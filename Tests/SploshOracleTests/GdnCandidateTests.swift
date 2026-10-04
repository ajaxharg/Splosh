import Accelerate
import Foundation
import Metal
import Testing
import SploshCore
import SploshRuntime

/// Candidate gated-delta kernels against engine.metal's split path (sp_gdn_prepare, sp_gdn_chains,
/// sp_gdn_finish, sp_gdn_history, with sp_gdn_commit before them) and against a CPU reference of
/// the same layer, on synthetic inputs at the model's dimensions.
///
/// SPLOSH_GDN_CANDIDATES lists what to test, comma separated. An item is a kernel name, placed
/// by its prefix (sp_gdn_prepare*, sp_gdn_chains*, sp_gdn_finish*, sp_gdn_history*) in the stage
/// it replaces; `a+b` replaces several stages at once; `name@64` runs a chain kernel with 64
/// threads a threadgroup instead of 128 (the engine's SPLOSH_GDN_CHAIN_THREADS). Unset, the
/// kernels of candidates/gdn_b.metal are tested.
///
/// An item goes through GdnKernelChoice.validate, the check Engine.init makes: kernels that do
/// not work together (a `_bn` prepare kernel without a `_bn` chain kernel, say) are an error
/// here as they are there, not something the test repairs.
///
/// A `_bn` prepare kernel hands the chain q and k already scaled by their norms, and (decay,
/// beta) where the norms were; an `s` one stores q and k at a key head's first value head only.
/// What the existing kernels and the CPU reference leave in those buffers is put in the same
/// form before it is compared.
///
/// The steps alternate between the split path and sp_gdn_fast (which the engine uses for steps
/// under gdnSplitRows whatever the candidates), so that journal entries and conv history written
/// by either are read by the other.
///
/// SPLOSH_GDN_CANDIDATE_TIMING=<repeats> enables the timing test: 48 dispatches of one stage for
/// one 128-row run, reference and candidates interleaved. It is a relative figure for choosing
/// between candidates, not a step time.
@Suite("GdnCandidateTests", .serialized)
struct GdnCandidateTests {
    static let valueHeads = 48, keyHeads = 16, d = 128
    static let channels = 2 * keyHeads * d + valueHeads * d
    static let slots = 4, units = 84, maxRows = 192, maxRuns = 8
    static let journalRows = 16, journalWidth = 260
    static let stateUnit = valueHeads * d * d
    static let convUnit = valueHeads * 3 * d * 3
    static let eps: Float = 1e-6
    /// Tolerances on a group's largest difference over its largest magnitude (see `group`). The
    /// elements are sums of 128 products that partly cancel, so a group can be a hundred
    /// roundings from another summation order: the existing kernels are 1.1e-4 from the CPU
    /// reference on their worst state row and the candidates 3.5e-5 from the existing kernels on
    /// their worst output vector. A wrong factor of a part in a thousand fails either (`controls`).
    static let againstExisting: Float = 1e-4, againstCPU: Float = 5e-4

    static let defaultCandidates = [
        "sp_gdn_prepare_b1", "sp_gdn_chains_b1", "sp_gdn_chains_b2", "sp_gdn_chains_b2@64", "sp_gdn_history_b1",
        "sp_gdn_prepare_b1+sp_gdn_chains_b2@64+sp_gdn_history_b1",
        "sp_gdn_prepare_bn1+sp_gdn_chains_bn1", "sp_gdn_prepare_bn1+sp_gdn_chains_bn2@64+sp_gdn_history_b1",
        "sp_gdn_chains_bs2", "sp_gdn_prepare_bs1+sp_gdn_chains_bs2@64+sp_gdn_history_b1",
        "sp_gdn_prepare_bn1+sp_gdn_chains_bns2", "sp_gdn_prepare_bns1+sp_gdn_chains_bns2@64+sp_gdn_history_b1",
    ]

    private struct Params {
        var rows, runs, keyHeads, valueHeads, headDim, channels: UInt32
        var eps: Float
        var debug: UInt32 = 0
    }

    struct Kernels: CustomStringConvertible {
        var choice = GdnKernelChoice()
        var prepare: String { choice.prepare }
        var chains: String { choice.chains }
        var finish: String { choice.finish }
        var history: String { choice.history }
        var chainThreads: Int { choice.chainThreads }
        var description: String { choice.description }

        init() {}
        /// `validate: false` is for the controls, which run choices the engine refuses.
        init(_ text: String, validate: Bool = true) throws {
            for part in text.split(separator: "+") {
                let pieces = part.split(separator: "@")
                let name = String(pieces[0])
                if name.hasPrefix("sp_gdn_prepare") { choice.prepare = name }
                else if name.hasPrefix("sp_gdn_chains") {
                    choice.chains = name
                    if pieces.count > 1, let threads = Int(pieces[1]) { choice.chainThreads = threads }
                } else if name.hasPrefix("sp_gdn_finish") { choice.finish = name }
                else if name.hasPrefix("sp_gdn_history") { choice.history = name }
                else { throw SploshError.capabilityGateFailure("cannot place candidate \(name) in a stage") }
            }
            if validate { try choice.validate() }
        }

        /// The prepare stage leaves q and k scaled by their norms and (decay, beta) in the norms' place.
        var normalised: Bool { choice.prepareForm.normalised }
        /// The prepare stage leaves q, k and their norms at a key head's first value head only.
        var shared: Bool { choice.prepareForm.shared }
    }

    struct Run { var slot, start, length: Int; var speculative: Bool; var pending: Int }

    typealias Fields = [(String, [Float])]

    /// One step's inputs, as the engine would hand them to the layer.
    struct Step {
        /// Run by sp_gdn_fast, as the engine runs a step of fewer than gdnSplitRows rows.
        var narrow: Bool
        var rows: Int
        var runs: [Run]
        var rowRead: [UInt32], rowWrite: [UInt32]
        var mixed, z, a, b, rowInv: [Float]
    }

    /// The weights and the starting state shared by every world.
    struct Fixture {
        var convWeight: [UInt16], aLog: [UInt16], dtBias: [UInt16], normWeight: [UInt16]
        var state: [Float], convState: [Float]
    }

    struct Generator {
        var seed: UInt64
        mutating func next() -> UInt64 {
            seed &+= 0x9E37_79B9_7F4A_7C15
            var x = seed
            x = (x ^ (x >> 30)) &* 0xBF58_476D_1CE4_E5B9
            x = (x ^ (x >> 27)) &* 0x94D0_49BB_1331_11EB
            return x ^ (x >> 31)
        }
        mutating func float(_ low: Float, _ high: Float) -> Float {
            low + (high - low) * Float(next() >> 40) / Float(1 << 24)
        }
        mutating func floats(_ count: Int, _ low: Float, _ high: Float) -> [Float] {
            (0..<count).map { _ in float(low, high) }
        }
    }

    static func bf16Bits(_ x: Float) -> UInt16 {
        // Round to nearest even, as the GPU's float to bfloat conversion does.
        let bits = x.bitPattern
        return UInt16(truncatingIfNeeded: (bits &+ 0x7FFF &+ ((bits >> 16) & 1)) >> 16)
    }
    static func bf16(_ bits: UInt16) -> Float { Float(bitPattern: UInt32(bits) << 16) }

    static func fixture() -> Fixture {
        var g = Generator(seed: 0x5EED_6D17)
        let convWeight = g.floats(channels * 4, -0.6, 0.6).map(bf16Bits)
        let aLog = g.floats(valueHeads, -2, 1).map(bf16Bits)
        let dtBias = g.floats(valueHeads, -1, 1).map(bf16Bits)
        let normWeight = g.floats(d, 0.5, 1.5).map(bf16Bits)
        let state = g.floats(slots * stateUnit, -0.5, 0.5)
        var convState = g.floats(units * convUnit, -1, 1)
        // The history of q and k is the same for the value heads of a key head: sp_gdn_history
        // writes it from the same inputs.
        let perKey = valueHeads / keyHeads
        for unit in 0..<units {
            for h in 0..<valueHeads where h % perKey != 0 {
                let first = (h / perKey) * perKey
                for element in 0..<(2 * d * 3) {
                    convState[unit * convUnit + h * 3 * d * 3 + element] = convState[unit * convUnit + first * 3 * d * 3 + element]
                }
            }
        }
        return Fixture(convWeight: convWeight, aLog: aLog, dtBias: dtBias, normWeight: normWeight, state: state, convState: convState)
    }

    /// Eight steps over four slots: run lengths from 1 to 128, speculative runs whose journal
    /// entries are accepted in part, in full and not at all, and the runs that then replay them.
    /// Steps 4 and 6 go through sp_gdn_fast: it replays in its own loop the entries the split
    /// path's speculative runs left (steps 3 and 5), reads the conv history the split path wrote,
    /// and leaves entries and history for the split path's steps 5 and 7 (sp_gdn_commit there).
    static func steps() -> [Step] {
        // (slot, length, speculative, entries accepted afterwards)
        let plan: [(Bool, [(Int, Int, Bool, Int)])] = [
            (false, [(0, 1, false, 0), (1, 7, true, 5), (2, 128, false, 0), (3, 33, false, 0)]),
            (false, [(1, 16, true, 16), (0, 64, false, 0), (3, 2, true, 2), (2, 17, false, 0)]),
            (false, [(3, 5, false, 0), (1, 128, false, 0), (0, 3, true, 0), (2, 31, false, 0)]),
            (false, [(0, 2, false, 0), (2, 127, false, 0), (1, 1, true, 1), (3, 40, false, 0)]),
            (true, [(1, 6, false, 0), (0, 4, true, 3), (2, 9, true, 9), (3, 3, false, 0)]),
            (false, [(0, 40, false, 0), (2, 5, true, 4), (1, 33, false, 0), (3, 12, true, 12)]),
            (true, [(2, 3, true, 2), (3, 8, false, 0), (0, 2, false, 0), (1, 1, true, 1)]),
            (false, [(2, 64, false, 0), (1, 20, false, 0), (0, 16, true, 0), (3, 37, false, 0)]),
        ]
        var g = Generator(seed: 0xC0FF_EE00)
        var slotUnit = Array(0..<slots), pending = Array(repeating: 0, count: slots)
        var nextUnit = slots
        var result: [Step] = []
        for (narrow, entries) in plan {
            var runs: [Run] = [], rowRead: [UInt32] = [], rowWrite: [UInt32] = []
            var row = 0
            for (slot, length, speculative, accepted) in entries {
                runs.append(Run(slot: slot, start: row, length: length, speculative: speculative, pending: pending[slot]))
                pending[slot] = 0
                var previous = slotUnit[slot]
                for t in 0..<length {
                    rowRead.append(UInt32(previous))
                    if speculative {
                        // A fresh unit per speculative row; the slot moves to the last accepted one.
                        let unit = nextUnit
                        nextUnit += 1
                        rowWrite.append(UInt32(unit))
                        previous = unit
                        if t + 1 == accepted { slotUnit[slot] = unit }
                    } else {
                        rowWrite.append(UInt32(previous))
                    }
                }
                if speculative { pending[slot] = accepted }
                row += length
            }
            precondition(row <= maxRows && nextUnit <= units)
            result.append(Step(narrow: narrow, rows: row, runs: runs, rowRead: rowRead, rowWrite: rowWrite,
                               mixed: g.floats(row * channels, -1, 1),
                               z: g.floats(row * valueHeads * d, -2, 2),
                               a: g.floats(row * valueHeads, -4, 4),
                               b: g.floats(row * valueHeads, -3, 3),
                               rowInv: g.floats(row, 0.5, 1.5)))
        }
        return result
    }

    // MARK: GPU

    final class World {
        let buffers: [Int: MTLBuffer]
        init(device: MTLDevice, fixture: Fixture) throws {
            let f = MemoryLayout<Float>.stride, u = MemoryLayout<UInt32>.stride, s = MemoryLayout<UInt16>.stride
            let rowHead = maxRows * valueHeads
            let sizes: [Int: Int] = [
                0: maxRows * channels * f, 1: rowHead * d * f, 2: rowHead * f, 3: rowHead * f,
                4: channels * 4 * s, 5: valueHeads * s, 6: valueHeads * s, 7: d * s,
                8: units * convUnit * f, 9: slots * stateUnit * f, 10: rowHead * d * f, 11: maxRuns * 3 * u,
                13: maxRows * u, 14: maxRows * u, 15: rowHead * d * s, 16: maxRows * (valueHeads * d / 64) * f,
                17: rowHead * d * f, 18: rowHead * d * f, 19: rowHead * d * f, 20: rowHead * 2 * f,
                21: slots * journalRows * valueHeads * journalWidth * f, 22: maxRuns * 2 * u, 23: maxRows * f,
            ]
            var made: [Int: MTLBuffer] = [:]
            for (index, bytes) in sizes {
                guard let buffer = device.makeBuffer(length: bytes, options: .storageModeShared) else {
                    throw SploshError.capabilityGateFailure("could not allocate synthetic GDN buffer \(index)")
                }
                memset(buffer.contents(), 0, bytes)
                made[index] = buffer
            }
            buffers = made
            write(4, fixture.convWeight); write(5, fixture.aLog); write(6, fixture.dtBias); write(7, fixture.normWeight)
            write(8, fixture.convState); write(9, fixture.state)
        }

        func write<T>(_ index: Int, _ values: [T]) {
            values.withUnsafeBytes { memcpy(buffers[index]!.contents(), $0.baseAddress!, $0.count) }
        }
        func fill(_ index: Int, count: Int, _ value: Float) {
            let pointer = buffers[index]!.contents().assumingMemoryBound(to: Float.self)
            for i in 0..<count { pointer[i] = value }
        }
        func floats(_ index: Int, from: Int = 0, count: Int) -> [Float] {
            Array(UnsafeBufferPointer(start: buffers[index]!.contents().assumingMemoryBound(to: Float.self) + from, count: count))
        }
        func bfloats(_ index: Int, count: Int) -> [Float] {
            UnsafeBufferPointer(start: buffers[index]!.contents().assumingMemoryBound(to: UInt16.self), count: count).map(GdnCandidateTests.bf16)
        }
    }

    private static func context() throws -> (MTLDevice, Metallib, MTLCommandQueue) {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw SploshError.capabilityGateFailure("no Metal device on this host")
        }
        let injected = ProcessInfo.processInfo.environment["SPLOSH_TEST_METALLIB"] ?? ""
        let library = injected.isEmpty ? try Metallib(device: device) : try makeTestMetallib(device: device)
        guard let queue = device.makeCommandQueue() else {
            throw SploshError.capabilityGateFailure("could not create an MTLCommandQueue")
        }
        return (device, library, queue)
    }

    private static func bind(_ encoder: MTLComputeCommandEncoder, _ world: World, rows: Int, runs: Int, state: MTLBuffer? = nil) {
        for (index, buffer) in world.buffers { encoder.setBuffer(buffer, offset: 0, index: index) }
        if let state { encoder.setBuffer(state, offset: 0, index: 9) }
        var p = Params(rows: UInt32(rows), runs: UInt32(runs), keyHeads: UInt32(keyHeads), valueHeads: UInt32(valueHeads),
                       headDim: UInt32(d), channels: UInt32(channels), eps: eps)
        encoder.setBytes(&p, length: MemoryLayout<Params>.stride, index: 12)
    }

    private static func finish(_ command: MTLCommandBuffer) throws {
        command.commit()
        command.waitUntilCompleted()
        guard command.status == .completed else {
            throw SploshError.capabilityGateFailure("GDN command failed: \(command.error?.localizedDescription ?? "unknown")")
        }
    }

    /// Load a step's inputs and poison everything the step is to write, so that a kernel that
    /// writes nothing cannot pass on what an earlier step or the allocation left there.
    private static func load(_ step: Step, into world: World) {
        world.write(0, step.mixed); world.write(1, step.z); world.write(2, step.a); world.write(3, step.b)
        world.write(23, step.rowInv); world.write(13, step.rowRead); world.write(14, step.rowWrite)
        world.write(11, step.runs.flatMap { [UInt32($0.slot), UInt32($0.start), UInt32($0.length)] })
        world.write(22, step.runs.flatMap { [$0.speculative ? UInt32(1) : 0, UInt32($0.pending)] })
        let rowHead = step.rows * valueHeads
        for index in [10, 17, 18, 19] { world.fill(index, count: rowHead * d, .nan) }
        world.fill(20, count: rowHead * 2, .nan)
        world.fill(16, count: step.rows * (valueHeads * d / 64), .nan)
        world.write(15, Array(repeating: UInt16(0x7FC0), count: rowHead * d))
        let journal = world.buffers[21]!.contents().assumingMemoryBound(to: Float.self)
        for run in step.runs where run.speculative && run.pending == 0 {
            // (With entries pending the commit before the step still has to read them.)
            for t in 0..<run.length {
                let base = (run.slot * journalRows * valueHeads + t * valueHeads) * journalWidth
                for i in 0..<(valueHeads * journalWidth) { journal[base + i] = .nan }
            }
        }
    }

    /// The layer as Engine.encodeLinear dispatches it: sp_gdn_fast for a narrow step, otherwise
    /// the split path. Returns what the recurrence left in the output buffer, read before the
    /// finish stage normalises it in place (the split path only): a factor applied to all of one
    /// (row, head) there cancels in the finish stage's RMSNorm and would not show after it.
    @discardableResult
    private static func run(_ step: Step, in world: World, kernels: Kernels, library: Metallib, queue: MTLCommandQueue) throws -> [Float]? {
        load(step, into: world)
        guard var command = queue.makeCommandBuffer() else { throw SploshError.capabilityGateFailure("no command buffer") }
        if step.narrow {
            guard let encoder = command.makeComputeCommandEncoder() else { throw SploshError.capabilityGateFailure("no encoder") }
            encoder.setComputePipelineState(try library.pipeline("sp_gdn_fast"))
            bind(encoder, world, rows: step.rows, runs: step.runs.count)
            encoder.dispatchThreadgroups(MTLSize(width: valueHeads, height: step.runs.count, depth: 1),
                                         threadsPerThreadgroup: MTLSize(width: 1024, height: 1, depth: 1))
            encoder.endEncoding()
            try finish(command)
            return nil
        }
        for run in step.runs where run.pending > 0 {
            guard let encoder = command.makeComputeCommandEncoder() else { throw SploshError.capabilityGateFailure("no encoder") }
            var cp = (UInt32(run.slot), UInt32(run.pending), UInt32(valueHeads), UInt32(d))
            encoder.setComputePipelineState(try library.pipeline("sp_gdn_commit"))
            encoder.setBuffer(world.buffers[9]!, offset: 0, index: 0)
            encoder.setBuffer(world.buffers[21]!, offset: 0, index: 1)
            encoder.setBytes(&cp, length: 16, index: 2)
            encoder.dispatchThreadgroups(MTLSize(width: valueHeads, height: 1, depth: 1),
                                         threadsPerThreadgroup: MTLSize(width: 1024, height: 1, depth: 1))
            encoder.endEncoding()
        }
        let stages: [(String, MTLSize, Int)] = [
            (kernels.prepare, MTLSize(width: valueHeads, height: step.rows, depth: 1), 96),
            (kernels.chains, MTLSize(width: valueHeads * 8, height: step.runs.count, depth: 1), kernels.chainThreads),
            (kernels.finish, MTLSize(width: valueHeads, height: step.rows, depth: 1), 32),
            (kernels.history, MTLSize(width: valueHeads, height: step.runs.count, depth: 1), 384),
        ]
        var chainOutput: [Float]? = nil
        for (index, (name, grid, threads)) in stages.enumerated() {
            guard let encoder = command.makeComputeCommandEncoder() else { throw SploshError.capabilityGateFailure("no encoder") }
            encoder.setComputePipelineState(try library.pipeline(name))
            bind(encoder, world, rows: step.rows, runs: step.runs.count)
            encoder.dispatchThreadgroups(grid, threadsPerThreadgroup: MTLSize(width: threads, height: 1, depth: 1))
            encoder.endEncoding()
            if index == 1 {
                try finish(command)
                chainOutput = world.floats(10, count: step.rows * valueHeads * d)
                guard let next = queue.makeCommandBuffer() else { throw SploshError.capabilityGateFailure("no command buffer") }
                command = next
            }
        }
        try finish(command)
        return chainOutput
    }

    /// The journal entries of a step's speculative runs, as three fields: k, delta and decay.
    private static func journalFields(_ step: Step, _ entry: (Int) -> ArraySlice<Float>) -> Fields {
        var k: [Float] = [], delta: [Float] = [], decay: [Float] = []
        for run in step.runs where run.speculative {
            for t in 0..<run.length {
                for h in 0..<valueHeads {
                    let line = entry((run.slot * journalRows * valueHeads + t * valueHeads + h) * journalWidth)
                    k += line.prefix(d); delta += line.dropFirst(d).prefix(d); decay.append(line[line.startIndex + 2 * d])
                }
            }
        }
        return [("journal k", k), ("journal delta", delta), ("journal decay", decay)]
    }

    /// Everything a step leaves behind, by name.
    private static func snapshot(_ step: Step, of world: World, chainOutput: [Float]?) -> Fields {
        let rowHead = step.rows * valueHeads
        var fields: Fields = [
            ("q", world.floats(17, count: rowHead * d)), ("k", world.floats(18, count: rowHead * d)),
            ("v", world.floats(19, count: rowHead * d)), ("norms", world.floats(20, count: rowHead * 2)),
            ("decay", world.floats(2, count: rowHead)), ("beta", world.floats(3, count: rowHead)),
            ("output", world.floats(10, count: rowHead * d)), ("output bf16", world.bfloats(15, count: rowHead * d)),
            ("output sums", world.floats(16, count: step.rows * (valueHeads * d / 64))),
            ("state", world.floats(9, count: slots * stateUnit)), ("conv history", world.floats(8, count: units * convUnit)),
        ]
        if let chainOutput { fields.append(("chain output", chainOutput)) }
        return fields + journalFields(step) { world.floats(21, from: $0, count: 2 * d + 1)[...] }
    }

    /// Whether every unit of a conv history has the same q and k history for the value heads of a
    /// key head: what the candidates' prepare kernels rely on, and what the engine checks of an
    /// imported state when such a kernel is chosen.
    private static func keyHistoryIsShared(_ conv: [Float]) -> Bool {
        conv.withUnsafeBufferPointer { pointer in
            (0..<units).allSatisfy {
                GdnKernelChoice.keyHistoryIsShared(pointer.baseAddress! + $0 * convUnit, valueHeads: valueHeads, keyHeads: keyHeads, headDim: d)
            }
        }
    }

    // MARK: CPU reference

    /// The layer in plain loops (the matrix products through vDSP), in single precision.
    final class Reference {
        var state: [Float], convState: [Float]
        var journal = [Float](repeating: 0, count: slots * journalRows * valueHeads * journalWidth)
        let fixture: Fixture

        init(_ fixture: Fixture) { self.fixture = fixture; state = fixture.state; convState = fixture.convState }

        private func update(_ s: UnsafeMutablePointer<Float>, decay: Float, k: UnsafePointer<Float>, delta: UnsafePointer<Float>) {
            // S[j][i] = S[j][i] * decay + k[i] * delta[j]
            var decay = decay
            for j in 0..<d {
                var dj = delta[j]
                vDSP_vsmsma(s + j * d, 1, &decay, k, 1, &dj, s + j * d, 1, vDSP_Length(d))
            }
        }

        func run(_ step: Step) -> Fields {
            let VH = valueHeads, KH = keyHeads, rowHead = step.rows * valueHeads
            let weight = fixture.convWeight.map(bf16)
            var q = [Float](repeating: 0, count: rowHead * d), k = q, v = q, core = q
            var norms = [Float](repeating: 0, count: rowHead * 2)
            var decay = [Float](repeating: 0, count: rowHead), beta = decay
            // Accepted journal entries first.
            for run in step.runs where run.pending > 0 {
                for h in 0..<VH {
                    for e in 0..<run.pending {
                        let entry = (run.slot * journalRows * VH + e * VH + h) * journalWidth
                        journal.withUnsafeBufferPointer { jp in
                            state.withUnsafeMutableBufferPointer { sp in
                                update(sp.baseAddress! + run.slot * stateUnit + h * d * d, decay: jp[entry + 2 * d],
                                       k: jp.baseAddress! + entry, delta: jp.baseAddress! + entry + d)
                            }
                        }
                    }
                }
            }
            // Conv + SiLU, norms, gates.
            let history = convState
            func input(_ run: Run, _ t: Int, _ h: Int, _ kind: Int, _ jc: Int, _ channel: Int) -> Float {
                if t >= 0 { return step.mixed[(run.start + t) * channels + channel] * step.rowInv[run.start + t] }
                return history[Int(step.rowRead[run.start]) * convUnit + ((h * 3 + kind) * d + jc) * 3 + 3 + t]
            }
            for run in step.runs {
                for t in 0..<run.length {
                    let row = run.start + t
                    for h in 0..<VH {
                        let hk = h / (VH / KH), index = row * VH + h
                        for kind in 0..<3 {
                            var squares: Float = 0
                            for jc in 0..<d {
                                let channel = kind == 0 ? hk * d + jc : kind == 1 ? KH * d + hk * d + jc : 2 * KH * d + h * d + jc
                                var sum: Float = 0
                                for tap in 0..<4 { sum += weight[channel * 4 + tap] * input(run, t - 3 + tap, h, kind, jc, channel) }
                                let out = sum / (1 + exp(-sum))
                                squares += out * out
                                if kind == 0 { q[index * d + jc] = out } else if kind == 1 { k[index * d + jc] = out } else { v[index * d + jc] = out }
                            }
                            if kind == 0 { norms[index * 2] = 1 / (squares + 1e-6).squareRoot() / Float(d).squareRoot() }
                            if kind == 1 { norms[index * 2 + 1] = 1 / (squares + 1e-6).squareRoot() }
                        }
                        let x = step.a[index] * step.rowInv[row] + bf16(fixture.dtBias[h])
                        let softplus = x > 20 ? x : log(1 + exp(x))
                        decay[index] = exp(-exp(bf16(fixture.aLog[h])) * softplus)
                        beta[index] = 1 / (1 + exp(-step.b[index] * step.rowInv[row]))
                    }
                }
            }
            // The recurrence.
            var memory = [Float](repeating: 0, count: d), delta = memory, out = memory
            for run in step.runs {
                for h in 0..<VH {
                    var s = Array(state[(run.slot * stateUnit + h * d * d)..<(run.slot * stateUnit + (h + 1) * d * d)])
                    for t in 0..<run.length {
                        let index = (run.start + t) * VH + h
                        let invQ = norms[index * 2], invK = norms[index * 2 + 1]
                        k.withUnsafeBufferPointer { kp in
                            vDSP_mmul(s, 1, kp.baseAddress! + index * d, 1, &memory, 1, vDSP_Length(d), 1, vDSP_Length(d))
                            for j in 0..<d { delta[j] = (v[index * d + j] - memory[j] * decay[index] * invK) * beta[index] * invK }
                            s.withUnsafeMutableBufferPointer { update($0.baseAddress!, decay: decay[index], k: kp.baseAddress! + index * d, delta: delta) }
                        }
                        q.withUnsafeBufferPointer { qp in
                            vDSP_mmul(s, 1, qp.baseAddress! + index * d, 1, &out, 1, vDSP_Length(d), 1, vDSP_Length(d))
                        }
                        for j in 0..<d { core[index * d + j] = out[j] * invQ }
                        if run.speculative {
                            let entry = (run.slot * journalRows * VH + t * VH + h) * journalWidth
                            for i in 0..<d { journal[entry + i] = k[index * d + i]; journal[entry + d + i] = delta[i] }
                            journal[entry + 2 * d] = decay[index]
                        }
                    }
                    if !run.speculative { state.replaceSubrange((run.slot * stateUnit + h * d * d)..<(run.slot * stateUnit + (h + 1) * d * d), with: s) }
                }
            }
            let journalOut = GdnCandidateTests.journalFields(step) { journal[$0..<($0 + 2 * d + 1)] }
            let chainOutput = core
            // Gated RMSNorm, the bf16 operand and its per-64 sums.
            var rounded = [Float](repeating: 0, count: rowHead * d)
            var sums = [Float](repeating: 0, count: step.rows * (VH * d / 64))
            for row in 0..<step.rows {
                for h in 0..<VH {
                    let base = (row * VH + h) * d
                    var squares: Float = 0
                    for j in 0..<d { squares += core[base + j] * core[base + j] }
                    let inv = 1 / (squares / Float(d) + eps).squareRoot()
                    for j in 0..<d {
                        let gate = step.z[base + j] * step.rowInv[row]
                        core[base + j] = bf16(fixture.normWeight[j]) * core[base + j] * inv * gate / (1 + exp(-gate))
                        rounded[base + j] = bf16(bf16Bits(core[base + j]))
                        sums[row * (VH * d / 64) + h * 2 + j / 64] += rounded[base + j]
                    }
                }
            }
            // The conv history every written unit keeps: the last three inputs up to its row.
            for run in step.runs {
                for t in 0..<run.length {
                    let row = run.start + t
                    if t + 1 != run.length && step.rowWrite[row + 1] == step.rowWrite[row] { continue }
                    for h in 0..<VH {
                        let hk = h / (VH / KH)
                        for kind in 0..<3 {
                            for jc in 0..<d {
                                let channel = kind == 0 ? hk * d + jc : kind == 1 ? KH * d + hk * d + jc : 2 * KH * d + h * d + jc
                                let target = Int(step.rowWrite[row]) * convUnit + ((h * 3 + kind) * d + jc) * 3
                                for back in 0..<3 { convState[target + 2 - back] = input(run, t - back, h, kind, jc, channel) }
                            }
                        }
                    }
                }
            }
            return [("q", q), ("k", k), ("v", v), ("norms", norms), ("decay", decay), ("beta", beta), ("output", core),
                    ("output bf16", rounded), ("output sums", sums), ("state", state), ("conv history", convState),
                    ("chain output", chainOutput)] + journalOut
        }
    }

    // MARK: Comparison

    /// A snapshot in sp_gdn_prepare's form, put in the form the candidate's prepare stage leaves.
    /// Normalised: q and k scaled by their norms, (decay, beta) in the norms' place, and in a
    /// journal entry k scaled and delta divided by k's norm (their product, which is what the
    /// replay uses, unchanged). Shared: see `keyHeadRows`.
    private static func inForm(of kernels: Kernels, _ fields: Fields, _ step: Step) -> Fields {
        var result = fields
        if kernels.normalised {
            var byName: [String: [Float]] = [:]
            for (name, values) in fields { byName[name] = values }
            let norms = byName["norms"]!, decay = byName["decay"]!, beta = byName["beta"]!
            var q = byName["q"]!, k = byName["k"]!, journalK = byName["journal k"]!, journalDelta = byName["journal delta"]!
            var gates = norms
            for index in 0..<(step.rows * valueHeads) {
                for i in 0..<d { q[index * d + i] *= norms[index * 2]; k[index * d + i] *= norms[index * 2 + 1] }
                gates[index * 2] = decay[index]; gates[index * 2 + 1] = beta[index]
            }
            var at = 0
            for run in step.runs where run.speculative {
                for t in 0..<run.length {
                    for h in 0..<valueHeads {
                        let invK = norms[((run.start + t) * valueHeads + h) * 2 + 1]
                        for i in 0..<d { journalK[at + i] *= invK; journalDelta[at + i] /= invK }
                        at += d
                    }
                }
            }
            let replaced = ["q": q, "k": k, "norms": gates, "journal k": journalK, "journal delta": journalDelta]
            result = result.map { ($0.0, replaced[$0.0] ?? $0.1) }
        }
        return kernels.shared ? keyHeadRows(result, norms: !kernels.normalised) : result
    }

    /// The rows of q and k (and of their norms) that belong to a key head's first value head: all
    /// that an `s` prepare kernel writes of them.
    private static func keyHeadRows(_ fields: Fields, norms: Bool) -> Fields {
        let perKey = valueHeads / keyHeads
        return fields.map { name, values in
            let width = name == "q" || name == "k" ? d : name == "norms" && norms ? 2 : 0
            guard width > 0 else { return (name, values) }
            var kept: [Float] = []
            for index in 0..<(values.count / width) where (index % valueHeads) % perKey == 0 {
                kept += values[(index * width)..<((index + 1) * width)]
            }
            return (name, kept)
        }
    }

    /// The elements of a field that are compared against a common magnitude. A vector of a (row,
    /// head), a state row, or a row's sums share one (their elements are sums that may cancel, so
    /// an element has no scale of its own); a norm, a gate or a stored input stands alone. With
    /// one magnitude for a whole field, an element far smaller than the field's largest (a q norm
    /// beside the k norms, a small state row) could be wholly wrong and pass.
    private static func group(_ name: String) -> Int {
        switch name {
        case "q", "k", "v", "output", "output bf16", "chain output", "state", "journal k", "journal delta": return d
        case "output sums": return valueHeads * d / 64
        default: return 1
        }
    }

    /// The largest difference in a group over the largest reference magnitude in that group, at
    /// its worst over the field's groups; infinite if anything is not finite or the lengths
    /// differ. `scale` is the largest reference magnitude in the field.
    static func relativeError(_ expected: [Float], _ actual: [Float], group: Int) -> (error: Float, scale: Float) {
        guard expected.count == actual.count, expected.count % group == 0 else { return (.infinity, 0) }
        return expected.withUnsafeBufferPointer { e in
            actual.withUnsafeBufferPointer { a in
                var worst: Float = 0, scale: Float = 0
                var at = 0
                while at < e.count {
                    var difference: Float = 0, magnitude: Float = 0
                    for i in at..<(at + group) {
                        let each = abs(e[i] - a[i])
                        if !(each <= .greatestFiniteMagnitude) { return (.infinity, scale) }
                        difference = max(difference, each)
                        magnitude = max(magnitude, abs(e[i]))
                    }
                    if difference > 0 { worst = max(worst, magnitude > 0 ? difference / magnitude : .infinity) }
                    scale = max(scale, magnitude)
                    at += group
                }
                return (worst, scale)
            }
        }
    }

    /// bf16 keeps eight bits of mantissa: an output a rounding apart in fp32 can land on the
    /// next bf16 value, and a sum of 64 of them moves with it.
    private static func tolerance(_ name: String, tight: Float) -> Float {
        switch name {
        case "output bf16": return 1.0 / 128
        case "output sums": return 1.0 / 16
        default: return tight
        }
    }

    struct Comparison {
        var failures: [String] = []
        /// Worst error over the fields kept in fp32.
        var worst: Float = 0
        var line = ""
    }

    /// Every field of `got` against the field of the same name in `expected`.
    static func compare(_ got: Fields, to expected: Fields, tight: Float) -> Comparison {
        var byName: [String: [Float]] = [:]
        for (name, values) in expected { byName[name] = values }
        var result = Comparison()
        for (name, values) in got {
            guard let reference = byName[name] else { result.failures.append("\(name): nothing to compare with"); continue }
            let error = relativeError(reference, values, group: group(name)).error
            if !(error <= tolerance(name, tight: tight)) { result.failures.append("\(name): \(error)") }
            if !name.hasPrefix("output ") { result.worst = max(result.worst, error) }
            result.line += " \(name)=\(error)"
        }
        return result
    }

    @Test("candidate kernels agree with the existing kernels and with a CPU reference")
    func candidatesAgree() throws {
        let (device, library, queue) = try Self.context()
        let listed = ProcessInfo.processInfo.environment["SPLOSH_GDN_CANDIDATES"] ?? ""
        let names = listed.isEmpty ? Self.defaultCandidates : listed.split(separator: ",").map { $0.trimmingCharacters(in: .whitespaces) }
        let candidates = try names.map { (name: $0, kernels: try Kernels($0)) }
        let fixture = Self.fixture(), steps = Self.steps()
        #expect(Self.keyHistoryIsShared(fixture.convState))
        let reference = Reference(fixture)
        let existing = try World(device: device, fixture: fixture)
        let worlds = try candidates.map { _ in try World(device: device, fixture: fixture) }
        var worstAgainstExisting = [Float](repeating: 0, count: candidates.count)
        var worstAgainstCPU = [Float](repeating: 0, count: candidates.count)
        var worstExistingAgainstCPU: Float = 0
        for (number, step) in steps.enumerated() {
            let path = step.narrow ? "sp_gdn_fast" : "split"
            let expected = reference.run(step)
            let chainOutput = try Self.run(step, in: existing, kernels: Kernels(), library: library, queue: queue)
            let base = Self.snapshot(step, of: existing, chainOutput: chainOutput)
            for (name, values) in expected where !values.isEmpty {
                // Every field has to carry real numbers: an all-zero reference would pass anything.
                let scale = Self.relativeError(values, values, group: 1).scale
                #expect(scale > 1e-3, "step \(number) \(name): the reference is empty")
            }
            let against = Self.compare(base, to: expected, tight: Self.againstCPU)
            #expect(against.failures.isEmpty, "step \(number) (\(path)) existing kernels vs CPU: \(against.failures)")
            worstExistingAgainstCPU = max(worstExistingAgainstCPU, against.worst)
            // The kernels that write conv history keep q's and k's the same for a key head's value heads.
            #expect(Self.keyHistoryIsShared(existing.floats(8, count: Self.units * Self.convUnit)), "step \(number) existing kernels: key-head history")
            for (c, candidate) in candidates.enumerated() {
                let chainOutput = try Self.run(step, in: worlds[c], kernels: candidate.kernels, library: library, queue: queue)
                var got = Self.snapshot(step, of: worlds[c], chainOutput: chainOutput)
                var base = base, expected = expected
                if !step.narrow {
                    // sp_gdn_fast leaves these buffers in its own form whatever the candidates.
                    base = Self.inForm(of: candidate.kernels, base, step)
                    expected = Self.inForm(of: candidate.kernels, expected, step)
                    if candidate.kernels.shared { got = Self.keyHeadRows(got, norms: !candidate.kernels.normalised) }
                }
                let existing = Self.compare(got, to: base, tight: Self.againstExisting), cpu = Self.compare(got, to: expected, tight: Self.againstCPU)
                #expect(existing.failures.isEmpty, "step \(number) (\(path)) \(candidate.name) vs existing: \(existing.failures)")
                #expect(cpu.failures.isEmpty, "step \(number) (\(path)) \(candidate.name) vs CPU: \(cpu.failures)")
                #expect(Self.keyHistoryIsShared(worlds[c].floats(8, count: Self.units * Self.convUnit)), "step \(number) \(candidate.name): key-head history")
                worstAgainstExisting[c] = max(worstAgainstExisting[c], existing.worst)
                worstAgainstCPU[c] = max(worstAgainstCPU[c], cpu.worst)
                print("GDN candidate \(candidate.name) step \(number) (\(path), \(step.rows) rows):\(existing.line)")
            }
        }
        print("GDN existing kernels vs CPU: worst relative error \(worstExistingAgainstCPU)")
        for (c, candidate) in candidates.enumerated() {
            print("GDN candidate \(candidate.name) [\(candidate.kernels)]: worst relative error vs existing \(worstAgainstExisting[c]), vs CPU \(worstAgainstCPU[c])")
        }
    }

    @Test("kernel choices that do not work together are refused")
    func choicesAreChecked() throws {
        func choice(_ environment: [String: String]) throws -> GdnKernelChoice { try GdnKernelChoice(environment: environment) }
        // The default, and the choices the lead is told to make.
        let standard = try choice([:])
        #expect(standard.prepare == "sp_gdn_prepare" && standard.chains == "sp_gdn_chains" && standard.finish == "sp_gdn_finish"
                && standard.history == "sp_gdn_history" && standard.chainGroups == 8 && standard.chainThreads == 128)
        #expect(!standard.readsKeyHeadHistory)
        let accepted: [[String: String]] = [
            ["SPLOSH_GDN_PREPARE": "sp_gdn_prepare_b1", "SPLOSH_GDN_CHAINS": "sp_gdn_chains_b2", "SPLOSH_GDN_HISTORY": "sp_gdn_history_b1"],
            ["SPLOSH_GDN_CHAINS": "sp_gdn_chains_b2", "SPLOSH_GDN_CHAIN_THREADS": "64"],
            ["SPLOSH_GDN_PREPARE": "sp_gdn_prepare_bn1", "SPLOSH_GDN_CHAINS": "sp_gdn_chains_bn2", "SPLOSH_GDN_CHAIN_THREADS": "64"],
            ["SPLOSH_GDN_PREPARE": "sp_gdn_prepare_bn1", "SPLOSH_GDN_CHAINS": "sp_gdn_chains_bns2"],
            ["SPLOSH_GDN_CHAINS": "sp_gdn_chains_bs2"],
            ["SPLOSH_GDN_PREPARE": "sp_gdn_prepare_bs1", "SPLOSH_GDN_CHAINS": "sp_gdn_chains_bs2", "SPLOSH_GDN_CHAIN_THREADS": "64"],
            ["SPLOSH_GDN_PREPARE": "sp_gdn_prepare_bns1", "SPLOSH_GDN_CHAINS": "sp_gdn_chains_bns2"],
            // A kernel from another file: nothing is known of it, so nothing is refused.
            ["SPLOSH_GDN_CHAINS": "sp_gdn_chains_other", "SPLOSH_GDN_CHAIN_GROUPS": "16"],
        ]
        for environment in accepted { #expect(throws: Never.self, "\(environment)") { _ = try choice(environment) } }
        #expect(try choice(accepted[0]).readsKeyHeadHistory && choice(accepted[4]).readsKeyHeadHistory)
        let refused: [[String: String]] = [
            ["SPLOSH_GDN_PREPARE": "sp_gdn_prepare_bn1"],
            ["SPLOSH_GDN_CHAINS": "sp_gdn_chains_bn2"],
            ["SPLOSH_GDN_PREPARE": "sp_gdn_prepare_bn1", "SPLOSH_GDN_CHAINS": "sp_gdn_chains_b2"],
            ["SPLOSH_GDN_PREPARE": "sp_gdn_prepare_b1", "SPLOSH_GDN_CHAINS": "sp_gdn_chains_bns2"],
            ["SPLOSH_GDN_PREPARE": "sp_gdn_prepare_bs1"],
            ["SPLOSH_GDN_PREPARE": "sp_gdn_prepare_bs1", "SPLOSH_GDN_CHAINS": "sp_gdn_chains_b2"],
            ["SPLOSH_GDN_PREPARE": "sp_gdn_prepare_bns1", "SPLOSH_GDN_CHAINS": "sp_gdn_chains_bn2"],
            ["SPLOSH_GDN_PREPARE": "sp_gdn_prepare_bns1", "SPLOSH_GDN_CHAINS": "sp_gdn_chains_bs2"],
            ["SPLOSH_GDN_CHAIN_THREADS": "64"],
            ["SPLOSH_GDN_CHAINS": "sp_gdn_chains_b1", "SPLOSH_GDN_CHAIN_THREADS": "64"],
            ["SPLOSH_GDN_CHAINS": "sp_gdn_chains_b2", "SPLOSH_GDN_CHAIN_THREADS": "32"],
            ["SPLOSH_GDN_CHAINS": "sp_gdn_chains_b2", "SPLOSH_GDN_CHAIN_THREADS": "80"],
            ["SPLOSH_GDN_CHAINS": "sp_gdn_chains_b2", "SPLOSH_GDN_CHAIN_GROUPS": "16"],
            ["SPLOSH_GDN_CHAIN_THREADS": "many"],
        ]
        for environment in refused { #expect(throws: EngineError.self, "\(environment)") { _ = try choice(environment) } }
        // The test's own parser goes through the same check.
        #expect(throws: EngineError.self) { _ = try Kernels("sp_gdn_prepare_bn1") }

        // The history check: the fixture's copies agree; one float changed in a second value
        // head's q history, or in a third's k history, and they do not. v's history is the head's own.
        var conv = Array(Self.fixture().convState.prefix(Self.convUnit))
        #expect(Self.keyHistoryIsShared(Self.fixture().convState))
        func shared() -> Bool {
            conv.withUnsafeBufferPointer { GdnKernelChoice.keyHistoryIsShared($0.baseAddress!, valueHeads: Self.valueHeads, keyHeads: Self.keyHeads, headDim: Self.d) }
        }
        #expect(shared())
        let perHead = 3 * Self.d * 3
        conv[4 * perHead + 2 * Self.d * 3 + 5] += 1          // head 4, v
        #expect(shared())
        conv[4 * perHead + 17] += 1                          // head 4, q
        #expect(!shared())
        conv[4 * perHead + 17] -= 1
        #expect(shared())
        conv[47 * perHead + Self.d * 3 + 383] += 1           // head 47, k, the last float
        #expect(!shared())
    }

    /// The comparison has to fail on kernels that are wrong, or it shows nothing when it passes.
    @Test("controls: wrong kernels and wrong numbers fail the comparison")
    func controls() throws {
        let (device, library, queue) = try Self.context()
        let fixture = Self.fixture(), step = Self.steps()[0]
        let existing = try World(device: device, fixture: fixture)
        let base = Self.snapshot(step, of: existing, chainOutput: try Self.run(step, in: existing, kernels: Kernels(), library: library, queue: queue))
        #expect(Self.compare(base, to: base, tight: 0).failures.isEmpty)

        func failures(_ text: String) throws -> [String] {
            let kernels = try Kernels(text, validate: false)
            let world = try World(device: device, fixture: fixture)
            let chainOutput = try Self.run(step, in: world, kernels: kernels, library: library, queue: queue)
            var got = Self.snapshot(step, of: world, chainOutput: chainOutput)
            if kernels.shared { got = Self.keyHeadRows(got, norms: !kernels.normalised) }
            let found = Self.compare(got, to: Self.inForm(of: kernels, base, step), tight: Self.againstExisting).failures
            print("GDN control \(text): \(found)")
            return found
        }
        func names(_ failures: [String]) -> Set<String> { Set(failures.map { String($0.split(separator: ":")[0]) }) }

        // A kernel that leaves columns of the state uncomputed: 32 threads where it needs 64.
        #expect(names(try failures("sp_gdn_chains_b2@32")).isSuperset(of: ["chain output", "output", "state", "journal delta"]))
        #expect(names(try failures("sp_gdn_chains_b1@64")).isSuperset(of: ["chain output", "output", "state", "journal delta"]))
        // Half of a pair with engine.metal's other half: finite, plausible and wrong.
        #expect(names(try failures("sp_gdn_prepare_bn1")).isSuperset(of: ["chain output", "state"]))
        #expect(names(try failures("sp_gdn_chains_bn2")).isSuperset(of: ["chain output", "state"]))
        // q and k stored once per key head, read by a kernel that looks at every value head.
        #expect(names(try failures("sp_gdn_prepare_bs1")).isSuperset(of: ["chain output", "state"]))

        // Numbers wrong in ways one magnitude per field, or the output after the finish stage, hide.
        func changed(_ name: String, _ change: (inout [Float]) -> Void) -> [String] {
            let got = base.map { field -> (String, [Float]) in
                guard field.0 == name else { return field }
                var values = field.1
                change(&values)
                return (field.0, values)
            }
            return Self.compare(got, to: base, tight: Self.againstExisting).failures
        }
        // One (row, head) of the recurrence's output off by a factor: this cancels in the finish
        // stage's RMSNorm, so only the output read before that stage shows it.
        let row = 40 * Self.valueHeads + 7
        #expect(names(changed("chain output") { for i in 0..<Self.d { $0[row * Self.d + i] *= 1.001 } }) == ["chain output"])
        // The smallest norm and the smallest state row, each wrong by a part in a thousand.
        let norms = base.first { $0.0 == "norms" }!.1
        let smallest = norms.indices.min { abs(norms[$0]) < abs(norms[$1]) }!
        #expect(norms.map { abs($0) }.max()! > 5 * abs(norms[smallest]))
        #expect(names(changed("norms") { $0[smallest] *= 1.001 }) == ["norms"])
        let state = base.first { $0.0 == "state" }!.1
        func magnitude(_ group: Int) -> Float { state[(group * Self.d)..<((group + 1) * Self.d)].map { abs($0) }.max()! }
        let quiet = (0..<(state.count / Self.d)).min { magnitude($0) < magnitude($1) }!
        #expect(names(changed("state") { for i in 0..<Self.d { $0[quiet * Self.d + i] *= 1.001 } }) == ["state"])
        // An element that was never written.
        #expect(names(changed("v") { $0[12345] = .nan }) == ["v"])
    }

    @Test("timing of one stage, 48 dispatches, reference and candidates interleaved",
          .enabled(if: ProcessInfo.processInfo.environment["SPLOSH_GDN_CANDIDATE_TIMING"] != nil))
    func timing() throws {
        let (device, library, queue) = try Self.context()
        let repeats = Int(ProcessInfo.processInfo.environment["SPLOSH_GDN_CANDIDATE_TIMING"] ?? "") ?? 7
        let listed = ProcessInfo.processInfo.environment["SPLOSH_GDN_CANDIDATES"] ?? ""
        let names = listed.isEmpty ? Self.defaultCandidates : listed.split(separator: ",").map { $0.trimmingCharacters(in: .whitespaces) }
        let fixture = Self.fixture()
        let world = try World(device: device, fixture: fixture)
        let layers = 48
        let rows = max(1, min(128, Int(ProcessInfo.processInfo.environment["SPLOSH_GDN_TIMING_ROWS"] ?? "") ?? 128))
        var g = Generator(seed: 0x71AE)
        // The range the decay's input is drawn from: "-4,4" gives decays from 1e-8 to 1, "-9,-5"
        // decays within a percent of 1 (which is what lets a kernel carry the decay as a scalar).
        let range = (ProcessInfo.processInfo.environment["SPLOSH_GDN_TIMING_A"] ?? "-4,4").split(separator: ",").compactMap { Float($0) }
        let low = range.count == 2 ? range[0] : -4, high = range.count == 2 ? range[1] : 4
        let step = Step(narrow: false, rows: rows, runs: [Run(slot: 0, start: 0, length: rows, speculative: false, pending: 0)],
                        rowRead: Array(repeating: 0, count: rows), rowWrite: Array(repeating: 0, count: rows),
                        mixed: g.floats(rows * Self.channels, -1, 1), z: g.floats(rows * Self.valueHeads * Self.d, -2, 2),
                        a: g.floats(rows * Self.valueHeads, low, high), b: g.floats(rows * Self.valueHeads, -3, 3),
                        rowInv: g.floats(rows, 0.5, 1.5))
        // A state per layer, as in the model: 3 MB each, so the 48 do not stay in a cache.
        var states: [MTLBuffer] = []
        for _ in 0..<layers {
            guard let buffer = device.makeBuffer(length: Self.stateUnit * MemoryLayout<Float>.stride, options: .storageModeShared) else {
                throw SploshError.capabilityGateFailure("could not allocate a timing state")
            }
            memcpy(buffer.contents(), world.buffers[9]!.contents(), buffer.length)
            states.append(buffer)
        }
        // Valid operands for the chain and finish stages, in both forms; the prepare stage, which
        // overwrites them (and its own inputs), is timed in a world of its own.
        let paired = try World(device: device, fixture: fixture), scratch = try World(device: device, fixture: fixture)
        try Self.run(step, in: world, kernels: Kernels(), library: library, queue: queue)
        try Self.run(step, in: paired, kernels: try Kernels("sp_gdn_prepare_bn1+sp_gdn_chains_bn1"), library: library, queue: queue)
        try Self.run(step, in: scratch, kernels: Kernels(), library: library, queue: queue)

        // Dispatches a sample: fewer than the 48 gives shorter samples, of which more escape
        // whatever else is using the GPU. Figures are scaled to 48.
        let batch = max(1, min(layers, Int(ProcessInfo.processInfo.environment["SPLOSH_GDN_TIMING_BATCH"] ?? "") ?? layers))
        func time(_ name: String, grid: MTLSize, threads: Int, first: Int) throws -> Double {
            // (An `s` chain kernel reads rows that every prepare kernel of its form fills.)
            let target = name.hasPrefix("sp_gdn_prepare") ? scratch : name.contains("_bn") ? paired : world
            guard let command = queue.makeCommandBuffer(), let encoder = command.makeComputeCommandEncoder() else {
                throw SploshError.capabilityGateFailure("no command buffer")
            }
            encoder.setComputePipelineState(try library.pipeline(name))
            for layer in 0..<batch {
                Self.bind(encoder, target, rows: rows, runs: 1, state: states[(first + layer) % layers])
                encoder.dispatchThreadgroups(grid, threadsPerThreadgroup: MTLSize(width: threads, height: 1, depth: 1))
            }
            encoder.endEncoding()
            try Self.finish(command)
            return (command.gpuEndTime - command.gpuStartTime) * 1000 * Double(layers) / Double(batch)
        }
        var configurations: [(label: String, name: String, grid: MTLSize, threads: Int)] = [
            ("sp_gdn_prepare", "sp_gdn_prepare", MTLSize(width: Self.valueHeads, height: rows, depth: 1), 96),
            ("sp_gdn_chains", "sp_gdn_chains", MTLSize(width: Self.valueHeads * 8, height: 1, depth: 1), 128),
            ("sp_gdn_finish", "sp_gdn_finish", MTLSize(width: Self.valueHeads, height: rows, depth: 1), 32),
            ("sp_gdn_history", "sp_gdn_history", MTLSize(width: Self.valueHeads, height: 1, depth: 1), 384),
        ]
        for item in names where !item.contains("+") {
            let kernels = try Kernels(item, validate: false)
            if item.hasPrefix(kernels.prepare) { configurations.append((item, kernels.prepare, configurations[0].grid, 96)) }
            if item.hasPrefix(kernels.chains) { configurations.append((item, kernels.chains, configurations[1].grid, kernels.chainThreads)) }
            if item.hasPrefix(kernels.finish) { configurations.append((item, kernels.finish, configurations[2].grid, 32)) }
            if item.hasPrefix(kernels.history) { configurations.append((item, kernels.history, configurations[3].grid, 384)) }
        }
        var samples = [[Double]](repeating: [], count: configurations.count)
        for pass in 0...repeats {
            for (index, configuration) in configurations.enumerated() {
                let ms = try time(configuration.name, grid: configuration.grid, threads: configuration.threads, first: pass * batch)
                if pass > 0 { samples[index].append(ms) }      // the first pass builds pipelines and faults pages
            }
        }
        for (index, configuration) in configurations.enumerated() {
            let sorted = samples[index].sorted()
            print(String(format: "GDN timing %@: min %.2f ms, lower decile %.2f ms, median %.2f ms for %d dispatches of %d rows",
                         configuration.label, sorted[0], sorted[sorted.count / 10], sorted[sorted.count / 2], layers, rows))
        }
        // The stages run in place on their own outputs here; the figures must still be finite.
        #expect(world.floats(10, count: rows * Self.valueHeads * Self.d).allSatisfy { $0.isFinite })
    }
}
