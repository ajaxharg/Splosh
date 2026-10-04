import Testing
import Metal

import SploshCore
import SploshQuant
import SploshRuntime

// The MLP's gate/up/SiLU stage on synthetic q4 weights in the tiled layout: the shipped three
// dispatches (gate GEMM, up GEMM, sp_silu_mul_na) against every variant of the candidate (gate
// GEMM, then the up GEMM with the SiLU product in its epilogue:
// Sources/Shaders/candidates/mlp_fused.metal), and all against a CPU reference. What is compared
// is the down projection's operand: bf16 values and their per-64 sums. The candidate's
// dispatches come from MlpFusedPlan, which is also what the engine encodes from.

private struct NaParams {
    var rows, outDim, inner, groups, hasResidual, outStride: UInt32
}

private struct FusedParams {
    var rows, outDim, inner, groups, firstTile, outStride: UInt32
}

private struct SplitMix {
    var state: UInt64
    mutating func next() -> UInt64 {
        state &+= 0x9E37_79B9_7F4A_7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        return z ^ (z >> 31)
    }
    /// Uniform in [-1, 1).
    mutating func unit() -> Float { Float(Int64(next() >> 40) - (1 << 23)) / Float(1 << 23) }
}

/// One q4 weight in the tiled layout, with its values for the CPU reference.
private struct TiledWeight {
    let outDim: Int, inner: Int
    /// [tile of 128 outputs][quant group][output in tile][32 bytes: 64 codes, low nibble first]
    var packed: [UInt8]
    /// [tile][quant group][output in tile], bf16 bits
    var scales: [UInt16], biases: [UInt16]

    init(outDim: Int, inner: Int, seed: UInt64) {
        self.outDim = outDim; self.inner = inner
        let groups = inner / 64
        var random = SplitMix(state: seed)
        packed = [UInt8](repeating: 0, count: outDim * groups * 32)
        packed.withUnsafeMutableBytes { bytes in
            let words = bytes.bindMemory(to: UInt64.self)
            for index in 0..<words.count { words[index] = random.next() }
        }
        scales = [UInt16](repeating: 0, count: outDim * groups); biases = scales
        for index in 0..<(outDim * groups) {
            let scale = 0.0125 + 0.0075 * random.unit()
            scales[index] = Q4BufferLayout.bf16Bits(scale)
            biases[index] = Q4BufferLayout.bf16Bits(-7.5 * scale + 0.01 * random.unit())
        }
    }

    func slot(output n: Int, group g: Int) -> Int { ((n / 128) * (inner / 64) + g) * 128 + n % 128 }
}

/// The inputs of one step, and the dispatches on them.
private final class MlpCase {
    static let sentinel16: UInt16 = 0xFFFF     // a bf16 NaN no kernel here produces
    static let sentinel32 = Float(bitPattern: 0xFFFF_FFFF)

    let device: MTLDevice, library: Metallib, queue: MTLCommandQueue
    let rows: Int, cap: Int, inner: Int, outDim: Int
    var groups: Int { inner / 64 }
    var outGroups: Int { outDim / 64 }
    let x: [UInt16], xSums: [Float], rowInv: [Float]
    let gateWeight: TiledWeight, upWeight: TiledWeight
    let xBuffer, sumsBuffer, invBuffer: MTLBuffer
    let gatePacked, gateScales, gateBiases, upPacked, upScales, upBiases: MTLBuffer

    init(rows: Int, gate: TiledWeight, up: TiledWeight) throws {
        guard let device = MTLCreateSystemDefaultDevice() else { throw SploshError.capabilityGateFailure("no Metal device on this host") }
        self.device = device
        library = try Metallib(device: device)
        queue = try #require(device.makeCommandQueue())
        self.rows = rows; inner = gate.inner; outDim = gate.outDim
        gateWeight = gate; upWeight = up
        let cap = (rows + 127) / 128 * 128, inner = gate.inner, groups = inner / 64
        self.cap = cap
        var random = SplitMix(state: 0x51_1C0 &+ UInt64(rows))
        // Activations for every row the buffers hold (rows past the step are not zero, so a
        // kernel that lets them through is seen), their per-64 sums, and a per-row scale.
        let x: [UInt16] = (0..<(cap * inner)).map { _ in Q4BufferLayout.bf16Bits(random.unit()) }
        var xSums = [Float](repeating: 0, count: cap * groups)
        for row in 0..<cap {
            for g in 0..<groups {
                var total: Float = 0
                for k in 0..<64 { total += Q4BufferLayout.bf16(x[row * inner + g * 64 + k]) }
                xSums[row * groups + g] = total
            }
        }
        self.x = x; self.xSums = xSums
        rowInv = (0..<cap).map { _ in 1.0 + 0.5 * random.unit() }
        xBuffer = try Self.buffer(device, x); sumsBuffer = try Self.buffer(device, xSums); invBuffer = try Self.buffer(device, rowInv)
        gatePacked = try Self.buffer(device, gate.packed); gateScales = try Self.buffer(device, gate.scales); gateBiases = try Self.buffer(device, gate.biases)
        upPacked = try Self.buffer(device, up.packed); upScales = try Self.buffer(device, up.scales); upBiases = try Self.buffer(device, up.biases)
    }

    static func buffer<T>(_ device: MTLDevice, _ values: [T]) throws -> MTLBuffer {
        try #require(values.withUnsafeBytes { device.makeBuffer(bytes: $0.baseAddress!, length: $0.count, options: .storageModeShared) })
    }
    static func read<T>(_ buffer: MTLBuffer, _ count: Int, as type: T.Type) -> [T] {
        Array(UnsafeBufferPointer(start: buffer.contents().bindMemory(to: T.self, capacity: count), count: count))
    }

    struct Operand {
        var values: [UInt16], sums: [Float]
        /// The bf16 gate the up GEMM read, when the variant has one.
        var gate16: [UInt16]?
    }

    /// The operand of the down projection. `variant` nil: the shipped three dispatches;
    /// otherwise the candidate's, from MlpFusedPlan. Outputs start as sentinels, so an element
    /// nothing wrote stays one.
    func run(variant: Int?) throws -> Operand {
        let gate32 = try Self.buffer(device, [Float](repeating: Self.sentinel32, count: cap * outDim))
        let outB = try Self.buffer(device, [UInt16](repeating: Self.sentinel16, count: cap * outDim))
        let outSums = try Self.buffer(device, [Float](repeating: Self.sentinel32, count: cap * outGroups))
        let command = try #require(queue.makeCommandBuffer())
        let encoder = try #require(command.makeComputeCommandEncoder(dispatchType: .concurrent))
        // The wide GEMM the engine dispatches: the small kernel for whole 32-row tiles, the
        // general tiled one when the last tile is partial.
        let pGemm = try library.pipeline(rows % 32 == 0 ? "sp_gemm_q4_nat_m32n128s4_whole" : "sp_gemm_q4_nat_m32n128s4")
        func gemm(_ packed: MTLBuffer, _ scales: MTLBuffer, _ biases: MTLBuffer, into output: MTLBuffer) {
            var p = NaParams(rows: UInt32(rows), outDim: UInt32(outDim), inner: UInt32(inner), groups: UInt32(groups),
                             hasResidual: 0, outStride: UInt32(outDim))
            encoder.setComputePipelineState(pGemm)
            encoder.setBuffer(packed, offset: 0, index: 0); encoder.setBuffer(scales, offset: 0, index: 1); encoder.setBuffer(biases, offset: 0, index: 2)
            encoder.setBuffer(xBuffer, offset: 0, index: 3)
            encoder.setBuffer(output, offset: 0, index: 4); encoder.setBuffer(output, offset: 0, index: 5)
            encoder.setBuffer(sumsBuffer, offset: 0, index: 6)
            encoder.setBytes(&p, length: MemoryLayout<NaParams>.stride, index: 7)
            encoder.dispatchThreadgroups(MlpFusedPlan.grid(rowTiles: (rows + 31) / 32, outDim: outDim), threadsPerThreadgroup: MlpFusedPlan.threadsPerGroup)
        }
        var gate16: MTLBuffer?
        if let variant {
            // Candidate: gate, a barrier, then up with the product in its epilogue.
            let plan = try #require(MlpFusedPlan(variant: variant, rows: rows, outDim: outDim))
            func fused(_ dispatch: MlpFusedPlan.Dispatch, _ packed: MTLBuffer, _ scales: MTLBuffer, _ biases: MTLBuffer,
                       gate: MTLBuffer, product: Bool) throws {
                var p = FusedParams(rows: UInt32(dispatch.rows), outDim: UInt32(outDim), inner: UInt32(inner), groups: UInt32(groups),
                                    firstTile: UInt32(dispatch.firstTile), outStride: UInt32(outDim))
                encoder.setComputePipelineState(try library.pipeline(dispatch.kernel))
                encoder.setBuffer(packed, offset: 0, index: 0); encoder.setBuffer(scales, offset: 0, index: 1); encoder.setBuffer(biases, offset: 0, index: 2)
                encoder.setBuffer(xBuffer, offset: 0, index: 3)
                encoder.setBuffer(gate, offset: 0, index: 4)
                encoder.setBuffer(product ? outB : gate, offset: 0, index: 5)
                encoder.setBuffer(sumsBuffer, offset: 0, index: 6)
                encoder.setBytes(&p, length: MemoryLayout<FusedParams>.stride, index: 7)
                if product {
                    encoder.setBuffer(outSums, offset: 0, index: 8)
                    encoder.setBuffer(invBuffer, offset: 0, index: 9)
                }
                encoder.dispatchThreadgroups(dispatch.grid, threadsPerThreadgroup: MlpFusedPlan.threadsPerGroup)
            }
            let gateValues: MTLBuffer
            if let gate = plan.gate {
                let buffer = try Self.buffer(device, [UInt16](repeating: Self.sentinel16, count: cap * outDim))
                gate16 = buffer; gateValues = buffer
                try fused(gate, gatePacked, gateScales, gateBiases, gate: buffer, product: false)
            } else {
                gateValues = gate32
                gemm(gatePacked, gateScales, gateBiases, into: gate32)
            }
            encoder.memoryBarrier(scope: .buffers)
            for up in plan.up { try fused(up, upPacked, upScales, upBiases, gate: gateValues, product: true) }
        } else {
            // Shipped: gate and up in one stage, then the SiLU product.
            let up32 = try Self.buffer(device, [Float](repeating: Self.sentinel32, count: cap * outDim))
            gemm(gatePacked, gateScales, gateBiases, into: gate32)
            gemm(upPacked, upScales, upBiases, into: up32)
            encoder.memoryBarrier(scope: .buffers)
            var sp = (UInt32(rows), UInt32(outDim))
            encoder.setComputePipelineState(try library.pipeline("sp_silu_mul_na"))
            encoder.setBuffer(gate32, offset: 0, index: 0); encoder.setBuffer(up32, offset: 0, index: 1)
            encoder.setBuffer(outB, offset: 0, index: 2); encoder.setBuffer(outSums, offset: 0, index: 3)
            encoder.setBytes(&sp, length: 8, index: 4)
            encoder.setBuffer(invBuffer, offset: 0, index: 5)
            encoder.dispatchThreadgroups(MTLSize(width: outDim / 64, height: rows, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
        }
        encoder.endEncoding()
        command.commit(); command.waitUntilCompleted()
        if let error = command.error { throw SploshError.capabilityGateFailure("mlp command failed: \(error)") }
        return Operand(values: Self.read(outB, cap * outDim, as: UInt16.self), sums: Self.read(outSums, cap * outGroups, as: Float.self),
                       gate16: gate16.map { Self.read($0, cap * outDim, as: UInt16.self) })
    }

    /// One element of a projection on the CPU: the affine q4 product, in double precision.
    func project(_ w: TiledWeight, row: Int, output n: Int) -> Double {
        w.packed.withUnsafeBufferPointer { codes in
            x.withUnsafeBufferPointer { activations in
                var total = 0.0
                for g in 0..<groups {
                    let slot = w.slot(output: n, group: g)
                    var dot: Float = 0     // exact: 64 products of an 8-bit mantissa and a 4-bit code
                    for k in 0..<64 {
                        let code = (codes[slot * 32 + k / 2] >> UInt8(4 * (k & 1))) & 15
                        dot += Q4BufferLayout.bf16(activations[row * inner + g * 64 + k]) * Float(code)
                    }
                    total += Double(Q4BufferLayout.bf16(w.scales[slot])) * Double(dot)
                        + Double(Q4BufferLayout.bf16(w.biases[slot])) * Double(xSums[row * groups + g])
                }
                return total
            }
        }
    }

    /// Every check on one variant's operand. `reference`: (row, output, gate, up) on the CPU
    /// for the elements to compare (all of them, or a sample).
    func check(variant: Int, shipped: Operand, reference: [(row: Int, n: Int, gate: Double, up: Double)]) throws {
        let candidate = try run(variant: variant)
        let plan = try #require(MlpFusedPlan(variant: variant, rows: rows, outDim: outDim))
        let tag = "variant \(variant) rows=\(rows) \(inner)->\(outDim)"
        // Sums in registers add in another order than simd_sum; a bf16 gate changes the values.
        let registerSums = plan.up.contains { $0.kernel.hasSuffix("_regsums") }
        let exactGate = plan.gate == nil

        // 1. Against shipped on the step's rows; nothing written past them. With an fp32 gate
        // the arithmetic is the same but not compiled the same, so the fp32 product can differ
        // in its last bits and, on a rounding boundary, land on the neighbouring bf16. On a
        // step of whole tiles that is at most a few values in 100,000 (often none). On a step
        // with a partial tile it is a few in 10,000, because there the shipped path computes
        // every tile with sp_na_tiled, whose epilogue is written differently from the small
        // whole-tile kernel the candidate is built on. A value further off than a bf16 step
        // fails, unless both are within fp32 accumulation error of zero (an up result that
        // cancels to nearly nothing has no relative accuracy in either path).
        // A sum may differ by what its values differ by (plus, for sums taken in registers,
        // fp32 rounding of a different order of addition); read-back sums whose values are all
        // equal must be equal to the bit.
        var valueMismatches = 0, distant = 0, sumMismatches = 0, unwritten = 0, strayValues = 0, straySums = 0, strayGate = 0
        var largestSumDifference: Float = 0, sumsOutside = 0, sumsNotExact = 0
        var shippedError = 0.0, shippedTotal = 0.0, largestDistant = 0.0
        for row in 0..<cap {
            for n in 0..<outDim {
                let index = row * outDim + n
                if row < rows {
                    if candidate.values[index] == Self.sentinel16 { unwritten += 1 }
                    if let gate16 = candidate.gate16, gate16[index] == Self.sentinel16 { unwritten += 1 }
                    let expected = Double(Q4BufferLayout.bf16(shipped.values[index]))
                    let actual = Double(Q4BufferLayout.bf16(candidate.values[index]))
                    if candidate.values[index] != shipped.values[index] {
                        valueMismatches += 1
                        let difference = abs(actual - expected)
                        if !(difference <= max(abs(actual), abs(expected)) / 127 || difference <= 2e-5) {
                            distant += 1; largestDistant = max(largestDistant, difference)
                        }
                    }
                    shippedError += abs(actual - expected); shippedTotal += abs(expected)
                } else {
                    if candidate.values[index] != Self.sentinel16 { strayValues += 1 }
                    if let gate16 = candidate.gate16, gate16[index] != Self.sentinel16 { strayGate += 1 }
                }
            }
            for group in 0..<outGroups {
                let index = row * outGroups + group
                if row < rows {
                    if candidate.sums[index].bitPattern == Self.sentinel32.bitPattern { unwritten += 1 }
                    var valueDifference: Float = 0, magnitude: Float = 0, stored: Float = 0
                    for k in 0..<64 {
                        let at = row * outDim + group * 64 + k
                        let value = Q4BufferLayout.bf16(candidate.values[at])
                        valueDifference += abs(value - Q4BufferLayout.bf16(shipped.values[at]))
                        magnitude += abs(value); stored += value
                    }
                    // 2. Each sum against the values it sums.
                    if !(abs(stored - candidate.sums[index]) <= 1e-5 * magnitude + 1e-6) { sumsOutside += 1 }
                    if candidate.sums[index].bitPattern != shipped.sums[index].bitPattern {
                        sumMismatches += 1
                        let difference = abs(candidate.sums[index] - shipped.sums[index])
                        largestSumDifference = max(largestSumDifference, difference)
                        if !(difference <= valueDifference + 1e-5 * magnitude + 1e-6) { sumsOutside += 1 }
                        if valueDifference == 0, !registerSums || row >= rows / 32 * 32 { sumsNotExact += 1 }
                    }
                } else if candidate.sums[index].bitPattern != Self.sentinel32.bitPattern { straySums += 1 }
            }
        }
        let shippedRelative = shippedError / max(shippedTotal, 1e-9)
        #expect(unwritten == 0, "\(tag): \(unwritten) operand elements never written")
        #expect(strayValues == 0 && straySums == 0 && strayGate == 0,
                "\(tag): wrote \(strayValues) values, \(straySums) sums and \(strayGate) gate values past the step's rows")
        #expect(sumsOutside == 0, "\(tag): \(sumsOutside) sums are not the sums of their values (largest difference from shipped \(largestSumDifference))")
        #expect(sumsNotExact == 0, "\(tag): \(sumsNotExact) read-back sums differ from the shipped ones although their values do not")
        if exactGate {
            #expect(distant == 0, "\(tag): \(distant) bf16 values differ from the shipped path by more than one step (largest \(largestDistant))")
            let allowed = rows * outDim / (rows % 32 == 0 ? 20_000 : 2_000)
            #expect(valueMismatches <= allowed, "\(tag): \(valueMismatches) of \(rows * outDim) bf16 values differ from the shipped path (limit \(allowed))")
        } else {
            // The gate was rounded to bf16 before SiLU: each value may move by about a step.
            #expect(shippedRelative <= 6e-3, "\(tag): relErr \(shippedRelative) against the shipped path")
        }

        // 3. Against the CPU: the affine q4 products in double precision, the scale, SiLU, the
        // product, rounded to bf16. The GPU accumulates in fp32, so a value may land on the
        // neighbouring bf16 (2^-8 relative). A bf16 gate is checked in two steps: the stored
        // gate against the CPU's, then the product against the CPU's from the stored gate.
        var largestRelative = 0.0, absoluteError = 0.0, absoluteTotal = 0.0, outside = 0, nonzero = 0, gateOutside = 0
        var exactError = 0.0, exactTotal = 0.0     // against the CPU's own, unrounded, gate
        for element in reference {
            let index = element.row * outDim + element.n
            let scale = Double(rowInv[element.row])
            var gate = element.gate
            let exact = gate * scale / (1 + exp(-gate * scale)) * (element.up * scale)
            exactError += abs(Double(Q4BufferLayout.bf16(candidate.values[index])) - exact); exactTotal += abs(exact)
            if let gate16 = candidate.gate16 {
                let stored = Double(Q4BufferLayout.bf16(gate16[index]))
                if !(abs(stored - gate) / max(abs(gate), 1e-3) <= 0.012) { gateOutside += 1 }
                gate = stored
            }
            gate *= scale
            let expected = gate / (1 + exp(-gate)) * (element.up * scale)
            let actual = Double(Q4BufferLayout.bf16(candidate.values[index]))
            absoluteError += abs(actual - expected); absoluteTotal += abs(expected)
            if expected != 0 { nonzero += 1 }
            let relative = abs(actual - expected) / max(abs(expected), 1e-3)
            largestRelative = max(largestRelative, relative)
            if !(relative <= 0.012) { outside += 1 }
        }
        let relErr = absoluteError / max(absoluteTotal, 1e-9)
        print("MlpFused \(tag) [\(plan.up.map(\.kernel).joined(separator: " + "))]: vs shipped: \(valueMismatches) of \(rows * outDim) values differ "
              + "(\(distant) by more than one bf16 step, relErr \(shippedRelative)), \(sumMismatches) sums differ (largest \(largestSumDifference)); "
              + "vs CPU on \(reference.count) elements: relErr=\(relErr) largest element error=\(largestRelative) (limit 0.012)"
              + (exactGate ? "" : "; vs CPU with its own unrounded gate: relErr=\(exactError / max(exactTotal, 1e-9))"))
        #expect(nonzero > reference.count / 2, "\(tag): the reference is mostly zero; the test would prove nothing")
        #expect(gateOutside == 0, "\(tag): \(gateOutside) bf16 gate values further than a bf16 step from the CPU reference")
        #expect(outside == 0, "\(tag): \(outside) values further than a bf16 step from the CPU reference")
        #expect(relErr <= 4e-3, "\(tag): relErr=\(relErr) against the CPU reference")
        #expect(exactError / max(exactTotal, 1e-9) <= 6e-3, "\(tag): relErr=\(exactError / max(exactTotal, 1e-9)) against the CPU reference with an unrounded gate")
    }
}

@Suite("MlpFusedTests")
struct MlpFusedTests {
    // Row counts: below a tile, a tile, whole tiles short of a block of four (96), a partial
    // tile after a first tile that is not a multiple of four (100: a trimmed last layer), a
    // block (128), a partial tile after a block (129), a second block with empty slots (160),
    // the same with a partial tile (161), and several blocks.
    @Test("every variant's operand equals gate GEMM + up GEMM + sp_silu_mul_na, and the CPU reference",
          arguments: [1, 7, 32, 96, 100, 128, 129, 160, 161, 512])
    func fusedMatchesShippedAndReference(rows: Int) throws {
        let inner = 256, outDim = 384
        let mlp = try MlpCase(rows: rows, gate: TiledWeight(outDim: outDim, inner: inner, seed: 11),
                              up: TiledWeight(outDim: outDim, inner: inner, seed: 23))
        let shipped = try mlp.run(variant: nil)
        var reference: [(row: Int, n: Int, gate: Double, up: Double)] = []
        for row in 0..<rows {
            for n in 0..<outDim {
                reference.append((row, n, mlp.project(mlp.gateWeight, row: row, output: n), mlp.project(mlp.upWeight, row: row, output: n)))
            }
        }
        for variant in MlpFusedPlan.variants { try mlp.check(variant: variant, shipped: shipped, reference: reference) }
    }

    // The model's own MLP shape (5120 -> 17408: 80 quant groups, 136 column tiles), so index
    // arithmetic that only goes wrong at scale shows. Every value is compared with the shipped
    // path; the CPU reference is taken on a sample that touches every column tile and row.
    @Test("the model's MLP shape", arguments: [100, 128])
    func modelShape(rows: Int) throws {
        let inner = 5120, outDim = 17408
        let mlp = try MlpCase(rows: rows, gate: TiledWeight(outDim: outDim, inner: inner, seed: 31),
                              up: TiledWeight(outDim: outDim, inner: inner, seed: 47))
        let shipped = try mlp.run(variant: nil)
        var random = SplitMix(state: 0xA11CE &+ UInt64(rows))
        var reference: [(row: Int, n: Int, gate: Double, up: Double)] = []
        for sample in 0..<2720 {
            // Twenty per column tile, at a random column of it and a row that cycles.
            let n = (sample % 136) * 128 + Int(random.next() % 128), row = (sample * 7 + Int(random.next() % 7)) % rows
            reference.append((row, n, mlp.project(mlp.gateWeight, row: row, output: n), mlp.project(mlp.upWeight, row: row, output: n)))
        }
        for variant in MlpFusedPlan.variants { try mlp.check(variant: variant, shipped: shipped, reference: reference) }
    }

    @Test("the plan covers every row of a step exactly once")
    func planCoversRows() throws {
        for variant in MlpFusedPlan.variants {
            for rows in 1...300 {
                let plan = try #require(MlpFusedPlan(variant: variant, rows: rows, outDim: 17408))
                var covered = [Int](repeating: 0, count: rows + 64)
                for dispatch in plan.up {
                    // The kernels' own rule: tile = firstTile + z * 4 + x, live while its first row is below `rows`.
                    for z in 0..<dispatch.grid.depth {
                        for x in 0..<dispatch.grid.width {
                            let first = (dispatch.firstTile + z * 4 + x) * 32
                            guard first < dispatch.rows else { continue }
                            let whole = !dispatch.kernel.hasSuffix("_partial")
                            for row in first..<(whole ? first + 32 : min(first + 32, dispatch.rows)) { covered[row] += 1 }
                        }
                    }
                    #expect(dispatch.grid.height == 136)
                    #expect(MlpFusedPlan.kernels(variant: variant).contains(dispatch.kernel))
                }
                #expect(covered[..<rows].allSatisfy { $0 == 1 } && covered[rows...].allSatisfy { $0 == 0 }, "variant \(variant) rows=\(rows)")
                // A bf16 gate only on steps of whole tiles, over all of them.
                #expect((plan.gate != nil) == (variant >= 3 && rows % 32 == 0), "variant \(variant) rows=\(rows)")
                if let gate = plan.gate {
                    #expect(gate.rows == rows && gate.firstTile == 0 && gate.grid.width * gate.grid.depth >= rows / 32)
                    #expect(MlpFusedPlan.kernels(variant: variant).contains(gate.kernel))
                }
            }
        }
        #expect(MlpFusedPlan(variant: 0, rows: 128, outDim: 17408) == nil && MlpFusedPlan(variant: 5, rows: 128, outDim: 17408) == nil)
    }
}
