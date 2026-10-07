import Accelerate
import Foundation
import Metal
import Testing

import SploshCore
import SploshModel
import SploshQuant

// The GGUF weight kernels (Sources/Shaders/engine_gguf.metal) against a CPU reference, a format
// at a time. A synthetic tensor of random native blocks goes two ways: through the reference
// decoder (GgufTensorType.decode), each weight then rounded to fp16 as the kernels stage it, and
// through GgufPlanes.repack to the planes the kernels read. So a test covers the repack, the
// per-format decoding in the shader and the tile arithmetic together.
//
// Six tests a format. The first reads the staged weights back one at a time, through
// activations that are 1 at a single input, and compares them to the reference in fp16 units.
// The second runs every kernel shape on random activations at whole and partial steps and
// checks the sums, that every output of a row in the step was written and that none of a row
// past it was. The third runs each shape's `heads` form against the plain one on activations
// whose head blocks were moved to the file's order first. The fourth gathers embedding rows and
// the fifth runs the small GEMM on a 48-row weight; both decode to fp32, so they are held to
// the reference's values unrounded, the embedding bit for bit. The sixth runs each tile shape
// with one buffer bound as its out and as its residual, which is how the engine adds a
// projection to the hidden state, and holds it to the two-buffer result bit for bit.

private struct KernelParams {
    var rows, outDim, inner, groups, hasResidual, outStride: UInt32
}

private struct EmbedParams {
    var rows, strideWords, groupsPerRow, hidden: UInt32
}

private struct Generator {
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

/// Where a native block keeps its fp16 scale fields, each with the magnitude that puts a
/// decoded weight near 0.05: the scale is multiplied by a block's integer scales and codes,
/// which for random bytes are about half their range.
private func scaleFields(of type: GgufTensorType) -> [(offset: Int, magnitude: Float)] {
    switch type {
    case .q4K: return [(0, 2e-4), (2, 1.5e-3)]      // d * [0, 63] * [0, 15] - dmin * [0, 63]
    case .q5K: return [(0, 1e-4), (2, 1.5e-3)]      // d * [0, 63] * [0, 31] - dmin * [0, 63]
    case .q6K: return [(208, 5e-5)]                 // d * [-128, 127] * [-32, 31]
    case .q3K: return [(108, 1.5e-3)]               // d * [-32, 31] * [-4, 3]
    case .q8_0: return [(0, 8e-4)]                  // d * [-128, 127]
    case .iq4NL: return [(0, 1e-3)]                 // d * [-127, 113]
    case .iq4XS: return [(0, 6e-5)]                 // d * [-32, 31] * [-127, 113]
    case .iq3S: return [(0, 4e-4)]                  // d * [1, 31] * [-15, 15]
    case .f32, .f16, .bf16, .q4_0: return []
    }
}

/// Rows of native blocks of random bytes. A scale field is 0.75 to 1.25 of its magnitude, and
/// negative in one block of four: nothing in a format requires it positive.
private func syntheticRows(of type: GgufTensorType, rows: Int, inner: Int, seed: UInt64) -> [UInt8] {
    var random = Generator(state: seed)
    let size = type.blockBytes, blocks = rows * inner / type.blockElements
    var bytes = [UInt8](repeating: 0, count: blocks * size)
    bytes.withUnsafeMutableBytes { raw in
        for index in 0..<raw.count { raw[index] = UInt8(truncatingIfNeeded: random.next()) }
        for block in 0..<blocks {
            for field in scaleFields(of: type) {
                var value = field.magnitude * (1 + 0.25 * random.unit())
                if random.next() & 3 == 0 { value = -value }
                raw.storeBytes(of: Float16(value).bitPattern.littleEndian, toByteOffset: block * size + field.offset, as: UInt16.self)
            }
        }
    }
    return bytes
}

/// An fp16 value's place among the fp16 values, so that neighbours differ by one.
private func ordinal(_ value: Float16) -> Int {
    let magnitude = Int(value.bitPattern & 0x7FFF)
    return value.sign == .minus ? -magnitude : magnitude
}

private struct Harness {
    let device: MTLDevice
    let library: Metallib
    let queue: MTLCommandQueue

    init() throws {
        device = try #require(MTLCreateSystemDefaultDevice())
        library = try Metallib(device: device)
        queue = try #require(device.makeCommandQueue())
    }

    func buffer<T>(_ values: [T]) throws -> MTLBuffer {
        try #require(values.withUnsafeBytes { device.makeBuffer(bytes: $0.baseAddress!, length: $0.count, options: .storageModeShared) })
    }
}

/// A synthetic weight in the planes, with the reference's values.
private struct Weight {
    let outDim: Int, inner: Int
    let suffix: String
    let plane0: MTLBuffer, plane1: MTLBuffer, meta: MTLBuffer
    /// [output][input]: each weight as the reference decodes it.
    let decoded: [Float]
    /// The same rounded to fp16, as the tile kernels stage it.
    let values: [Float]

    /// The planes hold whole tiles: rows past `outDim` are zero blocks, as the converter pads a
    /// 48-row weight.
    init(_ type: GgufTensorType, outDim: Int, inner: Int, harness: Harness) throws {
        self.outDim = outDim; self.inner = inner
        suffix = try #require(GgufPlanes.geometry(of: type)).kernelSuffix
        var native = syntheticRows(of: type, rows: outDim, inner: inner, seed: 0x6600 + UInt64(type.rawValue))
        var decoded = [Float](repeating: 0, count: outDim * inner)
        native.withUnsafeBytes { raw in
            decoded.withUnsafeMutableBufferPointer { GgufTensorType.decode(type, blocks: raw, into: $0) }
        }
        self.decoded = decoded
        values = decoded.map { Float(Float16($0)) }
        let stored = (outDim + GgufPlanes.tileRows - 1) / GgufPlanes.tileRows * GgufPlanes.tileRows
        native += [UInt8](repeating: 0, count: (stored - outDim) * inner / type.blockElements * type.blockBytes)
        let planes = native.withUnsafeBytes { GgufPlanes.repack(type, native: $0, rows: stored, inner: inner) }
        plane0 = try harness.buffer(planes.plane0)
        // A format without a second plane never reads the binding.
        plane1 = planes.plane1.isEmpty ? plane0 : try harness.buffer(planes.plane1)
        meta = try harness.buffer(planes.meta)
    }
}

/// One kernel shape and the steps to run it at.
private struct Shape {
    let name: String
    /// The kernel's name up to its format suffix.
    let kernel: String
    /// The rows and the outputs one threadgroup takes, and its threads.
    let tileRows: Int, columns: Int, threads: Int
    let steps: [Int]
}

@Suite("GgufKernelTests", .serialized)
struct GgufKernelTests {
    /// The buffers hold `cap` rows, so a kernel that lets a row past the step through is seen.
    static let outDim = 256, inner = 1280, cap = 160
    static let formats: [GgufTensorType] = [.q4K, .q5K, .q6K, .q3K, .q8_0, .iq4NL, .iq4XS, .iq3S]
    /// The steps asked for, and one of several tiles of rows for each of the two grids: 43 rows
    /// are three 16-row tiles, 150 are two 128-row ones with a partial 32-row tile in the second.
    private static let shapes = [
        Shape(name: "split m16", kernel: "sp_gguf_split_m16_", tileRows: 16, columns: 32, threads: 64, steps: [16, 11, 43]),
        Shape(name: "split m32", kernel: "sp_gguf_split_m32_", tileRows: 32, columns: 32, threads: 64, steps: [27]),
        Shape(name: "wide", kernel: "sp_gguf_wide_", tileRows: 128, columns: 64, threads: 128, steps: [128, 96, 75, 150]),
    ]
    static let sentinel = Float(bitPattern: 0xFFFF_FFFF)

    /// One dispatch of a GEMM `kernel` over `rows` rows, on `grid` threadgroups of `threads`;
    /// the residual is added when there is one.
    private func dispatch(_ kernel: String, weight: Weight, activations: MTLBuffer, out: MTLBuffer, residual: MTLBuffer?,
                          rows: Int, outStride: Int, grid: MTLSize, threads: Int, harness: Harness) throws {
        let command = try #require(harness.queue.makeCommandBuffer())
        let encoder = try #require(command.makeComputeCommandEncoder(dispatchType: .concurrent))
        var p = KernelParams(rows: UInt32(rows), outDim: UInt32(weight.outDim), inner: UInt32(weight.inner),
                             groups: UInt32(weight.inner / GgufPlanes.groupElements),
                             hasResidual: residual == nil ? 0 : 1, outStride: UInt32(outStride))
        encoder.setComputePipelineState(try harness.library.pipeline(kernel))
        encoder.setBuffer(weight.plane0, offset: 0, index: 0)
        encoder.setBuffer(weight.plane1, offset: 0, index: 1)
        encoder.setBuffer(weight.meta, offset: 0, index: 2)
        encoder.setBuffer(activations, offset: 0, index: 3)
        encoder.setBuffer(out, offset: 0, index: 4)
        encoder.setBuffer(residual ?? out, offset: 0, index: 5)
        encoder.setBytes(&p, length: MemoryLayout<KernelParams>.stride, index: 7)
        encoder.dispatchThreadgroups(grid, threadsPerThreadgroup: MTLSize(width: threads, height: 1, depth: 1))
        encoder.endEncoding()
        command.commit()
        command.waitUntilCompleted()
        try #require(command.status == .completed, "\(kernel): \(String(describing: command.error))")
    }

    /// One dispatch of a tile `kernel` of `shape` over `rows` rows.
    private func run(_ kernel: String, _ shape: Shape, weight: Weight, activations: MTLBuffer, out: MTLBuffer,
                     residual: MTLBuffer?, rows: Int, harness: Harness) throws {
        let grid = MTLSize(width: weight.outDim / shape.columns, height: (rows + shape.tileRows - 1) / shape.tileRows, depth: 1)
        try dispatch(kernel, weight: weight, activations: activations, out: out, residual: residual, rows: rows,
                     outStride: weight.outDim, grid: grid, threads: shape.threads, harness: harness)
    }

    /// Every row's sums against the reference weights `weights` ([output][input]), in Double: x
    /// times the transpose.
    private func referenceSums(_ x: [UInt16], rows: Int, weights: [Float], outDim: Int, inner: Int) -> [Double] {
        let xValues = x.map { Double(Q4BufferLayout.bf16($0)) }
        var transposed = [Double](repeating: 0, count: inner * outDim)
        for n in 0..<outDim {
            for k in 0..<inner { transposed[k * outDim + n] = Double(weights[n * inner + k]) }
        }
        var expected = [Double](repeating: 0, count: rows * outDim)
        vDSP_mmulD(xValues, 1, transposed, 1, &expected, 1, vDSP_Length(rows), vDSP_Length(outDim), vDSP_Length(inner))
        return expected
    }

    @Test("Staged weights are the reference's to one fp16 unit in the last place", arguments: formats)
    func stagedWeights(_ type: GgufTensorType) throws {
        let harness = try Harness()
        let outDim = Self.outDim, inner = Self.inner, rows = 128
        let weight = try Weight(type, outDim: outDim, inner: inner, harness: harness)
        let wide = Self.shapes[2]
        let one = Q4BufferLayout.bf16Bits(1)
        var exact = 0, adjacent = 0, further = 0, unstaged = 0
        for pass in 0..<(inner / rows) {
            // Row r is 1 at input 128 * pass + r and 0 elsewhere, so its output n is weight
            // (n, 128 * pass + r) as staged: the sum of one weight and zeros is that weight.
            var x = [UInt16](repeating: 0, count: rows * inner)
            for row in 0..<rows { x[row * inner + pass * rows + row] = one }
            let out = try harness.buffer([Float](repeating: Self.sentinel, count: rows * outDim))
            try run(wide.kernel + weight.suffix, wide, weight: weight, activations: try harness.buffer(x), out: out,
                    residual: nil, rows: rows, harness: harness)
            let result = UnsafeBufferPointer(start: out.contents().bindMemory(to: Float.self, capacity: rows * outDim), count: rows * outDim)
            for row in 0..<rows {
                for n in 0..<outDim {
                    let observed = result[row * outDim + n], expected = weight.values[n * inner + pass * rows + row]
                    // Not an fp16 value: not a staged weight, whatever it is.
                    guard observed.isFinite, Float(Float16(observed)) == observed else { unstaged += 1; continue }
                    switch abs(ordinal(Float16(observed)) - ordinal(Float16(expected))) {
                    case 0: exact += 1
                    case 1: adjacent += 1
                    default: further += 1
                    }
                }
            }
        }
        print("gguf kernels \(type): \(exact) staged weights exact, \(adjacent) one fp16 unit off, \(further) further, \(unstaged) not fp16")
        #expect(unstaged == 0, "\(type): \(unstaged) outputs are not fp16 values")
        #expect(further == 0, "\(type): \(further) staged weights are more than one fp16 unit from the reference")
        #expect(exact + adjacent == outDim * inner)
    }

    @Test("Every kernel shape matches the reference at whole and partial steps", arguments: formats)
    func matchReference(_ type: GgufTensorType) throws {
        let harness = try Harness()
        let outDim = Self.outDim, inner = Self.inner, cap = Self.cap
        let weight = try Weight(type, outDim: outDim, inner: inner, harness: harness)
        var random = Generator(state: 0xAC71 + UInt64(type.rawValue))
        let x: [UInt16] = (0..<(cap * inner)).map { _ in Q4BufferLayout.bf16Bits(random.unit()) }
        let residual: [Float] = (0..<(cap * outDim)).map { _ in random.unit() }
        let xBuffer = try harness.buffer(x), residualBuffer = try harness.buffer(residual)

        let expected = referenceSums(x, rows: cap, weights: weight.values, outDim: outDim, inner: inner)

        // An output is a sum of 1280 products of an activation in [-1, 1) and a weight of about
        // 0.05, at times with a residual in [-1, 1): of order 1, a few at most. The kernels
        // accumulate it in fp32 in an order of their own where the reference sums in Double, so
        // each of some 1300 additions can lose half a unit in the last place of the running sum,
        // about 1e-7 at these sizes. As the losses fall that is a few 1e-6 (the largest seen is
        // 6e-6); it would take every one falling the same way to reach 1e-4. A staged weight may
        // also be the fp16 neighbour of the reference's: stagedWeights finds none in these
        // tensors, but a Q4_K or Q5_K weight is made with one fused multiply-add where the
        // reference rounds twice, and one such weight under 0.125 moves a sum by up to 6e-5.
        // The bound sits over both and under what a wrong code does: the smallest step of any of
        // these formats is one Q8_0 code, 8e-4 of weight and so 4e-4 of a sum at the mean
        // activation of 0.5.
        let bound = 1e-4
        var report: [String] = []
        let residualFirst = Self.formats.firstIndex(of: type)! % 2 == 0
        var step = 0
        for shape in Self.shapes {
            var shapeWorst = 0.0
            for rows in shape.steps {
                // The residual is on at every other step, starting on or off by the format.
                let withResidual = (step % 2 == 0) == residualFirst
                step += 1
                let out = try harness.buffer([Float](repeating: Self.sentinel, count: cap * outDim))
                try run(shape.kernel + weight.suffix, shape, weight: weight, activations: xBuffer, out: out,
                        residual: withResidual ? residualBuffer : nil, rows: rows, harness: harness)
                let result = UnsafeBufferPointer(start: out.contents().bindMemory(to: Float.self, capacity: cap * outDim), count: cap * outDim)
                var worst = 0.0, untouched = 0, invalid = 0, written = 0
                for row in 0..<cap {
                    for n in 0..<outDim {
                        let observed = result[row * outDim + n]
                        let isSentinel = observed.bitPattern == Self.sentinel.bitPattern
                        if row >= rows {
                            if !isSentinel { written += 1 }
                        } else if isSentinel {
                            untouched += 1
                        } else if !observed.isFinite {
                            invalid += 1
                        } else {
                            let sum = expected[row * outDim + n] + (withResidual ? Double(residual[row * outDim + n]) : 0)
                            worst = max(worst, abs(Double(observed) - sum))
                        }
                    }
                }
                let name = "\(shape.kernel + weight.suffix) at \(rows) rows\(withResidual ? " with a residual" : "")"
                #expect(untouched == 0, "\(name) left \(untouched) outputs unwritten")
                #expect(invalid == 0, "\(name) wrote \(invalid) outputs that are not finite")
                #expect(written == 0, "\(name) wrote \(written) outputs of rows past the step")
                #expect(worst < bound, "\(name): largest error \(worst)")
                shapeWorst = max(shapeWorst, worst)
            }
            report.append(String(format: "%@ %.2e", shape.name, shapeWorst))
        }
        print("gguf kernels \(type): largest error " + report.joined(separator: ", ") + String(format: " (bound %.0e)", bound))
    }

    @Test("A heads kernel on the engine's head order is the plain kernel on the file's", arguments: formats)
    func headOrder(_ type: GgufTensorType) throws {
        let harness = try Harness()
        // The out projection's inner: 48 value heads of 128 columns.
        let outDim = 128, heads = 48, headDim = 128, inner = heads * headDim, cap = Self.cap
        let weight = try Weight(type, outDim: outDim, inner: inner, harness: harness)
        var random = Generator(state: 0x4EAD + UInt64(type.rawValue))
        // Activations as the engine lays them out, and the same with each head's block moved to
        // where the file has that head: the file's head h is the engine's 3 (h % 16) + h / 16.
        let engineOrder: [UInt16] = (0..<(cap * inner)).map { _ in Q4BufferLayout.bf16Bits(random.unit()) }
        var fileOrder = engineOrder
        for row in 0..<cap {
            for head in 0..<heads {
                let from = row * inner + GgufNames.mlxValueHead(ofGgufHead: head) * headDim, to = row * inner + head * headDim
                fileOrder.replaceSubrange(to ..< to + headDim, with: engineOrder[from ..< from + headDim])
            }
        }
        #expect(fileOrder != engineOrder)
        let residual: [Float] = (0..<(cap * outDim)).map { _ in random.unit() }
        let engineBuffer = try harness.buffer(engineOrder), fileBuffer = try harness.buffer(fileOrder)
        let residualBuffer = try harness.buffer(residual)
        // The kernels are held to the reference at this inner too, so that the two do not agree
        // on something wrong. A sum is of 6144 products here, five times matchReference's, and
        // larger with it: the largest error seen is 3.4e-5. The bound is twice matchReference's,
        // still half of what one wrong Q8_0 code does.
        let expected = referenceSums(fileOrder, rows: cap, weights: weight.values, outDim: outDim, inner: inner)
        let bound = 2e-4
        var worst = 0.0, step = 0
        for shape in Self.shapes {
            for rows in shape.steps.suffix(2) {
                let withResidual = step % 2 == 0
                step += 1
                var results: [[UInt32]] = []
                for (kernel, x) in [(shape.kernel + weight.suffix, fileBuffer), (shape.kernel + "heads_" + weight.suffix, engineBuffer)] {
                    let out = try harness.buffer([Float](repeating: Self.sentinel, count: cap * outDim))
                    try run(kernel, shape, weight: weight, activations: x, out: out,
                            residual: withResidual ? residualBuffer : nil, rows: rows, harness: harness)
                    results.append(Array(UnsafeBufferPointer(start: out.contents().bindMemory(to: UInt32.self, capacity: cap * outDim), count: cap * outDim)))
                }
                let name = "\(shape.kernel)heads_\(weight.suffix) at \(rows) rows\(withResidual ? " with a residual" : "")"
                // The same matmuls on the same operands in the same order: equal bits, and the
                // rows past the step are the sentinel in both.
                let different = zip(results[0], results[1]).filter { $0 != $1 }.count
                #expect(different == 0, "\(name): \(different) outputs differ from the plain kernel's on the file's order")
                var untouched = 0
                for row in 0..<rows {
                    for n in 0..<outDim {
                        let bits = results[1][row * outDim + n]
                        if bits == Self.sentinel.bitPattern { untouched += 1; continue }
                        let sum = expected[row * outDim + n] + (withResidual ? Double(residual[row * outDim + n]) : 0)
                        worst = max(worst, abs(Double(Float(bitPattern: bits)) - sum))
                    }
                }
                #expect(untouched == 0, "\(name) left \(untouched) outputs unwritten")
            }
        }
        #expect(worst < bound, "\(type): largest error \(worst) against the reference")
        print("gguf kernels \(type): heads forms equal the plain ones bit for bit at \(step) steps, " + String(format: "largest error %.2e (bound %.0e)", worst, bound))
    }

    @Test("The embedding gathers rows by token id and decodes them as the reference does, bit for bit", arguments: formats)
    func embedding(_ type: GgufTensorType) throws {
        let harness = try Harness()
        // A vocabulary of four tiles. The ids are in each of them, at both ends of a tile, and
        // one is asked for twice; the ids past the step are valid but must not be gathered.
        let vocab = 512, hidden = Self.inner, cap = 16
        let weight = try Weight(type, outDim: vocab, inner: hidden, harness: harness)
        let tokens: [UInt32] = [0, 127, 128, 300, 511, 5, 256, 383, 300, 129, 64, 448]
        let rows = tokens.count, groups = hidden / GgufPlanes.groupElements
        let ids = try harness.buffer(tokens + [UInt32](repeating: 1, count: cap - rows))
        let out = try harness.buffer([Float](repeating: Self.sentinel, count: cap * hidden))
        let kernel = "sp_gguf_embed_" + weight.suffix
        let pipeline = try harness.library.pipeline(kernel)
        let command = try #require(harness.queue.makeCommandBuffer())
        let encoder = try #require(command.makeComputeCommandEncoder())
        var p = EmbedParams(rows: UInt32(rows), strideWords: 0, groupsPerRow: UInt32(groups), hidden: UInt32(hidden))
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(weight.plane0, offset: 0, index: 0)
        encoder.setBuffer(weight.plane1, offset: 0, index: 1)
        encoder.setBuffer(weight.meta, offset: 0, index: 2)
        encoder.setBuffer(ids, offset: 0, index: 3)
        encoder.setBuffer(out, offset: 0, index: 4)
        encoder.setBytes(&p, length: MemoryLayout<EmbedParams>.stride, index: 5)
        // A grid wider than a row's groups and taller than the step: the kernel bounds both.
        encoder.dispatchThreads(MTLSize(width: groups + 24, height: cap, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: pipeline.threadExecutionWidth, height: 1, depth: 1))
        encoder.endEncoding()
        command.commit()
        command.waitUntilCompleted()
        try #require(command.status == .completed, "\(kernel): \(String(describing: command.error))")

        let result = UnsafeBufferPointer(start: out.contents().bindMemory(to: Float.self, capacity: cap * hidden), count: cap * hidden)
        var different = 0, written = 0
        for row in 0..<cap {
            for k in 0..<hidden {
                let bits = result[row * hidden + k].bitPattern
                if row >= rows {
                    if bits != Self.sentinel.bitPattern { written += 1 }
                } else if bits != weight.decoded[Int(tokens[row]) * hidden + k].bitPattern {
                    different += 1
                }
            }
        }
        #expect(different == 0, "\(kernel): \(different) of \(rows * hidden) values are not the reference's bits")
        #expect(written == 0, "\(kernel) wrote \(written) values of rows past the step")
        print("gguf kernels \(type): \(rows * hidden - different) of \(rows * hidden) embedding values equal the reference's bits")
    }

    @Test("The small GEMM matches the reference on a 48-row weight and writes nothing past it", arguments: formats)
    func smallGemm(_ type: GgufTensorType) throws {
        let harness = try Harness()
        // The planes hold a tile; the 80 rows past the weight's 48 are zero. The out buffer is
        // wider than the weight and longer than a step, and so is the grid.
        let outDim = 48, inner = Self.inner, cap = 8, outStride = 64
        let weight = try Weight(type, outDim: outDim, inner: inner, harness: harness)
        var random = Generator(state: 0x5A11 + UInt64(type.rawValue))
        let x: [UInt16] = (0..<(cap * inner)).map { _ in Q4BufferLayout.bf16Bits(random.unit()) }
        let residual: [Float] = (0..<(cap * outStride)).map { _ in random.unit() }
        let xBuffer = try harness.buffer(x), residualBuffer = try harness.buffer(residual)
        let expected = referenceSums(x, rows: cap, weights: weight.decoded, outDim: outDim, inner: inner)

        // The weights are the reference's exactly, so what is left is the order of 1280 fp32
        // additions: a few 1e-6 as in matchReference, far under the 4e-4 of one wrong code.
        let bound = 2e-5
        let kernel = "sp_gguf_small_" + weight.suffix
        var worst = 0.0
        for (rows, withResidual) in [(5, false), (cap, true), (1, true)] {
            let out = try harness.buffer([Float](repeating: Self.sentinel, count: cap * outStride))
            try dispatch(kernel, weight: weight, activations: xBuffer, out: out, residual: withResidual ? residualBuffer : nil,
                         rows: rows, outStride: outStride, grid: MTLSize(width: outStride, height: cap, depth: 1), threads: 32, harness: harness)
            let result = UnsafeBufferPointer(start: out.contents().bindMemory(to: Float.self, capacity: cap * outStride), count: cap * outStride)
            var untouched = 0, invalid = 0, written = 0
            for row in 0..<cap {
                for n in 0..<outStride {
                    let observed = result[row * outStride + n]
                    let isSentinel = observed.bitPattern == Self.sentinel.bitPattern
                    if row >= rows || n >= outDim {
                        if !isSentinel { written += 1 }
                    } else if isSentinel {
                        untouched += 1
                    } else if !observed.isFinite {
                        invalid += 1
                    } else {
                        let sum = expected[row * outDim + n] + (withResidual ? Double(residual[row * outStride + n]) : 0)
                        worst = max(worst, abs(Double(observed) - sum))
                    }
                }
            }
            let name = "\(kernel) at \(rows) rows\(withResidual ? " with a residual" : "")"
            #expect(untouched == 0, "\(name) left \(untouched) outputs unwritten")
            #expect(invalid == 0, "\(name) wrote \(invalid) outputs that are not finite")
            #expect(written == 0, "\(name) wrote \(written) outputs past the weight's rows or the step")
        }
        #expect(worst < bound, "\(kernel): largest error \(worst)")
        print("gguf kernels \(type): small GEMM largest error " + String(format: "%.2e (bound %.0e)", worst, bound))
    }

    @Test("A residual in the out buffer itself, as the engine binds it, is added in place", arguments: formats)
    func residualInPlace(_ type: GgufTensorType) throws {
        let harness = try Harness()
        let outDim = Self.outDim, inner = Self.inner, cap = Self.cap
        let weight = try Weight(type, outDim: outDim, inner: inner, harness: harness)
        var random = Generator(state: 0x1A5E + UInt64(type.rawValue))
        let x: [UInt16] = (0..<(cap * inner)).map { _ in Q4BufferLayout.bf16Bits(random.unit()) }
        // What the buffer holds before the pass: the hidden state a projection is added to.
        let prior: [Float] = (0..<(cap * outDim)).map { _ in random.unit() }
        let xBuffer = try harness.buffer(x), priorBuffer = try harness.buffer(prior)
        let expected = referenceSums(x, rows: cap, weights: weight.values, outDim: outDim, inner: inner)

        // A tile loads its residual before its first matmul and stores after its last, and no
        // tile stores where another loads, so one buffer at both bindings has to give what two
        // give, bit for bit: the same matmuls on the same operands in the same order. It is held
        // to the reference as well, by matchReference's bound, and a row past the step has to
        // keep what the buffer held.
        let bound = 1e-4
        // The steps the engine gives each shape (up to 16 rows, 17 to 32, more), whole and partial.
        let steps = [[16, 11, 1], [32, 27], [128, 75, 150]]
        var worst = 0.0, passes = 0
        for (shape, rowCounts) in zip(Self.shapes, steps) {
            for rows in rowCounts {
                let kernel = shape.kernel + weight.suffix
                let inPlace = try harness.buffer(prior)
                try run(kernel, shape, weight: weight, activations: xBuffer, out: inPlace, residual: inPlace, rows: rows, harness: harness)
                let apart = try harness.buffer([Float](repeating: Self.sentinel, count: cap * outDim))
                try run(kernel, shape, weight: weight, activations: xBuffer, out: apart, residual: priorBuffer, rows: rows, harness: harness)
                let result = UnsafeBufferPointer(start: inPlace.contents().bindMemory(to: Float.self, capacity: cap * outDim), count: cap * outDim)
                let twoBuffers = UnsafeBufferPointer(start: apart.contents().bindMemory(to: Float.self, capacity: cap * outDim), count: cap * outDim)
                var different = 0, invalid = 0, disturbed = 0
                for row in 0..<cap {
                    for n in 0..<outDim {
                        let index = row * outDim + n, observed = result[index]
                        if row >= rows {
                            if observed.bitPattern != prior[index].bitPattern { disturbed += 1 }
                            continue
                        }
                        if observed.bitPattern != twoBuffers[index].bitPattern { different += 1 }
                        if !observed.isFinite { invalid += 1; continue }
                        worst = max(worst, abs(Double(observed) - expected[index] - Double(prior[index])))
                    }
                }
                let name = "\(kernel) at \(rows) rows, out and residual one buffer"
                #expect(different == 0, "\(name): \(different) outputs differ from the two-buffer pass")
                #expect(invalid == 0, "\(name) wrote \(invalid) outputs that are not finite")
                #expect(disturbed == 0, "\(name) changed \(disturbed) values of rows past the step")
                passes += 1
            }
        }
        #expect(worst < bound, "\(type): largest error \(worst) against the reference")
        print("gguf kernels \(type): in place equals two buffers bit for bit at \(passes) steps, " + String(format: "largest error %.2e (bound %.0e)", worst, bound))
    }
}
