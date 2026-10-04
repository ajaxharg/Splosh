import Testing
import Metal
import Foundation

import SploshCore
import SploshQuant

// The weight kernels that are not the shipped q4 ones, each against a CPU reference on
// synthetic weights in the tiled layout:
//
//   * the 8-bit forms of the shipped kernels (engine_na.metal: `sp_gemm_q8_...`),
//   * the group-K candidates (candidates/gemm_group.metal: `sp_gk_...`), for unsigned codes
//     with a scale and an offset per group and signed codes with a scale alone, in groups of
//     64, 32 and 16 columns,
//   * the staged candidates (candidates/gguf_staged.metal: `sp_gs_...`), which dequantise to
//     fp16 in threadgroup memory; their reference rounds each weight to fp16 the same way.
//
// A layout is [tile of 128 outputs][quant group][output in tile][codes of one group], with
// bf16 scales and offsets [tile][group][output]; activations are bf16 with fp32 sums per group.

private struct KernelParams {
    var rows, outDim, inner, groups, hasResidual, outStride: UInt32
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

/// One weight in the tiled layout, with its values for the reference.
private struct TiledWeight {
    let outDim: Int, inner: Int
    /// Columns per quant group, bits per code, whether codes are two's complement, and whether
    /// a group has an offset as well as a scale.
    let group: Int, bits: Int, signed: Bool, affine: Bool
    var packed: [UInt8]
    var scales: [UInt16], offsets: [UInt16]

    var groups: Int { inner / group }
    var elementBytes: Int { group * bits / 8 }

    init(outDim: Int, inner: Int, group: Int, bits: Int, signed: Bool, affine: Bool, seed: UInt64) {
        self.outDim = outDim; self.inner = inner; self.group = group; self.bits = bits; self.signed = signed; self.affine = affine
        var random = Generator(state: seed)
        let groups = inner / group
        packed = (0..<(outDim * groups * group * bits / 8)).map { _ in UInt8(truncatingIfNeeded: random.next()) }
        scales = [UInt16](repeating: 0, count: outDim * groups); offsets = scales
        let midpoint: Float = signed ? 0 : Float((1 << bits) - 1) / 2
        let range = Float(1 << bits)
        for index in 0..<(outDim * groups) {
            // Weights of a group span about +-0.1 whatever the code width.
            let scale = (0.2 + 0.1 * random.unit()) / range
            scales[index] = Q4BufferLayout.bf16Bits(scale)
            offsets[index] = Q4BufferLayout.bf16Bits(-midpoint * scale + 0.01 * random.unit())
        }
    }

    func slot(output n: Int, group g: Int) -> Int { ((n / 128) * groups + g) * 128 + n % 128 }

    func code(output n: Int, column k: Int) -> Float {
        let base = slot(output: n, group: k / group) * elementBytes, j = k % group
        if bits == 8 {
            return signed ? Float(Int8(bitPattern: packed[base + j])) : Float(packed[base + j])
        }
        let nibble = Int(j % 2 == 0 ? packed[base + j / 2] & 0xF : packed[base + j / 2] >> 4)
        return Float(signed && nibble >= 8 ? nibble - 16 : nibble)
    }

    /// The weight as the accelerator kernels apply it, or, staged, as rounded to fp16.
    func weight(output n: Int, column k: Int, staged: Bool) -> Float {
        let index = slot(output: n, group: k / group)
        let scale = Q4BufferLayout.bf16(scales[index]), offset = affine ? Q4BufferLayout.bf16(offsets[index]) : 0
        let value = code(output: n, column: k) * scale + offset
        return staged ? Float(Float16(value)) : value
    }
}

private struct Kernel {
    let name: String
    let weight: TiledWeight
    let rows: Int
    let grid: MTLSize
    let threads: Int
    var staged = false
    /// Whether rows past `rows` in the last tile must be left alone (the whole-tile kernels
    /// are only dispatched on whole tiles).
    var guardsRows = true
}

@Suite("GroupGemmTests")
struct GroupGemmTests {
    static let outDim = 256, inner = 1280, cap = 128

    private static func weight(_ group: Int, _ bits: Int, signed: Bool, affine: Bool = true) -> TiledWeight {
        TiledWeight(outDim: outDim, inner: inner, group: group, bits: bits, signed: signed, affine: affine,
                    seed: UInt64(group * 1000 + bits * 10 + (signed ? 1 : 0)))
    }

    private static var kernels: [Kernel] {
        let wideGrid = MTLSize(width: 4, height: outDim / 128, depth: 1)
        func splitGrid(_ columns: Int) -> MTLSize { MTLSize(width: outDim / columns, height: 1, depth: 1) }
        var list: [Kernel] = []
        // The 8-bit forms of the shipped kernels.
        let q8 = weight(64, 8, signed: false)
        list.append(Kernel(name: "sp_gemm_q8_nat_m32n128s4_whole", weight: q8, rows: 128, grid: wideGrid, threads: 128, guardsRows: false))
        list.append(Kernel(name: "sp_gemm_q8_nat_m32n128s4", weight: q8, rows: 75, grid: MTLSize(width: 3, height: outDim / 128, depth: 1), threads: 128))
        list.append(Kernel(name: "sp_gemm_q8_split_tiled_m16n64p4", weight: q8, rows: 16, grid: splitGrid(64), threads: 128))
        list.append(Kernel(name: "sp_gemm_q8_split_tiled_m16n64p4", weight: q8, rows: 11, grid: splitGrid(64), threads: 128))
        list.append(Kernel(name: "sp_gemm_q8_split_tiled_m32p4", weight: q8, rows: 27, grid: splitGrid(32), threads: 128))
        // Group-K candidates.
        let layouts: [(String, TiledWeight)] = [
            ("u4k64", weight(64, 4, signed: false)), ("u8k64", q8),
            ("u4k32", weight(32, 4, signed: false)), ("i4k32", weight(32, 4, signed: true, affine: false)),
            ("u8k32", weight(32, 8, signed: false)), ("i8k32", weight(32, 8, signed: true, affine: false)),
            ("i8k16", weight(16, 8, signed: true, affine: false)),
        ]
        for (suffix, w) in layouts {
            list.append(Kernel(name: "sp_gk_wide_" + suffix, weight: w, rows: 128, grid: wideGrid, threads: 128, guardsRows: false))
            list.append(Kernel(name: "sp_gk_split_" + suffix, weight: w, rows: 16, grid: splitGrid(64), threads: 128))
            list.append(Kernel(name: "sp_gk_split_" + suffix, weight: w, rows: 11, grid: splitGrid(64), threads: 128))
        }
        // Staged candidates: 4-bit codes in groups of 32.
        let k4 = weight(32, 4, signed: false)
        for partitions in [2, 4, 8] {
            for rows in [16, 11] {
                list.append(Kernel(name: "sp_gs_split_m16p\(partitions)", weight: k4, rows: rows, grid: splitGrid(32),
                                   threads: partitions * 32, staged: true))
            }
        }
        list.append(Kernel(name: "sp_gs_wide", weight: k4, rows: 128, grid: MTLSize(width: outDim / 64, height: 1, depth: 1), threads: 128, staged: true))
        // Whole 32-row tiles only, as the shipped whole-tile kernel: the fourth simdgroup has no rows here.
        list.append(Kernel(name: "sp_gs_wide", weight: k4, rows: 96, grid: MTLSize(width: outDim / 64, height: 1, depth: 1), threads: 128, staged: true))
        return list
    }

    @Test("8-bit, group-K and staged weight kernels match a CPU reference")
    func matchReference() throws {
        let device = try #require(MTLCreateSystemDefaultDevice())
        let library = try Metallib(device: device)
        let queue = try #require(device.makeCommandQueue())
        let outDim = Self.outDim, inner = Self.inner, cap = Self.cap
        func buffer<T>(_ values: [T]) throws -> MTLBuffer {
            try #require(values.withUnsafeBytes { device.makeBuffer(bytes: $0.baseAddress!, length: $0.count, options: .storageModeShared) })
        }
        // Activations for every row the buffers hold, so a kernel that lets a row past the
        // step through is seen.
        var random = Generator(state: 0xAC71)
        let x: [UInt16] = (0..<(cap * inner)).map { _ in Q4BufferLayout.bf16Bits(random.unit()) }
        let xValues = x.map { Q4BufferLayout.bf16($0) }
        let xBuffer = try buffer(x)
        let sentinel = Float(bitPattern: 0xFFFF_FFFF)

        for kernel in Self.kernels {
            let w = kernel.weight, groups = w.groups, rows = kernel.rows
            var sums = [Float](repeating: 0, count: cap * groups)
            for row in 0..<cap {
                for g in 0..<groups {
                    var total: Float = 0
                    for k in 0..<w.group { total += xValues[row * inner + g * w.group + k] }
                    sums[row * groups + g] = total
                }
            }
            let packed = try buffer(w.packed), scales = try buffer(w.scales), offsets = try buffer(w.offsets)
            let sumsBuffer = try buffer(sums)
            let out = try buffer([Float](repeating: sentinel, count: cap * outDim))
            let command = try #require(queue.makeCommandBuffer())
            let encoder = try #require(command.makeComputeCommandEncoder(dispatchType: .concurrent))
            var p = KernelParams(rows: UInt32(rows), outDim: UInt32(outDim), inner: UInt32(inner), groups: UInt32(groups),
                                 hasResidual: 0, outStride: UInt32(outDim))
            encoder.setComputePipelineState(try library.pipeline(kernel.name))
            encoder.setBuffer(packed, offset: 0, index: 0); encoder.setBuffer(scales, offset: 0, index: 1); encoder.setBuffer(offsets, offset: 0, index: 2)
            encoder.setBuffer(xBuffer, offset: 0, index: 3)
            encoder.setBuffer(out, offset: 0, index: 4); encoder.setBuffer(out, offset: 0, index: 5)
            encoder.setBuffer(sumsBuffer, offset: 0, index: 6)
            encoder.setBytes(&p, length: MemoryLayout<KernelParams>.stride, index: 7)
            encoder.dispatchThreadgroups(kernel.grid, threadsPerThreadgroup: MTLSize(width: kernel.threads, height: 1, depth: 1))
            encoder.endEncoding()
            command.commit()
            command.waitUntilCompleted()
            try #require(command.status == .completed, "\(kernel.name): \(String(describing: command.error))")
            let result = Array(UnsafeBufferPointer(start: out.contents().bindMemory(to: Float.self, capacity: cap * outDim), count: cap * outDim))

            var worst: Float = 0, untouched = 0, written = 0
            for row in 0..<cap {
                if row >= rows {
                    for n in 0..<outDim where result[row * outDim + n].bitPattern != sentinel.bitPattern { written += 1 }
                    continue
                }
                for n in 0..<outDim {
                    var expected: Double = 0
                    for k in 0..<inner { expected += Double(xValues[row * inner + k]) * Double(w.weight(output: n, column: k, staged: kernel.staged)) }
                    let observed = result[row * outDim + n]
                    if observed.bitPattern == sentinel.bitPattern { untouched += 1; continue }
                    worst = max(worst, abs(observed - Float(expected)))
                }
            }
            // Outputs are sums of 1280 products of magnitude about 0.05; fp32 accumulation in
            // a different order leaves errors of a few 1e-5.
            #expect(untouched == 0, "\(kernel.name) at \(rows) rows left \(untouched) outputs unwritten")
            #expect(worst < 5e-4, "\(kernel.name) at \(rows) rows: largest error \(worst)")
            if kernel.guardsRows {
                #expect(written == 0, "\(kernel.name) at \(rows) rows wrote \(written) outputs of rows past the step")
            }
        }
    }
}
