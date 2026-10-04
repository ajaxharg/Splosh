import Testing
import Metal
import Foundation

import SploshCore
import SploshModel
import SploshQuant

// Timing probe for the candidate weight kernels in Sources/Shaders/candidates/gemm_group.metal
// and gguf_staged.metal, against the shipped q4 kernels: what a quant group narrower than 64
// columns costs on the accelerator, what a byte per code costs instead of a nibble, and what it
// costs to dequantise a block into threadgroup memory before each matmul. With them, the GGUF
// kernels of Sources/Shaders/engine_gguf.metal for five of their formats, which are the staged
// candidates with a real format's decoding. One MLP-sized GEMM (17408 x 5120) over several
// distinct weight tensors per command buffer, so the weights stream from memory as they do in
// a step.
//
// The values are random and the outputs are not checked here: only GPU time is read. Variants
// run interleaved, several rounds, and the report gives the least and the median of each. Off
// unless SPLOSH_GK_PROBE is set, since it measures and asserts nothing;
// SPLOSH_GK_PROBE_ROUNDS, _TENSORS and _PASSES change the defaults.

private struct GkParams {
    var rows, outDim, inner, groups, hasResidual, outStride: UInt32
}

@Suite("GroupGemmProbeTests")
struct GroupGemmProbeTests {
    private struct Variant {
        let label: String
        let kernel: String
        /// Columns per quant group, as the kernel's parameters count them.
        let group: Int
        let bits: Int
        let rows: Int
        let grid: MTLSize
        let threads: Int
        /// The format of a GGUF kernel, which reads planes and not codes with sidecars.
        var gguf: GgufTensorType? = nil
    }

    /// Plane 1 and the meta plane of one GGUF format, each with its bytes a tensor.
    private struct GgufPlaneSet {
        let plane1: MTLBuffer
        let plane1Bytes: Int
        let meta: MTLBuffer
        let metaBytes: Int
    }

    @Test("group size, code width and staged dequantisation on the accelerator, timed",
          .enabled(if: ProcessInfo.processInfo.environment["SPLOSH_GK_PROBE"] != nil))
    func timings() throws {
        let environment = ProcessInfo.processInfo.environment
        let rounds = Int(environment["SPLOSH_GK_PROBE_ROUNDS"] ?? "") ?? 9
        let tensors = Int(environment["SPLOSH_GK_PROBE_TENSORS"] ?? "") ?? 4
        // Passes over the tensors in one command buffer, so its fixed cost is spread thin. The
        // tensors together are far larger than any cache, so a second pass streams them again.
        let passes = Int(environment["SPLOSH_GK_PROBE_PASSES"] ?? "") ?? 8
        let outDim = 17408, inner = 5120, wideRows = 128, narrowRows = 16
        let device = try #require(MTLCreateSystemDefaultDevice())
        let library = try Metallib(device: device)
        let queue = try #require(device.makeCommandQueue())

        // One buffer of codes at the 8-bit size, and one of sidecars at the finest group (16
        // columns), shared by every variant: each reads as much of them as its layout needs.
        let codeBytes = outDim * inner, sidecarBytes = outDim * (inner / 16) * 2
        let packed = try #require(device.makeBuffer(length: tensors * codeBytes, options: .storageModeShared))
        let sidecars = try #require(device.makeBuffer(length: tensors * sidecarBytes, options: .storageModeShared))
        var state: UInt64 = 0x5EED_0008
        func next() -> UInt64 {
            state &+= 0x9E37_79B9_7F4A_7C15
            var z = state
            z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
            z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
            return z ^ (z >> 31)
        }
        let words = packed.contents().bindMemory(to: UInt64.self, capacity: packed.length / 8)
        for index in 0..<(packed.length / 8) { words[index] = next() }
        let scale = Q4BufferLayout.bf16Bits(0.0125)
        let sidecar = sidecars.contents().bindMemory(to: UInt16.self, capacity: sidecars.length / 2)
        for index in 0..<(sidecars.length / 2) { sidecar[index] = scale }
        let activations = try #require(device.makeBuffer(length: wideRows * inner * 2, options: .storageModeShared))
        let a = activations.contents().bindMemory(to: UInt16.self, capacity: wideRows * inner)
        for index in 0..<(wideRows * inner) {
            a[index] = Q4BufferLayout.bf16Bits(Float(Int64(next() >> 40) - (1 << 23)) / Float(1 << 23))
        }
        let sums = try #require(device.makeBuffer(length: wideRows * (inner / 16) * 4, options: .storageModeShared))
        memset(sums.contents(), 0, sums.length)
        let out = try #require(device.makeBuffer(length: wideRows * outDim * 4, options: .storageModeShared))

        // The shipped tile shapes: 32 x 128 on four simdgroups, four row tiles across; and the
        // 16 x 64 split-K tile.
        let wideGrid = MTLSize(width: 4, height: outDim / 128, depth: 1)
        let splitGrid = MTLSize(width: outDim / 64, height: 1, depth: 1)
        // The GGUF formats timed, each with the fp16 scale fields of its meta record: where they
        // are and a value that makes weights of about 0.05.
        let ggufFormats: [(type: GgufTensorType, scales: [(offset: Int, value: Float)])] = [
            (.q4K, [(0, 2e-4), (2, 1.5e-3)]), (.q5K, [(0, 1e-4), (2, 1.5e-3)]), (.q6K, [(16, 5e-5)]),
            (.iq4XS, [(0, 6e-5)]), (.q8_0, [(0, 8e-4)]),
        ]
        var variants: [Variant] = []
        for wide in [true, false] {
            let shape = wide ? "wide" : "split", rows = wide ? wideRows : narrowRows, grid = wide ? wideGrid : splitGrid
            variants.append(Variant(label: "q4 shipped", kernel: wide ? "sp_gemm_q4_nat_m32n128s4_whole" : "sp_gemm_q4_split_tiled_m16n64p4",
                                    group: 64, bits: 4, rows: rows, grid: grid, threads: 128))
            variants.append(Variant(label: "q8 shipped", kernel: wide ? "sp_gemm_q8_nat_m32n128s4_whole" : "sp_gemm_q8_split_tiled_m16n64p4",
                                    group: 64, bits: 8, rows: rows, grid: grid, threads: 128))
            for (suffix, group, bits) in [("u4k64", 64, 4), ("u8k64", 64, 8), ("u4k32", 32, 4), ("i4k32", 32, 4),
                                          ("u8k32", 32, 8), ("i8k32", 32, 8), ("i8k16", 16, 8)] {
                variants.append(Variant(label: suffix, kernel: "sp_gk_\(shape)_\(suffix)", group: group, bits: bits,
                                        rows: rows, grid: grid, threads: 128))
            }
            // Staged: 4-bit codes in groups of 32, dequantised to fp16 in threadgroup memory.
            if wide {
                variants.append(Variant(label: "staged", kernel: "sp_gs_wide", group: 32, bits: 4, rows: rows,
                                        grid: MTLSize(width: outDim / 64, height: 1, depth: 1), threads: 128))
            } else {
                for partitions in [2, 4, 8] {
                    variants.append(Variant(label: "staged p\(partitions)", kernel: "sp_gs_split_m16p\(partitions)", group: 32, bits: 4,
                                            rows: rows, grid: MTLSize(width: outDim / 32, height: 1, depth: 1), threads: partitions * 32))
                }
            }
            // GGUF: codes at their native width in groups of 32, dequantised as the staged ones
            // are. The wide shape is the staged one; the split is its two-partition form.
            for format in ggufFormats {
                let geometry = try #require(GgufPlanes.geometry(of: format.type))
                variants.append(Variant(label: "gguf \(geometry.kernelSuffix)",
                                        kernel: (wide ? "sp_gguf_wide_" : "sp_gguf_split_m16_") + geometry.kernelSuffix,
                                        group: 32, bits: (geometry.plane0Bytes + geometry.plane1Bytes) / 4, rows: rows,
                                        grid: MTLSize(width: outDim / (wide ? 64 : 32), height: 1, depth: 1),
                                        threads: wide ? 128 : 64, gguf: format.type))
            }
        }
        let only = environment["SPLOSH_GK_PROBE_ONLY"].map { $0.split(separator: ",").map(String.init) }
        if let only { variants = variants.filter { variant in only.contains { variant.label.contains($0) } } }
        let pipelines = try variants.map { try library.pipeline($0.kernel) }

        // The planes of the GGUF formats left to time. Plane 0 is the shared codes, of which a
        // format reads its own width. Plane 1 and the meta plane are copies of some of them, so
        // random too, with the scale fields set: random bits there would be infinities and NaNs.
        let highBytes = outDim * (inner / 32) * 8
        var highBits: MTLBuffer?
        var planeSets: [GgufTensorType: GgufPlaneSet] = [:]
        for format in ggufFormats where variants.contains(where: { $0.gguf == format.type }) {
            let geometry = try #require(GgufPlanes.geometry(of: format.type))
            let sizes = try #require(GgufPlanes.sizes(of: format.type, rows: outDim, inner: inner))
            if highBits == nil {
                let buffer = try #require(device.makeBuffer(length: tensors * highBytes, options: .storageModeShared))
                memcpy(buffer.contents(), packed.contents(), buffer.length)
                highBits = buffer
            }
            let meta = try #require(device.makeBuffer(length: tensors * sizes.meta, options: .storageModeShared))
            memcpy(meta.contents(), packed.contents(), meta.length)
            for record in 0..<(meta.length / geometry.metaBytes) {
                for field in format.scales {
                    meta.contents().storeBytes(of: Float16(field.value).bitPattern,
                                               toByteOffset: record * geometry.metaBytes + field.offset, as: UInt16.self)
                }
            }
            planeSets[format.type] = GgufPlaneSet(plane1: try #require(highBits), plane1Bytes: highBytes, meta: meta, metaBytes: sizes.meta)
        }

        func run(_ index: Int) throws -> Double {
            let variant = variants[index]
            let command = try #require(queue.makeCommandBuffer())
            let encoder = try #require(command.makeComputeCommandEncoder(dispatchType: .concurrent))
            encoder.setComputePipelineState(pipelines[index])
            var p = GkParams(rows: UInt32(variant.rows), outDim: UInt32(outDim), inner: UInt32(inner),
                             groups: UInt32(inner / variant.group), hasResidual: 0, outStride: UInt32(outDim))
            encoder.setBuffer(activations, offset: 0, index: 3)
            encoder.setBuffer(out, offset: 0, index: 4)
            encoder.setBuffer(out, offset: 0, index: 5)
            encoder.setBuffer(sums, offset: 0, index: 6)
            encoder.setBytes(&p, length: MemoryLayout<GkParams>.stride, index: 7)
            for dispatch in 0..<(tensors * passes) {
                let tensor = dispatch % tensors
                encoder.setBuffer(packed, offset: tensor * codeBytes, index: 0)
                if let type = variant.gguf, let planes = planeSets[type] {
                    encoder.setBuffer(planes.plane1, offset: tensor * planes.plane1Bytes, index: 1)
                    encoder.setBuffer(planes.meta, offset: tensor * planes.metaBytes, index: 2)
                } else {
                    encoder.setBuffer(sidecars, offset: tensor * sidecarBytes, index: 1)
                    encoder.setBuffer(sidecars, offset: tensor * sidecarBytes, index: 2)
                }
                encoder.dispatchThreadgroups(variant.grid, threadsPerThreadgroup: MTLSize(width: variant.threads, height: 1, depth: 1))
            }
            encoder.endEncoding()
            command.commit()
            command.waitUntilCompleted()
            try #require(command.status == .completed, "\(variant.kernel): \(String(describing: command.error))")
            return (command.gpuEndTime - command.gpuStartTime) * 1000 / Double(tensors * passes)
        }

        var samples = [[Double]](repeating: [], count: variants.count)
        for index in variants.indices { _ = try run(index) }      // pipelines compiled, pages touched
        for round in 0..<rounds {
            for step in variants.indices {
                let index = (step + round) % variants.count
                samples[index].append(try run(index))
            }
        }
        print("weight kernel probe: \(outDim) x \(inner), \(passes) passes over \(tensors) tensors a command buffer, \(rounds) rounds; ms per GEMM (least / median)")
        for (index, variant) in variants.enumerated() {
            let sorted = samples[index].sorted()
            let megabytes = Double(outDim * inner * variant.bits / 8) / 1e6
            print(String(format: "  %3d rows  %-11@ K=%2d %d-bit  %7.3f / %7.3f ms   %6.1f GB/s of codes",
                         variant.rows, variant.label as NSString, variant.group, variant.bits,
                         sorted[0], sorted[sorted.count / 2], megabytes / sorted[0]))
        }
    }
}
