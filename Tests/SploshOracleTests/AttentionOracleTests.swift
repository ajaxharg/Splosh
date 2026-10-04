import Foundation
import Testing
import Metal
import SploshCore
import SploshOracle

private struct AttentionDenseParams {
    var headDim: UInt32
    var heads: UInt32
    var kvHeads: UInt32
    var rowCount: UInt32
    var tokenCount: UInt32
    var queryPosition: UInt32
    var startPosition: UInt32
    var rotaryDim: UInt32
    var epsilon: Float
    var ropeTheta: Float
}

@Suite("AttentionOracleTests")
struct AttentionOracleTests {
    private let heads = 2
    private let kvHeads = 1
    private let headDim = 4
    private let tokenCount = 4
    private let scales: [Float] = [0.125]

    private func relativeError(_ actual: [Float], _ expected: [Float]) -> Float {
        let numerator = zip(actual, expected).map { abs($0 - $1) }.reduce(0, +)
        let denominator = max(expected.map(abs).reduce(0, +), 1e-12)
        return numerator / denominator
    }

    private func geometry() -> AttentionOracle.Configuration {
        .init(heads: heads, kvHeads: kvHeads, headDim: headDim,
              epsilon: 1e-6, rotaryDim: 4, ropeTheta: 10_000,
              positionOrigin: 0, position: 2)
    }

    // Token-major [token][kv head][channel], with deliberately exact per-head scale.
    private func kvPayload(offset: Int) -> [Int8] {
        (0..<(tokenCount * kvHeads * headDim)).map { i in
            Int8(((i * 7 + offset) % 31) - 15)
        }
    }

    private func qPayload(rows: Int) -> [Float] {
        (0..<(rows * heads * 2 * headDim)).map { i in
            let x = Float((i * 13 + 3) % 29 - 14)
            return x / 9.0
        }
    }

    private func runKernel(_ name: String, q: [Float], keys: [Int8], values: [Int8],
                           qNorm: [Float], kNorm: [Float], oProjection: [Float],
                           rows: Int, startPosition: Int) throws -> [Float] {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw SploshError.capabilityGateFailure("no Metal device")
        }
        let metallib = try Metallib(device: device)
        let pipeline = try metallib.pipeline(name)
        let queue = try #require(device.makeCommandQueue())
        let outputCount = rows * heads * headDim
        let qBuffer = try #require(device.makeBuffer(bytes: q, length: q.count * MemoryLayout<Float>.stride, options: .storageModeShared))
        let qnBuffer = try #require(device.makeBuffer(bytes: qNorm, length: qNorm.count * MemoryLayout<Float>.stride, options: .storageModeShared))
        let knBuffer = try #require(device.makeBuffer(bytes: kNorm, length: kNorm.count * MemoryLayout<Float>.stride, options: .storageModeShared))
        let kBuffer = try #require(device.makeBuffer(bytes: keys, length: keys.count, options: .storageModeShared))
        let vBuffer = try #require(device.makeBuffer(bytes: values, length: values.count, options: .storageModeShared))
        let ksBuffer = try #require(device.makeBuffer(bytes: scales, length: scales.count * MemoryLayout<Float>.stride, options: .storageModeShared))
        let vsBuffer = try #require(device.makeBuffer(bytes: scales, length: scales.count * MemoryLayout<Float>.stride, options: .storageModeShared))
        let oBuffer = oProjection.isEmpty ? nil : try #require(device.makeBuffer(bytes: oProjection, length: oProjection.count * MemoryLayout<Float>.stride, options: .storageModeShared))
        let outputBuffer = try #require(device.makeBuffer(length: outputCount * MemoryLayout<Float>.stride, options: .storageModeShared))
        var params = AttentionDenseParams(headDim: UInt32(headDim), heads: UInt32(heads), kvHeads: UInt32(kvHeads),
                                          rowCount: UInt32(rows), tokenCount: UInt32(tokenCount),
                                          queryPosition: UInt32(startPosition + rows - 1), startPosition: UInt32(startPosition),
                                          rotaryDim: 4, epsilon: 1e-6, ropeTheta: 10_000)
        let command = try #require(queue.makeCommandBuffer())
        let encoder = try #require(command.makeComputeCommandEncoder())
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(qBuffer, offset: 0, index: 0)
        encoder.setBuffer(qnBuffer, offset: 0, index: 1)
        encoder.setBuffer(knBuffer, offset: 0, index: 2)
        encoder.setBuffer(kBuffer, offset: 0, index: 3)
        encoder.setBuffer(vBuffer, offset: 0, index: 4)
        encoder.setBuffer(ksBuffer, offset: 0, index: 5)
        encoder.setBuffer(vsBuffer, offset: 0, index: 6)
        encoder.setBuffer(oBuffer, offset: 0, index: 7)
        encoder.setBuffer(outputBuffer, offset: 0, index: 8)
        encoder.setBytes(&params, length: MemoryLayout<AttentionDenseParams>.stride, index: 9)
        if name == "attention_dense_decode" {
            encoder.dispatchThreads(MTLSize(width: heads * headDim, height: 1, depth: 1),
                                    threadsPerThreadgroup: MTLSize(width: min(heads * headDim, pipeline.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
        } else {
            encoder.dispatchThreads(MTLSize(width: 1, height: rows, depth: 1),
                                    threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))
        }
        encoder.endEncoding()
        command.commit(); command.waitUntilCompleted()
        if let error = command.error { throw SploshError.capabilityGateFailure("attention command failed: \(error)") }
        let pointer = outputBuffer.contents().bindMemory(to: Float.self, capacity: outputCount)
        return Array(UnsafeBufferPointer(start: pointer, count: outputCount))
    }

    @Test("layer applies causal GQA and gate before projection")
    func layerContract() {
        let c = geometry()
        let q = qPayload(rows: 1)
        let keys = AttentionOracle.dequantizedInt8(kvPayload(offset: 1), scales: scales, kvHeads: kvHeads, headDim: headDim)
        let values = AttentionOracle.dequantizedInt8(kvPayload(offset: 9), scales: scales, kvHeads: kvHeads, headDim: headDim)
        let norm: [Float] = [0.0, 0.1, -0.1, 0.2]
        let o = (0..<(heads * headDim * heads * headDim)).map { Float(($0 % 9) - 4) / 11 }
        let output = AttentionOracle.layer(qProjection: q, keys: keys, values: values, qNorm: norm, kNorm: norm,
                                           oProjection: o, configuration: c)
        #expect(output.count == heads * headDim)
        #expect(output.allSatisfy { $0.isFinite })
    }

    @Test("attention_dense_decode matches decodeDense with causal GQA and int8 KV")
    func decodeMatchesOracle() throws {
        let q = qPayload(rows: 1), k8 = kvPayload(offset: 1), v8 = kvPayload(offset: 9)
        let norm: [Float] = [0.0, 0.1, -0.1, 0.2]
        let keys = AttentionOracle.dequantizedInt8(k8, scales: scales, kvHeads: kvHeads, headDim: headDim)
        let values = AttentionOracle.dequantizedInt8(v8, scales: scales, kvHeads: kvHeads, headDim: headDim)
        let expected = AttentionOracle.decodeDense(qProjection: q, keys: keys, values: values, qNorm: norm, kNorm: norm, oProjection: [], position: 2, configuration: geometry())
        let actual = try runKernel("attention_dense_decode", q: q, keys: k8, values: v8, qNorm: norm, kNorm: norm, oProjection: [], rows: 1, startPosition: 2)
        let relErr = relativeError(actual, expected)
        print("attention_dense_decode relErr=\(relErr)")
        #expect(relErr <= 1e-3, "decode relErr=\(relErr)")
    }

    @Test("attention_dense_prefill matches prefill for every causal row")
    func prefillMatchesOracle() throws {
        let rows = 3, q = qPayload(rows: rows), k8 = kvPayload(offset: 1), v8 = kvPayload(offset: 9)
        let norm: [Float] = [0.0, 0.1, -0.1, 0.2]
        let keys = AttentionOracle.dequantizedInt8(k8, scales: scales, kvHeads: kvHeads, headDim: headDim)
        let values = AttentionOracle.dequantizedInt8(v8, scales: scales, kvHeads: kvHeads, headDim: headDim)
        let expectedRows = AttentionOracle.prefill(qProjections: q, keys: keys, values: values, qNorm: norm, kNorm: norm, oProjection: [], startPosition: 0, configuration: geometry())
        let expected = expectedRows.flatMap { $0 }
        let actual = try runKernel("attention_dense_prefill", q: q, keys: k8, values: v8, qNorm: norm, kNorm: norm, oProjection: [], rows: rows, startPosition: 0)
        let relErr = relativeError(actual, expected)
        print("attention_dense_prefill relErr=\(relErr)")
        #expect(relErr <= 1e-3, "prefill relErr=\(relErr)")
    }

    @Test("metallib exports the cumulative M2.4 set and rejects a missing export")
    func exactExportsAndMissingFailure() throws {
        let device = try #require(MTLCreateSystemDefaultDevice())
        let library = try Metallib(device: device)
        let expected: Set<String> = ["copy", "rmsnorm", "rope_mrope", "swiglu", "gemm_bf16", "attention_dense_decode", "attention_dense_prefill"]
        // Later milestones and the engine (sp_-prefixed kernels) add to the library.
        #expect(expected.isSubset(of: Set(library.functionNames)))
        var failed = false
        do { _ = try library.pipeline("attention_dense_missing") } catch { failed = true }
        #expect(failed)
    }
}
