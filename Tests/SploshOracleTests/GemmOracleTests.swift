import Testing
import Metal

import SploshCore
import SploshModel
import SploshOracle
import SploshQuant

private struct GemmBF16Params {
    var rows: UInt32
    var columns: UInt32
    var inner: UInt32
}

private struct GemmQ4Params {
    var rows: UInt32
    var columns: UInt32
    var inner: UInt32
    var q4RowStrideWords: UInt32
    var orientation: UInt32
    var groupsPerRow: UInt32
}

/// Isolated M2.3d gate: BF16 GEMM only, using the injected five-export metallib.
@Suite("M23dGemmTests")
struct M23dGemmTests {
    private let a: [UInt16] = [0x3f80, 0xc000, 0x4040, 0x3e80, 0x4120, 0xbf00]
    private let b: [UInt16] = [0x4000, 0xbf80, 0x3f00, 0x4080, 0xc040, 0x3fc0]

    @Test("isolated gemm_bf16 matches GemmOracle.bf16 with relErr <= 1e-2")
    func shaderMatchesOracle() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw SploshError.capabilityGateFailure("no Metal device on this host")
        }
        let metallib = try makeTestMetallib(device: device)
        let pipeline = try metallib.pipeline("gemm_bf16")
        let expected = GemmOracle.bf16(a: a, b: b, rows: 2, columns: 2, inner: 3)
        let aBuffer = try #require(device.makeBuffer(bytes: a, length: a.count * 2, options: .storageModeShared))
        let bBuffer = try #require(device.makeBuffer(bytes: b, length: b.count * 2, options: .storageModeShared))
        let cBuffer = try #require(device.makeBuffer(length: expected.count * 4, options: .storageModeShared))
        let queue = try #require(device.makeCommandQueue())
        let command = try #require(queue.makeCommandBuffer())
        let encoder = try #require(command.makeComputeCommandEncoder())
        var params = GemmBF16Params(rows: 2, columns: 2, inner: 3)
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(aBuffer, offset: 0, index: 0)
        encoder.setBuffer(bBuffer, offset: 0, index: 1)
        encoder.setBuffer(cBuffer, offset: 0, index: 2)
        encoder.setBytes(&params, length: MemoryLayout<GemmBF16Params>.stride, index: 3)
        encoder.dispatchThreads(MTLSize(width: 2, height: 2, depth: 1), threadsPerThreadgroup: MTLSize(width: 2, height: 2, depth: 1))
        encoder.endEncoding()
        command.commit(); command.waitUntilCompleted()
        if let error = command.error { throw SploshError.capabilityGateFailure("gemm command failed: \(error)") }
        let actual = Array(UnsafeBufferPointer(start: cBuffer.contents().bindMemory(to: Float.self, capacity: expected.count), count: expected.count))
        let numerator = zip(actual, expected).map { abs($0 - $1) }.reduce(0, +)
        let denominator = max(expected.map(abs).reduce(0, +), 1e-6)
        let relErr = numerator / denominator
        print("GemmOracle.bf16 relErr=\(relErr) tolerance=1e-2")
        #expect(relErr <= 1e-2, "relErr=\(relErr) exceeds bf16 tolerance 1e-2")
    }

    @Test("isolated metallib exports exactly the M2.3d five-kernel set")
    func exactExportSet() throws {
        let device = try #require(MTLCreateSystemDefaultDevice())
        let metallib = try makeTestMetallib(device: device)
        #expect(Set(metallib.functionNames) == ["copy", "rmsnorm", "rope_mrope", "swiglu", "gemm_bf16"])
    }
}

@Suite("GemmRegressionTests")
struct GemmOracleTests {
    // Independent bf16 payloads, intentionally non-trivial signs and magnitudes.
    private let a: [UInt16] = [0x3f80, 0xc000, 0x4040, 0x3e80, 0x4120, 0xbf00]
    private let b: [UInt16] = [0x4000, 0xbf80, 0x3f00, 0x4080, 0xc040, 0x3fc0]

    @Test("GemmOracle.bf16 uses row-major bf16 operands and fp32 accumulation")
    func independentVector() {
        let result = GemmOracle.bf16(a: a, b: b, rows: 2, columns: 2, inner: 3)
        #expect(result.count == 4)
        #expect(result[0] == -8.0)
        #expect(result[1] == -4.5)
        #expect(result[2] == 7.0)
        #expect(result[3] == 39.0)
    }

    @Test("gemm_bf16 matches GemmOracle.bf16 with fp32 accumulation and relErr <= 1e-2")
    func shaderMatchesOracle() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw SploshError.capabilityGateFailure("no Metal device on this host")
        }
        let metallib = try Metallib(device: device)
        let pipeline = try metallib.pipeline("gemm_bf16")
        let expected = GemmOracle.bf16(a: a, b: b, rows: 2, columns: 2, inner: 3)
        let aBuffer = try #require(device.makeBuffer(bytes: a, length: a.count * 2, options: .storageModeShared))
        let bBuffer = try #require(device.makeBuffer(bytes: b, length: b.count * 2, options: .storageModeShared))
        let cBuffer = try #require(device.makeBuffer(length: expected.count * 4, options: .storageModeShared))
        let queue = try #require(device.makeCommandQueue())
        let command = try #require(queue.makeCommandBuffer())
        let encoder = try #require(command.makeComputeCommandEncoder())
        var params = GemmBF16Params(rows: 2, columns: 2, inner: 3)
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(aBuffer, offset: 0, index: 0)
        encoder.setBuffer(bBuffer, offset: 0, index: 1)
        encoder.setBuffer(cBuffer, offset: 0, index: 2)
        encoder.setBytes(&params, length: MemoryLayout<GemmBF16Params>.stride, index: 3)
        encoder.dispatchThreads(MTLSize(width: 2, height: 2, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: 2, height: 2, depth: 1))
        encoder.endEncoding()
        command.commit(); command.waitUntilCompleted()
        if let error = command.error { throw SploshError.capabilityGateFailure("gemm command failed: \(error)") }
        let pointer = cBuffer.contents().bindMemory(to: Float.self, capacity: expected.count)
        let actual = Array(UnsafeBufferPointer(start: pointer, count: expected.count))
        let numerator = zip(actual, expected).map { abs($0 - $1) }.reduce(0, +)
        let denominator = max(expected.map(abs).reduce(0, +), 1e-6)
        let relErr = numerator / denominator
        print("GemmOracle.bf16 relErr=\(relErr) tolerance=1e-2")
        #expect(relErr <= 1e-2, "relErr=\(relErr) exceeds bf16 tolerance 1e-2")
    }

    @Test("gemm_q4 matches explicit affine dequantization then fp32 GEMM")
    func q4ShaderMatchesDequantThenGemm() throws {
        guard let device = MTLCreateSystemDefaultDevice() else { throw SploshError.capabilityGateFailure("no Metal device on this host") }
        let metallib = try Metallib(device: device)
        let pipeline = try metallib.pipeline("gemm_q4")
        let rows = 2, columns = 3, inner = 65
        let layout = try Q4BufferLayout(rows: columns, logicalK: inner)
        var words = Array(repeating: UInt32.zero, count: layout.packedWordCount)
        for n in 0..<columns { for k in 0..<inner { words[try layout.wordIndex(row: n, k: k)] |= UInt32((n + 3 * k) & 15) << (try layout.nibbleShift(row: n, k: k)) } }
        let scales = (0..<layout.sidecarCount).map { Q4BufferLayout.bf16Bits(Float(0.25 + Float($0) * 0.125)) }
        let biases = (0..<layout.sidecarCount).map { Q4BufferLayout.bf16Bits(Float(-0.5 + Float($0) * 0.25)) }
        let compactWords = (0..<columns).flatMap { n in
            (0..<layout.packedWordsPerRow).map { words[n * layout.rowStrideWords + $0] }
        }
        let tensor = try MlxAffine(shape: [columns, inner], words: compactWords, scales: scales.map(Q4BufferLayout.bf16), biases: biases.map(Q4BufferLayout.bf16))
        let dequant = DequantOracle.dequantize(tensor)
        let a: [UInt16] = (0..<(rows * inner)).map { Q4BufferLayout.bf16Bits(Float((($0 % 11) - 5)) * 0.125) }
        // DequantOracle supplies the affine Float32 weights; retain fp32 accumulation
        // while using the bf16 left operand, matching the two-step q4 oracle contract.
        var expected = Array(repeating: Float.zero, count: rows * columns)
        for row in 0..<rows {
            for column in 0..<columns {
                var sum: Float = 0
                for k in 0..<inner {
                    let lhs = Q4BufferLayout.bf16(a[row * inner + k])
                    sum += lhs * dequant[column * inner + k]
                }
                expected[row * columns + column] = sum
            }
        }
        let ab = try #require(device.makeBuffer(bytes: a, length: a.count * 2, options: .storageModeShared))
        let qb = try #require(device.makeBuffer(bytes: words, length: words.count * 4, options: .storageModeShared))
        let sb = try #require(device.makeBuffer(bytes: scales, length: scales.count * 2, options: .storageModeShared))
        let bb = try #require(device.makeBuffer(bytes: biases, length: biases.count * 2, options: .storageModeShared))
        let cb = try #require(device.makeBuffer(length: expected.count * 4, options: .storageModeShared))
        let queue = try #require(device.makeCommandQueue()); let command = try #require(queue.makeCommandBuffer()); let encoder = try #require(command.makeComputeCommandEncoder())
        var params = GemmQ4Params(rows: UInt32(rows), columns: UInt32(columns), inner: UInt32(inner), q4RowStrideWords: UInt32(layout.rowStrideWords), orientation: 0, groupsPerRow: UInt32(layout.groupsPerRow))
        encoder.setComputePipelineState(pipeline); encoder.setBuffer(ab, offset: 0, index: 0); encoder.setBuffer(qb, offset: 0, index: 1); encoder.setBuffer(sb, offset: 0, index: 2); encoder.setBuffer(bb, offset: 0, index: 3); encoder.setBuffer(cb, offset: 0, index: 4); encoder.setBytes(&params, length: MemoryLayout<GemmQ4Params>.stride, index: 5)
        encoder.dispatchThreads(MTLSize(width: columns, height: rows, depth: 1), threadsPerThreadgroup: MTLSize(width: columns, height: rows, depth: 1)); encoder.endEncoding(); command.commit(); command.waitUntilCompleted()
        if let error = command.error { throw SploshError.capabilityGateFailure("gemm_q4 command failed: \(error)") }
        let actual = Array(UnsafeBufferPointer(start: cb.contents().bindMemory(to: Float.self, capacity: expected.count), count: expected.count))
        let numerator = zip(actual, expected).map { abs($0 - $1) }.reduce(0, +); let denominator = max(expected.map(abs).reduce(0, +), 1e-6); let relErr = numerator / denominator
        print("GemmOracle.q4 relErr=\(relErr) tolerance=1e-2"); #expect(relErr <= 1e-2)
    }

    @Test("named q4 GEMM cases use explicit dequantize-then-GEMM with relErr <= 1e-2")
    func namedQ4Cases() throws {
        guard let device = MTLCreateSystemDefaultDevice() else { throw SploshError.capabilityGateFailure("no Metal device on this host") }
        let metallib = try Metallib(device: device)
        let pipeline = try metallib.pipeline("gemm_q4")
        let cases: [(name: String, rows: Int, columns: Int, inner: Int)] = [
            ("q4_gemm_small", 2, 3, 65),
            ("q4_gemm_group_boundary", 3, 2, 128),
            ("q4_gemm_single_row", 1, 1, 64)
        ]
        for testCase in cases {
            let layout = try Q4BufferLayout(rows: testCase.columns, logicalK: testCase.inner)
            var words = Array(repeating: UInt32.zero, count: layout.packedWordCount)
            for column in 0..<testCase.columns {
                for k in 0..<testCase.inner {
                    words[try layout.wordIndex(row: column, k: k)] |= UInt32((column + k * 3) & 15) << (try layout.nibbleShift(row: column, k: k))
                }
            }
            let scales = (0..<layout.sidecarCount).map { Q4BufferLayout.bf16Bits(0.25 + Float($0) * 0.125) }
            let biases = (0..<layout.sidecarCount).map { Q4BufferLayout.bf16Bits(-0.5 + Float($0) * 0.25) }
            let compact = (0..<testCase.columns).flatMap { row in (0..<layout.packedWordsPerRow).map { words[row * layout.rowStrideWords + $0] } }
            let tensor = try MlxAffine(shape: [testCase.columns, testCase.inner], words: compact, scales: scales.map(Q4BufferLayout.bf16), biases: biases.map(Q4BufferLayout.bf16))
            let dequant = DequantOracle.dequantize(tensor)
            let a = (0..<(testCase.rows * testCase.inner)).map { Q4BufferLayout.bf16Bits(Float(($0 % 11) - 5) * 0.125) }
            // DequantOracle returns Q in [N,K] row-major order, while GemmOracle
            // consumes B in [K,N] row-major order for A[M,K] × B[K,N].
            let b = (0..<testCase.inner).flatMap { k in
                (0..<testCase.columns).map { column in
                    Q4BufferLayout.bf16Bits(dequant[column * testCase.inner + k])
                }
            }
            let expected = GemmOracle.bf16(a: a, b: b, rows: testCase.rows, columns: testCase.columns, inner: testCase.inner)
            let ab = try #require(device.makeBuffer(bytes: a, length: a.count * 2, options: .storageModeShared))
            let qb = try #require(device.makeBuffer(bytes: words, length: words.count * 4, options: .storageModeShared))
            let sb = try #require(device.makeBuffer(bytes: scales, length: scales.count * 2, options: .storageModeShared))
            let bb = try #require(device.makeBuffer(bytes: biases, length: biases.count * 2, options: .storageModeShared))
            let cb = try #require(device.makeBuffer(length: expected.count * 4, options: .storageModeShared))
            let queue = try #require(device.makeCommandQueue()); let command = try #require(queue.makeCommandBuffer()); let encoder = try #require(command.makeComputeCommandEncoder())
            var params = GemmQ4Params(rows: UInt32(testCase.rows), columns: UInt32(testCase.columns), inner: UInt32(testCase.inner), q4RowStrideWords: UInt32(layout.rowStrideWords), orientation: 0, groupsPerRow: UInt32(layout.groupsPerRow))
            encoder.setComputePipelineState(pipeline); encoder.setBuffer(ab, offset: 0, index: 0); encoder.setBuffer(qb, offset: 0, index: 1); encoder.setBuffer(sb, offset: 0, index: 2); encoder.setBuffer(bb, offset: 0, index: 3); encoder.setBuffer(cb, offset: 0, index: 4); encoder.setBytes(&params, length: MemoryLayout<GemmQ4Params>.stride, index: 5)
            encoder.dispatchThreads(MTLSize(width: testCase.columns, height: testCase.rows, depth: 1), threadsPerThreadgroup: MTLSize(width: testCase.columns, height: testCase.rows, depth: 1)); encoder.endEncoding(); command.commit(); command.waitUntilCompleted()
            if let error = command.error { throw SploshError.capabilityGateFailure("gemm_q4 command failed: \(error)") }
            let actual = Array(UnsafeBufferPointer(start: cb.contents().bindMemory(to: Float.self, capacity: expected.count), count: expected.count))
            let numerator = zip(actual, expected).map { abs($0 - $1) }.reduce(0, +)
            let denominator = max(expected.map(abs).reduce(0, +), 1e-6)
            let relErr = numerator / denominator
            print("\(testCase.name) relErr=\(relErr) tolerance=1e-2")
            #expect(relErr <= 1e-2)
        }
    }

    @Test("q4 GEMM rejects malformed layouts before dispatch")
    func malformedLayoutFailsClosed() throws {
        let shape = try GemmQ4Shape(rows: 2, columns: 3, inner: 65)
        #expect(throws: GemmQ4Error.layoutMismatch) {
            try shape.validate(layout: Q4BufferLayout(rows: 4, logicalK: 65))
        }
        #expect(throws: GemmQ4Error.layoutMismatch) {
            try shape.validate(layout: Q4BufferLayout(rows: 3, logicalK: 64))
        }
    }

    @Test("metallib exports the oracle kernel set plus only sp_-prefixed engine kernels")
    func exactExportSet() throws {
        let device = try #require(MTLCreateSystemDefaultDevice())
        let metallib = try Metallib(device: device)
        let oracle: Set<String> = ["copy", "rmsnorm", "rope_mrope", "swiglu", "gemm_bf16", "attention_dense_decode", "attention_dense_prefill", "gdn_prepare", "gdn_decode", "gdn_gate", "gdn_commit", "gemm_q4"]
        let names = Set(metallib.functionNames)
        #expect(oracle.isSubset(of: names))
        #expect(names.subtracting(oracle).allSatisfy { $0.hasPrefix("sp_") })
    }
}
