import Testing
import Metal
import SploshCore
import SploshModel
import SploshQuant
import SploshRuntime

@Suite("Q4RuntimeTests")
struct Q4RuntimeTests {
    private func device() throws -> MTLDevice {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw SploshError.capabilityGateFailure("no Metal device on this host")
        }
        return device
    }

    @Test("runtime validates ABI and completes synthetic q4 dispatch")
    func validDispatch() throws {
        let device = try device()
        let runtime = try Q4GemmRuntime(device: device, metallib: try Metallib(device: device))
        let shape = try GemmQ4Shape(rows: 2, columns: 3, inner: 65)
        let layout = try Q4BufferLayout(rows: 3, logicalK: 65)
        var words = Array(repeating: UInt32.zero, count: layout.packedWordCount)
        for n in 0..<3 { for k in 0..<65 { words[try layout.wordIndex(row: n, k: k)] |= UInt32((n + k) & 15) << (try layout.nibbleShift(row: n, k: k)) } }
        let scales = Array(repeating: Q4BufferLayout.bf16Bits(0.25), count: layout.sidecarCount)
        let biases = Array(repeating: Q4BufferLayout.bf16Bits(-0.5), count: layout.sidecarCount)
        let a = Array(repeating: Q4BufferLayout.bf16Bits(0.125), count: 2 * 65)
        let ab = try #require(device.makeBuffer(bytes: a, length: a.count * 2, options: .storageModeShared))
        let qb = try #require(device.makeBuffer(bytes: words, length: words.count * 4, options: .storageModeShared))
        let sb = try #require(device.makeBuffer(bytes: scales, length: scales.count * 2, options: .storageModeShared))
        let bb = try #require(device.makeBuffer(bytes: biases, length: biases.count * 2, options: .storageModeShared))
        let cb = try #require(device.makeBuffer(length: 2 * 3 * 4, options: .storageModeShared))
        let queue = try #require(device.makeCommandQueue())
        try runtime.dispatch(shape: shape, layout: layout, a: ab, packed: qb, scales: sb, biases: bb, output: cb, commandQueue: queue)
        // dispatch waits for and checks the submitted command buffer before returning.
        let actual = Array(UnsafeBufferPointer(start: cb.contents().bindMemory(to: Float.self, capacity: 6), count: 6))
        #expect(actual.allSatisfy { $0.isFinite })
    }

    @Test("runtime rejects orientation, stride, sidecar geometry, and undersized buffers")
    func rejectsInvalidABI() throws {
        let shape = try GemmQ4Shape(rows: 2, columns: 3, inner: 65)
        #expect(throws: GemmQ4Error.orientationMismatch) { try shape.validate(layout: Q4BufferLayout(rows: 3, logicalK: 65, orientation: .kByColumns)) }
        #expect(throws: GemmQ4Error.invalidRowStride) { try shape.validate(layout: Q4BufferLayout(rows: 3, logicalK: 65, rowStrideBytes: 160)) }
        #expect(throws: GemmQ4Error.layoutMismatch) { try shape.validate(layout: Q4BufferLayout(rows: 4, logicalK: 65)) }
        #expect(throws: GemmQ4Error.bufferTooSmall(name: "a", expected: 260, observed: 2)) {
            let layout = try Q4BufferLayout(rows: 3, logicalK: 65)
            let device = try device()
            let runtime = try Q4GemmRuntime(device: device, metallib: try Metallib(device: device))
            let tiny = try #require(device.makeBuffer(length: 2, options: .storageModeShared))
            let full = try #require(device.makeBuffer(length: layout.packedWordCount * 4, options: .storageModeShared))
            let side = try #require(device.makeBuffer(length: layout.sidecarCount * 2, options: .storageModeShared))
            let out = try #require(device.makeBuffer(length: 24, options: .storageModeShared))
            try runtime.dispatch(shape: shape, layout: layout, a: tiny, packed: full, scales: side, biases: side, output: out, commandQueue: try #require(device.makeCommandQueue()))
        }
    }

    @Test("runtime requires exported gemm_q4 pipeline")
    func missingPipelineFailsClosed() throws {
        let device = try device()
        let metallib = try Metallib(device: device)
        #expect(metallib.functionNames.contains("gemm_q4"))
        #expect(throws: SploshError.missingMetallibFunction("not_gemm_q4")) { try metallib.pipeline("not_gemm_q4") }
    }
}
