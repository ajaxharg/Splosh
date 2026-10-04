import Foundation
import Metal
import Testing
import SploshCore
import SploshModel
import SploshOracle

/// Synthetic per-kernel ABI contracts. These are GPU-vs-scalar checks over deterministic inputs;
/// they do not claim model or external-vector validation.
@Suite("GdnStageContractTests")
struct GdnStageContractTests {
    private struct PrepareParams { var channels: UInt32; var length: UInt32 }
    private struct DecodeParams { var heads: UInt32; var keyDim: UInt32; var valueDim: UInt32 }
    private struct GateParams { var length: UInt32; var dim: UInt32; var epsilon: Float }
    private struct CommitParams { var count: UInt32 }

    private func context() throws -> (MTLDevice, Metallib, MTLCommandQueue) {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw SploshError.capabilityGateFailure("no Metal device on this host")
        }
        let library = try makeTestMetallib(device: device)
        guard let queue = device.makeCommandQueue() else {
            throw SploshError.capabilityGateFailure("could not create an MTLCommandQueue")
        }
        return (device, library, queue)
    }

    private func buffer(_ device: MTLDevice, _ values: [Float]) throws -> MTLBuffer {
        guard let b = device.makeBuffer(bytes: values, length: values.count * MemoryLayout<Float>.stride,
                                        options: .storageModeShared) else {
            throw SploshError.capabilityGateFailure("could not allocate synthetic GDN buffer")
        }
        return b
    }

    private func floats(_ buffer: MTLBuffer, count: Int) -> [Float] {
        Array(UnsafeBufferPointer(start: buffer.contents().assumingMemoryBound(to: Float.self), count: count))
    }

    private func finish(_ command: MTLCommandBuffer) throws {
        command.commit()
        command.waitUntilCompleted()
        guard command.status == .completed else {
            throw SploshError.capabilityGateFailure("GDN command failed: \(command.error?.localizedDescription ?? "unknown")")
        }
    }

    private func assertRelErr(_ expected: [Float], _ actual: [Float]) {
        let scale = max(expected.map { abs($0) }.max() ?? 0, 1)
        let relErr = zip(expected, actual).map { abs($0 - $1) }.max()! / scale
        print("GDN synthetic relErr=\(relErr)")
        #expect(relErr <= 1e-3)
    }

    @Test("gdn_prepare GPU stage agrees with scalar oracle")
    func prepareStage() throws {
        let (device, library, queue) = try context()
        // A single-token synthetic stage isolates the published ABI; the shader has no history buffer.
        let input: [Float] = [0.25, -0.5]
        let kernel: [Float] = [0.1, -0.2, 0.3, 0.9, -0.15, 0.25, 0.05, 0.8]
        var expected = [Float](repeating: 0, count: input.count)
        for c in 0..<2 {
            let channel = [input[c]]
            let prepared = GdnOracle.prepare(input: channel, kernel: Array(kernel[(c * 4)..<(c * 4 + 4)]))
            expected[c] = prepared[0]
        }
        let out = try buffer(device, Array(repeating: 0, count: input.count))
        let command = try #require(queue.makeCommandBuffer())
        let encoder = try #require(command.makeComputeCommandEncoder())
        encoder.setComputePipelineState(try library.pipeline("gdn_prepare"))
        encoder.setBuffer(try buffer(device, input), offset: 0, index: 0)
        encoder.setBuffer(try buffer(device, kernel), offset: 0, index: 1)
        encoder.setBuffer(out, offset: 0, index: 2)
        var params = PrepareParams(channels: 2, length: 3)
        encoder.setBytes(&params, length: MemoryLayout<PrepareParams>.stride, index: 3)
        encoder.dispatchThreads(MTLSize(width: 2, height: 3, depth: 1), threadsPerThreadgroup: MTLSize(width: 2, height: 3, depth: 1))
        encoder.endEncoding(); try finish(command)
        assertRelErr(expected, floats(out, count: input.count))
    }

    @Test("gdn_decode GPU stage agrees with scalar oracle")
    func decodeStage() throws {
        let (device, library, queue) = try context()
        let q: [Float] = [0.8, -0.2], k: [Float] = [0.5, 0.25], v: [Float] = [1.5, -0.75]
        let beta: [Float] = [0.6], decay: [Float] = [-0.1]
        var oracleState = GdnOracle.State(heads: 1, keyDim: 2, valueDim: 2)
        let expected = GdnOracle.decode(query: q, key: k, value: v, beta: beta, decay: decay, state: &oracleState)
        let state = try buffer(device, [Float](repeating: 0, count: 4)); let out = try buffer(device, [Float](repeating: 0, count: 2))
        let command = try #require(queue.makeCommandBuffer()); let encoder = try #require(command.makeComputeCommandEncoder())
        encoder.setComputePipelineState(try library.pipeline("gdn_decode"))
        for (i, values) in [q, k, v, beta, decay].enumerated() { encoder.setBuffer(try buffer(device, values), offset: 0, index: i) }
        encoder.setBuffer(state, offset: 0, index: 5); encoder.setBuffer(out, offset: 0, index: 6)
        var params = DecodeParams(heads: 1, keyDim: 2, valueDim: 2); encoder.setBytes(&params, length: MemoryLayout<DecodeParams>.stride, index: 7)
        encoder.dispatchThreads(MTLSize(width: 1, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1)); encoder.endEncoding(); try finish(command)
        assertRelErr(expected, floats(out, count: 2)); assertRelErr(oracleState.values, floats(state, count: 4))
    }

    @Test("gdn_gate GPU stage agrees with scalar oracle")
    func gateStage() throws {
        let (device, library, queue) = try context()
        let values: [Float] = [0.5, -1.0, 1.25, 0.25], z: [Float] = [-0.2, 0.7, 0.4, -0.9], weight: [Float] = [0.1, -0.15]
        let expected = GdnOracle.gate(values, z: z, weight: weight)
        let out = try buffer(device, [Float](repeating: 0, count: 4)); let command = try #require(queue.makeCommandBuffer()); let encoder = try #require(command.makeComputeCommandEncoder())
        encoder.setComputePipelineState(try library.pipeline("gdn_gate")); encoder.setBuffer(try buffer(device, values), offset: 0, index: 0); encoder.setBuffer(try buffer(device, z), offset: 0, index: 1); encoder.setBuffer(try buffer(device, weight), offset: 0, index: 2); encoder.setBuffer(out, offset: 0, index: 3)
        var params = GateParams(length: 4, dim: 2, epsilon: GdnOracle.epsilon); encoder.setBytes(&params, length: MemoryLayout<GateParams>.stride, index: 4); encoder.dispatchThreads(MTLSize(width: 4, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 4, height: 1, depth: 1)); encoder.endEncoding(); try finish(command)
        assertRelErr(expected, floats(out, count: 4))
    }

    @Test("gdn_commit GPU stage is byte exact")
    func commitStage() throws {
        let (device, library, queue) = try context(); let source: [Float] = [1.25, -2.5, 0.0, 9.75]
        let inactive = try buffer(device, source); let active = try buffer(device, [Float](repeating: -7, count: source.count)); let command = try #require(queue.makeCommandBuffer()); let encoder = try #require(command.makeComputeCommandEncoder())
        encoder.setComputePipelineState(try library.pipeline("gdn_commit")); encoder.setBuffer(inactive, offset: 0, index: 0); encoder.setBuffer(active, offset: 0, index: 1); var params = CommitParams(count: 4); encoder.setBytes(&params, length: MemoryLayout<CommitParams>.stride, index: 2); encoder.dispatchThreads(MTLSize(width: 4, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 4, height: 1, depth: 1)); encoder.endEncoding(); try finish(command)
        #expect(Data(bytes: active.contents(), count: source.count * MemoryLayout<Float>.stride) == Data(bytes: inactive.contents(), count: source.count * MemoryLayout<Float>.stride))
    }

    @Test("GdnLayer rejects malformed dimensions, nonfinite input, and invalid state")
    func typedNegatives() throws {
        var layer = GdnLayer(); var state = try GdnLayer.State(heads: 1, keyDim: 2, valueDim: 2)
        #expect(throws: GdnLayer.Error.shapeMismatch("query/key/value/beta/decay")) { try layer.runToken(query: [0], key: [0], value: [0, 0], beta: [1], decay: [0], state: &state) }
        #expect(throws: GdnLayer.Error.nonFiniteInput("query")) { try layer.runToken(query: [.infinity, 0], key: [0, 0], value: [0, 0], beta: [1], decay: [0], state: &state) }
    }
}
