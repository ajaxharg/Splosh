import Testing
import Metal
import SploshCore
import SploshOracle

private struct RopeMropeParams {
    var headDim: UInt32
    var rotaryDim: UInt32
    var positionT: UInt32
    var positionH: UInt32
    var positionW: UInt32
    var sectionH: UInt32
    var sectionW: UInt32
    var ropeTheta: Float
    var attentionScaling: Float
}

@Suite("M23bRopeTests")
struct M23bRopeTests {
    private func relativeError(_ actual: [Float], _ expected: [Float]) -> Float {
        let numerator = zip(actual, expected).map { abs($0 - $1) }.reduce(0, +)
        let denominator = max(expected.map(abs).reduce(0, +), 1e-12)
        return numerator / denominator
    }

    private func runKernel(_ input: [Float], positions: [Int]) throws -> [Float] {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw SploshError.capabilityGateFailure("no Metal device")
        }
        let metallib = try makeTestMetallib(device: device)
        let pipeline = try metallib.pipeline("rope_mrope")
        let queue = try #require(device.makeCommandQueue())
        let inputBuffer = try #require(device.makeBuffer(bytes: input, length: input.count * MemoryLayout<Float>.stride, options: .storageModeShared))
        let outputBuffer = try #require(device.makeBuffer(length: input.count * MemoryLayout<Float>.stride, options: .storageModeShared))
        var params = RopeMropeParams(
            headDim: 256, rotaryDim: 64,
            positionT: UInt32(positions[0]), positionH: UInt32(positions.count > 1 ? positions[1] : positions[0]),
            positionW: UInt32(positions.count > 2 ? positions[2] : positions[0]),
            sectionH: 11, sectionW: 10, ropeTheta: 10_000_000, attentionScaling: 1
        )
        let command = try #require(queue.makeCommandBuffer())
        let encoder = try #require(command.makeComputeCommandEncoder())
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(inputBuffer, offset: 0, index: 0)
        encoder.setBuffer(outputBuffer, offset: 0, index: 1)
        encoder.setBytes(&params, length: MemoryLayout<RopeMropeParams>.stride, index: 2)
        encoder.dispatchThreads(MTLSize(width: input.count, height: 1, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: min(pipeline.maxTotalThreadsPerThreadgroup, input.count), height: 1, depth: 1))
        encoder.endEncoding(); command.commit(); command.waitUntilCompleted()
        if let error = command.error { throw SploshError.capabilityGateFailure("rope command failed: \(error)") }
        let ptr = outputBuffer.contents().bindMemory(to: Float.self, capacity: input.count)
        return Array(UnsafeBufferPointer(start: ptr, count: input.count))
    }

    @Test("RopeOracle rotates only partial dimensions with mRoPE sections")
    func oracleVector() {
        let input = (0..<256).map { Float($0 + 1) / 17 }
        let output = RopeOracle.apply(input, positionIDs: [3, 5, 7])
        #expect(output.count == 256)
        #expect(output[64...] == input[64...])
        #expect(output != input)
    }

    @Test("rope_mrope matches RopeOracle with relErr <= 1e-3")
    func kernelMatchesOracle() throws {
        let input = (0..<256).map { sin(Float($0) * 0.17) + Float($0 % 11) * 0.03 }
        let expected = RopeOracle.apply(input, positionIDs: [3, 5, 7])
        let actual = try runKernel(input, positions: [3, 5, 7])
        let relErr = relativeError(actual, expected)
        print("rope_mrope relErr=\(relErr)")
        #expect(relErr <= 1e-3, "rope_mrope relErr=\(relErr)")
    }

    @Test("metallib exports exactly the M2.3b set")
    func exactExportSet() throws {
        let device = try #require(MTLCreateSystemDefaultDevice())
        let metallib = try makeTestMetallib(device: device)
        #expect(Set(metallib.functionNames) == ["copy", "rmsnorm", "rope_mrope"])
    }
}
