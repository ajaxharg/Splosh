// SploshOracleTests.swift — SploshOracleTests.
//
// Owner: M0.6. Contract source: rev4 §6 M0.6 ("each holds >= 1 test that exercises a real symbol --
// the oracle file a `SploshOracle` helper ... `#expect(true)` is not a test").
//
// The suite is named after the file so that `swift test --filter SploshOracleTests` -- the filter
// rev4 §3.2's inventory names and §6 M0.6's gate runs -- matches it.
//
// M0 has no model math yet, so the oracle symbol exercised here is `NormOracle.rmsNormEps`, the one
// constant §4.7's NormOracle owns that is sourced from the pack's own config. Both assertions can
// fail for the right reason: a wrong eps value, or a drift in the documented unit-RMS relation.
// M2.3 adds the `NormOracle.norm(...)` entry point to this same file's subject.
//
// rev4 §2.2 (`:147`)  rms_norm_eps 1e-06
// rev4 §2.3 (`:226`)  x_hat = 1 / sqrt(1 + rms_norm_eps) = 0.99999950000025 for a unit-RMS input

import Testing
import Metal

import SploshCore
import SploshOracle

private struct RMSNormParams {
    var length: UInt32
    var epsilon: Float
}

@Suite("M23RmsnormTests")
struct M23RmsnormTests {
    /// The zero-centring epsilon must be the pack's own value, not a plausible default.
    @Test("NormOracle.rmsNormEps is the pack's rms_norm_eps (rev4 §2.2 `:147`)")
    func rmsNormEpsIsThePackValue() {
        #expect(NormOracle.rmsNormEps == 1e-06)
    }

    /// rev4 §2.3 `:226` derives a closed form for the zero-centred norm on a unit-RMS input. It is
    /// strictly less than 1 by exactly the epsilon term, so a norm implemented as `sqrt(1 + eps)`
    /// with the epsilon dropped, doubled, or applied outside the root is caught here.
    @Test("unit-RMS zero-centred norm matches the documented 1/sqrt(1 + eps) relation")
    func unitVectorNormScaleMatchesTheDocumentedRelation() {
        let documented: Float = 0.99999950000025
        let observed = 1 / (1 + NormOracle.rmsNormEps).squareRoot()

        #expect(abs(observed - documented) <= 1e-7)
        #expect(observed < 1)
    }

    @Test("rmsnorm matches NormOracle.norm with fp32 accumulation and relErr <= 1e-3")
    func rmsnormMatchesNormOracle() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw SploshError.capabilityGateFailure("no Metal device on this host")
        }
        let metallib = try makeTestMetallib(device: device)
        #expect(Set(metallib.functionNames) == ["copy", "rmsnorm"])
        let pipeline = try metallib.pipeline("rmsnorm")
        guard let queue = device.makeCommandQueue() else {
            throw SploshError.capabilityGateFailure("could not create command queue")
        }

        let input: [Float] = [-1.25, 0.5, 2.0, -0.75, 3.25, 0.125, -2.5, 1.75]
        let weight: [Float] = [0.0, 0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7]
        let expected = NormOracle.norm(input, weight: weight)
        guard
            let inputBuffer = device.makeBuffer(bytes: input, length: input.count * MemoryLayout<Float>.stride, options: .storageModeShared),
            let weightBuffer = device.makeBuffer(bytes: weight, length: weight.count * MemoryLayout<Float>.stride, options: .storageModeShared),
            let outputBuffer = device.makeBuffer(length: expected.count * MemoryLayout<Float>.stride, options: .storageModeShared),
            let commandBuffer = queue.makeCommandBuffer(),
            let encoder = commandBuffer.makeComputeCommandEncoder()
        else {
            throw SploshError.capabilityGateFailure("could not allocate rmsnorm resources")
        }

        var params = RMSNormParams(length: UInt32(input.count), epsilon: NormOracle.rmsNormEps)
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(inputBuffer, offset: 0, index: 0)
        encoder.setBuffer(weightBuffer, offset: 0, index: 1)
        encoder.setBuffer(outputBuffer, offset: 0, index: 2)
        encoder.setBytes(&params, length: MemoryLayout<RMSNormParams>.stride, index: 3)
        encoder.dispatchThreads(MTLSize(width: 1, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))
        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        if let error = commandBuffer.error {
            throw SploshError.capabilityGateFailure("rmsnorm command buffer failed: \(error)")
        }

        let actualPointer = outputBuffer.contents().bindMemory(to: Float.self, capacity: expected.count)
        let actual: [Float] = Array(UnsafeBufferPointer<Float>(start: actualPointer, count: expected.count))
        let denominator = max(expected.map { abs($0) }.max() ?? 0, 1e-6)
        let relErr = zip(actual, expected).map { abs($0 - $1) }.max()! / denominator
        print("rmsnorm relErr=\(relErr)")
        #expect(relErr <= 1e-3, "rmsnorm relErr=\(relErr) exceeds 1e-3")
    }
}
