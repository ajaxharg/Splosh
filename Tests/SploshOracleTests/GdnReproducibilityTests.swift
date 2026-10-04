import Foundation
import Testing
import SploshModel
import SploshOracle

/// Synthetic, deterministic reproducibility evidence for the chunked GDN host contract.
/// This intentionally does not claim model, BF16, external-vector, or golden validation.
@Suite("GdnReproducibilityTests")
struct GdnReproducibilityTests {
    private struct Fixture {
        let query: [[Float]]
        let key: [[Float]]
        let value: [[Float]]
        let beta: [[Float]]
        let decay: [[Float]]
    }

    private func fixture(_ count: Int = 512) -> Fixture {
        var q = [[Float]](), k = [[Float]](), v = [[Float]](), b = [[Float]](), d = [[Float]]()
        for t in 0..<count {
            q.append([Float((t % 7) - 3) / 3, Float((t % 5) - 2) / 2])
            k.append([Float((t % 11) - 5) / 5, Float((t % 3) - 1)])
            v.append([Float((t % 13) - 6) / 6, Float((t % 17) - 8) / 8])
            b.append([0.25 + Float(t % 4) * 0.1])
            d.append([-0.01 * Float(t % 3)])
        }
        return Fixture(query: q, key: k, value: v, beta: b, decay: d)
    }

    private func appendUInt64(_ value: UInt64, to bytes: inout [UInt8]) {
        var value = value.littleEndian
        withUnsafeBytes(of: &value) { bytes.append(contentsOf: $0) }
    }

    private func digest(outputs: [[Float]], state: GdnLayer.State, layer: GdnLayer) -> String {
        var bytes = [UInt8]()
        for value in outputs.flatMap({ $0 }) { appendUInt64(UInt64(value.bitPattern), to: &bytes) }
        for value in state.values { appendUInt64(UInt64(value.bitPattern), to: &bytes) }
        appendUInt64(UInt64(layer.chunkBoundaryCount), to: &bytes)
        appendUInt64(UInt64(layer.dispatchedKernels.count), to: &bytes)
        for kernel in layer.dispatchedKernels {
            let id: UInt64
            switch kernel {
            case "gdn_prepare": id = 1
            case "gdn_decode": id = 2
            case "gdn_gate": id = 3
            case "gdn_commit": id = 4
            default: id = 0
            }
            appendUInt64(id, to: &bytes)
        }
        var hash: UInt64 = 0xcbf29ce484222325
        for byte in bytes {
            hash ^= UInt64(byte)
            hash &*= 0x100000001b3
        }
        return String(format: "%016llx", hash)
    }

    @Test("512-token chunked run is reproducible and oracle-equivalent")
    func reproducibility() throws {
        let input = fixture()
        var firstLayer = GdnLayer()
        var firstState = try GdnLayer.State(heads: 1, keyDim: 2, valueDim: 2)
        let first = try firstLayer.runChunked(query: input.query, key: input.key, value: input.value, beta: input.beta, decay: input.decay, state: &firstState)
        let firstDigest = digest(outputs: first, state: firstState, layer: firstLayer)
        print("GDN_REPRO_DIGEST=\(firstDigest)")

        var secondLayer = GdnLayer()
        var secondState = try GdnLayer.State(heads: 1, keyDim: 2, valueDim: 2)
        let second = try secondLayer.runChunked(query: input.query, key: input.key, value: input.value, beta: input.beta, decay: input.decay, state: &secondState)
        let secondDigest = digest(outputs: second, state: secondState, layer: secondLayer)
        print("GDN_REPRO_DIGEST=\(secondDigest)")

        #expect(firstDigest == secondDigest)
        #expect(firstLayer.chunkBoundaryCount == 8)
        #expect(secondLayer.chunkBoundaryCount == 8)
        #expect(firstLayer.dispatchedKernels == Array(repeating: GdnLayer.kernelOrder, count: 512).flatMap { $0 })
        #expect(secondLayer.dispatchedKernels == firstLayer.dispatchedKernels)

        var oracleState = GdnOracle.State(heads: 1, keyDim: 2, valueDim: 2)
        var expected = [[Float]]()
        for t in input.query.indices {
            expected.append(GdnOracle.decode(query: input.query[t], key: input.key[t], value: input.value[t], beta: input.beta[t], decay: input.decay[t], state: &oracleState))
        }
        #expect(first == expected)
        #expect(firstState.values == oracleState.values)
    }

    @Test("invalid chunked inputs do not mutate state or instrumentation")
    func validationDoesNotMutate() throws {
        let input = fixture()
        let cases: [(String, (inout GdnLayer, inout GdnLayer.State) throws -> Void, GdnLayer.Error)] = [
            ("incomplete", { layer, state in _ = try layer.runChunked(query: Array(input.query.prefix(65)), key: Array(input.key.prefix(65)), value: Array(input.value.prefix(65)), beta: Array(input.beta.prefix(65)), decay: Array(input.decay.prefix(65)), state: &state) }, .incompleteChunk(1)),
            ("wrong chunk", { layer, state in _ = try layer.runChunked(query: input.query, key: input.key, value: input.value, beta: input.beta, decay: input.decay, state: &state, chunkSize: 32) }, .invalidChunkSize(32)),
            ("token mismatch", { layer, state in _ = try layer.runChunked(query: input.query, key: Array(input.key.dropLast()), value: input.value, beta: input.beta, decay: input.decay, state: &state) }, .shapeMismatch("token counts")),
            ("nonfinite", { layer, state in var q = input.query; q[0][0] = .infinity; _ = try layer.runChunked(query: q, key: input.key, value: input.value, beta: input.beta, decay: input.decay, state: &state) }, .nonFiniteInput("query"))
        ]
        for (name, operation, expected) in cases {
            var layer = GdnLayer()
            var state = try GdnLayer.State(heads: 1, keyDim: 2, valueDim: 2)
            let stateBefore = state, boundariesBefore = layer.chunkBoundaryCount, kernelsBefore = layer.dispatchedKernels
            do { try operation(&layer, &state); Issue.record("expected \(name) validation to throw") }
            catch let error as GdnLayer.Error {
                #expect(error == expected)
            }
            catch { Issue.record("unexpected error for \(name): \(error)") }
            #expect(state == stateBefore)
            #expect(layer.chunkBoundaryCount == boundariesBefore)
            #expect(layer.dispatchedKernels == kernelsBefore)
        }
    }
}
