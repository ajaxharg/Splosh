import Foundation
import Testing
import SploshModel

/// Synthetic host-only validation contract. Every rejected call must be atomic:
/// state and instrumentation remain exactly as they were before validation.
@Suite("GdnValidationContractTests")
struct GdnValidationContractTests {
    private struct Fixture {
        var query: [[Float]]
        var key: [[Float]]
        var value: [[Float]]
        var beta: [[Float]]
        var decay: [[Float]]
    }

    private func fixture(_ count: Int = 64) -> Fixture {
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

    private func assertRejected(
        _ name: String,
        expected: GdnLayer.Error,
        operation: (inout GdnLayer, inout GdnLayer.State) throws -> Void
    ) throws {
        var layer = GdnLayer()
        var state = try GdnLayer.State(heads: 1, keyDim: 2, valueDim: 2)
        let stateBefore = state
        let boundariesBefore = layer.chunkBoundaryCount
        let kernelsBefore = layer.dispatchedKernels
        do {
            try operation(&layer, &state)
            Issue.record("expected \(name) to throw \(expected)")
        } catch let error as GdnLayer.Error {
            #expect(error == expected)
        } catch {
            Issue.record("\(name) threw unexpected error: \(error)")
        }
        #expect(state == stateBefore)
        #expect(layer.chunkBoundaryCount == boundariesBefore)
        #expect(layer.dispatchedKernels == kernelsBefore)
    }

    @Test("empty input is typed and atomic")
    func emptyInput() throws {
        let valid = fixture()
        try assertRejected("empty input", expected: .emptyInput) { layer, state in
            _ = try layer.runChunked(query: [], key: valid.key, value: valid.value, beta: valid.beta, decay: valid.decay, state: &state)
        }
    }

    @Test("invalid chunk size is typed and atomic")
    func invalidChunkSize() throws {
        let valid = fixture()
        try assertRejected("invalid chunk size", expected: .invalidChunkSize(32)) { layer, state in
            _ = try layer.runChunked(query: valid.query, key: valid.key, value: valid.value, beta: valid.beta, decay: valid.decay, state: &state, chunkSize: 32)
        }
    }

    @Test("incomplete chunk is typed and atomic")
    func incompleteChunk() throws {
        let valid = fixture(65)
        try assertRejected("incomplete chunk", expected: .incompleteChunk(1)) { layer, state in
            _ = try layer.runChunked(query: valid.query, key: valid.key, value: valid.value, beta: valid.beta, decay: valid.decay, state: &state)
        }
    }

    @Test("token count mismatch is typed and atomic")
    func tokenCountShapeMismatch() throws {
        let valid = fixture()
        try assertRejected("token count mismatch", expected: .shapeMismatch("token counts")) { layer, state in
            _ = try layer.runChunked(query: valid.query, key: Array(valid.key.dropLast()), value: valid.value, beta: valid.beta, decay: valid.decay, state: &state)
        }
    }

    @Test("each inner token dimension mismatch is typed and atomic")
    func perTokenShapeMismatch() throws {
        let valid = fixture()
        let cases: [(String, (inout Fixture) -> Void)] = [
            ("query", { $0.query[0].append(0) }),
            ("key", { $0.key[0].removeLast() }),
            ("value", { $0.value[0].append(0) }),
            ("beta", { $0.beta[0].append(0) }),
            ("decay", { $0.decay[0].append(0) })
        ]
        for (name, mutate) in cases {
            var malformed = valid
            mutate(&malformed)
            try assertRejected("inner \(name) dimension", expected: .shapeMismatch("query/key/value/beta/decay")) { layer, state in
                _ = try layer.runChunked(query: malformed.query, key: malformed.key, value: malformed.value, beta: malformed.beta, decay: malformed.decay, state: &state)
            }
        }
    }

    @Test("each accepted input array rejects nonfinite values atomically")
    func nonFiniteInputs() throws {
        let valid = fixture()
        let cases: [(String, (inout Fixture) -> Void, GdnLayer.Error)] = [
            ("query", { $0.query[0][0] = .infinity }, .nonFiniteInput("query")),
            ("key", { $0.key[0][0] = -.infinity }, .nonFiniteInput("key")),
            ("value", { $0.value[0][0] = .nan }, .nonFiniteInput("value")),
            ("beta", { $0.beta[0][0] = .infinity }, .nonFiniteInput("beta")),
            ("decay", { $0.decay[0][0] = -.infinity }, .nonFiniteInput("decay"))
        ]
        for (name, mutate, expected) in cases {
            var malformed = valid
            mutate(&malformed)
            try assertRejected("nonfinite \(name)", expected: expected) { layer, state in
                _ = try layer.runChunked(query: malformed.query, key: malformed.key, value: malformed.value, beta: malformed.beta, decay: malformed.decay, state: &state)
            }
        }
    }

    @Test("reachable nonfinite state is typed and atomic")
    func nonFiniteState() throws {
        let valid = fixture()
        var layer = GdnLayer()
        var state = try GdnLayer.State(heads: 1, keyDim: 2, valueDim: 2)
        state.values[0] = .nan
        let stateBefore = state
        let boundariesBefore = layer.chunkBoundaryCount
        let kernelsBefore = layer.dispatchedKernels
        do {
            _ = try layer.runChunked(query: valid.query, key: valid.key, value: valid.value, beta: valid.beta, decay: valid.decay, state: &state)
            Issue.record("expected nonfinite state to throw")
        } catch let error as GdnLayer.Error {
            #expect(error == .nonFiniteInput("state"))
        } catch {
            Issue.record("nonfinite state threw unexpected error: \(error)")
        }
        #expect(state == stateBefore)
        #expect(layer.chunkBoundaryCount == boundariesBefore)
        #expect(layer.dispatchedKernels == kernelsBefore)
    }
}
