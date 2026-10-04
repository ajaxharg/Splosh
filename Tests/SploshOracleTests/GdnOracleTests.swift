import Testing
import SploshOracle

@Suite("GdnOracleTests")
struct GdnOracleTests {
    @Test("GdnOracle.decode performs decay, delta write, then read")
    func decodeOrder() {
        var state = GdnOracle.State(heads: 1, keyDim: 2, valueDim: 2)
        let output = GdnOracle.decode(query: [1, 0], key: [1, 0], value: [2, 3], beta: [1], decay: [0], state: &state)
        #expect(output == [2, 3])
        #expect(state.values == [2, 3, 0, 0])
    }

    @Test("GdnOracle recurrence carries fp32 state across tokens")
    func recurrenceCarriesState() {
        let result = GdnOracle.recurrence(
            query: [[1, 0], [1, 0]], key: [[1, 0], [1, 0]],
            value: [[2, 3], [4, 5]], beta: [[1], [0.5]], decay: [[0], [0]])
        #expect(result.output.count == 4)
        #expect(abs(result.output[0] - 2) < 1e-6)
        #expect(abs(result.output[2] - 3) < 1e-6)
    }

    @Test("GdnOracle commit copies inactive parity exactly")
    func commitCopies() {
        let source = GdnOracle.State(values: [1, 2, 3, 4], heads: 1, keyDim: 2, valueDim: 2)
        #expect(GdnOracle.commit(source).values == source.values)
    }
}
