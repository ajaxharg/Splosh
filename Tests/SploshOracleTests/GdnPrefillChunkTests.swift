import Testing
import SploshModel
import SploshOracle

@Suite("GdnPrefillChunkTests")
struct GdnPrefillChunkTests {
    private func inputs(_ count: Int) -> ([[Float]], [[Float]], [[Float]], [[Float]], [[Float]]) {
        var q = [[Float]](), k = [[Float]](), v = [[Float]](), b = [[Float]](), d = [[Float]]()
        for t in 0..<count {
            q.append([Float((t % 7) - 3) / 3, Float((t % 5) - 2) / 2])
            k.append([Float((t % 11) - 5) / 5, Float((t % 3) - 1)])
            v.append([Float((t % 13) - 6) / 6, Float((t % 17) - 8) / 8])
            b.append([0.25 + Float(t % 4) * 0.1])
            d.append([-0.01 * Float(t % 3)])
        }
        return (q, k, v, b, d)
    }

    @Test("512 tokens use eight carried-state chunks")
    func eightChunks() throws {
        let (q,k,v,b,d) = inputs(512)
        var oracleState = GdnOracle.State(heads: 1, keyDim: 2, valueDim: 2)
        var expected = [Float]()
        for t in q.indices { expected += GdnOracle.decode(query:q[t], key:k[t], value:v[t], beta:b[t], decay:d[t], state:&oracleState) }
        var layer = GdnLayer()
        var state = try GdnLayer.State(heads: 1, keyDim: 2, valueDim: 2)
        let actual = try layer.runChunked(query:q,key:k,value:v,beta:b,decay:d,state:&state).flatMap{$0}
        let relErr = zip(expected, actual).map { abs($0-$1) }.max()! / max(expected.map(abs).max()!, 1)
        print("relErr=\(relErr) token=512 chunk=64 chunk-count=\(layer.chunkBoundaryCount) boundary-count=\(layer.chunkBoundaryCount)")
        #expect(relErr < 1e-6)
        #expect(layer.chunkBoundaryCount == 8)
        #expect(state.values == oracleState.values)
    }

    @Test("rejects incomplete and wrong chunks")
    func negatives() throws {
        let (q,k,v,b,d) = inputs(65)
        var layer = GdnLayer(); var state = try GdnLayer.State(heads:1,keyDim:2,valueDim:2)
        #expect(throws: GdnLayer.Error.incompleteChunk(1)) { try layer.runChunked(query:q,key:k,value:v,beta:b,decay:d,state:&state) }
        #expect(throws: GdnLayer.Error.invalidChunkSize(32)) { try layer.runChunked(query:Array(q.prefix(64)),key:Array(k.prefix(64)),value:Array(v.prefix(64)),beta:Array(b.prefix(64)),decay:Array(d.prefix(64)),state:&state,chunkSize:32) }
        #expect(throws: GdnLayer.Error.shapeMismatch("token counts")) { try layer.runChunked(query:q,key:Array(k.dropLast()),value:v,beta:b,decay:d,state:&state) }
    }
}
