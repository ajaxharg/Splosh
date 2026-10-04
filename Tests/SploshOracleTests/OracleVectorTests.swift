// OracleVectorTests.swift — M2.13 independent external vectors.
// The contract is intentionally fail-closed: vectors are never generated here.
import Foundation
import Testing
import SploshCore
import SploshOracle

@Suite("OracleVectorTests")
struct OracleVectorTests {
    private static let vectorDirectory = URL(fileURLWithPath: #filePath)
        .deletingLastPathComponent().deletingLastPathComponent()
        .appendingPathComponent("Goldens/oracle_vectors", isDirectory: true)

    private struct Provenance: Decodable {
        let producer, source, generatedAt, inputDigest, outputDigest: String
        enum CodingKeys: String, CodingKey, CaseIterable { case producer, source, generatedAt, inputDigest, outputDigest }
        init(from decoder: Decoder) throws {
            let c = try decoder.container(keyedBy: CodingKeys.self)
            try Self.exact(c, name: "provenance")
            producer = try c.decode(String.self, forKey: .producer)
            source = try c.decode(String.self, forKey: .source)
            generatedAt = try c.decode(String.self, forKey: .generatedAt)
            inputDigest = try c.decode(String.self, forKey: .inputDigest)
            outputDigest = try c.decode(String.self, forKey: .outputDigest)
            guard !producer.isEmpty, !source.isEmpty, !generatedAt.isEmpty,
                  !inputDigest.isEmpty, !outputDigest.isEmpty,
                  generatedAt.hasSuffix("Z") else { throw Provenance.bad("M2.13 invalid provenance") }
        }
        static func exact<K: CodingKey>(_ c: KeyedDecodingContainer<K>, name: String) throws where K: CaseIterable {
            guard Set(c.allKeys.map(\ .stringValue)) == Set(K.allCases.map(\ .stringValue)) else {
                throw Provenance.bad("M2.13 malformed \(name): unexpected or missing fields")
            }
        }
        static func bad(_ m: String) -> SploshError { .capabilityGateFailure(m) }
    }

    private struct NormInput: Decodable {
        let values, weight: [Float]; let epsilon: Float
        enum CodingKeys: String, CodingKey, CaseIterable { case values, weight, epsilon }
        init(from decoder: Decoder) throws {
            let c = try decoder.container(keyedBy: CodingKeys.self); try Provenance.exact(c, name: "norm input")
            values = try c.decode([Float].self, forKey: .values); weight = try c.decode([Float].self, forKey: .weight)
            epsilon = try c.decode(Float.self, forKey: .epsilon)
            guard !values.isEmpty, values.count == weight.count, epsilon.isFinite, epsilon > 0,
                  values.allSatisfy(\.isFinite), weight.allSatisfy(\.isFinite) else { throw Provenance.bad("M2.13 norm input has invalid finite values or shape") }
        }
    }
    private struct NormExpected: Decodable {
        let values: [Float]
        enum CodingKeys: String, CodingKey, CaseIterable { case values }
        init(from decoder: Decoder) throws {
            let c = try decoder.container(keyedBy: CodingKeys.self); try Provenance.exact(c, name: "norm expected")
            values = try c.decode([Float].self, forKey: .values)
            guard !values.isEmpty, values.allSatisfy(\.isFinite) else { throw Provenance.bad("M2.13 norm expected has invalid values") }
        }
    }
    private struct NormEnvelope: Decodable {
        let schema: String; let operation: String; let provenance: Provenance
        let tolerance, relErr: Float; let input: NormInput; let expected: NormExpected; let zeroCentredDistinction: Float
        enum CodingKeys: String, CodingKey, CaseIterable { case schema, operation, provenance, tolerance, relErr, input, expected, zeroCentredDistinction }
        init(from decoder: Decoder) throws {
            let c = try decoder.container(keyedBy: CodingKeys.self); try Provenance.exact(c, name: "norm envelope")
            schema = try c.decode(String.self, forKey: .schema); operation = try c.decode(String.self, forKey: .operation)
            provenance = try c.decode(Provenance.self, forKey: .provenance); tolerance = try c.decode(Float.self, forKey: .tolerance)
            relErr = try c.decode(Float.self, forKey: .relErr); input = try c.decode(NormInput.self, forKey: .input)
            expected = try c.decode(NormExpected.self, forKey: .expected)
            zeroCentredDistinction = try c.decode(Float.self, forKey: .zeroCentredDistinction)
            guard schema == "m2.13.external-vector.v1", operation == "norm", tolerance.isFinite, tolerance >= 0,
                  relErr.isFinite, relErr >= 0, zeroCentredDistinction.isFinite, zeroCentredDistinction >= 0,
                  input.values.count == expected.values.count else { throw Provenance.bad("M2.13 malformed norm envelope") }
        }
    }

    // The documented attention payload is tensor/shape based and does not map to the oracle's
    // undocumented projection arguments. Decode the envelope only, then reject rather than infer.
    private struct AttentionEnvelope: Decodable {
        enum CodingKeys: String, CodingKey { case schema, operation, provenance, tolerance, relErr, input, expected }
        let schema: String; let operation: String; let provenance: Provenance; let tolerance, relErr: Float
        init(from decoder: Decoder) throws {
            let c = try decoder.container(keyedBy: CodingKeys.self)
            schema = try c.decode(String.self, forKey: .schema); operation = try c.decode(String.self, forKey: .operation)
            provenance = try c.decode(Provenance.self, forKey: .provenance); tolerance = try c.decode(Float.self, forKey: .tolerance)
            relErr = try c.decode(Float.self, forKey: .relErr); _ = try c.decode([String: AnyDecodable].self, forKey: .input)
            _ = try c.decode([String: AnyDecodable].self, forKey: .expected)
            guard schema == "m2.13.external-vector.v1", operation == "attention_decode" || operation == "attention_prefill",
                  tolerance.isFinite, tolerance >= 0, relErr.isFinite, relErr >= 0 else { throw Provenance.bad("M2.13 malformed attention envelope") }
            throw Provenance.bad("M2.13 \(operation) rejected: documented tensor/shape payload is not supported by AttentionOracle without inference")
        }
    }
    private struct AnyDecodable: Decodable { init(from decoder: Decoder) throws { _ = try decoder.singleValueContainer() } }

    private struct GDNInput: Decodable {
        let tokens, chunkSize, chunks, chunkBoundaries: Int; let query, key, value, beta, decay: [[Float]]
        enum CodingKeys: String, CodingKey, CaseIterable { case tokens, chunkSize, chunks, chunkBoundaries, query, key, value, beta, decay }
        init(from decoder: Decoder) throws {
            let c = try decoder.container(keyedBy: CodingKeys.self); try Provenance.exact(c, name: "gdn input")
            tokens = try c.decode(Int.self, forKey: .tokens); chunkSize = try c.decode(Int.self, forKey: .chunkSize)
            chunks = try c.decode(Int.self, forKey: .chunks); chunkBoundaries = try c.decode(Int.self, forKey: .chunkBoundaries)
            query = try c.decode([[Float]].self, forKey: .query); key = try c.decode([[Float]].self, forKey: .key)
            value = try c.decode([[Float]].self, forKey: .value); beta = try c.decode([[Float]].self, forKey: .beta)
            decay = try c.decode([[Float]].self, forKey: .decay)
            guard tokens == 512, chunkSize == 64, chunks == 8, chunkBoundaries >= 8,
                  query.count == tokens, key.count == tokens, value.count == tokens, beta.count == tokens, decay.count == tokens,
                  query.allSatisfy({ $0.allSatisfy(\.isFinite) }), key.allSatisfy({ $0.allSatisfy(\.isFinite) }),
                  value.allSatisfy({ $0.allSatisfy(\.isFinite) }), beta.allSatisfy({ $0.allSatisfy(\.isFinite) }), decay.allSatisfy({ $0.allSatisfy(\.isFinite) }) else { throw Provenance.bad("M2.13 malformed gdn geometry or non-finite input") }
        }
    }
    private struct GDNExpected: Decodable {
        let output, state: [Float]
        enum CodingKeys: String, CodingKey, CaseIterable { case output, state }
        init(from decoder: Decoder) throws { let c = try decoder.container(keyedBy: CodingKeys.self); try Provenance.exact(c, name: "gdn expected"); output = try c.decode([Float].self, forKey: .output); state = try c.decode([Float].self, forKey: .state); guard output.allSatisfy(\.isFinite), state.allSatisfy(\.isFinite) else { throw Provenance.bad("M2.13 gdn expected contains non-finite values") } }
    }
    private struct GDNEnvelope: Decodable {
        let schema: String; let operation: String; let provenance: Provenance; let tolerance, relErr: Float; let input: GDNInput; let expected: GDNExpected
        enum CodingKeys: String, CodingKey, CaseIterable { case schema, operation, provenance, tolerance, relErr, input, expected }
        init(from decoder: Decoder) throws { let c = try decoder.container(keyedBy: CodingKeys.self); try Provenance.exact(c, name: "gdn envelope"); schema = try c.decode(String.self, forKey: .schema); operation = try c.decode(String.self, forKey: .operation); provenance = try c.decode(Provenance.self, forKey: .provenance); tolerance = try c.decode(Float.self, forKey: .tolerance); relErr = try c.decode(Float.self, forKey: .relErr); input = try c.decode(GDNInput.self, forKey: .input); expected = try c.decode(GDNExpected.self, forKey: .expected); guard schema == "m2.13.external-vector.v1", operation == "gdn", tolerance.isFinite, tolerance >= 0, relErr.isFinite, relErr >= 0 else { throw Provenance.bad("M2.13 malformed gdn envelope") } }
    }
    private static func bad(_ m: String) -> SploshError { .capabilityGateFailure(m) }
    private func data(_ name: String) throws -> Data { let u = Self.vectorDirectory.appendingPathComponent(name); guard FileManager.default.fileExists(atPath: u.path), let d = try? Data(contentsOf: u), !d.isEmpty else { throw Self.bad("M2.13 blocked: missing externally generated triple \(u.path); no self-consistency vector is accepted") }; return d }
    private func relErr(_ a: [Float], _ e: [Float]) throws -> Float { guard a.count == e.count, !a.isEmpty else { throw Self.bad("M2.13 output shape differs from external expected tensor") }; return zip(a,e).map { abs($0-$1) / max(abs($1), 1e-6) }.max()! }
    @Test("external norm triple is independently compared") func normExternalTriple() throws { let v = try JSONDecoder().decode(NormEnvelope.self, from: data("norm.json")); let a = NormOracle.norm(v.input.values, weight: v.input.weight, epsilon: v.input.epsilon); let e = try relErr(a,v.expected.values); #expect(v.zeroCentredDistinction >= 1e-2); #expect(v.relErr <= v.tolerance); #expect(e <= v.tolerance) }
    @Test("external attention triples are rejected until documented tensor shape is supported") func attentionExternalTriples() throws { for n in ["attention_decode.json", "attention_prefill.json"] { _ = try JSONDecoder().decode(AttentionEnvelope.self, from: data(n)) } }
    @Test("external GDN triple is independently compared") func gdnExternalTriple() throws { let v = try JSONDecoder().decode(GDNEnvelope.self, from: data("gdn_512.json")); let a = GdnOracle.recurrence(query:v.input.query,key:v.input.key,value:v.input.value,beta:v.input.beta,decay:v.input.decay); let e = try relErr(a.output,v.expected.output); #expect(v.relErr <= v.tolerance); #expect(e <= v.tolerance) }
    @Test("schema negatives reject flat and incomplete envelopes") func schemaNegatives() { let badJSON = ["{}", "{\"x\":[1],\"weight\":[0],\"expected\":[1]}"]; for s in badJSON { #expect(throws: (any Error).self) { _ = try JSONDecoder().decode(NormEnvelope.self, from: Data(s.utf8)) } } }
}
