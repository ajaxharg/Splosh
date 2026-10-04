import Foundation
import Testing
import SploshModel

@Suite("ModelConfig Codable")
struct ModelConfigTests {
    private var json: Data {
        let layers = Array(repeating: ["linear_attention", "linear_attention", "linear_attention", "full_attention"], count: 16).flatMap { $0 }
        let object: [String: Any] = [
            "architectures": ["Qwen3_5ForConditionalGeneration"],
            "model_type": "qwen3_5",
            "text_config": [
                "hidden_size": 5120, "intermediate_size": 17408,
                "num_hidden_layers": 64, "layer_types": layers,
                "num_key_value_heads": 4, "num_attention_heads": 24,
                "head_dim": 256, "vocab_size": 248320, "rms_norm_eps": 0.000001
            ]
        ]
        return try! JSONSerialization.data(withJSONObject: object)
    }

    private func data(changing key: String, to value: Any) throws -> Data {
        var object = try #require(JSONSerialization.jsonObject(with: json) as? [String: Any])
        var text = try #require(object["text_config"] as? [String: Any])
        text[key] = value
        object["text_config"] = text
        return try JSONSerialization.data(withJSONObject: object)
    }

    @Test("valid synthetic configuration validates")
    func validConfiguration() throws {
        let config = try ModelConfig(data: json)
        try config.validate(expectedHash: config.configHash)
    }

    @Test("structural validation rejects malformed configurations")
    func malformedConfigurations() throws {
        let cases: [(String, Data, String)] = [
            ("wrong layer count", try data(changing: "num_hidden_layers", to: 63), "layer count"),
            ("wrong dimensions", try data(changing: "hidden_size", to: 256), "hidden/intermediate size"),
            ("unknown layer type", try data(changing: "layer_types", to: Array(repeating: "mystery", count: 64)), "unknown layer type"),
            ("non-positive field", try data(changing: "head_dim", to: 0), "non-positive structural field")
        ]
        for (_, malformed, category) in cases {
            do {
                _ = try ModelConfig(data: malformed)
                Issue.record("Expected invalid configuration for \(category)")
            } catch let error as ModelConfigError {
                guard case .invalid(let diagnostic) = error else {
                    Issue.record("Expected invalid error for \(category), got \(error)")
                    continue
                }
                #expect(diagnostic.contains(category))
            }
        }
    }

    @Test("derived configHash is not part of Codable payload")
    func hashIsDerivedAndOmitted() throws {
        let config = try ModelConfig(data: json)
        #expect(config.configHash.count == 64)

        let encoded = try JSONEncoder().encode(config)
        let object = try #require(JSONSerialization.jsonObject(with: encoded) as? [String: Any])
        #expect(object["configHash"] == nil)
        #expect(object["config_hash"] == nil)

        let decoded = try JSONDecoder().decode(ModelConfig.self, from: encoded)
        #expect(decoded.configHash.isEmpty)
        #expect(decoded.architectures == config.architectures)
        #expect(decoded.modelType == config.modelType)
        #expect(decoded.text == config.text)
    }
}
