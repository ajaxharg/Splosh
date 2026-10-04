// ModelConfig.swift — validated Qwen3.5 model configuration.
import Foundation
import CryptoKit

public enum ModelConfigError: Error, Equatable, CustomStringConvertible {
    case invalid(String)
    case hashMismatch(expected: String, observed: String)
    public var description: String {
        switch self { case .invalid(let s): return "invalid model config: \(s)"; case .hashMismatch(let e, let o): return "model config hash mismatch: expected \(e), observed \(o)" }
    }
}

public struct ModelConfig: Codable, Sendable, Equatable {
    public struct Text: Codable, Sendable, Equatable {
        public let hiddenSize: Int; public let intermediateSize: Int; public let numHiddenLayers: Int; public let layerTypes: [String]
        public let numKeyValueHeads: Int; public let numAttentionHeads: Int; public let headDim: Int; public let vocabSize: Int; public let rmsNormEps: Double
        enum CodingKeys: String, CodingKey { case hiddenSize = "hidden_size", intermediateSize = "intermediate_size", numHiddenLayers = "num_hidden_layers", layerTypes = "layer_types", numKeyValueHeads = "num_key_value_heads", numAttentionHeads = "num_attention_heads", headDim = "head_dim", vocabSize = "vocab_size", rmsNormEps = "rms_norm_eps" }
    }
    public let architectures: [String]; public let modelType: String; public let text: Text
    public var configHash: String = ""
    private enum CodingKeys: String, CodingKey { case architectures; case modelType = "model_type"; case text = "text_config" }
    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        architectures = try c.decode([String].self, forKey: .architectures); modelType = try c.decode(String.self, forKey: .modelType); text = try c.decode(Text.self, forKey: .text); configHash = ""
    }
    public func encode(to encoder: Encoder) throws { var c = encoder.container(keyedBy: CodingKeys.self); try c.encode(architectures, forKey: .architectures); try c.encode(modelType, forKey: .modelType); try c.encode(text, forKey: .text) }
    public init(data: Data) throws { var d = try JSONDecoder().decode(Self.self, from: data); d.configHash = Self.sha256(data); self = d; try validate() }
    public init(url: URL) throws { try self.init(data: Data(contentsOf: url)) }
    public func validate(expectedHash: String? = nil) throws {
        guard architectures.contains("Qwen3_5ForConditionalGeneration"), modelType == "qwen3_5" else { throw ModelConfigError.invalid("architecture/model_type") }
        guard text.numHiddenLayers == 64, text.layerTypes.count == 64 else { throw ModelConfigError.invalid("layer count") }
        guard text.hiddenSize == 5120, text.intermediateSize == 17408 else { throw ModelConfigError.invalid("hidden/intermediate size") }
        guard text.layerTypes.allSatisfy({ $0 == "linear_attention" || $0 == "full_attention" }) else { throw ModelConfigError.invalid("unknown layer type") }
        let expectedLayerTypes = (0..<16).flatMap { _ in ["linear_attention", "linear_attention", "linear_attention", "full_attention"] }
        guard text.layerTypes == expectedLayerTypes else { throw ModelConfigError.invalid("layer type pattern") }
        guard text.numKeyValueHeads > 0, text.numAttentionHeads > 0, text.headDim > 0, text.vocabSize > 0, text.rmsNormEps > 0 else { throw ModelConfigError.invalid("non-positive structural field") }
        if let expectedHash, expectedHash.lowercased() != configHash.lowercased() { throw ModelConfigError.hashMismatch(expected: expectedHash, observed: configHash) }
    }
    public static func sha256(_ data: Data) -> String { SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined() }
}
