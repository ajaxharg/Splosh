// LayerGraph.swift — validated 64-layer Qwen3.5 graph.
import Foundation

public enum LayerKind: String, Sendable, Equatable, Codable { case linearAttention = "LA", fullAttention = "FA" }
public struct Layer: Sendable, Equatable {
    public let index: Int
    public let kind: LayerKind
    public let tensorNames: [String]
}
public enum LayerGraphError: Error, Equatable, CustomStringConvertible {
    case invalid(String)
    public var description: String {
        switch self {
        case .invalid(let message):
            return "invalid layer graph: \(message)"
        }
    }
}

public struct LayerGraph: Sendable, Equatable {
    public let layers: [Layer]
    public static let expectedKinds: [LayerKind] = (0..<16).flatMap { _ in [.linearAttention, .linearAttention, .linearAttention, .fullAttention] }

    private enum TensorRole {
        case mlpGate, mlpUp, mlpDown, norm, postNorm
        case linearQKV, linearA, linearB, linearZ, linearOut, linearConv, linearDecay, linearDt, linearNorm
        case fullQ, fullK, fullV, fullO, fullQNorm, fullKNorm

        var suffix: String {
            switch self {
            case .mlpGate: return "mlp.gate_proj"
            case .mlpUp: return "mlp.up_proj"
            case .mlpDown: return "mlp.down_proj"
            case .norm: return "input_layernorm"
            case .postNorm: return "post_attention_layernorm"
            case .linearQKV: return "linear_attn.in_proj_qkv"
            case .linearA: return "linear_attn.in_proj_a"
            case .linearB: return "linear_attn.in_proj_b"
            case .linearZ: return "linear_attn.in_proj_z"
            case .linearOut: return "linear_attn.out_proj"
            case .linearConv: return "linear_attn.conv1d"
            case .linearDecay: return "linear_attn.A_log"
            case .linearDt: return "linear_attn.dt_bias"
            case .linearNorm: return "linear_attn.norm"
            case .fullQ: return "self_attn.q_proj"
            case .fullK: return "self_attn.k_proj"
            case .fullV: return "self_attn.v_proj"
            case .fullO: return "self_attn.o_proj"
            case .fullQNorm: return "self_attn.q_norm"
            case .fullKNorm: return "self_attn.k_norm"
            }
        }

        func shape(config: ModelConfig) -> [Int] {
            let h = config.text.hiddenSize, m = config.text.intermediateSize
            let heads = config.text.numAttentionHeads, kv = config.text.numKeyValueHeads, d = config.text.headDim
            let linearValue = heads * d
            let linearChannels = 2 * h
            switch self {
            case .mlpGate, .mlpUp: return [m, h]
            case .mlpDown: return [h, m]
            case .norm, .postNorm: return [h]
            case .linearQKV: return [linearChannels, h]
            case .linearA, .linearB: return [2 * heads, h]
            case .linearZ, .linearOut: return [linearValue, h]
            case .linearConv: return [linearChannels, 1, 4]
            case .linearDecay, .linearDt: return [2 * heads]
            case .linearNorm: return [d / 2]
            case .fullQ: return [heads * d * 2, h]
            case .fullK, .fullV: return [kv * d, h]
            case .fullO: return [h, heads * d]
            case .fullQNorm, .fullKNorm: return [d]
            }
        }
    }

    // One descriptor table is the source of truth for names, roles and shape rules.
    private static let linearDescriptors: [TensorRole] = [.linearQKV, .linearA, .linearB, .linearZ, .linearOut, .linearConv, .linearDecay, .linearDt, .linearNorm, .mlpGate, .mlpUp, .mlpDown, .norm, .postNorm]
    private static let fullDescriptors: [TensorRole] = [.fullQ, .fullK, .fullV, .fullO, .mlpGate, .mlpUp, .mlpDown, .norm, .postNorm, .fullQNorm, .fullKNorm]

    public init(config: ModelConfig, index: WeightIndex) throws {
        do { try config.validate() } catch { throw LayerGraphError.invalid("configuration validation failed: \(error)") }
        guard index.entries.count >= 64 * 9 else { throw LayerGraphError.invalid("index has \(index.entries.count) entries; requires at least 576") }
        let kinds = config.text.layerTypes.map { $0 == "linear_attention" ? LayerKind.linearAttention : .fullAttention }
        guard kinds == Self.expectedKinds else { throw LayerGraphError.invalid("layer_types does not equal [LA,LA,LA,FA] x 16") }
        var result: [Layer] = []
        for i in kinds.indices {
            let descriptors = kinds[i] == .fullAttention ? Self.fullDescriptors : Self.linearDescriptors
            let prefix = "model.layers.\(i)."
            var names: [String] = []
            for role in descriptors {
                let name = prefix + role.suffix
                guard let entry = index.entry(name) else { throw LayerGraphError.invalid("missing tensor \(name)") }
                guard entry.tensorDType == .bf16 else { throw LayerGraphError.invalid("tensor \(name) has wrong dtype; expected BF16") }
                let expected = role.shape(config: config)
                guard !entry.shape.isEmpty, entry.shape.allSatisfy({ $0 >= 0 }) else { throw LayerGraphError.invalid("tensor \(name) has invalid shape \(entry.shape)") }
                guard entry.shape == expected else { throw LayerGraphError.invalid("tensor \(name) has wrong shape; expected \(expected), observed \(entry.shape)") }
                names.append(name)
            }
            result.append(Layer(index: i, kind: kinds[i], tensorNames: names))
        }
        self.layers = result
    }
}
