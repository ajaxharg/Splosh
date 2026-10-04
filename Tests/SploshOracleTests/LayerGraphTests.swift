import Foundation
import Testing
import SploshModel

/// Synthetic fixtures only: these entries are generated and do not represent a real model pack.
@Suite("LayerGraph synthetic fixtures")
struct LayerGraphTests {
    private let layerPattern = Array(repeating: ["linear_attention", "linear_attention", "linear_attention", "full_attention"], count: 16).flatMap { $0 }

    private func configData(layerTypes: [String]? = nil) -> Data {
        let types = layerTypes ?? layerPattern
        let encoded = types.map { "\"\($0)\"" }.joined(separator: ",")
        return Data("""
        {"architectures":["Qwen3_5ForConditionalGeneration"],"model_type":"qwen3_5","text_config":{"hidden_size":5120,"intermediate_size":17408,"num_hidden_layers":64,"layer_types":[\(encoded)],"num_key_value_heads":4,"num_attention_heads":24,"head_dim":256,"vocab_size":248320,"rms_norm_eps":0.000001}}
        """.utf8)
    }

    private func config(layerTypes: [String]? = nil) throws -> ModelConfig {
        try ModelConfig(data: configData(layerTypes: layerTypes))
    }

    private func suffixes(for kind: LayerKind) -> [String] {
        switch kind {
        case .linearAttention:
            return ["linear_attn.in_proj_qkv", "linear_attn.in_proj_a", "linear_attn.in_proj_b", "linear_attn.in_proj_z", "linear_attn.out_proj", "linear_attn.conv1d", "linear_attn.A_log", "linear_attn.dt_bias", "linear_attn.norm", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj", "input_layernorm", "post_attention_layernorm"]
        case .fullAttention:
            return ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj", "input_layernorm", "post_attention_layernorm", "self_attn.q_norm", "self_attn.k_norm"]
        }
    }

    private func shape(for suffix: String, config: ModelConfig) -> [Int] {
        let h = config.text.hiddenSize, m = config.text.intermediateSize
        switch suffix {
        case "mlp.gate_proj", "mlp.up_proj": return [m, h]
        case "mlp.down_proj": return [h, m]
        case "input_layernorm", "post_attention_layernorm": return [h]
        case "linear_attn.in_proj_qkv": return [2 * h, h]
        case "linear_attn.in_proj_a", "linear_attn.in_proj_b": return [2 * config.text.numAttentionHeads, h]
        case "linear_attn.in_proj_z", "linear_attn.out_proj": return [config.text.numAttentionHeads * config.text.headDim, h]
        case "linear_attn.conv1d": return [2 * h, 1, 4]
        case "linear_attn.A_log", "linear_attn.dt_bias": return [2 * config.text.numAttentionHeads]
        case "linear_attn.norm": return [config.text.headDim / 2]
        case "self_attn.q_proj": return [2 * config.text.numAttentionHeads * config.text.headDim, h]
        case "self_attn.k_proj", "self_attn.v_proj": return [config.text.numKeyValueHeads * config.text.headDim, h]
        case "self_attn.o_proj": return [h, config.text.numAttentionHeads * config.text.headDim]
        case "self_attn.q_norm", "self_attn.k_norm": return [config.text.headDim]
        default: fatalError("unhandled synthetic suffix \(suffix)")
        }
    }

    private func entries(config: ModelConfig, mutate: ((inout [TensorEntry]) -> Void)? = nil) throws -> [TensorEntry] {
        var result: [TensorEntry] = []
        for i in 0..<64 {
            let kind = LayerGraph.expectedKinds[i]
            for suffix in suffixes(for: kind) {
                result.append(TensorEntry(name: "model.layers.\(i).\(suffix)", shard: "synthetic-00001.safetensors", shape: shape(for: suffix, config: config), dtype: "BF16"))
            }
        }
        var filler = 0
        while result.count < 576 {
            result.append(TensorEntry(name: "synthetic.filler.\(filler)", shard: "synthetic-00001.safetensors", shape: [1], dtype: "BF16"))
            filler += 1
        }
        mutate?(&result)
        return result
    }

    private func graph(mutate: ((inout [TensorEntry]) -> Void)? = nil, layerTypes: [String]? = nil) throws -> LayerGraph {
        let config = try config(layerTypes: layerTypes)
        return try LayerGraph(config: config, index: WeightIndex(entries: entries(config: config, mutate: mutate)))
    }

    @Test("synthetic graph has the exact 64-layer sequence literal")
    func exactSequence() throws {
        let graph = try graph()
        #expect(graph.layers.map(\.kind) == LayerGraph.expectedKinds)
        #expect(graph.layers.map(\.index) == Array(0..<64))
    }

    @Test("synthetic fixture resolves every layer and required tensor")
    func completeResolution() throws {
        let graph = try graph()
        #expect(graph.layers.count == 64)
        #expect(graph.layers.allSatisfy { !$0.tensorNames.isEmpty })
        #expect(Set(graph.layers.flatMap(\.tensorNames)).count == 48 * 14 + 16 * 11)
    }

    @Test("synthetic required entries preserve BF16 and dimensional metadata")
    func bf16AndDimensions() throws {
        let c = try config()
        let index = try WeightIndex(entries: entries(config: c))
        let qkv = try #require(index.entry("model.layers.0.linear_attn.in_proj_qkv"))
        #expect(qkv.tensorDType == .bf16)
        #expect(qkv.shape == [10240, 5120])
        let key = try #require(index.entry("model.layers.3.self_attn.k_proj"))
        #expect(key.tensorDType == .bf16)
        #expect(key.shape == [1024, 5120])
        #expect(index.tensorCount >= 576)
    }

    @Test("synthetic missing tensor fails closed")
    func missingTensor() throws {
        #expect(throws: LayerGraphError.self) {
            try graph { entries in entries.removeAll { $0.name == "model.layers.0.linear_attn.in_proj_qkv" } }
        }
    }

    @Test("synthetic wrong layer sequence fails closed")
    func wrongLayerSequence() throws {
        var wrong = layerPattern
        wrong[3] = "linear_attention"
        #expect(throws: (any Error).self) { try graph(layerTypes: wrong) }
    }

    @Test("synthetic wrong shape fails closed")
    func wrongShape() throws {
        #expect(throws: LayerGraphError.self) {
            try graph { entries in
                let i = entries.firstIndex { $0.name == "model.layers.0.linear_attn.in_proj_qkv" }!
                entries[i] = TensorEntry(name: entries[i].name, shard: entries[i].shard, shape: [1], dtype: "BF16")
            }
        }
    }

    @Test("synthetic wrong dtype fails closed")
    func wrongDType() throws {
        #expect(throws: LayerGraphError.self) {
            try graph { entries in
                let i = entries.firstIndex { $0.name == "model.layers.3.self_attn.q_proj" }!
                entries[i] = TensorEntry(name: entries[i].name, shard: entries[i].shard, shape: entries[i].shape, dtype: "F32")
            }
        }
    }

    @Test("synthetic insufficient index fails closed")
    func insufficientIndex() throws {
        let c = try config()
        let all = try entries(config: c)
        let result = try? LayerGraph(config: c, index: WeightIndex(entries: Array(all.prefix(575))))
        #expect(result == nil)
    }
}
