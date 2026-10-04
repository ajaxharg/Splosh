import Foundation
import Testing
import SploshModel

@Suite("Tensor inventory")
struct TensorInventoryTests {
    private func temporaryDirectory() throws -> URL {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("SploshTensorInventory-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        return url
    }

    private func writeShard(_ url: URL, tensors: [(String, String, [Int], Data)], headerOverride: Data? = nil) throws {
        var header: [String: Any] = [:]
        var cursor: Int64 = 0
        for (name, dtype, shape, bytes) in tensors {
            header[name] = ["dtype": dtype, "shape": shape, "data_offsets": [cursor, cursor + Int64(bytes.count)]]
            cursor += Int64(bytes.count)
        }
        let headerData = try headerOverride ?? JSONSerialization.data(withJSONObject: header, options: [.sortedKeys])
        var data = Data()
        var length = UInt64(headerData.count)
        withUnsafeBytes(of: &length) { data.append(contentsOf: $0) }
        data.append(headerData)
        for tensor in tensors { data.append(tensor.3) }
        try data.write(to: url)
    }

    private func writeIndex(_ url: URL, map: [String: String], totalSize: Int64? = nil, sidecarDType: String? = nil, configHash: String? = nil) throws {
        var object: [String: Any] = ["weight_map": map]
        var metadata: [String: Any] = [:]
        if let totalSize { metadata["total_size"] = totalSize }
        if let sidecarDType { metadata["sidecar_dtype"] = sidecarDType }
        if !metadata.isEmpty { object["metadata"] = metadata }
        if let configHash { object["config_hash"] = configHash }
        try JSONSerialization.data(withJSONObject: object, options: [.sortedKeys]).write(to: url)
    }

    private func validConfigData() -> Data {
        Data("""
        {"architectures":["Qwen3_5ForConditionalGeneration"],"model_type":"qwen3_5","text_config":{"hidden_size":5120,"intermediate_size":17408,"num_hidden_layers":64,"layer_types":["linear_attention","linear_attention","linear_attention","full_attention","linear_attention","linear_attention","linear_attention","full_attention","linear_attention","linear_attention","linear_attention","full_attention","linear_attention","linear_attention","linear_attention","full_attention","linear_attention","linear_attention","linear_attention","full_attention","linear_attention","linear_attention","linear_attention","full_attention","linear_attention","linear_attention","linear_attention","full_attention","linear_attention","linear_attention","linear_attention","full_attention","linear_attention","linear_attention","linear_attention","full_attention","linear_attention","linear_attention","linear_attention","full_attention","linear_attention","linear_attention","linear_attention","full_attention","linear_attention","linear_attention","linear_attention","full_attention","linear_attention","linear_attention","linear_attention","full_attention","linear_attention","linear_attention","linear_attention","full_attention","linear_attention","linear_attention","linear_attention","full_attention","linear_attention","linear_attention","linear_attention","full_attention"],"num_key_value_heads":4,"num_attention_heads":24,"head_dim":256,"vocab_size":32,"rms_norm_eps":0.000001}}
        """.utf8)
    }

    @Test("valid multishard BF16/F32 inventory and data reads")
    func validMultishardInventory() throws {
        let root = try temporaryDirectory(); defer { try? FileManager.default.removeItem(at: root) }
        let bf16 = Data(repeating: 0x11, count: 4)
        let f32 = Data(repeating: 0x22, count: 8)
        try writeShard(root.appendingPathComponent("model-00001.safetensors"), tensors: [("embed", "BF16", [2], bf16)])
        try writeShard(root.appendingPathComponent("model-00002.safetensors"), tensors: [("proj", "F32", [2], f32)])
        try writeIndex(root.appendingPathComponent("model.safetensors.index.json"), map: ["embed": "model-00001.safetensors", "proj": "model-00002.safetensors"], totalSize: 12, sidecarDType: "BF16")

        let file = try WeightFile(indexURL: root.appendingPathComponent("model.safetensors.index.json"))
        let first = try file.tensor("embed", expectedDType: .bf16, expectedShape: [2])
        let second = try file.tensor("proj", expectedDType: .f32, expectedShape: [2])
        #expect(try file.data(for: first) == bf16)
        #expect(try file.data(for: second) == f32)
        #expect(file.index.tensorCount == 2)
        #expect(file.index.totalSize == 12)
    }

    @Test("missing shard and incomplete index fail closed")
    func missingShardAndIncompleteSet() throws {
        let root = try temporaryDirectory(); defer { try? FileManager.default.removeItem(at: root) }
        try writeIndex(root.appendingPathComponent("index.json"), map: ["missing": "missing.safetensors"])
        #expect(throws: WeightFileError.self) { try WeightFile(indexURL: root.appendingPathComponent("index.json")) }
        let shard = root.appendingPathComponent("present.safetensors")
        try writeShard(shard, tensors: [("present", "F32", [1], Data(repeating: 0, count: 4))])
        try writeIndex(root.appendingPathComponent("index2.json"), map: ["present": "present.safetensors", "absent": "absent.safetensors"])
        #expect(throws: WeightFileError.self) { try WeightFile(indexURL: root.appendingPathComponent("index2.json")) }
    }

    @Test("malformed and truncated safetensors headers are rejected")
    func malformedHeaders() throws {
        let root = try temporaryDirectory(); defer { try? FileManager.default.removeItem(at: root) }
        let short = root.appendingPathComponent("short.safetensors")
        try Data([1, 2, 3]).write(to: short)
        #expect(throws: TensorInventoryError.self) { try WeightIndex.parseShard(at: short) }
        let truncated = root.appendingPathComponent("truncated.safetensors")
        var bytes = Data(); var length: UInt64 = 100
        withUnsafeBytes(of: &length) { bytes.append(contentsOf: $0) }; bytes.append(Data("{}".utf8))
        try bytes.write(to: truncated)
        #expect(throws: TensorInventoryError.self) { try WeightIndex.parseShard(at: truncated) }
        let badJSON = root.appendingPathComponent("bad-json.safetensors")
        var bad = Data(); length = 3
        withUnsafeBytes(of: &length) { bad.append(contentsOf: $0) }; bad.append(Data("xxx".utf8))
        try bad.write(to: badJSON)
        #expect(throws: TensorInventoryError.self) { try WeightIndex.parseShard(at: badJSON) }
    }

    @Test("invalid offsets and shapes are rejected")
    func invalidOffsetsAndShapes() throws {
        let root = try temporaryDirectory(); defer { try? FileManager.default.removeItem(at: root) }
        let invalidOffset = root.appendingPathComponent("offset.safetensors")
        let header = Data("{\"x\":{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[0,8]}}".utf8)
        try writeShard(invalidOffset, tensors: [], headerOverride: header)
        #expect(throws: TensorInventoryError.self) { try WeightIndex.parseShard(at: invalidOffset) }
        let invalidShape = root.appendingPathComponent("shape.safetensors")
        let shapeHeader = Data("{\"x\":{\"dtype\":\"F32\",\"shape\":[-1],\"data_offsets\":[0,0]}}".utf8)
        try writeShard(invalidShape, tensors: [], headerOverride: shapeHeader)
        #expect(throws: TensorInventoryError.self) { try WeightIndex.parseShard(at: invalidShape) }
    }

    @Test("config hash match is accepted and mismatch is rejected")
    func configHashValidation() throws {
        let root = try temporaryDirectory(); defer { try? FileManager.default.removeItem(at: root) }
        let config = try ModelConfig(data: validConfigData())
        try writeShard(root.appendingPathComponent("weights.safetensors"), tensors: [("x", "F32", [1], Data(repeating: 0, count: 4))])
        let index = root.appendingPathComponent("index.json")
        try writeIndex(index, map: ["x": "weights.safetensors"], configHash: config.configHash)
        _ = try WeightFile(indexURL: index, root: root, config: config)
        try writeIndex(index, map: ["x": "weights.safetensors"], configHash: String(repeating: "0", count: 64))
        #expect(throws: WeightFileError.self) { try WeightFile(indexURL: index, root: root, config: config) }
    }

    @Test("declared total and observed shard bytes remain separate")
    func declaredVersusObservedTotals() throws {
        let root = try temporaryDirectory(); defer { try? FileManager.default.removeItem(at: root) }
        let payload = Data(repeating: 7, count: 4)
        try writeShard(root.appendingPathComponent("w.safetensors"), tensors: [("x", "F32", [1], payload)])
        let indexURL = root.appendingPathComponent("index.json")
        try writeIndex(indexURL, map: ["x": "w.safetensors"], totalSize: 999)
        let index = try WeightIndex(url: indexURL)
        let spec = try WeightFile(indexURL: indexURL, root: root).tensor("x")
        #expect(index.totalSize == 999)
        #expect(spec.byteLength == 4)
        #expect(try WeightFile(indexURL: indexURL, root: root).data(for: spec).count == 4)
    }

    @Test("duplicate programmatic entries are rejected")
    func duplicateEntries() throws {
        let entries = [TensorEntry(name: "x", shard: "a.safetensors", shape: [1], dtype: "F32"), TensorEntry(name: "x", shard: "b.safetensors", shape: [1], dtype: "F32")]
        #expect(throws: WeightFileError.self) { try WeightIndex(entries: entries) }
    }

    @Test("sidecar dtype metadata is exposed for caller validation")
    func sidecarMismatchIsObservable() throws {
        let root = try temporaryDirectory(); defer { try? FileManager.default.removeItem(at: root) }
        try writeShard(root.appendingPathComponent("w.safetensors"), tensors: [("x", "F32", [1], Data(repeating: 0, count: 4))])
        let indexURL = root.appendingPathComponent("index.json")
        try writeIndex(indexURL, map: ["x": "w.safetensors"], sidecarDType: "BF16")
        let index = try WeightIndex(url: indexURL)
        #expect(index.sidecarDType == .bf16)
        #expect(index.entry("x")?.tensorDType == nil)
    }
}
