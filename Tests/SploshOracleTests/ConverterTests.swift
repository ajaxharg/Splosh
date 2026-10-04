import Foundation
import Testing
import SploshModel

@Suite("Synthetic q4 converter")
struct ConverterTests {
    private func temporaryDirectory() throws -> URL {
        let workspace = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
        let url = workspace.appendingPathComponent(".build/SploshConverter-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        return url
    }

    private func writeShard(_ url: URL, tensors: [(String, String, [Int], Data)], headerOverride: Data? = nil) throws {
        var header: [String: Any] = [:]; var cursor: Int64 = 0
        for (name, dtype, shape, bytes) in tensors {
            header[name] = ["dtype": dtype, "shape": shape, "data_offsets": [cursor, cursor + Int64(bytes.count)]]
            cursor += Int64(bytes.count)
        }
        let json = try headerOverride ?? JSONSerialization.data(withJSONObject: header, options: [.sortedKeys])
        var data = Data(); var length = UInt64(json.count)
        withUnsafeBytes(of: &length) { data.append(contentsOf: $0) }; data.append(json)
        for tensor in tensors { data.append(tensor.3) }
        try data.write(to: url)
    }

    private func writeIndex(_ url: URL, map: [String: String], sidecar: String = "BF16") throws {
        let object: [String: Any] = ["weight_map": map, "metadata": ["sidecar_dtype": sidecar]]
        try JSONSerialization.data(withJSONObject: object, options: [.sortedKeys]).write(to: url)
    }

    private func fixture() throws -> (URL, [Data]) {
        let root = try temporaryDirectory()
        let q4 = Data((0..<256).map { UInt8($0 & 255) }) // 2 x 32 packed U32 words
        let scales = Data(repeating: 0, count: 16) // [2, 4] BF16 group-64 scales
        let biases = Data(repeating: 1, count: 16) // [2, 4] BF16 group-64 biases
        let dense = Data((0..<20).map { UInt8($0) }) // [2, 5] dense BF16
        try writeShard(root.appendingPathComponent("model-00001.safetensors"), tensors: [
            ("layer.weight", "U32", [2, 32], q4),
            ("layer.scales", "BF16", [2, 4], scales),
            ("layer.biases", "BF16", [2, 4], biases)
        ])
        try writeShard(root.appendingPathComponent("model-00002.safetensors"), tensors: [("dense", "BF16", [2, 5], dense)])
        try writeIndex(root.appendingPathComponent("model.safetensors.index.json"), map: [
            "layer.weight": "model-00001.safetensors", "layer.scales": "model-00001.safetensors",
            "layer.biases": "model-00001.safetensors", "dense": "model-00002.safetensors"
        ])
        try Data("{}".utf8).write(to: root.appendingPathComponent("config.json"))
        try Data("{\"vocab\":[]}".utf8).write(to: root.appendingPathComponent("tokenizer.json"))
        return (root, [q4, scales, biases, dense])
    }

    private func recordBytes(_ url: URL, _ record: SPLWTensorRecord) throws -> Data {
        let data = try Data(contentsOf: url)
        return data.subdata(in: Int(record.payloadOffset)..<Int(record.payloadOffset + record.rawByteLength))
    }

    @Test("q4 and dense BF16 shards convert, reopen, verify, and round-trip exactly")
    func conversionRoundTrip() throws {
        let (root, sourceBytes) = try fixture(); defer { try? FileManager.default.removeItem(at: root) }
        let output = root.appendingPathComponent("weights.splw")
        let report = try Converter.convert(inputRoot: root, outputURL: output)
        #expect(report.header.dtype == "MIXED")
        #expect(report.header.tensorRecords.map(\.name) == report.header.tensorRecords.map(\.name).sorted())
        #expect(report.header.tensorRecords.allSatisfy { $0.payloadOffset % 128 == 0 && $0.rowStride % 128 == 0 })
        let file = try WeightFile(splwURL: output)
        let q4 = try file.q4("layer.weight")
        #expect(q4.physicalShape == [2, 32]); #expect(q4.logicalShape == [2, 256])
        #expect(try file.denseBF16("dense", expectedShape: [2, 5]).tensor.dtype == .bf16)
        _ = try Converter.verify(outputURL: output, sourceRoot: root)
        let records = file.header.tensorRecords
        let expected: [String: Data] = ["layer.weight": sourceBytes[0], "layer.scales": sourceBytes[1], "layer.biases": sourceBytes[2], "dense": sourceBytes[3]]
        for record in records { #expect(try recordBytes(output, record) == expected[record.name]) }
    }

    @Test("conversion records and offsets are deterministic")
    func deterministicRecords() throws {
        let (root, _) = try fixture(); defer { try? FileManager.default.removeItem(at: root) }
        let a = root.appendingPathComponent("a.splw"), b = root.appendingPathComponent("b.splw")
        _ = try Converter.convert(inputRoot: root, outputURL: a); _ = try Converter.convert(inputRoot: root, outputURL: b)
        #expect(try Data(contentsOf: a) == Data(contentsOf: b))
    }

    private func writeMalformed(_ url: URL, header: Data, payload: Data = Data()) throws {
        var bytes = Data("SPLW".utf8); var version: UInt32 = 1; var length = UInt64(header.count)
        withUnsafeBytes(of: &version) { bytes.append(contentsOf: $0) }; withUnsafeBytes(of: &length) { bytes.append(contentsOf: $0) }
        bytes.append(header); let pad = (128 - bytes.count % 128) % 128; bytes.append(Data(repeating: 0, count: pad)); bytes.append(payload)
        try bytes.write(to: url)
    }

    @Test("malformed SPLW header, framing, records, and payload fail closed")
    func malformedContainers() throws {
        let root = try temporaryDirectory(); defer { try? FileManager.default.removeItem(at: root) }
        let valid = ConverterHeader(dtype: "MIXED", alignment: ConverterAlignment(), configHash: "c", tokenizerHash: "t", tensorRecords: [], headerLength: 0, payloadLength: 0)
        let encoder = JSONEncoder(); encoder.outputFormatting = [.sortedKeys]
        let validData = try encoder.encode(valid)
        func expectInvalid(_ data: Data, payload: Data = Data()) throws { let u = root.appendingPathComponent(UUID().uuidString); try writeMalformed(u, header: data, payload: payload); #expect(throws: WeightFileError.self) { try WeightFile(splwURL: u) } }
        var badMagic = Data("NOPE".utf8); var v: UInt32 = 1; var l = UInt64(validData.count); withUnsafeBytes(of: &v) { badMagic.append(contentsOf: $0) }; withUnsafeBytes(of: &l) { badMagic.append(contentsOf: $0) }; badMagic.append(validData); try badMagic.write(to: root.appendingPathComponent("magic.splw")); #expect(throws: WeightFileError.self) { try WeightFile(splwURL: root.appendingPathComponent("magic.splw")) }
        var altered = validData; altered[0] = 0x7b; try expectInvalid(altered)
        var duplicate = SPLWTensorRecord(name: "x", dtype: .bf16, shape: [1], payloadOffset: 128, rawByteLength: 2, rowStride: 128, alignmentPadding: 0, sourceShard: "x")
        let dupHeader = ConverterHeader(dtype: "MIXED", alignment: ConverterAlignment(), configHash: "c", tokenizerHash: "t", tensorRecords: [duplicate, duplicate], headerLength: UInt64(0), payloadLength: 2)
        try expectInvalid(try encoder.encode(dupHeader), payload: Data(repeating: 0, count: 2))
        duplicate = SPLWTensorRecord(name: "x", dtype: .unknown, shape: [1], payloadOffset: 128, rawByteLength: 2, rowStride: 128, alignmentPadding: 0, sourceShard: "x")
        let unknown = ConverterHeader(dtype: "MIXED", alignment: ConverterAlignment(), configHash: "c", tokenizerHash: "t", tensorRecords: [duplicate], headerLength: 0, payloadLength: 2)
        try expectInvalid(try encoder.encode(unknown), payload: Data(repeating: 0, count: 2))
        let unsupported = ConverterHeader(dtype: "MIXED", alignment: ConverterAlignment(), configHash: "c", tokenizerHash: "t", tensorRecords: [], headerLength: 0, payloadLength: 0, quantization: ConverterQuantization(bits: 8, groupSize: 32, mode: "symmetric"))
        try expectInvalid(try encoder.encode(unsupported))
        try expectInvalid(Data("{}".utf8))
    }

    @Test("missing q4 index and alignment evidence fail closed")
    func missingInputsFailClosed() throws {
        let root = try temporaryDirectory(); defer { try? FileManager.default.removeItem(at: root) }
        #expect(throws: ConverterError.self) { try Converter.requireAssets(at: root) }
        #expect(throws: ConverterError.self) { try Converter.alignmentEvidence(from: root) }
    }

    @Test("missing hashes, altered sources, and failed conversion leave no output")
    func sourceFailures() throws {
        let (root, _) = try fixture(); defer { try? FileManager.default.removeItem(at: root) }
        let output = root.appendingPathComponent("failed.splw")
        let sentinel = Data("pre-existing-output".utf8)
        try sentinel.write(to: output)
        try FileManager.default.removeItem(at: root.appendingPathComponent("config.json"))
        #expect(throws: ConverterError.self) { try Converter.convert(inputRoot: root, outputURL: output) }
        #expect(FileManager.default.fileExists(atPath: output.path))
        #expect(try Data(contentsOf: output) == sentinel)
        let (root2, _) = try fixture(); defer { try? FileManager.default.removeItem(at: root2) }
        let out2 = root2.appendingPathComponent("changed.splw"); try Data("changed".utf8).write(to: root2.appendingPathComponent("config.json"))
        _ = try Converter.convert(inputRoot: root2, outputURL: out2); try Data("altered".utf8).write(to: root2.appendingPathComponent("config.json")); #expect(throws: WeightFileError.self) { try Converter.verify(outputURL: out2, sourceRoot: root2) }
    }
}
