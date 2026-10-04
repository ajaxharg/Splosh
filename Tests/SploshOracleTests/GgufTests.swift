import Foundation
import Testing
import SploshModel

// GGUF v3 reader, quant decoders and name mapping. The synthetic file is written below; the golden
// blocks are real blocks of Qwen3.8-27B-UD-Q4_K_M.gguf (GgufGoldenBlocks.swift); the last test reads
// that file itself when the Hugging Face cache has it.

private func le<T: FixedWidthInteger>(_ value: T) -> [UInt8] {
    withUnsafeBytes(of: value.littleEndian) { Array($0) }
}

private func ggufString(_ text: String) -> [UInt8] { le(UInt64(text.utf8.count)) + Array(text.utf8) }

private func roundUp(_ value: Int, to multiple: Int) -> Int { (value + multiple - 1) / multiple * multiple }

/// A GGUF v3 file of two tensors: `norm.weight`, five F32, and `blk.0.ssm_alpha.weight`, a Q8_0
/// matrix of 3 rows of 64, six blocks of scale 0.5 and codes 5 * block + element - 100.
private struct SyntheticGguf {
    var bytes: [UInt8]
    let headerEnd: Int
    let dataStart: Int
    /// Where the matrix starts in the data section.
    let matrixOffset: Int
    let norm: [UInt8]
    let matrix: [UInt8]

    init(magic: String = "GGUF", version: UInt32 = 3, alignment: UInt32? = 64, matrixType: UInt32 = 8, matrixInner: UInt64 = 64) {
        func pair(_ key: String, _ type: UInt32, _ value: [UInt8]) -> [UInt8] { ggufString(key) + le(type) + value }
        var pairs: [[UInt8]] = []
        pairs.append(pair("general.architecture", 8, ggufString("synthetic")))
        pairs.append(pair("synthetic.layers", 4, le(UInt32(7))))
        pairs.append(pair("synthetic.offset", 5, le(Int32(-3))))
        pairs.append(pair("synthetic.byte", 0, [200]))
        pairs.append(pair("synthetic.ratio", 6, le(Float(0.5).bitPattern)))
        pairs.append(pair("synthetic.scale", 12, le(Double(1.25).bitPattern)))
        pairs.append(pair("synthetic.flag", 7, [1]))
        pairs.append(pair("synthetic.seed", 10, le(UInt64.max)))
        var sections = le(UInt32(5)) + le(UInt64(4))
        for value in [1, 2, 3, 4] as [Int32] { sections += le(value) }
        pairs.append(pair("synthetic.sections", 9, sections))
        pairs.append(pair("synthetic.weights", 9, le(UInt32(6)) + le(UInt64(2)) + le(Float(0.5).bitPattern) + le(Float(-1.5).bitPattern)))
        var tokens = le(UInt32(8)) + le(UInt64(3))
        for token in ["a", "bb", "ccc"] { tokens += ggufString(token) }
        pairs.append(pair("synthetic.tokens", 9, tokens))
        pairs.append(pair("synthetic.long", 9, le(UInt32(0)) + le(UInt64(1000)) + [UInt8](repeating: 9, count: 1000)))
        if let alignment { pairs.append(pair("general.alignment", 4, le(alignment))) }
        let align = Int(alignment ?? 32)

        var header = Array(magic.utf8)
        header += le(version)
        header += le(UInt64(2))
        header += le(UInt64(pairs.count))
        for item in pairs { header += item }
        header += ggufString("norm.weight") + le(UInt32(1)) + le(UInt64(5)) + le(UInt32(0)) + le(UInt64(0))
        let matrixOffset = roundUp(20, to: align)
        header += ggufString("blk.0.ssm_alpha.weight") + le(UInt32(2)) + le(matrixInner) + le(UInt64(3)) + le(matrixType) + le(UInt64(matrixOffset))

        var norm: [UInt8] = []
        for value in [1.5, -2, 0.25, 8, -0.125] as [Float] { norm += le(value.bitPattern) }
        var matrix: [UInt8] = []
        for block in 0..<6 {
            matrix += le(Float16(0.5).bitPattern)
            matrix += (0..<32).map { UInt8(bitPattern: Int8(5 * block + $0 - 100)) }
        }
        let dataStart = roundUp(header.count, to: align)
        var bytes = header
        bytes += [UInt8](repeating: 0, count: dataStart - header.count)
        bytes += norm
        bytes += [UInt8](repeating: 0, count: matrixOffset - norm.count)
        bytes += matrix
        self.bytes = bytes; self.headerEnd = header.count; self.dataStart = dataStart
        self.matrixOffset = matrixOffset; self.norm = norm; self.matrix = matrix
    }
}

private func inScratchDirectory<T>(_ body: (URL) throws -> T) throws -> T {
    let directory = FileManager.default.temporaryDirectory.appendingPathComponent("SploshGguf-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: directory) }
    return try body(directory)
}

private func write(_ bytes: [UInt8], in directory: URL) throws -> URL {
    let url = directory.appendingPathComponent("synthetic.gguf")
    try Data(bytes).write(to: url)
    return url
}

private func decode(_ type: GgufTensorType, _ bytes: some Collection<UInt8>) -> [Float] {
    let blocks = Array(bytes)
    var out = [Float](repeating: 0, count: blocks.count / type.blockBytes * type.blockElements)
    blocks.withUnsafeBytes { raw in
        out.withUnsafeMutableBufferPointer { GgufTensorType.decode(type, blocks: raw, into: $0) }
    }
    return out
}

/// The real model, when this machine has it.
private let realGgufPath = NSHomeDirectory()
    + "/.cache/huggingface/hub/models--unsloth--Qwen3.8-27B-GGUF/snapshots/4ca720788d1e01f1bff70c033e0d0028fd02e502/Qwen3.8-27B-UD-Q4_K_M.gguf"

@Suite("GgufTests")
struct GgufTests {
    @Test("A synthetic file parses to its metadata, shapes, offsets and bytes")
    func syntheticFile() throws {
        let synthetic = SyntheticGguf()
        try inScratchDirectory { directory in
            let file = try GgufFile(url: try write(synthetic.bytes, in: directory))
            defer { withExtendedLifetime(file) {} }
            #expect(file.version == 3)
            #expect(file.fileSize == synthetic.bytes.count)
            #expect(file.alignment == 64)
            #expect(file.dataStart == synthetic.dataStart)
            #expect(file.dataStart % 64 == 0 && file.dataStart >= synthetic.headerEnd && file.dataStart - synthetic.headerEnd < 64)

            #expect(file.metadata.count == 13)
            #expect(file.metadata["general.architecture"] == .string("synthetic"))
            #expect(file.metadata["synthetic.layers"] == .integer(7))
            #expect(file.metadata["synthetic.offset"] == .integer(-3))
            #expect(file.metadata["synthetic.byte"] == .integer(200))
            #expect(file.metadata["synthetic.ratio"] == .float(0.5))
            #expect(file.metadata["synthetic.scale"] == .float(1.25))
            #expect(file.metadata["synthetic.flag"] == .bool(true))
            #expect(file.metadata["synthetic.seed"] == .unsigned(UInt64.max))
            #expect(file.metadata["general.alignment"]?.integerValue == 64)
            #expect(file.metadata["synthetic.sections"] == .integers([1, 2, 3, 4]))
            #expect(file.metadata["synthetic.weights"] == .floats([0.5, -1.5]))
            #expect(file.metadata["synthetic.tokens"] == .skippedArray(count: 3))
            #expect(file.metadata["synthetic.long"] == .skippedArray(count: 1000))
            #expect(file.metadata["synthetic.long"]?.integerValue == nil)

            #expect(file.tensors.map(\.name) == ["norm.weight", "blk.0.ssm_alpha.weight"])
            let norm = try file.tensor(named: "norm.weight")
            #expect(norm.type == .f32)
            #expect(norm.dims == [5] && norm.inner == 5 && norm.rows == 1 && norm.elementCount == 5)
            #expect(norm.offset == synthetic.dataStart && norm.byteLength == 20)
            let matrix = try file.tensor(named: "blk.0.ssm_alpha.weight")
            #expect(matrix.type == .q8_0)
            #expect(matrix.dims == [64, 3] && matrix.inner == 64 && matrix.rows == 3 && matrix.elementCount == 192)
            #expect(matrix.rowBytes == 68 && matrix.byteLength == 204)
            #expect(matrix.offset == synthetic.dataStart + synthetic.matrixOffset)
            #expect(file.tensors == [norm, matrix])

            #expect(Array(file.bytes(of: norm)) == synthetic.norm)
            #expect(Array(try file.bytes(named: "blk.0.ssm_alpha.weight")) == synthetic.matrix)
            #expect(Array(try file.bytes(of: matrix, rows: 1..<3)) == Array(synthetic.matrix[68..<204]))
            #expect(Array(try file.bytes(of: matrix, rows: 0..<1)) == Array(synthetic.matrix[0..<68]))
            #expect(try file.bytes(of: matrix, rows: 3..<3).isEmpty)
            #expect(throws: GgufError.rowsOutOfRange(tensor: "blk.0.ssm_alpha.weight", rows: 2..<4, count: 3)) {
                try file.bytes(of: matrix, rows: 2..<4)
            }
            #expect(throws: GgufError.unknownTensor("nope")) { try file.tensor(named: "nope") }

            // The tensor bytes are what the decoders read: scale 0.5 and codes 5 * block + element - 100.
            let values = decode(.q8_0, file.bytes(of: matrix))
            #expect(values.count == 192)
            #expect((0..<192).allSatisfy { values[$0] == 0.5 * Float(5 * ($0 / 32) + $0 % 32 - 100) })
            #expect(decode(.f32, file.bytes(of: norm)) == [1.5, -2, 0.25, 8, -0.125])
        }
    }

    @Test("Without general.alignment the alignment is 32")
    func defaultAlignment() throws {
        let synthetic = SyntheticGguf(alignment: nil)
        try inScratchDirectory { directory in
            let file = try GgufFile(url: try write(synthetic.bytes, in: directory))
            defer { withExtendedLifetime(file) {} }
            #expect(file.alignment == 32)
            #expect(file.metadata.count == 12 && file.metadata["general.alignment"] == nil)
            #expect(file.dataStart == synthetic.dataStart && file.dataStart % 32 == 0)
            #expect(synthetic.matrixOffset == 32)
            let matrix = try file.tensor(named: "blk.0.ssm_alpha.weight")
            #expect(matrix.offset == synthetic.dataStart + 32)
            #expect(Array(file.bytes(of: matrix)) == synthetic.matrix)
        }
    }

    @Test("Bad magic, an unsupported version, truncation and bad tensors throw GgufError")
    func malformedFiles() throws {
        try inScratchDirectory { directory in
            #expect(throws: GgufError.badMagic) { try GgufFile(url: try write(SyntheticGguf(magic: "GGML").bytes, in: directory)) }
            #expect(throws: GgufError.unsupportedVersion(2)) { try GgufFile(url: try write(SyntheticGguf(version: 2).bytes, in: directory)) }
            #expect(throws: GgufError.unsupportedType(tensor: "blk.0.ssm_alpha.weight", id: 2)) {
                try GgufFile(url: try write(SyntheticGguf(matrixType: 2).bytes, in: directory))
            }
            #expect { try GgufFile(url: try write(SyntheticGguf(matrixInner: 40).bytes, in: directory)) } throws: { error in
                if case GgufError.malformedHeader = error { return true } else { return false }
            }
            #expect { try GgufFile(url: directory.appendingPathComponent("missing.gguf")) } throws: { error in
                if case GgufError.unreadable = error { return true } else { return false }
            }

            // Cut anywhere inside the header, and the header is truncated; an empty file included.
            let synthetic = SyntheticGguf()
            var wrong: [Int] = []
            for cut in 0..<synthetic.headerEnd {
                let url = try write(Array(synthetic.bytes[0..<cut]), in: directory)
                do { _ = try GgufFile(url: url); wrong.append(cut) } catch GgufError.truncatedHeader {} catch { wrong.append(cut) }
            }
            #expect(wrong.isEmpty)

            // A whole header with no data, and data that stops inside the second tensor.
            #expect(throws: GgufError.tensorPastEnd(tensor: "norm.weight", end: UInt64(synthetic.dataStart + 20), fileSize: UInt64(synthetic.headerEnd))) {
                try GgufFile(url: try write(Array(synthetic.bytes[0..<synthetic.headerEnd]), in: directory))
            }
            let cut = synthetic.dataStart + synthetic.matrixOffset + 100
            #expect(throws: GgufError.tensorPastEnd(tensor: "blk.0.ssm_alpha.weight", end: UInt64(synthetic.bytes.count), fileSize: UInt64(cut))) {
                try GgufFile(url: try write(Array(synthetic.bytes[0..<cut]), in: directory))
            }
        }
        #expect(GgufError.badMagic.description.contains("magic"))
        #expect(GgufError.unknownTensor("blk.9.nothing").description.contains("blk.9.nothing"))
        #expect(GgufError.unsupportedVersion(7).description.contains("7"))
    }

    @Test("Block geometry and ggml type ids")
    func geometry() {
        let expected: [(GgufTensorType, UInt32, Int, Int)] = [
            (.f32, 0, 1, 4), (.f16, 1, 1, 2), (.q8_0, 8, 32, 34), (.q3K, 11, 256, 110), (.q4K, 12, 256, 144),
            (.q5K, 13, 256, 176), (.q6K, 14, 256, 210), (.iq4NL, 20, 32, 18), (.iq3S, 21, 256, 110),
            (.iq4XS, 23, 256, 136), (.bf16, 30, 1, 2),
        ]
        #expect(GgufTensorType.allCases.count == expected.count)
        for (type, id, elements, bytes) in expected {
            #expect(type.rawValue == id && GgufTensorType(rawValue: id) == type)
            #expect(type.blockElements == elements && type.blockBytes == bytes)
        }
        #expect(GgufTensorType(rawValue: 2) == nil)
        #expect(GgufTensorType.iq4nlValues.count == 16 && GgufTensorType.iq4nlValues.sorted() == GgufTensorType.iq4nlValues)
        #expect(GgufTensorType.iq3sGrid.count == 512)
        // Every byte of every grid entry is an odd magnitude from 1 to 15.
        #expect(GgufTensorType.iq3sGrid.allSatisfy { entry in (0..<4).allSatisfy { let m = (entry >> UInt32(8 * $0)) & 0xFF; return m % 2 == 1 && m <= 15 } })
    }

    @Test("A real block of each quantised format decodes to the reference values", arguments: GgufGoldenBlocks.all)
    func goldenBlock(_ golden: GgufGoldenBlock) {
        #expect(golden.bytes.count == golden.type.blockBytes)
        #expect(golden.values.count == golden.type.blockElements)
        let decoded = decode(golden.type, golden.bytes)
        #expect(decoded.count == golden.values.count)
        let wrong = zip(decoded, golden.values).enumerated().filter { abs($0.element.0 - $0.element.1) > 1e-6 * abs($0.element.1) }
        #expect(wrong.isEmpty, "\(golden.type): \(wrong.count) of \(decoded.count) values differ, the first at \(wrong.first?.offset ?? -1)")
        // Whole blocks are decoded one after the other, each to its own slice.
        #expect(decode(golden.type, golden.bytes + golden.bytes) == decoded + decoded)
    }

    @Test("The float types decode to the value they hold")
    func floatTypes() {
        #expect(decode(.f32, le(Float(-3.25).bitPattern) + le(Float(0.5).bitPattern)) == [-3.25, 0.5])
        #expect(decode(.f16, le(Float16(1.5).bitPattern) + le(Float16(-0.25).bitPattern)) == [1.5, -0.25])
        #expect(decode(.bf16, le(UInt16(0x3FC0)) + le(UInt16(0xC120))) == [1.5, -10])
    }

    @Test("GGUF tensor names map to Splosh names")
    func nameMapping() {
        let expected: [String: String?] = [
            "token_embd.weight": "language_model.model.embed_tokens",
            "output.weight": "language_model.lm_head",
            "output_norm.weight": "language_model.model.norm",
            "blk.0.ffn_gate.weight": "language_model.model.layers.0.mlp.gate_proj",
            "blk.0.ffn_up.weight": "language_model.model.layers.0.mlp.up_proj",
            "blk.63.ffn_down.weight": "language_model.model.layers.63.mlp.down_proj",
            "blk.12.attn_qkv.weight": "language_model.model.layers.12.linear_attn.in_proj_qkv",
            "blk.12.attn_gate.weight": "language_model.model.layers.12.linear_attn.in_proj_z",
            "blk.12.ssm_alpha.weight": "language_model.model.layers.12.linear_attn.in_proj_a",
            "blk.12.ssm_beta.weight": "language_model.model.layers.12.linear_attn.in_proj_b",
            "blk.12.ssm_out.weight": "language_model.model.layers.12.linear_attn.out_proj",
            "blk.12.ssm_norm.weight": "language_model.model.layers.12.linear_attn.norm",
            "blk.12.ssm_conv1d.weight": "language_model.model.layers.12.linear_attn.conv1d",
            "blk.12.ssm_a": "language_model.model.layers.12.linear_attn.A_log",
            "blk.12.ssm_dt.bias": "language_model.model.layers.12.linear_attn.dt_bias",
            "blk.3.attn_q.weight": "language_model.model.layers.3.self_attn.q_proj",
            "blk.3.attn_k.weight": "language_model.model.layers.3.self_attn.k_proj",
            "blk.3.attn_v.weight": "language_model.model.layers.3.self_attn.v_proj",
            "blk.3.attn_output.weight": "language_model.model.layers.3.self_attn.o_proj",
            "blk.3.attn_q_norm.weight": "language_model.model.layers.3.self_attn.q_norm",
            "blk.3.attn_k_norm.weight": "language_model.model.layers.3.self_attn.k_norm",
            "blk.3.attn_norm.weight": "language_model.model.layers.3.input_layernorm",
            "blk.3.post_attention_norm.weight": "language_model.model.layers.3.post_attention_layernorm",
            "blk.64.nextn.eh_proj.weight": nil,
            "blk.3.ffn_up.bias": nil,
            "blk.x.ffn_up.weight": nil,
            "blk.3": nil,
            "rope_freqs.weight": nil,
        ]
        for (gguf, splosh) in expected { #expect(GgufNames.sploshName(forGguf: gguf) == splosh, "\(gguf)") }
        #expect(Set(GgufNames.blockTensors.values).count == GgufNames.blockTensors.count)
        #expect(GgufNames.blockIndex(of: "blk.12.ssm_a") == 12 && GgufNames.blockIndex(of: "output.weight") == nil)
    }

    @Test("The MTP block is the last block when a tensor is a nextn tensor")
    func mtpBlock() {
        let names = ["token_embd.weight", "blk.0.attn_norm.weight", "blk.9.ffn_up.weight", "blk.10.attn_norm.weight",
                     "blk.10.nextn.eh_proj.weight", "output.weight"]
        #expect(GgufNames.mtpBlock(among: names) == 10)
        #expect(GgufNames.belongsToMtp("blk.10.attn_norm.weight", mtpBlock: 10))
        #expect(GgufNames.belongsToMtp("blk.10.nextn.eh_proj.weight", mtpBlock: 10))
        #expect(!GgufNames.belongsToMtp("blk.9.ffn_up.weight", mtpBlock: 10))
        #expect(!GgufNames.belongsToMtp("blk.1.ffn_up.weight", mtpBlock: 10))
        #expect(!GgufNames.belongsToMtp("output.weight", mtpBlock: 10))
        // Without a nextn tensor there is no MTP block.
        #expect(GgufNames.mtpBlock(among: names.filter { !$0.contains(".nextn.") }) == nil)
        #expect(!GgufNames.belongsToMtp("blk.10.attn_norm.weight", mtpBlock: nil))
        #expect(GgufNames.mtpBlock(among: [String]()) == nil)
    }

    @Test("The value-head permutation is a bijection on 48 heads")
    func valueHeadPermutation() {
        let heads = (0..<48).map { GgufNames.mlxValueHead(ofGgufHead: $0) }
        #expect(Set(heads) == Set(0..<48))
        #expect(GgufNames.mlxValueHead(ofGgufHead: 0) == 0)
        #expect(GgufNames.mlxValueHead(ofGgufHead: 1) == 3)
        #expect(GgufNames.mlxValueHead(ofGgufHead: 16) == 1)
        #expect(GgufNames.mlxValueHead(ofGgufHead: 32) == 2)
        #expect(GgufNames.mlxValueHead(ofGgufHead: 47) == 47)
    }

    @Test("The real Qwen3.8-27B-UD-Q4_K_M file",
          .enabled(if: FileManager.default.fileExists(atPath: realGgufPath), "needs the Hugging Face cache copy of Qwen3.8-27B-UD-Q4_K_M.gguf"))
    func realFile() throws {
        let file = try GgufFile(url: URL(fileURLWithPath: realGgufPath))
        defer { withExtendedLifetime(file) {} }
        #expect(file.version == 3 && file.alignment == 32)
        #expect(file.metadata["general.architecture"] == .string("qwen35"))
        #expect(file.metadata["qwen35.block_count"]?.integerValue == 65)
        #expect(file.metadata["qwen35.rope.dimension_sections"] == .integers([11, 11, 10, 0]))
        #expect(file.metadata["tokenizer.ggml.tokens"] == .skippedArray(count: 248320))
        #expect(file.tensors.count == 866)

        let embedding = try file.tensor(named: "token_embd.weight")
        #expect(embedding.type == .q4K && embedding.rows == 248320 && embedding.inner == 5120 && embedding.dims == [5120, 248320])
        #expect(try file.tensor(named: "blk.0.ssm_alpha.weight").type == .q8_0)
        #expect(try file.tensor(named: "blk.0.ssm_alpha.weight").rows == 48)
        #expect(try file.tensor(named: "output_norm.weight").type == .f32)

        // Every tensor lies in the data section, inside the file and on an alignment boundary, and
        // no two overlap.
        for tensor in file.tensors {
            #expect(tensor.offset >= file.dataStart && tensor.offset % file.alignment == 0, "\(tensor.name)")
            #expect(tensor.byteLength <= file.fileSize - tensor.offset, "\(tensor.name)")
            #expect(tensor.byteLength == tensor.elementCount / tensor.type.blockElements * tensor.type.blockBytes, "\(tensor.name)")
        }
        let byOffset = file.tensors.sorted { $0.offset < $1.offset }
        for (current, next) in zip(byOffset, byOffset.dropFirst()) {
            #expect(current.offset + current.byteLength <= next.offset, "\(current.name) overlaps \(next.name)")
        }

        // The golden blocks are the first blocks of these tensors.
        for golden in GgufGoldenBlocks.all {
            let tensor = try file.tensor(named: golden.tensor)
            #expect(tensor.type == golden.type, "\(golden.tensor)")
            #expect(Array(file.bytes(of: tensor).prefix(golden.type.blockBytes)) == golden.bytes, "\(golden.tensor)")
        }
        // A row of the embedding is 20 blocks; its first 256 values are the golden block's.
        let row = decode(.q4K, try file.bytes(of: embedding, rows: 0..<1))
        #expect(row.count == 5120 && row.allSatisfy(\.isFinite))
        #expect(Array(row.prefix(256)) == GgufGoldenBlocks.q4K.values)
        #expect(try file.bytes(of: embedding, rows: 248319..<248320).count == embedding.rowBytes)

        // Every tensor outside the MTP block has a Splosh name, each its own; in the MTP block only
        // the nextn tensors have none.
        let mtp = GgufNames.mtpBlock(among: file.tensors.map(\.name))
        #expect(mtp == 64)
        var seen = Set<String>()
        for tensor in file.tensors {
            let splosh = GgufNames.sploshName(forGguf: tensor.name)
            if GgufNames.belongsToMtp(tensor.name, mtpBlock: mtp) {
                #expect((splosh == nil) == tensor.name.contains(".nextn."), "\(tensor.name)")
            } else {
                #expect(splosh != nil && seen.insert(splosh ?? "").inserted, "\(tensor.name)")
            }
        }
    }
}
