import CryptoKit
import Foundation
import Metal
import Testing

import SploshCore
import SploshModel

// The GGUF converter (Sources/SploshModel/GgufConvert.swift) against the file it converts.
//
// A converted weight is read back from the artifact a tile at a time, its planes unpacked to
// native blocks (GgufPlanes.unpack) and each row held against the row of the source that belongs
// there: the bytes, and the reference decoder's values bit for bit. Which row belongs where is
// stated here on its own, as the inverse of the documented order of the value heads: the file's
// head i is the engine's head 3 (i % 16) + i / 16, so the engine's head m is the file's
// 16 (m % 3) + m / 3. A dense tensor is held against the source's fp32 values, rounded here by
// comparing an fp32's two BF16 neighbours.
//
// Three files are converted: a synthetic one with every format and every kind of head order, two
// blocks of the real UD-Q5_K_M when the Hugging Face cache has it, and, when .build/gguf has the
// whole of it converted, that artifact is checked against the names and shapes the model loads
// and sampled against the source, a tile a weight. A fourth, small, is mapped as the loader
// maps an artifact and read by the GGUF kernels.
//
// The tests run one at a time: each spreads its work over every core, and a debug build takes
// some hundreds of nanoseconds an element to repack, unpack or decode.

private func le<T: FixedWidthInteger>(_ value: T) -> [UInt8] {
    withUnsafeBytes(of: value.littleEndian) { Array($0) }
}

private func ggufString(_ text: String) -> [UInt8] { le(UInt64(text.utf8.count)) + Array(text.utf8) }

private func roundUp(_ value: Int, to multiple: Int) -> Int { (value + multiple - 1) / multiple * multiple }

private func sha256(_ bytes: some DataProtocol) -> String {
    SHA256.hash(data: bytes).map { String(format: "%02x", $0) }.joined()
}

private struct Generator {
    var state: UInt64
    mutating func next() -> UInt64 {
        state &+= 0x9E37_79B9_7F4A_7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        return z ^ (z >> 31)
    }
    /// Uniform in [-1, 1).
    mutating func unit() -> Float { Float(Int64(next() >> 40) - (1 << 23)) / Float(1 << 23) }
}

/// One tensor of a synthetic file.
private struct SourceTensor {
    let name: String
    /// The extents as the file stores them, innermost first.
    let dims: [Int]
    /// The ggml type id: a `GgufTensorType`'s, or one the reader does not know.
    let typeID: UInt32
    let bytes: [UInt8]

    init(_ name: String, _ dims: [Int], _ type: GgufTensorType, _ bytes: [UInt8]) {
        self.name = name; self.dims = dims; typeID = type.rawValue; self.bytes = bytes
    }

    init(_ name: String, _ dims: [Int], typeID: UInt32, _ bytes: [UInt8]) {
        self.name = name; self.dims = dims; self.typeID = typeID; self.bytes = bytes
    }
}

/// A GGUF v3 file of these tensors, aligned to 32.
private func ggufFile(_ tensors: [SourceTensor]) -> [UInt8] {
    var header = Array("GGUF".utf8) + le(UInt32(3)) + le(UInt64(tensors.count)) + le(UInt64(2))
    header += ggufString("general.architecture") + le(UInt32(8)) + ggufString("synthetic")
    header += ggufString("general.alignment") + le(UInt32(4)) + le(UInt32(32))
    var offset = 0
    for tensor in tensors {
        header += ggufString(tensor.name) + le(UInt32(tensor.dims.count))
        for extent in tensor.dims { header += le(UInt64(extent)) }
        header += le(tensor.typeID) + le(UInt64(offset))
        offset = roundUp(offset + tensor.bytes.count, to: 32)
    }
    var bytes = header + [UInt8](repeating: 0, count: roundUp(header.count, to: 32) - header.count)
    for tensor in tensors {
        bytes += tensor.bytes
        bytes += [UInt8](repeating: 0, count: roundUp(tensor.bytes.count, to: 32) - tensor.bytes.count)
    }
    return bytes
}

/// Where a native block keeps its fp16 scale fields.
private func scaleOffsets(of type: GgufTensorType) -> [Int] {
    switch type {
    case .q4K, .q5K: return [0, 2]
    case .q6K: return [208]
    case .q3K: return [108]
    case .q8_0, .iq4NL, .iq4XS, .iq3S: return [0]
    case .f32, .f16, .bf16: return []
    }
}

/// Rows of native blocks of random bytes, each scale field a finite fp16 of either sign.
private func nativeRows(_ type: GgufTensorType, rows: Int, inner: Int, seed: UInt64) -> [UInt8] {
    var random = Generator(state: seed)
    let size = type.blockBytes, blocks = rows * inner / type.blockElements
    var bytes = [UInt8](repeating: 0, count: blocks * size)
    bytes.withUnsafeMutableBytes { raw in
        var offset = 0
        while offset + 8 <= raw.count {
            raw.storeBytes(of: random.next(), toByteOffset: offset, as: UInt64.self)
            offset += 8
        }
        while offset < raw.count {
            raw[offset] = UInt8(truncatingIfNeeded: random.next())
            offset += 1
        }
        for block in 0..<blocks {
            for field in scaleOffsets(of: type) {
                let scale = Float16(0.002 * random.unit())
                raw.storeBytes(of: scale.bitPattern.littleEndian, toByteOffset: block * size + field, as: UInt16.self)
            }
        }
    }
    return bytes
}

private func floatBytes(_ values: [Float]) -> [UInt8] {
    var bytes: [UInt8] = []
    bytes.reserveCapacity(4 * values.count)
    for value in values { bytes += le(value.bitPattern) }
    return bytes
}

/// fp32 values that round to BF16 every way: exact ones, ties above an even and above an odd
/// BF16, the fp32 either side of a tie, and random ones.
private func denseValues(_ count: Int, seed: UInt64) -> [Float] {
    var random = Generator(state: seed)
    let crafted: [UInt32] = [0x3F80_0000, 0x3F81_8000, 0x3F82_8000, 0x3F81_7FFF, 0x3F81_8001, 0xBF83_8000, 0xBF84_8000, 0x0000_0000, 0x8000_0000, 0x3F80_FFFF]
    return (0..<count).map { index in
        index < crafted.count ? Float(bitPattern: crafted[index]) : 1 + random.unit()
    }
}

/// The BF16 nearest an fp32, a tie going to the even one: by comparing its two neighbours.
private func nearestBF16(_ value: Float) -> UInt16 {
    let below = value.bitPattern & 0xFFFF_0000, above = below &+ 0x1_0000
    let x = Double(value), toBelow = abs(x - Double(Float(bitPattern: below))), toAbove = abs(Double(Float(bitPattern: above)) - x)
    if toBelow != toAbove { return UInt16((toBelow < toAbove ? below : above) >> 16) }
    return UInt16(((below >> 16) & 1 == 0 ? below : above) >> 16)
}

private func bf16(_ bits: UInt16) -> Float { Float(bitPattern: UInt32(bits) << 16) }

/// Where a tensor of the artifact has the 48 value heads of a gated-delta layer: `span` units a
/// head from unit `first`, a unit being a row, a channel or an element. nil for a tensor that
/// has none, or has them as columns (`out_proj`, which is not re-ordered).
private func headOrder(ofArtifactTensor name: String) -> (first: Int, span: Int)? {
    for suffix in ["linear_attn.in_proj_a.weight", "linear_attn.in_proj_b.weight", "linear_attn.A_log", "linear_attn.dt_bias"] where name.hasSuffix(suffix) {
        return (0, 1)
    }
    if name.hasSuffix("linear_attn.in_proj_z.weight") { return (0, 128) }
    if name.hasSuffix("linear_attn.in_proj_qkv.weight") || name.hasSuffix("linear_attn.conv1d.weight") { return (4096, 128) }
    return nil
}

/// The unit of the file that the artifact has at `unit`.
private func sourceUnit(ofArtifactUnit unit: Int, _ order: (first: Int, span: Int)?) -> Int {
    guard let order, unit >= order.first else { return unit }
    let head = (unit - order.first) / order.span
    return order.first + (16 * (head % 3) + head / 3) * order.span + (unit - order.first) % order.span
}

/// What a check of one weight found.
private struct WeightCheck {
    var tiles = 0
    /// Rows whose unpacked native bytes were held against the source row's, and those of them
    /// whose decoded values were too.
    var rows = 0, decoded = 0
    /// Rows whose bytes, or whose decoded values, are not the source row's.
    var differentBytes = 0, differentValues = 0
    /// Rows past the weight's own that are not zero blocks, and tiles that could not be read.
    var paddingNotZero = 0, unreadable = 0

    var clean: Bool { differentBytes == 0 && differentValues == 0 && paddingNotZero == 0 && unreadable == 0 }

    static func + (a: WeightCheck, b: WeightCheck) -> WeightCheck {
        WeightCheck(tiles: a.tiles + b.tiles, rows: a.rows + b.rows, decoded: a.decoded + b.decoded,
                    differentBytes: a.differentBytes + b.differentBytes, differentValues: a.differentValues + b.differentValues,
                    paddingNotZero: a.paddingNotZero + b.paddingNotZero, unreadable: a.unreadable + b.unreadable)
    }
}

/// Unpack one tile of a converted weight and hold each of its rows against the source row that
/// the documented head order puts there: the native bytes of every row, and of every
/// `decodeEvery`'th the reference decoder's values. A tile is 128 rows and is read on its own:
/// the planes keep a tile's records together.
private func check(tile: Int, of record: GgufWeightRecord, in artifact: WeightFile, against tensor: GgufFile.Tensor,
                   of source: GgufFile, decodeEvery: Int) -> WeightCheck {
    let type = record.type, rows = record.logicalShape[0], inner = record.logicalShape[1]
    var result = WeightCheck(tiles: 1)
    guard let sizes = GgufPlanes.sizes(of: type, rows: GgufPlanes.tileRows, inner: inner) else {
        result.unreadable = 1
        return result
    }
    func plane(_ spec: TensorSpec?, _ bytes: Int) -> [UInt8]? {
        guard let spec else { return bytes == 0 ? [] : nil }
        return (try? artifact.readRange(offset: UInt64(spec.byteOffset ?? 0) + UInt64(tile * bytes), length: UInt64(bytes))).map { [UInt8]($0) }
    }
    guard let plane0 = plane(record.plane0, sizes.plane0), let plane1 = plane(record.plane1, sizes.plane1),
          let meta = plane(record.meta, sizes.meta) else {
        result.unreadable = 1
        return result
    }
    let rowBytes = inner / type.blockElements * type.blockBytes
    let order = headOrder(ofArtifactTensor: record.name)
    let native = source.bytes(of: tensor)
    let unpacked = GgufPlanes.unpack(type, planes: GgufPlanes.Planes(plane0: plane0, plane1: plane1, meta: meta),
                                     rows: GgufPlanes.tileRows, inner: inner)
    var mine = [Float](repeating: 0, count: inner), theirs = [Float](repeating: 0, count: inner)
    unpacked.withUnsafeBytes { unpacked in
        for index in 0..<GgufPlanes.tileRows {
            let row = tile * GgufPlanes.tileRows + index
            let converted = UnsafeRawBufferPointer(rebasing: unpacked[index * rowBytes ..< (index + 1) * rowBytes])
            guard row < rows else {
                if converted.contains(where: { $0 != 0 }) { result.paddingNotZero += 1 }
                continue
            }
            let from = sourceUnit(ofArtifactUnit: row, order)
            let original = UnsafeRawBufferPointer(rebasing: native[from * rowBytes ..< (from + 1) * rowBytes])
            result.rows += 1
            if memcmp(converted.baseAddress!, original.baseAddress!, rowBytes) != 0 { result.differentBytes += 1 }
            guard index % decodeEvery == 0 else { continue }
            result.decoded += 1
            mine.withUnsafeMutableBufferPointer { GgufTensorType.decode(type, blocks: converted, into: $0) }
            theirs.withUnsafeMutableBufferPointer { GgufTensorType.decode(type, blocks: original, into: $0) }
            if memcmp(mine, theirs, inner * MemoryLayout<Float>.size) != 0 { result.differentValues += 1 }
        }
    }
    return result
}

/// The values of a dense tensor of the artifact that are not what the source's give: the
/// source's value at the place the head order puts there, rounded to BF16; for `A_log`, the
/// BF16 nearest the logarithm of the source's negated value, which is what exponentiating back
/// to it within BF16 rounding means.
private func differences(inDense name: String, shape: [Int], of artifact: WeightFile, against tensor: GgufFile.Tensor,
                         of source: GgufFile) throws -> Int {
    let record = try artifact.denseBF16(name, expectedShape: shape)
    let stored = try artifact.data(for: record.tensor)
    var values = [Float](repeating: 0, count: tensor.elementCount)
    values.withUnsafeMutableBufferPointer { GgufTensorType.decode(tensor.type, blocks: source.bytes(of: tensor), into: $0) }
    guard stored.count == 2 * values.count else { return values.count }
    let order = headOrder(ofArtifactTensor: name)
    // A unit of the convolution is a channel's taps.
    let unit = shape.count > 1 ? shape.dropFirst().reduce(1, *) : 1
    var different = 0
    for index in values.indices {
        let bits = UInt16(stored[stored.startIndex + 2 * index]) | UInt16(stored[stored.startIndex + 2 * index + 1]) << 8
        let value = values[sourceUnit(ofArtifactUnit: index / unit, order) * unit + index % unit]
        if name.hasSuffix(".A_log") {
            // Half a unit in the last place of the BF16 that was stored, with room for the
            // rounding of an fp32 logarithm.
            let held = Double(bf16(bits)), wanted = log(-Double(value))
            let halfUnit = Double(bf16(bits).ulp) * 32768
            if !(abs(held - wanted) <= halfUnit * 1.0001) { different += 1 }
        } else if bits != nearestBF16(value) {
            different += 1
        }
    }
    return different
}

/// The artifact's name for each tensor of a source that the model has: the name mapping of
/// GgufNames.swift, with the `.weight` the file's name carries.
private func artifactNames(of source: GgufFile) -> [String: GgufFile.Tensor] {
    let mtp = GgufNames.mtpBlock(among: source.tensors.map(\.name))
    var names: [String: GgufFile.Tensor] = [:]
    for tensor in source.tensors where !GgufNames.belongsToMtp(tensor.name, mtpBlock: mtp) {
        guard let base = GgufNames.sploshName(forGguf: tensor.name) else { continue }
        names[base + (tensor.name.hasSuffix(".weight") ? ".weight" : "")] = tensor
    }
    return names
}

/// The shape a dense tensor has in the artifact: the file's extents outermost first, and the
/// convolution with its trailing 1.
private func denseShape(_ name: String, _ tensor: GgufFile.Tensor) -> [Int] {
    name.hasSuffix("conv1d.weight") ? Array(tensor.dims.reversed()) + [1] : Array(tensor.dims.reversed())
}

/// Hold every record of an artifact against the source: each packed weight at the tiles
/// `tiles` picks of its count, each dense tensor whole. Returns what was checked, by the type in
/// the source; under a float type, `rows` counts dense tensors.
private func checkAll(_ artifact: WeightFile, against source: GgufFile, decodeEvery: Int = 1,
                      tiles: (Int) -> [Int]) throws -> [GgufTensorType: WeightCheck] {
    let names = artifactNames(of: source)
    var byType: [GgufTensorType: WeightCheck] = [:]
    var planes = 0
    var weights: [(record: GgufWeightRecord, tensor: GgufFile.Tensor)] = []
    var jobs: [(weight: Int, tile: Int)] = []
    for record in artifact.header.tensorRecords {
        if record.name.hasSuffix(".plane1") || record.name.hasSuffix(".meta") { planes += 1; continue }
        let tensor = try #require(names[record.name], "\(record.name) is no tensor of the source")
        if record.dtype == .bf16 {
            let different = try differences(inDense: record.name, shape: denseShape(record.name, tensor), of: artifact, against: tensor, of: source)
            #expect(different == 0, "\(record.name): \(different) of \(tensor.elementCount) values are not the source's")
            byType[tensor.type, default: WeightCheck()].rows += 1
            continue
        }
        let weight = try artifact.gguf(record.name)
        #expect(weight.type == tensor.type && weight.logicalShape == [tensor.rows, tensor.inner], "\(record.name) is \(weight.type) \(weight.logicalShape)")
        planes -= weight.plane1 == nil ? 1 : 2
        for tile in tiles(weight.storedRows / GgufPlanes.tileRows) { jobs.append((weights.count, tile)) }
        weights.append((weight, tensor))
    }
    // Every plane 1 and meta record belongs to a weight that was resolved.
    #expect(planes == 0, "\(planes) plane records are of no weight")

    // The tiles of every weight as one set of tasks: a tile is the unit of work, whichever
    // weight it is of.
    var results = [WeightCheck](repeating: WeightCheck(), count: jobs.count)
    results.withUnsafeMutableBufferPointer { slots in
        nonisolated(unsafe) let slots = slots
        DispatchQueue.concurrentPerform(iterations: jobs.count) { job in
            let weight = weights[jobs[job].weight]
            slots[job] = check(tile: jobs[job].tile, of: weight.record, in: artifact, against: weight.tensor, of: source, decodeEvery: decodeEvery)
        }
    }
    var perWeight = [WeightCheck](repeating: WeightCheck(), count: weights.count)
    for (job, result) in zip(jobs, results) { perWeight[job.weight] = perWeight[job.weight] + result }
    for (weight, result) in zip(weights, perWeight) {
        #expect(result.clean, "\(weight.record.name) (\(weight.tensor.name), \(weight.tensor.type)): \(result)")
        #expect(result.rows > 0 && result.decoded > 0, "\(weight.record.name): no row was checked")
        byType[weight.tensor.type, default: WeightCheck()] = byType[weight.tensor.type, default: WeightCheck()] + result
    }
    return byType
}

private func summary(_ checked: [GgufTensorType: WeightCheck]) -> String {
    checked.sorted { $0.key.rawValue < $1.key.rawValue }.map { type, result in
        GgufPlanes.geometry(of: type) == nil ? "\(result.rows) dense \(type)"
            : "\(type) \(result.rows) rows in \(result.tiles) tiles (\(result.decoded) decoded)"
    }.joined(separator: ", ")
}

private func inScratchDirectory<T>(_ body: (URL) throws -> T) throws -> T {
    let directory = FileManager.default.temporaryDirectory.appendingPathComponent("SploshGgufConvert-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: directory) }
    return try body(directory)
}

private let cache = NSHomeDirectory() + "/.cache/huggingface/hub/models--unsloth--Qwen3.8-27B-GGUF/snapshots/4ca720788d1e01f1bff70c033e0d0028fd02e502/"
/// The real model, when this machine has it, and the whole of it converted, when that was done.
private let realGgufPath = cache + "Qwen3.8-27B-UD-Q5_K_M.gguf"
private let realArtifactPath = FileManager.default.currentDirectoryPath + "/.build/gguf/ud-q5_k_m.splw"
private let tokenizerPath = FileManager.default.currentDirectoryPath + "/inputs/tokenizer/tokenizer.json"

@Suite("GgufConvertTests", .serialized)
struct GgufConvertTests {
    private static let layer0 = "language_model.model.layers.0.", layer3 = "language_model.model.layers.3."

    /// The synthetic file: the tensors of a gated-delta layer whose heads are re-ordered, in each
    /// way they are, the out projection whose are not, three weights of full-attention layers,
    /// and dense tensors of every shape, all at the model's sizes. The eight formats each occur.
    /// The last two are an MTP block, which is left out. `artifact` is nil for those.
    private static let synthetic: [(gguf: String, artifact: String?, dims: [Int], type: GgufTensorType)] = [
        ("blk.0.ssm_alpha.weight", layer0 + "linear_attn.in_proj_a.weight", [5120, 48], .q8_0),
        ("blk.0.ssm_beta.weight", layer0 + "linear_attn.in_proj_b.weight", [5120, 48], .iq4NL),
        ("blk.0.attn_gate.weight", layer0 + "linear_attn.in_proj_z.weight", [5120, 6144], .q4K),
        ("blk.0.attn_qkv.weight", layer0 + "linear_attn.in_proj_qkv.weight", [5120, 10240], .iq4XS),
        ("blk.0.ssm_out.weight", layer0 + "linear_attn.out_proj.weight", [6144, 5120], .q3K),
        ("blk.0.ssm_conv1d.weight", layer0 + "linear_attn.conv1d.weight", [4, 10240], .f32),
        ("blk.0.ssm_a", layer0 + "linear_attn.A_log", [48], .f32),
        ("blk.0.ssm_dt.bias", layer0 + "linear_attn.dt_bias", [48], .f32),
        ("blk.0.ssm_norm.weight", layer0 + "linear_attn.norm.weight", [128], .f32),
        ("blk.0.attn_norm.weight", layer0 + "input_layernorm.weight", [5120], .f32),
        ("blk.0.post_attention_norm.weight", layer0 + "post_attention_layernorm.weight", [5120], .f32),
        ("blk.3.attn_k.weight", layer3 + "self_attn.k_proj.weight", [5120, 1024], .q5K),
        ("blk.3.attn_v.weight", layer3 + "self_attn.v_proj.weight", [5120, 1024], .q6K),
        ("blk.7.attn_k.weight", "language_model.model.layers.7.self_attn.k_proj.weight", [5120, 1024], .iq3S),
        ("blk.3.attn_q_norm.weight", layer3 + "self_attn.q_norm.weight", [256], .f32),
        ("blk.3.attn_k_norm.weight", layer3 + "self_attn.k_norm.weight", [256], .f16),
        ("output_norm.weight", "language_model.model.norm.weight", [5120], .f32),
        ("blk.8.nextn.hnorm.weight", nil, [5120], .f32),
        ("blk.8.attn_norm.weight", nil, [5120], .f32),
    ]

    private static func syntheticTensors() -> [SourceTensor] {
        synthetic.enumerated().map { index, item in
            let seed = 0xC0DE + UInt64(index)
            let count = item.dims.reduce(1, *)
            switch item.type {
            case .f32:
                var values = denseValues(count, seed: seed)
                // The decay is -exp of something: negative.
                if item.gguf.hasSuffix("ssm_a") {
                    var random = Generator(state: seed)
                    values = (0..<count).map { _ in -exp(4 * random.unit() - 2) }
                }
                return SourceTensor(item.gguf, item.dims, item.type, floatBytes(values))
            case .f16:
                var bytes: [UInt8] = []
                for value in denseValues(count, seed: seed) { bytes += le(Float16(value).bitPattern) }
                return SourceTensor(item.gguf, item.dims, item.type, bytes)
            default:
                return SourceTensor(item.gguf, item.dims, item.type, nativeRows(item.type, rows: item.dims[1], inner: item.dims[0], seed: seed))
            }
        }
    }

    @Test("The head order here inverts the one GgufNames documents")
    func headOrderInverse() {
        for head in 0..<48 {
            #expect(sourceUnit(ofArtifactUnit: GgufNames.mlxValueHead(ofGgufHead: head), (0, 1)) == head)
            #expect(sourceUnit(ofArtifactUnit: 4096 + GgufNames.mlxValueHead(ofGgufHead: head) * 128 + 5, (4096, 128)) == 4096 + head * 128 + 5)
        }
        #expect(sourceUnit(ofArtifactUnit: 4095, (4096, 128)) == 4095)
        #expect(sourceUnit(ofArtifactUnit: 77, nil) == 77)
    }

    @Test("A synthetic file converts to planes and BF16 that decode to the source's values in the engine's order")
    func syntheticFile() throws {
        try inScratchDirectory { directory in
            let bytes = ggufFile(Self.syntheticTensors())
            let input = directory.appendingPathComponent("synthetic.gguf"), output = directory.appendingPathComponent("out/synthetic.splw")
            let tokenizer = directory.appendingPathComponent("tokenizer.json")
            try Data(bytes).write(to: input)
            try Data("{\"model\":{}}".utf8).write(to: tokenizer)
            let source = try GgufFile(url: input)

            let report = try Converter.convertGguf(inputURL: input, outputURL: output, tokenizerURL: tokenizer)
            let expected = Self.synthetic.filter { $0.artifact != nil }
            #expect(report.tensorCount == expected.count)
            #expect(report.mtpTensors == 2 && report.filteredTensors == 0)
            for type in GgufTensorType.allCases where type != .bf16 {
                #expect((report.tensorsByType[type] ?? 0) == expected.filter { $0.type == type }.count, "\(type)")
            }
            #expect(report.byteCount == UInt64(try Data(contentsOf: output, options: .alwaysMapped).count))
            #expect(report.bytesByType.values.reduce(0, +) == report.header.tensorRecords.map(\.rawByteLength).reduce(0, +))
            // Nothing is left beside the artifact.
            #expect(try FileManager.default.contentsOfDirectory(atPath: output.deletingLastPathComponent().path) == ["synthetic.splw"])

            let artifact = try WeightFile(splwURL: output)
            let header = artifact.header
            #expect(artifact.isGguf && header.format == "gguf.tiled128" && header.quantization == .gguf && header.dtype == "MIXED")
            #expect(header.configHash == sha256(bytes[..<source.dataStart]))
            #expect(header.tokenizerHash == sha256(Data("{\"model\":{}}".utf8)))
            #expect(header.tensorRecords.map(\.name) == header.tensorRecords.map(\.name).sorted())
            #expect(header.tensorRecords.allSatisfy { $0.payloadOffset % 128 == 0 && $0.sourceShard == "synthetic.gguf" })

            // The records are these and no others: a dense tensor is one, a weight is its
            // planes, the second only where the format has one.
            var names: Set<String> = []
            for item in expected {
                let name = try #require(item.artifact)
                names.insert(name)
                guard let geometry = GgufPlanes.geometry(of: item.type) else {
                    let record = try #require(header.tensorRecords.first { $0.name == name })
                    #expect(record.dtype == .bf16 && record.ggufType == nil && record.logicalShape == nil)
                    continue
                }
                let base = String(name.dropLast(".weight".count))
                names.insert(base + ".meta")
                if geometry.plane1Bytes > 0 { names.insert(base + ".plane1") }
                let record = try #require(header.tensorRecords.first { $0.name == name })
                let rows = item.dims[1], inner = item.dims[0], stored = (rows + 127) / 128 * 128
                #expect(record.dtype == .u8 && record.ggufType == item.type.rawValue && record.logicalShape == [rows, inner])
                #expect(record.shape == [stored, inner / 32 * geometry.plane0Bytes])
                let weight = try artifact.gguf(name)
                #expect(weight.type == item.type && weight.storedRows == stored && (weight.plane1 == nil) == (geometry.plane1Bytes == 0))
                // The planes are not an affine triplet, and say so.
                #expect(throws: (any Error).self) { try artifact.q4(name) }
            }
            #expect(Set(header.tensorRecords.map(\.name)) == names)
            for item in expected where GgufPlanes.geometry(of: item.type) == nil {
                let name = try #require(item.artifact)
                let shape = name.hasSuffix("conv1d.weight") ? [10240, 4, 1] : item.dims
                #expect(try artifact.denseBF16(name, expectedShape: shape).tensor.shape == shape)
            }

            // Every tile of every weight, and every dense value, against the source.
            let checked = try checkAll(artifact, against: source) { Array(0..<$0) }
            for item in expected where GgufPlanes.geometry(of: item.type) != nil {
                #expect(checked[item.type]?.rows == item.dims[1] && checked[item.type]?.decoded == item.dims[1], "\(item.type): every row of \(item.gguf)")
            }
            // The 48-row projections are a tile each, 80 rows of it padding.
            #expect(checked[.q8_0]?.tiles == 1 && checked[.iq4NL]?.tiles == 1)
            print("gguf convert: synthetic file, \(report.recordCount) records, \(report.byteCount) bytes: " + summary(checked))

            // The model needs far more than this: the header-only resolution says what is missing.
            #expect(throws: (any Error).self) { try ModelWeights.recordsUsed(in: artifact) }

            // The filter picks tensors by their names in the file, and the same input gives the
            // same bytes.
            let small = directory.appendingPathComponent("small.splw"), again = directory.appendingPathComponent("again.splw")
            let pick = { (name: String) in name.contains("ssm_") && !name.contains("ssm_out") }
            let filtered = try Converter.convertGguf(inputURL: input, outputURL: small, tokenizerURL: tokenizer, include: pick)
            #expect(filtered.tensorCount == 6 && filtered.filteredTensors == Self.synthetic.count - 6 && filtered.mtpTensors == 0)
            _ = try Converter.convertGguf(inputURL: input, outputURL: again, tokenizerURL: tokenizer, include: pick)
            #expect(try Data(contentsOf: small) == Data(contentsOf: again))
            let smallChecked = try checkAll(try WeightFile(splwURL: small), against: source) { Array(0..<$0) }
            #expect(smallChecked[.q8_0]?.decoded == 48 && smallChecked[.iq4NL]?.decoded == 48 && smallChecked[.f32]?.rows == 4)
        }
    }

    @Test("A converted weight, mapped and resolved as the loader does, is what its format's kernel reads")
    func residentWeights() throws {
        // A weight of each format, small ones: the two 48-row projections, whose rows are
        // re-ordered and padded, and key and value projections of full-attention layers.
        let weights: [(gguf: String, artifact: String, rows: Int, type: GgufTensorType)] = [
            ("blk.0.ssm_alpha.weight", Self.layer0 + "linear_attn.in_proj_a.weight", 48, .q8_0),
            ("blk.0.ssm_beta.weight", Self.layer0 + "linear_attn.in_proj_b.weight", 48, .iq4NL),
            ("blk.3.attn_k.weight", Self.layer3 + "self_attn.k_proj.weight", 1024, .q5K),
            ("blk.3.attn_v.weight", Self.layer3 + "self_attn.v_proj.weight", 1024, .q6K),
            ("blk.7.attn_k.weight", "language_model.model.layers.7.self_attn.k_proj.weight", 1024, .iq3S),
            ("blk.7.attn_v.weight", "language_model.model.layers.7.self_attn.v_proj.weight", 1024, .q4K),
            ("blk.11.attn_k.weight", "language_model.model.layers.11.self_attn.k_proj.weight", 1024, .iq4XS),
            ("blk.11.attn_v.weight", "language_model.model.layers.11.self_attn.v_proj.weight", 1024, .q3K),
        ]
        let inner = 5120
        let device = try #require(MTLCreateSystemDefaultDevice())
        let library = try Metallib(device: device)
        let queue = try #require(device.makeCommandQueue())
        try inScratchDirectory { directory in
            let input = directory.appendingPathComponent("weights.gguf"), output = directory.appendingPathComponent("weights.splw")
            let tokenizer = directory.appendingPathComponent("tokenizer.json")
            try Data(ggufFile(weights.enumerated().map { index, item in
                SourceTensor(item.gguf, [inner, item.rows], item.type, nativeRows(item.type, rows: item.rows, inner: inner, seed: 0x7E57 + UInt64(index)))
            })).write(to: input)
            try Data("{}".utf8).write(to: tokenizer)
            _ = try Converter.convertGguf(inputURL: input, outputURL: output, tokenizerURL: tokenizer)
            let source = try GgufFile(url: input)

            // The mapping ModelWeights makes: every record, from the first payload byte on.
            let artifact = try WeightFile(splwURL: output)
            let records = artifact.header.tensorRecords.map {
                ResidentTensorRecord(name: $0.name, payloadOffset: Int($0.payloadOffset), byteLength: Int($0.rawByteLength), shape: $0.shape)
            }
            let first = try #require(records.map(\.payloadOffset).min())
            let resident = try ResidentWeights(device: device, records: records, containerURL: output, payloadStart: first,
                                               payloadLength: Int(artifact.fileSize) - first)
            struct EmbedParams { var rows, strideWords, groupsPerRow, hidden: UInt32 }
            for item in weights {
                let record = try artifact.gguf(item.artifact)
                let geometry = try #require(GgufPlanes.geometry(of: record.type))
                let handle = try resident.gguf(item.artifact, kernel: geometry.kernelSuffix, rows: item.rows, inner: inner, secondPlane: record.plane1 != nil)
                #expect(handle.kernel == geometry.kernelSuffix && handle.bits == 0 && handle.rows == item.rows && handle.inner == inner)
                #expect(handle.groupsPerRow == inner / 32 && handle.rowStrideWords * 4 == inner / 32 * geometry.plane0Bytes)
                let stored = record.storedRows
                #expect(handle.scales.byteLength == stored * inner / 32 * geometry.plane1Bytes)

                // The embedding kernel is a row decoder: rows from both ends, the middle and,
                // for a padded weight, past its own rows.
                let rows = [0, 1, item.rows / 3, item.rows / 2 + 5, item.rows - 2, item.rows - 1, stored - 1].map(UInt32.init)
                let ids = try #require(rows.withUnsafeBytes { device.makeBuffer(bytes: $0.baseAddress!, length: $0.count, options: .storageModeShared) })
                let out = try #require(device.makeBuffer(length: rows.count * inner * MemoryLayout<Float>.size, options: .storageModeShared))
                let pipeline = try library.pipeline("sp_gguf_embed_" + geometry.kernelSuffix)
                let command = try #require(queue.makeCommandBuffer())
                let encoder = try #require(command.makeComputeCommandEncoder())
                var p = EmbedParams(rows: UInt32(rows.count), strideWords: UInt32(handle.rowStrideWords),
                                    groupsPerRow: UInt32(handle.groupsPerRow), hidden: UInt32(inner))
                encoder.setComputePipelineState(pipeline)
                // The three planes where the engine binds a weight's packed, scales and biases.
                encoder.setBuffer(resident.buffer(handle.packed), offset: handle.packed.offset, index: 0)
                encoder.setBuffer(resident.buffer(handle.scales), offset: handle.scales.offset, index: 1)
                encoder.setBuffer(resident.buffer(handle.biases), offset: handle.biases.offset, index: 2)
                encoder.setBuffer(ids, offset: 0, index: 3)
                encoder.setBuffer(out, offset: 0, index: 4)
                encoder.setBytes(&p, length: MemoryLayout<EmbedParams>.stride, index: 5)
                encoder.dispatchThreads(MTLSize(width: handle.groupsPerRow, height: rows.count, depth: 1),
                                        threadsPerThreadgroup: MTLSize(width: pipeline.threadExecutionWidth, height: 1, depth: 1))
                encoder.endEncoding()
                command.commit()
                command.waitUntilCompleted()
                try #require(command.status == .completed, "\(item.gguf): \(String(describing: command.error))")

                let tensor = try source.tensor(named: item.gguf)
                let order = headOrder(ofArtifactTensor: item.artifact)
                let result = UnsafeBufferPointer(start: out.contents().bindMemory(to: Float.self, capacity: rows.count * inner), count: rows.count * inner)
                var expected = [Float](repeating: 0, count: inner)
                var different = 0
                // A row of padding is zero blocks, which decode to zeros of the sign the format
                // gives them: IQ4_NL's first codebook value is negative.
                let padding = [UInt8](repeating: 0, count: tensor.rowBytes)
                for (index, row) in rows.enumerated() {
                    if Int(row) < item.rows {
                        let from = sourceUnit(ofArtifactUnit: Int(row), order)
                        let blocks = try source.bytes(of: tensor, rows: from ..< from + 1)
                        expected.withUnsafeMutableBufferPointer { GgufTensorType.decode(tensor.type, blocks: blocks, into: $0) }
                    } else {
                        padding.withUnsafeBytes { blocks in
                            expected.withUnsafeMutableBufferPointer { GgufTensorType.decode(tensor.type, blocks: blocks, into: $0) }
                        }
                        #expect(expected.allSatisfy { $0 == 0 })
                    }
                    for k in 0..<inner where result[index * inner + k].bitPattern != expected[k].bitPattern { different += 1 }
                }
                #expect(different == 0, "\(item.gguf) (\(item.type)): \(different) of \(rows.count * inner) values read by its kernel are not the source's")
            }
        }
    }

    @Test("A tensor the model cannot take is refused by name, and nothing is written")
    func refusals() throws {
        try inScratchDirectory { directory in
            let tokenizer = directory.appendingPathComponent("tokenizer.json")
            try Data("{}".utf8).write(to: tokenizer)
            let output = directory.appendingPathComponent("out.splw")
            let sentinel = Data("what was here before".utf8)
            try sentinel.write(to: output)
            let norm = SourceTensor("blk.0.ssm_norm.weight", [128], .f32, floatBytes(denseValues(128, seed: 1)))

            /// The error a file of these tensors is refused with; nil when it converts.
            func refusal(_ tensors: [SourceTensor], include: ((String) -> Bool)? = nil) throws -> String? {
                let input = directory.appendingPathComponent("refused.gguf")
                try Data(ggufFile(tensors)).write(to: input)
                do {
                    _ = try Converter.convertGguf(inputURL: input, outputURL: output, tokenizerURL: tokenizer, include: include)
                    return nil
                } catch {
                    return String(describing: error)
                }
            }
            let zeros = { (count: Int) in [UInt8](repeating: 0, count: count) }
            let cases: [(what: String, name: String, tensor: SourceTensor)] = [
                ("no counterpart", "blk.0.mystery.weight", SourceTensor("blk.0.mystery.weight", [4], .f32, zeros(16))),
                ("no counterpart outside the blocks", "rope_freqs.weight", SourceTensor("rope_freqs.weight", [4], .f32, zeros(16))),
                ("the wrong shape", "blk.0.ssm_dt.bias", SourceTensor("blk.0.ssm_dt.bias", [64], .f32, zeros(256))),
                ("a weight of the wrong shape", "blk.3.attn_k.weight", SourceTensor("blk.3.attn_k.weight", [5120, 512], .q8_0, zeros(512 * 160 * 34))),
                ("a dense tensor of a quantised type", "blk.0.attn_norm.weight", SourceTensor("blk.0.attn_norm.weight", [5120], .q8_0, zeros(160 * 34))),
                ("a weight of a float type", "blk.0.ssm_alpha.weight", SourceTensor("blk.0.ssm_alpha.weight", [5120, 48], .f16, zeros(48 * 5120 * 2))),
                ("an attention tensor in a gated-delta layer", "blk.0.attn_q_norm.weight", SourceTensor("blk.0.attn_q_norm.weight", [256], .f32, zeros(1024))),
                ("a gated-delta tensor in an attention layer", "blk.3.ssm_norm.weight", SourceTensor("blk.3.ssm_norm.weight", [128], .f32, zeros(512))),
                ("a block past the model's layers", "blk.64.attn_norm.weight", SourceTensor("blk.64.attn_norm.weight", [5120], .f32, zeros(20480))),
                ("a decay that is not negative", "blk.0.ssm_a", SourceTensor("blk.0.ssm_a", [48], .f32, zeros(192))),
                ("a type the reader does not know", "blk.0.ssm_beta.weight", SourceTensor("blk.0.ssm_beta.weight", [5120, 48], typeID: 99, zeros(64))),
            ]
            for item in cases {
                let message = try refusal([norm, item.tensor])
                #expect(message?.contains(item.name) == true, "\(item.what): \(message ?? "converted")")
            }
            // A filter that leaves the tensor out lets the rest convert; one that leaves
            // nothing does not.
            #expect(try refusal([norm, cases[0].tensor], include: { _ in false }) != nil)
            #expect(try Data(contentsOf: output) == sentinel)
            #expect(try FileManager.default.contentsOfDirectory(atPath: directory.path).sorted() == ["out.splw", "refused.gguf", "tokenizer.json"])
            #expect(try refusal([norm, cases[0].tensor], include: { $0 == norm.name }) == nil)
            #expect(try WeightFile(splwURL: output).header.tensorRecords.map(\.name) == [Self.layer0 + "linear_attn.norm.weight"])
            // No tokenizer, no artifact.
            #expect(throws: ConverterError.self) {
                try Converter.convertGguf(inputURL: directory.appendingPathComponent("refused.gguf"), outputURL: output,
                                          tokenizerURL: directory.appendingPathComponent("absent.json")) { $0 == norm.name }
            }
        }
    }

    @Test("Two blocks of the real UD-Q5_K_M convert to its values in the engine's order",
          .enabled(if: FileManager.default.fileExists(atPath: realGgufPath), "needs the Hugging Face cache copy of Qwen3.8-27B-UD-Q5_K_M.gguf"))
    func realBlocks() throws {
        try inScratchDirectory { directory in
            let input = URL(fileURLWithPath: realGgufPath), output = directory.appendingPathComponent("blocks.splw")
            var tokenizer = URL(fileURLWithPath: tokenizerPath)
            if !FileManager.default.fileExists(atPath: tokenizer.path) {
                tokenizer = directory.appendingPathComponent("tokenizer.json")
                try Data("{}".utf8).write(to: tokenizer)
            }
            let source = try GgufFile(url: input)
            let wanted = source.tensors.filter { $0.name.hasPrefix("blk.0.") || $0.name.hasPrefix("blk.3.") }
            // A gated-delta layer has fourteen tensors and a full-attention layer eleven.
            #expect(wanted.count == 25)

            let started = Date()
            let report = try Converter.convertGguf(inputURL: input, outputURL: output, tokenizerURL: tokenizer) {
                $0.hasPrefix("blk.0.") || $0.hasPrefix("blk.3.")
            }
            let seconds = Date().timeIntervalSince(started)
            #expect(report.tensorCount == wanted.count && report.filteredTensors == source.tensors.count - wanted.count && report.mtpTensors == 0)

            let artifact = try WeightFile(splwURL: output)
            #expect(artifact.isGguf && artifact.fileSize == report.byteCount)
            #expect(artifact.header.tokenizerHash == sha256(try Data(contentsOf: tokenizer)))
            let names = artifactNames(of: source)
            let converted = Set(artifact.header.tensorRecords.map(\.name).filter { !$0.hasSuffix(".plane1") && !$0.hasSuffix(".meta") })
            #expect(converted == Set(names.filter { $0.value.name.hasPrefix("blk.0.") || $0.value.name.hasPrefix("blk.3.") }.keys))

            // Every tile of every weight and every dense value; `A_log` among them, held to the
            // BF16 nearest log(-ssm_a).
            let checked = try checkAll(artifact, against: source) { Array(0..<$0) }
            let rows = Dictionary(grouping: wanted.filter { GgufPlanes.geometry(of: $0.type) != nil }, by: \.type).mapValues { $0.map(\.rows).reduce(0, +) }
            for (type, count) in rows { #expect(checked[type]?.decoded == count, "\(type): \(checked[type]?.decoded ?? 0) of \(count) rows decoded") }
            #expect(try artifact.denseBF16(Self.layer0 + "linear_attn.A_log", expectedShape: [48]).tensor.byteLength == 96)
            print("gguf convert: blk.0 and blk.3 of UD-Q5_K_M, \(report.recordCount) records, \(report.byteCount) bytes in "
                  + String(format: "%.1f s: ", seconds) + summary(checked))
        }
    }

    /// The names and shapes ModelWeights.swift resolves, written out from the geometry: the
    /// packed weights by base name as [rows, inner], the dense tensors by name.
    private static func modelTensors() -> (packed: [String: [Int]], dense: [String: [Int]]) {
        let g = ModelGeometry()
        var packed: [String: [Int]] = [:], dense: [String: [Int]] = [:]
        for layer in 0..<g.layers {
            let p = "language_model.model.layers.\(layer)."
            dense[p + "input_layernorm.weight"] = [g.hidden]
            dense[p + "post_attention_layernorm.weight"] = [g.hidden]
            packed[p + "mlp.gate_proj"] = [g.intermediate, g.hidden]
            packed[p + "mlp.up_proj"] = [g.intermediate, g.hidden]
            packed[p + "mlp.down_proj"] = [g.hidden, g.intermediate]
            if layer % 4 == 3 {
                packed[p + "self_attn.q_proj"] = [g.heads * g.headDim * 2, g.hidden]
                packed[p + "self_attn.k_proj"] = [g.kvHeads * g.headDim, g.hidden]
                packed[p + "self_attn.v_proj"] = [g.kvHeads * g.headDim, g.hidden]
                packed[p + "self_attn.o_proj"] = [g.hidden, g.heads * g.headDim]
                dense[p + "self_attn.q_norm.weight"] = [g.headDim]
                dense[p + "self_attn.k_norm.weight"] = [g.headDim]
            } else {
                packed[p + "linear_attn.in_proj_qkv"] = [g.gdnChannels, g.hidden]
                packed[p + "linear_attn.in_proj_a"] = [g.gdnValueHeads, g.hidden]
                packed[p + "linear_attn.in_proj_b"] = [g.gdnValueHeads, g.hidden]
                packed[p + "linear_attn.in_proj_z"] = [g.gdnValueDim, g.hidden]
                packed[p + "linear_attn.out_proj"] = [g.hidden, g.gdnValueDim]
                dense[p + "linear_attn.conv1d.weight"] = [g.gdnChannels, 4, 1]
                dense[p + "linear_attn.A_log"] = [g.gdnValueHeads]
                dense[p + "linear_attn.dt_bias"] = [g.gdnValueHeads]
                dense[p + "linear_attn.norm.weight"] = [g.gdnHeadDim]
            }
        }
        packed["language_model.model.embed_tokens"] = [g.vocab, g.hidden]
        packed["language_model.lm_head"] = [g.vocab, g.hidden]
        dense["language_model.model.norm.weight"] = [g.hidden]
        return (packed, dense)
    }

    @Test("The whole UD-Q5_K_M as converted has every tensor the model loads, and the source's values",
          .enabled(if: FileManager.default.fileExists(atPath: realArtifactPath) && FileManager.default.fileExists(atPath: realGgufPath),
                   "needs .build/gguf/ud-q5_k_m.splw, converted from the Hugging Face cache copy of Qwen3.8-27B-UD-Q5_K_M.gguf"))
    func realArtifact() throws {
        // Only the header and the sampled tiles are read: the artifact is not mapped, and
        // ModelWeights is not opened on it.
        let artifact = try WeightFile(splwURL: URL(fileURLWithPath: realArtifactPath))
        let source = try GgufFile(url: URL(fileURLWithPath: realGgufPath))
        let header = artifact.header
        #expect(artifact.isGguf && header.quantization == .gguf)
        let headerBytes = try #require(try FileHandle(forReadingFrom: source.url).read(upToCount: source.dataStart))
        #expect(headerBytes.count == source.dataStart && header.configHash == sha256(headerBytes))
        if FileManager.default.fileExists(atPath: tokenizerPath) {
            #expect(header.tokenizerHash == sha256(try Data(contentsOf: URL(fileURLWithPath: tokenizerPath))))
        }

        // The record set is exactly what the model looks up, name by name and shape by shape.
        let model = Self.modelTensors()
        var names: Set<String> = []
        var formats: [GgufTensorType: Int] = [:]
        for (base, shape) in model.packed {
            let weight = try artifact.gguf(base + ".weight")
            #expect(weight.logicalShape == shape, "\(base) is \(weight.logicalShape), the model's is \(shape)")
            names.formUnion([base + ".weight", base + ".meta"])
            if weight.plane1 != nil { names.insert(base + ".plane1") }
            formats[weight.type, default: 0] += 1
        }
        for (name, shape) in model.dense {
            #expect(try artifact.denseBF16(name, expectedShape: shape).tensor.shape == shape)
            names.insert(name)
        }
        #expect(Set(header.tensorRecords.map(\.name)) == names)
        // And the loader's own resolution, run on the header, finds them all and uses them all.
        #expect(try ModelWeights.recordsUsed(in: artifact) == names)

        // The formats are the source's, tensor for tensor.
        let mtp = GgufNames.mtpBlock(among: source.tensors.map(\.name))
        let sourceFormats = Dictionary(grouping: source.tensors.filter { !GgufNames.belongsToMtp($0.name, mtpBlock: mtp) && GgufPlanes.geometry(of: $0.type) != nil }, by: \.type).mapValues(\.count)
        #expect(formats == sourceFormats)

        // The middle tile of every weight, which is of a head the order moves (it leaves the
        // first and the last in place): the bytes of all its rows, the decoded values of one row
        // in sixteen. And every dense tensor whole.
        let checked = try checkAll(artifact, against: source, decodeEvery: 16) { tiles in [tiles / 2] }
        for type in sourceFormats.keys { #expect((checked[type]?.decoded ?? 0) > 0, "no row of a \(type) weight was decoded") }
        print("gguf convert: \(realArtifactPath), \(header.tensorRecords.count) records, \(artifact.fileSize) bytes, "
              + formats.sorted { $0.key.rawValue < $1.key.rawValue }.map { "\($0.value) \($0.key)" }.joined(separator: " ") + "; checked " + summary(checked))
    }
}
