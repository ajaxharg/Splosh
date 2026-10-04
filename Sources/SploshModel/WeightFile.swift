// WeightFile.swift — strict safetensors and SPLW container reader.
import Foundation

public enum WeightFileError: Error, Equatable, CustomStringConvertible {
    case missing(URL)
    case invalid(String)
    case configHashMismatch(expected: String, observed: String)
    public var description: String {
        switch self { case .missing(let u): return "weight file missing: \(u.path)"; case .invalid(let s): return "weight file invalid: \(s)"; case .configHashMismatch(let e, let o): return "weight file config hash mismatch: expected \(e), observed \(o)" }
    }
}

/// A validated MLX affine q4 tensor triplet in an SPLW container.
public struct Q4WeightRecord: Sendable, Equatable {
    public let name: String
    public let weight: TensorSpec
    public let scales: TensorSpec
    public let biases: TensorSpec
    public let physicalShape: [Int]
    public let logicalShape: [Int]

    public init(name: String, weight: TensorSpec, scales: TensorSpec, biases: TensorSpec,
                physicalShape: [Int], logicalShape: [Int]) {
        self.name = name; self.weight = weight; self.scales = scales; self.biases = biases
        self.physicalShape = physicalShape; self.logicalShape = logicalShape
    }
}

/// A GGUF weight's planes (GgufPlanes.swift) in an SPLW container.
public struct GgufWeightRecord: Sendable, Equatable {
    public let name: String
    public let type: GgufTensorType
    public let plane0: TensorSpec
    /// nil for a format with no second plane.
    public let plane1: TensorSpec?
    public let meta: TensorSpec
    /// The scalar matrix, `[rows, inner]`.
    public let logicalShape: [Int]
    /// The rows the planes hold: `rows` rounded up to whole tiles, the rows past it zero blocks.
    public let storedRows: Int

    public init(name: String, type: GgufTensorType, plane0: TensorSpec, plane1: TensorSpec?, meta: TensorSpec,
                logicalShape: [Int], storedRows: Int) {
        self.name = name; self.type = type; self.plane0 = plane0; self.plane1 = plane1; self.meta = meta
        self.logicalShape = logicalShape; self.storedRows = storedRows
    }
}

/// A dense BF16 component resolved from an SPLW container.
public struct DenseBF16Record: Sendable, Equatable {
    public let name: String
    public let tensor: TensorSpec
    public init(name: String, tensor: TensorSpec) { self.name = name; self.tensor = tensor }
}

public struct WeightFile: Sendable {
    public let index: WeightIndex
    public let root: URL
    public let header: ConverterHeader
    public let fileSize: UInt64
    private let containerURL: URL?

    public init(indexURL: URL) throws { try self.init(indexURL: indexURL, root: indexURL.deletingLastPathComponent()) }
    public init(indexURL: URL, root: URL, config: ModelConfig? = nil) throws {
        guard FileManager.default.fileExists(atPath: indexURL.path) else { throw WeightFileError.missing(indexURL) }
        let loaded = try WeightIndex(url: indexURL)
        guard !loaded.entries.isEmpty else { throw WeightFileError.invalid("empty index") }
        if let config, let expected = loaded.configHash, expected.lowercased() != config.configHash.lowercased() { throw WeightFileError.configHashMismatch(expected: expected, observed: config.configHash) }
        for entry in loaded.entries.values {
            let shardURL = root.appendingPathComponent(entry.shard)
            let canonicalRoot = root.standardizedFileURL.path + "/"
            guard shardURL.standardizedFileURL.path.hasPrefix(canonicalRoot) else { throw WeightFileError.invalid("shard escapes root: \(entry.shard)") }
            guard FileManager.default.fileExists(atPath: shardURL.path) else { throw WeightFileError.invalid("missing shard: \(entry.shard)") }
        }
        self.index = loaded; self.root = root; self.header = ConverterHeader(dtype: "", alignment: ConverterAlignment(), configHash: "", tokenizerHash: ""); self.fileSize = 0; self.containerURL = nil
    }

    public init(splwURL: URL) throws {
        guard FileManager.default.fileExists(atPath: splwURL.path) else { throw WeightFileError.missing(splwURL) }
        let attrs = try FileManager.default.attributesOfItem(atPath: splwURL.path)
        guard let number = attrs[.size] as? NSNumber, number.int64Value >= 0 else { throw WeightFileError.invalid("invalid file size") }
        let size = UInt64(number.int64Value)
        guard size >= 16 else { throw WeightFileError.invalid("truncated framing") }
        let handle = try FileHandle(forReadingFrom: splwURL)
        func readExact(_ offset: UInt64, _ count: Int) throws -> Data {
            guard offset <= size, UInt64(count) <= size - offset else { throw WeightFileError.invalid("truncated framing") }
            try handle.seek(toOffset: offset)
            let result = try handle.read(upToCount: count) ?? Data()
            guard result.count == count else { throw WeightFileError.invalid("truncated framing") }
            return result
        }
        let framing: Data
        do { framing = try readExact(0, 16) } catch { try? handle.close(); throw error }
        guard String(decoding: framing[0..<4], as: UTF8.self) == "SPLW" else { try? handle.close(); throw WeightFileError.invalid("bad magic") }
        let version = Self.readLE(UInt32.self, framing, 4); guard version == 1 else { try? handle.close(); throw WeightFileError.invalid("unsupported version") }
        let headerLength = Self.readLE(UInt64.self, framing, 8)
        guard headerLength <= size - 16, headerLength <= UInt64(Int.max) else { try? handle.close(); throw WeightFileError.invalid("header length overflow") }
        let data: Data
        do { data = try readExact(16, Int(headerLength)); try handle.close() } catch { try? handle.close(); throw error }
        let decoded: ConverterHeader
        do { decoded = try JSONDecoder().decode(ConverterHeader.self, from: data) } catch { throw WeightFileError.invalid("malformed header: \(error)") }
        guard decoded.magic == "SPLW", decoded.version == 1, decoded.headerLength == headerLength else { throw WeightFileError.invalid("contradictory header framing") }
        // MLX affine codes of one width, or what a GGUF-derived artifact declares; the format and
        // the quantization must say the same.
        let affine = (decoded.quantization.bits == 4 || decoded.quantization.bits == 8) && decoded.quantization.groupSize == 64 && decoded.quantization.mode == "affine"
        guard decoded.format == Converter.ggufFormat ? decoded.quantization == .gguf : affine else { throw WeightFileError.invalid("unsupported quantization") }
        guard decoded.alignment.pointerAlignmentBytes > 0 else { throw WeightFileError.invalid("invalid alignment") }
        let headerEnd = 16 + headerLength
        guard headerEnd >= 16 else { throw WeightFileError.invalid("header offset overflow") }
        let payloadStart = Self.align(headerEnd, UInt64(decoded.alignment.pointerAlignmentBytes))
        guard payloadStart <= size, decoded.payloadLength <= size - payloadStart,
              payloadStart + decoded.payloadLength == size else { throw WeightFileError.invalid("payload size mismatch") }
        var previous: UInt64 = payloadStart
        var names = Set<String>()
        for record in decoded.tensorRecords {
            guard names.insert(record.name).inserted else { throw WeightFileError.invalid("duplicate tensor: \(record.name)") }
            guard record.dtype != .unknown else { throw WeightFileError.invalid("unknown dtype: \(record.name)") }
            guard !record.name.isEmpty, record.shape.allSatisfy({ $0 >= 0 }), record.payloadOffset >= payloadStart,
                  record.payloadOffset >= previous, record.payloadOffset <= size,
                  record.rawByteLength <= size - record.payloadOffset else { throw WeightFileError.invalid("tensor bounds: \(record.name)") }
            guard record.payloadOffset % UInt64(decoded.alignment.pointerAlignmentBytes) == 0 else { throw WeightFileError.invalid("unaligned tensor: \(record.name)") }
            guard record.alignmentPadding == record.payloadOffset - previous else { throw WeightFileError.invalid("padding mismatch: \(record.name)") }
            previous = record.payloadOffset + record.rawByteLength
        }
        guard previous <= size, decoded.tensorRecords.isEmpty || previous == size else { throw WeightFileError.invalid("unaccounted payload bytes") }
        self.header = decoded; self.root = splwURL.deletingLastPathComponent(); self.fileSize = size; self.containerURL = splwURL
        self.index = try WeightIndex(entries: decoded.tensorRecords.map { TensorEntry(name: $0.name, shard: $0.sourceShard, shape: $0.shape, dtype: $0.dtype.rawValue) })
    }

    /// Resolve one q4 weight and its required BF16 affine sidecars.
    public func resolveQ4(_ name: String) throws -> Q4WeightRecord { try q4(name) }

    /// Whether the container is GGUF-derived: its packed weights are planes, resolved by
    /// `gguf(_:)`, not MLX affine triplets.
    public var isGguf: Bool { header.format == Converter.ggufFormat }

    /// Resolve one GGUF weight by the name of its plane 0, `<base>.weight`, with `<base>.meta`
    /// and, for a format that has one, `<base>.plane1`. Every plane must be the size the weight's
    /// type and logical shape give it.
    public func gguf(_ name: String) throws -> GgufWeightRecord {
        guard let record = header.tensorRecords.first(where: { $0.name == name }) else {
            throw TensorInventoryError.missingTensor(name)
        }
        guard record.dtype == .u8 else {
            throw TensorInventoryError.dtypeMismatch(name, expected: .u8, observed: record.dtype)
        }
        guard isGguf, let id = record.ggufType, let type = GgufTensorType(rawValue: id),
              let geometry = GgufPlanes.geometry(of: type) else {
            throw WeightFileError.invalid("\(name) is not a GGUF weight of a quantised type")
        }
        guard let logical = record.logicalShape, logical.count == 2, logical[0] > 0, logical[1] > 0 else {
            throw WeightFileError.invalid("\(name) has no logical shape [rows, inner]")
        }
        let storedRows = (logical[0] + GgufPlanes.tileRows - 1) / GgufPlanes.tileRows * GgufPlanes.tileRows
        guard let sizes = GgufPlanes.sizes(of: type, rows: storedRows, inner: logical[1]) else {
            throw WeightFileError.invalid("\(name): a row of \(logical[1]) is not a whole number of \(type) blocks")
        }
        let base = name.hasSuffix(".weight") ? String(name.dropLast(".weight".count)) : name
        // A plane is recorded as bytes, `[stored rows, bytes a row]`, though it is laid out in tiles.
        func plane(_ planeName: String, bytes: Int) throws -> TensorSpec {
            guard let found = planeName == name ? record : header.tensorRecords.first(where: { $0.name == planeName }) else {
                throw WeightFileError.invalid("\(name) is \(type) and has no \(planeName)")
            }
            let shape = [storedRows, bytes / storedRows]
            guard found.dtype == .u8, found.rawByteLength == UInt64(bytes), found.shape == shape else {
                throw WeightFileError.invalid("\(planeName) is \(found.rawByteLength) bytes of shape \(found.shape); \(type) \(logical) needs \(bytes) of shape \(shape)")
            }
            return TensorSpec(name: found.name, dtype: found.dtype, shape: found.shape, byteOffset: Int64(found.payloadOffset),
                              byteLength: Int64(found.rawByteLength), shard: found.sourceShard)
        }
        let plane1Name = base + ".plane1"
        if geometry.plane1Bytes == 0, header.tensorRecords.contains(where: { $0.name == plane1Name }) {
            throw WeightFileError.invalid("\(name) is \(type), which has no second plane, yet \(plane1Name) exists")
        }
        return GgufWeightRecord(
            name: name, type: type,
            plane0: try plane(name, bytes: sizes.plane0),
            plane1: geometry.plane1Bytes == 0 ? nil : try plane(plane1Name, bytes: sizes.plane1),
            meta: try plane(base + ".meta", bytes: sizes.meta),
            logicalShape: logical, storedRows: storedRows)
    }

    /// `physicalShape` is the packed U32 matrix; `logicalShape` is the scalar matrix.
    public func q4(_ name: String) throws -> Q4WeightRecord {
        guard let record = header.tensorRecords.first(where: { $0.name == name }) else {
            throw TensorInventoryError.missingTensor(name)
        }
        guard record.dtype == .u32 else {
            throw TensorInventoryError.dtypeMismatch(name, expected: .u32, observed: record.dtype)
        }
        guard !isGguf else {
            throw TensorInventoryError.mlxAffineInvalid("\(name) is in a GGUF-derived artifact, which has no affine triplets")
        }
        guard record.physicalShape == record.shape, record.physicalShape.count == 2,
              record.physicalShape.allSatisfy({ $0 > 0 }),
              let logical = record.logicalShape, logical.count == 2,
              logical[0] == record.physicalShape[0],
              logical[1] == record.physicalShape[1] * (32 / header.quantization.bits) else {
            throw TensorInventoryError.mlxAffineInvalid("invalid physical/logical shape metadata for \(name)")
        }
        let base = name.hasSuffix(".weight") ? String(name.dropLast(".weight".count)) : name
        guard let scalesRecord = header.tensorRecords.first(where: { $0.name == base + ".scales" }),
              let biasesRecord = header.tensorRecords.first(where: { $0.name == base + ".biases" }) else {
            throw TensorInventoryError.mlxAffineInvalid("missing sidecar for \(name); expected \(base).scales and \(base).biases")
        }
        guard scalesRecord.dtype == .bf16, biasesRecord.dtype == .bf16 else {
            throw TensorInventoryError.mlxAffineInvalid("sidecars for \(name) must be BF16")
        }
        let groups = (logical[1] + 63) / 64
        let expected = [logical[0], groups]
        guard scalesRecord.shape == expected, biasesRecord.shape == expected else {
            throw TensorInventoryError.mlxAffineInvalid("sidecar shape for \(name) must be \(expected)")
        }
        return Q4WeightRecord(name: name,
            weight: TensorSpec(name: name, dtype: record.dtype, shape: record.physicalShape,
                               byteOffset: Int64(record.payloadOffset), byteLength: Int64(record.rawByteLength), shard: record.sourceShard),
            scales: TensorSpec(name: scalesRecord.name, dtype: scalesRecord.dtype, shape: scalesRecord.shape,
                               byteOffset: Int64(scalesRecord.payloadOffset), byteLength: Int64(scalesRecord.rawByteLength), shard: scalesRecord.sourceShard),
            biases: TensorSpec(name: biasesRecord.name, dtype: biasesRecord.dtype, shape: biasesRecord.shape,
                               byteOffset: Int64(biasesRecord.payloadOffset), byteLength: Int64(biasesRecord.rawByteLength), shard: biasesRecord.sourceShard),
            physicalShape: record.physicalShape, logicalShape: logical)
    }

    /// Resolve a non-q4 dense component and require its SPLW dtype to be BF16.
    public func denseBF16(_ name: String, expectedShape: [Int]? = nil) throws -> DenseBF16Record {
        guard let record = header.tensorRecords.first(where: { $0.name == name }) else {
            throw TensorInventoryError.missingTensor(name)
        }
        guard record.dtype == .bf16 else {
            throw TensorInventoryError.dtypeMismatch(name, expected: .bf16, observed: record.dtype)
        }
        if let expectedShape, record.shape != expectedShape {
            throw TensorInventoryError.shapeMismatch(name, expected: expectedShape, observed: record.shape)
        }
        let spec = TensorSpec(name: name, dtype: .bf16, shape: record.shape,
                              byteOffset: Int64(record.payloadOffset), byteLength: Int64(record.rawByteLength), shard: record.sourceShard)
        return DenseBF16Record(name: name, tensor: spec)
    }

    public func tensor(_ name: String, expectedDType: TensorDType? = nil, expectedShape: [Int]? = nil) throws -> TensorSpec {
        let entry = try index.require(name)
        if !header.tensorRecords.isEmpty, let r = header.tensorRecords.first(where: { $0.name == name }) {
            if let expectedDType, r.dtype != expectedDType { throw TensorInventoryError.dtypeMismatch(name, expected: expectedDType, observed: r.dtype) }
            if let expectedShape, r.shape != expectedShape { throw TensorInventoryError.shapeMismatch(name, expected: expectedShape, observed: r.shape) }
            return TensorSpec(name: r.name, dtype: r.dtype, shape: r.shape, byteOffset: Int64(r.payloadOffset), byteLength: Int64(r.rawByteLength), shard: r.sourceShard)
        }
        let shardURL = root.appendingPathComponent(entry.shard); let specs = try WeightIndex.parseShard(at: shardURL)
        guard let spec = specs[name] else { throw TensorInventoryError.missingTensor(name) }
        if let expectedDType, spec.dtype != expectedDType { throw TensorInventoryError.dtypeMismatch(name, expected: expectedDType, observed: spec.dtype) }
        if let expectedShape, spec.shape != expectedShape { throw TensorInventoryError.shapeMismatch(name, expected: expectedShape, observed: spec.shape) }
        return spec
    }
    /// Read an exact byte range from the file-backed SPLW container.
    public func readRange(offset: UInt64, length: UInt64) throws -> Data {
        guard let containerURL else { throw WeightFileError.invalid("range reads require an SPLW container") }
        guard offset <= fileSize, length <= fileSize - offset, length <= UInt64(Int.max) else {
            throw WeightFileError.invalid("range out of bounds")
        }
        let handle = try FileHandle(forReadingFrom: containerURL)
        do {
            try handle.seek(toOffset: offset)
            let result = try handle.read(upToCount: Int(length)) ?? Data()
            try handle.close()
            guard UInt64(result.count) == length else { throw WeightFileError.invalid("truncated range") }
            return result
        } catch {
            try? handle.close()
            throw error
        }
    }

    public func data(for spec: TensorSpec) throws -> Data {
        guard let offset = spec.byteOffset, let length = spec.byteLength,
              offset >= 0, length >= 0, length <= Int64(Int.max) else {
            throw WeightFileError.invalid("tensor lacks resolved location: \(spec.name)")
        }
        let url: URL
        if let containerURL { url = containerURL }
        else {
            guard let shard = spec.shard else { throw WeightFileError.invalid("tensor lacks shard: \(spec.name)") }
            url = root.appendingPathComponent(shard)
        }
        let attrs = try FileManager.default.attributesOfItem(atPath: url.path)
        guard let size = (attrs[.size] as? NSNumber)?.int64Value,
              offset <= size, length <= size - offset else {
            throw TensorInventoryError.boundsViolation(spec.name)
        }
        let handle = try FileHandle(forReadingFrom: url)
        do {
            try handle.seek(toOffset: UInt64(offset))
            let result = try handle.read(upToCount: Int(length)) ?? Data()
            try handle.close()
            guard result.count == Int(length) else { throw TensorInventoryError.boundsViolation(spec.name) }
            return result
        } catch {
            try? handle.close()
            throw error
        }
    }
    private static func readLE<T: FixedWidthInteger>(_ type: T.Type, _ data: Data, _ offset: Int) -> T { data[offset..<offset + MemoryLayout<T>.size].withUnsafeBytes { T(littleEndian: $0.load(as: T.self)) } }
    private static func align(_ n: UInt64, _ a: UInt64) -> UInt64 { n + ((a - n % a) % a) }
}
