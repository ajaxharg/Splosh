// WeightIndex.swift — strict safetensors index inventory.
import Foundation

public struct TensorEntry: Codable, Sendable, Equatable {
    public let name: String
    public let shard: String
    public let shape: [Int]
    public let dtype: String
    public let dataOffsets: [Int64]?
    public init(name: String, shard: String, shape: [Int] = [], dtype: String = "unknown", dataOffsets: [Int64]? = nil) {
        self.name = name; self.shard = shard; self.shape = shape; self.dtype = dtype; self.dataOffsets = dataOffsets
    }
    public var tensorDType: TensorDType? { try? TensorDType(safetensorsName: dtype.uppercased()) }
}

public enum TensorSourceClass: String, Codable, Sendable {
    case q4Weight
    case q4Scales
    case q4Biases
    case dense
}

public struct ResolvedTensor: Codable, Sendable, Equatable {
    public let spec: TensorSpec
    public let sourceClass: TensorSourceClass
    public init(spec: TensorSpec, sourceClass: TensorSourceClass) { self.spec = spec; self.sourceClass = sourceClass }
}

public struct TensorInventory: Sendable, Equatable {
    public let tensors: [String: ResolvedTensor]
    public let declaredTotalSize: Int64?
    public let observedShardBytes: Int64
    public init(tensors: [String: ResolvedTensor], declaredTotalSize: Int64?, observedShardBytes: Int64) { self.tensors = tensors; self.declaredTotalSize = declaredTotalSize; self.observedShardBytes = observedShardBytes }
    public func require(_ name: String) throws -> ResolvedTensor { guard let value = tensors[name] else { throw TensorInventoryError.missingTensor(name) }; return value }
}

public struct WeightIndex: Sendable {
    public let entries: [String: TensorEntry]
    public let totalSize: Int64?
    public let sidecarDType: TensorDType?
    public let configHash: String?

    public init(url: URL) throws {
        guard FileManager.default.fileExists(atPath: url.path) else { throw WeightFileError.missing(url) }
        let data = try Data(contentsOf: url)
        guard let object = try JSONSerialization.jsonObject(with: data) as? [String: Any],
              let map = object["weight_map"] as? [String: String], !map.isEmpty else { throw WeightFileError.invalid("missing or empty weight_map") }
        var parsed: [String: TensorEntry] = [:]
        for (name, shard) in map {
            guard !name.isEmpty, !shard.isEmpty else { throw TensorInventoryError.malformedIndex("empty tensor or shard name") }
            try Self.validateShardName(shard)
            parsed[name] = TensorEntry(name: name, shard: shard)
        }
        let metadata = object["metadata"] as? [String: Any]
        let total = (metadata?["total_size"] as? NSNumber)?.int64Value
        let sidecar = try Self.SelfmetadataDType(object: object)
        self.entries = parsed; self.totalSize = total; self.sidecarDType = sidecar
        self.configHash = object["config_hash"] as? String ?? metadata?["config_hash"] as? String
    }

    public init(entries: [TensorEntry], totalSize: Int64? = nil, sidecarDType: TensorDType? = nil) throws {
        guard !entries.isEmpty else { throw WeightFileError.invalid("empty index") }
        var result: [String: TensorEntry] = [:]
        for entry in entries {
            guard result[entry.name] == nil else { throw WeightFileError.invalid("duplicate tensor \(entry.name)") }
            try Self.entryToSpec(entry).validate(); try Self.validateShardName(entry.shard)
            result[entry.name] = entry
        }
        self.entries = result; self.totalSize = totalSize; self.sidecarDType = sidecarDType; self.configHash = nil
    }

    private static func entryToSpec(_ e: TensorEntry) -> TensorSpec {
        TensorSpec(name: e.name, dtype: e.tensorDType ?? .unknown, shape: e.shape, byteOffset: e.dataOffsets?.first, byteLength: e.dataOffsets.flatMap { $0.count == 2 ? $0[1] - $0[0] : nil }, shard: e.shard)
    }
    private static func validateShardName(_ shard: String) throws {
        guard !shard.hasPrefix("/"), !shard.split(separator: "/").contains("..") else { throw TensorInventoryError.shardTraversal(shard) }
    }
    private static func SelfmetadataDType(object: [String: Any]) throws -> TensorDType? {
        let metadata = object["metadata"] as? [String: Any]
        let raw = (metadata?["sidecar_dtype"] as? String) ?? (object["sidecar_dtype"] as? String)
        guard let raw else { return nil }
        guard let dtype = try? TensorDType(safetensorsName: raw.uppercased()) else { throw TensorInventoryError.malformedIndex("invalid sidecar_dtype \(raw)") }
        return dtype
    }
    public func entry(_ name: String) -> TensorEntry? { entries[name] }
    public func require(_ name: String) throws -> TensorEntry { guard let e = entries[name] else { throw TensorInventoryError.missingTensor(name) }; return e }
    public var tensorCount: Int { entries.count }

    /// Resolve every indexed tensor, parsing each referenced shard exactly once.
    /// The declared index total is metadata only; observed bytes are measured from files.
    public func inventory(root: URL) throws -> TensorInventory {
        var parsed: [String: [String: TensorSpec]] = [:]
        var observed: Int64 = 0
        for shard in Set(entries.values.map(\.shard)) {
            try Self.validateShardName(shard)
            let url = root.appendingPathComponent(shard)
            guard url.standardizedFileURL.path.hasPrefix(root.standardizedFileURL.path + "/") else { throw TensorInventoryError.shardTraversal(shard) }
            guard FileManager.default.fileExists(atPath: url.path) else { throw TensorInventoryError.shardMissing(shard) }
            let shardSpecs = try Self.parseShard(at: url)
            // parseShard reports the filename; inventory must preserve the indexed
            // relative path so callers can safely reopen nested shards.
            parsed[shard] = shardSpecs.mapValues { spec in
                TensorSpec(name: spec.name, dtype: spec.dtype, shape: spec.shape,
                           byteOffset: spec.byteOffset, byteLength: spec.byteLength, shard: shard)
            }
            let attributes = try FileManager.default.attributesOfItem(atPath: url.path)
            let shardBytes = (attributes[.size] as? NSNumber)?.int64Value ?? 0
            let (newObserved, overflow) = observed.addingReportingOverflow(shardBytes)
            guard !overflow else { throw TensorInventoryError.boundsViolation("observed shard size overflow") }
            observed = newObserved
        }
        var resolved: [String: ResolvedTensor] = [:]
        for (name, entry) in entries {
            guard let spec = parsed[entry.shard]?[name] else { throw TensorInventoryError.indexCoverage(name) }
            guard resolved[name] == nil else { throw TensorInventoryError.duplicateTensor(name) }
            resolved[name] = ResolvedTensor(spec: spec, sourceClass: .dense)
        }
        let names = Set(resolved.keys)
        for name in names where name.hasSuffix(".weight") {
            let base = String(name.dropLast(".weight".count))
            let scalesName = base + ".scales", biasesName = base + ".biases"
            guard let weight = resolved[name] else { continue }
            let hasScales = names.contains(scalesName), hasBiases = names.contains(biasesName)
            if hasScales || hasBiases {
                guard hasScales && hasBiases else { throw TensorInventoryError.mlxAffineInvalid("incomplete triplet for \(base)") }
                guard let scales = resolved[scalesName], let biases = resolved[biasesName], scales.spec.shard == weight.spec.shard, biases.spec.shard == weight.spec.shard else { throw TensorInventoryError.mlxAffineInvalid("triplet crosses shards for \(base)") }
                try Self.validateMLX(weight: weight.spec, scales: scales.spec, biases: biases.spec)
                resolved[name] = ResolvedTensor(spec: weight.spec, sourceClass: .q4Weight)
                resolved[scalesName] = ResolvedTensor(spec: scales.spec, sourceClass: .q4Scales)
                resolved[biasesName] = ResolvedTensor(spec: biases.spec, sourceClass: .q4Biases)
            }
        }
        for name in names where name.hasSuffix(".scales") || name.hasSuffix(".biases") {
            let base = name.replacingOccurrences(of: name.hasSuffix(".scales") ? ".scales" : ".biases", with: "")
            guard names.contains(base + ".weight") else { throw TensorInventoryError.mlxAffineInvalid("orphan sidecar \(name)") }
        }
        return TensorInventory(tensors: resolved, declaredTotalSize: totalSize, observedShardBytes: observed)
    }

    private static func validateMLX(weight: TensorSpec, scales: TensorSpec, biases: TensorSpec) throws {
        guard weight.dtype == .u32, weight.shape.count == 2, weight.shape.allSatisfy({ $0 > 0 }) else { throw TensorInventoryError.mlxAffineInvalid("\(weight.name) must be positive rank-2 U32") }
        // MLX stores eight 4-bit values in each U32 word. The header shape is
        // the physical matrix [rows, packedWordColumns], not the logical scalar shape.
        let rows = weight.shape[0], rawWordColumns = weight.shape[1]
        let (rowWords, rowWordsOverflow) = Int64(rows).multipliedReportingOverflow(by: Int64(rawWordColumns))
        let (packed, packedOverflow) = rowWords.multipliedReportingOverflow(by: 4)
        guard !rowWordsOverflow, !packedOverflow else { throw TensorInventoryError.mlxAffineInvalid("packed length overflow for \(weight.name)") }
        guard weight.byteLength == packed else { throw TensorInventoryError.mlxAffineInvalid("packed length for \(weight.name)") }
        guard scales.dtype == .bf16, biases.dtype == .bf16 else { throw TensorInventoryError.mlxAffineInvalid("sidecars for \(weight.name) must be BF16") }
        guard rawWordColumns % 8 == 0 else { throw TensorInventoryError.mlxAffineInvalid("packed word columns for \(weight.name) must be divisible by 8") }
        let (logicalColumns, logicalOverflow) = Int64(rawWordColumns).multipliedReportingOverflow(by: 8)
        guard !logicalOverflow, logicalColumns > 0 else { throw TensorInventoryError.mlxAffineInvalid("logical column overflow for \(weight.name)") }
        // Eight U32 words represent one complete group of 64 logical columns.
        let groups = rawWordColumns / 8
        let (rowGroups, sidecarOverflow) = Int64(rows).multipliedReportingOverflow(by: Int64(groups))
        let (expectedBytes, bytesOverflow) = rowGroups.multipliedReportingOverflow(by: 2)
        guard !sidecarOverflow, !bytesOverflow else { throw TensorInventoryError.mlxAffineInvalid("sidecar length overflow for \(weight.name)") }
        let expectedShape = [rows, groups]
        guard scales.shape == expectedShape, biases.shape == expectedShape, scales.byteLength == expectedBytes, biases.byteLength == expectedBytes else { throw TensorInventoryError.mlxAffineInvalid("group-64 sidecar shape for \(weight.name)") }
    }

    /// Parse and validate every tensor header in a safetensors shard.
    public static func parseShard(at url: URL) throws -> [String: TensorSpec] {
        guard FileManager.default.fileExists(atPath: url.path) else { throw TensorInventoryError.shardMissing(url.path) }
        let data: Data
        do {
            data = try Data(contentsOf: url)
        } catch {
            throw TensorInventoryError.headerMalformed("unable to read shard: \(error.localizedDescription)")
        }
        guard data.count >= 8 else { throw TensorInventoryError.headerMalformed("header length missing") }
        let headerLength = data.prefix(8).enumerated().reduce(UInt64(0)) { $0 | UInt64($1.element) << (UInt64($1.offset) * 8) }
        guard headerLength <= UInt64(data.count - 8), headerLength <= UInt64(Int.max) else { throw TensorInventoryError.headerMalformed("header length out of bounds") }
        let start = 8, end = start + Int(headerLength)
        let object: [String: Any]
        do {
            guard let decoded = try JSONSerialization.jsonObject(with: data[start..<end]) as? [String: Any] else { throw TensorInventoryError.headerMalformed("header is not an object") }
            object = decoded
        } catch let error as TensorInventoryError {
            throw error
        } catch {
            throw TensorInventoryError.headerMalformed("invalid JSON")
        }
        let payload = Int64(end), fileSize = Int64(data.count)
        var result: [String: TensorSpec] = [:]
        for (name, value) in object {
            if name == "__metadata__" { continue }
            guard let t = value as? [String: Any], let rawDType = t["dtype"] as? String, let rawShape = t["shape"] as? [NSNumber], let offsets = t["data_offsets"] as? [NSNumber], offsets.count == 2 else { throw TensorInventoryError.headerMalformed(name) }
            let dtype = try TensorDType(safetensorsName: rawDType)
            let shape = rawShape.map(\.intValue)
            guard shape.allSatisfy({ $0 >= 0 }) else { throw TensorInventoryError.invalidShape(name, shape) }
            let a = offsets[0].int64Value, b = offsets[1].int64Value
            guard a >= 0, b >= a, payload <= fileSize, payload <= fileSize - (b - a) else { throw TensorInventoryError.boundsViolation(name) }
            let elementCount = shape.reduce(Int64(1)) { partial, d in
                let dimension = Int64(d)
                return dimension == 0 || partial == 0 ? 0 : (partial > Int64.max / dimension ? Int64.max : partial * dimension)
            }
            let width = dtype.byteWidth ?? 0
            let expected = elementCount > Int64.max / width ? Int64.max : elementCount * width
            guard expected == b - a else { throw TensorInventoryError.boundsViolation("\(name): expected \(expected), got \(b-a)") }
            let spec = TensorSpec(name: name, dtype: dtype, shape: shape, byteOffset: payload + a, byteLength: b - a, shard: url.lastPathComponent)
            try spec.validate(); result[name] = spec
        }
        return result
    }
}
