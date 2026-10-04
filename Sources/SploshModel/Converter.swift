// Converter.swift — deterministic, self-describing SPLW container writer.
import Foundation
import SploshQuant
import CryptoKit

public enum ConverterError: Error, Equatable, CustomStringConvertible {
    case missingSource(URL)
    case missingAlignmentEvidence(URL)
    case missingPinnedAssets(String)
    case invalidSource(String)
    case interruptedWrite
    public var description: String {
        switch self {
        case .missingSource(let u): return "converter source missing: \(u.path)"
        case .missingAlignmentEvidence(let u): return "converter alignment evidence missing: \(u.path)"
        case .missingPinnedAssets(let d): return "converter assets absent: \(d)"
        case .invalidSource(let s): return "invalid converter source: \(s)"
        case .interruptedWrite: return "converter interrupted write left no final artifact"
        }
    }
}

public struct ConverterAlignment: Codable, Equatable, Sendable {
    public let rowStrideBytes: Int
    public let pointerAlignmentBytes: Int
    public let padded: Bool
    public init(rowStrideBytes: Int = 128, pointerAlignmentBytes: Int = 128, padded: Bool = true) {
        self.rowStrideBytes = rowStrideBytes; self.pointerAlignmentBytes = pointerAlignmentBytes; self.padded = padded
    }
}

public struct ConverterQuantization: Codable, Equatable, Sendable {
    public let bits: Int
    public let groupSize: Int
    public let mode: String
    public init(bits: Int = 4, groupSize: Int = 64, mode: String = "affine") {
        self.bits = bits; self.groupSize = groupSize; self.mode = mode
    }

    /// What a GGUF-derived artifact declares: llama.cpp's formats in groups of 32, a format a
    /// weight (each `.weight` record's `ggufType`), so no one code width; `bits` is 0.
    public static let gguf = ConverterQuantization(bits: 0, groupSize: 32, mode: "gguf")
}

public struct SPLWTensorRecord: Codable, Equatable, Sendable {
    public let name: String
    public let dtype: TensorDType
    public let shape: [Int]
    /// Physical packed shape as stored by safetensors (for q4 weights this is U32 words).
    public let physicalShape: [Int]
    /// Logical scalar shape represented by a packed q4 weight; nil for non-packed tensors.
    public let logicalShape: [Int]?
    public let payloadOffset: UInt64
    public let rawByteLength: UInt64
    public let rowStride: UInt64
    public let alignmentPadding: UInt64
    public let sourceShard: String
    /// The ggml type id of a GGUF weight (`GgufTensorType`), on the record of its plane 0; nil
    /// for every other record, and absent from the headers written before there were any.
    public let ggufType: UInt32?
    public init(name: String, dtype: TensorDType, shape: [Int], physicalShape: [Int]? = nil, logicalShape: [Int]? = nil, payloadOffset: UInt64, rawByteLength: UInt64, rowStride: UInt64, alignmentPadding: UInt64, sourceShard: String, ggufType: UInt32? = nil) {
        self.name = name; self.dtype = dtype; self.shape = shape; self.physicalShape = physicalShape ?? shape; self.logicalShape = logicalShape; self.payloadOffset = payloadOffset; self.rawByteLength = rawByteLength; self.rowStride = rowStride; self.alignmentPadding = alignmentPadding; self.sourceShard = sourceShard; self.ggufType = ggufType
    }
}

public struct ConverterHeader: Codable, Equatable, Sendable {
    public let magic: String
    public let version: Int
    public let format: String
    public let dtype: String
    public let quantization: ConverterQuantization
    public let alignment: ConverterAlignment
    public let configHash: String
    public let tokenizerHash: String
    public let tensorRecords: [SPLWTensorRecord]
    public let headerLength: UInt64
    public let payloadLength: UInt64
    public init(format: String = "mlxAffine4", dtype: String, alignment: ConverterAlignment, configHash: String, tokenizerHash: String, tensorRecords: [SPLWTensorRecord] = [], headerLength: UInt64 = 0, payloadLength: UInt64 = 0, quantization: ConverterQuantization = ConverterQuantization()) {
        self.magic = "SPLW"; self.version = 1; self.format = format; self.dtype = dtype; self.quantization = quantization; self.alignment = alignment; self.configHash = configHash; self.tokenizerHash = tokenizerHash; self.tensorRecords = tensorRecords; self.headerLength = headerLength; self.payloadLength = payloadLength
    }
}

public struct ConversionReport: Sendable, Equatable {
    public let outputURL: URL
    public let tensorCount: Int
    public let byteCount: UInt64
    public let header: ConverterHeader
}
public struct VerificationReport: Sendable, Equatable {
    public let outputURL: URL
    public let tensorCount: Int
    public let byteCount: UInt64
}

public enum Converter {
    public static let expectedTextTowerBytes: Int64 = 15_159_640_064
    public static let alignmentEvidencePath = "audit/M3-alignment.md"
    public static let alignment = 128

    public static func alignmentEvidence(from root: URL) throws -> ConverterAlignment {
        let url = root.appendingPathComponent(alignmentEvidencePath)
        guard let text = try? String(contentsOf: url, encoding: .utf8), text.contains("Selected rule"), text.contains("128-byte") else { throw ConverterError.missingAlignmentEvidence(url) }
        return ConverterAlignment()
    }
    public static func requireAssets(at root: URL) throws {
        // Callers may provide either the materialised q4 directory or the
        // repository root.  Never fall back to the BF16 tree: conversion is
        // gated specifically on the pinned MLX q4 index.
        let candidates = [
            root.appendingPathComponent("model.safetensors.index.json"),
            root.appendingPathComponent("inputs/mlx-q4/model.safetensors.index.json")
        ]
        let fileManager = FileManager.default
        guard let index = candidates.first(where: { path in
            guard fileManager.fileExists(atPath: path.path) else { return false }
            return (try? path.resourceValues(forKeys: [.isRegularFileKey]).isRegularFile) == true
        }) else {
            let paths = candidates.map(\.path).joined(separator: ", ")
            throw ConverterError.missingPinnedAssets("missing q4 index (checked \(paths)); MLX q4 pack is not materialised")
        }
        _ = index
    }

    /// The code width of a pack, read from its tensors: a quant group is 64 columns, so a packed
    /// weight with G groups a row has 8 G words a row at 4 bits and 16 G at 8. Every packed
    /// weight must agree, and so must `config.json` where it declares a quantization. A pack
    /// with no packed weights is taken as 4-bit.
    static func packQuantization(specs: [TensorSpec], configURL: URL) throws -> ConverterQuantization {
        let byName = Dictionary(specs.map { ($0.name, $0) }, uniquingKeysWith: { first, _ in first })
        var widths = Set<Int>()
        for spec in specs where spec.dtype == .u32 && spec.shape.count == 2 && spec.name.hasSuffix(".weight") {
            guard let scales = byName[String(spec.name.dropLast(".weight".count)) + ".scales"],
                  scales.shape.count == 2, scales.shape[1] > 0 else { continue }
            let words = spec.shape[1], groups = scales.shape[1]
            guard words % (2 * groups) == 0 else {
                throw ConverterError.invalidSource("\(spec.name): \(words) words for \(groups) groups a row is not a whole code width")
            }
            widths.insert(words / (2 * groups))
        }
        guard widths.count <= 1, widths.isSubset(of: [4, 8]) else {
            throw ConverterError.invalidSource("unsupported code widths \(widths.sorted()); packs are 4-bit or 8-bit throughout")
        }
        let bits = widths.first ?? 4
        struct Config: Decodable {
            struct Quantization: Decodable {
                let bits: Int, groupSize: Int?, mode: String?
                enum CodingKeys: String, CodingKey { case bits, groupSize = "group_size", mode }
            }
            let quantization: Quantization?
        }
        if let data = try? Data(contentsOf: configURL), let declared = (try? JSONDecoder().decode(Config.self, from: data))?.quantization {
            guard declared.bits == bits, (declared.groupSize ?? 64) == 64, (declared.mode ?? "affine") == "affine" else {
                throw ConverterError.invalidSource("config.json declares \(declared.bits)-bit \(declared.mode ?? "affine") quantization in groups of \(declared.groupSize ?? 64); the tensors are \(bits)-bit affine in groups of 64")
            }
        }
        return ConverterQuantization(bits: bits)
    }

    public static func convert(inputRoot: URL, outputURL: URL, verify: Bool = true) throws -> ConversionReport {
        let indexURL = inputRoot.appendingPathComponent("model.safetensors.index.json")
        guard FileManager.default.fileExists(atPath: indexURL.path) else { throw ConverterError.missingSource(indexURL) }
        let index = try WeightIndex(url: indexURL)
        let inventory = try index.inventory(root: inputRoot)
        let evidenceRoot = inputRoot.deletingLastPathComponent().deletingLastPathComponent()
        let evidence = try alignmentEvidence(from: evidenceRoot)
        guard evidence.rowStrideBytes == Packing.alignment, evidence.pointerAlignmentBytes == Packing.alignment, evidence.padded else { throw ConverterError.invalidSource("alignment evidence disagrees with Packing") }
        let configHash = try hashRequired(inputRoot.appendingPathComponent("config.json"), label: "config.json")
        let tokenizerHash = try hashRequired(inputRoot.appendingPathComponent("tokenizer.json"), label: "tokenizer.json")
        let alignment = evidence
        let sortedSpecs = inventory.tensors.keys.sorted().compactMap { inventory.tensors[$0]?.spec }
        guard !sortedSpecs.isEmpty else { throw ConverterError.invalidSource("empty resolved inventory") }
        // Inventory only: payload bytes are fetched one tensor at a time below, so a
        // large q4 model never has all of its payload resident in memory.
        guard sortedSpecs.allSatisfy({ $0.byteLength != nil && $0.byteLength! >= 0 }) else { throw ConverterError.invalidSource("unresolved tensor byte length") }
        let quantization = try packQuantization(specs: sortedSpecs, configURL: inputRoot.appendingPathComponent("config.json"))
        let format = rowMajorFormat(bits: quantization.bits)
        // Logical columns per packed U32 word.
        let codesPerWord = 32 / quantization.bits
        var records = sortedSpecs.map { spec -> SPLWTensorRecord in
            let byteLength = spec.byteLength!
            let byteLengthInt = Int(byteLength)
            let rowBytes = spec.dtype == .u32 && spec.shape.count == 2 ? spec.shape[1] * 4 : (spec.shape.count == 2 ? spec.shape[1] * (spec.dtype.byteWidth.map(Int.init) ?? 1) : byteLengthInt)
            let stride = ((rowBytes + alignment.rowStrideBytes - 1) / alignment.rowStrideBytes) * alignment.rowStrideBytes
            return SPLWTensorRecord(name: spec.name, dtype: spec.dtype, shape: spec.shape, physicalShape: spec.shape, logicalShape: spec.dtype == .u32 && spec.shape.count == 2 ? [spec.shape[0], spec.shape[1] * codesPerWord] : nil, payloadOffset: 0, rawByteLength: UInt64(byteLength), rowStride: UInt64(stride), alignmentPadding: 0, sourceShard: spec.shard ?? "")
        }
        let source = try WeightFile(indexURL: indexURL, root: inputRoot)
        let base: UInt64 = 16 // magic + version + header length
        var header = ConverterHeader(format: format, dtype: sortedSpecs[0].dtype.rawValue, alignment: alignment, configHash: configHash, tokenizerHash: tokenizerHash, tensorRecords: records, quantization: quantization)
        var headerData = try canonicalJSON(header)
        for _ in 0..<8 {
            let payloadStart = align(base + UInt64(headerData.count), UInt64(alignment.pointerAlignmentBytes))
            var cursor = payloadStart
            for i in records.indices {
                let next = align(cursor, UInt64(alignment.pointerAlignmentBytes))
                records[i] = SPLWTensorRecord(name: records[i].name, dtype: records[i].dtype, shape: records[i].shape, physicalShape: records[i].physicalShape, logicalShape: records[i].logicalShape, payloadOffset: next, rawByteLength: records[i].rawByteLength, rowStride: records[i].rowStride, alignmentPadding: next - cursor, sourceShard: records[i].sourceShard)
                cursor = next + records[i].rawByteLength
            }
            header = ConverterHeader(format: format, dtype: "MIXED", alignment: alignment, configHash: configHash, tokenizerHash: tokenizerHash, tensorRecords: records, headerLength: UInt64(headerData.count), payloadLength: cursor - payloadStart, quantization: quantization)
            let nextData = try canonicalJSON(header)
            if nextData == headerData { break }
            headerData = nextData
        }
        let payloadStart = align(base + UInt64(headerData.count), UInt64(alignment.pointerAlignmentBytes))
        guard header.headerLength == UInt64(headerData.count) else { throw ConverterError.interruptedWrite }
        let temp = outputURL.deletingLastPathComponent().appendingPathComponent(".\(outputURL.lastPathComponent).tmp-\(UUID().uuidString)")
        let fileManager = FileManager.default
        var handle: FileHandle?
        do {
            try fileManager.createDirectory(at: outputURL.deletingLastPathComponent(), withIntermediateDirectories: true)
            fileManager.createFile(atPath: temp.path, contents: nil)
            handle = try FileHandle(forWritingTo: temp)
            var framing = Data("SPLW".utf8); appendLE(UInt32(1), to: &framing); appendLE(UInt64(headerData.count), to: &framing); framing.append(headerData)
            try handle!.write(contentsOf: framing)
            let initialPadding = Int(payloadStart) - framing.count
            if initialPadding > 0 { try handle!.write(contentsOf: Data(repeating: 0, count: initialPadding)) }
            var written = payloadStart
            for (i, spec) in sortedSpecs.enumerated() {
                let target = records[i].payloadOffset
                if target < written { throw ConverterError.interruptedWrite }
                let padding = target - written
                if padding > 0 { try handle!.write(contentsOf: Data(repeating: 0, count: Int(padding))) }
                let bytes = try source.data(for: spec)
                guard UInt64(bytes.count) == records[i].rawByteLength else { throw ConverterError.interruptedWrite }
                try handle!.write(contentsOf: bytes)
                written = target + UInt64(bytes.count)
            }
            guard written == payloadStart + header.payloadLength else { throw ConverterError.interruptedWrite }
            try handle!.synchronize()
            try handle!.close(); handle = nil
            if verify { _ = try Self.verify(outputURL: temp, sourceRoot: inputRoot) }
            if fileManager.fileExists(atPath: outputURL.path) { try fileManager.removeItem(at: outputURL) }
            try fileManager.moveItem(at: temp, to: outputURL)
        } catch {
            try? handle?.close()
            try? fileManager.removeItem(at: temp)
            throw error
        }
        return ConversionReport(outputURL: outputURL, tensorCount: records.count, byteCount: payloadStart + header.payloadLength, header: header)
    }

    public static func verify(outputURL: URL, sourceRoot: URL? = nil) throws -> VerificationReport {
        let file = try WeightFile(splwURL: outputURL)
        let records = file.header.tensorRecords
        guard records.map(\.name) == records.map(\.name).sorted() else { throw WeightFileError.invalid("records are not sorted") }
        if let sourceRoot {
            let index = try WeightIndex(url: sourceRoot.appendingPathComponent("model.safetensors.index.json"))
            let inventory = try index.inventory(root: sourceRoot)
            let configHash = try hashRequired(sourceRoot.appendingPathComponent("config.json"), label: "config.json")
            let tokenizerHash = try hashRequired(sourceRoot.appendingPathComponent("tokenizer.json"), label: "tokenizer.json")
            guard file.header.configHash == configHash, file.header.tokenizerHash == tokenizerHash else { throw WeightFileError.invalid("source hash mismatch") }
            guard Set(inventory.tensors.keys) == Set(records.map(\.name)) else { throw WeightFileError.invalid("source coverage mismatch") }
            let source = try WeightFile(indexURL: sourceRoot.appendingPathComponent("model.safetensors.index.json"), root: sourceRoot)
            // Mapped, not read: the 8-bit artifact is 27 GiB, and this may run beside a live server.
            let outputData = try Data(contentsOf: outputURL, options: .alwaysMapped)
            for record in records {
                guard let resolved = inventory.tensors[record.name] else { throw WeightFileError.invalid("source missing: \(record.name)") }
                guard resolved.spec.dtype == record.dtype, resolved.spec.shape == record.shape, resolved.spec.shard == record.sourceShard else { throw WeightFileError.invalid("source metadata mismatch: \(record.name)") }
                let bytes = try source.data(for: resolved.spec)
                guard bytes.count == record.rawByteLength, Data(outputData[Int(record.payloadOffset)..<Int(record.payloadOffset + record.rawByteLength)]) == bytes else { throw WeightFileError.invalid("source bytes mismatch: \(record.name)") }
            }
        }
        return VerificationReport(outputURL: outputURL, tensorCount: records.count, byteCount: file.fileSize)
    }
    private static func hashRequired(_ url: URL, label: String) throws -> String { guard let d = try? Data(contentsOf: url) else { throw ConverterError.missingSource(url) }; return SHA256.hash(data: d).map { String(format: "%02x", $0) }.joined() }
    static func canonicalJSON<T: Encodable>(_ value: T) throws -> Data { let e = JSONEncoder(); e.outputFormatting = [.sortedKeys]; return try e.encode(value) }
    static func align(_ n: UInt64, _ a: UInt64) -> UInt64 { n + ((a - n % a) % a) }
    static func appendLE<T: FixedWidthInteger>(_ value: T, to data: inout Data) { var v = value.littleEndian; withUnsafeBytes(of: &v) { data.append(contentsOf: $0) } }
}
