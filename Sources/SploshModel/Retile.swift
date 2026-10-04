// Retile.swift — rewrite an SPLW artifact with its q4 weights in the tiled layout.
//
// The engine's kernels read weights as [tile of 128 rows][quant group][row in tile][32 bytes of
// codes], with scales and biases as [tile][group][row]. MLX stores them row-major. Converting
// once on disk means the server maps a single copy instead of building the tiled form in
// memory beside the original.
//
// Tensor names, sizes and order are unchanged; only the bytes of eligible q4 triplets are
// permuted, and the header's `format` records the layout.

import Foundation

public extension Converter {
    static let rowMajorFormat = "mlxAffine4"
    static let tiledFormat = "mlxAffine4.tiled128"
    static let tileRows = 128

    /// The header's `format` for a pack of this code width, row-major and tiled. The 8-bit pack
    /// is the 4-bit one with a byte per code, so its tiled element is 64 bytes, not 32.
    static func rowMajorFormat(bits: Int) -> String { "mlxAffine\(bits)" }
    static func tiledFormat(bits: Int) -> String { rowMajorFormat(bits: bits) + ".tiled128" }
    /// The code width a `format` names and whether it is the tiled layout; nil for any other.
    static func layout(ofFormat format: String) -> (bits: Int, tiled: Bool)? {
        for bits in [4, 8] {
            if format == rowMajorFormat(bits: bits) { return (bits, false) }
            if format == tiledFormat(bits: bits) { return (bits, true) }
        }
        return nil
    }

    /// Whether a packed q4 weight is stored tiled in a tiled artifact. The embedding is a
    /// row gather, so it stays row-major; so does anything whose row count is not a tile multiple.
    static func isTiledWeight(name: String, rows: Int) -> Bool {
        name.hasPrefix("language_model.") && name.hasSuffix(".weight")
            && !name.contains("embed_tokens") && rows % tileRows == 0
    }

    struct RetileReport: Sendable {
        public let format: String
        public let outputURL: URL
        public let tensorCount: Int
        public let tiledWeights: Int
        public let byteCount: UInt64
    }

    static func retile(inputURL: URL, outputURL: URL) throws -> RetileReport {
        let source = try WeightFile(splwURL: inputURL)
        let old = source.header
        guard let layout = layout(ofFormat: old.format), !layout.tiled, layout.bits == old.quantization.bits else {
            throw ConverterError.invalidSource("retile expects a row-major artifact (\(rowMajorFormat) or \(rowMajorFormat(bits: 8))), found \(old.format)")
        }
        // One (row, quant group) of codes: 64 of them.
        let codeBytes = 8 * layout.bits
        let records = old.tensorRecords.sorted { $0.payloadOffset < $1.payloadOffset }
        guard let firstOffset = records.first?.payloadOffset else { throw ConverterError.invalidSource("artifact has no tensors") }
        let alignment = UInt64(old.alignment.pointerAlignmentBytes)
        let oldStart = align(16 + old.headerLength, alignment)

        // Which records are permuted, and how: packed weights, or their sidecars.
        var packedRows: [String: Int] = [:]
        for record in records where record.dtype == .u32 && record.shape.count == 2 && isTiledWeight(name: record.name, rows: record.shape[0]) {
            packedRows[String(record.name.dropLast(".weight".count))] = record.shape[0]
        }
        func sidecarBase(_ name: String) -> String? {
            for suffix in [".scales", ".biases"] where name.hasSuffix(suffix) {
                let base = String(name.dropLast(suffix.count))
                if packedRows[base] != nil { return base }
            }
            return nil
        }

        // The header length depends on the offsets it contains; iterate to a fixed point.
        var headerData = Data()
        var header = old
        var shift: UInt64 = 0
        for _ in 0..<16 {
            let newStart = align(16 + UInt64(headerData.count), alignment)
            shift = newStart &- oldStart
            let moved = old.tensorRecords.map {
                SPLWTensorRecord(name: $0.name, dtype: $0.dtype, shape: $0.shape, physicalShape: $0.physicalShape, logicalShape: $0.logicalShape,
                                 payloadOffset: $0.payloadOffset &+ shift, rawByteLength: $0.rawByteLength, rowStride: $0.rowStride,
                                 alignmentPadding: $0.alignmentPadding, sourceShard: $0.sourceShard)
            }
            header = ConverterHeader(format: tiledFormat(bits: layout.bits), dtype: old.dtype, alignment: old.alignment, configHash: old.configHash,
                                     tokenizerHash: old.tokenizerHash, tensorRecords: moved, headerLength: UInt64(headerData.count),
                                     payloadLength: old.payloadLength, quantization: old.quantization)
            let next = try canonicalJSON(header)
            if next == headerData { break }
            headerData = next
        }
        guard header.headerLength == UInt64(headerData.count) else { throw ConverterError.interruptedWrite }
        let payloadStart = align(16 + UInt64(headerData.count), alignment)
        guard firstOffset &+ shift >= payloadStart else { throw ConverterError.interruptedWrite }

        let fileManager = FileManager.default
        let temp = outputURL.deletingLastPathComponent().appendingPathComponent(".\(outputURL.lastPathComponent).tmp-\(UUID().uuidString)")
        try fileManager.createDirectory(at: outputURL.deletingLastPathComponent(), withIntermediateDirectories: true)
        fileManager.createFile(atPath: temp.path, contents: nil)
        var handle: FileHandle? = try FileHandle(forWritingTo: temp)
        var tiledCount = 0
        do {
            var framing = Data("SPLW".utf8); appendLE(UInt32(1), to: &framing); appendLE(UInt64(headerData.count), to: &framing); framing.append(headerData)
            try handle!.write(contentsOf: framing)
            var written = UInt64(framing.count)
            for record in records {
                let target = record.payloadOffset &+ shift
                guard target >= written else { throw ConverterError.interruptedWrite }
                if target > written { try handle!.write(contentsOf: Data(repeating: 0, count: Int(target - written))) }
                var bytes = try source.readRange(offset: record.payloadOffset, length: record.rawByteLength)
                if let base = record.name.hasSuffix(".weight") ? String(record.name.dropLast(".weight".count)) : nil, let rows = packedRows[base] {
                    bytes = tile(bytes, rows: rows, elementBytes: codeBytes)
                    tiledCount += 1
                } else if let base = sidecarBase(record.name), let rows = packedRows[base] {
                    bytes = tile(bytes, rows: rows, elementBytes: 2)
                }
                guard UInt64(bytes.count) == record.rawByteLength else { throw ConverterError.interruptedWrite }
                try handle!.write(contentsOf: bytes)
                written = target + record.rawByteLength
            }
            guard written == payloadStart + old.payloadLength else { throw ConverterError.interruptedWrite }
            try handle!.synchronize()
            try handle!.close(); handle = nil
            _ = try WeightFile(splwURL: temp)   // framing, bounds and alignment re-validated
            if fileManager.fileExists(atPath: outputURL.path) { try fileManager.removeItem(at: outputURL) }
            try fileManager.moveItem(at: temp, to: outputURL)
        } catch {
            try? handle?.close()
            try? fileManager.removeItem(at: temp)
            throw error
        }
        return RetileReport(format: header.format, outputURL: outputURL, tensorCount: records.count, tiledWeights: tiledCount, byteCount: payloadStart + old.payloadLength)
    }

    /// [row][group][element] -> [tile][group][row in tile][element].
    static func tile(_ data: Data, rows: Int, elementBytes: Int) -> Data {
        let groups = data.count / (rows * elementBytes)
        var out = Data(count: data.count)
        data.withUnsafeBytes { source in
            out.withUnsafeMutableBytes { target in
                let src = source.baseAddress!, dst = target.baseAddress!
                for tile in 0..<(rows / tileRows) {
                    for group in 0..<groups {
                        let base = (tile * groups + group) * tileRows
                        for n in 0..<tileRows {
                            memcpy(dst.advanced(by: (base + n) * elementBytes),
                                   src.advanced(by: ((tile * tileRows + n) * groups + group) * elementBytes), elementBytes)
                        }
                    }
                }
            }
        }
        return out
    }
}
