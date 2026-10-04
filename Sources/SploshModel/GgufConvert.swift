// GgufConvert.swift — a llama.cpp GGUF file as an SPLW artifact the engine maps.
//
// The artifact is the one ModelWeights.swift resolves: the SPLW framing of Converter.swift, the
// MLX pack's tensor names and shapes, and the engine's order of the gated-delta value heads. What
// differs is how a packed weight is held. Its codes stay in the file's format, at the file's
// size, as the three planes of GgufPlanes.swift, a record each:
//
//   <base>.weight   plane 0, with the weight's ggml type (`ggufType`) and its logical shape
//   <base>.plane1   plane 1; no record for a format without one
//   <base>.meta     the meta plane
//
// all U8 of shape [stored rows, bytes a row], the stored rows being the weight's rounded up to
// whole tiles of 128: the 48-row projections of a gated-delta layer are padded with zero blocks.
// Every 2-D weight goes this way, the embedding included. The header's `format` is
// `gguf.tiled128` and its quantization `ConverterQuantization.gguf`.
//
// The file orders the 48 value heads of a gated-delta layer differently from the engine: its
// head i is the engine's 3 (i % 16) + i / 16 (GgufNames.swift). Where a head is rows of a weight
// the rows are moved, which is lossless, a row being whole native blocks; where it is channels or
// elements of a dense tensor those are moved. The out projection has the heads as its columns,
// and moving columns would mean re-quantising every block, so it alone is left in the file's
// order: its kernels read the activations in that order instead (the `heads` kernels of
// Sources/Shaders/engine_gguf.metal).
//
// The dense tensors, fp32 in the file, are rounded to BF16, to nearest and ties to even: the
// norm weights (the +1 already in them, as in the MLX pack), the convolution as
// [channels, 4, 1], the time-step bias, and the decay, which the file holds as -exp(A_log) and
// the engine reads as A_log.
//
// `configHash` is the SHA-256 of the file's header, everything before its tensor data: the
// metadata and the tensor directory, so two quantisations of one model differ in it. The engine
// keys stored prefix snapshots by it (Engine.snapshotIdentity). `tokenizerHash` is that of the
// tokenizer.json the artifact is served with, the hash an MLX artifact of the same model has.

import CryptoKit
import Foundation

public struct GgufConversionReport: Sendable, Equatable {
    public let outputURL: URL
    /// The tensors converted, by their type in the file: the packed weights by format, the
    /// dense tensors under their float type.
    public let tensorsByType: [GgufTensorType: Int]
    /// The bytes written for them, by the same key.
    public let bytesByType: [GgufTensorType: UInt64]
    /// Tensors of the MTP block, which the engine does not run, and tensors the filter left out.
    public let mtpTensors: Int
    public let filteredTensors: Int
    public let recordCount: Int
    /// The size of the artifact.
    public let byteCount: UInt64
    public let header: ConverterHeader

    public var tensorCount: Int { tensorsByType.values.reduce(0, +) }
}

/// Where a tensor has the value heads of a gated-delta layer: `span` units a head from unit
/// `first`, a unit being a row of a weight, a channel of the convolution or an element of a
/// per-head vector.
private struct HeadOrder {
    let first: Int
    let span: Int

    /// The place in the engine's order of unit `unit` of the file's.
    func destination(of unit: Int, heads: Int) -> Int {
        guard unit >= first, (unit - first) / span < heads else { return unit }
        return first + GgufNames.mlxValueHead(ofGgufHead: (unit - first) / span) * span + (unit - first) % span
    }
}

private enum GgufRole {
    /// A packed weight of `rows` x `inner`, to planes; `heads` when its rows are by value head.
    case weight(rows: Int, inner: Int, heads: HeadOrder?)
    /// A dense tensor, to BF16 of `shape`, in units of `unit` values; `negatedLog` for the decay.
    case dense(shape: [Int], unit: Int, heads: HeadOrder?, negatedLog: Bool)
}

private struct GgufSource {
    let tensor: GgufFile.Tensor
    /// The artifact's name for it: Splosh's base name, with the `.weight` the file's name has.
    let name: String
    let role: GgufRole
}

/// One tensor converted, in memory.
private enum GgufConverted {
    case planes(GgufPlanes.Planes)
    case dense([UInt8])
}

private enum GgufPiece { case plane0, plane1, meta, dense }

public extension Converter {
    /// The header's `format` for an artifact converted from a GGUF file.
    static let ggufFormat = "gguf.tiled128"

    /// Convert a GGUF file to an artifact.
    ///
    /// One tensor is held at a time, and the artifact is written beside `outputURL` and renamed
    /// into place once it reopens, so a failure leaves whatever was there. `include`, given a
    /// tensor's name in the file, says whether to convert it; without it every tensor is, bar
    /// those of the MTP block. A tensor that is included must be one the model has, of the
    /// model's shape and of a type its role can take: anything else throws
    /// `ConverterError.invalidSource` naming the tensor.
    static func convertGguf(inputURL: URL, outputURL: URL, tokenizerURL: URL,
                            include: ((String) -> Bool)? = nil) throws -> GgufConversionReport {
        guard FileManager.default.fileExists(atPath: inputURL.path) else { throw ConverterError.missingSource(inputURL) }
        let file = try GgufFile(url: inputURL)
        let geometry = ModelGeometry()
        let plan = try ggufPlan(file, geometry, include: include)
        let configHash = try sha256(of: inputURL, prefix: file.dataStart)
        let tokenizerHash = try sha256(of: tokenizerURL, prefix: nil)

        // Every record's size follows from the directory, so the header is whole before any
        // tensor is read. Records are in name order, which keeps a tensor's together.
        let alignment = ConverterAlignment()
        let shard = inputURL.lastPathComponent
        var entries: [(record: SPLWTensorRecord, source: Int, piece: GgufPiece)] = []
        for (index, source) in plan.sources.enumerated() {
            func add(_ name: String, _ piece: GgufPiece, dtype: TensorDType, shape: [Int], bytes: Int,
                     logicalShape: [Int]? = nil, ggufType: UInt32? = nil) {
                let rowBytes = shape.count == 2 ? bytes / max(shape[0], 1) : bytes
                let stride = (rowBytes + alignment.rowStrideBytes - 1) / alignment.rowStrideBytes * alignment.rowStrideBytes
                entries.append((SPLWTensorRecord(name: name, dtype: dtype, shape: shape, logicalShape: logicalShape, payloadOffset: 0,
                                                 rawByteLength: UInt64(bytes), rowStride: UInt64(stride), alignmentPadding: 0,
                                                 sourceShard: shard, ggufType: ggufType), index, piece))
            }
            switch source.role {
            case .weight(let rows, let inner, _):
                let stored = storedRows(rows)
                guard let sizes = GgufPlanes.sizes(of: source.tensor.type, rows: stored, inner: inner) else {
                    throw ConverterError.invalidSource("tensor \(source.tensor.name): \(source.tensor.type) \(rows) x \(inner) has no plane form")
                }
                let base = String(source.name.dropLast(".weight".count))
                add(source.name, .plane0, dtype: .u8, shape: [stored, sizes.plane0 / stored], bytes: sizes.plane0,
                    logicalShape: [rows, inner], ggufType: source.tensor.type.rawValue)
                if sizes.plane1 > 0 { add(base + ".plane1", .plane1, dtype: .u8, shape: [stored, sizes.plane1 / stored], bytes: sizes.plane1) }
                add(base + ".meta", .meta, dtype: .u8, shape: [stored, sizes.meta / stored], bytes: sizes.meta)
            case .dense(let shape, _, _, _):
                add(source.name, .dense, dtype: .bf16, shape: shape, bytes: 2 * source.tensor.elementCount)
            }
        }
        entries.sort { $0.record.name < $1.record.name }
        var records = entries.map(\.record)

        // The header's length depends on the offsets it holds; iterate to a fixed point.
        let base: UInt64 = 16
        let pointerAlignment = UInt64(alignment.pointerAlignmentBytes)
        var header = ConverterHeader(format: ggufFormat, dtype: "MIXED", alignment: alignment, configHash: configHash,
                                     tokenizerHash: tokenizerHash, tensorRecords: records, quantization: .gguf)
        var headerData = try canonicalJSON(header)
        for _ in 0..<8 {
            let payloadStart = align(base + UInt64(headerData.count), pointerAlignment)
            var cursor = payloadStart
            for i in records.indices {
                let next = align(cursor, pointerAlignment), old = records[i]
                records[i] = SPLWTensorRecord(name: old.name, dtype: old.dtype, shape: old.shape, physicalShape: old.physicalShape,
                                              logicalShape: old.logicalShape, payloadOffset: next, rawByteLength: old.rawByteLength,
                                              rowStride: old.rowStride, alignmentPadding: next - cursor, sourceShard: old.sourceShard,
                                              ggufType: old.ggufType)
                cursor = next + old.rawByteLength
            }
            header = ConverterHeader(format: ggufFormat, dtype: "MIXED", alignment: alignment, configHash: configHash,
                                     tokenizerHash: tokenizerHash, tensorRecords: records, headerLength: UInt64(headerData.count),
                                     payloadLength: cursor - payloadStart, quantization: .gguf)
            let nextData = try canonicalJSON(header)
            if nextData == headerData { break }
            headerData = nextData
        }
        let payloadStart = align(base + UInt64(headerData.count), pointerAlignment)
        guard header.headerLength == UInt64(headerData.count) else { throw ConverterError.interruptedWrite }

        let fileManager = FileManager.default
        let temp = outputURL.deletingLastPathComponent().appendingPathComponent(".\(outputURL.lastPathComponent).tmp-\(UUID().uuidString)")
        var handle: FileHandle?
        var tensorsByType: [GgufTensorType: Int] = [:], bytesByType: [GgufTensorType: UInt64] = [:]
        for source in plan.sources { tensorsByType[source.tensor.type, default: 0] += 1 }
        do {
            try fileManager.createDirectory(at: outputURL.deletingLastPathComponent(), withIntermediateDirectories: true)
            fileManager.createFile(atPath: temp.path, contents: nil)
            handle = try FileHandle(forWritingTo: temp)
            // Past the buffer cache: the artifact is the size of the memory a running server
            // leaves free, and nothing here reads it back but its header.
            _ = fcntl(handle!.fileDescriptor, F_NOCACHE, 1)
            var framing = Data("SPLW".utf8); appendLE(UInt32(1), to: &framing); appendLE(UInt64(headerData.count), to: &framing); framing.append(headerData)
            try handle!.write(contentsOf: framing)
            var written = UInt64(framing.count)
            var current: (source: Int, converted: GgufConverted)?
            for (i, record) in records.enumerated() {
                let entry = entries[i]
                if current?.source != entry.source {
                    // The last tensor goes before the next is made.
                    current = nil
                    current = (entry.source, try converted(plan.sources[entry.source], in: file, geometry))
                }
                guard record.payloadOffset >= written, let converted = current?.converted else { throw ConverterError.interruptedWrite }
                if record.payloadOffset > written { try handle!.write(contentsOf: Data(repeating: 0, count: Int(record.payloadOffset - written))) }
                let bytes: [UInt8]
                switch (converted, entry.piece) {
                case (.planes(let planes), .plane0): bytes = planes.plane0
                case (.planes(let planes), .plane1): bytes = planes.plane1
                case (.planes(let planes), .meta): bytes = planes.meta
                case (.dense(let dense), .dense): bytes = dense
                default: throw ConverterError.interruptedWrite
                }
                guard UInt64(bytes.count) == record.rawByteLength else { throw ConverterError.interruptedWrite }
                try bytes.withUnsafeBytes { raw in
                    guard let start = raw.baseAddress else { return }
                    try handle!.write(contentsOf: Data(bytesNoCopy: UnsafeMutableRawPointer(mutating: start), count: raw.count, deallocator: .none))
                }
                written = record.payloadOffset + record.rawByteLength
                bytesByType[plan.sources[entry.source].tensor.type, default: 0] += record.rawByteLength
            }
            guard written == payloadStart + header.payloadLength else { throw ConverterError.interruptedWrite }
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
        return GgufConversionReport(outputURL: outputURL, tensorsByType: tensorsByType, bytesByType: bytesByType,
                                    mtpTensors: plan.mtp, filteredTensors: plan.filtered, recordCount: records.count,
                                    byteCount: payloadStart + header.payloadLength, header: header)
    }

    /// The rows a weight's planes hold: its own, rounded up to whole tiles.
    internal static func storedRows(_ rows: Int) -> Int {
        (rows + GgufPlanes.tileRows - 1) / GgufPlanes.tileRows * GgufPlanes.tileRows
    }

    /// An fp32 as BF16, rounded to nearest and ties to even.
    internal static func bf16Bits(roundingToEven value: Float) -> UInt16 {
        let bits = value.bitPattern
        return UInt16(truncatingIfNeeded: (bits &+ 0x7FFF &+ ((bits >> 16) & 1)) >> 16)
    }

    /// The tensors to convert, each with what the model makes of it, in the order of their
    /// names in the artifact.
    private static func ggufPlan(_ file: GgufFile, _ g: ModelGeometry,
                                 include: ((String) -> Bool)?) throws -> (sources: [GgufSource], mtp: Int, filtered: Int) {
        let mtpBlock = GgufNames.mtpBlock(among: file.tensors.map(\.name))
        var sources: [GgufSource] = [], mtp = 0, filtered = 0
        for tensor in file.tensors {
            if let include, !include(tensor.name) { filtered += 1; continue }
            if GgufNames.belongsToMtp(tensor.name, mtpBlock: mtpBlock) { mtp += 1; continue }
            guard let base = GgufNames.sploshName(forGguf: tensor.name) else {
                throw ConverterError.invalidSource("tensor \(tensor.name) has no counterpart in the model")
            }
            let role: GgufRole?
            if let block = GgufNames.blockIndex(of: tensor.name) {
                guard block < g.layers else {
                    throw ConverterError.invalidSource("tensor \(tensor.name) is of block \(block); the model has \(g.layers) layers")
                }
                let fullAttention = block % 4 == 3
                role = ggufRole(ofBlockTensor: String(tensor.name.dropFirst("blk.\(block).".count)), fullAttention: fullAttention, g)
                guard role != nil else {
                    throw ConverterError.invalidSource("tensor \(tensor.name) has no counterpart: layer \(block) is a \(fullAttention ? "full-attention" : "gated-delta") layer")
                }
            } else {
                role = ggufRole(ofGlobalTensor: tensor.name, g)
            }
            switch role {
            case .weight(let rows, let inner, _):
                guard tensor.dims == [inner, rows] else {
                    throw ConverterError.invalidSource("tensor \(tensor.name) has dims \(tensor.dims); the model's \(base) is \(rows) x \(inner), dims [\(inner), \(rows)]")
                }
                guard GgufPlanes.sizes(of: tensor.type, rows: storedRows(rows), inner: inner) != nil else {
                    throw ConverterError.invalidSource("tensor \(tensor.name) is \(tensor.type), a type with no plane form for a \(rows) x \(inner) weight")
                }
            case .dense(let shape, _, _, _):
                // The file's extents are innermost first and it keeps none of 1.
                let dims = Array(shape.reversed().filter { $0 != 1 })
                guard tensor.dims == dims else {
                    throw ConverterError.invalidSource("tensor \(tensor.name) has dims \(tensor.dims); the model's \(base) is \(shape), dims \(dims)")
                }
                guard [.f32, .f16, .bf16].contains(tensor.type) else {
                    throw ConverterError.invalidSource("tensor \(tensor.name) is \(tensor.type); a dense tensor must be of a float type")
                }
            case nil:
                throw ConverterError.invalidSource("tensor \(tensor.name) has no counterpart in the model")
            }
            sources.append(GgufSource(tensor: tensor, name: base + (tensor.name.hasSuffix(".weight") ? ".weight" : ""), role: role!))
        }
        guard !sources.isEmpty else {
            throw ConverterError.invalidSource("\(file.url.lastPathComponent) has no tensor to convert")
        }
        sources.sort { $0.name < $1.name }
        return (sources, mtp, filtered)
    }

    /// What the model makes of the tensor `tail` of a block, its name below `blk.<n>.`; nil when
    /// a layer of that kind has no such tensor. The shapes are those ModelWeights.swift expects.
    private static func ggufRole(ofBlockTensor tail: String, fullAttention: Bool, _ g: ModelGeometry) -> GgufRole? {
        func dense(_ shape: [Int], unit: Int = 1, heads: HeadOrder? = nil, negatedLog: Bool = false) -> GgufRole {
            .dense(shape: shape, unit: unit, heads: heads, negatedLog: negatedLog)
        }
        switch tail {
        case "attn_norm.weight", "post_attention_norm.weight": return dense([g.hidden])
        case "ffn_gate.weight", "ffn_up.weight": return .weight(rows: g.intermediate, inner: g.hidden, heads: nil)
        case "ffn_down.weight": return .weight(rows: g.hidden, inner: g.intermediate, heads: nil)
        default: break
        }
        if fullAttention {
            switch tail {
            case "attn_q.weight": return .weight(rows: g.heads * g.headDim * 2, inner: g.hidden, heads: nil)
            case "attn_k.weight", "attn_v.weight": return .weight(rows: g.kvHeads * g.headDim, inner: g.hidden, heads: nil)
            case "attn_output.weight": return .weight(rows: g.hidden, inner: g.heads * g.headDim, heads: nil)
            case "attn_q_norm.weight", "attn_k_norm.weight": return dense([g.headDim])
            default: return nil
            }
        }
        // The channels of a gated-delta layer are its query and key heads, then its value heads.
        let valueChannels = HeadOrder(first: g.gdnChannels - g.gdnValueDim, span: g.gdnHeadDim)
        switch tail {
        case "attn_qkv.weight": return .weight(rows: g.gdnChannels, inner: g.hidden, heads: valueChannels)
        case "attn_gate.weight": return .weight(rows: g.gdnValueDim, inner: g.hidden, heads: HeadOrder(first: 0, span: g.gdnHeadDim))
        case "ssm_alpha.weight", "ssm_beta.weight": return .weight(rows: g.gdnValueHeads, inner: g.hidden, heads: HeadOrder(first: 0, span: 1))
        // The heads are this weight's columns, which stay where they are.
        case "ssm_out.weight": return .weight(rows: g.hidden, inner: g.gdnValueDim, heads: nil)
        case "ssm_conv1d.weight": return dense([g.gdnChannels, convTaps, 1], unit: convTaps, heads: valueChannels)
        case "ssm_a": return dense([g.gdnValueHeads], heads: HeadOrder(first: 0, span: 1), negatedLog: true)
        case "ssm_dt.bias": return dense([g.gdnValueHeads], heads: HeadOrder(first: 0, span: 1))
        case "ssm_norm.weight": return dense([g.gdnHeadDim])
        default: return nil
        }
    }

    private static func ggufRole(ofGlobalTensor name: String, _ g: ModelGeometry) -> GgufRole? {
        switch name {
        case "token_embd.weight", "output.weight": return .weight(rows: g.vocab, inner: g.hidden, heads: nil)
        case "output_norm.weight": return .dense(shape: [g.hidden], unit: 1, heads: nil, negatedLog: false)
        default: return nil
        }
    }

    /// The taps of a gated-delta layer's convolution.
    private static let convTaps = 4

    private static func converted(_ source: GgufSource, in file: GgufFile, _ g: ModelGeometry) throws -> GgufConverted {
        switch source.role {
        case .weight(let rows, let inner, let heads):
            return .planes(ggufPlanes(of: source.tensor, in: file, rows: rows, inner: inner, heads: heads, valueHeads: g.gdnValueHeads))
        case .dense(_, let unit, let heads, let negatedLog):
            return .dense(try ggufDense(of: source.tensor, in: file, unit: unit, heads: heads, valueHeads: g.gdnValueHeads, negatedLog: negatedLog))
        }
    }

    /// A weight's planes: its rows put in the engine's order and padded to whole tiles where
    /// either is needed, then repacked a tile a task.
    private static func ggufPlanes(of tensor: GgufFile.Tensor, in file: GgufFile, rows: Int, inner: Int,
                                   heads: HeadOrder?, valueHeads: Int) -> GgufPlanes.Planes {
        let type = tensor.type, rowBytes = tensor.rowBytes, stored = storedRows(rows)
        guard let sizes = GgufPlanes.sizes(of: type, rows: stored, inner: inner) else {
            preconditionFailure("\(tensor.name): \(type) \(rows) x \(inner) has no plane form")
        }
        let native = file.bytes(of: tensor)
        var ordered: [UInt8] = []
        if heads != nil || stored != rows {
            ordered = [UInt8](repeating: 0, count: stored * rowBytes)
            ordered.withUnsafeMutableBytes { target in
                for row in 0..<rows {
                    let to = heads?.destination(of: row, heads: valueHeads) ?? row
                    target.baseAddress!.advanced(by: to * rowBytes).copyMemory(from: native.baseAddress!.advanced(by: row * rowBytes), byteCount: rowBytes)
                }
            }
        }
        var plane0 = [UInt8](repeating: 0, count: sizes.plane0), plane1 = [UInt8](repeating: 0, count: sizes.plane1)
        var meta = [UInt8](repeating: 0, count: sizes.meta)
        let tiles = stored / GgufPlanes.tileRows
        func repack(_ rowsInOrder: UnsafeRawBufferPointer) {
            plane0.withUnsafeMutableBytes { plane0 in
                plane1.withUnsafeMutableBytes { plane1 in
                    meta.withUnsafeMutableBytes { meta in
                        // A tile's records are together in each plane, so the tiles are independent
                        // and each task writes bytes no other touches.
                        let tileNative = GgufPlanes.tileRows * rowBytes
                        let tile0 = sizes.plane0 / tiles, tile1 = sizes.plane1 / tiles, tileMeta = sizes.meta / tiles
                        nonisolated(unsafe) let rowsInOrder = rowsInOrder, plane0 = plane0, plane1 = plane1, meta = meta
                        DispatchQueue.concurrentPerform(iterations: tiles) { tile in
                            GgufPlanes.repack(type, native: UnsafeRawBufferPointer(rebasing: rowsInOrder[tile * tileNative ..< (tile + 1) * tileNative]),
                                              rows: GgufPlanes.tileRows, inner: inner,
                                              plane0: UnsafeMutableRawBufferPointer(rebasing: plane0[tile * tile0 ..< (tile + 1) * tile0]),
                                              plane1: UnsafeMutableRawBufferPointer(rebasing: plane1[tile * tile1 ..< (tile + 1) * tile1]),
                                              meta: UnsafeMutableRawBufferPointer(rebasing: meta[tile * tileMeta ..< (tile + 1) * tileMeta]))
                        }
                    }
                }
            }
        }
        if ordered.isEmpty { repack(native) } else { ordered.withUnsafeBytes(repack) }
        return GgufPlanes.Planes(plane0: plane0, plane1: plane1, meta: meta)
    }

    /// A dense tensor as BF16 bytes: decoded to fp32, the decay turned back into A_log, its
    /// units put in the engine's order, each value rounded.
    private static func ggufDense(of tensor: GgufFile.Tensor, in file: GgufFile, unit: Int, heads: HeadOrder?,
                                  valueHeads: Int, negatedLog: Bool) throws -> [UInt8] {
        var values = [Float](repeating: 0, count: tensor.elementCount)
        values.withUnsafeMutableBufferPointer { GgufTensorType.decode(tensor.type, blocks: file.bytes(of: tensor), into: $0) }
        if negatedLog {
            // The file holds -exp(A_log), which is negative whatever A_log is.
            guard values.allSatisfy({ $0 < 0 }) else {
                throw ConverterError.invalidSource("tensor \(tensor.name) holds a value that is not negative, so is not -exp of anything")
            }
            values = values.map { Foundation.log(-$0) }
        }
        guard values.allSatisfy(\.isFinite) else {
            throw ConverterError.invalidSource("tensor \(tensor.name) holds a value that is not finite")
        }
        var bytes = [UInt8](repeating: 0, count: 2 * values.count)
        for index in values.indices {
            let to = (heads?.destination(of: index / unit, heads: valueHeads) ?? index / unit) * unit + index % unit
            let bits = bf16Bits(roundingToEven: values[index])
            bytes[2 * to] = UInt8(bits & 0xFF)
            bytes[2 * to + 1] = UInt8(bits >> 8)
        }
        return bytes
    }

    /// The SHA-256 of a file, or of its first `prefix` bytes, in hex.
    private static func sha256(of url: URL, prefix: Int?) throws -> String {
        guard let handle = try? FileHandle(forReadingFrom: url) else { throw ConverterError.missingSource(url) }
        defer { try? handle.close() }
        var hasher = SHA256()
        var remaining = prefix ?? Int.max
        while remaining > 0 {
            guard let chunk = try handle.read(upToCount: min(remaining, 1 << 22)), !chunk.isEmpty else { break }
            hasher.update(data: chunk)
            remaining -= chunk.count
        }
        guard prefix == nil || remaining == 0 else { throw ConverterError.invalidSource("\(url.lastPathComponent) is shorter than its header") }
        return hasher.finalize().map { String(format: "%02x", $0) }.joined()
    }
}
