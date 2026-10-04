// Deterministic packing rules selected by audit/M3-alignment.md.
import Foundation

public enum PackingError: Error, Equatable { case invalidRowBytes }

public struct PackedRows: Sendable {
    public let bytes: Data
    public let rowStride: Int
    public let paddingBytes: Int
}

public enum Packing {
    public static let alignment = 128

    /// M3.6 selected 128-byte padded rows for 4-bit Metal tensors.
    public static func paddedRows(_ rows: [Data], fourBit: Bool = true) throws -> PackedRows {
        guard !rows.isEmpty else { return PackedRows(bytes: Data(), rowStride: 0, paddingBytes: 0) }
        let raw = rows[0].count
        guard rows.allSatisfy({ $0.count == raw }) else { throw PackingError.invalidRowBytes }
        let stride = fourBit ? ((raw + alignment - 1) / alignment) * alignment : raw
        var output = Data(capacity: stride * rows.count)
        for row in rows { output.append(row); output.append(Data(repeating: 0, count: stride - row.count)) }
        return PackedRows(bytes: output, rowStride: stride, paddingBytes: stride - raw)
    }
}
