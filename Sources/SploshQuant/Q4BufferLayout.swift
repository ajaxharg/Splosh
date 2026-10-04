import Foundation

/// Orientation of the logical matrix represented by a packed q4 buffer.
///
/// `.rowsByK` is the MLX affine convention: one physical row contains the K
/// values for one output row. `.kByColumns` is the transposed view used when a
/// kernel addresses the same payload as K-by-column weights.
public enum Q4BufferOrientation: Sendable, Equatable {
    case rowsByK
    case kByColumns
}

public enum Q4BufferLayoutError: Error, Equatable {
    case invalidDimensions
    case invalidRowStride
    case packedLengthMismatch(expected: Int, observed: Int)
    case sidecarLengthMismatch(expected: Int, observedScales: Int, observedBiases: Int)
    case arithmeticOverflow
    case invalidIndex(row: Int, k: Int)
}

/// CPU-visible ABI contract for MLX affine q4 buffers.
///
/// Weights are low-first q4 nibbles in U32 words. Rows are padded independently
/// to `rowStrideBytes` (128-byte alignment by default); padding words are never
/// addressed logically. Scale and bias sidecars are BF16 bit patterns, laid out
/// by row then group. `groupStride` is therefore the number of sidecar entries
/// between adjacent physical rows.
public struct Q4BufferLayout: Sendable, Equatable {
    public static let groupSize = 64
    public static let wordBytes = 4
    public static let defaultRowAlignment = 128

    public let rows: Int
    public let logicalK: Int
    public let orientation: Q4BufferOrientation
    public let packedWordsPerRow: Int
    public let rowStrideBytes: Int
    public let rowStrideWords: Int
    public let paddingWordsPerRow: Int
    public let groupsPerRow: Int
    public let groupStride: Int

    public init(rows: Int, logicalK: Int,
                orientation: Q4BufferOrientation = .rowsByK,
                rowStrideBytes: Int? = nil,
                rowAlignment: Int = Q4BufferLayout.defaultRowAlignment) throws {
        guard rows > 0, logicalK > 0, rowAlignment > 0 else { throw Q4BufferLayoutError.invalidDimensions }
        guard logicalK <= Int.max - 7, rows <= Int.max - 7 else { throw Q4BufferLayoutError.arithmeticOverflow }
        let packed = orientation == .rowsByK ? (logicalK + 7) / 8 : (rows + 7) / 8
        guard packed <= Int.max / Self.wordBytes else { throw Q4BufferLayoutError.arithmeticOverflow }
        let minimum = packed * Self.wordBytes
        let stride: Int
        if let rowStrideBytes { stride = rowStrideBytes }
        else {
            guard minimum <= Int.max - rowAlignment + 1 else { throw Q4BufferLayoutError.arithmeticOverflow }
            let aligned = (minimum + rowAlignment - 1) / rowAlignment
            guard aligned <= Int.max / rowAlignment else { throw Q4BufferLayoutError.arithmeticOverflow }
            stride = aligned * rowAlignment
        }
        guard stride >= minimum, stride % Self.wordBytes == 0 else { throw Q4BufferLayoutError.invalidRowStride }
        guard logicalK <= Int.max - (Self.groupSize - 1) else { throw Q4BufferLayoutError.arithmeticOverflow }
        let groups = (logicalK + Self.groupSize - 1) / Self.groupSize
        let physicalRows = orientation == .rowsByK ? rows : logicalK
        guard physicalRows <= Int.max / (stride / Self.wordBytes), rows <= Int.max / groups else { throw Q4BufferLayoutError.arithmeticOverflow }
        self.rows = rows
        self.logicalK = logicalK
        self.orientation = orientation
        self.packedWordsPerRow = packed
        self.rowStrideBytes = stride
        self.rowStrideWords = stride / Self.wordBytes
        self.paddingWordsPerRow = stride / Self.wordBytes - packed
        self.groupsPerRow = groups
        self.groupStride = self.groupsPerRow
    }

    public var dimensions: (rows: Int, columns: Int) { (rows, logicalK) }
    public var packedWordCount: Int { (orientation == .rowsByK ? rows : logicalK) * rowStrideWords }
    public var sidecarCount: Int { rows * groupStride }

    /// Address of a logical matrix element in the packed U32 payload.
    public func wordIndex(row: Int, k: Int) throws -> Int {
        try validateIndex(row: row, k: k)
        switch orientation {
        case .rowsByK:
            guard row <= Int.max / rowStrideWords else { throw Q4BufferLayoutError.arithmeticOverflow }
            return row * rowStrideWords + k / 8
        case .kByColumns:
            guard k <= Int.max / rowStrideWords else { throw Q4BufferLayoutError.arithmeticOverflow }
            return k * rowStrideWords + row / 8
        }
    }

    public func nibbleShift(row: Int, k: Int) throws -> UInt32 {
        try validateIndex(row: row, k: k)
        let offset = orientation == .rowsByK ? k % 8 : row % 8
        return UInt32(offset * 4)
    }

    public func sidecarIndex(row: Int, k: Int) throws -> Int {
        try validateIndex(row: row, k: k)
        guard row <= Int.max / groupStride else { throw Q4BufferLayoutError.arithmeticOverflow }
        return row * groupStride + k / Self.groupSize
    }

    private func validateIndex(row: Int, k: Int) throws {
        guard row >= 0, row < rows, k >= 0, k < logicalK else {
            throw Q4BufferLayoutError.invalidIndex(row: row, k: k)
        }
    }

    public func decode(row: Int, k: Int, words: [UInt32], scalesBF16: [UInt16], biasesBF16: [UInt16]) throws -> Float {
        guard words.count == packedWordCount else { throw Q4BufferLayoutError.packedLengthMismatch(expected: packedWordCount, observed: words.count) }
        guard scalesBF16.count == sidecarCount, biasesBF16.count == sidecarCount else {
            throw Q4BufferLayoutError.sidecarLengthMismatch(expected: sidecarCount, observedScales: scalesBF16.count, observedBiases: biasesBF16.count)
        }
        let word = words[try wordIndex(row: row, k: k)]
        let q = (word >> (try nibbleShift(row: row, k: k))) & 0xF
        let group = try sidecarIndex(row: row, k: k)
        return Self.bf16(scalesBF16[group]) * Float(q) + Self.bf16(biasesBF16[group])
    }

    public static func bf16(_ bits: UInt16) -> Float {
        Float(bitPattern: UInt32(bits) << 16)
    }

    public static func bf16Bits(_ value: Float) -> UInt16 {
        UInt16((value.bitPattern + 0x8000) >> 16)
    }
}
