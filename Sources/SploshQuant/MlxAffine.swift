// MLX affine q4 layout: U32 words, 8 low-first nibbles per word, group size 64.
import Foundation

public enum MlxAffineError: Error, Equatable, CustomStringConvertible {
    case invalidShape
    case packedLengthMismatch(expected: Int, observed: Int)
    case sidecarLengthMismatch
    case physicalShapeOverflow
    case partialGroup
    public var description: String {
        switch self {
        case .invalidShape: return "MLX affine tensor shape must contain positive rows and columns"
        case .packedLengthMismatch(let e, let o): return "MLX packed length mismatch: expected \(e), observed \(o)"
        case .sidecarLengthMismatch: return "MLX affine sidecar length mismatch"
        case .physicalShapeOverflow: return "MLX affine physical shape overflow"
        case .partialGroup: return "MLX affine physical columns must contain complete 64-value groups"
        }
    }
}

public struct MlxAffine: Sendable {
    public static let groupSize = 64
    public let rows: Int
    public let columns: Int
    public let words: [UInt32]
    public let scales: [Float]
    public let biases: [Float]

    /// Compatibility initializer for callers that provide the logical scalar shape.
    /// New safetensors callers must use `init(physicalShape:...)`.
    public init(shape: [Int], words: [UInt32], scales: [Float], biases: [Float]) throws {
        guard shape.count == 2, shape.allSatisfy({ $0 > 0 }) else { throw MlxAffineError.invalidShape }
        let rows = shape[0], columns = shape[1]
        let (rounded, overflow) = columns.addingReportingOverflow(7)
        guard !overflow else { throw MlxAffineError.physicalShapeOverflow }
        let wordColumns = rounded / 8
        let (expected, expectedOverflow) = rows.multipliedReportingOverflow(by: wordColumns)
        guard !expectedOverflow else { throw MlxAffineError.physicalShapeOverflow }
        guard words.count == expected else { throw MlxAffineError.packedLengthMismatch(expected: expected, observed: words.count) }
        let (groupNumerator, groupNumeratorOverflow) = columns.addingReportingOverflow(Self.groupSize - 1)
        guard !groupNumeratorOverflow else { throw MlxAffineError.physicalShapeOverflow }
        let groupsPerRow = groupNumerator / Self.groupSize
        let (groups, groupsOverflow) = rows.multipliedReportingOverflow(by: groupsPerRow)
        guard !groupsOverflow else { throw MlxAffineError.physicalShapeOverflow }
        guard scales.count == groups, biases.count == groups else { throw MlxAffineError.sidecarLengthMismatch }
        self.rows = rows; self.columns = columns; self.words = words; self.scales = scales; self.biases = biases
    }

    /// Converts the physical MLX safetensors representation into logical dimensions.
    /// `physicalShape` is `[rows, packedWordColumns]`; each word contains eight
    /// low-first nibbles and each sidecar group describes 64 logical columns.
    public init(physicalShape: [Int], packedWords: [UInt32], scales: [Float], biases: [Float]) throws {
        guard physicalShape.count == 2, physicalShape.allSatisfy({ $0 > 0 }) else { throw MlxAffineError.invalidShape }
        let rows = physicalShape[0], wordColumns = physicalShape[1]
        guard wordColumns % 8 == 0 else { throw MlxAffineError.partialGroup }
        let (expected, expectedOverflow) = rows.multipliedReportingOverflow(by: wordColumns)
        guard !expectedOverflow else { throw MlxAffineError.physicalShapeOverflow }
        guard packedWords.count == expected else { throw MlxAffineError.packedLengthMismatch(expected: expected, observed: packedWords.count) }
        let (logicalColumns, logicalOverflow) = wordColumns.multipliedReportingOverflow(by: 8)
        guard !logicalOverflow else { throw MlxAffineError.physicalShapeOverflow }
        let groupsPerRow = wordColumns / 8
        let (groups, groupsOverflow) = rows.multipliedReportingOverflow(by: groupsPerRow)
        guard !groupsOverflow else { throw MlxAffineError.physicalShapeOverflow }
        guard scales.count == groups, biases.count == groups else { throw MlxAffineError.sidecarLengthMismatch }
        self.rows = rows; self.columns = logicalColumns; self.words = packedWords; self.scales = scales; self.biases = biases
    }

    public static func decode(physicalShape: [Int], packedWords: [UInt32], scales: [Float], biases: [Float]) throws -> MlxAffine {
        try MlxAffine(physicalShape: physicalShape, packedWords: packedWords, scales: scales, biases: biases)
    }

    public func unpack(row: Int) -> [UInt8] {
        precondition(row >= 0 && row < rows)
        let wordsPerRow = (columns + 7) / 8
        return (0..<columns).map { c in UInt8((words[row * wordsPerRow + c / 8] >> UInt32(4 * (c % 8))) & 0xF) }
    }

    public func dequantized(row: Int) -> [Float] {
        let q = unpack(row: row)
        let groupsPerRow = (columns + Self.groupSize - 1) / Self.groupSize
        return q.enumerated().map { c, value in
            let group = row * groupsPerRow + c / Self.groupSize
            return scales[group] * Float(value) + biases[group]
        }
    }
}
