// Gemm.swift — SploshModel.
//
// Runtime-neutral GEMM shape and q4 ABI description. Metal dispatch lives in SploshRuntime.

import SploshQuant

/// Row-major GEMM dimensions: A[M,K] × dequant(Q)[K,N] = C[M,N].
public struct GemmQ4Shape: Sendable, Equatable {
    public let rows: Int
    public let columns: Int
    public let inner: Int

    public init(rows: Int, columns: Int, inner: Int) throws {
        guard rows > 0, columns > 0, inner > 0 else { throw GemmQ4Error.invalidShape }
        self.rows = rows; self.columns = columns; self.inner = inner
    }

    /// Validates the packed-right-operand layout used by `gemm_q4`.
    public func validate(layout: Q4BufferLayout) throws {
        guard layout.rows == columns, layout.logicalK == inner else {
            throw GemmQ4Error.layoutMismatch
        }
        guard layout.orientation == .rowsByK else {
            throw GemmQ4Error.orientationMismatch
        }
        guard layout.rowStrideBytes >= Q4BufferLayout.defaultRowAlignment,
              layout.rowStrideBytes % Q4BufferLayout.defaultRowAlignment == 0,
              layout.rowStrideBytes % Q4BufferLayout.wordBytes == 0,
              layout.rowStrideBytes >= layout.packedWordsPerRow * Q4BufferLayout.wordBytes else {
            throw GemmQ4Error.invalidRowStride
        }
        guard Q4BufferLayout.groupSize == 64 else {
            throw GemmQ4Error.invalidGroupSize
        }
        guard inner <= Int.max - (Q4BufferLayout.groupSize - 1) else { throw GemmQ4Error.invalidShape }
        guard layout.groupsPerRow == (inner + Q4BufferLayout.groupSize - 1) / Q4BufferLayout.groupSize,
              layout.groupStride == layout.groupsPerRow else {
            throw GemmQ4Error.sidecarGeometryMismatch
        }
    }
}

public enum GemmQ4Error: Error, Sendable, Equatable {
    case invalidShape
    case layoutMismatch
    case invalidRowStride
    case orientationMismatch
    case invalidGroupSize
    case sidecarGeometryMismatch
    case sidecarMismatch
    case bufferTooSmall(name: String, expected: Int, observed: Int)
}
