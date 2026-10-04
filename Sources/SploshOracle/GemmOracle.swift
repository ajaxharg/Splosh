// GemmOracle.swift — independent scalar bf16 GEMM reference.
// Operands are bf16 payloads and every product is accumulated in Float32.

public enum GemmOracle {
    /// Computes row-major `C = A × B`, with A[M,K], B[K,N], and C[M,N].
    public static func bf16(
        a: [UInt16],
        b: [UInt16],
        rows: Int,
        columns: Int,
        inner: Int
    ) -> [Float] {
        precondition(rows >= 0 && columns >= 0 && inner >= 0)
        precondition(a.count == rows * inner, "A shape does not match rows × inner")
        precondition(b.count == inner * columns, "B shape does not match inner × columns")
        var output = Array(repeating: Float.zero, count: rows * columns)
        for row in 0..<rows {
            for column in 0..<columns {
                var sum: Float = 0
                for k in 0..<inner {
                    let lhs = Float(bitPattern: UInt32(a[row * inner + k]) << 16)
                    let rhs = Float(bitPattern: UInt32(b[k * columns + column]) << 16)
                    sum += lhs * rhs
                }
                output[row * columns + column] = sum
            }
        }
        return output
    }
}
