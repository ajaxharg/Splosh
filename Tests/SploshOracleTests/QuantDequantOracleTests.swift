import Testing
import SploshOracle
import SploshQuant

@Suite("QuantDequantOracleTests")
struct QuantDequantOracleTests {
    @Test("q4_0 uses GGML low/high nibble pair layout and fp16 scale")
    func q4_0Contract() {
        let values = DequantOracle.q4_0(data: [0x00, 0x3c] + Array(repeating: 0x10, count: 16))
        #expect(values.count == 32)
        #expect(values[0] == -8)
        #expect(values[1] == -7)
        #expect(values[16] == -8)
        #expect(values[31] == -7)
    }

    @Test("q4_K decodes eight affine groups with packed scale/min metadata")
    func q4_KContract() {
        // d=1, dmin=1; each group gets scale=1 and min=0. Nibbles 0..15 repeat.
        let metadata = [UInt8](repeating: 1, count: 8) + [0, 0, 0, 0]
        let payload = [UInt8](repeating: 0x10, count: 128)
        let values = DequantOracle.q4_K(data: [0x00, 0x3c, 0x00, 0x00] + metadata + payload)
        #expect(values.count == 256)
        #expect(values[0] == 0)
        #expect(values[1] == 0)
        #expect(values[16] == 1)
        #expect(values[31] == 1)
        #expect(values[32] == 1)
    }

    @Test("MLX affine q4 uses low-first group-64 nibbles and affine sidecars")
    func mlxAffineGroup64Contract() throws {
        let columns = 65
        let wordsPerRow = (columns + 7) / 8
        var words = Array(repeating: UInt32.zero, count: wordsPerRow)
        for column in 0..<columns {
            words[column / 8] |= UInt32(column % 16) << UInt32((column % 8) * 4)
        }
        let tensor = try MlxAffine(shape: [1, columns], words: words,
                                   scales: [2, 3], biases: [-1, 10])
        let expected = (0..<columns).map { c in
            let q = Float(c % 16)
            return Float(c < 64 ? 2 : 3) * q + Float(c < 64 ? -1 : 10)
        }
        #expect(tensor.unpack(row: 0).count == columns)
        #expect(DequantOracle.dequantize(tensor) == expected)
        #expect(tensor.dequantized(row: 0) == expected)
    }

    @Test("MLX affine q4 supports non-32 row counts and explicit dequantize-then-GEMM")
    func mlxAffineGemmOracleContract() throws {
        let rows = 3, inner = 65, columns = 2
        let wordsPerRow = (inner + 7) / 8
        var words: [UInt32] = []
        for row in 0..<rows {
            for wordIndex in 0..<wordsPerRow {
                var word: UInt32 = 0
                for nibble in 0..<8 {
                    let k = wordIndex * 8 + nibble
                    if k < inner { word |= UInt32((k + row) % 16) << UInt32(nibble * 4) }
                }
                words.append(word)
            }
        }
        let scales = (0..<(rows * 2)).map { _ in Float(1) }
        let biases = (0..<(rows * 2)).map { _ in Float(0) }
        let tensor = try MlxAffine(shape: [rows, inner], words: words, scales: scales, biases: biases)
        let weights = DequantOracle.dequantize(tensor)
        let a = (0..<(rows * inner)).map { UInt16(Float($0 % 7) / 2).littleEndian }
        let b = (0..<inner * columns).map { index -> UInt16 in
            let value = weights[index % inner]
            let bits = value.bitPattern
            return UInt16((bits + 0x8000) >> 16)
        }
        let expected = GemmOracle.bf16(a: a, b: b, rows: rows, columns: columns, inner: inner)
        #expect(weights.count == rows * inner)
        #expect(expected.count == rows * columns)
        let isFinite: (Float) -> Bool = { value in value.isFinite }
        #expect(expected.allSatisfy(isFinite))
        print("MLX affine dequantize-then-Gemm rows=\(rows) inner=\(inner) columns=\(columns)")
    }

    @Test("MLX affine rejects malformed sidecars")
    func mlxAffineValidation() {
        #expect(throws: MlxAffineError.sidecarLengthMismatch) {
            _ = try MlxAffine(shape: [3, 65], words: Array(repeating: 0, count: 27), scales: [1], biases: [0])
        }
    }

    @Test("q4 oracle does not invent model assets")
    func noImplicitAssets() {
        // The oracle is entirely input-driven: an empty payload is rejected rather than
        // replaced with synthetic weights. Model conversion tests own asset availability.
        #expect(DequantOracle.q4_0(data: []).isEmpty)
        #expect(DequantOracle.q4_K(data: []).isEmpty)
    }
}
