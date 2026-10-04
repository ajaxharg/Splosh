import Testing
import SploshQuant
import SploshOracle

@Suite("Q4BufferLayoutTests")
struct Q4BufferLayoutTests {
    @Test("row-major layout addresses padded words and partial logical columns")
    func paddedRowsAndPartialColumns() throws {
        let layout = try Q4BufferLayout(rows: 2, logicalK: 65)
        #expect(layout.packedWordsPerRow == 9)
        #expect(layout.rowStrideBytes == 128)
        #expect(layout.paddingWordsPerRow == 23)
        var words = Array(repeating: UInt32.zero, count: layout.packedWordCount)
        for row in 0..<2 {
            for k in 0..<65 { words[try layout.wordIndex(row: row, k: k)] |= UInt32((k + row) & 15) << (try layout.nibbleShift(row: row, k: k)) }
        }
        let scales = (0..<layout.sidecarCount).map { Q4BufferLayout.bf16Bits(Float(2 + $0)) }
        let biases = (0..<layout.sidecarCount).map { Q4BufferLayout.bf16Bits(Float(-$0)) }
        let firstValue = try layout.decode(row: 0, k: 0, words: words, scalesBF16: scales, biasesBF16: biases)
        let lastValue = try layout.decode(row: 0, k: 64, words: words, scalesBF16: scales, biasesBF16: biases)
        let secondRowLastValue = try layout.decode(row: 1, k: 64, words: words, scalesBF16: scales, biasesBF16: biases)
        #expect(firstValue == 0)
        #expect(lastValue == -1)
        #expect(secondRowLastValue == 2)
        let tensor = try MlxAffine(shape: [2, 65], words: words.enumerated().compactMap { index, word in
            index % layout.rowStrideWords < layout.packedWordsPerRow ? word : nil
        }, scales: scales.map { Q4BufferLayout.bf16($0) }, biases: biases.map { Q4BufferLayout.bf16($0) })
        let decodedLastValue = try layout.decode(row: 0, k: 64, words: words, scalesBF16: scales, biasesBF16: biases)
        #expect(DequantOracle.dequantize(tensor)[64] == decodedLastValue)
    }

    @Test("transposed orientation uses kernel-addressed K-major words")
    func transposedAddressing() throws {
        let layout = try Q4BufferLayout(rows: 3, logicalK: 65, orientation: .kByColumns)
        #expect(layout.rowStrideBytes == 128)
        var words = Array(repeating: UInt32.zero, count: layout.packedWordCount)
        for row in 0..<3 { for k in 0..<65 { words[try layout.wordIndex(row: row, k: k)] |= UInt32((row + 2 * k) & 15) << (try layout.nibbleShift(row: row, k: k)) } }
        let scales = Array(repeating: Q4BufferLayout.bf16Bits(1), count: layout.sidecarCount)
        let biases = Array(repeating: Q4BufferLayout.bf16Bits(0), count: layout.sidecarCount)
        let decodedValue = try layout.decode(row: 2, k: 64, words: words, scalesBF16: scales, biasesBF16: biases)
        #expect(decodedValue == 2)
        #expect(try layout.wordIndex(row: 2, k: 0) != layout.wordIndex(row: 0, k: 2))
    }
}
