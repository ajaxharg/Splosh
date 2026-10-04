import Foundation
import XCTest
@testable import SploshServer

final class TokenizerTests: XCTestCase {
    private struct Vector: Decodable { let text: String; let ids: [Int] }

    func testTokenizerGoldenVectors() throws {
        let root = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
        let tokenizerURL = root.appendingPathComponent("inputs/tokenizer/tokenizer.json")
        let configURL = root.appendingPathComponent("inputs/tokenizer/tokenizer_config.json")
        XCTAssertTrue(FileManager.default.fileExists(atPath: tokenizerURL.path), "Missing tokenizer.json")
        XCTAssertTrue(FileManager.default.fileExists(atPath: configURL.path), "Missing tokenizer_config.json")
        let vectorsURL = root.appendingPathComponent("Tests/Goldens/tokenizer_ids.json")
        let vectors = try JSONDecoder().decode([Vector].self, from: Data(contentsOf: vectorsURL))
        XCTAssertGreaterThanOrEqual(vectors.count, 20)
        let tokenizer = try Tokenizer(tokenizerURL: tokenizerURL, configURL: configURL)
        for vector in vectors {
            let actual = tokenizer.encode(vector.text)
            XCTAssertEqual(actual, vector.ids, "Tokenizer mismatch for \(vector.text.debugDescription): expected \(vector.ids), produced \(actual)")
        }
    }
}
