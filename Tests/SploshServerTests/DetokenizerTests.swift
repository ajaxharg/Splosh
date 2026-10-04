import XCTest
@testable import SploshServer

final class DetokenizerTests: XCTestCase {
    private func byteLevel(_ bytes: [UInt8]) -> String {
        let scalars = bytes.map { b -> UnicodeScalar in
            if (33...126).contains(Int(b)) || b >= 161 {
                return UnicodeScalar(Int(b))!
            }
            // Byte-level tokenizers represent non-printable bytes as Unicode
            // scalars offset by 256. Construct the scalar directly rather
            // than narrowing the offset value back to UInt8.
            return UnicodeScalar(256 + Int(b))!
        }
        return String(String.UnicodeScalarView(scalars))
    }

    func testFourByteScalarIsNotSplitAcrossFrames() {
        var d = Detokenizer(tokenText: [1: byteLevel([0xF0, 0x9F]), 2: byteLevel([0x98, 0x80])])
        XCTAssertTrue(d.append([1]).isEmpty)
        let frames = d.append([2])
        XCTAssertEqual(frames.compactMap(\.content), ["😀"])
        XCTAssertFalse(frames.contains { $0.content?.contains("\u{FFFD}") == true })
    }

    func testIncompleteTailFlushesReplacementCharacter() {
        var d = Detokenizer(tokenText: [1: byteLevel([0xF0])])
        _ = d.append([1])
        let frames = d.close()
        XCTAssertEqual(frames.compactMap(\.content), ["\u{FFFD}"])
    }

    func testBothEOSIdsMapToStopOnce() {
        for eos in [248046, 248044] {
            var d = Detokenizer(tokenText: [:])
            let frames = d.append([eos])
            XCTAssertEqual(frames.compactMap(\.finishReason), ["stop"])
        }
    }

    func testStopStringAndLength() {
        var stop = Detokenizer(tokenText: [1: "a", 2: "b"], stopStrings: ["ab"])
        XCTAssertEqual(stop.append([1, 2]).compactMap(\.finishReason), ["stop"])
        var length = Detokenizer(tokenText: [1: "x"], maxCompletionTokens: 1)
        XCTAssertEqual(length.append([1]).compactMap(\.finishReason), ["length"])
    }

    func testClosedToolCallMapsToolCallsOnce() {
        var d = Detokenizer(tokenText: [1: "<tool_call>", 2: "{}", 3: "</tool_call>"])
        XCTAssertEqual(d.append([1, 2, 3]).compactMap(\.finishReason), ["tool_calls"])
    }
}
