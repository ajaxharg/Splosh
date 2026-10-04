import Foundation

/// Incremental inverse of the tokenizer's byte-level representation.
public struct Detokenizer: Sendable {
    public struct Frame: Sendable, Equatable {
        public let content: String?
        public let finishReason: String?
        public init(content: String? = nil, finishReason: String? = nil) {
            self.content = content; self.finishReason = finishReason
        }
    }

    private var tokenText: [Int: String]
    private var pendingBytes: [UInt8] = []
    private var stopStrings: [String]
    private var held = ""
    private var inToolCall = false
    private var finished = false
    private var completionCount = 0
    private let maxCompletionTokens: Int?

    public init(tokenText: [Int: String], stopStrings: [String] = [], maxCompletionTokens: Int? = nil) {
        self.tokenText = tokenText
        self.stopStrings = stopStrings.filter { !$0.isEmpty }
        self.maxCompletionTokens = maxCompletionTokens
    }

    /// Compatibility spelling for callers that expose the tokenizer vocabulary.
    public init(vocab: [Int: String], stopStrings: [String] = [], maxCompletionTokens: Int? = nil) {
        self.init(tokenText: vocab, stopStrings: stopStrings, maxCompletionTokens: maxCompletionTokens)
    }

    public mutating func append(_ ids: [Int]) -> [Frame] {
        guard !finished else { return [] }
        var result: [Frame] = []
        for id in ids {
            if SpecialTokens.eosIDs.contains(id) {
                result += close(.stop); break
            }
            completionCount += 1
            if let text = tokenText[id] { pendingBytes += Self.byteLevelBytes(text) }
            result += drainCompleteBytes()
            if let limit = maxCompletionTokens, completionCount >= limit {
                result += close(.length); break
            }
        }
        return result
    }

    public mutating func push(_ ids: [Int]) -> [Frame] { append(ids) }

    public enum CloseReason: Sendable { case stop, length, toolCalls }

    public mutating func close(_ reason: CloseReason = .stop) -> [Frame] {
        guard !finished else { return [] }
        var result = drainCompleteBytes()
        if !pendingBytes.isEmpty {
            pendingBytes.removeAll(keepingCapacity: false)
            result += process("\u{FFFD}")
        }
        if !held.isEmpty && !inToolCall { result.append(Frame(content: held.precomposedStringWithCanonicalMapping)); held = "" }
        let mapped: String
        switch reason { case .stop: mapped = "stop"; case .length: mapped = "length"; case .toolCalls: mapped = "tool_calls" }
        finished = true
        result.append(Frame(finishReason: mapped))
        return result
    }

    public mutating func finish(_ reason: CloseReason = .stop) -> [Frame] { close(reason) }

    private mutating func drainCompleteBytes() -> [Frame] {
        guard !pendingBytes.isEmpty else { return [] }
        var count = pendingBytes.count
        while count > 0 && String(bytes: pendingBytes[0..<count], encoding: .utf8) == nil { count -= 1 }
        guard count > 0 else { return [] }
        let bytes = Array(pendingBytes[0..<count]); pendingBytes.removeFirst(count)
        return process(String(decoding: bytes, as: UTF8.self).precomposedStringWithCanonicalMapping)
    }

    private mutating func process(_ text: String) -> [Frame] {
        var output: [Frame] = []
        for scalar in text { held.append(scalar) }
        while true {
            if !inToolCall, let start = held.range(of: SpecialTokens.toolCallOpen) {
                let before = String(held[..<start.lowerBound]); held = String(held[start.upperBound...]);
                output += emitScanned(before); inToolCall = true; continue
            }
            if inToolCall, let end = held.range(of: SpecialTokens.toolCallClose) {
                held = String(held[end.upperBound...]); inToolCall = false
                output += close(.toolCalls); return output
            }
            if inToolCall { return output }
            // Keep enough cumulative text to match a stop crossing token/frame boundaries.
            let keep = stopStrings.map(\.count).max().map { max(0, $0 - 1) } ?? 0
            if let stop = stopStrings.first(where: { held.contains($0) }) {
                let before = String(held[..<held.range(of: stop)!.lowerBound]); held = ""
                output += emitScanned(before); output += close(.stop); return output
            }
            if held.count > keep {
                let index = held.index(held.startIndex, offsetBy: held.count - keep)
                let ready = String(held[..<index]); held = String(held[index...])
                output += emitScanned(ready)
            }
            return output
        }
    }

    private mutating func emitScanned(_ text: String) -> [Frame] {
        guard !text.isEmpty else { return [] }
        return [Frame(content: text.precomposedStringWithCanonicalMapping)]
    }

    private static func byteLevelBytes(_ text: String) -> [UInt8] {
        var result: [UInt8] = []
        for scalar in text.unicodeScalars {
            let v = scalar.value
            if v >= 256 && v <= 511 { result.append(UInt8(v - 256)) }
            else if v <= 255 { result.append(UInt8(v)) }
            else { result += Array(String(scalar).utf8) }
        }
        return result
    }
}
