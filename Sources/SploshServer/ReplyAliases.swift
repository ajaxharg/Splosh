// ReplyAliases.swift — a reply as the model wrote it, token for token.
//
// A conversation's next request carries the last reply back as text, and the text is tokenised
// again. Usually that gives the tokens the model generated, the prompt runs on from what the
// server already holds, and only what is new is evaluated. But the model does not always spell
// a word the way the tokeniser would ("Molto" came out as M-ol-to where the tokeniser makes
// M-olto): the same text, other tokens. Token for token the prompt then parts from what the
// server holds in the middle of the reply, and the whole reply is evaluated again. In a day's
// logs of a harness in use that was one reply in twelve, the long ones, and a tenth of all the
// tokens evaluated.
//
// So each reply is remembered as generated, with its bytes. A later prompt that begins with
// the reply's prompt and goes on with those same bytes has that stretch of its tokens replaced
// by the generated ones: what the model is then given is what it wrote, which is also what the
// server holds. A reply whose text comes back changed in any byte is left alone.

import Foundation

public final class ReplyAliases: @unchecked Sendable {
    private struct Record {
        /// The reply's prompt: its length and a hash of its tokens.
        let promptCount: Int
        let promptHash: UInt64
        let tokens: [Int]
        let bytes: [UInt8]
    }

    private let lock = NSLock()
    private var records: [Record] = []          // oldest first
    private let capacity: Int

    public init(capacity: Int = 512) { self.capacity = capacity }

    /// A finished reply: the prompt it answered and the tokens generated, with their bytes.
    public func remember(prompt: [Int], generated: [Int], bytes: [UInt8]) {
        guard !prompt.isEmpty, !generated.isEmpty, !bytes.isEmpty else { return }
        let record = Record(promptCount: prompt.count, promptHash: Self.hash(prompt[...]), tokens: generated, bytes: bytes)
        lock.lock(); defer { lock.unlock() }
        records.removeAll { $0.promptCount == record.promptCount && $0.promptHash == record.promptHash }
        records.append(record)
        if records.count > capacity { records.removeFirst(records.count - capacity) }
    }

    /// `prompt` with every remembered reply it contains put back as it was generated.
    /// `bytes` gives a token's bytes (none for a token that has none).
    public func apply(to prompt: [Int], bytes: (Int) -> [UInt8]) -> [Int] {
        lock.lock()
        // All of them: a record's length is of its prompt as generated, which may be longer than
        // the prompt as the client sent it. The loop below stops at the first that cannot fit.
        var pending = records
        lock.unlock()
        guard !pending.isEmpty else { return prompt }
        var tokens = prompt
        // A reply's prompt holds the replies before it as generated, so they go in order: the
        // earliest first, and each later prompt is looked for in what that has produced.
        pending.sort { $0.promptCount < $1.promptCount }
        var settled: [Record] = []                // found, and already as generated: nothing to keep them for
        // One pass: a replacement begins where its record's prompt ends, so what has been
        // hashed is never changed afterwards.
        var hash: UInt64 = 0xcbf2_9ce4_8422_2325, hashed = 0
        for record in pending {
            guard record.promptCount < tokens.count else { break }
            while hashed < record.promptCount {
                hash = (hash ^ UInt64(UInt32(truncatingIfNeeded: tokens[hashed]))) &* 0x0000_0100_0000_01b3
                hashed += 1
            }
            guard hash == record.promptHash else { continue }
            if tokens[record.promptCount...].starts(with: record.tokens) { settled.append(record); continue }
            // The prompt goes on with other tokens. The same bytes, ending where a token ends?
            var end = record.promptCount, matched = 0
            while matched < record.bytes.count, end < tokens.count {
                let piece = bytes(tokens[end])
                guard matched + piece.count <= record.bytes.count,
                      record.bytes[matched..<matched + piece.count].elementsEqual(piece) else { break }
                matched += piece.count
                end += 1
            }
            guard matched == record.bytes.count else { continue }
            tokens.replaceSubrange(record.promptCount..<end, with: record.tokens)
        }
        if !settled.isEmpty {
            lock.lock()
            records.removeAll { record in settled.contains { $0.promptCount == record.promptCount && $0.promptHash == record.promptHash } }
            lock.unlock()
        }
        return tokens
    }

    private static func hash(_ tokens: ArraySlice<Int>) -> UInt64 {
        var hash: UInt64 = 0xcbf2_9ce4_8422_2325
        for token in tokens { hash = (hash ^ UInt64(UInt32(truncatingIfNeeded: token))) &* 0x0000_0100_0000_01b3 }
        return hash
    }
}
