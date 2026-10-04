// LookupIndex.swift — drafts copied from the session's own text.
//
// When the model is reproducing something already in its context (a file it is editing, a
// record it was asked to repeat, the argument of a tool call it just read) the best draft is
// the text itself: find where the last few tokens occurred before and propose what followed
// them. The target verifies the block exactly as it verifies a block from the draft model, so
// a wrong guess costs a step and nothing else. There is no draft pass to pay for, and a copy
// is usually right for far longer than the draft model's block.
//
// The idea is the copy rule of Splish (github.com/publicExcess/splish), which has it from
// TensorFold (github.com/ashhart/TensorFold). The index and the choice between a copy and the
// draft model are written for Splosh.

import Foundation

public struct LookupIndex: Sendable {
    /// Tokens in a key. Shorter keys match everywhere; longer ones miss the start of a copy.
    public static let keyLength = 3

    private var tokens: [Int32] = []
    /// For each key, the index of the last token of its most recent occurrence.
    private var latest: [UInt64: Int32] = [:]
    /// For the occurrence ending at each index, the one before it with the same key (-1: none).
    private var earlier: [Int32] = []
    /// The text each token came from, in an index that holds several texts (see
    /// `append(contentsOf:owner:)`); empty in an index over one.
    private var owners: [Int32] = []
    /// Stands between two texts. It is no token, so no shared run and no proposal crosses it.
    private static let boundary: Int32 = -1

    public init() {}

    public var count: Int { tokens.count }

    private static func key(_ a: Int32, _ b: Int32, _ c: Int32) -> UInt64 {
        // Token ids fit in 18 bits (the vocabulary has 248,320 entries), so three fit a word.
        (UInt64(UInt32(bitPattern: a)) << 42) ^ (UInt64(UInt32(bitPattern: b)) << 21) ^ UInt64(UInt32(bitPattern: c))
    }

    public mutating func reset(_ history: [Int]) {
        tokens.removeAll(keepingCapacity: true)
        earlier.removeAll(keepingCapacity: true)
        latest.removeAll(keepingCapacity: true)
        owners.removeAll(keepingCapacity: true)
        tokens.reserveCapacity(history.count + 4096)
        earlier.reserveCapacity(history.count + 4096)
        latest.reserveCapacity(history.count + 4096)
        for token in history { append(token) }
    }

    /// Make the index hold `history`: keep what it has of it, drop what it has that is not, and
    /// add the rest. A conversation's next turn costs what the turn added, not its whole
    /// context (`reset` over 200K tokens is tens of milliseconds on the scheduler thread).
    public mutating func extend(to history: [Int]) {
        guard owners.isEmpty else { reset(history); return }
        var common = 0
        let limit = min(tokens.count, history.count)
        while common < limit, tokens[common] == Int32(truncatingIfNeeded: history[common]) { common += 1 }
        // Dropping a token costs about what adding one does.
        if tokens.count - common >= common { reset(history); return }
        truncate(to: common)
        for token in history[common...] { append(token) }
    }

    /// Keep the first `count` tokens. Taken off last first, each occurrence gives its key back
    /// to the one before it, so the index is exactly as it was when it held `count` tokens.
    public mutating func truncate(to count: Int) {
        while tokens.count > count {
            let end = tokens.count - 1
            if end >= Self.keyLength - 1 {
                let key = Self.key(tokens[end - 2], tokens[end - 1], tokens[end])
                latest[key] = earlier[end] >= 0 ? earlier[end] : nil
            }
            tokens.removeLast()
            earlier.removeLast()
        }
        if owners.count > count { owners.removeLast(owners.count - count) }
    }

    public mutating func append(_ token: Int) {
        tokens.append(Int32(truncatingIfNeeded: token))
        let end = tokens.count - 1
        guard end >= Self.keyLength - 1 else { earlier.append(-1); return }
        let key = Self.key(tokens[end - 2], tokens[end - 1], tokens[end])
        earlier.append(latest[key] ?? -1)
        latest[key] = Int32(end)
    }

    /// Add a whole text to an index that holds several, kept apart from the one before it.
    /// `owner` says whose it is, so that a lookup can pass over the asker's own text.
    public mutating func append(contentsOf text: some Sequence<Int>, owner: Int) {
        append(Int(Self.boundary))
        for token in text { append(token) }
        owners.append(contentsOf: repeatElement(Int32(truncatingIfNeeded: owner), count: tokens.count - owners.count))
    }

    /// The occurrence of `key` that shares the longest run with the present text, the most
    /// recent on a tie. `before(n)` is the present text's token n places before its last, nil
    /// past its start. Occurrences in the text of `owner` are passed over.
    private func best(_ key: UInt64, candidates: Int, longest: Int, skipping owner: Int? = nil, before: (Int) -> Int32?) -> (index: Int, match: Int)? {
        var candidate = latest[key] ?? -1
        var best = (index: -1, match: 0)
        var examined = 0, passed = 0
        while candidate >= 0, examined < candidates {
            let index = Int(candidate)
            candidate = earlier[index]
            if let owner, index < owners.count, owners[index] == Int32(truncatingIfNeeded: owner) {
                // The asker's own occurrences are the most recent; do not walk them for ever.
                passed += 1
                if passed > candidates * 16 { break }
                continue
            }
            // The key's own tokens match by construction; count how far back the run extends.
            var match = Self.keyLength
            while match < longest, index - match >= 0, let token = before(match), tokens[index - match] == token { match += 1 }
            if match > best.match { best = (index, match) }
            if match >= longest { break }
            examined += 1
        }
        return best.index >= 0 ? best : nil
    }

    /// What followed the best earlier occurrence of the text ending in `anchor` (a token that
    /// comes after everything appended so far). Returns up to `count` tokens and the number of
    /// tokens, counting back from the anchor, that the occurrence shares with the present text;
    /// nil if the last `keyLength` tokens have not been seen before.
    ///
    /// The best occurrence is the one with the longest shared run, the most recent on a tie.
    /// A continuation that runs into the present text carries on into the proposal itself,
    /// which is what a repeating pattern needs.
    public func continuation(after anchor: Int, count: Int, candidates: Int = 24, longest: Int = 64) -> (match: Int, tokens: [Int])? {
        let end = tokens.count          // the anchor's index
        guard end >= Self.keyLength - 1, count > 0 else { return nil }
        let anchor = Int32(truncatingIfNeeded: anchor)
        guard let best = best(Self.key(tokens[end - 2], tokens[end - 1], anchor), candidates: candidates, longest: longest,
                              before: { tokens[end - $0] }) else { return nil }
        var proposal: [Int] = []
        proposal.reserveCapacity(count)
        for offset in 0..<count {
            let source = best.index + 1 + offset
            if source < end {
                proposal.append(Int(tokens[source]))
            } else if source == end {
                proposal.append(Int(anchor))
            } else {
                proposal.append(proposal[source - end - 1])
            }
        }
        return (best.match, proposal)
    }

    /// The same for a text the index does not end with: `tail` is that text's last tokens, the
    /// anchor last. The proposal stops where the text it is copied from does; nil if nothing
    /// follows the occurrence.
    public func continuation(of tail: [Int], count: Int, candidates: Int = 24, longest: Int = 64, skipping owner: Int? = nil) -> (match: Int, tokens: [Int])? {
        let last = tail.count - 1
        guard last >= Self.keyLength - 1, count > 0 else { return nil }
        func present(_ back: Int) -> Int32? { back <= last ? Int32(truncatingIfNeeded: tail[last - back]) : nil }
        guard let best = best(Self.key(present(2)!, present(1)!, present(0)!), candidates: candidates, longest: longest,
                              skipping: owner, before: present) else { return nil }
        var proposal: [Int] = []
        proposal.reserveCapacity(count)
        var source = best.index + 1
        while proposal.count < count, source < tokens.count, tokens[source] != Self.boundary {
            proposal.append(Int(tokens[source]))
            source += 1
        }
        return proposal.isEmpty ? nil : (best.match, proposal)
    }
}

/// How a session's speculative blocks have fared, and what that says about the next one:
/// whether to copy or to ask the draft model, and what a double block's second half is worth.
public struct SpeculationState: Sendable {
    public struct BlockStats: Sendable {
        /// Tokens a step yields from the first `DraftModel.blockSize` rows.
        public var blockYield: Double
        /// How often the drafts in those rows are all accepted.
        public var wholeRate: Double
        /// Tokens the rows after them then add.
        public var tailYield: Double
        /// What the second half of a double block is expected to add to a step.
        public var tail: Double { wholeRate * tailYield }

        mutating func record(agreed: Int, drafts: Int) {
            let prefix = DraftModel.blockSize - 1
            blockYield += (Double(min(agreed, prefix) + 1) - blockYield) / 16
            wholeRate += ((agreed >= prefix ? 1 : 0) - wholeRate) / 12
            if drafts > prefix, agreed >= prefix { tailYield += (Double(agreed - prefix) - tailYield) / 4 }
        }
    }

    /// Everything the session has evaluated: the source of copied drafts.
    public var lookup = LookupIndex()
    public private(set) var drafted = BlockStats(blockYield: 3.5, wholeRate: 0.15, tailYield: 3)
    public private(set) var copied = BlockStats(blockYield: 6, wholeRate: 0.5, tailYield: 5)

    /// The shortest shared run worth copying from. Measured against the draft model from the
    /// same anchors: runs under ten tokens yield 1.5-5 tokens a step where the model yields
    /// 6-11 (repeated structure with different contents: JSON keys, a function signature);
    /// runs of twenty or more yield 15 of a possible 16.
    public static let shortestRun = 10
    /// Tokens a copied block yields, for runs of 10-19 tokens and of 20 or more: starting
    /// values from those measurements, corrected by what the session's own copies do.
    private static let copyPriors: [Double] = [5, 14]
    private var copyYield = SpeculationState.copyPriors
    private static func bucket(_ match: Int) -> Int { match < 20 ? 0 : 1 }
    /// A copied block needs no draft pass, which is about a tenth of a cycle.
    private static let copyCycle = 0.9

    public init() {}

    /// A copied draft for the block after `anchor`, when copying is expected to beat the draft
    /// model: `count` tokens and the length of the run the source shares with the text.
    public mutating func copy(after anchor: Int, count: Int) -> (match: Int, tokens: [Int])? {
        guard let found = lookup.continuation(after: anchor, count: count), found.match >= Self.shortestRun else { return nil }
        let bucket = Self.bucket(found.match)
        if copyYield[bucket] > Self.copyCycle * (drafted.blockYield + drafted.tail) { return found }
        // Not worth trying now; let the estimate drift back so a change in the text gets noticed.
        copyYield[bucket] += (Self.copyPriors[bucket] - copyYield[bucket]) / 32
        return nil
    }

    /// Record a verified block: `agreed` of its `drafts` tokens were accepted. `copyMatch` is
    /// the shared run of a copied block, nil for one from the draft model.
    public mutating func record(agreed: Int, drafts: Int, copyMatch: Int?) {
        if let copyMatch {
            let bucket = Self.bucket(copyMatch)
            copyYield[bucket] += (Double(agreed + 1) - copyYield[bucket]) / 4
            copied.record(agreed: agreed, drafts: drafts)
        } else {
            drafted.record(agreed: agreed, drafts: drafts)
        }
    }
}
