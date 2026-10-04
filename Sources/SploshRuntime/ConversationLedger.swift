// ConversationLedger.swift — which requests are turns of the same conversation.
//
// A chat API has no conversation id: each request carries the whole conversation so far. So a
// request belongs to the conversation whose latest prompt its own prompt continues, and to a
// new one if there is none. A repeat of the same prompt, a retry, is the same conversation too.
// Two conversations that only share a system prompt are not: neither continues the other.
//
// "Continues" cannot mean "begins with the whole earlier prompt". A prompt ends by opening the
// reply (`<|im_start|>assistant`, then usually `<think>`), and the next turn renders that reply
// from the conversation's history, where the thinking may have been dropped. What the next turn
// is sure to repeat is the prompt up to where the reply opens: its stable part.

import Foundation

public struct ConversationLedger: Sendable {
    /// Oldest first: a conversation's id and the length and hash of its latest prompt's stable part.
    private var entries: [(id: Int, stableCount: Int, stableHash: UInt64)] = []
    private var nextID = 1
    public let capacity: Int

    public init(capacity: Int) { self.capacity = capacity }

    /// The conversation `prompt` continues, or a new one; either way the prompt becomes that
    /// conversation's latest. `stable` is how many leading tokens the next turn will repeat
    /// (the whole prompt if nil). `dropped` lists conversations forgotten to stay within capacity.
    public mutating func join(_ prompt: [Int], stable: Int? = nil) -> (id: Int, isNew: Bool, dropped: [Int]) {
        let stable = min(max(stable ?? prompt.count, 0), prompt.count)
        var wanted: [Int: [Int]] = [:]
        for (index, entry) in entries.enumerated() where entry.stableCount > 0 && entry.stableCount <= prompt.count {
            wanted[entry.stableCount, default: []].append(index)
        }
        // FNV-1a over the token ids, compared at each length some conversation's stable part
        // has. The longest match wins.
        var hash: UInt64 = 0xcbf2_9ce4_8422_2325
        var stableHash = hash
        var match: Int?
        for (index, token) in prompt.enumerated() {
            hash = (hash ^ UInt64(UInt32(truncatingIfNeeded: token))) &* 0x0000_0100_0000_01b3
            if index + 1 == stable { stableHash = hash }
            if let candidates = wanted[index + 1], let found = candidates.last(where: { entries[$0].stableHash == hash }) { match = found }
        }
        if let match {
            let id = entries[match].id
            entries.remove(at: match)
            entries.append((id, stable, stableHash))
            return (id, false, [])
        }
        let id = nextID
        nextID += 1
        entries.append((id, stable, stableHash))
        var dropped: [Int] = []
        while entries.count > capacity { dropped.append(entries.removeFirst().id) }
        return (id, true, dropped)
    }
}
