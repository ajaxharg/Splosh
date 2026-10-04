import Foundation
import Testing
import SploshRuntime

@Suite("Prefix store and top-k selection")
struct PrefixStoreTests {
    private func scratchDirectory() -> URL {
        FileManager.default.temporaryDirectory.appendingPathComponent("splosh-prefix-tests-\(UUID().uuidString)", isDirectory: true)
    }

    private func snapshot(tokens: Int, fill: UInt8) -> SlotSnapshot {
        SlotSnapshot(tokenCount: tokens, state: Data(repeating: fill, count: 4096), kv: Data(repeating: fill &+ 1, count: 8192))
    }

    @Test("a stored prefix survives a new store instance and is found by a longer prompt")
    func roundTrip() throws {
        let directory = scratchDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let prefix = Array(0..<3000)
        let draft = DraftContextSnapshot(first: 700, end: 3000, rings: Data(repeating: 9, count: 512))
        do {
            let store = try PrefixStore(directory: directory, identity: "model-a", maxBytes: 1 << 30, minTokens: 1024)
            #expect(store.wants(prefix))
            store.save(tokens: prefix, snapshot: snapshot(tokens: prefix.count, fill: 3), draft: draft)
            store.flush()
            #expect(store.stats.entries == 1)
        }
        let reopened = try PrefixStore(directory: directory, identity: "model-a", maxBytes: 1 << 30, minTokens: 1024)
        #expect(reopened.stats.entries == 1)
        let prompt = prefix + [9001, 9002]
        let entry = try #require(reopened.longestPrefix(of: prompt, longerThan: 0))
        #expect(entry.tokenCount == prefix.count)
        let loaded = try reopened.load(entry)
        #expect(loaded.snapshot.tokenCount == prefix.count)
        #expect(loaded.snapshot.state == Data(repeating: 3, count: 4096))
        #expect(loaded.snapshot.kv == Data(repeating: 4, count: 8192))
        #expect(loaded.draft?.first == 700)
        #expect(loaded.draft?.end == 3000)
        #expect(loaded.draft?.rings == Data(repeating: 9, count: 512))
        #expect(reopened.stats.hits == 1)
    }

    @Test("lookup requires a proper prefix, a token-for-token match, and more than what is in memory")
    func lookupRules() throws {
        let directory = scratchDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let store = try PrefixStore(directory: directory, identity: "model-a", maxBytes: 1 << 30, minTokens: 1024)
        let prefix = Array(0..<2000)
        store.save(tokens: prefix, snapshot: snapshot(tokens: prefix.count, fill: 1), draft: nil)
        store.flush()
        // The identical prompt is not a proper prefix: one token must still be evaluated.
        #expect(store.longestPrefix(of: prefix, longerThan: 0) == nil)
        var different = prefix + [7]
        different[1500] = -1
        #expect(store.longestPrefix(of: different, longerThan: 0) == nil)
        #expect(store.longestPrefix(of: prefix + [7], longerThan: 2000) == nil)
        #expect(store.longestPrefix(of: prefix + [7], longerThan: 1999) != nil)
    }

    @Test("entries written by another model or KV format are ignored")
    func identity() throws {
        let directory = scratchDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let prefix = Array(0..<2000)
        let first = try PrefixStore(directory: directory, identity: "model-a", maxBytes: 1 << 30, minTokens: 1024)
        first.save(tokens: prefix, snapshot: snapshot(tokens: prefix.count, fill: 1), draft: nil)
        first.flush()
        let other = try PrefixStore(directory: directory, identity: "model-b", maxBytes: 1 << 30, minTokens: 1024)
        #expect(other.stats.entries == 0)
        #expect(other.longestPrefix(of: prefix + [1], longerThan: 0) == nil)
    }

    @Test("short prompts and small extensions are not stored; a longer prefix supersedes a shorter one")
    func growthPolicy() throws {
        let directory = scratchDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let store = try PrefixStore(directory: directory, identity: "model-a", maxBytes: 1 << 30, minTokens: 1024)
        #expect(!store.wants(Array(0..<500)))
        let base = Array(0..<4000)
        store.save(tokens: base, snapshot: snapshot(tokens: base.count, fill: 1), draft: nil)
        store.flush()
        #expect(!store.wants(base))
        // One more turn of the same conversation: restoring `base` and evaluating the rest is cheap.
        #expect(!store.wants(Array(0..<4500)))
        let grown = Array(0..<9000)
        #expect(store.wants(grown))
        store.save(tokens: grown, snapshot: snapshot(tokens: grown.count, fill: 2), draft: nil)
        store.flush()
        #expect(store.stats.entries == 1)
        #expect(store.longestPrefix(of: grown + [1], longerThan: 0)?.tokenCount == grown.count)
    }

    @Test("the byte budget evicts the least recently used entry")
    func eviction() throws {
        let directory = scratchDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        // Each entry is a little over 20 KiB (header, 2,000 tokens, 12 KiB of payload).
        let store = try PrefixStore(directory: directory, identity: "model-a", maxBytes: 50_000, minTokens: 1024)
        for family in 0..<3 {
            let tokens = (0..<2000).map { $0 + family * 100_000 }
            store.save(tokens: tokens, snapshot: snapshot(tokens: tokens.count, fill: UInt8(family)), draft: nil)
            store.flush()
        }
        #expect(store.stats.entries == 2)
        #expect(store.stats.bytes <= 50_000)
        let oldest = (0..<2000).map { $0 } + [1]
        #expect(store.longestPrefix(of: oldest, longerThan: 0) == nil)
    }

    @Test("a state checkpoint serves prompts that share a beginning, and is neither superseded nor stored twice")
    func stateCheckpoints() throws {
        let directory = scratchDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let store = try PrefixStore(directory: directory, identity: "model-a", maxBytes: 1 << 30, minTokens: 1024)
        let shared = Array(0..<3000)                       // what two conversations have in common
        let first = shared + Array(10_000..<10_200)
        #expect(store.wantsState(shared))
        store.saveState(tokens: shared, state: Data(repeating: 7, count: 4096))
        store.save(tokens: first, snapshot: snapshot(tokens: first.count, fill: 1), draft: nil)
        store.flush()
        #expect(!store.wantsState(shared))
        #expect(store.stats.entries == 2)

        // A second conversation: no whole stored prompt is its prefix, but the first one's KV
        // covers the shared part and the checkpoint has the state there.
        let second = shared + Array(20_000..<20_300)
        #expect(store.longestPrefix(of: second, longerThan: 0) == nil)
        let source = try #require(store.longestCommonPrefix(with: second))
        #expect(source.common == shared.count)
        #expect(source.entry.tokenCount == first.count)
        let state = try #require(store.bestState(for: second, upTo: source.common, longerThan: 0))
        #expect(state.tokenCount == shared.count)
        #expect(state.isStateOnly)
        #expect(try store.load(state).snapshot.state == Data(repeating: 7, count: 4096))
        // Out of reach if the KV on hand stops short of it, or if it is no gain.
        #expect(store.bestState(for: second, upTo: shared.count - 1, longerThan: 0) == nil)
        #expect(store.bestState(for: second, upTo: source.common, longerThan: shared.count) == nil)
        // The second prompt is not stored in full: the checkpoint takes it most of the way.
        #expect(!store.wants(second))

        // The first conversation grows and supersedes its own earlier prefix, not the checkpoint.
        let grown = first + Array(30_000..<36_000)
        #expect(store.wants(grown))
        store.save(tokens: grown, snapshot: snapshot(tokens: grown.count, fill: 2), draft: nil)
        store.flush()
        #expect(store.stats.entries == 2)
        #expect(store.bestState(for: second, upTo: shared.count, longerThan: 0) != nil)
        #expect(store.longestPrefix(of: grown + [1], longerThan: 0)?.tokenCount == grown.count)

        // And it is all still there for a new store instance.
        let reopened = try PrefixStore(directory: directory, identity: "model-a", maxBytes: 1 << 30, minTokens: 1024)
        #expect(reopened.bestState(for: second, upTo: shared.count, longerThan: 0)?.tokenCount == shared.count)
    }

    @Test("a prefix larger than one write(2) can take is written and read back",
          .enabled(if: ProcessInfo.processInfo.environment["SPLOSH_BIG_TESTS"] != nil, "set SPLOSH_BIG_TESTS=1; writes 2.3 GiB"))
    func largerThanTwoGigabytes() throws {
        let directory = scratchDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let store = try PrefixStore(directory: directory, identity: "model-a", maxBytes: 8 << 30, minTokens: 1024)
        let tokens = Array(0..<70_000)
        var kv = Data(count: 2_400_000_000)
        kv[0] = 7; kv[2_147_483_648] = 8; kv[kv.count - 1] = 9
        store.save(tokens: tokens, snapshot: SlotSnapshot(tokenCount: tokens.count, state: Data(repeating: 1, count: 4096), kv: kv), draft: nil)
        store.flush()
        let entry = try #require(store.longestPrefix(of: tokens + [1], longerThan: 0))
        let loaded = try store.load(entry)
        #expect(loaded.snapshot.kv.count == kv.count)
        let read = loaded.snapshot.kv
        #expect(read[read.startIndex] == 7)
        #expect(read[read.startIndex + 2_147_483_648] == 8)
        #expect(read[read.endIndex - 1] == 9)
    }

    @Test("top-k selection matches a full sort, including ties and limits that are not a chunk multiple")
    func topK() {
        var state: UInt64 = 0x2545_F491_4F6C_DD1D
        func next() -> Float {
            state ^= state << 13; state ^= state >> 7; state ^= state << 17
            // Coarse values, so equal logits are common.
            return Float(Int(state % 2001) - 1000) / 8
        }
        for limit in [100, 4096, 10_000, 248_077] {
            let logits = (0..<limit + 300).map { _ in next() }
            for k in [1, 16, 20, 64] where k <= limit {
                let expected = logits[0..<limit].enumerated()
                    .sorted { $0.element != $1.element ? $0.element > $1.element : $0.offset < $1.offset }
                    .prefix(k)
                let (ids, values) = logits.withUnsafeBufferPointer { TopK.select($0.baseAddress!, limit: limit, k: k) }
                #expect(ids == expected.map(\.offset))
                #expect(values == expected.map(\.element))
            }
            let best = logits.withUnsafeBufferPointer { TopK.argmax($0.baseAddress!, limit: limit) }
            #expect(logits[best] == logits[0..<limit].max())
        }
    }

    @Test("speculative rejection sampling reproduces the target distribution, whatever the draft proposes")
    func speculativeSamplingIsExact() {
        // Target over six tokens; drafts over candidate sets that overlap it partly, wholly, or not at all.
        let targetIDs = [10, 11, 12, 13, 14, 15]
        let target: [Float] = [0.45, 0.25, 0.15, 0.08, 0.05, 0.02]
        let drafts: [(candidates: [Int], q: [Float])] = [
            ([10, 11, 12, 13, 14, 15], [0.45, 0.25, 0.15, 0.08, 0.05, 0.02]),   // identical: always accepted
            ([11, 10, 99, 12], [0.6, 0.2, 0.15, 0.05]),                         // over-confident in the wrong token
            ([98, 99], [0.5, 0.5]),                                             // disjoint: always rejected
            ([15, 14, 13, 10], [0.7, 0.2, 0.05, 0.05]),                         // concentrated on the target's tail
        ]
        let draws = 400_000
        for (index, draft) in drafts.enumerated() {
            var sampler = Sampler(parameters: SamplingParameters(temperature: 1, topP: 1, topK: 0, seed: 1234 + UInt64(index)), vocabLimit: 100)
            var proposer = Sampler(parameters: SamplingParameters(temperature: 1, topP: 1, topK: 0, seed: 99 + UInt64(index)), vocabLimit: 100)
            var counts = [Int: Int](), accepted = 0
            for _ in 0..<draws {
                let proposed = draft.candidates[proposer.draw(draft.q)]
                let replacement = SpeculativeSampling.resolve(draft: proposed, candidates: draft.candidates, draftProbabilities: draft.q,
                                                              targetIDs: targetIDs, targetProbabilities: target, sampler: &sampler)
                if replacement == nil { accepted += 1 }
                counts[replacement ?? proposed, default: 0] += 1
            }
            #expect(Set(counts.keys).isSubset(of: Set(targetIDs)))
            for (slot, id) in targetIDs.enumerated() {
                let expected = Double(target[slot])
                let observed = Double(counts[id] ?? 0) / Double(draws)
                // Five standard deviations of a binomial proportion.
                #expect(abs(observed - expected) < 5 * (expected * (1 - expected) / Double(draws)).squareRoot(), "draft \(index) token \(id): \(observed) vs \(expected)")
            }
            // The acceptance rate is the overlap of the two distributions: sum of min(p, q).
            let overlap = targetIDs.enumerated().reduce(0.0) { sum, item in
                sum + Double(min(target[item.offset], draft.candidates.firstIndex(of: item.element).map { draft.q[$0] } ?? 0))
            }
            #expect(abs(Double(accepted) / Double(draws) - overlap) < 0.005, "draft \(index) acceptance \(Double(accepted) / Double(draws)) vs \(overlap)")
        }
    }
}
