import Testing
import SploshRuntime

@Suite("Lookup drafts")
struct LookupIndexTests {
    @Test("an index cut back and extended answers as one built from scratch")
    func extend() throws {
        var random = SplitMix(seed: 7)
        let text = (0..<3000).map { _ in Int(random.uniform() * 12) }
        let next = Array(text[0..<2600]) + (0..<500).map { _ in Int(random.uniform() * 12) }
        var kept = LookupIndex()
        kept.reset(text)
        kept.extend(to: next)
        var fresh = LookupIndex()
        fresh.reset(next)
        #expect(kept.count == fresh.count)
        for anchor in 0..<12 {
            let a = kept.continuation(after: anchor, count: 8), b = fresh.continuation(after: anchor, count: 8)
            #expect(a?.tokens == b?.tokens && a?.match == b?.match)
        }
        // Mostly different: rebuilt, with the same answers.
        kept.extend(to: Array(next.reversed()))
        fresh.reset(Array(next.reversed()))
        for anchor in 0..<12 { #expect(kept.continuation(after: anchor, count: 8)?.tokens == fresh.continuation(after: anchor, count: 8)?.tokens) }
    }

    @Test("the continuation is what followed the earlier occurrence")
    func continuation() throws {
        var index = LookupIndex()
        index.reset([1, 2, 3, 4, 5, 6, 7, 8, 9, 100, 101, 2, 3])
        // Present text ends ... 2, 3, 4: seen before at indices 1...3, followed by 5, 6, 7, ...
        let found = try #require(index.continuation(after: 4, count: 4))
        #expect(found.tokens == [5, 6, 7, 8])
        #expect(found.match == 3)
    }

    @Test("nothing is proposed for text not seen before")
    func noMatch() {
        var index = LookupIndex()
        index.reset([1, 2, 3, 4, 5])
        #expect(index.continuation(after: 9, count: 4) == nil)
        #expect(LookupIndex().continuation(after: 1, count: 4) == nil)
    }

    @Test("the longest shared run wins over a more recent, shorter one")
    func longestRun() throws {
        var index = LookupIndex()
        // "10 11 12 13 14 -> 20" early on; "12 13 14 -> 30" later; the present text ends 11 12 13 14.
        index.reset([10, 11, 12, 13, 14, 20, 21, 50, 12, 13, 14, 30, 31, 60, 10, 11, 12, 13])
        let found = try #require(index.continuation(after: 14, count: 2))
        #expect(found.tokens == [20, 21])
        #expect(found.match == 5)
    }

    @Test("a repeating pattern continues into the proposal itself")
    func periodic() throws {
        var index = LookupIndex()
        index.reset([7, 8, 9, 7, 8])
        let found = try #require(index.continuation(after: 9, count: 7))
        #expect(found.tokens == [7, 8, 9, 7, 8, 9, 7])
    }

    @Test("appending keeps the index current")
    func incremental() throws {
        var index = LookupIndex()
        index.reset([1, 2, 3, 4])
        for token in [9, 1, 2] { index.append(token) }
        #expect(index.count == 7)
        let found = try #require(index.continuation(after: 3, count: 3))
        #expect(found.tokens == [4, 9, 1])
    }
}
