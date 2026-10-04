import Testing
import SploshRuntime

@Suite("Lookup in text outside the session")
struct ForeignLookupTests {
    @Test("a tail that is the index's own end finds what the index's own lookup finds")
    func agreesWithOwnLookup() throws {
        let text = [10, 11, 12, 13, 14, 20, 21, 50, 12, 13, 14, 30, 31, 60, 10, 11, 12, 13]
        var index = LookupIndex()
        index.reset(text)
        let own = try #require(index.continuation(after: 14, count: 2))
        let foreign = try #require(index.continuation(of: text + [14], count: 2))
        #expect(foreign.tokens == own.tokens)
        #expect(foreign.match == own.match)
    }

    @Test("the asker's own text is passed over, and a proposal stops where its text does")
    func owners() throws {
        var index = LookupIndex()
        index.append(contentsOf: [1, 2, 3, 4, 5], owner: 1)
        index.append(contentsOf: [1, 2, 3, 9, 9], owner: 2)
        // The most recent occurrence on a tie.
        let latest = try #require(index.continuation(of: [1, 2, 3], count: 4))
        #expect(latest.tokens == [9, 9])
        let other = try #require(index.continuation(of: [0, 1, 2, 3], count: 4, skipping: 2))
        #expect(other.tokens == [4, 5])
        #expect(other.match == 3)
    }

    @Test("no run and no key crosses from one text into the next")
    func boundaries() {
        var index = LookupIndex()
        index.append(contentsOf: [7, 8], owner: 1)
        index.append(contentsOf: [9, 5, 5], owner: 2)
        #expect(index.continuation(of: [7, 8, 9], count: 2) == nil)
    }

    @Test("nothing for a tail shorter than a key, or an occurrence nothing follows")
    func nothing() {
        var index = LookupIndex()
        index.append(contentsOf: [1, 2, 3], owner: 1)
        #expect(index.continuation(of: [2, 3], count: 4) == nil)
        #expect(index.continuation(of: [1, 2, 3], count: 4) == nil)
    }
}

@Suite("Dictionary study")
struct DictionaryStudyTests {
    /// A session of conversation 8 that writes what conversation 7 holds: two tokens of its
    /// own, then 100 ..< 125.
    private static let text = [900, 901] + Array(100..<125)

    @Test("a proposal from another conversation is scored against what was generated")
    func scored() throws {
        var study = DictionaryStudy()
        study.add(100..<130, conversation: 7)
        // The anchor is 102, at index 4; the step taken from it yielded 4 tokens.
        let notes = study.look(tail: Array(Self.text[0...4]), conversation: 8, position: 4, own: 0, stepped: 4)
        #expect(notes.count == 1)
        #expect(notes.first?.tokens == Array(103..<118))
        let counted = study.score(notes, text: Self.text)
        #expect(counted)
        let tally = study.tallies[DictionaryStudy.Source.conversations.rawValue][0]      // the band of runs of 3
        #expect(tally.anchors == 1)
        #expect(tally.proposed == 16)
        #expect(tally.stepped == 4)
        let report = try #require(study.report.first)
        #expect(study.report.count == 1)
        #expect(report.contains("other conversations"))
        #expect(report.contains("run 3: 1 anchors, 16.00 a step against 4.00"))
    }

    @Test("a conversation is not looked up in itself, and a run its own text matches is not counted")
    func leftOut() {
        var study = DictionaryStudy()
        study.add(100..<130, conversation: 7)
        let itself = study.look(tail: Array(Self.text[0...4]), conversation: 7, position: 4, own: 0, stepped: 4)
        #expect(itself.isEmpty)
        #expect(study.redundant == [0, 0])
        let matched = study.look(tail: Array(Self.text[0...4]), conversation: 8, position: 4, own: 3, stepped: 4)
        #expect(matched.isEmpty)
        #expect(study.redundant == [1, 0])
        #expect(study.steps == 2)
        #expect(study.report.isEmpty)
    }

    @Test("an anchor too close to the end of the text is not scored")
    func nearTheEnd() {
        var study = DictionaryStudy()
        study.add(100..<130, conversation: 7)
        let notes = study.look(tail: Array(Self.text[0...4]), conversation: 8, position: 4, own: 0, stepped: 4)
        let counted = study.score(notes, text: Array(Self.text.prefix(12)))
        #expect(!counted)
        #expect(study.report.isEmpty)
    }

    @Test("the corpus is a source of its own")
    func corpus() {
        var study = DictionaryStudy(corpus: [Array(40..<70), Array(200..<210)])
        let notes = study.look(tail: [5, 40, 41, 42, 43], conversation: 1, position: 4, own: 0, stepped: 2)
        #expect(notes.map(\.source) == [.corpus])
        #expect(notes.first?.match == 4)
        #expect(notes.first?.tokens == Array(44..<59))
    }
}
