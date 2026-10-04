import Testing
import SploshRuntime

/// Ids used below: the chat template's close is a newline, the close token, a blank line.
private let newline = 10, closeToken = 100, blankLine = 11
private let closing = ThinkingClose(token: closeToken, tokens: [newline, closeToken, blankLine])
private let stopToken = 999

/// What the model chooses while it thinks and while it answers; none of them is a special id.
private func thinking(_ count: Int) -> [Int] { (0..<count).map { 500 + $0 } }
private func answer(_ count: Int) -> [Int] { (0..<count).map { 700 + $0 } }

/// A reply as the scheduler writes it: `chosen[i]` is what the model would pick at position i
/// whatever came before it, so where the watch substitutes a token that choice is dropped and the
/// model carries on with its choice for the next position.
private func reply(_ watch: inout ThinkingWatch, chosen: [Int]) -> [Int] {
    var out: [Int] = []
    for token in chosen { out.append(watch.next(token, generated: out.count)) }
    return out
}

/// A watch with a limit of 32 tokens (40 allowed, a quarter of it held back).
private func limit32(_ close: ThinkingClose = closing) throws -> ThinkingWatch {
    try #require(ThinkingWatch(close, maxTokens: 40, reserve: 8))
}

private func limit(maxTokens: Int, reserve: Int) throws -> Int {
    try #require(ThinkingWatch(closing, maxTokens: maxTokens, reserve: reserve)).limit
}

@Suite("Thinking watch")
struct ThinkingWatchTests {
    // MARK: When there is nothing to watch

    @Test("a reply that does not begin in thinking is not watched")
    func noClose() {
        #expect(ThinkingWatch(nil, maxTokens: 1000, reserve: 100) == nil)
    }

    @Test("a close with no tokens to write is not watched")
    func emptyClosing() {
        #expect(ThinkingWatch(ThinkingClose(token: closeToken, tokens: []), maxTokens: 1000, reserve: 100) == nil)
    }

    @Test("a reserve of nothing is not watched")
    func noReserve() {
        #expect(ThinkingWatch(closing, maxTokens: 1000, reserve: 0) == nil)
    }

    @Test("a reply too short for a quarter of it to hold a token is not watched")
    func tooShort() {
        #expect(ThinkingWatch(closing, maxTokens: 3, reserve: 8192) == nil)
        #expect(ThinkingWatch(closing, maxTokens: 0, reserve: 8192) == nil)
    }

    // MARK: Where thinking is ended

    @Test("thinking is ended where only the reserve is left")
    func limitIsWhatTheReserveLeaves() throws {
        #expect(try limit(maxTokens: 1000, reserve: 100) == 900)
    }

    @Test("the reserve is at most a quarter of the reply")
    func reserveCappedAtAQuarter() throws {
        #expect(try limit(maxTokens: 32768, reserve: 8192) == 24576)
        #expect(try limit(maxTokens: 800, reserve: 8192) == 600)
    }

    @Test("the smallest reply that is watched holds back one token")
    func smallestWatched() throws {
        #expect(try limit(maxTokens: 4, reserve: 8192) == 3)
    }

    // MARK: A reply that closes its own thinking

    @Test("a reply that closes its thinking before the limit is passed through unchanged")
    func closesItself() throws {
        var watch = try limit32()
        let chosen = thinking(10) + [closeToken] + answer(5)
        #expect(reply(&watch, chosen: chosen) == chosen)
    }

    @Test("a reply that closed its thinking is still unchanged when its answer runs past the limit")
    func answerRunsPastTheLimit() throws {
        var watch = try limit32()
        let chosen = thinking(10) + [closeToken] + answer(80)
        #expect(chosen.count > 32 + closing.tokens.count)
        #expect(reply(&watch, chosen: chosen) == chosen)
    }

    @Test("the model's own closing sequence is not added to")
    func ownClosingSequence() throws {
        var watch = try limit32()
        let chosen = thinking(10) + [newline, closeToken, blankLine] + answer(60)
        #expect(reply(&watch, chosen: chosen) == chosen)
    }

    // MARK: A reply still thinking at the limit

    @Test("a reply still thinking at the limit has its thinking closed there and then goes on")
    func closedAtTheLimit() throws {
        var watch = try limit32()
        let chosen = thinking(60)
        let out = reply(&watch, chosen: chosen)

        #expect(out.count == chosen.count)
        #expect(Array(out[0..<32]) == Array(chosen[0..<32]))
        #expect(Array(out[32..<35]) == closing.tokens)
        // Positions 32 to 34 were the closing's; the model's choices for them are gone, not delayed.
        #expect(Array(out[35...]) == Array(chosen[35...]))
    }

    @Test("the close token is the second token written, so thinking ends one token after the limit")
    func closePosition() throws {
        var watch = try limit32()
        let out = reply(&watch, chosen: thinking(60))
        #expect(out.firstIndex(of: closeToken) == 33)
        #expect(out.filter { $0 == closeToken }.count == 1)
    }

    @Test("the model's choices while the closing is being written are ignored, whatever they are")
    func choicesDuringTheClosing() throws {
        var watch = try limit32()
        // Position 33 is the close token the model happens to pick while the closing is going out.
        var chosen = thinking(60)
        chosen[33] = closeToken
        chosen[34] = stopToken
        let out = reply(&watch, chosen: chosen)

        #expect(Array(out[32..<35]) == closing.tokens)
        #expect(Array(out[35...]) == Array(chosen[35...]))
        #expect(out.filter { $0 == closeToken }.count == 1)
    }

    @Test("a closing of one token is written in place of the model's choice at the limit")
    func singleTokenClosing() throws {
        var watch = try limit32(ThinkingClose(token: closeToken, tokens: [closeToken]))
        let chosen = thinking(40)
        let out = reply(&watch, chosen: chosen)

        #expect(Array(out[0..<32]) == Array(chosen[0..<32]))
        #expect(out[32] == closeToken)
        #expect(Array(out[33...]) == Array(chosen[33...]))
    }

    @Test("a stop token chosen before the limit while still thinking is left alone")
    func stopsBeforeTheLimit() throws {
        var watch = try limit32()
        let chosen = thinking(31) + [stopToken]
        #expect(reply(&watch, chosen: chosen) == chosen)
    }

    // MARK: The model's choice at the limit

    @Test("a model that picks the close token exactly at the limit is left alone")
    func closesAtTheLimit() throws {
        var watch = try limit32()
        let chosen = thinking(32) + [closeToken] + answer(10)
        let out = reply(&watch, chosen: chosen)

        #expect(out == chosen)
        #expect(out.filter { $0 == closeToken }.count == 1)
    }

    @Test("a stop token chosen at the limit while still thinking is replaced by the closing")
    func stopAtTheLimit() throws {
        var watch = try limit32()
        let chosen = thinking(32) + [stopToken] + answer(20)
        let out = reply(&watch, chosen: chosen)

        #expect(Array(out[32..<35]) == closing.tokens)
        #expect(Array(out[35...]) == Array(chosen[35...]))
        #expect(!out.contains(stopToken))
    }

    @Test("a model that picks the first closing token anyway still has the rest of the closing written")
    func firstClosingTokenChosenAnyway() throws {
        var watch = try limit32()
        let chosen = thinking(32) + [newline] + answer(20)
        let out = reply(&watch, chosen: chosen)

        #expect(out[32] == newline)
        #expect(Array(out[32..<35]) == closing.tokens)
        #expect(Array(out[35...]) == Array(chosen[35...]))
    }

    // MARK: After the closing

    @Test("a close token the model picks after the closing was written is passed through and nothing is written twice")
    func laterCloseToken() throws {
        var watch = try limit32()
        let chosen = thinking(40) + [closeToken] + answer(10)
        let out = reply(&watch, chosen: chosen)

        #expect(Array(out[32..<35]) == closing.tokens)
        #expect(out[40] == closeToken)
        #expect(Array(out[35...]) == Array(chosen[35...]))
        #expect(out.count == chosen.count)
        #expect(out.filter { $0 == closeToken }.count == 2)
    }

    @Test("once the closing has been written the reply is the model's, however long it goes on")
    func answerAfterTheClosing() throws {
        var watch = try limit32()
        let chosen = thinking(32) + [stopToken] + answer(200)
        let out = reply(&watch, chosen: chosen)
        #expect(Array(out[35...]) == Array(chosen[35...]))
    }
}
