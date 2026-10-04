import Foundation
import Testing
import SploshRuntime
@testable import SploshServer

@Suite("Conversations")
struct ConversationLedgerTests {
    @Test("a prompt that continues an earlier one is the same conversation; a retry is too")
    func continuation() {
        var ledger = ConversationLedger(capacity: 10)
        let system = Array(0..<500)
        let first = system + [1000, 1001]
        let a = ledger.join(first)
        #expect(a.isNew)
        // The next turn: the reply and a tool result after the same beginning.
        let second = first + Array(2000..<2300)
        let b = ledger.join(second)
        #expect(!b.isNew && b.id == a.id)
        #expect(ledger.join(second).id == a.id)                         // the same request again
        #expect(ledger.join(second + [7, 8, 9]).id == a.id)
    }

    @Test("conversations that share only a system prompt are different")
    func sharedSystemPrompt() {
        var ledger = ConversationLedger(capacity: 10)
        let system = Array(0..<500)
        let a = ledger.join(system + [1000, 1001])
        let b = ledger.join(system + [3000, 3001, 3002])
        #expect(b.isNew && b.id != a.id)
        // Each carries on separately.
        #expect(ledger.join(system + [1000, 1001, 5, 6]).id == a.id)
        #expect(ledger.join(system + [3000, 3001, 3002, 5, 6]).id == b.id)
        // An earlier turn's prompt is no longer the conversation's latest: a prompt that
        // extends only that is a branch, and a conversation of its own.
        #expect(ledger.join(system + [1000, 1001, 9]).isNew)
    }

    @Test("the reply a prompt opens may be rendered differently in the next turn")
    func openedReply() {
        var ledger = ConversationLedger(capacity: 10)
        let system = Array(0..<500)
        // 90 opens a message, 91 92 are "assistant" and the newline, 93 94 open the thinking.
        let history = system + [90, 600, 601, 99]
        let a = ledger.join(history + [90, 91, 92, 93, 94], stable: history.count + 1)
        // Next turn: the reply is in the history without its thinking, then a tool result.
        let next = history + [90, 91, 92, 700, 701, 99, 90, 800, 99]
        let b = ledger.join(next + [90, 91, 92, 93, 94], stable: next.count + 1)
        #expect(!b.isNew && b.id == a.id)
        // The same first request again is a branch from before that turn, not a continuation.
        #expect(ledger.join(history + [90, 91, 92, 93, 94], stable: history.count + 1).isNew)
        // Another conversation under the same system prompt.
        let other = system + [90, 650, 99]
        #expect(ledger.join(other + [90, 91, 92, 93, 94], stable: other.count + 1).isNew)
    }

    @Test("the oldest conversation is forgotten past capacity")
    func capacity() {
        var ledger = ConversationLedger(capacity: 2)
        let a = ledger.join([1, 2, 3])
        _ = ledger.join([4, 5, 6])
        let c = ledger.join([7, 8, 9])
        #expect(c.dropped == [a.id])
        #expect(ledger.join([1, 2, 3, 4]).isNew)
    }

    /// Real prompts: the chat template and tokenizer the server uses, an agent's turns as a
    /// harness sends them (reply text and tool calls back, with or without the thinking).
    @Test("an agent's turns, as rendered and tokenised, are one conversation")
    func renderedTurns() throws {
        let root = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
        let tokenizer = try Tokenizer(tokenizerURL: root.appendingPathComponent("inputs/tokenizer/tokenizer.json"),
                                      configURL: root.appendingPathComponent("inputs/tokenizer/tokenizer_config.json"))
        func prompt(_ messages: String) throws -> (tokens: [Int], stable: Int?) {
            let rendered = try ChatTemplate.render(messages: JSONValue.parse(messages).arrayValue ?? [])
            let tokens = tokenizer.encode(rendered.prompt)
            return (tokens, tokens.lastIndex(of: SpecialTokens.imStartID).map { $0 + 1 })
        }
        let system = #"{"role":"system","content":"You are a coding agent. Use the tools."}"#
        let ask = #"{"role":"user","content":"List the files, then read the largest."}"#
        let call = #""tool_calls":[{"id":"c1","type":"function","function":{"name":"ls","arguments":"{\"path\":\".\"}"}}]"#
        let result = #"{"role":"tool","tool_call_id":"c1","content":"a.txt 10\nb.txt 900"}"#
        var ledger = ConversationLedger(capacity: 10)

        let first = try prompt("[\(system),\(ask)]")
        let a = ledger.join(first.tokens, stable: first.stable)
        #expect(a.isNew)
        // The harness sends the reply back without its thinking.
        let second = try prompt(#"[\#(system),\#(ask),{"role":"assistant","content":"Listing.",\#(call)},\#(result)]"#)
        #expect(!second.tokens.starts(with: first.tokens))               // why the stable part is what is compared
        #expect(ledger.join(second.tokens, stable: second.stable).id == a.id)
        // And a third turn, this time with the thinking kept.
        let third = try prompt(#"[\#(system),\#(ask),{"role":"assistant","content":"Listing.",\#(call)},\#(result),{"role":"assistant","reasoning_content":"b.txt is the largest.","content":"b.txt is the largest."},{"role":"user","content":"Now read it."}]"#)
        #expect(ledger.join(third.tokens, stable: third.stable).id == a.id)
        // A second agent with the same instructions and a different task is its own conversation.
        let other = try prompt(#"[\#(system),{"role":"user","content":"List the files, then delete the smallest."}]"#)
        #expect(ledger.join(other.tokens, stable: other.stable).isNew)
    }
}
