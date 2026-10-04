import Foundation
import Testing
import Hummingbird
import SploshRuntime
@testable import SploshServer

/// A reply cut inside a tool call holds a call that cannot be made, and the harness's turn ends
/// there. A call that is open at the limit is let run on to its close.
@Suite("Reply limit", .serialized)
struct ReplyLimitTests {
    private static let open = 100, close = 101, thinkClose = 200
    private static let marks = ToolCallMarks(open: open, close: close)

    /// How many of `tokens` the reply has when it ends for length; nil if it never does.
    private func length(_ tokens: [Int], maxTokens: Int, overrun: Int, marks: ToolCallMarks? = marks) -> Int? {
        var limit = ReplyLimit(maxTokens: maxTokens, marks: marks, overrun: overrun, resets: Self.thinkClose)
        for (index, token) in tokens.enumerated() where limit.reached(by: token, generated: index + 1) { return index + 1 }
        return nil
    }

    private func text(_ count: Int) -> [Int] { Array(repeating: 1, count: count) }

    @Test("a reply in no tool call ends at its limit")
    func plain() {
        #expect(length(text(50), maxTokens: 20, overrun: 30) == 20)
        #expect(length(text(10), maxTokens: 20, overrun: 30) == nil)
        #expect(length(text(5) + [Self.open] + text(3) + [Self.close] + text(50), maxTokens: 20, overrun: 30) == 20)   // a call made and closed earlier
    }

    @Test("a tool call open at the limit runs on to its close, and the reply ends there")
    func finishesTheCall() {
        let reply = text(15) + [Self.open] + text(12) + [Self.close] + text(40)
        #expect(length(reply, maxTokens: 20, overrun: 30) == 29)                // the close is its 29th token
        // The close falling exactly on the limit is the same reply ending in time.
        #expect(length(text(15) + [Self.open] + text(3) + [Self.close] + text(9), maxTokens: 20, overrun: 30) == 20)
    }

    @Test("a second call is not waited for")
    func oneCall() {
        let reply = text(15) + [Self.open] + text(8) + [Self.close, 1, Self.open] + text(40)
        #expect(length(reply, maxTokens: 20, overrun: 30) == 25)
    }

    @Test("a call that does not close is cut when the overrun is spent")
    func overrunSpent() {
        #expect(length(text(15) + [Self.open] + text(100), maxTokens: 20, overrun: 30) == 50)
    }

    @Test("no overrun, or no marks, is the limit as it was")
    func off() {
        let reply = text(15) + [Self.open] + text(100)
        #expect(length(reply, maxTokens: 20, overrun: 0) == 20)
        #expect(length(reply, maxTokens: 20, overrun: -5) == 20)
        #expect(length(reply, maxTokens: 20, overrun: 30, marks: nil) == 20)
    }

    @Test("an opening token inside the thinking is not a call in progress")
    func talkOfACall() {
        let reply = text(5) + [Self.open] + text(5) + [Self.thinkClose] + text(50)
        #expect(length(reply, maxTokens: 20, overrun: 30) == 20)
    }

    /// What the endpoint makes of the reply the scheduler then ends: the scheduler says it
    /// ended for length, and the client is told it ended in tool calls, with the call whole.
    @Test("a reply that ends for length on a whole tool call is reported as tool calls")
    func reported() async throws {
        let root = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
        let tokenizer = try Tokenizer(tokenizerURL: root.appendingPathComponent("inputs/tokenizer/tokenizer.json"),
                                      configURL: root.appendingPathComponent("inputs/tokenizer/tokenizer_config.json"))
        let marks = try #require(ChatEndpoint.toolCallMarks(tokenizer))
        let reply = "<tool_call>\n<function=write>\n<parameter=path>\na.swift\n</parameter>\n<parameter=content>\nlet a = 1\n</parameter>\n</function>\n</tool_call>"
        let tokens = tokenizer.encode(reply)
        #expect(tokens.first == marks.open && tokens.last == marks.close)

        // The limit falls inside the call; the scheduler's rule ends the reply on the close.
        var limit = ReplyLimit(maxTokens: 5, marks: marks, overrun: 1000)
        let ends = tokens.indices.first { limit.reached(by: tokens[$0], generated: $0 + 1) }
        #expect(ends == tokens.count - 1)

        let port = 18_197
        let router = Router()
        router.post("/v1/chat/completions") { _, _ -> Response in
            let events = AsyncStream<GenerationEvent> { (continuation: AsyncStream<GenerationEvent>.Continuation) in
                for token in tokens { continuation.yield(.token(token)) }
                continuation.yield(.finished(.length, GenerationUsage(), message: nil))
                continuation.finish()
            }
            let prepared = ChatEndpoint.Prepared(
                request: GenerationRequest(promptTokens: [1, 2, 3], maxTokens: 5, stopTokenIDs: [], vocabLimit: tokenizer.idLimit),
                model: "m", stream: false, includeUsage: false, startsInThinking: false, tools: [], stopStrings: [])
            return await ChatEndpoint.complete(events, prepared: prepared, tokenizer: tokenizer, id: "x", created: 0)
        }
        let app = Application(responder: router.buildResponder(), configuration: .init(address: .hostname("127.0.0.1", port: port)))
        let server = Task { try await app.runService(gracefulShutdownSignals: [.sigusr2]) }
        defer { server.cancel() }
        try await Task.sleep(for: .milliseconds(700))

        var request = URLRequest(url: URL(string: "http://127.0.0.1:\(port)/v1/chat/completions")!)
        request.httpMethod = "POST"
        request.httpBody = Data("{}".utf8)
        let (data, _) = try await URLSession.shared.data(for: request)
        let choice = try JSONValue.parse(Array(data))["choices"]?.arrayValue?.first
        #expect(choice?["finish_reason"]?.stringValue == "tool_calls")
        let call = choice?["message"]?["tool_calls"]?.arrayValue?.first?["function"]
        #expect(call?["name"]?.stringValue == "write")
        let arguments = try JSONValue.parse(Array((call?["arguments"]?.stringValue ?? "").utf8))
        #expect(arguments["path"]?.stringValue == "a.swift")
        #expect(arguments["content"]?.stringValue == "let a = 1")
    }
}
