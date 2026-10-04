import Foundation
import Testing
import Hummingbird
import SploshRuntime
@testable import SploshServer

/// A reply that reaches its token limit while the model is still thinking is reasoning and
/// nothing else. A harness leaves an assistant message with no content and no tool call out of
/// the next request, and the reasoning goes with it: asked to continue, the model starts its
/// thinking again. So such a reply is given content, and comes back whole.
@Suite("Thinking and no answer", .serialized)
struct ThinkingOnlyReplyTests {
    private static let thinking = "The user wants one file. Let me work out what goes in it before I write anything: first the types, then the"
    private static let close = "\n</think>\n\n"

    private func loadTokenizer() throws -> Tokenizer {
        let root = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
        return try Tokenizer(tokenizerURL: root.appendingPathComponent("inputs/tokenizer/tokenizer.json"),
                             configURL: root.appendingPathComponent("inputs/tokenizer/tokenizer_config.json"))
    }

    /// What a client is sent when the model writes `text` and the session then ends: at the
    /// token limit, or with a stop token.
    private func cutReply(_ text: String, ending: FinishReason = .length, stream: Bool, port: Int) async throws -> String {
        let tokenizer = try loadTokenizer()
        let tokens = tokenizer.encode(text)
        let router = Router()
        router.post("/v1/chat/completions") { _, _ -> Response in
            let events = AsyncStream<GenerationEvent> { (continuation: AsyncStream<GenerationEvent>.Continuation) in
                for token in tokens { continuation.yield(.token(token)) }
                continuation.yield(.finished(ending, GenerationUsage(), message: nil))
                continuation.finish()
            }
            let prepared = ChatEndpoint.Prepared(
                request: GenerationRequest(promptTokens: [1, 2, 3], maxTokens: tokens.count, stopTokenIDs: [], vocabLimit: tokenizer.idLimit),
                model: "m", stream: stream, includeUsage: false, startsInThinking: true, tools: [], stopStrings: [])
            return stream
                ? ChatEndpoint.streaming(events, prepared: prepared, tokenizer: tokenizer, id: "x", created: 0)
                : await ChatEndpoint.complete(events, prepared: prepared, tokenizer: tokenizer, id: "x", created: 0)
        }
        let app = Application(responder: router.buildResponder(), configuration: .init(address: .hostname("127.0.0.1", port: port)))
        let server = Task { try await app.runService(gracefulShutdownSignals: [.sigusr2]) }
        defer { server.cancel() }
        try await Task.sleep(for: .milliseconds(700))

        var request = URLRequest(url: URL(string: "http://127.0.0.1:\(port)/v1/chat/completions")!)
        request.httpMethod = "POST"
        request.httpBody = Data("{}".utf8)
        let (data, _) = try await URLSession.shared.data(for: request)
        return String(decoding: data, as: UTF8.self)
    }

    @Test("a streamed reply cut while thinking ends with content, after its reasoning")
    func streamed() async throws {
        let body = try await cutReply(Self.thinking, stream: true, port: 18_194)
        let choices = try body.split(separator: "\n").compactMap { line -> JSONValue? in
            guard line.hasPrefix("data: "), line != "data: [DONE]" else { return nil }
            return try JSONValue.parse(Array(line.dropFirst(6).utf8))["choices"]?.arrayValue?.first
        }
        #expect(choices.compactMap { $0["delta"]?["reasoning_content"]?.stringValue }.joined() == Self.thinking)
        #expect(choices.compactMap { $0["delta"]?["content"]?.stringValue }.joined() == ChatEndpoint.cutWhileThinking)
        #expect(choices.compactMap { $0["finish_reason"]?.stringValue } == ["length"])
        #expect(body.contains("[DONE]"))
    }

    @Test("a whole-body reply cut while thinking has the same content")
    func whole() async throws {
        let body = try await cutReply(Self.thinking, stream: false, port: 18_195)
        let choice = try JSONValue.parse(Array(body.utf8))["choices"]?.arrayValue?.first
        #expect(choice?["message"]?["reasoning_content"]?.stringValue == Self.thinking)
        #expect(choice?["message"]?["content"]?.stringValue == ChatEndpoint.cutWhileThinking)
        #expect(choice?["finish_reason"]?.stringValue == "length")
    }

    @Test("a reply the model ended after thinking only is given content too")
    func endedByTheModel() async throws {
        let body = try await cutReply(Self.thinking + Self.close, ending: .stop, stream: false, port: 18_196)
        let choice = try JSONValue.parse(Array(body.utf8))["choices"]?.arrayValue?.first
        // The newline before </think> is the reasoning's, as the parser has it; the template trims it.
        #expect(choice?["message"]?["reasoning_content"]?.stringValue == Self.thinking + "\n")
        #expect(choice?["message"]?["content"]?.stringValue == ChatEndpoint.endedAfterThinking)
        #expect(choice?["finish_reason"]?.stringValue == "stop")
    }

    /// Thinking the scheduler ends itself (ThinkingWatch) is closed with these tokens.
    @Test("a thinking block closed by the scheduler reads as one the model closed, and comes back as generated")
    func closedByTheScheduler() throws {
        let tokenizer = try loadTokenizer()
        let close = try #require(ChatEndpoint.thinkingClose(tokenizer))
        #expect(tokenizer.decode(close.tokens) == Self.close)
        #expect(close.tokens.filter { $0 == close.token }.count == 1)

        // The reply: thinking cut where it stood, the closing, then the answer.
        let answer = "The file is written."
        let generated = tokenizer.encode(Self.thinking) + close.tokens + tokenizer.encode(answer)
        var parser = ChatOutputParser(startsInThinking: true)
        var events: [ChatOutputParser.Event] = []
        for token in generated { events += parser.push(tokenizer.bytes(for: token) ?? []) }
        events += parser.finish()
        var reasoning = "", content = ""
        for case .reasoning(let text) in events { reasoning += text }
        for case .content(let text) in events { content += text }
        #expect(reasoning == Self.thinking + "\n")
        #expect(content == answer)

        let history: [JSONValue] = [.object([("role", .string("user")), ("content", .string("Write the file."))])]
        let prompt = tokenizer.encode(try ChatTemplate.render(messages: history, tools: [], options: .init()).prompt)
        let next = try ChatTemplate.render(messages: history + [
            .object([("role", .string("assistant")), ("content", .string(content)), ("reasoning_content", .string(reasoning))]),
            .object([("role", .string("user")), ("content", .string("Thanks."))]),
        ], tools: [], options: .init()).prompt
        #expect(tokenizer.encode(next).starts(with: prompt + generated))
    }

    @Test("only a reply that is thinking and no answer is given it, and by how it ended")
    func scope() {
        func substance(_ text: String, startsInThinking: Bool = true) -> ChatEndpoint.Substance {
            var parser = ChatOutputParser(startsInThinking: startsInThinking)
            var substance = ChatEndpoint.Substance()
            substance.note(parser.push(Array(text.utf8)))
            substance.note(parser.finish())
            return substance
        }
        /// A reply that ends inside a tool call, which the client drops.
        func insideACall(_ text: String) -> String? {
            var parser = ChatOutputParser(startsInThinking: true)
            var substance = ChatEndpoint.Substance()
            substance.note(parser.push(Array(text.utf8)))
            substance.note(parser.finish())
            #expect(parser.unfinishedToolCall)
            return ChatEndpoint.standIn(for: .length, substance, unfinishedCall: parser.unfinishedToolCall)
        }
        let t = Self.thinking, c = Self.close
        #expect(ChatEndpoint.standIn(for: .length, substance(t)) == ChatEndpoint.cutWhileThinking)
        #expect(ChatEndpoint.standIn(for: .length, substance(t + c)) == ChatEndpoint.cutWhileThinking)   // the limit fell just after </think>
        #expect(ChatEndpoint.standIn(for: .stop, substance(t)) == ChatEndpoint.endedAfterThinking)       // the model stopped inside its thinking
        #expect(ChatEndpoint.standIn(for: .stop, substance(t + c)) == ChatEndpoint.endedAfterThinking)   // or closed it and said nothing
        #expect(ChatEndpoint.standIn(for: .stop, substance(t + c + "Done.")) == nil)
        #expect(ChatEndpoint.standIn(for: .stop, substance(t + c), stopString: true) == nil)             // what the request asked for
        #expect(ChatEndpoint.standIn(for: .cancelled, substance(t)) == nil)                              // a broken connection, not a reply
        #expect(ChatEndpoint.standIn(for: .length, substance(t + c + "The file is")) == nil)  // an answer was begun
        // Cut inside a tool call: the call is dropped, and what is left must not be thinking alone.
        let cutCall = "<tool_call>\n<function=write>\n<parameter=path>\na.sw"
        #expect(insideACall(t + c + cutCall) == ChatEndpoint.endedInsideToolCall)
        #expect(insideACall(t + c + "I'll write it now.\n\n" + cutCall) == nil)                         // there is content to keep
        #expect(ChatEndpoint.standIn(for: .length, substance("The file is", startsInThinking: false)) == nil)
        #expect(ChatEndpoint.standIn(for: .length, substance(" \n")) == nil)                   // no thinking to carry back
    }

    /// The message as a harness sends it back: the content it was given, and the reasoning.
    @Test("sent back, the thinking is in the next prompt, which runs on from what the server holds")
    func comesBack() throws {
        let tokenizer = try loadTokenizer()
        let history: [JSONValue] = [
            .object([("role", .string("system")), ("content", .string("You are a coding agent."))]),
            .object([("role", .string("user")), ("content", .string("Write the file."))]),
        ]
        let prompt = tokenizer.encode(try ChatTemplate.render(messages: history, tools: [], options: .init()).prompt)
        // At the token limit the last token sampled goes to the client and is never evaluated.
        let slot = prompt + tokenizer.encode(Self.thinking).dropLast()

        let next = try ChatTemplate.render(messages: history + [
            .object([("role", .string("assistant")), ("content", .string(ChatEndpoint.cutWhileThinking)),
                     ("reasoning_content", .string(Self.thinking))]),
            .object([("role", .string("user")), ("content", .string("continue"))]),
        ], tools: [], options: .init()).prompt
        #expect(next.contains(Self.thinking + Self.close + ChatEndpoint.cutWhileThinking))
        #expect(tokenizer.encode(next).starts(with: slot))
    }
}
