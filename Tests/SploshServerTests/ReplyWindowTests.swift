import Foundation
import Testing
import Hummingbird
import SploshRuntime
@testable import SploshServer

/// The dashboard shows a reply while it is written. What it is given must be what the client
/// was sent: the same thinking, the same answer, the same tool call, in the order they came.
@Suite("Reply being written", .serialized)
struct ReplyWindowTests {
    private static let text = "The directory first, then the file.\n</think>\n\nListing it now.\n\n<tool_call>\n<function=bash>\n<parameter=command>\nls -la\n</parameter>\n</function>\n</tool_call>"
    private static let tools: [JSONValue] = (try? JSONValue.parse("""
    [{"type": "function", "function": {"name": "bash", "parameters": {"type": "object", "properties": {"command": {"type": "string"}}}}}]
    """).arrayValue) ?? []

    private func loadTokenizer() throws -> Tokenizer {
        let root = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
        return try Tokenizer(tokenizerURL: root.appendingPathComponent("inputs/tokenizer/tokenizer.json"),
                             configURL: root.appendingPathComponent("inputs/tokenizer/tokenizer_config.json"))
    }

    private struct Seen {
        var client: String
        var window: JSONValue
        var missing: (status: Int, body: JSONValue)
    }

    /// The model writes `text` as session 7: what the client is sent, what the window is then
    /// given for that session, and what it is given for a session that is not running.
    private func write(stream: Bool, port: Int) async throws -> Seen {
        let tokenizer = try loadTokenizer()
        let tokens = tokenizer.encode(Self.text)
        let activity = SessionActivity()
        let router = Router()
        router.post("/v1/chat/completions") { _, _ -> Response in
            let events = AsyncStream<GenerationEvent> { (continuation: AsyncStream<GenerationEvent>.Continuation) in
                for token in tokens { continuation.yield(.token(token)) }
                continuation.yield(.finished(.stop, GenerationUsage(), message: nil))
                continuation.finish()
            }
            var request = GenerationRequest(promptTokens: [1, 2, 3], maxTokens: tokens.count, stopTokenIDs: [], vocabLimit: tokenizer.idLimit)
            request.activity = activity
            let prepared = ChatEndpoint.Prepared(request: request, model: "m", stream: stream, includeUsage: false,
                                                 startsInThinking: true, tools: Self.tools, stopStrings: [])
            return stream
                ? ChatEndpoint.streaming(events, prepared: prepared, tokenizer: tokenizer, id: "x", created: 0)
                : await ChatEndpoint.complete(events, prepared: prepared, tokenizer: tokenizer, id: "x", created: 0)
        }
        router.get("/v1/sessions/:id/reply") { _, context -> Response in
            let id = context.parameters.get("id", as: Int.self)
            return ChatEndpoint.reply(of: id, id == 7 ? activity : nil)
        }
        let app = Application(responder: router.buildResponder(), configuration: .init(address: .hostname("127.0.0.1", port: port)))
        let server = Task { try await app.runService(gracefulShutdownSignals: [.sigusr2]) }
        defer { server.cancel() }
        try await Task.sleep(for: .milliseconds(700))

        var request = URLRequest(url: URL(string: "http://127.0.0.1:\(port)/v1/chat/completions")!)
        request.httpMethod = "POST"
        request.httpBody = Data("{}".utf8)
        let (sent, _) = try await URLSession.shared.data(for: request)
        let (window, _) = try await URLSession.shared.data(from: URL(string: "http://127.0.0.1:\(port)/v1/sessions/7/reply")!)
        let (missing, response) = try await URLSession.shared.data(from: URL(string: "http://127.0.0.1:\(port)/v1/sessions/8/reply")!)
        return Seen(client: String(decoding: sent, as: UTF8.self), window: try JSONValue.parse(Array(window)),
                    missing: ((response as? HTTPURLResponse)?.statusCode ?? 0, try JSONValue.parse(Array(missing))))
    }

    private func parts(_ window: JSONValue) -> [[String]] {
        (window["parts"]?.arrayValue ?? []).map { [$0["kind"]?.stringValue ?? "", $0["name"]?.stringValue ?? "", $0["text"]?.stringValue ?? ""] }
    }

    @Test("the window is given a streamed reply as the client was sent it")
    func streamed() async throws {
        let seen = try await write(stream: true, port: 18_211)
        let deltas = try seen.client.split(separator: "\n").compactMap { line -> JSONValue? in
            guard line.hasPrefix("data: "), line != "data: [DONE]" else { return nil }
            return try JSONValue.parse(Array(line.dropFirst(6).utf8))["choices"]?.arrayValue?.first?["delta"]
        }
        let reasoning = deltas.compactMap { $0["reasoning_content"]?.stringValue }.joined()
        let content = deltas.compactMap { $0["content"]?.stringValue }.joined()
        let calls = deltas.compactMap { $0["tool_calls"]?.arrayValue?.first?["function"] }
        #expect(!reasoning.isEmpty && !content.isEmpty)
        #expect(seen.window["id"]?.intValue == 7)
        #expect(parts(seen.window) == [["thinking", "", reasoning], ["answer", "", content],
                                       ["tool call", "bash", calls.compactMap { $0["arguments"]?.stringValue }.joined()]])
        #expect(calls.compactMap { $0["name"]?.stringValue } == ["bash"])
        #expect(try JSONValue.parse(Array(parts(seen.window)[2][2].utf8))["command"]?.stringValue == "ls -la")
    }

    @Test("and a whole-body reply as it will be sent")
    func whole() async throws {
        let seen = try await write(stream: false, port: 18_212)
        let message = try JSONValue.parse(Array(seen.client.utf8))["choices"]?.arrayValue?.first?["message"]
        let call = message?["tool_calls"]?.arrayValue?.first?["function"]
        #expect(parts(seen.window) == [["thinking", "", message?["reasoning_content"]?.stringValue ?? "?"],
                                       ["answer", "", message?["content"]?.stringValue ?? "?"],
                                       ["tool call", "bash", call?["arguments"]?.stringValue ?? "?"]])
    }

    @Test("a session that is not running is not found")
    func missing() async throws {
        let seen = try await write(stream: true, port: 18_213)
        #expect(seen.missing.status == 404)
        #expect(seen.missing.body["error"]?["code"]?.stringValue == "session_not_found")
    }

    @Test("a session that has left the stats can still be read, until enough others have ended")
    func ended() {
        var index = SessionActivities()
        let first = SessionActivity(), second = SessionActivity()
        index.update([1: first, 2: second])
        #expect(index[1] === first && index[2] === second && index[3] == nil)
        index.update([2: second])
        #expect(index[1] === first)
        // One session after another beside the second, each ending as the next begins.
        let others = 10..<(10 + SessionActivities.endedKept)
        for id in others { index.update([2: second, id: SessionActivity()]) }
        index.update([2: second])
        #expect(index[1] == nil && index[2] === second)
        #expect(others.allSatisfy { index[$0] != nil })
    }
}
