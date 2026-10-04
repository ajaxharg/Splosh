import Foundation
import Testing
import Hummingbird
import SploshRuntime
@testable import SploshServer

/// A reply cut at the token limit its client set ends in a call to the client's shell tool,
/// when the server is set to, so the harness goes on where it would have ended the turn.
@Suite("Continue at the limit", .serialized)
struct LimitContinuationTests {
    private static let bash: JSONValue = (try? JSONValue.parse(Array(("""
    {"type":"function","function":{"name":"bash","parameters":{"type":"object",
     "properties":{"command":{"type":"string"},"description":{"type":"string"}},"required":["command","description"]}}}
    """).utf8))) ?? .null

    @Test("a shell tool's required arguments are filled in")
    func finds() throws {
        let call = try #require(LimitContinuation.find(in: [Self.bash]))
        #expect(call.name == "bash")
        let arguments = try JSONValue.parse(Array(call.arguments.utf8))
        #expect(arguments["command"]?.stringValue?.hasPrefix("echo '[Output limit reached") == true)
        #expect(arguments["description"]?.stringValue?.isEmpty == false)
    }

    @Test("no shell tool, or one needing an argument that is not text, gives none")
    func none() throws {
        let other = try JSONValue.parse(Array(#"{"type":"function","function":{"name":"read","parameters":{"properties":{"path":{"type":"string"}}}}}"#.utf8))
        #expect(LimitContinuation.find(in: [other]) == nil)
        #expect(LimitContinuation.find(in: []) == nil)
        let odd = try JSONValue.parse(Array(#"{"type":"function","function":{"name":"bash","parameters":{"properties":{"command":{"type":"string"},"timeout":{"type":"integer"}},"required":["command","timeout"]}}}"#.utf8))
        #expect(LimitContinuation.find(in: [odd]) == nil)
    }

    private func loadTokenizer() throws -> Tokenizer {
        let root = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
        return try Tokenizer(tokenizerURL: root.appendingPathComponent("inputs/tokenizer/tokenizer.json"),
                             configURL: root.appendingPathComponent("inputs/tokenizer/tokenizer_config.json"))
    }

    /// What a client is sent when the model writes `text` and the session then ends at its limit.
    private func cutReply(_ text: String, continuation: LimitContinuation?, stream: Bool, port: Int) async throws -> String {
        let tokenizer = try loadTokenizer()
        let tokens = tokenizer.encode(text)
        let router = Router()
        router.post("/v1/chat/completions") { _, _ -> Response in
            let events = AsyncStream<GenerationEvent> { (continuation: AsyncStream<GenerationEvent>.Continuation) in
                for token in tokens { continuation.yield(.token(token)) }
                continuation.yield(.finished(.length, GenerationUsage(), message: nil))
                continuation.finish()
            }
            let prepared = ChatEndpoint.Prepared(
                request: GenerationRequest(promptTokens: [1, 2, 3], maxTokens: tokens.count, stopTokenIDs: [], vocabLimit: tokenizer.idLimit),
                model: "m", stream: stream, includeUsage: false, startsInThinking: false, tools: [Self.bash], stopStrings: [],
                limitContinuation: continuation)
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

    private static let text = "Here is the first part of the file, and then"

    @Test("a streamed reply cut at the limit ends in the call, finishing as tool_calls")
    func streamed() async throws {
        let body = try await cutReply(Self.text, continuation: LimitContinuation.find(in: [Self.bash]), stream: true, port: 18_195)
        let choices = try body.split(separator: "\n").compactMap { line -> JSONValue? in
            guard line.hasPrefix("data: "), line != "data: [DONE]" else { return nil }
            return try JSONValue.parse(Array(line.dropFirst(6).utf8))["choices"]?.arrayValue?.first
        }
        #expect(choices.compactMap { $0["delta"]?["content"]?.stringValue }.joined() == Self.text)
        let calls = choices.compactMap { $0["delta"]?["tool_calls"]?.arrayValue?.first }
        #expect(calls.count == 1)
        #expect(calls.first?["function"]?["name"]?.stringValue == "bash")
        #expect(calls.first?["function"]?["arguments"]?.stringValue?.contains("echo") == true)
        #expect(choices.last?["finish_reason"]?.stringValue == "tool_calls")
    }

    @Test("a whole reply cut at the limit ends in the call")
    func whole() async throws {
        let body = try await cutReply(Self.text, continuation: LimitContinuation.find(in: [Self.bash]), stream: false, port: 18_196)
        let choice = try #require(try JSONValue.parse(Array(body.trimmingCharacters(in: .whitespacesAndNewlines).utf8))["choices"]?.arrayValue?.first)
        #expect(choice["finish_reason"]?.stringValue == "tool_calls")
        #expect(choice["message"]?["content"]?.stringValue == Self.text)
        #expect(choice["message"]?["tool_calls"]?.arrayValue?.first?["function"]?["name"]?.stringValue == "bash")
    }

    @Test("without the setting a cut reply is still told it hit the limit")
    func off() async throws {
        let body = try await cutReply(Self.text, continuation: nil, stream: false, port: 18_197)
        let choice = try #require(try JSONValue.parse(Array(body.trimmingCharacters(in: .whitespacesAndNewlines).utf8))["choices"]?.arrayValue?.first)
        #expect(choice["finish_reason"]?.stringValue == "length")
        #expect(choice["message"]?["tool_calls"] == nil)
    }
}
