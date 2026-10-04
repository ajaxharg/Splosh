import Foundation
import Testing
import Hummingbird
import SploshRuntime
@testable import SploshServer

/// A generator that takes its time: one word every `interval`, `count` of them.
private struct SlowService: InferenceService {
    let count: Int
    let interval: Duration
    func infer(_ request: InferenceRequest) -> AsyncThrowingStream<InferenceEvent, Error> {
        AsyncThrowingStream { continuation in
            let task = Task {
                for index in 0..<count {
                    try? await Task.sleep(for: interval)
                    if Task.isCancelled { break }
                    continuation.yield(.delta("w\(index) "))
                }
                continuation.yield(.finished(reason: Task.isCancelled ? "cancelled" : "stop"))
                continuation.finish()
            }
            continuation.onTermination = { _ in task.cancel() }
        }
    }
}

@Suite("Controlled stop", .serialized)
struct ControlledStopTests {
    /// The server is asked to stop (the signal a Ctrl+C sends) while a reply is streaming. The
    /// reply has to carry on to its end; only then may the server go.
    @Test("a reply in flight when the server is asked to stop is finished, not cut")
    func drains() async throws {
        let port = 18_191
        let app = Server.application(service: SlowService(count: 30, interval: .milliseconds(100)), port: port, contextWindow: 4096)
        let server = Task { try await app.runService(gracefulShutdownSignals: [.sigusr1]) }
        try await Task.sleep(for: .milliseconds(700))

        var request = URLRequest(url: URL(string: "http://127.0.0.1:\(port)/v1/chat/completions")!)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.httpBody = Data(#"{"model":"m","stream":true,"messages":[{"role":"user","content":"go"}]}"#.utf8)
        let started = Date()
        let (bytes, _) = try await URLSession.shared.bytes(for: request)
        var words = 0, done = false, signalled = false
        for try await line in bytes.lines {
            if line.contains("\"content\":\"w") { words += 1 }
            if line.contains("[DONE]") { done = true }
            if words == 3, !signalled { signalled = true; kill(getpid(), SIGUSR1) }
        }
        let took = Date().timeIntervalSince(started)
        #expect(words == 30, "the reply was cut after \(words) of 30 words, \(took) s in")
        #expect(done)
        try await server.value
    }

    /// What a client sees of a reply the scheduler ended because the server is stopping.
    private func cutReply(stream: Bool, port: Int) async throws -> (status: Int?, text: String, complete: Bool) {
        let root = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
        let tokenizer = try Tokenizer(tokenizerURL: root.appendingPathComponent("inputs/tokenizer/tokenizer.json"),
                                      configURL: root.appendingPathComponent("inputs/tokenizer/tokenizer_config.json"))
        let words = tokenizer.encode("The answer so far is only half of")
        let router = Router()
        router.post("/v1/chat/completions") { _, _ -> Response in
            let events = AsyncStream<GenerationEvent> { (continuation: AsyncStream<GenerationEvent>.Continuation) in
                Task {
                    // Long enough over the prompt to have reported progress, as a real one would.
                    continuation.yield(.progress(evaluated: 10, total: 100))
                    for token in words {
                        try? await Task.sleep(for: .milliseconds(20))
                        continuation.yield(.token(token))
                    }
                    continuation.yield(.finished(.cancelled, GenerationUsage(), message: "server shutting down"))
                    continuation.finish()
                }
            }
            let prepared = ChatEndpoint.Prepared(
                request: GenerationRequest(promptTokens: [1, 2, 3], maxTokens: 64, stopTokenIDs: [], vocabLimit: tokenizer.idLimit),
                model: "m", stream: stream, includeUsage: false, startsInThinking: false, tools: [], stopStrings: [])
            return stream
                ? ChatEndpoint.streaming(events, prepared: prepared, tokenizer: tokenizer, id: "x", created: 0)
                : await ChatEndpoint.complete(events, prepared: prepared, tokenizer: tokenizer, id: "x", created: 0)
        }
        let app = Application(responder: router.buildResponder(), configuration: .init(address: .hostname("127.0.0.1", port: port)))
        let server = Task { try await app.runService(gracefulShutdownSignals: [.sigusr2]) }
        defer { server.cancel() }
        try await Task.sleep(for: .milliseconds(700))

        return Self.rawExchange(port: port)
    }

    /// One request over a plain socket, and everything the server sent back until it closed
    /// the connection: URLSession forgives a body that stops without its last chunk, which is
    /// the very thing to be told apart here.
    private static func rawExchange(port: Int) -> (status: Int?, text: String, complete: Bool) {
        let fd = socket(AF_INET, SOCK_STREAM, 0)
        defer { close(fd) }
        var address = sockaddr_in()
        address.sin_family = sa_family_t(AF_INET)
        address.sin_port = in_port_t(port).bigEndian
        address.sin_addr.s_addr = inet_addr("127.0.0.1")
        let connected = withUnsafePointer(to: &address) {
            $0.withMemoryRebound(to: sockaddr.self, capacity: 1) { connect(fd, $0, socklen_t(MemoryLayout<sockaddr_in>.size)) }
        }
        guard connected == 0 else { return (nil, "", false) }
        var timeout = timeval(tv_sec: 10, tv_usec: 0)
        setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &timeout, socklen_t(MemoryLayout<timeval>.size))
        let request = Array("POST /v1/chat/completions HTTP/1.1\r\nHost: test\r\nConnection: close\r\nContent-Length: 2\r\n\r\n{}".utf8)
        _ = request.withUnsafeBytes { write(fd, $0.baseAddress, $0.count) }
        var received: [UInt8] = []
        var buffer = [UInt8](repeating: 0, count: 65_536)
        while true {
            let count = read(fd, &buffer, buffer.count)
            if count <= 0 { break }
            received.append(contentsOf: buffer[0..<count])
        }
        let text = String(decoding: received, as: UTF8.self)
        let status = text.split(separator: " ").dropFirst().first.flatMap { Int($0) }
        // A chunked body is whole only if it ends with the empty chunk.
        return (status, text, text.hasSuffix("0\r\n\r\n"))
    }

    @Test("a streamed reply cut by a stopping server does not look like one that finished")
    func cutStream() async throws {
        let seen = try await cutReply(stream: true, port: 18_192)
        #expect(seen.text.contains("half"))                             // what there was of it arrived
        #expect(!seen.text.contains("[DONE]"))
        #expect(!seen.text.contains("\"finish_reason\":\"stop\""))
        #expect(!seen.complete, "the body should be left unfinished, not ended cleanly")
    }

    @Test("a whole-body reply cut by a stopping server is a failure, not a short answer")
    func cutBody() async throws {
        let seen = try await cutReply(stream: false, port: 18_193)
        #expect(seen.status == 503 || !seen.complete)
        #expect(!seen.text.contains("\"finish_reason\""))
    }
}
