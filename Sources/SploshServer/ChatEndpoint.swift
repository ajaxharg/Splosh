import Foundation
import Hummingbird
import HTTPTypes
import NIOCore
import SploshRuntime

/// Everything the HTTP layer needs to serve the model.
public struct ChatBackend: Sendable {
    public let scheduler: BatchScheduler
    public let tokenizer: Tokenizer
    public let modelID: String
    public let maxContext: Int
    /// Replies as generated, for prompts that carry them back as text (see ReplyAliases).
    public let aliases = ReplyAliases()
    /// Settings that can change while the server runs.
    public let options: ReplyOptions

    public init(scheduler: BatchScheduler, tokenizer: Tokenizer, modelID: String, maxContext: Int,
                options: ReplyOptions = ReplyOptions()) {
        self.scheduler = scheduler; self.tokenizer = tokenizer
        self.modelID = modelID; self.maxContext = maxContext
        self.options = options
    }
}

/// How replies are ended, as the settings page has it.
public final class ReplyOptions: @unchecked Sendable {
    private let lock = NSLock()
    private var _continueOnLimit: Bool
    /// A reply cut at its client's token limit ends with a call to the client's shell tool
    /// instead (see `LimitContinuation`).
    public var continueOnLimit: Bool {
        get { lock.lock(); defer { lock.unlock() }; return _continueOnLimit }
        set { lock.lock(); _continueOnLimit = newValue; lock.unlock() }
    }
    public init(continueOnLimit: Bool = false) { _continueOnLimit = continueOnLimit }
}

/// A call to the client's shell tool, added to a reply that reached its token limit.
///
/// A harness takes `finish_reason: "length"` for the end of a turn: a one-shot sub-agent has
/// no "continue" to send, and the work is lost. A reply that ends in a tool call is a turn that
/// goes on: the harness runs the call and sends the conversation back with a fresh limit. The
/// call echoes a note to the model, so what it reads next is to carry on.
struct LimitContinuation: Sendable {
    let name: String
    let arguments: String

    static let note = "[Output limit reached. Carry on exactly where the last reply stopped; do not repeat what is above.]"
    private static let shellNames: Set<String> = ["bash", "shell", "sh", "run_command", "execute_bash", "terminal"]

    /// The first shell-like tool among `tools` whose required arguments can all be filled in:
    /// the command is an `echo` of the note, any other required text is a short fixed word.
    static func find(in tools: [JSONValue]) -> LimitContinuation? {
        for tool in tools {
            let function = tool["function"] ?? tool
            guard let name = function["name"]?.stringValue, shellNames.contains(name.lowercased()) else { continue }
            let schema = function["parameters"] ?? function["input_schema"]
            let properties = schema?["properties"]?.objectValue ?? []
            let required = Set((schema?["required"]?.arrayValue ?? []).compactMap(\.stringValue))
            var members: [(key: String, value: JSONValue)] = []
            var filled = false
            for (key, property) in properties {
                let type = property["type"]?.stringValue
                if key == "command" || key == "cmd" {
                    guard type == nil || type == "string" else { break }
                    members.append((key, .string("echo '\(note)'"))); filled = true
                } else if required.contains(key) {
                    guard type == nil || type == "string" else { filled = false; break }
                    members.append((key, .string(property["enum"]?.arrayValue?.first?.stringValue ?? "Continue after the output limit")))
                }
            }
            if filled, required.isSubset(of: Set(members.map(\.key))) {
                return LimitContinuation(name: name, arguments: JSONValue.object(members).compact)
            }
        }
        return nil
    }
}

extension SessionActivity {
    /// The reply as it is parsed, kept as text for whoever watches it being written.
    func note(_ events: [ChatOutputParser.Event]) {
        for event in events {
            switch event {
            case .reasoning(let text): write(text, as: .thinking)
            case .content(let text): write(text, as: .answer)
            case .toolCallStart(_, _, let name): beginToolCall(name)
            case .toolCallArguments(_, let piece): write(piece, as: .toolCall)
            case .toolCallEnd: break
            }
        }
    }
}

/// `POST /v1/chat/completions`.
enum ChatEndpoint {
    struct Prepared {
        let request: GenerationRequest
        let model: String
        let stream: Bool
        let includeUsage: Bool
        let startsInThinking: Bool
        let tools: [JSONValue]
        let stopStrings: [String]
        /// Set when a reply cut at the limit the client asked for is to end in a tool call.
        var limitContinuation: LimitContinuation? = nil
    }

    struct APIError: Error {
        let status: HTTPResponse.Status
        let type: String
        let code: String?
        let message: String

        var body: JSONValue {
            .object([("error", .object([("message", .string(message)), ("type", .string(type)),
                                         ("param", .null), ("code", code.map(JSONValue.string) ?? .null)]))])
        }
    }

    static func json(_ value: JSONValue, status: HTTPResponse.Status = .ok) -> Response {
        Response(status: status, headers: [.contentType: "application/json"],
                 body: ResponseBody(byteBuffer: ByteBuffer(string: value.compact)))
    }

    // MARK: Request

    static func prepare(_ bytes: [UInt8], backend: ChatBackend) throws -> Prepared {
        let started = DispatchTime.now().uptimeNanoseconds
        return try prepare(parse(bytes), backend: backend, started: started)
    }

    static func parse(_ bytes: [UInt8]) throws -> JSONValue {
        do { return try JSONValue.parse(bytes) } catch {
            throw APIError(status: .badRequest, type: "invalid_request_error", code: nil, message: "\(error)")
        }
    }

    /// `started` is when the request's body was in hand, before it was parsed.
    static func prepare(_ body: JSONValue, backend: ChatBackend, started: UInt64) throws -> Prepared {
        guard let rawMessages = body["messages"]?.arrayValue, !rawMessages.isEmpty else {
            throw APIError(status: .badRequest, type: "invalid_request_error", code: nil, message: "'messages' must be a non-empty array")
        }
        let messages = normalise(rawMessages)
        var tools = body["tools"]?.arrayValue ?? []
        if body["tool_choice"]?.stringValue == "none" { tools = [] }

        var options = ChatTemplate.Options()
        if let flag = body["enable_thinking"]?.boolValue ?? body["chat_template_kwargs"]?["enable_thinking"]?.boolValue {
            options.enableThinking = flag
        }
        if let effort = body["reasoning_effort"]?.stringValue ?? body["chat_template_kwargs"]?["reasoning_effort"]?.stringValue {
            switch effort {
            case "none", "minimal", "off": options.enableThinking = false
            case "low": options.reasoningEffort = "low"
            case "medium": options.reasoningEffort = "medium"
            default: options.reasoningEffort = "xhigh"
            }
        }
        if let flag = body["chat_template_kwargs"]?["preserve_thinking"]?.boolValue { options.preserveThinking = flag }

        let rendered: ChatTemplate.Rendered
        do { rendered = try ChatTemplate.render(messages: messages, tools: tools, options: options) } catch {
            throw APIError(status: .badRequest, type: "invalid_request_error", code: nil, message: "\(error)")
        }
        // Replies the model spelt its own way go back in as it spelt them: that is what the
        // server holds, and the prompt then runs on from it.
        let prompt = backend.aliases.apply(to: backend.tokenizer.encode(rendered.prompt)) { backend.tokenizer.bytes(for: $0) ?? [] }
        guard prompt.count < backend.maxContext else {
            throw APIError(status: .badRequest, type: "invalid_request_error", code: "context_length_exceeded",
                           message: "This model's maximum context length is \(backend.maxContext) tokens. Your messages resulted in \(prompt.count) tokens.")
        }
        let room = backend.maxContext - prompt.count
        let requested = body["max_completion_tokens"]?.intValue ?? body["max_tokens"]?.intValue
        let maxTokens = max(1, min(requested ?? room, room))

        var sampling = SamplingParameters()
        if let value = body["temperature"]?.doubleValue { sampling.temperature = Float(value) }
        if let value = body["top_p"]?.doubleValue { sampling.topP = Float(value) }
        if let value = body["top_k"]?.intValue { sampling.topK = value }
        if let value = body["seed"]?.intValue { sampling.seed = UInt64(bitPattern: Int64(value)) }

        var stops: [String] = []
        if let single = body["stop"]?.stringValue { stops = [single] }
        else if let many = body["stop"]?.arrayValue { stops = many.compactMap(\.stringValue) }

        let model = body["model"]?.stringValue ?? backend.modelID
        var request = GenerationRequest(promptTokens: prompt, maxTokens: maxTokens, sampling: sampling,
                                        stopTokenIDs: SpecialTokens.eosIDs, vocabLimit: backend.tokenizer.idLimit, label: model)
        // The system block (a harness's instructions and tool definitions) is what separate
        // conversations have in common: ask for the state to be kept where it ends.
        request.activity = SessionActivity()
        if rendered.prompt.hasPrefix("<|im_start|>system"), prompt.first == SpecialTokens.imStartID,
           let end = prompt.dropFirst().firstIndex(of: SpecialTokens.imStartID) {
            request.checkpointHints = [end]
        }
        if let opened = prompt.lastIndex(of: SpecialTokens.imStartID) { request.stablePromptTokens = opened + 1 }
        if rendered.startsInThinking { request.thinkingClose = thinkingClose(backend.tokenizer) }
        request.toolCall = toolCallMarks(backend.tokenizer)
        request.prepareSeconds = Double(DispatchTime.now().uptimeNanoseconds - started) / 1e9
        // Only for the client's own limit: a reply that ran out of context cannot go on.
        let continuation = backend.options.continueOnLimit && (requested ?? room) < room ? LimitContinuation.find(in: tools) : nil
        return Prepared(request: request, model: model, stream: body["stream"]?.boolValue ?? false,
                        includeUsage: body["stream_options"]?["include_usage"]?.boolValue ?? false,
                        startsInThinking: rendered.startsInThinking, tools: tools, stopStrings: stops,
                        limitContinuation: continuation)
    }

    /// How the scheduler is to close a thinking block it ends itself: as the template closes
    /// one, so the reply comes back in the next prompt as the tokens it was generated as.
    static func thinkingClose(_ tokenizer: Tokenizer) -> ThinkingClose? {
        let close = tokenizer.encode(SpecialTokens.thinkingClose)
        guard close.count == 1 else { return nil }
        return ThinkingClose(token: close[0], tokens: tokenizer.encode("\n" + SpecialTokens.thinkingClose + "\n\n"))
    }

    /// The tokens a tool call opens and closes with, for the scheduler to let one that is open
    /// at the reply's limit be finished (ReplyLimit).
    static func toolCallMarks(_ tokenizer: Tokenizer) -> ToolCallMarks? {
        let open = tokenizer.encode(SpecialTokens.toolCallOpen), close = tokenizer.encode(SpecialTokens.toolCallClose)
        guard open.count == 1, close.count == 1 else { return nil }
        return ToolCallMarks(open: open[0], close: close[0])
    }

    /// Accept what real clients send: `developer` is a system role, and several leading system
    /// messages are one system turn. The template itself allows a single leading system message.
    private static func normalise(_ messages: [JSONValue]) -> [JSONValue] {
        var leading: [String] = []
        var rest: [JSONValue] = []
        var inPrefix = true
        for message in messages {
            let role = message["role"]?.stringValue ?? ""
            let isSystem = role == "system" || role == "developer"
            if inPrefix && isSystem {
                leading.append(plainText(message["content"]))
                continue
            }
            inPrefix = false
            if isSystem {
                // A system message mid-conversation has no slot in the template; carry it as a user turn.
                rest.append(.object([("role", .string("user")), ("content", .string(plainText(message["content"])))]))
            } else {
                rest.append(message)
            }
        }
        guard !leading.isEmpty else { return rest }
        return [.object([("role", .string("system")), ("content", .string(leading.joined(separator: "\n\n")))])] + rest
    }

    private static func plainText(_ content: JSONValue?) -> String {
        guard let content else { return "" }
        if let text = content.stringValue { return text }
        return (content.arrayValue ?? []).compactMap { $0["text"]?.stringValue }.joined()
    }

    // MARK: Response

    static func usage(_ usage: GenerationUsage) -> JSONValue {
        let prefilled = usage.promptTokens - usage.cachedTokens
        return .object([
            ("prompt_tokens", .int(usage.promptTokens)),
            ("completion_tokens", .int(usage.completionTokens)),
            ("total_tokens", .int(usage.promptTokens + usage.completionTokens)),
            ("prompt_tokens_details", .object([("cached_tokens", .int(usage.cachedTokens))])),
            ("timings", .object([
                ("queue_ms", .double((usage.queueSeconds * 1000).rounded())),
                ("prefill_ms", .double((usage.prefillSeconds * 1000).rounded())),
                ("decode_ms", .double((usage.decodeSeconds * 1000).rounded())),
                ("prefill_tokens_per_second", .double(usage.prefillSeconds > 0 ? (Double(prefilled) / usage.prefillSeconds * 10).rounded() / 10 : 0)),
                ("decode_tokens_per_second", .double(usage.decodeSeconds > 0 ? (Double(usage.completionTokens) / usage.decodeSeconds * 10).rounded() / 10 : 0)),
                // Up to the reply: the disk store's copy at the end is not in it.
                ("outside_steps_ms", .double((usage.overhead.total * 1000).rounded())),
                ("first_step_ms", .double((usage.overhead.firstStep * 1000).rounded())),
            ])),
        ])
    }

    /// A request taken as far as the scheduler: its reply's events, or the answer to one that
    /// got no further.
    enum Admitted {
        case reply(Prepared, AsyncStream<GenerationEvent>)
        case refused(Response)
    }

    /// Read a request and give it to the scheduler. With a `catalog`, one that names a
    /// registered model this engine does not have is answered before its prompt is rendered:
    /// it is another engine's to prepare (see ModelCatalog).
    static func admit(_ bytes: [UInt8], backend: ChatBackend, catalog: ModelCatalog? = nil) -> Admitted {
        let started = DispatchTime.now().uptimeNanoseconds
        do {
            let body = try parse(bytes)
            if let catalog, let refusal = catalog.refusal(catalog.gate(body["model"]?.stringValue)) { return .refused(refusal) }
            let prepared = try prepare(body, backend: backend, started: started)
            return .reply(prepared, backend.scheduler.submit(prepared.request))
        } catch let error as APIError {
            return .refused(json(error.body, status: error.status))
        } catch {
            return .refused(json(APIError(status: .internalServerError, type: "server_error", code: nil, message: "\(error)").body, status: .internalServerError))
        }
    }

    static func respond(to admitted: Admitted, backend: ChatBackend) async -> Response {
        switch admitted {
        case .refused(let response):
            return response
        case .reply(let prepared, let events):
            let id = "chatcmpl-" + UUID().uuidString.replacingOccurrences(of: "-", with: "").lowercased()
            let created = Int(Date().timeIntervalSince1970)
            return prepared.stream
                ? streaming(events, prepared: prepared, tokenizer: backend.tokenizer, id: id, created: created, aliases: backend.aliases)
                : await complete(events, prepared: prepared, tokenizer: backend.tokenizer, id: id, created: created, aliases: backend.aliases)
        }
    }

    static func handle(_ bytes: [UInt8], backend: ChatBackend, catalog: ModelCatalog? = nil) async -> Response {
        await respond(to: admit(bytes, backend: backend, catalog: catalog), backend: backend)
    }

    /// A reply the server cut short because it is stopping. The client must not take what it
    /// has for a whole reply (an agent would act on half an answer and call its work done), so
    /// this is always a failure: one a client that retries can send again.
    static func interrupted() -> APIError {
        APIError(status: .serviceUnavailable, type: "server_error", code: "server_stopping",
                 message: "the server stopped before this reply was finished; send the request again")
    }

    private static func finishReason(_ reason: FinishReason, parser: ChatOutputParser) -> String {
        // A call that was part sent and never finished must not be made. "length" is what a
        // client is told of a reply cut short, and what it drops such a call on, however the
        // reply came to end there.
        if parser.unfinishedToolCall { return "length" }
        if parser.toolCallCount > 0 { return "tool_calls" }
        return reason == .length ? "length" : "stop"
    }

    /// The call a reply is to end with instead of its limit: one that was cut at the limit
    /// outside any tool call (a whole call already goes on, and half of one cannot be fixed
    /// once its start is sent).
    private static func continuation(_ reason: FinishReason, parser: ChatOutputParser, prepared: Prepared) -> LimitContinuation? {
        guard reason == .length, !parser.stopped, !parser.unfinishedToolCall, parser.toolCallCount == 0 else { return nil }
        return prepared.limitContinuation
    }

    /// What a reply holds that a client keeps: reasoning, and an answer (content or a whole
    /// tool call). Whitespace is neither.
    struct Substance: Sendable {
        var reasoning = false, content = false, call = false
        var answer: Bool { content || call }

        mutating func note(_ events: [ChatOutputParser.Event]) {
            for event in events {
                switch event {
                case .reasoning(let text): if text.contains(where: { !$0.isWhitespace }) { reasoning = true }
                case .content(let text): if text.contains(where: { !$0.isWhitespace }) { content = true }
                case .toolCallEnd: call = true
                case .toolCallStart, .toolCallArguments: break
                }
            }
        }
    }

    /// The content of a reply that is thinking and nothing else: one that reached its token
    /// limit while still thinking, and one the model ended without an answer.
    ///
    /// A harness leaves an assistant message with no content and no tool call out of the next
    /// request, the reasoning with it (the one this was found under did so 7 times in 7). Asked
    /// to go on, the model has none of its thinking and does it again. With content the
    /// message comes back whole, `reasoning_content` and all, and the next prompt runs on from
    /// what the server holds. The model did not write it, so it is not in the usage.
    static let cutWhileThinking = "[Output token limit reached while still thinking: no answer was written. The thinking so far is kept, so the next reply can carry on from it instead of starting again.]"
    static let endedAfterThinking = "[The reply ended after its thinking: no answer was written. The thinking is kept, so the next reply can answer from it instead of working it out again.]"
    /// For a reply that ended inside a tool call (see `finishReason`): the client drops the
    /// reply's calls, and what is left of it must not be nothing.
    static let endedInsideToolCall = "[The reply ended inside a tool call, before the call was whole, so the call was not made. The next reply should make it again, in smaller calls if it was a long one.]"

    /// `stopString` is a reply ended by one of the request's own stop strings: what it asked
    /// for. `unfinishedCall` is one that ended inside a tool call.
    static func standIn(for reason: FinishReason, _ substance: Substance, stopString: Bool = false, unfinishedCall: Bool = false) -> String? {
        guard reason == .length || reason == .stop, !stopString else { return nil }
        if unfinishedCall { return substance.content ? nil : endedInsideToolCall }
        guard substance.reasoning, !substance.answer else { return nil }
        switch reason {
        case .length: return cutWhileThinking
        case .stop: return endedAfterThinking
        case .cancelled, .error: return nil
        }
    }

    /// `GET /v1/sessions/:id/reply`: what a session has written so far, for the dashboard's
    /// reply window. `activity` is nil for a session that is not running and did not end lately.
    static func reply(of session: Int?, _ activity: SessionActivity?) -> Response {
        guard let session, let activity else {
            let error = APIError(status: .notFound, type: "invalid_request_error", code: "session_not_found",
                                 message: "no session with that id is running or ended lately")
            return json(error.body, status: error.status)
        }
        let parts = activity.reply.map { part -> JSONValue in
            var members: [(key: String, value: JSONValue)] = [("kind", .string(part.kind.rawValue))]
            if let name = part.name { members.append(("name", .string(name))) }
            members.append(("text", .string(part.text)))
            return .object(members)
        }
        var response = json(.object([("id", .int(session)), ("parts", .array(parts))]))
        response.headers[.cacheControl] = "no-store"
        return response
    }

    /// What a reply was, token for token, kept for the prompt that carries it back.
    private struct Generated {
        var tokens: [Int] = []
        var bytes: [UInt8] = []
        mutating func add(_ token: Int, _ piece: [UInt8]) { tokens.append(token); bytes.append(contentsOf: piece) }
    }

    static func streaming(_ events: AsyncStream<GenerationEvent>, prepared: Prepared, tokenizer: Tokenizer,
                          id: String, created: Int, aliases: ReplyAliases? = nil) -> Response {
        let body = ResponseBody { writer in
            var writer = writer
            var parser = ChatOutputParser(startsInThinking: prepared.startsInThinking, tools: prepared.tools, stopStrings: prepared.stopStrings)
            func chunk(delta: JSONValue, finish: String? = nil) -> ByteBuffer {
                let value = JSONValue.object([
                    ("id", .string(id)), ("object", .string("chat.completion.chunk")), ("created", .int(created)),
                    ("model", .string(prepared.model)),
                    ("choices", .array([.object([("index", .int(0)), ("delta", delta),
                                                 ("finish_reason", finish.map(JSONValue.string) ?? .null)])])),
                ])
                return ByteBuffer(string: "data: " + value.compact + "\n\n")
            }
            func deltas(_ parsed: [ChatOutputParser.Event]) -> [ByteBuffer] {
                parsed.compactMap { event in
                    switch event {
                    case .reasoning(let text): return chunk(delta: .object([("reasoning_content", .string(text))]))
                    case .content(let text): return chunk(delta: .object([("content", .string(text))]))
                    case .toolCallStart(let index, let callID, let name):
                        return chunk(delta: .object([("tool_calls", .array([.object([
                            ("index", .int(index)), ("id", .string(callID)), ("type", .string("function")),
                            ("function", .object([("name", .string(name)), ("arguments", .string(""))])),
                        ])]))]))
                    case .toolCallArguments(let index, let piece):
                        return chunk(delta: .object([("tool_calls", .array([.object([
                            ("index", .int(index)), ("function", .object([("arguments", .string(piece))])),
                        ])]))]))
                    case .toolCallEnd: return nil
                    }
                }
            }
            try await writer.write(chunk(delta: .object([("role", .string("assistant")), ("content", .string(""))])))
            // Until the session says how it ended, it has not: events that just stop coming
            // (this task cancelled, the scheduler gone) are a reply cut short, not a finished one.
            var final: (FinishReason, GenerationUsage, String?) = (.cancelled, GenerationUsage(), nil)
            var generated = Generated()
            var substance = Substance()
            loop: for await event in events {
                switch event {
                case .token(let token):
                    let piece = tokenizer.bytes(for: token) ?? []
                    generated.add(token, piece)
                    let parsed = parser.push(piece)
                    substance.note(parsed)
                    prepared.request.activity?.record(parser.phase)
                    prepared.request.activity?.note(parsed)
                    for buffer in deltas(parsed) { try await writer.write(buffer) }
                    if parser.stopped { final.0 = .stop; break loop }
                case .progress(let evaluated, let total):
                    // A comment line, which event-stream clients skip. It keeps the connection
                    // from looking dead while a long prompt is evaluated, and the write fails
                    // if the client has gone, which ends the session instead of orphaning it.
                    try await writer.write(ByteBuffer(string: ": evaluating prompt \(evaluated)/\(total)\n\n"))
                case .finished(let reason, let usage, let message):
                    final = (reason, usage, message)
                }
            }
            if final.0 == .cancelled {
                // The connection ends without its closing frames, and nothing is said first: a
                // broken connection is what clients take for a passing fault and send again,
                // where an error frame is taken for the server's answer (the harness this was
                // tried under retries the one and gives up on the other).
                throw interrupted()
            }
            if final.0 == .error, let message = final.2,
               !message.hasPrefix("context_length_exceeded"), message != "empty prompt" {
                // A fault of the engine or the store, which the same request may not meet again:
                // also a broken connection. Only what the request itself got wrong is answered.
                throw interrupted()
            }
            if !parser.stopped, final.0 != .error {
                aliases?.remember(prompt: prepared.request.promptTokens, generated: generated.tokens, bytes: generated.bytes)
            }
            let last = parser.finish()
            substance.note(last)
            prepared.request.activity?.note(last)
            for buffer in deltas(last) { try await writer.write(buffer) }
            if final.0 == .error {
                let error = APIError(status: .internalServerError, type: "server_error", code: nil, message: final.2 ?? "generation failed")
                try await writer.write(ByteBuffer(string: "data: " + error.body.compact + "\n\n"))
            } else {
                if let call = continuation(final.0, parser: parser, prepared: prepared) {
                    try await writer.write(chunk(delta: .object([("tool_calls", .array([.object([
                        ("index", .int(0)), ("id", .string("call_" + UUID().uuidString.replacingOccurrences(of: "-", with: "").lowercased())),
                        ("type", .string("function")),
                        ("function", .object([("name", .string(call.name)), ("arguments", .string(call.arguments))])),
                    ])]))])))
                    try await writer.write(chunk(delta: .object([]), finish: "tool_calls"))
                } else {
                    if let text = standIn(for: final.0, substance, stopString: parser.stopped, unfinishedCall: parser.unfinishedToolCall) {
                        try await writer.write(chunk(delta: .object([("content", .string(text))])))
                    }
                    try await writer.write(chunk(delta: .object([]), finish: finishReason(final.0, parser: parser)))
                }
                if prepared.includeUsage {
                    let value = JSONValue.object([
                        ("id", .string(id)), ("object", .string("chat.completion.chunk")), ("created", .int(created)),
                        ("model", .string(prepared.model)), ("choices", .array([])), ("usage", usage(final.1)),
                    ])
                    try await writer.write(ByteBuffer(string: "data: " + value.compact + "\n\n"))
                }
            }
            try await writer.write(ByteBuffer(string: "data: [DONE]\n\n"))
            try await writer.finish(nil)
        }
        return Response(status: .ok, headers: [.contentType: "text/event-stream", .cacheControl: "no-cache"], body: body)
    }

    /// A whole response in one JSON body.
    ///
    /// A short request is answered as usual, status and all. One whose prompt takes long enough
    /// to report progress is answered as a body written in pieces: whitespace while the prompt
    /// is evaluated (valid before a JSON value, and enough to keep the connection alive and to
    /// find out that the client has gone, which ends the session), then the JSON. By then the
    /// status has been sent, so a failure after that point arrives as an error body under 200.
    static func complete(_ events: AsyncStream<GenerationEvent>, prepared: Prepared, tokenizer: Tokenizer,
                         id: String, created: Int, aliases: ReplyAliases? = nil) async -> Response {
        struct Collected {
            var parser: ChatOutputParser
            var content = "", reasoning = ""
            var calls: [(id: String, name: String, arguments: String)] = []
            var final: (FinishReason, GenerationUsage, String?) = (.cancelled, GenerationUsage(), nil)
            var generated = Generated()
            var substance = Substance()

            mutating func absorb(_ parsed: [ChatOutputParser.Event]) {
                substance.note(parsed)
                for event in parsed {
                    switch event {
                    case .reasoning(let text): reasoning += text
                    case .content(let text): content += text
                    case .toolCallStart(_, let callID, let name): calls.append((callID, name, ""))
                    case .toolCallArguments(let index, let piece): if calls.indices.contains(index) { calls[index].arguments += piece }
                    case .toolCallEnd: break
                    }
                }
            }
        }
        @Sendable func fresh() -> Collected {
            Collected(parser: ChatOutputParser(startsInThinking: prepared.startsInThinking, tools: prepared.tools, stopStrings: prepared.stopStrings))
        }
        @Sendable func failure(_ collected: Collected) -> APIError? {
            if collected.final.0 == .cancelled { return interrupted() }
            guard collected.final.0 == .error else { return nil }
            let message = collected.final.2 ?? "generation failed"
            let contextual = message.hasPrefix("context_length_exceeded")
            return APIError(status: contextual ? .badRequest : .internalServerError,
                            type: contextual ? "invalid_request_error" : "server_error",
                            code: contextual ? "context_length_exceeded" : nil, message: message)
        }
        @Sendable func success(_ collected: Collected) -> JSONValue {
            let added = continuation(collected.final.0, parser: collected.parser, prepared: prepared)
            let content = collected.content + (added != nil ? "" : standIn(for: collected.final.0, collected.substance, stopString: collected.parser.stopped,
                                                                           unfinishedCall: collected.parser.unfinishedToolCall) ?? "")
            // The whole calls: one the reply ended inside is left out.
            var calls = collected.calls.prefix(collected.parser.toolCallCount).map { call in
                JSONValue.object([("id", .string(call.id)), ("type", .string("function")),
                                  ("function", .object([("name", .string(call.name)), ("arguments", .string(call.arguments))]))])
            }
            if let added {
                calls.append(.object([("id", .string("call_" + UUID().uuidString.replacingOccurrences(of: "-", with: "").lowercased())),
                                      ("type", .string("function")),
                                      ("function", .object([("name", .string(added.name)), ("arguments", .string(added.arguments))]))]))
            }
            var message: [(key: String, value: JSONValue)] = [("role", .string("assistant")),
                ("content", content.isEmpty && !calls.isEmpty ? .null : .string(content))]
            if !collected.reasoning.isEmpty { message.append(("reasoning_content", .string(collected.reasoning))) }
            if !calls.isEmpty { message.append(("tool_calls", .array(calls))) }
            return .object([
                ("id", .string(id)), ("object", .string("chat.completion")), ("created", .int(created)),
                ("model", .string(prepared.model)),
                ("choices", .array([.object([("index", .int(0)), ("message", .object(message)),
                                             ("finish_reason", .string(added != nil ? "tool_calls" : finishReason(collected.final.0, parser: collected.parser)))])])),
                ("usage", usage(collected.final.1)),
            ])
        }
        /// The reply's end: what the parser was still holding.
        @Sendable func finish(_ collected: inout Collected) {
            let last = collected.parser.finish()
            collected.absorb(last)
            prepared.request.activity?.note(last)
        }
        /// Take events until the session ends, or (with `untilProgress`) until it first reports
        /// progress. Returns false in the latter case.
        @Sendable func collect(into collected: inout Collected, untilProgress: Bool, onProgress: () async throws -> Void) async rethrows -> Bool {
            for await event in events {
                switch event {
                case .token(let token):
                    let piece = tokenizer.bytes(for: token) ?? []
                    collected.generated.add(token, piece)
                    let parsed = collected.parser.push(piece)
                    collected.absorb(parsed)
                    prepared.request.activity?.record(collected.parser.phase)
                    prepared.request.activity?.note(parsed)
                    if collected.parser.stopped { collected.final.0 = .stop; finish(&collected); return true }
                case .progress:
                    if untilProgress { return false }
                    try await onProgress()
                case .finished(let reason, let usage, let message):
                    collected.final = (reason, usage, message)
                }
            }
            if collected.final.0 == .stop || collected.final.0 == .length {
                aliases?.remember(prompt: prepared.request.promptTokens, generated: collected.generated.tokens, bytes: collected.generated.bytes)
            }
            finish(&collected)
            return true
        }

        var quick = fresh()
        if await collect(into: &quick, untilProgress: true, onProgress: {}) {
            if let error = failure(quick) { return json(error.body, status: error.status) }
            return json(success(quick))
        }
        let body = ResponseBody { writer in
            var writer = writer
            try await writer.write(ByteBuffer(string: " \n"))
            var collected = fresh()
            _ = try await collect(into: &collected, untilProgress: false) { try await writer.write(ByteBuffer(string: " \n")) }
            // Cut short with the status already sent: the body is left unfinished, which a
            // client reads as a broken connection and may retry, where an error body under
            // 200 might be read as a reply.
            if collected.final.0 == .cancelled { throw interrupted() }
            let value = failure(collected)?.body ?? success(collected)
            try await writer.write(ByteBuffer(string: value.compact))
            try await writer.finish(nil)
        }
        return Response(status: .ok, headers: [.contentType: "application/json"], body: body)
    }
}
