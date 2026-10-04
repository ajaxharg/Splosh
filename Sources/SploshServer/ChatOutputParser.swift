import Foundation
import SploshRuntime

/// Turns the model's raw output bytes into OpenAI-shaped pieces as they arrive: reasoning (inside
/// the `<think>` block the prompt opened), visible content, and tool calls in the model's
/// `<tool_call><function=…><parameter=…>` form.
///
/// A tool call goes out as it is written, not when it is whole: its name first, then its
/// arguments piece by piece. A client that hears nothing for long enough takes the reply for
/// dead and asks again (the harness this runs under allows 300 seconds), and a call that
/// writes a long file can take longer than that to generate.
public struct ChatOutputParser: Sendable {
    public enum Event: Sendable, Equatable {
        case reasoning(String)
        case content(String)
        /// A tool call begins: its place among the reply's calls, its id, the function's name.
        case toolCallStart(index: Int, id: String, name: String)
        /// The next piece of the call's arguments: the pieces of one call, joined, are a JSON
        /// object's text. An empty piece only says the call is still being written.
        case toolCallArguments(index: Int, String)
        /// The call is whole.
        case toolCallEnd(index: Int)
    }

    private enum Mode { case thinking, content, tool }

    /// A tool call whose function has been named: from there it is sent as it is written.
    private struct OpenCall {
        let index: Int
        let name: String
        var members = 0
        /// `</function>` has been seen: the call lacks only its closing tag.
        var functionClosed = false
        var parameter: OpenParameter?
    }

    private struct OpenParameter {
        let key: String
        let type: String?
        /// What has come of a value that has to be read whole.
        var raw = ""
        /// The value's first character has been seen (and a newline there dropped).
        var begun = false
        /// A declared string is sent as it comes. Anything else may be a number, a list or an
        /// object, and what it is cannot be said until all of it is there.
        var streamed: Bool { type == "string" }
    }

    static let thinkClose = "</think>"
    static let toolOpen = "<tool_call>"
    static let toolClose = "</tool_call>"
    private static let parameterOpen = "<parameter="
    private static let parameterClose = "</parameter>"
    private static let functionClose = "</function>"
    /// Tokens of a value read whole (or of anything else that sends nothing) after which an
    /// empty piece is sent, so the client goes on hearing from the call.
    private static let quietTokens = 64

    private var mode: Mode
    private var pending: [UInt8] = []
    private var held = ""
    private var skipLeadingWhitespace = false
    private let stopStrings: [String]
    /// Tool name -> parameter name -> declared JSON-schema type.
    private let parameterTypes: [String: [String: String]]
    private var call: OpenCall?
    private var started = 0
    private var quiet = 0

    /// What the output is at the moment: reasoning, the answer, or a tool call.
    public var phase: SessionActivity.Phase {
        switch mode {
        case .thinking: return .thinking
        case .content: return .answer
        case .tool: return .toolCall
        }
    }
    /// Tool calls that are whole.
    public private(set) var toolCallCount = 0
    /// The reply ended inside a tool call of which part has been sent: a call that must not
    /// be made.
    public private(set) var unfinishedToolCall = false
    public private(set) var stopped = false

    public init(startsInThinking: Bool, tools: [JSONValue] = [], stopStrings: [String] = []) {
        mode = startsInThinking ? .thinking : .content
        self.stopStrings = stopStrings.filter { !$0.isEmpty }
        var types: [String: [String: String]] = [:]
        for tool in tools {
            let function = tool["function"] ?? tool
            guard let name = function["name"]?.stringValue else { continue }
            var parameters: [String: String] = [:]
            for member in function["parameters"]?["properties"]?.objectValue ?? [] {
                if let type = member.value["type"]?.stringValue { parameters[member.key] = type }
            }
            types[name] = parameters
        }
        parameterTypes = types
    }

    public mutating func push(_ bytes: [UInt8]) -> [Event] {
        guard !stopped else { return [] }
        pending += bytes
        // Emit only the longest prefix that is complete UTF-8; a split scalar waits for its tail.
        var count = pending.count
        let floor = max(0, pending.count - 3)
        while count > floor, String(bytes: pending[0..<count], encoding: .utf8) == nil { count -= 1 }
        if count == floor, String(bytes: pending[0..<count], encoding: .utf8) == nil {
            count = pending.count // genuinely invalid: decode lossily rather than stall
        }
        guard count > 0 else { return [] }
        held += String(decoding: pending[0..<count], as: UTF8.self)
        pending.removeFirst(count)
        var events = drain(final: false)
        if let call, events.isEmpty {
            quiet += 1
            if quiet >= Self.quietTokens { events.append(.toolCallArguments(index: call.index, "")); quiet = 0 }
        } else {
            quiet = 0
        }
        return events
    }

    public mutating func finish() -> [Event] {
        guard !stopped else { return [] }
        if !pending.isEmpty {
            held += String(decoding: pending, as: UTF8.self)
            pending.removeAll()
        }
        return drain(final: true)
    }

    private mutating func drain(final: Bool) -> [Event] {
        var events: [Event] = []
        while true {
            switch mode {
            case .thinking:
                if let close = held.range(of: Self.thinkClose) {
                    emit(.reasoning(String(held[..<close.lowerBound])), into: &events)
                    held = String(held[close.upperBound...])
                    mode = .content
                    skipLeadingWhitespace = true
                    continue
                }
                flush(markers: [Self.thinkClose], final: final, into: &events) { .reasoning($0) }
                return events
            case .content:
                if skipLeadingWhitespace {
                    held = String(held.drop { $0.isWhitespace })
                    if held.isEmpty { return events }
                    skipLeadingWhitespace = false
                }
                var first: (range: Range<String.Index>, isTool: Bool)?
                for marker in [Self.toolOpen] + stopStrings {
                    if let range = held.range(of: marker), first == nil || range.lowerBound < first!.range.lowerBound {
                        first = (range, marker == Self.toolOpen)
                    }
                }
                if let first {
                    let before = String(held[..<first.range.lowerBound])
                    if first.isTool {
                        emit(.content(Self.trimTrailingWhitespace(before)), into: &events)
                        held = String(held[first.range.upperBound...])
                        mode = .tool
                        continue
                    }
                    emit(.content(before), into: &events)
                    held = ""
                    stopped = true
                    return events
                }
                flush(markers: [Self.toolOpen] + stopStrings, final: final, into: &events) { .content($0) }
                return events
            case .tool:
                if advanceCall(into: &events) { continue }
                if final { endCallWithReply(into: &events) }
                return events
            }
        }
    }

    /// One step through a tool call's text. False when it has to wait for more.
    private mutating func advanceCall(into events: inout [Event]) -> Bool {
        let close = held.range(of: Self.toolClose)
        // A tag after the call's close is not the call's.
        let limit = close?.lowerBound ?? held.endIndex

        guard var call else {
            // Not named yet, and nothing sent: a call that turns out not to be one is content.
            if let open = held.range(of: "<function=", range: held.startIndex..<limit),
               let nameEnd = held.range(of: ">", range: open.upperBound..<limit) {
                let name = held[open.upperBound..<nameEnd.lowerBound].trimmingCharacters(in: .whitespacesAndNewlines)
                if !name.isEmpty {
                    let id = "call_" + UUID().uuidString.replacingOccurrences(of: "-", with: "").lowercased().prefix(24)
                    events.append(.toolCallStart(index: started, id: id, name: name))
                    self.call = OpenCall(index: started, name: name)
                    started += 1
                    held = String(held[nameEnd.upperBound...])
                    return true
                }
            }
            guard let close else { return false }
            emit(.content(Self.toolOpen + String(held[..<close.upperBound])), into: &events)
            held = String(held[close.upperBound...])
            mode = .content
            skipLeadingWhitespace = true
            return true
        }
        defer { if self.call != nil { self.call = call } }

        if var parameter = call.parameter {
            if let end = held.range(of: Self.parameterClose, range: held.startIndex..<limit) {
                var rest = String(held[..<end.lowerBound])
                if parameter.streamed {
                    if !parameter.begun, rest.hasPrefix("\n") { rest.removeFirst() }
                    if rest.hasSuffix("\n") { rest.removeLast() }
                    events.append(.toolCallArguments(index: call.index, Self.escaped(rest) + "\""))
                } else {
                    // The model writes strings verbatim and everything else as JSON.
                    var raw = parameter.raw + rest
                    if raw.hasPrefix("\n") { raw.removeFirst() }
                    if raw.hasSuffix("\n") { raw.removeLast() }
                    events.append(.toolCallArguments(index: call.index, Self.memberOpening(parameter.key, first: call.members == 0)
                                                     + coerce(raw, type: parameter.type).compact))
                    call.members += 1
                }
                call.parameter = nil
                held = String(held[end.upperBound...])
                return true
            }
            if let close {
                // The call closes inside a value. One read whole is left out, as a value that
                // never ended always was; a string part sent is ended where it stands.
                if parameter.streamed {
                    var rest = String(held[..<close.lowerBound])
                    if !parameter.begun, rest.hasPrefix("\n") { rest.removeFirst() }
                    if rest.hasSuffix("\n") { rest.removeLast() }
                    events.append(.toolCallArguments(index: call.index, Self.escaped(rest) + "\""))
                }
                call.parameter = nil
                held = String(held[close.lowerBound...])
                return true
            }
            // The value goes on. What may be the start of a closing tag waits, and so does a
            // newline at the end: the one before `</parameter>` is not the value's.
            let keep = Self.tagStart(held, of: [Self.parameterClose, Self.toolClose])
            var ready = String(held.dropLast(keep))
            var tail = String(held.suffix(keep))
            if parameter.streamed {
                if !parameter.begun, !ready.isEmpty {
                    if ready.hasPrefix("\n") { ready.removeFirst() }
                    parameter.begun = true
                }
                if ready.hasSuffix("\n") { ready.removeLast(); tail = "\n" + tail }
                if !ready.isEmpty { events.append(.toolCallArguments(index: call.index, Self.escaped(ready))) }
            } else {
                parameter.raw += ready
            }
            held = tail
            call.parameter = parameter
            return false
        }

        // Between parameters.
        if let open = held.range(of: Self.parameterOpen, range: held.startIndex..<limit) {
            if let keyEnd = held.range(of: ">", range: open.upperBound..<limit) {
                let key = held[open.upperBound..<keyEnd.lowerBound].trimmingCharacters(in: .whitespacesAndNewlines)
                let parameter = OpenParameter(key: key, type: parameterTypes[call.name]?[key])
                if parameter.streamed {
                    events.append(.toolCallArguments(index: call.index, Self.memberOpening(key, first: call.members == 0) + "\""))
                    call.members += 1
                }
                call.functionClosed = false
                call.parameter = parameter
                held = String(held[keyEnd.upperBound...])
                return true
            }
            if close == nil {
                held = String(held[open.lowerBound...])       // the name is still being written
                return false
            }
        }
        if let close {
            events.append(.toolCallArguments(index: call.index, call.members == 0 ? "{}" : "}"))
            events.append(.toolCallEnd(index: call.index))
            toolCallCount += 1
            self.call = nil
            held = String(held[close.upperBound...])
            mode = .content
            skipLeadingWhitespace = true
            return true
        }
        if held.contains(Self.functionClose) { call.functionClosed = true }
        held = String(held.suffix(Self.tagStart(held, of: [Self.parameterOpen, Self.functionClose, Self.toolClose])))
        return false
    }

    /// The reply ended inside a tool call.
    private mutating func endCallWithReply(into events: inout [Event]) {
        defer { held = ""; call = nil }
        guard let call else {
            emit(.content(Self.toolOpen + held), into: &events)   // never named: the text it is
            return
        }
        if call.parameter == nil, call.functionClosed || held.contains(Self.functionClose) {
            // The model stopped after the function block, short of the closing tag: it is whole.
            events.append(.toolCallArguments(index: call.index, call.members == 0 ? "{}" : "}"))
            events.append(.toolCallEnd(index: call.index))
            toolCallCount += 1
        } else {
            unfinishedToolCall = true
        }
    }

    private func emit(_ event: Event, into events: inout [Event]) {
        switch event {
        case .reasoning(let text), .content(let text): if !text.isEmpty { events.append(event) }
        case .toolCallStart, .toolCallArguments, .toolCallEnd: events.append(event)
        }
    }

    /// Emit everything that cannot be the start of a marker; keep the rest for the next push.
    private mutating func flush(markers: [String], final: Bool, into events: inout [Event], _ make: (String) -> Event) {
        let keep = final ? 0 : Self.tagStart(held, of: markers)
        let cut = held.index(held.endIndex, offsetBy: -keep)
        emit(make(String(held[..<cut])), into: &events)
        held = String(held[cut...])
    }

    /// How many characters at the end of `text` may be the start of one of `markers`.
    private static func tagStart(_ text: String, of markers: [String]) -> Int {
        var keep = 0
        for marker in markers {
            var length = min(marker.count - 1, text.count)
            while length > keep {
                if marker.hasPrefix(text.suffix(length)) { keep = length; break }
                length -= 1
            }
        }
        return keep
    }

    /// `"key":`, after the `{` or `,` that opens an object's member.
    private static func memberOpening(_ key: String, first: Bool) -> String {
        (first ? "{" : ",") + JSONValue.string(key).compact + ":"
    }

    /// Text as it stands inside a JSON string. Scalars are escaped one by one, so pieces
    /// escaped apart join into the string escaped whole.
    private static func escaped(_ text: String) -> String {
        // The quotes come off as scalars: a piece may begin with one that joins the character
        // before it (an emoji's modifier), and as a character it would go with the quote.
        var quoted = JSONValue.string(text).compact.unicodeScalars
        quoted.removeFirst()
        quoted.removeLast()
        return String(quoted)
    }

    private static func trimTrailingWhitespace(_ text: String) -> String {
        var view = Substring(text)
        while let last = view.last, last.isWhitespace { view.removeLast() }
        return String(view)
    }

    /// The model writes strings verbatim and everything else as JSON. The declared schema type
    /// decides; without one, only unambiguous JSON literals are treated as non-strings.
    private func coerce(_ raw: String, type: String?) -> JSONValue {
        if type == "string" { return .string(raw) }
        if let parsed = try? JSONValue.parse(raw) {
            if type != nil { return parsed }
            switch parsed {
            case .string: return .string(raw)
            default: return parsed
            }
        }
        return .string(raw)
    }
}
