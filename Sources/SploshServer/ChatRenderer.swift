import Foundation

/// The OpenAI-shaped messages accepted by the prompt renderer.  Tool arguments are an
/// ordered sequence deliberately (JSON objects have no ordering guarantee in Swift).
public struct RenderMessage: Sendable {
    public enum Role: String, Sendable { case system, user, assistant, tool }
    public let role: Role
    public let content: String?
    public let toolCalls: [ToolCall]
    public let toolCallID: String?

    public init(role: Role, content: String? = nil, toolCalls: [ToolCall] = [], toolCallID: String? = nil) {
        self.role = role; self.content = content; self.toolCalls = toolCalls; self.toolCallID = toolCallID
    }
}

public struct ToolSchema: Equatable, Sendable {
    /// Raw JSON is retained to make schema rendering byte-for-byte transparent.
    public let rawJSON: String
    public init(rawJSON: String) { self.rawJSON = rawJSON }
}

public struct ToolCall: Sendable {
    public let name: String
    public let arguments: [(String, AnySendable)]
    public init(name: String, arguments: [(String, AnySendable)] = []) { self.name = name; self.arguments = arguments }
}

/// A small JSON value used by tool arguments. Strings are emitted verbatim; every other
/// value is encoded with JSONSerialization, matching Jinja's `tojson` filter.
public enum AnySendable: Equatable, Sendable {
    case string(String), number(Double), integer(Int), boolean(Bool), null, array([AnySendable]), object([(String, AnySendable)])
    public static func == (lhs: AnySendable, rhs: AnySendable) -> Bool {
        areEqual(lhs, rhs)
    }

    private static func areEqual(_ lhs: AnySendable, _ rhs: AnySendable) -> Bool {
        switch (lhs, rhs) {
        case (.string(let a), .string(let b)): return a == b
        case (.number(let a), .number(let b)): return a == b
        case (.integer(let a), .integer(let b)): return a == b
        case (.boolean(let a), .boolean(let b)): return a == b
        case (.null, .null): return true
        case (.array(let a), .array(let b)):
            return a.count == b.count && zip(a, b).allSatisfy { pair in
                areEqual(pair.0, pair.1)
            }
        case (.object(let a), .object(let b)):
            return a.count == b.count && zip(a, b).allSatisfy { pair in
                pair.0.0 == pair.1.0 && areEqual(pair.0.1, pair.1.1)
            }
        default: return false
        }
    }
    fileprivate var json: String {
        switch self {
        case .string(let s): return s
        case .number(let n): return Self.encode(n)
        case .integer(let n): return String(n)
        case .boolean(let b): return b ? "true" : "false"
        case .null: return "null"
        case .array(let a): return "[" + a.map { $0.encodedJSON }.joined(separator: ",") + "]"
        case .object(let o): return "{" + o.map { Self.encode($0.0) + ":" + $0.1.encodedJSON }.joined(separator: ",") + "}"
        }
    }
    fileprivate var encodedJSON: String { Self.encodeJSON(self) }
    private static func encodeJSON(_ value: AnySendable) -> String { value.json }
    private static func encode<T: Encodable>(_ value: T) -> String {
        let data = (try? JSONEncoder().encode(value)) ?? Data("null".utf8)
        return String(decoding: data, as: UTF8.self)
    }
}

public enum ChatRenderer {
    public typealias Message = RenderMessage
    public typealias Tool = ToolSchema
    public typealias Call = ToolCall
    public typealias JSONValue = AnySendable
    public enum ReasoningEffort: String, Sendable { case xhigh, medium, low }

    public static func render(messages: [RenderMessage], tools: [ToolSchema] = [], reasoningEffort: ReasoningEffort = .xhigh) -> String {
        var out = ""
        if !tools.isEmpty, let systemIndex = messages.firstIndex(where: { $0.role == .system }) {
            // The template adds the raw tool objects to the system turn. Build that turn
            // here rather than decoding/re-encoding schemas (which would alter bytes).
            let system = messages[systemIndex]
            let prefix = system.content ?? ""
            let effortText: String
            switch reasoningEffort {
            case .xhigh: effortText = "reasoning_effort: xhigh\n"
            case .medium: effortText = ""
            case .low: effortText = "reasoning_effort: low\n"
            }
            let toolBlock = "<tools>\n" + tools.map(\.rawJSON).joined(separator: "\n") + "\n</tools>\n"
            var rewritten = messages
            rewritten[systemIndex] = RenderMessage(role: .system, content: effortText + toolBlock + prefix)
            return render(messages: rewritten, tools: [], reasoningEffort: reasoningEffort)
        }
        var previousWasTool = false
        for message in messages {
            if message.role == .tool {
                if !previousWasTool { out += "<|im_start|>user\n" }
                out += "<tool_response>\n" + (message.content ?? "") + "\n</tool_response>\n"
                previousWasTool = true
                continue
            }
            if previousWasTool { out += "<|im_end|>\n"; previousWasTool = false }
            out += "<|im_start|>\(message.role.rawValue)\n"
            if message.role == .assistant, !message.toolCalls.isEmpty {
                if let content = message.content, !content.isEmpty { out += content + "\n" }
                for call in message.toolCalls { out += render(call) }
            } else { out += message.content ?? "" }
            out += "<|im_end|>\n"
        }
        if previousWasTool { out += "<|im_end|>\n" }
        if let last = messages.last, last.role != .assistant || last.toolCalls.isEmpty {
            out += "<|im_start|>assistant\n"
        }
        return out
    }

    private static func render(_ call: ToolCall) -> String {
        var result = "<tool_call>\n<function=\(call.name)>\n"
        for (key, value) in call.arguments {
            result += "<parameter=\(key)>\n\(value.json)\n</parameter>\n"
        }
        return result + "</function>\n</tool_call>\n"
    }
}
