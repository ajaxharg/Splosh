import Foundation
import Hummingbird

public struct ChatRequest: Codable, Sendable { public let model: String; public let messages: [WireChatMessage]; public let stream: Bool?; public init(model: String, messages: [WireChatMessage], stream: Bool? = nil) { self.model=model; self.messages=messages; self.stream=stream } }
public struct WireChatMessage: Codable, Sendable { public let role: String; public let content: String; public init(role: String, content: String) { self.role=role; self.content=content } }
public struct ContextWindowExceededResponse: Codable, Sendable, ResponseEncodable { public let error: String; public init(error: String = "CONTEXT_WINDOW_EXCEEDED") { self.error = error } }
public struct SSEFrame: Encodable, Sendable {
    public let id: String; public let model: String; public let choices: [Choice]
    public init(id: String, model: String, choices: [Choice]) { self.id=id; self.model=model; self.choices=choices }
    public struct Choice: Encodable, Sendable { public let delta: Delta; public let finish_reason: String?; public init(delta: Delta, finish_reason: String? = nil) { self.delta=delta; self.finish_reason=finish_reason } }
    public struct Delta: Encodable, Sendable {
        public let content: String?; public let reasoning_content: String?; public let tool_calls: [ToolCall]?
        public init(content: String? = nil, reasoning_content: String? = nil, tool_calls: [ToolCall]? = nil) { self.content=content; self.reasoning_content=reasoning_content; self.tool_calls=tool_calls }
    }
    public struct ToolCall: Encodable, Sendable { public let index: Int; public let id: String; public let type: String?; public let function: Function?
        public init(index: Int, id: String, type: String? = nil, function: Function? = nil) { self.index=index; self.id=id; self.type=type; self.function=function }
        public struct Function: Encodable, Sendable { public let name: String?; public let arguments: String?; public init(name: String? = nil, arguments: String? = nil) { self.name=name; self.arguments=arguments } }
    }
}
