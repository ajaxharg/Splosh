import Foundation

public struct InferenceRequest: Sendable, Codable {
    public let model: String
    public let messages: [InferenceMessage]
    public let stream: Bool
    public init(model: String, messages: [InferenceMessage], stream: Bool = true) {
        self.model = model; self.messages = messages; self.stream = stream
    }
}

public struct InferenceMessage: Sendable, Codable {
    public let role: String
    public let content: String
    public init(role: String, content: String) { self.role = role; self.content = content }
}

public enum InferenceEvent: Sendable {
    case delta(String)
    case reasoning(String)
    case toolCreated(index: Int, id: String, name: String)
    case toolArguments(index: Int, id: String, fragment: String)
    case finished(reason: String)
}

public enum InferenceServiceError: Error, Sendable, Equatable {
    case contextWindowExceeded
}

public extension InferenceService {
    /// Applies the shared history policy before invoking the model stream.
    func admit(_ request: InferenceRequest, contextWindow: Int = 262_144, tokenCount: (InferenceMessage) -> Int = { $0.content.utf8.count }) throws -> InferenceRequest {
        var messages = request.messages
        while messages.reduce(0, { $0 + tokenCount($1) }) > contextWindow {
            guard let index = messages.firstIndex(where: { $0.role != "system" && $0.role != "developer" }) else { throw InferenceServiceError.contextWindowExceeded }
            messages.remove(at: index)
        }
        guard messages.reduce(0, { $0 + tokenCount($1) }) <= contextWindow else { throw InferenceServiceError.contextWindowExceeded }
        return InferenceRequest(model: request.model, messages: messages, stream: request.stream)
    }
}

public protocol InferenceService: Sendable {
    func infer(_ request: InferenceRequest) -> AsyncThrowingStream<InferenceEvent, Error>
}
