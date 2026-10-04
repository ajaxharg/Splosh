import Foundation

/// A stand-in for the model: it says back the last message. With an `interval` it says it a
/// word at a time, that far apart, which is how the server's plumbing is tested against
/// replies that take a while.
public struct EchoInferenceService: InferenceService {
    public let interval: Duration?
    public init(interval: Duration? = nil) { self.interval = interval }
    public func infer(_ request: InferenceRequest) -> AsyncThrowingStream<InferenceEvent, Error> {
        let text = request.messages.last?.content ?? ""
        guard let interval else {
            return AsyncThrowingStream { continuation in
                continuation.yield(.delta(text))
                continuation.yield(.finished(reason: "stop"))
                continuation.finish()
            }
        }
        return AsyncThrowingStream { continuation in
            let task = Task {
                for word in text.split(separator: " ") {
                    try? await Task.sleep(for: interval)
                    if Task.isCancelled { continuation.finish(throwing: CancellationError()); return }
                    continuation.yield(.delta(word + " "))
                }
                continuation.yield(.finished(reason: "stop"))
                continuation.finish()
            }
            continuation.onTermination = { _ in task.cancel() }
        }
    }
}
