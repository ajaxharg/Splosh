import Foundation
import SploshRuntime

public enum SsePipelineError: Error, Sendable, Equatable {
    case eventAfterCompletion
}

/// Converts runtime inference events into OpenAI-compatible, encoded SSE records.
/// The returned records are ready for transport; no server writer is involved here.
public enum SsePipeline {
    public static func encode(
        _ events: AsyncThrowingStream<InferenceEvent, Error>,
        model: String,
        id: String = "chatcmpl-echo"
    ) async throws -> [Data] {
        var records: [Data] = []
        var completed = false
        let encoder = JSONEncoder()

        for try await event in events {
            if completed {
                // A producer may race completion. Ignore late events so the wire has
                // exactly one terminal frame, while preserving the successful stream.
                continue
            }
            let frame: SSEFrame
            switch event {
            case .delta(let text):
                frame = SSEFrame(id: id, model: model, choices: [.init(delta: .init(content: text))])
            case .reasoning(let text):
                frame = SSEFrame(id: id, model: model, choices: [.init(delta: .init(reasoning_content: text))])
            case .toolCreated(let index, let toolID, let name):
                frame = SSEFrame(id: id, model: model, choices: [.init(delta: .init(tool_calls: [.init(index: index, id: toolID, type: "function", function: .init(name: name))]))])
            case .toolArguments(let index, let toolID, let fragment):
                frame = SSEFrame(id: id, model: model, choices: [.init(delta: .init(tool_calls: [.init(index: index, id: toolID, function: .init(arguments: fragment))]))])
            case .finished(let reason):
                frame = SSEFrame(id: id, model: model, choices: [.init(delta: .init(), finish_reason: reason)])
                completed = true
            }
            let json = try encoder.encode(frame)
            records.append(Data("data: \(String(decoding: json, as: UTF8.self))\n\n".utf8))
        }
        if !completed {
            // A normally completed inference stream still needs a terminal frame.
            let frame = SSEFrame(id: id, model: model, choices: [.init(delta: .init(), finish_reason: "stop")])
            let json = try encoder.encode(frame)
            records.append(Data("data: \(String(decoding: json, as: UTF8.self))\n\n".utf8))
        }
        records.append(Data("data: [DONE]\n\n".utf8))
        return records
    }
}
