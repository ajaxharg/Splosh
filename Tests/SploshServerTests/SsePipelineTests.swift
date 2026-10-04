import Foundation
import Testing
import SploshRuntime
import SploshServer

@Suite("SsePipelineTests")
struct SsePipelineTests {
    private func stream(_ events: [InferenceEvent]) -> AsyncThrowingStream<InferenceEvent, Error> {
        AsyncThrowingStream { continuation in
            for event in events { continuation.yield(event) }
            continuation.finish()
        }
    }

    @Test("emits one terminal frame and DONE")
    func terminalFrame() async throws {
        let records = try await SsePipeline.encode(stream([.delta("hi"), .finished(reason: "stop"), .finished(reason: "length")]), model: "m")
        let text = records.reduce(into: "") { $0 += String(decoding: $1, as: UTF8.self) }
        #expect(text.components(separatedBy: "finish_reason").count - 1 == 1)
        #expect(text.hasSuffix("data: [DONE]\n\n"))
    }

    @Test("preserves reasoning and tool ordering")
    func ordering() async throws {
        let records = try await SsePipeline.encode(stream([
            .reasoning("think"), .toolCreated(index: 0, id: "deadbeef", name: "search"),
            .toolArguments(index: 0, id: "deadbeef", fragment: "{\"q\":\"x\"}"), .finished(reason: "stop")
        ]), model: "m")
        let text = records.reduce(into: "") { $0 += String(decoding: $1, as: UTF8.self) }
        #expect(text.range(of: "search")!.lowerBound < text.range(of: "arguments")!.lowerBound)
        #expect(text.contains("reasoning_content"))
        #expect(text.hasSuffix("data: [DONE]\n\n"))
    }

    @Test("streaming parser preserves split UTF8")
    func splitUTF8() {
        var parser = StreamingToolCallParser()
        _ = parser.append("<tool_call><function=echo>")
        let bytes = Array("é".utf8)
        #expect(parser.append(Data(bytes.prefix(1))).isEmpty)
        #expect(parser.append(Data(bytes.suffix(1))).isEmpty)
        let events = parser.append("</parameter></function></tool_call>")
        let arguments = events.compactMap { event -> String? in
            if case .arguments(_, _, let value) = event { return value }
            return nil
        }
        #expect(arguments.contains { $0.hasPrefix("é") })
        #expect(!arguments.contains { $0.contains("�") })
    }
}
