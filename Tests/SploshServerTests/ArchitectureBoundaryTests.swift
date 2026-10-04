// ArchitectureBoundaryTests.swift — M1.4 neutral module-boundary gate.
//
// The static architecture checker owns import/manifest scanning. These tests keep
// the named filter non-vacuous and exercise the actual Runtime→Server seam.

import Testing
import SploshRuntime
import SploshServer

@Suite("ArchitectureBoundaryTests")
struct ArchitectureBoundaryTests {
    @Test("echo service conforms to the neutral runtime protocol")
    func echoServiceUsesNeutralBoundary() async throws {
        let service: any InferenceService = EchoInferenceService()
        let request = InferenceRequest(
            model: Server.modelID,
            messages: [InferenceMessage(role: "user", content: "boundary")]
        )
        var events: [InferenceEvent] = []
        for try await event in service.infer(request) {
            events.append(event)
        }
        #expect(events.count == 2)
        if case .delta(let text) = events[0] {
            #expect(text == "boundary")
        } else {
            Issue.record("first echo event was not a delta")
        }
        if case .finished(let reason) = events[1] {
            #expect(reason == "stop")
        } else {
            Issue.record("second echo event was not finished")
        }
    }

    @Test("server owns the published wire identity")
    func serverPublishesExpectedIdentity() {
        #expect(Server.modelID == "qwen3.8-27b")
        #expect(Server.contextLength == 262_144)
    }
}
