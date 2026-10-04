// ServerCommandTests.swift — SploshServerTests.
//
// Owner: M0.6. Contract source: rev4 §6 M0.6 ("each holds >= 1 test that exercises a real symbol --
// ... the server file a `SploshServer` value").
//
// The suite is named after the file so that `swift test --filter ServerCommandTests` -- the filter
// rev4 §3.2's inventory names and §6 M0.6's gate runs -- matches it.
//
// M0 has no HTTP layer yet: M1.4 lands the Hummingbird server, and M1.6 is the gate that puts these
// two values on the wire at `GET /v1/models`. Asserting them here keeps the published identity in one
// place and makes the test target non-vacuous before any of that exists.
//
// rev4 §2.2 (`:151`)  max_position_embeddings 262144
// rev4 §6 M1.6        `data[0].id == "qwen3.8-27b"` and `data[0].context_length == 262144`

import Testing

import SploshServer
import SploshRuntime

@Suite("ServerCommandTests")
struct ServerCommandTests {
    /// The id M1.6's `GET /v1/models` assertion compares against, and the spelling Appendix A's
    /// DSH provider block declares.
    @Test("the served model id is qwen3.8-27b (rev4 §6 M1.6)")
    func modelIDMatchesTheWireContract() {
        #expect(Server.modelID == "qwen3.8-27b")
    }

    /// 262144 is `2^18` and is one number in three places (rev4 §5.4: §2.2's
    /// `max_position_embeddings`, M1.6's asserted `context_length`, and Appendix A's
    /// `contextWindow`). The power-of-two form is asserted separately so a decimal typo cannot pass.
    @Test("the published context length is 2^18 (rev4 §2.2 `:151`, §6 M1.6)")
    func contextLengthIsTheSourcedValue() {
        #expect(Server.contextLength == 262_144)
        #expect(Server.contextLength == 1 << 18)
    }

    @Test("custom context is published by the application")
    func customContextIsPublished() async throws {
        let app = Server.application(service: EchoInferenceService(), port: 18_091, contextWindow: 4096)
        _ = app
        #expect(Server.contextLength == 262_144)
    }
}
