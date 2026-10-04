import Foundation
import Testing
import Hummingbird
import SploshRuntime
@testable import SploshServer

@Suite("ModelGateTests", .serialized)
struct ModelGateTests {
    @Test("the loaded model answers a request that names it, no model, or none that is registered")
    func serves() {
        let catalog = testCatalog()
        #expect(catalog.gate("b") == .serve)
        #expect(catalog.gate(nil) == .serve)
        #expect(catalog.gate("qwen3.8-27b") == .serve)
        #expect(catalog.gate("") == .serve)
        #expect(catalog.refusal(.serve) == nil)
    }

    @Test("a registered model that is not loaded: loaded on request, refused when that is by hand, not found without its artifact")
    func gates() {
        #expect(testCatalog().gate("a") == .load("a"))
        #expect(testCatalog(mode: .manual).gate("a") == .notLoaded("a"))
        #expect(testCatalog(mode: .none).gate("a") == .notLoaded("a"))
        for mode in [ModelCatalog.Mode.request, .manual, .none] {
            #expect(testCatalog(mode: mode).gate("gone") == .missing(.init(id: "gone", path: "/models/gone.splw")))
        }
        // By hand, `manual` loads; a model already loaded is nothing to do; one not registered is not found.
        #expect(testCatalog(mode: .manual).gateLoad("a") == .load("a"))
        #expect(testCatalog(mode: .none).gateLoad("a") == .notLoaded("a"))
        #expect(testCatalog(mode: .manual).gateLoad("b") == .serve)
        #expect(testCatalog().gateLoad("nope") == .missing(.init(id: "nope", path: "")))
    }

    @Test("the marker is the status and header the holder looks for, and is the whole of what it is told")
    func marker() throws {
        let response = try #require(testCatalog().refusal(.load("a")))
        #expect(response.status == .misdirectedRequest)
        #expect(response.status.code == 421)
        #expect(response.headers[ModelCatalog.switchHeader] == "a")
        #expect(response.headers[ModelCatalog.switchAdminHeader] == nil)
        #expect(ModelCatalog.switchHeader.canonicalName == "x-splosh-switch")
        #expect(ModelCatalog.modelHeader.canonicalName == "x-splosh-model")
        let byHand = try #require(testCatalog().refusal(.load("a"), admin: true))
        #expect(byHand.headers[ModelCatalog.switchAdminHeader] == "1")
        #expect(testCatalog().refusal(.missing(.init(id: "gone", path: "/p")))?.status == .notFound)
        #expect(testCatalog(mode: .manual).refusal(.notLoaded("a"))?.status == .conflict)
    }

    @Test("over the wire: a chat response names the model that gave it; one for another model is the marker")
    func chat() async throws {
        func body(_ model: String) -> String { #"{"model":"\#(model)","stream":true,"messages":[{"role":"user","content":"hello there"}]}"# }
        try await withModelServer(port: 18_296, catalog: testCatalog()) { send in
            for name in ["b", "qwen3.8-27b"] {
                let served = try await send("POST", "/v1/chat/completions", body(name))
                #expect(served.status == 200, "\(name)")
                #expect(served.headers["x-splosh-model"] == "b", "\(name)")
                #expect(String(decoding: served.body, as: UTF8.self).contains("hello there"), "\(name)")
                #expect(served.headers["x-splosh-switch"] == nil)
            }
            let wanted = try await send("POST", "/v1/chat/completions", body("a"))
            #expect(wanted.status == 421)
            #expect(wanted.headers["x-splosh-switch"] == "a")
            #expect(wanted.headers["x-splosh-switch-admin"] == nil)
            #expect(wanted.json?["error"]?["code"]?.stringValue == "model_switching")
            let missing = try await send("POST", "/v1/chat/completions", body("gone"))
            #expect(missing.status == 404)
            #expect(missing.json?["error"]?["code"]?.stringValue == "model_not_found")
            #expect(missing.headers["x-splosh-model"] == "b")
            // None of them left anything in flight.
            #expect(try await send("GET", "/v1/busy", nil).json?["busy"]?.intValue == 0)
        }
        try await withModelServer(port: 18_296, catalog: testCatalog(mode: .manual)) { send in
            let refused = try await send("POST", "/v1/chat/completions", body("a"))
            #expect(refused.status == 409)
            #expect(refused.json?["error"]?["code"]?.stringValue == "model_not_loaded")
            #expect(refused.json?["error"]?["message"]?.stringValue?.contains("splosh models --load a") == true)
            #expect(refused.headers["x-splosh-switch"] == nil)
            #expect(try await send("POST", "/v1/chat/completions", body("b")).status == 200)
        }
    }

    @Test("without a catalog the echo server lists its one model as it always has")
    func uncatalogued() async throws {
        let app = Server.application(service: EchoInferenceService(), port: 18_297, contextWindow: 4096)
        let server = Task { try await app.runService(gracefulShutdownSignals: [.sigusr2]) }
        defer { server.cancel() }
        var listed: JSONValue?
        for _ in 0..<100 where listed == nil {
            if let (data, _) = try? await URLSession.shared.data(from: URL(string: "http://127.0.0.1:18297/v1/models")!) {
                listed = try? JSONValue.parse(Array(data))
            } else {
                try await Task.sleep(for: .milliseconds(50))
            }
        }
        #expect(listed?["data"]?.arrayValue?.first?["id"]?.stringValue == Server.modelID)
        #expect(listed?["data"]?.arrayValue?.first?["context_length"]?.intValue == 4096)
        #expect(listed?["loaded"] == nil)
    }
}
