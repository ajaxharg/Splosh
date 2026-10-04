import Foundation
import Testing
import Hummingbird
import SploshRuntime
@testable import SploshServer

/// A catalog of three models with `b` loaded: `a` and `b` have artifacts, `gone` has none.
func testCatalog(mode: ModelCatalog.Mode = .request, switching: ModelCatalog.Switching? = nil) -> ModelCatalog {
    ModelCatalog(models: [.init(id: "a", path: "/models/a.splw"), .init(id: "b", path: "/models/b.splw"), .init(id: "gone", path: "/models/gone.splw")],
                 loaded: "b", mode: mode,
                 size: { $0 == "/models/gone.splw" ? nil : $0 == "/models/a.splw" ? 1_000 : 2_000 },
                 switching: { switching })
}

/// What a test reads of a response.
struct ModelReply {
    let status: Int
    let headers: [String: String]
    let body: Data
    var json: JSONValue? { try? JSONValue.parse(Array(body)) }
}

/// The echo server with a catalog, on a port of its own, for as long as `body` runs.
func withModelServer(port: Int, catalog: ModelCatalog, interval: Duration? = nil,
                     _ body: (_ send: @Sendable (String, String, String?) async throws -> ModelReply) async throws -> Void) async throws {
    let app = Server.application(service: EchoInferenceService(interval: interval), port: port, contextWindow: 4096, catalog: catalog)
    let server = Task { try await app.runService(gracefulShutdownSignals: [.sigusr2]) }
    let session = URLSession(configuration: .ephemeral)
    let send: @Sendable (String, String, String?) async throws -> ModelReply = { method, path, body in
        var request = URLRequest(url: URL(string: "http://127.0.0.1:\(port)\(path)")!)
        request.httpMethod = method
        if let body {
            request.httpBody = Data(body.utf8)
            request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        }
        let (data, response) = try await session.data(for: request)
        let http = response as! HTTPURLResponse
        var headers: [String: String] = [:]
        for (name, value) in http.allHeaderFields { headers["\(name)".lowercased()] = "\(value)" }
        return ModelReply(status: http.statusCode, headers: headers, body: data)
    }
    var up = false
    for _ in 0..<100 where !up {
        up = (try? await send("GET", "/health", nil))?.status == 200
        if !up { try await Task.sleep(for: .milliseconds(50)) }
    }
    do {
        #expect(up, "the server did not come up on port \(port)")
        try await body(send)
    } catch {
        server.cancel()
        _ = try? await server.value
        throw error
    }
    server.cancel()
    _ = try? await server.value
}

@Suite("ModelsRouteTests", .serialized)
struct ModelsRouteTests {
    @Test("the list is OpenAI's, with where each model stands here")
    func list() async throws {
        let switching = ModelCatalog.Switching(target: "a", phase: "dwell", parked: 1)
        try await withModelServer(port: 18_291, catalog: testCatalog(switching: switching)) { send in
            let reply = try await send("GET", "/v1/models", nil)
            #expect(reply.status == 200)
            let list = try #require(reply.json)
            #expect(list["object"]?.stringValue == "list")
            #expect(list["loaded"]?.stringValue == "b")
            let data = try #require(list["data"]?.arrayValue)
            // The loaded model first, then the others as registered.
            #expect(data.compactMap { $0["id"]?.stringValue } == ["b", "a", "gone"])
            for model in data {
                #expect(model["object"]?.stringValue == "model")
                #expect(model["owned_by"]?.stringValue == "splosh")
                #expect(model["context_length"]?.intValue == 4096)
                #expect((model["created"]?.intValue ?? 0) > 1_700_000_000)
            }
            #expect(data.compactMap { $0["loaded"]?.boolValue } == [true, false, false])
            #expect(data.compactMap { $0["state"]?.stringValue } == ["loaded", "available", "missing"])
            #expect(data.compactMap { $0["path"]?.stringValue } == ["/models/b.splw", "/models/a.splw", "/models/gone.splw"])
            #expect(data[0]["size_bytes"]?.intValue == 2_000)
            #expect(data[1]["size_bytes"]?.intValue == 1_000)
            #expect(data[2]["size_bytes"]?.isNull == true)
            #expect(list["switch"]?["mode"]?.stringValue == "request")
            #expect(list["switch"]?["target"]?.stringValue == "a")
            #expect(list["switch"]?["phase"]?.stringValue == "dwell")
            #expect(list["switch"]?["parked"]?.intValue == 1)
        }
    }

    @Test("the loaded model is listed first, and the rest as they are registered")
    func loadedFirst() async throws {
        // A client that takes the first of the list names the model in memory, and so causes no switch.
        func ids(_ list: JSONValue?) -> [String] { list?["data"]?.arrayValue?.compactMap { $0["id"]?.stringValue } ?? [] }
        func catalog(loaded: String) -> ModelCatalog {
            var catalog = testCatalog()
            catalog.loaded = loaded
            return catalog
        }
        for (port, loaded, order) in [(18_300, "a", ["a", "b", "gone"]), (18_301, "b", ["b", "a", "gone"]), (18_302, "gone", ["gone", "a", "b"])] {
            try await withModelServer(port: port, catalog: catalog(loaded: loaded)) { send in
                let list = try await send("GET", "/v1/models", nil).json
                #expect(ids(list) == order, "with \(loaded) loaded")
                #expect(list?["loaded"]?.stringValue == loaded)
                #expect(list?["data"]?.arrayValue?.first?["loaded"]?.boolValue == true)
            }
        }
        // The registry keeps its own order.
        #expect(testCatalog().models.map(\.id) == ["a", "b", "gone"])
    }

    @Test("with no switch in hand the list says so; one model is its own object, and an unregistered name is the loaded model's")
    func one() async throws {
        try await withModelServer(port: 18_292, catalog: testCatalog(mode: .manual)) { send in
            let list = try #require(try await send("GET", "/v1/models", nil).json)
            #expect(list["switch"]?["mode"]?.stringValue == "manual")
            #expect(list["switch"]?["target"]?.isNull == true)
            #expect(list["switch"]?["phase"]?.isNull == true)
            #expect(list["switch"]?["parked"]?.intValue == 0)

            let a = try #require(try await send("GET", "/v1/models/a", nil).json)
            #expect(a["id"]?.stringValue == "a")
            #expect(a["object"]?.stringValue == "model")
            #expect(a["loaded"]?.boolValue == false)
            #expect(a["state"]?.stringValue == "available")
            let other = try #require(try await send("GET", "/v1/models/qwen3.8-27b", nil).json)
            #expect(other["id"]?.stringValue == "b")
            #expect(other["state"]?.stringValue == "loaded")
        }
    }

    @Test("the engine says what it has in flight, and which model it has")
    func busy() async throws {
        try await withModelServer(port: 18_293, catalog: testCatalog(), interval: .milliseconds(150)) { send in
            let idle = try #require(try await send("GET", "/v1/busy", nil).json)
            #expect(idle["busy"]?.intValue == 0)
            #expect(idle["model"]?.stringValue == "b")
            async let reply = send("POST", "/v1/chat/completions", #"{"model":"b","messages":[{"role":"user","content":"one two three four five six"}]}"#)
            try await Task.sleep(for: .milliseconds(300))
            #expect(try await send("GET", "/v1/busy", nil).json?["busy"]?.intValue == 1)
            #expect(try await reply.status == 200)
            try await Task.sleep(for: .milliseconds(100))
            #expect(try await send("GET", "/v1/busy", nil).json?["busy"]?.intValue == 0)
        }
    }

    @Test("a load asked for by hand: from this machine only, of a model that is registered and there")
    func load() async throws {
        try await withModelServer(port: 18_294, catalog: testCatalog(mode: .manual)) { send in
            // The one loaded: nothing to do.
            let same = try await send("POST", "/v1/models/load", #"{"model":"b"}"#)
            #expect(same.status == 200)
            #expect(same.json?["loaded"]?.stringValue == "b")
            // Another: the marker, for the holder, whatever the mode; and marked as asked for by hand.
            let other = try await send("POST", "/v1/models/load", #"{"model":"a"}"#)
            #expect(other.status == 421)
            #expect(other.headers["x-splosh-switch"] == "a")
            #expect(other.headers["x-splosh-switch-admin"] == "1")
            let missing = try await send("POST", "/v1/models/load", #"{"model":"gone"}"#)
            #expect(missing.status == 404)
            #expect(missing.json?["error"]?["code"]?.stringValue == "model_not_found")
            let unknown = try await send("POST", "/v1/models/load", #"{"model":"nope"}"#)
            #expect(unknown.status == 404)
            #expect(unknown.json?["error"]?["message"]?.stringValue == "no model called nope is registered")
            #expect(try await send("POST", "/v1/models/load", #"{"id":"a"}"#).status == 400)
        }
        // With no process holding the port there is nothing to start another engine.
        try await withModelServer(port: 18_294, catalog: testCatalog(mode: .none)) { send in
            let refused = try await send("POST", "/v1/models/load", #"{"model":"a"}"#)
            #expect(refused.status == 409)
            #expect(refused.json?["error"]?["code"]?.stringValue == "model_not_loaded")
        }
    }

    @Test("a load is guarded as a change to the settings is")
    func guarded() async throws {
        try await withModelServer(port: 18_295, catalog: testCatalog()) { _ in
            func post(_ headers: [String: String]) async throws -> Int {
                var request = URLRequest(url: URL(string: "http://127.0.0.1:18295/v1/models/load")!)
                request.httpMethod = "POST"
                request.httpBody = Data(#"{"model":"a"}"#.utf8)
                for (name, value) in headers { request.setValue(value, forHTTPHeaderField: name) }
                return ((try await URLSession.shared.data(for: request)).1 as! HTTPURLResponse).statusCode
            }
            let untyped = try await post(["Content-Type": "text/plain"])
            let elsewhere = try await post(["Content-Type": "application/json", "Origin": "http://elsewhere.example"])
            let own = try await post(["Content-Type": "application/json", "Origin": "http://127.0.0.1:18295"])
            #expect(untyped == 403)
            #expect(elsewhere == 403)
            #expect(own == 421)
        }
    }
}
