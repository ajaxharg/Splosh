import CryptoKit
import Foundation
import HTTPTypes
import Hummingbird
import NIOCore
import SploshRuntime
import Testing
@testable import SploshCLI
@testable import SploshServer

@Suite("ModelLibraryTests")
struct ModelLibraryTests {
    @Test("the built-in library names each model once, with whole hashes and a default that is there")
    func builtIn() {
        let library = ModelLibrary.builtIn
        #expect(Set(library.models.map(\.id)).count == library.models.count)
        #expect(library.model(library.defaultModel) != nil)
        let sources = library.models.map(\.source) + [library.tokenizer, library.draft]
        for source in sources {
            #expect(source.revision.count == 40, "\(source.repo) is pinned to a commit")
            #expect(!source.files.isEmpty)
            for file in source.files {
                #expect(file.bytes > 0)
                #expect(file.sha256.count == 64 && file.sha256.allSatisfy { $0.isHexDigit && !$0.isUppercase }, "\(file.name)")
            }
        }
        for model in library.models {
            #expect(ModelEntry.isValid(id: model.id))
            #expect(model.artifactBytes > 0 && model.conversionBytes == (model.kind == .mlx ? 2 : 1) * model.artifactBytes)
        }
        // The converter hashes the pack's own tokenizer.json: the pack is fetched with one, the same as the tokenizer's.
        let mq4 = library.model("mq4")
        #expect(mq4?.source.files.first { $0.name == "tokenizer.json" } == library.tokenizer.files.first { $0.name == "tokenizer.json" })
    }

    @Test("the splosh.toml in the repository registers every built-in model where the library puts it")
    func shippedRegistry() throws {
        // So a model downloaded from a page of a running server can be loaded with no restart.
        let config = try ServeConfig.load(path: "splosh.toml")
        for model in ModelLibrary.builtIn.models {
            #expect(config.models.contains(ModelEntry(id: model.id, path: model.artifact)), "\(model.id)")
            #expect(config.artifactPath(of: model) == model.artifact)
        }
    }

    @Test("an MLX pack's row-major artifact is beside its tiled one")
    func rowMajor() throws {
        let mq4 = try #require(ModelLibrary.builtIn.model("mq4"))
        #expect(mq4.rowMajorPath(tiled: ServeConfig.tiledWeightsPath) == ServeConfig.rowMajorWeightsPath)
        #expect(mq4.rowMajorPath(tiled: "/models/mine.splw") == "/models/mine.splw.rowmajor")
    }

    @Test("a registered model's artifact is where the file says, not where the library would put it")
    func registeredPath() throws {
        let uq5 = try #require(ModelLibrary.builtIn.model("uq5"))
        #expect(ServeConfig().artifactPath(of: uq5) == uq5.artifact)
        #expect(try ServeConfig.parse("model.uq5 = \"/elsewhere/q5.splw\"").artifactPath(of: uq5) == "/elsewhere/q5.splw")
    }

    @Test("SPLOSH_LIBRARY replaces the library, and a file that is not one is an error")
    func replaced() throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent("splosh-library-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        var library = ModelLibrary.builtIn
        library.defaultModel = "uq4"
        library.models.removeFirst()
        let good = directory.appendingPathComponent("library.json"), bad = directory.appendingPathComponent("bad.json")
        try JSONEncoder().encode(library).write(to: good)
        try Data("{}".utf8).write(to: bad)
        #expect(try ModelLibrary.current(environment: [:]) == .builtIn)
        #expect(try ModelLibrary.current(environment: ["SPLOSH_LIBRARY": good.path]) == library)
        #expect(throws: CLIError.self) { _ = try ModelLibrary.current(environment: ["SPLOSH_LIBRARY": bad.path]) }
        #expect(throws: CLIError.self) { _ = try ModelLibrary.current(environment: ["SPLOSH_LIBRARY": directory.appendingPathComponent("none").path]) }
    }

    @Test("files are fetched from Hugging Face, or the mirror HF_ENDPOINT names; the cache is where its variables say")
    func places() {
        #expect(ModelLibrary.endpoint(environment: [:]) == "https://huggingface.co")
        #expect(ModelLibrary.endpoint(environment: ["HF_ENDPOINT": "http://127.0.0.1:9000/"]) == "http://127.0.0.1:9000")
        #expect(HubCache.directory(environment: ["HF_HUB_CACHE": "/c"]).path == "/c")
        #expect(HubCache.directory(environment: ["HF_HOME": "/h"]).path == "/h/hub")
        #expect(HubCache.directory(environment: [:]).path.hasSuffix("/.cache/huggingface/hub"))
        let source = ModelLibrary.Source(repo: "org/name", revision: "abc", directory: "d", files: [])
        #expect(HubDownload(endpoint: "http://h").url(of: "a file.gguf", in: source)?.absoluteString == "http://h/org/name/resolve/abc/a%20file.gguf")
    }

    @Test("sizes and times are said as a person would")
    func human() {
        #expect(Human.bytes(16_054_541_349) == "16.05 GB")
        #expect(Human.bytes(19_989_325) == "20.0 MB")
        #expect(Human.bytes(4932) == "5 kB")
        #expect(Human.bytes(500) == "500 bytes")
        #expect(Human.duration(42) == "42 s")
        #expect(Human.duration(400) == "6 min 40 s")
        #expect(Human.duration(7260) == "2 h 1 min")
    }

    @Test("download takes a model's name, or none for the default")
    func arguments() throws {
        #expect(try DownloadCommand.parse([]) == DownloadArguments())
        var expected = DownloadArguments()
        expected.model = "uq5"; expected.draft = false; expected.config = "c.toml"
        #expect(try DownloadCommand.parse(["uq5", "--no-draft", "--config", "c.toml"]) == expected)
        #expect(try DownloadCommand.parse(["--list"]).list)
        #expect(throws: CLIError.self) { _ = try DownloadCommand.parse(["a", "b"]) }
        #expect(throws: CLIError.self) { _ = try DownloadCommand.parse(["--fast"]) }
        #expect(throws: CLIError.self) { _ = try DownloadCommand.parse(["a/b"]) }
        #expect(throws: CLIError.self) { _ = try DownloadCommand.parse(["--config"]) }
    }
}

@Suite("InstalledStartTests")
struct InstalledStartTests {
    private static let registry = "model = \"b\"\nmodel.a = \"/a.splw\"\nmodel.b = \"/b.splw\"\nmodel.c = \"/c.splw\"\n"

    @Test("the starting model is the configured one when its artifact is there")
    func configured() throws {
        let config = try ServeConfig.parse(Self.registry)
        let start = try config.installedStart { _ in true }
        #expect(start?.model.id == "b" && start?.note == nil)
    }

    @Test("a starting model with no artifact gives way to the first registered one that has one, and says so")
    func fallsBack() throws {
        let config = try ServeConfig.parse(Self.registry)
        let start = try config.installedStart { $0 == "/c.splw" }
        #expect(start?.model == ModelEntry(id: "c", path: "/c.splw"))
        #expect(start?.note?.contains("b, the model to start on, has no artifact at /b.splw; starting on c") == true)
    }

    @Test("with no artifact at all there is no model to start on")
    func none() throws {
        #expect(try ServeConfig.parse(Self.registry).installedStart { _ in false } == nil)
        // Without a registry the one model is at the default path.
        #expect(try ServeConfig.parse("").installedStart { _ in false } == nil)
        #expect(try ServeConfig.parse("").installedStart { $0 == ServeConfig.defaultWeightsPath }?.model.id == "qwen3.8-27b")
    }

    @Test("a model asked for by name is that one, there or not")
    func named() throws {
        let config = try ServeConfig.parse(Self.registry)
        // `--model`, and the model the process holding the port keeps across a restart.
        #expect(try config.installedStart(cli: "a") { _ in false }?.model.id == "a")
        #expect(try config.installedStart(held: "c") { $0 == "/a.splw" }?.model.id == "c")
        #expect(throws: CLIError.self) { _ = try config.installedStart(cli: "nothing") { _ in true } }
        // A model the holder kept that the file no longer registers is as if none were kept.
        #expect(try config.installedStart(held: "gone") { $0 == "/a.splw" }?.model.id == "a")
        // `weightsPath` with no registry names the one model.
        #expect(try ServeConfig.parse("weightsPath = \"/mine.splw\"").installedStart { _ in false }?.model.path == "/mine.splw")
    }
}

/// A stand-in for Hugging Face on a port of its own, for as long as `body` runs. `whole` has
/// it answer every request with the whole file, as a server that knows nothing of ranges does.
private func withHub(port: Int, files: [String: Data], whole: Bool = false,
                     _ body: (_ ranges: @escaping @Sendable () -> [String?]) async throws -> Void) async throws {
    final class Seen: @unchecked Sendable {
        private let lock = NSLock()
        private var ranges: [String?] = []
        func add(_ range: String?) { lock.lock(); ranges.append(range); lock.unlock() }
        var all: [String?] { lock.lock(); defer { lock.unlock() }; return ranges }
    }
    let seen = Seen()
    let router = Router<BasicRequestContext>()
    router.get("health") { _, _ -> String in "ok" }
    router.get("test/repo/resolve/rev/:name") { request, context -> Response in
        let asked = request.headers[.range]
        seen.add(asked)
        guard let data = context.parameters.get("name").flatMap({ files[$0] }) else { return Response(status: .notFound) }
        guard !whole, let asked, let start = Int(asked.dropFirst("bytes=".count).dropLast()) else {
            return Response(status: .ok, body: ResponseBody(byteBuffer: ByteBuffer(bytes: data)))
        }
        guard start < data.count else { return Response(status: .rangeNotSatisfiable) }
        return Response(status: .partialContent, headers: [.contentRange: "bytes \(start)-\(data.count - 1)/\(data.count)"],
                        body: ResponseBody(byteBuffer: ByteBuffer(bytes: data[start...])))
    }
    let app = Application(responder: router.buildResponder(), configuration: .init(address: .hostname("127.0.0.1", port: port)))
    let server = Task { try await app.runService(gracefulShutdownSignals: [.sigusr2]) }
    var up = false
    for _ in 0..<100 where !up {
        up = ((try? await URLSession.shared.data(from: URL(string: "http://127.0.0.1:\(port)/health")!))?.1 as? HTTPURLResponse)?.statusCode == 200
        if !up { try await Task.sleep(for: .milliseconds(50)) }
    }
    do {
        #expect(up, "the stand-in hub did not come up on port \(port)")
        try await body { seen.all }
    } catch {
        server.cancel()
        _ = try? await server.value
        throw error
    }
    server.cancel()
    _ = try? await server.value
}

@Suite("HubDownloadTests", .serialized)
struct HubDownloadTests {
    private static let source = ModelLibrary.Source(repo: "test/repo", revision: "rev", directory: "", files: [])
    private static let bytes = Data((0..<300_000).map { UInt8(truncatingIfNeeded: $0 &* 31 &+ $0 / 251) })
    private static let file = ModelLibrary.File(name: "weights.bin", bytes: bytes.count,
                                                sha256: SHA256.hash(data: bytes).map { String(format: "%02x", $0) }.joined())

    /// A directory of its own, removed afterwards.
    private static func withDirectory(_ body: (URL) async throws -> Void) async throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent("splosh-hub-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        try await body(directory)
    }

    /// The download, off the test's own thread: it waits on the network.
    private static func fetch(port: Int, to destination: URL, file: ModelLibrary.File = file) async throws -> Int {
        try await Task.detached {
            final class Most: @unchecked Sendable { var bytes = 0 }
            let most = Most()
            var hub = HubDownload(endpoint: "http://127.0.0.1:\(port)")
            hub.patience = 2
            try hub.fetch(file, from: source, to: destination, cancellation: Cancellation(), progress: { most.bytes = max(most.bytes, $0) })
            return most.bytes
        }.value
    }

    @Test("a file is fetched whole, checked, and left with no part beside it")
    func whole() async throws {
        try await Self.withDirectory { directory in
            try await withHub(port: 18_471, files: ["weights.bin": Self.bytes]) { ranges in
                let destination = directory.appendingPathComponent("weights.bin")
                let told = try await Self.fetch(port: 18_471, to: destination)
                #expect(try Data(contentsOf: destination) == Self.bytes)
                #expect(told == Self.bytes.count)
                #expect(!FileManager.default.fileExists(atPath: destination.path + ".part"))
                #expect(ranges() == [nil])
                #expect(try HubDownload.sha256(of: destination) == Self.file.sha256)
            }
        }
    }

    @Test("a part left by a stopped download is carried on from")
    func resumed() async throws {
        try await Self.withDirectory { directory in
            try await withHub(port: 18_472, files: ["weights.bin": Self.bytes]) { ranges in
                let destination = directory.appendingPathComponent("weights.bin")
                try Self.bytes.prefix(100_000).write(to: URL(fileURLWithPath: destination.path + ".part"))
                _ = try await Self.fetch(port: 18_472, to: destination)
                #expect(try Data(contentsOf: destination) == Self.bytes)
                #expect(ranges() == ["bytes=100000-"])
            }
        }
    }

    @Test("a part that is already the whole file is checked and put in place with nothing fetched")
    func complete() async throws {
        try await Self.withDirectory { directory in
            try await withHub(port: 18_473, files: [:]) { ranges in
                let destination = directory.appendingPathComponent("weights.bin")
                try Self.bytes.write(to: URL(fileURLWithPath: destination.path + ".part"))
                _ = try await Self.fetch(port: 18_473, to: destination)
                #expect(try Data(contentsOf: destination) == Self.bytes)
                #expect(ranges().isEmpty)
            }
        }
    }

    @Test("a server that sends the whole file where the rest was asked for has it taken from its start")
    func ignoresRange() async throws {
        try await Self.withDirectory { directory in
            try await withHub(port: 18_474, files: ["weights.bin": Self.bytes], whole: true) { _ in
                let destination = directory.appendingPathComponent("weights.bin")
                try Self.bytes.prefix(100_000).write(to: URL(fileURLWithPath: destination.path + ".part"))
                _ = try await Self.fetch(port: 18_474, to: destination)
                #expect(try Data(contentsOf: destination) == Self.bytes)
            }
        }
    }

    @Test("a file that is not the pinned one is refused, and its part removed")
    func mismatch() async throws {
        try await Self.withDirectory { directory in
            // The part is not the start of the file, so what arrives after it does not make the file.
            try await withHub(port: 18_475, files: ["weights.bin": Self.bytes]) { _ in
                let destination = directory.appendingPathComponent("weights.bin")
                try Data(repeating: 7, count: 100_000).write(to: URL(fileURLWithPath: destination.path + ".part"))
                let error = await #expect(throws: DownloadError.self) { _ = try await Self.fetch(port: 18_475, to: destination) }
                #expect(error?.message.contains("SHA-256") == true && error?.cancelled == false)
                #expect(!FileManager.default.fileExists(atPath: destination.path))
                #expect(!FileManager.default.fileExists(atPath: destination.path + ".part"))
            }
        }
    }

    @Test("a file the hub does not have, and a file longer than the pinned one, are errors")
    func refused() async throws {
        try await Self.withDirectory { directory in
            try await withHub(port: 18_476, files: ["long.bin": Self.bytes]) { _ in
                let missing = await #expect(throws: DownloadError.self) { _ = try await Self.fetch(port: 18_476, to: directory.appendingPathComponent("weights.bin")) }
                #expect(missing?.message.contains("404") == true)
                let short = ModelLibrary.File(name: "long.bin", bytes: 1000, sha256: Self.file.sha256)
                await #expect(throws: DownloadError.self) { _ = try await Self.fetch(port: 18_476, to: directory.appendingPathComponent("long.bin"), file: short) }
                #expect(!FileManager.default.fileExists(atPath: directory.appendingPathComponent("long.bin").path))
            }
        }
    }

    @Test("a download that is asked to stop keeps its part")
    func stopped() async throws {
        try await Self.withDirectory { directory in
            try await withHub(port: 18_477, files: ["weights.bin": Self.bytes]) { _ in
                let destination = directory.appendingPathComponent("weights.bin")
                try Self.bytes.prefix(50_000).write(to: URL(fileURLWithPath: destination.path + ".part"))
                let cancellation = Cancellation()
                cancellation.cancel()
                let error = await #expect(throws: DownloadError.self) {
                    try await Task.detached {
                        try HubDownload(endpoint: "http://127.0.0.1:18477").fetch(Self.file, from: Self.source, to: destination, cancellation: cancellation, progress: { _ in })
                    }.value
                }
                #expect(error?.cancelled == true)
                #expect(ModelCatalog.fileSize(destination.path + ".part") == 50_000)
            }
        }
    }
}

@Suite("DownloadsRouteTests", .serialized)
struct DownloadsRouteTests {
    /// What a stub service was asked to do.
    private final class Asked: @unchecked Sendable {
        private let lock = NSLock()
        private var calls: [String] = []
        func add(_ call: String) { lock.lock(); calls.append(call); lock.unlock() }
        var all: [String] { lock.lock(); defer { lock.unlock() }; return calls }
    }

    private static func service(_ asked: Asked) -> DownloadService {
        let state = JSONValue.object([("setup", .bool(true)), ("models", .array([]))])
        return DownloadService(
            read: { state },
            start: { id in
                asked.add("start \(id)")
                if id == "busy" { throw DownloadRefusal(status: 409, message: "another is being installed") }
                return state
            },
            cancel: { id in asked.add("cancel \(id)"); return state })
    }

    /// A server with no model, on a port of its own, for as long as `body` runs.
    private static func withSetupServer(port: Int, asked: Asked,
                                        _ body: (_ send: @Sendable (String, String, String?, [String: String]) async throws -> ModelReply) async throws -> Void) async throws {
        let app = Server.setupApplication(port: port, downloads: service(asked)) { .object([("data", .array([])), ("setup", .bool(true))]) }
        let server = Task { try await app.runService(gracefulShutdownSignals: [.sigusr2]) }
        let session = URLSession(configuration: .ephemeral)
        let send: @Sendable (String, String, String?, [String: String]) async throws -> ModelReply = { method, path, body, headers in
            var request = URLRequest(url: URL(string: "http://127.0.0.1:\(port)\(path)")!)
            request.httpMethod = method
            if let body {
                request.httpBody = Data(body.utf8)
                request.setValue("application/json", forHTTPHeaderField: "Content-Type")
            }
            for (name, value) in headers { request.setValue(value, forHTTPHeaderField: name) }
            let (data, response) = try await session.data(for: request)
            let http = response as! HTTPURLResponse
            var found: [String: String] = [:]
            for (name, value) in http.allHeaderFields { found["\(name)".lowercased()] = "\(value)" }
            return ModelReply(status: http.statusCode, headers: found, body: data)
        }
        var up = false
        for _ in 0..<100 where !up {
            up = (try? await send("GET", "/health", nil, [:]))?.status == 200
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

    @Test("a server with no model serves the page that offers the downloads, and says so to everything that needs one")
    func setup() async throws {
        try await Self.withSetupServer(port: 18_481, asked: Asked()) { send in
            let page = try await send("GET", "/", nil, [:])
            #expect(page.status == 200 && page.headers["content-type"]?.hasPrefix("text/html") == true)
            let html = String(decoding: page.body, as: UTF8.self)
            #expect(html.contains("Choose a model") && html.contains("/v1/downloads") && html.contains("dlPoll()"))
            // The panel's style and script are written into the page, not left as a placeholder.
            #expect(html.contains("function dlDraw()") && html.contains(".dlbar") && !html.contains("#(DownloadsPanel"))
            let listed = try await send("GET", "/v1/models", nil, [:])
            #expect(listed.status == 200 && listed.json?["setup"]?.boolValue == true)
            for (method, path, body) in [("POST", "/v1/chat/completions", "{\"messages\":[]}"), ("POST", "/v1/models/load", "{\"model\":\"a\"}"),
                                         ("GET", "/v1/stats", nil), ("GET", "/v1/models/a", nil)] as [(String, String, String?)] {
                let reply = try await send(method, path, body, [:])
                #expect(reply.status == 503, "\(method) \(path)")
                #expect(reply.json?["error"]?["code"]?.stringValue == "model_not_installed", "\(method) \(path)")
            }
            #expect(try await send("GET", "/v1/busy", nil, [:]).json?["busy"]?.intValue == 0)
        }
    }

    @Test("an install is started and stopped as the settings are changed: from this machine's own page, as JSON")
    func changes() async throws {
        let asked = Asked()
        try await Self.withSetupServer(port: 18_482, asked: asked) { send in
            let read = try await send("GET", "/v1/downloads", nil, [:])
            #expect(read.status == 200 && read.json?["setup"]?.boolValue == true && read.headers["cache-control"] == "no-store")
            #expect(try await send("POST", "/v1/downloads", "{\"model\":\"mq4\"}", ["Origin": "http://127.0.0.1:18482"]).status == 202)
            #expect(try await send("POST", "/v1/downloads/cancel", "{\"model\":\"mq4\"}", [:]).status == 200)
            let busy = try await send("POST", "/v1/downloads", "{\"model\":\"busy\"}", [:])
            #expect(busy.status == 409 && busy.json?["error"]?.stringValue == "another is being installed")
            #expect(try await send("POST", "/v1/downloads", "{\"other\":1}", [:]).status == 400)
            #expect(try await send("POST", "/v1/downloads", "{\"model\":\"mq4\"}", ["Origin": "http://elsewhere.example"]).status == 403)
            #expect(try await send("POST", "/v1/downloads", "{\"model\":\"mq4\"}", ["Content-Type": "text/plain"]).status == 403)
            #expect(asked.all == ["start mq4", "cancel mq4", "start busy"])
        }
    }

    @Test("the models have a page of their own, with the downloads; the settings page has the rest and a way there")
    func modelsPage() async throws {
        for page in [SettingsPage.html, SettingsPage.models] {
            #expect(page.contains("function dlDraw()") && page.contains(".dlbar") && page.contains("id=\"dl\"") && !page.contains("#("))
        }
        #expect(SettingsPage.models.contains("const MODELS=true") && SettingsPage.models.contains("<h1>Models</h1>") && SettingsPage.models.contains("href=\"/settings\""))
        #expect(SettingsPage.html.contains("const MODELS=false") && SettingsPage.html.contains("<h1>Settings</h1>") && SettingsPage.html.contains("href=\"/models\""))
        // Every setting is on one page or the other: the models' groups on theirs.
        #expect(SettingsPage.html.contains("(x.group==='Models'||x.group==='Model')===MODELS"))
    }

    @Test("a server with a model answers for the downloads too, and serves both pages")
    func withModel() async throws {
        let asked = Asked()
        let settings = SettingsService(read: { .object([("settings", .array([]))]) }, write: { _ in .null }, restart: nil)
        let app = Server.application(service: EchoInferenceService(interval: nil), port: 18_483, contextWindow: 4096, settings: settings,
                                     downloads: Self.service(asked))
        let server = Task { try await app.runService(gracefulShutdownSignals: [.sigusr2]) }
        defer { server.cancel() }
        func get(_ path: String) async -> (status: Int, text: String) {
            guard let (data, response) = try? await URLSession.shared.data(from: URL(string: "http://127.0.0.1:18483\(path)")!) else { return (0, "") }
            return ((response as? HTTPURLResponse)?.statusCode ?? 0, String(decoding: data, as: UTF8.self))
        }
        var status = 0
        for _ in 0..<100 where status != 200 {
            status = await get("/v1/downloads").status
            if status != 200 { try await Task.sleep(for: .milliseconds(50)) }
        }
        #expect(status == 200)
        let models = await get("/models"), page = await get("/settings")
        #expect(models.status == 200 && models.text.contains("<title>Splosh models</title>"))
        #expect(page.status == 200 && page.text.contains("<title>Splosh settings</title>"))
    }
}
