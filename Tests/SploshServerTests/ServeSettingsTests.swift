import Foundation
import Testing
@testable import SploshCLI
import SploshServer

@Suite("Settings page and splosh.toml")
struct ServeSettingsTests {
    private static func temporaryFile(_ contents: String) throws -> URL {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("splosh-settings-\(UUID().uuidString).toml")
        try contents.write(to: url, atomically: true, encoding: .utf8)
        return url
    }

    private static func setting(_ key: String, in snapshot: JSONValue) -> JSONValue? {
        snapshot["settings"]?.arrayValue?.first { $0["key"]?.stringValue == key }
    }

    @Test("a line that is only a comment is not a setting")
    func comments() throws {
        let config = try ServeConfig.parse("# the port\nport = 9000  # as agreed\n\n")
        #expect(config.port == 9000)
    }

    @Test("a key is changed where it stands, and the rest of the file is left as written")
    func changeInPlace() {
        var file = ConfigFileText("# mine\nslots = 4  # few\nport = 9000\n")
        file.set("slots", to: "12")
        file.set("kvFormat", to: "\"fp16\"")
        #expect(file.text == "# mine\nslots = 12  # few\nport = 9000\nkvFormat = \"fp16\"\n")
        #expect(file.value("kvFormat") == "fp16")
        file.set("port", to: nil)
        #expect(file.text == "# mine\nslots = 12  # few\nkvFormat = \"fp16\"\n")
        #expect(file.value("port") == nil)
    }

    @Test("every setting offered is one the file's reader takes, and reads back")
    func everyKey() throws {
        let samples = [
            "host": "0.0.0.0", "port": "9001", "modelID": "m", "requestLog": "false", "weightsPath": "/w", "tokenizerPath": "/t",
            "draftPath": "none", "contextWindow": "1000", "slots": "3", "kvPages": "640", "kvFormat": "fp16", "maxRows": "64",
            "concurrency": "2", "decodeWeight": "2.5", "prefixCacheDir": "none", "prefixCacheGiB": "20", "prefixCacheMinTokens": "10",
            "drainSeconds": "5", "restartDrainSeconds": "7", "dictionaryStudy": "true", "dictionaryCorpus": "/c",
            "attentionScan": "m48c64s8", "answerReserve": "4096", "toolCallOverrun": "0", "continueOnLimit": "true",
            // With no models registered, the one there is has the server's name for it.
            "model": "qwen3.8-27b", "modelSwitch": "manual", "modelDwellSeconds": "5", "switchWaitSeconds": "30",
        ]
        #expect(Set(samples.keys) == Set(ServeSetting.all.map(\.key)))
        for setting in ServeSetting.all {
            let sample = try #require(samples[setting.key])
            var file = ConfigFileText("")
            file.set(setting.key, to: setting.kind == .text || setting.kind == .choice ? "\"\(sample)\"" : sample)
            let config = try ServeConfig.parse(file.text)
            #expect(config.text(setting.key) == sample, "\(setting.key)")
            #expect(config != ServeConfig(), "\(setting.key)")
        }
    }

    @Test("a refused value names its setting")
    func refused() {
        #expect(throws: SettingsError(key: "slots", message: "Slots: 0 is not a value it can take")) {
            _ = try ServeSettings.change("", [("slots", "0")])
        }
        #expect(throws: SettingsError(key: "nonsense", message: "there is no setting called nonsense")) {
            _ = try ServeSettings.change("", [("nonsense", "1")])
        }
        #expect(throws: SettingsError(key: "weightsPath", message: "Weights: there is no file at /nowhere/weights.splw")) {
            _ = try ServeSettings.change("", [("weightsPath", "/nowhere/weights.splw")])
        }
        #expect(throws: SettingsError(key: "modelID", message: "Model name: quotes and # cannot be written to splosh.toml")) {
            _ = try ServeSettings.change("", [("modelID", "a # b")])
        }
        #expect(throws: SettingsError(key: "kvFormat", message: "KV precision: int4 is not one of int8, q4, fp16")) {
            _ = try ServeSettings.change("", [("kvFormat", "int4")])
        }
    }

    @Test("an empty value puts a setting back to its default")
    func unset() throws {
        let changed = try ServeSettings.change("slots = 4\nport = 9000\n", [("slots", ""), ("port", nil), ("concurrency", "3")])
        #expect(changed.text == "concurrency = 3\n")
        #expect(changed.config.slots == ServeConfig().slots)
        #expect(changed.config.concurrency == 3)
    }

    @Test("the page is told what is saved, what is running, and what waits for a restart")
    func snapshot() throws {
        let url = try Self.temporaryFile("slots = 12\nrequestLog = false\n")
        defer { try? FileManager.default.removeItem(at: url) }
        // The engine was started before the file said this, and has sized its own KV pool.
        let loaded = ServeConfig()
        var running = loaded
        running.kvPages = 4096
        let snapshot = ServeSettings.snapshot(path: url.path, cliPort: nil, loaded: loaded, running: running, canRestart: true)
        #expect(snapshot["canRestart"]?.boolValue == true)
        let slots = try #require(Self.setting("slots", in: snapshot))
        #expect(slots["value"]?.stringValue == "12")
        #expect(slots["running"]?.stringValue == "8")
        #expect(slots["pending"]?.boolValue == true)
        let log = try #require(Self.setting("requestLog", in: snapshot))
        #expect(log["running"]?.stringValue == "false")
        #expect(log["pending"]?.boolValue == false)
        let pages = try #require(Self.setting("kvPages", in: snapshot))
        #expect(pages["value"]?.isNull == true)
        #expect(pages["running"]?.stringValue == "4096")
        #expect(pages["unset"]?.stringValue == "sized to the machine's memory")
        #expect(pages["pending"]?.boolValue == false)
    }

    @Test("saving writes the file and puts the immediate settings into effect")
    func save() throws {
        final class Applied: @unchecked Sendable { var config: ServeConfig? }
        let url = try Self.temporaryFile("# kept\nslots = 4\n")
        defer { try? FileManager.default.removeItem(at: url) }
        let applied = Applied()
        let service = ServeSettings.service(path: url.path, cliPort: nil, loaded: ServeConfig(), running: ServeConfig(), holder: nil) { applied.config = $0 }
        #expect(service.restart == nil)
        let after = try service.write([("decodeWeight", "2"), ("dictionaryStudy", "true")])
        #expect(try String(contentsOf: url, encoding: .utf8) == "# kept\nslots = 4\ndecodeWeight = 2\ndictionaryStudy = true\n")
        #expect(applied.config?.decodeWeight == 2)
        #expect(applied.config?.dictionaryStudy == true)
        #expect(Self.setting("dictionaryStudy", in: after)?["running"]?.stringValue == "true")
        // A refused change leaves the file as it was.
        #expect(throws: SettingsError.self) { _ = try service.write([("requestLog", "true"), ("port", "0")]) }
        #expect(try String(contentsOf: url, encoding: .utf8) == "# kept\nslots = 4\ndecodeWeight = 2\ndictionaryStudy = true\n")
    }

    private static let registry = "model.zed = \"/models/zed.splw\"\nmodel.alpha = \"/models/alpha.splw\"\nmodel.mid = \"/models/mid.splw\"\n"

    @Test("with models registered, the starting model is a choice of their ids, in the order the file gives them")
    func startingModelChoice() throws {
        let url = try Self.temporaryFile(Self.registry + "model = \"alpha\"\n")
        defer { try? FileManager.default.removeItem(at: url) }
        let snapshot = ServeSettings.snapshot(path: url.path, cliPort: nil, loaded: ServeConfig(), running: ServeConfig(), canRestart: true)
        let model = try #require(Self.setting("model", in: snapshot))
        #expect(model["kind"]?.stringValue == "choice")
        #expect(model["choices"]?.arrayValue?.compactMap(\.stringValue) == ["zed", "alpha", "mid"])
        #expect(model["value"]?.stringValue == "alpha")
        #expect(model["default"]?.isNull == true)
        // Unset starts on the first, and the page's blank option says which that is.
        #expect(model["unset"]?.stringValue == "zed, the first registered")
        // No other setting changes with it.
        #expect(Self.setting("modelID", in: snapshot)?["kind"]?.stringValue == "text")
        #expect(Self.setting("modelSwitch", in: snapshot)?["choices"]?.arrayValue?.compactMap(\.stringValue) == ["request", "manual"])
        #expect(Self.setting("slots", in: snapshot)?["choices"]?.arrayValue?.isEmpty == true)
    }

    @Test("with none registered the starting model stays the text it was, and offers no empty list")
    func startingModelText() throws {
        let url = try Self.temporaryFile("slots = 4\n")
        defer { try? FileManager.default.removeItem(at: url) }
        let snapshot = ServeSettings.snapshot(path: url.path, cliPort: nil, loaded: ServeConfig(), running: ServeConfig(), canRestart: true)
        let model = try #require(Self.setting("model", in: snapshot))
        #expect(model["kind"]?.stringValue == "text")
        #expect(model["choices"]?.arrayValue?.isEmpty == true)
        #expect(model["unset"]?.stringValue == "the first model registered")
        let setting = try #require(ServeSetting.named("model"))
        #expect(setting.offered(registered: []).kind == .text)
        #expect(setting.offered(registered: ["a", "b"]).kind == .choice)
        #expect(setting.offered(registered: ["a", "b"]).choices == ["a", "b"])
        // Only the starting model is a choice of the registry.
        #expect(ServeSetting.named("slots")?.offered(registered: ["a", "b"]).kind == .integer)
        #expect(ServeSetting.named("modelSwitch")?.offered(registered: ["a", "b"]).choices == ["request", "manual"])
    }

    @Test("a starting model that is not registered is refused, whichever order the changes come in; a registered one is written")
    func startingModelChecked() throws {
        let chosen = try ServeSettings.change(Self.registry, [("model", "mid")])
        #expect(chosen.text == Self.registry + "model = \"mid\"\n")
        #expect(chosen.config.model == "mid")
        #expect(throws: SettingsError(key: "model", message: "Starting model: nope is not a value it can take")) {
            _ = try ServeSettings.change(Self.registry, [("model", "nope")])
        }
        // Registered by the same change that chooses it, the choice first.
        let artifact = FileManager.default.temporaryDirectory.appendingPathComponent("splosh-model-\(UUID().uuidString).splw")
        try Data("weights".utf8).write(to: artifact)
        defer { try? FileManager.default.removeItem(at: artifact) }
        let both = try ServeSettings.change(Self.registry, [("model", "new"), ("model.new", artifact.path)])
        #expect(both.config.model == "new")
        #expect(both.config.models.map(\.id) == ["zed", "alpha", "mid", "new"])
        // Empty puts it back to the first.
        #expect(try ServeSettings.change(chosen.text, [("model", "")]).config.model == nil)
        // A refused change through the service leaves the file as it was.
        let url = try Self.temporaryFile(Self.registry)
        defer { try? FileManager.default.removeItem(at: url) }
        let service = ServeSettings.service(path: url.path, cliPort: nil, loaded: ServeConfig(), running: ServeConfig(), holder: nil) { _ in }
        #expect(throws: SettingsError.self) { _ = try service.write([("model", "nope")]) }
        #expect(try String(contentsOf: url, encoding: .utf8) == Self.registry)
    }
}
