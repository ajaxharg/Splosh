import Foundation
import Testing
@testable import SploshCLI
import SploshServer

@Suite("ModelRegistryTests")
struct ModelRegistryTests {
    private static let registry = """
        model = "uq5"
        model.mq4 = ".build/q4/weights.tiled.splw"   # the MLX pack
        model.uq5 = ".build/gguf/ud-q5_k_m.splw"
        model.uq6 = '.build/gguf/ud-q6_k_m.splw'
        modelSwitch = "manual"
        modelDwellSeconds = 30
        switchWaitSeconds = 90.5
        """

    @Test("the registry is the model.<id> lines, in the order written")
    func parses() throws {
        let config = try ServeConfig.parse(Self.registry)
        #expect(config.models == [ModelEntry(id: "mq4", path: ".build/q4/weights.tiled.splw"),
                                  ModelEntry(id: "uq5", path: ".build/gguf/ud-q5_k_m.splw"),
                                  ModelEntry(id: "uq6", path: ".build/gguf/ud-q6_k_m.splw")])
        #expect(config.hasRegistry)
        #expect(config.registry == config.models)
        #expect(config.model == "uq5")
        #expect(config.modelSwitch == "manual")
        #expect(config.modelDwellSeconds == 30)
        #expect(config.switchWaitSeconds == 90.5)
        // The name given to requests that name no model is still its own setting.
        #expect(config.modelID == ServeConfig().modelID)
    }

    @Test("with no model.<id> lines there is the one model, named modelID, at weightsPath")
    func implicit() throws {
        let defaults = try ServeConfig.parse("")
        #expect(!defaults.hasRegistry)
        #expect(defaults.registry == [ModelEntry(id: "qwen3.8-27b", path: ServeConfig.defaultWeightsPath)])
        #expect(defaults.modelSwitch == "request")
        #expect(defaults.modelDwellSeconds == 60)
        #expect(defaults.switchWaitSeconds == 120)
        let named = try ServeConfig.parse("modelID = \"mine\"\nweightsPath = \"/w.splw\"\nmodel = \"mine\"\n")
        #expect(named.registry == [ModelEntry(id: "mine", path: "/w.splw")])
        #expect(try named.startingModel().model == ModelEntry(id: "mine", path: "/w.splw"))
    }

    @Test("a line given twice keeps its place and takes the later path")
    func repeated() throws {
        let config = try ServeConfig.parse("model.a = \"/1\"\nmodel.b = \"/2\"\nmodel.a = \"/3\"\n")
        #expect(config.models == [ModelEntry(id: "a", path: "/3"), ModelEntry(id: "b", path: "/2")])
    }

    @Test("ids are letters, digits, '.', '_' and '-'; the starting model is a registered one")
    func refused() {
        for text in ["model.a/b = \"/w\"", "model. = \"/w\"", "model.a b = \"/w\"", "model.é = \"/w\"", "model.a = \"\"",
                     "model.a = \"/w\"\nmodel = \"b\"", "model = \"other\"", "modelSwitch = \"always\"",
                     "modelDwellSeconds = -1", "switchWaitSeconds = soon", "[model]\na = \"/w\""] {
            #expect(throws: ServeConfigError.self, "\(text)") { _ = try ServeConfig.parse(text) }
        }
        #expect(ModelEntry.isValid(id: "ud-q5_k_m.v2"))
        #expect(throws: ServeConfigError.invalidValue("model")) { _ = try ServeConfig.parse("model = \"b\"\nmodel.a = \"/w\"") }
        // Wherever the line that names it stands.
        #expect((try? ServeConfig.parse("model = \"a\"\nmodel.a = \"/w\""))?.model == "a")
    }

    @Test("the starting model: the holder's, then --model, then `model`, then the first registered")
    func precedence() throws {
        let config = try ServeConfig.parse(Self.registry)
        #expect(try config.startingModel().model.id == "uq5")
        #expect(try config.startingModel(cli: "uq6").model.id == "uq6")
        #expect(try config.startingModel(held: "mq4", cli: "uq6").model.id == "mq4")
        #expect(try config.startingModel(held: "mq4").note == nil)
        var unnamed = config
        unnamed.model = nil
        #expect(try unnamed.startingModel().model.id == "mq4")
        // A model the holder had loaded that the file no longer registers is passed over, and said to be.
        let passed = try config.startingModel(held: "gone", cli: "uq6")
        #expect(passed.model.id == "uq6")
        #expect(passed.note?.contains("gone") == true)
        #expect(throws: CLIError.self) { _ = try config.startingModel(cli: "nope") }
    }

    @Test("serve --model and the models command's flags")
    func arguments() throws {
        #expect(try CLIArguments.parseServe(["--model", "uq5", "--echo"]).model == "uq5")
        var expected = ModelsArguments()
        expected.json = true; expected.port = 9000; expected.config = "c.toml"; expected.load = "uq6"
        #expect(try CLIArguments.parseModels(["--json", "--port", "9000", "--config", "c.toml", "--load", "uq6"]) == expected)
        #expect(try CLIArguments.parseModels([]) == ModelsArguments())
        #expect(throws: CLIError.self) { _ = try CLIArguments.parseModels(["--load"]) }
        #expect(throws: CLIError.self) { _ = try CLIArguments.parseModels(["--load", "a/b"]) }
        #expect(throws: CLIError.self) { _ = try CLIArguments.parseServe(["--model", "a b"]) }
    }

    @Test("the settings page has the model settings, and a line for each registered model")
    func settings() throws {
        let artifact = FileManager.default.temporaryDirectory.appendingPathComponent("splosh-model-\(UUID().uuidString).splw")
        try Data("weights".utf8).write(to: artifact)
        let file = FileManager.default.temporaryDirectory.appendingPathComponent("splosh-models-\(UUID().uuidString).toml")
        defer { try? FileManager.default.removeItem(at: artifact); try? FileManager.default.removeItem(at: file) }

        // A model is registered by its line, where there is a file; and is then one to start on.
        let added = try ServeSettings.change("slots = 4\n", [("model.a", artifact.path), ("model", "a"), ("modelSwitch", "manual")])
        #expect(added.text == "slots = 4\nmodel.a = \"\(artifact.path)\"\nmodel = \"a\"\nmodelSwitch = \"manual\"\n")
        #expect(added.config.models == [ModelEntry(id: "a", path: artifact.path)])
        #expect(added.config.text("model.a") == artifact.path)
        #expect(throws: SettingsError(key: "model.b", message: "Model b: there is no file at /nowhere/b.splw")) {
            _ = try ServeSettings.change(added.text, [("model.b", "/nowhere/b.splw")])
        }
        #expect(throws: SettingsError(key: "model", message: "Starting model: b is not a value it can take")) {
            _ = try ServeSettings.change(added.text, [("model", "b")])
        }
        #expect(throws: SettingsError(key: "model.a/b", message: "there is no setting called model.a/b")) {
            _ = try ServeSettings.change(added.text, [("model.a/b", artifact.path)])
        }
        // Taking away the model the file starts on is refused, as any file that cannot be read is.
        #expect(throws: SettingsError.self) { _ = try ServeSettings.change(added.text, [("model.a", nil)]) }

        try added.text.write(to: file, atomically: true, encoding: .utf8)
        let snapshot = ServeSettings.snapshot(path: file.path, cliPort: nil, loaded: ServeConfig(), running: ServeConfig(), canRestart: true)
        func setting(_ key: String) -> JSONValue? { snapshot["settings"]?.arrayValue?.first { $0["key"]?.stringValue == key } }
        let line = try #require(setting("model.a"))
        #expect(line["value"]?.stringValue == artifact.path)
        #expect(line["group"]?.stringValue == "Models")
        #expect(line["applies"]?.stringValue == "engine")
        #expect(line["pending"]?.boolValue == true)             // the engine was started before it was registered
        #expect(setting("modelSwitch")?["value"]?.stringValue == "manual")
        #expect(setting("modelSwitch")?["choices"]?.arrayValue?.compactMap(\.stringValue) == ["request", "manual"])
        #expect(setting("model")?["applies"]?.stringValue == "server")
        #expect(setting("modelDwellSeconds")?["default"]?.stringValue == "60")
        #expect(setting("switchWaitSeconds")?["default"]?.stringValue == "120")
    }

    @Test("the list the command prints: the loaded model marked, each with its state, size and artifact")
    func listing() throws {
        let listed = try JSONValue.parse(#"""
            {"object":"list","loaded":"uq5","data":[
              {"id":"mq4","loaded":false,"state":"available","path":"a.splw","size_bytes":16054846720},
              {"id":"uq5","loaded":true,"state":"loaded","path":"b.splw","size_bytes":19524540800},
              {"id":"uq6","loaded":false,"state":"missing","path":"c.splw","size_bytes":null}],
             "switch":{"mode":"request","target":"mq4","phase":"waiting","parked":2}}
            """#)
        let lines = ModelsCommand.table(listed).split(separator: "\n").map(String.init)
        #expect(lines.count == 4)
        #expect(lines[0].hasPrefix("  model"))
        #expect(lines[1] == "  mq4    available  14.95 GiB   a.splw")
        #expect(lines[2] == "* uq5    loaded     18.18 GiB   b.splw")
        #expect(lines[3] == "  uq6    missing    -           c.splw")
        let summary = ModelsCommand.summary(listed)
        #expect(summary.contains("switching on request: on"))
        #expect(summary.contains("switching to mq4: new requests wait while uq5 finishes the ones it has (2 requests kept until then)"))
        let manual = try JSONValue.parse(#"{"loaded":"a","data":[],"switch":{"mode":"manual","target":null,"phase":null,"parked":0}}"#)
        #expect(ModelsCommand.summary(manual) == "switching on request: off (modelSwitch = \"manual\"); `splosh models --load <id>` loads one")

        // With no server, the file's own registry in the same shape, none of it loaded.
        let config = try ServeConfig.parse("model.a = \"/nowhere/a.splw\"\n")
        let registered = ModelsCommand.registered(in: config)
        #expect(registered["loaded"]?.isNull == true)
        #expect(registered["data"]?.arrayValue?.first?["state"]?.stringValue == "missing")
        #expect(registered["data"]?.arrayValue?.first?["size_bytes"]?.isNull == true)
        #expect(registered["switch"]?["mode"]?.stringValue == "request")
    }
}
