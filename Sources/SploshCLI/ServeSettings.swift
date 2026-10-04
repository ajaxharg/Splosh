// ServeSettings.swift — the settings page's side of splosh.toml.
//
// The page at /settings shows every key `ServeConfig` reads and saves changes to the file
// itself, so the file stays the one place the settings live and a server started by hand
// reads what the page wrote. A saved change is in one of three states: in effect at once (the
// scheduler takes it between steps), waiting for the engine to be started again (the page can
// ask for that; the port stays open, see ServeSupervisor), or waiting for the whole server to
// be (the address it listens on).

import Foundation
import SploshRuntime
import SploshServer

/// A key of splosh.toml as the settings page offers it.
struct ServeSetting: Sendable {
    enum Kind: String, Sendable { case integer, number, boolean, text, choice }
    /// When a saved change takes effect.
    enum Applies: String, Sendable { case now, engine, server }

    let key: String, group: String, title: String, help: String
    let kind: Kind
    var choices: [String] = []
    let applies: Applies
    /// What leaving it unset means, where that is not simply a value.
    var unset: String?
    /// Its value is one of the registered models' ids: a choice of those once the file registers
    /// any (see `offered`), and the text it has always been until then.
    var ofRegistry = false

    /// The kind and the choices the page is given, for a file that registers `ids`.
    func offered(registered ids: [String]) -> (kind: Kind, choices: [String]) {
        ofRegistry && !ids.isEmpty ? (.choice, ids) : (kind, choices)
    }

    static let all: [ServeSetting] = [
        ServeSetting(key: "host", group: "Server", title: "Host", help: "Address the server listens on.", kind: .text, applies: .server),
        ServeSetting(key: "port", group: "Server", title: "Port", help: "Port the server listens on. A port given on the command line wins.", kind: .integer, applies: .server),
        ServeSetting(key: "modelID", group: "Server", title: "Model name", help: "The name clients are given for the model.", kind: .text, applies: .engine),
        ServeSetting(key: "requestLog", group: "Server", title: "Request log", help: "One line in the server's terminal per finished request.", kind: .boolean, applies: .now),
        ServeSetting(key: "openBrowser", group: "Server", title: "Open the page at start",
                     help: "A server started from a terminal shows its page in the browser once it is listening. `splosh serve --no-open` leaves it out for one start.",
                     kind: .boolean, applies: .server),

        ServeSetting(key: "weightsPath", group: "Model", title: "Weights", help: "The converted weight artifact. Not used when models are registered (model.<id> lines).", kind: .text, applies: .engine,
                     unset: "the tiled artifact in models/q4 when it exists, otherwise the row-major one"),
        ServeSetting(key: "tokenizerPath", group: "Model", title: "Tokenizer", help: "Directory holding tokenizer.json and tokenizer_config.json.", kind: .text, applies: .engine),
        ServeSetting(key: "draftPath", group: "Model", title: "Draft model", help: "DFlash 2 draft checkpoint for speculative decoding. \"none\" switches speculation off.",
                     kind: .text, applies: .engine, unset: "the copy in inputs/draft or the Hugging Face cache, when there is one"),

        // The registered models' own lines follow these on the page (see `artifact(of:)`).
        ServeSetting(key: "model", group: "Models", title: "Starting model", help: "The registered model the server starts on. A model loaded since is kept when the engine restarts.",
                     kind: .text, applies: .server, unset: "the first model registered", ofRegistry: true),
        ServeSetting(key: "modelSwitch", group: "Models", title: "Switch models",
                     help: "request: a request naming a registered model that is not loaded has it loaded in place of the one that is. manual: only `splosh models --load` does.",
                     kind: .choice, choices: ["request", "manual"], applies: .engine),
        ServeSetting(key: "modelDwellSeconds", group: "Models", title: "Model turn, seconds",
                     help: "How long a model just loaded is kept before a request may have another loaded in its place.", kind: .number, applies: .server),
        ServeSetting(key: "switchWaitSeconds", group: "Models", title: "Switch wait, seconds",
                     help: "How long a request for another model waits for the loaded one's requests to finish before it is refused.", kind: .number, applies: .server),

        ServeSetting(key: "contextWindow", group: "Memory", title: "Context window", help: "Most tokens a session may hold. The KV pool is a second limit.", kind: .integer, applies: .engine),
        ServeSetting(key: "slots", group: "Memory", title: "Slots", help: "Sessions that can hold state at once, active or cached.", kind: .integer, applies: .engine),
        ServeSetting(key: "kvPages", group: "Memory", title: "KV pool pages", help: "KV pool size in 256-token pages, shared by every session.", kind: .integer, applies: .engine,
                     unset: "sized to the machine's memory"),
        ServeSetting(key: "kvFormat", group: "Memory", title: "KV precision", help: "int8 is 32.5 KiB a token and the format the fused attention runs on; q4 is 18 KiB; fp16 is 64 KiB.",
                     kind: .choice, choices: ["int8", "q4", "fp16"], applies: .engine),
        // The names in the shader library (see ServeConfig.attentionScan). A name it does not
        // have stops the engine from starting, so the page offers these and no others.
        ServeSetting(key: "attentionScan", group: "Memory", title: "Attention scan kernel", help: "The attention scan kernel for int8 KV, by its shape name.",
                     kind: .choice, choices: ["r48c64s8", "m48c64s8", "r48c64s8q4", "r48c32s8q4", "b48c64s8", "h48c64s8"], applies: .engine,
                     unset: "the engine's default (r48c64s8)"),
        ServeSetting(key: "maxRows", group: "Memory", title: "Rows per step", help: "Rows evaluated per forward pass.", kind: .integer, applies: .engine),

        ServeSetting(key: "concurrency", group: "Scheduling", title: "Concurrency", help: "Most requests worked on at once; the rest queue. 0 is as many as there are slots.", kind: .integer, applies: .now),
        ServeSetting(key: "decodeWeight", group: "Scheduling", title: "Decode weight",
                     help: "What a generated token is worth in prompt tokens when decoding sessions and waiting prompts share a step. Low values favour waiting prompts.",
                     kind: .number, applies: .now),

        ServeSetting(key: "answerReserve", group: "Replies", title: "Answer reserve, tokens",
                     help: "Tokens of a reply's limit kept for its answer: thinking that has used the rest is closed by the server, so an answer is still written. At most a quarter of the limit. 0 lets thinking run to the limit.",
                     kind: .integer, applies: .now),
        ServeSetting(key: "toolCallOverrun", group: "Replies", title: "Tool call overrun, tokens",
                     help: "Tokens a tool call in progress at a reply's limit may run past it, so the reply ends on a whole call and an agent carries on. 0 ends every reply at its limit.",
                     kind: .integer, applies: .now),
        ServeSetting(key: "continueOnLimit", group: "Replies", title: "Carry on at the token limit",
                     help: "A reply cut at the token limit its client set ends with a call to the client's shell tool (bash) that echoes a note to carry on, so a harness goes on with a fresh limit instead of ending the turn. Only for a request with such a tool.",
                     kind: .boolean, applies: .now),

        ServeSetting(key: "prefixCacheDir", group: "Prefix cache", title: "Directory", help: "Where evaluated prompt prefixes are kept on disk. \"none\" switches the store off.", kind: .text, applies: .engine),
        ServeSetting(key: "prefixCacheGiB", group: "Prefix cache", title: "Disk budget, GiB", help: "Disk space for stored prefixes (a 50K-token prefix is about 1.8 GiB). 0 switches the store off.",
                     kind: .integer, applies: .engine, unset: "a tenth of the free space on its volume, between 16 and 96"),
        ServeSetting(key: "prefixCacheMinTokens", group: "Prefix cache", title: "Shortest prompt stored", help: "Prompts with fewer tokens than this are not stored.", kind: .integer, applies: .engine),

        ServeSetting(key: "drainSeconds", group: "Stopping", title: "Stop drain, seconds", help: "How long a stop waits for requests in flight before ending them.", kind: .number, applies: .engine),
        ServeSetting(key: "restartDrainSeconds", group: "Stopping", title: "Restart drain, seconds", help: "How long a restart lets requests in flight run on before it cuts them.", kind: .number, applies: .server),

        ServeSetting(key: "dictionaryStudy", group: "Studies", title: "Dictionary study",
                     help: "Report with the request log what drafts copied from other conversations, and from the corpus below, would have yielded. Nothing decoded changes.",
                     kind: .boolean, applies: .now),
        ServeSetting(key: "dictionaryCorpus", group: "Studies", title: "Dictionary corpus", help: "Directory of source files the study also looks drafts up in.", kind: .text, applies: .now,
                     unset: "no corpus"),
    ]

    /// A registered model's line, `model.<id>`: there is one for each model, so they are not in `all`.
    static func artifact(of id: String) -> ServeSetting {
        ServeSetting(key: ServeConfig.modelKeyPrefix + id, group: "Models", title: "Model \(id)", help: "The converted weight artifact served as \(id).",
                     kind: .text, applies: .engine, unset: "not registered")
    }

    /// The setting `key` names: one of `all`, or a model's line.
    static func named(_ key: String) -> ServeSetting? {
        if let setting = all.first(where: { $0.key == key }) { return setting }
        guard key.hasPrefix(ServeConfig.modelKeyPrefix) else { return nil }
        let id = String(key.dropFirst(ServeConfig.modelKeyPrefix.count))
        return ModelEntry.isValid(id: id) ? artifact(of: id) : nil
    }
}

extension ServeConfig {
    /// The value of a key as splosh.toml would give it; nil for one that is left to its default
    /// and has no value of its own (see `ServeSetting.unset`).
    func text(_ key: String) -> String? {
        func number(_ value: Double) -> String { value == value.rounded() && abs(value) < 1e15 ? String(Int(value)) : String(value) }
        switch key {
        case "host": return host
        case "port": return String(port)
        case "modelID": return modelID
        case "requestLog": return String(requestLog)
        case "openBrowser": return String(openBrowser)
        case "weightsPath": return weightsPath
        case "tokenizerPath": return tokenizerPath
        case "draftPath": return draftPath
        case "contextWindow": return String(contextWindow)
        case "slots": return String(slots)
        case "kvPages": return kvPages > 0 ? String(kvPages) : nil
        case "kvFormat": return kvFormat
        case "attentionScan": return attentionScan
        case "maxRows": return String(maxRows)
        case "concurrency": return String(concurrency)
        case "decodeWeight": return number(decodeWeight)
        case "answerReserve": return String(answerReserve)
        case "toolCallOverrun": return String(toolCallOverrun)
        case "continueOnLimit": return String(continueOnLimit)
        case "prefixCacheDir": return prefixCacheDir
        case "prefixCacheGiB": return prefixCacheAuto ? nil : String(prefixCacheGiB)
        case "prefixCacheMinTokens": return String(prefixCacheMinTokens)
        case "drainSeconds": return number(drainSeconds)
        case "restartDrainSeconds": return number(restartDrainSeconds)
        case "dictionaryStudy": return String(dictionaryStudy)
        case "dictionaryCorpus": return dictionaryCorpus
        case "model": return model
        case "modelSwitch": return modelSwitch
        case "modelDwellSeconds": return number(modelDwellSeconds)
        case "switchWaitSeconds": return number(switchWaitSeconds)
        case _ where key.hasPrefix(Self.modelKeyPrefix):
            return models.first { $0.id == key.dropFirst(Self.modelKeyPrefix.count) }?.path
        default: return nil
        }
    }
}

/// splosh.toml as written, to change one key and leave every other line as it is.
struct ConfigFileText {
    private(set) var lines: [String]

    init(_ text: String) {
        lines = text.split(separator: "\n", omittingEmptySubsequences: false).map(String.init)
        if lines.last == "" { lines.removeLast() }
    }

    var text: String { lines.isEmpty ? "" : lines.joined(separator: "\n") + "\n" }

    /// The key a line sets and what it sets it to, read as `ServeConfig.parse` reads it.
    private static func assignment(_ line: String) -> (key: String, value: String)? {
        let code = line.split(separator: "#", maxSplits: 1, omittingEmptySubsequences: false).first.map(String.init) ?? ""
        let parts = code.split(separator: "=", maxSplits: 1).map { $0.trimmingCharacters(in: .whitespaces) }
        return parts.count == 2 ? (parts[0], ServeConfig.unquote(parts[1])) : nil
    }

    /// What the file gives `key`; nil when it does not set it.
    func value(_ key: String) -> String? {
        lines.last { Self.assignment($0)?.key == key }.flatMap { Self.assignment($0)?.value }
    }

    /// Set `key` to `written` (as it is to appear, quoted if it is text) where the file sets
    /// it now, keeping that line's comment, or at the end; nil takes the key out.
    mutating func set(_ key: String, to written: String?) {
        let found = lines.indices.filter { Self.assignment(lines[$0])?.key == key }
        if let written {
            let line = "\(key) = \(written)"
            if let last = found.last {
                lines[last] = line + (lines[last].firstIndex(of: "#").map { "  " + lines[last][$0...] } ?? "")
            } else {
                lines.append(line)
            }
        }
        for index in (written == nil ? found : Array(found.dropLast())).reversed() { lines.remove(at: index) }
    }
}

enum ServeSettings {
    /// The page's view of the settings: what the file says, what each key is when unset, what
    /// the running server has, and which saved changes it has not taken yet.
    ///
    /// `loaded` is the configuration the engine was started on and `running` the same with the
    /// figures it sized for itself filled in. `cliPort` is a port given on the command line.
    static func snapshot(path: String, cliPort: Int?, loaded: ServeConfig, running: ServeConfig, canRestart: Bool) -> JSONValue {
        let text = (try? String(contentsOfFile: path, encoding: .utf8)) ?? ""
        let file = ConfigFileText(text)
        var saved = (try? ServeConfig.parse(text)) ?? loaded
        if let cliPort { saved.port = cliPort }
        let defaults = ServeConfig()
        func json(_ value: String?) -> JSONValue { value.map(JSONValue.string) ?? .null }
        // The models the file registers, and any the engine was started with that it no longer does.
        let registered = saved.models.map(\.id) + loaded.models.map(\.id).filter { id in !saved.models.contains { $0.id == id } }
        // The starting model is chosen from the models the file registers, as the page is to show them.
        let ids = saved.models.map(\.id)
        let settings = (ServeSetting.all + registered.map(ServeSetting.artifact(of:))).map { setting -> JSONValue in
            let live = setting.applies == .now
            let (kind, choices) = setting.offered(registered: ids)
            return .object([
                ("key", .string(setting.key)), ("group", .string(setting.group)), ("title", .string(setting.title)), ("help", .string(setting.help)),
                ("kind", .string(kind.rawValue)), ("choices", .array(choices.map(JSONValue.string))),
                ("applies", .string(setting.applies.rawValue)),
                ("value", json(file.value(setting.key))),
                ("default", json(defaults.text(setting.key))),
                ("unset", .string(kind == .choice && setting.ofRegistry ? "\(ids[0]), the first registered" : setting.unset ?? defaults.text(setting.key) ?? "")),
                ("running", json((live ? saved : running).text(setting.key))),
                ("pending", .bool(!live && saved.text(setting.key) != loaded.text(setting.key))),
            ])
        }
        return .object([
            ("file", .string(URL(fileURLWithPath: path).standardizedFileURL.path)),
            ("canRestart", .bool(canRestart)),
            ("restartDrainSeconds", .double(loaded.restartDrainSeconds)),
            ("settings", .array(settings)),
        ])
    }

    /// The file's text with `changes` made, and the configuration it then gives. Throws
    /// `SettingsError` for a key there is not, a value the file's reader refuses, or a path
    /// that would leave the engine unable to start.
    static func change(_ text: String, _ changes: [(key: String, value: String?)]) throws -> (text: String, config: ServeConfig) {
        var file = ConfigFileText(text)
        for (key, value) in changes {
            guard let setting = ServeSetting.named(key) else {
                throw SettingsError(key: key, message: "there is no setting called \(key)")
            }
            guard let value = value?.trimmingCharacters(in: .whitespaces), !value.isEmpty else { file.set(key, to: nil); continue }
            let quoted = setting.kind == .text || setting.kind == .choice
            if quoted, value.contains(where: { "\"'#\n\r".contains($0) }) {
                throw SettingsError(key: key, message: "\(setting.title): quotes and # cannot be written to splosh.toml")
            }
            if setting.kind == .choice, !setting.choices.contains(value) {
                throw SettingsError(key: key, message: "\(setting.title): \(value) is not one of \(setting.choices.joined(separator: ", "))")
            }
            try checkPath(setting, value)
            file.set(key, to: quoted ? "\"\(value)\"" : value)
        }
        do {
            return (file.text, try ServeConfig.parse(file.text))
        } catch ServeConfigError.invalidValue(let key) {
            let title = ServeSetting.named(key)?.title ?? key
            throw SettingsError(key: key, message: "\(title): \(file.value(key) ?? "that") is not a value it can take")
        } catch {
            throw SettingsError(key: nil, message: "splosh.toml cannot be read as it stands: \(error)")
        }
    }

    /// A path the engine would fail to start on is refused here: an engine that cannot start
    /// is started again and again, and takes the settings page with it.
    private static func checkPath(_ setting: ServeSetting, _ value: String) throws {
        let path = (value as NSString).expandingTildeInPath
        var isDirectory: ObjCBool = false
        let exists = FileManager.default.fileExists(atPath: path, isDirectory: &isDirectory)
        let noFile = exists && !isDirectory.boolValue ? nil : "there is no file at \(value)"
        let problem: String?
        switch setting.key {
        case "weightsPath": problem = noFile
        case _ where setting.key.hasPrefix(ServeConfig.modelKeyPrefix): problem = noFile
        case "draftPath": problem = value == "none" || exists ? nil : "there is nothing at \(value)"
        case "tokenizerPath": problem = FileManager.default.fileExists(atPath: path + "/tokenizer.json") ? nil : "there is no tokenizer.json in \(value)"
        case "dictionaryCorpus": problem = exists && isDirectory.boolValue ? nil : "there is no directory at \(value)"
        default: problem = nil
        }
        if let problem { throw SettingsError(key: setting.key, message: "\(setting.title): \(problem)") }
    }

    /// The service behind the settings page. `apply` puts a saved configuration's immediate
    /// settings into effect; `holder` is the process that holds the port, which replaces the
    /// engine when asked.
    static func service(path: String, cliPort: Int?, loaded: ServeConfig, running: ServeConfig, holder: pid_t?,
                        apply: @escaping @Sendable (ServeConfig) -> Void) -> SettingsService {
        let read: @Sendable () -> JSONValue = { snapshot(path: path, cliPort: cliPort, loaded: loaded, running: running, canRestart: holder != nil) }
        let saving = NSLock()
        var restart: (@Sendable () -> Void)?
        if let holder {
            // After the reply has gone: the engine that sends it is the one being replaced.
            restart = { DispatchQueue.global().asyncAfter(deadline: .now() + 0.3) { _ = kill(holder, SIGUSR1) } }
        }
        return SettingsService(
            read: read,
            write: { changes in
                saving.lock(); defer { saving.unlock() }
                let changed = try change((try? String(contentsOfFile: path, encoding: .utf8)) ?? "", changes)
                do {
                    try Data(changed.text.utf8).write(to: URL(fileURLWithPath: path), options: .atomic)
                } catch {
                    throw SettingsError(key: nil, message: "could not write \(path): \(error.localizedDescription)")
                }
                apply(changed.config)
                return read()
            },
            restart: restart)
    }
}
