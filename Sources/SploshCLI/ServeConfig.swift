import Foundation

/// A model the server can load: the name a request gives it, and its weight artifact.
public struct ModelEntry: Equatable, Sendable {
    public let id: String
    public var path: String

    public init(id: String, path: String) { self.id = id; self.path = path }

    /// Whether `id` can be a model's name: letters, digits, ".", "_" and "-".
    public static func isValid(id: String) -> Bool {
        !id.isEmpty && id.utf8.allSatisfy { byte in
            (byte >= 0x30 && byte <= 0x39) || (byte | 0x20 >= 0x61 && byte | 0x20 <= 0x7a) || byte == 0x2e || byte == 0x5f || byte == 0x2d
        }
    }
}

/// The small, dependency-free subset of splosh.toml consumed by `serve`.
public struct ServeConfig: Equatable, Sendable {
    public var port: Int = 8091
    /// The one model's artifact, when no models are registered (see `models`).
    public var weightsPath: String?
    public var contextWindow: Int = 262_144
    /// Directory holding tokenizer.json and tokenizer_config.json.
    public var tokenizerPath: String = "inputs/tokenizer"
    /// Sessions that can hold state at once, active or cached.
    public var slots: Int = 8
    /// KV pool size in 256-token pages, shared by every session.
    public var kvPages: Int = 0
    /// Rows evaluated per forward pass.
    public var maxRows: Int = 128
    /// DFlash 2 draft checkpoint for speculative decoding. nil uses the Hugging Face cache copy
    /// when present; "none" disables speculation.
    public var draftPath: String?
    public var host: String = "127.0.0.1"
    public var modelID: String = "qwen3.8-27b"
    /// Where evaluated prompt prefixes are kept on disk. "none" disables the store.
    public var prefixCacheDir: String = "~/Library/Caches/Splosh/prefix-cache"
    /// Disk budget for stored prefixes, in GiB (a 50K-token prefix is about 1.8 GiB).
    public var prefixCacheGiB: Int = 16
    /// False once `prefixCacheGiB` is given: left alone, the store may use a tenth of the free
    /// space on its volume, between 16 and 96 GiB (a long conversation is several GiB, and a
    /// batch of agents has more of them than fit in memory).
    public var prefixCacheAuto = true
    /// Prompts shorter than this are not stored: re-evaluating them is cheap.
    public var prefixCacheMinTokens: Int = 1024
    /// One line on standard error per finished request.
    public var requestLog: Bool = true
    /// A server started from a terminal shows its page in the browser once it is listening.
    public var openBrowser = true
    /// Most requests worked on at once (BatchScheduler.maxActive); 0 means as many as there
    /// are slots.
    public var concurrency: Int = 0
    /// How long a controlled stop waits for requests in flight before ending them.
    public var drainSeconds: Double = 600
    /// How long a restart (`splosh serve --restart`) lets requests in flight run on before it
    /// cuts them. Short, because the requests that are waiting for the new engine wait this
    /// long too, and a cut request loses little: its context is saved as far as it got and its
    /// client sends it again.
    public var restartDrainSeconds: Double = 30
    /// What a generated token is worth in prompt tokens when decoding sessions and waiting
    /// prompts share a step (BatchScheduler.decodeWeight).
    public var decodeWeight: Double = 1
    /// Tokens of a reply's limit kept for its answer: thinking that has used the rest is closed
    /// by the scheduler (BatchScheduler.answerReserve). 0 lets thinking run to the limit.
    public var answerReserve: Int = 8192
    /// Tokens a tool call in progress at a reply's limit may run past it, to be finished
    /// (BatchScheduler.toolCallOverrun). 0 ends every reply at its limit.
    public var toolCallOverrun: Int = 8192
    /// A reply cut at the token limit its client set ends with a call to the client's shell tool
    /// that echoes a note to carry on, so a harness goes on with a fresh limit where it would
    /// have ended the turn (LimitContinuation).
    public var continueOnLimit = false
    /// Report what drafts copied from other conversations, and from the source files under
    /// `dictionaryCorpus`, would have yielded (DictionaryStudy). Nothing decoded changes.
    public var dictionaryStudy = false
    public var dictionaryCorpus: String?
    /// The models the server can load, by the name a request gives one: the `model.<id>` lines,
    /// in the order written. One is loaded at a time. Empty when the file registers none: then
    /// there is the one model, `modelID`, at `weightsPath` (see `registry`).
    public var models: [ModelEntry] = []
    /// The registered model the server starts on; unset is the first.
    public var model: String?
    /// "request": a request naming a registered model that is not the one loaded has it loaded
    /// in the other's place. "manual": only `splosh models --load` does that.
    public var modelSwitch: String = "request"
    /// How long a model just loaded is kept before a request may have another loaded in its
    /// place, so that clients of two models each get a turn.
    public var modelDwellSeconds: Double = 60
    /// How long a request for another model waits for the loaded one's requests to finish
    /// before it is refused.
    public var switchWaitSeconds: Double = 120

    // Converted models are kept in `models/`, a directory of their own: what cleans the build
    // (`swift package clean` empties `.build`) is not to take 16 GB of model with it.
    public static let rowMajorWeightsPath = "models/q4/weights.splw"
    public static let tiledWeightsPath = "models/q4/weights.tiled.splw"
    /// The tiled artifact when it exists (one resident copy), otherwise the row-major one.
    public static var defaultWeightsPath: String {
        FileManager.default.fileExists(atPath: tiledWeightsPath) ? tiledWeightsPath : rowMajorWeightsPath
    }
    /// KV cache precision: "int8" (default, 32.5 KiB per token, the format the fused accelerator
    /// attention runs on), "q4" (18 KiB) or "fp16" (64 KiB).
    public var kvFormat: String = "int8"
    /// The attention scan kernel for int8 KV, by its shape name; unset is the engine's default.
    /// "r48c64s8" (the default) gains most at long context: against "m48c64s8", the kernel before
    /// it, 7 ms off a 128-row step at 8K, 33 at 36K, 117 at 100K. Others: "r48c64s8q4",
    /// "r48c32s8q4", "b48c64s8", "h48c64s8".
    public var attentionScan: String?

    public init(port: Int = 8091, weightsPath: String? = nil, contextWindow: Int = 262_144) {
        self.port = port
        self.weightsPath = weightsPath
        self.contextWindow = contextWindow
    }

    /// A session cannot outgrow the KV pool, whatever the configured window says.
    public var effectiveContext: Int { min(contextWindow, kvPages * 256) }

    /// Whether splosh.toml registers models (`model.<id>` lines).
    public var hasRegistry: Bool { !models.isEmpty }

    /// The models that can be loaded: the registered ones, or the one model there is without a
    /// registry, named `modelID`.
    public var registry: [ModelEntry] {
        hasRegistry ? models : [ModelEntry(id: modelID, path: weightsPath ?? Self.defaultWeightsPath)]
    }

    /// The model a server starts on, and a line for standard error when a choice was passed over.
    /// In order: `held`, the one the process holding the port asks for (the model that was
    /// loaded, across a restart, or the one a switch is to); `cli`, from `serve --model`;
    /// `model`; the first registered.
    public func startingModel(held: String? = nil, cli: String? = nil) throws -> (model: ModelEntry, note: String?) {
        let registry = self.registry
        var note: String?
        if let held {
            if let entry = registry.first(where: { $0.id == held }) { return (entry, nil) }
            note = "no model called \(held) is registered now; starting on the configured one\n"
        }
        if let cli {
            guard let entry = registry.first(where: { $0.id == cli }) else {
                throw CLIError("--model: no model called '\(cli)' is registered (splosh.toml has \(registry.map(\.id).joined(separator: ", ")))")
            }
            return (entry, note)
        }
        return (registry.first { $0.id == model } ?? registry[0], note)
    }

    /// The model a server starts on, of those whose artifact is there (`exists` says of a path):
    /// as `startingModel`, but a starting model with no artifact gives way to the first
    /// registered one that has one, with a line saying so. Nil when none has: the server then
    /// starts without a model, to offer the downloads (see ServeDownloads).
    ///
    /// A model asked for by name (`held`, `cli`, or `weightsPath` where no models are registered)
    /// is returned whether it is there or not: its absence is an error, and the caller's to report.
    public func installedStart(held: String? = nil, cli: String? = nil, exists: (String) -> Bool) throws -> (model: ModelEntry, note: String?)? {
        let chosen = try startingModel(held: held, cli: cli)
        if exists(chosen.model.path) || cli != nil || chosen.model.id == held || (!hasRegistry && weightsPath != nil) { return chosen }
        guard let other = registry.first(where: { exists($0.path) }) else { return nil }
        return (other, (chosen.note ?? "") + "\(chosen.model.id), the model to start on, has no artifact at \(chosen.model.path); starting on \(other.id)\n")
    }

    /// Resolve the effective configuration with the documented precedence: defaults, TOML, CLI.
    public static func resolve(path: String = "./splosh.toml", cliPort: Int? = nil) throws -> ServeConfig {
        var config = try load(path: path)
        if let cliPort { config.port = cliPort }
        return config
    }

    public static func load(path: String = "./splosh.toml") throws -> ServeConfig {
        let url = URL(fileURLWithPath: path)
        guard FileManager.default.fileExists(atPath: url.path) else { return ServeConfig() }
        return try parse(String(contentsOf: url, encoding: .utf8))
    }

    /// The configuration a splosh.toml's text gives.
    public static func parse(_ text: String) throws -> ServeConfig {
        var result = ServeConfig()
        for (lineNumber, rawLine) in text.split(whereSeparator: \.isNewline).enumerated() {
            // (Empty pieces kept: a line that is only a comment has nothing before its "#".)
            let line = rawLine.split(separator: "#", maxSplits: 1, omittingEmptySubsequences: false).first.map(String.init)?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
            guard !line.isEmpty else { continue }
            let parts = line.split(separator: "=", maxSplits: 1).map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            guard parts.count == 2 else { throw ServeConfigError.invalidLine(lineNumber + 1) }
            let key = parts[0]
            guard !key.isEmpty, parts[1].isEmpty == false else { throw ServeConfigError.invalidLine(lineNumber + 1) }
            let value = unquote(parts[1])
            switch key {
            case "port": result.port = try CLIArguments.portValue(value)
            case "contextWindow":
                guard let n = Int(value), n > 0 else { throw ServeConfigError.invalidValue(key) }
                result.contextWindow = n
            case "weightsPath":
                guard !value.isEmpty else { throw ServeConfigError.invalidValue(key) }
                result.weightsPath = value
            case "tokenizerPath":
                guard !value.isEmpty else { throw ServeConfigError.invalidValue(key) }
                result.tokenizerPath = value
            case "slots", "kvPages", "maxRows":
                guard let n = Int(value), n > 0 else { throw ServeConfigError.invalidValue(key) }
                if key == "slots" { result.slots = n } else if key == "kvPages" { result.kvPages = n } else { result.maxRows = n }
            case "prefixCacheDir":
                guard !value.isEmpty else { throw ServeConfigError.invalidValue(key) }
                result.prefixCacheDir = value
            case "prefixCacheGiB", "prefixCacheMinTokens":
                guard let n = Int(value), n >= 0 else { throw ServeConfigError.invalidValue(key) }
                if key == "prefixCacheGiB" { result.prefixCacheGiB = n; result.prefixCacheAuto = false } else { result.prefixCacheMinTokens = n }
            case "decodeWeight":
                guard let weight = Double(value), weight >= 0 else { throw ServeConfigError.invalidValue(key) }
                result.decodeWeight = weight
            case "drainSeconds":
                guard let seconds = Double(value), seconds >= 0 else { throw ServeConfigError.invalidValue(key) }
                result.drainSeconds = seconds
            case "restartDrainSeconds":
                guard let seconds = Double(value), seconds >= 0 else { throw ServeConfigError.invalidValue(key) }
                result.restartDrainSeconds = seconds
            case "concurrency":
                guard let n = Int(value), n >= 0 else { throw ServeConfigError.invalidValue(key) }
                result.concurrency = n
            case "answerReserve", "toolCallOverrun":
                guard let n = Int(value), n >= 0 else { throw ServeConfigError.invalidValue(key) }
                if key == "answerReserve" { result.answerReserve = n } else { result.toolCallOverrun = n }
            case "requestLog", "dictionaryStudy", "continueOnLimit", "openBrowser":
                guard value == "true" || value == "false" else { throw ServeConfigError.invalidValue(key) }
                switch key {
                case "requestLog": result.requestLog = value == "true"
                case "openBrowser": result.openBrowser = value == "true"
                case "continueOnLimit": result.continueOnLimit = value == "true"
                default: result.dictionaryStudy = value == "true"
                }
            case "dictionaryCorpus":
                guard !value.isEmpty else { throw ServeConfigError.invalidValue(key) }
                result.dictionaryCorpus = value
            case "kvFormat":
                guard value == "int8" || value == "q4" || value == "fp16" else { throw ServeConfigError.invalidValue(key) }
                result.kvFormat = value
            case "attentionScan":
                result.attentionScan = value
            case "draftPath":
                guard !value.isEmpty else { throw ServeConfigError.invalidValue(key) }
                result.draftPath = value
            case "host":
                guard !value.isEmpty else { throw ServeConfigError.invalidValue(key) }
                result.host = value
            case "modelID":
                guard !value.isEmpty else { throw ServeConfigError.invalidValue(key) }
                result.modelID = value
            case "model":
                guard !value.isEmpty else { throw ServeConfigError.invalidValue(key) }
                result.model = value
            case "modelSwitch":
                guard value == "request" || value == "manual" else { throw ServeConfigError.invalidValue(key) }
                result.modelSwitch = value
            case "modelDwellSeconds", "switchWaitSeconds":
                guard let seconds = Double(value), seconds >= 0 else { throw ServeConfigError.invalidValue(key) }
                if key == "modelDwellSeconds" { result.modelDwellSeconds = seconds } else { result.switchWaitSeconds = seconds }
            case _ where key.hasPrefix(modelKeyPrefix):
                // The registry. TOML would read the dot as a table; this reader takes the key whole.
                let id = String(key.dropFirst(modelKeyPrefix.count))
                guard ModelEntry.isValid(id: id), !value.isEmpty else { throw ServeConfigError.invalidValue(key) }
                if let index = result.models.firstIndex(where: { $0.id == id }) {
                    result.models[index].path = value
                } else {
                    result.models.append(ModelEntry(id: id, path: value))
                }
            default: continue
            }
        }
        // Whatever order the lines came in: the starting model is one that can be loaded.
        if let model = result.model, !result.registry.contains(where: { $0.id == model }) { throw ServeConfigError.invalidValue("model") }
        return result
    }

    /// What a registry line's key begins with: `model.<id> = "<path>"`.
    static let modelKeyPrefix = "model."

    static func unquote(_ value: String) -> String {
        guard value.count >= 2, (value.first == "\"" && value.last == "\"") || (value.first == "'" && value.last == "'") else { return value }
        return String(value.dropFirst().dropLast())
    }
}

public enum ServeConfigError: Error, Equatable, CustomStringConvertible {
    case invalidLine(Int)
    case invalidValue(String)
    public var description: String {
        switch self { case .invalidLine(let n): return "invalid config line \(n)"; case .invalidValue(let key): return "invalid config value for \(key)" }
    }
}
