import Foundation
import Hummingbird
import Metal
import SploshModel
import SploshRuntime
import SploshServer

public enum ServeCommand {
    public static func run(_ tokens: [String]) -> Int32 {
        do {
            let args = try CLIArguments.parseServe(tokens)
            let configPath = args.config ?? "./splosh.toml"
            var config = try ServeConfig.resolve(path: configPath, cliPort: args.port)
            if args.restart { return ServeSupervisor.requestRestart(port: config.port) }
            // The model to load, of those splosh.toml registers (see ServeConfig.startingModel).
            let environment = ProcessInfo.processInfo.environment
            let starting = try config.startingModel(held: environment[ServeSupervisor.modelVariable], cli: args.model)
            // The process started by hand holds the port and runs the engine, this same command,
            // as a child serving on a private socket (see ServeSupervisor). SPLOSH_SERVE_DIRECT
            // is the engine on the port itself, in one process, for profiling.
            let socketPath = environment[ServeSupervisor.socketVariable]
            if socketPath == nil, environment["SPLOSH_SERVE_DIRECT"] == nil {
                // With models registered, the engine is told which to load from here on: a
                // restart keeps the one that is loaded, whatever the file has come to say.
                return ServeSupervisor.run(tokens: tokens.filter { $0 != "--takeover" }, config: config, takeover: args.takeover,
                                           model: config.hasRegistry ? starting.model.id : nil)
            }
            if let note = starting.note { SploshCLI.writeStderr(note) }
            if config.hasRegistry, config.weightsPath != nil {
                SploshCLI.writeStderr("weightsPath is not used: splosh.toml registers models (its model.<id> lines), and the one loaded is \(starting.model.id)\n")
            }
            // Behind the holder: it writes to the terminal from the background, and with the
            // holder gone there is nobody to answer, from the first moment (the model takes a
            // while to load, and a holder killed meanwhile must not leave an engine behind).
            let orphaned = OrphanWatch()
            let holder: pid_t? = socketPath == nil ? nil : ProcessInfo.processInfo.environment[ServeSupervisor.holderVariable].flatMap { pid_t($0) } ?? getppid()
            if let holder {
                signal(SIGTTOU, SIG_IGN)
                orphaned.start(holder: holder)
            }
            // The settings page (ServeSettings): what this engine was started on, what it has
            // sized for itself since, and the process that can start another in its place.
            let loaded = config
            func settings(running: ServeConfig, apply: @escaping @Sendable (ServeConfig) -> Void) -> SettingsService {
                ServeSettings.service(path: configPath, cliPort: args.port, loaded: loaded, running: running, holder: holder, apply: apply)
            }
            // The models there are, and which this engine has: a request for another is the
            // holder's to see to, by starting an engine with that one (see ModelCatalog).
            let reportPath = environment[ServeSupervisor.reportVariable]
            let catalog = ModelCatalog(
                models: config.registry.map { ModelCatalog.Model(id: $0.id, path: $0.path) }, loaded: starting.model.id,
                mode: holder == nil ? .none : ModelCatalog.Mode(rawValue: config.modelSwitch) ?? .request,
                switching: { reportPath.flatMap { try? String(contentsOfFile: $0, encoding: .utf8) }.flatMap(SwitchReport.read) })
            if args.echo {
                return try serveEcho(config: config, socketPath: socketPath, settings: settings(running: config) { _ in }, catalog: catalog, model: starting.model)
            }
            guard let device = MTLCreateSystemDefaultDevice() else { throw CLIError("no Metal device available") }
            let weightsPath = starting.model.path
            guard FileManager.default.fileExists(atPath: weightsPath) else {
                throw CLIError(config.hasRegistry
                    ? "weight artifact not found at \(weightsPath), where splosh.toml registers the model \(starting.model.id)"
                    : "weight artifact not found at \(weightsPath); run `splosh convert` or set weightsPath in splosh.toml")
            }
            let tokenizerDir = URL(fileURLWithPath: config.tokenizerPath, isDirectory: true)
            let tokenizer = try Tokenizer(tokenizerURL: tokenizerDir.appendingPathComponent("tokenizer.json"),
                                          configURL: tokenizerDir.appendingPathComponent("tokenizer_config.json"))
            let loadStarted = Date()
            let weights = try ModelWeights(device: device, artifactURL: URL(fileURLWithPath: weightsPath))
            if config.kvPages <= 0 {
                // The KV pool is committed only as sessions fill it, so it is sized to what the
                // machine could hold with everything else loaded, not to a guess at the load:
                // most of the GPU's working set, less the weights and about 5 GiB of state,
                // scratch and the draft model.
                let perToken = config.kvFormat == "fp16" ? 65_536 : config.kvFormat == "q4" ? 18_432 : 33_280
                let budget = Int(Double(device.recommendedMaxWorkingSetSize) * 0.72) - weights.resident.residentBytes - (5 << 30)
                config.kvPages = min(8192, max(512, budget / (perToken * Engine.pageTokens) / 64 * 64))
            }
            var engineConfig = EngineConfig(maxSlots: config.slots, maxRows: config.maxRows,
                                            kvPages: config.kvPages, maxContext: config.effectiveContext)
            engineConfig.kvFormat = EngineConfig.KVFormat(rawValue: config.kvFormat) ?? .int8
            let engine = try Engine(device: device, weights: weights, config: engineConfig)
            if let shape = config.attentionScan, engineConfig.kvFormat == .int8 {
                try engine.setScanShape(shape)
                SploshCLI.writeStderr("attention    scan kernel \(shape)\n")
            }
            var draft: DraftModel?
            if config.draftPath != "none" {
                if let url = config.draftPath.map({ URL(fileURLWithPath: $0) }) ?? GenerateCommand.defaultDraftURL() {
                    draft = try DraftModel(engine: engine, safetensorsURL: url)
                    SploshCLI.writeStderr("draft        \(url.path)\n")
                } else {
                    SploshCLI.writeStderr("draft        none found; decoding one token per step (set draftPath in splosh.toml)\n")
                }
            }
            try engine.warmUp()
            report(engine: engine, weights: weights, path: weightsPath, seconds: Date().timeIntervalSince(loadStarted), device: device)
            if let draft {
                SploshCLI.writeStderr(String(format: "  draft      %7.2f GiB   DFlash 2, quantised to q4; %.2f GiB scratch and context rings\n",
                                             Double(draft.residentBytes) / 1_073_741_824, Double(draft.scratchBytes) / 1_073_741_824))
            }
            let scheduler = BatchScheduler(engine: engine, draft: draft)
            scheduler.requestLog = config.requestLog
            scheduler.decodeWeight = config.decodeWeight
            scheduler.answerReserve = config.answerReserve
            scheduler.toolCallOverrun = config.toolCallOverrun
            if config.concurrency > 0 { scheduler.maxActive = config.concurrency }
            if config.dictionaryStudy { scheduler.dictionaryStudy = dictionaryStudy(config, tokenizer: tokenizer) }
            let replyOptions = ReplyOptions(continueOnLimit: config.continueOnLimit)
            let live = LiveSettings(scheduler: scheduler, tokenizer: tokenizer, options: replyOptions, applied: config)
            if config.prefixCacheDir != "none", config.prefixCacheGiB > 0 {
                let directory = URL(fileURLWithPath: (config.prefixCacheDir as NSString).expandingTildeInPath, isDirectory: true)
                if config.prefixCacheAuto {
                    try? FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
                    let free = (try? directory.resourceValues(forKeys: [.volumeAvailableCapacityForImportantUsageKey]))?.volumeAvailableCapacityForImportantUsage ?? 0
                    config.prefixCacheGiB = min(96, max(16, Int(free >> 30) / 10))
                }
                let store = try PrefixStore(directory: directory, identity: engine.snapshotIdentity,
                                            maxBytes: config.prefixCacheGiB << 30, minTokens: config.prefixCacheMinTokens)
                scheduler.prefixStore = store
                let stats = store.stats
                SploshCLI.writeStderr(String(format: "prefix cache %@  (%d stored, %.2f of %d GiB; prompts of %d tokens or more)\n",
                                             directory.path, stats.entries, Double(stats.bytes) / 1_073_741_824,
                                             config.prefixCacheGiB, config.prefixCacheMinTokens))
            }
            let backend = ChatBackend(scheduler: scheduler, tokenizer: tokenizer,
                                      modelID: config.modelID, maxContext: config.effectiveContext, options: replyOptions)
            var running = config
            if scheduler.prefixStore != nil { running.prefixCacheAuto = false }
            let app = Server.application(backend: backend, host: config.host, port: config.port, socketPath: socketPath,
                                         settings: settings(running: running) { live.apply($0) }, catalog: catalog)
            if socketPath == nil {
                SploshCLI.writeStderr("listening on http://\(config.host):\(config.port)  (dashboard at /, API at /v1)\n")
            }
            // From here on there are conversations to save on the way out.
            orphaned.action = {
                SploshCLI.writeStderr("\nthe process holding the port has gone; stopping\n")
                kill(getpid(), SIGTERM)
                scheduler.shutdown()
            }
            // A controlled stop. The first signal: the HTTP layer stops taking connections and
            // waits for the requests in flight, which carry on to their ends; then every
            // conversation's context is written to the disk store, so the next server finds
            // each one where it was. A second signal cuts the requests short (their contexts
            // are still saved); a third leaves at once.
            let signals = [SIGINT, SIGTERM].map { number -> DispatchSourceSignal in
                let source = DispatchSource.makeSignalSource(signal: number, queue: .global())
                source.setEventHandler {
                    switch Self.stopping.next() {
                    case nil:
                        break
                    case 1:
                        let running = scheduler.stats().sessions.count
                        SploshCLI.writeStderr("\nstopping: " + (running > 0 ? "letting \(running) request\(running == 1 ? "" : "s") finish (Ctrl+C again to cut them short)\n" : "saving conversations\n"))
                        DispatchQueue.global().asyncAfter(deadline: .now() + config.drainSeconds) {
                            SploshCLI.writeStderr("stopping: \(Int(config.drainSeconds)) s have passed; ending the requests still running\n")
                            scheduler.shutdown()
                        }
                    case 2:
                        SploshCLI.writeStderr("\nstopping: ending the requests still running\n")
                        scheduler.shutdown()
                    default:
                        _exit(130)
                    }
                }
                source.resume()
                return source
            }
            defer { signals.forEach { $0.cancel() } }
            try awaitRun(app)
            let saved = scheduler.shutdownAndSave()
            if scheduler.prefixStore != nil {
                SploshCLI.writeStderr("stopped: \(saved) conversation\(saved == 1 ? "" : "s") written to the prefix store; a restart picks them up from there\n")
            }
            return ExitStatus.ok
        } catch {
            SploshCLI.writeStderr("serve failed: \(error)\n")
            return 1
        }
    }
    /// Print what the model costs in memory, measured, against what the artifact declares.
    private static func report(engine: Engine, weights: ModelWeights, path: String, seconds: Double, device: MTLDevice) {
        let m = engine.memory
        func gib(_ bytes: Int) -> String { String(format: "%7.2f GiB", Double(bytes) / 1_073_741_824) }
        let committed = m.weightBytes + engine.duplicatedWeightBytes + m.stateBytesTotal + m.scratchBytes
        let lines = [
            "model        \(path)  (\(weights.resident.tensorCount) text tensors, loaded in \(String(format: "%.2f", seconds))s)",
            "device       \(device.name)",
            "memory",
            "  weights    \(gib(m.weightBytes))   resident, mapped zero-copy from the artifact" + (engine.duplicatedWeightBytes > 0
                ? "\n  re-tiled   \(gib(engine.duplicatedWeightBytes))   SECOND COPY: this artifact is row-major; run `splosh convert --retile` to drop it" : ""),
            "  state      \(gib(m.stateBytesTotal))   \(engine.config.maxSlots) slots x \(gib(m.stateBytesPerSlot).trimmingCharacters(in: .whitespaces)) recurrent state",
            "  scratch    \(gib(m.scratchBytes))   activations for \(engine.config.maxRows) rows per step",
            "  committed  \(gib(committed))   of \(gib(m.deviceWorkingSetBytes).trimmingCharacters(in: .whitespaces)) working set (\(String(format: "%.1f", Double(committed) / Double(m.deviceWorkingSetBytes) * 100))%)",
            "  kv pool    \(gib(m.kvPoolBytes))   capacity for \(m.kvPagesTotal * Engine.pageTokens) tokens across all sessions, committed as used (\(m.kvBytesPerToken / 1024) KiB/token, \(engine.config.kvFormat.rawValue))",
        ]
        SploshCLI.writeStderr(lines.joined(separator: "\n") + "\n")
    }

    /// The dictionary study a configuration asks for, its corpus read and turned into tokens.
    static func dictionaryStudy(_ config: ServeConfig, tokenizer: Tokenizer) -> DictionaryStudy {
        let corpus = config.dictionaryCorpus.map { studyCorpus(directory: $0, tokenizer: tokenizer) } ?? []
        SploshCLI.writeStderr("dictionary study on: other conversations" + (corpus.isEmpty ? "" : " and a corpus of \(corpus.count) files, \(corpus.reduce(0) { $0 + $1.count }) tokens") + "\n")
        return DictionaryStudy(corpus: corpus)
    }

    /// The source files under a directory as tokens, one array per file, for the dictionary
    /// study: hidden directories and dependencies left out, up to the study's capacity.
    private static func studyCorpus(directory: String, tokenizer: Tokenizer) -> [[Int]] {
        let extensions: Set<String> = ["swift", "py", "ts", "tsx", "js", "jsx", "go", "rs", "c", "h", "cpp", "hpp", "m", "mm", "java", "kt",
                                       "rb", "sh", "sql", "metal", "json", "toml", "yaml", "yml", "md"]
        let root = URL(fileURLWithPath: (directory as NSString).expandingTildeInPath, isDirectory: true)
        guard let files = FileManager.default.enumerator(at: root, includingPropertiesForKeys: nil, options: [.skipsHiddenFiles]) else { return [] }
        var corpus: [[Int]] = [], total = 0
        for case let url as URL in files {
            if ["node_modules", "build", "DerivedData"].contains(url.lastPathComponent) { files.skipDescendants(); continue }
            guard extensions.contains(url.pathExtension.lowercased()), let text = try? String(contentsOf: url, encoding: .utf8),
                  text.utf8.count < 1 << 20 else { continue }
            let tokens = tokenizer.encode(text)
            corpus.append(tokens)
            total += tokens.count
            if total >= DictionaryStudy.capacity { break }
        }
        return corpus
    }

    private static let stopping = StopFlag()


    /// `--echo`: the server's plumbing without the model. It says back the last message, a
    /// word every SPLOSH_ECHO_INTERVAL_MS if that is set.
    ///
    /// A registered model is "loaded" as the real one is, as far as the plumbing can tell: its
    /// artifact has to be there; loading takes SPLOSH_ECHO_LOAD_MS (a time, which `<id>=<ms>`
    /// items after it set for single models: "200,big=5000"); and the model SPLOSH_ECHO_FAIL_MODEL
    /// names does not load.
    private static func serveEcho(config: ServeConfig, socketPath: String?, settings: SettingsService,
                                  catalog: ModelCatalog, model: ModelEntry) throws -> Int32 {
        let environment = ProcessInfo.processInfo.environment
        if config.hasRegistry {
            guard ModelCatalog.fileSize(model.path) != nil else {
                throw CLIError("weight artifact not found at \(model.path), where splosh.toml registers the model \(model.id)")
            }
            if environment["SPLOSH_ECHO_FAIL_MODEL"] == model.id { throw CLIError("the model \(model.id) is set not to load (SPLOSH_ECHO_FAIL_MODEL)") }
            var milliseconds = 0
            for item in (environment["SPLOSH_ECHO_LOAD_MS"] ?? "").split(separator: ",") {
                let parts = item.split(separator: "=", maxSplits: 1)
                if parts.count == 1 { milliseconds = Int(parts[0]) ?? milliseconds } else if parts[0] == model.id, let own = Int(parts[1]) { milliseconds = own; break }
            }
            if milliseconds > 0 { usleep(useconds_t(milliseconds) * 1000) }
        }
        let interval = environment["SPLOSH_ECHO_INTERVAL_MS"].flatMap(Int.init)
        let app = Server.application(service: EchoInferenceService(interval: interval.map { .milliseconds($0) }),
                                     port: config.port, contextWindow: Server.contextLength, socketPath: socketPath, settings: settings,
                                     catalog: catalog)
        // A second signal ends it at once; the first is the HTTP layer's, which finishes replies in flight.
        let signals = [SIGINT, SIGTERM].map { number -> DispatchSourceSignal in
            let source = DispatchSource.makeSignalSource(signal: number, queue: .global())
            source.setEventHandler { if (Self.stopping.next() ?? 0) > 1 { _exit(130) } }
            source.resume()
            return source
        }
        defer { signals.forEach { $0.cancel() } }
        try awaitRun(app)
        return ExitStatus.ok
    }

    private static func awaitRun(_ app: Application<RouterResponder<BasicRequestContext>>) throws {
        let sem = DispatchSemaphore(value: 0)
        let result = LockedResult()
        Task.detached {
            do {
                try await app.runService(gracefulShutdownSignals: [.sigterm, .sigint])
                result.store(.success(()))
            } catch {
                result.store(.failure(error))
            }
            sem.signal()
        }
        sem.wait()
        try result.get()
    }

    private final class LockedResult: @unchecked Sendable {
        private let lock = NSLock()
        private var result: Result<Void, Error>?

        func store(_ result: Result<Void, Error>) {
            lock.lock()
            self.result = result
            lock.unlock()
        }

        func get() throws {
            lock.lock()
            defer { lock.unlock() }
            if let result { try result.get() }
        }
    }
}

/// Puts into effect the settings a running scheduler can take: the ones the settings page
/// marks as applying at once.
final class LiveSettings: @unchecked Sendable {
    private let lock = NSLock()
    private var applied: ServeConfig
    private let scheduler: BatchScheduler
    private let tokenizer: Tokenizer
    private let options: ReplyOptions

    init(scheduler: BatchScheduler, tokenizer: Tokenizer, options: ReplyOptions, applied: ServeConfig) {
        self.scheduler = scheduler; self.tokenizer = tokenizer; self.options = options; self.applied = applied
    }

    func apply(_ config: ServeConfig) {
        lock.lock()
        let previous = applied
        applied = config
        lock.unlock()
        options.continueOnLimit = config.continueOnLimit
        scheduler.update {
            $0.requestLog = config.requestLog
            $0.decodeWeight = config.decodeWeight
            $0.answerReserve = config.answerReserve
            $0.toolCallOverrun = config.toolCallOverrun
            $0.maxActive = config.concurrency > 0 ? config.concurrency : .max
        }
        guard config.dictionaryStudy != previous.dictionaryStudy || config.dictionaryCorpus != previous.dictionaryCorpus else { return }
        guard config.dictionaryStudy else {
            if previous.dictionaryStudy {
                SploshCLI.writeStderr("dictionary study off\n")
                scheduler.update { $0.dictionaryStudy = nil }
            }
            return
        }
        // Reading a corpus takes a while: not on the thread that answers the page.
        DispatchQueue.global(qos: .utility).async { [scheduler, tokenizer] in
            let study = ServeCommand.dictionaryStudy(config, tokenizer: tokenizer)
            scheduler.update { $0.dictionaryStudy = study }
        }
    }
}

/// Counts termination signals.
final class StopFlag: @unchecked Sendable {
    private let lock = NSLock()
    private var count = 0
    private var last = Date.distantPast
    /// Counts a signal and returns how many there have been; nil for one that comes straight
    /// after another, which is the same request arriving twice (`pkill splosh` reaches the
    /// engine and, through the process that holds the port, reaches it again).
    func next() -> Int? {
        lock.lock(); defer { lock.unlock() }
        let now = Date()
        defer { last = now }
        guard now.timeIntervalSince(last) > 0.3 else { return nil }
        count += 1
        return count
    }
}

/// Ends an engine whose holder has gone. Until there is something to save, at once.
final class OrphanWatch: @unchecked Sendable {
    private let lock = NSLock()
    private var source: DispatchSourceProcess?
    private var handler: @Sendable () -> Void = { _exit(1) }

    var action: @Sendable () -> Void {
        get { lock.lock(); defer { lock.unlock() }; return handler }
        set { lock.lock(); handler = newValue; lock.unlock() }
    }

    func start(holder: pid_t) {
        let source = DispatchSource.makeProcessSource(identifier: holder, eventMask: .exit, queue: .global())
        source.setEventHandler { [self] in action() }
        source.resume()
        lock.lock(); self.source = source; lock.unlock()
        if !ServeSupervisor.isRunning(holder) { action() }          // it had gone already
    }
}
