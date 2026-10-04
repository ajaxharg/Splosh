// ServeDownloads.swift — the server's side of `splosh download`.
//
// The first-launch page and the models page list the library's models and start an install.
// The install is `splosh download <model>` as a process of its own: a conversion writes tens of
// gigabytes and must not share a process with the engine, and an install started from a page
// goes on while the engine is replaced (a restart, another model loaded). It writes how it
// stands to a file (DownloadFiles), which is all this reads; so an install started from a
// terminal is shown by the pages too.
//
// An install ends with the process that holds the port: a download is not left running behind
// a server that has been stopped. What it had fetched is kept, and carried on from.

import Foundation
import SploshServer

enum ServeDownloads {
    /// Set for an install the server starts: the process whose going ends it.
    static let parentVariable = "SPLOSH_DOWNLOAD_PARENT"

    /// `loaded` is the model the engine has, and `known` the ones it can load (what splosh.toml
    /// registered when it started); both nil for a server with no model.
    static func service(library: ModelLibrary, configPath: String, loaded: ModelEntry?, known: [String]?, parent: pid_t) -> DownloadService {
        let read: @Sendable () -> JSONValue = { snapshot(library: library, configPath: configPath, loaded: loaded, known: known) }
        return DownloadService(
            read: read,
            start: { id in
                guard library.model(id) != nil else {
                    throw DownloadRefusal(status: 404, message: "there is no model called \(id) to download")
                }
                if let active = DownloadFiles.active() {
                    throw DownloadRefusal(status: 409, message: active.model == id ? "\(id) is already being installed" : "\(active.model) is being installed, and one install runs at a time")
                }
                try spawn(id, configPath: configPath, parent: parent)
                return read()
            },
            cancel: { id in
                guard let active = DownloadFiles.active(), active.model == id, active.pid > 0 else {
                    throw DownloadRefusal(status: 409, message: "\(id) is not being installed")
                }
                kill(active.pid, SIGTERM)
                // It says where it stopped, and lets go of the lock, before this answers.
                for _ in 0..<60 where DownloadFiles.active()?.pid == active.pid { usleep(50_000) }
                return read()
            })
    }

    /// Start `splosh download <id>`, and return once it holds the install lock or has ended.
    private static func spawn(_ id: String, configPath: String, parent: pid_t) throws {
        var size: UInt32 = 0
        _NSGetExecutablePath(nil, &size)
        var buffer = [CChar](repeating: 0, count: Int(size) + 1)
        _NSGetExecutablePath(&buffer, &size)
        let process = Process()
        process.executableURL = URL(fileURLWithPath: String(cString: buffer))
        process.arguments = ["download", id, "--config", configPath]
        var environment = ProcessInfo.processInfo.environment
        environment[parentVariable] = "\(parent)"
        process.environment = environment
        // It reports to the server's terminal; it is not to read from it.
        process.standardInput = FileHandle.nullDevice
        let asked = Date().timeIntervalSince1970
        do {
            try process.run()
        } catch {
            throw DownloadRefusal(status: 500, message: "the install could not be started: \(error.localizedDescription)")
        }
        running.keep(process)
        for _ in 0..<60 where process.isRunning && DownloadFiles.active() == nil { usleep(50_000) }
        // One that ended at once without writing down why (splosh.toml it could not read, say)
        // has said so only in the server's terminal.
        if !process.isRunning, process.terminationStatus != 0, (DownloadFiles.status(of: id)?["updated"]?.doubleValue ?? 0) < asked {
            throw DownloadRefusal(status: 500, message: "the install of \(id) ended as it started (status \(process.terminationStatus)); the server's terminal says why")
        }
    }

    /// The installs this engine started, kept until they end so that each is collected.
    private final class Running: @unchecked Sendable {
        private let lock = NSLock()
        private var processes: [Process] = []
        func keep(_ process: Process) {
            lock.lock()
            processes.removeAll { !$0.isRunning }
            processes.append(process)
            lock.unlock()
        }
    }
    private static let running = Running()

    /// What the pages show: every model of the library, what of it is here, and its install.
    static func snapshot(library: ModelLibrary, configPath: String, loaded: ModelEntry?, known: [String]?) -> JSONValue {
        let config = (try? ServeConfig.load(path: configPath)) ?? ServeConfig()
        let active = DownloadFiles.active()
        let root = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
        let models = library.models.map { model -> JSONValue in
            let directory = root.appendingPathComponent(model.source.directory).standardizedFileURL
            let installed = config.isInstalled(model)
            // Without a registry the one model is served under another name, from this artifact.
            let isLoaded = loaded.map { $0.id == model.id || $0.path == config.artifactPath(of: model) } ?? false
            // The bytes of the source that need not be downloaded: whole files, the part of one
            // that has arrived, and files in the Hugging Face cache.
            let here = model.source.files.reduce(0) { sum, file in
                let path = directory.appendingPathComponent(file.name).path
                if let size = OnDisk.size(path) { return sum + min(size, file.bytes) }
                if HubCache.find(file.name, bytes: file.bytes, repo: model.source.repo) != nil { return sum + file.bytes }
                return sum + min(ModelCatalog.fileSize(path + ".part") ?? 0, file.bytes)
            }
            var install = DownloadFiles.status(of: model.id)
            if let status = install, status["state"]?.stringValue == "running", active?.model != model.id {
                // Its process went away without a word.
                install = .object((status.objectValue ?? []).map { member in
                    switch member.key {
                    case "state": return (member.key, JSONValue.string("cancelled"))
                    case "message": return (member.key, JSONValue.string("The install stopped before it had finished. What was downloaded is kept, and it carries on from there."))
                    default: return member
                    }
                })
            }
            return .object([
                ("id", .string(model.id)), ("title", .string(model.title)), ("summary", .string(model.summary)),
                ("repo", .string(model.source.repo)), ("bitsPerWeight", .double(model.bitsPerWeight)), ("memoryGiB", .double(model.memoryGiB)),
                ("downloadBytes", .int(model.source.bytes)), ("hereBytes", .int(installed ? model.source.bytes : here)),
                // Where the converted model is, or will be: as registered, and as a whole path.
                ("artifact", .string(config.artifactPath(of: model))),
                ("artifactPath", .string(URL(fileURLWithPath: (config.artifactPath(of: model) as NSString).expandingTildeInPath, relativeTo: root).standardizedFileURL.path)),
                ("artifactBytes", OnDisk.size(config.artifactPath(of: model)).map(JSONValue.int) ?? .null),
                ("installed", .bool(installed)),
                ("loaded", .bool(isLoaded)),
                // Installed since the engine started, under a name it did not know then.
                ("needsRestart", .bool(installed && !isLoaded && config.models.contains { $0.id == model.id } && known.map { !$0.contains(model.id) } ?? false)),
                ("place", .string(directory.path + "/" + (model.source.files.count == 1 ? model.source.files[0].name : ""))),
                ("files", .array(model.source.files.map { .string($0.name) })),
                ("install", install ?? .null),
            ])
        }
        let free = (try? root.resourceValues(forKeys: [.volumeAvailableCapacityForImportantUsageKey]))?.volumeAvailableCapacityForImportantUsage
        return .object([
            ("setup", .bool(loaded == nil)),
            ("default", .string(library.defaultModel)),
            ("active", active.map { .string($0.model) } ?? .null),
            ("loaded", loaded.map { .string($0.id) } ?? .null),
            ("freeBytes", free.map { .int(Int($0)) } ?? .null),
            ("cache", .string(HubCache.directory().path)),
            ("draft", .object([
                ("wanted", .bool(config.draftPath == nil)), ("present", .bool(GenerateCommand.defaultDraftURL() != nil)),
                ("bytes", .int(library.draft.bytes)),
            ])),
            ("models", .array(models)),
        ])
    }

    // MARK: A server with no model

    /// What a server that has no model says in its terminal: how to get one.
    static func welcome(library: ModelLibrary, config: ServeConfig, address: String, prompt: Bool) -> String {
        let fallback = library.model(library.defaultModel) ?? library.models[0]
        var ways = ["  in a browser      open \(address) and pick from the list; it shows each step"]
        if prompt {
            ways.append("  in this terminal  press Enter for the default (\(fallback.id): \(fallback.title), \(Human.bytes(fallback.source.bytes))),\n"
                        + "                    or type another's name and press Enter")
        }
        ways.append("  from a shell      splosh download <model>     (in this directory)")
        return """

            No model is installed yet, so the server is starting without one. To get one:

            \(ways.joined(separator: "\n"))

            \(DownloadCommand.table(library, config: config))

            They are the same model at different precisions: a smaller file decodes faster and is further
            from the full model. Splosh downloads the one you choose, checks it, converts it, and loads it.

            Downloaded the files yourself? Put them here, under the names they have on Hugging Face, and
            choose the model as above: what is already there is checked and not fetched again. Files in the
            Hugging Face cache (\(HubCache.directory().path)) are found too.

            \(DownloadCommand.ownFiles(library))


            """
    }

    /// Start the engine when a model has been installed: asked of the process that holds the
    /// port once no install is running and some registered model has its artifact. Without that
    /// process the server can only say so.
    static func startWhenInstalled(library: ModelLibrary, configPath: String, restart: (@Sendable () -> Void)?) {
        Thread.detachNewThread {
            while true {
                usleep(500_000)
                guard DownloadFiles.active() == nil, let config = try? ServeConfig.load(path: configPath),
                      let start = try? config.installedStart(exists: { ModelCatalog.fileSize($0) != nil }),
                      ModelCatalog.fileSize(start.model.path) != nil,
                      // A pack between its two conversions is not a model installed: an install
                      // stopped there (or one that failed) leaves the server waiting, as asked.
                      !library.models.contains(where: { $0.kind == .mlx && !config.isInstalled($0)
                                                        && $0.rowMajorPath(tiled: config.artifactPath(of: $0)) == start.model.path }) else { continue }
                if let restart {
                    SploshCLI.writeStderr("\na model is installed: starting the engine on it\n")
                    restart()
                } else {
                    SploshCLI.writeStderr("\na model is installed: stop this server and start it again to load it\n")
                }
                return
            }
        }
    }
}

/// The terminal's side of a server with no model: Enter downloads the default, a model's name
/// that one. It asks the server as its pages do, so the install is the same one the pages show.
enum SetupPrompt {
    /// Whether what is typed in the terminal reaches the process `leader` (the one that holds
    /// the port): its input is a terminal, and it is that terminal's foreground job. A server
    /// started in the background, or with its input from elsewhere, is not asked anything.
    static func interactive(leader: pid_t) -> Bool {
        isatty(STDIN_FILENO) != 0 && tcgetpgrp(STDIN_FILENO) == getpgid(leader)
    }

    static func start(library: ModelLibrary, config: ServeConfig) {
        // Put in the background later, a read from the terminal fails and is not a stop.
        signal(SIGTTIN, SIG_IGN)
        Thread.detachNewThread {
            while let line = readLine() {
                guard let state = ask("GET", "/v1/downloads", config: config), let setup = state["setup"]?.boolValue else {
                    SploshCLI.writeStderr("the server did not answer; try again in a moment\n")
                    continue
                }
                // Only while the server is waiting for a model: a stray Enter later starts nothing.
                guard setup else { return }
                if let active = state["active"]?.stringValue {
                    SploshCLI.writeStderr("\(active) is being installed; its steps are shown here as they happen\n")
                    continue
                }
                let typed = line.trimmingCharacters(in: .whitespaces)
                let id = typed.isEmpty ? library.defaultModel : typed
                guard library.model(id) != nil else {
                    SploshCLI.writeStderr("there is no model called '\(typed)': the names are \(library.models.map(\.id).joined(separator: ", ")); Enter alone is \(library.defaultModel)\n")
                    continue
                }
                let body = JSONValue.object([("model", .string(id))]).compact
                guard let answer = ask("POST", "/v1/downloads", config: config, body: Data(body.utf8)) else {
                    SploshCLI.writeStderr("the server did not answer; try again\n")
                    continue
                }
                if let refused = answer["error"]?.stringValue {
                    SploshCLI.writeStderr("\(id) was not started: \(refused)\n")
                    continue
                }
                return
            }
        }
    }

    private static func ask(_ method: String, _ path: String, config: ServeConfig, body: Data? = nil) -> JSONValue? {
        ModelsCommand.request(method, path, config: config, body: body, timeout: 30).flatMap { try? JSONValue.parse(Array($0.body)) }
    }
}
