// DownloadCommand.swift — `splosh download`: fetch a model and make it ready to serve.
//
// One command does what the README used to ask for by hand: the tokenizer, the weights from
// Hugging Face (resumed where a stopped download left off, each file checked against its pinned
// SHA-256), the conversion to the artifact the engine maps, the draft model for speculative
// decoding, and the line in splosh.toml that registers the model. Each step says what it is for,
// and a step whose result is already there is passed over: files fetched by hand into the
// directories `--list` names are used as they are.
//
// The server runs this same command as a process of its own when a model is asked for from its
// pages (ServeDownloads), so a conversion never shares a process with the engine. How the
// install stands is written to a file as it goes (DownloadFiles), which is what the pages show.

import Foundation
import SploshModel
import SploshServer

public struct DownloadArguments: Equatable, Sendable {
    public var model: String?
    public var list = false
    public var draft = true
    public var config: String?
    public var help = false

    public init() {}
}

public enum DownloadCommand {
    public static let usage = """
        usage: splosh download [<model>] [--list] [--no-draft] [--config <path>]

        Fetch a model from Hugging Face and make it ready for `splosh serve`: download, check,
        convert, register. With no model named it is the default, mq4 (the MLX 4-bit pack).
        A download that was stopped carries on from where it got to.

          <model>           one of the names `--list` shows: mq4, uq4, uq5, uq6, sq4, sq5, sq6
          --list            the models there are, which are installed, and the directories that
                            take files you have downloaded yourself
          --no-draft        leave out the draft model (speculative decoding needs it)
          --config <path>   default ./splosh.toml; the model is registered in it
        """

    public static func parse(_ tokens: [String]) throws -> DownloadArguments {
        var result = DownloadArguments()
        var stream = TokenStream(tokens)
        while let token = stream.current {
            switch token {
            case "--help", "-h": result.help = true
            case "--list": result.list = true
            case "--no-draft": result.draft = false
            case "--config": result.config = try stream.requireValue(for: token)
            case _ where token.hasPrefix("-"):
                throw CLIError(CLIArguments.unexpected(token, command: "download"))
            default:
                guard result.model == nil else { throw CLIError(CLIArguments.unexpected(token, command: "download")) }
                result.model = try CLIArguments.modelID(token, flag: "download")
            }
            stream.advance()
        }
        return result
    }

    public static func run(_ tokens: [String]) -> Int32 {
        do {
            let args = try parse(tokens)
            let library = try ModelLibrary.current()
            let configPath = args.config ?? "./splosh.toml"
            let config = try ServeConfig.load(path: configPath)
            if args.list {
                print(listing(library, config: config))
                return ExitStatus.ok
            }
            let id = args.model ?? library.defaultModel
            guard let model = library.model(id) else {
                throw CLIError("there is no model called '\(id)' to download; the names are \(library.models.map(\.id).joined(separator: ", ")) (`splosh download --list`)")
            }
            return Installer(library: library, model: model, config: config, configPath: configPath, draft: args.draft).run()
        } catch let error as CLIError {
            SploshCLI.writeStderr("download: \(error.message)\n")
            return ExitStatus.usage
        } catch {
            SploshCLI.writeStderr("download failed: \(error)\n")
            return 1
        }
    }

    // MARK: What there is

    /// The models, one a line: name, what it is, what it costs to fetch and to hold, and whether it is installed.
    static func table(_ library: ModelLibrary, config: ServeConfig) -> String {
        func pad(_ text: String, _ count: Int) -> String { text + String(repeating: " ", count: max(0, count - text.count)) }
        let ids = max(5, library.models.map(\.id.count).max() ?? 0), titles = max(5, library.models.map(\.title.count).max() ?? 0)
        var lines = ["  " + pad("model", ids) + "  " + pad("", titles) + "  " + pad("download", 9) + "  " + pad("in memory", 9) + "  bits a weight"]
        for model in library.models {
            let state = config.isInstalled(model) ? "installed" : model.id == library.defaultModel ? "the default" : ""
            lines.append("  " + pad(model.id, ids) + "  " + pad(model.title, titles) + "  " + pad(Human.bytes(model.source.bytes), 9) + "  "
                         + pad(String(format: "%.1f GiB", model.memoryGiB), 9) + "  " + String(format: "%.2f", model.bitsPerWeight)
                         + (state.isEmpty ? "" : "           " + state))
        }
        return lines.joined(separator: "\n")
    }

    /// Where files fetched by hand go, for each model, as whole paths.
    static func ownFiles(_ library: ModelLibrary) -> String {
        let root = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
        let width = max(5, library.models.map(\.id.count).max() ?? 0)
        var lines: [String] = []
        for model in library.models {
            let directory = root.appendingPathComponent(model.source.directory).standardizedFileURL.path
            let name = model.id + String(repeating: " ", count: width - model.id.count)
            if model.source.files.count == 1 {
                lines.append("  \(name)  \(directory)/\(model.source.files[0].name)")
            } else {
                lines.append("  \(name)  \(directory)/   every file of \(model.source.repo): " + model.source.files.map(\.name).joined(separator: ", "))
            }
        }
        return lines.joined(separator: "\n")
    }

    static func listing(_ library: ModelLibrary, config: ServeConfig) -> String {
        """
        Splosh runs Qwen3.8-27B from one of these. `splosh download <model>` fetches one and makes it ready to serve:

        \(table(library, config: config))

        A smaller file is faster to decode and further from the full model; prefill is much the same for all.

        Files you have downloaded yourself go here, under the names they have on Hugging Face. What is
        already there is checked and not fetched again; files in the Hugging Face cache
        (\(HubCache.directory().path)) are found too:

        \(ownFiles(library))
        """
    }
}

// MARK: - Where an install is written down

/// The files an install leaves for the server's pages: how each model's last install stands, and
/// the lock the one running holds.
enum DownloadFiles {
    static let directory = "models/.downloads"

    static func statusURL(_ id: String) -> URL { URL(fileURLWithPath: directory, isDirectory: true).appendingPathComponent(id + ".json") }
    private static var lockPath: String { directory + "/active.lock" }

    /// Take the lock an install holds for as long as its process lives: one runs at a time, for
    /// they share the tokenizer, the draft model and the disk. Returns nil when another has it.
    static func lock(model: String) -> Int32? {
        try? FileManager.default.createDirectory(atPath: directory, withIntermediateDirectories: true)
        let descriptor = open(lockPath, O_RDWR | O_CREAT, 0o644)
        guard descriptor >= 0 else { return nil }
        // A look by the server holds it for an instant (see `active`).
        var held = false
        for _ in 0..<20 where !held {
            held = flock(descriptor, LOCK_EX | LOCK_NB) == 0
            if !held { usleep(50_000) }
        }
        guard held else { close(descriptor); return nil }
        _ = ftruncate(descriptor, 0)
        let text = Array("\(model) \(getpid())\n".utf8)
        _ = text.withUnsafeBytes { pwrite(descriptor, $0.baseAddress, $0.count, 0) }
        return descriptor
    }

    /// The install that is running: its model and its process; nil when none is.
    static func active() -> (model: String, pid: pid_t)? {
        let descriptor = open(lockPath, O_RDONLY)
        guard descriptor >= 0 else { return nil }
        defer { close(descriptor) }
        if flock(descriptor, LOCK_SH | LOCK_NB) == 0 {
            flock(descriptor, LOCK_UN)
            return nil
        }
        var buffer = [UInt8](repeating: 0, count: 256)
        let count = pread(descriptor, &buffer, buffer.count, 0)
        let parts = String(decoding: buffer[0..<max(0, count)], as: UTF8.self).split(whereSeparator: { $0 == " " || $0 == "\n" })
        guard parts.count >= 2, let pid = pid_t(parts[1]) else { return ("", 0) }
        return (String(parts[0]), pid)
    }

    /// How a model's last install stands, as it wrote it down; nil when it has had none.
    static func status(of id: String) -> JSONValue? {
        (try? Data(contentsOf: statusURL(id))).flatMap { try? JSONValue.parse(Array($0)) }
    }
}

// MARK: - What an install says

/// How an install stands: said in the terminal as it goes, and written down for the pages.
final class InstallReport: @unchecked Sendable {
    struct Step {
        let title: String
        let detail: String
        var state = "waiting"
        var note: String?
        var item: String?
        var verb: String?
        var done: Int?
        var total: Int?
        var rate: Double?
    }

    private let lock = NSLock()
    private let model: String
    private let statusURL: URL?
    /// The terminal is this command's own: progress is one line, redrawn. Otherwise (under the
    /// server, or into a file) it is a line every few seconds.
    private let live: Bool
    private var steps: [Step]
    private var state = "running"
    private var message: String?
    private let started = Date()
    private var samples: [(at: Date, bytes: Int)] = []
    private var lastSaved = Date.distantPast, lastShown = Date.distantPast
    private var lineOpen = false

    init(model: String, steps: [(title: String, detail: String)], statusURL: URL?, live: Bool) {
        self.model = model; self.statusURL = statusURL; self.live = live
        self.steps = steps.map { Step(title: $0.title, detail: $0.detail) }
    }

    var elapsed: TimeInterval { Date().timeIntervalSince(started) }

    /// A line of its own in the terminal.
    func say(_ text: String) {
        lock.lock(); defer { lock.unlock() }
        write(text)
    }

    func begin(_ index: Int) {
        lock.lock(); defer { lock.unlock() }
        steps[index].state = "running"
        let width = steps.map(\.title.count).max() ?? 0
        let title = steps[index].title + String(repeating: " ", count: width - steps[index].title.count)
        write("[\(index + 1)/\(steps.count)] \(title)  \(steps[index].detail)")
        samples = []
        // Redrawn, progress is shown from the first of it; as lines, only for a step that takes a while.
        lastShown = live ? .distantPast : Date()
        save()
    }

    /// Bytes of a step dealt with so far: `verb` is what is being done to `item`.
    func progress(_ index: Int, item: String?, verb: String, done: Int, total: Int) {
        lock.lock(); defer { lock.unlock() }
        let now = Date()
        if steps[index].item != item || steps[index].verb != verb { samples = [] }
        steps[index].item = item; steps[index].verb = verb; steps[index].done = done; steps[index].total = total
        samples.append((now, done))
        samples.removeAll { now.timeIntervalSince($0.at) > 8 }
        var rate: Double?
        if let first = samples.first, now.timeIntervalSince(first.at) >= 1, done > first.bytes {
            rate = Double(done - first.bytes) / now.timeIntervalSince(first.at)
        }
        steps[index].rate = verb == "downloading" ? rate : nil
        if now.timeIntervalSince(lastSaved) >= 0.5 { save() }
        guard now.timeIntervalSince(lastShown) >= (live ? 0.2 : 10) else { return }
        lastShown = now
        var line = "      " + (verb == "downloading" ? "" : verb + " ") + (item.map { $0 + "  " } ?? "")
        line += total > 0 ? String(format: "%.0f%%  ", Double(done) / Double(total) * 100) : ""
        line += "\(Human.bytes(done)) of \(Human.bytes(total))"
        if verb == "downloading", let rate, rate > 0 {
            line += "  \(Human.bytes(Int(rate)))/s  about \(Human.duration(Double(total - done) / rate)) left"
        }
        if live {
            SploshCLI.writeStderr("\r\u{1B}[2K" + line)
            lineOpen = true
        } else {
            SploshCLI.writeStderr(line + "\n")
        }
    }

    func end(_ index: Int, state: String, note: String?) {
        lock.lock(); defer { lock.unlock() }
        steps[index].state = state; steps[index].note = note
        steps[index].item = nil; steps[index].verb = nil; steps[index].rate = nil
        if state == "done", let total = steps[index].total { steps[index].done = total }
        if let note { write("      " + note) } else { clearLine() }
        save()
    }

    /// The install is over: `state` is done, failed or cancelled.
    func finish(_ state: String, message: String) {
        lock.lock(); defer { lock.unlock() }
        self.state = state; self.message = message
        for index in steps.indices where steps[index].state == "running" { steps[index].state = state == "done" ? "done" : state }
        write(message)
        save()
    }

    private func clearLine() {
        if lineOpen { SploshCLI.writeStderr("\r\u{1B}[2K"); lineOpen = false }
    }

    private func write(_ text: String) {
        clearLine()
        SploshCLI.writeStderr(text + "\n")
    }

    private func save() {
        guard let statusURL else { return }
        lastSaved = Date()
        func number(_ value: Int?) -> JSONValue { value.map(JSONValue.int) ?? .null }
        func text(_ value: String?) -> JSONValue { value.map(JSONValue.string) ?? .null }
        let status = JSONValue.object([
            ("model", .string(model)), ("pid", .int(Int(getpid()))), ("state", .string(state)),
            ("started", .double(started.timeIntervalSince1970)), ("updated", .double(Date().timeIntervalSince1970)),
            ("message", text(message)),
            ("steps", .array(steps.map { step in
                .object([("title", .string(step.title)), ("detail", .string(step.detail)), ("state", .string(step.state)), ("note", text(step.note)),
                         ("item", text(step.item)), ("verb", text(step.verb)), ("done", number(step.done)), ("total", number(step.total)),
                         ("rate", step.rate.map { JSONValue.int(Int($0)) } ?? .null)])
            })),
        ])
        try? Data(status.compact.utf8).write(to: statusURL, options: .atomic)
    }
}

// MARK: - The install

/// One model, from nothing (or from whatever of it is already here) to an artifact the server
/// can load by name.
final class Installer: @unchecked Sendable {
    private let library: ModelLibrary
    private let model: ModelLibrary.Model
    private let config: ServeConfig
    private let configPath: String
    private let wantDraft: Bool
    private let environment = ProcessInfo.processInfo.environment
    private let cancellation = Cancellation()
    private let hub: HubDownload
    private let manager = FileManager.default
    private var report: InstallReport!

    /// How a file came to be where it is wanted.
    private enum Got { case present, kept(String), copied, cached, downloaded }
    private typealias Outcome = (state: String, note: String?)
    private struct Step {
        let title: String
        let detail: String
        let run: (Int) throws -> Outcome
    }

    init(library: ModelLibrary, model: ModelLibrary.Model, config: ServeConfig, configPath: String, draft: Bool) {
        self.library = library; self.model = model; self.config = config; self.configPath = configPath; wantDraft = draft
        hub = HubDownload(endpoint: ModelLibrary.endpoint())
    }

    private var artifact: String { config.artifactPath(of: model) }
    private var rowMajor: String { model.rowMajorPath(tiled: artifact) }
    private var tokenizerDirectory: URL { URL(fileURLWithPath: config.tokenizerPath, isDirectory: true) }
    private var sourceDirectory: URL { URL(fileURLWithPath: model.source.directory, isDirectory: true) }
    private var draftDirectory: URL { URL(fileURLWithPath: library.draft.directory, isDirectory: true) }
    private func exists(_ path: String) -> Bool { ModelCatalog.fileSize(path) != nil }
    /// The process that asked for this install (the server), when one did.
    private var parent: pid_t? { environment[ServeDownloads.parentVariable].flatMap { pid_t($0) } }

    func run() -> Int32 {
        guard let lock = DownloadFiles.lock(model: model.id) else {
            let other = DownloadFiles.active()
            SploshCLI.writeStderr("download: \(other.map { "\($0.model) is being installed by process \($0.pid)" } ?? "another install is running"); one runs at a time\n")
            return 1
        }
        defer { close(lock) }

        let steps = plan()
        report = InstallReport(model: model.id, steps: steps.map { ($0.title, $0.detail) }, statusURL: DownloadFiles.statusURL(model.id),
                               live: parent == nil && isatty(STDERR_FILENO) != 0)
        watchForStop()

        report.say("\n\(model.id): \(model.title) of Qwen3.8-27B (\(model.source.repo)). \(model.summary)")
        do {
            try checkDisk()
            for (index, step) in steps.enumerated() {
                try cancellation.check()
                report.begin(index)
                do {
                    let outcome = try step.run(index)
                    report.end(index, state: outcome.state, note: outcome.note)
                } catch let error as DownloadError where !error.cancelled && step.title == Self.draftTitle {
                    // The model serves without it, a token a step.
                    report.end(index, state: "failed", note: "not fetched: \(error.message)")
                    report.finish("done", message: "\(model.id) is installed (\(artifact)), without the draft model: the server writes one token a step until `splosh download \(model.id)` has fetched it.")
                    return ExitStatus.ok
                }
            }
        } catch let error as DownloadError where error.cancelled {
            report.finish("cancelled", message: "Stopped. What has been downloaded is kept: `splosh download \(model.id)` carries on from there.")
            return 130
        } catch {
            report.finish("failed", message: "\(model.id) was not installed: \(error)")
            return 1
        }
        report.finish("done", message: "Done in \(Human.duration(report.elapsed)). " + whereItIs() + "\n" + (parent == nil
            ? "`splosh serve` starts the server; one that is running lists the model already (`splosh models`, or its models page)."
            : "The server lists it now."))
        return ExitStatus.ok
    }

    /// Where the installed model is, what of the install the server does not read, and what
    /// would delete the model: said when an install ends.
    private func whereItIs() -> String {
        let path = URL(fileURLWithPath: artifact).standardizedFileURL.path
        var lines = ["\(model.id) is installed: \(path)" + (OnDisk.size(artifact).map { " (\(Human.bytes($0)))" } ?? "")]
        // Files that are here themselves, not links to the Hugging Face cache.
        let downloaded = model.source.files.filter { file in
            let path = sourceDirectory.appendingPathComponent(file.name).path
            return (try? manager.destinationOfSymbolicLink(atPath: path)) == nil && exists(path)
        }
        if !downloaded.isEmpty {
            let what = downloaded.count == 1 ? "\(model.source.directory)/\(downloaded[0].name)" : "the files in \(model.source.directory)"
            lines.append("That file is all the server reads of it: what it was converted from, \(what) (\(Human.bytes(downloaded.reduce(0) { $0 + $1.bytes }))), can be deleted.")
        }
        if artifact.hasPrefix(".build/") {
            lines.append("It is kept in .build with the build's own files: `swift package clean`, `swift package reset` and `rm -rf .build` delete it with them; `swift build` does not.")
        }
        return lines.joined(separator: "\n")
    }

    /// A stop: Ctrl+C, the server's Cancel, or the process that asked for the install going away.
    private func watchForStop() {
        for number in [SIGINT, SIGTERM, SIGHUP] {
            signal(number, SIG_IGN)
            let source = DispatchSource.makeSignalSource(signal: number, queue: .global())
            source.setEventHandler { [self] in
                if cancellation.isCancelled { _exit(130) }       // asked twice
                cancellation.cancel()
            }
            source.resume()
            Self.sources.append(source)
        }
        if let parent {
            // It writes to the server's terminal from the background.
            signal(SIGTTOU, SIG_IGN)
            let source = DispatchSource.makeProcessSource(identifier: parent, eventMask: .exit, queue: .global())
            source.setEventHandler { [self] in cancellation.cancel() }
            source.resume()
            Self.sources.append(source)
            if !ServeSupervisor.isRunning(parent) { cancellation.cancel() }
        }
    }
    nonisolated(unsafe) private static var sources: [any DispatchSourceProtocol] = []

    // MARK: The steps

    private static let draftTitle = "Draft model"

    private func plan() -> [Step] {
        let tokenizer = library.tokenizer, source = model.source
        var steps = [
            Step(title: "Tokenizer",
                 detail: "what turns text into tokens and back, and the chat template: \(tokenizer.files.count) files, \(Human.bytes(tokenizer.bytes)), into \(config.tokenizerPath)") { [self] index in
                try fetch(tokenizer.files, of: tokenizer, into: tokenizerDirectory, step: index)
            },
            Step(title: "Weights",
                 detail: "the model itself: \(source.files.count == 1 ? source.files[0].name : "\(source.files.count) files"), \(Human.bytes(source.bytes)), from \(source.repo) into \(source.directory)") { [self] index in
                if exists(artifact) { return ("skipped", "not needed: the converted artifact is already at \(artifact)") }
                if model.kind == .mlx, exists(rowMajor) { return ("skipped", "not needed: the converted pack is already at \(rowMajor)") }
                // Only converted from, so a copy in the Hugging Face cache is used where it is.
                return try fetch(source.files, of: source, into: sourceDirectory, also: [tokenizerDirectory], linked: true, step: index)
            },
        ]
        switch model.kind {
        case .mlx:
            steps.append(Step(title: "Convert", detail: "the pack's tensors written out in Splosh's format, and compared with the pack: \(rowMajor)") { [self] index in
                if exists(artifact) || exists(rowMajor) { return ("skipped", "already done") }
                try convert(to: rowMajor, from: largestSource, bytes: model.artifactBytes, step: index) { output in
                    _ = try Converter.alignmentEvidence(from: URL(fileURLWithPath: manager.currentDirectoryPath))
                    try Converter.requireAssets(at: sourceDirectory)
                    _ = try Converter.convert(inputRoot: sourceDirectory, outputURL: output, verify: true)
                }
                createdRowMajor = true
                return ("done", nil)
            })
            steps.append(Step(title: "Tile", detail: "the weights laid out in the 128-row tiles the engine reads, so the file is mapped once and not copied: \(artifact)") { [self] index in
                if exists(artifact) { return ("skipped", "already there") }
                try convert(to: artifact, from: URL(fileURLWithPath: rowMajor), bytes: model.artifactBytes, step: index) { output in
                    _ = try Converter.retile(inputURL: URL(fileURLWithPath: rowMajor), outputURL: output)
                }
                // The row-major artifact was only the way here.
                if createdRowMajor { try? manager.removeItem(atPath: rowMajor) }
                return ("done", createdRowMajor ? "\(rowMajor) removed: it was only the step before" : nil)
            })
        case .gguf:
            steps.append(Step(title: "Convert", detail: "the file's codes re-ordered into the layout the kernels read, at the size they are: \(artifact)") { [self] index in
                if exists(artifact) { return ("skipped", "already there") }
                let input = sourceDirectory.appendingPathComponent(source.files[0].name)
                try convert(to: artifact, from: input, bytes: model.artifactBytes, step: index) { output in
                    _ = try Converter.convertGguf(inputURL: input, outputURL: output, tokenizerURL: tokenizerDirectory.appendingPathComponent("tokenizer.json"))
                }
                return ("done", nil)
            })
        }
        steps.append(Step(title: "Register", detail: "a line in \(configPath) names the model, so the server and its clients can ask for it as \(model.id)") { [self] _ in
            let said = try register()
            return ("done", said)
        })
        let draft = library.draft
        steps.append(Step(title: Self.draftTitle,
                          detail: "DFlash 2, which guesses several tokens ahead so that a step can write more than one: \(Human.bytes(draft.bytes)), from \(draft.repo)") { [self] index in
            if !wantDraft { return ("skipped", "left out (--no-draft): the server writes one token a step without it") }
            if config.draftPath == "none" { return ("skipped", "splosh.toml switches speculative decoding off (draftPath = \"none\")") }
            if let own = config.draftPath { return ("skipped", "splosh.toml names its own: \(own)") }
            if let found = GenerateCommand.defaultDraftURL() { return ("skipped", "already there: \(found.path)") }
            return try fetch(draft.files, of: draft, into: draftDirectory, step: index)
        })
        return steps
    }
    private var createdRowMajor = false

    /// Refuse before anything is fetched if the install could not be finished: the volume
    /// cannot take it, or the conversion could not be run here.
    private func checkDisk() throws {
        // The converter of an MLX pack takes its alignment rule from a record in the repository.
        if model.kind == .mlx, !exists(artifact), !exists(rowMajor), environment["SPLOSH_ECHO_CONVERT"] == nil,
           !manager.fileExists(atPath: Converter.alignmentEvidencePath) {
            throw DownloadError("an MLX pack is converted in Splosh's own directory, where \(Converter.alignmentEvidencePath) is: run `splosh download \(model.id)` there")
        }
        func missing(_ files: [ModelLibrary.File], of source: ModelLibrary.Source, in directory: URL) -> Int {
            files.reduce(0) { sum, file in
                let destination = directory.appendingPathComponent(file.name).path
                if exists(destination) || HubCache.find(file.name, bytes: file.bytes, repo: source.repo) != nil { return sum }
                return sum + max(0, file.bytes - (ModelCatalog.fileSize(destination + ".part") ?? 0))
            }
        }
        var download = missing(library.tokenizer.files, of: library.tokenizer, in: tokenizerDirectory), written = 0
        if !exists(artifact) {
            let converted = model.kind == .mlx && exists(rowMajor)
            if !converted { download += missing(model.source.files, of: model.source, in: sourceDirectory) }
            written = converted ? model.artifactBytes : model.conversionBytes
        }
        if wantDraft, config.draftPath == nil, GenerateCommand.defaultDraftURL() == nil {
            download += missing(library.draft.files, of: library.draft, in: draftDirectory)
        }
        let here = URL(fileURLWithPath: manager.currentDirectoryPath)
        let free = (try? here.resourceValues(forKeys: [.volumeAvailableCapacityForImportantUsageKey]))?.volumeAvailableCapacityForImportantUsage.map(Int.init)
        var line = "To do: \(Human.bytes(download)) to download"
        if written > 0 { line += ", \(Human.bytes(written)) written in converting" + (model.kind == .mlx && written > model.artifactBytes ? " (half of it kept)" : "") }
        if let free { line += "; \(Human.bytes(free)) of disk free" }
        report.say(line + ".\n")
        if let free, download + written > free {
            throw DownloadError("it needs \(Human.bytes(download + written)) of disk and \(Human.bytes(free)) is free on this volume")
        }
    }

    /// Have every file of `files` in `directory`. `also` are directories a copy may already be
    /// in; `linked` has a file found in the Hugging Face cache used from there, by a link.
    private func fetch(_ files: [ModelLibrary.File], of source: ModelLibrary.Source, into directory: URL, also: [URL] = [], linked: Bool = false,
                       step: Int) throws -> Outcome {
        let total = files.reduce(0) { $0 + $1.bytes }
        var before = 0, got: [Got] = []
        for file in files {
            try cancellation.check()
            let base = before
            got.append(try ensure(file, of: source, in: directory, also: also, linked: linked) { [report] verb, bytes in
                report?.progress(step, item: file.name, verb: verb, done: base + bytes, total: total)
            })
            before += file.bytes
        }
        var parts: [String] = []
        func count(_ what: String, _ matches: (Got) -> Bool) {
            let n = got.filter(matches).count
            if n > 0 { parts.append(files.count == 1 ? what : "\(n) \(what)") }
        }
        count("downloaded and checked against the pinned SHA-256") { if case .downloaded = $0 { return true }; return false }
        count("already there, and the pinned file") { if case .present = $0 { return true }; return false }
        count("found in the Hugging Face cache, and the pinned file") { if case .cached = $0 { return true }; return false }
        count("copied from \(also.first?.relativePath ?? "beside it")") { if case .copied = $0 { return true }; return false }
        for case .kept(let why) in got { parts.append("kept as found, though \(why)") }
        return ("done", parts.joined(separator: "; "))
    }

    /// Have `file` in `directory`: as it is found there, from another directory, from the
    /// Hugging Face cache, or downloaded. A file found in `directory` is the user's own and is
    /// kept even when it is not the pinned one; a copy from anywhere else must be.
    ///
    /// A file in the cache is copied, since the server is to have it whatever becomes of the
    /// cache; with `linked` (a source, read once by a conversion) it is linked to instead.
    private func ensure(_ file: ModelLibrary.File, of source: ModelLibrary.Source, in directory: URL, also: [URL], linked: Bool,
                        progress: @escaping @Sendable (String, Int) -> Void) throws -> Got {
        let destination = directory.appendingPathComponent(file.name)
        try manager.createDirectory(at: directory, withIntermediateDirectories: true)
        func pinned(_ url: URL) throws -> Bool {
            try HubDownload.sha256(of: url, cancellation: cancellation) { progress("checking", $0) } == file.sha256
        }
        if let size = OnDisk.size(destination.path) {
            guard size == file.bytes else { return .kept("\(file.name) is \(size) bytes where the pinned file has \(file.bytes)") }
            return try pinned(destination) ? .present : .kept("\(file.name) is not the pinned file (its SHA-256 differs)")
        }
        // A link left pointing at nothing.
        if (try? manager.destinationOfSymbolicLink(atPath: destination.path)) != nil { try manager.removeItem(at: destination) }
        // A copy arrives under another name and takes the file's own when it is whole, so that a
        // copy cut short is never taken for the file.
        func copy(_ original: URL) throws {
            let arriving = URL(fileURLWithPath: destination.path + ".copy")
            try? manager.removeItem(at: arriving)
            try manager.copyItem(at: original.resolvingSymlinksInPath(), to: arriving)
            try manager.moveItem(at: arriving, to: destination)
        }
        for other in also.map({ $0.appendingPathComponent(file.name) }) where OnDisk.size(other.path) == file.bytes {
            if try pinned(other) {
                try copy(other)
                return .copied
            }
        }
        if let cached = HubCache.find(file.name, bytes: file.bytes, repo: source.repo), try pinned(cached) {
            if linked { try manager.createSymbolicLink(at: destination, withDestinationURL: cached.resolvingSymlinksInPath()) }
            else { try copy(cached) }
            return .cached
        }
        try hub.fetch(file, from: source, to: destination, cancellation: cancellation,
                      progress: { progress("downloading", $0) }, event: { [report] in report?.say("      " + $0) })
        return .downloaded
    }

    /// The largest file of the source: the weights, or the first of them.
    private var largestSource: URL {
        sourceDirectory.appendingPathComponent(model.source.files.max { $0.bytes < $1.bytes }?.name ?? "")
    }

    /// Run a conversion of `input` that writes `path`, showing how much of it is written. A
    /// conversion cannot be asked to stop, so a stop ends the process, with what it had written
    /// removed.
    ///
    /// With SPLOSH_ECHO_CONVERT set, for tests of the plumbing with files that are not models,
    /// the output is `input` itself after that many milliseconds, and the model
    /// SPLOSH_ECHO_CONVERT_FAIL names does not convert.
    private func convert(to path: String, from input: URL, bytes: Int, step: Int, _ work: (URL) throws -> Void) throws {
        let output = URL(fileURLWithPath: path)
        try manager.createDirectory(at: output.deletingLastPathComponent(), withIntermediateDirectories: true)
        // The converters write to a hidden file beside the output and rename it when it is whole.
        let directory = output.deletingLastPathComponent().path, prefix = ".\(output.lastPathComponent).tmp-"
        let partial: @Sendable () -> [String] = {
            ((try? FileManager.default.contentsOfDirectory(atPath: directory)) ?? []).filter { $0.hasPrefix(prefix) }.map { directory + "/" + $0 }
        }
        partial().forEach { try? manager.removeItem(atPath: $0) }               // left by a conversion that was killed
        let watch = DispatchSource.makeTimerSource(queue: .global())
        watch.schedule(deadline: .now() + 0.5, repeating: 0.5)
        watch.setEventHandler { [report] in
            guard let written = partial().compactMap({ ModelCatalog.fileSize($0) }).max() else { return }
            report?.progress(step, item: nil, verb: written >= bytes ? "checking" : "writing", done: min(written, bytes), total: bytes)
        }
        watch.resume()
        defer { watch.cancel() }
        cancellation.onCancel { [report, id = model.id] in
            partial().forEach { try? FileManager.default.removeItem(atPath: $0) }
            report?.finish("cancelled", message: "Stopped during the conversion, which begins again when `splosh download \(id)` is next run; what was downloaded is kept.")
            _exit(130)
        }
        defer { cancellation.onCancel(nil) }
        try cancellation.check()
        do {
            if let pause = environment["SPLOSH_ECHO_CONVERT"].flatMap(Int.init) {
                usleep(useconds_t(pause) * 1000)
                guard exists(input.path) else { throw DownloadError("there is nothing at \(input.path) to convert") }
                if environment["SPLOSH_ECHO_CONVERT_FAIL"] == model.id { throw DownloadError("\(model.id) is set not to convert (SPLOSH_ECHO_CONVERT_FAIL)") }
                try manager.copyItem(at: input.resolvingSymlinksInPath(), to: output)
            } else {
                try work(output)
            }
        } catch let error as DownloadError {
            throw error
        } catch {
            throw DownloadError("the conversion to \(path) failed: \(error)")
        }
    }

    /// Name the model in splosh.toml, and have the server start on it if the model it would
    /// start on is not there. Returns what was done, for the step's note.
    private func register() throws -> String {
        let text = (try? String(contentsOfFile: configPath, encoding: .utf8)) ?? ""
        let current = try ServeConfig.parse(text)
        var changes: [(key: String, value: String?)] = []
        if !current.hasRegistry {
            // The first registered model replaces the one model there was without a registry:
            // that one, and any other of the library's already here, are registered with it.
            // (The one model: at `weightsPath`, or a pack converted by hand and never tiled.)
            let path = current.weightsPath ?? ServeConfig.defaultWeightsPath
            if exists(path), ModelEntry.isValid(id: current.modelID), library.model(current.modelID) == nil, !library.models.contains(where: { $0.artifact == path }) {
                changes.append((ServeConfig.modelKeyPrefix + current.modelID, path))
            }
            for other in library.models where other.id != model.id && exists(other.artifact) {
                changes.append((ServeConfig.modelKeyPrefix + other.id, other.artifact))
            }
        }
        if !current.models.contains(where: { $0.id == model.id }) { changes.append((ServeConfig.modelKeyPrefix + model.id, artifact)) }
        do {
            let after = try ServeSettings.change(text, changes).config
            var said = changes.isEmpty ? ["already registered"] : ["registered"]
            let start = after.registry.first { $0.id == after.model } ?? after.registry[0]
            if start.id != model.id, !exists(start.path) {
                changes.append(("model", model.id))
                said.append("and made the model the server starts on, since \(start.id) is not installed")
            }
            guard !changes.isEmpty else { return said[0] }
            try Data(try ServeSettings.change(text, changes).text.utf8).write(to: URL(fileURLWithPath: configPath), options: .atomic)
            return said.joined(separator: ", ")
        } catch let error as SettingsError {
            throw DownloadError("\(configPath) could not be changed: \(error.message)")
        }
    }
}
