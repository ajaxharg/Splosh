// ServeSupervisor.swift — the process that holds the port.
//
// `splosh serve` is two processes. This one binds the address clients use and never loads the
// model. It starts the engine as a child, which serves HTTP on a private Unix-domain socket,
// and passes each connection's bytes to and fro without looking at them.
//
// The point is that the port outlives the engine. Asked to restart (`splosh serve --restart`,
// which sends SIGUSR1), it tells the engine to stop the controlled way: requests in flight
// carry on for a while, then are cut (their clients send them again), and every conversation's
// context is written to disk. Meanwhile it goes on accepting connections and keeps them,
// request and all, until a new engine (a new build, new settings) is up, and then hands them
// over. A client sees a reply that took longer, not a refused connection; the new engine finds
// each conversation on disk.
//
// A stop (Ctrl+C, SIGTERM, the terminal closing) is passed on to the engine and ends both.
// An engine that dies on its own is started again.
//
// The engine counts the signals it is sent: the first lets requests finish, the second cuts
// them short, the third leaves at once. This process sends each on purpose, once.
//
// A model is loaded in another's place the same way, since an engine holds one model and two
// do not fit: the engine is replaced by one started with the other. The one thing this process
// reads of what passes through it is the start of an engine's answer, for the marker of a
// request that names a registered model the engine does not have (see ModelSwitch). Such a
// request is kept here until the loaded model has had its turn and its engine has nothing in
// flight; then the engine is stopped, with nothing to cut, and the request given to the next.

import Foundation

enum ServeSupervisor {
    /// Set for the engine process: the socket it is to serve on, and this process's id.
    static let socketVariable = "SPLOSH_ENGINE_SOCKET"
    static let holderVariable = "SPLOSH_HOLDER_PID"
    /// Also set for it: the registered model it is to load (the one that was loaded, across a
    /// restart, or the one a switch is to), and the file it reads how a switch stands from.
    static let modelVariable = "SPLOSH_MODEL"
    static let reportVariable = "SPLOSH_SWITCH_REPORT"

    static func pidFile(port: Int) -> URL {
        let directory = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask)[0].appendingPathComponent("Splosh", isDirectory: true)
        return directory.appendingPathComponent("serve-\(port).pid")
    }

    /// `splosh serve --restart`: ask the server on `port` to replace its engine.
    static func requestRestart(port: Int) -> Int32 {
        guard let listening = listeningProcess(port: port) else {
            SploshCLI.writeStderr("no splosh server is listening on port \(port); start one with `splosh serve`\n")
            return 1
        }
        // The signal ends a process that does not expect it, so it goes only to one that wrote
        // its own id down as a holder of this port and is the one listening on it.
        let recorded = (try? String(contentsOf: pidFile(port: port), encoding: .utf8)).flatMap { pid_t($0.trimmingCharacters(in: .whitespacesAndNewlines)) }
        guard recorded == listening else {
            SploshCLI.writeStderr("the server on port \(port) (process \(listening)) is of a build that cannot replace its engine in place; start the new one in its place with `splosh serve --takeover`\n")
            return 1
        }
        guard kill(listening, SIGUSR1) == 0 else {
            SploshCLI.writeStderr("could not signal the server (process \(listening)): \(String(cString: strerror(errno)))\n")
            return 1
        }
        SploshCLI.writeStderr("asked the server on port \(port) (process \(listening)) to restart its engine; it reports in its own terminal\n")
        return 0
    }

    /// `model` is the registered model to start on; nil, when none are registered, leaves the
    /// engine to its configuration as before there was a registry.
    static func run(tokens: [String], config: ServeConfig, takeover: Bool, model: String? = nil) -> Int32 {
        signal(SIGPIPE, SIG_IGN)
        // Two descriptors a connection: the default limit of 256 is about 120 of them.
        var limit = rlimit()
        if getrlimit(RLIMIT_NOFILE, &limit) == 0, limit.rlim_cur < 10_240 {
            limit.rlim_cur = min(limit.rlim_max, 10_240)
            setrlimit(RLIMIT_NOFILE, &limit)
        }
        // Taking over: the server on the port is asked to stop, which closes its listener at
        // once and leaves it finishing its requests. This process binds the port the moment
        // it is free and keeps what arrives until that server has gone, model and all.
        var predecessor: pid_t?
        if takeover, let pid = listeningProcess(port: config.port) {
            SploshCLI.writeStderr("taking over from the server on port \(config.port) (process \(pid)): it finishes its requests and saves its conversations; new requests wait here\n")
            kill(pid, SIGINT)
            predecessor = pid
        }
        var listener: Int32 = -1
        let deadline = Date().addingTimeInterval(predecessor == nil ? 0 : 15)
        while listener < 0 {
            do { listener = try listen(host: config.host, port: config.port) } catch {
                guard Date() < deadline else {
                    SploshCLI.writeStderr("serve failed: \(error)\n")
                    return 1
                }
                usleep(200)
            }
        }
        return Supervisor(listener: listener, tokens: tokens, config: config, model: model).run(after: predecessor)
    }

    /// Whether a process is still running: one that has exited and not yet been collected by
    /// its parent still has its id, and still takes a signal, but has let go of everything.
    static func isRunning(_ pid: pid_t) -> Bool {
        var info = kinfo_proc()
        var size = MemoryLayout<kinfo_proc>.size
        var name: [Int32] = [CTL_KERN, KERN_PROC, KERN_PROC_PID, pid]
        guard sysctl(&name, 4, &info, &size, nil, 0) == 0, size > 0 else { return false }
        return info.kp_proc.p_stat != SZOMB
    }

    /// The `splosh` process listening on `port`, if there is one.
    private static func listeningProcess(port: Int) -> pid_t? {
        func output(_ path: String, _ arguments: [String]) -> String {
            let process = Process()
            process.executableURL = URL(fileURLWithPath: path)
            process.arguments = arguments
            let pipe = Pipe()
            process.standardOutput = pipe
            process.standardError = FileHandle.nullDevice
            guard (try? process.run()) != nil else { return "" }
            let data = pipe.fileHandleForReading.readDataToEndOfFile()
            process.waitUntilExit()
            return String(decoding: data, as: UTF8.self)
        }
        for line in output("/usr/sbin/lsof", ["-nP", "-iTCP:\(port)", "-sTCP:LISTEN", "-t"]).split(separator: "\n") {
            guard let pid = pid_t(line), pid != getpid() else { continue }
            if output("/bin/ps", ["-o", "command=", "-p", "\(pid)"]).contains("splosh") { return pid }
        }
        return nil
    }

    private static func listen(host: String, port: Int) throws -> Int32 {
        var hints = addrinfo()
        hints.ai_family = AF_UNSPEC
        hints.ai_socktype = SOCK_STREAM
        hints.ai_flags = AI_PASSIVE
        var found: UnsafeMutablePointer<addrinfo>?
        guard getaddrinfo(host, String(port), &hints, &found) == 0, let info = found else {
            throw CLIError("cannot resolve \(host)")
        }
        defer { freeaddrinfo(found) }
        let fd = socket(info.pointee.ai_family, SOCK_STREAM, 0)
        guard fd >= 0 else { throw CLIError("cannot create a socket: \(String(cString: strerror(errno)))") }
        var yes: Int32 = 1
        setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &yes, socklen_t(MemoryLayout<Int32>.size))
        guard bind(fd, info.pointee.ai_addr, info.pointee.ai_addrlen) == 0 else {
            let reason = String(cString: strerror(errno))
            close(fd)
            throw CLIError("cannot listen on \(host):\(port): \(reason)")
        }
        guard Darwin.listen(fd, 512) == 0 else {
            let reason = String(cString: strerror(errno))
            close(fd)
            throw CLIError("cannot listen on \(host):\(port): \(reason)")
        }
        _ = fcntl(fd, F_SETFL, fcntl(fd, F_GETFL) | O_NONBLOCK)
        return fd
    }
}

private final class Supervisor: @unchecked Sendable {
    private enum Event { case signal(Int32), exited(pid_t, Int32) }
    private enum State {
        case starting(first: Bool)
        case running
        /// A restart: the old engine is finishing, and its requests are cut at `cutAt`.
        case draining(cutAt: Date)
        /// No engine: the last one did not start. Another try at `retryAt`.
        case waiting(retryAt: Date)
        case stopping
    }

    private let listener: Int32
    private let tokens: [String]
    private let host: String
    private let port: Int
    private let restartDrain: TimeInterval
    private let executable: String

    /// Where connections go, nil while there is no engine to take them (they are kept); and
    /// whether the server is stopping (they are closed). Guarded by `gate`.
    private let gate = NSLock()
    private var enginePath: String?
    private var closing = false
    /// Written to when the server is stopping, so that the listener is closed at once and not
    /// at the accepting thread's next look.
    private var wake: [Int32] = [-1, -1]

    private let inbox = NSCondition()
    private var events: [Event] = []

    private var engine: pid_t = 0
    private var engineSocket = ""
    private var engineStarted = Date()
    private var generation = 0

    // Models (see ModelSwitch).
    private let dwell: TimeInterval
    private let switchWait: TimeInterval
    /// How long a model may take to load before the one before it is started again.
    private let loadLimit: TimeInterval
    /// Where the engine's socket and the report of a switch are put.
    private let directory: String
    /// The model engines are started with; nil leaves the choice to the engine's configuration.
    private var model: String?
    /// A switch under way: the engine is being replaced by one with `target`, and if that
    /// does not load, by one with `previous` again.
    private var switching: (target: String, previous: String?)?
    /// What the engine was last told of a switch (see `report`).
    private var reported = ""

    /// What is to become of a request kept for another model.
    private enum Verdict { case replay, refuse([UInt8]) }
    /// A connection, for as long as it is open. Guarded by `gate`.
    private struct Kept {
        let arrived: Date
        /// Its client has sent a request, which waits here for an engine to be given to.
        var waiting = false
        /// That request is a GET; nil until enough of it has come to tell.
        var get: Bool?
        /// It is kept until this model is loaded.
        var want: SwitchWant?
        var verdict: Verdict?
    }
    private var kept: [Int: Kept] = [:]
    private var lastConnection = 0
    /// While set, connections that have arrived since are kept, but for GETs: the engine is
    /// to finish what it has and be replaced. Guarded by `gate`.
    private var barrier: Date?

    init(listener: Int32, tokens: [String], config: ServeConfig, model: String?) {
        self.listener = listener; self.tokens = tokens; self.model = model
        host = config.host; port = config.port; restartDrain = config.restartDrainSeconds
        dwell = config.modelDwellSeconds; switchWait = config.switchWaitSeconds
        // (The variable is for tests of a model that never comes up.)
        loadLimit = ProcessInfo.processInfo.environment["SPLOSH_LOAD_SECONDS"].flatMap(Double.init) ?? 120
        // A directory only this user can reach, where the system provides one short enough for
        // a socket path.
        directory = NSTemporaryDirectory().utf8.count > 70 ? "/tmp/" : NSTemporaryDirectory()
        var size: UInt32 = 0
        _NSGetExecutablePath(nil, &size)
        var buffer = [CChar](repeating: 0, count: Int(size) + 1)
        _NSGetExecutablePath(&buffer, &size)
        // Not resolved to the file it names now: a restart is to run whatever is at this path then.
        executable = String(cString: buffer)
        pipe(&wake)
        for descriptor in wake { _ = fcntl(descriptor, F_SETFL, fcntl(descriptor, F_GETFL) | O_NONBLOCK) }
    }

    private var reportPath: String { directory + "splosh-\(getpid()).switch" }

    func run(after predecessor: pid_t? = nil) -> Int32 {
        // A restart is its own signal: the terminal closing sends SIGHUP, and that is a stop
        // (unless this was started to outlive its terminal, with SIGHUP already ignored).
        var sources: [DispatchSourceSignal] = []
        for number in [SIGINT, SIGTERM, SIGHUP, SIGUSR1] {
            let previous = signal(number, SIG_IGN)
            if number == SIGHUP, unsafeBitCast(previous, to: Int.self) == unsafeBitCast(SIG_IGN, to: Int.self) { continue }
            let source = DispatchSource.makeSignalSource(signal: number, queue: .global())
            source.setEventHandler { [self] in post(.signal(number)) }
            source.resume()
            sources.append(source)
        }
        defer { sources.forEach { $0.cancel() } }
        Thread.detachNewThread { [self] in acceptLoop() }

        if let predecessor {
            // Connections are being accepted and kept; the model cannot be loaded twice.
            // Its requests get the restart's time to finish, not the whole drain: the clients
            // kept here give up after five minutes without a byte. A second stop cuts them
            // (a third would end the old engine before it had saved, so only one is sent).
            let cutAt = Date().addingTimeInterval(max(restartDrain, 0.5))
            var cut = false
            while ServeSupervisor.isRunning(predecessor) {
                if case .signal(let number)? = next(timeout: 0.2), number != SIGUSR1 { return finish(1) }
                if !cut, Date() >= cutAt {
                    cut = true
                    SploshCLI.writeStderr("the previous server is still answering; ending its requests for their clients to send again\n")
                    kill(predecessor, SIGINT)
                }
            }
            SploshCLI.writeStderr("the previous server has gone; starting the engine\n")
        }
        guard startEngine() else { return finish(1) }
        let file = ServeSupervisor.pidFile(port: port)
        try? FileManager.default.createDirectory(at: file.deletingLastPathComponent(), withIntermediateDirectories: true)
        try? "\(getpid())\n".write(to: file, atomically: true, encoding: .utf8)

        var state = State.starting(first: true)
        var restartBegan = Date()
        var restartAsked = false                // asked for while an engine was still starting
        var startFailures = 0                   // engines in a row that did not start
        var shortLived = 0                      // engines in a row that started and soon stopped
        var stops = 0                           // stop signals this engine has had; its third ends it unsaved
        var readySince = Date()                 // when the engine in place began taking requests
        var idlePolls = 0                       // times in a row it has had nothing in flight, with new requests kept
        var lastPoll = Date.distantPast
        var loaded: String?                     // the model the engine in place says it has
        var overdue = false                     // a model being switched to has had as long as one may to load

        /// The engine is asked to finish; new connections are kept from here on.
        func beginRestart() {
            hold()
            restartBegan = Date()
            SploshCLI.writeStderr(String(format: "\nrestarting the engine: new requests wait and the port stays open; requests in flight have %.0f s to finish, then are cut for their clients to send again\n", restartDrain))
            kill(engine, SIGTERM)
            stops = 1
            // Not sooner than the engine tells two signals apart.
            state = .draining(cutAt: Date().addingTimeInterval(max(restartDrain, 0.5)))
        }
        /// Start an engine in place of one that has gone, or wait and try again.
        func replaceEngine() {
            stops = 0
            overdue = false
            if let target = switching?.target { report(target: target, phase: "loading") }
            if startEngine() {
                state = .starting(first: false)
            } else {
                startFailures += 1
                state = .waiting(retryAt: Date().addingTimeInterval(5))
            }
        }
        /// The engine has nothing in flight and new requests are being kept: it is stopped,
        /// which cuts nothing, and one with `target` loaded is started when it has gone.
        func beginSwitch(to target: String) {
            hold()
            restartBegan = Date()
            switching = (target, loaded ?? model)
            SploshCLI.writeStderr("\nswitching to \(target): \(loaded ?? model ?? "the loaded model") has nothing in flight, so its engine is stopped and its conversations saved; requests wait and the port stays open\n")
            kill(engine, SIGTERM)
            stops = 1
            state = .draining(cutAt: Date().addingTimeInterval(max(restartDrain, 0.5)))
            report(target: target, phase: "stopping")
        }
        /// The requests kept for a model that is not loaded, looked at while the engine serves.
        /// One that has waited as long as it may is refused. For the rest the loaded model's
        /// turn runs out; then new requests are kept too, and the engine, once it has nothing
        /// in flight, is replaced.
        func attend() {
            let now = Date()
            gate.lock()
            var wants: [(id: Int, want: SwitchWant)] = []
            for (id, entry) in kept {
                guard let want = entry.want else { continue }
                if SwitchPolicy.expired(want, now: now, wait: switchWait) {
                    kept[id]?.want = nil
                    kept[id]?.verdict = .refuse(SwitchRefusal.busy(model: want.model, loaded: loaded ?? model, waited: now.timeIntervalSince(want.since)))
                } else {
                    wants.append((id, want))
                }
            }
            wants.sort { ($0.want.since, $0.id) < ($1.want.since, $1.id) }
            let decision = SwitchPolicy.decide(now: now, readySince: readySince, wants: wants.map(\.want),
                                               idle: barrier != nil && idlePolls >= 2, dwell: dwell)
            var target: String?, phase: String?
            switch decision {
            case .stay:
                barrier = nil
            case .dwell(let wanted, _):
                barrier = nil
                (target, phase) = (wanted, "dwell")
            case .drain(let wanted), .swap(let wanted):
                if barrier == nil { barrier = now; idlePolls = 0 }
                (target, phase) = (wanted, "waiting")
            }
            // Requests on their way to the engine, which it cannot have counted yet: the ones
            // kept while there was no engine, and the ones kept for this model until it was loaded.
            let since = barrier
            let coming = kept.values.filter { entry in
                guard entry.waiting, entry.want == nil else { return false }
                if let since, entry.arrived >= since, entry.get != true { return false }       // kept, for the next engine
                return true
            }.count
            gate.unlock()

            if since == nil {
                idlePolls = 0
            } else if now.timeIntervalSince(lastPoll) >= 0.2 {
                // Twice in a row, a moment apart: a request given to the engine as it was last
                // asked is counted by the time it is asked again.
                lastPoll = now
                let busy = Self.busy(engineSocket)
                loaded = busy?.model ?? loaded
                idlePolls = busy?.count == 0 && coming == 0 ? idlePolls + 1 : 0
                if let target, target == busy?.model {
                    // The engine has the model after all (it was started on another than the
                    // one asked for, which the file had stopped registering): nothing to replace.
                    settle(wanting: target, .replay)
                    return
                }
            }
            report(target: target, phase: phase)
            if case .swap(let wanted) = decision, coming == 0 { beginSwitch(to: wanted) }
        }

        while true {
            let event = next(timeout: 0.1)
            switch (state, event) {
            // An engine starting.
            case (.starting(let first), nil):
                guard let probe = Self.connectUnix(engineSocket) else {
                    if switching != nil, !overdue, Date().timeIntervalSince(engineStarted) > loadLimit {
                        // Its exit is taken as any model's that does not load.
                        overdue = true
                        kill(engine, SIGKILL)
                    }
                    break
                }
                close(probe)
                open(engineSocket)
                state = .running
                startFailures = 0
                readySince = Date()
                idlePolls = 0
                if let (target, _) = switching {
                    // The requests kept for it are the first it is given.
                    switching = nil
                    model = target
                    loaded = target
                    settle(wanting: target, .replay)
                    SploshCLI.writeStderr(String(format: "switched to %@ in %.1f s: the new engine is taking requests\n", target, Date().timeIntervalSince(restartBegan)))
                } else {
                    SploshCLI.writeStderr(first
                        ? "listening on http://\(host):\(port)  (dashboard at /, API at /v1; `splosh serve --restart` to replace the engine with the port kept open)\n"
                        : String(format: "restarted in %.1f s: the new engine is taking requests\n", Date().timeIntervalSince(restartBegan)))
                }
                report(target: nil, phase: nil)
                if restartAsked { restartAsked = false; beginRestart() }
            case (.starting(let first), .exited(let pid, let status)) where pid == engine:
                if first { return finish(status == 0 ? 1 : status) }
                if let (target, previous) = switching {
                    // The model wanted did not load. The one before it is started again at
                    // once, and the requests kept for the other are told.
                    switching = nil
                    model = previous ?? model
                    loaded = model
                    SploshCLI.writeStderr("the model \(target) did not load (\(overdue ? String(format: "not ready in %.0f s", loadLimit) : "status \(status)")); starting \(model ?? "the model before it") again\n")
                    settle(wanting: target, .refuse(SwitchRefusal.loadFailed(model: target, loaded: model)))
                    replaceEngine()
                    break
                }
                startFailures += 1
                guard startFailures < 20 else {
                    SploshCLI.writeStderr("the engine has failed to start \(startFailures) times; stopping\n")
                    return finish(status == 0 ? 1 : status)
                }
                SploshCLI.writeStderr("the new engine did not start (status \(status)); trying again in 5 s. Requests are still being kept; Ctrl+C stops the server\n")
                state = .waiting(retryAt: Date().addingTimeInterval(5))
            case (.starting, .signal(SIGUSR1)):
                restartAsked = true
            case (.starting, .signal(let number)):
                refuse()
                forward(number)
                stops += 1
                state = .stopping

            // No engine.
            case (.waiting(let retryAt), nil) where Date() >= retryAt:
                replaceEngine()
            case (.waiting, .signal(SIGUSR1)):
                replaceEngine()
            case (.waiting, .signal):
                return finish(1)

            // Serving.
            case (.running, .signal(SIGUSR1)):
                beginRestart()
            case (.running, .signal(let number)):
                refuse()
                forward(number)
                stops += 1
                state = .stopping
            case (.running, .exited(let pid, let status)) where pid == engine:
                hold()
                restartBegan = Date()
                shortLived = Date().timeIntervalSince(engineStarted) > 60 ? 0 : shortLived + 1
                guard shortLived < 5 else {
                    SploshCLI.writeStderr("the engine keeps stopping soon after it starts (status \(status)); not starting it again\n")
                    return finish(status == 0 ? 1 : status)
                }
                SploshCLI.writeStderr("the engine stopped unexpectedly (status \(status)); starting it again. Requests it had not begun to answer are given to the new one; new ones wait\n")
                replaceEngine()
            case (.running, nil):
                attend()

            // A restart or a switch, the old engine finishing.
            case (.draining(let cutAt), nil) where Date() >= cutAt:
                SploshCLI.writeStderr("\(switching == nil ? "restarting the engine" : "switching models"): ending the requests still running\n")
                kill(engine, SIGTERM)
                stops = 2
                state = .draining(cutAt: .distantFuture)
            case (.draining(let cutAt), .signal(SIGUSR1)) where cutAt != .distantFuture:
                // The cut, brought forward; not sooner than the engine tells two signals apart.
                state = .draining(cutAt: min(cutAt, max(Date(), restartBegan.addingTimeInterval(0.5))))
            case (.draining, .signal(let number)) where number != SIGUSR1:
                // The engine is already finishing: a stop now only means no new engine after it.
                // A second stop cuts its requests short.
                refuse()
                SploshCLI.writeStderr(stops < 2 ? "\nstopping once the engine has finished (Ctrl+C again to cut its requests short)\n"
                                                : "\nstopping once the engine has saved its conversations\n")
                state = .stopping
            case (.draining, .exited(let pid, _)) where pid == engine:
                replaceEngine()

            // A stop.
            case (.stopping, .signal(let number)) where number != SIGUSR1:
                // The engine's third stop ends it at once, with nothing saved.
                if stops < 2 {
                    forward(number)
                    stops += 1
                } else {
                    SploshCLI.writeStderr("the engine is saving its conversations and will stop when it has\n")
                }
            case (.stopping, .exited(let pid, let status)) where pid == engine:
                return finish(status)
            default:
                break
            }
        }
    }

    private func forward(_ number: Int32) { kill(engine, number == SIGHUP ? SIGTERM : number) }

    // MARK: Events

    private func post(_ event: Event) {
        inbox.lock(); events.append(event); inbox.signal(); inbox.unlock()
    }

    private func next(timeout: TimeInterval) -> Event? {
        inbox.lock(); defer { inbox.unlock() }
        if events.isEmpty { _ = inbox.wait(until: Date().addingTimeInterval(timeout)) }
        return events.isEmpty ? nil : events.removeFirst()
    }

    private func open(_ path: String) { gate.lock(); enginePath = path; gate.unlock() }
    /// Keep new connections until there is an engine again.
    private func hold() { gate.lock(); enginePath = nil; barrier = nil; gate.unlock() }
    /// Stop taking connections: the server is going.
    private func refuse() {
        gate.lock(); enginePath = nil; closing = true; gate.unlock()
        var byte: UInt8 = 1
        _ = write(wake[1], &byte, 1)
    }

    private func destination() -> (path: String?, closing: Bool) {
        gate.lock(); defer { gate.unlock() }
        return (enginePath, closing)
    }

    private func finish(_ status: Int32) -> Int32 {
        refuse()
        if !engineSocket.isEmpty { unlink(engineSocket) }
        unlink(reportPath)
        try? FileManager.default.removeItem(at: ServeSupervisor.pidFile(port: port))
        return status
    }

    // MARK: Requests kept for another model

    private func arrived() -> (id: Int, at: Date) {
        gate.lock(); defer { gate.unlock() }
        lastConnection += 1
        let entry = Kept(arrived: Date())
        kept[lastConnection] = entry
        return (lastConnection, entry.arrived)
    }

    private func gone(_ id: Int) { gate.lock(); kept[id] = nil; gate.unlock() }

    /// Where connection `id` goes with what its client has sent so far: an engine's socket, or
    /// nowhere yet. With new requests being kept for the next engine, one that arrived since
    /// they were goes to this engine only once it is seen to be a GET.
    private func destination(of id: Int, request: [UInt8]) -> (path: String?, closing: Bool) {
        gate.lock(); defer { gate.unlock() }
        if closing { return (nil, true) }
        let get: Bool? = request.count < Self.get.count ? nil : request.starts(with: Self.get)
        kept[id]?.waiting = !request.isEmpty
        kept[id]?.get = get
        if let barrier, let entry = kept[id], entry.arrived >= barrier, get != true { return (nil, false) }
        return (enginePath, false)
    }
    private static let get = Array("GET ".utf8)

    /// Connection `id`'s request is with an engine.
    private func handed(_ id: Int) { gate.lock(); kept[id]?.waiting = false; gate.unlock() }

    /// Connection `id`'s request is kept until the model it names is loaded.
    private func park(_ id: Int, _ want: SwitchWant) {
        gate.lock(); kept[id]?.want = want; kept[id]?.waiting = false; kept[id]?.verdict = nil; gate.unlock()
    }

    /// What has been decided for connection `id`'s kept request, once it has been.
    private func verdict(of id: Int) -> (verdict: Verdict?, closing: Bool) {
        gate.lock(); defer { gate.unlock() }
        let verdict = kept[id]?.verdict
        kept[id]?.verdict = nil
        return (verdict, closing)
    }

    /// Decide every request kept for `model`. One to be given to the engine counts as on its
    /// way there from this moment (see `attend`).
    private func settle(wanting model: String, _ verdict: Verdict) {
        gate.lock(); defer { gate.unlock() }
        for (id, entry) in kept where entry.want?.model == model {
            kept[id]?.want = nil
            kept[id]?.verdict = verdict
            if case .replay = verdict { kept[id]?.waiting = true }
        }
    }

    /// Tell the engine how a switch stands, when that has changed: it is the engine that
    /// answers `GET /v1/models`, and this process that knows what it is keeping.
    private func report(target: String?, phase: String?) {
        gate.lock()
        let parked = kept.values.filter { $0.want != nil }.count
        gate.unlock()
        let text = target == nil && parked == 0 ? "" : SwitchReport.text(target: target, phase: phase, parked: parked)
        guard text != reported else { return }
        reported = text
        if text.isEmpty {
            unlink(reportPath)
        } else {
            try? Data(text.utf8).write(to: URL(fileURLWithPath: reportPath), options: .atomic)
        }
    }

    /// What the engine has in flight; nil if it does not say within half a second.
    private static func busy(_ path: String) -> EngineBusy? {
        guard let descriptor = connectUnix(path) else { return nil }
        defer { close(descriptor) }
        let request = Array("GET /v1/busy HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n".utf8)
        guard writeAll(descriptor, request, request.count) else { return nil }
        var response: [UInt8] = []
        var buffer = [UInt8](repeating: 0, count: 4096)
        let deadline = Date().addingTimeInterval(0.5)
        while response.count < 1 << 16 {
            if let busy = EngineBusy.parse(response) { return busy }
            var waiting = pollfd(fd: descriptor, events: Int16(POLLIN), revents: 0)
            let left = deadline.timeIntervalSinceNow
            guard left > 0, poll(&waiting, 1, Int32(left * 1000) + 1) > 0 else { return nil }
            let count = read(descriptor, &buffer, buffer.count)
            if count < 0 && errno == EINTR { continue }
            if count <= 0 { break }
            response.append(contentsOf: buffer[0..<count])
        }
        return EngineBusy.parse(response)
    }

    // MARK: The engine

    private func startEngine() -> Bool {
        if !engineSocket.isEmpty { unlink(engineSocket) }
        generation += 1
        engineSocket = directory + "splosh-\(getpid())-\(generation).sock"
        unlink(engineSocket)

        var attributes: posix_spawnattr_t?
        posix_spawnattr_init(&attributes)
        defer { posix_spawnattr_destroy(&attributes) }
        // Its own process group: the terminal's Ctrl+C reaches this process only, which passes
        // each signal on once. Default handling for the signals ignored here, and none of this
        // process's sockets.
        var defaults = sigset_t()
        sigemptyset(&defaults)
        for number in [SIGINT, SIGTERM, SIGHUP, SIGUSR1, SIGPIPE] { sigaddset(&defaults, number) }
        posix_spawnattr_setsigdefault(&attributes, &defaults)
        posix_spawnattr_setpgroup(&attributes, 0)
        posix_spawnattr_setflags(&attributes, Int16(POSIX_SPAWN_SETPGROUP | POSIX_SPAWN_SETSIGDEF | POSIX_SPAWN_CLOEXEC_DEFAULT))
        var actions: posix_spawn_file_actions_t?
        posix_spawn_file_actions_init(&actions)
        defer { posix_spawn_file_actions_destroy(&actions) }
        for descriptor: Int32 in 0...2 { posix_spawn_file_actions_addinherit_np(&actions, descriptor) }

        var environment = ProcessInfo.processInfo.environment
        environment[ServeSupervisor.socketVariable] = engineSocket
        environment[ServeSupervisor.holderVariable] = "\(getpid())"
        environment[ServeSupervisor.reportVariable] = reportPath
        if let model = switching?.target ?? model { environment[ServeSupervisor.modelVariable] = model }
        let argv: [UnsafeMutablePointer<CChar>?] = ([executable, "serve"] + tokens).map { strdup($0) } + [nil]
        let envp: [UnsafeMutablePointer<CChar>?] = environment.map { strdup("\($0.key)=\($0.value)") } + [nil]
        defer { (argv + envp).forEach { free($0) } }

        var pid: pid_t = 0
        let result = posix_spawn(&pid, executable, &actions, &attributes, argv, envp)
        guard result == 0 else {
            SploshCLI.writeStderr("cannot start the engine (\(executable)): \(String(cString: strerror(result)))\n")
            return false
        }
        engine = pid
        engineStarted = Date()
        Thread.detachNewThread { [self] in
            var status: Int32 = 0
            while waitpid(pid, &status, 0) < 0 && errno == EINTR {}
            post(.exited(pid, status & 0x7f == 0 ? (status >> 8) & 0xff : 128 + (status & 0x7f)))
        }
        return true
    }

    // MARK: Connections

    private func acceptLoop() {
        while true {
            if destination().closing { close(listener); return }
            var waiting = [pollfd(fd: listener, events: Int16(POLLIN), revents: 0), pollfd(fd: wake[0], events: Int16(POLLIN), revents: 0)]
            guard poll(&waiting, 2, 1000) > 0, waiting[0].revents != 0, waiting[1].revents == 0 else { continue }
            let client = accept(listener, nil, nil)
            guard client >= 0 else {
                if errno != EWOULDBLOCK && errno != EINTR && errno != ECONNABORTED { usleep(20_000) }   // out of descriptors: wait for some
                continue
            }
            _ = fcntl(client, F_SETFL, fcntl(client, F_GETFL) & ~O_NONBLOCK)
            var yes: Int32 = 1
            setsockopt(client, SOL_SOCKET, SO_NOSIGPIPE, &yes, socklen_t(MemoryLayout<Int32>.size))
            setsockopt(client, SOL_SOCKET, SO_KEEPALIVE, &yes, socklen_t(MemoryLayout<Int32>.size))
            setsockopt(client, IPPROTO_TCP, TCP_NODELAY, &yes, socklen_t(MemoryLayout<Int32>.size))
            Thread.detachNewThread { [self] in serve(client) }
        }
    }

    /// One connection: to the engine if there is one; otherwise kept, taking in what the client
    /// sends so that its write completes, until there is.
    ///
    /// What the client has sent is also kept until the engine answers. An engine that closes
    /// the connection without a byte of answer (it was told to stop as the request arrived, or
    /// it died) has not served the request, and the request is given to the next engine.
    ///
    /// So is a request the engine answers with the marker of a model it does not have: the
    /// request is kept until the model is loaded (or cannot be, and the client is told so
    /// here), and none of that answer is the client's.
    private func serve(_ client: Int32) {
        defer { close(client) }
        let (id, arrival) = arrived()
        defer { gone(id) }
        var buffer = [UInt8](repeating: 0, count: 65_536)
        var request: [UInt8] = []               // what the client has sent that no engine has begun to answer
        var began = arrival                     // when the first of it came
        var whole = true                        // `request` is all of that
        var sending = true                      // the client may send more
        var given = 0                           // engines that had it and did not answer; a wait for a model is not one
        while given < 4 {
            // An engine to give it to.
            var backend: Int32 = -1
            var refusedSince: Date?
            while backend < 0 {
                let (path, closing) = destination(of: id, request: request)
                if closing { return }
                if let path {
                    if let connected = Self.connectUnix(path) { backend = connected; break }
                    // That engine has gone and this process has not heard yet: wait for the next.
                    let since = refusedSince ?? Date()
                    refusedSince = since
                    if Date().timeIntervalSince(since) > 15 { return }
                } else {
                    refusedSince = nil
                }
                guard sending, request.count < Self.heldLimit else { usleep(100_000); continue }
                var waiting = pollfd(fd: client, events: Int16(POLLIN), revents: 0)
                guard poll(&waiting, 1, 100) > 0 else { continue }
                let count = read(client, &buffer, buffer.count)
                if count < 0 && errno == EINTR { continue }
                if count <= 0 { return }                             // the client gave up waiting
                if request.isEmpty { began = Date() }
                request.append(contentsOf: buffer[0..<count])
            }

            var answered = false
            var wanted: SwitchWant?
            do {
                defer { close(backend) }
                var head: [UInt8] = []          // the engine's answer so far, until it is known to be one
                var engineEnded = !Self.writeAll(backend, request, request.count)
                handed(id)
                var pair = [pollfd(fd: client, events: Int16(POLLIN), revents: 0), pollfd(fd: backend, events: Int16(POLLIN), revents: 0)]
                if !sending {
                    // Kept for a model after its client had finished sending: the engine is told again.
                    pair[0].fd = -1
                    shutdown(backend, SHUT_WR)
                }
                pump: while !engineEnded {
                    if poll(&pair, 2, -1) < 0 {
                        if errno == EINTR { continue }
                        return
                    }
                    if pair[1].revents != 0 {
                        let count = read(backend, &buffer, buffer.count)
                        if count < 0 && errno == EINTR { continue }
                        if count <= 0 { engineEnded = true; break pump }
                        if answered, request.isEmpty, head.isEmpty {
                            // More of an answer under way.
                            guard Self.writeAll(client, buffer, count) else { return }  // the client has gone
                        } else {
                            // The start of its answer to what the client last sent (a client may
                            // send a second request where it was asked to use the connection once).
                            head.append(contentsOf: buffer[0..<count])
                            switch SwitchMarker.inspect(head) {
                            case .undecided:
                                break
                            case .wanted(let model, let admin):
                                wanted = SwitchWant(model: model, since: began, admin: admin)
                                break pump
                            case .answer:
                                answered = true
                                request = []
                                whole = true
                                guard Self.writeAll(client, head, head.count) else { return }
                                head = []
                            }
                        }
                    }
                    if pair[0].fd >= 0, pair[0].revents != 0 {
                        let count = read(client, &buffer, buffer.count)
                        if count < 0 && errno == EINTR { continue }
                        if count <= 0 {
                            // The client has finished sending, or gone. The engine is told; what it
                            // has still to say is passed on, and fails to be if the client has gone.
                            sending = false
                            pair[0].fd = -1
                            shutdown(backend, SHUT_WR)
                            continue
                        }
                        // Kept until the engine begins to answer it: before its first answer, to
                        // give to the next engine if it never does; and always, to keep here if
                        // its answer is that the request is for a model it does not have.
                        if request.isEmpty { began = Date() }
                        if request.count + count <= Self.heldLimit { request.append(contentsOf: buffer[0..<count]) } else { whole = false }
                        if !Self.writeAll(backend, buffer, count) { engineEnded = true }
                    }
                }
                if wanted == nil, !head.isEmpty {
                    // It ended part-way through the head of an answer: that much is the client's.
                    answered = true
                    request = []
                    guard Self.writeAll(client, head, head.count) else { return }
                }
            }
            if let wanted {
                guard whole, !request.isEmpty else {
                    let refusal = SwitchRefusal.tooLarge(model: wanted.model, limit: Self.heldLimit)
                    if Self.writeAll(client, refusal, refusal.count), sending { Self.lastWords(client, &buffer) }
                    return
                }
                park(id, wanted)
                var verdict: Verdict?
                while verdict == nil {
                    let (decided, closing) = self.verdict(of: id)
                    if closing { return }
                    verdict = decided
                    guard verdict == nil else { break }
                    guard sending else { usleep(100_000); continue }
                    var waiting = pollfd(fd: client, events: Int16(POLLIN), revents: 0)
                    guard poll(&waiting, 1, 100) > 0 else { continue }
                    let count = read(client, &buffer, buffer.count)
                    if count < 0 && errno == EINTR { continue }
                    if count <= 0 { return }                         // the client gave up waiting
                    if request.count + count <= Self.heldLimit { request.append(contentsOf: buffer[0..<count]) } else { whole = false }
                }
                if case .refuse(let refusal)? = verdict {
                    if Self.writeAll(client, refusal, refusal.count), sending { Self.lastWords(client, &buffer) }
                    return
                }
                continue                                                                // to the engine that has the model
            }
            if !answered && whole && sending && !request.isEmpty { given += 1; continue }   // to the next engine
            if sending { Self.lastWords(client, &buffer) }
            return
        }
    }

    /// The connection is ending from this side. The client is to have all that was sent:
    /// closing with its bytes unread here would reset the connection instead.
    private static func lastWords(_ client: Int32, _ buffer: inout [UInt8]) {
        shutdown(client, SHUT_WR)
        let until = Date().addingTimeInterval(2)
        var rest = pollfd(fd: client, events: Int16(POLLIN), revents: 0)
        while Date() < until, poll(&rest, 1, 200) >= 0 {
            if rest.revents != 0, read(client, &buffer, buffer.count) <= 0 { break }
        }
    }

    /// As much of a waiting client's request as is kept in memory before it is left unread.
    private static let heldLimit = 64 << 20

    private static func writeAll(_ descriptor: Int32, _ bytes: [UInt8], _ count: Int) -> Bool {
        var sent = 0
        while sent < count {
            let written = bytes.withUnsafeBytes { write(descriptor, $0.baseAddress! + sent, count - sent) }
            if written < 0 && errno == EINTR { continue }
            if written <= 0 { return false }
            sent += written
        }
        return true
    }

    fileprivate static func connectUnix(_ path: String) -> Int32? {
        let descriptor = socket(AF_UNIX, SOCK_STREAM, 0)
        guard descriptor >= 0 else { return nil }
        var address = sockaddr_un()
        address.sun_family = sa_family_t(AF_UNIX)
        let capacity = MemoryLayout.size(ofValue: address.sun_path)
        guard path.utf8.count < capacity else { close(descriptor); return nil }
        withUnsafeMutableBytes(of: &address.sun_path) { raw in
            raw.copyBytes(from: path.utf8)
        }
        let result = withUnsafePointer(to: &address) {
            $0.withMemoryRebound(to: sockaddr.self, capacity: 1) { connect(descriptor, $0, socklen_t(MemoryLayout<sockaddr_un>.size)) }
        }
        guard result == 0 else { close(descriptor); return nil }
        var yes: Int32 = 1
        setsockopt(descriptor, SOL_SOCKET, SO_NOSIGPIPE, &yes, socklen_t(MemoryLayout<Int32>.size))
        return descriptor
    }
}
