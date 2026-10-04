// main.swift — SploshCLI.
//
// Owner: M0.5 lands the dispatch table and the `doctor`/`serve` stub tokens. M0.9 lands the flag
// grammar (`CLIArguments`), M0.10 replaces the `doctor` token with `DoctorCommand.run(_:)`, and
// M1.4 replaces the `serve` token with `ServeCommand.run(_:)`.
//
// Contract source: rev4 §4.6 (the command table), §6 M0.5 and §6 M0.9.
//
// The dispatch is hand-written because rev4 §3.3's one-dependency rule excludes Swift
// ArgumentParser: the package's single external dependency is Hummingbird. `doctor` is this
// execution plan's addition (`IMPLEMENTATION-PLAN.md` §4.2), which is why the table has seven
// commands where rev4 §4.6 names six.

import Foundation

/// The commands the executable dispatches on (rev4 §4.6 plus `doctor`).
public enum CommandName: String, CaseIterable, Sendable {
    case doctor
    case serve
    case models
    case download
    case convert
    case bench
    case cache
    case oracle
    case soak
    case generate
}

/// Process exit codes the CLI uses.
public enum ExitStatus {
    /// Success.
    public static let ok: Int32 = 0
    /// The arguments did not parse.
    public static let usage: Int32 = 2
    /// `EX_UNAVAILABLE` — the command exists but is not implemented yet.
    ///
    /// A stub must never exit 0: rev4 §6 M0.9's failure mode is "a stub that exits `0`", which lets
    /// a later gate mistake an unimplemented command for a working one.
    public static let notImplemented: Int32 = 69
}

/// The CLI entry point, split from `main.swift`'s top-level statement so tests can drive it.
public enum SploshCLI {
    /// Usage for a command still living as an inline token here.
    private static let doctorUsage = """
        usage: splosh doctor

        Report one line per prerequisite:
          PASS|FAIL|DEFERRED <check-id> <observed> <required> <remediation>

        Exits non-zero when any capability required through M1 fails.
        """

    private static let serveUsage = """
        usage: splosh serve [--config <path>] [--port <n>] [--model <id>] [--restart] [--no-open]

          --config <path>   default ./splosh.toml; an absent file means all defaults
          --port <n>        default 8091; the bind is 127.0.0.1 only
          --model <id>      start on this registered model (a `model.<id>` line of the config)
                            instead of the one the config names; see `splosh models`
          --restart         ask the server already running on that port to replace its engine
                            (a new build, changed settings) without closing the port: requests
                            in flight finish, new ones wait and are answered by the new engine
          --takeover        start in place of the server already on that port, of any build: it
                            is asked to stop (it finishes its requests and saves its
                            conversations) and requests arriving meanwhile wait for this one
          --no-open         do not show the server's page in the browser when it is up (a server
                            started from a terminal does, unless openBrowser = false)
          --echo            serve without the model, saying back what is sent (for testing)

        With no model installed the server starts all the same and offers the downloads: on its
        page, and in the terminal, where Enter fetches the default (see `splosh download`).
        """

    /// The top-level usage text.
    public static let globalUsage = """
        usage: splosh <command> [options]

        Commands:
          doctor    check the environment and print remediation for anything missing
          serve     start the loopback OpenAI-compatible HTTP service
          models    list the models the server can load, and load one
          download  fetch a model from Hugging Face and make it ready to serve
          convert   MLX safetensors or a GGUF file -> the Splosh weight format
          bench     step timing across 2K/8K/32K/128K contexts
          cache     inspect and purge the durable prefix cache
          oracle    diff the GPU engine against the CPU oracle
          soak      sustained-load harness
          generate  one-shot greedy generation through the engine

        Run `splosh <command> --help` for that command's flags.
        """

    /// Usage text for one command.
    public static func usage(for command: CommandName) -> String {
        switch command {
        case .doctor: return doctorUsage
        case .serve: return serveUsage
        case .models: return ModelsCommand.usage
        case .download: return DownloadCommand.usage
        case .convert: return ConvertCommand.usage
        case .bench: return BenchCommand.usage
        case .cache: return CacheCommand.usage
        case .oracle: return OracleCommand.usage
        case .soak: return SoakCommand.usage
        case .generate: return GenerateCommand.usage
        }
    }

    /// Report a command as present but unimplemented, on stderr, with a non-zero status.
    public static func notImplemented(_ command: CommandName) -> Int32 {
        writeStderr("not implemented: \(command.rawValue)\n")
        return ExitStatus.notImplemented
    }

    /// Write to stderr without requiring a `print` overload dance.
    public static func writeStderr(_ text: String) {
        // A plain write: the terminal may have gone, and FileHandle's raises then.
        let bytes = Array(text.utf8)
        bytes.withUnsafeBytes { _ = write(2, $0.baseAddress, $0.count) }
    }

    /// Run one invocation. `arguments` excludes `argv[0]`.
    ///
    /// Returns the process exit status; it never calls `exit`, so tests can call it directly.
    public static func run(arguments: [String]) -> Int32 {
        guard let first = arguments.first else {
            writeStderr(globalUsage + "\n")
            return ExitStatus.usage
        }

        if first == "--help" || first == "-h" {
            print(globalUsage)
            return ExitStatus.ok
        }

        guard let command = CommandName(rawValue: first) else {
            writeStderr("unknown command: \(first)\n")
            writeStderr(globalUsage + "\n")
            return ExitStatus.usage
        }

        let rest = Array(arguments.dropFirst())

        // `--help` wins over every other flag and exits 0 for all seven commands (rev4 §6 M0.5).
        if rest.contains("--help") || rest.contains("-h") {
            print(usage(for: command))
            return ExitStatus.ok
        }

        switch command {
        case .doctor:
            return DoctorCommand.run(rest)
        case .serve:
            return ServeCommand.run(rest)
        case .models:
            return ModelsCommand.run(rest)
        case .download:
            return DownloadCommand.run(rest)
        case .convert:
            return ConvertCommand.run(rest)
        case .bench:
            return BenchCommand.run(rest)
        case .cache:
            return CacheCommand.run(rest)
        case .oracle:
            return OracleCommand.run(rest)
        case .soak:
            return SoakCommand.run(rest)
        case .generate:
            return GenerateCommand.run(rest)
        }
    }
}

// MARK: - Argument grammar (M0.9)

// The grammar lives in `main.swift` rather than a file of its own because rev4 §3.2's tree names the
// `SploshCLI` files and `IMPLEMENTATION-PLAN.md` §15 registers no `CLIArguments.swift`; §15's closing
// rule is that a file with no producing row must not exist.

/// A CLI parse failure.
///
/// `Equatable` so M0.9's committed table can assert the *message*, not merely that something threw;
/// a rejection that names the wrong flag is a different defect from one that names none.
public struct CLIError: Error, Equatable, Sendable, CustomStringConvertible {
    public let message: String

    public init(_ message: String) {
        self.message = message
    }

    public var description: String {
        message
    }
}

/// `--context` values (rev4 §4.6, §6 M0.9).
public enum ContextLength: String, CaseIterable, Sendable {
    case k2 = "2K"
    case k8 = "8K"
    case k32 = "32K"
    case k128 = "128K"
}

/// `--speculative` (rev4 §6 M0.9).
public enum SpeculativeMode: String, CaseIterable, Sendable {
    case on
    case off
}

/// `--kv-format` (rev4 §6 M0.9).
///
/// `q4` is in the *grammar* but is only served if M4.5 lands: `IMPLEMENTATION-PLAN.md` §10 makes q4
/// an explicit owner opt-in, and rejects `--kv-format q4` when it has not happened. M0's command is a
/// stub, so parsing it here reserves the spelling without promising the behaviour.
public enum KvFormat: String, CaseIterable, Sendable {
    case int8
    case q4
}

/// The parsed form of `splosh bench`'s arguments (rev4 §4.6).
public struct BenchArguments: Equatable, Sendable {
    public var contexts: [ContextLength] = []
    public var weights: String?
    public var cacheDir: String?
    public var restartCycles: Int?
    public var speculative: SpeculativeMode?
    public var batch: Int?
    public var concurrency: Int?
    public var generate: Int?
    public var agentTurn = false
    public var prefix: Int?
    public var thinking: Int?
    public var greedyAgreement = false
    public var restore: String?
    public var kvFormat: KvFormat?
    public var help = false

    public init() {}
}

/// The parsed form of `splosh serve`'s arguments. `--echo` is reserved for M1.4.
public struct ServeArguments: Equatable, Sendable {
    public var config: String?
    public var port: Int?
    /// The registered model to start on.
    public var model: String?
    public var echo = false
    public var restart = false
    public var takeover = false
    /// Leave the browser alone at this start (see `ServeConfig.openBrowser`).
    public var noOpen = false
    public var help = false

    public init() {}
}

/// The parsed form of `splosh models`'s arguments.
public struct ModelsArguments: Equatable, Sendable {
    public var config: String?
    public var port: Int?
    public var json = false
    /// The registered model the server is asked to load.
    public var load: String?
    public var help = false

    public init() {}
}

/// The parsed form of `splosh doctor`'s arguments: no required arguments at all.
public struct DoctorArguments: Equatable, Sendable {
    public var help = false

    public init() {}
}

/// Hand-written argument parsing. No external dependency (rev4 §3.3's one-dependency rule).
public enum CLIArguments {
    // MARK: bench

    /// Parse `splosh bench`'s arguments.
    public static func parseBench(_ tokens: [String]) throws -> BenchArguments {
        var result = BenchArguments()
        var stream = TokenStream(tokens)

        while let token = stream.current {
            switch token {
            case "--help", "-h":
                result.help = true
            case "--context":
                // Accumulates across repeats as well as across a comma list, so
                // `--context 2K --context 128K` and `--context 2K,128K` agree.
                for context in try contextList(stream.requireValue(for: token))
                where !result.contexts.contains(context) {
                    result.contexts.append(context)
                }
            case "--weights":
                result.weights = try stream.requireValue(for: token)
            case "--cache-dir":
                result.cacheDir = try stream.requireValue(for: token)
            case "--restart-cycles":
                result.restartCycles = try positiveInt(
                    stream.requireValue(for: token), flag: token
                )
            case "--speculative":
                result.speculative = try enumerated(
                    SpeculativeMode.self, stream.requireValue(for: token), flag: token
                )
            case "--batch":
                result.batch = try batchValue(stream.requireValue(for: token))
            case "--concurrency":
                result.concurrency = try positiveInt(
                    stream.requireValue(for: token), flag: token
                )
            case "--generate":
                result.generate = try positiveInt(
                    stream.requireValue(for: token), flag: token
                )
            case "--agent-turn":
                result.agentTurn = true
            case "--prefix":
                result.prefix = try positiveInt(stream.requireValue(for: token), flag: token)
            case "--thinking":
                result.thinking = try positiveInt(stream.requireValue(for: token), flag: token)
            case "--greedy-agreement":
                result.greedyAgreement = true
            case "--restore":
                result.restore = try stream.requireValue(for: token)
            case "--kv-format":
                result.kvFormat = try enumerated(
                    KvFormat.self, stream.requireValue(for: token), flag: token
                )
            default:
                throw CLIError(unexpected(token, command: "bench"))
            }
            stream.advance()
        }

        return result
    }

    // MARK: serve

    /// Parse `splosh serve`'s arguments.
    public static func parseServe(_ tokens: [String]) throws -> ServeArguments {
        var result = ServeArguments()
        var stream = TokenStream(tokens)

        while let token = stream.current {
            switch token {
            case "--help", "-h":
                result.help = true
            case "--config":
                result.config = try stream.requireValue(for: token)
            case "--port":
                result.port = try portValue(stream.requireValue(for: token))
            case "--model":
                result.model = try modelID(stream.requireValue(for: token), flag: token)
            case "--echo":
                result.echo = true
            case "--restart":
                result.restart = true
            case "--takeover":
                result.takeover = true
            case "--no-open":
                result.noOpen = true
            default:
                throw CLIError(unexpected(token, command: "serve"))
            }
            stream.advance()
        }

        return result
    }

    // MARK: models

    /// Parse `splosh models`'s arguments.
    public static func parseModels(_ tokens: [String]) throws -> ModelsArguments {
        var result = ModelsArguments()
        var stream = TokenStream(tokens)

        while let token = stream.current {
            switch token {
            case "--help", "-h":
                result.help = true
            case "--config":
                result.config = try stream.requireValue(for: token)
            case "--port":
                result.port = try portValue(stream.requireValue(for: token))
            case "--json":
                result.json = true
            case "--load":
                result.load = try modelID(stream.requireValue(for: token), flag: token)
            default:
                throw CLIError(unexpected(token, command: "models"))
            }
            stream.advance()
        }

        return result
    }

    // MARK: doctor

    /// Parse `splosh doctor`'s arguments. `doctor` takes no arguments, so `--help` is the only
    /// accepted token (rev4 §6 M0.9: "`doctor` with no required arguments").
    public static func parseDoctor(_ tokens: [String]) throws -> DoctorArguments {
        var result = DoctorArguments()
        for token in tokens {
            switch token {
            case "--help", "-h":
                result.help = true
            default:
                throw CLIError(unexpected(token, command: "doctor"))
            }
        }
        return result
    }

    // MARK: - Value domains

    /// `--context` accepts one value **or a comma list** drawn from `{2K, 8K, 32K, 128K}`
    /// (rev4 §6 M0.9). Duplicates collapse; an empty element is rejected.
    public static func contextList(_ raw: String) throws -> [ContextLength] {
        let parts = raw.split(separator: ",", omittingEmptySubsequences: false).map(String.init)
        var values: [ContextLength] = []
        for part in parts {
            let parsed = try enumerated(ContextLength.self, part, flag: "--context")
            if !values.contains(parsed) {
                values.append(parsed)
            }
        }
        return values
    }

    /// `--batch` is an integer in `1...16` (rev4 §6 M0.9).
    public static func batchValue(_ raw: String) throws -> Int {
        guard let value = Int(raw), (1...16).contains(value) else {
            throw CLIError("--batch accepts an integer in 1...16; got '\(raw)'")
        }
        return value
    }

    /// `--port` is a TCP port.
    public static func portValue(_ raw: String) throws -> Int {
        guard let value = Int(raw), (1...65_535).contains(value) else {
            throw CLIError("--port accepts an integer in 1...65535; got '\(raw)'")
        }
        return value
    }

    /// `--model` and `--load` name a model as splosh.toml registers one (`model.<id>`).
    public static func modelID(_ raw: String, flag: String) throws -> String {
        guard ModelEntry.isValid(id: raw) else {
            throw CLIError("\(flag) accepts a model id of letters, digits, '.', '_' and '-'; got '\(raw)'")
        }
        return raw
    }

    // MARK: - Helpers

    static func positiveInt(_ raw: String, flag: String) throws -> Int {
        guard let value = Int(raw), value > 0 else {
            throw CLIError("\(flag) accepts a positive integer; got '\(raw)'")
        }
        return value
    }

    static func enumerated<T: RawRepresentable & CaseIterable & Sendable>(
        _ type: T.Type, _ raw: String, flag: String
    ) throws -> T where T.RawValue == String {
        guard let value = T(rawValue: raw) else {
            let allowed = T.allCases.map(\.rawValue).joined(separator: ", ")
            throw CLIError("\(flag) accepts one of \(allowed); got '\(raw)'")
        }
        return value
    }

    static func unexpected(_ token: String, command: String) -> String {
        token.hasPrefix("-")
            ? "unknown flag '\(token)' for command '\(command)'"
            : "unexpected argument '\(token)' for command '\(command)'"
    }
}

/// A minimal positional cursor over an argument vector.
struct TokenStream {
    private let tokens: [String]
    private var index = 0

    init(_ tokens: [String]) {
        self.tokens = tokens
    }

    var current: String? {
        index < tokens.count ? tokens[index] : nil
    }

    mutating func advance() {
        index += 1
    }

    /// Consume and return the token after `flag`, or throw when it is missing.
    mutating func requireValue(for flag: String) throws -> String {
        advance()
        guard index < tokens.count else {
            throw CLIError("\(flag) requires a value")
        }
        return tokens[index]
    }
}

exit(SploshCLI.run(arguments: Array(CommandLine.arguments.dropFirst())))
