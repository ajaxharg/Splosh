// CLIArgsTests.swift — SploshServerTests.
//
// Owner: M0.9. Contract source: rev4 §6 M0.9, §4.6 (the command table), §3.1 rule 3 (this test
// target is the one place allowed to import `SploshCLI`).
//
// The suite is named after the file so `swift test --filter CLIArgsTests` -- the filter rev4 §3.2's
// inventory names and §6 M0.9's gate runs -- matches it.
//
// Two committed tables drive it: an argv-to-structure table, and a rejected table that gives **every**
// flag at least one rejected form (rev4 §6 M0.9). Assertions compare whole parsed structures, so a
// flag that parses to the wrong field fails rather than passing on a non-nil check.

import Foundation
import Testing

import SploshCLI

@Suite("CLIArgsTests")
struct CLIArgsTests {
    // MARK: - Tables

    private enum Command {
        case bench
        case serve
        case models
        case doctor
    }

    private struct BenchRow {
        let name: String
        let argv: [String]
        let expected: BenchArguments
    }

    private struct ServeRow {
        let name: String
        let argv: [String]
        let expected: ServeArguments
    }

    private struct ModelsRow {
        let name: String
        let argv: [String]
        let expected: ModelsArguments
    }

    private struct RejectRow {
        let name: String
        let command: Command
        let argv: [String]
        /// A substring the rejection message must contain — usually the offending flag.
        let mentions: String
    }

    private static func bench(_ mutate: (inout BenchArguments) -> Void) -> BenchArguments {
        var value = BenchArguments()
        mutate(&value)
        return value
    }

    private static func serve(_ mutate: (inout ServeArguments) -> Void) -> ServeArguments {
        var value = ServeArguments()
        mutate(&value)
        return value
    }

    private static func models(_ mutate: (inout ModelsArguments) -> Void) -> ModelsArguments {
        var value = ModelsArguments()
        mutate(&value)
        return value
    }

    private static let benchTable: [BenchRow] = [
        BenchRow(name: "no arguments", argv: [], expected: bench { _ in }),
        BenchRow(
            name: "context, single value",
            argv: ["--context", "8K"],
            expected: bench { $0.contexts = [.k8] }
        ),
        BenchRow(
            name: "context, comma list",
            argv: ["--context", "2K,32K,128K"],
            expected: bench { $0.contexts = [.k2, .k32, .k128] }
        ),
        BenchRow(
            name: "context, comma list with a duplicate",
            argv: ["--context", "8K,8K"],
            expected: bench { $0.contexts = [.k8] }
        ),
        BenchRow(
            name: "context, repeated flag accumulates like a comma list",
            argv: ["--context", "2K", "--context", "128K"],
            expected: bench { $0.contexts = [.k2, .k128] }
        ),
        BenchRow(
            name: "speculative off",
            argv: ["--speculative", "off"],
            expected: bench { $0.speculative = .off }
        ),
        BenchRow(
            name: "speculative on",
            argv: ["--speculative", "on"],
            expected: bench { $0.speculative = .on }
        ),
        BenchRow(name: "batch lower bound", argv: ["--batch", "1"], expected: bench { $0.batch = 1 }),
        BenchRow(name: "batch upper bound", argv: ["--batch", "16"], expected: bench { $0.batch = 16 }),
        BenchRow(
            name: "kv-format int8",
            argv: ["--kv-format", "int8"],
            expected: bench { $0.kvFormat = .int8 }
        ),
        BenchRow(
            name: "kv-format q4 is grammatical now, served only if M4.5 lands",
            argv: ["--kv-format", "q4"],
            expected: bench { $0.kvFormat = .q4 }
        ),
        BenchRow(
            name: "weights and cache-dir are paths, not enums",
            argv: ["--weights", "/tmp/w", "--cache-dir", "/tmp/c"],
            expected: bench {
                $0.weights = "/tmp/w"
                $0.cacheDir = "/tmp/c"
            }
        ),
        BenchRow(
            name: "agent-turn with prefix and thinking",
            argv: ["--agent-turn", "--prefix", "100", "--thinking", "50"],
            expected: bench {
                $0.agentTurn = true
                $0.prefix = 100
                $0.thinking = 50
            }
        ),
        BenchRow(
            name: "greedy-agreement with restore",
            argv: ["--greedy-agreement", "--restore", "abc123"],
            expected: bench {
                $0.greedyAgreement = true
                $0.restore = "abc123"
            }
        ),
        BenchRow(
            name: "restart-cycles and concurrency",
            argv: ["--restart-cycles", "3", "--concurrency", "4"],
            expected: bench {
                $0.restartCycles = 3
                $0.concurrency = 4
            }
        ),
        BenchRow(name: "generate", argv: ["--generate", "128"], expected: bench { $0.generate = 128 }),
        BenchRow(
            name: "every flag at once",
            argv: [
                "--context", "2K,8K", "--weights", "/w", "--cache-dir", "/c",
                "--restart-cycles", "2", "--speculative", "on", "--batch", "4",
                "--concurrency", "3", "--generate", "64", "--agent-turn",
                "--prefix", "10", "--thinking", "5", "--greedy-agreement",
                "--restore", "e1", "--kv-format", "int8",
            ],
            expected: bench {
                $0.contexts = [.k2, .k8]
                $0.weights = "/w"
                $0.cacheDir = "/c"
                $0.restartCycles = 2
                $0.speculative = .on
                $0.batch = 4
                $0.concurrency = 3
                $0.generate = 64
                $0.agentTurn = true
                $0.prefix = 10
                $0.thinking = 5
                $0.greedyAgreement = true
                $0.restore = "e1"
                $0.kvFormat = .int8
            }
        ),
        BenchRow(name: "--help", argv: ["--help"], expected: bench { $0.help = true }),
    ]

    private static let serveTable: [ServeRow] = [
        ServeRow(name: "no arguments", argv: [], expected: serve { _ in }),
        ServeRow(
            name: "config and port",
            argv: ["--config", "/tmp/splosh.toml", "--port", "8091"],
            expected: serve {
                $0.config = "/tmp/splosh.toml"
                $0.port = 8091
            }
        ),
        ServeRow(name: "--echo is reserved for M1.4", argv: ["--echo"], expected: serve { $0.echo = true }),
        ServeRow(name: "the model to start on", argv: ["--model", "uq5"], expected: serve { $0.model = "uq5" }),
        ServeRow(
            name: "a restart of the server on a port",
            argv: ["--restart", "--port", "8092"],
            expected: serve {
                $0.restart = true
                $0.port = 8092
            }
        ),
        ServeRow(name: "--help", argv: ["--help"], expected: serve { $0.help = true }),
    ]

    private static let modelsTable: [ModelsRow] = [
        ModelsRow(name: "no arguments", argv: [], expected: models { _ in }),
        ModelsRow(name: "--json", argv: ["--json"], expected: models { $0.json = true }),
        ModelsRow(
            name: "config and port",
            argv: ["--config", "/tmp/splosh.toml", "--port", "8092"],
            expected: models {
                $0.config = "/tmp/splosh.toml"
                $0.port = 8092
            }
        ),
        ModelsRow(name: "a load by hand", argv: ["--load", "ud-q5_k_m.v2"], expected: models { $0.load = "ud-q5_k_m.v2" }),
        ModelsRow(
            name: "every flag at once",
            argv: ["--load", "uq6", "--json", "--port", "9000", "--config", "c.toml"],
            expected: models {
                $0.load = "uq6"
                $0.json = true
                $0.port = 9000
                $0.config = "c.toml"
            }
        ),
        ModelsRow(name: "--help", argv: ["--help"], expected: models { $0.help = true }),
    ]

    private static let rejectedTable: [RejectRow] = [
        // --- bench: one rejected form per declared flag ---
        RejectRow(name: "context: value outside the domain", command: .bench, argv: ["--context", "4K"], mentions: "--context"),
        RejectRow(name: "context: missing value", command: .bench, argv: ["--context"], mentions: "--context"),
        RejectRow(name: "context: empty value", command: .bench, argv: ["--context", ""], mentions: "--context"),
        RejectRow(name: "context: trailing comma", command: .bench, argv: ["--context", "2K,"], mentions: "--context"),
        RejectRow(name: "weights: missing value", command: .bench, argv: ["--weights"], mentions: "--weights"),
        RejectRow(name: "cache-dir: missing value", command: .bench, argv: ["--cache-dir"], mentions: "--cache-dir"),
        RejectRow(name: "restart-cycles: zero", command: .bench, argv: ["--restart-cycles", "0"], mentions: "--restart-cycles"),
        RejectRow(name: "restart-cycles: not a number", command: .bench, argv: ["--restart-cycles", "abc"], mentions: "--restart-cycles"),
        RejectRow(name: "speculative: outside {on, off}", command: .bench, argv: ["--speculative", "maybe"], mentions: "--speculative"),
        RejectRow(name: "batch: below the range", command: .bench, argv: ["--batch", "0"], mentions: "--batch"),
        RejectRow(name: "batch: above the range", command: .bench, argv: ["--batch", "17"], mentions: "--batch"),
        RejectRow(name: "batch: not an integer", command: .bench, argv: ["--batch", "2.5"], mentions: "--batch"),
        RejectRow(name: "concurrency: zero", command: .bench, argv: ["--concurrency", "0"], mentions: "--concurrency"),
        RejectRow(name: "generate: negative", command: .bench, argv: ["--generate", "-5"], mentions: "--generate"),
        RejectRow(name: "agent-turn: takes no value", command: .bench, argv: ["--agent-turn", "yes"], mentions: "'yes'"),
        RejectRow(name: "prefix: zero", command: .bench, argv: ["--prefix", "0"], mentions: "--prefix"),
        RejectRow(name: "thinking: zero", command: .bench, argv: ["--thinking", "0"], mentions: "--thinking"),
        RejectRow(name: "greedy-agreement: takes no value", command: .bench, argv: ["--greedy-agreement", "extra"], mentions: "'extra'"),
        RejectRow(name: "restore: missing value", command: .bench, argv: ["--restore"], mentions: "--restore"),
        RejectRow(name: "kv-format: outside {int8, q4}", command: .bench, argv: ["--kv-format", "fp8"], mentions: "--kv-format"),
        RejectRow(name: "unknown flag", command: .bench, argv: ["--nope"], mentions: "'--nope'"),
        RejectRow(name: "stray positional", command: .bench, argv: ["7K"], mentions: "'7K'"),

        // --- serve ---
        RejectRow(name: "serve config: missing value", command: .serve, argv: ["--config"], mentions: "--config"),
        RejectRow(name: "serve port: zero", command: .serve, argv: ["--port", "0"], mentions: "--port"),
        RejectRow(name: "serve port: above 65535", command: .serve, argv: ["--port", "70000"], mentions: "--port"),
        RejectRow(name: "serve port: not a number", command: .serve, argv: ["--port", "abc"], mentions: "--port"),
        RejectRow(name: "serve echo: takes no value", command: .serve, argv: ["--echo", "extra"], mentions: "'extra'"),
        RejectRow(name: "serve model: missing value", command: .serve, argv: ["--model"], mentions: "--model"),
        RejectRow(name: "serve model: not an id", command: .serve, argv: ["--model", "a/b"], mentions: "--model"),
        RejectRow(name: "serve unknown flag", command: .serve, argv: ["--bogus"], mentions: "'--bogus'"),

        // --- models ---
        RejectRow(name: "models config: missing value", command: .models, argv: ["--config"], mentions: "--config"),
        RejectRow(name: "models port: zero", command: .models, argv: ["--port", "0"], mentions: "--port"),
        RejectRow(name: "models json: takes no value", command: .models, argv: ["--json", "yes"], mentions: "'yes'"),
        RejectRow(name: "models load: missing value", command: .models, argv: ["--load"], mentions: "--load"),
        RejectRow(name: "models load: not an id", command: .models, argv: ["--load", "two words"], mentions: "--load"),
        RejectRow(name: "models unknown flag", command: .models, argv: ["--eject"], mentions: "'--eject'"),
        RejectRow(name: "models stray positional", command: .models, argv: ["uq5"], mentions: "'uq5'"),

        // --- doctor: no arguments at all ---
        RejectRow(name: "doctor: stray positional", command: .doctor, argv: ["extra"], mentions: "'extra'"),
        RejectRow(name: "doctor: unknown flag", command: .doctor, argv: ["--verbose"], mentions: "'--verbose'"),
    ]

    // MARK: - Accepted forms

    @Test("bench: every committed argv parses to the expected structure")
    func benchTableParses() throws {
        for row in Self.benchTable {
            let parsed = try CLIArguments.parseBench(row.argv)
            #expect(parsed == row.expected, "row '\(row.name)': argv \(row.argv) parsed to \(parsed)")
        }
    }

    @Test("serve: every committed argv parses to the expected structure")
    func serveTableParses() throws {
        for row in Self.serveTable {
            let parsed = try CLIArguments.parseServe(row.argv)
            #expect(parsed == row.expected, "row '\(row.name)': argv \(row.argv) parsed to \(parsed)")
        }
    }

    @Test("models: every committed argv parses to the expected structure")
    func modelsTableParses() throws {
        for row in Self.modelsTable {
            let parsed = try CLIArguments.parseModels(row.argv)
            #expect(parsed == row.expected, "row '\(row.name)': argv \(row.argv) parsed to \(parsed)")
        }
    }

    @Test("doctor takes no arguments, and --help is the only accepted token")
    func doctorTakesNoArguments() throws {
        #expect(try CLIArguments.parseDoctor([]) == DoctorArguments())
        #expect(try CLIArguments.parseDoctor(["--help"]).help)
    }

    // MARK: - Rejected forms

    @Test("every flag has at least one rejected form, and the message names it")
    func everyFlagHasARejectedForm() {
        for row in Self.rejectedTable {
            do {
                switch row.command {
                case .bench: _ = try CLIArguments.parseBench(row.argv)
                case .serve: _ = try CLIArguments.parseServe(row.argv)
                case .models: _ = try CLIArguments.parseModels(row.argv)
                case .doctor: _ = try CLIArguments.parseDoctor(row.argv)
                }
                Issue.record("row '\(row.name)': argv \(row.argv) was accepted but must be rejected")
            } catch let error as CLIError {
                #expect(
                    error.message.contains(row.mentions),
                    "row '\(row.name)': message '\(error.message)' does not mention '\(row.mentions)'"
                )
            } catch {
                Issue.record("row '\(row.name)': threw \(error), expected CLIError")
            }
        }
    }

    // MARK: - Convert grammar

    @Test("convert parser is explicit and fail-closed")
    func convertGrammar() throws {
        #expect(try ConvertCommand.parse(["--input", "src", "--out", "dst", "--verify"]) == ConvertArguments(input: "src", output: "dst", verify: true))
        #expect(try ConvertCommand.parse(["--help"]).help)
        for argv in [["--input"], ["--out"]] {
            #expect(throws: CLIError.self) { _ = try ConvertCommand.parse(argv) }
        }
        for argv in [["--bogus"], ["--input", "a", "--input", "b"], ["--out", "a", "--out", "b"], ["--verify", "--verify"]] {
            #expect(throws: CLIError.self) { _ = try ConvertCommand.parse(argv) }
        }
    }

    @Test("convert help parses without requiring assets")
    func convertHelpDoesNotTouchAssets() {
        #expect(ConvertCommand.run(["--help"]) == ExitStatus.ok)
    }

    @Test("convert rejects a source without the q4 index")
    func convertRejectsMissingQ4Index() throws {
        let root = URL(fileURLWithPath: NSTemporaryDirectory()).appendingPathComponent("SploshCLI-\(UUID().uuidString)")
        let output = root.deletingLastPathComponent().appendingPathComponent("SploshCLI-out-\(UUID().uuidString).splw")
        defer {
            try? FileManager.default.removeItem(at: root)
            try? FileManager.default.removeItem(at: output)
        }
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        #expect(ConvertCommand.run(["--input", root.path, "--out", output.path]) != ExitStatus.ok)
        #expect(!FileManager.default.fileExists(atPath: output.path))
    }

    @Test("convert rejects an output whose parent is a regular file")
    func convertRejectsMalformedOutputParent() throws {
        let root = URL(fileURLWithPath: NSTemporaryDirectory()).appendingPathComponent("SploshCLI-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: root) }
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        let fileParent = root.appendingPathComponent("not-a-directory")
        try Data("sentinel".utf8).write(to: fileParent)
        let output = fileParent.appendingPathComponent("out.splw")
        #expect(ConvertCommand.run(["--input", root.path, "--out", output.path]) != ExitStatus.ok)
        #expect(try Data(contentsOf: fileParent) == Data("sentinel".utf8))
    }

    @Test("convert rejects output equal to or inside the source")
    func convertRejectsSourceCollisions() throws {
        let root = URL(fileURLWithPath: NSTemporaryDirectory()).appendingPathComponent("SploshCLI-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: root) }
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        let marker = root.appendingPathComponent("keep.txt")
        try Data("keep".utf8).write(to: marker)
        #expect(ConvertCommand.run(["--input", root.path, "--out", root.path]) != ExitStatus.ok)
        #expect(ConvertCommand.run(["--input", root.path, "--out", root.appendingPathComponent("nested/out.splw").path]) != ExitStatus.ok)
        #expect(try Data(contentsOf: marker) == Data("keep".utf8))
    }

    @Test("convert preserves a pre-existing output after conversion failure")
    func convertPreservesExistingOutputOnFailure() throws {
        let root = URL(fileURLWithPath: NSTemporaryDirectory()).appendingPathComponent("SploshCLI-\(UUID().uuidString)")
        let output = root.deletingLastPathComponent().appendingPathComponent("SploshCLI-existing-\(UUID().uuidString).splw")
        defer {
            try? FileManager.default.removeItem(at: root)
            try? FileManager.default.removeItem(at: output)
        }
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        let sentinel = Data("pre-existing-output".utf8)
        try sentinel.write(to: output)
        #expect(ConvertCommand.run(["--input", root.path, "--out", output.path]) != ExitStatus.ok)
        #expect(try Data(contentsOf: output) == sentinel)
    }

    // MARK: - The domains themselves

    @Test("the value domains are exactly rev4's")
    func valueDomainsAreExact() throws {
        #expect(ContextLength.allCases.map(\.rawValue) == ["2K", "8K", "32K", "128K"])
        #expect(SpeculativeMode.allCases.map(\.rawValue) == ["on", "off"])
        #expect(KvFormat.allCases.map(\.rawValue) == ["int8", "q4"])

        // The commands: rev4 §4.6's six, this plan's `doctor`, and `generate` and `models` since.
        #expect(
            CommandName.allCases.map(\.rawValue)
                == ["doctor", "serve", "models", "convert", "bench", "cache", "oracle", "soak", "generate"]
        )

        // The batch range is closed at both ends (rev4 §6 M0.9).
        let lower = try CLIArguments.batchValue("1")
        let upper = try CLIArguments.batchValue("16")
        #expect(lower == 1)
        #expect(upper == 16)
        #expect(throws: CLIError.self) { _ = try CLIArguments.batchValue("0") }
        #expect(throws: CLIError.self) { _ = try CLIArguments.batchValue("17") }
    }
}
