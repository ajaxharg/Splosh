import Foundation
import Testing
import SploshCLI

@Suite("OracleCommandTests")
struct OracleCommandTests {
    private func temporaryDirectory() throws -> URL {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("splosh-oracle-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        return url
    }

    @Test("no args and missing prompt fail closed")
    func missingPrompt() {
        #expect(OracleCommand.run([]) != ExitStatus.ok)
        #expect(OracleCommand.run(["--weights", "/tmp/no-such-weights"]) != ExitStatus.ok)
    }

    @Test("nonexistent prompt fails before asset validation")
    func nonexistentPrompt() throws {
        let root = try temporaryDirectory()
        #expect(OracleCommand.run(["--prompt", root.appendingPathComponent("missing.txt").path]) != ExitStatus.ok)
    }

    @Test("valid prompt with missing weights root is blocked")
    func missingWeights() throws {
        let root = try temporaryDirectory()
        let prompt = root.appendingPathComponent("prompt.txt")
        try Data("hello".utf8).write(to: prompt)
        #expect(OracleCommand.run(["--prompt", prompt.path, "--weights", root.appendingPathComponent("weights").path]) != ExitStatus.ok)
    }

    @Test("valid prompt with a root lacking pinned q4 index is blocked")
    func missingQ4Index() throws {
        let root = try temporaryDirectory()
        let prompt = root.appendingPathComponent("prompt.txt")
        try Data("hello".utf8).write(to: prompt)
        let weights = root.appendingPathComponent("weights")
        try FileManager.default.createDirectory(at: weights, withIntermediateDirectories: true)
        #expect(OracleCommand.run(["--prompt", prompt.path, "--weights", weights.path]) != ExitStatus.ok)
    }

    @Test("unknown flags and missing values are rejected")
    func malformedArguments() {
        #expect(OracleCommand.run(["--unknown"]) == ExitStatus.usage)
        #expect(OracleCommand.run(["--prompt"]) == ExitStatus.usage)
        #expect(OracleCommand.run(["--weights"]) == ExitStatus.usage)
    }

    @Test("help succeeds without assets")
    func help() {
        #expect(OracleCommand.run(["--help"]) == ExitStatus.ok)
    }
}
