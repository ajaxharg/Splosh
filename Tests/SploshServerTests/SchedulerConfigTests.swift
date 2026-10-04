import Foundation
import Testing
import SploshCLI
import SploshRuntime

@Suite("Scheduler and Serve config")
struct SchedulerConfigTests {
    @Test("single request admission is exclusive and release is idempotent")
    func schedulerLease() async {
        let scheduler = SingleRequestScheduler()
        let first = await scheduler.tryAdmit()
        #expect(first != nil)
        let second = await scheduler.tryAdmit()
        #expect(second == nil)
        first?.release()
        first?.release()
        try? await Task.sleep(nanoseconds: 10_000_000)
        let occupiedAfterRelease = await scheduler.isOccupied
        #expect(occupiedAfterRelease == false)
        let third = await scheduler.tryAdmit()
        #expect(third != nil)
    }

    @Test("cancellation-style lease release leaves scheduler reusable")
    func cancellationReleasesResource() async {
        let scheduler = SingleRequestScheduler()
        let lease = await scheduler.admit()
        #expect(lease != nil)
        lease?.cancel()
        try? await Task.sleep(nanoseconds: 10_000_000)
        #expect(await scheduler.activeCount == 0)
        #expect(await scheduler.admit() != nil)
    }

    @Test("serve config defaults, parses values, and permits CLI override")
    func serveConfig() throws {
        let path = FileManager.default.temporaryDirectory.appendingPathComponent("splosh-config-\(UUID().uuidString).toml")
        try "port = 8099\ncontextWindow = 4096\nweightsPath = \"/tmp/weights\"\n".write(to: path, atomically: true, encoding: .utf8)
        defer { try? FileManager.default.removeItem(at: path) }
        let config = try ServeConfig.load(path: path.path)
        #expect(config == ServeConfig(port: 8099, weightsPath: "/tmp/weights", contextWindow: 4096))
        let defaults = try ServeConfig.load(path: path.deletingLastPathComponent().appendingPathComponent("missing.toml").path)
        #expect(defaults == ServeConfig())
        #expect(try CLIArguments.parseServe(["--config", path.path, "--port", "8100"]).port == 8100)
        #expect(try ServeConfig.resolve(path: path.path, cliPort: 8100).port == 8100)
        #expect(try ServeConfig.resolve(path: path.path, cliPort: nil).contextWindow == 4096)
    }

    @Test("invalid known config values fail closed")
    func invalidServeConfig() throws {
        let cases = [
            "port = 0", "port = 65536", "port = nope",
            "contextWindow = 0", "contextWindow = -1", "contextWindow = nope",
            "port = 8091\nmalformed"
        ]
        for contents in cases {
            let path = FileManager.default.temporaryDirectory.appendingPathComponent("splosh-invalid-\(UUID().uuidString).toml")
            defer { try? FileManager.default.removeItem(at: path) }
            try contents.write(to: path, atomically: true, encoding: .utf8)
            #expect(throws: Error.self) { _ = try ServeConfig.load(path: path.path) }
        }
    }
}
