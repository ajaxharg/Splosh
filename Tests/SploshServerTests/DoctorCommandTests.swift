// DoctorCommandTests.swift — SploshServerTests.
// Owner: M0.10. Contract: IMPLEMENTATION-PLAN.md §4.2 and M0-WORK-ORDER.md M0.10.

import Testing
import SploshCLI

@Suite("DoctorCommandTests")
struct DoctorCommandTests {
    struct FakeProbe: DoctorProbe {
        var workspaceRoot: String = "/tmp/splosh-doctor-test"
        var unavailableTools: Set<String> = []
        var writablePaths = true
        var freeBytes: Int64 = 100 * 1024 * 1024 * 1024
        var freePort = true
        var assetBytes: Int64?
        var artifacts: Int64 = 0

        func run(_ executable: String, _ arguments: [String]) -> DoctorProcessResult? {
            if executable == "/usr/bin/xcrun", arguments.first == "--find", unavailableTools.contains(arguments.last ?? "") { return nil }
            if executable == "/usr/bin/swift", arguments.contains("dump-package") { return .init(exitCode: 0, stdout: "{}") }
            if executable == "/usr/bin/sw_vers" { return .init(exitCode: 0, stdout: "27.0\n") }
            if executable == "/usr/bin/uname" { return .init(exitCode: 0, stdout: "arm64\n") }
            if executable == "/usr/bin/xcodebuild" { return .init(exitCode: 0, stdout: "Xcode 27.0\n") }
            if executable == "/usr/bin/env" && arguments.first == "swift" { return .init(exitCode: 0, stdout: "Apple Swift version 6.4\n") }
            if executable == "/usr/bin/env" && arguments.first == "python3" { return .init(exitCode: 0) }
            if executable == "/usr/bin/xcrun" && arguments.contains("--show-sdk-version") { return .init(exitCode: 0, stdout: "27.0\n") }
            if executable == "/usr/bin/xcrun" && arguments.contains("metal") { return .init(exitCode: 0) }
            if executable == "/usr/bin/xcrun" && arguments.contains("metallib") { return .init(exitCode: 0) }
            return .init(exitCode: 0)
        }
        func writable(_ path: String) -> Bool { writablePaths }
        func ensureDirectory(_ path: String) -> Bool { writablePaths }
        func freeDiskBytes(at path: String) -> Int64? { freeBytes }
        func portIsFree(_ port: Int) -> Bool { freePort }
        func environment(_ key: String) -> String? { nil }
        func requiredAssetBytes() -> Int64? { assetBytes }
        func artifactBytes() -> Int64 { artifacts }
        func write(_ path: String, _ contents: String) -> Bool { writablePaths }
        func remove(_ path: String) -> Bool { true }
    }

    private func result(_ probe: FakeProbe) -> [DoctorCheckResult] { Doctor(probe: probe).results() }

    @Test("unavailable Metal tools fail with actionable remediation")
    func unavailableToolsFail() {
        let checks = result(FakeProbe(unavailableTools: ["metal"]))
        let check = checks.first { $0.id == "metal-tools" }
        #expect(check?.status == .fail)
        #expect(check?.observed.contains("metal") == true)
        #expect(check?.remediation.contains("Xcode") == true)
    }

    @Test("occupied port fails without crashing")
    func occupiedPortFails() {
        let checks = result(FakeProbe(freePort: false))
        #expect(checks.first { $0.id == "port-8091" }?.status == .fail)
    }

    @Test("unwritable paths fail with remediation")
    func unwritablePathsFail() {
        let checks = result(FakeProbe(writablePaths: false))
        let check = checks.first { $0.id == "writable-paths" }
        #expect(check?.status == .fail)
        #expect(check?.remediation.isEmpty == false)
    }

    @Test("low disk fails both M0/M1 and asset-plus-reserve rules")
    func insufficientDiskFailsBothRules() {
        let checks = result(FakeProbe(freeBytes: 1 * 1024 * 1024 * 1024, assetBytes: 100 * 1024 * 1024 * 1024))
        #expect(checks.first { $0.id == "disk-m0m1" }?.status == .fail)
        #expect(checks.first { $0.id == "disk-asset-phase" }?.status == .fail)
        #expect(checks.first { $0.id == "disk-asset-phase" }?.observed.contains("reserve") == true)
    }

    @Test("doctor help and invalid arguments are handled")
    func argumentHandling() {
        #expect(DoctorCommand.run(["--help"], probe: FakeProbe()) == 0)
        #expect(DoctorCommand.run(["--unknown"], probe: FakeProbe()) != 0)
    }
}
