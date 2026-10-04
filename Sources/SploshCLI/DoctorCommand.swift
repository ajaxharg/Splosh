// DoctorCommand.swift — SploshCLI.
// Owner: M0.10. Contract: IMPLEMENTATION-PLAN.md §4.2 and M0-WORK-ORDER.md M0.10.

import Foundation
import Darwin
import Metal
import SploshCore

public enum DoctorStatus: String, Sendable, Equatable {
    case pass = "PASS"
    case fail = "FAIL"
    case deferred = "DEFERRED"
}

public struct DoctorCheckResult: Sendable, Equatable {
    public let id: String
    public let status: DoctorStatus
    public let observed: String
    public let required: String
    public let remediation: String
    public init(id: String, status: DoctorStatus, observed: String, required: String, remediation: String) {
        self.id = id; self.status = status; self.observed = observed; self.required = required; self.remediation = remediation
    }
    public var line: String { "\(status.rawValue) \(id) \(observed) \(required) \(remediation)" }
}

public struct DoctorProcessResult: Sendable, Equatable {
    public let exitCode: Int32
    public let stdout: String
    public let stderr: String
    public init(exitCode: Int32, stdout: String = "", stderr: String = "") {
        self.exitCode = exitCode; self.stdout = stdout; self.stderr = stderr
    }
}

public protocol DoctorProbe: Sendable {
    var workspaceRoot: String { get }
    func run(_ executable: String, _ arguments: [String]) -> DoctorProcessResult?
    func writable(_ path: String) -> Bool
    func ensureDirectory(_ path: String) -> Bool
    func freeDiskBytes(at path: String) -> Int64?
    func portIsFree(_ port: Int) -> Bool
    func environment(_ key: String) -> String?
    func requiredAssetBytes() -> Int64?
    func artifactBytes() -> Int64
    func write(_ path: String, _ contents: String) -> Bool
    func remove(_ path: String) -> Bool
}

public struct SystemDoctorProbe: DoctorProbe {
    public let workspaceRoot: String
    public init(workspaceRoot: String = FileManager.default.currentDirectoryPath) { self.workspaceRoot = workspaceRoot }
    public func run(_ executable: String, _ arguments: [String]) -> DoctorProcessResult? {
        let p = Process(); let out = Pipe(); let err = Pipe()
        p.executableURL = URL(fileURLWithPath: executable); p.arguments = arguments; p.standardOutput = out; p.standardError = err
        do { try p.run(); p.waitUntilExit() } catch { return nil }
        return DoctorProcessResult(exitCode: p.terminationStatus,
            stdout: String(data: out.fileHandleForReading.readDataToEndOfFile(), encoding: .utf8) ?? "",
            stderr: String(data: err.fileHandleForReading.readDataToEndOfFile(), encoding: .utf8) ?? "")
    }
    public func writable(_ path: String) -> Bool {
        let fm = FileManager.default; let url = URL(fileURLWithPath: path)
        if !fm.fileExists(atPath: path) { return ensureDirectory(path) && fm.isWritableFile(atPath: path) }
        return fm.isWritableFile(atPath: path)
    }
    public func ensureDirectory(_ path: String) -> Bool {
        do { try FileManager.default.createDirectory(atPath: path, withIntermediateDirectories: true); return true } catch { return false }
    }
    public func freeDiskBytes(at path: String) -> Int64? {
        guard let bytes = try? URL(fileURLWithPath: path).resourceValues(forKeys: [.volumeAvailableCapacityForImportantUsageKey]).volumeAvailableCapacityForImportantUsage else { return nil }
        return Int64(bytes)
    }
    public func portIsFree(_ port: Int) -> Bool {
        let fd = socket(AF_INET, SOCK_STREAM, 0); guard fd >= 0 else { return false }; defer { close(fd) }
        var address = sockaddr_in(); address.sin_len = UInt8(MemoryLayout<sockaddr_in>.size); address.sin_family = sa_family_t(AF_INET); address.sin_port = in_port_t(port).bigEndian; address.sin_addr = in_addr(s_addr: inet_addr("127.0.0.1"))
        return withUnsafePointer(to: &address) { $0.withMemoryRebound(to: sockaddr.self, capacity: 1) { Darwin.bind(fd, $0, socklen_t(MemoryLayout<sockaddr_in>.size)) == 0 } }
    }
    public func environment(_ key: String) -> String? { ProcessInfo.processInfo.environment[key] }
    public func requiredAssetBytes() -> Int64? {
        let path = URL(fileURLWithPath: workspaceRoot).appendingPathComponent("inputs/asset-manifest.json")
        guard let data = try? Data(contentsOf: path), let object = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any], let value = object["required_asset_bytes"] as? NSNumber else { return nil }
        return value.int64Value
    }
    public func artifactBytes() -> Int64 {
        let path = URL(fileURLWithPath: workspaceRoot).appendingPathComponent("artifacts").path
        guard let e = FileManager.default.enumerator(atPath: path) else { return 0 }
        return e.compactMap { item -> Int64? in
            let p = URL(fileURLWithPath: path).appendingPathComponent(item as! String).path
            return (try? FileManager.default.attributesOfItem(atPath: p)[.size] as? NSNumber)?.int64Value
        }.reduce(0, +)
    }
    public func write(_ path: String, _ contents: String) -> Bool { do { try contents.data(using: .utf8)!.write(to: URL(fileURLWithPath: path)); return true } catch { return false } }
    public func remove(_ path: String) -> Bool { (try? FileManager.default.removeItem(atPath: path)) != nil }
}

public struct Doctor {
    public let probe: any DoctorProbe
    public init(probe: any DoctorProbe) { self.probe = probe }
    public func results() -> [DoctorCheckResult] {
        var r: [DoctorCheckResult] = []
        func add(_ id: String, _ status: DoctorStatus, _ observed: String, _ required: String, _ fix: String) { r.append(.init(id: id, status: status, observed: observed, required: required, remediation: fix)) }
        let uname = probe.run("/usr/bin/uname", ["-m"])?.stdout.trimmingCharacters(in: .whitespacesAndNewlines) ?? "unavailable"
        add("arch-arm64", uname == "arm64" ? .pass : .fail, uname, "arm64", "run on Apple silicon")
        let mac = probe.run("/usr/bin/sw_vers", ["-productVersion"])?.stdout.trimmingCharacters(in: .whitespacesAndNewlines) ?? "unavailable"
        add("macos-27", mac.hasPrefix("27.") ? .pass : .fail, mac, "27.x", "install/use macOS 27")
        let swift = probe.run("/usr/bin/env", ["swift", "--version"])?.stdout ?? probe.run("/usr/bin/swift", ["--version"])?.stdout ?? "unavailable"
        add("swift-6.4", swift.contains("Swift version: 6.4") || swift.contains("Swift version 6.4") ? .pass : .fail, swift.split(separator: "\n").first.map(String.init) ?? "unavailable", "Swift 6.4", "select Swift 6.4 toolchain")
        let xcode = probe.run("/usr/bin/xcodebuild", ["-version"])?.stdout ?? "unavailable"
        let sdk = probe.run("/usr/bin/xcrun", ["--sdk", "macosx", "--show-sdk-version"])?.stdout.trimmingCharacters(in: .whitespacesAndNewlines) ?? "unavailable"
        add("xcode-sdk-27", xcode.contains("Xcode 27") && sdk.hasPrefix("27." ) ? .pass : .fail, "Xcode/SDK \(sdk)", "Xcode 27 / SDK 27.x", "select Xcode 27 and SDK 27")
        var missing: [String] = []
        for tool in ["metal", "metallib", "metal-nm"] { if probe.run("/usr/bin/xcrun", ["--find", tool]) == nil { missing.append(tool) } }
        add("metal-tools", missing.isEmpty ? .pass : .fail, missing.isEmpty ? "metal metallib metal-nm" : missing.joined(separator: ","), "all three available", "install/select Xcode Metal tools")
        let probeDir = URL(fileURLWithPath: probe.workspaceRoot).appendingPathComponent(".build/doctor-probe"); _ = probe.ensureDirectory(probeDir.path)
        let src = probeDir.appendingPathComponent("probe.metal").path, air = probeDir.appendingPathComponent("probe.air").path, lib = probeDir.appendingPathComponent("probe.metallib").path
        let shader = "#include <metal_stdlib>\nusing namespace metal; kernel void doctor_probe(uint3 id [[thread_position_in_grid]]) {}\n"
        let cache = URL(fileURLWithPath: probe.workspaceRoot).appendingPathComponent(".build/metal-module-cache").path
        let wrote = probe.write(src, shader); let compiled = wrote && probe.run("/usr/bin/xcrun", ["-sdk", "macosx", "metal", "-std=metal4.0", "-fmodules-cache-path=\(cache)", "-c", src, "-o", air])?.exitCode == 0 && probe.run("/usr/bin/xcrun", ["-sdk", "macosx", "metallib", air, "-o", lib])?.exitCode == 0
        _ = probe.remove(src); _ = probe.remove(air); _ = probe.remove(lib)
        add("metal-module-cache-compile", compiled ? .pass : .fail, compiled ? "compiled with \(cache)" : "compile failed", "real Metal compile", "use an in-workspace -fmodules-cache-path")
        guard let device = Device.shared else { add("metal4-family", .fail, "no Metal device", "Metal4", "use a Metal 4 GPU"); add("gpu-families", .fail, "no Metal device", "Metal4 + Apple10", "use the target GPU"); add("threadgroup-memory", .fail, "no Metal device", ">=32768", "use a supported GPU"); add("working-set", .fail, "no Metal device", ">0", "use a supported GPU"); return finish(r) }
        add("metal4-family", device.supportsMetal4 ? .pass : .fail, device.supportsMetal4 ? "supported" : "unsupported", "MTLGPUFamilyMetal4 (5002)", "use a Metal 4 GPU")
        let families = device.supports(MTLGPUFamily(rawValue: 5002)!) && device.supports(MTLGPUFamily(rawValue: 1010)!)
        add("gpu-families", families ? .pass : .fail, families ? "Metal4 + Apple10" : "required family missing", "Metal4 + Apple10", "use the target M5 GPU")
        add("threadgroup-memory", device.maxThreadgroupMemoryLength >= 32768 ? .pass : .fail, "\(device.maxThreadgroupMemoryLength) bytes", ">=32768 bytes", "use a GPU with required threadgroup memory")
        add("working-set", device.recommendedMaxWorkingSetSize > 0 ? .pass : .fail, "\(device.recommendedMaxWorkingSetSize) bytes", ">0 bytes", "use a Metal device exposing a working-set budget")
        let paths = [probe.workspaceRoot, URL(fileURLWithPath: probe.workspaceRoot).appendingPathComponent(".build").path, cache, URL(fileURLWithPath: probe.workspaceRoot).appendingPathComponent("inputs").path, URL(fileURLWithPath: probe.workspaceRoot).appendingPathComponent("artifacts").path]
        let pathsOK = paths.allSatisfy { probe.ensureDirectory($0) && probe.writable($0) }
        add("writable-paths", pathsOK ? .pass : .fail, pathsOK ? "workspace/build/cache/inputs/artifacts writable" : "one or more paths unwritable", "all required paths writable", "create/chmod the required workspace paths")
        let swiftpm = probe.run("/usr/bin/swift", ["package", "--disable-sandbox", "dump-package"])?.exitCode == 0
        add("swiftpm-functional", swiftpm ? .pass : .fail, swiftpm ? "dump-package exit 0" : "dump-package failed", "SwiftPM usable", "run SwiftPM with --disable-sandbox")
        add("port-8091", probe.portIsFree(8091) ? .pass : .fail, probe.portIsFree(8091) ? "free" : "occupied", "127.0.0.1:8091 free", "stop the listener or choose the configured port")
        let py = probe.run("/usr/bin/env", ["python3", URL(fileURLWithPath: probe.workspaceRoot).appendingPathComponent("tools/fetch_inputs.py").path, "--help"])?.exitCode == 0
        add("python-fetch-tool", py ? .pass : .fail, py ? "python3 + fetch_inputs.py --help" : "fetch tool unavailable", "Python 3 and locked fetch tool", "install Python 3 and use requirements.lock")
        let free = probe.freeDiskBytes(at: probe.workspaceRoot) ?? -1, tenGiB: Int64 = 10 * 1024 * 1024 * 1024
        add("disk-m0m1", free >= tenGiB ? .pass : .fail, "\(free) bytes free", ">=10737418240 bytes", "free at least 10 GiB")
        if let required = probe.requiredAssetBytes() { let reserve: Int64 = 20 * 1024 * 1024 * 1024; let need = required + reserve + probe.artifactBytes(); add("disk-asset-phase", free >= need ? .pass : .fail, "free=\(free), required=\(required)+reserve=\(reserve)+artifacts=\(probe.artifactBytes())", ">=\(need) bytes", "free the required asset bytes plus 20 GiB reserve") } else { add("disk-asset-phase", .deferred, "asset manifest absent", "M1.7 asset manifest", "M1.7 fetch_inputs.py produces inputs/asset-manifest.json") }
        add("hf-token", .deferred, "remote assets not requested", "M1.7 remote fetch", "set HF_TOKEN before requesting remote assets")
        add("cache-dir", .deferred, "cacheDir unset; cache is off", "M5.5 configured cache", "configure cacheDir only when durable cache is enabled")
        add("optional-profilers", .deferred, "non-blocking until used", "M3.3 xctrace / M7.2 powermetrics", "install privileges when profiling/soak requires them")
        return finish(r)
    }
    private func finish(_ r: [DoctorCheckResult]) -> [DoctorCheckResult] { r }
}

public enum DoctorCommand {
    public static func run(_ arguments: [String], probe: any DoctorProbe = SystemDoctorProbe()) -> Int32 {
        do { let parsed = try CLIArguments.parseDoctor(arguments); if parsed.help { print(SploshCLI.usage(for: .doctor)); return 0 } } catch let e as CLIError { SploshCLI.writeStderr("doctor: \(e.message)\n"); return 2 } catch { return 2 }
        let results = Doctor(probe: probe).results(); results.forEach { print($0.line) }
        return results.contains { $0.status == .fail && $0.id != "disk-asset-phase" } || results.contains { $0.status == .fail && $0.id == "disk-asset-phase" } ? 1 : 0
    }
}
