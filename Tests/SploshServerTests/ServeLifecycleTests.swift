import Foundation
import Testing
#if canImport(FoundationNetworking)
import FoundationNetworking
#endif
#if os(macOS)
import Darwin
#elseif os(Linux)
import Glibc
#endif

@Suite("Serve lifecycle")
struct ServeLifecycleTests {
    @Test("serve exits on SIGTERM and immediately rebinds the same port")
    func gracefulShutdownAndRebind() async throws {
        let port = try unusedLoopbackPort()
        let executable = try locateExecutable()
        var first: Process?
        var second: Process?
        defer {
            terminateAndReap(first, label: "first")
            terminateAndReap(second, label: "second")
        }

        do {
            first = try launch(executable, port: port)
            try await waitForHealth(port: port, process: first!, timeout: 120)
            try sendSIGTERM(first!, label: "first")
            try waitForExit(first!, timeout: 30, label: "first")
            #expect(first!.terminationStatus == 0, "first serve exited with status \(first!.terminationStatus); stderr: \(diagnostics(first!))")

            second = try launch(executable, port: port)
            try await waitForHealth(port: port, process: second!, timeout: 120)
            try sendSIGTERM(second!, label: "second")
            try waitForExit(second!, timeout: 30, label: "second")
            #expect(second!.terminationStatus == 0, "second serve exited with status \(second!.terminationStatus); stderr: \(diagnostics(second!))")
        } catch {
            let firstDiagnostics = first.map { diagnostics($0) } ?? "not launched"
            let secondDiagnostics = second.map { diagnostics($0) } ?? "not launched"
            Issue.record("serve lifecycle failed: \(error)\nfirst: \(firstDiagnostics)\nsecond: \(secondDiagnostics)")
            throw error
        }
    }

    private func locateExecutable() throws -> URL {
        let root = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let candidates = [
            root.appendingPathComponent(".build/debug/splosh"),
            root.appendingPathComponent(".build/arm64-apple-macosx/debug/splosh")
        ]
        if let found = candidates.first(where: { FileManager.default.isExecutableFile(atPath: $0.path) }) {
            return found
        }
        throw LifecycleError("built splosh executable not found; run swift build before this suite (looked in \(candidates.map(\.path)) )")
    }

    private func unusedLoopbackPort() throws -> Int {
        let fd = socket(AF_INET, SOCK_STREAM, 0)
        guard fd >= 0 else { throw LifecycleError("socket() failed") }
        defer { _ = close(fd) }
        var address = sockaddr_in()
        address.sin_len = UInt8(MemoryLayout<sockaddr_in>.size)
        address.sin_family = sa_family_t(AF_INET)
        address.sin_port = 0
        address.sin_addr = in_addr(s_addr: inet_addr("127.0.0.1"))
        let bound = withUnsafePointer(to: &address) {
            $0.withMemoryRebound(to: sockaddr.self, capacity: 1) { bind(fd, $0, socklen_t(MemoryLayout<sockaddr_in>.size)) }
        }
        guard bound == 0 else { throw LifecycleError("bind(port 0) failed") }
        var length = socklen_t(MemoryLayout<sockaddr_in>.size)
        let result = withUnsafeMutablePointer(to: &address) {
            $0.withMemoryRebound(to: sockaddr.self, capacity: 1) { getsockname(fd, $0, &length) }
        }
        guard result == 0 else { throw LifecycleError("getsockname() failed") }
        return Int(UInt16(bigEndian: address.sin_port))
    }

    private func launch(_ executable: URL, port: Int) throws -> Process {
        let process = Process()
        process.executableURL = executable
        process.arguments = ["serve", "--port", String(port)]
        process.standardOutput = Pipe()
        process.standardError = Pipe()
        try process.run()
        return process
    }

    private func waitForHealth(port: Int, process: Process, timeout: TimeInterval) async throws {
        let deadline = Date().addingTimeInterval(timeout)
        let url = URL(string: "http://127.0.0.1:\(port)/health")!
        while Date() < deadline {
            if !process.isRunning { throw LifecycleError("serve exited before /health; \(diagnostics(process))") }
            do {
                var request = URLRequest(url: url)
                request.timeoutInterval = 0.5
                let (data, response) = try await URLSession.shared.data(for: request)
                if (response as? HTTPURLResponse)?.statusCode == 200, String(data: data, encoding: .utf8) == "ok" { return }
            } catch { /* startup race; retry until the bounded deadline */ }
            try await Task.sleep(for: .milliseconds(100))
        }
        throw LifecycleError("/health did not become ready within \(timeout)s; \(diagnostics(process))")
    }

    private func sendSIGTERM(_ process: Process, label: String) throws {
        guard process.isRunning else { throw LifecycleError("\(label) process was not running before SIGTERM") }
        guard kill(process.processIdentifier, SIGTERM) == 0 else { throw LifecycleError("SIGTERM failed for \(label)") }
    }

    private func waitForExit(_ process: Process, timeout: TimeInterval, label: String) throws {
        let deadline = Date().addingTimeInterval(timeout)
        while process.isRunning && Date() < deadline { Thread.sleep(forTimeInterval: 0.05) }
        guard !process.isRunning else { throw LifecycleError("\(label) process did not exit within \(timeout)s; forcing cleanup") }
    }

    private func terminateAndReap(_ process: Process?, label: String) {
        guard let process else { return }
        if process.isRunning {
            _ = kill(process.processIdentifier, SIGTERM)
            let deadline = Date().addingTimeInterval(1)
            while process.isRunning && Date() < deadline { Thread.sleep(forTimeInterval: 0.05) }
            if process.isRunning { _ = kill(process.processIdentifier, SIGKILL) }
        }
        if process.isRunning { process.waitUntilExit() }
        _ = label // retained for useful debugger labels without emitting test output
    }

    private func diagnostics(_ process: Process) -> String {
        let output = (process.standardOutput as? Pipe)?.fileHandleForReading.readDataToEndOfFile() ?? Data()
        let error = (process.standardError as? Pipe)?.fileHandleForReading.readDataToEndOfFile() ?? Data()
        let stdout = String(data: output, encoding: .utf8) ?? "<non-UTF8>"
        let stderr = String(data: error, encoding: .utf8) ?? "<non-UTF8>"
        return "stdout=\(stdout) stderr=\(stderr)"
    }

    private struct LifecycleError: Error, CustomStringConvertible {
        let description: String
        init(_ description: String) { self.description = description }
    }
}
