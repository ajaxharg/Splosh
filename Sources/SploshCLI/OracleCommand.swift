// OracleCommand.swift — SploshCLI.
//
// M2.10 command surface: argument and asset validation is deliberately fail-closed.
// Numerical oracle execution remains unavailable until the real runtime prerequisites exist.

import Foundation
import SploshModel

public struct OracleArguments: Equatable, Sendable {
    public var prompt: String?
    public var weights: String?
    public var help: Bool

    public init(prompt: String? = nil, weights: String? = nil, help: Bool = false) {
        self.prompt = prompt
        self.weights = weights
        self.help = help
    }
}

public enum OracleCommand {
    public static let usage = """
        usage: splosh oracle --prompt <file> [options]

          --prompt <file>   fixed prompt input
          --weights <path>  q4 weight root; default $PWD/splosh-weights/
          --help             show this help
        """

    /// Parse the command's small, explicit grammar without touching assets or runtime state.
    public static func parse(_ tokens: [String]) throws -> OracleArguments {
        var result = OracleArguments()
        var stream = TokenStream(tokens)
        while let token = stream.current {
            switch token {
            case "--help", "-h":
                result.help = true
            case "--prompt":
                result.prompt = try stream.requireValue(for: token)
            case "--weights":
                result.weights = try stream.requireValue(for: token)
            default:
                throw CLIError(token.hasPrefix("-")
                    ? "unknown flag '\(token)' for command 'oracle'"
                    : "unexpected argument '\(token)' for command 'oracle'")
            }
            stream.advance()
        }
        return result
    }

    /// Run validation and fail closed before any oracle or GPU path.
    public static func run(_ arguments: [String]) -> Int32 {
        do {
            let parsed = try parse(arguments)
            if parsed.help {
                print(usage)
                return ExitStatus.ok
            }
            guard let promptPath = parsed.prompt else {
                return blocked("missing required --prompt <file>")
            }
            let promptURL = URL(fileURLWithPath: promptPath)
            let fm = FileManager.default
            guard fm.fileExists(atPath: promptURL.path),
                  (try? promptURL.resourceValues(forKeys: [.isRegularFileKey]).isRegularFile) == true,
                  fm.isReadableFile(atPath: promptURL.path) else {
                return blocked("prompt is missing or unreadable: \(promptPath)")
            }

            let weightsPath = parsed.weights ?? URL(fileURLWithPath: fm.currentDirectoryPath)
                .appendingPathComponent("splosh-weights").path
            let weightsURL = URL(fileURLWithPath: weightsPath)
            guard fm.fileExists(atPath: weightsURL.path),
                  (try? weightsURL.resourceValues(forKeys: [.isDirectoryKey]).isDirectory) == true,
                  fm.isReadableFile(atPath: weightsURL.path) else {
                return blocked("weights root is missing or unreadable: \(weightsPath)")
            }
            do {
                try Converter.requireAssets(at: weightsURL)
            } catch {
                return blocked(errorMessage(error))
            }

            // Deliberately no stub success: real CPU/GPU execution is not yet available.
            return blocked("oracle runtime prerequisites are unavailable")
        } catch let error as CLIError {
            SploshCLI.writeStderr("oracle: \(error.message)\n")
            return ExitStatus.usage
        } catch {
            return blocked("validation failed")
        }
    }

    private static func blocked(_ detail: String) -> Int32 {
        SploshCLI.writeStderr("oracle blocked: \(detail)\n")
        return ExitStatus.notImplemented
    }

    private static func errorMessage(_ error: Error) -> String {
        String(describing: error)
    }
}
