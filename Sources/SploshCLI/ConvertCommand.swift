// M3.1 converter command. It consumes the recorded M3.6 alignment rule, then fails closed on absent packs.
import Foundation
import SploshModel

public struct ConvertArguments: Equatable, Sendable {
    public var input: String?
    public var output: String?
    public var verify: Bool
    public var retile: Bool = false
    public var tokenizer: String?
    public var help: Bool

    public init(input: String? = nil, output: String? = nil, verify: Bool = false, help: Bool = false) {
        self.input = input; self.output = output; self.verify = verify; self.help = help
    }
}

public enum ConvertCommand {
    public static let usage = """
        usage: splosh convert --input <dir> --out <path> [--verify]
               splosh convert --input <file.gguf> --out <path> [--tokenizer <tokenizer.json>]
               splosh convert --retile --input <file.splw> --out <path>

          --input <dir>   source MLX 4-bit or 8-bit safetensors directory
          --input <file.gguf>
                          source llama.cpp GGUF file: its weights are kept in their own formats,
                          at their own size, in the tiled planes the GGUF kernels read
          --out <path>    destination; default $PWD/splosh-weights/ (a GGUF file has no default)
          --verify        verify deterministic coverage sample and source bytes (directories)
          --tokenizer <tokenizer.json>
                          the tokenizer a GGUF artifact is served with, for the hash in its
                          header; default inputs/tokenizer/tokenizer.json
          --retile        rewrite an existing artifact (--input <file.splw>) in the tiled weight
                          layout the engine runs from, so it is mapped as a single copy
          --help
        """

    /// The tokenizer whose hash a GGUF artifact records when none is named.
    public static let defaultTokenizerPath = "inputs/tokenizer/tokenizer.json"

    public static func parse(_ tokens: [String]) throws -> ConvertArguments {
        var result = ConvertArguments()
        var stream = TokenStream(tokens)
        while let token = stream.current {
            switch token {
            case "--help", "-h": result.help = true
            case "--verify":
                guard !result.verify else { throw CLIError("duplicate option '--verify' for command 'convert'") }
                result.verify = true
            case "--retile":
                guard !result.retile else { throw CLIError("duplicate option '--retile' for command 'convert'") }
                result.retile = true
            case "--input":
                guard result.input == nil else { throw CLIError("duplicate option '--input' for command 'convert'") }
                result.input = try stream.requireValue(for: token)
            case "--out":
                guard result.output == nil else { throw CLIError("duplicate option '--out' for command 'convert'") }
                result.output = try stream.requireValue(for: token)
            case "--tokenizer":
                guard result.tokenizer == nil else { throw CLIError("duplicate option '--tokenizer' for command 'convert'") }
                result.tokenizer = try stream.requireValue(for: token)
            default: throw CLIError(CLIArguments.unexpected(token, command: "convert"))
            }
            stream.advance()
        }
        return result
    }

    public static func run(_ arguments: [String]) -> Int32 {
        do {
            let parsed = try parse(arguments)
            if parsed.help { print(usage); return ExitStatus.ok }
            let fm = FileManager.default
            if parsed.retile {
                guard let inputPath = parsed.input, let outputPath = parsed.output else {
                    return fail("--retile needs --input <artifact.splw> and --out <path>", status: ExitStatus.usage)
                }
                let source = URL(fileURLWithPath: inputPath), destination = URL(fileURLWithPath: outputPath)
                guard source.standardizedFileURL != destination.standardizedFileURL else { return fail("output must differ from input") }
                let report = try Converter.retile(inputURL: source, outputURL: destination)
                print("retiled format=\(report.format) tensors=\(report.tensorCount) tiled_weights=\(report.tiledWeights) bytes=\(report.byteCount)")
                return ExitStatus.ok
            }
            if let inputPath = parsed.input, inputPath.lowercased().hasSuffix(".gguf"),
               (try? URL(fileURLWithPath: inputPath).resourceValues(forKeys: [.isDirectoryKey]).isDirectory) != true {
                return try convertGguf(parsed, inputPath: inputPath)
            }
            guard parsed.tokenizer == nil else {
                return fail("--tokenizer goes with a GGUF file; a directory brings its own tokenizer.json", status: ExitStatus.usage)
            }
            let input = URL(fileURLWithPath: parsed.input ?? fm.currentDirectoryPath)
            let output = URL(fileURLWithPath: parsed.output ?? input.appendingPathComponent("splosh-weights").path)
            guard (try? input.resourceValues(forKeys: [.isDirectoryKey]).isDirectory) == true else {
                return fail("input is missing or not a directory: \(input.path)")
            }
            let inputCanonical = input.standardizedFileURL.resolvingSymlinksInPath()
            let outputCanonical = output.standardizedFileURL.resolvingSymlinksInPath()
            guard inputCanonical != outputCanonical,
                  !outputCanonical.path.hasPrefix(inputCanonical.path + "/") else {
                return fail("output must not equal or be inside input tree")
            }
            let parent = output.deletingLastPathComponent()
            if fm.fileExists(atPath: parent.path) && (try? parent.resourceValues(forKeys: [.isDirectoryKey]).isDirectory) != true {
                return fail("output parent is not a directory: \(parent.path)")
            }
            let root = URL(fileURLWithPath: fm.currentDirectoryPath)
            _ = try Converter.alignmentEvidence(from: root)
            try Converter.requireAssets(at: input)
            let report = try Converter.convert(inputRoot: input, outputURL: output, verify: parsed.verify)
            print("converted format=\(report.header.format) tensors=\(report.tensorCount) bytes=\(report.byteCount) dtype=\(report.header.dtype)")
            return ExitStatus.ok
        } catch let error as CLIError { return fail(error.message, status: ExitStatus.usage) }
        catch { return fail(String(describing: error)) }
    }

    /// A GGUF file to an artifact of planes (SploshModel/GgufConvert.swift).
    private static func convertGguf(_ parsed: ConvertArguments, inputPath: String) throws -> Int32 {
        guard let outputPath = parsed.output else {
            return fail("a GGUF file needs --out <path>", status: ExitStatus.usage)
        }
        guard !parsed.verify else {
            return fail("--verify compares an artifact with a safetensors directory; it does not apply to a GGUF file", status: ExitStatus.usage)
        }
        let source = URL(fileURLWithPath: inputPath), destination = URL(fileURLWithPath: outputPath)
        guard FileManager.default.fileExists(atPath: source.path) else { return fail("input is missing: \(source.path)") }
        guard source.standardizedFileURL.resolvingSymlinksInPath() != destination.standardizedFileURL.resolvingSymlinksInPath() else {
            return fail("output must differ from input")
        }
        let tokenizer = URL(fileURLWithPath: parsed.tokenizer ?? defaultTokenizerPath)
        guard FileManager.default.fileExists(atPath: tokenizer.path) else {
            return fail("tokenizer is missing: \(tokenizer.path); name the one the artifact is served with by --tokenizer")
        }
        let started = Date()
        let report = try Converter.convertGguf(inputURL: source, outputURL: destination, tokenizerURL: tokenizer)
        // The packed formats by their kernel names, then the dense tensors by their float type.
        let formats = report.tensorsByType.sorted { $0.key.rawValue < $1.key.rawValue }
            .map { "\(GgufPlanes.geometry(of: $0.key)?.kernelSuffix ?? "dense.\($0.key)")=\($0.value)" }.joined(separator: " ")
        print("converted format=\(report.header.format) tensors=\(report.tensorCount) records=\(report.recordCount) bytes=\(report.byteCount) \(formats) mtp_skipped=\(report.mtpTensors) seconds=\(String(format: "%.1f", Date().timeIntervalSince(started)))")
        return ExitStatus.ok
    }

    private static func fail(_ message: String, status: Int32 = ExitStatus.notImplemented) -> Int32 {
        SploshCLI.writeStderr("convert: \(message)\n")
        return status
    }
}
