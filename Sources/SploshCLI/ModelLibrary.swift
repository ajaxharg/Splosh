// ModelLibrary.swift — the models Splosh knows how to fetch.
//
// A model here is a set of files of a Hugging Face repository at one revision, each with its
// size and SHA-256, and the artifact they are converted to. The revisions are the ones Splosh
// was measured on, so what is downloaded is checked against that and not against whatever the
// repository holds today. `splosh download` fetches and converts one (DownloadCommand); the
// server lists them on its first-launch page and its models page (ServeDownloads).
//
// The same directories take files fetched by hand: a file already where the library would put
// it is checked and not downloaded again.

import Foundation
import SploshServer

struct ModelLibrary: Codable, Equatable, Sendable {
    struct File: Codable, Equatable, Sendable {
        var name: String
        var bytes: Int
        var sha256: String
    }

    /// Files of one repository at one revision, and the directory they are kept in.
    struct Source: Codable, Equatable, Sendable {
        var repo: String
        var revision: String
        var directory: String
        var files: [File]

        var bytes: Int { files.reduce(0) { $0 + $1.bytes } }
    }

    struct Model: Codable, Equatable, Sendable {
        /// What the source is: an MLX safetensors pack (converted, then tiled) or a GGUF file.
        enum Kind: String, Codable, Sendable { case mlx, gguf }

        var id: String
        var title: String
        var summary: String
        var kind: Kind
        var source: Source
        /// Where its artifact goes unless splosh.toml registers the model somewhere else.
        var artifact: String
        var artifactBytes: Int
        var bitsPerWeight: Double
        var memoryGiB: Double
    }

    /// The model fetched when none is named.
    var defaultModel: String
    /// The tokenizer and chat template every model is served with.
    var tokenizer: Source
    /// The DFlash 2 draft model for speculative decoding, shared by every model.
    var draft: Source
    var models: [Model]

    func model(_ id: String) -> Model? { models.first { $0.id == id } }

    /// The library in use: the one built in, or the JSON file SPLOSH_LIBRARY names (for tests,
    /// which cannot serve 16 GB files with the built-in hashes).
    static func current(environment: [String: String] = ProcessInfo.processInfo.environment) throws -> ModelLibrary {
        guard let path = environment["SPLOSH_LIBRARY"] else { return builtIn }
        do {
            return try JSONDecoder().decode(ModelLibrary.self, from: Data(contentsOf: URL(fileURLWithPath: path)))
        } catch {
            throw CLIError("SPLOSH_LIBRARY: \(path) is not a model library: \(error)")
        }
    }

    /// Where files are fetched from: Hugging Face, or the mirror HF_ENDPOINT names.
    static func endpoint(environment: [String: String] = ProcessInfo.processInfo.environment) -> String {
        var endpoint = environment["HF_ENDPOINT"] ?? "https://huggingface.co"
        while endpoint.hasSuffix("/") { endpoint.removeLast() }
        return endpoint
    }

    // MARK: The models

    private static let mlxRepo = "mlx-community/Qwen3.8-27B-4bit"
    private static let mlxRevision = "3e6447f082e89cc7f0bc6e5441afd38dfce760ff"
    private static let tokenizerJSON = File(name: "tokenizer.json", bytes: 19_989_325, sha256: "06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523")

    private static func gguf(_ id: String, _ quant: String, bytes: Int, sha256: String, artifactBytes: Int, bits: Double, memoryGiB: Double,
                             summary: String) -> Model {
        Model(id: id, title: "Unsloth UD-\(quant)", summary: summary, kind: .gguf,
              source: Source(repo: "unsloth/Qwen3.8-27B-GGUF", revision: "4ca720788d1e01f1bff70c033e0d0028fd02e502", directory: "inputs/gguf",
                             files: [File(name: "Qwen3.8-27B-UD-\(quant).gguf", bytes: bytes, sha256: sha256)]),
              artifact: "models/gguf/ud-\(quant.lowercased()).splw", artifactBytes: artifactBytes, bitsPerWeight: bits, memoryGiB: memoryGiB)
    }

    static let builtIn = ModelLibrary(
        defaultModel: "mq4",
        tokenizer: Source(repo: mlxRepo, revision: mlxRevision, directory: "inputs/tokenizer", files: [
            tokenizerJSON,
            File(name: "tokenizer_config.json", bytes: 1165, sha256: "792fa3f0cb88b111e54ef3134c873531008c4df471d108da17903426e308aa7b"),
            File(name: "chat_template.jinja", bytes: 8952, sha256: "c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041"),
        ]),
        draft: Source(repo: "incoai/Qwen3.8-27B-DFlash2", revision: "015e795645c74b1a0eeef3b570031fb62e769bc5", directory: "inputs/draft", files: [
            File(name: "model.safetensors", bytes: 3_848_817_896, sha256: "67fc76d68dc5a9415511a4f394ef744d67510cd20e93b37cc2cc7d28e4bab65c"),
        ]),
        models: [
            Model(id: "mq4", title: "MLX 4-bit pack",
                  summary: "The smallest and the fastest to decode: the pack Splosh's published figures were measured on. The one to start with.",
                  kind: .mlx,
                  source: Source(repo: mlxRepo, revision: mlxRevision, directory: "inputs/mlx-q4", files: [
                      File(name: "config.json", bytes: 4932, sha256: "14b65a0ee06517060a6bbd979bb1a8ff54e7b304b1a1f01d54344b88b8285e85"),
                      File(name: "model.safetensors.index.json", bytes: 218_281, sha256: "13b840162b4cb35c66fef7df072f7dbb4717908204364f5e5d9f9655a2758fa8"),
                      tokenizerJSON,
                      File(name: "model-00001-of-00003.safetensors", bytes: 5_343_268_662, sha256: "6cc1508e96fb5d0865dfd5753a79f4ec60651bf3e2a82844a7e8ae9c60528c0d"),
                      File(name: "model-00002-of-00003.safetensors", bytes: 5_354_185_130, sha256: "83f2a20ca8058f486a3634a27faf99587f4cd3c156a83dee34fb99e6ac178670"),
                      File(name: "model-00003-of-00003.safetensors", bytes: 5_357_087_557, sha256: "31b8c91ef899f79efaaa69e3d2c096f6e2ebeb2ff20e29222abbd9ebc79e560a"),
                  ]),
                  artifact: ServeConfig.tiledWeightsPath, artifactBytes: 16_054_846_720, bitsPerWeight: 4.5, memoryGiB: 14.1),
            gguf("uq4", "Q4_K_M", bytes: 16_464_440_224, sha256: "322e194ff79741c7baa497c240f677f54b201b0efab44ca8e50f122b39123482",
                 artifactBytes: 16_186_089_600, bits: 4.79, memoryGiB: 15.0,
                 summary: "Unsloth's calibrated 4-bit file: closer to the full model than the MLX pack for 1 GiB more memory. Decode a little slower; not yet timed."),
            gguf("uq5", "Q5_K_M", bytes: 19_771_509_664, sha256: "2de73110cb254cbf09b54b717578dadff12ef1194e7271527e68202f39ba4bfd",
                 artifactBytes: 19_524_540_800, bits: 5.77, memoryGiB: 18.1,
                 summary: "Closer still to the full model, for 4 GiB more memory than the MLX pack. Decode measured at 0.7 to 0.85 of its rate; prefill the same."),
            gguf("uq6", "Q6_K_M", bytes: 23_088_409_504, sha256: "493301830a596b8ad56dc1329f80bbcb578c8e910da395feafdc9cd8263430bb",
                 artifactBytes: 22_891_686_784, bits: 6.76, memoryGiB: 21.2,
                 summary: "The closest to the full model here, and the largest. Decode slower again than uq5; not yet timed."),
        ])
}

extension ModelLibrary.Model {
    /// The row-major artifact an MLX pack is converted to on its way to the tiled one at `tiled`.
    func rowMajorPath(tiled: String) -> String {
        let url = URL(fileURLWithPath: tiled)
        return url.lastPathComponent == "weights.tiled.splw" ? url.deletingLastPathComponent().appendingPathComponent("weights.splw").relativePath
                                                             : tiled + ".rowmajor"
    }

    /// Disk the conversion writes: the artifact, and for an MLX pack the row-major one before it.
    var conversionBytes: Int { kind == .mlx ? 2 * artifactBytes : artifactBytes }
}

extension ServeConfig {
    /// Where a library model's artifact is: where splosh.toml registers the model, or the
    /// library's own place for it.
    func artifactPath(of model: ModelLibrary.Model) -> String {
        models.first { $0.id == model.id }?.path ?? model.artifact
    }

    /// Whether a library model's artifact is there.
    func isInstalled(_ model: ModelLibrary.Model) -> Bool {
        ModelCatalog.fileSize(artifactPath(of: model)) != nil
    }
}

/// The Hugging Face cache on this machine, where `hf download` without `--local-dir` puts files.
enum HubCache {
    static func directory(environment: [String: String] = ProcessInfo.processInfo.environment) -> URL {
        if let cache = environment["HF_HUB_CACHE"] { return URL(fileURLWithPath: (cache as NSString).expandingTildeInPath, isDirectory: true) }
        if let home = environment["HF_HOME"] { return URL(fileURLWithPath: (home as NSString).expandingTildeInPath, isDirectory: true).appendingPathComponent("hub") }
        return FileManager.default.homeDirectoryForCurrentUser.appendingPathComponent(".cache/huggingface/hub")
    }

    /// A file of `repo` in the cache, of any revision, with the size wanted; nil when none is.
    static func find(_ name: String, bytes: Int, repo: String, environment: [String: String] = ProcessInfo.processInfo.environment) -> URL? {
        let snapshots = directory(environment: environment).appendingPathComponent("models--" + repo.replacingOccurrences(of: "/", with: "--"))
            .appendingPathComponent("snapshots")
        let revisions = (try? FileManager.default.contentsOfDirectory(atPath: snapshots.path)) ?? []
        return revisions.sorted().map { snapshots.appendingPathComponent($0).appendingPathComponent(name) }.first { OnDisk.size($0.path) == bytes }
    }
}

enum OnDisk {
    /// The size of the file a path leads to, links followed; nil when there is none. The
    /// Hugging Face cache is links (a snapshot's files point into its blobs), and so is a source
    /// file used from there.
    static func size(_ path: String) -> Int? {
        ModelCatalog.fileSize(URL(fileURLWithPath: (path as NSString).expandingTildeInPath).resolvingSymlinksInPath().path)
    }
}

/// Sizes and times as the terminal and the pages give them.
enum Human {
    /// Decimal units, as download sizes are quoted: "16.05 GB", "20.0 MB".
    static func bytes(_ count: Int) -> String {
        let value = Double(count)
        if value >= 1e9 { return String(format: value >= 1e11 ? "%.0f GB" : "%.2f GB", value / 1e9) }
        if value >= 1e6 { return String(format: "%.1f MB", value / 1e6) }
        if value >= 1e3 { return String(format: "%.0f kB", value / 1e3) }
        return "\(count) bytes"
    }

    static func duration(_ seconds: Double) -> String {
        let whole = Int(seconds.rounded())
        if whole < 90 { return "\(whole) s" }
        if whole < 5400 { return "\(whole / 60) min \(whole % 60) s" }
        return "\(whole / 3600) h \(whole % 3600 / 60) min"
    }
}
