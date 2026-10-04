import Foundation

/// Qwen byte-level BPE tokenizer. `tokenizer.json` is the source of truth; the config is
/// loaded as well so a missing/invalid materialized asset cannot be silently ignored.
///
/// Encoding follows the asset's pipeline: NFC normalisation, added-token isolation, the
/// `Split` pre-tokenizer regex, byte-level mapping, then BPE merges within each piece.
public struct Tokenizer: @unchecked Sendable {
    private struct Model: Decodable {
        let vocab: [String: Int]
        let merges: [[String]]
    }
    private struct AddedToken: Decodable {
        let id: Int
        let content: String
        let special: Bool?
    }
    private struct Asset: Decodable {
        let model: Model
        let addedTokens: [AddedToken]
        enum CodingKeys: String, CodingKey { case model; case addedTokens = "added_tokens" }
    }
    private struct Config: Decodable {}

    /// The `Split` pattern from `tokenizer.json`'s `pre_tokenizer`.
    static let splitPattern =
        "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"

    private let vocab: [String: Int]
    private let ranks: [String: Int]
    private let addedIDs: [String: Int]
    private let addedRegex: NSRegularExpression?
    private let splitRegex: NSRegularExpression
    private let byteSymbols: [String]
    /// Raw bytes each id decodes to.
    private let idBytes: [Int: [UInt8]]
    private let specialIDs: Set<Int>

    /// One past the largest id the tokenizer defines. Model rows at or above this are padding.
    public let idLimit: Int

    public init(tokenizerURL: URL = Tokenizer.defaultTokenizerURL,
                configURL: URL = Tokenizer.defaultConfigURL) throws {
        let data = try Data(contentsOf: tokenizerURL)
        let asset = try JSONDecoder().decode(Asset.self, from: data)
        _ = try JSONDecoder().decode(Config.self, from: Data(contentsOf: configURL))
        vocab = asset.model.vocab
        var rankMap: [String: Int] = [:]
        rankMap.reserveCapacity(asset.model.merges.count)
        for (index, merge) in asset.model.merges.enumerated() { rankMap[merge.joined(separator: " ")] = index }
        ranks = rankMap

        let byteToUnicode = Self.makeByteToUnicode()
        byteSymbols = (0...255).map { String(byteToUnicode[UInt8($0)]!) }
        // Keyed by scalar: grapheme clustering must never merge two byte symbols.
        var unicodeToByte: [UnicodeScalar: UInt8] = [:]
        for (byte, character) in byteToUnicode { unicodeToByte[character.unicodeScalars.first!] = byte }

        var bytes: [Int: [UInt8]] = [:]
        bytes.reserveCapacity(vocab.count + asset.addedTokens.count)
        for (text, id) in vocab { bytes[id] = text.unicodeScalars.compactMap { unicodeToByte[$0] } }
        var added: [String: Int] = [:]
        var special = Set<Int>()
        for token in asset.addedTokens {
            added[token.content] = token.id
            bytes[token.id] = Array(token.content.utf8)
            if token.special ?? false { special.insert(token.id) }
        }
        addedIDs = added
        idBytes = bytes
        specialIDs = special
        idLimit = (bytes.keys.max() ?? -1) + 1

        // Longest content first so overlapping added tokens resolve the way the reference does.
        let alternation = added.keys.sorted { $0.utf8.count > $1.utf8.count }
            .map { NSRegularExpression.escapedPattern(for: $0) }.joined(separator: "|")
        addedRegex = alternation.isEmpty ? nil : try NSRegularExpression(pattern: alternation)
        splitRegex = try NSRegularExpression(pattern: Self.splitPattern)
    }

    public static var defaultTokenizerURL: URL {
        URL(fileURLWithPath: "inputs/tokenizer/tokenizer.json", isDirectory: false)
    }
    public static var defaultConfigURL: URL {
        URL(fileURLWithPath: "inputs/tokenizer/tokenizer_config.json", isDirectory: false)
    }

    public func encode(_ input: String) -> [Int] {
        let normalized = input.precomposedStringWithCanonicalMapping as NSString
        var output: [Int] = []
        var cursor = 0
        let whole = NSRange(location: 0, length: normalized.length)
        addedRegex?.enumerateMatches(in: normalized as String, range: whole) { match, _, _ in
            guard let range = match?.range else { return }
            if range.location > cursor {
                encodeText(normalized.substring(with: NSRange(location: cursor, length: range.location - cursor)), into: &output)
            }
            if let id = addedIDs[normalized.substring(with: range)] { output.append(id) }
            cursor = range.location + range.length
        }
        if cursor < normalized.length {
            encodeText(normalized.substring(from: cursor), into: &output)
        }
        return output
    }

    /// Raw bytes for one id, or nil when the id is not defined by the tokenizer.
    public func bytes(for id: Int) -> [UInt8]? { idBytes[id] }

    public func isSpecial(_ id: Int) -> Bool { specialIDs.contains(id) }

    public func decode(_ ids: [Int]) -> String {
        String(decoding: ids.flatMap { idBytes[$0] ?? [] }, as: UTF8.self)
    }

    /// Byte-level token text for every id, in the form `Detokenizer` consumes.
    public func tokenTextTable() -> [Int: String] {
        var table: [Int: String] = [:]
        table.reserveCapacity(idBytes.count)
        for (id, bytes) in idBytes { table[id] = bytes.map { byteSymbols[Int($0)] }.joined() }
        return table
    }

    private func encodeText(_ text: String, into output: inout [Int]) {
        let ns = text as NSString
        var cursor = 0
        splitRegex.enumerateMatches(in: text, range: NSRange(location: 0, length: ns.length)) { match, _, _ in
            guard let range = match?.range, range.length > 0 else { return }
            if range.location > cursor {
                output.append(contentsOf: encodePiece(ns.substring(with: NSRange(location: cursor, length: range.location - cursor))))
            }
            output.append(contentsOf: encodePiece(ns.substring(with: range)))
            cursor = range.location + range.length
        }
        if cursor < ns.length { output.append(contentsOf: encodePiece(ns.substring(from: cursor))) }
    }

    private func encodePiece(_ text: String) -> [Int] {
        var pieces = text.utf8.map { byteSymbols[Int($0)] }
        guard !pieces.isEmpty else { return [] }
        while pieces.count > 1 {
            var best: (rank: Int, index: Int)?
            for i in 0..<(pieces.count - 1) {
                if let rank = ranks[pieces[i] + " " + pieces[i + 1]], best == nil || rank < best!.rank { best = (rank, i) }
            }
            guard let selected = best else { break }
            pieces[selected.index] += pieces.remove(at: selected.index + 1)
        }
        return pieces.compactMap { vocab[$0] }
    }

    private static func makeByteToUnicode() -> [UInt8: Character] {
        var bytes = Array(UInt8(33)...UInt8(126)) + Array(UInt8(161)...UInt8(172)) + Array(UInt8(174)...UInt8(255))
        var unicode = bytes.map { Character(UnicodeScalar($0)) }
        var next = 0
        for value in UInt8(0)...UInt8(255) where !bytes.contains(value) {
            bytes.append(value)
            unicode.append(Character(UnicodeScalar(256 + next)!))
            next += 1
        }
        var result: [UInt8: Character] = [:]
        for (byte, character) in zip(bytes, unicode) { result[byte] = character }
        return result
    }
}
