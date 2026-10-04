// PrefixStore.swift — prompt prefixes on disk.
//
// A slot keeps its evaluated prefix in memory for as long as it is not needed for something else,
// which covers the next turn of a conversation but not a restart, an eviction, or a long prompt
// that comes back an hour later. Evaluating a 50K-token prompt takes minutes; reading its KV and
// recurrent state back from an SSD takes about a second. So every prompt of useful length is
// also written here when its session ends, and looked up when a new prompt arrives.
//
// One file per prefix:
//
//   "SPLPFX01"                       8 bytes
//   header length                    UInt32, little endian
//   header                           JSON (PrefixHeader)
//   tokens                           Int32 x tokenCount
//   state                            gated-delta and conv state at the end of the prefix
//   kv                               KV pages covering the prefix, in Engine.exportKV order
//   draft                            the draft model's context ring for the slot (optional)
//
// Two kinds of file are stored. A prompt's prefix, as above, belongs to one conversation: the
// next, longer one supersedes it. A state checkpoint is the recurrent state alone, at some
// position inside a prompt, with no KV and no draft section. KV depends only on the tokens
// before it, so the KV of any stored prompt (or any slot in memory) that begins with the same
// tokens serves; what cannot be had from another prompt is the recurrent state at the point
// where two prompts part, and without it nothing before that point can be reused. Checkpoints
// are what let a new conversation start from a system prompt another one has already
// evaluated, even when the two differ before the end of it.
//
// Files carry the identity of the engine that wrote them (weights, KV format, layout version);
// entries with another identity are ignored, not deleted.

import Foundation

public struct PrefixStoreStats: Sendable, Codable {
    public let directory: String
    public let entries: Int
    public let bytes: Int
    public let maxBytes: Int
    public let hits: Int
    public let saves: Int
    public let tokensRestored: Int
}

/// The draft model's context for a slot at the end of a stored prefix.
public struct DraftContextSnapshot: Sendable {
    public let first: Int
    public let end: Int
    public let rings: Data
    public init(first: Int, end: Int, rings: Data) { self.first = first; self.end = end; self.rings = rings }
}

public final class PrefixStore: @unchecked Sendable {
    fileprivate struct PrefixHeader: Codable {
        let identity: String
        let tokenCount: Int
        let stateBytes: Int
        let kvBytes: Int
        let draftFirst: Int
        let draftEnd: Int
        let draftBytes: Int
        /// A state checkpoint: no KV. Absent in files written before these existed.
        var stateOnly: Bool?
    }

    public struct Entry: Sendable {
        public let tokens: [Int32]
        fileprivate let url: URL
        fileprivate let bytes: Int
        fileprivate let payloadOffset: Int
        fileprivate let header: PrefixHeader
        fileprivate var lastUsed: Date
        public var tokenCount: Int { tokens.count }
        public var isStateOnly: Bool { header.stateOnly ?? false }
    }

    private static let magic = Array("SPLPFX01".utf8)
    private static let fileExtension = "splpfx"

    public let directory: URL
    public let identity: String
    public let maxBytes: Int
    /// Prefixes shorter than this are cheaper to re-evaluate than to store.
    public let minTokens: Int

    private let lock = NSLock()
    private var entries: [String: Entry] = [:]
    private var hits = 0, saves = 0, tokensRestored = 0
    private let writer = DispatchQueue(label: "splosh.prefix-store.write", qos: .utility)

    public init(directory: URL, identity: String, maxBytes: Int, minTokens: Int) throws {
        self.directory = directory
        self.identity = identity
        self.maxBytes = maxBytes
        self.minTokens = minTokens
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let files = (try? FileManager.default.contentsOfDirectory(at: directory, includingPropertiesForKeys: [.contentModificationDateKey, .fileSizeKey])) ?? []
        for url in files where url.pathExtension == Self.fileExtension {
            guard let entry = Self.readIndexEntry(url), entry.header.identity == identity else { continue }
            entries[url.lastPathComponent] = entry
        }
    }

    public var stats: PrefixStoreStats {
        lock.lock(); defer { lock.unlock() }
        return PrefixStoreStats(directory: directory.path, entries: entries.count, bytes: entries.values.reduce(0) { $0 + $1.bytes },
                                maxBytes: maxBytes, hits: hits, saves: saves, tokensRestored: tokensRestored)
    }

    /// The longest stored prefix that is a proper prefix of `prompt` and longer than `longerThan`.
    public func longestPrefix(of prompt: [Int], longerThan: Int) -> Entry? {
        lock.lock(); defer { lock.unlock() }
        var best: Entry?
        for entry in entries.values where !entry.isStateOnly && entry.tokenCount > longerThan && entry.tokenCount < prompt.count
            && entry.tokenCount > (best?.tokenCount ?? 0) {
            var matches = true
            for index in 0..<entry.tokenCount where Int(entry.tokens[index]) != prompt[index] { matches = false; break }
            if matches { best = entry }
        }
        return best
    }

    /// The stored prompt that shares the most leading tokens with `prompt`, and how many: a
    /// source of KV for that many tokens, whatever comes after them in either.
    public func longestCommonPrefix(with prompt: [Int]) -> (entry: Entry, common: Int)? {
        lock.lock(); defer { lock.unlock() }
        var best: (entry: Entry, common: Int)?
        for entry in entries.values where !entry.isStateOnly {
            let limit = min(entry.tokenCount, prompt.count)
            var common = 0
            while common < limit, Int(entry.tokens[common]) == prompt[common] { common += 1 }
            if common > (best?.common ?? 0) { best = (entry, common) }
        }
        return best
    }

    /// The longest state checkpoint that is a prefix of `prompt`, no longer than `limit` tokens
    /// (what a KV source covers) and longer than `longerThan`.
    public func bestState(for prompt: [Int], upTo limit: Int, longerThan: Int) -> Entry? {
        lock.lock(); defer { lock.unlock() }
        var best: Entry?
        for entry in entries.values where entry.isStateOnly && entry.tokenCount > longerThan && entry.tokenCount <= limit
            && entry.tokenCount < prompt.count && entry.tokenCount > (best?.tokenCount ?? 0) {
            if Self.isPrefix(entry, of: prompt) { best = entry }
        }
        return best
    }

    /// Whether a state checkpoint at the end of `tokens` is worth writing.
    public func wantsState(_ tokens: [Int]) -> Bool {
        guard tokens.count >= minTokens else { return false }
        lock.lock(); defer { lock.unlock() }
        return entries[Self.fileName(tokens, stateOnly: true)] == nil
    }

    /// Store the recurrent state at the end of `tokens`.
    public func saveState(tokens: [Int], state: Data) {
        save(tokens: tokens, snapshot: SlotSnapshot(tokenCount: tokens.count, state: state, kv: Data()), draft: nil, stateOnly: true)
    }

    /// Whether a prefix is worth writing: long enough, not stored already, and not a small
    /// extension of something that is. A growing conversation would otherwise write its whole
    /// history again on every turn; restoring the slightly shorter prefix and re-evaluating the
    /// difference costs a few seconds at most.
    ///
    /// `exactly` asks only whether these very tokens are stored: for a server about to stop,
    /// which wants each conversation back as it stands, not a little short of it.
    public func wants(_ tokens: [Int], exactly: Bool = false) -> Bool {
        guard tokens.count >= minTokens else { return false }
        lock.lock(); defer { lock.unlock() }
        guard entries[Self.fileName(tokens)] == nil else { return false }
        if exactly { return true }
        let slack = max(2048, tokens.count / 16)
        for entry in entries.values where !entry.isStateOnly && entry.tokenCount <= tokens.count && entry.tokenCount + slack > tokens.count {
            if Self.isPrefix(entry, of: tokens) { return false }
        }
        // The same goes for a prompt that a state checkpoint already takes most of the way,
        // when some stored prompt has the KV for it: a new conversation that begins like an
        // old one would otherwise store the shared beginning again in full.
        for state in entries.values where state.isStateOnly && state.tokenCount <= tokens.count && state.tokenCount + slack > tokens.count
            && Self.isPrefix(state, of: tokens) {
            for entry in entries.values where !entry.isStateOnly && entry.tokenCount >= state.tokenCount {
                var covered = true
                for index in 0..<state.tokenCount where entry.tokens[index] != state.tokens[index] { covered = false; break }
                if covered { return false }
            }
        }
        return true
    }

    private static func isPrefix(_ entry: Entry, of tokens: [Int]) -> Bool {
        guard entry.tokenCount <= tokens.count else { return false }
        for index in 0..<entry.tokenCount where Int(entry.tokens[index]) != tokens[index] { return false }
        return true
    }

    /// Read a stored prefix back. The file is mapped, so the copies into the engine's buffers
    /// are the only pass over the data.
    public func load(_ entry: Entry) throws -> (snapshot: SlotSnapshot, draft: DraftContextSnapshot?) {
        let data = try Data(contentsOf: entry.url, options: .alwaysMapped)
        let header = entry.header
        var cursor = entry.payloadOffset + header.tokenCount * MemoryLayout<Int32>.stride
        guard data.count == cursor + header.stateBytes + header.kvBytes + header.draftBytes else {
            throw EngineError.snapshotMismatch("stored prefix \(entry.url.lastPathComponent) is truncated")
        }
        // Slices of the mapping, not copies: the engine copies them into its own buffers once.
        let state = data[cursor..<cursor + header.stateBytes]; cursor += header.stateBytes
        let kv = data[cursor..<cursor + header.kvBytes]; cursor += header.kvBytes
        let draft = header.draftBytes > 0
            ? DraftContextSnapshot(first: header.draftFirst, end: header.draftEnd, rings: data[cursor..<cursor + header.draftBytes])
            : nil
        try? FileManager.default.setAttributes([.modificationDate: Date()], ofItemAtPath: entry.url.path)
        lock.lock()
        hits += 1
        tokensRestored += header.tokenCount
        entries[entry.url.lastPathComponent]?.lastUsed = Date()
        lock.unlock()
        return (SlotSnapshot(tokenCount: header.tokenCount, state: state, kv: kv), draft)
    }

    /// Store a prefix. The write happens off the caller's thread; the entry becomes visible to
    /// `longestPrefix` once the file is complete.
    public func save(tokens: [Int], snapshot: SlotSnapshot, draft: DraftContextSnapshot?) {
        save(tokens: tokens, snapshot: snapshot, draft: draft, stateOnly: false)
    }

    private func save(tokens: [Int], snapshot: SlotSnapshot, draft: DraftContextSnapshot?, stateOnly: Bool) {
        let name = Self.fileName(tokens, stateOnly: stateOnly)
        let url = directory.appendingPathComponent(name)
        let header = PrefixHeader(identity: identity, tokenCount: tokens.count, stateBytes: snapshot.state.count,
                                  kvBytes: snapshot.kv.count, draftFirst: draft?.first ?? 0, draftEnd: draft?.end ?? 0,
                                  draftBytes: draft?.rings.count ?? 0, stateOnly: stateOnly ? true : nil)
        let tokens32 = tokens.map { Int32(truncatingIfNeeded: $0) }
        writer.async { [self] in
            do {
                let headerData = try JSONEncoder().encode(header)
                var prefix = Data(Self.magic)
                withUnsafeBytes(of: UInt32(headerData.count).littleEndian) { prefix.append(contentsOf: $0) }
                prefix.append(headerData)
                let payloadOffset = prefix.count
                tokens32.withUnsafeBufferPointer { prefix.append(Data(buffer: $0)) }
                let temporary = directory.appendingPathComponent(".\(name).tmp")
                FileManager.default.createFile(atPath: temporary.path, contents: nil)
                do {
                    let handle = try FileHandle(forWritingTo: temporary)
                    try Self.write(prefix, to: handle)
                    try Self.write(snapshot.state, to: handle)
                    try Self.write(snapshot.kv, to: handle)
                    if let draft { try Self.write(draft.rings, to: handle) }
                    try handle.close()
                    _ = try? FileManager.default.removeItem(at: url)
                    try FileManager.default.moveItem(at: temporary, to: url)
                } catch {
                    try? FileManager.default.removeItem(at: temporary)
                    throw error
                }
                let bytes = prefix.count + snapshot.state.count + snapshot.kv.count + (draft?.rings.count ?? 0)
                lock.lock()
                // Shorter prefixes of the same prompt are superseded by this one. Checkpoints
                // neither supersede nor are superseded: other prompts branch off at them.
                var superseded: [URL] = []
                for (key, entry) in entries where !stateOnly && !entry.isStateOnly && entry.tokenCount < tokens.count && Self.isPrefix(entry, of: tokens) {
                    superseded.append(entry.url)
                    entries[key] = nil
                }
                entries[name] = Entry(tokens: tokens32, url: url, bytes: bytes, payloadOffset: payloadOffset, header: header, lastUsed: Date())
                saves += 1
                lock.unlock()
                for old in superseded { try? FileManager.default.removeItem(at: old) }
                evict()
            } catch {
                writeStandardError(Data("prefix store: could not write \(name): \(error)\n".utf8))
            }
        }
    }

    /// A single write(2) is limited to 2 GiB and a long context's KV is larger than that.
    private static func write(_ data: Data, to handle: FileHandle) throws {
        let piece = 256 << 20
        var offset = data.startIndex
        while offset < data.endIndex {
            let end = min(offset + piece, data.endIndex)
            try handle.write(contentsOf: data[offset..<end])
            offset = end
        }
    }

    /// Block until queued writes have finished (shutdown, tests).
    public func flush() { writer.sync {} }

    /// Delete least-recently-used entries until the store fits its budget.
    private func evict() {
        lock.lock()
        var total = entries.values.reduce(0) { $0 + $1.bytes }
        var victims: [URL] = []
        for (name, entry) in entries.sorted(by: { $0.value.lastUsed < $1.value.lastUsed }) where total > maxBytes {
            total -= entry.bytes
            victims.append(entry.url)
            entries[name] = nil
        }
        lock.unlock()
        for url in victims { try? FileManager.default.removeItem(at: url) }
    }

    /// One stored prefix as it sits on disk, whatever engine identity wrote it.
    public struct InventoryItem: Sendable {
        public let url: URL
        public let tokenCount: Int
        public let bytes: Int
        public let identity: String
        public let modified: Date
        public let stateOnly: Bool
    }

    /// Every readable prefix file in a directory (for `splosh cache`).
    public static func inventory(directory: URL) -> [InventoryItem] {
        let files = (try? FileManager.default.contentsOfDirectory(at: directory, includingPropertiesForKeys: [.contentModificationDateKey, .fileSizeKey])) ?? []
        return files.filter { $0.pathExtension == fileExtension }.compactMap { url in
            guard let entry = readIndexEntry(url) else { return nil }
            return InventoryItem(url: url, tokenCount: entry.tokenCount, bytes: entry.bytes, identity: entry.header.identity,
                                 modified: entry.lastUsed, stateOnly: entry.isStateOnly)
        }.sorted { $0.modified > $1.modified }
    }

    private static func fileName(_ tokens: [Int], stateOnly: Bool = false) -> String {
        // FNV-1a over the token ids; the token list in the file is what is actually compared.
        var hash: UInt64 = 0xcbf2_9ce4_8422_2325
        for token in tokens {
            var value = UInt32(truncatingIfNeeded: token)
            for _ in 0..<4 { hash = (hash ^ UInt64(value & 0xff)) &* 0x0000_0100_0000_01b3; value >>= 8 }
        }
        return String(format: "%016llx-%d%@.%@", hash, tokens.count, stateOnly ? "-state" : "", fileExtension)
    }

    private static func readIndexEntry(_ url: URL) -> Entry? {
        guard let handle = try? FileHandle(forReadingFrom: url) else { return nil }
        defer { try? handle.close() }
        guard let lead = try? handle.read(upToCount: 12), lead.count == 12, Array(lead.prefix(8)) == magic else { return nil }
        let headerLength = Int(lead.subdata(in: 8..<12).withUnsafeBytes { $0.loadUnaligned(as: UInt32.self) }.littleEndian)
        guard headerLength > 0, headerLength < 1 << 20,
              let headerData = try? handle.read(upToCount: headerLength), headerData.count == headerLength,
              let header = try? JSONDecoder().decode(PrefixHeader.self, from: headerData),
              let tokenData = try? handle.read(upToCount: header.tokenCount * MemoryLayout<Int32>.stride),
              tokenData.count == header.tokenCount * MemoryLayout<Int32>.stride else { return nil }
        let tokens = tokenData.withUnsafeBytes { Array($0.bindMemory(to: Int32.self)) }
        let values = try? url.resourceValues(forKeys: [.contentModificationDateKey, .fileSizeKey])
        let payloadOffset = 12 + headerLength
        let expected = payloadOffset + tokenData.count + header.stateBytes + header.kvBytes + header.draftBytes
        guard let size = values?.fileSize, size == expected else { return nil }
        return Entry(tokens: tokens, url: url, bytes: size, payloadOffset: payloadOffset, header: header,
                     lastUsed: values?.contentModificationDate ?? .distantPast)
    }
}
