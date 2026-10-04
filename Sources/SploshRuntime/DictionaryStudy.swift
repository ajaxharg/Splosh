// DictionaryStudy.swift — would drafts copied from text outside the session pay?
//
// A session's copied drafts come from its own text (LookupIndex). This asks, without changing
// anything that is decoded, what two other bodies of text would have added: what the other
// conversations have evaluated and written, and a fixed corpus of source files. At every
// verify step the present text is looked up in both. A proposal found on a longer run than
// the session's own text shares is kept, and when the request ends it is scored against what
// was generated, beside what the step taken from the same anchor yielded.
//
// A source is worth building only where its proposals beat the steps that were taken, on
// enough of the steps to matter; the report gives both.

import Foundation

public struct DictionaryStudy: Sendable {
    public enum Source: Int, CaseIterable, Sendable {
        case conversations, corpus
        var label: String { self == .conversations ? "other conversations" : "corpus" }
    }

    /// A proposal waiting for the text it is to be scored against.
    public struct Note: Sendable {
        public let source: Source
        /// Where the anchor sits in the session's text.
        public let position: Int
        public let match: Int
        public let tokens: [Int]
        /// Tokens the step taken from this anchor yielded.
        public let stepped: Int
    }

    public struct Tally: Sendable, Equatable {
        public var anchors = 0
        /// Tokens the proposals would have yielded, and tokens the steps taken did.
        public var proposed = 0, stepped = 0
    }

    /// Shared runs are counted in the bands the copied drafts were measured in.
    public static let bands: [(label: String, range: ClosedRange<Int>)] = [("3", 3...3), ("4-5", 4...5), ("6-9", 6...9), ("10-19", 10...19), ("20+", 20...Int.max)]
    /// Tokens of the present text a lookup is given, the anchor last.
    public static let tailLength = 65
    /// Most tokens kept of what conversations have evaluated, and of a corpus.
    public static let capacity = 4_000_000

    /// One index per source.
    private var indexes: [LookupIndex]
    /// Verify steps looked up.
    public private(set) var steps = 0
    /// Per source: proposals found on a run no longer than the session's own text shares.
    public private(set) var redundant: [Int]
    /// Per source and band, over the proposals that could be scored.
    public private(set) var tallies: [[Tally]]

    /// `corpus`: the fixed texts, one array of tokens per file.
    public init(corpus: [[Int]] = []) {
        var fixed = LookupIndex()
        for (number, file) in corpus.enumerated() { fixed.append(contentsOf: file, owner: number) }
        indexes = [LookupIndex(), fixed]
        redundant = [Int](repeating: 0, count: Source.allCases.count)
        tallies = [[Tally]](repeating: [Tally](repeating: Tally(), count: Self.bands.count), count: Source.allCases.count)
    }

    /// Text a conversation has evaluated or generated, for the others to be looked up in.
    public mutating func add(_ text: some Sequence<Int>, conversation: Int) {
        guard indexes[Source.conversations.rawValue].count < Self.capacity else { return }
        indexes[Source.conversations.rawValue].append(contentsOf: text, owner: conversation)
    }

    /// Look up the text ending in `tail` (the anchor last, at `position` in the session's
    /// text). `own` is the run the session's own text shares there and `stepped` what the
    /// step taken from the anchor yielded. Returns the proposals worth scoring.
    public mutating func look(tail: [Int], conversation: Int, position: Int, own: Int, stepped: Int) -> [Note] {
        steps += 1
        var notes: [Note] = []
        for source in Source.allCases {
            guard let found = indexes[source.rawValue].continuation(of: tail, count: DraftModel.maxBlock - 1,
                                                                    skipping: source == .conversations ? conversation : nil) else { continue }
            if found.match > own {
                notes.append(Note(source: source, position: position, match: found.match, tokens: found.tokens, stepped: stepped))
            } else {
                redundant[source.rawValue] += 1
            }
        }
        return notes
    }

    /// Score a finished session's notes against its text: the agreeing prefix plus the token
    /// the target would then have given, as a verify step yields. Anchors too close to the end
    /// are left out. Returns whether any were counted.
    public mutating func score(_ notes: [Note], text: [Int]) -> Bool {
        var counted = false
        for note in notes where note.position + DraftModel.maxBlock <= text.count {
            guard let band = Self.bands.firstIndex(where: { $0.range.contains(note.match) }) else { continue }
            var keep = 0
            while keep < note.tokens.count, note.tokens[keep] == text[note.position + 1 + keep] { keep += 1 }
            tallies[note.source.rawValue][band].anchors += 1
            tallies[note.source.rawValue][band].proposed += keep + 1
            tallies[note.source.rawValue][band].stepped += note.stepped
            counted = true
        }
        return counted
    }

    /// A line per source that has found anything, everything so far: by band, the anchors,
    /// the tokens a step its proposals would have yielded and the tokens a step the steps
    /// taken from the same anchors did.
    public var report: [String] {
        Source.allCases.compactMap { source in
            let row = tallies[source.rawValue]
            let anchors = row.reduce(0) { $0 + $1.anchors }
            guard anchors > 0 else { return nil }
            let bands = zip(Self.bands, row).filter { $0.1.anchors > 0 }.map { band, tally in
                String(format: "run %@: %d anchors, %.2f a step against %.2f", band.label, tally.anchors,
                       Double(tally.proposed) / Double(tally.anchors), Double(tally.stepped) / Double(tally.anchors))
            }
            return String(format: "dictionary study, %@: a longer run than the session's own text at %d of %d verify steps (%d more no longer) — ",
                          source.label, anchors, steps, redundant[source.rawValue]) + bands.joined(separator: "; ")
        }
    }
}
