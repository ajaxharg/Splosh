// BatchScheduler.swift — continuous batching over the engine.
//
// One thread owns the engine. Every step it packs rows from all live sessions into a single
// forward pass: one row per decoding session, and the remaining row budget to sessions that are
// still prefilling, shortest first. Sessions join and leave between steps.
//
// Session state outlives a request. A slot that finishes is kept as a cached prefix, so the
// next request that extends it — the normal shape of a chat or agent loop — evaluates only the
// new tokens. Gated-delta layers hold a recurrent state that cannot be rewound, so a slot is
// reusable at exactly two points: its final token, and the end-of-prompt checkpoint.

import Foundation

public struct GenerationRequest: Sendable {
    public var promptTokens: [Int]
    public var maxTokens: Int
    public var sampling: SamplingParameters
    public var stopTokenIDs: Set<Int>
    /// Token ids at or above this are never sampled.
    public var vocabLimit: Int
    /// Free-form tag shown in stats (for example the requested model name).
    public var label: String
    /// Positions in the prompt where other prompts are likely to branch off: the scheduler
    /// keeps the recurrent state as it stood after that many tokens, so a prompt that shares
    /// only that much with this one does not start from nothing. The server passes the end of
    /// the system block (a harness's instructions and tool definitions).
    public var checkpointHints: [Int] = []
    /// How many leading prompt tokens the conversation's next request is sure to repeat: the
    /// prompt up to where it opens the reply (see ConversationLedger). Nil means all of it.
    public var stablePromptTokens: Int?
    /// Filled in by whoever parses the output, for the stats (see SessionActivity).
    public var activity: SessionActivity?
    /// Seconds the server spent making this request out of what it was sent (rendering the
    /// template, tokenising), for RequestOverhead.
    public var prepareSeconds = 0.0
    /// Set when the prompt opens a thinking block, which the reply begins inside: how the
    /// block is closed, so the scheduler can close it (see ThinkingWatch).
    public var thinkingClose: ThinkingClose?
    /// The tokens a tool call opens and closes with, so the scheduler can see one in progress
    /// (see ReplyLimit).
    public var toolCall: ToolCallMarks?

    public init(promptTokens: [Int], maxTokens: Int, sampling: SamplingParameters = SamplingParameters(),
                stopTokenIDs: Set<Int>, vocabLimit: Int, label: String = "") {
        self.promptTokens = promptTokens; self.maxTokens = maxTokens; self.sampling = sampling
        self.stopTokenIDs = stopTokenIDs; self.vocabLimit = vocabLimit; self.label = label
    }
}

/// How a thinking block is closed: the token that closes it, and the tokens to write to close
/// it the way the chat template does (the token with the newlines around it).
public struct ThinkingClose: Sendable, Equatable {
    public var token: Int
    public var tokens: [Int]
    public init(token: Int, tokens: [Int]) { self.token = token; self.tokens = tokens }
}

/// Keeps part of a reply's token limit for its answer.
///
/// Thinking and the answer come out of one limit, which the model does not know of: a reply
/// that thinks to the end of it has written nothing a client can use (eight agents at once
/// spent 32,768 tokens each that way). So the thinking is ended when only the reserve is left:
/// from there the closing tokens are the reply's next tokens whatever the model chose, and the
/// model goes on after them, with the reserve, to the answer.
public struct ThinkingWatch: Sendable {
    private let close: ThinkingClose
    /// Tokens of thinking after which it is ended.
    public let limit: Int
    private var thinking = true
    private var owed: [Int] = []              // closing tokens still to write, last first

    /// Nil where there is nothing to do: the reply does not begin in thinking, or nothing is
    /// held back. The reserve is at most a quarter of the limit, so a small limit is not all
    /// reserve.
    public init?(_ close: ThinkingClose?, maxTokens: Int, reserve: Int) {
        guard let close, !close.tokens.isEmpty else { return nil }
        let held = min(reserve, maxTokens / 4)
        guard held > 0 else { return nil }
        self.close = close
        limit = maxTokens - held
    }

    /// The reply's next token, given the one the model chose and how many it has so far.
    public mutating func next(_ chosen: Int, generated: Int) -> Int {
        if let token = owed.popLast() { return token }
        guard thinking else { return chosen }
        if chosen == close.token { thinking = false; return chosen }
        guard generated >= limit else { return chosen }
        thinking = false
        owed = close.tokens.reversed()
        return owed.removeLast()
    }
}

/// The tokens that open and close a tool call in the model's output.
public struct ToolCallMarks: Sendable, Equatable {
    public var open: Int
    public var close: Int
    public init(open: Int, close: Int) { self.open = open; self.close = close }
}

/// A reply's token limit, which a tool call in progress may run past.
///
/// A reply cut inside a tool call holds a call that cannot be made: the harness drops it, and
/// its turn ends there until someone types "continue". So a call that is open when the limit
/// is reached runs on to its close, for at most `overrun` tokens more, and the reply ends
/// there: on a whole call, which the harness makes and goes on from.
public struct ReplyLimit: Sendable {
    private let maxTokens: Int
    private let marks: ToolCallMarks?
    private let overrun: Int
    /// A token after which the reply is in no tool call: the close of its thinking, where an
    /// opening token was only talk of one.
    private let resets: Int?
    private var inCall = false

    public init(maxTokens: Int, marks: ToolCallMarks?, overrun: Int, resets: Int? = nil) {
        self.maxTokens = maxTokens; self.marks = marks; self.overrun = max(0, overrun); self.resets = resets
    }

    /// Whether the reply ends for length with `token`, its `generated`th.
    public mutating func reached(by token: Int, generated: Int) -> Bool {
        if let marks {
            if token == marks.open { inCall = true }
            else if token == marks.close || token == resets { inCall = false }
        }
        guard generated >= maxTokens else { return false }
        return !(inCall && generated - maxTokens < overrun)
    }
}

public enum FinishReason: String, Sendable, Codable { case stop, length, cancelled, error }

/// What a session's output consists of, as whoever parses it reports: the scheduler deals in
/// tokens and does not know reasoning from an answer from a tool call. The server's output
/// parser records each token here, and the stats read it.
public final class SessionActivity: @unchecked Sendable {
    public enum Phase: String, Sendable { case thinking, answer, toolCall = "tool call" }
    /// A stretch of the reply that is all of one kind, as the parser gave it. A tool call is
    /// one part: the function's name, and for text its arguments.
    public struct Part: Sendable, Equatable {
        public let kind: Phase
        public var name: String?
        public var text: String
        public init(kind: Phase, name: String? = nil, text: String) { self.kind = kind; self.name = name; self.text = text }
    }
    private let lock = NSLock()
    private var current: Phase?
    private var counts: [Phase: Int] = [:]
    private var parts: [Part] = []

    public init() {}

    /// One generated token, in the phase the output was in once it had been parsed.
    public func record(_ phase: Phase) {
        lock.lock(); defer { lock.unlock() }
        current = phase
        counts[phase, default: 0] += 1
    }

    /// Parsed text of the reply, kept so that it can be read while it is being written.
    public func write(_ text: String, as kind: Phase) {
        guard !text.isEmpty else { return }
        lock.lock(); defer { lock.unlock() }
        if let last = parts.indices.last, parts[last].kind == kind { parts[last].text += text }
        else { parts.append(Part(kind: kind, text: text)) }
    }

    /// A tool call begins: what is written as a tool call from here is its arguments.
    public func beginToolCall(_ name: String) {
        lock.lock(); defer { lock.unlock() }
        parts.append(Part(kind: .toolCall, name: name, text: ""))
    }

    public var snapshot: (current: String, thinking: Int, answer: Int, toolCall: Int) {
        lock.lock(); defer { lock.unlock() }
        return (current?.rawValue ?? "", counts[.thinking] ?? 0, counts[.answer] ?? 0, counts[.toolCall] ?? 0)
    }

    /// The reply so far, in the order it was written.
    public var reply: [Part] {
        lock.lock(); defer { lock.unlock() }
        return parts
    }
}

/// The activities of the sessions in the published stats, by session id, and of the last few
/// to have left them. A reply is read while it is written by asking for it again and again;
/// its end is written as its session leaves the stats, and is asked for after that.
public struct SessionActivities: Sendable {
    public static let endedKept = 16
    private var live: [Int: SessionActivity] = [:]
    private var ended: [(id: Int, activity: SessionActivity)] = []

    public init() {}

    /// The sessions that are in the stats now.
    public mutating func update(_ now: [Int: SessionActivity]) {
        ended += live.filter { now[$0.key] == nil }.sorted { $0.key < $1.key }.map { (id: $0.key, activity: $0.value) }
        ended.removeFirst(max(0, ended.count - Self.endedKept))
        live = now
    }

    public subscript(session: Int) -> SessionActivity? {
        live[session] ?? ended.last { $0.id == session }?.activity
    }
}

public struct GenerationUsage: Sendable, Codable, Equatable {
    public var promptTokens = 0
    public var cachedTokens = 0
    public var completionTokens = 0
    public var queueSeconds = 0.0
    public var prefillSeconds = 0.0
    public var decodeSeconds = 0.0
    public var overhead = RequestOverhead()
    public init() {}
}

/// Where a request's time went outside the engine's steps, in seconds. Admission and restores
/// are otherwise counted as queue time and the server's own work not at all; a short turn (a
/// tool call with a small result) can spend more here than in its steps.
public struct RequestOverhead: Sendable, Codable, Equatable {
    /// Rendering the chat template and tokenising, before the request reached the scheduler.
    public var prepare = 0.0
    /// Finding which conversation the prompt continues (ConversationLedger).
    public var join = 0.0
    /// Choosing a slot and restoring its prefix, from memory or disk; what the slot held before
    /// is written out first, which is counted under `storeCopy`.
    public var admit = 0.0
    /// Copies of the recurrent state at checkpoints, the turn mark and the prompt's end.
    public var stateExports = 0.0
    public var stateExportCount = 0
    /// Rebuilding the index copied drafts are looked up in, when the prompt is done.
    public var lookupReset = 0.0
    /// Copying conversations out for the disk store: a displaced one at admission, this one when
    /// it finished. Done on the scheduler thread; every other session waits for it.
    public var storeCopy = 0.0
    /// The request's first step, and how long the GPU had been idle before it (0: it was busy).
    /// Not part of `total`: it is a step, and is compared with the steps of its tier.
    public var firstStep = 0.0
    public var idleBeforeFirstStep = 0.0
    public init() {}

    public var total: Double { prepare + join + admit + stateExports + lookupReset + storeCopy }

    public static func += (sum: inout RequestOverhead, other: RequestOverhead) {
        sum.prepare += other.prepare; sum.join += other.join; sum.admit += other.admit
        sum.stateExports += other.stateExports; sum.stateExportCount += other.stateExportCount
        sum.lookupReset += other.lookupReset; sum.storeCopy += other.storeCopy
        sum.firstStep += other.firstStep; sum.idleBeforeFirstStep += other.idleBeforeFirstStep
    }

    /// For the request log: the parts that took a millisecond or more.
    public var summary: String {
        let parts: [(String, Double)] = [("prepare", prepare), ("join", join), ("admit", admit),
                                         ("\(stateExportCount) state export\(stateExportCount == 1 ? "" : "s")", stateExports),
                                         ("lookup", lookupReset), ("store copy", storeCopy)]
        let shown = parts.filter { $0.1 >= 0.001 }.map { String(format: "%@ %.0f", $0.0, $0.1 * 1000) }
        var text = String(format: "outside steps %.0f ms", total * 1000) + (shown.isEmpty ? "" : " (" + shown.joined(separator: ", ") + ")")
        if firstStep > 0 {
            text += String(format: "; first step %.0f ms", firstStep * 1000)
            if idleBeforeFirstStep > 0 { text += String(format: " after %.1f s idle", idleBeforeFirstStep) }
        }
        return text
    }
}

/// What speculative steps were made of: blocks copied from the context (no draft pass) and
/// blocks from the draft model, with the tokens each kind produced (accepted drafts plus the
/// target's own token).
public struct BlockKinds: Sendable, Codable, Equatable {
    public var copySteps = 0
    public var copyTokens = 0
    public var draftSteps = 0
    public var draftTokens = 0
    public init() {}

    mutating func record(copied: Bool, produced: Int) {
        if copied { copySteps += 1; copyTokens += produced } else { draftSteps += 1; draftTokens += produced }
    }

    public static func += (sum: inout BlockKinds, other: BlockKinds) {
        sum.copySteps += other.copySteps; sum.copyTokens += other.copyTokens
        sum.draftSteps += other.draftSteps; sum.draftTokens += other.draftTokens
    }
}

/// Engine steps of one row tier at one context range, as they have run: what the planner's
/// tier costs (`BatchScheduler.tierMilliseconds`) are supposed to be.
public struct StepTierStats: Sendable, Codable, Equatable {
    /// The accelerator tier the step's rows fall in: 16, then multiples of 32.
    public let rows: Int
    /// The furthest position any row of the step was at: "<16K", "16-64K" or ">64K".
    public let context: String
    /// Whether the step carried prompt rows, or only decode and verify rows.
    public let prompt: Bool
    public var steps = 0
    /// Means: the engine step, its GPU time, and the scheduler's whole cycle around it.
    public var milliseconds = 0.0
    public var gpuMilliseconds = 0.0
    public var cycleMilliseconds = 0.0
    /// The fastest step: what the tier costs on a cool GPU.
    public var fastestMilliseconds = 0.0
}

public enum GenerationEvent: Sendable {
    case token(Int)
    /// Sent every couple of seconds while a session has produced nothing yet (queued, or
    /// evaluating its prompt): how much of the prompt has been evaluated. It gives the
    /// server something to write, which is how it learns that a client has gone.
    case progress(evaluated: Int, total: Int)
    case finished(FinishReason, GenerationUsage, message: String?)
}

public struct SessionStats: Sendable, Codable {
    public let id: Int
    public let label: String
    public let state: String
    public let slot: Int?
    public let promptTokens: Int
    public let cachedTokens: Int
    public let evaluatedTokens: Int
    public let generatedTokens: Int
    public let contextTokens: Int
    public let kvBytes: Int
    public let stateBytes: Int
    /// Averages since the session was admitted (prefill) and since it began generating.
    public let prefillTokensPerSecond: Double
    public let decodeTokensPerSecond: Double
    /// The same over the last few seconds: what the session is getting now.
    public var recentPrefillTokensPerSecond = 0.0
    public var recentDecodeTokensPerSecond = 0.0
    /// Draft tokens proposed and kept over the last five seconds.
    public var recentDraftedTokens = 0
    public var recentAcceptedTokens = 0
    /// What is being generated at the moment ("thinking", "answer", "tool call"; empty when
    /// nothing has been yet or nobody is parsing the output) and the tokens of each so far.
    /// The conversation this request belongs to and which of its requests it is.
    public var conversation = 0
    public var turn = 0
    /// Tokens drafted for this request so far and how many the target accepted.
    public var draftedTokens = 0
    public var acceptedTokens = 0
    public var blocks = BlockKinds()
    public var producing = ""
    public var thinkingTokens = 0
    public var answerTokens = 0
    public var toolCallTokens = 0
    /// Tokens produced per target verify step (1.0 without a draft model).
    public let tokensPerStep: Double
    public let ageSeconds: Double
}

public struct CachedPrefixStats: Sendable, Codable {
    public let slot: Int
    public let tokens: Int
    public let checkpointTokens: Int
    public let kvBytes: Int
    public let stateBytes: Int
    public let idleSeconds: Double
}

public struct MemoryStats: Sendable, Codable {
    public let weightBytes: Int
    public let kvPoolBytes: Int
    public let kvBytesUsed: Int
    public let kvPagesTotal: Int
    public let kvPagesUsed: Int
    public let kvBytesPerToken: Int
    public let stateBytesPerSlot: Int
    public let stateBytesTotal: Int
    public let checkpointBytes: Int
    public let scratchBytes: Int
    public let deviceAllocatedBytes: Int
    public let deviceWorkingSetBytes: Int
}

public struct ThroughputStats: Sendable, Codable {
    /// Tokens per second over the recent window, summed across sessions.
    public let prefillTokensPerSecond: Double
    public let decodeTokensPerSecond: Double
    public let stepsPerSecond: Double
    public let lastStepRows: Int
    public let lastStepMilliseconds: Double
    /// Draft tokens proposed and kept over the last five seconds.
    public var recentDraftedTokens = 0
    public var recentAcceptedTokens = 0
}

public struct TotalStats: Sendable, Codable {
    public var requests = 0
    public var completed = 0
    public var promptTokens = 0
    public var cachedTokens = 0
    public var completionTokens = 0
    public var steps = 0
    /// Everything the server has done so far, finished requests or not: prompt tokens
    /// evaluated, tokens generated, and the wall-clock time it was working. Their ratios are
    /// the server's combined rates, which with several sessions at once are higher than any
    /// one session's.
    public var evaluatedTokens = 0
    public var generatedTokens = 0
    public var busySeconds = 0.0
    /// The parts of that time in which prompts were being evaluated and in which tokens were
    /// being generated (a step that did both counts towards both), so that each combined rate
    /// is over the time its kind of work was actually going on.
    public var prefillBusySeconds = 0.0
    public var decodeBusySeconds = 0.0
    /// Speculative decoding: tokens drafted (by the draft model or copied from the context)
    /// and how many of them the target accepted.
    public var draftedTokens = 0
    public var acceptedTokens = 0
    /// Summed over finished requests, each on its own clock: the time they spent evaluating
    /// their prompts and generating, and the prompt tokens they had to evaluate. Their ratios
    /// are what a request experienced on average.
    public var finishedEvaluatedTokens = 0
    public var finishedPrefillSeconds = 0.0
    public var finishedDecodeSeconds = 0.0
    /// Summed over finished requests: their time outside the engine's steps, by part.
    public var finishedOverhead = RequestOverhead()
    /// Speculative steps by kind, over all requests.
    public var blocks = BlockKinds()
}

/// One conversation: the requests whose prompts each extend the one before (an agent's turns,
/// one request per tool result), added up. A harness sends hundreds of requests for a handful
/// of conversations, and it is the conversations one wants to see.
public struct ConversationStats: Sendable, Codable {
    public let id: Int
    public var label: String
    /// Requests finished so far.
    public var turns = 0
    /// The latest prompt's length: the conversation's context.
    public var promptTokens = 0
    /// Summed over its requests.
    public var reusedTokens = 0
    public var evaluatedTokens = 0
    public var completionTokens = 0
    public var verifySteps = 0
    public var queueSeconds = 0.0
    public var prefillSeconds = 0.0
    public var decodeSeconds = 0.0
    public var thinkingTokens = 0
    public var answerTokens = 0
    public var toolCallTokens = 0
    public var draftedTokens = 0
    public var acceptedTokens = 0
    public var blocks = BlockKinds()
    /// Time outside the engine's steps, summed over its requests.
    public var overhead = RequestOverhead()
    public var lastFinish = ""
    public var endedSecondsAgo = 0.0
    /// A request of this conversation is queued or being worked on now.
    public var running = false
}

public struct SchedulerStats: Sendable, Codable {
    public let sessions: [SessionStats]
    public let queued: Int
    public let cached: [CachedPrefixStats]
    public let memory: MemoryStats
    public let throughput: ThroughputStats
    public let totals: TotalStats
    public let maxSlots: Int
    public let maxRows: Int
    public let maxContext: Int
    public let speculative: Bool
    public let uptimeSeconds: Double
    /// Present when prompt prefixes are also kept on disk.
    public var prefixStore: PrefixStoreStats?
    /// Conversations, most recently active first.
    public var conversations: [ConversationStats] = []
    /// Engine steps since the server started, by row tier and context.
    public var stepTiers: [StepTierStats] = []
    /// The model loaded, when the server is told which it is (the scheduler is not).
    public var model: LoadedModelStats?
}

/// Which of the server's models these are the stats of.
public struct LoadedModelStats: Sendable, Codable {
    public let id: String
    public let path: String
    /// Another model is to be loaded in this one's place.
    public let switching: Bool

    public init(id: String, path: String, switching: Bool) {
        self.id = id; self.path = path; self.switching = switching
    }
}

public final class BatchScheduler: @unchecked Sendable {
    private final class Session {
        let id: Int
        let request: GenerationRequest
        let continuation: AsyncStream<GenerationEvent>.Continuation
        var sampler: Sampler
        var slot = -1
        var cached = 0
        var fed = 0
        var generated = 0
        var pending: Int?
        /// Made with the reply's first token (see `next`).
        var thinkingWatch: ThinkingWatch?
        var limit: ReplyLimit?
        var verifySteps = 0
        /// What the session's blocks have yielded so far, and the text drafts can be copied from.
        var speculation = SpeculationState()
        /// Proposals from outside the session's text, to be scored when it ends (DictionaryStudy).
        var studyNotes: [DictionaryStudy.Note] = []
        /// Where a reused prefix came from, when not from the slot's own text (for the log).
        var reusedFrom: String?
        /// Positions at which this session's prefill keeps the recurrent state: in memory and
        /// on disk, or on disk only.
        var stateMarks: Set<Int> = []
        var diskMarks: Set<Int> = []
        /// Where the slot's checkpoint is taken: the prompt's stable point (see
        /// GenerationRequest.stablePromptTokens), a few tokens short of its end. The next
        /// request of the conversation begins with the prompt up to there however it renders
        /// the reply this prompt opened; with the thinking dropped it does not begin with the
        /// whole prompt, and a checkpoint at the very end would be of no use to it.
        var turnMark: Int?
        /// The slot's checkpoint is already where this prompt would put it.
        var checkpointKept = false
        /// KV pages the session was admitted for: its prompt and some generation.
        var wantedPages = 0
        var lastProgress = Date()
        var conversation = 0
        var turn = 0
        var drafted = 0
        var accepted = 0
        var blocks = BlockKinds()
        var overhead: RequestOverhead
        var steppedOnce = false
        /// (time, prompt tokens evaluated, tokens generated) a moment ago, for current rates.
        var random: SplitMix
        let submitted = Date()
        var admitted: Date?
        var prefillDone: Date?

        init(id: Int, request: GenerationRequest, continuation: AsyncStream<GenerationEvent>.Continuation) {
            self.id = id; self.request = request; self.continuation = continuation
            self.sampler = Sampler(parameters: request.sampling, vocabLimit: request.vocabLimit)
            self.random = SplitMix(seed: (request.sampling.seed ?? UInt64.random(in: 1...UInt64.max)) ^ 0xD1F7)
            self.overhead = RequestOverhead()
            self.overhead.prepare = request.prepareSeconds
        }
    }

    private static func seconds(since start: UInt64) -> Double { Double(DispatchTime.now().uptimeNanoseconds - start) / 1e9 }
    private static func now() -> UInt64 { DispatchTime.now().uptimeNanoseconds }

    private struct Checkpoint { let tokens: [Int]; let state: Data }
    /// Where the slot chosen for a prompt gets its reused prefix from.
    private enum ReuseSource {
        case nothing
        /// The slot's whole evaluated text is a prefix of the prompt.
        case tokens
        /// The slot's state at the end of its last prompt.
        case promptEnd
        /// A shared prefix held by `from`, which may be another slot and may be busy.
        case boundary(from: Int)
    }
    private enum Planned {
        case decode(Session)
        case verify(Session, DraftProposal)
        case prefill(Session, count: Int, complete: Bool)
    }

    private let engine: Engine
    /// Present when speculative decoding is enabled.
    private let draft: DraftModel?
    private let condition = NSCondition()
    private var inbox: [Session] = []
    /// Sessions submitted and not yet finished, wherever they are (see `requestsInFlight`).
    private var inFlight = 0
    private var stopping = false
    private var cancelRequests = Set<Int>()
    /// Sessions the last plan left out because the KV pool had no page for them.
    private var waitingForMemory: [Session] = []
    private var nextID = 1
    private var published: SchedulerStats
    /// The activity of each session in `published`, by its id there.
    private var publishedActivities = SessionActivities()

    // Owned by the scheduler thread.
    private var waiting: [Session] = []
    private var active: [Session] = []
    private var slotTokens: [[Int]]
    private var slotBusy: [Bool]
    private var slotLastUsed: [Date]
    private var checkpoints: [Checkpoint?]
    /// The copy-draft index over each slot's text, kept between its requests (see
    /// `LookupIndex.extend`); a session holds its slot's while it decodes.
    private var slotLookups: [LookupIndex]
    /// The state at a slot's shared prefix (see GenerationRequest.checkpointHints): valid for
    /// as long as the slot's text starts with those tokens.
    private var boundaries: [Checkpoint?]
    private var totals = TotalStats()
    /// Which conversation a request belongs to, and what each conversation has added up to.
    private var ledger = ConversationLedger(capacity: 200)
    private var conversations: [Int: (stats: ConversationStats, lastActive: Date)] = [:]

    private func joinConversation(_ session: Session) {
        let joined = ledger.join(session.request.promptTokens, stable: session.request.stablePromptTokens)
        session.conversation = joined.id
        if joined.isNew { conversations[joined.id] = (ConversationStats(id: joined.id, label: session.request.label), Date()) }
        for dropped in joined.dropped { conversations[dropped] = nil }
        conversations[joined.id]?.lastActive = Date()
        session.turn = (conversations[joined.id]?.stats.turns ?? 0) + 1
    }
    /// The steps of the last few seconds: when each began and ended, the tokens it evaluated
    /// and generated, and which sessions they were for. Every "now" figure is read from this,
    /// the server's and each session's alike, so that they agree.
    private typealias StepShare = (id: Int, prefill: Int, decode: Int, drafted: Int, accepted: Int)
    private var window: [(start: Double, end: Double, prefill: Int, decode: Int, shares: [StepShare])] = []
    /// How far back the rates look, and how far back the share of drafts accepted does.
    private static let rateSpan = 3.0, draftSpan = 5.0
    private var cycleStarted = 0.0
    private var lastStep: (rows: Int, seconds: Double) = (0, 0)
    /// When the last engine step ended (uptime nanoseconds; 0 before the first).
    private var lastStepEnded: UInt64 = 0
    private struct TierKey: Hashable { let rows: Int; let context: Int; let prompt: Bool }
    private var tiers: [TierKey: (steps: Int, seconds: Double, gpu: Double, cycle: Double, fastest: Double)] = [:]
    private static let contextRanges = ["<16K", "16-64K", ">64K"]
    private static func contextRange(_ position: Int) -> Int { position < 16_384 ? 0 : position < 65_536 ? 1 : 2 }
    private let started = Date()
    private let trace = ProcessInfo.processInfo.environment["SPLOSH_TRACE"] != nil
    private var traceTotals = (0.0, 0.0, 0.0, 0.0)
    private var traceSteps = 0, traceRows = 0

    /// Rows granted to prefill in a step that also carries decode rows: at least this many,
    /// then as many more as fit in the accelerator tiles the step is then paying for anyway.
    /// The floor is what bounds the cost of someone else's prompt to a running stream.
    public var prefillRowsWhileDecoding = 8

    /// What one generated token is worth in prompt tokens when a step has to be shared between
    /// sessions that are decoding and prompts that are waiting (see `prefillCap`). Low values
    /// favour getting waiting prompts evaluated, which is what a batch of agents wants; high
    /// values keep a running stream at full speed and make the others wait. At 1 a waiting
    /// prompt always gets a full-width step, with the decoding sessions' rows riding in it; at
    /// 3 the rule preferred 32-row steps once the prompt's context passed about 100K, which
    /// held prefill to 120 tok/s where a full step gives 180.
    public var decodeWeight = 1.0

    /// How many rows a step may hold when it carries both decode rows and prefill rows.
    ///
    /// Rows come in tiers (16, then tiles of 32), and a wider step is slower for everyone in
    /// it: with one session decoding, a 16-row step leaves prefill 8 rows and about a quarter
    /// of its full rate; a 128-row step gives prefill its full rate and the decoder a quarter
    /// of its own. Neither end is right in general. The width chosen is the one that makes the
    /// most progress per millisecond, counting a generated token as `decodeWeight` prompt
    /// tokens: wide while few sessions decode and much is waiting, narrow when many decode
    /// and a newcomer has a little to catch up.
    private func prefillCap(decodeRows: Int, decodeTokens: Double, decodeAttention: Double, prefillers: [Session]) -> Int {
        let backlog = prefillers.reduce(0) { $0 + $1.request.promptTokens.count - $1.fed }
        guard backlog > 0 else { return decodeRows }
        // Prefill rows pay for attention over their own context: 2.8 ms per thousand tokens
        // for 128 rows. The prompt with least left goes first, so its length stands for all.
        let perRow = 0.022 * Double(prefillers.first.map { $0.fed } ?? 0) / 1000
        var best = (cap: min(engine.config.maxRows, Self.tier(decodeRows + prefillRowsWhileDecoding)), rate: 0.0)
        for width in [16, 32, 64, 96, 128] where width <= engine.config.maxRows && width > decodeRows {
            let prefill = min(backlog, width - decodeRows)
            let milliseconds = Self.cycleMilliseconds(rows: decodeRows + prefill) + decodeAttention + perRow * Double(prefill)
            let rate = (decodeWeight * decodeTokens + Double(prefill)) / milliseconds
            if rate > best.rate { best = (width, rate) }
        }
        return best.cap
    }

    /// Prompt prefixes on disk: looked up when a prompt arrives, written when a session ends.
    public var prefixStore: PrefixStore?

    /// Most rows a step spends on speculative verification. Measured: a step costs about 4 ms
    /// per row up to 64 rows and far more beyond, so extra sessions take turns instead.
    public var maxVerifyRows = 64

    /// Draft double-length blocks where they pay (see `planBlocks`). SPLOSH_LONG_BLOCKS=0
    /// keeps every block at the checkpoint's own eight.
    public var longBlocks = ProcessInfo.processInfo.environment["SPLOSH_LONG_BLOCKS"] != "0"

    /// Draft from the session's own text where it is repeating itself or its prompt (see
    /// LookupIndex). SPLOSH_COPY_DRAFTS=0 leaves every block to the draft model.
    public var copyDrafts = ProcessInfo.processInfo.environment["SPLOSH_COPY_DRAFTS"] != "0"

    /// When set, every verify step is also looked up in text outside its session, and what
    /// that would have yielded is reported as requests finish (see DictionaryStudy). Nothing
    /// decoded changes. Set before the first request.
    public var dictionaryStudy: DictionaryStudy?

    /// Write a line to standard error for every finished request.
    public var requestLog = false

    /// Most requests worked on at once; the rest wait in the queue, in order of arrival, with
    /// progress events. Slots still hold every conversation's context, so this limits how the
    /// GPU is shared, not what is cached. At 1 requests run strictly one after another, each
    /// at full speed.
    public var maxActive = Int.max

    /// Tokens of a reply's limit kept for its answer: thinking that has used the rest is ended
    /// by the scheduler (see ThinkingWatch). 0 leaves the thinking to run to the limit.
    public var answerReserve = 8192

    /// Tokens a tool call in progress at a reply's limit may run past it, to be finished (see
    /// ReplyLimit). 0 ends every reply at its limit, whatever it is in the middle of.
    public var toolCallOverrun = 8192

    /// Seconds between progress events for a session that has not produced a token yet.
    public var progressInterval = 2.0

    /// Shared prefixes shorter than this are not worth a checkpoint (150 MB of state each).
    public var sharedPrefixMinTokens = 1024
    /// Text a slot holds beyond a shared prefix that may be dropped to reuse the slot in place;
    /// more than this is someone's conversation, and the prefix is copied to another slot.
    private static let cheapToLose = 2048
    /// Tokens a prompt must lose for want of a checkpoint before one is taken where it diverged.
    private static let worthACheckpoint = 512
    /// Checkpoints written to the disk store during a prefill: every `checkpointInterval`
    /// tokens up to `denseCheckpoints`, then every `denseCheckpoints`.
    private static let checkpointInterval = 4096
    private static let denseCheckpoints = 16384

    private static func commonPrefix(_ a: [Int], _ b: [Int]) -> Int {
        let limit = min(a.count, b.count)
        var count = 0
        while count < limit, a[count] == b[count] { count += 1 }
        return count
    }

    /// Milliseconds a decode-only step takes, by accelerator tile tier: up to 16 rows cost
    /// one figure, and beyond that rows come in tiles of 32. Only the ratios matter, and they
    /// come from the tile structure, so these are constants: correcting them from the steps
    /// that run made a hot machine's 32-row steps look as dear as a 64-row step nobody had
    /// tried yet, and the plan oscillated.
    private static let tierMilliseconds: [Int: Double] = [16: 62, 32: 86, 64: 165, 96: 230, 128: 295]
    private static func tier(_ rows: Int) -> Int { rows <= 16 ? 16 : ((rows + 31) / 32) * 32 }
    /// A whole cycle: the verify step, plus the draft pass, context push and CPU work.
    private static func cycleMilliseconds(rows: Int) -> Double {
        (tierMilliseconds[tier(rows)] ?? Double(rows) * 2.3) + 0.45 * Double(rows) + 10
    }

    /// Attention is the part of a verify step that grows with a session's context: milliseconds
    /// per thousand tokens for a block's first eight rows and for its second eight. Measured:
    /// an 8-row step is 62 ms at 1K, 78 at 50K, 105 at 150K; a 16-row one 65, 90, 141.
    private static let attentionMilliseconds = (first: 0.29, second: 0.22)

    /// Which sessions get a double block this step.
    ///
    /// A double block's first eight rows are isolated from the rest, so they are the same
    /// drafts an ordinary block would give; the second eight only count when the first seven
    /// drafts are all accepted. They cost rows. Inside a tile tier the step is already paying
    /// for (one session: rows 9-16) that is close to free; across a tier boundary it is not, so
    /// sessions are taken in order of what their tail is expected to yield for what it costs
    /// and the set that gives the most tokens per millisecond wins. At long context the cost
    /// is mostly the tail's attention: at 50K a second half adds 15% to a lone session's
    /// cycle, so it is only drafted where blocks are usually accepted whole.
    private func planBlocks(_ speculative: [Session], copying: Set<Int>, otherRows: Int) -> [Int: Int] {
        func stats(_ session: Session) -> SpeculationState.BlockStats {
            copying.contains(session.id) ? session.speculation.copied : session.speculation.drafted
        }
        let block = DraftModel.blockSize
        var blocks = [Int: Int](uniqueKeysWithValues: speculative.map { ($0.id, block) })
        guard longBlocks, DraftModel.maxBlock >= 2 * block else { return blocks }
        let base = speculative.count * block + otherRows
        func thousands(_ session: Session) -> Double { Double(slotTokens[session.slot].count) / 1000 }
        func tailCost(_ session: Session) -> Double { 0.45 * Double(block) + Self.attentionMilliseconds.second * thousands(session) }
        let candidates = speculative
            .filter { slotTokens[$0.slot].count + 2 * block < engine.config.maxContext }
            .sorted { stats($0).tail / tailCost($0) > stats($1).tail / tailCost($1) }
        var tokens = speculative.reduce(0.0) { $0 + stats($1).blockYield } + Double(otherRows)
        var attention = speculative.reduce(0.0) { $0 + Self.attentionMilliseconds.first * thousands($1) }
        let trace = ProcessInfo.processInfo.environment["SPLOSH_PLAN_TRACE"] != nil
        var best = (count: 0, rate: tokens / (Self.cycleMilliseconds(rows: base) + attention))
        for (index, session) in candidates.enumerated() {
            let rows = base + (index + 1) * block
            guard rows <= engine.config.maxRows, rows <= engine.config.maxLogitRows,
                  (speculative.count + index + 1) * block <= maxVerifyRows else { break }
            tokens += stats(session).tail
            attention += Self.attentionMilliseconds.second * thousands(session)
            // The estimates are noisy; moving up a tier has to clear a margin.
            let rate = tokens / (Self.cycleMilliseconds(rows: rows) + attention) * (Self.tier(rows) == Self.tier(base) ? 1 : 0.88)
            if rate > best.rate { best = (index + 1, rate) }
        }
        for session in candidates.prefix(best.count) { blocks[session.id] = 2 * block }
        if trace {
            writeStandardError("plan: \(speculative.count) sessions, base \(base) rows, \(best.count) doubled; tails \(candidates.map { String(format: "%.2f", stats($0).tail) }), \(copying.count) copying\n".data(using: .utf8)!)
        }
        return blocks
    }

    public init(engine: Engine, draft: DraftModel? = nil) {
        self.engine = engine
        self.draft = draft
        let slots = engine.config.maxSlots
        slotTokens = Array(repeating: [], count: slots)
        slotBusy = Array(repeating: false, count: slots)
        slotLastUsed = Array(repeating: .distantPast, count: slots)
        checkpoints = Array(repeating: nil, count: slots)
        boundaries = Array(repeating: nil, count: slots)
        slotLookups = Array(repeating: LookupIndex(), count: slots)
        published = SchedulerStats(sessions: [], queued: 0, cached: [], memory: Self.memoryStats(engine.memory, checkpointBytes: 0),
                                   throughput: ThroughputStats(prefillTokensPerSecond: 0, decodeTokensPerSecond: 0, stepsPerSecond: 0, lastStepRows: 0, lastStepMilliseconds: 0),
                                   totals: TotalStats(), maxSlots: slots, maxRows: engine.config.maxRows,
                                   maxContext: engine.config.maxContext, speculative: draft != nil, uptimeSeconds: 0)
        let thread = Thread { [weak self] in self?.run() }
        thread.name = "splosh.scheduler"
        thread.qualityOfService = .userInteractive
        thread.start()
    }

    // MARK: - Public API

    public func submit(_ request: GenerationRequest) -> AsyncStream<GenerationEvent> {
        AsyncStream(bufferingPolicy: .unbounded) { continuation in
            condition.lock()
            if stopping {
                // The scheduler's thread may have taken its last look at the inbox: answer here,
                // as a request cut by the stop is answered, for the client to send it again.
                condition.unlock()
                continuation.yield(.finished(.cancelled, GenerationUsage(), message: "server shutting down"))
                continuation.finish()
                return
            }
            let session = Session(id: nextID, request: request, continuation: continuation)
            nextID += 1
            inbox.append(session)
            inFlight += 1
            condition.signal()
            condition.unlock()
            let id = session.id
            continuation.onTermination = { [weak self] _ in
                guard let self else { return }
                self.condition.lock()
                self.cancelRequests.insert(id)
                self.condition.signal()
                self.condition.unlock()
            }
        }
    }

    public func stats() -> SchedulerStats {
        condition.lock(); defer { condition.unlock() }
        return published
    }

    /// Requests submitted whose replies have not ended: queued, being worked on, or about to be.
    /// At 0 the engine can be stopped with nothing cut (the stats are a step behind; this is not).
    public var requestsInFlight: Int {
        condition.lock(); defer { condition.unlock() }
        return inFlight
    }

    /// What the parser of a session's output has recorded, the session named by its id in the
    /// stats: one that is in them, or left them lately. Nil for any other, and for one whose
    /// output nobody parses.
    public func activity(of session: Int) -> SessionActivity? {
        condition.lock(); defer { condition.unlock() }
        return publishedActivities[session]
    }

    /// Seconds between small passes over the KV pool while idle after work (0: none). The pool
    /// is outside the residency set, and after a few seconds idle the first step waits for the
    /// driver to make it resident again: measured at 35K context after 3 s idle, a turn's first
    /// step took 560-620 ms without these passes and 325-340 with them (its steady figure is
    /// 390). SPLOSH_KEEPWARM sets the interval; 0 turns it off.
    private let keepWarmInterval = Double(ProcessInfo.processInfo.environment["SPLOSH_KEEPWARM"] ?? "") ?? 0.5
    /// How long after the last step the passes go on: an agent's tool call, not a night.
    private static let keepWarmSeconds = 600.0

    /// Change the settings of a running scheduler (`requestLog`, `decodeWeight`, `maxActive`,
    /// `dictionaryStudy`): `change` runs on the scheduler thread, between steps.
    public func update(_ change: @escaping @Sendable (BatchScheduler) -> Void) {
        condition.lock(); changes.append(change); condition.signal(); condition.unlock()
    }
    private var changes: [@Sendable (BatchScheduler) -> Void] = []

    /// End every session and stop the scheduler; it writes each conversation's context to the
    /// disk store on its way out.
    public func shutdown() {
        condition.lock(); stopping = true; condition.signal(); condition.unlock()
    }

    /// `shutdown`, then wait for the contexts to be on disk. Returns how many were written.
    public func shutdownAndSave(timeout: TimeInterval = 300) -> Int {
        shutdown()
        _ = stopped.wait(timeout: .now() + timeout)
        return savedAtShutdown
    }
    private let stopped = DispatchSemaphore(value: 0)
    private var savedAtShutdown = 0

    // MARK: - Scheduler thread

    private func run() {
        while true {
            condition.lock()
            while inbox.isEmpty && active.isEmpty && waiting.isEmpty && cancelRequests.isEmpty && changes.isEmpty && !stopping {
                let warm = keepWarmInterval > 0 && lastStepEnded > 0 && Self.seconds(since: lastStepEnded) < Self.keepWarmSeconds
                if window.isEmpty && !warm { condition.wait(); continue }
                // Idle with recent activity: wake once a second so published rates decay to zero,
                // and keep the KV pool resident (see `keepWarmInterval`).
                _ = condition.wait(until: Date().addingTimeInterval(warm ? keepWarmInterval : 1))
                if inbox.isEmpty && !stopping {
                    condition.unlock()
                    if warm { engine.wake() }
                    publish()
                    condition.lock()
                }
            }
            let stop = stopping
            let arrived = inbox
            waiting.append(contentsOf: inbox)
            totals.requests += inbox.count
            inbox.removeAll()
            let cancelledIDs = cancelRequests
            cancelRequests.removeAll()
            let changed = changes
            changes.removeAll()
            condition.unlock()
            for change in changed { change(self) }
            for session in arrived {
                let start = Self.now()
                joinConversation(session)
                session.overhead.join = Self.seconds(since: start)
            }

            if stop {
                // A prompt still being evaluated is kept as far as it got: its request, sent
                // again, carries on from there instead of from the last turn's end. (Without the
                // draft's context, which is collected only near a prompt's end.)
                var partial = Set<Int>()
                for session in active where session.prefillDone == nil && session.slot >= 0 && session.fed > session.cached
                    && slotTokens[session.slot].count == session.fed {
                    guard let state = try? engine.exportState(session.slot) else { continue }
                    checkpoints[session.slot] = Checkpoint(tokens: slotTokens[session.slot], state: state)
                    partial.insert(session.slot)
                }
                let cut = Set(active.map(\.slot))
                for session in waiting + active { finish(session, .cancelled, message: "server shutting down") }
                // Every conversation as it stands, whether or not something close to it is
                // stored already: the next server finds each one where it was. One that was
                // idle goes whole, reply included; one cut off in the middle goes as far as
                // its checkpoint, since its request will be sent again.
                for slot in slotBusy.indices where !slotTokens[slot].isEmpty {
                    if cut.contains(slot) {
                        if persistPrefix(slot: slot, exactly: true, draftContext: !partial.contains(slot)) { savedAtShutdown += 1 }
                    } else {
                        retire(slot)
                        savedAtShutdown += 1
                    }
                }
                prefixStore?.flush()
                stopped.signal()
                return
            }
            for session in waiting where cancelledIDs.contains(session.id) { finish(session, .cancelled, message: nil) }
            for session in active where cancelledIDs.contains(session.id) { finish(session, .cancelled, message: nil) }
            // A session with nothing to show yet reports its progress instead.
            let now = Date()
            for session in waiting + active where session.prefillDone == nil && now.timeIntervalSince(session.lastProgress) >= progressInterval {
                session.lastProgress = now
                session.continuation.yield(.progress(evaluated: max(session.fed, session.cached), total: session.request.promptTokens.count))
            }
            admit()

            do {
                let t0 = DispatchTime.now().uptimeNanoseconds
                cycleStarted = Date().timeIntervalSinceReferenceDate
                let (rows, plan) = try buildRows()
                guard !rows.isEmpty else {
                    if !waitingForMemory.isEmpty { relieveMemory() }
                    publish(); continue
                }
                let t1 = DispatchTime.now().uptimeNanoseconds
                let step = try engine.step(rows, captureFeatures: draft != nil)
                let t2 = DispatchTime.now().uptimeNanoseconds
                let idle = lastStepEnded > 0 && t1 - lastStepEnded > 50_000_000 ? Double(t1 - lastStepEnded) / 1e9 : 0
                lastStepEnded = t2
                for entry in plan {
                    let session: Session
                    switch entry { case .decode(let s), .verify(let s, _), .prefill(let s, _, _): session = s }
                    guard !session.steppedOnce else { continue }
                    session.steppedOnce = true
                    session.overhead.firstStep = step.wallSeconds
                    session.overhead.idleBeforeFirstStep = idle
                }
                try consume(plan, step: step)
                let t3 = DispatchTime.now().uptimeNanoseconds
                let carriesPrompt = plan.contains { if case .prefill = $0 { return true } else { return false } }
                let key = TierKey(rows: Self.tier(rows.count), context: Self.contextRange(rows.map(\.position).max() ?? 0), prompt: carriesPrompt)
                var tier = tiers[key] ?? (0, 0, 0, 0, .infinity)
                tier.steps += 1; tier.seconds += step.wallSeconds; tier.gpu += step.gpuSeconds
                tier.cycle += Double(t3 - t0) / 1e9; tier.fastest = min(tier.fastest, step.wallSeconds)
                tiers[key] = tier
                totals.busySeconds += Double(t3 - t0) / 1e9
                if window.last.map({ $0.prefill > 0 }) ?? false { totals.prefillBusySeconds += Double(t3 - t0) / 1e9 }
                if window.last.map({ $0.decode > 0 }) ?? false { totals.decodeBusySeconds += Double(t3 - t0) / 1e9 }
                if trace {
                    traceTotals.0 += Double(t1 - t0) / 1e6; traceTotals.1 += Double(t2 - t1) / 1e6
                    traceTotals.2 += Double(t3 - t2) / 1e6; traceTotals.3 += step.gpuSeconds * 1000; traceSteps += 1; traceRows += rows.count
                    if traceSteps == 20 {
                        writeStandardError(Data(String(format: "trace: rows %.0f  draft+build %.1f ms  step %.1f ms (gpu %.1f)  consume %.1f ms\n",
                            Double(traceRows) / 20, traceTotals.0 / 20, traceTotals.1 / 20, traceTotals.3 / 20, traceTotals.2 / 20).utf8))
                        traceTotals = (0, 0, 0, 0); traceSteps = 0; traceRows = 0
                    }
                }
            } catch EngineError.kvPoolExhausted {
                // The plan reserves its pages, so this is not expected; make room and go on.
                if !evictIdleSlot() { relieveMemory() }
            } catch {
                for session in active { finish(session, .error, message: String(describing: error)) }
            }
            publish()
        }
    }

    private func admit() {
        var index = 0
        while index < waiting.count, active.count < maxActive {
            let session = waiting[index]
            let prompt = session.request.promptTokens
            let admitStart = Self.now(), copiedBefore = storeCopySeconds
            if prompt.isEmpty {
                waiting.remove(at: index); finish(session, .error, message: "empty prompt"); continue
            }
            if prompt.count >= engine.config.maxContext {
                waiting.remove(at: index)
                finish(session, .error, message: "context_length_exceeded: prompt is \(prompt.count) tokens; limit \(engine.config.maxContext)")
                continue
            }
            guard let choice = chooseSlot(for: prompt) else { index += 1; continue }
            let slot = choice.slot
            var reuse = choice.reuse
            // How far the prompt runs along something whose KV is to hand: a slot's text or a
            // stored prompt. Where that stops is where this prompt parts from everything seen
            // so far, and the furthest a state checkpoint could take it.
            let inMemory = slotBusy.indices.map { (slot: $0, common: Self.commonPrefix(slotTokens[$0], prompt)) }.max { $0.common < $1.common }
            let onDisk = prefixStore?.longestCommonPrefix(with: prompt)
            let divergence = min(max(inMemory?.common ?? 0, onDisk?.common ?? 0), prompt.count - 1)
            // Three ways to start, best first: a state checkpoint with KV from whichever prompt
            // covers it, a whole stored prompt, or what the slot choice found in memory.
            var stored: PrefixStore.Entry?
            var checkpoint: PrefixStore.Entry?
            if let store = prefixStore {
                stored = store.longestPrefix(of: prompt, longerThan: reuse + Engine.pageTokens)
                checkpoint = store.bestState(for: prompt, upTo: divergence, longerThan: max(reuse, stored?.tokenCount ?? 0) + Engine.pageTokens)
                if checkpoint != nil { stored = nil }
                reuse = checkpoint?.tokenCount ?? stored?.tokenCount ?? reuse
            }
            // Whether the slot keeps its own pages, and which slots are about to be read from.
            var inPlace = false
            var sources: Set<Int> = [slot]
            if let checkpoint {
                if let inMemory, inMemory.common >= checkpoint.tokenCount { inPlace = inMemory.slot == slot; sources.insert(inMemory.slot) }
            } else if stored == nil {
                switch choice.source {
                case .nothing: break
                case .tokens, .promptEnd: inPlace = true
                case .boundary(let from): inPlace = from == slot; sources.insert(from)
                }
            }
            // Room in the KV pool for the prompt and some generation, or the session waits.
            // What is available: free pages, what this slot gives back, and idle
            // conversations, which are evicted (least recently used first) to make room.
            let wanted = Self.pages(min(prompt.count + min(session.request.maxTokens, Self.generationHeadroom), engine.config.maxContext))
            guard wanted <= engine.config.kvPages else {
                waiting.remove(at: index)
                finish(session, .error, message: "context_length_exceeded: the prompt needs \(wanted * Engine.pageTokens) tokens of KV cache and the pool holds \(engine.config.kvPages * Engine.pageTokens); raise kvPages")
                continue
            }
            let kept = inPlace ? min(Self.pages(reuse), engine.pagesHeld(by: slot)) : 0
            let released = engine.pagesHeld(by: slot) - kept
            var evictable = 0
            for other in slotBusy.indices where !sources.contains(other) && !slotBusy[other] { evictable += engine.pagesHeld(by: other) }
            // Pages are taken as a session goes, so what the sessions already running were
            // admitted for and have not taken yet is spoken for.
            let promised = active.reduce(0) { $0 + max(0, $1.wantedPages - engine.pagesHeld(by: $1.slot)) }
            guard wanted - kept <= engine.freePageCount - promised + released + evictable else {
                index += 1; continue
            }
            while wanted - kept > engine.freePageCount - promised + released, evictIdleSlot(except: sources) {}
            session.wantedPages = wanted
            waiting.remove(at: index)
            // A slot kept in place through a shared prefix (a disk checkpoint, or its own boundary
            // when no other slot is free) may hold another conversation, which goes to disk first:
            // `finish` does not always store one (see there).
            var throughShared = checkpoint != nil
            if stored == nil, checkpoint == nil, case .boundary = choice.source { throughShared = true }
            let displaced = throughShared && inPlace && slotTokens[slot].count > reuse + Self.cheapToLose
            if !inPlace || displaced { retire(slot) }
            do {
                if let store = prefixStore, let entry = checkpoint {
                    // The state from the checkpoint, the KV from whichever prompt covers it.
                    let state = Data(try store.load(entry).snapshot.state)
                    if inPlace {
                        try engine.importState(slot, state: state)
                        engine.truncate(slot, tokenCount: reuse)
                        draft?.truncate(slot, tokenCount: reuse)
                    } else {
                        if let inMemory, inMemory.common >= reuse {
                            try engine.importSlot(slot, tokenCount: reuse, state: state, kvFrom: inMemory.slot)
                        } else {
                            // Read from the mapped file straight into the pool.
                            let source = onDisk!.entry
                            try engine.importSlot(slot, snapshot: SlotSnapshot(tokenCount: reuse, state: state, kv: try store.load(source).snapshot.kv),
                                                  kvStoredTokens: source.tokenCount)
                        }
                        draft?.resetSlot(slot)
                    }
                    slotTokens[slot] = Array(prompt[0..<reuse])
                    checkpoints[slot] = nil
                    boundaries[slot] = Checkpoint(tokens: slotTokens[slot], state: state)
                    session.reusedFrom = "a checkpoint"
                } else if let store = prefixStore, let entry = stored {
                    // A prefix on disk that is worth more than what is in memory replaces it.
                    let loaded = try store.load(entry)
                    try engine.importSlot(slot, snapshot: loaded.snapshot)
                    draft?.resetSlot(slot)
                    if let context = loaded.draft { draft?.importContext(slot, snapshot: context) }
                    slotTokens[slot] = entry.tokens.map(Int.init)
                    // A copy: the loaded snapshot is a mapping of the whole file.
                    checkpoints[slot] = Checkpoint(tokens: slotTokens[slot], state: Data(loaded.snapshot.state))
                    boundaries[slot] = nil
                    session.reusedFrom = "disk"
                } else {
                    switch choice.source {
                    case .nothing:
                        engine.resetSlot(slot)
                        draft?.resetSlot(slot)
                        slotTokens[slot] = []
                        checkpoints[slot] = nil
                        boundaries[slot] = nil
                    case .tokens:
                        break
                    case .promptEnd:
                        let state = checkpoints[slot]!
                        try engine.importState(slot, state: state.state)
                        engine.truncate(slot, tokenCount: reuse)
                        draft?.truncate(slot, tokenCount: reuse)
                        slotTokens[slot] = state.tokens
                    case .boundary(let from):
                        let boundary = boundaries[from]!
                        if from == slot {
                            try engine.importState(slot, state: boundary.state)
                            engine.truncate(slot, tokenCount: reuse)
                            draft?.truncate(slot, tokenCount: reuse)
                        } else {
                            // The prefix's KV does not change while its slot carries on, so it
                            // can be copied out from under a running session.
                            try engine.importSlot(slot, tokenCount: reuse, state: boundary.state, kvFrom: from)
                            draft?.resetSlot(slot)
                            boundaries[slot] = boundary
                        }
                        slotTokens[slot] = boundary.tokens
                        checkpoints[slot] = nil
                        session.reusedFrom = "a shared prefix"
                    }
                }
            } catch {
                finish(session, .error, message: String(describing: error)); continue
            }
            // Where this prefill keeps its state. The server's hints (the end of the system
            // block); the point where the prompt left what was known, if that is well past
            // what could be reused, so the next prompt to leave there need not start so far
            // back; and, for the disk store, regular positions, close together through the
            // part prompts tend to share and sparse after it.
            var marks = Set(session.request.checkpointHints)
            if divergence > reuse + Self.worthACheckpoint { marks.insert(divergence) }
            session.stateMarks = marks.filter { $0 > reuse && $0 >= sharedPrefixMinTokens && $0 < prompt.count }
            session.diskMarks = []
            if prefixStore != nil {
                var position = Self.checkpointInterval
                while position < prompt.count {
                    if position > reuse, position >= sharedPrefixMinTokens { session.diskMarks.insert(position) }
                    position += position < Self.denseCheckpoints ? Self.checkpointInterval : Self.denseCheckpoints
                }
            }
            if let stable = session.request.stablePromptTokens, stable > 0, stable < prompt.count {
                if stable > reuse {
                    session.turnMark = stable
                } else if stable == reuse, checkpoints[slot]?.tokens.count == stable {
                    session.checkpointKept = true                    // the same prompt again
                }
            }
            slotBusy[slot] = true
            session.slot = slot
            session.cached = reuse
            session.fed = reuse
            session.admitted = Date()
            let copied = storeCopySeconds - copiedBefore
            session.overhead.storeCopy += copied
            session.overhead.admit += Self.seconds(since: admitStart) - copied
            active.append(session)
        }
    }

    /// Time spent copying conversations out for the disk store (`persistPrefix`, `retire`).
    private var storeCopySeconds = 0.0

    /// Longest reusable prefix; otherwise an empty slot; otherwise the least recently used idle
    /// slot. Returns nil when every slot is busy.
    ///
    /// An idle slot can be carried on in place: from all of its text, or from the state at the
    /// end of its last prompt. A shared prefix can come from any slot, busy or not, and is
    /// copied to a free slot unless its own slot is idle and has little else to lose: starting
    /// a conversation from another one's system block should not cost that one its context.
    private func chooseSlot(for prompt: [Int]) -> (slot: Int, reuse: Int, source: ReuseSource)? {
        var inPlace: (slot: Int, reuse: Int, source: ReuseSource)?
        var empty: Int?
        var idleByAge: [Int] = []
        for slot in slotBusy.indices where !slotBusy[slot] {
            let tokens = slotTokens[slot]
            if tokens.isEmpty { empty = empty ?? slot }
            idleByAge.append(slot)
            // The slot must be a proper prefix: at least one token has to be evaluated to
            // produce logits for the first sampled token.
            if !tokens.isEmpty, tokens.count < prompt.count, prompt.starts(with: tokens) {
                if inPlace == nil || tokens.count > inPlace!.reuse { inPlace = (slot, tokens.count, .tokens) }
            } else if let checkpoint = checkpoints[slot], !checkpoint.tokens.isEmpty,
                      checkpoint.tokens.count < prompt.count, prompt.starts(with: checkpoint.tokens) {
                if inPlace == nil || checkpoint.tokens.count > inPlace!.reuse { inPlace = (slot, checkpoint.tokens.count, .promptEnd) }
            }
        }
        idleByAge.sort { slotLastUsed[$0] < slotLastUsed[$1] }
        var shared: (slot: Int, reuse: Int)?
        for slot in slotBusy.indices {
            guard let boundary = boundaries[slot], prompt.count > boundary.tokens.count, prompt.starts(with: boundary.tokens) else { continue }
            // The longest; between equals, an idle holder, which may need no copy.
            let length = boundary.tokens.count
            if let current = shared, length < current.reuse || (length == current.reuse && (slotBusy[slot] || !slotBusy[current.slot])) { continue }
            shared = (slot, length)
        }
        if let shared, shared.reuse > (inPlace?.reuse ?? 0) {
            let holder = shared.slot
            let idle = !slotBusy[holder]
            if idle, slotTokens[holder].count - shared.reuse <= Self.cheapToLose { return (holder, shared.reuse, .boundary(from: holder)) }
            if let destination = empty ?? idleByAge.first(where: { $0 != holder }) { return (destination, shared.reuse, .boundary(from: holder)) }
            if idle { return (holder, shared.reuse, .boundary(from: holder)) }
        }
        if let inPlace { return inPlace }
        if let empty { return (empty, 0, .nothing) }
        if let oldest = idleByAge.first { return (oldest, 0, .nothing) }
        return nil
    }

    /// Write the slot's prompt-boundary prefix to disk if it is long enough and not there yet.
    /// The KV of a prefix is untouched by what was generated after it, and the recurrent state
    /// at the boundary is the slot's checkpoint.
    @discardableResult
    private func persistPrefix(slot: Int, exactly: Bool = false, draftContext: Bool = true) -> Bool {
        guard let store = prefixStore, let checkpoint = checkpoints[slot],
              slotTokens[slot].count >= checkpoint.tokens.count,
              slotTokens[slot].starts(with: checkpoint.tokens), store.wants(checkpoint.tokens, exactly: exactly) else { return false }
        let start = Self.now()
        defer { storeCopySeconds += Self.seconds(since: start) }
        let count = checkpoint.tokens.count
        let snapshot = SlotSnapshot(tokenCount: count, state: checkpoint.state, kv: engine.exportKV(slot, tokenCount: count))
        store.save(tokens: checkpoint.tokens, snapshot: snapshot, draft: draftContext ? draft?.exportContext(slot, tokenCount: count) : nil)
        return true
    }

    /// An idle slot is about to be given to something else. Its conversation goes to disk as it
    /// stands, reply and all (the state is the slot's own, at the end of its text), with the
    /// checkpoint where its last prompt opened the reply as a state on its own: the next request
    /// either carries the text on exactly or parts from it there. Otherwise a conversation that
    /// comes back after its slot was taken starts from whatever older prefix the store has.
    private func retire(_ slot: Int) {
        guard let store = prefixStore, !slotBusy[slot], !slotTokens[slot].isEmpty else { return }
        let start = Self.now()
        defer { storeCopySeconds += Self.seconds(since: start) }
        let tokens = slotTokens[slot]
        if let checkpoint = checkpoints[slot], checkpoint.tokens.count < tokens.count, tokens.starts(with: checkpoint.tokens),
           store.wantsState(checkpoint.tokens) {
            store.saveState(tokens: checkpoint.tokens, state: checkpoint.state)
        }
        guard store.wants(tokens, exactly: true), let state = try? engine.exportState(slot) else { return }
        store.save(tokens: tokens, snapshot: SlotSnapshot(tokenCount: tokens.count, state: state, kv: engine.exportKV(slot, tokenCount: tokens.count)),
                   draft: draft?.exportContext(slot, tokenCount: tokens.count))
    }

    /// Tokens of generation a session is given room for when it is admitted.
    private static let generationHeadroom = 4096
    private static func pages(_ tokens: Int) -> Int { (tokens + Engine.pageTokens - 1) / Engine.pageTokens }

    private func evictIdleSlot(except keep: Set<Int> = []) -> Bool {
        var victim: Int?
        for slot in slotBusy.indices where !keep.contains(slot) && !slotBusy[slot] && !slotTokens[slot].isEmpty {
            if victim == nil || slotLastUsed[slot] < slotLastUsed[victim!] { victim = slot }
        }
        guard let victim else { return false }
        retire(victim)
        engine.resetSlot(victim)
        draft?.resetSlot(victim)
        slotTokens[victim] = []
        checkpoints[victim] = nil
        boundaries[victim] = nil
        slotLookups[victim] = LookupIndex()
        return true
    }

    /// Every active session is waiting for KV pages and nothing idle is left to evict, so none
    /// of them can finish and free any. Put the newest prompt still being evaluated back in
    /// the queue (it is admitted again when there is room for all of it); if they are all
    /// generating, end the one with the most context as if it had reached its length limit.
    private func relieveMemory() {
        if let session = active.last(where: { $0.pending == nil }) {
            engine.resetSlot(session.slot)
            draft?.resetSlot(session.slot)
            slotTokens[session.slot] = []
            checkpoints[session.slot] = nil
            boundaries[session.slot] = nil
            slotBusy[session.slot] = false
            active.removeAll { $0 === session }
            session.slot = -1; session.cached = 0; session.fed = 0; session.admitted = nil; session.wantedPages = 0
            waiting.append(session)
        } else if let session = active.max(by: { slotTokens[$0.slot].count < slotTokens[$1.slot].count }) {
            // Not a reply that reached its length: cut, for the client to send it again.
            finish(session, .cancelled, message: nil)
        }
        waitingForMemory.removeAll()
    }

    private func buildRows() throws -> ([EngineRow], [Planned]) {
        var rows: [EngineRow] = []
        var plan: [Planned] = []
        let block = DraftModel.blockSize
        let decoders = Array(active.filter { $0.pending != nil }.prefix(engine.config.maxSlots))
        var logitRows = 0
        // What the decoding sessions are expected to yield this step, and what their attention
        // costs: one side of the bargain with waiting prompts (see `prefillCap`).
        var decodeTokens = 0.0, decodeAttention = 0.0
        // KV pages this step's rows will take. A session whose rows need more than the pool
        // can give (after evicting idle conversations) sits the step out and waits.
        var reserved = 0
        waitingForMemory.removeAll(keepingCapacity: true)
        func reserve(_ session: Session, through position: Int) -> Bool {
            let need = position / Engine.pageTokens + 1 - engine.pagesHeld(by: session.slot)
            guard need > 0 else { return true }
            while need > engine.freePageCount - reserved, evictIdleSlot() {}
            guard need <= engine.freePageCount - reserved else { waitingForMemory.append(session); return false }
            reserved += need
            return true
        }

        // Speculative sessions verify a drafted block; the rest (no draft model, or too close to
        // the context limit for a whole block) decode one token.
        var speculative: [Session] = []
        var plain: [Session] = []
        for session in decoders {
            let position = slotTokens[session.slot].count
            let fits = (speculative.count + 1) * block <= maxVerifyRows
                && rows.count + (speculative.count + 1) * block + plain.count <= engine.config.maxRows
                && logitRows + (speculative.count + 1) * block + plain.count <= engine.config.maxLogitRows
            if draft != nil, position + block < engine.config.maxContext {
                // Room for a double block is reserved; a single one may use less.
                if fits, reserve(session, through: position + 2 * block - 1) { speculative.append(session) }   // otherwise it waits for the next step
            } else if reserve(session, through: position) {
                plain.append(session)
            }
        }
        if let draft, !speculative.isEmpty {
            // Blocks copied from a session's own text, where that looks better than the draft model.
            var copies: [Int: (match: Int, tokens: [Int])] = [:]
            if copyDrafts {
                for session in speculative {
                    copies[session.id] = session.speculation.copy(after: session.pending!, count: DraftModel.maxBlock - 1)
                }
            }
            // A waiting prompt's rows come first: they are certain progress, a tail is a bet.
            let prefilling = active.contains { $0.pending == nil }
            let blocks = prefilling ? [Int: Int](uniqueKeysWithValues: speculative.map { ($0.id, block) })
                                    : planBlocks(speculative, copying: Set(copies.keys), otherRows: plain.count)
            for session in speculative {
                decodeTokens += (copies[session.id] == nil ? session.speculation.drafted : session.speculation.copied).blockYield
                decodeAttention += Self.attentionMilliseconds.first * Double(slotTokens[session.slot].count) / 1000
            }
            var proposals: [Int: DraftProposal] = [:]
            for session in speculative {
                guard let copy = copies[session.id] else { continue }
                proposals[session.id] = DraftProposal(copied: Array(copy.tokens.prefix(blocks[session.id]! - 1)), match: copy.match,
                                                      sampling: session.request.sampling.temperature > 0)
            }
            // One draft pass per sampling temperature; sessions usually share one.
            for (temperature, group) in Dictionary(grouping: speculative.filter { copies[$0.id] == nil }, by: { $0.request.sampling.temperature }) {
                var random = group[0].random
                let drafted = try draft.propose(group.map { DraftRequest(slot: $0.slot, anchor: $0.pending!, start: slotTokens[$0.slot].count, block: blocks[$0.id]!,
                                                                         isolatePrefix: blocks[$0.id]! > DraftModel.blockSize) },
                                                temperature: max(0, temperature), vocabLimit: group[0].request.vocabLimit, random: &random)
                group[0].random = random
                for (session, proposal) in zip(group, drafted) { proposals[session.id] = proposal }
            }
            for session in speculative {
                let proposal = proposals[session.id]!
                let start = slotTokens[session.slot].count
                for (offset, token) in ([session.pending!] + proposal.tokens).enumerated() {
                    rows.append(EngineRow(slot: session.slot, token: token, position: start + offset, wantLogits: true, verify: true))
                }
                logitRows += proposal.tokens.count + 1
                plan.append(.verify(session, proposal))
            }
        }
        for session in plain {
            rows.append(EngineRow(slot: session.slot, token: session.pending!,
                                  position: slotTokens[session.slot].count, wantLogits: true))
            logitRows += 1
            plan.append(.decode(session))
        }

        if speculative.count < decoders.count {
            let served = Set(speculative.map(\.id))
            active = active.filter { !served.contains($0.id) } + active.filter { served.contains($0.id) }
        }
        let prefillers = active.filter { $0.pending == nil }
            .sorted { $0.request.promptTokens.count - $0.fed < $1.request.promptTokens.count - $1.fed }
        let cap = decoders.isEmpty ? engine.config.maxRows
            : prefillCap(decodeRows: rows.count, decodeTokens: decodeTokens + Double(plain.count), decodeAttention: decodeAttention, prefillers: prefillers)
        for session in prefillers {
            let budget = cap - rows.count
            guard budget > 0 else { break }
            let prompt = session.request.promptTokens
            var count = min(prompt.count - session.fed, budget)
            // A step ends exactly where the state is to be kept.
            if let mark = session.stateMarks.union(session.diskMarks).union(session.turnMark.map { [$0] } ?? []).filter({ $0 > session.fed }).min(),
               mark < session.fed + count { count = mark - session.fed }
            if !reserve(session, through: session.fed + count - 1) {
                // Whatever still fits in the pages the slot has.
                count = min(count, engine.pagesHeld(by: session.slot) * Engine.pageTokens - session.fed)
                guard count > 0 else { continue }
                waitingForMemory.removeAll { $0 === session }
            }
            let complete = session.fed + count == prompt.count
            if complete && logitRows >= engine.config.maxLogitRows { continue }
            for offset in 0..<count {
                let position = session.fed + offset
                rows.append(EngineRow(slot: session.slot, token: prompt[position], position: position,
                                      wantLogits: complete && offset == count - 1))
            }
            if complete { logitRows += 1 }
            plan.append(.prefill(session, count: count, complete: complete))
        }
        return (rows, plan)
    }

    private func consume(_ plan: [Planned], step: EngineStepStats) throws {
        var logitIndex = 0, rowIndex = 0
        var prefillRows = 0, decodeRows = 0
        var shares: [StepShare] = []
        var context: [(slot: Int, position: Int, engineRow: Int)] = []
        for entry in plan {
            switch entry {
            case .decode(let session):
                context.append((session.slot, slotTokens[session.slot].count, rowIndex))
                slotTokens[session.slot].append(session.pending!)
                session.speculation.lookup.append(session.pending!)
                session.pending = nil
                session.verifySteps += 1
                decodeRows += 1
                shares.append((session.id, 0, 1, 0, 0))
                handleSample(session, session.sampler.sample(engine.logits(logitIndex)))
                logitIndex += 1
                rowIndex += 1
            case .verify(let session, let proposal):
                let drafts = proposal.tokens
                let (agreed, bonus) = accept(session, proposal: proposal, firstLogit: logitIndex)
                session.speculation.record(agreed: agreed, drafts: drafts.count, copyMatch: proposal.copyMatch)
                session.drafted += drafts.count; session.accepted += agreed
                totals.draftedTokens += drafts.count; totals.acceptedTokens += agreed
                session.blocks.record(copied: proposal.copyMatch != nil, produced: agreed + 1)
                totals.blocks.record(copied: proposal.copyMatch != nil, produced: agreed + 1)
                // Emit accepted drafts until a stop token or the token budget ends the session.
                let anchor = session.pending!
                session.pending = nil
                session.verifySteps += 1
                var usable = 0
                var ended: FinishReason?
                // The scheduler's own token, where it is ending the thinking: the drafts from
                // there on were for a reply that is not being written.
                var written: Int?
                while usable < agreed {
                    let token = next(session, drafts[usable])
                    if token != drafts[usable] { written = token; break }
                    if session.request.stopTokenIDs.contains(drafts[usable]) { ended = .stop; break }
                    session.generated += 1
                    session.continuation.yield(.token(drafts[usable]))
                    usable += 1
                    if atLimit(session, token) { ended = .length; break }
                }
                // The anchor and the usable drafts are now evaluated; everything after is dropped.
                let start = slotTokens[session.slot].count
                if dictionaryStudy != nil {
                    let own = session.speculation.lookup.continuation(after: anchor, count: 1)?.match ?? 0
                    session.studyNotes += dictionaryStudy!.look(tail: Array(slotTokens[session.slot].suffix(DictionaryStudy.tailLength - 1)) + [anchor],
                                                                conversation: session.conversation, position: start, own: own, stepped: usable + 1)
                }
                slotTokens[session.slot].append(anchor)
                slotTokens[session.slot].append(contentsOf: drafts[0..<usable])
                session.speculation.lookup.append(anchor)
                for token in drafts[0..<usable] { session.speculation.lookup.append(token) }
                engine.acceptVerify(session.slot, rows: usable + 1, tokenCount: slotTokens[session.slot].count)
                for offset in 0...usable { context.append((session.slot, start + offset, rowIndex + offset)) }
                decodeRows += usable + 1
                shares.append((session.id, 0, usable + 1, drafts.count, agreed))
                if let ended { finish(session, ended, message: nil) }
                else if let written { emit(session, written) }
                else { handleSample(session, bonus) }
                logitIndex += drafts.count + 1
                rowIndex += drafts.count + 1
            case .prefill(let session, let count, let complete):
                let prompt = session.request.promptTokens
                // Only the positions that can still be inside the draft's window when decoding
                // starts need their hidden states pushed.
                let firstUseful = prompt.count - DraftModel.ringCapacity + DraftModel.maxBlock
                for offset in 0..<count where session.fed + offset >= firstUseful {
                    context.append((session.slot, session.fed + offset, rowIndex + offset))
                }
                slotTokens[session.slot].append(contentsOf: prompt[session.fed..<session.fed + count])
                session.fed += count
                prefillRows += count
                shares.append((session.id, count, 0, 0, 0))
                rowIndex += count
                // One copy of the state however many marks fall here (Data is shared, not copied).
                var exported: Data?
                func state() throws -> Data {
                    if let exported { return exported }
                    let start = Self.now()
                    let state = try engine.exportState(session.slot)
                    session.overhead.stateExports += Self.seconds(since: start)
                    session.overhead.stateExportCount += 1
                    exported = state
                    return state
                }
                let marked = session.stateMarks.contains(session.fed)
                if marked || session.diskMarks.contains(session.fed) {
                    let tokens = Array(prompt[0..<session.fed])
                    let wanted = prefixStore?.wantsState(tokens) ?? false
                    if marked || wanted {
                        let state = try state()
                        if marked { boundaries[session.slot] = Checkpoint(tokens: tokens, state: state) }
                        if wanted { prefixStore?.saveState(tokens: tokens, state: state) }
                    }
                }
                if session.fed == session.turnMark {
                    checkpoints[session.slot] = Checkpoint(tokens: Array(prompt[0..<session.fed]), state: try state())
                    session.checkpointKept = true
                }
                if complete {
                    // The slot's index, as its last session left it, extended by this prompt.
                    let lookupStart = Self.now()
                    var lookup = slotLookups[session.slot]
                    slotLookups[session.slot] = LookupIndex()
                    lookup.extend(to: slotTokens[session.slot])
                    session.speculation.lookup = lookup
                    session.overhead.lookupReset = Self.seconds(since: lookupStart)
                    // What this prompt added to what was already evaluated; a reused prefix
                    // went in when it was first evaluated, if that was since the server started.
                    dictionaryStudy?.add(prompt[min(session.cached, prompt.count)...], conversation: session.conversation)
                    session.prefillDone = Date()
                    if !session.checkpointKept {
                        checkpoints[session.slot] = Checkpoint(tokens: prompt, state: try state())
                    }
                    handleSample(session, session.sampler.sample(engine.logits(logitIndex)))
                    logitIndex += 1
                }
            }
        }
        // The draft conditions on the target's hidden states for every evaluated position.
        try draft?.pushContext(context)
        totals.steps += 1
        let now = Date().timeIntervalSinceReferenceDate
        window.append((cycleStarted, now, prefillRows, decodeRows, shares))
        totals.evaluatedTokens += prefillRows
        totals.generatedTokens += decodeRows
        lastStep = (step.rows, step.wallSeconds)

    }

    /// How many drafted tokens the target accepts, and the token that follows them.
    /// Greedy: the agreeing prefix, then the target's own arg-max. Sampling: speculative
    /// rejection sampling, which preserves the target's distribution exactly.
    private func accept(_ session: Session, proposal: DraftProposal, firstLogit: Int) -> (agreed: Int, bonus: Int) {
        let drafts = proposal.tokens
        guard session.request.sampling.temperature > 0, let draftProbabilities = proposal.probabilities else {
            var agreed = 0
            while agreed < drafts.count, session.sampler.argmax(engine.logits(firstLogit + agreed)) == drafts[agreed] { agreed += 1 }
            return (agreed, session.sampler.argmax(engine.logits(firstLogit + agreed)))
        }
        for index in drafts.indices {
            let target = session.sampler.distribution(engine.logits(firstLogit + index))
            if let replacement = SpeculativeSampling.resolve(draft: drafts[index], candidates: proposal.candidates[index],
                                                             draftProbabilities: draftProbabilities[index], targetIDs: target.ids,
                                                             targetProbabilities: target.probabilities, sampler: &session.sampler) {
                return (index, replacement)
            }
        }
        let last = session.sampler.distribution(engine.logits(firstLogit + drafts.count))
        return (drafts.count, last.ids[session.sampler.draw(last.probabilities)])
    }

    /// The reply's next token: the one the model chose, unless its thinking is being ended.
    private func next(_ session: Session, _ chosen: Int) -> Int {
        if session.generated == 0 {
            // The reply's first token: its limits are made here, from the settings as they stand.
            let request = session.request
            session.thinkingWatch = ThinkingWatch(request.thinkingClose, maxTokens: request.maxTokens, reserve: answerReserve)
            session.limit = ReplyLimit(maxTokens: request.maxTokens, marks: request.toolCall, overrun: toolCallOverrun,
                                       resets: request.thinkingClose?.token)
        }
        return session.thinkingWatch?.next(chosen, generated: session.generated) ?? chosen
    }

    /// Whether the reply ends for length with `token`, which has just been written.
    private func atLimit(_ session: Session, _ token: Int) -> Bool {
        session.limit?.reached(by: token, generated: session.generated) ?? (session.generated >= session.request.maxTokens)
    }

    private func handleSample(_ session: Session, _ chosen: Int) {
        emit(session, next(session, chosen))
    }

    private func emit(_ session: Session, _ token: Int) {
        if session.request.stopTokenIDs.contains(token) {
            finish(session, .stop, message: nil); return
        }
        session.generated += 1
        session.continuation.yield(.token(token))
        if atLimit(session, token) || slotTokens[session.slot].count + 1 >= engine.config.maxContext {
            finish(session, .length, message: nil); return
        }
        session.pending = token
    }

    private func finish(_ session: Session, _ reason: FinishReason, message: String?) {
        let now = Date()
        var usage = GenerationUsage()
        usage.promptTokens = session.request.promptTokens.count
        usage.cachedTokens = session.cached
        usage.completionTokens = session.generated
        usage.queueSeconds = (session.admitted ?? now).timeIntervalSince(session.submitted)
        if let admitted = session.admitted {
            usage.prefillSeconds = (session.prefillDone ?? now).timeIntervalSince(admitted)
        }
        if let done = session.prefillDone { usage.decodeSeconds = now.timeIntervalSince(done) }
        usage.overhead = session.overhead
        session.continuation.yield(.finished(reason, usage, message: message))
        session.continuation.finish()
        condition.lock(); inFlight -= 1; condition.unlock()
        if session.slot >= 0 {
            slotBusy[session.slot] = false
            slotLastUsed[session.slot] = now
            if session.prefillDone != nil {
                slotLookups[session.slot] = session.speculation.lookup
                session.speculation.lookup = LookupIndex()
            }
            // The copy for the disk store runs on this thread and every other session's steps
            // wait for it: at long context it is gigabytes. With nobody waiting it is taken now,
            // as insurance against a crash; otherwise the conversation goes to disk when its slot
            // is given up (`retire`) or the server stops.
            condition.lock(); let arriving = !inbox.isEmpty; condition.unlock()
            if !arriving && !active.contains(where: { $0 !== session }) && !waiting.contains(where: { $0 !== session }) {
                let copiedBefore = storeCopySeconds
                persistPrefix(slot: session.slot)
                session.overhead.storeCopy += storeCopySeconds - copiedBefore
            }
        }
        let evaluated = max(0, (session.prefillDone == nil ? max(session.fed, session.cached) : usage.promptTokens) - usage.cachedTokens)
        if var entry = conversations[session.conversation] {
            entry.lastActive = now
            entry.stats.turns += 1
            entry.stats.promptTokens = max(entry.stats.promptTokens, usage.promptTokens)
            entry.stats.reusedTokens += usage.cachedTokens
            entry.stats.evaluatedTokens += evaluated
            entry.stats.completionTokens += usage.completionTokens
            entry.stats.verifySteps += session.verifySteps
            entry.stats.queueSeconds += usage.queueSeconds
            entry.stats.prefillSeconds += usage.prefillSeconds
            entry.stats.decodeSeconds += usage.decodeSeconds
            entry.stats.draftedTokens += session.drafted
            entry.stats.acceptedTokens += session.accepted
            entry.stats.blocks += session.blocks
            entry.stats.overhead += session.overhead
            entry.stats.lastFinish = reason.rawValue
            if let activity = session.request.activity?.snapshot {
                entry.stats.thinkingTokens += activity.thinking
                entry.stats.answerTokens += activity.answer
                entry.stats.toolCallTokens += activity.toolCall
            }
            conversations[session.conversation] = entry
        }
        totals.finishedEvaluatedTokens += evaluated
        totals.finishedPrefillSeconds += usage.prefillSeconds
        totals.finishedDecodeSeconds += usage.decodeSeconds
        totals.finishedOverhead += session.overhead
        if requestLog {
            // One line per request: what was reused, what had to be evaluated, how fast.
            var line = String(format: "request %d (conversation %d, turn %d): prompt %d tokens (%d reused%@), %d evaluated in %.1f s",
                              session.id, session.conversation, session.turn, usage.promptTokens, usage.cachedTokens, session.reusedFrom.map { ", from " + $0 } ?? "", evaluated, usage.prefillSeconds)
            if usage.prefillSeconds > 0.5 { line += String(format: " = %.0f tok/s", Double(evaluated) / usage.prefillSeconds) }
            if usage.completionTokens > 1, usage.decodeSeconds > 0 {
                line += String(format: "; %d tokens in %.1f s = %.0f tok/s, %.1f a step", usage.completionTokens, usage.decodeSeconds,
                               Double(usage.completionTokens) / usage.decodeSeconds, Double(session.generated) / Double(max(session.verifySteps, 1)))
                let b = session.blocks
                if b.copySteps > 0 {
                    line += String(format: " (%d copied steps gave %d tokens, %d drafted gave %d)", b.copySteps, b.copyTokens, b.draftSteps, b.draftTokens)
                }
            }
            line += "; " + session.overhead.summary
            line += "; \(reason)" + (message.map { " (\($0))" } ?? "") + "\n"
            writeStandardError(Data(line.utf8))
        }
        if dictionaryStudy != nil, session.slot >= 0 {
            let text = slotTokens[session.slot]
            let counted = dictionaryStudy!.score(session.studyNotes, text: text)
            dictionaryStudy!.add(text.dropFirst(session.request.promptTokens.count), conversation: session.conversation)
            if counted { FileHandle.standardError.write(Data(dictionaryStudy!.report.map { $0 + "\n" }.joined().utf8)) }
        }
        active.removeAll { $0 === session }
        waiting.removeAll { $0 === session }
        totals.completed += 1
        totals.promptTokens += usage.promptTokens
        totals.cachedTokens += usage.cachedTokens
        totals.completionTokens += usage.completionTokens
    }

    // MARK: - Stats

    private static func memoryStats(_ m: EngineMemory, checkpointBytes: Int) -> MemoryStats {
        MemoryStats(weightBytes: m.weightBytes, kvPoolBytes: m.kvPoolBytes, kvBytesUsed: m.kvBytesUsed,
                    kvPagesTotal: m.kvPagesTotal, kvPagesUsed: m.kvPagesUsed, kvBytesPerToken: m.kvBytesPerToken,
                    stateBytesPerSlot: m.stateBytesPerSlot, stateBytesTotal: m.stateBytesTotal,
                    checkpointBytes: checkpointBytes, scratchBytes: m.scratchBytes,
                    deviceAllocatedBytes: m.deviceAllocatedBytes, deviceWorkingSetBytes: m.deviceWorkingSetBytes)
    }

    private func publish() {
        let now = Date()
        let memory = engine.memory
        let reference = now.timeIntervalSinceReferenceDate
        window.removeAll { reference - $0.end > Self.draftSpan }
        let rated = window.drop { reference - $0.end > Self.rateSpan }
        // From the start of the oldest step counted: a step's tokens over the time it took,
        // not over the gap between two steps' ends, which counts one step too many.
        let elapsed = max(reference - (rated.first?.start ?? reference), 1e-3)
        var throughput = ThroughputStats(
            prefillTokensPerSecond: rated.isEmpty ? 0 : Double(rated.reduce(0) { $0 + $1.prefill }) / elapsed,
            decodeTokensPerSecond: rated.isEmpty ? 0 : Double(rated.reduce(0) { $0 + $1.decode }) / elapsed,
            stepsPerSecond: rated.isEmpty ? 0 : Double(rated.count) / elapsed,
            lastStepRows: lastStep.rows, lastStepMilliseconds: lastStep.seconds * 1000)
        for entry in window {
            for share in entry.shares { throughput.recentDraftedTokens += share.drafted; throughput.recentAcceptedTokens += share.accepted }
        }

        func describe(_ session: Session, state: String) -> SessionStats {
            let context = session.slot >= 0 ? slotTokens[session.slot].count : 0
            var prefillRate = 0.0, decodeRate = 0.0
            if let admitted = session.admitted {
                let seconds = (session.prefillDone ?? now).timeIntervalSince(admitted)
                if seconds > 0 { prefillRate = Double(session.fed - session.cached) / seconds }
            }
            if let done = session.prefillDone {
                let seconds = now.timeIntervalSince(done)
                if seconds > 0 { decodeRate = Double(session.generated) / seconds }
            }
            var stats = SessionStats(id: session.id, label: session.request.label, state: state,
                                slot: session.slot >= 0 ? session.slot : nil,
                                promptTokens: session.request.promptTokens.count, cachedTokens: session.cached,
                                evaluatedTokens: session.fed, generatedTokens: session.generated, contextTokens: context,
                                kvBytes: session.slot >= 0 ? engine.pagesHeld(by: session.slot) * memory.kvBytesPerPage : 0,
                                stateBytes: session.slot >= 0 ? memory.stateBytesPerSlot : 0,
                                prefillTokensPerSecond: prefillRate, decodeTokensPerSecond: decodeRate,
                                tokensPerStep: session.verifySteps > 0 ? Double(session.generated) / Double(session.verifySteps) : 1,
                                ageSeconds: now.timeIntervalSince(session.submitted))
            stats.conversation = session.conversation
            stats.turn = session.turn
            stats.draftedTokens = session.drafted
            stats.acceptedTokens = session.accepted
            stats.blocks = session.blocks
            if let activity = session.request.activity?.snapshot {
                stats.producing = activity.current
                stats.thinkingTokens = activity.thinking; stats.answerTokens = activity.answer; stats.toolCallTokens = activity.toolCall
            }
            // This session's part of the same steps, since the first of them it was in: alone,
            // it reads what the server reads; together, the sessions add up to it.
            var recent = (prefill: 0, decode: 0), since: Double?
            for entry in window {
                for share in entry.shares where share.id == session.id {
                    stats.recentDraftedTokens += share.drafted; stats.recentAcceptedTokens += share.accepted
                    guard reference - entry.end <= Self.rateSpan else { continue }
                    recent.prefill += share.prefill; recent.decode += share.decode
                    since = since ?? entry.start
                }
            }
            if let since {
                let seconds = max(reference - since, 1e-3)
                stats.recentPrefillTokensPerSecond = Double(recent.prefill) / seconds
                stats.recentDecodeTokensPerSecond = Double(recent.decode) / seconds
            }
            return stats
        }
        let sessions = active.map { describe($0, state: $0.prefillDone == nil ? "prefill" : "decode") }
            + waiting.map { describe($0, state: "queued") }
        var cached: [CachedPrefixStats] = []
        var checkpointBytes = 0
        for slot in slotBusy.indices {
            checkpointBytes += (checkpoints[slot]?.state.count ?? 0) + (boundaries[slot]?.state.count ?? 0)
            guard !slotBusy[slot], !slotTokens[slot].isEmpty else { continue }
            cached.append(CachedPrefixStats(slot: slot, tokens: slotTokens[slot].count,
                                            checkpointTokens: checkpoints[slot]?.tokens.count ?? 0,
                                            kvBytes: engine.pagesHeld(by: slot) * memory.kvBytesPerPage,
                                            stateBytes: memory.stateBytesPerSlot,
                                            idleSeconds: now.timeIntervalSince(slotLastUsed[slot])))
        }
        var snapshot = SchedulerStats(sessions: sessions, queued: waiting.count, cached: cached,
                                      memory: Self.memoryStats(memory, checkpointBytes: checkpointBytes),
                                      throughput: throughput, totals: totals,
                                      maxSlots: engine.config.maxSlots, maxRows: engine.config.maxRows,
                                      maxContext: engine.config.maxContext, speculative: draft != nil,
                                      uptimeSeconds: now.timeIntervalSince(started))
        snapshot.prefixStore = prefixStore?.stats
        let live = Set((active + waiting).map(\.conversation))
        snapshot.conversations = conversations.values.sorted { $0.lastActive > $1.lastActive }.map { entry in
            var stats = entry.stats
            stats.endedSecondsAgo = now.timeIntervalSince(entry.lastActive)
            stats.running = live.contains(stats.id)
            return stats
        }
        snapshot.stepTiers = tiers.sorted { ($0.key.prompt ? 1 : 0, $0.key.rows, $0.key.context) < ($1.key.prompt ? 1 : 0, $1.key.rows, $1.key.context) }
            .map { key, sums in
                StepTierStats(rows: key.rows, context: Self.contextRanges[key.context], prompt: key.prompt, steps: sums.steps,
                              milliseconds: sums.seconds / Double(sums.steps) * 1000, gpuMilliseconds: sums.gpu / Double(sums.steps) * 1000,
                              cycleMilliseconds: sums.cycle / Double(sums.steps) * 1000, fastestMilliseconds: sums.fastest * 1000)
            }
        var activities: [Int: SessionActivity] = [:]
        for session in active + waiting { activities[session.id] = session.request.activity }
        condition.lock(); published = snapshot; publishedActivities.update(activities); condition.unlock()
    }
}

/// To standard error without raising when it has gone (a closed terminal).
func writeStandardError(_ data: Data) {
    data.withUnsafeBytes { _ = write(2, $0.baseAddress, $0.count) }
}
func writeStandardError(_ text: String) { writeStandardError(Data(text.utf8)) }
