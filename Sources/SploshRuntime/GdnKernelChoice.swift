import Foundation

/// The kernels of the gated-delta layer's split path as chosen by name, and the checks that a
/// choice is one that computes the layer.
///
/// The names come from SPLOSH_GDN_PREPARE, SPLOSH_GDN_CHAINS, SPLOSH_GDN_FINISH and
/// SPLOSH_GDN_HISTORY (unset: the defaults below), and SPLOSH_GDN_CHAIN_THREADS gives the
/// threads of a recurrence threadgroup (unset: 1024 / the threadgroups a head).
///
/// The candidates of Sources/Shaders/candidates/gdn_b.metal are named `_b`, then `n` and `s`
/// as they apply, then a number:
///   n   the prepare stage hands the recurrence q and k scaled by their norms and (decay, beta)
///       where the norms were; the recurrence expects exactly that
///   s   q and k (and their norms) are stored once per key head, at its first value head
/// A prepare kernel and a recurrence kernel that disagree on `n`, or an `s` prepare kernel with
/// a recurrence that reads every value head's own q and k, give finite, plausible and wrong
/// numbers, and the wrong state is then kept. Such a choice is refused here.
public struct GdnKernelChoice: Sendable, CustomStringConvertible {
    public var prepare = "sp_gdn_prepare_b1", chains = "sp_gdn_chains_b2", finish = "sp_gdn_finish", history = "sp_gdn_history_b1"
    /// Threadgroups a head for the recurrence.
    public var chainGroups = 8
    /// Threads of a recurrence threadgroup.
    public var chainThreads = 128

    public var description: String { "\(prepare) / \(chains) (\(chainGroups) x \(chainThreads) threads) / \(finish) / \(history)" }

    public init() {}

    public init(environment: [String: String]) throws {
        prepare = environment["SPLOSH_GDN_PREPARE"] ?? prepare
        chains = environment["SPLOSH_GDN_CHAINS"] ?? chains
        finish = environment["SPLOSH_GDN_FINISH"] ?? finish
        history = environment["SPLOSH_GDN_HISTORY"] ?? history
        chainGroups = Int(environment["SPLOSH_GDN_CHAIN_GROUPS"] ?? "") ?? 8
        chainThreads = chainGroups > 0 ? 1024 / chainGroups : 0
        if let text = environment["SPLOSH_GDN_CHAIN_THREADS"] {
            guard let threads = Int(text) else {
                throw EngineError.invalidKernelChoice("SPLOSH_GDN_CHAIN_THREADS=\(text) is not a number")
            }
            chainThreads = threads
        }
        try validate()
    }

    /// What a candidate's name says about the operands it writes or reads.
    public struct Form: Equatable, Sendable {
        public var candidate = false, normalised = false, shared = false, number = 0
    }

    public static func form(_ name: String, stage: String) -> Form {
        guard name.hasPrefix(stage + "_b") else { return Form() }
        var rest = Substring(name.dropFirst(stage.count + 2))
        var form = Form(candidate: true)
        if rest.hasPrefix("n") { form.normalised = true; rest = rest.dropFirst() }
        if rest.hasPrefix("s") { form.shared = true; rest = rest.dropFirst() }
        // Anything else is not one of this family's names: nothing is known about it.
        guard let number = Int(rest), number > 0 else { return Form() }
        form.number = number
        return form
    }

    public var prepareForm: Form { Self.form(prepare, stage: "sp_gdn_prepare") }
    public var chainForm: Form { Self.form(chains, stage: "sp_gdn_chains") }

    /// The simdgroups of a threadgroup the recurrence kernel uses. engine.metal's kernel and the
    /// candidates with sixteen state elements a lane use four; the candidates with thirty-two
    /// (numbered 2) use the first two and return at once in any others.
    public var chainSimdgroupsUsed: Int { chainForm.candidate && chainForm.number == 2 ? 2 : 4 }

    public func validate() throws {
        let prepared = prepareForm, chain = chainForm
        guard prepared.normalised == chain.normalised else {
            throw EngineError.invalidKernelChoice(
                "\(prepare) and \(chains) do not agree on normalised operands; the `_bn` kernels are a pair and must be chosen together")
        }
        guard !prepared.shared || chain.shared else {
            throw EngineError.invalidKernelChoice(
                "\(prepare) stores q and k once per key head, which \(chains) does not read; choose an `s` recurrence kernel with it")
        }
        guard chainGroups > 0, chainThreads >= 32, chainThreads % 32 == 0, chainThreads <= 1024 else {
            throw EngineError.invalidKernelChoice(
                "the recurrence's threadgroups are whole simdgroups, 32 to 1024 threads; \(chainGroups) threadgroups a head of \(chainThreads) threads is not")
        }
        // What is known of the kernel's layout: engine.metal's kernel and the candidates spread a
        // head over 8 threadgroups. A kernel under another name is taken at its word.
        if chain.candidate || chains == "sp_gdn_chains" {
            guard chainGroups == 8 else {
                throw EngineError.invalidKernelChoice("\(chains) needs SPLOSH_GDN_CHAIN_GROUPS=8, not \(chainGroups)")
            }
            guard chainThreads >= chainSimdgroupsUsed * 32 else {
                throw EngineError.invalidKernelChoice(
                    "\(chains) needs \(chainSimdgroupsUsed * 32) threads a threadgroup or more, not \(chainThreads): columns of the state would not be computed")
            }
        }
    }

    /// Whether the choice reads the conv history of q and k from a key head's first value head
    /// only (the candidates' prepare kernels), or q and k themselves from there (an `s`
    /// recurrence). The history is kept per value head; the kernels that write it (sp_gdn_fast,
    /// sp_gdn_history and its candidate) write the same values for the value heads of a key head,
    /// so the copies only differ in a state that came from somewhere else.
    public var readsKeyHeadHistory: Bool { prepareForm.candidate || chainForm.shared }

    /// Whether one unit of conv history (valueHeads x 3 kinds x headDim channels x 3 inputs) has
    /// the same q and k history for every value head of a key head.
    public static func keyHistoryIsShared(_ conv: UnsafePointer<Float>, valueHeads: Int, keyHeads: Int, headDim: Int) -> Bool {
        let perKey = valueHeads / keyHeads, perHead = 3 * headDim * 3, shared = 2 * headDim * 3
        for h in 0..<valueHeads where h % perKey != 0 {
            let first = (h / perKey) * perKey
            if memcmp(conv + h * perHead, conv + first * perHead, shared * MemoryLayout<Float>.stride) != 0 { return false }
        }
        return true
    }
}
