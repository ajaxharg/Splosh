// Engine.swift — the whole-model execution path.
//
// A step evaluates a batch of rows through all 64 blocks in one command buffer. A row is one
// token position of one session ("slot"). Rows from many slots in one step is batched decode;
// consecutive rows of one slot is prefill; a step may contain both.
//
// Per-session state lives in two places:
//   * full-attention layers: K/V in a paged fp16 pool shared by all slots (256 tokens per page),
//     addressed through a per-slot page table, so long and short sessions share one pool;
//   * gated-delta layers: a fixed-size recurrent state and a 3-sample conv history per slot.
//
// The engine is not thread-safe; the scheduler owns it and serialises steps.

import Foundation
@preconcurrency import Metal
import SploshCore
import SploshModel

public struct EngineConfig: Sendable, Equatable {
    /// Concurrent sessions holding state (active or cached).
    public var maxSlots: Int
    /// Rows evaluated per step.
    public var maxRows: Int
    /// KV pool size in pages of 256 tokens, shared by all slots.
    public var kvPages: Int
    /// Longest single session, in tokens.
    public var maxContext: Int

    /// Gated-delta state units in the pool. Each slot holds one; a speculative verify of n rows
    /// briefly holds n more. Units are committed lazily, so spare capacity costs nothing.
    public var stateUnits: Int
    /// Rows per step that may request logits.
    public var maxLogitRows: Int
    /// KV cache precision. `int8` stores symmetric 8-bit codes with one scale per vector
    /// (32.5 KiB per token) and is the format the fused accelerator attention runs on; `q4`
    /// stores 4-bit codes with per-64 fp16 scale and bias (18 KiB per token); `fp16` stores
    /// half floats (64 KiB per token).
    public var kvFormat: KVFormat = .int8
    public enum KVFormat: String, Sendable { case int8, q4, fp16 }

    public init(maxSlots: Int = 8, maxRows: Int = 64, kvPages: Int = 256, maxContext: Int = 32768,
                stateUnits: Int? = nil, maxLogitRows: Int? = nil) {
        self.maxSlots = maxSlots; self.maxRows = maxRows
        self.kvPages = kvPages; self.maxContext = maxContext
        self.stateUnits = stateUnits ?? maxSlots * (Engine.journalRows + 1)
        self.maxLogitRows = maxLogitRows ?? maxRows
    }
}

public struct EngineRow: Sendable, Equatable {
    public let slot: Int
    public let token: Int
    public let position: Int
    public let wantLogits: Bool
    /// Speculative row: its recurrent state is written to a fresh unit so that any prefix of the
    /// run can be accepted afterwards with `acceptVerify`.
    public let verify: Bool
    public init(slot: Int, token: Int, position: Int, wantLogits: Bool, verify: Bool = false) {
        self.slot = slot; self.token = token; self.position = position
        self.wantLogits = wantLogits; self.verify = verify
    }
}

public struct EngineStepStats: Sendable {
    public let rows: Int
    public let dispatches: Int
    public let gpuSeconds: Double
    public let wallSeconds: Double
}

public struct EngineMemory: Sendable, Equatable {
    public let weightBytes: Int
    public let kvPoolBytes: Int
    public let kvBytesPerPage: Int
    public let kvPagesTotal: Int
    public let kvPagesUsed: Int
    public let stateBytesPerSlot: Int
    public let stateBytesTotal: Int
    public let stateUnitsUsed: Int
    public let scratchBytes: Int
    public let deviceWorkingSetBytes: Int
    public let deviceAllocatedBytes: Int

    public var kvBytesUsed: Int { kvPagesUsed * kvBytesPerPage }
    public var kvBytesPerToken: Int { kvBytesPerPage / Engine.pageTokens }
}

public enum EngineError: Error, CustomStringConvertible {
    case kvPoolExhausted(needed: Int, free: Int)
    case contextExceeded(position: Int, limit: Int)
    case unsupportedDevice(String)
    case allocationFailed(String, bytes: Int)
    case invalidRows(String)
    case snapshotMismatch(String)
    case invalidKernelChoice(String)

    public var description: String {
        switch self {
        case .kvPoolExhausted(let needed, let free): return "KV pool exhausted: need \(needed) page(s), \(free) free"
        case .contextExceeded(let position, let limit): return "position \(position) exceeds the context limit \(limit)"
        case .unsupportedDevice(let reason): return "unsupported device: \(reason)"
        case .allocationFailed(let what, let bytes): return "unable to allocate \(what) (\(bytes) bytes)"
        case .invalidRows(let reason): return "invalid step rows: \(reason)"
        case .snapshotMismatch(let reason): return "snapshot does not match this engine: \(reason)"
        case .invalidKernelChoice(let reason): return "gated-delta kernels: \(reason)"
        }
    }
}

/// Everything needed to resume a session at a token boundary.
public struct SlotSnapshot: Sendable {
    public let tokenCount: Int
    /// Recurrent + conv state for every gated-delta layer, concatenated in layer order.
    public let state: Data
    /// K then V for every full-attention layer, whole pages, concatenated in layer order.
    public let kv: Data
    public init(tokenCount: Int, state: Data, kv: Data) {
        self.tokenCount = tokenCount; self.state = state; self.kv = kv
    }
    public var byteCount: Int { state.count + kv.count }
}

public final class Engine: @unchecked Sendable {
    public static let pageTokens = 256
    private static let lanes = 32
    private static let rowBlock = 8

    public let device: MTLDevice
    public let weights: ModelWeights
    public let config: EngineConfig
    public let maxPagesPerSlot: Int
    let g: ModelGeometry
    let graph: CommandGraph

    // Pipelines
    private let pEmbed, pGemm, pNorm, pSiluMul, pSiluMulNa, pGather: MTLComputePipelineState
    private let pMlpIn, pGdn, pAttentionFused, pProbe: MTLComputePipelineState
    /// Fixed-width GEMM / MLP-input kernels, keyed by rows per pass (1, 2, 4, 8).
    private let gemmByWidth: [Int: MTLComputePipelineState]
    private let mlpByWidth: [Int: MTLComputePipelineState]
    private let fixedWidth: Bool
    private let skip: Set<String>
    private let pGemmMM: MTLComputePipelineState
    /// Neural-accelerator GEMM: (pipeline, rows per tile, columns per tile, threads per group).
    private let na: (pipeline: MTLComputePipelineState, m: Int, n: Int, threads: Int)?
    private let pNaPrepare: MTLComputePipelineState
    /// Split-K kernel for passes of at most eight rows: (pipeline, partitions).
    private let split: (pipeline: MTLComputePipelineState, partitions: Int)?
    private let splitMinRows: Int
    private var splitTiled: MTLComputePipelineState?
    private var splitColumns = 32
    /// Lane kernels reading the tiled layout, keyed by rows per pass (1, 2).
    private var tiledLane: [Int: (gemm: MTLComputePipelineState, mlp: MTLComputePipelineState)] = [:]
    private var tiledLanePartitions = 8
    /// Split-K kernels for taller tiles: (rows per tile, partitions, pipeline), shortest first.
    private var splitWide: [(m: Int, partitions: Int, columns: Int, pipeline: MTLComputePipelineState)] = []
    private let naInput, naSums: MTLBuffer
    /// Weights re-ordered into 128-row tiles for the accelerator, keyed by packed tensor name.
    private var tiled: [String: (packed: TensorHandle, scales: TensorHandle, biases: TensorHandle)] = [:]
    private var naTiled: MTLComputePipelineState?
    /// Weight bytes held a second time because the artifact was not tiled on disk.
    public private(set) var duplicatedWeightBytes = 0
    private var pNaPrepareTiled: MTLComputePipelineState?
    private var naRowMajorSums = false
    /// The buffer `naInput`/`naSums` currently mirror. Cleared at every stage boundary.
    private var naSource: MTLBuffer?
    /// KV pages visible to the longest row of the current step.
    private var stepPages = 1
    /// Operands already emitted by the kernel that produced a buffer, keyed by that buffer.
    private var ready: [ObjectIdentifier: (values: MTLBuffer, sums: MTLBuffer)] = [:]
    private var pNormNa: MTLComputePipelineState!
    /// Two-phase attention (q4 KV): span scan, then merge.
    private var pAttnScan, pAttnMerge: MTLComputePipelineState?
    /// Accelerator attention: its own pool layout (codes and metadata in separate buffers).
    private var pAttnScanNa, pAttnQPrepNa, pAttnKVStoreNa: MTLComputePipelineState?
    private var kMeta: [MTLBuffer] = [], vMeta: [MTLBuffer] = []
    private var attnBlocks, attnQueries, attnQuerySums: MTLBuffer!
    private var attnBlockCount = 0
    /// Rows per attention block (queries per block / heads per KV head).
    private var attnBlockRows = 4
    /// Dense three-pass attention: scratch for scores, value operands and per-query statistics.
    private var pAttnScores, pAttnSoftmax, pAttnValues, pAttnValuesMerge: MTLComputePipelineState?
    private var attnValuePartials: MTLBuffer!
    private static let maxValueSegments = 16
    private var attnScores, attnWeighted, attnStats, attnRowQ: MTLBuffer!
    private var attnQrowCap = 0
    private var attnScratchBytes = 0
    private let separateKVMeta: Bool
    private let kvMetaBytesPerVector: Int
    private let kvCodeBytesPerVector: Int
    private var attnScanThreads = 256
    private let hB, hSums, hSquares, rowInv, rowOnes: MTLBuffer
    private let pRowInv, pGemmBf16, pGemmEmitWide, pGemmEmitSplit: MTLComputePipelineState
    /// The per-row scale the consumers of the current GEMM outputs must apply: `rowInv` while a
    /// norm is deferred, otherwise ones.
    private var currentInv: MTLBuffer
    /// SPLOSH_DEFER_NORM=0 keeps a normalisation stage in front of every GEMM.
    private let deferNorms = ProcessInfo.processInfo.environment["SPLOSH_DEFER_NORM"] != "0"
    private let extraStages = Int(ProcessInfo.processInfo.environment["SPLOSH_EXTRA_STAGES"] ?? "") ?? 0
    private let overlapProbe = ProcessInfo.processInfo.environment["SPLOSH_OVERLAP_PROBE"]
    private lazy var pAluProbe = try? library.pipeline("sp_probe_alu_f32")
    private lazy var aluProbeOut = device.makeBuffer(length: (1 << 16) * 4, options: .storageModeShared)

    /// Scheduling probe (see sp_probe_spin): GPU seconds for one dispatch of the given grid.
    /// Sixteen rounds of [a spin dispatch of `spin` iterations (none if 0), then a matmul probe
    /// dispatch], each its own stage: what an accelerator dispatch costs after ordinary work.
    public func afterSpinProbe(grid: (Int, Int), spin: Int, matmul: Bool) throws -> Double {
        let spinPipeline = try library.pipeline("sp_probe_spin"), probe = try library.pipeline("sp_probe_matmul")
        guard let out = aluProbeOut else { return 0 }
        var sp = (UInt32(spin), UInt32(0)), pp = (UInt32(3), UInt32(31))
        try graph.begin(label: "splosh.afterspin", concurrent: true)
        for _ in 0..<16 {
            if spin > 0 {
                graph.dispatch(spinPipeline, grid: MTLSize(width: 64, height: 16, depth: 1), threadsPerGroup: MTLSize(width: 256, height: 1, depth: 1)) { e in
                    e.setBuffer(out, offset: 0, index: 0); e.setBytes(&sp, length: 8, index: 1)
                }
                stage()
            }
            if matmul {
                graph.dispatch(probe, grid: MTLSize(width: grid.0, height: grid.1, depth: 1), threadsPerGroup: MTLSize(width: 256, height: 1, depth: 1)) { e in
                    e.setBuffer(attnQueries, offset: 0, index: 0); e.setBuffer(kPools[0], offset: 0, index: 1)
                    e.setBuffer(out, offset: 0, index: 2); e.setBytes(&pp, length: 8, index: 3)
                }
                stage()
            }
        }
        return try graph.commitAndWait()
    }

    public func spinProbe(grid: (Int, Int), threads: Int, iterations: Int, barriers: Int, repeats: Int = 1, alternate: Bool = false) throws -> Double {
        let matmul = threads == 0
        let pipeline = try library.pipeline(matmul ? "sp_probe_matmul" : "sp_probe_spin")
        let other = try library.pipeline("sp_probe_matmul_b")
        guard let out = aluProbeOut else { return 0 }
        var p = (UInt32(iterations), UInt32(barriers))
        try graph.begin(label: "splosh.spin", concurrent: true)
        for index in 0..<repeats {
            if alternate, index % 2 == 1 {
                // The other kind of matmul, in threadgroups of four simdgroups.
                graph.dispatch(other, grid: MTLSize(width: grid.0, height: grid.1, depth: 1),
                               threadsPerGroup: MTLSize(width: 128, height: 1, depth: 1)) { e in
                    e.setBuffer(attnQueries, offset: 0, index: 0); e.setBuffer(kPools[0], offset: 0, index: 1)
                    e.setBuffer(out, offset: 0, index: 2); e.setBytes(&p, length: 8, index: 3)
                }
                stage()
                continue
            }
            graph.dispatch(pipeline, grid: MTLSize(width: grid.0, height: grid.1, depth: 1),
                           threadsPerGroup: MTLSize(width: matmul ? 256 : threads, height: 1, depth: 1)) { e in
                if matmul {
                    // Any operands will do: the queries scratch and the first key pool.
                    e.setBuffer(attnQueries, offset: 0, index: 0); e.setBuffer(kPools[0], offset: 0, index: 1)
                    e.setBuffer(out, offset: 0, index: 2); e.setBytes(&p, length: 8, index: 3)
                } else {
                    e.setBuffer(out, offset: 0, index: 0); e.setBytes(&p, length: 8, index: 1)
                }
            }
            stage()
        }
        return try graph.commitAndWait()
    }

    private func aluProbe() {
        guard let pAluProbe, let aluProbeOut else { return }
        var p = (UInt32(1 << 16), UInt32(Int(ProcessInfo.processInfo.environment["SPLOSH_OVERLAP_ITERATIONS"] ?? "") ?? 4000))
        graph.dispatchThreads(pAluProbe, threads: MTLSize(width: 1 << 16, height: 1, depth: 1)) { e in
            e.setBuffer(aluProbeOut, offset: 0, index: 0); e.setBytes(&p, length: 8, index: 1)
        }
    }
    private let emitProbe = ProcessInfo.processInfo.environment["SPLOSH_SPLIT_PROBE"]
    private let narrowSpanLimit = Int(ProcessInfo.processInfo.environment["SPLOSH_ATTN_SPANS"] ?? "") ?? 16
    public var wideSpanLimit = Int(ProcessInfo.processInfo.environment["SPLOSH_ATTN_WIDE_SPANS"] ?? "") ?? 0
    private let pGdnCommit: MTLComputePipelineState
    private let pGdnPrepare, pGdnScan, pGdnFinish, pGdnHistory: MTLComputePipelineState
    /// From this many rows a gated-delta layer runs as three dispatches (see sp_gdn_prepare);
    /// below it, as the one kernel, where two more stages would cost more than they save.
    /// 4-5 ms a 128-row step; at 32 and 64 rows it neither gains nor loses.
    public var gdnSplitRows = Int(ProcessInfo.processInfo.environment["SPLOSH_GDN_SPLIT"] ?? "") ?? 32
    /// Timing probe: parts of the split path left out (1 prepare, 2 chains, 4 finish, 8 history).
    public var gdnSkipMask = 0
    /// The split path's kernels by name and the recurrence's threadgroup shape (GdnKernelChoice).
    private let gdnChoice: GdnKernelChoice
    /// From this many rows the norm runs with a row spread over ten simdgroups.
    public var normWideRows = Int(ProcessInfo.processInfo.environment["SPLOSH_NORM_WIDE"] ?? "") ?? 32
    private lazy var pNormWide: MTLComputePipelineState = try! library.pipeline("sp_rmsnorm_wide")
    /// Probe: the scan's grid made this many times wider with threadgroups that return at once.
    public var attnGridPad = Int(ProcessInfo.processInfo.environment["SPLOSH_ATTN_PAD"] ?? "") ?? 1
    /// Timing probe: switch the int8 scan kernel between steps ("m48c32s8"; 48-query shapes only).
    public func setScanShape(_ shape: String) throws {
        pAttnScanNa = try library.pipeline("sp_attn_scan_i8_" + shape)
        let digits = shape.split(whereSeparator: { !$0.isNumber }).compactMap { Int($0) }
        attnScanThreads = max(digits.count > 2 ? digits[2] : 8, shape.contains("v") && digits.count > 3 ? digits[3] : 0) * 32
    }
    /// The scan's grid with the blocks along x (see spi_scan).
    public var attnBlocksAcross = ProcessInfo.processInfo.environment["SPLOSH_ATTN_ACROSS"] == "1"
    /// Timing probe for attention: 1 query prepare, 2 key/value store, 4 scan, 8 merge,
    /// 16 the q, k, v projections, 32 the output projection, 64 the norm stages, 128 the SiLU pass.
    public var attnSkipMask = Int(ProcessInfo.processInfo.environment["SPLOSH_ATTN_SKIPMASK"] ?? "") ?? 0
    /// Speculative rows per slot per step whose state updates the journal can hold.
    public static let journalRows = 16
    private static let journalWidth = 260
    private var gdnJournal: [MTLBuffer] = []
    /// Accepted journal entries per slot, not yet folded into the stored state.
    private var journalPending: [Int] = []
    private let runInfo: MTLBuffer
    private let attnOperandArrays = 4
    private let kvMetaPageBytes: Int
    private let scratchRowCap: Int
    private var acceleratedAttention: Bool { separateKVMeta }
    private var attnPartials: MTLBuffer!
    private static let maxSpans = Int(ProcessInfo.processInfo.environment["SPLOSH_ATTN_MAXSPANS"] ?? "") ?? 16
    /// Probe: every attention chunk read from the first this-many tokens of the pool, which
    /// takes the memory traffic out of the scan and leaves its arithmetic.
    private let attnAliasTokens = UInt32(ProcessInfo.processInfo.environment["SPLOSH_ATTN_ALIAS"] ?? "") ?? 0
    private var xnB, xnSums, coreB, coreSums, gateB, gateSums: MTLBuffer!
    /// Whether the mirrored sums use the tiled layout rather than row-major.
    private var naSourceTiled = false
    private let matrixRows: Int
    /// Largest pass the split-K kernels take; wider ones go to the plain tile kernel.
    private let splitMaxRows = Int(ProcessInfo.processInfo.environment["SPLOSH_SPLIT_MAX"] ?? "") ?? 64
    let library: Metallib
    /// Register-tiled kernels for multi-row passes: (rows per pass, weight rows per lane).
    private let tile: (rows: Int, columns: Int, gemm: MTLComputePipelineState, mlp: MTLComputePipelineState)?
    private let pQPrep, pKVStore, pAttention: MTLComputePipelineState
    private let pConv, pL2, pScan, pGatedNorm: MTLComputePipelineState

    // Step scratch
    private let tokens, rowSlot, rowPos, runs, gatherIndex, pageTable, rowRead, rowWrite, features: MTLBuffer
    private let pCopyRows: MTLComputePipelineState
    /// Target layers whose outputs the draft model conditions on.
    public static let featureLayers = [5, 19, 33, 47, 61]
    private let h, xn, proj, z, aBuf, bBuf, core, kproj, vproj, qn, gate, up, hs, hsn, logits: MTLBuffer
    private let gdnQ, gdnK, gdnV, gdnNorm: MTLBuffer
    private let scratchBytes: Int

    // Session state
    private var kPools: [MTLBuffer] = []
    private var vPools: [MTLBuffer] = []
    private var gdnState: [MTLBuffer] = []
    private var convState: [MTLBuffer] = []
    private var freePages: [UInt32]
    private var slotPages: [[UInt32]]
    /// Weight buffers by span index: the resident artifact's spans, then registered extras.
    private var spans: [MTLBuffer] = []
    /// The state unit holding each slot's current recurrent state.
    private var slotUnit: [Int] = []
    private var freeUnits: [Int] = []
    /// Units written by a slot's most recent verify run, in row order.
    private var pendingVerify: [[Int]] = []

    private let kvLayerPageBytes: Int
    private let stateLayerBytes: Int
    private let convLayerBytes: Int

    public init(device: MTLDevice, weights: ModelWeights, config: EngineConfig = EngineConfig()) throws {
        self.device = device
        self.weights = weights
        self.config = config
        self.g = weights.geometry
        self.maxPagesPerSlot = (config.maxContext + Self.pageTokens - 1) / Self.pageTokens

        // The 8-bit kernels exist (engine_q8.metal, and the q8 forms in engine_na.metal) and
        // the artifact converts, but `gemm`, `mlpIn` and `embed` still choose q4 kernels only.
        // A GGUF-derived artifact reports no code width: each of its handles names its format's
        // kernels (engine_gguf.metal), and those three dispatch them.
        guard weights.ggufDerived || weights.codeBits == 4 else {
            throw EngineError.unsupportedDevice("\(weights.codeBits)-bit weights: the engine does not dispatch their kernels yet")
        }

        guard let queue = device.makeCommandQueue() else {
            throw EngineError.unsupportedDevice("no command queue")
        }
        queue.label = "splosh.engine"
        self.graph = CommandGraph(queue: queue)

        let library = try Metallib(device: device)
        self.library = library
        pGemmMM = try library.pipeline("sp_gemm_q4_mm")
        // 0: narrow passes use the wide split-K kernels (a 16-row tile beats the 8-row one even
        // for eight rows); a positive count selects the 8 x 32 kernel with that many partitions.
        let partitions = Int(ProcessInfo.processInfo.environment["SPLOSH_SPLIT"] ?? "") ?? 0
        split = partitions > 0 ? (try library.pipeline("sp_gemm_q4_split\(partitions)"), partitions) : nil
        splitMinRows = Int(ProcessInfo.processInfo.environment["SPLOSH_SPLIT_MIN"] ?? "") ?? 3
        pNaPrepare = try library.pipeline((ProcessInfo.processInfo.environment["SPLOSH_NA"] ?? "").contains("half") ? "sp_na_prepare_half" : "sp_na_prepare")
        // Variant name encodes the tile: m<rows>n<columns>s<simdgroups>.
        let naName = ProcessInfo.processInfo.environment["SPLOSH_NA"] ?? "m32n128s4"
        if naName == "off" {
            na = nil
        } else {
            let digits = naName.split(whereSeparator: { !$0.isNumber }).compactMap { Int($0) }
            guard digits.count >= 3 else { throw EngineError.unsupportedDevice("bad SPLOSH_NA '\(naName)'") }
            // Probe suffixes (for example "_noepi") exist only for the tiled kernels.
            let tiledRun = ProcessInfo.processInfo.environment["SPLOSH_TILED"] != "0"
            let base = tiledRun ? String(naName.split(separator: "_")[0]) : naName
            na = (try library.pipeline("sp_gemm_q4_na_" + base), digits[0], digits[1], digits[2] * 32)
        }
        matrixRows = Int(ProcessInfo.processInfo.environment["SPLOSH_MM_ROWS"] ?? "") ?? 3
        skip = Set((ProcessInfo.processInfo.environment["SPLOSH_SKIP"] ?? "").split(separator: ",").map(String.init))
        // A GGUF embedding names its format's row gather; it takes sp_embed_q4's bindings.
        pEmbed = try library.pipeline(weights.embed.kernel.map { "sp_gguf_embed_" + $0 } ?? "sp_embed_q4")
        pGemm = try library.pipeline("sp_gemm_q4")
        pNorm = try library.pipeline("sp_rmsnorm")
        pMlpIn = try library.pipeline("sp_mlp_in")
        pGdn = try library.pipeline("sp_gdn_fast")
        // The kernels by name, so that a candidate in another file can be tried against them
        // (SPLOSH_GDN_CHAIN_GROUPS: threadgroups a head for the recurrence, of 1024 / that many
        // threads unless SPLOSH_GDN_CHAIN_THREADS says otherwise). A choice of kernels that do
        // not work together is refused here.
        gdnChoice = try GdnKernelChoice(environment: ProcessInfo.processInfo.environment)
        pGdnPrepare = try library.pipeline(gdnChoice.prepare)
        pGdnScan = try library.pipeline(gdnChoice.chains)
        pGdnFinish = try library.pipeline(gdnChoice.finish)
        pGdnHistory = try library.pipeline(gdnChoice.history)
        pGdnCommit = try library.pipeline("sp_gdn_commit")
        pCopyRows = try library.pipeline("sp_copy_rows")
        pAttentionFused = try library.pipeline(config.kvFormat == .q4 ? "sp_attention_fused_q4" : "sp_attention_fused")
        pProbe = try library.pipeline("sp_probe_read")
        var gemms: [Int: MTLComputePipelineState] = [:], mlps: [Int: MTLComputePipelineState] = [:]
        for width in [1, 2, 4, 8] {
            gemms[width] = try library.pipeline("sp_gemm_q4_r\(width)")
            mlps[width] = try library.pipeline("sp_mlp_in_r\(width)")
        }
        gemmByWidth = gemms; mlpByWidth = mlps
        fixedWidth = ProcessInfo.processInfo.environment["SPLOSH_GEMM"] != "variable"
        switch ProcessInfo.processInfo.environment["SPLOSH_TILE"] ?? "r8t4" {
        case "r8t2": tile = (8, 2, try library.pipeline("sp_gemm_q4_r8t2"), try library.pipeline("sp_mlp_in_r8t2"))
        case "r8t4": tile = (8, 4, try library.pipeline("sp_gemm_q4_r8t4"), try library.pipeline("sp_mlp_in_r8t4"))
        case "r8t8": tile = (8, 8, try library.pipeline("sp_gemm_q4_r8t8"), try library.pipeline("sp_mlp_in_r8t4"))
        case "r16t4": tile = (16, 4, try library.pipeline("sp_gemm_q4_r16t4"), try library.pipeline("sp_mlp_in_r16t4"))
        default: tile = nil
        }
        pSiluMul = try library.pipeline("sp_silu_mul")
        pSiluMulNa = try library.pipeline("sp_silu_mul_na")
        if MlpFusedPlan.variants.contains(mlpFused) {
            for name in MlpFusedPlan.kernels(variant: mlpFused) { mlpFusedPipelines[name] = try library.pipeline(name) }
        }
        pGather = try library.pipeline("sp_gather_rows")
        pQPrep = try library.pipeline("sp_attn_q_prepare")
        pKVStore = try library.pipeline(config.kvFormat == .q4 ? "sp_attn_kv_store_q4" : "sp_attn_kv_store")
        scratchRowCap = ((config.maxRows + 127) / 128) * 128
        pAttention = try library.pipeline("sp_attention")
        pConv = try library.pipeline("sp_gdn_conv")
        pL2 = try library.pipeline("sp_gdn_l2norm")
        pScan = try library.pipeline("sp_gdn_scan")
        pGatedNorm = try library.pipeline("sp_gdn_gated_norm")

        // The reduction kernels assume one simdgroup per 32-thread group.
        guard pGemm.threadExecutionWidth == Self.lanes else {
            throw EngineError.unsupportedDevice("thread execution width \(pGemm.threadExecutionWidth); kernels require \(Self.lanes)")
        }
        guard pAttention.maxTotalThreadsPerThreadgroup >= g.headDim else {
            throw EngineError.unsupportedDevice("attention needs \(g.headDim) threads per group")
        }
        guard g.headDim == Self.pageTokens else {
            throw EngineError.unsupportedDevice("attention kernel requires headDim == page size")
        }

        var scratch = 0
        var owned: [MTLBuffer] = []
        func make(_ label: String, _ bytes: Int) throws -> MTLBuffer {
            guard let buffer = device.makeBuffer(length: max(bytes, 16), options: .storageModeShared) else {
                throw EngineError.allocationFailed(label, bytes: bytes)
            }
            buffer.label = "splosh." + label
            scratch += bytes
            owned.append(buffer)
            return buffer
        }
        // Matrix-tiled GEMMs read and write whole 32-row tiles, so row scratch is padded to 32.
        let rows = ((config.maxRows + 127) / 128) * 128, f = MemoryLayout<Float>.stride, u = MemoryLayout<UInt32>.stride
        tokens = try make("tokens", rows * u)
        // Widest GEMM input: the draft model's feature projection reads five hidden states.
        let widest = max(g.intermediate, g.hidden * Self.featureLayers.count)
        naInput = try make("naInput", rows * widest * MemoryLayout<UInt16>.stride)
        naSums = try make("naSums", rows * (widest / 64) * f)
        xnB = try make("xnB", rows * g.hidden * 2); xnSums = try make("xnSums", rows * (g.hidden / 64) * f)
        // Deferred normalisation (see SpNaEmit): the operand a residual GEMM emits for the next
        // norm, the sums of squares its scale comes from, and that per-row scale.
        hB = try make("hB", rows * g.hidden * 2); hSums = try make("hSums", rows * (g.hidden / 64) * f)
        hSquares = try make("hSquares", rows * (g.hidden / 64) * f)
        rowInv = try make("rowInv", rows * f); rowOnes = try make("rowOnes", rows * f)
        let ones = rowOnes.contents().assumingMemoryBound(to: Float.self)
        for index in 0..<rows { ones[index] = 1 }
        currentInv = rowOnes
        pRowInv = try library.pipeline("sp_row_inv")
        pGemmBf16 = try library.pipeline("sp_gemm_q4_b")
        pGemmEmitWide = try library.pipeline("sp_gemm_q4_nat_m32n128s4_emit")
        pGemmEmitSplit = try library.pipeline("sp_gemm_q4_split_tiled_m16n64p4_emit")
        coreB = try make("coreB", rows * g.gdnValueDim * 2); coreSums = try make("coreSums", rows * (g.gdnValueDim / 64) * f)
        gateB = try make("gateB", rows * g.intermediate * 2); gateSums = try make("gateSums", rows * (g.intermediate / 64) * f)
        pNormNa = try library.pipeline("sp_rmsnorm_na")
        // Attention implementations over the q4 pool: "dense" (three accelerator passes, the
        // default), "na" (accelerator span scan), "scan" (ordinary arithmetic span scan).
        let attentionMode: String
        switch config.kvFormat {
        case .int8: attentionMode = "i8"
        case .q4: attentionMode = ProcessInfo.processInfo.environment["SPLOSH_ATTN"] ?? "dense"
        case .fp16: attentionMode = "fused"
        }
        let int8Layout = attentionMode == "i8"
        let acceleratorLayout = attentionMode == "na" || attentionMode == "dense" || int8Layout
        if attentionMode == "scan" || attentionMode == "na" || int8Layout {
            pAttnMerge = try library.pipeline("sp_attn_merge")
            attnPartials = try make("attnPartials", rows * g.heads * Self.maxSpans * 258 * f)
        }
        if attentionMode == "scan" { pAttnScan = try library.pipeline("sp_attn_scan_q4") }
        if acceleratorLayout {
            pAttnQPrepNa = try library.pipeline("sp_attn_q_prepare_na")
            pAttnKVStoreNa = try library.pipeline(int8Layout ? "sp_attn_kv_store_i8"
                                                  : "sp_attn_kv_store_na")
            attnBlocks = try make("attnBlocks", rows * 2 * u)
            attnRowQ = try make("attnRowQ", rows * u)
            // Query rows are block-padded: every slot run can leave one partly filled block.
            attnQrowCap = (rows / 8 + config.maxSlots + 2) * 8
            attnQueries = try make("attnQueries", g.kvHeads * (attnQrowCap + 8) * (g.heads / g.kvHeads) * g.headDim * 2)
            attnQuerySums = try make("attnQuerySums", g.kvHeads * (attnQrowCap + 8) * (g.heads / g.kvHeads) * 4 * f)
        }
        if int8Layout {
            // SPLOSH_ATTN_SHAPE = "m<queries>c<tokens>s<simdgroups>", e.g. m48c64s8.
            let shape = ProcessInfo.processInfo.environment["SPLOSH_ATTN_SHAPE"] ?? "r48c64s8"
            pAttnScanNa = try library.pipeline("sp_attn_scan_i8_" + shape)
            let digits = shape.split(whereSeparator: { !$0.isNumber }).compactMap { Int($0) }
            attnBlockRows = (digits.first ?? 48) / (g.heads / g.kvHeads)
            // "s<n>" simdgroups for the score matmul, optional "v<n>" for the value matmul.
            attnScanThreads = max(digits.count > 2 ? digits[2] : 8, shape.contains("v") && digits.count > 3 ? digits[3] : 0) * 32
        }
        if attentionMode == "na" {
            // SPLOSH_ATTN_SHAPE = "m<queries>c<tokens>", e.g. m48c64; unset uses the fixed kernel.
            if let shape = ProcessInfo.processInfo.environment["SPLOSH_ATTN_SHAPE"] {
                pAttnScanNa = try library.pipeline("sp_attn_scan_na_" + shape)
                let digits = shape.split(whereSeparator: { !$0.isNumber }).compactMap { Int($0) }
                attnBlockRows = (digits.first ?? 24) / (g.heads / g.kvHeads)
            } else {
                pAttnScanNa = try library.pipeline("sp_attn_scan_na")
            }
        }
        if attentionMode == "dense" {
            pAttnScores = try library.pipeline("sp_attn_scores_na")
            pAttnSoftmax = try library.pipeline("sp_attn_softmax")
            pAttnValues = try library.pipeline("sp_attn_values_na")
            pAttnValuesMerge = try library.pipeline("sp_attn_values_merge")
            // Scratch per query-token: fp32 score plus one fp16 operand (four for per-group affine).
            // [kvHead][block][4 groups][segment][48 x 64] floats
            attnValuePartials = try make("attnValuePartials", g.kvHeads * (attnQrowCap / 8) * 4 * Self.maxValueSegments * 48 * 64 * f)
            attnBlockRows = 8
            // Scores are fp32 and the four value operands fp16: 12 bytes per query-token.
            attnScratchBytes = (Int(ProcessInfo.processInfo.environment["SPLOSH_ATTN_SCRATCH_MB"] ?? "") ?? 1536) << 20
            let perToken = 4 + 2 * attnOperandArrays
            attnScores = try make("attnScores", attnScratchBytes / perToken * 4)
            attnWeighted = try make("attnWeighted", attnScratchBytes / perToken * 2 * attnOperandArrays)
            attnStats = try make("attnStats", g.heads * attnQrowCap * 8 * f)
        }
        separateKVMeta = acceleratorLayout
        kvMetaBytesPerVector = int8Layout ? 4 : 16
        kvCodeBytesPerVector = int8Layout ? 256 : 128
        rowSlot = try make("rowSlot", rows * u)
        rowPos = try make("rowPos", rows * u)
        runs = try make("runs", rows * 3 * u)
        runInfo = try make("runInfo", rows * 2 * u)
        gatherIndex = try make("gatherIndex", rows * u)
        rowRead = try make("rowRead", rows * u)
        rowWrite = try make("rowWrite", rows * u)
        features = try make("features", rows * Self.featureLayers.count * g.hidden * f)
        pageTable = try make("pageTable", config.maxSlots * maxPagesPerSlot * u)
        h = try make("h", rows * g.hidden * f)
        xn = try make("xn", rows * g.hidden * f)
        proj = try make("proj", rows * g.heads * g.headDim * 2 * f)
        z = try make("z", rows * g.gdnValueDim * f)
        aBuf = try make("a", rows * g.gdnValueHeads * f)
        bBuf = try make("b", rows * g.gdnValueHeads * f)
        core = try make("core", rows * g.gdnValueDim * f)
        // Post-conv q, k and v per value head, for the barrier-free recurrence.
        gdnQ = try make("gdnQ", rows * g.gdnValueDim * f)
        gdnK = try make("gdnK", rows * g.gdnValueDim * f)
        gdnV = try make("gdnV", rows * g.gdnValueDim * f)
        gdnNorm = try make("gdnNorm", rows * g.gdnValueHeads * 2 * f)
        kproj = try make("kproj", rows * g.kvHeads * g.headDim * f)
        vproj = try make("vproj", rows * g.kvHeads * g.headDim * f)
        qn = try make("qn", rows * g.heads * g.headDim * f)
        gate = try make("gate", rows * g.intermediate * f)
        up = try make("up", rows * g.intermediate * f)
        let logitRows = ((config.maxLogitRows + 127) / 128) * 128
        hs = try make("hs", logitRows * g.hidden * f)
        hsn = try make("hsn", logitRows * g.hidden * f)
        logits = try make("logits", config.maxLogitRows * g.vocab * f)
        scratchBytes = scratch

        // Session state. Shared-storage buffers are zero-filled and committed lazily, so the KV
        // pool costs physical memory only for pages a session has actually written.
        // Bytes per (token, KV head) vector: a 144-byte q4 record, or 256 half floats.
        // "na": 128 code bytes per vector here plus 16 metadata bytes in a second buffer.
        let vectorBytes = config.kvFormat == .fp16 ? g.headDim * MemoryLayout<UInt16>.stride : (separateKVMeta ? kvCodeBytesPerVector : 144)
        kvLayerPageBytes = Self.pageTokens * g.kvHeads * vectorBytes
        kvMetaPageBytes = separateKVMeta ? Self.pageTokens * g.kvHeads * kvMetaBytesPerVector : 0
        stateLayerBytes = g.gdnValueHeads * g.gdnHeadDim * g.gdnHeadDim * f
        // Conv history per value head: q, k and v channels, three samples each.
        convLayerBytes = g.gdnValueHeads * 3 * g.gdnHeadDim * 3 * f
        for index in 0..<g.fullLayerCount {
            guard let k = device.makeBuffer(length: config.kvPages * kvLayerPageBytes, options: .storageModeShared),
                  let v = device.makeBuffer(length: config.kvPages * kvLayerPageBytes, options: .storageModeShared) else {
                throw EngineError.allocationFailed("KV pool", bytes: config.kvPages * kvLayerPageBytes)
            }
            k.label = "splosh.kv.k\(index)"; v.label = "splosh.kv.v\(index)"
            kPools.append(k); vPools.append(v)
            if kvMetaPageBytes > 0 {
                guard let km = device.makeBuffer(length: config.kvPages * kvMetaPageBytes, options: .storageModeShared),
                      let vm = device.makeBuffer(length: config.kvPages * kvMetaPageBytes, options: .storageModeShared) else {
                    throw EngineError.allocationFailed("KV metadata", bytes: config.kvPages * kvMetaPageBytes)
                }
                kMeta.append(km); vMeta.append(vm)
            }
        }
        for index in 0..<g.linearLayerCount {
            // The recurrent state has one unit per slot; a speculative row journals its update
            // instead of writing a state. Conv history is small and keeps a unit per row.
            let journalBytes = config.maxSlots * Self.journalRows * g.gdnValueHeads * Self.journalWidth * f
            guard let s = device.makeBuffer(length: config.maxSlots * stateLayerBytes, options: .storageModeShared),
                  let c = device.makeBuffer(length: config.stateUnits * convLayerBytes, options: .storageModeShared),
                  let j = device.makeBuffer(length: journalBytes, options: .storageModeShared) else {
                throw EngineError.allocationFailed("GDN state", bytes: config.maxSlots * stateLayerBytes)
            }
            s.label = "splosh.gdn.state\(index)"; c.label = "splosh.gdn.conv\(index)"
            gdnState.append(s); convState.append(c); gdnJournal.append(j)
        }
        freePages = (0..<UInt32(config.kvPages)).reversed()
        slotPages = Array(repeating: [], count: config.maxSlots)
        guard config.stateUnits >= config.maxSlots else { throw EngineError.invalidRows("stateUnits must be at least maxSlots") }
        slotUnit = Array(0..<config.maxSlots)
        spans = weights.resident.buffers
        freeUnits = Array((config.maxSlots..<config.stateUnits).reversed())
        pendingVerify = Array(repeating: [], count: config.maxSlots)
        journalPending = Array(repeating: 0, count: config.maxSlots)
        if weights.tiledLayout || ProcessInfo.processInfo.environment["SPLOSH_TILED"] != "0", let na {
            let name = ProcessInfo.processInfo.environment["SPLOSH_NA"] ?? "m32n128s4"
            naTiled = try library.pipeline("sp_gemm_q4_nat_" + name)
            if name == "m32n128s4", ProcessInfo.processInfo.environment["SPLOSH_WHOLE_TILES"] != "0" {
                pGemmWhole = try library.pipeline("sp_gemm_q4_nat_m32n128s4_whole")
                pGemmWholeResidual = try library.pipeline("sp_gemm_q4_nat_m32n128s4_whole_residual")
            }
            pNaPrepareTiled = try library.pipeline("sp_na_prepare_tiled")
            if ProcessInfo.processInfo.environment["SPLOSH_TILED_LANE"] != "0" {
                tiledLanePartitions = Int(ProcessInfo.processInfo.environment["SPLOSH_TILED_LANE"] ?? "") ?? 8
                for width in [1, 2] {
                    tiledLane[width] = (try library.pipeline("sp_gemm_q4_tiled_r\(width)p\(tiledLanePartitions)"),
                                        try library.pipeline("sp_mlp_in_tiled_r\(width)p\(tiledLanePartitions)"))
                }
            }
            splitColumns = Int(ProcessInfo.processInfo.environment["SPLOSH_SPLIT_N"] ?? "") ?? 32
            if let split {
                splitTiled = try library.pipeline(splitColumns == 32 ? "sp_gemm_q4_split_tiled\(split.partitions)"
                                                  : "sp_gemm_q4_split_tiled_n\(splitColumns)p\(split.partitions)")
            }
            // SPLOSH_SPLIT_WIDE="16:8,32:4" selects (tile rows : partitions); "off" disables.
            let wideSpec = ProcessInfo.processInfo.environment["SPLOSH_SPLIT_WIDE"] ?? "16:4:64,32:4"
            if wideSpec != "off" {
                for item in wideSpec.split(separator: ",") {
                    let parts = item.split(separator: ":").compactMap { Int($0) }
                    guard parts.count >= 2 else { continue }
                    let probe = ProcessInfo.processInfo.environment["SPLOSH_SPLIT_PROBE"].map { "_" + $0 } ?? ""
                    // Optional third field: tile columns (32 unless given).
                    let columns = parts.count > 2 ? parts[2] : 32
                    let name = columns == 32 ? "sp_gemm_q4_split_tiled_m\(parts[0])p\(parts[1])" : "sp_gemm_q4_split_tiled_m\(parts[0])n\(columns)p\(parts[1])"
                    splitWide.append((parts[0], parts[1], columns, try library.pipeline(name + probe)))
                }
            }
            naRowMajorSums = name.hasSuffix("_biasmm")
            _ = na
            try buildTiles()
        }
        try buildGgufPipelines()
        if ProcessInfo.processInfo.environment["SPLOSH_RESIDENCY"] != "0" {
            // The KV pool is left out: it is far larger than what is in use, residency commits
            // all of it (26 GB of footprint instead of 11), and the waits were longer with it
            // in the set than without. What it still costs after an idle spell grows with the
            // pool: 0.16 s with a 2 GiB pool, 0.28 s with 18 GiB.
            let descriptor = MTLResidencySetDescriptor()
            descriptor.label = "splosh.engine"
            descriptor.initialCapacity = owned.count + spans.count + 3 * g.linearLayerCount + 256
            let set = try device.makeResidencySet(descriptor: descriptor)
            set.addAllocations(owned + spans + gdnState + convState + gdnJournal)
            set.commit()
            set.requestResidency()
            graph.queue.addResidencySet(set)
            residency = set
        }
    }

    /// Make every eligible q4 weight available in the tiled layout. A tiled artifact already
    /// stores them that way, so its resident tensors are used as they are. A row-major artifact
    /// is re-ordered into separate buffers, which doubles the weight memory — `splosh convert
    /// --retile` avoids that.
    private func buildTiles() throws {
        var handles: [Q4Handle] = [weights.lmHead]
        for block in weights.blocks {
            handles += [block.gate, block.up, block.down]
            switch block.mixer {
            case .linear(let m): handles += [m.qkv, m.z, m.out]
            case .full(let m): handles += [m.q, m.k, m.v, m.o]
            }
        }
        // A GGUF weight's planes are not affine codes: nothing that reads this map may take them.
        let list = handles.filter { $0.kernel == nil && Converter.isTiledWeight(name: $0.packed.name, rows: $0.rows) }
        if weights.tiledLayout {
            for w in list { tiled[w.packed.name] = (w.packed, w.scales, w.biases) }
            return
        }
        var made: [(MTLBuffer, MTLBuffer, MTLBuffer)] = []
        for w in list {
            guard let packed = device.makeBuffer(length: w.packed.byteLength, options: .storageModeShared),
                  let scales = device.makeBuffer(length: w.scales.byteLength, options: .storageModeShared),
                  let biases = device.makeBuffer(length: w.biases.byteLength, options: .storageModeShared) else {
                throw EngineError.allocationFailed("tiled weights", bytes: w.residentBytes)
            }
            made.append((packed, scales, biases))
            duplicatedWeightBytes += w.residentBytes
        }
        let resident = weights.resident
        let buffers = made
        DispatchQueue.concurrentPerform(iterations: list.count) { index in
            let w = list[index]
            let groups = w.groupsPerRow, stride = w.rowStrideWords * 4
            let src = resident.buffer(w.packed).contents().advanced(by: w.packed.offset)
            let dst = buffers[index].0.contents()
            let srcS = resident.buffer(w.scales).contents().advanced(by: w.scales.offset).assumingMemoryBound(to: UInt16.self)
            let srcB = resident.buffer(w.biases).contents().advanced(by: w.biases.offset).assumingMemoryBound(to: UInt16.self)
            let dstS = buffers[index].1.contents().assumingMemoryBound(to: UInt16.self)
            let dstB = buffers[index].2.contents().assumingMemoryBound(to: UInt16.self)
            for tile in 0..<(w.rows / 128) {
                for group in 0..<groups {
                    let base = (tile * groups + group) * 128
                    for n in 0..<128 {
                        let row = tile * 128 + n
                        memcpy(dst.advanced(by: (base + n) * 32), src.advanced(by: row * stride + group * 32), 32)
                        dstS[base + n] = srcS[row * groups + group]
                        dstB[base + n] = srcB[row * groups + group]
                    }
                }
            }
        }
        for (w, buffers) in zip(list, made) {
            func handle(_ original: TensorHandle, _ buffer: MTLBuffer) -> TensorHandle {
                TensorHandle(name: original.name, span: registerSpan(buffer), offset: 0, byteLength: buffer.length, shape: original.shape)
            }
            tiled[w.packed.name] = (handle(w.packed, buffers.0), handle(w.scales, buffers.1), handle(w.biases, buffers.2))
        }
    }

    /// One GGUF format's tile kernels (engine_gguf.metal): split-K over 16-row and 32-row tiles,
    /// and the wide one, 128 rows a threadgroup.
    private struct GgufTiles { let split16, split32, wide: MTLComputePipelineState }
    /// The GGUF GEMM kernels by format token (`Q4Handle.kernel`): the tiles, their `heads` forms
    /// and the small GEMM for a weight that is not whole 32-row tiles. Empty for an MLX artifact.
    private var ggufTiles: [String: GgufTiles] = [:], ggufHeadTiles: [String: GgufTiles] = [:]
    private var ggufSmall: [String: MTLComputePipelineState] = [:]
    /// The gated-delta out projections of a GGUF artifact, by packed tensor name: their columns
    /// are in the file's order of the value heads, which the `heads` kernels read.
    private var ggufHeadOrder: Set<String> = []

    /// Compile every GGUF GEMM kernel the artifact's handles can take, so that none is first
    /// built inside a step: for each format token present the three tiles, their `heads` forms
    /// for the tokens of the gated-delta out projections, and the small GEMM for the tokens of
    /// the weights that are not whole tiles (the 48-row projections). A handle with no token,
    /// which is every handle of an MLX artifact, adds nothing.
    private func buildGgufPipelines() throws {
        func tiles(_ format: String, heads: String = "") throws -> GgufTiles {
            GgufTiles(split16: try library.pipeline("sp_gguf_split_m16_" + heads + format),
                      split32: try library.pipeline("sp_gguf_split_m32_" + heads + format),
                      wide: try library.pipeline("sp_gguf_wide_" + heads + format))
        }
        var handles: [Q4Handle] = [weights.lmHead]
        for block in weights.blocks {
            handles += [block.gate, block.up, block.down]
            switch block.mixer {
            case .linear(let m):
                handles += [m.qkv, m.z, m.a, m.b, m.out]
                if let format = m.out.kernel {
                    ggufHeadOrder.insert(m.out.packed.name)
                    if ggufHeadTiles[format] == nil { ggufHeadTiles[format] = try tiles(format, heads: "heads_") }
                }
            case .full(let m): handles += [m.q, m.k, m.v, m.o]
            }
        }
        for w in handles {
            guard let format = w.kernel else { continue }
            if ggufTiles[format] == nil { ggufTiles[format] = try tiles(format) }
            if w.rows % 32 != 0, ggufSmall[format] == nil { ggufSmall[format] = try library.pipeline("sp_gguf_small_" + format) }
        }
    }

    // MARK: - Memory

    public var memory: EngineMemory {
        let perPage = (kvLayerPageBytes + kvMetaPageBytes) * 2 * g.fullLayerCount
        let perSlot = (stateLayerBytes + convLayerBytes) * g.linearLayerCount
        return EngineMemory(
            weightBytes: weights.resident.residentBytes,
            kvPoolBytes: perPage * config.kvPages,
            kvBytesPerPage: perPage,
            kvPagesTotal: config.kvPages,
            kvPagesUsed: config.kvPages - freePages.count,
            stateBytesPerSlot: perSlot,
            stateBytesTotal: stateLayerBytes * g.linearLayerCount * config.maxSlots
                + convLayerBytes * g.linearLayerCount * (config.stateUnits - freeUnits.count),
            stateUnitsUsed: config.stateUnits - freeUnits.count,
            scratchBytes: scratchBytes,
            deviceWorkingSetBytes: Int(device.recommendedMaxWorkingSetSize),
            deviceAllocatedBytes: device.currentAllocatedSize)
    }

    public func pagesHeld(by slot: Int) -> Int { slotPages[slot].count }
    public var freePageCount: Int { freePages.count }

    // MARK: - Slot lifecycle

    /// Clear a slot's recurrent state and return its KV pages to the pool.
    public func resetSlot(_ slot: Int) {
        freePages.append(contentsOf: slotPages[slot])
        slotPages[slot].removeAll(keepingCapacity: true)
        discardVerify(slot)
        journalPending[slot] = 0
        let unit = slotUnit[slot]
        for index in 0..<g.linearLayerCount {
            memset(gdnState[index].contents().advanced(by: slot * stateLayerBytes), 0, stateLayerBytes)
            memset(convState[index].contents().advanced(by: unit * convLayerBytes), 0, convLayerBytes)
        }
    }

    /// Drop the units of an unaccepted verify run.
    public func discardVerify(_ slot: Int) {
        freeUnits.append(contentsOf: pendingVerify[slot])
        pendingVerify[slot].removeAll(keepingCapacity: true)
    }

    /// Keep the first `rows` rows of the slot's last verify run: the state after that row
    /// becomes current, the other units are released, and KV past `tokenCount` is dropped.
    public func acceptVerify(_ slot: Int, rows: Int, tokenCount: Int) {
        let units = pendingVerify[slot]
        precondition(rows >= 1 && rows <= units.count, "acceptVerify outside the verified run")
        freeUnits.append(slotUnit[slot])
        slotUnit[slot] = units[rows - 1]
        for (index, unit) in units.enumerated() where index != rows - 1 { freeUnits.append(unit) }
        pendingVerify[slot].removeAll(keepingCapacity: true)
        // The accepted rows' state updates are replayed from the journal by the slot's next step.
        journalPending[slot] = rows
        truncate(slot, tokenCount: tokenCount)
    }

    private func ensurePage(slot: Int, position: Int) throws {
        guard position < config.maxContext else {
            throw EngineError.contextExceeded(position: position, limit: config.maxContext)
        }
        let index = position / Self.pageTokens
        let table = pageTable.contents().assumingMemoryBound(to: UInt32.self)
        while slotPages[slot].count <= index {
            guard let page = freePages.popLast() else {
                throw EngineError.kvPoolExhausted(needed: index + 1 - slotPages[slot].count, free: 0)
            }
            table[slot * maxPagesPerSlot + slotPages[slot].count] = page
            slotPages[slot].append(page)
        }
    }

    /// Keep only the first `tokenCount` tokens of a slot's KV, returning whole unused pages.
    /// The recurrent state is not rewound; pair this with `importState`.
    public func truncate(_ slot: Int, tokenCount: Int) {
        let keep = (tokenCount + Self.pageTokens - 1) / Self.pageTokens
        while slotPages[slot].count > keep { freePages.append(slotPages[slot].removeLast()) }
    }

    /// The gated-delta recurrent and conv state of one slot (about 150 MB).
    public func exportState(_ slot: Int) throws -> Data {
        try commitJournal(slot)
        // Copied once, straight into the result (`Data(bytes:)` then `append` copies twice).
        var state = Data(count: (stateLayerBytes + convLayerBytes) * g.linearLayerCount)
        state.withUnsafeMutableBytes { raw in
            var cursor = raw.baseAddress!
            for index in 0..<g.linearLayerCount {
                memcpy(cursor, gdnState[index].contents().advanced(by: slot * stateLayerBytes), stateLayerBytes)
                cursor += stateLayerBytes
                memcpy(cursor, convState[index].contents().advanced(by: slotUnit[slot] * convLayerBytes), convLayerBytes)
                cursor += convLayerBytes
            }
        }
        return state
    }

    /// Fold a slot's accepted journal entries into its stored state. A step does this itself
    /// for every slot it touches; this is for reading the state between steps.
    private func commitJournal(_ slot: Int) throws {
        guard journalPending[slot] > 0 else { return }
        var p = (UInt32(slot), UInt32(journalPending[slot]), UInt32(g.gdnValueHeads), UInt32(g.gdnHeadDim))
        try graph.begin(label: "splosh.gdn.commit", concurrent: true)
        for layer in 0..<g.linearLayerCount {
            graph.dispatch(pGdnCommit, grid: MTLSize(width: g.gdnValueHeads, height: 1, depth: 1),
                           threadsPerGroup: MTLSize(width: 1024, height: 1, depth: 1)) { e in
                e.setBuffer(gdnState[layer], offset: 0, index: 0)
                e.setBuffer(gdnJournal[layer], offset: 0, index: 1)
                e.setBytes(&p, length: 16, index: 2)
            }
        }
        try graph.commitAndWait()
        journalPending[slot] = 0
    }

    public func importState(_ slot: Int, state: Data) throws {
        guard state.count == (stateLayerBytes + convLayerBytes) * g.linearLayerCount else {
            throw EngineError.snapshotMismatch("unexpected state byte count")
        }
        if gdnChoice.readsKeyHeadHistory {
            // The chosen kernels read a key head's q and k history from its first value head.
            // Every state this engine exports has the same history in the other two; one that
            // does not would be continued differently from the default kernels, so it is refused.
            let shared = state.withUnsafeBytes { raw -> Bool in
                (0..<g.linearLayerCount).allSatisfy { index in
                    let conv = raw.baseAddress!.advanced(by: index * (stateLayerBytes + convLayerBytes) + stateLayerBytes)
                    return GdnKernelChoice.keyHistoryIsShared(conv.assumingMemoryBound(to: Float.self), valueHeads: g.gdnValueHeads,
                                                              keyHeads: g.gdnKeyHeads, headDim: g.gdnHeadDim)
                }
            }
            guard shared else {
                throw EngineError.snapshotMismatch("conv history of q and k differs between the value heads of a key head, which \(gdnChoice.prepare) / \(gdnChoice.chains) do not read")
            }
        }
        journalPending[slot] = 0
        state.withUnsafeBytes { raw in
            var cursor = raw.baseAddress!
            for index in 0..<g.linearLayerCount {
                memcpy(gdnState[index].contents().advanced(by: slot * stateLayerBytes), cursor, stateLayerBytes)
                cursor = cursor.advanced(by: stateLayerBytes)
                memcpy(convState[index].contents().advanced(by: slotUnit[slot] * convLayerBytes), cursor, convLayerBytes)
                cursor = cursor.advanced(by: convLayerBytes)
            }
        }
    }

    // MARK: - Snapshots (prefix cache)

    /// Every per-layer KV buffer with its bytes per page, in a fixed order.
    private func kvBuffers(_ layer: Int) -> [(buffer: MTLBuffer, pageBytes: Int)] {
        var list = [(kPools[layer], kvLayerPageBytes), (vPools[layer], kvLayerPageBytes)]
        if kvMetaPageBytes > 0 { list += [(kMeta[layer], kvMetaPageBytes), (vMeta[layer], kvMetaPageBytes)] }
        return list
    }

    private var kvSnapshotBytesPerPage: Int { (kvLayerPageBytes + kvMetaPageBytes) * 2 * g.fullLayerCount }

    public func exportSlot(_ slot: Int, tokenCount: Int) throws -> SlotSnapshot {
        SlotSnapshot(tokenCount: tokenCount, state: try exportState(slot), kv: exportKV(slot, tokenCount: tokenCount))
    }

    /// The KV pages covering a slot's first `tokenCount` tokens. KV is append-only, so this is
    /// valid for any prefix of what the slot has evaluated, whatever was evaluated after it.
    public func exportKV(_ slot: Int, tokenCount: Int) -> Data {
        let pages = min((tokenCount + Self.pageTokens - 1) / Self.pageTokens, slotPages[slot].count)
        var kv = Data(count: pages * kvSnapshotBytesPerPage)
        kv.withUnsafeMutableBytes { raw in
            guard var cursor = raw.baseAddress else { return }
            for index in 0..<g.fullLayerCount {
                for (pool, pageBytes) in kvBuffers(index) {
                    for page in slotPages[slot].prefix(pages) {
                        memcpy(cursor, pool.contents().advanced(by: Int(page) * pageBytes), pageBytes)
                        cursor += pageBytes
                    }
                }
            }
        }
        return kv
    }

    /// The first `keepTokens` tokens' worth of a KV snapshot of `storedTokens` tokens. KV for a
    /// position depends only on the tokens before it, so a prefix of one prompt's KV is the KV
    /// of any prompt that begins the same way. The last page kept may hold later positions too;
    /// they are never read, and are overwritten as the slot goes on.
    public func kvPrefix(_ kv: Data, storedTokens: Int, keepTokens: Int) throws -> Data {
        let stored = (storedTokens + Self.pageTokens - 1) / Self.pageTokens
        let keep = (keepTokens + Self.pageTokens - 1) / Self.pageTokens
        guard kv.count == stored * kvSnapshotBytesPerPage, keep <= stored else {
            throw EngineError.snapshotMismatch("unexpected KV byte count")
        }
        if keep == stored { return kv }
        var prefix = Data(capacity: keep * kvSnapshotBytesPerPage)
        var cursor = kv.startIndex
        for index in 0..<g.fullLayerCount {
            for (_, pageBytes) in kvBuffers(index) {
                prefix.append(kv[cursor..<cursor + keep * pageBytes])
                cursor += stored * pageBytes
            }
        }
        return prefix
    }

    /// Identifies what a stored snapshot is only valid for: these weights, this KV format and
    /// this snapshot layout.
    public var snapshotIdentity: String {
        "\(weights.configHash)-\(weights.resident.residentBytes)-kv.\(config.kvFormat.rawValue)-state.v2"
    }

    /// `kvStoredTokens`: the snapshot's KV is that of a longer prefix, as stored (a mapping of
    /// the file, say), and only the pages covering `snapshot.tokenCount` are read from it; the
    /// same as `kvPrefix` without the copy in between.
    public func importSlot(_ slot: Int, snapshot: SlotSnapshot, kvStoredTokens: Int? = nil) throws {
        let pages = (snapshot.tokenCount + Self.pageTokens - 1) / Self.pageTokens
        let stored = kvStoredTokens.map { ($0 + Self.pageTokens - 1) / Self.pageTokens } ?? pages
        guard snapshot.state.count == (stateLayerBytes + convLayerBytes) * g.linearLayerCount,
              stored >= pages, snapshot.kv.count == stored * kvSnapshotBytesPerPage else {
            throw EngineError.snapshotMismatch("unexpected byte counts")
        }
        resetSlot(slot)
        if snapshot.tokenCount > 0 { try ensurePage(slot: slot, position: snapshot.tokenCount - 1) }
        try importState(slot, state: snapshot.state)
        snapshot.kv.withUnsafeBytes { raw in
            guard var cursor = raw.baseAddress else { return }
            for index in 0..<g.fullLayerCount {
                for (pool, pageBytes) in kvBuffers(index) {
                    for page in slotPages[slot].prefix(pages) {
                        memcpy(pool.contents().advanced(by: Int(page) * pageBytes), cursor, pageBytes)
                        cursor = cursor.advanced(by: pageBytes)
                    }
                    cursor = cursor.advanced(by: (stored - pages) * pageBytes)
                }
            }
        }
    }

    /// `importSlot` with the KV read from another slot's pages directly: KV is append-only, so
    /// the first `tokenCount` tokens' worth stays valid while that slot carries on.
    public func importSlot(_ slot: Int, tokenCount: Int, state: Data, kvFrom source: Int) throws {
        let pages = (tokenCount + Self.pageTokens - 1) / Self.pageTokens
        guard source != slot, pages <= slotPages[source].count else {
            throw EngineError.snapshotMismatch("slot \(source) does not hold \(tokenCount) tokens of KV")
        }
        resetSlot(slot)
        if tokenCount > 0 { try ensurePage(slot: slot, position: tokenCount - 1) }
        try importState(slot, state: state)
        for index in 0..<g.fullLayerCount {
            for (pool, pageBytes) in kvBuffers(index) {
                for (from, to) in zip(slotPages[source].prefix(pages), slotPages[slot].prefix(pages)) {
                    memcpy(pool.contents().advanced(by: Int(to) * pageBytes), pool.contents().advanced(by: Int(from) * pageBytes), pageBytes)
                }
            }
        }
    }

    /// Measure raw GPU read bandwidth over the first weight span, in bytes per second.
    public func probeReadBandwidth(bytes requested: Int = 8 << 30) throws -> Double {
        let source = weights.resident.buffers[0]
        let chunks = min(requested, source.length) / 65536
        guard let out = device.makeBuffer(length: chunks * 4, options: .storageModeShared) else {
            throw EngineError.allocationFailed("probe", bytes: chunks * 4)
        }
        var best = 0.0
        for _ in 0..<3 {
            try graph.begin(label: "splosh.probe")
            var p = UInt32(chunks)
            graph.dispatch(pProbe, grid: MTLSize(width: chunks, height: 1, depth: 1), threadsPerGroup: groupSize) { e in
                e.setBuffer(source, offset: 0, index: 0)
                e.setBuffer(out, offset: 0, index: 1)
                e.setBytes(&p, length: 4, index: 2)
            }
            let seconds = try graph.commitAndWait()
            best = max(best, Double(chunks * 65536) / seconds)
        }
        return best
    }

    /// One small pass that references the KV pool (the rest is in the residency set), committed
    /// and not waited for. Run every half second while idle (BatchScheduler's keep-warm), it
    /// keeps the driver from letting the pool go, and the first step after a pause does not wait
    /// for it to be made resident again. Call it between steps, from the thread that steps.
    public func wake() {
        guard let buffer = graph.queue.makeCommandBuffer(), let e = buffer.makeComputeCommandEncoder() else { return }
        buffer.label = "splosh.wake"
        e.useResources(kPools + vPools + kMeta + vMeta, usage: [.read, .write])
        var p = NormParams(count: 1, dim: UInt32(g.hidden), mode: 1, eps: g.normEps)
        e.setComputePipelineState(pNorm)
        e.setBuffer(h, offset: 0, index: 0)
        bind(e, weights.finalNorm, 1)
        e.setBuffer(wakeOut, offset: 0, index: 2)
        e.setBytes(&p, length: MemoryLayout<NormParams>.stride, index: 3)
        e.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1), threadsPerThreadgroup: groupSize)
        e.endEncoding()
        buffer.commit()
    }
    private lazy var wakeOut = device.makeBuffer(length: g.hidden * 4, options: .storageModeShared)

    /// Fixed cost of one small dispatch inside a command buffer, in seconds.
    public func probeDispatchOverhead(count: Int = 2000, concurrent: Bool = false, barriers: Bool = false, split: Bool = false) throws -> Double {
        try graph.begin(label: "splosh.probe.dispatch", concurrent: concurrent)
        for _ in 0..<count {
            norm(h, weights.finalNorm, xn, count: 1, dim: g.hidden, mode: 1)
            if barriers { graph.barrier() }
            if split { try graph.splitEncoder(concurrent: concurrent) }
        }
        return try graph.commitAndWait() / Double(count)
    }

    /// Cost of a stage boundary expressed as one command buffer per dispatch, committed without
    /// waiting; only the last is awaited.
    public func probeCommandBufferOverhead(count: Int = 2000) throws -> Double {
        let queue = graph.queue
        var p = NormParams(count: 1, dim: UInt32(g.hidden), mode: 1, eps: g.normEps)
        let started = DispatchTime.now().uptimeNanoseconds
        var last: MTLCommandBuffer?
        for _ in 0..<count {
            guard let buffer = queue.makeCommandBuffer(), let e = buffer.makeComputeCommandEncoder() else { throw CommandGraphError.encoderUnavailable }
            e.setComputePipelineState(pNorm)
            e.setBuffer(h, offset: 0, index: 0)
            bind(e, weights.finalNorm, 1)
            e.setBuffer(xn, offset: 0, index: 2)
            e.setBytes(&p, length: MemoryLayout<NormParams>.stride, index: 3)
            e.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1), threadsPerThreadgroup: groupSize)
            e.endEncoding()
            buffer.commit()
            last = buffer
        }
        last?.waitUntilCompleted()
        return Double(DispatchTime.now().uptimeNanoseconds - started) / 1e9 / Double(count)
    }

    /// GPU time for every GEMM of one decode step with no stage boundaries at all: the cost of
    /// the weight traffic alone.
    public func probeWeightPass(rows: Int = 1) throws -> Double {
        var best = Double.infinity
        for _ in 0..<3 {
            try beginPass(label: "splosh.probe.weights")
            // One prepare per distinct input, as a real step would pay, then every GEMM with no
            // further stage boundaries.
            for block in weights.blocks {
                switch block.mixer {
                case .linear(let m):
                    gemm(m.qkv, input: xn, output: proj, residual: false, rows: rows)
                    gemm(m.z, input: xn, output: z, residual: false, rows: rows)
                case .full(let m):
                    gemm(m.q, input: xn, output: proj, residual: false, rows: rows)
                    gemm(m.k, input: xn, output: kproj, residual: false, rows: rows)
                    gemm(m.v, input: xn, output: vproj, residual: false, rows: rows)
                }
                if rows >= matrixRows {
                    gemm(block.gate, input: xn, output: gate, residual: false, rows: rows)
                    gemm(block.up, input: xn, output: up, residual: false, rows: rows)
                } else {
                    mlpIn(block, rows: rows, source: xn)
                }
            }
            gemm(weights.lmHead, input: xn, output: logits, residual: false, rows: rows)
            for block in weights.blocks {
                switch block.mixer {
                case .linear(let m): gemm(m.out, input: core, output: hs, residual: false, rows: rows)
                case .full(let m): gemm(m.o, input: core, output: hs, residual: false, rows: rows)
                }
            }
            for block in weights.blocks { gemm(block.down, input: up, output: hsn, residual: false, rows: rows) }
            best = min(best, try graph.commitAndWait())
        }
        return best
    }

    /// Register-only arithmetic ceilings, in scalar multiply-adds per second.
    public func probeArithmetic() throws -> [(String, Double)] {
        let threads = 1 << 20
        guard let out = device.makeBuffer(length: threads * 8, options: .storageModeShared) else {
            throw EngineError.allocationFailed("probe", bytes: threads * 8)
        }
        var results: [(String, Double)] = []
        // (kernel, iterations, multiply-adds per thread per iteration, matrix kernel)
        let cases: [(String, Int, Double, Bool)] = [
            ("sp_probe_alu_f32", 2000, 8, false), ("sp_probe_alu_wide", 1000, 32, false),
            ("sp_probe_mm_f32", 1000, 4 * 512 / 32, true), ("sp_probe_mm_f16", 1000, 4 * 512 / 32, true),
        ]
        for (name, iterations, perIteration, matrix) in cases {
            let pipeline = try library.pipeline(name)
            try graph.begin(label: "splosh.probe.alu")
            var p = (UInt32(threads), UInt32(iterations))
            if matrix {
                graph.dispatch(pipeline, grid: MTLSize(width: threads / 32, height: 1, depth: 1), threadsPerGroup: groupSize) { e in
                    e.setBuffer(out, offset: 0, index: 0); e.setBytes(&p, length: 8, index: 1)
                }
            } else {
                graph.dispatchThreads(pipeline, threads: MTLSize(width: threads, height: 1, depth: 1)) { e in
                    e.setBuffer(out, offset: 0, index: 0); e.setBytes(&p, length: 8, index: 1)
                }
            }
            let seconds = try graph.commitAndWait()
            results.append((name, Double(threads) * Double(iterations) * perIteration / seconds))
        }
        return results
    }

    /// Run both kernel paths once so the first real request does not pay for pipeline warm-up,
    /// first-touch faults on scratch memory, or paging the weights in. Leaves slot 0 empty.
    public func warmUp() throws {
        resetSlot(0)
        let wide = (0..<config.maxRows).map { EngineRow(slot: 0, token: 0, position: $0, wantLogits: $0 == config.maxRows - 1) }
        try step(wide)
        try step([EngineRow(slot: 0, token: 0, position: config.maxRows, wantLogits: true)])
        resetSlot(0)
    }

    // MARK: - Step

    /// Logits for the i-th row that requested them, in row order. Valid until the next step.
    public func logits(_ index: Int) -> UnsafeBufferPointer<Float> {
        let base = logits.contents().assumingMemoryBound(to: Float.self).advanced(by: index * g.vocab)
        return UnsafeBufferPointer(start: base, count: g.vocab)
    }

    /// Debugging: stop a step after this many blocks, leaving intermediate buffers inspectable.
    public var debugMaxBlocks: Int?

    /// Debugging: the mixer output (`core`) of the last evaluated block, `rows * 6144` floats.
    public func debugCore(rows: Int) -> [Float] {
        Array(UnsafeBufferPointer(start: core.contents().assumingMemoryBound(to: Float.self), count: rows * g.gdnValueDim))
    }

    /// Benchmarking: fill every KV page the slot holds with pseudo-random codes and plausible
    /// scales, so a synthetic context costs what a real one does (untouched pages all map to
    /// one zero page and read far too cheaply).
    public func debugFillKV(_ slot: Int) {
        guard config.kvFormat == .int8 else { return }
        var state: UInt64 = 0x9E37_79B9_7F4A_7C15
        for layer in 0..<g.fullLayerCount {
            for page in slotPages[slot] {
                for pool in [kPools[layer], vPools[layer]] {
                    let words = pool.contents().advanced(by: Int(page) * kvLayerPageBytes).assumingMemoryBound(to: UInt64.self)
                    for index in 0..<kvLayerPageBytes / 8 {
                        state ^= state << 13; state ^= state >> 7; state ^= state << 17
                        words[index] = state
                    }
                }
                for meta in [kMeta[layer], vMeta[layer]] {
                    let scales = meta.contents().advanced(by: Int(page) * kvMetaPageBytes).assumingMemoryBound(to: Float.self)
                    for index in 0..<kvMetaPageBytes / 4 { scales[index] = 0.01 }
                }
            }
        }
    }

    /// Debugging: compare the operand a residual GEMM emitted against the hidden state it was
    /// emitted from. Returns the worst absolute error in the per-64 sums of squares, the worst in
    /// the per-64 operand sums, and how many values are not finite.
    public func debugEmitCheck(rows: Int) -> (squares: Float, sums: Float, nonFinite: Int, worstSquareAt: Int) {
        let hidden = g.hidden, groups = hidden / 64
        let hp = h.contents().assumingMemoryBound(to: Float.self)
        let bp = hB.contents().assumingMemoryBound(to: UInt16.self)
        let sp = hSums.contents().assumingMemoryBound(to: Float.self)
        let qp = hSquares.contents().assumingMemoryBound(to: Float.self)
        var worstSquare: Float = 0, worstSum: Float = 0, bad = 0, at = -1
        for row in 0..<rows {
            for group in 0..<groups {
                var square: Float = 0, sum: Float = 0
                for index in 0..<64 {
                    let value = hp[row * hidden + group * 64 + index]
                    square += value * value
                    sum += Float(bitPattern: UInt32(bp[row * hidden + group * 64 + index]) << 16)
                }
                let es = abs(square - qp[row * groups + group]) / max(square, 1e-6)
                let eu = abs(sum - sp[row * groups + group]) / max(abs(sum), 1e-3)
                if !es.isFinite || !eu.isFinite { bad += 1; continue }
                if es > worstSquare { worstSquare = es; at = row * groups + group }
                worstSum = max(worstSum, eu)
            }
        }
        return (worstSquare, worstSum, bad, at)
    }

    /// Debugging: fill the accelerator operand rows from `first` up to the next multiple of 32 with
    /// a constant, to test whether a tile's unused rows influence its used ones.
    public var debugPadding: Float?

    private func fillPadding(from first: Int, value: Float) {
        let end = ((first + 31) / 32) * 32
        guard end > first else { return }
        let bits = UInt16(truncatingIfNeeded: value.bitPattern >> 16)
        for (buffer, width) in [(xnB, g.hidden), (hB, g.hidden), (coreB, g.gdnValueDim), (gateB, g.intermediate), (naInput, g.intermediate)] {
            let pointer = buffer!.contents().assumingMemoryBound(to: UInt16.self)
            for index in first * width..<min(end * width, buffer!.length / 2) { pointer[index] = bits }
        }
        // Query rows past the last one, in each KV head's region (fp16).
        if let attnQueries {
            let perRow = (g.heads / g.kvHeads) * g.headDim
            let pointer = attnQueries.contents().assumingMemoryBound(to: Float16.self)
            for kvHead in 0..<g.kvHeads {
                for row in first..<first + 8 {
                    let base = (kvHead * (attnQrowCap + 8) + row) * perRow
                    for index in base..<min(base + perRow, attnQueries.length / 2) { pointer[index] = Float16(value) }
                }
            }
        }
    }

    /// Debugging: the first floats of the mixer projection buffer.
    public func debugProjection(count: Int) -> [Float] {
        Array(UnsafeBufferPointer(start: proj.contents().assumingMemoryBound(to: Float.self), count: count))
    }

    /// Debugging: the first floats of the attention partials buffer.
    public func debugPartials(count: Int) -> [Float] {
        guard let attnPartials else { return [] }
        return Array(UnsafeBufferPointer(start: attnPartials.contents().assumingMemoryBound(to: Float.self), count: count))
    }

    /// Debugging: the emitted accelerator operand for `core`: bf16 values (as floats) and sums.
    public func debugCoreOperand(rows: Int) -> (values: [Float], sums: [Float]) {
        let raw = UnsafeBufferPointer(start: coreB.contents().assumingMemoryBound(to: UInt16.self), count: rows * g.gdnValueDim)
        let sums = UnsafeBufferPointer(start: coreSums.contents().assumingMemoryBound(to: Float.self), count: rows * g.gdnValueDim / 64)
        return (raw.map { Float(bitPattern: UInt32($0) << 16) }, Array(sums))
    }

    /// Captured hidden-state taps for step row `row`: `featureLayers.count * hidden` floats.
    public func featureRow(_ row: Int) -> UnsafeBufferPointer<Float> {
        let width = Self.featureLayers.count * g.hidden
        return UnsafeBufferPointer(start: features.contents().assumingMemoryBound(to: Float.self).advanced(by: row * width), count: width)
    }

    @discardableResult
    public func step(_ rows: [EngineRow], captureFeatures: Bool = false) throws -> EngineStepStats {
        let count = rows.count
        guard count > 0, count <= config.maxRows else {
            throw EngineError.invalidRows("\(count) rows; limit \(config.maxRows)")
        }
        let started = DispatchTime.now().uptimeNanoseconds
        let tokenPtr = tokens.contents().assumingMemoryBound(to: UInt32.self)
        let slotPtr = rowSlot.contents().assumingMemoryBound(to: UInt32.self)
        let posPtr = rowPos.contents().assumingMemoryBound(to: UInt32.self)
        let runPtr = runs.contents().assumingMemoryBound(to: UInt32.self)
        let gatherPtr = gatherIndex.contents().assumingMemoryBound(to: UInt32.self)
        let readPtr = rowRead.contents().assumingMemoryBound(to: UInt32.self)
        let writePtr = rowWrite.contents().assumingMemoryBound(to: UInt32.self)
        let infoPtr = runInfo.contents().assumingMemoryBound(to: UInt32.self)
        for row in rows where row.verify && row.slot >= 0 && row.slot < config.maxSlots { discardVerify(row.slot) }

        var runCount = 0, logitCount = 0
        stepPages = (rows.reduce(0) { max($0, $1.position) }) / Self.pageTokens + 1
        if acceleratedAttention {
            // Attention blocks: up to `attnBlockRows` consecutive rows of one slot.
            let blockPtr = attnBlocks.contents().assumingMemoryBound(to: UInt32.self)
            let rowQPtr = attnRowQ.contents().assumingMemoryBound(to: UInt32.self)
            attnBlockCount = 0
            var index = 0
            while index < count {
                var length = 1
                while length < attnBlockRows, index + length < count, rows[index + length].slot == rows[index].slot { length += 1 }
                blockPtr[attnBlockCount * 2] = UInt32(index)
                blockPtr[attnBlockCount * 2 + 1] = UInt32(length)
                // Dense attention pads every block to eight query rows; the scan indexes by row.
                for offset in 0..<length { rowQPtr[index + offset] = UInt32(pAttnScores != nil ? attnBlockCount * 8 + offset : index + offset) }
                attnBlockCount += 1
                index += length
            }
        }
        var seen = Set<Int>()
        for (index, row) in rows.enumerated() {
            guard row.slot >= 0, row.slot < config.maxSlots, row.token >= 0, row.token < g.vocab else {
                throw EngineError.invalidRows("row \(index) out of range")
            }
            try ensurePage(slot: row.slot, position: row.position)
            tokenPtr[index] = UInt32(row.token)
            slotPtr[index] = UInt32(row.slot)
            posPtr[index] = UInt32(row.position)
            // State units: in place for ordinary rows, a fresh unit per speculative row.
            let previous = pendingVerify[row.slot].last ?? slotUnit[row.slot]
            readPtr[index] = UInt32(previous)
            if row.verify {
                guard let unit = freeUnits.popLast() else { throw EngineError.invalidRows("state unit pool exhausted") }
                pendingVerify[row.slot].append(unit)
                writePtr[index] = UInt32(unit)
            } else {
                guard pendingVerify[row.slot].isEmpty else { throw EngineError.invalidRows("ordinary row after speculative rows in slot \(row.slot)") }
                writePtr[index] = UInt32(previous)
            }
            if index > 0, rows[index - 1].slot == row.slot {
                guard row.position == rows[index - 1].position + 1 else {
                    throw EngineError.invalidRows("slot \(row.slot) positions are not consecutive")
                }
                guard row.verify == rows[index - 1].verify else {
                    throw EngineError.invalidRows("slot \(row.slot) mixes speculative and ordinary rows")
                }
                runPtr[(runCount - 1) * 3 + 2] += 1
                guard !row.verify || runPtr[(runCount - 1) * 3 + 2] <= UInt32(Self.journalRows) else {
                    throw EngineError.invalidRows("more than \(Self.journalRows) speculative rows in slot \(row.slot)")
                }
            } else {
                guard seen.insert(row.slot).inserted else {
                    throw EngineError.invalidRows("slot \(row.slot) rows are not contiguous")
                }
                runPtr[runCount * 3] = UInt32(row.slot)
                runPtr[runCount * 3 + 1] = UInt32(index)
                runPtr[runCount * 3 + 2] = 1
                infoPtr[runCount * 2] = row.verify ? 1 : 0
                infoPtr[runCount * 2 + 1] = UInt32(journalPending[row.slot])
                runCount += 1
            }
            if row.wantLogits {
                guard logitCount < config.maxLogitRows else { throw EngineError.invalidRows("too many logit rows") }
                gatherPtr[logitCount] = UInt32(index)
                logitCount += 1
            }
        }

        if let debugPadding { fillPadding(from: count, value: debugPadding) }
        // Concurrent encoder: dispatches overlap unless separated by `stage()`.
        try graph.begin(label: "splosh.step", concurrent: true)
        if dumpDispatches == count { graph.trace = [] }
        naSource = nil
        ready.removeAll(keepingCapacity: true)
        // From three rows up the GEMMs run on the accelerator and take bf16 operands, so the
        // kernels that produce GEMM inputs emit those operands directly.
        let wide = count >= splitMinRows && na != nil
        encodeEmbed(rows: count)
        stage()
        var linearIndex = 0, fullIndex = 0, blockIndex = 0
        // The norm weight the current (hB, hSums) operand was emitted for, if any.
        var deferred: String?
        // Deferred norms pay for narrow steps, where a stage is a large share of the time. A
        // whole-tile step is faster with the small GEMM kernels and an ordinary norm stage
        // (265 ms against 281 for 128 rows): the emitting kernel is the larger one.
        let canDefer = deferNorms && wide && config.kvFormat == .int8 && pAttnScanNa != nil
            && (count <= 16 || deferWide || pGemmWhole == nil || count % 32 != 0)
        // The last layer's output is read only for rows that want logits (nothing taps it for
        // the draft). For the rest, a prompt being evaluated, the layer has only to store its
        // keys and values: its attention, output projection and MLP are skipped. Rows that
        // want logits come first in a step, so the rows the layer is computed for are a
        // leading run (not fewer than the accelerator kernels take).
        var liveRows = count
        if trimLastLayer, wide, count >= 8, config.kvFormat == .int8, pAttnScanNa != nil, debugMaxBlocks == nil,
           !Self.featureLayers.contains(weights.blocks.count - 1) {
            let wanted = (rows.lastIndex(where: { $0.wantLogits }) ?? -1) + 1
            liveRows = wanted == 0 ? 0 : min(count, max(wanted, 8))
        }
        currentInv = rowOnes
        for block in weights.blocks {
            if let debugMaxBlocks, blockIndex >= debugMaxBlocks { break }
            // Encoding a step takes a few milliseconds. Submitting the first block, then the
            // next dozen, lets the GPU work through them while the remainder is encoded.
            if blockIndex == 1 || blockIndex == 12 { try graph.flush() }
            // A norm whose operand the previous residual GEMM already emitted needs no stage:
            // only its per-row scale is computed, alongside the GEMMs that consume the operand,
            // and the consumers of their outputs apply it.
            func normalise(_ weight: TensorHandle, rows count: Int) -> MTLBuffer {
                if deferred == weight.name, ready[ObjectIdentifier(h)] != nil {
                    rowInverse(rows: count)
                    currentInv = rowInv
                    deferred = nil
                    return h
                }
                deferred = nil
                if attnSkipMask & 64 == 0 { norm(h, weight, xn, count: count, dim: g.hidden, mode: 1, emit: wide) }
                if attnSkipMask & 16384 == 0 { stage() }
                currentInv = rowOnes
                return xn
            }
            let lastBlock = blockIndex + 1 == (debugMaxBlocks ?? weights.blocks.count) || blockIndex + 1 == weights.blocks.count
            // Rows this block's output is computed for: all, but for the model's last block.
            let live = blockIndex + 1 == weights.blocks.count ? liveRows : count
            var source = normalise(block.inputNorm, rows: count)
            var emitted = false
            if wide { ready[ObjectIdentifier(core)] = (coreB, coreSums) }
            switch block.mixer {
            case .linear(let m):
                emitted = encodeLinear(m, layer: linearIndex, rows: count, runs: runCount, source: source, emit: canDefer ? block.postNorm : nil)
                linearIndex += 1
            case .full(let m):
                emitted = encodeFull(m, layer: fullIndex, rows: count, live: live, source: source, emit: canDefer ? block.postNorm : nil)
                fullIndex += 1
            }
            if emitted { deferred = block.postNorm.name }
            stage()
            if live > 0 {
                source = normalise(block.postNorm, rows: live)
                mlpIn(block, rows: live, source: source)
                stage()
                let next = lastBlock ? nil : weights.blocks[blockIndex + 1].inputNorm
                if gemm(block.down, input: gate, output: h, residual: true, rows: live, emit: canDefer ? next : nil), let next {
                    deferred = next.name
                }
            }
            // Timing probe: ordinary-arithmetic work alongside the accelerator GEMM ("with"), or
            // in a stage of its own ("after"), to see whether the two kinds of work overlap.
            if overlapProbe == "with" { aluProbe() }
            stage()
            if overlapProbe == "after" { aluProbe(); stage() }
            currentInv = rowOnes
            // Timing probe: extra dependent stages per block, each a one-row norm into scratch.
            for _ in 0..<extraStages {
                norm(hs, weights.finalNorm, hsn, count: 1, dim: g.hidden, mode: 1)
                stage()
            }
            if captureFeatures, let tap = Self.featureLayers.firstIndex(of: blockIndex) {
                // Reads h only, so it shares a stage with the next block's input norm.
                var cp = CopyParams(rows: UInt32(count), dim: UInt32(g.hidden),
                                    dstStride: UInt32(Self.featureLayers.count * g.hidden), dstOffset: UInt32(tap * g.hidden))
                graph.dispatchThreads(pCopyRows, threads: MTLSize(width: g.hidden, height: count, depth: 1)) { e in
                    e.setBuffer(h, offset: 0, index: 0)
                    e.setBuffer(features, offset: 0, index: 1)
                    e.setBytes(&cp, length: MemoryLayout<CopyParams>.stride, index: 2)
                }
            }
            blockIndex += 1
        }
        if logitCount > 0 {
            var p = GatherParams(count: UInt32(logitCount), dim: UInt32(g.hidden))
            graph.dispatchThreads(pGather, threads: MTLSize(width: g.hidden, height: logitCount, depth: 1)) { e in
                e.setBuffer(h, offset: 0, index: 0)
                e.setBuffer(gatherIndex, offset: 0, index: 1)
                e.setBuffer(hs, offset: 0, index: 2)
                e.setBytes(&p, length: MemoryLayout<GatherParams>.stride, index: 3)
            }
            stage()
            norm(hs, weights.finalNorm, hsn, count: logitCount, dim: g.hidden, mode: 1, emit: logitCount >= splitMinRows && na != nil)
            stage()
            gemm(weights.lmHead, input: hsn, output: logits, residual: false, rows: logitCount)
        }
        let dispatches = graph.dispatchCount
        if let trace = graph.trace {
            // SPLOSH_DUMP_DISPATCHES=<rows>: the first step of that many rows, stage by stage.
            graph.trace = nil
            dumpDispatches = nil
            writeStandardError(Data(("dispatches:\n" + trace.joined(separator: "\n") + "\n").utf8))
        }
        let gpu = try graph.commitAndWait()
        if stepTiming {
            // Where a step's wall time goes besides GPU work: the wait before the first
            // command buffer starts, gaps between buffers, and the return after the last.
            let pass = graph.lastPass
            let now = ProcessInfo.processInfo.systemUptime
            var line = String(format: "step timing: %d rows; prepared %.1f ms before the pass", count, (pass.began - Double(started) / 1e9) * 1000)
            var previousEnd = pass.began
            for buffer in pass.buffers {
                line += String(format: "; committed +%.1f, gpu +%.1f..+%.1f (idle %.1f)", (buffer.committed - pass.began) * 1000,
                               (buffer.gpuStart - pass.began) * 1000, (buffer.gpuEnd - pass.began) * 1000, (buffer.gpuStart - previousEnd) * 1000)
                previousEnd = buffer.gpuEnd
            }
            line += String(format: "; returned +%.1f\n", (now - pass.began) * 1000)
            writeStandardError(Data(line.utf8))
        }
        // Every slot in the step had its accepted journal entries replayed into its state.
        for index in 0..<runCount { journalPending[Int(runPtr[index * 3])] = 0 }
        let wall = Double(DispatchTime.now().uptimeNanoseconds - started) / 1e9
        return EngineStepStats(rows: count, dispatches: dispatches, gpuSeconds: gpu, wallSeconds: wall)
    }

    // MARK: - Encoding

    private struct EmbedParams { var rows, strideWords, groupsPerRow, hidden: UInt32 }
    private struct GemmParams { var rows, outDim, inner, strideWords, groupsPerRow, hasResidual, inStride, outStride: UInt32 }
    private struct NormParams { var count, dim, mode: UInt32; var eps: Float }
    private struct CountParams { var count: UInt32 }
    private struct CopyParams { var rows, dim, dstStride, dstOffset: UInt32 }
    private struct NaPrepareParams { var rows, groups, inner: UInt32 }
    private struct NaParams { var rows, outDim, inner, groups, hasResidual, outStride: UInt32 }
    private struct GatherParams { var count, dim: UInt32 }
    private struct AttnPrepParams { var rows, heads, rotary, inHeadStride, maxPages: UInt32; var eps, theta: Float }
    private struct NaQPrepParams { var rows, heads, kvHeads, rotary, rowCap: UInt32; var eps, theta: Float }
    private struct NaKVParams { var rows, kvHeads, rotary, maxPages: UInt32; var eps, theta: Float }
    private struct NaScanParams { var blocks, heads, kvHeads, maxPages, spans, pagesPerSpan, rowCap: UInt32; var scale: Float; var aliasTokens: UInt32 = 0; var blocksAcross: UInt32 = 0 }
    private struct AttnScanParams { var rows, heads, kvHeads, maxPages, spans, pagesPerSpan: UInt32; var scale: Float }
    private struct AttnFusedParams { var rows, heads, kvHeads, maxPages, rotary: UInt32; var scale, eps, theta: Float }
    private struct GdnParams { var rows, runs, keyHeads, valueHeads, headDim, channels: UInt32; var eps: Float; var debug: UInt32 = 0 }

    /// End a stage: everything encoded so far completes before anything encoded after.
    func stage() { graph.barrier(); naSource = nil }

    /// Open a concurrent pass on the engine's queue for work outside `step` (the draft model).
    func beginPass(label: String) throws {
        try graph.begin(label: label, concurrent: true)
        naSource = nil
        ready.removeAll(keepingCapacity: true)
        currentInv = rowOnes
    }

    var groupSize: MTLSize { MTLSize(width: Self.lanes, height: 1, depth: 1) }

    func bind(_ e: MTLComputeCommandEncoder, _ handle: TensorHandle, _ index: Int) {
        e.setBuffer(spans[handle.span], offset: handle.offset, index: index)
    }

    /// Make an extra buffer addressable through `TensorHandle.span` (used by the draft model,
    /// whose weights are not part of the resident artifact).
    /// A q4 weight from outside the artifact (the draft model's) whose own buffers are in the
    /// tiled layout.
    func registerTiled(_ w: Q4Handle) {
        tiled[w.packed.name] = (w.packed, w.scales, w.biases)
        tiledInPlace.insert(w.packed.name)
    }
    private var tiledInPlace: Set<String> = []
    private var pGemmWhole: MTLComputePipelineState?, pGemmWholeResidual: MTLComputePipelineState?
    private let deferWide = ProcessInfo.processInfo.environment["SPLOSH_DEFER_WIDE"] == "1"
    /// Everything a step reads or writes except the KV pool, kept resident on the GPU. Without
    /// it the driver makes a step's resources resident when the step is committed, and after
    /// the process has idled that takes long enough to notice: the first 128-row step after
    /// 25 s of idling waited 1.5-1.8 s for the GPU to start, a narrow one 0.27 s.
    private var residency: MTLResidencySet?

    /// Keep buffers created outside the engine (the draft model's) resident as well.
    func keepResident(_ buffers: [MTLBuffer]) {
        guard let residency, !buffers.isEmpty else { return }
        residency.addAllocations(buffers)
        residency.commit()
        residency.requestResidency()
    }
    private let stepTiming = ProcessInfo.processInfo.environment["SPLOSH_STEP_TIMING"] != nil
    private var dumpDispatches = Int(ProcessInfo.processInfo.environment["SPLOSH_DUMP_DISPATCHES"] ?? "")
    /// SPLOSH_FULL_LAST_LAYER=1 computes the last layer for every row, as before, for comparison.
    private let trimLastLayer = ProcessInfo.processInfo.environment["SPLOSH_FULL_LAST_LAYER"] == nil

    func registerSpan(_ buffer: MTLBuffer) -> Int {
        spans.append(buffer)
        return spans.count - 1
    }

    private func encodeEmbed(rows: Int) { embed(tokens: tokens, output: h, rows: rows) }

    func embed(tokens: MTLBuffer, output: MTLBuffer, rows: Int) {
        let w = weights.embed
        var p = EmbedParams(rows: UInt32(rows), strideWords: UInt32(w.rowStrideWords),
                            groupsPerRow: UInt32(w.groupsPerRow), hidden: UInt32(g.hidden))
        // One thread a packed word of a row. For a GGUF embedding `pEmbed` is sp_gguf_embed: one
        // thread a group of 32, which is what its `groupsPerRow` counts, and no stride is read.
        graph.dispatchThreads(pEmbed, threads: MTLSize(width: w.kernel == nil ? w.rowStrideWords : w.groupsPerRow, height: rows, depth: 1)) { e in
            bind(e, w.packed, 0); bind(e, w.scales, 1); bind(e, w.biases, 2)
            e.setBuffer(tokens, offset: 0, index: 3)
            e.setBuffer(output, offset: 0, index: 4)
            e.setBytes(&p, length: MemoryLayout<EmbedParams>.stride, index: 5)
        }
    }

    /// The accelerator operand for a GEMM input: its bf16 copy and per-64 sums. Producers that
    /// emit these register them in `ready`; otherwise a conversion stage runs here.
    private func operand(for input: MTLBuffer, w: Q4Handle, rows: Int) -> (values: MTLBuffer, sums: MTLBuffer) {
        if let emitted = ready[ObjectIdentifier(input)] { return emitted }
        if naSource !== input {
            // The operand's sums are per 64 columns whatever the weight's groups are. An MLX
            // handle's `groupsPerRow` is this count; a GGUF handle's counts groups of 32.
            let groups = w.inner / 64
            var pp = NaPrepareParams(rows: UInt32(rows), groups: UInt32(groups), inner: UInt32(w.inner))
            graph.dispatch(pNaPrepare, grid: MTLSize(width: groups, height: rows, depth: 1), threadsPerGroup: groupSize) { e in
                e.setBuffer(input, offset: 0, index: 0)
                e.setBuffer(naInput, offset: 0, index: 1)
                e.setBuffer(naSums, offset: 0, index: 2)
                e.setBytes(&pp, length: MemoryLayout<NaPrepareParams>.stride, index: 3)
            }
            graph.barrier()
            naSource = input
        }
        return (naInput, naSums)
    }

    /// `emit`: the weight of the RMSNorm that follows this (residual) projection. When the
    /// selected kernel can, it also writes the next GEMM's operand for that norm and the sums of
    /// squares its scale needs, and the result is true: no normalisation stage is then required.
    @discardableResult
    func gemm(_ w: Q4Handle, input: MTLBuffer, output: MTLBuffer, residual: Bool, rows: Int, emit: TensorHandle? = nil) -> Bool {
        var p = GemmParams(rows: UInt32(rows), outDim: UInt32(w.rows), inner: UInt32(w.inner),
                           strideWords: UInt32(w.rowStrideWords), groupsPerRow: UInt32(w.groupsPerRow),
                           hasResidual: residual ? 1 : 0, inStride: UInt32(w.inner), outStride: UInt32(w.rows))
        if skip.contains("gemm") { return false }
        if output === h { ready[ObjectIdentifier(h)] = nil }
        func bindEmit(_ e: MTLComputeCommandEncoder, _ weight: TensorHandle) {
            bind(e, weight, 8)
            e.setBuffer(hB, offset: 0, index: 9)
            e.setBuffer(hSums, offset: 0, index: 10)
            e.setBuffer(hSquares, offset: 0, index: 11)
        }
        var np = NaParams(rows: UInt32(rows), outDim: UInt32(w.rows), inner: UInt32(w.inner),
                          groups: UInt32(w.groupsPerRow), hasResidual: residual ? 1 : 0, outStride: UInt32(w.rows))
        // A GGUF weight (engine_gguf.metal). Its planes are bound where an affine weight's codes,
        // scales and biases are, and `np` is its parameter block, a GGUF handle's `groupsPerRow`
        // being its groups of 32. The kernels read the bf16 operand and no sums, and none emits
        // the next norm's operand, so the result is false and the caller normalises explicitly.
        // A step of up to 16 rows is one split-K tile of 16 rows, one of up to 32 a tile of 32,
        // and a wider one goes to the wide kernel; a weight that is not whole 32-row tiles (the
        // 48-row projections of a gated-delta layer) goes to the small GEMM at any step.
        if let format = w.kernel {
            let source = operand(for: input, w: w, rows: rows)
            // (kernel, rows and outputs a threadgroup takes, its threads)
            let shape: (pipeline: MTLComputePipelineState?, tileRows: Int, columns: Int, threads: Int)
            if w.rows % 32 != 0 {
                shape = (ggufSmall[format], 1, 1, 32)
            } else {
                let tiles = (ggufHeadOrder.contains(w.packed.name) ? ggufHeadTiles : ggufTiles)[format]
                shape = rows <= 16 ? (tiles?.split16, 16, 32, 64) : rows <= 32 ? (tiles?.split32, 32, 32, 64) : (tiles?.wide, 128, 64, 128)
            }
            guard let pipeline = shape.pipeline else {
                preconditionFailure("no GGUF kernel was compiled for \(w.packed.name) (\(format))")
            }
            graph.dispatch(pipeline, grid: MTLSize(width: w.rows / shape.columns, height: (rows + shape.tileRows - 1) / shape.tileRows, depth: 1),
                           threadsPerGroup: MTLSize(width: shape.threads, height: 1, depth: 1)) { e in
                bind(e, w.packed, 0); bind(e, w.scales, 1); bind(e, w.biases, 2)
                e.setBuffer(source.values, offset: 0, index: 3)
                e.setBuffer(output, offset: 0, index: 4)
                e.setBuffer(output, offset: 0, index: 5)
                e.setBytes(&np, length: MemoryLayout<NaParams>.stride, index: 7)
            }
            return false
        }
        let tiles = tiled[w.packed.name]
        func bindWeights(_ e: MTLComputeCommandEncoder, tiledLayout: Bool) {
            if tiledLayout, let tiles {
                bind(e, tiles.packed, 0); bind(e, tiles.scales, 1); bind(e, tiles.biases, 2)
            } else {
                bind(e, w.packed, 0); bind(e, w.scales, 1); bind(e, w.biases, 2)
            }
        }
        // Narrow passes: split-K accelerator kernel.
        if let split, rows >= splitMinRows, rows <= 8, w.rows % 128 == 0, w.groupsPerRow % split.partitions == 0 {
            let source = operand(for: input, w: w, rows: rows)
            let useTiles = splitTiled != nil && tiles != nil
            graph.dispatch(useTiles ? splitTiled! : split.pipeline,
                           grid: MTLSize(width: w.rows / (useTiles ? splitColumns : 32), height: (rows + 7) / 8, depth: 1),
                           threadsPerGroup: MTLSize(width: split.partitions * 32, height: 1, depth: 1)) { e in
                bindWeights(e, tiledLayout: useTiles)
                e.setBuffer(source.values, offset: 0, index: 3)
                e.setBuffer(output, offset: 0, index: 4)
                e.setBuffer(output, offset: 0, index: 5)
                e.setBuffer(source.sums, offset: 0, index: 6)
                e.setBytes(&np, length: MemoryLayout<NaParams>.stride, index: 7)
            }
            return false
        }
        // Wider passes on tiled weights: the same split-K scheme with taller tiles.
        // Measured: split-K wins up to two 32-row tiles; beyond that the plain tile kernel has
        // enough threadgroups to fill the GPU on its own and is faster. Whole 32-row tiles past
        // the first go to the small whole-tile kernels, which also defer the norms: a 64-row
        // step (four sessions' double blocks) is 3 ms faster there, while 40 and 48 rows are
        // 9-10 ms slower than with split-K.
        if rows > (split == nil ? matrixRows - 1 : 8), rows <= splitMaxRows, !Self.wholeTiles(rows), tiles != nil, w.rows % 128 == 0,
           let wide = splitWide.first(where: { $0.m >= min(rows, splitWide.last!.m) }), w.groupsPerRow % wide.partitions == 0 {
            let source = operand(for: input, w: w, rows: rows)
            let emitting = emit != nil && output === h && w.rows == g.hidden && wide.m == 16 && wide.columns == 64 && wide.partitions == 4 && emitProbe == nil
            graph.dispatch(emitting ? pGemmEmitSplit : wide.pipeline,
                           grid: MTLSize(width: w.rows / wide.columns, height: (rows + wide.m - 1) / wide.m, depth: 1),
                           threadsPerGroup: MTLSize(width: wide.partitions * 32, height: 1, depth: 1)) { e in
                bindWeights(e, tiledLayout: true)
                e.setBuffer(source.values, offset: 0, index: 3)
                e.setBuffer(output, offset: 0, index: 4)
                e.setBuffer(output, offset: 0, index: 5)
                e.setBuffer(source.sums, offset: 0, index: 6)
                e.setBytes(&np, length: MemoryLayout<NaParams>.stride, index: 7)
                if emitting { bindEmit(e, emit!) }
            }
            if emitting { ready[ObjectIdentifier(h)] = (hB, hSums) }
            return emitting
        }
        // Wide passes: one accelerator tile per threadgroup.
        if let na, rows >= matrixRows, w.rows % na.n == 0, w.inner % 64 == 0 {
            let source = operand(for: input, w: w, rows: rows)
            let useTiles = naTiled != nil && tiles != nil
            let rowTiles = (rows + na.m - 1) / na.m
            let emitting = emit != nil && output === h && w.rows == g.hidden && useTiles && na.m == 32 && na.n == 128 && na.threads == 128 && emitProbe == nil
            // Tiled kernels take row tiles in blocks of four (see SP_NA_ROW_BLOCK).
            // Whole 32-row tiles and the shipped tile shape: the small kernels (see sp_na_wide).
            let whole = useTiles && rows % na.m == 0 ? (residual ? pGemmWholeResidual : pGemmWhole) : nil
            graph.dispatch(emitting ? pGemmEmitWide : whole ?? (useTiles ? naTiled! : na.pipeline),
                           grid: useTiles ? MTLSize(width: min(rowTiles, 4), height: w.rows / na.n, depth: (rowTiles + 3) / 4)
                                          : MTLSize(width: w.rows / na.n, height: rowTiles, depth: 1),
                           threadsPerGroup: MTLSize(width: na.threads, height: 1, depth: 1)) { e in
                bindWeights(e, tiledLayout: useTiles)
                e.setBuffer(source.values, offset: 0, index: 3)
                e.setBuffer(output, offset: 0, index: 4)
                e.setBuffer(output, offset: 0, index: 5)
                e.setBuffer(source.sums, offset: 0, index: 6)
                e.setBytes(&np, length: MemoryLayout<NaParams>.stride, index: 7)
                if emitting { bindEmit(e, emit!) }
            }
            if emitting { ready[ObjectIdentifier(h)] = (hB, hSums) }
            return emitting
        }
        // One or two rows on tiled weights.
        if rows <= 2, let tiles, let lane = tiledLane[rows], w.groupsPerRow % tiledLanePartitions == 0 {
            graph.dispatch(lane.gemm, grid: MTLSize(width: w.rows / 32, height: 1, depth: 1),
                           threadsPerGroup: MTLSize(width: tiledLanePartitions * 32, height: 1, depth: 1)) { e in
                bind(e, tiles.packed, 0); bind(e, tiles.scales, 1); bind(e, tiles.biases, 2)
                e.setBuffer(input, offset: 0, index: 3)
                e.setBuffer(output, offset: 0, index: 4)
                e.setBuffer(output, offset: 0, index: 5)
                e.setBytes(&p, length: MemoryLayout<GemmParams>.stride, index: 6)
            }
            return false
        }
        // A small row-major weight whose input exists only as a deferred-norm operand.
        if input === h, let emitted = ready[ObjectIdentifier(h)] {
            graph.dispatch(pGemmBf16, grid: MTLSize(width: w.rows, height: rows, depth: 1), threadsPerGroup: groupSize) { e in
                bind(e, w.packed, 0); bind(e, w.scales, 1); bind(e, w.biases, 2)
                e.setBuffer(emitted.values, offset: 0, index: 3)
                e.setBuffer(output, offset: 0, index: 4)
                e.setBytes(&p, length: MemoryLayout<GemmParams>.stride, index: 6)
            }
            return false
        }
        // Row-major weights (untiled tensors, or no tiles at all): lane kernels.
        precondition(!(weights.tiledLayout && tiles != nil) && !tiledInPlace.contains(w.packed.name),
                     "row-major kernel selected for tiled weight \(w.packed.name)")
        var pipeline = fixedWidth ? gemmByWidth[Self.passWidth(rows)]! : pGemm
        var width = fixedWidth ? Self.passWidth(rows) : Self.rowBlock
        var columns = 1
        if let tile, rows >= 5, w.rows % tile.columns == 0 { pipeline = tile.gemm; width = tile.rows; columns = tile.columns }
        let blocks = (rows + width - 1) / width
        graph.dispatch(pipeline, grid: MTLSize(width: w.rows / columns, height: blocks, depth: 1), threadsPerGroup: groupSize) { e in
            bind(e, w.packed, 0); bind(e, w.scales, 1); bind(e, w.biases, 2)
            e.setBuffer(input, offset: 0, index: 3)
            e.setBuffer(output, offset: 0, index: 4)
            e.setBuffer(output, offset: 0, index: 5)
            e.setBytes(&p, length: MemoryLayout<GemmParams>.stride, index: 6)
        }
        return false
    }

    /// The deferred norm's per-row scale, from the sums of squares the emitting GEMM wrote. It
    /// shares a stage with the GEMMs that consume the emitted operand.
    private func rowInverse(rows: Int) {
        var p = (UInt32(rows), UInt32(g.hidden / 64), UInt32(g.hidden), g.normEps)
        graph.dispatch(pRowInv, grid: MTLSize(width: rows, height: 1, depth: 1), threadsPerGroup: groupSize) { e in
            e.setBuffer(hSquares, offset: 0, index: 0)
            e.setBuffer(rowInv, offset: 0, index: 1)
            e.setBytes(&p, length: 16, index: 2)
        }
    }

    /// Steps the split-K kernels leave to whole accelerator tiles (see `gemm`). SPLOSH_SPLIT_WHOLE=0
    /// keeps split-K for them.
    private static let splitLeavesWholeTiles = ProcessInfo.processInfo.environment["SPLOSH_SPLIT_WHOLE"] != "0"
    private static func wholeTiles(_ rows: Int) -> Bool { splitLeavesWholeTiles && rows > 32 && rows % 32 == 0 }

    /// Rows evaluated per pass over the weights: the smallest fixed width that covers `rows`,
    /// capped at eight.
    private static func passWidth(_ rows: Int) -> Int { rows >= 5 ? 8 : rows >= 3 ? 4 : rows }

    /// SPLOSH_MLP_FUSED=1...4 (a candidate, off by default): on wide steps the up GEMM forms the
    /// SiLU product in its epilogue and writes the down projection's operand itself, so the SiLU
    /// stage and the up projection's fp32 output go. The value picks the variant: see
    /// MlpFusedPlan and Sources/Shaders/candidates/mlp_fused.metal.
    private let mlpFused = Int(ProcessInfo.processInfo.environment["SPLOSH_MLP_FUSED"] ?? "") ?? 0
    private var mlpFusedPipelines: [String: MTLComputePipelineState] = [:]
    private struct MlpFusedParams { var rows, outDim, inner, groups, firstTile, outStride: UInt32 }

    /// Whether `gemm` runs this weight, at this row count, as 32 x 128 accelerator tiles on four
    /// simdgroups over tiled weights: the conditions of its branches, in its order. This only
    /// decides where the candidate is used (narrower steps keep the split-K kernels); the
    /// candidate dispatches both of its GEMMs itself, so it does not depend on `gemm` agreeing.
    private func runsWideTiles(_ w: Q4Handle, rows: Int) -> Bool {
        let tiles = tiled[w.packed.name]
        if let split, rows >= splitMinRows, rows <= 8, w.rows % 128 == 0, w.groupsPerRow % split.partitions == 0 { return false }
        if rows > (split == nil ? matrixRows - 1 : 8), rows <= splitMaxRows, !Self.wholeTiles(rows), tiles != nil, w.rows % 128 == 0,
           let wide = splitWide.first(where: { $0.m >= min(rows, splitWide.last!.m) }), w.groupsPerRow % wide.partitions == 0 { return false }
        guard let na, rows >= matrixRows, w.rows % na.n == 0, w.inner % 64 == 0 else { return false }
        return naTiled != nil && tiles != nil && na.m == 32 && na.n == 128 && na.threads == 128
    }

    /// The candidate form of the wide gate/up/SiLU stage; false when it does not apply.
    /// The gate GEMM runs first: the shipped wide kernel into `output` as fp32, or, in the
    /// bf16-gate variants, a kernel that stores it as bf16 into `scratch`. After a barrier,
    /// since the up GEMM's epilogue reads the gate, the up GEMM runs with the product in its
    /// epilogue: whole 32-row tiles by one kernel, a partial last tile by another.
    private func mlpInFused(gate gateWeights: Q4Handle, up upWeights: Q4Handle, input: MTLBuffer, output: MTLBuffer,
                            scratch: MTLBuffer, rows: Int) -> Bool {
        // The candidate's kernels read affine q4 codes: not for a GGUF weight.
        guard !skip.contains("silu"), gateWeights.kernel == nil, upWeights.kernel == nil,
              gateWeights.rows == upWeights.rows, gateWeights.inner == upWeights.inner,
              gateWeights.groupsPerRow == upWeights.groupsPerRow,
              runsWideTiles(gateWeights, rows: rows), runsWideTiles(upWeights, rows: rows),
              let gateTiles = tiled[gateWeights.packed.name], let upTiles = tiled[upWeights.packed.name],
              let naTiled, let plan = MlpFusedPlan(variant: mlpFused, rows: rows, outDim: upWeights.rows),
              plan.up.allSatisfy({ mlpFusedPipelines[$0.kernel] != nil }),
              plan.gate.map({ mlpFusedPipelines[$0.kernel] != nil }) ?? true else { return false }
        let w = upWeights
        let source = operand(for: input, w: w, rows: rows)
        func dispatch(_ pipeline: MTLComputePipelineState, _ tiles: (packed: TensorHandle, scales: TensorHandle, biases: TensorHandle),
                      rows: Int, firstTile: Int, grid: MTLSize, gate: MTLBuffer, product: Bool) {
            // The same six words as NaParams, the residual flag's slot being the first tile
            // (zero, and no residual, for the shipped gate kernel).
            var p = MlpFusedParams(rows: UInt32(rows), outDim: UInt32(w.rows), inner: UInt32(w.inner),
                                   groups: UInt32(w.groupsPerRow), firstTile: UInt32(firstTile), outStride: UInt32(w.rows))
            graph.dispatch(pipeline, grid: grid, threadsPerGroup: MlpFusedPlan.threadsPerGroup) { e in
                bind(e, tiles.packed, 0); bind(e, tiles.scales, 1); bind(e, tiles.biases, 2)
                e.setBuffer(source.values, offset: 0, index: 3)
                e.setBuffer(gate, offset: 0, index: 4)
                e.setBuffer(product ? gateB : gate, offset: 0, index: 5)
                e.setBuffer(source.sums, offset: 0, index: 6)
                e.setBytes(&p, length: MemoryLayout<MlpFusedParams>.stride, index: 7)
                if product {
                    e.setBuffer(gateSums, offset: 0, index: 8)
                    e.setBuffer(currentInv, offset: 0, index: 9)
                }
            }
        }
        let gateValues: MTLBuffer
        if let gate = plan.gate {
            gateValues = scratch
            dispatch(mlpFusedPipelines[gate.kernel]!, gateTiles, rows: gate.rows, firstTile: gate.firstTile, grid: gate.grid,
                     gate: scratch, product: false)
        } else {
            gateValues = output
            dispatch((rows % 32 == 0 ? pGemmWhole : nil) ?? naTiled, gateTiles, rows: rows, firstTile: 0,
                     grid: MlpFusedPlan.grid(rowTiles: (rows + 31) / 32, outDim: w.rows), gate: output, product: false)
        }
        stage()
        for up in plan.up {
            dispatch(mlpFusedPipelines[up.kernel]!, upTiles, rows: up.rows, firstTile: up.firstTile, grid: up.grid,
                     gate: gateValues, product: true)
        }
        ready[ObjectIdentifier(output)] = (gateB, gateSums)
        return true
    }

    /// act = silu(gate_proj(x)) * up_proj(x), written to `gate`.
    private func mlpIn(_ block: BlockWeights, rows: Int, source: MTLBuffer) {
        mlpIn(gate: block.gate, up: block.up, input: source, output: gate, scratch: up, rows: rows)
    }

    /// output = silu(gate(input)) * up(input). `scratch` is used only by the multi-row path.
    func mlpIn(gate gateWeights: Q4Handle, up upWeights: Q4Handle, input: MTLBuffer, output: MTLBuffer, scratch: MTLBuffer, rows: Int) {
        let w = gateWeights
        var p = GemmParams(rows: UInt32(rows), outDim: UInt32(w.rows), inner: UInt32(w.inner),
                           strideWords: UInt32(w.rowStrideWords), groupsPerRow: UInt32(w.groupsPerRow),
                           hasResidual: 0, inStride: UInt32(w.inner), outStride: UInt32(w.rows))
        if skip.contains("gemm") { return }
        ready[ObjectIdentifier(output)] = nil
        // A GGUF pair takes the two-GEMM form at any row count: the lane kernels below, which
        // fuse the pair for one or two rows, read affine q4 codes.
        if rows >= matrixRows || gateWeights.kernel != nil || upWeights.kernel != nil {
            if mlpFused != 0, na != nil, rows >= splitMinRows, rows <= config.maxRows, w.rows == g.intermediate,
               mlpInFused(gate: gateWeights, up: upWeights, input: input, output: output, scratch: scratch, rows: rows) { return }
            gemm(gateWeights, input: input, output: output, residual: false, rows: rows)
            gemm(upWeights, input: input, output: scratch, residual: false, rows: rows)
            stage()
            if na != nil, rows >= splitMinRows, rows <= config.maxRows, w.rows == g.intermediate {
                // Only the accelerator operand is produced; the down-projection is its consumer.
                var sp = (UInt32(rows), UInt32(w.rows))
                if !skip.contains("silu") { graph.dispatch(pSiluMulNa, grid: MTLSize(width: w.rows / 64, height: rows, depth: 1), threadsPerGroup: groupSize) { e in
                    e.setBuffer(output, offset: 0, index: 0)
                    e.setBuffer(scratch, offset: 0, index: 1)
                    e.setBuffer(gateB, offset: 0, index: 2)
                    e.setBuffer(gateSums, offset: 0, index: 3)
                    e.setBytes(&sp, length: 8, index: 4)
                    e.setBuffer(currentInv, offset: 0, index: 5)
                } }
                ready[ObjectIdentifier(output)] = (gateB, gateSums)
            } else {
                siluMul(gate: output, up: scratch, count: rows * w.rows)
            }
            return
        }
        if rows <= 2, let lane = tiledLane[rows], let gateTiles = tiled[gateWeights.packed.name], let upTiles = tiled[upWeights.packed.name],
           w.groupsPerRow % tiledLanePartitions == 0 {
            graph.dispatch(lane.mlp, grid: MTLSize(width: w.rows / 32, height: 1, depth: 1),
                           threadsPerGroup: MTLSize(width: tiledLanePartitions * 32, height: 1, depth: 1)) { e in
                bind(e, gateTiles.packed, 0); bind(e, gateTiles.scales, 1); bind(e, gateTiles.biases, 2)
                bind(e, upTiles.packed, 3); bind(e, upTiles.scales, 4); bind(e, upTiles.biases, 5)
                e.setBuffer(input, offset: 0, index: 6)
                e.setBuffer(output, offset: 0, index: 7)
                e.setBytes(&p, length: MemoryLayout<GemmParams>.stride, index: 8)
            }
            return
        }
        var pipeline = fixedWidth ? mlpByWidth[Self.passWidth(rows)]! : pMlpIn
        var width = fixedWidth ? Self.passWidth(rows) : Self.rowBlock
        var columns = 1
        if let tile, rows >= 5 { pipeline = tile.mlp; width = tile.rows; columns = tile.columns == 8 ? 4 : tile.columns }
        let blocks = (rows + width - 1) / width
        graph.dispatch(pipeline, grid: MTLSize(width: w.rows / columns, height: blocks, depth: 1), threadsPerGroup: groupSize) { e in
            bind(e, w.packed, 0); bind(e, w.scales, 1); bind(e, w.biases, 2)
            bind(e, upWeights.packed, 3); bind(e, upWeights.scales, 4); bind(e, upWeights.biases, 5)
            e.setBuffer(input, offset: 0, index: 6)
            e.setBuffer(output, offset: 0, index: 7)
            e.setBytes(&p, length: MemoryLayout<GemmParams>.stride, index: 8)
        }
    }

    /// RMSNorm. With `emit`, also writes the accelerator operand for `output` and registers it.
    func norm(_ input: MTLBuffer, _ weight: TensorHandle, _ output: MTLBuffer, count: Int, dim: Int, mode: UInt32, emit: Bool = false) {
        var p = NormParams(count: UInt32(count), dim: UInt32(dim), mode: mode, eps: g.normEps)
        let emitting = emit && dim == g.hidden
        // A wide step spreads each row over the GPU (sp_rmsnorm_wide); a narrow one has rows to spare.
        let spread = emitting && count >= normWideRows && dim % 512 == 0 && dim / 512 <= 32
        graph.dispatch(spread ? pNormWide : emitting ? pNormNa : pNorm, grid: MTLSize(width: count, height: 1, depth: 1),
                       threadsPerGroup: spread ? MTLSize(width: dim / 512 * 32, height: 1, depth: 1) : groupSize) { e in
            e.setBuffer(input, offset: 0, index: 0)
            bind(e, weight, 1)
            e.setBuffer(output, offset: 0, index: 2)
            e.setBytes(&p, length: MemoryLayout<NormParams>.stride, index: 3)
            if emitting {
                e.setBuffer(xnB, offset: 0, index: 4)
                e.setBuffer(xnSums, offset: 0, index: 5)
            }
        }
        if emitting { ready[ObjectIdentifier(output)] = (xnB, xnSums) } else { ready[ObjectIdentifier(output)] = nil }
    }

    /// gate = silu(gate) * up, in place.
    func siluMul(gate: MTLBuffer, up: MTLBuffer, count: Int) {
        var p = CountParams(count: UInt32(count))
        graph.dispatchThreads(pSiluMul, threads: MTLSize(width: count, height: 1, depth: 1)) { e in
            e.setBuffer(gate, offset: 0, index: 0)
            e.setBuffer(up, offset: 0, index: 1)
            e.setBuffer(gate, offset: 0, index: 2)
            e.setBytes(&p, length: MemoryLayout<CountParams>.stride, index: 3)
        }
    }

    /// Returns whether the output projection emitted the operand for `emit` (the following norm).
    private func encodeLinear(_ m: LinearMixerWeights, layer: Int, rows: Int, runs runCount: Int,
                              source: MTLBuffer, emit: TensorHandle?) -> Bool {
        gemm(m.qkv, input: source, output: proj, residual: false, rows: rows)
        gemm(m.z, input: source, output: z, residual: false, rows: rows)
        gemm(m.a, input: source, output: aBuf, residual: false, rows: rows)
        gemm(m.b, input: source, output: bBuf, residual: false, rows: rows)
        stage()
        var p = GdnParams(rows: UInt32(rows), runs: UInt32(runCount), keyHeads: UInt32(g.gdnKeyHeads),
                          valueHeads: UInt32(g.gdnValueHeads), headDim: UInt32(g.gdnHeadDim),
                          channels: UInt32(g.gdnChannels), eps: g.normEps,
                          debug: UInt32(ProcessInfo.processInfo.environment["SPLOSH_GDN_DEBUG"] ?? "") ?? 0)
        if skip.contains("gdn") { return false }
        let bindGdn: (MTLComputeCommandEncoder) -> Void = { [self] e in
            e.setBuffer(proj, offset: 0, index: 0)
            e.setBuffer(z, offset: 0, index: 1)
            e.setBuffer(aBuf, offset: 0, index: 2)
            e.setBuffer(bBuf, offset: 0, index: 3)
            bind(e, m.conv, 4); bind(e, m.aLog, 5); bind(e, m.dtBias, 6); bind(e, m.norm, 7)
            e.setBuffer(convState[layer], offset: 0, index: 8)
            e.setBuffer(gdnState[layer], offset: 0, index: 9)
            e.setBuffer(core, offset: 0, index: 10)
            e.setBuffer(runs, offset: 0, index: 11)
            e.setBytes(&p, length: MemoryLayout<GdnParams>.stride, index: 12)
            e.setBuffer(rowRead, offset: 0, index: 13)
            e.setBuffer(rowWrite, offset: 0, index: 14)
            e.setBuffer(coreB, offset: 0, index: 15)
            e.setBuffer(coreSums, offset: 0, index: 16)
            e.setBuffer(gdnQ, offset: 0, index: 17)
            e.setBuffer(gdnK, offset: 0, index: 18)
            e.setBuffer(gdnV, offset: 0, index: 19)
            e.setBuffer(gdnNorm, offset: 0, index: 20)
            e.setBuffer(gdnJournal[layer], offset: 0, index: 21)
            e.setBuffer(runInfo, offset: 0, index: 22)
            e.setBuffer(currentInv, offset: 0, index: 23)
        }
        if gdnSkipMask & 16 != 0, let probe = try? library.pipeline("sp_probe_matmul"), let out = aluProbeOut {
            // Timing: the synthetic accelerator probe as a stage of its own in a gated-delta
            // layer (mask 16: reading the attention scratch; with 32 as well: reading its own buffer).
            var pp = (UInt32(3), UInt32(31))
            let own = gdnSkipMask & 32 != 0
            graph.dispatch(probe, grid: MTLSize(width: 4, height: 16, depth: 1), threadsPerGroup: MTLSize(width: 256, height: 1, depth: 1)) { e in
                e.setBuffer(own ? features : attnQueries, offset: 0, index: 0); e.setBuffer(own ? features : kPools[0], offset: 0, index: 1)
                e.setBuffer(out, offset: 0, index: 2); e.setBytes(&pp, length: 8, index: 3)
            }
            stage()
        }
        if rows >= gdnSplitRows {
            // Updates accepted from a slot's last speculative run go into its stored state
            // first: the scan's threadgroups cannot replay them between them.
            let runPtr = runs.contents().assumingMemoryBound(to: UInt32.self)
            let infoPtr = runInfo.contents().assumingMemoryBound(to: UInt32.self)
            for run in 0..<runCount where infoPtr[run * 2 + 1] > 0 {
                var cp = (runPtr[run * 3], infoPtr[run * 2 + 1], UInt32(g.gdnValueHeads), UInt32(g.gdnHeadDim))
                graph.dispatch(pGdnCommit, grid: MTLSize(width: g.gdnValueHeads, height: 1, depth: 1),
                               threadsPerGroup: MTLSize(width: 1024, height: 1, depth: 1)) { e in
                    e.setBuffer(gdnState[layer], offset: 0, index: 0)
                    e.setBuffer(gdnJournal[layer], offset: 0, index: 1)
                    e.setBytes(&cp, length: 16, index: 2)
                }
            }
            if gdnSkipMask & 1 == 0 { graph.dispatch(pGdnPrepare, grid: MTLSize(width: g.gdnValueHeads, height: rows, depth: 1),
                           threadsPerGroup: MTLSize(width: 96, height: 1, depth: 1), bindGdn) }
            stage()
            if gdnSkipMask & 2 == 0 { graph.dispatch(pGdnScan, grid: MTLSize(width: g.gdnValueHeads * gdnChoice.chainGroups, height: runCount, depth: 1),
                           threadsPerGroup: MTLSize(width: gdnChoice.chainThreads, height: 1, depth: 1), bindGdn) }
            stage()
            if gdnSkipMask & 4 == 0 { graph.dispatch(pGdnFinish, grid: MTLSize(width: g.gdnValueHeads, height: rows, depth: 1),
                           threadsPerGroup: MTLSize(width: 32, height: 1, depth: 1), bindGdn) }
            if gdnSkipMask & 8 == 0 { graph.dispatch(pGdnHistory, grid: MTLSize(width: g.gdnValueHeads, height: runCount, depth: 1),
                           threadsPerGroup: MTLSize(width: 384, height: 1, depth: 1), bindGdn) }
        } else {
            graph.dispatch(pGdn, grid: MTLSize(width: g.gdnValueHeads, height: runCount, depth: 1),
                           threadsPerGroup: MTLSize(width: 1024, height: 1, depth: 1), bindGdn)
        }
        stage()
        return gemm(m.out, input: core, output: h, residual: true, rows: rows, emit: emit)
    }

    private struct ValueParams { var segments, chunksPerSegment: UInt32 }
    private struct DenseParams { var blocks, blockBase, heads, kvHeads, maxPages, qrowCap, tcap, rowBase, rowCount: UInt32; var scale: Float }

    /// Scores, softmax and values as three passes, in sub-batches of blocks sized to the scratch.
    private func encodeDenseAttention(scores pScores: MTLComputePipelineState, softmax pSoftmax: MTLComputePipelineState,
                                      values pValues: MTLComputePipelineState, layer: Int, rows: Int) {
        let tcap = ((stepPages * Self.pageTokens + 63) / 64) * 64
        // 12 bytes per query-token; a block holds 8 rows x 24 heads of queries.
        let perBlock = 8 * g.heads * tcap * (4 + 2 * attnOperandArrays)
        let blocksPerPass = max(1, attnScratchBytes / perBlock)
        let blockPtr = attnBlocks.contents().assumingMemoryBound(to: UInt32.self)
        var first = 0
        while first < attnBlockCount {
            let blockCount = min(blocksPerPass, attnBlockCount - first)
            let rowBase = Int(blockPtr[first * 2])
            let lastBlock = first + blockCount - 1
            let rowEnd = Int(blockPtr[lastBlock * 2]) + Int(blockPtr[lastBlock * 2 + 1])
            var p = DenseParams(blocks: UInt32(blockCount), blockBase: UInt32(first), heads: UInt32(g.heads), kvHeads: UInt32(g.kvHeads),
                                maxPages: UInt32(maxPagesPerSlot), qrowCap: UInt32(attnQrowCap + 8), tcap: UInt32(tcap),
                                rowBase: UInt32(rowBase), rowCount: UInt32(rowEnd - rowBase), scale: 1 / Float(g.headDim).squareRoot())
            let size = MemoryLayout<DenseParams>.stride
            let group = MTLSize(width: 128, height: 1, depth: 1)
            if !skip.contains("scores") {
            graph.dispatch(pScores, grid: MTLSize(width: tcap / 256, height: blockCount, depth: g.kvHeads), threadsPerGroup: group) { e in
                e.setBuffer(attnQueries, offset: 0, index: 0)
                e.setBuffer(attnQuerySums, offset: 0, index: 1)
                e.setBuffer(attnBlocks, offset: 0, index: 2)
                e.setBuffer(rowSlot, offset: 0, index: 3)
                e.setBuffer(rowPos, offset: 0, index: 4)
                e.setBuffer(pageTable, offset: 0, index: 5)
                e.setBuffer(kPools[layer], offset: 0, index: 6)
                e.setBuffer(kMeta[layer], offset: 0, index: 7)
                e.setBuffer(attnScores, offset: 0, index: 8)
                e.setBytes(&p, length: size, index: 9)
            }
            }
            stage()
            if !skip.contains("softmax") {
            graph.dispatch(pSoftmax, grid: MTLSize(width: g.heads, height: rowEnd - rowBase, depth: 1), threadsPerGroup: groupSize) { e in
                e.setBuffer(attnScores, offset: 0, index: 0)
                e.setBuffer(attnRowQ, offset: 0, index: 1)
                e.setBuffer(attnBlocks, offset: 0, index: 2)
                e.setBuffer(rowSlot, offset: 0, index: 3)
                e.setBuffer(rowPos, offset: 0, index: 4)
                e.setBuffer(pageTable, offset: 0, index: 5)
                e.setBuffer(vMeta[layer], offset: 0, index: 6)
                e.setBuffer(attnWeighted, offset: 0, index: 7)
                e.setBuffer(attnStats, offset: 0, index: 8)
                e.setBytes(&p, length: size, index: 9)
            }
            }
            stage()
            if !skip.contains("values") {
            let chunks = tcap / 64
            let chunksPerSegment = max(8, (chunks + Self.maxValueSegments - 1) / Self.maxValueSegments)
            var vp = ValueParams(segments: UInt32((chunks + chunksPerSegment - 1) / chunksPerSegment), chunksPerSegment: UInt32(chunksPerSegment))
            graph.dispatch(pValues, grid: MTLSize(width: 4 * Int(vp.segments), height: blockCount, depth: g.kvHeads), threadsPerGroup: group) { e in
                e.setBuffer(attnWeighted, offset: 0, index: 0)
                e.setBuffer(attnBlocks, offset: 0, index: 1)
                e.setBuffer(rowSlot, offset: 0, index: 2)
                e.setBuffer(rowPos, offset: 0, index: 3)
                e.setBuffer(pageTable, offset: 0, index: 4)
                e.setBuffer(vPools[layer], offset: 0, index: 5)
                e.setBuffer(attnValuePartials, offset: 0, index: 6)
                e.setBytes(&p, length: size, index: 7)
                e.setBytes(&vp, length: MemoryLayout<ValueParams>.stride, index: 8)
            }
            stage()
            graph.dispatch(pAttnValuesMerge!, grid: MTLSize(width: 4, height: blockCount, depth: g.kvHeads),
                           threadsPerGroup: MTLSize(width: 64, height: 1, depth: 1)) { e in
                e.setBuffer(attnValuePartials, offset: 0, index: 0)
                e.setBuffer(attnStats, offset: 0, index: 1)
                e.setBuffer(attnBlocks, offset: 0, index: 2)
                e.setBuffer(proj, offset: 0, index: 3)
                e.setBuffer(core, offset: 0, index: 4)
                e.setBuffer(coreB, offset: 0, index: 5)
                e.setBuffer(coreSums, offset: 0, index: 6)
                e.setBytes(&p, length: size, index: 7)
                e.setBytes(&vp, length: MemoryLayout<ValueParams>.stride, index: 8)
            }
            }
            first += blockCount
            if first < attnBlockCount { stage() }
        }
    }

    /// `live` is how many leading rows the layer's output is wanted for (see `step`): every row's
    /// keys and values are stored; queries, the scan and the output projection are for those.
    private func encodeFull(_ m: FullMixerWeights, layer: Int, rows allRows: Int, live: Int, source: MTLBuffer, emit: TensorHandle?) -> Bool {
        var rows = allRows
        if live < allRows, separateKVMeta, pAttnQPrepNa != nil, pAttnKVStoreNa != nil, pAttnScanNa != nil { rows = live }
        if attnSkipMask & 16 != 0 { stage() } else {
        if rows > 0 { gemm(m.q, input: source, output: proj, residual: false, rows: rows) }
        gemm(m.k, input: source, output: kproj, residual: false, rows: allRows)
        gemm(m.v, input: source, output: vproj, residual: false, rows: allRows)

        stage() }
        /// Timing: the synthetic accelerator probe as a stage of its own (see attnSkipMask 512, 1024, 2048).
        func probeStage(own: Bool) {
            guard let probe = try? library.pipeline("sp_probe_matmul"), let out = aluProbeOut else { return }
            var pp = (UInt32(3), UInt32(31))
            graph.dispatch(probe, grid: MTLSize(width: 4, height: 16, depth: 1), threadsPerGroup: MTLSize(width: 256, height: 1, depth: 1)) { e in
                e.setBuffer(own ? features : attnQueries, offset: 0, index: 0); e.setBuffer(own ? features : kPools[layer], offset: 0, index: 1)
                e.setBuffer(out, offset: 0, index: 2); e.setBytes(&pp, length: 8, index: 3)
            }
            stage()
        }
        if attnSkipMask & 512 != 0 { probeStage(own: true) }             // before the query prepare and key store
        if separateKVMeta, let pAttnQPrepNa, let pAttnKVStoreNa {
            let perKV = g.heads / g.kvHeads
            var qp = NaQPrepParams(rows: UInt32(rows), heads: UInt32(g.heads), kvHeads: UInt32(g.kvHeads), rotary: UInt32(g.rotaryDim),
                                   rowCap: UInt32(attnQrowCap + 8), eps: g.normEps, theta: g.ropeTheta)
            if rows > 0, attnSkipMask & 1 == 0 { graph.dispatch(pAttnQPrepNa, grid: MTLSize(width: g.heads, height: rows, depth: 1), threadsPerGroup: groupSize) { e in
                e.setBuffer(proj, offset: 0, index: 0)
                bind(e, m.qNorm, 1)
                e.setBuffer(rowPos, offset: 0, index: 2)
                e.setBuffer(attnQueries, offset: 0, index: 3)
                e.setBuffer(attnQuerySums, offset: 0, index: 4)
                e.setBytes(&qp, length: MemoryLayout<NaQPrepParams>.stride, index: 5)
                e.setBuffer(attnRowQ, offset: 0, index: 6)
                e.setBuffer(currentInv, offset: 0, index: 7)
            } }
            var kp = NaKVParams(rows: UInt32(allRows), kvHeads: UInt32(g.kvHeads), rotary: UInt32(g.rotaryDim),
                                maxPages: UInt32(maxPagesPerSlot), eps: g.normEps, theta: g.ropeTheta)
            if attnSkipMask & 2 == 0 { graph.dispatch(pAttnKVStoreNa, grid: MTLSize(width: g.kvHeads, height: allRows, depth: 1), threadsPerGroup: groupSize) { e in
                e.setBuffer(kproj, offset: 0, index: 0)
                e.setBuffer(vproj, offset: 0, index: 1)
                bind(e, m.kNorm, 2)
                e.setBuffer(rowSlot, offset: 0, index: 3)
                e.setBuffer(rowPos, offset: 0, index: 4)
                e.setBuffer(pageTable, offset: 0, index: 5)
                e.setBuffer(kPools[layer], offset: 0, index: 6)
                e.setBuffer(kMeta[layer], offset: 0, index: 7)
                e.setBuffer(vPools[layer], offset: 0, index: 8)
                e.setBuffer(vMeta[layer], offset: 0, index: 9)
                e.setBytes(&kp, length: MemoryLayout<NaKVParams>.stride, index: 10)
                e.setBuffer(currentInv, offset: 0, index: 11)
            } }
            // Keys and values only: the caller ends the stage.
            if rows == 0 { return false }
            stage()
            if attnSkipMask & 1024 != 0 { probeStage(own: true) }        // after them, reading nothing they wrote
            if attnSkipMask & 2048 != 0 { probeStage(own: false) }       // after them, reading what they wrote
            // The attention blocks that hold the rows computed for: blocks are in row order.
            var attnBlockCount = attnBlockCount
            if rows < allRows {
                let blockPtr = attnBlocks.contents().assumingMemoryBound(to: UInt32.self)
                attnBlockCount = (0..<attnBlockCount).firstIndex(where: { blockPtr[$0 * 2] >= UInt32(rows) }) ?? attnBlockCount
            }
            if !skip.contains("attn"), let pAttnScores, let pAttnSoftmax, let pAttnValues {
                encodeDenseAttention(scores: pAttnScores, softmax: pAttnSoftmax, values: pAttnValues, layer: layer, rows: rows)
                stage()   // the output projection reads what the value pass wrote
            } else {
                // A span is a chain of chunks one threadgroup walks in sequence. A full step has
                // enough blocks to fill the GPU with sixteen spans; a narrow one (a verify block)
                // does not, and is paced by the length of that chain, so it gets more, shorter
                // spans within the same partials budget.
                let spanLimit = wideSpanLimit > 0 && attnBlockCount > 1 ? wideSpanLimit
                    : max(Self.maxSpans, min(narrowSpanLimit, Self.maxSpans * config.maxRows / max(rows, 1)))
                let pagesPerSpan = (stepPages + spanLimit - 1) / spanLimit
                let spans = (stepPages + pagesPerSpan - 1) / pagesPerSpan
                if !skip.contains("attn"), let pAttnScanNa, let pAttnMerge {
                    var sp = NaScanParams(blocks: UInt32(attnBlockCount), heads: UInt32(g.heads), kvHeads: UInt32(g.kvHeads),
                                          maxPages: UInt32(maxPagesPerSlot), spans: UInt32(spans), pagesPerSpan: UInt32(pagesPerSpan),
                                          rowCap: UInt32(attnQrowCap + 8), scale: 1 / Float(g.headDim).squareRoot(),
                                          aliasTokens: attnAliasTokens, blocksAcross: attnBlocksAcross ? 1 : 0)
                    if attnSkipMask & 256 != 0, let probe = try? library.pipeline("sp_probe_matmul"), let out = aluProbeOut {
                        // Timing: the synthetic probe in the scan's place, same grid, three rounds.
                        var pp = (UInt32(3), UInt32(31))
                        graph.dispatch(probe, grid: MTLSize(width: g.kvHeads * spans, height: attnBlockCount, depth: 1),
                                       threadsPerGroup: MTLSize(width: 256, height: 1, depth: 1)) { e in
                            e.setBuffer(attnQueries, offset: 0, index: 0); e.setBuffer(kPools[layer], offset: 0, index: 1)
                            e.setBuffer(out, offset: 0, index: 2); e.setBytes(&pp, length: 8, index: 3)
                        }
                    }
                    if attnSkipMask & 4 == 0 { graph.dispatch(pAttnScanNa, grid: attnBlocksAcross ? MTLSize(width: g.kvHeads * attnBlockCount, height: spans, depth: 1)
                                                                                : MTLSize(width: g.kvHeads * spans * attnGridPad, height: attnBlockCount, depth: 1),
                                   threadsPerGroup: MTLSize(width: attnScanThreads, height: 1, depth: 1)) { e in
                        e.setBuffer(attnQueries, offset: 0, index: 0)
                        e.setBuffer(attnQuerySums, offset: 0, index: 1)
                        e.setBuffer(attnBlocks, offset: 0, index: 2)
                        e.setBuffer(rowSlot, offset: 0, index: 3)
                        e.setBuffer(rowPos, offset: 0, index: 4)
                        e.setBuffer(pageTable, offset: 0, index: 5)
                        e.setBuffer(kPools[layer], offset: 0, index: 6)
                        e.setBuffer(kMeta[layer], offset: 0, index: 7)
                        e.setBuffer(vPools[layer], offset: 0, index: 8)
                        e.setBuffer(vMeta[layer], offset: 0, index: 9)
                        e.setBuffer(attnPartials, offset: 0, index: 10)
                        e.setBytes(&sp, length: MemoryLayout<NaScanParams>.stride, index: 11)
                    } }
                    stage()
                    var mp = AttnScanParams(rows: UInt32(rows), heads: UInt32(g.heads), kvHeads: UInt32(g.kvHeads),
                                            maxPages: UInt32(maxPagesPerSlot), spans: UInt32(spans), pagesPerSpan: UInt32(pagesPerSpan),
                                            scale: 1 / Float(g.headDim).squareRoot())
                    if attnSkipMask & 8 == 0 { graph.dispatch(pAttnMerge, grid: MTLSize(width: g.heads, height: rows, depth: 1),
                                   threadsPerGroup: MTLSize(width: g.headDim, height: 1, depth: 1)) { e in
                        e.setBuffer(attnPartials, offset: 0, index: 0)
                        e.setBuffer(proj, offset: 0, index: 1)
                        e.setBuffer(core, offset: 0, index: 2)
                        e.setBuffer(coreB, offset: 0, index: 3)
                        e.setBuffer(coreSums, offset: 0, index: 4)
                        e.setBytes(&mp, length: MemoryLayout<AttnScanParams>.stride, index: 5)
                        e.setBuffer(currentInv, offset: 0, index: 6)
                    } }
                }
                stage()
            }
            _ = perKV
            if attnSkipMask & 32 != 0 { return false }
            return gemm(m.o, input: core, output: h, residual: true, rows: rows, emit: emit)
        }
        let size = MemoryLayout<AttnPrepParams>.stride
        var kp = AttnPrepParams(rows: UInt32(rows), heads: UInt32(g.kvHeads), rotary: UInt32(g.rotaryDim),
                                inHeadStride: UInt32(g.headDim), maxPages: UInt32(maxPagesPerSlot),
                                eps: g.normEps, theta: g.ropeTheta)
        graph.dispatch(pKVStore, grid: MTLSize(width: g.kvHeads, height: rows, depth: 1), threadsPerGroup: groupSize) { e in
            e.setBuffer(kproj, offset: 0, index: 0)
            e.setBuffer(vproj, offset: 0, index: 1)
            bind(e, m.kNorm, 2)
            e.setBuffer(rowSlot, offset: 0, index: 3)
            e.setBuffer(rowPos, offset: 0, index: 4)
            e.setBuffer(pageTable, offset: 0, index: 5)
            e.setBuffer(kPools[layer], offset: 0, index: 6)
            e.setBuffer(vPools[layer], offset: 0, index: 7)
            e.setBytes(&kp, length: size, index: 8)
        }
        if let pAttnScan, let pAttnMerge {
            // Query prep shares the stage with the KV store: both only read the projections.
            var qp = AttnPrepParams(rows: UInt32(rows), heads: UInt32(g.heads), rotary: UInt32(g.rotaryDim),
                                    inHeadStride: UInt32(g.headDim * 2), maxPages: UInt32(maxPagesPerSlot),
                                    eps: g.normEps, theta: g.ropeTheta)
            graph.dispatch(pQPrep, grid: MTLSize(width: g.heads, height: rows, depth: 1), threadsPerGroup: groupSize) { e in
                e.setBuffer(proj, offset: 0, index: 0)
                bind(e, m.qNorm, 1)
                e.setBuffer(rowPos, offset: 0, index: 2)
                e.setBuffer(qn, offset: 0, index: 3)
                e.setBytes(&qp, length: size, index: 4)
            }
            stage()
            let pagesPerSpan = (stepPages + Self.maxSpans - 1) / Self.maxSpans
            let spans = (stepPages + pagesPerSpan - 1) / pagesPerSpan
            var sp = AttnScanParams(rows: UInt32(rows), heads: UInt32(g.heads), kvHeads: UInt32(g.kvHeads),
                                    maxPages: UInt32(maxPagesPerSlot), spans: UInt32(spans), pagesPerSpan: UInt32(pagesPerSpan),
                                    scale: 1 / Float(g.headDim).squareRoot())
            if !skip.contains("attn") {
                graph.dispatch(pAttnScan, grid: MTLSize(width: g.kvHeads * spans, height: rows, depth: 1),
                               threadsPerGroup: MTLSize(width: g.headDim, height: 1, depth: 1)) { e in
                    e.setBuffer(qn, offset: 0, index: 0)
                    e.setBuffer(rowSlot, offset: 0, index: 1)
                    e.setBuffer(rowPos, offset: 0, index: 2)
                    e.setBuffer(pageTable, offset: 0, index: 3)
                    e.setBuffer(kPools[layer], offset: 0, index: 4)
                    e.setBuffer(vPools[layer], offset: 0, index: 5)
                    e.setBuffer(attnPartials, offset: 0, index: 6)
                    e.setBytes(&sp, length: MemoryLayout<AttnScanParams>.stride, index: 7)
                }
                stage()
                graph.dispatch(pAttnMerge, grid: MTLSize(width: g.heads, height: rows, depth: 1),
                               threadsPerGroup: MTLSize(width: g.headDim, height: 1, depth: 1)) { e in
                    e.setBuffer(attnPartials, offset: 0, index: 0)
                    e.setBuffer(proj, offset: 0, index: 1)
                    e.setBuffer(core, offset: 0, index: 2)
                    e.setBuffer(coreB, offset: 0, index: 3)
                    e.setBuffer(coreSums, offset: 0, index: 4)
                    e.setBytes(&sp, length: MemoryLayout<AttnScanParams>.stride, index: 5)
                    e.setBuffer(currentInv, offset: 0, index: 6)
                }
            }
            stage()
        } else {
        stage()
        var ap = AttnFusedParams(rows: UInt32(rows), heads: UInt32(g.heads), kvHeads: UInt32(g.kvHeads),
                                 maxPages: UInt32(maxPagesPerSlot), rotary: UInt32(g.rotaryDim),
                                 scale: 1 / Float(g.headDim).squareRoot(), eps: g.normEps, theta: g.ropeTheta)
        if !skip.contains("attn") {
        graph.dispatch(pAttentionFused, grid: MTLSize(width: g.heads, height: rows, depth: 1),
                       threadsPerGroup: MTLSize(width: g.headDim, height: 1, depth: 1)) { e in
            e.setBuffer(proj, offset: 0, index: 0)
            bind(e, m.qNorm, 1)
            e.setBuffer(rowSlot, offset: 0, index: 2)
            e.setBuffer(rowPos, offset: 0, index: 3)
            e.setBuffer(pageTable, offset: 0, index: 4)
            e.setBuffer(kPools[layer], offset: 0, index: 5)
            e.setBuffer(vPools[layer], offset: 0, index: 6)
            e.setBuffer(core, offset: 0, index: 7)
            e.setBytes(&ap, length: MemoryLayout<AttnFusedParams>.stride, index: 8)
            e.setBuffer(coreB, offset: 0, index: 9)
            e.setBuffer(coreSums, offset: 0, index: 10)
        }
        }
        stage()
        }
        return gemm(m.o, input: core, output: h, residual: true, rows: rows)
    }
}
