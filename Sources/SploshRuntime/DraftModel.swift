// DraftModel.swift — the DFlash 2 block-diffusion drafter.
//
// A draft pass proposes a whole block of tokens at once: the block is [anchor, mask x 7], and
// five non-causal layers attend over the block plus a sliding window of context derived from
// the target model's hidden states at five tap layers. A small selector then traces one path
// through the top candidates at each position. The target verifies the block in one step.
//
// Formulas follow dflash/model.py in z-lab/dflash (Apache-2.0). Speculative decoding is
// lossless — the target decides every token — so the only thing draft accuracy affects is how
// many tokens each verify step accepts.
//
// The checkpoint ships as BF16 safetensors; its linear layers are quantised here to the same
// MLX affine q4 form as the target so one pass over the draft is ~1.1 GB instead of 7.2 GB.

import Accelerate
import Foundation
@preconcurrency import Metal
import SploshCore
import SploshModel

public enum DraftError: Error, CustomStringConvertible {
    case invalid(String)
    public var description: String {
        switch self { case .invalid(let reason): return "draft model: \(reason)" }
    }
}

/// One block to draft: `block - 1` tokens after `anchor`, which sits at position `start`.
public struct DraftRequest: Sendable {
    public var slot: Int
    public var anchor: Int
    public var start: Int
    /// `DraftModel.blockSize`, or a larger multiple of it up to `DraftModel.maxBlock`.
    public var block: Int
    /// The first `blockSize` rows attend only to each other (and the context), so they are
    /// exactly the block the checkpoint was trained on; the rows after them see the whole block.
    public var isolatePrefix: Bool

    public init(slot: Int, anchor: Int, start: Int, block: Int = DraftModel.blockSize, isolatePrefix: Bool = false) {
        self.slot = slot; self.anchor = anchor; self.start = start; self.block = block; self.isolatePrefix = isolatePrefix
    }
}

public struct DraftProposal: Sendable {
    /// The chosen token at each drafted position.
    public let tokens: [Int]
    /// The candidates considered at each position.
    public let candidates: [[Int]]
    /// The selector's sampling distribution over `candidates` (nil for greedy).
    public let probabilities: [[Float]]?
    /// For a block copied from the session's own text: the length of the run its source
    /// shares with the text. Nil for a block from the draft model.
    public var copyMatch: Int?

    public init(tokens: [Int], candidates: [[Int]], probabilities: [[Float]]?, copyMatch: Int? = nil) {
        self.tokens = tokens; self.candidates = candidates; self.probabilities = probabilities; self.copyMatch = copyMatch
    }

    /// A block copied from the text: each position proposes its one token with certainty,
    /// which makes the acceptance test at temperature > 0 the target's own probability of it.
    public init(copied tokens: [Int], match: Int, sampling: Bool) {
        self.init(tokens: tokens, candidates: tokens.map { [$0] }, probabilities: sampling ? tokens.map { _ in [1] } : nil, copyMatch: match)
    }
}

public final class DraftModel: @unchecked Sendable {
    /// The block the checkpoint was trained on: [anchor, mask x 7].
    public static let blockSize = 8
    /// The longest block a request may ask for (a multiple of `blockSize`). Positions past the
    /// eighth are outside what the checkpoint was trained on, and it still gets them right
    /// often enough on text it is already getting right; see `propose`.
    public static let maxBlock = 16
    public static let window = 2048
    /// Ring positions per slot: the window plus one block, rounded to a chunk multiple.
    public static let ringCapacity = 2304
    public static let candidateCount = 16
    public static let maskToken = 248_070

    private static let layers = 5
    private static let heads = 32, kvHeads = 8, headDim = 128
    private static let groupSize = 16, groups = 320
    private static let rank = 256

    private struct Layer {
        let inputNorm, postNorm, qNorm, kNorm, attentionBase, mlpBase: TensorHandle
        let q, k, v, o, gate, up, down, attentionKernel, mlpKernel: Q4Handle
    }

    private let engine: Engine
    private let layers: [Layer]
    private let fc: Q4Handle
    private let hiddenNorm, finalNorm: TensorHandle
    private let hiddenProjection: [Float]
    private let file: MappedFile
    private let predecessorOffset, successorOffset: Int

    private let pConv, pKVStore, pAttention, pAttentionMerge, pQueryPrepare, pGather: MTLComputePipelineState
    /// The accelerator kernel (default) or the ordinary-arithmetic one (SPLOSH_DRAFT_ATTN=alu).
    private let attentionOnAccelerator: Bool
    private let preparedQueries: MTLBuffer
    private let attentionPartials: MTLBuffer
    private static let attentionSpans = 8
    private let contextRows: Int
    private let blockRows: Int

    // Context pass
    private let cFeatures, cHidden, cNormed, cSlot, cPos: MTLBuffer
    private let cK: [MTLBuffer], cV: [MTLBuffer]
    // Block pass
    private let bTokens, bSlot, bPos, bLo, bHi, bGather, bOffset: MTLBuffer
    private let dh, x, xc, t, dyn, qp, kp, vp, ao, gate, up, dn, selected, logits: MTLBuffer
    private let kRing: [MTLBuffer], vRing: [MTLBuffer]

    /// Context positions [first, end) currently valid in each slot's ring.
    private var contextFirst: [Int]
    private var contextEnd: [Int]

    public private(set) var residentBytes = 0
    public private(set) var scratchBytes = 0

    public init(engine: Engine, safetensorsURL: URL) throws {
        self.engine = engine
        let device = engine.device
        let g = engine.g
        let file = try MappedFile(url: safetensorsURL)
        self.file = file
        let tensors = try SafetensorsHeader(file: file)

        var resident = 0
        var kept: [MTLBuffer] = []
        func dense(_ name: String, _ shape: [Int]) throws -> TensorHandle {
            let entry = try tensors.require(name, dtype: "BF16", shape: shape)
            guard let buffer = device.makeBuffer(bytes: file.base.advanced(by: entry.offset), length: entry.length, options: .storageModeShared) else {
                throw DraftError.invalid("unable to allocate \(name)")
            }
            resident += entry.length
            kept.append(buffer)
            return TensorHandle(name: "draft." + name, span: engine.registerSpan(buffer), offset: 0, byteLength: entry.length, shape: shape)
        }
        func linear(_ name: String, rows: Int, inner: Int) throws -> Q4Handle {
            let entry = try tensors.require(name, dtype: "BF16", shape: [rows, inner])
            let groupsPerRow = inner / 64
            guard inner % 64 == 0,
                  let packed = device.makeBuffer(length: rows * inner / 2, options: .storageModeShared),
                  let scales = device.makeBuffer(length: rows * groupsPerRow * 2, options: .storageModeShared),
                  let biases = device.makeBuffer(length: rows * groupsPerRow * 2, options: .storageModeShared) else {
                throw DraftError.invalid("unable to allocate \(name)")
            }
            Self.quantize(source: file.base.advanced(by: entry.offset).assumingMemoryBound(to: UInt16.self),
                          rows: rows, inner: inner, tiled: rows % Converter.tileRows == 0,
                          packed: packed.contents().assumingMemoryBound(to: UInt32.self),
                          scales: scales.contents().assumingMemoryBound(to: UInt16.self),
                          biases: biases.contents().assumingMemoryBound(to: UInt16.self))
            resident += packed.length + scales.length + biases.length
            kept += [packed, scales, biases]
            func handle(_ suffix: String, _ buffer: MTLBuffer, _ shape: [Int]) -> TensorHandle {
                TensorHandle(name: "draft." + name + suffix, span: engine.registerSpan(buffer), offset: 0, byteLength: buffer.length, shape: shape)
            }
            let weight = Q4Handle(packed: handle("", packed, [rows, inner / 8]), scales: handle(".scales", scales, [rows, groupsPerRow]),
                                  biases: handle(".biases", biases, [rows, groupsPerRow]),
                                  rows: rows, inner: inner, rowStrideWords: inner / 8, groupsPerRow: groupsPerRow)
            // The accelerator kernels that read weights at streaming speed need the tiled
            // layout; row-major, a draft pass spent 11 of its 17 ms in the MLP matmuls.
            if rows % Converter.tileRows == 0 { engine.registerTiled(weight) }
            return weight
        }

        let hidden = g.hidden, mlp = g.intermediate
        let attention = Self.heads * Self.headDim, kv = Self.kvHeads * Self.headDim
        var built: [Layer] = []
        for index in 0..<Self.layers {
            let p = "layers.\(index)."
            built.append(Layer(
                inputNorm: try dense(p + "input_layernorm.weight", [hidden]),
                postNorm: try dense(p + "post_attention_layernorm.weight", [hidden]),
                qNorm: try dense(p + "self_attn.q_norm.weight", [Self.headDim]),
                kNorm: try dense(p + "self_attn.k_norm.weight", [Self.headDim]),
                attentionBase: try dense(p + "attention_conv.base_kernel", [2, 2, hidden]),
                mlpBase: try dense(p + "mlp_conv.base_kernel", [2, 2, hidden]),
                q: try linear(p + "self_attn.q_proj.weight", rows: attention, inner: hidden),
                k: try linear(p + "self_attn.k_proj.weight", rows: kv, inner: hidden),
                v: try linear(p + "self_attn.v_proj.weight", rows: kv, inner: hidden),
                o: try linear(p + "self_attn.o_proj.weight", rows: hidden, inner: attention),
                gate: try linear(p + "mlp.gate_proj.weight", rows: mlp, inner: hidden),
                up: try linear(p + "mlp.up_proj.weight", rows: mlp, inner: hidden),
                down: try linear(p + "mlp.down_proj.weight", rows: hidden, inner: mlp),
                attentionKernel: try linear(p + "attention_conv.kernel_projection.weight", rows: 4 * Self.groups, inner: hidden),
                mlpKernel: try linear(p + "mlp_conv.kernel_projection.weight", rows: 4 * Self.groups, inner: hidden)))
        }
        layers = built
        fc = try linear("fc.weight", rows: hidden, inner: hidden * Engine.featureLayers.count)
        hiddenNorm = try dense("hidden_norm.weight", [hidden])
        finalNorm = try dense("norm.weight", [hidden])

        let projection = try tensors.require("candidate_selector.hidden_projection.weight", dtype: "BF16", shape: [Self.rank, hidden])
        let raw = file.base.advanced(by: projection.offset).assumingMemoryBound(to: UInt16.self)
        hiddenProjection = (0..<Self.rank * hidden).map { Float(bitPattern: UInt32(raw[$0]) << 16) }
        predecessorOffset = try tensors.require("candidate_selector.predecessor_codebook", dtype: "BF16", shape: [g.vocab, Self.rank]).offset
        successorOffset = try tensors.require("candidate_selector.successor_codebook", dtype: "BF16", shape: [g.vocab, Self.rank]).offset
        // The codebooks (127 MB each) are read by token id on every draft pass straight from the
        // file mapping. Touch them once so a lookup never waits on the disk.
        let codebookBytes = g.vocab * Self.rank * 2
        for offset in [predecessorOffset, successorOffset] {
            let start = file.base.advanced(by: offset)
            madvise(UnsafeMutableRawPointer(mutating: start), codebookBytes, MADV_WILLNEED)
            var sink: UInt8 = 0
            for page in stride(from: 0, to: codebookBytes, by: 16384) { sink &+= start.load(fromByteOffset: page, as: UInt8.self) }
            _ = sink
        }
        residentBytes = resident

        pConv = try engine.library.pipeline("sp_dyn_conv")
        pKVStore = try engine.library.pipeline("sp_draft_kv_store")
        attentionOnAccelerator = ProcessInfo.processInfo.environment["SPLOSH_DRAFT_ATTN"] != "alu"
        pAttention = try engine.library.pipeline(attentionOnAccelerator ? "sp_draft_attention_na" : "sp_draft_attention_span")
        pQueryPrepare = try engine.library.pipeline("sp_draft_q_prepare")
        pAttentionMerge = try engine.library.pipeline("sp_draft_attention_merge")
        pGather = try engine.library.pipeline("sp_gather_rows")

        // Accelerator tiles read whole 128-row-padded blocks.
        let contextRows = ((engine.config.maxRows + 127) / 128) * 128
        let blockRows = ((engine.config.maxSlots * Self.maxBlock + 127) / 128) * 128
        self.contextRows = contextRows
        self.blockRows = blockRows
        var scratch = 0
        func make(_ label: String, _ bytes: Int) throws -> MTLBuffer {
            guard let buffer = device.makeBuffer(length: max(bytes, 16), options: .storageModeShared) else {
                throw DraftError.invalid("unable to allocate \(label)")
            }
            buffer.label = "splosh.draft." + label
            scratch += bytes
            kept.append(buffer)
            return buffer
        }
        let f = MemoryLayout<Float>.stride, u = MemoryLayout<UInt32>.stride
        cFeatures = try make("cFeatures", contextRows * fc.inner * f)
        cHidden = try make("cHidden", contextRows * hidden * f)
        cNormed = try make("cNormed", contextRows * hidden * f)
        cSlot = try make("cSlot", contextRows * u)
        cPos = try make("cPos", contextRows * u)
        cK = try (0..<Self.layers).map { try make("cK\($0)", contextRows * kv * f) }
        cV = try (0..<Self.layers).map { try make("cV\($0)", contextRows * kv * f) }
        bTokens = try make("bTokens", blockRows * u)
        bSlot = try make("bSlot", blockRows * u)
        bPos = try make("bPos", blockRows * u)
        bLo = try make("bLo", blockRows * u)
        bHi = try make("bHi", blockRows * u)
        bGather = try make("bGather", blockRows * u)
        bOffset = try make("bOffset", blockRows * u)
        dh = try make("dh", blockRows * hidden * f)
        x = try make("x", blockRows * hidden * f)
        xc = try make("xc", blockRows * hidden * f)
        t = try make("t", blockRows * hidden * f)
        dyn = try make("dyn", blockRows * 4 * Self.groups * f)
        qp = try make("qp", blockRows * attention * f)
        kp = try make("kp", blockRows * kv * f)
        vp = try make("vp", blockRows * kv * f)
        ao = try make("ao", blockRows * attention * f)
        attentionPartials = try make("attentionPartials", blockRows * Self.heads * Self.attentionSpans * (Self.headDim + 2) * f)
        preparedQueries = try make("preparedQueries", blockRows * Self.heads * Self.headDim * MemoryLayout<UInt16>.stride)
        gate = try make("gate", blockRows * mlp * f)
        up = try make("up", blockRows * mlp * f)
        dn = try make("dn", blockRows * hidden * f)
        selected = try make("selected", blockRows * hidden * f)
        logits = try make("logits", engine.config.maxSlots * (Self.maxBlock - 1) * g.vocab * f)
        let ringBytes = engine.config.maxSlots * Self.ringCapacity * kv * MemoryLayout<UInt16>.stride
        kRing = try (0..<Self.layers).map { try make("kRing\($0)", ringBytes) }
        vRing = try (0..<Self.layers).map { try make("vRing\($0)", ringBytes) }
        scratchBytes = scratch
        engine.keepResident(kept)
        contextFirst = Array(repeating: 0, count: engine.config.maxSlots)
        contextEnd = Array(repeating: 0, count: engine.config.maxSlots)
    }

    // MARK: - Context

    public func resetSlot(_ slot: Int) {
        contextFirst[slot] = 0
        contextEnd[slot] = 0
    }

    /// Forget context at or beyond `tokenCount` (the session was rolled back).
    /// The slot's context rings as they stand for a prefix of `tokenCount` tokens: the rings
    /// themselves plus the range of positions still valid in them. nil when nothing useful is.
    public func exportContext(_ slot: Int, tokenCount: Int) -> DraftContextSnapshot? {
        guard contextEnd[slot] >= tokenCount else { return nil }
        // Positions evaluated after the prefix overwrote ring entries one capacity back.
        let first = max(contextFirst[slot], contextEnd[slot] + Self.maxBlock - Self.ringCapacity)
        guard first < tokenCount else { return nil }
        let perSlot = kRing[0].length / engine.config.maxSlots
        var rings = Data(capacity: perSlot * 2 * Self.layers)
        for index in 0..<Self.layers {
            rings.append(Data(bytes: kRing[index].contents().advanced(by: slot * perSlot), count: perSlot))
            rings.append(Data(bytes: vRing[index].contents().advanced(by: slot * perSlot), count: perSlot))
        }
        return DraftContextSnapshot(first: first, end: tokenCount, rings: rings)
    }

    public func importContext(_ slot: Int, snapshot: DraftContextSnapshot) {
        let perSlot = kRing[0].length / engine.config.maxSlots
        guard snapshot.rings.count == perSlot * 2 * Self.layers else { resetSlot(slot); return }
        snapshot.rings.withUnsafeBytes { raw in
            var cursor = raw.baseAddress!
            for index in 0..<Self.layers {
                memcpy(kRing[index].contents().advanced(by: slot * perSlot), cursor, perSlot); cursor += perSlot
                memcpy(vRing[index].contents().advanced(by: slot * perSlot), cursor, perSlot); cursor += perSlot
            }
        }
        contextFirst[slot] = snapshot.first
        contextEnd[slot] = snapshot.end
    }

    public func truncate(_ slot: Int, tokenCount: Int) {
        guard tokenCount < contextEnd[slot] else { return }
        // Later positions may have overwritten ring entries older than one capacity back.
        contextFirst[slot] = max(contextFirst[slot], contextEnd[slot] + Self.maxBlock - Self.ringCapacity)
        contextEnd[slot] = tokenCount
        if contextFirst[slot] >= contextEnd[slot] { contextFirst[slot] = tokenCount }
    }

    /// Number of context positions the draft can see for a block starting at `start`.
    public func contextLength(_ slot: Int, start: Int) -> Int {
        contextEnd[slot] == start ? start - max(contextFirst[slot], start - Self.window + 1) : 0
    }

    /// Fold target hidden states into the draft's context. Call straight after an engine step
    /// run with `captureFeatures`, naming the step rows to keep and their positions.
    public func pushContext(_ entries: [(slot: Int, position: Int, engineRow: Int)]) throws {
        guard !entries.isEmpty else { return }
        guard entries.count <= contextRows else { throw DraftError.invalid("too many context rows") }
        let g = engine.g
        let width = fc.inner
        let features = cFeatures.contents().assumingMemoryBound(to: Float.self)
        let slots = cSlot.contents().assumingMemoryBound(to: UInt32.self)
        let positions = cPos.contents().assumingMemoryBound(to: UInt32.self)
        for (index, entry) in entries.enumerated() {
            memcpy(features.advanced(by: index * width), engine.featureRow(entry.engineRow).baseAddress!, width * MemoryLayout<Float>.stride)
            slots[index] = UInt32(entry.slot)
            positions[index] = UInt32(entry.position)
            if entry.position != contextEnd[entry.slot] { contextFirst[entry.slot] = entry.position }
            contextEnd[entry.slot] = entry.position + 1
        }
        let rows = entries.count
        try engine.beginPass(label: "splosh.draft.context")
        engine.gemm(fc, input: cFeatures, output: cHidden, residual: false, rows: rows)
        engine.stage()
        engine.norm(cHidden, hiddenNorm, cNormed, count: rows, dim: g.hidden, mode: 1, emit: rows <= engine.config.maxRows)
        engine.stage()
        for (index, layer) in layers.enumerated() {
            engine.gemm(layer.k, input: cNormed, output: cK[index], residual: false, rows: rows)
            engine.gemm(layer.v, input: cNormed, output: cV[index], residual: false, rows: rows)
        }
        engine.stage()
        for (index, layer) in layers.enumerated() {
            storeKV(layer, index: index, k: cK[index], v: cV[index], slots: cSlot, positions: cPos, rows: rows)
        }
        try engine.graph.commitAndWait()
    }

    // MARK: - Proposal

    /// Draft `block - 1` tokens for each request. `start` is the anchor's position; `block` is
    /// `blockSize`, or a larger multiple of it up to `maxBlock`.
    ///
    /// A longer block costs the draft pass little and the verify step nothing while the step
    /// stays inside one 16-row tile, but it changes what the draft proposes at the early
    /// positions too, for the worse on text it finds hard (prose: 3.5 -> 3.15 tokens a step at
    /// sixteen). So callers ask for one only while a session's blocks are being accepted whole.
    /// Accumulated time in the block pass on the GPU and in candidate selection on the CPU.
    public private(set) var gpuSeconds = 0.0
    public private(set) var selectSeconds = 0.0
    public private(set) var topKSeconds = 0.0
    /// Timing probes: parts of the block pass to leave out (SPLOSH_DRAFT_SKIP=lmhead,attn,layers).
    private let debugSkip = Set((ProcessInfo.processInfo.environment["SPLOSH_DRAFT_SKIP"] ?? "").split(separator: ",").map(String.init))
    /// Timing probe: encode only the first so many stages of the block pass (proposals are
    /// then meaningless). `stageCount` is how many the last pass had.
    public var debugStageLimit: Int?
    public private(set) var stageCount = 0

    public func propose(_ requests: [DraftRequest], temperature: Float,
                        vocabLimit: Int, random: inout SplitMix) throws -> [DraftProposal] {
        guard !requests.isEmpty else { return [] }
        let g = engine.g
        let rows = requests.reduce(0) { $0 + $1.block }
        guard rows <= blockRows, requests.count <= engine.config.maxSlots else { throw DraftError.invalid("too many draft requests") }
        guard requests.allSatisfy({ $0.block >= Self.blockSize && $0.block <= Self.maxBlock && $0.block % Self.blockSize == 0 }) else {
            throw DraftError.invalid("draft block must be a multiple of \(Self.blockSize) up to \(Self.maxBlock)")
        }
        let tokens = bTokens.contents().assumingMemoryBound(to: UInt32.self)
        let slots = bSlot.contents().assumingMemoryBound(to: UInt32.self)
        let positions = bPos.contents().assumingMemoryBound(to: UInt32.self)
        let lo = bLo.contents().assumingMemoryBound(to: UInt32.self)
        let hi = bHi.contents().assumingMemoryBound(to: UInt32.self)
        let gather = bGather.contents().assumingMemoryBound(to: UInt32.self)
        let offsets = bOffset.contents().assumingMemoryBound(to: UInt32.self)
        // First logits row of each request; a request has `block - 1` of them.
        var firstPicked: [Int] = []
        var row = 0, picked = 0
        for request in requests {
            // Without contiguous context up to the anchor, the block attends only to itself.
            let first = contextEnd[request.slot] == request.start ? contextFirst[request.slot] : request.start
            firstPicked.append(picked)
            for offset in 0..<request.block {
                tokens[row] = UInt32(offset == 0 ? request.anchor : Self.maskToken)
                slots[row] = UInt32(request.slot)
                positions[row] = UInt32(request.start + offset)
                lo[row] = UInt32(first)
                hi[row] = UInt32(request.start + (request.isolatePrefix && offset < Self.blockSize ? Self.blockSize : request.block))
                offsets[row] = UInt32(offset)
                if offset > 0 { gather[picked] = UInt32(row); picked += 1 }
                row += 1
            }
        }

        let pickedRows = picked
        // The pass as a list of stages, each complete before the next starts.
        var stages: [() -> Void] = []
        stages.append { [self] in engine.embed(tokens: bTokens, output: dh, rows: rows) }
        for (index, layer) in layers.enumerated() {
            if debugSkip.contains("layers") { break }
            // The norms also emit the accelerator operand, which saves the GEMM behind each of
            // them a conversion stage.
            stages.append { [self] in engine.norm(dh, layer.inputNorm, x, count: rows, dim: g.hidden, mode: 1, emit: rows <= engine.config.maxRows) }
            stages.append { [self] in engine.gemm(layer.attentionKernel, input: x, output: dyn, residual: false, rows: rows) }
            stages.append { [self] in conv(input: x, base: layer.attentionBase, output: xc, which: 0, accumulate: false, rows: rows) }
            stages.append { [self] in
                engine.gemm(layer.q, input: xc, output: qp, residual: false, rows: rows)
                engine.gemm(layer.k, input: xc, output: kp, residual: false, rows: rows)
                engine.gemm(layer.v, input: xc, output: vp, residual: false, rows: rows)
            }
            stages.append { [self] in
                storeKV(layer, index: index, k: kp, v: vp, slots: bSlot, positions: bPos, rows: rows)
                if attentionOnAccelerator { prepareQueries(layer, rows: rows) }
            }
            // Two stages: the span scan and its merge.
            stages.append { [self] in if !debugSkip.contains("attn") { attention(layer, index: index, rows: rows) } }
            stages.append { [self] in engine.gemm(layer.o, input: ao, output: t, residual: false, rows: rows) }
            stages.append { [self] in conv(input: t, base: layer.attentionBase, output: dh, which: 1, accumulate: true, rows: rows) }
            stages.append { [self] in engine.norm(dh, layer.postNorm, x, count: rows, dim: g.hidden, mode: 1, emit: rows <= engine.config.maxRows) }
            stages.append { [self] in engine.gemm(layer.mlpKernel, input: x, output: dyn, residual: false, rows: rows) }
            stages.append { [self] in conv(input: x, base: layer.mlpBase, output: xc, which: 0, accumulate: false, rows: rows) }
            // Two stages: gate and up, then SiLU and the product.
            stages.append { [self] in engine.mlpIn(gate: layer.gate, up: layer.up, input: xc, output: gate, scratch: up, rows: rows) }
            stages.append { [self] in engine.gemm(layer.down, input: gate, output: t, residual: false, rows: rows) }
            stages.append { [self] in conv(input: t, base: layer.mlpBase, output: dh, which: 1, accumulate: true, rows: rows) }
        }
        stages.append { [self] in engine.norm(dh, finalNorm, dn, count: rows, dim: g.hidden, mode: 1) }
        stages.append { [self] in
            var gp = (UInt32(pickedRows), UInt32(g.hidden))
            engine.graph.dispatchThreads(pGather, threads: MTLSize(width: g.hidden, height: pickedRows, depth: 1)) { e in
                e.setBuffer(dn, offset: 0, index: 0)
                e.setBuffer(bGather, offset: 0, index: 1)
                e.setBuffer(selected, offset: 0, index: 2)
                e.setBytes(&gp, length: 8, index: 3)
            }
        }
        stages.append { [self] in
            if !debugSkip.contains("lmhead") {
                engine.gemm(engine.weights.lmHead, input: selected, output: logits, residual: false, rows: pickedRows)
            }
        }
        try engine.beginPass(label: "splosh.draft.block")
        for stage in stages.prefix(debugStageLimit ?? stages.count) {
            stage()
            engine.stage()
        }
        stageCount = stages.count
        gpuSeconds += try engine.graph.commitAndWait()

        let selectStart = DispatchTime.now().uptimeNanoseconds
        defer { selectSeconds += Double(DispatchTime.now().uptimeNanoseconds - selectStart) / 1e9 }
        return requests.indices.map {
            select(firstRow: firstPicked[$0], drafted: requests[$0].block - 1, anchor: requests[$0].anchor, temperature: temperature, vocabLimit: vocabLimit, random: &random)
        }
    }

    /// CandidateSelector.select: unary logits of the top candidates plus a low-rank pairwise
    /// term between the previous choice and each candidate, traced left to right.
    private func select(firstRow: Int, drafted: Int, anchor: Int, temperature: Float, vocabLimit: Int, random: inout SplitMix) -> DraftProposal {
        let g = engine.g
        let k = Self.candidateCount, rank = Self.rank
        let allLogits = logits.contents().assumingMemoryBound(to: Float.self)
        let hiddenStates = selected.contents().assumingMemoryBound(to: Float.self)
        let codebooks = file.base.assumingMemoryBound(to: UInt8.self)
        func codebook(_ offset: Int, _ token: Int) -> [Float] {
            let row = UnsafeRawPointer(codebooks).advanced(by: offset + token * rank * 2).assumingMemoryBound(to: UInt16.self)
            return (0..<rank).map { Float(bitPattern: UInt32(row[$0]) << 16) }
        }
        var predecessor = anchor
        var tokens: [Int] = [], candidateRows: [[Int]] = [], probabilityRows: [[Float]] = []
        var projected = [Float](repeating: 0, count: rank)
        for position in 0..<drafted {
            let row = firstRow + position
            let rowLogits = allLogits.advanced(by: row * g.vocab)
            let t0 = DispatchTime.now().uptimeNanoseconds
            let (ids, values) = TopK.select(rowLogits, limit: min(vocabLimit, g.vocab), k: k)
            topKSeconds += Double(DispatchTime.now().uptimeNanoseconds - t0) / 1e9
            hiddenProjection.withUnsafeBufferPointer { matrix in
                cblas_sgemv(CblasRowMajor, CblasNoTrans, Int32(rank), Int32(g.hidden), 1, matrix.baseAddress!, Int32(g.hidden),
                            hiddenStates.advanced(by: row * g.hidden), 1, 0, &projected, 1)
            }
            let before = codebook(predecessorOffset, predecessor)
            var gated = [Float](repeating: 0, count: rank)
            vDSP_vmul(before, 1, projected, 1, &gated, 1, vDSP_Length(rank))
            var scores = values
            for candidate in 0..<k {
                var pairwise: Float = 0
                vDSP_dotpr(gated, 1, codebook(successorOffset, ids[candidate]), 1, &pairwise, vDSP_Length(rank))
                scores[candidate] += pairwise
            }
            var choice = 0
            if temperature > 0 {
                let top = scores.max() ?? 0
                var q = scores.map { exp(($0 - top) / temperature) }
                let total = q.reduce(0, +)
                for index in q.indices { q[index] /= total }
                var draw = random.uniform()
                choice = k - 1
                for index in 0..<k { draw -= q[index]; if draw <= 0 { choice = index; break } }
                probabilityRows.append(q)
            } else {
                for index in 1..<k where scores[index] > scores[choice] { choice = index }
            }
            predecessor = ids[choice]
            tokens.append(predecessor)
            candidateRows.append(ids)
        }
        return DraftProposal(tokens: tokens, candidates: candidateRows, probabilities: temperature > 0 ? probabilityRows : nil)
    }

    // MARK: - Encoding

    private struct ConvParams { var rows, hidden, groupSize, groups, which, accumulate: UInt32 }
    private struct KVParams { var rows, kvHeads, capacity: UInt32; var eps, theta: Float }
    private struct AttentionParams { var rows, heads, kvHeads, capacity, window: UInt32; var scale, eps, theta: Float; var spans: UInt32 }

    private func conv(input: MTLBuffer, base: TensorHandle, output: MTLBuffer, which: UInt32, accumulate: Bool, rows: Int) {
        let g = engine.g
        var p = ConvParams(rows: UInt32(rows), hidden: UInt32(g.hidden),
                           groupSize: UInt32(Self.groupSize), groups: UInt32(Self.groups), which: which, accumulate: accumulate ? 1 : 0)
        engine.graph.dispatchThreads(pConv, threads: MTLSize(width: g.hidden, height: rows, depth: 1)) { e in
            e.setBuffer(input, offset: 0, index: 0)
            e.setBuffer(dyn, offset: 0, index: 1)
            engine.bind(e, base, 2)
            e.setBuffer(output, offset: 0, index: 3)
            e.setBytes(&p, length: MemoryLayout<ConvParams>.stride, index: 4)
            e.setBuffer(bOffset, offset: 0, index: 5)
        }
    }

    private func storeKV(_ layer: Layer, index: Int, k: MTLBuffer, v: MTLBuffer, slots: MTLBuffer, positions: MTLBuffer, rows: Int) {
        var p = KVParams(rows: UInt32(rows), kvHeads: UInt32(Self.kvHeads), capacity: UInt32(Self.ringCapacity),
                         eps: engine.g.normEps, theta: engine.g.ropeTheta)
        engine.graph.dispatch(pKVStore, grid: MTLSize(width: Self.kvHeads, height: rows, depth: 1), threadsPerGroup: engine.groupSize) { e in
            e.setBuffer(k, offset: 0, index: 0)
            e.setBuffer(v, offset: 0, index: 1)
            engine.bind(e, layer.kNorm, 2)
            e.setBuffer(slots, offset: 0, index: 3)
            e.setBuffer(positions, offset: 0, index: 4)
            e.setBuffer(kRing[index], offset: 0, index: 5)
            e.setBuffer(vRing[index], offset: 0, index: 6)
            e.setBytes(&p, length: MemoryLayout<KVParams>.stride, index: 7)
        }
    }

    private struct QueryPrepareParams { var rows, heads, kvHeads, rowCap: UInt32; var eps, theta: Float }
    private struct AcceleratorAttentionParams { var requests, heads, kvHeads, capacity, window, spans, rowCap: UInt32; var scale: Float }

    /// Queries for the accelerator attention; shares a stage with the block's KV store.
    private func prepareQueries(_ layer: Layer, rows: Int) {
        var p = QueryPrepareParams(rows: UInt32(rows), heads: UInt32(Self.heads), kvHeads: UInt32(Self.kvHeads),
                                   rowCap: UInt32(blockRows), eps: engine.g.normEps, theta: engine.g.ropeTheta)
        engine.graph.dispatch(pQueryPrepare, grid: MTLSize(width: Self.heads, height: rows, depth: 1),
                              threadsPerGroup: MTLSize(width: Self.headDim, height: 1, depth: 1)) { e in
            e.setBuffer(qp, offset: 0, index: 0)
            engine.bind(e, layer.qNorm, 1)
            e.setBuffer(bPos, offset: 0, index: 2)
            e.setBuffer(preparedQueries, offset: 0, index: 3)
            e.setBytes(&p, length: MemoryLayout<QueryPrepareParams>.stride, index: 4)
        }
    }

    private func attention(_ layer: Layer, index: Int, rows: Int) {
        var p = AttentionParams(rows: UInt32(rows), heads: UInt32(Self.heads), kvHeads: UInt32(Self.kvHeads),
                                capacity: UInt32(Self.ringCapacity), window: UInt32(Self.window),
                                scale: 1 / Float(Self.headDim).squareRoot(), eps: engine.g.normEps, theta: engine.g.ropeTheta,
                                spans: UInt32(Self.attentionSpans))
        // The window is scanned as parallel spans and merged; see draft.metal.
        if attentionOnAccelerator {
            var ap = AcceleratorAttentionParams(requests: UInt32(rows / Self.blockSize), heads: UInt32(Self.heads), kvHeads: UInt32(Self.kvHeads),
                                                capacity: UInt32(Self.ringCapacity), window: UInt32(Self.window),
                                                spans: UInt32(Self.attentionSpans), rowCap: UInt32(blockRows), scale: p.scale)
            engine.graph.dispatch(pAttention, grid: MTLSize(width: Self.kvHeads * Self.attentionSpans, height: rows / Self.blockSize, depth: 1),
                                  threadsPerGroup: MTLSize(width: 256, height: 1, depth: 1)) { e in
                e.setBuffer(preparedQueries, offset: 0, index: 0)
                e.setBuffer(bSlot, offset: 0, index: 1)
                e.setBuffer(bPos, offset: 0, index: 2)
                e.setBuffer(bLo, offset: 0, index: 3)
                e.setBuffer(bHi, offset: 0, index: 4)
                e.setBuffer(kRing[index], offset: 0, index: 5)
                e.setBuffer(vRing[index], offset: 0, index: 6)
                e.setBuffer(attentionPartials, offset: 0, index: 7)
                e.setBytes(&ap, length: MemoryLayout<AcceleratorAttentionParams>.stride, index: 8)
            }
        } else {
        engine.graph.dispatch(pAttention, grid: MTLSize(width: Self.heads, height: rows, depth: Self.attentionSpans),
                              threadsPerGroup: MTLSize(width: Self.headDim, height: 1, depth: 1)) { e in
            e.setBuffer(qp, offset: 0, index: 0)
            engine.bind(e, layer.qNorm, 1)
            e.setBuffer(bSlot, offset: 0, index: 2)
            e.setBuffer(bPos, offset: 0, index: 3)
            e.setBuffer(bLo, offset: 0, index: 4)
            e.setBuffer(bHi, offset: 0, index: 5)
            e.setBuffer(kRing[index], offset: 0, index: 6)
            e.setBuffer(vRing[index], offset: 0, index: 7)
            e.setBuffer(attentionPartials, offset: 0, index: 8)
            e.setBytes(&p, length: MemoryLayout<AttentionParams>.stride, index: 9)
        }
        }
        engine.stage()
        engine.graph.dispatch(pAttentionMerge, grid: MTLSize(width: Self.heads, height: rows, depth: 1),
                              threadsPerGroup: MTLSize(width: Self.headDim, height: 1, depth: 1)) { e in
            e.setBuffer(attentionPartials, offset: 0, index: 0)
            e.setBuffer(ao, offset: 0, index: 1)
            e.setBytes(&p, length: MemoryLayout<AttentionParams>.stride, index: 2)
        }
    }

    // MARK: - Quantisation

    /// MLX affine q4, group 64: per group `scale = (max - min) / 15`, `bias = min`, both stored
    /// as BF16, and codes rounded against the stored (rounded) parameters.
    /// Affine q4 in groups of 64, as MLX does it. `tiled` writes the layout of Retile.swift,
    /// [tile of 128 rows][group][row in tile], instead of row-major.
    private static func quantize(source: UnsafePointer<UInt16>, rows: Int, inner: Int, tiled: Bool,
                                 packed: UnsafeMutablePointer<UInt32>, scales: UnsafeMutablePointer<UInt16>,
                                 biases: UnsafeMutablePointer<UInt16>) {
        let groups = inner / 64, words = inner / 8, tileRows = Converter.tileRows
        let fitted = rangeRule == "fitted"
        nonisolated(unsafe) let source = source, packed = packed, scales = scales, biases = biases
        DispatchQueue.concurrentPerform(iterations: rows) { row in
            var values = [Float](repeating: 0, count: 64)
            for group in 0..<groups {
                let input = source.advanced(by: row * inner + group * 64)
                var lowest = Float.infinity, highest = -Float.infinity
                for index in 0..<64 {
                    let value = Float(bitPattern: UInt32(input[index]) << 16)
                    values[index] = value
                    lowest = min(lowest, value); highest = max(highest, value)
                }
                var scaleBits = bf16(highest > lowest ? (highest - lowest) / 15 : 1)
                var biasBits = bf16(lowest)
                if fitted, highest > lowest {
                    (scaleBits, biasBits) = fitRange(values, lowest: lowest, highest: highest, start: (scaleBits, biasBits))
                }
                let scale = Float(bitPattern: UInt32(scaleBits) << 16), bias = Float(bitPattern: UInt32(biasBits) << 16)
                let slot = tiled ? ((row / tileRows) * groups + group) * tileRows + row % tileRows : row * groups + group
                scales[slot] = scaleBits
                biases[slot] = biasBits
                for word in 0..<8 {
                    var bits: UInt32 = 0
                    for nibble in 0..<8 {
                        let code = min(15, max(0, ((values[word * 8 + nibble] - bias) / scale).rounded()))
                        bits |= UInt32(code) << UInt32(nibble * 4)
                    }
                    packed[(tiled ? slot * 8 : row * words + group * 8) + word] = bits
                }
            }
        }
    }

    /// How a group's range is chosen. "minmax" is MLX's rule: the whole 4-bit range spans the
    /// group's extremes, so one outlier coarsens the step for the other 63 values. "fitted" (the
    /// default) tries a few clipped ranges, refits scale and bias to the codes each gives by
    /// least squares, and keeps the one with the least squared error (after bf16 rounding). The
    /// format is the same; only what the draft proposes can change, never what is generated.
    /// Measured over thirteen prompts (SPLOSH_BLOCK_STUDY, same anchors): 4.65 -> 4.75 tokens a
    /// step for the isolated double block, better on ten; quantising takes 2.7 s instead of 0.8.
    /// SPLOSH_DRAFT_QUANT=minmax restores MLX's rule.
    static let rangeRule = ProcessInfo.processInfo.environment["SPLOSH_DRAFT_QUANT"] ?? "fitted"

    private static func fitRange(_ x: [Float], lowest: Float, highest: Float, start: (UInt16, UInt16)) -> (UInt16, UInt16) {
        func float(_ bits: UInt16) -> Float { Float(bitPattern: UInt32(bits) << 16) }
        func error(_ parameters: (UInt16, UInt16)) -> Float {
            let scale = float(parameters.0), bias = float(parameters.1)
            guard scale > 0, scale.isFinite, bias.isFinite else { return .infinity }
            var sum: Float = 0
            for value in x {
                let code = min(15, max(0, ((value - bias) / scale).rounded()))
                let difference = value - (bias + scale * code)
                sum += difference * difference
            }
            return sum
        }
        var best = start, bestError = error(start)
        let range = highest - lowest, count = Float(x.count)
        for low: Float in [0, 0.04, 0.08] {
            for high: Float in [0, 0.04, 0.08] {
                var scale = (range * (1 - low - high)) / 15, bias = lowest + low * range
                for _ in 0..<2 {
                    var sumCode: Float = 0, sumValue: Float = 0, sumCode2: Float = 0, sumCodeValue: Float = 0
                    for value in x {
                        let code = min(15, max(0, ((value - bias) / scale).rounded()))
                        sumCode += code; sumValue += value; sumCode2 += code * code; sumCodeValue += code * value
                    }
                    let determinant = count * sumCode2 - sumCode * sumCode
                    guard determinant > 0 else { break }
                    let refitted = (count * sumCodeValue - sumCode * sumValue) / determinant
                    guard refitted > 0 else { break }
                    scale = refitted
                    bias = (sumValue - refitted * sumCode) / count
                }
                let candidate = (bf16(scale), bf16(bias))
                let candidateError = error(candidate)
                if candidateError < bestError { best = candidate; bestError = candidateError }
            }
        }
        return best
    }

    private static func bf16(_ value: Float) -> UInt16 {
        let bits = value.bitPattern
        return UInt16(truncatingIfNeeded: (bits &+ 0x7FFF &+ ((bits >> 16) & 1)) >> 16)
    }
}

/// SplitMix64, shared by the sampler and the draft selector.
public struct SplitMix: Sendable {
    private var state: UInt64
    public init(seed: UInt64) { state = seed }
    public mutating func uniform() -> Float {
        state &+= 0x9E37_79B9_7F4A_7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        z ^= z >> 31
        return Float(z >> 40) / Float(1 << 24)
    }
}

/// A read-only memory mapping of a whole file.
final class MappedFile: @unchecked Sendable {
    let base: UnsafeRawPointer
    let length: Int

    init(url: URL) throws {
        let descriptor = open(url.path, O_RDONLY)
        guard descriptor >= 0 else { throw DraftError.invalid("cannot open \(url.path)") }
        defer { close(descriptor) }
        var info = stat()
        guard fstat(descriptor, &info) == 0, info.st_size > 16,
              let mapped = mmap(nil, Int(info.st_size), PROT_READ, MAP_PRIVATE, descriptor, 0),
              mapped != UnsafeMutableRawPointer(bitPattern: UInt(bitPattern: -1)) else {
            throw DraftError.invalid("cannot map \(url.path)")
        }
        base = UnsafeRawPointer(mapped)
        length = Int(info.st_size)
    }

    deinit { munmap(UnsafeMutableRawPointer(mutating: base), length) }
}

/// The tensor directory of a safetensors file.
struct SafetensorsHeader {
    struct Entry { let dtype: String; let shape: [Int]; let offset: Int; let length: Int }
    private let entries: [String: Entry]

    init(file: MappedFile) throws {
        let headerLength = Int(file.base.loadUnaligned(as: UInt64.self))
        guard headerLength > 0, headerLength <= file.length - 8 else { throw DraftError.invalid("bad safetensors header") }
        let data = Data(bytes: file.base.advanced(by: 8), count: headerLength)
        guard let object = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw DraftError.invalid("safetensors header is not an object")
        }
        var parsed: [String: Entry] = [:]
        for (name, value) in object where name != "__metadata__" {
            guard let tensor = value as? [String: Any], let dtype = tensor["dtype"] as? String,
                  let shape = tensor["shape"] as? [Int], let offsets = tensor["data_offsets"] as? [Int], offsets.count == 2,
                  offsets[0] >= 0, offsets[1] >= offsets[0], 8 + headerLength + offsets[1] <= file.length else {
                throw DraftError.invalid("malformed tensor entry \(name)")
            }
            parsed[name] = Entry(dtype: dtype, shape: shape, offset: 8 + headerLength + offsets[0], length: offsets[1] - offsets[0])
        }
        entries = parsed
    }

    func require(_ name: String, dtype: String, shape: [Int]) throws -> Entry {
        guard let entry = entries[name] else { throw DraftError.invalid("missing tensor \(name)") }
        guard entry.dtype == dtype, entry.shape == shape else {
            throw DraftError.invalid("\(name) is \(entry.dtype) \(entry.shape); expected \(dtype) \(shape)")
        }
        return entry
    }
}
