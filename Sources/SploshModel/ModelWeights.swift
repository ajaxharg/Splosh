// ModelWeights.swift — the text model resolved against a resident SPLW artifact.
//
// Names and shapes follow audit/QWEN35-REFERENCE-MATH.md §7. Only `language_model.*` tensors
// are made GPU-resident: Splosh serves text, for which mRoPE reduces to RoPE and the vision
// tower is never evaluated.
//
// An artifact converted from a GGUF file (GgufConvert.swift) has the same names, shapes and row
// order, and resolves into the same structure. Its packed weights are planes of llama.cpp's
// formats, a format a weight, so each `Q4Handle` names its kernels and `codeBits` is 0. One of
// its weights differs in more than format: a gated-delta layer's `out_proj` has its columns in
// the file's order of the value heads, for the `heads` kernels of engine_gguf.metal.

import Foundation
@preconcurrency import Metal
import SploshCore

public enum ModelWeightsError: Error, CustomStringConvertible {
    case shape(String, expected: [Int], observed: [Int])
    case unsupported(String)

    public var description: String {
        switch self {
        case .shape(let name, let expected, let observed):
            return "tensor \(name) has shape \(observed); expected \(expected)"
        case .unsupported(let reason): return "unsupported model artifact: \(reason)"
        }
    }
}

public struct LinearMixerWeights: Sendable {
    public let qkv: Q4Handle
    public let a: Q4Handle
    public let b: Q4Handle
    public let z: Q4Handle
    public let out: Q4Handle
    public let conv: TensorHandle
    public let aLog: TensorHandle
    public let dtBias: TensorHandle
    public let norm: TensorHandle
}

public struct FullMixerWeights: Sendable {
    public let q: Q4Handle
    public let k: Q4Handle
    public let v: Q4Handle
    public let o: Q4Handle
    public let qNorm: TensorHandle
    public let kNorm: TensorHandle
}

public enum MixerWeights: Sendable {
    case linear(LinearMixerWeights)
    case full(FullMixerWeights)
}

public struct BlockWeights: Sendable {
    public let inputNorm: TensorHandle
    public let postNorm: TensorHandle
    public let gate: Q4Handle
    public let up: Q4Handle
    public let down: Q4Handle
    public let mixer: MixerWeights
}

/// Fixed geometry of the served checkpoint, validated against the artifact at load.
public struct ModelGeometry: Sendable, Equatable {
    public let hidden = 5120
    public let intermediate = 17408
    public let vocab = 248320
    public let layers = 64
    // Full attention
    public let heads = 24
    public let kvHeads = 4
    public let headDim = 256
    public let rotaryDim = 64
    public let ropeTheta: Float = 10_000_000
    // Gated DeltaNet
    public let gdnKeyHeads = 16
    public let gdnValueHeads = 48
    public let gdnHeadDim = 128
    public let gdnChannels = 10240
    public let gdnValueDim = 6144
    public let normEps: Float = 1e-6
    public init() {}

    public var linearLayerCount: Int { 48 }
    public var fullLayerCount: Int { 16 }
}

public struct ModelWeights: Sendable {
    public static let prefix = "language_model."

    public let resident: ResidentWeights
    public let geometry: ModelGeometry
    public let embed: Q4Handle
    public let lmHead: Q4Handle
    public let finalNorm: TensorHandle
    public let blocks: [BlockWeights]
    public let configHash: String
    /// True when the artifact stores eligible q4 weights in the tiled layout (see Retile.swift).
    /// A GGUF-derived artifact is tiled throughout.
    public let tiledLayout: Bool
    /// Bits per weight code, the same for every packed weight of the artifact: 4 or 8. A
    /// GGUF-derived artifact has no one width and reports 0, so that nothing takes its planes
    /// for affine codes: each of its handles names its format's kernels instead (`Q4Handle.kernel`).
    public let codeBits: Int
    /// True when the artifact was converted from a GGUF file (GgufConvert.swift): its packed
    /// weights are planes of llama.cpp's formats, for the kernels of engine_gguf.metal.
    public let ggufDerived: Bool
    public let tokenizerHash: String

    public init(device: MTLDevice, artifactURL: URL) throws {
        let file = try WeightFile(splwURL: artifactURL)
        let header = file.header
        let layout = try Self.layout(of: header)
        tiledLayout = layout.tiled
        codeBits = layout.bits
        ggufDerived = file.isGguf
        let records = header.tensorRecords
            .filter { $0.name.hasPrefix(Self.prefix) }
            .map { ResidentTensorRecord(name: $0.name, payloadOffset: Int($0.payloadOffset),
                                        byteLength: Int($0.rawByteLength), shape: $0.shape) }
        guard let first = header.tensorRecords.map(\.payloadOffset).min() else {
            throw ModelWeightsError.unsupported("artifact has no tensors")
        }
        let resident = try ResidentWeights(device: device, records: records, containerURL: artifactURL,
                                           payloadStart: Int(first),
                                           payloadLength: Int(file.fileSize) - Int(first))
        let tensors = try Self.resolve(file, bits: layout.bits, lookup: resident.handle)

        self.resident = resident
        self.geometry = ModelGeometry()
        self.embed = tensors.embed
        self.lmHead = tensors.lmHead
        self.finalNorm = tensors.finalNorm
        self.blocks = tensors.blocks
        self.configHash = header.configHash
        self.tokenizerHash = header.tokenizerHash
    }

    /// The names of the records the model resolves from an artifact, found by resolving every
    /// tensor against the header as `init` does, with nothing mapped: what `init` would throw
    /// for a missing or mis-shaped tensor is thrown here. It is how an artifact too large to
    /// open beside a live server is checked.
    public static func recordsUsed(in file: WeightFile) throws -> Set<String> {
        let layout = try layout(of: file.header)
        var handles: [String: TensorHandle] = [:]
        for record in file.header.tensorRecords where record.name.hasPrefix(prefix) {
            handles[record.name] = TensorHandle(name: record.name, span: 0, offset: Int(record.payloadOffset),
                                                byteLength: Int(record.rawByteLength), shape: record.shape)
        }
        var used = Set<String>()
        _ = try resolve(file, bits: layout.bits) { name in
            guard let handle = handles[name] else { throw ResidentWeightsError.missingTensor(name) }
            used.insert(name)
            return handle
        }
        return used
    }

    /// What a header's `format` says of its packed weights.
    private static func layout(of header: ConverterHeader) throws -> (bits: Int, tiled: Bool) {
        if header.format == Converter.ggufFormat { return (0, true) }
        guard let layout = Converter.layout(ofFormat: header.format), layout.bits == header.quantization.bits else {
            throw ModelWeightsError.unsupported("format \(header.format)")
        }
        return layout
    }

    private struct Tensors {
        let embed: Q4Handle
        let lmHead: Q4Handle
        let finalNorm: TensorHandle
        let blocks: [BlockWeights]
    }

    /// Every tensor of the model by name, each checked against the geometry. `lookup` gives a
    /// record's handle: the resident mapping's, or one made from the header.
    private static func resolve(_ file: WeightFile, bits: Int, lookup: (String) throws -> TensorHandle) throws -> Tensors {
        let g = ModelGeometry()
        let gguf = file.isGguf

        func q4(_ name: String, rows: Int, inner: Int) throws -> Q4Handle {
            if gguf {
                // The header carries what the planes' shapes do not: the format and the
                // logical shape. `file.gguf` has checked each plane's size against them.
                let record = try file.gguf(Self.prefix + name + ".weight")
                guard record.logicalShape == [rows, inner], let geometry = GgufPlanes.geometry(of: record.type) else {
                    throw ModelWeightsError.shape(name, expected: [rows, inner], observed: record.logicalShape)
                }
                return try ResidentWeights.gguf(record.name, kernel: geometry.kernelSuffix, rows: rows, inner: inner,
                                                secondPlane: record.plane1 != nil, handle: lookup)
            }
            let handle = try ResidentWeights.q4(Self.prefix + name + ".weight", handle: lookup)
            guard handle.rows == rows, handle.inner == inner, handle.groupsPerRow * 64 == inner else {
                throw ModelWeightsError.shape(name, expected: [rows, inner], observed: [handle.rows, handle.inner])
            }
            guard handle.bits == bits else {
                throw ModelWeightsError.unsupported("\(name) is \(handle.bits)-bit in a \(bits)-bit artifact")
            }
            return handle
        }
        func dense(_ name: String, _ shape: [Int]) throws -> TensorHandle {
            let handle = try lookup(Self.prefix + name)
            guard handle.shape == shape else {
                throw ModelWeightsError.shape(name, expected: shape, observed: handle.shape)
            }
            return handle
        }

        var blocks: [BlockWeights] = []
        for index in 0..<g.layers {
            let p = "model.layers.\(index)."
            let mixer: MixerWeights
            if index % 4 == 3 {
                mixer = .full(FullMixerWeights(
                    q: try q4(p + "self_attn.q_proj", rows: g.heads * g.headDim * 2, inner: g.hidden),
                    k: try q4(p + "self_attn.k_proj", rows: g.kvHeads * g.headDim, inner: g.hidden),
                    v: try q4(p + "self_attn.v_proj", rows: g.kvHeads * g.headDim, inner: g.hidden),
                    o: try q4(p + "self_attn.o_proj", rows: g.hidden, inner: g.heads * g.headDim),
                    qNorm: try dense(p + "self_attn.q_norm.weight", [g.headDim]),
                    kNorm: try dense(p + "self_attn.k_norm.weight", [g.headDim])))
            } else {
                mixer = .linear(LinearMixerWeights(
                    qkv: try q4(p + "linear_attn.in_proj_qkv", rows: g.gdnChannels, inner: g.hidden),
                    a: try q4(p + "linear_attn.in_proj_a", rows: g.gdnValueHeads, inner: g.hidden),
                    b: try q4(p + "linear_attn.in_proj_b", rows: g.gdnValueHeads, inner: g.hidden),
                    z: try q4(p + "linear_attn.in_proj_z", rows: g.gdnValueDim, inner: g.hidden),
                    out: try q4(p + "linear_attn.out_proj", rows: g.hidden, inner: g.gdnValueDim),
                    conv: try dense(p + "linear_attn.conv1d.weight", [g.gdnChannels, 4, 1]),
                    aLog: try dense(p + "linear_attn.A_log", [g.gdnValueHeads]),
                    dtBias: try dense(p + "linear_attn.dt_bias", [g.gdnValueHeads]),
                    norm: try dense(p + "linear_attn.norm.weight", [g.gdnHeadDim])))
            }
            blocks.append(BlockWeights(
                inputNorm: try dense(p + "input_layernorm.weight", [g.hidden]),
                postNorm: try dense(p + "post_attention_layernorm.weight", [g.hidden]),
                gate: try q4(p + "mlp.gate_proj", rows: g.intermediate, inner: g.hidden),
                up: try q4(p + "mlp.up_proj", rows: g.intermediate, inner: g.hidden),
                down: try q4(p + "mlp.down_proj", rows: g.hidden, inner: g.intermediate),
                mixer: mixer))
        }

        return Tensors(embed: try q4("model.embed_tokens", rows: g.vocab, inner: g.hidden),
                       lmHead: try q4("lm_head", rows: g.vocab, inner: g.hidden),
                       finalNorm: try dense("model.norm.weight", [g.hidden]),
                       blocks: blocks)
    }
}
