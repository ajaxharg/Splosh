// GgufNames.swift — a Qwen3.5-family GGUF's tensor names in Splosh's terms, the MTP block, and the
// value-head order of its gated-delta layers.
//
// Splosh names tensors as the MLX pack does: `language_model.model.layers.<n>.<module>`, with the
// `.weight` of a module left off in a base name. The names below are those of the pack, and each
// GGUF tensor maps to exactly one of them. tools/gguf_check.py established the mapping by decoding
// the same tensors from both and comparing them.

import Foundation

public enum GgufNames {
    /// Splosh's name, below `language_model.model.layers.<n>.`, for each tensor of a block, keyed by
    /// the tensor's name below `blk.<n>.` in the file. `ssm_a` holds -exp(A_log), not A_log itself.
    public static let blockTensors: [String: String] = [
        "attn_norm.weight": "input_layernorm",
        "post_attention_norm.weight": "post_attention_layernorm",
        "attn_qkv.weight": "linear_attn.in_proj_qkv",
        "attn_gate.weight": "linear_attn.in_proj_z",
        "ssm_alpha.weight": "linear_attn.in_proj_a",
        "ssm_beta.weight": "linear_attn.in_proj_b",
        "ssm_out.weight": "linear_attn.out_proj",
        "ssm_norm.weight": "linear_attn.norm",
        "ssm_conv1d.weight": "linear_attn.conv1d",
        "ssm_a": "linear_attn.A_log",
        "ssm_dt.bias": "linear_attn.dt_bias",
        "attn_q.weight": "self_attn.q_proj",
        "attn_k.weight": "self_attn.k_proj",
        "attn_v.weight": "self_attn.v_proj",
        "attn_output.weight": "self_attn.o_proj",
        "attn_q_norm.weight": "self_attn.q_norm",
        "attn_k_norm.weight": "self_attn.k_norm",
        "ffn_gate.weight": "mlp.gate_proj",
        "ffn_up.weight": "mlp.up_proj",
        "ffn_down.weight": "mlp.down_proj",
    ]

    /// Splosh's name for each tensor outside the blocks.
    public static let globalTensors: [String: String] = [
        "token_embd.weight": "language_model.model.embed_tokens",
        "output.weight": "language_model.lm_head",
        "output_norm.weight": "language_model.model.norm",
    ]

    /// Splosh's base name for a GGUF tensor, nil for a tensor with no counterpart (the `nextn`
    /// tensors of the MTP block, or a name this model does not have). The mapping is by name alone:
    /// the MTP block's ordinary tensors do map, to a layer the model does not have, so a caller that
    /// skips the MTP block asks `belongsToMtp` first.
    public static func sploshName(forGguf name: String) -> String? {
        if let global = globalTensors[name] { return global }
        guard let (block, tail) = split(name), let module = blockTensors[String(tail)] else { return nil }
        return "language_model.model.layers.\(block).\(module)"
    }

    /// The block number of a `blk.<n>.…` name; nil for any other.
    public static func blockIndex(of name: String) -> Int? { split(name)?.block }

    /// The block that is the MTP head: the last `blk.<n>` when any tensor name contains `.nextn.`,
    /// nil when none does. It is numbered after the model's layers and is not one of them.
    public static func mtpBlock(among names: some Sequence<String>) -> Int? {
        var last: Int?, hasNextn = false
        for name in names {
            if name.contains(".nextn.") { hasNextn = true }
            if let block = blockIndex(of: name), block > (last ?? -1) { last = block }
        }
        return hasNextn ? last : nil
    }

    /// Whether `name` is a tensor of the block `mtpBlock`, as `mtpBlock(among:)` found it.
    public static func belongsToMtp(_ name: String, mtpBlock: Int?) -> Bool {
        guard let mtpBlock, let block = blockIndex(of: name) else { return false }
        return block == mtpBlock
    }

    /// The index in the MLX pack of the gated-delta value head that the GGUF stores as head `i`
    /// of 48: `3 * (i % 16) + i / 16`. A tensor is converted by moving GGUF head `i` to head
    /// `mlxValueHead(ofGgufHead: i)`. The permutation applies to:
    ///   - the rows of `ssm_alpha`, `ssm_beta` (one row a head) and `attn_gate` (128 rows a head);
    ///   - the value rows of `attn_qkv`, from row 4096, 128 rows a head;
    ///   - the value channels of `ssm_conv1d`, from channel 4096, 128 channels a head;
    ///   - `ssm_a` and `ssm_dt.bias` (one element a head);
    ///   - the columns of `ssm_out`, 128 columns a head.
    /// Nothing else is permuted: not the query and key rows of `attn_qkv`, not `ssm_norm`, which is
    /// the same 128 values for every head.
    public static func mlxValueHead(ofGgufHead i: Int) -> Int { 3 * (i % 16) + i / 16 }

    /// Split `blk.<n>.<tail>` into n and the tail.
    private static func split(_ name: String) -> (block: Int, tail: Substring)? {
        guard name.hasPrefix("blk.") else { return nil }
        let rest = name.dropFirst(4)
        let digits = rest.prefix { $0.isASCII && $0.isNumber }
        let tail = rest.dropFirst(digits.count)
        guard !digits.isEmpty, tail.first == ".", let block = Int(digits) else { return nil }
        return (block, tail.dropFirst())
    }
}
