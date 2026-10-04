import Foundation

/// Validated full-attention projection layout and buffer ordering.
public struct AttentionLayer: Sendable, Equatable {
    public let dense: AttentionDense
    public init(dense: AttentionDense? = nil) {
        // The public default geometry is guaranteed valid.
        self.dense = dense ?? AttentionDense()!
    }

    /// Metal buffer ABI order (all tensors are contiguous):
    /// 0 raw interleaved qProjection (`[query | gate]` per head),
    /// 1 q RMS-norm weights, 2 k RMS-norm weights,
    /// 3 token-major int8 K, 4 token-major int8 V,
    /// 5 per-KV-head fp32 K scales, 6 per-KV-head fp32 V scales,
    /// 7 optional dense row-major o_proj, 8 output, 9 Parameters.
    public enum Buffer: Int, Sendable {
        case qProjection = 0, qNorm = 1, kNorm = 2, keys = 3, values = 4
        case keyScales = 5, valueScales = 6, oProjection = 7, output = 8, parameters = 9
    }

    public var qProjectionWidth: Int { dense.qProjectionWidth }
    public var outputWidth: Int { dense.queryWidth }
    public static let gate = "sigmoid"
    public static let order = ["qk_rmsnorm", "rope", "causal_gqa", "sigmoid_gate", "o_proj"]
}
