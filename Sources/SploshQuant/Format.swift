// Format.swift — SploshQuant.
//
// Owner: M0.1 creates the enum; M2.9 records the chosen case in the weight-file header and
// §4.2's invariant makes it part of the cache key. Contract source: rev4 §4.2 (WeightFormat).
//
// rev4 §4.2 names the three cases exactly. No raw values are assigned here: the on-disk
// spelling belongs to the header that M2.9 specifies, and inventing one early would create a
// second, unauthorised source of truth for a durable format.

/// The supported on-disk weight formats (rev4 §4.2).
public enum WeightFormat: Sendable, CaseIterable {
    /// MLX 4-bit affine: packed `U32` weights with separate `scales` and `biases` sidecars.
    case mlxAffine4
    /// `mxfp4` with 32-element blocks and `e8m0` scales (F1 Route B; see rev4 §1.2).
    case mxfp4_32_e8m0
    /// Dense bf16, used by the M2 full-precision path.
    case bf16
}
