// NormOracle.swift — SploshOracle.
//
// Owner: M2.3 lands the `norm(...)` entry point. M0.1 creates the file with the one constant
// M0.6's non-vacuity gate exercises, so the filter is real from M0 rather than empty.
// Contract source: rev4 §4.7 (NormOracle), §2.2 (`:147`).

/// Scalar reference for the model's zero-centred RMSNorm (rev4 §4.7).
///
/// rev4 §4.7: `out = x_hat * (1 + weight)` with the variance accumulated in fp32 — `1 + weight`,
/// never `weight` alone (§2.3). Public entry point `NormOracle.norm(...)`, landed by M2.3.
public enum NormOracle {
    /// `rms_norm_eps` read from the pack's own config (rev4 §2.2, `:147`): `1e-06`.
    public static let rmsNormEps: Float = 1e-06

    /// Apply zero-centred RMSNorm using fp32 accumulators.
    ///
    /// The checkpoint stores a delta weight, so the scale is `1 + weight`. Inputs and
    /// weights are accepted as Float values to make the reference independent of Metal.
    public static func norm(
        _ input: [Float],
        weight: [Float],
        epsilon: Float = rmsNormEps
    ) -> [Float] {
        precondition(input.count == weight.count, "NormOracle input and weight shapes must match")
        guard !input.isEmpty else { return [] }

        var sumSquares: Float = 0
        for value in input {
            sumSquares += value * value
        }
        let inverseRMS = 1 / (sumSquares / Float(input.count) + epsilon).squareRoot()
        return zip(input, weight).map { value, delta in
            value * inverseRMS * (1 + delta)
        }
    }
}
