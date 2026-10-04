// RopeOracle.swift — scalar partial-dimension/interleaved mRoPE reference.
// Contract: M2.3b / qwen35-hybrid-reference-math.md §2.3.

import Foundation

/// Independent CPU reference for Qwen's partial, interleaved mRoPE.
public enum RopeOracle {
    public static let defaultHeadDim = 256
    public static let defaultRotaryDim = 64
    public static let defaultRopeTheta: Float = 10_000_000
    public static let defaultSections = [11, 11, 10]

    /// Applies RoPE to one head. `positionIDs` are T/H/W positions; a two-element or
    /// one-element array is accepted as T-only input for ordinary text tokens.
    public static func apply(
        _ input: [Float],
        positionIDs: [Int],
        headDim: Int = defaultHeadDim,
        rotaryDim: Int = defaultRotaryDim,
        ropeTheta: Float = defaultRopeTheta,
        mropeSection: [Int] = defaultSections,
        attentionScaling: Float = 1
    ) -> [Float] {
        precondition(input.count == headDim, "RoPE input must have headDim elements")
        precondition(rotaryDim > 0 && rotaryDim <= headDim && rotaryDim.isMultiple(of: 2))
        precondition(positionIDs.count >= 1)
        precondition(mropeSection.count == 3)

        // This is the reference's apply_interleaved_mrope loop literally: start with
        // T and overwrite H/W slots with slice(offset, section * 3, 3).
        var positions = Array(repeating: positionIDs[0], count: rotaryDim / 2)
        if positionIDs.count >= 2 {
            for index in stride(from: 0, to: mropeSection[1] * 3, by: 3) where index + 1 < positions.count {
                positions[index + 1] = positionIDs[1]
            }
        }
        if positionIDs.count >= 3 {
            for index in stride(from: 0, to: mropeSection[2] * 3, by: 3) where index + 2 < positions.count {
                positions[index + 2] = positionIDs[2]
            }
        }

        var output = input
        let half = rotaryDim / 2
        for i in 0..<half {
            let exponent = Float(2 * i) / Float(rotaryDim)
            let inverseFrequency = 1 / pow(ropeTheta, exponent)
            let angle = Float(positions[i]) * inverseFrequency
            let cosine = cos(angle) * attentionScaling
            let sine = sin(angle) * attentionScaling
            // NeoX rotate_half: cat(-x[d/2:], x[:d/2]), not adjacent pairing.
            output[i] = input[i] * cosine - input[i + half] * sine
            output[i + half] = input[i + half] * cosine + input[i] * sine
        }
        return output
    }

    public static func apply(
        input: [Float],
        position: Int,
        headDim: Int = defaultHeadDim,
        rotaryDim: Int = defaultRotaryDim,
        ropeTheta: Float = defaultRopeTheta
    ) -> [Float] {
        apply(input, positionIDs: [position], headDim: headDim, rotaryDim: rotaryDim, ropeTheta: ropeTheta)
    }

    public static func apply(_ input: [Float], position: Int) -> [Float] {
        apply(input: input, position: position)
    }
}
