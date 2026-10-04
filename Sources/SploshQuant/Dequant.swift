// CPU affine dequantisation shared by conversion and later kernel checks.
import Foundation

public enum Dequant {
    /// Validate and convert the physical MLX safetensors q4 representation.
    /// The returned tensor exposes logical scalar columns while retaining packed words.
    public static func mlxAffine(physicalShape: [Int], packedWords: [UInt32], scales: [Float], biases: [Float]) throws -> MlxAffine {
        try MlxAffine(physicalShape: physicalShape, packedWords: packedWords, scales: scales, biases: biases)
    }

    public static func affine(q: [UInt8], scales: [Float], biases: [Float], groupSize: Int = 64) -> [Float] {
        guard !q.isEmpty, groupSize > 0 else { return [] }
        return q.enumerated().map { index, value in
            let group = index / groupSize
            guard group < scales.count, group < biases.count else { return 0 }
            return scales[group] * Float(value) + biases[group]
        }
    }
}
