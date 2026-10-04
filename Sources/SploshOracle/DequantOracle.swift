// DequantOracle.swift — independent scalar q4 reference implementations.
// These are format contracts, not model fixtures. They consume bytes supplied by callers and
// never synthesize or substitute model weights.

import Foundation
import SploshQuant

public enum DequantOracle {
    /// MLX affine q4: row-major U32 words with low-first nibbles and group-64
    /// scale/bias sidecars. The result is row-major and remains Float32.
    public static func dequantize(_ tensor: MlxAffine) -> [Float] {
        let wordsPerRow = (tensor.columns + 7) / 8
        let groupsPerRow = (tensor.columns + MlxAffine.groupSize - 1) / MlxAffine.groupSize
        var output: [Float] = []
        output.reserveCapacity(tensor.rows * tensor.columns)
        for row in 0..<tensor.rows {
            for column in 0..<tensor.columns {
                let word = tensor.words[row * wordsPerRow + column / 8]
                let nibble = Float((word >> UInt32(4 * (column % 8))) & 0xF)
                let group = row * groupsPerRow + column / MlxAffine.groupSize
                output.append(tensor.scales[group] * nibble + tensor.biases[group])
            }
        }
        return output
    }

    public struct Q4_0Block: Sendable, Equatable {
        public let scale: UInt16
        public let qs: [UInt8]
        public init(scale: UInt16, qs: [UInt8]) {
            precondition(qs.count == 16, "q4_0 block requires 16 packed bytes")
            self.scale = scale; self.qs = qs
        }
    }

    /// GGML q4_0: 32 values, fp16 d, and 16 bytes with x[j] in the low nibble
    /// and x[j+16] in the high nibble.
    public static func q4_0(_ blocks: [Q4_0Block]) -> [Float] {
        blocks.flatMap { block in
            let d = Float(Float16(bitPattern: block.scale))
            return (0..<16).flatMap { i -> [Float] in
                let byte = block.qs[i]
                return [d * (Float(byte & 0x0f) - 8), d * (Float(byte >> 4) - 8)]
            }
        }
    }

    /// Convenience form for a contiguous q4_0 payload (18 bytes per block).
    public static func q4_0(data: [UInt8]) -> [Float] {
        precondition(data.count % 18 == 0, "q4_0 payload must contain complete 18-byte blocks")
        return stride(from: 0, to: data.count, by: 18).flatMap { i in
            q4_0([Q4_0Block(scale: UInt16(data[i]) | UInt16(data[i + 1]) << 8,
                            qs: Array(data[(i + 2)..<(i + 18)]))])
        }
    }

    public struct Q4_KBlock: Sendable, Equatable {
        public let scale: UInt16
        public let min: UInt16
        public let scales: [UInt8]
        public let qs: [UInt8]
        public init(scale: UInt16, min: UInt16, scales: [UInt8], qs: [UInt8]) {
            precondition(scales.count == 12, "q4_K block requires 12 scale bytes")
            precondition(qs.count == 128, "q4_K block requires 128 packed bytes")
            self.scale = scale; self.min = min; self.scales = scales; self.qs = qs
        }
    }

    /// GGML q4_K: 256 values, 8 affine groups of 32. The 12-byte scale area uses
    /// six-bit scale/min fields for groups 0...3 and split fields for groups 4...7.
    public static func q4_K(_ blocks: [Q4_KBlock]) -> [Float] {
        blocks.flatMap { block in
            let d = Float(Float16(bitPattern: block.scale))
            let dmin = Float(Float16(bitPattern: block.min))
            var out = Array(repeating: Float.zero, count: 256)
            for group in 0..<8 {
                let sc: UInt8
                let mn: UInt8
                if group < 4 {
                    sc = block.scales[group] & 0x3f
                    mn = block.scales[group + 4] & 0x3f
                } else {
                    sc = (block.scales[group + 4] & 0x0f) | ((block.scales[group - 4] >> 6) << 4)
                    mn = (block.scales[group + 4] >> 4) | ((block.scales[group - 4] >> 4) & 0xF0)
                }
                let scale = d * Float(sc)
                let offset = dmin * Float(mn)
                let packedOffset = (group / 2) * 16
                let high = group % 2 == 1
                for i in 0..<16 {
                    let q = block.qs[packedOffset + i]
                    let nibble = high ? (q >> 4) : (q & 0x0f)
                    out[group * 32 + i] = scale * Float(nibble) - offset
                    let paired = high ? (q & 0x0f) : (q >> 4)
                    out[group * 32 + i + 16] = scale * Float(paired) - offset
                }
            }
            return out
        }
    }

    /// Convenience form for contiguous q4_K blocks (144 bytes each).
    public static func q4_K(data: [UInt8]) -> [Float] {
        precondition(data.count % 144 == 0, "q4_K payload must contain complete 144-byte blocks")
        return stride(from: 0, to: data.count, by: 144).flatMap { i in
            let s = UInt16(data[i]) | UInt16(data[i + 1]) << 8
            let m = UInt16(data[i + 2]) | UInt16(data[i + 3]) << 8
            return q4_K([Q4_KBlock(scale: s, min: m,
                                   scales: Array(data[(i + 4)..<(i + 16)]),
                                   qs: Array(data[(i + 16)..<(i + 144)]))])
        }
    }
}
