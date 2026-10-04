import Foundation

/// Independent scalar reference for the dense Qwen full-attention block.
///
/// This implementation intentionally contains no Metal-facing code. It is the scalar
/// comparison authority for both decode and prefill, including their causal boundary.
public enum AttentionOracle {
    public struct Configuration: Sendable {
        public var heads: Int = 24
        public var kvHeads: Int = 4
        public var headDim: Int = 256
        public var epsilon: Float = 1e-6
        /// Rotary channels and the RoPE base are explicit so tests can mirror a model
        /// configuration rather than inheriting shader constants.
        public var rotaryDim: Int = RopeOracle.defaultRotaryDim
        public var ropeTheta: Float = RopeOracle.defaultRopeTheta
        /// Absolute position represented by cache element zero. This permits testing a
        /// page/cache slice whose first key is not token zero.
        public var positionOrigin: Int = 0
        public var position: Int = 0

        public init(heads: Int = 24, kvHeads: Int = 4, headDim: Int = 256,
                    epsilon: Float = 1e-6,
                    rotaryDim: Int = RopeOracle.defaultRotaryDim,
                    ropeTheta: Float = RopeOracle.defaultRopeTheta,
                    positionOrigin: Int = 0, position: Int = 0) {
            self.heads = heads; self.kvHeads = kvHeads; self.headDim = headDim
            self.epsilon = epsilon; self.rotaryDim = rotaryDim; self.ropeTheta = ropeTheta
            self.positionOrigin = positionOrigin; self.position = position
        }
    }

    private static func norm(_ x: ArraySlice<Float>, _ w: ArraySlice<Float>, _ eps: Float) -> [Float] {
        let mean = x.reduce(0) { $0 + $1 * $1 } / Float(x.count)
        let r = 1 / sqrt(mean + eps)
        return zip(x, w).map { $0 * r * (1 + $1) }
    }
    private static func dot(_ a: [Float], _ b: ArraySlice<Float>) -> Float {
        zip(a, b).reduce(0) { $0 + $1.0 * $1.1 }
    }
    private static func softmax(_ scores: [Float]) -> [Float] {
        // A row with no causal keys is empty, not a uniform distribution over masks.
        guard scores.contains(where: { $0 > -.greatestFiniteMagnitude / 2 }),
              let m = scores.max(), m.isFinite else {
            return Array(repeating: 0, count: scores.count)
        }
        let e = scores.map { exp($0 - m) }, s = e.reduce(0, +)
        return s.isFinite && s > 0 ? e.map { $0 / s } : Array(repeating: 0, count: scores.count)
    }

    /// Converts token-major signed-int8 KV data to scalar floats using one scale per
    /// KV head. This helper is deliberately a plain CPU operation for constructing
    /// deterministic test inputs; it does not call Metal or share shader code.
    public static func dequantizedInt8(_ values: [Int8], scales: [Float],
                                       kvHeads: Int, headDim: Int) -> [Float] {
        precondition(kvHeads > 0 && headDim > 0 && values.count % (kvHeads * headDim) == 0)
        precondition(scales.count == kvHeads && scales.allSatisfy { $0.isFinite })
        return values.enumerated().map { index, value in
            let head = (index / headDim) % kvHeads
            return Float(value) * scales[head]
        }
    }

    /// Computes one full attention row. qProjection is interleaved per head as
    /// `[query headDim | gate headDim]`; k/v are token-major KV heads.
    public static func layer(qProjection: [Float], keys: [Float], values: [Float],
                             qNorm: [Float], kNorm: [Float], oProjection: [Float],
                             configuration c: Configuration = .init()) -> [Float] {
        let h = c.heads, kh = c.kvHeads, d = c.headDim
        precondition(h > 0 && kh > 0 && h % kh == 0 && d > 0)
        precondition(c.epsilon.isFinite && c.epsilon > 0 && c.ropeTheta.isFinite && c.ropeTheta > 0)
        precondition(c.rotaryDim > 0 && c.rotaryDim.isMultiple(of: 2))
        precondition(qProjection.count == h * 2 * d)
        precondition(qNorm.count == d && kNorm.count == d)
        precondition(keys.count == values.count && keys.count % (kh * d) == 0)
        let tokens = keys.count / (kh * d), group = h / kh
        precondition(tokens > 0)
        let rotaryDim = min(c.rotaryDim, d)
        precondition(rotaryDim.isMultiple(of: 2))
        var mixed = Array(repeating: Float(0), count: h * d)
        for head in 0..<h {
            let base = head * 2 * d
            let q = norm(qProjection[base..<(base+d)], qNorm[0..<d], c.epsilon)
            let gate = qProjection[(base+d)..<(base+2*d)].map { 1 / (1 + exp(-$0)) }
            let qRot = RopeOracle.apply(input: q, position: c.position, headDim: d,
                                        rotaryDim: rotaryDim, ropeTheta: c.ropeTheta)
            var scores: [Float] = []
            var normKeys: [[Float]] = []
            for token in 0..<tokens {
                let kb = (token * kh + head / group) * d
                let k = norm(keys[kb..<(kb+d)], kNorm[0..<d], c.epsilon)
                let keyPosition = c.positionOrigin + token
                normKeys.append(RopeOracle.apply(input: k, position: keyPosition, headDim: d,
                                                  rotaryDim: rotaryDim, ropeTheta: c.ropeTheta))
            }
            for token in 0..<tokens {
                let keyPosition = c.positionOrigin + token
                scores.append(keyPosition <= c.position
                    ? dot(qRot, normKeys[token][...]) * (1 / sqrt(Float(d)))
                    : -.greatestFiniteMagnitude)
            }
            let p = softmax(scores)
            for j in 0..<d {
                // Gating is part of the attention value path and precedes o_proj.
                mixed[head*d+j] = zip(p, 0..<tokens).reduce(0) { acc, pair in
                    acc + pair.0 * values[(pair.1 * kh + head / group) * d + j]
                } * gate[j]
            }
        }
        guard oProjection.count == h*d*h*d else { return mixed }
        return (0..<(h*d)).map { row in
            (0..<(h*d)).reduce(0) { $0 + oProjection[row*h*d+$1] * mixed[$1] }
        }
    }

    public static func decodeDense(qProjection: [Float], keys: [Float], values: [Float], qNorm: [Float], kNorm: [Float], oProjection: [Float], position: Int, configuration: Configuration = .init()) -> [Float] {
        var c = configuration; c.position = position
        return layer(qProjection: qProjection, keys: keys, values: values, qNorm: qNorm, kNorm: kNorm, oProjection: oProjection, configuration: c)
    }

    public static func prefill(qProjections: [Float], keys: [Float], values: [Float], qNorm: [Float], kNorm: [Float], oProjection: [Float], startPosition: Int = 0, configuration: Configuration = .init()) -> [[Float]] {
        let width = configuration.heads * 2 * configuration.headDim
        precondition(qProjections.count % width == 0)
        let rows = qProjections.count / width
        return (0..<rows).map { row in
            let start = row * width
            var c = configuration; c.position = startPosition + row
            return layer(qProjection: Array(qProjections[start..<(start+width)]), keys: keys, values: values, qNorm: qNorm, kNorm: kNorm, oProjection: oProjection, configuration: c)
        }
    }
}
