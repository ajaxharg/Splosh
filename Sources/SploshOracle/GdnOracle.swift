// GdnOracle.swift — scalar gated-delta-rule reference (M2.5).
import Foundation

/// Independent CPU reference for Qwen3.5's gated delta network recurrence.
public enum GdnOracle {
    public static let epsilon: Float = 1e-6
    public static let defaultChunkSize = 64

    public struct State: Sendable, Equatable {
        public var values: [Float]
        public let heads: Int
        public let keyDim: Int
        public let valueDim: Int
        public init(values: [Float], heads: Int, keyDim: Int, valueDim: Int) {
            precondition(values.count == heads * keyDim * valueDim)
            self.values = values; self.heads = heads; self.keyDim = keyDim; self.valueDim = valueDim
        }
        public init(heads: Int, keyDim: Int, valueDim: Int) {
            self.init(values: Array(repeating: 0, count: heads * keyDim * valueDim), heads: heads, keyDim: keyDim, valueDim: valueDim)
        }
    }

    public struct RecurrenceResult: Sendable, Equatable {
        public let output: [Float]
        public let state: State
        public init(output: [Float], state: State) { self.output = output; self.state = state }
    }

    /// Prepare one token: causal depthwise convolution followed by SiLU.
    public static func prepare(input: [Float], kernel: [Float], history: [Float] = []) -> [Float] {
        precondition(kernel.count == 4)
        var window = Array(repeating: Float(0), count: 3)
        if history.count >= 3 { window = Array(history.suffix(3)) }
        return input.enumerated().map { i, x in
            let h0 = i >= 2 ? input[i - 2] : window[i + 1]
            let h1 = i >= 1 ? input[i - 1] : window[i + 2]
            let h2 = i == 0 ? window[2] : (i == 1 ? input[0] : input[i - 2])
            let y = kernel[0] * h0 + kernel[1] * h1 + kernel[2] * h2 + kernel[3] * x
            return y / (1 + exp(-y))
        }
    }

    /// One fp32 gated-delta update. State is indexed [head, key, value].
    public static func decode(query: [Float], key: [Float], value: [Float], beta: [Float], decay: [Float], state: inout State) -> [Float] {
        precondition(query.count == state.heads * state.keyDim && key.count == query.count)
        precondition(value.count == state.heads * state.valueDim && beta.count == state.heads && decay.count == state.heads)
        var output = Array(repeating: Float(0), count: value.count)
        for h in 0..<state.heads {
            let base = h * state.keyDim * state.valueDim
            let q = Array(query[(h * state.keyDim)..<((h + 1) * state.keyDim)])
            let k = Array(key[(h * state.keyDim)..<((h + 1) * state.keyDim)])
            let v = Array(value[(h * state.valueDim)..<((h + 1) * state.valueDim)])
            let d = exp(decay[h])
            for i in 0..<state.keyDim * state.valueDim { state.values[base + i] *= d }
            var predicted = Array(repeating: Float(0), count: state.valueDim)
            for i in 0..<state.keyDim { for j in 0..<state.valueDim { predicted[j] += state.values[base + i * state.valueDim + j] * k[i] } }
            for j in 0..<state.valueDim {
                let delta = beta[h] * (v[j] - predicted[j])
                for i in 0..<state.keyDim { state.values[base + i * state.valueDim + j] += k[i] * delta }
            }
            for j in 0..<state.valueDim { for i in 0..<state.keyDim { output[h * state.valueDim + j] += state.values[base + i * state.valueDim + j] * q[i] } }
        }
        return output
    }

    /// RMS-normalise each value head, then apply SiLU gate.
    public static func gate(_ values: [Float], z: [Float], weight: [Float] = [], epsilon: Float = Self.epsilon) -> [Float] {
        precondition(values.count == z.count)
        let dim = weight.isEmpty ? (values.count == 0 ? 1 : values.count) : weight.count
        precondition(weight.isEmpty || values.count % dim == 0)
        return values.enumerated().map { idx, x in
            let offset = (idx / dim) * dim
            let end = min(offset + dim, values.count)
            let slice = values[offset..<end]
            let inv = 1 / (slice.reduce(0) { $0 + $1 * $1 } / Float(slice.count) + epsilon).squareRoot()
            let scale = weight.isEmpty ? 1 : (1 + weight[idx % dim])
            let g = z[idx]
            return x * inv * scale * (g / (1 + exp(-g)))
        }
    }

    /// Explicit copy/swap commit; the returned state is an independent value.
    public static func commit(_ inactive: State) -> State { State(values: inactive.values, heads: inactive.heads, keyDim: inactive.keyDim, valueDim: inactive.valueDim) }

    /// Run the authoritative per-token recurrence. Inputs are token-major and flattened.
    public static func recurrence(query: [[Float]], key: [[Float]], value: [[Float]], beta: [[Float]], decay: [[Float]], initialState: State? = nil) -> RecurrenceResult {
        precondition(query.count == key.count && query.count == value.count && query.count == beta.count && query.count == decay.count)
        let heads = beta.first?.count ?? initialState?.heads ?? 0
        let kd = heads == 0 ? 0 : query.first!.count / heads
        let vd = heads == 0 ? 0 : value.first!.count / heads
        var state = initialState ?? State(heads: heads, keyDim: kd, valueDim: vd)
        var output: [Float] = []
        for t in query.indices { output += decode(query: query[t], key: key[t], value: value[t], beta: beta[t], decay: decay[t], state: &state) }
        return RecurrenceResult(output: output, state: state)
    }
}
