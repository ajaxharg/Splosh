// Sampler.swift — token selection from a logits row.

import Accelerate
import Foundation

/// Largest-value selection over a logits row of ~250K entries, which sits on the per-token path.
public enum TopK {
    private static let chunk = 256

    /// Index of the largest of the first `limit` values.
    public static func argmax(_ logits: UnsafePointer<Float>, limit: Int) -> Int {
        var value: Float = 0
        var index: vDSP_Length = 0
        vDSP_maxvi(logits, 1, &value, &index, vDSP_Length(limit))
        return Int(index)
    }

    /// The `k` largest of the first `limit` values in descending order, with their indices;
    /// among equal values the lower index comes first.
    ///
    /// Chunk maxima are found with vector code; the k chunks with the largest maxima contain
    /// every one of the k largest values, so only those are scanned element by element.
    public static func select(_ logits: UnsafePointer<Float>, limit: Int, k: Int) -> (ids: [Int], values: [Float]) {
        var ids = [Int](repeating: 0, count: k)
        var values = [Float](repeating: -.infinity, count: k)
        func consider(_ range: Range<Int>) {
            for index in range {
                let value = logits[index]
                guard value > values[k - 1] else { continue }
                var slot = k - 1
                while slot > 0, values[slot - 1] < value { values[slot] = values[slot - 1]; ids[slot] = ids[slot - 1]; slot -= 1 }
                values[slot] = value; ids[slot] = index
            }
        }
        let chunks = limit / chunk
        guard chunks > k else { consider(0..<limit); return (ids, values) }
        var maxima = [Float](repeating: 0, count: chunks)
        for index in 0..<chunks { vDSP_maxv(logits + index * chunk, 1, &maxima[index], vDSP_Length(chunk)) }
        var largest = [Float](repeating: -.infinity, count: k)
        for value in maxima where value > largest[k - 1] {
            var slot = k - 1
            while slot > 0, largest[slot - 1] < value { largest[slot] = largest[slot - 1]; slot -= 1 }
            largest[slot] = value
        }
        let threshold = largest[k - 1]
        for index in 0..<chunks where maxima[index] >= threshold { consider(index * chunk..<(index + 1) * chunk) }
        consider(chunks * chunk..<limit)
        return (ids, values)
    }
}

public struct SamplingParameters: Sendable, Equatable {
    /// 0 selects the arg-max token.
    public var temperature: Float
    public var topP: Float
    /// Candidates considered before nucleus filtering. 0 falls back to a bounded default.
    public var topK: Int
    public var seed: UInt64?

    /// Defaults follow the checkpoint's generation_config (temperature 1.0, top_k 20, top_p 0.95).
    public init(temperature: Float = 1.0, topP: Float = 0.95, topK: Int = 20, seed: UInt64? = nil) {
        self.temperature = temperature; self.topP = topP; self.topK = topK; self.seed = seed
    }

    public static let greedy = SamplingParameters(temperature: 0)
}

public struct Sampler: Sendable {
    private var state: UInt64
    public let parameters: SamplingParameters
    /// Token ids at or above this are padding rows of the head and are never selected.
    public let vocabLimit: Int

    public init(parameters: SamplingParameters, vocabLimit: Int) {
        self.parameters = parameters
        self.vocabLimit = vocabLimit
        self.state = parameters.seed ?? UInt64.random(in: 1...UInt64.max)
    }

    private mutating func uniform() -> Float {
        // SplitMix64
        state &+= 0x9E37_79B9_7F4A_7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        z ^= z >> 31
        return Float(z >> 40) / Float(1 << 24)
    }

    /// Uniform draw in [0, 1).
    public mutating func uniform01() -> Float { uniform() }

    /// The arg-max token, ignoring padding rows of the head.
    public func argmax(_ logits: UnsafeBufferPointer<Float>) -> Int {
        TopK.argmax(logits.baseAddress!, limit: min(vocabLimit, logits.count))
    }

    /// The sampling distribution this sampler draws from: token ids and probabilities after
    /// temperature, top-k and nucleus filtering. Requires temperature > 0.
    public func distribution(_ logits: UnsafeBufferPointer<Float>) -> (ids: [Int], probabilities: [Float]) {
        let limit = min(vocabLimit, logits.count)
        let k = max(1, min(parameters.topK > 0 ? parameters.topK : 64, limit))
        var (ids, values) = TopK.select(logits.baseAddress!, limit: limit, k: k)
        let top = values[0]
        var probabilities = values.map { exp(($0 - top) / max(parameters.temperature, 1e-6)) }
        var total = probabilities.reduce(0, +)
        for index in probabilities.indices { probabilities[index] /= total }
        var kept = k
        if parameters.topP < 1 {
            var mass: Float = 0
            for index in 0..<k { mass += probabilities[index]; if mass >= parameters.topP { kept = index + 1; break } }
        }
        ids.removeLast(k - kept); probabilities.removeLast(k - kept)
        total = probabilities.reduce(0, +)
        for index in probabilities.indices { probabilities[index] /= total }
        return (ids, probabilities)
    }

    /// Draw an index from a normalised distribution.
    public mutating func draw(_ probabilities: [Float]) -> Int {
        var remaining = uniform()
        for (index, probability) in probabilities.enumerated() {
            remaining -= probability
            if remaining <= 0 { return index }
        }
        return probabilities.count - 1
    }

    public mutating func sample(_ logits: UnsafeBufferPointer<Float>) -> Int {
        let limit = min(vocabLimit, logits.count)
        if parameters.temperature <= 0 { return TopK.argmax(logits.baseAddress!, limit: limit) }
        // Keep the k largest logits in descending order.
        let k = max(1, min(parameters.topK > 0 ? parameters.topK : 64, limit))
        let (ids, values) = TopK.select(logits.baseAddress!, limit: limit, k: k)
        let top = values[0]
        var probabilities = values.map { exp(($0 - top) / parameters.temperature) }
        let total = probabilities.reduce(0, +)
        for index in probabilities.indices { probabilities[index] /= total }
        // Nucleus: the smallest prefix whose mass reaches topP.
        var kept = k
        if parameters.topP < 1 {
            var mass: Float = 0
            for index in 0..<k {
                mass += probabilities[index]
                if mass >= parameters.topP { kept = index + 1; break }
            }
        }
        let keptMass = probabilities[0..<kept].reduce(0, +)
        var draw = uniform() * keptMass
        for index in 0..<kept {
            draw -= probabilities[index]
            if draw <= 0 { return ids[index] }
        }
        return ids[kept - 1]
    }
}

/// Speculative rejection sampling (Leviathan et al. 2023, Chen et al. 2023): a token drawn from
/// the draft's distribution q is kept with probability min(1, p/q); otherwise the position is
/// redrawn from the normalised excess max(0, p - q). The token that results is distributed
/// exactly as p, whatever q is.
public enum SpeculativeSampling {
    /// Nil if `draft` is accepted, else the token that replaces it. `draft` must have been drawn
    /// from `draftProbabilities` over `candidates`; the target's distribution is given over
    /// `targetIDs`.
    public static func resolve(draft: Int, candidates: [Int], draftProbabilities: [Float], targetIDs: [Int],
                               targetProbabilities: [Float], sampler: inout Sampler) -> Int? {
        let q = candidates.firstIndex(of: draft).map { draftProbabilities[$0] } ?? 0
        let p = targetIDs.firstIndex(of: draft).map { targetProbabilities[$0] } ?? 0
        if sampler.uniform01() * q < p { return nil }
        var residual = targetProbabilities
        for (slot, id) in targetIDs.enumerated() {
            if let candidate = candidates.firstIndex(of: id) { residual[slot] = max(0, residual[slot] - draftProbabilities[candidate]) }
        }
        let total = residual.reduce(0, +)
        if total > 0 { for slot in residual.indices { residual[slot] /= total } } else { residual = targetProbabilities }
        return targetIDs[sampler.draw(residual)]
    }
}
