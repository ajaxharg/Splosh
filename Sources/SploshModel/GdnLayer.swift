import Foundation

/// Input-driven, fp32 gated-delta recurrent layer.
///
/// The host contract is deliberately independent of model weights: callers provide the
/// token-major projections and an explicit carried state.  The four stages are kept in a
/// single ordered execution path so a GPU backend can replace the scalar kernels without
/// changing validation or state ownership.
public struct GdnLayer: Sendable {
    public static let defaultChunkSize = 64
    public static let kernelOrder = ["gdn_prepare", "gdn_decode", "gdn_gate", "gdn_commit"]

    public struct State: Sendable, Equatable {
        public var values: [Float]
        public let heads: Int
        public let keyDim: Int
        public let valueDim: Int
        public init(values: [Float], heads: Int, keyDim: Int, valueDim: Int) throws {
            guard heads > 0, keyDim > 0, valueDim > 0,
                  values.count == heads * keyDim * valueDim else { throw Error.invalidState }
            self.values = values; self.heads = heads; self.keyDim = keyDim; self.valueDim = valueDim
        }
        public init(heads: Int, keyDim: Int, valueDim: Int) throws {
            try self.init(values: Array(repeating: 0, count: heads * keyDim * valueDim), heads: heads, keyDim: keyDim, valueDim: valueDim)
        }
    }

    public enum Error: Swift.Error, Equatable, Sendable {
        case emptyInput
        case invalidState
        case shapeMismatch(String)
        case invalidChunkSize(Int)
        case incompleteChunk(Int)
        case incompleteFinalState
        case nonFiniteInput(String)
    }

    public private(set) var dispatchedKernels: [String] = []
    public private(set) var chunkBoundaryCount: Int = 0
    public init() {}

    public mutating func runToken(query: [Float], key: [Float], value: [Float], beta: [Float], decay: [Float], state: inout State) throws -> [Float] {
        try validate(query: query, key: key, value: value, beta: beta, decay: decay, state: state)
        dispatchedKernels += Self.kernelOrder
        var output = Array(repeating: Float(0), count: value.count)
        for h in 0..<state.heads {
            let base = h * state.keyDim * state.valueDim
            let qo = h * state.keyDim, vo = h * state.valueDim
            let d = exp(decay[h])
            for i in 0..<(state.keyDim * state.valueDim) { state.values[base + i] *= d }
            for j in 0..<state.valueDim {
                var predicted: Float = 0
                for i in 0..<state.keyDim { predicted += state.values[base + i * state.valueDim + j] * key[qo + i] }
                let delta = beta[h] * (value[vo + j] - predicted)
                for i in 0..<state.keyDim { state.values[base + i * state.valueDim + j] += key[qo + i] * delta }
            }
            for j in 0..<state.valueDim {
                for i in 0..<state.keyDim { output[vo + j] += state.values[base + i * state.valueDim + j] * query[qo + i] }
            }
        }
        return output
    }

    public mutating func runChunk(query: [[Float]], key: [[Float]], value: [[Float]], beta: [[Float]], decay: [[Float]], state: inout State) throws -> [[Float]] {
        guard !query.isEmpty else { throw Error.emptyInput }
        guard query.count == key.count, query.count == value.count, query.count == beta.count, query.count == decay.count else { throw Error.shapeMismatch("token counts") }
        for t in query.indices { try validate(query: query[t], key: key[t], value: value[t], beta: beta[t], decay: decay[t], state: state) }
        var result: [[Float]] = []
        for t in query.indices { result.append(try runToken(query: query[t], key: key[t], value: value[t], beta: beta[t], decay: decay[t], state: &state)) }
        return result
    }

    public mutating func runChunked(query: [[Float]], key: [[Float]], value: [[Float]], beta: [[Float]], decay: [[Float]], state: inout State, chunkSize: Int = GdnLayer.defaultChunkSize) throws -> [[Float]] {
        guard !query.isEmpty else { throw Error.emptyInput }
        guard chunkSize == Self.defaultChunkSize else { throw Error.invalidChunkSize(chunkSize) }
        guard query.count == key.count, query.count == value.count, query.count == beta.count, query.count == decay.count else { throw Error.shapeMismatch("token counts") }
        guard query.count % chunkSize == 0 else { throw Error.incompleteChunk(query.count % chunkSize) }
        // Validate every token before touching state or instrumentation counters.
        for t in query.indices { try validate(query: query[t], key: key[t], value: value[t], beta: beta[t], decay: decay[t], state: state) }
        var all: [[Float]] = []
        for start in stride(from: 0, to: query.count, by: chunkSize) {
            chunkBoundaryCount += 1
            let end = start + chunkSize
            all += try runChunk(query: Array(query[start..<end]), key: Array(key[start..<end]), value: Array(value[start..<end]), beta: Array(beta[start..<end]), decay: Array(decay[start..<end]), state: &state)
        }
        guard state.values.count == state.heads * state.keyDim * state.valueDim else { throw Error.incompleteFinalState }
        return all
    }

    private func validate(query: [Float], key: [Float], value: [Float], beta: [Float], decay: [Float], state: State) throws {
        guard !query.isEmpty, !key.isEmpty, !value.isEmpty, !beta.isEmpty, !decay.isEmpty else { throw Error.emptyInput }
        guard state.values.count == state.heads * state.keyDim * state.valueDim else { throw Error.invalidState }
        guard query.count == state.heads * state.keyDim, key.count == query.count,
              value.count == state.heads * state.valueDim,
              beta.count == state.heads, decay.count == state.heads else { throw Error.shapeMismatch("query/key/value/beta/decay") }
        let inputs: [(String, [Float])] = [("query", query), ("key", key), ("value", value), ("beta", beta), ("decay", decay), ("state", state.values)]
        for (name, values) in inputs where values.contains(where: { !$0.isFinite }) {
            throw Error.nonFiniteInput(name)
        }
    }
}
