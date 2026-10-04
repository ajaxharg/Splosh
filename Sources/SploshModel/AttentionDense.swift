import Foundation

/// Geometry and host/shader ABI for dense (unpaged) full attention.
public struct AttentionDense: Sendable, Equatable {
    public enum ValidationError: Error, Equatable, CustomStringConvertible {
        case nonPositiveGeometry
        case queryHeadsNotDivisible
        case abiFieldOutOfRange(String)
        case invalidBuffer(String)
        case arithmeticOverflow(String)
        public var description: String {
            switch self {
            case .nonPositiveGeometry: return "attention geometry must be positive"
            case .queryHeadsNotDivisible: return "query heads must be divisible by KV heads"
            case .abiFieldOutOfRange(let field): return "attention ABI field out of range: \(field)"
            case .invalidBuffer(let field): return "invalid attention buffer: \(field)"
            case .arithmeticOverflow(let field): return "attention size overflow: \(field)"
            }
        }
    }

    /// Must remain byte-for-byte identical to `AttentionDenseParams` in Metal.
    /// The eight uint32 values precede the two float32 values; the struct is 40 bytes.
    public struct Parameters: Sendable, Equatable {
        public let headDim: UInt32
        public let heads: UInt32
        public let kvHeads: UInt32
        public let rowCount: UInt32
        public let tokenCount: UInt32
        public let queryPosition: UInt32
        public let startPosition: UInt32
        public let rotaryDim: UInt32
        public let epsilon: Float
        public let ropeTheta: Float

        public init(headDim: UInt32, heads: UInt32, kvHeads: UInt32, rowCount: UInt32,
                    tokenCount: UInt32, queryPosition: UInt32, startPosition: UInt32,
                    rotaryDim: UInt32, epsilon: Float, ropeTheta: Float) {
            self.headDim = headDim; self.heads = heads; self.kvHeads = kvHeads
            self.rowCount = rowCount; self.tokenCount = tokenCount
            self.queryPosition = queryPosition; self.startPosition = startPosition
            self.rotaryDim = rotaryDim; self.epsilon = epsilon; self.ropeTheta = ropeTheta
        }
    }

    public static let defaultRotaryDim = 64
    public static let defaultRopeTheta: Float = 10_000
    public static let abiSize = MemoryLayout<Parameters>.size
    public static let abiStride = MemoryLayout<Parameters>.stride

    public let heads: Int
    public let kvHeads: Int
    public let headDim: Int

    /// Invalid geometry returns nil rather than trapping.
    public init?(heads: Int = 24, kvHeads: Int = 4, headDim: Int = 256) {
        guard heads > 0, kvHeads > 0, headDim > 0, heads % kvHeads == 0 else { return nil }
        self.heads = heads; self.kvHeads = kvHeads; self.headDim = headDim
    }

    public init(validatingHeads heads: Int, kvHeads: Int, headDim: Int) throws {
        guard heads > 0, kvHeads > 0, headDim > 0 else { throw ValidationError.nonPositiveGeometry }
        guard heads % kvHeads == 0 else { throw ValidationError.queryHeadsNotDivisible }
        guard UInt32(exactly: heads) != nil, UInt32(exactly: kvHeads) != nil,
              UInt32(exactly: headDim) != nil else { throw ValidationError.abiFieldOutOfRange("geometry") }
        self.heads = heads; self.kvHeads = kvHeads; self.headDim = headDim
    }

    public var queryGroupSize: Int { heads / kvHeads }
    public var queryWidth: Int { heads * headDim }
    public var qProjectionWidth: Int { queryWidth * 2 }
    public var kvWidth: Int { kvHeads * headDim }
    public var bytesPerToken: Int { bytesPerLayerToken }
    public var bytesPerLayerToken: Int { kvWidth * 2 + kvHeads * 2 * MemoryLayout<Float>.stride }

    private func mul(_ a: Int, _ b: Int, _ name: String) throws -> Int {
        let result = a.multipliedReportingOverflow(by: b)
        guard !result.overflow else { throw ValidationError.arithmeticOverflow(name) }
        return result.partialValue
    }
    public func decodeDimensions(tokenCount: Int, queryPosition: Int) throws -> Parameters {
        try parameters(rows: 1, tokenCount: tokenCount, queryPosition: queryPosition, startPosition: queryPosition)
    }
    public func prefillDimensions(rowCount: Int, tokenCount: Int, startPosition: Int) throws -> Parameters {
        guard rowCount > 0 else { throw ValidationError.abiFieldOutOfRange("rowCount") }
        guard startPosition >= 0, rowCount <= tokenCount - startPosition else { throw ValidationError.abiFieldOutOfRange("startPosition/rowCount") }
        return try parameters(rows: rowCount, tokenCount: tokenCount, queryPosition: startPosition, startPosition: startPosition)
    }
    public func parameters(rows: Int, tokenCount: Int, queryPosition: Int, startPosition: Int = 0,
                           epsilon: Float = 1e-6, rotaryDim: Int = defaultRotaryDim,
                           ropeTheta: Float = defaultRopeTheta) throws -> Parameters {
        guard rows > 0, tokenCount > 0 else { throw ValidationError.abiFieldOutOfRange("rowCount/tokenCount") }
        guard queryPosition >= 0, queryPosition < tokenCount, startPosition >= 0 else { throw ValidationError.abiFieldOutOfRange("position") }
        guard rotaryDim > 0, rotaryDim <= headDim, rotaryDim % 2 == 0 else { throw ValidationError.abiFieldOutOfRange("rotaryDim") }
        guard epsilon.isFinite, epsilon > 0, ropeTheta.isFinite, ropeTheta > 0 else { throw ValidationError.abiFieldOutOfRange("epsilon/ropeTheta") }
        let values = [headDim, heads, kvHeads, rows, tokenCount, queryPosition, startPosition, rotaryDim]
        guard values.allSatisfy({ UInt32(exactly: $0) != nil }) else { throw ValidationError.abiFieldOutOfRange("uint32") }
        return Parameters(headDim: UInt32(headDim), heads: UInt32(heads), kvHeads: UInt32(kvHeads), rowCount: UInt32(rows), tokenCount: UInt32(tokenCount), queryPosition: UInt32(queryPosition), startPosition: UInt32(startPosition), rotaryDim: UInt32(rotaryDim), epsilon: epsilon, ropeTheta: ropeTheta)
    }

    public func validateBufferShapes(qProjection: Int, qNorm: Int, kNorm: Int, keys: Int, values: Int,
                                     kScales: Int, vScales: Int, output: Int, rows: Int = 1,
                                     denseOProjection: Int? = nil) throws {
        let expectedQ = try mul(qProjectionWidth, rows, "qProjection")
        let expectedOutput = try mul(queryWidth, rows, "output")
        guard qProjection == expectedQ, output == expectedOutput else { throw ValidationError.invalidBuffer("qProjection/output") }
        guard qNorm == headDim, kNorm == headDim else { throw ValidationError.invalidBuffer("norm weights") }
        let kvElements = try mul(kvWidth, max(1, rows), "KV")
        guard keys > 0, values == keys, keys % kvWidth == 0 else { throw ValidationError.invalidBuffer("token-major K/V") }
        let tokens = keys / kvWidth
        guard kScales == kvHeads, vScales == kvHeads, tokens > 0 else { throw ValidationError.invalidBuffer("K/V scales") }
        if let denseOProjection { guard denseOProjection == queryWidth * queryWidth else { throw ValidationError.invalidBuffer("o_proj") } }
        _ = kvElements
    }

    public func validateABI(tokenCount: Int, queryPosition: Int) throws { _ = try decodeDimensions(tokenCount: tokenCount, queryPosition: queryPosition) }
}
