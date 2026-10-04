// TensorSpec.swift — strict, self-describing safetensors tensor inventory.
import Foundation

public enum TensorDType: String, Codable, Sendable {
    case bf16 = "BF16"
    case f32 = "F32"
    case f16 = "F16"
    case u32 = "U32"
    case i32 = "I32"
    case i64 = "I64"
    case u8 = "U8"
    case unknown = "UNKNOWN"

    public init(safetensorsName: String) throws {
        switch safetensorsName {
        case "BF16": self = .bf16
        case "F32": self = .f32
        case "F16": self = .f16
        case "U32": self = .u32
        case "I32": self = .i32
        case "I64": self = .i64
        case "U8": self = .u8
        default: throw TensorInventoryError.unsupportedDType(safetensorsName)
        }
    }
    public var byteWidth: Int64? {
        switch self { case .bf16, .f16: return 2; case .f32, .u32, .i32: return 4; case .i64: return 8; case .u8: return 1; case .unknown: return nil }
    }
}

public struct TensorSpec: Codable, Sendable, Equatable {
    public let name: String
    public let dtype: TensorDType
    public let shape: [Int]
    public let byteOffset: Int64?
    public let byteLength: Int64?
    public let shard: String?
    public init(name: String, dtype: TensorDType, shape: [Int], byteOffset: Int64? = nil, byteLength: Int64? = nil, shard: String? = nil) {
        self.name = name; self.dtype = dtype; self.shape = shape; self.byteOffset = byteOffset; self.byteLength = byteLength; self.shard = shard
    }
    public func validate() throws {
        guard !name.isEmpty, !name.contains("\0") else { throw TensorInventoryError.invalidTensorName(name) }
        guard shape.allSatisfy({ $0 >= 0 }) else { throw TensorInventoryError.invalidShape(name, shape) }
        if let o = byteOffset, o < 0 { throw TensorInventoryError.invalidOffset(name, o) }
        if let l = byteLength, l < 0 { throw TensorInventoryError.invalidByteLength(name, l) }
        if let o = byteOffset, let l = byteLength, o > Int64.max - l { throw TensorInventoryError.offsetOverflow(name) }
    }
}

public enum TensorInventoryError: Error, Equatable, CustomStringConvertible {
    case missingIndex(URL)
    case malformedIndex(String)
    case missingTensor(String)
    case configHashMismatch(expected: String, observed: String)
    case sidecarDTypeMismatch(expected: TensorDType, observed: TensorDType)
    case unsupportedDType(String)
    case invalidTensorName(String)
    case invalidShape(String, [Int])
    case invalidOffset(String, Int64)
    case invalidByteLength(String, Int64)
    case offsetOverflow(String)
    case shardMissing(String)
    case shardTraversal(String)
    case headerMalformed(String)
    case boundsViolation(String)
    case dtypeMismatch(String, expected: TensorDType, observed: TensorDType)
    case shapeMismatch(String, expected: [Int], observed: [Int])
    case duplicateTensor(String)
    case indexCoverage(String)
    case mlxAffineInvalid(String)
    public var description: String {
        switch self {
        case .missingIndex(let u): return "tensor inventory missing index: \(u.path)"
        case .malformedIndex(let s): return "tensor inventory malformed index: \(s)"
        case .missingTensor(let s): return "required tensor missing: \(s)"
        case .configHashMismatch(let e, let o): return "config hash mismatch: expected \(e), observed \(o)"
        case .sidecarDTypeMismatch(let e, let o): return "sidecar dtype mismatch: expected \(e.rawValue), observed \(o.rawValue)"
        case .unsupportedDType(let s): return "unsupported safetensors dtype: \(s)"
        case .invalidTensorName(let s): return "invalid tensor name: \(s)"
        case .invalidShape(let s, let sh): return "invalid shape for \(s): \(sh)"
        case .invalidOffset(let s, let o): return "invalid offset for \(s): \(o)"
        case .invalidByteLength(let s, let l): return "invalid byte length for \(s): \(l)"
        case .offsetOverflow(let s): return "offset overflow for \(s)"
        case .shardMissing(let s): return "safetensors shard missing: \(s)"
        case .shardTraversal(let s): return "unsafe shard path: \(s)"
        case .headerMalformed(let s): return "malformed safetensors header: \(s)"
        case .boundsViolation(let s): return "safetensors bounds violation: \(s)"
        case .dtypeMismatch(let n, let e, let o): return "dtype mismatch for \(n): expected \(e.rawValue), observed \(o.rawValue)"
        case .shapeMismatch(let n, let e, let o): return "shape mismatch for \(n): expected \(e), observed \(o)"
        case .duplicateTensor(let n): return "duplicate tensor: \(n)"
        case .indexCoverage(let s): return "index coverage error: \(s)"
        case .mlxAffineInvalid(let s): return "MLX affine validation failed: \(s)"
        }
    }
}
