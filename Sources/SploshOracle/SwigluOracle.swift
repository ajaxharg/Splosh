// SwigluOracle.swift — SploshOracle.
//
// Scalar reference for the dense MLP SwiGLU activation.  Inputs are interpreted as
// bf16 values promoted to Float before evaluation; the result is rounded to bf16 to
// mirror the shader's output contract.

import Darwin

public enum SwigluOracle {
    /// Evaluate `silu(gate) * up` elementwise using Float32 arithmetic and bf16 I/O.
    ///
    /// The `UInt16` representation is used deliberately: Swift's Float16 is IEEE
    /// binary16 and cannot represent the model's bfloat16 payload.
    public static func gate(gate: [UInt16], up: [UInt16]) -> [UInt16] {
        precondition(gate.count == up.count, "gate and up must have equal lengths")
        return zip(gate, up).map { gateValue, upValue in
            let g = Float.fromBFloat16(gateValue)
            let u = Float.fromBFloat16(upValue)
            let silu = g / (1 + exp(-g))
            return Float(silu * u).bfloat16Bits
        }
    }

    /// Float32 oracle useful for independently checking the math before bf16 output
    /// rounding.  It intentionally has no dependency on Metal or Runtime.
    public static func gateFloat(gate: [Float], up: [Float]) -> [Float] {
        precondition(gate.count == up.count, "gate and up must have equal lengths")
        return zip(gate, up).map { g, u in g / (1 + exp(-g)) * u }
    }
}

private extension Float {
    static func fromBFloat16(_ bits: UInt16) -> Float {
        Float(bitPattern: UInt32(bits) << 16)
    }

    var bfloat16Bits: UInt16 {
        // Round-to-nearest-even when narrowing Float32 to bfloat16.
        let bits = bitPattern
        let upper = bits >> 16
        let lower = bits & 0xffff
        let rounded = upper + ((lower > 0x8000 || (lower == 0x8000 && (upper & 1) == 1)) ? 1 : 0)
        return UInt16(truncatingIfNeeded: rounded)
    }
}
