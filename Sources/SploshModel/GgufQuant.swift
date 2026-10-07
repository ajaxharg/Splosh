// GgufQuant.swift — the tensor types a GGUF file stores, and a reference decoder for their blocks.
//
// A quantised type packs a fixed number of consecutive elements into one block of fixed size, and a
// row of a tensor is a whole number of blocks. The layouts follow llama.cpp's ggml-quants.c. The
// decoder is the plain reading of those layouts, one element at a time, with the arithmetic in the
// order of the numpy decoder it was checked against (tools/gguf_check.py), which was itself checked
// against a second copy of the same model. It is a reference for tests and conversion checks, not a
// fast path.

import Foundation

/// A ggml tensor type that this reader knows the block geometry of. The raw value is ggml's type id.
public enum GgufTensorType: UInt32, Sendable, CaseIterable {
    case f32 = 0
    case f16 = 1
    case q4_0 = 2
    case q8_0 = 8
    case q3K = 11
    case q4K = 12
    case q5K = 13
    case q6K = 14
    case iq4NL = 20
    case iq3S = 21
    case iq4XS = 23
    case bf16 = 30

    /// Elements in one block: 1 for the float types, 32 for Q4_0, Q8_0 and IQ4_NL, 256 for the rest.
    public var blockElements: Int {
        switch self {
        case .f32, .f16, .bf16: return 1
        case .q4_0, .q8_0, .iq4NL: return 32
        case .q3K, .q4K, .q5K, .q6K, .iq4XS, .iq3S: return 256
        }
    }

    /// Bytes in one block.
    public var blockBytes: Int {
        switch self {
        case .f32: return 4
        case .f16, .bf16: return 2
        case .q8_0: return 34
        case .q4_0, .iq4NL: return 18
        case .q3K, .iq3S: return 110
        case .q4K: return 144
        case .q5K: return 176
        case .q6K: return 210
        case .iq4XS: return 136
        }
    }
}

public extension GgufTensorType {
    /// The sixteen values that the 4-bit codes of IQ4_NL and IQ4_XS index.
    ///
    /// This table and `iq3sGrid` are llama.cpp's `kvalues_iq4nl` and `iq3s_grid` (ggml-common.h),
    /// copied under llama.cpp's MIT licence, Copyright (c) 2023-2026 The ggml authors.
    static let iq4nlValues: [Int8] = [-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113]

    /// The IQ3_S grid, indexed by a 9-bit code. An entry holds the magnitudes of four consecutive
    /// elements in its bytes 0 to 3, byte 0 first. Magnitudes are the odd numbers 1 to 15.
    static let iq3sGrid: [UInt32] = [
        0x01010101, 0x01010103, 0x01010105, 0x0101010b, 0x0101010f, 0x01010301, 0x01010303, 0x01010305,
        0x01010309, 0x0101030d, 0x01010501, 0x01010503, 0x0101050b, 0x01010707, 0x01010901, 0x01010905,
        0x0101090b, 0x0101090f, 0x01010b03, 0x01010b07, 0x01010d01, 0x01010d05, 0x01010f03, 0x01010f09,
        0x01010f0f, 0x01030101, 0x01030103, 0x01030105, 0x01030109, 0x01030301, 0x01030303, 0x0103030b,
        0x01030501, 0x01030507, 0x0103050f, 0x01030703, 0x0103070b, 0x01030909, 0x01030d03, 0x01030d0b,
        0x01030f05, 0x01050101, 0x01050103, 0x0105010b, 0x0105010f, 0x01050301, 0x01050307, 0x0105030d,
        0x01050503, 0x0105050b, 0x01050701, 0x01050709, 0x01050905, 0x0105090b, 0x0105090f, 0x01050b03,
        0x01050b07, 0x01050f01, 0x01050f07, 0x01070107, 0x01070303, 0x0107030b, 0x01070501, 0x01070505,
        0x01070703, 0x01070707, 0x0107070d, 0x01070909, 0x01070b01, 0x01070b05, 0x01070d0f, 0x01070f03,
        0x01070f0b, 0x01090101, 0x01090307, 0x0109030f, 0x01090503, 0x01090509, 0x01090705, 0x01090901,
        0x01090907, 0x01090b03, 0x01090f01, 0x010b0105, 0x010b0109, 0x010b0501, 0x010b0505, 0x010b050d,
        0x010b0707, 0x010b0903, 0x010b090b, 0x010b090f, 0x010b0d0d, 0x010b0f07, 0x010d010d, 0x010d0303,
        0x010d0307, 0x010d0703, 0x010d0b05, 0x010d0f03, 0x010f0101, 0x010f0105, 0x010f0109, 0x010f0501,
        0x010f0505, 0x010f050d, 0x010f0707, 0x010f0b01, 0x010f0b09, 0x03010101, 0x03010103, 0x03010105,
        0x03010109, 0x03010301, 0x03010303, 0x03010307, 0x0301030b, 0x0301030f, 0x03010501, 0x03010505,
        0x03010703, 0x03010709, 0x0301070d, 0x03010b09, 0x03010b0d, 0x03010d03, 0x03010f05, 0x03030101,
        0x03030103, 0x03030107, 0x0303010d, 0x03030301, 0x03030309, 0x03030503, 0x03030701, 0x03030707,
        0x03030903, 0x03030b01, 0x03030b05, 0x03030f01, 0x03030f0d, 0x03050101, 0x03050305, 0x0305030b,
        0x0305030f, 0x03050501, 0x03050509, 0x03050705, 0x03050901, 0x03050907, 0x03050b0b, 0x03050d01,
        0x03050f05, 0x03070103, 0x03070109, 0x0307010f, 0x03070301, 0x03070307, 0x03070503, 0x0307050f,
        0x03070701, 0x03070709, 0x03070903, 0x03070d05, 0x03070f01, 0x03090107, 0x0309010b, 0x03090305,
        0x03090309, 0x03090703, 0x03090707, 0x03090905, 0x0309090d, 0x03090b01, 0x03090b09, 0x030b0103,
        0x030b0301, 0x030b0307, 0x030b0503, 0x030b0701, 0x030b0705, 0x030b0b03, 0x030d0501, 0x030d0509,
        0x030d050f, 0x030d0909, 0x030d090d, 0x030f0103, 0x030f0107, 0x030f0301, 0x030f0305, 0x030f0503,
        0x030f070b, 0x030f0903, 0x030f0d05, 0x030f0f01, 0x05010101, 0x05010103, 0x05010107, 0x0501010b,
        0x0501010f, 0x05010301, 0x05010305, 0x05010309, 0x0501030d, 0x05010503, 0x05010507, 0x0501050f,
        0x05010701, 0x05010705, 0x05010903, 0x05010907, 0x0501090b, 0x05010b01, 0x05010b05, 0x05010d0f,
        0x05010f01, 0x05010f07, 0x05010f0b, 0x05030101, 0x05030105, 0x05030301, 0x05030307, 0x0503030f,
        0x05030505, 0x0503050b, 0x05030703, 0x05030709, 0x05030905, 0x05030b03, 0x05050103, 0x05050109,
        0x0505010f, 0x05050503, 0x05050507, 0x05050701, 0x0505070f, 0x05050903, 0x05050b07, 0x05050b0f,
        0x05050f03, 0x05050f09, 0x05070101, 0x05070105, 0x0507010b, 0x05070303, 0x05070505, 0x05070509,
        0x05070703, 0x05070707, 0x05070905, 0x05070b01, 0x05070d0d, 0x05090103, 0x0509010f, 0x05090501,
        0x05090507, 0x05090705, 0x0509070b, 0x05090903, 0x05090f05, 0x05090f0b, 0x050b0109, 0x050b0303,
        0x050b0505, 0x050b070f, 0x050b0901, 0x050b0b07, 0x050b0f01, 0x050d0101, 0x050d0105, 0x050d010f,
        0x050d0503, 0x050d0b0b, 0x050d0d03, 0x050f010b, 0x050f0303, 0x050f050d, 0x050f0701, 0x050f0907,
        0x050f0b01, 0x07010105, 0x07010303, 0x07010307, 0x0701030b, 0x0701030f, 0x07010505, 0x07010703,
        0x07010707, 0x0701070b, 0x07010905, 0x07010909, 0x0701090f, 0x07010b03, 0x07010d07, 0x07010f03,
        0x07030103, 0x07030107, 0x0703010b, 0x07030309, 0x07030503, 0x07030507, 0x07030901, 0x07030d01,
        0x07030f05, 0x07030f0d, 0x07050101, 0x07050305, 0x07050501, 0x07050705, 0x07050709, 0x07050b01,
        0x07070103, 0x07070301, 0x07070309, 0x07070503, 0x07070507, 0x0707050f, 0x07070701, 0x07070903,
        0x07070907, 0x0707090f, 0x07070b0b, 0x07070f07, 0x07090107, 0x07090303, 0x0709030d, 0x07090505,
        0x07090703, 0x07090b05, 0x07090d01, 0x07090d09, 0x070b0103, 0x070b0301, 0x070b0305, 0x070b050b,
        0x070b0705, 0x070b0909, 0x070b0b0d, 0x070b0f07, 0x070d030d, 0x070d0903, 0x070f0103, 0x070f0107,
        0x070f0501, 0x070f0505, 0x070f070b, 0x09010101, 0x09010109, 0x09010305, 0x09010501, 0x09010509,
        0x0901050f, 0x09010705, 0x09010903, 0x09010b01, 0x09010f01, 0x09030105, 0x0903010f, 0x09030303,
        0x09030307, 0x09030505, 0x09030701, 0x0903070b, 0x09030907, 0x09030b03, 0x09030b0b, 0x09050103,
        0x09050107, 0x09050301, 0x0905030b, 0x09050503, 0x09050707, 0x09050901, 0x09050b0f, 0x09050d05,
        0x09050f01, 0x09070109, 0x09070303, 0x09070307, 0x09070501, 0x09070505, 0x09070703, 0x0907070b,
        0x09090101, 0x09090105, 0x09090509, 0x0909070f, 0x09090901, 0x09090f03, 0x090b010b, 0x090b010f,
        0x090b0503, 0x090b0d05, 0x090d0307, 0x090d0709, 0x090d0d01, 0x090f0301, 0x090f030b, 0x090f0701,
        0x090f0907, 0x090f0b03, 0x0b010105, 0x0b010301, 0x0b010309, 0x0b010505, 0x0b010901, 0x0b010909,
        0x0b01090f, 0x0b010b05, 0x0b010d0d, 0x0b010f09, 0x0b030103, 0x0b030107, 0x0b03010b, 0x0b030305,
        0x0b030503, 0x0b030705, 0x0b030f05, 0x0b050101, 0x0b050303, 0x0b050507, 0x0b050701, 0x0b05070d,
        0x0b050b07, 0x0b070105, 0x0b07010f, 0x0b070301, 0x0b07050f, 0x0b070909, 0x0b070b03, 0x0b070d0b,
        0x0b070f07, 0x0b090103, 0x0b090109, 0x0b090501, 0x0b090705, 0x0b09090d, 0x0b0b0305, 0x0b0b050d,
        0x0b0b0b03, 0x0b0b0b07, 0x0b0d0905, 0x0b0f0105, 0x0b0f0109, 0x0b0f0505, 0x0d010303, 0x0d010307,
        0x0d01030b, 0x0d010703, 0x0d010707, 0x0d010d01, 0x0d030101, 0x0d030501, 0x0d03050f, 0x0d030d09,
        0x0d050305, 0x0d050709, 0x0d050905, 0x0d050b0b, 0x0d050d05, 0x0d050f01, 0x0d070101, 0x0d070309,
        0x0d070503, 0x0d070901, 0x0d09050b, 0x0d090907, 0x0d090d05, 0x0d0b0101, 0x0d0b0107, 0x0d0b0709,
        0x0d0b0d01, 0x0d0d010b, 0x0d0d0901, 0x0d0f0303, 0x0d0f0307, 0x0f010101, 0x0f010109, 0x0f01010f,
        0x0f010501, 0x0f010505, 0x0f01070d, 0x0f010901, 0x0f010b09, 0x0f010d05, 0x0f030105, 0x0f030303,
        0x0f030509, 0x0f030907, 0x0f03090b, 0x0f050103, 0x0f050109, 0x0f050301, 0x0f05030d, 0x0f050503,
        0x0f050701, 0x0f050b03, 0x0f070105, 0x0f070705, 0x0f07070b, 0x0f070b07, 0x0f090103, 0x0f09010b,
        0x0f090307, 0x0f090501, 0x0f090b01, 0x0f0b0505, 0x0f0b0905, 0x0f0d0105, 0x0f0d0703, 0x0f0f0101,
    ]
}

public extension GgufTensorType {
    /// Decode whole native blocks to Float.
    ///
    /// `blocks` holds a whole number of blocks of `type`, and `into` has room for
    /// `blockElements` values per block; both are checked. Scales are fp16 widened to Float, and
    /// every product and sum is made in Float in the order the numpy reference makes it, so a
    /// decoded value equals the reference's bit for bit. Nothing is allocated per block.
    static func decode(_ type: GgufTensorType, blocks: UnsafeRawBufferPointer, into output: UnsafeMutableBufferPointer<Float>) {
        let size = type.blockBytes, elements = type.blockElements
        precondition(blocks.count % size == 0, "\(blocks.count) bytes are not a whole number of \(type) blocks")
        let count = blocks.count / size
        precondition(output.count == count * elements, "output holds \(output.count) values, \(count * elements) expected")
        for index in 0..<count {
            let block = UnsafeRawBufferPointer(rebasing: blocks[index * size ..< (index + 1) * size])
            let values = UnsafeMutableBufferPointer(rebasing: output[index * elements ..< (index + 1) * elements])
            switch type {
            case .f32: values[0] = Float(bitPattern: UInt32(littleEndian: block.loadUnaligned(as: UInt32.self)))
            case .f16: values[0] = block.half(at: 0)
            case .bf16: values[0] = Float(bitPattern: UInt32(UInt16(littleEndian: block.loadUnaligned(as: UInt16.self))) << 16)
            case .q4_0: decodeQ4_0(block, values)
            case .q8_0: decodeQ8_0(block, values)
            case .iq4NL: decodeIQ4NL(block, values)
            case .q4K: decodeQ45K(block, values, fifthBit: false)
            case .q5K: decodeQ45K(block, values, fifthBit: true)
            case .q6K: decodeQ6K(block, values)
            case .q3K: decodeQ3K(block, values)
            case .iq4XS: decodeIQ4XS(block, values)
            case .iq3S: decodeIQ3S(block, values)
            }
        }
    }

    /// Q4_0: fp16 scale, then 16 bytes of two codes; the low nibbles are elements 0 to 15 and the
    /// high nibbles 16 to 31. A code is a value from -8 to 7, stored with 8 added.
    private static func decodeQ4_0(_ b: UnsafeRawBufferPointer, _ out: UnsafeMutableBufferPointer<Float>) {
        let d = b.half(at: 0)
        for e in 0..<16 {
            out[e] = d * Float(Int(b[2 + e] & 15) - 8)
            out[16 + e] = d * Float(Int(b[2 + e] >> 4) - 8)
        }
    }

    /// Q8_0: fp16 scale, then 32 signed bytes.
    private static func decodeQ8_0(_ b: UnsafeRawBufferPointer, _ out: UnsafeMutableBufferPointer<Float>) {
        let d = b.half(at: 0)
        for e in 0..<32 { out[e] = d * Float(Int8(bitPattern: b[2 + e])) }
    }

    /// IQ4_NL: fp16 scale, then 16 bytes of two codes; the low nibbles are elements 0 to 15 and the
    /// high nibbles 16 to 31. A code indexes `iq4nlValues`.
    private static func decodeIQ4NL(_ b: UnsafeRawBufferPointer, _ out: UnsafeMutableBufferPointer<Float>) {
        let d = b.half(at: 0), values = iq4nlValues
        for e in 0..<16 {
            out[e] = d * Float(values[Int(b[2 + e] & 15)])
            out[16 + e] = d * Float(values[Int(b[2 + e] >> 4)])
        }
    }

    /// Q4_K and Q5_K: fp16 scale and minimum, twelve bytes of eight 6-bit scales and eight 6-bit
    /// minimums, for Q5_K 32 bytes of fifth bits, then the 4-bit codes. The 256 elements are eight
    /// groups of 32. Codes are in four runs of 32 bytes: group 2h takes the low nibbles of run h and
    /// group 2h + 1 the high nibbles. Bit j of fifth-bit byte e is the top bit of element e of group j.
    private static func decodeQ45K(_ b: UnsafeRawBufferPointer, _ out: UnsafeMutableBufferPointer<Float>, fifthBit: Bool) {
        let d = b.half(at: 0), dmin = b.half(at: 2)
        let codes = fifthBit ? 48 : 16
        func packed(_ k: Int) -> UInt8 { b[4 + k] }
        for j in 0..<8 {
            let scale: UInt8, minimum: UInt8
            if j < 4 {
                scale = packed(j) & 63
                minimum = packed(j + 4) & 63
            } else {
                scale = (packed(j + 4) & 0xF) | ((packed(j - 4) >> 6) << 4)
                minimum = (packed(j + 4) >> 4) | ((packed(j) >> 6) << 4)
            }
            let scaled = d * Float(scale), offset = dmin * Float(minimum)
            for e in 0..<32 {
                let byte = b[codes + 32 * (j / 2) + e]
                var code = Float(j % 2 == 0 ? byte & 15 : byte >> 4)
                if fifthBit { code += 16 * Float((b[16 + e] >> UInt8(j)) & 1) }
                out[32 * j + e] = code * scaled - offset
            }
        }
    }

    /// Q6_K: 128 bytes of low nibbles, 64 bytes of top bit pairs, sixteen signed 8-bit scales, fp16
    /// scale. In each half of 128 elements, quarter q takes its low nibble from the first or second
    /// 32 bytes of the half (q & 1), high nibble when q is 2 or 3, and its top two bits from the
    /// pair at bit 2q of the half's 32 top-bit bytes. The code is 6 bits, less 32.
    private static func decodeQ6K(_ b: UnsafeRawBufferPointer, _ out: UnsafeMutableBufferPointer<Float>) {
        let d = b.half(at: 208)
        for j in 0..<8 {
            let half = j / 4, quarter = j % 4
            for e in 0..<32 {
                let low = (b[64 * half + 32 * (quarter & 1) + e] >> (4 * (quarter >> 1))) & 15
                let high = (b[128 + 32 * half + e] >> (2 * quarter)) & 3
                let code = Float(low | (high << 4)) - 32
                let scale = Float(Int8(bitPattern: b[192 + 2 * j + e / 16]))
                out[32 * j + e] = d * scale * code
            }
        }
    }

    /// Q3_K: 32 bytes of high bits, 64 bytes of 2-bit codes, twelve bytes holding sixteen 6-bit
    /// scales, fp16 scale. Group j takes its low bits from the 32 bytes of half j / 4 at bit
    /// 2 (j % 4), and its high bit from bit j of the high-bit bytes; the code is 3 bits, less 4.
    /// The twelve scale bytes are three little-endian words. Scale k, with w = k / 4, is made of the
    /// low nibble (w < 2) or high nibble (w >= 2) of byte k % 4 of word w % 2, and, as its top two
    /// bits, bits 2w of byte k % 4 of the third word; it is applied less 32, to 16 elements.
    private static func decodeQ3K(_ b: UnsafeRawBufferPointer, _ out: UnsafeMutableBufferPointer<Float>) {
        let d = b.half(at: 108)
        func scale(_ k: Int) -> Float {
            let word = k / 4, byte = k % 4
            let low = (b[96 + 4 * (word % 2) + byte] >> (4 * (word / 2))) & 0xF
            let high = (b[104 + byte] >> (2 * word)) & 3
            return Float(low | (high << 4)) - 32
        }
        for j in 0..<8 {
            let half = j / 4, shift = 2 * (j % 4)
            for e in 0..<32 {
                let low = (b[32 + 32 * half + e] >> shift) & 3
                let high = (b[e] >> UInt8(j)) & 1
                let code = Float(low | (high << 2)) - 4
                out[32 * j + e] = d * scale(2 * j + e / 16) * code
            }
        }
    }

    /// IQ4_XS: fp16 scale, 16 bits of high scale bits, four bytes of low scale nibbles, then eight
    /// runs of 16 bytes of two codes. Group j has a 6-bit scale, less 32: its low four bits are the
    /// nibble j % 2 of byte j / 2 of the low scale bytes, its top two bits are bits 2j of the high
    /// scale bits. Within a run, low nibbles are elements 0 to 15 of the group, high nibbles 16 to 31.
    private static func decodeIQ4XS(_ b: UnsafeRawBufferPointer, _ out: UnsafeMutableBufferPointer<Float>) {
        let d = b.half(at: 0), values = iq4nlValues
        let high = UInt32(UInt16(littleEndian: b.loadUnaligned(fromByteOffset: 2, as: UInt16.self)))
        for j in 0..<8 {
            let low = (UInt32(b[4 + j / 2]) >> UInt32(4 * (j % 2))) & 0xF
            let scale = (low | (((high >> UInt32(2 * j)) & 3) << 4))
            let scaled = d * (Float(scale) - 32)
            for e in 0..<16 {
                let byte = b[8 + 16 * j + e]
                out[32 * j + e] = scaled * Float(values[Int(byte & 15)])
                out[32 * j + 16 + e] = scaled * Float(values[Int(byte >> 4)])
            }
        }
    }

    /// IQ3_S: fp16 scale, 64 bytes of low grid codes, eight bytes of ninth bits, 32 bytes of sign
    /// bits, four bytes of 4-bit scales. Group j of 32 elements is eight grid entries of four: entry
    /// k has the code byte 8j + k and ninth bit k of the group's ninth-bit byte. A scale s weighs the
    /// group by d (1 + 2s); sign bit e of the group's four sign bytes, counted from the low bit of
    /// the first, negates element e.
    private static func decodeIQ3S(_ b: UnsafeRawBufferPointer, _ out: UnsafeMutableBufferPointer<Float>) {
        let d = b.half(at: 0), grid = iq3sGrid
        for j in 0..<8 {
            let scale = (b[106 + j / 2] >> UInt8(4 * (j % 2))) & 15
            let weight = d * (1 + 2 * Float(scale))
            let ninth = b[66 + j]
            for k in 0..<8 {
                let entry = grid[Int(b[2 + 8 * j + k]) | (Int((ninth >> UInt8(k)) & 1) << 8)]
                for t in 0..<4 {
                    let e = 4 * k + t
                    let magnitude = Float((entry >> UInt32(8 * t)) & 0xFF)
                    let negative = (b[74 + 4 * j + e / 8] >> UInt8(e % 8)) & 1
                    out[32 * j + e] = weight * magnitude * (1 - 2 * Float(negative))
                }
            }
        }
    }
}

private extension UnsafeRawBufferPointer {
    /// The little-endian fp16 at a byte offset, widened to Float.
    func half(at offset: Int) -> Float {
        Float(Float16(bitPattern: UInt16(littleEndian: loadUnaligned(fromByteOffset: offset, as: UInt16.self))))
    }
}
