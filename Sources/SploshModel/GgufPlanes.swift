// GgufPlanes.swift — the layout the GGUF kernels read quantised weights in, the repack to it and
// the unpack from it.
//
// llama.cpp stores a row as native blocks, each with its scale fields at one end and the codes of
// its elements interleaved in whatever order suited its CPU decoder (GgufQuant.swift). The
// accelerator kernels (Sources/Shaders/engine_gguf.metal) dequantise a group of 32 inputs of
// several rows at once, so they want a row's group as one small record beside the same group of
// the neighbouring rows. A weight of `rows` outputs and `inner` inputs becomes three planes, in
// tiles of 128 rows like the engine's other weights:
//
//   plane0 [rows / 128][inner / 32][128][plane0Bytes]                  the codes of a row's group
//   plane1 [rows / 128][inner / 32][128][plane1Bytes]                  their high bits, where a code is split
//   meta   [rows / 128][inner / 32 / metaGroups][128][metaBytes]       a native block's scale fields
//
// The codes keep their native width; only their order changes. Inside a group the elements are in
// the chunk order Sources/Shaders/gguf_formats.h describes, and the fields are packed as it says.
// The encoding is Splash's (Apache-2.0: runtime/metal/abi/QuantFormat.h and
// kernels/shared/gguf_repack.metal), re-tiled from its 256 rows to this engine's 128; the native
// layouts are llama.cpp's (MIT).

import Foundation

public enum GgufPlanes {
    /// Rows in one tile of a plane.
    public static let tileRows = 128
    /// Elements in one group: the unit the kernels dequantise and multiply.
    public static let groupElements = 32

    /// The record sizes of one format.
    public struct Geometry: Sendable, Equatable {
        /// Bytes of plane 0, and of plane 1, for one row's group. A format with no plane 1 has 0.
        public let plane0Bytes: Int
        public let plane1Bytes: Int
        /// Bytes of one meta record: the scale fields of one native block of one row.
        public let metaBytes: Int
        /// Groups that one meta record covers: the native block, in groups.
        public let metaGroups: Int
        /// The format's token in kernel names, as in `sp_gguf_wide_q4k`.
        public let kernelSuffix: String
    }

    /// The byte counts of a weight's three planes.
    public struct Sizes: Sendable, Equatable {
        public let plane0: Int
        public let plane1: Int
        public let meta: Int
    }

    /// A weight's three planes.
    public struct Planes: Sendable, Equatable {
        public var plane0: [UInt8]
        public var plane1: [UInt8]
        public var meta: [UInt8]

        public init(plane0: [UInt8], plane1: [UInt8], meta: [UInt8]) {
            self.plane0 = plane0; self.plane1 = plane1; self.meta = meta
        }
    }

    /// The record sizes of a quantised type; nil for the float types and for Q4_0, which have no
    /// plane form (Q4_0 is read so that a file keeping its MTP block in it can be opened).
    public static func geometry(of type: GgufTensorType) -> Geometry? {
        switch type {
        case .q4K: return Geometry(plane0Bytes: 16, plane1Bytes: 0, metaBytes: 16, metaGroups: 8, kernelSuffix: "q4k")
        case .q5K: return Geometry(plane0Bytes: 16, plane1Bytes: 4, metaBytes: 16, metaGroups: 8, kernelSuffix: "q5k")
        case .q6K: return Geometry(plane0Bytes: 16, plane1Bytes: 8, metaBytes: 20, metaGroups: 8, kernelSuffix: "q6k")
        case .q3K: return Geometry(plane0Bytes: 8, plane1Bytes: 4, metaBytes: 16, metaGroups: 8, kernelSuffix: "q3k")
        case .q8_0: return Geometry(plane0Bytes: 32, plane1Bytes: 0, metaBytes: 2, metaGroups: 1, kernelSuffix: "q80")
        case .iq4NL: return Geometry(plane0Bytes: 16, plane1Bytes: 0, metaBytes: 2, metaGroups: 1, kernelSuffix: "iq4nl")
        case .iq4XS: return Geometry(plane0Bytes: 16, plane1Bytes: 0, metaBytes: 8, metaGroups: 8, kernelSuffix: "iq4xs")
        case .iq3S: return Geometry(plane0Bytes: 16, plane1Bytes: 0, metaBytes: 2, metaGroups: 8, kernelSuffix: "iq3s")
        case .f32, .f16, .bf16, .q4_0: return nil
        }
    }

    /// The plane sizes of a `rows` x `inner` weight; nil when the type has no plane form, the rows
    /// are not whole tiles or a row is not a whole number of native blocks.
    public static func sizes(of type: GgufTensorType, rows: Int, inner: Int) -> Sizes? {
        guard let geometry = geometry(of: type), rows > 0, rows % tileRows == 0,
              inner > 0, inner % type.blockElements == 0 else { return nil }
        let groups = inner / groupElements
        return Sizes(plane0: rows * groups * geometry.plane0Bytes, plane1: rows * groups * geometry.plane1Bytes,
                     meta: rows * (groups / geometry.metaGroups) * geometry.metaBytes)
    }

    /// The index of the record of (row, block) in a plane with `blocks` records a row: the groups
    /// of a code plane, or the native blocks of the meta plane.
    public static func record(row: Int, block: Int, blocks: Int) -> Int {
        ((row / tileRows) * blocks + block) * tileRows + row % tileRows
    }

    /// The slot of element `element` (0 to 31) of a group in chunk order.
    public static func slot(of element: Int) -> Int {
        8 * ((element >> 2) & 3) + 4 * (element >> 4) + (element & 3)
    }

    /// Repack native rows into the three planes.
    ///
    /// `native` holds `rows` rows of `inner` elements as whole native blocks, row after row, and
    /// each plane is exactly the size `sizes(of:rows:inner:)` gives; all of it is checked. The
    /// two bytes that pad a Q3_K or Q6_K meta record are written as zero. Nothing is allocated
    /// per row, group or element.
    public static func repack(_ type: GgufTensorType, native: UnsafeRawBufferPointer, rows: Int, inner: Int,
                              plane0: UnsafeMutableRawBufferPointer, plane1: UnsafeMutableRawBufferPointer,
                              meta: UnsafeMutableRawBufferPointer) {
        guard let geometry = geometry(of: type), let sizes = sizes(of: type, rows: rows, inner: inner) else {
            preconditionFailure("\(type) \(rows) x \(inner) has no plane form")
        }
        let blockBytes = type.blockBytes
        let rowBytes = inner / type.blockElements * blockBytes
        precondition(native.count == rows * rowBytes, "\(native.count) native bytes, \(rows * rowBytes) expected")
        precondition(plane0.count == sizes.plane0 && plane1.count == sizes.plane1 && meta.count == sizes.meta,
                     "planes of \(plane0.count), \(plane1.count) and \(meta.count) bytes, \(sizes) expected")
        let groups = inner / groupElements, blocks = groups / geometry.metaGroups
        // The low and the high field of each slot of the group in hand.
        withUnsafeTemporaryAllocation(of: UInt8.self, capacity: 2 * groupElements) { scratch in
            let low = UnsafeMutableBufferPointer(rebasing: scratch[..<groupElements])
            let high = UnsafeMutableBufferPointer(rebasing: scratch[groupElements...])
            for row in 0..<rows {
                for group in 0..<groups {
                    let block = group / geometry.metaGroups, j = group % geometry.metaGroups
                    let start = row * rowBytes + block * blockBytes
                    let b = UnsafeRawBufferPointer(rebasing: native[start ..< start + blockBytes])
                    let codes = record(row: row, block: group, blocks: groups)
                    let out0 = UnsafeMutableRawBufferPointer(
                        rebasing: plane0[codes * geometry.plane0Bytes ..< (codes + 1) * geometry.plane0Bytes])
                    let out1 = UnsafeMutableRawBufferPointer(
                        rebasing: plane1[codes * geometry.plane1Bytes ..< (codes + 1) * geometry.plane1Bytes])
                    let scales = record(row: row, block: block, blocks: blocks)
                    let header = UnsafeMutableRawBufferPointer(
                        rebasing: meta[scales * geometry.metaBytes ..< (scales + 1) * geometry.metaBytes])
                    pack(type, block: b, group: j, low: low, high: high, plane0: out0, plane1: out1, meta: header)
                }
            }
        }
    }

    /// Repack native rows into three new arrays.
    public static func repack(_ type: GgufTensorType, native: UnsafeRawBufferPointer, rows: Int, inner: Int) -> Planes {
        guard let sizes = sizes(of: type, rows: rows, inner: inner) else {
            preconditionFailure("\(type) \(rows) x \(inner) has no plane form")
        }
        var planes = Planes(plane0: [UInt8](repeating: 0, count: sizes.plane0), plane1: [UInt8](repeating: 0, count: sizes.plane1),
                            meta: [UInt8](repeating: 0, count: sizes.meta))
        planes.plane0.withUnsafeMutableBytes { plane0 in
            planes.plane1.withUnsafeMutableBytes { plane1 in
                planes.meta.withUnsafeMutableBytes { meta in
                    repack(type, native: native, rows: rows, inner: inner, plane0: plane0, plane1: plane1, meta: meta)
                }
            }
        }
        return planes
    }

    /// Unpack the three planes into native rows: the inverse of `repack`.
    ///
    /// Each plane is exactly the size `sizes(of:rows:inner:)` gives and `native` is `rows` rows of
    /// whole native blocks; all of it is checked. Every bit of a native block has a place in the
    /// planes, so a repacked tensor unpacks to the bytes it was made of, and a converted weight
    /// can be put through the reference decoder. Nothing is allocated per row, group or element.
    public static func unpack(_ type: GgufTensorType, plane0: UnsafeRawBufferPointer, plane1: UnsafeRawBufferPointer,
                              meta: UnsafeRawBufferPointer, rows: Int, inner: Int, native: UnsafeMutableRawBufferPointer) {
        guard let geometry = geometry(of: type), let sizes = sizes(of: type, rows: rows, inner: inner) else {
            preconditionFailure("\(type) \(rows) x \(inner) has no plane form")
        }
        let blockBytes = type.blockBytes
        let rowBytes = inner / type.blockElements * blockBytes
        precondition(native.count == rows * rowBytes, "\(native.count) native bytes, \(rows * rowBytes) expected")
        precondition(plane0.count == sizes.plane0 && plane1.count == sizes.plane1 && meta.count == sizes.meta,
                     "planes of \(plane0.count), \(plane1.count) and \(meta.count) bytes, \(sizes) expected")
        let groups = inner / groupElements, blocks = groups / geometry.metaGroups
        // A block's groups share its code bytes, so each group's bits are or-ed into them.
        native.initializeMemory(as: UInt8.self, repeating: 0)
        withUnsafeTemporaryAllocation(of: UInt8.self, capacity: 2 * groupElements) { scratch in
            let low = UnsafeMutableBufferPointer(rebasing: scratch[..<groupElements])
            let high = UnsafeMutableBufferPointer(rebasing: scratch[groupElements...])
            for row in 0..<rows {
                for group in 0..<groups {
                    let block = group / geometry.metaGroups, j = group % geometry.metaGroups
                    let start = row * rowBytes + block * blockBytes
                    let b = UnsafeMutableRawBufferPointer(rebasing: native[start ..< start + blockBytes])
                    let codes = record(row: row, block: group, blocks: groups)
                    let in0 = UnsafeRawBufferPointer(
                        rebasing: plane0[codes * geometry.plane0Bytes ..< (codes + 1) * geometry.plane0Bytes])
                    let in1 = UnsafeRawBufferPointer(
                        rebasing: plane1[codes * geometry.plane1Bytes ..< (codes + 1) * geometry.plane1Bytes])
                    let scales = record(row: row, block: block, blocks: blocks)
                    let header = UnsafeRawBufferPointer(
                        rebasing: meta[scales * geometry.metaBytes ..< (scales + 1) * geometry.metaBytes])
                    unpack(type, plane0: in0, plane1: in1, meta: header, group: j, low: low, high: high, block: b)
                }
            }
        }
    }

    /// Unpack the three planes into a new array of native rows.
    public static func unpack(_ type: GgufTensorType, planes: Planes, rows: Int, inner: Int) -> [UInt8] {
        var native = [UInt8](repeating: 0, count: rows * (inner / type.blockElements) * type.blockBytes)
        planes.plane0.withUnsafeBytes { plane0 in
            planes.plane1.withUnsafeBytes { plane1 in
                planes.meta.withUnsafeBytes { meta in
                    native.withUnsafeMutableBytes {
                        unpack(type, plane0: plane0, plane1: plane1, meta: meta, rows: rows, inner: inner, native: $0)
                    }
                }
            }
        }
        return native
    }

    /// The records of group `j` of native block `b`: its codes, and with the block's first group
    /// the block's meta record. The element order of each native layout is GgufQuant.swift's.
    private static func pack(_ type: GgufTensorType, block b: UnsafeRawBufferPointer, group j: Int,
                             low: UnsafeMutableBufferPointer<UInt8>, high: UnsafeMutableBufferPointer<UInt8>,
                             plane0: UnsafeMutableRawBufferPointer, plane1: UnsafeMutableRawBufferPointer,
                             meta: UnsafeMutableRawBufferPointer) {
        func copyMeta(_ range: Range<Int>, to offset: Int = 0) {
            for index in range { meta[offset + index - range.lowerBound] = b[index] }
        }
        switch type {
        case .q4K, .q5K:
            let fifthBit = type == .q5K, codes = fifthBit ? 48 : 16
            for e in 0..<32 {
                low[slot(of: e)] = (b[codes + 32 * (j / 2) + e] >> UInt8(4 * (j % 2))) & 15
                if fifthBit { high[slot(of: e)] = (b[16 + e] >> UInt8(j)) & 1 }
            }
            storePairs(low, to: plane0)
            if fifthBit { storeBits(high, width: 1, to: plane1) }
            if j == 0 { copyMeta(0..<16) }
        case .q6K:
            let half = j / 4, quarter = j % 4
            for e in 0..<32 {
                low[slot(of: e)] = (b[64 * half + 32 * (quarter & 1) + e] >> UInt8(4 * (quarter >> 1))) & 15
                high[slot(of: e)] = (b[128 + 32 * half + e] >> UInt8(2 * quarter)) & 3
            }
            storePairs(low, to: plane0)
            storeBits(high, width: 2, to: plane1)
            if j == 0 {
                copyMeta(192..<208)
                copyMeta(208..<210, to: 16)
                meta[18] = 0; meta[19] = 0
            }
        case .q3K:
            for e in 0..<32 {
                low[slot(of: e)] = (b[32 + 32 * (j / 4) + e] >> UInt8(2 * (j % 4))) & 3
                high[slot(of: e)] = (b[e] >> UInt8(j)) & 1
            }
            storeBits(low, width: 2, to: plane0)
            storeBits(high, width: 1, to: plane1)
            if j == 0 {
                copyMeta(108..<110)
                meta[2] = 0; meta[3] = 0
                copyMeta(96..<108, to: 4)
            }
        case .q8_0:
            for e in 0..<32 { low[slot(of: e)] = b[2 + e] }
            storeBits(low, width: 8, to: plane0)
            copyMeta(0..<2)
        case .iq4NL, .iq4XS:
            // Sixteen bytes of two indices: the low nibbles are elements 0 to 15, the high 16 to 31.
            let codes = type == .iq4NL ? 2 : 8 + 16 * j
            for e in 0..<16 {
                low[slot(of: e)] = b[codes + e] & 15
                low[slot(of: 16 + e)] = b[codes + e] >> 4
            }
            storeBits(low, width: 4, to: plane0)
            if j == 0 { copyMeta(0 ..< (type == .iq4NL ? 2 : 8)) }
        case .iq3S:
            // Grid entry t covers elements 4t to 4t + 3: its low index bits are byte t of the
            // group's eight, its ninth bit is bit t of the group's ninth-bit byte.
            for e in 0..<32 { high[slot(of: e)] = (b[74 + 4 * j + e / 8] >> UInt8(e % 8)) & 1 }
            let ninth = UInt32(b[66 + j]), scale = UInt32((b[106 + j / 2] >> UInt8(4 * (j % 2))) & 15)
            for c in 0..<4 {
                var signs: UInt32 = 0
                for i in 0..<8 { signs |= UInt32(high[8 * c + i]) << UInt32(i) }
                var word = UInt32(b[2 + 8 * j + c]) | UInt32(b[2 + 8 * j + 4 + c]) << 8 | signs << 16
                word |= ((ninth >> UInt32(c)) & 1) << 24 | ((ninth >> UInt32(4 + c)) & 1) << 25 | scale << 26
                plane0.storeBytes(of: word.littleEndian, toByteOffset: 4 * c, as: UInt32.self)
            }
            if j == 0 { copyMeta(0..<2) }
        case .f32, .f16, .bf16, .q4_0:
            preconditionFailure("\(type) has no plane form")
        }
    }

    /// 4-bit linear codes: word c holds slots 8c to 8c + 7, pair p's first element at bits 4p and
    /// its second at bits 16 + 4p.
    private static func storePairs(_ slots: UnsafeMutableBufferPointer<UInt8>, to plane: UnsafeMutableRawBufferPointer) {
        for c in 0..<4 {
            var word: UInt32 = 0
            for i in 0..<8 { word |= UInt32(slots[8 * c + i]) << UInt32((i & 1) * 16 + 4 * (i >> 1)) }
            plane.storeBytes(of: word.littleEndian, toByteOffset: 4 * c, as: UInt32.self)
        }
    }

    /// The 32 slots as a little-endian bit string of fields `width` bits wide (1, 2, 4 or 8).
    private static func storeBits(_ slots: UnsafeMutableBufferPointer<UInt8>, width: Int, to plane: UnsafeMutableRawBufferPointer) {
        let perWord = 32 / width
        for w in 0..<width {
            var word: UInt32 = 0
            for i in 0..<perWord { word |= UInt32(slots[w * perWord + i]) << UInt32(width * i) }
            plane.storeBytes(of: word.littleEndian, toByteOffset: 4 * w, as: UInt32.self)
        }
    }

    /// `pack` undone: the records of group `j` back into native block `b`, whose code bytes are
    /// zero or hold the block's earlier groups, and with the block's first group its scale fields.
    private static func unpack(_ type: GgufTensorType, plane0: UnsafeRawBufferPointer, plane1: UnsafeRawBufferPointer,
                               meta: UnsafeRawBufferPointer, group j: Int,
                               low: UnsafeMutableBufferPointer<UInt8>, high: UnsafeMutableBufferPointer<UInt8>,
                               block b: UnsafeMutableRawBufferPointer) {
        func copyMeta(_ range: Range<Int>, from offset: Int = 0) {
            for index in range { b[index] = meta[offset + index - range.lowerBound] }
        }
        switch type {
        case .q4K, .q5K:
            let fifthBit = type == .q5K, codes = fifthBit ? 48 : 16
            loadPairs(plane0, to: low)
            if fifthBit { loadBits(plane1, width: 1, to: high) }
            for e in 0..<32 {
                b[codes + 32 * (j / 2) + e] |= low[slot(of: e)] << UInt8(4 * (j % 2))
                if fifthBit { b[16 + e] |= high[slot(of: e)] << UInt8(j) }
            }
            if j == 0 { copyMeta(0..<16) }
        case .q6K:
            let half = j / 4, quarter = j % 4
            loadPairs(plane0, to: low)
            loadBits(plane1, width: 2, to: high)
            for e in 0..<32 {
                b[64 * half + 32 * (quarter & 1) + e] |= low[slot(of: e)] << UInt8(4 * (quarter >> 1))
                b[128 + 32 * half + e] |= high[slot(of: e)] << UInt8(2 * quarter)
            }
            if j == 0 {
                copyMeta(192..<208)
                copyMeta(208..<210, from: 16)
            }
        case .q3K:
            loadBits(plane0, width: 2, to: low)
            loadBits(plane1, width: 1, to: high)
            for e in 0..<32 {
                b[32 + 32 * (j / 4) + e] |= low[slot(of: e)] << UInt8(2 * (j % 4))
                b[e] |= high[slot(of: e)] << UInt8(j)
            }
            if j == 0 {
                copyMeta(108..<110)
                copyMeta(96..<108, from: 4)
            }
        case .q8_0:
            loadBits(plane0, width: 8, to: low)
            for e in 0..<32 { b[2 + e] = low[slot(of: e)] }
            copyMeta(0..<2)
        case .iq4NL, .iq4XS:
            let codes = type == .iq4NL ? 2 : 8 + 16 * j
            loadBits(plane0, width: 4, to: low)
            for e in 0..<16 { b[codes + e] = low[slot(of: e)] | low[slot(of: 16 + e)] << 4 }
            if j == 0 { copyMeta(0 ..< (type == .iq4NL ? 2 : 8)) }
        case .iq3S:
            var ninth: UInt32 = 0
            for c in 0..<4 {
                let word = UInt32(littleEndian: plane0.loadUnaligned(fromByteOffset: 4 * c, as: UInt32.self))
                b[2 + 8 * j + c] = UInt8(word & 0xFF)
                b[2 + 8 * j + 4 + c] = UInt8((word >> 8) & 0xFF)
                for i in 0..<8 { high[8 * c + i] = UInt8((word >> UInt32(16 + i)) & 1) }
                ninth |= ((word >> 24) & 1) << UInt32(c) | ((word >> 25) & 1) << UInt32(4 + c)
                // Every chunk carries the group's scale; the kernels read the first chunk's.
                if c == 0 { b[106 + j / 2] |= UInt8((word >> 26) & 15) << UInt8(4 * (j % 2)) }
            }
            for e in 0..<32 { b[74 + 4 * j + e / 8] |= high[slot(of: e)] << UInt8(e % 8) }
            b[66 + j] = UInt8(ninth)
            if j == 0 { copyMeta(0..<2) }
        case .f32, .f16, .bf16, .q4_0:
            preconditionFailure("\(type) has no plane form")
        }
    }

    /// `storePairs` undone.
    private static func loadPairs(_ plane: UnsafeRawBufferPointer, to slots: UnsafeMutableBufferPointer<UInt8>) {
        for c in 0..<4 {
            let word = UInt32(littleEndian: plane.loadUnaligned(fromByteOffset: 4 * c, as: UInt32.self))
            for i in 0..<8 { slots[8 * c + i] = UInt8((word >> UInt32((i & 1) * 16 + 4 * (i >> 1))) & 15) }
        }
    }

    /// `storeBits` undone.
    private static func loadBits(_ plane: UnsafeRawBufferPointer, width: Int, to slots: UnsafeMutableBufferPointer<UInt8>) {
        let perWord = 32 / width, mask = (UInt32(1) << UInt32(width)) - 1
        for w in 0..<width {
            let word = UInt32(littleEndian: plane.loadUnaligned(fromByteOffset: 4 * w, as: UInt32.self))
            for i in 0..<perWord { slots[w * perWord + i] = UInt8((word >> UInt32(width * i)) & mask) }
        }
    }
}
