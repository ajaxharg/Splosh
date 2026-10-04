// GgufFile.swift — a read-only map of a GGUF v3 file: its metadata, its tensor directory and the
// bytes of its tensors.
//
// A GGUF file is a header, padded to `general.alignment`, then the tensor data. The header is the
// magic "GGUF", the version, the tensor and metadata counts, the metadata pairs, and one record per
// tensor: its name, its dimensions innermost first, its type, and an offset counted from the start
// of the data section. All integers are little-endian. The file is mapped and only the header is
// parsed (11 MB of a 16 GB file); tensor bytes are not read until a caller touches them.

import Foundation

public enum GgufError: Error, Equatable, CustomStringConvertible {
    case unreadable(path: String, reason: String)
    case badMagic
    case unsupportedVersion(UInt32)
    case truncatedHeader
    case malformedHeader(String)
    case unknownTensor(String)
    case unsupportedType(tensor: String, id: UInt32)
    case tensorPastEnd(tensor: String, end: UInt64, fileSize: UInt64)
    case rowsOutOfRange(tensor: String, rows: Range<Int>, count: Int)

    public var description: String {
        switch self {
        case .unreadable(let path, let reason): return "GGUF file unreadable: \(path): \(reason)"
        case .badMagic: return "not a GGUF file: bad magic"
        case .unsupportedVersion(let version): return "unsupported GGUF version \(version); only version 3 is read"
        case .truncatedHeader: return "GGUF header is truncated"
        case .malformedHeader(let reason): return "GGUF header is malformed: \(reason)"
        case .unknownTensor(let name): return "no tensor named \(name) in the GGUF file"
        case .unsupportedType(let tensor, let id): return "tensor \(tensor) has ggml type \(id), which is not supported"
        case .tensorPastEnd(let tensor, let end, let fileSize): return "tensor \(tensor) ends at byte \(end), past the end of the file (\(fileSize) bytes)"
        case .rowsOutOfRange(let tensor, let rows, let count): return "rows \(rows.lowerBound)..<\(rows.upperBound) of tensor \(tensor) are outside its \(count) rows"
        }
    }
}

/// A metadata value. Arrays of numbers are kept when they are short; a longer array, an array of
/// strings or of bools, is skipped and only its count is kept (a tokenizer's vocabulary is a
/// quarter of a million strings and is of no use to the engine).
public enum GgufValue: Sendable, Equatable {
    /// Any integer type up to 64 bits, signed or unsigned.
    case integer(Int64)
    /// A UINT64 above `Int64.max`, the only integer `integer` cannot hold.
    case unsigned(UInt64)
    /// FLOAT32 widened exactly, or FLOAT64.
    case float(Double)
    case bool(Bool)
    case string(String)
    case integers([Int64])
    case floats([Double])
    /// An array that was not kept.
    case skippedArray(count: Int)

    public var integerValue: Int? { if case .integer(let value) = self { return Int(exactly: value) } else { return nil } }
    public var floatValue: Double? { if case .float(let value) = self { return value } else { return nil } }
    public var stringValue: String? { if case .string(let value) = self { return value } else { return nil } }
}

public struct GgufFile: Sendable {
    /// One tensor of the directory.
    public struct Tensor: Sendable, Equatable {
        public let name: String
        public let type: GgufTensorType
        /// The extents as stored, innermost first: a matrix is `[inner, rows]`.
        public let dims: [Int]
        /// Where the bytes start, from the start of the file.
        public let offset: Int
        public let byteLength: Int

        /// Elements in a row: the innermost extent. A whole number of blocks of `type`.
        public var inner: Int { dims[0] }
        /// Rows: the product of the other extents, so 1 for a vector.
        public var rows: Int { dims.dropFirst().reduce(1, *) }
        public var elementCount: Int { dims.reduce(1, *) }
        /// Bytes in a row.
        public var rowBytes: Int { inner / type.blockElements * type.blockBytes }
    }

    /// Arrays of numbers longer than this are not kept in the metadata.
    public static let maxKeptArrayCount = 256

    public let url: URL
    public let fileSize: Int
    public let version: UInt32
    public let metadata: [String: GgufValue]
    /// `general.alignment`, 32 when the file does not say.
    public let alignment: Int
    /// Where the tensor data starts: the end of the header rounded up to `alignment`.
    public let dataStart: Int
    /// The tensors in the order of the file.
    public let tensors: [Tensor]

    private let indexByName: [String: Int]
    private let mapping: Mapping

    /// Map the file and parse its header. Every tensor is checked to lie inside the file, so a
    /// download that stopped early fails here, not on first touch.
    public init(url: URL) throws {
        let mapping = try Mapping(url: url)
        var reader = Reader(base: mapping.base, count: mapping.length)
        // "GGUF" read as a little-endian word.
        guard try reader.read(UInt32.self) == 0x4655_4747 else { throw GgufError.badMagic }
        let version = try reader.read(UInt32.self)
        guard version == 3 else { throw GgufError.unsupportedVersion(version) }
        let tensorCount = try reader.read(UInt64.self)
        let pairCount = try reader.read(UInt64.self)

        var metadata: [String: GgufValue] = [:]
        for _ in 0..<pairCount {
            let key = try reader.string()
            let value = try reader.value(ofType: try reader.read(UInt32.self))
            guard metadata.updateValue(value, forKey: key) == nil else { throw GgufError.malformedHeader("duplicate metadata key \(key)") }
        }
        var alignment = 32
        if let declared = metadata["general.alignment"] {
            guard let value = declared.integerValue, value > 0 else { throw GgufError.malformedHeader("general.alignment is not a positive integer") }
            alignment = value
        }

        struct Pending { let name: String; let type: GgufTensorType; let dims: [Int]; let relative: UInt64; let byteLength: Int }
        var pending: [Pending] = []
        for _ in 0..<tensorCount {
            let name = try reader.string()
            let rank = try reader.read(UInt32.self)
            // ggml has at most four dimensions.
            guard rank >= 1, rank <= 4 else { throw GgufError.malformedHeader("tensor \(name) has \(rank) dimensions") }
            var dims: [Int] = []
            for _ in 0..<rank {
                guard let extent = Int(exactly: try reader.read(UInt64.self)) else { throw GgufError.malformedHeader("tensor \(name) has an extent beyond Int") }
                dims.append(extent)
            }
            let id = try reader.read(UInt32.self)
            let relative = try reader.read(UInt64.self)
            guard let type = GgufTensorType(rawValue: id) else { throw GgufError.unsupportedType(tensor: name, id: id) }
            guard dims[0] % type.blockElements == 0 else {
                throw GgufError.malformedHeader("tensor \(name): a row of \(dims[0]) elements is not a whole number of \(type.blockElements)-element blocks")
            }
            var elements = 1
            for extent in dims {
                let (product, overflow) = elements.multipliedReportingOverflow(by: extent)
                guard !overflow else { throw GgufError.malformedHeader("tensor \(name) has more elements than Int holds") }
                elements = product
            }
            let (byteLength, overflow) = (elements / type.blockElements).multipliedReportingOverflow(by: type.blockBytes)
            guard !overflow else { throw GgufError.malformedHeader("tensor \(name) has more bytes than Int holds") }
            pending.append(Pending(name: name, type: type, dims: dims, relative: relative, byteLength: byteLength))
        }

        let remainder = reader.position % alignment
        let (dataStart, alignOverflow) = remainder == 0 ? (reader.position, false) : reader.position.addingReportingOverflow(alignment - remainder)
        guard !alignOverflow else { throw GgufError.malformedHeader("general.alignment is too large") }
        var tensors: [Tensor] = []
        var indexByName: [String: Int] = [:]
        for item in pending {
            let (start, startOverflow) = UInt64(dataStart).addingReportingOverflow(item.relative)
            let (end, endOverflow) = start.addingReportingOverflow(UInt64(item.byteLength))
            guard !startOverflow, !endOverflow, end <= UInt64(mapping.length) else {
                throw GgufError.tensorPastEnd(tensor: item.name, end: startOverflow || endOverflow ? UInt64.max : end, fileSize: UInt64(mapping.length))
            }
            guard indexByName.updateValue(tensors.count, forKey: item.name) == nil else { throw GgufError.malformedHeader("duplicate tensor \(item.name)") }
            tensors.append(Tensor(name: item.name, type: item.type, dims: item.dims, offset: Int(start), byteLength: item.byteLength))
        }

        self.url = url
        self.fileSize = mapping.length
        self.version = version
        self.metadata = metadata
        self.alignment = alignment
        self.dataStart = dataStart
        self.tensors = tensors
        self.indexByName = indexByName
        self.mapping = mapping
    }

    /// The tensor of this name; throws `unknownTensor` when the file has none.
    public func tensor(named name: String) throws -> Tensor {
        guard let index = indexByName[name] else { throw GgufError.unknownTensor(name) }
        return tensors[index]
    }

    /// The bytes of a tensor of this file.
    ///
    /// The pointer is into the mapping and is valid only while a copy of this `GgufFile` is alive;
    /// keep the file in scope, or wrap the use in `withExtendedLifetime`.
    public func bytes(of tensor: Tensor) -> UnsafeRawBufferPointer {
        precondition(tensor.offset >= 0 && tensor.byteLength <= fileSize - tensor.offset, "tensor \(tensor.name) is not a tensor of this file")
        return UnsafeRawBufferPointer(start: mapping.base + tensor.offset, count: tensor.byteLength)
    }

    /// The bytes of the rows `rows` of a tensor of this file, a whole number of blocks a row. The
    /// same lifetime rule as `bytes(of:)` applies.
    public func bytes(of tensor: Tensor, rows: Range<Int>) throws -> UnsafeRawBufferPointer {
        guard rows.lowerBound >= 0, rows.upperBound <= tensor.rows else {
            throw GgufError.rowsOutOfRange(tensor: tensor.name, rows: rows, count: tensor.rows)
        }
        let whole = bytes(of: tensor), stride = tensor.rowBytes
        return UnsafeRawBufferPointer(rebasing: whole[rows.lowerBound * stride ..< rows.upperBound * stride])
    }

    public func bytes(named name: String) throws -> UnsafeRawBufferPointer { bytes(of: try tensor(named: name)) }
}

/// The whole file, mapped read-only and unmapped when the last `GgufFile` that holds it goes.
private final class Mapping: @unchecked Sendable {
    let base: UnsafeRawPointer
    let length: Int

    init(url: URL) throws {
        let descriptor = open(url.path, O_RDONLY)
        guard descriptor >= 0 else { throw GgufError.unreadable(path: url.path, reason: String(cString: strerror(errno))) }
        defer { close(descriptor) }
        var info = stat()
        guard fstat(descriptor, &info) == 0 else { throw GgufError.unreadable(path: url.path, reason: String(cString: strerror(errno))) }
        // An empty file cannot be mapped, and has no header.
        guard info.st_size > 0 else { throw GgufError.truncatedHeader }
        guard let mapped = mmap(nil, Int(info.st_size), PROT_READ, MAP_PRIVATE, descriptor, 0),
              mapped != UnsafeMutableRawPointer(bitPattern: UInt(bitPattern: -1)) else {
            throw GgufError.unreadable(path: url.path, reason: String(cString: strerror(errno)))
        }
        base = UnsafeRawPointer(mapped)
        length = Int(info.st_size)
    }

    deinit { munmap(UnsafeMutableRawPointer(mutating: base), length) }
}

/// A cursor over the header that throws `truncatedHeader` instead of reading past the end.
private struct Reader {
    let base: UnsafeRawPointer
    let count: Int
    var position = 0

    init(base: UnsafeRawPointer, count: Int) { self.base = base; self.count = count }

    var remaining: Int { count - position }

    mutating func read<T: FixedWidthInteger>(_ type: T.Type) throws -> T {
        let size = MemoryLayout<T>.size
        guard remaining >= size else { throw GgufError.truncatedHeader }
        defer { position += size }
        return T(littleEndian: base.loadUnaligned(fromByteOffset: position, as: T.self))
    }

    /// A string: a 64-bit byte count, then that many bytes of UTF-8.
    mutating func string() throws -> String {
        let length = try read(UInt64.self)
        guard length <= UInt64(remaining) else { throw GgufError.truncatedHeader }
        defer { position += Int(length) }
        return String(decoding: UnsafeRawBufferPointer(start: base + position, count: Int(length)), as: UTF8.self)
    }

    mutating func skipString() throws {
        let length = try read(UInt64.self)
        guard length <= UInt64(remaining) else { throw GgufError.truncatedHeader }
        position += Int(length)
    }

    /// The byte width of a metadata value type that has a fixed one; nil for a string or an array.
    static func width(ofValueType type: UInt32) -> Int? {
        switch type {
        case 0, 1, 7: return 1
        case 2, 3: return 2
        case 4, 5, 6: return 4
        case 10, 11, 12: return 8
        default: return nil
        }
    }

    /// A metadata value of GGUF value type `type`: 0 to 5 are UINT8, INT8, UINT16, INT16, UINT32
    /// and INT32, 6 FLOAT32, 7 BOOL, 8 string, 9 array, 10 to 12 UINT64, INT64 and FLOAT64.
    mutating func value(ofType type: UInt32) throws -> GgufValue {
        switch type {
        case 0: return .integer(Int64(try read(UInt8.self)))
        case 1: return .integer(Int64(try read(Int8.self)))
        case 2: return .integer(Int64(try read(UInt16.self)))
        case 3: return .integer(Int64(try read(Int16.self)))
        case 4: return .integer(Int64(try read(UInt32.self)))
        case 5: return .integer(Int64(try read(Int32.self)))
        case 6: return .float(Double(Float(bitPattern: try read(UInt32.self))))
        case 7: return .bool(try read(UInt8.self) != 0)
        case 8: return .string(try string())
        case 9: return try array()
        case 10:
            let value = try read(UInt64.self)
            if let fits = Int64(exactly: value) { return .integer(fits) }
            return .unsigned(value)
        case 11: return .integer(try read(Int64.self))
        case 12: return .float(Double(bitPattern: try read(UInt64.self)))
        default: throw GgufError.malformedHeader("unknown metadata value type \(type)")
        }
    }

    /// An array: the element type, a 64-bit count, then the elements. Arrays of arrays do not occur
    /// in GGUF files and are refused.
    private mutating func array() throws -> GgufValue {
        let elementType = try read(UInt32.self)
        let count = try read(UInt64.self)
        if elementType == 8 {
            for _ in 0..<count { try skipString() }
            return .skippedArray(count: Int(count))
        }
        guard let width = Self.width(ofValueType: elementType) else {
            throw GgufError.malformedHeader("array of metadata value type \(elementType)")
        }
        let (bytes, overflow) = count.multipliedReportingOverflow(by: UInt64(width))
        guard !overflow, bytes <= UInt64(remaining) else { throw GgufError.truncatedHeader }
        let n = Int(count)
        guard elementType != 7, n <= GgufFile.maxKeptArrayCount else {
            position += Int(bytes)
            return .skippedArray(count: n)
        }
        var integers: [Int64] = [], floats: [Double] = []
        var representable = true
        for _ in 0..<n {
            switch try value(ofType: elementType) {
            case .integer(let element): integers.append(element)
            case .float(let element): floats.append(element)
            default: representable = false   // a UINT64 above Int64.max
            }
        }
        guard representable else { return .skippedArray(count: n) }
        return elementType == 6 || elementType == 12 ? .floats(floats) : .integers(integers)
    }
}
