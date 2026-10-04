// Allocator.swift — resident, mmap-backed weight storage.
//
// The q4 artifact is ~15 GiB and the observed working set on this host is ~51.8 GiB, so the
// whole model stays resident for the lifetime of the process. Nothing is streamed and nothing
// is copied: the container is mapped once and Metal buffers are created `bytesNoCopy` over the
// mapping, so the GPU reads the same physical pages the page cache holds.
//
// `MTLDevice.maxBufferLength` is smaller than the artifact on some configurations, so the
// payload is divided into spans. Split points are always tensor boundaries, which keeps every
// tensor contiguous inside exactly one buffer.

import Foundation
@preconcurrency import Metal

public enum ResidentWeightsError: Error, CustomStringConvertible {
    case mapFailed(String)
    case bufferCreationFailed(span: Int, bytes: Int)
    case tensorTooLarge(name: String, bytes: Int, limit: Int)
    case missingTensor(String)
    case unalignedSpan(Int)

    public var description: String {
        switch self {
        case .mapFailed(let reason): return "unable to map weight container: \(reason)"
        case .bufferCreationFailed(let span, let bytes): return "unable to create Metal buffer for span \(span) (\(bytes) bytes)"
        case .tensorTooLarge(let name, let bytes, let limit): return "tensor \(name) is \(bytes) bytes, exceeding the device buffer limit of \(limit)"
        case .missingTensor(let name): return "resident weights do not contain \(name)"
        case .unalignedSpan(let offset): return "span offset \(offset) is not page aligned"
        }
    }
}

/// Where one tensor lives in GPU-addressable memory.
public struct TensorHandle: Sendable, Equatable {
    public let name: String
    /// Index into `ResidentWeights.buffers`.
    public let span: Int
    /// Byte offset within that buffer. Always a multiple of the container's pointer alignment.
    public let offset: Int
    public let byteLength: Int
    public let shape: [Int]

    public init(name: String, span: Int, offset: Int, byteLength: Int, shape: [Int]) {
        self.name = name; self.span = span; self.offset = offset
        self.byteLength = byteLength; self.shape = shape
    }
}

/// An MLX affine weight resolved to its three resident buffers plus the geometry the kernels
/// need. The name is from when every weight was 4-bit; `bits` says which this one is.
///
/// A GGUF weight is held the same way, its three planes where the kernels bind them: plane 0 as
/// `packed`, plane 1 as `scales` (empty for a format that has none) and the meta plane as
/// `biases`. `kernel` then names its format and `bits` is 0.
public struct Q4Handle: Sendable, Equatable {
    public let packed: TensorHandle
    public let scales: TensorHandle
    public let biases: TensorHandle
    /// Output rows of the logical matrix (`[rows, inner]`). A GGUF weight's planes hold these
    /// rounded up to whole tiles of 128.
    public let rows: Int
    /// Logical inner dimension, i.e. the contracted axis.
    public let inner: Int
    /// U32 words per packed row: eight logical columns per word at 4 bits, four at 8. For a
    /// GGUF weight, the words of plane 0 a row.
    public let rowStrideWords: Int
    /// Affine groups per row, i.e. `inner / 64`. For a GGUF weight, its groups of 32 a row:
    /// `inner / 32`, the `groups` of its kernels.
    public let groupsPerRow: Int
    /// Bits per code: 4 or 8; 0 for a GGUF weight, which no affine kernel can read.
    public let bits: Int
    /// The format's token in the names of the GGUF kernels, as the `q5k` of `sp_gguf_wide_q5k`;
    /// nil for an MLX affine weight.
    public let kernel: String?

    public init(packed: TensorHandle, scales: TensorHandle, biases: TensorHandle,
                rows: Int, inner: Int, rowStrideWords: Int, groupsPerRow: Int, bits: Int = 4, kernel: String? = nil) {
        self.packed = packed; self.scales = scales; self.biases = biases
        self.rows = rows; self.inner = inner
        self.rowStrideWords = rowStrideWords; self.groupsPerRow = groupsPerRow
        self.bits = bits; self.kernel = kernel
    }

    public var residentBytes: Int { packed.byteLength + scales.byteLength + biases.byteLength }
}

/// The whole model, mapped once and held resident.
public final class ResidentWeights: @unchecked Sendable {
    public let buffers: [MTLBuffer]
    /// Total bytes the GPU can address through `buffers`, i.e. the model's footprint.
    public let residentBytes: Int
    /// Bytes of the mapping that are padding rather than tensor payload.
    public let paddingBytes: Int

    private let handles: [String: TensorHandle]
    private let mapping: UnsafeMutableRawPointer
    private let mappingLength: Int

    /// Spans are kept well under the device limit so a span boundary never splits a tensor.
    public init(device: MTLDevice, records: [ResidentTensorRecord], containerURL: URL,
                payloadStart: Int, payloadLength: Int) throws {
        let pageSize = Int(getpagesize())
        // Map from the start of the file so span offsets stay page aligned relative to the
        // mapping base; the header is a few hundred KiB and costs nothing to carry.
        let totalLength = payloadStart + payloadLength
        let fd = open(containerURL.path, O_RDONLY)
        guard fd >= 0 else { throw ResidentWeightsError.mapFailed(String(cString: strerror(errno))) }
        defer { close(fd) }
        let mapLength = ((totalLength + pageSize - 1) / pageSize) * pageSize
        guard let base = mmap(nil, mapLength, PROT_READ, MAP_PRIVATE, fd, 0),
              base != UnsafeMutableRawPointer(bitPattern: UInt(bitPattern: -1)) else {
            throw ResidentWeightsError.mapFailed(String(cString: strerror(errno)))
        }
        // Sequential-then-random is the honest access pattern: load touches everything once,
        // then decode revisits every layer each step.
        madvise(base, mapLength, MADV_WILLNEED)
        self.mapping = base
        self.mappingLength = mapLength

        let limit = device.maxBufferLength
        // Leave headroom so a span is never exactly at the limit.
        let spanTarget = max(pageSize, (limit / pageSize) * pageSize)

        var spans: [(start: Int, end: Int)] = []
        var resolved: [String: TensorHandle] = [:]
        var spanStart = (payloadStart / pageSize) * pageSize
        var spanEnd = spanStart
        var payloadBytes = 0

        for record in records.sorted(by: { $0.payloadOffset < $1.payloadOffset }) {
            let begin = record.payloadOffset
            let end = begin + record.byteLength
            guard record.byteLength <= spanTarget else {
                munmap(base, mapLength)
                throw ResidentWeightsError.tensorTooLarge(name: record.name, bytes: record.byteLength, limit: spanTarget)
            }
            if end - spanStart > spanTarget {
                // Close the current span on a page boundary at or before this tensor.
                spans.append((spanStart, spanEnd))
                spanStart = (begin / pageSize) * pageSize
                spanEnd = spanStart
            }
            spanEnd = max(spanEnd, end)
            resolved[record.name] = TensorHandle(name: record.name, span: spans.count,
                                                offset: begin - spanStart,
                                                byteLength: record.byteLength, shape: record.shape)
            payloadBytes += record.byteLength
        }
        if spanEnd > spanStart { spans.append((spanStart, spanEnd)) }

        var created: [MTLBuffer] = []
        var residentTotal = 0
        for (index, span) in spans.enumerated() {
            guard span.start % pageSize == 0 else {
                munmap(base, mapLength)
                throw ResidentWeightsError.unalignedSpan(span.start)
            }
            // Round the span length up to a page so `bytesNoCopy` accepts it. The rounding can
            // read past the last tensor but never past the mapping, which is page-rounded too.
            let rawLength = span.end - span.start
            let length = min(((rawLength + pageSize - 1) / pageSize) * pageSize, mapLength - span.start)
            guard let buffer = device.makeBuffer(bytesNoCopy: base.advanced(by: span.start),
                                                 length: length,
                                                 options: .storageModeShared,
                                                 deallocator: nil) else {
                munmap(base, mapLength)
                throw ResidentWeightsError.bufferCreationFailed(span: index, bytes: length)
            }
            buffer.label = "splosh.weights.span\(index)"
            created.append(buffer)
            residentTotal += length
        }

        self.buffers = created
        self.handles = resolved
        self.residentBytes = residentTotal
        self.paddingBytes = residentTotal - payloadBytes
    }

    deinit { munmap(mapping, mappingLength) }

    public func handle(_ name: String) throws -> TensorHandle {
        guard let handle = handles[name] else { throw ResidentWeightsError.missingTensor(name) }
        return handle
    }

    public func buffer(_ handle: TensorHandle) -> MTLBuffer { buffers[handle.span] }

    public var tensorCount: Int { handles.count }

    /// Resolve an MLX affine triplet and derive the kernel geometry from its shapes: a quant
    /// group is 64 columns, so the sidecars give the inner dimension and the packed words per
    /// row then give the code width (8 words a group at 4 bits, 16 at 8).
    public func q4(_ name: String) throws -> Q4Handle { try Self.q4(name, handle: handle) }

    /// The same from any source of handles, so that an artifact's header can be resolved as the
    /// mapping will be without mapping it.
    public static func q4(_ name: String, handle: (String) throws -> TensorHandle) throws -> Q4Handle {
        let base = name.hasSuffix(".weight") ? String(name.dropLast(".weight".count)) : name
        let packed = try handle(name)
        let scales = try handle(base + ".scales")
        let biases = try handle(base + ".biases")
        guard packed.shape.count == 2, scales.shape.count == 2, scales.shape[1] > 0 else {
            throw ResidentWeightsError.missingTensor("\(name) (expected rank-2 affine triplet)")
        }
        let rows = packed.shape[0]
        let words = packed.shape[1]
        let groups = scales.shape[1]
        let bits = words / (2 * groups)
        guard words == 2 * groups * bits, bits == 4 || bits == 8 else {
            throw ResidentWeightsError.missingTensor("\(name) (\(words) words for \(groups) groups a row is neither 4-bit nor 8-bit)")
        }
        return Q4Handle(packed: packed, scales: scales, biases: biases,
                        rows: rows, inner: groups * 64,
                        rowStrideWords: words, groupsPerRow: groups, bits: bits)
    }

    /// Resolve a GGUF weight's planes: `name` is plane 0, `<base>.weight`, with `<base>.meta`
    /// and, where the format has one, `<base>.plane1`. The logical shape and the kernel suffix
    /// are the caller's, from the artifact's header: the planes' own shapes are in bytes and
    /// whole tiles, and say neither.
    public func gguf(_ name: String, kernel: String, rows: Int, inner: Int, secondPlane: Bool) throws -> Q4Handle {
        try Self.gguf(name, kernel: kernel, rows: rows, inner: inner, secondPlane: secondPlane, handle: handle)
    }

    public static func gguf(_ name: String, kernel: String, rows: Int, inner: Int, secondPlane: Bool,
                            handle: (String) throws -> TensorHandle) throws -> Q4Handle {
        let base = name.hasSuffix(".weight") ? String(name.dropLast(".weight".count)) : name
        let plane0 = try handle(name)
        let meta = try handle(base + ".meta")
        guard plane0.shape.count == 2, plane0.shape[1] % 4 == 0, rows > 0, plane0.shape[0] >= rows, inner > 0, inner % 32 == 0 else {
            throw ResidentWeightsError.missingTensor("\(name) (planes of shape \(plane0.shape) do not hold \(rows) rows of \(inner))")
        }
        // A format with no second plane never reads the binding, but something must be bound:
        // an empty handle at plane 0's place.
        let plane1 = secondPlane ? try handle(base + ".plane1")
            : TensorHandle(name: base + ".plane1", span: plane0.span, offset: plane0.offset, byteLength: 0, shape: [plane0.shape[0], 0])
        return Q4Handle(packed: plane0, scales: plane1, biases: meta, rows: rows, inner: inner,
                        rowStrideWords: plane0.shape[1] / 4, groupsPerRow: inner / 32, bits: 0, kernel: kernel)
    }
}

/// The subset of an SPLW tensor record the resident allocator needs, so `SploshCore` does not
/// have to depend on `SploshModel`.
public struct ResidentTensorRecord: Sendable, Equatable {
    public let name: String
    public let payloadOffset: Int
    public let byteLength: Int
    public let shape: [Int]

    public init(name: String, payloadOffset: Int, byteLength: Int, shape: [Int]) {
        self.name = name; self.payloadOffset = payloadOffset
        self.byteLength = byteLength; self.shape = shape
    }
}
