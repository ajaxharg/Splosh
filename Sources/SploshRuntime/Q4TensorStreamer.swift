import Foundation
import Metal
import SploshModel
import SploshQuant
import SploshCore

/// A checked byte interval in an SPLW container.
public struct Q4TensorByteRange: Sendable, Equatable {
    public let offset: UInt64
    public let length: UInt64
    public init(offset: UInt64, length: UInt64) { self.offset = offset; self.length = length }
    public var end: UInt64 { offset + length }
}

/// Auditable source for the exact ranges consumed by a Q4 stream chunk.
public protocol Q4TensorRangeReader: Sendable {
    func readRange(offset: UInt64, length: UInt64) throws -> Data
}

extension WeightFile: Q4TensorRangeReader {}

public enum Q4TensorStreamError: Error, Sendable, Equatable, CustomStringConvertible {
    case invalidPlan(String)
    case residentBudgetExceeded(required: UInt64, maximum: UInt64)
    case unsupportedOutputColumnBase(Int)
    case invalidChunk(String)
    case shortRead(expected: UInt64, actual: UInt64, offset: UInt64)
    public var description: String {
        switch self {
        case .invalidPlan(let s): return "invalid q4 stream plan: \(s)"
        case .residentBudgetExceeded(let r, let m): return "q4 stream resident budget exceeded: required \(r), maximum \(m)"
        case .unsupportedOutputColumnBase(let b): return "q4 shader does not support output-column base \(b)"
        case .invalidChunk(let s): return "invalid q4 stream chunk: \(s)"
        case .shortRead(let expected, let actual, let offset): return "q4 stream short read at \(offset): expected \(expected), actual \(actual)"
        }
    }
}

/// A bounded, contiguous output-row transfer plan. Ranges are absolute SPLW offsets.
public struct Q4TensorStreamPlan: Sendable, Equatable {
    public let tensorName: String
    public let logicalShape: [Int]
    public let outputColumnRange: Range<Int>
    public let rowStrideBytes: Int
    public let packedRange: Q4TensorByteRange
    public let sidecarRanges: [Q4TensorByteRange]
    public let maximumResidentBytes: UInt64

    public init(file: WeightFile, tensorName: String, outputColumnRange: Range<Int>, maximumResidentBytes: UInt64) throws {
        guard maximumResidentBytes > 0 else { throw Q4TensorStreamError.invalidPlan("maximum resident bytes must be positive") }
        let record = try file.q4(tensorName)
        guard record.logicalShape.count == 2, record.physicalShape.count == 2 else { throw Q4TensorStreamError.invalidPlan("q4 tensor must be rank-2") }
        guard let container = file.header.tensorRecords.first(where: { $0.name == tensorName }), container.rowStride <= UInt64(Int.max) else { throw Q4TensorStreamError.invalidPlan("missing row stride") }
        let rows = record.logicalShape[0]
        guard outputColumnRange.lowerBound >= 0, outputColumnRange.upperBound <= rows, !outputColumnRange.isEmpty else { throw Q4TensorStreamError.invalidPlan("output-row range is empty or out of bounds") }
        guard file.header.alignment.rowStrideBytes == Q4BufferLayout.defaultRowAlignment else { throw Q4TensorStreamError.invalidPlan("unsupported row alignment") }
        let stride = Int(container.rowStride)
        guard stride > 0, stride % Q4BufferLayout.wordBytes == 0 else { throw Q4TensorStreamError.invalidPlan("invalid row stride") }
        let layout: Q4BufferLayout
        do { layout = try Q4BufferLayout(rows: rows, logicalK: record.logicalShape[1], rowStrideBytes: stride) }
        catch { throw Q4TensorStreamError.invalidPlan("invalid q4 layout: \(error)") }
        guard layout.rows == rows else { throw Q4TensorStreamError.invalidPlan("layout mismatch") }
        let count = outputColumnRange.count
        func multiply(_ a: Int, _ b: Int, _ message: String) throws -> UInt64 {
            guard a >= 0, b >= 0, a <= Int.max / max(b, 1) else { throw Q4TensorStreamError.invalidPlan(message) }
            return UInt64(a * b)
        }
        func add(_ a: UInt64, _ b: UInt64, _ message: String) throws -> UInt64 {
            let (v, overflow) = a.addingReportingOverflow(b); guard !overflow else { throw Q4TensorStreamError.invalidPlan(message) }; return v
        }
        func makeRange(base: Int64?, delta: UInt64, length: UInt64, missing: String, overflow: String) throws -> Q4TensorByteRange {
            guard let raw = base, raw >= 0, let b = UInt64(exactly: raw) else { throw Q4TensorStreamError.invalidPlan(missing) }
            return Q4TensorByteRange(offset: try add(b, delta, overflow), length: length)
        }
        let packedLength = try multiply(count, stride, "packed range overflow")
        let packedDelta = try multiply(outputColumnRange.lowerBound, stride, "packed offset overflow")
        let packed = try makeRange(base: record.weight.byteOffset, delta: packedDelta, length: packedLength, missing: "missing packed offset", overflow: "packed offset overflow")
        let groups = layout.groupsPerRow
        let sidecarLength = try multiply(groups, MemoryLayout<UInt16>.stride, "sidecar range overflow")
        func ranges(for spec: TensorSpec) throws -> [Q4TensorByteRange] {
            var result: [Q4TensorByteRange] = []; result.reserveCapacity(count)
            for row in outputColumnRange {
                let rowGroups = try multiply(row, groups, "sidecar row overflow")
                let rowDelta = try multiply(Int(rowGroups), MemoryLayout<UInt16>.stride, "sidecar row overflow")
                result.append(try makeRange(base: spec.byteOffset, delta: rowDelta, length: sidecarLength, missing: "missing sidecar offset", overflow: "sidecar offset overflow"))
            }
            return result
        }
        let scales = try ranges(for: record.scales)
        let biases = try ranges(for: record.biases)
        func validateContained(_ range: Q4TensorByteRange, in spec: TensorSpec, label: String) throws {
            guard let rawOffset = spec.byteOffset, let rawLength = spec.byteLength,
                  rawOffset >= 0, rawLength >= 0,
                  let base = UInt64(exactly: rawOffset), let length = UInt64(exactly: rawLength) else {
                throw Q4TensorStreamError.invalidPlan("invalid \(label) tensor extent")
            }
            let (end, overflow) = base.addingReportingOverflow(length)
            guard !overflow,
                  range.offset >= base, range.offset <= end,
                  range.length <= end - range.offset else {
                throw Q4TensorStreamError.invalidPlan("planned \(label) range escapes tensor extent")
            }
        }
        try validateContained(packed, in: record.weight, label: "packed")
        for range in scales { try validateContained(range, in: record.scales, label: "scales") }
        for range in biases { try validateContained(range, in: record.biases, label: "biases") }
        func multiplyU(_ a: UInt64, _ b: UInt64, _ message: String) throws -> UInt64 {
            let (v, overflow) = a.multipliedReportingOverflow(by: b); guard !overflow else { throw Q4TensorStreamError.invalidPlan(message) }; return v
        }
        let sidecarResident = try multiplyU(sidecarLength, UInt64(count), "resident size overflow")
        let required = try add(packedLength, try multiplyU(sidecarResident, 2, "resident size overflow"), "resident size overflow")
        guard required <= maximumResidentBytes else { throw Q4TensorStreamError.residentBudgetExceeded(required: required, maximum: maximumResidentBytes) }
        for range in [packed] + scales + biases {
            guard range.offset <= file.fileSize, range.length <= file.fileSize - range.offset else { throw Q4TensorStreamError.invalidPlan("planned range out of file bounds") }
        }
        self.tensorName = tensorName; self.logicalShape = record.logicalShape; self.outputColumnRange = outputColumnRange
        self.rowStrideBytes = stride; self.packedRange = packed; self.sidecarRanges = scales + biases; self.maximumResidentBytes = maximumResidentBytes
    }

    public var requiredResidentBytes: UInt64 {
        let half = sidecarRanges.count / 2
        return packedRange.length + sidecarRanges[half...].reduce(UInt64(0)) { $0 + $1.length } * 2
    }

    /// Partitions a requested output-row range into the largest contiguous chunks
    /// that fit the resident budget. Planning is metadata-only and performs no I/O.
    public static func partition(file: WeightFile, tensorName: String,
                                 maximumResidentBytes: UInt64,
                                 outputColumnRange: Range<Int>) throws -> [Q4TensorStreamPlan] {
        guard maximumResidentBytes > 0 else {
            throw Q4TensorStreamError.invalidPlan("maximum resident bytes must be positive")
        }

        // Validate the requested range before constructing any partial plans.
        // The initializer remains the single source of truth for all geometry,
        // extent, and budget validation.
        let record = try file.q4(tensorName)
        guard record.logicalShape.count == 2, record.physicalShape.count == 2 else {
            throw Q4TensorStreamError.invalidPlan("q4 tensor must be rank-2")
        }
        let rows = record.logicalShape[0]
        guard outputColumnRange.lowerBound >= 0,
              outputColumnRange.upperBound <= rows,
              !outputColumnRange.isEmpty else {
            throw Q4TensorStreamError.invalidPlan("output-row range is empty or out of bounds")
        }

        // Constructing the first one-row plan both validates the complete tensor
        // and establishes that the budget can hold at least one row, before any
        // result is exposed to the caller.
        let firstEnd = outputColumnRange.lowerBound + 1
        _ = try Q4TensorStreamPlan(file: file, tensorName: tensorName,
                                   outputColumnRange: outputColumnRange.lowerBound..<firstEnd,
                                   maximumResidentBytes: maximumResidentBytes)

        var plans: [Q4TensorStreamPlan] = []
        plans.reserveCapacity(outputColumnRange.count)
        var start = outputColumnRange.lowerBound
        while start < outputColumnRange.upperBound {
            let remainingEnd = outputColumnRange.upperBound
            var bestEnd = start + 1
            var probeEnd = bestEnd
            var fittingUpperBound: Int? = nil

            // Exponentially find a fitting upper bound, then binary-search the
            // largest fitting end. This avoids trying every prefix of a chunk.
            while probeEnd < remainingEnd {
                let span = probeEnd - start
                let doubled = span <= (Int.max - start) / 2 ? start + span * 2 : remainingEnd
                let candidateEnd = min(doubled, remainingEnd)
                do {
                    _ = try Q4TensorStreamPlan(file: file, tensorName: tensorName,
                                                outputColumnRange: start..<candidateEnd,
                                                maximumResidentBytes: maximumResidentBytes)
                    bestEnd = candidateEnd
                    probeEnd = candidateEnd
                } catch Q4TensorStreamError.residentBudgetExceeded {
                    fittingUpperBound = candidateEnd
                    break
                }
            }
            if fittingUpperBound == nil && bestEnd == remainingEnd {
                fittingUpperBound = remainingEnd
            }

            var low = bestEnd
            var high = (fittingUpperBound ?? remainingEnd) - 1
            if fittingUpperBound == nil { high = remainingEnd }
            if high < low { high = low }
            while low < high {
                let mid = low + (high - low + 1) / 2
                do {
                    let candidate = try Q4TensorStreamPlan(file: file, tensorName: tensorName,
                                                            outputColumnRange: start..<mid,
                                                            maximumResidentBytes: maximumResidentBytes)
                    _ = candidate
                    low = mid
                    bestEnd = mid
                } catch Q4TensorStreamError.residentBudgetExceeded {
                    high = mid - 1
                }
            }
            let plan = try Q4TensorStreamPlan(file: file, tensorName: tensorName,
                                               outputColumnRange: start..<bestEnd,
                                               maximumResidentBytes: maximumResidentBytes)
            plans.append(plan)
            start = bestEnd
        }
        return plans
    }

    /// Partitions all output rows into budget-fitting contiguous chunks.
    public static func partition(file: WeightFile, tensorName: String,
                                 maximumResidentBytes: UInt64) throws -> [Q4TensorStreamPlan] {
        let record = try file.q4(tensorName)
        guard record.logicalShape.count == 2, record.physicalShape.count == 2 else {
            throw Q4TensorStreamError.invalidPlan("q4 tensor must be rank-2")
        }
        return try partition(file: file, tensorName: tensorName,
                             maximumResidentBytes: maximumResidentBytes,
                             outputColumnRange: 0..<record.logicalShape[0])
    }
}

public struct Q4TensorStreamRead: Sendable, Equatable {
    public let plannedOffset: UInt64
    public let plannedLength: UInt64
    public let actualOffset: UInt64
    public let actualLength: UInt64
    public init(plannedOffset: UInt64, plannedLength: UInt64, actualOffset: UInt64, actualLength: UInt64) {
        self.plannedOffset = plannedOffset; self.plannedLength = plannedLength
        self.actualOffset = actualOffset; self.actualLength = actualLength
    }
}

public struct Q4TensorStreamLoadReport: Sendable, Equatable {
    public let reads: [Q4TensorStreamRead]
    public let plannedTotalBytes: UInt64
    public let actualTotalBytes: UInt64
    public let packedBytes: UInt64
    public let scalesBytes: UInt64
    public let biasesBytes: UInt64
    public let readCount: UInt64
    public let requiredResidentBytes: UInt64
    public let maximumResidentBytes: UInt64
    /// Plan identity and ordered ranges, retained for auditability of a streamed load.
    public let tensorName: String
    public let outputColumnRange: Range<Int>
    public let plannedRanges: [Q4TensorByteRange]
    public let logicalShape: [Int]
    public let rowStrideBytes: Int
    public init(reads: [Q4TensorStreamRead], plannedTotalBytes: UInt64, actualTotalBytes: UInt64,
                packedBytes: UInt64, scalesBytes: UInt64, biasesBytes: UInt64, readCount: UInt64,
                requiredResidentBytes: UInt64, maximumResidentBytes: UInt64,
                tensorName: String = "", outputColumnRange: Range<Int> = 0..<0,
                plannedRanges: [Q4TensorByteRange] = [], logicalShape: [Int] = [], rowStrideBytes: Int = 0) {
        self.reads = reads; self.plannedTotalBytes = plannedTotalBytes; self.actualTotalBytes = actualTotalBytes
        self.packedBytes = packedBytes; self.scalesBytes = scalesBytes; self.biasesBytes = biasesBytes
        self.readCount = readCount; self.requiredResidentBytes = requiredResidentBytes
        self.maximumResidentBytes = maximumResidentBytes; self.tensorName = tensorName
        self.outputColumnRange = outputColumnRange; self.plannedRanges = plannedRanges
        self.logicalShape = logicalShape; self.rowStrideBytes = rowStrideBytes
    }
}

public struct Q4TensorStreamLoad: Sendable {
    public let chunk: Q4TensorStreamChunk
    public let report: Q4TensorStreamLoadReport
    public init(chunk: Q4TensorStreamChunk, report: Q4TensorStreamLoadReport) { self.chunk = chunk; self.report = report }
}

public struct Q4TensorStreamChunk: Sendable {
    public let packed: Data
    public let scales: Data
    public let biases: Data
    public let outputColumnRange: Range<Int>
    private let expectedPlan: Q4TensorStreamPlan?

    public init(packed: Data, scales: Data, biases: Data, outputColumnRange: Range<Int>) {
        self.packed = packed; self.scales = scales; self.biases = biases; self.outputColumnRange = outputColumnRange
        self.expectedPlan = nil
    }

    init(packed: Data, scales: Data, biases: Data, outputColumnRange: Range<Int>, plan: Q4TensorStreamPlan) {
        self.packed = packed; self.scales = scales; self.biases = biases; self.outputColumnRange = outputColumnRange
        self.expectedPlan = plan
    }

    /// Verifies that this chunk exactly fulfils the ranges and resident budget of a plan.
    public func validate(against plan: Q4TensorStreamPlan) throws {
        guard outputColumnRange == plan.outputColumnRange else {
            throw Q4TensorStreamError.invalidChunk("output column range does not match plan")
        }
        guard UInt64(packed.count) == plan.packedRange.length else {
            throw Q4TensorStreamError.invalidChunk("packed byte count does not match planned range")
        }
        let half = plan.sidecarRanges.count / 2
        guard plan.sidecarRanges.count.isMultiple(of: 2) else {
            throw Q4TensorStreamError.invalidChunk("sidecar ranges are not split evenly")
        }
        func total(_ ranges: ArraySlice<Q4TensorByteRange>) throws -> UInt64 {
            var result: UInt64 = 0
            for range in ranges {
                let (next, overflow) = result.addingReportingOverflow(range.length)
                guard !overflow else { throw Q4TensorStreamError.invalidChunk("planned sidecar total overflow") }
                result = next
            }
            return result
        }
        let scalesTotal = try total(plan.sidecarRanges[..<half])
        let biasesTotal = try total(plan.sidecarRanges[half...])
        guard UInt64(scales.count) == scalesTotal else {
            throw Q4TensorStreamError.invalidChunk("scales byte count does not match planned ranges")
        }
        guard UInt64(biases.count) == biasesTotal else {
            throw Q4TensorStreamError.invalidChunk("biases byte count does not match planned ranges")
        }
        let (sidecars, sidecarOverflow) = scalesTotal.addingReportingOverflow(biasesTotal)
        let (total, totalOverflow) = UInt64(packed.count).addingReportingOverflow(sidecars)
        guard !sidecarOverflow, !totalOverflow else { throw Q4TensorStreamError.invalidChunk("resident total overflow") }
        guard total == plan.requiredResidentBytes, total <= plan.maximumResidentBytes else {
            if total > plan.maximumResidentBytes {
                throw Q4TensorStreamError.residentBudgetExceeded(required: total, maximum: plan.maximumResidentBytes)
            }
            throw Q4TensorStreamError.invalidChunk("resident total does not match plan")
        }
    }

    /// Validates a chunk against the plan captured when it was loaded.
    func validate() throws {
        guard let expectedPlan else { throw Q4TensorStreamError.invalidChunk("chunk has no stream plan") }
        try validate(against: expectedPlan)
    }
}

public final class Q4TensorStreamer: @unchecked Sendable {
    private let device: MTLDevice
    private let runtime: Q4GemmRuntime
    public init(runtime: Q4GemmRuntime, device: MTLDevice) { self.runtime = runtime; self.device = device }

    public func loadChunk(_ file: WeightFile, plan: Q4TensorStreamPlan) throws -> Q4TensorStreamChunk {
        try loadChunk(reader: file, plan: plan)
    }

    public func loadChunk<R: Q4TensorRangeReader>(reader: R, plan: Q4TensorStreamPlan) throws -> Q4TensorStreamChunk {
        try loadChunkWithReport(reader: reader, plan: plan).chunk
    }

    public func loadChunkWithReport<R: Q4TensorRangeReader>(reader: R, plan: Q4TensorStreamPlan) throws -> Q4TensorStreamLoad {
        // Keep this check before any I/O.
        guard plan.requiredResidentBytes <= plan.maximumResidentBytes else {
            throw Q4TensorStreamError.residentBudgetExceeded(required: plan.requiredResidentBytes, maximum: plan.maximumResidentBytes)
        }
        let half = plan.sidecarRanges.count / 2
        let allRanges = [plan.packedRange] + plan.sidecarRanges
        var plannedTotal: UInt64 = 0
        for range in allRanges {
            let (end, overflow) = range.offset.addingReportingOverflow(range.length)
            guard !overflow else { throw Q4TensorStreamError.invalidPlan("planned range end overflow") }
            let (next, totalOverflow) = plannedTotal.addingReportingOverflow(range.length)
            guard !totalOverflow else { throw Q4TensorStreamError.invalidPlan("planned total overflow") }
            plannedTotal = next
            _ = end
        }
        var reads: [Q4TensorStreamRead] = []; reads.reserveCapacity(allRanges.count)
        var actualTotal: UInt64 = 0
        func read(_ range: Q4TensorByteRange) throws -> Data {
            let data = try reader.readRange(offset: range.offset, length: range.length)
            let actual = UInt64(data.count)
            guard actual == range.length else { throw Q4TensorStreamError.shortRead(expected: range.length, actual: actual, offset: range.offset) }
            let (next, overflow) = actualTotal.addingReportingOverflow(actual)
            guard !overflow else { throw Q4TensorStreamError.invalidChunk("actual total overflow") }
            actualTotal = next
            reads.append(Q4TensorStreamRead(plannedOffset: range.offset, plannedLength: range.length, actualOffset: range.offset, actualLength: actual))
            return data
        }
        let packed = try read(plan.packedRange)
        func join(_ ranges: ArraySlice<Q4TensorByteRange>) throws -> Data {
            var result = Data()
            for range in ranges { result.append(try read(range)) }
            return result
        }
        let scales = try join(plan.sidecarRanges[..<half])
        let biases = try join(plan.sidecarRanges[half...])
        guard actualTotal == plannedTotal else { throw Q4TensorStreamError.invalidChunk("actual total does not equal planned total") }
        guard actualTotal <= plan.maximumResidentBytes else { throw Q4TensorStreamError.residentBudgetExceeded(required: actualTotal, maximum: plan.maximumResidentBytes) }
        let report = Q4TensorStreamLoadReport(reads: reads, plannedTotalBytes: plannedTotal, actualTotalBytes: actualTotal,
            packedBytes: UInt64(packed.count), scalesBytes: UInt64(scales.count), biasesBytes: UInt64(biases.count),
            readCount: UInt64(reads.count), requiredResidentBytes: plan.requiredResidentBytes, maximumResidentBytes: plan.maximumResidentBytes,
            tensorName: plan.tensorName, outputColumnRange: plan.outputColumnRange,
            plannedRanges: allRanges, logicalShape: plan.logicalShape, rowStrideBytes: plan.rowStrideBytes)
        let chunk = Q4TensorStreamChunk(packed: packed, scales: scales, biases: biases, outputColumnRange: plan.outputColumnRange, plan: plan)
        try chunk.validate(against: plan)
        return Q4TensorStreamLoad(chunk: chunk, report: report)
    }

    /// The current gemm_q4 ABI has no output-column base. Non-zero chunks are therefore rejected.
    public func dispatchChunk(_ chunk: Q4TensorStreamChunk, outputColumnBase: Int, shape: GemmQ4Shape, layout: Q4BufferLayout, a: MTLBuffer, output: MTLBuffer, commandQueue: MTLCommandQueue) throws {
        // Preserve the ABI's explicit unsupported-base error before validation/allocation.
        guard outputColumnBase == 0 else { throw Q4TensorStreamError.unsupportedOutputColumnBase(outputColumnBase) }
        guard chunk.outputColumnRange.lowerBound == 0 else {
            throw Q4TensorStreamError.invalidChunk("output-row range must start at zero")
        }
        try chunk.validate()
        guard shape.columns == chunk.outputColumnRange.count else {
            throw Q4TensorStreamError.invalidChunk("shape columns do not match output-row range")
        }
        guard layout.rows == chunk.outputColumnRange.count else {
            throw Q4TensorStreamError.invalidChunk("layout rows do not match output-row range")
        }
        guard layout.logicalK == shape.inner else {
            throw Q4TensorStreamError.invalidChunk("layout logical K does not match shape inner dimension")
        }
        guard layout.orientation == .rowsByK else {
            throw Q4TensorStreamError.invalidChunk("layout orientation is not rows-by-K")
        }
        guard layout.rowStrideBytes >= layout.packedWordsPerRow * Q4BufferLayout.wordBytes,
              layout.rowStrideBytes % Q4BufferLayout.wordBytes == 0 else {
            throw Q4TensorStreamError.invalidChunk("layout packed row geometry is invalid")
        }
        guard layout.groupsPerRow == layout.groupStride,
              layout.sidecarCount == layout.rows * layout.groupStride else {
            throw Q4TensorStreamError.invalidChunk("layout sidecar geometry is invalid")
        }
        let expectedPackedBytes: Int
        let expectedSidecarBytes: Int
        guard layout.packedWordCount <= Int.max / Q4BufferLayout.wordBytes,
              layout.sidecarCount <= Int.max / MemoryLayout<UInt16>.stride else {
            throw Q4TensorStreamError.invalidChunk("layout byte geometry overflows")
        }
        expectedPackedBytes = layout.packedWordCount * Q4BufferLayout.wordBytes
        expectedSidecarBytes = layout.sidecarCount * MemoryLayout<UInt16>.stride
        guard chunk.packed.count == expectedPackedBytes else {
            throw Q4TensorStreamError.invalidChunk("packed bytes do not match local layout geometry")
        }
        guard chunk.scales.count == expectedSidecarBytes,
              chunk.biases.count == expectedSidecarBytes else {
            throw Q4TensorStreamError.invalidChunk("sidecar bytes do not match local layout geometry")
        }
        let packed: MTLBuffer? = chunk.packed.withUnsafeBytes { bytes in
            guard let base = bytes.baseAddress else { return nil }
            return device.makeBuffer(bytes: base, length: chunk.packed.count, options: .storageModeShared)
        }
        let scales: MTLBuffer? = chunk.scales.withUnsafeBytes { bytes in
            guard let base = bytes.baseAddress else { return nil }
            return device.makeBuffer(bytes: base, length: chunk.scales.count, options: .storageModeShared)
        }
        let biases: MTLBuffer? = chunk.biases.withUnsafeBytes { bytes in
            guard let base = bytes.baseAddress else { return nil }
            return device.makeBuffer(bytes: base, length: chunk.biases.count, options: .storageModeShared)
        }
        guard let packed, let scales, let biases else { throw SploshError.capabilityGateFailure("unable to allocate q4 stream buffers") }
        try runtime.dispatch(shape: shape, layout: layout, a: a, packed: packed, scales: scales, biases: biases, output: output, commandQueue: commandQueue)
    }

    /// `[shape.rows, requestedRange.count]`, with columns in absolute requested
    /// range order. Reports and ranges are in dispatch order.
    public struct PartitionedResult: Sendable {
        public let values: [Float]
        public let reports: [Q4TensorStreamLoadReport]
        public let ranges: [Range<Int>]

        public init(values: [Float], reports: [Q4TensorStreamLoadReport], ranges: [Range<Int>]) {
            self.values = values
            self.reports = reports
            self.ranges = ranges
        }
    }

    private func checkedMultiply(_ lhs: Int, _ rhs: Int, _ message: String) throws -> Int {
        guard lhs >= 0, rhs >= 0, lhs <= Int.max / max(rhs, 1) else {
            throw Q4TensorStreamError.invalidPlan(message)
        }
        return lhs * rhs
    }

    private func checkedAdd(_ lhs: Int, _ rhs: Int, _ message: String) throws -> Int {
        guard lhs >= 0, rhs >= 0, lhs <= Int.max - rhs else {
            throw Q4TensorStreamError.invalidPlan(message)
        }
        return lhs + rhs
    }

    /// Loads and dispatches budget-fitting chunks using the existing zero-based
    /// q4 ABI, then copies each local output into one absolute host result.
    /// The input activation buffer is borrowed; chunk and output buffers are
    /// local to this call and are safe to release after `runtime.dispatch`
    /// returns because that API waits for command completion.
    public func dispatchPartitioned(
        file: WeightFile,
        tensorName: String,
        outputColumnRange: Range<Int>,
        maximumResidentBytes: UInt64,
        shape: GemmQ4Shape,
        a: MTLBuffer,
        commandQueue: MTLCommandQueue
    ) throws -> PartitionedResult {
        // Partitioning performs all range, geometry, and resident-budget
        // validation before the first read or dispatch.
        let plans = try Q4TensorStreamPlan.partition(file: file, tensorName: tensorName,
                                                      maximumResidentBytes: maximumResidentBytes,
                                                      outputColumnRange: outputColumnRange)
        guard !plans.isEmpty else {
            throw Q4TensorStreamError.invalidPlan("stream partition produced no plans")
        }
        guard let first = plans.first,
              first.logicalShape.count == 2,
              shape.rows > 0, shape.inner == first.logicalShape[1],
              shape.columns == outputColumnRange.count else {
            throw Q4TensorStreamError.invalidPlan("GEMM shape does not match requested tensor range")
        }
        let activationElements = try checkedMultiply(shape.rows, shape.inner, "activation element count overflow")
        let activationBytes = try checkedMultiply(activationElements, MemoryLayout<UInt16>.stride, "activation byte count overflow")
        guard a.length >= activationBytes else {
            throw GemmQ4Error.bufferTooSmall(name: "a", expected: activationBytes, observed: a.length)
        }
        let outputCount = try checkedMultiply(shape.rows, outputColumnRange.count, "host result size overflow")
        var values = Array(repeating: Float.zero, count: outputCount)
        var reports: [Q4TensorStreamLoadReport] = []
        var ranges: [Range<Int>] = []
        var previousEnd = outputColumnRange.lowerBound
        reports.reserveCapacity(plans.count)
        ranges.reserveCapacity(plans.count)

        for plan in plans {
            guard !plan.outputColumnRange.isEmpty,
                  plan.outputColumnRange.lowerBound >= outputColumnRange.lowerBound,
                  plan.outputColumnRange.upperBound <= outputColumnRange.upperBound,
                  plan.outputColumnRange.lowerBound == previousEnd else {
                throw Q4TensorStreamError.invalidPlan("stream plans are not ordered contiguous subranges")
            }
            previousEnd = plan.outputColumnRange.upperBound
            guard plan.logicalShape.count == 2,
                  plan.logicalShape[1] == shape.inner,
                  plan.rowStrideBytes > 0 else {
                throw Q4TensorStreamError.invalidPlan("stream plan geometry mismatch")
            }
            let packedBytes = try checkedMultiply(plan.outputColumnRange.count, plan.rowStrideBytes, "packed bytes overflow")
            let sidecarCount = try checkedMultiply(plan.outputColumnRange.count, 2, "sidecar range count overflow")
            guard plan.packedRange.length == UInt64(packedBytes),
                  plan.sidecarRanges.count == sidecarCount else {
                throw Q4TensorStreamError.invalidPlan("stream plan geometry mismatch")
            }
            let load = try loadChunkWithReport(reader: file, plan: plan)
            let chunk = load.chunk
            let localLayout = try Q4BufferLayout(rows: plan.outputColumnRange.count,
                                                  logicalK: shape.inner,
                                                  rowStrideBytes: plan.rowStrideBytes)
            let localShape = try GemmQ4Shape(rows: shape.rows,
                                             columns: plan.outputColumnRange.count,
                                             inner: shape.inner)
            let localCount = try checkedMultiply(shape.rows, plan.outputColumnRange.count, "local output element count overflow")
            let localOutputBytes = try checkedMultiply(localCount, MemoryLayout<Float>.stride, "local output byte count overflow")
            guard let output = device.makeBuffer(length: localOutputBytes,
                                                  options: .storageModeShared) else {
                throw SploshError.capabilityGateFailure("unable to allocate q4 stream output buffer")
            }
            try dispatchLocalChunk(chunk, shape: localShape, layout: localLayout,
                                   a: a, output: output, commandQueue: commandQueue)
            let local = output.contents().bindMemory(to: Float.self, capacity: localCount)
            let localColumns = plan.outputColumnRange.count
            let destinationColumn = plan.outputColumnRange.lowerBound - outputColumnRange.lowerBound
            let destinationEnd = try checkedAdd(destinationColumn, localColumns, "destination interval overflow")
            guard destinationColumn >= 0, destinationEnd <= outputColumnRange.count else {
                throw Q4TensorStreamError.invalidPlan("destination interval escapes result")
            }
            for row in 0..<shape.rows {
                let sourceIndex = try checkedMultiply(row, localColumns, "source interval overflow")
                let destinationRow = try checkedMultiply(row, outputColumnRange.count, "destination interval overflow")
                let destinationIndex = try checkedAdd(destinationRow, destinationColumn, "destination interval overflow")
                guard sourceIndex <= localCount - localColumns,
                      destinationIndex <= outputCount - localColumns else {
                    throw Q4TensorStreamError.invalidPlan("copy interval escapes buffer")
                }
                let source = local.advanced(by: sourceIndex)
                values.withUnsafeMutableBufferPointer { destination in
                    destination.baseAddress!.advanced(by: destinationIndex).assign(from: source, count: plan.outputColumnRange.count)
                }
            }
            reports.append(load.report)
            ranges.append(plan.outputColumnRange)
        }
        guard previousEnd == outputColumnRange.upperBound else {
            throw Q4TensorStreamError.invalidPlan("stream plans do not cover requested range")
        }
        return PartitionedResult(values: values, reports: reports, ranges: ranges)
    }

    private func dispatchLocalChunk(_ chunk: Q4TensorStreamChunk, shape: GemmQ4Shape,
                                    layout: Q4BufferLayout, a: MTLBuffer, output: MTLBuffer,
                                    commandQueue: MTLCommandQueue) throws {
        try chunk.validate()
        guard chunk.outputColumnRange.lowerBound >= 0,
              chunk.outputColumnRange.count == layout.rows,
              shape.columns == layout.rows else {
            throw Q4TensorStreamError.invalidChunk("local chunk geometry mismatch")
        }
        let packed = chunk.packed.withUnsafeBytes { bytes in
            bytes.baseAddress.flatMap { device.makeBuffer(bytes: $0, length: chunk.packed.count, options: .storageModeShared) }
        }
        let scales = chunk.scales.withUnsafeBytes { bytes in
            bytes.baseAddress.flatMap { device.makeBuffer(bytes: $0, length: chunk.scales.count, options: .storageModeShared) }
        }
        let biases = chunk.biases.withUnsafeBytes { bytes in
            bytes.baseAddress.flatMap { device.makeBuffer(bytes: $0, length: chunk.biases.count, options: .storageModeShared) }
        }
        guard let packedBuffer = packed, let scalesBuffer = scales, let biasesBuffer = biases else {
            throw SploshError.capabilityGateFailure("unable to allocate q4 stream buffers")
        }
        try runtime.dispatch(shape: shape, layout: layout, a: a, packed: packedBuffer,
                             scales: scalesBuffer, biases: biasesBuffer, output: output,
                             commandQueue: commandQueue)
    }
}
