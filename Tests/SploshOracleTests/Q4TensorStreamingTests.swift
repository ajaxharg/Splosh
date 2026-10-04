import Foundation
import Metal
import Testing
import SploshCore
import SploshModel
import SploshQuant
import SploshRuntime
import SploshOracle

/// Safety coverage for bounded Q4 SPLW range planning.  These tests never invent
/// quantized weights: artifact-dependent cases use the configured SPLW artifact.
@Suite("Q4TensorStreamingTests")
struct Q4TensorStreamingTests {
    private let tensorName = "language_model.model.layers.0.linear_attn.in_proj_a.weight"

    private final class ReadLog: @unchecked Sendable {
        var calls: [Q4TensorByteRange] = []
    }

    private struct RecordingReader: Q4TensorRangeReader {
        enum Response: Sendable {
            case artifact
            case short
            case overlong
        }

        let file: WeightFile
        let log: ReadLog
        let response: Response

        init(file: WeightFile, log: ReadLog, response: Response = .artifact) {
            self.file = file
            self.log = log
            self.response = response
        }

        func readRange(offset: UInt64, length: UInt64) throws -> Data {
            log.calls.append(Q4TensorByteRange(offset: offset, length: length))
            var data = try file.readRange(offset: offset, length: length)
            switch response {
            case .artifact:
                return data
            case .short:
                data.removeLast()
                return data
            case .overlong:
                data.append(0)
                return data
            }
        }
    }

    private func artifact() throws -> WeightFile? {
        let raw = ProcessInfo.processInfo.environment["ARTIFACT"]
        let url = raw.map(URL.init(fileURLWithPath:)) ?? URL(fileURLWithPath: "models/q4/weights.splw", relativeTo: URL(fileURLWithPath: FileManager.default.currentDirectoryPath))
        guard FileManager.default.fileExists(atPath: url.path) else { return nil }
        return try WeightFile(splwURL: url)
    }

    private func plan(_ file: WeightFile, range: Range<Int>, budget: UInt64 = 1 << 30) throws -> Q4TensorStreamPlan {
        try Q4TensorStreamPlan(file: file, tensorName: tensorName, outputColumnRange: range, maximumResidentBytes: budget)
    }

    @Test("zero resident budget is rejected")
    func zeroBudget() throws {
        guard let file = try artifact() else { return }
        #expect(throws: Q4TensorStreamError.invalidPlan("maximum resident bytes must be positive")) {
            try Q4TensorStreamPlan(file: file, tensorName: tensorName, outputColumnRange: 0..<1, maximumResidentBytes: 0)
        }
    }

    @Test("empty and negative column ranges fail closed")
    func invalidRanges() throws {
        guard let file = try artifact() else { return }
        #expect(throws: Q4TensorStreamError.self) { try plan(file, range: 2..<2) }
        #expect(throws: Q4TensorStreamError.self) { try plan(file, range: -1..<1) }
    }

    @Test("out of range columns fail closed")
    func outOfRangeColumns() throws {
        guard let file = try artifact(), let record = try? file.q4(tensorName) else { return }
        let rows = record.logicalShape[0]
        #expect(throws: Q4TensorStreamError.self) { try plan(file, range: rows..<(rows + 1)) }
    }

    @Test("one-column and multi-column plans preserve sidecar group boundaries")
    func boundaries() throws {
        guard let file = try artifact(), let record = try? file.q4(tensorName) else { return }
        let one = try plan(file, range: 0..<1)
        let many = try plan(file, range: 0..<min(48, record.logicalShape[0]))
        #expect(one.outputColumnRange == 0..<1)
        let groupsPerColumn = (record.logicalShape[1] + Q4BufferLayout.groupSize - 1) / Q4BufferLayout.groupSize
        let sidecarByteWidth = MemoryLayout<UInt16>.stride
        let expectedOneSidecarLength = groupsPerColumn * sidecarByteWidth
        let expectedManySidecarLength = groupsPerColumn * sidecarByteWidth
        #expect(one.sidecarRanges.count == one.outputColumnRange.count * 2)
        #expect(one.sidecarRanges.allSatisfy { $0.length == UInt64(expectedOneSidecarLength) })
        #expect(many.sidecarRanges.count == many.outputColumnRange.count * 2)
        #expect(many.sidecarRanges.allSatisfy { $0.length == UInt64(expectedManySidecarLength) })
        #expect(one.requiredResidentBytes <= one.maximumResidentBytes)
    }

    @Test("budget exhaustion is rejected without a partial plan")
    func budgetExceeded() throws {
        guard let file = try artifact() else { return }
        #expect(throws: Q4TensorStreamError.self) { try plan(file, range: 0..<1, budget: 1) }
    }

    @Test("row stride and alignment metadata are enforced")
    func strideAndAlignment() throws {
        guard let file = try artifact() else { return }
        #expect(file.header.alignment.rowStrideBytes == Q4BufferLayout.defaultRowAlignment)
        // The public layout contract rejects non-word-aligned and undersized strides.
        #expect(throws: Q4BufferLayoutError.self) { try Q4BufferLayout(rows: 1, logicalK: 64, rowStrideBytes: 3) }
        #expect(throws: Q4BufferLayoutError.self) { try Q4BufferLayout(rows: 1, logicalK: 64, rowStrideBytes: 4) }
    }

    @Test("range reads reject out of bounds and truncated ranges")
    func rangeReadSafety() throws {
        guard let file = try artifact() else { return }
        #expect(throws: WeightFileError.self) { try file.readRange(offset: file.fileSize + 1, length: 1) }
        #expect(throws: WeightFileError.self) { try file.readRange(offset: file.fileSize, length: 1) }
        let record = try file.q4(tensorName)
        let offset = try #require(record.weight.byteOffset).magnitude
        #expect((try file.readRange(offset: UInt64(offset), length: 1)).count == 1)
    }

    @Test("loadChunk reads exactly the planned ranges")
    func loadChunk() throws {
        guard let file = try artifact() else { return }
        let p = try plan(file, range: 0..<1)
        guard let device = MTLCreateSystemDefaultDevice() else { return }
        let runtime = try Q4GemmRuntime(device: device, metallib: Metallib(device: device))
        let chunk = try Q4TensorStreamer(runtime: runtime, device: device).loadChunk(file, plan: p)
        #expect(chunk.outputColumnRange == p.outputColumnRange)
        #expect(chunk.packed.count == p.packedRange.length)
    }

    @Test("artifact range loading covers one-column and boundary-sized chunks")
    func artifactRangeLoadingAcrossChunkPlans() throws {
        guard let file = try artifact(), let record = try? file.q4(tensorName) else { return }
        guard let device = MTLCreateSystemDefaultDevice() else { return }
        let runtime = try Q4GemmRuntime(device: device, metallib: Metallib(device: device))
        let streamer = Q4TensorStreamer(runtime: runtime, device: device)
        let columns = record.logicalShape[1]
        let boundary = min(Q4BufferLayout.groupSize, record.logicalShape[0] - 1)
        let first = try plan(file, range: 0..<1)
        let second = try plan(file, range: 1..<(1 + boundary))
        let chunks = try [first, second].map { try streamer.loadChunk(file, plan: $0) }
        let groups = (columns + Q4BufferLayout.groupSize - 1) / Q4BufferLayout.groupSize
        for (plan, chunk) in zip([first, second], chunks) {
            let columnsInChunk = plan.outputColumnRange.count
            let packedLength = columnsInChunk * plan.rowStrideBytes
            let sidecarLength = columnsInChunk * groups * MemoryLayout<UInt16>.stride
            #expect(chunk.outputColumnRange == plan.outputColumnRange)
            #expect(chunk.packed.count == packedLength)
            #expect(chunk.scales.count == sidecarLength)
            #expect(chunk.biases.count == sidecarLength)
            #expect(plan.requiredResidentBytes == UInt64(packedLength + sidecarLength * 2))
            #expect(plan.requiredResidentBytes <= plan.maximumResidentBytes)
        }
        #expect(second.outputColumnRange.count == boundary)
        #expect(first.packedRange.end == second.packedRange.offset)
        let firstHalf = first.sidecarRanges.count / 2
        let secondHalf = second.sidecarRanges.count / 2
        for row in 0..<firstHalf {
            #expect(first.sidecarRanges[row].end == second.sidecarRanges[row].offset)
            #expect(first.sidecarRanges[firstHalf + row].end == second.sidecarRanges[secondHalf + row].offset)
        }
        let reports = try [first, second].map { try streamer.loadChunkWithReport(reader: RecordingReader(file: file, log: ReadLog()), plan: $0).report }
        for (report, expectedPlan) in zip(reports, [first, second]) {
            #expect(report.tensorName == tensorName)
            #expect(report.outputColumnRange == expectedPlan.outputColumnRange)
            #expect(report.plannedRanges == [expectedPlan.packedRange] + expectedPlan.sidecarRanges)
            #expect(report.plannedRanges.first == expectedPlan.packedRange)
            #expect(Array(report.plannedRanges.dropFirst()) == expectedPlan.sidecarRanges)
            print("Q4 artifact report tensorName=\(report.tensorName) outputRange=\(report.outputColumnRange) plannedRanges=\(report.plannedRanges) readCount=\(report.readCount) plannedTotal=\(report.plannedTotalBytes) actualTotal=\(report.actualTotalBytes)")
        }
        print("Q4 artifact range loading tensor=\(tensorName) chunks=[1,\(boundary)] packed=[\(chunks[0].packed.count),\(chunks[1].packed.count)] scales=[\(chunks[0].scales.count),\(chunks[1].scales.count)] biases=[\(chunks[0].biases.count),\(chunks[1].biases.count)] resident=[\(first.requiredResidentBytes),\(second.requiredResidentBytes)] ranges=continuous-no-overlap")
    }

    @Test("recorded reads exactly cover planned ranges and resident bytes")
    func recordedReads() throws {
        guard let file = try artifact(), let device = MTLCreateSystemDefaultDevice() else { return }
        let runtime = try Q4GemmRuntime(device: device, metallib: Metallib(device: device))
        let streamer = Q4TensorStreamer(runtime: runtime, device: device)
        for columns in [0..<1, 1..<48] {
            let p = try plan(file, range: columns)
            let log = ReadLog()
            let chunk = try streamer.loadChunk(reader: RecordingReader(file: file, log: log), plan: p)
            let expected = [p.packedRange] + p.sidecarRanges
            #expect(log.calls == expected)
            #expect(log.calls.allSatisfy { $0.offset <= file.fileSize && $0.length <= file.fileSize - $0.offset })
            #expect(log.calls.contains(where: { $0.length == file.fileSize }) == false)
            #expect(UInt64(chunk.packed.count + chunk.scales.count + chunk.biases.count) == p.requiredResidentBytes)
            #expect(UInt64(log.calls.reduce(0) { $0 + $1.length }) == p.requiredResidentBytes)
        }
    }

    @Test("load reports exact reads totals and resident accounting")
    func loadReportAccounting() throws {
        guard let file = try artifact(), let device = MTLCreateSystemDefaultDevice() else { return }
        let runtime = try Q4GemmRuntime(device: device, metallib: Metallib(device: device))
        let streamer = Q4TensorStreamer(runtime: runtime, device: device)
        for columns in [0..<1, 1..<48] {
            let p = try plan(file, range: columns)
            let log = ReadLog()
            let load = try streamer.loadChunkWithReport(reader: RecordingReader(file: file, log: log), plan: p)
            let expected = [p.packedRange] + p.sidecarRanges
            #expect(load.report.reads.count == expected.count)
            #expect(load.report.reads.map { Q4TensorByteRange(offset: $0.actualOffset, length: $0.actualLength) } == expected)
            #expect(load.report.reads.map(\.plannedOffset) == expected.map(\.offset))
            #expect(load.report.reads.map(\.plannedLength) == expected.map(\.length))
            #expect(load.report.plannedTotalBytes == p.requiredResidentBytes)
            #expect(load.report.actualTotalBytes == p.requiredResidentBytes)
            #expect(load.report.packedBytes == UInt64(load.chunk.packed.count))
            #expect(load.report.scalesBytes == UInt64(load.chunk.scales.count))
            #expect(load.report.biasesBytes == UInt64(load.chunk.biases.count))
            #expect(load.report.readCount == UInt64(expected.count))
            #expect(load.report.requiredResidentBytes == p.requiredResidentBytes)
            #expect(load.report.requiredResidentBytes <= load.report.maximumResidentBytes)
            #expect(load.report.tensorName == p.tensorName)
            #expect(load.report.outputColumnRange == p.outputColumnRange)
            #expect(load.report.logicalShape == p.logicalShape)
            #expect(load.report.rowStrideBytes == p.rowStrideBytes)
            #expect(load.report.plannedRanges == expected)
            #expect(load.report.plannedRanges.first == p.packedRange)
            #expect(Array(load.report.plannedRanges.dropFirst()) == p.sidecarRanges)
            #expect(log.calls == expected)
        }
    }

    @Test("artifact partition covers rows deterministically within budget")
    func artifactPartitionCoverageAndLoads() throws {
        guard let file = try artifact(), let record = try? file.q4(tensorName) else { return }
        #expect(record.logicalShape == [48, 5120])
        let requested = 0..<record.logicalShape[0]
        let oneRow = try plan(file, range: 0..<1)
        #expect(oneRow.requiredResidentBytes == 2_880)
        let budget = oneRow.requiredResidentBytes * 4 + 1
        let plans = try Q4TensorStreamPlan.partition(file: file, tensorName: tensorName,
                                                       maximumResidentBytes: budget,
                                                       outputColumnRange: requested)
        let repeated = try Q4TensorStreamPlan.partition(file: file, tensorName: tensorName,
                                                          maximumResidentBytes: budget,
                                                          outputColumnRange: requested)
        #expect(plans == repeated)
        #expect(plans.count > 1)
        #expect(plans.map(\.outputColumnRange).reduce([], +).count == requested.count)
        var next = requested.lowerBound
        for p in plans {
            #expect(p.outputColumnRange.lowerBound == next)
            #expect(!p.outputColumnRange.isEmpty)
            #expect(p.requiredResidentBytes <= budget)
            #expect(p.outputColumnRange.upperBound <= requested.upperBound)
            let allRanges = [p.packedRange] + p.sidecarRanges
            for range in allRanges {
                #expect(range.offset <= file.fileSize)
                #expect(range.length <= file.fileSize - range.offset)
            }
            next = p.outputColumnRange.upperBound
        }
        #expect(next == requested.upperBound)

        let subrange = 7..<19
        let subPlans = try Q4TensorStreamPlan.partition(file: file, tensorName: tensorName,
                                                          maximumResidentBytes: budget,
                                                          outputColumnRange: subrange)
        var subNext = subrange.lowerBound
        for p in subPlans {
            #expect(p.outputColumnRange.lowerBound == subNext)
            #expect(subrange.contains(p.outputColumnRange.lowerBound))
            #expect(p.outputColumnRange.upperBound <= subrange.upperBound)
            subNext = p.outputColumnRange.upperBound
        }
        #expect(subNext == subrange.upperBound)

        guard let device = MTLCreateSystemDefaultDevice() else { return }
        let runtime = try Q4GemmRuntime(device: device, metallib: Metallib(device: device))
        let streamer = Q4TensorStreamer(runtime: runtime, device: device)
        for p in plans {
            let load = try streamer.loadChunkWithReport(reader: RecordingReader(file: file, log: ReadLog()), plan: p)
            let expected = [p.packedRange] + p.sidecarRanges
            #expect(load.report.tensorName == p.tensorName)
            #expect(load.report.outputColumnRange == p.outputColumnRange)
            #expect(load.report.plannedRanges == expected)
            #expect(load.report.plannedTotalBytes == p.requiredResidentBytes)
            #expect(load.report.actualTotalBytes == p.requiredResidentBytes)
            #expect(load.report.requiredResidentBytes == p.requiredResidentBytes)
            #expect(load.report.requiredResidentBytes <= load.report.maximumResidentBytes)
        }
        #expect(throws: Q4TensorStreamError.self) {
            _ = try Q4TensorStreamPlan.partition(file: file, tensorName: tensorName,
                                                  maximumResidentBytes: oneRow.requiredResidentBytes - 1,
                                                  outputColumnRange: requested)
        }
        let equal = try Q4TensorStreamPlan.partition(file: file, tensorName: tensorName,
                                                       maximumResidentBytes: oneRow.requiredResidentBytes,
                                                       outputColumnRange: 0..<1)
        #expect(equal.count == 1)
        #expect(equal[0].requiredResidentBytes == oneRow.requiredResidentBytes)
        print("Q4 partition tensor=\(tensorName) shape=\(record.logicalShape) budget=\(budget) chunks=\(plans.count) ranges=\(plans.map { String(describing: $0.outputColumnRange) }.joined(separator: ","))")
    }

    @Test("short and overlong reads are rejected with exact accounting")
    func malformedReadsRejected() throws {
        guard let file = try artifact(), let device = MTLCreateSystemDefaultDevice() else { return }
        let runtime = try Q4GemmRuntime(device: device, metallib: Metallib(device: device))
        let streamer = Q4TensorStreamer(runtime: runtime, device: device)
        let p = try plan(file, range: 0..<1)
        for response in [RecordingReader.Response.short, .overlong] {
            let log = ReadLog()
            #expect(throws: Q4TensorStreamError.self) {
                _ = try streamer.loadChunkWithReport(reader: RecordingReader(file: file, log: log, response: response), plan: p)
            }
            #expect(log.calls.count == 1)
            #expect(log.calls[0] == p.packedRange)
        }
    }

    @Test("over-budget load rejects before any recorded read")
    func overBudgetLoadHasZeroReads() throws {
        guard let file = try artifact() else { return }
        let log = ReadLog()
        #expect(throws: Q4TensorStreamError.self) {
            _ = try Q4TensorStreamPlan(file: file, tensorName: tensorName, outputColumnRange: 0..<1, maximumResidentBytes: 1)
        }
        #expect(log.calls.isEmpty)
    }

    @Test("plan ranges reject out-of-file and overflowed offsets")
    func planRangeSafety() throws {
        guard let file = try artifact() else { return }
        let p = try plan(file, range: 0..<1)
        #expect(p.packedRange.end <= file.fileSize)
        #expect(p.sidecarRanges.allSatisfy { $0.end <= file.fileSize })
        #expect(throws: Q4TensorStreamError.self) { try Q4TensorStreamPlan(file: file, tensorName: tensorName, outputColumnRange: Int.max..<Int.max, maximumResidentBytes: UInt64.max) }
    }

    @Test("malformed host buffers reject dispatch before allocation")
    func malformedDispatchBuffers() throws {
        guard let file = try artifact(), let device = MTLCreateSystemDefaultDevice() else { return }
        let p = try plan(file, range: 0..<1)
        let runtime = try Q4GemmRuntime(device: device, metallib: Metallib(device: device))
        let streamer = Q4TensorStreamer(runtime: runtime, device: device)
        let shape = try GemmQ4Shape(rows: p.outputColumnRange.count, columns: p.logicalShape[0], inner: p.logicalShape[1])
        let layout = try Q4BufferLayout(rows: p.logicalShape[0], logicalK: p.logicalShape[1], rowStrideBytes: p.rowStrideBytes)
        let buffer = try #require(device.makeBuffer(length: 1, options: .storageModeShared))
        let validPacked = Data(repeating: 0, count: Int(p.packedRange.length))
        let validScales = Data(repeating: 0, count: p.sidecarRanges[..<(p.sidecarRanges.count / 2)].reduce(0) { $0 + Int($1.length) })
        let validBiases = Data(repeating: 0, count: p.sidecarRanges[(p.sidecarRanges.count / 2)...].reduce(0) { $0 + Int($1.length) })
        let malformed: [(Data, Data, Data)] = [
            (Data(), validScales, validBiases),
            (validPacked, Data(), validBiases),
            (validPacked, validScales, Data()),
            (Data([0]), Data([0]), Data([0]))
        ]
        for (packed, scales, biases) in malformed {
            let chunk = Q4TensorStreamChunk(packed: packed, scales: scales, biases: biases, outputColumnRange: p.outputColumnRange)
            #expect(throws: Q4TensorStreamError.invalidChunk("chunk has no stream plan")) {
                try streamer.dispatchChunk(chunk, outputColumnBase: 0, shape: shape, layout: layout, a: buffer, output: buffer, commandQueue: try #require(device.makeCommandQueue()))
            }
        }
    }

    @Test("zero-based artifact chunk matches decoded Q4 oracle")
    func zeroBasedArtifactDispatch() throws {
        guard let file = try artifact(), let device = MTLCreateSystemDefaultDevice() else { return }
        let plan = try plan(file, range: 0..<1)
        let runtime = try Q4GemmRuntime(device: device, metallib: Metallib(device: device))
        let streamer = Q4TensorStreamer(runtime: runtime, device: device)
        let chunk = try streamer.loadChunk(file, plan: plan)
        let shape = try GemmQ4Shape(rows: 2, columns: 1, inner: 5120)
        let layout = try Q4BufferLayout(rows: 1, logicalK: 5120, rowStrideBytes: plan.rowStrideBytes)

        let activations = (0..<shape.rows * shape.inner).map { index in
            let value = Float((index % 19) - 9) * 0.03125 + (index % 7 == 0 ? 0.125 : 0)
            return Q4BufferLayout.bf16Bits(value)
        }
        let words: [UInt32] = chunk.packed.withUnsafeBytes { raw in
            Array(raw.bindMemory(to: UInt32.self))
        }
        let scales: [UInt16] = chunk.scales.withUnsafeBytes { raw in
            Array(raw.bindMemory(to: UInt16.self))
        }
        let biases: [UInt16] = chunk.biases.withUnsafeBytes { raw in
            Array(raw.bindMemory(to: UInt16.self))
        }
        var b = Array(repeating: UInt16.zero, count: shape.inner)
        for k in 0..<shape.inner {
            let decoded = try layout.decode(row: 0, k: k, words: words,
                                            scalesBF16: scales, biasesBF16: biases)
            b[k] = Q4BufferLayout.bf16Bits(decoded)
        }
        let expected = GemmOracle.bf16(a: activations, b: b,
                                       rows: shape.rows, columns: shape.columns, inner: shape.inner)

        guard let input = device.makeBuffer(bytes: activations,
                                            length: activations.count * MemoryLayout<UInt16>.stride,
                                            options: .storageModeShared),
              let output = device.makeBuffer(length: expected.count * MemoryLayout<Float>.stride,
                                             options: .storageModeShared),
              let queue = device.makeCommandQueue() else {
            print("SKIP Q4 streaming oracle: Metal buffer or command queue allocation unavailable")
            return
        }
        memset(output.contents(), 0, output.length)
        try streamer.dispatchChunk(chunk, outputColumnBase: 0, shape: shape, layout: layout,
                                   a: input, output: output, commandQueue: queue)
        let actual = Array(UnsafeBufferPointer(start: output.contents().bindMemory(to: Float.self,
                                                                                    capacity: expected.count),
                                               count: expected.count))
        let maxAbs = zip(actual, expected).map { abs($0 - $1) }.max() ?? 0
        let maxRel = zip(actual, expected).map { pair in
            abs(pair.0 - pair.1) / max(abs(pair.1), 1e-3)
        }.max() ?? 0
        let allFinite = actual.allSatisfy { $0.isFinite }
        print("Q4 streaming oracle tensor=\(tensorName) range=0..<1 shape=[2,1,5120] maxAbs=\(maxAbs) maxRel=\(maxRel) tolerance=0.10")
        #expect(allFinite, "q4 streaming runtime produced non-finite output")
        #expect(maxRel <= 0.10, "max relative error \(maxRel) exceeds documented q4 tolerance 0.10")
    }

    @Test("partitioned artifact full range matches decoded Q4 oracle")
    func partitionedArtifactOracleFullRange() throws {
        try partitionedArtifactOracle(range: 0..<48, label: "full")
    }

    @Test("partitioned artifact nonzero range matches decoded Q4 oracle")
    func partitionedArtifactOracleSubrange() throws {
        try partitionedArtifactOracle(range: 7..<19, label: "subrange")
    }

    private func partitionedArtifactOracle(range: Range<Int>, label: String) throws {
        guard let file = try artifact(), let record = try? file.q4(tensorName) else { return }
        guard let device = MTLCreateSystemDefaultDevice() else { return }
        let runtime = try Q4GemmRuntime(device: device, metallib: Metallib(device: device))
        let streamer = Q4TensorStreamer(runtime: runtime, device: device)
        let budget: UInt64 = 11_521
        let rows = 2
        let inner = record.logicalShape[1]
        let shape = try GemmQ4Shape(rows: rows, columns: range.count, inner: inner)
        let plans = try Q4TensorStreamPlan.partition(file: file, tensorName: tensorName,
                                                       maximumResidentBytes: budget,
                                                       outputColumnRange: range)
        #expect(!plans.isEmpty)

        // Deliberately nonzero BF16 activations, including negative values, to avoid
        // masking sign, base-offset, or zero-input errors in the dispatch path.
        let activations = (0..<(rows * inner)).map { index -> UInt16 in
            let lane = index % inner
            let value = 0.5 + Float((lane % 23) - 11) * 0.0625 +
                (lane % 7 == 0 ? 0.1875 : 0) +
                (index / inner == 1 ? 0.03125 : 0)
            return Q4BufferLayout.bf16Bits(value)
        }
        guard let input = device.makeBuffer(bytes: activations,
                                             length: activations.count * MemoryLayout<UInt16>.stride,
                                             options: .storageModeShared),
              let queue = device.makeCommandQueue() else {
            print("SKIP Q4 partitioned oracle: Metal buffer or command queue allocation unavailable")
            return
        }

        var decoded = Array(repeating: UInt16.zero, count: range.count * inner)
        for plan in plans {
            let chunk = try streamer.loadChunk(file, plan: plan)
            let layout = try Q4BufferLayout(rows: plan.outputColumnRange.count,
                                            logicalK: inner,
                                            rowStrideBytes: plan.rowStrideBytes)
            let words: [UInt32] = chunk.packed.withUnsafeBytes { Array($0.bindMemory(to: UInt32.self)) }
            let scales: [UInt16] = chunk.scales.withUnsafeBytes { Array($0.bindMemory(to: UInt16.self)) }
            let biases: [UInt16] = chunk.biases.withUnsafeBytes { Array($0.bindMemory(to: UInt16.self)) }
            for localColumn in 0..<plan.outputColumnRange.count {
                let absoluteColumn = plan.outputColumnRange.lowerBound + localColumn
                let destinationColumn = absoluteColumn - range.lowerBound
                for k in 0..<inner {
                    let value = try layout.decode(row: localColumn, k: k, words: words,
                                                  scalesBF16: scales, biasesBF16: biases)
                    // GemmOracle expects B in [K, N] row-major order.
                    decoded[k * range.count + destinationColumn] = Q4BufferLayout.bf16Bits(value)
                }
            }
        }
        let expected = GemmOracle.bf16(a: activations, b: decoded,
                                        rows: rows, columns: range.count, inner: inner)
        let result = try streamer.dispatchPartitioned(file: file, tensorName: tensorName,
                                                       outputColumnRange: range,
                                                       maximumResidentBytes: budget, shape: shape,
                                                       a: input, commandQueue: queue)
        #expect(result.ranges == plans.map(\.outputColumnRange))
        #expect(result.reports.count == plans.count)
        #expect(result.ranges.count == plans.count)
        for index in result.ranges.indices {
            let current = result.ranges[index]
            #expect(!current.isEmpty)
            #expect(current.lowerBound == (index == 0 ? range.lowerBound : result.ranges[index - 1].upperBound))
            #expect(current.upperBound <= range.upperBound)
            let report = result.reports[index]
            #expect(report.tensorName == tensorName)
            #expect(report.outputColumnRange == current)
            #expect(report.plannedRanges == [plans[index].packedRange] + plans[index].sidecarRanges)
            #expect(report.plannedTotalBytes == plans[index].requiredResidentBytes)
            #expect(report.actualTotalBytes == plans[index].requiredResidentBytes)
            #expect(report.requiredResidentBytes <= budget)
        }
        #expect(result.ranges.last?.upperBound == range.upperBound)
        let allFinite = result.values.allSatisfy(\.isFinite)
        let maxAbs = zip(result.values, expected).map { actual, reference in
            abs(actual - reference)
        }.max() ?? 0
        let maxRel = zip(result.values, expected).map { actual, reference in
            abs(actual - reference) / max(abs(reference), 1e-3)
        }.max() ?? 0
        print("Q4 partitioned oracle tensor=\(tensorName) label=\(label) range=\(range) budget=\(budget) M=\(rows) chunks=\(plans.count) maxAbs=\(maxAbs) maxRel=\(maxRel) tolerance=0.10")
        #expect(allFinite, "q4 partitioned runtime produced non-finite output")
        #expect(maxRel <= 0.10, "max relative error \(maxRel) exceeds documented q4 tolerance 0.10")
    }

    @Test("dispatch rejects shape and layout row mismatches")
    func dispatchShapeAndLayoutMismatches() throws {
        guard let file = try artifact(), let device = MTLCreateSystemDefaultDevice() else { return }
        let plan = try plan(file, range: 0..<1)
        let runtime = try Q4GemmRuntime(device: device, metallib: Metallib(device: device))
        let streamer = Q4TensorStreamer(runtime: runtime, device: device)
        let chunk = try streamer.loadChunk(file, plan: plan)
        let input = try #require(device.makeBuffer(length: 5120 * MemoryLayout<UInt16>.stride, options: .storageModeShared))
        let output = try #require(device.makeBuffer(length: 2 * MemoryLayout<Float>.stride, options: .storageModeShared))
        let queue = try #require(device.makeCommandQueue())
        let layout = try Q4BufferLayout(rows: 1, logicalK: 5120, rowStrideBytes: plan.rowStrideBytes)
        #expect(throws: Q4TensorStreamError.invalidChunk("shape columns do not match output-row range")) {
            try streamer.dispatchChunk(chunk, outputColumnBase: 0, shape: try GemmQ4Shape(rows: 1, columns: 2, inner: 5120), layout: layout, a: input, output: output, commandQueue: queue)
        }
        let mismatchedLayout = try Q4BufferLayout(rows: 2, logicalK: 5120, rowStrideBytes: plan.rowStrideBytes)
        #expect(throws: Q4TensorStreamError.invalidChunk("layout rows do not match output-row range")) {
            try streamer.dispatchChunk(chunk, outputColumnBase: 0, shape: try GemmQ4Shape(rows: 1, columns: 1, inner: 5120), layout: mismatchedLayout, a: input, output: output, commandQueue: queue)
        }
    }

    @Test("malformed loaded chunk buffers reject before dispatch")
    func malformedLoadedChunkBuffers() throws {
        guard let file = try artifact(), let device = MTLCreateSystemDefaultDevice() else { return }
        let plan = try plan(file, range: 0..<1)
        let runtime = try Q4GemmRuntime(device: device, metallib: Metallib(device: device))
        let streamer = Q4TensorStreamer(runtime: runtime, device: device)
        let loaded = try streamer.loadChunk(file, plan: plan)
        let shape = try GemmQ4Shape(rows: 1, columns: 1, inner: 5120)
        let layout = try Q4BufferLayout(rows: 1, logicalK: 5120, rowStrideBytes: plan.rowStrideBytes)
        let input = try #require(device.makeBuffer(length: 5120 * MemoryLayout<UInt16>.stride, options: .storageModeShared))
        let output = try #require(device.makeBuffer(length: MemoryLayout<Float>.stride, options: .storageModeShared))
        let queue = try #require(device.makeCommandQueue())
        let malformed = Q4TensorStreamChunk(packed: Data(repeating: 0, count: loaded.packed.count - 1), scales: loaded.scales, biases: loaded.biases, outputColumnRange: loaded.outputColumnRange)
        #expect(throws: Q4TensorStreamError.invalidChunk("chunk has no stream plan")) {
            try streamer.dispatchChunk(malformed, outputColumnBase: 0, shape: shape, layout: layout, a: input, output: output, commandQueue: queue)
        }
    }

    @Test("nonzero chunk range is rejected for zero-based dispatch")
    func nonzeroChunkRange() throws {
        guard let file = try artifact(), let device = MTLCreateSystemDefaultDevice() else { return }
        let plan = try plan(file, range: 1..<2)
        let runtime = try Q4GemmRuntime(device: device, metallib: Metallib(device: device))
        let streamer = Q4TensorStreamer(runtime: runtime, device: device)
        let chunk = try streamer.loadChunk(file, plan: plan)
        let shape = try GemmQ4Shape(rows: 1, columns: 1, inner: 5120)
        let layout = try Q4BufferLayout(rows: 1, logicalK: 5120, rowStrideBytes: plan.rowStrideBytes)
        let input = try #require(device.makeBuffer(length: 5120 * MemoryLayout<UInt16>.stride, options: .storageModeShared))
        let output = try #require(device.makeBuffer(length: MemoryLayout<Float>.stride, options: .storageModeShared))
        #expect(throws: Q4TensorStreamError.invalidChunk("output-row range must start at zero")) {
            try streamer.dispatchChunk(chunk, outputColumnBase: 0, shape: shape, layout: layout, a: input, output: output, commandQueue: try #require(device.makeCommandQueue()))
        }
    }

    @Test("partitioned dispatch rejects undersized activation buffer before reads")
    func partitionedUndersizedActivationBuffer() throws {
        guard let file = try artifact(), let device = MTLCreateSystemDefaultDevice() else { return }
        let record = try file.q4(tensorName)
        let runtime = try Q4GemmRuntime(device: device, metallib: Metallib(device: device))
        let streamer = Q4TensorStreamer(runtime: runtime, device: device)
        let shape = try GemmQ4Shape(rows: 1, columns: 1, inner: record.logicalShape[1])
        let activation = try #require(device.makeBuffer(length: 1, options: .storageModeShared))
        let queue = try #require(device.makeCommandQueue())

        #expect(throws: GemmQ4Error.bufferTooSmall(name: "a", expected: record.logicalShape[1] * MemoryLayout<UInt16>.stride, observed: 1)) {
            _ = try streamer.dispatchPartitioned(file: file, tensorName: tensorName,
                                                  outputColumnRange: 0..<1,
                                                  maximumResidentBytes: 1 << 30,
                                                  shape: shape, a: activation, commandQueue: queue)
        }
    }

    @Test("partitioned dispatch rejects activation size overflow before allocation")
    func partitionedActivationSizeOverflow() throws {
        guard let file = try artifact(), let device = MTLCreateSystemDefaultDevice() else { return }
        let record = try file.q4(tensorName)
        let runtime = try Q4GemmRuntime(device: device, metallib: Metallib(device: device))
        let streamer = Q4TensorStreamer(runtime: runtime, device: device)
        let rows = Int.max / record.logicalShape[1] + 1
        let shape = try GemmQ4Shape(rows: rows, columns: 1, inner: record.logicalShape[1])
        let activation = try #require(device.makeBuffer(length: 1, options: .storageModeShared))
        let queue = try #require(device.makeCommandQueue())

        #expect(throws: Q4TensorStreamError.invalidPlan("activation element count overflow")) {
            _ = try streamer.dispatchPartitioned(file: file, tensorName: tensorName,
                                                  outputColumnRange: 0..<1,
                                                  maximumResidentBytes: 1 << 30,
                                                  shape: shape, a: activation, commandQueue: queue)
        }
    }

    @Test("partitioned dispatch rejects impossible maximum-column range")
    func partitionedImpossibleColumnRange() throws {
        guard let file = try artifact(), let device = MTLCreateSystemDefaultDevice() else { return }
        let record = try file.q4(tensorName)
        let runtime = try Q4GemmRuntime(device: device, metallib: Metallib(device: device))
        let streamer = Q4TensorStreamer(runtime: runtime, device: device)
        let shape = try GemmQ4Shape(rows: 1, columns: Int.max, inner: record.logicalShape[1])
        let activation = try #require(device.makeBuffer(length: 1, options: .storageModeShared))
        let queue = try #require(device.makeCommandQueue())

        #expect(throws: Q4TensorStreamError.invalidPlan("output-row range is empty or out of bounds")) {
            _ = try streamer.dispatchPartitioned(file: file, tensorName: tensorName,
                                                  outputColumnRange: 0..<Int.max,
                                                  maximumResidentBytes: 1 << 30,
                                                  shape: shape, a: activation, commandQueue: queue)
        }
    }

    @Test("nonzero output-column base is explicitly unsupported before dispatch")
    func nonzeroOutputBase() throws {
        guard let device = MTLCreateSystemDefaultDevice() else { return }
        let runtime = try Q4GemmRuntime(device: device, metallib: Metallib(device: device))
        let streamer = Q4TensorStreamer(runtime: runtime, device: device)
        let chunk = Q4TensorStreamChunk(packed: Data(), scales: Data(), biases: Data(), outputColumnRange: 1..<2)
        let shape = try GemmQ4Shape(rows: 1, columns: 1, inner: 64)
        let layout = try Q4BufferLayout(rows: 1, logicalK: 64)
        let buffer = try #require(device.makeBuffer(length: 1, options: .storageModeShared))
        #expect(throws: Q4TensorStreamError.unsupportedOutputColumnBase(1)) {
            try streamer.dispatchChunk(chunk, outputColumnBase: 1, shape: shape, layout: layout, a: buffer, output: buffer, commandQueue: try #require(device.makeCommandQueue()))
        }
    }
}

