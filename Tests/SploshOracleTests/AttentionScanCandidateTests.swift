import Accelerate
import Foundation
import Metal
import SploshCore
import Testing

/// Candidate int8 attention scan kernels against the shipped one and a CPU reference.
///
/// Every kernel named in `SPLOSH_SCAN_CANDIDATES` (comma-separated shapes, e.g. `b48c64s8`; the
/// kernel is `sp_attn_scan_i8_<shape>`) runs on the same synthetic keys, values and queries as
/// `sp_attn_scan_i8_m48c64s8`. Its span records are merged here in double precision the way
/// `sp_attn_merge` does (log-sum-exp across spans), and the merged vector of every (row, head)
/// is compared with the shipped kernel's and with a double-precision reference computed from
/// the same int8 codes and fp32 scales.
///
/// The error of a (row, head) is max |a - b| over the 256 dimensions divided by max |reference|
/// over them. Both kernels hand the value product bf16 operands (8 significant bits), so each
/// weighted probability is off by up to 2^-8 of itself; measured, that is 0.003 to 0.004 for
/// every kernel against the reference, whether a row's weight sits on one token or on many.
///
/// The shipped kernel clamps a score at 60 above its row's reference, so with keys that raise
/// a score by hundreds its normaliser is not the true one, and with two such keys in a span
/// its vector can be wrong as well. Those scenes require the candidates to match the reference
/// and only report the shipped kernel.
///
/// Beyond the merged result, every record is checked on its own: no element may be NaN, and a
/// record with reference -inf (a row that sees no token of the span) must have total 0 and a
/// zero vector. The scenes whose blocks straddle the first page of a span name the records
/// that must be empty, since the merge would give such a record no weight whatever it held.
///
/// `SPLOSH_SCAN_TIMING=1` adds an interleaved GPU timing of the kernels on a wider step.
@Suite("AttentionScanCandidateTests", .serialized)
struct AttentionScanCandidateTests {
    static let headDim = 256, perKV = 6, record = 258, blockRows = 8
    static let existing = "m48c64s8"
    /// Against the CPU reference: one bf16 rounding of a dominant probability, with margin.
    static let toleranceReference = 1.0 / 128.0
    /// Between two kernels, each within that of the reference.
    static let toleranceKernels = 1.0 / 64.0
    /// log of the merged normaliser, absolute.
    static let toleranceLogSum = 5e-4

    /// Kernels whose tile holds more than eight rows (the first number is queries, six a row).
    static func isWide(_ shape: String) -> Bool {
        (shape.split(whereSeparator: { !$0.isNumber }).compactMap { Int($0) }.first ?? 48) / perKV > blockRows
    }

    static var candidates: [String] {
        let raw = ProcessInfo.processInfo.environment["SPLOSH_SCAN_CANDIDATES"] ?? "b48c64s8,b48c32s8,h48c64s8,h48c32s8"
        return raw.split(separator: ",").map { $0.trimmingCharacters(in: .whitespaces) }.filter { !$0.isEmpty }
    }

    static func loadLibrary(_ device: MTLDevice) throws -> Metallib {
        if let raw = ProcessInfo.processInfo.environment["SPLOSH_TEST_METALLIB"], !raw.isEmpty {
            return try makeTestMetallib(device: device)
        }
        return try Metallib(device: device)
    }

    struct Generator {
        var state: UInt64
        mutating func next() -> UInt64 {
            state &+= 0x9E37_79B9_7F4A_7C15
            var z = state
            z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
            z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
            return z ^ (z >> 31)
        }
        /// Uniform in [0, 1).
        mutating func unit() -> Float { Float(next() >> 40) / Float(1 << 24) }
    }

    /// Mirrors SpaScanParams.
    struct ScanParams {
        var blocks, heads, kvHeads, maxPages, spans, pagesPerSpan, rowCap: UInt32
        var scale: Float
        var aliasTokens: UInt32 = 0
        var blocksAcross: UInt32 = 0
    }

    /// A key whose scale is multiplied by `factor`, at `token` of the long slot.
    struct LargeKey { var token: Int; var factor: Float }

    /// Two slots with their own page tables over one shuffled pool. Rows are the last tokens of
    /// each slot's context, in blocks of eight with a short last block; the long slot's rows
    /// come first so that neither r0 nor the slot index is zero for every block.
    final class Fixture {
        let device: MTLDevice
        let contexts: [Int]            // per slot
        let slotRows: [Int]            // per slot
        let rowSlot: [Int], rowPos: [Int]
        let blockList: [(first: Int, count: Int)]
        /// The same rows in blocks of ten, for kernels whose tile holds ten rows (r60...).
        let wideBlockList: [(first: Int, count: Int)]
        let wideBlocks: MTLBuffer
        let rows: Int, rowCap: Int, maxPages: Int, stepPages: Int
        let kvHeads: Int, heads: Int
        /// Every pool token has finite scales (for aliasTokens, which reads tokens no slot stored).
        let allScales: Bool
        let scale = 1 / Float(256).squareRoot()
        let queries, blocks, rowSlotBuffer, rowPosBuffer, pageTable, kCodes, kScale, vCodes, vScale: MTLBuffer

        /// `nanUnusedQueries`: the query rows after the step's last row are NaN, not random. The
        /// last block of the step is short and reads them as its queries beyond the count.
        init(device: MTLDevice, context: Int, large: [LargeKey], seed: UInt64, slotRows: [Int] = [11, 13],
             kvHeads: Int = 2, allScales: Bool = false, nanUnusedQueries: Bool = false) throws {
            self.device = device
            self.slotRows = slotRows
            self.kvHeads = kvHeads
            self.allScales = allScales
            heads = kvHeads * AttentionScanCandidateTests.perKV
            contexts = [context * 2 / 5 + 17, context]
            var rowSlot = [Int](), rowPos = [Int](), blockList = [(first: Int, count: Int)]()
            for slot in [1, 0] {
                let first = contexts[slot] - slotRows[slot]
                var done = 0
                while done < slotRows[slot] {
                    let count = min(AttentionScanCandidateTests.blockRows, slotRows[slot] - done)
                    blockList.append((rowSlot.count, count))
                    for i in 0..<count { rowSlot.append(slot); rowPos.append(first + done + i) }
                    done += count
                }
            }
            self.rowSlot = rowSlot; self.rowPos = rowPos; self.blockList = blockList
            var wide = [(first: Int, count: Int)]()
            var start = 0
            for slot in [1, 0] {
                var done = 0
                while done < slotRows[slot] {
                    let count = min(10, slotRows[slot] - done)
                    wide.append((start + done, count))
                    done += count
                }
                start += slotRows[slot]
            }
            wideBlockList = wide
            rows = rowSlot.count
            rowCap = rows + 8
            let slotPages = contexts.map { ($0 + 255) / 256 }
            maxPages = slotPages.max()! + 3
            stepPages = rowPos.max()! / 256 + 1      // as Engine.swift computes it

            func buffer(_ bytes: Int) throws -> MTLBuffer {
                try #require(device.makeBuffer(length: max(bytes, 16), options: .storageModeShared))
            }
            let kv = kvHeads, dim = AttentionScanCandidateTests.headDim
            var generator = Generator(state: seed)

            // The page table: the slots' pages are spread over the pool in a shuffled order,
            // with a few pool pages left unused.
            let poolPages = slotPages.reduce(0, +) + 5
            var order = Array(0..<poolPages)
            for i in stride(from: poolPages - 1, to: 0, by: -1) { order.swapAt(i, Int(generator.next() % UInt64(i + 1))) }
            pageTable = try buffer(2 * maxPages * 4)
            let table = pageTable.contents().bindMemory(to: UInt32.self, capacity: 2 * maxPages)
            for i in 0..<(2 * maxPages) { table[i] = 0 }
            var used = 0
            for slot in 0..<2 { for page in 0..<slotPages[slot] { table[slot * maxPages + page] = UInt32(order[used]); used += 1 } }

            // Codes: every byte of the pool random, tokens beyond a context included.
            let poolTokens = poolPages * kv * 256
            kCodes = try buffer(poolTokens * dim); vCodes = try buffer(poolTokens * dim)
            for pool in [kCodes, vCodes] {
                let words = pool.contents().bindMemory(to: UInt64.self, capacity: poolTokens * dim / 8)
                for i in 0..<(poolTokens * dim / 8) { words[i] = generator.next() }
            }
            // Scales: NaN wherever no token was stored, so a kernel that lets a masked token
            // through shows it.
            kScale = try buffer(poolTokens * 4); vScale = try buffer(poolTokens * 4)
            let ks = kScale.contents().bindMemory(to: Float.self, capacity: poolTokens)
            let vs = vScale.contents().bindMemory(to: Float.self, capacity: poolTokens)
            for i in 0..<poolTokens {
                ks[i] = allScales ? 0.04 * (0.5 + generator.unit()) : .nan
                vs[i] = allScales ? 0.01 * (0.5 + generator.unit()) : .nan
            }
            for slot in 0..<2 { for head in 0..<kv { for token in 0..<contexts[slot] {
                let index = (Int(table[slot * maxPages + token / 256]) * kv + head) * 256 + token % 256
                // A code dot a unit-variance query is about 1,200; times 1/16 and this, about 3.
                ks[index] = 0.04 * (0.5 + generator.unit())
                vs[index] = 0.01 * (0.5 + generator.unit())
            } } }
            for key in large { for head in 0..<kv {
                ks[(Int(table[1 * maxPages + key.token / 256]) * kv + head) * 256 + key.token % 256] *= key.factor
            } }

            // Queries, fp16: [kvHead][rowCap][perKV][256], the unused rows random too.
            let queryCount = kv * rowCap * AttentionScanCandidateTests.perKV * dim
            queries = try buffer(queryCount * 2)
            let q = queries.contents().bindMemory(to: Float16.self, capacity: queryCount)
            for i in 0..<queryCount { q[i] = Float16((generator.unit() - 0.5) * 3.4641) }
            if nanUnusedQueries {
                let perRow = AttentionScanCandidateTests.perKV * dim
                for head in 0..<kv { for i in ((head * rowCap + rows) * perRow)..<((head + 1) * rowCap * perRow) { q[i] = .nan } }
            }

            blocks = try buffer(blockList.count * 8)
            let b = blocks.contents().bindMemory(to: UInt32.self, capacity: blockList.count * 2)
            for (i, block) in blockList.enumerated() { b[i * 2] = UInt32(block.first); b[i * 2 + 1] = UInt32(block.count) }
            wideBlocks = try buffer(wide.count * 8)
            let w = wideBlocks.contents().bindMemory(to: UInt32.self, capacity: wide.count * 2)
            for (i, block) in wide.enumerated() { w[i * 2] = UInt32(block.first); w[i * 2 + 1] = UInt32(block.count) }
            rowSlotBuffer = try buffer(rows * 4); rowPosBuffer = try buffer(rows * 4)
            let rs = rowSlotBuffer.contents().bindMemory(to: UInt32.self, capacity: rows)
            let rp = rowPosBuffer.contents().bindMemory(to: UInt32.self, capacity: rows)
            for i in 0..<rows { rs[i] = UInt32(rowSlot[i]); rp[i] = UInt32(rowPos[i]) }
        }

        /// Index of a slot's token in the scale arrays (and, times 256, in the code pools).
        /// With `alias` the kernels read each `chunk`-token chunk from the pool's first tokens
        /// instead (SpaScanParams.aliasTokens), which depends on the kernel's chunk size.
        func tokenIndex(slot: Int, head: Int, token: Int, alias: Int = 0, chunk: Int = 64) -> Int {
            if alias != 0 {
                let t0 = token / chunk * chunk
                return head * 256 + (t0 % alias) * kvHeads + token - t0
            }
            let table = pageTable.contents().bindMemory(to: UInt32.self, capacity: 2 * maxPages)
            return (Int(table[slot * maxPages + token / 256]) * kvHeads + head) * 256 + token % 256
        }

        /// The (row, span) pairs whose span runs chunks for the row's block while the row itself
        /// sees none of its tokens: the span starts after the row's position and not after the
        /// block's last. Their records must be reference -inf, total 0 and a zero vector.
        func emptyRunRecords(spanLimit: Int, wide: Bool = false) -> [(row: Int, span: Int)] {
            let (spans, pagesPerSpan) = self.spans(limit: spanLimit)
            var found = [(row: Int, span: Int)]()
            for block in wide ? wideBlockList : blockList {
                let last = rowPos[block.first + block.count - 1]
                for row in block.first..<(block.first + block.count) {
                    for span in 0..<spans where span * pagesPerSpan * 256 > rowPos[row] && span * pagesPerSpan * 256 <= last {
                        found.append((row, span))
                    }
                }
            }
            return found
        }

        /// Problems of single records: a NaN anywhere, a record with reference -inf that is not
        /// empty, or one of `empty` that is not (-inf, 0, zero vector).
        func recordProblems(_ partials: [Float], spanLimit: Int, empty: [(row: Int, span: Int)]) -> [String] {
            let record = AttentionScanCandidateTests.record
            let (spans, _) = self.spans(limit: spanLimit)
            var problems = [String]()
            for entry in 0..<(rows * heads) { for s in 0..<spans {
                let base = (entry * spans + s) * record
                var nans = 0, nonZero = 0
                for i in 0..<record {
                    if partials[base + i].isNaN { nans += 1 }
                    if i > 0 && partials[base + i] != 0 { nonZero += 1 }
                }
                if nans > 0 { problems.append("entry \(entry) span \(s): \(nans) NaN elements") }
                if partials[base] == -.infinity && nonZero > 0 {
                    problems.append("entry \(entry) span \(s): reference -inf with \(nonZero) non-zero elements")
                }
            } }
            for (row, span) in empty { for head in 0..<heads {
                let base = ((row * heads + head) * spans + span) * record
                if partials[base] != -.infinity || partials[base + 1] != 0 || (2..<record).contains(where: { partials[base + $0] != 0 }) {
                    problems.append("row \(row) head \(head) span \(span): not an empty record (reference \(partials[base]), total \(partials[base + 1]))")
                }
            } }
            return Array(problems.prefix(4))
        }

        /// Spans as Engine.swift chooses them for a span limit.
        func spans(limit: Int) -> (spans: Int, pagesPerSpan: Int) {
            let pagesPerSpan = (stepPages + limit - 1) / limit
            return ((stepPages + pagesPerSpan - 1) / pagesPerSpan, pagesPerSpan)
        }

        /// Runs one scan kernel and returns its partial records and the GPU time.
        func run(_ library: Metallib, shape: String, spanLimit: Int, across: Bool, alias: Int = 0) throws -> (partials: [Float], seconds: Double) {
            let pipeline = try library.pipeline("sp_attn_scan_i8_" + shape)
            // Threads per threadgroup from the shape's simdgroup counts, as Engine.setScanShape.
            let digits = shape.split(whereSeparator: { !$0.isNumber }).compactMap { Int($0) }
            let threads = max(digits.count > 2 ? digits[2] : 8, shape.contains("v") && digits.count > 3 ? digits[3] : 0) * 32
            let wide = AttentionScanCandidateTests.isWide(shape)
            let blockList = wide ? wideBlockList : self.blockList, blocks = wide ? wideBlocks : self.blocks
            let (spans, pagesPerSpan) = self.spans(limit: spanLimit)
            let floats = rows * heads * spans * AttentionScanCandidateTests.record
            let partials = try #require(device.makeBuffer(length: floats * 4, options: .storageModeShared))
            let out = partials.contents().bindMemory(to: Float.self, capacity: floats)
            for i in 0..<floats { out[i] = AttentionScanCandidateTests.unwritten }
            var params = ScanParams(blocks: UInt32(blockList.count), heads: UInt32(heads),
                                    kvHeads: UInt32(kvHeads), maxPages: UInt32(maxPages),
                                    spans: UInt32(spans), pagesPerSpan: UInt32(pagesPerSpan), rowCap: UInt32(rowCap),
                                    scale: scale, aliasTokens: UInt32(alias), blocksAcross: across ? 1 : 0)
            let queue = try #require(device.makeCommandQueue())
            let command = try #require(queue.makeCommandBuffer())
            let encoder = try #require(command.makeComputeCommandEncoder())
            encoder.setComputePipelineState(pipeline)
            encoder.setBuffer(queries, offset: 0, index: 0)
            encoder.setBuffer(blocks, offset: 0, index: 2)
            encoder.setBuffer(rowSlotBuffer, offset: 0, index: 3)
            encoder.setBuffer(rowPosBuffer, offset: 0, index: 4)
            encoder.setBuffer(pageTable, offset: 0, index: 5)
            encoder.setBuffer(kCodes, offset: 0, index: 6)
            encoder.setBuffer(kScale, offset: 0, index: 7)
            encoder.setBuffer(vCodes, offset: 0, index: 8)
            encoder.setBuffer(vScale, offset: 0, index: 9)
            encoder.setBuffer(partials, offset: 0, index: 10)
            encoder.setBytes(&params, length: MemoryLayout<ScanParams>.stride, index: 11)
            let kv = kvHeads
            let grid = across ? MTLSize(width: kv * blockList.count, height: spans, depth: 1)
                              : MTLSize(width: kv * spans, height: blockList.count, depth: 1)
            encoder.dispatchThreadgroups(grid, threadsPerThreadgroup: MTLSize(width: threads, height: 1, depth: 1))
            encoder.endEncoding()
            command.commit()
            command.waitUntilCompleted()
            if let error = command.error { throw error }
            return (Array(UnsafeBufferPointer(start: out, count: floats)), command.gpuEndTime - command.gpuStartTime)
        }

        /// Softmax-weighted values and log normaliser per (row, head), in double precision,
        /// from the int8 codes and fp32 scales the kernels read.
        func reference(alias: Int = 0, chunk: Int = 64) -> Merged {
            let kv = kvHeads, perKV = AttentionScanCandidateTests.perKV
            let dim = AttentionScanCandidateTests.headDim
            var vectors = [Double](repeating: 0, count: rows * heads * dim)
            var logSums = [Double](repeating: 0, count: rows * heads)
            let q = queries.contents().bindMemory(to: Float16.self, capacity: kv * rowCap * perKV * dim)
            let kCode = kCodes.contents().bindMemory(to: Int8.self, capacity: 1)
            let vCode = vCodes.contents().bindMemory(to: Int8.self, capacity: 1)
            let ks = kScale.contents().bindMemory(to: Float.self, capacity: 1)
            let vs = vScale.contents().bindMemory(to: Float.self, capacity: 1)
            var page = [Double](repeating: 0, count: 256 * dim), transposed = [Double](repeating: 0, count: 256 * dim)
            var codes = [Int8](repeating: 0, count: 256 * dim)
            // A page's 256 tokens as indices into the scale arrays, and their codes gathered.
            func indices(_ slot: Int, _ head: Int, _ pg: Int) -> [Int] {
                (0..<256).map { tokenIndex(slot: slot, head: head, token: pg * 256 + $0, alias: alias, chunk: chunk) }
            }
            func gather(_ pool: UnsafePointer<Int8>, _ index: [Int]) {
                codes.withUnsafeMutableBufferPointer { c in
                    for t in 0..<256 { memcpy(c.baseAddress! + t * dim, pool + index[t] * dim, dim) }
                }
                vDSP_vflt8D(codes, 1, &page, 1, vDSP_Length(256 * dim))
            }
            for slot in 0..<2 {
                let slotFirstRow = rowSlot.firstIndex(of: slot)!
                let count = slotRows[slot], fused = count * perKV
                let pages = (contexts[slot] + 255) / 256
                for head in 0..<kv {
                    var query = [Double](repeating: 0, count: fused * dim)
                    for i in 0..<fused { for d in 0..<dim {
                        query[i * dim + d] = Double(q[((head * rowCap + slotFirstRow) * perKV + i) * dim + d])
                    } }
                    // Scores, then weights, as [fused query][page * 256 + token].
                    var weights = [Double](repeating: 0, count: fused * pages * 256)
                    var product = [Double](repeating: 0, count: fused * 256)
                    let index = (0..<pages).map { indices(slot, head, $0) }
                    for pg in 0..<pages {
                        gather(kCode, index[pg])
                        vDSP_mtransD(page, 1, &transposed, 1, vDSP_Length(dim), 256)
                        vDSP_mmulD(query, 1, transposed, 1, &product, 1, vDSP_Length(fused), 256, vDSP_Length(dim))
                        for i in 0..<fused { for t in 0..<256 { weights[i * pages * 256 + pg * 256 + t] = product[i * 256 + t] } }
                    }
                    var tops = [Double](repeating: -.infinity, count: fused), sums = [Double](repeating: 0, count: fused)
                    weights.withUnsafeMutableBufferPointer { w in
                        for i in 0..<fused {
                            // A row sees tokens up to and including its own position.
                            let limit = rowPos[slotFirstRow + i / perKV] + 1
                            let row = w.baseAddress! + i * pages * 256
                            var top = -Double.infinity
                            for pg in 0..<pages {
                                let end = min(256, limit - pg * 256)
                                if end <= 0 { break }
                                for t in 0..<end {
                                    let s = row[pg * 256 + t] * Double(ks[index[pg][t]]) * Double(scale)
                                    row[pg * 256 + t] = s
                                    if s > top { top = s }
                                }
                            }
                            var sum = 0.0
                            for pg in 0..<pages {
                                let end = max(0, min(256, limit - pg * 256))
                                for t in 0..<end {
                                    let probability = exp(row[pg * 256 + t] - top)
                                    sum += probability
                                    row[pg * 256 + t] = probability * Double(vs[index[pg][t]])
                                }
                                for t in end..<256 { row[pg * 256 + t] = 0 }
                            }
                            tops[i] = top; sums[i] = sum
                        }
                    }
                    var accumulated = [Double](repeating: 0, count: fused * dim)
                    var pageWeights = [Double](repeating: 0, count: fused * 256)
                    var contribution = [Double](repeating: 0, count: fused * dim)
                    for pg in 0..<pages {
                        gather(vCode, index[pg])
                        for i in 0..<fused { for t in 0..<256 { pageWeights[i * 256 + t] = weights[i * pages * 256 + pg * 256 + t] } }
                        vDSP_mmulD(pageWeights, 1, page, 1, &contribution, 1, vDSP_Length(fused), vDSP_Length(dim), 256)
                        vDSP_vaddD(accumulated, 1, contribution, 1, &accumulated, 1, vDSP_Length(fused * dim))
                    }
                    for i in 0..<fused {
                        let entry = (slotFirstRow + i / perKV) * heads + head * perKV + i % perKV
                        for d in 0..<dim { vectors[entry * dim + d] = accumulated[i * dim + d] / sums[i] }
                        logSums[entry] = tops[i] + log(sums[i])
                    }
                }
            }
            return Merged(vectors: vectors, logSums: logSums, problems: [])
        }

        /// What sp_attn_merge computes from the span records (before the output gate), in double.
        func merge(_ partials: [Float], spanLimit: Int) -> Merged {
            let dim = AttentionScanCandidateTests.headDim
            let record = AttentionScanCandidateTests.record
            let (spans, _) = self.spans(limit: spanLimit)
            var vectors = [Double](repeating: 0, count: rows * heads * dim)
            var logSums = [Double](repeating: 0, count: rows * heads)
            var problems = [String]()
            for entry in 0..<(rows * heads) {
                let base = entry * spans * record
                var top = -Double.infinity
                for s in 0..<spans {
                    for i in 0..<record where partials[base + s * record + i] == AttentionScanCandidateTests.unwritten {
                        problems.append("entry \(entry) span \(s) element \(i) not written"); break
                    }
                    let reference = Double(partials[base + s * record])
                    if reference.isNaN { problems.append("entry \(entry) span \(s): reference is NaN") }
                    top = max(top, reference)
                }
                var sum = 0.0
                var acc = [Double](repeating: 0, count: dim)
                for s in 0..<spans {
                    let weight = exp(Double(partials[base + s * record]) - top)
                    if weight > 0 {
                        sum += Double(partials[base + s * record + 1]) * weight
                        for d in 0..<dim { acc[d] += Double(partials[base + s * record + 2 + d]) * weight }
                    }
                }
                for d in 0..<dim { vectors[entry * dim + d] = acc[d] / sum }
                logSums[entry] = top + log(sum)
                if !(sum > 0) || !sum.isFinite { problems.append("entry \(entry): merged sum \(sum)") }
            }
            return Merged(vectors: vectors, logSums: logSums, problems: Array(problems.prefix(4)))
        }
    }

    static let unwritten: Float = -7777.25

    struct Merged {
        var vectors: [Double]      // [row][head][256]
        var logSums: [Double]      // [row][head]
        var problems: [String]

        /// Worst over (row, head) of max |self - other| / max |other|, and of |log sum| difference.
        func distance(to other: Merged) -> (vector: Double, logSum: Double) {
            var worst = 0.0, worstLog = 0.0
            for entry in 0..<logSums.count {
                var difference = 0.0, magnitude = 0.0
                for d in 0..<256 {
                    let a = vectors[entry * 256 + d], b = other.vectors[entry * 256 + d]
                    if a.isNaN || b.isNaN { difference = .infinity }
                    difference = max(difference, abs(a - b))
                    magnitude = max(magnitude, abs(b))
                }
                worst = max(worst, difference / magnitude)
                let logDifference = abs(logSums[entry] - other.logSums[entry])
                worstLog = max(worstLog, logDifference.isNaN ? .infinity : logDifference)
            }
            return (worst, worstLog)
        }
    }

    /// Worst errors seen by a kernel over the scenes of one test.
    struct Worst { var reference = 0.0, existing = 0.0, logSum = 0.0 }

    /// What a scene requires of the shipped kernel (the candidates must always match the reference).
    enum Shipped {
        case exact        // vector and normaliser match the reference, and the candidates match it
        case vector       // the vector does; its normaliser is clamped
        case reported     // nothing: its error is printed
    }

    /// Runs the shipped kernel and the candidates on one scene for each span limit and checks them.
    /// `straddles`: the scene must have rows that see nothing of a span their block runs.
    /// `alias`: SpaScanParams.aliasTokens; the reference then depends on the kernel's chunk size,
    /// and a candidate is compared with the shipped kernel only when it has the same one.
    func check(_ name: String, context: Int, large: [LargeKey], spanLimits: [Int], across: Bool = false,
               shipped gate: Shipped = .exact, kvHeads: Int = 2, alias: Int = 0, straddles: Bool = false,
               nanUnusedQueries: Bool = false, worst: inout [String: Worst]) throws {
        let device = try #require(MTLCreateSystemDefaultDevice())
        let library = try Self.loadLibrary(device)
        let fixture = try Fixture(device: device, context: context, large: large, seed: UInt64(context) &* 31 &+ UInt64(large.count),
                                  kvHeads: kvHeads, allScales: alias != 0, nanUnusedQueries: nanUnusedQueries)
        func chunk(_ shape: String) -> Int {
            let digits = shape.split(whereSeparator: { !$0.isNumber }).compactMap { Int($0) }
            return digits.count > 1 ? digits[1] : 64
        }
        var references = [Int: Merged]()
        func expected(_ shape: String) -> Merged {
            let key = alias != 0 ? chunk(shape) : 0
            if let known = references[key] { return known }
            let made = fixture.reference(alias: alias, chunk: chunk(shape))
            references[key] = made
            return made
        }
        for limit in spanLimits {
            let (spans, pagesPerSpan) = fixture.spans(limit: limit)
            let label = "\(name) context=\(context) kvHeads=\(kvHeads) spans=\(spans) pagesPerSpan=\(pagesPerSpan)\(across ? " across" : "")\(alias != 0 ? " alias=\(alias)" : "")"
            let empty = fixture.emptyRunRecords(spanLimit: limit)
            if straddles { #expect(!empty.isEmpty, "\(label): no row is blind to a span its block runs") }
            let base = try fixture.run(library, shape: Self.existing, spanLimit: limit, across: across, alias: alias)
            let shipped = fixture.merge(base.partials, spanLimit: limit)
            let shippedError = shipped.distance(to: expected(Self.existing))
            let shippedRecords = fixture.recordProblems(base.partials, spanLimit: limit, empty: empty)
            print("scan \(label): \(Self.existing) vs cpu \(shippedError.vector) logsum \(shippedError.logSum) gpu \(base.seconds * 1e3) ms")
            worst[Self.existing, default: Worst()].reference = max(worst[Self.existing, default: Worst()].reference, shippedError.vector)
            worst[Self.existing, default: Worst()].logSum = max(worst[Self.existing, default: Worst()].logSum, shippedError.logSum)
            if gate != .reported {
                #expect(shipped.problems.isEmpty, "\(label) \(Self.existing): \(shipped.problems)")
                #expect(shippedRecords.isEmpty, "\(label) \(Self.existing): \(shippedRecords)")
                #expect(shippedError.vector < Self.toleranceReference, "\(label) \(Self.existing) vs cpu: \(shippedError.vector)")
            }
            if gate == .exact {
                #expect(shippedError.logSum < Self.toleranceLogSum, "\(label) \(Self.existing) log sum: \(shippedError.logSum)")
            }
            for shape in Self.candidates {
                let result = try fixture.run(library, shape: shape, spanLimit: limit, across: across, alias: alias)
                let merged = fixture.merge(result.partials, spanLimit: limit)
                let records = fixture.recordProblems(result.partials, spanLimit: limit,
                                                     empty: Self.isWide(shape) ? fixture.emptyRunRecords(spanLimit: limit, wide: true) : empty)
                let comparable = gate != .reported && (alias == 0 || chunk(shape) == chunk(Self.existing))
                let toReference = merged.distance(to: expected(shape)), toShipped = merged.distance(to: shipped)
                print("scan \(label): \(shape) vs cpu \(toReference.vector) vs \(Self.existing) \(comparable ? String(toShipped.vector) : "-") logsum \(toReference.logSum) gpu \(result.seconds * 1e3) ms")
                var w = worst[shape, default: Worst()]
                w.reference = max(w.reference, toReference.vector); w.logSum = max(w.logSum, toReference.logSum)
                if comparable { w.existing = max(w.existing, toShipped.vector) }
                worst[shape] = w
                #expect(merged.problems.isEmpty, "\(label) \(shape): \(merged.problems)")
                #expect(records.isEmpty, "\(label) \(shape): \(records)")
                #expect(toReference.vector < Self.toleranceReference, "\(label) \(shape) vs cpu: \(toReference.vector)")
                #expect(toReference.logSum < Self.toleranceLogSum, "\(label) \(shape) log sum: \(toReference.logSum)")
                if comparable {
                    #expect(toShipped.vector < Self.toleranceKernels, "\(label) \(shape) vs \(Self.existing): \(toShipped.vector)")
                }
            }
        }
    }

    func report(_ test: String, _ worst: [String: Worst]) {
        for (shape, w) in worst.sorted(by: { $0.key < $1.key }) {
            print("scan worst [\(test)] \(shape): vs cpu \(w.reference), vs \(Self.existing) \(w.existing), log sum \(w.logSum)")
        }
    }

    @Test("full and short blocks at 300, 5,000 and 40,000 tokens with 1, 4 and 16 spans")
    func contextsAndSpans() throws {
        var worst = [String: Worst]()
        for context in [300, 5_003, 40_011] {
            try check("plain", context: context, large: [], spanLimits: [1, 4, 16], worst: &worst)
        }
        try check("plain", context: 5_003, large: [], spanLimits: [4], across: true, worst: &worst)
        report("contexts and spans", worst)
    }

    /// Blocks whose rows straddle the first page of a span, so some rows see nothing of a span
    /// that runs chunks for the rows after them. The long slot's rows are context-13 ... context-1
    /// in a block of eight and one of five; 5,120 is the first token of page 20, which starts a
    /// span with 16 or more spans.
    @Test("rows that see nothing of a span their block runs")
    func straddlingBlocks() throws {
        var worst = [String: Worst]()
        // Full block 5117...5124: rows 0...2 are blind to the span of page 20.
        try check("straddle full", context: 5_130, large: [], spanLimits: [16, 64], straddles: true, worst: &worst)
        try check("straddle full", context: 5_130, large: [], spanLimits: [16], across: true, straddles: true, worst: &worst)
        // Short block 5117...5121 (five rows), and NaN queries beyond the step's last short block.
        try check("straddle short", context: 5_122, large: [], spanLimits: [16, 64], straddles: true, nanUnusedQueries: true, worst: &worst)
        // A large key in the straddled span, seen by the block's later rows only.
        try check("straddle large", context: 5_130, large: [LargeKey(token: 5_121, factor: 50)], spanLimits: [16],
                  shipped: .vector, straddles: true, worst: &worst)
        report("straddling blocks", worst)
    }

    /// The candidates take their select-free form for a chunk that ends exactly at a block's
    /// first position (firstPos % chunk == chunk - 1), where the shipped kernel still masks; and
    /// they take it for short blocks too, whose queries beyond the count are then computed on
    /// real tokens and dropped. NaN queries after the step's rows show that nothing of those
    /// dropped queries reaches a written record.
    @Test("the last unmasked chunk, full and short blocks")
    func unmaskedBoundary() throws {
        var worst = [String: Worst]()
        // Full block first at 5,119 = 80 * 64 - 1 (and 160 * 32 - 1); it also straddles page 20.
        try check("boundary full", context: 5_132, large: [], spanLimits: [1, 16], worst: &worst)
        try check("boundary full", context: 5_132, large: [], spanLimits: [64], straddles: true, worst: &worst)
        // Short block (five rows) first at 5,119.
        try check("boundary short", context: 5_124, large: [], spanLimits: [1, 16], nanUnusedQueries: true, worst: &worst)
        // Short blocks far from any boundary, with NaN in the queries beyond the count.
        try check("short nan queries", context: 40_011, large: [], spanLimits: [4], nanUnusedQueries: true, worst: &worst)
        report("unmasked boundary", worst)
    }

    /// Four KV heads (24 query heads) on both grids, and aliasTokens (the SPLOSH_ATTN_ALIAS
    /// timing probe), where every chunk is read from the pool's first tokens.
    @Test("four KV heads, and aliasTokens")
    func headsAndAlias() throws {
        var worst = [String: Worst]()
        try check("four kv heads", context: 5_003, large: [], spanLimits: [4], kvHeads: 4, worst: &worst)
        try check("four kv heads", context: 5_130, large: [], spanLimits: [16], across: true, kvHeads: 4, straddles: true, worst: &worst)
        try check("alias", context: 5_003, large: [], spanLimits: [1, 16], alias: 448, worst: &worst)
        try check("alias", context: 5_003, large: [], spanLimits: [4], across: true, kvHeads: 4, alias: 4_096, worst: &worst)
        report("heads and alias", worst)
    }

    @Test("a key with a very large scale late, early and among the step's own rows")
    func largeKeys() throws {
        var worst = [String: Worst]()
        // With one span the late key comes after more than 600 chunks of the same threadgroup;
        // for the queries it suits, the score rises some hundreds above everything before it.
        try check("large late", context: 40_011, large: [LargeKey(token: 40_011 - 150, factor: 50)], spanLimits: [1, 16],
                  shipped: .vector, worst: &worst)
        // In the span's first chunk, so the shipped kernel's reference starts on it.
        try check("large early", context: 40_011, large: [LargeKey(token: 5, factor: 50)], spanLimits: [1, 16], worst: &worst)
        // Visible to the long slot's last four rows only.
        try check("large in rows", context: 5_003, large: [LargeKey(token: 5_003 - 4, factor: 50)], spanLimits: [1, 16],
                  shipped: .vector, worst: &worst)
        report("large keys", worst)
    }

    /// Two large keys in one span, each far above the row's ordinary scores: the second chunk of
    /// the context and a late one, or both in one chunk. The shipped kernel clamps both at 60
    /// above its reference and so can weigh them in the wrong order; it is reported, not required.
    @Test("two large keys in one span")
    func twoLargeKeys() throws {
        var worst = [String: Worst]()
        try check("large early and late", context: 5_003,
                  large: [LargeKey(token: 70, factor: 30), LargeKey(token: 5_003 - 300, factor: 60)], spanLimits: [1, 4],
                  shipped: .reported, worst: &worst)
        let first = (5_003 - 400) / 64 * 64 + 3
        try check("two large in a chunk", context: 5_003, large: [LargeKey(token: first, factor: 50), LargeKey(token: first + 9, factor: 44)],
                  spanLimits: [1, 16], shipped: .reported, worst: &worst)
        report("two large keys in one span", worst)
    }

    /// Opt-in (SPLOSH_SCAN_TIMING=1): GPU time of each kernel on a 128-row step at 40,000 tokens
    /// with 16 spans, the kernels taken in turn for several rounds. Synthetic keys: an
    /// indication only, the real model is what decides.
    @Test("interleaved timing on a 128-row step")
    func timing() throws {
        guard ProcessInfo.processInfo.environment["SPLOSH_SCAN_TIMING"] == "1" else { return }
        let device = try #require(MTLCreateSystemDefaultDevice())
        let library = try Self.loadLibrary(device)
        for context in [5_003, 40_011] {
            let fixture = try Fixture(device: device, context: context, large: [], seed: 11, slotRows: [64, 64])
            let shapes = [Self.existing] + Self.candidates
            var times = [String: [Double]]()
            for round in 0..<9 {
                for shape in shapes {
                    let seconds = try fixture.run(library, shape: shape, spanLimit: 16, across: false).seconds
                    if round > 0 { times[shape, default: []].append(seconds * 1e3) }
                }
            }
            for shape in shapes {
                let sorted = times[shape]!.sorted()
                print("scan timing context=\(context) rows=128 \(shape): median \(sorted[sorted.count / 2]) ms, least \(sorted[0]) ms")
            }
        }
    }

    /// A kernel that finds a score partition other than the one it expects must write NaN
    /// references, and this test's merge must report them.
    @Test("a kernel that expects another score partition poisons its output")
    func poison() throws {
        let device = try #require(MTLCreateSystemDefaultDevice())
        let library = try Self.loadLibrary(device)
        let fixture = try Fixture(device: device, context: 300, large: [], seed: 7)
        let result = try fixture.run(library, shape: "x48c64s8", spanLimit: 4, across: false)
        let (spans, _) = fixture.spans(limit: 4)
        var references = 0, poisoned = 0
        for entry in 0..<(fixture.rows * fixture.heads * spans) {
            references += 1
            if result.partials[entry * Self.record].isNaN { poisoned += 1 }
        }
        #expect(poisoned == references, "\(poisoned) of \(references) references are NaN")
        #expect(!fixture.merge(result.partials, spanLimit: 4).problems.isEmpty)
    }
}
