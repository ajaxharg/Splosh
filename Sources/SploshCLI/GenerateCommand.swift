// GenerateCommand.swift — one-shot greedy generation straight through the engine.
//
// This is the shortest path from a prompt to tokens: no HTTP, no scheduler. It exists so the
// whole-model path can be exercised and timed directly against the artifact.

import Foundation
import Metal
import SploshModel
import SploshRuntime
import SploshServer

public enum GenerateCommand {
    public static let usage = """
        usage: splosh generate --prompt <text> [options]

          --prompt <text>        text to complete
          --chat                 wrap the prompt as a single user turn of the chat template
          --weights <path>       SPLW artifact; default the tiled artifact in models/q4 if present
          --tokenizer <dir>      default inputs/tokenizer
          --max-tokens <n>       default 64
          --ids                  print token ids as well as text
          --speculate            decode speculatively with the DFlash 2 draft model
          --draft <path>         draft safetensors; default: the Hugging Face cache copy
          --help
        """

    public static func run(_ arguments: [String]) -> Int32 {
        var prompt: String?
        var weights = ServeConfig.defaultWeightsPath
        var tokenizerDir = "inputs/tokenizer"
        var maxTokens = 64
        var chat = false, showIDs = false, speculate = false
        var draftPath: String?
        var index = 0
        func value() -> String? {
            index += 1
            return index < arguments.count ? arguments[index] : nil
        }
        while index < arguments.count {
            switch arguments[index] {
            case "--prompt": prompt = value()
            case "--weights": weights = value() ?? weights
            case "--tokenizer": tokenizerDir = value() ?? tokenizerDir
            case "--max-tokens": maxTokens = value().flatMap(Int.init) ?? maxTokens
            case "--chat": chat = true
            case "--ids": showIDs = true
            case "--draft": draftPath = value()
            case "--speculate": speculate = true
            default:
                SploshCLI.writeStderr("generate: unexpected argument '\(arguments[index])'\n\(usage)\n")
                return ExitStatus.usage
            }
            index += 1
        }
        guard let prompt else {
            SploshCLI.writeStderr("generate: --prompt is required\n\(usage)\n")
            return ExitStatus.usage
        }
        do {
            guard let device = MTLCreateSystemDefaultDevice() else { throw CLIError("no Metal device") }
            let dir = URL(fileURLWithPath: tokenizerDir, isDirectory: true)
            let tokenizer = try Tokenizer(tokenizerURL: dir.appendingPathComponent("tokenizer.json"),
                                          configURL: dir.appendingPathComponent("tokenizer_config.json"))
            let loadStart = Date()
            let model = try ModelWeights(device: device, artifactURL: URL(fileURLWithPath: weights))
            var engineConfig = EngineConfig(maxSlots: 1, maxRows: Int(ProcessInfo.processInfo.environment["SPLOSH_ROWS"] ?? "") ?? 64)
            if let format = ProcessInfo.processInfo.environment["SPLOSH_KV"].flatMap(EngineConfig.KVFormat.init(rawValue:)) { engineConfig.kvFormat = format }
            // SPLOSH_CONTEXT: room for a longer prompt than the default 32K.
            if let context = Int(ProcessInfo.processInfo.environment["SPLOSH_CONTEXT"] ?? "") {
                engineConfig.maxContext = context
                engineConfig.kvPages = context / 256 + 4
            }
            let engine = try Engine(device: device, weights: model, config: engineConfig)
            let memory = engine.memory
            SploshCLI.writeStderr(String(format: "loaded %d tensors, %.2f GiB resident, in %.2fs\n",
                                         model.resident.tensorCount, Double(memory.weightBytes) / 1_073_741_824,
                                         Date().timeIntervalSince(loadStart)))

            if ProcessInfo.processInfo.environment["SPLOSH_PROBE"] != nil {
                let rate = try engine.probeReadBandwidth()
                for (name, rate) in try engine.probeArithmetic() {
                    SploshCLI.writeStderr(String(format: "arithmetic %@: %.2f T mul-add/s\n", name, rate / 1e12))
                }
                SploshCLI.writeStderr(String(format: "command-buffer-per-dispatch: %.1f us\n", try engine.probeCommandBufferOverhead() * 1e6))
                for rows in [1, 2, 4, 8, 16, 32, 64, 128] where rows <= engine.config.maxRows {
                    SploshCLI.writeStderr(String(format: "all GEMMs, %2d rows, no stage boundaries: %.1f ms\n", rows, try engine.probeWeightPass(rows: rows) * 1000))
                }
                SploshCLI.writeStderr(String(format: "dispatch overhead: serial %.1f us, concurrent %.1f us, concurrent+barrier %.1f us, encoder-per-dispatch %.1f us\n",
                                             try engine.probeDispatchOverhead() * 1e6,
                                             try engine.probeDispatchOverhead(concurrent: true) * 1e6,
                                             try engine.probeDispatchOverhead(concurrent: true, barriers: true) * 1e6,
                                             try engine.probeDispatchOverhead(concurrent: true, split: true) * 1e6))
                SploshCLI.writeStderr(String(format: "read bandwidth: %.1f GB/s -> %.1f ms to read the weights once\n",
                                             rate / 1e9, Double(memory.weightBytes) / rate * 1000))
            }
            if let modes = ProcessInfo.processInfo.environment["SPLOSH_ATTNDIFF"] {
                // Compare the first full-attention layer's output between two attention modes.
                let names = modes.split(separator: ",").map(String.init)
                let rowsPerStep = Int(ProcessInfo.processInfo.environment["SPLOSH_ROWS"] ?? "") ?? 12
                let text = chat ? ChatRenderer.render(messages: [RenderMessage(role: .user, content: prompt)]) : prompt
                let ids = tokenizer.encode(text)
                var engines: [Engine] = []
                for name in names {
                    setenv("SPLOSH_ATTN", name, 1)
                    var diffConfig = EngineConfig(maxSlots: 1, maxRows: max(rowsPerStep, 8))
                    diffConfig.kvFormat = name == "fp16" ? .fp16 : name == "i8" ? .int8 : .q4
                    let e = try Engine(device: device, weights: model, config: diffConfig)
                    e.debugMaxBlocks = Int(ProcessInfo.processInfo.environment["SPLOSH_DEBUG_BLOCKS"] ?? "") ?? 4
                    e.resetSlot(0)
                    engines.append(e)
                }
                var position = 0, reported = 0
                var worstRatio: Float = 0
                while position < ids.count, reported < 6 {
                    let end = min(position + rowsPerStep, ids.count)
                    let rows = (position..<end).map { EngineRow(slot: 0, token: ids[$0], position: $0, wantLogits: false) }
                    var outputs: [[Float]] = []
                    var operands: [(values: [Float], sums: [Float])] = []
                    for e in engines { try e.step(rows); outputs.append(e.debugCore(rows: rows.count)); operands.append(e.debugCoreOperand(rows: rows.count)) }
                    if rows.count >= 3, reported < 6 {
                        var worstValue: Float = 0, worstSum: Float = 0, sumIndex = 0
                        for i in operands[0].values.indices { worstValue = max(worstValue, abs(operands[0].values[i] - operands[1].values[i])) }
                        for i in operands[0].sums.indices where abs(operands[0].sums[i] - operands[1].sums[i]) > worstSum { worstSum = abs(operands[0].sums[i] - operands[1].sums[i]); sumIndex = i }
                        if worstValue > 0.05 || worstSum > 0.05 {
                            SploshCLI.writeStderr("operand mismatch in step at position \(position): bf16 max diff \(worstValue); sums max diff \(worstSum) at index \(sumIndex) (row \(sumIndex / 96), group \(sumIndex % 96)): \(operands[0].sums[sumIndex]) vs \(operands[1].sums[sumIndex])\n")
                            reported += 1
                        }
                    }
                    for row in 0..<rows.count {
                        var worst: Float = 0, worstHead = 0, scale: Float = 0
                        for head in 0..<24 {
                            var diff: Float = 0, magnitude: Float = 0
                            for d in 0..<256 {
                                let a = outputs[0][(row * 24 + head) * 256 + d], b = outputs[1][(row * 24 + head) * 256 + d]
                                diff = max(diff, abs(a - b)); magnitude = max(magnitude, abs(a))
                            }
                            if diff > worst { worst = diff; worstHead = head; scale = magnitude }
                        }
                        if scale > 0 { worstRatio = max(worstRatio, worst / scale) }
                        if worst > 0.05 * max(scale, 1e-3), reported < 6 {
                            SploshCLI.writeStderr("mismatch at position \(position + row) (step row \(row) of \(rows.count)): head \(worstHead) max diff \(worst) vs magnitude \(scale)\n")
                            reported += 1
                        }
                    }
                    position = end
                }
                SploshCLI.writeStderr("worst per-head relative difference: \(worstRatio)\n")
                SploshCLI.writeStderr(reported == 0 ? "attention outputs agree across \(ids.count) positions\n" : "\(reported)+ mismatches\n")
                return ExitStatus.ok
            }
            if ProcessInfo.processInfo.environment["SPLOSH_PADCHECK"] != nil {
                // Does what sits in the unused rows of an accelerator tile change the used rows?
                let ids = tokenizer.encode(prompt)
                guard ids.count >= 40 else { throw CLIError("prompt too short for the padding check") }
                var reference: [[Float]] = []
                for padding: Float in [0, 0, 1, 100, 10000] {
                    let e = try Engine(device: device, weights: model, config: EngineConfig(maxSlots: 1, maxRows: 64))
                    e.resetSlot(0)
                    e.debugPadding = padding
                    try e.step((0..<24).map { EngineRow(slot: 0, token: ids[$0], position: $0, wantLogits: false) })
                    let checkRows = Int(ProcessInfo.processInfo.environment["SPLOSH_PADCHECK"] ?? "") ?? 8
                    try e.step((24..<24 + checkRows).map { EngineRow(slot: 0, token: ids[$0], position: $0, wantLogits: true) })
                    let rows = (0..<8).map { $0 < checkRows ? Array(e.logits($0).prefix(tokenizer.idLimit)) : [Float](repeating: 0, count: tokenizer.idLimit) }
                    if reference.isEmpty { reference = rows; continue }
                    var worst: Float = 0
                    for row in 0..<8 { for token in 0..<tokenizer.idLimit { worst = max(worst, abs(rows[row][token] - reference[row][token])) } }
                    SploshCLI.writeStderr(String(format: "padding rows = %g: worst logit difference from zero padding %.4f\n", padding, worst))
                }
                return ExitStatus.ok
            }
            if ProcessInfo.processInfo.environment["SPLOSH_VERIFYCHECK"] != nil {
                // Speculative rows against ordinary rows over the same tokens: the logits of an
                // eight-row block must not depend on which kind of row evaluated it, nor on how
                // many of the previous block's rows were accepted.
                let ids = tokenizer.encode(prompt)
                let width = Int(ProcessInfo.processInfo.environment["SPLOSH_VERIFYCHECK_WIDTH"] ?? "") ?? 8
                let accept = min(Int(ProcessInfo.processInfo.environment["SPLOSH_VERIFYCHECK"] ?? "") ?? width, width)
                let lead = Int(ProcessInfo.processInfo.environment["SPLOSH_VERIFYCHECK_LEAD"] ?? "") ?? 24
                // SPLOSH_VERIFYCHECK_REF=first: the reference is itself a speculative block of
                // the same width with only its first row kept, one token a step. Both sides then
                // run the same kernels, and what is measured is whether a row's logits depend
                // on where in the block it sits.
                let firstRowReference = ProcessInfo.processInfo.environment["SPLOSH_VERIFYCHECK_REF"] == "first"
                // SPLOSH_VERIFYCHECK_REF=single: the reference is ordinary one-row steps, the
                // path plain decoding takes. How far that is from a multi-row step is the size
                // of the rounding differences between kernel paths.
                let singleRowReference = ProcessInfo.processInfo.environment["SPLOSH_VERIFYCHECK_REF"] == "single"
                var worstKL = 0.0, totalKL = 0.0, worstTop: Float = 0
                let positions = Int(ProcessInfo.processInfo.environment["SPLOSH_VERIFYCHECK_POSITIONS"] ?? "") ?? width * 12
                guard ids.count >= lead + positions else { throw CLIError("prompt too short for the verify check") }
                let plainEngine = try Engine(device: device, weights: model, config: EngineConfig(maxSlots: 1, maxRows: 64))
                let specEngine = try Engine(device: device, weights: model, config: EngineConfig(maxSlots: 1, maxRows: 64))
                for e in [plainEngine, specEngine] {
                    e.resetSlot(0)
                    try e.step((0..<lead).map { EngineRow(slot: 0, token: ids[$0], position: $0, wantLogits: false) })
                }
                var position = lead, worst: Float = 0, flips = 0, compared = 0
                var worstByRow = [Float](repeating: 0, count: width)
                var reference = [Float](repeating: 0, count: tokenizer.idLimit)
                while position + width <= ids.count, compared < positions {
                    let block = (0..<width).map { EngineRow(slot: 0, token: ids[position + $0], position: position + $0, wantLogits: true, verify: true) }
                    try specEngine.step(block)
                    // Ordinary engine: exactly the rows the speculative one will keep.
                    if !firstRowReference && !singleRowReference {
                        try plainEngine.step((0..<accept).map { EngineRow(slot: 0, token: ids[position + $0], position: position + $0, wantLogits: true) })
                    }
                    for row in 0..<accept {
                        if firstRowReference {
                            let at = position + row
                            try plainEngine.step((0..<width).map { EngineRow(slot: 0, token: ids[at], position: at + $0, wantLogits: $0 == 0, verify: true) })
                            plainEngine.acceptVerify(0, rows: 1, tokenCount: at + 1)
                        }
                        if singleRowReference {
                            try plainEngine.step([EngineRow(slot: 0, token: ids[position + row], position: position + row, wantLogits: true)])
                        }
                        let a = plainEngine.logits(firstRowReference || singleRowReference ? 0 : row), b = specEngine.logits(row)
                        // What the difference does to the distribution: KL divergence, and the
                        // largest difference among the reference's twenty likeliest tokens.
                        do {
                            let limit = tokenizer.idLimit
                            let top = TopK.select(a.baseAddress!, limit: limit, k: 20)
                            for id in top.ids { worstTop = max(worstTop, abs(a[id] - b[id])) }
                            var maxA = -Float.infinity, maxB = -Float.infinity
                            for token in 0..<limit { maxA = max(maxA, a[token]); maxB = max(maxB, b[token]) }
                            var sumA = 0.0, sumB = 0.0
                            for token in 0..<limit { sumA += Double(exp(a[token] - maxA)); sumB += Double(exp(b[token] - maxB)) }
                            var kl = 0.0
                            for token in 0..<limit {
                                let pa = Double(exp(a[token] - maxA)) / sumA
                                if pa > 1e-12 { kl += pa * (Double(a[token] - maxA) - log(sumA) - Double(b[token] - maxB) + log(sumB)) }
                            }
                            worstKL = max(worstKL, kl); totalKL += kl
                        }
                        var difference: Float = 0
                        for token in 0..<tokenizer.idLimit { difference = max(difference, abs(a[token] - b[token])); reference[token] = a[token] }
                        if ProcessInfo.processInfo.environment["SPLOSH_VERIFYCHECK_TRACE"] != nil, difference > 0 {
                            SploshCLI.writeStderr(String(format: "  position %d (row %d of its block): %.5f\n", position + row, row, difference))
                        }
                        worst = max(worst, difference)
                        worstByRow[row] = max(worstByRow[row], difference)
                        if argmax(a, limit: tokenizer.idLimit) != argmax(b, limit: tokenizer.idLimit) { flips += 1 }
                        compared += 1
                    }
                    specEngine.acceptVerify(0, rows: accept, tokenCount: position + accept)
                    position += accept
                }
                SploshCLI.writeStderr(String(format: "verify check, %d-row blocks, %d rows kept per block: %d positions, worst logit difference %.4f, %d arg-max flips\n", width, accept, compared, worst, flips))
                SploshCLI.writeStderr("  worst by row of the block: " + worstByRow.map { String(format: "%.4f", $0) }.joined(separator: " ") + "\n")
                SploshCLI.writeStderr(String(format: "  KL divergence from the reference: mean %.6f, worst %.6f; worst difference among the top 20 tokens %.4f\n", totalKL / Double(max(compared, 1)), worstKL, worstTop))
                return ExitStatus.ok
            }
            if let rowsText = ProcessInfo.processInfo.environment["SPLOSH_EMITCHECK"], let checkRows = Int(rowsText) {
                // Stop after the first block's mixer: h then holds the residual output whose
                // operand the output projection emitted.
                let ids = tokenizer.encode(prompt)
                let e = try Engine(device: device, weights: model, config: EngineConfig(maxSlots: 1, maxRows: max(checkRows, 8)))
                e.resetSlot(0)
                e.debugMaxBlocks = Int(ProcessInfo.processInfo.environment["SPLOSH_DEBUG_BLOCKS"] ?? "") ?? 1
                let rows = (0..<min(checkRows, ids.count)).map { EngineRow(slot: 0, token: ids[$0], position: $0, wantLogits: false) }
                try e.step(rows)
                let check = e.debugEmitCheck(rows: rows.count)
                SploshCLI.writeStderr("emit check over \(rows.count) rows: worst relative error in sums of squares \(check.squares) (at \(check.worstSquareAt)), in operand sums \(check.sums); non-finite groups \(check.nonFinite)\n")
                return ExitStatus.ok
            }
            if let modes = ProcessInfo.processInfo.environment["SPLOSH_KVQUALITY"] {
                // Teacher-forced quality of each KV format against the first one listed: next-token
                // loss on the prompt's own continuation, plus divergence of the full distributions.
                let ids = tokenizer.encode(prompt)
                let window = Int(ProcessInfo.processInfo.environment["SPLOSH_KVQUALITY_WINDOW"] ?? "") ?? 64
                guard ids.count > window * 2 else { throw CLIError("prompt too short for the quality probe") }
                let usable = (ids.count / window) * window
                var reference: [[Float]] = []
                for name in modes.split(separator: ",").map(String.init) {
                    setenv("SPLOSH_ATTN", name, 1)
                    var config = EngineConfig(maxSlots: 1, maxRows: window, kvPages: usable / 256 + 2, maxContext: usable + 256)
                    config.kvFormat = name == "fp16" ? .fp16 : name == "i8" ? .int8 : .q4
                    let e = try Engine(device: device, weights: model, config: config)
                    e.resetSlot(0)
                    var logProbs: [[Float]] = []
                    var position = 0
                    // Logits for the last `tail` windows only.
                    let tail = Int(ProcessInfo.processInfo.environment["SPLOSH_KVQUALITY_WINDOWS"] ?? "") ?? 4
                    while position < usable {
                        let want = position >= usable - tail * window
                        let rows = (position..<position + window).map { EngineRow(slot: 0, token: ids[$0], position: $0, wantLogits: want) }
                        try e.step(rows)
                        if want {
                            for row in 0..<window {
                                let raw = e.logits(row)
                                var peak = -Float.infinity
                                for value in raw where value > peak { peak = value }
                                var total: Double = 0
                                for value in raw { total += Double(exp(value - peak)) }
                                let shift = peak + Float(log(total))
                                logProbs.append(raw.map { $0 - shift })
                            }
                        }
                        position += window
                    }
                    let first = usable - tail * window
                    var loss: Double = 0, counted = 0
                    for (index, row) in logProbs.enumerated() where first + index + 1 < ids.count {
                        loss -= Double(row[ids[first + index + 1]]); counted += 1
                    }
                    if reference.isEmpty {
                        reference = logProbs
                        SploshCLI.writeStderr(String(format: "%-7@ loss %.4f over %d positions at context %d (reference)\n", name as NSString, loss / Double(counted), counted, usable))
                        continue
                    }
                    var meanKL: Double = 0, worstKL: Double = 0, agree = 0
                    for (index, row) in logProbs.enumerated() {
                        let ref = reference[index]
                        var kl: Double = 0, bestRef = 0, bestRow = 0
                        for token in 0..<ref.count {
                            let pr = Double(exp(ref[token]))
                            if pr > 1e-9 { kl += pr * Double(ref[token] - row[token]) }
                            if ref[token] > ref[bestRef] { bestRef = token }
                            if row[token] > row[bestRow] { bestRow = token }
                        }
                        meanKL += kl; worstKL = max(worstKL, kl)
                        if bestRef == bestRow { agree += 1 }
                    }
                    SploshCLI.writeStderr(String(format: "%-7@ loss %.4f; KL vs reference mean %.5f, worst %.4f; top-1 agreement %d/%d\n",
                                                 name as NSString, loss / Double(counted), meanKL / Double(logProbs.count), worstKL, agree, logProbs.count))
                }
                return ExitStatus.ok
            }
            if let spec = ProcessInfo.processInfo.environment["SPLOSH_SPINPROBE"] {
                // How the GPU spreads a dispatch over its cores: threadgroups that only spin, in
                // grids of several shapes. SPLOSH_SPINPROBE="<threads per group>:<iterations>:<barriers>".
                let parts = spec.split(separator: ":").compactMap { Int($0) }
                let threads = parts.count > 0 ? parts[0] : 256, iterations = parts.count > 1 ? parts[1] : 2000, barriers = parts.count > 2 ? parts[2] : 0
                let probe = try Engine(device: device, weights: model, config: EngineConfig(maxSlots: 1, maxRows: 16, kvPages: 64, maxContext: 4096))
                try probe.warmUp()
                func time(_ grid: (Int, Int)) throws -> Double {
                    var best = Double.infinity
                    for _ in 0..<5 { best = min(best, try probe.spinProbe(grid: grid, threads: threads, iterations: iterations, barriers: barriers, repeats: 8) / 8) }
                    return best
                }
                _ = try time((64, 16))
                if ProcessInfo.processInfo.environment["SPLOSH_SPINPROBE_AFTER"] != nil {
                    func best(_ body: () throws -> Double) rethrows -> Double { var b = Double.infinity; for _ in 0..<5 { b = min(b, try body()) }; return b }
                    let alone = try best { try probe.afterSpinProbe(grid: (4, 16), spin: 0, matmul: true) }
                    SploshCLI.writeStderr(String(format: "matmul dispatches alone: %.1f us each\n", alone * 1e6 / 16))
                    for spin in [50, 200, 1000, 4000, 16000] {
                        let spinOnly = try best { try probe.afterSpinProbe(grid: (4, 16), spin: spin, matmul: false) }
                        let both = try best { try probe.afterSpinProbe(grid: (4, 16), spin: spin, matmul: true) }
                        SploshCLI.writeStderr(String(format: "after a spin dispatch of %.0f us: the matmul dispatch costs %.1f us\n", spinOnly * 1e6 / 16, (both - spinOnly) * 1e6 / 16))
                    }
                    return ExitStatus.ok
                }
                if ProcessInfo.processInfo.environment["SPLOSH_SPINPROBE_SWITCH"] != nil {
                    // Sixteen dispatches in one pass: all of one kind, or the two kinds in turn.
                    for grid in [(4, 16), (16, 16), (64, 16)] {
                        var same = Double.infinity, mixed = Double.infinity
                        for _ in 0..<5 {
                            same = min(same, try probe.spinProbe(grid: grid, threads: 0, iterations: iterations, barriers: barriers, repeats: 16))
                            mixed = min(mixed, try probe.spinProbe(grid: grid, threads: 0, iterations: iterations, barriers: barriers, repeats: 16, alternate: true))
                        }
                        SploshCLI.writeStderr(String(format: "switch probe, grid %d x %d: sixteen of one kind %.1f us; eight and eight in turn %.1f us\n",
                                                     grid.0, grid.1, same * 1e6, mixed * 1e6))
                    }
                    return ExitStatus.ok
                }
                let one = try time((1, 1))
                SploshCLI.writeStderr(String(format: "spin probe: %d threads a group, %d iterations, %d barriers; one threadgroup alone %.1f us\n", threads, iterations, barriers, one * 1e6))
                for grid in [(2, 1), (4, 1), (8, 1), (16, 1), (20, 1), (32, 1), (64, 1), (1, 64), (4, 16), (16, 4), (8, 8), (12, 64), (64, 12), (16, 16), (36, 16), (64, 16), (256, 1), (1024, 1), (32, 32)] {
                    let seconds = try time(grid)
                    let count = Double(grid.0 * grid.1)
                    SploshCLI.writeStderr(String(format: "  grid %4d x %-3d (%5.0f groups): %8.1f us a dispatch, %6.2f us a group, as if on %.1f cores\n",
                                                 grid.0, grid.1, count, seconds * 1e6, seconds * 1e6 / count, count * one / seconds))
                }
                return ExitStatus.ok
            }
            if let contexts = ProcessInfo.processInfo.environment["SPLOSH_CTXBENCH"] {
                // Step cost at a given context length without paying for the prefill: rows are
                // placed at the end of a context whose KV pages are allocated but unwritten.
                // The arithmetic is the same as with real keys; only the values differ.
                let benchRows = Int(ProcessInfo.processInfo.environment["SPLOSH_CTXBENCH_ROWS"] ?? "") ?? 128
                let verifyRows = Int(ProcessInfo.processInfo.environment["SPLOSH_CTXBENCH_VERIFY"] ?? "") ?? 8
                var benchConfig = EngineConfig(maxSlots: 1, maxRows: benchRows, kvPages: 1024, maxContext: 262_144)
                if let format = ProcessInfo.processInfo.environment["SPLOSH_KV"].flatMap(EngineConfig.KVFormat.init(rawValue:)) { benchConfig.kvFormat = format }
                let wide = try Engine(device: device, weights: model, config: benchConfig)
                try wide.warmUp()
                for context in contexts.split(separator: ",").compactMap({ Int($0) }) {
                    wide.resetSlot(0)
                    if ProcessInfo.processInfo.environment["SPLOSH_CTXBENCH_EMPTY"] == nil {
                        // Touch the pages first: one throwaway row at the end of the context
                        // allocates them, then they are filled with non-zero data.
                        try wide.step([EngineRow(slot: 0, token: 1000, position: max(context - 1, 0), wantLogits: false)])
                        wide.debugFillKV(0)
                    }
                    var prefill: [Double] = [], verify: [Double] = []
                    let iterations = Int(ProcessInfo.processInfo.environment["SPLOSH_CTXBENCH_ITERATIONS"] ?? "") ?? 5
                    // SPLOSH_CTXBENCH_AB=gdn: alternate steps between two settings of one engine,
                    // so that whatever else the machine is doing falls on both alike.
                    let alternate = ProcessInfo.processInfo.environment["SPLOSH_CTXBENCH_AB"]
                    var sides: [[Double]] = [[], []]
                    for iteration in 0..<iterations {
                        if alternate == "gdn" { wide.gdnSplitRows = iteration % 2 == 0 ? Int.max : 32 }
                        if alternate == "attnmask" { wide.attnSkipMask = iteration % 2 == 0 ? 0 : (Int(ProcessInfo.processInfo.environment["SPLOSH_ATTN_MASK"] ?? "") ?? 0) }
                        if alternate == "spans" { wide.wideSpanLimit = iteration % 2 == 0 ? 0 : (Int(ProcessInfo.processInfo.environment["SPLOSH_AB_SPANS"] ?? "") ?? 4) }
                        if alternate == "shape" {
                            try wide.setScanShape(iteration % 2 == 0 ? (ProcessInfo.processInfo.environment["SPLOSH_AB_BASE"] ?? "m48c64s8")
                                                 : (ProcessInfo.processInfo.environment["SPLOSH_AB_SHAPE"] ?? "m48c32s8"))
                        }
                        if alternate == "pad" { wide.attnGridPad = iteration % 2 == 0 ? 1 : (Int(ProcessInfo.processInfo.environment["SPLOSH_AB_PAD"] ?? "") ?? 16) }
                        if alternate == "norm" { wide.normWideRows = iteration % 2 == 0 ? Int.max : 32 }
                        if alternate == "across" { wide.attnBlocksAcross = iteration % 2 == 1 }
                        if alternate == "gdnmask" { wide.gdnSkipMask = iteration % 2 == 0 ? 0 : (Int(ProcessInfo.processInfo.environment["SPLOSH_GDN_MASK"] ?? "") ?? 0) }
                        let base = context + iteration * (benchRows + verifyRows)
                        let rows = (0..<benchRows).map { EngineRow(slot: 0, token: 1000 + $0, position: base + $0, wantLogits: false) }
                        prefill.append(try wide.step(rows).gpuSeconds)
                        if alternate != nil, iteration >= 4 { sides[iteration % 2].append(prefill.last!) }
                        let block = (0..<verifyRows).map { EngineRow(slot: 0, token: 2000 + $0, position: base + benchRows + $0, wantLogits: true, verify: true) }
                        verify.append(try wide.step(block, captureFeatures: true).gpuSeconds)
                        if ProcessInfo.processInfo.environment["SPLOSH_DUMP_GEMM_PARTITION"] != nil {
                            let dump = wide.debugProjection(count: 128 * 80)
                            for thread in [0, 1, 2, 3, 4, 15, 16, 31] {
                                let capacity = Int(dump[thread * 80])
                                let pairs = (0..<min(capacity, 39)).map { "(\(Int(dump[thread * 80 + 1 + $0 * 2])),\(Int(dump[thread * 80 + 2 + $0 * 2])))" }
                                SploshCLI.writeStderr("thread \(thread): capacity \(capacity): \(pairs.joined(separator: " "))\n")
                            }
                            return ExitStatus.ok
                        }
                        wide.acceptVerify(0, rows: verifyRows, tokenCount: base + benchRows + verifyRows)
                    }
                    if ProcessInfo.processInfo.environment["SPLOSH_DUMP_SCOREMAP"] != nil {
                        let dump = wide.debugPartials(count: 256 * 40)
                        for thread in [0, 1, 2, 31, 32, 33, 64, 127, 128, 255] {
                            let pairs = (0..<16).map { "(\(Int(dump[thread * 40 + $0 * 2])),\(Int(dump[thread * 40 + $0 * 2 + 1])))" }.joined(separator: " ")
                            SploshCLI.writeStderr("thread \(thread): scores (token,query) \(pairs); values (dim,query) first (\(Int(dump[thread * 40 + 32])),\(Int(dump[thread * 40 + 33]))) last (\(Int(dump[thread * 40 + 34])),\(Int(dump[thread * 40 + 35]))) of \(Int(dump[thread * 40 + 36]))\n")
                        }
                    }
                    if ProcessInfo.processInfo.environment["SPLOSH_DUMP_PARTIALS"] != nil {
                        let dump = wide.debugPartials(count: 1024)
                        var histogram: [String: Int] = [:]
                        for thread in 0..<(Int(ProcessInfo.processInfo.environment["SPLOSH_DUMP_THREADS"] ?? "") ?? 256) { histogram["\(Int(dump[thread * 4]))/\(Int(dump[thread * 4 + 1])) \(Int(dump[thread * 4 + 2]))/\(Int(dump[thread * 4 + 3]))", default: 0] += 1 }
                        SploshCLI.writeStderr("partition (score capacity/valid, output capacity/valid): threads \(histogram)\n")
                    }
                    if ProcessInfo.processInfo.environment["SPLOSH_CTXBENCH_TRACE"] != nil {
                        SploshCLI.writeStderr("prefill ms: " + prefill.map { String(format: "%.0f", $0 * 1000) }.joined(separator: " ") + "\n")
                        SploshCLI.writeStderr("verify ms: " + verify.map { String(format: "%.0f", $0 * 1000) }.joined(separator: " ") + "\n")
                    }
                    if alternate != nil {
                        func summary(_ times: [Double]) -> String {
                            let sorted = times.sorted()
                            return String(format: "median %.1f ms, fastest %.1f", sorted[sorted.count / 2] * 1000, (sorted.first ?? 0) * 1000)
                        }
                        // Step by step: each second-setting step against the first-setting step before it.
                        let pairs = zip(sides[0], sides[1]).map { ($0 - $1) * 1000 }
                        let mean = pairs.reduce(0, +) / Double(max(pairs.count, 1))
                        let spread = (pairs.map { ($0 - mean) * ($0 - mean) }.reduce(0, +) / Double(max(pairs.count - 1, 1))).squareRoot() / Double(max(pairs.count, 1)).squareRoot()
                        let middle = pairs.sorted()[pairs.count / 2]
                        SploshCLI.writeStderr("alternating at \(context): first setting \(summary(sides[0])); second \(summary(sides[1]))\n")
                        SploshCLI.writeStderr(String(format: "paired at %d: first minus second %.1f ms (median %.1f, standard error %.1f, %d pairs)\n", context, mean, middle, spread, pairs.count))
                    }
                    // The GPU runs about a third faster for the first second after being idle.
                    // With more than five iterations the first four are dropped, so the figure
                    // is the sustained rate a long prefill actually gets.
                    // Every step, in order: what a prompt of that many tokens pays, heat included.
                    SploshCLI.writeStderr(String(format: "sustained: %d rows in %.2f s of prefill steps = %.0f tok/s\n",
                                                 prefill.count * benchRows, prefill.reduce(0, +), Double(prefill.count * benchRows) / prefill.reduce(0, +)))
                    if iterations > 5 { prefill.removeFirst(4); verify.removeFirst(4) }
                    prefill.sort(); verify.sort()
                    SploshCLI.writeStderr(String(format: "context %6d: prefill step (%d rows) %.0f ms = %.0f tok/s; verify step (%d rows) %.1f ms\n",
                                                 context, benchRows, prefill[prefill.count / 2] * 1000, Double(benchRows) / prefill[prefill.count / 2], verifyRows, verify[verify.count / 2] * 1000))
                }
                return ExitStatus.ok
            }
            if ProcessInfo.processInfo.environment["SPLOSH_MIXBENCH"] != nil {
                // What a step costs when prefill rows of one session share it with a verify
                // block of another, against the same rows on their own.
                let mixed = try Engine(device: device, weights: model, config: EngineConfig(maxSlots: 2, maxRows: 128))
                try mixed.warmUp()
                func time(_ label: String, prefill: Int, verify: Int, speculative: Bool, features: Bool) throws {
                    mixed.resetSlot(0); mixed.resetSlot(1)
                    var times: [Double] = []
                    for iteration in 0..<7 {
                        var rows = (0..<prefill).map { EngineRow(slot: 0, token: 1000 + $0, position: iteration * prefill + $0, wantLogits: false) }
                        rows += (0..<verify).map { EngineRow(slot: 1, token: 2000 + $0, position: iteration * verify + $0, wantLogits: true, verify: speculative) }
                        times.append(try mixed.step(rows, captureFeatures: features).gpuSeconds)
                        if speculative, verify > 0 { mixed.acceptVerify(1, rows: verify, tokenCount: (iteration + 1) * verify) }
                    }
                    times.sort()
                    SploshCLI.writeStderr(String(format: "mix: %@ %.0f ms\n", label.padding(toLength: 58, withPad: " ", startingAt: 0), times[3] * 1000))
                }
                try time("128 prefill rows", prefill: 128, verify: 0, speculative: false, features: false)
                try time("120 prefill rows", prefill: 120, verify: 0, speculative: false, features: false)
                try time("120 prefill rows, hidden states captured", prefill: 120, verify: 0, speculative: false, features: true)
                try time("120 prefill + 8 ordinary rows of another slot, with logits", prefill: 120, verify: 8, speculative: false, features: false)
                try time("120 prefill + 8 verify rows of another slot", prefill: 120, verify: 8, speculative: true, features: false)
                try time("120 prefill + 8 verify rows, hidden states captured", prefill: 120, verify: 8, speculative: true, features: true)
                try time("8 verify rows alone, hidden states captured", prefill: 0, verify: 8, speculative: true, features: true)
                return ExitStatus.ok
            }
            if ProcessInfo.processInfo.environment["SPLOSH_STEPBENCH"] != nil {
                // Verify-shaped steps: k sessions x 8 speculative rows each.
                let wide = try Engine(device: device, weights: model, config: EngineConfig(maxSlots: 8, maxRows: 128))
                try wide.warmUp()
                for sessions in [1, 2, 3, 4, 6, 8] {
                    for slot in 0..<8 { wide.resetSlot(slot) }
                    // Consecutive steps with no idle gap between them, so the GPU stays clocked up.
                    var times: [Double] = [], walls: [Double] = []
                    for iteration in 0..<10 {
                        var rows: [EngineRow] = []
                        for slot in 0..<sessions {
                            for offset in 0..<8 { rows.append(EngineRow(slot: slot, token: 1000 + slot * 8 + offset, position: iteration * 8 + offset, wantLogits: true, verify: true)) }
                        }
                        let stats = try wide.step(rows, captureFeatures: true)
                        times.append(stats.gpuSeconds)
                        walls.append(stats.wallSeconds)
                        for slot in 0..<sessions { wide.acceptVerify(slot, rows: 8, tokenCount: (iteration + 1) * 8) }
                    }
                    times.sort(); walls.sort()
                    SploshCLI.writeStderr(String(format: "step: %d x 8 = %2d rows: min %.1f ms, median %.1f ms; wall min %.1f ms, median %.1f ms\n", sessions, sessions * 8, times[0] * 1000, times[5] * 1000, walls[0] * 1000, walls[5] * 1000))
                }
                return ExitStatus.ok
            }
            let text = chat ? ChatRenderer.render(messages: [RenderMessage(role: .user, content: prompt)]) : prompt
            let ids = tokenizer.encode(text)
            guard !ids.isEmpty else { throw CLIError("prompt encoded to zero tokens") }
            if showIDs { SploshCLI.writeStderr("prompt ids: \(ids)\n") }

            try engine.warmUp()
            let prefillStart = Date()
            let tracePrefill = ProcessInfo.processInfo.environment["SPLOSH_TRACE_PREFILL"] != nil
            var position = 0
            while position < ids.count {
                let end = min(position + engine.config.maxRows, ids.count)
                let rows = (position..<end).map {
                    EngineRow(slot: 0, token: ids[$0], position: $0, wantLogits: $0 == ids.count - 1)
                }
                let stats = try engine.step(rows)
                if tracePrefill, (position / engine.config.maxRows) % 16 == 0 {
                    SploshCLI.writeStderr(String(format: "prefill step at %6d: gpu %.0f ms, wall %.0f ms, since start %.2fs\n",
                                                 position, stats.gpuSeconds * 1000, stats.wallSeconds * 1000, Date().timeIntervalSince(prefillStart)))
                }
                position = end
            }
            let prefillSeconds = Date().timeIntervalSince(prefillStart)

            if speculate {
                guard let url = draftPath.map({ URL(fileURLWithPath: $0) }) ?? Self.defaultDraftURL() else {
                    throw CLIError("no draft model found; pass --draft <model.safetensors>")
                }
                return try speculativeGenerate(engine: engine, draftURL: url, ids: ids, tokenizer: tokenizer, count: maxTokens, showIDs: showIDs)
            }
            if ProcessInfo.processInfo.environment["SPLOSH_SPECTEST"] != nil {
                return try speculationSelfTest(engine: engine, ids: ids, tokenizer: tokenizer, count: maxTokens)
            }
            var generated: [Int] = []
            var pending: [UInt8] = []
            let decodeStart = Date()
            var gpuSeconds = 0.0
            while generated.count < maxTokens {
                let next = argmax(engine.logits(0), limit: tokenizer.idLimit)
                if SpecialTokens.eosIDs.contains(next) { break }
                generated.append(next)
                pending += tokenizer.bytes(for: next) ?? []
                if let chunk = String(bytes: pending, encoding: .utf8) {
                    FileHandle.standardOutput.write(Data(chunk.utf8))
                    pending.removeAll()
                }
                let stats = try engine.step([EngineRow(slot: 0, token: next, position: ids.count + generated.count - 1, wantLogits: true)])
                gpuSeconds += stats.gpuSeconds
            }
            let decodeSeconds = Date().timeIntervalSince(decodeStart)
            print("")
            if showIDs { SploshCLI.writeStderr("generated ids: \(generated)\n") }
            SploshCLI.writeStderr(String(format: "prefill: %d tokens in %.2fs (%.1f tok/s)\n",
                                         ids.count, prefillSeconds, Double(ids.count) / prefillSeconds))
            if !generated.isEmpty {
                SploshCLI.writeStderr(String(format: "decode:  %d tokens in %.2fs (%.2f tok/s, gpu %.1f ms/token)\n",
                                             generated.count, decodeSeconds, Double(generated.count) / decodeSeconds,
                                             gpuSeconds / Double(generated.count) * 1000))
            }
            return ExitStatus.ok
        } catch {
            SploshCLI.writeStderr("generate failed: \(error)\n")
            return 1
        }
    }

    /// Bookkeeping check for the verify path. A reference decodes one token a step; the same
    /// tokens are then evaluated again in speculative blocks whose drafts are the true
    /// continuation corrupted at a rotating position, so every length of partial acceptance
    /// (and every so often a whole block) is exercised. Each kept row's distribution must be
    /// the reference's.
    ///
    /// "The same" is a bound on KL divergence, not token equality. A row evaluated after one
    /// accepted row and the same row evaluated after five differ in the last bits, and
    /// activations are rounded to bf16 at every matmul, so those bits decide a rounding now
    /// and then. Measured over 200 positions, for 8- and 16-row blocks alike: mean KL
    /// 0.00003-0.0001, worst 0.001-0.013 (a position where two tokens are nearly tied), and
    /// an occasional arg-max flip there, after which a free-running comparison fails on
    /// everything that follows. State restored wrongly after a partial acceptance is a
    /// different order of magnitude (SPLOSH_SPECTEST_BREAK=1 shows what that looks like).
    static func speculationSelfTest(engine: Engine, ids: [Int], tokenizer: Tokenizer, count: Int) throws -> Int32 {
        let limit = tokenizer.idLimit
        func prefill() throws {
            engine.resetSlot(0)
            var position = 0
            while position < ids.count {
                let end = min(position + engine.config.maxRows, ids.count)
                try engine.step((position..<end).map { EngineRow(slot: 0, token: ids[$0], position: $0, wantLogits: $0 == ids.count - 1) })
                position = end
            }
        }
        func divergence(_ a: [Float], _ b: UnsafeBufferPointer<Float>) -> Double {
            var maxA = -Float.infinity, maxB = -Float.infinity
            for token in 0..<limit { maxA = max(maxA, a[token]); maxB = max(maxB, b[token]) }
            var sumA = 0.0, sumB = 0.0
            for token in 0..<limit { sumA += Double(exp(a[token] - maxA)); sumB += Double(exp(b[token] - maxB)) }
            var kl = 0.0
            for token in 0..<limit {
                let pa = Double(exp(a[token] - maxA)) / sumA
                if pa > 1e-12 { kl += pa * (Double(a[token] - maxA) - log(sumA) - Double(b[token] - maxB) + log(sumB)) }
            }
            return kl
        }
        let worstBound = 0.1, meanBound = 0.001
        let broken = ProcessInfo.processInfo.environment["SPLOSH_SPECTEST_BREAK"] != nil
        var passed = true
        // Once for the draft model's own block and once for the longest block a session verifies.
        for width in [DraftModel.blockSize, Engine.journalRows] {
            // The reference goes through the same speculative step (only its first row is
            // kept), so both sides run the same kernels.
            try prefill()
            var plain: [Int] = [argmax(engine.logits(0), limit: limit)]
            var reference: [[Float]] = []
            while plain.count < count + width {
                let position = ids.count + plain.count - 1
                let block = [Int](repeating: plain.last!, count: width)
                try engine.step(block.enumerated().map { EngineRow(slot: 0, token: $1, position: position + $0, wantLogits: $0 == 0, verify: true) })
                reference.append(Array(engine.logits(0)[0..<limit]))
                plain.append(argmax(engine.logits(0), limit: limit))
                engine.acceptVerify(0, rows: 1, tokenCount: position + 1)
            }
            try prefill()
            var done = 1, start = ids.count, cycle = 0, flips = 0
            var worst = 0.0, total = 0.0, compared = 0
            let begun = Date()
            while done < count {
                var drafts = Array(plain[done..<done + width - 1])
                // Every position in turn, and every so often none: a block accepted whole.
                let corrupt = cycle % width
                if corrupt < width - 1 { drafts[corrupt] = (drafts[corrupt] + 1) % 1000 }
                let block = [plain[done - 1]] + drafts
                try engine.step(block.enumerated().map { EngineRow(slot: 0, token: $1, position: start + $0, wantLogits: true, verify: true) })
                let keep = min(corrupt, width - 1)
                for row in 0...keep {
                    let kl = divergence(reference[done - 1 + row], engine.logits(row))
                    worst = max(worst, kl); total += kl; compared += 1
                    if argmax(engine.logits(row), limit: limit) != plain[done + row] { flips += 1 }
                }
                // The deliberate fault: one accepted row's state is left out.
                engine.acceptVerify(0, rows: broken && keep > 0 ? keep : keep + 1, tokenCount: start + keep + 1)
                start += keep + 1; done += keep + 1; cycle += 1
            }
            let seconds = Date().timeIntervalSince(begun)
            let same = worst < worstBound && total / Double(max(compared, 1)) < meanBound
            SploshCLI.writeStderr(String(format: "speculation self-test, %d-row blocks: %@ — %d verify steps over %d positions; KL from the reference mean %.6f (bound %.3f), worst %.6f (bound %.1f); %d arg-max flips; %.1f ms/step\n",
                                         width, same ? "PASS" : "FAIL", cycle, compared, total / Double(max(compared, 1)), meanBound, worst, worstBound, flips, seconds / Double(max(cycle, 1)) * 1000))
            passed = passed && same
        }
        return passed ? ExitStatus.ok : 1
    }

    /// The DFlash 2 checkpoint where `splosh download` puts it, or in the Hugging Face cache, if present.
    static func defaultDraftURL() -> URL? {
        let draft = ((try? ModelLibrary.current()) ?? .builtIn).draft
        let name = draft.files.first?.name ?? "model.safetensors"
        let own = URL(fileURLWithPath: draft.directory, isDirectory: true).appendingPathComponent(name)
        if FileManager.default.fileExists(atPath: own.path) { return own }
        let snapshots = HubCache.directory().appendingPathComponent("models--" + draft.repo.replacingOccurrences(of: "/", with: "--"))
            .appendingPathComponent("snapshots")
        let names = (try? FileManager.default.contentsOfDirectory(atPath: snapshots.path)) ?? []
        return names.map { snapshots.appendingPathComponent($0).appendingPathComponent(name) }
            .first { FileManager.default.fileExists(atPath: $0.path) }
    }

    /// Greedy speculative decoding: draft a block, verify it in one target step, keep the
    /// agreeing prefix plus the target's own next token.
    static func speculativeGenerate(engine: Engine, draftURL: URL, ids: [Int], tokenizer: Tokenizer, count: Int, showIDs: Bool) throws -> Int32 {
        let loadStart = Date()
        let draft = try DraftModel(engine: engine, safetensorsURL: draftURL)
        SploshCLI.writeStderr(String(format: "draft loaded and quantised: %.2f GiB resident, in %.2fs\n",
                                     Double(draft.residentBytes) / 1_073_741_824, Date().timeIntervalSince(loadStart)))
        engine.resetSlot(0)
        draft.resetSlot(0)
        let prefillStart = Date()
        var position = 0
        while position < ids.count {
            let end = min(position + engine.config.maxRows, ids.count)
            try engine.step((position..<end).map { EngineRow(slot: 0, token: ids[$0], position: $0, wantLogits: $0 == ids.count - 1) }, captureFeatures: true)
            // Only positions still inside the draft's window when decoding starts are needed.
            let firstUseful = ids.count - DraftModel.ringCapacity + DraftModel.maxBlock
            try draft.pushContext((max(position, firstUseful)..<max(end, firstUseful)).map { (slot: 0, position: $0, engineRow: $0 - position) }.filter { $0.position < end })
            position = end
        }
        let prefillSeconds = Date().timeIntervalSince(prefillStart)
        var out: [Int] = [argmax(engine.logits(0), limit: tokenizer.idLimit)]
        var start = ids.count, steps = 0
        var draftSeconds = 0.0, verifySeconds = 0.0, contextSeconds = 0.0
        var random = SplitMix(seed: 1)
        let branchStudy = ProcessInfo.processInfo.environment["SPLOSH_BRANCH_STUDY"] != nil
        var missAt = [Int](repeating: 0, count: DraftModel.maxBlock)
        // A double block with an isolated first half, as the scheduler gives a lone session.
        // SPLOSH_DRAFT_BLOCK=8 keeps the checkpoint's own block; SPLOSH_DRAFT_ISOLATE=0 lets the
        // first half see the second.
        let blockLength = Int(ProcessInfo.processInfo.environment["SPLOSH_DRAFT_BLOCK"] ?? "") ?? DraftModel.maxBlock
        let isolate = ProcessInfo.processInfo.environment["SPLOSH_DRAFT_ISOLATE"] != "0" && blockLength > DraftModel.blockSize
        // SPLOSH_BLOCK_STUDY: at every step also draft the other block shapes from the same
        // anchor, and score them all against the finished output. Greedy decoding is lossless,
        // so the output does not depend on which shape the run itself followed.
        let blockStudy = ProcessInfo.processInfo.environment["SPLOSH_BLOCK_STUDY"] != nil
        let studyShapes: [(name: String, block: Int, isolate: Bool)] = [("8", 8, false), ("16", 16, false), ("8+8", 16, true)]
        var studied: [(at: Int, tokens: [[Int]])] = []
        var copies: [(at: Int, match: Int, tokens: [Int])] = []
        // Everything evaluated so far, for drafts copied from the text itself, and the record
        // that decides between copying and the draft model. SPLOSH_COPY_DRAFTS=0: never copy.
        var speculation = SpeculationState()
        speculation.lookup.reset(ids)
        let copyDrafts = ProcessInfo.processInfo.environment["SPLOSH_COPY_DRAFTS"] != "0"
        var copiedSteps = 0, copiedTokens = 0
        var rankOfTarget: [Int: Int] = [:]
        var topLogitWasRight = 0, selectorFixed = 0
        if ProcessInfo.processInfo.environment["SPLOSH_DRAFT_PROFILE"] != nil {
            // Cumulative GPU time of the draft pass truncated after each stage: the differences
            // are what each stage adds. Best of several passes, run back to back.
            func pass(_ limit: Int?) throws -> Double {
                draft.debugStageLimit = limit
                var best = Double.infinity
                for _ in 0..<8 {
                    let before = draft.gpuSeconds
                    _ = try draft.propose([DraftRequest(slot: 0, anchor: out[0], start: start, block: blockLength, isolatePrefix: isolate)],
                                          temperature: 0, vocabLimit: tokenizer.idLimit, random: &random)
                    best = min(best, draft.gpuSeconds - before)
                }
                return best * 1000
            }
            let whole = try pass(nil)
            var previous = 0.0, line = ""
            for limit in 0...draft.stageCount {
                let cumulative = try pass(limit)
                line += String(format: "%d:%.2f(+%.2f) ", limit, cumulative, cumulative - previous)
                previous = cumulative
            }
            SploshCLI.writeStderr(String(format: "draft pass %.2f ms over %d stages; cumulative (and added) ms after each stage:\n%@\n", whole, draft.stageCount, line))
            return ExitStatus.ok
        }
        var stopped = SpecialTokens.eosIDs.contains(out[0])
        let decodeStart = Date()
        while out.count < count, !stopped {
            var mark = Date()
            if blockStudy {
                if let copy = speculation.lookup.continuation(after: out.last!, count: DraftModel.maxBlock - 1) { copies.append((out.count, copy.match, copy.tokens)) }
                studied.append((out.count, try studyShapes.map {
                    try draft.propose([DraftRequest(slot: 0, anchor: out.last!, start: start, block: $0.block, isolatePrefix: $0.isolate)],
                                      temperature: 0, vocabLimit: tokenizer.idLimit, random: &random)[0].tokens
                }))
            }
            let proposal: DraftProposal
            if copyDrafts, let copy = speculation.copy(after: out.last!, count: blockLength - 1) {
                proposal = DraftProposal(copied: copy.tokens, match: copy.match, sampling: false)
            } else {
                proposal = try draft.propose([DraftRequest(slot: 0, anchor: out.last!, start: start, block: blockLength, isolatePrefix: isolate)],
                                             temperature: 0, vocabLimit: tokenizer.idLimit, random: &random)[0]
            }
            draftSeconds += Date().timeIntervalSince(mark); mark = Date()
            let block = [out.last!] + proposal.tokens
            try engine.step(block.enumerated().map { EngineRow(slot: 0, token: $1, position: start + $0, wantLogits: true, verify: true) }, captureFeatures: true)
            let posterior = (0..<block.count).map { argmax(engine.logits($0), limit: tokenizer.idLimit) }
            var keep = 0
            while keep < proposal.tokens.count, proposal.tokens[keep] == posterior[keep] { keep += 1 }
            if branchStudy {
                // Where the first miss falls, and whether the target's token was a lower-ranked
                // draft candidate there: the headroom for verifying a second branch.
                missAt[keep] += 1
                // Accepted positions where the selector's pick differs from the top logit: the
                // selector turned a miss into a hit there.
                for index in 0..<keep where proposal.candidates[index][0] != proposal.tokens[index] { selectorFixed += 1 }
                if keep < proposal.tokens.count {
                    let rank = proposal.candidates[keep].firstIndex(of: posterior[keep]) ?? -1
                    // Candidates are in logit order; the draft's pick is not necessarily first.
                    let picked = proposal.candidates[keep].firstIndex(of: proposal.tokens[keep]) ?? -1
                    rankOfTarget[rank < 0 ? 17 : rank, default: 0] += 1
                    if picked != 0 && rank == 0 { topLogitWasRight += 1 }
                }
            }
            engine.acceptVerify(0, rows: keep + 1, tokenCount: start + keep + 1)
            verifySeconds += Date().timeIntervalSince(mark); mark = Date()
            try draft.pushContext((0...keep).map { (slot: 0, position: start + $0, engineRow: $0) })
            contextSeconds += Date().timeIntervalSince(mark)
            speculation.record(agreed: keep, drafts: proposal.tokens.count, copyMatch: proposal.copyMatch)
            if proposal.copyMatch != nil { copiedSteps += 1; copiedTokens += keep + 1 }
            speculation.lookup.append(block[0])
            for token in proposal.tokens[0..<keep] { speculation.lookup.append(token) }
            for token in proposal.tokens[0..<keep] + [posterior[keep]] {
                if SpecialTokens.eosIDs.contains(token) { stopped = true; break }
                out.append(token)
            }
            start += keep + 1
            steps += 1
        }
        let decodeSeconds = Date().timeIntervalSince(decodeStart)
        if branchStudy {
            SploshCLI.writeStderr("first miss at draft index (last = all accepted): \(missAt)\n")
            SploshCLI.writeStderr("rank of the target's token among draft candidates at the miss (17 = absent): \(rankOfTarget.sorted { $0.key < $1.key })\n")
            SploshCLI.writeStderr("misses where the top-logit candidate was right but the selector moved off it: \(topLogitWasRight); accepted positions the selector moved onto the right token: \(selectorFixed)\n")
        }
        if blockStudy {
            // Tokens a step from each studied anchor would have produced: the agreeing prefix
            // plus the target's own next token. Anchors too close to the end are left out.
            var totals = [Int](repeating: 0, count: studyShapes.count), anchors = 0
            var wins = [Int](repeating: 0, count: studyShapes.count)
            for entry in studied where entry.at + DraftModel.maxBlock <= out.count {
                anchors += 1
                let yields = entry.tokens.map { tokens -> Int in
                    var keep = 0
                    while keep < tokens.count, tokens[keep] == out[entry.at + keep] { keep += 1 }
                    return keep + 1
                }
                for index in yields.indices { totals[index] += yields[index]; if yields[index] == yields.max() { wins[index] += 1 } }
            }
            let summary = studyShapes.indices.map { String(format: "%@: %.2f (best or tied %d)", studyShapes[$0].name, Double(totals[$0]) / Double(max(anchors, 1)), wins[$0]) }
            SploshCLI.writeStderr("block study, tokens per step over \(anchors) anchors — " + summary.joined(separator: "; ") + "\n")
            // Copied drafts, by how many tokens the occurrence shared with the text, against
            // the isolated double block from the same anchors.
            func yield(_ tokens: [Int], at: Int) -> Int {
                var keep = 0
                while keep < tokens.count, tokens[keep] == out[at + keep] { keep += 1 }
                return keep + 1
            }
            let double = Dictionary(uniqueKeysWithValues: studied.map { ($0.at, $0.tokens[2]) })
            for (label, range) in [("3", 3...3), ("4-5", 4...5), ("6-9", 6...9), ("10-19", 10...19), ("20+", 20...64)] {
                let bucket = copies.filter { range.contains($0.match) && $0.at + DraftModel.maxBlock <= out.count }
                guard !bucket.isEmpty else { continue }
                let copied = bucket.reduce(0) { $0 + yield($1.tokens, at: $1.at) }
                let drafted = bucket.reduce(0) { $0 + yield(double[$1.at]!, at: $1.at) }
                SploshCLI.writeStderr(String(format: "  copy, shared run %@: %d anchors, %.2f tokens per step copied, %.2f drafted\n", label, bucket.count,
                                             Double(copied) / Double(bucket.count), Double(drafted) / Double(bucket.count)))
            }
        }
        print(tokenizer.decode(out.filter { !SpecialTokens.eosIDs.contains($0) }))
        if showIDs { SploshCLI.writeStderr("generated ids: \(out)\n") }
        SploshCLI.writeStderr(String(format: "prefill: %d tokens in %.2fs (%.1f tok/s, including draft context)\n", ids.count, prefillSeconds, Double(ids.count) / prefillSeconds))
        if steps > 0 {
            if copiedSteps > 0 {
                SploshCLI.writeStderr(String(format: "copied: %d of %d steps drafted from the text itself, %.2f tokens/step\n", copiedSteps, steps, Double(copiedTokens) / Double(copiedSteps)))
            }
            SploshCLI.writeStderr(String(format: "decode:  %d tokens in %.2fs (%.2f tok/s) — %d verify steps, %.2f tokens/step; per step: draft %.1f ms (gpu %.1f, select %.1f of which top-k %.1f), verify %.1f ms, context %.1f ms\n",
                                         out.count - 1, decodeSeconds, Double(out.count - 1) / decodeSeconds, steps, Double(out.count - 1) / Double(steps),
                                         draftSeconds / Double(steps) * 1000, draft.gpuSeconds / Double(steps) * 1000, draft.selectSeconds / Double(steps) * 1000, draft.topKSeconds / Double(steps) * 1000,
                                         verifySeconds / Double(steps) * 1000, contextSeconds / Double(steps) * 1000))
        }
        return ExitStatus.ok
    }

    static func argmax(_ logits: UnsafeBufferPointer<Float>, limit: Int) -> Int {
        TopK.argmax(logits.baseAddress!, limit: min(limit, logits.count))
    }
}
