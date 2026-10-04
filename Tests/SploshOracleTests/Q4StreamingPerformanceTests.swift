import Foundation
import Metal
import Testing
import SploshModel
import SploshRuntime
import SploshCore

/// Measurement-only evidence for bounded Q4 file-backed range loading.
/// This deliberately does not dispatch a GEMM kernel or claim GPU streaming performance.
@Suite("Q4StreamingPerformanceTests")
struct Q4StreamingPerformanceTests {
    private let tensorName = "language_model.model.layers.0.linear_attn.in_proj_a.weight"
    private let warmups = 2
    private let repetitions = 10

    @Test("artifact range-loading performance evidence")
    func artifactRangeLoadingPerformance() throws {
        let raw = ProcessInfo.processInfo.environment["ARTIFACT"]
        let url = raw.map(URL.init(fileURLWithPath:)) ?? URL(fileURLWithPath: "models/q4/weights.splw", relativeTo: URL(fileURLWithPath: FileManager.default.currentDirectoryPath))
        let file = try WeightFile(splwURL: url)
        let record = try file.q4(tensorName)
        let device = try #require(MTLCreateSystemDefaultDevice())
        let runtime = try Q4GemmRuntime(device: device, metallib: Metallib(device: device))
        let streamer = Q4TensorStreamer(runtime: runtime, device: device)
        let rows = record.logicalShape[0]
        let boundary = min(64, rows - 1)
        let ranges = [0..<1, 1..<(1 + boundary)]
        var lines: [String] = []
        lines.append("artifact=\(url.path) tensor=\(tensorName) warmups=\(warmups) repetitions=\(repetitions) device=\(device.name)")
        for range in ranges {
            let plan = try Q4TensorStreamPlan(file: file, tensorName: tensorName, outputColumnRange: range, maximumResidentBytes: 1 << 30)
            for _ in 0..<warmups { _ = try streamer.loadChunk(file, plan: plan) }
            var samples: [Double] = []
            var loadedBytes = 0
            for _ in 0..<repetitions {
                let start = ContinuousClock.now
                let chunk = try streamer.loadChunk(file, plan: plan)
                let elapsed = start.duration(to: .now)
                let nanos = Double(elapsed.components.attoseconds) / 1e9 + Double(elapsed.components.seconds) * 1e9
                samples.append(nanos / 1e6)
                loadedBytes = chunk.packed.count + chunk.scales.count + chunk.biases.count
            }
            let total = samples.reduce(0, +)
            let mean = total / Double(samples.count)
            let throughput = Double(loadedBytes) / (mean / 1000.0) / 1_000_000.0
            let sampleText = samples.map { String(format: "%.6f", $0) }.joined(separator: ",")
            let meanText = String(format: "%.6f", mean)
            let throughputText = String(format: "%.6f", throughput)
            lines.append("range=\(range.lowerBound)..<\(range.upperBound) columns=\(range.count) chunkCount=1 residentBytes=\(plan.requiredResidentBytes) loadedBytes=\(loadedBytes) elapsedMsMean=\(meanText) throughputMBps=\(throughputText) rawSamplesMs=[\(sampleText)]")
        }
        print(lines.joined(separator: "\n"))
    }
}
