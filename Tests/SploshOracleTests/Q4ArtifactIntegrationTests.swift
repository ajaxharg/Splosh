import Foundation
import Metal
import Testing

import SploshCore
import SploshModel
import SploshOracle
import SploshQuant
import SploshRuntime

/// Artifact-backed Q4 integration gate. This deliberately exercises only one confirmed tensor;
/// it is not a whole-model runner or a substitute for BF16 acquisition.
@Suite("Q4ArtifactIntegrationTests")
struct Q4ArtifactIntegrationTests {
    private static let tensorName = "language_model.model.layers.0.linear_attn.in_proj_a.weight"
    private static let logicalRows = 48
    private static let logicalK = 5120
    private static let rows = 2

    private func artifactURL() -> URL {
        if let raw = ProcessInfo.processInfo.environment["ARTIFACT"], !raw.isEmpty {
            return URL(fileURLWithPath: raw)
        }
        return URL(fileURLWithPath: ".build/q4/weights.splw", relativeTo: URL(fileURLWithPath: FileManager.default.currentDirectoryPath))
    }

    private func bf16Activations() -> [UInt16] {
        (0..<(Self.rows * Self.logicalK)).map { index in
            let value = (Float((index % 17) - 8) * 0.03125) + (index % 5 == 0 ? 0.125 : 0)
            return Q4BufferLayout.bf16Bits(value)
        }
    }

    private func words(_ data: Data) -> [UInt32] {
        data.withUnsafeBytes { raw in Array(raw.bindMemory(to: UInt32.self)) }
    }

    private func halfWords(_ data: Data) -> [UInt16] {
        data.withUnsafeBytes { raw in Array(raw.bindMemory(to: UInt16.self)) }
    }

    @Test("confirmed artifact tensor dispatch matches decoded Q4 oracle")
    func confirmedTensorMatchesOracle() throws {
        let url = artifactURL()
        guard FileManager.default.fileExists(atPath: url.path) else {
            print("SKIP Q4 artifact integration: missing artifact at \(url.path)")
            return
        }
        guard let device = MTLCreateSystemDefaultDevice() else {
            print("SKIP Q4 artifact integration: no Metal device on host")
            return
        }

        let file = try WeightFile(splwURL: url)
        let record = try file.q4(Self.tensorName)
        #expect(record.logicalShape == [Self.logicalRows, Self.logicalK])
        #expect(record.physicalShape == [Self.logicalRows, 640])
        let runtime = try Q4GemmRuntime(device: device, metallib: Metallib(device: device))
        let loaded = try runtime.loadQ4(file, name: Self.tensorName)
        #expect(loaded.logicalShape == [Self.logicalRows, Self.logicalK])
        #expect(loaded.layout.rowStrideBytes == 2560)

        let packedWords = words(try file.data(for: record.weight))
        let scales = halfWords(try file.data(for: record.scales))
        let biases = halfWords(try file.data(for: record.biases))
        let activations = bf16Activations()

        // Decode each packed word through the public layout contract, then transpose [N,K]
        // into the [K,N] BF16 operand consumed by GemmOracle.
        var b = Array(repeating: UInt16.zero, count: Self.logicalK * Self.logicalRows)
        for column in 0..<Self.logicalRows {
            for k in 0..<Self.logicalK {
                let decoded = try loaded.layout.decode(row: column, k: k, words: packedWords,
                                                       scalesBF16: scales, biasesBF16: biases)
                b[k * Self.logicalRows + column] = Q4BufferLayout.bf16Bits(decoded)
            }
        }
        let expected = GemmOracle.bf16(a: activations, b: b,
                                       rows: Self.rows, columns: Self.logicalRows, inner: Self.logicalK)

        guard let a = device.makeBuffer(bytes: activations, length: activations.count * 2, options: .storageModeShared),
              let output = device.makeBuffer(length: expected.count * MemoryLayout<Float>.stride, options: .storageModeShared),
              let queue = device.makeCommandQueue() else {
            print("SKIP Q4 artifact integration: Metal buffer or command queue allocation unavailable")
            return
        }
        let started = DispatchTime.now().uptimeNanoseconds
        try runtime.dispatch(shape: try GemmQ4Shape(rows: Self.rows, columns: Self.logicalRows, inner: Self.logicalK),
                             layout: loaded.layout, a: a, packed: loaded.packed, scales: loaded.scales,
                             biases: loaded.biases, output: output, commandQueue: queue)
        let elapsedMS = Double(DispatchTime.now().uptimeNanoseconds - started) / 1_000_000
        let actual = Array(UnsafeBufferPointer(start: output.contents().bindMemory(to: Float.self, capacity: expected.count), count: expected.count))
        let maxAbs = zip(actual, expected).map { abs($0 - $1) }.max() ?? 0
        let maxRel = zip(actual, expected).map { pair -> Float in abs(pair.0 - pair.1) / max(abs(pair.1), 1e-3) }.max() ?? 0
        print("Q4 artifact tensor=\(Self.tensorName) shape=M\(Self.rows)×N\(Self.logicalRows)×K\(Self.logicalK) maxAbs=\(maxAbs) maxRel=\(maxRel) tolerance=0.10 elapsedMs=\(elapsedMS) (q4 dequantized reference vs GPU fp32; not BF16 parity)")
        let allFinite = actual.allSatisfy { $0.isFinite }
        #expect(allFinite, "q4 runtime produced non-finite output")
        #expect(maxRel <= 0.10, "max relative error \(maxRel) exceeds documented q4 tolerance 0.10")
    }

    @Test("missing artifact is reported without fabrication")
    func missingArtifactFailsClosed() {
        let missing = URL(fileURLWithPath: "/definitely/missing/weights.splw")
        #expect(throws: WeightFileError.missing(missing)) { try WeightFile(splwURL: missing) }
    }

    @Test("unknown tensor is rejected by the artifact API")
    func unknownTensorFailsClosed() throws {
        let url = artifactURL()
        guard FileManager.default.fileExists(atPath: url.path) else { return }
        let file = try WeightFile(splwURL: url)
        #expect(throws: TensorInventoryError.missingTensor("q4.integration.unknown")) {
            try file.q4("q4.integration.unknown")
        }
    }

    @Test("activation shape mismatch is rejected before dispatch")
    func activationShapeFailsClosed() throws {
        let shape = try GemmQ4Shape(rows: Self.rows, columns: Self.logicalRows, inner: Self.logicalK)
        let layout = try Q4BufferLayout(rows: Self.logicalRows, logicalK: Self.logicalK)
        #expect(throws: GemmQ4Error.layoutMismatch) {
            try shape.validate(layout: try Q4BufferLayout(rows: Self.logicalRows, logicalK: Self.logicalK - 1))
        }
        #expect(layout.logicalK == Self.logicalK)
    }
}
