import Foundation
import Metal
import SploshCore
import SploshModel
import SploshQuant

/// Runtime-facing submission API for the exported `gemm_q4` pipeline.
/// Conversion and transport are deliberately outside this type.
public struct Q4LoadedBuffers: @unchecked Sendable {
    public let packed: MTLBuffer
    public let scales: MTLBuffer
    public let biases: MTLBuffer
    public let layout: Q4BufferLayout
    public let logicalShape: [Int]

    public init(packed: MTLBuffer, scales: MTLBuffer, biases: MTLBuffer,
                layout: Q4BufferLayout, logicalShape: [Int]) {
        self.packed = packed; self.scales = scales; self.biases = biases
        self.layout = layout; self.logicalShape = logicalShape
    }
}

public struct DenseBF16Buffer: @unchecked Sendable {
    public let buffer: MTLBuffer
    public let shape: [Int]
    public init(buffer: MTLBuffer, shape: [Int]) { self.buffer = buffer; self.shape = shape }
}

public final class Q4GemmRuntime: @unchecked Sendable {
    private let device: MTLDevice
    private let metallib: Metallib
    private let pipeline: MTLComputePipelineState

    public init(device: MTLDevice, metallib: Metallib? = nil) throws {
        self.device = device
        self.metallib = try metallib ?? Metallib(device: device)
        self.pipeline = try self.metallib.pipeline("gemm_q4")
    }

    /// Uploads a validated q4 triplet directly to GPU buffers. Packed weights remain packed.
    public func loadQ4(_ file: WeightFile, name: String) throws -> Q4LoadedBuffers {
        let record = try file.q4(name)
        guard record.logicalShape.count == 2, record.physicalShape.count == 2 else {
            throw TensorInventoryError.mlxAffineInvalid("q4 shape must be rank-2")
        }
        guard file.header.alignment.pointerAlignmentBytes == Q4BufferLayout.defaultRowAlignment,
              file.header.alignment.rowStrideBytes == Q4BufferLayout.defaultRowAlignment else {
            throw TensorInventoryError.mlxAffineInvalid("unsupported q4 alignment")
        }
        let rows = record.logicalShape[0], inner = record.logicalShape[1]
        guard rows > 0, inner > 0, inner <= Int.max - 63 else {
            throw TensorInventoryError.mlxAffineInvalid("q4 dimensions overflow or are non-positive")
        }
        guard let containerRecord = file.header.tensorRecords.first(where: { $0.name == name }), containerRecord.rowStride <= UInt64(Int.max) else {
            throw TensorInventoryError.mlxAffineInvalid("missing q4 row stride metadata")
        }
        let layout = try Q4BufferLayout(rows: rows, logicalK: inner, rowStrideBytes: Int(containerRecord.rowStride))
        let packedData = try file.data(for: record.weight)
        let scalesData = try file.data(for: record.scales)
        let biasesData = try file.data(for: record.biases)
        guard layout.packedWordCount <= Int.max / MemoryLayout<UInt32>.stride,
              layout.sidecarCount <= Int.max / MemoryLayout<UInt16>.stride else {
            throw TensorInventoryError.mlxAffineInvalid("q4 byte count overflow")
        }
        let expectedPacked = layout.packedWordCount * MemoryLayout<UInt32>.stride
        let expectedSidecar = layout.sidecarCount * MemoryLayout<UInt16>.stride
        guard packedData.count == expectedPacked, scalesData.count == expectedSidecar, biasesData.count == expectedSidecar else {
            throw TensorInventoryError.mlxAffineInvalid("q4 payload size does not match aligned layout")
        }
        let packed = packedData.withUnsafeBytes { device.makeBuffer(bytes: $0.baseAddress!, length: packedData.count, options: .storageModeShared) }
        let scales = scalesData.withUnsafeBytes { device.makeBuffer(bytes: $0.baseAddress!, length: scalesData.count, options: .storageModeShared) }
        let biases = biasesData.withUnsafeBytes { device.makeBuffer(bytes: $0.baseAddress!, length: biasesData.count, options: .storageModeShared) }
        guard let packed, let scales, let biases else {
            throw SploshError.capabilityGateFailure("unable to allocate q4 buffers")
        }
        return Q4LoadedBuffers(packed: packed, scales: scales, biases: biases, layout: layout, logicalShape: record.logicalShape)
    }

    /// Uploads a dense BF16 component without converting or densifying q4 payloads.
    public func loadDenseBF16(_ file: WeightFile, name: String, expectedShape: [Int]? = nil) throws -> DenseBF16Buffer {
        let record = try file.denseBF16(name, expectedShape: expectedShape)
        let data = try file.data(for: record.tensor)
        let expected = record.tensor.shape.reduce(1, *) * MemoryLayout<UInt16>.stride
        guard data.count == expected else { throw TensorInventoryError.boundsViolation(name) }
        let buffer = data.withUnsafeBytes { device.makeBuffer(bytes: $0.baseAddress!, length: data.count, options: .storageModeShared) }
        guard let buffer else {
            throw SploshError.capabilityGateFailure("unable to allocate BF16 buffer")
        }
        return DenseBF16Buffer(buffer: buffer, shape: record.tensor.shape)
    }

    /// Encodes one q4 GEMM and dispatches one thread per output element.
    /// `a` is bf16 [M,K], q4 is physically [N,rowStrideWords], and C is Float32 [M,N].
    public func dispatch(
        shape: GemmQ4Shape,
        layout: Q4BufferLayout,
        a: MTLBuffer,
        packed: MTLBuffer,
        scales: MTLBuffer,
        biases: MTLBuffer,
        output: MTLBuffer,
        commandQueue: MTLCommandQueue
    ) throws {
        try shape.validate(layout: layout)
        guard shape.rows <= Int.max / shape.inner,
              shape.rows * shape.inner <= Int.max / MemoryLayout<UInt16>.stride,
              layout.packedWordCount <= Int.max / MemoryLayout<UInt32>.stride,
              layout.sidecarCount <= Int.max / MemoryLayout<UInt16>.stride,
              shape.rows <= Int.max / shape.columns,
              shape.rows * shape.columns <= Int.max / MemoryLayout<Float>.stride else {
            throw GemmQ4Error.invalidShape
        }
        let expectedA = shape.rows * shape.inner * MemoryLayout<UInt16>.stride
        let expectedPacked = layout.packedWordCount * MemoryLayout<UInt32>.stride
        let expectedSidecar = layout.sidecarCount * MemoryLayout<UInt16>.stride
        let expectedOutput = shape.rows * shape.columns * MemoryLayout<Float>.stride
        try require(a, atLeast: expectedA, name: "a")
        try require(packed, atLeast: expectedPacked, name: "packed")
        try require(scales, atLeast: expectedSidecar, name: "scales")
        try require(biases, atLeast: expectedSidecar, name: "biases")
        try require(output, atLeast: expectedOutput, name: "output")
        guard let command = commandQueue.makeCommandBuffer(),
              let encoder = command.makeComputeCommandEncoder() else {
            throw SploshError.capabilityGateFailure("unable to create q4 command encoder")
        }
        struct Parameters { var rows: UInt32; var columns: UInt32; var inner: UInt32; var q4RowStrideWords: UInt32; var orientation: UInt32; var groupsPerRow: UInt32 }
        var params = Parameters(rows: UInt32(shape.rows), columns: UInt32(shape.columns), inner: UInt32(shape.inner), q4RowStrideWords: UInt32(layout.rowStrideWords), orientation: layout.orientation == .rowsByK ? 0 : 1, groupsPerRow: UInt32(layout.groupsPerRow))
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(a, offset: 0, index: 0)
        encoder.setBuffer(packed, offset: 0, index: 1)
        encoder.setBuffer(scales, offset: 0, index: 2)
        encoder.setBuffer(biases, offset: 0, index: 3)
        encoder.setBuffer(output, offset: 0, index: 4)
        encoder.setBytes(&params, length: MemoryLayout<Parameters>.stride, index: 5)
        let grid = MTLSize(width: shape.columns, height: shape.rows, depth: 1)
        let width = min(shape.columns, max(1, pipeline.threadExecutionWidth))
        let height = min(shape.rows, max(1, pipeline.maxTotalThreadsPerThreadgroup / width))
        encoder.dispatchThreads(grid, threadsPerThreadgroup: MTLSize(width: width, height: height, depth: 1))
        encoder.endEncoding()
        command.commit()
        command.waitUntilCompleted()
        if let error = command.error {
            throw SploshError.capabilityGateFailure("gemm_q4 command failed: \(error)")
        }
    }

    private func makeBuffer(from data: Data) -> MTLBuffer? {
        data.withUnsafeBytes { bytes in
            guard let baseAddress = bytes.baseAddress else { return nil }
            return device.makeBuffer(bytes: baseAddress, length: data.count, options: .storageModeShared)
        }
    }

    private func require(_ buffer: MTLBuffer, atLeast expected: Int, name: String) throws {
        guard buffer.length >= expected else { throw GemmQ4Error.bufferTooSmall(name: name, expected: expected, observed: buffer.length) }
    }
}
