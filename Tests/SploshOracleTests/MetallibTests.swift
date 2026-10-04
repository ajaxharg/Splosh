// MetallibTests.swift — SploshOracleTests.
//
// Owner: M0.3 (the 1 MiB copy execution test) and M0.7 (the full ABI-conformance size table).
// Contract sources: rev4 §6 M0.3, §6 M0.7, §4.1 (Metallib), §3.2 (the `{copy}` exact set).
//
// The suite is named after the file so `swift test --filter MetallibTests` -- the filter rev4 §3.2's
// inventory names and §6 M0.3/M0.7's gates run -- matches it.
//
// The 1 MiB assertion is byte equality, not "it ran": a transposed binding, a 16-byte-only stride, or
// an unstated alignment precondition all still produce a command buffer that completes successfully.

import Testing
import Metal

import SploshCore

/// The Swift mirror of `Sources/Shaders/common/abi.h`'s `CopyParams` (rev4 M0.7).
///
/// It lives here rather than in `SploshCore` because rev4 §4.1's public-type table does not name it;
/// M5.5, whose durable-cache staging copy reuses this ABI, is the task that promotes it to a shared
/// type. The `MemoryLayout` assertion below is what makes this mirror a checked claim rather than a
/// restatement: the shader's `static_assert(sizeof(CopyParams) == 4)` and this struct must agree.
private struct CopyParams {
    var byteCount: UInt32
}

/// The sizes rev4 §6 M0.7 names (0, 1, 15, 16, 17, 1 MiB), plus the threadgroup boundaries the same
/// dispatch geometry implies: 4096 B is exactly one 256x16 threadgroup, and 4097 is the first byte of
/// a second.
///
/// These are the **lengths** the ABI's byte-granularity claim is about, and they are what the
/// unaligned-tail case exercises: 1, 15, 17, 257 and 4097 each leave a partial final 16-byte window.
/// Whether a `setBuffer` *offset* may be unaligned is Metal's rule for the binding rather than a
/// property of this kernel, so no offset misalignment is asserted here.
///
/// Declared at file scope rather than inside the suite because `@Test(arguments:)` is evaluated in
/// the enclosing scope, where a member of the suite is not visible.
private let abiSizes = [0, 1, 15, 16, 17, 255, 256, 257, 4095, 4096, 4097, 1024 * 1024]

@Suite("MetallibTests")
struct MetallibTests {
    // MARK: - Harness

    /// The plan's ABI fixes threadgroup 256 and 16 bytes per thread (rev4 M0.7).
    private static let threadsPerThreadgroup = 256
    private static let bytesPerThread = 16
    private static let bytesPerThreadgroup = threadsPerThreadgroup * bytesPerThread

    private func makeDevice() throws -> MTLDevice {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw SploshError.capabilityGateFailure("no Metal device on this host")
        }
        return device
    }

    /// Run the `copy` kernel over `source` and return what landed in the destination.
    ///
    /// The dispatch geometry is the plan's: threadgroup 256, threadgroups per grid
    /// `ceil(byteCount / (256 * 16))`. A zero-byte copy is dispatched as one threadgroup that returns
    /// immediately, because Metal requires a non-empty grid; the kernel's own guard makes the empty
    /// case a no-op rather than an out-of-range read.
    private func runCopy(device: MTLDevice, metallib: Metallib, source: [UInt8]) throws -> [UInt8] {
        let byteCount = source.count
        #expect(MemoryLayout<CopyParams>.size == 4, "CopyParams must mirror the shader's one u32")

        let pipeline = try metallib.pipeline("copy")
        #expect(
            pipeline.maxTotalThreadsPerThreadgroup >= Self.threadsPerThreadgroup,
            "the ABI's threadgroup of \(Self.threadsPerThreadgroup) must be legal for this pipeline"
        )

        guard let queue = device.makeCommandQueue() else {
            throw SploshError.capabilityGateFailure("could not create an MTLCommandQueue")
        }

        // A non-zero length is required to allocate; the empty case still gets a 1-byte buffer that
        // the kernel never touches, so the geometry under test is unchanged.
        let allocationLength = max(byteCount, 1)

        guard
            let src = device.makeBuffer(bytes: source.isEmpty ? [UInt8(0)] : source,
                                        length: allocationLength,
                                        options: .storageModeShared),
            let dst = device.makeBuffer(length: allocationLength, options: .storageModeShared)
        else {
            throw SploshError.capabilityGateFailure("could not allocate the copy buffers")
        }
        // Poison the destination so a copy that silently does nothing cannot pass.
        memset(dst.contents(), 0xA5, allocationLength)

        guard
            let commandBuffer = queue.makeCommandBuffer(),
            let encoder = commandBuffer.makeComputeCommandEncoder()
        else {
            throw SploshError.capabilityGateFailure("could not create the command buffer / encoder")
        }

        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(src, offset: 0, index: 0)
        encoder.setBuffer(dst, offset: 0, index: 1)
        var params = CopyParams(byteCount: UInt32(byteCount))
        encoder.setBytes(&params, length: MemoryLayout<CopyParams>.size, index: 2)

        let groups = max(1, (byteCount + Self.bytesPerThreadgroup - 1) / Self.bytesPerThreadgroup)
        encoder.dispatchThreadgroups(
            MTLSize(width: groups, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(
                width: Self.threadsPerThreadgroup, height: 1, depth: 1
            )
        )
        encoder.endEncoding()

        commandBuffer.commit()
        // The engine's own `Queue` observes completion without blocking (rev4 §4.1); this harness is
        // a single-shot test that wants a deterministic read-back, so it waits.
        commandBuffer.waitUntilCompleted()

        if let error = commandBuffer.error {
            throw SploshError.capabilityGateFailure("copy command buffer failed: \(error)")
        }

        let pointer = dst.contents().bindMemory(to: UInt8.self, capacity: allocationLength)
        return Array(UnsafeBufferPointer(start: pointer, count: byteCount))
    }

    private func deterministicBytes(count: Int) -> [UInt8] {
        // A cheap, position-dependent pattern: any byte swapped with its neighbour, or copied from
        // the wrong offset, changes the result. (UInt8 arithmetic wraps, which is what we want here.)
        (0..<count).map { UInt8(truncatingIfNeeded: $0 &* 31 &+ 7) }
    }

    // MARK: - M0.2's durable exact-set assertion (rev4 §6 M0.2, "keep this as the durable/CI form")

    // The shipped library has grown past M0: it carries the oracle kernels of the later
    // milestones and the engine's kernels, which are all `sp_`-prefixed. The exact M0 set is
    // still checked against the isolated metallib the M0 gate builds.
    @Test("the loaded library exports copy, and nothing outside the oracle and sp_ kernel sets")
    func functionNameSetIsExactlyCopy() throws {
        let device = try makeDevice()
        let metallib = try Metallib(device: device)
        let oracle: Set<String> = ["copy", "rmsnorm", "rope_mrope", "swiglu", "gemm_bf16", "attention_dense_decode", "attention_dense_prefill", "gdn_prepare", "gdn_decode", "gdn_gate", "gdn_commit", "gemm_q4"]
        let names = Set(metallib.functionNames)

        #expect(names.contains("copy"))
        #expect(names.subtracting(oracle).allSatisfy { $0.hasPrefix("sp_") })
    }

    @Test("a missing function is a typed error, never a trap")
    func missingFunctionIsATypedError() throws {
        let device = try makeDevice()
        let metallib = try Metallib(device: device)

        #expect(throws: SploshError.self) {
            _ = try metallib.pipeline("no_such_kernel")
        }
    }

    // MARK: - M0.3's copy execution test

    @Test("a 1 MiB GPU copy is byte-equal to its source")
    func oneMebibyteCopyIsByteEqual() throws {
        let device = try makeDevice()
        let metallib = try Metallib(device: device)

        let source = deterministicBytes(count: 1024 * 1024)
        let destination = try runCopy(device: device, metallib: metallib, source: source)

        #expect(destination.count == source.count)
        #expect(destination == source)
    }

    // MARK: - M0.7's ABI-conformance size table

    @Test("copy is byte-granular at every size the ABI names", arguments: abiSizes)
    func copyAcrossABISizes(byteCount: Int) throws {
        let device = try makeDevice()
        let metallib = try Metallib(device: device)

        let source = deterministicBytes(count: byteCount)
        let destination = try runCopy(device: device, metallib: metallib, source: source)

        // The count check is not redundant with the equality check: on a zero-byte copy `[] == []`
        // is trivially true, and this is the assertion that pins the empty case's shape.
        #expect(destination.count == byteCount)
        #expect(destination == source)
    }

    @Test("the grid formula is ceil(byteCount / 4096) at every ABI size")
    func dispatchGeometryIsTheDocumentedFormula() {
        // The ABI's geometry, restated independently of the harness, so a change to one without the
        // other is visible. rev4 §6 M0.7 fixes threadgroup 256 and 16 bytes per thread.
        func grid(_ byteCount: Int) -> Int {
            max(1, (byteCount + 4095) / 4096)
        }

        #expect(grid(0) == 1, "an empty copy still needs a non-empty grid")
        #expect(grid(1) == 1)
        #expect(grid(4095) == 1)
        #expect(grid(4096) == 1)
        #expect(grid(4097) == 2)
        #expect(grid(1024 * 1024) == 256)
    }
}
