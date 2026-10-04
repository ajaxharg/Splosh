// CommandGraph.swift — one command buffer per forward pass.
//
// A decode step issues ~870 dispatches (64 layers x ~13, plus the head). Submitting each one
// with `waitUntilCompleted` would spend more time in round trips than in arithmetic, so every
// dispatch for a step is encoded into a single command buffer and committed once.
//
// The encoder stays open across dispatches. Metal orders dispatches within one compute encoder
// according to their data dependencies only when they are in separate encoders; within a single
// encoder, concurrent dispatch is opt-in. We use the default (serial) ordering because the
// forward pass is a dependency chain, and serial ordering is exactly the semantics we need.

import Foundation
@preconcurrency import Metal

public enum CommandGraphError: Error, CustomStringConvertible {
    case encoderUnavailable
    case executionFailed(String)

    public var description: String {
        switch self {
        case .encoderUnavailable: return "unable to create a Metal compute encoder"
        case .executionFailed(let reason): return "GPU execution failed: \(reason)"
        }
    }
}

/// Accumulates dispatches into one command buffer and commits them together.
public final class CommandGraph {
    public let queue: MTLCommandQueue
    private var command: MTLCommandBuffer?
    private var encoder: MTLComputeCommandEncoder?
    /// Buffers of the current pass already handed to the GPU by `flush()`.
    private var submitted: [MTLCommandBuffer] = []
    private var concurrentPass = false
    private(set) public var dispatchCount: Int = 0
    /// When set, every dispatch and barrier of a pass is noted here (kernel, grid, threads):
    /// what a step consists of, for looking at.
    public var trace: [String]?
    private static let nameLock = NSLock()
    nonisolated(unsafe) private static var names: [ObjectIdentifier: String] = [:]
    public static func name(_ pipeline: MTLComputePipelineState, _ name: String) {
        nameLock.lock(); names[ObjectIdentifier(pipeline)] = name; nameLock.unlock()
    }
    private func note(_ pipeline: MTLComputePipelineState, _ groups: Int, _ threads: Int) {
        guard trace != nil else { return }
        Self.nameLock.lock(); let name = Self.names[ObjectIdentifier(pipeline)] ?? "?"; Self.nameLock.unlock()
        trace?.append("\(name) \(groups)x\(threads)")
    }
    /// Diagnostics for the last committed pass, in seconds of system uptime: when it began,
    /// when each of its command buffers was committed, and when the GPU started and finished it.
    private(set) public var lastPass: (began: Double, buffers: [(committed: Double, gpuStart: Double, gpuEnd: Double)]) = (0, [])
    private var began = 0.0
    private var commitTimes: [Double] = []

    public init(queue: MTLCommandQueue) { self.queue = queue }

    /// `concurrent` lets the GPU overlap dispatches; the caller must then call `barrier()`
    /// between a dispatch and any later dispatch that reads what it wrote.
    public func begin(label: String, concurrent: Bool = false) throws {
        precondition(command == nil, "CommandGraph.begin called while a pass was open")
        guard let buffer = queue.makeCommandBuffer(),
              let compute = buffer.makeComputeCommandEncoder(dispatchType: concurrent ? .concurrent : .serial) else {
            throw CommandGraphError.encoderUnavailable
        }
        buffer.label = label
        command = buffer
        encoder = compute
        dispatchCount = 0
        submitted.removeAll(keepingCapacity: true)
        concurrentPass = concurrent
        began = ProcessInfo.processInfo.systemUptime
        commitTimes.removeAll(keepingCapacity: true)
    }

    /// Hand what has been encoded so far to the GPU and keep encoding into a new command buffer.
    /// Buffers on one queue run in order, so this is also a dependency barrier. It lets the GPU
    /// start on the first layers while the CPU is still encoding the rest of the pass.
    public func flush() throws {
        guard let current = command, let open = encoder else { preconditionFailure("flush outside a pass") }
        open.endEncoding()
        current.commit()
        commitTimes.append(ProcessInfo.processInfo.systemUptime)
        submitted.append(current)
        guard let buffer = queue.makeCommandBuffer(),
              let compute = buffer.makeComputeCommandEncoder(dispatchType: concurrentPass ? .concurrent : .serial) else {
            throw CommandGraphError.encoderUnavailable
        }
        buffer.label = current.label
        command = buffer
        encoder = compute
    }

    /// Encode one dispatch. `body` binds buffers and sets bytes; the grid is supplied here so
    /// every call site is forced to state its geometry.
    public func dispatch(_ pipeline: MTLComputePipelineState,
                         grid: MTLSize,
                         threadsPerGroup: MTLSize,
                         label: String? = nil,
                         _ body: (MTLComputeCommandEncoder) -> Void) {
        guard let encoder else { preconditionFailure("dispatch outside a pass") }
        if let label { encoder.label = label }
        encoder.setComputePipelineState(pipeline)
        body(encoder)
        encoder.dispatchThreadgroups(grid, threadsPerThreadgroup: threadsPerGroup)
        note(pipeline, grid.width * grid.height * grid.depth, threadsPerGroup.width * threadsPerGroup.height * threadsPerGroup.depth)
        dispatchCount += 1
    }

    /// End the current encoder and open a new one on the same command buffer.
    public func splitEncoder(concurrent: Bool) throws {
        guard let command else { preconditionFailure("splitEncoder outside a pass") }
        encoder?.endEncoding()
        guard let next = command.makeComputeCommandEncoder(dispatchType: concurrent ? .concurrent : .serial) else { throw CommandGraphError.encoderUnavailable }
        encoder = next
    }

    /// Order everything encoded so far before everything encoded after.
    public func barrier() {
        trace?.append("--")
        encoder?.memoryBarrier(scope: .buffers)
    }

    /// Encode a dispatch over an exact thread grid (no threadgroup-level reduction).
    public func dispatchThreads(_ pipeline: MTLComputePipelineState,
                                threads: MTLSize,
                                _ body: (MTLComputeCommandEncoder) -> Void) {
        guard let encoder else { preconditionFailure("dispatch outside a pass") }
        encoder.setComputePipelineState(pipeline)
        body(encoder)
        let width = min(threads.width, pipeline.threadExecutionWidth)
        let height = min(threads.height, max(1, pipeline.maxTotalThreadsPerThreadgroup / width))
        encoder.dispatchThreads(threads, threadsPerThreadgroup: MTLSize(width: width, height: height, depth: 1))
        note(pipeline, (threads.width * threads.height * threads.depth + width * height - 1) / (width * height), width * height)
        dispatchCount += 1
    }

    /// Commit and block until the GPU finishes. Returns the GPU-reported execution time.
    @discardableResult
    public func commitAndWait() throws -> Double {
        guard let command, let encoder else { preconditionFailure("commitAndWait outside a pass") }
        encoder.endEncoding()
        command.commit()
        commitTimes.append(ProcessInfo.processInfo.systemUptime)
        command.waitUntilCompleted()
        self.encoder = nil
        self.command = nil
        let buffers = submitted + [command]
        lastPass = (began, zip(commitTimes, buffers).map { ($0, $1.gpuStartTime, $1.gpuEndTime) })
        submitted.removeAll(keepingCapacity: true)
        for buffer in buffers {
            if let error = buffer.error { throw CommandGraphError.executionFailed(String(describing: error)) }
        }
        return command.gpuEndTime - buffers[0].gpuStartTime
    }

    /// Commit without blocking, invoking `completion` when the GPU finishes.
    public func commit(completion: @escaping @Sendable (Result<Double, Error>) -> Void) {
        guard let command, let encoder else { preconditionFailure("commit outside a pass") }
        encoder.endEncoding()
        command.addCompletedHandler { buffer in
            if let error = buffer.error {
                completion(.failure(CommandGraphError.executionFailed(String(describing: error))))
            } else {
                completion(.success(buffer.gpuEndTime - buffer.gpuStartTime))
            }
        }
        command.commit()
        self.encoder = nil
        self.command = nil
    }
}
