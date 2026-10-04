import Foundation

/// B=1 admission used before runtime-width batching is available.
/// A lease is exclusive, idempotently releasable, and releases on cancellation/deinit.
public actor SingleRequestScheduler {
    public final class Lease: @unchecked Sendable {
        private let lock = NSLock()
        private var released = false
        private let onRelease: @Sendable () -> Void

        fileprivate init(onRelease: @escaping @Sendable () -> Void) {
            self.onRelease = onRelease
        }

        public func release() {
            lock.lock()
            guard !released else { lock.unlock(); return }
            released = true
            lock.unlock()
            onRelease()
        }

        public func cancel() { release() }
        deinit { release() }
    }

    private var occupied = false
    public init() {}

    /// Attempts admission without waiting. A second in-flight request is refused.
    public func tryAdmit() -> Lease? {
        guard !occupied else { return nil }
        occupied = true
        return Lease { [weak self] in
            guard let self else { return }
            Task { await self.release() }
        }
    }

    /// Alias spelling useful to callers that model admission as a reservation.
    public func admit() -> Lease? { tryAdmit() }

    public func release() {
        occupied = false
    }

    public var isOccupied: Bool { occupied }
    public var activeCount: Int { occupied ? 1 : 0 }
}
