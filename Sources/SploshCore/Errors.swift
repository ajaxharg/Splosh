// Errors.swift — SploshCore.
//
// Owner: M0.1. Contract source: rev4 §4.1 (SploshError).

// `LocalizedError` and `errorDescription` are Foundation's; the engine's typed error conforms so a
// caller can print a remediation without re-deriving it from the case.
import Foundation

/// The typed error enum thrown across the engine (rev4 §4.1).
///
/// rev4 §4.1 defines four failures: a missing metallib function, a snapshot layout mismatch,
/// a no-trace admission failure, and a capability gate failure. The enum is `Error`-conforming
/// and descriptive; a recoverable failure never becomes a stringly-typed or `fatalError` path.
public enum SploshError: Error, Sendable, Equatable {
    /// A named compute function was not present in the loaded metallib.
    case missingMetallibFunction(String)
    /// The metallib resource could not be located, loaded, or parsed.
    case metallibUnavailable(String)
    /// A snapshot's layout did not match the layout it is being restored into.
    case snapshotLayoutMismatch(String)
    /// A cache admission failed and must leave no trace (rev4 §4.4, `StateCache`).
    case noTraceAdmissionFailure(String)
    /// The host GPU does not satisfy a capability the caller requires.
    case capabilityGateFailure(String)
}

extension SploshError: LocalizedError {
    public var errorDescription: String? {
        switch self {
        case .missingMetallibFunction(let name):
            return "metallib is missing compute function '\(name)'"
        case .metallibUnavailable(let detail):
            return "default.metallib unavailable: \(detail)"
        case .snapshotLayoutMismatch(let detail):
            return "snapshot layout mismatch: \(detail)"
        case .noTraceAdmissionFailure(let detail):
            return "cache admission failed: \(detail)"
        case .capabilityGateFailure(let detail):
            return "GPU capability gate failed: \(detail)"
        }
    }
}
