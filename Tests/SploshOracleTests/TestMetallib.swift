import Foundation
import Metal
import SploshCore

/// Loads only the metallib named by the isolated gate environment.
/// Missing or invalid injection fails closed; it never falls back to default.metallib.
func makeTestMetallib(device: MTLDevice) throws -> Metallib {
    guard let raw = ProcessInfo.processInfo.environment["SPLOSH_TEST_METALLIB"], !raw.isEmpty else {
        throw SploshError.metallibUnavailable("SPLOSH_TEST_METALLIB is required for isolated oracle tests")
    }
    return try Metallib(device: device, metallibURL: URL(fileURLWithPath: raw))
}
