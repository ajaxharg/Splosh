// Metallib.swift — SploshCore.
//
// Owner: M0.2. Contract source: rev4 §4.1 (Metallib), §3.3 (the `Bundle.module` resource route).

import Foundation
import Metal

/// Loads `default.metallib` and hands out cached compute pipeline states (rev4 §4.1).
///
/// The resource is a build product of `make shaders`, declared with `.copy` in `Package.swift`, and
/// therefore found through `Bundle.module`. rev4 §3.3's ordering trap applies: the Makefile must run
/// *before* SwiftPM lays out resources, so `make shaders` precedes `swift build`.
///
/// `functionNames` is the thin wrapper over `MTLLibrary.functionNames` that M0.2's exact-set kernel
/// check reads. Every kernel-introducing milestone extends the set that check asserts (rev4 §3.2).
///
/// Marked `@unchecked Sendable`: the two Metal references are immutable for the object's lifetime and
/// the pipeline cache is guarded by `lock`.
public final class Metallib: @unchecked Sendable {
    /// The resource basename; `Package.swift` copies `Resources/default.metallib`.
    public static let resourceName = "default"
    /// The resource extension.
    public static let resourceExtension = "metallib"

    private let device: MTLDevice
    private let library: MTLLibrary
    private let lock = NSLock()
    private var pipelines: [String: MTLComputePipelineState] = [:]

    /// Load `default.metallib` from the package's own resource bundle.
    ///
    /// - Throws: `SploshError.metallibUnavailable` when the resource is absent or cannot be parsed.
    public convenience init(device: MTLDevice) throws {
        try self.init(device: device, bundle: .module)
    }

    /// Load a metallib from an explicit URL. This is intended for isolated tests only;
    /// production callers should use `init(device:)` so Bundle.module remains authoritative.
    ///
    /// - Throws: `SploshError.metallibUnavailable` when the URL is absent or cannot be parsed.
    public convenience init(device: MTLDevice, metallibURL: URL) throws {
        guard FileManager.default.fileExists(atPath: metallibURL.path) else {
            throw SploshError.metallibUnavailable("no metallib at \(metallibURL.path)")
        }
        do {
            try self.init(device: device, library: device.makeLibrary(URL: metallibURL))
        } catch let error as SploshError {
            throw error
        } catch {
            throw SploshError.metallibUnavailable("\(metallibURL.path): \(error)")
        }
    }

    private init(device: MTLDevice, library: MTLLibrary) {
        self.device = device
        self.library = library
    }

    /// Load `default.metallib` from an explicit bundle.
    ///
    /// This is a second initializer rather than a `bundle: Bundle = .module` default on one, because
    /// SwiftPM generates `Bundle.module` as `internal` and a `public` declaration may not reference
    /// an internal symbol from a default argument value.
    ///
    /// - Throws: `SploshError.metallibUnavailable` when the resource is absent or cannot be parsed.
    ///   A recoverable failure never becomes a `fatalError` (rev4 §4.1).
    public init(device: MTLDevice, bundle: Bundle) throws {
        guard
            let url = bundle.url(
                forResource: Self.resourceName,
                withExtension: Self.resourceExtension
            )
        else {
            throw SploshError.metallibUnavailable(
                "no \(Self.resourceName).\(Self.resourceExtension) in bundle "
                    + "\(bundle.bundlePath); run `make shaders` before `swift build`"
            )
        }
        do {
            self.library = try device.makeLibrary(URL: url)
        } catch {
            throw SploshError.metallibUnavailable("\(url.path): \(error)")
        }
        self.device = device
    }

    /// The names of the compute functions the library exports (rev4 §4.1).
    ///
    /// Exposed in `MTLLibrary`'s own order; callers that assert a set must compare as a set.
    public var functionNames: [String] {
        library.functionNames
    }

    /// The compute pipeline state for `name`, built once and cached.
    ///
    /// - Throws: `SploshError.missingMetallibFunction` when the library has no such function.
    public func pipeline(_ name: String) throws -> MTLComputePipelineState {
        lock.lock()
        defer { lock.unlock() }

        if let cached = pipelines[name] {
            return cached
        }
        guard let function = library.makeFunction(name: name) else {
            throw SploshError.missingMetallibFunction(name)
        }
        do {
            let pipeline = try device.makeComputePipelineState(function: function)
            pipelines[name] = pipeline
            CommandGraph.name(pipeline, name)
            return pipeline
        } catch {
            throw SploshError.metallibUnavailable("compute pipeline '\(name)': \(error)")
        }
    }
}
