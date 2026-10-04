// Device.swift — SploshCore.
//
// Owner: M0.4. Contract source: rev4 §4.1 (Device), §2.1 (GPU family ids, working-set query).

import Metal

/// Wraps `MTLDevice` and owns the engine's capability gate (rev4 §4.1).
///
/// rev4 §4.1's invariant is *"single shared instance; nothing else creates an `MTLDevice`"*.
/// `Device.shared` is therefore the one creation site in the engine — and it is optional rather
/// than trapping, because a host with no Metal device is a capability failure to report
/// (`SploshError.capabilityGateFailure`) and not a crash.
public final class Device: @unchecked Sendable {
    /// The process-wide device, or `nil` when this host exposes no Metal device.
    public static let shared: Device? = Device()

    /// The wrapped Metal device.
    public let metal: MTLDevice

    /// Create the shared instance from the system default device.
    public convenience init?() {
        guard let metal = MTLCreateSystemDefaultDevice() else {
            return nil
        }
        self.init(metal: metal)
    }

    /// Wrap an already-created Metal device.
    public init(metal: MTLDevice) {
        self.metal = metal
    }

    /// `MTLDevice.name`, e.g. `Apple M5 Pro`.
    public var name: String {
        metal.name
    }

    /// `MTLDevice.registryID` — the device's stable identity, printed by M0.10's doctor.
    public var registryID: UInt64 {
        metal.registryID
    }

    /// `MTLDevice.recommendedMaxWorkingSetSize`.
    ///
    /// rev4 §2.1 reports **51 GiB** on the target machine and §4.4 makes it the budget
    /// `MemoryMonitor` never exceeds; this accessor is the query, not a constant.
    public var recommendedMaxWorkingSetSize: UInt64 {
        metal.recommendedMaxWorkingSetSize
    }

    /// `MTLDevice.maxThreadgroupMemoryLength`.
    ///
    /// `IMPLEMENTATION-PLAN.md` §4.2 requires >= **32,768** bytes for the required-through-M1 set.
    public var maxThreadgroupMemoryLength: Int {
        metal.maxThreadgroupMemoryLength
    }

    /// The capability gate (rev4 §4.1).
    ///
    /// Kernel code must ask this before claiming a family it needs. The failure this exists to
    /// prevent is two-sided: a family the kernels need returning `false`, *and* this returning
    /// `true` for a family the host does not have.
    public func supports(_ family: MTLGPUFamily) -> Bool {
        // `MTLDevice`'s own spelling is `supportsFamily(_:)` (`MTLDevice.h:882`); `supports(_:)` is
        // the name rev4 §4.1 gives this wrapper.
        metal.supportsFamily(family)
    }

    /// The concrete Metal 4 gate (rev4 §4.1).
    ///
    /// `MTLGPUFamilyMetal4` is **5002** (rev4 §2.1, `MTLDevice.h:255`). Spelled with the named case
    /// rather than the raw value so a renamed SDK constant is a compile error rather than a silent
    /// change of meaning; M0.4's test also asserts the raw-value form, so the number stays pinned.
    public var supportsMetal4: Bool {
        supports(.metal4)
    }
}
