// DeviceTests.swift — SploshOracleTests.
//
// Owner: M0.4. Contract source: rev4 §6 M0.4, §4.1 (Device), §2.1 (GPU family ids and the
// measured hardware facts).
//
// The suite is named after the file so `swift test --filter DeviceTests` -- the filter rev4 §3.2's
// inventory names and §6 M0.4's gate runs -- matches it.
//
// Failure mode this row exists to catch (rev4 §6 M0.4): a family the kernels need returns `false`,
// or `supports(_:)` returns `true` for a family the host does not have. `Device.supportsMetal4` is
// deliberately *not* the gate -- it cannot fail on an M5 Pro -- so the raw-value family assertions
// below carry the gate.

import Testing
import Metal

import SploshCore

@Suite("DeviceTests")
struct DeviceTests {
    /// The GPU family ids the plan's kernels require (rev4 §2.1, `MTLDevice.h:242,255`).
    private static let metal4Raw = 5002
    private static let apple10Raw = 1010

    @Test("the host has a Metal device and the plan's required GPU families")
    func requiredFamiliesAreSupported() throws {
        let device = try #require(Device.shared, "this host exposes no Metal device")

        // The two families §2.1 pins, asserted through their raw values so the *numbers* are gated
        // and not only the SDK's spelling of them.
        #expect(device.supports(MTLGPUFamily(rawValue: Self.metal4Raw)!) == true)
        #expect(device.supports(MTLGPUFamily(rawValue: Self.apple10Raw)!) == true)

        // Precondition only: rev4 §6 M0.4 says this "cannot fail on the target machine (an M5 Pro
        // reports Metal 4) and so gates nothing". It is kept so a divergence from the raw-value
        // assertion above -- i.e. a renumbered SDK constant -- shows up as a failure here.
        #expect(device.supportsMetal4)
    }

    @Test("device facts the plan budgets against (rev4 §2.1, §4.2)")
    func deviceFactsMatchThePlannedBudget() throws {
        let device = try #require(Device.shared, "this host exposes no Metal device")

        let report = """
            GPU report (rev4 §2.1, §4.2):
              name                        = \(device.name)
              registryID                  = \(device.registryID)
              macOS                       = \(ProcessInfo.processInfo.operatingSystemVersionString)
              maxThreadgroupMemoryLength  = \(device.maxThreadgroupMemoryLength) bytes \
            (required >= 32768)
              recommendedMaxWorkingSetSize= \(device.recommendedMaxWorkingSetSize) bytes
            """
        print(report)

        // IMPLEMENTATION-PLAN.md §4.2's required-through-M1 set: threadgroup memory >= 32 KiB.
        #expect(device.maxThreadgroupMemoryLength >= 32_768)

        // The working-set budget must be non-zero, or every §4.4 MemoryMonitor calculation divides
        // into nothing. The 51 GiB figure is an observation on the target machine, not asserted
        // here: this test must stay true on any host that runs it.
        #expect(device.recommendedMaxWorkingSetSize > 0)

        #expect(!device.name.isEmpty)
        #expect(device.registryID != 0)
    }

    @Test("the capability gate is a real answer, not a constant true")
    func capabilityGateDiscriminates() throws {
        let device = try #require(Device.shared, "this host exposes no Metal device")

        // The gate must be *evaluated*, not stubbed. `MTLGPUFamily(rawValue:)` is failable, so an
        // undefined id cannot be constructed; instead assert that at least one family the plan does
        // not require is rejected, which a `true`-returning stub could not do.
        let candidates: [MTLGPUFamily] = [
            MTLGPUFamily(rawValue: 1001)!,  // Apple1
            MTLGPUFamily(rawValue: 2002)!,  // Mac2 (deprecated on macOS 27)
            MTLGPUFamily(rawValue: 4002)!,  // MacCatalyst2
        ]
        let answers = candidates.map { device.supports($0) }
        print("capability gate sample: \(zip(candidates.map(\.rawValue), answers).map { "\($0)=\($1)" }.joined(separator: " "))")

        #expect(answers.contains(false), "supports(_:) returned true for every family tried")
    }
}
