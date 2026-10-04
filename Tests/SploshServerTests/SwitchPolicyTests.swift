import Foundation
import Testing
@testable import SploshCLI
import SploshServer

@Suite("SwitchPolicyTests")
struct SwitchPolicyTests {
    private static let ready = Date(timeIntervalSinceReferenceDate: 1_000)
    private static func want(_ model: String, after seconds: TimeInterval, admin: Bool = false) -> SwitchWant {
        SwitchWant(model: model, since: ready.addingTimeInterval(seconds), admin: admin)
    }
    private static func decide(at seconds: TimeInterval, _ wants: [SwitchWant], idle: Bool = false, dwell: TimeInterval = 60) -> SwitchPolicy.Decision {
        SwitchPolicy.decide(now: ready.addingTimeInterval(seconds), readySince: ready, wants: wants, idle: idle, dwell: dwell)
    }

    @Test("with nothing kept for another model, the loaded one stays")
    func stays() {
        #expect(Self.decide(at: 500, []) == .stay)
        #expect(Self.decide(at: 500, [], idle: true) == .stay)
    }

    @Test("the loaded model keeps its turn for the dwell, however idle it is")
    func dwell() {
        let wants = [Self.want("b", after: 5)]
        #expect(Self.decide(at: 10, wants, idle: true) == .dwell(target: "b", until: Self.ready.addingTimeInterval(60)))
        #expect(Self.decide(at: 59.9, wants, idle: true) == .dwell(target: "b", until: Self.ready.addingTimeInterval(60)))
        #expect(Self.decide(at: 60, wants) == .drain(target: "b"))
        #expect(Self.decide(at: 10, wants, dwell: 0) == .drain(target: "b"))
    }

    @Test("after the dwell new requests are kept, and the engine is replaced only once it is idle")
    func drainsThenSwaps() {
        let wants = [Self.want("b", after: 5), Self.want("b", after: 30)]
        #expect(Self.decide(at: 70, wants) == .drain(target: "b"))
        #expect(Self.decide(at: 70, wants, idle: true) == .swap(target: "b"))
    }

    @Test("the oldest kept request's model is next; a third model takes the turn after")
    func oldestFirst() {
        let wants = [Self.want("c", after: 3), Self.want("b", after: 8), Self.want("c", after: 9)]
        #expect(Self.decide(at: 70, wants, idle: true) == .swap(target: "c"))
        // Once c is loaded its requests have gone to it, and b's waits out c's turn.
        #expect(Self.decide(at: 20, [Self.want("b", after: -62)], idle: true) == .dwell(target: "b", until: Self.ready.addingTimeInterval(60)))
    }

    @Test("a load asked for by hand goes first and does not wait for the dwell, but does for the engine to be idle")
    func byHand() {
        let wants = [Self.want("b", after: 1), Self.want("c", after: 2, admin: true)]
        #expect(Self.decide(at: 3, wants) == .drain(target: "c"))
        #expect(Self.decide(at: 3, wants, idle: true) == .swap(target: "c"))
    }

    @Test("a kept request has waited long enough at the wait, counted from its arrival")
    func expiry() {
        let want = Self.want("b", after: 0)
        #expect(!SwitchPolicy.expired(want, now: Self.ready.addingTimeInterval(119.9), wait: 120))
        #expect(SwitchPolicy.expired(want, now: Self.ready.addingTimeInterval(120), wait: 120))
        #expect(SwitchPolicy.expired(want, now: Self.ready, wait: 0))
    }

    @Test("the holder's own answers are whole responses with the error codes clients are told of")
    func refusals() throws {
        func parts(_ response: [UInt8]) throws -> (head: String, body: JSONValue) {
            let end = try #require(SwitchMarker.headEnd(response))
            return (String(decoding: response[..<end], as: UTF8.self), try JSONValue.parse(Array(response[(end + 4)...])))
        }
        let busy = try parts(SwitchRefusal.busy(model: "b", loaded: "a", waited: 120.4))
        #expect(busy.head.hasPrefix("HTTP/1.1 503 Service Unavailable\r\n"))
        #expect(busy.head.contains("\r\nRetry-After: 5"))
        #expect(busy.head.contains("\r\nConnection: close"))
        #expect(busy.head.contains("\r\nContent-Length: \(busy.body.compact.utf8.count)"))
        #expect(busy.body["error"]?["code"]?.stringValue == "model_busy")
        #expect(busy.body["error"]?["type"]?.stringValue == "server_error")
        #expect(busy.body["error"]?["message"]?.stringValue?.contains("120 s") == true)

        let failed = try parts(SwitchRefusal.loadFailed(model: "b", loaded: "a"))
        #expect(failed.head.hasPrefix("HTTP/1.1 503 "))
        #expect(!failed.head.contains("Retry-After"))
        #expect(failed.body["error"]?["code"]?.stringValue == "model_load_failed")

        let large = try parts(SwitchRefusal.tooLarge(model: "b", limit: 64 << 20))
        #expect(large.head.hasPrefix("HTTP/1.1 413 "))
        #expect(large.body["error"]?["code"]?.stringValue == "request_too_large")
        #expect(large.body["error"]?["message"]?.stringValue?.contains("64 MiB") == true)
        // None of them can be taken for the marker they stand in for.
        for response in [SwitchRefusal.busy(model: "b", loaded: nil, waited: 1), SwitchRefusal.loadFailed(model: "b", loaded: nil)] {
            #expect(SwitchMarker.inspect(response) == .answer)
        }
    }

    @Test("what the holder tells the engine of a switch reads back")
    func report() {
        let text = SwitchReport.text(target: "b", phase: "waiting", parked: 3)
        #expect(SwitchReport.read(text) == ModelCatalog.Switching(target: "b", phase: "waiting", parked: 3))
        #expect(SwitchReport.read(SwitchReport.text(target: nil, phase: nil, parked: 1)) == ModelCatalog.Switching(parked: 1))
        #expect(SwitchReport.read("") == nil)
        #expect(SwitchReport.read("{\"target\":") == nil)
    }
}
