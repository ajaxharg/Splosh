// ModelSwitch.swift — when the engine is replaced by one with another model.
//
// An engine holds one model for as long as it lives. A request that names a registered model
// the engine does not have is answered by it with a marker (see ModelCatalog), which the process
// holding the port takes for itself: it keeps the request, and when the loaded model has had
// its turn and its engine has nothing in flight, stops that engine and starts one with the
// model wanted (ServeSupervisor). The engine's exit is what frees the memory: two models do
// not fit.
//
// This file is the parts of that with no sockets in them: what an engine's first bytes of
// answer say, what is to be done about the requests kept, the answers the holder gives when a
// model cannot be had, and what it tells the engine of a switch for `GET /v1/models`.

import Foundation
import SploshServer

/// A request kept by the holder until the model it names is the one loaded.
struct SwitchWant: Equatable {
    let model: String
    /// When its connection arrived.
    let since: Date
    /// Asked for by hand (`splosh models --load`): not held back for the loaded model's turn.
    let admin: Bool
}

/// What the first bytes of an engine's answer are.
enum SwitchMarker: Equatable {
    /// An answer, for the client.
    case answer
    /// Not enough of it yet to tell.
    case undecided
    /// The engine does not have the model the request names: none of this is the client's.
    case wanted(model: String, admin: Bool)

    /// As much of an answer as is looked at for its head to end.
    static let headLimit = 16 << 10
    private static let statusLine = Array("HTTP/1.1 \(ModelCatalog.switchStatus.code) ".utf8)

    /// Where the head of an HTTP message ends: the offset of its blank line.
    static func headEnd(_ bytes: [UInt8]) -> Int? {
        guard bytes.count >= 4 else { return nil }
        return (0...(bytes.count - 4)).first { bytes[$0] == 13 && bytes[$0 + 1] == 10 && bytes[$0 + 2] == 13 && bytes[$0 + 3] == 10 }
    }

    static func inspect(_ head: [UInt8]) -> SwitchMarker {
        // Any answer but the marker's status is known at its first byte that differs (the
        // minor version, at 7, may be either).
        for (index, byte) in head.prefix(statusLine.count).enumerated() where index != 7 && byte != statusLine[index] { return .answer }
        guard head.count >= statusLine.count, let end = headEnd(head) else { return head.count < headLimit ? .undecided : .answer }
        var model: String?, admin = false
        for line in String(decoding: head[..<end], as: UTF8.self).components(separatedBy: "\r\n").dropFirst() {
            guard let colon = line.firstIndex(of: ":") else { continue }
            let name = line[..<colon].lowercased()
            if name == ModelCatalog.switchHeader.canonicalName {
                model = line[line.index(after: colon)...].trimmingCharacters(in: .whitespaces)
            } else if name == ModelCatalog.switchAdminHeader.canonicalName {
                admin = true
            }
        }
        guard let model, !model.isEmpty else { return .answer }
        return .wanted(model: model, admin: admin)
    }
}

enum SwitchPolicy {
    enum Decision: Equatable {
        /// Nothing kept wants another model.
        case stay
        /// The loaded model keeps its turn until `until`; then `target`'s begins.
        case dwell(target: String, until: Date)
        /// New requests are kept, and the engine is waited on to finish the ones it has.
        case drain(target: String)
        /// The engine has nothing in flight: it is stopped, and one with `target` started.
        case swap(target: String)
    }

    /// Whether a kept request has waited as long as one may. It is refused then: its client
    /// gives up on a request that has had no answer for five minutes, and a save and a load
    /// have still to fit in that.
    static func expired(_ want: SwitchWant, now: Date, wait: TimeInterval) -> Bool {
        now.timeIntervalSince(want.since) >= wait
    }

    /// What is to be done for `wants`: the requests kept, in order of arrival, none of them
    /// for the loaded model or expired. The oldest's model is next, after the loaded model
    /// has had `dwell` seconds from `readySince` (so that clients of two models each get a
    /// turn); one asked for by hand goes first and does not wait for that. `idle` is whether
    /// the engine has nothing in flight with new requests already being kept.
    static func decide(now: Date, readySince: Date, wants: [SwitchWant], idle: Bool, dwell: TimeInterval) -> Decision {
        guard let want = wants.first(where: \.admin) ?? wants.first else { return .stay }
        let until = readySince.addingTimeInterval(dwell)
        if !want.admin, now < until { return .dwell(target: want.model, until: until) }
        return idle ? .swap(target: want.model) : .drain(target: want.model)
    }
}

/// The answers the holder gives itself, as whole HTTP responses: the engine that would give
/// them is the one that cannot be had.
enum SwitchRefusal {
    static func response(status: Int, reason: String, type: String = "server_error", code: String, message: String, retryAfter: Int? = nil) -> [UInt8] {
        let body = JSONValue.object([("error", .object([("message", .string(message)), ("type", .string(type)),
                                                        ("param", .null), ("code", .string(code))]))]).compact
        var head = "HTTP/1.1 \(status) \(reason)\r\nContent-Type: application/json\r\nContent-Length: \(body.utf8.count)\r\nConnection: close\r\n"
        if let retryAfter { head += "Retry-After: \(retryAfter)\r\n" }
        return Array((head + "\r\n" + body).utf8)
    }

    /// The loaded model's requests did not finish in the time a kept request may wait.
    static func busy(model: String, loaded: String?, waited: TimeInterval) -> [UInt8] {
        response(status: 503, reason: "Service Unavailable", code: "model_busy",
                 message: "the model \(model) could not be loaded in \(Int(waited.rounded())) s: \(loaded ?? "the loaded model") is still answering requests; send the request again",
                 retryAfter: 5)
    }

    /// The model did not load; the one before it is being started again.
    static func loadFailed(model: String, loaded: String?) -> [UInt8] {
        response(status: 503, reason: "Service Unavailable", code: "model_load_failed",
                 message: "the model \(model) did not load; \(loaded ?? "the model before it") is being loaded again in its place")
    }

    /// The request is more than the holder keeps, so it cannot be given to another engine.
    static func tooLarge(model: String, limit: Int) -> [UInt8] {
        response(status: 413, reason: "Content Too Large", type: "invalid_request_error", code: "request_too_large",
                 message: "the model \(model) is not loaded, and a request of more than \(limit >> 20) MiB cannot be kept while it is")
    }
}

/// What an engine says it has in flight (`GET /v1/busy`), and the model it has loaded.
struct EngineBusy: Equatable {
    let count: Int
    let model: String?

    /// From the engine's whole response; nil for anything but a 200 with the count in it.
    static func parse(_ response: [UInt8]) -> EngineBusy? {
        let status = Array("HTTP/1.1 200 ".utf8)
        guard response.count > status.count, let end = SwitchMarker.headEnd(response),
              !response.prefix(status.count).enumerated().contains(where: { $0.offset != 7 && $0.element != status[$0.offset] }),
              let body = try? JSONValue.parse(Array(response[(end + 4)...])), let count = body["busy"]?.intValue else { return nil }
        return EngineBusy(count: count, model: body["model"]?.stringValue)
    }
}

/// How a switch stands, as the holder writes it down for the engine to pass on: the engine
/// answers `GET /v1/models`, and only the holder knows what it is keeping.
enum SwitchReport {
    static func text(target: String?, phase: String?, parked: Int) -> String {
        JSONValue.object([("target", target.map(JSONValue.string) ?? .null), ("phase", phase.map(JSONValue.string) ?? .null),
                          ("parked", .int(parked))]).compact + "\n"
    }

    static func read(_ text: String) -> ModelCatalog.Switching? {
        guard let value = try? JSONValue.parse(text), value.objectValue != nil else { return nil }
        return ModelCatalog.Switching(target: value["target"]?.stringValue, phase: value["phase"]?.stringValue, parked: value["parked"]?.intValue ?? 0)
    }
}
