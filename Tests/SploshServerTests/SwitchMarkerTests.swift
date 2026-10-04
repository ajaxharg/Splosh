import Foundation
import Testing
@testable import SploshCLI

@Suite("SwitchMarkerTests")
struct SwitchMarkerTests {
    private static func inspect(_ text: String) -> SwitchMarker { SwitchMarker.inspect(Array(text.utf8)) }
    private static let marker = "HTTP/1.1 421 Misdirected Request\r\nContent-Type: application/json\r\nx-splosh-switch: uq5\r\nConnection: close\r\nContent-Length: 2\r\n\r\n{}"

    @Test("the marker is the status and the header, and names the model wanted")
    func wanted() {
        #expect(Self.inspect(Self.marker) == .wanted(model: "uq5", admin: false))
        #expect(Self.inspect("HTTP/1.1 421 Misdirected Request\r\nX-Splosh-Switch:mq4 \r\nx-splosh-switch-admin: 1\r\n\r\n") == .wanted(model: "mq4", admin: true))
        #expect(Self.inspect("HTTP/1.0 421 Misdirected Request\r\nx-splosh-switch: a.b_c-d\r\n\r\n") == .wanted(model: "a.b_c-d", admin: false))
    }

    @Test("any other answer is the client's from its first bytes")
    func answers() {
        #expect(Self.inspect("HTTP/1.1 200 OK\r\n") == .answer)
        #expect(Self.inspect("HTTP/1.1 2") == .answer)
        #expect(Self.inspect("HTTP/1.1 404 Not Found\r\nx-splosh-switch: uq5\r\n\r\n") == .answer)
        #expect(Self.inspect("data: {\"content\":\"HTTP/1.1 421\"}\n\n") == .answer)
        // The status alone is not the marker, nor is the header with nothing in it.
        #expect(Self.inspect("HTTP/1.1 421 Misdirected Request\r\nContent-Length: 0\r\n\r\n") == .answer)
        #expect(Self.inspect("HTTP/1.1 421 Misdirected Request\r\nx-splosh-switch: \r\n\r\n") == .answer)
        // A header's name in a body is not a header.
        #expect(Self.inspect("HTTP/1.1 421 Misdirected Request\r\nContent-Length: 24\r\n\r\nx-splosh-switch: uq5\r\n\r\n") == .answer)
    }

    @Test("an answer that may yet be the marker is not passed on until its head is whole")
    func undecided() {
        let bytes = Array(Self.marker.utf8)
        let headLength = Self.marker.range(of: "\r\n\r\n")!.upperBound.utf16Offset(in: Self.marker)
        for count in 0..<headLength {
            #expect(SwitchMarker.inspect(Array(bytes[..<count])) == .undecided, "\(count) bytes")
        }
        #expect(SwitchMarker.inspect(Array(bytes[..<headLength])) == .wanted(model: "uq5", admin: false))
        // A head that never ends is given up on, and passed on as it is.
        let endless = Array(("HTTP/1.1 421 Misdirected Request\r\n" + String(repeating: "x-pad: 0123456789\r\n", count: 1000)).utf8)
        #expect(SwitchMarker.inspect(Array(endless[..<1000])) == .undecided)
        #expect(SwitchMarker.inspect(endless) == .answer)
    }

    @Test("what the engine has in flight is read from its whole response and nothing less")
    func busy() {
        let response = "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 24\r\n\r\n{\"busy\":2,\"model\":\"uq5\"}"
        #expect(EngineBusy.parse(Array(response.utf8)) == EngineBusy(count: 2, model: "uq5"))
        #expect(EngineBusy.parse(Array("HTTP/1.1 200 OK\r\n\r\n{\"busy\":0}".utf8)) == EngineBusy(count: 0, model: nil))
        #expect(EngineBusy.parse(Array(response.utf8.dropLast(3))) == nil)
        #expect(EngineBusy.parse(Array("HTTP/1.1 200 OK\r\nContent-Length: 24\r\n".utf8)) == nil)
        #expect(EngineBusy.parse(Array("HTTP/1.1 404 Not Found\r\n\r\n{\"busy\":0}".utf8)) == nil)
        #expect(EngineBusy.parse(Array("HTTP/1.1 200 OK\r\n\r\n{\"error\":\"no\"}".utf8)) == nil)
        #expect(EngineBusy.parse([]) == nil)
    }
}
