import Foundation
import Testing
@testable import SploshServer

/// A tool call is sent as it is written: its name, then its arguments in pieces. The pieces,
/// joined, must be the arguments the parser made when it read a call whole, however the
/// model's text happens to be cut into tokens.
@Suite("Tool call stream")
struct ToolCallStreamTests {
    private static let tools: [JSONValue] = (try? JSONValue.parse("""
    [
      {"type": "function", "function": {"name": "write_file", "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}}}},
      {"type": "function", "function": {"name": "bash", "parameters": {"type": "object", "properties": {"command": {"type": "string"}, "timeout": {"type": "integer"}, "background": {"type": "boolean"}, "ratio": {"type": "number"}}}}},
      {"type": "function", "function": {"name": "edit_file", "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "edits": {"type": "array"}, "options": {"type": "object"}}}}},
      {"type": "function", "function": {"name": "noargs", "parameters": {"type": "object", "properties": {}}}}
    ]
    """).arrayValue) ?? []

    private static func types(_ name: String) -> [String: String] {
        for tool in tools where tool["function"]?["name"]?.stringValue == name {
            var types: [String: String] = [:]
            for member in tool["function"]?["parameters"]?["properties"]?.objectValue ?? [] { types[member.key] = member.value["type"]?.stringValue }
            return types
        }
        return [:]
    }

    /// The arguments as they were made from a call read whole (the parser before it streamed).
    private static func whole(_ body: String) -> (name: String, arguments: String)? {
        guard let open = body.range(of: "<function="), let nameEnd = body.range(of: ">", range: open.upperBound..<body.endIndex) else { return nil }
        let name = String(body[open.upperBound..<nameEnd.lowerBound]).trimmingCharacters(in: .whitespacesAndNewlines)
        guard !name.isEmpty else { return nil }
        let types = types(name)
        var members: [(key: String, value: JSONValue)] = []
        var cursor = nameEnd.upperBound
        while let parameter = body.range(of: "<parameter=", range: cursor..<body.endIndex),
              let keyEnd = body.range(of: ">", range: parameter.upperBound..<body.endIndex),
              let close = body.range(of: "</parameter>", range: keyEnd.upperBound..<body.endIndex) {
            let key = String(body[parameter.upperBound..<keyEnd.lowerBound]).trimmingCharacters(in: .whitespacesAndNewlines)
            var value = Substring(body[keyEnd.upperBound..<close.lowerBound])
            if value.hasPrefix("\n") { value.removeFirst() }
            if value.hasSuffix("\n") { value.removeLast() }
            let raw = String(value), type = types[key]
            if type == "string" { members.append((key, .string(raw))) }
            else if let parsed = try? JSONValue.parse(raw) {
                if type != nil { members.append((key, parsed)) }
                else if case .string = parsed { members.append((key, .string(raw))) }
                else { members.append((key, parsed)) }
            } else { members.append((key, .string(raw))) }
            cursor = close.upperBound
        }
        return (name, JSONValue.object(members).compact)
    }

    private struct Parsed: Equatable {
        var reasoning = "", content = ""
        var calls: [Call] = []
        var whole = 0
        var unfinished = false
        var pieces = 0, empty = 0
        struct Call: Equatable { var name: String; var arguments: String }
    }

    private static func parse(_ chunks: [[UInt8]], startsInThinking: Bool = false) -> Parsed {
        var parser = ChatOutputParser(startsInThinking: startsInThinking, tools: tools)
        var events: [ChatOutputParser.Event] = []
        for chunk in chunks { events += parser.push(chunk) }
        events += parser.finish()
        var parsed = Parsed()
        for event in events {
            switch event {
            case .reasoning(let text): parsed.reasoning += text
            case .content(let text): parsed.content += text
            case .toolCallStart(let index, _, let name):
                #expect(index == parsed.calls.count)
                parsed.calls.append(.init(name: name, arguments: ""))
            case .toolCallArguments(let index, let piece):
                parsed.calls[index].arguments += piece
                parsed.pieces += 1
                if piece.isEmpty { parsed.empty += 1 }
            case .toolCallEnd(let index): #expect(index == parsed.whole); parsed.whole += 1
            }
        }
        #expect(parsed.whole == parser.toolCallCount)
        parsed.unfinished = parser.unfinishedToolCall
        return parsed
    }

    /// The same text cut up in ways tokens might cut it: whole, by character, by byte (which
    /// splits characters), and in runs of a few bytes.
    private static func cuts(_ text: String) -> [[[UInt8]]] {
        let bytes = Array(text.utf8)
        var cuts: [[[UInt8]]] = [[bytes], text.map { Array(String($0).utf8) }, bytes.map { [$0] }]
        for size in [2, 3, 5, 7, 11] {
            cuts.append(stride(from: 0, to: bytes.count, by: size).map { Array(bytes[$0..<min($0 + size, bytes.count)]) })
        }
        return cuts
    }

    private static func call(_ body: String) -> String { "<tool_call>" + body + "</tool_call>" }

    /// Every cut of `text` gives the same reply: `expected`, counts of pieces aside. (And space
    /// at the end of content aside: before a call it is dropped only if the call's opening
    /// tag came with it, and the template trims it either way.)
    private func expect(_ text: String, _ expected: Parsed, startsInThinking: Bool = false, _ comment: Comment) {
        for (number, chunks) in Self.cuts(text).enumerated() {
            var got = Self.parse(chunks, startsInThinking: startsInThinking)
            got.pieces = 0; got.empty = 0
            while got.content.last?.isWhitespace == true { got.content.removeLast() }
            #expect(got == expected, "\(comment), cut \(number)")
        }
    }

    private static let source = "func a() -> String {\n    let s = \"tab\\there\"\t// café ☕️ 👩‍💻\n\n    return s  \n}\n\n// </p> </parameterless> <parameter> </ <\n"

    @Test("string values go out in pieces that join into the value, whatever is in it")
    func strings() {
        let body = "\n<function=write_file>\n<parameter=path>\nSources/a b.swift\n</parameter>\n<parameter=content>\n" + Self.source + "\n</parameter>\n</function>\n"
        let wanted = Self.whole(body)!
        #expect((try? JSONValue.parse(wanted.arguments))?["content"]?.stringValue == Self.source)
        expect(Self.call(body), Parsed(calls: [.init(name: "write_file", arguments: wanted.arguments)], whole: 1), "a file written")
        // By character, the value is under way long before it is whole.
        let streamed = Self.parse(Self.cuts(Self.call(body))[1])
        #expect(streamed.pieces > 50)
    }

    @Test("values that are not strings are read whole and written as JSON")
    func typed() {
        let bodies = [
            "\n<function=bash>\n<parameter=command>\nsleep 5\n</parameter>\n<parameter=timeout>\n30\n</parameter>\n<parameter=background>\ntrue\n</parameter>\n<parameter=ratio>\n1.50\n</parameter>\n</function>\n",
            "\n<function=edit_file>\n<parameter=path>\na.swift\n</parameter>\n<parameter=edits>\n[{\"old\": \"let a = 1\", \"new\": \"let a = 2\\n\"}, {\"old\": \"é\", \"new\": \"e\"}]\n</parameter>\n<parameter=options>\n{\n  \"dry_run\": false, \"limit\": null\n}\n</parameter>\n</function>\n",
            "\n<function=bash>\n<parameter=background>\nTrue\n</parameter>\n<parameter=command>\nls\n</parameter>\n</function>\n",        // not JSON: kept as the text it is
            "\n<function=edit_file>\n<parameter=edits>\n[]\n</parameter>\n<parameter=path>\na.swift\n</parameter>\n</function>\n",       // a string after a value read whole
        ]
        for body in bodies {
            let wanted = Self.whole(body)!
            expect(Self.call(body), Parsed(calls: [.init(name: wanted.name, arguments: wanted.arguments)], whole: 1), "\(wanted.name)")
        }
    }

    @Test("a tool or parameter that was not declared is read whole, as before")
    func undeclared() {
        let body = "\n<function=mystery>\n<parameter=count>\n3\n</parameter>\n<parameter=note>\nthree of them\n</parameter>\n<parameter=quoted>\n\"x\"\n</parameter>\n</function>\n"
        let wanted = Self.whole(body)!
        #expect(wanted.arguments == #"{"count":3,"note":"three of them","quoted":"\"x\""}"#)
        expect(Self.call(body), Parsed(calls: [.init(name: "mystery", arguments: wanted.arguments)], whole: 1), "mystery")
    }

    @Test("empty values, values that are only newlines, and a call with no parameters")
    func empties() {
        for value in ["", "\n", "\n\n", "\n\n\n", " ", "x", "\nx", "x\n", "\n\nx\n\n"] {
            let body = "\n<function=write_file>\n<parameter=path>" + value + "</parameter>\n<parameter=content>" + value + "</parameter>\n</function>\n"
            let wanted = Self.whole(body)!
            expect(Self.call(body), Parsed(calls: [.init(name: "write_file", arguments: wanted.arguments)], whole: 1), "value \(value.debugDescription)")
        }
        expect(Self.call("\n<function=noargs>\n</function>\n"), Parsed(calls: [.init(name: "noargs", arguments: "{}")], whole: 1), "no parameters")
    }

    @Test("thinking, content and two calls in one reply")
    func wholeReply() {
        let first = "\n<function=bash>\n<parameter=command>\ngrep -rn \"chooseSlot\" Sources/ | head -20\n</parameter>\n</function>\n"
        let second = "\n<function=write_file>\n<parameter=path>\na.txt\n</parameter>\n<parameter=content>\none\ntwo\n</parameter>\n</function>\n"
        let text = "Let me look.\n</think>\n\nI'll search first.\n\n" + Self.call(first) + "\n" + Self.call(second)
        expect(text, Parsed(reasoning: "Let me look.\n", content: "I'll search first.",
                            calls: [.init(name: "bash", arguments: Self.whole(first)!.arguments), .init(name: "write_file", arguments: Self.whole(second)!.arguments)],
                            whole: 2), startsInThinking: true, "two calls")
    }

    @Test("what is not a call is content, as before")
    func notACall() {
        expect("See:" + Self.call("\nno function here\n") + " there.", Parsed(content: "See:<tool_call>\nno function here\n</tool_call>there."), "no function")
        expect(Self.call("\n<function=>\n</function>\n"), Parsed(content: "<tool_call>\n<function=>\n</function>\n</tool_call>"), "no name")
        expect("<tool_call>\n<func", Parsed(content: "<tool_call>\n<func"), "cut before its name")
    }

    @Test("a reply that ends inside a call leaves it unfinished, with what was sent a beginning of it")
    func cutShort() {
        let body = "\n<function=write_file>\n<parameter=path>\na.txt\n</parameter>\n<parameter=content>\none\ntwo\nthree\n</parameter>\n</function>\n"
        let wanted = Self.whole(body)!.arguments
        let text = Self.call(body)
        for length in [40, 60, 75, 85, 95] {
            let cut = String(text.prefix(length))
            for chunks in Self.cuts(cut) {
                let got = Self.parse(chunks)
                #expect(got.unfinished && got.whole == 0 && got.content.isEmpty, "cut at \(length)")
                #expect(got.calls.count == 1 && wanted.hasPrefix(got.calls[0].arguments), "cut at \(length): \(got.calls.first?.arguments ?? "")")
            }
        }
        // The function block whole and only the closing tag missing: the call is whole.
        let closed = String(text.dropLast("</tool_call>".count))
        expect(closed, Parsed(calls: [.init(name: "write_file", arguments: wanted)], whole: 1), "no closing tag")
    }

    @Test("a call that closes inside a string value ends the value where it stands")
    func closedInsideAValue() {
        let text = Self.call("\n<function=write_file>\n<parameter=path>\na.txt\n</parameter>\n<parameter=content>\none\ntwo\n") + "after"
        expect(text, Parsed(content: "after", calls: [.init(name: "write_file", arguments: #"{"path":"a.txt","content":"one\ntwo"}"#)], whole: 1), "closed early")
    }

    @Test("a long value that has to be read whole still sends word that the call is alive")
    func heartbeat() {
        let list = "[" + (0..<400).map { "{\"old\": \"line \($0)\", \"new\": \"LINE \($0)\"}" }.joined(separator: ", ") + "]"
        let body = "\n<function=edit_file>\n<parameter=path>\na.swift\n</parameter>\n<parameter=edits>\n" + list + "\n</parameter>\n</function>\n"
        let wanted = Self.whole(body)!.arguments
        // About four characters to a token.
        let bytes = Array(Self.call(body).utf8)
        let chunks = stride(from: 0, to: bytes.count, by: 4).map { Array(bytes[$0..<min($0 + 4, bytes.count)]) }
        let got = Self.parse(chunks)
        #expect(got.calls == [.init(name: "edit_file", arguments: wanted)])
        #expect(got.empty >= chunks.count / 64 - 2, "\(got.empty) empty pieces over \(chunks.count) tokens")
        #expect(got.empty <= chunks.count / 64 + 1)
    }
}
