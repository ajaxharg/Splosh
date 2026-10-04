import Foundation
import XCTest
@testable import SploshServer

/// Does the next turn's prompt continue a slot's text (the last prompt plus the reply generated
/// after it), or does it leave that text somewhere inside the reply?
///
/// A reply is generated as tokens, parsed into reasoning / content / tool calls for the client,
/// and comes back in the next request as JSON that the chat template renders again. The slot can
/// only be carried on from all of its text if those two forms are the same tokens. Each case
/// here writes a reply the way the model emits it, runs it through the server's own parser one
/// token at a time (as the streaming endpoint does), builds the message a client sends back,
/// renders and tokenises the next prompt, and reports where the two sequences part.
///
/// The "generated" tokens are the tokenizer's own encoding of the reply text, piece by piece:
/// the model is not loaded. What the model writes that the tokenizer would have split
/// differently is not covered.
final class ReplyRoundTripTests: XCTestCase {
    private enum Ending { case stop, length }

    private struct Outcome {
        let prompt: Int, generated: Int, slot: Int, next: Int, common: Int
        /// What `BatchScheduler.chooseSlot` would find for the idle slot: "tokens" (all of its
        /// text is a prefix of the new prompt), "promptEnd" (only the last prompt is), "neither".
        let source: String
        let evaluated: Int
        let slotSide: String, nextSide: String
    }

    private static let toolsJSON = """
    [
      {"type": "function", "function": {"name": "bash", "description": "Run a shell command.", "parameters": {"type": "object", "properties": {"command": {"type": "string"}, "timeout": {"type": "integer"}, "background": {"type": "boolean"}, "ratio": {"type": "number"}}, "required": ["command"]}}},
      {"type": "function", "function": {"name": "write_file", "description": "Write a file.", "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]}}},
      {"type": "function", "function": {"name": "edit_file", "description": "Edit a file.", "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "edits": {"type": "array", "items": {"type": "object"}}, "options": {"type": "object"}}, "required": ["path", "edits"]}}}
    ]
    """

    private static let thinking = """
    The user wants me to find where the server decides which slot to reuse. Let me think about this step by step.

    1. First, list the Swift sources — `ls Sources/SploshRuntime`.
    2. Then grep for "chooseSlot" (it's probably in BatchScheduler.swift).

    Caveat: the path might contain spaces, e.g. "My Files/a b.txt"; I'll quote it. Cost ≈ 3 × 16 = 48 rows → fine.
    """

    private static let close = "\n</think>\n\n"
    private static let bash = "<tool_call>\n<function=bash>\n<parameter=command>\ngrep -rn \"chooseSlot\" Sources/ | head -20\n</parameter>\n</function>\n</tool_call>"
    private static let write = "<tool_call>\n<function=write_file>\n<parameter=path>\nSources/a b.swift\n</parameter>\n<parameter=content>\nfunc a() {\n    return 1  \n}\n\n// café\n\n</parameter>\n</function>\n</tool_call>"

    private func loadTokenizer() throws -> Tokenizer {
        let root = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
        return try Tokenizer(tokenizerURL: root.appendingPathComponent("inputs/tokenizer/tokenizer.json"),
                             configURL: root.appendingPathComponent("inputs/tokenizer/tokenizer_config.json"))
    }

    private func show(_ tokenizer: Tokenizer, _ tokens: ArraySlice<Int>) -> String {
        tokens.isEmpty ? "(end)" : tokenizer.decode(Array(tokens)).debugDescription
    }

    /// One turn and the request after it.
    private func turn(_ tokenizer: Tokenizer, pieces: [String], ending: Ending = .stop, stops: [String] = [],
                      enableThinking: Bool? = nil, client: (String) -> String = { $0 }) throws -> Outcome {
        let tools = try JSONValue.parse(Self.toolsJSON).arrayValue ?? []
        let options = ChatTemplate.Options(enableThinking: enableThinking)
        let history: [JSONValue] = [
            .object([("role", .string("system")), ("content", .string("You are a coding agent. Work in the repository at hand."))]),
            .object([("role", .string("user")), ("content", .string("Where does the scheduler choose a slot?"))]),
        ]
        let rendered = try ChatTemplate.render(messages: history, tools: tools, options: options)
        let prompt = tokenizer.encode(rendered.prompt)

        // The reply as the model emits it: tokens after the prompt, never merged with it.
        let generated = pieces.flatMap { tokenizer.encode($0) }

        // The server: one token at a time through the output parser, as the streaming endpoint does.
        var parser = ChatOutputParser(startsInThinking: rendered.startsInThinking, tools: tools, stopStrings: stops)
        var events: [ChatOutputParser.Event] = []
        var pushed = 0
        for token in generated {
            events += parser.push(tokenizer.bytes(for: token) ?? [])
            pushed += 1
            if parser.stopped { break }
        }
        events += parser.finish()

        // The slot's text when the session ends. A stop token is sampled and never evaluated; at
        // the token limit the last sampled token is sent to the client and never evaluated.
        let evaluated = ending == .length ? max(0, pushed - 1) : pushed
        let slot = prompt + generated[0..<evaluated]

        // The client: deltas joined, and sent back as the assistant message of the next request.
        var reasoning = "", content = ""
        var streamed: [(id: String, name: String, arguments: String)] = []
        for event in events {
            switch event {
            case .reasoning(let text): reasoning += text
            case .content(let text): content += text
            case .toolCallStart(_, let id, let name): streamed.append((id, name, ""))
            case .toolCallArguments(let index, let piece): streamed[index].arguments += piece
            case .toolCallEnd: break
            }
        }
        // Calls come in pieces; a client makes the ones that are whole.
        var calls: [JSONValue] = []
        var results: [JSONValue] = []
        for call in streamed.prefix(parser.toolCallCount) {
            calls.append(.object([("id", .string(call.id)), ("type", .string("function")),
                                  ("function", .object([("name", .string(call.name)), ("arguments", .string(client(call.arguments)))]))]))
            results.append(.object([("role", .string("tool")), ("tool_call_id", .string(call.id)),
                                    ("content", .string("Sources/SploshRuntime/BatchScheduler.swift:806: private func chooseSlot"))]))
        }
        var assistant: [(key: String, value: JSONValue)] = [("role", .string("assistant")), ("content", .string(content))]
        if !reasoning.isEmpty { assistant.append(("reasoning_content", .string(reasoning))) }
        if !calls.isEmpty { assistant.append(("tool_calls", .array(calls))) }
        if results.isEmpty { results = [.object([("role", .string("user")), ("content", .string("Go on."))])] }

        let next = tokenizer.encode(try ChatTemplate.render(messages: history + [.object(assistant)] + results, tools: tools, options: options).prompt)
        var common = 0
        while common < min(slot.count, next.count), slot[common] == next[common] { common += 1 }

        let source: String
        let reused: Int
        if slot.count < next.count, common == slot.count { source = "tokens"; reused = slot.count }
        else if prompt.count < next.count, next.starts(with: prompt) { source = "promptEnd"; reused = prompt.count }
        else { source = "neither"; reused = 0 }
        return Outcome(prompt: prompt.count, generated: generated.count, slot: slot.count, next: next.count, common: common,
                       source: source, evaluated: next.count - reused,
                       slotSide: show(tokenizer, slot[common..<min(slot.count, common + 3)]),
                       nextSide: show(tokenizer, next[common..<min(next.count, common + 3)]))
    }

    private func report(_ name: String, _ outcome: Outcome) {
        var line = "RT| \(name): prompt \(outcome.prompt) + reply \(outcome.generated) -> slot \(outcome.slot); next prompt \(outcome.next); "
            + "common prefix \(outcome.common); reuse .\(outcome.source); \(outcome.evaluated) to evaluate"
        if outcome.source != "tokens" {
            line += " (\(outcome.next - outcome.common) lie past where they part); parts at slot \(outcome.slotSide) vs next \(outcome.nextSide)"
        }
        print(line)
    }

    /// Replies written the way the template writes them: the next prompt must continue the slot.
    func testCanonicalRepliesAreExactContinuations() throws {
        let tokenizer = try loadTokenizer()
        let t = Self.thinking, c = Self.close
        let cases: [(String, [String])] = [
            ("thinking, one tool call", [t + c + Self.bash]),
            ("thinking, content, one tool call", [t + c + "I'll search for it first.\n\n" + Self.bash]),
            ("thinking, two tool calls, multi-line string value", [t + c + Self.bash + "\n" + Self.write]),
            ("thinking, typed values in tojson form", [t + c + "<tool_call>\n<function=bash>\n<parameter=command>\nsleep 5\n</parameter>\n<parameter=timeout>\n30\n</parameter>\n<parameter=background>\ntrue\n</parameter>\n<parameter=ratio>\n1.50\n</parameter>\n</function>\n</tool_call>\n<tool_call>\n<function=edit_file>\n<parameter=path>\na.swift\n</parameter>\n<parameter=edits>\n[{\"old\": \"let a = 1\", \"new\": \"let a = 2\\n\"}, {\"old\": \"é\", \"new\": \"e\"}]\n</parameter>\n<parameter=options>\n{\"dry_run\": false, \"limit\": null}\n</parameter>\n</function>\n</tool_call>"]),
            ("thinking, final answer, no tool call", [t + c + "It is chosen in `chooseSlot` (BatchScheduler.swift:806).\n\n- longest prefix first\n- then an empty slot"]),
            ("boolean written Python-style (True)", [t + c + "<tool_call>\n<function=bash>\n<parameter=command>\nls\n</parameter>\n<parameter=background>\nTrue\n</parameter>\n</function>\n</tool_call>"]),
        ]
        for (name, pieces) in cases {
            let outcome = try turn(tokenizer, pieces: pieces)
            report(name, outcome)
            XCTAssertEqual(outcome.source, "tokens", name)
            XCTAssertEqual(outcome.common, outcome.slot, name)
        }
    }

    /// Replies that differ from the template's form in small ways, and replies that were cut.
    /// Nothing is asserted beyond the prompt's own stability: this reports what happens today.
    func testDeviationsReport() throws {
        let tokenizer = try loadTokenizer()
        let t = Self.thinking, c = Self.close
        let cases: [(String, [String], Ending, [String])] = [
            ("reasoning opens with a blank line", ["\n" + t + c + Self.bash], .stop, []),
            ("two newlines before </think>", [t + "\n\n</think>\n\n" + Self.bash], .stop, []),
            ("one newline after </think>", [t + "\n</think>\n" + Self.bash], .stop, []),
            ("no newline before </think>", [t + "</think>\n\n" + Self.bash], .stop, []),
            ("content, one newline before <tool_call>", [t + c + "I'll search for it first.\n" + Self.bash], .stop, []),
            ("two newlines between tool calls", [t + c + Self.bash + "\n\n" + Self.write], .stop, []),
            ("newline after the last </tool_call>", [t + c + Self.bash + "\n"], .stop, []),
            ("content after the tool call", [t + c + Self.bash + "\n\nThat should find it."], .stop, []),
            ("final answer ending in a newline", [t + c + "It is chosen in `chooseSlot`.\n"], .stop, []),
            ("array value written compactly", [t + c + "<tool_call>\n<function=edit_file>\n<parameter=path>\na.swift\n</parameter>\n<parameter=edits>\n[{\"old\":\"a\",\"new\":\"b\"}]\n</parameter>\n</function>\n</tool_call>"], .stop, []),
            ("object value written over several lines", [t + c + "<tool_call>\n<function=edit_file>\n<parameter=path>\na.swift\n</parameter>\n<parameter=edits>\n[]\n</parameter>\n<parameter=options>\n{\n  \"dry_run\": false\n}\n</parameter>\n</function>\n</tool_call>"], .stop, []),
            ("string value without its newlines", [t + c + "<tool_call>\n<function=bash>\n<parameter=command>ls</parameter>\n</function>\n</tool_call>"], .stop, []),
            ("empty thinking: newline, </think>", ["\n</think>\n\n" + Self.bash], .stop, []),
            ("empty thinking: </think> at once", ["</think>\n\n" + Self.bash], .stop, []),
            ("</think> and newlines as separate tokens", [t, "\n", "</think>", "\n", "\n", Self.bash], .stop, []),
            ("token limit inside thinking", [String(t.prefix(150))], .length, []),
            ("token limit inside content", [t + c + "It is chosen in `chooseSl"], .length, []),
            ("token limit inside a tool call", [t + c + "<tool_call>\n<function=bash>\n<parameter=command>\ngrep -rn \"cho"], .length, []),
            ("stop string in content", [t + c + "It is chosen in chooseSlot.\nObservation: none needed"], .stop, ["\nObservation:"]),
        ]
        for (name, pieces, ending, stops) in cases {
            let outcome = try turn(tokenizer, pieces: pieces, ending: ending, stops: stops)
            report(name, outcome)
        }
        // A client that parses the arguments and writes them again (JavaScript's JSON.stringify
        // turns 1.0 into 1) changes a number's spelling.
        let respelled = try turn(tokenizer, pieces: [t + c + "<tool_call>\n<function=bash>\n<parameter=command>\nls\n</parameter>\n<parameter=ratio>\n1.0\n</parameter>\n</function>\n</tool_call>"],
                                 client: { $0.replacingOccurrences(of: "1.0", with: "1") })
        report("number respelled by the client (1.0 -> 1)", respelled)
        // Thinking off: the prompt itself closes the think block, and the reply is the answer.
        let off = try turn(tokenizer, pieces: [Self.bash], enableThinking: false)
        report("thinking disabled, one tool call", off)
        XCTAssertEqual(off.source, "tokens")
    }
}
