import Foundation
import Testing
import SploshServer

@Suite("ToolCallParserTests")
struct ToolCallParserTests {
    @Test("parses renderer tool-call grammar and JSON values")
    func parsesToolCall() throws {
        let text = """
        <tool_call>
        <function=list>
        <parameter=path>
        \"Sources\"
        </parameter>
        <parameter=depth>
        2
        </parameter>
        <parameter=hidden>
        false
        </parameter>
        </function>
        </tool_call>
        """
        let calls = ToolCallParser.parse(text)
        #expect(calls.count == 1)
        guard let call = calls.first else { return }
        #expect(call.type == "function")
        #expect(call.function.name == "list")
        #expect(call.id.count <= 40)
        #expect(!call.id.contains("|"))
        if case .string(let value) = call.function.arguments["path"] { #expect(value == "Sources") } else { #expect(Bool(false)) }
        if case .integer(let value) = call.function.arguments["depth"] { #expect(value == 2) } else { #expect(Bool(false)) }
        if case .boolean(let value) = call.function.arguments["hidden"] { #expect(!value) } else { #expect(Bool(false)) }
    }

    @Test("parses nested JSON argument values")
    func parsesNestedJSON() {
        let calls = ToolCallParser.parse("<tool_call><function=search><parameter=query>{\"terms\":[\"swift\",\"metal\"],\"limit\":3}</parameter></function></tool_call>")
        #expect(calls.first?.function.name == "search")
        guard let value = calls.first?.function.arguments["query"] else { #expect(Bool(false)); return }
        if case .object(let object) = value {
            #expect(object.count == 2)
        } else { #expect(Bool(false)) }
    }
}
