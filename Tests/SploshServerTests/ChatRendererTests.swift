import Foundation
import Testing
import SploshServer

@Suite("ChatRendererTests")
struct ChatRendererTests {
    private func golden(_ name: String) -> String {
        let root = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
        return (try? String(contentsOf: root.appendingPathComponent("Tests/Goldens/renderer/\(name).txt"), encoding: .utf8)) ?? ""
    }

    @Test("system and user")
    func systemAndUser() {
        let actual = ChatRenderer.render(messages: [
            .init(role: .system, content: "Be concise."),
            .init(role: .user, content: "Hello")
        ])
        #expect(actual == golden("system-user"), "rendered prompt differs from system-user golden")
    }

    @Test("assistant tool call")
    func assistantToolCall() {
        let actual = ChatRenderer.render(messages: [
            .init(role: .user, content: "List files"),
            .init(role: .assistant, toolCalls: [.init(name: "list", arguments: [
                ("path", .string("Sources")), ("depth", .integer(2)), ("hidden", .boolean(false))
            ])])
        ])
        #expect(actual == golden("assistant-tool-call"), "rendered prompt differs from assistant tool-call golden")
    }

    @Test("tool result")
    func toolResult() {
        let actual = ChatRenderer.render(messages: [
            .init(role: .user, content: "List files"),
            .init(role: .assistant, toolCalls: [.init(name: "list")]),
            .init(role: .tool, content: "a.swift", toolCallID: "call")
        ])
        #expect(actual == golden("tool-result"), "rendered prompt differs from tool-result golden")
    }

    @Test("two consecutive tool results merge")
    func consecutiveToolResults() {
        let actual = ChatRenderer.render(messages: [
            .init(role: .assistant, toolCalls: [.init(name: "one")]),
            .init(role: .tool, content: "one", toolCallID: "1"),
            .init(role: .tool, content: "two", toolCallID: "2")
        ])
        #expect(actual == golden("two-tool-results"), "rendered prompt differs from two-tool-results golden")
    }
}
