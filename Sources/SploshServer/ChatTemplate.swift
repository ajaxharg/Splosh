import Foundation

/// A statement-for-statement port of `inputs/tokenizer/chat_template.jinja`.
///
/// The prompt must match what the model was trained on, and it must be byte-stable across turns
/// so the prefix cache hits, so this follows the template literally rather than approximating
/// it. Vision content is rejected: Splosh serves text only.
public enum ChatTemplate {
    public struct Failure: Error, CustomStringConvertible, Equatable {
        public let message: String
        public init(_ message: String) { self.message = message }
        public var description: String { message }
    }

    public struct Options: Sendable, Equatable {
        /// nil is the template's "undefined", which enables thinking.
        public var enableThinking: Bool?
        /// "xhigh" (the template default), "medium" or "low".
        public var reasoningEffort: String?
        /// nil is "undefined", which preserves every assistant turn's thinking block.
        public var preserveThinking: Bool?
        public var addGenerationPrompt = true
        public init(enableThinking: Bool? = nil, reasoningEffort: String? = nil, preserveThinking: Bool? = nil) {
            self.enableThinking = enableThinking; self.reasoningEffort = reasoningEffort
            self.preserveThinking = preserveThinking
        }
    }

    public struct Rendered: Sendable, Equatable {
        public let prompt: String
        /// True when the prompt ends inside an open `<think>` block, so generation starts with
        /// reasoning rather than the answer.
        public let startsInThinking: Bool
    }

    static let toolInstructions = "\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n<tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n</parameter>\n<parameter=example_parameter_2>\nThis is the value for the second parameter\nthat can span\nmultiple lines\n</parameter>\n</function>\n</tool_call>\n\n<IMPORTANT>\nReminder:\n- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags\n- Required parameters MUST be specified\n- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after\n- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls\n</IMPORTANT>"

    public static func render(messages: [JSONValue], tools: [JSONValue] = [], options: Options = Options()) throws -> Rendered {
        guard !messages.isEmpty else { throw Failure("No messages provided.") }
        var out = ""

        var reasoningInstructions = ""
        if options.enableThinking != false {
            let effort = options.reasoningEffort ?? "xhigh"
            switch effort {
            case "xhigh":
                reasoningInstructions = "Reasoning effort is set to xhigh. Please think carefully through the task, validate key assumptions, consider plausible alternatives, and prioritize correctness, consistency, and clarity in the final answer."
            case "low":
                reasoningInstructions = "Reasoning effort is set to low. Keep your thinking brief and focused, moving directly to the conclusion without unnecessary elaboration."
            case "medium": break
            default:
                throw Failure("Unexpected reasoning effort \(effort). Supported types are xhigh (default), medium, and low.")
            }
        }

        let firstIsSystem = role(messages[0]) == "system"
        if !tools.isEmpty {
            out += "<|im_start|>system\n"
            if !reasoningInstructions.isEmpty { out += reasoningInstructions + "\n\n" }
            out += "# Tools\n\nYou have access to the following functions:\n\n<tools>"
            for tool in tools { out += "\n" + tool.pythonStyle }
            out += "\n</tools>"
            out += toolInstructions
            if firstIsSystem {
                let content = try trim(renderContent(messages[0]["content"], system: true))
                if !content.isEmpty { out += "\n\n" + content }
            }
            out += "<|im_end|>\n"
        } else if firstIsSystem {
            let content = try trim(renderContent(messages[0]["content"], system: true))
            if !content.isEmpty {
                out += "<|im_start|>system\n" + (reasoningInstructions.isEmpty ? "" : reasoningInstructions + "\n\n") + content + "<|im_end|>\n"
            } else if !reasoningInstructions.isEmpty {
                out += "<|im_start|>system\n" + reasoningInstructions + "<|im_end|>\n"
            }
        } else if !reasoningInstructions.isEmpty {
            out += "<|im_start|>system\n" + reasoningInstructions + "<|im_end|>\n"
        }

        // The last user turn that is a real query rather than a wrapped tool response.
        var lastQueryIndex: Int?
        for index in messages.indices.reversed() where role(messages[index]) == "user" {
            let content = try trim(renderContent(messages[index]["content"], system: false))
            if !(content.hasPrefix("<tool_response>") && content.hasSuffix("</tool_response>")) {
                lastQueryIndex = index
                break
            }
        }
        guard let lastQueryIndex else { throw Failure("No user query found in messages.") }

        for (index, message) in messages.enumerated() {
            let content = try trim(renderContent(message["content"], system: false))
            switch role(message) {
            case "system":
                guard index == 0 else { throw Failure("System message must be at the beginning.") }
            case "user":
                out += "<|im_start|>user\n" + content + "<|im_end|>\n"
            case "assistant":
                let reasoning = trim(message["reasoning_content"]?.stringValue ?? "")
                if options.preserveThinking != false || index > lastQueryIndex {
                    out += "<|im_start|>assistant\n<think>\n" + reasoning + "\n</think>\n\n" + content
                } else {
                    out += "<|im_start|>assistant\n" + content
                }
                if let calls = message["tool_calls"]?.arrayValue {
                    for (callIndex, wrapped) in calls.enumerated() {
                        let call = wrapped["function"] ?? wrapped
                        let name = call["name"]?.stringValue ?? ""
                        if callIndex == 0 {
                            out += (content.isEmpty ? "" : "\n\n") + "<tool_call>\n<function=" + name + ">\n"
                        } else {
                            out += "\n<tool_call>\n<function=" + name + ">\n"
                        }
                        for (key, value) in try argumentMembers(call["arguments"]) {
                            out += "<parameter=" + key + ">\n"
                            out += value.stringValue ?? value.pythonStyle
                            out += "\n</parameter>\n"
                        }
                        out += "</function>\n</tool_call>"
                    }
                }
                out += "<|im_end|>\n"
            case "tool":
                if index > 0, role(messages[index - 1]) != "tool" { out += "<|im_start|>user" }
                out += "\n<tool_response>\n" + content + "\n</tool_response>"
                let isLast = index == messages.count - 1
                if isLast || role(messages[index + 1]) != "tool" { out += "<|im_end|>\n" }
            default:
                throw Failure("Unexpected message role.")
            }
        }

        var startsInThinking = false
        if options.addGenerationPrompt {
            out += "<|im_start|>assistant\n"
            if options.enableThinking == false {
                out += "<think>\n\n</think>\n\n"
            } else {
                out += "<think>\n"
                startsInThinking = true
            }
        }
        return Rendered(prompt: out, startsInThinking: startsInThinking)
    }

    private static func role(_ message: JSONValue) -> String { message["role"]?.stringValue ?? "" }

    /// Python `str.strip()`.
    private static func trim(_ text: String) -> String { text.trimmingCharacters(in: .whitespacesAndNewlines) }

    /// OpenAI clients send `arguments` as a JSON string; the template iterates a mapping.
    private static func argumentMembers(_ arguments: JSONValue?) throws -> [(key: String, value: JSONValue)] {
        guard let arguments else { return [] }
        switch arguments {
        case .object(let members): return members
        case .string(let text):
            if text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty { return [] }
            guard case .object(let members)? = try? JSONValue.parse(text) else {
                throw Failure("tool_call arguments must be a JSON object")
            }
            return members
        case .null: return []
        default: throw Failure("tool_call arguments must be a JSON object")
        }
    }

    private static func renderContent(_ content: JSONValue?, system: Bool) throws -> String {
        guard let content else { return "" }
        switch content {
        case .string(let text): return text
        case .null: return ""
        case .array(let items):
            var out = ""
            for item in items {
                let type = item["type"]?.stringValue
                if item["image"] != nil || item["image_url"] != nil || type == "image" || type == "image_url"
                    || item["video"] != nil || type == "video" {
                    throw Failure(system ? "System message cannot contain images." : "Image and video content is not supported; this server is text-only.")
                }
                guard let text = item["text"] else { throw Failure("Unexpected item type in content.") }
                out += text.stringValue ?? ""
            }
            return out
        default: throw Failure("Unexpected content type.")
        }
    }
}
