import Foundation
import Testing
@testable import SploshServer

@Suite("Replies as generated")
struct ReplyAliasesTests {
    private func tokenizer() throws -> Tokenizer {
        let root = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
        return try Tokenizer(tokenizerURL: root.appendingPathComponent("inputs/tokenizer/tokenizer.json"),
                             configURL: root.appendingPathComponent("inputs/tokenizer/tokenizer_config.json"))
    }

    /// The model spells a word its own way (the same text, other tokens); the next prompt has
    /// the tokeniser's spelling. The prompt is given the model's back, and the turn after that
    /// finds its own prompt in the result.
    @Test("a reply that comes back as the same text gets its generated tokens back")
    func sameTextOtherTokens() throws {
        let tokenizer = try tokenizer()
        func bytes(_ token: Int) -> [UInt8] { tokenizer.bytes(for: token) ?? [] }
        func text(_ tokens: [Int]) -> [UInt8] { tokens.flatMap(bytes) }
        let aliases = ReplyAliases()

        let opening = "<|im_start|>user\nSay a word.<|im_end|>\n<|im_start|>assistant\n<think>\n"
        let prompt = tokenizer.encode(opening)
        let reply = "The word is Molto, as asked.\n</think>\n\nMolto"
        // As the model might write it: the same characters, split where the tokeniser would not.
        let generated = ["The word is M", "ol", "to, as asked.\n</think>\n\nM", "ol", "to"].flatMap { tokenizer.encode($0) }
        #expect(text(generated) == Array(reply.utf8))
        let second = opening + reply + "<|im_end|>\n<|im_start|>user\nAgain.<|im_end|>\n<|im_start|>assistant\n<think>\n"
        let next = tokenizer.encode(second)
        #expect(next.starts(with: prompt))
        #expect(!next.starts(with: prompt + generated), "the tokeniser spells it the model's way: the case is not exercised")

        aliases.remember(prompt: prompt, generated: generated, bytes: text(generated))
        let carried = aliases.apply(to: next, bytes: bytes)
        #expect(carried.starts(with: prompt + generated))
        #expect(text(carried) == text(next))                              // the same text, to the byte

        // The next turn: its reply's prompt is the carried one, and is found in a prompt that
        // arrives with both replies in the tokeniser's spelling.
        let generated2 = ["Still M", "ol", "to."].flatMap { tokenizer.encode($0) }
        aliases.remember(prompt: carried, generated: generated2, bytes: text(generated2))
        let third = tokenizer.encode(second + "Still Molto.<|im_end|>\n<|im_start|>user\nMore.<|im_end|>\n<|im_start|>assistant\n<think>\n")
        let carried3 = aliases.apply(to: third, bytes: bytes)
        #expect(carried3.starts(with: carried + generated2))
        #expect(text(carried3) == text(third))
    }

    @Test("a reply that comes back changed, or was spelt the usual way, changes nothing")
    func leftAlone() throws {
        let tokenizer = try tokenizer()
        func bytes(_ token: Int) -> [UInt8] { tokenizer.bytes(for: token) ?? [] }
        let aliases = ReplyAliases()
        let opening = "<|im_start|>user\nSay a word.<|im_end|>\n<|im_start|>assistant\n<think>\n"
        let prompt = tokenizer.encode(opening)
        let generated = ["The word is M", "ol", "to."].flatMap { tokenizer.encode($0) }
        aliases.remember(prompt: prompt, generated: generated, bytes: generated.flatMap(bytes))

        // One character different: not the same reply.
        let changed = tokenizer.encode(opening + "The word is Molto!<|im_end|>\n")
        #expect(aliases.apply(to: changed, bytes: bytes) == changed)
        // A different conversation of the same length.
        let other = tokenizer.encode("<|im_start|>user\nSay a noun.<|im_end|>\n<|im_start|>assistant\n<think>\nThe word is Molto.<|im_end|>\n")
        #expect(aliases.apply(to: other, bytes: bytes) == other)
        // Spelt the tokeniser's way: the prompt is unchanged.
        let usual = tokenizer.encode("A plain reply.")
        aliases.remember(prompt: prompt, generated: usual, bytes: usual.flatMap(bytes))
        let plain = tokenizer.encode(opening + "A plain reply.<|im_end|>\n")
        #expect(aliases.apply(to: plain, bytes: bytes) == plain)
    }
}
