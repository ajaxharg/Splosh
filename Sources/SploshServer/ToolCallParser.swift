import Foundation
#if canImport(CryptoKit)
import CryptoKit
#endif

public struct ParsedToolCall: Sendable {
    public let id: String
    public let type: String
    public let function: Function
    public init(id: String, type: String = "function", function: Function) { self.id = id; self.type = type; self.function = function }
    public struct Function: Sendable {
        public let name: String
        public let arguments: [String: AnySendable]
        public init(name: String, arguments: [String: AnySendable]) { self.name = name; self.arguments = arguments }
    }
}

/// Incremental UTF-8/tool-call parser. `append` accepts arbitrary byte chunks and never
/// decodes an incomplete scalar as U+FFFD.
public struct StreamingToolCallParser: Sendable {
    public enum Event: Sendable { case text(String); case created(index: Int, id: String, name: String); case arguments(index: Int, id: String, fragment: String); case finished }
    private var bytes: [UInt8] = []
    private var text = ""
    private var nextIndex = 0
    public init() {}
    public mutating func append(_ data: Data) -> [Event] { bytes.append(contentsOf: data); drainUTF8(); return drainToolCalls() }
    public mutating func append(_ value: String) -> [Event] { append(Data(value.utf8)) }
    public mutating func finish() -> [Event] { drainUTF8(force: true); return drainToolCalls() }
    private mutating func drainUTF8(force: Bool = false) {
        guard !bytes.isEmpty else { return }
        if let s = String(bytes: bytes, encoding: .utf8) { text += s; bytes.removeAll() }
        else if force { text += String(decoding: bytes, as: UTF8.self); bytes.removeAll() }
        else { while !bytes.isEmpty { let n = bytes.count - 1; if let s = String(bytes: bytes[0...n], encoding: .utf8) { text += s; bytes.removeFirst(n + 1); break }; if n == 0 { break } } }
    }
    private mutating func drainToolCalls() -> [Event] {
        var out: [Event] = []
        while let open = text.range(of: SpecialTokens.toolCallOpen), let close = text.range(of: SpecialTokens.toolCallClose, range: open.upperBound..<text.endIndex) {
            let before = String(text[..<open.lowerBound]); if !before.isEmpty { out.append(.text(before)) }
            let body = String(text[open.upperBound..<close.lowerBound]); text = String(text[close.upperBound...])
            guard let fs = body.range(of: "<function="), let ne = body.range(of: ">", range: fs.upperBound..<body.endIndex) else { continue }
            let name = String(body[fs.upperBound..<ne.lowerBound]); let raw = String(body[ne.upperBound...])
            let args = raw.trimmingCharacters(in: .whitespacesAndNewlines)
            let id = ToolCallParser.makeID(name + "‖" + args); let index = nextIndex; nextIndex += 1
            out.append(.created(index: index, id: id, name: name))
            out.append(.arguments(index: index, id: id, fragment: args))
            out.append(.finished)
        }
        return out
    }
}

public enum ToolCallParser {
    public typealias Call = ParsedToolCall
    public static func parse(_ text: String) -> [ParsedToolCall] {
        var calls: [ParsedToolCall] = []; var search = text.startIndex
        while let openRange = text.range(of: SpecialTokens.toolCallOpen, range: search..<text.endIndex), let closeRange = text.range(of: SpecialTokens.toolCallClose, range: openRange.upperBound..<text.endIndex) {
            if let call = parseBody(String(text[openRange.upperBound..<closeRange.lowerBound])) { calls.append(call) }; search = closeRange.upperBound
        }; return calls
    }
    private static func parseBody(_ body: String) -> ParsedToolCall? {
        guard let fs = body.range(of: "<function="), let ne = body.range(of: ">", range: fs.upperBound..<body.endIndex) else { return nil }
        let name = String(body[fs.upperBound..<ne.lowerBound]); guard !name.isEmpty else { return nil }
        let end = body.range(of: "</function>", range: ne.upperBound..<body.endIndex)?.lowerBound ?? body.endIndex
        let parameters = String(body[ne.upperBound..<end]); var args: [String: AnySendable] = [:]; var cursor = parameters.startIndex
        while let p = parameters.range(of: "<parameter=", range: cursor..<parameters.endIndex) { guard let ke = parameters.range(of: ">", range: p.upperBound..<parameters.endIndex), let ce = parameters.range(of: "</parameter>", range: ke.upperBound..<parameters.endIndex) else { break }; args[String(parameters[p.upperBound..<ke.lowerBound])] = decodeJSON(String(parameters[ke.upperBound..<ce.lowerBound]).trimmingCharacters(in: .whitespacesAndNewlines)); cursor = ce.upperBound }
        let canonical = "{" + args.keys.sorted().map { "\"\($0)\":\(args[$0]!.encodedJSONForParser)" }.joined(separator: ",") + "}"
        return ParsedToolCall(id: makeID(name + canonical), function: .init(name: name, arguments: args))
    }
    private static func decodeJSON(_ raw: String) -> AnySendable { guard let d = raw.data(using: .utf8), let o = try? JSONSerialization.jsonObject(with: d, options: [.fragmentsAllowed]) else { return .string(raw) }; return convert(o) }
    private static func convert(_ v: Any) -> AnySendable { if v is NSNull { return .null }; if let x = v as? Bool { return .boolean(x) }; if let x = v as? Int { return .integer(x) }; if let x = v as? NSNumber { return .number(x.doubleValue) }; if let x = v as? String { return .string(x) }; if let x = v as? [Any] { return .array(x.map(convert)) }; if let x = v as? [String: Any] { return .object(x.map { ($0.key, convert($0.value)) }.sorted { $0.0 < $1.0 }) }; return .null }
    fileprivate static func makeID(_ input: String) -> String {
#if canImport(CryptoKit)
        return SHA256.hash(data: Data(input.utf8)).map { String(format: "%02x", $0) }.joined().prefix(8).description
#else
        var h: UInt64 = 14695981039346656037; for b in input.utf8 { h = (h ^ UInt64(b)) &* 1099511628211 }; return String(format: "%016llx", h).prefix(8).description
#endif
    }
}
private extension AnySendable { var encodedJSONForParser: String { switch self { case .string(let v): return String(decoding: try! JSONEncoder().encode(v), as: UTF8.self); case .number(let v): return String(v); case .integer(let v): return String(v); case .boolean(let v): return v ? "true" : "false"; case .null: return "null"; case .array(let v): return "[" + v.map{$0.encodedJSONForParser}.joined(separator:",") + "]"; case .object(let v): return "{" + v.map{"\"\($0.0)\":\($0.1.encodedJSONForParser)"}.joined(separator:",") + "}" } } }
