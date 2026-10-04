import Foundation

/// An order-preserving JSON value.
///
/// The chat template serialises tool schemas with `tojson`, and the rendered prompt has to be
/// byte-stable for the prefix cache to hit, so object key order from the request must survive.
/// `JSONSerialization` discards it; this parser keeps it. Numbers keep their source literal.
public indirect enum JSONValue: Sendable, Equatable {
    case null
    case bool(Bool)
    case number(String)
    case string(String)
    case array([JSONValue])
    case object([(key: String, value: JSONValue)])

    public static func == (lhs: JSONValue, rhs: JSONValue) -> Bool {
        switch (lhs, rhs) {
        case (.null, .null): return true
        case (.bool(let a), .bool(let b)): return a == b
        case (.number(let a), .number(let b)): return a == b
        case (.string(let a), .string(let b)): return a == b
        case (.array(let a), .array(let b)): return a == b
        case (.object(let a), .object(let b)):
            return a.count == b.count && zip(a, b).allSatisfy { $0.key == $1.key && $0.value == $1.value }
        default: return false
        }
    }

    // MARK: Access

    public subscript(key: String) -> JSONValue? {
        guard case .object(let members) = self else { return nil }
        return members.first { $0.key == key }?.value
    }

    public var stringValue: String? { if case .string(let s) = self { return s }; return nil }
    public var boolValue: Bool? { if case .bool(let b) = self { return b }; return nil }
    public var arrayValue: [JSONValue]? { if case .array(let a) = self { return a }; return nil }
    public var objectValue: [(key: String, value: JSONValue)]? { if case .object(let o) = self { return o }; return nil }
    public var doubleValue: Double? { if case .number(let n) = self { return Double(n) }; return nil }
    public var intValue: Int? {
        guard case .number(let n) = self else { return nil }
        return Int(n) ?? Double(n).flatMap { $0.isFinite ? Int($0) : nil }
    }
    public var isNull: Bool { if case .null = self { return true }; return false }

    public static func int(_ value: Int) -> JSONValue { .number(String(value)) }
    public static func double(_ value: Double) -> JSONValue { .number(value.isFinite ? String(value) : "0") }

    // MARK: Serialisation

    /// Compact form (`,` and `:` separators) for wire responses.
    public var compact: String { var out = ""; write(into: &out, item: ",", key: ":"); return out }

    /// Python `json.dumps(x, ensure_ascii=False)` form (`, ` and `: `), which is what the chat
    /// template's `tojson` filter emits.
    public var pythonStyle: String { var out = ""; write(into: &out, item: ", ", key: ": "); return out }

    private func write(into out: inout String, item: String, key: String) {
        switch self {
        case .null: out += "null"
        case .bool(let b): out += b ? "true" : "false"
        case .number(let n): out += n
        case .string(let s): Self.writeString(s, into: &out)
        case .array(let values):
            out += "["
            for (index, value) in values.enumerated() {
                if index > 0 { out += item }
                value.write(into: &out, item: item, key: key)
            }
            out += "]"
        case .object(let members):
            out += "{"
            for (index, member) in members.enumerated() {
                if index > 0 { out += item }
                Self.writeString(member.key, into: &out)
                out += key
                member.value.write(into: &out, item: item, key: key)
            }
            out += "}"
        }
    }

    private static func writeString(_ value: String, into out: inout String) {
        out += "\""
        for scalar in value.unicodeScalars {
            switch scalar {
            case "\"": out += "\\\""
            case "\\": out += "\\\\"
            case "\n": out += "\\n"
            case "\r": out += "\\r"
            case "\t": out += "\\t"
            case "\u{08}": out += "\\b"
            case "\u{0C}": out += "\\f"
            default:
                if scalar.value < 0x20 { out += String(format: "\\u%04x", scalar.value) }
                else { out.unicodeScalars.append(scalar) }
            }
        }
        out += "\""
    }

    // MARK: Parsing

    public struct ParseError: Error, CustomStringConvertible {
        public let message: String
        public let offset: Int
        public var description: String { "invalid JSON at byte \(offset): \(message)" }
    }

    public static func parse(_ text: String) throws -> JSONValue { try parse(Array(text.utf8)) }

    public static func parse(_ bytes: [UInt8]) throws -> JSONValue {
        var parser = Parser(bytes: bytes)
        parser.skipWhitespace()
        let value = try parser.value(depth: 0)
        parser.skipWhitespace()
        guard parser.index == bytes.count else { throw ParseError(message: "trailing data", offset: parser.index) }
        return value
    }

    private struct Parser {
        let bytes: [UInt8]
        var index = 0

        mutating func skipWhitespace() {
            while index < bytes.count, [0x20, 0x09, 0x0A, 0x0D].contains(bytes[index]) { index += 1 }
        }

        func fail(_ message: String) -> ParseError { ParseError(message: message, offset: index) }

        mutating func value(depth: Int) throws -> JSONValue {
            guard depth < 256 else { throw fail("nesting too deep") }
            guard index < bytes.count else { throw fail("unexpected end") }
            switch bytes[index] {
            case UInt8(ascii: "{"):
                index += 1
                var members: [(key: String, value: JSONValue)] = []
                skipWhitespace()
                if index < bytes.count, bytes[index] == UInt8(ascii: "}") { index += 1; return .object(members) }
                while true {
                    skipWhitespace()
                    guard index < bytes.count, bytes[index] == UInt8(ascii: "\"") else { throw fail("expected object key") }
                    let key = try string()
                    skipWhitespace()
                    guard index < bytes.count, bytes[index] == UInt8(ascii: ":") else { throw fail("expected ':'") }
                    index += 1
                    skipWhitespace()
                    members.append((key, try value(depth: depth + 1)))
                    skipWhitespace()
                    guard index < bytes.count else { throw fail("unterminated object") }
                    if bytes[index] == UInt8(ascii: ",") { index += 1; continue }
                    if bytes[index] == UInt8(ascii: "}") { index += 1; return .object(members) }
                    throw fail("expected ',' or '}'")
                }
            case UInt8(ascii: "["):
                index += 1
                var values: [JSONValue] = []
                skipWhitespace()
                if index < bytes.count, bytes[index] == UInt8(ascii: "]") { index += 1; return .array(values) }
                while true {
                    skipWhitespace()
                    values.append(try value(depth: depth + 1))
                    skipWhitespace()
                    guard index < bytes.count else { throw fail("unterminated array") }
                    if bytes[index] == UInt8(ascii: ",") { index += 1; continue }
                    if bytes[index] == UInt8(ascii: "]") { index += 1; return .array(values) }
                    throw fail("expected ',' or ']'")
                }
            case UInt8(ascii: "\""): return .string(try string())
            case UInt8(ascii: "t"): try literal("true"); return .bool(true)
            case UInt8(ascii: "f"): try literal("false"); return .bool(false)
            case UInt8(ascii: "n"): try literal("null"); return .null
            default: return try number()
            }
        }

        mutating func literal(_ word: String) throws {
            let expected = Array(word.utf8)
            guard index + expected.count <= bytes.count, Array(bytes[index..<index + expected.count]) == expected else {
                throw fail("unexpected token")
            }
            index += expected.count
        }

        mutating func number() throws -> JSONValue {
            let start = index
            while index < bytes.count, "+-0123456789.eE".utf8.contains(bytes[index]) { index += 1 }
            let text = String(decoding: bytes[start..<index], as: UTF8.self)
            guard !text.isEmpty, Double(text) != nil else { index = start; throw fail("invalid number") }
            return .number(text)
        }

        mutating func hex4() throws -> UInt32 {
            guard index + 4 <= bytes.count,
                  let value = UInt32(String(decoding: bytes[index..<index + 4], as: UTF8.self), radix: 16) else {
                throw fail("invalid \\u escape")
            }
            index += 4
            return value
        }

        mutating func string() throws -> String {
            index += 1 // opening quote
            var out: [UInt8] = []
            while index < bytes.count {
                let byte = bytes[index]
                if byte == UInt8(ascii: "\"") { index += 1; return String(decoding: out, as: UTF8.self) }
                if byte == UInt8(ascii: "\\") {
                    index += 1
                    guard index < bytes.count else { break }
                    let escape = bytes[index]
                    index += 1
                    switch escape {
                    case UInt8(ascii: "n"): out.append(0x0A)
                    case UInt8(ascii: "r"): out.append(0x0D)
                    case UInt8(ascii: "t"): out.append(0x09)
                    case UInt8(ascii: "b"): out.append(0x08)
                    case UInt8(ascii: "f"): out.append(0x0C)
                    case UInt8(ascii: "u"):
                        var code = try hex4()
                        if (0xD800...0xDBFF).contains(code), index + 6 <= bytes.count,
                           bytes[index] == UInt8(ascii: "\\"), bytes[index + 1] == UInt8(ascii: "u") {
                            index += 2
                            let low = try hex4()
                            if (0xDC00...0xDFFF).contains(low) { code = 0x10000 + ((code - 0xD800) << 10) + (low - 0xDC00) }
                        }
                        out += Array(String(UnicodeScalar(code) ?? "\u{FFFD}").utf8)
                    default: out.append(escape) // \" \\ \/
                    }
                    continue
                }
                out.append(byte)
                index += 1
            }
            throw fail("unterminated string")
        }
    }
}
