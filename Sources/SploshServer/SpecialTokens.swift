import Foundation

/// Token ids and delimiters shared by prompt rendering and streaming detokenization.
public enum SpecialTokens {
    public static let endOfTextID = 248044
    public static let imStartID = 248045
    public static let imEndID = 248046
    public static let eosIDs: Set<Int> = [imEndID, endOfTextID]
    public static let toolCallOpen = "<tool_call>"
    public static let toolCallClose = "</tool_call>"
    public static let toolCallOpenID = 248058
    public static let toolCallCloseID = 248059
    public static let toolResponseOpen = "<tool_response>"
    public static let toolResponseClose = "</tool_response>"
    public static let thinkingOpen = " thinking"
    public static let thinkingClose = "</think>"
}
