import Foundation
import Hummingbird
import NIOCore

public enum SseWriter {
    public static func body(_ frames: [Data]) -> ResponseBody {
        ResponseBody { writer in
            var writer = writer
            for frame in frames { try await writer.write(ByteBuffer(bytes: frame)) }
            try await writer.finish(nil)
        }
    }
}
