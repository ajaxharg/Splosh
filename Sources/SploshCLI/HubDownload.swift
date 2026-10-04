// HubDownload.swift — one file from Hugging Face: resumed where it stopped, and checked.
//
// A file arrives in `<name>.part` beside where it is to go and is renamed once its size and
// SHA-256 are the pinned ones. A part left by a download that was stopped (or lost its
// connection) is carried on from with a Range request; its bytes are hashed first, so the whole
// file is hashed once however many times the download was taken up again.

import CryptoKit
import Foundation

/// Why a download or an install did not finish.
struct DownloadError: Error, CustomStringConvertible, Equatable {
    let message: String
    /// It was asked to stop: nothing went wrong.
    var cancelled = false

    init(_ message: String, cancelled: Bool = false) { self.message = message; self.cancelled = cancelled }
    var description: String { message }

    static let stopped = DownloadError("stopped", cancelled: true)
}

/// A stop asked for while something long is under way.
final class Cancellation: @unchecked Sendable {
    private let lock = NSLock()
    private var cancelled = false
    private var atOnce: (@Sendable () -> Void)?

    var isCancelled: Bool { lock.lock(); defer { lock.unlock() }; return cancelled }

    func cancel() {
        lock.lock()
        cancelled = true
        let atOnce = self.atOnce
        lock.unlock()
        atOnce?()
    }

    func check() throws { if isCancelled { throw DownloadError.stopped } }

    /// What a stop does at once, where the work under way cannot be asked to end (a conversion):
    /// nil when it can, and checks for itself.
    func onCancel(_ action: (@Sendable () -> Void)?) { lock.lock(); atOnce = action; lock.unlock() }
}

struct HubDownload: Sendable {
    /// Hugging Face, or a mirror of it (see ModelLibrary.endpoint).
    var endpoint: String
    /// Attempts in a row that bring nothing before a download is given up.
    var patience = 5
    /// Seconds without a byte before a connection is taken as lost.
    var idleTimeout: TimeInterval = 60

    func url(of name: String, in source: ModelLibrary.Source) -> URL? {
        let path = name.split(separator: "/").map { $0.addingPercentEncoding(withAllowedCharacters: .urlPathAllowed) ?? String($0) }.joined(separator: "/")
        return URL(string: "\(endpoint)/\(source.repo)/resolve/\(source.revision)/\(path)")
    }

    /// The SHA-256 of a file, as lowercase hex. `progress` is told the bytes read so far.
    static func sha256(of url: URL, cancellation: Cancellation? = nil, progress: ((Int) -> Void)? = nil) throws -> String {
        let handle = try FileHandle(forReadingFrom: url)
        defer { try? handle.close() }
        var hasher = SHA256()
        var read = 0
        while let chunk = try handle.read(upToCount: 4 << 20), !chunk.isEmpty {
            try cancellation?.check()
            hasher.update(data: chunk)
            read += chunk.count
            progress?(read)
        }
        return hex(hasher.finalize())
    }

    static func hex(_ digest: SHA256.Digest) -> String { digest.map { String(format: "%02x", $0) }.joined() }

    /// Have `file` at `destination`, from `source`. `progress` is told how many of the file's
    /// bytes are on disk, and `event` what happened on the way that is worth a line.
    func fetch(_ file: ModelLibrary.File, from source: ModelLibrary.Source, to destination: URL, cancellation: Cancellation,
               progress: @escaping @Sendable (Int) -> Void, event: @escaping @Sendable (String) -> Void = { _ in }) throws {
        guard let url = url(of: file.name, in: source) else { throw DownloadError("\(file.name) of \(source.repo) cannot be made into an address") }
        let part = try PartFile(url: URL(fileURLWithPath: destination.path + ".part"), limit: file.bytes, cancellation: cancellation, progress: progress)
        defer { part.close() }
        if part.bytes > 0, part.bytes < file.bytes {
            event("carrying on from \(Human.bytes(part.bytes)) of \(file.name) already here")
        }

        let configuration = URLSessionConfiguration.ephemeral
        configuration.timeoutIntervalForRequest = idleTimeout
        configuration.timeoutIntervalForResource = 7 * 24 * 3600
        configuration.urlCache = nil
        configuration.requestCachePolicy = .reloadIgnoringLocalCacheData
        var fruitless = 0, restarts = 0
        while part.bytes < file.bytes {
            try cancellation.check()
            let before = part.bytes
            let attempt = Attempt(part: part, cancellation: cancellation, progress: progress)
            let session = URLSession(configuration: configuration, delegate: attempt, delegateQueue: nil)
            var request = URLRequest(url: url)
            request.setValue("splosh (model download)", forHTTPHeaderField: "User-Agent")
            // As stored: a range of an encoded body is not a range of the file.
            request.setValue("identity", forHTTPHeaderField: "Accept-Encoding")
            if before > 0 { request.setValue("bytes=\(before)-", forHTTPHeaderField: "Range") }
            let task = session.dataTask(with: request)
            task.resume()
            while attempt.done.wait(timeout: .now() + 0.25) == .timedOut {
                if cancellation.isCancelled { task.cancel() }
            }
            session.invalidateAndCancel()
            try cancellation.check()

            var lost: String?
            switch attempt.outcome {
            case .finished:
                // An answer that ended short of the file is a connection lost like any other.
                if part.bytes < file.bytes { lost = "the answer ended early" }
            case .refused(let status):
                let why = status == 404 ? "there is no such file at that revision" : status == 401 || status == 403 ? "it is not open to download" : "the request was refused"
                throw DownloadError("\(url.absoluteString) answered \(status): \(why)")
            case .restart(let why):
                restarts += 1
                guard restarts <= 2 else { throw DownloadError("\(file.name) could not be carried on with, nor begun again: \(why)") }
                event("\(file.name): \(why); beginning it again")
                try part.reset()
                continue
            case .failed(let why):
                lost = why
            }
            guard let lost else { continue }
            fruitless = part.bytes > before ? 0 : fruitless + 1
            guard fruitless < patience else {
                throw DownloadError("\(file.name) stopped at \(Human.bytes(part.bytes)) of \(Human.bytes(file.bytes)) after \(patience) tries: \(lost). What has arrived is kept, and the download carries on from there when it is asked for again")
            }
            let pause = min(30, 1 << fruitless)
            event("\(file.name): \(lost); trying again in \(pause) s from \(Human.bytes(part.bytes))")
            for _ in 0..<(pause * 4) { try cancellation.check(); usleep(250_000) }
        }
        guard part.bytes == file.bytes else {
            try? part.reset()
            throw DownloadError("\(file.name) came to \(part.bytes) bytes where the pinned file has \(file.bytes)")
        }
        let digest = part.digest()
        part.close()
        guard digest == file.sha256 else {
            try? FileManager.default.removeItem(at: part.url)
            throw DownloadError("\(file.name) arrived whole but is not the pinned file: its SHA-256 is \(digest), not \(file.sha256). It has been removed")
        }
        if (try? FileManager.default.destinationOfSymbolicLink(atPath: destination.path)) != nil || FileManager.default.fileExists(atPath: destination.path) {
            try FileManager.default.removeItem(at: destination)
        }
        try FileManager.default.moveItem(at: part.url, to: destination)
    }
}

/// The part of a file that has arrived: its bytes on disk, and their hash so far.
private final class PartFile: @unchecked Sendable {
    let url: URL
    /// The size of the whole file.
    let limit: Int
    private var handle: FileHandle?
    private var hasher = SHA256()
    private(set) var bytes = 0

    /// Opens the part at `url`, hashing what is there; a part longer than the file is to be is begun again.
    init(url: URL, limit: Int, cancellation: Cancellation, progress: (Int) -> Void) throws {
        self.url = url
        self.limit = limit
        let manager = FileManager.default
        try manager.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        if !manager.fileExists(atPath: url.path) { manager.createFile(atPath: url.path, contents: nil) }
        let handle = try FileHandle(forUpdating: url)
        self.handle = handle
        let size = Int(try handle.seekToEnd())
        guard size > 0, size <= limit else { try reset(); return }
        try handle.seek(toOffset: 0)
        while bytes < size {
            try cancellation.check()
            guard let chunk = try handle.read(upToCount: min(4 << 20, size - bytes)), !chunk.isEmpty else { break }
            hasher.update(data: chunk)
            bytes += chunk.count
            progress(bytes)
        }
        try handle.truncate(atOffset: UInt64(bytes))
        try handle.seekToEnd()
    }

    func append(_ data: Data) throws {
        guard let handle else { throw DownloadError("\(url.lastPathComponent) is closed") }
        do {
            try handle.write(contentsOf: data)
        } catch {
            // Some of it may have been written (a full disk): the file is put back to what was
            // hashed, so that what arrives next goes where it belongs.
            try? handle.truncate(atOffset: UInt64(bytes))
            try? handle.seekToEnd()
            throw error
        }
        hasher.update(data: data)
        bytes += data.count
    }

    func reset() throws {
        try handle?.truncate(atOffset: 0)
        try handle?.seek(toOffset: 0)
        hasher = SHA256()
        bytes = 0
    }

    func digest() -> String { HubDownload.hex(hasher.finalize()) }

    func close() {
        try? handle?.synchronize()
        try? handle?.close()
        handle = nil
    }
}

/// One request for the rest of a file.
private final class Attempt: NSObject, URLSessionDataDelegate, @unchecked Sendable {
    enum Outcome {
        case finished
        /// The connection was lost, or the server could not answer for now.
        case failed(String)
        /// The server will not send this file.
        case refused(Int)
        /// The server would not send the rest: the file is to be fetched from its start.
        case restart(String)
    }

    let done = DispatchSemaphore(value: 0)
    private let part: PartFile
    private let cancellation: Cancellation
    private let progress: @Sendable (Int) -> Void
    private var decided: Outcome?

    var outcome: Outcome { decided ?? .failed("the connection ended") }

    init(part: PartFile, cancellation: Cancellation, progress: @escaping @Sendable (Int) -> Void) {
        self.part = part; self.cancellation = cancellation; self.progress = progress
    }

    func urlSession(_ session: URLSession, dataTask: URLSessionDataTask, didReceive response: URLResponse,
                    completionHandler: @escaping @Sendable (URLSession.ResponseDisposition) -> Void) {
        guard let http = response as? HTTPURLResponse else {
            decided = .failed("the answer was not HTTP")
            return completionHandler(.cancel)
        }
        switch http.statusCode {
        case 206:
            // The rest, from where the part ends.
            let range = http.value(forHTTPHeaderField: "Content-Range") ?? ""
            guard range.hasPrefix("bytes \(part.bytes)-") else {
                decided = .restart("the server sent another range than the one asked for (\(range))")
                return completionHandler(.cancel)
            }
        case 200:
            // The whole file, whatever was asked for.
            if part.bytes > 0 {
                do { try part.reset() } catch {
                    decided = .failed("the part could not be begun again: \(error.localizedDescription)")
                    return completionHandler(.cancel)
                }
            }
        case 416:
            decided = .restart("the server has no such range")
            return completionHandler(.cancel)
        case 408, 429, 500...599:
            decided = .failed("the server answered \(http.statusCode)")
            return completionHandler(.cancel)
        default:
            decided = .refused(http.statusCode)
            return completionHandler(.cancel)
        }
        completionHandler(.allow)
    }

    func urlSession(_ session: URLSession, dataTask: URLSessionDataTask, didReceive data: Data) {
        guard decided == nil else { return }
        if cancellation.isCancelled { return dataTask.cancel() }
        guard part.bytes + data.count <= part.limit else {
            decided = .restart("more arrived than the pinned file has")
            return dataTask.cancel()
        }
        do {
            try part.append(data)
            progress(part.bytes)
        } catch {
            decided = .failed("could not write \(part.url.path): \(error.localizedDescription)")
            dataTask.cancel()
        }
    }

    func urlSession(_ session: URLSession, task: URLSessionTask, didCompleteWithError error: (any Error)?) {
        if decided == nil { decided = error.map { .failed($0.localizedDescription) } ?? .finished }
        done.signal()
    }
}
