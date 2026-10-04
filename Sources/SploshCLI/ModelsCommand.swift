// ModelsCommand.swift — `splosh models`.
//
// The models splosh.toml registers, as the running server has them: which one is loaded, which
// artifacts are where the file says, and whether a request naming another has it loaded. It
// asks the server (`GET /v1/models`); with none on the port it reads the file. `--load` has the
// server load a model now (`POST /v1/models/load`), which is the only way one is loaded when
// `modelSwitch` is "manual".

import Foundation
import SploshServer

public enum ModelsCommand {
    public static let usage = """
        usage: splosh models [--json] [--port <n>] [--config <path>] [--load <id>]

        List the models the server can load (the `model.<id> = "<path>"` lines of splosh.toml):
        which is loaded, whether each artifact is there, and whether a request that names
        another model has it loaded. With no server on the port, the list is the file's.

          --json            the list as the server gives it (GET /v1/models)
          --port <n>        default the config's port, 8091 unless it says otherwise
          --config <path>   default ./splosh.toml
          --load <id>       have the server load this model in place of the one it has. It waits
                            for the requests in flight to finish, and for the model to answer
        """

    public static func run(_ tokens: [String]) -> Int32 {
        do {
            let args = try CLIArguments.parseModels(tokens)
            let config = try ServeConfig.resolve(path: args.config ?? "./splosh.toml", cliPort: args.port)
            if let id = args.load { return load(id, config: config, json: args.json) }
            guard let answer = request("GET", "/v1/models", config: config, timeout: 300),
                  answer.status == 200, let listed = try? JSONValue.parse(Array(answer.body)), listed["data"]?.arrayValue != nil else {
                let listed = registered(in: config)
                if args.json {
                    print(listed.compact)
                } else {
                    print("no server is answering on port \(config.port); splosh.toml registers:\n")
                    print(table(listed))
                    print("\nit would start on \((try? config.startingModel().model.id) ?? config.registry[0].id), with switching on request "
                          + (config.modelSwitch == "request" ? "on" : "off (modelSwitch = \"manual\")"))
                    if let hint = missingHint(listed) { print(hint) }
                }
                return ExitStatus.ok
            }
            if args.json {
                print(String(decoding: answer.body, as: UTF8.self))
            } else {
                print(table(listed))
                print("\n" + summary(listed))
                if let hint = missingHint(listed) { print(hint) }
            }
            return ExitStatus.ok
        } catch {
            SploshCLI.writeStderr("models failed: \(error)\n")
            return 1
        }
    }

    /// `--load`: the server answers once the model is loaded, or cannot be.
    private static func load(_ id: String, config: ServeConfig, json: Bool) -> Int32 {
        let body = JSONValue.object([("model", .string(id))]).compact
        // As long as a switch may take: the wait for requests in flight, a save and a load.
        guard let answer = request("POST", "/v1/models/load", config: config, body: Data(body.utf8), timeout: 900) else {
            SploshCLI.writeStderr("no splosh server is answering on port \(config.port); start one with `splosh serve --model \(id)`\n")
            return 1
        }
        let value = try? JSONValue.parse(Array(answer.body))
        if json { print(String(decoding: answer.body, as: UTF8.self)) }
        guard answer.status == 200, let loaded = value?["loaded"]?.stringValue else {
            let message = value?["error"]?["message"]?.stringValue ?? value?["error"]?.stringValue ?? "the server answered \(answer.status)"
            SploshCLI.writeStderr("\(id) was not loaded: \(message)\n")
            return 1
        }
        if !json { print("\(loaded) is loaded") }
        return loaded == id ? ExitStatus.ok : 1
    }

    // MARK: The list

    /// The registry as the file gives it, in the shape the server lists it in, with none loaded.
    /// `setup` is a server's own list when it has no model: it is up to offer the downloads.
    static func registered(in config: ServeConfig, setup: Bool = false) -> JSONValue {
        let models = config.registry.map { entry -> JSONValue in
            let size = ModelCatalog.fileSize(entry.path)
            return .object([
                ("id", .string(entry.id)), ("object", .string("model")), ("owned_by", .string("splosh")),
                ("loaded", .bool(false)), ("state", .string(size == nil ? "missing" : "available")),
                ("path", .string(entry.path)), ("size_bytes", size.map(JSONValue.int) ?? .null),
            ])
        }
        return .object([
            ("object", .string("list")), ("data", .array(models)), ("loaded", .null), ("setup", .bool(setup)),
            ("switch", .object([("mode", .string(config.modelSwitch)), ("target", .null), ("phase", .null), ("parked", .int(0))])),
        ])
    }

    /// One line a model: the loaded one marked, then its state, its size and its artifact.
    static func table(_ listed: JSONValue) -> String {
        let models = listed["data"]?.arrayValue ?? []
        let width = max(5, models.map { $0["id"]?.stringValue?.count ?? 0 }.max() ?? 0)
        func pad(_ text: String, _ count: Int) -> String { text + String(repeating: " ", count: max(0, count - text.count)) }
        var lines = ["  " + pad("model", width) + "  " + pad("state", 9) + "  " + pad("size", 10) + "  artifact"]
        for model in models {
            let size = model["size_bytes"]?.intValue.map { String(format: "%.2f GiB", Double($0) / 1_073_741_824) } ?? "-"
            lines.append((model["loaded"]?.boolValue == true ? "* " : "  ") + pad(model["id"]?.stringValue ?? "?", width) + "  "
                         + pad(model["state"]?.stringValue ?? "?", 9) + "  " + pad(size, 10) + "  " + (model["path"]?.stringValue ?? ""))
        }
        return lines.joined(separator: "\n")
    }

    /// What the server says of switching: whether a request has a model loaded, and the switch
    /// it has in hand.
    static func summary(_ listed: JSONValue) -> String {
        if listed["setup"]?.boolValue == true {
            return "the server has no model yet: it is up to offer the downloads, on its page and with `splosh download`"
        }
        let loaded = listed["loaded"]?.stringValue ?? "the loaded model"
        var lines: [String]
        switch listed["switch"]?["mode"]?.stringValue {
        case "request": lines = ["switching on request: on (a request that names another registered model has it loaded in place of \(loaded))"]
        case "manual": lines = ["switching on request: off (modelSwitch = \"manual\"); `splosh models --load <id>` loads one"]
        case "none": lines = ["switching on request: off (this server runs without a process holding its port, so it keeps the model it started on)"]
        default: lines = []
        }
        let parked = listed["switch"]?["parked"]?.intValue ?? 0
        let kept = parked == 0 ? "" : " (\(parked) request\(parked == 1 ? "" : "s") kept until then)"
        if let target = listed["switch"]?["target"]?.stringValue {
            let phase: String
            switch listed["switch"]?["phase"]?.stringValue {
            case "dwell": phase = "\(loaded) has its turn first"
            case "waiting": phase = "new requests wait while \(loaded) finishes the ones it has"
            case "stopping": phase = "\(loaded) is being stopped"
            case "loading": phase = "it is being loaded"
            default: phase = "in hand"
            }
            lines.append("switching to \(target): \(phase)\(kept)")
        }
        return lines.joined(separator: "\n")
    }

    /// What fetches the models listed as missing, of those `splosh download` knows; nil when none is.
    static func missingHint(_ listed: JSONValue) -> String? {
        let library = (try? ModelLibrary.current()) ?? .builtIn
        let missing = (listed["data"]?.arrayValue ?? []).filter { $0["state"]?.stringValue == "missing" }.compactMap { $0["id"]?.stringValue }
            .filter { library.model($0) != nil }
        return missing.isEmpty ? nil : "missing: `splosh download <model>` fetches \(missing.joined(separator: ", "))"
    }

    // MARK: The server

    /// One request to the server on this machine; nil when nothing answers.
    static func request(_ method: String, _ path: String, config: ServeConfig, body: Data? = nil,
                                timeout: TimeInterval) -> (status: Int, body: Data)? {
        // The address the server listens on, or this machine's own where it listens on all.
        let host = ["", "0.0.0.0", "::", "[::]"].contains(config.host) ? "127.0.0.1" : config.host
        guard let url = URL(string: "http://\(host.contains(":") && !host.hasPrefix("[") ? "[\(host)]" : host):\(config.port)\(path)") else { return nil }
        var request = URLRequest(url: url, cachePolicy: .reloadIgnoringLocalCacheData, timeoutInterval: timeout)
        request.httpMethod = method
        if let body {
            request.httpBody = body
            request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        }
        let answer = Answer()
        let done = DispatchSemaphore(value: 0)
        let session = URLSession(configuration: .ephemeral)
        session.dataTask(with: request) { data, response, _ in
            if let response = response as? HTTPURLResponse { answer.set((response.statusCode, data ?? Data())) }
            done.signal()
        }.resume()
        done.wait()
        session.invalidateAndCancel()
        return answer.value
    }

    private final class Answer: @unchecked Sendable {
        private let lock = NSLock()
        private var stored: (status: Int, body: Data)?
        func set(_ answer: (status: Int, body: Data)) { lock.lock(); stored = answer; lock.unlock() }
        var value: (status: Int, body: Data)? { lock.lock(); defer { lock.unlock() }; return stored }
    }
}
