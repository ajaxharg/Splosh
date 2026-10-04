import Foundation
import Hummingbird
import HTTPTypes
import NIOCore

/// The models a server can load, and the one this engine has. The command that runs the server
/// supplies it: the registry is splosh.toml's, and an engine holds one model for as long as it
/// lives, so loading another is the business of the process that holds the port, which starts
/// a new engine in this one's place.
///
/// A request that names a registered model this engine does not have is answered with
/// `switchStatus` and the model's name in `switchHeader`, before anything else. The holder
/// reads that much of an engine's answer; it keeps the request, has the model loaded, and gives
/// the request to the engine that has it. The client sees a reply that took longer.
public struct ModelCatalog: Sendable {
    public struct Model: Sendable, Equatable {
        public let id: String
        public let path: String
        public init(id: String, path: String) { self.id = id; self.path = path }
    }

    /// When a request for a registered model that is not loaded has it loaded: on `request`,
    /// only when asked (`manual`, by `POST /v1/models/load`), or never (`none`: no process
    /// holds the port to start another engine).
    public enum Mode: String, Sendable { case request, manual, none }

    /// A switch the process holding the port has in hand.
    public struct Switching: Sendable, Equatable {
        /// The model to be loaded, and how far that has got; nil when none is.
        public var target: String?
        public var phase: String?
        /// Requests kept until the model they name is loaded.
        public var parked: Int
        public init(target: String? = nil, phase: String? = nil, parked: Int = 0) {
            self.target = target; self.phase = phase; self.parked = parked
        }
    }

    /// What becomes of a request, by the model it names.
    public enum Gate: Equatable, Sendable {
        /// The loaded model answers it: it names that one, none, or none that is registered.
        case serve
        /// The model is to be loaded first.
        case load(String)
        /// The model is registered and its artifact is not where the registry says.
        case missing(Model)
        /// The model is not loaded, and a request does not have it loaded here.
        case notLoaded(String)
    }

    /// The registered models, in the order splosh.toml gives them.
    public var models: [Model]
    /// The id of the one this engine has loaded.
    public var loaded: String
    public var mode: Mode
    /// The size of the file at a path; nil when there is none.
    public var size: @Sendable (String) -> Int?
    /// The switch under way, as the process holding the port reports it; nil when none is.
    public var switching: @Sendable () -> Switching?

    public init(models: [Model], loaded: String, mode: Mode,
                size: @escaping @Sendable (String) -> Int? = ModelCatalog.fileSize,
                switching: @escaping @Sendable () -> Switching? = { nil }) {
        self.models = models; self.loaded = loaded; self.mode = mode
        self.size = size; self.switching = switching
    }

    /// The status and headers the holder knows a wanted switch by, and the header every chat
    /// response names the answering model in.
    public static let switchStatus = HTTPResponse.Status.misdirectedRequest
    public static let switchHeader = HTTPField.Name("x-splosh-switch")!
    /// With `switchHeader`: asked for by hand, so not held back for the loaded model's turn.
    public static let switchAdminHeader = HTTPField.Name("x-splosh-switch-admin")!
    public static let modelHeader = HTTPField.Name("x-splosh-model")!

    public static let fileSize: @Sendable (String) -> Int? = { path in
        let expanded = (path as NSString).expandingTildeInPath
        var isDirectory: ObjCBool = false
        guard FileManager.default.fileExists(atPath: expanded, isDirectory: &isDirectory), !isDirectory.boolValue else { return nil }
        return ((try? FileManager.default.attributesOfItem(atPath: expanded))?[.size] as? NSNumber)?.intValue ?? 0
    }

    public func gate(_ name: String?) -> Gate {
        guard let name, name != loaded, let model = models.first(where: { $0.id == name }) else { return .serve }
        guard size(model.path) != nil else { return .missing(model) }
        return mode == .request ? .load(name) : .notLoaded(name)
    }

    /// The same for a load asked for by hand, which `manual` allows.
    public func gateLoad(_ name: String) -> Gate {
        guard let model = models.first(where: { $0.id == name }) else { return .missing(Model(id: name, path: "")) }
        if name == loaded { return .serve }
        guard size(model.path) != nil else { return .missing(model) }
        return mode == .none ? .notLoaded(name) : .load(name)
    }

    /// The answer to a request the loaded model is not to serve; nil for one it is.
    func refusal(_ gate: Gate, admin: Bool = false) -> Response? {
        switch gate {
        case .serve:
            return nil
        case .load(let id):
            // For the holder, which gives the client none of it.
            let error = ChatEndpoint.APIError(status: Self.switchStatus, type: "server_error", code: "model_switching",
                                              message: "the model \(id) is being loaded in place of \(loaded); send the request again")
            var response = ChatEndpoint.json(error.body, status: error.status)
            response.headers[Self.switchHeader] = id
            if admin { response.headers[Self.switchAdminHeader] = "1" }
            return response
        case .missing(let model):
            let message = model.path.isEmpty ? "no model called \(model.id) is registered"
                                             : "the model \(model.id) is registered, but there is no artifact at \(model.path)"
            let error = ChatEndpoint.APIError(status: .notFound, type: "invalid_request_error", code: "model_not_found", message: message)
            return ChatEndpoint.json(error.body, status: error.status)
        case .notLoaded(let id):
            let how = mode == .none ? "this server has no process holding its port to load another: stop it and start it with `--model \(id)`"
                                    : "this server loads another only when asked: `splosh models --load \(id)`"
            let error = ChatEndpoint.APIError(status: .conflict, type: "invalid_request_error", code: "model_not_loaded",
                                              message: "the model \(id) is not loaded (\(loaded) is), and \(how)")
            return ChatEndpoint.json(error.body, status: error.status)
        }
    }

    // MARK: Listing

    func state(of model: Model) -> (state: String, size: Int?) {
        let bytes = size(model.path)
        return (model.id == loaded ? "loaded" : bytes != nil ? "available" : "missing", bytes)
    }

    /// A model as `GET /v1/models` lists it: the OpenAI object, and where it stands here.
    func object(_ model: Model, created: Int, contextLength: Int) -> JSONValue {
        let (state, bytes) = state(of: model)
        return .object([
            ("id", .string(model.id)), ("object", .string("model")), ("created", .int(created)),
            ("owned_by", .string("splosh")), ("context_length", .int(contextLength)),
            ("loaded", .bool(model.id == loaded)), ("state", .string(state)),
            ("path", .string(model.path)), ("size_bytes", bytes.map(JSONValue.int) ?? .null),
        ])
    }

    /// The models with the loaded one first and the rest as registered, so that a client that
    /// takes the first of the list names the model in memory and causes no switch.
    var listed: [Model] { models.filter { $0.id == loaded } + models.filter { $0.id != loaded } }

    func list(created: Int, contextLength: Int) -> JSONValue {
        let switching = self.switching() ?? Switching()
        return .object([
            ("object", .string("list")),
            ("data", .array(listed.map { object($0, created: created, contextLength: contextLength) })),
            ("loaded", .string(loaded)),
            ("switch", .object([
                ("mode", .string(mode.rawValue)),
                ("target", switching.target.map(JSONValue.string) ?? .null),
                ("phase", switching.phase.map(JSONValue.string) ?? .null),
                ("parked", .int(switching.parked)),
            ])),
        ])
    }
}

/// Counts the requests a router has in hand, for `GET /v1/busy`.
final class InFlight: @unchecked Sendable {
    private let lock = NSLock()
    private var count = 0

    func enter() { lock.lock(); count += 1; lock.unlock() }
    func leave() { lock.lock(); count -= 1; lock.unlock() }
    var value: Int { lock.lock(); defer { lock.unlock() }; return count }
}

extension Routes {
    /// The model routes: the list, one model, what the engine has in flight, and a load asked
    /// for by hand. `inFlight` is the number of chat requests not yet answered in full.
    static func addModels(_ catalog: ModelCatalog, to router: Router<BasicRequestContext>, contextLength: Int,
                          inFlight: @escaping @Sendable () -> Int) {
        let started = Int(Date().timeIntervalSince1970)
        @Sendable func refused(_ status: HTTPResponse.Status, _ type: String, _ code: String?, _ message: String) -> Response {
            ChatEndpoint.json(ChatEndpoint.APIError(status: status, type: type, code: code, message: message).body, status: status)
        }
        router.get("v1/models") { _, _ -> Response in
            var response = ChatEndpoint.json(catalog.list(created: started, contextLength: contextLength))
            response.headers[.cacheControl] = "no-store"
            return response
        }
        // A name that is not registered is the loaded model's, as every name was before there
        // was a registry: clients ask for the one they were configured with.
        router.get("v1/models/:id") { _, context -> Response in
            let id = context.parameters.get("id")
            let model = catalog.models.first { $0.id == id } ?? catalog.models.first { $0.id == catalog.loaded }
                ?? ModelCatalog.Model(id: catalog.loaded, path: "")
            return ChatEndpoint.json(catalog.object(model, created: started, contextLength: contextLength))
        }
        // For the holder: at 0 this engine can be stopped with nothing cut.
        router.get("v1/busy") { _, _ -> Response in
            var response = ChatEndpoint.json(.object([("busy", .int(inFlight())), ("model", .string(catalog.loaded))]))
            response.headers[.cacheControl] = "no-store"
            return response
        }
        // `splosh models --load`. The engine that has the model answers it: asked of another,
        // it is a wanted switch like a chat request's, kept by the holder until that engine is up.
        router.post("v1/models/load") { request, _ -> Response in
            if let reason = changeRefusal(request) {
                return refused(.forbidden, "invalid_request_error", nil, "a model is loaded as the settings are changed: \(reason)")
            }
            let buffer = try await request.body.collect(upTo: 1 << 20)
            guard let id = (try? JSONValue.parse(Array(buffer.readableBytesView)))?["model"]?.stringValue else {
                return refused(.badRequest, "invalid_request_error", nil, "expected a JSON object naming the model: {\"model\": \"<id>\"}")
            }
            if let refusal = catalog.refusal(catalog.gateLoad(id), admin: true) { return refusal }
            return ChatEndpoint.json(.object([("loaded", .string(catalog.loaded))]))
        }
    }
}
