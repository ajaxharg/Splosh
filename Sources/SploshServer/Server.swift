import Hummingbird
import SploshRuntime

public enum Server {
    public static let modelID = "qwen3.8-27b"
    public static let contextLength = 262_144
    public static func application(service: any InferenceService, port: Int = 8091, contextWindow: Int = contextLength,
                                   socketPath: String? = nil, settings: SettingsService? = nil,
                                   catalog: ModelCatalog? = nil, downloads: DownloadService? = nil) -> Application<RouterResponder<BasicRequestContext>> {
        let router = Routes.make(service: service, contextWindow: contextWindow, behindHolder: socketPath != nil, settings: settings, catalog: catalog,
                                 downloads: downloads)
        return Application(responder: router.buildResponder(), configuration: .init(address: address("127.0.0.1", port, socketPath)))
    }

    /// The model-backed application. `host` defaults to loopback. With `socketPath` it listens
    /// on that Unix-domain socket instead: the engine behind the process that holds the port.
    /// `catalog` names the model it has among those the server can load (see ModelCatalog), and
    /// `downloads` the ones that can be fetched (see DownloadService).
    public static func application(backend: ChatBackend, host: String = "127.0.0.1", port: Int = 8091,
                                   socketPath: String? = nil, settings: SettingsService? = nil,
                                   catalog: ModelCatalog? = nil, downloads: DownloadService? = nil) -> Application<RouterResponder<BasicRequestContext>> {
        let router = Routes.make(backend: backend, behindHolder: socketPath != nil, settings: settings, catalog: catalog, downloads: downloads)
        return Application(responder: router.buildResponder(), configuration: .init(address: address(host, port, socketPath)))
    }

    /// The application of a server that has no model yet (see Routes.makeSetup): it offers the
    /// downloads. `models` is its answer to `GET /v1/models`.
    public static func setupApplication(host: String = "127.0.0.1", port: Int = 8091, socketPath: String? = nil,
                                        settings: SettingsService? = nil, downloads: DownloadService,
                                        models: @escaping @Sendable () -> JSONValue) -> Application<RouterResponder<BasicRequestContext>> {
        let router = Routes.makeSetup(downloads: downloads, settings: settings, behindHolder: socketPath != nil, models: models)
        return Application(responder: router.buildResponder(), configuration: .init(address: address(host, port, socketPath)))
    }

    private static func address(_ host: String, _ port: Int, _ socketPath: String?) -> BindAddress {
        socketPath.map { .unixDomainSocket(path: $0) } ?? .hostname(host, port: port)
    }
}
