import Foundation
import Hummingbird
import HTTPTypes
import NIOCore
import SploshRuntime

public enum Routes {
    public static func make(service: any InferenceService, contextWindow: Int = Server.contextLength,
                            behindHolder: Bool = false, settings: SettingsService? = nil,
                            catalog: ModelCatalog? = nil) -> Router<BasicRequestContext> {
        let router = Router<BasicRequestContext>()
        if behindHolder { router.add(middleware: OneRequestPerConnection()) }
        if let settings { addSettings(settings, to: router) }
        router.get("health") { _, _ -> String in "ok" }
        // The whole of a reply is made before any of it is sent, so a request is in flight for
        // as long as its handler runs.
        let inFlight = InFlight()
        if let catalog {
            addModels(catalog, to: router, contextLength: contextWindow) { inFlight.value }
        } else {
            router.get("v1/models") { _, _ -> ModelsResponse in
                ModelsResponse(data: [ModelInfo(id: Server.modelID, context_length: contextWindow)])
            }
        }
        router.post("v1/chat/completions") { request, context -> Response in
            inFlight.enter()
            defer { inFlight.leave() }
            let chat = try await request.decode(as: ChatRequest.self, context: context)
            if let catalog, let refusal = catalog.refusal(catalog.gate(chat.model)) { return answered(refusal, by: catalog) }
            let inference = InferenceRequest(model: chat.model, messages: chat.messages.map { InferenceMessage(role: $0.role, content: $0.content) }, stream: chat.stream ?? true)
            let admitted: InferenceRequest
            do {
                admitted = try service.admit(inference, contextWindow: contextWindow)
            } catch InferenceServiceError.contextWindowExceeded {
                let body = ContextWindowExceededResponse(error: "CONTEXT_WINDOW_EXCEEDED")
                let encoded = try JSONEncoder().encode(body)
                return answered(Response(status: .badRequest, headers: [.contentType: "application/json"], body: ResponseBody { writer in
                    var writer = writer
                    try await writer.write(ByteBuffer(bytes: encoded))
                    try await writer.finish(nil)
                }), by: catalog)
            }
            let records = try await SsePipeline.encode(service.infer(admitted), model: chat.model)
            return answered(Response(status: .ok, headers: [.contentType: "text/event-stream"], body: SseWriter.body(records)), by: catalog)
        }
        return router
    }

    /// A chat response, naming the model that gave it.
    static func answered(_ response: Response, by catalog: ModelCatalog?) -> Response {
        guard let catalog else { return response }
        var response = response
        response.headers[ModelCatalog.modelHeader] = catalog.loaded
        return response
    }
}

/// Asks the client to use each connection once. Behind the process that holds the port, an
/// engine that stops closes its idle connections, and a request sent on one at that moment
/// would be lost; with no idle connections, every request arrives on a new one, which the
/// holder keeps until there is an engine to take it.
struct OneRequestPerConnection: RouterMiddleware {
    func handle(_ request: Request, context: BasicRequestContext,
                next: (Request, BasicRequestContext) async throws -> Response) async throws -> Response {
        var response = try await next(request, context)
        response.headers[.connection] = "close"
        return response
    }
}

extension Routes {
    /// The production router: the model-backed OpenAI-compatible API, live stats, the dashboard
    /// and the settings page.
    public static func make(backend: ChatBackend, behindHolder: Bool = false, settings: SettingsService? = nil,
                            catalog: ModelCatalog? = nil) -> Router<BasicRequestContext> {
        let router = Router<BasicRequestContext>()
        if behindHolder { router.add(middleware: OneRequestPerConnection()) }
        if let settings { addSettings(settings, to: router) }
        router.get("health") { _, _ -> String in "ok" }
        // In flight: a request being read and prepared here, then one the scheduler has.
        let arriving = InFlight()
        if let catalog {
            addModels(catalog, to: router, contextLength: backend.maxContext) { arriving.value + backend.scheduler.requestsInFlight }
        } else {
            let started = Int(Date().timeIntervalSince1970)
            let model = JSONValue.object([
                ("id", .string(backend.modelID)), ("object", .string("model")), ("created", .int(started)),
                ("owned_by", .string("splosh")), ("context_length", .int(backend.maxContext)),
            ])
            router.get("v1/models") { _, _ -> Response in
                ChatEndpoint.json(.object([("object", .string("list")), ("data", .array([model]))]))
            }
            router.get("v1/models/:id") { _, _ -> Response in ChatEndpoint.json(model) }
        }
        router.post("v1/chat/completions") { request, _ -> Response in
            arriving.enter()
            let admitted: ChatEndpoint.Admitted
            do {
                // Until the scheduler has it, and counts it: there is no moment between the two
                // at which a request in hand is counted by neither.
                defer { arriving.leave() }
                let buffer = try await request.body.collect(upTo: 256 * 1024 * 1024)
                admitted = ChatEndpoint.admit(Array(buffer.readableBytesView), backend: backend, catalog: catalog)
            }
            return answered(await ChatEndpoint.respond(to: admitted, backend: backend), by: catalog)
        }
        router.get("v1/stats") { _, _ -> Response in
            var stats = backend.scheduler.stats()
            if let catalog, let model = catalog.models.first(where: { $0.id == catalog.loaded }) {
                stats.model = LoadedModelStats(id: model.id, path: model.path, switching: catalog.switching()?.target != nil)
            }
            let data = try JSONEncoder().encode(stats)
            return Response(status: .ok, headers: [.contentType: "application/json", .cacheControl: "no-store"],
                            body: ResponseBody(byteBuffer: ByteBuffer(bytes: data)))
        }
        router.get("v1/sessions/:id/reply") { _, context -> Response in
            let id = context.parameters.get("id", as: Int.self)
            return ChatEndpoint.reply(of: id, id.flatMap(backend.scheduler.activity(of:)))
        }
        router.get("/") { _, _ -> Response in
            Response(status: .ok, headers: [.contentType: "text/html; charset=utf-8"],
                     body: ResponseBody(byteBuffer: ByteBuffer(string: Dashboard.html)))
        }
        return router
    }
}

struct ModelsResponse: ResponseEncodable { let data: [ModelInfo] }
struct ModelInfo: Encodable { let id: String; let context_length: Int }
