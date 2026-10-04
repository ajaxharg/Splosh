import Foundation
import Hummingbird
import HTTPTypes
import NIOCore

/// The models that can be downloaded, and the install under way. The command that runs the
/// server supplies it: an install is a process of its own (`splosh download`), so that a
/// conversion never shares a process with the engine, and this is the server's view of it.
public struct DownloadService: Sendable {
    /// What there is to download and how each model stands, in the shape `DownloadsPanel` draws.
    public var read: @Sendable () -> JSONValue
    /// Start the install of a model. Returns the same as `read`, or throws `DownloadRefusal`.
    public var start: @Sendable (String) throws -> JSONValue
    /// Stop the install of a model; what it has downloaded is kept.
    public var cancel: @Sendable (String) throws -> JSONValue

    public init(read: @escaping @Sendable () -> JSONValue, start: @escaping @Sendable (String) throws -> JSONValue,
                cancel: @escaping @Sendable (String) throws -> JSONValue) {
        self.read = read; self.start = start; self.cancel = cancel
    }
}

/// An install that was not started or stopped, and why, for the page to show.
public struct DownloadRefusal: Error, Equatable, Sendable {
    /// The HTTP status it is answered with.
    public let status: Int
    public let message: String
    public init(status: Int, message: String) { self.status = status; self.message = message }
}

extension Routes {
    static func addDownloads(_ downloads: DownloadService, to router: Router<BasicRequestContext>) {
        @Sendable func refused(_ status: HTTPResponse.Status, _ message: String) -> Response {
            ChatEndpoint.json(.object([("error", .string(message))]), status: status)
        }
        // A change, taken as the settings' are: from a page this server served to this machine.
        @Sendable func change(_ request: Request, _ act: @Sendable (String) throws -> JSONValue, status: HTTPResponse.Status) async throws -> Response {
            if let reason = changeRefusal(request) { return refused(.forbidden, reason) }
            let buffer = try await request.body.collect(upTo: 1 << 20)
            guard let id = (try? JSONValue.parse(Array(buffer.readableBytesView)))?["model"]?.stringValue else {
                return refused(.badRequest, "expected a JSON object naming the model: {\"model\": \"<id>\"}")
            }
            do {
                return ChatEndpoint.json(try act(id), status: status)
            } catch let refusal as DownloadRefusal {
                return refused(HTTPResponse.Status(code: refusal.status), refusal.message)
            }
        }
        router.get("v1/downloads") { _, _ -> Response in
            var response = ChatEndpoint.json(downloads.read())
            response.headers[.cacheControl] = "no-store"
            return response
        }
        router.post("v1/downloads") { request, _ -> Response in try await change(request, downloads.start, status: .accepted) }
        router.post("v1/downloads/cancel") { request, _ -> Response in try await change(request, downloads.cancel, status: .ok) }
    }

    /// The router of a server that has no model yet: the page that offers the downloads, the
    /// settings, and for everything that needs a model an answer saying that there is none.
    /// `models` is the answer to `GET /v1/models`.
    public static func makeSetup(downloads: DownloadService, settings: SettingsService? = nil, behindHolder: Bool = false,
                                 models: @escaping @Sendable () -> JSONValue) -> Router<BasicRequestContext> {
        let router = Router<BasicRequestContext>()
        if behindHolder { router.add(middleware: OneRequestPerConnection()) }
        if let settings { addSettings(settings, to: router) }
        addDownloads(downloads, to: router)
        router.get("health") { _, _ -> String in "ok" }
        router.get("/") { _, _ -> Response in
            Response(status: .ok, headers: [.contentType: "text/html; charset=utf-8", .cacheControl: "no-store"],
                     body: ResponseBody(byteBuffer: ByteBuffer(string: SetupPage.html)))
        }
        router.get("v1/models") { _, _ -> Response in
            var response = ChatEndpoint.json(models())
            response.headers[.cacheControl] = "no-store"
            return response
        }
        // For the process that holds the port: nothing is ever in flight here.
        router.get("v1/busy") { _, _ -> Response in ChatEndpoint.json(.object([("busy", .int(0)), ("model", .null)])) }
        let none: @Sendable (Request, BasicRequestContext) async throws -> Response = { _, _ in
            let error = ChatEndpoint.APIError(status: .serviceUnavailable, type: "server_error", code: "model_not_installed",
                                              message: "no model is installed yet: open this server's address in a browser to download one, or run `splosh download`")
            return ChatEndpoint.json(error.body, status: error.status)
        }
        router.post("v1/chat/completions", use: none)
        router.post("v1/models/load", use: none)
        router.get("v1/models/:id", use: none)
        router.get("v1/stats", use: none)
        return router
    }
}

/// The part of a page that lists the models to download and shows an install going on: the
/// first-launch page is little else, and the settings page has it as a card. A page that uses it
/// has `$`, `esc` and `send` (as the settings page defines them) and an element with id "dl";
/// it calls `dlPoll()` once, and may set `DL.onChange` to hear of a model installed.
enum DownloadsPanel {
    static let style = #"""
.dlm{padding:12px 0;border-bottom:1px solid var(--line)}.dlm:first-child{padding-top:2px}
.dlh{display:flex;justify-content:space-between;align-items:center;gap:12px;margin-bottom:3px}.dlh button{flex:none}
.dlsteps{margin-top:10px}.dls{display:flex;gap:8px;padding:3px 0}.dls>div{min-width:0;flex:1}.dlk{width:1.1em;flex:none;color:var(--mute);text-align:center}
.dls.done .dlk{color:var(--b)}.dls.running .dlk{color:var(--a)}.dls.failed .dlk{color:var(--e)}.dls.waiting{opacity:.55}
.dlbar{height:6px;border-radius:3px;background:var(--line);margin:6px 0 3px;overflow:hidden}.dlbar i{display:block;height:100%;background:var(--a);transition:width .4s}
.pill.loaded,.pill.installed{color:var(--b);border-color:var(--b)}.pill.installing,.pill.first{color:var(--a);border-color:var(--a)}
.own{padding-top:12px}.own table{border-collapse:collapse;margin-top:4px}.own td{padding:1px 12px 1px 0;vertical-align:top;font-size:12px;color:var(--mute);overflow-wrap:anywhere}
.own td:first-child{white-space:nowrap}
code{font:12px ui-monospace,SFMono-Regular,Menlo,monospace}
"""#

    static let script = #"""
// The models that can be downloaded (/v1/downloads): what there is, what is here, and the install under way.
const DL={d:null,msg:null,busy:false,key:null,timer:null,turn:0,onChange:null};
const dsz=b=>b>=1e9?(b/1e9).toFixed(b<1e10?2:1)+' GB':b>=1e6?(b/1e6).toFixed(0)+' MB':(b/1e3).toFixed(0)+' kB';
const drate=r=>r>=1e9?(r/1e9).toFixed(2)+' GB/s':(r/1e6).toFixed(r<1e7?1:0)+' MB/s';
const ddur=s=>s<90?Math.round(s)+' s':s<5400?Math.round(s/60)+' min':(s/3600).toFixed(1)+' h';
const dmark={done:'✓',skipped:'–',running:'●',failed:'✕',cancelled:'■',waiting:'○'};
function dlStep(s){let h=`<div class="dls ${esc(s.state)}"><span class="dlk">${dmark[s.state]||'○'}</span><div><b>${esc(s.title)}</b> <span class="help">${esc(s.detail)}</span>`;
 if(s.state==='running'&&s.total){const p=Math.min(100,s.done/s.total*100);
  h+=`<div class="dlbar"><i style="width:${p}%"></i></div><div class="note">${s.verb&&s.verb!=='downloading'?esc(s.verb)+' ':''}${s.item?esc(s.item)+' · ':''}${p.toFixed(0)}% · ${dsz(s.done)} of ${dsz(s.total)}`
   +(s.rate>0?` · ${drate(s.rate)} · about ${ddur((s.total-s.done)/s.rate)} left`:'')+'</div>'}
 if(s.note)h+=`<div class="note${s.state==='failed'?' bad':''}">${esc(s.note)}</div>`;
 return h+'</div></div>'}
function dlModel(m,d){const j=m.install,busy=d.active===m.id,need=Math.max(0,m.downloadBytes-m.hereBytes);
 const state=m.loaded?'loaded':m.installed?'installed':busy?'installing':'';
 const act=busy?`<button data-dlstop="${esc(m.id)}"${DL.busy?' disabled':''}>Stop</button>`
  :m.installed?'':`<button class="go" data-dl="${esc(m.id)}"${d.active||DL.busy?' disabled':''}>${need===0?'Convert':(m.hereBytes>0?'Carry on · ':'Download · ')+dsz(need)}</button>`;
 let h=`<div class="dlm"><div class="dlh"><div><span class="name">${esc(m.id)}</span> · ${esc(m.title)}`
  +(m.id===d.default&&!m.installed&&!busy?' <span class="pill first">the one to start with</span>':'')+(state?` <span class="pill ${state}">${state}</span>`:'')+`</div>${act}</div>`
  +`<div class="help">${esc(m.summary)}</div><div class="help">${dsz(m.downloadBytes)} to download · ${m.memoryGiB.toFixed(1)} GiB in memory · ${m.bitsPerWeight} bits a weight · from <code>${esc(m.repo)}</code></div>`
  +`<div class="help">${m.installed?'Installed at':'Converted to'} <code>${esc(m.artifactPath)}</code>${m.installed&&m.artifactBytes?' · '+dsz(m.artifactBytes):''}</div>`;
 if(busy&&j&&j.steps)h+=`<div class="dlsteps">${j.steps.map(dlStep).join('')}</div>`;
 else if(busy)h+='<div class="note wait">starting…</div>';
 else if(j&&j.message&&j.state!=='done'&&!m.installed)h+=`<div class="note ${j.state==='failed'?'bad':'wait'}">${esc(j.message)}</div>`;
 else if(m.needsRestart)h+='<div class="note wait">Installed, and registered in splosh.toml. The engine reads the registry when it starts: restart it and the model can be loaded.</div>';
 return h+'</div>'}
function dlDraw(){const el=$('dl');if(!el)return;const d=DL.d;
 if(!d){el.innerHTML=`<div class="help">${esc(DL.msg||'loading…')}</div>`;return}
 const have=d.models.find(m=>m.installed);
 let h=d.models.map(m=>dlModel(m,d)).join('');
 if(have&&!d.active&&d.draft.wanted&&!d.draft.present)h+=`<div class="dlm"><div class="dlh"><div><span class="name">Draft model</span> <span class="pill">missing</span></div><button data-dl="${esc(have.id)}"${DL.busy?' disabled':''}>Download · ${dsz(d.draft.bytes)}</button></div>`
  +`<div class="help">DFlash 2 guesses several tokens ahead, so a step can write more than one. Without it the server writes one token a step. The engine picks it up when it next starts.</div></div>`;
 if(DL.msg)h+=`<div class="note bad" style="padding-top:10px">${esc(DL.msg)}</div>`;
 h+=`<div class="own"><div class="help"><b>Where they are kept.</b> A converted model is the one file the server reads, at the place shown with each model; what it was converted from, under <code>inputs/</code>, can be deleted once it is installed.`
  +(d.models.some(m=>m.artifact.startsWith('.build/'))?' Models in <code>.build</code> are kept with the build’s own files: <code>swift package clean</code>, <code>swift package reset</code> and <code>rm -rf .build</code> delete them too; <code>swift build</code> does not.':'')+'</div></div>';
 h+=`<div class="own"><div class="help"><b>Downloaded the files yourself?</b> Put them where Splosh looks, under the names they have on Hugging Face, and press the model’s button: what is already there is checked and not fetched again. Files in the Hugging Face cache (<code>${esc(d.cache)}</code>) are found too.</div><table>`
  +d.models.map(m=>`<tr><td>${esc(m.id)}</td><td><code>${esc(m.place)}</code>${m.files.length>1?' · '+m.files.map(esc).join(', '):''}</td></tr>`).join('')+'</table></div>';
 el.innerHTML=h}
// One chain of polls: a poll asked for while another waits for its answer takes its place.
async function dlPoll(){clearTimeout(DL.timer);const turn=++DL.turn;
 try{const r=await fetch('/v1/downloads',{cache:'no-store'});if(!r.ok)throw 0;const d=await r.json();if(turn!==DL.turn)return;DL.d=d;dlDraw();
  const key=d.setup+'|'+d.models.map(m=>m.id+(m.installed?1:0)).join();if(DL.key!==null&&key!==DL.key&&DL.onChange)DL.onChange(d);DL.key=key}catch(e){}
 if(turn===DL.turn)DL.timer=setTimeout(dlPoll,DL.d&&(DL.d.active||DL.d.setup)?1000:document.hidden?20000:4000)}
async function dlSend(path,id){DL.busy=true;DL.msg=null;dlDraw();
 try{DL.d=await send(path,{model:id})}catch(e){DL.msg=e.message}
 DL.busy=false;dlDraw();dlPoll()}
document.addEventListener('click',e=>{const t=e.target.dataset||{};if(t.dl)dlSend('/v1/downloads',t.dl);else if(t.dlstop)dlSend('/v1/downloads/cancel',t.dlstop)});
"""#
}

/// The page a server with no model serves at `/`: what there is to download, and the install
/// going on. When a model is installed the engine is started on it, and this page gives way to
/// the dashboard.
enum SetupPage {
    static let html = #"""
<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Splosh: choose a model</title>
<style>
:root{color-scheme:light dark;--bg:#f6f6f4;--card:#fff;--ink:#1b1b1a;--mute:#6d6d68;--line:#e2e2dd;--a:#2f6fde;--b:#1f9d6b;--c:#c9861a;--e:#c0392b}
@media(prefers-color-scheme:dark){:root{--bg:#141413;--card:#1e1e1c;--ink:#ecece8;--mute:#96968f;--line:#30302d}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.45 -apple-system,system-ui,sans-serif;padding:20px}
main{max-width:860px;margin:0 auto}h1{font-size:18px;margin:0 0 2px}h2{font-size:12px;text-transform:uppercase;letter-spacing:.06em;color:var(--mute);margin:0 0 6px}
a{color:var(--a);text-decoration:none}.sub{color:var(--mute);margin-bottom:18px}p{margin:0 0 10px}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px;margin-bottom:14px}
.name{font-weight:550}.help,.note{color:var(--mute);font-size:12px;overflow-wrap:anywhere}.note.wait{color:var(--c)}.note.bad{color:var(--e)}
.pill{font-size:11px;padding:1px 7px;border-radius:9px;border:1px solid var(--line);color:var(--mute);white-space:nowrap}
button{font:inherit;border:1px solid var(--line);background:var(--card);color:var(--ink);border-radius:6px;padding:6px 14px;cursor:pointer}
button.go{background:var(--a);border-color:var(--a);color:#fff}button:disabled{opacity:.5;cursor:default}
ol{margin:0;padding-left:20px}li{margin:2px 0}
\#(DownloadsPanel.style)
</style></head><body><main>
<h1>Splosh</h1><div class="sub">No model is installed yet · <a href="/models">Models</a> · <a href="/settings">Settings</a></div>
<div class="card"><h2>Choose a model</h2>
<p>The server is up, without a model. Pick one below and Splosh does the rest; this page shows each step as it happens, and becomes the dashboard when the model is loaded.</p>
<ol class="help"><li><b>Download</b> from Hugging Face. A download that is stopped, or loses its connection, carries on from where it got to.</li>
<li><b>Check</b> every file against the SHA-256 Splosh was tested with.</li>
<li><b>Convert</b> the weights into the form the engine maps straight into memory.</li>
<li><b>Fetch the draft model</b> that lets a step write several tokens, and <b>start serving</b>.</li></ol>
<p class="help" style="margin-top:10px">They are all the same model, Qwen3.8-27B, at different precisions. A smaller file decodes faster and is further from the full model; prompts are read at much the same rate by all. Others can be added later from the Models page, and the server changes between them on request.</p></div>
<div class="card"><h2>Models</h2><div id="ready" class="note wait" hidden></div><div id="dl"></div></div>
<div class="card"><h2>From a terminal</h2><div class="help">The same, in the directory the server was started in: <code>splosh download</code> fetches the default, <code>splosh download uq5</code> another, and <code>splosh download --list</code> says what there is. In the terminal the server is running in, pressing Enter starts the default.</div></div>
</main><script>
const $=id=>document.getElementById(id);
const esc=s=>String(s).replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
async function send(path,body){const r=await fetch(path,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
 const j=await r.json().catch(()=>({error:`the server answered ${r.status}`}));if(!r.ok)throw new Error((j.error&&j.error.message)||j.error||`the server answered ${r.status}`);return j}
\#(DownloadsPanel.script)
// A model is installed: the server starts its engine on it, and then answers as one that has a model.
DL.onChange=d=>{if(!d.setup){location.replace('/');return}
 const m=d.models.find(m=>m.installed);if(m&&!d.active){$('ready').hidden=false;$('ready').textContent=`${m.id} is installed. The server is loading it; the dashboard opens when it is ready.`}};
dlPoll();
</script></body></html>
"""#
}
