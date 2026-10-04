import Foundation
import Hummingbird
import HTTPTypes
import NIOCore

/// What the settings page reads and changes. The command that runs the server supplies it: the
/// configuration file, and what can change without a restart, are its business.
public struct SettingsService: Sendable {
    /// The settings as they stand, in the shape `SettingsPage` renders.
    public var read: @Sendable () -> JSONValue
    /// Save new values; a nil value puts a setting back to its default. Returns the settings
    /// as they then stand. Throws `SettingsError` when a value is refused, and saves nothing.
    public var write: @Sendable ([(key: String, value: String?)]) throws -> JSONValue
    /// Replace the engine with one started on the saved settings; nil when no process holds
    /// the port to do it.
    public var restart: (@Sendable () -> Void)?

    public init(read: @escaping @Sendable () -> JSONValue, write: @escaping @Sendable ([(key: String, value: String?)]) throws -> JSONValue,
                restart: (@Sendable () -> Void)?) {
        self.read = read; self.write = write; self.restart = restart
    }
}

/// A change to the settings that was not made, and why, for the page to show.
public struct SettingsError: Error, Equatable, Sendable {
    public let key: String?
    public let message: String
    public init(key: String?, message: String) { self.key = key; self.message = message }
}

extension Routes {
    static func addSettings(_ settings: SettingsService, to router: Router<BasicRequestContext>) {
        @Sendable func refused(_ status: HTTPResponse.Status, _ message: String, key: String? = nil) -> Response {
            ChatEndpoint.json(.object([("error", .string(message)), ("key", key.map(JSONValue.string) ?? .null)]), status: status)
        }
        router.get("settings") { _, _ -> Response in
            Response(status: .ok, headers: [.contentType: "text/html; charset=utf-8"],
                     body: ResponseBody(byteBuffer: ByteBuffer(string: SettingsPage.html)))
        }
        // The models have a page of their own: the same settings file, the keys that are theirs.
        router.get("models") { _, _ -> Response in
            Response(status: .ok, headers: [.contentType: "text/html; charset=utf-8"],
                     body: ResponseBody(byteBuffer: ByteBuffer(string: SettingsPage.models)))
        }
        router.get("v1/settings") { _, _ -> Response in
            var response = ChatEndpoint.json(settings.read())
            response.headers[.cacheControl] = "no-store"
            return response
        }
        router.post("v1/settings") { request, _ -> Response in
            if let reason = changeRefusal(request) { return refused(.forbidden, reason) }
            let buffer = try await request.body.collect(upTo: 1 << 20)
            guard let members = (try? JSONValue.parse(Array(buffer.readableBytesView)))?.objectValue else {
                return refused(.badRequest, "expected a JSON object of settings")
            }
            var changes: [(key: String, value: String?)] = []
            for (key, value) in members {
                switch value {
                case .null: changes.append((key, nil))
                case .string(let text), .number(let text): changes.append((key, text))
                case .bool(let on): changes.append((key, on ? "true" : "false"))
                default: return refused(.badRequest, "\(key) must be a string, a number, true, false or null", key: key)
                }
            }
            do {
                return ChatEndpoint.json(try settings.write(changes))
            } catch let error as SettingsError {
                return refused(.badRequest, error.message, key: error.key)
            }
        }
        router.post("v1/settings/restart") { request, _ -> Response in
            if let reason = changeRefusal(request) { return refused(.forbidden, reason) }
            guard let restart = settings.restart else {
                return refused(.conflict, "this server has no process holding its port to restart the engine; stop it and start it again")
            }
            restart()
            return ChatEndpoint.json(.object([("restarting", .bool(true))]), status: .accepted)
        }
    }

    /// Why a request to change the settings is not taken; nil if it is. Changes come only from
    /// a page this server served to this machine: the request names a loopback host, comes
    /// from no other origin, and carries JSON, which a page elsewhere cannot send unasked.
    static func changeRefusal(_ request: Request) -> String? {
        func host(_ authority: String) -> String {
            if authority.hasPrefix("["), let end = authority.firstIndex(of: "]") { return String(authority[...end]) }
            return String(authority.split(separator: ":", maxSplits: 1, omittingEmptySubsequences: false).first ?? "")
        }
        let loopback: Set<String> = ["localhost", "127.0.0.1", "[::1]"]
        guard let authority = request.head.authority, loopback.contains(host(authority).lowercased()) else {
            return "settings can be changed only from the machine the server runs on, at a localhost address"
        }
        if let origin = request.headers[.origin] {
            guard let separator = origin.range(of: "://"), loopback.contains(host(String(origin[separator.upperBound...])).lowercased()) else {
                return "settings can be changed only from the server's own settings page"
            }
        }
        guard request.headers[.contentType]?.lowercased().hasPrefix("application/json") == true else {
            return "a change to the settings must be sent as application/json"
        }
        return nil
    }
}

/// The settings page served at `/settings`, and the models page at `/models`. Each reads
/// `/v1/settings`, posts changes back, and can ask for the engine to be restarted. They are one
/// page that shows one part of the settings or the other: the models page has the groups
/// "Models" and "Model", the model in memory and the ones to download; the settings page has
/// the rest.
enum SettingsPage {
    static let html = page(models: false)
    static let models = page(models: true)

    private static func page(models: Bool) -> String {
        #"""
<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Splosh \#(models ? "models" : "settings")</title>
<style>
:root{color-scheme:light dark;--bg:#f6f6f4;--card:#fff;--ink:#1b1b1a;--mute:#6d6d68;--line:#e2e2dd;--a:#2f6fde;--b:#1f9d6b;--c:#c9861a;--e:#c0392b}
@media(prefers-color-scheme:dark){:root{--bg:#141413;--card:#1e1e1c;--ink:#ecece8;--mute:#96968f;--line:#30302d}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.45 -apple-system,system-ui,sans-serif;padding:20px 20px 90px}
h1{font-size:18px;margin:0 0 2px}h2{font-size:12px;text-transform:uppercase;letter-spacing:.06em;color:var(--mute);margin:0 0 6px}
a{color:var(--a);text-decoration:none}.sub{color:var(--mute);margin-bottom:18px;overflow-wrap:anywhere}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px;margin-bottom:14px}
.row{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,300px);gap:6px 20px;padding:10px 0;border-bottom:1px solid var(--line);align-items:start}
.row:last-child{border-bottom:0;padding-bottom:0}.name{font-weight:550}.help,.note{color:var(--mute);font-size:12px;overflow-wrap:anywhere}
.note.wait{color:var(--c)}.note.bad{color:var(--e)}
input[type=text],input[type=number],select{width:100%;font:inherit;color:inherit;background:var(--bg);border:1px solid var(--line);border-radius:6px;padding:6px 8px}
input:focus,select:focus{outline:2px solid var(--a);outline-offset:-1px}label.sw{display:flex;align-items:center;gap:8px;padding:5px 0}
.pill{font-size:11px;padding:1px 7px;border-radius:9px;border:1px solid var(--line);color:var(--mute);white-space:nowrap}
button{font:inherit;border:1px solid var(--line);background:var(--card);color:var(--ink);border-radius:6px;padding:6px 14px;cursor:pointer}
button.go{background:var(--a);border-color:var(--a);color:#fff}button:disabled{opacity:.5;cursor:default}
button.link{border:0;background:none;color:var(--a);padding:0;font-size:12px}
.foot{position:fixed;left:0;right:0;bottom:0;background:var(--card);border-top:1px solid var(--line);padding:12px 20px;display:flex;flex-wrap:wrap;gap:10px 14px;align-items:center}
.foot .msg{flex:1 1 240px;color:var(--mute)}.foot .msg.bad{color:var(--e)}.foot .msg.wait{color:var(--c)}
.pick{display:flex;gap:8px}.pick select{min-width:0}.pick button{flex:none}
@media(max-width:640px){.row{grid-template-columns:minmax(0,1fr)}}
\#(DownloadsPanel.style)
</style></head><body>
<h1>\#(models ? "Models" : "Settings")</h1><div class="sub"><a href="/">← Dashboard</a> · <a href="\#(models ? "/settings" : "/models")">\#(models ? "Settings" : "Models")</a> · <span id="file">loading…</span></div>
<div id="groups"></div>
<div class="foot"><div class="msg" id="msg"></div><button id="restart" hidden>Restart engine</button><button id="discard" disabled>Discard</button><button class="go" id="save" disabled>Save</button></div>
<script>
// Which of the two pages this is: the models', or the rest of the settings.
const MODELS=\#(models ? "true" : "false"),ofPage=x=>(x.group==='Models'||x.group==='Model')===MODELS;
const $=id=>document.getElementById(id);
const esc=s=>String(s).replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const when={now:'applies at once',engine:'needs an engine restart',server:'needs the server stopped and started'};
let S=null,dirty={},busy=false;
const shown=x=>x.key in dirty?dirty[x.key]:x.value;
function control(x){const v=shown(x),id=`f-${x.key}`;
 if(x.kind==='boolean'){const on=(v??x.default)==='true';return `<label class="sw"><input type="checkbox" id="${id}" data-k="${x.key}"${on?' checked':''}> <span>${on?'On':'Off'}</span></label>`}
 if(x.kind==='choice')return `<select id="${id}" data-k="${x.key}">`+(x.default===null?`<option value=""${v===null?' selected':''}>${esc(x.unset)}</option>`:'')+x.choices.map(c=>`<option${c===(v??x.default)?' selected':''}>${esc(c)}</option>`).join('')+'</select>';
 return `<input id="${id}" data-k="${x.key}" type="${x.kind==='text'?'text':'number'}"${x.kind==='number'?' step="any"':''}${x.kind==='text'?' spellcheck="false"':''} value="${esc(v??'')}" placeholder="${esc(x.unset)}">`}
function note(x){const v=shown(x),out=[];
 if(x.key in dirty)out.push('<span class="note wait">not saved</span>');
 else if(x.pending)out.push(`<span class="note wait">saved; the running ${x.applies==='server'?'server':'engine'} has ${esc(x.running??x.unset)}</span>`);
 else if(v===null&&x.running!==null&&x.running!==x.default)out.push(`<span class="note">now ${esc(x.running)}</span>`);
 if(v!==null)out.push(`<button class="link" data-reset="${x.key}">use the default${(x.kind==='boolean'||x.kind==='choice')&&x.default!==null?` (${esc(x.default==='true'?'on':x.default==='false'?'off':x.default)})`:''}</button>`);
 return out.join(' · ')}
function render(){if(!S)return;const groups=[];
 for(const x of S.settings.filter(ofPage)){let g=groups.find(g=>g.name===x.group);if(!g)groups.push(g={name:x.group,rows:[]});
  g.rows.push(`<div class="row"><div><label class="name" for="f-${x.key}">${esc(x.title)}</label> <span class="pill">${when[x.applies]}</span><div class="help">${esc(x.help)} <code>${x.key}</code></div></div><div>${control(x)}<div class="note" id="n-${x.key}">${note(x)}</div></div></div>`)}
 const mg=groups.find(g=>g.name==='Models');
 if(mg&&ofModels())mg.rows.unshift(`<div class="row"><div><span class="name">Loaded model</span> <span class="pill">loads at once</span><div class="help">The model in memory. Pick another and press Load to have it loaded in its place: requests in flight finish first, requests that arrive meanwhile wait, and the load takes a while. This is not saved to splosh.toml: after the server is stopped and started it loads the starting model below.</div></div><div id="lm"></div></div>`);
 // On the models page: the registered models first, then the ones that can be downloaded (see
 // DownloadsPanel), then where the tokenizer and the draft model are.
 groups.sort((a,b)=>(b.name==='Models')-(a.name==='Models'));
 $('groups').innerHTML=groups.map(g=>`<div class="card"><h2>${esc(g.name==='Model'?'Files':g.name)}</h2>${g.rows.join('')}</div>`+(g.name==='Models'?'<div class="card" id="downloads"><h2>Download models</h2><div id="dl"></div></div>':'')).join('');
 if($('lm'))$('lm').innerHTML=pickbox();
 dlDraw();
 $('file').textContent=S.file;foot()}
function foot(message,kind){const changed=Object.keys(dirty).length,waiting=S?S.settings.filter(x=>x.pending):[];
 $('save').disabled=$('discard').disabled=busy||!changed;
 const engine=waiting.filter(x=>x.applies==='engine').length,server=waiting.filter(x=>x.applies==='server');
 $('restart').hidden=!(S&&S.canRestart&&engine);$('restart').disabled=busy;
 const m=$('msg');m.className='msg'+(kind?' '+kind:waiting.length&&!message?' wait':'');
 m.textContent=message||(changed?`${changed} change${changed===1?'':'s'} not saved`:waiting.length?
  (engine?`${engine} saved setting${engine===1?' waits':'s wait'} for the engine to restart${S.canRestart?'':' (stop the server and start it again)'}. `:'')+
  (server.length?`${server.map(x=>x.title).join(' and ')} wait${server.length===1?'s':''} for the server to be stopped and started.`:''):'')}
function edited(el){const x=S.settings.find(x=>x.key===el.dataset.k);if(!x)return;
 let v=x.kind==='boolean'?String(el.checked):el.value.trim();if(v==='')v=null;
 if(x.kind==='boolean')el.nextElementSibling.textContent=el.checked?'On':'Off';
 if(v===x.value||(x.value===null&&x.kind!=='text'&&x.kind!=='integer'&&x.kind!=='number'&&v===x.default))delete dirty[x.key];else dirty[x.key]=v;
 $('n-'+x.key).innerHTML=note(x);foot()}
async function send(path,body){const r=await fetch(path,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
 const j=await r.json().catch(()=>({error:`the server answered ${r.status}`}));if(!r.ok)throw Object.assign(new Error((j.error&&j.error.message)||j.error||`the server answered ${r.status}`),{key:j.key});return j}
// The loaded model, picked from what the server lists (/v1/models: the loaded one first; shown here in the order the file registers them).
// The button's answer comes when the model is in memory, or has failed to be; meanwhile the control says so. A model with no artifact cannot be picked.
let M=null,pick=null,loading=null,told=null,sig='',seq=0,seen=0,asking=false;
const stage={dwell:'the loaded model has its turn first',waiting:'new requests wait while the loaded model finishes the ones it has',stopping:'the loaded model is being stopped',loading:'it is being loaded'};
const ofModels=()=>MODELS&&!!(M&&S&&Array.isArray(M.data)&&M.switch&&M.loaded&&S.settings.some(x=>x.key.startsWith('model.')));
const gib=b=>(b/1073741824).toFixed(2)+' GiB',at=id=>S.settings.findIndex(x=>x.key==='model.'+id),ok=id=>M.data.some(m=>m.id===id&&m.state!=='missing');
function pickstate(){const sw=M.switch,cur=ok(pick)?pick:M.loaded,m=M.data.find(m=>m.id===cur),off=!!(loading||sw.target||sw.mode==='none');
 const [cls,text]=loading?['wait',`loading ${loading}…`]
  :sw.target?['wait',`switching to ${sw.target}: ${stage[sw.phase]||'in hand'}`+(sw.parked?`; ${sw.parked} request${sw.parked===1?'':'s'} kept until then`:'')]
  :sw.mode==='none'?['','This server runs without a process holding its port, so it keeps the model it started on.']
  :told?[told.bad?'bad':'',told.text]:['',m?(m.size_bytes==null?'':gib(m.size_bytes)+' · ')+m.path:''];
 return{cur,off,cls,text}}
function pickbox(){const p=pickstate();
 return `<div class="pick"><select id="lm-sel"${p.off?' disabled':''}>`+[...M.data].sort((a,b)=>at(a.id)-at(b.id)).map(m=>`<option value="${esc(m.id)}"${m.id===p.cur?' selected':''}${m.state==='missing'?' disabled':''}>${esc(m.id)} · ${m.size_bytes==null?'':gib(m.size_bytes)+' · '}${esc(m.path.split('/').pop())}${m.state==='missing'?' · missing':m.loaded?' · loaded':''}</option>`).join('')
  +`</select><button class="go" id="lm-go"${p.off||p.cur===M.loaded?' disabled':''}>Load</button></div><div class="note ${p.cls}" id="lm-note">${esc(p.text)}</div>`}
function paintLoad(){if(ofModels()!==!!$('lm'))render();else if($('lm'))$('lm').innerHTML=pickbox()}
async function models(){const k=++seq;try{const j=await (await fetch('/v1/models',{cache:'no-store'})).json();if(k>seen){seen=k;const t=JSON.stringify(j);if(t!==sig){sig=t;M=j;told=null;paintLoad()}}}catch(e){}}
async function loadModel(){const id=pick;if(!id||loading)return;let said;loading=id;told=null;paintLoad();
 try{await send('/v1/models/load',{model:id});said={text:`${id} is loaded.`}}catch(e){said={text:e.message,bad:true}}
 loading=null;pick=null;await models();told=said;paintLoad()}
async function load(){const got=await Promise.all([fetch('/v1/settings',{cache:'no-store'}).then(r=>r.json()),fetch('/v1/models',{cache:'no-store'}).then(r=>r.json()).catch(()=>null)]);
 S=got[0];if(got[1]){M=got[1];sig=JSON.stringify(M)}render()}
$('groups').addEventListener('change',e=>{if(e.target.id!=='lm-sel')return;pick=e.target.value;told=null;const p=pickstate();
 $('lm-go').disabled=p.off||p.cur===M.loaded;$('lm-note').className='note '+p.cls;$('lm-note').textContent=p.text});
$('groups').addEventListener('click',e=>{if(e.target.id==='lm-go')loadModel()});
// What the server lists changes when another client has a model loaded, so it is asked for again.
if(MODELS){setInterval(async()=>{if(asking||document.hidden)return;asking=true;await models();asking=false},3000);
 document.addEventListener('visibilitychange',()=>{if(!document.hidden)models()})}
document.addEventListener('input',e=>{if(e.target.dataset.k)edited(e.target)});
document.addEventListener('click',e=>{const k=e.target.dataset.reset;if(!k)return;const x=S.settings.find(x=>x.key===k);
 if(x.value===null)delete dirty[k];else dirty[k]=null;render()});
$('discard').onclick=()=>{dirty={};render()};
$('save').onclick=async()=>{busy=true;foot('Saving…');
 try{S=await send('/v1/settings',dirty);dirty={};busy=false;render();if(!S.settings.some(x=>x.pending))foot('Saved. In effect now.')}
 catch(e){busy=false;foot(e.message,'bad');if(e.key&&$('f-'+e.key))$('f-'+e.key).focus()}};
$('restart').onclick=async()=>{
 if(!confirm(`Restart the engine?\n\nThe port stays open. Requests in flight get up to ${S.restartDrainSeconds} s to finish, then are cut and their contexts saved; requests that arrive meanwhile wait for the new engine.`))return;
 busy=true;foot('Asking the engine to restart…','wait');
 try{await send('/v1/settings/restart',{})}catch(e){busy=false;foot(e.message,'bad');return}
 foot('The engine is restarting: saving conversations, then loading the model again. This page carries on when it is back.','wait');
 const again=async()=>{try{const s=await (await fetch('/v1/settings',{cache:'no-store'})).json();
   if(s.settings.some(x=>x.pending&&x.applies==='engine'))throw 0;S=s;busy=false;render();foot('The engine is back, on the saved settings.')}catch(e){setTimeout(again,1500)}};
 setTimeout(again,2500)};
\#(DownloadsPanel.script)
// A model installed is one more to load, and may be a line more in splosh.toml: both are read again.
DL.onChange=()=>{if(!Object.keys(dirty).length)load().catch(()=>{});else models()};
load().then(()=>{if(MODELS)dlPoll()}).catch(()=>{$('file').textContent='could not read the settings'});
</script></body></html>
"""#
    }
}
