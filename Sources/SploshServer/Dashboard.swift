import Foundation

/// The live stats page served at `/`. It polls `/v1/stats` and renders in place.
enum Dashboard {
    static let html = #"""
<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Splosh</title>
<style>
:root{color-scheme:light dark;--bg:#f6f6f4;--card:#fff;--ink:#1b1b1a;--mute:#6d6d68;--line:#e2e2dd;--a:#2f6fde;--b:#1f9d6b;--c:#c9861a;--d:#8a5cd6;--e:#c0392b}
@media(prefers-color-scheme:dark){:root{--bg:#141413;--card:#1e1e1c;--ink:#ecece8;--mute:#96968f;--line:#30302d}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.45 -apple-system,system-ui,sans-serif;padding:20px}
h1{font-size:18px;margin:0 0 2px}h2{font-size:12px;text-transform:uppercase;letter-spacing:.06em;color:var(--mute);margin:0 0 10px}
.sub{color:var(--mute);margin-bottom:18px}.grid{display:grid;gap:14px;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));margin-bottom:14px}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px}
.big{font-size:26px;font-weight:650;font-variant-numeric:tabular-nums}.unit{font-size:12px;color:var(--mute);font-weight:400}
table{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums}th{text-align:left;font-weight:500;color:var(--mute);font-size:12px}
th,td{padding:6px 10px 6px 0;border-bottom:1px solid var(--line);white-space:nowrap}tr:last-child td{border-bottom:0}
.scroll{overflow-x:auto}.bar{display:flex;height:16px;border-radius:5px;overflow:hidden;background:var(--line);margin:8px 0}
.bar i{display:block;height:100%}.key{display:flex;flex-wrap:wrap;gap:4px 16px;color:var(--mute);font-size:12px}
.key b{display:inline-block;width:9px;height:9px;border-radius:2px;margin-right:5px}
.pill{font-size:11px;padding:1px 7px;border-radius:9px;border:1px solid var(--line)}
#mp select,#mp button{font:inherit;font-size:12px;padding:1px 8px;border-radius:7px;border:1px solid var(--line);background:var(--bg);color:var(--ink)}
#mp button{cursor:pointer}#mp select:disabled,#mp button:disabled{opacity:.5;cursor:default}#mpn.wait{color:var(--c)}#mpn.bad{color:var(--e)}
.prefill{color:var(--c)}.decode{color:var(--b)}.queued{color:var(--mute)}.empty{color:var(--mute);padding:6px 0}
.mini{height:6px;border-radius:3px;background:var(--line);min-width:80px;overflow:hidden}.mini i{display:block;height:100%;background:var(--a)}
#sessions tr[data-s]{cursor:pointer}#sessions tr[data-s]:hover td{background:rgba(127,127,127,.1)}
#shade{position:fixed;inset:0;z-index:19;background:rgba(0,0,0,.4);display:none}#shade.open{display:block}
#win{position:fixed;z-index:20;display:none;flex-direction:column;width:min(440px,calc(100vw - 16px));height:250px;background:var(--card);border:1px solid var(--line);border-radius:10px;box-shadow:0 10px 34px rgba(0,0,0,.28);overflow:hidden}
#win.open{display:flex}#win.large{left:50%;top:50%;transform:translate(-50%,-50%);width:min(980px,calc(100vw - 32px));height:calc(100vh - 64px)}
.whead{display:flex;align-items:center;gap:8px;padding:7px 10px;border-bottom:1px solid var(--line)}
.whead .t{flex:1;min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;color:var(--mute);font-size:12px}.whead .t b{color:var(--ink)}
.whead button{font:inherit;font-size:12px;padding:2px 10px;border-radius:7px;border:1px solid var(--line);background:var(--bg);color:var(--ink);cursor:pointer;white-space:nowrap}
#win.large #wbig,#win:not(.large) #wclose{display:none}
#wtext{flex:1;overflow-y:auto;overscroll-behavior:contain;padding:10px 12px;font-size:12px;line-height:1.5}#wtext:empty::before{content:'Nothing written yet';color:var(--mute)}
#win.large #wtext{font-size:14px;padding:14px 18px}
.part{white-space:pre-wrap;overflow-wrap:anywhere;margin-bottom:10px}.part.thinking{color:var(--mute)}
.part.tool{font-family:ui-monospace,Menlo,monospace;font-size:.92em;border:1px solid var(--line);border-radius:7px;padding:6px 9px;background:var(--bg)}
.part b{display:block;font:500 11px/1.6 -apple-system,system-ui,sans-serif;text-transform:uppercase;letter-spacing:.06em;color:var(--mute)}
</style></head><body>
<h1>Splosh</h1><div class="sub"><span id="mp"></span><span id="sub">connecting…</span> · <a href="/models" style="color:var(--a);text-decoration:none">Models</a> · <a href="/settings" style="color:var(--a);text-decoration:none">Settings</a></div>
<div class="grid">
<div class="card"><h2>Decode</h2><div class="big"><span id="dec">–</span> <span class="unit">tok/s</span></div></div>
<div class="card"><h2>Prefill</h2><div class="big"><span id="pre">–</span> <span class="unit">tok/s</span></div></div>
<div class="card"><h2>Sessions</h2><div class="big"><span id="act">–</span> <span class="unit" id="que"></span></div></div>
<div class="card"><h2>Step</h2><div class="big"><span id="step">–</span> <span class="unit" id="rows"></span></div></div>
</div>
<div class="card" style="margin-bottom:14px"><h2>Memory</h2>
<div><span class="big" id="memused">–</span> <span class="unit" id="memtotal"></span></div>
<div class="bar" id="membar"></div><div class="key" id="memkey"></div></div>
<div class="card" style="margin-bottom:14px"><h2>Sessions</h2><div class="scroll" id="sessions"></div><div class="key" style="margin-top:8px">Rest the pointer on a session to watch its reply being written; click it for a large window.</div></div>
<div class="card" style="margin-bottom:14px"><h2>Cached prefixes</h2><div class="scroll" id="cached"></div><div class="key" id="disk" style="margin-top:8px"></div></div>
<div class="card" style="margin-bottom:14px"><h2>Totals</h2><div class="scroll" id="totals"></div>
<div class="grid" style="margin:14px 0 0">
<div><h2>Combined, generating</h2><div class="big"><span id="cdec">–</span> <span class="unit">tok/s</span></div><div class="key" id="cdecnote"></div></div>
<div><h2>Combined, prompt</h2><div class="big"><span id="cpre">–</span> <span class="unit">tok/s</span></div><div class="key" id="cprenote"></div></div>
<div><h2>Average session, generating</h2><div class="big"><span id="adec">–</span> <span class="unit">tok/s</span></div><div class="key">finished requests, each on its own clock</div></div>
<div><h2>Average session, prompt</h2><div class="big"><span id="apre">–</span> <span class="unit">tok/s</span></div><div class="key">tokens it had to evaluate, after reuse</div></div>
<div><h2>Drafts accepted, last 5 s</h2><div class="big"><span id="acc">–</span></div><div class="key" id="accnote"></div></div>
</div><div class="key" id="blocks" style="margin-top:8px"></div></div>
<div class="card" style="margin-bottom:14px"><h2>Time outside the steps, average per finished request</h2><div class="scroll" id="overhead"></div><div class="key" style="margin-top:8px">Store copy is the disk store's copy of a conversation, on the scheduler thread; first step is a step, not outside one, and is to be compared with its tier below.</div></div>
<div class="card" style="margin-bottom:14px"><h2>Steps by tier and context</h2><div class="scroll" id="tiers"></div></div>
<div class="card"><h2>Conversations</h2><div class="key" style="margin-bottom:8px">Requests whose prompts each continue the one before are one conversation; its figures are summed over them.</div><div class="scroll" id="recent"></div></div>
<div id="shade"></div>
<div id="win" role="dialog" aria-label="Session reply"><div class="whead"><span class="t" id="wtitle"></span><button id="wbig">Make larger</button><button id="wclose">Close</button></div><div id="wtext"></div></div>
<script>
const $=id=>document.getElementById(id);
const gb=b=>(b/1073741824).toFixed(2)+' GiB',mb=b=>b>=1073741824?gb(b):(b/1048576).toFixed(0)+' MiB';
const n=x=>x.toLocaleString(),r=x=>x>=100?x.toFixed(0):x.toFixed(1);
const rate=(tokens,seconds)=>seconds>0.05&&tokens>0?r(tokens/seconds):'–';
const kinds=x=>[['thinking',x.thinkingTokens],['answer',x.answerTokens],['tool call',x.toolCallTokens]].filter(p=>p[1]>0);
const made=x=>kinds(x).map(p=>`${p[0]} ${n(p[1])}`).join(' · ')||'–';
// A session's state, said once: a generating session is named by what it is writing, in the one colour.
const state=x=>`<span class="pill ${x.state}">${esc(x.state==='decode'&&x.producing||x.state)}</span>`;
// What a generating session has written: the kinds are given only where they say more than the state does.
const written=x=>{const k=kinds(x);return `${n(x.generatedTokens)} generated`+(k.length>1||k.length&&k[0][0]!==x.producing?` <span class="queued">${made(x)}</span>`:'')};
const pct=(a,b)=>b>0?(a/b*100).toFixed(0)+'%':'–';
const left=x=>{const v=x.recentPrefillTokensPerSecond||x.prefillTokensPerSecond,l=x.promptTokens-x.evaluatedTokens;if(x.state!=='prefill'||!(v>0)||l<=0)return'';const t=l/v;return ` <span class="queued">· about ${t<90?t.toFixed(0)+' s':t<5400?(t/60).toFixed(1)+' min':(t/3600).toFixed(1)+' h'} left</span>`};
const ago=x=>x<90?x.toFixed(0)+'s ago':x<5400?(x/60).toFixed(0)+'m ago':(x/3600).toFixed(1)+'h ago';
const esc=s=>String(s).replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
function table(head,rows,empty,attr){if(!rows.length)return `<div class="empty">${empty}</div>`;
 return '<table><tr>'+head.map(h=>`<th>${h}</th>`).join('')+'</tr>'+rows.map((c,i)=>`<tr${attr?attr(i):''}>`+c.map(v=>`<td>${v}</td>`).join('')+'</tr>').join('')+'</table>'}
// The reply window: a session's reply as it is written. Resting on a session's row opens it
// small, beside the pointer; its button, or a click on the row, makes it large. It stays with
// the conversation: when the request ends and the conversation's next arrives, it goes on with that.
const win=$('win'),wtext=$('wtext'),shade=$('shade'),sess=$('sessions');
let stats=null,view=null,busy=false,opening=0,pending=0,closing=0,down=null;
const keep=()=>{clearTimeout(closing);closing=0},drop=()=>{clearTimeout(opening);pending=0};
const mine=(id,conv)=>view&&(id===view.id||conv>0&&conv===view.conv);
function aim(id,conv){if(!mine(id,conv)){view={id,conv,now:0,shown:0,turn:0,after:0,large:false};wtext.textContent=''}}
function shut(){keep();view=null;win.className='';shade.className=''}
function leave(){if(view&&!view.large&&!closing)closing=setTimeout(shut,300)}
function peek(id,conv,x,over,under){if(view&&view.large)return;aim(id,conv);keep();
 // Under the row, or over it where there is no room under: the row itself stays in sight.
 win.style.left=Math.max(8,Math.min(x+14,innerWidth-448))+'px';win.style.top=Math.max(8,under+252>innerHeight?over-248:under-2)+'px';
 win.className='open';refresh()}
function enlarge(){if(!view)return;keep();drop();view.large=true;win.style.left=win.style.top='';win.className='open large';shade.className='open';wtext.scrollTop=wtext.scrollHeight;refresh()}
function headline(v,x){const who=`<b>#${v.id}</b>`+(v.conv>0&&v.large?` · conversation ${v.conv}`:'');
 if(!x)return `${who} · ended`+(v.conv>0?', waiting for the conversation’s next request':'');
 const now=x.state==='decode'?` ${n(x.generatedTokens)} tokens · ${r(x.recentDecodeTokensPerSecond||0)} tok/s`
  :x.state==='prefill'?` ${n(x.evaluatedTokens)} of ${n(x.promptTokens)} tokens`:'';
 return `${who}${x.turn&&v.large?` turn ${x.turn}`:''} · ${state(x)}${now}`+(v.shown&&v.shown!==x.id?` · showing turn ${v.turn}`:'')}
// Text is added to what is there and never replaced, so a selection and the scroll position
// hold. A tool call's arguments are JSON text: shown with its escapes undone, whole escapes only.
const plain={n:'\n',t:'\t',r:'',b:'',f:''};
function paint(v,parts){const end=wtext.scrollHeight-wtext.scrollTop-wtext.clientHeight<30;
 if(v.shown!==v.id){wtext.textContent='';v.shown=v.id;v.turn=v.now}
 parts.forEach((p,i)=>{const tool=p.kind==='tool call';let el=wtext.children[i];
  if(!el){el=document.createElement('div');el.className='part '+(tool?'tool':p.kind);el.len=0;
   const b=document.createElement('b');b.textContent=tool?`tool call · ${p.name||''}`:p.kind;el.append(b);wtext.append(el)}
  let to=p.text.length;if(tool){let k=to;while(k>0&&p.text[k-1]==='\\')k--;if((to-k)%2)to--}
  if(to>el.len){const s=p.text.slice(el.len,to);el.append(tool?s.replace(/\\(.)/g,(m,c)=>c==='u'?m:c in plain?plain[c]:c):s);el.len=to}});
 if(end)wtext.scrollTop=wtext.scrollHeight}
async function refresh(){if(!view||busy||!stats)return;busy=true;const v=view;
 try{const x=(v.conv>0&&stats.sessions.filter(y=>y.conversation===v.conv).pop())||stats.sessions.find(y=>y.id===v.id);
  if(x){v.id=x.id;v.conv=x.conversation||v.conv;v.now=x.turn;v.after=0}
  // A reply's end is written as its session leaves the stats, so it is asked for twice more then.
  if(x||v.after++<2){const got=await fetch(`/v1/sessions/${v.id}/reply`,{cache:'no-store'});
   // A request that has written nothing yet leaves the reply before it in view.
   if(got.ok){const parts=(await got.json()).parts;if(view===v&&(parts.length||!v.shown||v.shown===v.id))paint(v,parts)}}
  if(view===v)$('wtitle').innerHTML=headline(v,x);
 }catch(e){}finally{busy=false}}
// The table is rebuilt twice a second, so its rows are found from the events, not listened on:
// a pointer that rests on a row for a moment opens the window, and a press and release on one
// (which may be two different elements by then) is a click.
const rowOf=e=>e.target.closest?e.target.closest('tr[data-s]'):null;
sess.addEventListener('mousemove',e=>{if(view&&view.large)return;
 const row=rowOf(e);if(!row){drop();leave();return}
 const id=+row.dataset.s,conv=+row.dataset.c;if(mine(id,conv)){drop();keep();return}
 if(id===pending)return;
 drop();pending=id;const x=e.clientX,box=row.getBoundingClientRect();
 opening=setTimeout(()=>{pending=0;peek(id,conv,x,box.top,box.bottom)},250)});
sess.addEventListener('mouseleave',()=>{drop();leave()});
sess.addEventListener('mousedown',e=>{down=e.button?null:[e.clientX,e.clientY]});
sess.addEventListener('mouseup',e=>{const row=rowOf(e),at=down;down=null;
 if(row&&at&&Math.hypot(e.clientX-at[0],e.clientY-at[1])<5){aim(+row.dataset.s,+row.dataset.c);enlarge()}});
win.addEventListener('mouseenter',()=>{keep();drop()});win.addEventListener('mouseleave',leave);
$('wbig').onclick=enlarge;$('wclose').onclick=shut;shade.onclick=shut;
addEventListener('keydown',e=>{if(e.key==='Escape'&&view)shut()});
// The loaded model, and where there is more than one a way to load another. The list is the server's (/v1/models):
// the loaded model first, then the rest as registered; the one picked is loaded by the button, and its answer comes
// when the model is in memory, or has failed to be. A model with no artifact cannot be picked.
const mp=$('mp'),stage={dwell:'the loaded model has its turn first',waiting:'new requests wait while the loaded model finishes the ones it has',stopping:'the loaded model is being stopped',loading:'it is being loaded'};
let M=null,pick=null,loading=null,told=null,sig='',seq=0,seen=0,asking=false,age=0;
const many=()=>!!(M&&Array.isArray(M.data)&&M.data.length>1&&M.switch),ok=id=>M.data.some(m=>m.id===id&&m.state!=='missing');
function pickstate(){const sw=M.switch,cur=ok(pick)?pick:M.loaded,off=!!(loading||sw.target||sw.mode==='none');
 const [cls,text]=loading?['wait',`loading ${loading}…`]:sw.target?['wait',`switching to ${sw.target}: ${stage[sw.phase]||'in hand'}`]
  :sw.mode==='none'?['','this server keeps the model it started on']:told?['bad',told]:['',''];
 return{cur,off,cls,text}}
function paintPick(){if(!many()){mp.innerHTML='';return}const p=pickstate();
 mp.innerHTML=`<select id="msel"${p.off?' disabled':''}>`+M.data.map(m=>`<option value="${esc(m.id)}"${m.id===p.cur?' selected':''}${m.state==='missing'?' disabled':''}>${esc(m.id)} · ${m.size_bytes==null?'':gb(m.size_bytes)+' · '}${esc(m.path.split('/').pop())}${m.state==='missing'?' · missing':m.loaded?' · loaded':''}</option>`).join('')
  +`</select> <button id="mgo"${p.off||p.cur===M.loaded?' disabled':''}>Load</button>${p.text?` <span id="mpn" class="${p.cls}">${esc(p.text)}</span>`:''} · `}
async function models(){const k=++seq;try{const j=await (await fetch('/v1/models',{cache:'no-store'})).json();if(k>seen){seen=k;const t=JSON.stringify(j);if(t!==sig){sig=t;M=j;told=null;paintPick()}}}catch(err){}}
mp.addEventListener('change',e=>{if(e.target.id!=='msel')return;pick=e.target.value;$('mgo').disabled=pickstate().off||pick===M.loaded});
mp.addEventListener('click',async e=>{if(e.target.id!=='mgo'||loading||!pick)return;const id=pick;let said=null;loading=id;told=null;paintPick();
 try{const got=await fetch('/v1/models/load',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({model:id})}),j=await got.json().catch(()=>({}));
  if(!got.ok)throw new Error((j.error&&j.error.message)||`the server answered ${got.status}`)}catch(err){said=err.message}
 loading=null;pick=null;await models();told=said;paintPick()});
async function tick(){try{
 const s=await (await fetch('/v1/stats',{cache:'no-store'})).json(),m=s.memory,t=s.throughput;
 $('sub').textContent=`${s.model&&!many()?`${s.model.id}${s.model.switching?' (another model is to be loaded)':''} · `:''}up${Math.floor(s.uptimeSeconds/60)}m · ${s.maxSlots} slots · ${n(s.maxContext)} token context · ${s.maxRows} rows/step`;
 // The models are asked for again when the stats say the loaded one or a switch has changed, and every ten seconds.
 if(s.model&&!asking&&(age++%20===0||M&&M.switch&&(M.loaded!==s.model.id||!!M.switch.target!==!!s.model.switching))){asking=true;models().finally(()=>asking=false)}
 $('dec').textContent=r(t.decodeTokensPerSecond);$('pre').textContent=r(t.prefillTokensPerSecond);
 const live=s.sessions.filter(x=>x.state!=='queued').length;$('act').textContent=live;$('que').textContent=s.queued?`+ ${s.queued} queued`:'active';
 $('step').textContent=t.lastStepMilliseconds?r(t.lastStepMilliseconds):'–';$('rows').textContent=t.lastStepRows?`ms · ${t.lastStepRows} rows`:'ms';
 const parts=[['Weights',m.weightBytes,'var(--a)'],['KV in use',m.kvBytesUsed,'var(--b)'],['Session state',m.stateBytesPerSlot*(live+s.cached.length),'var(--c)'],['Checkpoints',m.checkpointBytes,'var(--d)'],['Scratch',m.scratchBytes,'var(--mute)']];
 const used=parts.reduce((a,p)=>a+p[1],0),tot=m.deviceWorkingSetBytes;
 $('memused').textContent=gb(used);$('memtotal').textContent=`of ${gb(tot)} working set · ${(used/tot*100).toFixed(1)}% · KV pool ${m.kvPagesUsed}/${m.kvPagesTotal} pages (${(m.kvBytesPerToken/1024).toFixed(1)} KiB/token)`;
 $('membar').innerHTML=parts.map(p=>`<i style="width:${p[1]/tot*100}%;background:${p[2]}"></i>`).join('');
 $('memkey').innerHTML=parts.map(p=>`<span><b style="background:${p[2]}"></b>${p[0]} ${mb(p[1])}</span>`).join('');
 $('sessions').innerHTML=table(['#','Conversation','State','Model','Context','Progress','Prompt tok/s now','avg','Generating tok/s now','avg','Tok/step','Drafts accepted, last 5 s','avg','KV','State mem','Age'],
  s.sessions.map(x=>[x.id,x.conversation?`${x.conversation} <span class="queued">turn ${x.turn}</span>`:'–',state(x),esc(x.label),n(x.contextTokens),
   x.state==='decode'?written(x):`<div class="mini"><i style="width:${x.promptTokens?x.evaluatedTokens/x.promptTokens*100:0}%"></i></div> ${n(x.evaluatedTokens)}/${n(x.promptTokens)}${x.cachedTokens?` (${n(x.cachedTokens)} cached)`:''}${left(x)}`,
   x.state==='prefill'?r(x.recentPrefillTokensPerSecond||0):'–',r(x.prefillTokensPerSecond),x.state==='decode'?r(x.recentDecodeTokensPerSecond||0):'–',r(x.decodeTokensPerSecond),x.state==='decode'&&s.speculative?x.tokensPerStep.toFixed(2):'–',pct(x.recentAcceptedTokens,x.recentDraftedTokens),pct(x.acceptedTokens,x.draftedTokens),mb(x.kvBytes),mb(x.stateBytes),x.ageSeconds.toFixed(1)+'s']),'No active sessions',
  i=>` data-s="${s.sessions[i].id}" data-c="${s.sessions[i].conversation||0}"`);
 stats=s;refresh();
 $('cached').innerHTML=table(['Slot','Tokens','Checkpoint at','KV','State mem','Idle'],
  s.cached.map(x=>[x.slot,n(x.tokens),n(x.checkpointTokens),mb(x.kvBytes),mb(x.stateBytes),x.idleSeconds.toFixed(0)+'s']),'Nothing cached yet');
 const d=s.prefixStore;$('disk').textContent=d?`On disk: ${d.entries} prefix${d.entries===1?'':'es'}, ${gb(d.bytes)} of ${gb(d.maxBytes)} · ${n(d.hits)} restored (${n(d.tokensRestored)} tokens) · ${n(d.saves)} written · ${d.directory}`:'Disk prefix store off';
 const o=s.totals;$('totals').innerHTML=table(['Requests','Completed','Prompt tokens','Served from cache','Completion tokens','Steps'],
  [[n(o.requests),n(o.completed),n(o.promptTokens),`${n(o.cachedTokens)} (${o.promptTokens?(o.cachedTokens/o.promptTokens*100).toFixed(0):0}%)`,n(o.completionTokens),n(o.steps)]],'');
 const span=x=>x<120?x.toFixed(0)+' s':(x/60).toFixed(1)+' min';
 $('cdec').textContent=rate(o.generatedTokens,o.decodeBusySeconds);$('cpre').textContent=rate(o.evaluatedTokens,o.prefillBusySeconds);
 $('cdecnote').textContent=`${n(o.generatedTokens)} tokens, all sessions together, over the ${span(o.decodeBusySeconds||0)} in which any were generating`;
 $('cprenote').textContent=`${n(o.evaluatedTokens)} tokens over the ${span(o.prefillBusySeconds||0)} in which prompts were being evaluated`;
 $('acc').textContent=pct(t.recentAcceptedTokens,t.recentDraftedTokens);$('accnote').textContent=o.draftedTokens?`since the server started: ${pct(o.acceptedTokens,o.draftedTokens)}, ${n(o.acceptedTokens)} of ${n(o.draftedTokens)} drafted tokens kept by the model`:'no drafts yet';
 $('adec').textContent=rate(o.completionTokens,o.finishedDecodeSeconds);$('apre').textContent=rate(o.finishedEvaluatedTokens,o.finishedPrefillSeconds);
 const b=o.blocks,per=(t,k)=>k?(t/k).toFixed(1):'–';
 $('blocks').textContent=b&&b.copySteps+b.draftSteps?`Speculative steps: ${n(b.copySteps)} copied from the context (${pct(b.copySteps,b.copySteps+b.draftSteps)}), ${n(b.copyTokens)} tokens, ${per(b.copyTokens,b.copySteps)} a step · ${n(b.draftSteps)} from the draft model, ${n(b.draftTokens)} tokens, ${per(b.draftTokens,b.draftSteps)} a step`:'';
 const h=o.finishedOverhead,c=o.completed,ms=x=>(x/c*1000).toFixed(0)+' ms';
 $('overhead').innerHTML=h&&c?table(['Prepare','Join','Admit and restore','State exports','Lookup','Store copy','Total','First step'],
  [[ms(h.prepare),ms(h.join),ms(h.admit),`${ms(h.stateExports)} <span class="queued">${(h.stateExportCount/c).toFixed(1)} copies</span>`,ms(h.lookupReset),ms(h.storeCopy),
    ms(h.prepare+h.join+h.admit+h.stateExports+h.lookupReset+h.storeCopy),ms(h.firstStep)]],''):'<div class="empty">No finished requests yet</div>';
 $('tiers').innerHTML=table(['Rows','Context','Kind','Steps','Step ms','GPU ms','Fastest ms','Cycle ms'],
  (s.stepTiers||[]).map(x=>[x.rows,x.context,x.prompt?'with prompt rows':'decode only',n(x.steps),r(x.milliseconds),r(x.gpuMilliseconds),r(x.fastestMilliseconds),r(x.cycleMilliseconds)]),'No steps yet');
 $('recent').innerHTML=table(['#','Turns','Model','Context','Reused','Evaluated','Prompt tok/s','Generated','Of which','Generating tok/s','Tok/step','Copied steps','Drafts accepted','Outside steps','Working time','Last finish','Last active'],
  (s.conversations||[]).map(x=>[x.id,n(x.turns)+(x.running?' +1':''),esc(x.label),n(x.promptTokens),n(x.reusedTokens),n(x.evaluatedTokens),rate(x.evaluatedTokens,x.prefillSeconds),
   n(x.completionTokens),made(x),rate(x.completionTokens,x.decodeSeconds),s.speculative&&x.verifySteps?(x.completionTokens/x.verifySteps).toFixed(2):'–',
   x.blocks?pct(x.blocks.copySteps,x.blocks.copySteps+x.blocks.draftSteps):'–',pct(x.acceptedTokens,x.draftedTokens),
   x.overhead&&x.turns?((x.overhead.prepare+x.overhead.join+x.overhead.admit+x.overhead.stateExports+x.overhead.lookupReset+x.overhead.storeCopy)/x.turns*1000).toFixed(0)+' ms a turn':'–',
   span(x.queueSeconds+x.prefillSeconds+x.decodeSeconds),esc(x.lastFinish||'–'),x.running?'<span class="pill decode">running</span>':ago(x.endedSecondsAgo)]),'No conversations yet');
}catch(e){$('sub').textContent='disconnected — retrying'}}
tick();setInterval(tick,500);
</script></body></html>
"""#
}
