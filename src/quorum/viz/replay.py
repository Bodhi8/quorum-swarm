"""Turn an event log into a single self-contained HTML replay.

No external scripts or fonts: the file opens anywhere, can be hosted as a
static page (the low-cost way to run the public demo), and replays every
tick of every episode: the field over the problem graph, what the scouts
did, what the gate decided and which policy changes the twin accepted.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..events import read_events


def _compact(events: list[dict[str, Any]]) -> dict[str, Any]:
    episodes: dict[int, dict[str, Any]] = {}
    deltas = []
    for e in events:
        ep = episodes.setdefault(e["episode"], {"ticks": [], "gate": None, "end": None, "start": None, "scenario": ""})
        k, d = e["kind"], e["data"]
        if k == "episode_start":
            ep["scenario"] = d.get("scenario", "")
        elif k == "search_start":
            ep["start"] = d
        elif k == "tick":
            ep["ticks"].append({"field": d["field"], "scouts": d["scouts"], "best": d.get("best"),
                                "entropy": d.get("entropy"), "spent": d.get("spent")})
        elif k == "gate":
            ep["gate"] = {kk: d.get(kk) for kk in ("decision", "approved", "p_approve", "reason", "summary", "risk")}
        elif k == "episode_end":
            ep["end"] = {"metrics": d["metrics"], "reward": d["reward"], "arm_usage": d.get("arm_usage", {})}
        elif k == "policy_delta":
            deltas.append({"episode": e["episode"], **d})
    eps = [dict(episode=i, **episodes[i]) for i in sorted(episodes) if episodes[i]["start"] is not None]
    return {"episodes": eps, "deltas": deltas}


TEMPLATE = r"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>__TITLE__</title>
<style>
:root{--bg:#fbfaf7;--fg:#1f1e1b;--mut:#6b6a64;--line:#dedbd2;--card:#fff;--acc:#534AB7;--hot:#D85A30;--ok:#1D9E75;--warn:#BA7517}
@media (prefers-color-scheme:dark){:root{--bg:#1c1b19;--fg:#ecebe6;--mut:#a3a19a;--line:#3a3935;--card:#262522;--acc:#AFA9EC;--hot:#F0997B;--ok:#5DCAA5;--warn:#EF9F27}}
*{box-sizing:border-box}body{margin:0;font:15px/1.55 system-ui,-apple-system,Segoe UI,sans-serif;background:var(--bg);color:var(--fg)}
header{padding:28px 32px 8px}h1{font-size:24px;font-weight:600;margin:0}h2{font-size:15px;font-weight:600;margin:0 0 10px}
.sub{color:var(--mut);margin-top:4px}.wrap{padding:8px 32px 40px;display:grid;gap:18px;grid-template-columns:minmax(0,1.35fr) minmax(0,1fr)}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:16px 18px}
.kpis{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;padding:0 32px}.kpi{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:12px 14px}
.kpi b{display:block;font-size:22px;font-weight:600}.kpi span{color:var(--mut);font-size:13px}
.row{display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin-bottom:8px}select,button,input{font:inherit;color:inherit}
select,button{background:var(--bg);border:1px solid var(--line);border-radius:8px;padding:4px 10px;cursor:pointer}
input[type=range]{flex:1;min-width:120px;accent-color:var(--acc)}.mut{color:var(--mut);font-size:13px}table{width:100%;border-collapse:collapse;font-size:13px}
td,th{text-align:left;padding:5px 6px;border-bottom:1px solid var(--line);vertical-align:top}th{color:var(--mut);font-weight:500}
.pill{display:inline-block;padding:1px 8px;border-radius:99px;font-size:12px;border:1px solid var(--line)}
.auto{color:var(--ok)}.escalate{color:var(--warn)}.block{color:var(--hot)}.full{grid-column:1/-1}
svg text{fill:var(--fg);font-size:11px}.legend{display:flex;gap:14px;flex-wrap:wrap;color:var(--mut);font-size:12px;margin-top:6px}
.sw{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:5px;vertical-align:-1px}
@media (max-width:900px){.wrap{grid-template-columns:1fr}.kpis{grid-template-columns:repeat(2,1fr)}}
</style></head><body>
<header><h1>__TITLE__</h1><div class="sub" id="subtitle"></div></header>
<div class="kpis" id="kpis"></div>
<div class="wrap">
 <div class="card"><h2>Swarm field over the problem graph</h2>
  <div class="row"><label class="mut">Episode</label><select id="ep"></select>
   <button id="play">Play</button><input type="range" id="tick" min="0" value="0"><span class="mut" id="ticklbl"></span></div>
  <div class="mut" id="scenario"></div>
  <svg id="graph" viewBox="0 0 640 420" width="100%" role="img" aria-label="Field over the problem graph"></svg>
  <div class="legend"><span><i class="sw" style="background:var(--acc)"></i>promise (agents converge)</span>
   <span><i class="sw" style="background:var(--hot)"></i>stress / overloaded line</span>
   <span><i class="sw" style="background:var(--mut)"></i>dead end</span><span>dashed = tripped line, ring = protected</span></div>
 </div>
 <div class="card"><h2>This episode</h2><div id="epinfo"></div><h2 style="margin-top:14px">Scouts this tick</h2><div id="scouts"></div></div>
 <div class="card full"><h2>Learning over episodes</h2><svg id="curves" viewBox="0 0 960 220" width="100%" role="img" aria-label="Reward, compute cost and interrupts per episode"></svg>
  <div class="legend"><span><i class="sw" style="background:var(--acc)"></i>reward</span><span><i class="sw" style="background:var(--warn)"></i>compute cost</span><span><i class="sw" style="background:var(--hot)"></i>human interrupt</span><span><i class="sw" style="background:var(--ok)"></i>field policy change accepted by shadow twin</span></div></div>
 <div class="card full"><h2>Gate decisions</h2><div style="max-height:320px;overflow:auto"><table id="gates"></table></div></div>
</div>
<footer style="padding:0 32px 32px;color:var(--mut);font-size:13px">Made with Quorum, the open-core swarm control plane by
 <a href="https://vector1.ai" style="color:var(--acc)">Vector1</a>. The learned gate, self-tuning field and energy domain pack are part of
 <a href="https://vector1.ai" style="color:var(--acc)">Quorum Pro</a>.</footer>
<script>
const RUN = __DATA__;
const $ = id => document.getElementById(id);
const eps = RUN.episodes;
function css(v){return getComputedStyle(document.documentElement).getPropertyValue(v).trim()}
function mean(a){return a.length? a.reduce((x,y)=>x+y,0)/a.length : 0}
function pct(x){return (100*x).toFixed(0)+'%'}
const M = k => eps.filter(e=>e.end).map(e=>e.end.metrics[k]);
$('subtitle').textContent = `${eps.length} episodes · ${RUN.meta.domain} domain · mode: ${RUN.meta.mode}`;
const kp = [["Survived", pct(mean(M('survived')))], ["Human interrupts / episode", mean(M('interrupts')).toFixed(2)],
  ["Compute cost / episode", mean(M('compute_cost')).toFixed(2)], ["Policy changes accepted", RUN.deltas.filter(d=>d.accepted).length+' / '+RUN.deltas.length]];
$('kpis').innerHTML = kp.map(([l,v])=>`<div class="kpi"><b>${v}</b><span>${l}</span></div>`).join('');
eps.forEach((e,i)=>{const o=document.createElement('option');o.value=i;o.textContent=`#${e.episode}`+(e.end&&e.end.metrics.survived<0.5?' (blackout)':'');$('ep').appendChild(o)});

function layout(start){
  const L = (start.view||{}).layout; const nodes = start.nodes; const pos = {};
  if (L){ const xs=nodes.map(n=>L[n][0]), ys=nodes.map(n=>L[n][1]);
    const [x0,x1,y0,y1]=[Math.min(...xs),Math.max(...xs),Math.min(...ys),Math.max(...ys)];
    nodes.forEach(n=>pos[n]=[40+560*(L[n][0]-x0)/((x1-x0)||1), 30+360*(L[n][1]-y0)/((y1-y0)||1)]); }
  else nodes.forEach((n,i)=>{const a=2*Math.PI*i/nodes.length;pos[n]=[320+170*Math.cos(a),210+170*Math.sin(a)]});
  return pos;
}
let cur=0, timer=null;
function draw(){
  const e = eps[+$('ep').value]; const s = e.start; const t = Math.min(cur, e.ticks.length-1); const tk = e.ticks[t];
  $('tick').max = e.ticks.length-1; $('ticklbl').textContent = `tick ${t+1} / ${e.ticks.length}`; $('scenario').textContent = e.scenario;
  const pos = layout(s), v = s.view||{}, idx = {}; s.nodes.forEach((n,i)=>idx[n]=i);
  const pr = tk.field.promise, de = tk.field.dead_end, st = tk.field.stress; const pmax = Math.max(0.2,...pr), smax=Math.max(1e-6,...st);
  let h = '';
  const lines = v.lines || s.edges.map(([a,b])=>[a,b]);
  lines.forEach((ln,li)=>{const [a,b]=ln.map(String); const r = v.rho? v.rho[li] : 0; const out = v.outage===li;
    const col = r>1? css('--hot') : r>0.85? css('--warn') : css('--line');
    h += `<line x1="${pos[a][0]}" y1="${pos[a][1]}" x2="${pos[b][0]}" y2="${pos[b][1]}" stroke="${out?css('--mut'):col}" stroke-width="${out?1.5:1.5+3*Math.max(0,r-0.8)}" ${out?'stroke-dasharray="5 5"':''}><title>line ${li}${v.rho?' loading '+pct(r):''}${out?' (tripped)':''}</title></line>`});
  const best = tk.best? tk.best.nodes : [];
  s.nodes.forEach(n=>{const i=idx[n], [x,y]=pos[n]; const p=pr[i]/pmax, d=Math.min(1,de[i]), sr=st[i]/smax;
    const prot = (v.protected||[]).map(String).includes(n);
    h += `<circle cx="${x}" cy="${y}" r="${10+8*sr}" fill="${css('--hot')}" opacity="${0.12+0.25*sr}"/>`;
    h += `<circle cx="${x}" cy="${y}" r="9" fill="${p>0.02?css('--acc'):css('--card')}" fill-opacity="${0.15+0.85*p}" stroke="${prot?css('--fg'):css('--mut')}" stroke-width="${prot?2.2:0.8}"><title>node ${n}\npromise ${pr[i].toFixed(2)}\ndead end ${de[i].toFixed(2)}\nstress ${st[i].toFixed(2)}</title></circle>`;
    if (d>0.05) h += `<circle cx="${x}" cy="${y}" r="13" fill="none" stroke="${css('--mut')}" stroke-dasharray="2 3" opacity="${d}"/>`;
    if (best.includes(n)) h += `<circle cx="${x}" cy="${y}" r="16" fill="none" stroke="${css('--ok')}" stroke-width="2"/>`;
    h += `<text x="${x+12}" y="${y-10}">${n}</text>`});
  $('graph').innerHTML = h;
  $('scouts').innerHTML = '<table><tr><th>strategy</th><th>looked at</th><th>quality</th></tr>' + tk.scouts.map(sc=>`<tr><td>${sc.arm}</td><td>${sc.nodes.join(', ')||'–'}</td><td>${sc.quality}</td></tr>`).join('') + '</table>';
  const g = e.gate, m = e.end? e.end.metrics : {};
  $('epinfo').innerHTML = `<div class="mut">Best plan so far: ${tk.best? 'robust score '+tk.best.robust.toFixed(2)+' at '+tk.best.nodes.join(', ') : 'none yet'} · spent ${tk.spent.toFixed(2)}</div>` +
    (g? `<p>${g.summary||''}<br><span class="pill ${g.decision}">${g.decision}</span> ${g.approved?'applied':'not applied'} · ${g.reason}</p>`:'<p class="mut">No plan beat doing nothing.</p>') +
    `<p class="mut">Outcome: ${m.survived>0.5?'grid/system held':'<b style="color:var(--hot)">failed</b>'} · overload steps ${m.overload_steps??'–'} · peak ${m.max_rho? pct(m.max_rho):'–'} · interrupts ${m.interrupts}</p>`;
}
function curves(){
  const R = eps.map(e=>e.end?e.end.reward:0), C = eps.map(e=>e.end?e.end.metrics.compute_cost:0), I = eps.map(e=>e.end?e.end.metrics.interrupts:0);
  const W=960,H=220,P=30, n=Math.max(2,eps.length), x=i=>P+(W-2*P)*i/(n-1);
  const sc=(a,lo,hi)=>v=>H-P-(H-2*P)*(v-lo)/((hi-lo)||1);
  const yr=sc(R,Math.min(...R),Math.max(...R)), yc=sc(C,0,Math.max(...C));
  const path=(a,y)=>a.map((v,i)=>(i?'L':'M')+x(i).toFixed(1)+' '+y(v).toFixed(1)).join(' ');
  let h=`<line x1="${P}" y1="${H-P}" x2="${W-P}" y2="${H-P}" stroke="${css('--line')}"/>`;
  RUN.deltas.forEach(d=>{const i=eps.findIndex(e=>e.episode===d.episode); if(i>=0) h+=`<line x1="${x(i)}" y1="${P}" x2="${x(i)}" y2="${H-P}" stroke="${d.accepted?css('--ok'):css('--line')}" stroke-dasharray="${d.accepted?'':'2 4'}"><title>episode ${d.episode}: field policy change ${d.accepted?'accepted':'rejected'} by shadow twin (Δ ${d.delta.toFixed(3)})</title></line>`});
  I.forEach((v,i)=>{ if(v>0) h+=`<circle cx="${x(i)}" cy="${H-P+12}" r="4" fill="${css('--hot')}"><title>episode ${eps[i].episode}: human interrupted</title></circle>`});
  h+=`<path d="${path(C,yc)}" fill="none" stroke="${css('--warn')}" stroke-width="1.5"/><path d="${path(R,yr)}" fill="none" stroke="${css('--acc')}" stroke-width="2"/>`;
  h+=`<text x="${P}" y="18">episode 1</text><text x="${W-P}" y="18" text-anchor="end">episode ${eps.length}</text>`;
  $('curves').innerHTML=h;
}
function gates(){
  $('gates').innerHTML='<tr><th>episode</th><th>plan</th><th>gate</th><th>p(approve)</th><th>applied</th><th>reason</th></tr>'+
   eps.filter(e=>e.gate).map(e=>`<tr><td>${e.episode}</td><td>${e.gate.summary}</td><td><span class="pill ${e.gate.decision}">${e.gate.decision}</span></td><td>${(+e.gate.p_approve).toFixed(2)}</td><td>${e.gate.approved?'yes':'no'}</td><td class="mut">${e.gate.reason}</td></tr>`).join('');
}
$('ep').onchange=()=>{cur=0;$('tick').value=0;draw()}; $('tick').oninput=ev=>{cur=+ev.target.value;draw()};
$('play').onclick=()=>{ if(timer){clearInterval(timer);timer=null;$('play').textContent='Play';return}
  $('play').textContent='Pause'; timer=setInterval(()=>{const e=eps[+$('ep').value]; if(cur<e.ticks.length-1){cur++;$('tick').value=cur;draw()} else {const nx=+$('ep').value+1; if(nx<eps.length){$('ep').value=nx;cur=0;$('tick').value=0;draw()} else {clearInterval(timer);timer=null;$('play').textContent='Play'}}},700)};
draw(); curves(); gates();
</script></body></html>"""


def render(events_path: str | Path, out_path: str | Path, title: str = "Quorum run replay",
           meta: dict[str, Any] | None = None) -> Path:
    data = _compact(read_events(events_path))
    data["meta"] = {"domain": "?", "mode": "?", **(meta or {})}
    html = TEMPLATE.replace("__TITLE__", title).replace("__DATA__", json.dumps(data, separators=(",", ":")))
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    return out
