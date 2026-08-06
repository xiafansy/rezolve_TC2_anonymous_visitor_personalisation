"""
demo.py (v3) -- watch the homepage evolve click by click.

Replays three scripted visitor journeys through the two-stage engine, prints
the decision after EVERY event, and writes demo.html -- an interactive
step-through you can open in a browser.
"""

import json

from intent_engine import TwoStageEngine
from personalisation import render

FITTED = dict(temperature=1.0, gate=0.43)  # from evaluate_synthetic.py

JOURNEYS = [
    dict(
        name="Instagram scroller who falls for one lamp",
        ctx={"referrer": "instagram", "device": "mobile", "landing": "home", "hour": 21},
        events=[
            {"type": "view", "ts": 0, "item": "fas-001", "category": "fashion"},
            {"type": "view", "ts": 25, "item": "hom-004", "category": "home"},
            {"type": "view", "ts": 60, "item": "ele-002", "category": "electronics"},
            {"type": "view", "ts": 95, "item": "hom-004", "category": "home"},
            {"type": "sort", "ts": 115, "sort_key": "rating"},
            {"type": "view", "ts": 190, "item": "hom-004", "category": "home"},
            {"type": "view", "ts": 320, "item": "hom-007", "category": "home"},
            {"type": "view", "ts": 460, "item": "hom-004", "category": "home"},
        ]),
    dict(
        name="Google searcher on a mission (turns decisive)",
        ctx={"referrer": "google", "device": "mobile", "landing": "product", "hour": 13},
        events=[
            {"type": "view", "ts": 0, "item": "run-002", "category": "running"},
            {"type": "view", "ts": 40, "item": "run-002", "category": "running"},
            {"type": "addtocart", "ts": 70, "item": "run-002", "category": "running"},
        ]),
    dict(
        name="Email deal-hunter working the sale rail",
        ctx={"referrer": "email", "device": "desktop", "landing": "sale", "hour": 19},
        events=[
            {"type": "sort", "ts": 0, "sort_key": "price_asc"},
            {"type": "view", "ts": 20, "item": "sal-001", "category": "sale"},
            {"type": "view", "ts": 75, "item": "sal-003", "category": "sale"},
            {"type": "filter", "ts": 95, "filter_key": "price"},
            {"type": "view", "ts": 130, "item": "sal-001", "category": "sale"},
            {"type": "view", "ts": 210, "item": "sal-001", "category": "sale"},
        ]),
]


def describe(evt):
    t = evt["type"]
    if t == "view":
        return f"views {evt['item']} ({evt['category']})"
    if t == "search":
        return f"searches '{evt['query']}'"
    if t == "sort":
        return f"sorts by {evt['sort_key']}"
    if t == "filter":
        return f"filters on {evt['filter_key']}"
    if t == "addtocart":
        return f"ADDS TO CART: {evt['item']}"
    return t


def snapshot(inf):
    page = render(inf)
    return dict(intent=page.intent, confidence=round(inf.confidence, 3),
                probs={k: round(v, 3) for k, v in inf.probs.items()},
                mode=page.mode, hero=page.hero,
                blocks=[list(b) for b in page.blocks],
                reasons=inf.reasons[:3])


def main():
    export = []
    for j in JOURNEYS:
        print("=" * 72)
        print(f"JOURNEY: {j['name']}")
        print("=" * 72)
        eng = TwoStageEngine(**FITTED)
        steps = []

        inf = eng.start_session(j["ctx"])
        s = snapshot(inf)
        steps.append(dict(label=f"arrival ({j['ctx']['referrer']}, "
                                f"{j['ctx']['device']}, lands on {j['ctx']['landing']})",
                          **s))
        print(f"\n[arrival] -> {s['mode'].upper()} | {inf.explain()}")

        for evt in j["events"]:
            inf = eng.observe(evt)
            s = snapshot(inf)
            steps.append(dict(label=describe(evt), **s))
            print(f"\n[{describe(evt)}]")
            print(f"  -> {s['mode'].upper()} homepage | {inf.intent} "
                  f"(conf {inf.confidence:.0%})")
            print(f"     hero: {s['hero']}")

        export.append(dict(name=j["name"], ctx=j["ctx"], steps=steps))
        print()

    html = TEMPLATE.replace("__DATA__", json.dumps(export))
    with open("demo.html", "w") as f:
        f.write(html)
    print("wrote demo.html -- open it in a browser and step through the journeys")


TEMPLATE = r"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>v3 two-stage intent demo</title>
<style>
  :root { --bg:#0f1115; --card:#181c24; --ink:#e8eaf0; --dim:#8a90a0; --acc:#e07a3f; }
  * { box-sizing:border-box; margin:0; }
  body { background:var(--bg); color:var(--ink); font:15px/1.5 system-ui, sans-serif; padding:24px; }
  h1 { font-size:20px; margin-bottom:4px; } .sub{color:var(--dim); margin-bottom:20px;}
  select,button { background:var(--card); color:var(--ink); border:1px solid #2a3040;
    border-radius:8px; padding:8px 14px; font-size:14px; cursor:pointer; }
  button:hover{border-color:var(--acc)}
  .wrap { display:grid; grid-template-columns: 300px 1fr; gap:18px; margin-top:18px; }
  @media (max-width:800px){ .wrap{grid-template-columns:1fr} }
  .timeline { display:flex; flex-direction:column; gap:6px; }
  .step { background:var(--card); border:1px solid #2a3040; border-radius:10px;
    padding:10px 12px; cursor:pointer; }
  .step.active { border-color:var(--acc); background:#211a14; }
  .step .n { color:var(--dim); font-size:12px; }
  .panel { background:var(--card); border:1px solid #2a3040; border-radius:14px; padding:20px; }
  .mode { display:inline-block; padding:3px 10px; border-radius:99px; font-size:12px;
    letter-spacing:.05em; text-transform:uppercase; background:#2a3040; margin-bottom:10px;}
  .mode.personalised{background:#1d3a2a;color:#7ee2a8}.mode.checkout{background:#3a1d1d;color:#ff9d9d}
  .mode.cold-accent{background:#1d2c3a;color:#8fc7ff}.mode.neutral,.mode.neutral-light{background:#2a3040;color:#aab}
  .bars{margin:14px 0 18px} .bar{display:grid;grid-template-columns:110px 1fr 44px;
    gap:8px;align-items:center;margin:4px 0;font-size:13px}
  .track{background:#0c0e12;border-radius:6px;height:14px;overflow:hidden}
  .fill{height:100%;background:var(--acc);border-radius:6px;transition:width .35s}
  .hero{border:1px dashed #3a4050;border-radius:10px;
    padding:14px;margin:8px 0 14px;font-weight:600}
  .block{background:#12151c;border-radius:10px;padding:10px 14px;margin:6px 0}
  .block small{color:var(--dim);display:block}
  .why{color:var(--dim);font-size:13px;margin-top:14px}
  .nav{display:flex;gap:8px;margin-top:16px}
</style></head><body>
<h1>Two-stage intent &rarr; homepage, click by click</h1>
<div class="sub">Stage 1: cold-start prior from arrival context &middot; Stage 2: incremental scoring of the last few actions &middot; fitted T=1.0, gate=0.43</div>
<select id="j"></select>
<div class="wrap">
  <div class="timeline" id="tl"></div>
  <div class="panel" id="panel"></div>
</div>
<script>
const DATA = __DATA__;
let ji = 0, si = 0;
const jsel = document.getElementById('j');
DATA.forEach((j,i)=>{ const o=document.createElement('option'); o.value=i; o.textContent=j.name; jsel.appendChild(o); });
jsel.onchange = e => { ji = +e.target.value; si = 0; draw(); };
function draw(){
  const j = DATA[ji];
  const tl = document.getElementById('tl'); tl.innerHTML='';
  j.steps.forEach((s,i)=>{
    const d=document.createElement('div'); d.className='step'+(i===si?' active':'');
    d.innerHTML=`<div class="n">${i===0?'stage 1':'event '+i}</div>${s.label}`;
    d.onclick=()=>{si=i;draw();}; tl.appendChild(d);
  });
  const s = j.steps[si];
  const order = Object.keys(s.probs).sort((a,b)=>s.probs[b]-s.probs[a]);
  document.getElementById('panel').innerHTML = `
    <span class="mode ${s.mode}">${s.mode}</span>
    <h2>${s.intent} <span style="color:var(--dim);font-size:15px">conf ${(s.confidence*100).toFixed(0)}%</span></h2>
    <div class="bars">${order.map(k=>`
      <div class="bar"><span>${k}</span>
        <div class="track"><div class="fill" style="width:${s.probs[k]*100}%"></div></div>
        <span>${(s.probs[k]*100).toFixed(0)}%</span></div>`).join('')}</div>
    <div><strong>Homepage now:</strong></div>
    <div class="hero">HERO &mdash; ${s.hero}</div>
    ${s.blocks.map(b=>`<div class="block">${b[0]}<small>${b[1]}</small></div>`).join('')}
    <div class="why"><strong>why:</strong> ${s.reasons.join(' &middot; ')||'&mdash;'}</div>
    <div class="nav">
      <button onclick="si=Math.max(0,si-1);draw()">&larr; prev</button>
      <button onclick="si=Math.min(${j.steps.length-1},si+1);draw()">next &rarr;</button>
    </div>`;
}
draw();
</script></body></html>"""


if __name__ == "__main__":
    main()
