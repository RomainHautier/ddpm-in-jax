import json
M = json.load(open("/home/rhautier/ddpm-jax/base_results/ood_metrics.json"))
# compact payload: scalars + downsampled curves
RES = ["500","1000","2000"]
MODELS = ["base","grad_frozen60","grad_full60","field_frozen60","field_full60"]
payload = {"re": {}}
for re in RES:
    node = M["re"][re]
    gt = node["GT"]
    def hik(sp): return sum(sp[32:])
    ghik = hik(gt["spectrum"])
    entry = {"gt_residual": gt["residual"], "gt_spectrum": gt["spectrum"], "gt_hist": gt["hist"], "models": {}}
    for m in MODELS:
        if m not in node["models"]: continue
        d = node["models"][m]
        entry["models"][m] = {"residual": d["residual"], "mse": d["mse"],
                              "spectrum": d["spectrum"], "hist": d["hist"],
                              "hik": hik(d["spectrum"])/ghik}
    payload["re"][re] = entry
DATA = json.dumps(payload)

html = '''<style>
:root{
  --bg:#f6f8fa; --panel:#ffffff; --ink:#0e1620; --muted:#5b6675; --faint:#8b95a3;
  --line:#e3e8ee; --line2:#eef2f6; --accent:#0891b2; --accent-soft:#cff3fb;
  --helps:#0f9d8f; --hurts:#c26a11; --grid:#eceff3;
  --font-sans:system-ui,-apple-system,'Segoe UI',Roboto,sans-serif;
  --font-mono:ui-monospace,'SF Mono','JetBrains Mono',Menlo,monospace;
}
@media (prefers-color-scheme:dark){:root{
  --bg:#0b0f14; --panel:#121821; --ink:#e6edf3; --muted:#9aa7b5; --faint:#697585;
  --line:#222c38; --line2:#1a222c; --accent:#22b8cf; --accent-soft:#0c2b33;
  --helps:#2bb7a6; --hurts:#e0902f; --grid:#1b232d;
}}
:root[data-theme="light"]{
  --bg:#f6f8fa; --panel:#ffffff; --ink:#0e1620; --muted:#5b6675; --faint:#8b95a3;
  --line:#e3e8ee; --line2:#eef2f6; --accent:#0891b2; --accent-soft:#cff3fb;
  --helps:#0f9d8f; --hurts:#c26a11; --grid:#eceff3;
}
:root[data-theme="dark"]{
  --bg:#0b0f14; --panel:#121821; --ink:#e6edf3; --muted:#9aa7b5; --faint:#697585;
  --line:#222c38; --line2:#1a222c; --accent:#22b8cf; --accent-soft:#0c2b33;
  --helps:#2bb7a6; --hurts:#e0902f; --grid:#1b232d;
}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font-family:var(--font-sans);
  line-height:1.5;-webkit-font-smoothing:antialiased}
.wrap{max-width:1080px;margin:0 auto;padding:48px 24px 80px}
.eyebrow{font-family:var(--font-mono);font-size:12px;letter-spacing:.14em;text-transform:uppercase;
  color:var(--accent);font-weight:600}
h1{font-size:clamp(28px,4vw,42px);line-height:1.08;margin:14px 0 0;font-weight:800;letter-spacing:-.02em;
  text-wrap:balance;max-width:20ch}
.lede{color:var(--muted);font-size:17px;max-width:62ch;margin:16px 0 0}
.meta{margin-top:18px;display:flex;flex-wrap:wrap;gap:8px}
.tag{font-family:var(--font-mono);font-size:11.5px;color:var(--muted);border:1px solid var(--line);
  border-radius:999px;padding:4px 10px;background:var(--panel)}
section{margin-top:52px}
.sec-h{font-family:var(--font-mono);font-size:12px;letter-spacing:.12em;text-transform:uppercase;
  color:var(--faint);font-weight:600;margin:0 0 16px;display:flex;align-items:center;gap:12px}
.sec-h::after{content:"";flex:1;height:1px;background:var(--line)}
.cards{display:grid;grid-template-columns:1fr 1fr;gap:16px}
@media(max-width:680px){.cards{grid-template-columns:1fr}}
.card{background:var(--panel);border:1px solid var(--line);border-radius:14px;padding:22px 22px 20px;
  position:relative;overflow:hidden}
.card .rail{position:absolute;left:0;top:0;bottom:0;width:4px}
.card h3{margin:0 0 4px;font-size:15px;font-family:var(--font-mono);font-weight:600}
.card .big{font-size:34px;font-weight:800;letter-spacing:-.02em;font-family:var(--font-mono);
  font-variant-numeric:tabular-nums;margin:6px 0 2px}
.card p{margin:8px 0 0;color:var(--muted);font-size:13.5px;max-width:40ch}
.mtx{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums}
.mtx th,.mtx td{padding:11px 12px;text-align:right;border-bottom:1px solid var(--line2)}
.mtx th{font-family:var(--font-mono);font-size:11px;letter-spacing:.06em;text-transform:uppercase;
  color:var(--faint);font-weight:600;border-bottom:1px solid var(--line)}
.mtx td:first-child,.mtx th:first-child{text-align:left;font-family:var(--font-mono);font-size:13px}
.mtx tbody tr:hover{background:var(--line2)}
.delta{font-family:var(--font-mono);font-weight:700;font-variant-numeric:tabular-nums;
  padding:3px 9px;border-radius:6px;display:inline-block;min-width:64px;font-size:13px}
.scale-wrap{margin-top:14px;display:flex;align-items:center;gap:12px;color:var(--faint);
  font-family:var(--font-mono);font-size:11px}
.scale{height:8px;flex:1;border-radius:4px;max-width:260px;
  background:linear-gradient(90deg,var(--helps),var(--line),var(--hurts))}
.panels{display:grid;grid-template-columns:1fr;gap:16px}
.panel{background:var(--panel);border:1px solid var(--line);border-radius:14px;padding:20px}
.panel .ph{display:flex;align-items:baseline;justify-content:space-between;gap:12px;margin-bottom:4px}
.panel .ph .re{font-size:20px;font-weight:800;letter-spacing:-.01em}
.panel .ph .re small{font-family:var(--font-mono);font-weight:500;color:var(--faint);font-size:12px;
  margin-left:8px;letter-spacing:.04em}
.panel .ph .gtres{font-family:var(--font-mono);font-size:12px;color:var(--muted)}
.charts{display:grid;grid-template-columns:1fr 1fr 1fr;gap:20px;margin-top:16px}
@media(max-width:820px){.charts{grid-template-columns:1fr}}
.chart h4{margin:0 0 2px;font-size:12px;font-family:var(--font-mono);font-weight:600;color:var(--ink)}
.chart .sub{font-size:11px;color:var(--faint);margin:0 0 8px;font-family:var(--font-mono)}
svg{width:100%;height:auto;display:block;overflow:visible}
.legend{display:flex;flex-wrap:wrap;gap:10px 16px;margin-top:24px;padding:16px 18px;background:var(--panel);
  border:1px solid var(--line);border-radius:12px}
.lg{display:flex;align-items:center;gap:7px;font-family:var(--font-mono);font-size:12px;color:var(--muted)}
.lg .sw{width:14px;height:3px;border-radius:2px}
.foot{margin-top:40px;color:var(--faint);font-size:12.5px;font-family:var(--font-mono);
  border-top:1px solid var(--line);padding-top:18px;line-height:1.7}
.note{color:var(--muted);font-size:14px;max-width:64ch;margin:14px 0 0}
b.h{color:var(--helps)} b.x{color:var(--hurts)}
</style>

<div class="wrap">
  <div class="eyebrow">Physics-conditioned diffusion &middot; OOD generalisation</div>
  <h1>Does residual conditioning survive out of distribution?</h1>
  <p class="lede">Five reconstruction models &mdash; a plain diffusion base and four residual-conditioned
  adapters, all trained at Re=1000 &mdash; tested on Kolmogorov flow at Re&nbsp;500, 1000, and 2000.
  Each conditional model is fed the Navier&ndash;Stokes residual evaluated at the <em>target</em> Reynolds
  number. The question: does the learned physics correction still help when the flow is out of distribution?</p>
  <div class="meta">
    <span class="tag">8 sequences / Re</span><span class="tag">2544 frames in-dist</span>
    <span class="tag">learned guidance only (&lambda;=0)</span><span class="tag">metric: mean NS residual&sup2;</span>
  </div>

  <section>
    <div class="sec-h">The two findings</div>
    <div class="cards">
      <div class="card"><div class="rail" style="background:var(--helps)"></div>
        <h3>grad_frozen60 &middot; the reliable one</h3>
        <div class="big" style="color:var(--helps)">&minus;5.5%</div>
        <p>The faithful Shu&nbsp;et&nbsp;al. adapter (gradient signal, frozen base) lowers the PDE residual
        by a near-constant <b>&minus;5.8 / &minus;5.4 / &minus;5.3%</b> across Re 500 / 1000 / 2000 &mdash;
        Reynolds-number-agnostic. It generalises.</p>
      </div>
      <div class="card"><div class="rail" style="background:var(--accent)"></div>
        <h3>field_full60 &middot; the OOD one</h3>
        <div class="big" style="color:var(--accent)">&minus;9.7%</div>
        <p>The ENS-style raw-residual signal with full fine-tuning barely moves in-distribution, but its
        benefit <em>grows with turbulence</em>: &minus;0.4% &rarr; &minus;4.1% &rarr; <b>&minus;9.7%</b> at
        Re=2000 &mdash; the largest gain in the study, exactly where the flow is most OOD.</p>
      </div>
    </div>
    <p class="note">Both beat the unconditional base. The other two adapters don't:
    <b class="x">grad_full60</b> consistently <i>degrades</i> physics (unfreezing the base with the gradient
    signal is the wrong move), and <b>field_frozen60</b> sits at roughly no-change everywhere.</p>
  </section>

  <section>
    <div class="sec-h">Residual vs the plain base &mdash; every model, every Re</div>
    <table class="mtx"><thead><tr>
      <th>model</th><th>Re 500</th><th>Re 1000<small style="font-weight:400"> (in-dist)</small></th>
      <th>Re 2000</th><th>signal / scope</th></tr></thead>
      <tbody id="mtxbody"></tbody></table>
    <div class="scale-wrap"><span>helps (lower residual)</span><span class="scale"></span><span>hurts</span>
      <span style="margin-left:auto">&Delta; = (model &minus; base) / base</span></div>
  </section>

  <section>
    <div class="sec-h">Per-Re detail &mdash; residual, spectrum match, vorticity</div>
    <div class="panels" id="panels"></div>
  </section>

  <div class="legend" id="legend"></div>

  <div class="foot" id="foot"></div>
</div>

<script id="ood-data" type="application/json">__DATA__</script>
<script>
const D = JSON.parse(document.getElementById('ood-data').textContent);
const MODELS = ["base","grad_frozen60","grad_full60","field_frozen60","field_full60"];
const LABEL = {base:"base",grad_frozen60:"grad·frozen",grad_full60:"grad·full",
  field_frozen60:"field·frozen",field_full60:"field·full"};
const SCOPE = {base:"— unconditional",grad_frozen60:"gradient · frozen",grad_full60:"gradient · full-ft",
  field_frozen60:"field (ENS) · frozen",field_full60:"field (ENS) · full-ft"};
const COL = {base:"var(--faint)",grad_frozen60:"#0f9d8f",grad_full60:"#c26a11",
  field_frozen60:"#7b8794",field_full60:"var(--accent)"};
const RES = ["500","1000","2000"];
const css = v => getComputedStyle(document.documentElement).getPropertyValue(v).trim();

function deltaCell(v){
  const pct = v*100;
  const mag = Math.min(Math.abs(pct)/12,1);
  const helps = pct<0;
  const base = helps? css('--helps'):css('--hurts');
  const bg = `color-mix(in srgb, ${base} ${18+mag*46}%, transparent)`;
  const s = (pct>=0?'+':'') + pct.toFixed(1) + '%';
  return `<span class="delta" style="background:${bg};color:${helps?'var(--helps)':'var(--hurts)'}">${s}</span>`;
}
// matrix
const tb = document.getElementById('mtxbody');
MODELS.forEach(m=>{
  const cells = RES.map(re=>{
    const nd=D.re[re]; if(!nd.models[m]) return '<td>—</td>';
    const b=nd.models['base'].residual, v=nd.models[m].residual;
    return m==='base'? `<td style="color:var(--faint);font-family:var(--font-mono)">${v.toFixed(1)}</td>`
                     : `<td>${deltaCell(v/b-1)}</td>`;
  }).join('');
  tb.insertAdjacentHTML('beforeend',
    `<tr><td>${LABEL[m]}</td>${cells}<td style="color:var(--muted);font-family:var(--font-mono);font-size:12px">${SCOPE[m]}</td></tr>`);
});

// ---- charts ----
function svg(w,h){const s=document.createElementNS('http://www.w3.org/2000/svg','svg');
  s.setAttribute('viewBox',`0 0 ${w} ${h}`);return s;}
function el(t,a){const e=document.createElementNS('http://www.w3.org/2000/svg',t);
  for(const k in a)e.setAttribute(k,a[k]);return e;}

function residualBars(re){
  const nd=D.re[re], W=300,H=200,ml=8,mr=8,mt=14,mb=40, iw=W-ml-mr, ih=H-mt-mb;
  const s=svg(W,H); const vals=MODELS.map(m=>nd.models[m].residual);
  const gt=nd.gt_residual, mx=Math.max(...vals,gt)*1.12;
  const bw=iw/MODELS.length;
  // GT reference line
  const gy=mt+ih-(gt/mx)*ih;
  s.appendChild(el('line',{x1:ml,y1:gy,x2:ml+iw,y2:gy,stroke:css('--accent'),'stroke-width':1.3,'stroke-dasharray':'4 3',opacity:.9}));
  s.appendChild(el('text',{x:ml+2,y:gy-5,fill:css('--accent'),'font-size':10,'font-family':'var(--font-mono)'})).textContent=`GT ${gt.toFixed(1)}`;
  MODELS.forEach((m,i)=>{
    const v=nd.models[m].residual, bh=(v/mx)*ih, x=ml+i*bw+bw*0.16, y=mt+ih-bh, bwid=bw*0.68;
    s.appendChild(el('rect',{x,y,width:bwid,height:bh,rx:3,fill:COL[m],opacity:.92}));
    const t=el('text',{x:x+bwid/2,y:H-mb+14,fill:css('--muted'),'font-size':9,'text-anchor':'middle','font-family':'var(--font-mono)'});
    t.textContent=LABEL[m].replace('·','\\n'); 
    // two-line label
    const parts=LABEL[m].split('·');
    t.textContent=parts[0];
    s.appendChild(t);
    if(parts[1]){const t2=el('text',{x:x+bwid/2,y:H-mb+25,fill:css('--faint'),'font-size':8.5,'text-anchor':'middle','font-family':'var(--font-mono)'});t2.textContent=parts[1];s.appendChild(t2);}
    const vt=el('text',{x:x+bwid/2,y:y-4,fill:css('--muted'),'font-size':9.5,'text-anchor':'middle','font-family':'var(--font-mono)'});vt.textContent=v.toFixed(1);s.appendChild(vt);
  });
  return s;
}
function line(pts){return pts.map((p,i)=>(i?'L':'M')+p[0].toFixed(1)+' '+p[1].toFixed(1)).join(' ');}
function spectrumChart(re){
  const nd=D.re[re], W=300,H=200,ml=30,mr=8,mt=12,mb=26,iw=W-ml-mr,ih=H-mt-mb;
  const s=svg(W,H); const gsp=nd.gt_spectrum, NK=gsp.length;
  const kmin=1,kmax=NK-1, lx=k=>ml+(Math.log(k)-Math.log(kmin))/(Math.log(kmax)-Math.log(kmin))*iw;
  const ymax=2, ly=r=>mt+ih-Math.min(r,ymax)/ymax*ih;
  // grid: y=1 and y=0.5
  [0.5,1,1.5].forEach(g=>{const yy=ly(g);s.appendChild(el('line',{x1:ml,y1:yy,x2:ml+iw,y2:yy,stroke:css('--grid'),'stroke-width':1}));
    const t=el('text',{x:ml-5,y:yy+3,fill:css('--faint'),'font-size':9,'text-anchor':'end','font-family':'var(--font-mono)'});t.textContent=g.toFixed(1);s.appendChild(t);});
  // k=32 marker
  const x32=lx(32); s.appendChild(el('line',{x1:x32,y1:mt,x2:x32,y2:mt+ih,stroke:css('--faint'),'stroke-width':1,'stroke-dasharray':'2 3',opacity:.6}));
  const t32=el('text',{x:x32,y:mt+ih+16,fill:css('--faint'),'font-size':8.5,'text-anchor':'middle','font-family':'var(--font-mono)'});t32.textContent='k=32';s.appendChild(t32);
  // perfect-match line y=1 emphasised
  s.appendChild(el('line',{x1:ml,y1:ly(1),x2:ml+iw,y2:ly(1),stroke:css('--accent'),'stroke-width':1,opacity:.5}));
  MODELS.forEach(m=>{
    const sp=nd.models[m].spectrum, pts=[];
    for(let k=kmin;k<=kmax;k++){const r=sp[k]/(gsp[k]||1e-30);pts.push([lx(k),ly(r)]);}
    s.appendChild(el('path',{d:line(pts),fill:'none',stroke:COL[m],'stroke-width':m==='base'?1.6:1.4,opacity:m==='base'?.9:.85}));
  });
  const xl=el('text',{x:ml+iw/2,y:H-2,fill:css('--faint'),'font-size':9,'text-anchor':'middle','font-family':'var(--font-mono)'});xl.textContent='wavenumber k (log)';s.appendChild(xl);
  return s;
}
function pdfChart(re){
  const nd=D.re[re],W=300,H=200,ml=8,mr=8,mt=12,mb=26,iw=W-ml-mr,ih=H-mt-mb;
  const s=svg(W,H); const nb=nd.gt_hist.length; const xs=i=>ml+i/(nb-1)*iw;
  // log-y: find range over GT+models
  let vmax=0,vmin=1;const all=[nd.gt_hist,...MODELS.map(m=>nd.models[m].hist)];
  all.forEach(h=>h.forEach(v=>{if(v>vmax)vmax=v;}));
  const lo=1e-5, ly=v=>{const c=Math.max(v,lo);return mt+ih-(Math.log10(c)-Math.log10(lo))/(Math.log10(vmax)-Math.log10(lo))*ih;};
  [1e-1,1e-2,1e-3,1e-4].forEach(g=>{if(g>vmax)return;const yy=ly(g);s.appendChild(el('line',{x1:ml,y1:yy,x2:ml+iw,y2:yy,stroke:css('--grid'),'stroke-width':1}));});
  // GT filled-ish (thick)
  const gpts=nd.gt_hist.map((v,i)=>[xs(i),ly(v)]);
  s.appendChild(el('path',{d:line(gpts),fill:'none',stroke:css('--ink'),'stroke-width':1.8,opacity:.9}));
  MODELS.forEach(m=>{if(m==='base')return;
    const pts=nd.models[m].hist.map((v,i)=>[xs(i),ly(v)]);
    s.appendChild(el('path',{d:line(pts),fill:'none',stroke:COL[m],'stroke-width':1.2,opacity:.8}));
  });
  const bpts=nd.models['base'].hist.map((v,i)=>[xs(i),ly(v)]);
  s.appendChild(el('path',{d:line(bpts),fill:'none',stroke:COL['base'],'stroke-width':1.2,opacity:.8,'stroke-dasharray':'3 2'}));
  const xl=el('text',{x:ml+iw/2,y:H-2,fill:css('--faint'),'font-size':9,'text-anchor':'middle','font-family':'var(--font-mono)'});xl.textContent='vorticity  (PDF, log)';s.appendChild(xl);
  return s;
}

const panels=document.getElementById('panels');
RES.forEach(re=>{
  const nd=D.re[re];
  const p=document.createElement('div');p.className='panel';
  p.innerHTML=`<div class="ph"><div class="re">Re&nbsp;${re}${re==='1000'?'<small>in-distribution</small>':'<small>out of distribution</small>'}</div>
    <div class="gtres">GT residual ${nd.gt_residual.toFixed(2)} &middot; hi-k ret ${(nd.models.base.hik).toFixed(2)}&rarr;base</div></div>
    <div class="charts">
      <div class="chart"><h4>PDE residual</h4><div class="sub">lower = more physical &middot; cyan = GT</div><div class="c1"></div></div>
      <div class="chart"><h4>Energy spectrum E(k) / GT</h4><div class="sub">1.0 = perfect match</div><div class="c2"></div></div>
      <div class="chart"><h4>Vorticity PDF</h4><div class="sub">black = GT &middot; dashed = base</div><div class="c3"></div></div>
    </div>`;
  panels.appendChild(p);
  p.querySelector('.c1').appendChild(residualBars(re));
  p.querySelector('.c2').appendChild(spectrumChart(re));
  p.querySelector('.c3').appendChild(pdfChart(re));
});
// legend
const lg=document.getElementById('legend');
lg.innerHTML=MODELS.map(m=>`<span class="lg"><span class="sw" style="background:${COL[m]}"></span>${LABEL[m]}</span>`).join('')
  +`<span class="lg"><span class="sw" style="background:var(--accent)"></span>GT reference</span>`;
document.getElementById('foot').innerHTML=
  `Task: sparse reconstruction (K=3, S=[150,100,50], 1024-pt sample + NN-fill). All models trained at Re=1000; `
  +`residual signal evaluated at the target Re (viscosity 1/Re). Datasets: kf_2d_re1000 (seqs 32–39), `
  +`generated kf_re500 / kf_re2000 (seqs 0–7). MSE differences between models are &lt;2% everywhere — the signal is in the residual.`;
</script>'''

html = html.replace("__DATA__", DATA)
open("/home/rhautier/ddpm-jax/base_results/ood_results.html","w").write(html)
print("wrote base_results/ood_results.html", len(html), "bytes")
