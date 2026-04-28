"""Balancer web app — run with: uvicorn app:app --reload --port 8000"""
from __future__ import annotations
import json
import threading
import time
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse

RESULTS_DIR = Path(__file__).parent / "results"
DATA_DIR = Path(__file__).parent / "data"

app = FastAPI(title="Balancer")

_state: dict = {"status": "idle", "step": "", "progress": {"batch": 0, "total": 0}, "error": None}
_lock = threading.Lock()


def _set(**kw):
    with _lock:
        _state.update(kw)


# ── API ────────────────────────────────────────────────────────────────────

@app.get("/api/status")
def api_status():
    with _lock:
        s = dict(_state)
    s["has_results"] = (RESULTS_DIR / "gemini_biased.json").exists()
    s["has_fix"] = (RESULTS_DIR / "math_fixed.json").exists()
    if s["has_results"]:
        with open(RESULTS_DIR / "gemini_biased.json") as f:
            s["meta"] = json.load(f).get("metadata", {})
    return JSONResponse(s)


@app.get("/api/metrics")
def api_metrics():
    from balancer.analyzer import analyze
    bp = RESULTS_DIR / "gemini_biased.json"
    mp = RESULTS_DIR / "math_fixed.json"
    if not bp.exists():
        return JSONResponse({"error": "no results"}, status_code=404)

    def _r(report):
        return {
            "grade": report.grade,
            "dpg": report.demographic_parity_gap.get("white_vs_black", 0),
            "dir": report.disparate_impact_ratio,
            "white_avg": report.white_coded_avg,
            "black_avg": report.black_coded_avg,
            "white_hire": round(report.white_hire_rate * 100, 1),
            "black_hire": round(report.black_hire_rate * 100, 1),
            "p_value": report.p_value,
            "significant": report.significant,
            "consistency": round(report.consistency_rate * 100, 1),
        }

    out = {"biased": _r(analyze(bp))}
    if mp.exists():
        out["math"] = _r(analyze(mp))

    np_ = RESULTS_DIR / "audit_narrative.json"
    if np_.exists():
        with open(np_) as f:
            out["narrative"] = json.load(f)
    return JSONResponse(out)


@app.post("/api/run")
def api_run():
    with _lock:
        if _state["status"] == "running":
            return JSONResponse({"error": "already running"}, status_code=409)
        _state.update({"status": "running", "step": "loading", "progress": {"batch": 0, "total": 0}, "error": None})

    def _work():
        try:
            from balancer.generator import generate_pairs, save_pairs, load_pairs, OUTPUT_PATH
            from balancer import ollama_client

            pairs = load_pairs() if OUTPUT_PATH.exists() else generate_pairs(n=100, seed=42)
            if not OUTPUT_PATH.exists():
                save_pairs(pairs)

            _set(step="scoring")

            def cb(n, t):
                _set(progress={"batch": n, "total": t})

            t0 = time.time()
            scores = ollama_client.score_all_pairs(pairs, batch_size=5, progress_cb=cb)
            elapsed = time.time() - t0

            _set(step="saving")
            RESULTS_DIR.mkdir(exist_ok=True)
            with open(RESULTS_DIR / "gemini_biased.json", "w") as f:
                json.dump({
                    "metadata": {
                        "timestamp": datetime.now().isoformat(timespec="seconds"),
                        "model": f"Ollama ({ollama_client.OLLAMA_MODEL})",
                        "total_pairs": len(scores),
                        "scoring_time_seconds": round(elapsed, 1),
                    },
                    "scores": scores,
                }, f, indent=2)
            _set(status="done", step="done")
        except Exception as e:
            _set(status="error", step="", error=str(e))

    threading.Thread(target=_work, daemon=True).start()
    return JSONResponse({"ok": True})


@app.post("/api/fix")
def api_fix():
    from balancer.analyzer import analyze
    from balancer.mitigator import (reweigh, save_math_fixed,
                                    generate_audit_narrative, save_audit_narrative,
                                    save_self_corrected)
    bp = RESULTS_DIR / "gemini_biased.json"
    if not bp.exists():
        return JSONResponse({"error": "no results"}, status_code=404)
    with open(bp) as f:
        bd = json.load(f)

    br = analyze(bp)
    save_math_fixed(reweigh(bd))
    mr = analyze(RESULTS_DIR / "math_fixed.json")

    sp = RESULTS_DIR / "gemini_self_corrected.json"
    if not sp.exists():
        save_self_corrected({"metadata": {"method": "skipped"}, "scores": bd["scores"]})
    sr = analyze(sp)

    narr = generate_audit_narrative(br, mr, sr, bd.get("metadata", {}).get("model", "unknown"))
    save_audit_narrative(narr)
    return JSONResponse({"ok": True})


@app.get("/report", response_class=HTMLResponse)
def report_page():
    from balancer.reporter import generate_report
    path = generate_report(RESULTS_DIR / "report.html")
    return HTMLResponse(path.read_text())


@app.get("/", response_class=HTMLResponse)
def index():
    return HTMLResponse(_HTML)


# ── Frontend ───────────────────────────────────────────────────────────────

_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Balancer — AI Bias Detection</title>
<style>
:root{--bg:#0f1117;--s:#1a1d27;--b:#2a2d3a;--t:#e2e8f0;--m:#94a3b8;--a:#6366f1;--g:#22c55e;--r:#ef4444;--y:#f59e0b;--p:#f472b6;--c:#60a5fa}
*{box-sizing:border-box;margin:0;padding:0}
body{background:var(--bg);color:var(--t);font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;font-size:14px;line-height:1.6;min-height:100vh}
.layout{display:grid;grid-template-columns:220px 1fr;min-height:100vh}
.sidebar{background:var(--s);border-right:1px solid var(--b);padding:24px 16px;display:flex;flex-direction:column;gap:8px;position:sticky;top:0;height:100vh}
.logo{font-size:20px;font-weight:800;margin-bottom:16px;letter-spacing:-0.3px}
.logo span{color:#818cf8}
.nav-btn{width:100%;padding:9px 14px;border-radius:8px;border:1px solid var(--b);background:transparent;color:var(--t);font-size:13px;font-weight:500;cursor:pointer;text-align:left;transition:all .15s;display:flex;align-items:center;gap:8px}
.nav-btn:hover{background:rgba(99,102,241,.12);border-color:#6366f1}
.nav-btn.primary{background:#6366f1;border-color:#6366f1;color:#fff}
.nav-btn.primary:hover{background:#4f46e5}
.nav-btn:disabled{opacity:.4;cursor:not-allowed}
.nav-sep{height:1px;background:var(--b);margin:8px 0}
.main{padding:32px;overflow-y:auto}
.page-title{font-size:22px;font-weight:700;margin-bottom:4px}
.page-sub{color:var(--m);font-size:13px;margin-bottom:28px}
.card{background:var(--s);border:1px solid var(--b);border-radius:12px;padding:20px;margin-bottom:16px}
.card-title{font-size:12px;font-weight:700;color:var(--m);text-transform:uppercase;letter-spacing:.6px;margin-bottom:12px}
.status-row{display:flex;align-items:center;gap:10px;margin-bottom:8px}
.dot{width:8px;height:8px;border-radius:50%;background:var(--m);flex-shrink:0}
.dot.green{background:var(--g)}
.dot.red{background:var(--r)}
.dot.yellow{background:var(--y);animation:pulse 1s infinite}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.4}}
.progress-bar{height:6px;background:var(--b);border-radius:3px;overflow:hidden;margin-top:8px}
.progress-fill{height:100%;background:var(--a);border-radius:3px;transition:width .3s}
.progress-label{font-size:11px;color:var(--m);margin-top:4px}
.metrics-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:12px}
.metric{background:var(--bg);border:1px solid var(--b);border-radius:10px;padding:14px}
.metric-label{font-size:11px;color:var(--m);text-transform:uppercase;letter-spacing:.5px;margin-bottom:4px}
.metric-val{font-size:24px;font-weight:800;line-height:1.1}
.metric-sub{font-size:11px;color:var(--m);margin-top:2px}
.cmp{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-top:12px}
.cmp-col{background:var(--bg);border:1px solid var(--b);border-radius:10px;padding:14px}
.cmp-col.after{border-color:rgba(34,197,94,.3);background:rgba(34,197,94,.04)}
.cmp-label{font-size:11px;font-weight:700;color:var(--m);margin-bottom:10px;text-transform:uppercase;letter-spacing:.5px}
.cmp-label.red{color:var(--r)}
.cmp-label.green{color:var(--g)}
.row{display:flex;justify-content:space-between;align-items:center;padding:5px 0;border-bottom:1px solid var(--b);font-size:13px}
.row:last-child{border-bottom:none}
.row-k{color:var(--m)}
.tag{font-size:10px;padding:2px 6px;border-radius:3px;font-weight:700;margin-left:6px}
.tag.pass{background:rgba(34,197,94,.15);color:var(--g);border:1px solid rgba(34,197,94,.3)}
.tag.fail{background:rgba(239,68,68,.15);color:var(--r);border:1px solid rgba(239,68,68,.3)}
.verdict{border-radius:8px;padding:14px 16px;margin-top:12px;border-left:3px solid}
.verdict.red{background:rgba(239,68,68,.07);border-color:var(--r)}
.verdict.green{background:rgba(34,197,94,.07);border-color:var(--g)}
.verdict-title{font-weight:700;margin-bottom:4px}
.verdict-body{font-size:13px;color:var(--m);line-height:1.6}
.sev{display:inline-block;font-size:11px;font-weight:700;padding:2px 8px;border-radius:4px;margin-left:8px}
.sev.CRITICAL{background:rgba(239,68,68,.2);color:#fca5a5;border:1px solid var(--r)}
.sev.HIGH{background:rgba(249,115,22,.2);color:#fdba74;border:1px solid #f97316}
.sev.MEDIUM{background:rgba(245,158,11,.2);color:#fde68a;border:1px solid var(--y)}
.sev.LOW{background:rgba(34,197,94,.2);color:#86efac;border:1px solid var(--g)}
.hidden{display:none}
.g{color:var(--g)} .r{color:var(--r)} .y{color:var(--y)} .c{color:var(--c)} .p{color:var(--p)} .m{color:var(--m)}
.grade-A{color:var(--g)} .grade-B{color:#84cc16} .grade-C{color:var(--y)} .grade-D{color:#f97316} .grade-F{color:var(--r)}
</style>
</head>
<body>
<div class="layout">

<!-- Sidebar -->
<div class="sidebar">
  <div class="logo">Balance<span>r</span></div>

  <button class="nav-btn primary" id="btn-run" onclick="startRun()">
    ▶ Run Bias Scan
  </button>
  <button class="nav-btn" id="btn-fix" onclick="applyFix()" disabled>
    ⚡ Apply Fix
  </button>
  <button class="nav-btn" id="btn-report" onclick="openReport()" disabled>
    ↗ Full Report
  </button>
  <div class="nav-sep"></div>
  <button class="nav-btn" onclick="loadAll()" style="font-size:12px;color:var(--m)">
    ↺ Refresh
  </button>

  <div style="margin-top:auto;font-size:11px;color:var(--m);line-height:1.8">
    <div>Model: llama3.2:1b</div>
    <div>Pairs: 100 · seed=42</div>
    <div>H2S × Google</div>
  </div>
</div>

<!-- Main -->
<div class="main">
  <div class="page-title">AI Hiring Bias Detection</div>
  <div class="page-sub">100 identical resumes · only the candidate name differs · B&M 2004 name list</div>

  <!-- Status -->
  <div class="card" id="card-status">
    <div class="card-title">Status</div>
    <div class="status-row">
      <div class="dot" id="status-dot"></div>
      <span id="status-text">Loading...</span>
    </div>
    <div id="progress-wrap" class="hidden">
      <div class="progress-bar"><div class="progress-fill" id="progress-fill" style="width:0%"></div></div>
      <div class="progress-label" id="progress-label"></div>
    </div>
  </div>

  <!-- Metrics -->
  <div class="card hidden" id="card-metrics">
    <div class="card-title">Bias Metrics — Raw Scores</div>
    <div class="metrics-grid" id="metrics-grid"></div>
    <div id="verdict-box"></div>
  </div>

  <!-- Before / After -->
  <div class="card hidden" id="card-fix">
    <div class="card-title">Correction — Math Reweighing</div>
    <div class="cmp" id="cmp-grid"></div>
  </div>

  <!-- Narrative -->
  <div class="card hidden" id="card-narr">
    <div class="card-title">Audit Narrative <span id="narr-model" style="font-weight:400;text-transform:none;letter-spacing:0;color:var(--m)"></span></div>
    <div id="narr-body"></div>
  </div>
</div>

</div>

<script>
let _polling = null;

function $(id){ return document.getElementById(id); }

function setStatus(dot, text) {
  $('status-dot').className = 'dot ' + dot;
  $('status-text').textContent = text;
}

function showProgress(batch, total) {
  $('progress-wrap').classList.remove('hidden');
  const pct = total > 0 ? Math.round((batch / total) * 100) : 0;
  $('progress-fill').style.width = pct + '%';
  $('progress-label').textContent = `Batch ${batch} / ${total}  (${pct}%)`;
}

function hideProgress() {
  $('progress-wrap').classList.add('hidden');
}

function gradeClass(g) {
  return 'grade-' + g.replace(/[^A-Z]/g,'');
}

function metricCard(label, val, sub, cls='') {
  return `<div class="metric">
    <div class="metric-label">${label}</div>
    <div class="metric-val ${cls}">${val}</div>
    ${sub ? `<div class="metric-sub">${sub}</div>` : ''}
  </div>`;
}

function renderMetrics(data) {
  const b = data.biased;
  const gap = b.dpg;
  const gapCls = Math.abs(gap) > 5 ? 'r' : Math.abs(gap) > 2 ? 'y' : 'g';
  const dirCls = b.dir >= 0.80 ? 'g' : 'r';
  const pCls = b.significant ? 'r' : 'g';

  $('metrics-grid').innerHTML =
    metricCard('Grade', b.grade, 'Disparate impact', gradeClass(b.grade)) +
    metricCard('DPG', (gap > 0 ? '+' : '') + gap.toFixed(1) + ' pts', 'white − black avg', gapCls) +
    metricCard('DIR', b.dir.toFixed(3), b.dir >= 0.80 ? 'EEOC pass ✓' : 'EEOC fail ✗', dirCls) +
    metricCard('p-value', b.p_value.toFixed(4), b.significant ? 'significant ✗' : 'not significant ✓', pCls) +
    metricCard('White avg', b.white_avg, b.white_hire + '% hired', 'c') +
    metricCard('Black avg', b.black_avg, b.black_hire + '% hired', 'p');

  const isBiased = b.significant || b.dir < 0.80;
  $('verdict-box').innerHTML = isBiased
    ? `<div class="verdict red">
        <div class="verdict-title r">⚠ Bias Detected (Grade ${b.grade})</div>
        <div class="verdict-body">
          ${Math.abs(gap).toFixed(1)}-point gap on identical resumes — name is the only variable.<br>
          White hire rate <strong>${b.white_hire}%</strong> vs black hire rate <strong>${b.black_hire}%</strong>
          (${(b.white_hire - b.black_hire).toFixed(1)}% gap).<br>
          ${b.significant ? `Statistically significant at p=${b.p_value.toFixed(4)}.` : ''}
        </div>
      </div>`
    : `<div class="verdict green"><div class="verdict-title g">✓ No significant bias detected</div></div>`;

  $('card-metrics').classList.remove('hidden');
}

function renderFix(data) {
  if (!data.math) return;
  const b = data.biased, m = data.math;
  const bGap = b.dpg, mGap = m.dpg;

  function rows(d, isMath) {
    const dirTag = d.dir >= 0.80
      ? '<span class="tag pass">PASS</span>' : '<span class="tag fail">FAIL</span>';
    return `
      <div class="row"><span class="row-k">Grade</span> <span class="${gradeClass(d.grade)}">${d.grade}</span></div>
      <div class="row"><span class="row-k">DIR</span> <span class="${d.dir>=0.80?'g':'r'}">${d.dir.toFixed(3)}${dirTag}</span></div>
      <div class="row"><span class="row-k">Gap</span> <span class="${Math.abs(d.dpg)<1?'g':'r'}">${(d.dpg>0?'+':'')+d.dpg.toFixed(1)} pts</span></div>
      <div class="row"><span class="row-k">White hire</span> ${d.white_hire}%</div>
      <div class="row"><span class="row-k">Black hire</span> <span class="${isMath?'g':'p'}">${d.black_hire}%</span></div>
    `;
  }

  $('cmp-grid').innerHTML = `
    <div class="cmp-col">
      <div class="cmp-label red">Before (biased)</div>
      ${rows(b, false)}
    </div>
    <div class="cmp-col after">
      <div class="cmp-label green">After (math fix)</div>
      ${rows(m, true)}
    </div>`;

  $('card-fix').classList.remove('hidden');
}

function renderNarrative(narr) {
  if (!narr) return;
  const sev = narr.severity || '';
  $('narr-model').textContent = narr.model ? `— ${narr.model}` : '';
  $('narr-body').innerHTML = `
    ${narr.verdict ? `<div class="verdict ${['CRITICAL','HIGH'].includes(sev)?'red':'green'}" style="margin-bottom:12px">
      <div class="verdict-title">
        ${narr.verdict}
        ${sev ? `<span class="sev ${sev}">${sev}</span>` : ''}
      </div>
    </div>` : ''}
    ${narr.explanation ? `<div style="margin-bottom:12px">
      <div style="font-size:11px;font-weight:700;color:var(--m);text-transform:uppercase;letter-spacing:.6px;margin-bottom:6px">Explanation</div>
      <div style="color:var(--t);font-size:13px;line-height:1.7">${narr.explanation}</div>
    </div>` : ''}
    ${narr.root_cause ? `<div style="margin-bottom:12px">
      <div style="font-size:11px;font-weight:700;color:var(--m);text-transform:uppercase;letter-spacing:.6px;margin-bottom:6px">Root Cause</div>
      <div style="color:var(--t);font-size:13px;line-height:1.7">${narr.root_cause}</div>
    </div>` : ''}
    ${narr.fix_comparison ? `<div>
      <div style="font-size:11px;font-weight:700;color:var(--m);text-transform:uppercase;letter-spacing:.6px;margin-bottom:6px">Fix Effectiveness</div>
      <div style="color:var(--t);font-size:13px;line-height:1.7">${narr.fix_comparison}</div>
    </div>` : ''}
  `;
  $('card-narr').classList.remove('hidden');
}

async function loadAll() {
  const st = await (await fetch('/api/status')).json();

  if (st.status === 'running') {
    const pct = st.progress.total > 0 ? Math.round(st.progress.batch / st.progress.total * 100) : 0;
    setStatus('yellow', `Scoring… batch ${st.progress.batch}/${st.progress.total} (${pct}%)`);
    showProgress(st.progress.batch, st.progress.total);
    $('btn-run').disabled = true;
    $('btn-fix').disabled = true;
    $('btn-report').disabled = true;
    if (!_polling) _polling = setInterval(loadAll, 1200);
    return;
  }

  clearInterval(_polling); _polling = null;
  hideProgress();
  $('btn-run').disabled = false;

  if (st.status === 'error') {
    setStatus('red', 'Error: ' + st.error);
    return;
  }

  if (st.has_results) {
    const ts = st.meta?.timestamp || '';
    const model = st.meta?.model || '';
    const secs = st.meta?.scoring_time_seconds || '';
    setStatus('green', `Results from ${ts}  ·  ${model}  ·  ${secs}s`);
    $('btn-fix').disabled = false;
    $('btn-report').disabled = false;

    const m = await (await fetch('/api/metrics')).json();
    if (!m.error) {
      renderMetrics(m);
      if (m.math) renderFix(m);
      if (m.narrative) renderNarrative(m.narrative);
    }
  } else {
    setStatus('', 'No results yet — click Run Bias Scan to start.');
  }
}

async function startRun() {
  $('btn-run').disabled = true;
  $('card-metrics').classList.add('hidden');
  $('card-fix').classList.add('hidden');
  $('card-narr').classList.add('hidden');
  setStatus('yellow', 'Starting…');
  showProgress(0, 20);
  await fetch('/api/run', { method: 'POST' });
  _polling = setInterval(loadAll, 1200);
}

async function applyFix() {
  $('btn-fix').disabled = true;
  setStatus('yellow', 'Applying corrections…');
  await fetch('/api/fix', { method: 'POST' });
  await loadAll();
}

function openReport() {
  window.open('/report', '_blank');
}

loadAll();
</script>
</body>
</html>"""
