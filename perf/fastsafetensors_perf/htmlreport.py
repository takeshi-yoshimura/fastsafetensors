"""Self-contained HTML report generator (fixed design, baked into the package).

`render()` turns JSONL aggregate records + `--trace` time-series into one
standalone HTML file: throughput bars, tensor-parallel scaling, a resource
timeline (small multiples with a real y-axis and a synced hover cursor), a
detail table, and an auto-filled methodology block. The design lives in
:data:`TEMPLATE`; only the data changes between runs, so the output is
reproducible from the CLI (`fastsafetensors-perf html`) rather than hand-authored.

CPU-only, no torch.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from .report import metric_value

_BAR_COLORS = ["var(--series-fst)", "var(--series-vllm)", "var(--series-tp)"]
_BASE_COLOR = "var(--series-base-solid)"


def _cfg(a: Dict[str, Any]) -> Dict[str, Any]:
    ident = a["identity"]
    return {
        "mode": ident["mode"], "ws": ident["world_size"], "backend": ident["backend"],
        "cache": ident["cache_policy"], "consumer": ident["consumer"],
        "gbps": metric_value(a, "delivery_gbps"),
        "read_gbps": metric_value(a, "read_gbps"),
        "wall": metric_value(a, "wall_s"), "ttf": metric_value(a, "ttf_s"),
        "cov": a["aggregate"]["stats"].get("wall_seconds", {}).get("cov", 0.0),
        "gpu_mem": metric_value(a, "gpu_mem_gb"),
        "disk_gbps": metric_value(a, "disk_gbps"),
    }


def _fmt_gb(nbytes: float) -> str:
    return f"{nbytes / 1e9:.1f} GB"


def _assemble(aggregates: List[Dict[str, Any]], traces: List[Dict[str, Any]],
              title: str, subtitle: str) -> Dict[str, Any]:
    if not aggregates:
        raise ValueError("no aggregate records to report")
    cfgs = [_cfg(a) for a in aggregates]
    first = aggregates[0]
    env = first.get("environment", {})
    versions = env.get("versions", {})
    ckpt = first.get("case", {}).get("checkpoint", {})
    gpu = env.get("gpu", {}).get("name", "GPU")
    model_bytes = ckpt.get("source_checkpoint_bytes", 0)
    repeat = first.get("case", {}).get("repeat", 0)
    cache = cfgs[0]["cache"]
    # Chip should name the fastsafetensors backend, not a stray safetensors mmap.
    backend = next((c["backend"] for c in cfgs if c["mode"] != "safetensors"),
                   cfgs[0]["backend"])

    chips = [
        f'{gpu}', _fmt_gb(model_bytes) + " checkpoint",
        f'{cache} cache', f'{repeat} reps · medians', f'backend {backend}',
    ]

    # --- single-GPU throughput bars, speedup vs a safetensors ws1 baseline ---
    ws1 = [c for c in cfgs if c["ws"] == 1]
    base = next((c for c in ws1 if c["mode"] == "safetensors"), None)
    base_gbps = base["gbps"] if base else (min((c["gbps"] for c in ws1), default=0.0) or 1.0)
    bars_modes = []
    ci = 0
    for c in sorted(ws1, key=lambda x: (x["mode"] != "safetensors", x["mode"])):
        is_base = base is not None and c is base
        color = _BASE_COLOR if is_base else _BAR_COLORS[ci % len(_BAR_COLORS)]
        if not is_base:
            ci += 1
        bars_modes.append({
            "label": c["mode"], "sub": f'{c["backend"]} · {c["consumer"]}',
            "gbps": round(c["gbps"], 2), "color": color, "base": is_base,
            "speedup": round(c["gbps"] / base_gbps, 2) if base_gbps else 1.0,
        })

    # --- scaling bars: a mode present at multiple world sizes ----------------
    bars_scale: Optional[List[Dict[str, Any]]] = None
    by_mode: Dict[str, List[Dict[str, Any]]] = {}
    for c in cfgs:
        by_mode.setdefault(c["mode"], []).append(c)
    for mode, group in by_mode.items():
        wss = sorted({c["ws"] for c in group})
        if len(wss) > 1:
            base_c = min(group, key=lambda x: x["ws"])
            bars_scale = []
            for i, c in enumerate(sorted(group, key=lambda x: x["ws"])):
                bars_scale.append({
                    "label": f'{c["ws"]} × GPU', "sub": f'{mode} · {c["backend"]}',
                    "gbps": round(c["read_gbps"] or c["gbps"], 2),
                    "wall": round(c["wall"], 3),
                    "color": _BASE_COLOR if c is base_c else _BAR_COLORS[i % len(_BAR_COLORS)],
                    "base": c is base_c,
                    "speedup": round(base_c["wall"] / c["wall"], 2) if c["wall"] else 1.0,
                })
            break

    # --- KPIs ----------------------------------------------------------------
    kpis = []
    fst_ws1 = [c for c in ws1 if c["mode"] != "safetensors"]
    if fst_ws1:
        best = max(fst_ws1, key=lambda x: x["gbps"])
        kpis.append({"n": f'{best["gbps"]:.1f}', "u": "GB/s",
                     "k": "fastsafetensors, single GPU",
                     "sub": (f'vs {base["gbps"]:.1f} GB/s safetensors' if base else "")})
        if base and base["gbps"]:
            kpis.append({"n": f'{best["gbps"] / base["gbps"]:.2f}', "u": "×",
                         "k": "faster than safetensors",
                         "sub": "same GPU · files · cache"})
    if bars_scale and len(bars_scale) > 1:
        top = bars_scale[-1]
        kpis.append({"n": f'{top["speedup"]:.1f}', "u": "×",
                     "k": f'faster at {top["label"]}', "sub": "tensor-parallel load"})

    # --- detail table --------------------------------------------------------
    detail = [{
        "name": c["mode"], "cfg": f'{c["consumer"]} · {c["cache"]}',
        "gpus": c["ws"], "backend": c["backend"], "gbps": round(c["gbps"], 2),
        "wall": round(c["wall"], 3), "ttf": round(c["ttf"], 3), "cov": c["cov"],
    } for c in sorted(cfgs, key=lambda x: (x["ws"], x["mode"]))]

    # --- traces (timeline) ---------------------------------------------------
    tr = []
    for i, t in enumerate(traces):
        ident = t.get("identity", {})
        tr.append({
            "key": f"t{i}",
            "label": f'{ident.get("mode", "?")} · {ident.get("world_size", "?")}× GPU',
            "wall": t.get("wall_seconds_median", 0.0),
            "ranks": [r["samples"] for r in t.get("ranks", [])],
        })

    # --- auto notes / methodology -------------------------------------------
    notes = []
    if any(c["backend"] == "nogds" for c in cfgs):
        notes.append("fastsafetensors ran on the <code>nogds</code> path "
                     "(GDS unavailable on this host); a GDS-capable NVMe would differ.")
    if cfgs and all(c["disk_gbps"] == 0 for c in cfgs) and any(c["read_gbps"] > 0 for c in cfgs):
        notes.append("Block-layer disk I/O read 0 (tmpfs or warm cache); "
                     "<code>read GB/s</code> is bytes via <code>read()</code> syscalls.")
    notes.append("Loader throughput, not a full <code>model.load_weights()</code>.")
    notes.append("NVML GPU utilization updates on a ~100&nbsp;ms window, so that "
                 "line is stepped even at 20&nbsp;ms sampling.")

    method = (
        f'torch {versions.get("torch", "?")} · CUDA {versions.get("cuda", "?")} · '
        f'vLLM {versions.get("vllm", "?")} · fastsafetensors {versions.get("fastsafetensors", "?")} · '
        f'{gpu} · git {env.get("git_sha", "")[:9]}'
    )

    return {
        "title": title, "subtitle": subtitle, "chips": chips, "kpis": kpis,
        "bars_modes": bars_modes, "bars_scale": bars_scale, "detail": detail,
        "traces": tr, "notes": notes, "method": method,
    }


def render(aggregates: List[Dict[str, Any]], traces: List[Dict[str, Any]],
           *, title: str = "fastsafetensors weight-load benchmark",
           subtitle: str = "") -> str:
    data = _assemble(aggregates, traces, title, subtitle)
    return TEMPLATE.replace("__DATA_BLOB__", json.dumps(data, separators=(",", ":")))


# The fixed design. __DATA_BLOB__ is replaced with the assembled JSON.
TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>fastsafetensors weight-load benchmark</title>
<style>
  :root {
    color-scheme: light;
    --bg:#eef1f5; --surface-1:#fbfcfd; --surface-2:#f2f4f8; --border:#d8dee6;
    --border-strong:#c3ccd6; --text-primary:#0e141b; --text-secondary:#4a5763;
    --text-muted:#7c8894; --series-fst:#2a78d6; --series-vllm:#1baf7a;
    --series-base-solid:#64748b; --series-tp:#4a3aa7; --track:#e7ebf1;
    --grid:#dbe1e9; --accent-ink:#1f5fb0; --good:#1a8f5a;
    --shadow:0 1px 2px rgba(14,20,27,.06), 0 6px 20px rgba(14,20,27,.05);
  }
  @media (prefers-color-scheme: dark) {
    :root:where(:not([data-theme="light"])) {
      color-scheme: dark;
      --bg:#0d1116; --surface-1:#161b22; --surface-2:#1b2029; --border:#262d38;
      --border-strong:#38404d; --text-primary:#f2f5f8; --text-secondary:#b3bdc9;
      --text-muted:#7d8896; --series-fst:#3987e5; --series-vllm:#199e70;
      --series-base-solid:#8b98a8; --series-tp:#9085e9; --track:#222834;
      --grid:#2a323e; --accent-ink:#7fb2f4; --good:#35b979;
      --shadow:0 1px 2px rgba(0,0,0,.4), 0 8px 26px rgba(0,0,0,.35);
    }
  }
  :root[data-theme="dark"] {
    color-scheme: dark;
    --bg:#0d1116; --surface-1:#161b22; --surface-2:#1b2029; --border:#262d38;
    --border-strong:#38404d; --text-primary:#f2f5f8; --text-secondary:#b3bdc9;
    --text-muted:#7d8896; --series-fst:#3987e5; --series-vllm:#199e70;
    --series-base-solid:#8b98a8; --series-tp:#9085e9; --track:#222834;
    --grid:#2a323e; --accent-ink:#7fb2f4; --good:#35b979;
    --shadow:0 1px 2px rgba(0,0,0,.4), 0 8px 26px rgba(0,0,0,.35);
  }
  * { box-sizing:border-box; }
  html,body { margin:0; padding:0; }
  body { background:var(--bg); color:var(--text-primary); font-family:system-ui,-apple-system,"Segoe UI",Roboto,sans-serif; line-height:1.5; -webkit-font-smoothing:antialiased; }
  .mono { font-family:ui-monospace,"SF Mono","Cascadia Code",Menlo,Consolas,monospace; font-variant-numeric:tabular-nums; }
  .wrap { max-width:920px; margin:0 auto; padding:40px 24px 64px; }
  .themebtn { position:fixed; top:16px; right:16px; z-index:30; width:38px; height:38px; border-radius:10px; border:1px solid var(--border); background:var(--surface-1); color:var(--text-secondary); cursor:pointer; font-size:17px; box-shadow:var(--shadow); }
  .themebtn:focus-visible { outline:2px solid var(--series-fst); outline-offset:2px; }
  header { margin-bottom:26px; }
  .eyebrow { font-size:12px; letter-spacing:.12em; text-transform:uppercase; color:var(--accent-ink); font-weight:700; margin:0 0 10px; }
  h1 { font-size:clamp(25px,4.4vw,36px); line-height:1.12; margin:0 0 12px; letter-spacing:-.02em; text-wrap:balance; font-weight:800; }
  h1 .light { color:var(--text-muted); font-weight:700; }
  .lede { color:var(--text-secondary); font-size:16px; max-width:64ch; margin:0 0 18px; }
  .chips { display:flex; flex-wrap:wrap; gap:8px; }
  .chip { display:inline-flex; align-items:center; gap:6px; background:var(--surface-2); border:1px solid var(--border); border-radius:999px; padding:4px 11px; font-size:12.5px; color:var(--text-secondary); }
  .kpis { display:grid; grid-template-columns:repeat(auto-fit,minmax(180px,1fr)); gap:14px; margin:24px 0 28px; }
  .kpi { background:var(--surface-1); border:1px solid var(--border); border-radius:14px; padding:18px; box-shadow:var(--shadow); }
  .kpi .n { font-size:clamp(28px,5vw,36px); font-weight:800; letter-spacing:-.02em; line-height:1; }
  .kpi .n small { font-size:.5em; font-weight:700; color:var(--text-muted); margin-left:2px; }
  .kpi .k { font-size:13px; color:var(--text-secondary); margin-top:8px; }
  .kpi .sub { font-size:12px; color:var(--text-muted); margin-top:3px; }
  .card { background:var(--surface-1); border:1px solid var(--border); border-radius:16px; padding:22px 22px 18px; box-shadow:var(--shadow); margin-bottom:20px; }
  .card h2 { font-size:17px; margin:0 0 3px; letter-spacing:-.01em; }
  .card .note { font-size:13px; color:var(--text-muted); margin:0 0 18px; }
  .bars { display:flex; flex-direction:column; gap:15px; }
  .bar-row { display:grid; grid-template-columns:190px 1fr; gap:14px; align-items:center; }
  .bar-label { font-size:13.5px; }
  .bar-label .swatch { display:inline-block; width:9px; height:9px; border-radius:2px; margin-right:7px; vertical-align:middle; }
  .bar-label .cfg { color:var(--text-muted); font-size:12px; }
  .bar-track { position:relative; height:30px; background:var(--track); border-radius:4px 7px 7px 4px; overflow:hidden; }
  .bar-fill { position:absolute; left:0; top:0; bottom:0; border-radius:4px 6px 6px 4px; display:flex; align-items:center; justify-content:flex-end; transition:width .8s cubic-bezier(.22,.61,.36,1); }
  .bar-val { font-size:13px; font-weight:700; color:#fff; padding-right:9px; white-space:nowrap; }
  .badge { display:inline-block; margin-left:8px; font-size:11.5px; font-weight:700; color:var(--good); background:color-mix(in srgb,var(--good) 14%,transparent); border-radius:999px; padding:1px 7px; vertical-align:middle; }
  .badge.base { color:var(--text-muted); background:var(--surface-2); }
  /* timeline */
  .runtoggle { display:inline-flex; border:1px solid var(--border); border-radius:9px; overflow:hidden; margin-bottom:6px; }
  .runtoggle button { border:0; background:var(--surface-2); color:var(--text-secondary); font:inherit; font-size:13px; font-weight:600; padding:6px 14px; cursor:pointer; }
  .runtoggle button[aria-pressed="true"] { background:var(--series-fst); color:#fff; }
  .runtoggle button:focus-visible { outline:2px solid var(--series-fst); outline-offset:-2px; }
  .rank-legend { display:flex; gap:16px; font-size:12px; color:var(--text-secondary); margin:2px 0 14px; min-height:14px; }
  .rank-legend .swatch { display:inline-block; width:16px; height:3px; border-radius:2px; margin-right:6px; vertical-align:middle; }
  .charts { display:flex; flex-direction:column; gap:16px; }
  .tchart .thead { display:flex; justify-content:space-between; align-items:baseline; margin-bottom:4px; }
  .tchart .ttitle { font-size:13px; font-weight:600; }
  .tchart .ttitle .u { color:var(--text-muted); font-weight:400; font-size:11.5px; margin-left:6px; }
  .tchart .tval { font-size:12.5px; color:var(--text-secondary); }
  .tbody { display:grid; grid-template-columns:52px 1fr; column-gap:8px; }
  .yaxis { display:flex; flex-direction:column; justify-content:space-between; align-items:flex-end; height:74px; font-size:10.5px; color:var(--text-muted); padding:1px 0; }
  .plot { min-width:0; }
  .plot svg { display:block; width:100%; height:74px; background:var(--surface-2); border-radius:8px; border:1px solid var(--border); }
  .tfoot { display:flex; justify-content:space-between; font-size:10.5px; color:var(--text-muted); margin-top:2px; }
  polyline, line.cur, .gridline { vector-effect:non-scaling-stroke; }
  .gridline { stroke:var(--grid); stroke-width:1; }
  line.cur { stroke:var(--text-secondary); stroke-width:1; stroke-dasharray:3 3; opacity:0; }
  table.data { width:100%; border-collapse:collapse; font-size:13px; }
  .table-wrap { overflow-x:auto; }
  table.data th, table.data td { text-align:right; padding:8px 10px; border-bottom:1px solid var(--border); white-space:nowrap; }
  table.data th:first-child, table.data td:first-child { text-align:left; }
  table.data thead th { color:var(--text-secondary); font-weight:600; font-size:12px; border-bottom:1px solid var(--border-strong); }
  table.data tbody tr:last-child td { border-bottom:none; }
  details.method { margin-top:22px; }
  details.method summary { cursor:pointer; font-weight:600; font-size:14px; color:var(--text-secondary); padding:6px 0; list-style:none; }
  details.method summary::before { content:"▸"; display:inline-block; margin-right:8px; transition:transform .15s; color:var(--text-muted); }
  details.method[open] summary::before { transform:rotate(90deg); }
  details.method ul { margin:8px 0 0; padding-left:20px; color:var(--text-secondary); font-size:13.5px; }
  details.method li { margin:6px 0; }
  footer { margin-top:26px; color:var(--text-muted); font-size:12.5px; }
  code { font-family:ui-monospace,Menlo,monospace; background:var(--surface-2); padding:1px 5px; border-radius:4px; font-size:.9em; }
  @media (max-width:600px) { .bar-row { grid-template-columns:1fr; gap:6px; } }
  @media (prefers-reduced-motion:reduce) { .bar-fill { transition:none; } }
</style>
</head>
<body>
<button class="themebtn" id="themebtn" aria-label="Toggle light/dark theme">◐</button>
<div class="wrap">
  <header>
    <p class="eyebrow">fastsafetensors-perf report</p>
    <h1 id="title"></h1>
    <p class="lede" id="subtitle"></p>
    <div class="chips" id="chips"></div>
  </header>
  <section class="kpis" id="kpis"></section>
  <div class="card" id="card-modes">
    <h2>Single-GPU load throughput</h2>
    <p class="note">Higher is better. Speedup is vs the safetensors baseline on identical files, cache, and consumer.</p>
    <div class="bars" id="chart-modes"></div>
  </div>
  <div class="card" id="card-scale" hidden>
    <h2>Tensor-parallel scaling</h2>
    <p class="note">Effective load rate = checkpoint bytes ÷ wall time; speedup is wall-time reduction vs one GPU.</p>
    <div class="bars" id="chart-scale"></div>
  </div>
  <div class="card" id="card-timeline" hidden>
    <h2>Resource timeline</h2>
    <p class="note">Every metric sampled across one representative load. Hover to scrub — the cursor and per-chart readouts follow the same instant. Each rank is a separate line.</p>
    <div class="runtoggle" id="runtoggle" role="group" aria-label="Select run"></div>
    <div class="rank-legend" id="rank-legend"></div>
    <div class="charts" id="charts"></div>
  </div>
  <div class="card">
    <h2>All measurements</h2>
    <p class="note">Medians across repetitions. CoV is the run-to-run coefficient of variation on wall time.</p>
    <div class="table-wrap"><table class="data" id="data-table"><thead><tr>
      <th>Configuration</th><th>GPUs</th><th>backend</th><th>load GB/s</th><th>wall s</th><th>ttf s</th><th>CoV</th>
    </tr></thead><tbody></tbody></table></div>
  </div>
  <details class="method">
    <summary>Methodology &amp; notes</summary>
    <ul id="notes"></ul>
    <p class="mono" id="method" style="font-size:12px;color:var(--text-muted);margin-top:10px"></p>
  </details>
  <footer>Generated by <code>fastsafetensors-perf html</code> from JSONL results and <code>--trace</code> time-series — the same records the regression gate consumes.</footer>
</div>
<script>
  const DATA = __DATA_BLOB__;
  const RANK_COLORS = ["var(--series-fst)", "var(--series-tp)"];
  const METRICS = [
    { key:"cpu_user_pct", label:"CPU user", unit:"% of 1 core", fmt:v=>v.toFixed(0) },
    { key:"cpu_system_pct", label:"CPU system", unit:"% of 1 core", fmt:v=>v.toFixed(0) },
    { key:"read_gbps", label:"read throughput", unit:"GB/s (read syscalls)", fmt:v=>v.toFixed(1) },
    { key:"gpu_util_pct", label:"GPU utilization", unit:"% (NVML)", fmt:v=>v.toFixed(0) },
    { key:"gpu_mem_gb", label:"GPU memory used", unit:"GB", fmt:v=>v.toFixed(1) },
    { key:"host_rss_gb", label:"host memory (RSS)", unit:"GB", fmt:v=>v.toFixed(1) },
    { key:"nvlink_gbps", label:"NVLink", unit:"GB/s", fmt:v=>v.toFixed(2) },
  ];
  const $ = id => document.getElementById(id);
  $("title").innerHTML = DATA.title;
  $("subtitle").textContent = DATA.subtitle || "";
  if (!DATA.subtitle) $("subtitle").style.display = "none";
  $("chips").innerHTML = DATA.chips.map(c => `<span class="chip">${c}</span>`).join("");
  $("kpis").innerHTML = DATA.kpis.map(k => `
    <div class="kpi"><div class="n mono">${k.n}<small>${k.u}</small></div>
      <div class="k">${k.k}</div><div class="sub">${k.sub||""}</div></div>`).join("");

  function renderBars(elId, rows) {
    if (!rows) return;
    const max = Math.max(...rows.map(r => r.gbps));
    $(elId).innerHTML = rows.map(r => {
      const pct = Math.max(6, (r.gbps / max) * 100);
      const badge = r.base ? `<span class="badge base">baseline</span>`
                           : `<span class="badge">${r.speedup.toFixed(2)}× faster</span>`;
      return `<div class="bar-row"><div class="bar-label"><span class="swatch" style="background:${r.color}"></span>
          <b>${r.label}</b>${badge}<br><span class="cfg mono">${r.sub}</span></div>
        <div class="bar-track"><div class="bar-fill" style="background:${r.color};width:${pct}%">
          <span class="bar-val mono">${r.gbps.toFixed(2)} GB/s</span></div></div></div>`;
    }).join("");
  }
  renderBars("chart-modes", DATA.bars_modes);
  if (DATA.bars_scale) { renderBars("chart-scale", DATA.bars_scale); $("card-scale").hidden = false; }

  document.querySelector("#data-table tbody").innerHTML = DATA.detail.map(r => `
    <tr><td>${r.name} <span class="mono" style="color:var(--text-muted)">${r.cfg}</span></td>
      <td class="mono">${r.gpus}</td><td class="mono">${r.backend}</td>
      <td class="mono">${r.gbps.toFixed(2)}</td><td class="mono">${r.wall.toFixed(3)}</td>
      <td class="mono">${r.ttf.toFixed(3)}</td><td class="mono">${(r.cov*100).toFixed(1)}%</td></tr>`).join("");

  $("notes").innerHTML = DATA.notes.map(n => `<li>${n}</li>`).join("");
  $("method").textContent = DATA.method;

  // ---- timeline small multiples with a real y-axis + synced cursor ----
  const H = 74, PADT = 7, PADB = 6, VBW = 1000;
  let curRun = 0;
  function niceMax(v) {
    if (v <= 0) return 1;
    const p = Math.pow(10, Math.floor(Math.log10(v))), n = v / p;
    return (n <= 1 ? 1 : n <= 2 ? 2 : n <= 5 ? 5 : 10) * p;
  }
  function buildToggle() {
    if (!DATA.traces.length) return;
    $("card-timeline").hidden = false;
    $("runtoggle").innerHTML = DATA.traces.map((t, i) =>
      `<button data-i="${i}" aria-pressed="${i === curRun}">${t.label}</button>`).join("");
  }
  function buildCharts() {
    const run = DATA.traces[curRun];
    if (!run) return;
    const ranks = run.ranks;
    const tmax = Math.max(...ranks.map(s => s[s.length-1].t));
    $("rank-legend").innerHTML = ranks.length > 1
      ? ranks.map((_, i) => `<span><span class="swatch" style="background:${RANK_COLORS[i]}"></span>rank ${i}</span>`).join("")
      : "";
    const plotH = H - PADT - PADB;
    $("charts").innerHTML = METRICS.map((m, mi) => {
      const vmax = niceMax(Math.max(1e-9, ...ranks.flatMap(s => s.map(p => p[m.key]))));
      const gy = PADT + plotH/2;
      const polys = ranks.map((samples, ri) => {
        const pts = samples.map(p =>
          `${(p.t/tmax*VBW).toFixed(1)},${(PADT + plotH - (p[m.key]/vmax)*plotH).toFixed(1)}`).join(" ");
        const color = RANK_COLORS[ri];
        const area = ranks.length === 1
          ? `<polygon points="0,${PADT+plotH} ${pts} ${VBW},${PADT+plotH}" fill="${color}" opacity="0.10"/>` : "";
        return `${area}<polyline points="${pts}" fill="none" stroke="${color}" stroke-width="1.8"/>`;
      }).join("");
      return `<div class="tchart" data-mi="${mi}">
        <div class="thead"><div class="ttitle">${m.label}<span class="u">${m.unit}</span></div>
          <div class="tval mono" data-val></div></div>
        <div class="tbody">
          <div class="yaxis"><span>${m.fmt(vmax)}</span><span>0</span></div>
          <div class="plot"><svg viewBox="0 0 ${VBW} ${H}" preserveAspectRatio="none" aria-hidden="true">
            <line class="gridline" x1="0" y1="${gy}" x2="${VBW}" y2="${gy}"/>
            <line class="gridline" x1="0" y1="${PADT}" x2="${VBW}" y2="${PADT}"/>
            ${polys}<line class="cur" x1="0" y1="0" x2="0" y2="${H}"/></svg>
            <div class="tfoot"><span>0 s</span><span>${tmax.toFixed(2)} s</span></div>
          </div>
        </div></div>`;
    }).join("");
    $("charts")._tmax = tmax; $("charts")._ranks = ranks;
    scrub(null);
  }
  function nearest(samples, t) {
    let lo = 0, hi = samples.length - 1;
    while (lo < hi) { const m = (lo+hi)>>1; if (samples[m].t < t) lo = m+1; else hi = m; }
    if (lo > 0 && Math.abs(samples[lo-1].t - t) < Math.abs(samples[lo].t - t)) return samples[lo-1];
    return samples[lo];
  }
  function scrub(frac) {
    const charts = $("charts"), ranks = charts._ranks, tmax = charts._tmax;
    if (!ranks) return;
    const t = frac == null ? null : frac * tmax;
    charts.querySelectorAll(".tchart").forEach(tc => {
      const m = METRICS[+tc.dataset.mi];
      const cur = tc.querySelector(".cur"), val = tc.querySelector("[data-val]");
      if (t == null) {
        cur.style.opacity = "0";
        const peak = Math.max(...ranks.flatMap(s => s.map(p => p[m.key])));
        val.textContent = "peak " + m.fmt(peak);
        return;
      }
      cur.setAttribute("x1", frac*VBW); cur.setAttribute("x2", frac*VBW); cur.style.opacity = "1";
      const parts = ranks.map((s, ri) => {
        const p = nearest(s, t);
        return ranks.length > 1 ? `r${ri} ${m.fmt(p[m.key])}` : m.fmt(p[m.key]);
      });
      val.textContent = `${t.toFixed(2)}s · ${parts.join("  ")}`;
    });
  }
  const chartsEl = $("charts");
  chartsEl.addEventListener("pointermove", e => {
    const svg = chartsEl.querySelector("svg"); if (!svg) return;
    const r = svg.getBoundingClientRect();
    scrub(Math.max(0, Math.min(1, (e.clientX - r.left) / r.width)));
  });
  chartsEl.addEventListener("pointerleave", () => scrub(null));
  $("runtoggle").addEventListener("click", e => {
    const b = e.target.closest("button"); if (!b) return;
    curRun = +b.dataset.i;
    document.querySelectorAll("#runtoggle button").forEach(x =>
      x.setAttribute("aria-pressed", String(+x.dataset.i === curRun)));
    buildCharts();
  });
  buildToggle();
  buildCharts();

  $("themebtn").addEventListener("click", () => {
    const cur = document.documentElement.getAttribute("data-theme");
    const dark = cur ? cur === "dark" : matchMedia("(prefers-color-scheme: dark)").matches;
    document.documentElement.setAttribute("data-theme", dark ? "light" : "dark");
  });
</script>
</body>
</html>
"""
