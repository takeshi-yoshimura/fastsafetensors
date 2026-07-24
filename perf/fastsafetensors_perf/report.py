"""Cross-configuration performance reporting (non-gating).

This is the counterpart to :mod:`fastsafetensors_perf.compare`. Where ``compare``
*refuses* to compare across different identities (that is the regression gate),
``report`` deliberately lines results up across an axis -- mode, world size,
queue size, cache policy, model -- to answer "how much faster is X than Y?".

It reads the same JSONL aggregate records, so any data collected for regression
testing is reusable here for a talk/benchmark comparison. CPU-only, no torch.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from .results import RECORD_KIND_AGGREGATE, iter_records

# Metrics a row can be ranked on. "higher is better" flips the speedup ratio so
# a speedup > 1 always means "the candidate is better".
HIGHER_IS_BETTER = {"delivery_gbps", "storage_gbps"}

_METRIC_LABELS = {
    "delivery_gbps": "delivery GB/s",
    "storage_gbps": "storage GB/s",
    "wall_s": "wall s",
    "ttf_s": "time-to-first s",
    "peak_cuda_gb": "peak CUDA GB",
    "peak_rss_gb": "peak RSS GB",
}


def _stats(agg: Dict[str, Any]) -> Dict[str, Any]:
    return agg.get("aggregate", {}).get("stats", {})


def _median(agg: Dict[str, Any], metric: str) -> float:
    m = _stats(agg).get(metric)
    if isinstance(m, dict):
        return float(m.get("median", 0.0))
    return 0.0


def metric_value(agg: Dict[str, Any], name: str) -> float:
    """Extract a scalar metric from an aggregate record."""
    stats = _stats(agg)
    if name == "delivery_gbps":
        return float(stats.get("delivery_throughput_bps", 0.0)) / 1e9
    if name == "storage_gbps":
        return float(stats.get("storage_throughput_bps", 0.0)) / 1e9
    if name == "wall_s":
        return _median(agg, "wall_seconds")
    if name == "ttf_s":
        return _median(agg, "time_to_first_seconds")
    if name == "peak_cuda_gb":
        return _median(agg, "peak_cuda_allocated_bytes") / 1e9
    if name == "peak_rss_gb":
        return _median(agg, "host_peak_rss_bytes") / 1e9
    raise ValueError(f"unknown metric: {name}")


@dataclass
class ReportRow:
    identity: Dict[str, Any]
    group: Dict[str, Any]  # the group_by field -> value subset
    metric_name: str
    value: float
    delivery_gbps: float
    wall_s: float
    ttf_s: float
    peak_cuda_gb: float
    cov: float
    n_reps: int
    status: str
    speedup: Optional[float] = None  # vs the chosen baseline, if any

    def label(self) -> str:
        return " ".join(f"{k}={v}" for k, v in self.group.items())


def load_aggregates(paths: Sequence[str]) -> List[Dict[str, Any]]:
    """Read every aggregate record from one or more JSONL files."""
    out: List[Dict[str, Any]] = []
    for p in paths:
        for rec in iter_records(p):
            if rec.get("kind") == RECORD_KIND_AGGREGATE:
                out.append(rec)
    return out


def _row(agg: Dict[str, Any], group_by: Sequence[str], metric: str) -> ReportRow:
    ident = agg.get("identity", {})
    agg_block = agg.get("aggregate", {})
    return ReportRow(
        identity=ident,
        group={f: ident.get(f) for f in group_by},
        metric_name=metric,
        value=metric_value(agg, metric),
        delivery_gbps=metric_value(agg, "delivery_gbps"),
        wall_s=metric_value(agg, "wall_s"),
        ttf_s=metric_value(agg, "ttf_s"),
        peak_cuda_gb=metric_value(agg, "peak_cuda_gb"),
        cov=_stats(agg).get("wall_seconds", {}).get("cov", 0.0),
        n_reps=int(agg_block.get("n_repetitions", 0)),
        status=agg_block.get("worst_status", "ok"),
    )


def build_report(aggregates: Sequence[Dict[str, Any]], group_by: Sequence[str],
                 metric: str = "delivery_gbps",
                 baseline_field: Optional[str] = None,
                 baseline_value: Optional[str] = None) -> List[ReportRow]:
    """Build comparison rows, optionally with speedup vs a baseline.

    ``group_by`` names the identity fields that define a row (e.g. ``["mode"]``
    or ``["world_size"]``). When ``baseline_field``/``baseline_value`` are given
    (e.g. ``mode``/``safetensors``), each row's ``speedup`` is computed against
    the row that shares all *other* group fields but has the baseline value.
    """
    rows = [_row(agg, group_by, metric) for agg in aggregates]

    if baseline_field and baseline_value is not None:
        def rest_key(r: ReportRow):
            return tuple((f, r.identity.get(f)) for f in group_by if f != baseline_field)

        baselines: Dict[Any, ReportRow] = {}
        for r in rows:
            if str(r.identity.get(baseline_field)) == str(baseline_value):
                baselines[rest_key(r)] = r
        for r in rows:
            base = baselines.get(rest_key(r))
            if base and base.value:
                if metric in HIGHER_IS_BETTER:
                    r.speedup = r.value / base.value
                else:
                    r.speedup = base.value / r.value

    rows.sort(key=lambda r: (tuple(str(v) for v in r.group.values())))
    return rows


def format_table(rows: Sequence[ReportRow]) -> str:
    """Render a compact fixed-width comparison table."""
    if not rows:
        return "(no aggregate records)"
    metric = rows[0].metric_name
    metric_label = _METRIC_LABELS.get(metric, metric)
    has_speedup = any(r.speedup is not None for r in rows)

    header = ["config", metric_label, "wall s", "GB/s", "ttf s", "CoV", "reps", "status"]
    if has_speedup:
        header.insert(2, "speedup")

    table: List[List[str]] = [header]
    for r in rows:
        row = [
            r.label(),
            f"{r.value:.3f}",
            f"{r.wall_s:.3f}",
            f"{r.delivery_gbps:.2f}",
            f"{r.ttf_s:.3f}",
            f"{r.cov:.1%}",
            str(r.n_reps),
            r.status,
        ]
        if has_speedup:
            row.insert(2, f"{r.speedup:.2f}x" if r.speedup is not None else "-")
        table.append(row)

    widths = [max(len(r[i]) for r in table) for i in range(len(header))]
    lines = []
    for i, r in enumerate(table):
        lines.append("  ".join(cell.ljust(widths[j]) for j, cell in enumerate(r)))
        if i == 0:
            lines.append("  ".join("-" * widths[j] for j in range(len(header))))
    return "\n".join(lines)


def to_chart_data(rows: Sequence[ReportRow]) -> Dict[str, Any]:
    """Machine-readable series for plotting (labels + parallel metric arrays)."""
    return {
        "metric": rows[0].metric_name if rows else "",
        "labels": [r.label() for r in rows],
        "value": [r.value for r in rows],
        "delivery_gbps": [r.delivery_gbps for r in rows],
        "wall_s": [r.wall_s for r in rows],
        "ttf_s": [r.ttf_s for r in rows],
        "peak_cuda_gb": [r.peak_cuda_gb for r in rows],
        "cov": [r.cov for r in rows],
        "speedup": [r.speedup for r in rows],
    }
