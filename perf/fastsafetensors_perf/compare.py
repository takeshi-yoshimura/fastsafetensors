"""Regression comparison of candidate vs baseline aggregate records.

CPU-only. Pairs aggregate records by baseline identity, applies the regression
policy, and produces per-case outcomes plus an overall process exit code.

Only records whose identity matches exactly are compared. Records that differ in
identity are *incompatible* and are refused by default (they represent different
baseline series); ``allow_incompatible`` downgrades that to an informational,
non-gating note.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Any, Dict, List, Optional

from .results import (
    IDENTITY_FIELDS,
    RECORD_KIND_AGGREGATE,
    SCHEMA_VERSION,
    identity_key,
)


class ExitCode(IntEnum):
    OK = 0
    REGRESSION = 1
    INCOMPATIBLE = 2
    UNSTABLE = 3


class Outcome(IntEnum):
    """Ordered by severity so ``max`` picks the worst."""

    OK = 0
    WARN = 1
    UNSTABLE = 2
    FAIL = 3


@dataclass
class Thresholds:
    """Regression thresholds. Fractions are relative increases (candidate/base)."""

    time_fail: float = 0.15  # median wall regression >= 15% -> fail
    time_warn: float = 0.10  # >= 10% -> warn
    ttf_warn: float = 0.20  # time-to-first regression >= 20% -> warn
    mem_frac_warn: float = 0.10  # peak memory increase >= 10% ...
    mem_abs_warn_bytes: int = 512 * 1024 * 1024  # ... or >= 512 MiB -> warn
    mem_fail: bool = False  # if True, a memory-warn escalates to fail
    cov_unstable: float = 0.10  # CoV >= 10% -> mark unstable

    def to_dict(self) -> Dict[str, Any]:
        return {
            "time_fail": self.time_fail,
            "time_warn": self.time_warn,
            "ttf_warn": self.ttf_warn,
            "mem_frac_warn": self.mem_frac_warn,
            "mem_abs_warn_bytes": self.mem_abs_warn_bytes,
            "mem_fail": self.mem_fail,
            "cov_unstable": self.cov_unstable,
        }


@dataclass
class CaseResult:
    identity: Dict[str, Any]
    outcome: Outcome
    messages: List[str]
    time_delta: Optional[float] = None  # relative wall-time change
    ttf_delta: Optional[float] = None
    mem_delta: Optional[float] = None
    baseline_cov: Optional[float] = None
    candidate_cov: Optional[float] = None

    def label(self) -> str:
        ident = self.identity
        return (
            f"{ident.get('model_alias', '?')}"
            f"/{ident.get('mode', '?')}"
            f"/{ident.get('consumer', '?')}"
            f"/ws{ident.get('world_size', '?')}"
            f"/q{ident.get('queue_size', '?')}"
            f"/{ident.get('backend', '?')}"
            f"/{ident.get('cache_policy', '?')}"
        )


def _rel_delta(candidate: float, baseline: float) -> Optional[float]:
    if baseline == 0:
        return None
    return (candidate - baseline) / baseline


def _index_aggregates(records: List[Dict[str, Any]]) -> Dict[tuple, Dict[str, Any]]:
    out: Dict[tuple, Dict[str, Any]] = {}
    for rec in records:
        if rec.get("kind") != RECORD_KIND_AGGREGATE:
            continue
        out[identity_key(rec)] = rec
    return out


def _stat(rec: Dict[str, Any], metric: str, key: str = "median") -> Optional[float]:
    stats = rec.get("aggregate", {}).get("stats", {})
    m = stats.get(metric)
    if not isinstance(m, dict):
        return None
    return m.get(key)


def compare_case(baseline: Dict[str, Any], candidate: Dict[str, Any],
                 thresholds: Thresholds) -> CaseResult:
    """Compare one matched (baseline, candidate) aggregate pair."""
    messages: List[str] = []
    outcome = Outcome.OK

    # --- stability first: unstable results cannot drive a hard decision -----
    base_cov = _stat(baseline, "wall_seconds", "cov") or 0.0
    cand_cov = _stat(candidate, "wall_seconds", "cov") or 0.0
    unstable = base_cov >= thresholds.cov_unstable or cand_cov >= thresholds.cov_unstable

    # --- correctness / completion ------------------------------------------
    base_status = baseline.get("aggregate", {}).get("worst_status", "ok")
    cand_status = candidate.get("aggregate", {}).get("worst_status", "ok")
    if cand_status not in ("ok",):
        messages.append(f"candidate did not complete cleanly (status={cand_status})")
        return CaseResult(
            identity=candidate.get("identity", {}),
            outcome=Outcome.FAIL,
            messages=messages,
            baseline_cov=base_cov,
            candidate_cov=cand_cov,
        )

    # --- wall time ----------------------------------------------------------
    base_wall = _stat(baseline, "wall_seconds")
    cand_wall = _stat(candidate, "wall_seconds")
    time_delta = _rel_delta(cand_wall, base_wall) if (base_wall and cand_wall) else None
    if time_delta is not None:
        if time_delta >= thresholds.time_fail:
            outcome = max(outcome, Outcome.FAIL)
            messages.append(f"wall time regression {time_delta:+.1%} (>= {thresholds.time_fail:.0%})")
        elif time_delta >= thresholds.time_warn:
            outcome = max(outcome, Outcome.WARN)
            messages.append(f"wall time regression {time_delta:+.1%} (>= {thresholds.time_warn:.0%})")

    # --- time to first tensor ----------------------------------------------
    base_ttf = _stat(baseline, "time_to_first_seconds")
    cand_ttf = _stat(candidate, "time_to_first_seconds")
    ttf_delta = _rel_delta(cand_ttf, base_ttf) if (base_ttf and cand_ttf) else None
    if ttf_delta is not None and ttf_delta >= thresholds.ttf_warn:
        outcome = max(outcome, Outcome.WARN)
        messages.append(f"time-to-first regression {ttf_delta:+.1%} (>= {thresholds.ttf_warn:.0%})")

    # --- peak memory (CUDA allocated and host RSS) -------------------------
    mem_delta = None
    for metric, human in (
        ("peak_cuda_allocated_bytes", "CUDA allocated"),
        ("host_peak_rss_bytes", "host RSS"),
    ):
        base_mem = _stat(baseline, metric)
        cand_mem = _stat(candidate, metric)
        if not base_mem or not cand_mem:
            continue
        d = _rel_delta(cand_mem, base_mem)
        abs_inc = cand_mem - base_mem
        if d is None:
            continue
        if d >= thresholds.mem_frac_warn and abs_inc >= thresholds.mem_abs_warn_bytes:
            level = Outcome.FAIL if thresholds.mem_fail else Outcome.WARN
            outcome = max(outcome, level)
            messages.append(
                f"{human} peak +{d:.1%} (+{abs_inc / (1024 * 1024):.0f} MiB)"
            )
            if metric == "peak_cuda_allocated_bytes":
                mem_delta = d

    if unstable:
        # Unstable measurements downgrade a hard FAIL to a non-gating UNSTABLE:
        # we refuse to call a regression when the noise floor is too high.
        messages.append(
            f"unstable: CoV baseline={base_cov:.1%} candidate={cand_cov:.1%} "
            f"(>= {thresholds.cov_unstable:.0%}); not making a hard decision"
        )
        outcome = Outcome.UNSTABLE if outcome >= Outcome.WARN else max(outcome, Outcome.OK)

    if not messages:
        messages.append("within thresholds")

    return CaseResult(
        identity=candidate.get("identity", {}),
        outcome=outcome,
        messages=messages,
        time_delta=time_delta,
        ttf_delta=ttf_delta,
        mem_delta=mem_delta,
        baseline_cov=base_cov,
        candidate_cov=cand_cov,
    )


@dataclass
class ComparisonReport:
    results: List[CaseResult]
    incompatible: List[Dict[str, Any]]  # identity dicts present on only one side
    exit_code: ExitCode

    def worst_outcome(self) -> Outcome:
        return max((r.outcome for r in self.results), default=Outcome.OK)


def compare(baseline_records: List[Dict[str, Any]],
            candidate_records: List[Dict[str, Any]],
            thresholds: Optional[Thresholds] = None,
            *, allow_incompatible: bool = False) -> ComparisonReport:
    """Compare two sets of records and produce a gating report.

    * Matched identities are compared case-by-case.
    * Identities present on only one side (or with a mismatched schema major
      version) are *incompatible*. By default their presence sets the
      INCOMPATIBLE exit code; ``allow_incompatible`` makes them informational.
    """
    thresholds = thresholds or Thresholds()
    base_idx = _index_aggregates(baseline_records)
    cand_idx = _index_aggregates(candidate_records)

    results: List[CaseResult] = []
    incompatible: List[Dict[str, Any]] = []

    common = set(base_idx) & set(cand_idx)
    for key in common:
        b = base_idx[key]
        c = cand_idx[key]
        if b.get("schema_version", "").split(".")[0] != c.get("schema_version", "").split(".")[0]:
            incompatible.append(c.get("identity", {}))
            continue
        results.append(compare_case(b, c, thresholds))

    for key in set(cand_idx) - set(base_idx):
        incompatible.append(cand_idx[key].get("identity", {}))
    for key in set(base_idx) - set(cand_idx):
        incompatible.append(base_idx[key].get("identity", {}))

    # Sort results by severity (worst first) for reporting.
    results.sort(key=lambda r: r.outcome, reverse=True)

    worst = max((r.outcome for r in results), default=Outcome.OK)
    if worst == Outcome.FAIL:
        exit_code = ExitCode.REGRESSION
    elif worst == Outcome.UNSTABLE:
        exit_code = ExitCode.UNSTABLE
    else:
        exit_code = ExitCode.OK

    if incompatible and not allow_incompatible:
        exit_code = ExitCode.INCOMPATIBLE

    return ComparisonReport(results=results, incompatible=incompatible, exit_code=exit_code)
