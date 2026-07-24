"""Comparison decisions, thresholds, and exit codes."""

from __future__ import annotations

from fastsafetensors_perf import metrics
from fastsafetensors_perf.compare import (
    ExitCode,
    Outcome,
    Thresholds,
    compare,
)
from fastsafetensors_perf.results import (
    Identity,
    RankMetrics,
    make_aggregate_record,
)


def _identity(**over) -> Identity:
    base = dict(
        hardware_profile="a100", gpu_model="NVIDIA A100-SXM4-80GB", world_size=1,
        storage_id="nvme:ext4", model_alias="qwen3-8b", model_revision="rev1",
        model_fingerprint="abcd", mode="parallel", consumer="iterate",
        queue_size=0, backend="nogds", cache_policy="cold", tuning_key="",
        vllm_series="0.25", torch_series="2.6", cuda_series="12.4",
        fastsafetensors_series="0.3",
    )
    base.update(over)
    return Identity(**base)


def _agg(wall_values, ident=None, ttf=0.1, mem=1000, status="ok"):
    ident = ident or _identity()
    reps = [
        [RankMetrics(rank=0, repetition=i, wall_seconds=w,
                     time_to_first_seconds=ttf, logical_bytes=100,
                     peak_cuda_allocated_bytes=mem,
                     host_peak_rss_bytes=mem).to_dict()]
        for i, w in enumerate(wall_values)
    ]
    stats = metrics.aggregate_repetitions(reps)
    return make_aggregate_record(ident, {}, {}, stats,
                                 cache_eviction_available=True,
                                 n_repetitions=len(wall_values), n_ranks=1,
                                 worst_status=status)


def test_within_threshold_is_ok():
    base = [_agg([2.00, 2.01, 2.00, 2.02, 2.01])]
    cand = [_agg([2.00, 2.02, 2.01, 2.01, 2.00])]
    report = compare(base, cand)
    assert report.exit_code == ExitCode.OK
    assert report.results[0].outcome == Outcome.OK


def test_time_regression_fails():
    base = [_agg([2.00, 2.00, 2.00, 2.00, 2.00])]
    cand = [_agg([2.40, 2.40, 2.40, 2.40, 2.40])]  # +20% -> fail
    report = compare(base, cand)
    assert report.exit_code == ExitCode.REGRESSION
    assert report.results[0].outcome == Outcome.FAIL


def test_time_regression_warns():
    base = [_agg([2.00, 2.00, 2.00, 2.00, 2.00])]
    cand = [_agg([2.24, 2.24, 2.24, 2.24, 2.24])]  # +12% -> warn
    report = compare(base, cand)
    assert report.exit_code == ExitCode.OK  # warn does not gate
    assert report.results[0].outcome == Outcome.WARN


def test_unstable_downgrades_hard_decision():
    base = [_agg([2.00, 2.00, 2.00, 2.00, 2.00])]
    # candidate median ~2.4 (would fail) but extreme variance -> unstable
    cand = [_agg([1.0, 3.8, 1.2, 3.6, 2.4])]
    report = compare(base, cand)
    assert report.results[0].outcome == Outcome.UNSTABLE
    assert report.exit_code == ExitCode.UNSTABLE


def test_memory_regression_warns():
    base = [_agg([2.0, 2.0, 2.0, 2.0, 2.0], mem=1 * 1024 * 1024 * 1024)]
    cand = [_agg([2.0, 2.0, 2.0, 2.0, 2.0], mem=2 * 1024 * 1024 * 1024)]
    report = compare(base, cand)
    assert report.results[0].outcome == Outcome.WARN
    assert any("peak" in m for m in report.results[0].messages)


def test_candidate_failure_is_fail():
    base = [_agg([2.0, 2.0, 2.0])]
    cand = [_agg([2.0, 2.0, 2.0], status="timeout")]
    report = compare(base, cand)
    assert report.results[0].outcome == Outcome.FAIL
    assert report.exit_code == ExitCode.REGRESSION


def test_incompatible_identity_refused_by_default():
    base = [_agg([2.0, 2.0, 2.0])]
    cand = [_agg([2.0, 2.0, 2.0], ident=_identity(gpu_model="NVIDIA H100"))]
    report = compare(base, cand)
    assert report.exit_code == ExitCode.INCOMPATIBLE
    assert report.incompatible

    report2 = compare(base, cand, allow_incompatible=True)
    assert report2.exit_code == ExitCode.OK
