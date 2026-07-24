"""Cross-configuration reporting and speedup computation."""

from __future__ import annotations

from fastsafetensors_perf import metrics, report
from fastsafetensors_perf.results import Identity, RankMetrics, make_aggregate_record


def _identity(**over) -> Identity:
    base = dict(
        hardware_profile="h100", gpu_model="NVIDIA H100 80GB HBM3", world_size=1,
        storage_id="nvme:ext4", model_alias="synthetic", model_revision="main",
        model_fingerprint="abcd", mode="parallel", consumer="iterate",
        queue_size=0, backend="nogds", cache_policy="warm", tuning_key="",
        vllm_series="0.25", torch_series="2.11", cuda_series="13.0",
        fastsafetensors_series="0.3",
    )
    base.update(over)
    return Identity(**base)


def _agg(wall, logical_bytes, ident):
    reps = [
        [RankMetrics(rank=0, repetition=i, wall_seconds=w, logical_bytes=logical_bytes,
                     time_to_first_seconds=0.1, peak_cuda_allocated_bytes=10**9).to_dict()]
        for i, w in enumerate(wall)
    ]
    stats = metrics.aggregate_repetitions(reps)
    return make_aggregate_record(ident, {}, {}, stats,
                                 cache_eviction_available=True,
                                 n_repetitions=len(wall), n_ranks=1, worst_status="ok")


def test_speedup_vs_baseline_higher_is_better():
    # safetensors: 2 GB in 1.0 s -> 2 GB/s ; parallel: 2 GB in 0.5 s -> 4 GB/s
    two_gb = 2 * 10**9
    aggs = [
        _agg([1.0, 1.0, 1.0], two_gb, _identity(mode="safetensors", backend="mmap")),
        _agg([0.5, 0.5, 0.5], two_gb, _identity(mode="parallel")),
    ]
    rows = report.build_report(aggs, ["mode"], metric="delivery_gbps",
                               baseline_field="mode", baseline_value="safetensors")
    by_mode = {r.group["mode"]: r for r in rows}
    assert by_mode["safetensors"].speedup == 1.0
    assert by_mode["parallel"].speedup == 2.0  # 4 / 2 GB/s


def test_speedup_wall_lower_is_better():
    two_gb = 2 * 10**9
    aggs = [
        _agg([1.0, 1.0], two_gb, _identity(mode="safetensors", backend="mmap")),
        _agg([0.25, 0.25], two_gb, _identity(mode="parallel")),
    ]
    rows = report.build_report(aggs, ["mode"], metric="wall_s",
                               baseline_field="mode", baseline_value="safetensors")
    by_mode = {r.group["mode"]: r for r in rows}
    assert by_mode["parallel"].speedup == 4.0  # 1.0 / 0.25


def test_scaling_group_by_world_size():
    two_gb = 2 * 10**9
    aggs = [
        _agg([1.0, 1.0], two_gb, _identity(world_size=1, mode="vllm")),
        _agg([0.55, 0.55], two_gb, _identity(world_size=2, mode="vllm")),
    ]
    rows = report.build_report(aggs, ["world_size"], metric="delivery_gbps")
    labels = [r.label() for r in rows]
    assert "world_size=1" in labels and "world_size=2" in labels
    data = report.to_chart_data(rows)
    assert len(data["delivery_gbps"]) == 2


def test_format_table_smoke():
    two_gb = 2 * 10**9
    aggs = [_agg([1.0, 1.0], two_gb, _identity(mode="parallel"))]
    rows = report.build_report(aggs, ["mode"], metric="delivery_gbps")
    out = report.format_table(rows)
    assert "delivery GB/s" in out
    assert "parallel" in out
