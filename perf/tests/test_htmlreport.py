"""HTML report generation (fixed design, data-driven)."""

from __future__ import annotations

from fastsafetensors_perf import htmlreport, metrics
from fastsafetensors_perf.results import Identity, RankMetrics, make_aggregate_record


def _identity(**over) -> Identity:
    base = dict(
        hardware_profile="h100", gpu_model="NVIDIA H100 80GB HBM3", world_size=1,
        storage_id="tmpfs", model_alias="synthetic", model_revision="main",
        model_fingerprint="abcd", mode="parallel", consumer="iterate",
        queue_size=0, backend="nogds", cache_policy="warm", tuning_key="",
        vllm_series="0.25", torch_series="2.11", cuda_series="13.0",
        fastsafetensors_series="0.3",
    )
    base.update(over)
    return Identity(**base)


def _agg(mode, ws, wall, gbps_bytes, backend, **tel):
    reps = []
    for i in range(3):
        m = RankMetrics(rank=0, repetition=i, wall_seconds=wall,
                        logical_bytes=gbps_bytes, source_checkpoint_bytes=gbps_bytes,
                        gpu_util_pct=tel.get("gpu_util", 30), **{})
        reps.append([m.to_dict()])
    stats = metrics.aggregate_repetitions(reps)
    env = {"gpu": {"name": "NVIDIA H100 80GB HBM3"},
           "versions": {"torch": "2.11.0", "cuda": "13.0", "vllm": "0.25.1",
                        "fastsafetensors": "0.3.3"}, "git_sha": "abc123def"}
    case = {"repeat": 3, "checkpoint": {"source_checkpoint_bytes": gbps_bytes}}
    return make_aggregate_record(_identity(mode=mode, world_size=ws, backend=backend),
                                 env, case, stats, cache_eviction_available=True,
                                 n_repetitions=3, n_ranks=ws, worst_status="ok")


def _trace(mode, ws):
    ranks = [{"rank": r, "samples": [
        {"t": 0.0, "cpu_user_pct": 0, "cpu_system_pct": 0, "gpu_util_pct": 0,
         "gpu_mem_gb": 0.0, "host_rss_gb": 1.0, "disk_gbps": 0.0, "read_gbps": 0.0,
         "nvlink_gbps": 0.0},
        {"t": 0.5, "cpu_user_pct": 50, "cpu_system_pct": 30, "gpu_util_pct": 40,
         "gpu_mem_gb": 3.0, "host_rss_gb": 1.1, "disk_gbps": 0.0, "read_gbps": 8.0,
         "nvlink_gbps": 1.2 if ws > 1 else 0.0},
    ]} for r in range(ws)]
    return {"schema": "trace-1", "identity": {"mode": mode, "world_size": ws},
            "wall_seconds_median": 0.5, "ranks": ranks}


def test_render_produces_valid_html():
    aggs = [
        _agg("safetensors", 1, 1.0, 2_000_000_000, "mmap"),
        _agg("parallel", 1, 0.8, 2_000_000_000, "nogds"),
        _agg("vllm", 1, 0.8, 2_000_000_000, "nogds"),
        _agg("vllm", 2, 0.5, 2_000_000_000, "nogds"),  # vllm at 2 world sizes -> scaling
    ]
    traces = [_trace("vllm", 1), _trace("vllm", 2)]
    html = htmlreport.render(aggs, traces, title="Test report")
    assert html.startswith("<!doctype html>")
    assert "__DATA_BLOB__" not in html  # placeholder was replaced
    assert "Test report" in html
    # data-driven pieces present
    assert "Resource timeline" in html
    assert "NVIDIA H100" in html  # from environment chip
    assert "backend nogds" in html  # not "mmap"
    import json, re
    data = json.loads(re.search(r"const DATA = (\{.*?\});", html).group(1))
    labels = [b["label"] for b in data["bars_modes"]]
    assert labels[0] == "safetensors"  # baseline first
    assert data["bars_modes"][0]["base"] is True
    # fastsafetensors speedup > 1 vs safetensors baseline
    assert any(b["speedup"] > 1 for b in data["bars_modes"] if not b["base"])
    assert data["bars_scale"] is not None  # world_size 1 and 2 present
    assert len(data["traces"]) == 2


def test_render_without_traces():
    aggs = [_agg("vllm", 1, 0.8, 2_000_000_000, "nogds")]
    html = htmlreport.render(aggs, [])
    assert "<!doctype html>" in html
    import json, re
    data = json.loads(re.search(r"const DATA = (\{.*?\});", html).group(1))
    assert data["traces"] == []
