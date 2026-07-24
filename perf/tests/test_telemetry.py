"""Resource telemetry monitor and its aggregation (CPU-only, no GPU needed)."""

from __future__ import annotations

import time

from fastsafetensors_perf import metrics
from fastsafetensors_perf.telemetry import ResourceMonitor, TelemetryResult


def test_monitor_runs_without_gpu_and_measures_cpu():
    # NVML disabled so this is portable; a busy loop should register user CPU.
    with ResourceMonitor(enable_nvml=False, sample_interval=0.005) as mon:
        end = time.perf_counter() + 0.1
        x = 0
        while time.perf_counter() < end:
            x += 1  # burn user CPU
    r = mon.result
    assert isinstance(r, TelemetryResult)
    assert r.samples >= 1
    assert r.cpu_user_pct >= 0.0  # nonnegative; typically > 0 after a busy loop
    assert r.gpu_util_pct == 0.0  # NVML disabled
    assert r.nvlink_bytes == 0


def test_monitor_context_manager_populates_dict():
    with ResourceMonitor(enable_nvml=False) as mon:
        time.sleep(0.02)
    d = mon.result.to_dict()
    for key in ("cpu_user_pct", "cpu_system_pct", "host_mem_increase_bytes",
                "disk_read_bytes", "disk_read_bps", "gpu_util_pct",
                "gpu_mem_used_bytes", "nvlink_bytes", "nvlink_bps"):
        assert key in d


def _rank_metric(rank, **over):
    base = dict(
        wall_seconds=1.0, time_to_first_seconds=0.1, consumer_copy_seconds=0.0,
        peak_cuda_allocated_bytes=0, peak_cuda_reserved_bytes=0, host_peak_rss_bytes=0,
        logical_bytes=0, storage_bytes=0, source_checkpoint_bytes=0,
        cpu_user_pct=100.0, cpu_system_pct=20.0, host_mem_increase_bytes=10**9,
        disk_read_bps=2e9, gpu_util_pct=80.0, gpu_mem_used_bytes=5 * 10**9,
        nvlink_bps=3e9,
    )
    base.update(over)
    base["rank"] = rank
    return base


def test_aggregate_reduces_telemetry_across_ranks():
    # Two ranks in one repetition.
    rep = [
        _rank_metric(0, cpu_user_pct=100.0, gpu_util_pct=60.0,
                     gpu_mem_used_bytes=4 * 10**9, disk_read_bps=2e9, nvlink_bps=1e9,
                     host_mem_increase_bytes=10**9),
        _rank_metric(1, cpu_user_pct=150.0, gpu_util_pct=80.0,
                     gpu_mem_used_bytes=6 * 10**9, disk_read_bps=3e9, nvlink_bps=2e9,
                     host_mem_increase_bytes=2 * 10**9),
    ]
    stats = metrics.aggregate_repetitions([rep])
    # CPU / disk / NVLink / host-mem sum across ranks.
    assert stats["cpu_user_pct"]["median"] == 250.0
    assert stats["disk_read_bps"]["median"] == 5e9
    assert stats["nvlink_bps"]["median"] == 3e9
    assert stats["host_mem_increase_bytes"]["median"] == 3 * 10**9
    # GPU utilization averages across GPUs; GPU memory is the worst device.
    assert stats["gpu_util_pct"]["median"] == 70.0
    assert stats["gpu_mem_used_bytes"]["median"] == 6 * 10**9
