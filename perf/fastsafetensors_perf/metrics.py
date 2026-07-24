"""Robust statistics and byte/dtype accounting.

CPU-only, no torch import. Given a list of per-repetition rank records, produce
the aggregate statistics required by the schema (median, p90, MAD, coefficient
of variation) using the slowest rank per repetition as the distributed
completion time.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Sequence

# Bytes per element for the dtypes we account for. Packed sub-byte dtypes have a
# storage size below one byte per logical element; we track logical vs storage
# separately, so this table is the *logical* element size in bits.
DTYPE_LOGICAL_BITS = {
    "F64": 64,
    "F32": 32,
    "F16": 16,
    "BF16": 16,
    "F8_E4M3": 8,
    "F8_E5M2": 8,
    "I64": 64,
    "I32": 32,
    "I16": 16,
    "I8": 8,
    "U8": 8,
    "BOOL": 8,
    # Packed 4-bit floats: two logical elements per storage byte.
    "F4": 4,
    "FP4": 4,
    "F4_E2M1": 4,
}


def median(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    n = len(s)
    mid = n // 2
    if n % 2 == 1:
        return float(s[mid])
    return (s[mid - 1] + s[mid]) / 2.0


def percentile(values: Sequence[float], pct: float) -> float:
    """Linear-interpolation percentile (pct in [0, 100])."""
    if not values:
        return 0.0
    s = sorted(values)
    if len(s) == 1:
        return float(s[0])
    rank = (pct / 100.0) * (len(s) - 1)
    lo = math.floor(rank)
    hi = math.ceil(rank)
    if lo == hi:
        return float(s[int(rank)])
    frac = rank - lo
    return float(s[lo] * (1 - frac) + s[hi] * frac)


def mad(values: Sequence[float]) -> float:
    """Median absolute deviation from the median (unscaled)."""
    if not values:
        return 0.0
    med = median(values)
    return median([abs(v - med) for v in values])


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def stddev(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    m = mean(values)
    var = sum((v - m) ** 2 for v in values) / (len(values) - 1)
    return math.sqrt(var)


def coefficient_of_variation(values: Sequence[float]) -> float:
    """Sample stddev / mean. 0 when mean is 0 or fewer than 2 samples."""
    m = mean(values)
    if m == 0 or len(values) < 2:
        return 0.0
    return stddev(values) / m


def summarize(values: Sequence[float]) -> Dict[str, float]:
    """The standard robust-statistics bundle for one metric across repetitions."""
    return {
        "median": median(values),
        "p90": percentile(values, 90.0),
        "mad": mad(values),
        "cov": coefficient_of_variation(values),
        "min": float(min(values)) if values else 0.0,
        "max": float(max(values)) if values else 0.0,
        "n": len(values),
    }


def distributed_wall_seconds(per_rank_wall: Sequence[float]) -> float:
    """Distributed completion time for one repetition = slowest rank."""
    return float(max(per_rank_wall)) if per_rank_wall else 0.0


def aggregate_repetitions(
    rank_metrics_by_rep: List[List[Dict[str, Any]]],
) -> Dict[str, Any]:
    """Compute aggregate statistics across repetitions.

    ``rank_metrics_by_rep`` is a list (one entry per repetition) of lists of
    per-rank metric dicts (as produced by :meth:`RankMetrics.to_dict`).

    For time-like metrics that describe distributed completion (``wall_seconds``,
    ``time_to_first_seconds``) we take the slowest rank within a repetition and
    then summarize across repetitions. For resource metrics
    (peak memory, bytes) we take the max/sum across ranks per repetition as
    appropriate, then summarize.
    """
    wall_per_rep = []
    ttf_per_rep = []
    copy_per_rep = []
    peak_cuda_alloc_per_rep = []
    peak_cuda_reserved_per_rep = []
    host_rss_per_rep = []
    logical_bytes_per_rep = []
    storage_bytes_per_rep = []
    source_bytes_per_rep = []

    # Telemetry, reduced across ranks per repetition. CPU / host-mem / disk /
    # NVLink are host-or-node totals -> sum across ranks; GPU utilization is an
    # average across the GPUs; GPU memory is the worst single device -> max.
    def _get(rep, key):
        return [m.get(key, 0) or 0 for m in rep]

    telemetry_specs = {
        "cpu_user_pct": sum,
        "cpu_system_pct": sum,
        "host_mem_increase_bytes": sum,
        "disk_read_bps": sum,
        "read_char_bps": sum,
        "gpu_util_pct": lambda xs: (sum(xs) / len(xs)) if xs else 0.0,
        "gpu_mem_used_bytes": max,
        "nvlink_bps": sum,
    }
    telemetry_per_rep = {k: [] for k in telemetry_specs}

    for rep in rank_metrics_by_rep:
        if not rep:
            continue
        wall_per_rep.append(distributed_wall_seconds([m["wall_seconds"] for m in rep]))
        ttf_per_rep.append(distributed_wall_seconds([m["time_to_first_seconds"] for m in rep]))
        copy_per_rep.append(distributed_wall_seconds([m["consumer_copy_seconds"] for m in rep]))
        peak_cuda_alloc_per_rep.append(max(m["peak_cuda_allocated_bytes"] for m in rep))
        peak_cuda_reserved_per_rep.append(max(m["peak_cuda_reserved_bytes"] for m in rep))
        host_rss_per_rep.append(max(m["host_peak_rss_bytes"] for m in rep))
        logical_bytes_per_rep.append(sum(m["logical_bytes"] for m in rep))
        storage_bytes_per_rep.append(sum(m["storage_bytes"] for m in rep))
        source_bytes_per_rep.append(sum(m["source_checkpoint_bytes"] for m in rep))
        for key, reducer in telemetry_specs.items():
            telemetry_per_rep[key].append(reducer(_get(rep, key)))

    wall_median = median(wall_per_rep) if wall_per_rep else 0.0
    logical_median = median(logical_bytes_per_rep) if logical_bytes_per_rep else 0.0
    source_median = median(source_bytes_per_rep) if source_bytes_per_rep else 0.0

    out = {
        "wall_seconds": summarize(wall_per_rep),
        "time_to_first_seconds": summarize(ttf_per_rep),
        "consumer_copy_seconds": summarize(copy_per_rep),
        "peak_cuda_allocated_bytes": summarize(peak_cuda_alloc_per_rep),
        "peak_cuda_reserved_bytes": summarize(peak_cuda_reserved_per_rep),
        "host_peak_rss_bytes": summarize(host_rss_per_rep),
        "logical_bytes": summarize(logical_bytes_per_rep),
        "storage_bytes": summarize(storage_bytes_per_rep),
        "source_checkpoint_bytes": summarize(source_bytes_per_rep),
        # Derived throughput using medians -- convenient headline numbers.
        "delivery_throughput_bps": (logical_median / wall_median) if wall_median > 0 else 0.0,
        "storage_throughput_bps": (source_median / wall_median) if wall_median > 0 else 0.0,
    }
    for key in telemetry_specs:
        out[key] = summarize(telemetry_per_rep[key])
    return out
