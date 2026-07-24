"""Robust statistics."""

from __future__ import annotations

import pytest

from fastsafetensors_perf import metrics


def test_median():
    assert metrics.median([3, 1, 2]) == 2
    assert metrics.median([1, 2, 3, 4]) == 2.5
    assert metrics.median([]) == 0.0


def test_percentile():
    assert metrics.percentile([1, 2, 3, 4, 5], 50) == 3
    assert metrics.percentile([1, 2, 3, 4, 5], 90) == pytest.approx(4.6)
    assert metrics.percentile([10], 90) == 10


def test_mad():
    # median 3; abs devs [2,1,0,1,2]; median of those = 1
    assert metrics.mad([1, 2, 3, 4, 5]) == 1


def test_cov():
    assert metrics.coefficient_of_variation([5, 5, 5]) == 0.0
    assert metrics.coefficient_of_variation([5]) == 0.0
    cov = metrics.coefficient_of_variation([10, 12])
    assert cov > 0


def test_distributed_wall_is_slowest_rank():
    assert metrics.distributed_wall_seconds([1.0, 3.0, 2.0]) == 3.0


def test_aggregate_uses_slowest_rank_per_rep():
    # rep0: ranks [1.0, 4.0] -> 4.0 ; rep1: ranks [2.0, 3.0] -> 3.0
    reps = [
        [
            {"wall_seconds": 1.0, "time_to_first_seconds": 0.1, "consumer_copy_seconds": 0.0,
             "peak_cuda_allocated_bytes": 10, "peak_cuda_reserved_bytes": 10,
             "host_peak_rss_bytes": 10, "logical_bytes": 50, "storage_bytes": 50,
             "source_checkpoint_bytes": 100},
            {"wall_seconds": 4.0, "time_to_first_seconds": 0.2, "consumer_copy_seconds": 0.0,
             "peak_cuda_allocated_bytes": 20, "peak_cuda_reserved_bytes": 20,
             "host_peak_rss_bytes": 20, "logical_bytes": 50, "storage_bytes": 50,
             "source_checkpoint_bytes": 100},
        ],
        [
            {"wall_seconds": 2.0, "time_to_first_seconds": 0.1, "consumer_copy_seconds": 0.0,
             "peak_cuda_allocated_bytes": 10, "peak_cuda_reserved_bytes": 10,
             "host_peak_rss_bytes": 10, "logical_bytes": 50, "storage_bytes": 50,
             "source_checkpoint_bytes": 100},
            {"wall_seconds": 3.0, "time_to_first_seconds": 0.2, "consumer_copy_seconds": 0.0,
             "peak_cuda_allocated_bytes": 20, "peak_cuda_reserved_bytes": 20,
             "host_peak_rss_bytes": 20, "logical_bytes": 50, "storage_bytes": 50,
             "source_checkpoint_bytes": 100},
        ],
    ]
    stats = metrics.aggregate_repetitions(reps)
    # wall per rep = [4.0, 3.0]; median 3.5
    assert stats["wall_seconds"]["median"] == pytest.approx(3.5)
    # logical bytes summed across ranks per rep = 100; median 100
    assert stats["logical_bytes"]["median"] == 100
    # peak cuda alloc is max across ranks per rep = 20
    assert stats["peak_cuda_allocated_bytes"]["median"] == 20
