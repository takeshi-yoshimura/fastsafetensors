#!/usr/bin/env python3
"""Summarize FASTSAFETENSORS_PROFILE=1 output from a vLLM log."""

import re
import sys
from collections import defaultdict

if len(sys.argv) != 2:
    raise SystemExit(f"usage: {sys.argv[0]} VLLM_LOG")

kv_re = re.compile(r"([a-z_]+)=([^ ]+)")
batch_re = re.compile(
    r"Batch \d+ summary: .*?copy_files=(?P<copy>[0-9.]+)ms, "
    r"queue_wait=(?P<queue>[0-9.]+)ms, "
    r"get_tensor_total=(?P<consume>[0-9.]+)ms, "
    r"close=(?P<close>[0-9.]+)ms"
)
rows = defaultdict(list)
batches = defaultdict(float)
with open(sys.argv[1], encoding="utf-8", errors="replace") as stream:
    for line in stream:
        if "[FST_PROFILE]" in line:
            kind = line.split("[FST_PROFILE]", 1)[1].strip().split()[0]
            values = {}
            for key, value in kv_re.findall(line):
                try:
                    values[key] = float(value)
                except ValueError:
                    pass
            rows[kind].append(values)
        match = batch_re.search(line)
        if match:
            for key, value in match.groupdict().items():
                batches[key] += float(value)


def total(kind, key):
    return sum(row.get(key, 0) for row in rows[kind])


def ms(kind, key):
    return total(kind, key) / 1000.0


requested = total("dma", "requested")
read = total("dma", "read")
dma_wall_s = ms("dma", "wall_ms")
print(f"dma_calls={len(rows['dma'])}")
print(f"logical_gib={requested / 2**30:.3f}")
print(f"physical_read_gib={read / 2**30:.3f}")
print(f"read_amplification={read / requested if requested else 0:.4f}x")
print(f"dma_wall_sum_s={dma_wall_s:.3f}")
print(
    f"logical_throughput_gib_s={requested / 2**30 / dma_wall_s if dma_wall_s else 0:.3f}"
)
print(
    f"physical_read_throughput_gib_s={read / 2**30 / dma_wall_s if dma_wall_s else 0:.3f}"
)
print(f"alloc_sum_s={ms('submit', 'alloc_ms'):.3f}")
print(f"pread_worker_sum_s={ms('dma', 'pread_worker_ms'):.3f}")
print(f"memcpy_worker_sum_s={ms('dma', 'memcpy_worker_ms'):.3f}")
print(f"sync_worker_sum_s={ms('dma', 'sync_worker_ms'):.3f}")
print(f"pin_worker_sum_s={ms('dma', 'pin_worker_ms'):.3f}")
print(f"materialize_sync_sum_s={ms('materialize', 'sync_ms'):.3f}")
print(f"dlpack_sum_s={ms('materialize', 'dlpack_ms'):.3f}")
print(f"batch_copy_sum_s={batches['copy'] / 1000:.3f}")
print(f"batch_queue_wait_sum_s={batches['queue'] / 1000:.3f}")
print(f"batch_consumer_sum_s={batches['consume'] / 1000:.3f}")
print(f"batch_close_sum_s={batches['close'] / 1000:.3f}")
