# DGX Spark performance tuning

## GB10 CPU build

On AArch64, set `FASTSAFETENSORS_CPU_TARGET=gb10`. The configured C++ compiler
is feature-tested; `-mcpu=gb10` is added only when accepted. Other architectures,
unset targets, and rejecting compilers keep portable flags.

```console
FASTSAFETENSORS_CPU_TARGET=gb10 python -m build
```

Compare against a portable build on identical model/storage, recording loader
CPU time and end-to-end load time; no CPU speedup is claimed without results.

## Loader profiling

Set `FASTSAFETENSORS_PROFILE=1` to emit one structured DMA, submit, and
materialization record per shard, plus producer/consumer batch timings:

```console
FASTSAFETENSORS_PROFILE=1 vllm serve ... 2>&1 | tee /tmp/fst-profile.log
python benchmarks/dgxspark/summarize_profile.py /tmp/fst-profile.log
```

To reduce CUDA caching-allocator growth caused by differently sized shards,
round transient loader buffers to a reusable capacity. For checkpoints with
shards no larger than 4 GiB (including Qwen3.6-35B-A3B), use:

```bash
export FASTSAFETENSORS_ALLOC_GRANULARITY_MB=4096
```

This does not increase disk I/O, but an unbuffered pipeline may keep two rounded
buffers live at once. Leave it unset (or set it to `0`) when that extra transient
GPU-memory headroom is unavailable.

To pipeline within each shard, set a maximum tensor-chunk span. The producer
then makes each chunk visible to the weight iterator as soon as it is ready,
while reading the following chunk. Qwen3.6-35B-A3B has individual tensors up
to 1 GiB, so use 1 GiB for both chunking and reusable allocation capacity:

```bash
export FASTSAFETENSORS_MAX_BATCH_MB=1024
export FASTSAFETENSORS_ALLOC_GRANULARITY_MB=1024
```

A chunk cannot split one tensor. If the configured size is smaller than the
checkpoint's largest tensor, loading fails before data transfer and reports
the required tensor size. Unset `FASTSAFETENSORS_MAX_BATCH_MB` (or set it to
`0`) to retain the original shard-at-a-time behavior.

`wall_ms` is elapsed time; fields ending in `worker_ms` are sums across worker
threads and therefore may exceed wall time. `pread_worker_ms` versus
`memcpy_worker_ms` indicates whether deeper asynchronous I/O or asynchronous
CUDA copies are more promising. `queue_wait` measures consumer starvation;
`get_tensor_total` includes time while vLLM consumes yielded tensors.
`physical_read_gib / logical_gib` reports O_DIRECT alignment/read amplification.
Profiling is disabled unless the environment variable is exactly `1`.
