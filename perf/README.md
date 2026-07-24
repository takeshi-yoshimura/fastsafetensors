# fastsafetensors-perf

A regression benchmark for the `fastsafetensors` `ParallelLoader`, focused on how
vLLM uses it to load model weights. It replaces the old
`fastsafetensors_perf/perf.py` GPT-2 script (see
[`docs/amd-perf.md`](../docs/amd-perf.md) for the historical numbers).

The benchmark answers one question: **did a change to `fastsafetensors` make
weight loading slower or heavier, for a fixed model / GPU / storage / vLLM
version?** It is not a serving-throughput or GEMM benchmark.

## Install

Always install the checkout under test together with this package, so the
benchmark cannot silently pick up the PyPI wheel:

```bash
uv venv --python 3.12
uv pip install -e . -e ./perf
# vLLM modes are optional and pin the whole baseline series:
uv pip install -e '.[vllm]' -e ./perf   # or: pip install 'fastsafetensors-perf[vllm]'
```

`inspect`, the `parallel` and `safetensors` modes, and all unit tests run
without vLLM. Only `mode=vllm` / `vllm-model-load` require the `vllm` extra.

## Commands

```bash
# Header-only inventory: shards, tensors, dtype histogram, logical/storage bytes.
fastsafetensors-perf inspect /models/Qwen3-8B

# Single case (single GPU). Emits rank + aggregate JSONL records.
fastsafetensors-perf run /models/Qwen3-8B \
  --mode vllm --consumer copy --queue-size 0 \
  --cache cold --repeat 5 --output results/a100-qwen3-8b.jsonl

# Multi-GPU via the standard launcher (respects CUDA_VISIBLE_DEVICES):
torchrun --standalone --nproc-per-node=4 \
  -m fastsafetensors_perf.worker run /models/Qwen3-8B \
  --mode vllm --world-size 4 --output results/a100-qwen3-8b-ws4.jsonl

# A whole matrix from a host config:
fastsafetensors-perf matrix configs/a100.json \
  --models configs/models.local.json --output-dir results/a100

# Gate a candidate against a baseline (exit code drives CI):
fastsafetensors-perf compare results/baseline.jsonl results/candidate.jsonl

# Cross-configuration comparison for a report/talk (NON-gating):
fastsafetensors-perf report results/*.jsonl --group-by mode \
  --baseline-field mode --baseline-value safetensors
```

`compare` and `report` are complementary. `compare` is the **regression gate**:
it refuses to compare across different identities (a different GPU/model/vLLM
version is a different baseline series). `report` is the opposite — it
**deliberately** lines results up across an axis (mode, world size, queue size,
cache, model) to answer "how much faster is X than Y?", e.g. safetensors vs
fastsafetensors, or 1- vs 2-GPU scaling. `--json` emits chart-ready series. The
same JSONL feeds both, so data collected for regression testing is reusable for
a performance write-up.

## Benchmark modes

| Mode | What it exercises |
|---|---|
| `vllm` | The **installed** `vllm...weight_utils.fastsafetensors_weights_iterator`. Covers vLLM's file ordering, process-group selection, GDS/nogds choice, fallback, and cleanup. Never a local copy. |
| `parallel` | `fastsafetensors.parallel_loader.ParallelLoader` directly, exposing `nogds`, thread count, bounce-buffer size, `max_batch_bytes`, unified-memory settings. |
| `safetensors` | The stock `safetensors` mmap→device path, as an external baseline. |
| `vllm-model-load` | Optional canary: instantiate a vLLM model in-process with `load_format="fastsafetensors"`, no server. Release/weekly only. |

Consumers: `iterate` (consume + record metadata/bytes) and `copy` (add a
device-to-device copy per tensor as a proxy for `model.load_weights()` pressure —
a proxy, **not** a full model load). Both consume the full iterator, synchronize
CUDA before the final timestamp, and verify tensor count and byte totals.

## Result schema

One JSONL file per case, holding:

* one `kind="rank"` record **per rank per repetition** — the raw measurement; and
* one `kind="aggregate"` record written by rank 0 — robust statistics across
  repetitions, using the **slowest rank** as the distributed completion time.

`schema_version` is `MAJOR.MINOR`; readers reject a different MAJOR. Every record
carries `identity`, `environment`, and `case` blocks. Human-readable stdout is
never parsed to build results.

### Baseline identity

Two results may only be compared as the same series when **every** identity field
matches. Any difference is a new baseline series, not a regression:

`hardware_profile`, `gpu_model`, `world_size`, `storage_id`, `model_alias`,
`model_revision`, `model_fingerprint`, `mode`, `consumer`, `queue_size`,
`backend`, `cache_policy`, `tuning_key`, `vllm_series`, `torch_series`,
`cuda_series`, `fastsafetensors_series`.

Version fields are compared as **series** (`major.minor`): a patch/local bump
stays comparable; a minor bump starts a new series. A vLLM upgrade therefore
starts a new baseline series unless explicitly cross-validated.

### Example aggregate record

```json
{
  "schema_version": "1.0",
  "kind": "aggregate",
  "identity": {
    "hardware_profile": "a100", "gpu_model": "NVIDIA A100-SXM4-80GB",
    "world_size": 1, "storage_id": "nvme0n1:ext4", "model_alias": "qwen3-8b",
    "model_revision": "main", "model_fingerprint": "9f3c1a2b4d5e6f70",
    "mode": "vllm", "consumer": "copy", "queue_size": 0, "backend": "nogds",
    "cache_policy": "cold", "tuning_key": "",
    "vllm_series": "0.25", "torch_series": "2.6", "cuda_series": "12.4",
    "fastsafetensors_series": "0.3"
  },
  "environment": {"hostname": "gpu-a100-01", "git_sha": "abcd123", "git_dirty": false},
  "case": {"case_id": "qwen3-8b/vllm/copy/ws1/q0/cold", "repeat": 5},
  "aggregate": {
    "n_repetitions": 5, "n_ranks": 1, "worst_status": "ok",
    "cache_eviction_available": true,
    "stats": {"wall_seconds": {"median": 3.21, "p90": 3.30, "mad": 0.04, "cov": 0.02}}
  }
}
```

## Regression policy

Applied by `compare`, worst outcome first, with a process exit code:

| Condition | Outcome | Exit |
|---|---|---|
| Candidate did not complete cleanly | fail | 1 |
| Median wall time regression ≥ 15% | fail | 1 |
| Median wall time regression ≥ 10% | warn | 0 |
| Time-to-first-tensor regression ≥ 20% | warn | 0 |
| Peak CUDA/host memory +≥ 10% **and** ≥ 512 MiB | warn (fail per profile) | 0/1 |
| CoV ≥ 10% (baseline or candidate) | unstable — no hard decision | 3 |
| Identity mismatch / schema major mismatch | incompatible (refused) | 2 |

`compare` refuses incompatible series by default; `--allow-incompatible` makes
them informational and non-gating.

## Cache protocol

* `cold`: rank 0 applies `posix_fadvise(POSIX_FADV_DONTNEED)` to every shard,
  then all ranks barrier. Records when cold eviction is unavailable/ineffective.
* `warm`: one unrecorded full warmup before the recorded repetitions.
* Never writes `/proc/sys/vm/drop_caches` and never evicts unrelated cache.
* GDS/O_DIRECT paths may bypass page cache; the cold/warm label is retained so
  nogds/unified/mmap paths stay comparable.

Use ≥ 5 repetitions for standard cases.

## Local model configuration

Committed configs use **model aliases**, never machine-specific absolute paths.
Aliases resolve through `models.local.json` (git-ignored) or `--model-root`.
Result JSONL files and traces are git-ignored.

## Layout

```
perf/
├── README.md
├── pyproject.toml
├── configs/            # host matrices + example model map
├── fastsafetensors_perf/
│   ├── cli.py          # typer app: inspect / run / matrix / compare
│   ├── compare.py      # regression gate
│   ├── environment.py  # host / GPU / version inventory
│   ├── loaders.py      # vllm / parallel / safetensors modes + consumers
│   ├── metrics.py      # robust statistics, byte/dtype accounting
│   ├── models.py       # discovery, index parsing, natural sort, inspect
│   ├── results.py      # versioned schema, identity, JSONL I/O
│   └── worker.py       # single-GPU + torchrun worker
└── tests/              # CPU unit tests (no torch/CUDA required)
```
