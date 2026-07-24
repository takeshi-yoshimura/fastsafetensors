"""Versioned result schema, baseline identity, and JSONL I/O.

This module is the single source of truth for the on-disk benchmark contract. It
deliberately has no torch/vllm/CUDA imports so it can be validated in CPU-only
unit tests and reused by the ``compare`` command.

Record kinds
------------
Each benchmark case emits, to one JSONL file:

* one ``kind="rank"`` record per rank per repetition -- the raw measurement; and
* one ``kind="aggregate"`` record written by rank 0 -- robust statistics across
  repetitions, using the slowest rank as the distributed completion time.

Both record kinds share an ``identity`` block. Two results may only be compared
as the same baseline series when every field in :data:`IDENTITY_FIELDS` matches
(see :func:`identity_key` and :mod:`fastsafetensors_perf.compare`).
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Iterator, List, Optional

# Bump the MINOR component for additive, backward-compatible fields; bump MAJOR
# when a reader must change to interpret existing fields. ``compare`` refuses to
# gate across different MAJOR versions.
SCHEMA_VERSION = "1.0"

RECORD_KIND_RANK = "rank"
RECORD_KIND_AGGREGATE = "aggregate"

# --- Baseline identity ------------------------------------------------------
#
# A comparison is only valid when the candidate and baseline describe the same
# thing. These are the fields that must match exactly; any difference means a new
# baseline series, not a regression. Keep this list in one place so the schema
# docs, the runner, and the comparison gate cannot drift apart.
IDENTITY_FIELDS = (
    "hardware_profile",  # logical profile name, e.g. "a100", "dgx-spark"
    "gpu_model",  # exact GPU marketing name, e.g. "NVIDIA A100-SXM4-80GB"
    "world_size",
    "storage_id",  # device + filesystem + mount identity (see environment)
    "model_alias",
    "model_revision",  # pinned HF revision / checkpoint fingerprint
    "model_fingerprint",  # content hash of the shard set (size+name based)
    "mode",  # vllm | parallel | safetensors | vllm-model-load
    "consumer",  # iterate | copy
    "queue_size",
    "backend",  # effective backend: gds | nogds | mmap | unified
    "cache_policy",  # cold | warm
    "tuning_key",  # canonical string of direct-loader tuning knobs
    # Compatibility *series*, not exact build strings: a patch bump of the same
    # series is still comparable; a minor/major bump starts a new series.
    "vllm_series",
    "torch_series",
    "cuda_series",
    "fastsafetensors_series",
)


def series_of(version: Optional[str]) -> str:
    """Reduce a version string to its comparability series (major.minor).

    ``0.25.1`` -> ``0.25``; ``2.6.0+cu124`` -> ``2.6``; ``None``/``""`` -> ``""``.
    A patch/local bump stays in the same series (comparable); a minor bump
    starts a new series.
    """
    if not version:
        return ""
    core = str(version).split("+", 1)[0].strip()
    parts = core.split(".")
    if len(parts) >= 2:
        return f"{parts[0]}.{parts[1]}"
    return core


@dataclass
class Identity:
    """The baseline-identity block embedded in every record."""

    hardware_profile: str = ""
    gpu_model: str = ""
    world_size: int = 1
    storage_id: str = ""
    model_alias: str = ""
    model_revision: str = ""
    model_fingerprint: str = ""
    mode: str = ""
    consumer: str = ""
    queue_size: int = 0
    backend: str = ""
    cache_policy: str = ""
    tuning_key: str = ""
    vllm_series: str = ""
    torch_series: str = ""
    cuda_series: str = ""
    fastsafetensors_series: str = ""

    def key(self) -> tuple:
        return tuple(getattr(self, f) for f in IDENTITY_FIELDS)

    def to_dict(self) -> Dict[str, Any]:
        return {f: getattr(self, f) for f in IDENTITY_FIELDS}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Identity":
        return cls(**{f: d[f] for f in IDENTITY_FIELDS if f in d})


def identity_key(record: Dict[str, Any]) -> tuple:
    """Extract the identity tuple from a raw record dict."""
    ident = record.get("identity", {})
    return tuple(ident.get(f) for f in IDENTITY_FIELDS)


# --- Metric payloads --------------------------------------------------------


@dataclass
class RankMetrics:
    """Per-rank, per-repetition raw measurement.

    Byte fields are split into ``storage_bytes`` (what was read off disk,
    including packing) and ``logical_bytes`` (element_count * dtype_size of the
    yielded tensors). For packed sub-byte checkpoints (native F4, NVFP4) these
    differ and both are required.
    """

    rank: int = 0
    repetition: int = 0
    ok: bool = True
    status: str = "ok"  # ok | error | timeout
    error: str = ""

    wall_seconds: float = 0.0
    time_to_first_seconds: float = 0.0
    consumer_copy_seconds: float = 0.0

    source_checkpoint_bytes: int = 0  # bytes of the shard files this rank read
    storage_bytes: int = 0  # storage-layout bytes of yielded tensors
    logical_bytes: int = 0  # logical element bytes of yielded tensors

    shard_count: int = 0
    tensor_count: int = 0
    max_shard_bytes: int = 0

    # per-dtype: {dtype: {"count": n, "storage_bytes": s, "logical_bytes": l}}
    dtype_histogram: Dict[str, Dict[str, int]] = field(default_factory=dict)

    peak_cuda_allocated_bytes: int = 0
    peak_cuda_reserved_bytes: int = 0
    host_peak_rss_bytes: int = 0

    requested_backend: str = ""
    effective_backend: str = ""
    fallback: bool = False

    def storage_throughput_bps(self) -> float:
        return self.source_checkpoint_bytes / self.wall_seconds if self.wall_seconds > 0 else 0.0

    def delivery_throughput_bps(self) -> float:
        return self.logical_bytes / self.wall_seconds if self.wall_seconds > 0 else 0.0

    def to_dict(self) -> Dict[str, Any]:
        d = dataclasses.asdict(self)
        d["storage_throughput_bps"] = self.storage_throughput_bps()
        d["delivery_throughput_bps"] = self.delivery_throughput_bps()
        return d


# --- Full records -----------------------------------------------------------


def _base_record(kind: str, identity: Identity, environment: Dict[str, Any],
                 case: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": kind,
        "identity": identity.to_dict(),
        "environment": environment,
        "case": case,
    }


def make_rank_record(identity: Identity, environment: Dict[str, Any],
                     case: Dict[str, Any], metrics: RankMetrics) -> Dict[str, Any]:
    rec = _base_record(RECORD_KIND_RANK, identity, environment, case)
    rec["metrics"] = metrics.to_dict()
    return rec


def make_aggregate_record(identity: Identity, environment: Dict[str, Any],
                          case: Dict[str, Any], stats: Dict[str, Any],
                          cache_eviction_available: bool,
                          n_repetitions: int, n_ranks: int,
                          worst_status: str) -> Dict[str, Any]:
    rec = _base_record(RECORD_KIND_AGGREGATE, identity, environment, case)
    rec["aggregate"] = {
        "n_repetitions": n_repetitions,
        "n_ranks": n_ranks,
        "worst_status": worst_status,
        "cache_eviction_available": cache_eviction_available,
        "stats": stats,
    }
    return rec


# --- Validation -------------------------------------------------------------


class SchemaError(ValueError):
    """Raised when a record does not satisfy the schema contract."""


_REQUIRED_TOP = ("schema_version", "kind", "identity", "environment", "case")


def validate_record(record: Dict[str, Any]) -> None:
    """Validate a single record dict, raising :class:`SchemaError` on failure."""
    for key in _REQUIRED_TOP:
        if key not in record:
            raise SchemaError(f"missing required field: {key!r}")

    major = str(record["schema_version"]).split(".", 1)[0]
    if major != SCHEMA_VERSION.split(".", 1)[0]:
        raise SchemaError(
            f"incompatible schema major version {record['schema_version']!r} "
            f"(reader supports {SCHEMA_VERSION})"
        )

    kind = record["kind"]
    if kind not in (RECORD_KIND_RANK, RECORD_KIND_AGGREGATE):
        raise SchemaError(f"unknown record kind: {kind!r}")

    ident = record["identity"]
    missing = [f for f in IDENTITY_FIELDS if f not in ident]
    if missing:
        raise SchemaError(f"identity missing fields: {missing}")

    if kind == RECORD_KIND_RANK and "metrics" not in record:
        raise SchemaError("rank record missing 'metrics'")
    if kind == RECORD_KIND_AGGREGATE and "aggregate" not in record:
        raise SchemaError("aggregate record missing 'aggregate'")


# --- JSONL I/O --------------------------------------------------------------


def write_records(path: str, records: Iterable[Dict[str, Any]], *, append: bool = True) -> None:
    """Append records to a JSONL file (one compact JSON object per line)."""
    mode = "a" if append else "w"
    with open(path, mode, encoding="utf-8") as fh:
        for rec in records:
            validate_record(rec)
            fh.write(json.dumps(rec, sort_keys=True, separators=(",", ":")))
            fh.write("\n")


def read_records(path: str, *, validate: bool = True) -> List[Dict[str, Any]]:
    """Read all records from a JSONL file."""
    out: List[Dict[str, Any]] = []
    for rec in iter_records(path, validate=validate):
        out.append(rec)
    return out


def iter_records(path: str, *, validate: bool = True) -> Iterator[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SchemaError(f"{path}:{lineno}: invalid JSON: {exc}") from exc
            if validate:
                validate_record(rec)
            yield rec


def aggregate_records(records: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [r for r in records if r.get("kind") == RECORD_KIND_AGGREGATE]
