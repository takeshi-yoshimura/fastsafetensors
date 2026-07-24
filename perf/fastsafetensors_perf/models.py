"""Local checkpoint discovery, index parsing, and inspection.

CPU-only. Reads only safetensors *headers* (never tensor data) to build a
deterministic inventory: shard list, tensor count, per-dtype histogram, and
logical vs storage byte totals. The shard ordering matches vLLM's natural sort
so ``mode=vllm`` and ``mode=parallel`` see the same file order.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import struct
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .metrics import DTYPE_LOGICAL_BITS

SAFETENSORS_SUFFIX = ".safetensors"
INDEX_SUFFIX = ".safetensors.index.json"

# safetensors header length prefix is a little-endian u64.
_HEADER_LEN_STRUCT = struct.Struct("<Q")
# Guard against a corrupt/absurd length prefix (headers are JSON metadata, tiny
# relative to weights). 256 MiB is far beyond any real checkpoint header.
_MAX_HEADER_BYTES = 256 * 1024 * 1024


def natural_sort_key(filepath: str) -> List[object]:
    """vLLM-compatible natural sort key.

    Mirrors ``vllm...weight_utils._natural_sort_key`` so shard ordering is
    identical: ``model-00001-of-00005.safetensors`` ->
    ``['model-', 1, '-of-', 5, '.safetensors']``.
    """
    return [
        int(s) if s.isdigit() else s
        for s in re.split(r"(\d+)", os.path.basename(filepath))
    ]


def _read_header(path: str) -> Dict[str, object]:
    """Read and parse the JSON header of a safetensors file."""
    with open(path, "rb") as fh:
        prefix = fh.read(8)
        if len(prefix) != 8:
            raise ValueError(f"{path}: truncated safetensors header length prefix")
        (header_len,) = _HEADER_LEN_STRUCT.unpack(prefix)
        if header_len == 0 or header_len > _MAX_HEADER_BYTES:
            raise ValueError(f"{path}: implausible header length {header_len}")
        raw = fh.read(header_len)
        if len(raw) != header_len:
            raise ValueError(f"{path}: truncated safetensors header")
    return json.loads(raw)


@dataclass
class TensorInfo:
    name: str
    dtype: str
    shape: Tuple[int, ...]
    storage_bytes: int  # data_offsets[1] - data_offsets[0]

    @property
    def n_elements(self) -> int:
        n = 1
        for d in self.shape:
            n *= d
        return n

    @property
    def logical_bytes(self) -> int:
        bits = DTYPE_LOGICAL_BITS.get(self.dtype.upper())
        if bits is None:
            # Unknown dtype: fall back to the storage size so totals stay sane.
            return self.storage_bytes
        return (self.n_elements * bits + 7) // 8


@dataclass
class ShardInfo:
    path: str
    file_bytes: int
    tensors: List[TensorInfo] = field(default_factory=list)


@dataclass
class CheckpointInventory:
    """Deterministic summary of a checkpoint's shard/tensor/dtype layout."""

    model_dir: str
    shards: List[ShardInfo]

    @property
    def shard_count(self) -> int:
        return len(self.shards)

    @property
    def files(self) -> List[str]:
        return [s.path for s in self.shards]

    @property
    def tensor_count(self) -> int:
        return sum(len(s.tensors) for s in self.shards)

    @property
    def source_checkpoint_bytes(self) -> int:
        return sum(s.file_bytes for s in self.shards)

    @property
    def max_shard_bytes(self) -> int:
        return max((s.file_bytes for s in self.shards), default=0)

    @property
    def storage_bytes(self) -> int:
        return sum(t.storage_bytes for s in self.shards for t in s.tensors)

    @property
    def logical_bytes(self) -> int:
        return sum(t.logical_bytes for s in self.shards for t in s.tensors)

    def dtype_histogram(self) -> Dict[str, Dict[str, int]]:
        hist: Dict[str, Dict[str, int]] = {}
        for shard in self.shards:
            for t in shard.tensors:
                entry = hist.setdefault(
                    t.dtype, {"count": 0, "storage_bytes": 0, "logical_bytes": 0}
                )
                entry["count"] += 1
                entry["storage_bytes"] += t.storage_bytes
                entry["logical_bytes"] += t.logical_bytes
        return hist

    def fingerprint(self) -> str:
        """Content-independent fingerprint of the shard *set*.

        Uses shard basenames and sizes plus tensor count -- stable across hosts
        and cheap (no weight hashing), enough to detect a different checkpoint or
        a changed revision as a distinct baseline series.
        """
        h = hashlib.sha256()
        for shard in sorted(self.shards, key=lambda s: os.path.basename(s.path)):
            h.update(os.path.basename(shard.path).encode())
            h.update(struct.pack("<Q", shard.file_bytes))
            h.update(struct.pack("<I", len(shard.tensors)))
        h.update(struct.pack("<I", self.tensor_count))
        return h.hexdigest()[:16]

    def summary_dict(self) -> Dict[str, object]:
        return {
            "model_dir": self.model_dir,
            "shard_count": self.shard_count,
            "tensor_count": self.tensor_count,
            "source_checkpoint_bytes": self.source_checkpoint_bytes,
            "storage_bytes": self.storage_bytes,
            "logical_bytes": self.logical_bytes,
            "max_shard_bytes": self.max_shard_bytes,
            "fingerprint": self.fingerprint(),
            "dtype_histogram": self.dtype_histogram(),
            "files": [os.path.basename(f) for f in self.files],
        }


def _index_shard_files(model_dir: str) -> Optional[List[str]]:
    """Return the shard files listed by a ``*.safetensors.index.json``, if any.

    Handles both the standard per-tensor ``weight_map`` and consolidated indexes.
    Returns None when no index is present (single-file or bare directory).
    """
    index_files = [
        os.path.join(model_dir, f)
        for f in os.listdir(model_dir)
        if f.endswith(INDEX_SUFFIX)
    ]
    if not index_files:
        return None
    # If multiple indexes exist, prefer the canonical model.safetensors.index.json.
    index_files.sort(key=lambda p: (os.path.basename(p) != "model" + INDEX_SUFFIX,
                                     natural_sort_key(p)))
    index_path = index_files[0]
    with open(index_path, "r", encoding="utf-8") as fh:
        index = json.load(fh)
    weight_map = index.get("weight_map", {})
    shard_names = sorted(set(weight_map.values()), key=natural_sort_key)
    return [os.path.join(model_dir, name) for name in shard_names]


def discover_shards(model_dir: str) -> List[str]:
    """Return the ordered list of safetensors shard files for a checkpoint.

    Rules:
    * If an index file is present, use exactly the shards it references. This
      filters out a stray consolidated ``model.safetensors`` living beside a
      sharded set.
    * Otherwise, use every ``*.safetensors`` file in the directory.
    * Order is always vLLM natural sort.
    """
    if not os.path.isdir(model_dir):
        if os.path.isfile(model_dir) and model_dir.endswith(SAFETENSORS_SUFFIX):
            return [model_dir]
        raise FileNotFoundError(f"not a model directory or safetensors file: {model_dir}")

    indexed = _index_shard_files(model_dir)
    if indexed is not None:
        missing = [p for p in indexed if not os.path.isfile(p)]
        if missing:
            raise FileNotFoundError(
                f"index references missing shards: {[os.path.basename(m) for m in missing]}"
            )
        return sorted(indexed, key=natural_sort_key)

    files = [
        os.path.join(model_dir, f)
        for f in os.listdir(model_dir)
        if f.endswith(SAFETENSORS_SUFFIX)
    ]
    if not files:
        raise FileNotFoundError(f"no .safetensors files in {model_dir}")
    return sorted(files, key=natural_sort_key)


def inspect_checkpoint(model_dir: str) -> CheckpointInventory:
    """Build the full inventory by reading every shard header."""
    shard_paths = discover_shards(model_dir)
    shards: List[ShardInfo] = []
    for path in shard_paths:
        header = _read_header(path)
        file_bytes = os.path.getsize(path)
        tensors: List[TensorInfo] = []
        for name, meta in header.items():
            if name == "__metadata__":
                continue
            if not isinstance(meta, dict):
                continue
            offsets = meta.get("data_offsets", [0, 0])
            storage_bytes = int(offsets[1]) - int(offsets[0])
            tensors.append(
                TensorInfo(
                    name=name,
                    dtype=str(meta.get("dtype", "")),
                    shape=tuple(int(x) for x in meta.get("shape", [])),
                    storage_bytes=storage_bytes,
                )
            )
        tensors.sort(key=lambda t: t.name)
        shards.append(ShardInfo(path=path, file_bytes=file_bytes, tensors=tensors))
    return CheckpointInventory(model_dir=os.path.abspath(model_dir), shards=shards)
