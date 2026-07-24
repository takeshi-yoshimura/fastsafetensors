"""Shared test fixtures: synthetic safetensors checkpoints, no torch required."""

from __future__ import annotations

import json
import os
import struct
from typing import Dict, List, Sequence, Tuple

DTYPE_STORAGE_BITS = {
    "F64": 64, "F32": 32, "F16": 16, "BF16": 16,
    "F8_E4M3": 8, "F8_E5M2": 8,
    "I64": 64, "I32": 32, "I16": 16, "I8": 8, "U8": 8, "BOOL": 8,
    # Packed: two 4-bit elements per storage byte.
    "F4_E2M1": 4,
}


def _storage_bytes(dtype: str, shape: Sequence[int]) -> int:
    n = 1
    for d in shape:
        n *= d
    bits = DTYPE_STORAGE_BITS[dtype.upper()]
    return (n * bits + 7) // 8


def write_safetensors(path: str, tensors: Dict[str, Tuple[str, Sequence[int]]],
                      metadata: Dict[str, str] | None = None) -> int:
    """Write a valid safetensors file with zero-filled tensor data.

    ``tensors`` maps name -> (dtype, shape). Returns total file size in bytes.
    """
    header: Dict[str, object] = {}
    offset = 0
    for name, (dtype, shape) in tensors.items():
        nbytes = _storage_bytes(dtype, shape)
        header[name] = {
            "dtype": dtype,
            "shape": list(shape),
            "data_offsets": [offset, offset + nbytes],
        }
        offset += nbytes
    if metadata is not None:
        header["__metadata__"] = metadata

    header_bytes = json.dumps(header).encode("utf-8")
    with open(path, "wb") as fh:
        fh.write(struct.pack("<Q", len(header_bytes)))
        fh.write(header_bytes)
        fh.write(b"\x00" * offset)
    return os.path.getsize(path)


def make_sharded_checkpoint(model_dir: str,
                            shards: List[Dict[str, Tuple[str, Sequence[int]]]],
                            write_index: bool = True) -> None:
    """Create a multi-shard checkpoint with an optional index file."""
    os.makedirs(model_dir, exist_ok=True)
    n = len(shards)
    weight_map: Dict[str, str] = {}
    for i, tensors in enumerate(shards, 1):
        name = f"model-{i:05d}-of-{n:05d}.safetensors"
        write_safetensors(os.path.join(model_dir, name), tensors)
        for tname in tensors:
            weight_map[tname] = name
    if write_index:
        index = {"metadata": {"total_size": 0}, "weight_map": weight_map}
        with open(os.path.join(model_dir, "model.safetensors.index.json"), "w") as fh:
            json.dump(index, fh)
