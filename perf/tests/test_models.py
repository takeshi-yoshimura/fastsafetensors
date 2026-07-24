"""Discovery, index filtering, natural sort, and inspection."""

from __future__ import annotations

import os

from conftest import make_sharded_checkpoint, write_safetensors
from fastsafetensors_perf.models import (
    discover_shards,
    inspect_checkpoint,
    natural_sort_key,
)


def test_natural_sort_matches_vllm_shape():
    key = natural_sort_key("/x/model-00001-of-00005.safetensors")
    assert key == ["model-", 1, "-of-", 5, ".safetensors"]


def test_natural_sort_orders_numerically_not_lexically():
    files = [
        "model-00010-of-00012.safetensors",
        "model-00002-of-00012.safetensors",
        "model-00001-of-00012.safetensors",
    ]
    ordered = sorted(files, key=natural_sort_key)
    assert ordered[0].endswith("00001-of-00012.safetensors")
    assert ordered[1].endswith("00002-of-00012.safetensors")
    assert ordered[2].endswith("00010-of-00012.safetensors")


def test_discover_single_file(tmp_path):
    p = tmp_path / "model.safetensors"
    write_safetensors(str(p), {"w": ("F32", [2, 2])})
    assert discover_shards(str(tmp_path)) == [str(p)]


def test_index_filters_stray_consolidated_file(tmp_path):
    d = str(tmp_path)
    make_sharded_checkpoint(d, [
        {"a": ("BF16", [4, 4])},
        {"b": ("BF16", [4, 4])},
    ])
    # A stray consolidated file that is NOT in the index must be excluded.
    write_safetensors(os.path.join(d, "model.safetensors"), {"a": ("BF16", [4, 4]),
                                                             "b": ("BF16", [4, 4])})
    shards = discover_shards(d)
    names = [os.path.basename(s) for s in shards]
    assert names == [
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors",
    ]
    assert "model.safetensors" not in names


def test_inspect_byte_and_dtype_accounting(tmp_path):
    d = str(tmp_path)
    make_sharded_checkpoint(d, [
        {"a": ("F32", [10])},           # 40 storage / 40 logical bytes
        {"b": ("BF16", [10])},          # 20 / 20
    ])
    inv = inspect_checkpoint(d)
    assert inv.shard_count == 2
    assert inv.tensor_count == 2
    hist = inv.dtype_histogram()
    assert hist["F32"]["count"] == 1
    assert hist["F32"]["storage_bytes"] == 40
    assert hist["BF16"]["storage_bytes"] == 20
    assert inv.storage_bytes == 60
    assert inv.logical_bytes == 60


def test_inspect_packed_f4_logical_vs_storage(tmp_path):
    d = str(tmp_path)
    # 100 packed 4-bit elements => 50 storage bytes, 50 logical bytes (F4 is 4
    # logical bits per element in our accounting), plus an FP8 scale tensor.
    make_sharded_checkpoint(d, [
        {"w_packed": ("F4_E2M1", [100]), "w_scale": ("F8_E4M3", [4])},
    ])
    inv = inspect_checkpoint(d)
    hist = inv.dtype_histogram()
    assert hist["F4_E2M1"]["storage_bytes"] == 50
    assert hist["F4_E2M1"]["logical_bytes"] == 50
    assert hist["F8_E4M3"]["storage_bytes"] == 4


def test_fingerprint_stable_and_distinct(tmp_path):
    d1 = str(tmp_path / "m1")
    d2 = str(tmp_path / "m2")
    make_sharded_checkpoint(d1, [{"a": ("F32", [10])}])
    make_sharded_checkpoint(d2, [{"a": ("F32", [10])}])
    make_sharded_checkpoint(str(tmp_path / "m3"), [{"a": ("F32", [11])}])
    fp1 = inspect_checkpoint(d1).fingerprint()
    fp2 = inspect_checkpoint(d2).fingerprint()
    fp3 = inspect_checkpoint(str(tmp_path / "m3")).fingerprint()
    assert fp1 == fp2  # identical layout -> identical fingerprint
    assert fp1 != fp3  # different tensor size -> different fingerprint
