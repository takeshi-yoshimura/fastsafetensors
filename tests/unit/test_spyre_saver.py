# SPDX-License-Identifier: Apache-2.0
"""Spyre checkpoint writing without Spyre hardware or an inference framework."""

import builtins
import json
import sys
from types import ModuleType

import pytest
from safetensors import safe_open

from fastsafetensors import ParallelSaver
from fastsafetensors.common import SafeTensorsMetadata
from fastsafetensors.copier.spyre import (
    NATIVE_FORMAT,
    native_entry,
    native_fill,
    native_shard_metadata,
)
from fastsafetensors.native import native_format, native_specs
from fastsafetensors.saver import plain_entry
from fastsafetensors.st_types import DType

torch = pytest.importorskip("torch")


@pytest.fixture(autouse=True)
def no_inference_imports(monkeypatch):
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name.split(".")[0] in {"vllm", "spyre_inference"}:
            raise AssertionError(f"checkpoint writing imported {name}")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)


@pytest.mark.parametrize("num_shards", [1, 3])
def test_native_and_plain_entries_match_legacy_metadata(
    tmp_path, monkeypatch, framework, num_shards
):
    src = torch.arange(16, dtype=torch.float16).reshape(4, 4)
    dst = src.t()
    payload = bytes(range(48))
    plain = torch.ones(3)
    calls = []
    runtime = ModuleType("torch_spyre._C")

    def encode(work, num_threads):
        calls.append(num_threads)
        for buf, source, destination in work:
            assert source is src and destination is dst
            buf[:] = payload

    runtime.convert_h2d_into = encode
    monkeypatch.setitem(sys.modules, "torch_spyre._C", runtime)
    paths = ParallelSaver(
        num_shards=num_shards,
        fill=native_fill,
        shard_metadata=native_shard_metadata,
    ).save_entries(
        [
            native_entry("w", src, dst, nbytes=len(payload), layout={"tiles": 2}),
            plain_entry("plain", plain),
        ],
        str(tmp_path / "native"),
    )
    seen = set()
    for path in paths:
        meta = SafeTensorsMetadata.from_file(path, framework)
        assert native_format(meta) == NATIVE_FORMAT
        specs = native_specs(meta)
        with safe_open(path, framework="pt") as checkpoint:
            metadata = checkpoint.metadata()
            assert metadata["fst.native.version"] == "1"
            assert metadata["fst.native.device"] == "spyre"
            assert metadata["fst.native.format"] == "torch_spyre/1"
            expected_specs = {}
            names = set(checkpoint.keys())
            if "w" in names:
                expected_specs["w"] = {
                    "dtype": "F16",
                    "shape": [4, 4],
                    "stride": [1, 4],
                    "layout": {"tiles": 2},
                }
                assert checkpoint.get_tensor("w").numpy().tobytes() == payload
                assert specs["w"].dtype == DType.F16
                assert specs["w"].nbytes == len(payload)
            if "plain" in names:
                torch.testing.assert_close(checkpoint.get_tensor("plain"), plain)
            assert metadata["fst.native.tensors"] == json.dumps(
                expected_specs, separators=(",", ":")
            )
            assert set(specs) == names - {"plain"}
            seen.update(names)
    assert seen == {"w", "plain"}
    assert len(calls) == 1 and 1 <= calls[0] <= 64


@pytest.mark.parametrize("size", [0, -1])
def test_invalid_native_size(size):
    t = torch.ones(2, dtype=torch.float16)
    with pytest.raises(ValueError, match="must be positive"):
        native_entry("w", t, t, nbytes=size, layout={})


def test_native_entry_size_is_not_a_backend_transfer_limit():
    t = torch.ones(2, dtype=torch.float16)
    nbytes = (4 << 30) + 1
    assert native_entry("w", t, t, nbytes=nbytes, layout={}).nbytes == nbytes


def test_plain_fill_and_metadata_do_not_import_spyre_or_vllm(tmp_path, monkeypatch):
    original_import = builtins.__import__

    def no_backend(name, *args, **kwargs):
        if name.split(".")[0] in {"torch_spyre", "vllm", "spyre_inference"}:
            raise AssertionError(f"unexpected optional import: {name}")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_backend)
    native_fill([])
    tensor = torch.arange(6.0).reshape(2, 3).t()
    entry = native_entry("w", tensor, tensor, nbytes=tensor.nbytes, layout={})
    assert native_shard_metadata([entry])["fst.native.device"] == "spyre"
    paths = ParallelSaver(
        num_shards=1, fill=native_fill, shard_metadata=native_shard_metadata
    ).save({"plain": tensor}, str(tmp_path / "plain"))
    with safe_open(paths[0], framework="pt") as checkpoint:
        assert checkpoint.metadata()["fst.native.tensors"] == "{}"
        torch.testing.assert_close(checkpoint.get_tensor("plain"), tensor)


def test_native_fill_propagates_conversion_error_and_cleans_files(
    tmp_path, monkeypatch
):
    runtime = ModuleType("torch_spyre._C")

    def fail(*args, **kwargs):
        raise RuntimeError("conversion failed")

    runtime.convert_h2d_into = fail
    monkeypatch.setitem(sys.modules, "torch_spyre._C", runtime)
    tensor = torch.zeros(4, dtype=torch.float16)
    saver = ParallelSaver(
        num_shards=2, fill=native_fill, shard_metadata=native_shard_metadata
    )
    with pytest.raises(RuntimeError, match="conversion failed"):
        saver.save_entries(
            [native_entry("w", tensor, tensor, nbytes=tensor.nbytes, layout={})],
            str(tmp_path / "failed"),
        )
    assert list(tmp_path.iterdir()) == []
