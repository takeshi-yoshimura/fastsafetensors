# SPDX-License-Identifier: Apache-2.0
"""Native metadata parsing with checkpoints written through the upstream saver."""

import json

import pytest
from safetensors import safe_open

from fastsafetensors.common import SafeTensorsMetadata
from fastsafetensors.native import (
    KEY_DEVICE,
    KEY_FORMAT,
    KEY_TENSORS,
    KEY_VERSION,
    NativeFormat,
    NativeFormatError,
    native_format,
    native_specs,
)
from fastsafetensors.saver import (
    ParallelSaver,
    WriteEntry,
    plain_entry,
    tensor_fill,
    validate_shard_set,
)
from fastsafetensors.st_types import DType

torch = pytest.importorskip("torch")


@pytest.mark.parametrize("num_shards", [1, 3])
def test_native_and_plain_entries_with_upstream_saver(tmp_path, framework, num_shards):
    payload = bytes(range(48))
    plain = torch.ones(3)
    spec = {
        "dtype": "F16",
        "shape": [4, 4],
        "stride": [4, 1],
        "layout": {"tiles": 2},
    }

    def fill(work):
        tensor_fill()([(buf, e) for buf, e in work if e.name != "w"])
        for buf, entry in work:
            if entry.name == "w":
                buf[:] = entry.source

    paths = ParallelSaver(
        num_shards=num_shards,
        fill=fill,
        shard_metadata=lambda entries: {
            KEY_VERSION: "1",
            KEY_DEVICE: "fakedev",
            KEY_FORMAT: "fake/1",
            KEY_TENSORS: json.dumps({e.name: spec for e in entries if e.name == "w"}),
        },
    ).save_entries(
        [WriteEntry("w", DType.U8, [48], 48, payload), plain_entry("plain", plain)],
        str(tmp_path / "native"),
    )
    metas = [SafeTensorsMetadata.from_file(p, framework) for p in paths]
    validate_shard_set(metas)
    for meta in metas:
        assert native_format(meta) == NativeFormat("fakedev", "fake/1")
        specs = native_specs(meta)
        if "w" in meta.tensors:
            s = specs["w"]
            assert (s.dtype, s.shape, s.stride, s.layout, s.nbytes) == (
                DType.F16,
                [4, 4],
                [4, 1],
                {"tiles": 2},
                48,
            )
        else:
            assert specs == {}
        with safe_open(meta.src, framework="pt") as checkpoint:
            for name in checkpoint.keys():
                actual = checkpoint.get_tensor(name)
                if name == "w":
                    assert actual.numpy().tobytes() == payload
                else:
                    torch.testing.assert_close(actual, plain)


@pytest.fixture
def plain_meta(tmp_path, framework):
    paths = ParallelSaver(num_shards=1).save(
        {"plain": torch.ones(3)}, str(tmp_path / "plain")
    )
    return SafeTensorsMetadata.from_file(paths[0], framework)


def test_plain_checkpoint_has_no_native_specs(plain_meta):
    assert native_format(plain_meta) is None
    assert native_specs(plain_meta) == {}


@pytest.mark.parametrize(
    "metadata, message",
    [
        ({KEY_VERSION: "invalid"}, "bad fst.native.version"),
        ({KEY_VERSION: "99"}, "version 99"),
        ({KEY_VERSION: "1"}, "missing"),
        ({KEY_VERSION: "1", KEY_DEVICE: "x"}, "missing"),
        ({KEY_VERSION: "1", KEY_FORMAT: "y"}, "missing"),
    ],
)
def test_native_format_validation(plain_meta, metadata, message):
    plain_meta.metadata.update(metadata)
    with pytest.raises(NativeFormatError, match=message):
        native_format(plain_meta)


@pytest.mark.parametrize(
    "specs, message",
    [
        ("{", "bad fst.native.tensors"),
        (json.dumps({"ghost": {"dtype": "F16", "shape": [1]}}), "ghost"),
        (json.dumps({"plain": {"dtype": "unknown", "shape": [3]}}), "bad spec"),
        (json.dumps({"plain": {"dtype": "F16"}}), "bad spec"),
    ],
)
def test_native_spec_validation(plain_meta, specs, message):
    plain_meta.metadata.update(
        {KEY_VERSION: "1", KEY_DEVICE: "x", KEY_FORMAT: "y", KEY_TENSORS: specs}
    )
    with pytest.raises(NativeFormatError, match=message):
        native_specs(plain_meta)
