# Copyright 2026 The Torch-Spyre Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Spyre-native checkpoints round-trip byte-identically to copy_tensor."""

import gc
import json
import os
import weakref

import pytest

if os.environ.get("TEST_FASTSAFETENSORS_SPYRE") != "1":
    pytest.skip(
        "set TEST_FASTSAFETENSORS_SPYRE=1 to run on Spyre", allow_module_level=True
    )

import torch
import torch_spyre  # noqa: F401 -- registers the device
from safetensors import safe_open
from torch_spyre._C import (
    SpyreTensorLayout,
    convert_h2d_into,
    copy_tensor,
    dma_bytes,
    get_device_dtype,
    get_elem_in_stick,
    get_spyre_tensor_layout,
    spyre_empty_with_layout,
)
from torch_spyre.layout import layout_to_json

from fastsafetensors import ParallelLoader, ParallelSaver, fastsafe_open
from fastsafetensors.copier import spyre as transfer
from fastsafetensors.copier.spyre import NATIVE_FORMAT
from fastsafetensors.frameworks._torch import fst_dtype
from fastsafetensors.native import NativeFormatError
from fastsafetensors.saver import WriteEntry, plain_entry, tensor_fill
from fastsafetensors.st_types import DType

DEVICE = torch.device("spyre")


def _rows_outermost(t):
    rows, width = t.shape
    stick = get_elem_in_stick(t.dtype)
    return SpyreTensorLayout(
        [rows, width // stick, stick], [width, stick, 1], get_device_dtype(t.dtype)
    )


def _tensors():
    g = torch.Generator().manual_seed(0)
    half = torch.float16
    return {
        "linear": torch.randn(1024, 5376, generator=g).to(half),
        "norm": torch.randn(5376, generator=g).to(half),
        "odd": torch.randn(37, 100, generator=g).to(half),
        "transposed": torch.randn(256, 128, generator=g).to(half).t(),
        "table": torch.randn(512, 128, generator=g).to(half),
        "swapped": torch.randn(512, 1024, generator=g).to(half),
    }


def _layouts(tensors):
    s = tensors["swapped"]
    return {
        "table": _rows_outermost(tensors["table"]),
        "swapped": SpyreTensorLayout(list(s.shape), list(s.stride()), s.dtype, [1, 0]),
    }


def _reference(tensors, layouts):
    out = {}
    for name, t in tensors.items():
        if name in layouts:
            dst = spyre_empty_with_layout(
                t.size(), t.stride(), t.dtype, layouts[name], device=DEVICE
            )
        else:
            dst = torch.empty_strided(
                t.size(), t.stride(), dtype=t.dtype, device=DEVICE
            )
        copy_tensor(t, dst, non_blocking=False)
        out[name] = dst.cpu()
    return out


def _write_checkpoint(tensors, base, *, layouts=None, num_shards=2, plain=False):
    layouts = layouts or {}
    entries = []
    moved = {}
    specs = {}
    for name, t in tensors.items():
        layout = layouts.get(name)
        if plain:
            entries.append(plain_entry(name, t))
            moved[name] = None if layout is None else layout_to_json(layout)
        else:
            dst = (
                torch.empty_strided(t.size(), t.stride(), dtype=t.dtype, device=DEVICE)
                if layout is None
                else spyre_empty_with_layout(
                    t.size(), t.stride(), t.dtype, layout, device=DEVICE
                )
            )
            nbytes = dma_bytes(dst)
            specs[name] = {
                "dtype": fst_dtype(dst.dtype).value,
                "shape": list(dst.shape),
                "stride": list(dst.stride()),
                "layout": layout_to_json(get_spyre_tensor_layout(dst)),
            }
            entries.append(
                WriteEntry(
                    name,
                    DType.U8,
                    [nbytes],
                    nbytes,
                    source=(t, dst),
                )
            )

    def fill(work):
        tensor_fill()([(buf, e) for buf, e in work if e.name not in specs])
        native_work = [(buf, *e.source) for buf, e in work if e.name in specs]
        if native_work:
            convert_h2d_into(native_work)

    saver = ParallelSaver(
        num_shards=num_shards,
        fill=fill,
        shard_metadata=lambda shard: {
            "fst.native.version": str(NATIVE_FORMAT.version),
            "fst.native.device": NATIVE_FORMAT.device,
            "fst.native.format": NATIVE_FORMAT.format,
            "fst.native.tensors": json.dumps(
                {e.name: specs[e.name] for e in shard if e.name in specs}
            ),
        },
    )
    return saver.save_entries(
        entries,
        base,
        metadata={"torch_spyre.moved": json.dumps(moved)},
    )


def test_round_trip(tmp_path):
    tensors = _tensors()
    layouts = _layouts(tensors)
    expected = _reference(tensors, layouts)
    paths = _write_checkpoint(
        tensors, str(tmp_path / "ckpt"), layouts=layouts, num_shards=4
    )
    assert len(paths) == 4

    loader = ParallelLoader(None, paths, device="spyre", all_local=True, set_numa=False)
    try:
        got = dict(loader.iterate_weights())
    finally:
        loader.close()
    assert got.keys() == tensors.keys()
    for name, t in got.items():
        assert t.device.type == "spyre", name
        torch.testing.assert_close(t.cpu(), expected[name], rtol=0, atol=0, msg=name)


def test_reuses_host_buffers_across_shards(tmp_path, monkeypatch):
    from fastsafetensors.copier import spyre

    generator = torch.Generator().manual_seed(42)
    tensors = {
        f"weight_{i}": torch.randn(128, width, generator=generator).half()
        for i, width in enumerate((256, 128, 256, 64))
    }
    expected = _reference(tensors, {})
    paths = _write_checkpoint(tensors, str(tmp_path / "reuse"), num_shards=4)
    allocated = []
    allocate = spyre._allocate_host_buffer

    def track_allocation(*args):
        buffer = allocate(*args)
        allocated.append(weakref.ref(buffer))
        return buffer

    monkeypatch.setattr(spyre, "_allocate_host_buffer", track_allocation)
    loader = ParallelLoader(None, paths, device="spyre", all_local=True, set_numa=False)
    try:
        got = dict(loader.iterate_weights())
    finally:
        loader.close()
    assert 0 < len(allocated) < len(paths)
    assert got.keys() == expected.keys()
    for name, tensor in got.items():
        torch.testing.assert_close(tensor.cpu(), expected[name], rtol=0, atol=0)
    gc.collect()
    assert all(ref() is None for ref in allocated)


def test_into_destinations_and_by_name(tmp_path):
    tensors = _tensors()
    layouts = _layouts(tensors)
    expected = _reference(tensors, layouts)
    paths = _write_checkpoint(tensors, str(tmp_path / "ckpt"), layouts=layouts)
    t = tensors["table"]
    dst = spyre_empty_with_layout(
        t.size(), t.stride(), t.dtype, layouts["table"], device=DEVICE
    )

    loader = ParallelLoader(
        None,
        paths,
        device="spyre",
        destinations={"table": dst},
        tensor_filter=lambda name: name == "table",
        all_local=True,
        set_numa=False,
        accumulate_resident=False,
    )
    try:
        got = dict(loader.iterate_weights())
    finally:
        loader.close()
    assert list(got) == ["table"] and got["table"] is dst
    torch.testing.assert_close(dst.cpu(), expected["table"], rtol=0, atol=0)


def test_mismatched_destination_fails_before_transfer(tmp_path, monkeypatch):
    tensors = _tensors()
    paths = _write_checkpoint(tensors, str(tmp_path / "ckpt"), num_shards=2)
    submitted = []
    monkeypatch.setattr(
        transfer.TransferPipeline,
        "run",
        lambda self, jobs: submitted.append(jobs),
    )
    wrong = torch.empty(5, 5, dtype=torch.float16, device=DEVICE)
    loader = ParallelLoader(
        None,
        paths,
        device="spyre",
        destinations={"linear": wrong},
        tensor_filter=lambda name: name == "linear",
        all_local=True,
        set_numa=False,
    )
    try:
        with pytest.raises(NativeFormatError, match="does not match"):
            dict(loader.iterate_weights())
    finally:
        loader.close()
    assert not submitted


def test_plain_moved_tensors_with_recorded_layouts(tmp_path):
    tensors = _tensors()
    layouts = _layouts(tensors)
    expected = _reference(tensors, layouts)
    paths = _write_checkpoint(
        tensors, str(tmp_path / "ckpt"), layouts=layouts, num_shards=2, plain=True
    )
    for path in paths:
        with safe_open(path, framework="pt", device="cpu") as f:
            assert set(json.loads(f.metadata()["torch_spyre.moved"])) == set(tensors)
            for name in f.keys():
                assert torch.equal(f.get_tensor(name), tensors[name])

    loader = ParallelLoader(None, paths, device="spyre", all_local=True, set_numa=False)
    try:
        got = dict(loader.iterate_weights())
    finally:
        loader.close()
    for name, t in got.items():
        assert t.device.type == "spyre", name
        torch.testing.assert_close(t.cpu(), expected[name], rtol=0, atol=0, msg=name)


def test_spyre_copier_is_selected_through_fastsafetensors(tmp_path):
    tensors = {"w": torch.randn(64, 64).half()}
    expected = _reference(tensors, {})
    paths = _write_checkpoint(tensors, str(tmp_path / "ckpt"), num_shards=1)
    with fastsafe_open(paths, device="spyre") as f:
        torch.testing.assert_close(
            f.get_tensor("w").cpu(), expected["w"], rtol=0, atol=0
        )
