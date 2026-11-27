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
    copy_tensor,
    get_device_dtype,
    get_elem_in_stick,
    get_spyre_tensor_layout,
    spyre_empty_with_layout,
)

from fastsafetensors import ParallelLoader, ParallelSaver, fastsafe_open
from fastsafetensors.common import SafeTensorsMetadata
from fastsafetensors.copier import spyre as transfer
from fastsafetensors.frameworks import get_framework_op
from fastsafetensors.frameworks._torch_spyre import layout_to_json
from fastsafetensors.native import NativeFormatError
from fastsafetensors.saver import plain_entry
from fastsafetensors.st_types import Device

DEVICE = torch.device("spyre")


def dma_bytes(tensor):
    return (
        get_framework_op("pytorch")
        .get_native_ops(Device.from_str("spyre:0"))
        .tensor_nbytes(tensor)
    )


def _rows_outermost(t):
    rows, width = t.shape
    stick = get_elem_in_stick(t.dtype)
    return SpyreTensorLayout(
        [rows, width // stick, stick],
        [width, stick, 1],
        get_device_dtype(t.dtype),
        {(0,): rows, (1, 2): width},
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
            try:
                nbytes = dma_bytes(dst)
                entry = transfer.native_entry(
                    name,
                    t,
                    dst,
                    nbytes=nbytes,
                    layout=layout_to_json(get_spyre_tensor_layout(dst)),
                )
            except (NativeFormatError, ValueError):
                # Unsupported views/layouts are separate portable entries;
                # this happens during fixture planning, before native DMA.
                entry = plain_entry(name, t)
                moved[name] = None if layout is None else layout_to_json(layout)
            entries.append(entry)

    native, portable = transfer.split_native_entries(entries)
    saver = ParallelSaver(
        num_shards=num_shards,
        fill=transfer.native_fill,
        shard_metadata=transfer.native_shard_metadata,
    )
    paths = saver.save_entries(native, base)
    return paths + ParallelSaver(num_shards=1, fill=transfer.native_fill).save_entries(
        portable,
        base + "-portable",
        metadata={"torch_spyre.moved": json.dumps(moved)},
    )


def test_round_trip(tmp_path):
    tensors = _tensors()
    layouts = _layouts(tensors)
    expected = _reference(tensors, layouts)
    paths = _write_checkpoint(
        tensors, str(tmp_path / "ckpt"), layouts=layouts, num_shards=4
    )
    assert len(paths) == 5

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
    from fastsafetensors.frameworks._torch_spyre import TorchSpyreOp

    generator = torch.Generator().manual_seed(42)
    tensors = {
        f"weight_{i}": torch.randn(128, width, generator=generator).half()
        for i, width in enumerate((256, 128, 256, 64))
    }
    expected = _reference(tensors, {})
    paths = _write_checkpoint(tensors, str(tmp_path / "reuse"), num_shards=4)
    allocated = []
    allocate = TorchSpyreOp.allocate_host_buffer

    def track_allocation(*args):
        buffer = allocate(*args)
        allocated.append(weakref.ref(buffer.get_raw()))
        return buffer

    monkeypatch.setattr(TorchSpyreOp, "allocate_host_buffer", track_allocation)
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


@pytest.mark.parametrize("queue_size", [-1, 0, 2])
def test_parallel_loader_chunks_native_shard(tmp_path, queue_size):
    tensors = _tensors()
    layouts = _layouts(tensors)
    expected = _reference(tensors, layouts)
    paths = _write_checkpoint(
        tensors, str(tmp_path / "chunked"), layouts=layouts, num_shards=1
    )
    metadata = SafeTensorsMetadata.from_file(paths[0], get_framework_op("pytorch"))
    largest = max(
        frame.data_offsets[1] - frame.data_offsets[0]
        for frame in metadata.tensors.values()
    )
    loader = ParallelLoader(
        None,
        paths,
        device="spyre",
        all_local=True,
        set_numa=False,
        max_batch_bytes=largest,
        queue_size=queue_size,
        use_tqdm_on_load=False,
    )
    try:
        assert len(loader.weight_files_batches) > 1
        got = dict(loader.iterate_weights())
    finally:
        loader.close()
    assert got.keys() == expected.keys()
    for name, tensor in got.items():
        torch.testing.assert_close(tensor.cpu(), expected[name], rtol=0, atol=0)


def test_mismatched_destination_fails_before_transfer(tmp_path, monkeypatch):
    tensors = _tensors()
    paths = _write_checkpoint(tensors, str(tmp_path / "ckpt"), num_shards=2)
    submitted = []
    monkeypatch.setattr(
        transfer.HostFileReader,
        "submit_read",
        lambda self, *args: submitted.append(args),
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
            if not f.keys():
                continue
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
