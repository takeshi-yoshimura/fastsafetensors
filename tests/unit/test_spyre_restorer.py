# SPDX-License-Identifier: Apache-2.0
"""Transfer replay and record matching without Spyre hardware or model traversal."""

import builtins
import copy
import sys
from types import ModuleType

import pytest
import torch

from fastsafetensors import ParallelSaver, SafeTensorsMetadata
from fastsafetensors.copier import spyre
from fastsafetensors.frameworks import get_framework_op
from fastsafetensors.native import NativeFormatError


@pytest.fixture(autouse=True)
def no_inference_imports(monkeypatch):
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name.split(".")[0] in {"vllm", "spyre_inference"}:
            raise AssertionError(f"tensor replay imported {name}")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)


def _save(tmp_path, recorder):
    records, entries = recorder.checkpoint_entries()
    paths = ParallelSaver(
        num_shards=2,
        fill=spyre.native_fill,
        shard_metadata=spyre.native_shard_metadata,
    ).save_entries(entries, str(tmp_path / "transfers"))
    framework = get_framework_op("pytorch")
    metas = [SafeTensorsMetadata.from_file(path, framework) for path in paths]
    headers = {name: meta for meta in metas for name in meta.tensors}
    return records, headers


@pytest.fixture
def plain(tmp_path):
    source = torch.arange(12.0).reshape(3, 4).t()
    recorder = spyre.TransferRecorder(torch.device("meta"))
    destination = recorder(source)
    recorder(destination)
    records, headers = _save(tmp_path, recorder)
    return source, records, headers


def test_plain_replay_recovers_strides_and_revisits_without_runtime(plain, monkeypatch):
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name.split(".")[0] == "torch_spyre":
            raise AssertionError("ordinary replay must not import torch-spyre")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    source, records, headers = plain
    restorer = spyre.TransferRestorer(torch.device("cpu"), headers, records)
    # Ordinary payloads restore the saved strides, not the meta placeholder's.
    destination = restorer(torch.empty(source.shape, device="meta"))
    assert restorer(destination) is destination
    assert restorer.finish() is None
    assert restorer.destinations == {}
    assert destination.stride() == source.stride()
    torch.testing.assert_close(destination, source)


@pytest.mark.parametrize(
    "field, value", [("size", [2, 6]), ("dtype", "float16"), ("placed", {"other": 1})]
)
def test_reject_tensor_record_mismatch_before_read(plain, monkeypatch, field, value):
    source, records, headers = plain
    records[0][field] = value
    monkeypatch.setattr(
        spyre, "read_recorded_tensor", lambda *args: pytest.fail("must validate first")
    )
    restorer = spyre.TransferRestorer(torch.device("cpu"), headers, records)
    with pytest.raises(NativeFormatError, match="does not match"):
        restorer(torch.empty_like(source, device="meta"))


def test_record_exhaustion_checked_by_backend(plain):
    source, records, headers = plain
    restorer = spyre.TransferRestorer(torch.device("cpu"), headers, records)
    destination = restorer(torch.empty_like(source, device="meta"))
    with pytest.raises(NativeFormatError, match="moved 1 tensors, recorded 2"):
        restorer.finish()
    restorer(destination)
    restorer.finish()
    with pytest.raises(NativeFormatError, match="more tensors"):
        restorer(destination)


@pytest.mark.parametrize("kind", ["unknown", "defer"])
def test_unknown_record_kind_is_not_treated_as_raw(plain, kind):
    source, records, headers = plain
    records[0]["kind"] = kind
    restorer = spyre.TransferRestorer(torch.device("cpu"), headers, records)
    with pytest.raises(NativeFormatError, match="unknown transfer kind"):
        restorer(torch.empty_like(source, device="meta"))


def test_revisit_requires_destination_device(plain):
    source, records, headers = plain
    restorer = spyre.TransferRestorer(torch.device("cpu"), headers, records[1:])
    with pytest.raises(NativeFormatError, match="was a revisit"):
        restorer(torch.empty_like(source, device="meta"))


@pytest.mark.parametrize("field, value", [("size", [2, 6]), ("dtype", "float16")])
def test_plain_payload_must_match_record(plain, field, value):
    _, records, headers = plain
    records[0][field] = value
    with pytest.raises(NativeFormatError, match="stored tensor does not match"):
        spyre.read_recorded_tensor(headers, records[0])


def test_non_dense_source_uses_contiguous_payload(tmp_path):
    source = torch.arange(24.0).reshape(4, 6)[:, :3]
    recorder = spyre.TransferRecorder(torch.device("meta"))
    recorder(source)
    records, headers = _save(tmp_path, recorder)
    assert not records[0]["dense"]
    restored = spyre.TransferRestorer(torch.device("cpu"), headers, records)
    tensor = restored(torch.empty_like(source, device="meta"))
    restored.finish()
    assert tensor.is_contiguous()
    torch.testing.assert_close(tensor, source)


@pytest.fixture
def raw(tmp_path, monkeypatch):
    runtime = ModuleType("torch_spyre._C")
    calls = []

    def allocate(size, stride, dtype, layout, device):
        calls.append((size, stride, dtype, layout, device))
        return torch.empty_strided(size, stride, dtype=dtype, device=device)

    def encode(work, num_threads):
        for buffer, source, destination in work:
            buffer[:] = (
                source.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
            )

    runtime.spyre_empty_with_layout = allocate
    runtime.dma_bytes = lambda tensor: tensor.nbytes
    runtime.get_spyre_tensor_layout = lambda tensor: {"physical": 1}
    runtime.convert_h2d_into = encode
    layouts = ModuleType("torch_spyre.layout")
    layouts.layout_to_json = lambda layout: layout
    layouts.layout_from_json = lambda layout: layout
    monkeypatch.setitem(sys.modules, "torch_spyre._C", runtime)
    monkeypatch.setitem(sys.modules, "torch_spyre.layout", layouts)
    source = torch.arange(12, dtype=torch.float16).reshape(3, 4).t()
    recorder = spyre.TransferRecorder(torch.device("cpu"))
    recorder.records = [
        spyre.TransferRecord("defer", source, torch.empty_like(source), {"placed": 1})
    ]
    records, headers = _save(tmp_path, recorder)
    return source, records, headers, runtime, calls


def test_raw_replay_allocates_without_filling(raw):
    source, records, headers, _, calls = raw
    restorer = spyre.TransferRestorer(torch.device("cpu"), headers, records)
    destination = restorer.place_with_layout(
        torch.empty_like(source, device="meta"), {"placed": 1}
    )
    restorer.finish()
    assert list(restorer.destinations) == ["t0"]
    assert restorer.destinations["t0"] is destination
    assert calls == [
        (
            list(source.shape),
            list(source.stride()),
            source.dtype,
            {"physical": 1},
            torch.device("cpu"),
        )
    ]


def test_raw_rejects_stride_mismatch_before_allocation(raw):
    source, records, headers, _, calls = raw
    restorer = spyre.TransferRestorer(torch.device("cpu"), headers, records)
    with pytest.raises(NativeFormatError, match="strides"):
        restorer.place_with_layout(
            torch.empty(source.shape, dtype=source.dtype, device="meta"), {"placed": 1}
        )
    assert not calls


@pytest.mark.parametrize(
    "field, value",
    [
        ("size", [2, 6]),
        ("stride", [3, 1]),
        ("dtype", "float32"),
        ("layout", {"physical": 2}),
    ],
)
def test_raw_payload_description_must_match_record(raw, field, value):
    source, records, headers, _, calls = raw
    records = copy.deepcopy(records)
    records[0][field] = value
    tensor = torch.empty_strided(
        records[0]["size"],
        records[0]["stride"],
        dtype=getattr(torch, records[0]["dtype"]),
        device="meta",
    )
    restorer = spyre.TransferRestorer(torch.device("cpu"), headers, records)
    with pytest.raises(NativeFormatError, match="native payload does not match"):
        restorer.place_with_layout(tensor, {"placed": 1})
    assert not calls


def test_raw_rejects_incompatible_format(raw):
    source, records, headers, _, calls = raw
    headers["t0"].metadata["fst.native.format"] = "other/1"
    restorer = spyre.TransferRestorer(torch.device("cpu"), headers, records)
    with pytest.raises(NativeFormatError, match="incompatible native format"):
        restorer.place_with_layout(
            torch.empty_like(source, device="meta"), {"placed": 1}
        )
    assert not calls


def test_raw_rejects_device_byte_size_mismatch(raw):
    source, records, headers, runtime, _ = raw
    runtime.dma_bytes = lambda tensor: tensor.nbytes + 128
    restorer = spyre.TransferRestorer(torch.device("cpu"), headers, records)
    with pytest.raises(NativeFormatError, match="device size differs"):
        restorer.place_with_layout(
            torch.empty_like(source, device="meta"), {"placed": 1}
        )
    assert not restorer.destinations


def test_native_payload_cannot_be_read_as_ordinary(raw):
    _, records, headers, _, _ = raw
    with pytest.raises(NativeFormatError, match="expected an ordinary tensor"):
        spyre.read_recorded_tensor(headers, records[0])


@pytest.mark.parametrize("kind", ["same", "to"])
def test_explicit_layout_normal_move_uses_recorded_placement(plain, monkeypatch, kind):
    source, records, headers = plain
    records[0].update(kind=kind, placed={"placed": 1})
    tensor = torch.empty_like(source, device="cpu" if kind == "same" else "meta")
    calls = []

    def to(self, device, **kwargs):
        calls.append((self, device, kwargs))
        return source

    layouts = ModuleType("torch_spyre.layout")
    layouts.layout_to_json = lambda layout: layout
    monkeypatch.setitem(sys.modules, "torch_spyre.layout", layouts)
    monkeypatch.setattr(torch.Tensor, "to", to)
    restorer = spyre.TransferRestorer(torch.device("cpu"), headers, records[:1])
    assert restorer.place_with_layout(tensor, {"placed": 1}) is source
    restorer.finish()
    assert len(calls) == 1
    assert calls[0][1:] == (
        torch.device("cpu"),
        {"dtype": source.dtype, "device_layout": {"placed": 1}},
    )
