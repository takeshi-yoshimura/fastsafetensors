# SPDX-License-Identifier: Apache-2.0
"""Deferred tensor allocation and fallback, without an inference framework."""

import builtins
import sys
from types import ModuleType

import pytest
import torch

from fastsafetensors.copier import spyre


@pytest.fixture(autouse=True)
def no_inference_imports(monkeypatch):
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name.split(".")[0] in {"vllm", "spyre_inference"}:
            raise AssertionError(f"tensor transfer imported {name}")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)


@pytest.mark.parametrize(
    "tensor, dense",
    [
        (torch.empty(4, 8), True),
        (torch.empty(8, 4).t(), True),
        (torch.empty(4, 1, 8), True),
        (torch.empty(4, 8)[:, :4], False),
        (torch.empty(8).expand(4, 8), False),
        (torch.tensor(1.0), True),
    ],
)
def test_is_dense(tensor, dense):
    assert spyre.is_dense_tensor(tensor) == dense


def test_deferred_allocation_preserves_strides_and_source_storage(monkeypatch):
    def no_runtime(name, *args, **kwargs):
        if name.split(".")[0] == "torch_spyre":
            raise AssertionError("allocation must not import the Spyre runtime")
        return original_import(name, *args, **kwargs)

    original_import = builtins.__import__
    monkeypatch.setattr(builtins, "__import__", no_runtime)
    source = torch.nn.Parameter(
        torch.arange(12, dtype=torch.float16).reshape(3, 4).t(), requires_grad=False
    )
    expected = source.detach().clone()
    recorder = spyre.TransferRecorder(torch.device("meta"))
    destination = recorder(source)
    assert destination.device.type == "meta"
    assert destination.shape == source.shape and destination.stride() == source.stride()
    record = recorder.records[0]
    assert record.kind == "defer" and record.dst is destination
    assert record.src is not source
    assert record.src.data_ptr() == source.data_ptr()
    source.data = torch.zeros_like(source)
    torch.testing.assert_close(record.src, expected)


@pytest.mark.parametrize(
    "source",
    [
        torch.ones(3, dtype=torch.float32),
        torch.ones(3, dtype=torch.int32),
        torch.empty(0, dtype=torch.float16),
        torch.ones(3, dtype=torch.float16, requires_grad=True),
        torch.ones(3, 4, dtype=torch.float16)[:, :2],
        torch.ones(1, dtype=torch.float16).expand(4),
    ],
)
def test_unstageable_tensors_use_normal_to(source):
    recorder = spyre.TransferRecorder(torch.device("meta"))
    destination = recorder(source)
    assert destination.device.type == "meta"
    assert destination.shape == source.shape and destination.dtype == source.dtype
    assert recorder.records[0].kind == "to"
    assert not recorder.records[0].src.requires_grad


def test_repeat_destination_is_recorded_as_same():
    recorder = spyre.TransferRecorder(torch.device("meta"))
    destination = recorder(torch.ones(4, dtype=torch.float16))
    assert recorder(destination) is destination
    assert [record.kind for record in recorder.records] == ["defer", "same"]


def test_explicit_layout_allocation(monkeypatch):
    runtime = ModuleType("torch_spyre._C")
    calls = []

    def allocate(size, stride, dtype, layout, device):
        calls.append((size, stride, dtype, layout, device))
        return torch.empty_strided(size, stride, dtype=dtype, device=device)

    runtime.spyre_empty_with_layout = allocate
    monkeypatch.setitem(sys.modules, "torch_spyre._C", runtime)
    recorder = spyre.TransferRecorder(torch.device("meta"))
    source = torch.ones(4, 8, dtype=torch.float16).t()
    layout = object()
    destination = recorder.place_with_layout(source, layout)
    assert calls == [
        (source.size(), source.stride(), source.dtype, layout, recorder.device)
    ]
    record = recorder.records[0]
    assert (
        record.kind == "defer" and record.layout is layout and record.dst is destination
    )
    assert record.src.data_ptr() == source.data_ptr()


@pytest.mark.parametrize(
    "source_device, dtype, kind",
    [("cpu", torch.float32, "to"), ("meta", torch.float16, "same")],
)
def test_explicit_layout_normal_to(monkeypatch, source_device, dtype, kind):
    runtime = ModuleType("torch_spyre._C")
    runtime.spyre_empty_with_layout = lambda *args, **kwargs: pytest.fail(
        "must use .to()"
    )
    monkeypatch.setitem(sys.modules, "torch_spyre._C", runtime)
    source = torch.ones(4, dtype=dtype, device=source_device)
    destination = torch.empty_like(source, device="meta")
    calls = []

    def to(self, device, **kwargs):
        assert self is source
        calls.append((device, kwargs))
        return destination

    monkeypatch.setattr(torch.Tensor, "to", to)
    layout = object()
    recorder = spyre.TransferRecorder(torch.device("meta"))
    assert recorder.place_with_layout(source, layout) is destination
    assert calls == [(recorder.device, {"dtype": dtype, "device_layout": layout})]
    assert recorder.records[0].kind == kind and recorder.records[0].layout is layout


def test_fill_submits_only_deferred_and_counts_fallback(monkeypatch):
    recorder = spyre.TransferRecorder(torch.device("meta"))
    deferred = recorder(torch.ones(4, dtype=torch.float16))
    recorder(deferred)
    recorder(torch.ones(3, dtype=torch.float32))
    counts = []
    pipeline = type(
        "Pipeline", (), {"count": lambda self, *args: counts.append(args)}
    )()

    def stage(pairs, device):
        assert device == recorder.device
        assert len(pairs) == 1
        assert pairs[0][0] is recorder.records[0].src and pairs[0][1] is deferred
        return pipeline

    monkeypatch.setattr(spyre, "stage_to_spyre", stage)
    assert recorder.fill() is pipeline
    assert counts == [(".to()", 12)]


def test_fill_propagates_pipeline_failure(monkeypatch):
    recorder = spyre.TransferRecorder(torch.device("meta"))
    recorder(torch.ones(4, dtype=torch.float16))

    def fail(*args):
        raise RuntimeError("transfer failed")

    monkeypatch.setattr(spyre, "stage_to_spyre", fail)
    with pytest.raises(RuntimeError, match="transfer failed"):
        recorder.fill()
    assert len(recorder.records) == 1


@pytest.mark.parametrize("nbytes, kind", [(0, "to"), (8, "raw"), (17, "to")])
def test_checkpoint_entries_preserve_transfer_recipe(monkeypatch, nbytes, kind):
    from fastsafetensors.native import NativeFormatError

    runtime = ModuleType("torch_spyre._C")
    runtime.dma_bytes = lambda dst: nbytes
    runtime.get_spyre_tensor_layout = lambda dst: {"physical": 1}
    layouts = ModuleType("torch_spyre.layout")
    layouts.layout_to_json = lambda layout: layout
    monkeypatch.setitem(sys.modules, "torch_spyre._C", runtime)
    monkeypatch.setitem(sys.modules, "torch_spyre.layout", layouts)
    monkeypatch.setattr(spyre, "MAX_TRANSFER_BYTES", 16)
    source = torch.arange(4, dtype=torch.float16).reshape(2, 2).t()
    layout = {"placed": 1}
    recorder = spyre.TransferRecorder(torch.device("cpu"))
    recorder.records = [
        spyre.TransferRecord("defer", source, torch.empty_like(source), layout),
        spyre.TransferRecord("same", source, source),
        spyre.TransferRecord("to", source.float(), source.float()),
    ]
    records, entries = recorder.checkpoint_entries(prefix="weight")
    assert [record["kind"] for record in records] == [kind, "same", "to"]
    assert [entry.name for entry in entries] == ["weight0", "weight2"]
    assert records[0]["placed"] == layout
    assert records[0]["stride"] == [1, 2]
    assert records[1] == {
        "kind": "same",
        **spyre.tensor_metadata(source),
        "placed": None,
    }
    assert records[2] == {
        "kind": "to",
        "name": "weight2",
        "placed": None,
        "dense": True,
        **spyre.tensor_metadata(source.float()),
    }
    if kind == "raw":
        assert records[0]["layout"] == {"physical": 1}
        assert entries[0].nbytes == nbytes
    else:
        assert records[0]["dense"] is True
        assert "layout" not in records[0]
    recorder.records = [
        spyre.TransferRecord("to", torch.ones(2, dtype=torch.complex64), source)
    ]
    with pytest.raises(NativeFormatError, match="no safetensors equivalent"):
        recorder.checkpoint_entries()


def test_ordinary_checkpoint_records_need_no_spyre_runtime(monkeypatch):
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name.split(".")[0] == "torch_spyre":
            raise AssertionError(
                "ordinary transfer records must not import torch-spyre"
            )
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    source = torch.ones(2)
    recorder = spyre.TransferRecorder(torch.device("meta"))
    recorder(source)
    recorder(torch.ones(2, device="meta"))
    records, entries = recorder.checkpoint_entries()
    assert [record["kind"] for record in records] == ["to", "same"]
    assert len(entries) == 1
