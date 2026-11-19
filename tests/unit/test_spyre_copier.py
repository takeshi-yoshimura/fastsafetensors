# SPDX-License-Identifier: Apache-2.0
"""Spyre selection and device-native loader contracts, tested without hardware."""

import builtins
import json
import os
import subprocess
import sys
import textwrap
import weakref
from types import ModuleType, SimpleNamespace
from typing import Dict, List, Optional

import pytest

from fastsafetensors import SafeTensorsFileLoader, fastsafe_open
from fastsafetensors.common import SafeTensorsMetadata
from fastsafetensors.copier import registry
from fastsafetensors.copier.base import CopierInterface, DummyDeviceBuffer
from fastsafetensors.frameworks import TensorBase
from fastsafetensors.parallel_loader import ParallelLoader
from fastsafetensors.st_types import Device, DeviceType, DType


class _FakeNativeCopier(CopierInterface):
    device_native = True
    owns_tensor_storage = True
    selected: List[Optional[List[str]]] = []

    def __init__(self, metadata: SafeTensorsMetadata, device: Device, framework):
        self.metadata = metadata
        self.device = device
        self.framework = framework
        self.names: Optional[List[str]] = None

    def select_tensors(self, keep):
        self.names = [n for n in self.metadata.tensors if keep(n)]

    def submit_io(self, use_buf_register, max_copy_block_size):
        import torch

        from fastsafetensors.frameworks._torch import TorchTensor, dtype_convert

        names = self.names if self.names is not None else list(self.metadata.tensors)
        _FakeNativeCopier.selected.append(self.names)
        self.out: Dict[str, TensorBase] = {}
        with open(self.metadata.src, "rb") as f:
            for name in names:
                frame = self.metadata.tensors[name]
                begin, end = frame.data_offsets
                f.seek(self.metadata.header_length + begin)
                data = bytearray(f.read(end - begin))
                t = torch.frombuffer(data, dtype=dtype_convert[frame.dtype])
                self.out[name] = TorchTensor(
                    self.device, frame.dtype, t.reshape(frame.shape)
                )
        return DummyDeviceBuffer()

    def wait_io(self, gbuf, dtype=DType.AUTO, noalign=False, owner=None):
        return self.out


@pytest.fixture
def fake_spyre(monkeypatch, framework):
    if framework.get_name() != "pytorch":
        pytest.skip("Spyre uses the torch framework")
    import torch

    calls = []
    monkeypatch.setitem(sys.modules, "torch_spyre", ModuleType("torch_spyre"))
    monkeypatch.setattr(
        torch, "spyre", SimpleNamespace(set_device=calls.append), raising=False
    )

    def factory(device, **kwargs):
        def construct(metadata, device, framework):
            return _FakeNativeCopier(metadata, device, framework)

        return construct

    # Keep the real registration intact after the fixture exits.
    monkeypatch.setattr(registry, "_copier_registry", dict(registry._copier_registry))
    monkeypatch.setattr(
        registry, "_copier_class_registry", dict(registry._copier_class_registry)
    )
    registry.register_copier_constructor("spyre_native", _FakeNativeCopier)(factory)
    yield calls
    _FakeNativeCopier.selected.clear()


def test_spyre_device_parses_and_prints():
    dev = Device.from_str("spyre:1")
    assert dev == Device(DeviceType.SPYRE, 1)
    assert not dev.is_gpu
    assert dev.as_str() == "spyre:1"
    assert Device.from_str("Spyre").as_str() == "spyre"
    assert Device.from_str("cuda:0").is_gpu


def test_unknown_device_is_rejected():
    with pytest.raises(ValueError, match="Unknown device type: nosuchdev"):
        Device.from_str("nosuchdev:0")


def test_torch_spyre_device_operations(fake_spyre, framework, monkeypatch):
    import torch

    dev = framework.get_device("spyre:2", framework.get_process_group(None))
    assert dev == Device(DeviceType.SPYRE, 2)
    framework.set_device(dev)
    assert fake_spyre == [2]
    with pytest.raises(NotImplementedError, match="device-native copier"):
        framework.alloc_tensor_memory(16, dev)
    with pytest.raises(NotImplementedError, match="device-native copier"):
        framework.free_tensor_memory(DummyDeviceBuffer(), dev)

    sync_calls = []
    c = ModuleType("torch_spyre._C")
    c.synchronize = sync_calls.append
    monkeypatch.setitem(sys.modules, "torch_spyre._C", c)
    real_device = torch.device
    monkeypatch.setattr(
        torch,
        "device",
        lambda d: real_device("cpu") if d == "spyre:2" else real_device(d),
    )
    framework.synchronize(dev)
    assert sync_calls == [real_device("cpu")]


def test_cpu_loading_does_not_import_spyre(input_files):
    # A fresh interpreter catches imports during package initialization too.
    script = """
import builtins
import sys
original_import = builtins.__import__
def no_spyre(name, *args, **kwargs):
    if name == "torch_spyre" or name.startswith("torch_spyre."):
        raise AssertionError("CPU loading attempted to import torch-spyre")
    return original_import(name, *args, **kwargs)
builtins.__import__ = no_spyre
from fastsafetensors import fastsafe_open
from fastsafetensors.copier.registry import get_copier_class
assert get_copier_class("spyre_native").device_native
with fastsafe_open(sys.argv[1], device="cpu", nogds=True) as f:
    assert f.keys()
    assert f.get_tensor(f.keys()[0]).device.type == "cpu"
"""
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script), input_files[0]],
        env={
            **os.environ,
            "TORCH_DEVICE_BACKEND_AUTOLOAD": "0",
            "CUDA_VISIBLE_DEVICES": "",
        },
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("failure", ["absent", "broken_dependency", "runtime"])
def test_spyre_dependency_errors_are_reported(monkeypatch, framework, failure):
    if framework.get_name() != "pytorch":
        pytest.skip("Spyre uses the torch framework")
    original_import = builtins.__import__
    errors = {
        "absent": ModuleNotFoundError(
            "No module named 'torch_spyre'", name="torch_spyre"
        ),
        "broken_dependency": ModuleNotFoundError("No module named 'flex'", name="flex"),
        "runtime": RuntimeError("Spyre initialization failed"),
    }
    error = errors[failure]

    def fail_spyre(name, *args, **kwargs):
        if name == "torch_spyre":
            raise error
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fail_spyre)
    if failure == "absent":
        with pytest.raises(ImportError, match="requires torch-spyre") as exc:
            SafeTensorsFileLoader(None, "spyre", set_numa=False)
        assert exc.value.__cause__ is error
    else:
        with pytest.raises(type(error)) as exc:
            SafeTensorsFileLoader(None, "spyre", set_numa=False)
        assert exc.value is error


@pytest.mark.parametrize("copier_type", ["gds", "nogds", "fgds", "unified"])
def test_byte_copiers_reject_spyre(copier_type, framework):
    if framework.get_name() != "pytorch":
        pytest.skip("Spyre uses the torch framework")
    with pytest.raises(ValueError, match="cannot load onto Spyre"):
        SafeTensorsFileLoader(None, "spyre", copier_type=copier_type, set_numa=False)


def test_spyre_copier_rejects_cpu():
    with pytest.raises(ValueError, match="requires device='spyre'"):
        registry.create_copier_constructor("spyre_native", Device())


def test_spyre_copier_rejects_other_frameworks():
    with pytest.raises(ValueError, match="requires the pytorch framework"):
        registry.create_copier_constructor(
            "spyre_native",
            Device(DeviceType.SPYRE),
            framework=SimpleNamespace(get_name=lambda: "paddle"),
        )


def test_unknown_copier_is_rejected():
    with pytest.raises(KeyError, match="is not registered"):
        registry.create_copier_constructor("nosuchcopier", Device())


def test_device_native_copier_must_own_its_tensors():
    class Borrowing(CopierInterface):
        device_native = True

    with pytest.raises(ValueError, match="own their tensors"):
        registry.register_copier_constructor("borrowing", Borrowing)


# --- device-native copiers through the loaders --------------------------------------


def _expected(input_files, framework) -> Dict[str, "object"]:
    from safetensors.torch import load_file

    out: Dict[str, object] = {}
    for path in input_files:
        out.update(load_file(path))
    return out


def test_loader_uses_the_device_default_copier(fake_spyre, input_files, framework):
    if framework.get_name() != "pytorch":
        pytest.skip("fake copier builds torch tensors")
    import torch

    loader = SafeTensorsFileLoader(None, "spyre:0", set_numa=False)
    assert loader.copier_class is _FakeNativeCopier
    loader.add_filenames({0: input_files})
    fb = loader.copy_files_to_device()
    expected = _expected(input_files, framework)
    got = {k: fb.get_tensor(k) for k in loader.get_keys()}
    fb.close()
    loader.close()
    # Owned tensors stay valid after close().
    for name, t in got.items():
        assert torch.equal(t, expected[name]), name


def test_device_native_refuses_conversion_and_slicing(
    fake_spyre, input_files, framework
):
    if framework.get_name() != "pytorch":
        pytest.skip("fake copier builds torch tensors")
    loader = SafeTensorsFileLoader(None, "spyre:0", set_numa=False)
    loader.add_filenames({0: input_files})
    with pytest.raises(ValueError, match="dtype conversion"):
        loader.copy_files_to_device(dtype=DType.F32)
    fb = loader.copy_files_to_device()
    name = loader.get_keys()[0]
    with pytest.raises(ValueError, match="converted or moved"):
        fb.get_tensor(name, dtype=DType.F32)
    with pytest.raises(ValueError, match="sliced"):
        fb.get_multi_cols([name], dim=0)
    fb.close()
    loader.close()


def test_device_native_needs_a_single_process_group(fake_spyre, framework, monkeypatch):
    class _TwoRanks:
        def size(self):
            return 2

        def rank(self):
            return 0

    monkeypatch.setattr(
        type(framework), "get_process_group", lambda self, pg: _TwoRanks()
    )
    with pytest.raises(ValueError, match="single-process"):
        SafeTensorsFileLoader(None, "spyre:0", set_numa=False)


def test_tensor_filter_passes_exact_names(fake_spyre, input_files, framework):
    if framework.get_name() != "pytorch":
        pytest.skip("fake copier builds torch tensors")
    loader = SafeTensorsFileLoader(None, "spyre:0", set_numa=False)
    loader.add_filenames({0: input_files})
    keep = sorted(loader.get_keys())[:2]
    loader.set_tensor_filter(lambda n: n in keep)
    fb = loader.copy_files_to_device()
    assert sorted(n for s in _FakeNativeCopier.selected for n in (s or [])) == keep
    fb.close()
    loader.close()


def test_parallel_loader_yields_owned_tensors_without_clone(
    fake_spyre, input_files, framework
):
    if framework.get_name() != "pytorch":
        pytest.skip("fake copier builds torch tensors")
    import torch

    pl = ParallelLoader(
        None,
        input_files,
        device="spyre:0",
        copier_type="spyre_native",
        use_tqdm_on_load=False,
        set_numa=False,
    )
    assert pl.need_clone is False
    expected = _expected(input_files, framework)
    got = dict(pl.iterate_weights())
    pl.close()
    assert got.keys() == expected.keys()
    for name, t in got.items():
        assert torch.equal(t, expected[name]), name


def test_fastsafe_open_takes_a_copier_type(fake_spyre, input_files, framework):
    if framework.get_name() != "pytorch":
        pytest.skip("fake copier builds torch tensors")
    import torch

    expected = _expected(input_files, framework)
    with fastsafe_open(input_files, device="spyre:0", copier_type="spyre_native") as f:
        for name in f.keys():
            assert torch.equal(f.get_tensor(name), expected[name]), name


def test_extension_config_passes_custom_copiers_through():
    ext = SafeTensorsFileLoader.process_extension_config({"copier_type": "fake_native"})
    assert ext == {"copier_type": "fake_native"}
    assert SafeTensorsFileLoader.process_extension_config({"copier_type": "nogds"}) == {
        "nogds": True,
        "use_fgds": False,
    }


@pytest.fixture
def emulated_spyre(monkeypatch, framework):
    """Run the real copier with CPU allocations and a synchronous fake DMA.

    This checks file offsets, shard batching and destinations, not Spyre's
    physical layout, pinned memory or hardware transfer behavior.
    """
    if framework.get_name() != "pytorch":
        pytest.skip("Spyre uses the torch framework")
    import torch

    state = SimpleNamespace(
        batches=[], allocated=[], host_allocated=[], moves=[], layout={"test_layout": 1}
    )
    real_device = torch.device
    real_to = torch.Tensor.to

    def to(tensor, *args, **kwargs):
        if args and isinstance(args[0], real_device):
            state.moves.append(
                (tensor, kwargs.get("device_layout"), len(state.batches))
            )
            return tensor.clone()
        return real_to(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "to", to)
    monkeypatch.setattr(
        torch,
        "device",
        lambda d: real_device("cpu") if str(d).startswith("spyre") else real_device(d),
    )
    monkeypatch.setattr(
        torch, "spyre", SimpleNamespace(set_device=lambda index: None), raising=False
    )

    def allocate(shape, stride, dtype, layout, device):
        assert layout == state.layout
        tensor = torch.empty_strided(shape, stride, dtype=dtype, device=device)
        state.allocated.append(tensor)
        return tensor

    def copy_batch(entries):
        state.batches.append(list(entries))
        for buffer, dst in entries:
            dst.view(-1).view(torch.uint8).copy_(buffer[: dst.nbytes])

    c = ModuleType("torch_spyre._C")
    c.dma_bytes = lambda t: t.numel() * t.element_size()
    c.spyre_empty_with_layout = allocate
    c.get_spyre_tensor_layout = lambda t: state.layout
    c.synchronize = lambda device: None

    from fastsafetensors.copier import spyre

    real_allocate = spyre._allocate_host_buffer

    def allocate_host_buffer(nbytes):
        buffer = real_allocate(nbytes)
        state.host_allocated.append(weakref.ref(buffer))
        return buffer

    monkeypatch.setattr(spyre, "_allocate_host_buffer", allocate_host_buffer)
    c.copy_raw_h2d_batch = copy_batch

    def copy_tensor(src, dst, non_blocking):
        assert not non_blocking
        state.moves.append((src, state.layout, len(state.batches)))
        dst.copy_(src)

    c.copy_tensor = copy_tensor
    layouts = ModuleType("torch_spyre.layout")
    layouts.layout_from_json = lambda layout: layout
    layouts.layout_to_json = lambda layout: layout
    for name, module in {
        "torch_spyre": ModuleType("torch_spyre"),
        "torch_spyre._C": c,
        "torch_spyre.layout": layouts,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)
    return state


def _native_files(tmp_path, *, fmt=None, moved=None):
    import torch

    from fastsafetensors.native import NativeFormat
    from fastsafetensors.saver import WriteEntry, save_sharded

    tensors = {
        "a": torch.arange(8, dtype=torch.float32).reshape(2, 4),
        "b": torch.arange(4, dtype=torch.float32).reshape(2, 2),
        "plain": torch.arange(3, dtype=torch.float32),
    }
    if moved is not None:
        tensors["kept"] = torch.arange(2, dtype=torch.float32)
    entries = []
    specs = {}
    for name, tensor in tensors.items():
        nbytes = tensor.numel() * tensor.element_size()
        spec = (
            None
            if name in {"plain", "kept"}
            else {
                "dtype": DType.F32.value,
                "shape": list(tensor.shape),
                "stride": list(tensor.stride()),
                "layout": {"test_layout": 1},
            }
        )
        if spec is not None:
            specs[name] = spec
        entries.append(
            WriteEntry(
                name,
                DType.F32 if spec is None else DType.U8,
                list(tensor.shape) if spec is None else [nbytes],
                nbytes,
                source=tensor,
            )
        )
    fmt = fmt or NativeFormat("spyre", "torch_spyre/1")
    return (
        save_sharded(
            entries,
            str(tmp_path / "native"),
            num_shards=2,
            shard_metadata=lambda shard: {
                "fst.native.version": str(fmt.version),
                "fst.native.device": fmt.device,
                "fst.native.format": fmt.format,
                "fst.native.tensors": json.dumps(
                    {e.name: specs[e.name] for e in shard if e.name in specs}
                ),
            },
            metadata=(
                None if moved is None else {"torch_spyre.moved": json.dumps(moved)}
            ),
        ),
        tensors,
    )


@pytest.mark.parametrize("filtered", [False, True])
def test_real_copier_reads_shards_and_preserves_destinations(
    emulated_spyre, tmp_path, filtered
):
    import torch

    from fastsafetensors.copier.spyre import SpyreNativeCopier

    paths, expected = _native_files(tmp_path)
    destination = torch.empty_like(expected["b"])
    loader = SafeTensorsFileLoader(
        None, "spyre:0", destinations={"b": destination}, set_numa=False
    )
    assert loader.copier_class is SpyreNativeCopier
    loader.add_filenames({0: paths})
    if filtered:
        loader.set_tensor_filter(lambda name: name == "b")
        expected = {"b": expected["b"]}
    fb = loader.copy_files_to_device()
    got = {name: fb.get_tensor(name) for name in fb.key_to_rank_lidx}
    fb.close()
    loader.close()
    assert got.keys() == expected.keys()
    assert got["b"] is destination
    for name in expected:
        assert torch.equal(got[name], expected[name])
    assert len(emulated_spyre.batches) == 1
    assert len(emulated_spyre.batches[0]) == (1 if filtered else 2)
    assert len(emulated_spyre.allocated) == (0 if filtered else 1)


def test_backend_writer_round_trip_into_existing_destination(
    emulated_spyre, tmp_path, monkeypatch
):
    import torch

    from fastsafetensors.copier.spyre import (
        native_entry,
        native_fill,
        native_shard_metadata,
    )
    from fastsafetensors.saver import ParallelSaver, plain_entry

    def encode(work, num_threads):
        for buf, src, dst in work:
            assert len(buf) == dst.nbytes
            buf[:] = src.reshape(-1).view(torch.uint8).numpy().tobytes()

    monkeypatch.setattr(
        sys.modules["torch_spyre._C"], "convert_h2d_into", encode, raising=False
    )
    tensor = torch.arange(8, dtype=torch.float32).reshape(2, 4)
    plain = torch.arange(3, dtype=torch.float32)
    destination = torch.empty_like(tensor)
    paths = ParallelSaver(
        num_shards=2, fill=native_fill, shard_metadata=native_shard_metadata
    ).save_entries(
        [
            native_entry(
                "native",
                tensor,
                destination,
                nbytes=destination.nbytes,
                layout=emulated_spyre.layout,
            ),
            plain_entry("plain", plain),
        ],
        str(tmp_path / "writer"),
    )
    loader = ParallelLoader(
        None,
        paths,
        device="spyre:0",
        destinations={"native": destination},
        all_local=True,
        set_numa=False,
        use_tqdm_on_load=False,
    )
    try:
        got = dict(loader.iterate_weights())
    finally:
        loader.close()
    assert got["native"] is destination
    torch.testing.assert_close(destination, tensor, rtol=0, atol=0)
    torch.testing.assert_close(got["plain"], plain, rtol=0, atol=0)


@pytest.mark.parametrize("mismatch", ["shape", "dtype", "layout", "size", "format"])
def test_real_copier_rejects_incompatible_payload_before_dma(
    emulated_spyre, tmp_path, mismatch, monkeypatch
):
    import torch

    from fastsafetensors.native import NativeFormat, NativeFormatError

    fmt = NativeFormat("spyre", "unsupported/1") if mismatch == "format" else None
    paths, expected = _native_files(tmp_path, fmt=fmt)
    dst = torch.empty_like(expected["b"])
    if mismatch == "shape":
        dst = torch.empty(3, 3)
    elif mismatch == "dtype":
        dst = dst.half()
    elif mismatch == "layout":
        emulated_spyre.layout = {"test_layout": 2}
    elif mismatch == "size":
        monkeypatch.setattr(sys.modules["torch_spyre._C"], "dma_bytes", lambda t: 1)
    loader = SafeTensorsFileLoader(
        None, "spyre", destinations={"b": dst}, set_numa=False
    )
    try:
        loader.add_filenames({0: paths})
        loader.set_tensor_filter(lambda name: name == "b")
        with pytest.raises(NativeFormatError):
            loader.copy_files_to_device()
        assert not emulated_spyre.batches
    finally:
        loader.close()


@pytest.mark.parametrize("layout", [None, {"test_layout": 1}])
def test_fastsafe_open_restores_plain_moved_entries(emulated_spyre, tmp_path, layout):
    import torch

    paths, expected = _native_files(tmp_path, moved={"plain": layout})
    with fastsafe_open(paths, device="spyre:0", set_numa=False) as f:
        got = {name: f.get_tensor(name) for name in f.keys()}
        assert f.get_tensor_wrapped("plain").device.type == DeviceType.SPYRE
        assert f.get_tensor_wrapped("kept").device.type == DeviceType.CPU
    for name, want in expected.items():
        assert torch.equal(got[name], want)
    assert len(emulated_spyre.moves) == 1
    source, placed, completed_batches = emulated_spyre.moves[0]
    assert torch.equal(source, expected["plain"])
    assert placed == layout
    assert completed_batches == 1  # All native transfers finished first.


@pytest.mark.parametrize("selected", [{"plain"}, {"a"}, set()])
def test_loader_filters_before_restoring_moved_entries(
    emulated_spyre, tmp_path, selected
):
    import torch

    paths, expected = _native_files(tmp_path, moved={"plain": {"test_layout": 1}})
    loader = SafeTensorsFileLoader(None, "spyre:0", set_numa=False)
    try:
        loader.add_filenames({0: paths})
        loader.set_tensor_filter(selected.__contains__)
        loaded = loader.copy_files_to_device()
        try:
            assert set(loaded.key_to_rank_lidx) == selected
            for name in selected:
                assert torch.equal(loaded.get_tensor(name), expected[name])
        finally:
            loaded.close()
    finally:
        loader.close()
    assert len(emulated_spyre.moves) == int("plain" in selected)


@pytest.mark.parametrize("layout", [None, {"test_layout": 1}])
def test_moved_entries_fill_supplied_destinations(emulated_spyre, tmp_path, layout):
    import torch

    paths, expected = _native_files(tmp_path, moved={"plain": layout})
    destinations = {name: torch.empty_like(expected[name]) for name in ("a", "plain")}
    with fastsafe_open(
        paths, device="spyre:0", destinations=destinations, set_numa=False
    ) as f:
        for name, dst in destinations.items():
            assert f.get_tensor(name) is dst
            assert torch.equal(dst, expected[name])
    assert len(emulated_spyre.moves) == 1


@pytest.mark.parametrize("moved", [[], {"plain": 3}, {"a": None}])
def test_bad_moved_metadata_fails_before_dma(emulated_spyre, tmp_path, moved):
    from fastsafetensors.native import NativeFormatError

    paths, _ = _native_files(tmp_path, moved=moved)
    with pytest.raises(NativeFormatError, match="torch_spyre.moved"):
        fastsafe_open(paths, device="spyre:0", set_numa=False)
    assert not emulated_spyre.batches and not emulated_spyre.moves


def test_moved_destination_mismatch_fails_before_dma(emulated_spyre, tmp_path):
    import torch

    from fastsafetensors.native import NativeFormatError

    paths, _ = _native_files(tmp_path, moved={"plain": {"test_layout": 1}})
    with pytest.raises(NativeFormatError, match="destination does not match"):
        fastsafe_open(
            paths,
            device="spyre:0",
            destinations={"plain": torch.empty(4)},
            set_numa=False,
        )
    assert not emulated_spyre.batches and not emulated_spyre.moves


def test_native_validation_failure_closes_staged_shards(
    emulated_spyre, tmp_path, monkeypatch
):
    from fastsafetensors.native import NativeFormatError

    paths, _ = _native_files(tmp_path, moved={"plain": 3})
    real_open, real_close = os.open, os.close
    live = set()

    def tracked_open(path, *args, **kwargs):
        fd = real_open(path, *args, **kwargs)
        if str(path) in paths:
            live.add(fd)
        return fd

    def tracked_close(fd):
        real_close(fd)
        live.discard(fd)

    monkeypatch.setattr(os, "open", tracked_open)
    monkeypatch.setattr(os, "close", tracked_close)
    with pytest.raises(NativeFormatError, match="torch_spyre.moved"):
        fastsafe_open(paths, device="spyre:0", set_numa=False)
    assert not live
    assert not emulated_spyre.batches and not emulated_spyre.moves


@pytest.mark.parametrize("queue_size", [-1, 0, 2])
@pytest.mark.parametrize("selected", [{"a", "b", "plain", "kept"}, {"plain"}, set()])
def test_parallel_loader_restores_selected_tensors_into_destinations(
    emulated_spyre, tmp_path, selected, queue_size, monkeypatch
):
    import threading

    import torch

    paths, expected = _native_files(tmp_path, moved={"plain": {"test_layout": 1}})
    destinations = {
        name: torch.empty_like(expected[name]) for name in selected - {"kept"}
    }
    threads = []
    c = sys.modules["torch_spyre._C"]
    copy_batch = c.copy_raw_h2d_batch

    def record_batch(*args):
        threads.append(threading.get_ident())
        copy_batch(*args)

    monkeypatch.setattr(c, "copy_raw_h2d_batch", record_batch)
    loader = ParallelLoader(
        None,
        paths,
        device="spyre:0",
        destinations=destinations,
        tensor_filter=selected.__contains__,
        all_local=True,
        accumulate_resident=False,
        queue_size=queue_size,
        set_numa=False,
        use_tqdm_on_load=False,
    )
    try:
        assert not loader.need_clone
        got = dict(loader.iterate_weights())
    finally:
        loader.close()
    assert set(got) == selected
    for name, t in got.items():
        assert torch.equal(t, expected[name])
        if name in destinations:
            assert t is destinations[name]
    assert len(emulated_spyre.moves) == int("plain" in selected)
    assert not emulated_spyre.allocated
    if selected & {"a", "b"}:
        assert len(set(threads)) == 1
        assert threads[0] != threading.get_ident()
    else:
        assert not threads


def test_parallel_loader_reuses_host_buffers_across_shards_and_releases_on_close(
    emulated_spyre, tmp_path
):
    import gc

    import torch

    paths, expected = _native_files(tmp_path)
    loader = ParallelLoader(
        None,
        paths,
        device="spyre:0",
        copier_type="spyre_native",
        queue_size=-1,
        use_tqdm_on_load=False,
        set_numa=False,
    )
    got = dict(loader.iterate_weights())
    assert got.keys() == expected.keys()
    for name, tensor in got.items():
        assert torch.equal(tensor, expected[name]), name

    # The two native tensors are loaded from separate shard flushes. The
    # larger first-shard slot is reused as an exact-length view for the second.
    assert len(emulated_spyre.host_allocated) == 1
    assert all(ref() is not None for ref in emulated_spyre.host_allocated)
    emulated_spyre.batches.clear()
    loader.close()
    gc.collect()
    assert all(ref() is None for ref in emulated_spyre.host_allocated)


def test_parallel_loader_propagates_destination_mismatch(emulated_spyre, tmp_path):
    import torch

    from fastsafetensors.native import NativeFormatError

    paths, _ = _native_files(tmp_path)
    loader = ParallelLoader(
        None,
        paths,
        device="spyre:0",
        destinations={"b": torch.empty(3)},
        tensor_filter=lambda name: name == "b",
        all_local=True,
        set_numa=False,
        use_tqdm_on_load=False,
    )
    try:
        with pytest.raises(NativeFormatError, match="does not match"):
            list(loader.iterate_weights())
    finally:
        loader.close()
    assert not emulated_spyre.batches


def test_malformed_moved_json_is_rejected(emulated_spyre, tmp_path):
    import torch

    from fastsafetensors.native import NativeFormatError
    from fastsafetensors.saver import plain_entry, save_sharded

    paths = save_sharded(
        [plain_entry("w", torch.ones(2))],
        str(tmp_path / "bad"),
        num_shards=1,
        metadata={"torch_spyre.moved": "{invalid"},
    )
    with pytest.raises(NativeFormatError, match="torch_spyre.moved"):
        fastsafe_open(paths, device="spyre:0", set_numa=False)
    assert not emulated_spyre.batches and not emulated_spyre.moves
