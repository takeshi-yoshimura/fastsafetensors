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
from unittest.mock import Mock

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

    def set_chunk(self, byte_ranges, names, allocation_size=None):
        self.names = [n for n in self.metadata.tensors if n in names]

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
    monkeypatch.setattr(
        registry, "_copier_library_loaders", dict(registry._copier_library_loaders)
    )
    monkeypatch.setattr(
        registry, "_copier_init_checks", dict(registry._copier_init_checks)
    )
    registry.register_copier_constructor("spyre", _FakeNativeCopier)(factory)
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
assert get_copier_class("spyre").device_native
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


@pytest.mark.parametrize("copier_type", ["gds", "nogds", "fgds", "unified", "dstorage"])
def test_byte_copiers_reject_spyre(copier_type, framework, monkeypatch):
    if framework.get_name() != "pytorch":
        pytest.skip("Spyre uses the torch framework")
    initialize = Mock(side_effect=AssertionError("must reject before initialization"))
    monkeypatch.setattr(registry, "load_and_init_and_check_copier", initialize)
    with pytest.raises(ValueError, match="cannot load onto Spyre"):
        SafeTensorsFileLoader(None, "spyre", copier_type=copier_type, set_numa=False)
    initialize.assert_not_called()


def test_spyre_copier_rejects_cpu():
    with pytest.raises(ValueError, match="requires device='spyre'"):
        registry.select_copier("spyre", Device())


def test_spyre_copier_rejects_other_frameworks():
    with pytest.raises(ValueError, match="requires the pytorch framework"):
        registry.select_copier(
            "spyre",
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
    with pytest.raises(ValueError, match="cannot be sliced"):
        loader.set_slice(loader.get_keys()[0], slice(0, 1))
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
        copier_type="spyre",
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
    with fastsafe_open(input_files, device="spyre:0", copier_type="spyre") as f:
        for name in f.keys():
            assert torch.equal(f.get_tensor(name), expected[name]), name


def test_extension_config_passes_custom_copiers_through():
    ext = SafeTensorsFileLoader.process_extension_config({"copier_type": "fake_native"})
    assert ext == {"copier_type": "fake_native"}
    assert SafeTensorsFileLoader.process_extension_config({"copier_type": "nogds"}) == {
        "copier_type": "nogds",
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
        tensor = torch.empty_strided(shape, stride, dtype=dtype, device=device)
        state.allocated.append(tensor)
        return tensor

    def copy_batch(entries):
        state.batches.append(list(entries))
        for buffer, dst, offset, size in entries:
            count = max(0, min(size, dst.nbytes - offset))
            dst.view(-1).view(torch.uint8)[offset : offset + count].copy_(
                buffer[:count]
            )

    c = ModuleType("torch_spyre._C")
    from fastsafetensors.frameworks._torch import fst_dtype
    from fastsafetensors.native import canonical_layout

    c.dma_bytes = lambda t: canonical_layout(list(t.shape), fst_dtype(t.dtype))[1]
    c.spyre_empty_with_layout = allocate
    c.get_spyre_tensor_layout = lambda t: (
        state.layout
        if t.numel() == 3
        else (
            canonical_layout(list(t.shape), fst_dtype(t.dtype))[0]
            if state.layout == {"test_layout": 1}
            else state.layout
        )
    )
    c.synchronize = lambda device: None

    from fastsafetensors.frameworks._torch_spyre import TorchSpyreOp

    monkeypatch.setattr(TorchSpyreOp, "load_library", lambda self: None)
    monkeypatch.setattr("fastsafetensors.frameworks._torch_spyre.backend", lambda: c)
    monkeypatch.setattr(
        TorchSpyreOp, "tensor_nbytes", lambda self, t: c.dma_bytes(self.unwrap(t))
    )

    def emulate_copy(self, entries):
        c.copy_raw_h2d_batch(
            [(self.unwrap(a), self.unwrap(b), *rest) for a, b, *rest in entries]
        )

    monkeypatch.setattr(TorchSpyreOp, "copy_raw_tensors", emulate_copy)

    real_allocate = TorchSpyreOp.allocate_host_buffer

    def allocate_host_buffer(self, nbytes):
        buffer = real_allocate(self, nbytes)
        state.host_allocated.append(weakref.ref(buffer.get_raw()))
        return buffer

    monkeypatch.setattr(TorchSpyreOp, "allocate_host_buffer", allocate_host_buffer)
    c.copy_raw_h2d_batch = copy_batch

    def copy_tensor(src, dst, non_blocking):
        assert not non_blocking
        state.moves.append((src, state.layout, len(state.batches)))
        dst.copy_(src)

    c.copy_tensor = copy_tensor
    monkeypatch.setattr(TorchSpyreOp, "layout_from_json", lambda self, layout: layout)
    monkeypatch.setattr(TorchSpyreOp, "layout_to_json", lambda self, layout: layout)
    for name, module in {
        "torch_spyre": ModuleType("torch_spyre"),
        "torch_spyre._C": c,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)
    return state


def _native_files(tmp_path, *, fmt=None, moved=None, num_shards=2, extra_native=False):
    import torch

    from fastsafetensors.copier.spyre import NATIVE_FORMAT
    from fastsafetensors.native import (
        PROFILE,
        NativeTensorSpec,
        canonical_layout,
        native_metadata,
    )
    from fastsafetensors.saver import WriteEntry, save_sharded, tensor_fill

    tensors = {
        "a": torch.arange(8, dtype=torch.float32).reshape(2, 4),
        "b": torch.arange(4, dtype=torch.float32).reshape(2, 2),
        "plain": torch.arange(3, dtype=torch.float32),
    }
    if extra_native:
        tensors["c"] = torch.arange(8, dtype=torch.float32).reshape(2, 4) + 10
    if moved is not None:
        tensors["kept"] = torch.arange(2, dtype=torch.float32)
    entries = []
    specs = {}
    for name, tensor in tensors.items():
        nbytes = tensor.numel() * tensor.element_size()
        spec = None
        if name not in {"plain", "kept"}:
            layout, nbytes = canonical_layout(list(tensor.shape), DType.F32)
            spec = NativeTensorSpec(
                name,
                DType.F32,
                list(tensor.shape),
                list(tensor.stride()),
                layout,
                nbytes,
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

    def fill(work):
        for buffer, entry in work:
            if entry.name in specs:
                buffer[:] = bytes(entry.nbytes)
                data = entry.source.numpy().tobytes()
                buffer[: len(data)] = data
            else:
                tensor_fill()([(buffer, entry)])

    def shard_metadata(shard):
        md = native_metadata(NATIVE_FORMAT, {e.name: specs[e.name] for e in shard})
        if fmt is not None:
            md["profile"] = fmt.format
        return md

    paths = save_sharded(
        [e for e in entries if e.name in specs],
        str(tmp_path / "native"),
        num_shards=num_shards,
        shard_metadata=shard_metadata,
        fill=fill,
    )
    paths += save_sharded(
        [e for e in entries if e.name not in specs],
        str(tmp_path / "portable"),
        num_shards=1,
        metadata=None if moved is None else {"torch_spyre.moved": json.dumps(moved)},
    )
    return paths, tensors


@pytest.mark.parametrize("filtered", [False, True])
def test_real_copier_reads_shards_and_preserves_destinations(
    emulated_spyre, tmp_path, filtered
):
    import torch

    from fastsafetensors.copier.spyre import SpyreCopier

    paths, expected = _native_files(tmp_path)
    destination = torch.empty_like(expected["b"])
    loader = SafeTensorsFileLoader(
        None, "spyre:0", destinations={"b": destination}, set_numa=False
    )
    assert loader.copier_class is SpyreCopier
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
    assert sum(map(len, emulated_spyre.batches)) == (1 if filtered else 2)
    assert all(len(batch) == 1 for batch in emulated_spyre.batches)
    assert len(emulated_spyre.allocated) == (0 if filtered else 1)


@pytest.mark.parametrize("chunked", [False, True])
@pytest.mark.parametrize("selected", [{"a", "c"}, set()])
def test_loader_selection_uses_exact_names_not_coalesced_ranges(
    emulated_spyre, tmp_path, monkeypatch, chunked, selected
):
    import torch

    from fastsafetensors.copier import host
    from fastsafetensors.copier.spyre import SpyreCopier

    paths, expected = _native_files(tmp_path, num_shards=1, extra_native=True)
    loader = SafeTensorsFileLoader(None, "spyre:0", set_numa=False)
    loader.add_filenames({0: paths})
    plans, reads = {}, []
    set_chunk = SpyreCopier.set_chunk
    pread = host.pread_into

    def record_plan(self, ranges, names, allocation_size=None):
        plans[self.metadata.src] = (ranges, set(names))
        set_chunk(self, ranges, names, allocation_size)

    def record_read(fd, buffer, offset):
        reads.append((offset, len(buffer)))
        pread(fd, buffer, offset)

    monkeypatch.setattr(SpyreCopier, "set_chunk", record_plan)
    monkeypatch.setattr(host, "pread_into", record_read)
    if chunked:
        loader._set_chunk_plan(
            {
                path: (
                    selected & meta.tensors.keys(),
                    meta.select_byte_ranges(selected.__contains__),
                    None,
                )
                for path, (meta, _) in loader.meta.items()
            }
        )
    else:
        loader.set_tensor_filter(selected.__contains__)
    try:
        loaded = loader.copy_files_to_device()
        try:
            assert set(loaded.key_to_rank_lidx) == selected
            for name in selected:
                assert torch.equal(loaded.get_tensor(name), expected[name])
        finally:
            loaded.close()
        assert len(plans) == len(paths)
        for path, (meta, _) in loader.meta.items():
            ranges, names = plans[path]
            assert names == selected & meta.tensors.keys()
            if selected and "b" in meta.tensors:
                start, end = (
                    meta.header_length + offset
                    for offset in meta.tensors["b"].data_offsets
                )
                assert any(lo <= start < end <= hi for lo, hi in ranges)
        assert len(emulated_spyre.allocated) == len(selected)
        assert sum(map(len, emulated_spyre.batches)) == len(selected)
        assert sum(size for _, size in reads) == 256 * len(selected)
        if not selected:
            assert not emulated_spyre.host_allocated
    finally:
        loader.close()


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

    def encode(work):
        for buf, src, shape, dtype in work:
            assert shape == [2, 4] and dtype == "F32"
            buf[:] = bytes(len(buf))
            data = bytes(src)
            buf[: len(data)] = data

    monkeypatch.setattr(
        sys.modules["torch_spyre._C"], "convert_buffers", encode, raising=False
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
                nbytes=256,
                layout=__import__(
                    "fastsafetensors.native", fromlist=["canonical_layout"]
                ).canonical_layout([2, 4], DType.F32)[0],
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
    with fastsafe_open(paths, device="spyre:0") as f:
        got = {name: f.get_tensor(name) for name in f.keys()}
        assert f.get_tensor_wrapped("plain").device.type == DeviceType.SPYRE
        assert f.get_tensor_wrapped("kept").device.type == DeviceType.CPU
    for name, want in expected.items():
        assert torch.equal(got[name], want)
    assert len(emulated_spyre.moves) == 1
    source, placed, completed_batches = emulated_spyre.moves[0]
    assert torch.equal(source, expected["plain"])
    assert placed == layout
    # Native DMA for this file completes before its ordinary moved entries.
    assert completed_batches >= 1


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
    loader = SafeTensorsFileLoader(
        None, "spyre:0", destinations=destinations, set_numa=False
    )
    loader.add_filenames({0: paths})
    f = loader.copy_files_to_device()
    try:
        for name, dst in destinations.items():
            assert f.get_tensor(name) is dst
            assert torch.equal(dst, expected[name])
    finally:
        f.close()
        loader.close()
    assert len(emulated_spyre.moves) == 1


@pytest.mark.parametrize("moved", [[], {"plain": 3}])
def test_bad_moved_metadata_fails_before_dma(emulated_spyre, tmp_path, moved):
    from fastsafetensors.native import NativeFormatError

    paths, _ = _native_files(tmp_path, moved=moved, num_shards=1)
    with pytest.raises(NativeFormatError, match="torch_spyre.moved"):
        fastsafe_open(paths, device="spyre:0")
    assert not emulated_spyre.batches and not emulated_spyre.moves


@pytest.mark.parametrize("target", ["plain", "b"])
def test_destination_mismatch_precedes_every_file_submission(
    emulated_spyre, tmp_path, monkeypatch, target
):
    import torch

    from fastsafetensors.copier.spyre import SpyreCopier
    from fastsafetensors.native import NativeFormatError

    paths, _ = _native_files(
        tmp_path, moved={"plain": {"test_layout": 1}}, num_shards=1
    )
    loader = SafeTensorsFileLoader(
        None, "spyre:0", destinations={target: torch.empty(4)}, set_numa=False
    )
    loader.add_filenames({0: paths})

    def forbidden(*args, **kwargs):
        pytest.fail("a file was submitted before all destinations were validated")

    monkeypatch.setattr(SpyreCopier, "submit_io", forbidden)
    try:
        with pytest.raises(NativeFormatError, match="destination does not match"):
            loader.copy_files_to_device()
    finally:
        loader.close()
    assert not emulated_spyre.batches and not emulated_spyre.moves
    assert not emulated_spyre.allocated and not emulated_spyre.host_allocated


@pytest.mark.parametrize("chunked", [False, True])
def test_destination_preflight_only_checks_selected_tensors(
    emulated_spyre, tmp_path, chunked
):
    import torch

    paths, expected = _native_files(
        tmp_path, moved={"plain": {"test_layout": 1}}, num_shards=1
    )
    loader = SafeTensorsFileLoader(
        None,
        "spyre:0",
        destinations={"b": torch.empty(4), "plain": torch.empty(4)},
        set_numa=False,
    )
    loader.add_filenames({0: paths})
    if chunked:
        loader._set_chunk_plan(
            {
                path: (
                    {"a"} & meta.tensors.keys(),
                    meta.select_byte_ranges(lambda name: name == "a"),
                    None,
                )
                for path, (meta, _) in loader.meta.items()
            }
        )
    else:
        loader.set_tensor_filter(lambda name: name == "a")
    try:
        fb = loader.copy_files_to_device()
        try:
            assert set(fb.key_to_rank_lidx) == {"a"}
            assert torch.equal(fb.get_tensor("a"), expected["a"])
        finally:
            fb.close()
    finally:
        loader.close()
    assert not emulated_spyre.moves


@pytest.mark.parametrize("mode", ["full", "filtered", "chunked"])
def test_spyre_finalizes_every_plan_once_before_submission(
    emulated_spyre, tmp_path, monkeypatch, mode
):
    import torch

    from fastsafetensors.copier import host, spyre

    paths, expected = _native_files(
        tmp_path, moved={"plain": {"test_layout": 1}}, num_shards=1
    )
    selected = set(expected) if mode == "full" else {"a", "plain"}
    loader = SafeTensorsFileLoader(
        None,
        "spyre:0",
        destinations={name: torch.empty_like(expected[name]) for name in selected},
        set_numa=False,
    )
    loader.add_filenames({0: paths})
    planned = {}
    set_chunk, submit = spyre.SpyreCopier.set_chunk, spyre.SpyreCopier.submit_io

    def forbidden(*args, **kwargs):
        pytest.fail("planning must not do IO; submission must reuse the checked plan")

    def record_plan(self, ranges, names, allocation_size=None):
        assert self.metadata.src not in planned
        with monkeypatch.context() as patch:
            patch.setattr(spyre, "pread_into", forbidden)
            patch.setattr(host, "pread_into", forbidden)
            patch.setattr(self.ops, "empty_tensor", forbidden)
            patch.setattr(self.ops, "allocate_host_buffer", forbidden)
            patch.setattr(self.reader, "submit_read", forbidden)
            set_chunk(self, ranges, names, allocation_size)
        planned[self.metadata.src] = list(self.plan[1])
        assert not emulated_spyre.batches and not emulated_spyre.moves
        assert not emulated_spyre.allocated and not emulated_spyre.host_allocated

    def record_submit(self, *args):
        assert planned.keys() == loader.meta.keys()
        with monkeypatch.context() as patch:
            patch.setattr(spyre, "native_specs", forbidden)
            patch.setattr(self.ops, "layout_from_json", forbidden)
            patch.setattr(self.ops, "validate_destination", forbidden)
            patch.setattr(self.ops, "validate_plain_destination", forbidden)
            return submit(self, *args)

    monkeypatch.setattr(spyre.SpyreCopier, "set_chunk", record_plan)
    monkeypatch.setattr(spyre.SpyreCopier, "submit_io", record_submit)
    if mode == "filtered":
        loader.set_tensor_filter(selected.__contains__)
    elif mode == "chunked":
        loader._set_chunk_plan(
            {
                path: (
                    selected & meta.tensors.keys(),
                    meta.select_byte_ranges(selected.__contains__),
                    None,
                )
                for path, (meta, _) in loader.meta.items()
            }
        )
    try:
        loaded = loader.copy_files_to_device()
        try:
            assert set(loaded.key_to_rank_lidx) == selected
            for name in selected:
                assert torch.equal(loaded.get_tensor(name), expected[name])
            for path, (meta, _) in loader.meta.items():
                assert planned[path] == [n for n in meta.tensors if n in selected]
        finally:
            loaded.close()
    finally:
        loader.close()


def test_spyre_submission_requires_a_successful_plan(emulated_spyre, tmp_path):
    paths, _ = _native_files(tmp_path, num_shards=1)
    loader = SafeTensorsFileLoader(None, "spyre:0", set_numa=False)
    metadata = SafeTensorsMetadata.from_file(paths[0], loader.framework)
    copier = loader.copier_constructor(metadata, loader.device, loader.framework)
    try:
        with pytest.raises(RuntimeError, match="set_chunk must finalize"):
            copier.submit_io(False, 1024)
        copier.set_chunk([], set())
        with pytest.raises(ValueError, match="unknown chunk"):
            copier.set_chunk([], {"unknown"})
        with pytest.raises(RuntimeError, match="set_chunk must finalize"):
            copier.submit_io(False, 1024)
        assert copier.fd is None and copier.request is None
        assert not emulated_spyre.batches and not emulated_spyre.allocated
        assert not emulated_spyre.host_allocated
    finally:
        copier.finish_io()
        loader.close()


def test_native_validation_failure_closes_staged_shards(
    emulated_spyre, tmp_path, monkeypatch
):
    from fastsafetensors.native import NativeFormatError

    paths, _ = _native_files(tmp_path, moved={"plain": 3}, num_shards=1)
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
        fastsafe_open(paths, device="spyre:0")
    assert not live
    assert not emulated_spyre.batches and not emulated_spyre.moves


def test_later_file_submission_failure_drains_already_submitted_file(
    emulated_spyre, tmp_path, monkeypatch
):
    import torch

    from fastsafetensors.copier.spyre import SpyreCopier
    from fastsafetensors.native import NativeFormatError

    paths, _ = _native_files(tmp_path, moved={"plain": {"test_layout": 1}})
    loader = SafeTensorsFileLoader(None, "spyre:0", set_numa=False)
    loader.add_filenames({0: paths})
    copiers = []
    submit = SpyreCopier.submit_io

    def record(self, *args):
        copiers.append(self)
        if len(copiers) == 3:
            raise NativeFormatError("injected later-file submission failure")
        result = submit(self, *args)
        # Force earlier DMA to finish before a later resource/submission error.
        assert self.request.done.wait(5)
        return result

    monkeypatch.setattr(SpyreCopier, "submit_io", record)
    try:
        with pytest.raises(NativeFormatError, match="injected later-file"):
            loader.copy_files_to_device()
        assert emulated_spyre.batches
        assert len(copiers) == 3
        assert all(c.fd is None and c.request is None for c in copiers)
        assert not emulated_spyre.moves
    finally:
        loader.close()


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
        copier_type="spyre",
        queue_size=-1,
        use_tqdm_on_load=False,
        set_numa=False,
    )
    got = dict(loader.iterate_weights())
    assert got.keys() == expected.keys()
    for name, tensor in got.items():
        assert torch.equal(tensor, expected[name]), name

    # Both fixed slots are retained across separate shard flushes. Copies use
    # exact-length views without allocating tensor-sized staging buffers.
    assert len(emulated_spyre.host_allocated) == 2
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
    loader = SafeTensorsFileLoader(None, "spyre:0", set_numa=False)
    loader.add_filenames({0: paths})
    try:
        with pytest.raises(NativeFormatError, match="torch_spyre.moved"):
            loader.copy_files_to_device()
    finally:
        loader.close()
    assert not emulated_spyre.batches and not emulated_spyre.moves


def test_moved_json_uses_last_duplicate_value(emulated_spyre, tmp_path):
    import torch

    from fastsafetensors.saver import plain_entry, save_sharded

    expected = torch.arange(3, dtype=torch.float32)
    paths = save_sharded(
        [plain_entry("plain", expected)],
        str(tmp_path / "moved"),
        num_shards=1,
        metadata={"torch_spyre.moved": '{"plain":3,"plain":null}'},
    )
    with fastsafe_open(paths, device="spyre:0") as loaded:
        assert torch.equal(loaded.get_tensor("plain"), expected)
        assert loaded.get_tensor_wrapped("plain").device.type == DeviceType.SPYRE
    assert len(emulated_spyre.moves) == 1 and not emulated_spyre.batches


@pytest.mark.parametrize("queue_size", [-1, 0, 2])
@pytest.mark.parametrize("destinations", [False, True])
@pytest.mark.parametrize("filtered", [False, True])
def test_parallel_loader_owns_spyre_chunk_planning(
    emulated_spyre, tmp_path, queue_size, destinations, filtered, monkeypatch
):
    import torch

    from fastsafetensors.copier.spyre import SpyreCopier

    paths, expected = _native_files(
        tmp_path, moved={"plain": {"test_layout": 1}}, num_shards=1
    )
    selected = {"a", "plain"} if filtered else set(expected)
    supplied = (
        {name: torch.empty_like(expected[name]) for name in selected - {"kept"}}
        if destinations
        else {}
    )
    chunks = []
    allocated_per_chunk = []
    set_chunk = SpyreCopier.set_chunk
    submit = SpyreCopier.submit_io

    def record_chunk(self, ranges, names, allocation_size=None):
        chunks.append(set(names))
        set_chunk(self, ranges, names, allocation_size)

    def record_submit(self, *args):
        before = len(emulated_spyre.allocated)
        result = submit(self, *args)
        allocated_per_chunk.append(len(emulated_spyre.allocated) - before)
        return result

    monkeypatch.setattr(SpyreCopier, "set_chunk", record_chunk)
    monkeypatch.setattr(SpyreCopier, "submit_io", record_submit)
    loader = ParallelLoader(
        None,
        paths,
        device="spyre:0",
        all_local=True,
        destinations=supplied,
        tensor_filter=selected.__contains__,
        max_batch_bytes=256,
        queue_size=queue_size,
        use_chunk_budget_as_allocation_size=True,
        use_tqdm_on_load=False,
        set_numa=False,
    )
    try:
        got = dict(loader.iterate_weights())
    finally:
        loader.close()
    assert len(chunks) > 1
    assert set.union(*chunks) == selected
    assert sum(map(len, chunks)) == len(selected)
    assert set(got) == selected
    assert all(n <= 1 for n in allocated_per_chunk)
    for name in selected:
        assert torch.equal(got[name], expected[name])
        if name in supplied:
            assert got[name] is supplied[name]
    assert sum(map(len, emulated_spyre.batches)) == len(selected & {"a", "b"})


def test_spyre_chunk_validation_and_empty_chunk(emulated_spyre, tmp_path):
    paths, _ = _native_files(tmp_path, num_shards=1)
    loader = SafeTensorsFileLoader(None, "spyre:0", set_numa=False)
    metadata = SafeTensorsMetadata.from_file(paths[0], loader.framework)
    copier = loader.copier_constructor(metadata, loader.device, loader.framework)
    start = metadata.header_length + metadata.tensors["a"].data_offsets[0]
    with pytest.raises(ValueError, match="whole tensor"):
        copier.set_chunk([(start, start + 1)], {"a"})
    with pytest.raises(ValueError, match="unknown chunk"):
        copier.set_chunk([], {"unknown"})
    copier.set_chunk([], set())
    gbuf = copier.submit_io(False, 1024)
    assert copier.wait_io(gbuf) == {}
    copier.finish_io()
    loader.close()
    assert not emulated_spyre.batches and not emulated_spyre.allocated


def test_spyre_budget_is_not_guessed_from_native_byte_sizes(emulated_spyre, tmp_path):
    paths, _ = _native_files(tmp_path)
    with pytest.raises(NotImplementedError, match="Spyre device_memory_budget"):
        ParallelLoader(
            None,
            paths,
            device="spyre:0",
            all_local=True,
            device_memory_budget=1024,
            use_tqdm_on_load=False,
            set_numa=False,
        )


def test_copier_cleanup_only_cancels_its_own_file(emulated_spyre, tmp_path):
    import torch

    paths, expected = _native_files(tmp_path)
    loader = SafeTensorsFileLoader(None, "spyre:0", set_numa=False)
    copiers = [
        loader.copier_constructor(
            SafeTensorsMetadata.from_file(path, loader.framework),
            loader.device,
            loader.framework,
        )
        for path in paths
    ]
    try:
        for copier in copiers:
            names = set(copier.metadata.tensors)
            copier.set_chunk(
                copier.metadata.select_byte_ranges(names.__contains__), names
            )
        buffers = [copier.submit_io(False, 1024) for copier in copiers]
        copiers[0].finish_io()
        copiers[0].finish_io()
        assert copiers[0].fd is None
        got = copiers[1].wait_io(buffers[1])
        assert got
        for name, tensor in got.items():
            assert torch.equal(tensor.get_raw(), expected[name])
        assert emulated_spyre.batches
        assert copiers[1].fd is None
    finally:
        for copier in copiers:
            copier.finish_io()
        loader.close()


def test_spyre_read_failure_releases_file_and_allows_retry(
    emulated_spyre, tmp_path, monkeypatch
):
    from fastsafetensors.copier import host
    from fastsafetensors.copier.spyre import SpyreCopier

    paths, _ = _native_files(tmp_path, num_shards=1)
    loader = SafeTensorsFileLoader(None, "spyre:0", set_numa=False)
    loader.add_filenames({0: paths})
    copiers = []
    real_submit = SpyreCopier.submit_io
    real_read = host.pread_into

    def record(self, *args):
        result = real_submit(self, *args)
        copiers.append(self)
        return result

    def fail(*args):
        raise OSError("read failed")

    monkeypatch.setattr(SpyreCopier, "submit_io", record)
    monkeypatch.setattr(host, "pread_into", fail)
    try:
        with pytest.raises(OSError, match="read failed"):
            loader.copy_files_to_device()
        assert copiers and all(c.fd is None and c.request is None for c in copiers)
        assert not emulated_spyre.batches
        monkeypatch.setattr(host, "pread_into", real_read)
        fb = loader.copy_files_to_device()
        fb.close()
        assert emulated_spyre.batches
    finally:
        loader.close()


def test_copier_finish_keeps_fd_open_until_running_pread_drains(
    emulated_spyre, tmp_path, monkeypatch
):
    import threading
    from concurrent.futures import ThreadPoolExecutor

    from fastsafetensors.copier import host

    paths, _ = _native_files(tmp_path, num_shards=1)
    active, release = threading.Event(), threading.Event()
    real_read = host.pread_into

    def read(fd, view, offset):
        os.fstat(fd)
        active.set()
        assert release.wait(5)
        real_read(fd, view, offset)
        os.fstat(fd)

    monkeypatch.setattr(host, "pread_into", read)
    loader = SafeTensorsFileLoader(None, "spyre:0", set_numa=False)
    metadata = SafeTensorsMetadata.from_file(paths[0], loader.framework)
    copier = loader.copier_constructor(metadata, loader.device, loader.framework)
    copier.set_chunk(metadata.select_byte_ranges(lambda name: name == "a"), {"a"})
    try:
        copier.submit_io(False, 1024)
        fd = copier.fd
        with ThreadPoolExecutor(1) as waiter:
            try:
                assert active.wait(5)
                request = copier.request
                finished = waiter.submit(copier.finish_io)
                assert request.cancelled.wait(5)
                assert not finished.done()
                os.fstat(fd)
            finally:
                release.set()
            finished.result(5)
        assert copier.fd is None and not emulated_spyre.batches
        with pytest.raises(OSError):
            os.fstat(fd)
    finally:
        release.set()
        copier.finish_io()
        loader.close()


@pytest.mark.parametrize("bbuf_size_kb", [None, 8])
def test_spyre_loader_passes_reader_and_block_settings(
    emulated_spyre, tmp_path, monkeypatch, bbuf_size_kb
):
    from fastsafetensors.copier import host

    paths, _ = _native_files(tmp_path, num_shards=1)
    reader_type = host.HostFileReader
    settings, lengths = [], []

    class RecordingReader(reader_type):
        def __init__(self, ops, max_threads, *, max_bytes):
            settings.append((max_threads, max_bytes))
            super().__init__(ops, max_threads, max_bytes=max_bytes)

    preadv = os.preadv

    def record_read(fd, buffers, offset):
        lengths.append(len(buffers[0]))
        return preadv(fd, buffers, offset)

    monkeypatch.setattr("fastsafetensors.copier.spyre.HostFileReader", RecordingReader)
    monkeypatch.setattr(os, "preadv", record_read)
    loader = SafeTensorsFileLoader(
        None, "spyre:0", bbuf_size_kb=bbuf_size_kb, max_threads=2, set_numa=False
    )
    loader.add_filenames({0: paths})
    try:
        for _ in range(2):
            fb = loader.copy_files_to_device(max_copy_block_size=7)
            fb.close()
    finally:
        loader.close()
    assert settings == [(2, (16 * 1024 if bbuf_size_kb is None else 8) * 1024)]
    assert len(lengths) > 2 and max(lengths) <= 7


def test_parallel_spyre_early_stop_does_not_leak_files(
    emulated_spyre, tmp_path, monkeypatch
):
    import contextlib

    paths, _ = _native_files(tmp_path, num_shards=1)
    real_open, real_close = os.open, os.close
    live = set()

    def track_open(path, *args, **kwargs):
        fd = real_open(path, *args, **kwargs)
        if str(path) in paths:
            live.add(fd)
        return fd

    def track_close(fd):
        real_close(fd)
        live.discard(fd)

    monkeypatch.setattr(os, "open", track_open)
    monkeypatch.setattr(os, "close", track_close)
    loader = ParallelLoader(
        None,
        paths,
        device="spyre:0",
        max_batch_bytes=256,
        queue_size=2,
        use_tqdm_on_load=False,
        set_numa=False,
    )
    try:
        with contextlib.closing(loader.iterate_weights()) as weights:
            assert next(weights)[0]
    finally:
        loader.close()
    assert not live
