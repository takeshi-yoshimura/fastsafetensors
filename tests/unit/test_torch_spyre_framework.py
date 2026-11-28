# SPDX-License-Identifier: Apache-2.0
"""Framework-owned Spyre allocation and tensor operations, without hardware."""

import ast
import builtins
import gc
import json
import mmap
import runpy
import subprocess
import sys
import threading
import weakref
from enum import Enum
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import setuptools
import torch

from fastsafetensors.frameworks import FrameworkOpBase, TensorBase
from fastsafetensors.frameworks import _torch_spyre as bridge
from fastsafetensors.frameworks._torch import TorchOp
from fastsafetensors.native import NativeFormatError, NativeTensorSpec, canonical_layout
from fastsafetensors.st_types import Device, DeviceType, DType


@pytest.fixture
def ops():
    return TorchOp().get_native_ops(Device(DeviceType.SPYRE))


@pytest.fixture
def runtime(monkeypatch):
    package = ModuleType("torch_spyre")
    runtime = ModuleType("torch_spyre._C")
    for name in ("copy_buffer_ranges", "convert_buffers"):
        setattr(runtime, name, lambda *a, **kw: pytest.fail("not called at import"))
    monkeypatch.setitem(sys.modules, "torch_spyre", package)
    monkeypatch.setitem(sys.modules, "torch_spyre._C", runtime)
    bridge.backend.cache_clear()
    yield runtime
    bridge.backend.cache_clear()


def test_backend_contract_is_checked_and_cached(runtime):
    assert bridge.backend() is runtime
    assert bridge.backend() is runtime


def test_backend_needs_only_the_two_callable_apis(runtime):
    assert sorted(name for name in vars(runtime) if not name.startswith("__")) == [
        "convert_buffers",
        "copy_buffer_ranges",
    ]
    assert bridge.backend() is runtime


@pytest.mark.parametrize("name", ["copy_buffer_ranges", "convert_buffers"])
@pytest.mark.parametrize("value", [None, 123])
def test_reject_missing_or_noncallable_capability(runtime, name, value):
    setattr(runtime, name, value)
    with pytest.raises(RuntimeError, match="require callable"):
        bridge.backend()


@pytest.mark.parametrize("name", ["copy_buffer_ranges", "convert_buffers"])
def test_reject_absent_capability(runtime, name):
    delattr(runtime, name)
    with pytest.raises(RuntimeError, match="require callable"):
        bridge.backend()


def test_missing_api_does_not_use_legacy_bindings(runtime, monkeypatch):
    monkeypatch.setitem(sys.modules, "torch_spyre", None)
    with pytest.raises(ImportError, match="No legacy DMA fallback"):
        bridge.backend()


@pytest.fixture
def layouts(monkeypatch):
    class DataFormats(Enum):
        FP16 = 1

    class ElementArrangement(Enum):
        CONTIGUOUS = 1

    def layout(
        device_size, stride_map, device_dtype, valid_elements, element_arrangement
    ):
        return SimpleNamespace(
            device_size=device_size,
            stride_map=stride_map,
            device_dtype=device_dtype,
            valid_elements=valid_elements,
            element_arrangement=element_arrangement,
        )

    runtime = ModuleType("torch_spyre._C")
    runtime.DataFormats = DataFormats
    runtime.ElementArrangement = ElementArrangement
    runtime.SpyreTensorLayout = layout
    monkeypatch.setitem(sys.modules, "torch_spyre._C", runtime)
    return bridge, layout(
        [2, 4], [4, 1], DataFormats.FP16, {}, ElementArrangement.CONTIGUOUS
    )


@pytest.mark.parametrize("valid_elements", [{}, {(0,): 2, (1, 2): 7}])
def test_layout_round_trip(layouts, valid_elements):
    module, layout = layouts
    layout.valid_elements = valid_elements
    encoded = json.loads(json.dumps(module.layout_to_json(layout)))
    assert encoded == {
        "device_size": [2, 4],
        "stride_map": [4, 1],
        "device_dtype": "FP16",
        "element_arrangement": "CONTIGUOUS",
        "valid_elements": [
            [list(dims), count] for dims, count in valid_elements.items()
        ],
    }
    assert module.layout_from_json(encoded) == layout


def test_legacy_layout_without_valid_elements(layouts):
    module, layout = layouts
    encoded = module.layout_to_json(layout)
    del encoded["valid_elements"]
    assert module.layout_from_json(encoded) == layout


@pytest.mark.parametrize("field", ["device_dtype", "element_arrangement"])
def test_unknown_layout_enum_is_rejected(layouts, field):
    module, layout = layouts
    encoded = module.layout_to_json(layout)
    encoded[field] = "unknown"
    with pytest.raises(KeyError):
        module.layout_from_json(encoded)


def test_layout_helper_import_does_not_load_torch_spyre():
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; "
            "from fastsafetensors.frameworks import _torch_spyre; "
            "assert not any(n == 'torch_spyre' or n.startswith('torch_spyre.') "
            "for n in sys.modules)",
        ],
        check=True,
    )


def test_setup_never_imports_torch_with_sdk_opt_in(monkeypatch):
    original = builtins.__import__
    calls = []

    def guarded(name, *args, **kwargs):
        if name.split(".")[0] in {"torch", "torch_spyre", "flex", "deeptools"}:
            raise AssertionError(f"build imported {name}")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    monkeypatch.setattr(setuptools, "setup", lambda **kw: calls.append(kw))
    monkeypatch.setenv("FASTSAFETENSORS_BUILD_SPYRE", "1")
    runpy.run_path(
        str(Path(__file__).resolve().parents[2] / "setup.py"), run_name="__main__"
    )
    assert len(calls) == 1
    assert [ext.name for ext in calls[0]["ext_modules"]] == ["fastsafetensors.cpp"]
    extension = calls[0]["ext_modules"][0]
    assert extension.libraries == ["stdc++"]
    assert all("torch" not in path.lower() for path in extension.include_dirs)


def test_no_spyre_bridge_or_torch_headers_in_cpp():
    source = Path(__file__).resolve().parents[2] / "fastsafetensors/cpp"
    assert not (source / "spyre.cpp").exists()
    for pattern in ("*.cpp", "*.hpp", "*.h"):
        for path in source.glob(pattern):
            text = path.read_text()
            assert "<ATen/" not in text
            assert "<c10/" not in text
            assert "<torch/" not in text
            assert "_ZN5spyre" not in text
            assert "#include <flex/" not in text
            assert "spyrecode-host-functions" not in text


def test_backend_contract_has_no_binary_layout_or_dso_dependency():
    source = Path(bridge.__file__).read_text()
    function = next(
        node
        for node in ast.parse(source).body
        if isinstance(node, ast.FunctionDef) and node.name == "backend"
    )
    imports = []
    for node in ast.walk(function):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imports.append(node.module or "")
    assert imports == ["torch_spyre"]
    assert "dlsym" not in source and "_spyre_cpp" not in source


def test_native_adapter_and_plain_allocation_need_no_spyre(monkeypatch):
    original = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name.split(".")[0] == "torch_spyre":
            raise AssertionError(f"unexpected runtime import: {name}")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    framework = TorchOp()
    ops = framework.get_native_ops(Device(DeviceType.SPYRE))
    tensor = framework.get_empty_tensor([2, 3], DType.F32, Device(DeviceType.CPU))
    assert tensor.get_shape() == [2, 3] and tensor.get_raw().dtype == torch.float32
    view = ops.host_buffer_view(tensor)
    assert view.nbytes == 24
    view[:] = bytes(24)
    assert torch.equal(tensor.get_raw(), torch.zeros(2, 3))
    layout, size = canonical_layout([2, 3], DType.F32)
    assert ops.tensor_spec("w", tensor, size, layout).dtype == DType.F32


@pytest.mark.parametrize("nbytes", [1, 127, 128, 4096, 65537, 1048576])
def test_aligned_host_allocation_and_mapping_lifetime(ops, monkeypatch, nbytes):
    from fastsafetensors.frameworks import _torch_spyre

    allocate = mmap.mmap
    mappings = []

    def track(*args):
        mapping = allocate(*args)
        mappings.append(weakref.ref(mapping))
        return mapping

    monkeypatch.setattr(_torch_spyre.mmap, "mmap", track)
    monkeypatch.setattr(ops, "_device", Device())
    tensor = ops.allocate_host_buffer(nbytes)
    assert isinstance(tensor, TensorBase) and tensor.get_nbytes() == nbytes
    assert tensor.data_ptr() % mmap.PAGESIZE == 0
    view = tensor[:]
    del tensor
    gc.collect()
    assert mappings[0]() is not None
    ops.host_buffer_view(view)[:] = bytes([45]) * nbytes
    assert view.get_raw()[-1].item() == 45
    del view
    gc.collect()
    assert mappings[0]() is None


def test_mapped_host_allocation_uses_existing_pinned_cpu_factory(ops, monkeypatch):
    calls = []
    owner = torch.empty(4096, dtype=torch.uint8)
    reference = weakref.ref(owner)
    device = SimpleNamespace(type="spyre", index=0)
    monkeypatch.setattr(ops, "_torch_device", lambda d: device)
    monkeypatch.setattr(
        torch,
        "spyre",
        SimpleNamespace(_lazy_init=lambda: calls.append("init")),
        raising=False,
    )

    def allocate(nbytes, **kwargs):
        calls.append((nbytes, kwargs))
        return reference()

    monkeypatch.setattr(torch, "empty", allocate)
    tensor = ops.allocate_host_buffer(4096)
    assert isinstance(tensor, TensorBase) and tensor.get_nbytes() == 4096
    assert calls == [
        "init",
        (4096, dict(dtype=torch.uint8, device="cpu", pin_memory=True)),
    ]
    view = tensor[:]
    del owner, tensor
    gc.collect()
    assert reference() is not None
    view.get_raw().fill_(42)
    assert reference()[0].item() == 42
    del view
    gc.collect()
    assert reference() is None


@pytest.mark.parametrize("nbytes", [0, -1])
def test_host_allocation_rejects_invalid_size_before_runtime(ops, nbytes):
    with pytest.raises(ValueError, match="positive"):
        ops.allocate_host_buffer(nbytes)


def test_prepared_copy_passes_captured_sdk_stream_to_worker(ops, monkeypatch):
    runtime = ModuleType("torch_spyre")
    token = object()
    runtime.current_stream = lambda device: token
    monkeypatch.setitem(sys.modules, "torch_spyre", runtime)
    monkeypatch.setattr(ops, "_torch_device", lambda device: device)
    calls = []
    monkeypatch.setattr(
        ops, "submit_raw_tensors", lambda entries, **kw: calls.append((entries, kw))
    )
    copy = ops.prepare_copy(Device(DeviceType.SPYRE, 2))
    worker = threading.Thread(target=copy, args=([],))
    worker.start()
    worker.join(5)
    assert not worker.is_alive()
    assert calls == [([], {"stream": token, "device": Device(DeviceType.SPYRE, 2)})]


@pytest.mark.parametrize("size", [0, (4 << 30) + 1])
def test_raw_transfer_rejects_unstageable_payload(ops, monkeypatch, size):
    monkeypatch.setattr(ops, "tensor_nbytes", lambda tensor: size)
    with pytest.raises(ValueError, match="invalid raw transfer size"):
        ops.validate_transfer(ops.as_tensor(torch.empty(1)), Device())


def test_optional_layout_serialization_needs_no_backend(ops, monkeypatch):
    monkeypatch.setitem(sys.modules, "torch_spyre", None)
    assert ops.layout_from_json(None) is None
    assert ops.layout_to_json(None) is None


def test_native_layout_allocation_uses_fixed_description(ops, monkeypatch):
    runtime = ModuleType("torch_spyre._C")
    layout = object()
    fixed, size = canonical_layout([3, 2], DType.F16)
    monkeypatch.setattr(
        ops, "layout_from_json", lambda data: layout if data == fixed else None
    )
    monkeypatch.setattr(ops, "tensor_layout", lambda t: fixed)
    monkeypatch.setattr(ops, "tensor_nbytes", lambda t: size)
    calls = []

    def allocate(shape, stride, dtype, placed, device):
        calls.append((shape, stride, dtype, placed, device))
        return torch.empty_strided(shape, stride, dtype=dtype, device=device)

    runtime.spyre_empty_with_layout = allocate
    monkeypatch.setitem(sys.modules, "torch_spyre._C", runtime)
    spec = NativeTensorSpec("w", DType.F16, [3, 2], [2, 1], fixed, size)
    tensor = ops.empty_tensor(spec, "cpu")
    assert calls == [([3, 2], [2, 1], torch.float16, layout, torch.device("cpu"))]
    assert list(tensor.get_raw().stride()) == [2, 1]


def test_explicit_layout_allocator_initializes_runtime_before_pybind(ops, monkeypatch):
    runtime = ModuleType("torch_spyre._C")
    events = []
    monkeypatch.setattr(
        torch,
        "spyre",
        SimpleNamespace(_lazy_init=lambda: events.append("init")),
        raising=False,
    )
    monkeypatch.setattr(
        ops, "_torch_device", lambda device: SimpleNamespace(type="spyre")
    )
    monkeypatch.setattr(ops, "layout_from_json", lambda data: data)

    def allocate(*args, **kwargs):
        events.append("allocate")
        return torch.empty(4)

    runtime.spyre_empty_with_layout = allocate
    monkeypatch.setitem(sys.modules, "torch_spyre._C", runtime)
    fixed, size = canonical_layout([4], DType.F32)
    monkeypatch.setattr(ops, "validate_destination", lambda *a, **kw: None)
    monkeypatch.setattr(ops, "tensor_nbytes", lambda t: size)
    spec = NativeTensorSpec("w", DType.F32, [4], [1], fixed, size)
    ops.empty_tensor(spec, "spyre:0")
    assert events == ["init", "allocate"]


@pytest.mark.parametrize("mismatch", ["shape", "stride", "dtype", "layout", "device"])
def test_destination_validation_rejects_mismatch(ops, monkeypatch, mismatch):
    monkeypatch.setattr(ops, "tensor_layout", lambda tensor: {"tiles": 2})
    tensor = torch.empty_strided([3, 2], [1, 3])
    arguments = dict(stride=[1, 3], layout={"tiles": 2})
    shape, dtype, device = [3, 2], DType.F32, "cpu"
    if mismatch == "shape":
        shape = [2, 3]
    elif mismatch == "stride":
        arguments["stride"] = [2, 1]
    elif mismatch == "dtype":
        dtype = DType.F16
    elif mismatch == "layout":
        arguments["layout"] = {"tiles": 3}
    else:
        device = "meta"
    with pytest.raises(NativeFormatError, match="destination"):
        ops.validate_destination("w", tensor, shape, dtype, device, **arguments)


def test_plain_packed_tensor_uses_framework_native_shape(ops):
    if not hasattr(torch, "float4_e2m1fn_x2"):
        pytest.skip("PyTorch has no packed float4 dtype")
    tensor = ops.framework.get_empty_tensor([2, 8], DType.F4, Device(DeviceType.CPU))
    assert tensor.get_raw().shape == (2, 4)
    assert tensor.get_shape() == [2, 8]
    assert ops.host_buffer_view(tensor).nbytes == 8
    ops.validate_plain_destination("w", tensor.get_raw(), [2, 8], DType.F4, "cpu")


@pytest.mark.parametrize("field", ["layout", "stride"])
def test_native_null_metadata_does_not_disable_validation(ops, monkeypatch, field):
    monkeypatch.setattr(ops, "tensor_layout", lambda tensor: {"tiles": 2})
    with pytest.raises(NativeFormatError, match="destination"):
        ops.validate_destination(
            "w", torch.empty(2, 3), [2, 3], DType.F32, "cpu", **{field: None}
        )


def test_plain_unspecified_layout_needs_no_runtime(ops):
    ops.validate_plain_destination(
        "w", torch.empty(2, 3), [2, 3], DType.F32, "cpu", layout=None
    )


def test_runtime_dependency_error_is_reported_by_copier(monkeypatch):
    from fastsafetensors.copier import spyre

    original = builtins.__import__

    def missing(name, *args, **kwargs):
        if name == "torch_spyre":
            raise ModuleNotFoundError("no torch-spyre", name="torch_spyre")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", missing)
    with pytest.raises(ImportError, match="install torch-spyre"):
        spyre.load_library_spyre(Device(DeviceType.SPYRE), framework=TorchOp())


def test_library_load_checks_backend_contract_before_allocation(ops, monkeypatch):
    package = ModuleType("torch_spyre")
    events = []
    package._autoload = lambda: events.append("autoload")
    monkeypatch.setitem(sys.modules, "torch_spyre", package)
    monkeypatch.setattr(
        torch._C, "_get_privateuse1_backend_name", lambda: "privateuseone"
    )

    def missing():
        events.append("contract")
        raise ImportError("SDK extension API v1 required")

    monkeypatch.setattr("fastsafetensors.frameworks._torch_spyre.backend", missing)
    with pytest.raises(ImportError, match="API v1"):
        ops.load_library()
    assert events == ["autoload", "contract"]


def test_raw_copy_keeps_exact_physical_ranges_at_python_boundary(ops, monkeypatch):
    source = torch.empty(4096, dtype=torch.uint8)
    destination = torch.empty(1)
    calls = []
    runtime = SimpleNamespace(
        copy_buffer_ranges=lambda *args, **kw: calls.append((args, kw)),
        stream_get_error=lambda stream: 0,
        SpyreStreamError=SimpleNamespace(Success=0),
    )
    monkeypatch.setattr(
        "fastsafetensors.frameworks._torch_spyre.backend", lambda: runtime
    )
    src, dst = ops.as_tensor(source), ops.as_tensor(destination)
    capture = ModuleType("torch_spyre._C")
    capture.capture_flex_stream_v1 = lambda device: pytest.fail("recaptured stream")
    monkeypatch.setitem(sys.modules, "torch_spyre._C", capture)
    owner = object()
    geometry = (3, 0, 8192, 0)
    monkeypatch.setattr(ops, "allocation_descriptor", lambda dst: (geometry, owner))
    stream = SimpleNamespace(synchronize=lambda: None)
    ops.copy_raw_tensors(
        [(src, dst, 4096, 128), (src, dst, 0, 4096)],
        stream=stream,
        device=destination.device,
    )
    assert len(calls) == 1
    assert [(offset, size) for _, _, offset, size in calls[0][0][0]] == [
        (4096, 128),
        (0, 4096),
    ]
    assert calls[0][1] == {"stream": stream}
    assert all(a.nbytes == 4096 and o is owner for a, o, _, _ in calls[0][0][0])


def test_raw_buffer_allocation_stays_fail_closed():
    framework = TorchOp()
    with pytest.raises(NotImplementedError, match="opaque handle"):
        framework.alloc_tensor_memory(16, Device(DeviceType.SPYRE))
    assert framework.get_mem_used() == 0


@pytest.mark.parametrize("failure", [None, "enqueue", "sync", "shutdown", "after-sync"])
def test_async_copy_owns_exports_until_drain_and_quarantines_unknown(
    ops, monkeypatch, failure
):
    from fastsafetensors.copier import host
    from fastsafetensors.frameworks import TransferDrainedError, _torch_spyre

    monkeypatch.setattr(host, "_failed_transfers", [])
    monkeypatch.setattr(host, "_transfer_poisoned", False)
    monkeypatch.setattr(_torch_spyre, "_raw_transfers", [])
    source = ops.as_tensor(torch.arange(128, dtype=torch.uint8))
    target = ops.as_tensor(torch.empty(128, dtype=torch.uint8))

    class Owner:
        pass

    owners, events = [], []

    def descriptor(dst):
        owner = Owner()
        owners.append(weakref.ref(owner))
        return None, owner

    monkeypatch.setattr(ops, "allocation_descriptor", descriptor)

    def enqueue(work, *, stream):
        events.append("enqueue")
        assert _torch_spyre._raw_transfers
        assert owners[0]() is work[0][1]
        if failure == "enqueue":
            raise RuntimeError("partial submission")

    def sync():
        events.append("sync")
        gc.collect()
        assert owners[0]() is not None
        # The producer may prepare the other slot while DMA is in flight.
        host.check_transfer_state()
        assert _torch_spyre._raw_transfers[0].lifetime[1][0][0].nbytes == 128
        if failure == "sync":
            raise RuntimeError("failed drain")

    stream = SimpleNamespace(synchronize=sync, _cdata=object())
    runtime = SimpleNamespace(
        copy_buffer_ranges=enqueue,
        stream_get_error=lambda s: int(
            failure == "shutdown" or (failure == "after-sync" and "sync" in events)
        ),
        SpyreStreamError=SimpleNamespace(Success=0),
    )
    monkeypatch.setattr(_torch_spyre, "backend", lambda: runtime)
    entries = [(source, target, 0, 128)]
    if failure:
        error = TransferDrainedError if failure == "enqueue" else RuntimeError
        with pytest.raises(error, match="partial submission|completion unknown"):
            ops.copy_raw_tensors(entries, stream=stream, device=torch.device("cpu"))
    else:
        ops.copy_raw_tensors(entries, stream=stream, device=torch.device("cpu"))
    assert events == (["enqueue"] if failure == "shutdown" else ["enqueue", "sync"])
    gc.collect()
    if failure in {"sync", "shutdown", "after-sync"}:
        assert owners[0]() is not None and _torch_spyre._raw_transfers
        with pytest.raises(RuntimeError, match="restart"):
            ops.copy_raw_tensors(entries, stream=stream, device=torch.device("cpu"))
        assert events.count("enqueue") == 1
    else:
        assert not _torch_spyre._raw_transfers
        host.check_transfer_state()


@pytest.mark.parametrize("failure", [None, "second-enqueue", "first-wait", "shutdown"])
def test_deferred_tickets_own_each_submission_and_never_retry_unknown_completion(
    ops, monkeypatch, failure
):
    from fastsafetensors.copier import host
    from fastsafetensors.frameworks import TransferDrainedError, _torch_spyre

    monkeypatch.setattr(host, "_failed_transfers", [])
    monkeypatch.setattr(host, "_transfer_poisoned", False)
    monkeypatch.setattr(_torch_spyre, "_raw_transfers", [])
    monkeypatch.setattr(ops, "allocation_descriptor", lambda dst: (None, dst))
    events = []

    def enqueue(work, *, stream):
        events.append(("enqueue", stream.name))
        if failure == "second-enqueue" and stream.name == "B":
            raise RuntimeError("partial second enqueue")

    def sync(name):
        events.append(("wait", name))
        if failure == "first-wait" and name == "A":
            raise RuntimeError("unknown wait")

    api = SimpleNamespace(
        copy_buffer_ranges=enqueue,
        stream_get_error=lambda stream: int(failure == "shutdown"),
        SpyreStreamError=SimpleNamespace(Success=0),
    )
    monkeypatch.setattr(_torch_spyre, "backend", lambda: api)
    streams = [
        SimpleNamespace(name=name, synchronize=lambda n=name: sync(n)) for name in "AB"
    ]
    sources = [ops.as_tensor(torch.arange(128, dtype=torch.uint8)) for _ in streams]
    targets = [ops.as_tensor(torch.empty(128, dtype=torch.uint8)) for _ in streams]
    references = [weakref.ref(t.get_raw()) for t in sources + targets]
    tickets = []
    for index, stream in enumerate(streams):
        entries = [(sources[index], targets[index], 0, 128)]
        if index == 1 and failure == "second-enqueue":
            with pytest.raises(TransferDrainedError, match="partial second"):
                ops.submit_raw_tensors(
                    entries, stream=stream, device=torch.device("cpu")
                )
        else:
            tickets.append(
                ops.submit_raw_tensors(
                    entries, stream=stream, device=torch.device("cpu")
                )
            )
    assert events[:2] == [("enqueue", "A"), ("enqueue", "B")]
    del sources, targets, entries
    gc.collect()
    if failure == "second-enqueue":
        # Removing B must not pop A's registry entry or release its owners.
        assert len(_torch_spyre._raw_transfers) == 1
        assert references[0]() is not None and references[2]() is not None
        tickets[0].wait()
        assert events[-2:] == [("wait", "B"), ("wait", "A")]
    elif failure:
        for ticket in tickets:
            with pytest.raises(RuntimeError, match="completion unknown"):
                ticket.wait()
        previous = list(events)
        for ticket in tickets:
            with pytest.raises(RuntimeError, match="completion unknown"):
                ticket.wait()
        assert events == previous
        assert len(_torch_spyre._raw_transfers) == 2
        assert all(ref() is not None for ref in references)
        assert [e for e in events if e[0] == "wait"] == (
            [("wait", "A")] if failure == "first-wait" else []
        )
    else:
        assert len(_torch_spyre._raw_transfers) == 2
        assert all(ref() is not None for ref in references)
        tickets[1].wait()
        tickets[1].wait()  # Successful wait is idempotent too.
        assert len(_torch_spyre._raw_transfers) == 1
        tickets[0].wait()
        assert events[2:] == [("wait", "B"), ("wait", "A")]
    if not failure or failure == "second-enqueue":
        assert not _torch_spyre._raw_transfers
        gc.collect()
        assert all(ref() is None for ref in references)


def test_dropping_unwaited_ticket_retains_pending_owners(ops, monkeypatch):
    from fastsafetensors.copier import host
    from fastsafetensors.frameworks import _torch_spyre

    monkeypatch.setattr(host, "_failed_transfers", [])
    monkeypatch.setattr(host, "_transfer_poisoned", False)
    monkeypatch.setattr(_torch_spyre, "_raw_transfers", [])
    monkeypatch.setattr(ops, "allocation_descriptor", lambda dst: (None, dst))
    monkeypatch.setattr(
        _torch_spyre,
        "backend",
        lambda: SimpleNamespace(copy_buffer_ranges=lambda *a, **kw: None),
    )
    source = ops.as_tensor(torch.zeros(128, dtype=torch.uint8))
    target = ops.as_tensor(torch.empty(128, dtype=torch.uint8))
    references = [weakref.ref(t.get_raw()) for t in (source, target)]
    ticket = ops.submit_raw_tensors(
        [(source, target, 0, 128)], stream=object(), device=torch.device("cpu")
    )
    del source, target, ticket
    gc.collect()
    assert all(ref() is not None for ref in references)
    assert len(_torch_spyre._raw_transfers) == 1


def test_reader_enqueues_both_slots_before_wait_and_never_reuses_pending_slot(
    ops, monkeypatch, tmp_path
):
    from concurrent.futures import ThreadPoolExecutor

    from fastsafetensors.copier import host
    from fastsafetensors.frameworks import _torch_spyre

    waiting, release, read_second = (threading.Event() for _ in range(3))
    monkeypatch.setattr(host, "_failed_transfers", [])
    monkeypatch.setattr(host, "_transfer_poisoned", False)
    monkeypatch.setattr(_torch_spyre, "_raw_transfers", [])
    path = tmp_path / "weights"
    path.write_bytes(bytes(range(128)) * 3)
    preadv = host.os.preadv

    def read(fd, buffers, offset):
        assert offset < 256 or release.is_set(), "reused a pending DMA slot"
        result = preadv(fd, buffers, offset)
        if offset == 128:
            read_second.set()
        return result

    monkeypatch.setattr(host.os, "preadv", read)
    monkeypatch.setattr(ops, "transfer_alignment", lambda: 128)
    monkeypatch.setattr(
        ops,
        "allocate_host_buffer",
        lambda size: ops.as_tensor(torch.empty(size, dtype=torch.uint8)),
    )
    monkeypatch.setattr(ops, "validate_transfer", lambda dst, device: 384)
    monkeypatch.setattr(ops, "allocation_descriptor", lambda dst: (None, dst))
    launches = []

    def enqueue(work, **kwargs):
        launches.append(bytes(work[0][0]))

    def sync():
        assert len(launches) == 2
        waiting.set()
        assert release.wait(5)

    stream = SimpleNamespace(synchronize=sync)
    runtime = SimpleNamespace(
        copy_buffer_ranges=enqueue,
        stream_get_error=lambda s: 0,
        SpyreStreamError=SimpleNamespace(Success=0),
    )
    monkeypatch.setattr(_torch_spyre, "backend", lambda: runtime)
    monkeypatch.setattr(
        ops,
        "prepare_copy",
        lambda device: lambda entries: ops.submit_raw_tensors(
            entries, stream=stream, device=torch.device("cpu")
        ),
    )
    reader = host.HostFileReader(ops, max_threads=1, max_bytes=256)
    try:
        with path.open("rb") as file, ThreadPoolExecutor(1) as waiter:
            request = reader.submit_read(
                [
                    (
                        (file.fileno(), 0),
                        ops.as_tensor(torch.empty(384, dtype=torch.uint8)),
                    )
                ],
                Device(),
            )
            assert waiting.wait(5) and read_second.wait(5)
            assert len(launches) == 2 and len(_torch_spyre._raw_transfers) == 2
            assert len(reader._state.pool.busy) == 2
            assert not request.done.is_set()
            closed = waiter.submit(reader.close)
            assert request.cancelled.wait(5) and not closed.done()
            release.set()
            closed.result(5)
            assert not _torch_spyre._raw_transfers
    finally:
        release.set()
        reader.close()


@pytest.mark.parametrize(
    "fault", [None, "cpu", "view", "offset", "stride", "chunks", "size"]
)
def test_allocation_descriptor_uses_chunk_values_and_storage_owner(
    ops, monkeypatch, fault
):
    runtime = ModuleType("torch_spyre._C")
    chunk = SimpleNamespace(region_id=9, offset=128, size=4096, domain_id=2)
    owner = SimpleNamespace(total_size=4096, num_chunks=1, chunks=lambda: [chunk])
    raw = SimpleNamespace(
        device=SimpleNamespace(type="cpu" if fault == "cpu" else "spyre"),
        storage_offset=lambda: int(fault == "offset"),
        _is_view=lambda: fault == "view",
        is_contiguous=lambda: fault != "stride",
        data_ptr=lambda: pytest.fail("destination data_ptr must never be read"),
    )
    if fault == "chunks":
        owner.num_chunks = 2
    elif fault == "size":
        owner.total_size = 8192
    runtime.get_composite_address = lambda tensor: owner
    monkeypatch.setitem(sys.modules, "torch_spyre._C", runtime)
    monkeypatch.setattr(ops, "as_tensor", lambda t: t)
    monkeypatch.setattr(ops, "unwrap", lambda t: t)
    if fault:
        with pytest.raises(ValueError, match="destination"):
            ops.allocation_descriptor(raw)
    else:
        assert ops.allocation_descriptor(raw) == ((9, 128, 4096, 2), owner)


@pytest.mark.parametrize(
    "fault", ["shape", "dtype", "stride", "offset", "layout", "size"]
)
def test_fixed_conversion_rejects_batch_before_cpp(ops, monkeypatch, fault):
    source = torch.ones(2, 3)
    destination = torch.empty_like(source)
    bad = source
    layout, size = canonical_layout([2, 3], DType.F32)
    monkeypatch.setattr(
        ops, "tensor_layout", lambda t: {} if fault == "layout" else layout
    )
    monkeypatch.setattr(ops, "tensor_nbytes", lambda t: size + int(fault == "size"))
    if fault == "shape":
        bad = torch.ones(3, 2)
    elif fault == "dtype":
        bad = source.half()
    elif fault == "stride":
        bad = torch.ones(3, 2).t()
    elif fault == "offset":
        bad = torch.ones(9)[3:].reshape(2, 3)
    runtime = SimpleNamespace(
        convert_buffers=lambda *a, **kw: pytest.fail("submitted invalid batch")
    )
    monkeypatch.setattr(
        "fastsafetensors.frameworks._torch_spyre.backend", lambda: runtime
    )
    with pytest.raises(ValueError, match="fixed-profile"):
        ops.encode_tensors(
            [
                (
                    memoryview(bytearray(size)),
                    ops.as_tensor(source),
                    ops.as_tensor(destination),
                ),
                (
                    memoryview(bytearray(size)),
                    ops.as_tensor(bad),
                    ops.as_tensor(destination),
                ),
            ]
        )


def test_other_native_backends_remain_unsupported():
    with pytest.raises(NotImplementedError, match="native tensor operations"):
        TorchOp().get_native_ops(Device(DeviceType.CPU))
    with pytest.raises(NotImplementedError, match="native tensor operations"):
        FrameworkOpBase.get_native_ops(TorchOp(), Device(DeviceType.SPYRE))


@pytest.mark.parametrize("module", ["spyre", "host"])
def test_copier_has_no_framework_implementation_imports(module):
    from importlib import import_module

    tree = ast.parse(
        Path(import_module(f"fastsafetensors.copier.{module}").__file__).read_text()
    )
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            assert (node.module or "").split(".")[0] not in {"torch", "torch_spyre"}
            assert not (node.module or "").startswith("torch_spyre.")
            assert "frameworks._torch" not in (node.module or "")
        elif isinstance(node, ast.Import):
            assert not any(
                alias.name.split(".")[0] in {"torch", "torch_spyre"}
                for alias in node.names
            )
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            assert not (
                isinstance(node.func.value, ast.Name) and node.func.value.id == "torch"
            )


@pytest.mark.parametrize("device", ["cpu", "meta"])
def test_boundary_wrap_reuses_framework_and_preserves_object(ops, monkeypatch, device):
    raw = torch.empty_strided((3, 2), (1, 3), device=device)
    calls = []
    wrap = ops.framework.wrap_tensor

    def tracked(tensor):
        calls.append(tensor)
        return wrap(tensor)

    monkeypatch.setattr(ops.framework, "wrap_tensor", tracked)
    tensor = ops.as_tensor(raw)
    assert isinstance(tensor, TensorBase) and ops.as_tensor(tensor) is tensor
    assert len(calls) == 1 and calls[0] is raw
    assert tensor.device == Device.from_str(device)
    assert list(tensor.get_raw().stride()) == [1, 3] and ops.unwrap(tensor) is raw


def test_native_calls_unwrap_only_at_framework_boundary(ops, monkeypatch):
    runtime = ModuleType("torch_spyre._C")
    src, dst = torch.ones(4), torch.empty(4)
    host = torch.empty(16, dtype=torch.uint8)
    source, destination, scratch = map(ops.as_tensor, (src, dst, host))
    view = memoryview(bytearray(128))
    calls = []
    runtime.convert_buffers = lambda work, **kw: calls.append(("encode", work))
    runtime.copy_buffer_ranges = lambda work, **kw: calls.append(("raw", work))
    runtime.stream_get_error = lambda stream: 0
    runtime.SpyreStreamError = SimpleNamespace(Success=0)
    package = ModuleType("torch_spyre")
    package.current_stream = lambda device: SimpleNamespace(synchronize=lambda: None)
    monkeypatch.setitem(sys.modules, "torch_spyre", package)
    runtime.copy_tensor = lambda a, b, **kw: calls.append(("copy", [(a, b)]))
    monkeypatch.setitem(sys.modules, "torch_spyre._C", runtime)
    monkeypatch.setattr(
        "fastsafetensors.frameworks._torch_spyre.backend", lambda: runtime
    )
    layout, size = canonical_layout([4], DType.F32)
    monkeypatch.setattr(ops, "tensor_layout", lambda t: layout)
    monkeypatch.setattr(ops, "tensor_nbytes", lambda t: size)
    owner = object()
    monkeypatch.setattr(
        ops, "allocation_descriptor", lambda t: ((1, 0, size, 0), owner)
    )
    ops.encode_tensors([(view, source, destination)])
    ops.copy_raw_tensors([(scratch, destination)])
    ops.move_tensor(source, torch.device("cpu"), destination=destination)
    assert calls[0][1][0][0] is view
    assert bytes(calls[0][1][0][1]) == src.numpy().tobytes()
    assert calls[0][1][0][2:] == ([4], "F32")
    assert calls[1][1][0][1] is owner
    assert calls[2][1][0][0] is src and calls[2][1][0][1] is dst


def test_native_entry_owns_wrapped_sources_until_encode(ops, monkeypatch):
    import gc
    import weakref

    from fastsafetensors.native import native_entry, native_fill

    raw = torch.arange(4.0)
    reference = weakref.ref(raw)
    entry = native_entry(
        "w",
        raw,
        torch.empty_like(raw),
        nbytes=128,
        layout=canonical_layout([4], DType.F32)[0],
    )
    assert isinstance(entry.source.src, TensorBase)
    assert isinstance(entry.source.dst, TensorBase)
    del raw
    gc.collect()
    assert reference() is entry.source.src.get_raw()
    runtime = ModuleType("torch_spyre._C")

    def encode(work, **kwargs):
        assert bytes(work[0][1]) == reference().numpy().tobytes()

    runtime.convert_buffers = encode
    monkeypatch.setitem(sys.modules, "torch_spyre._C", runtime)
    monkeypatch.setattr(
        "fastsafetensors.frameworks._torch_spyre.backend", lambda: runtime
    )
    monkeypatch.setattr(
        type(ops), "tensor_layout", lambda self, t: canonical_layout([4], DType.F32)[0]
    )
    monkeypatch.setattr(type(ops), "tensor_nbytes", lambda self, t: 128)
    native_fill([(memoryview(bytearray(128)), entry)])
    del entry
    gc.collect()
    assert reference() is None


@pytest.mark.parametrize(
    "copier", ["gds", "nogds", "fgds", "unified", "dstorage", "spyre"]
)
def test_meta_load_rejected_before_backend_initialization(monkeypatch, copier):
    from fastsafetensors.copier import registry

    monkeypatch.setattr(
        registry,
        "load_and_init_and_check_copier",
        lambda *a, **kw: pytest.fail("must not initialize"),
    )
    with pytest.raises(ValueError, match="not a checkpoint load destination"):
        registry.select_copier(copier, Device(DeviceType.META))
