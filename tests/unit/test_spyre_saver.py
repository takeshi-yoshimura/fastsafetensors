# SPDX-License-Identifier: Apache-2.0
"""Spyre checkpoint writing without Spyre hardware or an inference framework."""

import builtins
import gc
import itertools
import json
import math
import sys
import weakref
from types import ModuleType

import pytest
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from fastsafetensors import ParallelSaver
from fastsafetensors.common import SafeTensorsMetadata
from fastsafetensors.copier.spyre import (
    NATIVE_FORMAT,
    native_entry,
    native_fill,
    native_shard_metadata,
)
from fastsafetensors.frameworks._torch import TorchOp
from fastsafetensors.native import (
    NativeFormatError,
    canonical_layout,
    checkpoint_paths,
    convert_to_spyre,
    native_format,
    native_specs,
)
from fastsafetensors.saver import plain_entry
from fastsafetensors.st_types import Device, DType

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
def test_native_entries_match_profile_metadata(
    tmp_path, monkeypatch, framework, num_shards
):
    src = torch.arange(16, dtype=torch.float16).reshape(4, 4)
    dst = torch.empty_like(src)
    layout, size = canonical_layout([4, 4], DType.F16)
    payload = bytes(range(256)) * (size // 256)
    calls = []
    runtime = ModuleType("torch_spyre._C")
    from fastsafetensors.frameworks._torch_spyre import TorchSpyreOp

    monkeypatch.setattr(TorchSpyreOp, "tensor_layout", lambda self, t: layout)
    monkeypatch.setattr(TorchSpyreOp, "tensor_nbytes", lambda self, t: size)

    def encode(work):
        calls.append(len(work))
        for buf, source, shape, dtype in work:
            assert bytes(source) == src.view(torch.uint8).numpy().tobytes()
            assert shape == [4, 4]
            assert dtype == "F16"
            buf[:] = payload

    runtime.convert_buffers = encode
    monkeypatch.setitem(sys.modules, "torch_spyre._C", runtime)
    monkeypatch.setattr(
        "fastsafetensors.frameworks._torch_spyre.backend", lambda: runtime
    )
    paths = ParallelSaver(
        num_shards=num_shards,
        fill=native_fill,
        shard_metadata=native_shard_metadata,
    ).save_entries(
        [
            native_entry("w", src, dst, nbytes=len(payload), layout=layout),
        ],
        str(tmp_path / "native"),
    )
    seen = set()
    for path in paths:
        meta = SafeTensorsMetadata.from_file(path, framework)
        assert native_format(meta) == (NATIVE_FORMAT if meta.tensors else None)
        specs = native_specs(meta)
        with safe_open(path, framework="pt") as checkpoint:
            metadata = checkpoint.metadata()
            expected_specs = {}
            names = set(checkpoint.keys())
            if "w" in names:
                expected_specs["w"] = {
                    "logical_dtype": "F16",
                    "logical_shape": [4, 4],
                }
                assert checkpoint.get_tensor("w").numpy().tobytes() == payload
                assert specs["w"].dtype == DType.F16
                assert specs["w"].nbytes == len(payload)
            if expected_specs:
                assert metadata["profile"] == "spyre.native.v1"
                assert json.loads(metadata["spyre.overrides"]) == {
                    "tensors": expected_specs
                }
            assert set(specs) == names
            seen.update(names)
    assert seen == {"w"}
    assert len(calls) == 1 and 1 <= calls[0] <= 64


@pytest.mark.parametrize("size", [0, -1])
def test_invalid_native_size(size):
    t = torch.ones(2, dtype=torch.float16)
    with pytest.raises(ValueError, match="must be positive"):
        native_entry("w", t, t, nbytes=size, layout={})


def test_native_entry_rejects_nonconforming_size():
    t = torch.ones(2, dtype=torch.float16)
    nbytes = (4 << 30) + 1
    with pytest.raises(NativeFormatError, match="does not conform"):
        native_entry("w", t, t, nbytes=nbytes, layout={})


def test_plain_fill_and_metadata_do_not_import_spyre_or_vllm(tmp_path, monkeypatch):
    original_import = builtins.__import__

    def no_backend(name, *args, **kwargs):
        if name.split(".")[0] in {"torch_spyre", "vllm", "spyre_inference"}:
            raise AssertionError(f"unexpected optional import: {name}")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_backend)
    native_fill([])
    tensor = torch.arange(6.0).reshape(2, 3).t()
    assert native_shard_metadata([]) == {}
    paths = ParallelSaver(
        num_shards=1, fill=native_fill, shard_metadata=native_shard_metadata
    ).save({"plain": tensor}, str(tmp_path / "plain"))
    with safe_open(paths[0], framework="pt") as checkpoint:
        assert "profile" not in checkpoint.metadata()
        torch.testing.assert_close(checkpoint.get_tensor("plain"), tensor)


def test_native_fill_propagates_conversion_error_and_cleans_files(
    tmp_path, monkeypatch
):
    runtime = ModuleType("torch_spyre._C")

    def fail(*args, **kwargs):
        raise RuntimeError("conversion failed")

    runtime.convert_buffers = fail
    monkeypatch.setitem(sys.modules, "torch_spyre._C", runtime)
    monkeypatch.setattr(
        "fastsafetensors.frameworks._torch_spyre.backend", lambda: runtime
    )
    tensor = torch.zeros(4, dtype=torch.float16)
    from fastsafetensors.frameworks._torch_spyre import TorchSpyreOp

    monkeypatch.setattr(
        TorchSpyreOp,
        "tensor_layout",
        lambda self, t: canonical_layout([4], DType.F16)[0],
    )
    monkeypatch.setattr(TorchSpyreOp, "tensor_nbytes", lambda self, t: 128)
    saver = ParallelSaver(
        num_shards=2, fill=native_fill, shard_metadata=native_shard_metadata
    )
    with pytest.raises(RuntimeError, match="conversion failed"):
        saver.save_entries(
            [
                native_entry(
                    "w",
                    tensor,
                    tensor,
                    nbytes=128,
                    layout=canonical_layout([4], DType.F16)[0],
                )
            ],
            str(tmp_path / "failed"),
        )
    assert list(tmp_path.iterdir()) == []


@pytest.fixture
def emulation(monkeypatch):
    from fastsafetensors import frameworks

    framework = TorchOp()
    ops = framework.get_native_ops(Device.from_str("spyre:0"))
    allocate = framework.get_empty_tensor
    references = []

    def empty(shape, dtype, device):
        gc.collect()
        assert sum(ref() is not None for ref in references) == 0
        assert device == Device(), "converter must not allocate a device tensor"
        tensor = allocate(shape, dtype, Device())
        references.append(weakref.ref(tensor))
        return tensor

    monkeypatch.setattr(framework, "get_empty_tensor", empty)
    monkeypatch.setattr(frameworks, "get_framework_op", lambda name: framework)
    monkeypatch.setattr(framework, "get_native_ops", lambda device: ops)
    monkeypatch.setattr(ops, "load_library", lambda: None)
    monkeypatch.setattr(ops, "device", lambda device: Device())
    monkeypatch.setattr(
        ops,
        "tensor_nbytes",
        lambda tensor: canonical_layout(tensor.get_shape(), tensor.dtype)[1],
    )
    monkeypatch.setattr(
        ops,
        "tensor_layout",
        lambda tensor: canonical_layout(tensor.get_shape(), tensor.dtype)[0],
    )
    monkeypatch.setattr(
        ops, "empty_tensor", lambda spec, device: empty(spec.shape, spec.dtype, device)
    )

    def encode(buffer, src, shape, dtype):
        # Independent coordinate mapping, not the descriptor/tile implementation.
        raw = src.get_raw().reshape(-1).view(torch.uint8).numpy().tobytes()
        physical = canonical_layout(shape, dtype)[0]["device_size"]
        width = 2 if dtype.value in {"F16", "BF16"} else 4
        stick = 128 // width
        buffer[:] = bytes(len(buffer))
        coords = itertools.product(*(range(n) for n in shape)) if shape else [()]
        for i, index in enumerate(coords):
            axes = (
                list(index[1:-1]) + [index[-1] // stick, index[0], index[-1] % stick]
                if len(shape) > 1
                else [index[0] // stick, index[0] % stick] if shape else [0, 0]
            )
            offset = 0
            for coordinate, extent in zip(axes, physical):
                offset = offset * extent + coordinate
            buffer[offset * width : (offset + 1) * width] = raw[
                i * width : (i + 1) * width
            ]

    monkeypatch.setattr(ops, "encode_buffer", encode)
    return framework, ops


def test_conversion_streams_tensors_preserves_metadata_and_casts(tmp_path, emulation):
    source = tmp_path / "source"
    source.mkdir()
    tensors = {
        "a": torch.arange(15).reshape(3, 5).bfloat16(),
        "b": torch.arange(9).bfloat16(),
    }
    save_file(tensors, source / "model.safetensors")
    completed = []
    paths = convert_to_spyre(
        str(source),
        str(tmp_path / "native"),
        num_shards=2,
        progress=lambda name, size: completed.append((name, size)),
    )
    assert len(paths) == 2
    assert {name for name, _ in completed} == tensors.keys()
    for path in paths:
        meta = SafeTensorsMetadata.from_file(path, emulation[0])
        specs = native_specs(meta)
        payloads = load_file(path)
        for name, spec in specs.items():
            assert spec.shape == list(tensors[name].shape)
            shape = spec.shape
            if len(shape) == 2:
                restored = payloads[name].view(torch.float16).reshape(-1, shape[0], 64)
                restored = restored.permute(1, 0, 2).reshape(shape[0], -1)[
                    :, : shape[1]
                ]
            else:
                restored = payloads[name].view(torch.float16)[: tensors[name].numel()]
            assert torch.equal(restored, tensors[name].half())
    with pytest.raises(FileExistsError, match="already exists"):
        convert_to_spyre(str(source), str(tmp_path / "native"), num_shards=2)


def test_conversion_failure_removes_temporary_outputs(tmp_path, emulation, monkeypatch):
    path = tmp_path / "in.safetensors"
    save_file({"w": torch.ones(4)}, path)

    def fail(*args):
        raise RuntimeError("conversion failed")

    monkeypatch.setattr(emulation[1], "encode_buffer", fail)
    with pytest.raises(RuntimeError, match="conversion failed"):
        convert_to_spyre(str(path), str(tmp_path / "out"), num_shards=1)
    assert not list(tmp_path.glob("out*"))


@pytest.mark.parametrize("shape", [[], [4097], [71, 129], [3, 2, 5, 131], [129, 2]])
def test_conversion_bounds_reads_casts_and_matches_physical_order(
    tmp_path, emulation, monkeypatch, shape
):
    source = (torch.arange(math.prod(shape), dtype=torch.float64) % 23).reshape(shape)
    path = tmp_path / "in.safetensors"
    save_file({"w": source}, path)
    framework, ops = emulation
    allocate, encode = framework.get_empty_tensor, ops.encode_buffer
    peak = []

    def bounded(shape, dtype, device):
        assert math.prod(shape) * framework.get_dtype_size(dtype) <= 512
        return allocate(shape, dtype, device)

    def tile(output, src, shape, dtype):
        assert len(output) <= 512 and src.get_nbytes() <= 512
        peak.append(len(output))
        encode(output, src, shape, dtype)

    monkeypatch.setattr(framework, "get_empty_tensor", bounded)
    monkeypatch.setattr(ops, "encode_buffer", tile)
    paths = convert_to_spyre(
        str(path), str(tmp_path / "out"), num_shards=1, workspace_bytes=512
    )
    actual = load_file(paths[0])["w"].numpy().tobytes()
    size = canonical_layout(shape, DType.F16)[1]
    expected = memoryview(bytearray(size))
    encode(expected, framework.wrap_tensor(source.half()), shape, DType.F16)
    assert actual == expected.tobytes()
    assert peak and (len(peak) > 1 or not shape)


def test_checkpoint_paths_index_and_rejections(tmp_path):
    shard = tmp_path / "one.safetensors"
    save_file({"w": torch.ones(1)}, shard)
    index = tmp_path / "model.safetensors.index.json"
    index.write_text(json.dumps({"weight_map": {"a": shard.name, "b": shard.name}}))
    assert checkpoint_paths(str(tmp_path)) == [str(shard)]
    # HF snapshots symlink both the index and shards to extensionless blobs.
    blob = tmp_path / "blob"
    index.rename(blob)
    index.symlink_to(blob.name)
    assert checkpoint_paths(str(tmp_path)) == [str(shard)]
    index.write_text(json.dumps({"weight_map": {"a": "../outside.safetensors"}}))
    with pytest.raises(ValueError, match="unsafe"):
        checkpoint_paths(str(index))
    with pytest.raises(FileNotFoundError):
        checkpoint_paths(str(tmp_path / "missing"))
