# SPDX-License-Identifier: Apache-2.0
"""Static admission and fixed-layout fixtures; no runtime or hardware required."""

import json

import pytest
import torch
from safetensors import safe_open

from fastsafetensors.common import SafeTensorsMetadata
from fastsafetensors.frameworks import get_framework_op
from fastsafetensors.native import (
    PROFILE,
    NativeFormat,
    NativeFormatError,
    NativeTensorSpec,
    canonical_layout,
    dense_stride,
    native_format,
    native_metadata,
    native_specs,
)
from fastsafetensors.saver import ParallelSaver
from fastsafetensors.st_types import DType


def metadata(descriptor=None, *, raw=None, carrier_bytes=49152, extra=None):
    md = {
        "profile": PROFILE,
        "spyre.overrides": (
            json.dumps(
                {
                    "tensors": {
                        "w": (
                            descriptor
                            if descriptor is not None
                            else {"logical_shape": [128, 150]}
                        )
                    }
                }
            )
            if raw is None
            else raw
        ),
    }
    md.update(extra or {})
    header = json.dumps(
        {
            "__metadata__": md,
            "w": {
                "dtype": "U8",
                "shape": [carrier_bytes],
                "data_offsets": [0, carrier_bytes],
            },
        }
    )
    return SafeTensorsMetadata(
        header,
        len(header) + 8,
        len(header) + 8 + carrier_bytes,
        get_framework_op("pytorch"),
    )


@pytest.mark.parametrize(
    "shape,sizes,strides,valid",
    [
        ([], [1, 64], [0, 0], [[[0, 1], 1]]),
        ([65], [2, 64], [64, 1], [[[0, 1], 65]]),
        ([2, 65], [2, 2, 64], [64, 65, 1], [[[0, 2], 65], [[1], 2]]),
        (
            [2, 3, 65],
            [3, 2, 2, 64],
            [65, 64, 195, 1],
            [[[0], 3], [[1, 3], 65], [[2], 2]],
        ),
        ([2, 1, 3], [1, 1, 2, 64], [0, 3, 3, 1], [[[0], 1], [[1, 3], 3], [[2], 2]]),
    ],
)
def test_frozen_layout_geometry(shape, sizes, strides, valid):
    layout, size = canonical_layout(shape, DType.F16)
    assert layout == {
        "device_size": sizes,
        "stride_map": strides,
        "valid_elements": valid,
        "device_dtype": "SEN169_FP16",
        "element_arrangement": "STANDARD",
    }
    product = 2
    for extent in sizes:
        product *= extent
    assert size == product


def test_paper_example_and_minimal_metadata():
    spec = native_specs(metadata())["w"]
    assert spec.dtype == DType.BF16 and spec.nbytes == 49152
    result = native_metadata(NativeFormat("spyre", PROFILE), {"w": spec})
    assert set(result) == {"profile", "spyre.overrides"}
    assert json.loads(result["spyre.overrides"]) == {
        "tensors": {"w": {"logical_shape": [128, 150]}}
    }
    f16 = NativeTensorSpec(
        "w", DType.F16, spec.shape, spec.stride, spec.layout, spec.nbytes
    )
    assert (
        json.loads(
            native_metadata(NativeFormat("spyre", PROFILE), {"w": f16})[
                "spyre.overrides"
            ]
        )["tensors"]["w"]["logical_dtype"]
        == "F16"
    )


@pytest.mark.parametrize(
    "descriptor",
    [
        {},
        {"logical_shape": [True]},
        {"logical_shape": [1.0]},
        {"logical_shape": [-1]},
        {"logical_shape": [0]},
        {"logical_shape": [2**31]},
        {"logical_shape": [1] * 17},
        {"logical_shape": [2**30, 64]},
        {"logical_shape": [128, 150], "stride": [150, 1]},
        {"logical_shape": [128, 150], "layout": {}},
        {"logical_shape": [128, 150], "format": None},
        {"logical_shape": [128, 150], "logical_dtype": None},
        {"logical_shape": [128, 150], "layout": "qfp8wt.v1"},
        {"logical_shape": [128, 150], "format": "ieee-fp32"},
        {"logical_shape": [128, 150], "logical_dtype": "BOOL"},
        {"logical_shape": [65], "logical_dtype": "I64", "format": "int32"},
        {"logical_shape": [128, 150], "logical_dtype": "F8_E4M3"},
    ],
)
def test_malformed_descriptors_rejected(descriptor):
    with pytest.raises(NativeFormatError):
        native_specs(metadata(descriptor))


@pytest.mark.parametrize("shape", [[], [1], [65], [1, 65], [2, 1, 3], [2, 3, 65]])
def test_unqualified_i64_rejected_before_runtime(shape):
    with pytest.raises(NativeFormatError, match="unsupported native dtype"):
        canonical_layout(shape, DType.I64)


@pytest.mark.parametrize(
    "raw",
    [
        "[]",
        "{}",
        '{"tensors":{}}',
        '{"tensors":{},"module":"evil"}',
        '{"tensors":{"w":{"logical_shape":[128,150],"logical_shape":[1]}}}',
        '{"tensors":{"w":{"logical_shape":[NaN]}}}',
        '{"target":{"tp_size":true,"tp_rank":0},"tensors":{}}',
        '{"target":{"tp_size":1,"tp_rank":1},"tensors":{}}',
        '{"target":{"tp_size":1},"tensors":{}}',
        "[" * 33 + "0" + "]" * 33,
    ],
)
def test_invalid_envelope_rejected(raw):
    with pytest.raises(NativeFormatError):
        native_specs(metadata(raw=raw))


def test_profile_ownership_size_and_tp():
    for extra in (
        {"profile": "unknown"},
        {"spyre.foo": "x"},
        {"fst.native.version": "1"},
        {"torch_spyre.moved": "{}"},
    ):
        with pytest.raises(NativeFormatError):
            native_specs(metadata(extra=extra))
    with pytest.raises(NativeFormatError, match="carrier size"):
        native_specs(metadata(carrier_bytes=128))
    with pytest.raises(NativeFormatError, match="TP target"):
        native_specs(metadata(), tp_size=4, tp_rank=0)
    native_specs(metadata(), tp_size=1, tp_rank=0)


@pytest.mark.parametrize(
    "text",
    [
        '{"w":{},"w":{}}',
        '{"x":Infinity}',
        '{"x":1e999}',
        '{"x":-1e999}',
        '{"x":{"y":0,"y":1}}',
    ],
)
def test_outer_json_strict(text):
    with pytest.raises(ValueError):
        SafeTensorsMetadata(
            text, len(text) + 8, len(text) + 8, get_framework_op("pytorch")
        )


def test_legacy_reader_extracts_carrier(tmp_path):
    from fastsafetensors import ParallelSaver
    from fastsafetensors.saver import WriteEntry

    layout, size = canonical_layout([1, 65], DType.F16)
    spec = NativeTensorSpec("w", DType.F16, [1, 65], [65, 1], layout, size)

    def fill(work):
        for buffer, _ in work:
            buffer[:] = bytes(size)

    paths = ParallelSaver(num_shards=1, fill=fill).save_entries(
        [WriteEntry("w", DType.U8, [size], size, None)],
        str(tmp_path / "native"),
        metadata=native_metadata(NativeFormat("spyre", PROFILE), {"w": spec}),
    )
    with safe_open(paths[0], framework="pt") as reader:
        assert reader.get_tensor("w").numel() == size
        assert reader.metadata()["profile"] == PROFILE


@pytest.mark.parametrize("parallel", [False, True])
def test_all_files_admitted_before_any_allocation(tmp_path, monkeypatch, parallel):
    from fastsafetensors import ParallelLoader, SafeTensorsFileLoader
    from fastsafetensors.copier.spyre import new_spyre_copier
    from fastsafetensors.frameworks._torch_spyre import TorchSpyreOp
    from fastsafetensors.st_types import Device

    paths = []
    for i, target in enumerate((1, 4)):
        text = json.dumps(
            {
                "__metadata__": {
                    "profile": PROFILE,
                    "spyre.overrides": json.dumps(
                        {
                            "target": {"tp_size": target, "tp_rank": 0},
                            "tensors": {f"w{i}": {"logical_shape": [1]}},
                        }
                    ),
                },
                f"w{i}": {"dtype": "U8", "shape": [128], "data_offsets": [0, 128]},
            }
        ).encode()
        path = tmp_path / f"{i}.safetensors"
        path.write_bytes(len(text).to_bytes(8, "little") + text + bytes(128))
        paths.append(str(path))
    monkeypatch.setattr(TorchSpyreOp, "load_library", lambda self: None)
    monkeypatch.setattr(
        TorchSpyreOp,
        "empty_tensor",
        lambda *a: pytest.fail("allocation before admission"),
    )
    monkeypatch.setattr(
        TorchSpyreOp, "copy_raw_tensors", lambda *a: pytest.fail("DMA before admission")
    )
    if parallel:
        with pytest.raises(NativeFormatError, match="TP target"):
            ParallelLoader(
                None,
                paths,
                device="spyre:0",
                all_local=True,
                set_numa=False,
                max_batch_bytes=128,
            )
    else:
        loader = SafeTensorsFileLoader(None, "spyre:0", set_numa=False)
        try:
            loader.add_filenames({0: paths})
            with pytest.raises(NativeFormatError, match="TP target"):
                loader.copy_files_to_device()
        finally:
            loader.close()
    constructor = new_spyre_copier(Device.from_str("spyre:0"))
    from fastsafetensors import native

    monkeypatch.setattr(native, "MAX_TOTAL_BYTES", 200)
    meta = metadata({"logical_shape": [1]}, carrier_bytes=128)
    with pytest.raises(NativeFormatError, match="aggregate"):
        constructor.validate_metadata([meta, meta])


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
    "key",
    [
        "fst.native.version",
        "fst.native.device",
        "fst.native.format",
        "fst.native.tensors",
    ],
)
def test_legacy_metadata_requires_regeneration(plain_meta, key):
    plain_meta.metadata[key] = "1"
    with pytest.raises(NativeFormatError, match="legacy.*regenerate"):
        native_format(plain_meta)


@pytest.mark.parametrize(
    "md",
    [
        {"profile": "spyre.native.v2"},
        {"profile": "spyre.native.v1"},
        {"spyre.overrides": "{}"},
        {"spyre.unknown": "1"},
    ],
)
def test_incomplete_or_unknown_profile_is_not_portable(plain_meta, md):
    plain_meta.metadata.update(md)
    with pytest.raises(NativeFormatError):
        native_specs(plain_meta)
