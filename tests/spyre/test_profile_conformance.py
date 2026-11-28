# SPDX-License-Identifier: Apache-2.0
"""Byte-level fixtures for the candidate profile, independent of shape derivation."""

import itertools
import math
import os
import struct

import pytest

if os.environ.get("TEST_FASTSAFETENSORS_SPYRE") != "1":
    pytest.skip("set TEST_FASTSAFETENSORS_SPYRE=1 on Spyre", allow_module_level=True)

import torch
from torch_spyre._C import SpyreTensorLayout

from fastsafetensors.frameworks import get_framework_op
from fastsafetensors.frameworks._torch_spyre import layout_to_json
from fastsafetensors.native import NativeTensorSpec, canonical_layout, dense_stride
from fastsafetensors.st_types import Device, DType


@pytest.mark.parametrize("shape", [[], [1], [65], [1, 65], [2, 1, 3], [2, 3, 65]])
@pytest.mark.parametrize("dtype", [DType.F16, DType.BF16, DType.F32, DType.I32])
def test_fixed_layout_and_numeric_payload(shape, dtype, record_property):
    from fastsafetensors.frameworks._torch import dtype_convert

    op = get_framework_op("pytorch").get_native_ops(Device.from_str("spyre:0"))
    tdtype = dtype_convert[dtype]
    layout, size = canonical_layout(shape, dtype)
    assert layout_to_json(SpyreTensorLayout(shape, tdtype)) == layout
    spec = NativeTensorSpec("w", dtype, shape, dense_stride(shape), layout, size)
    source = torch.arange(max(1, math.prod(shape)), dtype=torch.float32).reshape(shape)
    source = source.to(tdtype)
    destination = op.empty_tensor(spec, "spyre:0")
    payloads = []
    for poison in (0x35, 0xA7):
        buffer = bytearray([poison]) * size
        op.encode_tensors([(memoryview(buffer), op.as_tensor(source), destination)])
        payloads.append(bytes(buffer))
    assert (
        payloads[0] == payloads[1]
    ), "conversion must define padding, not leak scratch bytes"
    host = op.allocate_host_buffer(size)
    op.host_buffer_view(host)[:] = payloads[0]
    op.copy_raw_tensors([(host, destination)])
    torch.testing.assert_close(op.unwrap(destination).cpu(), source, rtol=0, atol=0)
    if dtype in (DType.F32, DType.I32):
        unit = "<f" if dtype == DType.F32 else "<i"
        stick = 32
        physical = layout["device_size"]
        expected = bytearray(size)
        indices = itertools.product(*(range(n) for n in shape)) if shape else [()]
        for index in indices:
            axes = (
                list(index[1:-1]) + [index[-1] // stick, index[0], index[-1] % stick]
                if len(shape) > 1
                else [index[0] // stick, index[0] % stick] if shape else [0, 0]
            )
            offset = 0
            for axis, extent in zip(axes, physical):
                offset = offset * extent + axis
            struct.pack_into(unit, expected, offset * 4, source[index].item())
        assert payloads[0] == expected
    record_property("payload_hex_prefix", payloads[0][:128].hex())


@pytest.mark.parametrize("dtype", [DType.F16, DType.BF16])
def test_dlfloat16_numeric_encoding_fixture(dtype, record_property):
    from fastsafetensors.frameworks._torch import dtype_convert

    op = get_framework_op("pytorch").get_native_ops(Device.from_str("spyre:0"))
    # DLFloat16: sign / six exponent bits (bias 31) / nine fraction bits.
    source = torch.tensor(
        [0.0, 1.0, -1.0, 0.5, 2.0, 1.5, -2.0], dtype=dtype_convert[dtype]
    )
    expected = struct.pack(
        "<7H", 0x0000, 0x3E00, 0xBE00, 0x3C00, 0x4000, 0x3F00, 0xC000
    )
    layout, size = canonical_layout([7], dtype)
    destination = op.empty_tensor(
        NativeTensorSpec("w", dtype, [7], [1], layout, size), "spyre:0"
    )
    buffer = bytearray(size)
    op.encode_tensors([(memoryview(buffer), op.as_tensor(source), destination)])
    assert buffer[:14] == expected
    assert buffer[14:] == bytes(size - 14)
    record_property("dlfloat16_little_endian_hex", bytes(buffer).hex())


def test_i64_stays_portable(tmp_path):
    """Unsupported native dtypes stay in a separate ordinary safetensors file."""
    from safetensors import safe_open

    from fastsafetensors.common import SafeTensorsMetadata
    from fastsafetensors.native import (
        native_fill,
        native_format,
        native_shard_metadata,
        split_native_entries,
    )
    from fastsafetensors.saver import ParallelSaver, plain_entry

    with pytest.raises(ValueError):
        canonical_layout([65], DType.I64)
    source = torch.arange(65, dtype=torch.int64)
    native, portable = split_native_entries([plain_entry("w", source)])
    assert not native and len(portable) == 1
    paths = ParallelSaver(
        num_shards=1,
        fill=native_fill,
        shard_metadata=native_shard_metadata,
    ).save_entries(portable, str(tmp_path / "portable"))
    metadata = SafeTensorsMetadata.from_file(paths[0], get_framework_op("pytorch"))
    assert native_format(metadata) is None
    with safe_open(paths[0], framework="pt") as checkpoint:
        torch.testing.assert_close(checkpoint.get_tensor("w"), source, rtol=0, atol=0)
