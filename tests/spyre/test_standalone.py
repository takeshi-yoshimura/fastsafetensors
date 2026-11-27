# SPDX-License-Identifier: Apache-2.0
"""Optional hardware checks for pinned range DMA; not run in CPU preflight."""

import gc
import os
import weakref
from concurrent.futures import ThreadPoolExecutor

import pytest

if os.environ.get("TEST_FASTSAFETENSORS_SPYRE") != "1":
    pytest.skip("set TEST_FASTSAFETENSORS_SPYRE=1 on Spyre", allow_module_level=True)

import torch
import torch_spyre
import torch_spyre._C
import torch_spyre.streams
from safetensors.torch import save_file

from fastsafetensors import ParallelLoader
from fastsafetensors.frameworks import get_framework_op
from fastsafetensors.native import convert_to_spyre
from fastsafetensors.st_types import Device


@pytest.fixture
def ops(monkeypatch):
    ops = get_framework_op("pytorch").get_native_ops(Device.from_str("spyre:0"))
    ops.load_library()

    def forbidden(*args, **kwargs):
        raise AssertionError("legacy torch-spyre transfer wrapper called")

    for name in ("dma_bytes", "copy_raw_h2d_batch", "convert_h2d_into"):
        monkeypatch.setattr(torch_spyre._C, name, forbidden, raising=False)
    return ops


def test_partial_dma_and_validate_batch_before_first_launch(ops):
    dst = torch.empty((4, 64), dtype=torch.float16, device="spyre:0")
    target = ops.as_tensor(dst)
    size = ops.tensor_nbytes(target)
    host = ops.allocate_host_buffer(4096)
    assert host.get_raw().is_pinned()
    encoded = []
    for value in (1, 2):
        data = bytearray(size)
        src = ops.as_tensor(torch.full((4, 64), value, dtype=torch.float16))
        ops.encode_tensors([(memoryview(data), src, target)])
        encoded.append(data)
    ops.host_buffer_view(host)[:size] = encoded[0]
    ops.copy_raw_tensors([(host, target, 0, size)])
    for offset in (0, 128, size - 128):
        ops.host_buffer_view(host)[:size] = encoded[1]
        ops.copy_raw_tensors([(host, target, offset, 128)])
        got = dst.cpu()
        assert (got == 2).sum().item() == 64
        assert (got == 1).sum().item() == 192
        ops.host_buffer_view(host)[:size] = encoded[0]
        ops.copy_raw_tensors([(host, target, offset, 128)])
    ops.host_buffer_view(host)[:size] = encoded[1]
    for offset, length in ((1, 128), (0, 127), (size, 128), (0, 0)):
        with pytest.raises(RuntimeError, match="range"):
            ops.copy_raw_tensors(
                [(host, target, 0, 128), (host, target, offset, length)]
            )
        assert torch.equal(dst.cpu(), torch.ones_like(dst.cpu()))


def test_mapped_storage_views_keep_owner_and_support_source_offsets(ops):
    dst = torch.empty((4, 64), dtype=torch.float16, device="spyre:0")
    target = ops.as_tensor(dst)
    size = ops.tensor_nbytes(target)
    host = ops.allocate_host_buffer(8192)
    raw_ref = weakref.ref(host.get_raw())
    view = host[4096:8192]
    assert view.get_nbytes() == 4096
    del host
    gc.collect()
    assert view.get_raw().is_pinned()
    # A tensor view owns the same storage even if the original TensorImpl dies.
    assert view.get_raw().untyped_storage().nbytes() == 8192
    src = ops.as_tensor(torch.full((4, 64), 3, dtype=torch.float16))
    encoded = bytearray(size)
    ops.encode_tensors([(memoryview(encoded), src, target)])
    ops.host_buffer_view(view)[:size] = encoded
    ops.copy_raw_tensors([(view, target, 0, size)])
    torch.testing.assert_close(dst.cpu(), torch.full((4, 64), 3, dtype=torch.float16))
    del view
    gc.collect()
    assert raw_ref() is None


def test_pageable_sources_are_rejected_without_staging_fallback(ops):
    import mmap

    dst = torch.empty((4, 64), dtype=torch.float16, device="spyre:0")
    target = ops.as_tensor(dst)
    size = ops.tensor_nbytes(target)
    host = ops.host_tensor_from_buffer(mmap.mmap(-1, 4096))
    assert not host.get_raw().is_pinned()
    src = ops.as_tensor(torch.full((4, 64), 4, dtype=torch.float16))
    encoded = bytearray(size)
    ops.encode_tensors([(memoryview(encoded), src, target)])
    ops.host_buffer_view(host)[:size] = encoded
    with pytest.raises(RuntimeError, match="pinned buffer"):
        ops.copy_raw_tensors([(host, target, 0, size)])


def test_convert_then_load_tensor_much_larger_than_two_slots(
    ops, tmp_path, monkeypatch
):
    source = tmp_path / "source.safetensors"
    cpu = torch.arange(1024 * 256, dtype=torch.float32).reshape(1024, 256) / 65536
    save_file({"weight": cpu.bfloat16()}, source)
    paths = convert_to_spyre(
        str(source), str(tmp_path / "native"), num_shards=1, workspace_bytes=4096
    )
    allocations = []
    allocate = type(ops).allocate_host_buffer

    def track(self, size):
        allocations.append(size)
        return allocate(self, size)

    monkeypatch.setattr(type(ops), "allocate_host_buffer", track)
    loader = ParallelLoader(
        None, paths, device="spyre:0", all_local=True, set_numa=False, bbuf_size_kb=8
    )
    try:
        result = dict(loader.iterate_weights())["weight"]
    finally:
        loader.close()
    expected = cpu.bfloat16().half().to("spyre:0").cpu()
    torch.testing.assert_close(result.cpu(), expected, rtol=0, atol=0)
    assert allocations == [4096, 4096]


def test_page_range_with_128_byte_tail_on_captured_nondefault_stream(ops):
    dst = torch.empty((33, 64), dtype=torch.float16, device="spyre:0")
    target = ops.as_tensor(dst)
    size = ops.tensor_nbytes(target)
    assert size == 4096 + 128
    src = ops.as_tensor(torch.arange(33 * 64, dtype=torch.float16).reshape(33, 64))
    encoded = bytearray(size)
    ops.encode_tensors([(memoryview(encoded), src, target)])
    host = ops.allocate_host_buffer(4096)
    stream = torch_spyre.streams.Stream()
    with stream:
        copy = ops.prepare_copy(Device.from_str("spyre:0"))
    # Run after leaving the stream context, on another thread like the reader.
    with ThreadPoolExecutor(max_workers=1) as pool:
        for offset, length in ((0, 4096), (4096, 128)):
            ops.host_buffer_view(host)[:length] = encoded[offset : offset + length]
            ticket = pool.submit(copy, [(host, target, offset, length)]).result()
            # The same source slot must not be overwritten before completion.
            assert ticket is not None
            pool.submit(ticket.wait).result()
    expected = src.get_raw().to("spyre:0").cpu()
    torch.testing.assert_close(dst.cpu(), expected, rtol=0, atol=0)
