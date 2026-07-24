# SPDX-License-Identifier: Apache-2.0

"""Lifetime / ownership tests for the shared DLPack allocation.

The contract under test: a tensor returned from FilesBufferOnDevice retains
shared ownership of its backing device allocation, so it stays valid after
FilesBufferOnDevice.close(). The physical buffer is released only once the
buffer and every exported tensor are gone.
"""

import gc
from typing import Tuple

import pytest

from fastsafetensors import SafeTensorsFileLoader, SingleGroup
from fastsafetensors.allocation import SharedDeviceAllocation
from fastsafetensors.common import is_gpu_found
from fastsafetensors.frameworks import FrameworkOpBase
from fastsafetensors.st_types import Device, DType


def _device(framework: FrameworkOpBase) -> Device:
    if is_gpu_found():
        dev = "cuda:0" if framework.get_name() == "pytorch" else "gpu:0"
    else:
        dev = "cpu"
    return Device.from_str(dev)


def _open_first_tensor(
    framework: FrameworkOpBase, input_files, nogds: bool = True
) -> Tuple[SafeTensorsFileLoader, "object", str]:
    device = _device(framework)
    loader = SafeTensorsFileLoader(
        SingleGroup(), device.as_str(), nogds=nogds, framework=framework.get_name()
    )
    loader.add_filenames({0: [input_files[0]]})
    fb = loader.copy_files_to_device()
    key = loader.get_keys()[0]
    return loader, fb, key


# --------------------------------------------------------------------------- #
# SharedDeviceAllocation unit behavior (no framework/device needed)
# --------------------------------------------------------------------------- #


class _FakeGbuf:
    def __init__(self, addr: int, length: int):
        self._addr = addr
        self._length = length

    def get_base_address(self) -> int:
        return self._addr

    def get_length(self) -> int:
        return self._length


class _RecordingFramework:
    def __init__(self):
        self.freed = []

    def free_tensor_memory(self, gbuf, device):
        self.freed.append(gbuf)


def test_allocation_frees_exactly_once_after_last_reference():
    fw = _RecordingFramework()
    gbuf = _FakeGbuf(0x1000, 128)
    alloc = SharedDeviceAllocation(gbuf, fw, Device.from_str("cpu"))
    assert alloc.refcount() == 1 and alloc.live

    alloc.acquire()  # simulate an exported tensor
    alloc.acquire()  # and another
    assert alloc.refcount() == 3

    alloc.release()  # one tensor gone
    alloc.release()  # buffer-side (factory) gone
    assert fw.freed == [] and alloc.live  # last tensor still holds it

    alloc.release()  # final reference
    assert fw.freed == [gbuf] and not alloc.live

    # Idempotent past zero: extra releases must not double-free.
    alloc.release()
    alloc.release()
    assert fw.freed == [gbuf]


def test_allocation_acquire_after_free_raises():
    fw = _RecordingFramework()
    alloc = SharedDeviceAllocation(_FakeGbuf(0x2000, 64), fw, Device.from_str("cpu"))
    alloc.release()
    assert not alloc.live
    with pytest.raises(RuntimeError):
        alloc.acquire()


def test_allocation_non_owning_never_frees():
    fw = _RecordingFramework()
    alloc = SharedDeviceAllocation(
        _FakeGbuf(0x3000, 64), fw, Device.from_str("cpu"), owns_memory=False
    )
    alloc.release()
    assert fw.freed == []  # DummyDeviceBuffer-style: nothing to free


# --------------------------------------------------------------------------- #
# End-to-end lifetime through the loader
# --------------------------------------------------------------------------- #


def test_single_group_tensor_valid_after_close(fstcpp_log, input_files, framework):
    """A zero-copy SingleGroup tensor must survive fb.close()."""
    loader, fb, key = _open_first_tensor(framework, input_files)
    t = fb.get_tensor_wrapped(key)
    expected = t.clone().get_raw()  # independent snapshot

    fb.close()
    loader.close()

    # The returned tensor and its data are still valid after close.
    assert framework.is_equal(t, expected)


def test_view_and_reshape_keep_allocation_alive(fstcpp_log, input_files, framework):
    loader, fb, key = _open_first_tensor(framework, input_files)
    t = fb.get_tensor_wrapped(key)
    raw = t.get_raw()
    expected = raw.clone()
    flat = raw.reshape([-1])  # derived view sharing storage

    fb.close()
    loader.close()
    del t, raw
    gc.collect()

    # The view still references live memory after the source tensor is gone.
    if framework.get_name() == "pytorch":
        import torch

        assert torch.equal(flat.reshape(expected.shape), expected)
    else:
        import paddle

        assert bool(paddle.all(flat.reshape(expected.shape) == expected))


def test_allocation_released_after_final_tensor_dropped(
    fstcpp_log, input_files, framework
):
    base = framework.get_mem_used()
    loader, fb, key = _open_first_tensor(framework, input_files)
    t = fb.get_tensor_wrapped(key)

    assert framework.get_mem_used() > base  # buffer is allocated

    fb.close()
    loader.close()
    # Still alive: the exported tensor holds the buffer past close.
    assert framework.get_mem_used() > base

    del t
    gc.collect()
    # Final reference gone -> buffer released exactly back to baseline.
    assert framework.get_mem_used() == base


def test_repeated_close_no_double_free(fstcpp_log, input_files, framework):
    base = framework.get_mem_used()
    loader, fb, key = _open_first_tensor(framework, input_files)
    t = fb.get_tensor_wrapped(key)
    expected = t.clone().get_raw()

    fb.close()
    fb.close()  # idempotent
    loader.close()
    loader.close()

    assert framework.is_equal(t, expected)  # still valid despite double close

    del t, expected
    gc.collect()
    assert framework.get_mem_used() == base


def test_clone_outlives_buffer(fstcpp_log, input_files, framework):
    """An independent clone must outlive the buffer and its zero-copy tensor."""
    base = framework.get_mem_used()
    loader, fb, key = _open_first_tensor(framework, input_files)
    t = fb.get_tensor_wrapped(key)
    independent = t.clone().get_raw()
    snapshot = independent.clone()  # value captured while the buffer is alive

    fb.close()
    loader.close()
    del t
    gc.collect()
    # The zero-copy source and its buffer are gone; the clone must be unchanged.
    assert framework.get_mem_used() == base
    if framework.get_name() == "pytorch":
        import torch

        assert torch.equal(independent, snapshot)
    else:
        import paddle

        assert bool(paddle.all(independent == snapshot))
