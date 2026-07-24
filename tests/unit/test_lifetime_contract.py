# SPDX-License-Identifier: Apache-2.0

"""auto_mem_delete deprecation and live-allocation metric tests."""

import gc
import warnings

import pytest

from fastsafetensors import (
    SafeTensorsFileLoader,
    SingleGroup,
    live_allocation_bytes,
    live_allocation_count,
)
from fastsafetensors.common import is_gpu_found
from fastsafetensors.file_buffer import FilesBufferOnDevice
from fastsafetensors.frameworks import FrameworkOpBase
from fastsafetensors.st_types import Device


def _device(framework: FrameworkOpBase) -> Device:
    if is_gpu_found():
        dev = "cuda:0" if framework.get_name() == "pytorch" else "gpu:0"
    else:
        dev = "cpu"
    return Device.from_str(dev)


def _open(framework, files):
    device = _device(framework)
    loader = SafeTensorsFileLoader(
        SingleGroup(), device.as_str(), nogds=True, framework=framework.get_name()
    )
    loader.add_filenames({0: files})
    fb = loader.copy_files_to_device()
    return loader, fb


def test_auto_mem_delete_explicit_warns(fstcpp_log, framework):
    pg = framework.get_process_group(SingleGroup())
    with pytest.warns(DeprecationWarning, match="auto_mem_delete is deprecated"):
        FilesBufferOnDevice({}, pg, framework, auto_mem_delete=True)
    with pytest.warns(DeprecationWarning):
        FilesBufferOnDevice({}, pg, framework, auto_mem_delete=False)


def test_auto_mem_delete_default_does_not_warn(fstcpp_log, framework):
    pg = framework.get_process_group(SingleGroup())
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any warning becomes an error
        FilesBufferOnDevice({}, pg, framework)


def test_live_allocation_metrics_track_ownership(fstcpp_log, input_files, framework):
    c0 = live_allocation_count()
    b0 = live_allocation_bytes()

    loader, fb = _open(framework, [input_files[0]])
    # One file -> one owned allocation.
    assert live_allocation_count() == c0 + 1
    assert live_allocation_bytes() > b0

    key = fb.keys()[0]
    t = fb.get_tensor_wrapped(key)  # zero-copy, no new allocation
    assert live_allocation_count() == c0 + 1

    fb.close()
    loader.close()
    # The exported tensor still owns the allocation after close.
    assert live_allocation_count() == c0 + 1

    del t
    gc.collect()
    # Final owner gone -> metrics return exactly to baseline.
    assert live_allocation_count() == c0
    assert live_allocation_bytes() == b0


def test_live_allocation_bytes_released_by_consume(fstcpp_log, input_files, framework):
    c0 = live_allocation_count()
    b0 = live_allocation_bytes()

    loader, fb = _open(framework, [input_files[0]])
    assert live_allocation_count() == c0 + 1

    # Consume everything and drop the exported tensors.
    drained = list(fb.drain_tensors())
    assert live_allocation_count() == c0 + 1  # exported tensors still hold it
    del drained
    gc.collect()
    # No exported references and every name consumed -> allocation freed even
    # before close().
    assert live_allocation_count() == c0
    assert live_allocation_bytes() == b0

    fb.close()
    loader.close()
    assert live_allocation_count() == c0
