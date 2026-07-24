# SPDX-License-Identifier: Apache-2.0

"""Consuming-access / availability tests for get_and_remove_tensor() and keys().

Reusable access (get_tensor) leaves a name available; consuming access
(get_and_remove_tensor) hands the name over once and removes it from keys(). A name is
marked consumed only after acquisition succeeds. Metadata stays queryable after
consumption.
"""

import gc

import pytest

from fastsafetensors import SafeTensorsFileLoader, SingleGroup, TensorConsumedError
from fastsafetensors.common import is_gpu_found
from fastsafetensors.frameworks import FrameworkOpBase
from fastsafetensors.st_types import Device


def _device(framework: FrameworkOpBase) -> Device:
    if is_gpu_found():
        dev = "cuda:0" if framework.get_name() == "pytorch" else "gpu:0"
    else:
        dev = "cpu"
    return Device.from_str(dev)


def _open(framework, input_files):
    device = _device(framework)
    loader = SafeTensorsFileLoader(
        SingleGroup(), device.as_str(), nogds=True, framework=framework.get_name()
    )
    loader.add_filenames({0: [input_files[0]]})
    fb = loader.copy_files_to_device()
    return loader, fb


def test_remove_marks_name_consumed(fstcpp_log, input_files, framework):
    loader, fb = _open(framework, input_files)
    key = fb.keys()[0]

    assert key in fb.keys() and key in fb.all_keys()
    fb.get_and_remove_tensor(key)

    # keys() drops the consumed name; all_keys() (immutable metadata) keeps it.
    assert key not in fb.keys()
    assert key in fb.all_keys()

    with pytest.raises(TensorConsumedError):
        fb.get_tensor(key)
    with pytest.raises(TensorConsumedError):
        fb.get_and_remove_tensor(key)

    fb.close()
    loader.close()


def test_remove_after_get_both_valid(fstcpp_log, input_files, framework):
    loader, fb = _open(framework, input_files)
    key = fb.keys()[0]

    a = fb.get_tensor_wrapped(key)  # reusable
    b = fb.get_and_remove_tensor_wrapped(key)  # consuming after a reusable get

    # Both references remain valid; only the name is no longer available.
    assert framework.is_equal(a, b.get_raw())
    assert key not in fb.keys()

    fb.close()
    loader.close()
    assert framework.is_equal(a, b.get_raw())  # still valid after close


def test_metadata_available_after_remove(fstcpp_log, input_files, framework):
    loader, fb = _open(framework, input_files)
    key = fb.keys()[0]
    shape_before = fb.get_shape(key)
    src_before = fb.get_filename(key)

    fb.get_and_remove_tensor(key)

    # Shape / source filename lookups keep working after consumption.
    assert fb.get_shape(key) == shape_before
    assert fb.get_filename(key) == src_before

    fb.close()
    loader.close()


def test_remove_unknown_name_raises_value_error(fstcpp_log, input_files, framework):
    loader, fb = _open(framework, input_files)
    # Unknown name: plain ValueError, not TensorConsumedError.
    with pytest.raises(ValueError) as ei:
        fb.get_and_remove_tensor("does-not-exist")
    assert not isinstance(ei.value, TensorConsumedError)
    assert "does-not-exist" not in fb._consumed_keys
    fb.close()
    loader.close()


def test_failed_acquisition_does_not_consume(fstcpp_log, input_files, framework):
    loader, fb = _open(framework, input_files)
    key = fb.keys()[0]

    # Simulate a materialization failure: the call must propagate and NOT mark the
    # name consumed, leaving it available for retry.
    orig = fb.get_sharded_wrapped

    def boom(*args, **kwargs):
        raise RuntimeError("materialization failed")

    fb.get_sharded_wrapped = boom
    with pytest.raises(RuntimeError):
        fb.get_and_remove_tensor(key)
    assert key not in fb._consumed_keys
    assert key in fb.keys()

    # Retry after the failure clears: the name is still consumable.
    fb.get_sharded_wrapped = orig
    t = fb.get_and_remove_tensor(key)
    assert t is not None
    assert key not in fb.keys()

    fb.close()
    loader.close()


def test_reusable_methods_reject_consumed_name(fstcpp_log, input_files, framework):
    loader, fb = _open(framework, input_files)
    key = fb.keys()[0]
    fb.get_and_remove_tensor(key)

    from collections import OrderedDict

    with pytest.raises(TensorConsumedError):
        fb.get_sharded(key, dim=-1)
    with pytest.raises(TensorConsumedError):
        fb.get_multi_cols([key], dim=0)
    with pytest.raises(TensorConsumedError):
        fb.as_dict(OrderedDict([(key, -1)]))

    fb.close()
    loader.close()


def test_removed_tensor_released_after_drop(fstcpp_log, input_files, framework):
    base = framework.get_mem_used()
    loader, fb = _open(framework, input_files)
    key = fb.keys()[0]
    t = fb.get_and_remove_tensor_wrapped(key)

    fb.close()
    loader.close()
    assert framework.get_mem_used() > base  # removed tensor keeps buffer alive

    del t
    gc.collect()
    assert framework.get_mem_used() == base
