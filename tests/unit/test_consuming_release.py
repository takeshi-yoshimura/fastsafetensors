# SPDX-License-Identifier: Apache-2.0

"""Release-timing tests for consuming access and the auto_mem_delete rework.

PR 3 replaces the count-based ``instantiated`` accounting with reference drops:
consuming access (get_and_remove_*) and auto_mem_delete drop the factory's
internal reference to a name, and the backing buffer is released by shared
ownership once the factory's retained set drains and no exported tensor still
references it -- there is no explicit "last tensor" free.
"""

import gc

import pytest

from fastsafetensors import (
    SafeTensorsFileLoader,
    SingleGroup,
    TensorConsumedError,
)
from fastsafetensors.common import is_gpu_found
from fastsafetensors.frameworks import FrameworkOpBase
from fastsafetensors.st_types import Device


def _device(framework: FrameworkOpBase) -> Device:
    if is_gpu_found():
        dev = "cuda:0" if framework.get_name() == "pytorch" else "gpu:0"
    else:
        dev = "cpu"
    return Device.from_str(dev)


def _open(framework, files, keep=None):
    device = _device(framework)
    loader = SafeTensorsFileLoader(
        SingleGroup(), device.as_str(), nogds=True, framework=framework.get_name()
    )
    if keep is not None:
        loader.set_tensor_filter(keep)
    loader.add_filenames({0: files})
    fb = loader.copy_files_to_device()
    return loader, fb


def test_get_and_remove_sharded_consumes(fstcpp_log, input_files, framework):
    loader, fb = _open(framework, [input_files[0]])
    key = fb.keys()[0]

    t = fb.get_and_remove_sharded(key, dim=-1)
    assert t is not None
    assert key not in fb.keys()
    with pytest.raises(TensorConsumedError):
        fb.get_and_remove_sharded(key, dim=-1)

    fb.close()
    loader.close()


def test_get_and_remove_multi_cols_consumes(fstcpp_log, input_files, framework):
    loader, fb = _open(framework, [input_files[0]])
    # two tensors with the same shape so they concatenate cleanly
    names = [f"h.{i}.attn.c_proj.weight" for i in range(2)]
    for n in names:
        assert n in fb.keys()

    out = fb.get_and_remove_multi_cols(names, dim=0)
    assert out is not None
    for n in names:
        assert n not in fb.keys()
        with pytest.raises(TensorConsumedError):
            fb.get_tensor(n)

    fb.close()
    loader.close()


def test_consuming_frees_buffer_before_close(fstcpp_log, input_files, framework):
    """Consuming every name drains the factory; dropping the exported tensors
    then frees the buffer without waiting for close()."""
    base = framework.get_mem_used()
    loader, fb = _open(framework, [input_files[0]])
    factory = fb.rank_loaders[0][0]

    held = []
    for name in list(fb.keys()):
        held.append(fb.get_and_remove_tensor(name))
    assert fb.keys() == []

    # All names consumed -> the factory's retained set is empty and its
    # buffer-side reference is released; only the exported tensors keep the
    # allocation alive now.
    assert factory.gbuf is None
    assert framework.get_mem_used() > base

    del held
    gc.collect()
    # Last references gone -> buffer freed, before close() was ever called.
    assert framework.get_mem_used() == base

    fb.close()  # no double free
    loader.close()
    assert framework.get_mem_used() == base


def test_reusable_access_holds_until_close_and_allows_reaccess(
    fstcpp_log, input_files, framework
):
    base = framework.get_mem_used()
    loader, fb = _open(framework, [input_files[0]])
    factory = fb.rank_loaders[0][0]
    key = fb.keys()[0]

    a = fb.get_tensor_wrapped(key)
    b = fb.get_tensor_wrapped(key)  # re-access allowed (reusable)
    assert framework.is_equal(a, b.get_raw())
    # Reusable access keeps the buffer's internal references until close.
    assert factory.gbuf is not None
    assert framework.get_mem_used() > base

    fb.close()
    loader.close()
    del a, b
    gc.collect()
    assert framework.get_mem_used() == base


def test_auto_mem_delete_with_filter_releases_buffer(
    fstcpp_log, input_files, framework
):
    """With a tensor filter, retain_only prunes filtered names so consuming the
    kept ones drains the factory and releases the buffer -- the old count check
    compared against the full file metadata and never released here."""
    from fastsafetensors.common import SafeTensorsMetadata

    meta = SafeTensorsMetadata.from_file(input_files[0], framework)
    all_names = sorted(meta.tensors.keys())
    kept = set(all_names[::2])
    assert 0 < len(kept) < len(all_names)  # a strict subset is filtered out

    loader, fb = _open(framework, [input_files[0]], keep=lambda n: n in kept)
    factory = fb.rank_loaders[0][0]
    assert set(fb.keys()) == kept

    held = [fb.get_and_remove_tensor(n) for n in list(fb.keys())]
    # Even though filtered-out names still exist in the file metadata, the
    # factory's retained set is now empty, so its buffer is released.
    assert factory.gbuf is None

    del held
    gc.collect()
    fb.close()
    loader.close()


def test_repeated_close_after_consume_no_double_free(
    fstcpp_log, input_files, framework
):
    base = framework.get_mem_used()
    loader, fb = _open(framework, [input_files[0]])
    keys = list(fb.keys())
    held = [fb.get_and_remove_tensor(k) for k in keys]

    fb.close()
    fb.close()
    loader.close()
    del held
    gc.collect()
    assert framework.get_mem_used() == base
