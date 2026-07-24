# SPDX-License-Identifier: Apache-2.0

"""Bulk draining (drain_tensors) and ParallelLoader consuming-migration tests.

drain_tensors() visits every available name once, consuming each; the yielded
tensors retain shared ownership so they stay valid after close(). ParallelLoader
now uses this path (no lifetime clone), and its shutdown closes buffers that were
produced but never consumed.
"""

import gc
import os

import pytest

from fastsafetensors import ParallelLoader, SafeTensorsFileLoader, SingleGroup
from fastsafetensors.common import is_gpu_found
from fastsafetensors.frameworks import FrameworkOpBase
from fastsafetensors.parallel_loader import FileBatch, PipelineParallel
from fastsafetensors.st_types import Device, DType


def _device(framework: FrameworkOpBase) -> Device:
    if is_gpu_found():
        dev = "cuda:0" if framework.get_name() == "pytorch" else "gpu:0"
    else:
        dev = "cpu"
    return Device.from_str(dev)


def _save(path, tensors, framework):
    if framework.get_name() == "pytorch":
        from safetensors.torch import save_file
    else:
        from safetensors.paddle import save_file
    save_file({k: v.get_raw() for k, v in tensors.items()}, path, metadata={"fst": "t"})


def _make_files(tmp_dir, framework, n_files=4, per_file=3):
    device = _device(framework)
    files = []
    for fi in range(n_files):
        path = os.path.join(tmp_dir, f"drain_{fi}.safetensors")
        tensors = {
            f"f{fi}.t{ti}": framework.randn((4, 8), device=device, dtype=DType.F32)
            for ti in range(per_file)
        }
        _save(path, tensors, framework)
        files.append(path)
    return files


def _load_ref(path, device, framework):
    if framework.get_name() == "pytorch":
        from safetensors.torch import load_file
    else:
        from safetensors.paddle import load_file
    return load_file(path, device.as_str())


def test_drain_tensors_consumes_all_and_valid_after_close(
    fstcpp_log, input_files, framework
):
    device = _device(framework)
    loader = SafeTensorsFileLoader(
        SingleGroup(), device.as_str(), nogds=True, framework=framework.get_name()
    )
    loader.add_filenames({0: [input_files[0]]})
    fb = loader.copy_files_to_device()

    expected = _load_ref(input_files[0], device, framework)
    available = list(fb.keys())

    drained = {}
    for name, t in fb.drain_tensors():
        drained[name] = t
    # Visited every available name exactly once and consumed them all.
    assert set(drained.keys()) == set(available)
    assert fb.keys() == []

    fb.close()
    loader.close()

    # Tensors remain valid and correct after close (shared ownership).
    for name, exp in expected.items():
        assert _eq(drained[name], exp, framework)


def _eq(raw, exp, framework):
    if framework.get_name() == "pytorch":
        import torch

        return bool(torch.equal(raw, exp))
    import paddle

    return bool(paddle.all(raw == exp))


def test_drain_tensors_wrapped_values(fstcpp_log, input_files, framework):
    device = _device(framework)
    loader = SafeTensorsFileLoader(
        SingleGroup(), device.as_str(), nogds=True, framework=framework.get_name()
    )
    loader.add_filenames({0: [input_files[0]]})
    fb = loader.copy_files_to_device()
    expected = _load_ref(input_files[0], device, framework)

    drained = dict(fb.drain_tensors_wrapped())
    fb.close()
    loader.close()

    for name, exp in expected.items():
        assert framework.is_equal(drained[name], exp)


def _collect_all(loader):
    # Collect in a helper so the loop variables do not linger in the caller's
    # frame and pin a zero-copy tensor's shared buffer past the assertion.
    out = {}
    for name, tensor in loader.iterate_weights():
        out[name] = tensor
    return out


def _first_then_stop(loader):
    # Return after the first tensor; returning tears down this frame, which
    # closes the still-open generator and runs its cleanup (stop + drain).
    for _name, tensor in loader.iterate_weights():
        return tensor
    return None


def test_parallel_loader_iterate_weights(fstcpp_log, tmp_dir, framework):
    base = framework.get_mem_used()
    device = _device(framework)
    files = _make_files(tmp_dir, framework, n_files=4, per_file=3)
    refs = {}
    for f in files:
        refs.update(_load_ref(f, device, framework))

    loader = ParallelLoader(
        None,
        files,
        device=device.as_str(),
        nogds=True,
        framework=framework.get_name(),
        queue_size=2,
    )
    got = _collect_all(loader)
    assert set(got.keys()) == set(refs.keys())
    for name, exp in refs.items():
        assert _eq(got[name], exp, framework)

    loader.close()
    del got
    gc.collect()
    assert framework.get_mem_used() == base


def test_iterate_weights_early_termination_frees_buffers(
    fstcpp_log, tmp_dir, framework
):
    base = framework.get_mem_used()
    device = _device(framework)
    files = _make_files(tmp_dir, framework, n_files=6, per_file=3)

    loader = ParallelLoader(
        None,
        files,
        device=device.as_str(),
        nogds=True,
        framework=framework.get_name(),
        queue_size=4,
    )

    # Stop after the very first tensor (generator closed inside the helper).
    first = _first_then_stop(loader)
    assert first is not None

    loader.close()
    del first
    gc.collect()
    # Every buffer -- consumed, in-flight, and queued-but-unconsumed -- is freed.
    assert framework.get_mem_used() == base


def test_drain_and_close_queue_closes_buffers(fstcpp_log, tmp_dir, framework):
    """The shutdown drain closes buffers that were produced but never consumed."""
    base = framework.get_mem_used()
    device = _device(framework)
    files = _make_files(tmp_dir, framework, n_files=1, per_file=3)

    pipeline = ParallelLoader(
        None,
        files,
        device=device.as_str(),
        nogds=True,
        framework=framework.get_name(),
    )

    # Produce a real buffer via a separate loader and enqueue it as if the
    # producer had, without ever consuming it.
    side = SafeTensorsFileLoader(
        SingleGroup(), device.as_str(), nogds=True, framework=framework.get_name()
    )
    side.add_filenames({0: files})
    fb = side.copy_files_to_device()
    assert fb.rank_loaders != {}
    assert framework.get_mem_used() > base

    pipeline.batch_queue.put(FileBatch(fb, list(fb.keys()), 0))
    pipeline._drain_and_close_queue()

    # The drained batch's buffer was closed.
    assert fb.rank_loaders == {}
    side.close()
    pipeline.close()
    gc.collect()
    assert framework.get_mem_used() == base


def test_pipeline_repeated_close_is_safe(fstcpp_log, tmp_dir, framework):
    device = _device(framework)
    files = _make_files(tmp_dir, framework, n_files=1, per_file=2)
    loader = ParallelLoader(
        None, files, device=device.as_str(), nogds=True, framework=framework.get_name()
    )
    loader.close()
    loader.close()  # idempotent
    assert isinstance(loader, PipelineParallel)
