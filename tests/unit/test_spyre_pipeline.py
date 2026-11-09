# SPDX-License-Identifier: Apache-2.0
"""Copier scheduling and real nogds reads with mocked Spyre primitives."""

import builtins
import gc
import mmap
import sys
import weakref
from types import ModuleType, SimpleNamespace

import pytest
import torch

from fastsafetensors.copier import spyre


@pytest.mark.parametrize("nbytes", [1, 127, 128, 4096, 65537, 1048576])
def test_host_buffer_alignment_and_mapping_lifetime(monkeypatch, nbytes):
    mappings = []
    allocate = mmap.mmap

    def track_mapping(*args, **kwargs):
        mapping = allocate(*args, **kwargs)
        mappings.append(weakref.ref(mapping))
        return mapping

    monkeypatch.setattr(spyre.mmap, "mmap", track_mapping)
    buffer = spyre._allocate_host_buffer(nbytes)
    assert buffer.dtype == torch.uint8 and buffer.device.type == "cpu"
    assert buffer.is_contiguous() and buffer.nbytes == nbytes
    assert buffer.data_ptr() % mmap.PAGESIZE == 0
    view = buffer[:]
    del buffer
    gc.collect()
    assert mappings[0]() is not None
    view.fill_(45)
    assert view[-1].item() == 45
    del view
    gc.collect()
    assert mappings[0]() is None


@pytest.fixture
def runtime(monkeypatch):
    real_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "torch_spyre.serialization" or name.startswith(
            "torch_spyre.serialization."
        ):
            raise AssertionError("the copier must use only Spyre primitives")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    module = spyre
    monkeypatch.setattr(module, "_failed_transfers", [])
    monkeypatch.setattr(module, "_MAX_BATCH_BYTES", 20)
    monkeypatch.setattr(module, "_MAX_BUFFERS", 4)
    state = SimpleNamespace(
        allocated=[],
        enqueued=[],
        pending=[],
        submissions=0,
        syncs=0,
        enqueue_error_at=None,
        sync_error=False,
        fallback=[],
        peak_host_allocation_bytes=0,
    )

    real_allocate = module._allocate_host_buffer

    def allocate(nbytes):
        buffer = real_allocate(nbytes)
        state.allocated.append(weakref.ref(buffer))
        live_bytes = sum(
            current.nbytes for ref in state.allocated if (current := ref()) is not None
        )
        state.peak_host_allocation_bytes = max(
            state.peak_host_allocation_bytes, live_bytes
        )
        return buffer

    def enqueue(buffer, dst):
        state.enqueued.append((buffer.data_ptr(), buffer.nbytes, dst.nbytes))
        state.submissions += 1
        if state.submissions == state.enqueue_error_at:
            raise RuntimeError("submission failed")
        state.pending.append(
            (weakref.ref(buffer), weakref.ref(dst), buffer[: dst.nbytes].clone())
        )

    def synchronize(device):
        state.syncs += 1
        if state.sync_error:
            raise RuntimeError("stream shut down")
        for buffer_ref, dst_ref, expected in state.pending:
            buffer, dst = buffer_ref(), dst_ref()
            assert buffer is not None and dst is not None
            # The host buffer must not be freed or overwritten while DMA is pending.
            assert torch.equal(buffer[: dst.nbytes], expected)
            dst.copy_(expected)
        state.pending.clear()

    def copy_batch(entries):
        try:
            for buffer, dst in entries:
                enqueue(buffer, dst)
        finally:
            synchronize(torch.device("cpu"))

    def convert(entries, num_threads):
        for buffer, src, dst in entries:
            assert len(buffer) == dst.nbytes
            buffer[:] = src.numpy().tobytes()

    def copy_tensor(src, dst, non_blocking):
        assert not non_blocking
        state.fallback.append(dst)
        dst.copy_(src)

    c = ModuleType("torch_spyre._C")
    c.dma_bytes = lambda dst: dst.nbytes
    c.copy_raw_h2d_batch = copy_batch
    c.convert_h2d_into = convert
    c.copy_tensor = copy_tensor
    monkeypatch.setitem(sys.modules, "torch_spyre", ModuleType("torch_spyre"))
    monkeypatch.setitem(sys.modules, "torch_spyre._C", c)
    monkeypatch.setattr(module, "_allocate_host_buffer", allocate)
    return module, c, state


def jobs_for(module, sizes):
    jobs = []
    for i, size in enumerate(sizes):
        dst = torch.zeros(size, dtype=torch.uint8)

        def fill(buffer, i=i):
            buffer[:] = bytes([i + 1]) * len(buffer)

        jobs.append(module.TransferJob(size, dst, fill))
    return jobs


@pytest.mark.parametrize("max_buffers", [1, 2, 4])
def test_batch_limits_and_buffer_lifetime(runtime, monkeypatch, max_buffers):
    module, _, state = runtime
    monkeypatch.setattr(module, "_MAX_BUFFERS", max_buffers)
    jobs = jobs_for(module, [5, 12, 5, 8, 5, 5])
    pipeline = module.run_transfer_jobs(jobs, torch.device("cpu"))
    assert pipeline.peak_host_bytes <= 20
    assert state.submissions == len(jobs)
    assert state.syncs >= 2
    assert not state.pending
    for i, job in enumerate(jobs):
        assert torch.equal(job.dst, torch.full_like(job.dst, i + 1))
    gc.collect()
    assert all(ref() is None for ref in state.allocated)


def test_buffers_are_reused_after_sync(runtime, monkeypatch):
    module, _, state = runtime
    monkeypatch.setattr(module, "_MAX_BUFFERS", 2)
    module.run_transfer_jobs(jobs_for(module, [8] * 6), torch.device("cpu"))
    assert len(state.allocated) == 2
    assert state.syncs == 3


def test_caller_pool_reuses_buffers_across_runs_and_bounds_ragged_batches(
    runtime, monkeypatch
):
    module, _, state = runtime
    monkeypatch.setattr(module, "_MAX_BATCH_BYTES", 20)
    pool = []
    pipeline = module.TransferPipeline(torch.device("cpu"), host_buffer_pool=pool)

    pipeline.run(jobs_for(module, [12, 8]))
    assert sorted(buffer.nbytes for buffer in pool) == [8, 12]
    pipeline.run(jobs_for(module, [9, 7]))
    assert len(state.allocated) == 2
    assert sorted(buffer.nbytes for buffer in pool) == [8, 12]
    assert [entry[0] for entry in state.enqueued[2:]] == [
        entry[0] for entry in state.enqueued[:2]
    ]
    assert all(buffer_size == dst_size for _, buffer_size, dst_size in state.enqueued)

    # The best-fit 8-byte slot is too large to retain beside a new 15-byte
    # slot under the cap. Both old allocations are released before replacement.
    final_jobs = jobs_for(module, [15, 5])
    pipeline.run(final_jobs)
    assert len(state.allocated) == 4
    assert sorted(buffer.nbytes for buffer in pool) == [5, 15]
    assert sum(buffer.nbytes for buffer in pool) <= 20
    assert state.peak_host_allocation_bytes <= 20
    for i, job in enumerate(final_jobs):
        assert torch.equal(job.dst, torch.full_like(job.dst, i + 1))


def test_one_shot_run_does_not_retain_its_host_buffers(runtime):
    module, _, state = runtime
    module.run_transfer_jobs(jobs_for(module, [8, 8]), torch.device("cpu"))
    gc.collect()
    assert all(ref() is None for ref in state.allocated)


def test_pool_slot_limit_includes_new_allocations(runtime, monkeypatch):
    module, _, state = runtime
    monkeypatch.setattr(module, "_MAX_BUFFERS", 2)
    pool = []
    pipeline = module.TransferPipeline(torch.device("cpu"), host_buffer_pool=pool)
    pipeline.run(jobs_for(module, [3, 3]))
    pipeline.run(jobs_for(module, [4]))
    assert sorted(buffer.nbytes for buffer in pool) == [3, 4]
    pipeline.run(jobs_for(module, [2, 2]))
    assert len(state.allocated) == 3
    assert len(pool) == 2


def test_large_tensor_runs_alone(runtime):
    module, _, state = runtime
    pipeline = module.run_transfer_jobs(
        jobs_for(module, [32, 8, 8]), torch.device("cpu")
    )
    assert pipeline.peak_host_bytes == 32
    assert state.syncs == 2


def test_empty_pipeline_does_not_touch_runtime(runtime):
    module, _, state = runtime
    module.run_transfer_jobs([], torch.device("cpu"))
    assert not state.allocated and not state.syncs


def test_fill_failure_issues_no_dma(runtime):
    module, _, state = runtime
    jobs = jobs_for(module, [8, 8])

    def fail(buffer):
        raise OSError("read failed")

    jobs[1].fill = fail
    with pytest.raises(OSError, match="read failed"):
        module.run_transfer_jobs(jobs, torch.device("cpu"))
    assert state.submissions == state.syncs == 0
    assert not module._failed_transfers


def test_batch_failure_stops_pool_reuse(runtime):
    module, _, state = runtime
    jobs = jobs_for(module, [8, 8])
    state.enqueue_error_at = 2
    with pytest.raises(RuntimeError, match="submission failed"):
        module.run_transfer_jobs(jobs, torch.device("cpu"))
    assert state.syncs == 1 and not state.pending
    assert torch.equal(jobs[0].dst, torch.ones_like(jobs[0].dst))
    assert len(module._failed_transfers) == 1
    with pytest.raises(RuntimeError, match="restart"):
        module.run_transfer_jobs([], torch.device("cpu"))


@pytest.mark.parametrize("shared_pool", [False, True])
def test_failed_sync_retains_buffers_and_destinations(runtime, shared_pool):
    module, _, state = runtime
    state.sync_error = True
    jobs = jobs_for(module, [8, 8])
    destinations = [weakref.ref(job.dst) for job in jobs]
    pool = [] if shared_pool else None
    pipeline = module.TransferPipeline(torch.device("cpu"), host_buffer_pool=pool)
    with pytest.raises(RuntimeError, match="stream shut down"):
        pipeline.run(jobs)
    if pool is not None:
        pool.clear()
    del jobs, pipeline
    gc.collect()
    assert all(ref() is not None for ref in state.allocated + destinations)
    assert len(module._failed_transfers) == 1
    with pytest.raises(RuntimeError, match="restart"):
        module.run_transfer_jobs([], torch.device("cpu"))


@pytest.mark.parametrize("mismatch", ["zero", "oversize", "capacity", "device"])
def test_invalid_jobs_fail_before_allocation(runtime, mismatch):
    module, _, state = runtime
    job = jobs_for(module, [8])[0]
    if mismatch == "zero":
        job.nbytes = 0
    elif mismatch == "oversize":
        job.nbytes = module.MAX_TRANSFER_BYTES + 1
    elif mismatch == "capacity":
        job.nbytes = 7
    else:
        job.dst = torch.empty(8, dtype=torch.uint8, device="meta")
    with pytest.raises(ValueError):
        module.run_transfer_jobs([job], torch.device("cpu"))
    assert not state.allocated and not state.submissions


def test_conversion_and_ordinary_copy_fallback(runtime, monkeypatch):
    module, c, state = runtime
    pairs = [
        (torch.arange(n, dtype=torch.uint8), torch.zeros(n, dtype=torch.uint8))
        for n in (8, 12, 16)
    ]
    monkeypatch.setattr(module, "MAX_TRANSFER_BYTES", 12)
    monkeypatch.setattr(
        c, "dma_bytes", lambda dst: 0 if dst.nbytes == 12 else dst.nbytes
    )
    pipeline = module.stage_to_spyre(pairs, torch.device("cpu"))
    assert state.submissions == 1 and len(state.fallback) == 2
    assert pipeline.routes["raw"] == [1, 8]
    for src, dst in pairs:
        assert torch.equal(src, dst)


def test_nogds_reads_offsets_and_chunks_before_dma(runtime, monkeypatch, tmp_path):
    module, _, state = runtime
    monkeypatch.setattr(module, "_READ_BLOCK_BYTES", 3)
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.write_bytes(b"headerABCDEFGHtail")
    second.write_bytes(b"prefix12345end")
    with first.open("rb") as a, second.open("rb") as b:
        jobs = [
            module.TransferJob(
                8, torch.zeros(8, dtype=torch.uint8), module.FileRead(a.fileno(), 6)
            ),
            module.TransferJob(
                5, torch.zeros(5, dtype=torch.uint8), module.FileRead(b.fileno(), 6)
            ),
        ]
        pipeline = module.run_transfer_jobs(jobs, torch.device("cpu"))
    assert bytes(jobs[0].dst.tolist()) == b"ABCDEFGH"
    assert bytes(jobs[1].dst.tolist()) == b"12345"
    assert state.submissions == 2 and state.syncs == 1
    assert pipeline.peak_host_bytes <= 20


def test_nogds_short_file_issues_no_dma(runtime, tmp_path):
    module, _, state = runtime
    path = tmp_path / "short"
    path.write_bytes(b"short")
    with path.open("rb") as source:
        jobs = [
            module.TransferJob(
                8,
                torch.zeros(8, dtype=torch.uint8),
                module.FileRead(source.fileno(), 0),
            ),
            module.TransferJob(
                4,
                torch.zeros(4, dtype=torch.uint8),
                module.FileRead(source.fileno(), 1),
            ),
        ]
        with pytest.raises(OSError, match="host read failed"):
            module.run_transfer_jobs(jobs, torch.device("cpu"))
    assert state.submissions == state.syncs == 0
    assert not module._failed_transfers


def test_nogds_submission_failure_drains_reads(runtime, monkeypatch, tmp_path):
    module, _, state = runtime
    reader_type = module.fstcpp.nogds_file_reader
    waited = []

    class FailingReader:
        def __init__(self, *args):
            self.reader = reader_type(*args)
            self.submitted = 0

        def submit_read(self, *args):
            self.submitted += 1
            if self.submitted == 2:
                return -1
            return self.reader.submit_read(*args)

        def wait_read(self, req):
            waited.append(req)
            return self.reader.wait_read(req)

    monkeypatch.setattr(module.fstcpp, "nogds_file_reader", FailingReader)
    monkeypatch.setattr(module, "_READ_BLOCK_BYTES", 4)
    path = tmp_path / "data"
    path.write_bytes(b"12345678")
    with path.open("rb") as source:
        job = module.TransferJob(
            8, torch.zeros(8, dtype=torch.uint8), module.FileRead(source.fileno(), 0)
        )
        with pytest.raises(OSError, match="submission failed"):
            module.run_transfer_jobs([job], torch.device("cpu"))
    assert len(waited) == 1
    assert state.submissions == state.syncs == 0


def test_mixed_file_read_and_conversion(runtime, tmp_path):
    module, _, state = runtime
    path = tmp_path / "data"
    path.write_bytes(b"abcd")
    jobs = jobs_for(module, [8])
    with path.open("rb") as source:
        jobs.append(
            module.TransferJob(
                4,
                torch.zeros(4, dtype=torch.uint8),
                module.FileRead(source.fileno(), 0),
            )
        )
        module.run_transfer_jobs(jobs, torch.device("cpu"))
    assert torch.equal(jobs[0].dst, torch.ones(8, dtype=torch.uint8))
    assert bytes(jobs[1].dst.tolist()) == b"abcd"
    assert state.syncs == 1


@pytest.mark.parametrize("fd, offset", [(-1, 0), (0, -1)])
def test_invalid_file_read_fails_before_allocation(runtime, fd, offset):
    module, _, state = runtime
    job = module.TransferJob(
        8, torch.zeros(8, dtype=torch.uint8), module.FileRead(fd, offset)
    )
    with pytest.raises(ValueError, match="host read source"):
        module.run_transfer_jobs([job], torch.device("cpu"))
    assert not state.allocated and not state.submissions
