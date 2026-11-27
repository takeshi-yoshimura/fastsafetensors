# SPDX-License-Identifier: Apache-2.0
"""Real preadv with a framework-independent TensorBase/HostTransferOp adapter."""

import gc
import os
import threading
import weakref
from concurrent.futures import ThreadPoolExecutor

import pytest

from fastsafetensors.copier import host
from fastsafetensors.frameworks import TensorBase
from fastsafetensors.st_types import Device, DType


class Storage(bytearray):
    pass


class ByteTensor(TensorBase):
    def __init__(self, storage, start=0, end=None):
        super().__init__(Device(), DType.U8)
        self.storage = storage
        self.start = start
        self.end = len(storage) if end is None else end

    def get_raw(self):
        raise AssertionError("the reader must never unwrap tensors")

    def get_nbytes(self):
        return self.end - self.start

    def get_shape(self):
        return [self.get_nbytes()]

    def __getitem__(self, key):
        start, end, step = key.indices(self.get_nbytes())
        assert step == 1
        return ByteTensor(self.storage, self.start + start, self.start + end)


class ByteOps:
    def __init__(self):
        self.allocations = []
        self.peak = 0
        self.copies = []
        self.local = threading.local()
        self.before_copy = lambda entries: None

    def allocate_host_buffer(self, nbytes):
        storage = Storage(nbytes)
        self.allocations.append(weakref.ref(storage))
        self.peak = max(
            self.peak,
            sum(len(s) for ref in self.allocations if (s := ref()) is not None),
        )
        return ByteTensor(storage)

    def transfer_alignment(self):
        return 1

    def host_buffer_view(self, tensor):
        assert isinstance(tensor, TensorBase)
        return memoryview(tensor.storage)[tensor.start : tensor.end]

    def validate_transfer(self, tensor, device):
        assert isinstance(tensor, TensorBase) and device == Device()
        return tensor.get_nbytes()

    def prepare_copy(self, device):
        context = getattr(self.local, "context", None)

        def copy(entries):
            self.local.context = context
            expected = [bytes(self.host_buffer_view(b)) for b, _, _, _ in entries]
            self.before_copy(entries)
            assert [
                bytes(self.host_buffer_view(b)) for b, _, _, _ in entries
            ] == expected
            for data, (_, dst, offset, size) in zip(expected, entries):
                self.host_buffer_view(dst)[offset : offset + size] = data
            self.copies.extend(expected)

        return copy


@pytest.fixture
def ops(monkeypatch):
    monkeypatch.setattr(host, "_failed_transfers", [])
    monkeypatch.setattr(host, "_transfer_poisoned", False)
    monkeypatch.setattr(host, "_HOST_BYTES", 16)
    return ByteOps()


def entries_for(fd, sizes, offset=0):
    entries = []
    for size in sizes:
        entries.append(((fd, offset), ByteTensor(Storage(size))))
        offset += size
    return entries


def submit(reader, entries):
    return reader.submit_read(entries, Device())


def test_copy_overlaps_next_direct_read_and_preserves_context(
    ops, tmp_path, monkeypatch
):
    path = tmp_path / "data"
    path.write_bytes(b"headerABCDEFGHIJKLMNOtail")
    copying, next_read, release = (threading.Event() for _ in range(3))
    preadv = os.preadv

    def read(fd, buffers, offset):
        if offset == 14:
            assert copying.wait(5)
            next_read.set()
        return preadv(fd, buffers, offset)

    def copy(entries):
        assert ops.local.context == "caller stream"
        copying.set()
        assert release.wait(5)

    monkeypatch.setattr(os, "preadv", read)
    ops.before_copy = copy
    ops.local.context = "caller stream"
    reader = host.HostFileReader(ops, max_threads=2)
    try:
        with path.open("rb") as f:
            entries = entries_for(f.fileno(), [8, 7], 6)
            request = submit(reader, entries)
            try:
                assert copying.wait(5)
                assert next_read.wait(5), "next read must run while copy is blocked"
                assert not request.done.is_set()
                assert len(ops.allocations) == 2
            finally:
                release.set()
            reader.wait_read(request)
            assert ops.copies == [b"ABCDEFGH", b"IJKLMNO"]
            assert bytes(ops.host_buffer_view(entries[1][1])) == b"IJKLMNO"
    finally:
        release.set()
        reader.close()
    assert ops.peak <= 20
    gc.collect()
    assert all(ref() is None for ref in ops.allocations)


def test_multiple_files_progress_without_wait_and_submit_never_waits_for_pool(
    ops, tmp_path, monkeypatch
):
    path = tmp_path / "data"
    path.write_bytes(b"abcdefgh")
    copying, release, both = (threading.Event() for _ in range(3))
    calls = []

    def copy(entries):
        calls.append(entries[0][1])
        copying.set()
        assert release.wait(5)
        if len(calls) == 2:
            both.set()

    ops.before_copy = copy
    reader = host.HostFileReader(ops)
    try:
        with path.open("rb") as first, path.open("rb") as second:
            one = submit(reader, entries_for(first.fileno(), [8]))
            assert copying.wait(5)
            # Would deadlock if submit acquired a buffer owned by the first DMA.
            two = submit(reader, entries_for(second.fileno(), [8]))
            assert not two.done.is_set()
            release.set()
            assert both.wait(5), "second file must progress before either wait_read"
            reader.wait_read(two)
            reader.wait_read(one)
            assert ops.copies == [b"abcdefgh", b"abcdefgh"]
    finally:
        release.set()
        reader.close()
    assert len(ops.allocations) == 2 and ops.peak == 16


@pytest.mark.parametrize("capacity", [4, 16, 20])
def test_ragged_groups_account_for_busy_ready_and_retained_buffers(
    ops, tmp_path, capacity, monkeypatch
):
    monkeypatch.setattr(host, "_HOST_BYTES", capacity)
    data = bytes(range(128))
    path = tmp_path / "data"
    path.write_bytes(data)
    reader = host.HostFileReader(ops)
    try:
        with path.open("rb") as f:
            for sizes in ([5, 12, 5, 8, 5, 5], [15, 5], [3, 3], [4], [2, 2]):
                entries = entries_for(f.fileno(), sizes)
                reader.wait_read(submit(reader, entries))
                for (fd, begin), dst in entries:
                    assert (
                        bytes(ops.host_buffer_view(dst))
                        == data[begin : begin + dst.get_nbytes()]
                    )
                assert len(reader._state.pool.buffers) == 2
                assert (
                    sum(b.get_nbytes() for b in reader._state.pool.buffers) <= capacity
                )
    finally:
        reader.close()
    assert ops.peak <= capacity
    assert len(ops.allocations) == 2


def test_large_tensor_is_split_without_oversized_allocation(ops, tmp_path):
    path = tmp_path / "data"
    path.write_bytes(bytes(range(64)))
    reader = host.HostFileReader(ops)
    try:
        with path.open("rb") as f:
            reader.wait_read(submit(reader, entries_for(f.fileno(), [8, 32, 8])))
            assert list(map(len, ops.copies)) == [8] * 6
    finally:
        reader.close()
    assert ops.peak == 16 and len(ops.allocations) == 2


def test_short_reads_eintr_block_limit_and_eof_retry(ops, tmp_path, monkeypatch):
    path = tmp_path / "data"
    path.write_bytes(b"12345678")
    preadv = os.preadv
    lengths = []

    def short_read(fd, buffers, offset):
        lengths.append(len(buffers[0]))
        if len(lengths) == 1:
            raise InterruptedError()
        return preadv(fd, [buffers[0][:2]], offset)

    monkeypatch.setattr(os, "preadv", short_read)
    reader = host.HostFileReader(ops, block_size=4)
    try:
        with path.open("rb") as f:
            bad = reader.submit_read(
                entries_for(f.fileno(), [7], offset=2), Device(), 3
            )
            with pytest.raises(OSError, match="unexpected end of file"):
                reader.wait_read(bad)
            assert not ops.copies and not host._failed_transfers
            reader.wait_read(
                reader.submit_read(entries_for(f.fileno(), [8]), Device(), 3)
            )
    finally:
        reader.close()
    assert max(lengths) <= 3 and ops.copies == [b"12345678"]


@pytest.mark.parametrize("failure", ["read", "submit"])
def test_read_failure_drains_other_workers_before_return(
    ops, tmp_path, monkeypatch, failure
):
    path = tmp_path / "data"
    path.write_bytes(b"12345678")
    active, release, draining = (threading.Event() for _ in range(3))
    preadv = os.preadv
    real_submit = ThreadPoolExecutor.submit
    submitted = 0

    def read(fd, buffers, offset):
        if offset == 0:
            active.set()
            assert release.wait(5)
        else:
            assert active.wait(5)
            draining.set()
            raise OSError("read failed")
        return preadv(fd, buffers, offset)

    def fail_submit(self, function, *args, **kwargs):
        nonlocal submitted
        submitted += 1
        if submitted == 2:
            assert active.wait(5)
            draining.set()
            raise OSError("submit failed")
        return real_submit(self, function, *args, **kwargs)

    monkeypatch.setattr(os, "preadv", read)
    if failure == "submit":
        monkeypatch.setattr(ThreadPoolExecutor, "submit", fail_submit)
    reader = host.HostFileReader(ops, max_threads=2, block_size=4)
    try:
        with path.open("rb") as f:
            request = submit(reader, entries_for(f.fileno(), [8]))
            assert active.wait(5)
            assert draining.wait(5)
            assert not request.done.is_set()
            release.set()
            with pytest.raises(OSError, match=f"{failure} failed"):
                reader.wait_read(request)
            assert not ops.copies
    finally:
        release.set()
        reader.close()


def test_cancel_drains_read_and_never_starts_copy(ops, tmp_path, monkeypatch):
    path = tmp_path / "data"
    path.write_bytes(b"abcdefgh")
    active, release = threading.Event(), threading.Event()
    preadv = os.preadv

    def read(*args):
        active.set()
        assert release.wait(5)
        return preadv(*args)

    monkeypatch.setattr(os, "preadv", read)
    reader = host.HostFileReader(ops)
    try:
        with path.open("rb") as f, ThreadPoolExecutor(1) as waiter:
            request = submit(reader, entries_for(f.fileno(), [8]))
            assert active.wait(5)
            cancelled = waiter.submit(reader.cancel_read, request)
            assert request.cancelled.wait(5)
            assert not cancelled.done()
            release.set()
            cancelled.result(5)
            assert not ops.copies and not reader._state.pool.busy
    finally:
        release.set()
        reader.close()


def test_close_drains_running_copy_cancels_pending_and_joins_threads(ops, tmp_path):
    path = tmp_path / "data"
    path.write_bytes(b"abcdefgh" * 20)
    active, release = threading.Event(), threading.Event()

    def copy(entries):
        active.set()
        assert release.wait(5)

    ops.before_copy = copy
    reader = host.HostFileReader(ops)
    state = reader._state
    with path.open("rb") as f, ThreadPoolExecutor(1) as waiter:
        request = submit(reader, entries_for(f.fileno(), [8] * 20))
        try:
            assert active.wait(5)
            closed = waiter.submit(reader.close)
            assert request.cancelled.wait(5)
            assert not closed.done()
        finally:
            release.set()
        closed.result(5)
        assert request.done.is_set()
        assert len(ops.copies) == 1
        assert not state.threads and not state.pool.buffers and not state.requests
        reader.close()
        with pytest.raises(RuntimeError, match="closed"):
            submit(reader, [])


def test_dma_failure_retains_storage_and_prevents_other_requests(ops, tmp_path):
    path = tmp_path / "data"
    path.write_bytes(b"abcdefgh")

    def fail(entries):
        raise RuntimeError("uncertain DMA")

    ops.before_copy = fail
    reader = host.HostFileReader(ops)
    with path.open("rb") as f:
        entries = entries_for(f.fileno(), [8])
        destination = weakref.ref(entries[0][1].storage)
        request = submit(reader, entries)
        with pytest.raises(RuntimeError, match="uncertain DMA"):
            reader.wait_read(request)
        with pytest.raises(RuntimeError, match="restart"):
            submit(reader, entries)
    del entries, request
    reader.close()
    gc.collect()
    assert destination() is not None
    assert any(ref() is not None for ref in ops.allocations)
    assert len(host._failed_transfers) == 1


def test_dma_failure_drains_queued_and_pool_blocked_requests(ops, tmp_path):
    path = tmp_path / "data"
    path.write_bytes(b"abcdefgh" * 10)
    active, release = threading.Event(), threading.Event()

    def fail(entries):
        active.set()
        assert release.wait(5)
        raise RuntimeError("uncertain DMA")

    ops.before_copy = fail
    reader = host.HostFileReader(ops)
    try:
        with path.open("rb") as f:
            one = submit(reader, entries_for(f.fileno(), [8] * 10))
            assert active.wait(5)
            two = submit(reader, entries_for(f.fileno(), [8]))
            release.set()
            assert one.done.wait(5) and two.done.wait(5)
            for request in (one, two):
                with pytest.raises(RuntimeError, match="uncertain DMA|restart"):
                    reader.wait_read(request)
            assert not reader._state.pool.busy
    finally:
        release.set()
        reader.close()


def test_demonstrably_drained_enqueue_failure_does_not_poison(ops, tmp_path):
    from fastsafetensors.frameworks import TransferDrainedError

    path = tmp_path / "data"
    path.write_bytes(b"abcdefgh")

    def fail(entries):
        raise TransferDrainedError("partial batch drained")

    ops.before_copy = fail
    reader = host.HostFileReader(ops)
    try:
        with path.open("rb") as file:
            entries = entries_for(file.fileno(), [8])
            with pytest.raises(TransferDrainedError, match="drained"):
                reader.wait_read(submit(reader, entries))
            host.check_transfer_state()
            assert not host._failed_transfers
            ops.before_copy = lambda entries: None
            reader.wait_read(submit(reader, entries))
    finally:
        reader.close()


def test_worker_start_failure_is_closeable(ops, monkeypatch):
    start = threading.Thread.start

    def fail(thread):
        if thread.name == "fst-host-copy":
            raise RuntimeError("thread start failed")
        start(thread)

    monkeypatch.setattr(threading.Thread, "start", fail)
    reader = host.HostFileReader(ops)
    with pytest.raises(RuntimeError, match="thread start failed"):
        submit(reader, entries_for(0, [8]))
    reader.close()
    assert not reader._state.threads


def test_owner_collection_stops_idle_workers_and_frees_pool(ops, tmp_path):
    path = tmp_path / "data"
    path.write_bytes(b"abcdefgh")
    reader = host.HostFileReader(ops)
    state = reader._state
    with path.open("rb") as f:
        reader.wait_read(submit(reader, entries_for(f.fileno(), [8])))
    threads = list(state.threads)
    del reader
    gc.collect()
    assert threads and all(not t.is_alive() for t in threads)
    assert all(ref() is None for ref in ops.allocations)


def test_empty_request_and_invalid_sizes_allocate_nothing(ops):
    reader = host.HostFileReader(ops)
    try:
        reader.wait_read(submit(reader, []))
        with pytest.raises(ValueError, match="positive"):
            submit(reader, [((0, 0), ByteTensor(Storage()))])
        with pytest.raises(ValueError, match="positive"):
            reader.submit_read([], Device(), 0)
        assert not reader._state.threads and not ops.allocations
    finally:
        reader.close()


def test_slot_capacity_rounds_down_and_rejects_less_than_two_aligned_slots(
    ops, monkeypatch
):
    monkeypatch.setattr(ops, "transfer_alignment", lambda: 8)
    with pytest.raises(ValueError, match="two 8-byte aligned slots"):
        host.HostFileReader(ops, max_bytes=15)
    reader = host.HostFileReader(ops, max_bytes=39)
    try:
        pool = reader._state.pool
        assert pool.slot_bytes == 16
        first = pool.acquire(lambda: False)
        second = pool.acquire(lambda: False)
        assert first is not second
        assert pool.peak_bytes == 32 <= 39
        pool.release(first)
        pool.release(second)
    finally:
        reader.close()


def test_failed_second_slot_allocation_does_not_leave_a_half_pool(ops, monkeypatch):
    allocate = ops.allocate_host_buffer
    calls = 0

    def fail(size):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise MemoryError("second slot")
        return allocate(size)

    monkeypatch.setattr(ops, "allocate_host_buffer", fail)
    reader = host.HostFileReader(ops)
    try:
        pool = reader._state.pool
        with pytest.raises(MemoryError, match="second slot"):
            pool.acquire(lambda: False)
        assert not pool.buffers and not pool.busy
        leased = pool.acquire(lambda: False)
        assert len(pool.buffers) == 2
        pool.release(leased)
    finally:
        reader.close()


class DeferredOps(ByteOps):
    """No data moves until wait; pending entries keep their real source views."""

    def __init__(self):
        super().__init__()
        self.events = []
        self.active = []
        self.before_wait = lambda ticket: None
        self.after_enqueue = lambda ticket: None

    def prepare_copy(self, device):
        context = getattr(self.local, "context", None)
        ops = self

        class Ticket:
            def __init__(self, entries):
                self.entries = entries
                self.context = context
                self.expected = [
                    bytes(ops.host_buffer_view(b)) for b, _, _, _ in entries
                ]
                self.index = len([e for e in ops.events if e[0] == "enqueue"])

            def wait(self):
                ops.events.append(("wait", self.index, self.context))
                ops.before_wait(self)
                assert [
                    bytes(ops.host_buffer_view(b)) for b, _, _, _ in self.entries
                ] == self.expected, "source slot overwritten before completion"
                for data, (_, dst, offset, size) in zip(self.expected, self.entries):
                    ops.host_buffer_view(dst)[offset : offset + size] = data
                ops.active.remove(self)
                self.entries = []
                ops.events.append(("done", self.index, self.context))

        def enqueue(entries):
            ticket = Ticket(entries)
            ops.active.append(ticket)
            assert len(ops.active) <= 2
            ops.events.append(("enqueue", ticket.index, context))
            ops.after_enqueue(ticket)
            return ticket

        return enqueue


@pytest.mark.parametrize("sizes", [[8], [3], [16], [24], [33], [3, 9, 5], [8] * 9])
def test_deferred_copy_fills_two_slots_then_waits_and_tail_progresses(
    ops, tmp_path, sizes
):
    deferred = DeferredOps()
    path = tmp_path / "data"
    data = bytes(range(sum(sizes)))
    path.write_bytes(data)
    reader = host.HostFileReader(deferred, max_threads=2)
    try:
        with path.open("rb") as file:
            entries = entries_for(file.fileno(), sizes)
            request = submit(reader, entries)
            assert request.done.wait(5), "tail must progress without wait_read/close"
            reader.wait_read(request)
            for (_, begin), dst in entries:
                assert (
                    bytes(deferred.host_buffer_view(dst))
                    == data[begin : begin + dst.get_nbytes()]
                )
        events = [event[0] for event in deferred.events]
        count = sum((size + 7) // 8 for size in sizes)
        assert events.count("enqueue") == events.count("wait") == count
        if count > 1:
            assert events[:3] == ["enqueue", "enqueue", "wait"]
        assert not deferred.active and not reader._state.pool.busy
        assert len(deferred.allocations) == 2 and deferred.peak == 16
    finally:
        reader.close()


@pytest.mark.parametrize("action", ["cancel", "close", "finish"])
def test_deferred_pending_slots_not_reused_and_cancel_close_drain_both(
    ops, tmp_path, monkeypatch, action
):
    deferred = DeferredOps()
    waiting, release = threading.Event(), threading.Event()
    path = tmp_path / "data"
    path.write_bytes(bytes(range(32)))
    read = host.pread_into

    def check_read(fd, view, offset):
        assert offset < 16 or release.is_set(), "third read reused a pending slot"
        return read(fd, view, offset)

    def wait(ticket):
        if ticket.index == 0:
            assert [e[0] for e in deferred.events] == ["enqueue", "enqueue", "wait"]
            waiting.set()
            assert release.wait(5)

    deferred.before_wait = wait
    monkeypatch.setattr(host, "pread_into", check_read)
    reader = host.HostFileReader(deferred, max_threads=1)
    try:
        with path.open("rb") as file, ThreadPoolExecutor(1) as executor:
            request = submit(reader, entries_for(file.fileno(), [32]))
            assert waiting.wait(5)
            assert len(reader._state.pool.busy) == len(deferred.active) == 2
            assert not request.done.is_set()
            if action != "finish":
                operation = (
                    reader.close
                    if action == "close"
                    else lambda: reader.cancel_read(request)
                )
                future = executor.submit(operation)
                assert request.cancelled.wait(5) and not future.done()
            release.set()
            if action != "finish":
                future.result(5)
            assert request.done.wait(5)
            reader.wait_read(request)
            assert not deferred.active
            assert [e[1] for e in deferred.events if e[0] == "wait"] == list(
                range(4 if action == "finish" else 2)
            )
    finally:
        release.set()
        reader.close()


def test_deferred_different_requests_keep_captured_contexts(ops, tmp_path):
    deferred = DeferredOps()
    first_enqueued, release = threading.Event(), threading.Event()

    def enqueue(ticket):
        if ticket.index == 0:
            first_enqueued.set()
            assert release.wait(5)

    deferred.after_enqueue = enqueue
    path = tmp_path / "data"
    path.write_bytes(bytes(range(16)))
    reader = host.HostFileReader(deferred)
    try:
        with path.open("rb") as file:
            deferred.local.context = "stream A"
            first = submit(reader, entries_for(file.fileno(), [8]))
            assert first_enqueued.wait(5)
            deferred.local.context = "stream B"
            second = submit(reader, entries_for(file.fileno(), [8], 8))
            release.set()
            assert first.done.wait(5) and second.done.wait(5)
            reader.wait_read(first)
            reader.wait_read(second)
            assert deferred.events == [
                ("enqueue", 0, "stream A"),
                ("enqueue", 1, "stream B"),
                ("wait", 0, "stream A"),
                ("done", 0, "stream A"),
                ("wait", 1, "stream B"),
                ("done", 1, "stream B"),
            ]
    finally:
        release.set()
        reader.close()


@pytest.mark.parametrize(
    "failure", ["read", "enqueue-drained", "enqueue-unknown", "wait"]
)
def test_deferred_failure_with_earlier_pending_work_retains_or_drains(
    ops, tmp_path, monkeypatch, failure
):
    from fastsafetensors.frameworks import TransferDrainedError

    deferred = DeferredOps()
    path = tmp_path / "data"
    path.write_bytes(bytes(range(24)))
    read = host.pread_into
    first_enqueued = threading.Event()

    def check_read(fd, view, offset):
        if offset == 8 and failure == "read":
            assert first_enqueued.wait(5)
            raise OSError("second read failed")
        return read(fd, view, offset)

    def enqueue(ticket):
        first_enqueued.set()
        if ticket.index == 1 and failure.startswith("enqueue"):
            if failure == "enqueue-drained":
                # Simulate the adapter having proved completion before raising.
                ticket.wait()
                raise TransferDrainedError("partial enqueue drained")
            raise RuntimeError("partial enqueue unknown")

    def wait(ticket):
        if failure == "wait":
            raise RuntimeError("wait completion unknown")

    deferred.after_enqueue, deferred.before_wait = enqueue, wait
    monkeypatch.setattr(host, "pread_into", check_read)
    reader = host.HostFileReader(deferred, max_threads=1)
    try:
        with path.open("rb") as file:
            entries = entries_for(file.fileno(), [24])
            destination = weakref.ref(entries[0][1].storage)
            request = submit(reader, entries)
            assert request.done.wait(5), "failure stranded pending work"
            with pytest.raises((RuntimeError, OSError), match="failed|unknown|drained"):
                reader.wait_read(request)
            assert not reader._state.pool.busy
            if failure in {"wait", "enqueue-unknown"}:
                with pytest.raises(RuntimeError, match="restart"):
                    submit(reader, entries)
            else:
                host.check_transfer_state()
                assert not deferred.active and not host._failed_transfers
    finally:
        reader.close()
    if failure == "wait":
        # Generic quarantine retains both buffers and destination after close.
        assert len(host._failed_transfers) == 2
        deferred.active.clear()
        del entries, request
        gc.collect()
        assert destination() is not None
        assert all(ref() is not None for ref in deferred.allocations)
