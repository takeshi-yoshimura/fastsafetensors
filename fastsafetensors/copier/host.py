# SPDX-License-Identifier: Apache-2.0
"""Bounded preadv -> enqueue -> deferred completion pipeline, not a load planner.

The loader chooses files/chunks; copiers own descriptors and destinations.
This reader executes their requests without requiring wait_read() for progress.
No device pointer arithmetic, checkpoint format or raw framework tensors live here.
"""

from __future__ import annotations

import os
import threading
from collections import deque
from collections.abc import Callable, Iterator, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field

from ..frameworks import HostTransfer, HostTransferOp, TensorBase, TransferDrainedError
from ..st_types import Device

_HOST_BYTES = 16 << 20
_READ_BLOCK_BYTES = 256 << 20

# Also used by ordinary-copy/conversion fallbacks. Never hold this while waiting
# for a reader request: the copy worker needs it to make progress.
TRANSFER_LOCK = threading.RLock()
_failed_transfers: list[object] = []
_transfer_poisoned = False


def poison_transfer_state() -> None:
    global _transfer_poisoned
    _transfer_poisoned = True


def check_transfer_state() -> None:
    if _failed_transfers or _transfer_poisoned:
        raise RuntimeError("raw H2D completion previously failed; restart the process")


def pread_into(fd: int, buf: memoryview, offset: int) -> None:
    """Read directly into the supplied writable host view, including short reads."""
    done = 0
    while done < len(buf):
        try:
            count = os.preadv(fd, [buf[done:]], offset + done)
        except InterruptedError:
            continue
        if count <= 0:
            raise OSError(f"unexpected end of file at {offset + done}")
        done += count


def _read_block(fd: int, view: memoryview, offset: int) -> None:
    # Executor work items can retain their arguments briefly after Future.result
    # returns. Release the export before completion advertises buffer reuse.
    with view:
        pread_into(fd, view, offset)


class HostBufferPool:
    """Exactly two fixed-capacity slots, shared by all files in one loader."""

    def __init__(self, ops: HostTransferOp, max_bytes: int) -> None:
        alignment = ops.transfer_alignment()
        if alignment <= 0:
            raise ValueError("transfer alignment must be positive")
        self.slot_bytes = max_bytes // (2 * alignment) * alignment
        if self.slot_bytes <= 0:
            raise ValueError(
                f"host capacity must fit two {alignment}-byte aligned slots"
            )
        self.ops = ops
        self.max_bytes = max_bytes
        self.buffers: list[TensorBase] = []
        self.busy: set[int] = set()
        self.condition = threading.Condition()
        self.peak_bytes = 0

    def wake(self) -> None:
        with self.condition:
            self.condition.notify_all()

    def acquire(self, cancelled: Callable[[], bool]) -> TensorBase | None:
        with self.condition:
            while True:
                if cancelled():
                    return None
                check_transfer_state()
                if not self.buffers:
                    self.buffers = [
                        self.ops.allocate_host_buffer(self.slot_bytes) for _ in range(2)
                    ]
                    self.peak_bytes = 2 * self.slot_bytes
                for buffer in self.buffers:
                    if id(buffer) not in self.busy:
                        self.busy.add(id(buffer))
                        return buffer
                self.condition.wait()

    def release(self, buffer: TensorBase) -> None:
        with self.condition:
            self.busy.remove(id(buffer))
            self.condition.notify_all()


Copy = Callable[[list[tuple[TensorBase, TensorBase, int, int]]], HostTransfer | None]
FileEntry = tuple[tuple[int, int], TensorBase]


@dataclass(eq=False)
class HostReadRequest:
    entries: list[FileEntry]
    sizes: list[int]
    copy: Copy
    block_size: int
    cancelled: threading.Event = field(default_factory=threading.Event)
    done: threading.Event = field(default_factory=threading.Event)
    error: BaseException | None = None
    producing: bool = True
    inflight: int = 0


@dataclass
class _Group:
    request: HostReadRequest
    index: int
    offset: int
    size: int
    buffer: TensorBase
    ticket: HostTransfer | None = None


def host_ranges(sizes: list[int], slot_bytes: int) -> Iterator[tuple[int, int, int]]:
    """Physical byte ranges; no logical slicing or destination reallocation."""
    for index, size in enumerate(sizes):
        for offset in range(0, size, slot_bytes):
            yield index, offset, min(slot_bytes, size - offset)


class _ReaderState:
    """Workers own state, not HostFileReader, so dropping the owner can join them."""

    def __init__(self, ops: HostTransferOp, max_threads: int, max_bytes: int) -> None:
        self.ops = ops
        self.max_threads = max_threads
        self.pool = HostBufferPool(ops, max_bytes)
        self.condition = threading.Condition()
        self.pending: deque[HostReadRequest] = deque()
        self.ready: deque[_Group] = deque()
        self.requests: set[HostReadRequest] = set()
        self.closed = False
        self.producer_done = False
        self.threads: list[threading.Thread] = []

    def complete(self, request: HostReadRequest) -> None:
        # Called under condition, only after reads AND copies have drained.
        if not request.producing and request.inflight == 0:
            request.entries.clear()
            self.requests.discard(request)
            request.done.set()

    def fail(self, request: HostReadRequest, error: BaseException) -> None:
        with self.condition:
            if request.error is None:
                request.error = error.with_traceback(None)
            request.cancelled.set()
        self.pool.wake()

    def read_group(self, group: _Group, executor: ThreadPoolExecutor) -> None:
        request = group.request

        def blocks():
            (fd, begin), _ = request.entries[group.index]
            with self.ops.host_buffer_view(group.buffer[: group.size]) as view:
                size = group.size
                for offset in range(0, size, request.block_size):
                    yield fd, view[
                        offset : offset + request.block_size
                    ], begin + group.offset + offset

        futures: deque[Future] = deque()
        try:
            for block in blocks():
                if request.cancelled.is_set():
                    break
                # Bound outstanding futures even for tiny read block settings.
                if len(futures) >= self.max_threads:
                    futures.popleft().result()
                futures.append(executor.submit(_read_block, *block))
            while futures:
                futures.popleft().result()
        finally:
            # A submit/read failure or cancellation cannot let fd/views escape
            # before all started syscalls finish. Do not mask the first error.
            # Do not merely cancel futures: a cancelled executor work item may
            # still retain its host view until a worker dequeues it.
            for future in futures:
                try:
                    future.result()
                except BaseException:
                    pass

    def produce(self) -> None:
        try:
            with ThreadPoolExecutor(
                max_workers=self.max_threads, thread_name_prefix="fst-pread"
            ) as executor:
                while True:
                    with self.condition:
                        self.condition.wait_for(lambda: self.pending or self.closed)
                        if not self.pending:
                            return
                        request = self.pending.popleft()
                    try:
                        for index, offset, size in host_ranges(
                            request.sizes, self.pool.slot_bytes
                        ):
                            buffer = self.pool.acquire(request.cancelled.is_set)
                            if buffer is None:
                                break
                            group = _Group(request, index, offset, size, buffer)
                            del buffer
                            try:
                                self.read_group(group, executor)
                            except BaseException:
                                self.pool.release(group.buffer)
                                del group
                                raise
                            with self.condition:
                                request.inflight += 1
                                self.ready.append(group)
                                self.condition.notify_all()
                            del group
                    except BaseException as error:
                        self.fail(request, error)
                    finally:
                        with self.condition:
                            request.producing = False
                            self.complete(request)
                            self.condition.notify_all()
        except BaseException as error:
            # Executor construction/teardown failures must not strand waiters.
            with self.condition:
                self.closed = True
                for request in list(self.requests):
                    self.fail(request, error)
                    request.producing = False
                    self.complete(request)
                self.pending.clear()
        finally:
            with self.condition:
                self.producer_done = True
                self.condition.notify_all()

    def transfer(self, group: _Group) -> None:
        request = group.request
        with TRANSFER_LOCK:
            if group.ticket is None:
                if request.cancelled.is_set():
                    return
                check_transfer_state()
            destination = request.entries[group.index][1]
            try:
                if group.ticket is None:
                    group.ticket = request.copy(
                        [
                            (
                                group.buffer[: group.size],
                                destination,
                                group.offset,
                                group.size,
                            )
                        ]
                    )
                else:
                    # Already-started work must be drained even after cancel.
                    group.ticket.wait()
                    group.ticket = None
            except BaseException as error:
                if group.ticket is not None or not isinstance(
                    error, TransferDrainedError
                ):
                    _failed_transfers.append((group.buffer, destination, group.ticket))
                    poison_transfer_state()
                raise

    def consume(self) -> None:
        pending: deque[_Group] = deque()
        while True:
            with self.condition:
                # Fill both slots before waiting. A single/tail transfer must
                # progress without another submit_read, wait_read or close call.
                self.condition.wait_for(
                    lambda: self.ready
                    or self.producer_done
                    or len(pending) == 2
                    or (pending and not any(r.producing for r in self.requests))
                )
                if pending and (len(pending) == 2 or not self.ready):
                    group = pending.popleft()
                elif self.ready:
                    group = self.ready.popleft()
                else:
                    return
            request = group.request
            try:
                self.transfer(group)
            except BaseException as error:
                self.fail(request, error)
            else:
                if group.ticket is not None:
                    pending.append(group)
                    continue
            # Failed transfers are quarantined; poison prevents slot reuse.
            self.pool.release(group.buffer)
            with self.condition:
                request.inflight -= 1
                self.complete(request)
            del group, request


class HostFileReader:
    """Loader-shared two-slot reader with bounded enqueue and deferred waits."""

    def __init__(
        self,
        ops: HostTransferOp,
        max_threads: int = 16,
        block_size: int = _READ_BLOCK_BYTES,
        *,
        max_bytes: int | None = None,
    ) -> None:
        if max_threads <= 0 or block_size <= 0:
            raise ValueError("read block size and thread count must be positive")
        self._state = _ReaderState(
            ops, max_threads, _HOST_BYTES if max_bytes is None else max_bytes
        )
        self.block_size = min(
            block_size,
            _READ_BLOCK_BYTES,
            max(1, self._state.pool.slot_bytes // max_threads),
        )

    def submit_read(
        self,
        entries: Sequence[FileEntry],
        device: Device,
        max_copy_block_size: int = _READ_BLOCK_BYTES,
    ) -> HostReadRequest:
        if max_copy_block_size <= 0:
            raise ValueError("max_copy_block_size must be positive")
        state = self._state
        check_transfer_state()
        sizes = [state.ops.validate_transfer(dst, device) for _, dst in entries]
        if any(size <= 0 for size in sizes):
            raise ValueError("raw transfer sizes must be positive")
        request = HostReadRequest(
            list(entries),
            sizes,
            state.ops.prepare_copy(device) if entries else lambda entries: None,
            min(self.block_size, max_copy_block_size),
        )
        with state.condition:
            if state.closed:
                raise RuntimeError("host file reader is closed")
            if not entries:
                request.producing = False
                request.done.set()
                return request
            if not state.threads:
                try:
                    for target, name in (
                        (state.produce, "fst-host-read"),
                        (state.consume, "fst-host-copy"),
                    ):
                        thread = threading.Thread(target=target, name=name, daemon=True)
                        thread.start()
                        state.threads.append(thread)
                except BaseException:
                    state.closed = True
                    state.condition.notify_all()
                    raise
            state.requests.add(request)
            state.pending.append(request)
            state.condition.notify_all()
        return request

    def wait_read(self, request: HostReadRequest) -> None:
        request.done.wait()
        if request.error is not None:
            raise request.error

    def cancel_read(self, request: HostReadRequest) -> None:
        request.cancelled.set()
        self._state.pool.wake()
        self.wait_read(request)

    def close(self) -> None:
        state = self._state
        with state.condition:
            state.closed = True
            for request in state.requests:
                request.cancelled.set()
            state.condition.notify_all()
        state.pool.wake()
        for thread in state.threads:
            thread.join()
        state.threads.clear()
        state.pool.buffers.clear()

    def __del__(self) -> None:
        if hasattr(self, "_state"):
            self.close()
