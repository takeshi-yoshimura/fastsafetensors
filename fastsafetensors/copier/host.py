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
from collections.abc import Callable, Generator, Iterator, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field

from ..frameworks import HostTransfer, HostTransferOp, TensorBase, TransferDrainedError
from ..st_types import Device

_HOST_BYTES = 16 << 20
_READ_BLOCK_BYTES = 256 << 20

# Also used by ordinary-copy/conversion fallbacks. Never hold this while waiting
# for a reader request: the reader worker needs it to make progress.
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


@dataclass(eq=False)
class _Group:
    request: HostReadRequest
    index: int
    offset: int
    size: int
    buffer: TensorBase
    reads: deque[Future] = field(default_factory=deque)
    ticket: HostTransfer | None = None


_ReadBlocks = Generator[tuple[int, memoryview, int], None, None]


def host_ranges(sizes: list[int], slot_bytes: int) -> Iterator[tuple[int, int, int]]:
    """Physical byte ranges; no logical slicing or destination reallocation."""
    for index, size in enumerate(sizes):
        for offset in range(0, size, slot_bytes):
            yield index, offset, min(slot_bytes, size - offset)


class _ReaderState:
    """The worker owns slot transitions; pread workers only fill their views."""

    def __init__(self, ops: HostTransferOp, max_threads: int, max_bytes: int) -> None:
        self.ops = ops
        self.max_threads = max_threads
        alignment = ops.transfer_alignment()
        if alignment <= 0:
            raise ValueError("transfer alignment must be positive")
        self.slot_bytes = max_bytes // (2 * alignment) * alignment
        if self.slot_bytes <= 0:
            raise ValueError(
                f"host capacity must fit two {alignment}-byte aligned slots"
            )
        self.buffers: list[TensorBase] = []
        self.condition = threading.Condition()
        self.pending: deque[HostReadRequest] = deque()
        self.requests: set[HostReadRequest] = set()
        self.closed = False
        self.thread: threading.Thread | None = None

    def complete(self, request: HostReadRequest) -> None:
        # Called only after all of this request's slots have retired.
        with self.condition:
            request.entries.clear()
            self.requests.discard(request)
            request.done.set()

    def fail(self, request: HostReadRequest, error: BaseException) -> None:
        if request.error is None:
            request.error = error.with_traceback(None)
        request.cancelled.set()

    def read_blocks(self, group: _Group) -> _ReadBlocks:
        request = group.request
        (fd, begin), _ = request.entries[group.index]
        with self.ops.host_buffer_view(group.buffer[: group.size]) as view:
            for offset in range(0, group.size, request.block_size):
                yield fd, view[
                    offset : offset + request.block_size
                ], begin + group.offset + offset

    def fill_reads(
        self, group: _Group, blocks: _ReadBlocks, executor: ThreadPoolExecutor
    ) -> None:
        while (
            len(group.reads) < self.max_threads and not group.request.cancelled.is_set()
        ):
            try:
                block = next(blocks, None)
                if block is None:
                    break
                group.reads.append(executor.submit(_read_block, *block))
            except BaseException as error:
                self.fail(group.request, error)

    def drain_reads(
        self, group: _Group, blocks: _ReadBlocks, executor: ThreadPoolExecutor
    ) -> None:
        # Do not cancel futures: queued work still owns its view and fd.
        while group.reads:
            try:
                group.reads.popleft().result()
            except BaseException as error:
                self.fail(group.request, error)
            self.fill_reads(group, blocks, executor)
        blocks.close()

    def transfer(self, group: _Group) -> None:
        request = group.request
        with TRANSFER_LOCK:
            if group.ticket is None:
                if request.cancelled.is_set():
                    return
                try:
                    check_transfer_state()
                except RuntimeError as error:
                    self.fail(request, error)
                    return
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
                    check_transfer_state()
                    group.ticket.wait()
                    group.ticket = None
            except BaseException as error:
                if group.ticket is not None or not isinstance(
                    error, TransferDrainedError
                ):
                    _failed_transfers.append((group.buffer, destination, group.ticket))
                    poison_transfer_state()
                self.fail(request, error)

    def pipeline(self, executor: ThreadPoolExecutor, slots: list[_Group]) -> None:
        # FIFO request cursor plus at most two occupied slots. Only this thread
        # changes slot state; no worker waits for another worker to free a slot.
        request: HostReadRequest | None = None
        ranges: Iterator[tuple[int, int, int]] = iter(())

        def complete_if_finished(owner: HostReadRequest) -> None:
            if request is not owner and not any(g.request is owner for g in slots):
                self.complete(owner)

        def start_read() -> tuple[_Group, _ReadBlocks] | None:
            nonlocal request, ranges
            if len(slots) == 2:
                return None
            while True:
                if request is None:
                    with self.condition:
                        if not self.pending:
                            return None
                        request = self.pending.popleft()
                    ranges = host_ranges(request.sizes, self.slot_bytes)
                try:
                    if not request.cancelled.is_set():
                        check_transfer_state()
                        region = next(ranges, None)
                        if region is not None:
                            if not self.buffers:
                                self.buffers = [
                                    self.ops.allocate_host_buffer(self.slot_bytes)
                                    for _ in range(2)
                                ]
                            break
                except BaseException as error:
                    self.fail(request, error)
                owner, request = request, None
                complete_if_finished(owner)
            buffer = next(
                b for b in self.buffers if all(g.buffer is not b for g in slots)
            )
            group = _Group(request, *region, buffer)
            slots.append(group)
            blocks = self.read_blocks(group)
            self.fill_reads(group, blocks, executor)
            return group, blocks

        reading = None
        while True:
            if reading is None:
                reading = start_read()
            if reading is not None:
                group, blocks = reading
                self.drain_reads(group, blocks, executor)
                # Start lookahead before enqueue, even for a synchronous adapter.
                reading = start_read()
                self.transfer(group)
                if group.ticket is not None:
                    continue
            elif slots:
                group = slots[0]
                self.transfer(group)
            else:
                with self.condition:
                    self.condition.wait_for(lambda: self.pending or self.closed)
                    if not self.pending:
                        return
                continue
            # A failed wait leaves the owners quarantined; poison bars reuse.
            slots.remove(group)
            complete_if_finished(group.request)
            del group

    def run(self) -> None:
        slots: list[_Group] = []
        try:
            with ThreadPoolExecutor(
                max_workers=self.max_threads, thread_name_prefix="fst-pread"
            ) as executor:
                self.pipeline(executor, slots)
        except BaseException as error:
            # Executor exit has drained reads. If the scheduler itself failed,
            # any ticket still in a slot must retain its owners until restart.
            with TRANSFER_LOCK:
                for group in slots:
                    if group.ticket is not None:
                        dst = group.request.entries[group.index][1]
                        _failed_transfers.append((group.buffer, dst, group.ticket))
                        poison_transfer_state()
            with self.condition:
                self.closed = True
                for request in list(self.requests):
                    self.fail(request, error)
                    self.complete(request)
                self.pending.clear()


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
            max(1, self._state.slot_bytes // max_threads),
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
                request.done.set()
                return request
            if state.thread is None:
                try:
                    thread = threading.Thread(
                        target=state.run, name="fst-host-reader", daemon=True
                    )
                    thread.start()
                    state.thread = thread
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
        self.wait_read(request)

    def close(self) -> None:
        state = self._state
        with state.condition:
            state.closed = True
            for request in state.requests:
                request.cancelled.set()
            state.condition.notify_all()
            thread = state.thread
        if thread is not None:
            thread.join()
        with state.condition:
            state.thread = None
            state.buffers.clear()

    def __del__(self) -> None:
        if hasattr(self, "_state"):
            self.close()
