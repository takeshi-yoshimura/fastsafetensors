# Copyright 2026 The Torch-Spyre Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Spyre-native checkpoint encoding, loading and host-to-device staging.

Importing this module registers the copier without importing torch-spyre.
Spyre operations load the optional dependency when needed. This module
schedules checkpoint encoding, reads, conversion, and transfers; torch-spyre
provides device primitives and flex owns IOMMU mapping. Buffers are reused only after the
current stream completes; failed synchronization retains buffers and
destinations because a shut-down stream may return before its DMA finishes.
"""

from __future__ import annotations

import json
import mmap
import os
import threading
import time
from collections import defaultdict
from collections.abc import Callable, Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Mapping

from .. import cpp as fstcpp
from ..common import SafeTensorsMetadata
from ..native import (
    NativeFormat,
    NativeFormatError,
    NativeTensorSpec,
    native_format,
    native_metadata,
    native_specs,
)
from ..saver import WriteEntry, tensor_fill
from ..st_types import Device, DeviceType, DType
from .base import CopierInterface, DummyDeviceBuffer
from .registry import register_copier_constructor

if TYPE_CHECKING:
    import torch

    from ..frameworks._torch import TorchTensor

DEVICE_NAME = "spyre"
COPIER_TYPE = "spyre_native"
FORMAT = "torch_spyre/1"
NATIVE_FORMAT = NativeFormat(device=DEVICE_NAME, format=FORMAT)
__all__ = [
    "COPIER_TYPE",
    "FORMAT",
    "NATIVE_FORMAT",
    "SpyreNativeCopier",
    "MAX_TRANSFER_BYTES",
    "TransferPipeline",
    "stage_to_spyre",
    "native_entry",
    "native_fill",
    "native_shard_metadata",
]
# Plain payloads that must be restored to Spyre, with an optional device layout.
KEY_MOVED = "torch_spyre.moved"


# Preserve the native checkpoint admission limit; larger or multi-chunk tensors
# use the ordinary copy_tensor path. This is no longer a pinned-pool size.
MAX_TRANSFER_BYTES = 4 << 30
_MAX_BATCH_BYTES = 4 << 30
_MAX_BUFFERS = 64
_READ_BLOCK_BYTES = 256 << 20

# Serialize these pipelines with their ordinary-copy fallback paths.
MAP_LOCK = threading.RLock()
# Retain uncertain in-flight memory until process exit; do not reuse this runtime.
_failed_transfers: list[tuple[list[torch.Tensor], list[torch.Tensor]]] = []


@dataclass(frozen=True)
class _NativePayload:
    src: torch.Tensor
    dst: torch.Tensor
    spec: NativeTensorSpec


def native_entry(
    name: str,
    src: torch.Tensor,
    dst: torch.Tensor,
    *,
    nbytes: int,
    layout: Any,
) -> WriteEntry:
    """Describe a host tensor encoded for an allocated Spyre destination.

    ``nbytes`` and ``layout`` describe dst's physical storage, as reported by
    torch-spyre. No conversion or device transfer happens until native_fill.
    """
    from ..frameworks._torch import fst_dtype

    if nbytes <= 0:
        raise ValueError(f"{name}: native payload size must be positive")
    spec = NativeTensorSpec(
        name,
        fst_dtype(dst.dtype),
        list(dst.shape),
        list(dst.stride()),
        layout,
        nbytes,
    )
    return WriteEntry(name, DType.U8, [nbytes], nbytes, _NativePayload(src, dst, spec))


def native_shard_metadata(entries: Sequence[WriteEntry]) -> dict[str, str]:
    """ParallelSaver callback describing this shard's Spyre-native payloads."""
    return native_metadata(
        NATIVE_FORMAT,
        {
            e.name: e.source.spec
            for e in entries
            if isinstance(e.source, _NativePayload)
        },
    )


def native_fill(work: list[tuple[memoryview, WriteEntry]]) -> None:
    """ParallelSaver callback encoding native entries and copying plain tensors.

    Conversion writes directly into the saver-owned buffers; allocated Spyre
    destinations describe the layout but are not filled by this operation.
    """
    plain, native = [], []
    for buf, entry in work:
        (native if isinstance(entry.source, _NativePayload) else plain).append(
            (buf, entry)
        )
    if plain:
        tensor_fill()(plain)
    if native:
        from torch_spyre._C import convert_h2d_into

        convert_h2d_into(
            [(buf, entry.source.src, entry.source.dst) for buf, entry in native],
            num_threads=min(64, os.cpu_count() or 1),
        )


@dataclass(frozen=True)
class FileRead:
    """Read device-format bytes from an open fd; the caller owns the fd."""

    fd: int
    offset: int


@dataclass
class TransferJob:
    """Fill nbytes of device-format data, then copy it directly to dst."""

    nbytes: int
    dst: torch.Tensor
    fill: Callable[[memoryview], None] | FileRead


def _allocate_host_buffer(nbytes: int) -> torch.Tensor:
    import torch

    return torch.frombuffer(mmap.mmap(-1, nbytes), dtype=torch.uint8)


class TransferPipeline:
    def __init__(
        self,
        device: torch.device,
        host_buffer_pool: list[torch.Tensor] | None = None,
    ) -> None:
        import torch

        self.device = torch.device(device)
        if self.device.type == "spyre" and self.device.index is None:
            self.device = torch.device(
                "spyre", torch.spyre.current_device()  # type: ignore[attr-defined]
            )
        self.routes: dict[str, list[int]] = defaultdict(lambda: [0, 0])
        self.fill_s = self.wait_s = 0.0
        self.peak_host_bytes = 0
        # A caller-owned pool lets a long-lived loader reuse buffers across
        # shard flushes. The default remains local to run(), as before.
        self._host_buffer_pool = host_buffer_pool

    def count(self, route: str, nbytes: int) -> None:
        self.routes[route][0] += 1
        self.routes[route][1] += nbytes

    def _batches(self, jobs: list[TransferJob]) -> Iterator[list[TransferJob]]:
        batch: list[TransferJob] = []
        nbytes = 0
        for job in sorted(jobs, key=lambda job: -job.nbytes):
            if batch and (
                len(batch) >= _MAX_BUFFERS or nbytes + job.nbytes > _MAX_BATCH_BYTES
            ):
                yield batch
                batch, nbytes = [], 0
            # A tensor larger than the batch target runs alone.
            batch.append(job)
            nbytes += job.nbytes
        if batch:
            yield batch

    def _read_files(self, work: list[tuple[TransferJob, torch.Tensor]]) -> None:
        if not work:
            return
        # nogds submissions are single-producer; its C++ workers fill the
        # caller-owned buffers directly on the CPU path.
        reader = fstcpp.nogds_file_reader(False, 1, min(_MAX_BUFFERS, 64), False, -1)
        pending: list[int] = []
        failed = False
        try:
            try:
                for job, buffer in work:
                    source = job.fill
                    assert isinstance(source, FileRead)
                    target = fstcpp.gds_device_buffer(
                        buffer.data_ptr(), job.nbytes, False
                    )
                    for offset in range(0, job.nbytes, _READ_BLOCK_BYTES):
                        length = min(_READ_BLOCK_BYTES, job.nbytes - offset)
                        req = reader.submit_read(
                            source.fd, target, source.offset + offset, length, offset
                        )
                        if req < 0:
                            raise OSError("nogds host read submission failed")
                        pending.append(req)
            finally:
                # wait_read returns a pointer, with zero indicating failure.
                # Drain all reads before the caller closes any fd or frees storage.
                for req in pending:
                    if reader.wait_read(req) == 0:
                        failed = True
            if failed:
                raise OSError("nogds host read failed")
        finally:
            # The reader destructor joins its workers even if waiting was interrupted.
            del reader

    @staticmethod
    def _batch_buffers(
        batch: list[TransferJob],
        pool: list[torch.Tensor],
    ) -> list[torch.Tensor]:
        """Return exact-size views for ``batch`` and update the bounded pool.

        Reuse is best-fit so large slots remain available for later large jobs.
        The DMA source view starts at the aligned allocation's base and has the
        exact transfer length expected by copyRaw.
        """
        limit = max(_MAX_BATCH_BYTES, max(job.nbytes for job in batch))
        sizes = [buffer.nbytes for buffer in pool]
        available = set(range(len(pool)))
        selected: list[tuple[int | None, int]] = []
        for job in batch:
            candidates = [
                (sizes[index], index)
                for index in available
                if sizes[index] >= job.nbytes
            ]
            if candidates:
                _, slot_index = min(candidates)
                available.remove(slot_index)
                selected.append((slot_index, job.nbytes))
            else:
                selected.append((None, job.nbytes))

        # A ragged batch can match a job to an oversized slot and otherwise
        # exceed the pool budget. Replace the most wasteful such matches with
        # exact allocations; the exact-size plan always fits the batch limit.
        active_bytes = sum(
            sizes[index] if index is not None else size for index, size in selected
        )
        while active_bytes > limit:
            replaceable = [
                (sizes[index] - size, pos)
                for pos, (index, size) in enumerate(selected)
                if index is not None and sizes[index] > size
            ]
            if not replaceable:
                raise RuntimeError("host buffer batch exceeds its staging limit")
            _, pos = max(replaceable)
            index, size = selected[pos]
            assert index is not None
            selected[pos] = (None, size)
            active_bytes += size - sizes[index]

        # Keep unused slots only while the whole pool stays within the same
        # bound. Drop the largest unused slots first, preserving small slots
        # that often fit ragged tail batches exactly.
        used = {index for index, _ in selected if index is not None}
        new_bytes = sum(size for index, size in selected if index is None)
        new_count = sum(index is None for index, _ in selected)
        unused = available.copy()
        retained = used | unused
        retained_bytes = sum(sizes[index] for index in retained)
        while (
            retained_bytes + new_bytes > limit
            or len(retained) + new_count > _MAX_BUFFERS
        ) and unused:
            evict = max(unused, key=lambda index: sizes[index])
            unused.remove(evict)
            retained.remove(evict)
            retained_bytes -= sizes[evict]

        # Remove slots that will be replaced or evicted before allocating their
        # replacements. This keeps even the transient allocation peak bounded.
        old_to_new = {old: new for new, old in enumerate(sorted(retained))}
        pool[:] = [pool[index] for index in sorted(retained)]
        active: list[torch.Tensor] = []
        views: list[torch.Tensor] = []
        for job, (index, size) in zip(batch, selected):
            buffer = (
                pool[old_to_new[index]]
                if index is not None
                else _allocate_host_buffer(size)
            )
            active.append(buffer)
            views.append(buffer[: job.nbytes])
        pool[:] = active + [pool[old_to_new[index]] for index in sorted(unused)]
        return views

    def run(self, jobs: list[TransferJob]) -> None:
        from torch_spyre._C import copy_raw_h2d_batch, dma_bytes

        if _MAX_BUFFERS < 1 or _MAX_BATCH_BYTES < 1 or _READ_BLOCK_BYTES < 1:
            raise ValueError("host staging limits must be positive")
        with MAP_LOCK:
            if _failed_transfers:
                raise RuntimeError(
                    "raw H2D completion previously failed; restart the process"
                )
            for job in jobs:
                if not 0 < job.nbytes <= MAX_TRANSFER_BYTES:
                    raise ValueError(f"invalid raw transfer size: {job.nbytes}")
                if job.dst.device != self.device:
                    raise ValueError(
                        "raw transfer destination is on a different device"
                    )
                if dma_bytes(job.dst) != job.nbytes:
                    raise ValueError("raw transfer size differs from destination")
                if isinstance(job.fill, FileRead) and (
                    job.fill.fd < 0 or job.fill.offset < 0
                ):
                    raise ValueError("invalid host read source")
            buffers = (
                self._host_buffer_pool if self._host_buffer_pool is not None else []
            )

            def fill(job: TransferJob, buffer: torch.Tensor) -> None:
                assert callable(job.fill)
                with memoryview(buffer.numpy()) as view:
                    job.fill(view[: job.nbytes])

            with ThreadPoolExecutor(min(_MAX_BUFFERS, max(1, len(jobs)))) as workers:
                for batch in self._batches(jobs):
                    batch_buffers = self._batch_buffers(batch, buffers)
                    self.peak_host_bytes = max(
                        self.peak_host_bytes, sum(buffer.nbytes for buffer in buffers)
                    )
                    t0 = time.perf_counter()
                    filled = [
                        workers.submit(fill, job, buffer)
                        for job, buffer in zip(batch, batch_buffers)
                        if callable(job.fill)
                    ]
                    self._read_files(
                        [
                            (job, buffer)
                            for job, buffer in zip(batch, batch_buffers)
                            if isinstance(job.fill, FileRead)
                        ]
                    )
                    for future in filled:
                        future.result()
                    self.fill_s += time.perf_counter() - t0
                    t0 = time.perf_counter()
                    try:
                        copy_raw_h2d_batch(
                            [
                                (buffer, job.dst)
                                for job, buffer in zip(batch, batch_buffers)
                            ]
                        )
                    except BaseException:
                        # Do not let a caller-owned pool reuse uncertain storage.
                        _failed_transfers.append(
                            (list(batch_buffers), [job.dst for job in batch])
                        )
                        raise
                    finally:
                        self.wait_s += time.perf_counter() - t0
                    for job in batch:
                        self.count("raw", job.nbytes)
                    del batch_buffers

    def summary(self) -> str:
        routes = ", ".join(
            f"{route}: {n} ({nbytes / 2**30:.2f} GiB)"
            for route, (n, nbytes) in self.routes.items()
        )
        return (
            f"peak host buffers {self.peak_host_bytes / 2**20:.0f} MiB; "
            f"fill {self.fill_s:.1f}s, DMA wait {self.wait_s:.1f}s; {routes}"
        )


def run_transfer_jobs(
    jobs: list[TransferJob], device: torch.device
) -> TransferPipeline:
    pipeline = TransferPipeline(device)
    pipeline.run(list(jobs))
    return pipeline


def stage_to_spyre(
    pairs: Sequence[tuple[torch.Tensor, torch.Tensor]], device: torch.device
) -> TransferPipeline:
    """Convert into aligned host buffers and raw-copy, with ordinary-copy fallback."""
    from torch_spyre._C import convert_h2d_into, copy_tensor, dma_bytes

    def convert(src: torch.Tensor, dst: torch.Tensor) -> Callable[[memoryview], None]:
        def fill(buffer: memoryview) -> None:
            convert_h2d_into([(buffer, src, dst)], num_threads=1)

        return fill

    jobs: list[TransferJob] = []
    direct: list[tuple[torch.Tensor, torch.Tensor]] = []
    for src, dst in pairs:
        nbytes = dma_bytes(dst)
        if 0 < nbytes <= MAX_TRANSFER_BYTES:
            jobs.append(TransferJob(nbytes, dst, convert(src, dst)))
        else:
            direct.append((src, dst))
    with MAP_LOCK:
        pipeline = run_transfer_jobs(jobs, device)
        for src, dst in direct:
            copy_tensor(src, dst, non_blocking=False)
            pipeline.count("copy_tensor (multi-chunk/over cap)", src.nbytes)
    return pipeline


def pread_into(fd: int, buf: memoryview, offset: int) -> None:
    done = 0
    while done < len(buf):
        n = os.preadv(fd, [buf[done:]], offset + done)
        if n <= 0:
            raise OSError(f"unexpected end of file at {offset + done}")
        done += n


class _Engine:
    """Collects one file batch's jobs and reuses host buffers across flushes."""

    def __init__(self) -> None:
        self.jobs: list[TransferJob] = []
        self.fds: list[int] = []
        self.lock = threading.Lock()
        self.last_pipeline: TransferPipeline | None = None
        self.host_buffers: list[torch.Tensor] = []

    def flush(self, device: torch.device) -> None:
        with self.lock:
            jobs, self.jobs = self.jobs, []
            fds, self.fds = self.fds, []
        try:
            if jobs:
                pipeline = TransferPipeline(device, host_buffer_pool=self.host_buffers)
                pipeline.run(jobs)
                with self.lock:
                    self.last_pipeline = pipeline
        finally:
            for fd in fds:
                os.close(fd)


class SpyreNativeCopier(CopierInterface):
    device_native = True
    owns_tensor_storage = True

    def __init__(
        self,
        metadata: SafeTensorsMetadata,
        device: Device,
        framework: Any,
        engine: _Engine,
        destinations: Mapping[str, torch.Tensor],
    ) -> None:
        self.metadata = metadata
        self.device = device
        self.framework = framework
        self.engine = engine
        self.destinations = destinations
        self.names: list[str] | None = None
        self.out: dict[str, TorchTensor] = {}
        self.moved: dict[str, Any] = {}

    def select_tensors(self, keep: Callable[[str], bool]) -> None:
        self.names = [n for n in self.metadata.tensors if keep(n)]

    def submit_io(self, use_buf_register: bool, max_copy_block_size: int):
        import torch
        from torch_spyre._C import (
            dma_bytes,
            get_spyre_tensor_layout,
            spyre_empty_with_layout,
        )
        from torch_spyre.layout import layout_from_json, layout_to_json

        from ..frameworks._torch import TorchTensor, dtype_convert

        fmt = native_format(self.metadata)
        if fmt is not None and (fmt.device != DEVICE_NAME or fmt.format != FORMAT):
            raise NativeFormatError(
                f"{self.metadata.src}: {fmt.device}/{fmt.format} payloads, "
                f"this copier reads {DEVICE_NAME}/{FORMAT}"
            )
        specs = native_specs(self.metadata)
        names = self.names if self.names is not None else list(self.metadata.tensors)
        torch_device = torch.device(self.device.as_str())
        md = self.metadata.metadata
        try:
            moved = json.loads(md.get(KEY_MOVED, "{}")) if isinstance(md, dict) else {}
            if not isinstance(moved, dict):
                raise ValueError("expected a name-to-layout object")
            for name in names:
                if name not in moved:
                    continue
                if name in specs:
                    raise ValueError(f"{name}: a native payload cannot also be moved")
                layout = moved[name]
                if layout is not None and not isinstance(layout, dict):
                    raise ValueError(f"{name}: layout must be an object or null")
                self.moved[name] = None if layout is None else layout_from_json(layout)
        except (KeyError, TypeError, ValueError) as e:
            raise NativeFormatError(f"{self.metadata.src}: bad {KEY_MOVED}: {e}") from e
        fd = os.open(self.metadata.src, os.O_RDONLY)
        jobs: list[TransferJob] = []
        # Everything is validated and allocated here, before any data moves.
        try:
            for name in names:
                frame = self.metadata.tensors[name]
                begin = self.metadata.header_length + frame.data_offsets[0]
                nbytes = frame.data_offsets[1] - frame.data_offsets[0]
                spec = specs.get(name)
                if spec is None:
                    host = torch.empty(frame.shape, dtype=dtype_convert[frame.dtype])
                    dst = self.destinations.get(name)
                    if name in self.moved and dst is not None:
                        if (
                            list(dst.shape) != list(host.shape)
                            or dst.dtype != host.dtype
                            or dst.device.type != torch_device.type
                            or (
                                torch_device.index is not None
                                and dst.device.index != torch_device.index
                            )
                            or (
                                moved[name] is not None
                                and layout_to_json(get_spyre_tensor_layout(dst))
                                != moved[name]
                            )
                        ):
                            raise NativeFormatError(
                                f"{name}: destination does not match the stored tensor"
                            )
                    if nbytes:
                        pread_into(
                            fd,
                            memoryview(host.view(-1).view(torch.uint8).numpy()),
                            begin,
                        )
                    self.out[name] = TorchTensor(
                        Device(DeviceType.CPU), frame.dtype, host
                    )
                    continue
                dst = self.destinations.get(name)
                if dst is None:
                    dst = spyre_empty_with_layout(
                        spec.shape,
                        spec.stride,
                        dtype_convert[spec.dtype],
                        layout_from_json(spec.layout),
                        device=torch_device,
                    )
                elif (
                    list(dst.shape) != spec.shape
                    or list(dst.stride()) != spec.stride
                    or dst.dtype != dtype_convert[spec.dtype]
                    or layout_to_json(get_spyre_tensor_layout(dst)) != spec.layout
                ):
                    raise NativeFormatError(
                        f"{name}: destination does not match the stored tensor"
                    )
                if dma_bytes(dst) != nbytes:
                    raise NativeFormatError(f"{name}: device size differs from payload")

                jobs.append(TransferJob(nbytes, dst, FileRead(fd, begin)))
                self.out[name] = TorchTensor(self.device, spec.dtype, dst)
        except BaseException:
            os.close(fd)
            raise
        with self.engine.lock:
            self.engine.jobs.extend(jobs)
            self.engine.fds.append(fd)
        return DummyDeviceBuffer()

    def finish_io(self) -> None:
        # Submission only stages validated jobs. Abort them without starting DMA
        # when a later shard fails validation; wait_io performs transfers itself.
        with self.engine.lock:
            self.engine.jobs.clear()
            fds, self.engine.fds = self.engine.fds, []
        for fd in fds:
            os.close(fd)
        self.out.clear()
        self.moved.clear()

    def wait_io(
        self, gbuf, dtype: DType = DType.AUTO, noalign: bool = False, owner=None
    ):
        import torch

        from ..frameworks._torch import TorchTensor

        device = torch.device(self.device.as_str())
        self.engine.flush(device)
        # This batch's raw reads/transfers have completed before ordinary copies.
        with MAP_LOCK:
            for name, layout in self.moved.items():
                source = self.out[name]
                host = source.get_raw()
                dst = self.destinations.get(name)
                if dst is not None:
                    from torch_spyre._C import copy_tensor

                    copy_tensor(host, dst, non_blocking=False)
                elif layout is None:
                    dst = host.to(device)
                else:
                    # torch-spyre extends Tensor.to with device_layout.
                    dst = host.to(device, dtype=host.dtype, device_layout=layout)  # type: ignore[call-overload]
                self.out[name] = TorchTensor(self.device, source.dtype, dst)
        self.moved.clear()
        out, self.out = self.out, {}
        return out


@register_copier_constructor(COPIER_TYPE, SpyreNativeCopier)
def new_spyre_native_copier(
    device: Device,
    destinations: Mapping[str, torch.Tensor] | None = None,
    engine: _Engine | None = None,
    **_ignored: Any,
):
    """``destinations`` (name -> Spyre tensor) are filled in place of new tensors."""
    if device.type != DeviceType.SPYRE:
        raise ValueError("spyre_native copier requires device='spyre'")
    framework = _ignored.get("framework")
    if framework is not None and framework.get_name() != "pytorch":
        raise ValueError("spyre_native copier requires the pytorch framework")
    try:
        import torch_spyre  # noqa: F401 -- registers the torch device
    except ModuleNotFoundError as e:
        if e.name != "torch_spyre":
            raise
        raise ImportError(
            "spyre_native copier requires torch-spyre; install torch-spyre "
            "to load onto Spyre"
        ) from e
    shared = engine or _Engine()

    def construct(metadata, device, framework):
        return SpyreNativeCopier(
            metadata, device, framework, shared, destinations or {}
        )

    return construct
