# SPDX-License-Identifier: Apache-2.0

"""Unified memory copier for systems with shared CPU/GPU memory (DGX Spark, Grace Hopper).

Uses mmap → pin_memory → cudaMemcpyAsync instead of the bounce buffer approach.
On unified memory with ATS, pin_memory on mmap'd pages triggers kernel readahead
and page pinning in a single optimized path, then async DMA transfers at full
memory bandwidth.

All framework-specific operations (mmap + pinning, device synchronization,
device-name detection) go through the FrameworkOpBase abstraction so this
module never imports torch or paddle directly.
"""

import os
import sys
import threading
import time
from typing import Dict, List, Optional, Set, Tuple

from .. import cpp as fstcpp
from ..common import SafeTensorsMetadata, get_fs_type, init_logger
from ..frameworks import FrameworkOpBase, TensorBase
from ..st_types import Device, DeviceType, DType
from .base import (
    CopierInterface,
    validated_byte_ranges,
    validated_chunk_allocation_size,
)
from .registry import CopierConstructFunc, register_copier_constructor

logger = init_logger(__name__)

# O_DIRECT bypasses the page cache, which wins on local block devices but
# forfeits kernel readahead / client caching on network filesystems, where the
# buffered mmap + pin path performs better. Gate the fast path by fs type;
# FASTSAFETENSORS_ODIRECT=1/0 forces it on/off regardless.
_NETWORK_FS = {
    "nfs",
    "nfs4",
    "cifs",
    "smb3",
    "smbfs",
    "sshfs",
    "fuse.sshfs",
    "lustre",
    "gpfs",
    "beegfs",
    "glusterfs",
    "ceph",
    "9p",
    "virtiofs",
}
_warned_fs: set = set()


def _allocation_length(required: int) -> int:
    """Round transient buffers so PyTorch's allocator can reuse them.

    Safetensors shards commonly differ by a few hundred MiB. Allocating their
    exact lengths leaves differently-sized blocks in the CUDA caching allocator
    and repeatedly grows its pool. An opt-in granularity makes consecutive
    buffers interchangeable without changing tensor offsets or bytes read.
    """
    granularity_mb = int(os.environ.get("FASTSAFETENSORS_ALLOC_GRANULARITY_MB", "0"))
    if granularity_mb < 0:
        raise ValueError("FASTSAFETENSORS_ALLOC_GRANULARITY_MB must be >= 0")
    if granularity_mb == 0 or required == 0:
        return required
    granularity = granularity_mb * 1024 * 1024
    return ((required + granularity - 1) // granularity) * granularity


def _odirect_ok(path: str) -> bool:
    override = os.environ.get("FASTSAFETENSORS_ODIRECT")
    if override is not None:
        return override == "1"
    fstype = get_fs_type(path)
    if fstype in _NETWORK_FS:
        if fstype not in _warned_fs:
            _warned_fs.add(fstype)
            logger.info(
                "checkpoint on network filesystem (%s): using buffered reads "
                "instead of O_DIRECT (set FASTSAFETENSORS_ODIRECT=1 to force)",
                fstype,
            )
        return False
    return True


class UnifiedMemCopier(CopierInterface):
    """Copier using mmap → pin_memory → cudaMemcpyAsync for unified memory.

    On systems where CPU and GPU share the same physical memory (DGX Spark,
    Grace Hopper), this avoids the unnecessary bounce buffer used by NoGdsFileCopier.
    The mmap + pin_memory path lets the kernel handle readahead and page pinning
    in a single step, then async DMA copies at full memory bandwidth.
    """

    def __init__(
        self,
        metadata: SafeTensorsMetadata,
        device: Device,
        framework: FrameworkOpBase,
    ):
        self.metadata = metadata
        self.device = device
        self.framework = framework
        self._pinned: List[TensorBase] = []
        self.byte_ranges: Optional[List[Tuple[int, int]]] = None
        self._chunk_names: Optional[Set[str]] = None
        self._chunk_allocation_size: Optional[int] = None
        self._base_off = metadata.header_length
        self._submit_path = "unknown"
        self._materialize_thread: Optional[threading.Thread] = None
        self._materialized: Optional[Dict[str, TensorBase]] = None
        self._materialize_error: Optional[BaseException] = None
        self._materialize_ms = 0.0
        # Worker count for the C++ O_DIRECT range reader (dma_load_runs). Single
        # -thread pin_memory is page-cache-bound (~2.5 GB/s); O_DIRECT threads
        # bypass the cache and drive NVMe queue depth. Falls back to pin_memory
        # if the reader is unavailable.
        self._dma_threads = int(os.environ.get("FASTSAFETENSORS_DMA_THREADS", "8"))

    def _start_materialization(self, gbuf: fstcpp.gds_device_buffer) -> None:
        """Create tensor views while O_DIRECT workers populate ``gbuf``.

        Constructing DLPack/Torch views only records addresses, shapes, and
        strides; it does not read tensor contents. Consumers cannot observe the
        views until submit_io has completed and wait_io joins this thread.
        """
        if os.environ.get("FASTSAFETENSORS_OVERLAP_MATERIALIZE", "1") != "1":
            return

        self._materialized = None
        self._materialize_error = None
        self._materialize_ms = 0.0

        def materialize() -> None:
            begin = time.perf_counter()
            try:
                self.framework.set_device(self.device)
                self._materialized = self.metadata._get_tensors(
                    gbuf,
                    self.device,
                    self._base_off,
                    names=self._chunk_names,
                )
            except BaseException as exc:
                self._materialize_error = exc
            finally:
                self._materialize_ms = (time.perf_counter() - begin) * 1000

        self._materialize_thread = threading.Thread(
            target=materialize,
            name="fastsafetensors-materialize",
        )
        self._materialize_thread.start()

    def _finish_materialization(self) -> Optional[Dict[str, TensorBase]]:
        thread = self._materialize_thread
        if thread is None:
            return None
        thread.join()
        self._materialize_thread = None
        if self._materialize_error is not None:
            error = self._materialize_error
            self._materialize_error = None
            raise error
        tensors = self._materialized
        self._materialized = None
        return tensors

    def set_byte_ranges(self, byte_ranges: Optional[List[Tuple[int, int]]]) -> None:
        """Restrict reads to these ``[start, end)`` absolute file-offset runs.

        Only the bytes in the given runs are mmap-faulted, pinned, and copied;
        the rest of the device buffer is left uninitialized (so the corresponding
        tensors must not be requested). Tensor offsets are unchanged. ``None``
        reads the whole data section. Build runs with
        ``SafeTensorsMetadata.select_byte_ranges``.
        """
        self.byte_ranges = validated_byte_ranges(self.metadata, byte_ranges)

    def set_chunk(
        self,
        byte_ranges: List[Tuple[int, int]],
        names: Set[str],
        allocation_size: Optional[int] = None,
    ) -> None:
        """Load only ``names`` into a compact or fixed-size device buffer.

        Allocates ``max_end - min_start`` of the runs by default, or
        ``allocation_size`` when provided, and instantiates just ``names``.
        This supports bounded sub-file batching and fixed allocation buckets.
        """
        self.byte_ranges = validated_byte_ranges(self.metadata, byte_ranges)
        self._chunk_names = names
        self._chunk_allocation_size = validated_chunk_allocation_size(
            self.byte_ranges, allocation_size
        )

    @classmethod
    def chunk_transient_multiplier(cls, paths: List[str]) -> int:
        """Per in-flight-chunk transient cost, as a multiple of chunk span.

        The O_DIRECT reader (dma_load_runs) reads runs straight into the device
        buffer: each live chunk costs ~1x its span plus a small fixed thread
        pool (measured +~150 MB on GB10 regardless of chunk size; not charged
        here). The mmap+pin_memory fallback additionally pins the chunk's file
        pages for the copy's lifetime; on unified-memory systems both draws
        come from one physical pool, so each live chunk costs ~2x its span.
        Mirrors submit_io's own path selection.
        """
        if getattr(fstcpp, "dma_load_runs", None) is None:
            return 2
        if int(os.environ.get("FASTSAFETENSORS_DMA_THREADS", "8")) <= 0:
            return 2
        if any(not _odirect_ok(p) for p in paths):
            return 2
        return 1

    def submit_io(
        self, use_buf_register: bool, max_copy_block_size: int
    ) -> fstcpp.gds_device_buffer:
        header_length = self.metadata.header_length

        # Default to the whole data section, reproducing the full-file read.
        # An empty list (vs None) reads nothing — same semantics as nogds.
        runs = self.byte_ranges
        if runs is None:
            runs = [(header_length, self.metadata.size_bytes)]

        if self._chunk_names is not None:
            # Compact chunk: allocate only the runs' span and map gbuf[0] to the
            # first run's start, so peak memory tracks the chunk, not the shard.
            base_off = min(s for s, _ in runs)
            chunk_span = max(e for _, e in runs) - base_off
            alloc_length = self._chunk_allocation_size or chunk_span
        else:
            base_off = header_length
            alloc_length = self.metadata.size_bytes - header_length
        self._base_off = base_off

        # Allocate CUDA buffer via framework's allocator (proper lifecycle)
        profile = os.environ.get("FASTSAFETENSORS_PROFILE") == "1"
        submit_begin = time.perf_counter() if profile else 0.0
        buffer_length = _allocation_length(alloc_length)
        gbuf = self.framework.alloc_tensor_memory(buffer_length, self.device)
        alloc_done = time.perf_counter() if profile else 0.0

        # Fast path: multithreaded O_DIRECT reader copies only the runs straight
        # into gbuf (byte F -> gbuf[F - base_off]), bypassing the page cache and
        # single-thread pin. Works for both full and compact-chunk buffers.
        # FASTSAFETENSORS_DMA_THREADS=0 disables it (falls back to mmap + pin);
        # network filesystems fall back automatically (see _odirect_ok).
        dma_load_runs = getattr(fstcpp, "dma_load_runs", None)
        dma_attempt_ms = 0.0
        if (
            dma_load_runs is not None
            and self._dma_threads > 0
            and _odirect_ok(self.metadata.src)
        ):
            starts = [s for s, _ in runs]
            ends = [e for _, e in runs]
            # The reader's worker threads must select this loader's device
            # before any CUDA call (thread-local current device defaults to 0);
            # -1 = cpu device, leave the workers' default untouched.
            if self.device.type == DeviceType.CPU:
                device_id = -1
            else:
                device_id = self.device.index if self.device.index is not None else 0
            self._start_materialization(gbuf)
            rc = dma_load_runs(
                gbuf.get_base_address(),
                self.metadata.src,
                base_off,
                starts,
                ends,
                self._dma_threads,
                device_id,
            )
            if profile:
                dma_attempt_ms = (time.perf_counter() - alloc_done) * 1000
            if rc == 0:
                self._submit_path = "dma"
                if profile:
                    submit_done = time.perf_counter()
                    print(
                        f"[FST_PROFILE] submit path=dma file={self.metadata.src} "
                        f"bytes={alloc_length} capacity={buffer_length} "
                        f"alloc_ms={(alloc_done - submit_begin) * 1000:.3f} "
                        f"dma_ms={(submit_done - alloc_done) * 1000:.3f} "
                        f"total_ms={(submit_done - submit_begin) * 1000:.3f}",
                        file=sys.stderr,
                        flush=True,
                    )
                return gbuf
            # Non-zero: reader unusable here; fall back to mmap + pin_memory.
            if profile:
                print(
                    f"[FST_PROFILE] dma_fallback file={self.metadata.src} "
                    f"rc={rc} dma_attempt_ms={dma_attempt_ms:.3f}",
                    file=sys.stderr,
                    flush=True,
                )

        base_address = gbuf.get_base_address()
        self._pinned = []
        self._submit_path = "mmap"
        pin_ms = 0.0
        copy_submit_ms = 0.0
        copied_bytes = 0
        for start, end in runs:
            # mmap_file_pinned faults in + pins only this run's pages.
            stage_begin = time.perf_counter() if profile else 0.0
            pinned = self.framework.mmap_file_pinned(
                self.metadata.src, end - start, start
            )
            if profile:
                pin_ms += (time.perf_counter() - stage_begin) * 1000
            self._pinned.append(pinned)
            stage_begin = time.perf_counter() if profile else 0.0
            ret = fstcpp.memcpy_h2d_async(  # type: ignore[attr-defined]
                base_address + (start - base_off), pinned.data_ptr(), end - start
            )
            if profile:
                copy_submit_ms += (time.perf_counter() - stage_begin) * 1000
                copied_bytes += end - start
            if ret != 0:
                try:
                    self._finish_materialization()
                finally:
                    self.framework.free_tensor_memory(gbuf, self.device)
                self._pinned = []
                raise RuntimeError(
                    f"cudaMemcpyAsync failed with error {ret} for {self.metadata.src}"
                )

        if profile:
            submit_done = time.perf_counter()
            print(
                f"[FST_PROFILE] submit path=mmap file={self.metadata.src} "
                f"runs={len(runs)} bytes={copied_bytes} capacity={buffer_length} "
                f"alloc_ms={(alloc_done - submit_begin) * 1000:.3f} "
                f"dma_attempt_ms={dma_attempt_ms:.3f} pin_ms={pin_ms:.3f} "
                f"copy_submit_ms={copy_submit_ms:.3f} "
                f"total_ms={(submit_done - submit_begin) * 1000:.3f}",
                file=sys.stderr, flush=True,
            )

        return gbuf

    def wait_io(
        self,
        gbuf: fstcpp.gds_device_buffer,
        dtype: DType = DType.AUTO,
        noalign: bool = False,
    ) -> Dict[str, TensorBase]:
        profile = os.environ.get("FASTSAFETENSORS_PROFILE") == "1"
        wait_begin = time.perf_counter() if profile else 0.0
        self.framework.synchronize(self.device)
        sync_done = time.perf_counter() if profile else 0.0

        # Alignment note: unlike the GDS copier, we only copy the data section
        # (not the header) into gbuf, so gbuf starts at a CUDA-allocator-aligned
        # address. The copy_start_offset=header_length cancels out in get_tensors'
        # pointer arithmetic, giving correct offsets. No memmove fixup needed.
        tensors = self._finish_materialization() if dtype == DType.AUTO else None
        if tensors is None:
            # A speculative AUTO materialization, if any, must finish before a
            # dtype-converting pass can safely replace it.
            self._finish_materialization()
            tensors = self.metadata._get_tensors(
                gbuf,
                self.device,
                self._base_off,
                dtype=dtype,
                names=self._chunk_names,
            )

        # Release pinned mmap pages and time unpin separately.
        tensors_done = time.perf_counter() if profile else 0.0
        pinned_runs = len(self._pinned)
        self._pinned = []
        if profile:
            release_done = time.perf_counter()
            print(
                f"[FST_PROFILE] materialize path={self._submit_path} "
                f"file={self.metadata.src} tensors={len(tensors)} pinned_runs={pinned_runs} "
                f"sync_ms={(sync_done - wait_begin) * 1000:.3f} "
                f"dlpack_ms={(tensors_done - sync_done) * 1000:.3f} "
                f"dlpack_worker_ms={self._materialize_ms:.3f} "
                f"release_ms={(release_done - tensors_done) * 1000:.3f} "
                f"total_ms={(release_done - wait_begin) * 1000:.3f}",
                file=sys.stderr, flush=True,
            )

        return tensors


def is_unified_memory_system(framework: Optional[FrameworkOpBase] = None) -> bool:
    """Detect if this system has unified CPU/GPU memory.

    Currently verified on DGX Spark (GB10). Other unified memory
    platforms (Grace Hopper GH200) may also benefit but are untested.

    Can be overridden via the FASTSAFETENSORS_UNIFIED_MEM environment
    variable: set to "1" to force enable, "0" to force disable.
    Device-name detection requires *framework*; with framework=None only
    the environment override can enable it.
    """
    override = os.environ.get("FASTSAFETENSORS_UNIFIED_MEM")
    if override is not None:
        return override == "1"

    if framework is None:
        return False
    return "gb10" in framework.get_device_name(0).lower()


@register_copier_constructor("unified", UnifiedMemCopier)
def new_unified_copier(device: Device, **kwargs) -> CopierConstructFunc:
    """Factory function for UnifiedMemCopier.

    Returns a constructor that creates UnifiedMemCopier instances.
    """
    from .nogds import load_library_func

    load_library_func(kwargs.get("framework"))

    def construct_unified_copier(
        metadata: SafeTensorsMetadata,
        device: Device,
        framework: FrameworkOpBase,
    ) -> CopierInterface:
        return UnifiedMemCopier(metadata, device, framework)

    return construct_unified_copier
