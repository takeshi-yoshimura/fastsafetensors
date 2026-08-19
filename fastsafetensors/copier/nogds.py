# SPDX-License-Identifier: Apache-2.0

import os
import sys
import time
from typing import Dict, List, Optional, Set, Tuple

from .. import cpp as fstcpp
from ..common import (
    SafeTensorsMetadata,
    get_device_numa_node,
    is_gpu_found,
    resolve_runtime_lib_name,
)
from ..frameworks import FrameworkOpBase, TensorBase
from ..st_types import Device, DeviceType, DType
from .base import (
    CopierInterface,
    validated_byte_ranges,
    validated_chunk_allocation_size,
)
from .registry import CopierConstructFunc, register_copier_constructor

_MIN_COPY_BLOCK_SIZE = 256 * 1024 * 1024
_COPY_MODE_ENV = "FASTSAFETENSORS_NOGDS_COPY_MODE"
_MAX_THREADS_ENV = "FASTSAFETENSORS_NOGDS_MAX_THREADS"


def _use_async_copy() -> bool:
    copy_mode = os.environ.get(_COPY_MODE_ENV, "sync").lower()
    if copy_mode not in {"sync", "async"}:
        raise ValueError(
            f"{_COPY_MODE_ENV} must be 'sync' or 'async', got {copy_mode!r}"
        )
    return copy_mode == "async"


def _resolve_max_threads(configured: int) -> int:
    raw_value = os.environ.get(_MAX_THREADS_ENV)
    if raw_value is None:
        max_threads = configured
    else:
        try:
            max_threads = int(raw_value)
        except ValueError:
            raise ValueError(
                f"{_MAX_THREADS_ENV} must be a positive integer, got {raw_value!r}"
            ) from None
    if max_threads <= 0:
        source = _MAX_THREADS_ENV if raw_value is not None else "max_threads"
        raise ValueError(f"{source} must be positive, got {max_threads}")
    return max_threads


class NoGdsFileCopier(CopierInterface):
    def __init__(
        self,
        metadata: SafeTensorsMetadata,
        device: Device,
        reader: fstcpp.nogds_file_reader,
        framework: FrameworkOpBase,
        max_threads: int = 16,
    ):
        if max_threads <= 0:
            raise ValueError("max_threads must be positive")
        self.framework = framework
        self.metadata = metadata
        self.reader = reader
        self.max_threads = max_threads
        flags = os.O_RDONLY
        # On Windows, O_RDONLY defaults to text mode which translates \r\n
        # and stops at 0x1A (Ctrl+Z), corrupting binary tensor data.
        if sys.platform == "win32" and hasattr(os, "O_BINARY"):
            flags |= os.O_BINARY
        self.fd = os.open(metadata.src, flags, 0o644)
        if self.fd < 0:
            raise Exception(
                f"NoGdsFileCopier.__init__: failed to open, file={metadata.src}"
            )
        self.device = device
        self.reqs: List[int] = []
        self.byte_ranges: Optional[List[Tuple[int, int]]] = None
        self._chunk_names: Optional[Set[str]] = None
        self._chunk_allocation_size: Optional[int] = None
        self._base_off = metadata.header_length

    def set_byte_ranges(self, byte_ranges: Optional[List[Tuple[int, int]]]) -> None:
        """Restrict reads to these ``[start, end)`` absolute file-offset runs.

        Bytes outside the given runs are not read; their regions of the device
        buffer are left uninitialized, so the corresponding tensors must not be
        requested. ``None`` (the default) reads the whole data section. Build
        runs with ``SafeTensorsMetadata.select_byte_ranges``.
        """
        self.byte_ranges = validated_byte_ranges(self.metadata, byte_ranges)

    def set_chunk(
        self,
        byte_ranges: List[Tuple[int, int]],
        names: Set[str],
        allocation_size: Optional[int] = None,
    ) -> None:
        """Load only ``names`` into a compact or fixed-size device buffer.

        Unlike ``set_byte_ranges`` (which still allocates the whole data section
        and leaves it sparsely filled), this allocates only
        ``max_end - min_start`` of the runs by default, or ``allocation_size``
        when provided, and instantiates just ``names``. This is the building
        block for bounded sub-file batching and fixed allocation buckets.
        """
        self.byte_ranges = validated_byte_ranges(self.metadata, byte_ranges)
        self._chunk_names = names
        self._chunk_allocation_size = validated_chunk_allocation_size(
            self.byte_ranges, allocation_size
        )

    @classmethod
    def chunk_transient_multiplier(cls, paths: List[str]) -> int:
        """Per in-flight-chunk transient cost, as a multiple of chunk span: 1.

        Reads land in the reader's fixed pool of host bounce buffers
        (``bbuf_size_kb`` x ``max_threads``, sized independently of the chunk),
        so the only device-side allocation that scales with a chunk is the
        chunk buffer itself.
        """
        return 1

    def submit_io(
        self, use_buf_register: bool, max_copy_block_size: int
    ) -> fstcpp.gds_device_buffer:
        debug_log = os.getenv("FASTSAFETENSORS_DEBUG", "false").lower() == "true"
        header_length = self.metadata.header_length
        # Default to a single run spanning the whole data section, which
        # reproduces the original full-file read.
        runs = self.byte_ranges
        if runs is None:
            runs = [(header_length, self.metadata.size_bytes)]
        span = max(end for _, end in runs) - min(start for start, _ in runs)
        parallel_block_size = max(
            _MIN_COPY_BLOCK_SIZE,
            (span + self.max_threads - 1) // self.max_threads,
        )
        max_copy_block_size = min(max_copy_block_size, parallel_block_size)
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
        gbuf = self.framework.alloc_tensor_memory(alloc_length, self.device)
        submit_begin = time.perf_counter_ns()
        submitted_bytes = 0
        submitted_requests = 0
        for start, end in runs:
            count = start
            while count < end:
                length = min(end - count, max_copy_block_size)
                req = self.reader.submit_read(
                    self.fd, gbuf, count, length, count - base_off
                )
                if req < 0:
                    raise Exception(f"submit_io: submit_nogds_read failed, err={req}")
                self.reqs.append(req)
                submitted_bytes += length
                submitted_requests += 1
                count += length
        if debug_log:
            submit_us = (time.perf_counter_ns() - submit_begin) // 1000
            print(
                "[DEBUG] NoGdsFileCopier.submit_io: "
                f"submitted_bytes={submitted_bytes}, "
                f"requests={submitted_requests}, "
                f"max_copy_block_size={max_copy_block_size}, "
                f"elapsed={submit_us} us",
                flush=True,
            )
        return gbuf

    def wait_io(
        self,
        gbuf: fstcpp.gds_device_buffer,
        dtype: DType = DType.AUTO,
        noalign: bool = False,
    ) -> Dict[str, TensorBase]:
        # Drain every request before closing the fd so no in-flight read can
        # observe a closed descriptor, then report failures.
        debug_log = os.getenv("FASTSAFETENSORS_DEBUG", "false").lower() == "true"
        wait_begin = time.perf_counter_ns()
        failed = []
        for req in self.reqs:
            count = self.reader.wait_read(req)
            if count < 0:
                failed.append(req)
        if debug_log:
            wait_us = (time.perf_counter_ns() - wait_begin) // 1000
            print(
                "[DEBUG] NoGdsFileCopier.wait_io: "
                f"requests={len(self.reqs)}, failed={len(failed)}, "
                f"elapsed={wait_us} us",
                flush=True,
            )
        if self.fd > 0:
            os.close(self.fd)
            self.fd = 0
        if len(failed) > 0:
            raise Exception(f"wait_io: wait_nogds_read failed, reqs={failed}")
        tensors_begin = time.perf_counter_ns()
        tensors = self.metadata._get_tensors(
            gbuf, self.device, self._base_off, dtype=dtype, names=self._chunk_names
        )
        if debug_log:
            tensors_us = (time.perf_counter_ns() - tensors_begin) // 1000
            print(
                "[DEBUG] NoGdsFileCopier.wait_io: "
                f"registered_tensors={len(tensors)}, elapsed={tensors_us} us",
                flush=True,
            )
        return tensors


_loaded_library = False


def load_library_func(framework=None):
    global _loaded_library
    if _loaded_library:
        return

    lib = resolve_runtime_lib_name(framework)
    fstcpp.load_library_functions(lib)
    if lib and not is_gpu_found():
        # The framework hinted a specific vendor's runtime but loading it found
        # no GPU. A GPU-built framework only reports a vendor when it sees a
        # device, so this is a real mismatch (wrong/missing runtime for that
        # vendor).
        raise Exception(
            f"[FAIL] framework hinted GPU runtime '{lib}' but no GPU was found "
            "after loading it (runtime/devices for that vendor not present)"
        )
    _loaded_library = True


@register_copier_constructor("nogds", NoGdsFileCopier)
def new_nogds_file_copier(
    device: Device,
    bbuf_size_kb: int = 16 * 1024,
    max_threads: int = 16,
    set_numa: bool = True,
    **kwargs,
) -> CopierConstructFunc:
    load_library_func(kwargs.get("framework"))
    max_threads = _resolve_max_threads(max_threads)
    device_is_not_cpu = device.type != DeviceType.CPU
    if device_is_not_cpu and not is_gpu_found():
        raise Exception(
            "[FAIL] GPU runtime library not found (expected libcudart.so, libamdhip64.so, or cudart64_XX.dll)"
        )

    device_id = device.index if device.index is not None else 0
    numa_node = (
        get_device_numa_node(device_id) if set_numa and device_is_not_cpu else None
    )
    nogds_reader = fstcpp.nogds_file_reader(
        False,
        bbuf_size_kb,
        max_threads,
        device_is_not_cpu,
        device_id,
        _use_async_copy(),
        numa_node if numa_node is not None else -1,
    )

    def construct_nogds_copier(
        metadata: SafeTensorsMetadata,
        device: Device,
        framework: FrameworkOpBase,
    ) -> CopierInterface:
        return NoGdsFileCopier(metadata, device, nogds_reader, framework, max_threads)

    return construct_nogds_copier
