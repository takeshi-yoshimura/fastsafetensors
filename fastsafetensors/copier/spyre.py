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
Spyre operations load the optional dependency when needed. ParallelLoader owns
checkpoint scheduling; each copier prepares and completes only its own file or
chunk. The host reader bounds scratch and torch-spyre provides DMA and
IOMMU primitives, including synchronization before scratch can be reused.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Mapping

from ..common import SafeTensorsMetadata, strict_json_loads
from ..frameworks import TensorBase, get_framework_op
from ..native import (
    MAX_TENSOR_BYTES,
    NativeFormat,
    NativeFormatError,
    NativeTensorSpec,
    native_metadata,
    native_specs,
)
from ..saver import WriteEntry, tensor_fill
from ..st_types import Device, DeviceType, DType
from .base import (
    CopierInterface,
    DummyDeviceBuffer,
    validated_byte_ranges,
    validated_chunk_allocation_size,
)
from .host import TRANSFER_LOCK, HostFileReader, HostReadRequest, pread_into
from .registry import CopierConstructFunc, register_copier_constructor

DEVICE_NAME = "spyre"
COPIER_TYPE = "spyre"
FORMAT = "spyre.native.v1"
MAX_TRANSFER_BYTES = MAX_TENSOR_BYTES
NATIVE_FORMAT = NativeFormat(device=DEVICE_NAME, format=FORMAT)
__all__ = [
    "COPIER_TYPE",
    "FORMAT",
    "NATIVE_FORMAT",
    "SpyreCopier",
    "MAX_TRANSFER_BYTES",
    "native_entry",
    "native_fill",
    "native_shard_metadata",
    "split_native_entries",
]

# Plain payloads that must be restored to Spyre, with an optional device layout.
KEY_MOVED = "torch_spyre.moved"


def _tensor_ops():
    return get_framework_op("pytorch").get_native_ops(Device(DeviceType.SPYRE))


@dataclass(frozen=True)
class _NativePayload:
    src: TensorBase
    dst: TensorBase
    spec: NativeTensorSpec


def native_entry(
    name: str,
    src: Any,
    dst: Any,
    *,
    nbytes: int,
    layout: Any,
) -> WriteEntry:
    """Describe a host tensor encoded for an allocated Spyre destination.

    ``nbytes`` and ``layout`` describe dst's physical storage, as reported by
    torch-spyre. No conversion or device transfer happens until native_fill.
    """
    if nbytes <= 0:
        raise ValueError(f"{name}: native payload size must be positive")
    ops = _tensor_ops()
    src, dst = ops.as_tensor(src), ops.as_tensor(dst)
    spec = ops.tensor_spec(name, dst, nbytes, layout)
    return WriteEntry(name, DType.U8, [nbytes], nbytes, _NativePayload(src, dst, spec))


def native_shard_metadata(
    entries: Sequence[WriteEntry], *, tp_size=1, tp_rank=0
) -> dict[str, str]:
    """ParallelSaver callback describing this shard's Spyre-native payloads."""
    native = [e for e in entries if isinstance(e.source, _NativePayload)]
    if not native:
        return {}
    if len(native) != len(entries):
        raise NativeFormatError("native and portable entries require separate files")
    return native_metadata(
        NATIVE_FORMAT,
        {e.name: e.source.spec for e in native},
        tp_size=tp_size,
        tp_rank=tp_rank,
    )


def split_native_entries(entries):
    """Separate files are required by the all-native profile contract."""
    return (
        [e for e in entries if isinstance(e.source, _NativePayload)],
        [e for e in entries if not isinstance(e.source, _NativePayload)],
    )


def native_fill(work: list[tuple[memoryview, WriteEntry]]) -> None:
    """ParallelSaver callback encoding native entries and copying plain tensors.

    Conversion writes directly into the saver-owned buffers; allocated Spyre
    destinations describe the layout but are not filled by this operation.
    """
    plain: list[tuple[memoryview, WriteEntry]] = []
    native: list[tuple[memoryview, WriteEntry]] = []
    for buf, entry in work:
        (native if isinstance(entry.source, _NativePayload) else plain).append(
            (buf, entry)
        )
    if plain:
        tensor_fill()(plain)
    if native:
        _tensor_ops().encode_tensors(
            [(buf, entry.source.src, entry.source.dst) for buf, entry in native]
        )


class SpyreCopier(CopierInterface):
    device_native = True
    owns_tensor_storage = True

    def __init__(
        self,
        metadata: SafeTensorsMetadata,
        device: Device,
        framework: Any,
        reader: HostFileReader,
        destinations: Mapping[str, Any],
        tp_size: int = 1,
        tp_rank: int = 0,
    ) -> None:
        self.metadata = metadata
        self.device = device
        self.framework = framework
        self.ops = framework.get_native_ops(device)
        self.reader = reader
        self.request: HostReadRequest | None = None
        self.destinations: dict[str, TensorBase] = {
            name: self.ops.as_tensor(tensor) for name, tensor in destinations.items()
        }
        self.names: list[str] | None = None
        self.out: dict[str, TensorBase] = {}
        self.moved: dict[str, Any] = {}
        self.fd: int | None = None
        self.tp_size, self.tp_rank = tp_size, tp_rank

    def select_tensors(self, keep: Callable[[str], bool]) -> None:
        self.names = [n for n in self.metadata.tensors if keep(n)]

    def set_chunk(self, byte_ranges, names, allocation_size=None) -> None:
        """Accept ParallelLoader's whole-tensor chunk, never slice native bytes.

        File-span allocation padding is unnecessary with individual tensors.
        A supplied allocation_size is validated but does not pad destinations.
        """
        ranges = validated_byte_ranges(self.metadata, byte_ranges)
        if ranges is None:
            raise ValueError("chunk byte_ranges must be explicit")
        validated_chunk_allocation_size(ranges, allocation_size)
        unknown = names - self.metadata.tensors.keys()
        if unknown:
            raise ValueError(f"unknown chunk tensor names: {sorted(unknown)}")
        for name in names:
            frame = self.metadata.tensors[name]
            start, end = (
                self.metadata.header_length + offset for offset in frame.data_offsets
            )
            covered = start
            for lo, hi in ranges:
                if lo <= covered:
                    covered = max(covered, hi)
            if covered < end:
                raise ValueError(f"{name}: chunk must cover the whole tensor")
        self.select_tensors(names.__contains__)

    @classmethod
    def chunk_transient_multiplier(cls, paths) -> int:
        raise NotImplementedError(
            "Spyre device_memory_budget is unsupported: payload byte sizes do not "
            "account for logical-storage padding and moved tensor layouts; "
            "use max_batch_bytes for whole-tensor chunking instead"
        )

    def _validate_plan(self):
        specs = native_specs(self.metadata, tp_size=self.tp_size, tp_rank=self.tp_rank)
        names = self.names if self.names is not None else list(self.metadata.tensors)
        device = self.ops.device(self.device.as_str())
        md = self.metadata.metadata
        layouts = {}
        try:
            moved = (
                strict_json_loads(md.get(KEY_MOVED, "{}"))
                if isinstance(md, dict)
                else {}
            )
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
                layouts[name] = self.ops.layout_from_json(layout)
        except (KeyError, TypeError, ValueError) as e:
            raise NativeFormatError(f"{self.metadata.src}: bad {KEY_MOVED}: {e}") from e
        for name in names:
            dst = self.destinations.get(name)
            if dst is None:
                continue
            spec = specs.get(name)
            if spec is not None:
                self.ops.validate_destination(
                    name,
                    dst,
                    spec.shape,
                    spec.dtype,
                    device,
                    stride=spec.stride,
                    layout=spec.layout,
                )
                if self.ops.tensor_nbytes(dst) != spec.nbytes:
                    raise NativeFormatError(f"{name}: device size differs from payload")
            elif name in layouts:
                frame = self.metadata.tensors[name]
                self.ops.validate_plain_destination(
                    name, dst, frame.shape, frame.dtype, device, layout=moved[name]
                )
        return specs, names, device, layouts

    def validate_io(self) -> None:
        self._validate_plan()

    def submit_io(self, use_buf_register: bool, max_copy_block_size: int):
        if max_copy_block_size <= 0:
            raise ValueError("max_copy_block_size must be positive")
        specs, names, device, self.moved = self._validate_plan()
        fd = os.open(self.metadata.src, os.O_RDONLY)
        self.fd = fd
        transfers = []
        # Everything is validated and allocated here, before any data moves.
        try:
            for name in names:
                frame = self.metadata.tensors[name]
                begin = self.metadata.header_length + frame.data_offsets[0]
                nbytes = frame.data_offsets[1] - frame.data_offsets[0]
                spec = specs.get(name)
                if spec is None:
                    host = self.framework.get_empty_tensor(
                        frame.shape, frame.dtype, Device(DeviceType.CPU)
                    )
                    if nbytes:
                        view = self.ops.host_buffer_view(host)
                        for offset in range(0, nbytes, max_copy_block_size):
                            pread_into(
                                fd,
                                view[offset : offset + max_copy_block_size],
                                begin + offset,
                            )
                    self.out[name] = host
                    continue
                dst = self.destinations.get(name)
                if dst is None:
                    dst = self.ops.empty_tensor(spec, device)
                transfers.append(((fd, begin), dst))
                self.out[name] = self.ops.wrap_tensor(dst, self.device, spec.dtype)
            self.request = self.reader.submit_read(
                transfers, device, max_copy_block_size
            )
        except BaseException:
            try:
                self.finish_io()
            except BaseException:
                pass  # Preserve submission failure after draining/closing.
            raise
        return DummyDeviceBuffer()

    def finish_io(self) -> None:
        # Cancel only this file; drain running reads/copies before fd release.
        try:
            if self.request is not None:
                request, self.request = self.request, None
                self.reader.cancel_read(request)
        finally:
            if self.fd is not None:
                fd, self.fd = self.fd, None
                os.close(fd)
            self.out.clear()
            self.moved.clear()

    def wait_io(
        self, gbuf, dtype: DType = DType.AUTO, noalign: bool = False, owner=None
    ):
        device = self.ops.device(self.device.as_str())
        try:
            if self.request is not None:
                self.reader.wait_read(self.request)
                self.request = None
            return self._materialize_moved(device)
        finally:
            self.finish_io()

    def _materialize_moved(self, device):
        with TRANSFER_LOCK:
            for name, layout in self.moved.items():
                source = self.out[name]
                dst = self.destinations.get(name)
                dst = self.ops.move_tensor(source, device, layout, dst)
                self.out[name] = self.ops.wrap_tensor(dst, self.device, source.dtype)
        self.moved.clear()
        out, self.out = self.out, {}
        return out


def load_library_spyre(device: Device, **kwargs) -> None:
    if device.type != DeviceType.SPYRE:
        raise ValueError("spyre copier requires device='spyre'")
    framework = kwargs.get("framework")
    if framework is not None and framework.get_name() != "pytorch":
        raise ValueError("spyre copier requires the pytorch framework")
    try:
        (
            framework.get_native_ops(device) if framework is not None else _tensor_ops()
        ).load_library()
    except ModuleNotFoundError as e:
        if e.name != "torch_spyre":
            raise
        raise ImportError(
            "spyre copier requires torch-spyre; install torch-spyre "
            "to load onto Spyre"
        ) from e


@register_copier_constructor(
    COPIER_TYPE,
    SpyreCopier,
    load_library=load_library_spyre,
)
def new_spyre_copier(
    device: Device,
    destinations: Mapping[str, Any] | None = None,
    bbuf_size_kb: int | None = None,
    max_threads: int = 16,
    tp_size: int = 1,
    tp_rank: int = 0,
    **kwargs: Any,
) -> CopierConstructFunc:
    """``destinations`` (name -> Spyre tensor) are filled in place of new tensors."""
    capacity = (16 * 1024 if bbuf_size_kb is None else bbuf_size_kb) * 1024
    if capacity <= 0 or max_threads <= 0:
        raise ValueError("host capacity and thread count must be positive")
    reader: HostFileReader | None = None

    def construct(metadata, device, framework):
        nonlocal reader
        if reader is None:
            reader = HostFileReader(
                framework.get_native_ops(device), max_threads, max_bytes=capacity
            )
        return SpyreCopier(
            metadata, device, framework, reader, destinations or {}, tp_size, tp_rank
        )

    def validate_metadata(metas):
        total = 0
        for meta in metas:
            specs = native_specs(meta, tp_size=tp_size, tp_rank=tp_rank)
            total += sum(spec.nbytes for spec in specs.values())
        from ..native import MAX_TOTAL_BYTES

        if total > MAX_TOTAL_BYTES:
            raise NativeFormatError("native load plan exceeds aggregate byte budget")

    setattr(construct, "validate_metadata", validate_metadata)

    return construct
