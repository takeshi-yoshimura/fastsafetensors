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
from dataclasses import dataclass, fields, is_dataclass
from types import SimpleNamespace
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
    "TransferRecorder",
    "TransferRecord",
    "TransferRestorer",
    "is_dense_tensor",
    "tensor_metadata",
    "read_recorded_tensor",
    "check_materialized",
    "KEY_TRANSFERS",
    "KEY_HOST",
    "CACHE_FORMAT",
    "KEY_CACHE_FORMAT",
    "KEY_CACHE_KEY",
    "KEY_FP8",
    "cache_metadata",
    "read_cache_metadata",
    "native_entry",
    "native_fill",
    "native_shard_metadata",
]
# Plain payloads that must be restored to Spyre, with an optional device layout.
KEY_MOVED = "torch_spyre.moved"
KEY_TRANSFERS = "torch_spyre.transfers"
KEY_HOST = "torch_spyre.host"
CACHE_FORMAT = "torch_spyre-cache/3"
KEY_CACHE_FORMAT = "torch_spyre.cache.format"
KEY_CACHE_KEY = "torch_spyre.cache.key"
KEY_FP8 = "torch_spyre.cache.fp8"


def cache_metadata(key: Mapping[str, Any]) -> tuple[dict[str, str], dict[str, str]]:
    """Encode opaque caller identity in the backend's cache envelope.

    Returns metadata for every shard and metadata for shard zero, respectively.
    The caller computes identity; this backend owns its on-disk keys
    and format. FP8 payload recipes are reserved and currently unsupported.
    """
    return (
        {
            KEY_CACHE_FORMAT: CACHE_FORMAT,
            KEY_CACHE_KEY: json.dumps(dict(key), sort_keys=True),
        },
        {KEY_FP8: "[]"},
    )


def read_cache_metadata(metadata: Mapping[str, str], key: Mapping[str, Any]) -> None:
    """Validate cache format, caller identity and the absence of FP8 recipes."""
    try:
        if metadata[KEY_CACHE_FORMAT] != CACHE_FORMAT or metadata[
            KEY_CACHE_KEY
        ] != json.dumps(dict(key), sort_keys=True):
            raise NativeFormatError("cache is for another identity or format")
        if json.loads(metadata[KEY_FP8]) != []:
            raise NativeFormatError("FP8 cache payload recipes are not supported")
    except (KeyError, TypeError, ValueError) as e:
        raise NativeFormatError(f"invalid Spyre cache metadata: {e}") from e


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


def is_dense_tensor(t: torch.Tensor) -> bool:
    """Whether a tensor is non-overlapping and dense, including transposes."""
    expected = 1
    for stride, size in sorted((s, n) for s, n in zip(t.stride(), t.shape) if n != 1):
        if stride != expected:
            return False
        expected *= size
    return True


def _unstageable_reason(t: torch.Tensor) -> str | None:
    import torch

    if t.device.type not in ("cpu", "meta"):
        return f"on {t.device.type}"
    if t.dtype != torch.float16:
        return str(t.dtype)
    if t.numel() == 0 or t.requires_grad or not is_dense_tensor(t):
        return "empty/grad/non-dense"
    return None


def tensor_metadata(t: torch.Tensor) -> dict:
    """Logical tensor description used by transfer and host-state records."""
    from ..frameworks._torch import fst_dtype

    try:
        fst_dtype(t.dtype)
    except KeyError:
        raise NativeFormatError(
            f"dtype {t.dtype} has no safetensors equivalent"
        ) from None
    return {
        "size": list(t.shape),
        "stride": list(t.stride()),
        "dtype": str(t.dtype).removeprefix("torch."),
    }


def _layout_json(layout) -> dict | None:
    if layout is None:
        return None
    from torch_spyre.layout import layout_to_json

    return layout_to_json(layout)


def read_recorded_tensor(
    headers: Mapping[str, SafeTensorsMetadata], record: Mapping[str, Any]
) -> torch.Tensor:
    """Read an ordinary tensor and recover recorded dense strides, if needed."""
    import torch
    from safetensors import safe_open

    name = record["name"]
    if name in native_specs(headers[name]):
        raise NativeFormatError(
            f"{name}: expected an ordinary tensor, got a native payload"
        )
    with safe_open(headers[name].src, framework="pt", device="cpu") as checkpoint:
        tensor = checkpoint.get_tensor(name)
    if (
        list(tensor.shape) != record["size"]
        or str(tensor.dtype).removeprefix("torch.") != record["dtype"]
    ):
        raise NativeFormatError(f"{name}: stored tensor does not match its record")
    if record["dense"] and list(tensor.stride()) != record["stride"]:
        out = torch.empty_strided(record["size"], record["stride"], dtype=tensor.dtype)
        out.copy_(tensor)
        return out
    return tensor


def _tensor_paths(model):
    """Tensor references in module state, without inspecting arbitrary objects."""
    import torch

    def walk(value, path, ancestors):
        if isinstance(value, torch.Tensor):
            yield path, value
            return
        if id(value) in ancestors or isinstance(value, torch.nn.Module):
            return
        ancestors = ancestors | {id(value)}
        if isinstance(value, dict):
            items = (
                (["key", k], v)
                for k, v in value.items()
                if k is None or type(k) in (str, int, bool)
            )
        elif isinstance(value, (list, tuple)):
            items = ((["index", i], v) for i, v in enumerate(value))
        elif is_dataclass(value) and not isinstance(value, type):
            items = ((["attr", f.name], getattr(value, f.name)) for f in fields(value))
        elif isinstance(value, SimpleNamespace):
            items = ((["attr", k], v) for k, v in vars(value).items())
        else:
            return
        for step, item in items:
            yield from walk(item, path + [step], ancestors)

    for prefix, module in model.named_modules():
        for name, value in vars(module).items():
            if name != "_modules":
                for path, tensor in walk(value, [["attr", name]], frozenset()):
                    yield prefix, path, tensor


def _resolve_tensor(model, module, path):
    import torch

    if not isinstance(module, str) or not isinstance(path, list) or not path:
        raise NativeFormatError("invalid host tensor path")
    value = model
    try:
        for name in module.split(".") if module else ():
            value = value._modules[name]
        for kind, key in path:
            if kind == "attr" and isinstance(key, str):
                if (
                    is_dataclass(value)
                    and not isinstance(value, type)
                    and key in {f.name for f in fields(value)}
                ):
                    value = getattr(value, key)
                else:
                    value = vars(value)[key]
            elif (
                kind == "key"
                and isinstance(value, dict)
                and (key is None or type(key) in (str, int, bool))
            ):
                value = value[key]
            elif (
                kind == "index"
                and isinstance(value, (list, tuple))
                and type(key) is int
                and 0 <= key < len(value)
            ):
                value = value[key]
            else:
                raise NativeFormatError("invalid host tensor path step")
    except (KeyError, TypeError, ValueError, AttributeError) as e:
        raise NativeFormatError(
            f"host tensor path {module!r} {path!r} is missing or invalid"
        ) from e
    if not isinstance(value, torch.Tensor):
        raise NativeFormatError(f"host path {module!r} {path!r} is not a tensor")
    return value


def check_materialized(model) -> None:
    """Reject remaining meta tensors in the supported module-state containers."""
    for module, path, tensor in _tensor_paths(model):
        if tensor.is_meta:
            raise NativeFormatError(f"tensor {module!r} {path!r} is still on meta")


def _swap_in(target, data) -> None:
    import torch

    replacement = torch.Tensor._make_subclass(type(target), data, target.requires_grad)
    replacement.__dict__.update(target.__dict__)
    torch.utils.swap_tensors(target, replacement)


@dataclass
class TransferRecord:
    """One tensor transfer: already present, deferred, or moved immediately."""

    kind: str  # "same", "defer", or "to"
    src: torch.Tensor
    dst: torch.Tensor
    layout: Any = None


class TransferRecorder:
    """Allocate tensor destinations now and fill eligible fp16 tensors in bulk.

    Callable as a tensor conversion function, with ``place_with_layout`` for
    explicit Spyre placement. Deferred destinations contain no data until
    ``fill`` completes; callers must not read them during traversal. The caller
    owns model traversal and model-level checkpoint policy.
    """

    def __init__(self, device: torch.device) -> None:
        self.device = device
        self.records: list[TransferRecord] = []
        self._host_paths: list[tuple[str, list]] | None = None

    def capture_host_state(self, model) -> None:
        """Remember tensor paths before traversal; derived transfer caches are not saved."""
        check_materialized(model)
        self._host_paths = [
            (module, path)
            for module, path, tensor in _tensor_paths(model)
            if tensor.device.type == "cpu"
        ]

    def checkpoint_metadata(self, model) -> tuple[dict[str, str], list[WriteEntry]]:
        """Encode transfers and surviving CPU state using backend-owned metadata keys."""
        from ..saver import plain_entry

        if self._host_paths is None:
            raise NativeFormatError("capture_host_state must precede model traversal")
        check_materialized(model)
        records, entries = self.checkpoint_entries()
        host, names = [], {}
        for module, path in self._host_paths:
            try:
                tensor = _resolve_tensor(model, module, path)
            except NativeFormatError:
                # _apply may remove or replace a transfer source.
                continue
            if tensor.device.type != "cpu":
                continue
            name = names.get(id(tensor))
            if name is None:
                name = f"h{len(names)}"
                names[id(tensor)] = name
                entries.append(plain_entry(name, tensor.detach()))
            host.append(
                {
                    "module": module,
                    "path": path,
                    "device": "cpu",
                    "name": name,
                    **tensor_metadata(tensor),
                    "dense": is_dense_tensor(tensor),
                }
            )
        return {KEY_TRANSFERS: json.dumps(records), KEY_HOST: json.dumps(host)}, entries

    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        import torch

        if t.device.type == self.device.type:
            dst = t.to(self.device)
            kind = "same"
        elif _unstageable_reason(t) is not None:
            dst = t.to(self.device)
            kind = "to"
        else:
            dst = torch.empty_strided(
                t.size(), t.stride(), dtype=t.dtype, device=self.device
            )
            kind = "defer"
        # _apply may rebind a Parameter's .data; retain its original storage.
        self.records.append(TransferRecord(kind, t.detach(), dst))
        return dst

    def place_with_layout(self, src: torch.Tensor, layout) -> torch.Tensor:
        """Defer ``src.to(device, dtype=src.dtype, device_layout=layout)``."""
        from torch_spyre._C import spyre_empty_with_layout

        if src.device.type == self.device.type:
            dst = src.to(self.device, dtype=src.dtype, device_layout=layout)
            kind = "same"
        elif _unstageable_reason(src) is not None:
            dst = src.to(self.device, dtype=src.dtype, device_layout=layout)
            kind = "to"
        else:
            dst = spyre_empty_with_layout(
                src.size(), src.stride(), src.dtype, layout, device=self.device
            )
            kind = "defer"
        self.records.append(TransferRecord(kind, src.detach(), dst, layout))
        return dst

    def fill(self) -> TransferPipeline:
        """Complete deferred transfers, propagating pipeline failures."""
        pipeline = stage_to_spyre(
            [(r.src, r.dst) for r in self.records if r.kind == "defer"], self.device
        )
        for r in self.records:
            if r.kind == "to":
                pipeline.count(".to()", r.src.nbytes)
        return pipeline

    def checkpoint_entries(
        self, *, prefix: str = "t"
    ) -> tuple[list[dict], list[WriteEntry]]:
        """Describe recorded transfers for saving, without filling destinations.

        Returns JSON-serializable replay records and saver entries. Persist the
        records with KEY_TRANSFERS and use native_fill/native_shard_metadata
        to write the entries. Deferred destinations still contain no data.
        """
        from ..saver import plain_entry

        records, entries = [], []
        for seq, transfer in enumerate(self.records):
            record = {
                "kind": transfer.kind,
                **tensor_metadata(transfer.src),
                "placed": _layout_json(transfer.layout),
            }
            nbytes = 0
            if transfer.kind == "defer":
                from torch_spyre._C import dma_bytes, get_spyre_tensor_layout

                nbytes = dma_bytes(transfer.dst)
            name = f"{prefix}{seq}"
            if transfer.kind == "defer" and 0 < nbytes <= MAX_TRANSFER_BYTES:
                layout = _layout_json(get_spyre_tensor_layout(transfer.dst))
                record.update(kind="raw", name=name, layout=layout)
                entries.append(
                    native_entry(
                        name, transfer.src, transfer.dst, nbytes=nbytes, layout=layout
                    )
                )
            elif transfer.kind != "same":
                record.update(kind="to", name=name, dense=is_dense_tensor(transfer.src))
                entries.append(plain_entry(name, transfer.src))
            records.append(record)
        return records, entries


class TransferRestorer:
    """Replay tensor transfers in call order, independently of model traversal.

    Raw destinations are allocated but not filled. Call finish after traversal
    to check record exhaustion, then fill destinations with ParallelLoader.
    Any mismatch raises NativeFormatError; discard partially restored state.
    """

    def __init__(
        self,
        device: torch.device,
        headers: Mapping[str, SafeTensorsMetadata],
        records: Sequence[dict],
    ) -> None:
        self.device = device
        self.headers = headers
        self.records = records
        self.seq = 0
        self.destinations: dict[str, torch.Tensor] = {}
        self.host: list[dict] = []

    @classmethod
    def from_metadata(cls, device, headers, metadata):
        """Read the backend's transfer and CPU-state recipes, without inference imports."""
        try:
            records, host = json.loads(metadata[KEY_TRANSFERS]), json.loads(
                metadata[KEY_HOST]
            )
            if not isinstance(records, list) or not isinstance(host, list):
                raise ValueError("transfer and host recipes must be lists")
            if not all(isinstance(record, dict) for record in records + host):
                raise ValueError("transfer and host records must be objects")
        except (KeyError, TypeError, ValueError) as e:
            raise NativeFormatError(f"invalid Spyre replay metadata: {e}") from e
        restorer = cls(device, headers, records)
        restorer.host = host
        return restorer

    def restore_host_state(self, model) -> None:
        """Resolve recorded CPU paths directly and validate all aliases before swapping."""
        targets, aliases, names, paths = [], {}, {}, set()
        try:
            for record in self.host:
                path = json.dumps([record["module"], record["path"]])
                if path in paths or record["device"] != "cpu":
                    raise NativeFormatError(
                        "duplicate host path or invalid host device"
                    )
                paths.add(path)
                target = _resolve_tensor(model, record["module"], record["path"])
                if (
                    target.device.type not in ("cpu", "meta")
                    or list(target.shape) != record["size"]
                    or tensor_metadata(target)["dtype"] != record["dtype"]
                ):
                    raise NativeFormatError(
                        f"host tensor {path} differs from its record"
                    )
                name, identity = record["name"], id(target)
                if (
                    aliases.setdefault(name, identity) != identity
                    or names.setdefault(identity, name) != name
                ):
                    raise NativeFormatError(f"host tensor aliases differ at {path}")
                targets.append((target, record))
        except (KeyError, TypeError, ValueError) as e:
            raise NativeFormatError(f"invalid host tensor recipe: {e}") from e
        restored = set()
        for target, record in targets:
            if id(target) not in restored:
                _swap_in(target, read_recorded_tensor(self.headers, record))
                restored.add(id(target))

    def _next(self, tensor: torch.Tensor, placed) -> dict:
        if self.seq >= len(self.records):
            raise NativeFormatError("more tensors to move than recorded")
        record = self.records[self.seq]
        if (
            list(tensor.shape) != record["size"]
            or str(tensor.dtype).removeprefix("torch.") != record["dtype"]
            or _layout_json(placed) != record["placed"]
        ):
            raise NativeFormatError(
                f"move #{self.seq}: tensor or placed layout does not match its record"
            )
        if record["kind"] not in ("same", "to", "raw"):
            raise NativeFormatError(
                f"move #{self.seq}: unknown transfer kind {record['kind']!r}"
            )
        self.seq += 1
        return record

    def _restore(self, tensor: torch.Tensor, placed=None) -> torch.Tensor:
        record = self._next(tensor, placed)
        if record["kind"] == "same":
            if tensor.device.type != self.device.type:
                raise NativeFormatError(
                    f"move #{self.seq - 1} was a revisit, got {tensor.device}"
                )
            return self._move(tensor, placed)
        if record["kind"] == "to":
            return self._move(read_recorded_tensor(self.headers, record), placed)
        if list(tensor.stride()) != record["stride"]:
            raise NativeFormatError(
                f"move #{self.seq - 1}: strides {list(tensor.stride())} != {record['stride']}"
            )
        from torch_spyre._C import dma_bytes, spyre_empty_with_layout
        from torch_spyre.layout import layout_from_json

        from ..frameworks._torch import dtype_convert

        name = record["name"]
        metadata = self.headers[name]
        if native_format(metadata) != NATIVE_FORMAT:
            raise NativeFormatError(f"{name}: incompatible native format")
        spec = native_specs(metadata)[name]
        dtype = dtype_convert[spec.dtype]
        if (
            spec.shape != record["size"]
            or spec.stride != record["stride"]
            or str(dtype).removeprefix("torch.") != record["dtype"]
            or spec.layout != record["layout"]
        ):
            raise NativeFormatError(f"{name}: native payload does not match its record")
        dst = spyre_empty_with_layout(
            spec.shape,
            spec.stride,
            dtype,
            layout_from_json(spec.layout),
            device=self.device,
        )
        if dma_bytes(dst) != spec.nbytes:
            raise NativeFormatError(
                f"move #{self.seq - 1}: device size differs from payload"
            )
        self.destinations[name] = dst
        return dst

    def _move(self, source: torch.Tensor, placed) -> torch.Tensor:
        if placed is None:
            return source.to(self.device)
        return source.to(self.device, dtype=source.dtype, device_layout=placed)  # type: ignore[call-overload]

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        return self._restore(tensor)

    def place_with_layout(self, source: torch.Tensor, layout) -> torch.Tensor:
        return self._restore(source, layout)

    def finish(self) -> None:
        """Check that traversal consumed all records; does not start any DMA."""
        if self.seq != len(self.records):
            raise NativeFormatError(
                f"moved {self.seq} tensors, recorded {len(self.records)}"
            )


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
