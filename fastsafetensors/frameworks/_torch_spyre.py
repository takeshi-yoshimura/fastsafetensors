# SPDX-License-Identifier: Apache-2.0
"""PyTorch/torch-spyre tensor operations, without load scheduling or cache policy.

Importing this adapter does not import torch-spyre or initialize its runtime.
Spyre storage is tensor-owned: its data_ptr is not a raw DMA destination.
"""

from __future__ import annotations

import mmap
from collections.abc import Mapping
from functools import lru_cache
from typing import Any, Callable

from ..native import (
    NativeFormatError,
    NativeTensorSpec,
    canonical_layout,
    dense_stride,
    validate_native_spec,
)
from ..st_types import Device, DeviceType, DType
from . import HostTransfer, TensorBase, TransferDrainedError
from ._torch import TorchTensor, dtype_convert, fst_dtype

_UNSPECIFIED = object()
# Registered before enqueue; unknown completion deliberately never removes an
# entry. Do not confuse an active transfer with a failed one in producer checks.
_raw_transfers: list[_RawTransfer] = []


@lru_cache(maxsize=1)
def backend():
    """Load only the two callable buffer APIs; never use a legacy DMA fallback."""
    try:
        from torch_spyre import _C
    except ImportError as error:
        raise ImportError(
            "Spyre-native loading requires torch-spyre buffer APIs. "
            "No legacy DMA fallback."
        ) from error
    if any(
        not callable(getattr(_C, name, None))
        for name in ("copy_buffer_ranges", "convert_buffers")
    ):
        raise RuntimeError(
            "incompatible torch-spyre buffer API; "
            "require callable copy_buffer_ranges and convert_buffers"
        )
    return _C


def layout_to_json(layout) -> dict:
    return {
        "device_size": list(layout.device_size),
        "stride_map": list(layout.stride_map),
        "device_dtype": layout.device_dtype.name,
        "element_arrangement": layout.element_arrangement.name,
        "valid_elements": [
            [list(dims), count] for dims, count in layout.valid_elements.items()
        ],
    }


def layout_from_json(d: Mapping[str, Any]):
    from torch_spyre._C import DataFormats, ElementArrangement, SpyreTensorLayout

    return SpyreTensorLayout(
        device_size=list(d["device_size"]),
        stride_map=list(d["stride_map"]),
        device_dtype=DataFormats.__members__[d["device_dtype"]],
        valid_elements={
            tuple(dims): count for dims, count in d.get("valid_elements", [])
        },
        element_arrangement=ElementArrangement.__members__[d["element_arrangement"]],
    )


class _RawTransfer:
    """Private owning ticket; registry survives abandoned or failed tickets."""

    def __init__(self, entries, work, stream, api) -> None:
        self.lifetime: tuple[Any, Any, Any, Any] | None = (entries, work, stream, api)
        self.error: BaseException | None = None
        _raw_transfers.append(self)

    def wait(self) -> None:
        from ..copier.host import (
            TRANSFER_LOCK,
            check_transfer_state,
            poison_transfer_state,
        )

        with TRANSFER_LOCK:
            if self.error is not None:
                raise self.error
            if self.lifetime is None:
                return
            _, _, stream, api = self.lifetime
            stream_value = getattr(stream, "_cdata", stream)
            try:
                # An earlier unknown wait already poisoned the process. Retain
                # this ticket too, without retrying synchronization on that runtime.
                check_transfer_state()
                if api.stream_get_error(stream_value) != api.SpyreStreamError.Success:
                    raise RuntimeError("stream shut down")
                stream.synchronize()
                if api.stream_get_error(stream_value) != api.SpyreStreamError.Success:
                    raise RuntimeError("stream failed")
            except BaseException as error:
                poison_transfer_state()
                self.error = RuntimeError(
                    "raw H2D completion unknown; restart the process"
                )
                raise self.error from error
            _raw_transfers.remove(self)  # Ticket identity, never Tensor equality.
            self.lifetime = None


class TorchSpyreOp:
    def __init__(self, framework, device: Device) -> None:
        self.framework = framework
        self._device = device

    def load_library(self) -> None:
        import torch
        import torch_spyre

        if torch._C._get_privateuse1_backend_name() != "spyre":
            torch_spyre._autoload()
        # Fail before allocation/file staging when the required API is missing.
        backend()

    def _torch_device(self, device):
        import torch

        device = torch.device(device.as_str() if isinstance(device, Device) else device)
        if device.type == "spyre" and device.index is None:
            device = torch.device("spyre", torch.spyre.current_device())  # type: ignore[attr-defined]
        return device

    def device(self, device) -> Device:
        device = self._torch_device(device)
        return Device(DeviceType(device.type), device.index)

    def as_tensor(self, tensor: Any) -> TensorBase:
        """Wrap at the native API boundary; preserve existing storage and identity."""
        if isinstance(tensor, TensorBase):
            return tensor
        try:
            return self.framework.wrap_tensor(tensor)
        except KeyError:
            # AUTO is only a boundary placeholder for an unrepresentable dtype.
            # Metadata creation rejects it before any checkpoint entry is made.
            return TorchTensor(self.device(tensor.device), DType.AUTO, tensor)

    def unwrap(self, tensor: TensorBase) -> Any:
        return tensor.get_raw()

    def tensor_spec(self, name, tensor: TensorBase, nbytes, layout) -> NativeTensorSpec:
        raw = self.unwrap(tensor)
        spec = NativeTensorSpec(
            name,
            fst_dtype(raw.dtype),
            list(raw.shape),
            list(raw.stride()),
            layout,
            nbytes,
        )
        validate_native_spec(spec)
        return spec

    def wrap_tensor(
        self, tensor: TensorBase, device: Device, dtype: DType
    ) -> TensorBase:
        return TorchTensor(device, dtype, self.unwrap(tensor))

    def validate_device(self, name, tensor, device) -> None:
        if self.as_tensor(tensor).device != self.device(device):
            raise NativeFormatError(f"{name}: destination is on a different device")

    def empty_tensor(self, spec: NativeTensorSpec, device) -> TensorBase:
        validate_native_spec(spec)
        import torch
        from torch_spyre._C import spyre_empty_with_layout

        device = self._torch_device(device)
        if device.type == "spyre":
            # The explicit-layout pybind allocator bypasses aten's lazy init.
            torch.spyre._lazy_init()  # type: ignore[attr-defined]
        tensor = spyre_empty_with_layout(
            spec.shape,
            spec.stride,
            dtype_convert[spec.dtype],
            self.layout_from_json(spec.layout),
            device=device,
        )
        wrapped = self.as_tensor(tensor)
        self.validate_destination(
            spec.name,
            wrapped,
            spec.shape,
            spec.dtype,
            device,
            stride=spec.stride,
            layout=spec.layout,
        )
        if self.tensor_nbytes(wrapped) != spec.nbytes:
            raise NativeFormatError(
                f"{spec.name}: runtime allocation differs from fixed layout"
            )
        return wrapped

    def validate_destination(
        self,
        name,
        tensor,
        shape,
        dtype,
        device,
        *,
        stride=_UNSPECIFIED,
        layout=_UNSPECIFIED,
    ) -> None:
        tensor = self.as_tensor(tensor)
        raw = self.unwrap(tensor)
        if (
            list(raw.shape) != list(shape)
            or raw.dtype != dtype_convert[dtype]
            or (stride is not _UNSPECIFIED and list(raw.stride()) != stride)
            or (layout is not _UNSPECIFIED and self.tensor_layout(tensor) != layout)
        ):
            raise NativeFormatError(
                f"{name}: destination does not match the stored tensor"
            )
        self.validate_device(name, tensor, device)

    def validate_plain_destination(
        self, name, tensor, shape, dtype, device, *, layout=None
    ) -> None:
        self.validate_destination(
            name,
            tensor,
            self.framework.get_native_shape(dtype, shape),
            dtype,
            device,
            layout=_UNSPECIFIED if layout is None else layout,
        )

    def layout_from_json(self, layout):
        if layout is None:
            return None
        return layout_from_json(layout)

    def layout_to_json(self, layout):
        if layout is None:
            return None
        return layout_to_json(layout)

    def tensor_layout(self, tensor):
        from torch_spyre._C import get_spyre_tensor_layout

        return self.layout_to_json(
            get_spyre_tensor_layout(self.unwrap(self.as_tensor(tensor)))
        )

    def allocation_descriptor(self, tensor):
        """Snapshot geometry and a storage-owning handle, never dst.data_ptr()."""
        from torch_spyre._C import get_composite_address

        raw = self.unwrap(self.as_tensor(tensor))
        if (
            raw.device.type != "spyre"
            or raw.storage_offset() != 0
            or raw._is_view()
            or not raw.is_contiguous()
        ):
            raise ValueError("raw destination must be a contiguous base Spyre tensor")
        owner = get_composite_address(raw)
        chunks = owner.chunks()
        if owner.num_chunks != 1 or len(chunks) != 1:
            raise ValueError("raw destination must be a single-chunk allocation")
        chunk = chunks[0]
        geometry = (chunk.region_id, chunk.offset, chunk.size, chunk.domain_id)
        if not 0 < chunk.size <= 4 << 30 or chunk.size != owner.total_size:
            raise ValueError("invalid raw destination allocation capacity")
        return geometry, owner

    def tensor_nbytes(self, tensor) -> int:
        geometry, _owner = self.allocation_descriptor(tensor)
        return geometry[2]

    def transfer_alignment(self) -> int:
        return mmap.PAGESIZE

    def cast_tensor(self, tensor: TensorBase, dtype: DType) -> TensorBase:
        return self.as_tensor(self.unwrap(tensor).to(dtype=dtype_convert[dtype]))

    def validate_transfer(self, tensor, device) -> int:
        size = self.tensor_nbytes(tensor)
        if self.as_tensor(tensor).device != self.device(device):
            raise ValueError("raw transfer destination is on a different device")
        if not 0 < size <= 4 << 30 or (device.type == DeviceType.SPYRE and size % 128):
            raise ValueError(f"invalid raw transfer size: {size}")
        return size

    def allocate_host_buffer(self, nbytes: int) -> TensorBase:
        """Keep both fixed-slot storage and device mapping alive through views."""
        import torch

        if nbytes <= 0:
            raise ValueError("host buffer size must be positive")
        device = self._torch_device(self._device)
        if device.type == "spyre":
            torch.spyre._lazy_init()  # type: ignore[attr-defined]
            return self.as_tensor(
                torch.empty(nbytes, dtype=torch.uint8, device="cpu", pin_memory=True)
            )
        # CPU emulation must not initialize hardware or import the backend.
        return self.host_tensor_from_buffer(mmap.mmap(-1, nbytes))

    def prepare_copy(
        self, device: Device
    ) -> Callable[[list[tuple[TensorBase, TensorBase, int, int]]], HostTransfer | None]:
        """Capture the caller's stream; enqueue now and return an owning ticket."""
        if device.type != DeviceType.SPYRE:
            # Allows CPU emulation without importing/initializing torch-spyre.
            return self.copy_raw_tensors
        from torch_spyre import current_stream

        target = self._torch_device(device)
        stream = current_stream(target)

        def copy(
            entries: list[tuple[TensorBase, TensorBase, int, int]],
        ) -> HostTransfer | None:
            return self.submit_raw_tensors(entries, stream=stream, device=target)

        return copy

    def host_tensor_from_buffer(self, buffer) -> TensorBase:
        import torch

        return self.as_tensor(torch.frombuffer(buffer, dtype=torch.uint8))

    def host_buffer_view(self, tensor) -> memoryview:
        import torch

        tensor = self.unwrap(self.as_tensor(tensor))
        return memoryview(tensor.view(-1).view(torch.uint8).numpy())

    def encode_tensors(
        self, work: list[tuple[memoryview, TensorBase, TensorBase]]
    ) -> None:
        import torch

        entries = []
        for output, src, dst in work:
            source, destination = self.unwrap(src), self.unwrap(dst)
            dtype = fst_dtype(destination.dtype)
            shape = list(destination.shape)
            layout, size = canonical_layout(shape, dtype)
            if (
                source.device.type != "cpu"
                or source.dtype != destination.dtype
                or list(source.shape) != shape
                or list(source.stride()) != dense_stride(shape)
                or source.storage_offset() != 0
                or source.is_conj()
                or source.is_neg()
                or self.tensor_layout(dst) != layout
                or list(destination.stride()) != dense_stride(shape)
                or self.tensor_nbytes(dst) != size
            ):
                raise ValueError("conversion requires dense base fixed-profile tensors")
            # U8 numpy is supported for BF16 too; memoryview owns numpy's Tensor
            # storage reference. No worker reads a Tensor or arbitrary data_ptr.
            view = memoryview(source.detach().reshape(-1).view(torch.uint8).numpy())
            entries.append((output, view, shape, dtype.value))
        backend().convert_buffers(entries)

    def encode_buffer(self, output, source, shape, dtype) -> None:
        """Convert a compact CPU tile; never allocate or inspect a device tensor."""
        backend().convert_buffers(
            [
                (
                    output,
                    self.host_buffer_view(source),
                    list(shape),
                    dtype.value,
                )
            ]
        )

    def copy_raw_tensors(self, entries, *, stream=None, device=None) -> None:
        """Blocking convenience boundary; the reader uses submit_raw_tensors."""
        ticket = self.submit_raw_tensors(entries, stream=stream, device=device)
        if ticket is not None:
            ticket.wait()

    def submit_raw_tensors(
        self, entries, *, stream=None, device=None
    ) -> HostTransfer | None:
        import torch

        if not entries:
            return None
        if device is None:
            device = self.unwrap(self.as_tensor(entries[0][1])).device
        # Whole payloads remain useful to callers outside the file reader.
        work = []
        for entry in entries:
            buffer, dst = entry[:2]
            offset, size = entry[2:] if len(entry) == 4 else (0, buffer.get_nbytes())
            raw = self.unwrap(buffer)
            target = self.unwrap(self.as_tensor(dst))
            if (
                raw.device.type != "cpu"
                or raw.dtype != torch.uint8
                or not raw.is_contiguous()
                or target.device != device
            ):
                raise ValueError("raw copy requires CPU uint8 and one Spyre device")
            _geometry, owner = self.allocation_descriptor(dst)
            work.append((self.host_buffer_view(buffer), owner, offset, size))
        if stream is None:
            from torch_spyre import current_stream

            stream = current_stream(device)
        from ..copier.host import TRANSFER_LOCK, check_transfer_state

        api = backend()
        with TRANSFER_LOCK:
            check_transfer_state()
            # Registry owns the ticket before even a partial enqueue can occur.
            ticket = _RawTransfer(entries, work, stream, api)
            try:
                api.copy_buffer_ranges(work, stream=stream)
            except BaseException as error:
                # Earlier entries may already be enqueued. Never let an enqueue
                # exception bypass drain/quarantine, even without a returned ticket.
                ticket.wait()
                raise TransferDrainedError(str(error)) from error
            return ticket

    def move_tensor(
        self,
        source: TensorBase,
        device,
        layout=None,
        destination: TensorBase | None = None,
    ) -> TensorBase:
        if destination is not None:
            from torch_spyre._C import copy_tensor

            copy_tensor(
                self.unwrap(source), self.unwrap(destination), non_blocking=False
            )
            return destination
        raw = self.unwrap(source)
        device = self._torch_device(device)
        if layout is None:
            return self.as_tensor(raw.to(device))
        return self.as_tensor(raw.to(device, dtype=raw.dtype, device_layout=layout))
