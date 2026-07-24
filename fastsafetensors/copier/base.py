# SPDX-License-Identifier: Apache-2.0

import operator
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

from .. import cpp as fstcpp
from ..common import SafeTensorsMetadata
from ..frameworks import TensorBase
from ..st_types import DType

if TYPE_CHECKING:
    from ..allocation import SharedDeviceAllocation


def validated_byte_ranges(
    metadata: SafeTensorsMetadata,
    byte_ranges: Optional[List[Tuple[int, int]]],
) -> Optional[List[Tuple[int, int]]]:
    """Validate ``[start, end)`` absolute file-offset runs against *metadata*
    and return a defensive copy.

    Runs must be integer pairs within the data section
    (``header_length <= start < end <= size_bytes``), sorted and
    non-overlapping. ``None`` (full read) passes through; an empty list reads
    nothing. Shared by every copier so range checks live at one API boundary.
    """
    if byte_ranges is None:
        return None
    checked: List[Tuple[int, int]] = []
    prev_end = metadata.header_length
    for i, run in enumerate(byte_ranges):
        try:
            start, end = run
        except (TypeError, ValueError):
            raise ValueError(
                f"byte_ranges[{i}]: expected a (start, end) pair, got {run!r}"
            ) from None
        if isinstance(start, bool) or isinstance(end, bool):
            raise ValueError(f"byte_ranges[{i}]: offsets must be int, got {run!r}")
        try:
            start, end = operator.index(start), operator.index(end)
        except TypeError:
            raise ValueError(
                f"byte_ranges[{i}]: offsets must be int, got {run!r}"
            ) from None
        if end <= start:
            raise ValueError(f"byte_ranges[{i}]: empty or reversed run {run!r}")
        if start < prev_end:
            raise ValueError(
                f"byte_ranges[{i}]: runs must be sorted, non-overlapping, and "
                f"start at or after the data section "
                f"(start={start}, minimum here {prev_end})"
            )
        if end > metadata.size_bytes:
            raise ValueError(
                f"byte_ranges[{i}]: end={end} is beyond the end of "
                f"{metadata.src} ({metadata.size_bytes} bytes)"
            )
        checked.append((start, end))
        prev_end = end
    return checked


class CopierInterface(ABC):
    metadata: SafeTensorsMetadata

    def set_byte_ranges(self, byte_ranges: Optional[List[Tuple[int, int]]]) -> None:
        """Restrict reads to these ``[start, end)`` absolute file-offset runs.

        The default implementation validates the runs but reads the whole file,
        so the byte-range filter is a correct no-op on copiers that don't
        implement partial reads. Range-capable copiers (``nogds``, ``unified``)
        override this to read only the given runs, leaving the rest of the
        device buffer uninitialized (so skipped tensors must not be requested).
        Build runs with ``SafeTensorsMetadata.select_byte_ranges``; ``None``
        means full read.
        """
        validated_byte_ranges(self.metadata, byte_ranges)

    @abstractmethod
    def submit_io(
        self, use_buf_register: bool, max_copy_block_size: int
    ) -> fstcpp.gds_device_buffer:
        pass

    @abstractmethod
    def wait_io(
        self,
        gbuf: fstcpp.gds_device_buffer,
        dtype: DType = DType.AUTO,
        noalign: bool = False,
        owner: Optional["SharedDeviceAllocation"] = None,
    ) -> Dict[str, TensorBase]:
        """Materialize tensors from the completed I/O.

        *owner*, when given, is the ``SharedDeviceAllocation`` backing *gbuf*;
        it must be forwarded to ``metadata.get_tensors`` so every returned
        tensor shares ownership of the buffer and stays valid after close.
        """
        pass


class DummyDeviceBuffer(fstcpp.gds_device_buffer):
    def __init__(self):
        super().__init__(0, 0, False)
