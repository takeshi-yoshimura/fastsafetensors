# SPDX-License-Identifier: Apache-2.0
"""Device-native payloads in standard safetensors files.

A device-native tensor is stored as the bytes a device keeps it in (usually
``U8 [nbytes]``), which only that device's backend can interpret. The files stay
standard safetensors; everything else lives in ``__metadata__`` (string values
only):

``fst.native.version``  this convention's version (``NATIVE_VERSION``)
``fst.native.device``   the device backend the payloads are for (e.g. ``spyre``)
``fst.native.format``   the backend's own format id; only the backend judges it
``fst.native.tensors``  JSON ``{name: {"dtype", "shape", "stride", "layout"}}``:
                        the logical tensor each payload holds; ``layout`` is
                        opaque here and handed to the backend untouched

Tensors not listed in ``fst.native.tensors`` are ordinary safetensors tensors.
A set of shards written together (``fastsafetensors.saver``) also carries
``fst.shard.index`` / ``fst.shard.count`` / ``fst.shard.set`` (a shared id).
"""

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional

from .common import SafeTensorsMetadata
from .st_types import DType

NATIVE_VERSION = 1

KEY_VERSION = "fst.native.version"
KEY_DEVICE = "fst.native.device"
KEY_FORMAT = "fst.native.format"
KEY_TENSORS = "fst.native.tensors"


class NativeFormatError(ValueError):
    """A file's device-native metadata is missing, malformed or incompatible."""


@dataclass(frozen=True)
class NativeFormat:
    device: str
    format: str
    version: int = NATIVE_VERSION


@dataclass(frozen=True)
class NativeTensorSpec:
    name: str
    dtype: DType
    shape: List[int]
    stride: Optional[List[int]]
    layout: Any
    # Size of the stored payload.
    nbytes: int


def _metadata_dict(meta: SafeTensorsMetadata) -> Mapping[str, str]:
    return meta.metadata if isinstance(meta.metadata, dict) else {}


def native_format(meta: SafeTensorsMetadata) -> Optional[NativeFormat]:
    """The file's native format, or None for a plain safetensors file."""
    md = _metadata_dict(meta)
    if KEY_VERSION not in md:
        return None
    try:
        version = int(md[KEY_VERSION])
    except ValueError:
        raise NativeFormatError(f"{meta.src}: bad {KEY_VERSION}={md[KEY_VERSION]!r}")
    if version != NATIVE_VERSION:
        raise NativeFormatError(
            f"{meta.src}: native metadata version {version} is not supported "
            f"(expected {NATIVE_VERSION})"
        )
    if KEY_DEVICE not in md or KEY_FORMAT not in md:
        raise NativeFormatError(f"{meta.src}: {KEY_DEVICE}/{KEY_FORMAT} missing")
    return NativeFormat(md[KEY_DEVICE], md[KEY_FORMAT], version)


def native_specs(meta: SafeTensorsMetadata) -> Dict[str, NativeTensorSpec]:
    """The file's device-native tensors, checked against its tensor table."""
    if native_format(meta) is None:
        return {}
    try:
        raw = json.loads(_metadata_dict(meta).get(KEY_TENSORS, "{}"))
    except ValueError as e:
        raise NativeFormatError(f"{meta.src}: bad {KEY_TENSORS}: {e}")
    specs = {}
    for name, spec in raw.items():
        frame = meta.tensors.get(name)
        if frame is None:
            raise NativeFormatError(f"{meta.src}: native tensor {name} has no payload")
        try:
            specs[name] = NativeTensorSpec(
                name=name,
                dtype=DType(spec["dtype"]),
                shape=list(spec["shape"]),
                stride=spec.get("stride"),
                layout=spec.get("layout"),
                nbytes=frame.data_offsets[1] - frame.data_offsets[0],
            )
        except (KeyError, TypeError, ValueError) as e:
            raise NativeFormatError(f"{meta.src}: bad spec for {name}: {e}")
    return specs
