# SPDX-License-Identifier: Apache-2.0
"""The fixed, declarative ``spyre.native.v1`` safetensors profile.

Every tensor in a profiled file is a U8 byte carrier. Only ``profile`` and
``spyre.overrides`` describe its native representation; portable tensors belong
in separate files. Layout dictionaries below are trusted derived runtime ABI,
never objects supplied by the artifact.
"""

import json
import os
from dataclasses import dataclass
from math import prod
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

from .common import SafeTensorsMetadata
from .st_types import DType

NATIVE_VERSION = 1

PROFILE = "spyre.native.v1"
KEY_PROFILE = "profile"
KEY_TENSORS = "spyre.overrides"
MAX_TENSORS = 100_000
MAX_RANK = 16
MAX_EXTENT = (1 << 31) - 1
MAX_TENSOR_BYTES = 4 << 30
MAX_TOTAL_BYTES = 1 << 40


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


def dense_stride(shape):
    stride, result = 1, []
    for extent in reversed(shape):
        result.append(stride)
        stride *= max(extent, 1)
    return list(reversed(result))


def canonical_layout(shape, dtype, format=None, layout="stick-last.v1"):
    """Derive the frozen stick-last mapping, not the runtime's current default.

    Physical axes are [1, ..., rank-1, 0, stick]; the last logical axis is
    split into outer sticks and inner lanes. Singleton axes are retained.
    Scalars occupy one stick. Empty tensors are not admitted by this backend.
    """
    formats = {
        DType.F16: ("dlfloat16", "SEN169_FP16", 64),
        DType.BF16: ("dlfloat16", "SEN169_FP16", 64),
        DType.F32: ("ieee-fp32", "IEEE_FP32", 32),
        DType.I32: ("int32", "IEEE_INT32", 32),
    }
    if dtype not in formats or layout != "stick-last.v1":
        raise NativeFormatError("unsupported native dtype/layout combination")
    expected, device_dtype, stick = formats[dtype]
    if format is not None and format != expected:
        raise NativeFormatError("unsupported native dtype/format combination")
    if not isinstance(shape, list) or len(shape) > MAX_RANK:
        raise NativeFormatError("invalid logical_shape rank")
    if any(type(n) is not int or not 0 < n <= MAX_EXTENT for n in shape):
        raise NativeFormatError(
            "logical_shape extents must be positive bounded integers"
        )
    if not shape:
        return {
            "device_size": [1, stick],
            "stride_map": [0, 0],
            "device_dtype": device_dtype,
            "element_arrangement": "STANDARD",
            "valid_elements": [[[0, 1], 1]],
        }, 128
    strides = dense_stride(shape)
    rank = len(shape)
    sizes = list(shape)
    sizes[-1] = (sizes[-1] + stick - 1) // stick
    sizes = sizes[1:] + sizes[:1] + [stick]
    nbytes = 128
    for size in sizes[:-1]:
        nbytes *= size
        if nbytes > MAX_TENSOR_BYTES:
            raise NativeFormatError("native tensor exceeds byte limit")
    stride_map = [0] * (rank + 1)
    last = [-1] * rank
    groups = {}
    for host_axis, dev_axis in [(rank - 1, rank)] + [
        (i, rank - 1 if i == 0 else i - 1) for i in reversed(range(rank))
    ]:
        if shape[host_axis] != 1:
            stride_map[dev_axis] = (
                strides[host_axis] if last[host_axis] == -1 else last[host_axis]
            )
            last[host_axis] = min(
                stride_map[dev_axis] * sizes[dev_axis],
                strides[host_axis] * shape[host_axis],
            )
        groups.setdefault(host_axis, []).append(dev_axis)
    valid = sorted([[sorted(axes), shape[i]] for i, axes in groups.items()])
    return {
        "device_size": sizes,
        "stride_map": stride_map,
        "device_dtype": device_dtype,
        "element_arrangement": "STANDARD",
        "valid_elements": valid,
    }, nbytes


def validate_native_spec(spec):
    expected, nbytes = canonical_layout(spec.shape, spec.dtype)
    if (
        spec.layout != expected
        or spec.stride != dense_stride(spec.shape)
        or spec.nbytes != nbytes
    ):
        raise NativeFormatError(
            f"{spec.name}: tensor does not conform to stick-last.v1"
        )


def _target(value):
    if not isinstance(value, dict) or set(value) != {"tp_size", "tp_rank"}:
        raise NativeFormatError("target requires exactly tp_size and tp_rank")
    size, rank = value["tp_size"], value["tp_rank"]
    if (
        type(size) is not int
        or type(rank) is not int
        or not 0 <= rank < size <= MAX_TENSORS
    ):
        raise NativeFormatError("invalid TP target identity")
    return size, rank


def native_metadata(
    fmt: NativeFormat, specs: Mapping[str, NativeTensorSpec], *, tp_size=1, tp_rank=0
) -> Dict[str, str]:
    """Encode only necessary logical facts; reject arbitrary runtime layouts."""
    if fmt != NativeFormat("spyre", PROFILE):
        raise NativeFormatError("unsupported native profile")
    _target({"tp_size": tp_size, "tp_rank": tp_rank})
    if len(specs) > MAX_TENSORS:
        raise NativeFormatError("native tensor count exceeds limit")
    tensors = {}
    total = 0
    for name, spec in specs.items():
        if not isinstance(name, str) or spec.name != name:
            raise NativeFormatError("native spec name differs from storage key")
        validate_native_spec(spec)
        total += spec.nbytes
        if total > MAX_TOTAL_BYTES:
            raise NativeFormatError("native shard exceeds aggregate byte budget")
        descriptor: Dict[str, Any] = {"logical_shape": spec.shape}
        if spec.dtype != DType.BF16:
            descriptor["logical_dtype"] = spec.dtype.value
        if spec.dtype == DType.F32:
            descriptor["format"] = "ieee-fp32"
        elif spec.dtype == DType.I32:
            descriptor["format"] = "int32"
        tensors[name] = descriptor
    raw = {"tensors": tensors}
    if (tp_size, tp_rank) != (1, 0):
        raw["target"] = {"tp_size": tp_size, "tp_rank": tp_rank}
    encoded = json.dumps(raw, separators=(",", ":"))
    return {KEY_PROFILE: PROFILE, KEY_TENSORS: encoded}


def _metadata_dict(meta: SafeTensorsMetadata) -> Mapping[str, str]:
    return meta.metadata if isinstance(meta.metadata, dict) else {}


def native_format(meta: SafeTensorsMetadata) -> Optional[NativeFormat]:
    """The file's native format, or None for a plain safetensors file."""
    md = _metadata_dict(meta)
    if any(k.startswith("fst.native.") for k in md):
        raise NativeFormatError(
            "legacy fst.native metadata is unsupported; regenerate the artifact"
        )
    if KEY_PROFILE not in md and KEY_TENSORS not in md:
        if any(k.startswith("spyre.") for k in md):
            raise NativeFormatError("Spyre metadata requires a profile")
        return None
    if md.get(KEY_PROFILE) != PROFILE or KEY_TENSORS not in md:
        raise NativeFormatError(f"{meta.src}: unsupported or incomplete native profile")
    if any(k.startswith("spyre.") and k != KEY_TENSORS for k in md):
        raise NativeFormatError("unknown Spyre metadata key")
    if "torch_spyre.moved" in md:
        raise NativeFormatError("native and portable tensors must be in separate files")
    return NativeFormat("spyre", PROFILE)


def native_specs(
    meta: SafeTensorsMetadata, *, tp_size=None, tp_rank=None
) -> Dict[str, NativeTensorSpec]:
    """The file's device-native tensors, checked against its tensor table."""
    if native_format(meta) is None:
        return {}
    try:
        raw = json.loads(_metadata_dict(meta)[KEY_TENSORS])
        if not isinstance(raw, dict) or not {"tensors"} <= raw.keys() <= {
            "target",
            "tensors",
        }:
            raise ValueError("overrides requires tensors and optional target only")
        target = _target(raw.get("target", {"tp_size": 1, "tp_rank": 0}))
        if tp_size is not None or tp_rank is not None:
            if _target({"tp_size": tp_size, "tp_rank": tp_rank}) != target:
                raise ValueError("TP target does not match requested partition")
        tensors = raw["tensors"]
        if (
            not isinstance(tensors, dict)
            or set(tensors) != set(meta.tensors)
            or len(tensors) > MAX_TENSORS
        ):
            raise ValueError("native descriptors must match every storage key exactly")
    except (KeyError, TypeError, ValueError) as e:
        raise NativeFormatError(f"{meta.src}: bad {KEY_TENSORS}: {e}") from e
    specs = {}
    total = 0
    for name, spec in tensors.items():
        frame = meta.tensors[name]
        try:
            if not isinstance(spec, dict) or not {"logical_shape"} <= spec.keys() <= {
                "logical_shape",
                "logical_dtype",
                "format",
                "layout",
            }:
                raise ValueError("invalid tensor descriptor fields")
            if any(
                not isinstance(spec[field], str)
                for field in ("logical_dtype", "format", "layout")
                if field in spec
            ):
                raise ValueError("dtype, format and layout identifiers must be strings")
            dtype = DType(spec.get("logical_dtype", "BF16"))
            layout, nbytes = canonical_layout(
                spec["logical_shape"],
                dtype,
                spec.get("format", "dlfloat16"),
                spec.get("layout", "stick-last.v1"),
            )
            if (
                frame.dtype != DType.U8
                or frame.shape != [nbytes]
                or frame.data_offsets[1] - frame.data_offsets[0] != nbytes
            ):
                raise ValueError("native U8 carrier size differs from derived layout")
            total += nbytes
            if total > MAX_TOTAL_BYTES:
                raise ValueError("native shard exceeds aggregate byte budget")
            specs[name] = NativeTensorSpec(
                name,
                dtype,
                spec["logical_shape"],
                dense_stride(spec["logical_shape"]),
                layout,
                nbytes,
            )
        except (KeyError, TypeError, ValueError) as e:
            raise NativeFormatError(f"{meta.src}: bad spec for {name}: {e}")
    return specs


def checkpoint_paths(source: str) -> List[str]:
    """Resolve a file, HF safetensors index, or checkpoint directory (no download)."""
    # HF snapshots use symlinks to extensionless blobs. Keep the index's name
    # and parent directory when resolving its relative shard references.
    path = Path(os.path.abspath(source))
    if path.is_dir():
        index = path / "model.safetensors.index.json"
        if index.exists():
            return checkpoint_paths(str(index))
        paths = sorted(str(p) for p in path.glob("*.safetensors"))
    elif path.name.endswith(".index.json"):
        with path.open() as stream:
            names = sorted(set(json.load(stream)["weight_map"].values()))
        paths = []
        for name in names:
            candidate = path.parent / name
            if Path(name).is_absolute() or ".." in Path(name).parts:
                raise ValueError(f"unsafe checkpoint shard path: {name}")
            paths.append(str(candidate))
    else:
        paths = [str(path)]
    if not paths or any(not os.path.isfile(p) for p in paths):
        raise FileNotFoundError(f"checkpoint files not found: {source}")
    return paths


def conversion_tiles(shape, dtype, source_width, workspace_bytes):
    """Yield compact shape, source runs, output offset and length for bounded I/O.

    Input/cast tensors and each output interval are individually <= workspace_bytes.
    No device tensor or full-sized host source is needed, including wide rows.
    """
    canonical_layout(shape, dtype)
    width = 2 if dtype in {DType.F16, DType.BF16} else 4
    stick = 128 // width
    # A complete padded stick is the smallest independent conversion unit.
    row_budget = workspace_bytes // max(128, stick * source_width)
    if row_budget < 1:
        raise ValueError("conversion workspace must fit one input/output stick")
    if not shape:
        yield [], [(0, 1)], 0, 128
        return
    if len(shape) == 1:
        columns_per_tile = row_budget * stick
        for col in range(0, shape[0], columns_per_tile):
            count = min(columns_per_tile, shape[0] - col)
            yield [count], [(col, count)], col // stick * 128, (
                (count + stick - 1) // stick * 128
            )
        return
    rows, columns = shape[0], shape[-1]
    middle_count = prod(shape[1:-1])
    blocks = (columns + stick - 1) // stick
    rows_per_tile = min(rows, row_budget)
    blocks_per_tile = max(1, row_budget // rows) if rows_per_tile == rows else 1
    for middle in range(middle_count):
        for block in range(0, blocks, blocks_per_tile):
            tile_blocks = min(blocks_per_tile, blocks - block)
            col = block * stick
            count = min(tile_blocks * stick, columns - col)
            for row in range(0, rows, rows_per_tile):
                tile_rows = min(rows_per_tile, rows - row)
                runs = [
                    ((r * middle_count + middle) * columns + col, count)
                    for r in range(row, row + tile_rows)
                ]
                offset = ((middle * blocks + block) * rows + row) * 128
                yield [tile_rows, count], runs, offset, tile_rows * tile_blocks * 128


def convert_to_spyre(
    source: str,
    base: str,
    *,
    device: str = "spyre:0",
    dtype: DType = DType.F16,
    num_shards: int = 16,
    fsync: bool = False,
    progress=None,
    workspace_bytes: int = 8 << 20,
) -> List[str]:
    """Convert a plain checkpoint without importing spyre-inference.

    No device tensor is allocated. Input reads and CPU casts use compact tiles;
    each is <= workspace_bytes (both may coexist). ConvertData writes one small
    physical interval of the output mmap at a time. File mappings/page cache are
    not an RSS limit. The frozen stick-last layout and 4 GiB limit are unchanged.
    Existing output shards are never overwritten. No model/TP transforms.
    """
    from .copier.host import pread_into
    from .frameworks import get_framework_op
    from .saver import WriteEntry, plan_shards, shard_paths, write_shards
    from .st_types import Device, DeviceType

    dtype = DType(dtype)
    if dtype not in {DType.F16, DType.F32, DType.BF16}:
        raise ValueError("conversion dtype must be F16, F32, or BF16")
    if num_shards <= 0:
        raise ValueError("num_shards must be positive")
    if type(workspace_bytes) is not int or workspace_bytes < 512:
        raise ValueError("conversion workspace must be an integer >= 512 bytes")
    target = Device.from_str(device)
    if target.type != DeviceType.SPYRE:
        raise ValueError("conversion destination must be Spyre")
    framework = get_framework_op("pytorch")
    ops = framework.get_native_ops(target)
    ops.load_library()
    entries = []
    names = set()
    for path in checkpoint_paths(source):
        meta = SafeTensorsMetadata.from_file(path, framework)
        if native_format(meta) is not None:
            raise NativeFormatError(f"{path}: conversion requires a plain checkpoint")
        for name, frame in meta.tensors.items():
            if frame.dtype not in {DType.F16, DType.F32, DType.BF16, DType.F64}:
                raise ValueError(f"{name}: conversion requires floating-point weights")
            if name in names:
                raise ValueError(f"duplicate tensor name: {name}")
            names.add(name)
            layout, size = canonical_layout(frame.shape, dtype)
            spec = NativeTensorSpec(
                name, dtype, frame.shape, dense_stride(frame.shape), layout, size
            )
            entries.append(
                WriteEntry(
                    name,
                    DType.U8,
                    [size],
                    size,
                    (path, meta.header_length, frame, spec),
                )
            )

    def metadata(shard):
        return native_metadata(
            NativeFormat("spyre", PROFILE), {e.name: e.source[3] for e in shard}
        )

    paths = shard_paths(base, num_shards)
    if any(os.path.exists(path) for path in paths):
        raise FileExistsError("conversion output already exists; choose a new base")
    plans = plan_shards(entries, paths, shard_metadata=metadata)

    def fill(work):
        for buffer, entry in work:
            path, header, frame, spec = entry.source
            source_width = int(framework.get_dtype_size(frame.dtype))
            with open(path, "rb") as stream:
                for shape, runs, offset, length in conversion_tiles(
                    spec.shape, dtype, source_width, workspace_bytes
                ):
                    src = framework.get_empty_tensor(shape, frame.dtype, Device())
                    with ops.host_buffer_view(src) as view:
                        cursor = 0
                        for start, count in runs:
                            end = cursor + count * source_width
                            with view[cursor:end] as part:
                                pread_into(
                                    stream.fileno(),
                                    part,
                                    header
                                    + frame.data_offsets[0]
                                    + start * source_width,
                                )
                            cursor = end
                    cast = ops.cast_tensor(src, dtype)
                    with buffer[offset : offset + length] as output:
                        ops.encode_buffer(output, cast, shape, dtype)
                    del cast, src
            if progress is not None:
                progress(entry.name, entry.nbytes)

    return write_shards(plans, fill, fsync=fsync)
