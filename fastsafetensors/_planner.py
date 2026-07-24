# SPDX-License-Identifier: Apache-2.0

"""Internal planning helpers for sub-file chunked loading.

Not part of the public API: the entry point is
``ParallelLoader(max_batch_bytes=...)``, which plans chunks here and feeds
them to partial-read copiers via the loader's chunk plan.
"""

from typing import Callable, List, Optional, Set, Tuple

from .common import SafeTensorsMetadata


def plan_chunks(
    metadata: SafeTensorsMetadata,
    max_batch_bytes: int,
    keep_tensor: Optional[Callable[[str], bool]] = None,
    merge_gap: int = 4096,
) -> List[Tuple[Set[str], List[Tuple[int, int]]]]:
    """Partition a shard's tensors into byte-budgeted sub-file chunks.

    Walks the (optionally ``keep_tensor``-filtered) tensors in file-offset
    order and greedily packs consecutive tensors into a chunk while the
    chunk's byte span stays within ``max_batch_bytes``. Returns a list of
    ``(names, byte_ranges)`` pairs; feed each to a copier's ``set_chunk`` to
    load only that chunk, bounding peak device memory to ~``max_batch_bytes``
    (the device buffer covers the chunk's span). The byte ranges are the
    kept tensors' runs within the chunk (kept tensors separated by at most
    ``merge_gap`` bytes are coalesced), so holes left by a filter are not
    read even though the buffer spans them.

    A tensor is the atomic load unit, so ``max_batch_bytes`` must be at least
    the largest kept tensor -- otherwise that tensor fits in no chunk and a
    ``ValueError`` is raised naming it.
    """
    chunks: List[Tuple[Set[str], List[Tuple[int, int]]]] = []
    cur: List[str] = []
    cur_runs: List[List[int]] = []  # merged kept runs within the chunk
    cur_start = 0

    def _push(name: str, s: int, e: int) -> None:
        if cur_runs and s - cur_runs[-1][1] <= merge_gap:
            cur_runs[-1][1] = max(cur_runs[-1][1], e)
        else:
            cur_runs.append([s, e])

    for name, frame in metadata.tensors.items():  # insertion order == offset order
        if keep_tensor is not None and not keep_tensor(name):
            continue
        s = metadata.header_length + frame.data_offsets[0]
        e = metadata.header_length + frame.data_offsets[1]
        tensor_bytes = frame.data_offsets[1] - frame.data_offsets[0]
        if tensor_bytes > max_batch_bytes:
            raise ValueError(
                f"max_batch_bytes={max_batch_bytes} is smaller than tensor "
                f"'{name}' ({tensor_bytes} bytes); it must be at least the "
                f"largest kept tensor so every tensor fits in one chunk."
            )
        if cur and (e - cur_start) > max_batch_bytes:
            chunks.append((set(cur), [(s0, e0) for s0, e0 in cur_runs]))
            cur, cur_runs = [], []
        if not cur:
            cur_start = s
        cur.append(name)
        _push(name, s, e)
    if cur:
        chunks.append((set(cur), [(s0, e0) for s0, e0 in cur_runs]))
    return chunks
