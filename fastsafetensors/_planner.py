# SPDX-License-Identifier: Apache-2.0

"""Internal planners for bounded-memory loading.

Not part of the public API: the entry points are
``ParallelLoader(max_batch_bytes=..., device_memory_budget=...)``.

plan_chunks partitions one shard into byte-budgeted sub-file chunks; the
fit planner below turns a whole-load device memory budget into per-file
chunk budgets.


Given the byte sizes of every (kept) tensor -- all known upfront from
safetensors headers -- and a total ``device_memory_budget`` the load may
occupy at peak, compute a per-file chunk budget so that

    resident_bytes + live_transient_buffers <= budget

holds at every moment of the load. Budgets are large while cumulative
resident bytes are small (whole-file loads, no chunking overhead) and
decline only as the device fills, so chunking cost is paid only where the
fit actually requires it. The plan is precomputed and deterministic: same
files, same filter, same budget -> same plan. There is no runtime feedback.

Why per-file budgets are safe (the bound): let ``G(i)`` be the last file of
file ``i``'s batch group -- the ``group_size`` files loaded concurrently, one
per rank. While any chunk of file ``i`` is alive, resident bytes are at most
``R[G(i)+1]`` (only tensors of files ``<= G(i)`` have been materialized; the
group's kept bytes are charged up front, because broadcast leaves every rank
holding every file of the group) and every live transient buffer belongs to
file ``i`` or a later file ``j > i``. The per-file budget ``B`` declines
monotonically with ``R``, so every live buffer span is ``<= B[i]``. With at
most ``depth * transient_multiplier`` buffers plus one yield clone alive on
any one rank,

    peak <= R[G(i)+1]
            + (depth * transient_multiplier + yield_clone) * B[i]
         <= budget   (by choice of B[i]).

With ``group_size == 1``, ``G(i) == i`` and this reduces to ``R[i+1]``.
``yield_clone`` is 1 when a non-resident yielded tensor is cloned, else 0.

``fit_queue_size`` inverts this bound: given a budget it returns the deepest
queue size whose plan still fits, so a caller can clamp its pipeline depth to
what the device affords instead of failing the load. The inversion is exact
and needs no trial loads -- every term is known from the safetensors headers.

The budget itself is the caller's to choose -- only the caller knows what
else will live on the device. A caller sizing it from free memory should
keep a reserve for allocator rounding and the copier's fixed pools (5% or
1 GiB, whichever is larger, is a reasonable starting point), e.g.::

    free, _ = torch.cuda.mem_get_info(dev)
    budget = free - max(free // 20, 1 << 30)

and, under broadcast loading, all-reduce(MIN) that value before passing it:
per-rank readings diverge, and differing budgets would give ranks different
plans and deadlock the lockstep broadcast sequence.
"""

from dataclasses import dataclass
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


class BudgetInfeasibleError(ValueError):
    """The model cannot be loaded within the given device memory budget."""


@dataclass(frozen=True)
class FileWeightStats:
    """Per-file byte accounting for the fit plan (kept tensors only)."""

    path: str
    kept_bytes: int  # sum of kept tensor bytes: resident growth
    span_bytes: int  # last kept byte - first kept byte: single-chunk buffer size
    largest_tensor: int  # chunk floor: a tensor is the atomic load unit


def pipeline_depth(queue_size: int) -> int:
    """Worst-case number of concurrently live device buffers for a queue size.

    queue_size -1: fully serial, 1 buffer. 0: unbuffered pipeline, the producer
    materializes the next buffer while the consumer holds one -> 2. n > 0: n in
    the queue + 1 being produced + 1 being consumed -> n + 2.
    """
    if queue_size < 0:
        return 1
    return queue_size + 2


def load_depth(queue_size: int, group_size: int = 1) -> int:
    """Concurrently live device buffers for a queue size and group width.

    ``pipeline_depth`` counts the producer/consumer buffers; broadcast loading
    (``group_size > 1``) adds one more for the in-flight receive tensor. This
    is the ``depth`` that ``plan_file_budgets`` expects, and ``fit_queue_size``
    inverts exactly this function -- both live here so a change to the
    broadcast term cannot make a plan and its suggested queue size disagree.
    """
    return pipeline_depth(queue_size) + (1 if group_size > 1 else 0)


def _queue_size_of_depth(depth: int, group_size: int = 1) -> int:
    """Inverse of ``load_depth``.

    Not injective at the bottom -- ``queue_size=-1`` and no queue at all both
    give base depth 1 -- so base depth 1 maps back to -1 and anything deeper
    to ``base - 2``.
    """
    base = depth - (1 if group_size > 1 else 0)
    return -1 if base <= 1 else base - 2


def _effective_depth(
    depth: int, transient_multiplier: int, account_for_yield_clone: bool
) -> int:
    """Live transient buffers, each charged one chunk budget."""
    return depth * transient_multiplier + int(account_for_yield_clone)


def collect_file_stats(
    metas: List[Tuple[str, SafeTensorsMetadata]],
    keep_tensor: Optional[Callable[[str], bool]] = None,
) -> List[FileWeightStats]:
    """Byte accounting per file from already-parsed headers."""
    stats = []
    for path, meta in metas:
        kept = span_start = span_end = largest = 0
        first = True
        for name, frame in meta.tensors.items():
            if keep_tensor is not None and not keep_tensor(name):
                continue
            s, e = frame.data_offsets[0], frame.data_offsets[1]
            kept += e - s
            largest = max(largest, e - s)
            if first:
                span_start, first = s, False
            span_end = max(span_end, e)
        stats.append(
            FileWeightStats(path, kept, span_end - span_start if kept else 0, largest)
        )
    return stats


def _group_resident(stats: List[FileWeightStats], group_size: int) -> List[int]:
    """Cumulative kept bytes through the end of each file's batch group.

    With ``group_size == 1`` this is just ``R[i+1]``; under broadcast the whole
    group is in flight at once, so every file in it is charged the group total.
    """
    kept_through_group = []
    running = 0
    for st in stats:
        running += st.kept_bytes
        kept_through_group.append(running)
    return [
        kept_through_group[min((i // group_size + 1) * group_size, len(stats)) - 1]
        for i in range(len(stats))
    ]


def plan_file_budgets(
    stats: List[FileWeightStats],
    device_memory_budget: int,
    depth: int,
    max_batch_bytes: Optional[int] = None,
    accumulate_resident: bool = True,
    transient_multiplier: int = 1,
    group_size: int = 1,
    account_for_yield_clone: bool = False,
) -> List[int]:
    """Per-file chunk budgets satisfying the peak-memory bound.

    ``accumulate_resident=True`` models consumers that keep every yielded
    tensor (resident grows by cumulative kept bytes). ``False`` models
    consumers whose destination memory is already allocated before the load
    (e.g. copying into preallocated model parameters): resident growth is 0
    and the plan degenerates to a uniform budget determined by the transient
    depth.

    ``transient_multiplier`` scales the per-buffer transient cost: how many
    times over the copier stages each in-flight chunk. Only the copier knows
    (its reader path decides), so callers pass
    ``CopierInterface.chunk_transient_multiplier(paths)``. Fixed overheads
    that do not scale with chunk size -- bounce-buffer pools, the O_DIRECT
    reader's thread pool (measured on GB10 unified memory: +~150 MB regardless
    of chunk size) -- are not modelled here and must be left outside the
    budget the caller passes.

    ``group_size`` is the number of files loaded concurrently, one per rank
    (``pg.size()`` under broadcast loading, 1 otherwise). Every rank ends up
    with every tensor of the group, so a whole group's kept bytes become
    resident together and each file in it is charged the group total rather
    than its own prefix -- otherwise the bound below under-counts by up to
    ``group_size - 1`` files whenever shard sizes are uneven.

    ``account_for_yield_clone`` reserves one chunk budget for a non-resident
    yielded tensor clone.

    Returns one budget per file; feed each to
    ``SafeTensorsMetadata.plan_chunks``. A budget >= the file's span yields a
    single whole-span chunk (no splitting). Raises ``BudgetInfeasibleError``
    at plan time when some file's largest tensor cannot fit.
    """
    if device_memory_budget <= 0:
        raise BudgetInfeasibleError(
            f"device_memory_budget={device_memory_budget} must be positive"
        )
    if depth < 1:
        raise ValueError(f"depth must be >= 1, got {depth}")
    if transient_multiplier < 1:
        raise ValueError(
            f"transient_multiplier must be >= 1, got {transient_multiplier}"
        )
    if group_size < 1:
        raise ValueError(f"group_size must be >= 1, got {group_size}")
    eff_depth = _effective_depth(depth, transient_multiplier, account_for_yield_clone)
    group_resident = _group_resident(stats, group_size)
    budgets = []
    for i, st in enumerate(stats):
        resident = group_resident[i] if accumulate_resident else 0
        b = (device_memory_budget - resident) // eff_depth
        if max_batch_bytes is not None:
            b = min(b, max_batch_bytes)
        if b < st.largest_tensor:
            if max_batch_bytes is not None and max_batch_bytes < st.largest_tensor:
                # The cap binds before the budget does. Blaming the budget here
                # sends the user to free device memory, which cannot help: a
                # tensor is the atomic load unit, so no depth and no budget
                # make it fit a chunk smaller than itself.
                raise BudgetInfeasibleError(
                    f"max_batch_bytes={max_batch_bytes} is smaller than the "
                    f"largest kept tensor of '{st.path}' "
                    f"({st.largest_tensor} bytes); a tensor is the atomic load "
                    f"unit, so no queue_size or budget can fit it. Raise "
                    f"max_batch_bytes to at least {st.largest_tensor}."
                )
            required = resident + eff_depth * st.largest_tensor
            # Say what to retry with rather than only that something is too
            # big: the fix is a number the caller cannot get from the message
            # otherwise, and it is exactly computable here.
            requested_qs = _queue_size_of_depth(depth, group_size)
            fitted = fit_queue_size(
                requested_qs,
                stats,
                device_memory_budget,
                max_batch_bytes=max_batch_bytes,
                accumulate_resident=accumulate_resident,
                transient_multiplier=transient_multiplier,
                group_size=group_size,
                account_for_yield_clone=account_for_yield_clone,
            )
            if fitted is not None and fitted < requested_qs:
                advice = (
                    f"Retry with queue_size <= {fitted} (currently "
                    f"{requested_qs}), free device memory, or pass a larger "
                    f"explicit budget."
                )
            else:
                serial = (
                    resident
                    + _effective_depth(
                        load_depth(-1, group_size),
                        transient_multiplier,
                        account_for_yield_clone,
                    )
                    * st.largest_tensor
                )
                advice = (
                    f"No queue_size fits: even a fully serial load "
                    f"(queue_size=-1) needs >= {serial} bytes for this file. "
                    f"Free device memory or pass a larger explicit budget."
                )
            raise BudgetInfeasibleError(
                f"Model does not fit device_memory_budget: loading '{st.path}' "
                f"needs >= {required} bytes ({resident} resident + {eff_depth} x "
                f"{st.largest_tensor} transient), budget is {device_memory_budget}. "
                + advice
            )
        budgets.append(b)
    return budgets


def fit_queue_size(
    requested: int,
    stats: List[FileWeightStats],
    device_memory_budget: int,
    max_batch_bytes: Optional[int] = None,
    accumulate_resident: bool = True,
    transient_multiplier: int = 1,
    group_size: int = 1,
    account_for_yield_clone: bool = False,
) -> Optional[int]:
    """The deepest queue size ``<= requested`` whose plan fits the budget.

    Returns *requested* unchanged when it already fits, a smaller queue size
    when the budget only admits a shallower pipeline, and ``None`` when no
    queue size fits -- some file's largest tensor overflows the budget even
    loaded serially, so only freeing device memory (or a larger budget) helps.

    This is the exact inverse of the bound ``plan_file_budgets`` enforces, not
    a search: a plan with the returned queue size is guaranteed not to raise,
    and, whenever the result is smaller than *requested*, a plan one deeper
    than the result is guaranteed to raise.
    Every input comes from safetensors headers, so no trial load is needed --
    which is the point, since a trial load costs the whole model's bytes and
    a warm page cache would make repeated trials incomparable anyway.

    Callers pass the same arguments they would pass to ``plan_file_budgets``,
    minus ``depth``: that is what is being solved for. Under broadcast loading
    every rank must pass the same budget (see the module docstring), which then
    makes the clamp identical across ranks and keeps the plans in lockstep.
    """
    if transient_multiplier < 1:
        raise ValueError(
            f"transient_multiplier must be >= 1, got {transient_multiplier}"
        )
    if group_size < 1:
        raise ValueError(f"group_size must be >= 1, got {group_size}")
    if device_memory_budget <= 0:
        return None
    group_resident = _group_resident(stats, group_size)
    # Largest eff_depth every file can still afford. The plan's own test,
    # (budget - resident) // eff_depth >= largest, is exactly
    # budget - resident >= largest * eff_depth for eff_depth >= 1, so the
    # constraint inverts with no floor-division slack in either direction.
    cap = None  # None: no kept tensor constrains the depth
    for i, st in enumerate(stats):
        if st.largest_tensor <= 0:
            continue  # nothing kept in this file: no chunk, no constraint
        if max_batch_bytes is not None and max_batch_bytes < st.largest_tensor:
            return None  # the cap binds before the budget; depth cannot help
        headroom = device_memory_budget - (
            group_resident[i] if accumulate_resident else 0
        )
        if headroom < st.largest_tensor:
            return None
        limit = headroom // st.largest_tensor
        cap = limit if cap is None else min(cap, limit)
    if cap is None:
        return requested
    # eff_depth = load_depth * multiplier + clone <= cap, solved for the
    # pipeline_depth term, then mapped back through load_depth's inverse.
    room = cap - int(account_for_yield_clone)
    if room < transient_multiplier:
        return None  # not even one live buffer fits alongside the clone
    max_base = room // transient_multiplier - (1 if group_size > 1 else 0)
    if max_base < 1:
        return None
    return min(requested, -1 if max_base == 1 else max_base - 2)
