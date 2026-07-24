"""Benchmark run engine: single-GPU and torchrun multi-GPU.

Responsibilities:
* process-group / CUDA init (single process, or torchrun with RANK/WORLD_SIZE);
* model discovery and per-rank environment/identity;
* cold/warm cache protocol with barriers;
* the timed region (start after init+discovery+cache+barrier; end after a local
  CUDA sync, with a barrier so the aggregate can use the slowest rank);
* per-repetition timeout, error propagation, and loader cleanup; and
* gathering rank metrics to rank 0 and writing rank + aggregate JSONL records.

Runnable as ``python -m fastsafetensors_perf.worker run <model> [opts]`` and, for
multi-GPU, ``torchrun --standalone --nproc-per-node=N -m
fastsafetensors_perf.worker run ...``.
"""

from __future__ import annotations

import os
import resource
import signal
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from . import environment, metrics as metrics_mod
from .loaders import LoaderOptions, run_consume
from .models import inspect_checkpoint
from .results import (
    Identity,
    RankMetrics,
    make_aggregate_record,
    make_rank_record,
    write_records,
)

CACHE_COLD = "cold"
CACHE_WARM = "warm"


@dataclass
class RunConfig:
    model_path: str
    model_alias: str = ""
    model_revision: str = "main"
    hardware_profile: str = "unknown"
    case_id: str = ""
    world_size: int = 1
    repeat: int = 5
    warmup: int = 0
    cache_policy: str = CACHE_COLD
    timeout_seconds: float = 600.0
    output: str = ""
    options: LoaderOptions = field(default_factory=LoaderOptions)

    def resolved_case_id(self) -> str:
        if self.case_id:
            return self.case_id
        o = self.options
        return (f"{self.model_alias or os.path.basename(self.model_path.rstrip('/'))}"
                f"/{o.mode}/{o.consumer}/ws{self.world_size}/q{o.queue_size}"
                f"/{'nogds' if o.nogds else 'auto'}/{self.cache_policy}")


# --- distributed helpers ----------------------------------------------------


@dataclass
class Dist:
    rank: int = 0
    world_size: int = 1
    local_rank: int = 0
    distributed: bool = False
    pg: Any = None

    def barrier(self) -> None:
        if self.distributed:
            import torch.distributed as dist

            dist.barrier()

    def gather_objects(self, obj: Any) -> List[Any]:
        if not self.distributed:
            return [obj]
        import torch.distributed as dist

        out: List[Any] = [None] * self.world_size
        dist.all_gather_object(out, obj)
        return out


def _init_dist(requested_world_size: int) -> Dist:
    """Initialize distributed state from the torchrun environment, if present."""
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    if world_size <= 1:
        return Dist(rank=0, world_size=1, local_rank=local_rank, distributed=False)

    import torch
    import torch.distributed as dist

    backend = "nccl" if torch.cuda.is_available() else "gloo"
    if not dist.is_initialized():
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            # Bind the collective device explicitly: without device_id NCCL
            # guesses the device from the global rank, which can deadlock
            # barrier()/all_gather_object() when the mapping is not obvious.
            dist.init_process_group(
                backend=backend, device_id=torch.device(f"cuda:{local_rank}")
            )
        else:
            dist.init_process_group(backend=backend)
    return Dist(rank=rank, world_size=world_size, local_rank=local_rank,
                distributed=True, pg=dist.group.WORLD)


def _device_str() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            return f"cuda:{torch.cuda.current_device()}"
    except Exception:
        pass
    return "cpu"


# --- cache protocol ---------------------------------------------------------


def _evict_page_cache(files: List[str]) -> bool:
    """posix_fadvise(POSIX_FADV_DONTNEED) each file. Returns availability.

    Never touches drop_caches or unrelated cache. On platforms without
    posix_fadvise (or on failure) returns False so the record can note that cold
    eviction was unavailable/ineffective.
    """
    fadvise = getattr(os, "posix_fadvise", None)
    dontneed = getattr(os, "POSIX_FADV_DONTNEED", None)
    if fadvise is None or dontneed is None:
        return False
    ok = True
    for path in files:
        try:
            fd = os.open(path, os.O_RDONLY)
            try:
                os.fsync(fd)
            except Exception:
                pass
            try:
                fadvise(fd, 0, 0, dontneed)
            finally:
                os.close(fd)
        except Exception:
            ok = False
    return ok


# --- memory helpers ---------------------------------------------------------


def _reset_peak_memory(device: str) -> None:
    try:
        import torch

        if device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass


def _peak_cuda(device: str) -> Tuple[int, int]:
    try:
        import torch

        if device.startswith("cuda") and torch.cuda.is_available():
            return (int(torch.cuda.max_memory_allocated()),
                    int(torch.cuda.max_memory_reserved()))
    except Exception:
        pass
    return (0, 0)


def _peak_rss_bytes() -> int:
    # ru_maxrss is KiB on Linux, bytes on macOS. Assume Linux (KiB) here.
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024


# --- one repetition ---------------------------------------------------------


class _Timeout(Exception):
    pass


def _run_once_with_timeout(files: List[str], device: str, pg, options: LoaderOptions,
                           start_time: float, world_size: int,
                           timeout_seconds: float):
    """Run one consume inline, bounded by ``timeout_seconds`` via SIGALRM.

    Returns (ConsumeResult|None, status, error). The consume runs on the calling
    (main) thread so distributed NCCL collectives issued by the loader are never
    moved off-thread -- doing so deadlocks against the main thread's barriers.
    The timeout is a SIGALRM that raises in the consume loop; a rank that times
    out cannot safely resume, so the caller aborts its remaining repetitions.
    """
    use_alarm = (
        timeout_seconds > 0
        and hasattr(signal, "SIGALRM")
        and threading.current_thread() is threading.main_thread()
    )

    def _handler(signum, frame):
        raise _Timeout()

    prev_handler = None
    if use_alarm:
        prev_handler = signal.signal(signal.SIGALRM, _handler)
        signal.setitimer(signal.ITIMER_REAL, max(1.0, timeout_seconds))
    try:
        result = run_consume(files, device, pg, options, start_time, world_size)
        return result, "ok", ""
    except _Timeout:
        return None, "timeout", f"exceeded {timeout_seconds}s"
    except BaseException as exc:  # noqa: BLE001 -- propagate as a case result
        return None, "error", f"{type(exc).__name__}: {exc}"
    finally:
        if use_alarm:
            signal.setitimer(signal.ITIMER_REAL, 0)
            if prev_handler is not None:
                signal.signal(signal.SIGALRM, prev_handler)


def _repetition(rep: int, dist: Dist, files: List[str], device: str,
                config: RunConfig,
                expected_names_digest: Optional[int] = None) -> RankMetrics:
    options = config.options
    # cold: evict before every recorded repetition; barrier so all ranks start
    # from the same cache state.
    if config.cache_policy == CACHE_COLD:
        if dist.rank == 0:
            _evict_page_cache(files)
        dist.barrier()

    _reset_peak_memory(device)
    # Sync + barrier so the timer starts clean and simultaneously across ranks.
    from_time = time.perf_counter  # local alias
    try:
        import torch

        if device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        pass
    dist.barrier()

    start = from_time()
    result, status, error = _run_once_with_timeout(
        files, device, dist.pg, options, start, dist.world_size, config.timeout_seconds
    )
    # Local end timestamp (result already synchronized CUDA). Per-rank wall is
    # this rank's own time; the aggregate takes the slowest rank.
    end = from_time()
    dist.barrier()

    m = RankMetrics(rank=dist.rank, repetition=rep, status=status, ok=(status == "ok"),
                    error=error)
    if result is None:
        m.wall_seconds = end - start
        return m

    peak_alloc, peak_reserved = _peak_cuda(device)
    m.wall_seconds = end - start
    m.time_to_first_seconds = result.time_to_first_seconds
    m.consumer_copy_seconds = result.consumer_copy_seconds
    m.tensor_count = result.tensor_count
    m.logical_bytes = result.logical_bytes

    # Correctness: the delivered key set must match the header inventory. Without
    # a tensor_filter every rank receives every tensor (broadcast), so this holds
    # for all ranks. A mismatch turns an otherwise-ok repetition into an error.
    if expected_names_digest is not None and result.names_digest != expected_names_digest:
        m.status = "error"
        m.ok = False
        m.error = "delivered tensor key set does not match checkpoint inventory"
    m.peak_cuda_allocated_bytes = peak_alloc
    m.peak_cuda_reserved_bytes = peak_reserved
    m.host_peak_rss_bytes = _peak_rss_bytes()
    m.requested_backend = result.requested_backend
    m.effective_backend = result.effective_backend
    m.fallback = result.fallback
    return m


# --- top-level case ---------------------------------------------------------


def run_case(config: RunConfig) -> int:
    """Run a full case (all repetitions), write records, return a process exit code."""
    dist = _init_dist(config.world_size)
    device = _device_str()

    # Discovery + header inventory (cheap; each rank reads locally).
    inv = inspect_checkpoint(config.model_path)
    files = inv.files

    env = environment.collect(model_path=config.model_path)
    ident = Identity(
        model_alias=config.model_alias or os.path.basename(config.model_path.rstrip("/")),
        model_revision=config.model_revision,
        model_fingerprint=inv.fingerprint(),
        mode=config.options.mode,
        consumer=config.options.consumer,
        queue_size=config.options.queue_size,
        backend="",  # filled from effective backend after first repetition
        cache_policy=config.cache_policy,
        tuning_key=config.options.tuning_key(),
    )
    env.fill_identity(ident, config.hardware_profile, dist.world_size)

    case_block = {
        "case_id": config.resolved_case_id(),
        "repeat": config.repeat,
        "warmup": config.warmup,
        "timeout_seconds": config.timeout_seconds,
        "loader_options": {
            "mode": config.options.mode,
            "consumer": config.options.consumer,
            "queue_size": config.options.queue_size,
            "nogds": config.options.nogds,
            "tuning_key": config.options.tuning_key(),
        },
        "checkpoint": inv.summary_dict(),
    }

    cache_eviction_available = True
    # Warm: one unrecorded full warmup before recorded repetitions.
    if config.cache_policy == CACHE_WARM:
        try:
            _reset_peak_memory(device)
            dist.barrier()
            run_consume(files, device, dist.pg, config.options, time.perf_counter(),
                        dist.world_size)
        except Exception:
            pass
        dist.barrier()
    elif config.cache_policy == CACHE_COLD:
        if dist.rank == 0:
            cache_eviction_available = _evict_page_cache(files)

    # Expected key-set digest from the header inventory, for correctness checks.
    from .loaders import names_digest as _names_digest_fn

    expected_digest = _names_digest_fn(
        t.name for shard in inv.shards for t in shard.tensors
    )

    # Recorded repetitions. Abort this rank's remaining reps after a timeout.
    my_metrics: List[RankMetrics] = []
    aborted = False
    for rep in range(config.repeat):
        if aborted:
            my_metrics.append(RankMetrics(rank=dist.rank, repetition=rep,
                                          status="timeout", ok=False,
                                          error="aborted after prior timeout"))
            continue
        m = _repetition(rep, dist, files, device, config,
                        expected_names_digest=expected_digest)
        if m.status == "timeout":
            aborted = True
        my_metrics.append(m)

    # Fill effective backend on the identity from the first successful rep.
    for m in my_metrics:
        if m.ok and m.effective_backend:
            ident.backend = m.effective_backend
            break
    if not ident.backend:
        ident.backend = "nogds" if config.options.nogds else "gds"

    # Gather all ranks' metrics to everyone; rank 0 writes.
    gathered: List[List[Dict[str, Any]]] = dist.gather_objects(
        [m.to_dict() for m in my_metrics]
    )

    exit_code = 0
    if dist.rank == 0:
        # Reorganize gathered[rank][rep] -> per-rep list of rank dicts.
        rank_dicts: List[Dict[str, Any]] = []
        by_rep: List[List[Dict[str, Any]]] = [[] for _ in range(config.repeat)]
        worst_status = "ok"
        for rank_list in gathered:
            for md in rank_list:
                rank_dicts.append(md)
                rep = md["repetition"]
                if 0 <= rep < config.repeat:
                    by_rep[rep].append(md)
                if md["status"] != "ok":
                    worst_status = md["status"]

        stats = metrics_mod.aggregate_repetitions(
            [rep for rep in by_rep if rep]
        )

        records: List[Dict[str, Any]] = [
            make_rank_record(ident, env.to_dict(), case_block,
                             _rank_metrics_from_dict(md))
            for md in rank_dicts
        ]
        records.append(make_aggregate_record(
            ident, env.to_dict(), case_block, stats,
            cache_eviction_available=cache_eviction_available,
            n_repetitions=config.repeat, n_ranks=dist.world_size,
            worst_status=worst_status,
        ))

        if config.output:
            os.makedirs(os.path.dirname(os.path.abspath(config.output)), exist_ok=True)
            write_records(config.output, records, append=True)

        _print_summary(config, ident, stats, worst_status)
        if worst_status != "ok":
            exit_code = 1

    _teardown(dist)
    return exit_code


def _rank_metrics_from_dict(md: Dict[str, Any]) -> RankMetrics:
    """Rebuild a RankMetrics from a gathered dict (drops derived-only keys)."""
    fields = {f.name for f in RankMetrics.__dataclass_fields__.values()}
    return RankMetrics(**{k: v for k, v in md.items() if k in fields})


def _teardown(dist: Dist) -> None:
    if dist.distributed:
        try:
            import torch.distributed as d

            d.barrier()
            d.destroy_process_group()
        except Exception:
            pass


def _print_summary(config: RunConfig, ident: Identity, stats: Dict[str, Any],
                   worst_status: str) -> None:
    wall = stats.get("wall_seconds", {})
    ttf = stats.get("time_to_first_seconds", {})
    thr = stats.get("delivery_throughput_bps", 0.0) / 1e9
    print(f"[fastsafetensors-perf] {config.resolved_case_id()}")
    print(f"  status={worst_status} backend={ident.backend} "
          f"world_size={ident.world_size}")
    print(f"  wall median={wall.get('median', 0):.4f}s p90={wall.get('p90', 0):.4f}s "
          f"cov={wall.get('cov', 0):.1%}")
    print(f"  ttf median={ttf.get('median', 0):.4f}s  delivery={thr:.2f} GB/s")


# --- module CLI (for torchrun) ---------------------------------------------


def _build_arg_parser():
    import argparse

    p = argparse.ArgumentParser(prog="fastsafetensors_perf.worker")
    sub = p.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    run.add_argument("model_path")
    run.add_argument("--model-alias", default="")
    run.add_argument("--model-revision", default="main")
    run.add_argument("--hardware-profile", default="unknown")
    run.add_argument("--case-id", default="")
    run.add_argument("--mode", default="parallel")
    run.add_argument("--consumer", default="iterate")
    run.add_argument("--world-size", type=int, default=1)
    run.add_argument("--queue-size", type=int, default=0)
    run.add_argument("--nogds", action="store_true")
    run.add_argument("--max-threads", type=int, default=16)
    run.add_argument("--bbuf-size-kb", type=int, default=16 * 1024)
    run.add_argument("--max-batch-bytes", type=int, default=0)
    run.add_argument("--cache", default=CACHE_COLD, choices=[CACHE_COLD, CACHE_WARM])
    run.add_argument("--repeat", type=int, default=5)
    run.add_argument("--warmup", type=int, default=0)
    run.add_argument("--timeout", type=float, default=600.0)
    run.add_argument("--output", default="")
    return p


def config_from_namespace(ns) -> RunConfig:
    options = LoaderOptions(
        mode=ns.mode, consumer=ns.consumer, queue_size=ns.queue_size, nogds=ns.nogds,
        max_threads=ns.max_threads, bbuf_size_kb=ns.bbuf_size_kb,
        max_batch_bytes=ns.max_batch_bytes,
    )
    world_size = int(os.environ.get("WORLD_SIZE", str(ns.world_size)))
    return RunConfig(
        model_path=ns.model_path, model_alias=ns.model_alias,
        model_revision=ns.model_revision, hardware_profile=ns.hardware_profile,
        case_id=ns.case_id, world_size=world_size, repeat=ns.repeat,
        warmup=ns.warmup, cache_policy=ns.cache, timeout_seconds=ns.timeout,
        output=ns.output, options=options,
    )


def main(argv: Optional[List[str]] = None) -> int:
    parser = _build_arg_parser()
    ns = parser.parse_args(argv)
    if ns.command == "run":
        return run_case(config_from_namespace(ns))
    parser.error(f"unknown command {ns.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
