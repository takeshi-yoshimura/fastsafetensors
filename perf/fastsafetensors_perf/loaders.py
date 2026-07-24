"""Benchmark modes and consumers.

Torch, vLLM, and fastsafetensors are imported lazily inside the functions so
this module imports on a CPU-only host; only actually *running* a loader needs
them. Each mode produces a ``(name, tensor)`` iterator; each consumer drains it
and records timing/byte/count metrics.

Design rules from the roadmap:
* ``vllm`` mode calls the *installed* iterator, never a local reimplementation.
* Every consumer drains the full iterator, synchronizes CUDA before the final
  timestamp, and reports tensor count + logical bytes for verification.
"""

from __future__ import annotations

import hashlib
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple


def _name_hash(name: str) -> int:
    """Stable 64-bit hash of a tensor name for an order-independent digest."""
    return int.from_bytes(hashlib.blake2b(name.encode("utf-8"), digest_size=8).digest(), "big")

MODE_VLLM = "vllm"
MODE_PARALLEL = "parallel"
MODE_SAFETENSORS = "safetensors"
MODE_VLLM_MODEL_LOAD = "vllm-model-load"

CONSUMER_ITERATE = "iterate"
CONSUMER_COPY = "copy"


@dataclass
class LoaderOptions:
    """Mode/consumer configuration and direct-ParallelLoader tuning."""

    mode: str = MODE_PARALLEL
    consumer: str = CONSUMER_ITERATE
    queue_size: int = 0
    nogds: bool = False
    # direct-ParallelLoader tuning (mode=parallel)
    max_threads: int = 16
    bbuf_size_kb: int = 16 * 1024
    max_batch_bytes: int = 0  # 0 -> loader default
    use_shm: bool = False
    set_numa: bool = True
    extra_env: Dict[str, str] = field(default_factory=dict)

    def tuning_key(self) -> str:
        """Canonical identity string for the direct-loader knobs.

        Only meaningful (and only populated) for mode=parallel; other modes leave
        this empty so they don't fragment into per-tuning baseline series.
        """
        if self.mode != MODE_PARALLEL:
            return ""
        return (
            f"threads={self.max_threads},bbuf_kb={self.bbuf_size_kb},"
            f"max_batch={self.max_batch_bytes},shm={int(self.use_shm)},"
            f"numa={int(self.set_numa)}"
        )


@dataclass
class ConsumeResult:
    tensor_count: int = 0
    logical_bytes: int = 0
    time_to_first_seconds: float = 0.0
    consumer_copy_seconds: float = 0.0
    dtype_counts: Dict[str, int] = field(default_factory=dict)
    # XOR of per-name hashes: order-independent digest of the yielded key set,
    # used to validate the delivered tensors against the header inventory
    # without storing every name in the result record.
    names_digest: int = 0
    requested_backend: str = ""
    effective_backend: str = ""
    fallback: bool = False


def names_digest(names: Iterator[str]) -> int:
    """The digest that a correct run's yielded key set must match."""
    d = 0
    for n in names:
        d ^= _name_hash(n)
    return d


def _reset_gds_fallback_flag() -> bool:
    """Reset fastsafetensors' one-shot GDS-fallback flag so a fallback during the
    next run is observable. Returns True if the internal flag is reachable.

    This reads a private module global; guarded so a fastsafetensors change only
    disables fallback *detection*, never the benchmark.
    """
    try:
        from fastsafetensors.copier import gds

        gds._warned_gds_fallback = False
        return True
    except Exception:
        return False


def _gds_fell_back() -> bool:
    try:
        from fastsafetensors.copier import gds

        return bool(gds._warned_gds_fallback)
    except Exception:
        return False


def _effective_backend(options: LoaderOptions, world_size: int) -> str:
    if options.mode == MODE_SAFETENSORS:
        return "mmap"
    if options.nogds:
        return "nogds"
    if options.mode == MODE_VLLM:
        # vLLM forces nogds when TP>1 to avoid cuFileDriverOpen() side effects.
        return "nogds" if world_size > 1 else "gds"
    return "gds"


# --- iterators --------------------------------------------------------------


def _vllm_iterator(files: List[str], options: LoaderOptions):
    """The installed vLLM fastsafetensors iterator. No local reimplementation."""
    # VLLM_FASTSAFETENSORS_QUEUE_SIZE is read from the environment at iteration
    # time; set it before we start iterating.
    os.environ["VLLM_FASTSAFETENSORS_QUEUE_SIZE"] = str(options.queue_size)
    from vllm.model_executor.model_loader.weight_utils import (
        fastsafetensors_weights_iterator,
    )

    return fastsafetensors_weights_iterator(files, use_tqdm_on_load=False)


def _parallel_iterator(files: List[str], device: str, pg, options: LoaderOptions):
    from fastsafetensors.parallel_loader import ParallelLoader

    kwargs: Dict[str, Any] = dict(
        pg=pg,
        hf_weights_files=files,
        queue_size=options.queue_size,
        use_tqdm_on_load=False,
        device=device,
        nogds=options.nogds,
        max_threads=options.max_threads,
        bbuf_size_kb=options.bbuf_size_kb,
        set_numa=options.set_numa,
    )
    if options.max_batch_bytes > 0:
        kwargs["max_batch_bytes"] = options.max_batch_bytes
    loader = ParallelLoader(**kwargs)
    return loader


def _safetensors_iterator(files: List[str], device: str):
    """Stock safetensors mmap→device path."""
    from safetensors import safe_open

    def gen():
        for path in files:
            with safe_open(path, framework="pt", device=device) as f:
                for key in f.keys():
                    yield key, f.get_tensor(key)

    return gen()


# --- consumer ---------------------------------------------------------------


def _synchronize(device: str) -> None:
    try:
        import torch

        if str(device).startswith("cuda") and torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        pass


def _consume(iterator: Iterator[Tuple[str, Any]], options: LoaderOptions,
             device: str, start_time: float) -> ConsumeResult:
    """Drain the iterator, optionally copying each tensor, with sync at the end."""
    result = ConsumeResult()
    first_seen = False
    copy_seconds = 0.0
    do_copy = options.consumer == CONSUMER_COPY

    for name, tensor in iterator:
        if not first_seen:
            result.time_to_first_seconds = time.perf_counter() - start_time
            first_seen = True
        result.tensor_count += 1
        result.names_digest ^= _name_hash(name)
        try:
            numel = tensor.numel()
            elt = tensor.element_size()
            result.logical_bytes += numel * elt
            dt = str(tensor.dtype).replace("torch.", "")
            result.dtype_counts[dt] = result.dtype_counts.get(dt, 0) + 1
        except Exception:
            pass

        if do_copy:
            t0 = time.perf_counter()
            try:
                # Device-to-device copy: proxy for load_weights() write pressure.
                _ = tensor.clone()
            except Exception:
                pass
            copy_seconds += time.perf_counter() - t0

    # Synchronize before the caller takes the final timestamp so async H2D/D2D
    # work is included in wall time.
    _synchronize(device)
    result.consumer_copy_seconds = copy_seconds
    return result


# --- top-level entry --------------------------------------------------------


def run_consume(files: List[str], device: str, pg, options: LoaderOptions,
                start_time: float, world_size: int = 1) -> ConsumeResult:
    """Build the mode's iterator, drain it with the chosen consumer, and clean up.

    Cleanup runs on normal completion *and* on iterator error/timeout: the
    ParallelLoader is always closed.
    """
    requested = "nogds" if options.nogds else ("mmap" if options.mode == MODE_SAFETENSORS else "gds")
    effective = _effective_backend(options, world_size)

    # If a GDS attempt is expected, arm fallback detection so the *effective*
    # backend recorded in the identity reflects what actually ran. Resetting the
    # one-shot flag re-enables fastsafetensors' fallback warning every repetition,
    # so mute that logger for the duration (the flag is still set on fallback).
    detect_fallback = effective == "gds"
    muted_logger = None
    prev_level = 0
    if detect_fallback:
        detect_fallback = _reset_gds_fallback_flag()
        if detect_fallback:
            import logging

            muted_logger = logging.getLogger("fastsafetensors.copier.gds")
            prev_level = muted_logger.level
            muted_logger.setLevel(logging.ERROR)

    loader = None
    try:
        if options.mode == MODE_VLLM:
            iterator = _vllm_iterator(files, options)
        elif options.mode == MODE_PARALLEL:
            loader = _parallel_iterator(files, device, pg, options)
            iterator = loader.iterate_weights()
        elif options.mode == MODE_SAFETENSORS:
            iterator = _safetensors_iterator(files, device)
        elif options.mode == MODE_VLLM_MODEL_LOAD:
            # Optional release/weekly canary: an in-process vLLM model load with
            # load_format="fastsafetensors". It does not fit the tensor-iterator
            # measurement path (it builds a full model, not a stream of tensors),
            # so it is handled by a dedicated runner rather than run_consume.
            raise NotImplementedError(
                "mode 'vllm-model-load' is a release/weekly canary and is not run "
                "from run_consume; see run_vllm_model_load()"
            )
        else:
            raise ValueError(f"unsupported mode for run_consume: {options.mode}")

        result = _consume(iterator, options, device, start_time)
    finally:
        # Close the direct ParallelLoader; the vLLM iterator closes its own.
        if loader is not None:
            try:
                loader.close()
            except Exception:
                pass
        if muted_logger is not None:
            muted_logger.setLevel(prev_level)

    if detect_fallback and _gds_fell_back():
        effective = "nogds"

    result.requested_backend = requested
    result.effective_backend = effective
    result.fallback = requested != effective
    return result
