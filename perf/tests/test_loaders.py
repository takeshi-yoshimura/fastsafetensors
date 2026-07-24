"""CPU-safe loader helpers (no torch/vllm import required)."""

from __future__ import annotations

from fastsafetensors_perf.loaders import (
    MODE_PARALLEL,
    MODE_SAFETENSORS,
    MODE_VLLM,
    LoaderOptions,
    names_digest,
)
from fastsafetensors_perf.loaders import _effective_backend


def test_names_digest_order_independent():
    a = names_digest(iter(["w1", "w2", "w3"]))
    b = names_digest(iter(["w3", "w1", "w2"]))
    assert a == b


def test_names_digest_detects_missing_or_extra_key():
    full = names_digest(iter(["w1", "w2", "w3"]))
    missing = names_digest(iter(["w1", "w2"]))
    extra = names_digest(iter(["w1", "w2", "w3", "w4"]))
    assert full != missing
    assert full != extra


def test_effective_backend():
    assert _effective_backend(LoaderOptions(mode=MODE_SAFETENSORS), 1) == "mmap"
    assert _effective_backend(LoaderOptions(mode=MODE_PARALLEL, nogds=True), 1) == "nogds"
    assert _effective_backend(LoaderOptions(mode=MODE_PARALLEL, nogds=False), 1) == "gds"
    # vLLM forces nogds when TP > 1.
    assert _effective_backend(LoaderOptions(mode=MODE_VLLM, nogds=False), 4) == "nogds"
    assert _effective_backend(LoaderOptions(mode=MODE_VLLM, nogds=False), 1) == "gds"
