"""Configuration, loader-option identity, model resolution, and shipped configs."""

from __future__ import annotations

import glob
import json
import os

from fastsafetensors_perf.cli import _load_model_map, _resolve_model
from fastsafetensors_perf.loaders import (
    CONSUMER_ITERATE,
    MODE_PARALLEL,
    MODE_SAFETENSORS,
    MODE_VLLM,
    MODE_VLLM_MODEL_LOAD,
    LoaderOptions,
)

CONFIG_DIR = os.path.join(os.path.dirname(__file__), "..", "configs")
KNOWN_MODES = {MODE_VLLM, MODE_PARALLEL, MODE_SAFETENSORS, MODE_VLLM_MODEL_LOAD}


def test_tuning_key_only_for_parallel():
    parallel = LoaderOptions(mode=MODE_PARALLEL, max_threads=8, bbuf_size_kb=1024)
    assert "threads=8" in parallel.tuning_key()
    # Other modes must not fragment into per-tuning baseline series.
    assert LoaderOptions(mode=MODE_VLLM, max_threads=8).tuning_key() == ""
    assert LoaderOptions(mode=MODE_SAFETENSORS, max_threads=8).tuning_key() == ""


def test_model_resolution_alias_and_root():
    model_map = {"qwen3-8b": {"path": "Qwen3-8B", "revision": "abc123"}}
    r = _resolve_model("qwen3-8b", model_map, "/models")
    assert r == {"path": "/models/Qwen3-8B", "alias": "qwen3-8b", "revision": "abc123"}


def test_model_resolution_literal_path():
    r = _resolve_model("/data/Foo", {}, None)
    assert r["path"] == "/data/Foo"
    assert r["alias"] == "Foo"
    assert r["revision"] == "main"


def test_model_map_supports_flat_and_nested():
    assert _load_model_map(None) == {}


def test_shipped_configs_are_valid():
    configs = glob.glob(os.path.join(CONFIG_DIR, "*.json"))
    assert configs, "no config files found"
    for path in configs:
        with open(path) as fh:
            cfg = json.load(fh)
        base = os.path.basename(path)
        if base in ("common.json", "models.example.json"):
            continue
        assert "hardware_profile" in cfg, f"{base} missing hardware_profile"
        for case in cfg.get("cases", []):
            assert "model" in case, f"{base}: case missing model"
            mode = {**cfg.get("defaults", {}), **case}.get("mode", MODE_PARALLEL)
            assert mode in KNOWN_MODES, f"{base}: unknown mode {mode}"


def test_example_model_map_covers_matrix_models():
    with open(os.path.join(CONFIG_DIR, "models.example.json")) as fh:
        model_map = json.load(fh)["models"]
    for path in glob.glob(os.path.join(CONFIG_DIR, "*.json")):
        base = os.path.basename(path)
        if base in ("common.json", "models.example.json"):
            continue
        with open(path) as fh:
            cfg = json.load(fh)
        for case in cfg.get("cases", []):
            assert case["model"] in model_map, (
                f"{base}: model alias {case['model']!r} not in models.example.json"
            )
