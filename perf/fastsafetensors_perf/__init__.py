"""fastsafetensors-perf: regression benchmark for fastsafetensors ParallelLoader.

The package is intentionally split so that discovery, schema, and comparison
logic import cleanly on a CPU-only host (no torch/vllm/CUDA required). Only the
loader execution paths (``loaders`` and ``worker``) pull in torch and, for the
``vllm`` modes, vLLM itself.
"""

from .results import SCHEMA_VERSION

__all__ = ["SCHEMA_VERSION"]
