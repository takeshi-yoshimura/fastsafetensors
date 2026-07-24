"""Command-line interface: inspect / run / matrix / compare.

Multi-GPU runs are delegated to the standard launcher
``torchrun --standalone --nproc-per-node=N -m fastsafetensors_perf.worker`` so
there is one launch path. CUDA_VISIBLE_DEVICES is respected and never rewritten.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from typing import Any, Dict, List, Optional

import typer

from . import compare as compare_mod
from . import report as report_mod
from .loaders import LoaderOptions
from .models import inspect_checkpoint
from .results import iter_records
from .worker import CACHE_COLD, RunConfig, run_case

app = typer.Typer(add_completion=False, help="fastsafetensors regression benchmark")


# --- model alias resolution -------------------------------------------------


def _load_model_map(models_file: Optional[str]) -> Dict[str, Dict[str, Any]]:
    if not models_file:
        return {}
    with open(models_file, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    return data.get("models", data)


def _resolve_model(alias_or_path: str, model_map: Dict[str, Dict[str, Any]],
                   model_root: Optional[str]) -> Dict[str, Any]:
    """Resolve an alias (or literal path) to {path, alias, revision}."""
    if alias_or_path in model_map:
        entry = model_map[alias_or_path]
        path = entry["path"] if isinstance(entry, dict) else str(entry)
        revision = entry.get("revision", "main") if isinstance(entry, dict) else "main"
        alias = alias_or_path
    else:
        path = alias_or_path
        alias = os.path.basename(alias_or_path.rstrip("/"))
        revision = "main"
    if model_root and not os.path.isabs(path):
        path = os.path.join(model_root, path)
    return {"path": path, "alias": alias, "revision": revision}


# --- inspect ----------------------------------------------------------------


@app.command()
def inspect(model: str = typer.Argument(..., help="model directory or .safetensors file"),
            json_out: bool = typer.Option(False, "--json", help="emit JSON")):
    """Print a deterministic shard/tensor/dtype/byte inventory (headers only)."""
    inv = inspect_checkpoint(model)
    summary = inv.summary_dict()
    if json_out:
        typer.echo(json.dumps(summary, indent=2, sort_keys=True))
        return
    typer.echo(f"model: {summary['model_dir']}")
    typer.echo(f"fingerprint: {summary['fingerprint']}")
    typer.echo(f"shards: {summary['shard_count']}  tensors: {summary['tensor_count']}")
    typer.echo(f"source bytes: {summary['source_checkpoint_bytes'] / 1e9:.3f} GB")
    typer.echo(f"logical bytes: {summary['logical_bytes'] / 1e9:.3f} GB  "
               f"storage bytes: {summary['storage_bytes'] / 1e9:.3f} GB")
    typer.echo(f"max shard: {summary['max_shard_bytes'] / 1e9:.3f} GB")
    typer.echo("dtype histogram:")
    for dt, e in sorted(summary["dtype_histogram"].items()):
        typer.echo(f"  {dt:>10}  count={e['count']:<6} "
                   f"logical={e['logical_bytes'] / 1e6:.1f} MB "
                   f"storage={e['storage_bytes'] / 1e6:.1f} MB")


# --- run --------------------------------------------------------------------


def _launch_torchrun(model_path: str, opts: LoaderOptions, world_size: int,
                     cache: str, repeat: int, warmup: int, timeout: float,
                     output: str, model_alias: str, model_revision: str,
                     hardware_profile: str, case_id: str) -> int:
    cmd = [
        "torchrun", "--standalone", f"--nproc-per-node={world_size}",
        "-m", "fastsafetensors_perf.worker", "run", model_path,
        "--mode", opts.mode, "--consumer", opts.consumer,
        "--world-size", str(world_size), "--queue-size", str(opts.queue_size),
        "--max-threads", str(opts.max_threads), "--bbuf-size-kb", str(opts.bbuf_size_kb),
        "--max-batch-bytes", str(opts.max_batch_bytes),
        "--cache", cache, "--repeat", str(repeat), "--warmup", str(warmup),
        "--timeout", str(timeout), "--model-alias", model_alias,
        "--model-revision", model_revision, "--hardware-profile", hardware_profile,
    ]
    if opts.nogds:
        cmd.append("--nogds")
    if output:
        cmd += ["--output", output]
    if case_id:
        cmd += ["--case-id", case_id]
    env = dict(os.environ)
    for k, v in opts.extra_env.items():
        env[k] = v
    return subprocess.call(cmd, env=env)


@app.command()
def run(model: str = typer.Argument(...),
        mode: str = typer.Option("parallel"),
        consumer: str = typer.Option("iterate"),
        world_size: int = typer.Option(1, "--world-size"),
        queue_size: int = typer.Option(0, "--queue-size"),
        nogds: bool = typer.Option(False, "--nogds"),
        max_threads: int = typer.Option(16, "--max-threads"),
        bbuf_size_kb: int = typer.Option(16 * 1024, "--bbuf-size-kb"),
        max_batch_bytes: int = typer.Option(0, "--max-batch-bytes"),
        cache: str = typer.Option(CACHE_COLD, "--cache"),
        repeat: int = typer.Option(5, "--repeat"),
        warmup: int = typer.Option(0, "--warmup"),
        timeout: float = typer.Option(600.0, "--timeout"),
        output: str = typer.Option("", "--output"),
        models: Optional[str] = typer.Option(None, "--models", help="model alias map json"),
        model_root: Optional[str] = typer.Option(None, "--model-root"),
        hardware_profile: str = typer.Option("unknown", "--hardware-profile")):
    """Run one benchmark case. world_size>1 delegates to torchrun."""
    resolved = _resolve_model(model, _load_model_map(models), model_root)
    opts = LoaderOptions(
        mode=mode, consumer=consumer, queue_size=queue_size, nogds=nogds,
        max_threads=max_threads, bbuf_size_kb=bbuf_size_kb,
        max_batch_bytes=max_batch_bytes,
    )
    if world_size > 1:
        code = _launch_torchrun(
            resolved["path"], opts, world_size, cache, repeat, warmup, timeout,
            output, resolved["alias"], resolved["revision"], hardware_profile, "")
        raise typer.Exit(code)

    config = RunConfig(
        model_path=resolved["path"], model_alias=resolved["alias"],
        model_revision=resolved["revision"], hardware_profile=hardware_profile,
        world_size=1, repeat=repeat, warmup=warmup, cache_policy=cache,
        timeout_seconds=timeout, output=output, options=opts,
    )
    raise typer.Exit(run_case(config))


# --- matrix -----------------------------------------------------------------


@app.command()
def matrix(config_file: str = typer.Argument(..., help="host matrix json"),
           models: Optional[str] = typer.Option(None, "--models"),
           model_root: Optional[str] = typer.Option(None, "--model-root"),
           output_dir: str = typer.Option("results", "--output-dir")):
    """Run every case listed in a host matrix config.

    Cartesian expansion is intentionally *not* done here: each case is listed
    explicitly (optionally inheriting from a common defaults block).
    """
    with open(config_file, "r", encoding="utf-8") as fh:
        cfg = json.load(fh)
    defaults = cfg.get("defaults", {})
    hardware_profile = cfg.get("hardware_profile", "unknown")
    model_map = _load_model_map(models)
    os.makedirs(output_dir, exist_ok=True)

    overall = 0
    for case in cfg.get("cases", []):
        merged = {**defaults, **case}
        resolved = _resolve_model(merged["model"], model_map, model_root)
        opts = LoaderOptions(
            mode=merged.get("mode", "parallel"),
            consumer=merged.get("consumer", "iterate"),
            queue_size=merged.get("queue_size", 0),
            nogds=merged.get("nogds", False),
            max_threads=merged.get("max_threads", 16),
            bbuf_size_kb=merged.get("bbuf_size_kb", 16 * 1024),
            max_batch_bytes=merged.get("max_batch_bytes", 0),
            extra_env=merged.get("env", {}),
        )
        world_size = merged.get("world_size", 1)
        case_id = merged.get("case_id", "")
        out = os.path.join(output_dir, f"{merged.get('model')}.jsonl")
        typer.echo(f"--- case: {case_id or merged.get('model')} "
                   f"(mode={opts.mode} ws={world_size} q={opts.queue_size}) ---")

        if world_size > 1:
            code = _launch_torchrun(
                resolved["path"], opts, world_size, merged.get("cache", CACHE_COLD),
                merged.get("repeat", 5), merged.get("warmup", 0),
                merged.get("timeout", 600.0), out, resolved["alias"],
                resolved["revision"], hardware_profile, case_id)
        else:
            rc = RunConfig(
                model_path=resolved["path"], model_alias=resolved["alias"],
                model_revision=resolved["revision"], hardware_profile=hardware_profile,
                case_id=case_id, world_size=1, repeat=merged.get("repeat", 5),
                warmup=merged.get("warmup", 0), cache_policy=merged.get("cache", CACHE_COLD),
                timeout_seconds=merged.get("timeout", 600.0), output=out, options=opts,
            )
            for k, v in opts.extra_env.items():
                os.environ[k] = str(v)
            code = run_case(rc)
        overall = overall or code
    raise typer.Exit(overall)


# --- compare ----------------------------------------------------------------


@app.command()
def compare(baseline: str = typer.Argument(...),
            candidate: str = typer.Argument(...),
            fail_threshold: float = typer.Option(0.15, "--fail-threshold"),
            warn_threshold: float = typer.Option(0.10, "--warn-threshold"),
            allow_incompatible: bool = typer.Option(False, "--allow-incompatible")):
    """Compare candidate results against a baseline and gate via exit code."""
    base = list(iter_records(baseline))
    cand = list(iter_records(candidate))
    thresholds = compare_mod.Thresholds(time_fail=fail_threshold, time_warn=warn_threshold)
    report = compare_mod.compare(base, cand, thresholds,
                                 allow_incompatible=allow_incompatible)

    for r in report.results:
        typer.echo(f"[{r.outcome.name}] {r.label()}")
        for msg in r.messages:
            typer.echo(f"    - {msg}")
    if report.incompatible:
        typer.echo(f"incompatible / unmatched cases: {len(report.incompatible)}")
        for ident in report.incompatible[:10]:
            typer.echo(f"    ! {ident.get('model_alias', '?')} "
                       f"{ident.get('gpu_model', '?')} "
                       f"vllm={ident.get('vllm_series', '?')} "
                       f"fst={ident.get('fastsafetensors_series', '?')}")
    typer.echo(f"=> {report.exit_code.name} (exit {int(report.exit_code)})")
    raise typer.Exit(int(report.exit_code))


# --- report -----------------------------------------------------------------


@app.command()
def report(results: List[str] = typer.Argument(..., help="one or more result JSONL files"),
           group_by: str = typer.Option("mode", "--group-by",
                                         help="comma-separated identity fields, e.g. mode or world_size"),
           metric: str = typer.Option("delivery_gbps", "--metric",
                                       help="delivery_gbps|storage_gbps|wall_s|ttf_s|peak_cuda_gb|peak_rss_gb"),
           baseline_field: Optional[str] = typer.Option(None, "--baseline-field"),
           baseline_value: Optional[str] = typer.Option(None, "--baseline-value"),
           json_out: bool = typer.Option(False, "--json", help="emit chart-ready JSON")):
    """Cross-configuration performance comparison (non-gating).

    Unlike `compare`, this deliberately lines results up across an axis to show
    relative speed. Example (safetensors vs fastsafetensors):

        fastsafetensors-perf report results/*.jsonl --group-by mode \\
            --baseline-field mode --baseline-value safetensors
    """
    fields = [f.strip() for f in group_by.split(",") if f.strip()]
    aggs = report_mod.load_aggregates(results)
    rows = report_mod.build_report(aggs, fields, metric=metric,
                                   baseline_field=baseline_field,
                                   baseline_value=baseline_value)
    if json_out:
        typer.echo(json.dumps(report_mod.to_chart_data(rows), indent=2))
    else:
        typer.echo(report_mod.format_table(rows))


if __name__ == "__main__":
    app()
