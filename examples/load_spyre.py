#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Convert, load and verify Spyre-native checkpoints with bounded host scratch.

This loads weights, not a model: no spyre-inference, Transformers, or TP transform.
Tensors are released after each chunk; use ParallelLoader directly to retain them.
"""

import argparse
import json
import sys
import time

from fastsafetensors import ParallelLoader
from fastsafetensors.frameworks import get_framework_op
from fastsafetensors.native import checkpoint_paths, convert_to_spyre
from fastsafetensors.st_types import Device, DType


def load(args):
    import torch
    import torch_spyre

    ops = get_framework_op("pytorch").get_native_ops(Device.from_str(args.device))
    ops.load_library()
    reference_paths = {}
    if args.verify_source:
        from safetensors import safe_open

        for path in checkpoint_paths(args.verify_source):
            with safe_open(path, framework="pt", device="cpu") as source:
                for name in source.keys():
                    if name in reference_paths:
                        raise ValueError(f"duplicate reference tensor: {name}")
                    reference_paths[name] = path
    start = time.monotonic()
    count = nbytes = 0
    loader = ParallelLoader(
        None,
        checkpoint_paths(args.source),
        device=args.device,
        all_local=True,
        set_numa=False,
        accumulate_resident=False,
        max_batch_bytes=args.batch_mib << 20,
        queue_size=0,
        bbuf_size_kb=args.host_buffer_mib * 1024,
        max_threads=args.max_threads,
        use_tqdm_on_load=False,
    )
    seen = set()
    try:
        for name, tensor in loader.iterate_weights():
            count += 1
            seen.add(name)
            nbytes += ops.tensor_nbytes(tensor)
            if args.verify_source:
                with safe_open(
                    reference_paths[name], framework="pt", device="cpu"
                ) as source:
                    cpu = source.get_tensor(name).to(dtype=tensor.dtype)
                # Normal framework copy accounts for backend rounding.
                expected = cpu.to(args.device).cpu()
                torch.testing.assert_close(
                    tensor.cpu(), expected, rtol=0, atol=0, equal_nan=True, msg=name
                )
                del cpu, expected
            print(f"{count}: {name} {tuple(tensor.shape)} {tensor.dtype}", flush=True)
            del tensor
    finally:
        loader.close()
    if args.verify_source and seen != reference_paths.keys():
        raise AssertionError("native/reference tensor name sets differ")
    if any(
        name == "spyre_inference" or name.startswith("spyre_inference.")
        for name in sys.modules
    ):
        raise AssertionError("unexpected spyre-inference import")
    print(
        json.dumps(
            {
                "tensors": count,
                "native_bytes": nbytes,
                "host_capacity_bytes": args.host_buffer_mib << 20,
                "elapsed_s": time.monotonic() - start,
                "verified": bool(args.verify_source),
            }
        ),
        flush=True,
    )


def convert(args):
    start = time.monotonic()
    count = 0

    def progress(name, nbytes):
        nonlocal count
        count += 1
        print(f"{count}: {name} ({nbytes} native bytes)", flush=True)

    paths = convert_to_spyre(
        args.source,
        args.base,
        device=args.device,
        dtype=DType(args.dtype),
        num_shards=args.num_shards,
        fsync=args.fsync,
        progress=progress,
        workspace_bytes=args.workspace_bytes,
    )
    print(f"Converted {count} tensors in {time.monotonic() - start:.3f}s")
    for path in paths:
        print(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    convert_parser = commands.add_parser(
        "convert", help="encode a local plain checkpoint"
    )
    convert_parser.add_argument(
        "source", help="checkpoint file, HF index, or directory"
    )
    convert_parser.add_argument(
        "base", help="new output prefix, e.g. /output/gemma/model"
    )
    convert_parser.add_argument("--device", default="spyre:0")
    convert_parser.add_argument(
        "--dtype", choices=["F16", "F32", "BF16"], default="F16"
    )
    convert_parser.add_argument("--num-shards", type=int, default=16)
    convert_parser.add_argument("--fsync", action="store_true")
    convert_parser.add_argument("--workspace-bytes", type=int, default=8 << 20)
    convert_parser.set_defaults(run=convert)
    load_parser = commands.add_parser(
        "load", help="load shards and optionally verify values"
    )
    load_parser.add_argument("source", help="native safetensors file or directory")
    load_parser.add_argument("--device", default="spyre:0")
    load_parser.add_argument(
        "--host-buffer-mib",
        type=int,
        default=16,
        help="total capacity of the two host slots",
    )
    load_parser.add_argument("--max-threads", type=int, default=16)
    load_parser.add_argument(
        "--batch-mib",
        type=int,
        default=3072,
        help="whole-tensor delivery chunk target, not host scratch",
    )
    load_parser.add_argument("--verify-source", help="original plain checkpoint")
    load_parser.set_defaults(run=load)
    args = parser.parse_args()
    args.run(args)


if __name__ == "__main__":
    main()
