#!/usr/bin/env python3
"""
Profile the baseline and the optimized model on one shape and print a
per-operator CUDA time breakdown for each.

The point is to find where time actually goes rather than inferring it from
the shape parameters. Run it on shape 8 first (the shape that has resisted
three rounds of optimization) and compare the two tables: an operator that
is large in both is a bottleneck the current optimizations never touched.

Defaults to published shape 8. Override with the usual flags for other shapes.

Usage:
  python3 profile_shapes.py --module torch_transformer_benchmark_v3
  python3 profile_shapes.py --module torch_transformer_benchmark_v3 \
      --batch-size 1 --seq-len 128 --d-model 128 --ffn-dim 128
"""

import argparse
import importlib

import torch
from torch.profiler import ProfilerActivity, profile


def build_models(module, config, device, dtype, compile_mode):
    baseline = module.BaselineTransformer(config)
    try:
        optimized = module.UserOptimizedTransformer(config, compile_mode=compile_mode)
    except TypeError:
        # Older versions take no compile_mode argument.
        optimized = module.UserOptimizedTransformer(config)

    module.copy_model_weights(baseline, optimized, strict=True)
    baseline = baseline.to(device=device, dtype=dtype).eval()
    optimized = optimized.to(device=device, dtype=dtype).eval()
    return baseline, optimized


def profile_model(model, x, mask, device, label, warmup, active, row_limit):
    # Warm up outside the profiler: first calls include compilation and
    # autotuning, which would otherwise dominate the table.
    with torch.inference_mode():
        for _ in range(warmup):
            model(x, mask)
    torch.cuda.synchronize(device)

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
    ) as prof:
        with torch.inference_mode():
            for _ in range(active):
                model(x, mask)
        torch.cuda.synchronize(device)

    print()
    print("=" * 78)
    print(f"{label}: top operators by self CUDA time ({active} iterations)")
    print("=" * 78)
    print(
        prof.key_averages().table(
            sort_by="self_cuda_time_total",
            row_limit=row_limit,
        )
    )
    return prof


def main() -> int:
    parser = argparse.ArgumentParser(description="Profile baseline vs optimized")
    parser.add_argument(
        "--module",
        default="torch_transformer_benchmark_v3",
        help="benchmark module to import the models from (no .py extension)",
    )
    # Published shape 8 defaults.
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--d-model", type=int, default=1024)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--ffn-dim", type=int, default=1024)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--causal", action="store_true", default=True)
    parser.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="float32")
    parser.add_argument(
        "--user-compile-mode",
        choices=("off", "default", "reduce-overhead", "max-autotune"),
        default="off",
        help=(
            "compile mode for the optimized model. Default 'off': a compiled "
            "model collapses into a few opaque graph/kernel entries, which "
            "hides the per-operator breakdown this script exists to show."
        ),
    )
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--active", type=int, default=10)
    parser.add_argument("--row-limit", type=int, default=20)
    parser.add_argument(
        "--trace-prefix",
        default="",
        help="if set, also write Chrome traces to <prefix>-baseline.json and <prefix>-optimized.json",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA is not available; this script profiles GPU kernels.")
        return 1

    module = importlib.import_module(args.module)
    dtype = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[args.dtype]
    device = torch.device("cuda")

    config = module.TransformerConfig(
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        d_model=args.d_model,
        num_heads=args.heads,
        ffn_dim=args.ffn_dim,
        num_layers=args.layers,
        causal=args.causal,
    )
    config.validate()

    print("=== Profiling configuration ===")
    print(config)
    print(f"module={args.module}, dtype={dtype}, user_compile_mode={args.user_compile_mode}")
    print(f"gpu={torch.cuda.get_device_name(device)}")

    torch.manual_seed(1234)
    torch.cuda.manual_seed_all(1234)

    baseline, optimized = build_models(module, config, device, dtype, args.user_compile_mode)

    x = torch.randn(
        config.batch_size, config.seq_len, config.d_model, device=device, dtype=dtype
    )
    mask = torch.ones(config.batch_size, config.seq_len, device=device, dtype=torch.bool)

    prof_base = profile_model(
        baseline, x, mask, device, "BASELINE", args.warmup, args.active, args.row_limit
    )
    prof_opt = profile_model(
        optimized, x, mask, device, "OPTIMIZED", args.warmup, args.active, args.row_limit
    )

    if args.trace_prefix:
        prof_base.export_chrome_trace(f"{args.trace_prefix}-baseline.json")
        prof_opt.export_chrome_trace(f"{args.trace_prefix}-optimized.json")
        print(f"\nChrome traces written to {args.trace_prefix}-baseline.json and "
              f"{args.trace_prefix}-optimized.json (open in chrome://tracing)")

    print()
    print("How to read this: compare the two tables. An operator with a large")
    print("self CUDA time in BOTH is work the current optimizations never")
    print("removed, and is where the next optimization should go.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
