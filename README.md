# MegaMind - GPU Kernel Optimization for a Transformer Layer

TechJam 2026, Track 3: *Implement a GPU Kernel for a Transformer Layer*

A drop-in replacement for the benchmark's `UserOptimizedTransformer` that computes
the same Transformer block within the required error tolerance, substantially
faster.

**Headline result (NVIDIA H100 NVL, float32, causal, first 13 published shapes):**

| | Geometric mean | Range | Accuracy |
|---|---|---|---|
| Fusion only (eager) | **2.54x** | 1.35x - 3.91x | 13/13 PASS |
| Fusion + compilation | **4.52x** | 1.46x - 8.40x | 13/13 PASS |

Compilation is baked into the model constructor, so running the benchmark with no
extra flags gives the compiled result.

---

## Overview

The reference implementation computes attention the explicit way: it materializes
a full `[batch, heads, seq_len, seq_len]` score matrix in GPU memory, then reads
and writes it back for scaling, masking, and softmax. The arithmetic is not the
bottleneck; the memory traffic is.

Our implementation removes that traffic and the per-call overhead around it, in
four measured steps:

1. **Fused attention.** `F.scaled_dot_product_attention` replaces the explicit
   score matrix and softmax, and we drop the baseline's forced `.contiguous()`
   copies before attention since the fused backend accepts strided inputs.
2. **Hoisted padding mask.** The published shapes have no padding, so the
   baseline's per-layer `masked_fill` is a no-op executed `num_layers` times per
   call. We check once whether the mask is trivial and skip it - caching the check
   by tensor identity so it does not force a GPU sync inside a CUDA graph.
3. **Packed QKV projection.** Q, K and V share an input, so their three separate
   `nn.Linear` calls become one matmul against a concatenated weight, built once
   after weight loading and registered as buffers.
4. **Static shape specialization.** Each benchmark process runs exactly one fixed
   shape, so `torch.compile(dynamic=False)` removes the dynamic-shape guards the
   compiler otherwise emits. This is what moved the small shapes.

Parameter names are unchanged throughout, so the harness's strict weight copy
works without modification.

---

## Repository contents

| Path | What it is |
|---|---|
| `torch_transformer_old_versions/torch_transformer_benchmark_v4.py` | **Final submission.** Full benchmark harness with our optimized model. |
| `torch_transformer_old_versions/torch_transformer_benchmark_v3.py` | Packed QKV, without static shape specialization. |
| `torch_transformer_old_versions/torch_transformer_benchmark_v2.py` | Hoisted mask and compile-in-constructor. |
| `torch_transformer_old_versions/torch_transformer_benchmark_v1.py` | Fused attention only. |
| `profile_shapes.py` | Per-operator CUDA time breakdown, baseline vs optimized. |
| `run_sweep_h100.sh` | Sweep across shapes 1–13, eager and compiled. |
| `logs/` | Raw benchmark and profiler output. |

---

## Setup

Requires Python 3.12 and an NVIDIA GPU. PyTorch ships its own CUDA runtime, so a
system CUDA toolkit is not needed for the benchmark itself.

```bash
python3 -m venv ~/kernel-env
source ~/kernel-env/bin/activate
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install numpy
```

Verify the GPU is visible:

```bash
python3 -c "import torch; print(torch.cuda.get_device_name(0))"
```

## Reproducing our results (On a GPU)

### Single shape

```bash
python3 torch_transformer_benchmark.py --causal --dtype float32 \
  --batch-size 64 --seq-len 128 --d-model 128 --heads 4 --ffn-dim 128 --layers 4
```

With no compile flag this uses `--user-compile-mode reduce-overhead`, the default
baked into the model. Pass `--user-compile-mode off` for the fusion-only number.

### Full sweep, shapes 1–13

The numbers in the table above come from the harness's full timing settings
(`--warmup 20 --repeats 100 --benchmark-rounds 3 --accuracy-trials 5`):

```bash
run_shape() {
  local id="$1"; shift
  echo "### SHAPE ${id} EAGER ###"
  python3 torch_transformer_benchmark.py --causal --dtype float32 \
    --warmup 20 --repeats 100 --benchmark-rounds 3 --accuracy-trials 5 \
    --user-compile-mode off "$@"
  echo "### SHAPE ${id} COMPILED ###"
  python3 torch_transformer_benchmark.py --causal --dtype float32 \
    --warmup 20 --repeats 100 --benchmark-rounds 3 --accuracy-trials 5 \
    --user-compile-mode reduce-overhead "$@"
}

run_shape 1  --batch-size 64    --seq-len 128  --d-model 128  --heads 4  --ffn-dim 128  --layers 4
run_shape 2  --batch-size 1     --seq-len 128  --d-model 128  --heads 4  --ffn-dim 128  --layers 4
run_shape 3  --batch-size 4     --seq-len 128  --d-model 128  --heads 4  --ffn-dim 128  --layers 4
run_shape 4  --batch-size 16    --seq-len 128  --d-model 128  --heads 4  --ffn-dim 128  --layers 4
run_shape 5  --batch-size 128   --seq-len 128  --d-model 128  --heads 4  --ffn-dim 128  --layers 4
run_shape 6  --batch-size 10000 --seq-len 128  --d-model 128  --heads 4  --ffn-dim 128  --layers 4
run_shape 7  --batch-size 64    --seq-len 128  --d-model 32   --heads 4  --ffn-dim 32   --layers 4
run_shape 8  --batch-size 64    --seq-len 128  --d-model 1024 --heads 4  --ffn-dim 1024 --layers 4
run_shape 9  --batch-size 64    --seq-len 128  --d-model 128  --heads 1  --ffn-dim 128  --layers 4
run_shape 10 --batch-size 64    --seq-len 128  --d-model 128  --heads 2  --ffn-dim 128  --layers 4
run_shape 11 --batch-size 64    --seq-len 128  --d-model 128  --heads 16 --ffn-dim 128  --layers 4
run_shape 12 --batch-size 64    --seq-len 32   --d-model 128  --heads 4  --ffn-dim 128  --layers 4
run_shape 13 --batch-size 64    --seq-len 1024 --d-model 128  --heads 4  --ffn-dim 128  --layers 4
```

### Profiling

```bash
python3 profile_shapes.py --module torch_transformer_benchmark
```

Defaults to shape 8. Uses `--user-compile-mode off` by default, because a compiled
model collapses into a few opaque graph entries and hides the per-operator
breakdown the script exists to show.

---

## Full results

NVIDIA H100 NVL, float32, causal, driver 580.173.02, PyTorch 2.5.1+cu121.
All shapes(1-13) pass the accuracy criterion (`abs_error <= 0.002 OR rel_error <= 2%`).

| Shape | batch | seq_len | d_model | heads | ffn_dim | Eager | Compiled |
|---|---|---|---|---|---|---|---|
| 1 | 64 | 128 | 128 | 4 | 128 | 2.803x | 4.202x |
| 2 | 1 | 128 | 128 | 4 | 128 | 3.152x | **8.398x** |
| 3 | 4 | 128 | 128 | 4 | 128 | 3.026x | 7.488x |
| 4 | 16 | 128 | 128 | 4 | 128 | 2.557x | 6.778x |
| 5 | 128 | 128 | 128 | 4 | 128 | 1.989x | 2.978x |
| 6 | 10000 | 128 | 128 | 4 | 128 | 2.143x | 3.042x |
| 7 | 64 | 128 | 32 | 4 | 32 | 2.550x | 5.138x |
| 8 | 64 | 128 | 1024 | 4 | 1024 | 1.349x | 1.461x |
| 9 | 64 | 128 | 128 | 1 | 128 | 2.560x | 4.959x |
| 10 | 64 | 128 | 128 | 2 | 128 | 2.789x | 4.892x |
| 11 | 64 | 128 | 128 | 16 | 128 | 2.428x | 3.074x |
| 12 | 64 | 32 | 128 | 4 | 128 | 2.729x | 7.970x |
| 13 | 64 | 1024 | 128 | 4 | 128 | 3.913x | 4.517x |
| | | | | | **geomean** | **2.543x** | **4.518x** |

---

## Limitations and what we would improve

**Shape 8 is near a structural ceiling, and we can show why.** Profiling shows
`aten::addmm` is 80% of baseline runtime and 90% of ours. The shape is wide
(`d_model=1024`) and short (`seq_len=128`), so dense matrix multiplication
dominates and attention fusion can only address the remainder. The 1.46x we
measure is close to the limit of what a precision-preserving change can achieve
here.

**Custom GEMM kernels were evaluated and rejected on evidence.** Running Inductor
in `max-autotune` generates and benchmarks dozens of Triton matmul kernels against
cuBLAS. cuBLAS won every comparison; the best Triton candidates reached 86% and
79% of library performance on our two hottest matmuls, and end-to-end
`max-autotune` came out slightly slower than `reduce-overhead` (1.449x vs 1.461x)
after roughly 250 seconds of compilation. Given the profile, hand-writing Triton
GEMM kernels was not a good use of remaining effort.

**Reduced precision does not meet the tolerance.** bfloat16 fails badly.
float16 comes close but still fails on roughly 130 elements out of 84 million.
The subtlety is that the target is not the mathematically correct answer but the
reference's own float16 arithmetic - we tried keeping the residual stream in
float32 to limit error accumulation and it made agreement *worse*, because being
more accurate than the reference is still being different from it.

**We are on the memory-efficient SDPA backend, not FlashAttention.** Flash rejects
float32 outright, and the benchmark runs in float32. Casting only Q, K and V to
bfloat16 inside attention, leaving the residual stream and FFN in float32, would
make flash eligible. Our profile suggests this mainly helps long-sequence shapes
where attention is a large fraction of runtime, so it would be worth targeting
rather than applying everywhere.

**The fast path does not mask padded keys.** With `padding_ratio > 0`, the
implementation falls back to a slower path that handles masking correctly. The
published shapes all use the harness default of `padding_ratio=0`, so this does
not affect the reported results, but a variable-length attention path would be
needed for general use.

---

## Environment

Results were produced on:

- **GPU:** NVIDIA H100 NVL, 95830 MiB, driver 580.173.02
- **PyTorch:** 2.5.1+cu121, Python 3.12.3
- **Cluster:** NUS SoC compute cluster (SLURM)

---

## Team

- **Shubhan Gabra**
- **Anugrah Bagla**
- **Saayuj Ion Mathew**
