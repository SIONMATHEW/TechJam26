# MegaMind - GPU Kernel Optimization for a Transformer Layer

TechJam 2026, Track 3: *Implement a GPU Kernel for a Transformer Layer*

A drop-in replacement for the benchmark's `UserOptimizedTransformer` that computes
the same Transformer block within the required error tolerance, substantially
faster, and that executes the largest published shape - which the supplied
reference cannot run on any existing GPU.

**Results (NVIDIA H100 NVL, float32, causal, published shapes 1–13):**

| | Geometric mean | Range | Accuracy |
|---|---|---|---|
| Fusion only (eager) | **2.45x** | 1.40x - 3.58x | 13/13 PASS |
| Fusion + compilation | **4.49x** | 1.53x - 8.21x | 13/13 PASS |

**Shape 14** (`batch=32, seq_len=100000`) cannot be scored against the supplied
baseline, which would need a 20.48 TB score tensor. Our implementation executes it
and passes a full element-by-element correctness check against an independent
tiled reference: `failed=0 / 3,276,800,000`.

Compilation is baked into the model constructor, so running the benchmark with no
extra flags gives the compiled result.

---

## Overview

The supplied reference computes attention the explicit way: it materializes a full
`[batch, heads, seq_len, seq_len]` score matrix in GPU memory, then reads and
writes it back for scaling, masking and softmax. The arithmetic is not the
bottleneck; the memory traffic is.

Our implementation removes that traffic and the per-call overhead around it, in
four measured steps:

1. **Fused attention.** `F.scaled_dot_product_attention` replaces the explicit
   score matrix and softmax, and we drop the baseline's forced `.contiguous()`
   copies before attention since the fused backend accepts strided inputs.
2. **Hoisted padding mask.** The published shapes have no padding, so the
   baseline's per-layer `masked_fill` is a no-op executed `num_layers` times per
   call. We check once whether the mask is trivial and skip it, caching the check
   by tensor identity *and version* so that in-place mutation of a mask is never
   missed.
3. **Packed QKV projection.** Q, K and V share an input, so their three separate
   `nn.Linear` calls become one matmul against a concatenated weight, built once
   and repacked whenever weights change or the module is moved or recast.
4. **Static shape specialization.** Each benchmark process runs one fixed shape,
   so `torch.compile(dynamic=False)` removes the dynamic-shape guards the compiler
   otherwise emits. This is what moved the small shapes.

For sequences at or above `--long-seq-threshold` (default 32768), a separate eager
path processes the batch in microbatches so activations fit in memory, restricted
to memory-efficient SDPA backends. Padded or gradient-enabled calls fall back to a
correct tiled path rather than the fast path.

Parameter names are unchanged throughout, so the harness's strict weight copy works
without modification.

---

## Repository contents

| Path | What it is |
|---|---|
| `techjam_final.py` | **Final submission.** Optimized model, long-sequence path, independent tiled reference, self-test suite, per-shape JSON output. |
| `run_finale.sh` | Full run: shapes 1–13 eager and compiled, plus shape 14 validation. |
| `run_final.sh` | Earlier variant of the full run script. |
| `torch_transformer_benchmark.py` | The supplied harness with our optimized model (pre-`techjam_final` version). |
| `torch_transformer_old_versions/` | Development history, one optimization per version (v1–v3). |
| `Profile_shapes.py` | Per-operator CUDA time breakdown, baseline vs optimized. |
| `bash_run_scripts/` | Earlier sweep scripts. |

---

## Setup

Requires Python 3.12 and an NVIDIA GPU. PyTorch ships its own CUDA runtime, so a
system CUDA toolkit is not required.

```bash
python3 -m venv ~/techjam
source ~/techjam/bin/activate
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install numpy
```

Verify the GPU is visible:

```bash
python3 -c "import torch; print(torch.cuda.get_device_name(0))"
```

### On a SLURM cluster

The login node has no GPU. Allocate one, then connect to it:

```bash
salloc --partition=gpu --gres=gpu:h100-96:1 --time=02:00:00
squeue -u $USER          # note the allocated node
ssh <node>
```

---

## Reproducing our results

**Correctness regressions first** (fast, runs on CPU or GPU):

```bash
python3 techjam_final.py --self-test
```

This checks the tiled reference against the original baseline, the packed-QKV
path, mask mutation, weight mutation and reload, microbatch tails, and the
gradient-enabled fallback. It must print `SELF_TESTS: PASS` before any result is
meaningful.

**Full run, all shapes:**

```bash
bash run_finale.sh
```

**All 14 shapes directly**, each in an isolated subprocess:

```bash
python3 techjam_final.py --all
```

**A single shape:**

```bash
python3 techjam_final.py --shape 1
python3 techjam_final.py --shape 8 --user-compile-mode off   # fusion-only number
python3 techjam_final.py --shape 14                          # long-sequence validation
```

Every run prints a `source_sha256` for the file it executed, so results can be
tied to an exact code version, and writes a JSON record per shape under
`--output-dir` (default `results/final-run`).

---

## Full results

NVIDIA H100 NVL (93.09 GiB), PyTorch 2.5.1+cu121, Python 3.12.3, float32, causal,
`matmul_precision=high`, `allow_tf32=True`, `rtol=0.02`, `atol=0.002`.
Timing settings for this run: `warmup=5, repeats=20, rounds=2`.
`source_sha256=784f3e746af3eece0aecce42d2efb371259bd4ac6d48efab8e642d0fb02c05ac`

| Shape | batch | seq_len | d_model | heads | ffn_dim | Eager | Compiled |
|---|---|---|---|---|---|---|---|
| 1 | 64 | 128 | 128 | 4 | 128 | 2.692x | 4.235x |
| 2 | 1 | 128 | 128 | 4 | 128 | 2.958x | **8.212x** |
| 3 | 4 | 128 | 128 | 4 | 128 | 2.810x | 7.414x |
| 4 | 16 | 128 | 128 | 4 | 128 | 2.386x | 6.711x |
| 5 | 128 | 128 | 128 | 4 | 128 | 1.975x | 2.938x |
| 6 | 10000 | 128 | 128 | 4 | 128 | 2.134x | 3.037x |
| 7 | 64 | 128 | 32 | 4 | 32 | 2.396x | 5.085x |
| 8 | 64 | 128 | 1024 | 4 | 1024 | 1.401x | 1.530x |
| 9 | 64 | 128 | 128 | 1 | 128 | 2.447x | 4.920x |
| 10 | 64 | 128 | 128 | 2 | 128 | 2.731x | 4.759x |
| 11 | 64 | 128 | 128 | 16 | 128 | 2.412x | 3.033x |
| 12 | 64 | 32 | 128 | 4 | 128 | 2.560x | 8.094x |
| 13 | 64 | 1024 | 128 | 4 | 128 | 3.578x | 4.396x |
| | | | | | **geomean** | **2.445x** | **4.492x** |

Geometric mean is used because these are ratios. Arithmetic mean on speedups is
biased upward: a 4x gain and a 4x loss average to 2.1x arithmetically but
correctly to 1.0x geometrically.

An earlier sweep of the same optimizations at the harness's full timing settings
(`warmup=20, repeats=100, rounds=3`) produced a 4.518x compiled geometric mean,
within noise of the 4.492x above.

### Shape 14

`batch=32, seq_len=100000, d_model=1024, heads=16, ffn_dim=1024, layers=2`

The supplied baseline computes `matmul(q, k.transpose(-2, -1))` explicitly, which
at this shape is a `[32, 16, 100000, 100000]` float32 tensor: **20.48 TB** for one
tensor in one layer. No single GPU holds that, so no baseline output exists and no
official speedup can be computed. The script reports this explicitly rather than
substituting a proxy score:

```
Original baseline not attempted: 20.48 TB score tensor; official-baseline speedup=N/A
Tiled-reference ratio is experimental, not an organizer-approved score.
```

To validate correctness anyway, we built an independent tiled reference
(`tiled_reference_attention`) that computes the original attention formula in query
blocks without materializing the full score matrix. It deliberately does **not**
reuse the optimized path: it uses the original separate Q/K/V projections, explicit
`torch.matmul`, and fp32 softmax, never SDPA and never the packed weights.

Before it is used as ground truth, the self-test suite checks it reproduces the
*original* baseline at small shapes:

```
self-test PASS: tiled reference causal=False; max_abs=0
self-test PASS: tiled reference causal=True;  max_abs=0
```

Shape 14 then passes in full:

```
full_shape_accuracy: PASS | failed=0/3276800000 | max_abs=0.000704706
```

`compare_streamed_outputs` applies the unmodified supplied checker to every
element in memory-safe slices and asserts that the element count visited equals
`reference.numel()`, so this is a complete check, not a sample.

---

## Limitations and what we would improve

**Shape 8 is near a structural ceiling, and we can show why.** Profiling with
`torch.profiler` shows `aten::addmm` is 80% of baseline runtime and 90% of ours.
The shape is wide (`d_model=1024`) and short (`seq_len=128`), so dense matrix
multiplication dominates and attention fusion can only address the remainder. Our
packed-QKV change cut the matmul call count from 240 to 160 without reducing total
matmul time, confirming the work itself was already efficient. The 1.53x measured
is close to the limit for a precision-preserving change here.

**Custom GEMM kernels were evaluated and rejected on evidence.** Running Inductor
in `max-autotune` generates and benchmarks dozens of Triton matmul kernels against
cuBLAS. cuBLAS won every comparison — the best Triton candidates reached 86% and
79% of library performance on our two hottest matmuls — and end-to-end
`max-autotune` came out slightly slower than `reduce-overhead` (1.449x vs 1.461x)
after roughly 250 seconds of compilation.

**Reduced precision does not meet the tolerance.** bfloat16 fails badly. float16
comes close but still fails on roughly 130 elements out of 84 million. The subtlety
is that the target is not the mathematically correct answer but the reference's own
float16 arithmetic — we tried keeping the residual stream in float32 to limit error
accumulation and agreement got *worse*, because being more accurate than the
reference is still being different from it.

**We are on the memory-efficient SDPA backend, not FlashAttention.** Flash rejects
float32 and the benchmark runs in float32; our profile confirms the
`fmha_cutlassF_f32` kernel. Casting only Q, K and V to bfloat16 inside attention,
leaving the residual stream and FFN in float32, would make flash eligible. The
profile suggests this mainly helps long-sequence shapes.

**Shape 14's reference is ours, not the organizers'.** It is validated bit-exact
against the original where the original runs, but it is still our implementation,
and no speedup is claimed for this shape. We have raised the shape 14 feasibility
question with the organizers and would prefer an official validation path.

---

## Environment

- **GPU:** NVIDIA H100 NVL, 93.09 GiB
- **PyTorch:** 2.5.1+cu121, CUDA build 12.1
- **Python:** 3.12.3
- **Cluster:** NUS SoC compute cluster (SLURM), node `xgpi8`

---

## Team

- **Shubhan Gabra**
- **Anugrah Bagla**
- **Saayuj Ion Mathew**
