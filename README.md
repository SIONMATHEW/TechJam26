# TechJam Track 3 — Shape-Aware Transformer Kernel Optimization

This repository optimizes the supplied PyTorch Transformer benchmark while
preserving its numerical accuracy contract. The current V1 replaces explicit
attention-score materialization with PyTorch scaled-dot-product attention
(SDPA) and evaluates an optional `torch.compile(mode="reduce-overhead")` path.

## Current result

On an NVIDIA H100 NVL MIG 3g.47gb allocation, all published shapes 1–13 passed
the organizer's accuracy rule with zero failed elements.

- Eager SDPA geometric-mean speedup: **1.835x**
- Compiled SDPA geometric-mean speedup: **3.316x**
- Compiled per-shape range: **1.451x–7.100x**

These are exploratory measurements using 2 accuracy trials, 5 warmups,
20 repeats, and 2 timing rounds. See `RESULTS_H100_EXPLORATORY.md` and the raw
Slurm log for the complete configuration and caveats.

## Optimization

Only `UserOptimizedTransformer` changes the model implementation. V1:

- calls `torch.nn.functional.scaled_dot_product_attention` so CUDA can select
  a fused attention backend;
- avoids the reference implementation's forced contiguous Q/K/V copies;
- preserves parameter names and values for strict weight copying;
- preserves LayerNorm, residuals, GELU, FFN, and final normalization;
- supports the published causal, all-valid-token (`padding_ratio=0`) matrix.

The benchmark's baseline, accuracy checker, random input generator, and CUDA
event timer remain intact.

## Files

- `torch_transformer_benchmark_v1.py` — supplied benchmark plus SDPA V1.
- `run_sweep_h100.sh` — reproducible combined install-and-run Slurm sweep for
  shapes 1–13 on an H100 47 GB allocation.
- `RESULTS_H100_EXPLORATORY.md` — compact results table and interpretation.
- `results/raw/` — immutable Slurm logs used as benchmark evidence.

Temporary environments are created inside the Slurm allocation because the
cluster's `/tmp` is job-isolated and the tested home quota cannot hold the CUDA
PyTorch environment. The script downloads PyTorch once per allocation and then
runs the complete sweep.

## Reproduce the H100 exploratory sweep

From an NUS SoC cluster login node:

```bash
cd ~/techjam3
bash -n run_sweep_h100.sh
sbatch run_sweep_h100.sh
```

Slurm prints a job ID. Inspect it with:

```bash
squeue -j JOB_ID
tail -f track3-v1-sweep-JOB_ID.out
```

The final line should be:

```text
=== Shapes 1-13 exploratory sweep completed ===
```

## Accuracy contract

An element passes when its absolute error is at most `0.002` **or** its
relative error is at most `2%`. Large maximum relative-error values can occur
for reference values near zero; the authoritative evidence is `failed=0`.

## Known limitations and next work

- V1 does not implement padded-key masking and should only be claimed for the
  published `padding_ratio=0` cases.
- The compiled measurements compare a compiled optimized model with the eager
  supplied baseline. Reports must state this explicitly.
- Shape 8 is the weakest V1 regime and is the first profiling target.
- Shape 14 cannot be validated literally with the supplied reference: its
  explicit float32 attention-score tensor would require roughly 20.5 TB. It
  needs organizer clarification plus a memory-safe validation strategy.
- Final results need the full timing configuration, repeated runs, variance,
  and H200 measurements.

## Collaboration

Use one branch per experiment and merge through pull requests. Never commit
credentials, access tokens, SSH keys, cluster passwords, or virtual
environments. Each teammate must use their own authorized cluster account.
