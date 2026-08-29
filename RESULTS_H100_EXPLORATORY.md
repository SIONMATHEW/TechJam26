# H100 Exploratory Results — SDPA V1

## Environment

- GPU allocation: NVIDIA H100 NVL MIG 3g.47gb
- PyTorch: 2.11.0+cu128
- CUDA build: 12.8
- Data type: float32
- Workload: causal Transformer inference
- Accuracy tolerance: absolute error <= 0.002 **or** relative error <= 2%
- Exploratory timing: 2 accuracy trials, 5 warmups, 20 repeats, 2 rounds

## Results

Every tested implementation passed accuracy with zero failed elements.

| Shape | Eager SDPA speedup | Compiled SDPA speedup | Accuracy |
|---:|---:|---:|:---:|
| 1 | 1.555x | 2.438x | PASS |
| 2 | 1.905x | 7.100x | PASS |
| 3 | 1.885x | 6.872x | PASS |
| 4 | 1.901x | 4.639x | PASS |
| 5 | 1.758x | 2.717x | PASS |
| 6 | 1.854x | 2.605x | PASS |
| 7 | 1.842x | 3.258x | PASS |
| 8 | 1.254x | 1.451x | PASS |
| 9 | 1.437x | 2.412x | PASS |
| 10 | 1.497x | 2.533x | PASS |
| 11 | 2.204x | 2.808x | PASS |
| 12 | 2.026x | 5.138x | PASS |
| 13 | 3.492x | 3.600x | PASS |

- Eager SDPA geometric-mean speedup: **1.835x**
- Compiled SDPA geometric-mean speedup: **3.316x**
- Compiled SDPA range: **1.451x–7.100x**

## Interpretation and limitations

- The eager result isolates the benefit of replacing explicit attention with
  PyTorch scaled-dot-product attention (SDPA).
- The compiled result compares an eager reference baseline against SDPA V1
  compiled with `torch.compile(mode="reduce-overhead")`. It must be reported
  with that qualification rather than presented as an equal-compilation
  comparison.
- These are exploratory measurements, not final submission numbers. The final
  report should use the full accuracy and timing settings and include repeated
  runs with variance.
- Shape 8 is the weakest V1 case and should be profiled first for the next
  optimization pass.
- Shape 14 was not run. Its explicit float32 reference attention-score tensor
  would require roughly 20.5 TB, so it needs organizer clarification and a
  memory-safe validation strategy.

## Evidence

- Slurm job: `770672`
- Expected raw log in the cluster workspace:
  `results/raw/shapes01-13-h100-job770672.out`
