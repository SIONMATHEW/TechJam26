#!/usr/bin/env bash
#SBATCH --job-name=techjam-v1-sweep
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100-47:1
#SBATCH --time=00:45:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=track3-v1-sweep-%j.out

set -u -o pipefail

PROJECT_DIR="$HOME/techjam3"
ENV_ROOT="/tmp/${USER}/techjam3-sweep-${SLURM_JOB_ID}"

cd "$PROJECT_DIR"
mkdir -p "$ENV_ROOT"

echo "=== Installing isolated PyTorch environment ==="
python3 -m venv "$ENV_ROOT/.venv"
source "$ENV_ROOT/.venv/bin/activate"
python -m pip install --upgrade pip
python -m pip install \
  "torch==2.11.0+cu128" \
  --index-url https://download.pytorch.org/whl/cu128 \
  --extra-index-url https://pypi.org/simple \
  --no-cache-dir \
  --timeout 120 \
  --retries 20 \
  --resume-retries 50

echo
echo "=== Environment ==="
hostname
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("compiled CUDA:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")
PY

COMMON_ARGS=(
  --device cuda
  --dtype float32
  --causal
  --accuracy-trials 2
  --warmup 5
  --repeats 20
  --benchmark-rounds 2
)

run_shape() {
  local shape_id="$1"
  shift

  echo
  echo "########################################################################"
  echo "SHAPE ${shape_id}: eager baseline versus eager SDPA"
  echo "########################################################################"
  if ! python torch_transformer_benchmark_v1.py "${COMMON_ARGS[@]}" "$@"; then
    echo "SHAPE ${shape_id} EAGER FAILED; continuing sweep"
  fi

  echo
  echo "########################################################################"
  echo "SHAPE ${shape_id}: eager baseline versus compiled SDPA"
  echo "########################################################################"
  if ! python torch_transformer_benchmark_v1.py \
    "${COMMON_ARGS[@]}" "$@" \
    --compile-user \
    --compile-mode reduce-overhead; then
    echo "SHAPE ${shape_id} COMPILED FAILED; continuing sweep"
  fi
}

# Published test matrix. Shape 14 is intentionally excluded: its explicit
# reference attention scores alone require roughly 20.5 TB in float32.
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

echo
echo "=== Shapes 1-13 exploratory sweep completed ==="
