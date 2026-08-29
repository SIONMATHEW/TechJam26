#!/usr/bin/env bash
set -u -o pipefail

cd ~/techjam3
mkdir -p logs

# Set up venv once; reuse it on later runs.
if [ ! -d ~/kernel-env-h100 ]; then
  echo "=== Creating venv (first run only) ==="
  python3 -m venv ~/kernel-env-h100
  source ~/kernel-env-h100/bin/activate
  pip install --upgrade pip
  pip install torch --index-url https://download.pytorch.org/whl/cu121
  pip install numpy
else
  source ~/kernel-env-h100/bin/activate
fi

RUN_ID="$(date +%Y%m%d-%H%M%S)"
LOGFILE="logs/sweep-v2-${RUN_ID}.log"
SUMMARYFILE="logs/summary-v2-${RUN_ID}.tsv"

exec > >(tee "$LOGFILE") 2>&1

echo "=== Run ID: ${RUN_ID} ==="
echo "Script: torch_transformer_benchmark_v2.py"
echo "Full log: ${LOGFILE}"

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
  if ! python torch_transformer_benchmark_v2.py "${COMMON_ARGS[@]}" "$@" \
    --user-compile-mode off; then
    echo "SHAPE ${shape_id} EAGER FAILED; continuing sweep"
  fi

  echo
  echo "########################################################################"
  echo "SHAPE ${shape_id}: eager baseline versus compiled SDPA"
  echo "########################################################################"
  if ! python torch_transformer_benchmark_v2.py \
    "${COMMON_ARGS[@]}" "$@" \
    --user-compile-mode reduce-overhead; then
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

# Build summary table from the log we just wrote.
{
  echo -e "shape\teager_status\teager_speedup\tcompiled_status\tcompiled_speedup"
  for i in 1 2 3 4 5 6 7 8 9 10 11 12 13; do
    block=$(awk -v s="SHAPE ${i}:" '
      $0 ~ s {found=1}
      found {print}
      found && /^SHAPE '"$((i+1))"':/ {exit}
    ' "$LOGFILE")

    eager_block=$(echo "$block" | awk '/eager SDPA/{flag=1} /compiled SDPA/{flag=0} flag')
    compiled_block=$(echo "$block" | awk '/compiled SDPA/{flag=1} flag')

    estatus=$(echo "$eager_block" | grep -m1 "^summary:" | grep -oE "PASS|FAIL")
    espeedup=$(echo "$eager_block" | grep -m1 "^speedup" | grep -oE "[0-9]+\.[0-9]+x")
    cstatus=$(echo "$compiled_block" | grep -m1 "^summary:" | grep -oE "PASS|FAIL")
    cspeedup=$(echo "$compiled_block" | grep -m1 "^speedup" | grep -oE "[0-9]+\.[0-9]+x")

    echo -e "${i}\t${estatus:-ERR}\t${espeedup:-N/A}\t${cstatus:-ERR}\t${cspeedup:-N/A}"
  done
} | tee "$SUMMARYFILE"

echo
echo "=== SUMMARY TABLE (Run ID: ${RUN_ID}) ==="
column -t "$SUMMARYFILE"
echo
echo "Full log:    ${LOGFILE}"
echo "Summary tsv: ${SUMMARYFILE}"