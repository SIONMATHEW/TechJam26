cd ~/techjam3
mkdir -p logs

RUN_ID="$(date +%Y%m%d-%H%M%S)"
LOG="logs/final-v4-${RUN_ID}.log"

exec > >(tee "$LOG") 2>&1

run_shape() {
  local id="$1"; shift
  echo; echo "### SHAPE ${id} EAGER ###"
  python3 torch_transformer_benchmark_v4.py --causal --dtype float32 \
    --warmup 20 --repeats 100 --benchmark-rounds 3 --accuracy-trials 5 \
    --user-compile-mode off "$@"
  echo; echo "### SHAPE ${id} COMPILED ###"
  python3 torch_transformer_benchmark_v4.py --causal --dtype float32 \
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

echo; echo "=== SUMMARY ==="
grep -E "^### SHAPE|^speedup" "$LOG" | paste - - | sed 's/### //'