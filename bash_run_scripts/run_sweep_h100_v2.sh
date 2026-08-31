# padding-ratio is deliberately left at 0. v2's padded path does not exclude
# padded keys inside SDPA, so it does not reproduce the baseline under padding.
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
  # v2 compiles itself in __init__, so the eager arm must switch it off
  # explicitly. Omitting this flag silently gives a compiled model.
  if ! python torch_transformer_benchmark_v2.py \
    "${COMMON_ARGS[@]}" "$@" \
    --user-compile-mode off; then
    echo "SHAPE ${shape_id} EAGER FAILED; continuing sweep"
  fi

  echo
  echo "########################################################################"
  echo "SHAPE ${shape_id}: eager baseline versus compiled SDPA (CUDA graphs)"
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
