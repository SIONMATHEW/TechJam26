#!/usr/bin/env bash

# Use your ALREADY ACTIVE Python environment. Nothing is installed or activated.
# Run on your allocated GPU, with techjam active: ./run_final.sh
# This script does not request resources or submit jobs. Keep the session open.
set -euo pipefail
if (( $# > 0 )); then
    echo "This simple runner takes no arguments. Edit COMMON_ARGS below instead." >&2
    exit 2
fi

SCRIPT_SOURCE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/$(basename -- "${BASH_SOURCE[0]}")"
PROJECT_DIR="$(dirname -- "$SCRIPT_SOURCE")"
cd "$PROJECT_DIR"
[[ -f techjam_final.py ]] || { echo "Missing techjam_final.py in $PWD" >&2; exit 2; }
mkdir -p results
RUN_DIR="$(mktemp -d "$PWD/results/final-$(date +%Y%m%d-%H%M%S)-XXXXXX")"
exec > >(tee "$RUN_DIR/run.log") 2>&1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-8}}"
echo "Results: $RUN_DIR"
hostname

# Check and record the CURRENT environment; never create or switch one.
python - <<'PY' | tee "$RUN_DIR/environment.txt"
import sys
import torch
from importlib.metadata import distributions
print("Python executable:", sys.executable)
print("Python:", sys.version)
print("PyTorch:", torch.__version__)
print("CUDA build:", torch.version.cuda)
if not torch.cuda.is_available():
    raise SystemExit("This Python environment cannot access CUDA. Nothing was installed or changed.")
print("GPU:", torch.cuda.get_device_name(0))
print("GPU memory GiB:", torch.cuda.get_device_properties(0).total_memory / 1024**3)
print("Installed packages:")
print("\n".join(sorted(f"{d.metadata['Name']}=={d.version}" for d in distributions())))
PY

# Run a snapshot so later edits cannot change this sweep halfway through.
cp -- techjam_final.py "$RUN_DIR/techjam_final.py"
cp -- "$SCRIPT_SOURCE" "$RUN_DIR/run_final.sh"
CODE="$RUN_DIR/techjam_final.py"
printf 'shape\tmode\texit_code\n' > "$RUN_DIR/case-exit-codes.tsv"

# Short exploratory settings, matching your example. Accuracy limits unchanged.
# For the previous longer settings: trials=5, warmup=20, repeats=100, rounds=3.
# Padding=0 matches the published unpadded sweep; the old v2 padding warning
# does not apply to techjam_final.py, which has correctness-safe padding paths.
COMMON_ARGS=(
    --device cuda --dtype float32 --causal --padding-ratio 0
    --accuracy-trials 2 --warmup 5 --repeats 20 --benchmark-rounds 2
)

run_case() {
    local shape_id="$1" mode="$2" label="$3"
    shift 3
    local stem code=0 compile_mode="$mode"
    printf -v stem 'shape%02d-%s' "$shape_id" "$mode"
    [[ "$mode" != "long-fused-eager" ]] || compile_mode=off
    echo
    echo "########################################################################"
    echo "SHAPE ${shape_id}: ${label}"
    echo "########################################################################"
    python "$CODE" --shape "$shape_id" "${COMMON_ARGS[@]}" \
        --user-compile-mode "$compile_mode" --result-json "$RUN_DIR/$stem.json" \
        "$@" 2>&1 | tee "$RUN_DIR/$stem.log" || code=$?
    printf '%s\t%s\t%s\n' "$shape_id" "$mode" "$code" >> "$RUN_DIR/case-exit-codes.tsv"
    if (( code == 130 || code == 143 )); then
        echo "Interrupted; stopping the sweep. Logs remain in $RUN_DIR."
        exit "$code"
    fi
    if (( code != 0 )); then
        echo "SHAPE ${shape_id} ${mode} FAILED (exit ${code}); continuing sweep."
    fi
}

run_shape() {
    run_case "$1" off "eager baseline versus EAGER optimized SDPA"
    run_case "$1" reduce-overhead "eager baseline versus COMPILED optimized SDPA"
}

# --shape selects the exact published dimensions already stored in the Python.
for shape_id in {1..13}; do
    run_shape "$shape_id"
done

# Shape 14: B=32, S=100000, D=1024, H=16, FFN=1024, layers=2, causal.
# Full accuracy against an independent tiled reference, then time both.
# NEVER attempt the 20.48 TB explicit baseline or label this an official ratio.
run_case 14 long-fused-eager "FULL accuracy + tiled reference versus optimized" \
    --microbatch 1 --reference-query-block 256 \
    --shape14-accuracy-trials 1 --shape14-warmup 1 \
    --shape14-repeats 3 --shape14-rounds 1

# Build summaries from structured results, not by scraping the live log.
STATUS=0
python - "$RUN_DIR" <<'PY' || STATUS=$?
import csv
import json
import sys
from pathlib import Path
root = Path(sys.argv[1])
sys.path.insert(0, str(root))
from techjam_final import write_summary
records = []
with (root / "case-exit-codes.tsv").open() as stream:
    for entry in csv.DictReader(stream, delimiter="\t"):
        shape, mode, code = int(entry["shape"]), entry["mode"], int(entry["exit_code"])
        path = root / f"shape{shape:02d}-{mode}.json"
        row = dict(shape=shape, mode=mode, status="ERROR", accuracy_passed=False)
        try:
            row.update(json.loads(path.read_text()))
        except (OSError, ValueError) as exc:
            row["error"] = str(exc)
        row.update(shape=shape, mode=mode, exit_code=code,
                   reference_kind="tiled" if shape == 14 else "original")
        if code != 0 or row.get("status") != "PASS" or not row.get("accuracy_passed"):
            row.update(status="ERROR", baseline_speedup=None, tiled_reference_speedup=None)
        records.append(row)
aggregates = write_summary(root, records)
print("\n=== FINAL SUMMARY: shape 14 uses a DIFFERENT reference ===")
print((root / "summary.tsv").read_text())
print("Shapes 1-13 geometric means:", json.dumps(aggregates))
sys.exit(0 if len(records) == 27 and all(r["status"] == "PASS" for r in records) else 2)
PY
echo "Suite exit code: $STATUS (zero means all 27 cases passed)"
echo "Full log: $RUN_DIR/run.log"
echo "Summary: $RUN_DIR/summary.tsv"
echo "JSON:    $RUN_DIR/summary.json"
exit "$STATUS"
