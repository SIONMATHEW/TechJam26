#!/usr/bin/env bash
#SBATCH --job-name=techjam-final
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100-96:1
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=techjam-final-%j.out

# Inside an existing GPU allocation: bash run_final.sh
# From the login node:               sbatch run_final.sh
# H200 from the login node:          sbatch --gres=gpu:h200-141:1 run_final.sh
# Only compiled shapes 1-13 + full14: bash run_final.sh --modes compiled
# The dedicated full-size reference can be slow; use a sufficiently long job.

set -euo pipefail
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo "No Slurm allocation detected. From this folder, use: sbatch run_final.sh" >&2
    echo "Do not run GPU benchmarks on a login node." >&2
    exit 2
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_SOURCE="$SCRIPT_DIR/$(basename -- "${BASH_SOURCE[0]}")"
if [[ -f "$SCRIPT_DIR/techjam_final.py" ]]; then
    DEFAULT_PROJECT_DIR="$SCRIPT_DIR"
else
    DEFAULT_PROJECT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
fi
PROJECT_DIR="${TECHJAM_PROJECT_DIR:-$DEFAULT_PROJECT_DIR}"
VENV_DIR="${TECHJAM_VENV:-$HOME/kernel-env-h100}"
EXPECTED_TORCH="2.11.0+cu128"
cd "$PROJECT_DIR"
if [[ ! -f techjam_final.py ]]; then
    echo "Missing $PROJECT_DIR/techjam_final.py. Put both files in the same folder." >&2
    exit 2
fi

mkdir -p results
RUN_DIR="$(mktemp -d "$PROJECT_DIR/results/final-$(date +%Y%m%d-%H%M%S)-job${SLURM_JOB_ID}-XXXXXX")"
exec > >(tee "$RUN_DIR/run.log") 2>&1
echo "=== TechJam final: shapes 1-14 ==="
echo "Run directory: $RUN_DIR"
echo "Python environment: $VENV_DIR"
echo "One Slurm allocation; no additional GPU jobs will be submitted."
hostname
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
fi

# Serialize first-time installation across jobs. Existing environments are
# reused but never silently upgraded/downgraded while another job may use them.
if ! command -v flock >/dev/null 2>&1; then
    echo "flock is required for safe shared-environment setup." >&2
    exit 3
fi
mkdir -p "$(dirname -- "$VENV_DIR")"
exec 9>"${VENV_DIR}.setup.lock"
flock 9
if [[ ! -e "$VENV_DIR" ]]; then
    echo "Creating persistent environment once. Future runs reuse it."
    python3 -m venv "$VENV_DIR"
    "$VENV_DIR/bin/python" -m pip install --no-cache-dir \
        "torch==$EXPECTED_TORCH" --index-url https://download.pytorch.org/whl/cu128
    "$VENV_DIR/bin/python" -m pip install --no-cache-dir "numpy==2.2.6"
fi
if [[ ! -x "$VENV_DIR/bin/python" ]] || ! "$VENV_DIR/bin/python" -c \
    "import torch; assert torch.__version__ == '$EXPECTED_TORCH', torch.__version__"; then
    echo "Environment is incomplete or differs from $EXPECTED_TORCH; no files were deleted." >&2
    echo "Choose a new environment path, for example:" >&2
    echo 'TECHJAM_VENV="$HOME/techjam-final-cu128" bash run_final.sh' >&2
    exit 3
fi
flock -u 9
exec 9>&-
source "$VENV_DIR/bin/activate"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

python - <<'PY'
import sys
import torch
if not torch.cuda.is_available():
    raise SystemExit("PyTorch cannot access CUDA inside this allocation; stopping without reinstalling.")
print("Python:", sys.version)
print("PyTorch:", torch.__version__)
print("CUDA build:", torch.version.cuda)
print("Visible GPU:", torch.cuda.get_device_name(0))
print("GPU memory GiB:", torch.cuda.get_device_properties(0).total_memory / 1024**3)
PY
python -m pip freeze > "$RUN_DIR/environment.txt"
cp -- techjam_final.py "$RUN_DIR/techjam_final.py"
cp -- "$SCRIPT_SOURCE" "$RUN_DIR/run_final.sh"

echo "=== Small correctness/cache/reference self-tests ==="
python techjam_final.py --self-test --device cuda

echo "=== Full suite ==="
echo "Shapes 1-13: original baseline speedup. Shape 14: FULL accuracy and separate tiled-reference ratio."
STATUS=0
python techjam_final.py --all --device cuda --dtype float32 --output-dir "$RUN_DIR" "$@" || STATUS=$?
echo "Suite exit code: $STATUS (zero means ALL requested cases passed)"
echo "Summary: $RUN_DIR/summary.tsv"
echo "Machine-readable results: $RUN_DIR/summary.json"
echo "Per-case logs, source snapshots and environment are in the same folder."
exit "$STATUS"

