#!/usr/bin/env bash
#SBATCH --job-name=evoCAST_cnn_xgb_sur
#SBATCH --partition=mit_normal
#SBATCH --time=00:05:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.err

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  REPO_ROOT="$(cd "${SLURM_SUBMIT_DIR}" && pwd)"
else
  REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
fi
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:$PYTHONPATH}"
cd "${REPO_ROOT}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-8}}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-$OMP_NUM_THREADS}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-$OMP_NUM_THREADS}"

# CNN checkpoint (same format as genomic inference / explain_cnn_layers).
CHECKPOINT="${CHECKPOINT:-models/CNN_best_model.pt}"
OUT_DIR="${OUT_DIR:-output/interpretation/cnn_surrogate_xgboost}"
EXCEL="${EXCEL:-output/data_processing/251129_P8_120Complete_dataset_woCtrl_exogenous_with_splits.xlsx}"
SHEET="${SHEET:-0}"
TARGET_COL="${TARGET_COL:-FE}"
SPLIT_COL="${SPLIT_COL:-Split}"
# Set NO_REAL_EVAL=1 to skip measured CV evaluation (CNN vs XGB vs FE).
NO_REAL_EVAL="${NO_REAL_EVAL:-0}"
N_SYNTHETIC="${N_SYNTHETIC:-100000}"
DEVICE="${DEVICE:-auto}"
BATCH_SIZE="${BATCH_SIZE:-512}"
RANDOM_STATE="${RANDOM_STATE:-42}"
SAMPLE_ROWS_CSV="${SAMPLE_ROWS_CSV:-2000}"
XGB_N_JOBS="${XGB_N_JOBS:-${SLURM_CPUS_PER_TASK:-8}}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

mkdir -p logs

echo "=== evoCAST cnn_surrogate_xgboost job ${SLURM_JOB_ID:-local} $(date -Is) ==="
echo "REPO_ROOT=${REPO_ROOT}"
echo "CHECKPOINT=${CHECKPOINT}"
echo "OUT_DIR=${OUT_DIR}"
echo "N_SYNTHETIC=${N_SYNTHETIC} DEVICE=${DEVICE} BATCH_SIZE=${BATCH_SIZE}"
echo "NO_REAL_EVAL=${NO_REAL_EVAL} EXCEL=${EXCEL}"
echo "XGB_N_JOBS=${XGB_N_JOBS} SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-}"
echo "================================================================"

module load miniforge
mamba activate evoCAST

while [[ $# -gt 0 ]]; do
  case "$1" in
    --checkpoint)
      CHECKPOINT="$2"
      shift 2
      ;;
    --checkpoint=*)
      CHECKPOINT="${1#*=}"
      shift 1
      ;;
    --out-dir|--output-dir)
      OUT_DIR="$2"
      shift 2
      ;;
    --out-dir=*|--output-dir=*)
      OUT_DIR="${1#*=}"
      shift 1
      ;;
    --n-synthetic)
      N_SYNTHETIC="$2"
      shift 2
      ;;
    --n-synthetic=*)
      N_SYNTHETIC="${1#*=}"
      shift 1
      ;;
    --device)
      DEVICE="$2"
      shift 2
      ;;
    --device=*)
      DEVICE="${1#*=}"
      shift 1
      ;;
    *)
      echo "Unknown arg: $1 (use env vars CHECKPOINT, OUT_DIR, … or --checkpoint PATH)" >&2
      exit 2
      ;;
  esac
done

REAL_EVAL_CLI=()
if [[ "${NO_REAL_EVAL}" == "1" ]]; then
  REAL_EVAL_CLI=(--no-real-eval)
else
  REAL_EVAL_CLI=(--excel "${EXCEL}" --sheet "${SHEET}" --target-col "${TARGET_COL}" --split-col "${SPLIT_COL}")
fi

python -u -m spacerModel.interpretation.cnn_surrogate_xgboost \
  --checkpoint "${CHECKPOINT}" \
  --output-dir "${OUT_DIR}" \
  --n-synthetic "${N_SYNTHETIC}" \
  --random-state "${RANDOM_STATE}" \
  --device "${DEVICE}" \
  --batch-size "${BATCH_SIZE}" \
  --sample-rows-csv "${SAMPLE_ROWS_CSV}" \
  --xgb-n-jobs "${XGB_N_JOBS}" \
  "${REAL_EVAL_CLI[@]}" \
  $EXTRA_ARGS
