#!/usr/bin/env bash
#SBATCH --job-name=evoCAST_cnn_xgb_sur
#SBATCH --partition=mit_normal
# pred_contribs scales with boosted rounds × explained rows (default 8192×2); large
# n_estimators (e.g. 3000) can take tens of minutes on 2 CPUs. Pass extra Python flags either:
#   EXTRA_ARGS='--shap-contrib-max-rounds 400 --shap-synthetic-max-samples 2048' sbatch …/orcd_cnn_surrogate_xgboost.sbatch
# or append them after the sbatch script path (unknown flags are forwarded to python -m …):
#   sbatch …/orcd_cnn_surrogate_xgboost.sbatch --shap-contrib-max-rounds 400 --shap-synthetic-max-samples 2048
#SBATCH --time=01:40:00
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

# Thread caps: without Slurm, default OpenMP/BLAS to 1 thread per library. Otherwise
# XGBoost n_jobs>1 multiplies threads (libgomp: Thread creation failed on login nodes).
if [[ -n "${SLURM_CPUS_PER_TASK:-}" ]]; then
  export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${SLURM_CPUS_PER_TASK}}"
else
  export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
fi
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-$OMP_NUM_THREADS}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-$OMP_NUM_THREADS}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-$OMP_NUM_THREADS}"

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
# Match Slurm CPUs when scheduled; modest default when running `bash …sbatch` locally.
XGB_N_JOBS="${XGB_N_JOBS:-${SLURM_CPUS_PER_TASK:-4}}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
# Any argv not consumed by the case below is appended to the python invocation (quoted).
PASS_ARGS=()

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
      # Forward any other flags to `python -m spacerModel.interpretation.cnn_surrogate_xgboost`
      # (e.g. --shap-contrib-max-rounds 400 --shap-synthetic-max-samples 2048).
      PASS_ARGS+=("$1")
      shift 1
      ;;
  esac
done

if ((${#PASS_ARGS[@]} > 0)); then
  echo "Forwarded to python: ${PASS_ARGS[*]}"
fi

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
  ${EXTRA_ARGS} \
  "${PASS_ARGS[@]}"

# Also regenerate corrected/aggregated SHAP plots from raw CSV outputs (fast, no contrib recompute).
python -u -m spacerModel.interpretation.replot_surrogate_shap \
  --output-dir "${OUT_DIR}" \
  --stems synthetic,real_cv
