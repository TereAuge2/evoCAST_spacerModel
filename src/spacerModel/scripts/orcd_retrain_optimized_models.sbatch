#!/usr/bin/env bash
#SBATCH --job-name=evoCAST_retrain_optimized
#SBATCH --partition=mit_normal
#SBATCH --time=00:15:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.err

set -euo pipefail

# Slurm may run this file from a spool copy; use submit cwd as repo root.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  REPO_ROOT="$(cd "${SLURM_SUBMIT_DIR}" && pwd)"
else
  REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
fi
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:$PYTHONPATH}"
cd "${REPO_ROOT}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-4}}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-$OMP_NUM_THREADS}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-$OMP_NUM_THREADS}"

EXCEL="${EXCEL:-output/data_processing/251129_P8_120Complete_dataset_woCtrl_exogenous_with_splits.xlsx}"
SHEET="${SHEET:-0}"
SPLIT_COL="${SPLIT_COL:-Split}"
TARGET_COL="${TARGET_COL:-FE}"
EMBEDDING="${EMBEDDING:-kmer_gc_onehot}"
KMER_K="${KMER_K:-2}"
SVD_COMPONENTS="${SVD_COMPONENTS:-82}"

# Point these at your *best_hyperparameters*.json files (the ones you pasted above).
CNN_JSON="${CNN_JSON:-output/modeling/CNN_hyperparameter_tune/11209673_20260330_202342_CNN/best_hyperparameters.json}"
MLP_JSON="${MLP_JSON:-output/modeling/MLP_hyperparameter_tune/11203389_20260330_170824_MLP/best_hyperparameters.json}"
XGB_JSON="${XGB_JSON:-output/modeling/xgboost_hyperparameter_tune/11203391_20260330_170825_XGBOOST/best_hyperparameters.json}"
SVR_JSON="${SVR_JSON:-output/modeling/svr_hyperparameter_tune/11203390_20260330_170824_SVR/best_hyperparameters_svr_rbf.json}"
NUSVR_JSON="${NUSVR_JSON:-output/modeling/svr_hyperparameter_tune/11203390_20260330_170824_SVR/best_hyperparameters_nusvr_rbf.json}"
KRR_JSON="${KRR_JSON:-output/modeling/svr_hyperparameter_tune/11203390_20260330_170824_SVR/best_hyperparameters_kernel_ridge_rbf.json}"

# Optional: if set to a path, we will load that instead of the per-model JSONs.
BUNDLE_JSON="${BUNDLE_JSON:-}"

# Set to 1 to run the extra "evaluate_test" mode.
EVALUATE_TEST="${EVALUATE_TEST:-0}"

RUN_NAME="${RUN_NAME:-}"
OUT_DIR="${OUT_DIR:-output/modeling/optimized_retrain}"

mkdir -p logs

module load miniforge
mamba activate evoCAST

CMD=(python -u -m spacerModel.training.retrain_optimized_models
  --excel "${EXCEL}"
  --sheet "${SHEET}"
  --split-col "${SPLIT_COL}"
  --target-col "${TARGET_COL}"
  --embedding "${EMBEDDING}"
  --kmer-k "${KMER_K}"
  --svd-components "${SVD_COMPONENTS}"
  --output-dir "${OUT_DIR}"
  --evaluate-test
)

if [[ -n "${RUN_NAME}" ]]; then
  CMD+=(--run-name "${RUN_NAME}")
fi

if [[ -n "${BUNDLE_JSON}" ]]; then
  CMD+=(--bundle "${BUNDLE_JSON}")
else
  CMD+=(--cnn-json "${CNN_JSON}" --mlp-json "${MLP_JSON}" --xgb-json "${XGB_JSON}"
        --svr-json "${SVR_JSON}" --nusvr-json "${NUSVR_JSON}" --krr-json "${KRR_JSON}")
fi

if [[ "${EVALUATE_TEST}" == "1" ]]; then
  CMD+=(--evaluate-test)
fi

echo "Running:"
printf ' %q' "${CMD[@]}"
echo

PYTHONUNBUFFERED=1 "${CMD[@]}"

