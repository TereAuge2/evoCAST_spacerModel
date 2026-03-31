#!/usr/bin/env bash
#SBATCH --job-name=evoCAST_tune_svr
#SBATCH --partition=mit_normal
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.err

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Slurm may run this file from a spool copy; SCRIPT_DIR/../../.. can be wrong. Prefer submit cwd.
if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  REPO_ROOT="$(cd "${SLURM_SUBMIT_DIR}" && pwd)"
else
  REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
fi
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:$PYTHONPATH}"
cd "${REPO_ROOT}"

# shellcheck disable=SC1091
source "${REPO_ROOT}/src/spacerModel/scripts/load_wandb_env.sh"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-8}}"

# W&B on by default (USE_WANDB=0 to skip). Loads REPO_ROOT/.wandb_env if present.
# RUN_NAME empty -> job id + timestamp under output/modeling/svr_hyperparameter_tune/
EXCEL="${EXCEL:-output/data_processing/251129_P8_120Complete_dataset_woCtrl_exogenous_with_splits.xlsx}"
SPLIT_COL="${SPLIT_COL:-Split}"
EMBEDDING="${EMBEDDING:-kmer_gc_onehot}"
KMER_K="${KMER_K:-2}"
RUN_NAME="${RUN_NAME:-}"
N_ITER="${N_ITER:-200}"
MODELS="${MODELS:-svr_rbf,nusvr_rbf,kernel_ridge_rbf}"
USE_WANDB="${USE_WANDB:-1}"
# RAW_ONEHOT=0 -> --no-raw-onehot
RAW_ONEHOT="${RAW_ONEHOT:-}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

mkdir -p logs

module load miniforge
mamba activate evoCAST

WANDB_CLI=()
if [[ "${USE_WANDB}" == "1" ]]; then
  WANDB_CLI=(--use-wandb)
fi
RAW_CLI=()
if [[ "${RAW_ONEHOT}" == "0" ]]; then
  RAW_CLI=(--no-raw-onehot)
fi

python -u -m spacerModel.training.tune_svr_kernel \
  --excel "$EXCEL" \
  --split-col "$SPLIT_COL" \
  --embedding "$EMBEDDING" \
  --kmer-k "$KMER_K" \
  --run-name "$RUN_NAME" \
  --n-iter "$N_ITER" \
  --models "$MODELS" \
  "${WANDB_CLI[@]}" \
  "${RAW_CLI[@]}" \
  $EXTRA_ARGS
