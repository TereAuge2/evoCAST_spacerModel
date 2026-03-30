#!/usr/bin/env bash
# Small CNN hyperparameter search on GPU + wandb (smoke test; increase N_ITER for real runs).
#
# Submit from repo root:
#   sbatch src/spacerModel/scripts/orcd_train_torch_hyperopt_gpu_test.sbatch
#
# Overrides (examples):
#   sbatch --export=ALL,N_ITER=8,RUN_NAME=my_cnn_wandb_try src/spacerModel/scripts/orcd_train_torch_hyperopt_gpu_test.sbatch
#
# W&B: repo-root .wandb_env (see .wandb_env.example) or ~/.wandb_env — gitignored, set once.

#SBATCH --job-name=evoCAST_cnn_hopt_gpu
# ORCD Engaging: GPU partition is mit_normal_gpu (not "gpu"). Override: sbatch --partition=... ...
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:1
# Minimal request (single CNN CV ~tens of seconds on CPU); raise --time/--mem for large N_ITER.
#SBATCH --time=00:45:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.err

set -euo pipefail

# Slurm copies this script to /var/spool/slurmd/.../slurm_script — dirname is not the repo.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  REPO_ROOT="$(cd "${SLURM_SUBMIT_DIR}" && pwd)"
else
  REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
fi
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:$PYTHONPATH}"
cd "${REPO_ROOT}"

# shellcheck disable=SC1091
source "${REPO_ROOT}/src/spacerModel/scripts/load_wandb_env.sh"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-2}}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-$OMP_NUM_THREADS}"

EXCEL="${EXCEL:-output/data_processing/251129_P8_120Complete_dataset_woCtrl_exogenous_with_splits.xlsx}"
SPLIT_COL="${SPLIT_COL:-Split}"
# CNN requires one-hot only (not kmer_gc_onehot).
EMBEDDING="${EMBEDDING:-onehot}"
KMER_K="${KMER_K:-2}"
RUN_NAME="${RUN_NAME:-}"
MODEL_KIND="${MODEL_KIND:-cnn}"
DEVICE="${DEVICE:-cuda}"
# Few trials for a pipeline test; bump to 20–40 for exploratory search.
N_ITER="${N_ITER:-5}"
USE_WANDB="${USE_WANDB:-1}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
RAW_ONEHOT="${RAW_ONEHOT:-}"

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

python -u -m spacerModel.models.torch_hyperopt \
  --excel "$EXCEL" \
  --split-col "$SPLIT_COL" \
  --embedding "$EMBEDDING" \
  --kmer-k "$KMER_K" \
  ${RUN_NAME:+--run-name "$RUN_NAME"} \
  --model-kind "$MODEL_KIND" \
  --device "$DEVICE" \
  --n-iter "$N_ITER" \
  "${WANDB_CLI[@]}" \
  "${RAW_CLI[@]}" \
  $EXTRA_ARGS
