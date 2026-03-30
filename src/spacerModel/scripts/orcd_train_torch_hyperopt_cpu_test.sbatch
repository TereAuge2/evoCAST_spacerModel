#!/usr/bin/env bash
# Same CNN hyperparameter smoke test as orcd_train_torch_hyperopt_gpu_test.sbatch, but CPU-only
# (mit_normal). Use when GPU queue/partition is unavailable or the model is fast on CPU.
#
# Submit from repo root:
#   sbatch src/spacerModel/scripts/orcd_train_torch_hyperopt_cpu_test.sbatch
#
# Overrides:
#   sbatch --export=ALL,N_ITER=8,RUN_NAME=my_cnn_wandb_try src/spacerModel/scripts/orcd_train_torch_hyperopt_cpu_test.sbatch
#
# W&B: put API key once in repo-root .wandb_env (copy from .wandb_env.example); gitignored.
#      Or ~/.wandb_env on the cluster. Optional: USE_WANDB=0 to skip logging.

#SBATCH --job-name=evoCAST_cnn_hopt_cpu
#SBATCH --partition=mit_normal
#SBATCH --time=00:15:00
# 4 CPUs, 8 GB RAM (override: sbatch --cpus-per-task=N --mem=...)
#SBATCH --cpus-per-task=4
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

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-4}}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-$OMP_NUM_THREADS}"

EXCEL="${EXCEL:-output/data_processing/251129_P8_120Complete_dataset_woCtrl_exogenous_with_splits.xlsx}"
SPLIT_COL="${SPLIT_COL:-Split}"
EMBEDDING="${EMBEDDING:-onehot}"
KMER_K="${KMER_K:-2}"
RUN_NAME="${RUN_NAME:-}"
MODEL_KIND="${MODEL_KIND:-cnn}"
DEVICE="${DEVICE:-cpu}"
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
