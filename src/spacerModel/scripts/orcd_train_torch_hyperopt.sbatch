#!/usr/bin/env bash
# SBATCH: n_iter × CV is usually well under an hour for this stack; raise --time/--mem if needed.
#SBATCH --job-name=evoCAST_torch_hopt
#SBATCH --partition=mit_normal
#SBATCH --time=02:00:00
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

# RUN_NAME empty -> Python uses ${SLURM_JOB_ID}_<timestamp> under CNN_hyperparameter_tune or MLP_hyperparameter_tune
# USE_WANDB=1 -> pass --use-wandb. W&B key: repo .wandb_env or ~/.wandb_env (see .wandb_env.example)
EXCEL="${EXCEL:-output/data_processing/251129_P8_120Complete_dataset_woCtrl_exogenous_with_splits.xlsx}"
SPLIT_COL="${SPLIT_COL:-Split}"
EMBEDDING="${EMBEDDING:-kmer_gc_onehot}"
KMER_K="${KMER_K:-2}"
RUN_NAME="${RUN_NAME:-}"
MODEL_KIND="${MODEL_KIND:-mlp}"
DEVICE="${DEVICE:-auto}"
N_ITER="${N_ITER:-40}"
USE_WANDB="${USE_WANDB:-0}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
# RAW_ONEHOT=0 -> pass --no-raw-onehot (SVD on sparse); unset/1 -> Python default (dense)
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
  --run-name "$RUN_NAME" \
  --model-kind "$MODEL_KIND" \
  --device "$DEVICE" \
  --n-iter "$N_ITER" \
  "${WANDB_CLI[@]}" \
  "${RAW_CLI[@]}" \
  $EXTRA_ARGS
