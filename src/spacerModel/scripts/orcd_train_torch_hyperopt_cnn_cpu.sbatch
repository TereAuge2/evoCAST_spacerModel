#!/usr/bin/env bash
# CNN hyperparameter search (torch_hyperopt) on CPU — mit_normal. No GPU queue.
# W&B on by default (USE_WANDB=1). Set USE_WANDB=0 to skip logging.
#
# Features (matches torch_hyperopt / train_models):
#   --embedding onehot  →  DNA one-hot (A,C,G,T) only. No k-mer frequency block, no GC column.
#   There is no --kmer-k in this script: k is irrelevant for onehot (build_feature_matrix ignores it).
#   --raw-onehot (default ON): dense L×4 flat input to the CNN. RAW_ONEHOT=0 → SVD on sparse one-hot instead.
#   Do not set EMBEDDING to kmer* for CNN; torch_hyperopt exits unless embedding is onehot.
#
# Submit from repo root:
#   sbatch src/spacerModel/scripts/orcd_train_torch_hyperopt_cnn_cpu.sbatch
#
# Examples:
#   sbatch --export=ALL,N_ITER=40,RUN_NAME=my_cnn_run src/spacerModel/scripts/orcd_train_torch_hyperopt_cnn_cpu.sbatch
#   sbatch --export=ALL,USE_WANDB=0 src/spacerModel/scripts/orcd_train_torch_hyperopt_cnn_cpu.sbatch

#SBATCH --job-name=evoCAST_cnn_hopt
#SBATCH --partition=mit_normal
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=4
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

# shellcheck disable=SC1091
source "${REPO_ROOT}/src/spacerModel/scripts/load_wandb_env.sh"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-4}}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-$OMP_NUM_THREADS}"

EXCEL="${EXCEL:-output/data_processing/251129_P8_120Complete_dataset_woCtrl_exogenous_with_splits.xlsx}"
SPLIT_COL="${SPLIT_COL:-Split}"
EMBEDDING="${EMBEDDING:-onehot}"
RUN_NAME="${RUN_NAME:-}"
DEVICE="${DEVICE:-cpu}"
N_ITER="${N_ITER:-200}"
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

python -u -m spacerModel.training.torch_hyperopt \
  --excel "$EXCEL" \
  --split-col "$SPLIT_COL" \
  --embedding "$EMBEDDING" \
  ${RUN_NAME:+--run-name "$RUN_NAME"} \
  --model-kind cnn \
  --device "$DEVICE" \
  --n-iter "$N_ITER" \
  "${WANDB_CLI[@]}" \
  "${RAW_CLI[@]}" \
  $EXTRA_ARGS
