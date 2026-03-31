#!/usr/bin/env bash
# MLP hyperparameter search (torch_hyperopt) on CPU — mit_normal. No GPU queue.
# W&B on by default (USE_WANDB=1). Set USE_WANDB=0 to skip logging.
#
# Default embedding kmer_gc_onehot (see train_models.build_feature_matrix):
#   sparse stack of  (DNA one-hot) + (k-mer frequency columns for KMER_K) + (GC content).
# With --raw-onehot default ON (RAW_ONEHOT unset or non-0): that full design is materialized to
# one dense numeric matrix (one-hot + k-mers + GC) for the MLP. RAW_ONEHOT=0 uses SVD on the
# sparse blocks instead of full dense features.
#
# Submit from repo root:
#   sbatch src/spacerModel/scripts/orcd_train_torch_hyperopt_mlp_cpu.sbatch
#
# Examples:
#   sbatch --export=ALL,N_ITER=40,RUN_NAME=my_mlp_run src/spacerModel/scripts/orcd_train_torch_hyperopt_mlp_cpu.sbatch
#   sbatch --export=ALL,USE_WANDB=0 src/spacerModel/scripts/orcd_train_torch_hyperopt_mlp_cpu.sbatch

#SBATCH --job-name=evoCAST_mlp_hopt
#SBATCH --partition=mit_normal
#SBATCH --time=02:00:00
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
EMBEDDING="${EMBEDDING:-kmer_gc_onehot}"
KMER_K="${KMER_K:-2}"
RUN_NAME="${RUN_NAME:-}"
DEVICE="${DEVICE:-cpu}"
N_ITER="${N_ITER:-100}"
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
  --kmer-k "$KMER_K" \
  ${RUN_NAME:+--run-name "$RUN_NAME"} \
  --model-kind mlp \
  --device "$DEVICE" \
  --n-iter "$N_ITER" \
  "${WANDB_CLI[@]}" \
  "${RAW_CLI[@]}" \
  $EXTRA_ARGS
