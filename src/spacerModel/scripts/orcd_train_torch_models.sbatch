#!/usr/bin/env bash
#SBATCH --job-name=evoCAST_torch
#SBATCH --partition=mit_normal
#SBATCH --time=00:20:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.err
# Optional GPU (uncomment on clusters that provide it):
# #SBATCH --partition=mit_normal_gpu
# #SBATCH --gres=gpu:1

set -euo pipefail

# Slurm copies this script to /var/spool/.../slurm_script — use submit cwd as repo root.
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

# Defaults: CNN, raw one-hot (4ch), Split column.
# Best hyperparameters from torch_random_search_Split_cnn.csv trial 5 (lowest mean_rmse),
# run 11074812_20260327_162051 — override via EXTRA_ARGS if you re-tune.
EXCEL="${EXCEL:-output/data_processing/251129_P8_120Complete_dataset_woCtrl_exogenous_with_splits.xlsx}"
SPLIT_COL="${SPLIT_COL:-Split}"
EMBEDDING="${EMBEDDING:-onehot}"
KMER_K="${KMER_K:-2}"
RUN_NAME="${RUN_NAME:-Split_onehot_raw_cnn_bestHopt}"
MODEL_KIND="${MODEL_KIND:-cnn}"
DEVICE="${DEVICE:-auto}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

mkdir -p logs

module load miniforge
mamba activate evoCAST

python -u -m spacerModel.models.train_torch \
  --excel "$EXCEL" \
  --split-col "$SPLIT_COL" \
  --embedding "$EMBEDDING" \
  --kmer-k "$KMER_K" \
  --raw-onehot \
  --output-dir output/modeling \
  --run-name "$RUN_NAME" \
  --model-kind "$MODEL_KIND" \
  --device "$DEVICE" \
  --lr 9.494586914206585e-05 \
  --weight-decay 1.2923469377951069e-05 \
  --batch-size 32 \
  --dropout 0.1 \
  --max-epochs 216 \
  --patience 13 \
  --min-delta 3.3089181258956105e-08 \
  --cnn-conv1-channels 98 \
  --cnn-conv2-channels 78 \
  --cnn-head-hidden 256 \
  --cnn-kernel1 3 \
  --cnn-kernel2 5 \
  --cnn-pool-out 16 \
  $EXTRA_ARGS
