#!/usr/bin/env bash
#SBATCH --job-name=evoCAST_kmer_gc_onehot_raw_SR
#SBATCH --partition=mit_normal
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --array=0-2
#SBATCH --output=logs/slurm-%x-%A_%a.out
#SBATCH --error=logs/slurm-%x-%A_%a.err

set -euo pipefail

# Same sweep as orcd_train_models_array.sbatch, but CV uses SplitRandom (--split-col SplitRandom).
# kmer_gc_onehot + --raw-onehot for k in {2,3,4}.
#
# Submit:
#   sbatch src/spacerModel/scripts/orcd_train_models_array_splitrandom.sbatch
#
# Optional env overrides:
#   EXCEL, OUT_BASE, EXTRA_ARGS

EXCEL="${EXCEL:-output/data_processing/251129_P8_120Complete_dataset_woCtrl_exogenous_with_splits.xlsx}"
OUT_BASE="${OUT_BASE:-output/modeling}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

mkdir -p logs

module load miniforge
mamba activate evoCAST

K_LIST=(2 3 4)
task_id="${SLURM_ARRAY_TASK_ID}"
KMER_K="${K_LIST[$task_id]}"

stamp="$(date +%Y%m%d_%H%M%S)"
RUN_NAME="SplitRandom__kmer_gc_onehot_raw__k${KMER_K}__A${SLURM_ARRAY_JOB_ID}_T${task_id}__${stamp}"

python -u src/spacerModel/simple_modeling/train_models.py \
  --excel "$EXCEL" \
  --split-col SplitRandom \
  --output-dir "$OUT_BASE" \
  --run-name "$RUN_NAME" \
  --embedding kmer_gc_onehot \
  --kmer-k "$KMER_K" \
  --raw-onehot \
  $EXTRA_ARGS
