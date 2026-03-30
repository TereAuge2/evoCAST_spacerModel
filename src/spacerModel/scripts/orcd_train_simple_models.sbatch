#!/usr/bin/env bash
#SBATCH --job-name=evoCAST_train
#SBATCH --partition=mit_normal
#SBATCH --time=00:50:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.err

set -euo pipefail

# Match BLAS/PyTorch threads to the Slurm CPU allocation (avoids oversubscription and very slow runs).
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-2}}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-$OMP_NUM_THREADS}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-$OMP_NUM_THREADS}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-$OMP_NUM_THREADS}"

# Usage examples:
#   sbatch src/spacerModel/scripts/orcd_train_simple_models.sbatch
#   sbatch --export=ALL,EMBEDDING=kmer_gc,KMER_K=3,RUN_NAME=my_kmer3_gc src/spacerModel/scripts/orcd_train_simple_models.sbatch
#
# Env vars you can override via sbatch --export:
#   EXCEL: path to exported excel with Split columns
#   SPLIT_COL: Split or SplitRandom
#   EMBEDDING: default kmer_gc_onehot (raw one-hot + GC + k-mer freqs); also onehot | onehot_gc | kmer | kmer_gc
#   KMER_K: 2 | 3 | 4 (dinucleotide when 2 and embedding uses k-mers)
#   RUN_NAME: output subfolder under output/modeling
#   EXTRA_ARGS: any extra args passed to train_models.py

EXCEL="${EXCEL:-output/data_processing/251129_P8_120Complete_dataset_woCtrl_exogenous_with_splits.xlsx}"
SPLIT_COL="${SPLIT_COL:-Split}"
EMBEDDING="${EMBEDDING:-kmer_gc_onehot}"
KMER_K="${KMER_K:-2}"
RUN_NAME="${RUN_NAME:-}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

mkdir -p logs

module load miniforge
mamba activate evoCAST

python -u src/spacerModel/simple_modeling/train_models.py \
  --excel "$EXCEL" \
  --split-col "$SPLIT_COL" \
  --output-dir output/modeling \
  ${RUN_NAME:+--run-name "$RUN_NAME"} \
  --embedding "$EMBEDDING" \
  --kmer-k "$KMER_K" \
  $EXTRA_ARGS

