#!/usr/bin/env bash
#SBATCH --job-name=evoCAST_picking
#SBATCH --partition=mit_normal
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.err

set -euo pipefail

# Submit from the repository root (or set SLURM_SUBMIT_DIR to the repo when submitting).
cd "${SLURM_SUBMIT_DIR:-${PWD}}"
mkdir -p logs

# Match cluster Python env (override with sbatch --export or edit here).
module load miniforge 2>/dev/null || true
if command -v mamba >/dev/null 2>&1; then
  mamba activate evoCAST 2>/dev/null || true
fi

# Override paths via: sbatch --export=ALL,PICK_INPUT=/path/to/ranked.xlsx,...
PICK_INPUT="${PICK_INPUT:-output/inference/ranked_sequences.xlsx}"
PICK_FASTA="${PICK_FASTA:-data/input_fasta_edge.fa}"
PICK_OUTPUT="${PICK_OUTPUT:-output/inference/Model_ranked_spacers_all_genes_picked_for_testing_with_primers.xlsx}"
PICK_GENE="${PICK_GENE:-all}"
PICK_TOP="${PICK_TOP:-3}"
PICK_BOTTOM="${PICK_BOTTOM:-3}"
PICK_RANDOM="${PICK_RANDOM:-3}"
PICK_NCBI_N="${PICK_NCBI_N:-50}"
PICK_RANKED_OUT="${PICK_RANKED_OUT:-output/inference/Model_ranked_spacers_all_genes.csv}"
EXTRA_PICK_ARGS="${EXTRA_PICK_ARGS:-}"

export PYTHONPATH="${PWD}/src"

echo "[sbatch] cwd=$(pwd) PYTHONPATH=${PYTHONPATH}" >&2
echo "[sbatch] picking input=${PICK_INPUT} fasta=${PICK_FASTA} output=${PICK_OUTPUT}" >&2

python -u -m spacerModel.inference.picking \
  --input "${PICK_INPUT}" \
  --fasta "${PICK_FASTA}" \
  --output "${PICK_OUTPUT}" \
  --gene "${PICK_GENE}" \
  --number-top-picked "${PICK_TOP}" \
  --number-bottom-picked "${PICK_BOTTOM}" \
  --number-random-picked "${PICK_RANDOM}" \
  --ncbi-num-primer-pairs "${PICK_NCBI_N}" \
  --output-ranked "${PICK_RANKED_OUT}" \
  ${EXTRA_PICK_ARGS}

echo "[sbatch] done" >&2
