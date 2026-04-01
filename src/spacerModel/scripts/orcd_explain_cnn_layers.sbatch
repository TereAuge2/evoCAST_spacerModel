#!/usr/bin/env bash
#SBATCH --job-name=evoCAST_explain_cnn
#SBATCH --partition=mit_normal
#SBATCH --time=00:25:00
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

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-4}}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-$OMP_NUM_THREADS}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-$OMP_NUM_THREADS}"

EXCEL="${EXCEL:-output/data_processing/251129_P8_120Complete_dataset_woCtrl_exogenous_with_splits.xlsx}"
SHEET="${SHEET:-0}"
SPLIT_COL="${SPLIT_COL:-Split}"
TARGET_COL="${TARGET_COL:-FE}"

# Point at your saved best CNN checkpoint:
CHECKPOINT="${CHECKPOINT:-}"

# Which rows to explain: test | cv | all
WHICH="${WHICH:-all}"

# 10 random + 10 top-FE + 10 bottom-FE + 10 worst-error (default)
#
# NOTE: bash has a special $GROUPS variable (user's unix group IDs), so do NOT use that name here.
N_PER_GROUP="${N_PER_GROUP:-10}"
SELECT_GROUPS="${SELECT_GROUPS:-random,top_fe,bottom_fe,worst_error}"

OUT_DIR="${OUT_DIR:-output/modeling/cnn_explainability}"
DEVICE="${DEVICE:-auto}"

mkdir -p logs

module load miniforge
mamba activate evoCAST

# Optional CLI flags (override env vars):
#   sbatch .../orcd_explain_cnn_layers.sbatch -- --checkpoint PATH [--which all|cv|test] [--out-dir DIR] [--device auto|cpu|cuda]
while [[ $# -gt 0 ]]; do
  case "$1" in
    --checkpoint)
      if [[ $# -lt 2 ]]; then
        echo "ERROR: --checkpoint requires a value" >&2
        exit 2
      fi
      CHECKPOINT="$2"
      shift 2
      ;;
    --checkpoint=*)
      CHECKPOINT="${1#*=}"
      shift 1
      ;;
    --which)
      if [[ $# -lt 2 ]]; then
        echo "ERROR: --which requires a value" >&2
        exit 2
      fi
      WHICH="$2"
      shift 2
      ;;
    --which=*)
      WHICH="${1#*=}"
      shift 1
      ;;
    --out-dir)
      if [[ $# -lt 2 ]]; then
        echo "ERROR: --out-dir requires a value" >&2
        exit 2
      fi
      OUT_DIR="$2"
      shift 2
      ;;
    --out-dir=*)
      OUT_DIR="${1#*=}"
      shift 1
      ;;
    --device)
      if [[ $# -lt 2 ]]; then
        echo "ERROR: --device requires a value" >&2
        exit 2
      fi
      DEVICE="$2"
      shift 2
      ;;
    --device=*)
      DEVICE="${1#*=}"
      shift 1
      ;;
    --n-per-group|--groups)
      echo "NOTE: $1 is fixed by defaults; set N_PER_GROUP/SELECT_GROUPS env vars instead." >&2
      exit 2
      ;;
    *)
      echo "Unknown arg: $1" >&2
      exit 2
      ;;
  esac
done

if [[ -z "${CHECKPOINT}" ]]; then
  echo "ERROR: CHECKPOINT is empty. Set CHECKPOINT to final_models/CNN_best_model.pt" >&2
  exit 2
fi

python -u -m spacerModel.interpretation.explain_cnn_layers \
  --checkpoint "${CHECKPOINT}" \
  --excel "${EXCEL}" \
  --sheet "${SHEET}" \
  --split-col "${SPLIT_COL}" \
  --target-col "${TARGET_COL}" \
  --which "${WHICH}" \
  --n-per-group "${N_PER_GROUP}" \
  --groups "${SELECT_GROUPS}" \
  --out-dir "${OUT_DIR}" \
  --device "${DEVICE}"

