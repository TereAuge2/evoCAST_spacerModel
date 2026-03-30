#!/usr/bin/env bash
# Source after REPO_ROOT points at the repo root (same pattern as orcd_*.sbatch).
# Loads ${REPO_ROOT}/.wandb_env then ${HOME}/.wandb_env if they exist (gitignore the repo file).
# shellcheck shell=bash
if [[ -z "${REPO_ROOT:-}" ]]; then
  echo "load_wandb_env.sh: set REPO_ROOT to the repository root before sourcing" >&2
  return 1 2>/dev/null || exit 1
fi
for _wandb_f in "${REPO_ROOT}/.wandb_env" "${HOME}/.wandb_env"; do
  if [[ -f "${_wandb_f}" ]]; then
    set -a
    # shellcheck disable=SC1091
    source "${_wandb_f}"
    set +a
  fi
done
unset _wandb_f
