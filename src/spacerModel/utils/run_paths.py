"""Standard output folders for hyperparameter tuning runs (Slurm job ID prefixes)."""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

# Subfolders under output/modeling/
HYPEROPT_SUBDIR_CNN = "CNN_hyperparameter_tune"
HYPEROPT_SUBDIR_MLP = "MLP_hyperparameter_tune"
HYPEROPT_SUBDIR_SVR = "svr_hyperparameter_tune"
HYPEROPT_SUBDIR_XGBOOST = "xgboost_hyperparameter_tune"


def slurm_job_id_prefix() -> str:
    """Batch job id for folder names: array job uses ARRAY_JOB_ID_TASK_ID, else SLURM_JOB_ID."""
    aj = os.environ.get("SLURM_ARRAY_JOB_ID", "").strip()
    task = os.environ.get("SLURM_ARRAY_TASK_ID", "").strip()
    if aj and task != "":
        return f"{aj}_{task}"
    jid = os.environ.get("SLURM_JOB_ID", "").strip()
    return jid if jid else "local"


def default_run_name(model_kind: str | None = None) -> str:
    """
    Default run folder name: ``<job_id>_<timestamp>`` plus optional ``_<KIND>``
    (e.g. ``12345_20260327_143022_CNN``).
    """
    base = f"{slurm_job_id_prefix()}_{datetime.now():%Y%m%d_%H%M%S}"
    if model_kind is None:
        return base
    mk = str(model_kind).strip()
    if not mk:
        return base
    label = {
        "cnn": "CNN",
        "mlp": "MLP",
        "svr": "SVR",
        "xgboost": "XGBOOST",
    }.get(mk.lower())
    if label is None:
        label = mk.upper().replace(" ", "_")
    return f"{base}_{label}"


def default_hyperopt_output_dir(modeling_root: str | Path, tune_kind: str) -> Path:
    """
    tune_kind: 'cnn' | 'mlp' | 'svr' | 'xgboost'
    -> output/modeling/<CNN_hyperparameter_tune|...>/
    """
    root = Path(modeling_root)
    key = tune_kind.strip().lower()
    sub = {
        "cnn": HYPEROPT_SUBDIR_CNN,
        "mlp": HYPEROPT_SUBDIR_MLP,
        "svr": HYPEROPT_SUBDIR_SVR,
        "xgboost": HYPEROPT_SUBDIR_XGBOOST,
    }.get(key)
    if sub is None:
        raise ValueError(f"Unknown tune_kind: {tune_kind}")
    return root / sub
