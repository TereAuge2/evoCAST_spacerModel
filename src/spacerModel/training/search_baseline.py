"""
Baseline (default hyperparameter) CV rows for randomized searches.

Prepends one trial with library defaults so sweeps always include a reference point.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import cross_val_score


def sklearn_baseline_cv_row(
    estimator: Any,
    param_dict: dict[str, Any],
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    gkf: Any,
    df_search: pd.DataFrame,
    *,
    scoring: str = "neg_root_mean_squared_error",
    n_jobs: int = 1,
) -> dict[str, Any]:
    """
    One CV mean RMSE row with the same columns as ``df_search`` (RandomizedSearchCV trials).
    ``iter`` is 0; caller should shift search trial iters to 1..n via
    :func:`prepend_baseline_sklearn_trials_df`.
    """
    est = clone(estimator)
    est.set_params(**param_dict)
    scores = cross_val_score(
        est,
        X,
        y,
        cv=gkf,
        groups=groups,
        scoring=scoring,
        n_jobs=n_jobs,
    )
    neg = scoring.startswith("neg_")
    mean_rmse = float(-scores.mean()) if neg else float(scores.mean())
    std_rmse = float(scores.std())
    ep = est.get_params()
    row: dict[str, Any] = {
        "iter": 0,
        "mean_rmse": mean_rmse,
        "std_rmse": std_rmse,
    }
    for c in df_search.columns:
        if c in ("iter", "mean_rmse", "std_rmse"):
            continue
        row[c] = ep.get(c)
    return row


def prepend_baseline_sklearn_trials_df(df_search: pd.DataFrame, baseline_row: dict[str, Any]) -> pd.DataFrame:
    """Insert baseline (iter=0) and renumber RandomizedSearchCV iters to 1..n."""
    df = df_search.copy()
    df["iter"] = df["iter"].astype(int) + 1
    return pd.concat([pd.DataFrame([baseline_row]), df], ignore_index=True)
