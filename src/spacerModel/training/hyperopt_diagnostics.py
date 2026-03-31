"""
Hyperparameter search diagnostics: RMSE vs each explored hyperparameter, and shared plot style
(aligned with regression_scatter_plot hexbin aesthetics — sans-serif, grey/blue palette).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def apply_hyperopt_plot_style() -> None:
    mpl.rcParams.update(mpl.rcParamsDefault)
    mpl.rcParams["font.family"] = "sans-serif"
    mpl.rcParams["font.size"] = 14
    mpl.rcParams["axes.facecolor"] = "white"


# Colors echo regression hexbin (grey → dark blue)
_ACCENT = "#0d1b2a"
_GRID = "#d3d3d3"


def _is_categorical_series(s: pd.Series) -> bool:
    """
    True for small discrete grids (batch size, kernel {3,5,7}, dropout levels).

    False for sampled numeric scales (log-uniform C/gamma/lr, many distinct ints in a wide range)
    so the x-axis can be continuous (scatter of trials).

    Sklearn ``cv_results_`` tables often use ``object`` dtype for floats; those are coerced
    so they are not misclassified as categorical.
    """
    v = s.dropna()
    if v.empty:
        return True
    if pd.api.types.is_bool_dtype(v):
        return True

    # Object columns may still be numeric (common for sklearn param DataFrames)
    if v.dtype == object:
        vc = pd.to_numeric(v, errors="coerce")
        if vc.notna().all() and len(vc) == len(v):
            v = vc
        else:
            return True

    nuniq = int(v.nunique())
    if nuniq <= 1:
        return True

    if pd.api.types.is_integer_dtype(v):
        iv = v.astype(np.int64, copy=False)
        span = int(iv.max()) - int(iv.min())
        if nuniq >= 10 or span > 50:
            return False
        return True

    if pd.api.types.is_float_dtype(v):
        xf = np.asarray(v, dtype=float)
        n_round = len(np.unique(np.round(xf, 8)))
        # Need ≥3 distinct values so tiny searches (e.g. 3 trials) still get a scatter axis;
        # ≥4 was too strict and forced all floats into bar charts when n_iter was small.
        if nuniq >= 3 or n_round >= 3:
            return False
        return True

    return False


def _format_tick_label(x: Any) -> str:
    """Short numeric labels on categorical bar x-axis (avoid 7+ decimal clutter)."""
    try:
        xf = float(x)
        if not np.isfinite(xf):
            return str(x)
        ax_ = abs(xf)
        if ax_ >= 1000 or (ax_ >= 1.0 and ax_ < 1000.0):
            return f"{xf:.4g}"
        if ax_ >= 0.01:
            return f"{xf:.4g}"
        return f"{xf:.6g}"
    except (TypeError, ValueError):
        return str(x)


def _plot_continuous_hyperparam_vs_rmse(
    ax: Any,
    x: np.ndarray,
    y: np.ndarray,
    *,
    param_col: str,
) -> None:
    """Scatter (each trial); log-x if positive values span orders of magnitude."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if x.size == 0:
        return

    ax.scatter(
        x,
        y,
        c=_ACCENT,
        edgecolors="black",
        linewidths=0.4,
        s=40,
        alpha=0.85,
        zorder=3,
    )

    ax.axhline(float(np.min(y)), color="grey", linestyle=":", linewidth=0.8, alpha=0.7)
    ax.set_xlabel(param_col, fontsize=14)
    pos = x[x > 0]
    if pos.size >= 3 and float(np.nanmax(pos)) / float(np.nanmin(pos)) > 50.0:
        ax.set_xscale("log")


def plot_hyperparameter_vs_rmse(
    df: pd.DataFrame,
    *,
    param_col: str,
    rmse_col: str = "mean_rmse",
    outpath: Path,
    title: str | None = None,
) -> None:
    """
    Categorical: bar = mean RMSE per level, jittered dots = each trial.

    Continuous (sampled reals / many distinct ints): y = CV RMSE per trial, x = hyperparameter;
    scatter only. Log-scaled x when positive values span ~>50×.
    """
    apply_hyperopt_plot_style()
    sub = df[[param_col, rmse_col]].dropna().copy()
    if sub.empty:
        return

    fig, ax = plt.subplots(figsize=(18.0 / 2.54, 12.0 / 2.54))
    t = title or f"RMSE vs {param_col}"

    if _is_categorical_series(sub[param_col]):
        agg = sub.groupby(param_col, dropna=False)[rmse_col].mean().sort_values(ascending=True)
        x_labels = [_format_tick_label(x) for x in agg.index.tolist()]
        x_pos = np.arange(len(agg))
        cat_to_i = {k: i for i, k in enumerate(agg.index)}
        ax.bar(x_pos, agg.values, color=_GRID, edgecolor=_ACCENT, linewidth=0.8, zorder=2, label="mean")
        rng = np.random.default_rng(42)
        xs: list[float] = []
        ys: list[float] = []
        for _, row in sub.iterrows():
            k = row[param_col]
            if k not in cat_to_i:
                continue
            i = cat_to_i[k]
            xs.append(float(i) + float(rng.uniform(-0.32, 0.32)))
            ys.append(float(row[rmse_col]))
        ax.scatter(
            xs,
            ys,
            c=_ACCENT,
            edgecolors="black",
            linewidths=0.35,
            s=28,
            alpha=0.9,
            zorder=5,
            label="trials",
        )
        ax.set_xticks(x_pos)
        ax.set_xticklabels(x_labels, rotation=90, ha="center", fontsize=11)
        if len(xs) > 0:
            ax.legend(loc="upper right", fontsize=9, frameon=True)
    else:
        x_num = pd.to_numeric(sub[param_col], errors="coerce")
        yv = sub[rmse_col].to_numpy(dtype=float)
        xm = np.asarray(x_num, dtype=float)
        ok = np.isfinite(xm) & np.isfinite(yv)
        _plot_continuous_hyperparam_vs_rmse(
            ax,
            xm[ok],
            yv[ok],
            param_col=param_col,
        )

    ax.set_ylabel("mean RMSE (CV)", fontsize=14)
    ax.set_title(t, fontsize=14)
    ax.tick_params(axis="both", which="both", direction="out", length=5, width=1, labelsize=11)
    sns.despine(ax=ax)
    ax.grid(False)
    fig.tight_layout()
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_param_level_dataframe(df: pd.DataFrame, param_col: str, rmse_col: str, out_csv: Path) -> None:
    """One CSV per hyperparameter: aggregated (categorical) or per-trial (continuous)."""
    sub = df[[param_col, rmse_col]].dropna().copy()
    if sub.empty:
        return
    if _is_categorical_series(sub[param_col]):
        g = sub.groupby(param_col, dropna=False)[rmse_col].agg(["mean", "std", "count"]).reset_index()
        g.rename(columns={"mean": "mean_rmse", "std": "std_rmse", "count": "n_trials"}, inplace=True)
        g.to_csv(out_csv, index=False)
    else:
        sub.rename(columns={param_col: "value", rmse_col: "trial_rmse"}).to_csv(out_csv, index=False)


def infer_param_columns(df: pd.DataFrame, exclude: set[str]) -> list[str]:
    return [c for c in df.columns if c not in exclude]


def run_torch_hyperparam_overview(
    df_trials: pd.DataFrame,
    out_dir: Path,
    *,
    rmse_col: str = "mean_rmse",
    exclude_cols: frozenset[str] | None = None,
) -> list[str]:
    """
    For each explored hyperparameter column, write CSV + PNG.
    Returns list of param column names processed.

    ``exclude_cols`` is **unioned** with mandatory exclusions (scores and ids), so callers
    like sklearn tune scripts can pass ``{"iter", "std_rmse"}`` without accidentally
    treating ``mean_rmse`` as a hyperparameter (which would break plotting).
    """
    mandatory = frozenset(
        {
            "trial_idx",
            "mean_rmse",
            "mean_r2",
            "model_kind",
            "mlp_hidden",
            rmse_col,
        }
    )
    ex = mandatory | frozenset(exclude_cols or ())
    param_cols = infer_param_columns(df_trials, set(ex))
    diag_dir = out_dir
    df_dir = diag_dir / "dataframes"
    plot_dir = diag_dir / "plots"
    df_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    processed: list[str] = []
    for col in param_cols:
        if col not in df_trials.columns:
            continue
        if df_trials[col].isna().all():
            continue
        processed.append(col)
        safe = str(col).replace("/", "_")
        save_param_level_dataframe(df_trials, col, rmse_col, df_dir / f"{safe}.csv")
        plot_hyperparameter_vs_rmse(
            df_trials,
            param_col=col,
            rmse_col=rmse_col,
            outpath=plot_dir / f"{safe}.png",
            title=f"CV RMSE vs {col}",
        )
    (diag_dir / "processed_params.json").write_text(json.dumps(processed, indent=2), encoding="utf-8")
    return processed


def sklearn_random_search_to_trials_df(search: Any, *, rmse_neg_score: bool = True) -> pd.DataFrame:
    """Build a trial table from RandomizedSearchCV.cv_results_."""
    r = search.cv_results_
    n = len(r["params"])
    rows: list[dict[str, Any]] = []
    for i in range(n):
        row: dict[str, Any] = {"iter": i}
        score = float(r["mean_test_score"][i])
        row["mean_rmse"] = float(-score) if rmse_neg_score else float(score)
        row["std_rmse"] = float(r["std_test_score"][i]) if rmse_neg_score else float("nan")
        for k, v in r["params"][i].items():
            row[str(k)] = v
        rows.append(row)
    return pd.DataFrame(rows)


def plot_learning_curve_epochs(
    curves_df: pd.DataFrame,
    outpath: Path,
    *,
    title: str,
    y_metric: str = "val_rmse",
    trial_idx: int | None = None,
) -> None:
    """Line plot epoch vs metric; one line per fold (overview)."""
    apply_hyperopt_plot_style()
    fig, ax = plt.subplots(figsize=(20.0 / 2.54, 12.0 / 2.54))
    for fold in sorted(curves_df["fold"].unique()):
        sub = curves_df[curves_df["fold"] == fold]
        ax.plot(
            sub["epoch"],
            sub[y_metric],
            label=f"fold {int(fold)}",
            linewidth=1.5,
            alpha=0.85,
        )
    ax.set_xlabel("epoch", fontsize=14)
    ax.set_ylabel(y_metric, fontsize=14)
    ax.set_title(title, fontsize=14)
    ax.legend(frameon=True, fontsize=10)
    sns.despine(ax=ax)
    fig.tight_layout()
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_learning_curve_r2_epochs(
    curves_df: pd.DataFrame,
    outpath: Path,
    *,
    title: str,
    trial_idx: int | None = None,
) -> None:
    plot_learning_curve_epochs(curves_df, outpath, title=title, y_metric="val_r2")


def plot_xgboost_boosting_rmse(
    rounds: list[int],
    rmse: list[float],
    outpath: Path,
    *,
    title: str,
) -> None:
    """Validation RMSE vs boosting round (one fold, best params)."""
    apply_hyperopt_plot_style()
    fig, ax = plt.subplots(figsize=(20.0 / 2.54, 12.0 / 2.54))
    ax.plot(rounds, rmse, color=_ACCENT, linewidth=2.0, marker="o", markersize=3)
    ax.set_xlabel("boosting round", fontsize=14)
    ax.set_ylabel("val RMSE (one fold)", fontsize=14)
    ax.set_title(title, fontsize=14)
    sns.despine(ax=ax)
    fig.tight_layout()
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)
