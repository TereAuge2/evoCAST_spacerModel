from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import seaborn as sns
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.compose import TransformedTargetRegressor
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import ElasticNet, Lasso, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import NuSVR, SVR


@dataclass(frozen=True)
class Dataset:
    df: pd.DataFrame
    X: np.ndarray | sparse.csr_matrix
    y: np.ndarray


def one_hot_encode_dna(seqs: pd.Series, *, alphabet: str = "ACGT", pad_char: str = "N") -> sparse.csr_matrix:
    seq_list = ["" if pd.isna(x) else str(x) for x in seqs]
    seq_list = [s.upper().strip() for s in seq_list]
    n = len(seq_list)
    L = max((len(s) for s in seq_list), default=0)
    if n == 0 or L == 0:
        return sparse.csr_matrix((n, 0), dtype=np.uint8)

    alpha = list(alphabet)
    alpha_to_idx = {c: i for i, c in enumerate(alpha)}
    A = len(alpha)

    rows: list[int] = []
    cols: list[int] = []
    data: list[int] = []

    for r, s in enumerate(seq_list):
        if len(s) < L:
            s = s + (pad_char * (L - len(s)))
        for pos, ch in enumerate(s[:L]):
            j = alpha_to_idx.get(ch)
            if j is None:
                continue
            rows.append(r)
            cols.append(pos * A + j)
            data.append(1)

    return sparse.csr_matrix((data, (rows, cols)), shape=(n, L * A), dtype=np.uint8)


def gc_content(seqs: pd.Series) -> np.ndarray:
    s = seqs.fillna("").astype(str).str.upper().str.strip()
    gc = s.str.count("G") + s.str.count("C")
    atgc = gc + s.str.count("A") + s.str.count("T")
    with np.errstate(divide="ignore", invalid="ignore"):
        frac = (gc / atgc).astype(float)
    return np.nan_to_num(frac.to_numpy(dtype=float), nan=0.0, posinf=0.0, neginf=0.0)


def kmer_frequencies(seqs: pd.Series, *, k: int) -> np.ndarray:
    """
    Normalized k-mer frequencies over A/C/G/T only.
    Any window containing non-ACGT is skipped.
    """
    k = int(k)
    if k <= 0:
        raise ValueError("k must be >= 1")
    s_list = seqs.fillna("").astype(str).str.upper().str.strip().to_list()
    n = len(s_list)
    alpha = {"A": 0, "C": 1, "G": 2, "T": 3}
    n_feat = 4 ** k
    X = np.zeros((n, n_feat), dtype=np.float32)
    for i, s in enumerate(s_list):
        L = len(s)
        if L < k:
            continue
        denom = 0
        for j in range(L - k + 1):
            idx = 0
            ok = True
            for ch in s[j : j + k]:
                v = alpha.get(ch)
                if v is None:
                    ok = False
                    break
                idx = (idx * 4) + v
            if not ok:
                continue
            X[i, idx] += 1.0
            denom += 1
        if denom > 0:
            X[i, :] /= float(denom)
    return X


def build_feature_matrix(
    df: pd.DataFrame,
    *,
    embedding: str,
    kmer_k: int,
) -> np.ndarray | sparse.csr_matrix:
    if "SpacerSeq" not in df.columns:
        raise ValueError("Missing required column: SpacerSeq")

    embedding = embedding.strip().lower()
    if embedding == "onehot":
        return one_hot_encode_dna(df["SpacerSeq"])
    if embedding == "onehot_gc":
        X = one_hot_encode_dna(df["SpacerSeq"]).astype(np.float32)
        gc = gc_content(df["SpacerSeq"]).astype(np.float32).reshape(-1, 1)
        gc_col = sparse.csr_matrix(gc)
        return sparse.hstack([X, gc_col], format="csr")
    if embedding == "kmer":
        return kmer_frequencies(df["SpacerSeq"], k=int(kmer_k))
    if embedding == "kmer_gc":
        Xk = kmer_frequencies(df["SpacerSeq"], k=int(kmer_k)).astype(np.float32)
        gc = gc_content(df["SpacerSeq"]).astype(np.float32).reshape(-1, 1)
        return np.concatenate([Xk, gc], axis=1)
    if embedding == "kmer_gc_onehot":
        # Sparse one-hot + dense kmer freqs + GC appended as extra columns (all as sparse).
        Xo = one_hot_encode_dna(df["SpacerSeq"]).astype(np.float32)
        Xk = kmer_frequencies(df["SpacerSeq"], k=int(kmer_k)).astype(np.float32)
        gc = gc_content(df["SpacerSeq"]).astype(np.float32).reshape(-1, 1)
        Xk_sp = sparse.csr_matrix(Xk)
        gc_sp = sparse.csr_matrix(gc)
        return sparse.hstack([Xo, Xk_sp, gc_sp], format="csr")
    raise ValueError(f"Unknown embedding: {embedding}")

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    return {"rmse": rmse, "mae": mae, "r2": r2}

def _pearson_r(x: np.ndarray, y: np.ndarray) -> float:
    """Pearson r without scipy.stats (avoids heavy scipy import chain on slow filesystems)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size < 2:
        return float("nan")
    if np.allclose(np.std(x, ddof=0), 0.0) or np.allclose(np.std(y, ddof=0), 0.0):
        return float("nan")
    r = np.corrcoef(x, y)[0, 1]
    return float(r) if not np.isnan(r) else float("nan")


def _spearman_rho(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman ρ = Pearson correlation of average ranks (matches common scipy.stats.spearmanr)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size < 2:
        return float("nan")
    rx = pd.Series(x).rank(method="average").to_numpy(dtype=float)
    ry = pd.Series(y).rank(method="average").to_numpy(dtype=float)
    return _pearson_r(rx, ry)


def compute_corr_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.size < 2:
        return {"pearson_r": float("nan"), "spearman_rho": float("nan")}
    pr = _pearson_r(y_true, y_pred)
    sr = _spearman_rho(y_true, y_pred)
    return {"pearson_r": pr, "spearman_rho": sr}


def enrichment_factor_top_frac(y_true: np.ndarray, y_pred: np.ndarray, frac: float = 0.10) -> float:
    """
    EF@frac: enrichment of true top-frac items inside the top-frac predictions,
    relative to random selection.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    n_total = y_true.size
    if n_total == 0:
        return float("nan")

    n_top = max(1, int(round(frac * n_total)))

    # True top-frac set (by y_true)
    true_idx = np.argpartition(y_true, -n_top)[-n_top:]
    is_true_top = np.zeros(n_total, dtype=bool)
    is_true_top[true_idx] = True
    n_true_top = int(is_true_top.sum())
    frac_true_top = n_true_top / n_total if n_total > 0 else 0.0

    # Predicted top-frac set (by y_pred)
    pred_idx = np.argpartition(y_pred, -n_top)[-n_top:]
    is_pred_top = np.zeros(n_total, dtype=bool)
    is_pred_top[pred_idx] = True
    n_pred_top = int(is_pred_top.sum())

    if n_pred_top == 0 or frac_true_top == 0.0:
        return float("nan")

    tp = int((is_pred_top & is_true_top).sum())
    frac_tp_in_pred = tp / n_pred_top
    return float(frac_tp_in_pred / frac_true_top)


def regression_scatter_plot(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    title: str,
    xlab: str = "log2-FE (measured)",
    ylab: str = "log2-FE (predicted)",
    outpath: Path,
    width_cm: float = 18.0,
    height_cm: float = 12.0,
    use_hexbin: bool = False,
) -> None:
    mpl.rcParams.update(mpl.rcParamsDefault)
    mpl.rcParams["font.family"] = "sans-serif"
    mpl.rcParams["font.size"] = 14

    # Match requested sizing logic: widen plot when using hexbin so colorbar + metrics don't overlap.
    plot_width_cm = (width_cm + 8.0) if use_hexbin else width_cm
    plot_height_cm = height_cm
    fig, ax = plt.subplots(figsize=(plot_width_cm / 2.54, plot_height_cm / 2.54))

    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    if use_hexbin:
        import matplotlib.colors as mcolors

        _hex_cmap = mcolors.LinearSegmentedColormap.from_list("grey_to_blue", ["#d3d3d3", "#0d1b2a"], N=256)
        hb = ax.hexbin(
            y_true,
            y_pred,
            gridsize=50,
            mincnt=1,
            cmap=_hex_cmap,
            edgecolors="none",
            alpha=0.85,
            rasterized=True,
        )
        # Leave room for colorbar then metrics block to the right of plot (no overlap)
        fig.subplots_adjust(right=0.52)
        cbar = plt.colorbar(hb, ax=ax, pad=0.02, shrink=0.85, label="count")
        cbar.ax.tick_params(axis="y", labelsize=11)
    else:
        sns.scatterplot(
            x=y_true,
            y=y_pred,
            edgecolors="black",
            color="black",
            alpha=0.3,
            s=15,
            linewidth=0.3,
            rasterized=True,
            ax=ax,
        )

    ax.axvline(0, color="grey", linestyle=":", linewidth=1)
    ax.axhline(0, color="grey", linestyle=":", linewidth=1)

    r2 = float(r2_score(y_true, y_pred)) if y_true.size >= 2 else float("nan")
    corr = compute_corr_metrics(y_true, y_pred)
    ef10 = enrichment_factor_top_frac(y_true, y_pred, frac=0.10)

    metrics_text = (
        r"$\mathbf{Metrics}$"
        + "\n\n"
        + f"Pearson R: {corr['pearson_r']:.2f}\n"
        + f"Spearman ρ: {corr['spearman_rho']:.2f}\n"
        + f"R2: {r2:.2f}\n"
        + f"EF10%: {ef10:.2f}"
    )
    if use_hexbin:
        # Metrics block to the right of the colorbar (figure coords)
        fig.text(
            0.58,
            0.5,
            metrics_text,
            transform=fig.transFigure,
            ha="left",
            va="center",
            fontsize=12,
            bbox=dict(boxstyle="round", facecolor="white", edgecolor="black", linewidth=0.5),
        )
    else:
        ax.text(
            1.02,
            0.98,
            metrics_text,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=12,
            bbox=dict(boxstyle="round", facecolor="white", edgecolor="black", linewidth=0.5),
        )

    ax.set_xlabel(xlab, fontsize=14)
    ax.set_ylabel(ylab, fontsize=14)
    ax.set_title(title, fontsize=14)
    ax.tick_params(axis="both", which="both", direction="out", length=5, width=1, labelsize=11)
    sns.despine(ax=ax)
    ax.grid(False)
    if not use_hexbin:
        fig.tight_layout()

    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)


def y_cv_and_groups_from_excel(
    excel_path: Path,
    *,
    sheet: str | int = 0,
    split_col: str,
    target_col: str = "FE",
) -> tuple[np.ndarray, np.ndarray]:
    """
    CV rows only (split labels starting with ``cv``), same filter as ``train_models.main``.
    Returns ``(y_cv, groups)`` for median-baseline metrics.
    """
    df = pd.read_excel(excel_path, sheet_name=sheet, engine="openpyxl")
    needed = {split_col, target_col}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in Excel: {sorted(missing)}")
    df = df.copy()
    df[target_col] = pd.to_numeric(df[target_col], errors="coerce")
    df = df[df[target_col].notna()].reset_index(drop=True)
    mask_cv = df[split_col].astype(str).str.startswith("cv")
    df_cv = df.loc[mask_cv].reset_index(drop=True)
    if df_cv.empty:
        raise ValueError("No CV rows found (expected Split values like cv1..cv5).")
    y_cv = df_cv[target_col].to_numpy(dtype=float)
    groups = df_cv[split_col].astype(str).to_numpy()
    return y_cv, groups


def _safe_nanmean(values: list[float] | np.ndarray) -> float:
    """``np.nanmean`` on an empty array warns; return NaN instead.

    Do not use ``if not values``: NumPy arrays can be non-empty in Python truthiness
    but still yield ``size == 0`` after ``asarray`` (e.g. nested empty arrays), and
    ``bool(ndarray)`` is ambiguous for multi-element arrays.
    """
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return float("nan")
    return float(np.nanmean(arr))


def median_baseline_cv_metric_means(y_cv: np.ndarray, groups: np.ndarray) -> dict[str, float]:
    """
    GroupKFold on CV rows only: each fold predicts ``median(y_train)`` on validation.
    Returns mean of RMSE / MAE / R² / correlations / EF10% across folds (one scalar per metric).
    """
    keys = ["rmse", "mae", "r2", "pearson_r", "spearman_rho", "ef10"]
    nan_all = {k: float("nan") for k in keys}

    y_cv = np.asarray(y_cv, dtype=float)
    groups = np.asarray(groups)
    if y_cv.size == 0 or groups.size == 0:
        return nan_all

    unique_groups = sorted(set(groups.tolist()))
    # GroupKFold needs at least 2 groups; with 1 group there are no valid train/val splits.
    if len(unique_groups) < 2:
        return nan_all

    n_splits = min(5, len(unique_groups))
    gkf = GroupKFold(n_splits=n_splits)
    X_dummy = np.zeros((len(y_cv), 1))

    bucket: dict[str, list[float]] = {k: [] for k in keys}
    for tr, va in gkf.split(X_dummy, y_cv, groups=groups):
        med = float(np.median(y_cv[tr]))
        pred = np.full(y_cv[va].shape, med, dtype=float)
        yt = y_cv[va]
        m = compute_metrics(yt, pred)
        corr = compute_corr_metrics(yt, pred)
        ef10 = enrichment_factor_top_frac(yt, pred, frac=0.10)
        row = {**m, **corr, "ef10": ef10}
        for k in bucket:
            bucket[k].append(float(row[k]))

    # Pearson/Spearman are often NaN: predictions are constant per fold, so correlation is undefined.
    return {k: _safe_nanmean(vs) for k, vs in bucket.items()}


# CV summary bar colors: pastel palette by model family (see legend in ``plot_cv_summary``).
_CV_MODEL_TO_FAMILY: dict[str, str] = {
    "cnn_onehot": "cnn",
    "cnn1d": "cnn",
    "mlp": "mlp",
    "nusvr_rbf": "svr",
    "svr_rbf": "svr",
    "kernel_ridge_rbf": "svr",
    "elasticnet": "linear",
    "lasso": "linear",
    "ridge": "linear",
    "xgboost": "trees",
    "gbr": "trees",
    "extratrees": "trees",
    "rf": "trees",
    "rf_simon": "trees",
    "knn": "knn",
}
_CV_MODEL_FAMILY_ORDER: tuple[str, ...] = ("cnn", "mlp", "svr", "linear", "trees", "knn", "other")
# Pastels chosen so adjacent families read clearly (SVR = cool cyan/teal; trees = warm sage).
_CV_MODEL_FAMILY_COLORS: dict[str, str] = {
    "cnn": "#9BB8E8",
    "mlp": "#C8B0E8",
    "svr": "#7ECFE0",  # cyan-teal (distinct from yellow-greens)
    "linear": "#F2D4A8",
    "trees": "#A8C990",  # sage / yellow-green (distinct from SVR)
    "knn": "#E8A8B8",
    "other": "#D0D0D0",
}
# Short legend text only (model lists live in code comments / paper, not on the figure).
_CV_MODEL_FAMILY_LEGEND_LABELS: dict[str, str] = {
    "cnn": "CNN",
    "mlp": "MLP",
    "svr": "SVR / kernel ridge",
    "linear": "Linear",
    "trees": "Tree ensembles",
    "knn": "KNN",
    "other": "Other",
}


def _cv_model_family_key(model: str) -> str:
    key = str(model).strip().lower()
    if key in _CV_MODEL_TO_FAMILY:
        return _CV_MODEL_TO_FAMILY[key]
    return "other"


def _cv_summary_bar_color(model: str) -> str:
    return _CV_MODEL_FAMILY_COLORS[_cv_model_family_key(model)]


def plot_cv_summary(
    df_scores: pd.DataFrame,
    outpath: Path,
    *,
    title: str,
    median_baseline: dict[str, float] | None = None,
) -> None:
    mpl.rcParams.update(mpl.rcParamsDefault)
    # Prefer Arial (requested aesthetic) but fall back gracefully on clusters
    # where Arial isn't installed (avoids noisy "findfont" spam).
    try:
        import logging
        from matplotlib import font_manager

        logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
        arial_path = font_manager.findfont("Arial", fallback_to_default=False)  # type: ignore[arg-type]
        if arial_path:
            mpl.rcParams["font.family"] = "Arial"
        else:
            mpl.rcParams["font.family"] = "DejaVu Sans"
    except Exception:
        mpl.rcParams["font.family"] = "DejaVu Sans"
    mpl.rcParams["font.size"] = 14

    metrics_cols = ["rmse", "mae", "r2", "pearson_r", "spearman_rho", "ef10"]
    # If each model appears only once (e.g. test-set evaluation), plot raw values (no "mean across folds").
    # Otherwise, summarize as mean ± std across folds.
    per_model_n = df_scores.groupby("model").size()
    singleton_per_model = bool((per_model_n == 1).all())

    if singleton_per_model:
        dfm = df_scores.groupby("model", as_index=False)[metrics_cols].first()
    else:
        dfm = df_scores.groupby("model", as_index=False)[metrics_cols].agg(["mean", "std"]).reset_index()

    if not singleton_per_model:
        # Flatten columns safely (keeps "model" as "model")
        flat_cols: list[str] = []
        for c in dfm.columns:
            if isinstance(c, tuple):
                a, b = c
                if b == "":
                    flat_cols.append(str(a))
                elif a == "model":
                    flat_cols.append("model")
                else:
                    flat_cols.append(f"{a}_{b}")
            else:
                flat_cols.append(str(c))
        dfm.columns = flat_cols

    metrics_info: list[tuple[str, str, bool]] = [
        ("rmse", "RMSE", True),  # lower is better
        ("pearson_r", "Pearson R", False),  # higher is better
        ("spearman_rho", "Spearman ρ", False),
        ("r2", "R²", False),
        ("ef10", "EF10%", False),
    ]

    outpath.parent.mkdir(parents=True, exist_ok=True)

    for metric_key, metric_label, lower_is_better in metrics_info:
        if singleton_per_model:
            val_col = metric_key
            if val_col not in dfm.columns:
                continue
        else:
            mean_col = f"{metric_key}_mean"
            std_col = f"{metric_key}_std"
            if mean_col not in dfm.columns or std_col not in dfm.columns:
                continue

        # Sort each plot by its own metric
        sort_col = (metric_key if singleton_per_model else mean_col)
        order = dfm.sort_values(sort_col, ascending=lower_is_better)["model"].tolist()

        # Wide figure so many models + right-side legend do not compress the axes (width_cm × height_cm).
        fig, ax = plt.subplots(figsize=(32.0 / 2.54, 12.0 / 2.54))
        y_col = metric_key if singleton_per_model else mean_col
        x_pos = np.arange(len(order), dtype=float)
        y = dfm.set_index("model").loc[order, y_col].to_numpy(dtype=float)
        if not singleton_per_model:
            yerr = dfm.set_index("model").loc[order, std_col].to_numpy(dtype=float)
        else:
            yerr = None

        bar_colors = [_cv_summary_bar_color(m) for m in order]
        ax.bar(
            x_pos,
            y,
            width=0.72,
            color=bar_colors,
            edgecolor="0.35",
            linewidth=0.6,
            zorder=2,
        )
        ax.set_xticks(x_pos)
        ax.set_xticklabels(order)

        if not singleton_per_model:
            ax.errorbar(
                x_pos,
                y,
                yerr=yerr,
                fmt="none",
                ecolor="black",
                elinewidth=1.0,
                capsize=3,
                capthick=1.0,
                zorder=3,
            )

        if not singleton_per_model:
            # Overlay per-fold dots (scatter) on top of the mean ± std bars.
            # We jitter x slightly to avoid full overlap of identical values.
            rng = np.random.default_rng(0)
            for xi, model_name in zip(x_pos, order):
                y_folds = df_scores.loc[df_scores["model"] == model_name, metric_key].to_numpy(dtype=float)
                if y_folds.size == 0:
                    continue
                jitter = rng.uniform(-0.15, 0.15, size=y_folds.size)
                ax.scatter(
                    xi + jitter,
                    y_folds,
                    color="black",
                    s=14,
                    alpha=0.85,
                    zorder=4,
                )

        if median_baseline and metric_key in median_baseline:
            yb = float(median_baseline[metric_key])
            if np.isfinite(yb):
                y_lo0, y_hi0 = ax.get_ylim()
                pad = 0.04 * (y_hi0 - y_lo0) if y_hi0 > y_lo0 else 0.05
                if yb < y_lo0:
                    ax.set_ylim(yb - pad, y_hi0)
                elif yb > y_hi0:
                    ax.set_ylim(y_lo0, yb + pad)
                ax.axhline(
                    yb,
                    linestyle=":",
                    color="0.35",
                    linewidth=1.4,
                    zorder=10,
                    clip_on=False,
                )

        # Add score label (always show value on bar).
        y_bottom, y_top = ax.get_ylim()
        y_span_ylim = float(y_top - y_bottom)
        text_offset = 0.03 * y_span_ylim if y_span_ylim != 0 else 0.03
        if singleton_per_model:
            for xi, yv in zip(x_pos, y):
                ax.text(
                    xi,
                    float(yv) + text_offset,
                    f"{float(yv):.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=10,
                )
        else:
            min_y = y_bottom + 0.0005 * y_span_ylim if y_span_ylim != 0 else y_bottom
            for xi, y_mean, y_std in zip(x_pos, y, yerr):
                y_text = (y_mean - y_std) - (0.04 * y_span_ylim if y_span_ylim != 0 else 0.04)
                y_text = max(y_text, min_y)
                ax.text(
                    xi,
                    y_text,
                    f"{y_mean:.2f}",
                    ha="center",
                    va="top",
                    fontsize=10,
                )

        # Caller passes something like "<split> CV summary (RMSE)"; strip the placeholder
        # "(RMSE)" and let the metric_label below become the 2nd line.
        title_clean = title.replace(" (RMSE)", "")
        ax.set_title(f"{title_clean}\n{metric_label}")
        ax.set_xlabel("Model")
        ax.set_ylabel(f"{metric_label} (value)" if singleton_per_model else f"{metric_label} (mean ± std across folds)")
        ax.tick_params(axis="both", which="major", labelsize=10)
        plt.setp(ax.get_xticklabels(), rotation=90, ha="center", va="top")
        sns.despine(ax=ax)
        ax.grid(False)

        families_present = {_cv_model_family_key(m) for m in order}
        fam_order = sorted(
            families_present,
            key=lambda k: (
                _CV_MODEL_FAMILY_ORDER.index(k)
                if k in _CV_MODEL_FAMILY_ORDER
                else len(_CV_MODEL_FAMILY_ORDER)
            ),
        )
        leg_handles = [
            Patch(
                facecolor=_CV_MODEL_FAMILY_COLORS[k],
                edgecolor="0.35",
                linewidth=0.5,
                label=_CV_MODEL_FAMILY_LEGEND_LABELS[k],
            )
            for k in fam_order
        ]
        ax.legend(
            handles=leg_handles,
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            borderaxespad=0.0,
            fontsize=11,
            frameon=True,
            fancybox=False,
            edgecolor="0.75",
            title="Model type",
            title_fontsize=12,
        )

        # Leave space on the right for the legend; wide figure keeps the bar area comfortably wide.
        fig.tight_layout(rect=[0.025, 0.10, 0.80, 0.97])

        metric_suffix = metric_key.upper() if metric_key != "ef10" else "EF10"
        # RMSE plot is the canonical `outpath` (e.g. cv_summary_Split....png). Other metrics
        # get suffixed filenames only — avoids duplicating RMSE as both base and *_RMSE.png.
        if metric_key == "rmse":
            fig.savefig(outpath, dpi=300, bbox_inches="tight")
        else:
            out_metric = outpath.with_name(f"{outpath.stem}_{metric_suffix}.png")
            fig.savefig(out_metric, dpi=300, bbox_inches="tight")
        plt.close(fig)


class TorchCNNRegressor(BaseEstimator, RegressorMixin):
    """Compact 1D CNN regressor with sklearn-style fit/predict."""

    def __init__(
        self,
        *,
        random_state: int = 42,
        epochs: int = 50,
        batch_size: int = 64,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        verbose: int = 0,
    ) -> None:
        self.random_state = int(random_state)
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.verbose = int(verbose)
        self._model = None
        self._n_features = None

    @staticmethod
    def _to_channel_first(X: np.ndarray) -> np.ndarray:
        """(n, 4*L) one-hot → (n, 4, L) with channels A,C,G,T. No 1×L fallback."""
        n, d = X.shape
        if d % 4 != 0:
            raise ValueError(
                "cnn1d expects flat one-hot length divisible by 4 (sequence_length × 4 for ACGT); "
                f"got n_features={d}."
            )
        L = d // 4
        return X.reshape(n, L, 4).transpose(0, 2, 1).astype(np.float32, copy=False)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "TorchCNNRegressor":
        import torch
        import torch.nn as nn

        def _infer_torch_threads() -> int:
            for key in ("TORCH_NUM_THREADS", "OMP_NUM_THREADS", "SLURM_CPUS_PER_TASK"):
                v = os.environ.get(key)
                if v:
                    try:
                        return max(1, int(v))
                    except ValueError:
                        pass
            return max(1, os.cpu_count() or 2)

        torch.set_num_threads(_infer_torch_threads())
        # PyTorch allows set_num_interop_threads only once per process; each CV fold calls fit() again.
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass

        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32).reshape(-1)
        self._n_features = int(X.shape[1])
        Xc = self._to_channel_first(X)
        in_ch = int(Xc.shape[1])

        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        class _Net(nn.Module):
            def __init__(self, in_channels: int) -> None:
                super().__init__()
                # in_channels is always 4 (A,C,G,T) for DNA one-hot
                self.conv = nn.Sequential(
                    nn.Conv1d(in_channels, 64, kernel_size=5, padding=2),
                    nn.ReLU(),
                    nn.Conv1d(64, 64, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.AdaptiveMaxPool1d(8),
                )
                self.head = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(64 * 8, 128),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(128, 1),
                )

            def forward(self, xb: torch.Tensor) -> torch.Tensor:
                return self.head(self.conv(xb)).squeeze(-1)

        model = _Net(in_ch)
        opt = torch.optim.AdamW(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        loss_fn = nn.MSELoss()

        Xt = torch.from_numpy(Xc)
        yt = torch.from_numpy(y)
        n = int(Xt.shape[0])
        bs = max(8, int(self.batch_size))

        n_ep = max(1, int(self.epochs))
        print(f"[cnn1d] training: n={n}, epochs={n_ep}, batch_size={bs}", flush=True)
        model.train()
        for ep in range(n_ep):
            perm = torch.randperm(n)
            ep_loss = 0.0
            for i in range(0, n, bs):
                idx = perm[i : i + bs]
                xb = Xt[idx]
                yb = yt[idx]
                opt.zero_grad(set_to_none=True)
                pred = model(xb)
                loss = loss_fn(pred, yb)
                loss.backward()
                opt.step()
                ep_loss += float(loss.item()) * int(xb.shape[0])
            mean_loss = ep_loss / n
            # Always log each epoch so Slurm jobs do not look hung (verbose=0 used to print only every 10).
            if self.verbose or (ep + 1) == 1 or (ep + 1) % 5 == 0 or (ep + 1) == n_ep:
                print(f"[cnn1d] epoch {ep + 1}/{n_ep} loss={mean_loss:.6f}", flush=True)

        self._model = model.eval()
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        import torch

        if self._model is None:
            raise RuntimeError("Model is not fitted")
        X = np.asarray(X, dtype=np.float32)
        if self._n_features is None or int(X.shape[1]) != int(self._n_features):
            raise ValueError("Feature dimension mismatch at predict time")
        Xc = self._to_channel_first(X)
        Xt = torch.from_numpy(Xc)
        with torch.no_grad():
            pred = self._model(Xt).cpu().numpy().astype(float, copy=False)
        return pred


def get_models(random_state: int) -> dict[str, object]:
    # Note: we keep models simple + robust. Most use standardized SVD features.
    models: dict[str, object] = {
        "ridge": Ridge(alpha=1.0, random_state=random_state),
        "lasso": Lasso(alpha=1e-3, random_state=random_state, max_iter=10000),
        "elasticnet": ElasticNet(alpha=1e-3, l1_ratio=0.5, random_state=random_state, max_iter=10000),
        "knn": KNeighborsRegressor(n_neighbors=15, weights="distance"),
        "svr_rbf": SVR(C=10.0, gamma="scale", epsilon=0.1),
        "nusvr_rbf": NuSVR(C=10.0, nu=0.5, gamma="scale"),
        "kernel_ridge_rbf": KernelRidge(alpha=1.0, kernel="rbf", gamma=None),
        "rf": RandomForestRegressor(
            n_estimators=500, random_state=random_state, n_jobs=-1, min_samples_leaf=2
        ),
        "rf_Simon": RandomForestRegressor(n_estimators=800,max_depth=None,min_samples_leaf=1,max_features=1.0,bootstrap=True,max_samples=0.8,random_state=random_state
        ),
        "extratrees": ExtraTreesRegressor(
            n_estimators=800, random_state=random_state, n_jobs=-1, min_samples_leaf=2
        ),
        "gbr": GradientBoostingRegressor(random_state=random_state),
        "mlp": MLPRegressor(
            hidden_layer_sizes=(256, 128),
            activation="relu",
            alpha=1e-4,
            learning_rate_init=1e-3,
            max_iter=500,
            random_state=random_state,
        ),
    }

    # Optional: compact 1D CNN if torch is installed.
    try:
        import torch  # type: ignore

        _ = torch
        models["cnn1d"] = TorchCNNRegressor(
            random_state=random_state,
            epochs=50,
            batch_size=64,
            lr=1e-3,
            weight_decay=1e-4,
            verbose=0,
        )
    except Exception:
        pass

    # Optional: XGBoost if installed
    try:
        from xgboost import XGBRegressor  # type: ignore

        models["xgboost"] = XGBRegressor(
            n_estimators=2000,
            learning_rate=0.03,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            objective="reg:squarederror",
            random_state=random_state,
            n_jobs=-1,
        )
    except Exception:
        pass

    return models


def load_dataset(excel_path: Path, *, sheet: str | int = 0, split_col: str, target_col: str) -> Dataset:
    df = pd.read_excel(excel_path, sheet_name=sheet, engine="openpyxl")
    needed = {"SpacerSeq", target_col, split_col}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in excel: {sorted(missing)}")

    df = df.copy()
    df[target_col] = pd.to_numeric(df[target_col], errors="coerce")
    df = df[df[target_col].notna()].reset_index(drop=True)

    y = df[target_col].to_numpy(dtype=float)
    # X is constructed in main() so it can depend on embedding args.
    return Dataset(df=df, X=np.empty((len(df), 0), dtype=np.float32), y=y)


def max_svd_components(X: sparse.csr_matrix) -> int:
    """
    Largest valid n_components for TruncatedSVD on X (full numerical rank in the
    SVD basis; cumulative explained variance ratio sums to 1.0 up to float precision).
    """
    n_samples, n_features = X.shape
    return max(2, min(int(n_samples) - 1, int(n_features) - 1))


def fit_svd(
    X: sparse.csr_matrix, *, n_components: int, random_state: int
) -> tuple[np.ndarray, TruncatedSVD]:
    n_components = int(n_components)
    n_components = max(2, n_components)
    n_components = min(n_components, max(2, X.shape[0] - 1), max(2, X.shape[1] - 1))
    svd = TruncatedSVD(n_components=n_components, random_state=random_state)
    X_tf = svd.fit_transform(X)
    return X_tf, svd


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--excel",
        default="output/data_processing/251129_P8_120Complete_dataset_woCtrl_exogenous_with_splits.xlsx",
        help="Path to the exported Excel with Split columns",
    )
    ap.add_argument("--sheet", default=0, help="Sheet name or index")
    ap.add_argument("--target-col", default="FE")
    ap.add_argument(
        "--split-col",
        choices=["Split", "SplitRandom"],
        default="Split",
        help="Which split definition to use",
    )
    ap.add_argument("--random-state", type=int, default=42)
    ap.add_argument(
        "--svd-components",
        type=int,
        default=82,
        help="SVD components used as model features (ignored if --svd-full-variance)",
    )
    ap.add_argument(
        "--svd-full-variance",
        action="store_true",
        help="Use maximum SVD rank on CV data (~100%% cumulative explained variance). Overrides --svd-components.",
    )
    ap.add_argument(
        "--raw-onehot",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Dense features: no TruncatedSVD on sparse blocks (default: on for kmer_gc_onehot). "
        "Use --no-raw-onehot to use SVD on one-hot / kmer_gc_onehot sparse input.",
    )
    ap.add_argument(
        "--embedding",
        choices=["onehot", "onehot_gc", "kmer", "kmer_gc", "kmer_gc_onehot"],
        default="kmer_gc_onehot",
        help="Feature embedding from SpacerSeq. Default: raw one-hot + GC + k-mer freqs (dinucleotide when k=2).",
    )
    ap.add_argument(
        "--kmer-k",
        type=int,
        default=2,
        choices=[2, 3, 4],
        help="k for k-mer frequencies when embedding is kmer/kmer_gc (2=di, 3=tri, 4=quad).",
    )
    ap.add_argument("--output-dir", default="output/modeling", help="Where to write results")
    ap.add_argument(
        "--run-name",
        default="",
        help="Optional subfolder name under output-dir to avoid overwriting (e.g. 'kmer3_gc_try1').",
    )
    ap.add_argument(
        "--make-plots",
        action="store_true",
        default=True,
        help="Save measured-vs-predicted plots per model/fold (default: enabled)",
    )
    ap.add_argument(
        "--use-hexbin",
        action="store_true",
        default=True,
        help="Use hexbin instead of scatter for regression plots (default: enabled)",
    )
    ap.add_argument(
        "--models",
        default="all",
        help="Comma-separated model names to run (or 'all').",
    )
    ap.add_argument(
        "--cnn-epochs",
        type=int,
        default=50,
        help="Training epochs for cnn1d only (default 50).",
    )
    args = ap.parse_args(argv)

    t0 = time.time()

    def log(msg: str) -> None:
        dt = time.time() - t0
        print(f"[{dt:8.1f}s] {msg}", flush=True)

    # Line-buffered stderr so progress is visible under Slurm / no-TTY runs
    try:
        sys.stdout.reconfigure(line_buffering=True)  # type: ignore[attr-defined]
    except Exception:
        pass

    log("train_models: startup (imports finished, parsing done)")
    if args.raw_onehot and args.svd_full_variance:
        log("Note: --raw-onehot is set; --svd-full-variance is ignored")

    out_dir = Path(args.output_dir)
    if str(args.run_name).strip():
        out_dir = out_dir / str(args.run_name).strip()
    out_dir.mkdir(parents=True, exist_ok=True)

    log(f"Loading dataset: {args.excel} (split_col={args.split_col}, target_col={args.target_col})")
    ds = load_dataset(Path(args.excel), sheet=args.sheet, split_col=args.split_col, target_col=args.target_col)
    log(f"Loaded rows (after dropping NaN target): {len(ds.df)}")

    # Keep test completely untouched
    mask_cv = ds.df[args.split_col].astype(str).str.startswith("cv")
    df_cv = ds.df.loc[mask_cv].reset_index(drop=True)
    log(f"CV rows: {len(df_cv)} (test rows held out: {int((~mask_cv).sum())})")

    if df_cv.empty:
        raise ValueError("No CV rows found (expected Split values like cv1..cv5).")

    X_all = build_feature_matrix(ds.df, embedding=args.embedding, kmer_k=int(args.kmer_k))
    X_cv_all = X_all[mask_cv.to_numpy()]
    y_cv = ds.y[mask_cv.to_numpy()]

    out_tag = ""
    if args.embedding != "onehot":
        out_tag += f"_{args.embedding}"
    if args.embedding.startswith("kmer"):
        out_tag += f"_k{int(args.kmer_k)}"
    if args.raw_onehot and args.embedding in ("onehot", "onehot_gc", "kmer_gc_onehot"):
        out_tag += "_raw_onehot"

    if args.embedding in {"kmer", "kmer_gc"}:
        X_cv = np.asarray(X_cv_all, dtype=np.float64)
        log(f"Using k-mer feature matrix: shape={X_cv.shape} (k={int(args.kmer_k)})")
    else:
        # onehot / onehot_gc / kmer_gc_onehot
        if not sparse.issparse(X_cv_all):
            raise ValueError("Expected sparse matrix for onehot embeddings")
        X_cv_onehot = X_cv_all
        if args.raw_onehot:
            log(f"Using raw dense one-hot (no SVD); sparse shape={X_cv_onehot.shape}, nnz={X_cv_onehot.nnz}")
            X_cv = np.asarray(X_cv_onehot.toarray(), dtype=np.float64)
            log(f"Dense one-hot feature matrix: shape={X_cv.shape}")
        else:
            # Convert to dense SVD features for classic models
            if args.svd_full_variance:
                n_svd = max_svd_components(X_cv_onehot)
                log(f"Computing SVD features (--svd-full-variance): n_components={n_svd} (~100% cumulative variance)")
            else:
                n_svd = args.svd_components
                log(f"Computing SVD features (n_components={n_svd})")
            log(f"SVD input: sparse one-hot shape={X_cv_onehot.shape}, nnz={X_cv_onehot.nnz}")
            X_cv, svd_est = fit_svd(X_cv_onehot, n_components=n_svd, random_state=args.random_state)
            var_sum = float(np.sum(svd_est.explained_variance_ratio_))
            log(f"SVD feature matrix: shape={X_cv.shape}; cumulative explained variance ratio={var_sum:.6f}")

    # Groups: each cv fold label is its own held-out group
    groups = df_cv[args.split_col].astype(str).to_numpy()
    unique_groups = sorted(set(groups))
    log(f"CV groups: {unique_groups}")
    if len(unique_groups) != 5:
        # We still run GroupKFold, but warn via output file
        (out_dir / "split_warning.txt").write_text(
            f"Expected 5 CV folds (cv1..cv5), found: {unique_groups}\n"
        )

    gkf = GroupKFold(n_splits=min(5, len(unique_groups)))

    split_plot = f"{args.split_col}{out_tag}"

    models = get_models(args.random_state)
    if "cnn1d" in models:
        cnn = models["cnn1d"]
        if hasattr(cnn, "epochs"):
            cnn.epochs = int(args.cnn_epochs)
    available = list(models.keys())
    explicit_subset = args.models.strip().lower() != "all"
    if explicit_subset:
        wanted = [m.strip() for m in args.models.split(",") if m.strip()]
        missing = sorted(set(wanted) - set(available))
        if missing:
            raise ValueError(f"Unknown model(s): {missing}. Available: {available}")
        models = {k: models[k] for k in wanted}
    if "cnn1d" in models and args.embedding != "onehot":
        if explicit_subset:
            raise ValueError(
                "cnn1d requires --embedding onehot (flat length divisible by 4). "
                f"Got embedding={args.embedding!r}. Omit cnn1d or use --embedding onehot."
            )
        log(f"Skipping cnn1d (requires --embedding onehot; default embedding is {args.embedding}).")
        del models["cnn1d"]
    log(f"Models to run: {list(models.keys())}")

    rows: list[dict[str, object]] = []
    for model_name, model in models.items():
        log(f"Training model: {model_name}")
        # Standardize features for models that benefit from scaling
        pipe = Pipeline([("scaler", StandardScaler()), ("model", clone(model))])
        reg = TransformedTargetRegressor(regressor=pipe, transformer=StandardScaler())

        all_val_true: list[np.ndarray] = []
        all_val_pred: list[np.ndarray] = []

        for fold_idx, (tr, va) in enumerate(gkf.split(X_cv, y_cv, groups=groups), start=1):
            log(f"  Fold {fold_idx}: fitting (n_train={len(tr)}, n_val={len(va)})")
            reg_fold = clone(reg)
            reg_fold.fit(X_cv[tr], y_cv[tr])
            pred = reg_fold.predict(X_cv[va])

            m = compute_metrics(y_cv[va], pred)
            corr = compute_corr_metrics(y_cv[va], pred)
            ef10 = enrichment_factor_top_frac(y_cv[va], pred, frac=0.10)
            rows.append(
                {
                    "split_col": args.split_col,
                    "model": model_name,
                    "fold": fold_idx,
                    "val_group": sorted(set(groups[va])),
                    "n_train": int(len(tr)),
                    "n_val": int(len(va)),
                    **m,
                    **corr,
                    "ef10": ef10,
                }
            )

            all_val_true.append(y_cv[va])
            all_val_pred.append(np.asarray(pred, dtype=float))

            if args.make_plots:
                plot_dir = out_dir / "plots" / split_plot / model_name
                regression_scatter_plot(
                    y_cv[va],
                    pred,
                    title=f"{model_name} — {split_plot} fold {fold_idx}",
                    outpath=plot_dir / f"regression_fold{fold_idx}.png",
                    use_hexbin=bool(args.use_hexbin),
                )
        log(f"Finished model: {model_name}")

        if args.make_plots and all_val_true:
            y_all = np.concatenate(all_val_true, axis=0)
            p_all = np.concatenate(all_val_pred, axis=0)
            plot_dir = out_dir / "plots" / split_plot / model_name
            regression_scatter_plot(
                y_all,
                p_all,
                title=f"{model_name} — {split_plot} pooled CV",
                outpath=plot_dir / "regression_pooled.png",
                use_hexbin=bool(args.use_hexbin),
            )

    # Scores from this run
    df_scores_run = pd.DataFrame(rows)

    # Merge with any existing scores so summaries stay across all models
    scores_path = out_dir / f"cv_scores_{args.split_col}{out_tag}.csv"
    if scores_path.exists():
        df_prev = pd.read_csv(scores_path)
        # Drop old rows for models we just reran
        current_models = set(models.keys())
        df_prev = df_prev[~df_prev["model"].isin(current_models)]
        df_scores_all = pd.concat([df_prev, df_scores_run], ignore_index=True)
    else:
        df_scores_all = df_scores_run

    df_scores_all.to_csv(scores_path, index=False)
    log(f"Wrote per-fold scores (merged): {scores_path.name}")

    df_summary = (
        df_scores_all.groupby(["split_col", "model"], as_index=False)[["rmse", "mae", "r2"]]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary_csv = out_dir / f"cv_summary_{args.split_col}{out_tag}.csv"
    df_summary.to_csv(summary_csv, index=False)
    log(f"Wrote summary: {summary_csv.name}")
    plot_cv_summary(
        df_scores_all,
        out_dir / f"cv_summary_{args.split_col}{out_tag}.png",
        title=f"{split_plot} CV summary (RMSE)",
    )
    log(f"Wrote summary plot: cv_summary_{args.split_col}{out_tag}.png")
    log("Done")

    return 0


if __name__ == "__main__":
    # PYTHONUNBUFFERED=1 or `python -u` also helps on clusters where stdout is fully buffered
    raise SystemExit(main())

