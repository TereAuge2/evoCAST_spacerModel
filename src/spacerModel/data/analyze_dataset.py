from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from collections.abc import Callable
from typing import Iterable

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import sparse
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import logomaker


@dataclass(frozen=True)
class PlotStyle:
    width_cm: float = 12.0
    height_cm: float = 12.0
    rect_width_cm: float = 16.0
    rect_height_cm: float = 12.0
    cbar_width_cm: float = 0.6
    cbar_pad_cm: float = 0.4
    dpi: int = 300
    font_family: str = "Arial"
    font_size: int = 14


def set_plot_style(style: PlotStyle) -> None:
    mpl.rcParams.update(mpl.rcParamsDefault)
    # Prefer Arial, but fall back cleanly on HPC nodes where Arial isn't installed.
    preferred = style.font_family
    fallback_stack = [preferred, "DejaVu Sans", "Liberation Sans", "Nimbus Sans", "sans-serif"]
    try:
        from matplotlib import font_manager as fm

        _ = fm.findfont(preferred, fallback_to_default=False)
        mpl.rcParams["font.family"] = preferred
    except Exception:
        mpl.rcParams["font.family"] = "sans-serif"
        mpl.rcParams["font.sans-serif"] = fallback_stack
        try:
            import logging

            logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
        except Exception:
            pass
    mpl.rcParams["font.size"] = style.font_size


def savefig(path: Path, dpi: int, *, tight: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sns.despine()
    if tight:
        plt.tight_layout()
    plt.grid(False)
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close()


def to_float_series(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def one_hot_encode_dna(
    seqs: Iterable[str],
    *,
    alphabet: str = "ACGT",
    pad_char: str = "N",
) -> tuple[sparse.csr_matrix, int]:
    """
    One-hot encode DNA sequences into a sparse matrix of shape (n, L*|alphabet|).
    Sequences are uppercased and right-padded to max length with pad_char.

    Characters not in alphabet are encoded as all-zeros for that position.
    """
    seq_list = ["" if (x is None or (isinstance(x, float) and np.isnan(x))) else str(x) for x in seqs]
    seq_list = [s.upper().strip() for s in seq_list]
    n = len(seq_list)
    L = max((len(s) for s in seq_list), default=0)
    if n == 0 or L == 0:
        return sparse.csr_matrix((n, 0), dtype=np.uint8), L

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

    X = sparse.csr_matrix((data, (rows, cols)), shape=(n, L * A), dtype=np.uint8)
    return X, L


def dna_position_frequency_matrix(seqs: pd.Series, *, alphabet: str = "ACGT") -> tuple[pd.DataFrame, int]:
    seq_list = ["" if pd.isna(x) else str(x) for x in seqs]
    seq_list = [s.upper().strip() for s in seq_list]
    L = max((len(s) for s in seq_list), default=0)
    A = list(alphabet)
    counts = np.zeros((L, len(A)), dtype=float)
    for s in seq_list:
        if len(s) < L:
            s = s + ("N" * (L - len(s)))
        for i, ch in enumerate(s[:L]):
            if ch in A:
                counts[i, A.index(ch)] += 1.0
    with np.errstate(divide="ignore", invalid="ignore"):
        freqs = counts / counts.sum(axis=1, keepdims=True)
    freqs = np.nan_to_num(freqs, nan=0.0)
    df = pd.DataFrame(freqs, columns=A)
    df.index = np.arange(1, L + 1)
    df.index.name = "position"
    return df, L


def plot_sequence_logo(freq_df: pd.DataFrame, outpath: Path, *, style: PlotStyle, title: str) -> None:
    set_plot_style(style)
    # freq_df is a probability matrix already (position x base).
    # Convert to information content (bits); default background is uniform.
    info_df = logomaker.transform_matrix(freq_df, from_type="probability", to_type="information")

    plt.figure(figsize=(style.rect_width_cm / 2.54, style.rect_height_cm / 2.54))
    ax = plt.gca()
    logo = logomaker.Logo(info_df, ax=ax, color_scheme="classic")
    _ = logo
    ax.set_title(title.replace(" (", "\n("))
    ax.set_xlabel("Position")
    ax.set_ylabel("Bits")
    savefig(outpath, dpi=style.dpi)


def _relative_bits_matrix(freq_df: pd.DataFrame, background_df: pd.DataFrame, *, eps: float = 1e-9) -> pd.DataFrame:
    """
    Per-position bits logo normalized to an empirical background.

    For each position i and base b:
      weight(i,b) = p(i,b) * log2( p(i,b) / q(i,b) )

    where p is the selected-set probability matrix and q is the background/library probability matrix.
    This is the per-symbol contribution to KL divergence (in bits). Values can be negative when a base is
    depleted relative to background.
    """
    p = freq_df.copy()
    q = background_df.copy()
    # Align positions/bases
    q = q.reindex(index=p.index, columns=p.columns)
    p = p.astype(float)
    q = q.astype(float)
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)
    w = p * (np.log2(p) - np.log2(q))
    w.index = freq_df.index
    w.columns = freq_df.columns
    return w


def plot_sequence_logo_vs_background(
    freq_df: pd.DataFrame,
    background_df: pd.DataFrame,
    outpath: Path,
    *,
    style: PlotStyle,
    title: str,
) -> None:
    """
    Bits logo normalized to the empirical background (full library).
    Y-axis is "Bits vs background" (KL contribution per symbol).
    """
    set_plot_style(style)
    bits_df = _relative_bits_matrix(freq_df, background_df)
    plt.figure(figsize=(style.rect_width_cm / 2.54, style.rect_height_cm / 2.54))
    ax = plt.gca()
    logo = logomaker.Logo(bits_df, ax=ax, color_scheme="classic")
    _ = logo
    ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.6)
    ax.set_title(title.replace(" (", "\n("))
    ax.set_xlabel("Position")
    ax.set_ylabel("Bits vs background")
    savefig(outpath, dpi=style.dpi)


def pca_2d(
    X: sparse.csr_matrix,
    *,
    random_state: int = 0,
    mode: str = "auto",
    max_dense_cells: int = 50_000_000,
) -> tuple[np.ndarray, str]:
    """
    Compute a 2D linear embedding from the raw one-hot matrix.

    - mode="dense": use sklearn.PCA on a dense array (true PCA; may be huge)
    - mode="svd": use TruncatedSVD directly on the sparse one-hot (PCA-like for sparse)
    - mode="auto": use dense PCA only if matrix is small enough; otherwise SVD

    Returns (coords, method_name) where method_name is "PCA" or "TruncatedSVD".
    """
    mode = mode.lower().strip()
    if mode not in {"auto", "dense", "svd"}:
        raise ValueError("pca mode must be one of: auto, dense, svd")

    if sparse.issparse(X):
        n_cells = int(X.shape[0]) * int(X.shape[1])
        if mode == "dense" or (mode == "auto" and n_cells <= max_dense_cells):
            Xd = X.toarray().astype(np.float32, copy=False)
            pca = PCA(n_components=2, random_state=random_state)
            return pca.fit_transform(Xd), "PCA"
        svd = TruncatedSVD(n_components=2, random_state=random_state)
        return svd.fit_transform(X), "TruncatedSVD"

    pca = PCA(n_components=2, random_state=random_state)
    return pca.fit_transform(X), "PCA"


def elbow_kmeans(X2: np.ndarray, k_values: list[int], *, random_state: int = 0) -> pd.DataFrame:
    rows = []
    for k in k_values:
        km = KMeans(n_clusters=k, n_init="auto", random_state=random_state)
        km.fit(X2)
        rows.append({"k": k, "inertia": float(km.inertia_)})
    return pd.DataFrame(rows)

def kneedle_k_from_elbow(df_elbow: pd.DataFrame) -> int:
    """
    Simple Kneedle-style knee detection for a decreasing curve (inertia vs k).

    Steps:
    - Normalize k and inertia to [0, 1]
    - Convert decreasing curve to increasing via y_inc = 1 - y
    - Knee index is argmax(y_inc - x)

    This is the core idea behind Kneedle (Satopaa et al.) without smoothing.
    """
    df = df_elbow.sort_values("k").reset_index(drop=True)
    ks = df["k"].to_numpy(dtype=float)
    ys = df["inertia"].to_numpy(dtype=float)
    if len(ks) < 3:
        return int(df["k"].iloc[-1])

    x = (ks - ks.min()) / (ks.max() - ks.min())
    y = (ys - ys.min()) / (ys.max() - ys.min())  # decreasing typically
    y_inc = 1.0 - y
    g = y_inc - x
    idx = int(np.argmax(g))
    return int(df["k"].iloc[idx])


def plot_elbow(df_elbow: pd.DataFrame, outpath: Path, *, style: PlotStyle, knee_k: int | None = None) -> None:
    set_plot_style(style)
    plt.figure(figsize=(style.rect_width_cm / 2.54, style.rect_height_cm / 2.54))
    ax = sns.lineplot(data=df_elbow, x="k", y="inertia", marker="o", color="black", alpha=0.5)
    if knee_k is not None:
        ax.axvline(knee_k, color="black", linestyle="--", linewidth=1.0, alpha=0.8)
    ax.set_title("Elbow plot\n(KMeans)")
    ax.set_xlabel("k (clusters)")
    ax.set_ylabel("Inertia")
    savefig(outpath, dpi=style.dpi)


def plot_fe_density(df: pd.DataFrame, outpath: Path, *, style: PlotStyle) -> None:
    set_plot_style(style)
    plt.figure(figsize=(style.rect_width_cm / 2.54, style.rect_height_cm / 2.54))
    fe = to_float_series(df["FE"]) if "FE" in df.columns else pd.Series([], dtype=float)
    fe = fe.dropna()
    ax = sns.kdeplot(x=fe, fill=True)
    ax.set_title("FE density")
    ax.set_xlabel("FE")
    ax.set_ylabel("Density")
    savefig(outpath, dpi=style.dpi)


def plot_fe_density_by_split(
    df: pd.DataFrame,
    outpath: Path,
    *,
    style: PlotStyle,
    split_col: str = "Split",
) -> None:
    """Overlay KDE of FE for each split (e.g. cv1..cv5, test)."""
    if split_col not in df.columns or "FE" not in df.columns:
        return
    d = df[[split_col, "FE"]].copy()
    d["FE"] = to_float_series(d["FE"])
    d = d.dropna(subset=["FE"])
    d = d[d[split_col].notna()]
    if d.empty:
        return
    d[split_col] = d[split_col].astype(str)
    order = sorted(d[split_col].unique().tolist())
    set_plot_style(style)
    plt.figure(figsize=(style.rect_width_cm / 2.54, style.rect_height_cm / 2.54))
    ax = sns.kdeplot(
        data=d,
        x="FE",
        hue=split_col,
        hue_order=order,
        common_norm=False,
        linewidth=1.0,
    )
    ax.set_title("FE density by split")
    ax.set_xlabel("FE")
    ax.set_ylabel("Density")
    savefig(outpath, dpi=style.dpi)


def plot_fe_density_by_locus(df: pd.DataFrame, outpath: Path, *, style: PlotStyle) -> None:
    """Overlay KDE of FE for each Locus category."""
    if "Locus" not in df.columns or "FE" not in df.columns:
        return
    d = df[["Locus", "FE"]].copy()
    d["FE"] = to_float_series(d["FE"])
    d = d.dropna(subset=["FE", "Locus"])
    if d.empty:
        return
    d["Locus"] = d["Locus"].astype(str)
    order = sorted(d["Locus"].unique().tolist())
    set_plot_style(style)
    plt.figure(figsize=(style.rect_width_cm / 2.54, style.rect_height_cm / 2.54))
    ax = sns.kdeplot(
        data=d,
        x="FE",
        hue="Locus",
        hue_order=order,
        common_norm=False,
        linewidth=1.0,
    )
    ax.set_title("FE density by Locus")
    ax.set_xlabel("FE")
    ax.set_ylabel("Density")
    savefig(outpath, dpi=style.dpi)


def plot_fe_binned_sequence_logos(
    df: pd.DataFrame,
    logos_dir: Path,
    *,
    style: PlotStyle,
    n_bins: int = 10,
    log: Callable[[str], None] | None = None,
) -> None:
    """
    Equal-width bins over the observed FE range; one sequence logo per bin (bits / logomaker).
    Writes frequency CSVs and a small summary table.
    """
    if "SpacerSeq" not in df.columns or "FE" not in df.columns:
        return
    n_bins = max(2, int(n_bins))
    sub = df[["SpacerSeq", "FE"]].copy()
    sub["FE"] = to_float_series(sub["FE"])
    sub = sub[sub["FE"].notna() & sub["SpacerSeq"].notna()]
    if len(sub) < n_bins:
        if log:
            log(f"Skipping FE-binned logos: only n={len(sub)} rows with FE+SpacerSeq")
        return
    lo, hi = float(sub["FE"].min()), float(sub["FE"].max())
    if not np.isfinite(lo) or not np.isfinite(hi) or (hi - lo) <= 0:
        if log:
            log("Skipping FE-binned logos: constant or invalid FE range")
        return
    edges = np.linspace(lo, hi, n_bins + 1)
    sub["fe_bin"] = pd.cut(sub["FE"], bins=edges, include_lowest=True, labels=False)
    sub = sub.dropna(subset=["fe_bin"])
    sub["fe_bin"] = sub["fe_bin"].astype(int)

    meta_rows: list[dict[str, object]] = []
    for b in range(n_bins):
        part = sub[sub["fe_bin"] == b]
        n = len(part)
        left, right = float(edges[b]), float(edges[b + 1])
        meta_rows.append({"fe_bin": b + 1, "n": n, "fe_min": left, "fe_max": right})
        if n == 0:
            if log:
                log(f"FE bin {b + 1}/{n_bins} (FE [{left:.6g}, {right:.6g}]): n=0; skipping logo")
            continue
        bn = f"fe_bin{b + 1:02d}"
        freq_b, _ = dna_position_frequency_matrix(part["SpacerSeq"])
        freq_b.to_csv(logos_dir / f"logo_{bn}_freq.csv")
        plot_sequence_logo(
            freq_b,
            logos_dir / f"logo_{bn}.png",
            style=style,
            title=f"Sequence logo — FE bin {b + 1}/{n_bins}\n[{left:.4g}, {right:.4g}] (n={n})",
        )
        if log:
            log(f"Saved FE-binned logo bin {b + 1}/{n_bins} (n={n})")

    pd.DataFrame(meta_rows).to_csv(logos_dir / "logo_fe_bins_summary.csv", index=False)


def plot_cluster_size_density(cluster_sizes: pd.Series, outpath: Path, *, style: PlotStyle) -> None:
    set_plot_style(style)
    plt.figure(figsize=(style.rect_width_cm / 2.54, style.rect_height_cm / 2.54))
    sizes = pd.to_numeric(cluster_sizes, errors="coerce").dropna().astype(int)
    ax = sns.histplot(sizes, stat="density", bins=30, color="black", alpha=0.5)
    sns.kdeplot(sizes, color="black", alpha=0.5, ax=ax)
    ax.set_title("Cluster size distribution\n(KMeans cluster counts)")
    ax.set_xlabel("Sequences per cluster")
    ax.set_ylabel("Density")
    savefig(outpath, dpi=style.dpi)


def plot_embedding_clusters(
    df_plot: pd.DataFrame,
    outpath: Path,
    *,
    style: PlotStyle,
    title: str,
    x_col: str,
    y_col: str,
) -> None:
    set_plot_style(style)
    plt.figure(figsize=(style.width_cm / 2.54, style.height_cm / 2.54))
    ax = sns.scatterplot(
        data=df_plot,
        x=x_col,
        y=y_col,
        hue="cluster",
        palette="tab10",
        s=15,
        edgecolor="black",
        linewidth=0.3,
    )
    ax.set_title(title)
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    leg = ax.get_legend()
    if leg is not None:
        leg.remove()
    savefig(outpath, dpi=style.dpi)


def plot_embedding_categorical(
    df_plot: pd.DataFrame,
    outpath: Path,
    *,
    style: PlotStyle,
    title: str,
    x_col: str,
    y_col: str,
    category_col: str,
    max_legend_items: int = 20,
) -> None:
    set_plot_style(style)
    n_cats = int(pd.Series(df_plot[category_col]).nunique(dropna=True))
    show_legend = n_cats <= max_legend_items
    if show_legend:
        # Keep the main axes square and place legend in a side panel.
        legend_w_cm = 5.5
        legend_pad_cm = 0.4
        total_w_cm = style.width_cm + legend_pad_cm + legend_w_cm
        fig = plt.figure(figsize=(total_w_cm / 2.54, style.height_cm / 2.54))
        gs = fig.add_gridspec(
            1,
            3,
            width_ratios=[style.width_cm, legend_pad_cm, legend_w_cm],
            wspace=0.0,
        )
        ax = fig.add_subplot(gs[0, 0])
        legax = fig.add_subplot(gs[0, 2])
        legax.axis("off")

        sns.scatterplot(
            data=df_plot,
            x=x_col,
            y=y_col,
            hue=category_col,
            palette="tab20",
            s=15,
            edgecolor="black",
            linewidth=0.3,
            legend="brief",
            ax=ax,
        )
        ax.set_title(title)
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)

        handles, labels = ax.get_legend_handles_labels()
        leg = ax.get_legend()
        if leg is not None:
            leg.remove()
        # Drop seaborn's automatic legend title entry if present
        if labels and labels[0] == category_col:
            handles, labels = handles[1:], labels[1:]
        legax.legend(
            handles,
            labels,
            title=category_col,
            loc="center left",
            frameon=False,
            borderaxespad=0.0,
            handletextpad=0.4,
            labelspacing=0.4,
            fontsize=10,
            title_fontsize=11,
        )
        savefig(outpath, dpi=style.dpi, tight=False)
    else:
        plt.figure(figsize=(style.width_cm / 2.54, style.height_cm / 2.54))
        ax = sns.scatterplot(
            data=df_plot,
            x=x_col,
            y=y_col,
            hue=category_col,
            palette="tab20",
            s=15,
            edgecolor="black",
            linewidth=0.3,
            legend=False,
        )
        ax.set_title(title)
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        savefig(outpath, dpi=style.dpi)


def plot_embedding_continuous(
    df_plot: pd.DataFrame,
    outpath: Path,
    *,
    style: PlotStyle,
    title: str,
    x_col: str,
    y_col: str,
    value_col: str,
) -> None:
    set_plot_style(style)
    # Keep the *main axes* at a fixed size (width_cm x height_cm) and
    # add extra figure width for the colorbar so the scatter area isn't squished.
    total_w_cm = style.width_cm + style.cbar_pad_cm + style.cbar_width_cm
    fig = plt.figure(figsize=(total_w_cm / 2.54, style.height_cm / 2.54))
    gs = fig.add_gridspec(
        1,
        3,
        width_ratios=[style.width_cm, style.cbar_pad_cm, style.cbar_width_cm],
        wspace=0.0,
    )
    ax = fig.add_subplot(gs[0, 0])
    cax = fig.add_subplot(gs[0, 2])

    sns.scatterplot(
        data=df_plot,
        x=x_col,
        y=y_col,
        hue=value_col,
        palette="viridis",
        s=15,
        edgecolor="black",
        linewidth=0.3,
        legend=False,
        ax=ax,
    )
    ax.set_title(title)
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)

    import matplotlib.cm as cm
    import matplotlib.colors as colors

    vmin = float(df_plot[value_col].min())
    vmax = float(df_plot[value_col].max())
    norm = colors.Normalize(vmin=vmin, vmax=vmax)
    sm = cm.ScalarMappable(cmap="viridis", norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label(value_col)

    savefig(outpath, dpi=style.dpi, tight=False)


def plot_cumulative_explained_variance(
    explained_variance_ratio: np.ndarray,
    outpath: Path,
    *,
    style: PlotStyle,
    title: str,
) -> None:
    set_plot_style(style)
    plt.figure(figsize=(style.rect_width_cm / 2.54, style.rect_height_cm / 2.54))
    cum = np.cumsum(explained_variance_ratio)
    xs = np.arange(1, len(cum) + 1)
    ax = sns.lineplot(x=xs, y=cum, marker="o", color="black", alpha=0.5)
    ax.axhline(0.9, color="black", linestyle="--", linewidth=1.0, alpha=0.8)
    ax.set_title(title)
    ax.set_xlabel("Number of components")
    ax.set_ylabel("Cumulative explained\nvariance ratio")
    ax.set_ylim(0, 1.0)
    savefig(outpath, dpi=style.dpi)

def smallest_d_for_cumvar(explained_variance_ratio: np.ndarray, target: float) -> int | None:
    if explained_variance_ratio.size == 0:
        return None
    cum = np.cumsum(explained_variance_ratio)
    idx = np.searchsorted(cum, target, side="left")
    if idx >= len(cum):
        return None
    return int(idx + 1)  # 1-indexed components

def embed_tsne(
    X: np.ndarray,
    *,
    random_state: int = 0,
    perplexity: float = 30.0,
    early_exaggeration: float = 12.0,
    max_iter: int = 2000,
) -> np.ndarray:
    tsne = TSNE(
        n_components=2,
        init="pca",
        learning_rate="auto",
        perplexity=perplexity,
        early_exaggeration=early_exaggeration,
        max_iter=max_iter,
        random_state=random_state,
    )
    return tsne.fit_transform(X)


def embed_umap(X: np.ndarray, *, random_state: int = 0, n_neighbors: int = 15, min_dist: float = 0.1) -> np.ndarray:
    try:
        from umap import UMAP  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "UMAP requested but 'umap-learn' is not installed. Install it (e.g., `pip install umap-learn`)."
        ) from e

    um = UMAP(n_components=2, random_state=random_state, n_neighbors=n_neighbors, min_dist=min_dist)
    return um.fit_transform(X)


def assign_grouped_splits(
    group_ids: np.ndarray,
    *,
    test_frac: float = 0.15,
    n_folds: int = 5,
    random_state: int = 42,
) -> tuple[np.ndarray, dict[int, str]]:
    """
    Assign each row to one of: "test", "cv1".."cv{n_folds}" such that all rows
    from the same group_id stay in the same split.
    """
    if not (0.0 < test_frac < 1.0):
        raise ValueError("test_frac must be between 0 and 1")
    if n_folds < 2:
        raise ValueError("n_folds must be >= 2")

    group_ids = np.asarray(group_ids, dtype=int)
    uniq, counts = np.unique(group_ids, return_counts=True)

    rng = np.random.default_rng(int(random_state))
    order = rng.permutation(len(uniq))
    uniq = uniq[order]
    counts = counts[order]

    n = int(len(group_ids))
    target_test = int(round(test_frac * n))

    group_to_split: dict[int, str] = {}

    # Greedy fill test set by whole clusters
    test_total = 0
    for g, c in zip(uniq, counts):
        # Always take at least one group if target_test rounds to 0
        if test_total < target_test or (target_test == 0 and test_total == 0):
            group_to_split[int(g)] = "test"
            test_total += int(c)

    # Assign remaining groups to CV folds with greedy balancing by size
    fold_totals = [0 for _ in range(n_folds)]
    remaining = [(int(g), int(c)) for g, c in zip(uniq, counts) if int(g) not in group_to_split]
    remaining.sort(key=lambda t: t[1], reverse=True)

    for g, c in remaining:
        i = int(np.argmin(fold_totals))
        fold_totals[i] += c
        group_to_split[g] = f"cv{i+1}"

    split_labels = np.array([group_to_split[int(g)] for g in group_ids], dtype=object)
    return split_labels, group_to_split


def assign_random_splits(
    n_rows: int,
    *,
    test_frac: float = 0.15,
    n_folds: int = 5,
    random_state: int = 42,
) -> np.ndarray:
    """
    Randomly assign each row to one of: "test", "cv1".."cv{n_folds}".
    """
    if not (0.0 < test_frac < 1.0):
        raise ValueError("test_frac must be between 0 and 1")
    if n_folds < 2:
        raise ValueError("n_folds must be >= 2")
    n_rows = int(n_rows)
    if n_rows <= 0:
        return np.array([], dtype=object)

    rng = np.random.default_rng(int(random_state))
    idx = rng.permutation(n_rows)

    n_test = int(round(test_frac * n_rows))
    test_idx = idx[:n_test]
    cv_idx = idx[n_test:]

    labels = np.empty(n_rows, dtype=object)
    labels[test_idx] = "test"

    # Split remaining roughly evenly into folds
    folds = np.array_split(cv_idx, n_folds)
    for i, fold in enumerate(folds, start=1):
        labels[fold] = f"cv{i}"

    return labels


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input",
        default="data/251129_P8_120Complete_dataset_woCtrl_exogenous.xlsx",
        help="Path to the Excel file",
    )
    ap.add_argument(
        "--locus-excel",
        default="data/260317_P8_120Complete_dataset_woCtrl_exogenous_locus.xlsx",
        help="Excel that contains a 'Locus' column; merged onto input by SpacerSeq.",
    )
    ap.add_argument("--sheet", default=0, help="Excel sheet name or index (default: 0)")
    ap.add_argument("--output-dir", default="output/data_processing", help="Directory to write outputs")
    ap.add_argument("--min-k", type=int, default=2, help="Minimum k for elbow/cluster plots")
    ap.add_argument("--max-k", type=int, default=200, help="Maximum k for elbow/cluster plots")
    ap.add_argument("--random-state", type=int, default=42)
    ap.add_argument("--width-cm", type=float, default=12.0)
    ap.add_argument("--height-cm", type=float, default=12.0)
    ap.add_argument("--rect-width-cm", type=float, default=16.0, help="Figure width for non-cluster plots")
    ap.add_argument("--rect-height-cm", type=float, default=12.0, help="Figure height for non-cluster plots")
    ap.add_argument("--svd-components", type=int, default=50, help="SVD dims before KMeans/t-SNE/UMAP")
    ap.add_argument(
        "--pca-mode",
        choices=["auto", "dense", "svd"],
        default="auto",
        help="How to compute PC1/PC2 from one-hot (dense=PCA, svd=TruncatedSVD)",
    )
    ap.add_argument(
        "--variance-components",
        type=int,
        default=640,
        help="Number of components for cumulative explained variance plot",
    )
    ap.add_argument(
        "--no-tsne",
        dest="tsne",
        action="store_false",
        default=True,
        help="Do NOT compute t-SNE embedding/plot (default: compute)",
    )
    ap.add_argument(
        "--no-umap",
        dest="umap",
        action="store_false",
        default=True,
        help="Do NOT compute UMAP embedding/plot (default: compute; requires umap-learn)",
    )
    ap.add_argument("--tsne-perplexity", type=float, default=15.0)
    ap.add_argument("--tsne-early-exaggeration", type=float, default=24.0)
    ap.add_argument("--tsne-max-iter", type=int, default=2000)
    ap.add_argument("--umap-n-neighbors", type=int, default=5)
    ap.add_argument("--umap-min-dist", type=float, default=0.0)
    ap.add_argument(
        "--splits-excel",
        default="",
        help="Optional Excel with 'Split' column to make per-fold sequence logos",
    )
    ap.add_argument(
        "--activity-bins",
        type=int,
        default=10,
        help="Number of equal-width FE bins for activity-binned sequence logos",
    )
    ap.add_argument(
        "--no-standardize-embed",
        dest="standardize_embed",
        action="store_false",
        default=True,
        help="Do NOT standardize SVD features before t-SNE/UMAP (default: standardize)",
    )
    ap.add_argument("--test-frac", type=float, default=0.15, help="Fraction of data for held-out test split")
    ap.add_argument("--cv-folds", type=int, default=5, help="Number of CV folds for non-test data")
    ap.add_argument(
        "--no-export-excel",
        dest="export_excel",
        action="store_false",
        default=True,
        help="Do NOT write the augmented Excel (default: export)",
    )
    ap.add_argument(
        "--output-excel",
        default="",
        help="Path for exported Excel (default: output-dir/<input>_with_splits.xlsx)",
    )
    ap.add_argument(
        "--plot-all-k",
        action="store_true",
        help="Plot PCA/t-SNE/UMAP for all k in [min-k, max-k] (default: only knee k)",
    )
    args = ap.parse_args(argv)

    t0 = time.time()

    def log(msg: str) -> None:
        dt = time.time() - t0
        print(f"[{dt:8.1f}s] {msg}", flush=True)

    in_path = Path(args.input)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    log(f"Loading Excel: {in_path} (sheet={args.sheet})")
    df = pd.read_excel(in_path, sheet_name=args.sheet, engine="openpyxl")
    log(f"Loaded dataframe: shape={df.shape}")

    # Optional merge: add Locus column from a second dataset (by SpacerSeq).
    locus_path = Path(str(args.locus_excel).strip()) if str(args.locus_excel).strip() else None
    if locus_path is not None and locus_path.exists():
        if "SpacerSeq" not in df.columns:
            log("locus-excel provided but input is missing SpacerSeq; skipping locus merge")
        else:
            log(f"Loading locus Excel: {locus_path}")
            df_locus = pd.read_excel(locus_path, engine="openpyxl")
            if "SpacerSeq" not in df_locus.columns or "Locus" not in df_locus.columns:
                log("locus-excel missing required columns (SpacerSeq, Locus); skipping locus merge")
            else:
                # Filter locus table down to sequences that exist in the main dataset.
                # This avoids propagating metadata for extra sequences (e.g., controls) that
                # may exist only in the locus file.
                df_m = df.copy()
                df_m["SpacerSeq"] = df_m["SpacerSeq"].where(df_m["SpacerSeq"].notna(), other=np.nan)
                main_seqs = set(df_m["SpacerSeq"].dropna().astype(str).tolist())

                df_l = df_locus[["SpacerSeq", "Locus"]].copy()
                df_l = df_l.dropna(subset=["SpacerSeq"])
                df_l["SpacerSeq"] = df_l["SpacerSeq"].astype(str)
                df_l["Locus"] = df_l["Locus"].astype(str)
                df_l = df_l[df_l["SpacerSeq"].isin(main_seqs)]
                # Deduplicate by SpacerSeq to avoid explosive merges.
                df_l = df_l.drop_duplicates(subset=["SpacerSeq"], keep="first")

                df = df_m.merge(df_l, on="SpacerSeq", how="left")
                log(f"Merged locus info: now shape={df.shape}; Locus non-null={int(df['Locus'].notna().sum())}")
    elif locus_path is not None:
        log("locus-excel path not found; skipping locus merge")

    (out_dir / "data_shape.txt").write_text(f"rows\t{df.shape[0]}\ncols\t{df.shape[1]}\n")
    df.head(50).to_csv(out_dir / "data_head_50.csv", index=False)
    pd.DataFrame({"column": df.columns, "dtype": [str(t) for t in df.dtypes]}).to_csv(
        out_dir / "data_columns.csv", index=False
    )
    log(f"Wrote inspection files to: {out_dir}")

    style = PlotStyle(
        width_cm=args.width_cm,
        height_cm=args.height_cm,
        rect_width_cm=args.rect_width_cm,
        rect_height_cm=args.rect_height_cm,
    )

    if "FE" in df.columns:
        log("Plotting FE density")
        plot_fe_density(df, out_dir / "fe_density.png", style=style)
        log("Saved FE density plot")

        if "Locus" in df.columns and df["Locus"].notna().any():
            log("Plotting FE density by Locus")
            plot_fe_density_by_locus(df, out_dir / "fe_density_by_locus.png", style=style)
            log("Saved fe_density_by_locus.png")

    if "SpacerSeq" in df.columns:
        log("One-hot encoding SpacerSeq")
        seqs = df["SpacerSeq"]
        X, L = one_hot_encode_dna(seqs)
        log(f"One-hot matrix: shape={X.shape} nnz={X.nnz}")

        log(f"Computing 2D PCA from raw one-hot (mode={args.pca_mode})")
        pcs, pc_method = pca_2d(X, random_state=args.random_state, mode=args.pca_mode)
        if pc_method == "TruncatedSVD":
            x1, x2 = "SVD1", "SVD2"
        else:
            x1, x2 = "PC1", "PC2"
        df_pcs = pd.DataFrame({x1: pcs[:, 0], x2: pcs[:, 1]})
        log("Computed PC1/PC2")

        # Plot output subfolders
        pca_dir = out_dir / "pca"
        tsne_dir = out_dir / "tsne"
        umap_dir = out_dir / "umap"
        pca_dir.mkdir(parents=True, exist_ok=True)
        tsne_dir.mkdir(parents=True, exist_ok=True)
        umap_dir.mkdir(parents=True, exist_ok=True)

        fe_series: pd.Series | None = None
        if "FE" in df.columns:
            fe_series = to_float_series(df["FE"])

        # Sequence logo(s)
        log("Computing sequence logo for full library")
        freq_all, _ = dna_position_frequency_matrix(df["SpacerSeq"])
        logos_dir = out_dir / "logos"
        logos_dir.mkdir(parents=True, exist_ok=True)
        freq_all.to_csv(logos_dir / "logo_full_library_freq.csv")
        plot_sequence_logo(freq_all, logos_dir / "logo_full_library.png", style=style, title="Full library sequence logo")
        log("Saved full library sequence logo")

        # Sequence logos for the most active sequences (top-N by FE)
        if "FE" in df.columns:
            top_ns = [50, 100, 200, 500, 1000]
            sub_top = df[["SpacerSeq", "FE"]].copy()
            sub_top["FE"] = to_float_series(sub_top["FE"])
            sub_top = sub_top.dropna(subset=["SpacerSeq", "FE"])
            if not sub_top.empty:
                sub_top = sub_top.sort_values("FE", ascending=False)
                for n_top in top_ns:
                    part = sub_top.head(int(n_top))
                    if part.empty:
                        continue
                    freq_top, _ = dna_position_frequency_matrix(part["SpacerSeq"])
                    freq_top.to_csv(logos_dir / f"logo_top{n_top}_by_FE_freq.csv")
                    plot_sequence_logo(
                        freq_top,
                        logos_dir / f"logo_top{n_top}_by_FE.png",
                        style=style,
                        title=f"Sequence logo — top {n_top} by FE (n={len(part)})",
                    )
                    # Background-normalized bits logo: enrichment/depletion relative to full library composition.
                    plot_sequence_logo_vs_background(
                        freq_top,
                        freq_all,
                        logos_dir / f"logo_top{n_top}_by_FE_vs_library.png",
                        style=style,
                        title=f"Bits vs library — top {n_top} by FE (n={len(part)})",
                    )
                log("Saved top-N activity sequence logos (top 50/100/200/500/1000 by FE)")

        # Per-locus logos, if available from locus-excel merge
        if "Locus" in df.columns and df["Locus"].notna().any():
            locus_dir = logos_dir / "by_locus"
            locus_dir.mkdir(parents=True, exist_ok=True)
            for locus, subdf_l in df[df["Locus"].notna()].groupby("Locus"):
                locus_str = str(locus)
                freq_l, _ = dna_position_frequency_matrix(subdf_l["SpacerSeq"])
                safe = locus_str.replace("/", "_").replace(" ", "_")
                freq_l.to_csv(locus_dir / f"logo_locus_{safe}_freq.csv")
                plot_sequence_logo(
                    freq_l,
                    locus_dir / f"logo_locus_{safe}.png",
                    style=style,
                    title=f"Sequence logo — Locus {locus_str} (n={len(subdf_l)})",
                )
            log("Saved per-locus sequence logos")

        if "FE" in df.columns:
            log(f"Sequence logos binned by FE ({args.activity_bins} equal-width bins across observed range)")
            plot_fe_binned_sequence_logos(
                df,
                logos_dir,
                style=style,
                n_bins=args.activity_bins,
                log=log,
            )
            log("Saved FE-binned sequence logos (see logo_fe_bins_summary.csv)")

        fe_density_by_split_done = False
        if args.splits_excel.strip():
            split_path = Path(args.splits_excel.strip())
            if split_path.exists():
                log(f"Loading splits Excel for per-fold logos: {split_path}")
                df_s = pd.read_excel(split_path, engine="openpyxl")
                if "Split" in df_s.columns and "FE" in df_s.columns:
                    log("Plotting FE density by split (splits-excel)")
                    plot_fe_density_by_split(df_s, out_dir / "fe_density_by_split.png", style=style)
                    log("Saved fe_density_by_split.png")
                    fe_density_by_split_done = True
                if "Split" in df_s.columns and "SpacerSeq" in df_s.columns:
                    for split_name, subdf in df_s.groupby("Split"):
                        if pd.isna(split_name):
                            continue
                        split_name_str = str(split_name)
                        freq_split, _ = dna_position_frequency_matrix(subdf["SpacerSeq"])
                        freq_split.to_csv(logos_dir / f"logo_{split_name_str}_freq.csv")
                        plot_sequence_logo(
                            freq_split,
                            logos_dir / f"logo_{split_name_str}.png",
                            style=style,
                            title=f"Sequence logo — {split_name_str}",
                        )
                    log("Saved per-fold sequence logos (Split)")
                else:
                    log("splits-excel missing required columns (Split, SpacerSeq); skipping per-fold logos")

                # Per-cluster logos (KMeansCluster) + FE boxplot if available
                if "KMeansCluster" in df_s.columns and "SpacerSeq" in df_s.columns:
                    log("Computing sequence logos per KMeansCluster")
                    for clust_id, subdf_c in df_s.groupby("KMeansCluster"):
                        if pd.isna(clust_id):
                            continue
                        cid_int = int(clust_id)
                        cid_str = f"cluster{cid_int}"
                        freq_c, _ = dna_position_frequency_matrix(subdf_c["SpacerSeq"])
                        freq_c.to_csv(logos_dir / f"logo_{cid_str}_freq.csv")
                        plot_sequence_logo(
                            freq_c,
                            logos_dir / f"logo_{cid_str}.png",
                            style=style,
                            title=f"Sequence logo — KMeansCluster {cid_int}",
                        )
                    log("Saved per-cluster sequence logos (KMeansCluster)")

                    if "FE" in df_s.columns:
                        log("Plotting FE boxplot per KMeansCluster")
                        df_box = df_s.copy()
                        df_box["KMeansCluster"] = pd.to_numeric(df_box["KMeansCluster"], errors="coerce").astype("Int64")
                        df_box = df_box[df_box["KMeansCluster"].notna()]
                        df_box["FE"] = to_float_series(df_box["FE"])
                        df_box = df_box[df_box["FE"].notna()]
                        order = sorted(df_box["KMeansCluster"].dropna().unique().tolist())

                        set_plot_style(style)
                        plt.figure(figsize=(style.rect_width_cm / 2.54, style.rect_height_cm / 2.54))
                        ax = sns.boxplot(
                            data=df_box,
                            x="KMeansCluster",
                            y="FE",
                            order=order,
                            color="white",
                            fliersize=1.0,
                            linewidth=0.7,
                        )
                        ax.set_title("FE distribution per KMeansCluster")
                        ax.set_xlabel("KMeansCluster")
                        ax.set_ylabel("FE")
                        savefig(out_dir / "fe_boxplot_per_cluster.png", dpi=style.dpi)
                        log("Saved FE boxplot per cluster")
            else:
                log("splits-excel path not found; skipping per-fold logos")

        if not fe_density_by_split_done and "Split" in df.columns and "FE" in df.columns:
            log("Plotting FE density by split (input dataframe)")
            plot_fe_density_by_split(df, out_dir / "fe_density_by_split.png", style=style)
            log("Saved fe_density_by_split.png")

        # Cumulative explained variance (on one-hot; uses TruncatedSVD for sparse input)
        var_dim = max(2, int(args.variance_components))
        var_dim = min(var_dim, 640)
        var_dim = min(var_dim, max(2, X.shape[0] - 1), max(2, X.shape[1] - 1))
        log(f"Computing cumulative explained variance (variance_components={var_dim})")
        svd_var = TruncatedSVD(n_components=var_dim, random_state=args.random_state)
        svd_var.fit(X)
        np.savetxt(out_dir / "svd_explained_variance_ratio.csv", svd_var.explained_variance_ratio_, delimiter=",")
        plot_cumulative_explained_variance(
            svd_var.explained_variance_ratio_,
            out_dir / "svd_cumulative_explained_variance.png",
            style=style,
            title="Cumulative explained variance\n(TruncatedSVD on one-hot SpacerSeq)",
        )
        d90 = smallest_d_for_cumvar(svd_var.explained_variance_ratio_, 0.90)
        if d90 is None:
            log("Cumulative variance: did not reach 90% within computed components")
        else:
            log(f"Cumulative variance: smallest d for 90% is {d90}")
            (out_dir / "svd_cumvar_thresholds.txt").write_text(f"d_for_0.90\t{d90}\n")
        log("Saved cumulative explained variance plot")

        # Choose SVD dimensionality for clustering + embeddings.
        # Default: use the smallest d reaching 90% cumulative variance when available.
        requested_svd = max(2, int(args.svd_components))
        cluster_dim = int(d90) if (d90 is not None and d90 >= 2) else requested_svd
        log(f"Computing SVD representation for clustering/embeddings (ClusterSVD={cluster_dim})")
        svd = TruncatedSVD(n_components=cluster_dim, random_state=args.random_state)
        Xr = svd.fit_transform(X)
        log("Computed clustering/embedding SVD dense representation")

        X_embed = Xr
        if args.standardize_embed and (args.tsne or args.umap):
            log("Standardizing SVD features for t-SNE/UMAP")
            X_embed = StandardScaler().fit_transform(X_embed)

        tsne_xy: np.ndarray | None = None
        umap_xy: np.ndarray | None = None
        if args.tsne:
            log(
                f"Computing t-SNE (perplexity={args.tsne_perplexity}, early_exaggeration={args.tsne_early_exaggeration}, max_iter={args.tsne_max_iter})"
            )
            tsne_xy = embed_tsne(
                X_embed,
                random_state=args.random_state,
                perplexity=float(args.tsne_perplexity),
                early_exaggeration=float(args.tsne_early_exaggeration),
                max_iter=int(args.tsne_max_iter),
            )
            log("Computed t-SNE embedding")
        if args.umap:
            log(f"Computing UMAP (n_neighbors={args.umap_n_neighbors}, min_dist={args.umap_min_dist})")
            umap_xy = embed_umap(
                X_embed,
                random_state=args.random_state,
                n_neighbors=int(args.umap_n_neighbors),
                min_dist=float(args.umap_min_dist),
            )
            log("Computed UMAP embedding")

        # Color embeddings by FE (continuous), if available
        if fe_series is not None:
            log("Plotting embeddings colored by FE")
            fe_mask = fe_series.notna()

            if fe_mask.any():
                df_pca_fe = df_pcs.loc[fe_mask].copy()
                df_pca_fe["FE"] = fe_series.loc[fe_mask].to_numpy()
                plot_embedding_continuous(
                    df_pca_fe,
                    pca_dir / f"{'svd' if pc_method == 'TruncatedSVD' else 'pca'}_{x1.lower()}_{x2.lower()}_by_FE.png",
                    style=style,
                    title=f"{pc_method} (one-hot SpacerSeq)\nColored by FE",
                    x_col=x1,
                    y_col=x2,
                    value_col="FE",
                )

                if tsne_xy is not None:
                    df_tsne_fe = pd.DataFrame(
                        {"tSNE1": tsne_xy[:, 0], "tSNE2": tsne_xy[:, 1], "FE": fe_series.to_numpy()}
                    )
                    df_tsne_fe = df_tsne_fe.loc[fe_mask].copy()
                    plot_embedding_continuous(
                        df_tsne_fe,
                        tsne_dir / "tsne_1_2_by_FE.png",
                        style=style,
                        title="t-SNE (one-hot SpacerSeq)\nColored by FE",
                        x_col="tSNE1",
                        y_col="tSNE2",
                        value_col="FE",
                    )

                if umap_xy is not None:
                    df_umap_fe = pd.DataFrame(
                        {"UMAP1": umap_xy[:, 0], "UMAP2": umap_xy[:, 1], "FE": fe_series.to_numpy()}
                    )
                    df_umap_fe = df_umap_fe.loc[fe_mask].copy()
                    plot_embedding_continuous(
                        df_umap_fe,
                        umap_dir / "umap_1_2_by_FE.png",
                        style=style,
                        title="UMAP (one-hot SpacerSeq)\nColored by FE",
                        x_col="UMAP1",
                        y_col="UMAP2",
                        value_col="FE",
                    )

            log("Saved FE-colored embedding plots")

        # Color embeddings by Locus (categorical), if present
        if "Locus" in df.columns and df["Locus"].notna().any():
            log("Plotting embeddings colored by Locus")
            locus_series = df["Locus"].astype(str)
            locus_mask = df["Locus"].notna()

            if locus_mask.any():
                df_pca_l = df_pcs.loc[locus_mask].copy()
                df_pca_l["Locus"] = locus_series.loc[locus_mask].to_numpy()
                plot_embedding_categorical(
                    df_pca_l,
                    pca_dir / f"{'svd' if pc_method == 'TruncatedSVD' else 'pca'}_{x1.lower()}_{x2.lower()}_by_Locus.png",
                    style=style,
                    title=f"{pc_method} (one-hot SpacerSeq)\nColored by Locus",
                    x_col=x1,
                    y_col=x2,
                    category_col="Locus",
                )

                if tsne_xy is not None:
                    df_tsne_l = pd.DataFrame({"tSNE1": tsne_xy[:, 0], "tSNE2": tsne_xy[:, 1], "Locus": locus_series.to_numpy()})
                    df_tsne_l = df_tsne_l.loc[locus_mask].copy()
                    plot_embedding_categorical(
                        df_tsne_l,
                        tsne_dir / "tsne_1_2_by_Locus.png",
                        style=style,
                        title="t-SNE (one-hot SpacerSeq)\nColored by Locus",
                        x_col="tSNE1",
                        y_col="tSNE2",
                        category_col="Locus",
                    )

                if umap_xy is not None:
                    df_umap_l = pd.DataFrame({"UMAP1": umap_xy[:, 0], "UMAP2": umap_xy[:, 1], "Locus": locus_series.to_numpy()})
                    df_umap_l = df_umap_l.loc[locus_mask].copy()
                    plot_embedding_categorical(
                        df_umap_l,
                        umap_dir / "umap_1_2_by_Locus.png",
                        style=style,
                        title="UMAP (one-hot SpacerSeq)\nColored by Locus",
                        x_col="UMAP1",
                        y_col="UMAP2",
                        category_col="Locus",
                    )

            log("Saved Locus-colored embedding plots")

        k_values = list(range(args.min_k, args.max_k + 1))
        log(f"Computing elbow curve for k in [{args.min_k}, {args.max_k}]")
        df_elbow = elbow_kmeans(Xr, k_values, random_state=args.random_state)
        knee_k = kneedle_k_from_elbow(df_elbow)
        (out_dir / "kmeans_elbow_kneedle.txt").write_text(f"kneedle_k\t{knee_k}\n")
        df_elbow.to_csv(out_dir / "kmeans_elbow.csv", index=False)
        plot_elbow(df_elbow, out_dir / "kmeans_elbow.png", style=style, knee_k=knee_k)
        log("Saved elbow plot")

        # Final clustering at the elbow k
        log(f"Fitting final KMeans for split/labels (k={knee_k}, ClusterSVD={cluster_dim})")
        km_final = KMeans(n_clusters=int(knee_k), n_init="auto", random_state=args.random_state)
        cluster_labels = km_final.fit_predict(Xr).astype(int)

        # Export an augmented Excel (does not overwrite the input file)
        if args.export_excel:
            split_labels, _ = assign_grouped_splits(
                cluster_labels, test_frac=float(args.test_frac), n_folds=int(args.cv_folds), random_state=args.random_state
            )
            split_random = assign_random_splits(
                len(df),
                test_frac=float(args.test_frac),
                n_folds=int(args.cv_folds),
                random_state=args.random_state,
            )
            df_export = df.copy()
            df_export["KMeansCluster"] = cluster_labels
            df_export["Split"] = split_labels
            df_export["SplitRandom"] = split_random

            # Print counts per category for quick sanity checks
            log("Split counts (cluster-grouped):")
            for name, cnt in df_export["Split"].value_counts(dropna=False).sort_index().items():
                log(f"  {name}: {int(cnt)}")
            log("Split counts (random):")
            for name, cnt in df_export["SplitRandom"].value_counts(dropna=False).sort_index().items():
                log(f"  {name}: {int(cnt)}")

            # Save counts to CSV for downstream use
            df_split_counts = (
                pd.concat(
                    [
                        df_export["Split"].value_counts().rename("n").rename_axis("split").reset_index().assign(kind="cluster_grouped"),
                        df_export["SplitRandom"]
                        .value_counts()
                        .rename("n")
                        .rename_axis("split")
                        .reset_index()
                        .assign(kind="random"),
                    ],
                    ignore_index=True,
                )
                .sort_values(["kind", "split"])
                .reset_index(drop=True)
            )
            df_split_counts.to_csv(out_dir / "split_counts.csv", index=False)
            log("Wrote split counts table: split_counts.csv")

            # Cluster size distribution + plot
            cluster_sizes = pd.Series(cluster_labels).value_counts().sort_index()
            cluster_sizes.rename("n_sequences").rename_axis("cluster").to_csv(out_dir / "cluster_sizes.csv")
            plot_cluster_size_density(cluster_sizes, out_dir / "cluster_size_density.png", style=style)
            log("Wrote cluster sizes + density plot")

            out_excel = args.output_excel.strip()
            if not out_excel:
                stem = Path(args.input).stem
                out_excel = str(out_dir / f"{stem}_with_splits.xlsx")

            log(f"Writing augmented Excel: {out_excel}")
            with pd.ExcelWriter(out_excel, engine="openpyxl") as writer:
                df_export.to_excel(writer, index=False, sheet_name="data")
            log("Wrote augmented Excel with KMeansCluster + Split")

        plot_ks = k_values if args.plot_all_k else [knee_k]
        log(f"Plotting embeddings for k={plot_ks if args.plot_all_k else knee_k} (plot_all_k={args.plot_all_k})")

        for k in plot_ks:
            log(f"Clustering + plotting (k={k})")
            km = KMeans(n_clusters=k, n_init="auto", random_state=args.random_state)
            labels = km.fit_predict(Xr).astype(int).astype(str)

            df_plot_pca = df_pcs.copy()
            df_plot_pca["cluster"] = labels
            plot_embedding_clusters(
                df_plot_pca,
                pca_dir
                / f"{'svd' if pc_method == 'TruncatedSVD' else 'pca'}_{x1.lower()}_{x2.lower()}_kmeans_k{k}.png",
                style=style,
                title=f"{pc_method} (one-hot SpacerSeq)\nKMeans k={k} (ClusterSVD={cluster_dim})",
                x_col=x1,
                y_col=x2,
            )

            if tsne_xy is not None:
                df_tsne = pd.DataFrame({"tSNE1": tsne_xy[:, 0], "tSNE2": tsne_xy[:, 1], "cluster": labels})
                plot_embedding_clusters(
                    df_tsne,
                    tsne_dir / f"tsne_1_2_kmeans_k{k}.png",
                    style=style,
                    title=f"t-SNE (one-hot SpacerSeq)\nKMeans k={k} (EmbedSVD={cluster_dim})",
                    x_col="tSNE1",
                    y_col="tSNE2",
                )

            if umap_xy is not None:
                df_umap = pd.DataFrame({"UMAP1": umap_xy[:, 0], "UMAP2": umap_xy[:, 1], "cluster": labels})
                plot_embedding_clusters(
                    df_umap,
                    umap_dir / f"umap_1_2_kmeans_k{k}.png",
                    style=style,
                    title=f"UMAP (one-hot SpacerSeq)\nKMeans k={k} (EmbedSVD={cluster_dim})",
                    x_col="UMAP1",
                    y_col="UMAP2",
                )

        log("Done")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

