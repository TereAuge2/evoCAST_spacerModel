from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import sparse
from sklearn.decomposition import PCA, TruncatedSVD


@dataclass(frozen=True)
class PlotStyle:
    width_cm: float = 12.0
    height_cm: float = 12.0
    rect_width_cm: float = 16.0
    rect_height_cm: float = 12.0
    dpi: int = 300
    font_family: str = "Arial"
    font_size: int = 14


def set_plot_style(style: PlotStyle) -> None:
    mpl.rcParams.update(mpl.rcParamsDefault)
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


def savefig(path: Path, dpi: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sns.despine()
    plt.tight_layout()
    plt.grid(False)
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close()


def one_hot_encode_dna(seqs: pd.Series, *, alphabet: str = "ACGT", pad_char: str = "N") -> tuple[sparse.csr_matrix, int]:
    seq_list = ["" if pd.isna(x) else str(x) for x in seqs]
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


def fit_2d_linear(
    X: sparse.csr_matrix,
    *,
    mode: str = "svd",
    random_state: int = 42,
    max_dense_cells: int = 50_000_000,
) -> tuple[np.ndarray, np.ndarray, str]:
    """
    Returns (coords, components, method_name)
      - coords shape: (n_samples, 2)
      - components shape: (2, n_features) in feature space
    """
    mode = mode.lower().strip()
    if mode not in {"svd", "pca", "auto"}:
        raise ValueError("mode must be one of: svd, pca, auto")

    if sparse.issparse(X):
        n_cells = int(X.shape[0]) * int(X.shape[1])
        if mode == "pca" or (mode == "auto" and n_cells <= max_dense_cells):
            Xd = X.toarray().astype(np.float32, copy=False)
            pca = PCA(n_components=2, random_state=random_state)
            coords = pca.fit_transform(Xd)
            return coords, pca.components_, "PCA"

        svd = TruncatedSVD(n_components=2, random_state=random_state)
        coords = svd.fit_transform(X)
        return coords, svd.components_, "TruncatedSVD"

    pca = PCA(n_components=2, random_state=random_state)
    coords = pca.fit_transform(X)
    return coords, pca.components_, "PCA"


def components_to_long_df(components: np.ndarray, L: int, alphabet: str = "ACGT") -> pd.DataFrame:
    A = len(alphabet)
    if components.shape[1] != L * A:
        raise ValueError(f"Expected n_features={L*A}, got {components.shape[1]}")

    records: list[dict[str, object]] = []
    for pc_idx in range(components.shape[0]):
        w = components[pc_idx].reshape(L, A)
        for pos in range(L):
            for j, nuc in enumerate(alphabet):
                records.append(
                    {
                        "component": pc_idx + 1,
                        "position": pos + 1,  # 1-indexed for humans
                        "nucleotide": nuc,
                        "loading": float(w[pos, j]),
                        "abs_loading": float(abs(w[pos, j])),
                    }
                )
    return pd.DataFrame.from_records(records)


def plot_loading_heatmap(df_long: pd.DataFrame, outpath: Path, *, style: PlotStyle, component: int) -> None:
    set_plot_style(style)
    plt.figure(figsize=(style.rect_width_cm / 2.54, style.rect_height_cm / 2.54))
    sub = df_long[df_long["component"] == component]
    mat = sub.pivot(index="nucleotide", columns="position", values="loading")
    ax = sns.heatmap(mat, cmap="coolwarm", center=0.0)
    ax.set_title(f"Component {component} loadings\n(nucleotide × position)")
    ax.set_xlabel("Position")
    ax.set_ylabel("Nucleotide")
    savefig(outpath, dpi=style.dpi)


def plot_position_importance(df_long: pd.DataFrame, outpath: Path, *, style: PlotStyle, component: int) -> None:
    set_plot_style(style)
    plt.figure(figsize=(style.rect_width_cm / 2.54, style.rect_height_cm / 2.54))
    sub = df_long[df_long["component"] == component]
    pos_imp = sub.groupby("position", as_index=False)["abs_loading"].sum()
    ax = sns.lineplot(data=pos_imp, x="position", y="abs_loading", color="black", alpha=0.5, marker="o")
    ax.set_title(f"Component {component} position importance\n(sum |loading| across nucleotides)")
    ax.set_xlabel("Position")
    ax.set_ylabel("Sum |loading|")
    savefig(outpath, dpi=style.dpi)
    # Also write the table next to the plot
    out_csv = outpath.with_suffix(".csv")
    pos_imp.to_csv(out_csv, index=False)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--excel",
        default="data/251129_P8_120Complete_dataset_woCtrl_exogenous.xlsx",
        help="Input Excel file",
    )
    ap.add_argument("--sheet", default=0, help="Sheet name or index")
    ap.add_argument("--seq-col", default="SpacerSeq")
    ap.add_argument("--output-dir", default="output/interpretation", help="Output directory")
    ap.add_argument("--mode", choices=["svd", "pca", "auto"], default="svd")
    ap.add_argument("--random-state", type=int, default=42)
    ap.add_argument("--width-cm", type=float, default=12.0)
    ap.add_argument("--height-cm", type=float, default=12.0)
    ap.add_argument("--rect-width-cm", type=float, default=16.0)
    ap.add_argument("--rect-height-cm", type=float, default=12.0)
    args = ap.parse_args(argv)

    t0 = time.time()

    def log(msg: str) -> None:
        dt = time.time() - t0
        print(f"[{dt:8.1f}s] {msg}", flush=True)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    style = PlotStyle(
        width_cm=args.width_cm,
        height_cm=args.height_cm,
        rect_width_cm=args.rect_width_cm,
        rect_height_cm=args.rect_height_cm,
    )

    log(f"Loading Excel: {args.excel}")
    df = pd.read_excel(Path(args.excel), sheet_name=args.sheet, engine="openpyxl")
    if args.seq_col not in df.columns:
        raise ValueError(f"Missing sequence column: {args.seq_col}")

    log("One-hot encoding sequences")
    X, L = one_hot_encode_dna(df[args.seq_col])
    log(f"One-hot shape={X.shape} (L={L}) nnz={X.nnz}")

    log(f"Fitting 2D {args.mode.upper()} model to one-hot")
    coords, comps, method = fit_2d_linear(X, mode=args.mode, random_state=args.random_state)
    log(f"Fit done: method={method}")

    df_coords = pd.DataFrame({"comp1": coords[:, 0], "comp2": coords[:, 1]})
    df_coords.to_csv(out_dir / "components_scores.csv", index=False)

    df_long = components_to_long_df(comps, L)
    df_long.to_csv(out_dir / "components_loadings_long.csv", index=False)

    # Top features per component
    top = (
        df_long.sort_values(["component", "abs_loading"], ascending=[True, False])
        .groupby("component", as_index=False)
        .head(20)
        .reset_index(drop=True)
    )
    top.to_csv(out_dir / "top_loadings_per_component.csv", index=False)

    # Plots
    plot_loading_heatmap(df_long, out_dir / "component1_loadings_heatmap.png", style=style, component=1)
    plot_loading_heatmap(df_long, out_dir / "component2_loadings_heatmap.png", style=style, component=2)
    plot_position_importance(df_long, out_dir / "component1_position_importance.png", style=style, component=1)
    plot_position_importance(df_long, out_dir / "component2_position_importance.png", style=style, component=2)
    log("Wrote loadings tables + plots")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

