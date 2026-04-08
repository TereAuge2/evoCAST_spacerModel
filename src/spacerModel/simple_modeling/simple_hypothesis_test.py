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
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import r2_score


@dataclass(frozen=True)
class PlotStyle:
    width_cm: float = 18.0
    height_cm: float = 12.0
    use_hexbin: bool = True


def compute_corr_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.size < 2:
        return {"pearson_r": float("nan"), "spearman_rho": float("nan")}
    try:
        pr = float(pearsonr(y_true, y_pred).statistic)
    except Exception:
        pr = float("nan")
    try:
        sr = float(spearmanr(y_true, y_pred).statistic)
    except Exception:
        sr = float("nan")
    return {"pearson_r": pr, "spearman_rho": sr}


def regression_like_plot(
    x_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    title: str,
    xlab: str,
    ylab: str,
    outpath: Path,
    style: PlotStyle,
) -> None:
    mpl.rcParams.update(mpl.rcParamsDefault)
    mpl.rcParams["font.family"] = "sans-serif"
    mpl.rcParams["font.size"] = 14

    plot_width_cm = (style.width_cm + 8.0) if style.use_hexbin else style.width_cm
    fig, ax = plt.subplots(figsize=(plot_width_cm / 2.54, style.height_cm / 2.54))

    x_true = np.asarray(x_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    if style.use_hexbin:
        import matplotlib.colors as mcolors

        _hex_cmap = mcolors.LinearSegmentedColormap.from_list("grey_to_blue", ["#d3d3d3", "#0d1b2a"], N=256)
        hb = ax.hexbin(
            x_true,
            y_pred,
            gridsize=50,
            mincnt=1,
            cmap=_hex_cmap,
            edgecolors="none",
            alpha=0.85,
            rasterized=True,
        )
        fig.subplots_adjust(right=0.52)
        cbar = plt.colorbar(hb, ax=ax, pad=0.02, shrink=0.85, label="count")
        cbar.ax.tick_params(axis="y", labelsize=11)
    else:
        sns.scatterplot(
            x=x_true,
            y=y_pred,
            edgecolors="black",
            color="black",
            alpha=0.3,
            s=15,
            linewidth=0.3,
            rasterized=True,
            ax=ax,
        )

    # Reference lines (consistent with model plots)
    ax.axvline(0, color="grey", linestyle=":", linewidth=1)

    m = compute_corr_metrics(x_true, y_pred)
    metrics_text = (
        r"$\mathbf{Metrics}$"
        + "\n\n"
        + f"Pearson R: {m['pearson_r']:.2f}\n"
        + f"Spearman ρ: {m['spearman_rho']:.2f}"
    )
    if style.use_hexbin:
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
    ax.set_title(title.replace(" vs ", " vs\n"), fontsize=14)
    ax.tick_params(axis="both", which="both", direction="out", length=5, width=1, labelsize=11)
    sns.despine(ax=ax)
    ax.grid(False)
    if not style.use_hexbin:
        fig.tight_layout()

    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)


def gc_fraction(seq: str) -> float:
    if not seq:
        return float("nan")
    s = seq.upper()
    valid = [c for c in s if c in {"A", "C", "G", "T"}]
    if not valid:
        return float("nan")
    gc = sum(1 for c in valid if c in {"G", "C"})
    return gc / len(valid)


def c_fraction(seq: str) -> float:
    if not seq:
        return float("nan")
    s = seq.upper()
    valid = [c for c in s if c in {"A", "C", "G", "T"}]
    if not valid:
        return float("nan")
    c_count = sum(1 for c in valid if c == "C")
    return c_count / len(valid)


def tm_wallace(seq: str) -> float:
    """
    Wallace rule melting temperature estimate (good for short oligos).
    Tm ≈ 2*(A+T) + 4*(G+C)
    """
    if not seq:
        return float("nan")
    s = seq.upper()
    a = s.count("A")
    t = s.count("T")
    g = s.count("G")
    c = s.count("C")
    n = a + t + g + c
    if n == 0:
        return float("nan")
    return 2.0 * (a + t) + 4.0 * (g + c)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--excel",
        default="data/251129_P8_120Complete_dataset_woCtrl_exogenous.xlsx",
        help="Input Excel path",
    )
    ap.add_argument("--sheet", default=0, help="Sheet name or index")
    ap.add_argument("--seq-col", default="SpacerSeq")
    ap.add_argument("--target-col", default="FE")
    ap.add_argument("--output-dir", default="output/modeling/hypothesis_tests")
    ap.add_argument("--width-cm", type=float, default=18.0)
    ap.add_argument("--height-cm", type=float, default=12.0)
    ap.add_argument("--use-hexbin", action="store_true", default=True)
    ap.add_argument("--random-state", type=int, default=42)  # kept for parity / future use
    args = ap.parse_args(argv)

    t0 = time.time()

    def log(msg: str) -> None:
        dt = time.time() - t0
        print(f"[{dt:8.1f}s] {msg}", flush=True)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    log(f"Loading Excel: {args.excel}")
    df = pd.read_excel(Path(args.excel), sheet_name=args.sheet, engine="openpyxl")
    needed = {args.seq_col, args.target_col}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df = df.copy()
    df[args.target_col] = pd.to_numeric(df[args.target_col], errors="coerce")
    df = df[df[args.target_col].notna()].reset_index(drop=True)
    log(f"Rows after dropping NaN target: {len(df)}")

    seqs = df[args.seq_col].astype(str).fillna("")
    df["GC_frac"] = seqs.map(gc_fraction)
    df["C_frac"] = seqs.map(c_fraction)
    df["Tm_wallace"] = seqs.map(tm_wallace)

    df.to_csv(out_dir / "hypothesis_features.csv", index=False)
    log("Wrote hypothesis_features.csv")

    style = PlotStyle(width_cm=float(args.width_cm), height_cm=float(args.height_cm), use_hexbin=bool(args.use_hexbin))

    y = df[args.target_col].to_numpy(dtype=float)

    # GC% vs FE
    mask_gc = df["GC_frac"].notna()
    regression_like_plot(
        y[mask_gc],
        (df.loc[mask_gc, "GC_frac"] * 100.0).to_numpy(dtype=float),
        title="Measured log2-FE vs Predicted GC%",
        xlab="log2-FE (measured)",
        ylab="GC%",
        outpath=out_dir / "fe_vs_gc_percent.png",
        style=style,
    )
    log("Saved fe_vs_gc_percent.png")

    # C% vs FE
    mask_c = df["C_frac"].notna()
    regression_like_plot(
        y[mask_c],
        (df.loc[mask_c, "C_frac"] * 100.0).to_numpy(dtype=float),
        title="Measured log2-FE vs Predicted C%",
        xlab="log2-FE (measured)",
        ylab="C%",
        outpath=out_dir / "fe_vs_c_percent.png",
        style=style,
    )
    log("Saved fe_vs_c_percent.png")

    # Tm vs FE
    mask_tm = df["Tm_wallace"].notna()
    regression_like_plot(
        y[mask_tm],
        df.loc[mask_tm, "Tm_wallace"].to_numpy(dtype=float),
        title="Measured log2-FE vs Predicted Tm (Wallace)",
        xlab="log2-FE (measured)",
        ylab="Tm (Wallace, °C)",
        outpath=out_dir / "fe_vs_tm_wallace.png",
        style=style,
    )
    log("Saved fe_vs_tm_wallace.png")

    # Correlation table
    corr_rows = []
    m_gc = compute_corr_metrics(y[mask_gc], (df.loc[mask_gc, "GC_frac"] * 100.0).to_numpy(dtype=float))
    corr_rows.append({"feature": "GC_percent", **m_gc})
    m_c = compute_corr_metrics(y[mask_c], (df.loc[mask_c, "C_frac"] * 100.0).to_numpy(dtype=float))
    corr_rows.append({"feature": "C_percent", **m_c})
    m_tm = compute_corr_metrics(y[mask_tm], df.loc[mask_tm, "Tm_wallace"].to_numpy(dtype=float))
    corr_rows.append({"feature": "Tm_wallace", **m_tm})
    pd.DataFrame(corr_rows).to_csv(out_dir / "hypothesis_correlations.csv", index=False)
    log("Wrote hypothesis_correlations.csv")

    log("Done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

