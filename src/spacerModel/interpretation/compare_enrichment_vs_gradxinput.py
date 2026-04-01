"""
Compare a background-normalized enrichment matrix (top-FE vs library) to CNN mean grad×input.

The same enrichment vs grad×input scatter and per-position agreement are also produced by
`interpretation/explain_cnn_layers.py` under each `group_summaries/<group>/` (from frequencies
computed on the explainability mask). Use this script only when you want to pair external
CSV logos (e.g. from `analyze_dataset.py`) with a grad×input CSV by path.

Inputs:
  - top frequency CSV (position × A/C/G/T): e.g. logo_top500_by_FE_freq.csv
  - library frequency CSV (position × A/C/G/T): e.g. logo_full_library_freq.csv
  - grad×input mean CSV from explainability (position × A/C/G/T): e.g. group_summaries/top_fe/logo_grad_x_input_mean.csv

Outputs:
  - scatter plot of all (position,base) cells: enrichment bits vs grad×input
  - per-position agreement plot (Pearson r across bases at each position)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


BASES = ["A", "C", "G", "T"]


def _read_matrix(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Accept either explicit position column or index saved by pandas.
    if "position" in df.columns:
        df = df.set_index("position")
    # If first column is unnamed index, treat it as position.
    if df.columns[0].startswith("Unnamed"):
        df = df.set_index(df.columns[0])
    df.index = pd.to_numeric(df.index, errors="coerce")
    df = df.loc[df.index.notna()].copy()
    df.index = df.index.astype(int)
    # Ensure base columns
    df = df[[c for c in BASES if c in df.columns]].copy()
    return df


def _relative_bits(p: pd.DataFrame, q: pd.DataFrame, eps: float = 1e-9) -> pd.DataFrame:
    p2 = p.copy().astype(float)
    q2 = q.reindex(index=p2.index, columns=p2.columns).astype(float)
    p2 = np.clip(p2, eps, 1.0)
    q2 = np.clip(q2, eps, 1.0)
    w = p2 * (np.log2(p2) - np.log2(q2))
    w.index.name = "position"
    return w


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--top-freq", required=True, help="Top-set frequency CSV (position×A/C/G/T)")
    ap.add_argument("--library-freq", required=True, help="Library frequency CSV (position×A/C/G/T)")
    ap.add_argument("--gradxinput", required=True, help="Mean grad×input CSV (position×A/C/G/T)")
    ap.add_argument("--out-dir", required=True, help="Output directory")
    ap.add_argument("--title", default="Agreement: enrichment vs grad×input")
    args = ap.parse_args(argv)

    top = _read_matrix(Path(args.top_freq))
    lib = _read_matrix(Path(args.library_freq))
    gxi = _read_matrix(Path(args.gradxinput))

    # Align to common positions
    pos = sorted(set(top.index) & set(lib.index) & set(gxi.index))
    if not pos:
        raise SystemExit("No overlapping positions across inputs.")
    top = top.loc[pos, BASES]
    lib = lib.loc[pos, BASES]
    gxi = gxi.loc[pos, BASES]

    enr = _relative_bits(top, lib)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Scatter of all cells
    x = enr.to_numpy().reshape(-1)
    y = gxi.to_numpy().reshape(-1)
    ok = np.isfinite(x) & np.isfinite(y)
    x = x[ok]
    y = y[ok]

    fig, ax = plt.subplots(figsize=(16 / 2.54, 12 / 2.54))
    ax.scatter(x, y, s=18, alpha=0.6, color="black")
    ax.axhline(0.0, color="grey", linewidth=0.8)
    ax.axvline(0.0, color="grey", linewidth=0.8)
    if x.size >= 2:
        r = float(np.corrcoef(x, y)[0, 1])
        ax.set_title(f"{args.title}\nPearson r (all cells) = {r:.3f}")
    else:
        ax.set_title(args.title)
    ax.set_xlabel("enrichment bits vs library (p·log2(p/q))")
    ax.set_ylabel("mean grad×input")
    sns.despine(ax=ax)
    fig.tight_layout()
    fig.savefig(out_dir / "scatter_enrichment_vs_gradxinput.png", dpi=250, bbox_inches="tight")
    plt.close(fig)

    # Per-position Pearson r across bases (4 values each)
    rows = []
    for p in pos:
        xv = enr.loc[p, BASES].to_numpy(dtype=float)
        yv = gxi.loc[p, BASES].to_numpy(dtype=float)
        if np.allclose(np.std(xv), 0.0) or np.allclose(np.std(yv), 0.0):
            rr = float("nan")
        else:
            rr = float(np.corrcoef(xv, yv)[0, 1])
        rows.append({"position": int(p), "pearson_r_acgt": rr})
    dfp = pd.DataFrame(rows)
    dfp.to_csv(out_dir / "per_position_agreement.csv", index=False)

    fig, ax = plt.subplots(figsize=(18 / 2.54, 6 / 2.54))
    ax.plot(dfp["position"], dfp["pearson_r_acgt"], color="black", linewidth=1.5)
    ax.axhline(0.0, color="grey", linewidth=0.8)
    ax.set_xlabel("position")
    ax.set_ylabel("Pearson r across A/C/G/T")
    ax.set_title("Per-position agreement (enrichment vs grad×input)")
    sns.despine(ax=ax)
    fig.tight_layout()
    fig.savefig(out_dir / "per_position_agreement.png", dpi=250, bbox_inches="tight")
    plt.close(fig)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

