#!/usr/bin/env python3
"""
Replot surrogate SHAP/contribution summaries from already-written CSVs.

This script does NOT recompute SHAP/pred_contribs. It reads:
  - xgb_surrogate_<stem>_shap_mean_signed_by_position_base.csv
  - xgb_surrogate_<stem>_shap_mean_abs_by_position_base.csv

and rewrites a clean set of plots so you can compare signed vs magnitude views
without rerunning expensive attribution.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from spacerModel.interpretation.linear_position_weights import (
    plot_abs_weight_sum,
    plot_weights_by_position,
    savefig,
    set_plot_style,
)


def _read_pos_base_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    need = {"position", "A", "C", "G", "T"}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"{path.name}: missing columns {sorted(missing)}")
    out = df[["position", "A", "C", "G", "T"]].copy()
    out["position"] = pd.to_numeric(out["position"], errors="raise").astype(int)
    return out


def _plot_logo(df_pos: pd.DataFrame, outpath: Path, *, title: str) -> None:
    import logomaker
    import matplotlib.pyplot as plt
    import numpy as np

    set_plot_style()
    df = df_pos[["A", "C", "G", "T"]].copy().astype(float).reset_index(drop=True)
    npos = int(len(df))
    plt.figure(figsize=(18.0 / 2.54, 5.5 / 2.54))
    ax = plt.gca()
    logomaker.Logo(df, ax=ax, color_scheme="classic")
    ax.set_title(title)
    ax.set_ylabel("Contribution")
    if npos > 0:
        ax.set_xticks(np.arange(npos, dtype=float))
        ax.set_xticklabels([str(i + 1) for i in range(npos)], fontsize=7)
        ax.set_xlim(-0.5, npos - 0.5)
    ax.set_xlabel("Position")
    savefig(outpath)


def _replot_one_stem(out_dir: Path, stem: str, *, include_legacy_all_row: bool = False) -> None:
    # Active-only outputs (primary/recommended).
    signed_active_csv = out_dir / f"xgb_surrogate_{stem}_shap_mean_signed_active_only_by_position_base.csv"
    abs_active_csv = out_dir / f"xgb_surrogate_{stem}_shap_mean_abs_active_only_by_position_base.csv"
    if signed_active_csv.exists() and abs_active_csv.exists():
        df_signed_active = _read_pos_base_csv(signed_active_csv)
        df_abs_active = _read_pos_base_csv(abs_active_csv)
        plot_weights_by_position(
            df_signed_active,
            out_dir / f"xgb_surrogate_{stem}_shap_replot_mean_signed_active_only_by_position.png",
            title=f"XGBoost surrogate mean contribution (signed, active-only) — {stem.replace('_', ' ')}",
        )
        _plot_logo(
            df_signed_active,
            out_dir / f"xgb_surrogate_{stem}_shap_replot_mean_signed_active_only_logo.png",
            title=f"XGBoost surrogate mean contribution (signed, active-only) logo — {stem.replace('_', ' ')}",
        )
        plot_weights_by_position(
            df_abs_active,
            out_dir / f"xgb_surrogate_{stem}_shap_replot_mean_abs_active_only_by_position.png",
            title=f"XGBoost surrogate mean |contribution| (active-only) by position/base — {stem.replace('_', ' ')}",
        )
        _plot_logo(
            df_abs_active,
            out_dir / f"xgb_surrogate_{stem}_shap_replot_mean_abs_active_only_logo.png",
            title=f"XGBoost surrogate mean |contribution| (active-only) logo — {stem.replace('_', ' ')}",
        )
        plot_abs_weight_sum(
            df_abs_active,
            out_dir / f"xgb_surrogate_{stem}_shap_replot_sum_mean_abs_active_only_by_position.png",
            title=f"XGBoost surrogate Σ mean |contribution| (active-only) per position — {stem.replace('_', ' ')}",
        )
    else:
        raise FileNotFoundError(
            f"Missing active-only SHAP CSVs for stem={stem!r} in {out_dir}.\n"
            f"Expected:\n  {signed_active_csv.name}\n  {abs_active_csv.name}"
        )

    if not include_legacy_all_row:
        return

    # Optional legacy all-row outputs.
    signed_csv = out_dir / f"xgb_surrogate_{stem}_shap_mean_signed_by_position_base.csv"
    abs_csv = out_dir / f"xgb_surrogate_{stem}_shap_mean_abs_by_position_base.csv"
    if not signed_csv.exists() or not abs_csv.exists():
        return
    df_signed = _read_pos_base_csv(signed_csv)
    df_abs = _read_pos_base_csv(abs_csv)
    plot_weights_by_position(
        df_signed,
        out_dir / f"xgb_surrogate_{stem}_shap_replot_mean_signed_by_position.png",
        title=f"XGBoost surrogate mean contribution (all-row signed) — {stem.replace('_', ' ')}",
    )
    _plot_logo(
        df_signed,
        out_dir / f"xgb_surrogate_{stem}_shap_replot_mean_signed_logo.png",
        title=f"XGBoost surrogate mean contribution (all-row signed) logo — {stem.replace('_', ' ')}",
    )
    plot_abs_weight_sum(
        df_signed,
        out_dir / f"xgb_surrogate_{stem}_shap_replot_abs_of_mean_signed_by_position.png",
        title=f"XGBoost surrogate Σ|mean signed contribution| (all-row) — {stem.replace('_', ' ')}",
    )
    plot_weights_by_position(
        df_abs,
        out_dir / f"xgb_surrogate_{stem}_shap_replot_mean_abs_by_position.png",
        title=f"XGBoost surrogate mean |contribution| (all-row) — {stem.replace('_', ' ')}",
    )
    _plot_logo(
        df_abs,
        out_dir / f"xgb_surrogate_{stem}_shap_replot_mean_abs_logo.png",
        title=f"XGBoost surrogate mean |contribution| (all-row) logo — {stem.replace('_', ' ')}",
    )
    plot_abs_weight_sum(
        df_abs,
        out_dir / f"xgb_surrogate_{stem}_shap_replot_sum_mean_abs_by_position.png",
        title=f"XGBoost surrogate Σ mean |contribution| (all-row) — {stem.replace('_', ' ')}",
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/interpretation/cnn_surrogate_xgboost"),
        help="Directory containing xgb_surrogate_*_shap_mean_*_by_position_base.csv",
    )
    ap.add_argument(
        "--stems",
        type=str,
        default="synthetic,real_cv",
        help="Comma-separated stems to replot (default: synthetic,real_cv).",
    )
    ap.add_argument(
        "--include-legacy-all-row",
        action="store_true",
        help="Also regenerate legacy all-row SHAP plots when corresponding CSVs exist.",
    )
    args = ap.parse_args(argv)

    out_dir = Path(args.output_dir)
    if not out_dir.exists():
        raise SystemExit(f"--output-dir does not exist: {out_dir}")

    stems = [s.strip() for s in str(args.stems).split(",") if s.strip()]
    if not stems:
        raise SystemExit("No stems provided")

    for stem in stems:
        _replot_one_stem(out_dir, stem, include_legacy_all_row=bool(args.include_legacy_all_row))
        print(f"[replot-shap] wrote corrected plots for stem={stem!r} in {out_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

