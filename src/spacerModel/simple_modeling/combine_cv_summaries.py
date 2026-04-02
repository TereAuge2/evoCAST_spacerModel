#!/usr/bin/env python3
"""
Merge CV score tables from two modeling runs and draw a single `plot_cv_summary`
figure set (RMSE base PNG + R², Pearson, Spearman, EF10% suffixed PNGs).

Typical use: combine the main hyperopt-era CV run with a CNN-only one-hot rerun,
renaming the second run's `cnn1d` rows to `cnn_onehot` so they don't collide
with the first run's `cnn1d`.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path


def _default_repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-run-dir",
        type=Path,
        default=None,
        help="Primary run directory (e.g. 1_Best_initial_...). Receives cv_summary_combined_*.png",
    )
    parser.add_argument(
        "--extra-run-dir",
        type=Path,
        default=None,
        help="Second run directory (e.g. 2_Split__onehot_raw_CNN_only).",
    )
    parser.add_argument(
        "--base-cv-scores",
        type=Path,
        default=None,
        help="Explicit path to base cv_scores_*.csv (overrides glob in --base-run-dir).",
    )
    parser.add_argument(
        "--extra-cv-scores",
        type=Path,
        default=None,
        help="Explicit path to extra cv_scores_*.csv (overrides glob in --extra-run-dir).",
    )
    parser.add_argument(
        "--rename-model",
        nargs=2,
        metavar=("FROM", "TO"),
        default=("cnn1d", "cnn_onehot"),
        help="Rename model label in the extra run only (default: cnn1d cnn_onehot).",
    )
    parser.add_argument(
        "--modeling-root",
        type=Path,
        default=None,
        help="Directory for timestamped unique outputs (default: <repo>/output/modeling).",
    )
    parser.add_argument(
        "--combined-stem",
        type=str,
        default="k2_manual_plus_cnn_onehot",
        help="Short tag used in filenames (cv_summary_combined_<tag>.png in base dir).",
    )
    parser.add_argument(
        "--title",
        type=str,
        default=None,
        help="Plot title prefix (default: derived from split_col and --combined-stem).",
    )
    args = parser.parse_args()

    repo = _default_repo_root()
    sys.path.insert(0, str(repo / "src"))

    import pandas as pd  # pylint: disable=import-error

    from spacerModel.simple_modeling.train_models import plot_cv_summary  # pylint: disable=import-error

    modeling_root = args.modeling_root or (repo / "output" / "modeling")
    modeling_root.mkdir(parents=True, exist_ok=True)

    base_run = args.base_run_dir or (
        modeling_root / "1_Best_initial_CV_pre_hyperopt_Split__kmer_gc_onehot_raw__k2_manual"
    )
    extra_run = args.extra_run_dir or (modeling_root / "2_Split__onehot_raw_CNN_only")

    base_csv = args.base_cv_scores
    if base_csv is None:
        cands = sorted(base_run.glob("cv_scores_*.csv"))
        if len(cands) != 1:
            raise SystemExit(f"Expected exactly one cv_scores_*.csv in {base_run}, got {len(cands)}")
        base_csv = cands[0]

    extra_csv = args.extra_cv_scores
    if extra_csv is None:
        cands = sorted(extra_run.glob("cv_scores_*.csv"))
        if len(cands) != 1:
            raise SystemExit(f"Expected exactly one cv_scores_*.csv in {extra_run}, got {len(cands)}")
        extra_csv = cands[0]

    rename_from, rename_to = args.rename_model

    df_base = pd.read_csv(base_csv)
    df_extra = pd.read_csv(extra_csv)

    if "split_col" not in df_base.columns or "model" not in df_base.columns:
        raise SystemExit("base scores CSV must contain split_col and model columns")
    splits_base = set(df_base["split_col"].unique())
    splits_extra = set(df_extra["split_col"].unique())
    if splits_base != splits_extra:
        raise SystemExit(f"split_col mismatch: base {splits_base} vs extra {splits_extra}")

    mask = df_extra["model"] == rename_from
    if not bool(mask.any()):
        raise SystemExit(f"No rows with model={rename_from!r} in {extra_csv}")
    df_extra = df_extra.copy()
    df_extra.loc[mask, "model"] = rename_to

    overlap = set(df_base["model"].unique()) & set(df_extra["model"].unique())
    if overlap:
        raise SystemExit(
            f"After rename, model names still overlap between runs: {sorted(overlap)}. "
            "Adjust --rename-model or filter inputs."
        )

    df_merged = pd.concat([df_base, df_extra], ignore_index=True)

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    stem_tag = args.combined_stem

    merged_csv_base = base_run / f"cv_scores_combined_{stem_tag}.csv"
    merged_csv_unique = modeling_root / f"cv_scores_combined_{ts}_{stem_tag}.csv"
    df_merged.to_csv(merged_csv_base, index=False)
    df_merged.to_csv(merged_csv_unique, index=False)

    split_col = str(df_merged["split_col"].iloc[0])
    title = args.title or f"{split_col} CV summary — combined ({stem_tag})"

    out_base = base_run / f"cv_summary_combined_{stem_tag}.png"
    out_unique = modeling_root / f"cv_summary_combined_{ts}_{stem_tag}.png"

    plot_cv_summary(df_merged, out_base, title=f"{title} (RMSE)")
    plot_cv_summary(df_merged, out_unique, title=f"{title} (RMSE)")

    print(f"Wrote merged scores: {merged_csv_base}")
    print(f"Wrote merged scores (unique): {merged_csv_unique}")
    print(f"Wrote combined plots under: {base_run} (cv_summary_combined_{stem_tag}.*)")
    print(f"Wrote combined plots (unique): {out_unique.stem}.* in {modeling_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
