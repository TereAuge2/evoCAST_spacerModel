#!/usr/bin/env python3
"""
Re-generate CV summary plots (mean ± std with per-fold dots) from existing
`cv_scores_*.csv` files. This does NOT re-run any models.

Plot layout comes from `train_models.plot_cv_summary` (e.g. 90° x labels, tick size 10).

With ``--median-baseline`` and ``--excel``, adds a dotted horizontal line (mean
validation metric when predicting ``median(y_train)`` per fold, matching GroupKFold
in ``train_models``) plus a legend label "Median baseline model".
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-base",
        type=str,
        default="output/modeling",
        help="Base directory that contains run subdirectories (e.g. 90%_variance_SVD).",
    )
    parser.add_argument(
        "--run-dirs",
        nargs="*",
        default=None,
        help="Optional list of run directories (names under --input-base or absolute paths).",
    )
    parser.add_argument(
        "--excel",
        type=Path,
        default=None,
        help="Dataset Excel (with Split + FE) for --median-baseline. Default: train_models default path.",
    )
    parser.add_argument("--sheet", default=0, help="Excel sheet (name or index) for --median-baseline")
    parser.add_argument("--target-col", default="FE", help="Target column for --median-baseline")
    parser.add_argument(
        "--median-baseline",
        action="store_true",
        help="Draw dotted median-of-training baseline (requires --excel or default dataset)",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(repo_root / "src"))

    import pandas as pd  # pylint: disable=import-error

    from spacerModel.simple_modeling.train_models import (  # pylint: disable=import-error
        median_baseline_cv_metric_means,
        plot_cv_summary,
        y_cv_and_groups_from_excel,
    )

    input_base = Path(args.input_base)
    if not input_base.exists():
        raise SystemExit(f"--input-base does not exist: {input_base}")

    if args.run_dirs:
        run_dirs: list[Path] = []
        for p in args.run_dirs:
            pth = Path(p)
            if not pth.is_absolute():
                pth = input_base / p
            run_dirs.append(pth)
    else:
        # Auto-discover run directories that contain cv_scores_*.csv
        run_dirs = [
            d
            for d in sorted(input_base.iterdir())
            if d.is_dir() and any(d.glob("cv_scores_*.csv"))
        ]

    if not run_dirs:
        raise SystemExit(f"No run directories found under {input_base}")

    default_excel = repo_root / "output" / "data_processing" / "251129_P8_120Complete_dataset_woCtrl_exogenous_with_splits.xlsx"
    excel_path = args.excel if args.excel is not None else default_excel

    for run_dir in run_dirs:
        csv_paths = sorted(run_dir.glob("cv_scores_*.csv"))
        if not csv_paths:
            continue

        for csv_path in csv_paths:
            df_scores = pd.read_csv(csv_path)
            if "split_col" not in df_scores.columns:
                print(f"Skip (no split_col): {csv_path}", flush=True)
                continue
            split_col = str(df_scores["split_col"].iloc[0])

            stem = csv_path.stem
            suffix = stem[len("cv_scores_") :] if stem.startswith("cv_scores_") else stem
            outpath = run_dir / f"cv_summary_{suffix}.png"

            median_baseline = None
            if args.median_baseline:
                if not excel_path.exists():
                    raise SystemExit(f"--median-baseline: Excel not found: {excel_path}")
                y_cv, groups = y_cv_and_groups_from_excel(
                    excel_path,
                    sheet=args.sheet,
                    split_col=split_col,
                    target_col=str(args.target_col),
                )
                median_baseline = median_baseline_cv_metric_means(y_cv, groups)

            plot_cv_summary(
                df_scores,
                outpath,
                title=f"{split_col} CV summary (RMSE)",
                median_baseline=median_baseline,
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

