#!/usr/bin/env python3
"""
Re-generate CV summary plots (mean ± std with per-fold dots) from existing
`cv_scores_*.csv` files. This does NOT re-run any models.

Plot layout comes from `train_models.plot_cv_summary` (e.g. 90° x labels, tick size 10).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def parse_cv_scores_filename(csv_path: Path) -> tuple[str, str]:
    """
    Returns (split_col, out_tag) from a file like:
      - cv_scores_Split.csv -> ("Split", "")
      - cv_scores_SplitRandom.csv -> ("SplitRandom", "")
      - cv_scores_Split_raw_onehot.csv -> ("Split", "_raw_onehot")
    """
    stem = csv_path.stem
    prefix = "cv_scores_"
    if not stem.startswith(prefix):
        raise ValueError(f"Unexpected cv_scores filename: {csv_path}")

    remainder = stem[len(prefix) :]
    out_tag = ""
    raw_suffix = "_raw_onehot"
    if remainder.endswith(raw_suffix):
        out_tag = raw_suffix
        split_col = remainder[: -len(raw_suffix)]
    else:
        split_col = remainder
    return split_col, out_tag


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
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(repo_root / "src"))

    import pandas as pd  # pylint: disable=import-error

    from spacerModel.simple_modeling.train_models import plot_cv_summary  # pylint: disable=import-error

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

    for run_dir in run_dirs:
        csv_paths = sorted(run_dir.glob("cv_scores_*.csv"))
        if not csv_paths:
            continue

        for csv_path in csv_paths:
            split_col, out_tag = parse_cv_scores_filename(csv_path)
            df_scores = pd.read_csv(csv_path)

            outpath = run_dir / f"cv_summary_{split_col}{out_tag}.png"
            # plot_cv_summary writes RMSE to `outpath`; other metrics to *_METRIC.png (no duplicate RMSE).
            plot_cv_summary(df_scores, outpath, title=f"{split_col} CV summary (RMSE)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

