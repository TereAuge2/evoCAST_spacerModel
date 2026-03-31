"""
Generate regression_scatter_plot PNGs from predictions_vs_actual_*.csv (torch hyperopt retrain).

Writes per split (cv / test):
  regression_<split>.png        — scatter
  regression_<split>_plot2.png  — hexbin ("plot2" style in this project)

Examples (from repository root):
  python src/spacerModel/training/plot_hyperopt_predictions.py --run-dir output/modeling/CNN_hyperparameter_tune/<run>
  PYTHONPATH=src python -m spacerModel.training.plot_hyperopt_predictions --run-dir ...
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# .../src/spacerModel/training/this_file.py → add .../src so `spacerModel` imports work without PYTHONPATH.
_SRC = Path(__file__).resolve().parents[2]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import pandas as pd

from spacerModel.simple_modeling.train_models import regression_scatter_plot


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Regression plots from hyperopt predictions CSVs.")
    ap.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Folder containing predictions_vs_actual_cv.csv (and optional test CSV)",
    )
    ap.add_argument(
        "--title-prefix",
        default="torch_cnn best retrain",
        help="Title prefix for figure titles",
    )
    args = ap.parse_args(argv)

    run_dir = args.run_dir.resolve()
    if not run_dir.is_dir():
        raise SystemExit(f"Not a directory: {run_dir}")

    for _csv_name, split in [
        ("predictions_vs_actual_cv.csv", "cv"),
        ("predictions_vs_actual_test.csv", "test"),
    ]:
        p = run_dir / _csv_name
        if not p.exists():
            continue
        df = pd.read_csv(p)
        yt = df["y_true"].to_numpy(dtype=float)
        yp = df["y_pred"].to_numpy(dtype=float)
        prefix = str(args.title_prefix).strip()
        regression_scatter_plot(
            yt,
            yp,
            title=f"{prefix} — {split}",
            outpath=run_dir / f"regression_{split}.png",
            use_hexbin=False,
        )
        regression_scatter_plot(
            yt,
            yp,
            title=f"{prefix} — {split} (hexbin)",
            outpath=run_dir / f"regression_{split}_plot2.png",
            use_hexbin=True,
        )
        print(f"Wrote {run_dir / f'regression_{split}.png'}")
        print(f"Wrote {run_dir / f'regression_{split}_plot2.png'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
