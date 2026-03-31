"""
Single CNN GroupKFold trial for W&B Sweeps (Bayesian / random / grid).

Run from the repository root after creating a sweep, e.g.::

    cd /path/to/2603_evoCAST_spacer_model
    export PYTHONPATH=src
    wandb sweep configs/wandb/cnn_bayes_sweep.yaml
    wandb agent entity/project/sweep_id

Each agent process pulls the next ``wandb.config`` from the controller, trains once,
and logs ``mean_rmse`` / ``mean_r2`` (CV means) for the sweep metric.

This module is separate from ``torch_hyperopt.py`` (in-loop random search).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[2]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import pandas as pd
import wandb  # type: ignore[import-untyped]

from spacerModel.simple_modeling.train_models import load_dataset
from spacerModel.training.search_param_rounding import round_dropout
from spacerModel.training.train_torch import (
    resolve_device,
    round_lr_wd,
    run_torch_cross_validation,
)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="One CNN CV trial for a W&B sweep agent.")
    ap.add_argument(
        "--excel",
        default="output/data_processing/251129_P8_120Complete_dataset_woCtrl_exogenous_with_splits.xlsx",
    )
    ap.add_argument("--sheet", default=0)
    ap.add_argument("--target-col", default="FE")
    ap.add_argument("--split-col", choices=["Split", "SplitRandom"], default="Split")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--random-state", type=int, default=42)
    args = ap.parse_args(argv)

    wandb.init()
    cfg = dict(wandb.config)

    def req(key: str) -> object:
        if key not in cfg:
            raise KeyError(f"wandb.config missing key {key!r} (check configs/wandb/cnn_bayes_sweep.yaml)")
        return cfg[key]

    lr = round_lr_wd(float(req("lr")))
    wd = round_lr_wd(float(req("weight_decay")))

    ns = argparse.Namespace(
        model_kind="cnn",
        embedding="onehot",
        kmer_k=2,
        raw_onehot=True,
        svd_components=82,
        svd_full_variance=False,
        split_col=args.split_col,
        target_col=args.target_col,
        sheet=args.sheet,
        random_state=args.random_state,
        use_hexbin=True,
        max_epochs=200,
        patience=10,
        min_delta=1e-4,
        lr=lr,
        weight_decay=wd,
        batch_size=int(req("batch_size")),
        dropout=round_dropout(float(req("dropout"))),
        mlp_hidden="256,128",
    )
    cnn_kw = {
        "conv1_channels": int(req("cnn_conv1_channels")),
        "conv2_channels": int(req("cnn_conv2_channels")),
        "head_hidden": int(req("cnn_head_hidden")),
        "kernel1": int(req("cnn_kernel1")),
        "kernel2": int(req("cnn_kernel2")),
        "pool_out": int(req("cnn_pool_out")),
    }

    device = resolve_device(args.device)
    ds = load_dataset(
        Path(args.excel),
        sheet=args.sheet,
        split_col=args.split_col,
        target_col=args.target_col,
    )
    if ds.df[args.split_col].astype(str).str.startswith("cv").sum() == 0:
        raise ValueError("No CV rows found (expected cv* splits).")

    out_dir = Path("output/modeling/wandb_sweep_work") / (wandb.run.id or "local")
    out_dir.mkdir(parents=True, exist_ok=True)

    def log(msg: str) -> None:
        print(msg, flush=True)

    df_scores: pd.DataFrame = run_torch_cross_validation(
        ns,
        ds,
        device=device,
        mlp_hidden=(256, 128),
        out_dir=out_dir,
        make_plots=False,
        log=log,
        cnn_kwargs=cnn_kw,
        collect_learning_curves=False,
        trial_idx=None,
    )
    mean_rmse = float(df_scores["rmse"].mean())
    mean_r2 = float(df_scores["r2"].mean())
    wandb.log({"mean_rmse": mean_rmse, "mean_r2": mean_r2})
    wandb.summary["mean_rmse"] = mean_rmse
    wandb.summary["mean_r2"] = mean_r2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
