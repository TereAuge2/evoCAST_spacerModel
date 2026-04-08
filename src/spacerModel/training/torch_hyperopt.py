"""
Randomized hyperparameter search for PyTorch MLP/CNN (RandomizedSearchCV-style:
``n_iter`` independent draws). Each trial runs full GroupKFold CV with per-epoch
validation RMSE/R² learning curves (original target scale).

Writes:
  - torch_random_search_<split>_<kind>.csv
  - hyperparam_sweep/dataframes/*.csv, hyperparam_sweep/plots/*.png
  - learning_curves/all_trials_long.csv, by_trial/*.csv, plots/*.png
  - best_hyperparameters.json (lowest mean_rmse)

Optional full-CV retrain: ``--best-retrain`` (default: off).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from spacerModel.training.hyperopt_diagnostics import (
    plot_learning_curve_epochs,
    plot_learning_curve_r2_epochs,
    run_torch_hyperparam_overview,
)
from spacerModel.utils.run_paths import default_hyperopt_output_dir, default_run_name, slurm_job_id_prefix
from spacerModel.training.search_param_rounding import round_dropout
from spacerModel.training.train_torch import (
    apply_torch_hyperopt_row,
    retrain_best_torch_and_save_artifacts,
    resolve_device,
    round_lr_wd,
    run_torch_cross_validation,
)
from spacerModel.utils.wandb_util import finish_run, init_wandb_run, log_plot_images
from spacerModel.simple_modeling.train_models import load_dataset

_CNN_KEYS_TO_ARG: dict[str, str] = {
    "cnn_conv1_channels": "conv1_channels",
    "cnn_conv2_channels": "conv2_channels",
    "cnn_kernel1": "kernel1",
    "cnn_kernel2": "kernel2",
    "cnn_pool_out": "pool_out",
    "cnn_head_hidden": "head_hidden",
}

# Match tuned CNN defaults (see train_torch._TORCH_HYPEROPT_DEFAULTS / best_hyperparameters.json).
_TORCH_TRAIN_DEFAULTS: dict[str, Any] = {
    "lr": round_lr_wd(0.00013),
    "weight_decay": round_lr_wd(0.00091),
    "batch_size": 32,
    "dropout": 0.3,
    "max_epochs": 200,
    "patience": 10,
    "min_delta": 1e-4,
}

_MLP_HIDDEN_CHOICES = [
    "256,128",
    "512,256,128",
    "512,256,128,64",
    "384,256,128,64",
    "384,192,96",
    "768,512,256,128",
]


def _parse_mlp_hidden(s: str) -> tuple[int, ...]:
    return tuple(int(x.strip()) for x in s.split(",") if x.strip())


def sample_random_mlp(rng: np.random.Generator) -> dict[str, Any]:
    return {
        "lr": round_lr_wd(float(10 ** rng.uniform(-5.0, -2.0))),
        "weight_decay": round_lr_wd(float(10 ** rng.uniform(-6.0, -3.0))),
        "batch_size": int(rng.choice([32, 64, 128])),
        "dropout": round_dropout(float(rng.choice([0.1, 0.2, 0.3]))),
        "mlp_hidden": str(rng.choice(_MLP_HIDDEN_CHOICES)),
    }


def sample_random_cnn(rng: np.random.Generator) -> dict[str, Any]:
    return {
        "lr": round_lr_wd(float(10 ** rng.uniform(-5.0, -2.0))),
        "weight_decay": round_lr_wd(float(10 ** rng.uniform(-6.0, -3.0))),
        "batch_size": int(rng.choice([32, 64, 128])),
        "dropout": round_dropout(float(rng.choice([0.1, 0.2, 0.3]))),
        "cnn_conv1_channels": int(rng.integers(32, 129)),
        "cnn_conv2_channels": int(rng.integers(32, 129)),
        "cnn_head_hidden": int(rng.choice([128, 256, 384, 512])),
        "cnn_kernel1": int(rng.choice([3, 5, 7])),
        "cnn_kernel2": int(rng.choice([3, 5, 7])),
        "cnn_pool_out": int(rng.choice([4, 8, 16])),
    }


def default_baseline_row_mlp() -> dict[str, Any]:
    """Match ``_TORCH_TRAIN_DEFAULTS`` and a typical MLP hidden stack."""
    return {
        "lr": round_lr_wd(float(_TORCH_TRAIN_DEFAULTS["lr"])),
        "weight_decay": round_lr_wd(float(_TORCH_TRAIN_DEFAULTS["weight_decay"])),
        "batch_size": int(_TORCH_TRAIN_DEFAULTS["batch_size"]),
        "dropout": round_dropout(float(_TORCH_TRAIN_DEFAULTS["dropout"])),
        "mlp_hidden": "256,128",
    }


def default_baseline_row_cnn() -> dict[str, Any]:
    """Match ``_TORCH_TRAIN_DEFAULTS`` and tuned CNN architecture (train_torch CLI defaults)."""
    return {
        "lr": round_lr_wd(float(_TORCH_TRAIN_DEFAULTS["lr"])),
        "weight_decay": round_lr_wd(float(_TORCH_TRAIN_DEFAULTS["weight_decay"])),
        "batch_size": int(_TORCH_TRAIN_DEFAULTS["batch_size"]),
        "dropout": round_dropout(float(_TORCH_TRAIN_DEFAULTS["dropout"])),
        "cnn_conv1_channels": 114,
        "cnn_conv2_channels": 124,
        "cnn_head_hidden": 256,
        "cnn_kernel1": 3,
        "cnn_kernel2": 3,
        "cnn_pool_out": 16,
    }


def _wandb_log_learning_curve_scalars(wandb_run: Any, df_curves: pd.DataFrame, trial_idx: int) -> None:
    """
    Log epoch-wise metrics for Charts. Uses ``lc_epoch`` as custom x-axis (via ``define_metric``),
    not ``wandb.log(..., step=)``, so trial 2+ epochs are not dropped after trial 1 reaches ~200 steps.
    """
    if df_curves.empty:
        return

    mean_df = (
        df_curves.groupby("epoch", as_index=False)
        .agg({"train_loss": "mean", "val_loss": "mean", "val_rmse": "mean", "val_r2": "mean"})
        .sort_values("epoch")
    )
    prefix = f"curves/trial_{trial_idx}/mean"
    for _, row in mean_df.iterrows():
        ep = int(row["epoch"])
        wandb_run.log(
            {
                "lc_epoch": ep,
                f"{prefix}/train_loss": float(row["train_loss"]),
                f"{prefix}/val_loss": float(row["val_loss"]),
                f"{prefix}/val_rmse": float(row["val_rmse"]),
                f"{prefix}/val_r2": float(row["val_r2"]),
            }
        )


def cnn_kwargs_from_sample_row(row: dict[str, Any]) -> dict[str, Any] | None:
    out: dict[str, Any] = {}
    for sk, dk in _CNN_KEYS_TO_ARG.items():
        if sk in row and row[sk] is not None:
            out[dk] = int(row[sk])
    return out if out else None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Randomized search over torch MLP/CNN (n_iter trials × full GroupKFold CV each).",
    )
    ap.add_argument("--excel", default="output/data_processing/251129_P8_120Complete_dataset_woCtrl_exogenous_with_splits.xlsx")
    ap.add_argument("--sheet", default=0)
    ap.add_argument("--target-col", default="FE")
    ap.add_argument("--split-col", choices=["Split", "SplitRandom"], default="Split")
    ap.add_argument("--random-state", type=int, default=42)
    ap.add_argument("--svd-components", type=int, default=82)
    ap.add_argument("--svd-full-variance", action="store_true")
    ap.add_argument(
        "--raw-onehot",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Dense features (default on). Same semantics as train_torch / train_models.",
    )
    ap.add_argument(
        "--embedding",
        choices=["onehot", "onehot_gc", "kmer", "kmer_gc", "kmer_gc_onehot"],
        default="kmer_gc_onehot",
        help="Default: raw one-hot + GC + k-mer freqs (k from --kmer-k).",
    )
    ap.add_argument("--kmer-k", type=int, default=2, choices=[2, 3, 4])
    ap.add_argument(
        "--output-dir",
        default="",
        help="Default: output/modeling/CNN_hyperparameter_tune or MLP_hyperparameter_tune",
    )
    ap.add_argument(
        "--run-name",
        default="",
        help="Subfolder; default: <SLURM_JOB_ID>_<timestamp>",
    )
    ap.add_argument("--model-kind", choices=["mlp", "cnn"], required=True)
    ap.add_argument(
        "--n-iter",
        type=int,
        default=40,
        help="Number of random trials (like RandomizedSearchCV n_iter).",
    )
    ap.add_argument("--device", default="auto")
    ap.add_argument("--use-hexbin", action="store_true", default=True)
    ap.add_argument("--use-wandb", action="store_true", help="Log to Weights & Biases")
    ap.add_argument("--wandb-entity", default="", help="Default: WANDB_ENTITY or teresa_augustin-harvard-university")
    ap.add_argument("--wandb-project", default="", help="Default: WANDB_PROJECT or evoCAST_spacer")
    ap.add_argument(
        "--best-retrain",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="After search, retrain best config on full CV and save best_model.pt (default: false).",
    )
    args_base = ap.parse_args(argv)

    if args_base.model_kind == "cnn" and args_base.embedding != "onehot":
        raise SystemExit(
            "CNN hyperopt requires --embedding onehot. "
            f"Got embedding={args_base.embedding!r}. Use --model-kind mlp for kmer_gc_onehot."
        )

    try:
        sys.stdout.reconfigure(line_buffering=True)  # type: ignore[attr-defined]
    except Exception:
        pass

    t0 = time.time()

    def log(msg: str) -> None:
        print(f"[{time.time() - t0:8.1f}s] {msg}", flush=True)

    device = resolve_device(args_base.device)
    log(f"device={device}")

    modeling_root = Path("output/modeling")
    if str(args_base.output_dir).strip():
        base_out = Path(args_base.output_dir)
    else:
        base_out = default_hyperopt_output_dir(modeling_root, "cnn" if args_base.model_kind == "cnn" else "mlp")

    run_name = str(args_base.run_name).strip() or default_run_name(args_base.model_kind)
    out_dir = base_out / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    log(f"Output directory: {out_dir}")

    n_iter = max(1, int(args_base.n_iter))
    rng = np.random.default_rng(int(args_base.random_state))
    log(
        f"Random search: n_iter={n_iter} (model_kind={args_base.model_kind}); "
        "trial 1 = default baseline, trials 2..n_iter = random"
    )

    ds = load_dataset(
        Path(args_base.excel),
        sheet=args_base.sheet,
        split_col=args_base.split_col,
        target_col=args_base.target_col,
    )
    if ds.df[args_base.split_col].astype(str).str.startswith("cv").sum() == 0:
        raise ValueError("No CV rows found (expected cv* splits).")

    job_id = slurm_job_id_prefix()
    wandb_run = init_wandb_run(
        use_wandb=bool(args_base.use_wandb),
        name=run_name,
        job_id=job_id,
        tags=[f"tune_{args_base.model_kind}", "torch_random_search"],
        config={
            "model_kind": args_base.model_kind,
            "split_col": args_base.split_col,
            "embedding": args_base.embedding,
            "excel": str(args_base.excel),
            "n_iter": n_iter,
        },
        entity=str(args_base.wandb_entity).strip() or None,
        project=str(args_base.wandb_project).strip() or None,
    )
    if wandb_run is not None:
        import wandb  # type: ignore[import-untyped]

        try:
            # Hyperparameter search summary: x-axis = trial index (restarts OK; separate from global step).
            wandb.define_metric("hyperopt_trial")
            wandb.define_metric("trial_idx", step_metric="hyperopt_trial")
            wandb.define_metric("mean_rmse", step_metric="hyperopt_trial")
            wandb.define_metric("mean_r2", step_metric="hyperopt_trial")
            wandb.define_metric("mean_pearson_r", step_metric="hyperopt_trial")
            # Learning curves: x-axis = epoch within each trial (must not use wandb.log(step=epoch)
            # or trial 2+ is rejected after global step reached max_epochs).
            wandb.define_metric("lc_epoch")
            wandb.define_metric("curves/*", step_metric="lc_epoch")
        except Exception:
            pass

    results: list[dict[str, Any]] = []
    all_curves: list[pd.DataFrame] = []
    (out_dir / "learning_curves" / "plots").mkdir(parents=True, exist_ok=True)
    (out_dir / "learning_curves" / "by_trial").mkdir(parents=True, exist_ok=True)

    try:
        for ti in range(1, n_iter + 1):
            if ti == 1:
                row = default_baseline_row_mlp() if args_base.model_kind == "mlp" else default_baseline_row_cnn()
                log(f"Trial {ti}/{n_iter} (default baseline) {row}")
            else:
                if args_base.model_kind == "mlp":
                    row = sample_random_mlp(rng)
                else:
                    row = sample_random_cnn(rng)
                log(f"Trial {ti}/{n_iter} (random) {row}")

            ns = argparse.Namespace(**vars(deepcopy(args_base)))
            for k, v in _TORCH_TRAIN_DEFAULTS.items():
                if not hasattr(ns, k):
                    setattr(ns, k, v)
            for k, v in row.items():
                if k == "mlp_hidden":
                    continue
                if k in _CNN_KEYS_TO_ARG:
                    continue
                setattr(ns, k, v)
            ns.max_epochs = int(_TORCH_TRAIN_DEFAULTS["max_epochs"])
            ns.patience = int(_TORCH_TRAIN_DEFAULTS["patience"])
            ns.min_delta = float(_TORCH_TRAIN_DEFAULTS["min_delta"])
            ns.lr = round_lr_wd(float(ns.lr))
            ns.weight_decay = round_lr_wd(float(ns.weight_decay))
            ns.dropout = round_dropout(float(ns.dropout))

            if args_base.model_kind == "mlp":
                ns.mlp_hidden = str(row["mlp_hidden"])
                mlp_hidden = _parse_mlp_hidden(ns.mlp_hidden)
                cnn_kw = None
            else:
                ns.mlp_hidden = "256,128"
                mlp_hidden = (256, 128)
                cnn_kw = cnn_kwargs_from_sample_row(row)

            df_scores, df_curves = run_torch_cross_validation(
                ns,
                ds,
                device=device,
                mlp_hidden=mlp_hidden,
                out_dir=out_dir,
                make_plots=False,
                log=log,
                cnn_kwargs=cnn_kw,
                collect_learning_curves=True,
                trial_idx=ti,
            )
            mean_rmse = float(df_scores["rmse"].mean())
            mean_r2 = float(df_scores["r2"].mean())
            mean_pearson_r = float(df_scores["pearson_r"].mean())
            log(
                f"  -> mean_rmse={mean_rmse:.5f} mean_r2={mean_r2:.4f} mean_pearson_r={mean_pearson_r:.4f}"
            )

            lc_dir = out_dir / "learning_curves"
            by_trial = lc_dir / "by_trial"
            df_curves.to_csv(by_trial / f"trial_{ti:04d}.csv", index=False)
            all_curves.append(df_curves)

            plot_learning_curve_epochs(
                df_curves,
                lc_dir / "plots" / f"trial_{ti:04d}_val_rmse.png",
                title=f"Trial {ti} — val RMSE (original scale) per fold",
                y_metric="val_rmse",
                trial_idx=ti,
            )
            plot_learning_curve_r2_epochs(
                df_curves,
                lc_dir / "plots" / f"trial_{ti:04d}_val_r2.png",
                title=f"Trial {ti} — val R² per fold",
                trial_idx=ti,
            )

            if wandb_run is not None:
                import wandb  # type: ignore[import-untyped]

                wandb_run.log(
                    {
                        "hyperopt_trial": float(ti),
                        "trial_idx": ti,
                        "mean_rmse": mean_rmse,
                        "mean_r2": mean_r2,
                        "mean_pearson_r": mean_pearson_r,
                    }
                )
                try:
                    _wandb_log_learning_curve_scalars(wandb_run, df_curves, ti)
                except Exception as ex:
                    log(f"wandb learning curve scalars (trial {ti}) skipped: {ex}")
                try:
                    log_plot_images(
                        wandb_run,
                        {
                            f"plots/learning_curves/trial_{ti:04d}_val_rmse": lc_dir
                            / "plots"
                            / f"trial_{ti:04d}_val_rmse.png",
                            f"plots/learning_curves/trial_{ti:04d}_val_r2": lc_dir
                            / "plots"
                            / f"trial_{ti:04d}_val_r2.png",
                        },
                    )
                except Exception as ex:
                    log(f"wandb learning curve images (trial {ti}) skipped: {ex}")

            record: dict[str, Any] = {
                "trial_idx": ti,
                "mean_rmse": mean_rmse,
                "mean_r2": mean_r2,
                "mean_pearson_r": mean_pearson_r,
                "model_kind": args_base.model_kind,
            }
            record.update(row)
            record["max_epochs"] = int(_TORCH_TRAIN_DEFAULTS["max_epochs"])
            record["patience"] = int(_TORCH_TRAIN_DEFAULTS["patience"])
            record["min_delta"] = float(_TORCH_TRAIN_DEFAULTS["min_delta"])
            results.append(record)

        df_trials = pd.DataFrame(results)
        out_csv = out_dir / f"torch_random_search_{args_base.split_col}_{args_base.model_kind}.csv"
        df_trials.to_csv(out_csv, index=False)
        log(f"Wrote {out_csv}")

        if all_curves:
            pd.concat(all_curves, ignore_index=True).to_csv(out_dir / "learning_curves" / "all_trials_long.csv", index=False)
            log(f"Wrote {out_dir / 'learning_curves' / 'all_trials_long.csv'}")

        best = df_trials.sort_values("mean_rmse").iloc[0]
        best_dict = best.to_dict()
        log("Best trial by mean_rmse (CV):")
        for k, v in sorted(best_dict.items()):
            log(f"  {k}: {v}")
        log(
            f"Best hyperparameters — mean CV R²: {float(best['mean_r2']):.4f}, "
            f"mean CV Pearson r: {float(best['mean_pearson_r']):.4f}"
        )
        (out_dir / "best_hyperparameters.json").write_text(json.dumps(best_dict, indent=2, default=str), encoding="utf-8")
        log(f"Wrote {out_dir / 'best_hyperparameters.json'}")

        excl = frozenset(
            {
                "trial_idx",
                "mean_rmse",
                "mean_r2",
                "mean_pearson_r",
                "model_kind",
                "max_epochs",
                "patience",
                "min_delta",
            }
        )
        run_torch_hyperparam_overview(df_trials, out_dir / "hyperparam_sweep", exclude_cols=excl)

        if wandb_run is not None:
            sweep_plot_dir = out_dir / "hyperparam_sweep" / "plots"
            try:
                imgs = {
                    f"plots/hyperparam_sweep/{p.stem}": p
                    for p in sorted(sweep_plot_dir.glob("*.png"))
                    if p.is_file()
                }
                if imgs:
                    log_plot_images(wandb_run, imgs)
            except Exception as ex:
                log(f"wandb hyperparam_sweep images skipped: {ex}")
            wandb_run.summary["best_mean_rmse"] = float(best["mean_rmse"])
            wandb_run.summary["best_mean_r2"] = float(best["mean_r2"])
            wandb_run.summary["best_mean_pearson_r"] = float(best["mean_pearson_r"])
            wandb_run.summary["best_row"] = best_dict

        if args_base.best_retrain:
            log("Retraining best configuration on full CV (+ saving best_model.pt, predictions CSVs)...")
            ns_best, mlp_h, cnn_k = apply_torch_hyperopt_row(args_base, best, args_base.model_kind)
            retrain_best_torch_and_save_artifacts(
                args_ns=ns_best,
                ds=ds,
                mlp_hidden=mlp_h,
                cnn_kwargs=cnn_k,
                device=device,
                out_dir=out_dir,
                model_kind=args_base.model_kind,
                random_state=int(args_base.random_state),
                log=log,
            )
        else:
            log("Skipping full-CV retrain (--best-retrain not set).")
    finally:
        finish_run(wandb_run)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
