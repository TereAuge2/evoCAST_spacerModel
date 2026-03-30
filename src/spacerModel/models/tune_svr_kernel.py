"""
Randomized hyperparameter search for SVR, NuSVR, and KernelRidge (RBF kernel).
GroupKFold as in train_models. Writes trial tables, hyperparam sweep plots, best_hyperparameters per model.
Prediction CSVs only with --save-predictions.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import loguniform, uniform
from sklearn.base import clone
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import GroupKFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import NuSVR, SVR

from spacerModel.models.hyperopt_diagnostics import run_torch_hyperparam_overview, sklearn_random_search_to_trials_df
from spacerModel.models.run_paths import default_hyperopt_output_dir, default_run_name, slurm_job_id_prefix
from spacerModel.models.train_torch import build_cv_arrays, dense_from_embedding
from spacerModel.simple_modeling.train_models import build_feature_matrix, load_dataset, plot_cv_summary
from spacerModel.models.wandb_util import finish_run, init_wandb_run


def make_estimator(name: str, random_state: int) -> tuple[Pipeline, dict]:
    _ = random_state
    if name == "svr_rbf":
        est = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", SVR(kernel="rbf")),
            ]
        )
        param = {
            "model__C": loguniform(1e-2, 1e3),
            "model__gamma": loguniform(1e-7, 1.0),
            "model__epsilon": loguniform(1e-4, 0.5),
        }
        return est, param
    if name == "nusvr_rbf":
        est = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", NuSVR(kernel="rbf")),
            ]
        )
        param = {
            "model__C": loguniform(1e-2, 1e3),
            "model__gamma": loguniform(1e-7, 1.0),
            "model__nu": uniform(0.1, 0.8),
        }
        return est, param
    if name == "kernel_ridge_rbf":
        est = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", KernelRidge(kernel="rbf")),
            ]
        )
        param = {
            "model__alpha": loguniform(1e-6, 10.0),
            "model__gamma": loguniform(1e-7, 1.0),
        }
        return est, param
    raise ValueError(f"Unknown model: {name}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--excel",
        default="output/data_processing/251129_P8_120Complete_dataset_woCtrl_exogenous_with_splits.xlsx",
    )
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
        help="Dense features (default on). Use --no-raw-onehot for TruncatedSVD on sparse input.",
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
        help="Default: output/modeling/svr_hyperparameter_tune",
    )
    ap.add_argument("--run-name", default="", help="Default: <SLURM_JOB_ID>_<timestamp>")
    ap.add_argument(
        "--models",
        default="svr_rbf,nusvr_rbf,kernel_ridge_rbf",
        help="Comma-separated: svr_rbf, nusvr_rbf, kernel_ridge_rbf",
    )
    ap.add_argument("--n-iter", type=int, default=40, help="RandomizedSearchCV iterations per model")
    ap.add_argument("--n-jobs", type=int, default=-1)
    ap.add_argument("--use-wandb", action="store_true")
    ap.add_argument("--wandb-entity", default="")
    ap.add_argument("--wandb-project", default="")
    ap.add_argument(
        "--save-predictions",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Write predictions_vs_actual_* CSVs. Default: false.",
    )
    args = ap.parse_args(argv)

    try:
        sys.stdout.reconfigure(line_buffering=True)  # type: ignore[attr-defined]
    except Exception:
        pass

    t0 = time.time()

    def log(msg: str) -> None:
        print(f"[{time.time() - t0:8.1f}s] {msg}", flush=True)

    modeling_root = Path("output/modeling")
    base_out = Path(args.output_dir) if str(args.output_dir).strip() else default_hyperopt_output_dir(modeling_root, "svr")
    run_name = str(args.run_name).strip() or default_run_name()
    out_dir = base_out / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    log(f"Output directory: {out_dir}")

    ds = load_dataset(Path(args.excel), sheet=args.sheet, split_col=args.split_col, target_col=args.target_col)
    ns = argparse.Namespace(**vars(args))
    X_cv, y_cv, groups, out_tag, mask_cv = build_cv_arrays(ds, ns)

    gkf = GroupKFold(n_splits=min(5, len(set(groups))))
    wanted = [m.strip() for m in args.models.split(",") if m.strip()]

    job_id = slurm_job_id_prefix()
    wandb_run = init_wandb_run(
        use_wandb=bool(args.use_wandb),
        name=run_name,
        job_id=job_id,
        tags=["svr_tune", "kernel_ridge"],
        config={"split_col": args.split_col, "embedding": args.embedding, "models": wanted},
        entity=str(args.wandb_entity).strip() or None,
        project=str(args.wandb_project).strip() or None,
    )

    rows_scores: list[dict[str, object]] = []
    try:
        for model_name in wanted:
            log(f"RandomizedSearchCV: {model_name} (n_iter={args.n_iter})")
            est, param_dist = make_estimator(model_name, int(args.random_state))
            search = RandomizedSearchCV(
                est,
                param_distributions=param_dist,
                n_iter=int(args.n_iter),
                cv=gkf,
                scoring="neg_root_mean_squared_error",
                random_state=int(args.random_state),
                n_jobs=int(args.n_jobs),
                refit=True,
                verbose=0,
            )
            search.fit(X_cv, y_cv, groups=groups)
            log(f"  best RMSE (neg score): {-search.best_score_:.5f}")
            log(f"  best_params: {search.best_params_}")

            joblib.dump(search.best_estimator_, out_dir / f"best_model_{model_name}.joblib")
            log(f"  saved {out_dir / f'best_model_{model_name}.joblib'}")

            df_trials = sklearn_random_search_to_trials_df(search)
            df_trials.to_csv(out_dir / f"random_search_trials_{model_name}.csv", index=False)
            sweep_dir = out_dir / "hyperparam_sweep" / model_name
            run_torch_hyperparam_overview(
                df_trials,
                sweep_dir,
                rmse_col="mean_rmse",
                exclude_cols=frozenset({"iter", "std_rmse"}),
            )
            best_row = df_trials.loc[df_trials["mean_rmse"].idxmin()].to_dict()
            log(f"  Best hyperparameters ({model_name}):")
            for k, v in sorted(best_row.items()):
                log(f"    {k}: {v}")
            (out_dir / f"best_hyperparameters_{model_name}.json").write_text(
                json.dumps(best_row, indent=2, default=str), encoding="utf-8"
            )

            if args.save_predictions:
                pred_cv = search.best_estimator_.predict(X_cv)
                pd.DataFrame({"y_true": y_cv, "y_pred": pred_cv, "split": "cv"}).to_csv(
                    out_dir / f"predictions_vs_actual_cv_{model_name}.csv", index=False
                )
                mask_test = ~mask_cv
                if int(mask_test.sum()) > 0:
                    X_all = build_feature_matrix(ds.df, embedding=args.embedding, kmer_k=int(args.kmer_k))
                    X_full = dense_from_embedding(X_all, ns)
                    y_full = ds.y
                    pred_te = search.best_estimator_.predict(X_full[mask_test])
                    pd.DataFrame(
                        {"y_true": y_full[mask_test], "y_pred": pred_te, "split": "test"}
                    ).to_csv(out_dir / f"predictions_vs_actual_test_{model_name}.csv", index=False)

            if wandb_run is not None:
                wandb_run.log({f"{model_name}/best_rmse_cv": float(-search.best_score_)})

            for fold_idx, (tr, va) in enumerate(gkf.split(X_cv, y_cv, groups=groups), start=1):
                est_fold = clone(search.best_estimator_)
                est_fold.fit(X_cv[tr], y_cv[tr])
                pred = est_fold.predict(X_cv[va])
                from spacerModel.simple_modeling.train_models import (
                    compute_corr_metrics,
                    compute_metrics,
                    enrichment_factor_top_frac,
                )

                m = compute_metrics(y_cv[va], pred)
                m.update(compute_corr_metrics(y_cv[va], pred))
                m["ef10"] = enrichment_factor_top_frac(y_cv[va], pred, frac=0.10)
                rows_scores.append(
                    {
                        "split_col": args.split_col,
                        "model": model_name,
                        "fold": fold_idx,
                        "n_train": len(tr),
                        "n_val": len(va),
                        **m,
                    }
                )

            meta = {
                "model": model_name,
                "best_rmse_cv": float(-search.best_score_),
                "best_params_json": json.dumps(
                    {k: (float(v) if isinstance(v, (float, np.floating)) else v) for k, v in search.best_params_.items()}
                ),
            }
            (out_dir / f"best_params_{model_name}_{args.split_col}{out_tag}.json").write_text(
                json.dumps(meta, indent=2),
                encoding="utf-8",
            )

        df_scores = pd.DataFrame(rows_scores)
        scores_path = out_dir / f"cv_scores_{args.split_col}{out_tag}_tune_svr_kernel.csv"
        df_scores.to_csv(scores_path, index=False)
        log(f"Wrote {scores_path}")

        plot_cv_summary(
            df_scores,
            out_dir / f"cv_summary_{args.split_col}{out_tag}_tune_svr_kernel.png",
            title=f"{args.split_col}{out_tag} SVR/kernel tune (RMSE)",
        )
        log("Done")
    finally:
        finish_run(wandb_run)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
