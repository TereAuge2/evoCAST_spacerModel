"""
Randomized hyperparameter search for XGBoost regressor (GroupKFold).

Writes RandomizedSearchCV trial table, hyperparameter sweep plots/dataframes,
optional boosting-round learning curve for the best params (first CV fold),
and best_hyperparameters.json. Full-dataset prediction CSVs only with --save-predictions.
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
from scipy.stats import loguniform, randint, uniform
from sklearn.base import clone
from sklearn.model_selection import GroupKFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from spacerModel.models.hyperopt_diagnostics import (
    plot_xgboost_boosting_rmse,
    run_torch_hyperparam_overview,
    sklearn_random_search_to_trials_df,
)
from spacerModel.models.run_paths import default_hyperopt_output_dir, default_run_name, slurm_job_id_prefix
from spacerModel.models.train_torch import build_cv_arrays, dense_from_embedding
from spacerModel.simple_modeling.train_models import build_feature_matrix, load_dataset, plot_cv_summary
from spacerModel.models.wandb_util import finish_run, init_wandb_run


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
        help="Default: output/modeling/xgboost_hyperparameter_tune",
    )
    ap.add_argument("--run-name", default="", help="Default: <SLURM_JOB_ID>_<timestamp>")
    ap.add_argument("--n-iter", type=int, default=40)
    ap.add_argument("--n-jobs", type=int, default=1)
    ap.add_argument("--use-wandb", action="store_true")
    ap.add_argument("--wandb-entity", default="")
    ap.add_argument("--wandb-project", default="")
    ap.add_argument(
        "--save-predictions",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Write predictions_vs_actual_* CSVs (in-sample on CV / test). Default: false.",
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
    base_out = Path(args.output_dir) if str(args.output_dir).strip() else default_hyperopt_output_dir(modeling_root, "xgboost")
    run_name = str(args.run_name).strip() or default_run_name()
    out_dir = base_out / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    log(f"Output directory: {out_dir}")

    ds = load_dataset(Path(args.excel), sheet=args.sheet, split_col=args.split_col, target_col=args.target_col)
    ns = argparse.Namespace(**vars(args))
    X_cv, y_cv, groups, out_tag, mask_cv = build_cv_arrays(ds, ns)

    gkf = GroupKFold(n_splits=min(5, len(set(groups))))

    xgb = XGBRegressor(
        objective="reg:squarederror",
        random_state=int(args.random_state),
        n_jobs=1,
    )
    pipe = Pipeline([("scaler", StandardScaler()), ("model", xgb)])
    param_dist = {
        "model__n_estimators": randint(200, 2501),
        "model__learning_rate": loguniform(0.01, 0.3),
        "model__max_depth": randint(3, 13),
        "model__subsample": uniform(0.6, 0.35),
        "model__colsample_bytree": uniform(0.6, 0.35),
        "model__reg_lambda": loguniform(1e-3, 10.0),
        "model__reg_alpha": loguniform(1e-8, 1.0),
        "model__min_child_weight": loguniform(0.1, 10.0),
    }

    job_id = slurm_job_id_prefix()
    wandb_run = init_wandb_run(
        use_wandb=bool(args.use_wandb),
        name=run_name,
        job_id=job_id,
        tags=["xgboost_tune"],
        config={"split_col": args.split_col, "embedding": args.embedding, "n_iter": args.n_iter},
        entity=str(args.wandb_entity).strip() or None,
        project=str(args.wandb_project).strip() or None,
    )

    try:
        log(f"RandomizedSearchCV xgboost n_iter={args.n_iter}")
        search = RandomizedSearchCV(
            pipe,
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
        log(f"best RMSE: {-search.best_score_:.5f}")
        log(f"best_params: {search.best_params_}")

        joblib.dump(search.best_estimator_, out_dir / "best_model_xgboost.joblib")
        log(f"saved {out_dir / 'best_model_xgboost.joblib'}")

        df_trials = sklearn_random_search_to_trials_df(search)
        df_trials.to_csv(out_dir / "random_search_trials_xgboost.csv", index=False)
        run_torch_hyperparam_overview(
            df_trials,
            out_dir / "hyperparam_sweep",
            rmse_col="mean_rmse",
            exclude_cols=frozenset({"iter", "std_rmse"}),
        )

        best_row = df_trials.loc[df_trials["mean_rmse"].idxmin()].to_dict()
        log("Best hyperparameters (lowest mean CV RMSE):")
        for k, v in sorted(best_row.items()):
            log(f"  {k}: {v}")
        (out_dir / "best_hyperparameters.json").write_text(json.dumps(best_row, indent=2, default=str), encoding="utf-8")

        # Boosting-round curve: best params, first CV fold validation RMSE
        tr, va = next(iter(gkf.split(X_cv, y_cv, groups=groups)))
        best_pipe = clone(search.best_estimator_)
        best_pipe.fit(X_cv[tr], y_cv[tr])
        scaler = best_pipe.named_steps["scaler"]
        m = best_pipe.named_steps["model"]
        X_tr_s = scaler.transform(X_cv[tr])
        X_va_s = scaler.transform(X_cv[va])
        y_tr, y_va = y_cv[tr], y_cv[va]
        xgb_fit = XGBRegressor(**m.get_params())
        xgb_fit.fit(
            X_tr_s,
            y_tr,
            eval_set=[(X_va_s, y_va)],
            eval_metric="rmse",
            verbose=False,
        )
        er = xgb_fit.evals_result()
        val_key = list(er.keys())[0]
        inner = er[val_key]
        rmse_key = "rmse" if "rmse" in inner else list(inner.keys())[0]
        rmse_hist = inner[rmse_key]
        rounds = list(range(1, len(rmse_hist) + 1))
        pd.DataFrame({"boost_round": rounds, "val_rmse": rmse_hist}).to_csv(
            out_dir / "boosting_learning_curve_fold1.csv", index=False
        )
        plot_xgboost_boosting_rmse(
            rounds,
            rmse_hist,
            out_dir / "hyperparam_sweep" / "plots" / "xgboost_boosting_val_rmse_fold1.png",
            title="Best params — val RMSE vs boosting round (first CV fold)",
        )

        if args.save_predictions:
            pred_cv = search.best_estimator_.predict(X_cv)
            pd.DataFrame({"y_true": y_cv, "y_pred": pred_cv, "split": "cv"}).to_csv(
                out_dir / "predictions_vs_actual_cv_xgboost.csv", index=False
            )
            mask_test = ~mask_cv
            if int(mask_test.sum()) > 0:
                X_all = build_feature_matrix(ds.df, embedding=args.embedding, kmer_k=int(args.kmer_k))
                X_full = dense_from_embedding(X_all, ns)
                y_full = ds.y
                pred_te = search.best_estimator_.predict(X_full[mask_test])
                pd.DataFrame(
                    {"y_true": y_full[mask_test], "y_pred": pred_te, "split": "test"}
                ).to_csv(out_dir / "predictions_vs_actual_test_xgboost.csv", index=False)

        if wandb_run is not None:
            wandb_run.summary["best_rmse_cv"] = float(-search.best_score_)
            wandb_run.summary["best_params"] = search.best_params_

        (out_dir / f"best_params_xgboost_{args.split_col}{out_tag}.json").write_text(
            json.dumps(
                {
                    "best_rmse_cv": float(-search.best_score_),
                    "best_params": {
                        k: float(v)
                        if isinstance(v, (float, np.floating))
                        else int(v)
                        if isinstance(v, (int, np.integer))
                        else v
                        for k, v in search.best_params_.items()
                    },
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        best = search.best_estimator_
        rows_scores: list[dict[str, object]] = []
        for fold_idx, (tr2, va2) in enumerate(gkf.split(X_cv, y_cv, groups=groups), start=1):
            est_fold = clone(search.best_estimator_)
            est_fold.fit(X_cv[tr2], y_cv[tr2])
            pred = est_fold.predict(X_cv[va2])
            from spacerModel.simple_modeling.train_models import (
                compute_corr_metrics,
                compute_metrics,
                enrichment_factor_top_frac,
            )

            m_ = compute_metrics(y_cv[va2], pred)
            m_.update(compute_corr_metrics(y_cv[va2], pred))
            m_["ef10"] = enrichment_factor_top_frac(y_cv[va2], pred, frac=0.10)
            rows_scores.append(
                {
                    "split_col": args.split_col,
                    "model": "xgboost",
                    "fold": fold_idx,
                    "n_train": len(tr2),
                    "n_val": len(va2),
                    **m_,
                }
            )

        df_scores = pd.DataFrame(rows_scores)
        scores_path = out_dir / f"cv_scores_{args.split_col}{out_tag}_tune_xgboost.csv"
        df_scores.to_csv(scores_path, index=False)
        log(f"Wrote {scores_path}")

        plot_cv_summary(
            df_scores,
            out_dir / f"cv_summary_{args.split_col}{out_tag}_tune_xgboost.png",
            title=f"{args.split_col}{out_tag} XGBoost tune (RMSE)",
        )
        log("Done")
    finally:
        finish_run(wandb_run)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
