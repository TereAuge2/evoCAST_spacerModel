"""
Randomized hyperparameter search for XGBoost regressor (GroupKFold).

Writes RandomizedSearchCV trial table, hyperparameter sweep plots/dataframes,
optional boosting-round learning curve for the best params (first CV fold),
and best_hyperparameters.json (includes mean_r2 and mean_pearson_r for the best trial).
Full-dataset prediction CSVs only with --save-predictions.
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
from scipy.stats import randint
from sklearn.base import clone
from sklearn.model_selection import GroupKFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from spacerModel.training.search_baseline import prepend_baseline_sklearn_trials_df, sklearn_baseline_cv_row
from spacerModel.training.search_param_rounding import RoundedLogUniform, RoundedUniform
from spacerModel.training.hyperopt_diagnostics import (
    plot_xgboost_boosting_rmse,
    run_torch_hyperparam_overview,
    sklearn_random_search_to_trials_df,
)
from spacerModel.utils.run_paths import default_hyperopt_output_dir, default_run_name, slurm_job_id_prefix
from spacerModel.training.train_torch import build_cv_arrays, dense_from_embedding
from spacerModel.simple_modeling.train_models import (
    build_feature_matrix,
    compute_corr_metrics,
    compute_metrics,
    enrichment_factor_top_frac,
    load_dataset,
    plot_cv_summary,
)
from spacerModel.utils.wandb_util import finish_run, init_wandb_run, log_plot_images

_TRIALS_META_COLS = frozenset({"iter", "mean_rmse", "std_rmse"})


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
    run_name = str(args.run_name).strip() or default_run_name("xgboost")
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
        "model__learning_rate": RoundedLogUniform(0.01, 0.3),
        "model__max_depth": randint(3, 13),
        "model__subsample": RoundedUniform(0.6, 0.35, ndigits=4),
        "model__colsample_bytree": RoundedUniform(0.6, 0.35, ndigits=4),
        "model__reg_lambda": RoundedLogUniform(1e-3, 10.0),
        "model__reg_alpha": RoundedLogUniform(1e-8, 1.0),
        "model__min_child_weight": RoundedLogUniform(0.1, 10.0),
    }

    job_id = slurm_job_id_prefix()
    wandb_run = init_wandb_run(
        use_wandb=bool(args.use_wandb),
        name=run_name,
        job_id=job_id,
        tags=["xgboost_tune"],
        config={
            "split_col": args.split_col,
            "embedding": args.embedding,
            "excel": str(args.excel),
            "kmer_k": int(args.kmer_k),
            "raw_onehot": bool(args.raw_onehot),
            "svd_components": int(args.svd_components),
            "n_iter": int(args.n_iter),
        },
        entity=str(args.wandb_entity).strip() or None,
        project=str(args.wandb_project).strip() or None,
    )
    if wandb_run is not None:
        import wandb  # type: ignore[import-untyped]

        try:
            wandb.define_metric("hyperopt_trial")
            wandb.define_metric("trial_idx", step_metric="hyperopt_trial")
            wandb.define_metric("mean_rmse", step_metric="hyperopt_trial")
            wandb.define_metric("hparam/*", step_metric="hyperopt_trial")
            wandb.define_metric("boost_round")
            wandb.define_metric("curves/xgboost_best/val_rmse", step_metric="boost_round")
        except Exception:
            pass

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

        df_search = sklearn_random_search_to_trials_df(search)
        bl_row = sklearn_baseline_cv_row(
            pipe,
            {},
            X_cv,
            y_cv,
            groups,
            gkf,
            df_search,
            n_jobs=int(args.n_jobs),
        )
        df_trials = prepend_baseline_sklearn_trials_df(df_search, bl_row)
        log(
            f"default-baseline CV mean_rmse={bl_row['mean_rmse']:.5f} (iter=0; XGBRegressor defaults)"
        )
        df_trials.to_csv(out_dir / "random_search_trials_xgboost.csv", index=False)
        run_torch_hyperparam_overview(
            df_trials,
            out_dir / "hyperparam_sweep",
            rmse_col="mean_rmse",
            exclude_cols=frozenset({"iter", "std_rmse"}),
        )

        if wandb_run is not None:
            for ti, (_, trow) in enumerate(df_trials.iterrows(), start=1):
                payload: dict[str, object] = {
                    "hyperopt_trial": float(ti),
                    "trial_idx": ti,
                    "mean_rmse": float(trow["mean_rmse"]),
                }
                for hk, hv in trow.items():
                    if hk in ("iter", "mean_rmse", "std_rmse"):
                        continue
                    if isinstance(hv, (float, np.floating)) and pd.isna(hv):
                        continue
                    payload[f"hparam/{hk}"] = hv
                wandb_run.log(payload)
            sweep_plot_dir = out_dir / "hyperparam_sweep" / "plots"
            if sweep_plot_dir.is_dir():
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

        best_idx = df_trials["mean_rmse"].idxmin()
        best_series = df_trials.loc[best_idx]
        best_params_updates = {
            k: best_series[k]
            for k in df_trials.columns
            if k not in _TRIALS_META_COLS and pd.notna(best_series[k])
        }
        est_best = clone(pipe)
        est_best.set_params(**best_params_updates)

        rows_scores: list[dict[str, object]] = []
        for fold_idx, (tr2, va2) in enumerate(gkf.split(X_cv, y_cv, groups=groups), start=1):
            est_fold = clone(est_best)
            est_fold.fit(X_cv[tr2], y_cv[tr2])
            pred = est_fold.predict(X_cv[va2])
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
        mean_r2_cv = float(df_scores["r2"].mean())
        mean_pearson_r_cv = float(df_scores["pearson_r"].mean())

        best_row = best_series.to_dict()
        best_row["mean_r2"] = mean_r2_cv
        best_row["mean_pearson_r"] = mean_pearson_r_cv

        log("Best hyperparameters (lowest mean CV RMSE):")
        for k, v in sorted(best_row.items()):
            log(f"  {k}: {v}")
        log(
            f"Best trial — mean CV R²: {mean_r2_cv:.4f}, mean CV Pearson r: {mean_pearson_r_cv:.4f}"
        )
        (out_dir / "best_hyperparameters.json").write_text(json.dumps(best_row, indent=2, default=str), encoding="utf-8")
        log(f"Wrote {out_dir / 'best_hyperparameters.json'}")

        best_fitted = clone(est_best)
        best_fitted.fit(X_cv, y_cv)
        joblib.dump(best_fitted, out_dir / "best_model_xgboost.joblib")
        log(f"saved {out_dir / 'best_model_xgboost.joblib'}")

        # Boosting-round curve: best params, first CV fold validation RMSE
        tr, va = next(iter(gkf.split(X_cv, y_cv, groups=groups)))
        best_pipe = clone(est_best)
        best_pipe.fit(X_cv[tr], y_cv[tr])
        scaler = best_pipe.named_steps["scaler"]
        m = best_pipe.named_steps["model"]
        X_tr_s = scaler.transform(X_cv[tr])
        X_va_s = scaler.transform(X_cv[va])
        y_tr, y_va = y_cv[tr], y_cv[va]
        # eval_metric belongs on the estimator (XGBoost 2.x+), not on fit()
        _p = dict(m.get_params())
        _p.setdefault("eval_metric", "rmse")
        xgb_fit = XGBRegressor(**_p)
        xgb_fit.fit(
            X_tr_s,
            y_tr,
            eval_set=[(X_va_s, y_va)],
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
        boost_plot_path = out_dir / "hyperparam_sweep" / "plots" / "xgboost_boosting_val_rmse_fold1.png"
        plot_xgboost_boosting_rmse(
            rounds,
            rmse_hist,
            boost_plot_path,
            title="Best params — val RMSE vs boosting round (first CV fold)",
        )
        if wandb_run is not None:
            try:
                for i, rmse_v in enumerate(rmse_hist):
                    wandb_run.log(
                        {
                            "boost_round": float(i + 1),
                            "curves/xgboost_best/val_rmse": float(rmse_v),
                        }
                    )
                if boost_plot_path.is_file():
                    log_plot_images(wandb_run, {"plots/learning_curves/xgboost_boost_fold1": boost_plot_path})
            except Exception as ex:
                log(f"wandb boosting curve logging skipped: {ex}")

        if args.save_predictions:
            pred_cv = best_fitted.predict(X_cv)
            pd.DataFrame({"y_true": y_cv, "y_pred": pred_cv, "split": "cv"}).to_csv(
                out_dir / "predictions_vs_actual_cv_xgboost.csv", index=False
            )
            mask_test = ~mask_cv
            if int(mask_test.sum()) > 0:
                X_all = build_feature_matrix(ds.df, embedding=args.embedding, kmer_k=int(args.kmer_k))
                X_full = dense_from_embedding(X_all, ns)
                y_full = ds.y
                pred_te = best_fitted.predict(X_full[mask_test])
                pd.DataFrame(
                    {"y_true": y_full[mask_test], "y_pred": pred_te, "split": "test"}
                ).to_csv(out_dir / "predictions_vs_actual_test_xgboost.csv", index=False)

        log(f"best_r2_cv (mean across folds): {mean_r2_cv:.4f}")
        log(f"best_pearson_r_cv (mean across folds): {mean_pearson_r_cv:.4f}")

        (out_dir / f"best_params_xgboost_{args.split_col}{out_tag}.json").write_text(
            json.dumps(
                {
                    "best_rmse_cv": float(best_row["mean_rmse"]),
                    "best_r2_cv_mean": mean_r2_cv,
                    "best_pearson_r_cv_mean": mean_pearson_r_cv,
                    "best_params": {
                        k: float(v)
                        if isinstance(v, (float, np.floating))
                        else int(v)
                        if isinstance(v, (int, np.integer))
                        else v
                        for k, v in best_params_updates.items()
                    },
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        scores_path = out_dir / f"cv_scores_{args.split_col}{out_tag}_tune_xgboost.csv"
        df_scores.to_csv(scores_path, index=False)
        log(f"Wrote {scores_path}")

        cv_plot_path = out_dir / f"cv_summary_{args.split_col}{out_tag}_tune_xgboost.png"
        plot_cv_summary(
            df_scores,
            cv_plot_path,
            title=f"{args.split_col}{out_tag} XGBoost tune (RMSE)",
        )
        if wandb_run is not None:
            try:
                wandb_run.summary["best_mean_rmse"] = float(best_row["mean_rmse"])
                wandb_run.summary["best_rmse_cv"] = float(best_row["mean_rmse"])
                wandb_run.summary["best_mean_r2"] = mean_r2_cv
                wandb_run.summary["best_mean_pearson_r"] = mean_pearson_r_cv
                wandb_run.summary["best_r2_cv_mean"] = mean_r2_cv
                wandb_run.summary["best_pearson_r_cv_mean"] = mean_pearson_r_cv
                wandb_run.summary["best_params"] = best_params_updates
                wandb_run.summary["random_search_best_rmse"] = float(-search.best_score_)
                wandb_run.summary["best_row"] = {
                    k: (float(v) if isinstance(v, (float, np.floating)) else v) for k, v in best_row.items()
                }
            except Exception:
                pass
            try:
                if cv_plot_path.is_file():
                    log_plot_images(wandb_run, {"plots/cv_summary": cv_plot_path})
            except Exception as ex:
                log(f"wandb cv_summary image skipped: {ex}")
        log("Done")
    finally:
        finish_run(wandb_run)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
