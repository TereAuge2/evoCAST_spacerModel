"""
Retrain + compare optimized models.

Inputs: per-model best_hyperparameters.json files (CNN/MLP/XGBoost/SVR/NuSVR/KernelRidge)
or a single consolidated bundle JSON created by this script.

Outputs:
  - optimized_hyperparameters.json (bundle)
  - cv_scores_optimized_models.csv (+ cv_summary_optimized_models.png via plot_cv_summary)
  - per-model regression plots (fold + pooled)
  - optional --evaluate-test artifacts (test metrics + regression plot + summary barplot)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import TransformedTargetRegressor
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import NuSVR, SVR
from sklearn.kernel_ridge import KernelRidge

from spacerModel.simple_modeling.train_models import (
    build_feature_matrix,
    compute_corr_metrics,
    compute_metrics,
    enrichment_factor_top_frac,
    load_dataset,
    plot_cv_summary,
    regression_scatter_plot,
)
from spacerModel.training.train_torch import (
    apply_torch_hyperopt_row,
    build_cv_arrays,
    dense_from_embedding,
    resolve_device,
    retrain_best_torch_and_save_artifacts,
    run_torch_cross_validation,
)
from spacerModel.utils.run_paths import default_run_name


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _coerce_number(v: Any) -> Any:
    if isinstance(v, (int, float, str)):
        try:
            if isinstance(v, str) and v.strip() == "":
                return v
            f = float(v)
            if np.isfinite(f) and float(int(f)) == f:
                return int(f)
            return float(f)
        except Exception:
            return v
    return v


def load_or_build_bundle(
    *,
    bundle_path: Path | None,
    cnn_json: Path | None,
    mlp_json: Path | None,
    xgb_json: Path | None,
    svr_json: Path | None,
    nusvr_json: Path | None,
    krr_json: Path | None,
) -> dict[str, Any]:
    if bundle_path is not None and bundle_path.exists():
        b = _read_json(bundle_path)
        if not isinstance(b, dict) or "models" not in b:
            raise ValueError(f"bundle file missing top-level 'models': {bundle_path}")
        return b

    req = {
        "cnn": cnn_json,
        "mlp": mlp_json,
        "xgboost": xgb_json,
        "svr_rbf": svr_json,
        "nusvr_rbf": nusvr_json,
        "kernel_ridge_rbf": krr_json,
    }
    missing = [k for k, p in req.items() if p is None]
    if missing:
        raise ValueError(f"Missing required inputs (or pass --bundle): {missing}")

    models: dict[str, Any] = {}
    for name, p in req.items():
        assert p is not None
        d = _read_json(p)
        models[name] = d

    return {
        "schema": "spacerModel.optimized_hyperparameters.v1",
        "models": models,
    }


def _sklearn_cv_with_params(
    *,
    model_name: str,
    estimator: Any,
    model_params: dict[str, Any],
    X_cv: np.ndarray,
    y_cv: np.ndarray,
    groups: np.ndarray,
    out_dir: Path,
    split_plot: str,
    use_hexbin: bool,
    random_state: int,
) -> pd.DataFrame:
    unique_groups = sorted(set(groups))
    gkf = GroupKFold(n_splits=min(5, len(unique_groups)))
    rows: list[dict[str, object]] = []
    all_val_true: list[np.ndarray] = []
    all_val_pred: list[np.ndarray] = []

    # Standard scaling on X + target scaling on y, consistent with train_models.py.
    pipe = Pipeline([("scaler", StandardScaler()), ("model", clone(estimator))])
    pipe.set_params(**{k: _coerce_number(v) for k, v in model_params.items()})
    reg = TransformedTargetRegressor(regressor=pipe, transformer=StandardScaler())

    for fold_idx, (tr, va) in enumerate(gkf.split(X_cv, y_cv, groups=groups), start=1):
        reg_fold = clone(reg)
        reg_fold.fit(X_cv[tr], y_cv[tr])
        pred = reg_fold.predict(X_cv[va])

        m = compute_metrics(y_cv[va], pred)
        m.update(compute_corr_metrics(y_cv[va], pred))
        m["ef10"] = enrichment_factor_top_frac(y_cv[va], pred, frac=0.10)
        rows.append(
            {
                "split_col": split_plot,
                "model": model_name,
                "fold": fold_idx,
                "n_train": len(tr),
                "n_val": len(va),
                **m,
            }
        )
        all_val_true.append(np.asarray(y_cv[va], dtype=float))
        all_val_pred.append(np.asarray(pred, dtype=float))

        plot_dir = out_dir / "plots" / "cv" / model_name
        plot_dir.mkdir(parents=True, exist_ok=True)
        regression_scatter_plot(
            y_cv[va],
            pred,
            title=f"{model_name} — {split_plot} fold {fold_idx}",
            outpath=plot_dir / f"regression_fold{fold_idx}.png",
            use_hexbin=use_hexbin,
        )

    if all_val_true:
        y_all = np.concatenate(all_val_true, axis=0)
        p_all = np.concatenate(all_val_pred, axis=0)
        plot_dir = out_dir / "plots" / "cv" / model_name
        regression_scatter_plot(
            y_all,
            p_all,
            title=f"{model_name} — {split_plot} pooled CV",
            outpath=plot_dir / "regression_pooled.png",
            use_hexbin=use_hexbin,
        )
        pd.DataFrame({"y_true": y_all, "y_pred": p_all}).to_csv(
            out_dir / "predictions" / f"predictions_vs_actual_cv_{model_name}.csv", index=False
        )

    return pd.DataFrame(rows)


def _save_final_models(
    *,
    bundle: dict[str, Any],
    ds: Any,
    args_base: argparse.Namespace,
    out_dir: Path,
    random_state: int,
) -> None:
    """
    Save one fitted "best" model per algorithm under out_dir/final_models.

    Torch: saves checkpoints as CNN_best_model.pt / MLP_best_model.pt
    Sklearn: saves joblib as XGBOOST_best_model.joblib / SVR_best_model.joblib / NUSVR_best_model.joblib / KERNEL_RIDGE_best_model.joblib
    """
    final_dir = out_dir / "final_models"
    final_dir.mkdir(parents=True, exist_ok=True)

    # ---- Torch: train on all CV rows with internal 90/10 early-stop split and save checkpoint.
    for kind, fname in (("cnn", "CNN_best_model.pt"), ("mlp", "MLP_best_model.pt")):
        best = bundle["models"][kind]
        ns = argparse.Namespace(**vars(deepcopy(args_base)))
        ns.model_kind = kind
        if kind == "cnn":
            ns.embedding = "onehot"
            ns.raw_onehot = True
        for k in ("lr", "weight_decay", "batch_size", "dropout"):
            if k in best:
                setattr(ns, k, _coerce_number(best[k]))
        device = resolve_device(str(getattr(ns, "device", "auto")))
        ns2, mlp_hidden, cnn_kwargs = apply_torch_hyperopt_row(ns, pd.Series(best), kind)
        tmp_out = out_dir / "final_models" / f"_tmp_torch_{kind}"
        tmp_out.mkdir(parents=True, exist_ok=True)
        retrain_best_torch_and_save_artifacts(
            args_ns=ns2,
            ds=ds,
            mlp_hidden=mlp_hidden,
            cnn_kwargs=cnn_kwargs,
            device=device,
            out_dir=tmp_out,
            model_kind=kind,  # type: ignore[arg-type]
            random_state=int(random_state),
            log=lambda _m: None,
        )
        src = tmp_out / "best_model.pt"
        if src.is_file():
            (final_dir / fname).write_bytes(src.read_bytes())

    # ---- Sklearn: fit on all CV rows, with the same scaling conventions used elsewhere.
    X_cv, y_cv, _groups, _out_tag, _mask_cv = build_cv_arrays(ds, args_base)
    model_specs: list[tuple[str, Any, str]] = [
        ("svr_rbf", SVR(), "SVR_best_model.joblib"),
        ("nusvr_rbf", NuSVR(), "NUSVR_best_model.joblib"),
        ("kernel_ridge_rbf", KernelRidge(kernel="rbf"), "KERNEL_RIDGE_best_model.joblib"),
    ]
    # Optional XGBoost
    try:
        from xgboost import XGBRegressor  # type: ignore

        model_specs.append(
            ("xgboost", XGBRegressor(objective="reg:squarederror", random_state=int(random_state), n_jobs=1), "XGBOOST_best_model.joblib")
        )
    except Exception:
        pass

    for model_key, est, fname in model_specs:
        best = bundle["models"].get(model_key)
        if not isinstance(best, dict):
            continue
        params = {k: best[k] for k in best.keys() if str(k).startswith("model__")}
        pipe = Pipeline([("scaler", StandardScaler()), ("model", clone(est))])
        pipe.set_params(**{k: _coerce_number(v) for k, v in params.items()})
        reg = TransformedTargetRegressor(regressor=pipe, transformer=StandardScaler())
        reg.fit(X_cv, y_cv)
        joblib.dump(reg, final_dir / fname)


def _evaluate_test_mode(
    *,
    bundle: dict[str, Any],
    ds: Any,
    args_base: argparse.Namespace,
    out_dir: Path,
    use_hexbin: bool,
    random_state: int,
) -> pd.DataFrame:
    """
    Train once per model on CV rows with an internal 90/10 holdout; evaluate on true test set.
    For torch models, the 10% is used for early stopping. For sklearn models, it's only a held-out
    subset (no early stopping), but we keep the split consistent.
    """
    X_cv, y_cv, _groups, out_tag, mask_cv = build_cv_arrays(ds, args_base)
    mask_test = ~mask_cv
    if int(mask_test.sum()) <= 0:
        raise ValueError("No test rows found (mask_test is empty). Check split_col values.")

    X_all = build_feature_matrix(ds.df, embedding=args_base.embedding, kmer_k=int(args_base.kmer_k))
    X_full = dense_from_embedding(X_all, args_base)
    X_test = X_full[mask_test]
    y_test = ds.y[mask_test]

    n = int(X_cv.shape[0])
    rng = np.random.default_rng(int(random_state))
    perm = rng.permutation(n)
    n_va = max(1, int(round(0.1 * n)))
    va_sub = perm[:n_va]
    tr_sub = perm[n_va:]

    rows: list[dict[str, object]] = []
    pred_dir = out_dir / "predictions"
    plot_dir = out_dir / "plots" / "test"
    pred_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    # ---- Torch: reuse retrain_best_torch_and_save_artifacts but point it at our out_dir/model subfolders
    for kind in ("cnn", "mlp"):
        best = bundle["models"][kind]
        ns = argparse.Namespace(**vars(deepcopy(args_base)))
        ns.model_kind = kind
        if kind == "cnn":
            # CNN requires raw one-hot (4-channel) features, not kmer/gc embeddings.
            ns.embedding = "onehot"
            ns.raw_onehot = True
        for k in ("lr", "weight_decay", "batch_size", "dropout"):
            if k in best:
                setattr(ns, k, _coerce_number(best[k]))
        device = resolve_device(str(getattr(ns, "device", "auto")))
        # apply_torch_hyperopt_row gives us mlp_hidden/cnn_kwargs consistent with training code
        ns2, mlp_hidden, cnn_kwargs = apply_torch_hyperopt_row(ns, pd.Series(best), kind)

        # Train on CV with internal 90/10 split for early stopping and then evaluate test.
        sub_out = out_dir / "evaluate_test" / f"torch_{kind}"
        sub_out.mkdir(parents=True, exist_ok=True)
        ckpt_path = retrain_best_torch_and_save_artifacts(
            args_ns=ns2,
            ds=ds,
            mlp_hidden=mlp_hidden,
            cnn_kwargs=cnn_kwargs,
            device=device,
            out_dir=sub_out,
            model_kind=kind,  # type: ignore[arg-type]
            random_state=int(random_state),
            log=lambda _m: None,
        )
        _ = ckpt_path
        # The retrain helper already writes predictions_vs_actual_test.csv if test exists.
        dfp = pd.read_csv(sub_out / "predictions_vs_actual_test.csv")
        y_pred = dfp["y_pred"].to_numpy(dtype=float)
        m = compute_metrics(y_test, y_pred)
        m.update(compute_corr_metrics(y_test, y_pred))
        m["ef10"] = enrichment_factor_top_frac(y_test, y_pred, frac=0.10)
        rows.append({"split_col": f"{args_base.split_col}{out_tag}", "model": f"torch_{kind}", "fold": 0, **m})
        regression_scatter_plot(
            y_test,
            y_pred,
            title=f"torch_{kind} — test",
            outpath=plot_dir / f"regression_plot_2_torch_{kind}.png",
            use_hexbin=use_hexbin,
        )

    # ---- Sklearn models
    def _fit_predict_sklearn(model_key: str, est: Any, params: dict[str, Any]) -> tuple[np.ndarray, dict[str, float]]:
        pipe_x = Pipeline([("scaler", StandardScaler())])
        pipe_y = StandardScaler()
        X_tr = pipe_x.fit_transform(X_cv[tr_sub])
        X_te = pipe_x.transform(X_test)
        y_tr = pipe_y.fit_transform(y_cv[tr_sub].reshape(-1, 1)).ravel()
        base = Pipeline([("model", clone(est))])
        # keep parameter names consistent with your hyperopt JSONs: model__*
        base.set_params(**{k: _coerce_number(v) for k, v in params.items()})
        base.fit(X_tr, y_tr)
        pred_s = base.predict(X_te)
        pred = pipe_y.inverse_transform(np.asarray(pred_s).reshape(-1, 1)).ravel()
        m = compute_metrics(y_test, pred)
        m.update(compute_corr_metrics(y_test, pred))
        m["ef10"] = enrichment_factor_top_frac(y_test, pred, frac=0.10)
        return pred, m

    # XGBoost (if installed) – created dynamically to avoid hard dependency
    try:
        from xgboost import XGBRegressor  # type: ignore

        best = bundle["models"]["xgboost"]
        params = {k: best[k] for k in best.keys() if str(k).startswith("model__")}
        xgb = XGBRegressor(objective="reg:squarederror", random_state=int(random_state), n_jobs=1)
        pred, m = _fit_predict_sklearn("xgboost", xgb, params)
        rows.append({"split_col": f"{args_base.split_col}{out_tag}", "model": "xgboost", "fold": 0, **m})
        regression_scatter_plot(
            y_test,
            pred,
            title="xgboost — test",
            outpath=plot_dir / "regression_plot_2_xgboost.png",
            use_hexbin=use_hexbin,
        )
    except Exception:
        pass

    # SVR / NuSVR / KernelRidge
    for model_key, est in (
        ("svr_rbf", SVR()),
        ("nusvr_rbf", NuSVR()),
        ("kernel_ridge_rbf", KernelRidge(kernel="rbf")),
    ):
        best = bundle["models"][model_key]
        params = {k: best[k] for k in best.keys() if str(k).startswith("model__")}
        pred, m = _fit_predict_sklearn(model_key, est, params)
        rows.append({"split_col": f"{args_base.split_col}{out_tag}", "model": model_key, "fold": 0, **m})
        regression_scatter_plot(
            y_test,
            pred,
            title=f"{model_key} — test",
            outpath=plot_dir / f"regression_plot_2_{model_key}.png",
            use_hexbin=use_hexbin,
        )

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "test_scores_optimized_models.csv", index=False)
    plot_cv_summary(df, out_dir / "test_summary_optimized_models.png", title="Optimized models — test metrics")
    return df


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--excel",
        default="output/data_processing/251129_P8_120Complete_dataset_woCtrl_exogenous_with_splits.xlsx",
    )
    ap.add_argument(
        "--sheet",
        default=0,
        help="Excel sheet name or index (0-based). If an integer is provided, it is treated as an index.",
    )
    ap.add_argument("--target-col", default="FE")
    ap.add_argument("--split-col", choices=["Split", "SplitRandom"], default="Split")
    ap.add_argument("--random-state", type=int, default=42)
    ap.add_argument("--embedding", choices=["onehot", "onehot_gc", "kmer", "kmer_gc", "kmer_gc_onehot"], default="kmer_gc_onehot")
    ap.add_argument("--kmer-k", type=int, default=2, choices=[2, 3, 4])
    ap.add_argument("--svd-components", type=int, default=82)
    ap.add_argument("--svd-full-variance", action="store_true")
    ap.add_argument(
        "--raw-onehot",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    ap.add_argument("--output-dir", default="output/modeling/optimized_retrain")
    ap.add_argument("--run-name", default="", help="Default: timestamped run name")
    ap.add_argument("--use-hexbin", action="store_true", default=True)
    ap.add_argument("--device", default="auto", help="torch only: cpu|cuda|auto")

    ap.add_argument("--bundle", default="", help="Optional consolidated optimized_hyperparameters.json")
    ap.add_argument("--cnn-json", default="", help="CNN best_hyperparameters.json")
    ap.add_argument("--mlp-json", default="", help="MLP best_hyperparameters.json")
    ap.add_argument("--xgb-json", default="", help="XGBoost best_hyperparameters.json")
    ap.add_argument("--svr-json", default="", help="SVR best_hyperparameters_svr_rbf.json")
    ap.add_argument("--nusvr-json", default="", help="NuSVR best_hyperparameters_nusvr_rbf.json")
    ap.add_argument("--krr-json", default="", help="KernelRidge best_hyperparameters_kernel_ridge_rbf.json")

    ap.add_argument("--evaluate-test", action="store_true", default=False)

    args = ap.parse_args(argv)

    try:
        sys.stdout.reconfigure(line_buffering=True)  # type: ignore[attr-defined]
    except Exception:
        pass

    t0 = time.time()

    def log(msg: str) -> None:
        print(f"[{time.time() - t0:8.1f}s] {msg}", flush=True)

    out_root = Path(args.output_dir)
    run_name = str(args.run_name).strip() or default_run_name("optimized_retrain")
    out_dir = out_root / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "predictions").mkdir(parents=True, exist_ok=True)
    log(f"Output directory: {out_dir}")

    bundle_path = Path(args.bundle) if str(args.bundle).strip() else None
    bundle = load_or_build_bundle(
        bundle_path=bundle_path,
        cnn_json=Path(args.cnn_json) if str(args.cnn_json).strip() else None,
        mlp_json=Path(args.mlp_json) if str(args.mlp_json).strip() else None,
        xgb_json=Path(args.xgb_json) if str(args.xgb_json).strip() else None,
        svr_json=Path(args.svr_json) if str(args.svr_json).strip() else None,
        nusvr_json=Path(args.nusvr_json) if str(args.nusvr_json).strip() else None,
        krr_json=Path(args.krr_json) if str(args.krr_json).strip() else None,
    )
    bundle_out = out_dir / "optimized_hyperparameters.json"
    bundle_out.write_text(json.dumps(bundle, indent=2, default=str), encoding="utf-8")
    log(f"Wrote {bundle_out}")

    sheet: str | int
    if isinstance(args.sheet, int):
        sheet = int(args.sheet)
    else:
        s = str(args.sheet).strip()
        sheet = int(s) if s.lstrip("-").isdigit() else s
    ds = load_dataset(Path(args.excel), sheet=sheet, split_col=args.split_col, target_col=args.target_col)
    args_ns = argparse.Namespace(**vars(args))
    X_cv, y_cv, groups, out_tag, _mask_cv = build_cv_arrays(ds, args_ns)
    split_plot = f"{args.split_col}{out_tag}"

    # ---- Torch CV retrain (with those hyperparameters)
    device = resolve_device(str(args.device))
    rows_all: list[pd.DataFrame] = []
    for kind in ("mlp", "cnn"):
        best = bundle["models"][kind]
        ns = argparse.Namespace(**vars(deepcopy(args_ns)))
        ns.model_kind = kind
        if kind == "cnn":
            # CNN requires raw one-hot (4-channel) features, not kmer/gc embeddings.
            ns.embedding = "onehot"
            ns.raw_onehot = True
        for k in ("lr", "weight_decay", "batch_size", "dropout"):
            if k in best:
                setattr(ns, k, _coerce_number(best[k]))
        ns2, mlp_hidden, cnn_kwargs = apply_torch_hyperopt_row(ns, pd.Series(best), kind)
        log(f"Running torch_{kind} CV with optimized hyperparameters…")
        df_scores = run_torch_cross_validation(
            ns2,
            ds,
            device=device,
            mlp_hidden=mlp_hidden,
            cnn_kwargs=cnn_kwargs,
            out_dir=out_dir / "torch_cv" / kind,
            make_plots=True,
            log=log,
            collect_learning_curves=False,
        )
        rows_all.append(df_scores)

    # ---- Sklearn CV retrain
    log("Running sklearn CV with optimized hyperparameters…")
    # XGBoost (optional)
    try:
        from xgboost import XGBRegressor  # type: ignore

        best = bundle["models"]["xgboost"]
        params = {k: best[k] for k in best.keys() if str(k).startswith("model__")}
        xgb = XGBRegressor(objective="reg:squarederror", random_state=int(args.random_state), n_jobs=1)
        rows_all.append(
            _sklearn_cv_with_params(
                model_name="xgboost",
                estimator=xgb,
                model_params=params,
                X_cv=X_cv,
                y_cv=y_cv,
                groups=groups,
                out_dir=out_dir,
                split_plot=split_plot,
                use_hexbin=bool(args.use_hexbin),
                random_state=int(args.random_state),
            )
        )
    except Exception as ex:
        log(f"Skipping xgboost retrain (import failed): {ex}")

    for model_key, est in (
        ("svr_rbf", SVR()),
        ("nusvr_rbf", NuSVR()),
        ("kernel_ridge_rbf", KernelRidge(kernel="rbf")),
    ):
        best = bundle["models"][model_key]
        params = {k: best[k] for k in best.keys() if str(k).startswith("model__")}
        rows_all.append(
            _sklearn_cv_with_params(
                model_name=model_key,
                estimator=est,
                model_params=params,
                X_cv=X_cv,
                y_cv=y_cv,
                groups=groups,
                out_dir=out_dir,
                split_plot=split_plot,
                use_hexbin=bool(args.use_hexbin),
                random_state=int(args.random_state),
            )
        )

    df_scores_all = pd.concat(rows_all, ignore_index=True)
    scores_path = out_dir / "cv_scores_optimized_models.csv"
    df_scores_all.to_csv(scores_path, index=False)
    log(f"Wrote {scores_path}")
    plot_cv_summary(df_scores_all, out_dir / "cv_summary_optimized_models.png", title="Optimized models — CV metrics")
    log("Wrote cv_summary_optimized_models.png")

    log("Saving final best models to final_models/ …")
    _save_final_models(bundle=bundle, ds=ds, args_base=args_ns, out_dir=out_dir, random_state=int(args.random_state))
    log("Saved final_models/*.pt and final_models/*.joblib")

    if args.evaluate_test:
        log("Running --evaluate-test…")
        _evaluate_test_mode(
            bundle=bundle,
            ds=ds,
            args_base=args_ns,
            out_dir=out_dir,
            use_hexbin=bool(args.use_hexbin),
            random_state=int(args.random_state),
        )
        log("Wrote test artifacts (plots + scores).")

    log("Done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

