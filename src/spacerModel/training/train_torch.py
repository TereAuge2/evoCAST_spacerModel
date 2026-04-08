"""
Train / evaluate / test PyTorch MLP and CNN regressors with early stopping.
Plots reuse regression_scatter_plot aesthetics from simple_modeling.train_models.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from collections.abc import Callable
from typing import Any, Literal, overload

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy import sparse
from sklearn.compose import TransformedTargetRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from spacerModel.models import CNN1DRegressorNet, MLPRegressorNet, infer_cnn_in_channels, to_channel_first
from spacerModel.training.search_param_rounding import round_dropout
from spacerModel.simple_modeling.train_models import (
    Dataset,
    build_feature_matrix,
    compute_corr_metrics,
    compute_metrics,
    enrichment_factor_top_frac,
    fit_svd,
    load_dataset,
    max_svd_components,
    plot_cv_summary,
    regression_scatter_plot,
)

ModelKind = Literal["mlp", "cnn"]


@dataclass
class EarlyStoppingState:
    patience: int
    min_delta: float
    best_loss: float | None = None
    best_epoch: int = -1
    epochs_without_improve: int = 0
    best_state: dict[str, torch.Tensor] | None = None

    def update(self, epoch: int, val_loss: float, model: nn.Module) -> bool:
        """Return True if training should stop."""
        if self.best_loss is None or val_loss < self.best_loss - self.min_delta:
            self.best_loss = float(val_loss)
            self.best_epoch = int(epoch)
            self.epochs_without_improve = 0
            self.best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            return False
        self.epochs_without_improve += 1
        return self.epochs_without_improve >= self.patience


def resolve_device(explicit: str) -> torch.device:
    ex = explicit.strip().lower()
    if ex == "cpu":
        return torch.device("cpu")
    if ex in ("cuda", "gpu"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
        return torch.device("cuda")
    if ex == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    raise ValueError(f"Unknown device: {explicit}")


def _batch_indices(n: int, batch_size: int, rng: np.random.Generator) -> list[np.ndarray]:
    idx = np.arange(n)
    rng.shuffle(idx)
    return [idx[i : i + batch_size] for i in range(0, n, batch_size)]


def _forward(
    model: nn.Module,
    xb: torch.Tensor,
    kind: ModelKind,
) -> torch.Tensor:
    if kind == "mlp":
        return model(xb)
    # CNN: xb is (B, d) features -> (B, C, L)
    x_np = xb.detach().cpu().numpy()
    xc = to_channel_first(x_np)
    xt = torch.from_numpy(xc).to(xb.device, dtype=torch.float32)
    return model(xt)


def train_one_epoch(
    model: nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    kind: ModelKind,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    batch_size: int,
    rng: np.random.Generator,
) -> float:
    model.train()
    n = int(X_train.shape[0])
    bs = max(8, int(batch_size))
    total_loss = 0.0
    total_n = 0
    for batch_idx in _batch_indices(n, bs, rng):
        xb = torch.from_numpy(X_train[batch_idx]).to(device, dtype=torch.float32)
        yb = torch.from_numpy(y_train[batch_idx]).to(device, dtype=torch.float32)
        optimizer.zero_grad(set_to_none=True)
        pred = _forward(model, xb, kind)
        loss = loss_fn(pred, yb)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item()) * int(batch_idx.size)
        total_n += int(batch_idx.size)
    return total_loss / max(1, total_n)


@torch.no_grad()
def evaluate_loss(
    model: nn.Module,
    X: np.ndarray,
    y: np.ndarray,
    *,
    kind: ModelKind,
    device: torch.device,
    loss_fn: nn.Module,
    batch_size: int = 256,
) -> float:
    model.eval()
    n = int(X.shape[0])
    bs = max(32, int(batch_size))
    total_loss = 0.0
    total_n = 0
    for i in range(0, n, bs):
        sl = slice(i, min(i + bs, n))
        xb = torch.from_numpy(X[sl]).to(device, dtype=torch.float32)
        yb = torch.from_numpy(y[sl]).to(device, dtype=torch.float32)
        pred = _forward(model, xb, kind)
        loss = loss_fn(pred, yb)
        total_loss += float(loss.item()) * int(sl.stop - sl.start)
        total_n += int(sl.stop - sl.start)
    return total_loss / max(1, total_n)


@torch.no_grad()
def predict_numpy(
    model: nn.Module,
    X: np.ndarray,
    *,
    kind: ModelKind,
    device: torch.device,
    batch_size: int = 256,
) -> np.ndarray:
    model.eval()
    n = int(X.shape[0])
    bs = max(32, int(batch_size))
    outs: list[np.ndarray] = []
    for i in range(0, n, bs):
        sl = slice(i, min(i + bs, n))
        xb = torch.from_numpy(X[sl]).to(device, dtype=torch.float32)
        pred = _forward(model, xb, kind)
        outs.append(pred.detach().cpu().numpy().astype(np.float64, copy=False))
    return np.concatenate(outs, axis=0)


def train_with_early_stopping(
    model: nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    *,
    kind: ModelKind,
    device: torch.device,
    max_epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    patience: int,
    min_delta: float,
    random_state: int,
    log_prefix: str = "",
    y_val_orig: np.ndarray | None = None,
    pipe_y: StandardScaler | None = None,
) -> tuple[nn.Module, dict[str, Any]]:
    """Fit model with validation-based early stopping; restores best weights.

    If ``y_val_orig`` and ``pipe_y`` are set, ``history`` also records ``val_rmse`` and ``val_r2``
    on the **original** target scale each epoch (for learning curves).
    """
    loss_fn = nn.MSELoss()
    opt = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))
    rng = np.random.default_rng(int(random_state))
    stop = EarlyStoppingState(patience=int(patience), min_delta=float(min_delta))
    history: dict[str, list[float]] = {"train_loss": [], "val_loss": [], "val_rmse": [], "val_r2": []}
    track_metrics = y_val_orig is not None and pipe_y is not None

    for ep in range(max(1, int(max_epochs))):
        tr_loss = train_one_epoch(
            model,
            X_train,
            y_train,
            kind=kind,
            device=device,
            optimizer=opt,
            loss_fn=loss_fn,
            batch_size=batch_size,
            rng=rng,
        )
        va_loss = evaluate_loss(model, X_val, y_val, kind=kind, device=device, loss_fn=loss_fn, batch_size=batch_size)
        history["train_loss"].append(tr_loss)
        history["val_loss"].append(va_loss)
        if track_metrics:
            assert y_val_orig is not None and pipe_y is not None
            pred_s = predict_numpy(model, X_val, kind=kind, device=device, batch_size=batch_size)
            pred = pipe_y.inverse_transform(np.asarray(pred_s).reshape(-1, 1)).ravel()
            yo = np.asarray(y_val_orig, dtype=float)
            history["val_rmse"].append(float(np.sqrt(mean_squared_error(yo, pred))))
            history["val_r2"].append(float(r2_score(yo, pred)))
        else:
            history["val_rmse"].append(float("nan"))
            history["val_r2"].append(float("nan"))
        if log_prefix:
            extra = ""
            if track_metrics and not np.isnan(history["val_rmse"][-1]):
                extra = f" val_rmse={history['val_rmse'][-1]:.5f} val_r2={history['val_r2'][-1]:.4f}"
            print(
                f"{log_prefix} epoch {ep + 1}/{max_epochs} train_loss={tr_loss:.6f} val_loss={va_loss:.6f}{extra}",
                flush=True,
            )
        if stop.update(ep, va_loss, model):
            if log_prefix:
                print(f"{log_prefix} early stop at epoch {ep + 1} (best epoch {stop.best_epoch + 1})", flush=True)
            break

    if stop.best_state is not None:
        model.load_state_dict(stop.best_state)
    model.to(device)
    return model, {"best_epoch": stop.best_epoch, "best_val_loss": stop.best_loss, "history": history}


def evaluate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    m = compute_metrics(y_true, y_pred)
    m.update(compute_corr_metrics(y_true, y_pred))
    m["ef10"] = enrichment_factor_top_frac(y_true, y_pred, frac=0.10)
    return m


def test_predict(
    model: nn.Module,
    X: np.ndarray,
    *,
    kind: ModelKind,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    return predict_numpy(model, X, kind=kind, device=device, batch_size=batch_size)


def cnn_kwargs_from_args(args: argparse.Namespace) -> dict[str, object] | None:
    """Optional CNN architecture overrides from CLI (train_torch --cnn-*)."""
    aliases = [
        ("cnn_conv1_channels", "conv1_channels"),
        ("cnn_conv2_channels", "conv2_channels"),
        ("cnn_kernel1", "kernel1"),
        ("cnn_kernel2", "kernel2"),
        ("cnn_pool_out", "pool_out"),
        ("cnn_head_hidden", "head_hidden"),
    ]
    out: dict[str, object] = {}
    for attr, dest in aliases:
        if hasattr(args, attr):
            v = getattr(args, attr, None)
            if v is not None:
                out[dest] = int(v)
    return out if out else None


def _build_model(
    kind: ModelKind,
    n_features: int,
    *,
    mlp_hidden: tuple[int, ...],
    dropout: float,
    cnn_kwargs: dict[str, object] | None = None,
) -> nn.Module:
    if kind == "mlp":
        return MLPRegressorNet(int(n_features), hidden=mlp_hidden, dropout=dropout)
    ch = int(infer_cnn_in_channels(int(n_features)))
    kw: dict[str, object] = {"dropout": float(dropout)}
    if cnn_kwargs:
        kw.update(cnn_kwargs)
    return CNN1DRegressorNet(ch, **kw)


def _out_tag(args: argparse.Namespace) -> str:
    out_tag = ""
    if args.embedding != "onehot":
        out_tag += f"_{args.embedding}"
    if args.embedding.startswith("kmer"):
        out_tag += f"_k{int(args.kmer_k)}"
    if args.raw_onehot and args.embedding in ("onehot", "onehot_gc", "kmer_gc_onehot"):
        out_tag += "_raw_onehot"
    return out_tag


def dense_from_embedding(X_part: np.ndarray | sparse.csr_matrix, args: argparse.Namespace) -> np.ndarray:
    """Dense design matrix for sklearn/torch (matches train_models.py)."""
    if args.embedding in {"kmer", "kmer_gc"}:
        return np.asarray(X_part, dtype=np.float64)
    if not sparse.issparse(X_part):
        raise ValueError("Expected sparse matrix for onehot embeddings")
    if args.raw_onehot:
        return np.asarray(X_part.toarray(), dtype=np.float64)
    if args.svd_full_variance:
        n_svd = max_svd_components(X_part)
    else:
        n_svd = int(args.svd_components)
    X_d, _ = fit_svd(X_part, n_components=n_svd, random_state=int(args.random_state))
    return X_d


def round_lr_wd(x: float) -> float:
    """Round learning rate and weight decay to 5 decimal places before training."""
    return round(float(x), 5)


# Defaults aligned with tuned CNN run (e.g. best_hyperparameters.json from CNN hyperparameter search).
_TORCH_HYPEROPT_DEFAULTS: dict[str, Any] = {
    "lr": round_lr_wd(0.00013),
    "weight_decay": round_lr_wd(0.00091),
    "batch_size": 32,
    "dropout": 0.3,
    "max_epochs": 200,
    "patience": 10,
    "min_delta": 1e-4,
}

_CNN_GRID_KEYS_TO_ARG: dict[str, str] = {
    "cnn_conv1_channels": "conv1_channels",
    "cnn_conv2_channels": "conv2_channels",
    "cnn_kernel1": "kernel1",
    "cnn_kernel2": "kernel2",
    "cnn_pool_out": "pool_out",
    "cnn_head_hidden": "head_hidden",
}


def apply_torch_hyperopt_row(
    args_base: argparse.Namespace,
    row: dict[str, Any] | pd.Series,
    model_kind: str,
) -> tuple[argparse.Namespace, tuple[int, ...], dict[str, Any] | None]:
    """Rebuild Namespace + architecture from a grid-search result row (CSV / DataFrame row)."""
    if isinstance(row, pd.Series):
        row = {str(k): row[k] for k in row.index}
    ns = argparse.Namespace(**vars(deepcopy(args_base)))
    for k, v in _TORCH_HYPEROPT_DEFAULTS.items():
        if not hasattr(ns, k):
            setattr(ns, k, v)
    for k, v in row.items():
        if k in ("grid_idx", "trial_idx", "mean_rmse", "mean_r2", "mean_pearson_r", "model_kind", "trial"):
            continue
        if k == "mlp_hidden":
            continue
        if k in _CNN_GRID_KEYS_TO_ARG:
            continue
        if isinstance(v, (float, np.floating)) and np.isnan(v):
            continue
        setattr(ns, k, v)
    # Training controls were not tuned; always use current defaults.
    ns.max_epochs = int(_TORCH_HYPEROPT_DEFAULTS["max_epochs"])
    ns.patience = int(_TORCH_HYPEROPT_DEFAULTS["patience"])
    ns.min_delta = float(_TORCH_HYPEROPT_DEFAULTS["min_delta"])
    ns.lr = round_lr_wd(float(ns.lr))
    ns.weight_decay = round_lr_wd(float(ns.weight_decay))
    ns.dropout = round_dropout(float(ns.dropout))
    if model_kind == "mlp":
        if "mlp_hidden" not in row:
            raise KeyError("MLP grid row must include mlp_hidden")
        ns.mlp_hidden = str(row["mlp_hidden"])
        mlp_hidden = tuple(int(x.strip()) for x in str(ns.mlp_hidden).split(",") if x.strip())
        cnn_kw = None
    else:
        ns.mlp_hidden = "256,128"
        mlp_hidden = (256, 128)
        cnn_kw = {
            _CNN_GRID_KEYS_TO_ARG[sk]: int(row[sk])
            for sk in _CNN_GRID_KEYS_TO_ARG
            if sk in row and row[sk] is not None and not (isinstance(row[sk], float) and np.isnan(row[sk]))
        }
        cnn_kw = cnn_kw if cnn_kw else None
    return ns, mlp_hidden, cnn_kw


def save_torch_best_checkpoint(
    path: Path,
    model: nn.Module,
    *,
    model_kind: ModelKind,
    mlp_hidden: tuple[int, ...],
    cnn_kwargs: dict[str, Any] | None,
    dropout: float,
    n_features: int,
    pipe_x: Pipeline,
    pipe_y: StandardScaler,
) -> None:
    """Save weights + scaler stats for later inference (load_torch_best_checkpoint)."""
    scaler = pipe_x.named_steps["scaler"]
    pack: dict[str, Any] = {
        "model_kind": model_kind,
        "n_features": int(n_features),
        "dropout": float(dropout),
        "mlp_hidden": list(mlp_hidden),
        "cnn_kwargs": dict(cnn_kwargs or {}),
        "state_dict": model.state_dict(),
        "scaler_x_mean": np.asarray(scaler.mean_, dtype=np.float64),
        "scaler_x_scale": np.asarray(scaler.scale_, dtype=np.float64),
        "scaler_y_mean": np.asarray(pipe_y.mean_, dtype=np.float64),
        "scaler_y_scale": np.asarray(pipe_y.scale_, dtype=np.float64),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(pack, path)


def load_torch_best_checkpoint(path: Path, device: torch.device) -> tuple[nn.Module, dict[str, Any]]:
    """Load model and metadata from save_torch_best_checkpoint."""
    pack = torch.load(path, map_location=device, weights_only=False)
    kind = pack["model_kind"]
    n_features = int(pack["n_features"])
    dropout = float(pack.get("dropout", 0.3))
    mlp_hidden = tuple(int(x) for x in pack["mlp_hidden"])
    cnn_kwargs = pack.get("cnn_kwargs") or None
    model = _build_model(
        kind,
        n_features,
        mlp_hidden=mlp_hidden,
        dropout=dropout,
        cnn_kwargs=cnn_kwargs,
    )
    model.load_state_dict(pack["state_dict"])
    model.to(device)
    model.eval()
    return model, pack


def _restore_standard_scaler_1d(mean: np.ndarray, scale: np.ndarray) -> StandardScaler:
    s = StandardScaler()
    s.mean_ = np.asarray(mean, dtype=np.float64).ravel()
    s.scale_ = np.asarray(scale, dtype=np.float64).ravel()
    s.var_ = s.scale_**2
    s.n_features_in_ = int(s.mean_.size)
    return s


def predict_numpy_from_checkpoint(
    pack: dict[str, Any],
    model: nn.Module,
    X: np.ndarray,
    *,
    kind: ModelKind,
    device: torch.device,
    batch_size: int = 256,
) -> np.ndarray:
    """Apply saved scaler-X, model forward, inverse scaler-y."""
    sx = _restore_standard_scaler_1d(pack["scaler_x_mean"], pack["scaler_x_scale"])
    sy = _restore_standard_scaler_1d(pack["scaler_y_mean"], pack["scaler_y_scale"])
    Xs = sx.transform(np.asarray(X, dtype=np.float64))
    pred_s = predict_numpy(model, Xs.astype(np.float32), kind=kind, device=device, batch_size=batch_size)
    pred = sy.inverse_transform(np.asarray(pred_s).reshape(-1, 1)).ravel()
    return pred


def retrain_best_torch_and_save_artifacts(
    *,
    args_ns: argparse.Namespace,
    ds: Dataset,
    mlp_hidden: tuple[int, ...],
    cnn_kwargs: dict[str, Any] | None,
    device: torch.device,
    out_dir: Path,
    model_kind: ModelKind,
    random_state: int,
    log: Callable[[str], None],
) -> Path:
    """
    Train on all CV rows with 10% holdout for early stopping; save best_model.pt,
    predictions_vs_actual_cv.csv, and optional predictions_vs_actual_test.csv.
    """
    X_cv, y_cv, _g, _tag, mask_cv = build_cv_arrays(ds, args_ns)
    pipe_x = Pipeline([("scaler", StandardScaler())])
    pipe_y = StandardScaler()
    X_tr_full = pipe_x.fit_transform(X_cv)
    y_tr_full = pipe_y.fit_transform(y_cv.reshape(-1, 1)).ravel()
    n = int(X_tr_full.shape[0])
    rng = np.random.default_rng(int(random_state))
    perm = rng.permutation(n)
    n_va = max(1, int(round(0.1 * n)))
    va_sub = perm[:n_va]
    tr_sub = perm[n_va:]
    torch.manual_seed(int(random_state))
    np.random.seed(int(random_state))
    model = _build_model(
        model_kind,
        X_tr_full.shape[1],
        mlp_hidden=mlp_hidden,
        dropout=float(args_ns.dropout),
        cnn_kwargs=cnn_kwargs,
    )
    model.to(device)
    log(f"[best_retrain] training on n={n} (internal val={n_va})")
    model, info = train_with_early_stopping(
        model,
        X_tr_full[tr_sub].astype(np.float32),
        y_tr_full[tr_sub].astype(np.float32),
        X_tr_full[va_sub].astype(np.float32),
        y_tr_full[va_sub].astype(np.float32),
        kind=model_kind,
        device=device,
        max_epochs=int(args_ns.max_epochs),
        batch_size=int(args_ns.batch_size),
        lr=float(args_ns.lr),
        weight_decay=float(args_ns.weight_decay),
        patience=int(args_ns.patience),
        min_delta=float(args_ns.min_delta),
        random_state=int(random_state),
        log_prefix="[best_retrain]",
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = out_dir / "best_model.pt"
    save_torch_best_checkpoint(
        ckpt,
        model,
        model_kind=model_kind,
        mlp_hidden=mlp_hidden,
        cnn_kwargs=cnn_kwargs,
        dropout=float(args_ns.dropout),
        n_features=int(X_tr_full.shape[1]),
        pipe_x=pipe_x,
        pipe_y=pipe_y,
    )
    log(f"[best_retrain] saved {ckpt}")
    pred_cv = test_predict(
        model, X_tr_full.astype(np.float32), kind=model_kind, device=device, batch_size=int(args_ns.batch_size)
    )
    pred_cv = pipe_y.inverse_transform(np.asarray(pred_cv).reshape(-1, 1)).ravel()
    pd.DataFrame({"y_true": y_cv, "y_pred": pred_cv, "split": "cv"}).to_csv(
        out_dir / "predictions_vs_actual_cv.csv", index=False
    )
    mask_test = ~mask_cv
    if int(mask_test.sum()) > 0:
        X_all = build_feature_matrix(ds.df, embedding=args_ns.embedding, kmer_k=int(args_ns.kmer_k))
        X_full = dense_from_embedding(X_all, args_ns)
        X_te = pipe_x.transform(X_full[mask_test])
        y_te = ds.y[mask_test]
        pred_te = test_predict(
            model, X_te.astype(np.float32), kind=model_kind, device=device, batch_size=int(args_ns.batch_size)
        )
        pred_te = pipe_y.inverse_transform(np.asarray(pred_te).reshape(-1, 1)).ravel()
        pd.DataFrame({"y_true": y_te, "y_pred": pred_te, "split": "test"}).to_csv(
            out_dir / "predictions_vs_actual_test.csv", index=False
        )
        log(f"[best_retrain] wrote predictions_vs_actual_test.csv (n={len(y_te)})")
    meta = {
        "best_epoch_retrain": info.get("best_epoch"),
        "best_val_loss_retrain": info.get("best_val_loss"),
        "n_cv": int(len(y_cv)),
        "model_kind": model_kind,
    }
    (out_dir / "best_model_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return ckpt


def build_cv_arrays(ds: Dataset, args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, np.ndarray, str, np.ndarray]:
    """CV dense X, y, groups, plot tag, mask_cv (bool array aligned to ds.df)."""
    mask_cv = ds.df[args.split_col].astype(str).str.startswith("cv").to_numpy()
    X_all = build_feature_matrix(ds.df, embedding=args.embedding, kmer_k=int(args.kmer_k))
    X_cv_all = X_all[mask_cv]
    y_cv = ds.y[mask_cv]
    X_cv = dense_from_embedding(X_cv_all, args)
    groups = ds.df.loc[mask_cv, args.split_col].astype(str).to_numpy()
    return X_cv, y_cv, groups, _out_tag(args), mask_cv


@overload
def run_torch_cross_validation(
    args: argparse.Namespace,
    ds: Dataset,
    *,
    device: torch.device,
    mlp_hidden: tuple[int, ...],
    out_dir: Path,
    make_plots: bool,
    log: Callable[[str], None],
    cnn_kwargs: dict[str, object] | None = None,
    collect_learning_curves: Literal[False] = False,
    trial_idx: int | None = None,
) -> pd.DataFrame: ...


@overload
def run_torch_cross_validation(
    args: argparse.Namespace,
    ds: Dataset,
    *,
    device: torch.device,
    mlp_hidden: tuple[int, ...],
    out_dir: Path,
    make_plots: bool,
    log: Callable[[str], None],
    cnn_kwargs: dict[str, object] | None = None,
    collect_learning_curves: Literal[True] = True,
    trial_idx: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]: ...


def run_torch_cross_validation(
    args: argparse.Namespace,
    ds: Dataset,
    *,
    device: torch.device,
    mlp_hidden: tuple[int, ...],
    out_dir: Path,
    make_plots: bool,
    log: Callable[[str], None],
    cnn_kwargs: dict[str, object] | None = None,
    collect_learning_curves: bool = False,
    trial_idx: int | None = None,
) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]:
    """Run GroupKFold training; write per-fold plots if make_plots. Returns per-fold score rows.

    If ``collect_learning_curves`` is True, also returns a long DataFrame with
    epoch-wise ``val_rmse`` / ``val_r2`` (original target scale) per fold.
    """
    X_cv, y_cv, groups, out_tag, _mask_cv = build_cv_arrays(ds, args)
    unique_groups = sorted(set(groups))
    gkf = GroupKFold(n_splits=min(5, len(unique_groups)))

    model_label = f"torch_{args.model_kind}"
    split_plot = f"{args.split_col}{out_tag}"

    rows: list[dict[str, object]] = []
    all_val_true: list[np.ndarray] = []
    all_val_pred: list[np.ndarray] = []
    curve_rows: list[dict[str, object]] = []

    for fold_idx, (tr, va) in enumerate(gkf.split(X_cv, y_cv, groups=groups), start=1):
        log(f"Fold {fold_idx}: train={len(tr)} val={len(va)}")
        pipe_x = Pipeline([("scaler", StandardScaler())])
        pipe_y = StandardScaler()
        X_tr = pipe_x.fit_transform(X_cv[tr])
        X_va = pipe_x.transform(X_cv[va])
        y_tr = pipe_y.fit_transform(y_cv[tr].reshape(-1, 1)).ravel()
        y_va = pipe_y.transform(y_cv[va].reshape(-1, 1)).ravel()

        torch.manual_seed(int(args.random_state) + fold_idx)
        np.random.seed(int(args.random_state) + fold_idx)
        model = _build_model(
            args.model_kind,
            X_tr.shape[1],
            mlp_hidden=mlp_hidden,
            dropout=float(args.dropout),
            cnn_kwargs=cnn_kwargs,
        )
        model.to(device)

        log_prefix = f"[{model_label}]" if make_plots else ""
        t_kw: dict[str, Any] = {}
        if collect_learning_curves:
            t_kw["y_val_orig"] = y_cv[va]
            t_kw["pipe_y"] = pipe_y
        model, info = train_with_early_stopping(
            model,
            X_tr.astype(np.float32),
            y_tr.astype(np.float32),
            X_va.astype(np.float32),
            y_va.astype(np.float32),
            kind=args.model_kind,
            device=device,
            max_epochs=int(args.max_epochs),
            batch_size=int(args.batch_size),
            lr=float(args.lr),
            weight_decay=float(args.weight_decay),
            patience=int(args.patience),
            min_delta=float(args.min_delta),
            random_state=int(args.random_state) + fold_idx,
            log_prefix=log_prefix,
            **t_kw,
        )
        if collect_learning_curves:
            h = info["history"]
            for ep_idx in range(len(h["train_loss"])):
                curve_rows.append(
                    {
                        "fold": int(fold_idx),
                        "epoch": ep_idx + 1,
                        "train_loss": h["train_loss"][ep_idx],
                        "val_loss": h["val_loss"][ep_idx],
                        "val_rmse": h["val_rmse"][ep_idx],
                        "val_r2": h["val_r2"][ep_idx],
                    }
                )

        pred_va_scaled = test_predict(
            model, X_va.astype(np.float32), kind=args.model_kind, device=device, batch_size=int(args.batch_size)
        )
        pred_va = pipe_y.inverse_transform(np.asarray(pred_va_scaled).reshape(-1, 1)).ravel()

        m = evaluate_metrics(y_cv[va], pred_va)
        rows.append(
            {
                "split_col": args.split_col,
                "model": model_label,
                "fold": fold_idx,
                "val_group": sorted(set(groups[va])),
                "n_train": int(len(tr)),
                "n_val": int(len(va)),
                "best_epoch": info.get("best_epoch"),
                "best_val_loss": info.get("best_val_loss"),
                **m,
            }
        )
        all_val_true.append(y_cv[va])
        all_val_pred.append(np.asarray(pred_va, dtype=float))

        if make_plots:
            plot_dir = out_dir / "plots" / split_plot / model_label
            regression_scatter_plot(
                y_cv[va],
                pred_va,
                title=f"{model_label} — {split_plot} fold {fold_idx}",
                outpath=plot_dir / f"regression_fold{fold_idx}.png",
                use_hexbin=bool(args.use_hexbin),
            )

    if make_plots and all_val_true:
        y_all = np.concatenate(all_val_true, axis=0)
        p_all = np.concatenate(all_val_pred, axis=0)
        plot_dir = out_dir / "plots" / split_plot / model_label
        regression_scatter_plot(
            y_all,
            p_all,
            title=f"{model_label} — {split_plot} pooled CV",
            outpath=plot_dir / "regression_pooled.png",
            use_hexbin=bool(args.use_hexbin),
        )

    df_scores = pd.DataFrame(rows)
    if collect_learning_curves:
        df_curves = pd.DataFrame(curve_rows)
        if trial_idx is not None:
            df_curves["trial_idx"] = int(trial_idx)
        return df_scores, df_curves
    return df_scores


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Train PyTorch MLP or CNN with GroupKFold CV and early stopping.")
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
        help="Dense features (default on). CNN/MLP on kmer_gc_onehot: use --no-raw-onehot for SVD.",
    )
    ap.add_argument(
        "--embedding",
        choices=["onehot", "onehot_gc", "kmer", "kmer_gc", "kmer_gc_onehot"],
        default="kmer_gc_onehot",
        help="Default: raw one-hot + GC + k-mer freqs (k from --kmer-k; 2=dinucleotide).",
    )
    ap.add_argument("--kmer-k", type=int, default=2, choices=[2, 3, 4])
    ap.add_argument("--output-dir", default="output/modeling")
    ap.add_argument("--run-name", default="", help="Subfolder under output-dir")
    ap.add_argument("--model-kind", choices=["mlp", "cnn"], required=True)
    ap.add_argument("--max-epochs", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=0.00013)
    ap.add_argument("--weight-decay", type=float, default=0.00091)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--min-delta", type=float, default=1e-4)
    ap.add_argument("--mlp-hidden", default="256,128", help="Comma-separated hidden sizes for MLP")
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--cnn-conv1-channels", type=int, default=114, help="CNN only: first conv width")
    ap.add_argument("--cnn-conv2-channels", type=int, default=124, help="CNN only: second conv width")
    ap.add_argument("--cnn-head-hidden", type=int, default=256, help="CNN only: linear head hidden size")
    ap.add_argument("--cnn-kernel1", type=int, default=3, help="CNN only: first kernel size")
    ap.add_argument("--cnn-kernel2", type=int, default=3, help="CNN only: second kernel size")
    ap.add_argument("--cnn-pool-out", type=int, default=16, help="CNN only: adaptive pool length")
    ap.add_argument("--device", default="auto", help="auto | cpu | cuda")
    ap.add_argument("--make-plots", action="store_true", default=True)
    ap.add_argument("--use-hexbin", action="store_true", default=True)
    ap.add_argument("--eval-held-out-test", action="store_true", help="Retrain on all CV rows and evaluate on non-cv test rows")
    ap.add_argument("--no-plots", action="store_true", help="Skip regression plots (faster)")
    args = ap.parse_args(argv)
    args.lr = round_lr_wd(float(args.lr))
    args.weight_decay = round_lr_wd(float(args.weight_decay))
    args.dropout = round_dropout(float(args.dropout))

    if args.model_kind == "cnn" and args.embedding != "onehot":
        raise SystemExit(
            "CNN requires --embedding onehot (4-channel sequence). "
            f"Got embedding={args.embedding!r}. Use --model-kind mlp for kmer_gc_onehot."
        )

    t0 = time.time()

    def log(msg: str) -> None:
        print(f"[{time.time() - t0:8.1f}s] {msg}", flush=True)

    try:
        sys.stdout.reconfigure(line_buffering=True)  # type: ignore[attr-defined]
    except Exception:
        pass

    device = resolve_device(args.device)
    log(f"device={device}")

    mlp_hidden = tuple(int(x.strip()) for x in str(args.mlp_hidden).split(",") if x.strip())
    if not mlp_hidden:
        mlp_hidden = (256, 128)
    cnn_kw = cnn_kwargs_from_args(args)

    out_dir = Path(args.output_dir)
    if str(args.run_name).strip():
        out_dir = out_dir / str(args.run_name).strip()
    out_dir.mkdir(parents=True, exist_ok=True)

    ds = load_dataset(Path(args.excel), sheet=args.sheet, split_col=args.split_col, target_col=args.target_col)
    if ds.df[args.split_col].astype(str).str.startswith("cv").sum() == 0:
        raise ValueError("No CV rows found (expected cv* splits).")

    _xc, _yc, _g, out_tag, mask_cv = build_cv_arrays(ds, args)
    model_label = f"torch_{args.model_kind}"
    split_plot = f"{args.split_col}{out_tag}"
    make_plots = bool(args.make_plots) and not bool(args.no_plots)

    df_scores = run_torch_cross_validation(
        args,
        ds,
        device=device,
        mlp_hidden=mlp_hidden,
        out_dir=out_dir,
        make_plots=make_plots,
        log=log,
        cnn_kwargs=cnn_kw,
    )
    scores_path = out_dir / f"cv_scores_{args.split_col}{out_tag}_torch_{args.model_kind}.csv"
    df_scores.to_csv(scores_path, index=False)
    log(f"Wrote {scores_path}")

    df_summary = (
        df_scores.groupby(["split_col", "model"], as_index=False)[["rmse", "mae", "r2"]]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary_csv = out_dir / f"cv_summary_{args.split_col}{out_tag}_torch_{args.model_kind}.csv"
    df_summary.to_csv(summary_csv, index=False)
    plot_cv_summary(
        df_scores,
        out_dir / f"cv_summary_{args.split_col}{out_tag}_torch_{args.model_kind}.png",
        title=f"{split_plot} {model_label} CV summary (RMSE)",
    )
    log(f"Wrote summary plot and {summary_csv.name}")

    if args.eval_held_out_test:
        mask_test = ~mask_cv
        if int(mask_test.sum()) == 0:
            log("No held-out test rows; skipping --eval-held-out-test")
        else:
            log(f"Held-out test: n={int(mask_test.sum())}")
            X_all = build_feature_matrix(ds.df, embedding=args.embedding, kmer_k=int(args.kmer_k))
            X_full = dense_from_embedding(X_all, args)
            y_full = ds.y
            X_train_full = X_full[mask_cv]
            y_train_full = y_full[mask_cv]
            pipe_x = Pipeline([("scaler", StandardScaler())])
            pipe_y = StandardScaler()
            X_tr = pipe_x.fit_transform(X_train_full)
            y_tr = pipe_y.fit_transform(y_train_full.reshape(-1, 1)).ravel()
            X_te = pipe_x.transform(X_full[mask_test])
            y_te = y_full[mask_test]

            torch.manual_seed(int(args.random_state))
            np.random.seed(int(args.random_state))
            model = _build_model(
                args.model_kind,
                X_tr.shape[1],
                mlp_hidden=mlp_hidden,
                dropout=float(args.dropout),
                cnn_kwargs=cnn_kw,
            )
            model.to(device)
            n = int(X_tr.shape[0])
            val_frac = 0.1
            rng = np.random.default_rng(int(args.random_state))
            perm = rng.permutation(n)
            n_va = max(1, int(round(val_frac * n)))
            va_sub = perm[:n_va]
            tr_sub = perm[n_va:]
            model, _ = train_with_early_stopping(
                model,
                X_tr[tr_sub].astype(np.float32),
                y_tr[tr_sub].astype(np.float32),
                X_tr[va_sub].astype(np.float32),
                y_tr[va_sub].astype(np.float32),
                kind=args.model_kind,
                device=device,
                max_epochs=int(args.max_epochs),
                batch_size=int(args.batch_size),
                lr=float(args.lr),
                weight_decay=float(args.weight_decay),
                patience=int(args.patience),
                min_delta=float(args.min_delta),
                random_state=int(args.random_state),
                log_prefix=f"[{model_label} test-retrain]",
            )
            pred_te = test_predict(model, X_te.astype(np.float32), kind=args.model_kind, device=device, batch_size=int(args.batch_size))
            pred_te = pipe_y.inverse_transform(np.asarray(pred_te).reshape(-1, 1)).ravel()
            tm = evaluate_metrics(y_te, pred_te)
            log(f"Test metrics: rmse={tm['rmse']:.4f} r2={tm['r2']:.4f} pearson_r={tm['pearson_r']:.4f}")
            pd.DataFrame([tm]).to_csv(out_dir / f"test_metrics_{args.split_col}{out_tag}_torch_{args.model_kind}.csv", index=False)
            regression_scatter_plot(
                y_te,
                pred_te,
                title=f"{model_label} — {split_plot} held-out test",
                outpath=out_dir / "plots" / split_plot / model_label / "regression_test_holdout.png",
                use_hexbin=bool(args.use_hexbin),
            )

    log("Done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
