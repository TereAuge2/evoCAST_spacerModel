"""
CNN explainability utilities: per-layer activations + input attributions on one-hot DNA.

Designed for the saved torch checkpoint format produced by `training/train_torch.py`
and the consolidated retrain outputs (e.g. `final_models/CNN_best_model.pt`).
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch

from spacerModel.simple_modeling.train_models import build_feature_matrix, load_dataset
from spacerModel.models import to_channel_first
from spacerModel.training.train_torch import (
    _restore_standard_scaler_1d,
    load_torch_best_checkpoint,
)


_BASES = ["A", "C", "G", "T"]


@dataclass(frozen=True)
class ExplainExample:
    idx: int
    seq: str
    y_true: float
    y_pred: float


def _onehot_to_df(x4l: np.ndarray) -> pd.DataFrame:
    """(4, L) one-hot to logomaker-style DataFrame with columns A/C/G/T."""
    x = np.asarray(x4l, dtype=float)
    if x.shape[0] != 4:
        raise ValueError(f"Expected (4, L) one-hot; got {x.shape}")
    return pd.DataFrame(x.T, columns=_BASES)


def _attr_to_df(attr4l: np.ndarray) -> pd.DataFrame:
    """(4, L) attribution to logomaker-style DataFrame."""
    a = np.asarray(attr4l, dtype=float)
    if a.shape[0] != 4:
        raise ValueError(f"Expected (4, L) attributions; got {a.shape}")
    return pd.DataFrame(a.T, columns=_BASES)


def _plot_activation_heatmap(act: np.ndarray, outpath: Path, *, title: str) -> None:
    """
    act: (C, L) activation map.
    """
    a = np.asarray(act, dtype=float)
    fig, ax = plt.subplots(figsize=(18 / 2.54, 10 / 2.54))
    sns.heatmap(a, cmap="viridis", ax=ax, cbar=True)
    ax.set_xlabel("position")
    ax.set_ylabel("channel")
    ax.set_title(title)
    fig.tight_layout()
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, dpi=250, bbox_inches="tight")
    plt.close(fig)


def _plot_activation_heatmap_labeled(
    act: np.ndarray,
    outpath: Path,
    *,
    title: str,
    yticklabels: list[str] | None = None,
    ylabel: str = "channel",
) -> None:
    """Heatmap with optional y tick labels (e.g. A/C/G/T)."""
    a = np.asarray(act, dtype=float)
    fig, ax = plt.subplots(figsize=(18 / 2.54, 10 / 2.54))
    sns.heatmap(a, cmap="viridis", ax=ax, cbar=True)
    ax.set_xlabel("position")
    ax.set_ylabel(ylabel)
    if yticklabels is not None and len(yticklabels) == a.shape[0]:
        ax.set_yticks(np.arange(a.shape[0]) + 0.5)
        ax.set_yticklabels(yticklabels, rotation=0)
    ax.set_title(title)
    fig.tight_layout()
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, dpi=250, bbox_inches="tight")
    plt.close(fig)


def _plot_saliency_vs_top_filters(
    sal4l: np.ndarray,
    conv1: np.ndarray,
    conv2: np.ndarray,
    outpath: Path,
    *,
    title: str,
) -> None:
    """
    Connect input "importance" to detected conv features.

    Panels:
      1) saliency summed over bases
      2) conv1 top filter index by position (color strip) + max activation (line)
      3) conv2 top filter index by position (color strip) + max activation (line)
    """
    sal = np.asarray(sal4l, dtype=float)
    c1 = np.asarray(conv1, dtype=float)
    c2 = np.asarray(conv2, dtype=float)
    if sal.ndim != 2 or sal.shape[0] != 4:
        raise ValueError(f"Expected saliency (4, L); got {sal.shape}")
    L = int(sal.shape[1])
    if c1.ndim == 2 and c1.shape[1] > L:
        c1 = c1[:, :L]
    if c2.ndim == 2 and c2.shape[1] > L:
        c2 = c2[:, :L]

    sal_pos = sal.sum(axis=0)
    c1_max = c1.max(axis=0) if c1.size else np.zeros(L, dtype=float)
    c2_max = c2.max(axis=0) if c2.size else np.zeros(L, dtype=float)
    c1_arg = c1.argmax(axis=0) if c1.size else np.zeros(L, dtype=int)
    c2_arg = c2.argmax(axis=0) if c2.size else np.zeros(L, dtype=int)

    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(18 / 2.54, 14 / 2.54), sharex=True)
    ax0, ax1, ax2 = axes

    ax0.plot(np.arange(L), sal_pos, color="black", linewidth=1.6)
    ax0.set_ylabel("saliency\n(sum A/C/G/T)")
    ax0.set_title(title)
    sns.despine(ax=ax0)

    def _strip_and_line(ax: Any, arg: np.ndarray, mx: np.ndarray, *, label: str) -> None:
        strip = arg.reshape(1, -1)
        ax.imshow(strip, aspect="auto", interpolation="nearest", cmap="tab20")
        ax.set_yticks([])
        ax.set_ylabel(label)
        axr = ax.twinx()
        axr.plot(np.arange(L), mx, color="white", linewidth=1.2, alpha=0.9)
        axr.set_ylabel("max act.", color="white")
        axr.tick_params(axis="y", colors="white")
        sns.despine(ax=ax, left=True, bottom=True)
        sns.despine(ax=axr, left=False, right=True)

    _strip_and_line(ax1, c1_arg, c1_max, label="conv1\n(top filter)")
    _strip_and_line(ax2, c2_arg, c2_max, label="conv2\n(top filter)")
    ax2.set_xlabel("position")

    fig.tight_layout()
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, dpi=250, bbox_inches="tight")
    plt.close(fig)

def _plot_position_curve(y: np.ndarray, outpath: Path, *, title: str, ylabel: str) -> None:
    fig, ax = plt.subplots(figsize=(18 / 2.54, 6 / 2.54))
    ax.plot(np.arange(len(y)), y, color="black", linewidth=1.5)
    ax.set_xlabel("position")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    sns.despine(ax=ax)
    fig.tight_layout()
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, dpi=250, bbox_inches="tight")
    plt.close(fig)


def _plot_logo(df: pd.DataFrame, outpath: Path, *, title: str) -> None:
    import logomaker

    fig, ax = plt.subplots(figsize=(18 / 2.54, 5 / 2.54))
    logomaker.Logo(df, ax=ax, color_scheme="classic")
    ax.set_title(title)
    ax.set_xlabel("position")
    ax.set_ylabel("weight")
    sns.despine(ax=ax)
    fig.tight_layout()
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, dpi=250, bbox_inches="tight")
    plt.close(fig)


@torch.no_grad()
def _predict_with_pack(model: torch.nn.Module, pack: dict[str, Any], x_flat: np.ndarray, device: torch.device) -> float:
    """Scalar prediction on original y scale for one example."""
    sx = _restore_standard_scaler_1d(pack["scaler_x_mean"], pack["scaler_x_scale"])
    sy = _restore_standard_scaler_1d(pack["scaler_y_mean"], pack["scaler_y_scale"])
    xs = sx.transform(np.asarray(x_flat, dtype=np.float64).reshape(1, -1)).astype(np.float32)
    x4 = to_channel_first(xs).astype(np.float32)
    xt = torch.from_numpy(x4).to(device)
    pred_s = model(xt).detach().cpu().numpy().reshape(-1, 1)
    pred = sy.inverse_transform(pred_s).ravel()[0]
    return float(pred)


def _saliency_and_activations(
    model: torch.nn.Module,
    pack: dict[str, Any],
    x_flat: np.ndarray,
    *,
    device: torch.device,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """
    Returns:
      - saliency: (4, L) abs(grad) in original input space
      - activations: dict with keys conv1/conv2/pool and values (C, L') numpy arrays
    """
    sx = _restore_standard_scaler_1d(pack["scaler_x_mean"], pack["scaler_x_scale"])
    # Build differentiable scaling in torch: (x - mean) / scale
    mean = torch.from_numpy(np.asarray(sx.mean_, dtype=np.float32)).to(device)
    scale = torch.from_numpy(np.asarray(sx.scale_, dtype=np.float32)).to(device)

    x0 = np.asarray(x_flat, dtype=np.float32).reshape(1, -1)
    x = torch.from_numpy(x0).to(device).requires_grad_(True)  # original (unscaled) flat one-hot
    xs = (x - mean) / scale
    x4 = torch.from_numpy(to_channel_first(xs.detach().cpu().numpy()).astype(np.float32)).to(device)
    # Reconnect graph: reshape in torch (avoid detach)
    # xs: (1, 4*L) -> (1, L, 4) -> (1, 4, L)
    L = int(xs.shape[1] // 4)
    x4 = xs.reshape(1, L, 4).transpose(1, 2).contiguous()

    acts: dict[str, torch.Tensor] = {}

    def _hook(name: str):
        def fn(_m: Any, _inp: Any, out: Any) -> None:
            acts[name] = out.detach()

        return fn

    # CNN conv stack is `model.conv`: Conv1d -> ReLU -> Conv1d -> ReLU -> AdaptiveMaxPool1d
    h1 = model.conv[0].register_forward_hook(_hook("conv1"))
    h2 = model.conv[2].register_forward_hook(_hook("conv2"))
    h3 = model.conv[4].register_forward_hook(_hook("pool"))
    try:
        model.zero_grad(set_to_none=True)
        out = model(x4).sum()
        out.backward()
        grad = x.grad.detach().cpu().numpy().reshape(-1)  # (4*L,)
        sal = np.abs(grad).reshape(L, 4).T  # (4, L)
        # also compute grad*input (signed), useful for logos
        gxi = (x.detach().cpu().numpy().reshape(-1) * grad).reshape(L, 4).T
        # stash grad*input as extra entry
        activ_np = {
            "conv1": acts["conv1"].detach().cpu().numpy()[0],
            "conv2": acts["conv2"].detach().cpu().numpy()[0],
            "pool": acts["pool"].detach().cpu().numpy()[0],
            "grad_x_input": gxi,
        }
        return sal, activ_np
    finally:
        h1.remove()
        h2.remove()
        h3.remove()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="output/modeling/optimized_retrain/latest/final_models/CNN_best_model.pt")
    ap.add_argument("--excel", default="output/data_processing/251129_P8_120Complete_dataset_woCtrl_exogenous_with_splits.xlsx")
    ap.add_argument("--sheet", default=0, help="Sheet name or index (0-based)")
    ap.add_argument("--split-col", choices=["Split", "SplitRandom"], default="Split")
    ap.add_argument("--target-col", default="FE")
    ap.add_argument("--which", choices=["cv", "test", "all"], default="all")
    ap.add_argument("--n-per-group", type=int, default=10, help="How many sequences per selection group")
    ap.add_argument("--out-dir", default="output/modeling/cnn_explainability")
    ap.add_argument("--device", default="auto", help="cpu|cuda|auto")
    ap.add_argument(
        "--groups",
        default="random,top_fe,bottom_fe,worst_error",
        help="Comma-separated selection groups: random, top_fe, bottom_fe, worst_error",
    )
    args = ap.parse_args(argv)

    t0 = pd.Timestamp.now()
    def log(msg: str) -> None:
        dt = (pd.Timestamp.now() - t0).total_seconds()
        print(f"[{dt:7.1f}s] {msg}", flush=True)

    sheet: str | int
    if isinstance(args.sheet, int):
        sheet = int(args.sheet)
    else:
        s = str(args.sheet).strip()
        sheet = int(s) if s.lstrip("-").isdigit() else s

    dev = str(args.device).strip().lower()
    if dev == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif dev in ("cuda", "gpu"):
        device = torch.device("cuda")
    else:
        device = torch.device(dev)
    log(f"device={device}")

    log(f"Loading checkpoint: {args.checkpoint}")
    model, pack = load_torch_best_checkpoint(Path(args.checkpoint), device=device)

    log("Loading dataset + selecting rows...")
    ds = load_dataset(Path(args.excel), sheet=sheet, split_col=args.split_col, target_col=args.target_col)
    mask_cv = ds.df[args.split_col].astype(str).str.startswith("cv").to_numpy()
    if args.which == "cv":
        mask = mask_cv
    elif args.which == "test":
        mask = ~mask_cv
    else:
        mask = np.ones(len(ds.df), dtype=bool)

    df_sub = ds.df.loc[mask].reset_index(drop=False).rename(columns={"index": "orig_idx"})
    if df_sub.empty:
        raise SystemExit("No rows selected for explanation.")

    log(f"Building onehot features for n={len(df_sub)} rows...")
    X_sp = build_feature_matrix(df_sub, embedding="onehot", kmer_k=2)
    X = np.asarray(X_sp.toarray(), dtype=np.float64)  # raw one-hot
    y = df_sub[args.target_col].to_numpy(dtype=float)
    seqs = df_sub["SpacerSeq"].fillna("").astype(str).to_list()

    log("Computing predictions for selection...")
    preds = np.array([_predict_with_pack(model, pack, X[i], device) for i in range(X.shape[0])], dtype=float)
    err = np.abs(preds - y)

    rng = np.random.default_rng(0)
    groups = [g.strip().lower() for g in str(args.groups).split(",") if g.strip()]
    allowed = {"random", "top_fe", "bottom_fe", "worst_error"}
    bad = sorted(set(groups) - allowed)
    if bad:
        raise SystemExit(f"Unknown groups: {bad}. Allowed: {sorted(allowed)}")

    n_per = max(1, int(args.n_per_group))
    log(f"Selecting examples: groups={groups} n_per_group={n_per}")
    chosen: list[int] = []
    used: set[int] = set()

    def _take_from(order: np.ndarray, k: int) -> list[int]:
        out: list[int] = []
        for idx in order.tolist():
            if idx in used:
                continue
            out.append(int(idx))
            used.add(int(idx))
            if len(out) >= k:
                break
        return out

    group_assign: list[tuple[str, int]] = []
    for g in groups:
        if g == "random":
            pool = np.array([i for i in range(X.shape[0]) if i not in used], dtype=int)
            k = min(n_per, int(pool.size))
            if k > 0:
                pick = rng.choice(pool, size=k, replace=False)
                for idx in pick.tolist():
                    chosen.append(int(idx))
                    group_assign.append((g, int(idx)))
        elif g == "top_fe":
            order = np.argsort(-y)
            pick = _take_from(order, n_per)
            for idx in pick:
                chosen.append(int(idx))
                group_assign.append((g, int(idx)))
        elif g == "bottom_fe":
            order = np.argsort(y)
            pick = _take_from(order, n_per)
            for idx in pick:
                chosen.append(int(idx))
                group_assign.append((g, int(idx)))
        else:  # worst_error
            order = np.argsort(-err)
            pick = _take_from(order, n_per)
            for idx in pick:
                chosen.append(int(idx))
                group_assign.append((g, int(idx)))

    take = np.array(chosen[: min(len(chosen), n_per * len(groups))], dtype=int)
    log(f"Explaining n={len(take)} examples total")

    out_dir = Path(args.out_dir)
    run_dir = out_dir / Path(args.checkpoint).stem
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "examples").mkdir(parents=True, exist_ok=True)

    meta = {
        "checkpoint": str(args.checkpoint),
        "device": str(device),
        "n_per_group": int(n_per),
        "which": args.which,
        "groups": groups,
        "cnn_kwargs": pack.get("cnn_kwargs", {}),
    }
    (run_dir / "run_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    # Explain each example
    rows: list[dict[str, Any]] = []
    group_map = {idx: g for (g, idx) in group_assign}
    # Collect arrays per group for aggregate summaries.
    by_group: dict[str, dict[str, list[np.ndarray]]] = {}
    for j, i in enumerate(take, start=1):
        if j == 1 or j % 5 == 0 or j == len(take):
            log(f"Explaining example {j}/{len(take)} (row_idx={int(i)})")
        orig_idx = int(df_sub.loc[i, "orig_idx"])
        ex = ExplainExample(idx=orig_idx, seq=seqs[i], y_true=float(y[i]), y_pred=float(preds[i]))
        sal, acts = _saliency_and_activations(model, pack, X[i], device=device)

        grp = str(group_map.get(int(i), "unknown"))
        if grp not in by_group:
            by_group[grp] = {"sal": [], "gxi": [], "conv1": [], "conv2": [], "pool": []}
        by_group[grp]["sal"].append(np.asarray(sal, dtype=float))
        by_group[grp]["gxi"].append(np.asarray(acts["grad_x_input"], dtype=float))
        by_group[grp]["conv1"].append(np.asarray(acts["conv1"], dtype=float))
        by_group[grp]["conv2"].append(np.asarray(acts["conv2"], dtype=float))
        by_group[grp]["pool"].append(np.asarray(acts["pool"], dtype=float))
        ex_dir = run_dir / "examples" / f"{grp}_ex_{j:03d}_idx{orig_idx}"
        ex_dir.mkdir(parents=True, exist_ok=True)
        (ex_dir / "example.json").write_text(
            json.dumps({"orig_idx": ex.idx, "y_true": ex.y_true, "y_pred": ex.y_pred, "abs_err": float(abs(ex.y_pred - ex.y_true))}, indent=2),
            encoding="utf-8",
        )

        # Attribution plots
        _plot_activation_heatmap_labeled(
            sal,
            ex_dir / "saliency_abs_grad_heatmap.png",
            title="abs(∂y/∂x) (A,C,G,T × pos)",
            yticklabels=_BASES,
            ylabel="base",
        )
        _plot_position_curve(sal.sum(axis=0), ex_dir / "saliency_abs_grad_by_pos.png", title="abs(∂y/∂x) summed over bases", ylabel="saliency")

        # grad*input logos (signed; only the observed base at each position is non-zero)
        gxi = np.asarray(acts["grad_x_input"], dtype=float)
        _plot_logo(_attr_to_df(gxi), ex_dir / "logo_grad_x_input.png", title="grad × input (signed)")

        # Layer activations
        c1a = np.asarray(acts["conv1"], dtype=float)
        c2a = np.asarray(acts["conv2"], dtype=float)
        pa = np.asarray(acts["pool"], dtype=float)
        _plot_activation_heatmap_labeled(
            c1a,
            ex_dir / "conv1_activation.png",
            title="conv1 activation (filters × position)",
            yticklabels=[f"f{i}" for i in range(int(c1a.shape[0]))],
            ylabel="filter",
        )
        _plot_activation_heatmap_labeled(
            c2a,
            ex_dir / "conv2_activation.png",
            title="conv2 activation (filters × position)",
            yticklabels=[f"f{i}" for i in range(int(c2a.shape[0]))],
            ylabel="filter",
        )
        _plot_activation_heatmap_labeled(
            pa,
            ex_dir / "pool_activation.png",
            title="adaptive maxpool activation (filters × pooled_bin)",
            yticklabels=[f"f{i}" for i in range(int(pa.shape[0]))],
            ylabel="filter",
        )

        _plot_saliency_vs_top_filters(
            sal,
            c1a,
            c2a,
            ex_dir / "saliency_vs_top_filters.png",
            title=f"{grp} idx{orig_idx}: saliency vs top conv filters",
        )

        rows.append(
            {
                "group": grp,
                "orig_idx": ex.idx,
                "y_true": ex.y_true,
                "y_pred": ex.y_pred,
                "abs_err": float(abs(ex.y_pred - ex.y_true)),
                "seq_len": int(sal.shape[1]),
            }
        )

    pd.DataFrame(rows).to_csv(run_dir / "explained_examples.csv", index=False)

    # ---- Group-level aggregate summaries (reduces "too many plots" problem)
    log("Writing group-level summaries...")
    summ_dir = run_dir / "group_summaries"
    summ_dir.mkdir(parents=True, exist_ok=True)

    def _safe_stack_mean(arrs: list[np.ndarray]) -> np.ndarray:
        if not arrs:
            return np.empty((0,), dtype=float)
        # if lengths vary (shouldn't), truncate to min length
        shapes = [a.shape for a in arrs]
        min_shape = tuple(min(s[i] for s in shapes) for i in range(len(shapes[0])))
        trimmed = [a[tuple(slice(0, m) for m in min_shape)] for a in arrs]
        return np.mean(np.stack(trimmed, axis=0), axis=0)

    for grp, d in sorted(by_group.items()):
        if grp == "unknown":
            continue
        log(f"  summary: {grp} (n={len(d['sal'])})")
        gdir = summ_dir / grp
        gdir.mkdir(parents=True, exist_ok=True)
        sal_mean = _safe_stack_mean(d["sal"])  # (4, L)
        gxi_mean = _safe_stack_mean(d["gxi"])  # (4, L)
        conv1_mean = _safe_stack_mean(d["conv1"])  # (C1, L)
        conv2_mean = _safe_stack_mean(d["conv2"])  # (C2, L)
        pool_mean = _safe_stack_mean(d["pool"])  # (C2, T)

        # Save numeric summaries as CSV
        if sal_mean.size:
            _attr_to_df(sal_mean).to_csv(gdir / "saliency_abs_grad_mean.csv", index=False)
            _plot_activation_heatmap_labeled(
                sal_mean,
                gdir / "saliency_abs_grad_mean_heatmap.png",
                title=f"{grp}: mean abs(∂y/∂x)",
                yticklabels=_BASES,
                ylabel="base",
            )
            _plot_position_curve(
                sal_mean.sum(axis=0),
                gdir / "saliency_abs_grad_mean_by_pos.png",
                title=f"{grp}: mean abs(∂y/∂x) by position",
                ylabel="mean saliency",
            )
        if gxi_mean.size:
            _attr_to_df(gxi_mean).to_csv(gdir / "logo_grad_x_input_mean.csv", index=False)
            _plot_logo(_attr_to_df(gxi_mean), gdir / "logo_grad_x_input_mean.png", title=f"{grp}: mean grad×input")
        if conv1_mean.size:
            pd.DataFrame(conv1_mean).to_csv(gdir / "conv1_activation_mean.csv", index=False)
            _plot_activation_heatmap(conv1_mean, gdir / "conv1_activation_mean.png", title=f"{grp}: mean conv1 activation")
        if conv2_mean.size:
            pd.DataFrame(conv2_mean).to_csv(gdir / "conv2_activation_mean.csv", index=False)
            _plot_activation_heatmap(conv2_mean, gdir / "conv2_activation_mean.png", title=f"{grp}: mean conv2 activation")
        if pool_mean.size:
            pd.DataFrame(pool_mean).to_csv(gdir / "pool_activation_mean.csv", index=False)
            _plot_activation_heatmap(pool_mean, gdir / "pool_activation_mean.png", title=f"{grp}: mean pool activation")

    log(f"Done. Outputs in {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

