"""
Train an XGBoost surrogate on synthetic spacers labeled by a saved CNN checkpoint.

The CNN predicts log2-FE (scaled like training). XGBoost is fit on all synthetic rows
with aggressive capacity to approximate the CNN; feature importances are reshaped to
per-position and per-position×nucleotide tables and plots (same one-hot layout as
``linear_position_weights``).

By default, also evaluates on **measured CV rows** from ``--excel``: CNN vs surrogate vs
measured FE (metrics + scatter plots), to see how the surrogate tracks the teacher off
the synthetic training distribution.

For **gain** importance only, also writes heatmaps (raw + within-position share), a
top-features bar chart, a cumulative-mass curve, ``gain_feature_ranking.csv``, and
**sequence logos** of **within-position gain share** (letter height = fraction of that
position’s total gain; not biological base frequency).

**Spacer length:** this script only supports **32 bp** spacers (128 one-hot features). The
checkpoint must match; real-data rows must have ``SpacerSeq`` of length 32.

**Synthetic sequences:** each training row is **i.i.d.**; at every position the base is
drawn **uniformly** from ``{A,C,G,T}``, independently of other positions and other rows
(uniform prior over ``4^32`` sequences). Empirical position frequencies and sequence logos
confirm that prior.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor

from spacerModel.interpretation.linear_position_weights import (
    coefficient_matrix_from_flat,
    one_hot_encode_dna,
    plot_abs_weight_sum,
    plot_weights_by_position,
    savefig,
    set_plot_style,
)
from spacerModel.simple_modeling.train_models import regression_scatter_plot
from spacerModel.training.train_torch import (
    load_torch_best_checkpoint,
    predict_numpy_from_checkpoint,
    resolve_device,
)

# Project convention: CNN / one-hot spacers are 32 nt (128 flat features).
SPACER_LEN = 32

# Sequence logos: tick labels 1…L (slightly smaller than default body font).
_LOGO_POSITION_TICK_FONTSIZE = 7


def _style_logo_position_axis(ax, npos: int, *, xlabel: str = "Position") -> None:
    ax.set_xlabel(xlabel)
    if npos <= 0:
        return
    ax.set_xticks(np.arange(npos, dtype=float))
    ax.set_xticklabels([str(i + 1) for i in range(npos)], fontsize=_LOGO_POSITION_TICK_FONTSIZE)
    ax.set_xlim(-0.5, npos - 0.5)


def _random_onehot(n: int, seq_len: int, rng: np.random.Generator) -> np.ndarray:
    """Dense one-hot (N, 4*L) with one 1 per position, ACGT order per position."""
    a = 4
    x = np.zeros((n, seq_len * a), dtype=np.float32)
    pick = rng.integers(0, a, size=(n, seq_len))
    r = np.arange(n, dtype=np.int64)
    for pos in range(seq_len):
        x[r, pos * a + pick[:, pos]] = 1.0
    return x


def _dense_onehot_to_position_frequencies(x: np.ndarray, seq_len: int) -> pd.DataFrame:
    """Empirical P(base|position) from dense one-hot (n, 4*L); rows indexed 1…L."""
    x = np.asarray(x, dtype=np.float64)
    n, d = x.shape
    a = 4
    if d != seq_len * a:
        raise ValueError(f"Expected width {seq_len * a}, got {d}")
    xh = x.reshape(n, seq_len, a)
    freq = xh.mean(axis=0)
    df = pd.DataFrame(freq, columns=list("ACGT"))
    df.index = np.arange(1, seq_len + 1)
    df.index.name = "position"
    return df


def _plot_synthetic_sequence_logos(freq_df: pd.DataFrame, out_dir: Path) -> None:
    """Sequence logos for the synthetic training distribution (expect ~uniform / ~0 bits)."""
    import logomaker

    log = logging.getLogger(__name__)
    set_plot_style()
    df = freq_df[["A", "C", "G", "T"]].copy().astype(float)
    df_plot = df.reset_index(drop=True)
    npos = int(len(df_plot))

    plt.figure(figsize=(18.0 / 2.54, 5.5 / 2.54))
    ax = plt.gca()
    logomaker.Logo(df_plot, ax=ax, color_scheme="classic")
    ax.set_title("Synthetic training data: per-position base frequencies (uniform random)")
    ax.set_ylabel("Probability")
    ax.set_ylim(0.0, 1.0)
    _style_logo_position_axis(ax, npos)
    savefig(out_dir / "synthetic_sequence_logo_probability.png")

    info_df = logomaker.transform_matrix(df_plot, from_type="probability", to_type="information")
    plt.figure(figsize=(18.0 / 2.54, 5.5 / 2.54))
    ax = plt.gca()
    logomaker.Logo(info_df, ax=ax, color_scheme="classic")
    ymax = float(np.nanmax(info_df.to_numpy())) if info_df.size else 0.0
    ax.set_ylim(0.0, max(0.04, ymax * 1.35))
    ax.set_title("Synthetic training data: information vs uniform background (~0 bits if uniform)")
    ax.set_ylabel("Bits")
    _style_logo_position_axis(ax, npos)
    savefig(out_dir / "synthetic_sequence_logo_information_bits.png")
    log.info(
        "Wrote synthetic_sequence_logo_probability.png and synthetic_sequence_logo_information_bits.png"
    )


def _importance_full_vector(model: XGBRegressor, n_features: int, *, importance_type: str) -> np.ndarray:
    """Map booster get_score (sparse f*) to a dense length n_features vector."""
    booster = model.get_booster()
    scores = booster.get_score(importance_type=importance_type)
    vec = np.zeros(n_features, dtype=np.float64)
    for fname, val in scores.items():
        if not str(fname).startswith("f"):
            continue
        j = int(str(fname)[1:])
        if 0 <= j < n_features:
            vec[j] = float(val)
    return vec


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    err = y_pred - y_true
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(np.mean(np.abs(err))),
        "r2": float(r2_score(y_true, y_pred)),
        "n": int(y_true.size),
    }


def _plot_gain_feature_extras(imp_gain: np.ndarray, seq_len: int, out_dir: Path) -> None:
    """
    Extra views of one-hot *features* (position × A/C/G/T): heatmaps, ranking, cumulative mass.
    """
    vec = np.asarray(imp_gain, dtype=np.float64).ravel()
    n_feat = int(vec.size)
    bases = list("ACGT")
    rows: list[dict[str, object]] = []
    for j in range(n_feat):
        pos = j // 4 + 1
        base = bases[j % 4]
        rows.append({"feature_idx": j, "position": pos, "base": base, "label": f"pos {pos} · {base}", "gain": vec[j]})
    df_rank = pd.DataFrame(rows).sort_values("gain", ascending=False, kind="mergesort")
    df_rank.to_csv(out_dir / "gain_feature_ranking.csv", index=False)

    df_pos = coefficient_matrix_from_flat(vec, seq_len=seq_len, alphabet="ACGT")
    mat = df_pos[["A", "C", "G", "T"]].to_numpy(dtype=float)
    row_sum = mat.sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        share = np.divide(mat, row_sum[:, None], out=np.zeros_like(mat), where=row_sum[:, None] > 0)

    set_plot_style()
    pos_labels = [str(i) for i in range(1, seq_len + 1)]
    # Wide heatmap: rows = bases, cols = positions (read left→right as 5′→3′ spacer order).
    w_in = max(6.5, min(22.0, 0.22 * seq_len + 2.5))
    h_in = 4.2

    plt.figure(figsize=(w_in, h_in))
    ax = sns.heatmap(
        mat.T,
        xticklabels=pos_labels if seq_len <= 40 else False,
        yticklabels=bases,
        cmap="viridis",
        cbar_kws={"label": "Gain"},
    )
    ax.set_xlabel(f"Position (1 … {seq_len})")
    ax.set_ylabel("Nucleotide one-hot feature")
    ax.set_title("XGBoost surrogate: gain importance (position × base)")
    savefig(out_dir / "gain_importance_heatmap_position_x_base.png")

    plt.figure(figsize=(w_in, h_in))
    ax = sns.heatmap(
        share.T,
        xticklabels=pos_labels if seq_len <= 40 else False,
        yticklabels=bases,
        cmap="rocket_r",
        vmin=0.0,
        vmax=1.0,
        cbar_kws={"label": "Share of position total"},
    )
    ax.set_xlabel(f"Position (1 … {seq_len})")
    ax.set_ylabel("Nucleotide one-hot feature")
    ax.set_title("Within-position share of gain (each column sums to 1)")
    savefig(out_dir / "gain_importance_within_position_share_heatmap.png")

    _plot_gain_sequence_logos_from_share(share, seq_len, out_dir)

    top_k = min(32, n_feat)
    top = df_rank.head(top_k)
    plt.figure(figsize=(10.0 / 2.54, max(8.0, 0.35 * top_k) / 2.54))
    order = np.arange(top_k)
    plt.barh(order[::-1], top["gain"].to_numpy()[::-1], color="steelblue", alpha=0.85)
    plt.yticks(order[::-1], top["label"].to_numpy()[::-1], fontsize=8)
    plt.xlabel("Gain importance")
    plt.title(f"Top {top_k} one-hot features (by gain)")
    plt.tight_layout()
    savefig(out_dir / "gain_top_features_horizontal_bar.png")

    s = np.sort(vec)[::-1]
    total = float(np.sum(s))
    cum = np.cumsum(s) / total if total > 0 else np.zeros_like(s)
    plt.figure(figsize=(16.0 / 2.54, 10.0 / 2.54))
    x = np.arange(1, len(cum) + 1)
    plt.plot(x, cum, color="black", linewidth=1.2)
    for q in (0.5, 0.9):
        plt.axhline(q, color="gray", linestyle="--", linewidth=0.75, alpha=0.65)
    if len(cum) > 0 and total > 0:
        k90 = int(np.searchsorted(cum, 0.9, side="left")) + 1
        k90 = min(k90, len(cum))
        plt.scatter([k90], [cum[k90 - 1]], color="crimson", s=28, zorder=5, label=f"90% at rank {k90}")
        plt.legend(loc="lower right", fontsize=9)
    plt.xlabel("Ranked one-hot feature (by gain)")
    plt.ylabel("Cumulative fraction of total gain")
    plt.title(f"How concentrated gain is across the {n_feat} one-hot features (32 bp × 4)")
    plt.xlim(1, len(cum))
    plt.ylim(0, 1.02)
    plt.tight_layout()
    savefig(out_dir / "gain_cumulative_importance_curve.png")


def _plot_gain_sequence_logos_from_share(share: np.ndarray, seq_len: int, out_dir: Path) -> None:
    """
    Sequence-logo views of **gain**, using **within-position shares** (rows sum to 1).

    This is the logo analogue of ``gain_importance_within_position_share_heatmap``: letter
    height shows which one-hot dimension (base at that position) carries the largest fraction
    of total gain *at that position*, not P(base) in sequences.
    """
    import logomaker

    log = logging.getLogger(__name__)
    share = np.asarray(share, dtype=np.float64)
    if share.shape != (seq_len, 4):
        raise ValueError(f"Expected share shape ({seq_len}, 4), got {share.shape}")
    df_plot = pd.DataFrame(share, columns=list("ACGT")).reset_index(drop=True)
    npos = int(len(df_plot))

    set_plot_style()
    plt.figure(figsize=(18.0 / 2.54, 5.5 / 2.54))
    ax = plt.gca()
    logomaker.Logo(df_plot, ax=ax, color_scheme="classic")
    ax.set_title(
        "XGBoost gain: within-position share (height = fraction of total gain at that position)"
    )
    ax.set_xlabel("Position (1-based)")
    ax.set_ylabel("Fraction of position gain")
    ax.set_ylim(0.0, 1.0)
    _style_logo_position_axis(ax, npos)
    savefig(out_dir / "gain_sequence_logo_within_position_share.png")

    info_df = logomaker.transform_matrix(df_plot, from_type="probability", to_type="information")
    plt.figure(figsize=(18.0 / 2.54, 5.5 / 2.54))
    ax = plt.gca()
    logomaker.Logo(info_df, ax=ax, color_scheme="classic")
    ymax = float(np.nanmax(info_df.to_numpy())) if info_df.size else 0.0
    ax.set_ylim(0.0, max(0.08, ymax * 1.25))
    ax.set_title(
        "XGBoost gain: information of within-position share vs uniform (max 2 bits if one base dominates)"
    )
    ax.set_ylabel("Bits")
    _style_logo_position_axis(ax, npos)
    savefig(out_dir / "gain_sequence_logo_within_position_share_bits.png")
    log.info(
        "Wrote gain_sequence_logo_within_position_share.png and gain_sequence_logo_within_position_share_bits.png"
    )


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    if a.size < 2 or np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _parse_excel_sheet(value: str) -> int | str:
    """
    For ``pd.read_excel(..., sheet_name=...)``: numeric strings become 0-based **indices**;
    other strings are worksheet **names**. (Plain ``argparse`` leaves ``--sheet 0`` as str ``"0"``,
    which wrongly selects a sheet *named* ``"0"``.)
    """
    s = value.strip()
    if s.isdigit():
        return int(s)
    return s


def _real_data_cv_frame(
    excel: Path,
    *,
    sheet: str | int,
    target_col: str,
    split_col: str,
    seq_len: int,
) -> pd.DataFrame:
    """CV rows with valid target and spacer length exactly ``seq_len`` (32 bp in this pipeline)."""
    df = pd.read_excel(excel, sheet_name=sheet, engine="openpyxl")
    needed = {"SpacerSeq", target_col, split_col}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    d = df.copy()
    d[target_col] = pd.to_numeric(d[target_col], errors="coerce")
    d = d[d[target_col].notna() & d["SpacerSeq"].notna()].reset_index(drop=True)
    cv_mask = d[split_col].astype(str).str.startswith("cv")
    d = d.loc[cv_mask].reset_index(drop=True)
    d["_L"] = d["SpacerSeq"].astype(str).str.upper().str.strip().str.len()
    d = d.loc[d["_L"] == int(seq_len)].drop(columns=["_L"]).reset_index(drop=True)
    return d


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="XGBoost surrogate on random 32 bp spacers labeled by a CNN checkpoint (n_features must be 128)."
    )
    ap.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("models/CNN_best_model.pt"),
        help="CNN checkpoint from save_torch_best_checkpoint (default: models/CNN_best_model.pt).",
    )
    ap.add_argument("--n-synthetic", type=int, default=100_000, help="Number of random spacers.")
    ap.add_argument("--random-state", type=int, default=42)
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/interpretation/cnn_surrogate_xgboost"),
    )
    ap.add_argument("--device", default="auto", help="auto | cpu | cuda")
    ap.add_argument("--batch-size", type=int, default=512, help="Batch size for CNN forward.")
    ap.add_argument("--xgb-n-estimators", type=int, default=3000)
    ap.add_argument("--xgb-max-depth", type=int, default=12)
    ap.add_argument("--xgb-learning-rate", type=float, default=0.03)
    ap.add_argument("--xgb-reg-lambda", type=float, default=0.0, help="L2 on leaves (0 = strongest overfit).")
    ap.add_argument("--xgb-reg-alpha", type=float, default=0.0, help="L1 on leaves.")
    ap.add_argument("--xgb-min-child-weight", type=float, default=1.0)
    ap.add_argument("--xgb-subsample", type=float, default=1.0)
    ap.add_argument("--xgb-colsample-bytree", type=float, default=1.0)
    ap.add_argument("--xgb-n-jobs", type=int, default=-1)
    ap.add_argument(
        "--sample-rows-csv",
        type=int,
        default=2000,
        help="Write a CSV sample of this many synthetic rows (0 to skip).",
    )
    ap.add_argument(
        "--no-real-eval",
        action="store_true",
        help="Skip evaluation on measured CV rows from --excel (CNN vs XGB vs FE).",
    )
    ap.add_argument(
        "--excel",
        type=Path,
        default=Path(
            "output/data_processing/251129_P8_120Complete_dataset_woCtrl_exogenous_with_splits.xlsx"
        ),
        help="Dataset with SpacerSeq, target, and split (CV rows used for real-data check).",
    )
    ap.add_argument(
        "--sheet",
        default="0",
        type=_parse_excel_sheet,
        help="Sheet index (e.g. 0) or worksheet name for pandas.read_excel.",
    )
    ap.add_argument("--target-col", default="FE")
    ap.add_argument("--split-col", default="Split")
    ap.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        help="Logging verbosity (default: INFO).",
    )
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, str(args.log_level)),
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
        force=True,
    )
    log = logging.getLogger(__name__)

    if args.n_synthetic < 100:
        raise ValueError("--n-synthetic should be large (e.g. >= 100).")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()

    log.info("Starting CNN → XGBoost surrogate run")
    log.info("Output directory: %s", out_dir.resolve())
    log.info(
        "Config: checkpoint=%s | n_synthetic=%s | device=%s | batch_size=%s | xgb_jobs=%s",
        Path(args.checkpoint).resolve(),
        args.n_synthetic,
        args.device,
        args.batch_size,
        args.xgb_n_jobs,
    )
    log.info(
        "XGBoost: n_estimators=%s max_depth=%s lr=%s reg_lambda=%s",
        args.xgb_n_estimators,
        args.xgb_max_depth,
        args.xgb_learning_rate,
        args.xgb_reg_lambda,
    )

    rng = np.random.default_rng(int(args.random_state))
    device = resolve_device(str(args.device))
    log.info("Resolved torch device: %s", device)

    t_ckpt = time.perf_counter()
    model, pack = load_torch_best_checkpoint(Path(args.checkpoint), device=device)
    log.info("Loaded checkpoint in %.2fs", time.perf_counter() - t_ckpt)
    log.info("Checkpoint model_kind=%s n_features=%s", pack.get("model_kind"), pack.get("n_features"))
    if pack.get("model_kind") != "cnn":
        raise ValueError(
            f"This script expects a CNN checkpoint; got model_kind={pack.get('model_kind')!r}."
        )
    n_feat = int(pack["n_features"])
    expected_feat = SPACER_LEN * 4
    if n_feat != expected_feat:
        raise ValueError(
            f"Checkpoint n_features={n_feat}; this script only supports {SPACER_LEN} bp "
            f"spacers ({expected_feat} one-hot dimensions). Use a 32-mer CNN checkpoint."
        )
    seq_len = SPACER_LEN

    # 1) Synthetic one-hot + CNN teacher labels (original y scale: log2-FE).
    log.info("Generating %s random %s-bp one-hot spacers…", f"{args.n_synthetic:,}", seq_len)
    t_syn = time.perf_counter()
    x_syn = _random_onehot(int(args.n_synthetic), seq_len, rng)
    log.info("One-hot array ready: shape=%s (%.2fs)", x_syn.shape, time.perf_counter() - t_syn)

    freq_df = _dense_onehot_to_position_frequencies(x_syn, seq_len)
    freq_csv = out_dir / "synthetic_position_frequencies.csv"
    freq_df.to_csv(freq_csv, encoding="utf-8")
    log.info("Wrote %s (empirical P(base|position) over synthetic draws)", freq_csv)
    _plot_synthetic_sequence_logos(freq_df, out_dir)

    t_cnn = time.perf_counter()
    log.info("Running CNN teacher forward on synthetic batch (batch_size=%s)…", args.batch_size)
    y_syn = predict_numpy_from_checkpoint(
        pack,
        model,
        x_syn,
        kind="cnn",
        device=device,
        batch_size=int(args.batch_size),
    )
    y_syn = np.asarray(y_syn, dtype=np.float64).ravel()
    log.info(
        "CNN teacher predictions: n=%s | mean=%.6f | std=%.6f (%.2fs)",
        y_syn.size,
        float(np.mean(y_syn)),
        float(np.std(y_syn)),
        time.perf_counter() - t_cnn,
    )

    metrics_cnn_only: dict[str, object] = {
        "n_synthetic": int(args.n_synthetic),
        "seq_len": int(seq_len),
        "synthetic_prior": (
            "Each spacer row is i.i.d.: at every position the base is drawn uniformly from "
            "{A,C,G,T}, independent of other positions and other rows (uniform over 4^32 sequences)."
        ),
        "y_teacher_mean": float(np.mean(y_syn)),
        "y_teacher_std": float(np.std(y_syn)),
    }
    (out_dir / "synthetic_teacher_summary.json").write_text(
        json.dumps(metrics_cnn_only, indent=2), encoding="utf-8"
    )
    log.info("Wrote %s", out_dir / "synthetic_teacher_summary.json")

    if int(args.sample_rows_csv) > 0:
        m = min(int(args.sample_rows_csv), x_syn.shape[0])
        # Decode argmax base per position for readability.
        a = 4
        bases = np.array(list("ACGT"))
        idx = np.argmax(x_syn[:m].reshape(m, seq_len, a), axis=2)
        seqs = np.array(["".join(bases[idx[i]]) for i in range(m)])
        pd.DataFrame(
            {
                "spacer_seq": seqs,
                "y_cnn_teacher": y_syn[:m],
            }
        ).to_csv(out_dir / "synthetic_sample.csv", index=False)
        log.info("Wrote synthetic sample CSV (%s rows)", m)

    # 2) XGBoost surrogate (intentionally high capacity; evaluated on training set).
    log.info("Fitting XGBoost surrogate (n=%s features=%s)…", f"{args.n_synthetic:,}", n_feat)
    t_xgb = time.perf_counter()
    xgb = XGBRegressor(
        objective="reg:squarederror",
        n_estimators=int(args.xgb_n_estimators),
        max_depth=int(args.xgb_max_depth),
        learning_rate=float(args.xgb_learning_rate),
        reg_lambda=float(args.xgb_reg_lambda),
        reg_alpha=float(args.xgb_reg_alpha),
        min_child_weight=float(args.xgb_min_child_weight),
        subsample=float(args.xgb_subsample),
        colsample_bytree=float(args.xgb_colsample_bytree),
        random_state=int(args.random_state),
        n_jobs=int(args.xgb_n_jobs),
        tree_method="hist",
    )
    xgb.fit(x_syn, y_syn)
    log.info("XGBoost fit finished in %.2fs", time.perf_counter() - t_xgb)

    t_hat = time.perf_counter()
    y_hat = np.asarray(xgb.predict(x_syn), dtype=np.float64).ravel()
    log.info("XGBoost predict (train) in %.2fs", time.perf_counter() - t_hat)

    fit_metrics = _metrics(y_syn, y_hat)
    (out_dir / "xgb_surrogate_fit_metrics.json").write_text(
        json.dumps(fit_metrics, indent=2), encoding="utf-8"
    )
    log.info(
        "Surrogate vs CNN teacher (same synthetic data): RMSE=%.6f MAE=%.6f R²=%.6f | wrote %s",
        fit_metrics["rmse"],
        fit_metrics["mae"],
        fit_metrics["r2"],
        out_dir / "xgb_surrogate_fit_metrics.json",
    )

    syn_hex = int(y_syn.size) > 20_000
    regression_scatter_plot(
        y_syn,
        y_hat,
        title="Synthetic training: XGB surrogate vs CNN teacher",
        xlab="log2-FE (CNN)",
        ylab="log2-FE (XGB surrogate)",
        outpath=out_dir / "synthetic_eval_scatter_xgb_vs_cnn.png",
        use_hexbin=syn_hex,
    )
    log.info("Wrote %s (hexbin=%s)", out_dir / "synthetic_eval_scatter_xgb_vs_cnn.png", syn_hex)

    # 2b) Real CV rows: CNN vs XGB surrogate vs measured FE (generalization check).
    if not args.no_real_eval:
        log.info("Real-data evaluation: excel=%s", Path(args.excel).resolve())
        ex = Path(args.excel)
        if not ex.is_file():
            log.warning("Excel not found; skipping real CV eval (see real_data_eval_skipped.txt)")
            (out_dir / "real_data_eval_skipped.txt").write_text(
                f"Real-data eval skipped: file not found\n  {ex.resolve()}\n"
                "Pass a valid --excel path or use --no-real-eval.\n",
                encoding="utf-8",
            )
        else:
            d = _real_data_cv_frame(
                ex,
                sheet=args.sheet,
                target_col=str(args.target_col),
                split_col=str(args.split_col),
                seq_len=seq_len,
            )
            if d.empty:
                log.warning(
                    "No CV rows after filters (cv split, %s, length %s); skipping real eval",
                    args.target_col,
                    seq_len,
                )
                (out_dir / "real_data_eval_skipped.txt").write_text(
                    "Real-data eval skipped: no CV rows after filtering "
                    f"(need split prefix 'cv', valid {args.target_col}, SpacerSeq length {seq_len}).\n",
                    encoding="utf-8",
                )
            else:
                log.info("Real CV subset: %s rows (sheet=%s target=%s)", len(d), args.sheet, args.target_col)
                x_sp, l_enc = one_hot_encode_dna(d["SpacerSeq"])
                if l_enc != seq_len:
                    raise ValueError(
                        f"Encoded length L={l_enc} != checkpoint seq_len={seq_len} "
                        "(check SpacerSeq formatting)."
                    )
                x_real = x_sp.toarray().astype(np.float32)
                if x_real.shape[1] != n_feat:
                    raise ValueError(
                        f"One-hot width {x_real.shape[1]} != checkpoint n_features={n_feat}."
                    )
                y_true = d[str(args.target_col)].to_numpy(dtype=np.float64)
                t_re = time.perf_counter()
                log.info("CNN forward on real CV rows…")
                y_cnn = predict_numpy_from_checkpoint(
                    pack,
                    model,
                    x_real,
                    kind="cnn",
                    device=device,
                    batch_size=int(args.batch_size),
                )
                y_cnn = np.asarray(y_cnn, dtype=np.float64).ravel()
                log.info("XGBoost predict on real CV rows…")
                y_xgb = np.asarray(xgb.predict(x_real), dtype=np.float64).ravel()
                log.info("Real CV predictions done in %.2fs total", time.perf_counter() - t_re)

                real_summary: dict[str, object] = {
                    "n_rows": int(d.shape[0]),
                    "excel": str(ex.resolve()),
                    "split_filter": "cv prefix",
                    "spacer_length_filter": int(seq_len),
                    "cnn_vs_measured": _metrics(y_true, y_cnn),
                    "xgb_surrogate_vs_measured": _metrics(y_true, y_xgb),
                    "xgb_surrogate_vs_cnn": _metrics(y_cnn, y_xgb),
                    "pearson_xgb_vs_cnn": _pearson(y_cnn, y_xgb),
                    "pearson_cnn_vs_measured": _pearson(y_true, y_cnn),
                    "pearson_xgb_vs_measured": _pearson(y_true, y_xgb),
                }
                metrics_path = out_dir / "real_data_cv_eval_metrics.json"
                metrics_path.write_text(json.dumps(real_summary, indent=2), encoding="utf-8")
                log.info("Wrote %s", metrics_path)
                pd.DataFrame(
                    {
                        str(args.target_col): y_true,
                        "y_pred_cnn": y_cnn,
                        "y_pred_xgb_surrogate": y_xgb,
                        "xgb_minus_cnn": y_xgb - y_cnn,
                    }
                ).to_csv(out_dir / "real_data_cv_predictions.csv", index=False)
                log.info("Wrote %s", out_dir / "real_data_cv_predictions.csv")

                cvm = real_summary["cnn_vs_measured"]
                xvm = real_summary["xgb_surrogate_vs_measured"]
                xvc = real_summary["xgb_surrogate_vs_cnn"]
                log.info(
                    "Real CV metrics — CNN vs measured: RMSE=%.6f R²=%.6f | "
                    "XGB vs measured: RMSE=%.6f R²=%.6f | XGB vs CNN: RMSE=%.6f | "
                    "Pearson(XGB,CNN)=%.4f",
                    float(cvm["rmse"]),
                    float(cvm["r2"]),
                    float(xvm["rmse"]),
                    float(xvm["r2"]),
                    float(xvc["rmse"]),
                    float(real_summary["pearson_xgb_vs_cnn"]),
                )

                log.info("Writing real CV scatter plots…")
                regression_scatter_plot(
                    y_true,
                    y_cnn,
                    title="Real CV rows: CNN vs measured",
                    outpath=out_dir / "real_eval_scatter_cnn_vs_measured.png",
                )
                regression_scatter_plot(
                    y_true,
                    y_xgb,
                    title="Real CV rows: XGB surrogate vs measured",
                    outpath=out_dir / "real_eval_scatter_xgb_vs_measured.png",
                )
                regression_scatter_plot(
                    y_cnn,
                    y_xgb,
                    title="Real CV rows: XGB surrogate vs CNN",
                    xlab="log2-FE (CNN)",
                    ylab="log2-FE (XGB surrogate)",
                    outpath=out_dir / "real_eval_scatter_xgb_vs_cnn.png",
                )
                log.info("Real CV scatter plots done")
    else:
        log.info("Skipping real-data eval (--no-real-eval)")

    # 3) Importances: gain + cover + sklearn feature_importances_ (weight frequency).
    log.info("Computing feature importances and writing interpretation artifacts…")
    imp_gain = _importance_full_vector(xgb, n_feat, importance_type="gain")
    imp_cover = _importance_full_vector(xgb, n_feat, importance_type="cover")
    imp_weight = np.asarray(xgb.feature_importances_, dtype=np.float64).ravel()
    if imp_weight.size != n_feat:
        raise RuntimeError("Unexpected feature_importances_ length.")

    def _save_and_plot(vec: np.ndarray, stem: str, label: str) -> None:
        log.debug("Saving %s importance tables and plots…", stem)
        pd.DataFrame({"feature_idx": np.arange(vec.size), "importance": vec}).to_csv(
            out_dir / f"{stem}_importance_full.csv", index=False
        )
        df_pos = coefficient_matrix_from_flat(vec, seq_len=seq_len, alphabet="ACGT")
        df_pos.to_csv(out_dir / f"{stem}_importance_by_position_base.csv", index=False)
        pos_mean = df_pos[["A", "C", "G", "T"]].mean(axis=1).to_numpy(dtype=float)
        pos_total = df_pos[["A", "C", "G", "T"]].sum(axis=1).to_numpy(dtype=float)
        pd.DataFrame(
            {
                "position": df_pos["position"],
                "mean_importance": pos_mean,
                "total_importance": pos_total,
            }
        ).to_csv(out_dir / f"{stem}_importance_position_aggregate.csv", index=False)

        suffix = f" ({label})"
        plot_weights_by_position(
            df_pos,
            out_dir / f"{stem}_importance_by_position.png",
            title=f"XGBoost surrogate {label} by position/base{suffix}",
        )
        plot_abs_weight_sum(
            df_pos,
            out_dir / f"{stem}_abs_importance_sum_by_position.png",
            title=f"XGBoost surrogate sum of importances by position{suffix}",
        )

    _save_and_plot(imp_gain, "gain", "gain importance")
    _save_and_plot(imp_cover, "cover", "cover importance")
    _save_and_plot(imp_weight, "weight", "split weight (sklearn)")
    log.info("Wrote gain / cover / weight importance (CSVs + PNGs for each type)")
    t_extra = time.perf_counter()
    _plot_gain_feature_extras(imp_gain, seq_len, out_dir)
    log.info("Gain extra plots (heatmap, ranking, cumulative) in %.2fs", time.perf_counter() - t_extra)

    meta = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "n_synthetic": int(args.n_synthetic),
        "seq_len": int(seq_len),
        "random_state": int(args.random_state),
        "real_data_eval": (not bool(args.no_real_eval)),
        "real_data_excel": str(Path(args.excel).resolve()) if not args.no_real_eval else None,
        "xgb_params": {
            "n_estimators": int(args.xgb_n_estimators),
            "max_depth": int(args.xgb_max_depth),
            "learning_rate": float(args.xgb_learning_rate),
            "reg_lambda": float(args.xgb_reg_lambda),
            "reg_alpha": float(args.xgb_reg_alpha),
            "min_child_weight": float(args.xgb_min_child_weight),
            "subsample": float(args.xgb_subsample),
            "colsample_bytree": float(args.xgb_colsample_bytree),
        },
    }
    (out_dir / "run_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    log.info("Wrote run_metadata.json")
    log.info("Done. Total wall time %.2fs | outputs under %s", time.perf_counter() - t0, out_dir.resolve())

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
