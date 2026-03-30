from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import sparse
from sklearn.linear_model import ElasticNet, Lasso, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def set_plot_style() -> None:
    mpl.rcParams.update(mpl.rcParamsDefault)
    # Prefer Arial if available; otherwise use a sane sans fallback.
    try:
        from matplotlib import font_manager as fm

        _ = fm.findfont("Arial", fallback_to_default=False)
        mpl.rcParams["font.family"] = "Arial"
    except Exception:
        mpl.rcParams["font.family"] = "DejaVu Sans"
    mpl.rcParams["font.size"] = 14


def savefig(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sns.despine()
    plt.tight_layout()
    plt.grid(False)
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def one_hot_encode_dna(
    seqs: pd.Series, *, alphabet: str = "ACGT", pad_char: str = "N"
) -> tuple[sparse.csr_matrix, int]:
    seq_list = ["" if pd.isna(x) else str(x) for x in seqs]
    seq_list = [s.upper().strip() for s in seq_list]
    n = len(seq_list)
    L = max((len(s) for s in seq_list), default=0)
    if n == 0 or L == 0:
        return sparse.csr_matrix((n, 0), dtype=np.uint8), L

    alpha = list(alphabet)
    alpha_to_idx = {c: i for i, c in enumerate(alpha)}
    A = len(alpha)

    rows: list[int] = []
    cols: list[int] = []
    data: list[int] = []

    for r, s in enumerate(seq_list):
        if len(s) < L:
            s = s + (pad_char * (L - len(s)))
        for pos, ch in enumerate(s[:L]):
            j = alpha_to_idx.get(ch)
            if j is None:
                continue
            rows.append(r)
            cols.append(pos * A + j)
            data.append(1)

    X = sparse.csr_matrix((data, (rows, cols)), shape=(n, L * A), dtype=np.uint8)
    return X, L


def gc_content(seqs: pd.Series) -> np.ndarray:
    s = seqs.fillna("").astype(str).str.upper().str.strip()
    gc = s.str.count("G") + s.str.count("C")
    atgc = gc + s.str.count("A") + s.str.count("T")
    with np.errstate(divide="ignore", invalid="ignore"):
        frac = (gc / atgc).astype(float)
    return np.nan_to_num(frac.to_numpy(dtype=float), nan=0.0, posinf=0.0, neginf=0.0)


def kmer_frequencies(seqs: pd.Series, *, k: int) -> np.ndarray:
    k = int(k)
    if k <= 0:
        raise ValueError("k must be >= 1")
    s_list = seqs.fillna("").astype(str).str.upper().str.strip().to_list()
    n = len(s_list)
    alpha = {"A": 0, "C": 1, "G": 2, "T": 3}
    n_feat = 4 ** k
    X = np.zeros((n, n_feat), dtype=np.float32)
    for i, s in enumerate(s_list):
        L = len(s)
        if L < k:
            continue
        denom = 0
        for j in range(L - k + 1):
            idx = 0
            ok = True
            for ch in s[j : j + k]:
                v = alpha.get(ch)
                if v is None:
                    ok = False
                    break
                idx = (idx * 4) + v
            if not ok:
                continue
            X[i, idx] += 1.0
            denom += 1
        if denom > 0:
            X[i, :] /= float(denom)
    return X


def build_features(
    seqs: pd.Series, *, embedding: str, kmer_k: int
) -> tuple[np.ndarray, int]:
    """
    Returns:
      X_dense: feature matrix used for linear fit
      onehot_dim: number of leading one-hot columns (0 if no one-hot portion)
    """
    X_onehot_sp, seq_len = one_hot_encode_dna(seqs)
    X_onehot = X_onehot_sp.toarray().astype(np.float64)
    onehot_dim = int(X_onehot.shape[1])

    if embedding == "onehot":
        return X_onehot, onehot_dim
    if embedding == "kmer_gc_onehot":
        Xk = kmer_frequencies(seqs, k=int(kmer_k)).astype(np.float64)
        gc = gc_content(seqs).astype(np.float64).reshape(-1, 1)
        X = np.concatenate([X_onehot, Xk, gc], axis=1)
        return X, onehot_dim
    raise ValueError(f"Unsupported embedding: {embedding}")


def coefficient_matrix_from_flat(w: np.ndarray, seq_len: int, alphabet: str = "ACGT") -> pd.DataFrame:
    A = len(alphabet)
    if w.size != seq_len * A:
        raise ValueError(f"Coefficient length ({w.size}) does not match expected ({seq_len * A})")
    mat = w.reshape(seq_len, A)
    df = pd.DataFrame(mat, columns=list(alphabet))
    df.insert(0, "position", np.arange(1, seq_len + 1))
    return df


def fit_linear_models(X: np.ndarray, y: np.ndarray) -> dict[str, np.ndarray]:
    models = {
        "ridge": Ridge(alpha=1.0),
        "lasso": Lasso(alpha=1e-3, max_iter=10000, random_state=42),
        "elasticnet": ElasticNet(alpha=1e-3, l1_ratio=0.5, max_iter=10000, random_state=42),
    }
    out: dict[str, np.ndarray] = {}
    for name, model in models.items():
        pipe = Pipeline(
            [
                ("x_scaler", StandardScaler()),
                ("model", model),
            ]
        )
        pipe.fit(X, y)
        coef = np.asarray(pipe.named_steps["model"].coef_, dtype=float).reshape(-1)
        out[name] = coef
    return out


def plot_weights_by_position(df_coef: pd.DataFrame, outpath: Path, *, title: str) -> None:
    set_plot_style()
    df_long = df_coef.melt(id_vars="position", var_name="base", value_name="weight")
    plt.figure(figsize=(16.0 / 2.54, 12.0 / 2.54))
    ax = sns.lineplot(data=df_long, x="position", y="weight", hue="base", linewidth=1.8)
    ax.axhline(0.0, color="black", linewidth=0.8, linestyle="--", alpha=0.7)
    ax.set_title(title)
    ax.set_xlabel("Position")
    ax.set_ylabel("Coefficient weight")
    savefig(outpath)


def plot_abs_weight_sum(df_coef: pd.DataFrame, outpath: Path, *, title: str) -> None:
    set_plot_style()
    dfa = df_coef.copy()
    bases = [c for c in dfa.columns if c != "position"]
    dfa["abs_weight_sum"] = dfa[bases].abs().sum(axis=1)
    n_pos = int(len(dfa))
    # Wide figure + small horizontal tick labels (no rotation).
    width_cm = max(28.0, min(52.0, 0.95 * float(n_pos)))
    plt.figure(figsize=(width_cm / 2.54, 12.0 / 2.54))
    ax = sns.barplot(data=dfa, x="position", y="abs_weight_sum", color="black", alpha=0.5)
    ax.set_title(title)
    ax.set_xlabel("Position")
    ax.set_ylabel("Sum of |weights| across A/C/G/T")
    ax.tick_params(axis="x", labelsize=7)
    savefig(outpath)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--excel",
        default="output/data_processing/251129_P8_120Complete_dataset_woCtrl_exogenous_with_splits.xlsx",
    )
    ap.add_argument("--sheet", default=0)
    ap.add_argument("--target-col", default="FE")
    ap.add_argument("--split-col", default="Split")
    ap.add_argument(
        "--embeddings",
        default="onehot,kmer_gc_onehot",
        help="Comma-separated embeddings to run: onehot,kmer_gc_onehot",
    )
    ap.add_argument("--kmer-k", type=int, default=2, choices=[2, 3, 4], help="k for k-mer in kmer_gc_onehot")
    ap.add_argument(
        "--output-dir",
        default="output/interpretation/linear_position_weights",
        help="Directory for coefficients and plots",
    )
    args = ap.parse_args(argv)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_excel(args.excel, sheet_name=args.sheet, engine="openpyxl")
    needed = {"SpacerSeq", args.target_col, args.split_col}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    d = df.copy()
    d[args.target_col] = pd.to_numeric(d[args.target_col], errors="coerce")
    d = d[d[args.target_col].notna() & d["SpacerSeq"].notna()].reset_index(drop=True)

    # Match train_models behavior: only train on CV rows.
    cv_mask = d[args.split_col].astype(str).str.startswith("cv")
    d = d.loc[cv_mask].reset_index(drop=True)
    if d.empty:
        raise ValueError("No CV rows found after filtering by split_col prefix 'cv'.")

    y = d[args.target_col].to_numpy(dtype=float)
    _, seq_len = one_hot_encode_dna(d["SpacerSeq"])

    embeddings = [e.strip() for e in str(args.embeddings).split(",") if e.strip()]
    supported = {"onehot", "kmer_gc_onehot"}
    bad = sorted(set(embeddings) - supported)
    if bad:
        raise ValueError(f"Unsupported embeddings: {bad}. Supported: {sorted(supported)}")

    for emb in embeddings:
        emb_dir = out_dir / emb
        emb_dir.mkdir(parents=True, exist_ok=True)
        X, onehot_dim = build_features(d["SpacerSeq"], embedding=emb, kmer_k=int(args.kmer_k))
        model_coefs = fit_linear_models(X, y)
        for model_name, w in model_coefs.items():
            # Full coefficients for reproducibility/debugging.
            pd.DataFrame({"feature_idx": np.arange(w.size), "coef": w}).to_csv(
                emb_dir / f"{model_name}_coef_full.csv", index=False
            )
            # Position plots are based on the one-hot slice only.
            w_onehot = np.asarray(w[:onehot_dim], dtype=float)
            df_coef = coefficient_matrix_from_flat(w_onehot, seq_len=seq_len, alphabet="ACGT")
            df_coef.to_csv(emb_dir / f"{model_name}_coef_by_position_base.csv", index=False)

            suffix = "" if emb == "onehot" else f" ({emb}, k={int(args.kmer_k)})"
            plot_weights_by_position(
                df_coef,
                emb_dir / f"{model_name}_weights_by_position.png",
                title=f"{model_name} coefficients by position/base{suffix}",
            )
            plot_abs_weight_sum(
                df_coef,
                emb_dir / f"{model_name}_abs_weight_sum_by_position.png",
                title=f"{model_name} absolute weight sum by position{suffix}",
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

