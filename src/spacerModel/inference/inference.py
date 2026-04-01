"""
Genomic YN-PAM spacer scanning + CNN FE prediction.

Checkpoint I/O and scaling use ``training.train_torch``; one-hot features use
``simple_modeling.train_models.build_feature_matrix`` (``embedding="onehot"``). No duplicated helpers.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from spacerModel.simple_modeling.train_models import build_feature_matrix
from spacerModel.training.train_torch import (
    load_torch_best_checkpoint,
    predict_numpy_from_checkpoint,
    resolve_device,
)

DEFAULT_GENOMIC_SPACER_LEN = 32


def load_genomic_sequence(path: Path) -> str:
    """Load sequence from FASTA or plain text (skip '>' lines, concatenate sequence)."""
    lines = [ln.strip() for ln in path.read_text(encoding="utf-8", errors="replace").splitlines()]
    seq_lines = [ln for ln in lines if ln and not ln.startswith(">")]
    return "".join(seq_lines).upper()


def reverse_complement(seq: str) -> str:
    comp = str.maketrans("ACGTacgt", "TGCAtgca")
    return seq.translate(comp)[::-1]


def extract_spacers_with_yn_pam_both_strands(
    seq: str,
    *,
    spacer_len: int = DEFAULT_GENOMIC_SPACER_LEN,
) -> pd.DataFrame:
    """
    Scan both strands for ``[PAM(2)][SPACER(spacer_len)]`` with YN PAM (Y=C/T, N=any).

    Columns: Strand, Start, PAM, SpacerSeq, SpacerStart, SpacerEnd (1-based forward coords).
    """
    seq = seq.upper()
    n = len(seq)
    window_len = 2 + spacer_len
    records: list[dict[str, object]] = []

    for i in range(n - window_len + 1):
        pam = seq[i : i + 2]
        if len(pam) < 2:
            continue
        if pam[0] in ("C", "T"):
            spacer = seq[i + 2 : i + 2 + spacer_len]
            if len(spacer) == spacer_len and set(spacer).issubset({"A", "C", "G", "T"}):
                pam_start = i + 1
                spacer_start = i + 2 + 1
                spacer_end = spacer_start + spacer_len - 1
                records.append(
                    {
                        "Strand": "+",
                        "Start": pam_start,
                        "PAM": pam,
                        "SpacerSeq": spacer,
                        "SpacerStart": spacer_start,
                        "SpacerEnd": spacer_end,
                    }
                )

    rc = reverse_complement(seq)
    for j in range(n - window_len + 1):
        pam = rc[j : j + 2]
        if len(pam) < 2:
            continue
        if pam[0] in ("C", "T"):
            spacer = rc[j + 2 : j + 2 + spacer_len]
            if len(spacer) == spacer_len and set(spacer).issubset({"A", "C", "G", "T"}):
                spacer_first_rc = j + 2
                spacer_last_rc = j + 2 + spacer_len - 1
                spacer_first_orig0 = (n - 1) - spacer_first_rc
                spacer_last_orig0 = (n - 1) - spacer_last_rc
                spacer_start0 = min(spacer_first_orig0, spacer_last_orig0)
                spacer_end0 = max(spacer_first_orig0, spacer_last_orig0)
                spacer_start = spacer_start0 + 1
                spacer_end = spacer_end0 + 1
                pam_idx0_orig0 = (n - 1) - j
                pam_idx1_orig0 = (n - 1) - (j + 1)
                pam_start0 = min(pam_idx0_orig0, pam_idx1_orig0)
                pam_start = pam_start0 + 1
                records.append(
                    {
                        "Strand": "-",
                        "Start": pam_start,
                        "PAM": pam,
                        "SpacerSeq": spacer,
                        "SpacerStart": spacer_start,
                        "SpacerEnd": spacer_end,
                    }
                )

    return pd.DataFrame(records)


def run_genomic_cnn_spacer_prediction(
    checkpoint: Path,
    fasta: Path,
    *,
    device: str | torch.device = "auto",
    batch_size: int = 512,
    spacer_len: int | None = None,
) -> pd.DataFrame:
    """
    Load a CNN ``save_torch_best_checkpoint`` file, scan ``fasta`` for YN-PAM sites, predict FE.

    Returns a dataframe sorted by ``Predicted_FE`` (descending). Empty if no sites.
    ``spacer_len`` defaults to ``n_features//4`` from the checkpoint.
    """
    dev = device if isinstance(device, torch.device) else resolve_device(str(device))
    model, pack = load_torch_best_checkpoint(Path(checkpoint), device=dev)
    if pack.get("model_kind") != "cnn":
        raise ValueError(
            f"run_genomic_cnn_spacer_prediction expects model_kind='cnn'; got {pack.get('model_kind')!r}"
        )
    n_feat = int(pack["n_features"])
    if n_feat % 4 != 0:
        raise ValueError(f"Invalid checkpoint n_features={n_feat} (expected divisible by 4).")
    L_exp = n_feat // 4
    if spacer_len is None:
        spacer_len = L_exp
    elif int(spacer_len) != L_exp:
        raise ValueError(
            f"spacer_len={spacer_len} incompatible with checkpoint: need L={L_exp} (n_features={n_feat})."
        )

    genomic = load_genomic_sequence(Path(fasta))
    cand = extract_spacers_with_yn_pam_both_strands(genomic, spacer_len=int(spacer_len))
    if cand.empty:
        return cand

    # Same as CNN training / explain_cnn_layers: sparse one-hot → dense float64
    X_sp = build_feature_matrix(cand, embedding="onehot", kmer_k=2)
    X = np.asarray(X_sp.toarray(), dtype=np.float64)
    if X.shape[1] != n_feat:
        raise ValueError(f"Built features shape={X.shape} but checkpoint expects n_features={n_feat}.")

    pred = predict_numpy_from_checkpoint(
        pack,
        model,
        X,
        kind="cnn",
        device=dev,
        batch_size=int(batch_size),
    )
    out = cand.copy()
    out["Predicted_FE"] = np.asarray(pred, dtype=float)
    return out.sort_values("Predicted_FE", ascending=False)
