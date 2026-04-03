"""
Genomic YN-PAM spacer scanning + CNN log2-FE prediction.

Checkpoint I/O and scaling use ``training.train_torch``; one-hot features use
``simple_modeling.train_models.build_feature_matrix`` (``embedding="onehot"``). No duplicated helpers.
"""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch

from spacerModel.simple_modeling.train_models import build_feature_matrix

# PlotStyle imported lazily in plot_predicted_fe_density to match analyze_dataset (dpi/figsize/fonts).
from spacerModel.training.train_torch import (
    load_torch_best_checkpoint,
    predict_numpy_from_checkpoint,
    resolve_device,
)

DEFAULT_GENOMIC_SPACER_LEN = 32

# Bases trimmed from each end of each FASTA record before YN-PAM scanning; keeps flanks for primer
# design while coordinates in the output table remain 1-based on the full record.
DEFAULT_SPACER_SEARCH_EDGE_RESERVE = 400

# Output column for CNN predictions (log2-FE); matches plot axis titles.
PREDICTED_LOG2_FE_COL = "Predicted_log2_FE"


def parse_fasta_records(path: Path) -> list[tuple[str, str]]:
    """
    Split a FASTA file into ``(gene, sequence)`` pairs.

    ``gene`` is the header text after ``>`` (first line of each record), stripped.
    If the file has no ``>`` lines, the whole file is treated as one sequence with
    ``gene`` set to ``"(no_header)"``.
    """
    raw = path.read_text(encoding="utf-8", errors="replace")
    if ">" not in raw:
        seq_lines = [ln.strip() for ln in raw.splitlines() if ln.strip() and not ln.startswith(">")]
        seq = "".join(seq_lines).upper()
        return [("(no_header)", seq)] if seq else []

    records: list[tuple[str, str]] = []
    header: str | None = None
    seq_parts: list[str] = []

    for line in raw.splitlines():
        s = line.strip()
        if not s:
            continue
        if s.startswith(">"):
            if header is not None or seq_parts:
                seq = "".join(seq_parts).upper()
                records.append((header or "", seq))
            header = s[1:].strip()
            seq_parts = []
        else:
            seq_parts.append(s.upper())

    if header is not None or seq_parts:
        seq = "".join(seq_parts).upper()
        records.append((header or "", seq))

    return records


def load_genomic_sequence(path: Path) -> str:
    """Load sequence from FASTA or plain text (all records concatenated; prefer ``parse_fasta_records`` for per-gene)."""
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


def plot_predicted_fe_density(
    df: pd.DataFrame,
    outpath: Path,
    *,
    title: str | None = None,
) -> None:
    """
    Kernel density of ``Predicted_log2_FE``. Matches ``data/analyze_dataset.plot_fe_density`` (PlotStyle:
    300 dpi, 16×12 cm figure, Arial 14pt via ``set_plot_style``). Writes ``.png`` and ``.pdf`` next to
    the given path (strip ``.png``/``.pdf`` to get base stem, or use a stem without extension).
    """
    from spacerModel.data.analyze_dataset import PlotStyle, set_plot_style

    if PREDICTED_LOG2_FE_COL not in df.columns or df[PREDICTED_LOG2_FE_COL].empty:
        return
    vals = pd.to_numeric(df[PREDICTED_LOG2_FE_COL], errors="coerce").dropna()
    if vals.empty:
        return

    style = PlotStyle()
    set_plot_style(style)
    fig, ax = plt.subplots(figsize=(style.rect_width_cm / 2.54, style.rect_height_cm / 2.54))
    if len(vals) >= 2:
        try:
            sns.kdeplot(x=vals, ax=ax, fill=True, color="steelblue", linewidth=1.0)
        except Exception:
            sns.histplot(vals, ax=ax, stat="density", kde=True, color="steelblue", edgecolor="white")
    else:
        ax.axvline(float(vals.iloc[0]), color="steelblue", linewidth=2.0)
        ax.set_ylim(bottom=0.0)
    ax.set_xlabel("Predicted log2-FE")
    ax.set_ylabel("Density")
    ax.set_title(title or "Predicted log2-FE density")
    ax.grid(False)
    sns.despine(ax=ax)
    fig.tight_layout()
    outpath = Path(outpath)
    if outpath.suffix.lower() in (".png", ".pdf"):
        base = outpath.with_suffix("")
    else:
        base = outpath
    png_path = base.with_suffix(".png")
    pdf_path = base.with_suffix(".pdf")
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=style.dpi, bbox_inches="tight")
    fig.savefig(pdf_path, dpi=style.dpi, bbox_inches="tight")
    plt.close(fig)


def _safe_plot_stem_fragment(name: str, *, max_len: int = 80) -> str:
    """Filesystem-safe fragment from a gene / header string (for multi-gene plot stems)."""
    s = re.sub(r"[^\w\-.]+", "_", str(name).strip())
    s = s.strip("._-") or "gene"
    return s[:max_len]


def _gene_short_prefix(gene: str) -> str:
    """Substring before the first ``_`` (plot titles and stems); if none, full header stripped."""
    s = str(gene).strip()
    if not s:
        return "(unknown)"
    if "_" not in s:
        return s
    head, _ = s.split("_", 1)
    head = head.strip()
    return head if head else s


def _plot_stem_base(outpath: Path) -> Path:
    outpath = Path(outpath)
    if outpath.suffix.lower() in (".png", ".pdf"):
        return outpath.with_suffix("")
    return outpath


def plot_predicted_fe_density_by_gene(df: pd.DataFrame, outpath: Path) -> list[Path]:
    """
    KDE of ``Predicted_log2_FE`` per ``Gene``: one PNG+PDF pair per distinct gene. Filenames are always
    ``<base>_<short_prefix>`` (never ``<base>`` alone): the short prefix is the header text before the first
    ``_`` (then filesystem-safe); collisions become ``_2``, ``_3``, … If there is no ``Gene`` column,
    writes a single ``<base>`` plot without a gene suffix.
    Returns the list of ``.png`` paths written (empty if nothing could be plotted).
    """
    base = _plot_stem_base(outpath)
    if "Gene" not in df.columns:
        vals = pd.to_numeric(
            df.get(PREDICTED_LOG2_FE_COL, pd.Series(dtype=float)), errors="coerce"
        ).dropna()
        if vals.empty:
            return []
        plot_predicted_fe_density(df, outpath, title="Predicted log2-FE density")
        return [base.with_suffix(".png")]

    work = df.copy()
    work["Gene"] = work["Gene"].fillna("(unknown)")
    genes = sorted(work["Gene"].unique(), key=str)

    used: set[str] = set()
    png_paths: list[Path] = []

    def unique_frag(label: str) -> str:
        frag = _safe_plot_stem_fragment(label)
        if frag not in used:
            used.add(frag)
            return frag
        i = 2
        while True:
            cand = f"{frag}_{i}"
            if cand not in used:
                used.add(cand)
                return cand
            i += 1

    # One code path for one or many genes: always ``<base>_<short_prefix>.png`` (never write the base alone).
    for g in genes:
        sub = work[work["Gene"] == g]
        vals = pd.to_numeric(sub[PREDICTED_LOG2_FE_COL], errors="coerce").dropna()
        if vals.empty:
            continue
        short = _gene_short_prefix(str(g))
        frag = unique_frag(short)
        gene_base = base.parent / f"{base.name}_{frag}"
        plot_predicted_fe_density(
            sub,
            gene_base.with_suffix(".png"),
            title=f"{short}: Predicted log2-FE density",
        )
        png_paths.append(gene_base.with_suffix(".png"))

    return png_paths


def write_ranked_sequences_excel_per_gene(
    df: pd.DataFrame,
    out_dir: Path,
    *,
    stem: str = "ranked_sequences",
) -> list[tuple[Path, int]]:
    """
    Write ``{stem}.xlsx`` (all rows) and one ``{stem}_<gene>.xlsx`` per distinct ``Gene``.
    If ``Gene`` is missing, writes only ``{stem}.xlsx``. Returns ``(path, row_count)`` pairs.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if df.empty:
        return []

    if "Gene" not in df.columns:
        path = out_dir / f"{stem}.xlsx"
        df.to_excel(path, index=False, engine="openpyxl")
        return [(path, len(df))]

    work = df.copy()
    work["Gene"] = work["Gene"].fillna("(unknown)")
    genes = sorted(work["Gene"].unique(), key=str)
    used: set[str] = set()

    def unique_frag(label: str) -> str:
        frag = _safe_plot_stem_fragment(label)
        if frag not in used:
            used.add(frag)
            return frag
        i = 2
        while True:
            cand = f"{frag}_{i}"
            if cand not in used:
                used.add(cand)
                return cand
            i += 1

    written: list[tuple[Path, int]] = []
    for g in genes:
        sub = work[work["Gene"] == g]
        if sub.empty:
            continue
        frag = unique_frag(str(g))
        path = out_dir / f"{stem}_{frag}.xlsx"
        sub.to_excel(path, index=False, engine="openpyxl")
        written.append((path, len(sub)))

    combined_path = out_dir / f"{stem}.xlsx"
    df.to_excel(combined_path, index=False, engine="openpyxl")
    written.insert(0, (combined_path, len(df)))

    return written


def write_ranked_excel_per_gene_partitioned(
    df: pd.DataFrame,
    out_dir: Path,
    *,
    stem: str,
    partition_key: pd.Series,
) -> list[tuple[Path, int]]:
    """
    Write ``{stem}.xlsx`` (all rows) and one ``{stem}_{frag}.xlsx`` per distinct label in
    ``partition_key`` (must align with ``df.index``). Filenames use :func:`_safe_plot_stem_fragment`
    with collision suffixes ``_2``, ``_3``, … matching :func:`write_ranked_sequences_excel_per_gene`.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if df.empty:
        return []

    if len(partition_key) != len(df) or not partition_key.index.equals(df.index):
        raise ValueError("partition_key must be aligned with df (same length and index as df)")

    combined_path = out_dir / f"{stem}.xlsx"
    df.to_excel(combined_path, index=False, engine="openpyxl")
    written: list[tuple[Path, int]] = [(combined_path, len(df))]

    keys = sorted(partition_key.unique(), key=lambda x: str(x))
    used: set[str] = set()

    def unique_frag(label: str) -> str:
        frag = _safe_plot_stem_fragment(label)
        if frag not in used:
            used.add(frag)
            return frag
        i = 2
        while True:
            cand = f"{frag}_{i}"
            if cand not in used:
                used.add(cand)
                return cand
            i += 1

    for k in keys:
        sub = df.loc[partition_key == k]
        if sub.empty:
            continue
        frag = unique_frag(str(k))
        path = out_dir / f"{stem}_{frag}.xlsx"
        sub.to_excel(path, index=False, engine="openpyxl")
        written.append((path, len(sub)))

    return written


def run_genomic_cnn_spacer_prediction(
    checkpoint: Path,
    fasta: Path,
    *,
    device: str | torch.device = "auto",
    batch_size: int = 512,
    spacer_len: int | None = None,
    spacer_search_edge_reserve: int = DEFAULT_SPACER_SEARCH_EDGE_RESERVE,
) -> pd.DataFrame:
    """
    Load a CNN ``save_torch_best_checkpoint`` file, scan ``fasta`` for YN-PAM sites, predict log2-FE.

    YN-PAM scanning runs only on the **middle** of each sequence, excluding ``spacer_search_edge_reserve``
    bases from each end (default 400), so FASTA records can include flanking sequence for primer tools
    while spacer hits are kept away from the margins. ``Start`` / ``SpacerStart`` / ``SpacerEnd`` are
    still **1-based on the full FASTA record**. Use ``0`` to scan the entire sequence (legacy behavior).

    Returns a dataframe sorted by ``Predicted_log2_FE`` (descending). Empty if no sites.
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

    edge = int(spacer_search_edge_reserve)
    if edge < 0:
        raise ValueError(f"spacer_search_edge_reserve must be >= 0; got {edge}")

    fasta_path = Path(fasta)
    gene_seqs = parse_fasta_records(fasta_path)
    parts: list[pd.DataFrame] = []
    for gene, genomic in gene_seqs:
        if not genomic:
            continue
        n = len(genomic)
        if edge == 0:
            core = genomic
            coord_offset = 0
        elif n <= 2 * edge:
            continue
        else:
            core = genomic[edge : n - edge]
            coord_offset = edge

        sub = extract_spacers_with_yn_pam_both_strands(core, spacer_len=int(spacer_len))
        if sub.empty:
            continue
        sub = sub.copy()
        if coord_offset:
            for col in ("Start", "SpacerStart", "SpacerEnd"):
                sub[col] = sub[col].astype(int) + coord_offset
        sub.insert(0, "Gene", gene)
        parts.append(sub)

    if not parts:
        return pd.DataFrame()

    cand = pd.concat(parts, axis=0, ignore_index=True)

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
    out[PREDICTED_LOG2_FE_COL] = np.asarray(pred, dtype=float)
    out = out.sort_values(PREDICTED_LOG2_FE_COL, ascending=False)
    # Gene first for readability
    cols = ["Gene"] + [c for c in out.columns if c != "Gene"]
    return out[cols]
