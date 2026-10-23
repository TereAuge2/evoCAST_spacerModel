#!/usr/bin/env python3
"""
Pick spacer rows per gene: (1) ``number_top_picked`` best model ranks with primers;
(2) ``number_bottom_picked`` worst model ranks (scanning up from the bottom of the ranked list);
(3) ``number_random_picked`` additional spacers with primers at random from the remainder.
Runs NCBI Primer-BLAST via :func:`primer.process_dataframe` with ``per_spacer_template=True``.
Spacers whose ``SpacerSeq`` contains a run of **N** consecutive ``T`` (default **4**, i.e. ``TTTT+``)
are skipped before Primer-BLAST; set ``--min-consecutive-t-skip 0`` to disable.
Also skip if any **dinucleotide** repeats contiguously at least **M** times (default **6**, e.g.
``(AC)⁶`` / ``ACACACACACAC``); set ``--min-dinucleotide-repeat-skip 0`` to disable.

Output table shortens ``Gene`` to the symbol before the first underscore, adds
``total_spacers_gene`` from the ranked input (next to ``rank``), and drops ``ncbi_input_fasta``.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from spacerModel.inference.inference import PREDICTED_LOG2_FE_COL, add_spacer_uniqueness_columns
from spacerModel.inference.primer import (
    DEFAULT_DOWNSTREAM,
    DEFAULT_NCBI_NUM_PRIMER_PAIRS,
    DEFAULT_NCBI_ORGANISM,
    DEFAULT_NCBI_SPECIFICITY_DB,
    DEFAULT_UPSTREAM,
    FWD_MAX_PAST_SPACER_END,
    NCBI_POST_PRODUCT_MAX,
    NCBI_POST_PRODUCT_MIN,
    REV_MIN_PAST_SPACER_END,
    gene_matches_filter,
    load_gene_sequences,
    process_dataframe,
)


def load_ranked_table(path: Path) -> pd.DataFrame:
    """Load ``ranked_sequences`` table from ``.csv``, ``.tsv``, or Excel."""
    suf = path.suffix.lower()
    if suf == ".csv":
        return pd.read_csv(path)
    if suf == ".tsv":
        return pd.read_csv(path, sep="\t")
    if suf in (".xlsx", ".xls"):
        return pd.read_excel(path, engine="openpyxl")
    raise ValueError(f"Unsupported input suffix {path.suffix!r} (use .csv, .tsv, or .xlsx)")


def add_within_gene_rank(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add ``rank`` = 1..n within each ``Gene``, best model score first
    (``Predicted_log2_FE`` descending).
    """
    if "Gene" not in df.columns:
        raise ValueError("input table: missing Gene column")
    if PREDICTED_LOG2_FE_COL not in df.columns:
        raise ValueError(f"input table: missing {PREDICTED_LOG2_FE_COL!r} (needed for ranking)")
    out = df.copy()
    fe = pd.to_numeric(out[PREDICTED_LOG2_FE_COL], errors="coerce")
    out["_fe_sort"] = fe
    out = out.sort_values(
        ["Gene", "_fe_sort"],
        ascending=[True, False],
        na_position="last",
        kind="mergesort",
    )
    out["rank"] = out.groupby("Gene", sort=False).cumcount() + 1
    out = out.drop(columns=["_fe_sort"])
    return out.reset_index(drop=True)


def _spacer_key(row: pd.Series) -> tuple[Any, ...]:
    g = str(row["Gene"])
    ss = int(row["SpacerStart"])
    se = int(row["SpacerEnd"])
    st = str(row["Strand"]) if "Strand" in row.index and pd.notna(row["Strand"]) else "+"
    return (g, ss, se, st)


def _row_has_passing_primers(row: pd.Series) -> bool:
    c = row.get("primer_pairs_passing_count")
    if c is None or pd.isna(c):
        return False
    try:
        return int(c) > 0
    except (TypeError, ValueError):
        return False


def _should_skip_spacer_for_consecutive_t(row: pd.Series, min_len: int) -> bool:
    """
    True if ``SpacerSeq`` contains ``min_len`` or more consecutive T (case-insensitive).
    If ``min_len`` <= 0, never skip. Missing or empty ``SpacerSeq`` does not skip.
    """
    if min_len <= 0:
        return False
    if "SpacerSeq" not in row.index:
        return False
    v = row.get("SpacerSeq")
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return False
    s = str(v).strip().upper()
    if len(s) < min_len:
        return False
    return ("T" * min_len) in s


def _max_contiguous_dinucleotide_repeats(seq: str) -> int:
    """
    Largest ``k`` such that ``SpacerSeq`` contains a substring ``(UV)`` repeated ``k`` times in a row
    (non-overlapping 2-mers), for some dinucleotide ``UV`` ∈ {A,C,G,T}². Examples: ``ACACAC`` → 3 (AC);
    ``AAAAAAAAAAAA`` → 6 (AA).
    """
    s = "".join(c for c in str(seq).upper() if c in "ACGT")
    n = len(s)
    if n < 2:
        return 0
    best = 1
    for i in range(n - 1):
        unit = s[i : i + 2]
        rep = 1
        j = i + 2
        while j + 2 <= n and s[j : j + 2] == unit:
            rep += 1
            j += 2
        if rep > best:
            best = rep
    return best


def _should_skip_spacer_for_dinucleotide_repeat(row: pd.Series, min_repeats: int) -> bool:
    """
    True if some dinucleotide repeats contiguously at least ``min_repeats`` times.
    If ``min_repeats`` <= 0, never skip. Missing ``SpacerSeq`` does not skip.
    Non-ACGT letters are ignored for repeat detection.
    """
    if min_repeats <= 0:
        return False
    if "SpacerSeq" not in row.index:
        return False
    v = row.get("SpacerSeq")
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return False
    s_clean = "".join(c for c in str(v).upper() if c in "ACGT")
    if len(s_clean) < 2 * min_repeats:
        return False
    return _max_contiguous_dinucleotide_repeats(s_clean) >= min_repeats


def _spacer_row_fails_sequence_filters(
    row: pd.Series,
    *,
    min_consecutive_t_skip: int,
    min_dinucleotide_repeat_skip: int,
) -> tuple[bool, str]:
    """
    Returns ``(skip, reason)`` where ``reason`` is empty if ``skip`` is False.
    """
    if min_consecutive_t_skip > 0 and _should_skip_spacer_for_consecutive_t(row, min_consecutive_t_skip):
        return True, f"≥{min_consecutive_t_skip} consecutive T"
    if min_dinucleotide_repeat_skip > 0 and _should_skip_spacer_for_dinucleotide_repeat(
        row, min_dinucleotide_repeat_skip
    ):
        return (
            True,
            f"dinucleotide repeat ≥{min_dinucleotide_repeat_skip} (e.g. (XY)^{min_dinucleotide_repeat_skip})",
        )
    return False, ""


def _ensure_strand(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "Strand" not in out.columns:
        out["Strand"] = "+"
    return out


def short_gene_name(gene_cell: str) -> str:
    """Short symbol: substring of ``Gene`` before the first ``_`` (e.g. MYO7A from MYO7A_round2_...)."""
    s = str(gene_cell).strip()
    if not s:
        return s
    return s.split("_", 1)[0].strip()


def format_gene_columns(df: pd.DataFrame, counts_source: pd.DataFrame) -> pd.DataFrame:
    """
    Shorten ``Gene`` to the prefix before the first ``_``, add ``total_spacers_gene`` (row counts per
    short gene derived from ``counts_source``), drop ``ncbi_input_fasta`` if present, and place
    ``total_spacers_gene`` immediately after ``rank`` (if present; otherwise after ``Gene``).
    """
    if df.empty:
        return df
    out = df.copy()
    if "ncbi_input_fasta" in out.columns:
        out = out.drop(columns=["ncbi_input_fasta"])
    cs = counts_source.copy()
    cs["_gshort"] = cs["Gene"].map(short_gene_name)
    n_per_short = cs.groupby("_gshort", sort=False).size()
    gshort = out["Gene"].map(short_gene_name)
    out["total_spacers_gene"] = gshort.map(n_per_short).astype("Int64")
    out["Gene"] = gshort.astype(str)
    cols = [c for c in out.columns if c != "total_spacers_gene"]
    if "rank" in cols:
        ir = cols.index("rank")
        cols = cols[: ir + 1] + ["total_spacers_gene"] + cols[ir + 1 :]
    else:
        ig = cols.index("Gene")
        cols = cols[: ig + 1] + ["total_spacers_gene"] + cols[ig + 1 :]
    return out[cols]


def finalize_picks_output(combined: pd.DataFrame, ranked: pd.DataFrame) -> pd.DataFrame:
    """Apply :func:`format_gene_columns` to primer picks using the ranked table for counts."""
    return format_gene_columns(combined, ranked)


def pick_spacers_for_gene(
    df_g: pd.DataFrame,
    gene_to_seq: dict[str, str],
    *,
    number_top_picked: int,
    number_bottom_picked: int,
    number_random_picked: int,
    rng: np.random.Generator,
    dry_run: bool,
    verbose: bool,
    primer_kw: dict[str, Any],
    min_consecutive_t_skip: int = 4,
    min_dinucleotide_repeat_skip: int = 6,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Returns ``(top_df, worst_df, random_df)`` — primer-annotated rows only, each with ``pick_type``.
    Order of phases: best rank first (``top_model``), then worst rank upward (``worst_model``), then ``random``.
    """
    need_cols = {"Gene", "SpacerStart", "SpacerEnd"}
    missing = need_cols - set(df_g.columns)
    if missing:
        raise ValueError(f"Missing columns for primer step: {sorted(missing)}")

    df_g = _ensure_strand(df_g.copy())
    # Best rank first
    df_sorted = df_g.sort_values("rank", ascending=True, kind="mergesort").reset_index(drop=True)

    gene_short = short_gene_name(str(df_sorted["Gene"].iloc[0])) if len(df_sorted) else "?"

    top_rows: list[pd.Series] = []
    picked_keys: set[tuple[Any, ...]] = set()
    need = int(number_top_picked)

    # --- Top model order ---
    top_attempt = 0
    for _, row in df_sorted.iterrows():
        if len(top_rows) >= need:
            break
        skip, skip_reason = _spacer_row_fails_sequence_filters(
            row,
            min_consecutive_t_skip=min_consecutive_t_skip,
            min_dinucleotide_repeat_skip=min_dinucleotide_repeat_skip,
        )
        if skip:
            print(
                f"[picking] gene={gene_short} | top_model | skip — {skip_reason} | "
                f"model_rank={row['rank']} | SpacerStart={row['SpacerStart']}-{row['SpacerEnd']}",
                flush=True,
            )
            continue
        top_attempt += 1
        have = len(top_rows)
        print(
            f"[picking] gene={gene_short} | top_model | attempt {top_attempt} | "
            f"progress {have}/{need} | model_rank={row['rank']} | "
            f"SpacerStart={row['SpacerStart']}-{row['SpacerEnd']} | Strand={row.get('Strand', '+')}",
            flush=True,
        )
        one = pd.DataFrame([row.to_dict()])
        if dry_run:
            print(
                f"[picking] gene={gene_short} | top_model | attempt {top_attempt} | dry-run — skip NCBI",
                flush=True,
            )
            continue
        done = process_dataframe(one, gene_to_seq, **primer_kw)
        out_r = done.iloc[0]
        if _row_has_passing_primers(out_r):
            npp = int(out_r["primer_pairs_passing_count"])
            print(
                f"[picking] gene={gene_short} | top_model | attempt {top_attempt} | PASS "
                f"({npp} passing primer pair(s))",
                flush=True,
            )
            s = out_r.copy()
            s["pick_type"] = "top_model"
            top_rows.append(s)
            picked_keys.add(_spacer_key(row))
        else:
            print(
                f"[picking] gene={gene_short} | top_model | attempt {top_attempt} | FAIL "
                f"(no primers passing filters)",
                flush=True,
            )

    top_df = pd.DataFrame(top_rows) if top_rows else pd.DataFrame()
    print(
        f"[picking] gene={gene_short} | top_model | done — collected {len(top_df)}/{need} spacer(s) with primers",
        flush=True,
    )

    # --- Worst model order (highest rank / lowest FE first), skip already picked in top ---
    df_worst = df_sorted.sort_values("rank", ascending=False, kind="mergesort").reset_index(drop=True)
    need_w = int(number_bottom_picked)
    worst_rows: list[pd.Series] = []
    worst_attempt = 0
    for _, row in df_worst.iterrows():
        if len(worst_rows) >= need_w:
            break
        if _spacer_key(row) in picked_keys:
            continue
        skip, skip_reason = _spacer_row_fails_sequence_filters(
            row,
            min_consecutive_t_skip=min_consecutive_t_skip,
            min_dinucleotide_repeat_skip=min_dinucleotide_repeat_skip,
        )
        if skip:
            print(
                f"[picking] gene={gene_short} | worst_model | skip — {skip_reason} | "
                f"model_rank={row['rank']} | SpacerStart={row['SpacerStart']}-{row['SpacerEnd']}",
                flush=True,
            )
            continue
        worst_attempt += 1
        have_w = len(worst_rows)
        print(
            f"[picking] gene={gene_short} | worst_model | attempt {worst_attempt} | "
            f"progress {have_w}/{need_w} | model_rank={row['rank']} | "
            f"SpacerStart={row['SpacerStart']}-{row['SpacerEnd']} | Strand={row.get('Strand', '+')}",
            flush=True,
        )
        one = pd.DataFrame([row.to_dict()])
        if dry_run:
            print(
                f"[picking] gene={gene_short} | worst_model | attempt {worst_attempt} | dry-run — skip NCBI",
                flush=True,
            )
            continue
        done = process_dataframe(one, gene_to_seq, **primer_kw)
        out_r = done.iloc[0]
        if _row_has_passing_primers(out_r):
            npp = int(out_r["primer_pairs_passing_count"])
            print(
                f"[picking] gene={gene_short} | worst_model | attempt {worst_attempt} | PASS "
                f"({npp} passing primer pair(s))",
                flush=True,
            )
            s = out_r.copy()
            s["pick_type"] = "worst_model"
            worst_rows.append(s)
            picked_keys.add(_spacer_key(row))
        else:
            print(
                f"[picking] gene={gene_short} | worst_model | attempt {worst_attempt} | FAIL "
                f"(no primers passing filters)",
                flush=True,
            )

    worst_df = pd.DataFrame(worst_rows) if worst_rows else pd.DataFrame()
    print(
        f"[picking] gene={gene_short} | worst_model | done — collected {len(worst_df)}/{need_w} spacer(s) with primers",
        flush=True,
    )

    # --- Random from remaining (exclude already picked and sequence-filtered spacers) ---
    pool_idx = []
    for i in range(len(df_sorted)):
        if _spacer_key(df_sorted.iloc[i]) in picked_keys:
            continue
        sk, _ = _spacer_row_fails_sequence_filters(
            df_sorted.iloc[i],
            min_consecutive_t_skip=min_consecutive_t_skip,
            min_dinucleotide_repeat_skip=min_dinucleotide_repeat_skip,
        )
        if sk:
            continue
        pool_idx.append(i)
    pool_df = df_sorted.iloc[pool_idx].reset_index(drop=True)
    order = rng.permutation(len(pool_df)) if len(pool_df) else np.array([], dtype=int)

    rand_rows: list[pd.Series] = []
    rand_attempt = 0
    need_r = int(number_random_picked)
    for j in order:
        if len(rand_rows) >= need_r:
            break
        row = pool_df.iloc[j]
        rand_attempt += 1
        have_r = len(rand_rows)
        print(
            f"[picking] gene={gene_short} | random | attempt {rand_attempt} | "
            f"progress {have_r}/{need_r} | model_rank={row['rank']} | "
            f"SpacerStart={row['SpacerStart']}-{row['SpacerEnd']} | Strand={row.get('Strand', '+')}",
            flush=True,
        )
        one = pd.DataFrame([row.to_dict()])
        if dry_run:
            print(
                f"[picking] gene={gene_short} | random | attempt {rand_attempt} | dry-run — skip NCBI",
                flush=True,
            )
            continue
        done = process_dataframe(one, gene_to_seq, **primer_kw)
        out_r = done.iloc[0]
        if _row_has_passing_primers(out_r):
            npp = int(out_r["primer_pairs_passing_count"])
            print(
                f"[picking] gene={gene_short} | random | attempt {rand_attempt} | PASS "
                f"({npp} passing primer pair(s))",
                flush=True,
            )
            s = out_r.copy()
            s["pick_type"] = "random"
            rand_rows.append(s)
            picked_keys.add(_spacer_key(row))
        else:
            print(
                f"[picking] gene={gene_short} | random | attempt {rand_attempt} | FAIL "
                f"(no primers passing filters)",
                flush=True,
            )

    rand_df = pd.DataFrame(rand_rows) if rand_rows else pd.DataFrame()
    print(
        f"[picking] gene={gene_short} | random | done — collected {len(rand_df)}/{need_r} spacer(s) with primers",
        flush=True,
    )

    return top_df, worst_df, rand_df


def run_picking(
    ranked: pd.DataFrame,
    gene_to_seq: dict[str, str],
    *,
    gene_filter: str | None,
    number_top_picked: int,
    number_bottom_picked: int,
    number_random_picked: int,
    seed: int,
    dry_run: bool,
    verbose: bool,
    primer_kw: dict[str, Any],
    min_consecutive_t_skip: int = 4,
    min_dinucleotide_repeat_skip: int = 6,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns ``(ranked_with_rank, combined_picks)`` where ``combined_picks`` has primer columns
    plus ``pick_type``, original ``rank``, and uniqueness flags ``unique_in_locus`` /
    ``no_similar_spacers`` (computed only for picked rows; similarity is checked against
    the full ranked table per gene).
    """
    ranked = add_within_gene_rank(ranked)
    ranked = _ensure_strand(ranked)

    if gene_filter is not None and str(gene_filter).strip().lower() != "all":
        gwant = str(gene_filter).strip()
        mask = ranked["Gene"].map(lambda x: gene_matches_filter(str(x), gwant))
        ranked = ranked.loc[mask].reset_index(drop=True)
        print(f"[picking] gene filter {gwant!r} -> {len(ranked)} rows", flush=True)

    genes = ranked["Gene"].drop_duplicates().tolist()
    rng = np.random.default_rng(int(seed))

    all_top: list[pd.DataFrame] = []
    all_worst: list[pd.DataFrame] = []
    all_rand: list[pd.DataFrame] = []

    for gene in genes:
        df_g = ranked[ranked["Gene"] == gene].copy()
        print(
            f"[picking] Gene={str(gene)[:100]!r} rows={len(df_g)} "
            f"target top={number_top_picked} worst={number_bottom_picked} random={number_random_picked}",
            flush=True,
        )
        top_df, worst_df, rand_df = pick_spacers_for_gene(
            df_g,
            gene_to_seq,
            number_top_picked=number_top_picked,
            number_bottom_picked=number_bottom_picked,
            number_random_picked=number_random_picked,
            rng=rng,
            dry_run=dry_run,
            verbose=verbose,
            primer_kw=primer_kw,
            min_consecutive_t_skip=min_consecutive_t_skip,
            min_dinucleotide_repeat_skip=min_dinucleotide_repeat_skip,
        )
        if not top_df.empty:
            all_top.append(top_df)
        if not worst_df.empty:
            all_worst.append(worst_df)
        if not rand_df.empty:
            all_rand.append(rand_df)

    parts = [df for df in (all_top + all_worst + all_rand) if not df.empty]
    if not parts:
        combined = pd.DataFrame()
    else:
        combined = pd.concat(parts, axis=0, ignore_index=True)
        pick_order = {"top_model": 0, "worst_model": 1, "random": 2}
        combined["_po"] = combined["pick_type"].map(pick_order)
        combined = combined.sort_values(
            ["Gene", "_po", "rank"],
            ascending=[True, True, True],
            kind="mergesort",
        )
        combined = combined.drop(columns=["_po"])
        # Locus uniqueness / pairwise similarity only for picked rows (full table is too slow).
        combined = add_spacer_uniqueness_columns(
            combined,
            gene_to_seq,
            similarity_against=ranked,
        )

    return ranked, combined


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Per gene: best-ranked spacers with primers (top_model), worst-ranked with primers (worst_model), "
            "then random spacers with primers. Uses primer.process_dataframe(..., per_spacer_template=True). "
            "Output: primer columns plus pick_type and rank; short Gene; total_spacers_gene; no ncbi_input_fasta."
        ),
    )
    ap.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Ranked table: .xlsx (typical), .csv, or .tsv (e.g. output/inference/ranked_sequences.xlsx)",
    )
    ap.add_argument(
        "--fasta",
        type=Path,
        required=True,
        help="Reference FASTA (same as spacer scan / primer.py)",
    )
    ap.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output .xlsx: picked rows (primer columns + pick_type + rank; Gene shortened; total_spacers_gene after rank)",
    )
    ap.add_argument(
        "--output-ranked",
        type=Path,
        default=None,
        metavar="PATH",
        help="Optional: full ranked table (same layout as --output: total_spacers_gene after rank; csv or xlsx)",
    )
    ap.add_argument(
        "--gene",
        type=str,
        default="all",
        metavar="NAME",
        help=(
            "Process only this gene, or 'all' (default). Matches full Gene cell or the substring "
            "before the first '_' (case-insensitive), e.g. MYO7A matches MYO7A_round2__hg19_...."
        ),
    )
    ap.add_argument(
        "--number-top-picked",
        type=int,
        default=3,
        metavar="N",
        dest="number_top_picked",
        help="Target count of best-ranked spacers with primers per gene (default: 3)",
    )
    ap.add_argument(
        "--number-bottom-picked",
        type=int,
        default=0,
        metavar="N",
        dest="number_bottom_picked",
        help=(
            "Target count of worst-ranked spacers (scan from lowest model rank upward) with primers per gene "
            "(default: 0). Skips spacers already taken in top_model."
        ),
    )
    ap.add_argument(
        "--number-random-picked",
        type=int,
        default=3,
        metavar="N",
        dest="number_random_picked",
        help="Target count of random spacers with primers per gene (default: 3)",
    )
    ap.add_argument("--seed", type=int, default=42, help="RNG seed for random picks (default: 42)")
    ap.add_argument(
        "--min-consecutive-t-skip",
        type=int,
        default=4,
        metavar="N",
        dest="min_consecutive_t_skip",
        help=(
            "Skip spacers whose SpacerSeq contains N consecutive T (case-insensitive); "
            "default 4 (TTTT or longer). Use 0 to disable."
        ),
    )
    ap.add_argument(
        "--min-dinucleotide-repeat-skip",
        type=int,
        default=6,
        metavar="M",
        dest="min_dinucleotide_repeat_skip",
        help=(
            "Skip spacers where some dinucleotide repeats contiguously at least M times "
            "(e.g. M=6 matches ACACACACACAC or AAAAAA... as (AA)^6). Use 0 to disable."
        ),
    )
    ap.add_argument("--upstream", type=int, default=DEFAULT_UPSTREAM)
    ap.add_argument("--downstream", type=int, default=DEFAULT_DOWNSTREAM)
    ap.add_argument(
        "--spacer-coordinate-offset",
        type=int,
        default=0,
        metavar="N",
        dest="spacer_coordinate_offset",
        help="Added to SpacerStart/SpacerEnd before template build (same as primer.py)",
    )
    ap.add_argument(
        "--ncbi-num-primer-pairs",
        type=int,
        default=DEFAULT_NCBI_NUM_PRIMER_PAIRS,
        dest="ncbi_num_primer_pairs",
        metavar="N",
        help=f"PRIMER_NUM_RETURN (default: {DEFAULT_NCBI_NUM_PRIMER_PAIRS})",
    )
    ap.add_argument("--organism", type=str, default=DEFAULT_NCBI_ORGANISM)
    ap.add_argument(
        "--ncbi-database",
        type=str,
        default=DEFAULT_NCBI_SPECIFICITY_DB,
        metavar="NAME",
        dest="ncbi_database",
    )
    ap.add_argument("--ncbi-product-min", type=int, default=NCBI_POST_PRODUCT_MIN, dest="ncbi_product_min")
    ap.add_argument("--ncbi-product-max", type=int, default=NCBI_POST_PRODUCT_MAX, dest="ncbi_product_max")
    ap.add_argument(
        "--fwd-max-past-spacer-end",
        type=int,
        default=FWD_MAX_PAST_SPACER_END,
        dest="fwd_max_past_spacer_end",
    )
    ap.add_argument(
        "--rev-min-past-spacer-end",
        type=int,
        default=REV_MIN_PAST_SPACER_END,
        dest="rev_min_past_spacer_end",
    )
    ap.add_argument("--dry-run", action="store_true", help="Rank and plan only; no NCBI calls")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--ncbi-minimal-post", action="store_true", dest="ncbi_minimal_post")
    ap.add_argument("--no-ncbi-post-params", action="store_true", dest="no_ncbi_post_params")
    ap.add_argument("--ncbi-html-dir", type=Path, default=None, metavar="DIR")
    args = ap.parse_args(argv)

    if args.number_top_picked < 0 or args.number_bottom_picked < 0 or args.number_random_picked < 0:
        raise SystemExit("pick counts must be >= 0")
    if int(args.min_consecutive_t_skip) < 0:
        raise SystemExit("--min-consecutive-t-skip must be >= 0")
    if int(args.min_dinucleotide_repeat_skip) < 0:
        raise SystemExit("--min-dinucleotide-repeat-skip must be >= 0")

    ranked = load_ranked_table(Path(args.input))
    gene_to_seq = load_gene_sequences(Path(args.fasta))

    primer_kw: dict[str, Any] = dict(
        upstream=int(args.upstream),
        downstream=int(args.downstream),
        spacer_coordinate_offset=int(args.spacer_coordinate_offset),
        per_spacer_template=True,
        fwd_max_past_spacer_end=int(args.fwd_max_past_spacer_end),
        rev_min_past_spacer_end=int(args.rev_min_past_spacer_end),
        ncbi_product_min=int(args.ncbi_product_min),
        ncbi_product_max=int(args.ncbi_product_max),
        ncbi_num_primer_pairs=int(args.ncbi_num_primer_pairs),
        specificity_database=str(args.ncbi_database),
        organism=str(args.organism),
        dry_run=bool(args.dry_run),
        verbose=bool(args.verbose),
        ncbi_browser_post=not bool(args.ncbi_minimal_post),
        log_ncbi_post=not bool(args.no_ncbi_post_params),
        ncbi_html_dir=args.ncbi_html_dir,
    )

    ranked_out, combined = run_picking(
        ranked,
        gene_to_seq,
        gene_filter=args.gene,
        number_top_picked=int(args.number_top_picked),
        number_bottom_picked=int(args.number_bottom_picked),
        number_random_picked=int(args.number_random_picked),
        seed=int(args.seed),
        dry_run=bool(args.dry_run),
        verbose=bool(args.verbose),
        primer_kw=primer_kw,
        min_consecutive_t_skip=int(args.min_consecutive_t_skip),
        min_dinucleotide_repeat_skip=int(args.min_dinucleotide_repeat_skip),
    )

    combined = finalize_picks_output(combined, ranked_out)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    combined.to_excel(args.output, index=False, engine="openpyxl")
    outp = Path(args.output).resolve()
    print(f"[picking] wrote {len(combined)} picked row(s) to {outp}", flush=True)

    if args.output_ranked is not None:
        ranked_export = format_gene_columns(ranked_out.copy(), ranked_out)
        rp = Path(args.output_ranked)
        rp.parent.mkdir(parents=True, exist_ok=True)
        if rp.suffix.lower() == ".csv":
            ranked_export.to_csv(rp, index=False)
        else:
            ranked_export.to_excel(rp, index=False, engine="openpyxl")
        print(
            f"[picking] wrote ranked table ({len(ranked_export)} rows, short Gene + total_spacers_gene) -> {rp.resolve()}",
            flush=True,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
