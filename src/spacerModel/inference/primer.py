#!/usr/bin/env python3
"""
Primer-BLAST helper for ranked spacer tables: by default **one** Primer-BLAST run per (Gene, Strand)
on the **full** FASTA record (minus strand: whole sequence reverse-complemented for the form), then
**filter** parsed pairs per spacer row. Legacy mode (``--per-spacer-template``) builds one template per
row: plus = clip the forward reference; minus = reverse-complement the **full** record then clip on
that RC (not ``RC(clip(forward))``). ``query_sequence_ncbi`` is the template DNA; ``ncbi_input_fasta``
is the full FASTA string POSTed to NCBI (use to match the browser paste).
"""

from __future__ import annotations

import argparse
import json
import re
import time
import zlib
from pathlib import Path
from collections.abc import Callable
from typing import Any
import pandas as pd
import requests

from spacerModel.inference.inference import (
    DEFAULT_SPACER_SEARCH_EDGE_RESERVE,
    parse_fasta_records,
    reverse_complement,
)

PRIMER_BLAST_BASE = "https://www.ncbi.nlm.nih.gov/tools/primer-blast/"
PRIMER_BLAST_TOOL = "https://www.ncbi.nlm.nih.gov/tools/primer-blast/primertool.cgi"

DEFAULT_UPSTREAM = 188
DEFAULT_DOWNSTREAM = 370
# Target amplicon length (client filters + NCBI POST PRIMER_PRODUCT_MIN/MAX so search is constrained up front)
PRODUCT_MIN = 150
PRODUCT_MAX = 350
NCBI_POST_PRODUCT_MIN = PRODUCT_MIN
NCBI_POST_PRODUCT_MAX = PRODUCT_MAX
DEFAULT_NCBI_SPECIFICITY_DB = "refseq_mrna"
DEFAULT_NCBI_ORGANISM = "Homo sapiens"
NCBI_MAX_TARGET_AMPLICON = 4000
# NCBI PRIMER_NUM_RETURN (web form: "# of primers to return").
DEFAULT_NCBI_NUM_PRIMER_PAIRS = 200
# Primer-BLAST appears to cap pairs per job at 50 (same as the web UI dropdown max); larger requests
# are ignored/clamped server-side, so parsed HTML usually has at most this many pairs.
NCBI_SERVER_MAX_PRIMER_PAIRS = 50
# Search mode (index.cgi ``SEARCHMODE``): 0=Automatic, 1=User guided, 2=No user guidance.
NCBI_SEARCHMODE = 0
# Exon junction span: 0=No preference, 1=must span exon-exon junction (``PRIMER_ON_SPLICE_SITE``).
NCBI_PRIMER_ON_SPLICE_SITE = 0
NCBI_DEFAULT_SPLICE_SITE_OVERLAP_5END = 7
NCBI_DEFAULT_SPLICE_SITE_OVERLAP_3END = 4
NCBI_DEFAULT_SPLICE_SITE_OVERLAP_3END_MAX = 8
NCBI_DEFAULT_MIN_INTRON_SIZE = 1000
# NCBI web UI often uses 1e6 here (matches a typical browser / DevTools ``primertool.cgi`` POST).
NCBI_DEFAULT_MAX_INTRON_SIZE = 1_000_000
# Specificity + BLAST tuning (primertool.cgi). Browser example used 1/1 for stricter transcript specificity.
NCBI_DEFAULT_TOTAL_MISMATCH = 1
NCBI_DEFAULT_3END_MISMATCH = 1
NCBI_DEFAULT_MISMATCH_REGION_LENGTH = 5
NCBI_DEFAULT_TOTAL_MISMATCH_IGNORE = 6
NCBI_DEFAULT_HITSIZE = 50000
NCBI_DEFAULT_EVALUE = 30000
NCBI_DEFAULT_WORD_SIZE = 7
NCBI_DEFAULT_PRIMER_MIN_SIZE = 15
NCBI_DEFAULT_PRIMER_OPT_SIZE = 20
NCBI_DEFAULT_PRIMER_MAX_SIZE = 25
NCBI_DEFAULT_PRIMER_MIN_TM = 57.0
NCBI_DEFAULT_PRIMER_OPT_TM = 60.0
NCBI_DEFAULT_PRIMER_MAX_TM = 63.0
NCBI_DEFAULT_PRIMER_MAX_DIFF_TM = 3.0
NCBI_DEFAULT_PRIMER_MISPRIMING_LIBRARY = "AUTO"
NCBI_DEFAULT_LOW_COMPLEXITY_FILTER = "on"
# Same name as in NCBI example URLs (often equals max target / product deviation in the UI).
NCBI_DEFAULT_PRODUCT_SIZE_DEVIATION = 4000
# Advanced parameters (``index.cgi`` defVal / defaults; Primer Pair Specificity + Primer Parameters).
NCBI_DEFAULT_MAX_CANDIDATE_PRIMER = 500
NCBI_DEFAULT_NUM_TARGETS = 20
NCBI_DEFAULT_NUM_TARGETS_WITH_PRIMERS = 1000
NCBI_DEFAULT_MAX_TARGET_PER_TEMPLATE = 100
NCBI_DEFAULT_PRIMER_MIN_GC = 20.0
NCBI_DEFAULT_PRIMER_MAX_GC = 80.0
NCBI_DEFAULT_GC_CLAMP = 0
NCBI_DEFAULT_POLYX = 5
NCBI_DEFAULT_PRIMER_MAX_END_STABILITY = 9
NCBI_DEFAULT_PRIMER_MAX_END_GC = 5
# Thermodynamic alignment limits (used if TH checkboxes are on; NCBI still posts these with defaults).
NCBI_DEFAULT_PRIMER_MAX_TEMPLATE_MISPRIMING_TH = 40.00
NCBI_DEFAULT_PRIMER_PAIR_MAX_TEMPLATE_MISPRIMING_TH = 70.00
NCBI_DEFAULT_PRIMER_MAX_SELF_ANY_TH = 45.0
NCBI_DEFAULT_PRIMER_MAX_SELF_END_TH = 35.0
NCBI_DEFAULT_PRIMER_PAIR_MAX_COMPL_ANY_TH = 45.0
NCBI_DEFAULT_PRIMER_PAIR_MAX_COMPL_END_TH = 35.0
NCBI_DEFAULT_PRIMER_MAX_HAIRPIN_TH = 24.0
# Old secondary-structure alignment model (default when thermodynamic alignment boxes are unchecked).
NCBI_DEFAULT_PRIMER_MAX_TEMPLATE_MISPRIMING = 12.00
NCBI_DEFAULT_PRIMER_PAIR_MAX_TEMPLATE_MISPRIMING = 24.00
NCBI_DEFAULT_SELF_ANY = 8.00
NCBI_DEFAULT_SELF_END = 3.00
NCBI_DEFAULT_PRIMER_PAIR_MAX_COMPL_ANY = 8.00
NCBI_DEFAULT_PRIMER_PAIR_MAX_COMPL_END = 3.00
# Reaction conditions / thermo tables (SantaLucia 1998: ``SALT_FORMULAR`` / ``TM_METHOD`` = 1).
NCBI_DEFAULT_MONO_CATIONS = 50.0
NCBI_DEFAULT_DIVA_CATIONS = 1.5
NCBI_DEFAULT_CON_DNTPS = 0.6
NCBI_DEFAULT_CON_ANEAL_OLIGO = 50.0
NCBI_DEFAULT_SALT_FORMULAR = 1
NCBI_DEFAULT_TM_METHOD = 1
# Exon overlap (duplicate names used by some NCBI forms alongside ``SPLICE_SITE_OVERLAP_*``).
NCBI_DEFAULT_OVERLAP_5END = 7
NCBI_DEFAULT_OVERLAP_3END = 4
# Browser / DevTools primertool.cgi examples (user-guided diff display).
NCBI_DEFAULT_NUM_DIFFS = 5
NCBI_DEFAULT_NUM_OPTS_DIFFS = 0
NCBI_DEFAULT_NEWWIN = "on"
NCBI_DEFAULT_SHOW_SVIEWER = "on"
NCBI_DEFAULT_UNGAPPED_BLAST = "on"
NCBI_DEFAULT_USER_TYPE = "2"
# Internal oligo defaults (same as NCBI form; ``PICK_HYB_PROBE`` stays off unless set).
NCBI_DEFAULT_PRIMER_INTERNAL_OLIGO_MIN_SIZE = 18
NCBI_DEFAULT_PRIMER_INTERNAL_OLIGO_OPT_SIZE = 20
NCBI_DEFAULT_PRIMER_INTERNAL_OLIGO_MAX_SIZE = 27
NCBI_DEFAULT_PRIMER_INTERNAL_OLIGO_MIN_TM = 57.0
NCBI_DEFAULT_PRIMER_INTERNAL_OLIGO_OPT_TM = 60.0
NCBI_DEFAULT_PRIMER_INTERNAL_OLIGO_MAX_TM = 63.0
NCBI_DEFAULT_PRIMER_INTERNAL_OLIGO_MIN_GC = 20.0
NCBI_DEFAULT_PRIMER_INTERNAL_OLIGO_OPT_GC_PERCENT = 50
NCBI_DEFAULT_PRIMER_INTERNAL_OLIGO_MAX_GC = 80.0
# Forward primer 3′ on top strand (``f_hi``) must not lie more than this many nt past **spacer end**:
# ``f_hi <= se + FWD_MAX_PAST_SPACER_END``. May be arbitrarily far 5′ (lower coordinates).
FWD_MAX_PAST_SPACER_END = 20
# Reverse primer 5′ on top strand (``r_lo``) must be ≥ this many nt **past spacer end** ``se``:
# ``r_lo >= se + REV_MIN_PAST_SPACER_END``.
REV_MIN_PAST_SPACER_END = 130
NCBI_DELAY_SEC = 0.35
POLL_INTERVAL_SEC = 3.0
POLL_MAX_WAIT_SEC = 600.0
# NCBI often drops idle TLS connections (RemoteDisconnected); retry with backoff.
NCBI_HTTP_MAX_RETRIES = 6
NCBI_HTTP_RETRY_BASE_SEC = 5.0
NCBI_HTTP_RETRY_MAX_SLEEP_SEC = 120.0
NCBI_GET_TIMEOUT_SEC = 90
NCBI_POST_TIMEOUT_SEC = 300

USER_AGENT = "evoCAST-spacer-model/1.0 (primer.py; institutional research)"

# Shown with --help; client-side filters still differ from the raw NCBI table.
NCBI_BROWSER_VS_SCRIPT_HELP = """
NCBI POST body:
  • Default POST matches the expanded Primer-BLAST form (Tm, sizes, BLAST tuning, exon/intron,
    advanced Primer3 constraints, salt/thermo tables, specificity display limits, …), aligned with
    NCBI ``index.cgi`` defaults. Use --ncbi-minimal-post only for a smaller request (e.g. connection
    issues); primer quality may differ from the browser.
  • After NCBI returns, only pairs passing pair_matches_design_rules() are kept.
  • Typical product range: %d–%d bp (same as client filter).
""" % (NCBI_POST_PRODUCT_MIN, NCBI_POST_PRODUCT_MAX)


def load_gene_sequences(fasta: Path) -> dict[str, str]:
    """Map exact FASTA header (``Gene`` column) to uppercase sequence."""
    return {h: s for h, s in parse_fasta_records(fasta)}


def gene_matches_filter(gene_cell: str, want: str) -> bool:
    """True if ``want`` matches the full ``Gene`` string or its substring before the first ``_`` (case-insensitive)."""
    w = want.strip()
    if not w:
        return True
    g = str(gene_cell).strip()
    if g.upper() == w.upper():
        return True
    head = g.split("_", 1)[0].strip()
    return head.upper() == w.upper()


def resolve_fasta_for_gene(
    gene_cell: str, gene_to_seq: dict[str, str], *, verbose: bool = False, log_prefix: str = ""
) -> tuple[str, str] | None:
    """
    Map a table ``Gene`` cell to a FASTA header key and sequence.

    Tries exact header match first, then :func:`gene_matches_filter` so short names (e.g. ``RHO``)
    match long headers (e.g. ``RHO_hg19_...``).
    """
    g = str(gene_cell).strip()
    if not g:
        return None
    if g in gene_to_seq:
        return g, gene_to_seq[g]
    matches: list[tuple[str, str]] = []
    for header, seq in gene_to_seq.items():
        if gene_matches_filter(header, g):
            matches.append((header, seq))
    if not matches:
        return None
    if len(matches) > 1 and verbose:
        keys = [m[0][:80] for m in matches[:5]]
        print(
            f"{log_prefix}multiple FASTA headers match Gene={g!r} (using first): {keys!r}",
            flush=True,
        )
    return matches[0]


def build_primer_blast_query_sequence(
    seq: str,
    spacer_start_1b: int,
    spacer_end_1b: int,
    strand: str,
    *,
    upstream: int = DEFAULT_UPSTREAM,
    downstream: int = DEFAULT_DOWNSTREAM,
    precomputed_rc_full: str | None = None,
) -> tuple[str, int, int] | None:
    """
    Build the PCR template string sent to NCBI (5′→3′ on the targeting strand).

    **Plus:** clip ``[SpacerStart−upstream, SpacerEnd+downstream]`` on the forward reference.

    **Minus:** reverse-complement the **full** reference, map ``[SpacerStart, SpacerEnd]`` to that
    string, then clip ``[a−upstream, b+downstream]`` on the RC (``a,b`` = spacer interval 5′→3′ on
    the targeting strand). This is **not** the same as clipping the forward window and then RCing
    the clip.

    ``precomputed_rc_full``: for minus strand only, optional ``reverse_complement(seq)`` (same length
    as ``seq``) to avoid recomputing RC for every row of the same gene.

    Returns ``(dna, spacer_start_1b, spacer_end_1b)`` — 1-based inclusive spacer coordinates on
    ``dna`` (the posted query).
    """
    seq_u = seq.strip().upper()
    n = len(seq_u)
    ss = int(spacer_start_1b)
    se = int(spacer_end_1b)
    if ss < 1 or se > n or ss > se:
        return None

    st = str(strand).strip()
    if st in ("-", "-1", "minus", "reverse"):
        # Full-length RC; spacer [ss,se] on forward -> [n-se+1, n-ss+1] on T (5'->3' on minus strand).
        if precomputed_rc_full is not None:
            big_t = precomputed_rc_full.strip().upper()
            if len(big_t) != n:
                return None
        else:
            big_t = reverse_complement(seq_u)
        a = n - se + 1
        b = n - ss + 1
        if a < 1 or b > n or a > b:
            return None
        w_lo = max(1, a - upstream)
        w_hi = min(n, b + downstream)
        if w_lo > w_hi or w_hi - w_lo + 1 < 50:
            return None
        template = big_t[w_lo - 1 : w_hi]
        s1 = a - w_lo + 1
        s2 = b - w_lo + 1
        if s1 < 1 or s2 > len(template) or s1 > s2:
            return None
        return template, s1, s2

    lo = max(1, ss - upstream)
    hi = min(n, se + downstream)
    if lo > hi or hi - lo + 1 < 50:
        return None
    template_fwd = seq_u[lo - 1 : hi]
    L = len(template_fwd)
    sf = ss - lo + 1
    se_w = se - lo + 1
    if sf < 1 or se_w > L or sf > se_w:
        return None
    return template_fwd, sf, se_w


def full_sequence_blast_template(seq: str, strand: str) -> str:
    """Full FASTA record oriented for Primer-BLAST (same rules as window mode: RC whole sequence if minus)."""
    s = seq.strip().upper()
    st = str(strand).strip()
    if st in ("-", "-1", "minus", "reverse"):
        return reverse_complement(s)
    return s


def spacer_gc_content_on_ref(ref_seq: str, spacer_start_1b: int, spacer_end_1b: int) -> float | None:
    """
    Fraction of G+C in ``ref_seq[spacer_start_1b-1 : spacer_end_1b]`` (1-based, inclusive).
    Same for +/− targeting because RC preserves G+C count. Returns ``None`` if coordinates invalid.
    """
    s = ref_seq.strip().upper()
    n = len(s)
    ss = int(spacer_start_1b)
    se = int(spacer_end_1b)
    if ss < 1 or se > n or ss > se:
        return None
    frag = s[ss - 1 : se]
    if not frag:
        return None
    gc = sum(1 for c in frag if c in "GC")
    return gc / len(frag)


def add_spacer_gc_column(
    df: pd.DataFrame,
    gene_to_seq: dict[str, str],
    *,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Add ``spacer_GC_content`` (G+C fraction) on the reference for each row.

    Requires ``Gene``, ``SpacerStart``, ``SpacerEnd`` (1-based on the full FASTA record).
    Uses :func:`resolve_fasta_for_gene` and :func:`spacer_gc_content_on_ref`.
    """
    if df.empty:
        return df
    out = df.copy()
    gcs: list[float | None] = []
    for _, row in out.iterrows():
        resolved = resolve_fasta_for_gene(
            str(row["Gene"]), gene_to_seq, verbose=verbose, log_prefix=""
        )
        if resolved is None:
            gcs.append(None)
            continue
        _, seq = resolved
        ss = int(row["SpacerStart"])
        se = int(row["SpacerEnd"])
        gcs.append(spacer_gc_content_on_ref(seq, ss, se))
    out["spacer_GC_content"] = gcs
    return out


def spacer_coords_on_blast_template(
    ref_seq: str, spacer_start_1b: int, spacer_end_1b: int, strand: str
) -> tuple[int, int] | None:
    """
    Map spacer [ss, se] (1-based on ``ref_seq``) to coordinates on :func:`full_sequence_blast_template`
    output (same string length as ``ref_seq``).
    """
    n = len(ref_seq.strip())
    ss = int(spacer_start_1b)
    se = int(spacer_end_1b)
    if ss < 1 or se > n or ss > se:
        return None
    st = str(strand).strip()
    if st in ("-", "-1", "minus", "reverse"):
        return n - se + 1, n - ss + 1
    return ss, se


def _amplicon_length_from_pair(pair: dict[str, Any]) -> int | None:
    """PCR product size (bp) from NCBI ``product_len``, or from spans if missing."""
    plen = pair.get("product_len")
    if plen is not None:
        return int(plen)
    fs, rs = pair.get("fwd_span"), pair.get("rev_span")
    if not fs or not rs:
        return None
    f_hi = max(fs)
    r_lo = min(rs)
    if r_lo <= f_hi:
        return None
    return r_lo - f_hi - 1


def _pair_to_json_serializable(pair: dict[str, Any]) -> dict[str, Any]:
    d: dict[str, Any] = {}
    for k, v in pair.items():
        if isinstance(v, tuple) and len(v) == 2:
            d[k] = [int(v[0]), int(v[1])]
        else:
            d[k] = v
    return d


def matching_primer_pairs(
    pairs: list[dict[str, Any]],
    spacer_start_1b: int,
    spacer_end_1b: int,
    *,
    fwd_max_past_spacer_end: int = FWD_MAX_PAST_SPACER_END,
    rev_min_past_spacer_end: int = REV_MIN_PAST_SPACER_END,
) -> list[dict[str, Any]]:
    """All pairs that satisfy placement and product rules (not just the first)."""
    out: list[dict[str, Any]] = []
    for p in pairs:
        if pair_matches_design_rules(
            p,
            spacer_start_1b,
            spacer_end_1b,
            fwd_max_past_spacer_end=fwd_max_past_spacer_end,
            rev_min_past_spacer_end=rev_min_past_spacer_end,
        ):
            out.append(p)
    return out


def _parse_rid(html: str) -> str | None:
    """Return a plausible job id; ignore placeholders like ``--``."""
    candidates: list[str] = []
    for rx in (
        r'RID\s*=\s*["\']?([A-Za-z0-9_\-]{8,})',
        r'name="RID"\s+value="([^"]+)"',
        r"RID%3D([A-Za-z0-9_\-]{8,})",
    ):
        for m in re.finditer(rx, html):
            candidates.append(m.group(1).strip())
    for rid in candidates:
        if rid in ("--", "-", "") or rid.startswith("-"):
            continue
        if len(rid) >= 8 and re.match(r"^[A-Za-z0-9_\-]+$", rid):
            return rid
    return None


def _parse_meta_refresh_url(html: str) -> str | None:
    """
    NCBI often returns a short stub with ``META HTTP-EQUIV=Refresh`` to
    ``primertool.cgi?ctg_time=...&job_key=...`` instead of embedding ``RID`` in the body.
    """
    m = re.search(
        r'http-equiv\s*=\s*["\']?Refresh["\']?\s+content\s*=\s*["\']?\s*\d+\s*;\s*URL=([^"\'>\s]+)',
        html,
        re.I,
    )
    if m:
        return m.group(1).strip()
    m = re.search(r'CONTENT\s*=\s*["\']?\s*\d+\s*;\s*URL=([^"\'>\s]+)', html, re.I)
    if m:
        return m.group(1).strip()
    return None


def _status_waiting(html: str) -> bool:
    return bool(re.search(r"Status=\s*WAITING|still running|Please wait|job is queued", html, re.I))


def _primer_blast_finished(html: str) -> bool:
    """True when the job page has primer rows or a definitive no-primers outcome (not a wait stub)."""
    if _status_waiting(html):
        return False
    if re.search(r"Forward\s+primer\s+[ACGTacgt]{10,}", html, re.I):
        return True
    if re.search(r"No primers were found", html, re.I):
        return True
    if re.search(
        r"Primer pair\s*\d+.*?(?:Forward\s+primer|Oligo\s*\(\s*\d+\s+bp\s*\))",
        html,
        re.I | re.DOTALL,
    ):
        return True
    return False


def _status_ready(html: str) -> bool:
    """Compatibility alias for logging; prefer :func:`_primer_blast_finished`."""
    return _primer_blast_finished(html)


def build_ncbi_primer_blast_post_data(
    template_fasta: str,
    *,
    ncbi_product_min: int = NCBI_POST_PRODUCT_MIN,
    ncbi_product_max: int = NCBI_POST_PRODUCT_MAX,
    ncbi_num_primer_pairs: int = DEFAULT_NCBI_NUM_PRIMER_PAIRS,
    specificity_database: str = DEFAULT_NCBI_SPECIFICITY_DB,
    organism: str = DEFAULT_NCBI_ORGANISM,
    minimal: bool = False,
) -> dict[str, str]:
    """
    Form body for ``primertool.cgi``.

    **minimal=False** (default): send the same major fields as the Primer-BLAST web form (Tm, primer
    size/GC, specificity mismatches, BLAST E/word/hitsize, candidate-primer and target-display limits,
    old + thermodynamic secondary-structure limits, salt/thermo tables, exon/intron knobs, etc.) so
    behavior aligns with NCBI ``index.cgi`` defaults.

    **minimal=True**: only a handful of keys (product size, db, organism, …). The server then picks
    internal defaults for everything else — these are **not** documented to match the UI’s displayed
    defaults, so primer sets can differ a lot from the browser. Use only if you need a smaller POST.
    """
    nret = int(ncbi_num_primer_pairs)
    if minimal:
        return {
            "CMD": "request",
            "INPUT_SEQUENCE": template_fasta,
            "PRIMER_PRODUCT_MIN": str(int(ncbi_product_min)),
            "PRIMER_PRODUCT_MAX": str(int(ncbi_product_max)),
            "PRIMER_SPECIFICITY_DATABASE": specificity_database,
            "SEARCH_SPECIFIC_PRIMER": "on",
            "PRIMER_NUM_RETURN": str(nret),
            "TOTAL_PRIMER_SPECIFICITY_MISMATCH": str(NCBI_DEFAULT_TOTAL_MISMATCH),
            "ORGANISM": organism,
            "MAX_TARGET_SIZE": str(NCBI_MAX_TARGET_AMPLICON),
        }
    return {
        "CMD": "request",
        "INPUT_SEQUENCE": template_fasta,
        "SEARCHMODE": str(NCBI_SEARCHMODE),
        "PRIMER_PRODUCT_MIN": str(int(ncbi_product_min)),
        "PRIMER_PRODUCT_MAX": str(int(ncbi_product_max)),
        "PRIMER_SPECIFICITY_DATABASE": specificity_database,
        "SEARCH_SPECIFIC_PRIMER": "on",
        "PRIMER_NUM_RETURN": str(nret),
        "ORGANISM": organism,
        "TOTAL_PRIMER_SPECIFICITY_MISMATCH": str(NCBI_DEFAULT_TOTAL_MISMATCH),
        "PRIMER_3END_SPECIFICITY_MISMATCH": str(NCBI_DEFAULT_3END_MISMATCH),
        "MISMATCH_REGION_LENGTH": str(NCBI_DEFAULT_MISMATCH_REGION_LENGTH),
        "TOTAL_MISMATCH_IGNORE": str(NCBI_DEFAULT_TOTAL_MISMATCH_IGNORE),
        "MAX_TARGET_SIZE": str(NCBI_MAX_TARGET_AMPLICON),
        "PRODUCT_SIZE_DEVIATION": str(NCBI_DEFAULT_PRODUCT_SIZE_DEVIATION),
        "OVERLAP_5END": str(NCBI_DEFAULT_OVERLAP_5END),
        "OVERLAP_3END": str(NCBI_DEFAULT_OVERLAP_3END),
        "NUM_DIFFS": str(NCBI_DEFAULT_NUM_DIFFS),
        "NUM_OPTS_DIFFS": str(NCBI_DEFAULT_NUM_OPTS_DIFFS),
        "NEWWIN": NCBI_DEFAULT_NEWWIN,
        "SHOW_SVIEWER": NCBI_DEFAULT_SHOW_SVIEWER,
        "UNGAPPED_BLAST": NCBI_DEFAULT_UNGAPPED_BLAST,
        "USER_TYPE": NCBI_DEFAULT_USER_TYPE,
        "PRIMER_INTERNAL_OLIGO_MIN_SIZE": str(NCBI_DEFAULT_PRIMER_INTERNAL_OLIGO_MIN_SIZE),
        "PRIMER_INTERNAL_OLIGO_OPT_SIZE": str(NCBI_DEFAULT_PRIMER_INTERNAL_OLIGO_OPT_SIZE),
        "PRIMER_INTERNAL_OLIGO_MAX_SIZE": str(NCBI_DEFAULT_PRIMER_INTERNAL_OLIGO_MAX_SIZE),
        "PRIMER_INTERNAL_OLIGO_MIN_TM": str(NCBI_DEFAULT_PRIMER_INTERNAL_OLIGO_MIN_TM),
        "PRIMER_INTERNAL_OLIGO_OPT_TM": str(NCBI_DEFAULT_PRIMER_INTERNAL_OLIGO_OPT_TM),
        "PRIMER_INTERNAL_OLIGO_MAX_TM": str(NCBI_DEFAULT_PRIMER_INTERNAL_OLIGO_MAX_TM),
        "PRIMER_INTERNAL_OLIGO_MIN_GC": str(NCBI_DEFAULT_PRIMER_INTERNAL_OLIGO_MIN_GC),
        "PRIMER_INTERNAL_OLIGO_OPT_GC_PERCENT": str(NCBI_DEFAULT_PRIMER_INTERNAL_OLIGO_OPT_GC_PERCENT),
        "PRIMER_INTERNAL_OLIGO_MAX_GC": str(NCBI_DEFAULT_PRIMER_INTERNAL_OLIGO_MAX_GC),
        "HITSIZE": str(NCBI_DEFAULT_HITSIZE),
        "EVALUE": str(NCBI_DEFAULT_EVALUE),
        "WORD_SIZE": str(NCBI_DEFAULT_WORD_SIZE),
        "PRIMER_MIN_SIZE": str(NCBI_DEFAULT_PRIMER_MIN_SIZE),
        "PRIMER_OPT_SIZE": str(NCBI_DEFAULT_PRIMER_OPT_SIZE),
        "PRIMER_MAX_SIZE": str(NCBI_DEFAULT_PRIMER_MAX_SIZE),
        "PRIMER_MIN_TM": str(NCBI_DEFAULT_PRIMER_MIN_TM),
        "PRIMER_OPT_TM": str(NCBI_DEFAULT_PRIMER_OPT_TM),
        "PRIMER_MAX_TM": str(NCBI_DEFAULT_PRIMER_MAX_TM),
        "PRIMER_MAX_DIFF_TM": str(NCBI_DEFAULT_PRIMER_MAX_DIFF_TM),
        "PRIMER_MISPRIMING_LIBRARY": NCBI_DEFAULT_PRIMER_MISPRIMING_LIBRARY,
        "LOW_COMPLEXITY_FILTER": NCBI_DEFAULT_LOW_COMPLEXITY_FILTER,
        "PRIMER_ON_SPLICE_SITE": str(NCBI_PRIMER_ON_SPLICE_SITE),
        "SPLICE_SITE_OVERLAP_5END": str(NCBI_DEFAULT_SPLICE_SITE_OVERLAP_5END),
        "SPLICE_SITE_OVERLAP_3END": str(NCBI_DEFAULT_SPLICE_SITE_OVERLAP_3END),
        "SPLICE_SITE_OVERLAP_3END_MAX": str(NCBI_DEFAULT_SPLICE_SITE_OVERLAP_3END_MAX),
        "MIN_INTRON_SIZE": str(NCBI_DEFAULT_MIN_INTRON_SIZE),
        "MAX_INTRON_SIZE": str(NCBI_DEFAULT_MAX_INTRON_SIZE),
        "MAX_CANDIDATE_PRIMER": str(NCBI_DEFAULT_MAX_CANDIDATE_PRIMER),
        "NUM_TARGETS": str(NCBI_DEFAULT_NUM_TARGETS),
        "NUM_TARGETS_WITH_PRIMERS": str(NCBI_DEFAULT_NUM_TARGETS_WITH_PRIMERS),
        "MAX_TARGET_PER_TEMPLATE": str(NCBI_DEFAULT_MAX_TARGET_PER_TEMPLATE),
        "PRIMER_MIN_GC": str(NCBI_DEFAULT_PRIMER_MIN_GC),
        "PRIMER_MAX_GC": str(NCBI_DEFAULT_PRIMER_MAX_GC),
        "GC_CLAMP": str(NCBI_DEFAULT_GC_CLAMP),
        "POLYX": str(NCBI_DEFAULT_POLYX),
        "PRIMER_MAX_END_STABILITY": str(NCBI_DEFAULT_PRIMER_MAX_END_STABILITY),
        "PRIMER_MAX_END_GC": str(NCBI_DEFAULT_PRIMER_MAX_END_GC),
        "PRIMER_MAX_TEMPLATE_MISPRIMING_TH": str(NCBI_DEFAULT_PRIMER_MAX_TEMPLATE_MISPRIMING_TH),
        "PRIMER_PAIR_MAX_TEMPLATE_MISPRIMING_TH": str(NCBI_DEFAULT_PRIMER_PAIR_MAX_TEMPLATE_MISPRIMING_TH),
        "PRIMER_MAX_SELF_ANY_TH": str(NCBI_DEFAULT_PRIMER_MAX_SELF_ANY_TH),
        "PRIMER_MAX_SELF_END_TH": str(NCBI_DEFAULT_PRIMER_MAX_SELF_END_TH),
        "PRIMER_PAIR_MAX_COMPL_ANY_TH": str(NCBI_DEFAULT_PRIMER_PAIR_MAX_COMPL_ANY_TH),
        "PRIMER_PAIR_MAX_COMPL_END_TH": str(NCBI_DEFAULT_PRIMER_PAIR_MAX_COMPL_END_TH),
        "PRIMER_MAX_HAIRPIN_TH": str(NCBI_DEFAULT_PRIMER_MAX_HAIRPIN_TH),
        "PRIMER_MAX_TEMPLATE_MISPRIMING": str(NCBI_DEFAULT_PRIMER_MAX_TEMPLATE_MISPRIMING),
        "PRIMER_PAIR_MAX_TEMPLATE_MISPRIMING": str(NCBI_DEFAULT_PRIMER_PAIR_MAX_TEMPLATE_MISPRIMING),
        "SELF_ANY": str(NCBI_DEFAULT_SELF_ANY),
        "SELF_END": str(NCBI_DEFAULT_SELF_END),
        "PRIMER_PAIR_MAX_COMPL_ANY": str(NCBI_DEFAULT_PRIMER_PAIR_MAX_COMPL_ANY),
        "PRIMER_PAIR_MAX_COMPL_END": str(NCBI_DEFAULT_PRIMER_PAIR_MAX_COMPL_END),
        "MONO_CATIONS": str(NCBI_DEFAULT_MONO_CATIONS),
        "DIVA_CATIONS": str(NCBI_DEFAULT_DIVA_CATIONS),
        "CON_DNTPS": str(NCBI_DEFAULT_CON_DNTPS),
        "CON_ANEAL_OLIGO": str(NCBI_DEFAULT_CON_ANEAL_OLIGO),
        "SALT_FORMULAR": str(NCBI_DEFAULT_SALT_FORMULAR),
        "TM_METHOD": str(NCBI_DEFAULT_TM_METHOD),
    }


def _request_with_retries(
    do_request: Callable[[], requests.Response],
    *,
    log_prefix: str,
    what: str,
    max_attempts: int = NCBI_HTTP_MAX_RETRIES,
    base_sleep_sec: float = NCBI_HTTP_RETRY_BASE_SEC,
) -> requests.Response:
    """
    Retry transient HTTP failures (NCBI often closes the socket with ``RemoteDisconnected`` on POST).
    """
    transient = (
        requests.exceptions.ConnectionError,
        requests.exceptions.Timeout,
        requests.exceptions.ChunkedEncodingError,
    )
    for attempt in range(1, max_attempts + 1):
        try:
            resp = do_request()
            resp.raise_for_status()
            return resp
        except transient as e:
            if attempt >= max_attempts:
                print(f"{log_prefix}{what} failed after {max_attempts} attempts: {e!r}", flush=True)
                raise
            wait = min(base_sleep_sec * (2 ** (attempt - 1)), NCBI_HTTP_RETRY_MAX_SLEEP_SEC)
            print(
                f"{log_prefix}{what} transient error (attempt {attempt}/{max_attempts}): {e!r}; "
                f"sleep {wait:.1f}s and retry",
                flush=True,
            )
            time.sleep(wait)
    raise RuntimeError("_request_with_retries: logic error")


def _fasta_template_dna_bp(template_fasta: str) -> int | None:
    """Length of template DNA only (excludes ``>header`` line); None if empty."""
    lines = [ln.strip() for ln in template_fasta.strip().splitlines() if ln.strip()]
    if not lines:
        return None
    if lines[0].startswith(">"):
        return len("".join(lines[1:]))
    return len("".join(lines))


def _log_ncbi_post_body(data: dict[str, Any], *, log_prefix: str) -> None:
    """Print sorted CGI fields for diffing against browser DevTools → Network → primertool.cgi Form Data."""
    lines: list[str] = []
    for k in sorted(data.keys(), key=str):
        v = data[k]
        if k == "INPUT_SEQUENCE":
            s = str(v).strip()
            nlines = s.splitlines()
            hdr = nlines[0][:120] if nlines else ""
            body = "".join(nlines[1:]) if len(nlines) > 1 else ""
            lines.append(f"{k}={hdr!r} + sequence_len={len(body)}")
        else:
            lines.append(f"{k}={v!r}")
    print(f"{log_prefix}NCBI POST body ({len(data)} keys):\n" + "\n".join(lines), flush=True)


def _write_ncbi_result_html(
    html: str,
    dump_dir: Path,
    stem: str,
    *,
    log_prefix: str,
) -> Path:
    """
    Write final Primer-BLAST result page for troubleshooting (diff vs browser save).
    Filename uses RID from HTML when present, else a short hash.
    """
    dump_dir = Path(dump_dir)
    dump_dir.mkdir(parents=True, exist_ok=True)
    rid = _parse_rid(html)
    if rid:
        tag = re.sub(r"[^\w.\-]+", "_", str(rid).strip())[:120]
    else:
        tag = f"h{zlib.adler32(html.encode('utf-8', errors='replace')) & 0xFFFFFFFF:08x}"
    base = re.sub(r"[^\w.\-]+", "_", str(stem).strip())[:180] or "primer_blast"
    path = dump_dir / f"{base}_{tag}.html"
    path.write_text(html, encoding="utf-8")
    print(
        f"{log_prefix}saved NCBI result HTML ({len(html)} chars) -> {path.resolve()}",
        flush=True,
    )
    return path


def submit_primer_blast(
    template_fasta: str,
    *,
    ncbi_product_min: int = NCBI_POST_PRODUCT_MIN,
    ncbi_product_max: int = NCBI_POST_PRODUCT_MAX,
    ncbi_num_primer_pairs: int = DEFAULT_NCBI_NUM_PRIMER_PAIRS,
    specificity_database: str = DEFAULT_NCBI_SPECIFICITY_DB,
    organism: str = DEFAULT_NCBI_ORGANISM,
    session: requests.Session | None = None,
    verbose: bool = False,
    log_prefix: str = "",
    post_overrides: dict[str, str] | None = None,
    ncbi_browser_post: bool = True,
    log_ncbi_post: bool = True,
    ncbi_html_dir: Path | None = None,
    ncbi_html_stem: str = "primer_blast",
) -> str:
    """
    POST template (FASTA) to NCBI Primer-BLAST.

    By default sends the **expanded** form (``minimal=False``) to match the web UI. Set
    ``ncbi_browser_post=False`` (CLI: ``--ncbi-minimal-post``) only for a smaller POST body.
    ``post_overrides`` can replace or add CGI keys.

    By default prints the full CGI form (``log_ncbi_post=True``) for diffing against the browser;
    set ``log_ncbi_post=False`` to skip (CLI: ``--no-ncbi-post-params``).

    If ``ncbi_html_dir`` is set, writes the final result page HTML to that directory using
    ``ncbi_html_stem`` and the job RID in the filename (CLI: ``--ncbi-html-dir``).
    """
    sess = session or requests.Session()
    headers = {"User-Agent": USER_AGENT}
    print(f"{log_prefix}GET {PRIMER_BLAST_BASE} (session cookie)", flush=True)
    r0 = _request_with_retries(
        lambda: sess.get(PRIMER_BLAST_BASE, headers=headers, timeout=NCBI_GET_TIMEOUT_SEC),
        log_prefix=log_prefix,
        what="GET primer-blast landing page",
    )
    print(f"{log_prefix}GET primer-blast landing status={r0.status_code}", flush=True)

    data: dict[str, Any] = build_ncbi_primer_blast_post_data(
        template_fasta,
        ncbi_product_min=ncbi_product_min,
        ncbi_product_max=ncbi_product_max,
        ncbi_num_primer_pairs=ncbi_num_primer_pairs,
        specificity_database=specificity_database,
        organism=organism,
        minimal=not ncbi_browser_post,
    )
    if post_overrides:
        data.update(post_overrides)

    if log_ncbi_post:
        _log_ncbi_post_body(data, log_prefix=log_prefix)

    dna_bp = _fasta_template_dna_bp(template_fasta)
    dna_note = f"template_DNA={dna_bp} bp" if dna_bp is not None else "template_DNA=?"
    omit_note = "" if log_ncbi_post else " | CGI key=value list omitted (--no-ncbi-post-params)"
    print(
        f"{log_prefix}POST {PRIMER_BLAST_TOOL} INPUT_SEQUENCE={len(template_fasta)} chars FASTA "
        f"({dna_note}) ncbi_product={ncbi_product_min}-{ncbi_product_max} "
        f"PRIMER_NUM_RETURN={ncbi_num_primer_pairs} db={specificity_database!r} organism={organism!r} "
        f"post={'expanded' if ncbi_browser_post else 'minimal'}{omit_note}",
        flush=True,
    )
    if verbose:
        lines = template_fasta.strip().splitlines()
        preview = (lines[1][:120] + "...") if len(lines) > 1 and len(lines[1]) > 120 else template_fasta[:200]
        print(f"{log_prefix}  sequence preview: {preview!r}", flush=True)

    r = _request_with_retries(
        lambda: sess.post(
            PRIMER_BLAST_TOOL,
            data=data,
            headers=headers,
            timeout=NCBI_POST_TIMEOUT_SEC,
        ),
        log_prefix=log_prefix,
        what="POST primer-blast job",
    )
    html = r.text
    print(f"{log_prefix}POST response status={r.status_code} html_chars={len(html)}", flush=True)
    rid = _parse_rid(html)
    refresh_url = _parse_meta_refresh_url(html)
    if rid:
        print(
            f"{log_prefix}job RID={rid!r} waiting={_status_waiting(html)} "
            f"finished={_primer_blast_finished(html)}",
            flush=True,
        )
    if refresh_url:
        ru_show = refresh_url if len(refresh_url) <= 120 else refresh_url[:117] + "..."
        print(f"{log_prefix}META refresh job URL={ru_show!r}", flush=True)
    if not rid and not refresh_url:
        print(f"{log_prefix}no RID and no META refresh URL (sync results or unexpected layout)", flush=True)

    polls = 0
    deadline = time.time() + POLL_MAX_WAIT_SEC

    # Current NCBI: POST returns a ~21k stub with Refresh → job_key URL; results only after GET that URL.
    if refresh_url and not _primer_blast_finished(html):
        while time.time() < deadline:
            time.sleep(POLL_INTERVAL_SEC)
            polls += 1
            r2 = _request_with_retries(
                lambda: sess.get(refresh_url, headers=headers, timeout=NCBI_POST_TIMEOUT_SEC),
                log_prefix=log_prefix,
                what=f"GET poll refresh_url (poll {polls})",
            )
            html = r2.text
            print(
                f"{log_prefix}poll {polls} GET refresh_url status={r2.status_code} html_chars={len(html)} "
                f"waiting={_status_waiting(html)} finished={_primer_blast_finished(html)}",
                flush=True,
            )
            if _primer_blast_finished(html):
                break
    elif rid and len(rid) >= 8 and not _primer_blast_finished(html):
        while time.time() < deadline:
            time.sleep(POLL_INTERVAL_SEC)
            polls += 1
            r2 = _request_with_retries(
                lambda: sess.get(
                    PRIMER_BLAST_TOOL,
                    params={"CMD": "Get", "RID": rid},
                    headers=headers,
                    timeout=NCBI_POST_TIMEOUT_SEC,
                ),
                log_prefix=log_prefix,
                what=f"GET poll RID (poll {polls})",
            )
            html = r2.text
            print(
                f"{log_prefix}poll {polls} GET RID status={r2.status_code} html_chars={len(html)} "
                f"waiting={_status_waiting(html)} finished={_primer_blast_finished(html)}",
                flush=True,
            )
            if _primer_blast_finished(html):
                break

    if verbose and html and len(html) < 800:
        print(f"{log_prefix}full html:\n{html}", flush=True)
    elif "error" in html[:2000].lower() or "Error occurred" in html:
        snippet = html[:1200].replace("\n", " ")
        print(f"{log_prefix}possible error in HTML (snippet): {snippet!r}", flush=True)

    if ncbi_html_dir is not None:
        _write_ncbi_result_html(html, Path(ncbi_html_dir), ncbi_html_stem, log_prefix=log_prefix)

    return html


def _html_to_plain(s: str) -> str:
    """Strip tags/scripts so NCBI table rows become searchable text (Detailed primer reports)."""
    t = re.sub(r"<script[^>]*>[\s\S]*?</script>", " ", s, flags=re.I)
    t = re.sub(r"<style[^>]*>[\s\S]*?</style>", " ", t, flags=re.I)
    t = re.sub(r"<[^>]+>", " ", t)
    t = re.sub(r"\s+", " ", t)
    return t.strip()


def _parse_detailed_report_block(block_html: str) -> dict[str, Any] | None:
    """
    Parse NCBI 'Detailed primer reports' blocks: Forward/Reverse rows with Start/Stop (Plus strand;
    minus-strand reverse may have Start > Stop).

    Plain text after tag-stripping is often::

        Forward primer TGGATCACCACTCCGGATCT Plus 20 668 687 60.03 ...
        Reverse primer CGGAGCCTTCTGTTGGGAAT Minus 20 753 734 ...

    Older pages used ``Start:`` / ``Stop:`` labels instead.
    """
    block = _html_to_plain(block_html)

    fp = re.search(
        r"Forward\s+primer\s+([ACGTacgt]{10,})\s+Plus\s+\d+\s+(\d+)\s+(\d+)",
        block,
        re.I,
    )
    rp = re.search(
        r"Reverse\s+primer\s+([ACGTacgt]{10,})\s+Minus\s+\d+\s+(\d+)\s+(\d+)",
        block,
        re.I,
    )
    if not fp or not rp:
        fp = re.search(
            r"Forward\s+primer\s+([ACGTacgt]{10,}).*?Start:?\s*(\d+).*?Stop:?\s*(\d+)",
            block,
            re.I | re.DOTALL,
        )
        rp = re.search(
            r"Reverse\s+primer\s+([ACGTacgt]{10,}).*?Start:?\s*(\d+).*?Stop:?\s*(\d+)",
            block,
            re.I | re.DOTALL,
        )
    if not fp or not rp:
        return None
    fa, fb = int(fp.group(2)), int(fp.group(3))
    ra, rb = int(rp.group(2)), int(rp.group(3))
    fwd_span = (min(fa, fb), max(fa, fb))
    rev_span = (min(ra, rb), max(ra, rb))
    fwd = fp.group(1).upper()
    rev = rp.group(1).upper()
    plen: int | None = None
    plm = re.search(r"Product\s+length\s*:?\s*(\d+)", block, re.I)
    if plm:
        plen = int(plm.group(1))
    return {
        "forward_seq": fwd,
        "reverse_seq": rev,
        "product_len": plen,
        "fwd_span": fwd_span,
        "rev_span": rev_span,
    }


def _parse_legacy_block(block: str) -> dict[str, Any] | None:
    """Older/alternate HTML: primer sequences inline; Oligo(x,len) coordinates."""

    def _find_primer_seqs(b: str) -> tuple[str | None, str | None]:
        patterns_fwd = (
            r"Forward\s+primer\s*[:\s]*(?:5['′]?\s*[-–]\s*3['′]?\s*)?([ACGTacgt]{10,})",
            r"Forward\s+primer.*?(?:5['′]\s*[-–]\s*3['′])?\s*([ACGTacgt]{10,})",
            r"Forward\s+primer\s*\([^)]*\)\s*[^A]*?([ACGTacgt]{10,})",
        )
        patterns_rev = (
            r"Reverse\s+primer\s*[:\s]*(?:5['′]?\s*[-–]\s*3['′]?\s*)?([ACGTacgt]{10,})",
            r"Reverse\s+primer.*?(?:5['′]\s*[-–]\s*3['′])?\s*([ACGTacgt]{10,})",
            r"Reverse\s+primer\s*\([^)]*\)\s*[^A]*?([ACGTacgt]{10,})",
        )
        fm, rm = None, None
        for rx in patterns_fwd:
            fm = re.search(rx, b, re.I | re.DOTALL)
            if fm:
                break
        for rx in patterns_rev:
            rm = re.search(rx, b, re.I | re.DOTALL)
            if rm:
                break
        if not fm or not rm:
            return None, None
        return fm.group(1).upper(), rm.group(1).upper()

    fwd, rev = _find_primer_seqs(block)
    if not fwd or not rev:
        return None

    plen: int | None = None
    pm = re.search(r"Product length\s*[:\s]*(\d+)\s*bp", block, re.I)
    if pm:
        plen = int(pm.group(1))
    else:
        pm2 = re.search(r"Product length\s*[:\s]*(\d+)", block, re.I)
        if pm2:
            plen = int(pm2.group(1))

    def _oligo_span(after: str) -> tuple[int, int] | None:
        m = re.search(r"Oligo\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)", after[:800], re.I)
        if m:
            start, length = int(m.group(1)), int(m.group(2))
            return (start, start + length - 1)
        m = re.search(r"(\d+)\s*[-–]\s*(\d+)", after[:400])
        if m:
            return int(m.group(1)), int(m.group(2))
        return None

    fwd_m = re.search(r"Forward\s+primer", block, re.I)
    rev_m = re.search(r"Reverse\s+primer", block, re.I)
    tail_f = block[fwd_m.end() :] if fwd_m else block
    tail_r = block[rev_m.end() :] if rev_m else block
    fwd_span = _oligo_span(tail_f)
    rev_span = _oligo_span(tail_r)

    if plen is None and fwd_span and rev_span:
        f_lo, f_hi = min(fwd_span), max(fwd_span)
        r_lo, r_hi = min(rev_span), max(rev_span)
        if f_lo < r_lo:
            plen = max(1, r_lo - f_hi - 1)

    return {
        "forward_seq": fwd,
        "reverse_seq": rev,
        "product_len": plen,
        "fwd_span": fwd_span,
        "rev_span": rev_span,
    }


def parse_primer_blast_pairs(html: str) -> list[dict[str, Any]]:
    """
    Extract primer pairs from Primer-BLAST HTML. Prefer NCBI 'Detailed primer reports' tables
    (Forward/Reverse + Start/Stop + Product length); fall back to legacy patterns.
    """
    pairs: list[dict[str, Any]] = []
    blocks = re.split(r"Primer\s+pair\s*\d+", html, flags=re.I)
    if len(blocks) <= 1:
        blocks = [html]

    for block in blocks:
        if not block.strip():
            continue
        rec = _parse_detailed_report_block(block)
        if rec is None:
            rec = _parse_legacy_block(block)
        if rec is None:
            continue
        pairs.append(rec)

    return pairs


def pair_matches_design_rules(
    pair: dict[str, Any],
    spacer_start_1b: int,
    spacer_end_1b: int,
    *,
    product_min: int = PRODUCT_MIN,
    product_max: int = PRODUCT_MAX,
    fwd_max_past_spacer_end: int = FWD_MAX_PAST_SPACER_END,
    rev_min_past_spacer_end: int = REV_MIN_PAST_SPACER_END,
) -> bool:
    """
    Targeting-strand template (5′→3′): forward primer is left; spacer [ss, se] must lie in the amplicon.

    - Forward primer 3′ on the top strand (``f_hi``, inner edge toward the amplicon) may be arbitrarily
      far **5′** (lower coordinates); it must not lie more than ``fwd_max_past_spacer_end`` nt **past**
      **spacer end** ``se``: ``f_hi <= se + fwd_max_past_spacer_end``.
    - Reverse primer 5′ on the top strand (``r_lo``, inner edge toward the amplicon) must be at least
      ``rev_min_past_spacer_end`` nt **past** ``se``: ``r_lo >= se + rev_min_past_spacer_end``.
    Product length is bounded separately by ``product_min`` / ``product_max``.
    """
    plen = pair.get("product_len")
    if plen is None or plen < product_min or plen > product_max:
        return False

    fs, rs = pair.get("fwd_span"), pair.get("rev_span")
    if not fs or not rs:
        return False

    ss, se = int(spacer_start_1b), int(spacer_end_1b)

    f_lo, f_hi = min(fs), max(fs)
    r_lo, r_hi = min(rs), max(rs)
    if f_lo > r_lo:
        (f_lo, f_hi), (r_lo, r_hi) = (r_lo, r_hi), (f_lo, f_hi)
    if f_hi >= r_lo:
        return False
    if se >= r_lo:
        return False
    if f_hi > se + int(fwd_max_past_spacer_end):
        return False
    if r_lo < se + int(rev_min_past_spacer_end):
        return False
    return True


def first_matching_primer_pair(
    html: str,
    spacer_start_1b: int,
    spacer_end_1b: int,
    *,
    fwd_max_past_spacer_end: int = FWD_MAX_PAST_SPACER_END,
    rev_min_past_spacer_end: int = REV_MIN_PAST_SPACER_END,
    pairs: list[dict[str, Any]] | None = None,
) -> tuple[str, str] | tuple[None, None]:
    parsed = pairs if pairs is not None else parse_primer_blast_pairs(html)
    for p in parsed:
        if pair_matches_design_rules(
            p,
            spacer_start_1b,
            spacer_end_1b,
            fwd_max_past_spacer_end=fwd_max_past_spacer_end,
            rev_min_past_spacer_end=rev_min_past_spacer_end,
        ):
            return p["forward_seq"], p["reverse_seq"]
    return None, None


def _process_dataframe_full_gene_once(
    df: pd.DataFrame,
    gene_to_seq: dict[str, str],
    *,
    spacer_coordinate_offset: int = 0,
    fwd_max_past_spacer_end: int = FWD_MAX_PAST_SPACER_END,
    rev_min_past_spacer_end: int = REV_MIN_PAST_SPACER_END,
    ncbi_product_min: int = NCBI_POST_PRODUCT_MIN,
    ncbi_product_max: int = NCBI_POST_PRODUCT_MAX,
    ncbi_num_primer_pairs: int = DEFAULT_NCBI_NUM_PRIMER_PAIRS,
    specificity_database: str = DEFAULT_NCBI_SPECIFICITY_DB,
    organism: str = DEFAULT_NCBI_ORGANISM,
    dry_run: bool = False,
    verbose: bool = False,
    ncbi_browser_post: bool = True,
    log_ncbi_post: bool = True,
    ncbi_html_dir: Path | None = None,
) -> pd.DataFrame:
    """
    One NCBI Primer-BLAST job per distinct (Gene, Strand): full FASTA record (oriented for blast),
    then filter parsed pairs per spacer row.
    """
    out = df.copy().reset_index(drop=True)
    out["query_sequence_ncbi"] = pd.NA
    out["ncbi_input_fasta"] = pd.NA
    out["spacer_on_template_1b"] = pd.NA
    out["primer_pairs_passing_count"] = pd.NA
    out["primer_pairs_json"] = pd.NA
    out["primer_1_fwd"] = pd.NA
    out["primer_2_rev"] = pd.NA
    out["amplicon_length"] = pd.NA
    out["spacer_GC_content"] = pd.NA

    required = {"Gene", "SpacerStart", "SpacerEnd"}
    missing = required - set(out.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    fasta_headers = list(gene_to_seq.keys())
    print(
        f"[primer] mode=full_gene_once rows={len(out)} dry_run={dry_run} verbose={verbose} "
        f"ncbi_post={'expanded' if ncbi_browser_post else 'minimal'} "
        f"FASTA records={len(fasta_headers)} spacer_coordinate_offset={spacer_coordinate_offset}",
        flush=True,
    )
    if verbose:
        print(f"[primer] FASTA header sample: {fasta_headers[:3]!r}", flush=True)

    sess = requests.Session()
    n_processed = 0

    group_order: list[tuple[str, str]] = []
    group_indices: dict[tuple[str, str], list[Any]] = {}
    for i, row in out.iterrows():
        gene = str(row["Gene"])
        strand = str(row["Strand"]) if "Strand" in out.columns else "+"
        key = (gene, strand)
        if key not in group_indices:
            group_indices[key] = []
            group_order.append(key)
        group_indices[key].append(i)

    for key in group_order:
        gene, strand = key
        indices = group_indices[key]
        lp = f"[primer] group Gene={gene[:80]!r} strand={strand!r} rows={len(indices)}: "
        resolved = resolve_fasta_for_gene(gene, gene_to_seq, verbose=verbose, log_prefix=lp)
        if resolved is None:
            print(f"{lp}SKIP — no matching FASTA header", flush=True)
            if verbose:
                print(f"[primer]   available headers (first 5): {fasta_headers[:5]!r}", flush=True)
            continue

        fasta_key, seq = resolved
        if fasta_key != gene and verbose:
            print(f"{lp}resolved FASTA header {fasta_key[:80]!r} for Gene={gene!r}", flush=True)
        template = full_sequence_blast_template(seq, strand)
        fasta = f">{fasta_key}\n{template}\n"
        for idx in indices:
            out.loc[idx, "query_sequence_ncbi"] = str(template)
            out.loc[idx, "ncbi_input_fasta"] = fasta
        print(f"{lp}template_len={len(template)} (full FASTA record, blast-oriented)", flush=True)

        if dry_run:
            for idx in indices:
                row = out.loc[idx]
                ss = int(row["SpacerStart"]) + int(spacer_coordinate_offset)
                se = int(row["SpacerEnd"]) + int(spacer_coordinate_offset)
                mapped = spacer_coords_on_blast_template(seq, ss, se, strand)
                if mapped is None:
                    print(
                        f"[primer] row_index={idx} SKIP invalid spacer vs ref "
                        f"(ss={ss} se={se} ref_len={len(seq.strip())})",
                        flush=True,
                    )
                else:
                    out.loc[idx, "spacer_on_template_1b"] = f"{mapped[0]}-{mapped[1]}"
                    gc = spacer_gc_content_on_ref(seq, ss, se)
                    if gc is not None:
                        out.loc[idx, "spacer_GC_content"] = gc
                    print(
                        f"[primer] row_index={idx} spacer_on_template_1b={mapped[0]}-{mapped[1]}",
                        flush=True,
                    )
            continue

        time.sleep(NCBI_DELAY_SEC)
        html_stem = f"fullgene_{_safe_gene_filename_stem(gene)}_{strand}"
        html = submit_primer_blast(
            fasta,
            ncbi_product_min=ncbi_product_min,
            ncbi_product_max=ncbi_product_max,
            ncbi_num_primer_pairs=ncbi_num_primer_pairs,
            specificity_database=specificity_database,
            organism=organism,
            session=sess,
            verbose=verbose,
            log_prefix=lp,
            ncbi_browser_post=ncbi_browser_post,
            log_ncbi_post=log_ncbi_post,
            ncbi_html_dir=ncbi_html_dir,
            ncbi_html_stem=html_stem,
        )
        pairs = parse_primer_blast_pairs(html)
        print(
            f"{lp}NCBI detailed report: parsed {len(pairs)} primer pair block(s) from result HTML "
            f"(server returns up to ~{NCBI_SERVER_MAX_PRIMER_PAIRS} per job)",
            flush=True,
        )
        if verbose and pairs:
            print(f"{lp}  first pair raw: {pairs[0]!r}", flush=True)

        for idx in indices:
            row = out.loc[idx]
            lp_r = f"[primer] row_index={idx} Gene={str(row['Gene'])[:80]!r}: "
            ss = int(row["SpacerStart"]) + int(spacer_coordinate_offset)
            se = int(row["SpacerEnd"]) + int(spacer_coordinate_offset)
            gc = spacer_gc_content_on_ref(seq, ss, se)
            if gc is not None:
                out.loc[idx, "spacer_GC_content"] = gc
            mapped = spacer_coords_on_blast_template(seq, ss, se, strand)
            if mapped is None:
                out.loc[idx, "primer_pairs_json"] = "[]"
                out.loc[idx, "primer_pairs_passing_count"] = 0
                print(
                    f"{lp_r}SKIP — spacer outside reference (ss={ss} se={se} ref_len={len(seq.strip())})",
                    flush=True,
                )
                continue
            s1, s2 = mapped
            out.loc[idx, "spacer_on_template_1b"] = f"{s1}-{s2}"
            matched = matching_primer_pairs(
                pairs,
                s1,
                s2,
                fwd_max_past_spacer_end=fwd_max_past_spacer_end,
                rev_min_past_spacer_end=rev_min_past_spacer_end,
            )
            out.loc[idx, "primer_pairs_json"] = json.dumps(
                [_pair_to_json_serializable(p) for p in matched]
            )
            out.loc[idx, "primer_pairs_passing_count"] = len(matched)
            if matched:
                out.loc[idx, "primer_1_fwd"] = matched[0]["forward_seq"]
                out.loc[idx, "primer_2_rev"] = matched[0]["reverse_seq"]
                alen = _amplicon_length_from_pair(matched[0])
                if alen is not None:
                    out.loc[idx, "amplicon_length"] = alen
                n_processed += 1
                print(
                    f"{lp_r}OK — {len(matched)}/{len(pairs)} pair(s) pass evoCAST placement rules "
                    f"(spacer_on_template={s1}-{s2}; {len(pairs)} pairs from NCBI before this filter)",
                    flush=True,
                )
            else:
                if not pairs:
                    print(
                        f"{lp_r}no primers parsed from NCBI for this group (see group log above)",
                        flush=True,
                    )
                else:
                    print(
                        f"{lp_r}0 pairs pass filters for spacer_on_template={s1}-{s2} "
                        f"({len(pairs)} pair(s) from NCBI for whole gene)",
                        flush=True,
                    )

    print(f"[primer] done. rows_with_at_least_one_pair={n_processed} / {len(out)}", flush=True)
    return out


def _process_dataframe_per_spacer_window(
    df: pd.DataFrame,
    gene_to_seq: dict[str, str],
    *,
    upstream: int = DEFAULT_UPSTREAM,
    downstream: int = DEFAULT_DOWNSTREAM,
    spacer_coordinate_offset: int = 0,
    fwd_max_past_spacer_end: int = FWD_MAX_PAST_SPACER_END,
    rev_min_past_spacer_end: int = REV_MIN_PAST_SPACER_END,
    ncbi_product_min: int = NCBI_POST_PRODUCT_MIN,
    ncbi_product_max: int = NCBI_POST_PRODUCT_MAX,
    ncbi_num_primer_pairs: int = DEFAULT_NCBI_NUM_PRIMER_PAIRS,
    specificity_database: str = DEFAULT_NCBI_SPECIFICITY_DB,
    organism: str = DEFAULT_NCBI_ORGANISM,
    dry_run: bool = False,
    verbose: bool = False,
    ncbi_browser_post: bool = True,
    log_ncbi_post: bool = True,
    ncbi_html_dir: Path | None = None,
) -> pd.DataFrame:
    """Legacy: one NCBI job per row using a local window around each spacer."""
    out = df.copy().reset_index(drop=True)
    out["query_sequence_ncbi"] = pd.NA
    out["ncbi_input_fasta"] = pd.NA
    out["spacer_on_template_1b"] = pd.NA
    out["primer_pairs_passing_count"] = pd.NA
    out["primer_pairs_json"] = pd.NA
    out["primer_1_fwd"] = pd.NA
    out["primer_2_rev"] = pd.NA
    out["amplicon_length"] = pd.NA
    out["spacer_GC_content"] = pd.NA

    required = {"Gene", "SpacerStart", "SpacerEnd"}
    missing = required - set(out.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    fasta_headers = list(gene_to_seq.keys())
    print(
        f"[primer] mode=per_spacer_window rows={len(out)} dry_run={dry_run} verbose={verbose} "
        f"ncbi_post={'expanded' if ncbi_browser_post else 'minimal'} "
        f"FASTA records={len(fasta_headers)} upstream={upstream} downstream={downstream} "
        f"spacer_coordinate_offset={spacer_coordinate_offset}",
        flush=True,
    )
    if verbose:
        print(f"[primer] FASTA header sample: {fasta_headers[:3]!r}", flush=True)

    sess = requests.Session()
    n_processed = 0
    rc_full_cache: dict[str, str] = {}
    for i, row in out.iterrows():
        gene = str(row["Gene"])
        lp = f"[primer] row_index={i} Gene={gene[:80]!r}: "
        resolved = resolve_fasta_for_gene(gene, gene_to_seq, verbose=verbose, log_prefix=lp)
        if resolved is None:
            print(f"{lp}SKIP — no matching FASTA header", flush=True)
            if verbose:
                print(f"[primer]   available headers (first 5): {fasta_headers[:5]!r}", flush=True)
            continue

        fasta_key, seq = resolved
        if fasta_key != gene and verbose:
            print(f"{lp}resolved FASTA header {fasta_key[:80]!r} for Gene={gene!r}", flush=True)
        ss = int(row["SpacerStart"]) + int(spacer_coordinate_offset)
        se = int(row["SpacerEnd"]) + int(spacer_coordinate_offset)
        gc = spacer_gc_content_on_ref(seq, ss, se)
        if gc is not None:
            out.loc[i, "spacer_GC_content"] = gc
        strand = row["Strand"] if "Strand" in out.columns else "+"
        print(
            f"{lp}SpacerStart={ss} SpacerEnd={se} Strand={strand!r} ref_len={len(seq)} "
            f"(table {int(row['SpacerStart'])}-{int(row['SpacerEnd'])}, offset +{spacer_coordinate_offset})",
            flush=True,
        )

        st = str(strand).strip()
        prc: str | None = None
        if st in ("-", "-1", "minus", "reverse"):
            if fasta_key not in rc_full_cache:
                rc_full_cache[fasta_key] = reverse_complement(seq.strip().upper())
            prc = rc_full_cache[fasta_key]
        win = build_primer_blast_query_sequence(
            seq,
            ss,
            se,
            str(strand),
            upstream=upstream,
            downstream=downstream,
            precomputed_rc_full=prc,
        )
        if win is None:
            print(f"{lp}SKIP — could not build template window (coordinates vs reference length)", flush=True)
            continue
        template, s1, s2 = win
        out.loc[i, "query_sequence_ncbi"] = str(template)
        out.loc[i, "spacer_on_template_1b"] = f"{s1}-{s2}"
        fasta = f">{fasta_key}\n{template}\n"
        out.loc[i, "ncbi_input_fasta"] = fasta
        print(
            f"{lp}template_len={len(template)} spacer_on_template_1b={s1}-{s2}",
            flush=True,
        )

        if dry_run:
            print(f"{lp}dry-run — no NCBI request", flush=True)
            continue

        time.sleep(NCBI_DELAY_SEC)
        html_stem = f"row{i}_{_safe_gene_filename_stem(gene)}"
        html = submit_primer_blast(
            fasta,
            ncbi_product_min=ncbi_product_min,
            ncbi_product_max=ncbi_product_max,
            ncbi_num_primer_pairs=ncbi_num_primer_pairs,
            specificity_database=specificity_database,
            organism=organism,
            session=sess,
            verbose=verbose,
            log_prefix=lp,
            ncbi_browser_post=ncbi_browser_post,
            log_ncbi_post=log_ncbi_post,
            ncbi_html_dir=ncbi_html_dir,
            ncbi_html_stem=html_stem,
        )
        pairs = parse_primer_blast_pairs(html)
        print(
            f"{lp}NCBI detailed report: parsed {len(pairs)} primer pair block(s) from result HTML",
            flush=True,
        )
        if verbose and pairs:
            print(f"{lp}  first pair raw: {pairs[0]!r}", flush=True)

        matched = matching_primer_pairs(
            pairs,
            s1,
            s2,
            fwd_max_past_spacer_end=fwd_max_past_spacer_end,
            rev_min_past_spacer_end=rev_min_past_spacer_end,
        )
        out.loc[i, "primer_pairs_json"] = json.dumps(
            [_pair_to_json_serializable(p) for p in matched]
        )
        out.loc[i, "primer_pairs_passing_count"] = len(matched)
        if matched:
            out.loc[i, "primer_1_fwd"] = matched[0]["forward_seq"]
            out.loc[i, "primer_2_rev"] = matched[0]["reverse_seq"]
            alen = _amplicon_length_from_pair(matched[0])
            if alen is not None:
                out.loc[i, "amplicon_length"] = alen
            print(
                f"{lp}OK — {len(matched)}/{len(pairs)} pair(s) pass evoCAST placement rules "
                f"(spacer_on_template={s1}-{s2}; first pair also in primer_1_fwd / primer_2_rev)",
                flush=True,
            )
            n_processed += 1
        else:
            if not pairs:
                print(
                    f"{lp}no primers parsed (HTML may have changed or job failed); "
                    f"check html_chars above and NCBI page format",
                    flush=True,
                )
            else:
                print(
                    f"{lp}parsed {len(pairs)} pair(s) but none passed placement/product rules "
                    f"(spacer_in_template={s1}-{s2}; fwd_max_past_spacer_end={fwd_max_past_spacer_end}; "
                    f"rev_min_past_spacer_end={rev_min_past_spacer_end})",
                    flush=True,
                )
                if verbose:
                    print(f"{lp}  first pair: {pairs[0]!r}", flush=True)

    print(f"[primer] done. rows_with_at_least_one_pair={n_processed} / {len(out)}", flush=True)
    return out


def process_dataframe(
    df: pd.DataFrame,
    gene_to_seq: dict[str, str],
    *,
    upstream: int = DEFAULT_UPSTREAM,
    downstream: int = DEFAULT_DOWNSTREAM,
    spacer_coordinate_offset: int = 0,
    per_spacer_template: bool = False,
    fwd_max_past_spacer_end: int = FWD_MAX_PAST_SPACER_END,
    rev_min_past_spacer_end: int = REV_MIN_PAST_SPACER_END,
    ncbi_product_min: int = NCBI_POST_PRODUCT_MIN,
    ncbi_product_max: int = NCBI_POST_PRODUCT_MAX,
    ncbi_num_primer_pairs: int = DEFAULT_NCBI_NUM_PRIMER_PAIRS,
    specificity_database: str = DEFAULT_NCBI_SPECIFICITY_DB,
    organism: str = DEFAULT_NCBI_ORGANISM,
    dry_run: bool = False,
    verbose: bool = False,
    ncbi_browser_post: bool = True,
    log_ncbi_post: bool = True,
    ncbi_html_dir: Path | None = None,
) -> pd.DataFrame:
    """
    Default: one NCBI job per (Gene, Strand) on the full FASTA record; per-row ``primer_pairs_json``.

    With ``per_spacer_template=True``: one NCBI job per row; minus builds template by RC(full) then clip.
    """
    if per_spacer_template:
        return _process_dataframe_per_spacer_window(
            df,
            gene_to_seq,
            upstream=upstream,
            downstream=downstream,
            spacer_coordinate_offset=spacer_coordinate_offset,
            fwd_max_past_spacer_end=fwd_max_past_spacer_end,
            rev_min_past_spacer_end=rev_min_past_spacer_end,
            ncbi_product_min=ncbi_product_min,
            ncbi_product_max=ncbi_product_max,
            ncbi_num_primer_pairs=ncbi_num_primer_pairs,
            specificity_database=specificity_database,
            organism=organism,
            dry_run=dry_run,
            verbose=verbose,
            ncbi_browser_post=ncbi_browser_post,
            log_ncbi_post=log_ncbi_post,
            ncbi_html_dir=ncbi_html_dir,
        )
    return _process_dataframe_full_gene_once(
        df,
        gene_to_seq,
        spacer_coordinate_offset=spacer_coordinate_offset,
        fwd_max_past_spacer_end=fwd_max_past_spacer_end,
        rev_min_past_spacer_end=rev_min_past_spacer_end,
        ncbi_product_min=ncbi_product_min,
        ncbi_product_max=ncbi_product_max,
        ncbi_num_primer_pairs=ncbi_num_primer_pairs,
        specificity_database=specificity_database,
        organism=organism,
        dry_run=dry_run,
        verbose=verbose,
        ncbi_browser_post=ncbi_browser_post,
        log_ncbi_post=log_ncbi_post,
        ncbi_html_dir=ncbi_html_dir,
    )


def _safe_gene_filename_stem(gene: str) -> str:
    s = re.sub(r"[^\w.\-]+", "_", str(gene).strip())
    s = s.strip("._") or "gene"
    return s[:200]


def _gene_column_name(df: pd.DataFrame) -> str | None:
    for c in df.columns:
        if str(c) == "Gene":
            return "Gene"
    for c in df.columns:
        if str(c).lower() == "gene":
            return str(c)
    return None


def write_per_gene_excel_files(out: pd.DataFrame, main_output: Path) -> None:
    """
    Write ``{main_output.stem}_by_gene/<safe_gene>.xlsx`` for each distinct gene value
    (same columns as the combined table).
    """
    if out.empty:
        print("[primer] per-gene split: skipped (empty table)", flush=True)
        return
    gcol = _gene_column_name(out)
    if gcol is None:
        print(
            "[primer] per-gene split: skipped — no 'Gene' column in output (add Gene to input table)",
            flush=True,
        )
        return
    subdir = main_output.parent / f"{main_output.stem}_by_gene"
    subdir.mkdir(parents=True, exist_ok=True)
    stem_counts: dict[str, int] = {}
    n_files = 0
    for gene, gdf in out.groupby(gcol, sort=False):
        base = _safe_gene_filename_stem(str(gene))
        k = stem_counts.get(base, 0)
        stem_counts[base] = k + 1
        fname = f"{base}.xlsx" if k == 0 else f"{base}_{k + 1}.xlsx"
        path = subdir / fname
        gdf.to_excel(path, index=False, engine="openpyxl")
        n_files += 1
        print(f"[primer] per-gene: {len(gdf)} rows -> {path}", flush=True)
    print(f"[primer] wrote {n_files} per-gene file(s) under {subdir}", flush=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Annotate ranked_sequences Excel with NCBI Primer-BLAST primers. "
        "Default: one search per (Gene, Strand) on the full FASTA record. Columns include "
        "primer_pairs_json (pairs passing filters, or []), primer_pairs_passing_count, "
        "amplicon_length (bp, first passing pair), spacer_GC_content (G+C fraction on reference FASTA), "
        "spacer_on_template_1b, query_sequence_ncbi (DNA only), ncbi_input_fasta (exact FASTA sent as "
        "INPUT_SEQUENCE). With --per-spacer-template, minus: full RC then local window on that RC. "
        "NCBI POST defaults to the expanded form (match web UI); use --ncbi-minimal-post to send fewer fields. "
        "Also writes {output_stem}_by_gene/ unless --no-split-by-gene.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=NCBI_BROWSER_VS_SCRIPT_HELP,
    )
    ap.add_argument(
        "--input",
        action="append",
        dest="inputs",
        default=None,
        help="ranked_sequences*.xlsx (repeat for multiple genes/files)",
    )
    ap.add_argument(
        "--input-dir",
        type=Path,
        default=None,
        metavar="DIR",
        help="Load every ranked_sequences*.xlsx in this directory (in addition to --input)",
    )
    ap.add_argument(
        "--gene",
        default=None,
        metavar="NAME",
        help="Only process rows for this gene: matches full Gene column or the prefix before the first '_' (e.g. RHO for RHO__hg19_...)",
    )
    ap.add_argument(
        "--fasta",
        required=True,
        type=Path,
        help="Same FASTA used for spacer scanning (Gene column must match headers)",
    )
    ap.add_argument("--output", required=True, type=Path, help="Output .xlsx path")
    ap.add_argument(
        "--limit",
        type=int,
        default=None,
        metavar="N",
        help="Process only the first N rows after --gene filtering (e.g. 5 for quick tests)",
    )
    ap.add_argument(
        "--per-spacer-template",
        action="store_true",
        help=(
            "Legacy: one NCBI job per table row. Plus: clip forward ref around the spacer. Minus: "
            "reverse-complement the full FASTA record, then clip on that RC (not RC of the forward clip). "
            "Uses --upstream/--downstream. Default is one job per gene (full FASTA) and reuse pairs."
        ),
    )
    ap.add_argument("--upstream", type=int, default=DEFAULT_UPSTREAM)
    ap.add_argument("--downstream", type=int, default=DEFAULT_DOWNSTREAM)
    ap.add_argument(
        "--spacer-coordinate-offset",
        type=int,
        default=0,
        metavar="N",
        help=(
            "Add N to SpacerStart and SpacerEnd (1-based) before building the template. "
            f"When ranked coordinates are from an edge-trimmed spacer scan but --fasta is the full "
            f"reference, use N={DEFAULT_SPACER_SEARCH_EDGE_RESERVE} to match the left-edge reserve. "
            "Default: 0 (table coordinates already match the FASTA)."
        ),
    )
    ap.add_argument(
        "--organism",
        type=str,
        default=DEFAULT_NCBI_ORGANISM,
        help=f"NCBI Primer-BLAST organism line (default: {DEFAULT_NCBI_ORGANISM!r})",
    )
    ap.add_argument(
        "--ncbi-database",
        type=str,
        default=DEFAULT_NCBI_SPECIFICITY_DB,
        metavar="NAME",
        help=f"PRIMER_SPECIFICITY_DATABASE value (default: {DEFAULT_NCBI_SPECIFICITY_DB!r})",
    )
    ap.add_argument(
        "--ncbi-product-min",
        type=int,
        default=NCBI_POST_PRODUCT_MIN,
        help=f"NCBI PRIMER_PRODUCT_MIN (default: {NCBI_POST_PRODUCT_MIN}, same as client product filter)",
    )
    ap.add_argument(
        "--ncbi-product-max",
        type=int,
        default=NCBI_POST_PRODUCT_MAX,
        help=f"NCBI PRIMER_PRODUCT_MAX (default: {NCBI_POST_PRODUCT_MAX}, same as client product filter)",
    )
    ap.add_argument(
        "--ncbi-num-primer-pairs",
        type=int,
        default=DEFAULT_NCBI_NUM_PRIMER_PAIRS,
        metavar="N",
        dest="ncbi_num_primer_pairs",
        help=(
            f"NCBI PRIMER_NUM_RETURN (web form: # of primers to return; default: {DEFAULT_NCBI_NUM_PRIMER_PAIRS}). "
            f"Note: Primer-BLAST typically returns at most ~{NCBI_SERVER_MAX_PRIMER_PAIRS} pairs per job regardless "
            "of this value (server-side cap, same as the website)."
        ),
    )
    ap.add_argument(
        "--fwd-max-past-spacer-end",
        type=int,
        default=FWD_MAX_PAST_SPACER_END,
        metavar="N",
        dest="fwd_max_past_spacer_end",
        help=(
            "Forward primer 3′ (top strand, inner edge) at most N nt past spacer end se "
            "(default: 20). May be arbitrarily far 5′; product size also capped by client/NCBI limits."
        ),
    )
    ap.add_argument(
        "--rev-min-past-spacer-end",
        type=int,
        default=REV_MIN_PAST_SPACER_END,
        metavar="N",
        dest="rev_min_past_spacer_end",
        help=(
            "Reverse primer 5′ (top strand, inner edge) at least N nt past spacer end se (default: 130)."
        ),
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Load tables and extract windows only; do not call NCBI",
    )
    ap.add_argument(
        "--verbose",
        action="store_true",
        help="Extra logging (sequence preview, primer pair dicts, short HTML dumps)",
    )
    ap.add_argument(
        "--ncbi-minimal-post",
        action="store_true",
        help=(
            "Send only a small subset of CGI fields (product size, db, organism, mismatch count, "
            "PRIMER_NUM_RETURN, MAX_TARGET_SIZE). Expanded default adds advanced Primer3/NCBI fields "
            "(GC, Poly-X, stability, self/pair complementarity, salt tables, …); minimal omits those — "
            "they are not guaranteed to match the numbers shown in the browser. Default is expanded "
            "POST to align with the web form; use this flag only if you need a smaller request (e.g. "
            "unstable connections)."
        ),
    )
    ap.add_argument(
        "--no-ncbi-post-params",
        action="store_true",
        help=(
            "Do not print the full primertool.cgi form (all keys and values) before each POST. "
            "Default is to print it for comparison with the browser Network tab."
        ),
    )
    ap.add_argument(
        "--ncbi-html-dir",
        type=Path,
        default=None,
        metavar="DIR",
        help=(
            "Save each completed Primer-BLAST result page as UTF-8 HTML under DIR "
            "(filename: <stem>_<RID>.html) for diffing against a browser-saved page."
        ),
    )
    ap.add_argument(
        "--no-split-by-gene",
        action="store_true",
        help="Do not write per-gene Excel files under {output_stem}_by_gene/ (combined --output only).",
    )
    args = ap.parse_args(argv)

    input_paths: list[Path] = []
    if args.inputs:
        input_paths.extend(Path(p) for p in args.inputs)
    if args.input_dir is not None:
        input_paths.extend(sorted(args.input_dir.glob("ranked_sequences*.xlsx")))
    if not input_paths:
        raise SystemExit("Provide at least one --input file and/or --input-dir containing ranked_sequences*.xlsx")

    print(f"[primer] input files: {len(input_paths)} { [str(p) for p in input_paths]!r}", flush=True)
    print(f"[primer] FASTA: {args.fasta}", flush=True)
    print(
        f"[primer] NCBI POST: {'minimal (--ncbi-minimal-post)' if args.ncbi_minimal_post else 'expanded (default, matches web form)'}; "
        f"full CGI params logged each POST unless --no-ncbi-post-params",
        flush=True,
    )
    if args.ncbi_html_dir is not None:
        print(f"[primer] NCBI result HTML will be saved under {Path(args.ncbi_html_dir).resolve()}", flush=True)

    gene_to_seq = load_gene_sequences(args.fasta)
    frames: list[pd.DataFrame] = []
    for p in input_paths:
        frames.append(pd.read_excel(p, engine="openpyxl"))
    combined = pd.concat(frames, axis=0, ignore_index=True) if len(frames) > 1 else frames[0]

    if args.gene:
        if "Gene" not in combined.columns:
            raise SystemExit("Cannot use --gene: input table has no Gene column")
        mask = combined["Gene"].map(lambda x: gene_matches_filter(str(x), str(args.gene)))
        combined = combined.loc[mask].reset_index(drop=True)
        print(f"Filtered to {len(combined)} rows (--gene {args.gene!r})")

    if args.limit is not None:
        n = int(args.limit)
        if n < 1:
            raise SystemExit("--limit must be >= 1")
        combined = combined.iloc[:n].copy()
        print(f"[primer] using first {len(combined)} rows (--limit {n})", flush=True)

    npr = int(args.ncbi_num_primer_pairs)
    if npr < 1 or npr > 1000:
        raise SystemExit("--ncbi-num-primer-pairs must be between 1 and 1000")
    if npr > NCBI_SERVER_MAX_PRIMER_PAIRS:
        print(
            f"[primer] warning: --ncbi-num-primer-pairs {npr} is sent as PRIMER_NUM_RETURN, but NCBI "
            f"Primer-BLAST usually caps output at ~{NCBI_SERVER_MAX_PRIMER_PAIRS} pairs per run — "
            "expect parsed primer_pairs count to stop there.",
            flush=True,
        )

    out = process_dataframe(
        combined,
        gene_to_seq,
        upstream=int(args.upstream),
        downstream=int(args.downstream),
        spacer_coordinate_offset=int(args.spacer_coordinate_offset),
        per_spacer_template=bool(args.per_spacer_template),
        fwd_max_past_spacer_end=int(args.fwd_max_past_spacer_end),
        rev_min_past_spacer_end=int(args.rev_min_past_spacer_end),
        ncbi_product_min=int(args.ncbi_product_min),
        ncbi_product_max=int(args.ncbi_product_max),
        ncbi_num_primer_pairs=npr,
        specificity_database=str(args.ncbi_database),
        organism=str(args.organism),
        dry_run=bool(args.dry_run),
        verbose=bool(args.verbose),
        ncbi_browser_post=not bool(args.ncbi_minimal_post),
        log_ncbi_post=not bool(args.no_ncbi_post_params),
        ncbi_html_dir=args.ncbi_html_dir,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_excel(args.output, index=False, engine="openpyxl")
    outp = Path(args.output).resolve()
    print(f"Wrote {len(out)} rows to {outp}", flush=True)
    if not args.no_split_by_gene:
        print(
            f"[primer] splitting by gene -> {outp.parent / (outp.stem + '_by_gene')}",
            flush=True,
        )
        write_per_gene_excel_files(out, outp)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
