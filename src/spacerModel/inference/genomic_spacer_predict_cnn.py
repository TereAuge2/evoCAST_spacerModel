#!/usr/bin/env python3
"""
CLI wrapper for genomic YN-PAM scanning + CNN log2-FE prediction.

Core API: ``spacerModel.inference.inference`` (torch load/score via ``training.train_torch``).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .inference import (
    DEFAULT_GENOMIC_SPACER_LEN,
    DEFAULT_SPACER_SEARCH_EDGE_RESERVE,
    plot_predicted_fe_density,
    plot_predicted_fe_density_by_gene,
    run_genomic_cnn_spacer_prediction,
    write_ranked_sequences_excel_per_gene,
)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Genomic YN-PAM 32mer scan + CNN log2-FE prediction")
    ap.add_argument("--checkpoint", required=True, type=Path, help="Path to CNN_best_model.pt")
    ap.add_argument("--fasta", required=True, type=Path, help="Input FASTA or plain sequence file")
    ap.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Anchor path: per-gene workbooks are written as ranked_sequences_<gene>.xlsx in the same directory as this path",
    )
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument(
        "--spacer-len",
        type=int,
        default=None,
        help=f"Spacer length (default: inferred from checkpoint, typically {DEFAULT_GENOMIC_SPACER_LEN})",
    )
    ap.add_argument(
        "--no-omit-edge-bases",
        action="store_true",
        help="Scan the full FASTA record (do not exclude flanking bases). Overrides --spacer-search-edge-reserve.",
    )
    ap.add_argument(
        "--spacer-search-edge-reserve",
        type=int,
        default=DEFAULT_SPACER_SEARCH_EDGE_RESERVE,
        metavar="N",
        help=f"Bases excluded from each end during YN-PAM scan only (default {DEFAULT_SPACER_SEARCH_EDGE_RESERVE}; ignored with --no-omit-edge-bases)",
    )
    ap.add_argument(
        "--no-plot",
        action="store_true",
        help="Do not write Predicted_log2_FE density plots (PNG + PDF)",
    )
    ap.add_argument(
        "--plot-output",
        type=Path,
        default=None,
        help="Base path for density plot(s): .png and .pdf; multi-FASTA uses <stem>_<prefix>.png per gene (prefix = header before first '_') (default: <output_stem>_predicted_log2_fe_density)",
    )
    args = ap.parse_args(argv)

    edge = 0 if args.no_omit_edge_bases else int(args.spacer_search_edge_reserve)

    try:
        cand = run_genomic_cnn_spacer_prediction(
            args.checkpoint,
            args.fasta,
            device="cpu",
            batch_size=int(args.batch_size),
            spacer_len=args.spacer_len,
            spacer_search_edge_reserve=edge,
        )
    except ValueError as e:
        raise SystemExit(str(e)) from e

    if cand.empty:
        print("No YN-PAM sites found.")
        return 0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    written = write_ranked_sequences_excel_per_gene(cand, args.output.parent)
    for path, n in written:
        print(f"Wrote {n} rows to {path}")

    if not args.no_plot:
        plot_path = args.plot_output
        if plot_path is None:
            plot_path = args.output.with_name(f"{args.output.stem}_predicted_log2_fe_density.png")
        written = plot_predicted_fe_density_by_gene(cand, plot_path)
        for png in written:
            pdf = png.with_suffix(".pdf")
            print(f"Wrote density plots to {png} and {pdf}")
        if not written:
            print("Skipped density plots (no plottable Predicted_log2_FE values).")
        elif "Gene" in cand.columns and cand["Gene"].nunique(dropna=False) > 1:
            pp = Path(plot_path)
            base = pp.with_suffix("") if pp.suffix.lower() in (".png", ".pdf") else pp
            combined_plot = base.parent / f"{base.name}_combined.png"
            plot_predicted_fe_density(
                cand,
                combined_plot,
                title="Predicted log2-FE density (all genes)",
            )
            print(f"Wrote combined density plots to {combined_plot} and {combined_plot.with_suffix('.pdf')}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
