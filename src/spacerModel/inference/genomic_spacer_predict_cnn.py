#!/usr/bin/env python3
"""
CLI wrapper for genomic YN-PAM scanning + CNN FE prediction.

Core API: ``spacerModel.inference.inference`` (torch load/score via ``training.train_torch``).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .inference import (
    DEFAULT_GENOMIC_SPACER_LEN,
    run_genomic_cnn_spacer_prediction,
)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Genomic YN-PAM 32mer scan + CNN FE prediction")
    ap.add_argument("--checkpoint", required=True, type=Path, help="Path to CNN_best_model.pt")
    ap.add_argument("--fasta", required=True, type=Path, help="Input FASTA or plain sequence file")
    ap.add_argument("--output", required=True, type=Path, help="Output .xlsx or .csv path")
    ap.add_argument("--device", default="auto", help="auto|cpu|cuda")
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument(
        "--spacer-len",
        type=int,
        default=None,
        help=f"Spacer length (default: inferred from checkpoint, typically {DEFAULT_GENOMIC_SPACER_LEN})",
    )
    args = ap.parse_args(argv)

    try:
        cand = run_genomic_cnn_spacer_prediction(
            args.checkpoint,
            args.fasta,
            device=str(args.device),
            batch_size=int(args.batch_size),
            spacer_len=args.spacer_len,
        )
    except ValueError as e:
        raise SystemExit(str(e)) from e

    if cand.empty:
        print("No YN-PAM sites found.")
        return 0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    suf = args.output.suffix.lower()
    if suf in (".xlsx", ".xls"):
        cand.to_excel(args.output, index=False, engine="openpyxl")
    else:
        cand.to_csv(args.output, index=False)

    print(f"Wrote {len(cand)} rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
