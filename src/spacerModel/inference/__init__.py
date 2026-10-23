"""Genomic CNN inference; see ``inference.py`` for the handoff-friendly API."""

from .inference import (
    DEFAULT_GENOMIC_SPACER_LEN,
    DEFAULT_SIMILAR_SPACER_IDENTICAL_NT,
    DEFAULT_SPACER_SEARCH_EDGE_RESERVE,
    PREDICTED_LOG2_FE_COL,
    add_spacer_uniqueness_columns,
    extract_spacers_with_yn_pam_both_strands,
    load_genomic_sequence,
    parse_fasta_records,
    plot_predicted_fe_density,
    plot_predicted_fe_density_by_gene,
    reverse_complement,
    run_genomic_cnn_spacer_prediction,
    write_ranked_excel_per_gene_partitioned,
    write_ranked_sequences_excel_per_gene,
)

__all__ = [
    "DEFAULT_GENOMIC_SPACER_LEN",
    "DEFAULT_SIMILAR_SPACER_IDENTICAL_NT",
    "DEFAULT_SPACER_SEARCH_EDGE_RESERVE",
    "PREDICTED_LOG2_FE_COL",
    "add_spacer_uniqueness_columns",
    "extract_spacers_with_yn_pam_both_strands",
    "load_genomic_sequence",
    "parse_fasta_records",
    "plot_predicted_fe_density",
    "plot_predicted_fe_density_by_gene",
    "reverse_complement",
    "run_genomic_cnn_spacer_prediction",
    "write_ranked_excel_per_gene_partitioned",
    "write_ranked_sequences_excel_per_gene",
]
