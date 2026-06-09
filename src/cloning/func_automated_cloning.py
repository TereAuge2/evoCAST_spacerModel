"""In-silico insertion of sequences into plasmid maps (GenBank / SnapGene)."""

from pathlib import Path
from typing import BinaryIO, Optional, TextIO, Tuple, Union

from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqFeature import SeqFeature, FeatureLocation
from Bio.SeqRecord import SeqRecord

PathLike = Union[str, Path]
IOLike = Union[PathLike, BinaryIO, TextIO]


def _read_record(input_file: IOLike, fmt: Optional[str] = None) -> SeqRecord:
    if isinstance(input_file, (str, Path)):
        path = Path(input_file)
        if fmt is None:
            suffix = path.suffix.lower()
            fmt = "snapgene" if suffix == ".dna" else "genbank"
        return SeqIO.read(path, fmt)

    handle = input_file
    if fmt is None:
        fmt = "snapgene"
    return SeqIO.read(handle, fmt)


def _write_record(record: SeqRecord, output_file: IOLike, fmt: str = "genbank") -> None:
    if isinstance(output_file, (str, Path)):
        SeqIO.write(record, output_file, fmt)
        return
    SeqIO.write(record, output_file, fmt)


def snapgene_file_to_gbk(
    input_file: Union[PathLike, BinaryIO],
    output_file: Union[PathLike, TextIO],
) -> None:
    """Convert a SnapGene ``.dna`` file to GenBank format."""
    record = _read_record(input_file, fmt="snapgene")
    _write_record(record, output_file, fmt="genbank")


def _shift_feature(feat: SeqFeature, delta: int) -> SeqFeature:
    start = int(feat.location.start) + delta
    end = int(feat.location.end) + delta
    new_feat = SeqFeature(
        FeatureLocation(start, end, strand=feat.location.strand),
        type=feat.type,
        qualifiers=dict(feat.qualifiers),
    )
    return new_feat


def _replace_between_flanks(
    record: SeqRecord,
    flank_5_prime: str,
    flank_3_prime: str,
    insert: str,
) -> Tuple[SeqRecord, int, int]:
    seq = str(record.seq).upper()
    flank_5 = flank_5_prime.upper()
    flank_3 = flank_3_prime.upper()
    insert_seq = "".join(c for c in insert.upper() if c in "ACGT")

    i5 = seq.find(flank_5)
    if i5 < 0:
        raise ValueError(f"5' flank not found in plasmid: {flank_5}")
    after_5 = i5 + len(flank_5)
    i3 = seq.find(flank_3, after_5)
    if i3 < 0:
        raise ValueError(f"3' flank not found in plasmid: {flank_3}")

    replaced_len = i3 - after_5
    new_seq = seq[:after_5] + insert_seq + seq[i3:]
    delta = len(insert_seq) - replaced_len
    spacer_start = after_5
    spacer_end = after_5 + len(insert_seq)

    new_record = record[:]
    new_record.seq = Seq(new_seq)
    if "molecule_type" not in new_record.annotations:
        new_record.annotations["molecule_type"] = "DNA"

    kept_features = []  # type: list
    for feat in record.features:
        fstart = int(feat.location.start)
        fend = int(feat.location.end)
        if fend <= after_5:
            kept_features.append(feat)
        elif fstart >= i3:
            kept_features.append(_shift_feature(feat, delta))

    new_record.features = kept_features
    return new_record, spacer_start, spacer_end


def insert_sequence(
    input_file: PathLike,
    output_file: PathLike,
    flank_5_prime: str,
    flank_3_prime: str,
    insert: str,
) -> None:
    """Replace the sequence between two flanking motifs with ``insert``."""
    record = _read_record(input_file)
    new_record, _, _ = _replace_between_flanks(
        record, flank_5_prime, flank_3_prime, insert
    )
    _write_record(new_record, output_file)


def insert_sequence_with_feature(
    input_file: PathLike,
    output_file: PathLike,
    flank_5_prime: str,
    flank_3_prime: str,
    insert: str,
    feature_label: str,
) -> None:
    """Like :func:`insert_sequence`, but annotate the insert with a misc_feature."""
    record = _read_record(input_file)
    new_record, spacer_start, spacer_end = _replace_between_flanks(
        record, flank_5_prime, flank_3_prime, insert
    )
    spacer_feature = SeqFeature(
        FeatureLocation(spacer_start, spacer_end, strand=1),
        type="misc_feature",
        qualifiers={"label": [feature_label]},
    )
    new_record.features.append(spacer_feature)
    _write_record(new_record, output_file)
