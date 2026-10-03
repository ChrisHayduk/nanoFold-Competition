from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from nanofold.cath_boundaries import author_segments_to_label, domain_metadata, load_cath_boundaries


def _sources(tmp_path: Path, ranges: str, fasta: str) -> dict[str, dict[str, Any]]:
    (tmp_path / "cath-domain-boundaries-seqres.txt").write_text(ranges)
    (tmp_path / "cath-domain-seqs.fa").write_text(fasta)
    return load_cath_boundaries(tmp_path)


def _domain(identifier: str) -> dict[str, str]:
    return {"domain_id": identifier, "topology_id": "1.10.20", "homologous_superfamily_id": "1.10.20.30"}


def test_discontinuous_domains_retain_repeats_and_residue_order(tmp_path: Path) -> None:
    # Domain numbers do not reflect order. The repeated topology remains two
    # independent domain instances, and the outer domain spans a discontinuity.
    records = _sources(tmp_path, "1abcA01\t3-4\n1abcA02\t1-2,5-6\n", ">cath|4_4_0|1abcA01/13-14\nCD\n" + ">cath|4_4_0|1abcA02/10(A)-12_15-16\nABEF\n")
    result = domain_metadata([_domain("1abcA01"), _domain("1abcA02")], "ABCDEFGH", records)
    assert result["cath_topology_counts"] == {"1.10.20": 2}
    assert result["cath_domain_order"] == ["1abcA02", "1abcA01"]
    assert result["cath_annotated_residue_count"] == 6
    assert result["cath_unannotated_residue_count"] == 2
    assert result["cath_unannotated_fraction"] == 0.25
    assert result["cath_domains"][1]["domain_length"] == 4
    assert result["cath_domains"][1]["label_segments"] == [{"start": 1, "end": 2}, {"start": 5, "end": 6}]
    assert result["cath_domains"][1]["author_segments"][0]["start_insertion_code"] == "A"


@pytest.mark.parametrize("ranges,fasta,reason", [
    ("1abcA00\t1-4\n", ">cath|4_4_0|1abcA00/1-4\nAAAT\n", "domain_sequence_mismatch"),
    ("1abcA00\t1-9\n", ">cath|4_4_0|1abcA00/1-9\nAAAAA\n", "invalid_or_out_of_range_seqres_segments"),
    ("1abcA00\t3-1\n", ">cath|4_4_0|1abcA00/1-4\nAAAAA\n", "invalid_or_out_of_range_seqres_segments"),
    ("1abcA00\t1-3,3-4\n", ">cath|4_4_0|1abcA00/1-4\nAAAA\n", "invalid_or_out_of_range_seqres_segments"),
    ("1abcA00\t1-4\n", ">cath|4_4_0|2abcA00/1-4\nAAAA\n", "missing_release_domain_sequence"),
])
def test_unverified_boundaries_never_claim_lengths_or_coverage(
    tmp_path: Path, ranges: str, fasta: str, reason: str,
) -> None:
    result = domain_metadata([_domain("1abcA00")], "AAAA", _sources(tmp_path, ranges, fasta))
    assert result["cath_domains"][0]["boundary_status"] == reason
    assert result["cath_domains"][0]["domain_length"] is None
    assert result["cath_domains"][0]["label_segments"] == []
    assert result["cath_unannotated_fraction"] is None
    assert result["cath_domain_order"] is None


def test_overlapping_domains_are_explicitly_unresolved(tmp_path: Path) -> None:
    records = _sources(tmp_path, "1abcA01\t1-3\n1abcA02\t3-4\n", ">cath|4_4_0|1abcA01/1-3\nAAA\n>cath|4_4_0|1abcA02/3-4\nAA\n")
    result = domain_metadata([_domain("1abcA01"), _domain("1abcA02")], "AAAA", records)
    assert result["cath_domain_count"] == 2
    assert result["cath_boundary_status"] == "unresolved"
    assert result["cath_annotated_residue_count"] is None
    assert all(domain["boundary_status"] == "overlapping_domain_annotations" for domain in result["cath_domains"])


@pytest.mark.parametrize("ranges,fasta", [
    ("1abcA00 1-4\n1abcA00 2-4\n", ">cath|4_4_0|1abcA00/1-4\nAAAA\n"),
    ("1abcA00 1-4\n", ">cath|4_3_0|1abcA00/1-4\nAAAA\n"),
    ("1abcA00 1-4\n", ">cath|4_4_0|1abcA00/1-4\nAAAA\n>cath|4_4_0|1abcA00/1-4\nAAAA\n"),
    ("1abcA00 1-4\n", "AAAA\n"),
])
def test_malformed_conflicting_or_mixed_release_sources_fail(tmp_path: Path, ranges: str, fasta: str) -> None:
    with pytest.raises(ValueError):
        _sources(tmp_path, ranges, fasta)


def test_missing_annotation_is_not_fabricated() -> None:
    result = domain_metadata([], "AAAA", {})
    assert result["cath_boundary_status"] == "unannotated"
    assert result["cath_domains"] == []
    assert result["cath_unannotated_fraction"] == 1.0


def test_stale_chopping_can_use_unique_exact_contiguous_domain_sequence(tmp_path: Path) -> None:
    records = _sources(tmp_path, "1abcA00\t1-4\n", ">cath|4_4_0|1abcA00/1-4\nACDEFG\n")
    result = domain_metadata([_domain("1abcA00")], "MMACDEFGTT", records)
    domain = result["cath_domains"][0]
    assert domain["boundary_status"] == "verified_unique_domain_sequence"
    assert domain["domain_length"] == 6
    assert domain["label_segments"] == [{"start": 3, "end": 8}]
    assert domain["seqres_segments"] == [{"start": 1, "end": 4}]
    assert result["cath_unannotated_fraction"] == 0.4
    ambiguous = domain_metadata([_domain("1abcA00")], "ACDEFGTTACDEFG", records)
    assert ambiguous["cath_domains"][0]["domain_length"] is None
    assert ambiguous["cath_domain_order"] is None


def test_discontinuous_sequence_is_never_forced_into_contiguous_fallback(tmp_path: Path) -> None:
    records = _sources(tmp_path, "1abcA00\t1-2,5-6\n", ">cath|4_4_0|1abcA00/1-2_5-6\nACDE\n")
    result = domain_metadata([_domain("1abcA00")], "MMACDETT", records)
    assert result["cath_domains"][0]["domain_length"] is None


def _scheme(path: Path, rows: str) -> Path:
    path.write_text("data_test\nloop_\n" + "\n".join("_pdbx_poly_seq_scheme." + name for name in [
        "asym_id", "seq_id", "pdb_seq_num", "pdb_ins_code", "pdb_strand_id",
    ]) + "\n" + rows)
    return path


def test_author_numbering_maps_offsets_insertions_and_discontinuous_segments(tmp_path: Path) -> None:
    path = _scheme(tmp_path / "test.cif", "X 1 -2 . a\nX 2 10 A a\nX 3 10 B a\nX 4 11 . a\n")
    segments = [
        {"start_number": -2, "start_insertion_code": "", "end_number": 10, "end_insertion_code": "A"},
        {"start_number": 11, "start_insertion_code": "", "end_number": 11, "end_insertion_code": ""},
    ]
    assert author_segments_to_label(path, "a", segments, 4) == [{"start": 1, "end": 2}, {"start": 4, "end": 4}]
    assert author_segments_to_label(path, "A", segments, 4) is None


@pytest.mark.parametrize("rows", [
    "X 1 10 . A\nY 2 11 . A\n",  # Ambiguous author-to-label chain mapping.
    "X 1 10 . A\nX 2 10 . A\n",  # Duplicate author residue identity.
    "X 1 10 . A\n",  # Missing residue mapping.
])
def test_ambiguous_polymer_mapping_stays_unknown(tmp_path: Path, rows: str) -> None:
    path = _scheme(tmp_path / "test.cif", rows)
    segments = [{"start_number": 10, "start_insertion_code": "", "end_number": 11, "end_insertion_code": ""}]
    assert author_segments_to_label(path, "A", segments, 2) is None
