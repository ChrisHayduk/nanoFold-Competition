from __future__ import annotations

from pathlib import Path

import pytest

from nanofold.mmseqs import iter_mmseqs_hits, validate_search_parameters


@pytest.mark.parametrize(("row", "error"), [
    ("query\tsource", "eight columns"),
    ("unknown\tsource\t50\t20\t1\t1\t0.01\t20", "unknown sequence"),
    ("query\tsource\tnan\t20\t1\t1\t0.01\t20", "out-of-range"),
    ("query\tsource\t50\t0\t1\t1\t0.01\t20", "out-of-range"),
    ("query\tsource\t50\t20\t1.1\t1\t0.01\t20", "out-of-range"),
    ("query\tsource\t50\t20.5\t1\t1\t0.01\t20", "invalid numeric"),
    ("query\tsource\t20\t20\t1\t1\t0.01\t20", "requested thresholds"),
    ("query\tsource\t50\t20\t1\t0.7\t0.01\t20", "requested thresholds"),
])
def test_invalid_search_output_is_rejected(tmp_path: Path, row: str, error: str) -> None:
    path = tmp_path / "hits.m8"
    path.write_text(row + "\n")
    with pytest.raises(ValueError, match=error):
        list(iter_mmseqs_hits(path, query_ids={"query"}, target_ids={"source"}, min_seq_id=0.3, coverage=0.8))


def test_missing_search_output_is_not_an_empty_result(tmp_path: Path) -> None:
    path = tmp_path / "hits.m8"
    with pytest.raises(RuntimeError, match="without writing"):
        list(iter_mmseqs_hits(path, query_ids={"query"}, target_ids={"source"}, min_seq_id=0.3, coverage=0.8))
    path.write_text("")
    assert list(iter_mmseqs_hits(path, query_ids={"query"}, target_ids={"source"}, min_seq_id=0.3, coverage=0.8)) == []


@pytest.mark.parametrize(("identity", "coverage", "threads", "max_seqs"), [
    (float("nan"), 0.8, 1, 100), (0.3, float("inf"), 1, 100),
    (0.3, 0.0, 1, 100), (0.3, 0.8, 0, 100), (0.3, 0.8, 1, 0),
])
def test_invalid_search_parameters_are_rejected(identity: float, coverage: float, threads: int, max_seqs: int) -> None:
    with pytest.raises(ValueError):
        validate_search_parameters(min_seq_id=identity, coverage=coverage, threads=threads, max_seqs=max_seqs)


def test_native_count_limit_is_accepted() -> None:
    validate_search_parameters(min_seq_id=0.3, coverage=0.8, threads=2_147_483_647, max_seqs=2_147_483_647)


@pytest.mark.parametrize("value", [2_147_483_648, 4_294_967_297])
@pytest.mark.parametrize("argument", ["threads", "max_seqs"])
def test_native_count_overflow_is_rejected(argument: str, value: int) -> None:
    counts = {"threads": 1, "max_seqs": 100}
    counts[argument] = value
    with pytest.raises(ValueError, match=r"\[1, 2147483647\]"):
        validate_search_parameters(min_seq_id=0.3, coverage=0.8, **counts)


def test_exhaustive_search_rejects_overflow_before_launch_or_output_removal(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from scripts.build_msa_row_filter import _run_mmseqs

    source = tmp_path / "source.fasta"
    source.write_text(">source\nAAAA\n")
    output = tmp_path / "alignments.m8"
    output.write_text("previous audit evidence\n")
    monkeypatch.setattr("scripts.build_msa_row_filter.subprocess.run", lambda *args, **kwargs: pytest.fail("MMseqs was launched"))
    with pytest.raises(ValueError, match=r"\[1, 2147483647\]"):
        _run_mmseqs(
            mmseqs_bin="mmseqs", heldout_fasta=tmp_path / "heldout.fasta", source_fasta=source,
            output_m8=output, tmp_dir=tmp_path / "work", min_seq_id=0.3, coverage=0.8,
            threads=1, max_seqs=4_294_967_297, exhaustive=True,
        )
    assert output.read_text() == "previous audit evidence\n"


@pytest.mark.parametrize("corruption", ["missing_query", "masked_query", "masked_hit", "gapped_query", "float_hit", "wrong_chain", "negative_deletions"])
def test_audit_rejects_semantically_invalid_processed_sequences(tmp_path: Path, corruption: str) -> None:
    import numpy as np

    from nanofold.mmseqs import load_msa_for_audit

    arrays = {
        "aatype": np.array([0, 1, 2], dtype=np.int32),
        "msa": np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int32),
        "deletions": np.zeros((2, 3), dtype=np.int32),
        "chain_id": np.asarray("toy_A"),
    }
    if corruption == "missing_query":
        del arrays["aatype"]
    elif corruption == "masked_query":
        arrays["aatype"][0] = arrays["msa"][0, 0] = 22
    elif corruption == "masked_hit":
        arrays["msa"][1, 0] = 22
    elif corruption == "gapped_query":
        arrays["aatype"][0] = arrays["msa"][0, 0] = 21
    elif corruption == "float_hit":
        arrays["msa"] = arrays["msa"].astype(float) + 0.1
    elif corruption == "wrong_chain":
        arrays["chain_id"] = np.asarray("other_A")
    else:
        arrays["deletions"][1, 1] = -1
    path = tmp_path / "features.npz"
    np.savez(path, **arrays)
    with pytest.raises(ValueError):
        load_msa_for_audit(path, chain_id="toy_A")


def test_heldout_normalization_matches_feature_unknown_residues() -> None:
    from nanofold.mmseqs import canonical_target_sequence

    assert canonical_target_sequence("ABZJUOX") == "AXXXXXX"
    for invalid in ("AC D", "AC-D", "AC\n>injected", "", "AC*D"):
        with pytest.raises(ValueError, match="amino-acid"):
            canonical_target_sequence(invalid)
