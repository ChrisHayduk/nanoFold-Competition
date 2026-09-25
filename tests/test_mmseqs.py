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
