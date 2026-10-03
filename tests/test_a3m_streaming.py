from __future__ import annotations

import hashlib
import tracemalloc
from pathlib import Path

import numpy as np
import pytest

from nanofold.a3m import RESTYPES, iter_a3m_rows, read_a3m, sequence_to_ids
from nanofold.msa_filter_policy import MSARowFilter
from scripts.preprocess import _read_merged_msa


def test_streaming_rows_preserve_alignment_and_deletion_features(tmp_path: Path) -> None:
    path = tmp_path / "input.a3m"
    path.write_text(">query\nA.C-\nD.\n>hit\nAb.cCX\nD\n>unknown\nXZ--\n")
    materialized = read_a3m(path)
    expected_aligned, expected_deletions = materialized.to_aligned_msa()
    rows = list(iter_a3m_rows(path))
    assert [header for header, _, _ in rows] == materialized.headers
    assert [aligned for _, aligned, _ in rows] == expected_aligned
    np.testing.assert_array_equal(np.stack([deletions for _, _, deletions in rows]), expected_deletions)


@pytest.mark.parametrize("late_record, message", [
    (">late\nA1-D\n", "invalid sequence characters"),
    (">late\nACD\n", "inconsistent aligned lengths"),
    (">late\n", "empty sequence"),
    (">\nAC-D\n", "empty header"),
])
@pytest.mark.parametrize("second_source", [False, True])
def test_depth_cap_never_skips_validation_of_late_rows_or_sources(
    tmp_path: Path, late_record: str, message: str, second_source: bool,
) -> None:
    first = tmp_path / "first.a3m"
    first.write_text(">query\nAC-D\n>support\nA--D\n" + ("" if second_source else late_record))
    if second_source:
        (tmp_path / "second.a3m").write_text(">query\nAC-D\n" + late_record)
    with pytest.raises(ValueError, match=message):
        _read_merged_msa(tmp_path, msa_name="first.a3m", msa_names=["first.a3m", "second.a3m"], max_msa_seqs=2)


def test_depth_cap_still_checks_later_source_query(tmp_path: Path) -> None:
    (tmp_path / "first.a3m").write_text(">query\nACD\n")
    (tmp_path / "second.a3m").write_text(">query\nARD\n")
    with pytest.raises(ValueError, match="Mismatched query sequences"):
        _read_merged_msa(tmp_path, msa_name="first.a3m", msa_names=["first.a3m", "second.a3m"], max_msa_seqs=1)


def test_filter_counts_cover_all_sources_and_rows_after_retention_cap(tmp_path: Path) -> None:
    (tmp_path / "first.a3m").write_text(
        ">query\nAC-D\n>duplicate\nAC-D\n>excluded\nAXJD\n"
        ">different_deletions\nAaC-D\n>support\nA--D\n>excluded_after_cap\nAXJD\n"
    )
    (tmp_path / "second.a3m").write_text(
        ">query\nA-CD\n>excluded_after_projection\nAZXD\n>late_support\nR-CD\n"
    )
    row_filter = MSARowFilter(
        path=tmp_path / "filter.json", sha256="0" * 64,
        excluded_sequence_sha256=frozenset({hashlib.sha256(b"AXD").hexdigest()}),
    )
    msa, deletions, query, stats = _read_merged_msa(
        tmp_path, msa_name="first.a3m", msa_names=["first.a3m", "second.a3m"],
        max_msa_seqs=3, msa_row_filter=row_filter,
    )
    assert query == "ACD"
    np.testing.assert_array_equal(msa, [sequence_to_ids(sequence) for sequence in ("ACD", "ACD", "A-D")])
    np.testing.assert_array_equal(deletions, [[0, 0, 0], [0, 1, 0], [0, 0, 0]])
    assert stats.input_rows == 9
    assert stats.removed_rows == 3


def test_capped_deep_msa_does_not_materialize_all_rows(tmp_path: Path) -> None:
    path = tmp_path / "deep.a3m"
    depth, width = 3000, 256
    with path.open("w") as handle:
        for index in range(depth):
            prefix = "".join(RESTYPES[(index // 20**offset) % 20] for offset in range(3))
            handle.write(f">row{index}\n{prefix}" + "A" * (width - 3) + "\n")
    tracemalloc.start()
    try:
        msa, deletions, _, stats = _read_merged_msa(
            tmp_path, msa_name="deep.a3m", msa_names=None, max_msa_seqs=2,
        )
        _, peak_bytes = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert msa.shape == deletions.shape == (2, width)
    assert stats.input_rows == depth
    # One full int32 matrix alone exceeds 3 MB; retaining aligned strings,
    # insertion lists and both complete matrices would be substantially larger.
    assert peak_bytes < 2_000_000
