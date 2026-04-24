from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from nanofold.a3m import (
    GAP_ID,
    RESTYPE_TO_ID,
    UNK_ID,
    read_a3m,
    sequence_to_ids,
    ungap_query_columns,
)


def test_a3m_deletion_matrix_behavior(tmp_path: Path) -> None:
    content = """>query
ACD-E
>hit1
AcdCD-E
"""
    path = tmp_path / "toy.a3m"
    path.write_text(content)

    a3m = read_a3m(path)
    aligned, deletions = a3m.to_aligned_msa()

    assert aligned == ["ACD-E", "ACD-E"]
    assert deletions.shape == (2, 5)
    assert np.array_equal(deletions[0], np.array([0, 0, 0, 0, 0], dtype=np.int32))
    assert np.array_equal(deletions[1], np.array([0, 2, 0, 0, 0], dtype=np.int32))

    msa, del_tokens = a3m.to_tokens()
    assert msa.dtype == np.int32
    assert del_tokens.dtype == np.int32


def test_sequence_to_ids_maps_canonical_alphabet() -> None:
    ids = sequence_to_ids("ARNZX")
    assert ids.dtype == np.int32
    assert ids.shape == (5,)
    assert int(ids[0]) == RESTYPE_TO_ID["A"]
    assert int(ids[1]) == RESTYPE_TO_ID["R"]
    assert int(ids[2]) == RESTYPE_TO_ID["N"]
    # Z and X are non-standard — map to UNK_ID.
    assert int(ids[3]) == UNK_ID
    assert int(ids[4]) == UNK_ID


def test_ungap_query_columns_removes_query_gaps() -> None:
    # Query has a gap at column 2 → that column must be dropped from MSA/deletions.
    query_aligned = "AC-DE"
    msa = np.array(
        [
            [RESTYPE_TO_ID["A"], RESTYPE_TO_ID["C"], GAP_ID, RESTYPE_TO_ID["D"], RESTYPE_TO_ID["E"]],
            [RESTYPE_TO_ID["A"], RESTYPE_TO_ID["C"], RESTYPE_TO_ID["V"], RESTYPE_TO_ID["D"], RESTYPE_TO_ID["E"]],
        ],
        dtype=np.int32,
    )
    deletions = np.array(
        [[0, 0, 0, 0, 0], [0, 1, 0, 0, 0]],
        dtype=np.int32,
    )

    new_msa, new_deletions, target = ungap_query_columns(msa=msa, deletions=deletions, query_aligned=query_aligned)

    assert target == "ACDE"
    assert new_msa.shape == (2, 4)
    assert new_deletions.shape == (2, 4)
    # Column 2 was dropped — row 1 column 2 carried 'V' which should now be absent.
    assert np.array_equal(
        new_msa[1],
        np.array([RESTYPE_TO_ID["A"], RESTYPE_TO_ID["C"], RESTYPE_TO_ID["D"], RESTYPE_TO_ID["E"]], dtype=np.int32),
    )


def test_ungap_query_columns_rejects_length_mismatch() -> None:
    msa = np.zeros((1, 3), dtype=np.int32)
    deletions = np.zeros((1, 3), dtype=np.int32)
    with pytest.raises(ValueError):
        ungap_query_columns(msa=msa, deletions=deletions, query_aligned="ACDE")
