from __future__ import annotations

from pathlib import Path

import numpy as np

from nanofold.a3m import read_a3m


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
