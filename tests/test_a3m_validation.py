from pathlib import Path

import numpy as np
import pytest

from nanofold.a3m import GAP_ID, UNK_ID, read_a3m, ungap_query_columns


@pytest.mark.parametrize("character", ["1", "?", "*", " ", "\t", "é", "É", "α", "Ａ", "\x00"])
def test_invalid_homolog_characters_cannot_become_unknown_residues(tmp_path: Path, character: str) -> None:
    path = tmp_path / "invalid.a3m"
    path.write_text(f">query\nAAA\n>hit\nA{character}A\n")
    with pytest.raises(ValueError, match="invalid sequence characters"):
        read_a3m(path)


@pytest.mark.parametrize("contents, message", [
    ("AAA\n>query\nAAA\n", "before the first header"),
    (">\nAAA\n", "empty header"),
    (">  \nAAA\n", "empty header"),
    (">query\n>hit\nAAA\n", "empty sequence"),
    (">query\nAAA\n>hit\n", "empty sequence"),
    (">query\n...\n", "at least one aligned residue"),
    (">query\n---\n", "at least one aligned residue"),
    (">query\naaa\n", "at least one aligned residue"),
    (">query\na.-a\n", "at least one aligned residue"),
])
def test_malformed_records_are_rejected(tmp_path: Path, contents: str, message: str) -> None:
    path = tmp_path / "invalid.a3m"
    path.write_text(contents)
    with pytest.raises(ValueError, match=message):
        read_a3m(path)


def test_optional_insertion_gaps_do_not_create_columns_or_deletions(tmp_path: Path) -> None:
    # HH-suite A3M permits dots for gaps aligned to lowercase insertions.
    # They may also be omitted, so both representations must yield identical features.
    dotted = tmp_path / "dotted.a3m"
    omitted = tmp_path / "omitted.a3m"
    dotted.write_text(">query\n.A..C-D.\n>hit\nA.b.cC-D...\n")
    omitted.write_text(">query\nAC-D\n>hit\nAbcC-D\n")
    msa, deletions = read_a3m(dotted).to_tokens()
    expected_msa, expected_deletions = read_a3m(omitted).to_tokens()
    np.testing.assert_array_equal(msa, expected_msa)
    np.testing.assert_array_equal(deletions, expected_deletions)
    np.testing.assert_array_equal(deletions[1], [0, 2, 0, 0])
    query = read_a3m(dotted).to_aligned_msa()[0][0]
    ungapped_msa, ungapped_deletions, target = ungap_query_columns(msa, deletions, query)
    assert target == "ACD"
    assert ungapped_msa.shape == ungapped_deletions.shape == (2, 3)


def test_unknown_residues_query_gaps_and_wrapped_sequences_remain_valid(tmp_path: Path) -> None:
    path = tmp_path / "valid.a3m"
    # Six noncanonical uppercase letters remain explicit unknown residues.
    # Lowercase insertions carry deletion counts, without occupying columns.
    path.write_text(">query description\nABJ\nOUXZ-\n>hit description\naABJ\nouOUXZ-\n")
    msa, deletions = read_a3m(path).to_tokens()
    np.testing.assert_array_equal(msa[0], [0, UNK_ID, UNK_ID, UNK_ID, UNK_ID, UNK_ID, UNK_ID, GAP_ID])
    np.testing.assert_array_equal(msa[0], msa[1])
    np.testing.assert_array_equal(deletions[1], [1, 0, 0, 2, 0, 0, 0, 0])
