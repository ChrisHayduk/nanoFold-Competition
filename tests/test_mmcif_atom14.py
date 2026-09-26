"""Tests for the atom14-aware mmCIF parser."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("gemmi")

import gemmi  # noqa: E402

from nanofold.mmcif import (
    ChainAtoms,
    extract_chain_atoms,
)
from nanofold.residue_constants import ATOM14_INDEX, CA_ATOM14_SLOT


def _build_synthetic_mmcif(path: Path, *, chain_id: str = "A") -> None:
    """Write a minimal 3-residue alanine chain to ``path`` as mmCIF."""
    structure = gemmi.Structure()
    structure.name = "synth"
    model = gemmi.Model(1)
    chain = gemmi.Chain(chain_id)

    for seq_index in range(3):
        residue = gemmi.Residue()
        residue.name = "ALA"
        residue.seqid = gemmi.SeqId(seq_index + 1, " ")
        residue.entity_type = gemmi.EntityType.Polymer
        base_z = float(seq_index) * 3.8  # approximate Cα-Cα bond length

        for atom_name, dx, dy in (("N", 0.0, 1.0), ("CA", 0.0, 0.0), ("C", 1.0, 0.0), ("O", 1.0, -1.0)):
            atom = gemmi.Atom()
            atom.name = atom_name
            atom.element = gemmi.Element(atom_name[0])
            atom.pos = gemmi.Position(dx, dy, base_z)
            atom.occ = 1.0
            residue.add_atom(atom)

        chain.add_residue(residue)

    model.add_chain(chain)
    structure.add_model(model)
    structure.setup_entities()
    structure.make_mmcif_document().write_file(str(path))


def test_extract_chain_atoms_shapes_and_residue_index(tmp_path: Path) -> None:
    mmcif_path = tmp_path / "synth.cif"
    _build_synthetic_mmcif(mmcif_path)

    atoms = extract_chain_atoms(mmcif_path=mmcif_path, pdb_id="synth", chain_id="A")
    assert isinstance(atoms, ChainAtoms)
    assert atoms.sequence == "AAA"
    assert atoms.atom14_positions.shape == (3, 14, 3)
    assert atoms.atom14_mask.shape == (3, 14)
    assert atoms.residue_index.tolist() == [0, 1, 2]
    assert atoms.residue_index.dtype == np.int32

    # Every residue here has all 4 backbone atoms present.
    for slot in range(4):
        assert bool(atoms.atom14_mask[:, slot].all())


def test_ca_view_matches_atom14_ca_slot(tmp_path: Path) -> None:
    mmcif_path = tmp_path / "synth.cif"
    _build_synthetic_mmcif(mmcif_path)

    atoms = extract_chain_atoms(mmcif_path=mmcif_path, pdb_id="synth", chain_id="A")

    np.testing.assert_allclose(
        atoms.ca_coords,
        atoms.atom14_positions[:, CA_ATOM14_SLOT, :],
    )
    np.testing.assert_array_equal(
        atoms.ca_mask,
        atoms.atom14_mask[:, CA_ATOM14_SLOT] > 0.5,
    )


def test_extract_chain_atoms_requires_match_for_expected_sequence(tmp_path: Path) -> None:
    mmcif_path = tmp_path / "synth.cif"
    _build_synthetic_mmcif(mmcif_path)

    with pytest.raises(ValueError):
        extract_chain_atoms(
            mmcif_path=mmcif_path,
            pdb_id="synth",
            chain_id="A",
            expected_sequence="AAAA",
            require_full_match=True,
        )


def test_extract_chain_atoms_resolves_openfold_chain_key_by_sequence(tmp_path: Path) -> None:
    mmcif_path = tmp_path / "synth.cif"
    _build_synthetic_mmcif(mmcif_path, chain_id="A")

    atoms = extract_chain_atoms(
        mmcif_path=mmcif_path,
        pdb_id="synth",
        chain_id="AAA",
        expected_sequence="MAAA",
        require_full_match=False,
    )

    assert atoms.chain_id == "AAA"
    assert atoms.sequence == "AAA"


@pytest.mark.parametrize("a_occupancy,b_occupancy,expected", [(0.2, 0.8, 8.0), (0.5, 0.5, 2.0)])
def test_altloc_uses_highest_occupancy_before_conformer_preference(
    a_occupancy: float, b_occupancy: float, expected: float,
) -> None:
    from nanofold.mmcif import _select_chain_conformers

    residue = gemmi.Residue()
    residue.name = "ALA"
    for altloc, occupancy, x in (("A", a_occupancy, 2.0), ("B", b_occupancy, 8.0)):
        atom = gemmi.Atom()
        atom.name = "CA"
        atom.altloc = altloc
        atom.occ = occupancy
        atom.pos = gemmi.Position(x, 0, 0)
        residue.add_atom(atom)
    selected = _select_chain_conformers([residue])[0]
    assert selected.mask[CA_ATOM14_SLOT] == 1
    assert selected.positions[CA_ATOM14_SLOT, 0] == expected


@pytest.mark.parametrize("field", ["coordinate", "occupancy"])
def test_nonfinite_structure_values_fail_before_becoming_supervision(field: str) -> None:
    from nanofold.mmcif import _select_chain_conformers

    residue = gemmi.Residue()
    residue.name = "ALA"
    atom = gemmi.Atom()
    atom.name = "CA"
    atom.pos = gemmi.Position(float("nan") if field == "coordinate" else 0, 0, 0)
    atom.occ = float("nan") if field == "occupancy" else 1
    residue.add_atom(atom)
    with pytest.raises(ValueError, match="Nonfinite"):
        _select_chain_conformers([residue])


def _build_chains(path: Path, chains: list[tuple[str, str, str]]) -> None:
    structure = gemmi.Structure()
    model = gemmi.Model(1)
    for author_id, label_id, sequence in chains:
        chain = gemmi.Chain(author_id)
        for index, letter in enumerate(sequence):
            residue = gemmi.Residue()
            residue.name = {"A": "ALA", "G": "GLY"}[letter]
            residue.seqid = gemmi.SeqId(index + 1, " ")
            residue.subchain = label_id
            residue.entity_type = gemmi.EntityType.Polymer
            atom = gemmi.Atom()
            atom.name = "CA"
            atom.element = gemmi.Element("C")
            atom.pos = gemmi.Position(index * 3.8, 0, 0)
            residue.add_atom(atom)
            chain.add_residue(residue)
        model.add_chain(chain)
    structure.add_model(model)
    structure.setup_entities()
    structure.make_mmcif_document().write_file(str(path))


@pytest.mark.parametrize(
    "query,first,second",
    [("AAA", "AAA", "AAA"), ("A" * 20, "A" * 16, "A" * 20 + "G" * 5)],
)
def test_sequence_fallback_rejects_equally_matching_chains(
    tmp_path: Path, query: str, first: str, second: str,
) -> None:
    path = tmp_path / "ambiguous.cif"
    _build_chains(path, [("A", "X", first), ("B", "Y", second)])
    with pytest.raises(ValueError, match="Ambiguous sequence mapping.*A, B"):
        extract_chain_atoms(path, "synth", "missing", expected_sequence=query, require_full_match=False)


def test_label_id_mapping_resolves_homomer_without_sequence_guessing(tmp_path: Path) -> None:
    path = tmp_path / "homomer.cif"
    _build_chains(path, [("A", "X", "AAA"), ("B", "Y", "AAA")])
    atoms = extract_chain_atoms(path, "synth", "Y", expected_sequence="AAA")
    assert atoms.chain_id == "Y"
    assert atoms.resolved_author_chain_id == "B"
    assert atoms.chain_mapping_method == "label_id"
    assert atoms.sequence == "AAA"


def test_sequence_fallback_accepts_unique_best_candidate_and_records_author(tmp_path: Path) -> None:
    path = tmp_path / "unique.cif"
    _build_chains(path, [("A", "X", "AAAAAAA"), ("B", "Y", "AAAAAAG")])
    atoms = extract_chain_atoms(path, "synth", "missing", expected_sequence="AAAAAAA")
    assert atoms.chain_id == "missing"
    assert atoms.resolved_author_chain_id == "A"
    assert atoms.chain_mapping_method == "sequence"


def test_author_id_match_takes_precedence_over_sequence_fallback(tmp_path: Path) -> None:
    path = tmp_path / "homomer.cif"
    _build_chains(path, [("A", "X", "AAA"), ("B", "Y", "AAA")])
    atoms = extract_chain_atoms(path, "synth", "B", expected_sequence="AAA")
    assert atoms.chain_id == atoms.resolved_author_chain_id == "B"
    assert atoms.chain_mapping_method == "author_id"


def _alanine(index: int) -> gemmi.Residue:
    residue = gemmi.Residue()
    residue.name = "ALA"
    residue.seqid = gemmi.SeqId(index + 1, " ")
    residue.label_seq = index + 1
    residue.subchain = "X"
    residue.entity_type = gemmi.EntityType.Polymer
    return residue


def _add_conformer(
    residue: gemmi.Residue, altloc: str, occupancy: float, y: float = 0,
    names: tuple[str, ...] = ("N", "CA", "C", "O", "CB"),
) -> None:
    sequence_number = residue.seqid.num
    assert sequence_number is not None
    base = (sequence_number - 1) * 3.8
    coordinates = {
        "N": (base - 1.2, y, 0), "CA": (base, y, 0), "C": (base + 1.3, y, 0),
        "O": (base + 1.8, y + 1, 0), "CB": (base, y, 1.5),
    }
    for name in names:
        atom = gemmi.Atom()
        atom.name = name
        atom.element = gemmi.Element(name[0])
        atom.altloc = altloc or "\0"
        atom.occ = occupancy
        atom.pos = gemmi.Position(*coordinates[name])
        residue.add_atom(atom)


def _write_residues(path: Path, residues: list[gemmi.Residue]) -> None:
    structure = gemmi.Structure()
    model = gemmi.Model(1)
    chain = gemmi.Chain("A")
    for residue in residues:
        chain.add_residue(residue)
    model.add_chain(chain)
    structure.add_model(model)
    structure.setup_entities()
    structure.make_mmcif_document().write_file(str(path))


def test_neighbor_geometry_selects_one_deposited_conformer_not_individual_atoms(tmp_path: Path) -> None:
    left, right = _alanine(0), _alanine(1)
    _add_conformer(left, "A", 0.8)
    _add_conformer(left, "B", 0.2, y=8)
    _add_conformer(right, "C", 0.2)
    _add_conformer(right, "D", 0.8, y=8)
    # Per-atom occupancy creates both an inter-residue break and a mixed
    # right residue. The deposited lower-occupancy backbone avoids both.
    for atom in right:
        if atom.name == "CA":
            atom.occ = 0.9 if atom.altloc == "C" else 0.1
    path = tmp_path / "alternates.cif"
    _write_residues(path, [left, right])
    parsed = extract_chain_atoms(path, "synth", "A")
    assert parsed.atom14_mask[:, :5].all()
    np.testing.assert_array_equal(parsed.atom14_positions[:, :3, 1], np.zeros((2, 3)))
    assert np.linalg.norm(parsed.atom14_positions[0, 2] - parsed.atom14_positions[1, 0]) == pytest.approx(1.3)
    # Different alternate letters in adjacent residues are intentionally valid.
    for index, residue in enumerate((left, right)):
        selected_altloc = ("A", "C")[index]
        for atom in residue:
            if atom.altloc == selected_altloc:
                slot = ATOM14_INDEX["ALA"][atom.name]
                np.testing.assert_array_equal(
                    parsed.atom14_positions[index, slot],
                    np.asarray([atom.pos.x, atom.pos.y, atom.pos.z], dtype=np.float32),
                )


def test_conformer_selection_accounts_for_both_neighbors(tmp_path: Path) -> None:
    left, middle, right = (_alanine(index) for index in range(3))
    _add_conformer(left, "", 1)
    _add_conformer(middle, "A", 0.9)
    _add_conformer(middle, "B", 0.1, y=0.6)
    _add_conformer(right, "", 1, y=1.4)
    # Both middle choices connect to the left. The higher-occupancy choice
    # breaks the right peptide bond, which a left-to-right greedy rule misses.
    path = tmp_path / "lookahead.cif"
    _write_residues(path, [left, middle, right])
    parsed = extract_chain_atoms(path, "synth", "A")
    assert parsed.ca_coords[1, 1] == pytest.approx(0.6)
    peptide_distances = np.linalg.norm(parsed.atom14_positions[:-1, 2] - parsed.atom14_positions[1:, 0], axis=1)
    assert np.all((1 <= peptide_distances) & (peptide_distances <= 1.8))


def test_missing_backbone_cannot_win_by_hiding_bad_geometry(tmp_path: Path) -> None:
    left, right = _alanine(0), _alanine(1)
    _add_conformer(left, "", 1)
    _add_conformer(right, "A", 0.1, y=8)
    _add_conformer(right, "B", 0.9, names=("CA",))
    path = tmp_path / "incomplete.cif"
    _write_residues(path, [left, right])
    parsed = extract_chain_atoms(path, "synth", "A")
    assert parsed.atom14_mask[:, :4].all()
    assert parsed.ca_coords[1, 1] == 8
    assert np.linalg.norm(parsed.atom14_positions[0, 2] - parsed.atom14_positions[1, 0]) > 1.8


def test_equal_count_backbone_masks_are_resolved_without_geometry(tmp_path: Path) -> None:
    left, right = _alanine(0), _alanine(1)
    _add_conformer(left, "", 1)
    # Both alternates have three backbone atoms. Dropping N would make the
    # bad peptide bond uncheckable; the fixed mask tie-break retains N.
    _add_conformer(right, "A", 0.1, y=8, names=("N", "CA", "C"))
    _add_conformer(right, "B", 0.9, names=("CA", "C", "O"))
    path = tmp_path / "incomparable.cif"
    _write_residues(path, [left, right])
    parsed = extract_chain_atoms(path, "synth", "A")
    np.testing.assert_array_equal(parsed.atom14_mask[1, :4], [1, 1, 1, 0])
    assert np.linalg.norm(parsed.atom14_positions[0, 2] - parsed.atom14_positions[1, 0]) > 1.8


@pytest.mark.parametrize("right_label_seq", [None, 3])
def test_missing_or_nonadjacent_polymer_indices_do_not_imply_a_bond(
    tmp_path: Path, right_label_seq: int | None,
) -> None:
    left, right = _alanine(0), _alanine(1)
    right.label_seq = right_label_seq
    _add_conformer(left, "", 1)
    _add_conformer(right, "A", 0.1)
    _add_conformer(right, "B", 0.9, y=8)
    path = tmp_path / "gap.cif"
    _write_residues(path, [left, right])
    parsed = extract_chain_atoms(path, "synth", "A")
    assert parsed.ca_coords[1, 1] == 8
    assert parsed.polymer_residue_index.tolist() == [0, -1 if right_label_seq is None else 2]


def test_conformer_uses_shared_atoms_without_borrowing_other_alternate_slots(tmp_path: Path) -> None:
    residue = _alanine(0)
    _add_conformer(residue, "", 1, names=("N", "C", "O"))
    _add_conformer(residue, "A", 0.8, names=("CA",))
    _add_conformer(residue, "B", 0.2, y=8, names=("CA", "CB"))
    path = tmp_path / "shared.cif"
    _write_residues(path, [residue])
    parsed = extract_chain_atoms(path, "synth", "A")
    assert parsed.atom14_mask[0, :4].all()
    assert parsed.ca_coords[0, 1] == 0
    assert parsed.atom14_mask[0, 4] == 0  # CB belongs only to alternate B.
    np.testing.assert_array_equal(parsed.atom14_positions[0, 4], [0, 0, 0])


def test_zero_occupancy_is_unobserved_without_deleting_residues(tmp_path: Path) -> None:
    first, second = _alanine(0), _alanine(1)
    _add_conformer(first, "A", 0, y=8)
    _add_conformer(first, "B", 0.2, names=("CA",))
    _add_conformer(second, "", 0)
    path = tmp_path / "unobserved.cif"
    _write_residues(path, [first, second])
    parsed = extract_chain_atoms(path, "synth", "A")
    assert parsed.sequence == "AA"
    assert parsed.residue_index.tolist() == [0, 1]
    assert parsed.polymer_residue_index.tolist() == [0, 1]
    assert parsed.atom14_mask.sum() == 1
    assert parsed.ca_mask.tolist() == [True, False]
    assert not parsed.atom14_positions[parsed.atom14_mask == 0].any()


def test_negative_occupancy_is_rejected_and_positive_occupancy_has_no_upper_cutoff(tmp_path: Path) -> None:
    residue = _alanine(0)
    _add_conformer(residue, "", 1.2)
    path = tmp_path / "occupancy.cif"
    _write_residues(path, [residue])
    assert extract_chain_atoms(path, "synth", "A").atom14_mask[:, :5].all()
    residue[0].occ = -0.1
    _write_residues(path, [residue])
    with pytest.raises(ValueError, match="Negative occupancy"):
        extract_chain_atoms(path, "synth", "A")


def test_atom_order_does_not_change_equal_occupancy_conformer_selection(tmp_path: Path) -> None:
    expected = None
    for reverse in (False, True):
        residues = [_alanine(0), _alanine(1)]
        for residue in residues:
            for altloc, y in reversed((("A", 0), ("B", 8))) if reverse else (("A", 0), ("B", 8)):
                _add_conformer(residue, altloc, 0.5, y=y)
        path = tmp_path / f"order-{reverse}.cif"
        _write_residues(path, residues)
        parsed = extract_chain_atoms(path, "synth", "A")
        if expected is None:
            expected = parsed.atom14_positions
        else:
            np.testing.assert_array_equal(parsed.atom14_positions, expected)


def test_model_selection_preserves_one_whole_deposited_model(tmp_path: Path) -> None:
    left, right = _alanine(0), _alanine(1)
    _add_conformer(left, "", 1)
    _add_conformer(right, "", 1, y=8)
    path = tmp_path / "ensemble.cif"
    _write_residues(path, [left, right])
    structure = gemmi.read_structure(str(path))
    second = structure[0].clone()
    second.num = 2
    for atom in second[0][1]:
        atom.pos.y -= 8
    second[0][0][4].pos.z = 1.7
    structure.add_model(second)
    structure.make_mmcif_document().write_file(str(path))
    parsed = extract_chain_atoms(path, "synth", "A")
    assert parsed.ca_coords[1, 1] == 0
    assert parsed.atom14_mask[:, :5].all()
    for residue_index, residue in enumerate(second[0]):
        for atom in residue:
            slot = ATOM14_INDEX["ALA"][atom.name]
            np.testing.assert_array_equal(
                parsed.atom14_positions[residue_index, slot],
                np.asarray([atom.pos.x, atom.pos.y, atom.pos.z], dtype=np.float32),
            )


@pytest.mark.parametrize("mismatch", ["backbone_support", "sidechain_support", "support_swap", "residue", "polymer_index", "subchain", "chain_mapping", "missing_chain", "negative_occupancy", "nonfinite_coordinate"])
def test_ineligible_later_model_cannot_replace_first_model(tmp_path: Path, mismatch: str) -> None:
    left, right = _alanine(0), _alanine(1)
    _add_conformer(left, "", 1)
    _add_conformer(right, "", 1, y=8)
    path = tmp_path / "ineligible-ensemble.cif"
    _write_residues(path, [left, right])
    structure = gemmi.read_structure(str(path))
    second = structure[0].clone()
    second.num = 2
    for atom in second[0][1]:
        atom.pos.y -= 8
    if mismatch == "backbone_support":
        second[0][1][0].occ = 0
    elif mismatch == "sidechain_support":
        second[0][1][4].occ = 0
    elif mismatch == "support_swap":
        structure[0][0][0][0].occ = 0
        second[0][1][0].occ = 0
    elif mismatch == "residue":
        second[0][1].name = "GLY"
    elif mismatch == "polymer_index":
        second[0][1].label_seq = 3
    elif mismatch == "subchain":
        for residue in second[0]:
            residue.subchain = "Y"
    elif mismatch == "chain_mapping":
        second[0].name = "B"
    elif mismatch == "missing_chain":
        del second[0]
    elif mismatch == "negative_occupancy":
        second[0][1][0].occ = -0.1
    else:
        second[0][1][0].pos.x = float("nan")
    structure.add_model(second)
    structure.make_mmcif_document().write_file(str(path))
    parsed = extract_chain_atoms(path, "synth", "A", expected_sequence="AA")
    assert parsed.ca_coords[1, 1] == 8
    expected_mask = np.ones((2, 5))
    if mismatch == "support_swap":
        expected_mask[0, 0] = 0
    np.testing.assert_array_equal(parsed.atom14_mask[:, :5], expected_mask)
    assert np.linalg.norm(parsed.atom14_positions[0, 2] - parsed.atom14_positions[1, 0]) > 1.8


def test_equal_geometry_models_keep_deposited_order_without_occupancy_ranking(tmp_path: Path) -> None:
    left, right = _alanine(0), _alanine(1)
    _add_conformer(left, "", 0.5)
    _add_conformer(right, "", 0.5, y=8)
    path = tmp_path / "model-tie.cif"
    _write_residues(path, [left, right])
    structure = gemmi.read_structure(str(path))
    second = structure[0].clone()
    second.num = 2
    # Rigidly translating the whole model leaves every geometric score tied.
    for residue in second[0]:
        for atom in residue:
            atom.occ = 1
            atom.pos.z += 4
    structure.add_model(second)
    structure.make_mmcif_document().write_file(str(path))
    parsed = extract_chain_atoms(path, "synth", "A")
    np.testing.assert_array_equal(parsed.ca_coords[:, 2], [0, 0])
    assert parsed.ca_coords[1, 1] == 8


def test_invalid_first_model_is_not_replaced_by_valid_later_model(tmp_path: Path) -> None:
    residue = _alanine(0)
    _add_conformer(residue, "", 1)
    path = tmp_path / "invalid-anchor.cif"
    _write_residues(path, [residue])
    structure = gemmi.read_structure(str(path))
    second = structure[0].clone()
    second.num = 2
    structure[0][0][0][0].occ = -0.1
    structure.add_model(second)
    structure.make_mmcif_document().write_file(str(path))
    with pytest.raises(ValueError, match="Negative occupancy"):
        extract_chain_atoms(path, "synth", "A")


def test_single_model_unavoidable_geometry_remains_observed(tmp_path: Path) -> None:
    left, right = _alanine(0), _alanine(1)
    _add_conformer(left, "", 1)
    _add_conformer(right, "", 1, y=8)
    path = tmp_path / "unavoidable.cif"
    _write_residues(path, [left, right])
    parsed = extract_chain_atoms(path, "synth", "A")
    assert parsed.ca_coords[1, 1] == 8
    assert parsed.atom14_mask[:, :5].all()


def test_model_selection_does_not_score_across_missing_polymer_indices(tmp_path: Path) -> None:
    left, right = _alanine(0), _alanine(1)
    right.label_seq = None
    _add_conformer(left, "", 1)
    _add_conformer(right, "", 1, y=8)
    path = tmp_path / "unknown-model-adjacency.cif"
    _write_residues(path, [left, right])
    structure = gemmi.read_structure(str(path))
    second = structure[0].clone()
    second.num = 2
    for atom in second[0][1]:
        atom.pos.y -= 8
    structure.add_model(second)
    structure.make_mmcif_document().write_file(str(path))
    assert extract_chain_atoms(path, "synth", "A").ca_coords[1, 1] == 8
