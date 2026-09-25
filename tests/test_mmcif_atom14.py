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
from nanofold.residue_constants import CA_ATOM14_SLOT


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
    from nanofold.mmcif import _collect_atom14

    residue = gemmi.Residue()
    residue.name = "ALA"
    for altloc, occupancy, x in (("A", a_occupancy, 2.0), ("B", b_occupancy, 8.0)):
        atom = gemmi.Atom()
        atom.name = "CA"
        atom.altloc = altloc
        atom.occ = occupancy
        atom.pos = gemmi.Position(x, 0, 0)
        residue.add_atom(atom)
    coordinates, mask = _collect_atom14(residue, "ALA")
    assert mask[CA_ATOM14_SLOT] == 1
    assert coordinates[CA_ATOM14_SLOT, 0] == expected


@pytest.mark.parametrize("field", ["coordinate", "occupancy"])
def test_nonfinite_structure_values_fail_before_becoming_supervision(field: str) -> None:
    from nanofold.mmcif import _collect_atom14

    residue = gemmi.Residue()
    residue.name = "ALA"
    atom = gemmi.Atom()
    atom.name = "CA"
    atom.pos = gemmi.Position(float("nan") if field == "coordinate" else 0, 0, 0)
    atom.occ = float("nan") if field == "occupancy" else 1
    residue.add_atom(atom)
    with pytest.raises(ValueError, match="Nonfinite"):
        _collect_atom14(residue, "ALA")


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
