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
