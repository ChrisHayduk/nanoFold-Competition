from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from nanofold.structure_metadata import (
    dihedral_degrees,
    secondary_class_from_fractions,
    secondary_fractions_from_atom14,
    secondary_fractions_from_mmcif_annotations,
)


def _mmcif(path: Path, records: str, *, observed_length: int = 10) -> None:
    path.write_text(
        """data_test
loop_
_struct_conf.conf_type_id
_struct_conf.beg_label_asym_id
_struct_conf.end_label_asym_id
_struct_conf.beg_auth_asym_id
_struct_conf.end_auth_asym_id
_struct_conf.beg_label_seq_id
_struct_conf.end_label_seq_id
_struct_conf.beg_auth_seq_id
_struct_conf.end_auth_seq_id
"""
        + records
        + "loop_\n_atom_site.auth_asym_id\n_atom_site.label_asym_id\n_atom_site.label_atom_id\n_atom_site.label_seq_id\n"
        + "_atom_site.Cartn_x\n_atom_site.Cartn_y\n_atom_site.Cartn_z\n"
        + "".join(f"A X CA {position} {position}.0 0.0 0.0\n" for position in range(1, observed_length + 1))
    )


def test_annotations_use_author_chain_and_label_sequence_positions(tmp_path: Path) -> None:
    path = tmp_path / "test.cif"
    _mmcif(path, "HELX_P X X A A 2 5 102 105\nSTRN X X A A 8 9 108 109\nTURN_P X X A A 6 7 106 107\n")
    # Four helical and two strand residues in a ten-residue polymer; author
    # numbering begins at 101 and cannot be used as array indices.
    assert secondary_fractions_from_mmcif_annotations(str(path), chain_id="A", length=10) == (
        "alpha_beta",
        0.4,
        0.2,
        pytest.approx(0.4),
        "mmcif_annotations",
    )
    assert secondary_fractions_from_mmcif_annotations(str(path), chain_id="X", length=10) == (
        "unknown",
        None,
        None,
        None,
        None,
    )


def test_missing_chain_mapping_or_label_coordinates_do_not_invent_measurements(tmp_path: Path) -> None:
    path = tmp_path / "test.cif"
    _mmcif(path, "HELX_P A A ? ? 1 5 101 105\nHELX_P X X A A ? ? 101 105\n")
    assert secondary_fractions_from_mmcif_annotations(str(path), chain_id="A", length=10) == (
        "unknown",
        None,
        None,
        None,
        None,
    )


def test_conflicting_annotation_ranges_are_unknown(tmp_path: Path) -> None:
    path = tmp_path / "test.cif"
    _mmcif(path, "HELX_P X X A A 1 5 101 105\nSTRN X X A A 4 7 104 107\n")
    assert secondary_fractions_from_mmcif_annotations(str(path), chain_id="A", length=10) == (
        "unknown",
        None,
        None,
        None,
        None,
    )


def test_missing_author_fields_use_explicit_atom_chain_crosswalk(tmp_path: Path) -> None:
    path = tmp_path / "test.cif"
    _mmcif(path, "HELX_P X X ? ? 1 5 101 105\n")
    assert secondary_fractions_from_mmcif_annotations(str(path), chain_id="A", length=10) == (
        "alpha",
        0.5,
        0.0,
        0.5,
        "mmcif_annotations",
    )


@pytest.mark.parametrize("fractions", [(float("nan"), 0.2, 0.3), (1.2, 0.0, 0.0), (0.6, 0.6, 0.0), (-0.1, 0.5, 0.6)])
def test_invalid_secondary_fractions_are_unknown(fractions: tuple[float, float, float]) -> None:
    assert secondary_class_from_fractions(*fractions) == "unknown"


def test_nonfinite_coordinates_do_not_produce_a_torsion() -> None:
    assert dihedral_degrees([float("nan"), 0, 0], np.ones(3), np.zeros(3), [1, 0, 0]) is None


@pytest.mark.parametrize("observed_length", [0, 10])
def test_unresolved_or_unobserved_residues_are_not_called_coil(tmp_path: Path, observed_length: int) -> None:
    path = tmp_path / "test.cif"
    _mmcif(path, "HELX_P X X A A 1 10 101 110\n", observed_length=observed_length)
    assert secondary_fractions_from_mmcif_annotations(str(path), chain_id="A", length=100) == (
        "unknown",
        None,
        None,
        None,
        None,
    )


def _known_alpha_backbone() -> tuple[np.ndarray, np.ndarray]:
    # Independent analytic geometry: central phi=-60 and psi=-45 degrees;
    # both connecting peptide C-N lengths are exactly 1.33 angstrom.
    positions = np.zeros((3, 14, 3), dtype=np.float64)
    mask = np.zeros((3, 14), dtype=np.float64)
    positions[0, 2] = [0, 1.33 / 2, 1.33 * np.sqrt(3) / 2]
    positions[1, 0] = [0, 0, 0]
    positions[1, 1] = [1, 0, 0]
    positions[1, 2] = [1, 1, 0]
    positions[2, 0] = [1 - 1.33 / np.sqrt(2), 1, -1.33 / np.sqrt(2)]
    mask[0, 2] = mask[2, 0] = 1
    mask[1, :3] = 1
    return positions, mask


def test_secondary_torsion_connected_analytic_alpha_oracle() -> None:
    positions, mask = _known_alpha_backbone()
    assert secondary_fractions_from_atom14(positions, mask) == ("alpha", 1.0, 0.0, 0.0)


@pytest.mark.parametrize("side", [0, 2])
def test_secondary_torsions_do_not_bridge_disconnected_fragments(side: int) -> None:
    positions, mask = _known_alpha_backbone()
    positions[side] += 20
    assert secondary_fractions_from_atom14(positions, mask) == ("unknown", None, None, None)


@pytest.mark.parametrize("bad_mask", [float("nan"), -1.0, 0.5, 2.0])
def test_secondary_torsions_reject_nonbinary_observation_masks(bad_mask: float) -> None:
    positions, mask = _known_alpha_backbone()
    mask[1, 0] = bad_mask
    assert secondary_fractions_from_atom14(positions, mask) == ("unknown", None, None, None)


def test_secondary_torsions_ignore_unobserved_coordinate_values() -> None:
    positions, mask = _known_alpha_backbone()
    positions[mask == 0] = np.nan
    assert secondary_fractions_from_atom14(positions, mask) == ("alpha", 1.0, 0.0, 0.0)
