"""mmCIF parser → per-chain atom14 structures for ground-truth supervision.

Built on ``gemmi`` — a fast, strict, well-tested mmCIF/PDB parser. We keep
the gemmi dependency (rather than vendoring a pure-Python parser) because
the benchmark preprocesses tens of thousands of chains and parser
performance matters.

Output is a ``ChainAtoms`` dataclass carrying atom14 supervision
(``atom14_positions``, ``atom14_mask``, ``residue_index``, ``resolution``).
Cα coordinates and masks are exposed as derived views of the atom14 labels
for metrics and diagnostics. Residue indexing is **contiguous 0..N-1** from
sequence order, not the author numbering from the mmCIF — this matches AF2
supplement 1.2.9 and avoids the edge cases around insertion codes and gaps
in author numbering.

Altloc handling follows AF2 supplement 1.2.1 *"taking the one with the
largest occupancy"*: for each atom we pick the altloc with the highest
occupancy, breaking ties in favour of the "no altloc" / "A" conformer.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import gemmi
import numpy as np

from .a3m import sequence_to_ids
from .residue_constants import (
    ATOM14_INDEX,
    ATOM14_NUM_SLOTS,
    CA_ATOM14_SLOT,
    RESTYPE_3TO1,
)


def one_letter_from_resname(resname: str) -> str:
    return RESTYPE_3TO1.get(resname.upper(), "X")


@dataclass
class ChainAtoms:
    """One parsed chain ready to become a labelled training example.

    Fields match what the downstream data pipeline expects:

    * ``aatype``: ``(L,)`` int32 IDs using the alphabet from ``a3m.py``.
    * ``residue_index``: ``(L,)`` int32 contiguous 0..L-1 (AF2 supplement
      1.2.9 — *not* the author numbering from the mmCIF).
    * ``atom14_positions``: ``(L, 14, 3)`` Å coordinates in the per-residue
      atom14 slot ordering from ``residue_constants``.
    * ``atom14_mask``: ``(L, 14)`` 1.0 where a coordinate was present in
      the mmCIF, 0.0 otherwise. Atoms missing in ground truth are masked
      out of downstream losses.
    * ``resolution``: Å (0.0 when none was found). Consumed by loss
      resolution filters (supplement 1.9.6 / 1.9.10 train on resolutions
      in [0.1, 3.0]).
    """

    pdb_id: str
    chain_id: str
    sequence: str
    aatype: np.ndarray
    residue_index: np.ndarray
    atom14_positions: np.ndarray
    atom14_mask: np.ndarray
    resolution: float

    @property
    def ca_coords(self) -> np.ndarray:
        """(L, 3) float32 Cα coordinates — slot ``CA_ATOM14_SLOT`` of atom14."""
        return self.atom14_positions[:, CA_ATOM14_SLOT, :]

    @property
    def ca_mask(self) -> np.ndarray:
        """(L,) bool — True where the Cα atom was present in the mmCIF."""
        return self.atom14_mask[:, CA_ATOM14_SLOT] > 0.5


def _best_atom_by_occupancy(atoms: List[gemmi.Atom]) -> Optional[gemmi.Atom]:
    """Pick the best altloc per AF2 supplement 1.2.1: highest occupancy, with
    the no-altloc / "A" conformer breaking ties."""
    if not atoms:
        return None

    def priority(a: gemmi.Atom) -> Tuple[int, float]:
        altloc = a.altloc or "\0"
        preferred = 1 if altloc in ("\0", "A") else 0
        return (preferred, float(a.occ))

    return max(atoms, key=priority)


def _collect_atom14(
    residue: gemmi.Residue,
    residue_name_3: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """Fill atom14 slots for one residue using the per-atom best-altloc atom."""
    positions = np.zeros((ATOM14_NUM_SLOTS, 3), dtype=np.float32)
    mask = np.zeros((ATOM14_NUM_SLOTS,), dtype=np.float32)

    slot_table = ATOM14_INDEX.get(residue_name_3)
    if slot_table is None:
        return positions, mask

    atoms_by_name: dict[str, List[gemmi.Atom]] = {}
    for atom in residue:
        atoms_by_name.setdefault(atom.name, []).append(atom)

    for atom_name, slot in slot_table.items():
        best = _best_atom_by_occupancy(atoms_by_name.get(atom_name, []))
        if best is None:
            continue
        positions[slot] = np.array([best.pos.x, best.pos.y, best.pos.z], dtype=np.float32)
        mask[slot] = 1.0

    return positions, mask


def extract_chain_atoms(
    mmcif_path: str | Path,
    pdb_id: str,
    chain_id: str,
    expected_sequence: Optional[str] = None,
    require_full_match: bool = True,
) -> ChainAtoms:
    """Parse ``mmcif_path`` and return the atom14 structure for ``chain_id``.

    Follows AF2 supplement 1.2.1 Parsing: first model only (rejects NMR
    ensembles); polymer residues only (skips HETATM ligands/water);
    altlocs resolved by occupancy with no-altloc / "A" preferred; residue
    numbering is contiguous 0..N-1 from sequence order, not author
    numbering.

    If ``expected_sequence`` is provided and ``require_full_match=True``,
    the structure-derived sequence must match exactly. Set
    ``require_full_match=False`` to allow length mismatch (the caller
    typically projects structure→query via sequence alignment in that
    case).
    """
    mmcif_path = Path(mmcif_path)
    st = gemmi.read_structure(str(mmcif_path))
    st.setup_entities()

    model = st[0]
    chain = None
    for ch in model:
        if ch.name == chain_id:
            chain = ch
            break
    if chain is None:
        raise KeyError(f"Chain '{chain_id}' not found in {mmcif_path}")

    sequence_letters: List[str] = []
    atom14_positions_list: List[np.ndarray] = []
    atom14_mask_list: List[np.ndarray] = []

    for res in chain:
        # Keep only polymer residues. Filtering on het_flag would also drop
        # valid amino acids, so we use entity_type.
        if res.entity_type != gemmi.EntityType.Polymer:
            continue

        residue_name_3 = res.name.upper()
        one_letter = one_letter_from_resname(residue_name_3)
        sequence_letters.append(one_letter)

        positions, mask = _collect_atom14(res, residue_name_3)
        atom14_positions_list.append(positions)
        atom14_mask_list.append(mask)

    if not atom14_positions_list:
        raise ValueError(f"No polymer residues found for {pdb_id}_{chain_id} in {mmcif_path}")

    sequence = "".join(sequence_letters)

    if expected_sequence is not None and require_full_match and sequence != expected_sequence:
        raise ValueError(
            f"Sequence mismatch for {pdb_id}_{chain_id}: "
            f"structure seq len={len(sequence)}, expected len={len(expected_sequence)}"
        )

    atom14_positions = np.stack(atom14_positions_list, axis=0)
    atom14_mask = np.stack(atom14_mask_list, axis=0)
    residue_index = np.arange(len(sequence), dtype=np.int32)

    resolution_value = getattr(st, "resolution", 0.0) or 0.0
    try:
        resolution = float(resolution_value)
    except (TypeError, ValueError):
        resolution = 0.0
    if resolution < 0.0:
        resolution = 0.0

    return ChainAtoms(
        pdb_id=pdb_id.lower(),
        chain_id=chain_id,
        sequence=sequence,
        aatype=sequence_to_ids(sequence),
        residue_index=residue_index,
        atom14_positions=atom14_positions,
        atom14_mask=atom14_mask,
        resolution=resolution,
    )
