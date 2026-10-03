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

Alternate locations are selected as coherent residue conformers, retaining
shared atoms and positive occupancy only. Backbone support is fixed before
choosing conformers for continuity across deposited polymer neighbors;
occupancy breaks equal-geometry choices. Coordinates are never adjusted.
The first deposited model anchors identity and atom support. A later whole
model can replace it only with identical identity/support and fewer backbone
violations; equal-geometry models retain their deposited order.
"""

from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
from math import isclose
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
    * ``atom14_mask``: ``(L, 14)`` 1.0 for positive-occupancy coordinates
      in the selected residue conformer, 0.0 otherwise. Unobserved atoms
      are masked out of downstream losses.
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
    # Zero-based positions in the deposited polymer sequence, not the compact
    # observed-coordinate sequence. -1 means mmCIF did not supply label_seq_id.
    polymer_residue_index: np.ndarray
    # Complete deposited entity sequence, including residues without coordinates.
    polymer_sequence: str | None
    resolution: float
    resolved_author_chain_id: str
    chain_mapping_method: str

    @property
    def ca_coords(self) -> np.ndarray:
        """(L, 3) float32 Cα coordinates — slot ``CA_ATOM14_SLOT`` of atom14."""
        return self.atom14_positions[:, CA_ATOM14_SLOT, :]

    @property
    def ca_mask(self) -> np.ndarray:
        """(L,) bool — True where the selected Cα atom has positive occupancy."""
        return self.atom14_mask[:, CA_ATOM14_SLOT] > 0.5


@dataclass
class _ResidueConformer:
    positions: np.ndarray
    mask: np.ndarray
    occupancy: float


def _residue_conformers(residue: gemmi.Residue) -> list[_ResidueConformer]:
    """One named alternate plus shared atoms, with geometry-independent support.

    First maximize observed backbone atom count. Incomparable equal-count
    masks use the fixed CA, N, C, O preference. Every returned candidate has
    that exact backbone mask, so geometry cannot win by hiding a bond.
    """
    slot_table = ATOM14_INDEX.get(residue.name.upper(), {})
    atoms: dict[tuple[str, str], gemmi.Atom] = {}
    for atom in residue:
        if atom.name not in slot_table:
            continue
        occupancy = float(atom.occ)
        if not np.isfinite(occupancy):
            raise ValueError(f"Nonfinite occupancy for atom {atom.name}")
        if occupancy < 0:
            raise ValueError(f"Negative occupancy for atom {atom.name}")
        if occupancy == 0:
            continue
        coordinates = (atom.pos.x, atom.pos.y, atom.pos.z)
        if not np.isfinite(np.asarray(coordinates, dtype=np.float32)).all():
            raise ValueError(f"Nonfinite coordinates for {residue.name} atom {atom.name}")
        altloc = "" if atom.altloc in ("", "\0", " ", ".", "?") else atom.altloc
        key = (atom.name, altloc)
        previous = atoms.get(key)
        # Duplicate records for the same site must not make atom row order a
        # tie-breaker. This never mixes different named alternate locations.
        if previous is None or (occupancy, *coordinates) > (
            float(previous.occ), previous.pos.x, previous.pos.y, previous.pos.z,
        ):
            atoms[key] = atom

    altlocs = sorted({altloc for _, altloc in atoms if altloc}) or [""]
    candidates = []
    for altloc in altlocs:
        positions = np.zeros((ATOM14_NUM_SLOTS, 3), dtype=np.float32)
        mask = np.zeros((ATOM14_NUM_SLOTS,), dtype=np.float32)
        occupancies = []
        for atom_name, slot in slot_table.items():
            selected_atom = atoms.get((atom_name, altloc))
            if selected_atom is not None:
                occupancies.append(float(selected_atom.occ))
            else:
                selected_atom = atoms.get((atom_name, ""))
            if selected_atom is not None:
                positions[slot] = (selected_atom.pos.x, selected_atom.pos.y, selected_atom.pos.z)
                mask[slot] = 1.0
        occupancy = sum(occupancies) / len(occupancies) if occupancies else 0.0
        candidates.append(_ResidueConformer(positions, mask, occupancy))

    def support(candidate: _ResidueConformer) -> tuple[int, tuple[bool, ...]]:
        backbone = tuple(bool(candidate.mask[slot]) for slot in (1, 0, 2, 3))
        return sum(backbone), backbone

    best_support = max(support(candidate) for candidate in candidates)
    return [candidate for candidate in candidates if support(candidate) == best_support]


def _backbone_violations(left: _ResidueConformer, right: _ResidueConformer) -> tuple[int, int]:
    violations = []
    for left_slot, right_slot, lower, upper in ((2, 0, 1.0, 1.8), (1, 1, 2.5, 4.5)):
        outside = False
        if left.mask[left_slot] and right.mask[right_slot]:
            distance = float(np.linalg.norm(
                left.positions[left_slot].astype(np.float64) - right.positions[right_slot].astype(np.float64)
            ))
            outside = not lower <= distance <= upper
        violations.append(int(outside))
    return violations[0], violations[1]


def _polymer_neighbors(left: gemmi.Residue, right: gemmi.Residue) -> bool:
    return (
        left.subchain == right.subchain
        and left.label_seq is not None and right.label_seq is not None
        and right.label_seq == left.label_seq + 1
    )


def _select_chain_conformers(residues: list[gemmi.Residue]) -> list[_ResidueConformer]:
    """Minimize peptide, then CA violations; maximize occupancy on ties.

    Only consecutive known deposited polymer positions in the same subchain
    contribute geometry. Sorted alternate labels and predecessor order make
    ties deterministic, without requiring equal labels across residues.
    """
    candidates = [_residue_conformers(residue) for residue in residues]
    if all(len(choices) == 1 for choices in candidates):
        return [choices[0] for choices in candidates]

    costs = [(0, 0, -choice.occupancy) for choice in candidates[0]]
    predecessors: list[list[int]] = []
    for index in range(1, len(residues)):
        adjacent = _polymer_neighbors(residues[index - 1], residues[index])
        next_costs = []
        previous_indices = []
        for choice in candidates[index]:
            options = []
            for previous_index, previous in enumerate(candidates[index - 1]):
                peptide, ca = _backbone_violations(previous, choice) if adjacent else (0, 0)
                cost = costs[previous_index]
                options.append((cost[0] + peptide, cost[1] + ca, cost[2] - choice.occupancy, previous_index))
            best = min(options)
            next_costs.append(best[:3])
            previous_indices.append(best[3])
        costs = next_costs
        predecessors.append(previous_indices)

    selected_index = min(range(len(costs)), key=lambda index: costs[index])
    selected = [candidates[-1][selected_index]]
    for index in range(len(residues) - 2, -1, -1):
        selected_index = predecessors[index][selected_index]
        selected.append(candidates[index][selected_index])
    return selected[::-1]


def _chain_backbone_violations(
    residues: list[gemmi.Residue], conformers: list[_ResidueConformer],
) -> tuple[int, int]:
    peptide = ca = 0
    for index in range(len(residues) - 1):
        if _polymer_neighbors(residues[index], residues[index + 1]):
            edge_peptide, edge_ca = _backbone_violations(conformers[index], conformers[index + 1])
            peptide += edge_peptide
            ca += edge_ca
    return peptide, ca


def _residue_signature(residues: list[gemmi.Residue]) -> list[tuple[str, str, int | None, str | None]]:
    return [
        (residue.name.upper(), residue.subchain, residue.label_seq,
         str(residue.seqid) if residue.label_seq is None else None)
        for residue in residues
    ]


def _ordered_polymer_residues(chain: gemmi.Chain, label_chain_id: str | None = None) -> list[gemmi.Residue]:
    residues = [
        residue for residue in chain
        if residue.entity_type == gemmi.EntityType.Polymer
        and (label_chain_id is None or residue.subchain == label_chain_id)
    ]
    # Atom-site row order is not polymer sequence order. Where the deposited
    # indices fully specify one polymer, use them for sequence and geometry.
    # Missing indices retain deposited order rather than guessing from author IDs.
    if len({residue.subchain for residue in residues}) == 1 and all(
        residue.label_seq is not None for residue in residues
    ):
        residues.sort(key=lambda residue: residue.label_seq if residue.label_seq is not None else -1)
    return residues


def _chain_polymer_sequence(chain: gemmi.Chain) -> str:
    return "".join(one_letter_from_resname(residue.name) for residue in _ordered_polymer_residues(chain))


def _resolve_chain(
    model: gemmi.Model,
    *,
    chain_id: str,
    expected_sequence: Optional[str],
) -> Tuple[gemmi.Chain, Optional[str], str]:
    author_matches = [ch for ch in model if ch.name == chain_id]
    if len(author_matches) > 1:
        raise ValueError(f"Ambiguous author chain ID '{chain_id}'")
    if author_matches:
        return author_matches[0], None, "author_id"

    # Gemmi retains label_asym_id in Residue.subchain and auth_asym_id in Chain.name.
    label_matches = [
        ch for ch in model
        if any(res.entity_type == gemmi.EntityType.Polymer and res.subchain == chain_id for res in ch)
    ]
    if len(label_matches) > 1:
        raise ValueError(f"Ambiguous label chain ID '{chain_id}'")
    if label_matches:
        return label_matches[0], chain_id, "label_id"

    if expected_sequence:
        expected = expected_sequence.upper()
        candidates: list[tuple[float, gemmi.Chain]] = []
        for ch in model:
            seq = _chain_polymer_sequence(ch).upper()
            if not seq:
                continue
            ratio = SequenceMatcher(a=expected, b=seq, autojunk=False).ratio()
            if ratio >= 0.85:
                candidates.append((ratio, ch))
        if candidates:
            best_ratio = max(ratio for ratio, _ in candidates)
            best_matches = [ch for ratio, ch in candidates if isclose(ratio, best_ratio)]
            if len(best_matches) != 1:
                names = ", ".join(sorted(ch.name for ch in best_matches))
                raise ValueError(f"Ambiguous sequence mapping for chain '{chain_id}': equally matching author chains {names}")
            return best_matches[0], None, "sequence"

    raise KeyError


def extract_chain_atoms(
    mmcif_path: str | Path,
    pdb_id: str,
    chain_id: str,
    expected_sequence: Optional[str] = None,
    require_full_match: bool = True,
) -> ChainAtoms:
    """Parse ``mmcif_path`` and return the atom14 structure for ``chain_id``.

    The first deposited model anchors chain identity and observed atom support.
    A later whole model is eligible only with identical residue identity, chain
    mapping and atom14 masks; fewer peptide, then CA spacing violations choose
    between eligible models, with deposited order breaking ties. Invalid first
    models fail; malformed or noncomparable later models are ineligible.

    Within each model, positive-occupancy residue conformers are selected for
    observed backbone support, deposited-neighbor continuity, then occupancy.
    Polymer residues only are kept, numbered contiguously from sequence order.
    No coordinates are idealized or removed because of geometry.

    If ``expected_sequence`` is provided and ``require_full_match=True``,
    the structure-derived sequence must match exactly. Set
    ``require_full_match=False`` to allow length mismatch (the caller
    typically projects structure→query via sequence alignment in that
    case). IDs resolve by author ID, explicit label ID, then a unique best
    sequence match. Sequence ties are rejected; the requested ID and actual
    author ID are both preserved in the result.
    """
    mmcif_path = Path(mmcif_path)
    st = gemmi.read_structure(str(mmcif_path))
    st.setup_entities()

    model = st[0]
    try:
        chain, label_chain_id, chain_mapping_method = _resolve_chain(
            model, chain_id=chain_id, expected_sequence=expected_sequence,
        )
    except KeyError:
        raise KeyError(f"Chain '{chain_id}' not found in {mmcif_path}") from None

    sequence_letters: List[str] = []
    residues = _ordered_polymer_residues(chain, label_chain_id)
    polymer_residue_indices: List[int] = []
    selected_subchains = {
        res.subchain for res in chain
        if res.entity_type == gemmi.EntityType.Polymer
        and (label_chain_id is None or res.subchain == label_chain_id)
    }
    if len(selected_subchains) > 1:
        raise ValueError(f"Author chain '{chain.name}' contains multiple polymer label chains")
    entity = (
        st.get_entity_of(chain.get_subchain(next(iter(selected_subchains))))
        if selected_subchains else None
    )
    polymer_sequence = (
        "".join(one_letter_from_resname(name) for name in entity.full_sequence)
        if entity is not None and entity.full_sequence else None
    )

    for res in residues:
        residue_name_3 = res.name.upper()
        one_letter = one_letter_from_resname(residue_name_3)
        sequence_letters.append(one_letter)
        if res.label_seq is not None and res.label_seq < 1:
            raise ValueError("Polymer label_seq_id must be positive")
        polymer_residue_indices.append(res.label_seq - 1 if res.label_seq is not None else -1)

    if not residues:
        raise ValueError(f"No polymer residues found for {pdb_id}_{chain_id} in {mmcif_path}")
    known_polymer_indices = [index for index in polymer_residue_indices if index >= 0]
    if len(set(known_polymer_indices)) != len(known_polymer_indices):
        raise ValueError("Ambiguous observed residues: duplicate polymer label_seq_id")

    sequence = "".join(sequence_letters)

    if expected_sequence is not None and require_full_match and sequence != expected_sequence:
        raise ValueError(
            f"Sequence mismatch for {pdb_id}_{chain_id}: "
            f"structure seq len={len(sequence)}, expected len={len(expected_sequence)}"
        )

    conformers = _select_chain_conformers(residues)
    atom14_mask = np.stack([conformer.mask for conformer in conformers], axis=0)
    best_score = _chain_backbone_violations(residues, conformers) if len(st) > 1 else (0, 0)
    signature = _residue_signature(residues)
    for model_index, alternative_model in enumerate(st):
        if best_score == (0, 0):
            break
        if model_index == 0:
            continue
        try:
            alternative_chain, alternative_label, alternative_method = _resolve_chain(
                alternative_model, chain_id=chain_id, expected_sequence=expected_sequence,
            )
        except (KeyError, ValueError):
            continue
        if (alternative_chain.name, alternative_label, alternative_method) != (
            chain.name, label_chain_id, chain_mapping_method,
        ):
            continue
        alternative_residues = _ordered_polymer_residues(alternative_chain, alternative_label)
        if _residue_signature(alternative_residues) != signature:
            continue
        # Identical subchains in this Structure share the same deposited entity
        # sequence. The signature also preserves known polymer positions and,
        # when absent, author numbering including insertion codes.
        try:
            alternative_conformers = _select_chain_conformers(alternative_residues)
        except ValueError:
            # An invalid alternative cannot invalidate the first model or be
            # repaired/masked to become eligible for selection.
            continue
        alternative_mask = np.stack([item.mask for item in alternative_conformers], axis=0)
        if not np.array_equal(alternative_mask, atom14_mask):
            continue
        score = _chain_backbone_violations(alternative_residues, alternative_conformers)
        if score < best_score:
            conformers, best_score = alternative_conformers, score

    atom14_positions = np.stack([conformer.positions for conformer in conformers], axis=0)
    residue_index = np.arange(len(sequence), dtype=np.int32)

    resolution_value = getattr(st, "resolution", 0.0) or 0.0
    try:
        resolution = float(resolution_value)
    except (TypeError, ValueError):
        resolution = 0.0
    if not np.isfinite(resolution):
        raise ValueError(f"Nonfinite resolution in {mmcif_path}")
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
        polymer_residue_index=np.asarray(polymer_residue_indices, dtype=np.int32),
        polymer_sequence=polymer_sequence,
        resolution=resolution,
        resolved_author_chain_id=chain.name,
        chain_mapping_method=chain_mapping_method,
    )
