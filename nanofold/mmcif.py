from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import gemmi
import numpy as np

# Same restype mapping as in a3m.py
RESTYPES = "ARNDCQEGHILKMFPSTWYV"
RESTYPE_TO_ID = {aa: i for i, aa in enumerate(RESTYPES)}
UNK_ID = 20


THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}


def one_letter_from_resname(resname: str) -> str:
    return THREE_TO_ONE.get(resname.upper(), "X")


def aatype_from_sequence(seq: str) -> np.ndarray:
    out = np.zeros((len(seq),), dtype=np.int32)
    for i, aa in enumerate(seq):
        out[i] = RESTYPE_TO_ID.get(aa.upper(), UNK_ID)
    return out


@dataclass
class ChainStructure:
    pdb_id: str
    chain_id: str
    sequence: str
    ca_coords: np.ndarray  # (L,3) float32
    ca_mask: np.ndarray    # (L,) bool


def extract_chain_ca(
    mmcif_path: str | Path,
    pdb_id: str,
    chain_id: str,
    expected_sequence: Optional[str] = None,
    require_full_match: bool = True,
) -> ChainStructure:
    """Extract Cα coordinates for a single chain from an mmCIF.

    This is intentionally *strict* by default to keep the benchmark simple:
    - if expected_sequence is provided, we require the structure-derived sequence to match exactly
      (unless require_full_match=False)

    Returns:
      ChainStructure with:
        sequence: the structure-derived sequence (one-letter)
        ca_coords: (L,3) with 0s for missing
        ca_mask: (L,) True where Cα present
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
        # Sometimes auth_asym_id differs; fall back to first chain with same label?
        raise KeyError(f"Chain '{chain_id}' not found in {mmcif_path}")

    seq_letters: List[str] = []
    ca_coords: List[np.ndarray] = []
    ca_mask: List[bool] = []

    for res in chain:
        if res.is_water():
            continue
        if res.het_flag != " ":
            # skip hetero residues (ligands)
            continue
        aa = one_letter_from_resname(res.name)
        seq_letters.append(aa)

        atom = res.find_atom("CA", altloc="\0")
        if atom is None:
            ca_coords.append(np.zeros((3,), dtype=np.float32))
            ca_mask.append(False)
        else:
            pos = atom.pos
            ca_coords.append(np.array([pos.x, pos.y, pos.z], dtype=np.float32))
            ca_mask.append(True)

    seq = "".join(seq_letters)
    if expected_sequence is not None:
        if require_full_match and seq != expected_sequence:
            raise ValueError(
                f"Sequence mismatch for {pdb_id}_{chain_id}: structure seq len={len(seq)}, expected len={len(expected_sequence)}"
            )

    return ChainStructure(
        pdb_id=pdb_id.lower(),
        chain_id=chain_id,
        sequence=seq,
        ca_coords=np.stack(ca_coords, axis=0),
        ca_mask=np.asarray(ca_mask, dtype=bool),
    )
