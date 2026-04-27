"""Residue constants: atom14 slot tables for the 20 standard amino acids.

The atom14 layout assigns each residue a fixed 14-slot vector of atom
coordinates, with empty strings marking unused slots. Slot ordering matches
DeepMind's AF2 canonical layout — slot 0 is N, slot 1 is CA, slot 2 is C,
slot 3 is O, slot 4 is CB (or empty for glycine), and so on through the
sidechain atoms.

Only the atom14 slot tables are included here. The chi / rigid-group /
atom37 constants needed for full AlphaFold training live in the reference
``min-AlphaFold`` codebase, and submissions that want them can vendor them
separately.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

# Canonical AF2 one-letter ordering.
RESTYPES: List[str] = list("ARNDCQEGHILKMFPSTWYV")
RESTYPE_ORDER: Dict[str, int] = {restype: idx for idx, restype in enumerate(RESTYPES)}

RESTYPE_1TO3: Dict[str, str] = {
    "A": "ALA", "R": "ARG", "N": "ASN", "D": "ASP", "C": "CYS",
    "Q": "GLN", "E": "GLU", "G": "GLY", "H": "HIS", "I": "ILE",
    "L": "LEU", "K": "LYS", "M": "MET", "F": "PHE", "P": "PRO",
    "S": "SER", "T": "THR", "W": "TRP", "Y": "TYR", "V": "VAL",
}
RESTYPE_3TO1: Dict[str, str] = {three: one for one, three in RESTYPE_1TO3.items()}
RESTYPE_INDEX_TO_3: Tuple[str, ...] = tuple(RESTYPE_1TO3[one] for one in RESTYPES) + ("UNK",)

# Atom14 slot ordering: (N, CA, C, O, then up to 10 sidechain atoms).
# Slot 0 = N, Slot 1 = CA, Slot 2 = C, Slot 3 = O.
RESTYPE_NAME_TO_ATOM14_NAMES: Dict[str, List[str]] = {
    "ALA": ["N", "CA", "C", "O", "CB", "",    "",    "",    "",    "",    "",    "",    "",    ""],
    "ARG": ["N", "CA", "C", "O", "CB", "CG",  "CD",  "NE",  "CZ",  "NH1", "NH2", "",    "",    ""],
    "ASN": ["N", "CA", "C", "O", "CB", "CG",  "OD1", "ND2", "",    "",    "",    "",    "",    ""],
    "ASP": ["N", "CA", "C", "O", "CB", "CG",  "OD1", "OD2", "",    "",    "",    "",    "",    ""],
    "CYS": ["N", "CA", "C", "O", "CB", "SG",  "",    "",    "",    "",    "",    "",    "",    ""],
    "GLN": ["N", "CA", "C", "O", "CB", "CG",  "CD",  "OE1", "NE2", "",    "",    "",    "",    ""],
    "GLU": ["N", "CA", "C", "O", "CB", "CG",  "CD",  "OE1", "OE2", "",    "",    "",    "",    ""],
    "GLY": ["N", "CA", "C", "O", "",   "",    "",    "",    "",    "",    "",    "",    "",    ""],
    "HIS": ["N", "CA", "C", "O", "CB", "CG",  "ND1", "CD2", "CE1", "NE2", "",    "",    "",    ""],
    "ILE": ["N", "CA", "C", "O", "CB", "CG1", "CG2", "CD1", "",    "",    "",    "",    "",    ""],
    "LEU": ["N", "CA", "C", "O", "CB", "CG",  "CD1", "CD2", "",    "",    "",    "",    "",    ""],
    "LYS": ["N", "CA", "C", "O", "CB", "CG",  "CD",  "CE",  "NZ",  "",    "",    "",    "",    ""],
    "MET": ["N", "CA", "C", "O", "CB", "CG",  "SD",  "CE",  "",    "",    "",    "",    "",    ""],
    "PHE": ["N", "CA", "C", "O", "CB", "CG",  "CD1", "CD2", "CE1", "CE2", "CZ",  "",    "",    ""],
    "PRO": ["N", "CA", "C", "O", "CB", "CG",  "CD",  "",    "",    "",    "",    "",    "",    ""],
    "SER": ["N", "CA", "C", "O", "CB", "OG",  "",    "",    "",    "",    "",    "",    "",    ""],
    "THR": ["N", "CA", "C", "O", "CB", "OG1", "CG2", "",    "",    "",    "",    "",    "",    ""],
    "TRP": ["N", "CA", "C", "O", "CB", "CG",  "CD1", "CD2", "NE1", "CE2", "CE3", "CZ2", "CZ3", "CH2"],
    "TYR": ["N", "CA", "C", "O", "CB", "CG",  "CD1", "CD2", "CE1", "CE2", "CZ",  "OH",  "",    ""],
    "VAL": ["N", "CA", "C", "O", "CB", "CG1", "CG2", "",    "",    "",    "",    "",    "",    ""],
    "UNK": ["",  "",   "",  "",  "",   "",    "",    "",    "",    "",    "",    "",    "",    ""],
}

# (residue_name) -> (atom_name -> slot index). Empty slot strings are dropped.
ATOM14_INDEX: Dict[str, Dict[str, int]] = {
    residue_name: {atom_name: idx for idx, atom_name in enumerate(atoms) if atom_name}
    for residue_name, atoms in RESTYPE_NAME_TO_ATOM14_NAMES.items()
}

ATOM14_NAMES_BY_RESTYPE_INDEX: Tuple[Tuple[str, ...], ...] = tuple(
    tuple(RESTYPE_NAME_TO_ATOM14_NAMES[RESTYPE_1TO3[one]]) for one in RESTYPES
) + (tuple(RESTYPE_NAME_TO_ATOM14_NAMES["UNK"]),)

# Chi definitions for the first two side-chain torsions used by the official
# side-chain dihedral score. Residues without the relevant atoms are skipped.
CHI1_CHI2_ATOM_NAMES: Dict[str, Tuple[Tuple[str, str, str, str], ...]] = {
    "ALA": (),
    "ARG": (("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "CD")),
    "ASN": (("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "OD1")),
    "ASP": (("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "OD1")),
    "CYS": (("N", "CA", "CB", "SG"),),
    "GLN": (("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "CD")),
    "GLU": (("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "CD")),
    "GLY": (),
    "HIS": (("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "ND1")),
    "ILE": (("N", "CA", "CB", "CG1"), ("CA", "CB", "CG1", "CD1")),
    "LEU": (("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "CD1")),
    "LYS": (("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "CD")),
    "MET": (("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "SD")),
    "PHE": (("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "CD1")),
    "PRO": (("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "CD")),
    "SER": (("N", "CA", "CB", "OG"),),
    "THR": (("N", "CA", "CB", "OG1"),),
    "TRP": (("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "CD1")),
    "TYR": (("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "CD1")),
    "VAL": (("N", "CA", "CB", "CG1"),),
    "UNK": (),
}

# Some chi definitions are symmetric under a 180 degree flip of terminal atoms.
CHI_PI_PERIODIC: Dict[Tuple[str, int], bool] = {
    ("ASP", 1): True,
    ("PHE", 1): True,
    ("TYR", 1): True,
    ("VAL", 0): True,
    ("LEU", 1): True,
}

# Canonical atom14 slot for the Cα atom (same for every standard residue).
CA_ATOM14_SLOT = 1

ATOM14_NUM_SLOTS = 14

# (21, 14) 1.0 where a slot holds a real atom for that restype (20 AAs + UNK).
STANDARD_ATOM14_MASK: np.ndarray = np.zeros((21, ATOM14_NUM_SLOTS), dtype=np.float32)
for _restype_idx, _one in enumerate(RESTYPES):
    _three = RESTYPE_1TO3[_one]
    for _slot, _atom_name in enumerate(RESTYPE_NAME_TO_ATOM14_NAMES[_three]):
        if _atom_name:
            STANDARD_ATOM14_MASK[_restype_idx, _slot] = 1.0
# UNK (index 20) has no atoms — mask row stays all zeros.
