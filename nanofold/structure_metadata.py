from __future__ import annotations

from typing import Any

import gemmi
import numpy as np

STRUCTURE_METADATA_SCHEMA_VERSION = 2


def dihedral_degrees(a: Any, b: Any, c: Any, d: Any) -> float | None:
    p0 = np.asarray(a, dtype=np.float64)
    p1 = np.asarray(b, dtype=np.float64)
    p2 = np.asarray(c, dtype=np.float64)
    p3 = np.asarray(d, dtype=np.float64)
    if any(point.shape != (3,) or not np.isfinite(point).all() for point in (p0, p1, p2, p3)):
        return None
    b0 = -(p1 - p0)
    b1 = p2 - p1
    b2 = p3 - p2
    norm = np.linalg.norm(b1)
    if norm <= 1e-8:
        return None
    b1 = b1 / norm
    v = b0 - np.dot(b0, b1) * b1
    w = b2 - np.dot(b2, b1) * b1
    if np.linalg.norm(v) <= 1e-8 or np.linalg.norm(w) <= 1e-8:
        return None
    x = np.dot(v, w)
    y = np.dot(np.cross(b1, v), w)
    return float(np.degrees(np.arctan2(y, x)))


def secondary_class_from_fractions(
    helix_fraction: float | None,
    beta_fraction: float | None,
    coil_fraction: float | None,
) -> str:
    if helix_fraction is None or beta_fraction is None:
        return "unknown"
    helix = float(helix_fraction)
    beta = float(beta_fraction)
    fractions = [helix, beta] + ([] if coil_fraction is None else [float(coil_fraction)])
    if any(not np.isfinite(value) or not 0 <= value <= 1 for value in fractions) or sum(fractions) > 1.000001:
        return "unknown"
    if helix >= 0.40 and beta < 0.20:
        return "alpha"
    if beta >= 0.30 and helix < 0.20:
        return "beta"
    if helix >= 0.20 and beta >= 0.15:
        return "alpha_beta"
    if coil_fraction is not None and float(coil_fraction) >= 0.70:
        return "coil_or_sparse"
    return "mixed_low_confidence"


def normalize_secondary_class(value: object) -> str:
    text = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "alpha": "alpha",
        "all_alpha": "alpha",
        "mainly_alpha": "alpha",
        "helix": "alpha",
        "beta": "beta",
        "all_beta": "beta",
        "mainly_beta": "beta",
        "sheet": "beta",
        "strand": "beta",
        "alpha_beta": "alpha_beta",
        "mixed_alpha_beta": "alpha_beta",
        "alpha/beta": "alpha_beta",
        "alpha+beta": "alpha_beta",
        "mixed": "alpha_beta",
        "coil": "coil_or_sparse",
        "few_secondary_structures": "coil_or_sparse",
        "little_secondary_structure": "coil_or_sparse",
    }
    return aliases.get(text, text if text else "unknown")


def normalize_domain_architecture_class(value: object) -> str:
    text = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    text = text.replace("+", "_").replace("/", "_")
    if not text:
        return "unknown"
    if text in {"alpha", "all_alpha", "mainly_alpha", "alpha_bundle", "alpha_bundles", "alpha_array"}:
        return "alpha"
    if text in {
        "beta",
        "all_beta",
        "mainly_beta",
        "beta_sheet",
        "beta_sandwich",
        "beta_sandwiches",
        "beta_barrel",
    }:
        return "beta"
    if text in {"alpha_beta", "a_b", "a_b_two_layers", "a_b_three_layers", "a_b_four_layers", "mixed"}:
        return "alpha_beta"
    if text in {"coil", "few_secondary_structures", "little_secondary_structure", "small_protein"}:
        return "coil_or_sparse"
    if "alpha" in text and "beta" in text:
        return "alpha_beta"
    if text.startswith("a_b") or "a_b" in text:
        return "alpha_beta"
    if "alpha" in text or "helical" in text or "helix" in text:
        return "alpha"
    if "beta" in text or "sandwich" in text or "barrel" in text:
        return "beta"
    return text


def domain_architecture_from_cath_class(value: object) -> str:
    text = str(value or "").strip().lower()
    if text in {"1", "mainly alpha", "mainly_alpha"}:
        return "alpha"
    if text in {"2", "mainly beta", "mainly_beta"}:
        return "beta"
    if text in {"3", "alpha beta", "alpha_beta"}:
        return "alpha_beta"
    if text in {"4", "few secondary structures", "few_secondary_structures"}:
        return "coil_or_sparse"
    return normalize_domain_architecture_class(text)


def domain_architecture_from_scop_sccs(value: object) -> str:
    text = str(value or "").strip().lower()
    if not text:
        return "unknown"
    code = text[0]
    if code == "a":
        return "alpha"
    if code == "b":
        return "beta"
    if code in {"c", "d"}:
        return "alpha_beta"
    if code in {"g", "h"}:
        return "coil_or_sparse"
    return "other"


def _row_value(row: Any, *keys: str) -> str:
    for key in keys:
        try:
            value = row[key]
        except Exception:
            continue
        if value not in {None, "", "?", "."}:
            return str(value)
    return ""


def secondary_fractions_from_mmcif_annotations(
    mmcif_path: str,
    *,
    chain_id: str,
    length: int,
) -> tuple[str, float | None, float | None, float | None, str | None]:
    """Return broad secondary fractions from mmCIF annotation categories.

    This supports DSSP-annotated mmCIF files and depositor/MAXIT annotations.
    It intentionally collapses detailed states into helix, beta, and coil
    buckets for split stratification. Complete first-model CA coverage is
    required so unresolved residues are never counted as coil.
    """
    if length <= 0:
        return "unknown", None, None, None, None
    try:
        block = gemmi.cif.read_file(str(mmcif_path)).sole_block()
    except Exception:
        return "unknown", None, None, None, None

    helix_residues: set[int] = set()
    beta_residues: set[int] = set()
    author_to_label: dict[str, set[str]] = {}
    for category, author_key, label_key in (
        ("_atom_site.", "auth_asym_id", "label_asym_id"),
        ("_pdbx_poly_seq_scheme.", "pdb_strand_id", "asym_id"),
    ):
        table = block.find_mmcif_category(category)
        if table:
            for row in table:
                author = _row_value(row, category + author_key)
                label = _row_value(row, category + label_key)
                if author and label:
                    author_to_label.setdefault(author, set()).add(label)

    observed_residues: set[int] = set()
    atoms = block.find_mmcif_category("_atom_site.")
    first_model: str | None = None
    if atoms:
        for row in atoms:
            model = _row_value(row, "_atom_site.pdbx_PDB_model_num") or "1"
            if first_model is None:
                first_model = model
            if model != first_model:
                continue
            author = _row_value(row, "_atom_site.auth_asym_id")
            label = _row_value(row, "_atom_site.label_asym_id")
            matches_author = author == chain_id if author else label in author_to_label.get(chain_id, set())
            if not matches_author:
                continue
            if _row_value(row, "_atom_site.label_atom_id", "_atom_site.auth_atom_id") != "CA":
                continue
            seq_id = _row_value(row, "_atom_site.label_seq_id")
            if not seq_id.isdigit() or not 1 <= int(seq_id) <= length:
                continue
            try:
                coordinates = [float(_row_value(row, f"_atom_site.Cartn_{axis}")) for axis in ("x", "y", "z")]
                occupancy = float(_row_value(row, "_atom_site.occupancy") or "1")
            except ValueError:
                continue
            if np.isfinite(coordinates).all() and np.isfinite(occupancy) and occupancy > 0:
                observed_residues.add(int(seq_id))
    if observed_residues != set(range(1, length + 1)):
        return "unknown", None, None, None, None

    def matches_chain(row: Any, category: str, end: str) -> bool:
        author = _row_value(row, f"{category}{end}_auth_asym_id")
        if author:
            return author == chain_id
        label = _row_value(row, f"{category}{end}_label_asym_id")
        return bool(label and label in author_to_label.get(chain_id, set()))

    for category in ("_struct_conf.", "_struct_sheet_range."):
        table = block.find_mmcif_category(category)
        if not table:
            continue
        for row in table:
            if not (matches_chain(row, category, "beg") and matches_chain(row, category, "end")):
                continue
            # Label sequence IDs index the polymer sequence. Author residue numbers
            # can include offsets/insertion codes and must not be treated as indices.
            beg = _row_value(row, category + "beg_label_seq_id")
            end = _row_value(row, category + "end_label_seq_id")
            if not beg.isdigit() or not end.isdigit():
                continue
            start, stop = int(beg), int(end)
            if not 1 <= start <= stop <= length:
                continue
            if category == "_struct_sheet_range.":
                target = beta_residues
            else:
                conf_type = _row_value(row, "_struct_conf.conf_type_id").upper()
                if conf_type.startswith("HELX"):
                    target = helix_residues
                elif conf_type.startswith("STRN"):
                    target = beta_residues
                else:
                    continue
            target.update(range(start, stop + 1))

    if helix_residues & beta_residues:
        return "unknown", None, None, None, None
    beta_residues -= helix_residues
    annotated = len(helix_residues) + len(beta_residues)
    if annotated == 0:
        return "unknown", None, None, None, None
    helix_fraction = len(helix_residues) / float(length)
    beta_fraction = len(beta_residues) / float(length)
    coil_fraction = max(0.0, 1.0 - helix_fraction - beta_fraction)
    return (
        secondary_class_from_fractions(helix_fraction, beta_fraction, coil_fraction),
        helix_fraction,
        beta_fraction,
        coil_fraction,
        "mmcif_annotations",
    )


def secondary_fractions_from_atom14(
    atom14_positions: np.ndarray,
    atom14_mask: np.ndarray,
) -> tuple[str, float | None, float | None, float | None]:
    """Return alpha/beta/coil profile from atom14 backbone torsions.

    This is a deterministic split-stratification signal, not a DSSP replacement.
    It uses broad Ramachandran regions from N/CA/C slots so official split
    generation can avoid obvious alpha/beta imbalance when richer domain-class
    resources are unavailable.
    """
    pos = np.asarray(atom14_positions)
    mask = np.asarray(atom14_mask).astype(bool)
    if pos.ndim != 3 or pos.shape[1:] != (14, 3) or mask.shape != pos.shape[:2]:
        return "unknown", None, None, None

    n_slot, ca_slot, c_slot = 0, 1, 2
    helix = 0
    beta = 0
    assigned = 0
    for idx in range(1, pos.shape[0] - 1):
        required = (
            mask[idx - 1, c_slot]
            and mask[idx, n_slot]
            and mask[idx, ca_slot]
            and mask[idx, c_slot]
            and mask[idx + 1, n_slot]
        )
        if not required:
            continue
        phi = dihedral_degrees(pos[idx - 1, c_slot], pos[idx, n_slot], pos[idx, ca_slot], pos[idx, c_slot])
        psi = dihedral_degrees(pos[idx, n_slot], pos[idx, ca_slot], pos[idx, c_slot], pos[idx + 1, n_slot])
        if phi is None or psi is None:
            continue
        assigned += 1
        if -120.0 <= phi <= -25.0 and -100.0 <= psi <= 45.0:
            helix += 1
        elif (-180.0 <= phi <= -60.0 and 45.0 <= psi <= 180.0) or (45.0 <= phi <= 180.0 and -180.0 <= psi <= -120.0):
            beta += 1
    if assigned == 0:
        return "unknown", None, None, None
    helix_fraction = helix / float(assigned)
    beta_fraction = beta / float(assigned)
    coil_fraction = max(0.0, 1.0 - helix_fraction - beta_fraction)
    return (
        secondary_class_from_fractions(helix_fraction, beta_fraction, coil_fraction),
        helix_fraction,
        beta_fraction,
        coil_fraction,
    )
