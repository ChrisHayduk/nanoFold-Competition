"""NanoFold atom14 tensors -> Protenix compact-atom training dictionaries.

Only the explicitly whitelisted sequence/MSA tensors become model inputs.
Target coordinates and masks are emitted only in the two label dictionaries.
"""

from __future__ import annotations

import hashlib
import math
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


RESTYPE_ATOM14_NAMES: tuple[tuple[str, ...], ...] = tuple(
    tuple(row)
    for row in (
        ("N", "CA", "C", "O", "CB", "", "", "", "", "", "", "", "", ""),
        ("N", "CA", "C", "O", "CB", "CG", "CD", "NE", "CZ", "NH1", "NH2", "", "", ""),
        ("N", "CA", "C", "O", "CB", "CG", "OD1", "ND2", "", "", "", "", "", ""),
        ("N", "CA", "C", "O", "CB", "CG", "OD1", "OD2", "", "", "", "", "", ""),
        ("N", "CA", "C", "O", "CB", "SG", "", "", "", "", "", "", "", ""),
        ("N", "CA", "C", "O", "CB", "CG", "CD", "OE1", "NE2", "", "", "", "", ""),
        ("N", "CA", "C", "O", "CB", "CG", "CD", "OE1", "OE2", "", "", "", "", ""),
        ("N", "CA", "C", "O", "", "", "", "", "", "", "", "", "", ""),
        ("N", "CA", "C", "O", "CB", "CG", "ND1", "CD2", "CE1", "NE2", "", "", "", ""),
        ("N", "CA", "C", "O", "CB", "CG1", "CG2", "CD1", "", "", "", "", "", ""),
        ("N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "", "", "", "", "", ""),
        ("N", "CA", "C", "O", "CB", "CG", "CD", "CE", "NZ", "", "", "", "", ""),
        ("N", "CA", "C", "O", "CB", "CG", "SD", "CE", "", "", "", "", "", ""),
        ("N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ", "", "", ""),
        ("N", "CA", "C", "O", "CB", "CG", "CD", "", "", "", "", "", "", ""),
        ("N", "CA", "C", "O", "CB", "OG", "", "", "", "", "", "", "", ""),
        ("N", "CA", "C", "O", "CB", "OG1", "CG2", "", "", "", "", "", "", ""),
        ("N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "NE1", "CE2", "CE3", "CZ2", "CZ3", "CH2"),
        ("N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ", "OH", "", ""),
        ("N", "CA", "C", "O", "CB", "CG1", "CG2", "", "", "", "", "", "", ""),
    )
)
RESTYPE_3 = (
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
)

MODEL_INPUT_WHITELIST = frozenset(
    {
        "aatype",
        "restype",
        "msa",
        "msa_mask",
        "has_deletion",
        "deletion_value",
        "profile",
        "deletion_mean",
        "residue_index",
        "token_mask",
    }
)

def _seed64(*parts: object) -> int:
    digest = hashlib.sha256(
        "\x1f".join(str(part) for part in parts).encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "little", signed=False)


def _validate_reference_table(table: tuple[dict[str, Any], ...]) -> None:
    if len(table) != 20:
        raise ValueError(f"CCD reference table must have 20 residues, got {len(table)}")
    for restype, (resname, names) in enumerate(zip(RESTYPE_3, RESTYPE_ATOM14_NAMES)):
        row = table[restype]
        if row.get("resname") != resname:
            raise ValueError(f"CCD row {restype} is not {resname}")
        for key in ("positions", "masks", "charges", "atomic_numbers"):
            if len(row.get(key, [])) != 14:
                raise ValueError(f"{resname} CCD row has invalid {key}")
        for slot, name in enumerate(names):
            if name and not row["masks"][slot]:
                raise ValueError(f"{resname}/{name} missing from pinned CCD reference")
        atom_count = sum(bool(name) for name in names)
        atom_permutations = row.get("atom_permutations")
        if not isinstance(atom_permutations, list) or len(atom_permutations) != atom_count:
            raise ValueError(
                f"{resname} CCD row has invalid atom_permutations row count"
            )
        widths = {len(atom_row) for atom_row in atom_permutations}
        if len(widths) != 1 or not widths or next(iter(widths)) < 1:
            raise ValueError(
                f"{resname} CCD atom_permutations must be a non-empty matrix"
            )
        permutation_columns = list(zip(*atom_permutations))
        identity = tuple(range(atom_count))
        if identity not in permutation_columns:
            raise ValueError(f"{resname} CCD atom_permutations lacks identity")
        for permutation in permutation_columns:
            if tuple(sorted(permutation)) != identity:
                raise ValueError(
                    f"{resname} CCD atom_permutations contains a non-permutation"
                )


def _compact_atom_permutations(
    info: dict[str, Any], names: tuple[str, ...], *, resname: str
) -> list[list[int]]:
    """Project pinned CCD graph automorphisms onto NanoFold atom14 atoms.

    Protenix expects a residue-local ``[N_atom, N_permutation]`` matrix. CCD
    automorphisms that move an atom14 atom to an omitted atom (notably the
    terminal-only OXT atom) are invalid for an internal protein residue and are
    dropped. Duplicate projected columns are removed while preserving CCD
    order, with identity made the first column.
    """

    compact_names = [name for name in names if name]
    full_indices = [int(info["atom_map"][name]) for name in compact_names]
    full_to_local = {
        full_index: local_index
        for local_index, full_index in enumerate(full_indices)
    }
    raw_permutations = info.get("perm")
    if raw_permutations is None:
        raise RuntimeError(f"pinned Protenix returned no CCD permutations for {resname}")
    if len(raw_permutations) <= max(full_indices):
        raise RuntimeError(f"pinned Protenix returned malformed CCD permutations for {resname}")
    column_count = len(raw_permutations[0])
    candidates: list[tuple[int, ...]] = []
    for column in range(column_count):
        projected: list[int] = []
        for full_index in full_indices:
            mapped_full_index = int(raw_permutations[full_index][column])
            if mapped_full_index not in full_to_local:
                projected = []
                break
            projected.append(full_to_local[mapped_full_index])
        candidate = tuple(projected)
        if (
            candidate
            and tuple(sorted(candidate)) == tuple(range(len(compact_names)))
            and candidate not in candidates
        ):
            candidates.append(candidate)
    identity = tuple(range(len(compact_names)))
    if identity not in candidates:
        raise RuntimeError(
            f"pinned CCD permutations for {resname} do not preserve atom14 identity"
        )
    candidates.remove(identity)
    candidates.insert(0, identity)
    return [
        [permutation[atom_index] for permutation in candidates]
        for atom_index in range(len(compact_names))
    ]


@lru_cache(maxsize=1)
def load_protenix_ccd_reference() -> tuple[dict[str, Any], ...]:
    """Load exact reference conformers from the pinned Protenix CCD cache.

    ``PROTENIX_CCD_REFERENCE_JSON`` may point to a versioned export produced by
    ``export_ccd_reference.py``. Otherwise the table is read directly through
    ``protenix.data.core.ccd.get_ccd_ref_info``.
    """

    exported = os.environ.get("PROTENIX_CCD_REFERENCE_JSON")
    if exported:
        import json

        payload = json.loads(Path(exported).read_text(encoding="utf-8"))
        if payload.get("schema_version") != 2:
            raise ValueError(
                "CCD export predates chemically equivalent atom permutations; "
                "regenerate it with export_ccd_reference.py"
            )
        if payload.get("protenix_commit") != (
            "c3bfc365b3e1341a11935eddfe7bfdc308092147"
        ):
            raise ValueError("CCD export was not generated from the pinned commit")
        table = tuple(payload["residues"])
        _validate_reference_table(table)
        return table

    try:
        from protenix.data.core.ccd import get_ccd_ref_info
    except ImportError as error:
        raise RuntimeError(
            "Pinned Protenix must be importable to construct exact CCD references"
        ) from error
    rows: list[dict[str, Any]] = []
    for resname, names in zip(RESTYPE_3, RESTYPE_ATOM14_NAMES):
        try:
            info = get_ccd_ref_info(
                resname, return_perm=True, return_atomic_number=True
            )
        except FileNotFoundError as error:
            raise RuntimeError(
                "Protenix CCD cache is missing. Set PROTENIX_ROOT_DIR to a data "
                "root containing common/components.cif.rdkit_mol.pkl, or set "
                "PROTENIX_CCD_REFERENCE_JSON to a pinned export."
            ) from error
        if not info:
            raise RuntimeError(f"pinned Protenix returned no CCD reference for {resname}")
        positions = [[0.0, 0.0, 0.0] for _ in range(14)]
        masks = [False] * 14
        charges = [0] * 14
        atomic_numbers = [0] * 14
        for slot, name in enumerate(names):
            if not name:
                continue
            if name not in info["atom_map"]:
                raise RuntimeError(f"{resname}/{name} absent from pinned Protenix CCD")
            index = int(info["atom_map"][name])
            positions[slot] = [float(value) for value in info["coord"][index]]
            masks[slot] = bool(info["mask"][index])
            charges[slot] = int(info["charge"][index])
            atomic_numbers[slot] = int(info["atomic_number"][index])
        rows.append(
            {
                "resname": resname,
                "positions": positions,
                "masks": masks,
                "charges": charges,
                "atomic_numbers": atomic_numbers,
                "atom_permutations": _compact_atom_permutations(
                    info, names, resname=resname
                ),
            }
        )
    table = tuple(rows)
    _validate_reference_table(table)
    return table


def ccd_reference_sha256(table: tuple[dict[str, Any], ...]) -> str:
    import json

    return hashlib.sha256(
        json.dumps(table, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _augment_reference(
    positions: torch.Tensor, ref_space_uid: torch.Tensor, *, seed: int
) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed % (2**63 - 1))
    output = positions.clone()
    for uid in torch.unique(ref_space_uid).tolist():
        mask = ref_space_uid == int(uid)
        points = output[mask]
        points = points - points.mean(dim=0, keepdim=True)
        quaternion = torch.randn(4, generator=generator)
        quaternion = quaternion / quaternion.norm().clamp_min(1e-8)
        w, x, y, z = quaternion
        rotation = torch.stack(
            (
                1 - 2 * (y * y + z * z),
                2 * (x * y - z * w),
                2 * (x * z + y * w),
                2 * (y * z + x * w),
                1 - 2 * (x * x + z * z),
                2 * (y * z - x * w),
                2 * (x * z - y * w),
                2 * (y * z + x * w),
                1 - 2 * (x * x + y * y),
            )
        ).reshape(3, 3)
        translation = 2.0 * torch.rand(3, generator=generator) - 1.0
        output[mask] = (points + translation) @ rotation.T
    return output


def _atom_name_chars(names: list[str]) -> torch.Tensor:
    encoded = torch.zeros((len(names), 4, 64), dtype=torch.long)
    for atom_index, name in enumerate(names):
        for char_index, character in enumerate(name.ljust(4)[:4]):
            encoded[atom_index, char_index, max(0, min(63, ord(character) - 32))] = 1
    return encoded


def _nanofold_to_protenix(
    example: dict[str, Any],
    *,
    seed: int,
    step: int,
    rank: int,
    augment_reference: bool,
    ccd_reference: tuple[dict[str, Any], ...] | None = None,
    include_labels: bool,
) -> dict[str, Any]:
    """Build one unbatched Protenix item, optionally attaching supervision."""

    missing = MODEL_INPUT_WHITELIST.difference(example)
    if missing:
        raise KeyError(f"NanoFold item missing model-input keys: {sorted(missing)}")
    if include_labels:
        for required_target in (
            "atom14_positions",
            "atom14_positions_uncentered",
            "atom14_mask",
        ):
            if required_target not in example:
                raise KeyError(
                    f"NanoFold item missing required target key: {required_target}"
                )

    length = int(example["crop_length"])
    if not 1 <= length <= int(example["token_mask"].shape[0]):
        raise ValueError(f"invalid crop length: {length}")
    aatype = example["aatype"][:length].long()
    if torch.any((aatype < 0) | (aatype >= 20)):
        bad = torch.unique(aatype[(aatype < 0) | (aatype >= 20)]).tolist()
        raise ValueError(f"Protenix adapter currently supports 20 canonical amino acids; got {bad}")
    reference_table = (
        load_protenix_ccd_reference() if ccd_reference is None else ccd_reference
    )
    _validate_reference_table(reference_table)

    atom_to_token: list[int] = []
    atom_to_slot: list[int] = []
    atom_names: list[str] = []
    reference_positions: list[tuple[float, float, float]] = []
    frame_atom_index = torch.empty((length, 3), dtype=torch.long)
    distogram_rep: list[int] = []
    pae_rep: list[int] = []
    plddt_rep: list[int] = []
    ref_charge: list[int] = []
    atom_perm_list: list[list[int]] = []

    compact_index = 0
    for residue_index, restype_tensor in enumerate(aatype):
        restype = int(restype_tensor.item())
        names = RESTYPE_ATOM14_NAMES[restype]
        reference_row = reference_table[restype]
        per_residue: dict[str, int] = {}
        for slot, atom_name in enumerate(names):
            if not atom_name:
                continue
            per_residue[atom_name] = compact_index
            atom_to_token.append(residue_index)
            atom_to_slot.append(slot)
            atom_names.append(atom_name)
            reference_positions.append(tuple(reference_row["positions"][slot]))
            atom_perm_list.append(
                list(reference_row["atom_permutations"][len(per_residue) - 1])
            )
            ref_charge.append(int(reference_row["charges"][slot]))
            is_ca = atom_name == "CA"
            pae_rep.append(int(is_ca))
            plddt_rep.append(int(is_ca))
            distogram_rep.append(
                int(atom_name == ("CA" if restype == 7 else "CB"))
            )
            compact_index += 1
        frame_atom_index[residue_index] = torch.tensor(
            [per_residue["N"], per_residue["CA"], per_residue["C"]]
        )

    atom_to_token_tensor = torch.tensor(atom_to_token, dtype=torch.long)
    atom_to_slot_tensor = torch.tensor(atom_to_slot, dtype=torch.long)
    ref_space_uid = atom_to_token_tensor.clone()
    ref_pos = torch.tensor(reference_positions, dtype=torch.float32)
    if augment_reference:
        ref_pos = _augment_reference(
            ref_pos,
            ref_space_uid,
            seed=_seed64("protenix-ref", seed, step, rank, example["chain_id"]),
        )

    ref_element = torch.zeros((len(atom_names), 128), dtype=torch.long)
    compact_index = 0
    ref_mask: list[int] = []
    for restype_tensor in aatype:
        restype = int(restype_tensor.item())
        row = reference_table[restype]
        for slot, name in enumerate(RESTYPE_ATOM14_NAMES[restype]):
            if not name:
                continue
            atomic_number = int(row["atomic_numbers"][slot])
            if not 1 <= atomic_number <= 118:
                raise ValueError(f"invalid CCD atomic number for {RESTYPE_3[restype]}/{name}")
            ref_element[compact_index, atomic_number - 1] = 1
            ref_mask.append(int(bool(row["masks"][slot])))
            compact_index += 1

    valid_msa_rows = example["msa_mask"][:, :length].any(dim=-1)
    msa = example["msa"][valid_msa_rows, :length].long().clone()
    # Accept either the raw NanoFold/AF2 tokenization (gap=21) or the shared
    # common loader's Protenix-ready tokenization (gap=31).
    if torch.any((msa < 0) | ((msa > 20) & (msa != 21) & (msa != 31))):
        bad = torch.unique(
            msa[(msa < 0) | ((msa > 20) & (msa != 21) & (msa != 31))]
        ).tolist()
        raise ValueError(f"unexpected NanoFold MSA tokens: {bad}")
    msa[msa == 21] = 31
    has_deletion = example["has_deletion"][valid_msa_rows, :length].float()
    deletion_value = example["deletion_value"][valid_msa_rows, :length].float()
    if msa.shape[0] < 1:
        raise ValueError("MSA must contain its query row")

    restype = F.one_hot(aatype, num_classes=32).float()
    zeros_atom = torch.zeros(len(atom_names), dtype=torch.long)
    zeros_token = torch.zeros(length, dtype=torch.long)
    ca_mask = torch.tensor(pae_rep, dtype=torch.long)

    profile_source = example["profile"][:length].float()
    if profile_source.shape[-1] == 22:
        profile = profile_source.new_zeros((length, 32))
        profile[:, :21] = profile_source[:, :21]
        profile[:, 31] = profile_source[:, 21]
    elif profile_source.shape[-1] == 32:
        profile = profile_source
    else:
        raise ValueError(
            f"NanoFold profile must have 22 or 32 channels, got {profile_source.shape[-1]}"
        )
    input_feature_dict: dict[str, Any] = {
        "residue_index": example["residue_index"][:length].long(),
        "token_index": torch.arange(length, dtype=torch.long),
        "asym_id": zeros_token.clone(),
        "entity_id": zeros_token.clone(),
        "sym_id": zeros_token.clone(),
        "restype": restype,
        "entity_mol_id": zeros_atom.clone(),
        "mol_id": zeros_atom.clone(),
        "mol_atom_index": torch.arange(len(atom_names), dtype=torch.long),
        "ref_pos": ref_pos,
        "ref_mask": torch.tensor(ref_mask, dtype=torch.long),
        "ref_element": ref_element,
        "ref_charge": torch.tensor(ref_charge, dtype=torch.long),
        "ref_atom_name_chars": _atom_name_chars(atom_names),
        "ref_space_uid": ref_space_uid,
        "msa": msa,
        "has_deletion": has_deletion,
        "deletion_value": deletion_value,
        "profile": profile,
        "deletion_mean": example["deletion_mean"][:length].float(),
        "token_bonds": torch.zeros((length, length), dtype=torch.float32),
        "is_protein": torch.ones(len(atom_names), dtype=torch.long),
        "is_rna": zeros_atom.clone(),
        "is_dna": zeros_atom.clone(),
        "is_ligand": zeros_atom.clone(),
        "distogram_rep_atom_mask": torch.tensor(distogram_rep, dtype=torch.long),
        "pae_rep_atom_mask": ca_mask,
        "plddt_m_rep_atom_mask": torch.tensor(plddt_rep, dtype=torch.long),
        "modified_res_mask": zeros_atom.clone(),
        "bond_mask": torch.zeros((len(atom_names), len(atom_names)), dtype=torch.long),
        "resolution": torch.tensor([-1.0]),
        "atom_to_token_idx": atom_to_token_tensor,
        "atom_to_tokatom_idx": atom_to_slot_tensor,
        "is_distillation": torch.tensor([False]),
        "has_frame": torch.ones(length, dtype=torch.long),
        "frame_atom_index": frame_atom_index,
        # Protenix requires residue-local indices, not global compact indices.
        # Each row lists the image of that atom under every valid residue
        # automorphism projected from the exact pinned CCD.
        "atom_perm_list": atom_perm_list,
        "prot_pair_num_alignments": torch.tensor(0, dtype=torch.int32),
        "prot_unpair_num_alignments": torch.tensor(msa.shape[0], dtype=torch.int32),
        "rna_pair_num_alignments": torch.tensor(0, dtype=torch.int32),
        "rna_unpair_num_alignments": torch.tensor(0, dtype=torch.int32),
    }
    item = {
        "chain_id": example["chain_id"],
        "input_feature_dict": input_feature_dict,
        "atom_to_token_idx": atom_to_token_tensor,
        "atom_to_atom14_slot": atom_to_slot_tensor,
        "crop_length": length,
    }
    if include_labels:
        target = example["atom14_positions_uncentered"][:length].float()
        target_mask = example["atom14_mask"][:length].bool()
        coordinate = target[atom_to_token_tensor, atom_to_slot_tensor]
        coordinate_mask = target_mask[
            atom_to_token_tensor, atom_to_slot_tensor
        ].long()
        label_dict = {
            "coordinate": coordinate,
            "coordinate_mask": coordinate_mask,
        }
        item["label_dict"] = label_dict
        item["label_full_dict"] = {
            **label_dict,
            "entity_mol_id": zeros_atom.clone(),
            "mol_id": zeros_atom.clone(),
            "mol_atom_index": torch.arange(len(atom_names), dtype=torch.long),
            "pae_rep_atom_mask": ca_mask.clone(),
        }
    return item


def nanofold_to_protenix(
    example: dict[str, Any],
    *,
    seed: int,
    step: int,
    rank: int,
    augment_reference: bool,
    ccd_reference: tuple[dict[str, Any], ...] | None = None,
) -> dict[str, Any]:
    """Build one Protenix training item with labels kept out of model inputs."""

    return _nanofold_to_protenix(
        example,
        seed=seed,
        step=step,
        rank=rank,
        augment_reference=augment_reference,
        ccd_reference=ccd_reference,
        include_labels=True,
    )


def nanofold_inputs_to_protenix(
    example: dict[str, Any],
    *,
    seed: int,
    step: int,
    rank: int,
    augment_reference: bool,
    ccd_reference: tuple[dict[str, Any], ...] | None = None,
) -> dict[str, Any]:
    """Build a genuinely label-free Protenix inference item."""

    return _nanofold_to_protenix(
        example,
        seed=seed,
        step=step,
        rank=rank,
        augment_reference=augment_reference,
        ccd_reference=ccd_reference,
        include_labels=False,
    )


def compact_to_atom14(
    compact_coordinate: torch.Tensor,
    atom_to_token_idx: torch.Tensor,
    atom_to_atom14_slot: torch.Tensor,
    *,
    length: int,
) -> torch.Tensor:
    """Scatter compact Protenix coordinates into NanoFold's atom14 layout."""

    output = compact_coordinate.new_zeros(
        (*compact_coordinate.shape[:-2], length, 14, 3)
    )
    output[..., atom_to_token_idx, atom_to_atom14_slot, :] = compact_coordinate
    return output


def prediction_to_nanofold(
    prediction: dict[str, torch.Tensor],
    adapted_item: dict[str, Any],
    *,
    sample_index: int = 0,
) -> dict[str, torch.Tensor]:
    """Tested boundary from a Protenix prediction to NanoFold evaluator tensors.

    Protenix inference normally returns ``coordinate`` as
    ``[N_sample, N_atom, 3]``. A direct ``[N_atom, 3]`` tensor is also accepted
    for unit tests and one-sample callers.
    """

    coordinate = prediction["coordinate"]
    if coordinate.ndim == 3:
        if not 0 <= sample_index < coordinate.shape[0]:
            raise IndexError("sample_index outside Protenix diffusion samples")
        coordinate = coordinate[sample_index]
    if coordinate.ndim != 2 or coordinate.shape[-1] != 3:
        raise ValueError(
            f"expected [N_atom,3] or [N_sample,N_atom,3], got {tuple(coordinate.shape)}"
        )
    pred_atom14 = compact_to_atom14(
        coordinate,
        adapted_item["atom_to_token_idx"].to(coordinate.device),
        adapted_item["atom_to_atom14_slot"].to(coordinate.device),
        length=int(adapted_item["crop_length"]),
    )
    return {
        "pred_atom14": pred_atom14,
        "pred_ca": pred_atom14[:, 1],
    }

