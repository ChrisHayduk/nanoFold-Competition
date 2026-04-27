from __future__ import annotations

from typing import Dict, Iterable

import torch

from .residue_constants import (
    ATOM14_INDEX,
    ATOM14_NAMES_BY_RESTYPE_INDEX,
    ATOM14_NUM_SLOTS,
    CA_ATOM14_SLOT,
    CHI1_CHI2_ATOM_NAMES,
    CHI_PI_PERIODIC,
    RESTYPE_INDEX_TO_3,
)

FOLDSCORE_GDT_HA_CA_WEIGHT = 0.25
FOLDSCORE_LDDT_ATOM14_WEIGHT = 0.09375
FOLDSCORE_CAD_ATOM14_WEIGHT = 0.09375
FOLDSCORE_SG_ATOM14_WEIGHT = 0.09375
FOLDSCORE_SC_ATOM14_WEIGHT = 0.09375
FOLDSCORE_MOLPROBITY_CLASH_ATOM14_WEIGHT = 0.125
FOLDSCORE_BB_ATOM14_WEIGHT = 0.125
FOLDSCORE_DIPDIFF_ATOM14_WEIGHT = 0.125
BACKBONE_ATOM14_SLOTS = (0, 1, 2, 3)
SIDECHAIN_ATOM14_SLOTS = tuple(range(4, ATOM14_NUM_SLOTS))
MOLPROBITY_HEAVY_ATOM_VDW_RADII = {
    "C": 1.70,
    "N": 1.55,
    "O": 1.52,
    "S": 1.80,
}
SC_CHI_WEIGHTS = (2.0, 1.0)
FOLDSCORE_WEIGHT_BY_COMPONENT = {
    "gdt_ha_ca": FOLDSCORE_GDT_HA_CA_WEIGHT,
    "lddt_atom14": FOLDSCORE_LDDT_ATOM14_WEIGHT,
    "cad_atom14": FOLDSCORE_CAD_ATOM14_WEIGHT,
    "sg_atom14": FOLDSCORE_SG_ATOM14_WEIGHT,
    "sc_atom14": FOLDSCORE_SC_ATOM14_WEIGHT,
    "molprobity_clash_atom14": FOLDSCORE_MOLPROBITY_CLASH_ATOM14_WEIGHT,
    "bb_atom14": FOLDSCORE_BB_ATOM14_WEIGHT,
    "dipdiff_atom14": FOLDSCORE_DIPDIFF_ATOM14_WEIGHT,
}
FOLDSCORE_RANK_COMPONENT_NAMES = tuple(FOLDSCORE_WEIGHT_BY_COMPONENT)
FOLDSCORE_COMPONENT_NAMES = (
    "foldscore",
    *FOLDSCORE_RANK_COMPONENT_NAMES,
    "gdt_ts_ca",
    "lddt_ca",
    "lddt_backbone_atom14",
)
FOLDSCORE_DIAGNOSTIC_COMPONENT_NAMES = (
    "gdt_ts_ca",
    "lddt_ca",
    "lddt_backbone_atom14",
)
FOLDSCORE_CURVE_COMPONENT_NAMES = (
    *FOLDSCORE_RANK_COMPONENT_NAMES,
    *FOLDSCORE_DIAGNOSTIC_COMPONENT_NAMES,
)


@torch.no_grad()
def lddt_ca(
    pred_ca: torch.Tensor,
    true_ca: torch.Tensor,
    ca_mask: torch.Tensor,
    cutoff: float = 15.0,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Compute mean lDDT-Cα for a single example.

    Inputs:
      pred_ca: (L,3)
      true_ca: (L,3)
      ca_mask: (L,) bool

    Returns:
      scalar tensor in [0,1]
    """
    assert pred_ca.shape == true_ca.shape
    L = pred_ca.shape[0]
    device = pred_ca.device

    mask = ca_mask.to(device=device, dtype=torch.bool)
    if mask.sum() < 2:
        return torch.zeros((), device=device)

    # Pairwise distances (L,L)
    d_true = torch.cdist(true_ca, true_ca, p=2)
    d_pred = torch.cdist(pred_ca, pred_ca, p=2)

    # Consider pairs within cutoff in the true structure, excluding i==j.
    pair_mask = (d_true < cutoff) & mask[:, None] & mask[None, :] & (~torch.eye(L, dtype=torch.bool, device=device))

    # If a residue has zero neighbors, we exclude it from the per-residue average.
    per_res_counts = pair_mask.sum(dim=-1)  # (L,)
    valid_res = per_res_counts > 0

    if valid_res.sum() == 0:
        return torch.zeros((), device=device)

    diff = torch.abs(d_true - d_pred)

    thresholds = torch.tensor([0.5, 1.0, 2.0, 4.0], device=device)
    # (4, L, L)
    within = (diff[None, :, :] < thresholds[:, None, None]) & pair_mask[None, :, :]
    # fraction over neighbors per residue for each threshold
    frac = within.sum(dim=-1) / (per_res_counts[None, :] + eps)  # (4,L)
    per_res_lddt = frac.mean(dim=0)  # (L,)

    return per_res_lddt[valid_res].mean()


def _kabsch_align(
    pred_points: torch.Tensor,
    true_points: torch.Tensor,
    point_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    mask = point_mask.to(device=pred_points.device, dtype=torch.bool)
    if mask.sum() < 3:
        return pred_points, mask

    pred = pred_points.to(dtype=true_points.dtype)
    true = true_points.to(device=pred.device, dtype=pred.dtype)
    pred_sel = pred[mask]
    true_sel = true[mask]
    pred_centroid = pred_sel.mean(dim=0, keepdim=True)
    true_centroid = true_sel.mean(dim=0, keepdim=True)
    pred_centered = pred_sel - pred_centroid
    true_centered = true_sel - true_centroid

    covariance = pred_centered.transpose(0, 1) @ true_centered
    u, _s, vh = torch.linalg.svd(covariance)
    rotation = u @ vh
    if torch.det(rotation) < 0:
        u = u.clone()
        u[:, -1] *= -1
        rotation = u @ vh

    aligned = (pred - pred_centroid) @ rotation + true_centroid
    return aligned, mask


def _as_aatype_tensor(aatype: torch.Tensor, *, length: int, device: torch.device) -> torch.Tensor:
    if aatype.ndim != 1:
        raise ValueError(f"`aatype` must have shape (L,), got {tuple(aatype.shape)}")
    if int(aatype.shape[0]) < length:
        raise ValueError(f"`aatype` length {int(aatype.shape[0])} is shorter than scored length {length}")
    return aatype[:length].to(device=device, dtype=torch.long).clamp(min=0, max=len(RESTYPE_INDEX_TO_3) - 1)


def _residue_name_from_aatype(residue_type: int) -> str:
    if residue_type < 0 or residue_type >= len(RESTYPE_INDEX_TO_3):
        return "UNK"
    return RESTYPE_INDEX_TO_3[residue_type]


def _atom_name_to_element(atom_name: str) -> str:
    if not atom_name:
        return ""
    if atom_name[0].isdigit() and len(atom_name) > 1:
        return atom_name[1].upper()
    return atom_name[0].upper()


def _angular_error(pred_angle: torch.Tensor, true_angle: torch.Tensor, *, pi_periodic: bool = False) -> torch.Tensor:
    delta = torch.atan2(torch.sin(pred_angle - true_angle), torch.cos(pred_angle - true_angle)).abs()
    if pi_periodic:
        delta = torch.minimum(delta, torch.pi - delta)
    return delta / torch.pi


def _rmsd(points_a: torch.Tensor, points_b: torch.Tensor) -> torch.Tensor:
    if points_a.numel() == 0:
        return torch.zeros((), device=points_a.device, dtype=points_a.dtype)
    return torch.sqrt(torch.mean(torch.sum((points_a - points_b) ** 2, dim=-1)))


@torch.no_grad()
def _gdt_seed_masks(valid_mask: torch.Tensor, *, max_seeds_per_window: int = 48) -> list[torch.Tensor]:
    valid_indices = torch.nonzero(valid_mask, as_tuple=False).flatten()
    n_valid = int(valid_indices.numel())
    seeds: list[torch.Tensor] = [valid_mask]
    for window_len in (3, 5, 9):
        if n_valid < window_len:
            continue
        max_start = n_valid - window_len
        stride = max(1, max_start // max_seeds_per_window) if max_start else 1
        for start in range(0, max_start + 1, stride):
            seed = torch.zeros_like(valid_mask)
            seed[valid_indices[start : start + window_len]] = True
            seeds.append(seed)
    return seeds


@torch.no_grad()
def _gdt_fraction_at_threshold(
    pred_ca: torch.Tensor,
    true_ca: torch.Tensor,
    valid_mask: torch.Tensor,
    threshold: float,
) -> torch.Tensor:
    device = pred_ca.device
    n_valid = valid_mask.sum().to(dtype=pred_ca.dtype).clamp_min(1.0)
    if int(valid_mask.sum().item()) < 3:
        return torch.zeros((), device=device, dtype=pred_ca.dtype)

    best_fraction = torch.zeros((), device=device, dtype=pred_ca.dtype)
    best_count = -1
    for seed in _gdt_seed_masks(valid_mask):
        if int(seed.sum().item()) < 3:
            continue
        active = seed
        aligned = pred_ca
        for _ in range(8):
            aligned, _ = _kabsch_align(pred_ca, true_ca, active)
            distances = torch.linalg.vector_norm(aligned - true_ca, dim=-1)
            refined = valid_mask & (distances <= threshold)
            refined_count = int(refined.sum().item())
            if refined_count < 3 or torch.equal(refined, active):
                break
            active = refined
        distances = torch.linalg.vector_norm(aligned - true_ca, dim=-1)
        current_count = int(((distances <= threshold) & valid_mask).sum().item())
        if current_count > best_count:
            best_count = current_count
            best_fraction = torch.as_tensor(current_count, device=device, dtype=pred_ca.dtype) / n_valid
            if current_count == int(valid_mask.sum().item()):
                break
    return best_fraction


@torch.no_grad()
def gdt_ca(
    pred_ca: torch.Tensor,
    true_ca: torch.Tensor,
    ca_mask: torch.Tensor,
    *,
    thresholds: Iterable[float],
) -> torch.Tensor:
    """Compute C-alpha Global Distance Test with threshold-specific superpositions."""
    assert pred_ca.shape == true_ca.shape
    device = pred_ca.device
    true = true_ca.to(device=device, dtype=pred_ca.dtype)
    mask = ca_mask.to(device=device, dtype=torch.bool)
    if mask.sum() < 3:
        return torch.zeros((), device=device, dtype=pred_ca.dtype)
    threshold_values = tuple(thresholds)
    if not threshold_values:
        raise ValueError("GDT requires at least one distance threshold.")
    scores = [
        _gdt_fraction_at_threshold(pred_ca, true, mask, float(threshold))
        for threshold in threshold_values
    ]
    return torch.stack(scores).mean()


@torch.no_grad()
def gdt_ha_ca(pred_ca: torch.Tensor, true_ca: torch.Tensor, ca_mask: torch.Tensor) -> torch.Tensor:
    """Compute high-accuracy C-alpha GDT."""
    return gdt_ca(pred_ca, true_ca, ca_mask, thresholds=(0.5, 1.0, 2.0, 4.0))


@torch.no_grad()
def gdt_ts_ca(pred_ca: torch.Tensor, true_ca: torch.Tensor, ca_mask: torch.Tensor) -> torch.Tensor:
    """Compute C-alpha GDT_TS using 1/2/4/8 A cutoffs."""
    return gdt_ca(pred_ca, true_ca, ca_mask, thresholds=(1.0, 2.0, 4.0, 8.0))


@torch.no_grad()
def lddt_atom_points(
    pred_points: torch.Tensor,
    true_points: torch.Tensor,
    point_mask: torch.Tensor,
    residue_ids: torch.Tensor | None = None,
    cutoff: float = 15.0,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Compute mean lDDT over an arbitrary masked point cloud.

    Inputs:
      pred_points: (P,3)
      true_points: (P,3)
      point_mask: (P,) bool

    Returns:
      scalar tensor in [0,1]
    """
    assert pred_points.shape == true_points.shape
    P = pred_points.shape[0]
    device = pred_points.device

    mask = point_mask.to(device=device, dtype=torch.bool)
    if mask.sum() < 2:
        return torch.zeros((), device=device)

    d_true = torch.cdist(true_points, true_points, p=2)
    d_pred = torch.cdist(pred_points, pred_points, p=2)

    eye = torch.eye(P, dtype=torch.bool, device=device)
    pair_mask = (d_true < cutoff) & mask[:, None] & mask[None, :] & (~eye)
    if residue_ids is not None:
        residue_ids = residue_ids.to(device=device, dtype=torch.long)
        if residue_ids.ndim != 1 or int(residue_ids.shape[0]) != P:
            raise ValueError(f"`residue_ids` must have shape ({P},), got {tuple(residue_ids.shape)}")
        pair_mask &= residue_ids[:, None] != residue_ids[None, :]
    per_point_counts = pair_mask.sum(dim=-1)
    valid_points = per_point_counts > 0
    if valid_points.sum() == 0:
        return torch.zeros((), device=device)

    diff = torch.abs(d_true - d_pred)
    thresholds = torch.tensor([0.5, 1.0, 2.0, 4.0], device=device, dtype=diff.dtype)
    within = (diff[None, :, :] < thresholds[:, None, None]) & pair_mask[None, :, :]
    frac = within.sum(dim=-1) / (per_point_counts[None, :] + eps)
    per_point_lddt = frac.mean(dim=0)
    return per_point_lddt[valid_points].mean()


def _flatten_atom14(
    atom14_positions: torch.Tensor,
    atom14_mask: torch.Tensor,
    *,
    slots: Iterable[int] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if atom14_positions.ndim != 3 or atom14_positions.shape[1:] != (ATOM14_NUM_SLOTS, 3):
        raise ValueError(f"`atom14_positions` must have shape (L, 14, 3), got {tuple(atom14_positions.shape)}")
    if atom14_mask.ndim != 2 or atom14_mask.shape[1] != ATOM14_NUM_SLOTS:
        raise ValueError(f"`atom14_mask` must have shape (L, 14), got {tuple(atom14_mask.shape)}")
    if atom14_positions.shape[:2] != atom14_mask.shape:
        raise ValueError("`atom14_positions` and `atom14_mask` length/slot dimensions must match.")

    if slots is not None:
        slot_idx = torch.tensor(tuple(slots), dtype=torch.long, device=atom14_positions.device)
        atom14_positions = atom14_positions.index_select(dim=1, index=slot_idx)
        atom14_mask = atom14_mask.to(device=atom14_positions.device).index_select(dim=1, index=slot_idx)
    else:
        atom14_mask = atom14_mask.to(device=atom14_positions.device)

    return atom14_positions.reshape(-1, 3), atom14_mask.reshape(-1)


def _flatten_atom14_with_residue_ids(
    atom14_positions: torch.Tensor,
    atom14_mask: torch.Tensor,
    *,
    slots: Iterable[int] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    L = int(atom14_positions.shape[0])
    if slots is None:
        selected_slots = tuple(range(ATOM14_NUM_SLOTS))
    else:
        selected_slots = tuple(slots)
    points, point_mask = _flatten_atom14(atom14_positions, atom14_mask, slots=selected_slots)
    residue_ids = torch.arange(L, device=atom14_positions.device).repeat_interleave(len(selected_slots))
    return points, point_mask, residue_ids


@torch.no_grad()
def cad_atom14_score(
    pred_atom14: torch.Tensor,
    true_atom14: torch.Tensor,
    atom14_mask: torch.Tensor,
    contact_cutoff: float = 5.0,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Compute atom14 contact-preservation agreement from heavy-atom contacts."""
    device = pred_atom14.device
    L = int(pred_atom14.shape[0])
    if L < 2:
        return torch.ones((), device=device)

    true = true_atom14.to(device=device, dtype=pred_atom14.dtype)
    mask = atom14_mask.to(device=device, dtype=torch.bool)
    valid_pairs = mask[:, None, :, None] & mask[None, :, None, :]
    pair_denominator = valid_pairs.sum(dim=(-1, -2)).to(dtype=pred_atom14.dtype)
    residue_pair_mask = pair_denominator > 0
    residue_pair_mask &= ~torch.eye(L, dtype=torch.bool, device=device)
    if not bool(residue_pair_mask.any()):
        return torch.ones((), device=device)

    pred_dist = torch.linalg.vector_norm(
        pred_atom14[:, None, :, None, :] - pred_atom14[None, :, None, :, :],
        dim=-1,
    )
    true_dist = torch.linalg.vector_norm(
        true[:, None, :, None, :] - true[None, :, None, :, :],
        dim=-1,
    )
    pred_contact = ((pred_dist < contact_cutoff) & valid_pairs).sum(dim=(-1, -2)).to(dtype=pred_atom14.dtype)
    true_contact = ((true_dist < contact_cutoff) & valid_pairs).sum(dim=(-1, -2)).to(dtype=pred_atom14.dtype)
    pred_contact = pred_contact / pair_denominator.clamp_min(1.0)
    true_contact = true_contact / pair_denominator.clamp_min(1.0)

    numerator = torch.abs(pred_contact - true_contact)[residue_pair_mask].sum()
    denominator = torch.maximum(pred_contact, true_contact)[residue_pair_mask].sum()
    if float(denominator.detach().cpu()) <= eps:
        return torch.ones((), device=device)
    return torch.clamp(1.0 - numerator / denominator.clamp_min(eps), min=0.0, max=1.0)


@torch.no_grad()
def spheregrinder_atom14_score(
    pred_atom14: torch.Tensor,
    true_atom14: torch.Tensor,
    atom14_mask: torch.Tensor,
    sphere_radius: float = 6.0,
) -> torch.Tensor:
    """Compute a SphereGrinder-style local superposition score.

    For each resolved C-alpha, the target atoms inside a 6 A sphere define a
    local motif. The corresponding predicted atoms are optimally superposed to
    the target motif, a motif RMSD is computed, and the score is the average of
    the fractions of scorable residues with RMSD <= 2 A and <= 4 A.
    """
    device = pred_atom14.device
    pred_points, point_mask = _flatten_atom14(pred_atom14, atom14_mask)
    true_points, _ = _flatten_atom14(true_atom14.to(device=device, dtype=pred_atom14.dtype), atom14_mask)
    ca_mask = atom14_mask[:, CA_ATOM14_SLOT].to(device=device, dtype=torch.bool)
    if not bool(ca_mask.any()):
        return torch.ones((), device=device)

    local_rmsds = []
    true_ca = true_atom14[:, CA_ATOM14_SLOT, :].to(device=device, dtype=pred_atom14.dtype)
    for idx in torch.nonzero(ca_mask, as_tuple=False).flatten().tolist():
        sphere_mask = (torch.linalg.vector_norm(true_points - true_ca[idx], dim=-1) <= sphere_radius) & point_mask
        if int(sphere_mask.sum().item()) < 3:
            continue
        motif_mask = torch.ones((int(sphere_mask.sum().item()),), dtype=torch.bool, device=device)
        aligned, _ = _kabsch_align(pred_points[sphere_mask], true_points[sphere_mask], motif_mask)
        local_rmsds.append(_rmsd(aligned, true_points[sphere_mask]))
    if not local_rmsds:
        return torch.ones((), device=device)
    rmsd_tensor = torch.stack(local_rmsds)
    cutoff_scores = [
        (rmsd_tensor <= threshold).to(dtype=pred_atom14.dtype).mean()
        for threshold in (2.0, 4.0)
    ]
    return torch.stack(cutoff_scores).mean()


@torch.no_grad()
def _sidechain_burial_weights(
    true_atom14: torch.Tensor,
    atom14_mask: torch.Tensor,
    aatype: torch.Tensor,
    *,
    neighbor_cutoff: float = 10.0,
) -> torch.Tensor:
    device = true_atom14.device
    L = int(true_atom14.shape[0])
    representative = torch.zeros((L, 3), dtype=true_atom14.dtype, device=device)
    representative_mask = torch.zeros((L,), dtype=torch.bool, device=device)
    for idx, residue_type in enumerate(aatype.tolist()):
        residue_name = _residue_name_from_aatype(int(residue_type))
        cb_slot = ATOM14_INDEX.get(residue_name, {}).get("CB", CA_ATOM14_SLOT)
        if bool(atom14_mask[idx, cb_slot].item()):
            representative[idx] = true_atom14[idx, cb_slot]
            representative_mask[idx] = True
        elif bool(atom14_mask[idx, CA_ATOM14_SLOT].item()):
            representative[idx] = true_atom14[idx, CA_ATOM14_SLOT]
            representative_mask[idx] = True
    if int(representative_mask.sum().item()) < 2:
        return torch.ones((L,), dtype=true_atom14.dtype, device=device)
    distances = torch.cdist(representative, representative, p=2)
    eye = torch.eye(L, dtype=torch.bool, device=device)
    neighbors = (distances <= neighbor_cutoff) & representative_mask[:, None] & representative_mask[None, :] & (~eye)
    counts = neighbors.sum(dim=-1).to(dtype=true_atom14.dtype)
    burial = torch.clamp(counts / 20.0, min=0.0, max=1.0)
    return 0.25 + 0.75 * burial


def _chi_atom_slots(residue_name: str, chi_index: int) -> tuple[int, int, int, int] | None:
    chis = CHI1_CHI2_ATOM_NAMES.get(residue_name, ())
    if chi_index >= len(chis):
        return None
    slots = []
    atom_index = ATOM14_INDEX.get(residue_name, {})
    for atom_name in chis[chi_index]:
        slot = atom_index.get(atom_name)
        if slot is None:
            return None
        slots.append(slot)
    return (slots[0], slots[1], slots[2], slots[3])


@torch.no_grad()
def sidechain_atom14_score(
    pred_atom14: torch.Tensor,
    true_atom14: torch.Tensor,
    atom14_mask: torch.Tensor,
    aatype: torch.Tensor,
) -> torch.Tensor:
    """Compute a high-is-better chi1/chi2 side-chain dihedral score."""
    device = pred_atom14.device
    true = true_atom14.to(device=device, dtype=pred_atom14.dtype)
    mask = atom14_mask.to(device=device, dtype=torch.bool)
    aatype = _as_aatype_tensor(aatype, length=int(pred_atom14.shape[0]), device=device)
    burial_weights = _sidechain_burial_weights(true, mask, aatype)
    weighted_errors = []
    weights = []
    for residue_index, residue_type in enumerate(aatype.tolist()):
        residue_name = _residue_name_from_aatype(int(residue_type))
        for chi_index, chi_weight in enumerate(SC_CHI_WEIGHTS):
            slots = _chi_atom_slots(residue_name, chi_index)
            if slots is None:
                continue
            slot_tensor = torch.tensor(slots, dtype=torch.long, device=device)
            if not bool(mask[residue_index].index_select(0, slot_tensor).all().item()):
                continue
            pred_angle = _dihedral_angle(
                pred_atom14[residue_index, slots[0]],
                pred_atom14[residue_index, slots[1]],
                pred_atom14[residue_index, slots[2]],
                pred_atom14[residue_index, slots[3]],
            )
            true_angle = _dihedral_angle(
                true[residue_index, slots[0]],
                true[residue_index, slots[1]],
                true[residue_index, slots[2]],
                true[residue_index, slots[3]],
            )
            pi_periodic = CHI_PI_PERIODIC.get((residue_name, chi_index), False)
            weight = torch.as_tensor(chi_weight, device=device, dtype=pred_atom14.dtype) * burial_weights[residue_index]
            weighted_errors.append(_angular_error(pred_angle, true_angle, pi_periodic=pi_periodic) * weight)
            weights.append(weight)
    if not weighted_errors:
        return torch.ones((), device=device, dtype=pred_atom14.dtype)
    error = torch.stack(weighted_errors).sum() / torch.stack(weights).sum().clamp_min(1e-8)
    return torch.clamp(1.0 - error, min=0.0, max=1.0)


@torch.no_grad()
def molprobity_clash_atom14_score(
    pred_atom14: torch.Tensor,
    atom14_mask: torch.Tensor,
    aatype: torch.Tensor,
    overlap_cutoff: float = 0.4,
) -> torch.Tensor:
    """Compute an atom-name-aware heavy-atom clash score.

    This mirrors the MolProbity clash definition at the level available from
    atom14 outputs: a nonbonded heavy-atom pair clashes when van der Waals
    overlap exceeds 0.4 A. The result is mapped to a high-is-better [0, 1]
    component for FoldScore.
    """
    device = pred_atom14.device
    mask = atom14_mask.to(device=device, dtype=torch.bool)
    aatype = _as_aatype_tensor(aatype, length=int(pred_atom14.shape[0]), device=device)
    pred_points, point_mask = _flatten_atom14(pred_atom14, mask)
    if int(point_mask.sum().item()) < 2:
        return torch.ones((), device=device, dtype=pred_atom14.dtype)

    L = int(pred_atom14.shape[0])
    residue_indices = torch.arange(L, device=device).repeat_interleave(ATOM14_NUM_SLOTS)
    radii = torch.zeros((L, ATOM14_NUM_SLOTS), dtype=pred_atom14.dtype, device=device)
    for residue_index, residue_type in enumerate(aatype.tolist()):
        atom_names = ATOM14_NAMES_BY_RESTYPE_INDEX[int(residue_type)]
        for slot, atom_name in enumerate(atom_names):
            element = _atom_name_to_element(atom_name)
            if element:
                radii[residue_index, slot] = MOLPROBITY_HEAVY_ATOM_VDW_RADII.get(element, 1.70)
    flat_radii = radii.reshape(-1)
    valid = point_mask[:, None] & point_mask[None, :]
    valid &= residue_indices[:, None] < residue_indices[None, :]
    valid &= torch.abs(residue_indices[:, None] - residue_indices[None, :]) > 1
    valid &= flat_radii[:, None] > 0
    valid &= flat_radii[None, :] > 0
    if not bool(valid.any()):
        return torch.ones((), device=device, dtype=pred_atom14.dtype)

    distances = torch.cdist(pred_points, pred_points, p=2)
    overlap = flat_radii[:, None] + flat_radii[None, :] - distances
    clashes = ((overlap > overlap_cutoff) & valid).sum().to(dtype=pred_atom14.dtype)
    n_atoms = point_mask.sum().to(dtype=pred_atom14.dtype).clamp_min(1.0)
    clashes_per_1000_atoms = 1000.0 * clashes / n_atoms
    return 1.0 / (1.0 + clashes_per_1000_atoms / 100.0)


def _dihedral_angle(p0: torch.Tensor, p1: torch.Tensor, p2: torch.Tensor, p3: torch.Tensor) -> torch.Tensor:
    b0 = p0 - p1
    b1 = p2 - p1
    b2 = p3 - p2
    b1 = b1 / torch.linalg.vector_norm(b1, dim=-1, keepdim=True).clamp_min(1e-8)
    v = b0 - (b0 * b1).sum(dim=-1, keepdim=True) * b1
    w = b2 - (b2 * b1).sum(dim=-1, keepdim=True) * b1
    x = (v * w).sum(dim=-1)
    y = (torch.cross(b1, v, dim=-1) * w).sum(dim=-1)
    return torch.atan2(y, x)


@torch.no_grad()
def backbone_atom14_score(
    pred_atom14: torch.Tensor,
    true_atom14: torch.Tensor,
    atom14_mask: torch.Tensor,
) -> torch.Tensor:
    """Compute high-is-better phi/psi/omega backbone dihedral agreement."""
    device = pred_atom14.device
    true = true_atom14.to(device=device, dtype=pred_atom14.dtype)
    mask = atom14_mask.to(device=device, dtype=torch.bool)
    L = int(pred_atom14.shape[0])
    if L < 2:
        return torch.ones((), device=device, dtype=pred_atom14.dtype)

    errors_by_angle: Dict[str, list[torch.Tensor]] = {"phi": [], "psi": [], "omega": []}
    for idx in range(L):
        if idx > 0:
            phi_mask = mask[idx - 1, 2] & mask[idx, 0] & mask[idx, 1] & mask[idx, 2]
            if bool(phi_mask.item()):
                errors_by_angle["phi"].append(
                    _angular_error(
                        _dihedral_angle(
                            pred_atom14[idx - 1, 2],
                            pred_atom14[idx, 0],
                            pred_atom14[idx, 1],
                            pred_atom14[idx, 2],
                        ),
                        _dihedral_angle(true[idx - 1, 2], true[idx, 0], true[idx, 1], true[idx, 2]),
                    )
                )
        if idx < L - 1:
            psi_mask = mask[idx, 0] & mask[idx, 1] & mask[idx, 2] & mask[idx + 1, 0]
            omega_mask = mask[idx, 1] & mask[idx, 2] & mask[idx + 1, 0] & mask[idx + 1, 1]
            if bool(psi_mask.item()):
                errors_by_angle["psi"].append(
                    _angular_error(
                        _dihedral_angle(
                            pred_atom14[idx, 0],
                            pred_atom14[idx, 1],
                            pred_atom14[idx, 2],
                            pred_atom14[idx + 1, 0],
                        ),
                        _dihedral_angle(true[idx, 0], true[idx, 1], true[idx, 2], true[idx + 1, 0]),
                    )
                )
            if bool(omega_mask.item()):
                errors_by_angle["omega"].append(
                    _angular_error(
                        _dihedral_angle(
                            pred_atom14[idx, 1],
                            pred_atom14[idx, 2],
                            pred_atom14[idx + 1, 0],
                            pred_atom14[idx + 1, 1],
                        ),
                        _dihedral_angle(true[idx, 1], true[idx, 2], true[idx + 1, 0], true[idx + 1, 1]),
                    )
                )

    class_errors = [torch.stack(values).mean() for values in errors_by_angle.values() if values]
    if not class_errors:
        return torch.ones((), device=device, dtype=pred_atom14.dtype)
    error = torch.stack(class_errors).mean()
    return torch.clamp(1.0 - error, min=0.0, max=1.0)


@torch.no_grad()
def dipdiff_atom14_score(
    pred_atom14: torch.Tensor,
    true_atom14: torch.Tensor,
    atom14_mask: torch.Tensor,
) -> torch.Tensor:
    """Compute a DipDiff-inspired local CA/O backbone distance score."""
    device = pred_atom14.device
    true = true_atom14.to(device=device, dtype=pred_atom14.dtype)
    mask = atom14_mask.to(device=device, dtype=torch.bool)
    local_scores = []
    L = int(pred_atom14.shape[0])
    for idx in range(L):
        window_residues = [j for j in (idx - 1, idx, idx + 1) if 0 <= j < L]
        pred_points = []
        true_points = []
        for residue_index in window_residues:
            for slot in (CA_ATOM14_SLOT, 3):
                if bool(mask[residue_index, slot].item()):
                    pred_points.append(pred_atom14[residue_index, slot])
                    true_points.append(true[residue_index, slot])
        if len(pred_points) < 2:
            continue
        pred_window = torch.stack(pred_points)
        true_window = torch.stack(true_points)
        pred_dist = torch.cdist(pred_window, pred_window, p=2)
        true_dist = torch.cdist(true_window, true_window, p=2)
        pair_mask = ~torch.eye(len(pred_points), dtype=torch.bool, device=device)
        diff = torch.abs(pred_dist - true_dist)[pair_mask]
        thresholds = torch.tensor([0.25, 0.5, 1.0, 2.0], dtype=diff.dtype, device=device)
        local_scores.append((diff[None, :] < thresholds[:, None]).to(dtype=diff.dtype).mean())
    if not local_scores:
        return torch.ones((), device=device, dtype=pred_atom14.dtype)
    return torch.stack(local_scores).mean()


def foldscore_from_components(
    *,
    gdt_ha_ca_score: torch.Tensor | float,
    lddt_atom14_score: torch.Tensor | float,
    cad_atom14_score: torch.Tensor | float,
    sg_atom14_score: torch.Tensor | float,
    sc_atom14_score: torch.Tensor | float,
    molprobity_clash_atom14_score: torch.Tensor | float,
    bb_atom14_score: torch.Tensor | float,
    dipdiff_atom14_score: torch.Tensor | float,
) -> torch.Tensor:
    gdt = torch.as_tensor(gdt_ha_ca_score)
    atom14 = torch.as_tensor(lddt_atom14_score, device=gdt.device, dtype=gdt.dtype)
    cad = torch.as_tensor(cad_atom14_score, device=gdt.device, dtype=gdt.dtype)
    sg = torch.as_tensor(sg_atom14_score, device=gdt.device, dtype=gdt.dtype)
    sc = torch.as_tensor(sc_atom14_score, device=gdt.device, dtype=gdt.dtype)
    clash = torch.as_tensor(molprobity_clash_atom14_score, device=gdt.device, dtype=gdt.dtype)
    bb = torch.as_tensor(bb_atom14_score, device=gdt.device, dtype=gdt.dtype)
    dipdiff = torch.as_tensor(dipdiff_atom14_score, device=gdt.device, dtype=gdt.dtype)
    return (
        FOLDSCORE_GDT_HA_CA_WEIGHT * gdt
        + FOLDSCORE_LDDT_ATOM14_WEIGHT * atom14
        + FOLDSCORE_CAD_ATOM14_WEIGHT * cad
        + FOLDSCORE_SG_ATOM14_WEIGHT * sg
        + FOLDSCORE_SC_ATOM14_WEIGHT * sc
        + FOLDSCORE_MOLPROBITY_CLASH_ATOM14_WEIGHT * clash
        + FOLDSCORE_BB_ATOM14_WEIGHT * bb
        + FOLDSCORE_DIPDIFF_ATOM14_WEIGHT * dipdiff
    )


@torch.no_grad()
def foldscore_components(
    pred_atom14: torch.Tensor,
    true_atom14: torch.Tensor,
    atom14_mask: torch.Tensor,
    aatype: torch.Tensor,
    cutoff: float = 15.0,
) -> Dict[str, torch.Tensor]:
    """Compute nanoFold FoldScore components for one chain.

    Inputs are single-chain tensors:
      pred_atom14: (L,14,3)
      true_atom14: (L,14,3)
      atom14_mask: (L,14) bool
      aatype: (L,)
    """
    if pred_atom14.ndim != 3 or pred_atom14.shape[1:] != (ATOM14_NUM_SLOTS, 3):
        raise ValueError(f"`pred_atom14` must have shape (L, 14, 3), got {tuple(pred_atom14.shape)}")
    if true_atom14.ndim != 3 or true_atom14.shape[1:] != (ATOM14_NUM_SLOTS, 3):
        raise ValueError(f"`true_atom14` must have shape (L, 14, 3), got {tuple(true_atom14.shape)}")
    if atom14_mask.ndim != 2 or atom14_mask.shape[1] != ATOM14_NUM_SLOTS:
        raise ValueError(f"`atom14_mask` must have shape (L, 14), got {tuple(atom14_mask.shape)}")

    L = min(int(pred_atom14.shape[0]), int(true_atom14.shape[0]), int(atom14_mask.shape[0]))
    if L <= 0:
        zero = torch.zeros((), device=pred_atom14.device)
        return {name: zero for name in FOLDSCORE_COMPONENT_NAMES}

    pred = pred_atom14[:L]
    true = true_atom14[:L].to(device=pred.device, dtype=pred.dtype)
    mask = atom14_mask[:L].to(device=pred.device, dtype=torch.bool)
    residue_types = _as_aatype_tensor(aatype, length=L, device=pred.device)

    ca_mask = mask[:, CA_ATOM14_SLOT]
    ca_score = lddt_ca(
        pred[:, CA_ATOM14_SLOT, :],
        true[:, CA_ATOM14_SLOT, :],
        ca_mask,
        cutoff=cutoff,
    )
    gdt_ha_score = gdt_ha_ca(
        pred[:, CA_ATOM14_SLOT, :],
        true[:, CA_ATOM14_SLOT, :],
        ca_mask,
    )
    gdt_ts_score = gdt_ts_ca(
        pred[:, CA_ATOM14_SLOT, :],
        true[:, CA_ATOM14_SLOT, :],
        ca_mask,
    )

    pred_backbone, mask_backbone, backbone_residue_ids = _flatten_atom14_with_residue_ids(
        pred,
        mask,
        slots=BACKBONE_ATOM14_SLOTS,
    )
    true_backbone, _, _ = _flatten_atom14_with_residue_ids(true, mask, slots=BACKBONE_ATOM14_SLOTS)
    backbone_score = lddt_atom_points(
        pred_backbone,
        true_backbone,
        mask_backbone,
        residue_ids=backbone_residue_ids,
        cutoff=cutoff,
    )

    pred_all, mask_all, all_residue_ids = _flatten_atom14_with_residue_ids(pred, mask)
    true_all, _, _ = _flatten_atom14_with_residue_ids(true, mask)
    atom14_score = lddt_atom_points(
        pred_all,
        true_all,
        mask_all,
        residue_ids=all_residue_ids,
        cutoff=cutoff,
    )
    cad_score = cad_atom14_score(pred, true, mask)
    sg_score = spheregrinder_atom14_score(pred, true, mask)
    sc_score = sidechain_atom14_score(pred, true, mask, residue_types)
    clash_score = molprobity_clash_atom14_score(pred, mask, residue_types)
    bb_score = backbone_atom14_score(pred, true, mask)
    dipdiff_score = dipdiff_atom14_score(pred, true, mask)

    foldscore = foldscore_from_components(
        gdt_ha_ca_score=gdt_ha_score,
        lddt_atom14_score=atom14_score,
        cad_atom14_score=cad_score,
        sg_atom14_score=sg_score,
        sc_atom14_score=sc_score,
        molprobity_clash_atom14_score=clash_score,
        bb_atom14_score=bb_score,
        dipdiff_atom14_score=dipdiff_score,
    )
    return {
        "foldscore": foldscore,
        "gdt_ha_ca": gdt_ha_score,
        "lddt_atom14": atom14_score,
        "cad_atom14": cad_score,
        "sg_atom14": sg_score,
        "sc_atom14": sc_score,
        "molprobity_clash_atom14": clash_score,
        "bb_atom14": bb_score,
        "dipdiff_atom14": dipdiff_score,
        "gdt_ts_ca": gdt_ts_score,
        "lddt_ca": ca_score,
        "lddt_backbone_atom14": backbone_score,
    }


def foldscore_auc(points: Iterable[tuple[int | float, int | float, float]], *, sample_budget: int | float) -> float:
    """Trapezoidal AUC over `(step, cumulative_samples_seen, foldscore)` points."""
    rows = sorted((float(samples), float(score)) for _step, samples, score in points)
    if not rows:
        return float("nan")
    if len(rows) == 1:
        return rows[0][1]

    area = 0.0
    previous_samples = rows[0][0]
    for samples, _score in rows[1:]:
        if samples <= previous_samples:
            raise ValueError("AUC points must have strictly increasing cumulative sample counts.")
        previous_samples = samples

    for idx in range(1, len(rows)):
        x0, y0 = rows[idx - 1]
        x1, y1 = rows[idx]
        area += (x1 - x0) * (y0 + y1) * 0.5
    return area / max(float(sample_budget), 1.0)
