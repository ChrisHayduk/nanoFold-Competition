from __future__ import annotations

from typing import Dict, Iterable

import torch

from .residue_constants import ATOM14_NUM_SLOTS, CA_ATOM14_SLOT

FOLDSCORE_LDDT_CA_WEIGHT = 0.55
FOLDSCORE_LDDT_BACKBONE_WEIGHT = 0.30
FOLDSCORE_LDDT_ATOM14_WEIGHT = 0.15
BACKBONE_ATOM14_SLOTS = (0, 1, 2, 3)


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


@torch.no_grad()
def lddt_atom_points(
    pred_points: torch.Tensor,
    true_points: torch.Tensor,
    point_mask: torch.Tensor,
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


def foldscore_from_components(
    *,
    lddt_ca_score: torch.Tensor | float,
    lddt_backbone_atom14_score: torch.Tensor | float,
    lddt_atom14_score: torch.Tensor | float,
) -> torch.Tensor:
    ca = torch.as_tensor(lddt_ca_score)
    backbone = torch.as_tensor(lddt_backbone_atom14_score, device=ca.device, dtype=ca.dtype)
    atom14 = torch.as_tensor(lddt_atom14_score, device=ca.device, dtype=ca.dtype)
    return (
        FOLDSCORE_LDDT_CA_WEIGHT * ca
        + FOLDSCORE_LDDT_BACKBONE_WEIGHT * backbone
        + FOLDSCORE_LDDT_ATOM14_WEIGHT * atom14
    )


@torch.no_grad()
def foldscore_components(
    pred_atom14: torch.Tensor,
    true_atom14: torch.Tensor,
    atom14_mask: torch.Tensor,
    cutoff: float = 15.0,
) -> Dict[str, torch.Tensor]:
    """Compute nanoFold FoldScore components for one chain.

    Inputs are single-chain tensors:
      pred_atom14: (L,14,3)
      true_atom14: (L,14,3)
      atom14_mask: (L,14) bool
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
        return {
            "lddt_ca": zero,
            "lddt_backbone_atom14": zero,
            "lddt_atom14": zero,
            "foldscore": zero,
        }

    pred = pred_atom14[:L]
    true = true_atom14[:L].to(device=pred.device, dtype=pred.dtype)
    mask = atom14_mask[:L].to(device=pred.device, dtype=torch.bool)

    ca_mask = mask[:, CA_ATOM14_SLOT]
    ca_score = lddt_ca(
        pred[:, CA_ATOM14_SLOT, :],
        true[:, CA_ATOM14_SLOT, :],
        ca_mask,
        cutoff=cutoff,
    )

    pred_backbone, mask_backbone = _flatten_atom14(pred, mask, slots=BACKBONE_ATOM14_SLOTS)
    true_backbone, _ = _flatten_atom14(true, mask, slots=BACKBONE_ATOM14_SLOTS)
    backbone_score = lddt_atom_points(
        pred_backbone,
        true_backbone,
        mask_backbone,
        cutoff=cutoff,
    )

    pred_all, mask_all = _flatten_atom14(pred, mask)
    true_all, _ = _flatten_atom14(true, mask)
    atom14_score = lddt_atom_points(
        pred_all,
        true_all,
        mask_all,
        cutoff=cutoff,
    )

    foldscore = foldscore_from_components(
        lddt_ca_score=ca_score,
        lddt_backbone_atom14_score=backbone_score,
        lddt_atom14_score=atom14_score,
    )
    return {
        "lddt_ca": ca_score,
        "lddt_backbone_atom14": backbone_score,
        "lddt_atom14": atom14_score,
        "foldscore": foldscore,
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
