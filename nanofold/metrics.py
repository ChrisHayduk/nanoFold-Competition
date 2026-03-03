from __future__ import annotations

import torch


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
