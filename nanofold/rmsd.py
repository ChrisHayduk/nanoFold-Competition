"""C-alpha Kabsch RMSD with explicit observation coverage and cohort semantics."""
from __future__ import annotations

import math
from typing import Any

import torch

RMSD_CA_VERSION = "kabsch-ca-v1"
RMSD_CA_FIELDS = frozenset({"rmsd_ca", "rmsd_ca_atom_count", "rmsd_ca_status", "rmsd_ca_version"})


def masked_kabsch_rmsd(pred: torch.Tensor, true: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Align observed C-alpha coordinates with a proper rotation; require >=3.

    Collinear or collapsed predictions retain their actual least-squares error;
    nonunique optimal rotation is not a reason to drop a difficult target.
    """
    if pred.ndim != 2 or pred.shape[-1] != 3 or true.shape != pred.shape or mask.shape != pred.shape[:1]:
        raise ValueError("RMSD requires aligned (L, 3) coordinates and an (L,) observation mask")
    pred = pred.float()
    true = true.to(device=pred.device, dtype=pred.dtype)
    active = mask.to(device=pred.device, dtype=torch.bool)
    if not torch.isfinite(pred[active]).all() or not torch.isfinite(true[active]).all():
        raise ValueError("Observed RMSD coordinates must be finite")
    if int(active.sum().item()) < 3:
        return torch.tensor(float("nan"), device=pred.device, dtype=pred.dtype)
    pred_active, true_active = pred[active], true[active]
    pred_centered = pred_active - pred_active.mean(dim=0, keepdim=True)
    true_centered = true_active - true_active.mean(dim=0, keepdim=True)
    u, _, vh = torch.linalg.svd(pred_centered.transpose(0, 1) @ true_centered, full_matrices=False)
    correction = torch.ones(3, device=pred.device, dtype=pred.dtype)
    if torch.det(u @ vh) < 0:
        correction[-1] = -1.0
    rotation = u @ torch.diag(correction) @ vh
    return torch.sqrt((pred_centered @ rotation - true_centered).square().sum(dim=-1).mean().clamp_min(0.0))


def rmsd_ca_diagnostic(pred: torch.Tensor, true: torch.Tensor, mask: torch.Tensor) -> dict[str, Any]:
    count = int(mask.to(dtype=torch.bool).sum().item())
    value = float(masked_kabsch_rmsd(pred, true, mask).detach().cpu())
    if count >= 3 and not math.isfinite(value):
        raise ValueError("RMSD calculation failed for an observed target")
    return {"rmsd_ca": value if count >= 3 else None, "rmsd_ca_atom_count": count,
            "rmsd_ca_status": "defined" if count >= 3 else "undefined_insufficient_ca",
            "rmsd_ca_version": RMSD_CA_VERSION}


def summarize_rmsd_ca(rows: list[dict[str, Any]], *, expected_count: int) -> dict[str, Any]:
    """A full-cohort RMSD mean is absent when any target is undefined."""
    if expected_count <= 0 or len(rows) != expected_count:
        raise ValueError("RMSD requires the complete nonempty target cohort")
    values: list[float] = []
    for row in rows:
        if not RMSD_CA_FIELDS <= row.keys() or row["rmsd_ca_version"] != RMSD_CA_VERSION:
            raise ValueError("Missing or incompatible per-target RMSD evidence")
        count, value = row["rmsd_ca_atom_count"], row["rmsd_ca"]
        if type(count) is not int or count < 0:
            raise ValueError("Invalid RMSD observed C-alpha count")
        if count < 3:
            if value is not None or row["rmsd_ca_status"] != "undefined_insufficient_ca":
                raise ValueError("Insufficient C-alpha support must be explicitly undefined")
        else:
            if (type(value) not in (float, int) or not math.isfinite(value) or value < 0
                    or row["rmsd_ca_status"] != "defined"):
                raise ValueError("Observed RMSD must be finite, nonnegative and defined")
            values.append(float(value))
    complete = len(values) == expected_count
    return {"mean_rmsd_ca": math.fsum(values) / expected_count if complete else None,
            "rmsd_ca_valid_count": len(values), "rmsd_ca_undefined_count": expected_count - len(values),
            "rmsd_ca_status": "complete" if complete else "undefined_targets",
            "rmsd_ca_version": RMSD_CA_VERSION}
