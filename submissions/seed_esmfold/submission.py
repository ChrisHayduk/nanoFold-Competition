from __future__ import annotations

from typing import Any, Dict, Tuple

import torch
import torch.nn.functional as F

from nanofold.model import distogram_loss
from esmfold_seed_model import build_model


def build_optimizer(cfg: Dict[str, Any], model: torch.nn.Module) -> torch.optim.Optimizer:
    optim_cfg = cfg.get("optim", {})
    name = str(optim_cfg.get("name", "adamw")).lower()
    lr = float(optim_cfg.get("lr", 1.0e-4))
    weight_decay = float(optim_cfg.get("weight_decay", 1.0e-2))

    if name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)


def _valid_residue_mask(batch: Dict[str, torch.Tensor]) -> torch.Tensor:
    return batch["residue_mask"] & batch["ca_mask"]


def _pair_valid_mask(valid_res: torch.Tensor) -> torch.Tensor:
    B, L = valid_res.shape
    pair_mask = valid_res[:, :, None] & valid_res[:, None, :]
    tri = torch.triu(torch.ones((L, L), dtype=torch.bool, device=valid_res.device), diagonal=1)[None, :, :]
    return pair_mask & tri


def _true_distogram_bins(true_ca: torch.Tensor, n_bins: int, min_bin: float, max_bin: float) -> torch.Tensor:
    d_true = torch.cdist(true_ca, true_ca, p=2)
    boundaries = torch.linspace(min_bin, max_bin, n_bins - 1, device=true_ca.device, dtype=true_ca.dtype)
    return torch.bucketize(d_true, boundaries).long()


def _distogram_ce_loss(
    logits: torch.Tensor,
    true_ca: torch.Tensor,
    valid_res: torch.Tensor,
    min_bin: float,
    max_bin: float,
) -> torch.Tensor:
    # logits: (B,L,L,K)
    K = logits.shape[-1]
    targets = _true_distogram_bins(true_ca, n_bins=K, min_bin=min_bin, max_bin=max_bin)
    mask = _pair_valid_mask(valid_res)
    if not mask.any():
        return torch.zeros((), device=logits.device, dtype=logits.dtype)

    logits_flat = logits[mask]  # (M,K)
    target_flat = targets[mask]  # (M,)
    return F.cross_entropy(logits_flat, target_flat)


def _lddt_ca_per_residue(
    pred_ca: torch.Tensor,
    true_ca: torch.Tensor,
    valid_res: torch.Tensor,
    cutoff: float = 15.0,
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, torch.Tensor]:
    # Returns: (per_res_lddt [B,L], has_neighbors [B,L])
    B, L, _ = pred_ca.shape
    d_true = torch.cdist(true_ca, true_ca, p=2)
    d_pred = torch.cdist(pred_ca, pred_ca, p=2)

    eye = torch.eye(L, dtype=torch.bool, device=pred_ca.device)[None, :, :]
    pair_mask = (d_true < cutoff) & valid_res[:, :, None] & valid_res[:, None, :] & (~eye)

    counts = pair_mask.sum(dim=-1)  # (B,L)
    has_neighbors = counts > 0

    diff = torch.abs(d_true - d_pred)
    thresholds = torch.tensor([0.5, 1.0, 2.0, 4.0], device=pred_ca.device, dtype=pred_ca.dtype)
    within = (diff[:, None, :, :] < thresholds[None, :, None, None]) & pair_mask[:, None, :, :]
    frac = within.sum(dim=-1) / (counts[:, None, :] + eps)  # (B,4,L)
    per_res_lddt = frac.mean(dim=1).clamp(0.0, 1.0)  # (B,L)
    return per_res_lddt, has_neighbors


def _plddt_ce_loss(
    logits: torch.Tensor,
    pred_ca: torch.Tensor,
    true_ca: torch.Tensor,
    valid_res: torch.Tensor,
) -> torch.Tensor:
    # logits: (B,L,K)
    K = logits.shape[-1]
    target_scores, has_neighbors = _lddt_ca_per_residue(pred_ca=pred_ca, true_ca=true_ca, valid_res=valid_res)
    target_bins = torch.clamp((target_scores * K).long(), min=0, max=K - 1)
    mask = valid_res & has_neighbors
    if not mask.any():
        return torch.zeros((), device=logits.device, dtype=logits.dtype)

    logits_flat = logits[mask]  # (M,K)
    targets_flat = target_bins[mask]  # (M,)
    return F.cross_entropy(logits_flat, targets_flat)


def run_batch(
    model: torch.nn.Module,
    batch: Dict[str, torch.Tensor],
    cfg: Dict[str, Any],
    training: bool,
) -> Dict[str, torch.Tensor]:
    out = model(batch["aatype"], batch["residue_mask"])
    pred_ca = out["pred_ca"]
    distogram_logits = out["distogram_logits"]
    plddt_logits = out["plddt_logits"]
    if not training:
        return {"pred_ca": pred_ca}

    loss_cfg = cfg.get("loss", {})
    coord_weight = float(loss_cfg.get("coord_weight", 1.0))
    disto_weight = float(loss_cfg.get("distogram_weight", 0.3))
    plddt_weight = float(loss_cfg.get("plddt_weight", 0.1))

    coord_cutoff = float(loss_cfg.get("coord_cutoff", 15.0))
    disto_min_bin = float(loss_cfg.get("distogram_min_bin", 2.0))
    disto_max_bin = float(loss_cfg.get("distogram_max_bin", 22.0))

    valid_res = _valid_residue_mask(batch)

    coord = distogram_loss(
        pred_ca=pred_ca,
        true_ca=batch["ca_coords"],
        ca_mask=batch["ca_mask"],
        residue_mask=batch["residue_mask"],
        cutoff=coord_cutoff,
    )
    disto = _distogram_ce_loss(
        logits=distogram_logits,
        true_ca=batch["ca_coords"],
        valid_res=valid_res,
        min_bin=disto_min_bin,
        max_bin=disto_max_bin,
    )
    plddt = _plddt_ce_loss(
        logits=plddt_logits,
        pred_ca=pred_ca,
        true_ca=batch["ca_coords"],
        valid_res=valid_res,
    )

    loss = coord_weight * coord + disto_weight * disto + plddt_weight * plddt
    return {
        "pred_ca": pred_ca,
        "loss": loss,
        "coord_loss": coord.detach(),
        "distogram_ce_loss": disto.detach(),
        "plddt_ce_loss": plddt.detach(),
    }
