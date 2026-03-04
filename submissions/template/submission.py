from __future__ import annotations

from typing import Any, Dict

import torch

from nanofold.model import NanoFoldBaseline, distogram_loss


def build_model(cfg: Dict[str, Any]) -> torch.nn.Module:
    model_cfg = cfg["model"]
    return NanoFoldBaseline(
        d_model=int(model_cfg["d_model"]),
        n_layers=int(model_cfg["n_layers"]),
        n_heads=int(model_cfg["n_heads"]),
        dropout=float(model_cfg.get("dropout", 0.0)),
    )


def build_optimizer(cfg: Dict[str, Any], model: torch.nn.Module) -> torch.optim.Optimizer:
    optim_cfg = cfg["optim"]
    return torch.optim.AdamW(
        model.parameters(),
        lr=float(optim_cfg["lr"]),
        weight_decay=float(optim_cfg.get("weight_decay", 0.0)),
    )


def run_batch(
    model: torch.nn.Module,
    batch: Dict[str, torch.Tensor],
    cfg: Dict[str, Any],
    training: bool,
) -> Dict[str, torch.Tensor]:
    pred_ca = model(batch["aatype"], batch["msa"], batch["deletions"], batch["residue_mask"])
    loss = distogram_loss(
        pred_ca=pred_ca,
        true_ca=batch["ca_coords"],
        ca_mask=batch["ca_mask"],
        residue_mask=batch["residue_mask"],
    )
    return {
        "pred_ca": pred_ca,
        "loss": loss,
    }
