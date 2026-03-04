from __future__ import annotations

import math
from typing import Any, Dict

import torch
from torch.optim.lr_scheduler import LambdaLR

from .model import NanoFoldBaseline, baseline_composite_loss


def build_model(cfg: Dict[str, Any]) -> torch.nn.Module:
    model_cfg = cfg["model"]
    return NanoFoldBaseline(
        d_model=int(model_cfg["d_model"]),
        n_layers=int(model_cfg["n_layers"]),
        n_heads=int(model_cfg["n_heads"]),
        dropout=float(model_cfg.get("dropout", 0.0)),
        max_seq_len=int(model_cfg.get("max_seq_len", 1024)),
    )


def build_optimizer(cfg: Dict[str, Any], model: torch.nn.Module) -> torch.optim.Optimizer:
    optim_cfg = cfg["optim"]
    return torch.optim.AdamW(
        model.parameters(),
        lr=float(optim_cfg["lr"]),
        weight_decay=float(optim_cfg.get("weight_decay", 0.0)),
    )


def build_scheduler(cfg: Dict[str, Any], optimizer: torch.optim.Optimizer) -> Any:
    optim_cfg = cfg.get("optim", {})
    name = str(optim_cfg.get("scheduler", "none")).lower()
    if name in {"", "none", "off"}:
        return None
    if name != "warmup_cosine":
        raise ValueError(f"Unsupported optim.scheduler={name!r}; expected 'warmup_cosine' or 'none'.")

    max_steps = int(cfg.get("train", {}).get("max_steps", 0))
    if max_steps <= 0:
        raise ValueError("train.max_steps must be > 0 when using warmup_cosine scheduler.")

    base_lr = float(optim_cfg.get("lr", 0.0))
    min_lr = float(optim_cfg.get("min_lr", 0.0))
    warmup_steps = int(optim_cfg.get("warmup_steps", 0))
    if base_lr <= 0:
        raise ValueError("optim.lr must be > 0 when using warmup_cosine scheduler.")
    min_ratio = max(0.0, min(1.0, min_lr / base_lr))

    def lr_lambda(step: int) -> float:
        current = step + 1
        if warmup_steps > 0 and current <= warmup_steps:
            return max(current / warmup_steps, 1.0e-8)

        decay_steps = max(1, max_steps - warmup_steps)
        progress = min(max((current - warmup_steps) / decay_steps, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_ratio + (1.0 - min_ratio) * cosine

    return LambdaLR(optimizer, lr_lambda=lr_lambda)


def run_batch(
    model: torch.nn.Module,
    batch: Dict[str, torch.Tensor],
    cfg: Dict[str, Any],
    training: bool,
) -> Dict[str, torch.Tensor]:
    pred_ca = model(batch["aatype"], batch["msa"], batch["deletions"], batch["residue_mask"])
    if not training:
        return {"pred_ca": pred_ca}

    loss_cfg = cfg.get("loss", {})
    loss, terms = baseline_composite_loss(
        pred_ca=pred_ca,
        true_ca=batch["ca_coords"],
        ca_mask=batch["ca_mask"],
        residue_mask=batch["residue_mask"],
        local_cutoff=float(loss_cfg.get("local_cutoff", 15.0)),
        global_max_dist=float(loss_cfg.get("global_max_dist", 32.0)),
        local_weight=float(loss_cfg.get("local_weight", 1.0)),
        global_weight=float(loss_cfg.get("global_weight", 0.25)),
        bond_weight=float(loss_cfg.get("bond_weight", 0.10)),
    )
    return {
        "pred_ca": pred_ca,
        "loss": loss,
        "local_loss": terms["local_loss"].detach(),
        "global_loss": terms["global_loss"].detach(),
        "bond_loss": terms["bond_loss"].detach(),
    }
