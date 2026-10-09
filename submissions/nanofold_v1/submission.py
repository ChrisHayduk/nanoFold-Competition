from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any, Dict

import torch

# Ensure repo root is on sys.path
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from nanofold.models.nanofold_model import (
    NanoFoldConfig,
    NanoFoldLoss,
    NanoFoldModel,
)


def build_model(cfg: Dict[str, Any]) -> torch.nn.Module:
    """Builds and initializes the complete NanoFoldModel (~2.83M parameters)."""
    model_cfg = cfg.get("model", {})
    nano_cfg = NanoFoldConfig()

    # Configure dimensions from submission config if specified
    if "d_single" in model_cfg:
        nano_cfg.pairformer.d_single = int(model_cfg["d_single"])
    if "d_pair" in model_cfg:
        nano_cfg.pairformer.d_pair = int(model_cfg["d_pair"])
    if "n_pairformer_blocks" in model_cfg:
        nano_cfg.pairformer.n_blocks = int(model_cfg["n_pairformer_blocks"])
    if "thinker_steps" in model_cfg:
        nano_cfg.thinker.num_steps = int(model_cfg["thinker_steps"])
    if "d_workspace" in model_cfg:
        nano_cfg.thinker.d_h = int(model_cfg["d_workspace"])
    elif "d_h" in model_cfg:
        nano_cfg.thinker.d_h = int(model_cfg["d_h"])
    if "k_tokens" in model_cfg:
        nano_cfg.predictor.k_tokens = int(model_cfg["k_tokens"])
    if "d_tok" in model_cfg:
        nano_cfg.predictor.d_tok = int(model_cfg["d_tok"])
    if "trans_scale" in model_cfg:
        nano_cfg.thinker.trans_scale = float(model_cfg["trans_scale"])

    model = NanoFoldModel(nano_cfg)
    model.loss_fn = NanoFoldLoss(nano_cfg)
    return model


def build_optimizer(cfg: Dict[str, Any], model: torch.nn.Module) -> torch.optim.Optimizer:
    """Builds the AdamW optimizer with decoupled weight decay."""
    optim_cfg = cfg.get("optim", {})
    return torch.optim.AdamW(
        model.parameters(),
        lr=float(optim_cfg.get("lr", 1.0e-4)),
        weight_decay=float(optim_cfg.get("weight_decay", 1.0e-2)),
        betas=(float(optim_cfg.get("beta1", 0.9)), float(optim_cfg.get("beta2", 0.999))),
        eps=float(optim_cfg.get("eps", 1.0e-8)),
    )


def build_scheduler(cfg: Dict[str, Any], optimizer: torch.optim.Optimizer) -> Any:
    """Builds learning rate scheduler with linear warmup and cosine decay."""
    max_steps = int(cfg.get("train", {}).get("max_steps", 30000))
    warmup_steps = int(cfg.get("train", {}).get("warmup_steps", 1000))
    min_lr_ratio = float(cfg.get("optim", {}).get("min_lr_ratio", 0.1))

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, max_steps - warmup_steps))
        progress = min(max(progress, 0.0), 1.0)
        return min_lr_ratio + 0.5 * (1.0 - min_lr_ratio) * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def run_batch(
    model: torch.nn.Module,
    batch: Dict[str, torch.Tensor],
    cfg: Dict[str, Any],
    training: bool,
) -> Dict[str, torch.Tensor]:
    """Executes a single forward step (and loss computation when training=True).

    Args:
        model: NanoFoldModel instance.
        batch: Feature and label tensor dictionary.
        cfg: Configuration dictionary.
        training: Boolean flag indicating train vs inference mode.

    Returns:
        Dictionary containing 'pred_atom14' of shape (B, L, 14, 3) and 'loss' if training.
    """
    aatype = batch["aatype"]
    msa = batch["msa"]
    deletions = batch["deletions"].float()
    residue_mask = batch["residue_mask"].bool()
    residue_index = batch.get("residue_index")

    model_out = model(
        aatype=aatype,
        msa=msa,
        deletions=deletions,
        residue_mask=residue_mask,
        residue_index=residue_index,
    )

    pred_atom14 = model_out["pred_atom14"]

    has_supervision = "atom14_positions" in batch and "atom14_mask" in batch
    if not has_supervision:
        return {"pred_atom14": pred_atom14}

    loss_fn = getattr(model, "loss_fn", None)
    if loss_fn is None:
        loss_fn = NanoFoldLoss()
        model.loss_fn = loss_fn

    loss_dict = loss_fn(model_out, batch)

    return {
        "pred_atom14": pred_atom14,
        "loss": loss_dict["loss"],
        "loss_fape": loss_dict.get("loss_fape", torch.tensor(0.0)),
        "loss_lddt": loss_dict.get("loss_lddt", torch.tensor(0.0)),
        "loss_atom14": loss_dict["loss_atom14"],
        "loss_distogram": loss_dict["loss_distogram"],
    }
