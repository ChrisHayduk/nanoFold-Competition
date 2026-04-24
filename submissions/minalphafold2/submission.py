from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, Sequence

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
MINALPHAFOLD2_ROOT = REPO_ROOT / "third_party" / "minAlphaFold2"
if not (MINALPHAFOLD2_ROOT / "minalphafold").exists():
    raise ImportError(
        "Expected upstream minAlphaFold2 checkout at third_party/minAlphaFold2. "
        "Run `git submodule update --init --recursive`."
    )
if str(MINALPHAFOLD2_ROOT) not in sys.path:
    sys.path.insert(0, str(MINALPHAFOLD2_ROOT))

from minalphafold.data import (  # noqa: E402
    EXTRA_MSA_FEAT_DIM,
    MSA_FEAT_DIM,
    TEMPLATE_ANGLE_DIM,
    TEMPLATE_PAIR_DIM,
    build_extra_msa_feat,
    build_msa_feat,
    build_target_feat,
    cluster_statistics,
    sample_cluster_and_extra,
)
from minalphafold.model import AlphaFold2  # noqa: E402

MODEL_CONFIG_DEFAULTS: Dict[str, Any] = {
    "c_m": 32,
    "c_s": 32,
    "c_z": 16,
    "c_t": 16,
    "c_e": 24,
    "dim": 8,
    "num_heads": 4,
    "msa_transition_n": 2,
    "outer_product_dim": 8,
    "triangle_mult_c": 16,
    "triangle_dim": 8,
    "triangle_num_heads": 2,
    "pair_transition_n": 2,
    "template_pair_num_blocks": 1,
    "template_pair_dropout": 0.0,
    "template_pointwise_attention_dim": 8,
    "template_pointwise_num_heads": 2,
    "template_triangle_mult_c": 16,
    "template_triangle_attn_c": 8,
    "template_triangle_attn_num_heads": 2,
    "template_pair_transition_n": 2,
    "extra_msa_dim": 8,
    "extra_msa_dropout": 0.0,
    "extra_pair_dropout": 0.0,
    "msa_column_global_attention_dim": 8,
    "num_extra_msa": 1,
    "num_evoformer": 1,
    "evoformer_msa_dropout": 0.0,
    "evoformer_pair_dropout": 0.0,
    "structure_module_c": 16,
    "structure_module_layers": 2,
    "structure_module_dropout_ipa": 0.0,
    "structure_module_dropout_transition": 0.0,
    "sidechain_num_channel": 16,
    "sidechain_num_residual_block": 2,
    "position_scale": 10.0,
    "zero_init": True,
    "ipa_num_heads": 4,
    "ipa_c": 8,
    "ipa_n_query_points": 4,
    "ipa_n_value_points": 4,
    "n_dist_bins": 64,
    "plddt_hidden_dim": 32,
    "n_plddt_bins": 50,
    "n_msa_classes": 23,
    "n_pae_bins": 64,
}


def _model_cfg(cfg: Dict[str, Any]) -> SimpleNamespace:
    raw = dict(MODEL_CONFIG_DEFAULTS)
    raw.update(dict(cfg.get("model", {})))
    raw.setdefault("model_profile", "nanoFold_minAlphaFold2")
    return SimpleNamespace(**raw)


def build_model(cfg: Dict[str, Any]) -> torch.nn.Module:
    return AlphaFold2(_model_cfg(cfg))


def build_optimizer(cfg: Dict[str, Any], model: torch.nn.Module) -> torch.optim.Optimizer:
    optim_cfg = cfg["optim"]
    return torch.optim.AdamW(
        model.parameters(),
        lr=float(optim_cfg["lr"]),
        weight_decay=float(optim_cfg.get("weight_decay", 0.0)),
    )


def build_scheduler(cfg: Dict[str, Any], optimizer: torch.optim.Optimizer) -> Any:
    del cfg, optimizer
    return None


def _pad_tensor(tensor: torch.Tensor, target_shape: Sequence[int], value: float = 0.0) -> torch.Tensor:
    out = tensor.new_full(tuple(target_shape), value)
    slices = tuple(slice(0, int(size)) for size in tensor.shape)
    out[slices] = tensor
    return out


def _stack_padded(tensors: Iterable[torch.Tensor], target_shape: Sequence[int], value: float = 0.0) -> torch.Tensor:
    return torch.stack([_pad_tensor(t, target_shape=target_shape, value=value) for t in tensors], dim=0)


def _build_single_msa_features(
    *,
    msa: torch.Tensor,
    deletions: torch.Tensor,
    msa_depth: int,
    extra_msa_depth: int,
    training: bool,
) -> Dict[str, torch.Tensor]:
    cluster_msa, cluster_deletions, extra_msa, extra_deletions = sample_cluster_and_extra(
        msa,
        deletions,
        msa_depth=msa_depth,
        extra_msa_depth=extra_msa_depth,
        training=training,
    )
    cluster_profile, cluster_deletion_mean = cluster_statistics(
        cluster_msa,
        cluster_deletions,
        extra_msa,
        extra_deletions,
    )
    return {
        "msa_feat": build_msa_feat(cluster_msa, cluster_deletions, cluster_profile, cluster_deletion_mean),
        "extra_msa_feat": build_extra_msa_feat(extra_msa, extra_deletions),
        "msa_mask": torch.ones(cluster_msa.shape, dtype=torch.float32, device=msa.device),
        "extra_msa_mask": torch.ones(extra_msa.shape, dtype=torch.float32, device=msa.device),
    }


def _empty_template_features(batch_size: int, length: int, device: torch.device) -> Dict[str, torch.Tensor]:
    return {
        "template_pair_feat": torch.zeros(
            batch_size,
            0,
            length,
            length,
            TEMPLATE_PAIR_DIM,
            dtype=torch.float32,
            device=device,
        ),
        "template_angle_feat": torch.zeros(
            batch_size,
            0,
            length,
            TEMPLATE_ANGLE_DIM,
            dtype=torch.float32,
            device=device,
        ),
        "template_mask": torch.zeros(batch_size, 0, dtype=torch.float32, device=device),
        "template_residue_mask": torch.zeros(batch_size, 0, length, dtype=torch.float32, device=device),
    }


def _build_minalphafold_inputs(
    batch: Dict[str, torch.Tensor],
    cfg: Dict[str, Any],
    *,
    training: bool,
) -> Dict[str, torch.Tensor]:
    model_device = batch["aatype"].device
    aatype = batch["aatype"].detach().cpu().long()
    msa = batch["msa"].detach().cpu().long()
    deletions = batch["deletions"].detach().cpu().long()
    residue_mask = batch["residue_mask"].detach().cpu().float()
    batch_size, length = aatype.shape
    model_cfg = cfg.get("model", {})
    msa_depth = int(model_cfg.get("msa_depth", cfg.get("data", {}).get("msa_depth", msa.shape[1])))
    extra_msa_depth = int(model_cfg.get("extra_msa_depth", max(0, msa.shape[1] - msa_depth)))

    target_feat = torch.stack([build_target_feat(aatype[i]) for i in range(batch_size)], dim=0)
    residue_index = batch.get("residue_index")
    if residue_index is None:
        residue_index = torch.arange(length, device=aatype.device, dtype=torch.long).expand(batch_size, -1)
    else:
        residue_index = residue_index.detach().cpu().long()

    msa_features = [
        _build_single_msa_features(
            msa=msa[i],
            deletions=deletions[i],
            msa_depth=msa_depth,
            extra_msa_depth=extra_msa_depth,
            training=training,
        )
        for i in range(batch_size)
    ]
    max_cluster = max(item["msa_feat"].shape[0] for item in msa_features)
    max_extra = max(item["extra_msa_feat"].shape[0] for item in msa_features)

    out = {
        "target_feat": target_feat,
        "residue_index": residue_index,
        "aatype": aatype,
        "seq_mask": residue_mask,
        "msa_feat": _stack_padded(
            [item["msa_feat"] for item in msa_features],
            target_shape=(max_cluster, length, MSA_FEAT_DIM),
        ),
        "msa_mask": _stack_padded(
            [item["msa_mask"] for item in msa_features],
            target_shape=(max_cluster, length),
        ),
        "extra_msa_feat": _stack_padded(
            [item["extra_msa_feat"] for item in msa_features],
            target_shape=(max_extra, length, EXTRA_MSA_FEAT_DIM),
        ),
        "extra_msa_mask": _stack_padded(
            [item["extra_msa_mask"] for item in msa_features],
            target_shape=(max_extra, length),
        ),
    }
    out.update(_empty_template_features(batch_size=batch_size, length=length, device=aatype.device))
    return {
        key: value.to(device=model_device, non_blocking=True)
        for key, value in out.items()
    }


def _atom14_masked_mse(pred_atom14: torch.Tensor, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
    true_atom14 = batch["atom14_positions"].to(device=pred_atom14.device, dtype=pred_atom14.dtype)
    atom14_mask = batch["atom14_mask"].to(device=pred_atom14.device, dtype=pred_atom14.dtype)
    residue_mask = batch["residue_mask"].to(device=pred_atom14.device, dtype=pred_atom14.dtype)
    mask = atom14_mask * residue_mask[:, :, None]
    squared_error = (pred_atom14 - true_atom14).square().sum(dim=-1)
    return (squared_error * mask).sum() / mask.sum().clamp(min=1.0)


def run_batch(
    model: torch.nn.Module,
    batch: Dict[str, torch.Tensor],
    cfg: Dict[str, Any],
    training: bool,
) -> Dict[str, torch.Tensor]:
    features = _build_minalphafold_inputs(batch, cfg, training=training)
    model_out = model(
        target_feat=features["target_feat"],
        residue_index=features["residue_index"],
        msa_feat=features["msa_feat"],
        extra_msa_feat=features["extra_msa_feat"],
        template_pair_feat=features["template_pair_feat"],
        aatype=features["aatype"],
        template_angle_feat=features["template_angle_feat"],
        template_mask=features["template_mask"],
        template_residue_mask=features["template_residue_mask"],
        seq_mask=features["seq_mask"],
        msa_mask=features["msa_mask"],
        extra_msa_mask=features["extra_msa_mask"],
        n_cycles=int(cfg.get("model", {}).get("n_cycles", 1)),
        n_ensemble=int(cfg.get("model", {}).get("n_ensemble", 1)),
    )
    pred_atom14 = model_out["atom14_coords"] * batch["residue_mask"].to(
        device=model_out["atom14_coords"].device,
        dtype=model_out["atom14_coords"].dtype,
    )[:, :, None, None]

    if not training:
        return {"pred_atom14": pred_atom14}

    loss = _atom14_masked_mse(pred_atom14, batch)
    return {
        "pred_atom14": pred_atom14,
        "loss": loss,
        "atom14_mse": loss.detach(),
    }
