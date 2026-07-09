from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Iterable, NamedTuple, Sequence

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
    build_msa_features,
    build_supervision,
    build_target_feat,
)
from minalphafold.losses import AlphaFoldLoss  # noqa: E402
from minalphafold.model import AlphaFold2  # noqa: E402
from minalphafold.model_config import ModelConfig  # noqa: E402
from minalphafold.trainer import load_model_config, loss_inputs_from_batch  # noqa: E402

DEFAULT_MODEL_CONFIG_PATH = MINALPHAFOLD2_ROOT / "configs" / "tiny.toml"
AF2_INITIAL_SAMPLES = 10_000_000
AF2_FINETUNE_SAMPLES = 1_500_000
AF2_WARMUP_SAMPLES = 128_000
AF2_LR_DECAY_SAMPLES = 6_400_000
DEFAULT_FINETUNE_RAMP_STEPS = 500
FINETUNE_AUXILIARY_WEIGHT_ATTRS = (
    "structural_violation_weight",
    "experimentally_resolved_weight",
    "tm_score_weight",
)


class AlphaFoldBudgetSchedule(NamedTuple):
    max_steps: int
    finetune_start_step: int
    finetune_ramp_steps: int
    warmup_steps: int
    lr_decay_step: int
    finetune_lr_scale: float
    lr_decay_factor: float


def _bounded_step(value: int, *, max_steps: int) -> int:
    return max(0, min(int(value), int(max_steps)))


def _scaled_step(value_samples: int, total_samples: int, total_steps: int) -> int:
    if total_samples <= 0:
        raise ValueError("`total_samples` must be positive.")
    return int(round(int(total_steps) * float(value_samples) / float(total_samples)))


def _af2_budget_schedule(cfg: Dict[str, Any]) -> AlphaFoldBudgetSchedule:
    train_cfg = cfg.get("train", {})
    optim_cfg = cfg.get("optim", {})
    max_steps = int(train_cfg["max_steps"])
    default_finetune_start = _scaled_step(
        AF2_INITIAL_SAMPLES,
        AF2_INITIAL_SAMPLES + AF2_FINETUNE_SAMPLES,
        max_steps,
    )
    finetune_start_step = _bounded_step(
        int(train_cfg.get("finetune_start_step", default_finetune_start)),
        max_steps=max_steps,
    )
    warmup_steps = _bounded_step(
        int(
            train_cfg.get(
                "warmup_steps",
                _scaled_step(AF2_WARMUP_SAMPLES, AF2_INITIAL_SAMPLES, finetune_start_step),
            )
        ),
        max_steps=finetune_start_step,
    )
    lr_decay_step = _bounded_step(
        int(
            train_cfg.get(
                "lr_decay_step",
                _scaled_step(AF2_LR_DECAY_SAMPLES, AF2_INITIAL_SAMPLES, finetune_start_step),
            )
        ),
        max_steps=max_steps,
    )
    finetune_ramp_steps = max(
        0,
        int(train_cfg.get("finetune_ramp_steps", DEFAULT_FINETUNE_RAMP_STEPS)),
    )
    return AlphaFoldBudgetSchedule(
        max_steps=max_steps,
        finetune_start_step=finetune_start_step,
        finetune_ramp_steps=finetune_ramp_steps,
        warmup_steps=warmup_steps,
        lr_decay_step=lr_decay_step,
        finetune_lr_scale=float(train_cfg.get("finetune_lr_scale", 0.5)),
        lr_decay_factor=float(optim_cfg.get("lr_decay_factor", 0.95)),
    )


def _runtime_step(cfg: Dict[str, Any]) -> int:
    runtime = cfg.get("_runtime", {})
    if not isinstance(runtime, dict):
        return 0
    return int(runtime.get("step", 0))


def _use_finetune_loss(cfg: Dict[str, Any]) -> bool:
    return _runtime_step(cfg) >= _af2_budget_schedule(cfg).finetune_start_step


def _finetune_ramp_weight(cfg: Dict[str, Any]) -> float:
    schedule = _af2_budget_schedule(cfg)
    step = _runtime_step(cfg)
    if step < schedule.finetune_start_step:
        return 0.0
    if schedule.finetune_ramp_steps <= 0:
        return 1.0
    return max(
        0.0,
        min(
            1.0,
            float(step - schedule.finetune_start_step) / float(schedule.finetune_ramp_steps),
        ),
    )


def _finetune_target_weights(loss_fn: AlphaFoldLoss) -> Dict[str, float]:
    target_weights = getattr(loss_fn, "nanofold_finetune_target_weights", None)
    if isinstance(target_weights, dict):
        return {str(name): float(value) for name, value in target_weights.items()}
    captured = {
        attr: float(getattr(loss_fn, attr))
        for attr in FINETUNE_AUXILIARY_WEIGHT_ATTRS
        if hasattr(loss_fn, attr)
    }
    loss_fn.nanofold_finetune_target_weights = captured
    return captured


def _apply_finetune_ramp(loss_fn: AlphaFoldLoss, ramp_weight: float) -> None:
    target_weights = _finetune_target_weights(loss_fn)
    ramp_weight = max(0.0, min(1.0, float(ramp_weight)))
    for attr, target_weight in target_weights.items():
        if hasattr(loss_fn, attr):
            setattr(loss_fn, attr, target_weight * ramp_weight)


def _initial_loss_fn(model: torch.nn.Module, features: Dict[str, torch.Tensor]) -> AlphaFoldLoss:
    loss_fn = getattr(model, "nanofold_initial_loss_fn", None)
    if loss_fn is None:
        loss_fn = AlphaFoldLoss(finetune=False).to(features["aatype"].device)
        model.nanofold_initial_loss_fn = loss_fn
    return loss_fn


def _finetune_loss_fn(model: torch.nn.Module, features: Dict[str, torch.Tensor]) -> AlphaFoldLoss:
    loss_fn = getattr(model, "nanofold_finetune_loss_fn", None)
    if loss_fn is None:
        loss_fn = AlphaFoldLoss(finetune=True).to(features["aatype"].device)
        _finetune_target_weights(loss_fn)
        model.nanofold_finetune_loss_fn = loss_fn
    return loss_fn


def _model_cfg(cfg: Dict[str, Any]) -> ModelConfig:
    model_cfg = cfg.get("model", {})
    profile_path = Path(str(model_cfg.get("profile_path", DEFAULT_MODEL_CONFIG_PATH)))
    if not profile_path.is_absolute():
        profile_path = REPO_ROOT / profile_path
    return load_model_config(profile_path)


def build_model(cfg: Dict[str, Any]) -> torch.nn.Module:
    model = AlphaFold2(_model_cfg(cfg))
    model.nanofold_initial_loss_fn = AlphaFoldLoss(finetune=False)
    model.nanofold_finetune_loss_fn = AlphaFoldLoss(finetune=True)
    _finetune_target_weights(model.nanofold_finetune_loss_fn)
    if int(cfg.get("model", {}).get("n_cycles", 1)) <= 1:
        for module_name in ("recycle_norm_s", "recycle_norm_z"):
            module = getattr(model, module_name, None)
            if module is not None:
                for param in module.parameters():
                    param.requires_grad_(False)
    return model


def build_optimizer(cfg: Dict[str, Any], model: torch.nn.Module) -> torch.optim.Optimizer:
    optim_cfg = cfg["optim"]
    return torch.optim.Adam(
        model.parameters(),
        lr=float(optim_cfg["lr"]),
        betas=(
            float(optim_cfg.get("beta1", 0.9)),
            float(optim_cfg.get("beta2", 0.999)),
        ),
        eps=float(optim_cfg.get("eps", 1.0e-6)),
        weight_decay=float(optim_cfg.get("weight_decay", 0.0)),
    )


class _AlphaFoldBudgetLRScheduler:
    def __init__(self, cfg: Dict[str, Any], optimizer: torch.optim.Optimizer):
        self.optimizer = optimizer
        self.base_lr = float(cfg["optim"]["lr"])
        self.schedule = _af2_budget_schedule(cfg)
        self.completed_steps = 0
        self._apply_lr()

    def _lr_for_completed_steps(self) -> float:
        if self.completed_steps >= self.schedule.finetune_start_step:
            lr = self.base_lr * self.schedule.finetune_lr_scale
            if self.completed_steps >= self.schedule.lr_decay_step:
                lr *= self.schedule.lr_decay_factor
            return lr
        if self.schedule.warmup_steps > 0 and self.completed_steps < self.schedule.warmup_steps:
            return self.base_lr * float(self.completed_steps) / float(self.schedule.warmup_steps)
        if self.completed_steps >= self.schedule.lr_decay_step:
            return self.base_lr * self.schedule.lr_decay_factor
        return self.base_lr

    def _apply_lr(self) -> None:
        lr = self._lr_for_completed_steps()
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def step(self) -> None:
        self.completed_steps += 1
        self._apply_lr()

    def state_dict(self) -> Dict[str, Any]:
        return {"completed_steps": self.completed_steps}

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.completed_steps = int(state.get("completed_steps", 0))
        self._apply_lr()


def build_scheduler(cfg: Dict[str, Any], optimizer: torch.optim.Optimizer) -> Any:
    return _AlphaFoldBudgetLRScheduler(cfg, optimizer)


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
    return build_msa_features(
        {"msa": msa, "deletions": deletions},
        msa_depth=msa_depth,
        extra_msa_depth=extra_msa_depth,
        training=training,
        block_delete_training_msa=True,
        block_delete_msa_fraction=0.3,
        block_delete_msa_randomize_num_blocks=False,
        block_delete_msa_num_blocks=5,
        masked_msa_probability=0.15,
    )


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
    between_segment_residues = batch.get("between_segment_residues")
    if between_segment_residues is not None:
        between_segment_residues = between_segment_residues.detach().cpu().long()
    batch_size, length = aatype.shape
    model_cfg = cfg.get("model", {})
    msa_depth = int(model_cfg.get("msa_depth", cfg.get("data", {}).get("msa_depth", msa.shape[1])))
    extra_msa_depth = int(model_cfg.get("extra_msa_depth", max(0, msa.shape[1] - msa_depth)))

    target_feat = torch.stack(
        [
            build_target_feat(
                aatype[i],
                None if between_segment_residues is None else between_segment_residues[i],
            )
            for i in range(batch_size)
        ],
        dim=0,
    )
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
        "masked_msa_target": _stack_padded(
            [item["masked_msa_target"] for item in msa_features],
            target_shape=(max_cluster, length, 23),
        ),
        "masked_msa_mask": _stack_padded(
            [item["masked_msa_mask"] for item in msa_features],
            target_shape=(max_cluster, length),
        ),
    }
    if "atom14_positions" in batch and "atom14_mask" in batch:
        atom14_positions = batch["atom14_positions"].detach().cpu().float()
        atom14_mask = (
            batch["atom14_mask"].detach().cpu().float()
            * residue_mask[:, :, None]
        )
        supervision = [
            build_supervision(aatype[i], atom14_positions[i], atom14_mask[i])
            for i in range(batch_size)
        ]
        for key in supervision[0]:
            out[key] = torch.stack([item[key] for item in supervision], dim=0)
        if "resolution" in batch:
            out["resolution"] = batch["resolution"].detach().cpu().float()
        else:
            out["resolution"] = torch.zeros(batch_size, dtype=torch.float32)
    out.update(_empty_template_features(batch_size=batch_size, length=length, device=aatype.device))
    return {
        key: value.to(device=model_device, non_blocking=True)
        for key, value in out.items()
    }


def _alphafold_loss(
    model: torch.nn.Module,
    features: Dict[str, torch.Tensor],
    model_out: Dict[str, torch.Tensor],
    cfg: Dict[str, Any],
) -> torch.Tensor:
    loss_inputs = loss_inputs_from_batch(features, model_out)
    if not _use_finetune_loss(cfg):
        return _initial_loss_fn(model, features)(**loss_inputs).mean()

    ramp_weight = _finetune_ramp_weight(cfg)
    if ramp_weight <= 0.0:
        return _initial_loss_fn(model, features)(**loss_inputs).mean()
    if ramp_weight >= 1.0:
        return _finetune_loss_fn(model, features)(**loss_inputs).mean()

    initial_loss = _initial_loss_fn(model, features)(**loss_inputs)
    finetune_loss = _finetune_loss_fn(model, features)(**loss_inputs)
    blended_loss = initial_loss.lerp(finetune_loss, ramp_weight)
    return blended_loss.mean()


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

    has_supervision = "atom14_positions" in batch and "atom14_mask" in batch
    if not has_supervision:
        return {"pred_atom14": pred_atom14}

    loss = _alphafold_loss(model, features, model_out, cfg)
    return {
        "pred_atom14": pred_atom14,
        "loss": loss,
        "alphafold_loss": loss.detach(),
    }
