from __future__ import annotations

import math
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


class AlphaFoldBudgetSchedule(NamedTuple):
    max_steps: int
    finetune_start_step: int
    finetune_ramp_steps: int
    warmup_steps: int
    lr_decay_step: int
    finetune_lr_scale: float
    lr_decay_factor: float


def _bounded_step(value: int, *, max_steps: int) -> int:
    if type(value) is not int or not 0 <= value <= max_steps:
        raise ValueError(f"Schedule steps must be integers in [0, {max_steps}], got {value!r}.")
    return value


def _scaled_step(value_samples: int, total_samples: int, total_steps: int) -> int:
    if total_samples <= 0:
        raise ValueError("`total_samples` must be positive.")
    return int(round(int(total_steps) * float(value_samples) / float(total_samples)))


def _af2_budget_schedule(cfg: Dict[str, Any]) -> AlphaFoldBudgetSchedule:
    train_cfg = cfg.get("train", {})
    optim_cfg = cfg.get("optim", {})
    max_steps = train_cfg["max_steps"]
    if type(max_steps) is not int or max_steps <= 0:
        raise ValueError("`train.max_steps` must be a positive integer.")
    default_finetune_start = _scaled_step(
        AF2_INITIAL_SAMPLES,
        AF2_INITIAL_SAMPLES + AF2_FINETUNE_SAMPLES,
        max_steps,
    )
    finetune_start_step = _bounded_step(
        train_cfg.get("finetune_start_step", default_finetune_start),
        max_steps=max_steps,
    )
    warmup_steps = _bounded_step(
        train_cfg.get(
                "warmup_steps",
                _scaled_step(AF2_WARMUP_SAMPLES, AF2_INITIAL_SAMPLES, finetune_start_step),
        ),
        max_steps=finetune_start_step,
    )
    lr_decay_step = _bounded_step(
        train_cfg.get(
                "lr_decay_step",
                _scaled_step(AF2_LR_DECAY_SAMPLES, AF2_INITIAL_SAMPLES, finetune_start_step),
        ),
        max_steps=max_steps,
    )
    finetune_ramp_steps = train_cfg.get("finetune_ramp_steps", DEFAULT_FINETUNE_RAMP_STEPS)
    if type(finetune_ramp_steps) is not int or finetune_ramp_steps < 0:
        raise ValueError("`train.finetune_ramp_steps` must be a nonnegative integer.")
    finetune_lr_scale = float(train_cfg.get("finetune_lr_scale", 0.5))
    lr_decay_factor = float(optim_cfg.get("lr_decay_factor", 0.95))
    if not all(math.isfinite(v) and v >= 0.0 for v in (finetune_lr_scale, lr_decay_factor)):
        raise ValueError("Learning-rate scale factors must be finite and nonnegative.")
    return AlphaFoldBudgetSchedule(
        max_steps=max_steps,
        finetune_start_step=finetune_start_step,
        finetune_ramp_steps=finetune_ramp_steps,
        warmup_steps=warmup_steps,
        lr_decay_step=lr_decay_step,
        finetune_lr_scale=finetune_lr_scale,
        lr_decay_factor=lr_decay_factor,
    )


def _runtime_step(cfg: Dict[str, Any]) -> int:
    runtime = cfg.get("_runtime", {})
    if not isinstance(runtime, dict):
        return 0
    return int(runtime.get("step", 0))


def _experiment_section(cfg: Dict[str, Any], name: str) -> Dict[str, Any]:
    experiment = cfg.get("experiment", {})
    if not isinstance(experiment, dict):
        raise ValueError("`experiment` must be a mapping.")
    section = experiment.get(name, {})
    if not isinstance(section, dict):
        raise ValueError(f"`experiment.{name}` must be a mapping.")
    allowed = {
        "fape": {"backbone_clamp_mode", "clamped_probability", "eval_mode"},
        "recycle": {"train_mode", "min_train_cycles", "max_train_cycles", "eval_mode"},
        "loss_schedule": {"mode", "rationale"},
    }
    unknown = set(section) - allowed[name]
    if unknown:
        raise ValueError(f"Unknown experiment.{name} fields: {sorted(unknown)}.")
    return section


def _loss_schedule_mode(cfg: Dict[str, Any]) -> str:
    mode = _experiment_section(cfg, "loss_schedule").get("mode", "finetune")
    if mode not in {"finetune", "initial_only"}:
        raise ValueError(f"Unknown loss schedule mode: {mode!r}.")
    return str(mode)


def _fape_clamp_weight(
    cfg: Dict[str, Any], features: Dict[str, torch.Tensor], *, training: bool,
) -> torch.Tensor | None:
    fape = _experiment_section(cfg, "fape")
    mode = fape.get("backbone_clamp_mode", "fully_clamped")
    modes = {"fully_clamped", "fully_unclamped", "batchwise", "samplewise", "soft_mix", "finetune_unclamp"}
    if mode not in modes:
        raise ValueError(f"Unknown backbone FAPE clamp mode: {mode!r}.")
    probability = float(fape.get("clamped_probability", 0.9))
    if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
        raise ValueError("`experiment.fape.clamped_probability` must be finite and in [0, 1].")
    expected_eval = "fully_unclamped" if mode == "fully_unclamped" else "expected_mix"
    if fape.get("eval_mode", expected_eval) != expected_eval:
        raise ValueError(f"FAPE mode {mode!r} requires eval_mode={expected_eval!r}.")
    if mode == "fully_clamped":
        return None
    if mode == "finetune_unclamp":
        if not _use_finetune_loss(cfg):
            return None
        mode = "fully_unclamped"
    device = features["aatype"].device
    if mode == "fully_unclamped":
        return torch.tensor(0.0, device=device)
    if not training or mode == "soft_mix":
        return torch.tensor(probability, device=device)
    if mode == "batchwise":
        # Share one decision across every microbatch/rank of an optimizer step.
        # A private CPU generator makes it independent of evaluation and device RNG.
        generator = torch.Generator().manual_seed(int(cfg.get("seed", 0)) * 1_000_003 + _runtime_step(cfg))
        return (torch.rand((), generator=generator) < probability).to(device=device, dtype=torch.float32)
    return (torch.rand(features["aatype"].shape[0], device=device) < probability).float()


def _recycle_plan(cfg: Dict[str, Any], *, training: bool) -> tuple[int, bool]:
    cycles = cfg.get("model", {}).get("n_cycles", 1)
    if type(cycles) is not int or cycles <= 0:
        raise ValueError("`model.n_cycles` must be a positive integer.")
    recycle = _experiment_section(cfg, "recycle")
    mode = recycle.get("train_mode", "sample_uniform")
    if mode not in {"sample_uniform", "fixed"}:
        raise ValueError(f"Unknown recycle train_mode: {mode!r}.")
    if recycle.get("eval_mode", "fixed_max") != "fixed_max":
        raise ValueError("`experiment.recycle.eval_mode` must be fixed_max.")
    minimum = recycle.get("min_train_cycles", 1)
    maximum = recycle.get("max_train_cycles", cycles)
    if type(minimum) is not int or minimum != 1:
        raise ValueError("`experiment.recycle.min_train_cycles` must be 1.")
    if type(maximum) is not int or maximum != cycles:
        raise ValueError("`experiment.recycle.max_train_cycles` must equal model.n_cycles.")
    return cycles, training and mode == "sample_uniform"


def _use_finetune_loss(cfg: Dict[str, Any]) -> bool:
    return (
        _loss_schedule_mode(cfg) != "initial_only"
        and _runtime_step(cfg) >= _af2_budget_schedule(cfg).finetune_start_step
    )


def _finetune_ramp_weight(cfg: Dict[str, Any]) -> float:
    if _loss_schedule_mode(cfg) == "initial_only":
        return 0.0
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
        model.nanofold_finetune_loss_fn = loss_fn
    return loss_fn


def _model_cfg(cfg: Dict[str, Any]) -> ModelConfig:
    model_cfg = cfg.get("model", {})
    profile_path = Path(str(model_cfg.get("profile_path", DEFAULT_MODEL_CONFIG_PATH)))
    if not profile_path.is_absolute():
        profile_path = REPO_ROOT / profile_path
    return load_model_config(profile_path)


def validate_config(cfg: Dict[str, Any]) -> None:
    """Check the adapter's scientific controls before allocating a model."""
    _recycle_plan(cfg, training=True)
    _loss_schedule_mode(cfg)
    _fape_clamp_weight(cfg, {"aatype": torch.zeros((1, 1), dtype=torch.long)}, training=False)
    _af2_budget_schedule(cfg)
    model_cfg = cfg.get("model", {})
    for field, default, minimum in (("msa_depth", 192, 1), ("extra_msa_depth", 0, 0), ("n_ensemble", 1, 1)):
        value = model_cfg.get(field, default)
        if type(value) is not int or value < minimum:
            raise ValueError(f"`model.{field}` must be an integer >= {minimum}.")
    supported = {"family", "figure_label", "paper_claim", "budget_label", "fape", "recycle", "loss_schedule"}
    unknown = set(cfg.get("experiment", {})) - supported
    if unknown:
        raise ValueError(f"Unsupported experiment fields: {sorted(unknown)}.")


def build_model(cfg: Dict[str, Any]) -> torch.nn.Module:
    validate_config(cfg)
    model = AlphaFold2(_model_cfg(cfg))
    model.nanofold_initial_loss_fn = AlphaFoldLoss(finetune=False)
    model.nanofold_finetune_loss_fn = AlphaFoldLoss(finetune=True)
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
        if set(state) != {"completed_steps"}:
            raise ValueError("Scheduler state must contain exactly `completed_steps`.")
        completed_steps = state["completed_steps"]
        if type(completed_steps) is not int or not 0 <= completed_steps <= self.schedule.max_steps:
            raise ValueError("Scheduler completed_steps must be an integer within the training budget.")
        self.completed_steps = completed_steps
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
    msa_mask = batch.get("msa_mask")
    if msa_mask is None:
        msa_mask = residue_mask[:, None, :].bool().expand_as(msa)
    else:
        msa_mask = msa_mask.detach().cpu()
        if msa_mask.dtype != torch.bool or msa_mask.shape != msa.shape:
            raise ValueError("msa_mask must be a boolean tensor with the same shape as msa.")
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

    msa_features = []
    for i in range(batch_size):
        real_length = int(residue_mask[i].sum().item())
        real_columns = torch.arange(length) < real_length
        real_rows = msa_mask[i].any(dim=-1)
        if (
            real_length == 0 or not bool(real_rows[0])
            or not torch.equal(residue_mask[i].bool(), real_columns)
            or not torch.equal(msa_mask[i], real_rows[:, None] & real_columns[None, :])
        ):
            raise ValueError("MSA padding must contain complete biological rows and a nonempty residue prefix.")
        # Feature construction computes profiles and samples rows. Strip batch
        # padding first so companions cannot change a protein's biological MSA.
        msa_features.append(_build_single_msa_features(
            msa=msa[i, real_rows, :real_length],
            deletions=deletions[i, real_rows, :real_length],
            msa_depth=msa_depth,
            extra_msa_depth=extra_msa_depth,
            training=training,
        ))
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
        # Atom-type existence includes padded residues unless explicitly masked.
        out["atom37_exists"] *= residue_mask[:, :, None]
        consecutive = residue_index[:, 1:] - residue_index[:, :-1] == 1
        if between_segment_residues is not None:
            consecutive &= between_segment_residues[:, 1:] == 0
            # The upstream violation loss detects peptide breaks through indices.
            out["loss_residue_index"] = residue_index + (between_segment_residues != 0).long().cumsum(dim=1)
        out["true_torsion_mask"][:, 1:, :2] *= consecutive[:, :, None]
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
    if "loss_residue_index" in features:
        loss_inputs["residue_index"] = features["loss_residue_index"]
    clamp_weight = _fape_clamp_weight(cfg, features, training=model.training)
    initial_loss_fn = _initial_loss_fn(model, features)
    initial_loss_fn.use_clamped_fape = clamp_weight
    if not _use_finetune_loss(cfg):
        return initial_loss_fn(**loss_inputs).mean()

    ramp_weight = _finetune_ramp_weight(cfg)
    if ramp_weight <= 0.0:
        return initial_loss_fn(**loss_inputs).mean()
    finetune_loss_fn = _finetune_loss_fn(model, features)
    finetune_loss_fn.use_clamped_fape = clamp_weight
    if ramp_weight >= 1.0:
        return finetune_loss_fn(**loss_inputs).mean()

    initial_loss = initial_loss_fn(**loss_inputs)
    finetune_loss = finetune_loss_fn(**loss_inputs)
    blended_loss = initial_loss.lerp(finetune_loss, ramp_weight)
    return blended_loss.mean()


def run_batch(
    model: torch.nn.Module,
    batch: Dict[str, torch.Tensor],
    cfg: Dict[str, Any],
    training: bool,
) -> Dict[str, torch.Tensor]:
    model.train(training)
    n_cycles, sample_recycles = _recycle_plan(cfg, training=training)
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
        n_cycles=n_cycles,
        sample_recycles=sample_recycles,
        n_ensemble=int(cfg.get("model", {}).get("n_ensemble", 1)),
    )
    model_out["atom14_mask"] = model_out["atom14_mask"] * features["seq_mask"][:, :, None]
    pred_atom14 = model_out["atom14_coords"] * model_out["atom14_mask"].to(
        device=model_out["atom14_coords"].device,
        dtype=model_out["atom14_coords"].dtype,
    )[:, :, :, None]
    metrics = {
        "msa_cluster_rows": features["msa_mask"].bool().any(dim=-1).float().sum(dim=-1).mean(),
        "msa_extra_rows": features["extra_msa_mask"].bool().any(dim=-1).float().sum(dim=-1).mean(),
        "recycle_cycles": pred_atom14.new_tensor(float(model_out["sampled_n_cycles"])),
    }

    has_supervision = "atom14_positions" in batch and "atom14_mask" in batch
    if not has_supervision:
        return {"pred_atom14": pred_atom14, **metrics}

    loss = _alphafold_loss(model, features, model_out, cfg)
    return {
        "pred_atom14": pred_atom14,
        "loss": loss,
        "alphafold_loss": loss.detach(),
        **metrics,
    }
