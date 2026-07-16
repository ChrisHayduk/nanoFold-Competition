from __future__ import annotations

import math
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, cast

import torch
import torch.nn.functional as F
import torch.utils.checkpoint as torch_checkpoint

REPO_ROOT = Path(__file__).resolve().parents[2]
MINALPHAFOLD2_ROOT = REPO_ROOT / "third_party" / "minAlphaFold2"
if not (MINALPHAFOLD2_ROOT / "minalphafold").exists():
    raise ImportError(
        "Expected upstream minAlphaFold2 checkout at third_party/minAlphaFold2. "
        "Run `git submodule update --init --recursive`."
    )
if str(MINALPHAFOLD2_ROOT) not in sys.path:
    sys.path.insert(0, str(MINALPHAFOLD2_ROOT))

from minalphafold.embedders import (  # noqa: E402
    PairTransition,
    TriangleAttentionEndingNode,
    TriangleAttentionStartingNode,
    TriangleMultiplicationIncoming,
    TriangleMultiplicationOutgoing,
)
from minalphafold.evoformer import Evoformer  # noqa: E402
from minalphafold.heads import (  # noqa: E402
    DistogramHead,
    ExperimentallyResolvedHead,
    MaskedMSAHead,
    PLDDTHead,
    TMScoreHead,
)
from minalphafold.initialization import init_gate_linear, init_linear  # noqa: E402
from minalphafold.losses import AlphaFoldLoss  # noqa: E402
from minalphafold.model_config import ModelConfig  # noqa: E402
from minalphafold.structure_module import StructureModule  # noqa: E402
from minalphafold.utils import dropout_columnwise, dropout_rowwise  # noqa: E402

from submissions.minalphafold2.submission import (  # noqa: E402
    _af2_budget_schedule,
    _alphafold_loss,
    _build_minalphafold_inputs,
    _finetune_target_weights,
    load_model_config,
)
from submissions.minalphafold2.submission import (  # noqa: E402
    _finetune_ramp_weight as _base_finetune_ramp_weight,
)


def _finetune_ramp_weight(cfg: Dict[str, Any]) -> float:
    return _base_finetune_ramp_weight(cfg)


class PairBiasedSelfAttention(torch.nn.Module):
    """QK-normalized single attention conditioned by the pair representation."""

    def __init__(self, config: ModelConfig, *, num_heads: int, dropout: float):
        super().__init__()
        if config.c_s % num_heads != 0:
            raise ValueError(f"model.c_s={config.c_s} must be divisible by single_attention_heads={num_heads}.")
        self.num_heads = int(num_heads)
        self.head_dim = config.c_s // self.num_heads
        self.dropout = float(dropout)

        self.single_norm = torch.nn.LayerNorm(config.c_s)
        self.pair_norm = torch.nn.LayerNorm(config.c_z)
        self.linear_q = torch.nn.Linear(config.c_s, config.c_s, bias=False)
        self.linear_k = torch.nn.Linear(config.c_s, config.c_s, bias=False)
        self.linear_v = torch.nn.Linear(config.c_s, config.c_s, bias=False)
        self.linear_gate = torch.nn.Linear(config.c_s, config.c_s)
        self.linear_pair_bias = torch.nn.Linear(config.c_z, self.num_heads, bias=False)
        self.linear_output = torch.nn.Linear(config.c_s, config.c_s)

        for linear in (self.linear_q, self.linear_k, self.linear_v, self.linear_pair_bias):
            init_linear(linear, init="glorot")
        init_gate_linear(self.linear_gate)
        init_linear(self.linear_output, init="final")

    @staticmethod
    def _qk_norm(value: torch.Tensor) -> torch.Tensor:
        return value * torch.rsqrt(value.square().mean(dim=-1, keepdim=True) + 1.0e-6)

    def forward(
        self,
        single_representation: torch.Tensor,
        pair_representation: torch.Tensor,
        seq_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, length, _ = single_representation.shape
        single = self.single_norm(single_representation)
        pair = self.pair_norm(pair_representation)

        q = self.linear_q(single).reshape(batch_size, length, self.num_heads, self.head_dim)
        k = self.linear_k(single).reshape(batch_size, length, self.num_heads, self.head_dim)
        v = self.linear_v(single).reshape(batch_size, length, self.num_heads, self.head_dim)
        q = self._qk_norm(q)
        k = self._qk_norm(k)

        scores = torch.einsum("bihd,bjhd->bhij", q, k) / math.sqrt(self.head_dim)
        scores = scores + self.linear_pair_bias(pair).permute(0, 3, 1, 2)
        key_mask = seq_mask.to(dtype=torch.bool)[:, None, None, :]
        scores = scores.masked_fill(~key_mask, -1.0e4)
        attention = torch.softmax(scores.float(), dim=-1).to(dtype=v.dtype)
        attention = F.dropout(attention, p=self.dropout, training=self.training)

        values = torch.einsum("bhij,bjhd->bihd", attention, v).reshape(batch_size, length, -1)
        gate = torch.sigmoid(self.linear_gate(single))
        output = self.linear_output(gate * values)
        return output * seq_mask[..., None].to(dtype=output.dtype)


class ReLUSquaredTransition(torch.nn.Module):
    """Zero-initialized residual transition using the ReLU-squared activation."""

    def __init__(self, channels: int, multiplier: int):
        super().__init__()
        self.norm = torch.nn.LayerNorm(channels)
        self.linear_up = torch.nn.Linear(channels, multiplier * channels)
        self.linear_down = torch.nn.Linear(multiplier * channels, channels)
        init_linear(self.linear_up, init="relu")
        init_linear(self.linear_down, init="final")

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        hidden = F.relu(self.linear_up(self.norm(value))).square()
        return self.linear_down(hidden)


class CompactPairformerBlock(torch.nn.Module):
    """Pairformer-style pair geometry updates followed by pair-biased single attention."""

    def __init__(
        self,
        config: ModelConfig,
        *,
        single_attention_heads: int,
        single_transition_multiplier: int,
        single_dropout: float,
    ):
        super().__init__()
        self.pair_dropout = float(config.evoformer_pair_dropout)
        self.single_dropout = float(single_dropout)
        self.triangle_mult_out = TriangleMultiplicationOutgoing(config)
        self.triangle_mult_in = TriangleMultiplicationIncoming(config)
        self.triangle_att_start = TriangleAttentionStartingNode(config)
        self.triangle_att_end = TriangleAttentionEndingNode(config)
        self.pair_transition = PairTransition(config)
        self.single_attention = PairBiasedSelfAttention(
            config,
            num_heads=single_attention_heads,
            dropout=single_dropout,
        )
        self.single_transition = ReLUSquaredTransition(
            config.c_s,
            multiplier=single_transition_multiplier,
        )

    def forward(
        self,
        single_representation: torch.Tensor,
        pair_representation: torch.Tensor,
        seq_mask: torch.Tensor,
        pair_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pair_representation = pair_representation + dropout_rowwise(
            self.triangle_mult_out(pair_representation, pair_mask=pair_mask),
            p=self.pair_dropout,
            training=self.training,
        )
        pair_representation = pair_representation + dropout_rowwise(
            self.triangle_mult_in(pair_representation, pair_mask=pair_mask),
            p=self.pair_dropout,
            training=self.training,
        )
        pair_representation = pair_representation + dropout_rowwise(
            self.triangle_att_start(pair_representation, pair_mask=pair_mask),
            p=self.pair_dropout,
            training=self.training,
        )
        pair_representation = pair_representation + dropout_columnwise(
            self.triangle_att_end(pair_representation, pair_mask=pair_mask),
            p=self.pair_dropout,
            training=self.training,
        )
        pair_representation = pair_representation + self.pair_transition(pair_representation)
        pair_representation = pair_representation * pair_mask[..., None].to(dtype=pair_representation.dtype)

        single_representation = single_representation + F.dropout(
            self.single_attention(single_representation, pair_representation, seq_mask),
            p=self.single_dropout,
            training=self.training,
        )
        single_representation = single_representation + F.dropout(
            self.single_transition(single_representation),
            p=self.single_dropout,
            training=self.training,
        )
        single_representation = single_representation * seq_mask[..., None].to(dtype=single_representation.dtype)
        return single_representation, pair_representation


class CompactPairformerIPA(torch.nn.Module):
    """Short MSA trunk, compact Pairformer trunk, and AF2 IPA atom14 head."""

    def __init__(self, config: ModelConfig, model_cfg: Dict[str, Any]):
        super().__init__()
        from minalphafold.embedders import InputEmbedder

        self.config = config
        self.activation_checkpointing = bool(model_cfg.get("activation_checkpointing", True))
        self.checkpoint_preserve_rng_state = bool(model_cfg.get("checkpoint_preserve_rng_state", True))
        self.input_embedder = InputEmbedder(config)
        self.msa_blocks = torch.nn.ModuleList([Evoformer(config) for _ in range(config.num_evoformer)])
        self.single_rep_proj = torch.nn.Linear(config.c_m, config.c_s)
        init_linear(self.single_rep_proj, init="default")
        self.pairformer_blocks = torch.nn.ModuleList(
            [
                CompactPairformerBlock(
                    config,
                    single_attention_heads=int(model_cfg["single_attention_heads"]),
                    single_transition_multiplier=int(model_cfg.get("single_transition_multiplier", 4)),
                    single_dropout=float(model_cfg.get("single_dropout", 0.0)),
                )
                for _ in range(int(model_cfg["pairformer_blocks"]))
            ]
        )
        self.structure_model = StructureModule(config)
        self.distogram_head = DistogramHead(config)
        self.plddt_head = PLDDTHead(config)
        self.masked_msa_head = MaskedMSAHead(config)
        self.tm_score_head = TMScoreHead(config)
        self.experimentally_resolved_head = ExperimentallyResolvedHead(config)

        for module in self.modules():
            if module.__class__.__name__ == "InvariantPointAttention" and hasattr(module, "head_weights"):
                cast(torch.nn.Parameter, module.head_weights).data.fill_(math.log(math.e - 1.0))

    def forward(
        self,
        target_feat: torch.Tensor,
        residue_index: torch.Tensor,
        msa_feat: torch.Tensor,
        extra_msa_feat: torch.Tensor,
        template_pair_feat: torch.Tensor,
        aatype: torch.Tensor,
        template_angle_feat: torch.Tensor | None = None,
        template_mask: torch.Tensor | None = None,
        template_residue_mask: torch.Tensor | None = None,
        seq_mask: torch.Tensor | None = None,
        msa_mask: torch.Tensor | None = None,
        extra_msa_mask: torch.Tensor | None = None,
        n_cycles: int = 1,
        n_ensemble: int = 1,
    ) -> Dict[str, Any]:
        del extra_msa_feat, template_pair_feat, template_angle_feat, template_mask
        del template_residue_mask, extra_msa_mask
        if n_cycles != 1:
            raise ValueError("compact_pairformer_ipa currently requires model.n_cycles=1.")
        if n_ensemble != 1:
            raise ValueError("compact_pairformer_ipa currently requires model.n_ensemble=1.")

        batch_size, length = aatype.shape
        if seq_mask is None:
            seq_mask = target_feat.new_ones(batch_size, length)
        if msa_mask is None:
            msa_mask = target_feat.new_ones(batch_size, msa_feat.shape[1], length)
        pair_mask = seq_mask[:, :, None] * seq_mask[:, None, :]

        msa_representation, pair_representation = self.input_embedder(target_feat, residue_index, msa_feat)
        for block in self.msa_blocks:
            if self.activation_checkpointing and self.training and torch.is_grad_enabled():
                msa_representation, pair_representation = cast(
                    tuple[torch.Tensor, torch.Tensor],
                    torch_checkpoint.checkpoint(
                        block,
                        msa_representation,
                        pair_representation,
                        msa_mask,
                        pair_mask,
                        use_reentrant=False,
                        preserve_rng_state=self.checkpoint_preserve_rng_state,
                    ),
                )
            else:
                msa_representation, pair_representation = block(
                    msa_representation,
                    pair_representation,
                    msa_mask=msa_mask,
                    pair_mask=pair_mask,
                )

        single_representation = self.single_rep_proj(msa_representation[:, 0])
        for block in self.pairformer_blocks:
            if self.activation_checkpointing and self.training and torch.is_grad_enabled():
                single_representation, pair_representation = cast(
                    tuple[torch.Tensor, torch.Tensor],
                    torch_checkpoint.checkpoint(
                        block,
                        single_representation,
                        pair_representation,
                        seq_mask,
                        pair_mask,
                        use_reentrant=False,
                        preserve_rng_state=self.checkpoint_preserve_rng_state,
                    ),
                )
            else:
                single_representation, pair_representation = block(
                    single_representation,
                    pair_representation,
                    seq_mask,
                    pair_mask,
                )

        structure_predictions = self.structure_model(
            single_representation,
            pair_representation,
            aatype,
            seq_mask=seq_mask,
        )
        return {
            **structure_predictions,
            "distogram_logits": self.distogram_head(pair_representation),
            "masked_msa_logits": self.masked_msa_head(msa_representation),
            "experimentally_resolved_logits": self.experimentally_resolved_head(single_representation),
            "plddt_logits": self.plddt_head(structure_predictions["single"]),
            "tm_logits": self.tm_score_head(pair_representation),
            "pair_representation": pair_representation,
            "msa_representation": msa_representation,
            "single_representation": single_representation,
            "sampled_n_cycles": 1,
            "sampled_n_ensemble": 1,
        }


def zeropower_via_newtonschulz5(gradient: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """Orthogonalize a matrix update with the quintic Muon iteration."""

    if gradient.ndim != 2:
        raise ValueError(f"Muon expects a matrix gradient, got shape {tuple(gradient.shape)}.")
    a, b, c = 3.4445, -4.7750, 2.0315
    use_transpose = gradient.shape[0] > gradient.shape[1]
    value = gradient.transpose(0, 1) if use_transpose else gradient
    work_dtype = torch.bfloat16 if gradient.device.type == "cuda" else torch.float32
    value = value.to(dtype=work_dtype)
    value = value / (value.norm() + 1.0e-7)
    for _ in range(int(steps)):
        square = value @ value.transpose(0, 1)
        value = a * value + (b * square + c * (square @ square)) @ value
    if use_transpose:
        value = value.transpose(0, 1)
    return value.to(dtype=gradient.dtype)


class MuonWithAuxAdam(torch.optim.Optimizer):
    """Single-device Muon for trunk matrices and Adam for all auxiliary parameters."""

    def __init__(self, param_groups: list[dict[str, Any]]):
        super().__init__(param_groups, {})

    @torch.no_grad()
    def step(self, closure: Any = None) -> Any:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if bool(group["use_muon"]):
                self._step_muon_group(group)
            else:
                self._step_adam_group(group)
        return loss

    def _step_muon_group(self, group: dict[str, Any]) -> None:
        momentum = float(group["momentum"])
        for param in group["params"]:
            if param.grad is None:
                continue
            if param.grad.ndim != 2:
                raise ValueError(f"Muon parameter must be 2D, got {tuple(param.grad.shape)}.")
            state = self.state[param]
            if "momentum_buffer" not in state:
                state["momentum_buffer"] = torch.zeros_like(param, dtype=torch.float32)
            gradient = param.grad.detach().float()
            buffer = state["momentum_buffer"]
            buffer.lerp_(gradient, 1.0 - momentum)
            update = gradient.lerp(buffer, momentum) if bool(group["nesterov"]) else buffer
            update = zeropower_via_newtonschulz5(update, steps=int(group["ns_steps"]))
            update = update * math.sqrt(max(1.0, update.shape[0] / update.shape[1]))
            learning_rate = float(group["lr"])
            weight_decay = float(group["weight_decay"])
            if weight_decay:
                param.mul_(1.0 - learning_rate * weight_decay)
            param.add_(update.to(dtype=param.dtype), alpha=-learning_rate)

    def _step_adam_group(self, group: dict[str, Any]) -> None:
        beta1, beta2 = cast(tuple[float, float], group["betas"])
        eps = float(group["eps"])
        for param in group["params"]:
            if param.grad is None:
                continue
            state = self.state[param]
            if "step" not in state:
                state["step"] = 0
                state["exp_avg"] = torch.zeros_like(param, dtype=torch.float32)
                state["exp_avg_sq"] = torch.zeros_like(param, dtype=torch.float32)
            state["step"] += 1
            gradient = param.grad.detach().float()
            exp_avg = state["exp_avg"]
            exp_avg_sq = state["exp_avg_sq"]
            exp_avg.lerp_(gradient, 1.0 - beta1)
            exp_avg_sq.lerp_(gradient.square(), 1.0 - beta2)
            step = int(state["step"])
            corrected_avg = exp_avg / (1.0 - beta1**step)
            corrected_sq = exp_avg_sq / (1.0 - beta2**step)
            update = corrected_avg / (corrected_sq.sqrt() + eps)
            learning_rate = float(group["lr"])
            weight_decay = float(group["weight_decay"])
            if weight_decay:
                param.mul_(1.0 - learning_rate * weight_decay)
            param.add_(update.to(dtype=param.dtype), alpha=-learning_rate)


class GroupwiseBudgetLRScheduler:
    def __init__(self, cfg: Dict[str, Any], optimizer: torch.optim.Optimizer):
        self.optimizer = optimizer
        self.base_lrs = [float(group["lr"]) for group in optimizer.param_groups]
        self.schedule = _af2_budget_schedule(cfg)
        self.completed_steps = 0
        self._apply_lr()

    def _scale(self) -> float:
        if self.schedule.warmup_steps > 0 and self.completed_steps < self.schedule.warmup_steps:
            scale = float(self.completed_steps) / float(self.schedule.warmup_steps)
        else:
            scale = 1.0
        if self.completed_steps >= self.schedule.lr_decay_step:
            scale *= self.schedule.lr_decay_factor
        if self.completed_steps >= self.schedule.finetune_start_step:
            scale *= self.schedule.finetune_lr_scale
        return scale

    def _apply_lr(self) -> None:
        scale = self._scale()
        for group, base_lr in zip(self.optimizer.param_groups, self.base_lrs, strict=True):
            group["lr"] = base_lr * scale

    def step(self) -> None:
        self.completed_steps += 1
        self._apply_lr()

    def state_dict(self) -> Dict[str, Any]:
        return {"completed_steps": self.completed_steps}

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.completed_steps = int(state.get("completed_steps", 0))
        self._apply_lr()


def _compact_model_config(cfg: Dict[str, Any]) -> ModelConfig:
    model_cfg = cfg["model"]
    base = load_model_config(MINALPHAFOLD2_ROOT / "configs" / "tiny.toml")
    return replace(
        base,
        model_profile="compact_pairformer_ipa",
        c_m=int(model_cfg["c_m"]),
        c_s=int(model_cfg["c_s"]),
        c_z=int(model_cfg["c_z"]),
        c_t=int(model_cfg.get("c_t", 32)),
        c_e=int(model_cfg.get("c_e", 32)),
        dim=int(model_cfg["msa_head_dim"]),
        num_heads=int(model_cfg["msa_heads"]),
        msa_transition_n=int(model_cfg.get("msa_transition_multiplier", 2)),
        outer_product_dim=int(model_cfg["outer_product_dim"]),
        triangle_mult_c=int(model_cfg["triangle_mult_c"]),
        triangle_dim=int(model_cfg["triangle_head_dim"]),
        triangle_num_heads=int(model_cfg["triangle_heads"]),
        pair_transition_n=int(model_cfg.get("pair_transition_multiplier", 2)),
        num_extra_msa=0,
        num_evoformer=int(model_cfg["msa_blocks"]),
        evoformer_msa_dropout=float(model_cfg.get("msa_dropout", 0.0)),
        evoformer_pair_dropout=float(model_cfg.get("pair_dropout", 0.0)),
        structure_module_c=int(model_cfg["structure_c"]),
        structure_module_layers=int(model_cfg["structure_layers"]),
        structure_module_dropout_ipa=float(model_cfg.get("structure_dropout", 0.0)),
        structure_module_dropout_transition=float(model_cfg.get("structure_dropout", 0.0)),
        sidechain_num_channel=int(model_cfg["structure_c"]),
        ipa_num_heads=int(model_cfg["ipa_heads"]),
        ipa_c=int(model_cfg["ipa_head_dim"]),
        ipa_n_query_points=int(model_cfg.get("ipa_query_points", 4)),
        ipa_n_value_points=int(model_cfg.get("ipa_value_points", 4)),
        plddt_hidden_dim=int(model_cfg["structure_c"]),
        zero_init=True,
    )


def build_model(cfg: Dict[str, Any]) -> torch.nn.Module:
    model = CompactPairformerIPA(_compact_model_config(cfg), cfg["model"])
    model.nanofold_initial_loss_fn = AlphaFoldLoss(finetune=False)
    model.nanofold_finetune_loss_fn = AlphaFoldLoss(finetune=True)
    _finetune_target_weights(model.nanofold_finetune_loss_fn)
    return model


def _muon_parameter_split(model: torch.nn.Module) -> tuple[list[torch.nn.Parameter], list[torch.nn.Parameter]]:
    muon_params: list[torch.nn.Parameter] = []
    auxiliary_params: list[torch.nn.Parameter] = []
    trunk_prefixes = ("msa_blocks.", "pairformer_blocks.")
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        is_hidden_matrix = (
            param.ndim == 2
            and name.startswith(trunk_prefixes)
            and "gate" not in name
        )
        if is_hidden_matrix:
            muon_params.append(param)
        else:
            auxiliary_params.append(param)
    return muon_params, auxiliary_params


def build_optimizer(cfg: Dict[str, Any], model: torch.nn.Module) -> torch.optim.Optimizer:
    optim_cfg = cfg["optim"]
    name = str(optim_cfg.get("name", "adam")).lower()
    if name == "adam":
        return torch.optim.Adam(
            model.parameters(),
            lr=float(optim_cfg["lr"]),
            betas=(float(optim_cfg.get("beta1", 0.9)), float(optim_cfg.get("beta2", 0.999))),
            eps=float(optim_cfg.get("eps", 1.0e-6)),
            weight_decay=float(optim_cfg.get("weight_decay", 0.0)),
        )
    if name != "muon":
        raise ValueError(f"Unsupported optim.name={name!r}; expected 'adam' or 'muon'.")

    muon_params, auxiliary_params = _muon_parameter_split(model)
    if not muon_params or not auxiliary_params:
        raise ValueError("Muon optimizer requires both trunk matrix and auxiliary parameter groups.")
    return MuonWithAuxAdam(
        [
            {
                "name": "muon_trunk_matrices",
                "params": muon_params,
                "use_muon": True,
                "lr": float(optim_cfg["muon_lr"]),
                "momentum": float(optim_cfg.get("muon_momentum", 0.95)),
                "nesterov": bool(optim_cfg.get("muon_nesterov", True)),
                "ns_steps": int(optim_cfg.get("muon_ns_steps", 5)),
                "weight_decay": float(optim_cfg.get("muon_weight_decay", 0.0)),
            },
            {
                "name": "auxiliary_adam",
                "params": auxiliary_params,
                "use_muon": False,
                "lr": float(optim_cfg.get("aux_lr", optim_cfg["lr"])),
                "betas": (
                    float(optim_cfg.get("aux_beta1", optim_cfg.get("beta1", 0.9))),
                    float(optim_cfg.get("aux_beta2", optim_cfg.get("beta2", 0.999))),
                ),
                "eps": float(optim_cfg.get("aux_eps", optim_cfg.get("eps", 1.0e-6))),
                "weight_decay": float(optim_cfg.get("aux_weight_decay", optim_cfg.get("weight_decay", 0.0))),
            },
        ]
    )


def build_scheduler(cfg: Dict[str, Any], optimizer: torch.optim.Optimizer) -> GroupwiseBudgetLRScheduler:
    return GroupwiseBudgetLRScheduler(cfg, optimizer)


def _float_tensors(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.float() if value.is_floating_point() else value
    if isinstance(value, dict):
        return {key: _float_tensors(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_float_tensors(item) for item in value)
    if isinstance(value, list):
        return [_float_tensors(item) for item in value]
    return value


def _forward_model(
    model: torch.nn.Module,
    features: Dict[str, torch.Tensor],
    cfg: Dict[str, Any],
) -> Dict[str, Any]:
    kwargs = {
        "target_feat": features["target_feat"],
        "residue_index": features["residue_index"],
        "msa_feat": features["msa_feat"],
        "extra_msa_feat": features["extra_msa_feat"],
        "template_pair_feat": features["template_pair_feat"],
        "aatype": features["aatype"],
        "template_angle_feat": features["template_angle_feat"],
        "template_mask": features["template_mask"],
        "template_residue_mask": features["template_residue_mask"],
        "seq_mask": features["seq_mask"],
        "msa_mask": features["msa_mask"],
        "extra_msa_mask": features["extra_msa_mask"],
        "n_cycles": int(cfg["model"].get("n_cycles", 1)),
        "n_ensemble": int(cfg["model"].get("n_ensemble", 1)),
    }
    use_bf16 = bool(cfg["model"].get("bf16", False)) and features["aatype"].device.type == "cuda"
    if use_bf16:
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            return cast(Dict[str, Any], model(**kwargs))
    return cast(Dict[str, Any], model(**kwargs))


def run_batch(
    model: torch.nn.Module,
    batch: Dict[str, torch.Tensor],
    cfg: Dict[str, Any],
    training: bool,
) -> Dict[str, torch.Tensor]:
    features = _build_minalphafold_inputs(batch, cfg, training=training)
    model_out = _forward_model(model, features, cfg)
    pred_atom14 = model_out["atom14_coords"].float() * batch["residue_mask"].to(
        device=model_out["atom14_coords"].device,
        dtype=torch.float32,
    )[:, :, None, None]

    has_supervision = "atom14_positions" in batch and "atom14_mask" in batch
    if not has_supervision:
        return {"pred_atom14": pred_atom14}

    loss = _alphafold_loss(model, features, _float_tensors(model_out), cfg)
    return {
        "pred_atom14": pred_atom14,
        "loss": loss,
        "alphafold_loss": loss.detach(),
    }
