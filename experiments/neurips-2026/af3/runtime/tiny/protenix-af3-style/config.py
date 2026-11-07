"""Pinned Protenix-base configuration for NanoFold random-init training."""

from __future__ import annotations

import copy
from collections.abc import Mapping
from typing import Any


PROTENIX_COMMIT = "c3bfc365b3e1341a11935eddfe7bfdc308092147"


def _deep_update(destination: dict, update: Mapping[str, Any]) -> dict:
    for key, value in update.items():
        if isinstance(value, Mapping) and isinstance(destination.get(key), Mapping):
            _deep_update(destination[key], value)
        else:
            destination[key] = value
    return destination


def build_protenix_config() -> Any:
    """Return a ConfigDict for the pinned base architecture.

    Import happens lazily so adapter/checkpoint unit tests do not require a
    complete Protenix installation.
    """

    from ml_collections import ConfigDict
    from configs.configs_base import configs as base
    from configs.configs_data import data_configs
    from configs.configs_model_type import model_configs
    from protenix.config.config import ConfigManager

    raw = copy.deepcopy({**base, "data": data_configs})
    _deep_update(raw, model_configs["protenix_base_default_v1.0.0"])
    # ConfigManager resolves GlobalConfigValue/ListValue wrappers.
    raw.update(
        {
            "project": "nanofold-rebuttal",
            "run_name": "protenix-af3-style-random-init",
            "base_dir": "/workspace/runs/protenix-af3-style",
            "eval_interval": -1,
            "log_interval": 10,
            "checkpoint_interval": -1,
            "max_steps": 30_000,
        }
    )
    raw["model"]["N_cycle"] = 4
    raw["model"]["template_embedder"]["n_blocks"] = 0
    raw["esm"]["enable"] = False
    raw["diffusion_batch_size"] = 48
    raw["diffusion_chunk_size"] = 4
    raw["dtype"] = "bf16"
    raw["triangle_attention"] = "cuequivariance"
    raw["triangle_multiplicative"] = "cuequivariance"
    raw["enable_tf32"] = True
    raw["ema_decay"] = -1.0
    raw["use_wandb"] = False
    raw["seed"] = 20260725
    # Keep the executed optimizer/scheduler contract explicit here rather than
    # relying on inherited defaults. train.py constructs both objects through
    # the pinned Protenix helpers using these resolved values.
    raw["lr"] = 0.0018
    raw["lr_scheduler"] = "af3"
    raw["warmup_steps"] = 10
    raw["decay_every_n_steps"] = 50_000
    raw["grad_clip_norm"] = 10
    raw["adam"].update(
        {
            "beta1": 0.9,
            "beta2": 0.95,
            "weight_decay": 1e-8,
            "lr": 0.0018,
            "use_adamw": False,
        }
    )
    raw["af3_lr_scheduler"].update(
        {
            "warmup_steps": 10,
            "decay_every_n_steps": 50_000,
            "decay_factor": 0.95,
            "lr": 0.0018,
        }
    )
    raw["skip_amp"]["loss"] = True
    raw["deterministic"] = False
    raw["deterministic_seed"] = False
    raw["sample_diffusion"]["N_step_mini_rollout"] = 20
    raw["sample_diffusion"]["N_sample_mini_rollout"] = 1
    raw["loss"]["weight"]["alpha_confidence"] = 0.0
    raw["loss"]["weight"]["alpha_bond"] = 0.0
    raw["loss"]["weight"]["alpha_pae"] = 0.0
    # Additive scaling intervention: channel widths and block counts only.
    # All loss, noise, data, optimizer, recycle, and sampling settings above
    # remain the original NanoFold AF3-style settings.
    _deep_update(raw, {'c_s': 16, 'c_z': 4, 'c_atom': 4, 'c_atompair': 4, 'c_token': 4, 'c_s_inputs': 69, 'n_blocks': 1, 'model': {'input_embedder': {'c_atom': 4, 'c_atompair': 4, 'c_token': 4}, 'relative_position_encoding': {'c_z': 4}, 'template_embedder': {'c_z': 4}, 'msa_module': {'c_z': 4, 'c_s_inputs': 69, 'c_m': 4, 'n_blocks': 1}, 'pairformer': {'c_z': 4, 'c_s': 16, 'n_heads': 4, 'n_blocks': 1}, 'diffusion_module': {'c_atom': 4, 'c_atompair': 4, 'c_token': 16, 'c_s': 16, 'c_z': 4, 'c_s_inputs': 69, 'atom_encoder': {'n_blocks': 1}, 'transformer': {'n_blocks': 1, 'n_heads': 4}, 'atom_decoder': {'n_blocks': 1}}, 'confidence_head': {'c_z': 4, 'c_s': 16, 'c_s_inputs': 69, 'n_blocks': 1}, 'distogram_head': {'c_z': 4}, 'constraint_embedder': {'c_constraint_z': 4, 'c_constraint_s': 69, 'c_constraint_atom_pair': 4}}})
    # Outcome-blind operational layout, qualified on the exact 48-draw loss.
    raw["diffusion_chunk_size"] = 48
    raw["loss"]["diffusion_lddt_chunk_size"] = 48
    raw["model"]["diffusion_module"]["use_fine_grained_checkpoint"] = False
    manager = ConfigManager(raw, fill_required_with_null=True)
    return ConfigDict(manager.merge_configs({"blocks_per_ckpt": "None"}))
