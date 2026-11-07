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
    manager = ConfigManager(raw, fill_required_with_null=True)
    return ConfigDict(manager.merge_configs({}))


# Qualified execution-only layout: original architecture and training recipe.
_original_build_protenix_config = build_protenix_config

def build_protenix_config():
    config = _original_build_protenix_config()
    config.diffusion_chunk_size = 48
    config.loss.diffusion_lddt_chunk_size = 16
    return config
