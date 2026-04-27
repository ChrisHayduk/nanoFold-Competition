from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from nanofold.competition_policy import (
    apply_track_policy,
    enforce_model_param_limit,
    load_track_spec,
    validate_track_policy,
)


def _load_official_cfg() -> dict:
    cfg = yaml.safe_load(Path("configs/official_baseline.yaml").read_text())
    assert isinstance(cfg, dict)
    return cfg


def test_apply_track_policy_overrides_immutable_fields() -> None:
    track = load_track_spec("limited")
    cfg = _load_official_cfg()
    cfg["seed"] = 123
    cfg["data"]["crop_size"] = 999
    cfg["data"]["msa_depth"] = 64
    cfg["train"]["max_steps"] = 42
    cfg["data"]["val_crop_mode"] = "random"
    cfg["data"]["val_msa_sample_mode"] = "random"
    cfg["data"]["batch_size"] = 7
    cfg["train"]["grad_accum_steps"] = 9

    out = apply_track_policy(cfg, track_spec=track)
    assert out["seed"] == track.seed
    assert out["data"]["crop_size"] == track.crop_size
    assert out["data"]["msa_depth"] == track.msa_depth
    assert out["train"]["max_steps"] == track.max_steps
    assert out["data"]["val_crop_mode"] == track.val_crop_mode
    assert out["data"]["val_msa_sample_mode"] == track.val_msa_sample_mode
    assert out["data"]["batch_size"] * out["train"]["grad_accum_steps"] == track.effective_batch_size


def test_validate_track_policy_passes_after_apply() -> None:
    track = load_track_spec("limited")
    cfg = _load_official_cfg()
    cfg["data"]["crop_size"] = 128
    applied = apply_track_policy(cfg, track_spec=track)
    errors = validate_track_policy(
        applied,
        track_spec=track,
        enforce_manifest_paths=True,
        enforce_manifest_hashes=True,
    )
    assert errors == []


def test_model_param_limit_is_enforced() -> None:
    track = load_track_spec("limited")
    assert track.max_params is not None
    enforce_model_param_limit(track_spec=track, n_params=int(track.max_params))
    with pytest.raises(ValueError):
        enforce_model_param_limit(track_spec=track, n_params=int(track.max_params) + 1)
