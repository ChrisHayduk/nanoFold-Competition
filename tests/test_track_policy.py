from __future__ import annotations

from pathlib import Path

import yaml

from nanofold.competition_policy import load_track_spec, validate_config_against_track


def test_load_official_track() -> None:
    track = load_track_spec("limited_large_v3")
    assert track.track_id == "limited_large_v3"
    assert track.train_chain_count == 10000
    assert track.val_chain_count == 1000
    assert track.official is True


def test_official_config_matches_track() -> None:
    cfg = yaml.safe_load(Path("configs/limited_large_v3_official_baseline.yaml").read_text())
    track = load_track_spec("limited_large_v3")
    errors = validate_config_against_track(cfg, track_spec=track, enforce_manifest_paths=True)
    assert errors == []


def test_track_validation_rejects_budget_drift() -> None:
    cfg = yaml.safe_load(Path("configs/limited_large_v3_official_baseline.yaml").read_text())
    cfg["data"]["crop_size"] = 128
    track = load_track_spec("limited_large_v3")
    errors = validate_config_against_track(cfg, track_spec=track, enforce_manifest_paths=True)
    assert any("data.crop_size" in msg for msg in errors)
