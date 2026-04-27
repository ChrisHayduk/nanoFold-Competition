from __future__ import annotations

import json
from pathlib import Path

import yaml

from nanofold.competition_policy import load_track_spec, validate_config_against_track


def test_load_official_track() -> None:
    track = load_track_spec("limited")
    assert track.track_id == "limited"
    assert track.train_chain_count == 10000
    assert track.val_chain_count == 1000
    assert track.max_steps == 10000
    assert track.sample_budget == 20000
    assert track.residue_budget == 5120000
    assert track.official is True
    assert track.train_manifest_sha256 is not None
    assert track.val_manifest_sha256 is not None
    assert track.all_manifest_sha256 is not None


def test_official_config_matches_track() -> None:
    cfg = yaml.safe_load(Path("configs/official_baseline.yaml").read_text())
    track = load_track_spec("limited")
    errors = validate_config_against_track(
        cfg,
        track_spec=track,
        enforce_manifest_paths=True,
        enforce_manifest_hashes=True,
    )
    assert errors == []


def test_research_large_track_uses_same_data_with_larger_budget() -> None:
    limited = load_track_spec("limited")
    research = load_track_spec("research_large")

    assert research.track_id == "research_large"
    assert research.train_manifest_sha256 == limited.train_manifest_sha256
    assert research.val_manifest_sha256 == limited.val_manifest_sha256
    assert research.train_chain_count == limited.train_chain_count
    assert research.effective_batch_size == 2
    assert research.max_steps == 50000
    assert research.sample_budget == 100000
    assert research.fingerprint_path == "leaderboard/research_large_dataset_fingerprint.json"

    fingerprint = json.loads(Path(research.fingerprint_path).read_text())
    assert fingerprint["track_id"] == "research_large"


def test_unlimited_track_uses_same_data_without_budget_caps() -> None:
    limited = load_track_spec("limited")
    unlimited = load_track_spec("unlimited")

    assert unlimited.track_id == "unlimited"
    assert unlimited.train_manifest_sha256 == limited.train_manifest_sha256
    assert unlimited.val_manifest_sha256 == limited.val_manifest_sha256
    assert unlimited.sample_budget is None
    assert unlimited.residue_budget is None
    assert unlimited.max_params is None
    assert unlimited.rank_metric == "final_hidden_foldscore"
    assert unlimited.rank_tiebreak_metric is None
    assert unlimited.fingerprint_path == "leaderboard/unlimited_dataset_fingerprint.json"

    fingerprint = json.loads(Path(unlimited.fingerprint_path).read_text())
    assert fingerprint["track_id"] == "unlimited"


def test_track_validation_rejects_budget_drift() -> None:
    cfg = yaml.safe_load(Path("configs/official_baseline.yaml").read_text())
    cfg["data"]["crop_size"] = 128
    track = load_track_spec("limited")
    errors = validate_config_against_track(cfg, track_spec=track, enforce_manifest_paths=True)
    assert any("data.crop_size" in msg for msg in errors)


def test_track_validation_rejects_manifest_hash_mismatch(tmp_path: Path) -> None:
    cfg = yaml.safe_load(Path("configs/official_baseline.yaml").read_text())
    train_manifest = tmp_path / "data" / "manifests" / "train.txt"
    train_manifest.parent.mkdir(parents=True, exist_ok=True)
    train_manifest.write_text("wrong_chain_A\n")
    cfg["data"]["train_manifest"] = str(train_manifest)

    track = load_track_spec("limited")
    errors = validate_config_against_track(
        cfg,
        track_spec=track,
        enforce_manifest_paths=True,
        enforce_manifest_hashes=True,
    )
    assert any("Train manifest SHA256 mismatch" in msg for msg in errors)
