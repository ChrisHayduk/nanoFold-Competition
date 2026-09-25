from __future__ import annotations

import copy
import json
import sys
from dataclasses import replace
from pathlib import Path

import pytest
import torch
import yaml

import score
from nanofold.competition_policy import load_track_spec
from nanofold.data import collate_batch


def _contract(tmp_path: Path):
    manifest = tmp_path / "hidden.txt"
    manifest.write_text("1abc_A\n2abc_A\n")
    pred_root = tmp_path / "predictions"
    track = replace(load_track_spec(), official=False, max_steps=20, effective_batch_size=2, crop_size=8, hidden_chain_count=2)
    summary = {
        "mode": "predict", "official_mode": True, "track": track.track_id,
        "split": "hidden_val", "crop_size": 8, "crop_mode": "center",
        "effective_batch_size": 2, "sample_budget": 40, "residue_budget": 320,
        "manifest_path": str(manifest), "pred_out_dir": str(pred_root),
        "checkpoints": [],
    }
    for step, name in [(0, "ckpt_step_0"), (10, "ckpt_step_10"), (20, "ckpt_last")]:
        summary["checkpoints"].append({
            "step": step, "ckpt": f"/runs/checkpoints/{name}.pt", "num_chains": 2,
            "prediction_dir": str(pred_root / name), "cumulative_samples_seen": step * 2,
            "cumulative_cropped_residues_seen": step * 16,
            "cumulative_nonpad_residues_seen": step * 12,
        })
    return track, manifest, pred_root, summary


def _validate(tmp_path: Path, mutate=None):
    track, manifest, pred_root, summary = _contract(tmp_path)
    if mutate:
        mutate(summary)
    score._validate_official_summary(
        summary, track=track, manifest_path=manifest, pred_root=pred_root,
        split="hidden_val", checkpoint_steps="0,10,last",
    )


def test_valid_official_contract(tmp_path: Path) -> None:
    _validate(tmp_path)


@pytest.mark.parametrize("key,value", [
    ("manifest_path", "/tmp/subset.txt"), ("pred_out_dir", "/tmp/elsewhere"),
    ("crop_size", 1), ("crop_mode", "random"), ("sample_budget", 20),
    ("residue_budget", 160), ("effective_batch_size", 1), ("split", "val"),
    ("track", "custom"), ("official_mode", 1),
])
def test_official_summary_cannot_select_easier_contract(tmp_path: Path, key, value) -> None:
    with pytest.raises(ValueError):
        _validate(tmp_path, lambda summary: summary.update({key: value}))


@pytest.mark.parametrize("key,value", [
    ("step", 9), ("num_chains", 1), ("cumulative_samples_seen", 19),
    ("cumulative_cropped_residues_seen", 161), ("cumulative_nonpad_residues_seen", -1),
    ("ckpt", "/tmp/final.pt"), ("prediction_dir", "/tmp/elsewhere"),
])
def test_official_checkpoint_cannot_forge_counters(tmp_path: Path, key, value) -> None:
    with pytest.raises(ValueError):
        _validate(tmp_path, lambda summary: summary["checkpoints"][1].update({key: value}))


def test_official_checkpoint_cannot_omit_intermediate_milestone(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="schedule"):
        _validate(tmp_path, lambda summary: summary["checkpoints"].pop(1))


def test_competition_schedule_cannot_select_favorable_milestones(tmp_path: Path) -> None:
    track, manifest, pred_root, summary = _contract(tmp_path)
    track = replace(track, official=True, max_steps=30000)
    summary.update(sample_budget=60000, residue_budget=480000)
    with pytest.raises(ValueError, match="fixed competition checkpoint milestones"):
        score._validate_official_summary(
            summary, track=track, manifest_path=manifest, pred_root=pred_root,
            split="hidden_val", checkpoint_steps="0,last",
        )


def test_official_counts_accept_real_short_chain_batch(tmp_path: Path) -> None:
    track, manifest, pred_root, summary = _contract(tmp_path)
    examples = []
    for length in (3, 5):
        examples.append({
            "chain_id": f"chain{length}", "aatype": torch.zeros(length, dtype=torch.long),
            "msa": torch.zeros((1, length), dtype=torch.long), "deletions": torch.zeros((1, length)),
            "template_aatype": torch.zeros((0, length), dtype=torch.long),
            "template_ca_coords": torch.zeros((0, length, 3)),
            "template_ca_mask": torch.zeros((0, length), dtype=torch.bool),
        })
    batch = collate_batch(examples, crop_size=8, msa_depth=1, crop_mode="center", msa_sample_mode="top")
    batch_size, padded_length = batch["aatype"].shape
    for row in summary["checkpoints"]:
        row["cumulative_cropped_residues_seen"] = row["step"] * batch_size * padded_length
        row["cumulative_nonpad_residues_seen"] = row["step"] * int(batch["residue_mask"].sum())
    score._validate_official_summary(
        summary, track=track, manifest_path=manifest, pred_root=pred_root,
        split="hidden_val", checkpoint_steps="0,10,last",
    )


def test_unlimited_final_score_uses_trusted_config_not_prediction_metadata(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump({"data": {"batch_size": 2, "crop_size": 8}, "train": {"max_steps": 20}}))
    track = score._resolve_scoring_track(load_track_spec("unlimited"), str(config_path))
    assert (track.max_steps, track.crop_size, track.effective_batch_size) == (20, 8, 8)
    track = replace(track, hidden_chain_count=2)
    _, manifest, pred_root, summary = _contract(tmp_path)
    summary.update(track="unlimited", effective_batch_size=8, sample_budget=160, residue_budget=1280)
    row = summary["checkpoints"][-1]
    row.update(cumulative_samples_seen=160, cumulative_cropped_residues_seen=1280,
               cumulative_nonpad_residues_seen=960, prediction_dir=str(pred_root))
    summary["checkpoints"] = [row]
    score._validate_official_summary(
        summary, track=track, manifest_path=manifest, pred_root=pred_root,
        split="hidden_val", checkpoint_steps="last",
    )
    row["mean_foldscore"] = 0.75
    metrics = score._hidden_curve_metrics([row], sample_budget=160, compute_auc=False)
    assert metrics["final_hidden_foldscore"] == 0.75
    assert torch.isnan(torch.tensor(metrics["foldscore_auc_hidden"]))
    with pytest.raises(ValueError, match="trusted config"):
        score._resolve_scoring_track(load_track_spec("unlimited"), "")


def test_score_entrypoint_rejects_subset_before_reading_labels(tmp_path: Path, monkeypatch) -> None:
    track, manifest, pred_root, summary = _contract(tmp_path)
    subset = tmp_path / "subset.txt"
    subset.write_text("1abc_A\n")
    tampered = copy.deepcopy(summary)
    tampered["manifest_path"] = str(subset)
    summary_path = tmp_path / "predict.json"
    summary_path.write_text(json.dumps(tampered))
    monkeypatch.setattr(score, "load_track_spec", lambda _: track)
    monkeypatch.setattr(sys, "argv", [
        "score.py", "--official", "--track", track.track_id, "--split", "hidden_val",
        "--manifest", str(manifest), "--pred-out-dir", str(pred_root),
        "--checkpoint-steps", "0,10,last", "--prediction-summary", str(summary_path),
        "--features-dir", str(tmp_path / "absent-features"),
        "--labels-dir", str(tmp_path / "absent-labels"),
    ])
    with pytest.raises(ValueError, match="trusted manifest"):
        score.main()
