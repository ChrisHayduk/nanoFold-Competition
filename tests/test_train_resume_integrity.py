from __future__ import annotations

import pytest
import torch

from train import _reject_future_checkpoints, _restore_scheduler_state, resume_metadata_mismatches


def test_resume_metadata_mismatches_detects_all_key_differences() -> None:
    mismatches = resume_metadata_mismatches(
        ckpt_obj={
            "submission_entrypoint_sha256": "checkpoint-submission",
            "config_sha256": "checkpoint-config",
            "track_id": "checkpoint-track",
            "fingerprint_sha256": "checkpoint-fingerprint",
            "n_params": 123,
        },
        submission_entrypoint_sha256="runtime-submission",
        config_sha256="runtime-config",
        track_id="limited",
        fingerprint_sha256="runtime-fingerprint",
        n_params=456,
    )

    assert any(item.startswith("submission_entrypoint_sha256:") for item in mismatches)
    assert any(item.startswith("config_sha256:") for item in mismatches)
    assert any(item.startswith("track_id:") for item in mismatches)
    assert any(item.startswith("fingerprint_sha256:") for item in mismatches)
    assert any(item.startswith("n_params:") for item in mismatches)


def test_resume_metadata_mismatches_requires_track_id() -> None:
    mismatches = resume_metadata_mismatches(
        ckpt_obj={
            "submission_entrypoint_sha256": "same",
            "config_sha256": "same",
            "track": "limited",
            "fingerprint_sha256": None,
            "n_params": 42,
        },
        submission_entrypoint_sha256="same",
        config_sha256="same",
        track_id="limited",
        fingerprint_sha256=None,
        n_params=42,
    )

    assert any(item.startswith("track_id:") for item in mismatches)


def test_resume_rejects_scheduler_that_cannot_restore_its_clock() -> None:
    class StepOnlyScheduler:
        updates = 0

        def step(self):
            self.updates += 1

    scheduler = StepOnlyScheduler()
    with pytest.raises(ValueError, match="scheduler implementing load_state_dict"):
        _restore_scheduler_state(scheduler, {"scheduler": {"updates": 4}})
    assert scheduler.updates == 0


@pytest.mark.parametrize("saved", [None, "older-runtime"])
def test_resume_detects_missing_or_changed_trainer_implementation(saved) -> None:
    mismatches = resume_metadata_mismatches(
        ckpt_obj={"submission_entrypoint_sha256": "same", "config_sha256": "same",
                  "track_id": "limited", "fingerprint_sha256": None, "n_params": 42,
                  "training_runtime_sha256": saved},
        submission_entrypoint_sha256="same", config_sha256="same", track_id="limited",
        fingerprint_sha256=None, n_params=42, training_runtime_sha256="current-runtime",
    )
    assert len(mismatches) == 1
    assert mismatches[0].startswith("training_runtime_sha256:")


@pytest.mark.parametrize("numbered_future", [False, True])
def test_older_resume_rejects_future_checkpoint_artifacts_without_modifying_evidence(tmp_path, numbered_future) -> None:
    old = tmp_path / "ckpt_step_3.pt"
    latest = tmp_path / "ckpt_last.pt"
    torch.save({"step": 3}, old)
    torch.save({"step": 7}, latest)
    if numbered_future:
        torch.save({"step": 7}, tmp_path / "ckpt_step_7.pt")
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    with pytest.raises(ValueError, match="newer checkpoints remain"):
        _reject_future_checkpoints(tmp_path, resumed_step=3, resume_path=old)
    assert {p.name: p.read_bytes() for p in tmp_path.iterdir()} == before
    _reject_future_checkpoints(tmp_path, resumed_step=7, resume_path=latest)
