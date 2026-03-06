from __future__ import annotations

from train import resume_metadata_mismatches


def test_resume_metadata_mismatches_detects_all_key_differences() -> None:
    mismatches = resume_metadata_mismatches(
        ckpt_obj={
            "submission_entrypoint_sha256": "old-submission",
            "config_sha256": "old-config",
            "track_id": "old-track",
            "fingerprint_sha256": "old-fingerprint",
            "n_params": 123,
        },
        submission_entrypoint_sha256="new-submission",
        config_sha256="new-config",
        track_id="limited_large_v3",
        fingerprint_sha256="new-fingerprint",
        n_params=456,
    )

    assert any(item.startswith("submission_entrypoint_sha256:") for item in mismatches)
    assert any(item.startswith("config_sha256:") for item in mismatches)
    assert any(item.startswith("track_id:") for item in mismatches)
    assert any(item.startswith("fingerprint_sha256:") for item in mismatches)
    assert any(item.startswith("n_params:") for item in mismatches)


def test_resume_metadata_mismatches_accepts_legacy_track_field() -> None:
    mismatches = resume_metadata_mismatches(
        ckpt_obj={
            "submission_entrypoint_sha256": "same",
            "config_sha256": "same",
            "track": "limited_large_v3",
            "fingerprint_sha256": None,
            "n_params": 42,
        },
        submission_entrypoint_sha256="same",
        config_sha256="same",
        track_id="limited_large_v3",
        fingerprint_sha256=None,
        n_params=42,
    )

    assert mismatches == []
