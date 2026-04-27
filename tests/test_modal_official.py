from __future__ import annotations

from pathlib import Path


def _script_text() -> str:
    return Path("scripts/modal_official.py").read_text()


def test_modal_official_uses_separate_hidden_prediction_and_scoring_stages() -> None:
    text = _script_text()
    prediction_start = text.index("def run_prediction_stage")
    prediction_end = text.index("@app.function", prediction_start)
    scoring_start = text.index("def run_scoring_stage")
    scoring_decorator_start = text.rfind("@app.function", 0, scoring_start)
    local_start = text.index("def _write_local_result")
    prediction_block = text[prediction_start:prediction_end]
    scoring_block = text[scoring_decorator_start:local_start]

    assert "HIDDEN_LABELS_MOUNT" not in prediction_block
    assert "NANOFOLD_HIDDEN_LABELS_DIR" not in prediction_block
    assert "--skip-hidden-scoring" in prediction_block
    assert "HIDDEN_LABELS_MOUNT" in scoring_block
    assert "NANOFOLD_HIDDEN_LABELS_DIR" in scoring_block
    assert "--score-hidden-only" in scoring_block


def test_modal_official_blocks_network_in_remote_stages() -> None:
    text = _script_text()
    assert text.count("block_network=True") == 2
    assert text.count("gpu=GPU_SPEC") >= 2


def test_modal_official_keeps_hidden_assets_in_private_modal_volumes() -> None:
    text = _script_text()
    assert "nanofold-hidden-features" in text
    assert "nanofold-hidden-labels" in text
    assert "nanofold-hidden-meta" in text
    assert ".nanofold_private" in text
    assert 'hidden_fingerprint: str = "leaderboard/' not in text
    assert 'hidden_lock_file: str = "leaderboard/' not in text


def test_modal_official_updates_local_leaderboard_from_returned_result() -> None:
    text = _script_text()
    assert "modal_official_result.json" in text
    assert "scripts/add_leaderboard_entry.py" in text
    assert "--update-leaderboard" not in text
    assert "--commit" in text
    assert "--team" in text
    assert "team: str = \"\"" in text
    assert "resolve_leaderboard_team" in text
    assert "_resolve_local_commit" in text
    assert "upload_only: bool = False" in text
    assert "skip_predict: bool = False" in text
    assert "skip_score: bool = False" in text
