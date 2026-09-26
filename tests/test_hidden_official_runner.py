from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest
import yaml

from nanofold.chain_paths import chain_npz_path


def _load_run_official_module():
    module_path = Path("scripts/run_official.py").resolve()
    spec = importlib.util.spec_from_file_location("run_official_script", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module spec for {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_rank_metric_value_supports_auc_and_final_hidden() -> None:
    module = _load_run_official_module()
    metric_value = getattr(module, "_metric_value")
    hidden_results = {"foldscore_auc_hidden": 0.7, "final_hidden_foldscore": 0.9}
    eval_public_summary = {"mean_foldscore": 0.5}

    assert metric_value(
        "foldscore_auc_hidden",
        hidden_results=hidden_results,
        eval_public_summary=eval_public_summary,
    ) == 0.7
    assert metric_value(
        "final_hidden_foldscore",
        hidden_results=hidden_results,
        eval_public_summary=eval_public_summary,
    ) == 0.9
    assert metric_value(
        "public_val_foldscore",
        hidden_results=hidden_results,
        eval_public_summary=eval_public_summary,
    ) == 0.5


def test_hidden_lock_missing_or_incomplete_fails(tmp_path: Path) -> None:
    module = _load_run_official_module()
    validate_lock = getattr(module, "_validate_hidden_lock")

    hidden_manifest = tmp_path / "hidden_manifest.txt"
    hidden_manifest.write_text("1abc_A\n")
    hidden_features = tmp_path / "hidden_features"
    hidden_features.mkdir()
    hidden_labels = tmp_path / "hidden_labels"
    hidden_labels.mkdir()
    chain_npz_path(hidden_features, "1abc_A").write_bytes(b"features")
    chain_npz_path(hidden_labels, "1abc_A").write_bytes(b"labels")
    hidden_fingerprint = tmp_path / "hidden_fingerprint.json"
    hidden_fingerprint.write_text("{}")

    track = type(
        "Track",
        (),
        {"hidden_manifest_sha256": None, "hidden_fingerprint_sha256": None, "track_id": "limited", "hidden_chain_count": 1},
    )()

    with pytest.raises(FileNotFoundError):
        validate_lock(
            lock_path=tmp_path / "missing.lock.json",
            hidden_manifest=hidden_manifest,
            hidden_features_dir=hidden_features,
            hidden_labels_dir=hidden_labels,
            hidden_fingerprint=hidden_fingerprint,
            track_spec=track,
        )

    incomplete_lock = tmp_path / "incomplete.lock.json"
    incomplete_lock.write_text('{"hidden_manifest_sha256": null, "hidden_features_fingerprint_sha256": null, "hidden_labels_fingerprint_sha256": null, "hidden_fingerprint_sha256": null}')
    with pytest.raises(ValueError):
        validate_lock(
            lock_path=incomplete_lock,
            hidden_manifest=hidden_manifest,
            hidden_features_dir=hidden_features,
            hidden_labels_dir=hidden_labels,
            hidden_fingerprint=hidden_fingerprint,
            track_spec=track,
        )


def test_temp_predict_config_strips_label_path_and_resolves_submission_path(tmp_path: Path) -> None:
    module = _load_run_official_module()
    write_cfg = getattr(module, "_write_temp_predict_config")

    submission_dir = tmp_path / "submission"
    submission_dir.mkdir()
    source_cfg = submission_dir / "config.yaml"
    cfg = {
        "submission": {"path": "submission.py"},
        "data": {
            "processed_features_dir": "data/features",
            "processed_labels_dir": "data/labels",
        },
    }
    source_cfg.write_text(yaml.safe_dump(cfg))

    out_path = tmp_path / "predict.yaml"
    write_cfg(
        cfg=cfg,
        hidden_features_dir=tmp_path / "hidden_features",
        config_path=source_cfg,
        out_path=out_path,
    )
    saved = yaml.safe_load(out_path.read_text())
    assert saved["data"]["processed_labels_dir"] == ""
    assert Path(saved["submission"]["path"]).is_absolute()


def test_scrub_hidden_env_removes_hidden_asset_variables() -> None:
    module = _load_run_official_module()
    scrub = getattr(module, "_scrub_hidden_env")

    env = {
        "NANOFOLD_HIDDEN_MANIFEST": "/tmp/hidden_manifest.txt",
        "NANOFOLD_HIDDEN_LABELS_DIR": "/tmp/hidden_labels",
        "NANOFOLD_OFFICIAL_SEALED_RUNTIME": "1",
        "KEEP_ME": "yes",
    }
    out = scrub(stage="predict", env=env)
    assert "NANOFOLD_HIDDEN_MANIFEST" not in out
    assert "NANOFOLD_HIDDEN_LABELS_DIR" not in out
    assert out["KEEP_ME"] == "yes"
    assert out["NANOFOLD_OFFICIAL_RUNTIME_STAGE"] == "predict"


def test_hidden_runtime_requires_sealed_env() -> None:
    module = _load_run_official_module()
    require_runtime = getattr(module, "_require_sealed_hidden_runtime")

    with pytest.raises(ValueError, match="sealed runtime"):
        require_runtime(disable_hidden=False, env={})

    require_runtime(disable_hidden=False, env={"NANOFOLD_OFFICIAL_SEALED_RUNTIME": "1"})


def test_hidden_lock_validates_without_public_track_pins(tmp_path: Path) -> None:
    module = _load_run_official_module()
    validate_lock = getattr(module, "_validate_hidden_lock")

    hidden_manifest = tmp_path / "hidden_manifest.txt"
    hidden_manifest.write_text("1abc_A\n")
    hidden_features = tmp_path / "hidden_features"
    hidden_features.mkdir()
    hidden_labels = tmp_path / "hidden_labels"
    hidden_labels.mkdir()
    chain_npz_path(hidden_features, "1abc_A").write_bytes(b"features")
    chain_npz_path(hidden_labels, "1abc_A").write_bytes(b"labels")
    hidden_fingerprint = tmp_path / "hidden_fingerprint.json"
    hidden_fingerprint.write_text("{}")

    lock = tmp_path / "hidden.lock.json"
    lock.write_text(
        json.dumps(
            {
                "track": "limited",
                "hidden_chain_count": 1,
                "hidden_manifest_sha256": module._sha256(hidden_manifest),  # noqa: SLF001
                "hidden_features_fingerprint_sha256": module._tree_sha256(hidden_features),  # noqa: SLF001
                "hidden_labels_fingerprint_sha256": module._tree_sha256(hidden_labels),  # noqa: SLF001
                "hidden_fingerprint_sha256": module._sha256(hidden_fingerprint),  # noqa: SLF001
            }
        )
    )
    track = type("Track", (), {"hidden_manifest_sha256": None, "hidden_fingerprint_sha256": None, "track_id": "limited", "hidden_chain_count": 1})()

    meta = validate_lock(
        lock_path=lock,
        hidden_manifest=hidden_manifest,
        hidden_features_dir=hidden_features,
        hidden_labels_dir=hidden_labels,
        hidden_fingerprint=hidden_fingerprint,
        track_spec=track,
    )
    assert meta["status"] == "validated"


def test_official_runner_builds_predict_and_score_commands() -> None:
    module = _load_run_official_module()
    build_predict = getattr(module, "_build_predict_command")
    build_score = getattr(module, "_build_score_command")

    predict_cmd = build_predict(
        python="python",
        config_path=Path("/tmp/config.yaml"),
        split="hidden_val",
        track_id="limited",
        official=True,
        pred_out_dir=Path("/tmp/preds"),
        save_path=Path("/tmp/predict.json"),
        ckpt_dir=Path("/tmp/checkpoints"),
        ckpt_steps="0,1000,last",
        fingerprint=Path("/tmp/fingerprint.json"),
        hidden_manifest=Path("/tmp/hidden_manifest.txt"),
        forbid_labels_dir=Path("/tmp/forbid"),
    )
    score_cmd = build_score(
        python="python",
        prediction_summary=Path("/tmp/predict.json"),
        features_dir=Path("/tmp/features"),
        labels_dir=Path("/tmp/labels"),
        per_chain_out=Path("/tmp/per_chain.jsonl"),
        save_path=Path("/tmp/score.json"),
    )

    assert predict_cmd[1] == "predict.py"
    assert "--hidden-manifest" in predict_cmd
    assert score_cmd[1] == "score.py"
    assert "--features-dir" in score_cmd


@pytest.mark.parametrize("flags", [[], ["--skip-hidden-scoring", "--disable-hidden"], ["--score-hidden-only"]])
def test_hidden_prediction_only_cannot_mount_labels_or_disable_hidden(flags, monkeypatch) -> None:
    module = _load_run_official_module()
    monkeypatch.setattr(sys, "argv", ["run_official.py", "--submission", "unused", "--predict-hidden-only", *flags])
    with pytest.raises(ValueError, match="predict-hidden-only"):
        module.main()
