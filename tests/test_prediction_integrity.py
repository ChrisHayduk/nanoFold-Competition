from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import yaml

import eval as evaluate
import predict
import score
from nanofold.chain_paths import chain_npz_path
from nanofold.dataset_integrity import build_split_fingerprint
from nanofold.metrics import FOLDSCORE_COMPONENT_NAMES, FOLDSCORE_VERSION
from nanofold.prediction_contract import (
    load_prediction,
    prepare_prediction_directory,
    validate_checkpoint_for_prediction,
    validate_checkpoint_output_names,
)
from nanofold.rmsd import RMSD_CA_VERSION
from nanofold.scoring_io import load_feature_crop, load_label_crop, validate_scoring_alignment


def test_checkpoint_rejects_same_shape_semantic_config_and_code_changes() -> None:
    cfg = {"model": {"dropout": 0.1}, "submission": {"path": "training/submission.py", "recycles": 3}}
    checkpoint = {"config": copy.deepcopy(cfg), "submission_entrypoint_sha256": "a" * 64}
    cfg["submission"]["path"] = "relocated/submission.py"
    assert validate_checkpoint_for_prediction(checkpoint, cfg, submission_source_sha256="a" * 64, require_metadata=True)
    with pytest.raises(ValueError, match="source hash"):
        validate_checkpoint_for_prediction(checkpoint, cfg, submission_source_sha256="b" * 64, require_metadata=True)
    cfg["model"]["dropout"] = 0.5
    with pytest.raises(ValueError, match="model config"):
        validate_checkpoint_for_prediction(checkpoint, cfg, submission_source_sha256="a" * 64, require_metadata=True)
    with pytest.raises(ValueError, match="requires checkpoint config"):
        validate_checkpoint_for_prediction({}, cfg, submission_source_sha256="a" * 64, require_metadata=True)


def test_prediction_identity_and_floating_coordinates_are_verified(tmp_path: Path) -> None:
    path = tmp_path / "prediction.npz"
    arrays = {"pred_atom14": np.zeros((4, 14, 3), np.float32), "chain_id": "1abc_A", "ckpt_sha256": "a" * 64, "masked_length": 4}
    np.savez(path, **arrays)
    assert load_prediction(path, chain_id="1abc_A", checkpoint_sha256="a" * 64, require_binding=True).shape == (4, 14, 3)
    for cid, digest in (("2abc_A", "a" * 64), ("1abc_A", "b" * 64)):
        with pytest.raises(ValueError, match="scoring identity"):
            load_prediction(path, chain_id=cid, checkpoint_sha256=digest, require_binding=True)
    arrays["pred_atom14"] = np.zeros((4, 14, 3), np.int64)
    np.savez(path, **arrays)
    with pytest.raises(ValueError, match="floating point"):
        load_prediction(path, chain_id="1abc_A", checkpoint_sha256="a" * 64, require_binding=True)


def test_prediction_output_cannot_mix_checkpoints_or_overwrite_results(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="distinct checkpoint stems"):
        validate_checkpoint_output_names([tmp_path / "run_a/last.pt", tmp_path / "run_b/last.pt"])
    sentinel = tmp_path / "existing.npz"
    sentinel.write_bytes(b"previous prediction")
    with pytest.raises(FileExistsError, match="refusing to mix or overwrite"):
        prepare_prediction_directory(tmp_path)
    assert sentinel.read_bytes() == b"previous prediction"


def test_eval_uses_complete_manifest_denominator_and_separate_diagnostic_namespace() -> None:
    rows = [{"chain_id": cid, "rmsd_ca": None, "rmsd_ca_atom_count": 0, "rmsd_ca_status": "undefined_insufficient_ca", "rmsd_ca_version": RMSD_CA_VERSION, **{name: value for name in FOLDSCORE_COMPONENT_NAMES}} for cid, value in (("a", .25), ("b", .75))]
    result = evaluate._summarize_chain_metrics(rows, expected_chain_ids=["a", "b"], scored=True)
    assert result["mean_foldscore"] == .5
    assert result["rmsd_ca_valid_count"] == 0
    assert result["rmsd_ca_undefined_count"] == 2
    diagnostics = evaluate._summarize_submission_metrics({"foldscore": [(torch.tensor(99.), 2)]})
    assert "mean_foldscore" not in diagnostics
    assert diagnostics["mean_submission_foldscore"] == 99
    rows[1]["foldscore"] = float("nan")
    with pytest.raises(ValueError, match="Every manifest chain"):
        evaluate._summarize_chain_metrics(rows, expected_chain_ids=["a", "b"], scored=True)
    with pytest.raises(ValueError, match="exactly once"):
        evaluate._summarize_chain_metrics(rows[:1], expected_chain_ids=["a", "b"], scored=True)


def _fixture(tmp_path: Path, count: int = 3) -> tuple[dict, Path]:
    features = tmp_path / "features"
    labels = tmp_path / "labels"
    features.mkdir()
    labels.mkdir()
    chain_ids = [f"{idx + 1}abc_A" for idx in range(count)]
    manifest = tmp_path / "val.txt"
    manifest.write_text("\n".join(chain_ids) + "\n")
    for cid in chain_ids:
        np.savez(chain_npz_path(features, cid), chain_id=cid, aatype=np.zeros(4, np.int32), msa=np.zeros((2, 4), np.int32), deletions=np.zeros((2, 4), np.int32), residue_index=np.arange(4, dtype=np.int32))
        np.savez(chain_npz_path(labels, cid), chain_id=cid, ca_coords=np.zeros((4, 3), np.float32), ca_mask=np.ones(4, bool), atom14_positions=np.zeros((4, 14, 3), np.float32), atom14_mask=np.ones((4, 14), bool), residue_index=np.arange(4, dtype=np.int32))
    cfg = {"seed": 17, "submission": {"module": "nanofold.baseline_submission"}, "model": {}, "data": {"processed_features_dir": str(features), "processed_labels_dir": str(labels), "train_manifest": str(manifest), "val_manifest": str(manifest), "crop_size": 4, "msa_depth": 2, "batch_size": 2, "num_workers": 0}, "train": {"max_steps": 1}}
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump(cfg))
    return cfg, config


def _arguments(config: Path, checkpoint: Path, tmp_path: Path) -> argparse.Namespace:
    return argparse.Namespace(config=str(config), ckpt=str(checkpoint), ckpt_list="", ckpt_dir="", ckpt_steps="", split="val", track="limited", official=False, fingerprint="", verify_fingerprint=False, hidden_manifest="", score_labels_dir="", forbid_labels_dir="", allow_labels_mounted=False, pred_out_dir="", per_chain_out="", save=str(tmp_path / "summary.json"), submission_root="")


@pytest.mark.parametrize("entrypoint", [evaluate, predict])
def test_inference_refuses_missing_manifest_targets(tmp_path: Path, monkeypatch, entrypoint) -> None:
    cfg, config = _fixture(tmp_path)
    chain_npz_path(cfg["data"]["processed_features_dir"], "3abc_A").unlink()
    checkpoint = tmp_path / "checkpoint.pt"
    torch.save({"model": {}}, checkpoint)
    args = _arguments(config, checkpoint, tmp_path)
    monkeypatch.setattr(entrypoint, "parse_args", lambda: args)
    monkeypatch.setattr(entrypoint, "load_submission_hooks", lambda *a, **kw: SimpleNamespace())
    monkeypatch.setattr(entrypoint, "default_torch_device", lambda: torch.device("cpu"))
    with pytest.raises(FileNotFoundError, match="missing"):
        entrypoint.main()
    assert not Path(args.save).exists()


def test_eval_isolates_labels_and_replays_rng_per_checkpoint_with_complete_diagnostics(tmp_path: Path, monkeypatch) -> None:
    _cfg, config = _fixture(tmp_path)
    checkpoint = tmp_path / "checkpoint.pt"
    torch.save({"model": {}, "step": 0}, checkpoint)
    args = _arguments(config, checkpoint, tmp_path)
    second_checkpoint = tmp_path / "second.pt"
    torch.save({"model": {}, "step": 0}, second_checkpoint)
    args.ckpt_list = str(second_checkpoint)
    seen = []
    random_draws = []

    def run_batch(*, model, batch, cfg, training):
        assert not training
        assert not {"atom14_positions", "atom14_mask", "ca_coords", "ca_mask"}.intersection(batch)
        assert cfg["data"]["processed_labels_dir"] == ""
        seen.extend(batch["chain_id"])
        random_draws.append(float(torch.rand(())))
        return {"pred_atom14": torch.zeros((*batch["aatype"].shape, 14, 3)), "foldscore": torch.tensor(99.), "batch_size": torch.tensor(float(len(batch["chain_id"])))}

    hooks = SimpleNamespace(build_model=lambda cfg: torch.nn.Identity(), run_batch=run_batch, module_ref="fixture", source_sha256=None)
    monkeypatch.setattr(evaluate, "parse_args", lambda: args)
    monkeypatch.setattr(evaluate, "load_submission_hooks", lambda *a, **kw: hooks)
    monkeypatch.setattr(evaluate, "default_torch_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(evaluate, "_score_chain", lambda **kw: {**{name: .5 for name in FOLDSCORE_COMPONENT_NAMES}, "rmsd_ca": 0.0, "rmsd_ca_atom_count": 4, "rmsd_ca_status": "defined", "rmsd_ca_version": RMSD_CA_VERSION})
    evaluate.main()
    summary = json.loads(Path(args.save).read_text())
    assert seen == ["1abc_A", "2abc_A", "3abc_A"] * 2
    assert random_draws[:2] == random_draws[2:]
    assert summary["mean_foldscore"] == .5
    assert summary["mean_submission_foldscore"] == 99.
    assert summary["mean_submission_batch_size"] == pytest.approx(5 / 3)
    assert summary["foldscore_version"] == FOLDSCORE_VERSION
    assert summary["checkpoints"][0]["scored_chain_count"] == 3


def test_external_scoring_rejects_label_identity_and_alignment_mismatch(tmp_path: Path) -> None:
    cfg, _ = _fixture(tmp_path)
    labels = load_label_crop(labels_dir=Path(cfg["data"]["processed_labels_dir"]), chain_id="1abc_A", crop_size=2, crop_mode="center")
    features = load_feature_crop(features_dir=Path(cfg["data"]["processed_features_dir"]), chain_id="1abc_A", crop_size=2, crop_mode="center")
    features["residue_index"] += 1
    with pytest.raises(ValueError, match="residue_index disagree"):
        validate_scoring_alignment(labels, features)
    label_path = chain_npz_path(cfg["data"]["processed_labels_dir"], "1abc_A")
    with np.load(label_path) as source:
        arrays = {key: source[key] for key in source.files}
    arrays["chain_id"] = np.asarray("2abc_A")
    np.savez(label_path, **arrays)
    with pytest.raises(ValueError, match="chain_id"):
        load_label_crop(labels_dir=label_path.parent, chain_id="1abc_A", crop_size=2, crop_mode="center")


def test_saved_hidden_scoring_uses_bound_predictions_and_complete_curve(tmp_path: Path, monkeypatch) -> None:
    cfg, _ = _fixture(tmp_path, count=1)
    predictions = tmp_path / "predictions"
    checkpoints = []
    for step, value in ((0, "a"), (10, "b")):
        directory = predictions / f"ckpt_step_{step}"
        directory.mkdir(parents=True)
        digest = value * 64
        np.savez(chain_npz_path(directory, "1abc_A"), pred_atom14=np.zeros((4, 14, 3), np.float32), chain_id="1abc_A", ckpt_sha256=digest, masked_length=4)
        checkpoints.append({"ckpt": f"ckpt_step_{step}.pt", "ckpt_sha256": digest, "step": step, "cumulative_samples_seen": step})
    summary = tmp_path / "predict.json"
    summary.write_text(json.dumps({"mode": "predict", "prediction_format_version": 1, "official_mode": True, "track": "fixture", "split": "hidden_val", "checkpoints": checkpoints, "manifest_path": cfg["data"]["val_manifest"], "pred_out_dir": str(predictions), "crop_size": 4, "crop_mode": "center", "sample_budget": 10}))
    args = argparse.Namespace(prediction_summary=str(summary), manifest="", features_dir=cfg["data"]["processed_features_dir"], labels_dir=cfg["data"]["processed_labels_dir"], crop_size=-1, crop_mode="", official=False, device="cpu", per_chain_out=str(tmp_path / "scores.jsonl"), save=str(tmp_path / "scores.json"))
    monkeypatch.setattr(score, "parse_args", lambda: args)
    score.main()
    result = json.loads(Path(args.save).read_text())
    assert result["foldscore_version"] == FOLDSCORE_VERSION
    assert result["prediction_identity_verified"] is True
    assert result["official_mode"] is False  # A prediction summary cannot grant official scoring status.
    assert result["foldscore_auc_hidden"] == result["final_hidden_foldscore"]
    rows = [json.loads(line) for line in Path(args.per_chain_out).read_text().splitlines()]
    assert len(rows) == 2
    assert [row["ckpt_sha256"] for row in rows] == ["a" * 64, "b" * 64]
    malformed = json.loads(summary.read_text())
    malformed["sample_budget"] = 10.5
    summary.write_text(json.dumps(malformed))
    Path(args.per_chain_out).write_text("preserve prior output\n")
    with pytest.raises(ValueError, match="integer sample budget"):
        score.main()
    assert Path(args.per_chain_out).read_text() == "preserve prior output\n"
    malformed["sample_budget"] = 10
    summary.write_text(json.dumps(malformed))
    chain_npz_path(predictions / "ckpt_step_10", "1abc_A").unlink()
    with pytest.raises(FileNotFoundError, match="Missing prediction"):
        score.main()


@pytest.mark.parametrize("counts", [[0, 0, 10], [0, 1.5, 10], [0, True, 10], [0, 5, 9]])
def test_hidden_axis_does_not_truncate_or_repair_invalid_counters(counts) -> None:
    rows = [{"step": step, "cumulative_samples_seen": count, "mean_foldscore": .5} for step, count in zip((0, 5, 10), counts, strict=True)]
    with pytest.raises(ValueError):
        score._hidden_curve_metrics(rows, sample_budget=10)


@pytest.mark.parametrize("split_only", [False, True])
def test_research_prediction_and_scoring_verify_real_tensor_fingerprints(tmp_path: Path, monkeypatch, split_only: bool) -> None:
    cfg, config = _fixture(tmp_path, count=2)
    for chain in ("1abc_A", "2abc_A"):
        feature_path = chain_npz_path(cfg["data"]["processed_features_dir"], chain)
        with np.load(feature_path) as source:
            arrays = {key: source[key] for key in source.files}
        arrays.update(template_aatype=np.zeros((1, 4), np.int32), template_ca_coords=np.zeros((1, 4, 3), np.float32), template_ca_mask=np.ones((1, 4), bool))
        np.savez(feature_path, **arrays)
    train_manifest = tmp_path / "train.txt"
    train_manifest.write_text("1abc_A\n")
    Path(cfg["data"]["val_manifest"]).write_text("2abc_A\n")
    cfg["data"]["train_manifest"] = str(train_manifest)
    config.write_text(yaml.safe_dump(cfg))
    manifests = {"val": cfg["data"]["val_manifest"]}
    if not split_only:
        manifests = {"train": str(train_manifest), **manifests}
    fingerprint = build_split_fingerprint(
        processed_features_dir=cfg["data"]["processed_features_dir"],
        processed_labels_dir=cfg["data"]["processed_labels_dir"],
        manifest_paths=manifests, require_no_missing=True, require_labels=True, track_id="limited",
    )
    fingerprint_path = tmp_path / "fingerprint.json"
    fingerprint_path.write_text(json.dumps(fingerprint))
    # New-cohort evaluation must not need the training cohort to be mounted.
    if split_only:
        train_manifest.unlink()
        chain_npz_path(cfg["data"]["processed_features_dir"], "1abc_A").unlink()
    checkpoint = tmp_path / "checkpoint.pt"
    torch.save({"model": {}, "step": 0, "config": cfg}, checkpoint)
    args = _arguments(config, checkpoint, tmp_path)
    args.pred_out_dir = str(tmp_path / "predictions")
    args.verify_fingerprint = True
    args.fingerprint_split_only = split_only
    args.fingerprint = str(fingerprint_path)
    hooks = SimpleNamespace(
        build_model=lambda cfg: torch.nn.Identity(),
        run_batch=lambda **kw: {"pred_atom14": torch.zeros((*kw["batch"]["aatype"].shape, 14, 3))},
        module_ref="fixture", source_sha256=None, source_path=None,
    )
    monkeypatch.setattr(predict, "parse_args", lambda: args)
    monkeypatch.setattr(predict, "load_submission_hooks", lambda *a, **kw: hooks)
    monkeypatch.setattr(predict, "default_torch_device", lambda: torch.device("cpu"))
    predict.main()
    score_args = argparse.Namespace(
        prediction_summary=args.save, manifest=cfg["data"]["val_manifest"],
        features_dir=cfg["data"]["processed_features_dir"], labels_dir=cfg["data"]["processed_labels_dir"],
        crop_size=-1, crop_mode="", official=False, device="cpu", per_chain_out=str(tmp_path / "scores.jsonl"),
        save=str(tmp_path / "scores.json"), fingerprint=str(fingerprint_path), fingerprint_split_only=split_only,
        track="limited", split="val", config=str(config),
    )
    monkeypatch.setattr(score, "parse_args", lambda: score_args)
    score.main()
    assert json.loads(Path(score_args.save).read_text())["num_chains"] == 1
    original_output = Path(score_args.save).read_bytes()
    label_path = chain_npz_path(cfg["data"]["processed_labels_dir"], "2abc_A")
    with np.load(label_path) as source:
        arrays = {key: source[key] for key in source.files}
    arrays["atom14_positions"][0, 0, 0] += 1
    np.savez(label_path, **arrays)
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        score.main()
    assert Path(score_args.save).read_bytes() == original_output
