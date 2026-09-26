from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

import eval as evaluate
import score
from nanofold.chain_paths import chain_npz_path
from nanofold.experiment_reporting import validate_run_contract
from nanofold.prediction_contract import PREDICTION_FORMAT_VERSION
from nanofold.residue_constants import CA_ATOM14_SLOT
from nanofold.rmsd import masked_kabsch_rmsd, rmsd_ca_diagnostic, summarize_rmsd_ca
from nanofold.utils import sha256_file


def points() -> torch.Tensor:
    return torch.tensor([[0., 0., 0.], [1., 0., 0.], [0., 2., 0.], [0., 0., 3.]])


def test_coordinate_rmsd_preserves_frame_invariance_and_deformation() -> None:
    true = points()
    rotation = torch.tensor([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    pred = true @ rotation + torch.tensor([12., -8., 2.])
    assert masked_kabsch_rmsd(pred, true, torch.ones(4, dtype=torch.bool)) < 1.e-5
    # A collapsed prediction remains a defined, bad structure, rather than being
    # dropped because its alignment rotation is nonunique.
    collapsed = masked_kabsch_rmsd(torch.zeros_like(true), true, torch.ones(4, dtype=torch.bool))
    assert collapsed == pytest.approx(float(((true - true.mean(0)).square().sum(1).mean()).sqrt()), abs=1.e-6)
    assert masked_kabsch_rmsd(true * 2, true, torch.ones(4, dtype=torch.bool)) > 1.0


@pytest.mark.parametrize("count", [0, 1, 2])
def test_insufficient_observations_are_explicitly_undefined(count: int) -> None:
    observed = torch.arange(4) < count
    diagnostic = rmsd_ca_diagnostic(points(), points(), observed)
    assert diagnostic["rmsd_ca"] is None
    assert diagnostic["rmsd_ca_atom_count"] == count
    assert diagnostic["rmsd_ca_status"] == "undefined_insufficient_ca"
    summary = summarize_rmsd_ca([diagnostic], expected_count=1)
    assert summary["mean_rmsd_ca"] is None
    assert summary["rmsd_ca_undefined_count"] == 1


def test_partial_cohort_never_becomes_a_favorable_finite_only_mean() -> None:
    valid = rmsd_ca_diagnostic(points(), points(), torch.ones(4, dtype=torch.bool))
    undefined = rmsd_ca_diagnostic(points(), points(), torch.tensor([True, False, False, False]))
    summary = summarize_rmsd_ca([valid, undefined], expected_count=2)
    assert summary["mean_rmsd_ca"] is None
    assert summary["rmsd_ca_valid_count"] == summary["rmsd_ca_undefined_count"] == 1
    with pytest.raises(ValueError, match="complete nonempty"):
        summarize_rmsd_ca([valid], expected_count=2)
    with pytest.raises(ValueError, match="Missing"):
        summarize_rmsd_ca([{}], expected_count=1)
    undefined["rmsd_ca"] = 0.0
    with pytest.raises(ValueError, match="explicitly undefined"):
        summarize_rmsd_ca([undefined], expected_count=1)


def pin(path: Path) -> dict:
    return {"path": str(path), "sha256": sha256_file(path)}


@pytest.mark.parametrize("undefined", [False, True])
def test_saved_coordinate_producer_to_canonical_consumer(tmp_path: Path, monkeypatch, undefined: bool) -> None:
    features, labels, predictions = (tmp_path / name for name in ("features", "labels", "predictions"))
    for folder in (features, labels, predictions):
        folder.mkdir()
    manifest = tmp_path / "val.txt"
    manifest.write_text("a_A\nb_B\n")
    cfg = {"run_name": "test", "seed": 2, "data": {"batch_size": 1},
           "train": {"max_steps": 1, "grad_accum_steps": 1}}
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump(cfg))
    fingerprint = tmp_path / "fingerprint.json"
    fingerprint.write_text(json.dumps({"manifests": {"val": {"sha256": sha256_file(manifest), "chain_count": 2}}}))
    checkpoint = tmp_path / "ckpt_step_1.pt"
    torch.save({"step": 1, "cumulative_samples_seen": 1, "sample_budget": 1, "config": cfg,
                "config_sha256": sha256_file(config), "fingerprint_sha256": sha256_file(fingerprint)}, checkpoint)
    digest = sha256_file(checkpoint)
    for chain in ("a_A", "b_B"):
        coords = torch.zeros((4, 14, 3))
        coords[:, CA_ATOM14_SLOT] = points()
        mask = torch.zeros((4, 14), dtype=torch.bool)
        mask[:2 if undefined and chain == "b_B" else 4, CA_ATOM14_SLOT] = True
        np.savez(chain_npz_path(features, chain), chain_id=chain, aatype=np.zeros(4, np.int64), residue_index=np.arange(4))
        np.savez(chain_npz_path(labels, chain), chain_id=chain, atom14_positions=coords.numpy(), atom14_mask=mask.numpy(),
                 ca_coords=points().numpy(), ca_mask=mask[:, CA_ATOM14_SLOT].numpy(), residue_index=np.arange(4))
        pred = coords * (2.0 if chain == "b_B" else 1.0) + 8.0
        np.savez(chain_npz_path(predictions, chain), pred_atom14=pred.numpy(), chain_id=chain, ckpt_sha256=digest, masked_length=4)
        label_tensors = {"atom14_positions": coords, "atom14_mask": mask, "ca_coords": points(),
                         "ca_mask": mask[:, CA_ATOM14_SLOT], "residue_index": torch.arange(4)}
        feature_tensors = {"aatype": torch.zeros(4, dtype=torch.long), "residue_index": torch.arange(4)}
        eval_metrics = evaluate._score_chain(pred_atom14=pred, labels=label_tensors, features=feature_tensors)
        score_metrics = score._score_chain(pred_atom14=pred, labels=label_tensors, features=feature_tensors)
        assert eval_metrics == score_metrics
    prediction_summary = tmp_path / "prediction.json"
    checkpoint_row = {"ckpt": str(checkpoint), "ckpt_sha256": digest, "step": 1, "cumulative_samples_seen": 1}
    prediction_summary.write_text(json.dumps({"mode": "predict", "prediction_format_version": PREDICTION_FORMAT_VERSION,
        "split": "val", "track": "fixture", "manifest_path": str(manifest), "pred_out_dir": str(predictions),
        "crop_size": 4, "crop_mode": "center", "sample_budget": 1, "checkpoints": [checkpoint_row]}))
    summary, per_chain = tmp_path / "summary.json", tmp_path / "per_chain.jsonl"
    monkeypatch.setattr(score, "parse_args", lambda: argparse.Namespace(prediction_summary=str(prediction_summary),
        manifest="", features_dir=str(features), labels_dir=str(labels), crop_size=-1, crop_mode="", official=False,
        device="cpu", per_chain_out=str(per_chain), save=str(summary)))
    score.main()
    contract = {"schema_version": 1, "campaign_id": "fixture", "run_name": "test", "seed": 2,
        "final_step": 1, "sample_budget": 1, "metrics": ["lddt_ca", "rmsd_ca"],
        "checkpoints": [{**pin(checkpoint), "step": 1, "cumulative_samples_seen": 1}],
        "inputs": {name: pin(path) for name, path in {"config": config, "fingerprint": fingerprint,
            "manifest": manifest, "summary": summary, "per_chain": per_chain, "evaluator": Path(score.__file__)}.items()}}
    contract_path = tmp_path / "run_contract.json"
    contract_path.write_text(json.dumps(contract))
    receipt = validate_run_contract(contract_path)
    result = receipt["checkpoints"][0]
    assert result["rmsd_ca_valid_count"] == (1 if undefined else 2)
    assert result["rmsd_ca_undefined_count"] == (1 if undefined else 0)
    if undefined:
        assert result["mean_rmsd_ca"] is None
        assert result["rmsd_ca_status"] == "undefined_targets"
    else:
        assert result["mean_rmsd_ca"] > 0.5
        assert result["rmsd_ca_status"] == "complete"
    original = json.loads(summary.read_text())
    for change in ({"mean_rmsd_ca": 0.0}, {"rmsd_ca_valid_count": 99}, {"rmsd_ca_status": "complete" if undefined else "undefined_targets"}):
        changed = copy.deepcopy(original)
        changed["checkpoints"][0].update(change)
        summary.write_text(json.dumps(changed))
        contract["inputs"]["summary"] = pin(summary)
        contract_path.write_text(json.dumps(contract))
        with pytest.raises(ValueError, match="RMSD summary"):
            validate_run_contract(contract_path)
