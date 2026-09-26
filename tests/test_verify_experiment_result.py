from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import torch
import yaml

from nanofold.metrics import FOLDSCORE_COMPONENT_NAMES, FOLDSCORE_VERSION
from nanofold.utils import load_torch_checkpoint, sha256_file
from scripts.verify_experiment_result import verify_result


@pytest.fixture
def result_bundle(tmp_path: Path) -> dict[str, Any]:
    paths = {name: tmp_path / name for name in ("config", "fingerprint", "manifest", "checkpoint", "summary", "per_chain")}
    cfg = {"run_name": "test_seed1", "seed": 1, "train": {"max_steps": 2, "grad_accum_steps": 2},
           "data": {"batch_size": 1, "processed_features_dir": "data/features",
                    "processed_labels_dir": "data/labels", "val_manifest": "data/val.txt"}}
    paths["config"].write_text(yaml.safe_dump(cfg))
    paths["manifest"].write_text("1abc_A\n2def_A\n")
    paths["fingerprint"].write_text(json.dumps({"manifests": {"val": {
        "sha256": sha256_file(paths["manifest"]), "chain_count": 2,
    }}}))
    torch.save({"step": 2, "cumulative_samples_seen": 4, "sample_budget": 4, "config": cfg,
                "config_sha256": sha256_file(paths["config"]),
                "fingerprint_sha256": sha256_file(paths["fingerprint"])}, paths["checkpoint"])
    metrics = {name: 0.0 for name in FOLDSCORE_COMPONENT_NAMES}
    endpoint = {"ckpt": str(paths["checkpoint"]), "step": 2, "ckpt_sha256": sha256_file(paths["checkpoint"]),
                "num_chains": 2, "cumulative_samples_seen": 4,
                **{f"mean_{name}": value for name, value in metrics.items()}}
    paths["summary"].write_text(json.dumps({"split": "val", "official_mode": False,
                                          "foldscore_version": FOLDSCORE_VERSION,
                                          **endpoint, "checkpoints": [endpoint]}))
    paths["per_chain"].write_text("".join(json.dumps({"ckpt": str(paths["checkpoint"]), "step": 2,
                                                    "ckpt_sha256": sha256_file(paths["checkpoint"]),
                                                    "foldscore_version": FOLDSCORE_VERSION,
                                                    "chain_id": chain, **metrics}) + "\n"
                                           for chain in ("1abc_A", "2def_A")))
    return {**paths, "seed": 1, "final_step": 2, "sample_budget": 4}


def test_complete_zero_score_is_a_valid_endpoint(result_bundle: dict[str, Any]) -> None:
    result = verify_result(**result_bundle)
    assert result["status"] == "verified_final_public_endpoint"
    assert result["mean_foldscore"] == 0
    assert result["num_chains"] == 2


@pytest.mark.parametrize("defect", [None, "seed", "manifest", "batch_size", "empty_path", "missing_path"])
def test_runtime_staging_allows_only_data_directory_relocation(result_bundle: dict[str, Any], defect: str | None) -> None:
    checkpoint = result_bundle["checkpoint"]
    ckpt = load_torch_checkpoint(checkpoint)
    cfg = ckpt["config"]
    cfg["data"]["processed_features_dir"] = "/tmp/modal-stage/features"
    cfg["data"]["processed_labels_dir"] = "/tmp/modal-stage/labels"
    if defect == "seed":
        cfg["seed"] = 2
    elif defect == "manifest":
        cfg["data"]["val_manifest"] = "data/other_val.txt"
    elif defect == "batch_size":
        cfg["data"]["batch_size"] = 2
    elif defect == "empty_path":
        cfg["data"]["processed_labels_dir"] = ""
    elif defect == "missing_path":
        del cfg["data"]["processed_labels_dir"]
    torch.save(ckpt, checkpoint)
    checkpoint_hash = sha256_file(checkpoint)
    report = json.loads(result_bundle["summary"].read_text())
    report["ckpt_sha256"] = report["checkpoints"][0]["ckpt_sha256"] = checkpoint_hash
    result_bundle["summary"].write_text(json.dumps(report))
    rows = [json.loads(line) for line in result_bundle["per_chain"].read_text().splitlines()]
    for row in rows:
        row["ckpt_sha256"] = checkpoint_hash
    result_bundle["per_chain"].write_text("".join(json.dumps(row) + "\n" for row in rows))
    if defect is not None:
        with pytest.raises(ValueError, match="resolved config"):
            verify_result(**result_bundle)
    else:
        receipt = verify_result(**result_bundle)
        assert receipt["runtime_data_relocations"] == {
            "processed_features_dir": {"planned": "data/features", "resolved": "/tmp/modal-stage/features"},
            "processed_labels_dir": {"planned": "data/labels", "resolved": "/tmp/modal-stage/labels"},
        }


@pytest.mark.parametrize("defect", ["duplicate", "missing", "nonfinite", "wrong_checkpoint", "unbound_checkpoint", "old_row_version"])
def test_rejects_incomplete_or_mixed_per_target_records(result_bundle: dict[str, Any], defect: str) -> None:
    path = result_bundle["per_chain"]
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    if defect == "duplicate":
        rows[1]["chain_id"] = rows[0]["chain_id"]
    elif defect == "missing":
        rows.pop()
    elif defect == "nonfinite":
        rows[0]["foldscore"] = float("nan")
    elif defect == "wrong_checkpoint":
        rows[0]["ckpt"] = "another_experiment.pt"
    elif defect == "unbound_checkpoint":
        del rows[0]["ckpt_sha256"]
    else:
        rows[0]["foldscore_version"] = "old"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(ValueError):
        verify_result(**result_bundle)


@pytest.mark.parametrize("defect", ["stale_mean", "missing_denominator", "wrong_step", "old_metric", "official"])
def test_rejects_stale_or_incomplete_summary(result_bundle: dict[str, Any], defect: str) -> None:
    path = result_bundle["summary"]
    report = json.loads(path.read_text())
    if defect == "stale_mean":
        report["mean_foldscore"] = 0.8
    elif defect == "missing_denominator":
        del report["num_chains"]
    elif defect == "wrong_step":
        report["checkpoints"][0]["step"] = 1
    elif defect == "old_metric":
        report["foldscore_version"] = "old"
    else:
        report["official_mode"] = True
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError):
        verify_result(**result_bundle)


@pytest.mark.parametrize("field", ["fingerprint_sha256", "config_sha256", "cumulative_samples_seen"])
def test_rejects_other_data_or_budget_checkpoint(result_bundle: dict[str, Any], field: str) -> None:
    path = result_bundle["checkpoint"]
    ckpt = load_torch_checkpoint(path)
    ckpt[field] = 3 if field == "cumulative_samples_seen" else "0" * 64
    torch.save(ckpt, path)
    with pytest.raises(ValueError):
        verify_result(**result_bundle)


def test_rejects_composite_inconsistent_with_components_even_if_means_match(result_bundle: dict[str, Any]) -> None:
    rows_path = result_bundle["per_chain"]
    rows = [json.loads(line) for line in rows_path.read_text().splitlines()]
    for row in rows:
        row["foldscore"] = 0.8
    rows_path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    report_path = result_bundle["summary"]
    report = json.loads(report_path.read_text())
    report["mean_foldscore"] = report["checkpoints"][0]["mean_foldscore"] = 0.8
    report_path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="weighted components"):
        verify_result(**result_bundle)
