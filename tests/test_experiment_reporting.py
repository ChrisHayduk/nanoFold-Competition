from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
import yaml

from nanofold.experiment_reporting import validate_run_contract
from nanofold.metrics import FOLDSCORE_COMPONENT_NAMES, FOLDSCORE_VERSION
from nanofold.utils import load_torch_checkpoint, sha256_file


def pin(path: Path) -> dict:
    return {"path": str(path), "sha256": sha256_file(path)}


@pytest.fixture
def bundle(tmp_path: Path) -> Path:
    config, fingerprint, manifest = (tmp_path / name for name in ("config.yaml", "fingerprint.json", "val.txt"))
    cfg = {
        "run_name": "test",
        "seed": 1,
        "data": {"batch_size": 2},
        "train": {"max_steps": 30000, "grad_accum_steps": 1},
    }
    config.write_text(yaml.safe_dump(cfg))
    manifest.write_text("a_A\nb_B\n")
    fingerprint.write_text(json.dumps({"manifests": {"val": {"sha256": sha256_file(manifest), "chain_count": 2}}}))
    metrics = dict.fromkeys(FOLDSCORE_COMPONENT_NAMES, 0.0)
    metrics["lddt_ca"] = 0.25
    schedule, checkpoints, records = [], [], []
    for step in (0, 24000):
        checkpoint = tmp_path / f"step{step}.pt"
        torch.save(
            {
                "step": step,
                "cumulative_samples_seen": step * 2,
                "sample_budget": 60000,
                "config": cfg,
                "config_sha256": sha256_file(config),
                "fingerprint_sha256": sha256_file(fingerprint),
            },
            checkpoint,
        )
        schedule.append({**pin(checkpoint), "step": step, "cumulative_samples_seen": step * 2})
        checkpoint_row = {
            "step": step,
            "ckpt": str(checkpoint),
            "ckpt_sha256": sha256_file(checkpoint),
            "num_chains": 2,
            "cumulative_samples_seen": step * 2,
            **{f"mean_{name}": value for name, value in metrics.items()},
        }
        checkpoints.append(checkpoint_row)
        records.extend(
            {
                "step": step,
                "ckpt": str(checkpoint),
                "ckpt_sha256": sha256_file(checkpoint),
                "foldscore_version": FOLDSCORE_VERSION,
                "chain_id": chain,
                **metrics,
            }
            for chain in ("a_A", "b_B")
        )
    summary, per_chain, evaluator = (tmp_path / name for name in ("summary.json", "per_chain.jsonl", "eval.py"))
    evaluator.write_text("# synthetic fixture evaluator\n")
    summary.write_text(
        json.dumps(
            {
                "split": "val",
                "official_mode": False,
                "foldscore_version": FOLDSCORE_VERSION,
                **checkpoints[-1],
                "checkpoints": checkpoints,
            }
        )
    )
    per_chain.write_text("".join(json.dumps(row) + "\n" for row in records))
    contract = {
        "schema_version": 1,
        "campaign_id": "fixture",
        "run_name": "test",
        "seed": 1,
        "final_step": 30000,
        "sample_budget": 60000,
        "metrics": ["lddt_ca"],
        "checkpoints": schedule,
        "inputs": {
            name: pin(path)
            for name, path in {
                "config": config,
                "fingerprint": fingerprint,
                "manifest": manifest,
                "summary": summary,
                "per_chain": per_chain,
                "evaluator": evaluator,
            }.items()
        },
    }
    path = tmp_path / "run_contract.json"
    path.write_text(json.dumps(contract))
    return path


def alter(bundle: Path, kind: str, mutate) -> None:
    contract = json.loads(bundle.read_text())
    path = Path(contract["inputs"][kind]["path"])
    if kind == "per_chain":
        value = [json.loads(line) for line in path.read_text().splitlines()]
        mutate(value)
        path.write_text("".join(json.dumps(row) + "\n" for row in value))
    else:
        value = json.loads(path.read_text())
        mutate(value)
        path.write_text(json.dumps(value))
    contract["inputs"][kind] = pin(path)
    bundle.write_text(json.dumps(contract))


def test_preserves_24k_lddt_endpoint_and_valid_zero_foldscore(bundle: Path) -> None:
    receipt = validate_run_contract(bundle)
    assert receipt["final_step"] == 30000
    assert receipt["checkpoints"][-1]["step"] == 24000
    assert receipt["checkpoints"][-1]["mean_lddt_ca"] == 0.25
    assert receipt["checkpoints"][-1]["mean_foldscore"] == 0.0


@pytest.mark.parametrize("defect", ["duplicate", "missing", "nan", "checkpoint", "version", "step", "composite"])
def test_rejects_corrupt_target_evidence(bundle: Path, defect: str) -> None:
    def mutate(rows):
        if defect == "duplicate":
            rows[1]["chain_id"] = rows[0]["chain_id"]
        elif defect == "missing":
            rows.pop(0)
        elif defect == "nan":
            rows[0]["lddt_ca"] = float("nan")
        elif defect == "checkpoint":
            rows[0]["ckpt_sha256"] = "0" * 64
        elif defect == "version":
            rows[0]["foldscore_version"] = "old"
        elif defect == "step":
            rows[0]["step"] = 30000
        else:
            rows[0]["foldscore"] = 0.5

    alter(bundle, "per_chain", mutate)
    with pytest.raises(ValueError):
        validate_run_contract(bundle)


@pytest.mark.parametrize("defect", ["duplicate", "exposure", "mean", "denominator", "official"])
def test_rejects_corrupt_summary(bundle: Path, defect: str) -> None:
    def mutate(report):
        if defect == "duplicate":
            report["checkpoints"][1] = report["checkpoints"][0]
        elif defect == "exposure":
            report["checkpoints"][0]["cumulative_samples_seen"] = 1
        elif defect == "mean":
            report["checkpoints"][0]["mean_lddt_ca"] = 0.9
        elif defect == "denominator":
            del report["num_chains"]
        else:
            report["official_mode"] = True

    alter(bundle, "summary", mutate)
    with pytest.raises(ValueError):
        validate_run_contract(bundle)


@pytest.mark.parametrize(
    "key,value", [("config_sha256", "0" * 64), ("fingerprint_sha256", "0" * 64), ("cumulative_samples_seen", 3)]
)
def test_actual_checkpoint_identity_is_required(bundle: Path, key: str, value) -> None:
    contract = json.loads(bundle.read_text())
    checkpoint = Path(contract["checkpoints"][0]["path"])
    ckpt = load_torch_checkpoint(checkpoint)
    ckpt[key] = value
    torch.save(ckpt, checkpoint)
    contract["checkpoints"][0]["sha256"] = sha256_file(checkpoint)
    bundle.write_text(json.dumps(contract))
    with pytest.raises(ValueError):
        validate_run_contract(bundle)


def test_stale_contract_pin_is_rejected(bundle: Path) -> None:
    contract = json.loads(bundle.read_text())
    Path(contract["inputs"]["per_chain"]["path"]).write_text("")
    with pytest.raises(ValueError, match="hash disagreement"):
        validate_run_contract(bundle)


def add_evaluation_inputs(bundle: Path) -> None:
    contract = json.loads(bundle.read_text())
    cfg = yaml.safe_load(Path(contract["inputs"]["config"]["path"]).read_text())
    cfg["data"].update(
        {
            "processed_features_dir": "/new/features",
            "processed_labels_dir": "/new/labels",
            "val_manifest": "/new/val.txt",
        }
    )
    evaluation_config = bundle.parent / "evaluation.yaml"
    evaluation_config.write_text(yaml.safe_dump(cfg))
    manifest = Path(contract["inputs"]["manifest"]["path"])
    manifest.write_text("new_A\nnew_B\n")
    evaluation_fingerprint = bundle.parent / "evaluation_fingerprint.json"
    evaluation_fingerprint.write_text(
        json.dumps({"manifests": {"val": {"sha256": sha256_file(manifest), "chain_count": 2}}})
    )
    contract["inputs"]["manifest"] = pin(manifest)
    contract["inputs"]["evaluation_config"] = pin(evaluation_config)
    contract["inputs"]["evaluation_fingerprint"] = pin(evaluation_fingerprint)
    per_chain = Path(contract["inputs"]["per_chain"]["path"])
    rows = [json.loads(line) for line in per_chain.read_text().splitlines()]
    for row in rows:
        row["chain_id"] = {"a_A": "new_A", "b_B": "new_B"}[row["chain_id"]]
    per_chain.write_text("".join(json.dumps(row) + "\n" for row in rows))
    contract["inputs"]["per_chain"] = pin(per_chain)
    bundle.write_text(json.dumps(contract))


def test_separate_evaluation_cohort_preserves_training_identity(bundle: Path) -> None:
    add_evaluation_inputs(bundle)
    receipt = validate_run_contract(bundle)
    assert receipt["inputs"]["evaluation_fingerprint"]["sha256"] != receipt["inputs"]["fingerprint"]["sha256"]
    assert receipt["checkpoints"][-1]["mean_lddt_ca"] == 0.25


@pytest.mark.parametrize("defect", ["missing_fingerprint", "seed", "crop", "train_manifest", "empty_path"])
def test_evaluation_overrides_cannot_change_model_or_training_protocol(bundle: Path, defect: str) -> None:
    add_evaluation_inputs(bundle)
    contract = json.loads(bundle.read_text())
    if defect == "missing_fingerprint":
        del contract["inputs"]["evaluation_fingerprint"]
    else:
        path = Path(contract["inputs"]["evaluation_config"]["path"])
        cfg = yaml.safe_load(path.read_text())
        if defect == "seed":
            cfg["seed"] = 9
        elif defect == "crop":
            cfg["data"]["crop_size"] = 64
        elif defect == "train_manifest":
            cfg["data"]["train_manifest"] = "/other/train.txt"
        else:
            cfg["data"]["processed_labels_dir"] = ""
        path.write_text(yaml.safe_dump(cfg))
        contract["inputs"]["evaluation_config"] = pin(path)
    bundle.write_text(json.dumps(contract))
    with pytest.raises(ValueError):
        validate_run_contract(bundle)


def test_present_per_chain_exposure_must_agree(bundle: Path) -> None:
    alter(bundle, "per_chain", lambda rows: rows[0].update(cumulative_samples_seen=123))
    with pytest.raises(ValueError, match="Per-chain exposure"):
        validate_run_contract(bundle)


def test_present_summary_budget_must_agree(bundle: Path) -> None:
    alter(bundle, "summary", lambda report: report.update(sample_budget=123))
    with pytest.raises(ValueError, match="Summary sample budget"):
        validate_run_contract(bundle)
