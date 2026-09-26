"""Validate prespecified research endpoints against immutable execution artifacts.

This verifies evidence consistency, not training/source equivalence or scientific
independence. Evaluation inputs must come from the pinned evaluator's execution.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from pathlib import Path
from typing import Any

import yaml

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nanofold.data import read_manifest
from nanofold.metrics import FOLDSCORE_COMPONENT_NAMES, FOLDSCORE_VERSION, FOLDSCORE_WEIGHT_BY_COMPONENT
from nanofold.prediction_contract import validate_complete_chain_ids
from nanofold.rmsd import summarize_rmsd_ca
from nanofold.utils import load_torch_checkpoint, sha256_file


def _integer(value: Any, name: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


def _score(value: Any, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or not 0 <= value <= 1
    ):
        raise ValueError(f"{name} must be finite in [0, 1]")
    return float(value)


def _pin(spec: dict[str, Any], root: Path) -> Path:
    path = Path(spec["path"])
    if not path.is_absolute():
        path = root / path
    if sha256_file(path) != spec.get("sha256"):
        raise ValueError(f"Artifact hash disagreement: {path}")
    return path.resolve()



def _verify_rmsd_summary(observed: dict[str, Any], expected: dict[str, Any]) -> None:
    for key, value in expected.items():
        if key not in observed:
            raise ValueError(f"Missing RMSD summary evidence: {key}")
        actual = observed[key]
        if key == "mean_rmsd_ca" and value is not None:
            if (type(actual) not in (int, float) or not math.isfinite(actual)
                    or not math.isclose(actual, value, rel_tol=0, abs_tol=1e-7)):
                raise ValueError("RMSD summary disagrees with exact-cohort mean")
        elif type(actual) is not type(value) or actual != value:
            raise ValueError(f"RMSD summary coverage/status disagreement: {key}")


def validate_run_contract(contract_path: Path) -> dict[str, Any]:
    """Validate every declared checkpoint and recompute exact-cohort score means.

    ``final_step`` is the training horizon; the checkpoint schedule is the explicit
    analysis endpoint selection. No latest/best checkpoint is selected implicitly.
    All paths are pinned with SHA256, including the evaluator source file.
    """
    contract_hash = sha256_file(contract_path)
    contract = json.loads(contract_path.read_text())
    if contract.get("schema_version") != 1:
        raise ValueError("Unsupported experiment reporting contract schema")
    for key in ("campaign_id", "run_name"):
        if not isinstance(contract.get(key), str) or not contract[key].strip():
            raise ValueError(f"Missing {key}")
    seed = _integer(contract.get("seed"), "seed")
    final_step = _integer(contract.get("final_step"), "final_step")
    budget = _integer(contract.get("sample_budget"), "sample_budget")
    if final_step == 0 or budget == 0:
        raise ValueError("Training horizon and sample budget must be positive")
    specs = contract["inputs"]
    names: tuple[str, ...] = ("config", "fingerprint", "manifest", "summary", "per_chain", "evaluator")
    separate_evaluation = "evaluation_config" in specs or "evaluation_fingerprint" in specs
    if separate_evaluation:
        if not {"evaluation_config", "evaluation_fingerprint"} <= specs.keys():
            raise ValueError("Separate evaluation requires both config and fingerprint pins")
        names += ("evaluation_config", "evaluation_fingerprint")
    paths = {name: _pin(specs[name], contract_path.parent) for name in names}
    cfg = yaml.safe_load(paths["config"].read_text())
    fingerprint = json.loads(paths["fingerprint"].read_text())
    report = json.loads(paths["summary"].read_text())
    chains = read_manifest(paths["manifest"])
    if not chains or len(chains) != len(set(chains)):
        raise ValueError("Manifest must be nonempty and contain unique chain IDs")
    if cfg.get("run_name") != contract["run_name"] or _integer(cfg.get("seed"), "config seed") != seed:
        raise ValueError("Config run/seed identity disagreement")
    if _integer(cfg["train"].get("max_steps"), "configured horizon") != final_step:
        raise ValueError("Config training horizon disagreement")
    batch = _integer(cfg["data"].get("batch_size"), "batch size")
    accum = _integer(cfg["train"].get("grad_accum_steps"), "gradient accumulation")
    if batch * accum * final_step != budget:
        raise ValueError("Config does not implement the declared sample budget")
    if separate_evaluation:
        evaluation_cfg = yaml.safe_load(paths["evaluation_config"].read_text())
        normalised_evaluation_cfg = copy.deepcopy(evaluation_cfg)
        if not isinstance(normalised_evaluation_cfg, dict) or not isinstance(
            normalised_evaluation_cfg.get("data"), dict
        ):
            raise ValueError("Evaluation config must contain a data mapping")
        for key in ("processed_features_dir", "processed_labels_dir", "val_manifest"):
            if not isinstance(evaluation_cfg["data"].get(key), str) or not evaluation_cfg["data"][key].strip():
                raise ValueError(f"Evaluation config requires explicit nonempty {key}")
            if key in cfg["data"]:
                normalised_evaluation_cfg["data"][key] = cfg["data"][key]
            else:
                normalised_evaluation_cfg["data"].pop(key)
        if normalised_evaluation_cfg != cfg:
            raise ValueError("Evaluation config changes fields beyond the three permitted data overrides")
        evaluation_fingerprint = json.loads(paths["evaluation_fingerprint"].read_text())
    else:
        evaluation_fingerprint = fingerprint
    val = evaluation_fingerprint["manifests"]["val"]
    if val["sha256"] != specs["manifest"]["sha256"] or _integer(
        val.get("chain_count"), "fingerprint cohort size"
    ) != len(chains):
        raise ValueError("Manifest differs from evaluation fingerprint validation cohort")
    if report.get("official_mode") is not False or report.get("split") != "val":
        raise ValueError("Reporting requires explicit private research evaluation on public validation")
    if "sample_budget" in report and _integer(report["sample_budget"], "summary sample budget") != budget:
        raise ValueError("Summary sample budget disagreement")
    if report.get("foldscore_version") != FOLDSCORE_VERSION:
        raise ValueError("Unrecorded or incompatible FoldScore version")
    if _integer(report.get("num_chains"), "summary cohort size") != len(chains):
        raise ValueError("Summary cohort denominator disagreement")
    metrics = contract.get("metrics", list(FOLDSCORE_COMPONENT_NAMES))
    if not isinstance(metrics, list) or not metrics or len(metrics) != len(set(metrics)):
        raise ValueError("Declare a nonempty, unique metric list")
    supported = set(FOLDSCORE_COMPONENT_NAMES) | {"lddt_ca", "gdt_ts_ca", "lddt_backbone_atom14", "rmsd_ca"}
    if any(metric not in supported for metric in metrics):
        raise ValueError("Unsupported reporting metric")
    required_metrics = sorted((set(metrics) | set(FOLDSCORE_COMPONENT_NAMES)) - {"rmsd_ca"})
    require_rmsd = "rmsd_ca" in metrics
    schedule = contract.get("checkpoints")
    if not isinstance(schedule, list) or not schedule:
        raise ValueError("Explicit checkpoint schedule required")
    expected: dict[int, dict[str, Any]] = {}
    previous_step = previous_samples = -1
    relocations: dict[str, Any] = {}
    for planned in schedule:
        step = _integer(planned.get("step"), "planned checkpoint step")
        exposure = _integer(planned.get("cumulative_samples_seen"), "planned checkpoint exposure")
        if step <= previous_step or exposure <= previous_samples or step > final_step or exposure > budget:
            raise ValueError("Checkpoint schedule must strictly increase within training horizon/budget")
        previous_step, previous_samples = step, exposure
        checkpoint = _pin(planned, contract_path.parent)
        ckpt = load_torch_checkpoint(checkpoint)
        if (
            _integer(ckpt.get("step"), "actual checkpoint step") != step
            or _integer(ckpt.get("cumulative_samples_seen"), "actual exposure") != exposure
        ):
            raise ValueError("Actual checkpoint step/exposure disagreement")
        if _integer(ckpt.get("sample_budget"), "checkpoint sample budget") != budget:
            raise ValueError("Checkpoint budget disagreement")
        for key, source in (("config_sha256", "config"), ("fingerprint_sha256", "fingerprint")):
            if ckpt.get(key) != specs[source]["sha256"]:
                raise ValueError(f"Checkpoint {key} disagreement")
        resolved = copy.deepcopy(ckpt.get("config"))
        if isinstance(resolved, dict) and isinstance(resolved.get("data"), dict):
            for key in ("processed_features_dir", "processed_labels_dir"):
                planned_value, actual = cfg["data"].get(key), resolved["data"].get(key)
                if isinstance(planned_value, str) and planned_value and isinstance(actual, str) and actual:
                    if actual != planned_value:
                        relocations[f"{step}:{key}"] = {"planned": planned_value, "resolved": actual}
                    resolved["data"][key] = planned_value
        if resolved != cfg:
            raise ValueError("Checkpoint resolved config disagreement")
        expected[step] = planned
    items = report.get("checkpoints")
    if not isinstance(items, list) or len(items) != len(expected):
        raise ValueError("Summary does not contain the exact declared checkpoint schedule")
    by_step: dict[int, dict[str, Any]] = {}
    for item in items:
        step = _integer(item.get("step"), "summary step")
        if step not in expected or step in by_step:
            raise ValueError("Unexpected or duplicate checkpoint summary")
        if item.get("ckpt_sha256") != expected[step]["sha256"]:
            raise ValueError("Summary checkpoint hash disagreement")
        if (
            _integer(item.get("cumulative_samples_seen"), "summary exposure")
            != expected[step]["cumulative_samples_seen"]
        ):
            raise ValueError("Summary exposure disagreement")
        if _integer(item.get("num_chains"), "checkpoint cohort size") != len(chains):
            raise ValueError("Checkpoint cohort denominator disagreement")
        by_step[step] = item
    rows_by_step: dict[int, list[dict[str, Any]]] = {step: [] for step in expected}
    for line in paths["per_chain"].read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        step = _integer(row.get("step"), "per-chain step")
        if step not in expected:
            raise ValueError("Unexpected per-chain checkpoint step")
        if row.get("foldscore_version") != FOLDSCORE_VERSION or row.get("ckpt_sha256") != expected[step]["sha256"]:
            raise ValueError("Per-chain metric/checkpoint identity disagreement")
        if (
            "cumulative_samples_seen" in row
            and _integer(row["cumulative_samples_seen"], "per-chain exposure")
            != expected[step]["cumulative_samples_seen"]
        ):
            raise ValueError("Per-chain exposure disagreement")
        if row.get("ckpt") != by_step[step].get("ckpt"):
            raise ValueError("Per-chain checkpoint reference disagreement")
        if "status" in row and row["status"] != "ok":
            raise ValueError("Unsuccessful target in reporting cohort")
        composite = math.fsum(
            _score(row.get(name), name) * weight for name, weight in FOLDSCORE_WEIGHT_BY_COMPONENT.items()
        )
        if not math.isclose(_score(row.get("foldscore"), "foldscore"), composite, rel_tol=0, abs_tol=1e-7):
            raise ValueError("Per-chain FoldScore differs from its components")
        rows_by_step[step].append(row)
    recomputed = []
    for step, planned in expected.items():
        rows = rows_by_step[step]
        validate_complete_chain_ids([row["chain_id"] for row in rows], chains)
        result = {
            "step": step,
            "cumulative_samples_seen": planned["cumulative_samples_seen"],
            "ckpt_sha256": planned["sha256"],
            "num_chains": len(chains),
        }
        for metric in required_metrics:
            value = math.fsum(_score(row.get(metric), metric) for row in rows) / len(chains)
            key = f"mean_{metric}"
            if not math.isclose(value, _score(by_step[step].get(key), key), rel_tol=0, abs_tol=1e-7):
                raise ValueError(f"Checkpoint summary disagrees with exact-cohort {key}")
            result[key] = value
        if require_rmsd:
            diagnostic = summarize_rmsd_ca(rows, expected_count=len(chains))
            _verify_rmsd_summary(by_step[step], diagnostic)
            result.update(diagnostic)
        recomputed.append(result)
    last = recomputed[-1]
    if _integer(report.get("cumulative_samples_seen"), "final summary exposure") != last["cumulative_samples_seen"]:
        raise ValueError("Top-level summary exposure differs from final declared endpoint")
    for metric in required_metrics:
        key = f"mean_{metric}"
        if not math.isclose(last[key], _score(report.get(key), key), rel_tol=0, abs_tol=1e-7):
            raise ValueError(f"Top-level summary disagrees with final {key}")
    if require_rmsd:
        _verify_rmsd_summary(report, {key: value for key, value in last.items()
                                     if key == "mean_rmsd_ca" or key.startswith("rmsd_ca_")})
    for name in names:
        _pin(specs[name], contract_path.parent)
    for planned in expected.values():
        _pin(planned, contract_path.parent)
    if sha256_file(contract_path) != contract_hash:
        raise ValueError("Contract changed during validation")
    return {
        "schema_version": 1,
        "status": "verified_public_checkpoints",
        "campaign_id": contract["campaign_id"],
        "run_name": contract["run_name"],
        "seed": seed,
        "foldscore_version": FOLDSCORE_VERSION,
        "sample_budget": budget,
        "final_step": final_step,
        "num_chains": len(chains),
        "metrics": metrics,
        "checkpoints": recomputed,
        "contract_sha256": contract_hash,
        "inputs": {name: {"path": str(path), "sha256": specs[name]["sha256"]} for name, path in paths.items()},
        "runtime_data_relocations": relocations,
        "limitations": "Artifact consistency only; source/runtime execution and actual data materialization require separate receipts. No training-seed uncertainty inferred.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(validate_run_contract(args.contract), allow_nan=False))


if __name__ == "__main__":
    main()
