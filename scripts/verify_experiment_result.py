"""Verify a private research final public endpoint and recompute its exact cohort means."""
from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from pathlib import Path
from typing import Any

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nanofold.data import read_manifest
from nanofold.metrics import FOLDSCORE_COMPONENT_NAMES, FOLDSCORE_VERSION, FOLDSCORE_WEIGHT_BY_COMPONENT
from nanofold.prediction_contract import validate_complete_chain_ids
from nanofold.utils import load_torch_checkpoint, sha256_file


def _pin(path: Path) -> dict[str, Any]:
    return {"path": str(path.resolve()), "sha256": sha256_file(path), "bytes": path.stat().st_size}


def _integer(value: Any, expected: int, name: str) -> None:
    if type(value) is not int or value != expected:
        raise ValueError(f"{name} must equal the prespecified integer {expected}.")


def _score(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f"{name} must be a finite score in [0, 1].")
    return float(value)


def verify_result(
    *, config: Path, fingerprint: Path, manifest: Path, checkpoint: Path,
    summary: Path, per_chain: Path, seed: int, final_step: int, sample_budget: int,
) -> dict[str, Any]:
    """Verify one final endpoint; this does not certify AUC or source/runtime equivalence."""
    if final_step <= 0 or sample_budget <= 0 or seed < 0:
        raise ValueError("Endpoint steps/budget must be positive and seed nonnegative.")
    cfg = yaml.safe_load(config.read_text())
    expected_data = json.loads(fingerprint.read_text())
    ckpt = load_torch_checkpoint(checkpoint)
    report = json.loads(summary.read_text())
    chain_ids = read_manifest(manifest)
    if report.get("official_mode") is not False:
        raise ValueError("This verifier requires the explicit private research route, without official config overrides.")
    if report.get("foldscore_version") != FOLDSCORE_VERSION:
        raise ValueError("Summary uses a different or unrecorded FoldScore version.")
    _integer(cfg["seed"], seed, "Config seed")
    _integer(cfg["train"]["max_steps"], final_step, "Final step")
    if cfg["data"]["batch_size"] * cfg["train"]["grad_accum_steps"] * final_step != sample_budget:
        raise ValueError("Config does not implement the prespecified sample budget.")
    val_fingerprint = expected_data["manifests"]["val"]
    if val_fingerprint["sha256"] != sha256_file(manifest) or val_fingerprint["chain_count"] != len(chain_ids):
        raise ValueError("Evaluation manifest is not the fingerprint's public validation cohort.")
    for key, expected in (("config_sha256", sha256_file(config)), ("fingerprint_sha256", sha256_file(fingerprint))):
        if ckpt.get(key) != expected:
            raise ValueError(f"Checkpoint {key} does not match the planned experiment.")
    resolved_cfg = copy.deepcopy(ckpt.get("config"))
    relocations = {}
    if isinstance(resolved_cfg, dict) and isinstance(resolved_cfg.get("data"), dict):
        for key in ("processed_features_dir", "processed_labels_dir"):
            planned = cfg["data"].get(key)
            actual = resolved_cfg["data"].get(key)
            if isinstance(planned, str) and planned and isinstance(actual, str) and actual:
                if actual != planned:
                    relocations[key] = {"planned": planned, "resolved": actual}
                resolved_cfg["data"][key] = planned
    if resolved_cfg != cfg:
        raise ValueError("Checkpoint resolved config does not match the planned experiment.")
    _integer(ckpt.get("step"), final_step, "Checkpoint step")
    _integer(ckpt.get("cumulative_samples_seen"), sample_budget, "Checkpoint exposures")
    _integer(ckpt.get("sample_budget"), sample_budget, "Checkpoint sample budget")
    if report.get("split") != "val":
        raise ValueError("This verifier accepts only the public validation endpoint.")
    items = report.get("checkpoints")
    if not isinstance(items, list) or not items or items[-1].get("step") != final_step:
        raise ValueError("Summary must end at the prespecified final checkpoint.")
    matches = [item for item in items if item.get("step") == final_step]
    if len(matches) != 1:
        raise ValueError("Summary must contain exactly one final checkpoint entry.")
    endpoint = matches[0]
    checkpoint_hash = sha256_file(checkpoint)
    if endpoint.get("ckpt_sha256") != checkpoint_hash:
        raise ValueError("Summary final checkpoint hash does not match the actual checkpoint.")
    for record in (report, endpoint):
        _integer(record.get("num_chains"), len(chain_ids), "Summary denominator")
        _integer(record.get("cumulative_samples_seen"), sample_budget, "Summary exposures")
    rows = [json.loads(line) for line in per_chain.read_text().splitlines() if line.strip()]
    selected = [row for row in rows if row.get("step") == final_step]
    validate_complete_chain_ids([row["chain_id"] for row in selected], chain_ids)
    for row in selected:
        if row.get("foldscore_version") != FOLDSCORE_VERSION:
            raise ValueError("Per-chain record uses a different or unrecorded FoldScore version.")
        if Path(row["ckpt"]).resolve() != Path(endpoint["ckpt"]).resolve():
            raise ValueError("Per-chain record belongs to a different checkpoint.")
        if row.get("ckpt_sha256") != checkpoint_hash:
            raise ValueError("Per-chain checkpoint hash disagrees with the summary.")
        composite = math.fsum(_score(row.get(name), name) * weight
                              for name, weight in FOLDSCORE_WEIGHT_BY_COMPONENT.items())
        if not math.isclose(_score(row.get("foldscore"), "foldscore"), composite, rel_tol=0, abs_tol=1e-7):
            raise ValueError("Per-chain FoldScore differs from its declared weighted components.")
    means = {}
    for metric in FOLDSCORE_COMPONENT_NAMES:
        mean = math.fsum(_score(row.get(metric), metric) for row in selected) / len(chain_ids)
        for record in (report, endpoint):
            reported = _score(record.get(f"mean_{metric}"), f"mean_{metric}")
            if not math.isclose(mean, reported, rel_tol=0, abs_tol=1e-7):
                raise ValueError(f"Reported mean_{metric} differs from exact per-chain records.")
        means[f"mean_{metric}"] = mean
    return {
        "schema_version": 1, "status": "verified_final_public_endpoint",
        "run_name": cfg["run_name"], "seed": seed, "step": final_step,
        "cumulative_samples_seen": sample_budget, "num_chains": len(chain_ids), **means,
        "runtime_data_relocations": relocations,
        "inputs": {name: _pin(path) for name, path in {
            "config": config, "fingerprint": fingerprint, "manifest": manifest,
            "checkpoint": checkpoint, "summary": summary, "per_chain": per_chain,
        }.items()},
        "limitations": "Final endpoint consistency only. Separately verify runtime/source, actual data files, prediction scoring, AUC, and all prespecified runs.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("config", "fingerprint", "manifest", "checkpoint", "summary", "per-chain", "out"):
        parser.add_argument(f"--{field}", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--final-step", type=int, required=True)
    parser.add_argument("--sample-budget", type=int, required=True)
    args = vars(parser.parse_args())
    output = args.pop("out")
    receipt = verify_result(**args)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        json.dump(receipt, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(f"Verified final public endpoint; receipt: {output}")


if __name__ == "__main__":
    main()
