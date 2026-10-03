"""Export the repaired public and unchanged hidden cohorts from verified sources."""
from __future__ import annotations

import hashlib
import importlib
import os
from pathlib import Path
from typing import Any

from nanofold.representative_eligibility import file_record, read_json, require
from nanofold.revised_parent_leakage import POLICY as FRAME_POLICY
from nanofold.revised_parent_leakage import validate_parent_frame
from nanofold.structural_population import (
    qualification_predicate,
    read_rows,
    validate_file_binding,
    validate_qualification_predicate,
    write_json,
    write_rows,
)

POLICY = "source_qualified_repaired_heldout_export_v1"
BASELINE_POLICY = "fixed_baseline_verified_source_replay_v1"
COUNTS = {"val": 1000, "hidden": 1000, "unchanged_public": 786, "repaired_public": 214}
IMPLEMENTATIONS = ("nanofold/heldout_sources.py", "scripts/export_heldout_sources.py",
                   "scripts/analyze_structural_sampling.py")
ROOT = Path(__file__).resolve().parents[1]


def _checked(pin: dict[str, Any]) -> Path:
    path = Path(pin["path"])
    validate_file_binding(pin)
    return path


def _manifest(path: Path) -> list[str]:
    rows = [line.strip() for line in path.read_text().splitlines() if line.strip() and not line.lstrip().startswith("#")]
    require(bool(rows) and len(rows) == len(set(rows)), "Heldout source manifest is empty or repeats targets")
    return rows


def _implementations() -> dict[str, dict[str, str]]:
    return {name: file_record(ROOT / name) for name in IMPLEMENTATIONS}


def _qualified(row: dict[str, Any], *, verify_sources: bool) -> None:
    chain, sequence = row["chain_id"], row["sequence"]
    digest = hashlib.sha256(sequence.encode()).hexdigest()
    attempt = row["qualification"]
    require(row["status"] == attempt["status"] == attempt["structure_status"] == attempt["raw_status"] == "passed"
            and row["sequence_sha256"] == attempt["sequence_sha256"] == digest
            and attempt["chain_id"] == chain and row["length"] == attempt["target_length"] == len(sequence),
            "Heldout export lacks exact passing target qualification")
    raw, audit = attempt["raw"], attempt["raw_full_audit"]
    require(raw["query_sha256"] == digest and raw["query_length"] == audit["query_length"] == len(sequence)
            and all(raw[key] == audit[key] for key in ("path", "sha256", "bytes"))
            and type(audit["rows"]) is int and audit["rows"] > 0
            and type(audit["aligned_width"]) is int and audit["aligned_width"] >= len(sequence),
            "Heldout export raw query and complete MSA audit disagree")
    require(row["source_geometry"] == attempt["source_geometry"]
            and row["crop_geometry"] == attempt["crop_geometry"], "Heldout export geometry differs from qualification")
    if verify_sources:
        structure = attempt["structure"]
        require(file_record(Path(structure["path"]))["sha256"] == structure["sha256"],
                "Heldout structure source changed")
        actual = importlib.import_module("scripts.audit_target_identity").raw_record(Path(raw["path"]), sequence)
        require(actual == audit, "Heldout raw source differs from its complete MSA audit")


def _reconstruct(baseline_path: Path, frame_path: Path, *, verify_sources: bool) -> dict[str, Any]:
    baseline_pin, frame_pin = file_record(baseline_path), file_record(frame_path)
    receipt = read_json(baseline_path)
    require(receipt.get("policy") == BASELINE_POLICY, "Heldout export requires the final unified baseline replay")
    paths = {"candidate06": Path(receipt["inputs"]["train_manifest"]["path"]),
             "candidate06_rows": Path(receipt["outputs"]["train-qualified"]["path"])}
    importlib.import_module("scripts.analyze_structural_sampling").validate_fixed_baseline_replay(receipt, paths, 10000)
    frame = validate_parent_frame(frame_path)
    require(frame["policy"] == FRAME_POLICY, "Heldout export requires the source-qualified repaired frame")
    bindings = frame["source_bindings"]
    original = read_json(_checked(receipt["inputs"]["original_replay_receipt"]))
    manifests = {split: receipt["inputs"][split + "_manifest"] for split in ("train", "val", "hidden")}
    original_ids = {split: _manifest(_checked(pin)) for split, pin in manifests.items()}
    for split, pin in manifests.items():
        require(pin["sha256"] == original["inputs"][split]["sha256"], "Baseline manifest differs from original cohort")
    lock = read_json(_checked(bindings["baseline_public_lock"]))
    require(manifests["train"]["sha256"] == lock["outputs"]["train_manifest_sha256"]
            and manifests["val"]["sha256"] == bindings["historical_public"]["sha256"]
            and manifests["hidden"]["sha256"] == bindings["frozen_hidden"]["sha256"],
            "Baseline replay and repaired frame have different original cohorts")
    train = read_rows(_checked(receipt["outputs"]["train-qualified"]))
    heldout = read_rows(_checked(receipt["outputs"]["heldout-chains"]))
    all_rows = train + heldout
    by_id = {row["chain_id"]: row for row in all_rows}
    require(len(by_id) == len(all_rows) == 12000, "Baseline replay repeats or omits original targets")
    for split, count in (("train", 10000), ("val", 1000), ("hidden", 1000)):
        ids = [row["chain_id"] for row in all_rows if row["split"] == split]
        require(len(ids) == len(original_ids[split]) == count and set(ids) == set(original_ids[split]),
                "Baseline rows differ from exact original split membership")
    require({row["chain_id"] for row in train} == set(original_ids["train"])
            and {row["chain_id"] for row in heldout} == set(original_ids["val"] + original_ids["hidden"]),
            "Baseline output roles differ from original split membership")
    prior_rows = read_rows(_checked(receipt["inputs"]["train"])) + read_rows(_checked(receipt["inputs"]["heldout"]))
    prior = {row["chain_id"]: row for row in prior_rows}
    require(len(prior) == len(prior_rows) == len(by_id) and set(prior) == set(by_id),
            "Final baseline replay changed its original target rows")
    mutable = {"qualification", "source_geometry", "crop_geometry", "status", "original_source_bindings"}
    for chain, row in by_id.items():
        previous = prior[chain]
        require({key: value for key, value in row.items() if key not in mutable}
                == {key: value for key, value in previous.items() if key not in mutable},
                "Final baseline replay changed original target metadata")
        require(row["original_source_bindings"] == {key: previous["qualification"][key] for key in ("structure", "raw")}
                and all(row["qualification"][key]["sha256"] == previous["qualification"][key]["sha256"]
                        for key in ("structure", "raw")), "Final baseline replay changed original source bytes")
    records = {row["chain_id"]: row for row in frame["records"]}
    require(len(records) == len(frame["records"]), "Repaired frame repeats targets")
    for row in heldout:
        reference = records[row["chain_id"]]
        require(row["sequence"] == reference["sequence"] and row["sequence_sha256"] == reference["sequence_sha256"]
                and row["length"] == reference["length"], "Baseline heldout sequence differs from repaired frame")
        _qualified(row, verify_sources=False)
    active = _manifest(_checked(bindings["frozen_val"]))
    hidden = _manifest(_checked(bindings["frozen_hidden"]))
    cohorts = frame["evaluation_cohorts"]
    require(frame["frozen_splits"] == {"val": active, "hidden_val": hidden}
            and cohorts == {"historical_public": original_ids["val"], "active_public": active, "hidden": hidden}
            and hidden == original_ids["hidden"] and len(active) == len(hidden) == 1000,
            "Repaired frame changed active or original heldout membership")
    unchanged = set(active) & set(original_ids["val"])
    added = set(active) - set(original_ids["val"])
    qualifications = frame["selected_qualifications"]
    require(len(unchanged) == COUNTS["unchanged_public"] and len(added) == COUNTS["repaired_public"]
            and set(qualifications) == added and not set(active) & set(hidden),
            "Heldout export differs from the approved 786+214 public repair")
    exported = []
    for chain in active + hidden:
        if chain in added:
            attempt = qualifications[chain]
            row = records[chain] | {"split": "val", "status": "passed", "qualification": attempt,
                                    "source_geometry": attempt["source_geometry"], "crop_geometry": attempt["crop_geometry"]}
        else:
            row = by_id[chain]
        _qualified(row, verify_sources=verify_sources)
        exported.append(row)
    require(file_record(baseline_path) == baseline_pin and file_record(frame_path) == frame_pin,
            "Heldout export recipe changed during validation")
    return {"rows": exported, "baseline_replay_receipt": baseline_pin, "full_frame_receipt": frame_pin,
            "baseline_manifests": manifests, "public_manifest": bindings["frozen_val"],
            "hidden_manifest": bindings["frozen_hidden"], "selection_receipt": bindings["selection_receipt"]}


def export_heldout_sources(baseline_replay_receipt: Path, full_frame_receipt: Path, output: Path) -> dict[str, Any]:
    """Write a fresh private 2,000-target source index after complete validation."""
    require(not output.is_symlink(), "Heldout export output must not be a symlink")
    output = output.resolve()
    require(not output.is_relative_to(ROOT) or output.is_relative_to(ROOT / ".nanofold_private"),
            "Heldout source exports must remain outside the tracked repository")
    require(not output.exists(), "Heldout source export requires a fresh output directory")
    os.umask(0o077)
    predicate, code = qualification_predicate(), _implementations()
    context = _reconstruct(baseline_replay_receipt, full_frame_receipt, verify_sources=True)
    output.mkdir(parents=True, exist_ok=False, mode=0o700)
    heldout_path = output / "heldout-chains.jsonl"
    write_rows(heldout_path, context.pop("rows"))
    for name, filename in (("public_manifest", "val.txt"), ("hidden_manifest", "hidden_val.txt")):
        path = output / filename
        with path.open("xb") as handle:
            handle.write(_checked(context[name]).read_bytes())
        copied = file_record(path)
        require(copied["sha256"] == context[name]["sha256"], "Heldout manifest changed during export copying")
        context[name] = copied
    require(qualification_predicate() == predicate and _implementations() == code, "Heldout export implementation changed")
    receipt = {"schema_version": 1, "policy": POLICY, "status": "complete", **context,
               "heldout_chains": file_record(heldout_path), "counts": dict(COUNTS),
               "qualification_predicate": predicate, "code": code}
    write_json(output / "receipt.json", receipt)
    return receipt


def validate_heldout_sources(receipt_path: Path, verify_sources: bool = True) -> dict[str, Any]:
    """Reconstruct exact rows from owning evidence; updated output hashes cannot authorize changes."""
    pin = file_record(receipt_path)
    receipt = read_json(receipt_path)
    require(receipt.get("schema_version") == 1 and receipt.get("policy") == POLICY
            and receipt.get("status") == "complete" and receipt.get("counts") == COUNTS,
            "Heldout source export is incomplete or has another cohort policy")
    validate_qualification_predicate(receipt["qualification_predicate"])
    current = _implementations()
    require(set(receipt["code"]) == set(current), "Heldout export implementation bindings differ")
    for name, record in receipt["code"].items():
        _checked(record)
        require(record["sha256"] == current[name]["sha256"], "Heldout export implementation bytes changed")
    context = _reconstruct(_checked(receipt["baseline_replay_receipt"]), _checked(receipt["full_frame_receipt"]),
                           verify_sources=verify_sources)
    require(read_rows(_checked(receipt["heldout_chains"])) == context.pop("rows"),
            "Heldout source export rows differ from original and selected source qualifications")
    for name in ("public_manifest", "hidden_manifest"):
        require(_checked(receipt[name]).read_bytes() == _checked(context[name]).read_bytes(),
                "Heldout source export manifest differs from the active frame")
        context.pop(name)
    require(all(receipt[key] == value for key, value in context.items()), "Heldout export recipe bindings differ")
    require(file_record(receipt_path) == pin, "Heldout source export receipt changed during validation")
    return receipt
