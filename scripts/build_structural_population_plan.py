"""Freeze source qualification order for every barrier-admitted training copy.

The owning full-parent validator replays heldout exclusion evidence before this
script partitions training copies by exact sequence. No source is downloaded or
qualified here. Exhausted sequences and groups remain qualification evidence;
the later sampling stage decides whether its coverage requirements are met.
"""
from __future__ import annotations

import argparse
import importlib
import os
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
for directory in (REPO_ROOT, REPO_ROOT / "scripts"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from build_manifests import Candidate, _representative_rank

from nanofold.representative_eligibility import file_record, require
from nanofold.revised_parent_leakage import validate_parent_frame
from nanofold.structural_population import (
    POLICY,
    sequence_plan,
    validate_file_binding,
    write_json,
    write_rows,
)

COPY_FIELDS = ("chain_id", "sequence", "length", "resolution", "release_date", "group_id", "cluster40_id")


def copy_rank(row: dict[str, Any]) -> tuple[float, int, str, int, str]:
    """Use the owning representative rank with the fixed qualification seed."""
    candidate = Candidate(
        chain_id=row["chain_id"], sequence=row["sequence"], length=row["length"],
        resolution=row["resolution"], release_date=row["release_date"], cluster_id=row["cluster40_id"],
        pdb_id=row["chain_id"].split("_", 1)[0], secondary_structure_class="unknown",
        secondary_helix_fraction=None, secondary_beta_fraction=None, secondary_coil_fraction=None,
        domain_architecture_class="unknown", domain_architecture_source="unknown", metadata_source_count=0,
    )
    return _representative_rank(candidate, seed=0)


def training_copies(frame: dict[str, Any]) -> list[dict[str, Any]]:
    """Retain all admitted deposited copies before any exact-sequence collapse."""
    records = frame["records"]
    by_chain = {row["chain_id"]: row for row in records}
    admitted = frame["eligible_training_ids"]
    require(len(by_chain) == len(records) and len(set(admitted)) == len(admitted)
            and set(admitted) <= set(by_chain), "Full-parent training frame repeats or escapes source copies")
    copies = []
    for chain in sorted(admitted):
        original = by_chain[chain]
        require(not original["training_exclusion_reasons"]
                and isinstance(original["group_id"], str) and bool(original["group_id"])
                and isinstance(original["cluster40_id"], str) and bool(original["cluster40_id"]),
                "Admitted source copy lacks training ownership or diversity assignments")
        row = {key: original[key] for key in COPY_FIELDS}
        if row["resolution"] is None:
            row["resolution"] = 999.0
        copies.append(row)
    require(bool(copies), "Full-parent barrier admitted no training copies")
    return copies


def build(full_frame_receipt: Path, output: Path) -> dict[str, Any]:
    os.umask(0o077)
    output = output.resolve()
    require(not output.exists(), "Population plan requires a fresh output directory")
    require(not output.is_relative_to(REPO_ROOT) or output.is_relative_to(REPO_ROOT / ".nanofold_private"),
            "Population plans belong in the ignored private root")
    revised = importlib.import_module("scripts.build_revised_parent_frame")
    selector = importlib.import_module("nanofold.heldout_repair_selection")
    code_files = {
        "scripts/build_structural_population_plan.py", "scripts/build_manifests.py",
        "nanofold/structural_population.py", "nanofold/representative_eligibility.py",
        "nanofold/heldout_repair_selection.py", *revised.CODE_FILES, *selector.IMPLEMENTATION_DEPENDENCIES,
    }
    inputs = [file_record(full_frame_receipt), *(file_record(REPO_ROOT / name) for name in sorted(code_files))]
    frame = validate_parent_frame(full_frame_receipt)
    require(frame["receipt"] == inputs[0], "Full-parent receipt changed during validation")
    inputs.extend(frame["source_bindings"].values())
    copies = training_copies(frame)
    plans = sequence_plan(copies, rank=copy_rank)
    for binding in inputs:
        validate_file_binding(binding)
    output.mkdir(parents=True, mode=0o700)
    sequence_path = output / "sequence-plan.jsonl"
    write_rows(sequence_path, plans)
    for binding in inputs:
        validate_file_binding(binding)
    plan = {
        "policy": POLICY, "seed": 0, "require_all_groups": False,
        "full_frame_receipt": frame["receipt"], "inputs": inputs,
        "sequence_plan": file_record(sequence_path), "sequence_count": len(plans),
        "source_chain_count": len(copies), "training_group_count": len({row["group_id"] for row in copies}),
    }
    write_json(output / "plan.json", plan)
    return plan


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-frame-receipt", type=Path, required=True,
                        help="Completed full-parent frame receipt, replayed by its owning validator")
    parser.add_argument("--output", type=Path, required=True,
                        help="Fresh private directory for sequence-plan.jsonl and plan.json")
    args = parser.parse_args()
    build(args.full_frame_receipt, args.output)


if __name__ == "__main__":
    main()
