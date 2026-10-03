"""Qualify a frozen training sequence frame with a resumable private journal.

The plan contains every metadata-eligible training-owned chain, grouped only by
identical sequence, with copies ranked before source availability is examined.
Required sources must already be acquired by the owning raw-source retrievers.
Missing sources pause unless a pinned complete acquisition certificate proves
the required raw source unavailable. Every planned parent retains an outcome.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from nanofold.representative_eligibility import (
    file_record,
    inspect_candidate,
    read_json,
    require,
)
from nanofold.source_availability import validate_source_availability
from nanofold.structural_population import (
    POLICY,
    qualification_predicate,
    read_rows,
    summarize,
    unavailable_attempt,
    validate_file_binding,
    validate_outcome,
    validate_outcome_source_membership,
    validate_qualification_predicate,
    validate_receipt,
    validate_source_catalog,
    write_json,
    write_rows,
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def inspect_sequence(
    payload: tuple[dict[str, Any], dict[str, Any], list[str], dict[str, dict[str, Any]]],
) -> dict[str, Any]:
    plan, structures, raw_roots, availability = payload
    attempts: list[dict[str, Any]] = []
    selected = None
    for rank, candidate in enumerate(plan["ranked_copies"]):
        chain = candidate["chain_id"]
        if chain in availability:
            attempts.append(unavailable_attempt(candidate, rank, availability[chain]))
            continue
        pdb = chain.split("_", 1)[0]
        if pdb not in structures:
            return {"status": "sources_needed", "sequence_sha256": plan["sequence_sha256"],
                    "group_id": plan["group_id"], "attempts": attempts,
                    "pending_chain_id": chain, "missing": "historical_structure"}
        try:
            result = inspect_candidate(chain=chain, sequence=plan["sequence"], rank=rank,
                                       structure=structures[pdb], raw_roots=[Path(root) for root in raw_roots],
                                       include_geometry=True, full_msa_validation=True)
        except ValueError as exc:
            if str(exc) != "Required raw query source is missing; acquisition is incomplete":
                raise
            return {"status": "sources_needed", "sequence_sha256": plan["sequence_sha256"],
                    "group_id": plan["group_id"], "attempts": attempts,
                    "pending_chain_id": chain, "missing": "raw_query"}
        attempts.append(result)
        if result["status"] == "passed":
            selected = chain
            break
    outcome = {"sequence_sha256": plan["sequence_sha256"], "group_id": plan["group_id"],
               "selected_chain_id": selected, "attempts": attempts}
    validate_outcome(plan, outcome, verify_sources=False, source_availability=availability)
    return outcome


def _check_bindings(records: list[dict[str, str]]) -> None:
    for row in records:
        validate_file_binding(row)


def run(args: argparse.Namespace) -> dict[str, Any]:
    os.umask(0o077)
    output = args.output.resolve()
    require(not output.is_relative_to(REPO_ROOT) or output.is_relative_to(REPO_ROOT / ".nanofold_private"),
            "Qualification evidence belongs in the ignored private root")
    output.mkdir(parents=True, exist_ok=True, mode=0o700)
    with (output / ".writer.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return _run_locked(args, output)


def _run_locked(args: argparse.Namespace, output: Path) -> dict[str, Any]:
    plan_binding = file_record(args.plan)
    plan = read_json(args.plan)
    require(plan.get("policy") == POLICY, "Unexpected population plan policy")
    _check_bindings(plan["inputs"])
    sequence_binding = plan["sequence_plan"]
    require(file_record(Path(sequence_binding["path"])) == sequence_binding, "Sequence plan changed")
    plans = read_rows(Path(sequence_binding["path"]))
    require(len(plans) == plan["sequence_count"] and bool(plans), "Invalid sequence plan size")
    require(len({row["sequence_sha256"] for row in plans}) == len(plans), "Repeated planned sequence")
    catalog = read_json(args.source_catalog)
    validate_source_catalog(catalog)
    structures = catalog["members"]
    predicate = qualification_predicate()
    binding = {"schema_version": 1, "policy": POLICY, "plan": plan_binding,
               "sequence_plan": sequence_binding, "predicate": predicate,
               "historical_archive": catalog["historical_sources"]["archive_pin"],
               "raw_roots": sorted(str(path.resolve()) for path in args.raw_root)}
    availability_path = args.source_availability_certificate
    if availability_path is not None:
        binding["source_availability_certificate"] = file_record(availability_path)
    availability = (validate_source_availability(availability_path, plans, binding["raw_roots"])
                    if availability_path is not None else {})
    binding_path = output / "qualification-bindings.json"
    if binding_path.exists():
        previous = read_json(binding_path)
        require({key: value for key, value in previous.items() if key != "predicate"}
                == {key: value for key, value in binding.items() if key != "predicate"},
                "Resume differs from frozen qualification bindings")
        validate_qualification_predicate(previous["predicate"])
        binding = previous
        predicate = binding["predicate"]
    else:
        write_json(binding_path, binding)
    receipt_path = output / "receipt.json"
    if receipt_path.exists():
        return validate_receipt(receipt_path, verify_sources=True)
    records_dir = output / "sequence-results"
    records_dir.mkdir(exist_ok=True)
    completed: dict[str, dict[str, Any]] = {}
    pending = []
    for row in plans:
        path = records_dir / (row["sequence_sha256"] + ".json")
        if path.exists():
            result = read_json(path)
            validate_outcome_source_membership(result, catalog, binding["raw_roots"], availability)
            validate_outcome(row, result, verify_sources=True, source_availability=availability)
            completed[row["sequence_sha256"]] = result
        else:
            pending.append(row)
    progress_path = output / "progress.jsonl"
    started = _now()
    source_requests = []
    with progress_path.open("a") as progress:
        progress.write(json.dumps({"status": "running", "started_at": started, "pid": os.getpid(),
                                   "resumed_sequence_count": len(completed), "remaining": len(pending)}) + "\n")
        progress.flush()
        # Only each sequence's relevant structure records cross process boundaries.
        payloads = ((row, {copy["chain_id"].split("_", 1)[0]: structures[copy["chain_id"].split("_", 1)[0]]
                           for copy in row["ranked_copies"] if copy["chain_id"].split("_", 1)[0] in structures},
                     binding["raw_roots"], {copy["chain_id"]: availability[copy["chain_id"]]
                                            for copy in row["ranked_copies"] if copy["chain_id"] in availability})
                    for row in pending)
        try:
            with ProcessPoolExecutor(max_workers=args.workers) as pool:
                for result in pool.map(inspect_sequence, payloads, chunksize=1):
                    if result.get("status") == "sources_needed":
                        source_requests.append(result)
                        continue
                    write_json(records_dir / (result["sequence_sha256"] + ".json"), result)
                    completed[result["sequence_sha256"]] = result
                    if len(completed) % 100 == 0 or len(completed) == len(plans):
                        status = {"status": "running", "completed_sequence_count": len(completed),
                                  "source_sequence_count": len(plans), "updated_at": _now()}
                        progress.write(json.dumps(status) + "\n")
                        progress.flush()
                        os.fsync(progress.fileno())
                        print(json.dumps(status), flush=True)
        except BaseException as exc:
            progress.write(json.dumps({"status": "interrupted_or_failed", "updated_at": _now(),
                                       "completed_sequence_count": len(completed),
                                       "error_type": type(exc).__name__, "error": str(exc)}) + "\n")
            progress.flush()
            os.fsync(progress.fileno())
            raise
    _check_bindings(plan["inputs"] + catalog["inputs"] + [plan_binding, sequence_binding])
    validate_qualification_predicate(predicate)
    if availability_path is not None:
        validate_file_binding(binding["source_availability_certificate"])
        require(validate_source_availability(availability_path, plans, binding["raw_roots"]) == availability,
                "Source availability changed during qualification")
    if source_requests:
        request_path = output / ("sources-needed-" + _now().replace(":", "-") + ".jsonl")
        write_rows(request_path, source_requests)
        return {"status": "sources_needed", "counts": {"completed_sequence_count": len(completed),
                "pending_sequence_count": len(source_requests)}, "requests": file_record(request_path)}
    outcomes = [completed[row["sequence_sha256"]] for row in plans]
    counts = summarize(plans, outcomes, source_availability=availability)
    require(not plan.get("require_all_groups", True) or not counts["empty_group_ids"],
            "Qualification exhausted a required training group")
    qualified = []
    for row, outcome in zip(plans, outcomes, strict=True):
        if outcome["selected_chain_id"] is not None:
            selected = next(copy for copy in row["ranked_copies"] if copy["chain_id"] == outcome["selected_chain_id"])
            qualified.append(selected | {"sequence_sha256": row["sequence_sha256"],
                                         "reference_weight": 1.0 / counts["qualified_sequence_count"],
                                         "source_geometry": outcome["attempts"][-1]["source_geometry"],
                                         "crop_geometry": outcome["attempts"][-1]["crop_geometry"],
                                         "qualification": outcome["attempts"][-1]})
    outputs = {"outcomes": outcomes, "qualified-chains": qualified}
    for name, rows in outputs.items():
        path = output / (name + ".jsonl")
        if path.exists():
            require(read_rows(path) == rows, "Existing population export differs from durable sequence results")
        else:
            write_rows(path, rows)
    receipt = {"schema_version": 1, "status": "complete", "policy": POLICY,
               "reference_weighting": "uniform_per_source_qualified_exact_sequence",
               "copy_selection": "first_passing_copy_in_frozen_quality_rank",
               "plan": plan_binding, "sequence_plan": sequence_binding,
               "qualification_bindings": file_record(binding_path), "predicate": predicate,
               "source_catalog": file_record(args.source_catalog),
               "outcomes": file_record(output / "outcomes.jsonl"),
               "qualified_chains": file_record(output / "qualified-chains.jsonl"),
               "counts": counts, "completed_at": _now()}
    write_json(receipt_path, receipt)
    return validate_receipt(receipt_path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--source-catalog", type=Path, required=True,
                        help="Complete pinned historical source catalog; acquisition errors are fatal.")
    parser.add_argument("--raw-root", type=Path, action="append", required=True)
    parser.add_argument("--source-availability-certificate", type=Path,
                        help="Optional complete raw-acquisition proof; uncertified missing sources stay pending.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    require(1 <= args.workers <= 8, "Qualification worker count must be between one and eight")
    result = run(args)
    print(json.dumps({"status": result["status"], "counts": result["counts"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
