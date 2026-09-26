"""Qualify original sampling units from pinned historical structures and raw A3Ms.

Audit each complete unit's deterministic ranked prefix until its first passing
member. Missing sources, acquisition errors and unrecognized parser failures
abort qualification. Only complete group exhaustion permits an approved
exclusion. All outputs and source paths are private maintainer evidence.
Each fresh output also reserves a sibling ``<output>.diagnostics`` directory
with durable unit outcomes and failure details. These are diagnostic evidence,
never qualification acceptance or checkpoints that can skip a future audit.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
for root in (REPO_ROOT, Path(__file__).resolve().parent):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from build_homology_groups import validate_group_receipt
from build_manifests import (
    _build_candidates,
    _load_cluster_tsv,
    _load_structure_metadata,
    _original_unit_memberships,
)

from nanofold.representative_eligibility import (
    POLICY,
    file_record,
    historical_source_catalog,
    inspect_candidate,
    population_record,
    predicate_record,
    read_json,
    require,
    sha256_file,
    validate_exclusion_catalog,
    validate_representative_eligibility_receipt,
)


def audit_unit(
    unit_id: str, ranked: list[str], *, sequences: dict[str, str], full_group: str,
    historical_members: dict[str, dict[str, str]], raw_roots: list[Path],
) -> dict[str, Any]:
    outcomes = []
    selected = None
    for rank, chain in enumerate(ranked):
        pdb = chain.split("_", 1)[0]
        require(pdb in historical_members, "Ranked candidate lacks a pinned historical acquisition")
        row = inspect_candidate(
            chain=chain, sequence=sequences[chain], rank=rank, structure=historical_members[pdb], raw_roots=raw_roots,
        )
        outcomes.append(row)
        if row["status"] == "passed":
            selected = chain
            break
    return {"unit_id": unit_id, "full_group_id": full_group, "ranked_members": ranked,
            "outcomes": outcomes, "selected_chain_id": selected}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("chain-data-cache", "structure-metadata", "cluster-tsv", "split-group-tsv", "split-group-receipt",
                 "approved-group-exclusions", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--historical-retrieval-receipt", type=Path, action="append", required=True)
    parser.add_argument("--raw-root", type=Path, action="append", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--min-len", type=int, default=40)
    parser.add_argument("--max-len", type=int, default=256)
    parser.add_argument("--max-resolution", type=float, default=3.0)
    parser.add_argument("--require-monomer", action="store_true")
    parser.add_argument("--min-seq-id", type=float, default=0.3)
    parser.add_argument("--coverage", type=float, default=0.8)
    return parser.parse_args(argv)


def _write_diagnostic(path: Path, value: dict[str, Any]) -> None:
    with path.open("x") as handle:
        json.dump(value, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def run(args: argparse.Namespace) -> dict[str, Any]:
    os.umask(0o077)
    output = args.output.resolve()
    require(not output.exists(), "Eligibility output must be a fresh private path")
    require(not output.is_relative_to(REPO_ROOT) or output.is_relative_to(REPO_ROOT / ".nanofold_private"),
            "Repository eligibility evidence must remain under the ignored private root")
    input_bindings = [file_record(path) for path in (
        args.chain_data_cache, args.structure_metadata, args.cluster_tsv, args.split_group_tsv,
        args.split_group_receipt, args.approved_group_exclusions, *args.historical_retrieval_receipt,
    )]
    approved = validate_exclusion_catalog(args.approved_group_exclusions)
    cache = read_json(args.chain_data_cache)
    accepted, metadata = _load_structure_metadata(args.structure_metadata, require_schema_version=2)
    candidates, _rejections, _source = _build_candidates(
        cache, min_len=args.min_len, max_len=args.max_len, max_resolution=args.max_resolution,
        structure_metadata=metadata, structure_metadata_accepted=accepted,
        sampling_policy="cath-v3", require_monomer=args.require_monomer,
    )
    ids = {candidate.chain_id for candidate in candidates}
    clusters = _load_cluster_tsv(args.cluster_tsv, ids)
    groups = _load_cluster_tsv(args.split_group_tsv, ids)
    validate_group_receipt(
        args.split_group_receipt, split_group_tsv=args.split_group_tsv,
        chain_data_cache=args.chain_data_cache, structure_metadata=args.structure_metadata,
        cluster_map=clusters, candidate_ids=ids, min_seq_id=args.min_seq_id, coverage=args.coverage,
    )
    units = {unit: [candidate.chain_id for candidate in ranked]
             for unit, ranked in _original_unit_memberships(candidates, clusters, seed=args.seed).items()}
    population = population_record(
        chain_data_cache_path=args.chain_data_cache, structure_metadata_path=args.structure_metadata,
        cluster_map=clusters, split_group_map=groups, split_group_receipt_path=args.split_group_receipt,
        original_units=units, seed=args.seed,
    )
    predicate = predicate_record()
    sources, members = historical_source_catalog(args.historical_retrieval_receipt)
    roots = [path.resolve() for path in args.raw_root]
    sequences = {candidate.chain_id: candidate.sequence for candidate in candidates}
    output.parent.mkdir(parents=True, exist_ok=True)
    diagnostics = output.with_name(output.name + ".diagnostics")
    require(not diagnostics.exists() and not diagnostics.is_symlink(), "Eligibility diagnostics require a fresh private path")
    diagnostics.mkdir()
    _write_diagnostic(diagnostics / "bindings.json", {
        "schema_version": 1, "artifact_type": "representative_eligibility_diagnostic_bindings",
        "scope": "private_diagnostics_not_qualification_acceptance",
        "inputs": input_bindings, "population": population, "predicate": predicate,
        "historical_sources": sources, "raw_roots": [str(path) for path in roots],
    })
    journal = diagnostics / "units.jsonl"
    records = []
    phase = "auditing_units"
    mismatch: dict[str, Any] = {}
    try:
        with journal.open("x") as handle:
            for unit, ranked in sorted(units.items()):
                record = audit_unit(
                    unit, ranked, sequences=sequences, full_group=groups[ranked[0]], historical_members=members, raw_roots=roots,
                )
                handle.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
                records.append(record)
                if len(records) % 100 == 0:
                    print(json.dumps({"status": "running", "audited_unit_count": len(records), "original_unit_count": len(units)}), flush=True)
        passing = {record["full_group_id"] for record in records if record["selected_chain_id"] is not None}
        empty = set(groups.values()) - passing
        phase = "checking_excluded_groups"
        if empty != set(approved):
            mismatch = {
                "approved_excluded_groups": sorted(approved), "evaluated_empty_groups": sorted(empty),
                "newly_empty_groups": sorted(empty - set(approved)),
                "previously_excluded_now_nonempty_groups": sorted(set(approved) - empty),
                "evaluated_passing_group_count": len(passing),
            }
        require(empty == set(approved), "Approved exclusions differ from fully exhausted original groups")
        exhausted_count = sum(record["selected_chain_id"] is None for record in records)
        phase = "checking_input_pins"
        require(predicate_record() == predicate, "Eligibility predicate changed during qualification")
        require(all(file_record(Path(record["path"])) == record for record in input_bindings),
                "Eligibility input changed during qualification")
        result: dict[str, Any] = {
            "schema_version": 1, "policy": POLICY, "status": "complete", "population": population,
            "predicate": predicate, "historical_sources": sources,
            "original_structure_metadata": file_record(args.structure_metadata),
            "approved_group_exclusions_sha256": sha256_file(args.approved_group_exclusions), "units": records,
            "counts": {"eligible_unit_count": len(units) - exhausted_count, "exhausted_unit_count": exhausted_count,
                       "excluded_group_count": len(empty), "retained_group_count": len(passing)},
        }
        temporary = output.with_name(output.name + ".incomplete")
        require(not temporary.exists(), "Eligibility staging file already exists")
        with temporary.open("x") as handle:
            handle.write(json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n")
        phase = "validating_receipt"
        validate_representative_eligibility_receipt(
            temporary, approved_group_exclusions_path=args.approved_group_exclusions,
            chain_data_cache_path=args.chain_data_cache, structure_metadata_path=args.structure_metadata,
            cluster_map=clusters, split_group_map=groups, split_group_receipt_path=args.split_group_receipt,
            original_units=units, seed=args.seed,
        )
        phase = "publishing_receipt"
        # Link without replacement so another writer cannot be silently overwritten.
        os.link(temporary, output)
        temporary.unlink()
        print(json.dumps({"status": "complete", **population, **result["counts"]}, sort_keys=True))
        return result
    except BaseException as exc:
        try:
            pin_checks: dict[str, Any] = {"predicate_unchanged": None, "inputs_unchanged": None}
            try:
                pin_checks["predicate_unchanged"] = predicate_record() == predicate
                pin_checks["inputs_unchanged"] = all(file_record(Path(row["path"])) == row for row in input_bindings)
            except Exception as check_error:
                pin_checks["error_type"] = type(check_error).__name__
            _write_diagnostic(diagnostics / "failure.json", {
                "schema_version": 1, "artifact_type": "representative_eligibility_failure_diagnostic",
                "scope": "private_diagnostics_not_qualification_acceptance", "status": "failed",
                "phase": phase, "error_type": type(exc).__name__, "error": str(exc),
                "completed_unit_count": len(records), "original_unit_count": len(units),
                "full_population_evaluated": len(records) == len(units), "pin_checks": pin_checks,
                "units": file_record(journal), "exclusion_mismatch": mismatch,
                "qualification_receipt_published": output.exists(),
            })
        except Exception as diagnostic_error:
            exc.add_note(f"Private failure diagnostics could not be published: {type(diagnostic_error).__name__}")
        raise


def main() -> int:
    try:
        run(parse_args())
    except (OSError, ValueError, KeyError, RuntimeError) as exc:
        # Paths, targets and parser diagnostics may reveal hidden membership.
        print(json.dumps({"status": "failed", "error_type": type(exc).__name__,
                          "reason": "Eligibility source or predicate validation failed; no completed receipt published"}))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
