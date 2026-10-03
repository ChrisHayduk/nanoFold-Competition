"""Durable source qualification for a uniformly weighted sequence population.

An exact sequence enters the reference only after one of its deposited copies
passes the pinned structure and raw-query checks. Transport errors never count
as biological failures. The sequence frame and copy ordering are frozen before
qualification; neither availability nor previous qualification selects it.
"""
from __future__ import annotations

import hashlib
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable

from nanofold.representative_eligibility import (
    RAW_QUERY_REJECTION,
    STRUCTURE_REJECTIONS,
    THRESHOLDS,
    _raw_source,
    file_record,
    historical_source_catalog,
    inspect_candidate,
    predicate_record,
    read_json,
    require,
    sha256_file,
)

POLICY = "uniform_qualified_training_sequences_v1"


def validate_file_binding(binding: dict[str, Any]) -> None:
    require(set(binding) in ({"path", "sha256"}, {"path", "sha256", "bytes"}), "Unknown file binding fields")
    path = Path(binding["path"])
    require(file_record(path) == {"path": binding["path"], "sha256": binding["sha256"]},
            "Bound source file changed")
    require("bytes" not in binding or path.stat().st_size == binding["bytes"], "Bound source byte count changed")


def validate_source_catalog(catalog: dict[str, Any]) -> None:
    require(catalog.get("status") == "complete", "Historical source catalog is incomplete")
    for binding in catalog["inputs"]:
        validate_file_binding(binding)
    roots = {Path(row["path"]).parents[1] for row in catalog["inputs"]
             if Path(row["path"]).parts[-2:] == ("nanofold", "structure_metadata.py")}
    require(len(roots) == 1, "Historical catalog must bind one explicit acquisition code root")
    root = roots.pop()
    code_paths = {str(root / "nanofold/structure_metadata.py"), str(root / "scripts/fetch_openproteinset_mmcif.py")}
    require(code_paths <= {row["path"] for row in catalog["inputs"]}, "Historical catalog lacks acquisition code pins")
    relocation = None
    if catalog.get("source_relocations") is not None:
        validate_file_binding(catalog["source_relocations"])
        mapping = read_json(Path(catalog["source_relocations"]["path"]))
        require(mapping.get("status") == "verified_byte_identical_relocation", "Source relocation is not verified")
        relocation = {}
        for row in mapping["files"]:
            require(row["original_path"] not in relocation and row["sha256"] == row["original_sha256"],
                    "Source relocation has duplicate origins or changed bytes")
            relocation[row["original_path"]] = row["path"]
    sources, members = historical_source_catalog(
        [Path(row["path"]) for row in catalog["historical_sources"]["retrieval_receipts"]],
        acquisition_code_root=root, source_path_map=relocation,
    )
    require(sources == catalog["historical_sources"] and members == catalog["members"],
            "Historical catalog differs from its owning acquisition receipts")


def qualification_predicate() -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    return {"inspection": predicate_record(),
            "driver": file_record(root / "scripts/qualify_structural_population.py"),
            "journal": file_record(Path(__file__)),
            "geometry": file_record(root / "nanofold/structural_geometry.py")}


def validate_qualification_predicate(record: dict[str, Any]) -> None:
    """Allow only byte-identical implementation relocation, never data drift."""
    current = qualification_predicate()
    require(set(record) == set(current), "Qualification predicate structure differs")
    require(record["inspection"] == current["inspection"],
            "Qualification inspection code or runtime differs from its frozen execution")
    for name in ("driver", "journal", "geometry"):
        require(isinstance(record[name], dict) and set(record[name]) == {"path", "sha256"},
                "Qualification implementation binding differs")
        validate_file_binding(record[name])
        validate_file_binding(current[name])
        require(record[name]["sha256"] == current[name]["sha256"],
                "Qualification implementation bytes differ from its frozen execution")


def write_json(path: Path, value: Any) -> None:
    """Publish a fresh durable artifact without replacing another writer."""
    data = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    with temporary.open("xb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.link(temporary, path)
    finally:
        temporary.unlink()


def read_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open() as handle:
        for line in handle:
            require(line.endswith("\n"), "Truncated population journal row")
            value = json.loads(line)
            require(isinstance(value, dict), "Population row must be an object")
            rows.append(value)
    return rows


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("x") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def sequence_plan(
    records: list[dict[str, Any]], *, rank: Callable[[dict[str, Any]], Any],
) -> list[dict[str, Any]]:
    """Partition every eligible chain, deduplicating only identical sequences."""
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen = set()
    for record in records:
        chain = record["chain_id"]
        sequence = record["sequence"]
        require(chain not in seen and bool(sequence), "Duplicate chain or empty sequence in population")
        seen.add(chain)
        require(record["length"] == len(sequence), "Sequence length differs from metadata")
        groups[sequence].append(record)
    output = []
    for sequence, members in groups.items():
        require(len({member["group_id"] for member in members}) == 1,
                "Identical sequence crosses immutable split groups")
        output.append({"sequence_sha256": hashlib.sha256(sequence.encode()).hexdigest(),
                       "sequence": sequence, "group_id": members[0]["group_id"],
                       "ranked_copies": sorted(members, key=rank)})
    return sorted(output, key=lambda row: row["sequence_sha256"])


def validate_outcome(plan: dict[str, Any], outcome: dict[str, Any], *, verify_sources: bool) -> None:
    """Reject skipped candidates, partial exhaustion, and stale resumed sources."""
    require(outcome.get("sequence_sha256") == plan["sequence_sha256"]
            and outcome.get("group_id") == plan["group_id"], "Outcome differs from sequence plan")
    attempts = outcome.get("attempts")
    if not isinstance(attempts, list) or not 0 < len(attempts) <= len(plan["ranked_copies"]):
        raise ValueError("Qualification outcome has invalid attempt count")
    for index, attempt in enumerate(attempts):
        candidate = plan["ranked_copies"][index]
        require(attempt.get("chain_id") == candidate["chain_id"] and attempt.get("rank") == index
                and attempt.get("sequence_sha256") == plan["sequence_sha256"],
                "Qualification did not evaluate the exact ranked prefix")
        require(attempt.get("status") in {"passed", "failed"}, "Incomplete qualification outcome")
        require(index == len(attempts) - 1 or attempt["status"] == "failed",
                "Qualification continued after the first passing copy")
        require(attempt.get("structure_status") in {"passed", "failed"}, "Unrecognized structure status")
        statistics = attempt.get("projection_statistics")
        if statistics is None:
            require(attempt.get("structure_rejection") in STRUCTURE_REJECTIONS
                    and attempt["structure_status"] == "failed", "Unrecognized structural rejection")
        else:
            passed = all(statistics[key] >= threshold for key, threshold in THRESHOLDS.items())
            require(attempt["structure_status"] == ("passed" if passed else "failed"),
                    "Structure status differs from strict projection thresholds")
        if attempt["structure_status"] == "passed":
            require(attempt.get("raw_status") in {"passed", "failed"} and isinstance(attempt.get("raw"), dict),
                    "Structure-qualified candidate lacks raw query evidence")
            raw_passed = attempt["raw"]["query_sha256"] == plan["sequence_sha256"]
            require(attempt["raw_status"] == ("passed" if raw_passed else "failed")
                    and attempt["status"] == ("passed" if raw_passed else "failed"),
                    "Raw status differs from exact sequence identity")
            require(raw_passed or attempt.get("raw_rejection") == RAW_QUERY_REJECTION,
                    "Unrecognized raw source rejection")
        if attempt["status"] == "passed":
            require(attempt["structure_status"] == attempt.get("raw_status") == "passed",
                    "Passing candidate lacks complete source qualification")
            full_raw = attempt.get("raw_full_audit")
            require(isinstance(full_raw, dict) and full_raw.get("sha256") == attempt["raw"]["sha256"]
                    and full_raw.get("bytes") == attempt["raw"]["bytes"]
                    and full_raw.get("path") == attempt["raw"]["path"]
                    and full_raw.get("query_length") == len(plan["sequence"])
                    and type(full_raw.get("rows")) is int and full_raw["rows"] > 0
                    and type(full_raw.get("aligned_width")) is int and full_raw["aligned_width"] >= len(plan["sequence"]),
                    "Passing sequence lacks complete raw MSA validation")
        if verify_sources:
            for source in (attempt["structure"], attempt.get("raw")):
                if source is not None:
                    require(sha256_file(Path(source["path"])) == source["sha256"],
                            "Previously qualified source bytes changed")
            replay = inspect_candidate(chain=candidate["chain_id"], sequence=plan["sequence"], rank=index,
                                       structure=attempt["structure"],
                                       raw_path=Path(attempt["raw"]["path"]) if attempt.get("raw") else None,
                                       include_geometry=True, full_msa_validation=True)
            require(replay == attempt, "Stored qualification differs from current source replay")
    last = attempts[-1]
    selected = last["chain_id"] if last["status"] == "passed" else None
    require(outcome.get("selected_chain_id") == selected, "Selected copy differs from its qualification")
    require(selected is not None or len(attempts) == len(plan["ranked_copies"]),
            "Sequence was excluded without exhausting its eligible copies")


def validate_outcome_source_membership(
    outcome: dict[str, Any], catalog: dict[str, Any], raw_roots: list[str],
) -> None:
    """Bind replayed source bytes to their owning PDB and chain lookup."""
    for attempt in outcome["attempts"]:
        chain = attempt["chain_id"]
        require(attempt["structure"] == catalog["members"][chain.split("_", 1)[0]],
                "Inspected structure differs from historical acquisition catalog")
        if attempt.get("raw") is not None:
            require(Path(attempt["raw"]["path"]).resolve()
                    == _raw_source([Path(root) for root in raw_roots], chain).resolve(),
                    "Inspected raw source differs from frozen chain lookup")


def qualify_sequence(
    plan: dict[str, Any], inspect: Callable[[dict[str, Any], int], dict[str, Any]],
) -> dict[str, Any]:
    attempts = []
    selected = None
    for index, candidate in enumerate(plan["ranked_copies"]):
        result = inspect(candidate, index)
        attempts.append(result)
        if result["status"] == "passed":
            selected = candidate["chain_id"]
            break
    result = {"sequence_sha256": plan["sequence_sha256"], "group_id": plan["group_id"],
              "selected_chain_id": selected, "attempts": attempts}
    validate_outcome(plan, result, verify_sources=False)
    return result


def summarize(plans: list[dict[str, Any]], outcomes: list[dict[str, Any]]) -> dict[str, Any]:
    require(len(plans) == len(outcomes), "Qualification did not exhaust the sequence frame")
    by_hash = {outcome["sequence_sha256"]: outcome for outcome in outcomes}
    require(len(by_hash) == len(outcomes), "Repeated sequence outcome")
    for plan in plans:
        require(plan["sequence_sha256"] in by_hash, "Sequence outcome is missing")
        validate_outcome(plan, by_hash[plan["sequence_sha256"]], verify_sources=False)
    passed = [outcome for outcome in outcomes if outcome["selected_chain_id"] is not None]
    group_count = Counter(outcome["group_id"] for outcome in passed)
    empty = sorted({plan["group_id"] for plan in plans} - group_count.keys())
    return {"source_chain_count": sum(len(plan["ranked_copies"]) for plan in plans),
            "source_sequence_count": len(plans), "qualified_sequence_count": len(passed),
            "exhausted_sequence_count": len(plans) - len(passed),
            "attempted_chain_count": sum(len(outcome["attempts"]) for outcome in outcomes),
            "required_group_count": len({plan["group_id"] for plan in plans}),
            "qualified_group_count": len(group_count), "empty_group_ids": empty}


def validate_receipt(path: Path, *, verify_sources: bool = False) -> dict[str, Any]:
    receipt = read_json(path)
    require(receipt.get("status") == "complete" and receipt.get("policy") == POLICY,
            "Population receipt is not complete")
    for key in ("plan", "sequence_plan", "outcomes", "qualified_chains", "qualification_bindings", "source_catalog"):
        binding = receipt[key]
        require(file_record(Path(binding["path"])) == binding, "Population output binding changed")
    plan = read_json(Path(receipt["plan"]["path"]))
    require(plan["sequence_plan"] == receipt["sequence_plan"], "Receipt and plan bind different sequence frames")
    bindings = read_json(Path(receipt["qualification_bindings"]["path"]))
    require(bindings["plan"] == receipt["plan"] and bindings["sequence_plan"] == receipt["sequence_plan"]
            and bindings["predicate"] == receipt["predicate"],
            "Qualification bindings differ from their frozen execution")
    validate_qualification_predicate(receipt["predicate"])
    catalog = read_json(Path(receipt["source_catalog"]["path"]))
    validate_source_catalog(catalog)
    require(catalog["status"] == "complete"
            and catalog["historical_sources"]["archive_pin"] == bindings["historical_archive"],
            "Qualification source catalog differs from pinned historical archive")
    for binding in catalog["inputs"]:
        validate_file_binding(binding)
    for binding in plan["inputs"]:
        validate_file_binding(binding)
    rows = read_rows(Path(receipt["sequence_plan"]["path"]))
    outcomes = read_rows(Path(receipt["outcomes"]["path"]))
    require(summarize(rows, outcomes) == receipt["counts"], "Population summary differs from complete outcomes")
    require(not plan.get("require_all_groups", True) or not receipt["counts"]["empty_group_ids"],
            "Qualification lost a required training group")
    selected = {row["sequence_sha256"]: row["selected_chain_id"] for row in outcomes if row["selected_chain_id"]}
    qualified = read_rows(Path(receipt["qualified_chains"]["path"]))
    require(len(qualified) == len(selected)
            and {row["sequence_sha256"]: row["chain_id"] for row in qualified} == selected,
            "Qualified chain export differs from complete outcomes")
    by_outcome = {outcome["sequence_sha256"]: outcome for outcome in outcomes}
    by_qualified = {row["sequence_sha256"]: row for row in qualified}
    for row in rows:
        outcome = by_outcome[row["sequence_sha256"]]
        chain = outcome["selected_chain_id"]
        if chain is not None:
            candidate = next(copy for copy in row["ranked_copies"] if copy["chain_id"] == chain)
            expected = candidate | {"sequence_sha256": row["sequence_sha256"],
                                    "reference_weight": 1.0 / len(qualified),
                                    "source_geometry": outcome["attempts"][-1]["source_geometry"],
                                    "crop_geometry": outcome["attempts"][-1]["crop_geometry"],
                                    "qualification": outcome["attempts"][-1]}
            require(by_qualified[row["sequence_sha256"]] == expected,
                    "Qualified sequence fields differ from frozen plan and inspected source")
        validate_outcome_source_membership(outcome, catalog, bindings["raw_roots"])
    if verify_sources:
        for row in rows:
            validate_outcome(row, by_outcome[row["sequence_sha256"]], verify_sources=True)
    return receipt
