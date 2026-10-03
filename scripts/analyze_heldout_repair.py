"""Measure public-heldout repair feasibility without changing either manifest.

Every hidden target and every unaffected public target remains fixed. Alternatives
come only from each affected target's original public ownership group, remain
40--256 residues, and must pass both directional sequence barriers. The output
is private, metadata-only feasibility evidence, not qualified replacement data.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from build_full_parent_frame import json_once, verify_barrier, write_once

from nanofold.full_parent_leakage import file_record, read_groups, sequence_id, sha256, validate_parent_frame
from nanofold.mmseqs import iter_batched_mmseqs_hits


def distribution(values: Sequence[float | int]) -> dict[str, Any]:
    ordered = sorted(values)
    if not ordered:
        return {"count": 0}
    return {"count": len(ordered), "minimum": ordered[0], "median": ordered[len(ordered) // 2],
            "maximum": ordered[-1], "mean": sum(ordered) / len(ordered)}


def analyze_candidates(
    records: Sequence[Mapping[str, Any]], inherited_groups: Mapping[str, str],
    frozen: Mapping[str, Sequence[str]], barrier_hits: Iterable[tuple[str, str]],
) -> dict[str, Any]:
    """Derive replacement pools from the complete validated parent barrier."""
    by_id = {row["chain_id"]: row for row in records}
    val, hidden = set(frozen["val"]), set(frozen["hidden_val"])
    if len(by_id) != len(records) or val & hidden or not (val | hidden) <= by_id.keys():
        raise ValueError("Heldout targets must be disjoint members of the complete population")
    if not (val | hidden) <= inherited_groups.keys():
        raise ValueError("Original heldout ownership is incomplete")
    val_by_sequence: dict[str, list[str]] = defaultdict(list)
    hidden_by_sequence: dict[str, list[str]] = defaultdict(list)
    for chain in sorted(val):
        val_by_sequence[sequence_id(by_id[chain]["sequence"])].append(chain)
    for chain in sorted(hidden):
        hidden_by_sequence[sequence_id(by_id[chain]["sequence"])].append(chain)
    all_sequences = {sequence_id(row["sequence"]) for row in records}
    hidden_hit_sequences: set[str] = set()
    affected: set[str] = set()
    conflicts: list[dict[str, str]] = []
    for query, target in barrier_hits:
        if query not in val | hidden or target not in all_sequences:
            raise ValueError("Barrier hit escapes the complete frozen input frame")
        if query in hidden:
            hidden_hit_sequences.add(target)
            for public_chain in val_by_sequence.get(target, []):
                affected.add(public_chain)
                conflicts.append({"query_split": "hidden_val", "query": query,
                                  "target_split": "val", "target": public_chain})
        else:
            for hidden_chain in hidden_by_sequence.get(target, []):
                affected.add(query)
                conflicts.append({"query_split": "val", "query": query,
                                  "target_split": "hidden_val", "target": hidden_chain})
    public_groups = {inherited_groups[chain] for chain in val}
    hidden_groups = {inherited_groups[chain] for chain in hidden}
    if len(public_groups) != len(val) or public_groups & hidden_groups:
        raise ValueError("Original heldouts must own distinct groups without cross-split ownership")
    affected_by_group = {inherited_groups[chain]: chain for chain in affected}
    fixed = hidden | (val - affected)
    fixed_pdbs = {chain.split("_", 1)[0].lower() for chain in fixed}
    fixed_sequences = {by_id[chain]["sequence"] for chain in fixed}
    raw_pools: dict[str, list[dict[str, Any]]] = {group: [] for group in affected_by_group}
    rejections: dict[str, Counter[str]] = {group: Counter() for group in affected_by_group}
    for row in records:
        chain = row["chain_id"]
        group = inherited_groups.get(chain)
        if group not in affected_by_group:
            continue
        reason = None
        if chain in val:
            reason = "original_public_target"
        elif not 40 <= row["length"] <= 256:
            reason = "outside_short_evaluation_length"
        elif chain.split("_", 1)[0].lower() in fixed_pdbs:
            reason = "same_pdb_as_fixed_heldout"
        elif row["sequence"] in fixed_sequences:
            reason = "same_sequence_as_fixed_heldout"
        elif sequence_id(row["sequence"]) in hidden_hit_sequences:
            reason = "hidden_query_homology"
        if reason is not None:
            rejections[group][reason] += 1
        else:
            raw_pools[group].append({"chain_id": chain, "sequence_id": sequence_id(row["sequence"]),
                                     "length": row["length"], "resolution": row["resolution"]})
    groups = []
    for group, chain in sorted(affected_by_group.items()):
        candidates = sorted(raw_pools[group], key=lambda row: (row["resolution"], -row["length"], row["chain_id"]))
        groups.append({"original_group": group, "original_public_target": chain,
                       "original_target_length": by_id[chain]["length"],
                       "candidates_passing_hidden_query_direction": candidates,
                       "candidate_chain_count": len(candidates),
                       "candidate_sequence_count": len({row["sequence_id"] for row in candidates}),
                       "rejection_counts": dict(sorted(rejections[group].items()))})
    sequences = {row["sequence_id"] for group in groups for row in group["candidates_passing_hidden_query_direction"]}
    return {
        "status": "reverse_search_required" if sequences else "no_reverse_candidates",
        "scope": "Original-group alternatives are metadata-eligible only. No manifests or assets changed. Source and label qualification remains required.",
        "fixed_hidden_count": len(hidden), "fixed_unaffected_public_count": len(val - affected),
        "affected_public_count": len(affected), "affected_group_count": len(groups),
        "groups_with_forward_safe_alternatives": sum(bool(group["candidate_chain_count"]) for group in groups),
        "groups_without_forward_safe_alternatives": sum(not group["candidate_chain_count"] for group in groups),
        "candidate_chain_count": sum(group["candidate_chain_count"] for group in groups),
        "candidate_sequence_count": len(sequences),
        "candidate_length_distribution": distribution([row["length"] for group in groups for row in group["candidates_passing_hidden_query_direction"]]),
        "original_conflict_target_length_distribution": distribution([by_id[chain]["length"] for chain in affected]),
        "conflicts": sorted(conflicts, key=lambda row: (row["query_split"], row["query"], row["target"])),
        "groups": groups,
    }


def apply_reverse_hits(report: Mapping[str, Any], hits: Iterable[tuple[str, str]], hidden_sequences: set[str]) -> dict[str, Any]:
    """Remove alternatives whose own query-covered alignment reaches hidden."""
    result = json.loads(json.dumps(report))
    expected_queries = {row["sequence_id"] for group in result["groups"] for row in group["candidates_passing_hidden_query_direction"]}
    failed_queries: set[str] = set()
    for query, target in hits:
        if query not in expected_queries or target not in hidden_sequences:
            raise ValueError("Reverse hit escapes the declared candidate/hidden query frame")
        failed_queries.add(query)
    for group in result["groups"]:
        safe = [row for row in group["candidates_passing_hidden_query_direction"] if row["sequence_id"] not in failed_queries]
        group.update(candidates_passing_both_directions=safe, both_direction_chain_count=len(safe),
                     both_direction_sequence_count=len({row["sequence_id"] for row in safe}))
    result.update(status="metadata_feasibility_complete", reverse_failed_sequence_count=len(failed_queries),
                  groups_with_both_direction_alternatives=sum(bool(group["both_direction_chain_count"]) for group in result["groups"]),
                  groups_without_both_direction_alternatives=sum(not group["both_direction_chain_count"] for group in result["groups"]),
                  both_direction_candidate_chains=sum(group["both_direction_chain_count"] for group in result["groups"]))
    return result


def conflict_alignment_distributions(report: Mapping[str, Any], records: Sequence[Mapping[str, Any]],
                                     barrier: Mapping[str, Any]) -> dict[str, Any]:
    """Summarize already-verified native alignments for the frozen conflicts."""
    by_id = {row["chain_id"]: row for row in records}
    directions = {(row["query"], sequence_id(by_id[row["target"]]["sequence"])):
                  row["query_split"] + "_query_to_" + row["target_split"] + "_target" for row in report["conflicts"]}
    collected: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    seen = set()
    for job in barrier["jobs"]:
        path = Path(job["m8_path"])
        if sha256(path) != job["m8_sha256"]:
            raise ValueError("Verified conflict alignments changed before summarization")
        for line in path.read_text().splitlines():
            columns = line.split("\t")
            pair = columns[0], columns[1]
            if pair not in directions:
                continue
            if pair in seen:
                raise ValueError("Conflict alignments repeat a query/target pair")
            seen.add(pair)
            values = collected[directions[pair]]
            values["sequence_identity_percent"].append(float(columns[2]))
            values["query_coverage_fraction"].append(float(columns[4]))
            values["target_coverage_fraction"].append(float(columns[5]))
        if sha256(path) != job["m8_sha256"]:
            raise ValueError("Verified conflict alignments changed during summarization")
    if seen != set(directions):
        raise ValueError("Native alignment summary omits a derived heldout conflict")
    return {direction: {key: distribution(values) for key, values in fields.items()}
            for direction, fields in sorted(collected.items())}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frame-receipt", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True, help="New private output directory; never an official manifest directory")
    parser.add_argument("--reverse-receipt", type=Path, help="Optional completed native query-candidate/target-hidden search receipt")
    args = parser.parse_args()
    frame = validate_parent_frame(args.frame_receipt)
    receipt = json.loads(args.frame_receipt.read_text())
    report = analyze_candidates(frame["records"], read_groups(Path(frame["source_bindings"]["inherited_groups_tsv"]["path"])),
                                frame["frozen_splits"], iter_batched_mmseqs_hits(receipt["barrier"], min_seq_id=0.3, coverage=0.8, coverage_mode=2))
    report["conflicting_alignment_distributions"] = conflict_alignment_distributions(report, frame["records"], receipt["barrier"])
    output = args.out_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    sequences = {sequence_id(row["sequence"]): row["sequence"] for row in frame["records"]}
    candidates = {row["sequence_id"] for group in report["groups"] for row in group["candidates_passing_hidden_query_direction"]}
    by_id = {row["chain_id"]: row for row in frame["records"]}
    hidden_sequences = {sequence_id(by_id[chain]["sequence"]) for chain in frame["frozen_splits"]["hidden_val"]}
    query, source = output / "candidate-queries.fasta", output / "hidden-targets.fasta"
    for path, ids in ((query, candidates), (source, hidden_sequences)):
        write_once(path, "".join(f">{key}\n{sequences[key]}\n" for key in sorted(ids)))
    plan = {"schema_version": 1, "frame_receipt": file_record(args.frame_receipt), "analyzer": file_record(Path(__file__)),
            "query": file_record(query), "source": file_record(source), "query_count": len(candidates), "source_count": len(hidden_sequences),
            "min_seq_id": 0.3, "coverage": 0.8, "coverage_mode": 2,
            "required_search": "Exhaustive native no-prefilter query-candidate/target-hidden; no truncating count limits"}
    plan_path = output / "plan.json"
    json_once(plan_path, plan)
    report["inputs"] = {"plan": file_record(plan_path)}
    json_once(output / "forward-feasibility.json", report)
    if args.reverse_receipt is not None:
        reverse = {**json.loads(args.reverse_receipt.read_text()), "receipt_path": str(args.reverse_receipt.resolve()),
                   "receipt_sha256": sha256(args.reverse_receipt)}
        verify_barrier(reverse, query, source, sha256(plan_path))
        report = apply_reverse_hits(report, iter_batched_mmseqs_hits(reverse, min_seq_id=0.3, coverage=0.8, coverage_mode=2), hidden_sequences)
        report["inputs"]["reverse_receipt"] = file_record(args.reverse_receipt)
        json_once(output / "complete-feasibility.json", report)
    print(json.dumps({key: value for key, value in report.items() if key not in {"inputs", "groups", "conflicts"}}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
