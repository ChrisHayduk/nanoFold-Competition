"""Deterministic, source-qualified repair of public evaluation membership.

Historical public targets remain protected from training. This policy repairs
the active public cohort without choosing targets using source availability or
model performance. Pending qualification and pending safety evidence block
selection; they are never interpreted as biological rejection.
"""
from __future__ import annotations

import hashlib
import importlib
import json
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from nanofold.full_parent_leakage import file_record, sequence_id, sha256, validate_parent_frame
from nanofold.mmseqs import iter_batched_mmseqs_hits
from nanofold.structural_population import (
    read_rows,
    sequence_plan,
    validate_file_binding,
    validate_outcome,
    validate_outcome_source_membership,
    validate_qualification_predicate,
    validate_source_catalog,
    write_json,
    write_rows,
)

POLICY = "source_qualified_public_heldout_repair_v1"
REQUEST_POLICY = "evaluation_repair_source_request_v1"
IMPLEMENTATION_DEPENDENCIES = (
    "nanofold/full_parent_leakage.py", "nanofold/mmseqs.py", "nanofold/structural_population.py",
    "scripts/build_full_parent_frame.py", "scripts/analyze_heldout_repair.py", "scripts/plan_heldout_repair.py",
    "scripts/build_full_source_metadata.py", "scripts/build_structure_metadata.py",
    "nanofold/structure_metadata.py", "nanofold/cath_boundaries.py",
)
CRITERIA = {
    "unit": "exact_sequence; all ranked deposited copies precede the next sequence",
    "copy_rank": ["finite_positive_resolution_ascending_unknown_last", "seeded_chain_hash"],
    "same_group_sequence_rank": ["first_copy_resolution", "absolute_original_length_difference", "seeded_first_copy_hash"],
    "new_group_rank": ["known_unknown_stratum_mismatch", "topology_multiset_l1", "absolute_domain_count_difference",
                       "original_length_bin_mismatch", "seeded_group_hash"],
    "new_group_sequence_rank": ["structural_rank", "first_copy_resolution", "absolute_original_length_difference", "seeded_first_copy_hash"],
    "slot_order": "seeded_original_public_chain_hash",
    "length_bin_upper_bounds_exclusive": [64, 96, 128, 192],
    "pending_sources": "block; only complete biological rejection advances the rank",
    "safety": "both query-covered directions against hidden, identity0.3/coverage0.8/cov_mode2",
}


def stable_hash(value: str, seed: int = 0, kind: str = "chain") -> str:
    return hashlib.sha256(f"{seed}:{kind}:{value}".encode()).hexdigest()


def resolution_rank(value: Any) -> tuple[int, float]:
    if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value > 0:
        return 0, float(value)
    return 1, 0.0


def length_bin(length: int) -> int:
    if not 40 <= length <= 256:
        raise ValueError("Repair evaluation targets must remain 40--256 residues")
    return sum(length >= boundary for boundary in (64, 96, 128, 192))


def feature_record(metadata: Mapping[str, Any]) -> dict[str, Any]:
    required = {"length", "cath_annotation_status", "cath_topology_counts", "cath_domain_count"}
    if not required <= metadata.keys() or metadata["cath_annotation_status"] not in {"annotated", "unknown"}:
        raise ValueError("Repair ranking requires explicit verified CATH enrichment fields")
    counts = dict(metadata["cath_topology_counts"])
    if any(not isinstance(key, str) or type(value) is not int or value <= 0 for key, value in counts.items()):
        raise ValueError("CATH topology multiplicities must be positive integer counts")
    known = metadata["cath_annotation_status"] == "annotated"
    count = metadata["cath_domain_count"] if known else None
    if known and (not counts or type(count) is not int or count < sum(counts.values())):
        raise ValueError("CATH domain count is inconsistent with known topology multiplicities")
    if not known and (counts or metadata["cath_domain_count"] != 0):
        raise ValueError("Unknown CATH annotation contradicts declared domain counts")
    return {"annotation_stratum": "known" if known else "unknown",
            "topology_counts": dict(sorted(counts.items())) if known else {},
            "domain_count": count, "length_bin": length_bin(metadata["length"])}


def structural_rank(candidate: Mapping[str, Any], target: Mapping[str, Any]) -> tuple[int, int, int, int]:
    left, right = candidate["topology_counts"], target["topology_counts"]
    return (int(candidate["annotation_stratum"] != target["annotation_stratum"]),
            sum(abs(left.get(key, 0) - right.get(key, 0)) for key in left.keys() | right.keys()),
            abs((candidate["domain_count"] or 0) - (target["domain_count"] or 0)),
            int(candidate["length_bin"] != target["length_bin"]))


def sequence_rank(row: Mapping[str, Any], target: Mapping[str, Any], *, same_group: bool, seed: int = 0) -> tuple:
    copy = row["ranked_copies"][0]
    common = (*resolution_rank(copy["resolution"]), abs(copy["length"] - target["length"]),
              stable_hash(copy["chain_id"], seed))
    return common if same_group else (*structural_rank(copy["features"], target["features"]), *common)


def replay_selection(plan: Mapping[str, Any], sequence_plans: Sequence[Mapping[str, Any]],
                     outcomes: Mapping[str, Mapping[str, Any]], safety: Mapping[str, str]) -> dict[str, Any]:
    """Replay the deterministic ranked prefix using independently verified states."""
    seed = plan["seed"]
    indexed = {row["sequence_sha256"]: row for row in sequence_plans}
    if len(indexed) != len(sequence_plans):
        raise ValueError("Repeated exact sequence in repair plan")
    slots = plan["slots"]
    if slots != sorted(slots, key=lambda slot: (stable_hash(slot["original_public_id"], seed, "slot"), slot["original_public_id"])):
        raise ValueError("Repair slots differ from the frozen seeded order")
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in sequence_plans:
        groups[row["group_id"]].append(row)
    used_groups: set[str] = set()
    selected: list[dict[str, Any]] = []
    trace: list[dict[str, Any]] = []
    for slot in slots:
        target = slot["reference"]
        same = [indexed[key] for key in slot["same_group_sequences"]]
        if any(row["group_id"] != slot["original_group"] for row in same):
            raise ValueError("Same-group candidate escapes original ownership")
        tiers: list[tuple[str, str, list[Mapping[str, Any]]]] = []
        if same:
            tiers.append(("same_original_group", slot["original_group"], sorted(same, key=lambda row: sequence_rank(row, target, same_group=True, seed=seed))))
        new_groups = [group for group in plan["new_group_ids"] if group not in used_groups]
        if any(group not in groups for group in new_groups):
            raise ValueError("New replacement group has no frozen candidates")
        new_groups.sort(key=lambda group: (min(structural_rank(row["ranked_copies"][0]["features"], target["features"]) for row in groups[group]),
                                           stable_hash(group, seed, "group"), group))
        tiers.extend(("new_original_group", group, sorted(groups[group], key=lambda row: sequence_rank(row, target, same_group=False, seed=seed))) for group in new_groups)
        chosen = None
        for tier, group, candidates in tiers:
            if group in used_groups:
                continue
            for row in candidates:
                key = row["sequence_sha256"]
                state = safety.get(key)
                if state not in {None, "passed", "failed"}:
                    raise ValueError("Unrecognized candidate safety state")
                if state == "failed":
                    trace.append({"slot": slot["original_public_id"], "sequence_sha256": key, "decision": "safety_rejected"})
                    continue
                outcome = outcomes.get(key)
                if state is None or outcome is None:
                    return {"status": "pending", "selected": selected, "trace": trace,
                            "pending_sequence_sha256": [key], "pending_reason": "safety" if state is None else "qualification"}
                if outcome["selected_chain_id"] is None:
                    trace.append({"slot": slot["original_public_id"], "sequence_sha256": key, "decision": "biological_exhaustion"})
                    continue
                chain = outcome["selected_chain_id"]
                copies = {copy["chain_id"]: copy for copy in row["ranked_copies"]}
                if chain not in copies or outcome.get("group_id") != group or outcome.get("sequence_sha256") != key:
                    raise ValueError("Qualified copy escapes its frozen exact-sequence/group plan")
                chosen = {"original_public_id": slot["original_public_id"], "selected_chain_id": chain,
                          "sequence_sha256": key, "group_id": group, "tier": tier,
                          "selected_copy_feature_drift": copies[chain]["features"] != row["ranked_copies"][0]["features"]}
                break
            if chosen is not None:
                break
        if chosen is None:
            return {"status": "exhausted", "selected": selected, "trace": trace,
                    "unfilled_original_public_id": slot["original_public_id"], "pending_sequence_sha256": []}
        selected.append(chosen)
        used_groups.add(chosen["group_id"])
    return {"status": "complete", "selected": selected, "trace": trace, "pending_sequence_sha256": []}


def derive_plan(frame: Mapping[str, Any], feasibility: Mapping[str, Any], metadata: Mapping[str, Mapping[str, Any]],
                inherited: Mapping[str, str], *, seed: int = 0) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Freeze every eligible source copy before considering local availability."""
    records = {row["chain_id"]: row for row in frame["records"]}
    historical, hidden = (sorted(frame["frozen_splits"][key]) for key in ("val", "hidden_val"))
    affected = {group["original_public_target"] for group in feasibility["groups"]}
    if not affected <= set(historical):
        raise ValueError("Affected public targets escape the historical manifest")
    safe = {copy["chain_id"] for group in feasibility["groups"] for copy in group["candidates_passing_both_directions"]}
    protected_groups = {inherited[chain] for chain in historical + hidden}
    fixed = (set(historical) - affected) | set(hidden)
    fixed_pdb = {chain.split("_", 1)[0].lower() for chain in fixed}
    fixed_sequences = {records[chain]["sequence"] for chain in fixed}
    historical_and_hidden = set(historical + hidden)
    candidates = []
    new_groups = set()
    for chain, record in sorted(records.items()):
        same = chain in safe
        original_group = inherited.get(chain)
        new = (record["group_id"] is not None and original_group is not None and original_group not in protected_groups
               and 40 <= record["length"] <= 256)
        if not same and not new:
            continue
        if (original_group is None or chain in historical_and_hidden or record["sequence"] in fixed_sequences
                or chain.split("_", 1)[0].lower() in fixed_pdb):
            raise ValueError("Candidate pool violates frozen evaluation ownership")
        if chain not in metadata or metadata[chain]["length"] != record["length"]:
            raise ValueError("Candidate structural metadata is missing or mismatched")
        candidates.append({key: record[key] for key in ("chain_id", "sequence", "sequence_sha256", "length", "resolution")}
                          | {"group_id": original_group, "features": feature_record(metadata[chain]),
                             "candidate_tier": "same_original_group" if same else "new_original_group"})
        if new:
            new_groups.add(original_group)
    plans = sequence_plan(candidates, rank=lambda row: (*resolution_rank(row["resolution"]), stable_hash(row["chain_id"], seed), row["chain_id"]))
    indexed_groups: dict[str, list[str]] = defaultdict(list)
    for row in plans:
        indexed_groups[row["group_id"]].append(row["sequence_sha256"])
    slots = [{"original_public_id": chain, "original_group": inherited[chain],
              "reference": {"length": records[chain]["length"], "features": feature_record(metadata[chain])},
              "same_group_sequences": sorted(indexed_groups[inherited[chain]])} for chain in affected]
    slots.sort(key=lambda slot: (stable_hash(slot["original_public_id"], seed, "slot"), slot["original_public_id"]))
    plan = {"schema_version": 1, "policy": POLICY, "seed": seed, "criteria": CRITERIA,
            "historical_public_ids": historical, "hidden_ids": hidden,
            "fixed_public_ids": sorted(set(historical) - affected), "slots": slots,
            "new_group_ids": sorted(new_groups), "candidate_sequence_count": len(plans),
            "candidate_chain_count": len(candidates)}
    return plan, plans


def _bound_json(pin: Mapping[str, Any]) -> dict[str, Any]:
    validate_file_binding(dict(pin))
    data = Path(pin["path"]).read_bytes()
    if hashlib.sha256(data).hexdigest() != pin["sha256"]:
        raise ValueError("Bound JSON changed during reading")
    result = dict(json.loads(data))
    validate_file_binding(dict(pin))
    return result


def _bound_groups(pin: Mapping[str, Any]) -> dict[str, str]:
    validate_file_binding(dict(pin))
    data = Path(pin["path"]).read_bytes()
    if hashlib.sha256(data).hexdigest() != pin["sha256"]:
        raise ValueError("Bound original ownership changed during reading")
    mapping = {}
    for line in data.decode().splitlines():
        if not line or line.startswith("#"):
            continue
        columns = line.split("\t")
        if len(columns) != 2 or columns[1] in mapping:
            raise ValueError("Malformed or repeated original ownership row")
        mapping[columns[1]] = columns[0]
    validate_file_binding(dict(pin))
    return mapping


def _native_result(pin: Mapping[str, Any]) -> dict[str, Any]:
    return {**_bound_json(pin), "receipt_path": pin["path"], "receipt_sha256": pin["sha256"]}


def _validated_inputs(bindings: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, str]]:
    frame_pin = bindings["baseline_frame_receipt"]
    frame = validate_parent_frame(Path(frame_pin["path"]))
    if frame["receipt"] != frame_pin:
        raise ValueError("Baseline frame receipt differs from frozen repair input")
    frame_receipt = _bound_json(frame_pin)
    report = _bound_json(bindings["complete_feasibility"])
    forward_plan = _bound_json(report["inputs"]["plan"])
    if forward_plan["frame_receipt"] != frame_pin:
        raise ValueError("Repair feasibility belongs to a different baseline frame")
    analyze = importlib.import_module("scripts.analyze_heldout_repair")
    native = importlib.import_module("scripts.build_full_parent_frame")
    inherited = _bound_groups(frame["source_bindings"]["inherited_groups_tsv"])
    forward = analyze.analyze_candidates(frame["records"], inherited, frame["frozen_splits"],
                                        iter_batched_mmseqs_hits(frame_receipt["barrier"], min_seq_id=0.3, coverage=0.8, coverage_mode=2))
    sequences = {sequence_id(row["sequence"]): row["sequence"] for row in frame["records"]}
    rows = {row["chain_id"]: row for row in frame["records"]}
    expected_queries = {row["sequence_id"] for group in forward["groups"] for row in group["candidates_passing_hidden_query_direction"]}
    expected_hidden = {sequence_id(rows[chain]["sequence"]) for chain in frame["frozen_splits"]["hidden_val"]}
    for role, ids in (("query", expected_queries), ("source", expected_hidden)):
        pin = forward_plan[role]
        validate_file_binding(pin)
        expected = "".join(f">{key}\n{sequences[key]}\n" for key in sorted(ids))
        if Path(pin["path"]).read_text() != expected:
            raise ValueError("Repair feasibility search omitted or changed candidates")
    reverse = _native_result(report["inputs"]["reverse_receipt"])
    native.verify_barrier(reverse, Path(forward_plan["query"]["path"]), Path(forward_plan["source"]["path"]), report["inputs"]["plan"]["sha256"])
    replay = analyze.apply_reverse_hits(forward, iter_batched_mmseqs_hits(reverse, min_seq_id=0.3, coverage=0.8, coverage_mode=2), expected_hidden)
    if any(report.get(key) != value for key, value in replay.items()):
        raise ValueError("Complete feasibility differs from independently replayed barriers")
    for name in ("structural_metadata", "structural_metadata_receipt", "metadata_source_lock"):
        validate_file_binding(bindings[name])
    metadata_owner = importlib.import_module("scripts.build_full_source_metadata")
    document = metadata_owner.validate_full_source_metadata(
        metadata_path=Path(bindings["structural_metadata"]["path"]),
        receipt_path=Path(bindings["structural_metadata_receipt"]["path"]),
        chain_data_cache=Path(frame["source_bindings"]["chain_data_cache"]["path"]),
        metadata_sources_dir=Path(bindings["metadata_source_lock"]["path"]).parent,
        metadata_source_lock=Path(bindings["metadata_source_lock"]["path"]), replay=True)
    if document["source"]["chain_data_cache_sha256"] != frame["source_bindings"]["chain_data_cache"]["sha256"]:
        raise ValueError("Repair structural metadata uses a different underlying sequence cache")
    metadata = {row["chain_id"]: row for row in document["chains"]}
    if len(metadata) != len(document["chains"]):
        raise ValueError("Repair structural metadata repeats chains")
    for pin in [*bindings.values(), *frame["source_bindings"].values()]:
        validate_file_binding(pin)
    return frame, replay, metadata, inherited


def build_plan(*, baseline_frame_receipt: Path, complete_feasibility: Path, structural_metadata: Path,
               structural_metadata_receipt: Path, metadata_source_lock: Path, output: Path, seed: int = 0) -> Path:
    bindings = {"baseline_frame_receipt": file_record(baseline_frame_receipt),
                "complete_feasibility": file_record(complete_feasibility), "structural_metadata": file_record(structural_metadata),
                "structural_metadata_receipt": file_record(structural_metadata_receipt), "metadata_source_lock": file_record(metadata_source_lock)}
    frame, report, metadata, inherited = _validated_inputs(bindings)
    plan, sequences = derive_plan(frame, report, metadata, inherited, seed=seed)
    output.mkdir(parents=True, exist_ok=True)
    sequence_path = output / "sequence-plan.jsonl"
    write_rows(sequence_path, sequences)
    root = Path(__file__).resolve().parents[1]
    dependencies = {name: file_record(root / name) for name in IMPLEMENTATION_DEPENDENCIES}
    plan.update(source_bindings=bindings, implementation=file_record(Path(__file__)),
                implementation_dependencies=dependencies,
                inputs=[*bindings.values(), file_record(Path(__file__)), *dependencies.values()], sequence_plan=file_record(sequence_path))
    path = output / "plan.json"
    write_json(path, plan)
    return path


def validate_plan(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    plan = json.loads(path.read_text())
    if plan.get("policy") != POLICY or plan.get("schema_version") != 1 or plan.get("criteria") != CRITERIA:
        raise ValueError("Unknown or changed repair ranking policy")
    validate_file_binding(plan["implementation"])
    if plan["implementation"]["sha256"] != sha256(Path(__file__)):
        raise ValueError("Repair ranking implementation differs from the frozen plan")
    dependencies = plan["implementation_dependencies"]
    root = Path(__file__).resolve().parents[1]
    if set(dependencies) != set(IMPLEMENTATION_DEPENDENCIES):
        raise ValueError("Repair ranking implementation dependencies are incomplete")
    for name, pin in dependencies.items():
        validate_file_binding(pin)
        if pin["sha256"] != sha256(root / name):
            raise ValueError("Current repair verifier dependency differs from frozen implementation")
    frame, report, metadata, inherited = _validated_inputs(plan["source_bindings"])
    expected, sequences = derive_plan(frame, report, metadata, inherited, seed=plan["seed"])
    if any(plan.get(key) != value for key, value in expected.items()):
        raise ValueError("Repair plan differs from deterministic complete candidate derivation")
    validate_file_binding(plan["sequence_plan"])
    if read_rows(Path(plan["sequence_plan"]["path"])) != sequences:
        raise ValueError("Frozen repair sequence plan changed or omits source copies")
    expected_inputs = [*plan["source_bindings"].values(), plan["implementation"], *dependencies.values()]
    if sorted(plan["inputs"], key=lambda pin: pin["path"]) != sorted(expected_inputs, key=lambda pin: pin["path"]):
        raise ValueError("Repair request input bindings are incomplete")
    return plan, sequences, frame


def validate_request(path: Path) -> dict[str, Any]:
    request = json.loads(path.read_text())
    if request.get("schema_version") != 1 or request.get("policy") != REQUEST_POLICY:
        raise ValueError("Unknown evaluation source request")
    validate_file_binding(request["plan"])
    plan, sequences, _ = validate_plan(Path(request["plan"]["path"]))
    if request["sequence_plan"] != plan["sequence_plan"]:
        raise ValueError("Evaluation request does not bind its owning sequence plan")
    keys = request["requested_sequence_sha256"]
    if not keys or len(keys) != len(set(keys)) or not set(keys) <= {row["sequence_sha256"] for row in sequences}:
        raise ValueError("Evaluation request is empty, repeated, or outside its plan")
    return {"request": request, "plan": plan, "sequence_plans": sequences}


def optimistic_proposal(plan: Mapping[str, Any], sequences: Sequence[Mapping[str, Any]], *,
                        known_outcomes: Mapping[str, Mapping[str, Any]] | None = None,
                        known_safety: Mapping[str, str] | None = None) -> dict[str, Any]:
    """Inventory proposal only: assumed passes cannot be accepted as evidence."""
    outcomes = {row["sequence_sha256"]: {"sequence_sha256": row["sequence_sha256"], "group_id": row["group_id"],
                                        "selected_chain_id": row["ranked_copies"][0]["chain_id"]} for row in sequences}
    outcomes.update({key: dict(value) for key, value in (known_outcomes or {}).items()})
    safety = {key: "passed" for key in outcomes}
    safety.update(dict(known_safety or {}))
    result = replay_selection(plan, sequences, outcomes, safety)
    if result["status"] != "complete":
        raise ValueError("Metadata-only repair proposal cannot fill all public slots")
    return {"status": "metadata_only_proposal_not_evaluation_membership", "proposed": result["selected"],
            "requested_sequence_sha256": [row["sequence_sha256"] for row in result["selected"] if row["sequence_sha256"] not in (known_outcomes or {})],
            "safety_sequence_sha256": [row["sequence_sha256"] for row in result["selected"] if row["sequence_sha256"] not in (known_safety or {})]}


def _qualification_outcomes(pins: Sequence[Mapping[str, Any]], plan_pin: Mapping[str, Any],
                            plan: Mapping[str, Any], sequences: Sequence[Mapping[str, Any]], *, verify_sources: bool) -> dict[str, Any]:
    indexed = {row["sequence_sha256"]: row for row in sequences}
    all_outcomes: dict[str, Any] = {}
    for pin in pins:
        receipt = _bound_json(pin)
        if receipt.get("schema_version") != 1 or receipt.get("status") != "complete_request" or receipt.get("policy") != REQUEST_POLICY:
            raise ValueError("Repair requires complete source qualification request receipts")
        bindings = _bound_json(receipt["bindings"])
        validate_file_binding(bindings["worker"])
        validate_qualification_predicate(bindings["predicate"])
        if receipt["predicate"] != bindings["predicate"]:
            raise ValueError("Source qualification predicate changed")
        request = _bound_json(bindings["request"])
        if (request.get("schema_version") != 1 or request.get("policy") != REQUEST_POLICY
                or request["plan"] != plan_pin or request["sequence_plan"] != plan["sequence_plan"]):
            raise ValueError("Source request belongs to a different repair plan")
        keys = request["requested_sequence_sha256"]
        if not keys or len(keys) != len(set(keys)) or not set(keys) <= set(indexed):
            raise ValueError("Source request has repeated or foreign candidates")
        catalog = _bound_json(receipt["source_catalog"])
        validate_source_catalog(catalog)
        if catalog["historical_sources"]["archive_pin"] != bindings["historical_archive"]:
            raise ValueError("Source qualification changed its historical archive")
        validate_file_binding(receipt["outcomes"])
        outcomes = read_rows(Path(receipt["outcomes"]["path"]))
        if len(outcomes) != len(keys):
            raise ValueError("Source request outcomes omit candidates")
        qualified = []
        for key, outcome in zip(keys, outcomes, strict=True):
            row = indexed[key]
            validate_outcome(dict(row), outcome, verify_sources=verify_sources)
            validate_outcome_source_membership(outcome, catalog, bindings["raw_roots"])
            if key in all_outcomes and all_outcomes[key] != outcome:
                raise ValueError("Repeated source request changed a candidate outcome")
            all_outcomes[key] = outcome
            chain = outcome["selected_chain_id"]
            if chain is not None:
                copy = next(copy for copy in row["ranked_copies"] if copy["chain_id"] == chain)
                attempt = outcome["attempts"][-1]
                qualified.append(dict(copy) | {"sequence_sha256": key, "source_geometry": attempt["source_geometry"],
                                              "crop_geometry": attempt["crop_geometry"], "qualification": attempt})
        validate_file_binding(receipt["qualified_chains"])
        if (read_rows(Path(receipt["qualified_chains"]["path"])) != qualified
                or receipt["counts"] != {"requested_sequences": len(keys), "qualified_sequences": len(qualified),
                                         "exhausted_sequences": len(keys) - len(qualified)}):
            raise ValueError("Qualified evaluation source export differs from replayed attempts")
    return all_outcomes


def build_safety_plan(plan_path: Path, keys: Sequence[str], output: Path) -> Path:
    """Freeze an exact reverse-query search request; this launches no search."""
    plan, sequences, frame = validate_plan(plan_path)
    indexed = {row["sequence_sha256"]: row for row in sequences}
    if not keys or len(keys) != len(set(keys)) or not set(keys) <= indexed.keys():
        raise ValueError("Safety request must name unique frozen candidate sequences")
    records = {row["chain_id"]: row for row in frame["records"]}
    hidden = {sequence_id(records[chain]["sequence"]): records[chain]["sequence"] for chain in plan["hidden_ids"]}
    output.mkdir(parents=True, exist_ok=True)
    query, source = output / "queries.fasta", output / "hidden.fasta"
    for path, values in ((query, {"s_" + key: indexed[key]["sequence"] for key in keys}), (source, hidden)):
        with path.open("x") as handle:
            handle.write("".join(f">{key}\n{value}\n" for key, value in sorted(values.items())))
    result = {"schema_version": 1, "policy": "public_repair_hidden_reverse_v1", "repair_plan": file_record(plan_path),
              "sequence_sha256": sorted(keys), "query": file_record(query), "source": file_record(source)}
    path = output / "plan.json"
    write_json(path, result)
    return path


def _safety_states(pins: Sequence[Mapping[str, Any]], plan_pin: Mapping[str, Any], plan: Mapping[str, Any],
                   sequences: Sequence[Mapping[str, Any]], frame: Mapping[str, Any]) -> dict[str, str]:
    indexed = {row["sequence_sha256"]: row for row in sequences}
    states = {key: "passed" for slot in plan["slots"] for key in slot["same_group_sequences"]}
    records = {row["chain_id"]: row for row in frame["records"]}
    hidden = {sequence_id(records[chain]["sequence"]): records[chain]["sequence"] for chain in plan["hidden_ids"]}
    native = importlib.import_module("scripts.build_full_parent_frame")
    for pin in pins:
        safety_plan = _bound_json(pin["plan"])
        if (safety_plan.get("schema_version") != 1 or safety_plan.get("policy") != "public_repair_hidden_reverse_v1"
                or safety_plan["repair_plan"] != plan_pin):
            raise ValueError("Candidate reverse safety proof belongs to a different repair plan")
        keys = safety_plan["sequence_sha256"]
        if not keys or keys != sorted(set(keys)) or not set(keys) <= indexed.keys():
            raise ValueError("Candidate reverse proof repeats or substitutes exact sequences")
        expected_query = "".join(f">s_{key}\n{indexed[key]['sequence']}\n" for key in keys)
        expected_source = "".join(f">{key}\n{sequence}\n" for key, sequence in sorted(hidden.items()))
        for role, text in (("query", expected_query), ("source", expected_source)):
            validate_file_binding(safety_plan[role])
            if Path(safety_plan[role]["path"]).read_text() != text:
                raise ValueError("Candidate reverse proof omitted or changed a query/hidden sequence")
        result = _native_result(pin["result"])
        native.verify_barrier(result, Path(safety_plan["query"]["path"]), Path(safety_plan["source"]["path"]), pin["plan"]["sha256"])
        failed = set()
        for query, target in iter_batched_mmseqs_hits(result, min_seq_id=0.3, coverage=0.8, coverage_mode=2):
            if query.removeprefix("s_") not in keys or target not in hidden:
                raise ValueError("Candidate reverse hit escapes its bound query or hidden frame")
            failed.add(query.removeprefix("s_"))
        for key in keys:
            state = "failed" if key in failed else "passed"
            if key in states and states[key] != state:
                raise ValueError("Repeated candidate reverse search changed its safety outcome")
            states[key] = state
    return states


def selection_payload(plan: Mapping[str, Any], replay: Mapping[str, Any], frame: Mapping[str, Any]) -> dict[str, Any]:
    """Check membership invariants and provenance-only remaining capacity."""
    if replay["status"] != "complete":
        raise ValueError("Repair selection still has pending or exhausted candidates")
    selected = replay["selected"]
    additions = sorted(row["selected_chain_id"] for row in selected)
    replaced = sorted(row["original_public_id"] for row in selected)
    active = sorted(plan["fixed_public_ids"] + additions)
    if (len(active) != len(set(active)) or len(active) != len(plan["historical_public_ids"])
            or replaced != sorted(set(plan["historical_public_ids"]) - set(plan["fixed_public_ids"]))
            or set(additions) & set(plan["historical_public_ids"]) or set(active) & set(plan["hidden_ids"])):
        raise ValueError("Repair selection changed frozen membership or repeated targets")
    records = {row["chain_id"]: row for row in frame["records"]}
    inherited = _bound_groups(frame["source_bindings"]["inherited_groups_tsv"])
    active_groups = {inherited[chain] for chain in active}
    hidden_groups = {inherited[chain] for chain in plan["hidden_ids"]}
    if len(active_groups) != len(active) or active_groups & hidden_groups:
        raise ValueError("Active public ownership repeats or overlaps hidden ownership")
    for key in (lambda chain: chain.split("_", 1)[0].lower(), lambda chain: records[chain]["sequence"]):
        if {key(chain) for chain in active} & {key(chain) for chain in plan["hidden_ids"]}:
            raise ValueError("Active public PDB/exact-sequence identity overlaps hidden")
    withdrawn_groups = {row["group_id"] for row in selected if row["tier"] == "new_original_group"}
    heldout_pdbs = {chain.split("_", 1)[0].lower() for chain in additions}
    heldout_sequences = {records[chain]["sequence"] for chain in additions}
    retained = [row for row in frame["records"] if row["group_id"] is not None and inherited.get(row["chain_id"]) not in withdrawn_groups
                and row["chain_id"].split("_", 1)[0].lower() not in heldout_pdbs and row["sequence"] not in heldout_sequences]
    remaining = len({row["sequence_sha256"] for row in retained})
    if remaining < 10000:
        raise ValueError("Provisional evaluation repair leaves fewer than 10000 exact-sequence parents before its additional homology barrier")
    return {"active_public_ids": active, "historical_public_ids": list(plan["historical_public_ids"]),
            "hidden_ids": list(plan["hidden_ids"]), "replaced_public_ids": replaced, "added_public_ids": additions,
            "capacity_after_provenance_only": {"withdrawn_original_group_count": len(withdrawn_groups),
                                               "remaining_parent_chains": len(retained), "remaining_exact_sequence_parents": remaining,
                                               "additional_homology_barrier_pending": True}}


def validate_selection_receipt(path: Path, *, verify_sources: bool = True) -> dict[str, Any]:
    """Replay every ranked decision, actual qualification, and hidden barrier."""
    if verify_sources is not True:
        raise ValueError("Accepting evaluation membership requires actual source replay")
    original = file_record(path)
    receipt = _bound_json(original)
    if (receipt.get("schema_version") != 1 or receipt.get("policy") != POLICY
            or receipt.get("status") != "provisional_source_and_hidden_safety_verified"):
        raise ValueError("Evaluation repair has no complete provisional selection receipt")
    validate_file_binding(receipt["plan"])
    plan, sequences, frame = validate_plan(Path(receipt["plan"]["path"]))
    outcomes = _qualification_outcomes(receipt["qualification_receipts"], receipt["plan"], plan, sequences, verify_sources=verify_sources)
    safety = _safety_states(receipt["safety_receipts"], receipt["plan"], plan, sequences, frame)
    replay = replay_selection(plan, sequences, outcomes, safety)
    payload = selection_payload(plan, replay, frame)
    if receipt["selection"] != replay or any(receipt.get(key) != value for key, value in payload.items()):
        raise ValueError("Provisional repair membership differs from deterministic qualified selection")
    if receipt["source_bindings"] != plan["source_bindings"]:
        raise ValueError("Provisional repair input bindings differ from the frozen plan")
    qualifications = {row["selected_chain_id"]: outcomes[row["sequence_sha256"]]["attempts"][-1] for row in replay["selected"]}
    for outcome in outcomes.values():
        for attempt in outcome["attempts"]:
            for source in (attempt["structure"], attempt.get("raw")):
                if source is not None and sha256(Path(source["path"])) != source["sha256"]:
                    raise ValueError("Qualification source changed before membership acceptance")
    pins = [receipt["plan"], *plan["inputs"], plan["sequence_plan"], *frame["source_bindings"].values()]
    for source_pin in receipt["qualification_receipts"]:
        source_receipt = _bound_json(source_pin)
        bindings = _bound_json(source_receipt["bindings"])
        pins.extend([source_pin, source_receipt["bindings"], source_receipt["source_catalog"], source_receipt["outcomes"],
                     source_receipt["qualified_chains"], bindings["request"], bindings["worker"]])
    for pin in pins:
        validate_file_binding(pin)
    if file_record(path) != original:
        raise ValueError("Repair selection receipt changed during validation")
    return {**payload, "source_bindings": plan["source_bindings"], "qualified_added_ids": payload["added_public_ids"],
            "selected_qualifications": qualifications, "receipt": original}
