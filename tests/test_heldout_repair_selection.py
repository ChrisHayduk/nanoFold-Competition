from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path

import pytest

from nanofold.heldout_repair_selection import (
    POLICY,
    REQUEST_POLICY,
    build_plan,
    derive_plan,
    feature_record,
    length_bin,
    optimistic_proposal,
    replay_selection,
    resolution_rank,
    selection_payload,
    sequence_rank,
    stable_hash,
    structural_rank,
    validate_plan,
    validate_request,
    validate_selection_receipt,
)
from nanofold.structural_population import file_record


def _features(
    topology: str = "1.10.10", *, domains: int = 1, length: int = 80, known: bool = True,
) -> dict:
    return {"annotation_stratum": "known" if known else "unknown",
            "topology_counts": {topology: domains} if known else {},
            "domain_count": domains if known else None, "length_bin": length_bin(length)}


def _sequence(
    chain: str, group: str, *, resolution: float | None = 2.0,
    length: int = 80, features: dict | None = None,
) -> dict:
    residues = "".join("ACDEFGHIKLMNPQRS"[int(value, 16)] for value in hashlib.sha256(chain.encode()).hexdigest())
    sequence = (residues * ((length + 63) // 64))[:length]
    return {"sequence_sha256": hashlib.sha256(sequence.encode()).hexdigest(), "sequence": sequence,
            "group_id": group,
            "ranked_copies": [{"chain_id": chain, "group_id": group, "length": length,
                               "sequence": sequence, "resolution": resolution,
                               "features": features if features is not None else _features(length=length)}]}


def _slot(chain: str, group: str, sequences: list[dict], *, features: dict | None = None) -> dict:
    return {"original_public_id": chain, "original_group": group,
            "reference": {"length": 80, "features": features if features is not None else _features()},
            "same_group_sequences": [row["sequence_sha256"] for row in sequences]}


def _plan(slots: list[dict], new_groups: list[str] | None = None) -> dict:
    return {"seed": 0, "slots": sorted(slots, key=lambda row: stable_hash(row["original_public_id"], kind="slot")),
            "new_group_ids": new_groups or [], "fixed_public_ids": ["fixed_A"],
            "historical_public_ids": ["fixed_A", *(row["original_public_id"] for row in slots)],
            "hidden_ids": ["hidden_A"]}


def _outcome(sequence: dict, *, passed: bool = True, selected: str | None = None) -> dict:
    return {"sequence_sha256": sequence["sequence_sha256"], "group_id": sequence["group_id"],
            "selected_chain_id": (selected or sequence["ranked_copies"][0]["chain_id"]) if passed else None,
            "attempts": []}


def _outcomes(*rows: dict) -> dict[str, dict]:
    return {row["sequence_sha256"]: row for row in rows}


def _passed_safety(sequences: list[dict]) -> dict[str, str]:
    return {row["sequence_sha256"]: "passed" for row in sequences}


@pytest.mark.parametrize("invalid", [None, math.nan, math.inf, -math.inf, 0, -1])
def test_resolution_missing_or_nonfinite_cannot_outrank_a_finite_structure(invalid: float | None) -> None:
    assert resolution_rank(1.0) < resolution_rank(2.5) < resolution_rank(invalid)
    assert resolution_rank(invalid) == (1, 0.0)


@pytest.mark.parametrize("length,expected", [(40, 0), (63, 0), (64, 1), (95, 1), (96, 2), (127, 2),
                                             (128, 3), (191, 3), (192, 4), (256, 4)])
def test_evaluation_length_bins_cover_inclusive_short_chain_boundaries(length: int, expected: int) -> None:
    assert length_bin(length) == expected


def test_stable_ties_are_seeded_and_namespace_separated() -> None:
    value = stable_hash("candidate_A", seed=7, kind="chain")
    assert value == stable_hash("candidate_A", seed=7, kind="chain")
    assert len({value, stable_hash("candidate_A", seed=8, kind="chain"),
                stable_hash("candidate_A", seed=7, kind="group")}) == 3


def test_same_group_ranking_preserves_resolution_then_original_length_then_stable_tie() -> None:
    target = {"length": 80, "features": _features()}
    distant_structure = _sequence("best_A", "g", resolution=1.0, length=120,
                                  features=_features("2.20.20", domains=3, length=120))
    close_length = _sequence("close_A", "g", resolution=2.0, length=79)
    longer = _sequence("long_A", "g", resolution=2.0, length=90)
    missing = _sequence("unknown_A", "g", resolution=None, length=80)
    candidates = [missing, longer, close_length, distant_structure]
    def rank(row: dict) -> tuple:
        return sequence_rank(row, target, same_group=True, seed=7)

    assert sorted(candidates, key=rank) == [distant_structure, close_length, longer, missing]
    tied = [_sequence(f"tie{i}_A", "g") for i in range(5)]
    ordered = sorted(tied, key=rank)
    assert ordered == sorted(reversed(tied), key=rank)
    assert [row["ranked_copies"][0]["chain_id"] for row in ordered] == sorted(
        [row["ranked_copies"][0]["chain_id"] for row in tied],
        key=lambda chain: stable_hash(chain, seed=7),
    )


def test_structural_rank_preserves_topology_multiplicity_and_missing_annotation_strata() -> None:
    target = _features(domains=2)
    exact = _features(domains=2)
    wrong_length_bin = {**exact, "length_bin": 4}
    wrong_domain_count = {**exact, "domain_count": 3}
    wrong_topology = _features("2.20.20", domains=2)
    unknown = _features(known=False)
    assert structural_rank(exact, target) < structural_rank(wrong_length_bin, target)
    assert structural_rank(wrong_length_bin, target) < structural_rank(wrong_domain_count, target)
    assert structural_rank(wrong_domain_count, target) < structural_rank(wrong_topology, target)
    assert structural_rank(wrong_topology, target) < structural_rank(unknown, target)
    assert structural_rank(_features(domains=1), target)[1] == 1
    assert structural_rank(unknown, unknown) < structural_rank(exact, unknown)


def test_new_group_structural_match_precedes_better_resolution() -> None:
    target = {"length": 80, "features": _features()}
    matching = _sequence("match_A", "new1", resolution=3.0)
    different = _sequence("different_A", "new2", resolution=1.0, features=_features("2.20.20"))
    assert sequence_rank(matching, target, same_group=False) < sequence_rank(different, target, same_group=False)
    assert sequence_rank(different, target, same_group=True) < sequence_rank(matching, target, same_group=True)


def test_equal_structural_groups_follow_frozen_group_hash_before_copy_resolution() -> None:
    preferred_group, later_group = sorted(["new1", "new2"], key=lambda group: stable_hash(group, kind="group"))
    preferred = _sequence("preferred_A", preferred_group, resolution=3.0)
    later = _sequence("later_A", later_group, resolution=1.0)
    candidates = [later, preferred]
    result = replay_selection(_plan([_slot("retired_A", "original", [])], [later_group, preferred_group]),
                              candidates, _outcomes(*[_outcome(row) for row in candidates]), _passed_safety(candidates))
    assert result["status"] == "complete"
    assert result["selected"][0]["group_id"] == preferred_group


def test_qualified_later_copy_does_not_rerank_sequence_using_its_changed_features() -> None:
    preferred = _sequence("preferred_A", "original", resolution=1.0)
    preferred["ranked_copies"].append({**preferred["ranked_copies"][0], "chain_id": "deposited_twin_A",
                                       "resolution": 4.0, "features": _features("2.20.20")})
    later = _sequence("later_A", "original", resolution=2.0)
    candidates = [later, preferred]
    outcomes = _outcomes(_outcome(preferred, selected="deposited_twin_A"), _outcome(later))
    result = replay_selection(_plan([_slot("retired_A", "original", candidates)]),
                              candidates, outcomes, _passed_safety(candidates))
    assert result["status"] == "complete"
    assert result["selected"][0]["selected_chain_id"] == "deposited_twin_A"
    assert result["selected"][0]["selected_copy_feature_drift"] is True


def test_same_original_group_wins_before_new_group_and_preserves_frozen_membership() -> None:
    original = _sequence("original_alternative_A", "original", resolution=3.0)
    new = _sequence("new_alternative_A", "new", resolution=1.0)
    sequences = [original, new]
    plan = _plan([_slot("retired_A", "original", [original])], ["new"])
    before = copy.deepcopy(plan)
    result = replay_selection(plan, sequences, _outcomes(*[_outcome(row) for row in sequences]), _passed_safety(sequences))
    assert result["status"] == "complete"
    assert result["selected"][0]["selected_chain_id"] == "original_alternative_A"
    assert plan == before


@pytest.mark.parametrize("state", ["source_pending", "safety_missing"])
def test_pending_preferred_candidate_blocks_available_fallback(state: str) -> None:
    preferred = _sequence("preferred_A", "original", resolution=1.0)
    fallback = _sequence("available_A", "new", resolution=2.0)
    sequences = [preferred, fallback]
    outcomes = [_outcome(fallback)] if state == "source_pending" else [_outcome(row) for row in sequences]
    safety = _passed_safety(sequences)
    if state == "safety_missing":
        del safety[preferred["sequence_sha256"]]
    result = replay_selection(_plan([_slot("retired_A", "original", [preferred])], ["new"]),
                              sequences, _outcomes(*outcomes), safety)
    assert result["status"] == "pending"
    assert result["selected"] == []
    assert preferred["sequence_sha256"] in result["pending_sequence_sha256"]


@pytest.mark.parametrize("rejection", ["biological_exhaustion", "safety_failure"])
def test_completed_rejection_allows_the_next_ranked_sequence(rejection: str) -> None:
    preferred = _sequence("preferred_A", "original", resolution=1.0)
    fallback = _sequence("fallback_A", "original", resolution=2.0)
    sequences = [fallback, preferred]
    outcomes = [_outcome(fallback), _outcome(preferred, passed=rejection != "biological_exhaustion")]
    safety = _passed_safety(sequences)
    if rejection == "safety_failure":
        safety[preferred["sequence_sha256"]] = "failed"
    result = replay_selection(_plan([_slot("retired_A", "original", sequences)]), sequences, _outcomes(*outcomes), safety)
    assert result["status"] == "complete"
    assert result["selected"][0]["selected_chain_id"] == "fallback_A"


def test_top_sequence_cannot_be_skipped_for_an_available_sequence_in_the_same_group() -> None:
    preferred = _sequence("preferred_A", "original", resolution=1.0)
    available = _sequence("available_A", "original", resolution=2.0)
    sequences = [available, preferred]
    result = replay_selection(_plan([_slot("retired_A", "original", sequences)]), sequences,
                              _outcomes(_outcome(available)), _passed_safety(sequences))
    assert result["status"] == "pending"
    assert result["selected"] == []


def test_unresolved_new_group_is_not_replaced_by_available_worse_structural_match() -> None:
    preferred = _sequence("preferred_A", "preferred")
    available = _sequence("available_A", "available", features=_features("2.20.20"))
    sequences = [available, preferred]
    result = replay_selection(_plan([_slot("retired_A", "original", [])], ["available", "preferred"]),
                              sequences, _outcomes(_outcome(available)), _passed_safety(sequences))
    assert result["status"] == "pending"
    assert result["selected"] == []
    assert preferred["sequence_sha256"] in result["pending_sequence_sha256"]


def test_one_new_group_cannot_fill_two_public_slots() -> None:
    candidates = [_sequence("new1_A", "only_new_group"), _sequence("new2_A", "only_new_group")]
    slots = [_slot("old1_A", "original1", []), _slot("old2_A", "original2", [])]
    result = replay_selection(_plan(slots, ["only_new_group"]), candidates,
                              _outcomes(*[_outcome(row) for row in candidates]), _passed_safety(candidates))
    assert result["status"] == "exhausted"
    assert len(result["selected"]) == 1
    assert len({row["group_id"] for row in result["selected"]}) == len(result["selected"])


def test_pending_first_slot_stops_later_slots_and_input_order_does_not_change_selection() -> None:
    candidates = [_sequence("candidate1_A", "original1"), _sequence("candidate2_A", "original2")]
    slots = [_slot("old1_A", "original1", [candidates[0]]), _slot("old2_A", "original2", [candidates[1]])]
    plan = _plan(slots)
    first_hash = plan["slots"][0]["same_group_sequences"][0]
    available = next(row for row in candidates if row["sequence_sha256"] != first_hash)
    result = replay_selection(plan, candidates, _outcomes(_outcome(available)), _passed_safety(candidates))
    assert result["status"] == "pending"
    assert result["selected"] == []
    outcomes = [_outcome(row) for row in candidates]
    complete = replay_selection(plan, candidates, _outcomes(*outcomes), _passed_safety(candidates))
    assert complete["status"] == "complete"
    assert complete == replay_selection(plan, list(reversed(candidates)), _outcomes(*reversed(outcomes)),
                                        _passed_safety(candidates))


def test_no_qualified_candidate_is_reported_as_exhausted_only_after_all_outcomes_arrive() -> None:
    candidates = [_sequence("first_A", "original", resolution=1.0),
                  _sequence("last_A", "original", resolution=2.0)]
    plan = _plan([_slot("retired_A", "original", candidates)])
    outcomes = [_outcome(candidates[0], passed=False)]
    assert replay_selection(plan, candidates, _outcomes(*outcomes), _passed_safety(candidates))["status"] == "pending"
    outcomes.append(_outcome(candidates[1], passed=False))
    result = replay_selection(plan, candidates, _outcomes(*outcomes), _passed_safety(candidates))
    assert result["status"] == "exhausted"
    assert result["selected"] == []


def test_optimistic_proposal_preserves_known_failures_and_does_not_request_completed_sources() -> None:
    biological_failure = _sequence("biological_failure_A", "original", resolution=1.0)
    safety_failure = _sequence("safety_failure_A", "original", resolution=2.0)
    pending = _sequence("pending_A", "original", resolution=3.0)
    candidates = [pending, safety_failure, biological_failure]
    plan = _plan([_slot("retired_A", "original", candidates)])
    outcomes = _outcomes(_outcome(biological_failure, passed=False))
    safety = {safety_failure["sequence_sha256"]: "failed"}
    proposal = optimistic_proposal(plan, candidates, known_outcomes=outcomes, known_safety=safety)
    assert [row["selected_chain_id"] for row in proposal["proposed"]] == ["pending_A"]
    assert proposal["requested_sequence_sha256"] == [pending["sequence_sha256"]]
    outcomes[pending["sequence_sha256"]] = _outcome(pending)
    safety[pending["sequence_sha256"]] = "passed"
    resumed = optimistic_proposal(plan, candidates, known_outcomes=outcomes, known_safety=safety)
    assert resumed["proposed"] == proposal["proposed"]
    assert resumed["requested_sequence_sha256"] == resumed["safety_sequence_sha256"] == []


def test_unknown_annotation_remains_a_separate_stratum_without_inferred_domains() -> None:
    annotated = {"length": 80, "cath_annotation_status": "annotated",
                 "cath_topology_counts": {"1.10.10": 2}, "cath_domain_count": 2}
    assert feature_record(annotated) == _features(domains=2)
    unknown = {**annotated, "cath_annotation_status": "unknown", "cath_topology_counts": {}, "cath_domain_count": 0}
    assert feature_record(unknown) == _features(known=False)
    with pytest.raises(ValueError, match="domain count"):
        feature_record({**annotated, "cath_domain_count": 1})
    with pytest.raises(ValueError, match="multiplicities"):
        feature_record({**annotated, "cath_topology_counts": {"1.10.10": True}})


@pytest.mark.parametrize("change", ["length", "cath_annotation_status", "cath_topology_counts", "cath_domain_count",
                                     "unrecognized_status", "unknown_with_counts", "annotated_without_counts",
                                     "unknown_with_domains"])
def test_missing_or_inconsistent_structural_annotations_cannot_become_unknown_stratum(change: str) -> None:
    metadata = {"length": 80, "cath_annotation_status": "annotated",
                "cath_topology_counts": {"1.10.10": 2}, "cath_domain_count": 2}
    if change in metadata:
        del metadata[change]
    elif change == "unrecognized_status":
        metadata["cath_annotation_status"] = "unannotated"
    elif change == "unknown_with_counts":
        metadata["cath_annotation_status"] = "unknown"
    elif change == "annotated_without_counts":
        metadata["cath_topology_counts"] = {}
    else:
        metadata.update(cath_annotation_status="unknown", cath_topology_counts={})
    with pytest.raises(ValueError):
        feature_record(metadata)


@pytest.mark.parametrize("length", [39, 257])
def test_long_parents_and_undersized_chains_cannot_enter_evaluation_bins(length: int) -> None:
    with pytest.raises(ValueError, match="40--256"):
        length_bin(length)


def _derivation_fixture(*, public_count: int = 3, hidden_count: int = 1,
                        affected_count: int = 2, same_group_count: int = 1) -> tuple[dict, dict, dict, dict]:
    records, metadata, inherited = [], {}, {}

    def add(chain: str, group: str, *, admitted: bool = False, length: int = 80) -> None:
        sequence = _sequence(chain, group, length=length, features={})
        record = {**sequence["ranked_copies"][0], "sequence_sha256": sequence["sequence_sha256"],
                  "group_id": group if admitted else None}
        records.append(record)
        metadata[chain] = {"length": length, "cath_annotation_status": "annotated",
                           "cath_topology_counts": {"1.10.10": 1}, "cath_domain_count": 1}
        inherited[chain] = group

    public = [f"public{i:04}_A" for i in range(public_count)]
    hidden = [f"hidden{i:04}_A" for i in range(hidden_count)]
    for chain in public + hidden:
        add(chain, f"group:{chain}")
    groups = []
    for index in range(affected_count):
        chain = public[index]
        alternatives = []
        if index < same_group_count:
            alternative = f"same{index:04}_A"
            add(alternative, inherited[chain])
            alternatives.append({"chain_id": alternative})
        groups.append({"original_public_target": chain, "original_group": inherited[chain],
                       "candidates_passing_both_directions": alternatives})
    for index in range(affected_count - same_group_count):
        add(f"new{index:04}_A", f"new-group:{index}", admitted=True)
    add("short_A", "short-group", admitted=True, length=39)
    add("long_A", "long-group", admitted=True, length=257)
    add("excluded_A", "excluded-group")
    return {"records": records, "frozen_splits": {"val": public, "hidden_val": hidden}}, {"groups": groups}, metadata, inherited


def test_complete_repair_preserves_1000_hidden_786_fixed_public_and_41_original_groups() -> None:
    inputs = _derivation_fixture(public_count=1000, hidden_count=1000, affected_count=214, same_group_count=41)
    frame, _, _, _ = inputs
    plan, sequences = derive_plan(*inputs)
    assert plan["hidden_ids"] == sorted(frame["frozen_splits"]["hidden_val"])
    assert plan["historical_public_ids"] == sorted(frame["frozen_splits"]["val"])
    assert plan["fixed_public_ids"] == sorted(frame["frozen_splits"]["val"][214:])
    assert len(plan["hidden_ids"]) == 1000
    assert len(plan["fixed_public_ids"]) == 786
    assert len(plan["slots"]) == 214
    assert sum(bool(slot["same_group_sequences"]) for slot in plan["slots"]) == 41
    assert len(plan["new_group_ids"]) == 173
    assert len(sequences) == 214
    selected = replay_selection(plan, sequences, _outcomes(*[_outcome(row) for row in sequences]), _passed_safety(sequences))
    assert selected["status"] == "complete"
    assert len(selected["selected"]) == len({row["group_id"] for row in selected["selected"]}) == 214
    assert sum(row["tier"] == "same_original_group" for row in selected["selected"]) == 41
    active = plan["fixed_public_ids"] + [row["selected_chain_id"] for row in selected["selected"]]
    assert len(active) == len(set(active)) == 1000


def test_plan_does_not_use_source_availability_and_freezes_all_deposited_copies() -> None:
    inputs = _derivation_fixture()
    frame, _, metadata, inherited = inputs
    original = next(row for row in frame["records"] if row["chain_id"] == "same0000_A")
    twin = {**original, "chain_id": "same_twin_A", "resolution": 1.0}
    frame["records"].append(twin)
    metadata[twin["chain_id"]] = metadata[original["chain_id"]].copy()
    inherited[twin["chain_id"]] = inherited[original["chain_id"]]
    inputs[1]["groups"][0]["candidates_passing_both_directions"].append({"chain_id": twin["chain_id"]})
    expected = derive_plan(*inputs)
    for index, row in enumerate(frame["records"]):
        row.update(raw_available=index % 2 == 0, structure_available=False, qualified=index % 2 == 1,
                   cached_path=f"/unavailable/{index}")
    frame["records"].reverse()
    actual = derive_plan(*inputs)
    assert actual == expected
    copies = next(row["ranked_copies"] for row in actual[1] if len(row["ranked_copies"]) == 2)
    assert [row["chain_id"] for row in copies] == ["same_twin_A", "same0000_A"]


@pytest.mark.parametrize("collision", ["fixed_pdb", "hidden_sequence"])
def test_candidate_cannot_share_fixed_heldout_pdb_or_hidden_sequence(collision: str) -> None:
    inputs = _derivation_fixture()
    frame, _, metadata, inherited = inputs
    candidate = next(row for row in frame["records"] if row["chain_id"] == "new0000_A")
    hidden = next(row for row in frame["records"] if row["chain_id"] == "hidden0000_A")
    if collision == "fixed_pdb":
        old_id = candidate["chain_id"]
        candidate["chain_id"] = "PUBLIC0002_B"
        metadata[candidate["chain_id"]] = metadata.pop(old_id)
        inherited[candidate["chain_id"]] = inherited.pop(old_id)
    else:
        candidate["sequence"] = hidden["sequence"]
        candidate["sequence_sha256"] = hidden["sequence_sha256"]
    with pytest.raises(ValueError, match="frozen evaluation ownership"):
        derive_plan(*inputs)


def _saved_plan(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, inputs: tuple | None = None) -> Path:
    inputs = inputs if inputs is not None else _derivation_fixture()
    monkeypatch.setattr("nanofold.heldout_repair_selection._validated_inputs", lambda bindings: inputs)
    paths = [tmp_path / f"{name}.json" for name in ("frame", "feasibility", "metadata", "metadata_receipt", "source_lock")]
    for path in paths:
        path.write_text("{}\n")
    return build_plan(baseline_frame_receipt=paths[0], complete_feasibility=paths[1], structural_metadata=paths[2],
                      structural_metadata_receipt=paths[3], metadata_source_lock=paths[4],
                      output=tmp_path / "repair")


@pytest.mark.parametrize("change", ["bound_bytes", "current_bytes", "missing_dependency"])
def test_plan_requires_complete_unchanged_implementation_dependencies(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, change: str,
) -> None:
    path = _saved_plan(tmp_path, monkeypatch)
    plan, _, _ = validate_plan(path)
    dependencies = plan["implementation_dependencies"]
    assert set(dependencies) == {"nanofold/full_parent_leakage.py", "nanofold/mmseqs.py",
                                 "nanofold/structural_population.py", "scripts/build_full_parent_frame.py",
                                 "scripts/analyze_heldout_repair.py", "scripts/plan_heldout_repair.py",
                                 "scripts/build_full_source_metadata.py", "scripts/build_structure_metadata.py",
                                 "nanofold/structure_metadata.py", "nanofold/cath_boundaries.py"}
    name = "nanofold/structural_population.py"
    if change == "missing_dependency":
        del dependencies[name]
        expected_error = "dependencies are incomplete"
    else:
        copied_dependency = tmp_path / "structural_population.py"
        copied_dependency.write_bytes(Path(dependencies[name]["path"]).read_bytes())
        dependencies[name] = file_record(copied_dependency)
        copied_dependency.write_text(copied_dependency.read_text() + "\n# changed qualification behavior\n")
        if change == "current_bytes":
            dependencies[name] = file_record(copied_dependency)
            expected_error = "Current repair verifier dependency differs"
        else:
            expected_error = "Bound source file changed"
    plan["inputs"] = [*plan["source_bindings"].values(), plan["implementation"], *dependencies.values()]
    path.write_text(json.dumps(plan))
    with pytest.raises(ValueError, match=expected_error):
        validate_plan(path)


@pytest.mark.parametrize("field", ["hidden_ids", "historical_public_ids", "fixed_public_ids"])
def test_validator_rejects_membership_substitution_even_when_counts_match(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, field: str,
) -> None:
    path = _saved_plan(tmp_path, monkeypatch)
    plan, _, _ = validate_plan(path)
    plan[field][0] = "substituted_A"
    path.write_text(json.dumps(plan))
    with pytest.raises(ValueError, match="deterministic complete candidate derivation"):
        validate_plan(path)


@pytest.mark.parametrize("rebind", [False, True])
def test_sequence_plan_requires_exact_bytes_and_all_frozen_candidates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, rebind: bool,
) -> None:
    path = _saved_plan(tmp_path, monkeypatch)
    plan, _, _ = validate_plan(path)
    sequence_path = Path(plan["sequence_plan"]["path"])
    sequence_path.write_text(sequence_path.read_text().splitlines()[0] + "\n")
    if rebind:
        plan["sequence_plan"] = file_record(sequence_path)
        path.write_text(json.dumps(plan))
    with pytest.raises(ValueError, match="Bound source file changed|omits source copies"):
        validate_plan(path)


@pytest.mark.parametrize("change", ["plan_bytes", "sequence_binding", "duplicate_sequence", "foreign_sequence"])
def test_acquisition_request_requires_exact_owning_plan_and_sequence_frame(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, change: str,
) -> None:
    path = _saved_plan(tmp_path, monkeypatch)
    plan, sequences, _ = validate_plan(path)
    request = {"schema_version": 1, "policy": REQUEST_POLICY, "plan": file_record(path),
               "sequence_plan": plan["sequence_plan"], "requested_sequence_sha256": [sequences[0]["sequence_sha256"]]}
    request_path = tmp_path / "request.json"
    request_path.write_text(json.dumps(request))
    assert validate_request(request_path)["request"] == request
    if change == "plan_bytes":
        path.write_text(path.read_text() + "\n")
    elif change == "sequence_binding":
        request["sequence_plan"] = {**plan["sequence_plan"], "sha256": "0" * 64}
    elif change == "duplicate_sequence":
        request["requested_sequence_sha256"] *= 2
    else:
        request["requested_sequence_sha256"] = ["0" * 64]
    request_path.write_text(json.dumps(request))
    with pytest.raises(ValueError, match="Bound source file changed|owning sequence plan|outside its plan"):
        validate_request(request_path)


def _selection_fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, withdraw_training_copy: bool = False) -> tuple:
    inputs = _derivation_fixture()
    frame, _, _, inherited = inputs
    for row in frame["records"]:
        if row["chain_id"] in {"short_A", "long_A"}:
            row["group_id"] = None
    for index in range(10000):
        chain = f"training{index:05}_A"
        group = f"training-group:{index}"
        sequence = _sequence(chain, group, length=300, features={})
        frame["records"].append({**sequence["ranked_copies"][0], "sequence_sha256": sequence["sequence_sha256"]})
        inherited[chain] = "new-group:0" if withdraw_training_copy and index == 9999 else group
    groups_path = tmp_path / "groups.tsv"
    groups_path.write_text("".join(f"{group}\t{chain}\n" for chain, group in sorted(inherited.items())))
    frame["source_bindings"] = {"inherited_groups_tsv": file_record(groups_path)}
    plan_path = _saved_plan(tmp_path, monkeypatch, inputs=inputs)
    plan, sequences, _ = validate_plan(plan_path)
    outcomes = _outcomes(*[_outcome(row) for row in sequences])
    structure_path = tmp_path / "qualified-structure.cif"
    raw_path = tmp_path / "qualified-raw.a3m"
    structure_path.write_text("source bytes already inspected by the qualification fixture\n")
    raw_path.write_text(">synthetic\nACDE\n")
    for outcome in outcomes.values():
        outcome["attempts"] = [{"chain_id": outcome["selected_chain_id"], "status": "passed",
                                "structure": file_record(structure_path), "raw": file_record(raw_path)}]
    safety = _passed_safety(sequences)
    replay = replay_selection(plan, sequences, outcomes, safety)
    return plan_path, plan, sequences, frame, outcomes, safety, replay


def test_capacity_removes_every_parent_in_a_new_heldout_group_before_10000_guard(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, plan, _, frame, _, _, replay = _selection_fixture(tmp_path, monkeypatch, withdraw_training_copy=True)
    with pytest.raises(ValueError, match="fewer than 10000 exact-sequence parents"):
        selection_payload(plan, replay, frame)


def test_selection_rejects_ownership_file_drift_after_baseline_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, plan, _, frame, _, _, replay = _selection_fixture(tmp_path, monkeypatch)
    selection_payload(plan, replay, frame)
    groups_path = Path(frame["source_bindings"]["inherited_groups_tsv"]["path"])
    groups_path.write_text(groups_path.read_text() + "\n# appended after baseline validation\n")
    with pytest.raises(ValueError, match="Bound source file changed"):
        selection_payload(plan, replay, frame)


def test_selection_cannot_disable_actual_source_replay_before_reading_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unexpected_read(path: Path) -> dict:
        pytest.fail("Source replay must be required before reading any receipt")

    monkeypatch.setattr("nanofold.heldout_repair_selection.file_record", unexpected_read)
    with pytest.raises(ValueError, match="actual source replay"):
        validate_selection_receipt(tmp_path / "nonexistent.json", verify_sources=False)


@pytest.mark.parametrize("change", ["active_membership", "hidden_membership", "source_bindings", "selection",
                                     "capacity", "plan_bytes", "pending_qualification", "pending_safety",
                                     "structure_bytes", "raw_bytes"])
def test_selection_receipt_replays_exact_membership_proofs_and_bindings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, change: str,
) -> None:
    plan_path, plan, sequences, frame, outcomes, safety, replay = _selection_fixture(tmp_path, monkeypatch)
    payload = selection_payload(plan, replay, frame)
    assert payload["capacity_after_provenance_only"]["remaining_exact_sequence_parents"] == 10000
    assert payload["capacity_after_provenance_only"]["additional_homology_barrier_pending"] is True
    monkeypatch.setattr("nanofold.heldout_repair_selection._qualification_outcomes", lambda *args, **kwargs: outcomes)
    monkeypatch.setattr("nanofold.heldout_repair_selection._safety_states", lambda *args: safety)
    receipt = {"schema_version": 1, "policy": POLICY, "status": "provisional_source_and_hidden_safety_verified",
               "plan": file_record(plan_path), "qualification_receipts": [], "safety_receipts": [],
               "selection": replay, "source_bindings": plan["source_bindings"], **payload}
    receipt_path = tmp_path / "selection.json"
    receipt_path.write_text(json.dumps(receipt))
    validated = validate_selection_receipt(receipt_path)
    assert validated["active_public_ids"] == payload["active_public_ids"]
    assert set(validated["selected_qualifications"]) == set(payload["added_public_ids"])
    if change == "active_membership":
        receipt["active_public_ids"][0] = "substituted_A"
    elif change == "hidden_membership":
        receipt["hidden_ids"][0] = "substituted_A"
    elif change == "source_bindings":
        receipt["source_bindings"] = {**receipt["source_bindings"], "unbound": {"sha256": "0" * 64}}
    elif change == "selection":
        receipt["selection"]["selected"][0]["tier"] = "unverified_tier"
    elif change == "capacity":
        receipt["capacity_after_provenance_only"]["additional_homology_barrier_pending"] = False
    elif change == "plan_bytes":
        plan_path.write_text(plan_path.read_text() + "\n")
    elif change == "pending_qualification":
        outcomes.clear()
    elif change == "pending_safety":
        safety.clear()
    else:
        field = "structure" if change == "structure_bytes" else "raw"
        source_path = Path(next(iter(outcomes.values()))["attempts"][0][field]["path"])
        source_path.write_text(source_path.read_text() + "changed after qualification\n")
    receipt_path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="deterministic qualified selection|input bindings differ|Bound source file changed|pending or exhausted|Qualification source changed"):
        validate_selection_receipt(receipt_path)
