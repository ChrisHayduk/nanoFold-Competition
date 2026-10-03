from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from nanofold.representative_eligibility import THRESHOLDS, file_record, inspect_candidate
from nanofold.structural_population import (
    qualify_sequence,
    sequence_plan,
    summarize,
    unavailable_attempt,
    validate_outcome,
    validate_outcome_source_membership,
    validate_qualification_predicate,
)


def _copy(chain: str, sequence: str = "AAAA", group: str = "g", resolution: float = 2.0) -> dict:
    return {"chain_id": chain, "sequence": sequence, "group_id": group,
            "length": len(sequence), "resolution": resolution}


def _inspection(candidate: dict, rank: int, passed: bool, source: Path | None = None) -> dict:
    binding = {"path": str(source), "sha256": hashlib.sha256(source.read_bytes()).hexdigest(), "bytes": source.stat().st_size} if source else {"path": "fixture", "sha256": "fixture", "bytes": 0}
    return {"chain_id": candidate["chain_id"], "rank": rank,
            "sequence_sha256": hashlib.sha256(candidate["sequence"].encode()).hexdigest(),
            "status": "passed" if passed else "failed", "structure_status": "passed" if passed else "failed",
            "raw_status": "passed" if passed else "not_evaluated_structure_failed",
            "projection_statistics": dict(THRESHOLDS) if passed else None,
            "structure_rejection": None if passed else "No polymer residues found",
            "raw_full_audit": binding | {"query_length": len(candidate["sequence"]), "rows": 1, "aligned_width": len(candidate["sequence"])},
            "structure": binding, "raw": (binding | {"query_sha256": hashlib.sha256(candidate["sequence"].encode()).hexdigest()}) if passed else None}


def test_frame_retains_distinct_sequences_before_any_source_checks() -> None:
    rows = sequence_plan([_copy("a", resolution=3), _copy("b", resolution=1), _copy("c", "CCCC")],
                         rank=lambda row: (row["resolution"], row["chain_id"]))
    assert len(rows) == 2
    assert [copy["chain_id"] for row in rows if row["sequence"] == "AAAA" for copy in row["ranked_copies"]] == ["b", "a"]


def test_identical_sequences_cannot_cross_group_ownership() -> None:
    with pytest.raises(ValueError, match="crosses immutable"):
        sequence_plan([_copy("a", group="train"), _copy("b", group="val")], rank=lambda row: row["chain_id"])


def test_failed_copy_does_not_exclude_sequence_or_skip_rank() -> None:
    plan = sequence_plan([_copy("a"), _copy("b")], rank=lambda row: row["chain_id"])[0]
    result = qualify_sequence(plan, lambda candidate, rank: _inspection(candidate, rank, candidate["chain_id"] == "b"))
    assert result["selected_chain_id"] == "b"
    assert len(result["attempts"]) == 2
    assert summarize([plan], [result])["qualified_sequence_count"] == 1
    result["attempts"] = result["attempts"][1:]
    with pytest.raises(ValueError, match="exact ranked prefix"):
        validate_outcome(plan, result, verify_sources=False)


def test_acquisition_error_is_not_biological_exhaustion() -> None:
    plan = sequence_plan([_copy("a")], rank=lambda row: row["chain_id"])[0]

    def unavailable(candidate: dict, rank: int) -> dict:
        raise OSError("download unavailable")

    with pytest.raises(OSError, match="download unavailable"):
        qualify_sequence(plan, unavailable)


def test_exhaustion_requires_every_frozen_copy() -> None:
    plan = sequence_plan([_copy("a"), _copy("b")], rank=lambda row: row["chain_id"])[0]
    result = qualify_sequence(plan, lambda candidate, rank: _inspection(candidate, rank, False))
    assert summarize([plan], [result])["exhausted_sequence_count"] == 1
    result["attempts"].pop()
    with pytest.raises(ValueError, match="without exhausting"):
        validate_outcome(plan, result, verify_sources=False)


def test_resumed_result_rejects_source_byte_drift(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = tmp_path / "source"
    source.write_text("old")
    plan = sequence_plan([_copy("a")], rank=lambda row: row["chain_id"])[0]
    result = qualify_sequence(plan, lambda candidate, rank: _inspection(candidate, rank, True, source))
    monkeypatch.setattr("nanofold.structural_population.inspect_candidate", lambda **kwargs: result["attempts"][0])
    validate_outcome(plan, result, verify_sources=True)
    source.write_text("new")
    with pytest.raises(ValueError, match="source bytes changed"):
        validate_outcome(plan, result, verify_sources=True)


@pytest.mark.parametrize("later_row", ["AAA1", "AAA"])
def test_full_msa_qualification_rejects_malformed_nonquery_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, later_row: str,
) -> None:
    structure = tmp_path / "structure.cif"
    structure.write_text("synthetic structure parsed by isolated fixture")
    raw = tmp_path / "msa.a3m"
    raw.write_text(f">query\nAAAA\n>hit\n{later_row}\n")
    parsed = SimpleNamespace(sequence="AAAA", atom14_positions=np.zeros((4, 14, 3)),
                             atom14_mask=np.ones((4, 14)), polymer_sequence="AAAA",
                             polymer_residue_index=np.arange(4))
    monkeypatch.setattr("nanofold.representative_eligibility.extract_chain_atoms", lambda **kwargs: parsed)
    monkeypatch.setattr("scripts.preprocess._project_atom14_to_query",
                        lambda **kwargs: (parsed.atom14_positions, parsed.atom14_mask, dict(THRESHOLDS)))
    kwargs = {"chain": "1abc_A", "sequence": "AAAA", "rank": 0,
              "structure": {"path": str(structure), "sha256": hashlib.sha256(structure.read_bytes()).hexdigest()},
              "raw_path": raw}
    assert inspect_candidate(**kwargs)["status"] == "passed"
    with pytest.raises(ValueError, match="invalid character|inconsistent aligned widths"):
        inspect_candidate(**kwargs, full_msa_validation=True)


def _relocated_predicate(tmp_path: Path) -> tuple[dict, dict]:
    inspection = {"code": {"parser.py": "frozen-parser-hash"}, "runtime": {"python": "3.12.15"}}
    original: dict[str, Any] = {"inspection": inspection.copy()}
    relocated: dict[str, Any] = {"inspection": inspection.copy()}
    for name in ("driver", "journal", "geometry", "availability"):
        old = tmp_path / "original" / name
        new = tmp_path / "relocated" / name
        old.parent.mkdir(exist_ok=True)
        new.parent.mkdir(exist_ok=True)
        old.write_text(name + " implementation")
        new.write_bytes(old.read_bytes())
        original[name] = file_record(old)
        relocated[name] = file_record(new)
    return original, relocated


def test_outcome_rejects_same_query_raw_source_outside_frozen_chain_lookup(tmp_path: Path) -> None:
    from nanofold.chain_paths import chain_id_to_stem

    raw_root = tmp_path / "raw"
    raw = raw_root / "roda_pdb" / chain_id_to_stem("1abc_A") / "uniref90_hits.a3m"
    raw.parent.mkdir(parents=True)
    raw.write_text(">query\nAAAA\n")
    foreign = tmp_path / "foreign.a3m"
    foreign.write_bytes(raw.read_bytes())
    structure = {"path": "bound.cif", "sha256": "structure-hash"}
    attempt = {"chain_id": "1abc_A", "structure": structure, "raw": file_record(raw)}
    outcome = {"attempts": [attempt]}
    catalog = {"members": {"1abc": structure}}
    validate_outcome_source_membership(outcome, catalog, [str(raw_root)])
    attempt["raw"] = file_record(foreign)
    with pytest.raises(ValueError, match="frozen chain lookup"):
        validate_outcome_source_membership(outcome, catalog, [str(raw_root)])


def test_predicate_allows_verified_byte_identical_implementation_relocation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    original, relocated = _relocated_predicate(tmp_path)
    monkeypatch.setattr("nanofold.structural_population.qualification_predicate", lambda: relocated)
    validate_qualification_predicate(original)
    assert original["driver"]["path"] != relocated["driver"]["path"]


@pytest.mark.parametrize("changed", ["original", "current", "runtime", "parser"])
def test_predicate_relocation_rejects_code_and_runtime_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, changed: str,
) -> None:
    original, relocated = _relocated_predicate(tmp_path)
    if changed in {"original", "current"}:
        record = original if changed == "original" else relocated
        path = Path(record["journal"]["path"])
        path.write_text("changed implementation")
        if changed == "current":
            record["journal"] = file_record(path)
    else:
        relocated["inspection"] = {**relocated["inspection"],
                                  changed if changed == "runtime" else "code": {"changed": True}}
    monkeypatch.setattr("nanofold.structural_population.qualification_predicate", lambda: relocated)
    with pytest.raises(ValueError, match="Bound source file changed|implementation bytes differ|inspection code or runtime differs"):
        validate_qualification_predicate(original)


def test_certified_missing_copy_advances_rank_without_inventing_inspection(monkeypatch: pytest.MonkeyPatch) -> None:
    from scripts.qualify_structural_population import inspect_sequence

    plan = sequence_plan([_copy("1abc_A"), _copy("2abc_A")], rank=lambda row: row["chain_id"])[0]
    evidence = {"1abc_A": {"certificate": {"path": "verified-certificate", "sha256": "frozen"}}}
    inspected = []

    def inspect(**kwargs: Any) -> dict[str, Any]:
        inspected.append(kwargs["chain"])
        return _inspection(_copy(kwargs["chain"]), kwargs["rank"], True)

    monkeypatch.setattr("scripts.qualify_structural_population.inspect_candidate", inspect)
    outcome = inspect_sequence((plan, {"2abc": {}}, [], evidence))
    assert inspected == ["2abc_A"]
    assert outcome["selected_chain_id"] == "2abc_A"
    assert outcome["attempts"][0] == unavailable_attempt(plan["ranked_copies"][0], 0, evidence["1abc_A"])
    assert "structure" not in outcome["attempts"][0]
    counts = summarize([plan], [outcome], source_availability=evidence)
    assert counts["source_sequence_count"] == counts["qualified_sequence_count"] == 1
    assert counts["source_unavailable_chain_count"] == 1
    assert sum(counts["exhausted_sequence_counts_by_cause"].values()) == 0


def test_uncertified_missing_raw_stays_pending_before_later_rank(monkeypatch: pytest.MonkeyPatch) -> None:
    from scripts.qualify_structural_population import inspect_sequence

    plan = sequence_plan([_copy("1abc_A"), _copy("2abc_A")], rank=lambda row: row["chain_id"])[0]
    inspected = []

    def inspect(**kwargs: Any) -> dict[str, Any]:
        inspected.append(kwargs["chain"])
        raise ValueError("Required raw query source is missing; acquisition is incomplete")

    monkeypatch.setattr("scripts.qualify_structural_population.inspect_candidate", inspect)
    outcome = inspect_sequence((plan, {"1abc": {}, "2abc": {}}, [], {}))
    assert outcome["status"] == "sources_needed" and outcome["pending_chain_id"] == "1abc_A"
    assert outcome["attempts"] == [] and inspected == ["1abc_A"]


def test_certified_availability_losses_keep_every_parent_and_copy(monkeypatch: pytest.MonkeyPatch) -> None:
    from scripts.qualify_structural_population import inspect_sequence

    plans = sequence_plan([_copy("1abc_A"), _copy("2abc_A"), _copy("3abc_A", "CCCC"),
                           _copy("4abc_A", "CCCC"), _copy("5abc_A", "DDDD")],
                          rank=lambda row: row["chain_id"])
    evidence = {chain: {"certificate": {"path": "verified", "sha256": chain}}
                for chain in ("1abc_A", "2abc_A", "3abc_A")}
    monkeypatch.setattr("scripts.qualify_structural_population.inspect_candidate",
                        lambda **kwargs: _inspection(_copy(kwargs["chain"], kwargs["sequence"]), kwargs["rank"], False))
    outcomes = [inspect_sequence((plan, {"4abc": {}, "5abc": {}}, [], evidence)) for plan in plans]
    counts = summarize(plans, outcomes, source_availability=evidence)
    assert counts["source_sequence_count"] == counts["exhausted_sequence_count"] == 3
    assert counts["source_chain_count"] == counts["attempted_chain_count"] == 5
    assert counts["source_unavailable_chain_count"] == 3
    assert counts["exhausted_sequence_counts_by_cause"] == {
        "source_unavailable_only": 1, "source_unavailable_and_inspection_rejected": 1,
        "inspection_rejected_only": 1}
    index = next(i for i, plan in enumerate(plans) if plan["sequence"] == "AAAA")
    outcomes[index]["attempts"].pop()
    with pytest.raises(ValueError, match="without exhausting"):
        validate_outcome(plans[index], outcomes[index], verify_sources=False, source_availability=evidence)


def test_unavailable_attempt_requires_exact_verified_copy_evidence() -> None:
    plan = sequence_plan([_copy("1abc_A")], rank=lambda row: row["chain_id"])[0]
    evidence = {"1abc_A": {"certificate": {"path": "verified", "sha256": "original"}}}
    outcome = qualify_sequence(plan, lambda candidate, rank: unavailable_attempt(candidate, rank, evidence["1abc_A"]),
                               source_availability=evidence)
    with pytest.raises(ValueError, match="lacks verified acquisition"):
        validate_outcome(plan, outcome, verify_sources=False)
    with pytest.raises(ValueError, match="differs from its acquisition evidence"):
        validate_outcome(plan, outcome, verify_sources=False,
                         source_availability={"1abc_A": {"certificate": {"path": "verified", "sha256": "changed"}}})
    outcome["attempts"][0]["structure_status"] = "failed"
    with pytest.raises(ValueError, match="differs from its acquisition evidence"):
        validate_outcome(plan, outcome, verify_sources=False, source_availability=evidence)


def test_acquisition_certificate_to_qualification_preserves_uncovered_copies(tmp_path: Path) -> None:
    from test_source_availability import make_source_availability_case

    from nanofold.source_availability import validate_source_availability
    from scripts.qualify_structural_population import inspect_sequence

    certificate, plans, roots = make_source_availability_case(tmp_path)
    plan = plans[0]
    plan["ranked_copies"] = [copy | {"sequence": plan["sequence"], "length": len(plan["sequence"])}
                             for copy in plan["ranked_copies"]]
    availability = validate_source_availability(certificate, plans, roots)
    outcome = inspect_sequence((plan, {}, roots, availability))
    validate_outcome(plan, outcome, verify_sources=True, source_availability=availability)
    validate_outcome_source_membership(outcome, {"members": {}}, roots, availability)
    counts = summarize(plans, [outcome], source_availability=availability)
    assert counts["source_sequence_count"] == 1 and counts["attempted_chain_count"] == 3
    assert counts["exhausted_sequence_counts_by_cause"]["source_unavailable_only"] == 1

    # Another deposited copy of the same sequence is not proved unavailable by
    # any of the recorded routes. It must remain pending, preserving its parent.
    plan["ranked_copies"].append(_copy("2abc_A", plan["sequence"]))
    availability = validate_source_availability(certificate, plans, roots)
    pending = inspect_sequence((plan, {}, roots, availability))
    assert pending["status"] == "sources_needed" and pending["pending_chain_id"] == "2abc_A"
    with pytest.raises(ValueError, match="without exhausting"):
        validate_outcome(plan, outcome, verify_sources=True, source_availability=availability)
