from __future__ import annotations

import pytest

from nanofold.full_parent_leakage import sequence_id, sha256
from scripts.analyze_heldout_repair import analyze_candidates, apply_reverse_hits, conflict_alignment_distributions


def row(chain: str, amino_acid: str, length: int = 60) -> dict:
    return {"chain_id": chain, "sequence": amino_acid * length, "length": length, "resolution": 2.0}


def fixture() -> tuple[list[dict], dict[str, str], dict[str, list[str]], list[tuple[str, str]]]:
    records = [row("1val_A", "A"), row("2val_A", "C"), row("3val_A", "D"), row("4hid_A", "E"),
               row("5alt_A", "F"), row("6alt_A", "F"), row("7alt_A", "G"), row("8alt_A", "H"),
               row("9alt_A", "I", 400), row("4hid_B", "K"), row("0alt_A", "D")]
    groups = {"1val_A": "g1", "2val_A": "g2", "3val_A": "g3", "4hid_A": "gh", "5alt_A": "g1",
              "6alt_A": "g1", "7alt_A": "g1", "8alt_A": "g2", "9alt_A": "g1", "4hid_B": "g1", "0alt_A": "g1"}
    frozen = {"val": ["1val_A", "2val_A", "3val_A"], "hidden_val": ["4hid_A"]}
    hits = [("4hid_A", sequence_id("A" * 60)), ("2val_A", sequence_id("E" * 60)),
            ("4hid_A", sequence_id("G" * 60))]
    return records, groups, frozen, hits


def test_pool_preserves_hidden_and_unaffected_public_and_filters_all_candidate_copies() -> None:
    records, groups, frozen, hits = fixture()
    report = analyze_candidates(records, groups, frozen, hits)
    assert report["fixed_hidden_count"] == report["fixed_unaffected_public_count"] == 1
    assert report["affected_public_count"] == 2
    assert report["groups_with_forward_safe_alternatives"] == 2
    assert report["candidate_chain_count"] == 3
    assert report["candidate_sequence_count"] == 2
    by_group = {group["original_group"]: group for group in report["groups"]}
    assert {row["chain_id"] for row in by_group["g1"]["candidates_passing_hidden_query_direction"]} == {"5alt_A", "6alt_A"}
    assert by_group["g1"]["rejection_counts"] == {
        "original_public_target": 1, "outside_short_evaluation_length": 1,
        "same_pdb_as_fixed_heldout": 1, "same_sequence_as_fixed_heldout": 1, "hidden_query_homology": 1,
    }
    assert frozen == {"val": ["1val_A", "2val_A", "3val_A"], "hidden_val": ["4hid_A"]}


def test_reverse_only_conflict_removes_all_deposited_copies_and_exposes_unrepairable_group() -> None:
    report = analyze_candidates(*fixture())
    checked = apply_reverse_hits(report, [(sequence_id("F" * 60), sequence_id("E" * 60))], {sequence_id("E" * 60)})
    assert checked["status"] == "metadata_feasibility_complete"
    assert checked["groups_with_both_direction_alternatives"] == 1
    assert checked["groups_without_both_direction_alternatives"] == 1
    assert checked["both_direction_candidate_chains"] == 1
    assert report["status"] == "reverse_search_required"
    assert "both_direction_candidate_chains" not in report


def test_no_affected_groups_requires_no_reverse_search() -> None:
    records, groups, frozen, _ = fixture()
    report = analyze_candidates(records, groups, frozen, [])
    assert report["affected_group_count"] == 0
    assert report["fixed_unaffected_public_count"] == 3
    assert report["status"] == "no_reverse_candidates"


@pytest.mark.parametrize("query,target", [("unbound_query", sequence_id("A" * 60)), ("4hid_A", "unbound_target")])
def test_unbound_parent_hits_are_not_silently_ignored(query: str, target: str) -> None:
    records, groups, frozen, _ = fixture()
    with pytest.raises(ValueError, match="escapes"):
        analyze_candidates(records, groups, frozen, [(query, target)])


def test_reverse_hits_must_match_both_bound_inputs() -> None:
    report = analyze_candidates(*fixture())
    with pytest.raises(ValueError, match="escapes"):
        apply_reverse_hits(report, [(sequence_id("A" * 60), sequence_id("E" * 60))], {sequence_id("E" * 60)})
    with pytest.raises(ValueError, match="escapes"):
        apply_reverse_hits(report, [(sequence_id("F" * 60), sequence_id("A" * 60))], {sequence_id("E" * 60)})


def test_original_ownership_overlap_is_rejected() -> None:
    records, groups, frozen, hits = fixture()
    groups["4hid_A"] = "g1"
    with pytest.raises(ValueError, match="cross-split ownership"):
        analyze_candidates(records, groups, frozen, hits)


def test_native_conflict_statistics_require_pinned_complete_evidence(tmp_path) -> None:
    records, groups, frozen, hits = fixture()
    report = analyze_candidates(records, groups, frozen, hits)
    path = tmp_path / "alignments.m8"
    path.write_text(f"4hid_A\t{sequence_id('A' * 60)}\t100\t60\t1\t1\t0\t40\n"
                    f"2val_A\t{sequence_id('E' * 60)}\t45\t48\t0.8\t0.8\t0\t40\n")
    barrier = {"jobs": [{"m8_path": str(path), "m8_sha256": sha256(path)}]}
    result = conflict_alignment_distributions(report, records, barrier)
    assert result["hidden_val_query_to_val_target"]["sequence_identity_percent"]["maximum"] == 100
    assert result["val_query_to_hidden_val_target"]["query_coverage_fraction"]["minimum"] == 0.8
    path.write_text(path.read_text().splitlines()[0] + "\n")
    with pytest.raises(ValueError, match="changed"):
        conflict_alignment_distributions(report, records, barrier)
    barrier["jobs"][0]["m8_sha256"] = sha256(path)
    with pytest.raises(ValueError, match="omits"):
        conflict_alignment_distributions(report, records, barrier)
