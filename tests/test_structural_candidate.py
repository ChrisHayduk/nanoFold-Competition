from __future__ import annotations

import copy
from typing import Any

import pytest

from nanofold.structural_candidate import join_qualified_parents, verify_parent_selection


def sources() -> dict[str, dict[str, Any]]:
    a = {"chain_id": "aaaa_A", "sequence": "A" * 512, "length": 512, "resolution": 999.,
         "release_date": None, "group_id": "ga", "cluster40_id": "a"}
    b = {"chain_id": "bbbb_A", "sequence": "C" * 1000, "length": 1000, "resolution": 2.,
         "release_date": None, "group_id": "gb", "cluster40_id": "b"}
    return {row["chain_id"]: row for row in (a, b)}


def test_full_frame_retains_long_parents_and_explicit_unknown_resolution():
    records = sources()
    plans = [{"sequence": row["sequence"], "group_id": row["group_id"], "ranked_copies": [row]}
             for row in records.values()]
    args: dict[str, Any] = dict(expected_copies=records, ranks={chain: (chain,) for chain in records}, plans=plans,
                qualified=list(records.values()), metadata={chain: {"cath_domains": []} for chain in records})
    joined = join_qualified_parents(**args)
    assert [row["length"] for row in joined] == [512, 1000]
    assert joined[0]["resolution"] is None
    plans.pop()
    with pytest.raises(ValueError, match="omitted admitted source chains"):
        join_qualified_parents(**args)


def test_parent_selection_cannot_silently_relax_group_coverage():
    records = sources()
    args: dict[str, Any] = dict(qualified=list(records.values()), frozen_splits={"val": ["vvvv_A"], "hidden_val": ["hhhh_A"]},
                evaluation_cohorts={"historical_public": ["vvvv_A"], "active_public": ["vvvv_A"], "hidden": ["hhhh_A"]},
                all_sequences={"vvvv_A": "D" * 40, "hhhh_A": "E" * 40}, size=1,
                required_groups={"ga"}, coverage_policy="all_qualified_training_groups")
    with pytest.raises(ValueError, match="All-group policy"):
        verify_parent_selection(["aaaa_A"], **args)
    args["coverage_policy"] = "explicit_required_groups"
    report = verify_parent_selection(["aaaa_A"], **args)
    assert report["selected_training_group_count"] == 1
    assert report["expanded_all_pairs_homology_disjointness_claimed"] is False


def test_parent_selection_rejects_heldout_pdb_and_full_parent_sequence_leakage():
    records = list(sources().values())
    args: dict[str, Any] = dict(qualified=records, frozen_splits={"val": ["aaaa_B"], "hidden_val": ["hhhh_A"]},
                evaluation_cohorts={"historical_public": ["aaaa_B"], "active_public": ["aaaa_B"], "hidden": ["hhhh_A"]},
                all_sequences={"aaaa_B": "D" * 40, "hhhh_A": "E" * 40}, size=2,
                required_groups={"ga", "gb"}, coverage_policy="all_qualified_training_groups")
    with pytest.raises(ValueError, match="PDB overlap"):
        verify_parent_selection(["aaaa_A", "bbbb_A"], **args)
    args = copy.deepcopy(args)
    args["frozen_splits"]["val"] = ["vvvv_A"]
    args["evaluation_cohorts"]["active_public"] = ["vvvv_A"]
    args["evaluation_cohorts"]["historical_public"] = ["vvvv_A"]
    args["all_sequences"]["vvvv_A"] = "A" * 512
    with pytest.raises(ValueError, match="exact sequence overlap"):
        verify_parent_selection(["aaaa_A", "bbbb_A"], **args)


def test_soft_diversity_requires_explicit_zero_mandatory_anchors():
    records = list(sources().values())
    args: dict[str, Any] = dict(qualified=records, frozen_splits={"val": ["vvvv_A"], "hidden_val": ["hhhh_A"]},
                evaluation_cohorts={"historical_public": ["vvvv_A"], "active_public": ["vvvv_A"], "hidden": ["hhhh_A"]},
                all_sequences={"vvvv_A": "D" * 40, "hhhh_A": "E" * 40}, size=1,
                required_groups={"ga"}, coverage_policy="soft_diversity")
    with pytest.raises(ValueError, match="mandatory group anchors"):
        verify_parent_selection(["aaaa_A"], **args)
    args["required_groups"] = set()
    assert verify_parent_selection(["aaaa_A"], **args)["required_group_count"] == 0


def test_retired_public_targets_remain_protected_from_training():
    records = list(sources().values())
    args: dict[str, Any] = dict(
        qualified=records, frozen_splits={"val": ["vvvv_A"], "hidden_val": ["hhhh_A"]},
        evaluation_cohorts={"historical_public": ["rrrr_A"], "active_public": ["vvvv_A"], "hidden": ["hhhh_A"]},
        all_sequences={"vvvv_A": "D" * 40, "hhhh_A": "E" * 40, "rrrr_A": "A" * 512}, size=2,
        required_groups=set(), coverage_policy="soft_diversity",
    )
    with pytest.raises(ValueError, match="exact sequence overlap"):
        verify_parent_selection(["aaaa_A", "bbbb_A"], **args)
    args["all_sequences"]["rrrr_A"] = "G" * 40
    report = verify_parent_selection(["aaaa_A", "bbbb_A"], **args)
    assert report["historical_public_protected_from_training"] is True
    assert report["protected_public_union_count"] == 2
