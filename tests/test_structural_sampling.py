from collections import Counter

import numpy as np
import pytest

from nanofold.structural_geometry import describe_crop_ca
from nanofold.structural_sampling import (
    crop_structural_moments,
    qualified_sequence_reference,
    select_structural_sample,
    structural_moments,
)


def chain(index, group, topology: str | None = "1.10.10", *, repeats=1, resolution=2.0):
    # Distinct sequences, without introducing length as a changing covariate.
    alphabet = "ARNDCQEGHILKMFPSTWYV"
    sequence = "A" * 40 + alphabet[index // 20] + alphabet[index % 20]
    return {
        "chain_id": f"chain_{index}", "sequence": sequence, "group_id": group,
        "length": len(sequence), "resolution": resolution,
        "cath_domains": [
            {"domain_id": f"d{index}_{copy}", "topology_id": topology,
             "homologous_superfamily_id": f"{topology}.1", "domain_length": None}
            for copy in range(repeats)
        ] if topology else [],
    }


def test_domain_moments_retain_multiplicity_and_missing_boundaries():
    row = chain(1, "g", repeats=2)
    row["cath_domains"][0]["domain_length"] = 20
    feature = structural_moments(row)
    assert feature["topology"] == {"1.10.10": 2}
    assert feature["topology_domain_length"] == {"1.10.10|[0,40)": 1, "1.10.10|unknown": 1}
    assert feature["topology_domain_count"] == {"1.10.10|2": 2}
    assert feature["domain_composition"] == {"1.10.10*2": 1}
    assert feature["annotation_coverage"] == {"unknown": 1}
    unknown = structural_moments(chain(2, "g", None))
    assert unknown["topology"] == {"unknown": 1}
    assert unknown["domain_count"] == {"0": 1}


def test_complete_groups_exact_size_determinism_and_multiple_original_unit_members():
    records = [chain(i, f"g{i % 3}", repeats=1 + i % 2) for i in range(12)]
    for row in records:
        row["original_unit_id"] = "same_unit"
        row["pdb_id"] = "same_pdb"
    selected, report = select_structural_sample(records, size=8, seed=9)
    reverse, reversed_report = select_structural_sample(list(reversed(records)), size=8, seed=9)
    assert selected == reverse
    assert report == reversed_report
    assert len(selected) == 8
    assert {row["group_id"] for row in records if row["chain_id"] in selected} == {"g0", "g1", "g2"}
    assert report["final_objective"] <= report["greedy_objective"] + 1e-12
    assert report["covered_groups"] == 3
    assert report["reference_sequences"] == 12


def test_full_reference_precedes_selection_and_report_counts_are_exact():
    records = [chain(i, "dominant", "1.10.10") for i in range(12)]
    records += [chain(i, f"rare_{i}", "2.20.20", repeats=2) for i in range(12, 15)]
    selected, report = select_structural_sample(records, size=7)
    known = {row["chain_id"]: row for row in records}
    assert report["families"]["topology"]["counts"]["1.10.10"]["target"] == pytest.approx(12 / 15 * 7)
    assert report["families"]["topology"]["counts"]["2.20.20"]["reference"] == 6
    assert report["families"]["topology"]["counts"]["2.20.20"]["mandatory_group_minimum"] == 6
    assert report["families"]["topology"]["targets_below_mandatory_group_minimum"] == 1
    actual = Counter()
    for name in selected:
        actual.update(structural_moments(known[name])["topology"])
    for label, values in report["families"]["topology"]["counts"].items():
        assert values["selected"] == actual[label]
    # Required group coverage makes perfect distribution matching impossible.
    assert actual["2.20.20"] >= 6
    assert report["final_objective"] > 0


def test_copy_deduplication_uses_quality_without_changing_reference_weight():
    original = chain(1, "g", resolution=2.0)
    worse = {**original, "chain_id": "worse_copy", "resolution": 3.0}
    second = chain(2, "g")
    reference = qualified_sequence_reference([worse, second, original])
    assert {row["chain_id"] for row in reference} == {"chain_1", "chain_2"}
    selected, report = select_structural_sample([original, worse, second], size=2)
    assert selected == ["chain_1", "chain_2"]
    assert report["input_chains"] == 3
    assert report["reference_sequences"] == 2


def test_unknown_resolution_has_an_explicit_category_and_no_false_quality_advantage():
    known = chain(1, "g", resolution=2.0)
    unknown = {**known, "chain_id": "unknown_copy", "resolution": None}
    assert structural_moments(unknown)["resolution"] == {"unknown": 1}
    assert qualified_sequence_reference([unknown, known]) == [known]
    selected, report = select_structural_sample([unknown, chain(2, "g")], size=2)
    assert len(selected) == 2
    assert report["families"]["resolution"]["counts"]["unknown"]["selected"] == 1


def test_independent_reference_probabilities_change_target_but_not_selected_training_weights():
    records = [chain(1, "g", "1.10.10"), chain(2, "g", "2.20.20"), chain(3, "g", "2.20.20")]
    weights = {"chain_1": 9.0, "chain_2": 0.5, "chain_3": 0.5}
    selected, report = select_structural_sample(records, size=2, reference_weights=weights,
                                               reference_weighting="independent_declared_reference")
    assert len(selected) == 2
    assert report["families"]["topology"]["counts"]["1.10.10"]["target"] == pytest.approx(1.8)
    assert report["reference_probabilities"] == {"chain_1": 0.9, "chain_2": 0.05, "chain_3": 0.05}
    with pytest.raises(ValueError, match="explicit reference_weighting"):
        select_structural_sample(records, size=2, reference_weights=weights)
    with pytest.raises(ValueError, match="cover exactly"):
        select_structural_sample(records, size=2, reference_weights={"chain_1": 1.0})


def test_soft_group_policy_allows_complementary_members_from_the_same_diversity_group():
    records = [chain(i, "common", "1.10.10" if i % 2 else "2.20.20") for i in range(20)]
    records += [chain(20, "rare_b", "3.30.30"), chain(21, "rare_c", "4.40.40")]
    with pytest.raises(ValueError, match="mandatory groups"):
        select_structural_sample(records, size=2)
    selected, report = select_structural_sample(records, size=2, group_policy="soft_diversity")
    assert {row["group_id"] for row in records if row["chain_id"] in selected} == {"common"}
    assert report["mandatory_groups"] == 0
    assert report["covered_groups"] == 1
    assert report["reference_groups"] == 3
    assert report["uncovered_group_reference_probability"] == pytest.approx(2 / 22)
    assert report["group_semantics"] == "training_diversity_groups_not_leakage_certificate"


def test_crop_objective_retains_fractional_exposure_and_absolute_intact_mass():
    first = chain(1, "a")
    first.update(sequence="A" * 600, length=600)
    first["cath_domains"][0].update(domain_length=200, label_segments=[{"start": 1, "end": 200}])
    second = chain(2, "b")
    second.update(sequence="R" * 600, length=600)
    second["cath_domains"][0].update(domain_length=400, label_segments=[{"start": 101, "end": 500}])
    first_features = crop_structural_moments(first, crop_size=256)
    second_features = crop_structural_moments(second, crop_size=256)
    assert 0 < first_features["topology"]["1.10.10"] < 1
    assert first_features["topology_intact"]["1.10.10"] > 0
    assert second_features["topology_intact"]["1.10.10"] == 0
    assert second_features["topology_fragment"]["1.10.10"] > 0
    _, report = select_structural_sample([first, second], size=2, crop_size=256, group_policy="soft_diversity")
    exposure = report["families"]["topology"]["counts"]["1.10.10"]
    assert exposure["selected"] == pytest.approx(first_features["topology"]["1.10.10"] + second_features["topology"]["1.10.10"])
    assert report["families"]["topology_intact"]["selected_feature_mass_per_crop_or_chain"] == pytest.approx(first_features["topology_intact"]["1.10.10"] / 2)
    assert report["final_objective"] == pytest.approx(0, abs=1e-12)


def test_crop_geometry_must_match_parent_and_sampling_policy():
    row = chain(1, "g", None)
    row["crop_geometry"] = {"sampling": "uniform_contiguous_integer_start_v1", "mapping_status": "verified",
                            "crop_size": 256, "effective_crop_size": row["length"], "parent_length": row["length"],
                            "expected_valid_ca_count": row["length"] - 2, "expected_contacts": 12.5}
    features = crop_structural_moments(row)
    assert features["crop_contact_exposure"]["contacts_per_residue"] == 12.5 / row["length"]
    row["crop_geometry"]["crop_size"] = 128
    with pytest.raises(ValueError, match="declared parent and crop sampler"):
        crop_structural_moments(row)


def test_measured_geometry_pipeline_matches_crop_feature_contract():
    row = chain(1, "g", None)
    positions = np.zeros((row["length"], 14, 3))
    positions[:, 1, 0] = np.arange(row["length"]) * 1.5
    mask = np.zeros((row["length"], 14))
    mask[:, 1] = 1
    row["crop_geometry"] = describe_crop_ca(positions, mask, crop_size=16)
    features = crop_structural_moments(row, crop_size=16)
    assert features["crop_observation"] == {"observed_ca": 1.0, "missing_ca": 0.0}
    assert features["crop_contact_exposure"]["contacts_per_residue"] > 0


def test_scarce_groups_and_exact_sequence_leakage_fail_closed():
    first, second = chain(1, "a"), chain(2, "b")
    with pytest.raises(ValueError, match="mandatory groups"):
        select_structural_sample([first, second], size=1)
    with pytest.raises(ValueError, match="distinct sequences"):
        select_structural_sample([first, second], size=3)
    with pytest.raises(ValueError, match="spans supplied groups"):
        qualified_sequence_reference([first, {**first, "chain_id": "copy", "group_id": "b"}])
    with pytest.raises(ValueError, match="unique"):
        qualified_sequence_reference([first, first])


def test_corrupt_structural_annotations_fail_closed():
    row = chain(1, "g")
    row["cath_domains"][0]["homologous_superfamily_id"] = "2.20.20.1"
    with pytest.raises(ValueError, match="disagree"):
        structural_moments(row)
    row = chain(1, "g")
    row["cath_domains"].append(dict(row["cath_domains"][0]))
    with pytest.raises(ValueError, match="Repeated domain ID"):
        structural_moments(row)
    row = chain(1, "g")
    row["cath_unannotated_fraction"] = float("nan")
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        structural_moments(row)


def test_geometry_requires_nearly_complete_single_domain_and_measured_coordinates():
    row = chain(1, "g")
    row["cath_domains"][0]["domain_length"] = row["length"]
    row["source_geometry"] = {
        "mapping_status": "verified", "valid_ca_fraction": 1.0,
        "normalized_radius_of_gyration": 1.7, "relative_contact_order": 0.25,
    }
    assert structural_moments(row)["topology_geometry"] == {"1.10.10|rg:[1.5,2)|co:[0.2,0.3)": 1}
    row["cath_domains"][0]["domain_length"] = None
    assert structural_moments(row)["topology_geometry"] == {"unavailable": 1}
    row["cath_domains"][0]["domain_length"] = row["length"]
    row["source_geometry"]["valid_ca_fraction"] = 0.8
    assert structural_moments(row)["topology_geometry"] == {"unavailable": 1}
    row["source_geometry"]["valid_ca_fraction"] = 1.0
    row["source_geometry"]["normalized_radius_of_gyration"] = float("nan")
    with pytest.raises(ValueError, match="CA geometry"):
        structural_moments(row)


@pytest.mark.parametrize("size", [4, 12])
def test_swaps_improve_fit_and_can_replace_group_anchors(size):
    records = [chain(i, f"g{i % 4}", f"{1 + (i // 4) % 3}.10.10", repeats=1 + i % 3) for i in range(28)]
    greedy, _ = select_structural_sample(records, size=size, seed=7, swap_passes=0)
    swapped, report = select_structural_sample(records, size=size, seed=7, swap_passes=3, max_swap_attempts=100)
    assert report["accepted_swaps"] > 0
    assert report["final_objective"] < report["greedy_objective"]
    assert greedy != swapped
    assert report["swap_attempts"] <= 100
