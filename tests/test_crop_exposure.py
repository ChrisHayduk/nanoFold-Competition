import itertools
from typing import Any

import pytest

from nanofold.crop_exposure import (
    chain_crop_exposure,
    chain_observed_ca_crop_exposure,
    crop_pair_probability,
    domain_crop_exposure,
)


def segments(residues):
    result = []
    for residue in sorted(residues):
        if result and result[-1]["end"] + 1 == residue:
            result[-1]["end"] = residue
        else:
            result.append({"start": residue, "end": residue})
    return result


def test_exact_expectation_matches_all_small_domains_and_crops():
    for length in range(1, 8):
        for width in range(1, length + 2):
            actual_width = min(width, length)
            crops = [set(range(start, start + actual_width)) for start in range(1, length - actual_width + 2)]
            for bits in itertools.product((False, True), repeat=length):
                domain = {i + 1 for i, included in enumerate(bits) if included}
                if not domain:
                    continue
                exact = domain_crop_exposure(length, width, segments(domain))
                assert exact["expected_residues"] == pytest.approx(sum(len(domain & crop) for crop in crops) / len(crops))
                assert exact["presence_probability"] == pytest.approx(sum(bool(domain & crop) for crop in crops) / len(crops))
                assert exact["intact_probability"] == pytest.approx(sum(domain <= crop for crop in crops) / len(crops))
                assert exact["fragment_probability"] == pytest.approx(sum(bool(domain & crop) and not domain <= crop for crop in crops) / len(crops))


def test_discontinuous_domain_intactness_uses_full_span_not_summed_length():
    result = domain_crop_exposure(10, 4, [{"start": 1, "end": 1}, {"start": 10, "end": 10}])
    assert result["expected_residues"] == pytest.approx(2 / 7)
    assert result["presence_probability"] == pytest.approx(2 / 7)
    assert result["intact_probability"] == 0
    assert result["fragment_probability"] == pytest.approx(2 / 7)


def test_contact_probability_matches_window_enumeration():
    for length in range(1, 10):
        for crop_size in range(1, length + 2):
            width = min(crop_size, length)
            starts = range(1, length - width + 2)
            for first in range(1, length + 1):
                for last in range(first, length + 1):
                    actual = sum(start <= first <= last < start + width for start in starts) / len(starts)
                    assert crop_pair_probability(length, crop_size, first, last) == actual


def test_partial_annotation_retains_unknown_exposure_and_unmapped_domains():
    record = {"length": 10, "cath_domains": [
        {"domain_id": "mapped", "domain_length": 3, "label_segments": [{"start": 1, "end": 3}]},
        {"domain_id": "unmapped", "domain_length": None},
    ]}
    result = chain_crop_exposure(record, crop_size=4)
    assert result["domains"][1]["exposure"] is None
    assert result["unmapped_domain_count"] == 1
    assert result["expected_unknown_residue_fraction"] == pytest.approx(1 - (1 + 2 + 3) / (7 * 4))
    assert result["effective_crop_size"] == 4
    assert result["possible_starts"] == 7


def test_joint_crop_domain_count_and_composition_include_partial_domains():
    record = {"length": 10, "cath_domains": [
        {"domain_id": "a", "topology_id": "1.10.10", "domain_length": 2,
         "label_segments": [{"start": 1, "end": 1}, {"start": 10, "end": 10}]},
        {"domain_id": "b", "topology_id": "1.10.10", "domain_length": 2,
         "label_segments": [{"start": 3, "end": 4}]},
    ]}
    result = chain_crop_exposure(record, crop_size=4)
    counts = {"0": 0.0, "1": 0.0, "2": 0.0}
    for start in range(1, 8):
        crop = set(range(start, start + 4))
        count = int(bool(crop & {1, 10})) + int(bool(crop & {3, 4}))
        counts[str(count)] += 1 / 7
    assert result["mapped_domain_count_probabilities"] == pytest.approx(counts)
    assert sum(result["mapped_domain_composition_probabilities"].values()) == pytest.approx(1)
    for domain in result["domains"]:
        assert sum(domain["presence_by_mapped_domain_count"].values()) == pytest.approx(domain["exposure"]["presence_probability"])


@pytest.mark.parametrize("intervals", [[], [{"start": 0, "end": 2}], [{"start": 2, "end": 11}],
                                    [{"start": 2, "end": 4}, {"start": 4, "end": 6}]])
def test_invalid_or_ambiguous_boundaries_fail_closed(intervals):
    with pytest.raises(ValueError):
        domain_crop_exposure(10, 4, intervals)


def test_cross_domain_overlap_and_unbound_mapped_lengths_fail_closed():
    first = {"domain_id": "one", "domain_length": 3, "label_segments": [{"start": 1, "end": 3}]}
    second = {"domain_id": "two", "domain_length": 3, "label_segments": [{"start": 3, "end": 5}]}
    with pytest.raises(ValueError, match="domains overlap"):
        chain_crop_exposure({"length": 10, "cath_domains": [first, second]}, crop_size=4)
    with pytest.raises(ValueError, match="requires verified label segments"):
        chain_crop_exposure({"length": 10, "cath_domains": [{"domain_length": 3}]}, crop_size=4)
    with pytest.raises(ValueError, match="does not match"):
        chain_crop_exposure({"length": 10, "cath_domains": [{**first, "domain_length": 4}]}, crop_size=4)


def observed_record(mask, crop_size):
    length = len(mask)
    width = min(crop_size, length)
    observed = {index + 1 for index, present in enumerate(mask) if present}
    crops = [set(range(start, start + width)) for start in range(1, length - width + 2)]
    record: dict[str, Any] = {"length": length, "cath_domains": [
        {"domain_id": "discontinuous", "domain_length": 3,
         "label_segments": segments({1, 2, 5}), "topology_id": "1.10.10",
         "homologous_superfamily_id": "1.10.10.10"},
        {"domain_id": "unmapped", "domain_length": None, "topology_id": "2.20.20"},
    ], "crop_geometry": {
        "mapping_status": "verified", "sampling": "uniform_contiguous_integer_start_v1",
        "parent_length": length, "crop_size": crop_size, "effective_crop_size": width,
        "parent_valid_ca_count": len(observed), "observed_ca_segments": segments(observed),
        "expected_valid_ca_count": sum(len(observed & crop) for crop in crops) / len(crops),
    }}
    return record, observed, crops


def test_observed_ca_domain_exposure_matches_every_mask_and_crop():
    domain = {1, 2, 5}
    for mask in itertools.product((False, True), repeat=6):
        for crop_size in range(1, 8):
            record, observed, crops = observed_record(mask, crop_size)
            result = chain_observed_ca_crop_exposure(record, crop_size=crop_size)
            denominator = len(crops) * min(crop_size, 6)
            expected = sum(len(domain & observed & crop) for crop in crops) / denominator
            unknown = sum(len((observed - domain) & crop) for crop in crops) / denominator
            missing = sum(len(crop - observed) for crop in crops) / denominator
            for family, label in (("topology", "1.10.10"), ("superfamily", "1.10.10.10"),
                                  ("architecture", "1.10")):
                assert result["families"][family] == pytest.approx({
                    label: expected, "unknown_observed": unknown, "unobserved_ca": missing,
                })
            assert result["expected_mapped_observed_ca_fraction"] == pytest.approx(expected)


@pytest.mark.parametrize("field,value", [
    ("parent_valid_ca_count", 5), ("expected_valid_ca_count", 2.0),
    ("observed_ca_segments", [{"start": 1, "end": 3}, {"start": 4, "end": 6}]),
])
def test_observed_ca_exposure_rejects_inconsistent_measurements(field, value):
    record, _, _ = observed_record([True] * 6, 4)
    record["crop_geometry"][field] = value
    with pytest.raises(ValueError, match="Observed CA"):
        chain_observed_ca_crop_exposure(record, crop_size=4)


def test_missing_measured_ca_intervals_are_unavailable_not_unobserved():
    record, _, _ = observed_record([False] * 6, 4)
    del record["crop_geometry"]["observed_ca_segments"]
    result = chain_observed_ca_crop_exposure(record, crop_size=4)
    assert result["status"] == "unavailable"
    assert "families" not in result


@pytest.mark.parametrize("field,value", [("topology_id", "unobserved_ca"),
                                       ("homologous_superfamily_id", "unknown_observed"),
                                       ("homologous_superfamily_id", "2.20.20.20")])
def test_observed_ca_labels_cannot_corrupt_missingness_categories(field, value):
    record, _, _ = observed_record([True] * 6, 4)
    record["cath_domains"][0][field] = value
    with pytest.raises(ValueError, match="CATH"):
        chain_observed_ca_crop_exposure(record, crop_size=4)
