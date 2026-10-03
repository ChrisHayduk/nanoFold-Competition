from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from nanofold.structural_geometry import describe_crop_ca, describe_projected_ca


def test_descriptors_are_rigid_motion_invariant_and_ignore_masked_coordinates() -> None:
    rng = np.random.default_rng(42)
    xyz = rng.normal(size=(12, 14, 3)) * 4
    mask = np.zeros((12, 14))
    mask[:10, 1] = 1
    first = describe_projected_ca(xyz, mask)
    rotation = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    moved = xyz @ rotation + np.array([100, -77, 18])
    moved[10:] = np.nan
    second = describe_projected_ca(moved, mask)
    assert first["normalized_radius_of_gyration"] == pytest.approx(second["normalized_radius_of_gyration"])
    assert first["relative_contact_order"] == pytest.approx(second["relative_contact_order"])
    assert first["ca_distance_histogram"] == second["ca_distance_histogram"]
    assert first["valid_ca_count"] == 10
    assert sum(first["ca_distance_histogram"]) == pytest.approx(1)


def test_contact_order_uses_projected_query_separation_and_absent_contacts_are_explicit() -> None:
    xyz = np.zeros((8, 14, 3))
    mask = np.zeros((8, 14))
    mask[[0, 7], 1] = 1
    xyz[7, 1, 0] = 4
    close = describe_projected_ca(xyz, mask)
    assert close["relative_contact_order"] == 7 / 8
    assert close["nonlocal_pair_count"] == close["contact_count"] == 1
    xyz[7, 1, 0] = 100
    far = describe_projected_ca(xyz, mask)
    assert far["relative_contact_order"] is None
    assert far["contact_count"] == 0


def test_observed_nonfinite_ca_is_rejected() -> None:
    xyz = np.zeros((4, 14, 3))
    mask = np.zeros((4, 14))
    mask[:, 1] = 1
    xyz[0, 1] = np.nan
    with pytest.raises(ValueError, match="must be finite"):
        describe_projected_ca(xyz, mask)


@pytest.mark.parametrize("crop_size", [1, 2, 3, 4, 10, 23, 30])
def test_crop_expectations_match_enumerating_every_window(crop_size: int) -> None:
    rng = np.random.default_rng(73)
    xyz = rng.normal(size=(23, 14, 3)) * 5
    mask = np.zeros((23, 14))
    mask[[0, 3, 4, 6, 7, 9, 11, 14, 17, 18, 19, 22], 1] = 1
    width = min(crop_size, len(xyz))
    crops = [describe_projected_ca(xyz[start:start + width], mask[start:start + width])
             for start in range(len(xyz) - width + 1)]
    actual = describe_crop_ca(xyz, mask, crop_size=crop_size)
    parent = describe_projected_ca(xyz, mask)
    contacts = sum(crop["contact_count"] for crop in crops)
    pairs = sum(crop["nonlocal_pair_count"] for crop in crops)
    assert actual["expected_contacts"] == pytest.approx(contacts / len(crops))
    assert actual["expected_nonlocal_pair_count"] == pytest.approx(pairs / len(crops))
    assert actual["expected_valid_ca_count"] == pytest.approx(np.mean([crop["valid_ca_count"] for crop in crops]))
    assert actual["parent_contact_count"] == parent["contact_count"]
    assert actual["contact_retention_fraction"] == pytest.approx(contacts / len(crops) / parent["contact_count"])
    radii = [crop["normalized_radius_of_gyration"] for crop in crops if crop["valid_ca_count"]]
    assert actual["normalized_radius_of_gyration"] == pytest.approx(np.mean(radii), abs=1e-7)
    assert actual["empty_crop_probability"] == pytest.approx(1 - len(radii) / len(crops))
    valid_counts = [crop["valid_ca_count"] for crop in crops]
    assert actual["minimum_valid_ca_count"] == min(valid_counts)
    assert actual["zero_valid_ca_crop_count"] == sum(count == 0 for count in valid_counts)
    assert actual["below_threshold_valid_ca_crop_count"] == sum(count < 32 for count in valid_counts)
    assert actual["below_threshold_valid_ca_crop_probability"] == pytest.approx(np.mean(np.asarray(valid_counts) < 32))
    if pairs:
        histogram = sum(np.asarray(crop["ca_distance_histogram"]) * crop["nonlocal_pair_count"]
                        for crop in crops if crop["nonlocal_pair_count"]) / pairs
        assert actual["ca_distance_histogram"] == pytest.approx(histogram)
    else:
        assert actual["ca_distance_histogram"] is None
    if contacts:
        contact_order = sum(crop["relative_contact_order"] * crop["contact_count"]
                            for crop in crops if crop["contact_count"]) / contacts
        assert actual["relative_contact_order"] == pytest.approx(contact_order)
    else:
        assert actual["relative_contact_order"] is None


@pytest.mark.parametrize(("first", "last", "retention"), [(0, 3, 0.5), (1, 4, 1), (0, 255, 0.5), (0, 256, 0)])
def test_crop_contacts_respect_parent_indices_and_start_boundaries(first: int, last: int, retention: float) -> None:
    xyz = np.full((257, 14, 3), np.nan)
    mask = np.zeros((257, 14))
    mask[[first, last], 1] = 1
    xyz[first, 1] = [0, 0, 0]
    xyz[last, 1] = [4, 0, 0]
    result = describe_crop_ca(xyz, mask)
    assert result["parent_length"] == 257
    assert result["effective_crop_size"] == 256
    assert result["parent_contact_count"] == 1
    assert result["expected_contacts"] == retention
    assert result["contact_retention_fraction"] == retention


def test_crop_geometry_rigid_motion_and_masked_coordinates() -> None:
    rng = np.random.default_rng(981)
    xyz = rng.normal(size=(35, 14, 3)) * 7
    mask = np.zeros((35, 14))
    mask[rng.random(35) > 0.3, 1] = 1
    rotation = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    moved = xyz @ rotation + np.array([1e6, -2e6, 3e6])
    moved[mask == 0] = np.nan
    first = describe_crop_ca(xyz, mask, crop_size=16)
    second = describe_crop_ca(moved, mask, crop_size=16)
    for key in ("expected_contacts", "parent_contact_count", "contact_retention_fraction",
                "normalized_radius_of_gyration", "relative_contact_order", "ca_distance_histogram"):
        assert first[key] == pytest.approx(second[key])


def test_parent_contacts_cross_block_boundaries_and_strict_distance_cutoff() -> None:
    xyz = np.zeros((260, 14, 3))
    mask = np.zeros((260, 14))
    mask[:, 1] = 1
    # All pairs coincide. Only true sequence separation >= 3 counts; all parent
    # contacts must be counted once even across the 128-residue work blocks.
    result = describe_crop_ca(xyz, mask)
    assert result["parent_contact_count"] == 257 * 258 // 2
    assert result["expected_contacts"] == pytest.approx(253 * 254 // 2)
    mask[1:-1, 1] = 0
    xyz[-1, 1, 0] = 8
    result = describe_crop_ca(xyz, mask, crop_size=260)
    assert result["parent_contact_count"] == result["expected_contacts"] == 0
    assert result["contact_retention_fraction"] is None
    assert result["ca_distance_histogram"][2] == 1


def test_no_observed_ca_remains_explicit_and_invalid_crop_size_fails() -> None:
    xyz = np.full((7, 14, 3), np.nan)
    mask = np.zeros((7, 14))
    result = describe_crop_ca(xyz, mask, crop_size=4)
    assert result["expected_valid_ca_count"] == result["expected_contacts"] == 0
    assert result["normalized_radius_of_gyration"] is None
    assert result["ca_distance_histogram"] is None
    assert result["contact_retention_fraction"] is None
    assert result["empty_crop_probability"] == 1
    assert result["observed_ca_segments"] == []
    assert result["parent_valid_ca_count"] == 0
    invalid_sizes: tuple[Any, ...] = (0, -1, True, 1.5)
    for crop_size in invalid_sizes:
        with pytest.raises(ValueError, match="positive integer"):
            describe_crop_ca(xyz, mask, crop_size=crop_size)


def test_parent_coverage_does_not_hide_unsupervised_or_low_coverage_crops() -> None:
    xyz = np.zeros((1000, 14, 3))
    mask = np.zeros((1000, 14))
    mask[:700, 1] = 1
    result = describe_crop_ca(xyz, mask)
    assert mask[:, 1].mean() == 0.7
    # There are745 starts. Starts700..744 have no observed CA;669..744
    # contain fewer than32. This measures risk without changing eligibility.
    assert result["possible_starts"] == 745
    assert result["minimum_valid_ca_count"] == 0
    assert result["zero_valid_ca_crop_count"] == 45
    assert result["empty_crop_probability"] == pytest.approx(45 / 745)
    assert result["below_threshold_valid_ca_crop_count"] == 76
    assert result["below_threshold_valid_ca_crop_probability"] == pytest.approx(76 / 745)


def test_observed_ca_intervals_preserve_exact_parent_mask_and_count() -> None:
    xyz = np.full((8, 14, 3), np.nan)
    mask = np.zeros((8, 14))
    mask[[0, 1, 4, 6, 7], 1] = 1
    xyz[mask == 1] = 0
    result = describe_crop_ca(xyz, mask, crop_size=4)
    assert result["observed_ca_segments"] == [{"start": 1, "end": 2}, {"start": 5, "end": 5}, {"start": 7, "end": 8}]
    restored = np.zeros(8, dtype=bool)
    for segment in result["observed_ca_segments"]:
        restored[segment["start"] - 1:segment["end"]] = True
    assert np.array_equal(restored, mask[:, 1].astype(bool))
    assert restored.sum() == result["parent_valid_ca_count"] == 5
