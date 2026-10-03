"""Rigid-motion invariant, mask-aware descriptors of projected C-alpha atoms."""
from __future__ import annotations

from typing import Any

import numpy as np

DISTANCE_BIN_EDGES = (0.0, 4.0, 8.0, 12.0, 16.0, 24.0, 32.0, 48.0)


def _projected_ca(atom14_positions: Any, atom14_mask: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    positions = np.asarray(atom14_positions, dtype=np.float64)
    mask = np.asarray(atom14_mask)
    if positions.ndim != 3 or positions.shape[1:] != (14, 3) or mask.shape != positions.shape[:2]:
        raise ValueError("Geometry requires projected (L,14,3) positions and (L,14) masks")
    if len(positions) == 0 or not np.isfinite(mask).all() or not np.isin(mask, (0, 1)).all():
        raise ValueError("Geometry requires nonempty positions and finite binary atom masks")
    valid = mask[:, 1].astype(bool)
    ca = positions[valid, 1]
    if not np.isfinite(ca).all():
        raise ValueError("Observed C-alpha coordinates must be finite")
    return positions, valid, ca


def describe_projected_ca(atom14_positions: Any, atom14_mask: Any) -> dict[str, Any]:
    positions, valid, ca = _projected_ca(atom14_positions, atom14_mask)
    query_indices = np.flatnonzero(valid)
    count = len(ca)
    separation = np.abs(query_indices[:, None] - query_indices[None, :])
    pair_mask = np.triu(separation >= 3, k=1)
    distance = np.linalg.norm(ca[:, None] - ca[None, :], axis=-1)
    nonlocal_distances = distance[pair_mask]
    contact_mask = pair_mask & (distance < 8.0)
    contact_count = int(contact_mask.sum())
    contact_order = float((separation[contact_mask] / len(positions)).mean()) if contact_count else None
    radius = float(np.sqrt(np.square(ca - ca.mean(axis=0)).sum(axis=1).mean()) / np.sqrt(count)) if count else None
    histogram = np.histogram(nonlocal_distances, bins=(*DISTANCE_BIN_EDGES, np.inf))[0]
    normalized_histogram = (histogram / len(nonlocal_distances)).tolist() if len(nonlocal_distances) else None
    return {"schema_version": 1, "scope": "projected_chain_ca_nonlocal_geometry",
            "mapping_status": "verified", "radius_normalizer": "sqrt_observed_ca_count",
            "target_length": len(positions), "valid_ca_count": count,
            "valid_ca_fraction": count / len(positions), "minimum_sequence_separation": 3,
            "contact_cutoff_angstrom": 8.0, "contact_count": contact_count,
            "nonlocal_pair_count": len(nonlocal_distances),
            "normalized_radius_of_gyration": radius, "relative_contact_order": contact_order,
            "distance_bin_lower_edges_angstrom": list(DISTANCE_BIN_EDGES),
            "last_distance_bin_unbounded": True, "ca_distance_histogram": normalized_histogram}


def _parent_contact_count(ca: np.ndarray, query_indices: np.ndarray) -> int:
    """Count strict 8-Angstrom contacts with bounded intermediate memory."""
    total = 0
    for first in range(0, len(ca), 128):
        last = min(first + 128, len(ca))
        # Upper-triangular query separation keeps each observed pair once and
        # preserves the original sequence separation across missing residues.
        eligible = query_indices[None, :] - query_indices[first:last, None] >= 3
        delta = ca[first:last, None] - ca[None, :]
        squared_distance = np.einsum("ijk,ijk->ij", delta, delta)
        total += int(np.count_nonzero(eligible & (squared_distance < 64.0)))
    return total


def describe_crop_ca(
    atom14_positions: Any, atom14_mask: Any, *, crop_size: int = 256,
) -> dict[str, Any]:
    """Measure exact geometry exposure under uniform contiguous crops.

    Contact and pair expectations average all integer crop starts. Histograms
    and contact order aggregate expected pair counts, rather than averaging
    individually normalized crop histograms. Radius is averaged over crops
    having at least one observed CA; empty-crop probability is reported.

    Crop pairs take O(L * min(L, crop_size)) work. The parent contact count is
    computed in bounded-memory blocks, never an L-by-L coordinate tensor.
    """
    if isinstance(crop_size, bool) or not isinstance(crop_size, int) or crop_size <= 0:
        raise ValueError("Crop size must be a positive integer")
    positions, valid, ca = _projected_ca(atom14_positions, atom14_mask)
    length = len(positions)
    width = min(length, crop_size)
    starts = length - width + 1
    indices = np.arange(length)
    query_indices = np.flatnonzero(valid)
    parent_contacts = _parent_contact_count(ca, query_indices)

    # Centering before prefix sums avoids loss of translation invariance from
    # subtracting large squared coordinate offsets when measuring crop radius.
    centered = np.zeros((length, 3), dtype=np.float64)
    if len(ca):
        centered[valid] = ca - ca.mean(axis=0)
    prefix_count = np.r_[0, np.cumsum(valid, dtype=np.int64)]
    crop_counts = prefix_count[width:] - prefix_count[:-width]
    prefix_position = np.vstack((np.zeros((1, 3)), np.cumsum(centered, axis=0)))
    crop_sum = prefix_position[width:] - prefix_position[:-width]
    prefix_square = np.r_[0.0, np.cumsum(np.square(centered).sum(axis=1))]
    crop_square = prefix_square[width:] - prefix_square[:-width]
    observed = crop_counts > 0
    boundaries = np.diff(np.r_[0, valid.astype(np.int8), 0])
    observed_segments = [{"start": int(first + 1), "end": int(last)}
                         for first, last in zip(np.flatnonzero(boundaries == 1), np.flatnonzero(boundaries == -1), strict=True)]
    if observed.any():
        counts = crop_counts[observed]
        centered_square = crop_square[observed] - np.square(crop_sum[observed]).sum(axis=1) / counts
        mean_radius = float(np.mean(np.sqrt(np.maximum(centered_square, 0.0)) / counts))
    else:
        mean_radius = None

    histogram = np.zeros(len(DISTANCE_BIN_EDGES), dtype=np.float64)
    expected_contacts = 0.0
    expected_pairs = 0.0
    contact_separation_sum = 0.0
    for separation in range(3, width):
        eligible = valid[:-separation] & valid[separation:]
        first = indices[:-separation][eligible]
        last = first + separation
        # A zero-based start s includes both residues exactly when
        # last-width+1 <= s <= first and 0 <= s < starts.
        multiplicity = np.maximum(0, np.minimum(first, starts - 1) - np.maximum(0, last - width + 1) + 1)
        weights = multiplicity / starts
        distances = np.linalg.norm(centered[first] - centered[last], axis=1)
        histogram += np.histogram(distances, bins=(*DISTANCE_BIN_EDGES, np.inf), weights=weights)[0]
        pair_mass = float(weights.sum())
        contact_mass = float(weights[distances < 8.0].sum())
        expected_pairs += pair_mass
        expected_contacts += contact_mass
        contact_separation_sum += contact_mass * separation / width
    return {
        "schema_version": 1,
        "scope": "projected_contiguous_crop_ca_nonlocal_geometry",
        "mapping_status": "verified",
        "sampling": "uniform_contiguous_integer_start_v1",
        "parent_length": length,
        "parent_valid_ca_count": len(ca),
        "observed_ca_segments": observed_segments,
        "crop_size": crop_size,
        "effective_crop_size": width,
        "possible_starts": starts,
        "expected_valid_ca_count": float(crop_counts.mean()),
        "expected_valid_ca_fraction": float(crop_counts.mean() / width),
        "minimum_valid_ca_count": int(crop_counts.min()),
        "zero_valid_ca_crop_count": int(np.count_nonzero(crop_counts == 0)),
        "empty_crop_probability": float((~observed).mean()),
        "valid_ca_diagnostic_threshold": 32,
        "below_threshold_valid_ca_crop_count": int(np.count_nonzero(crop_counts < 32)),
        "below_threshold_valid_ca_crop_probability": float(np.mean(crop_counts < 32)),
        "minimum_sequence_separation": 3,
        "contact_cutoff_angstrom": 8.0,
        "expected_contacts": expected_contacts,
        "parent_contact_count": parent_contacts,
        "contact_retention_fraction": expected_contacts / parent_contacts if parent_contacts else None,
        "expected_nonlocal_pair_count": expected_pairs,
        "relative_contact_order": contact_separation_sum / expected_contacts if expected_contacts else None,
        "contact_order_aggregation": "expected_contact_weighted_crop_relative_separation",
        "normalized_radius_of_gyration": mean_radius,
        "radius_normalizer": "sqrt_observed_ca_count",
        "radius_aggregation": "mean_over_crops_with_observed_ca",
        "distance_bin_lower_edges_angstrom": list(DISTANCE_BIN_EDGES),
        "last_distance_bin_unbounded": True,
        "ca_distance_histogram": (histogram / expected_pairs).tolist() if expected_pairs else None,
        "distance_histogram_aggregation": "expected_pair_counts_normalized",
    }
