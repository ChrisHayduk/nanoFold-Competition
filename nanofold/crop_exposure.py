"""Exact expectations for uniformly sampled contiguous sequence crops.

Residue intervals are one-based and inclusive, matching validated CATH label
segments. Expectations average all possible starts; no crop seed or sampled
quadrature is used. Missing boundaries do not acquire inferred domain labels.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from typing import Any


def _dimensions(length: int, crop_size: int) -> tuple[int, int]:
    if isinstance(length, bool) or not isinstance(length, int) or length <= 0:
        raise ValueError("Parent length must be a positive integer")
    if isinstance(crop_size, bool) or not isinstance(crop_size, int) or crop_size <= 0:
        raise ValueError("Crop size must be a positive integer")
    width = min(length, crop_size)
    return width, length - width + 1


def crop_pair_probability(length: int, crop_size: int, first: int, last: int) -> float:
    """Probability that a crop contains both one-based residue positions."""
    width, starts = _dimensions(length, crop_size)
    if any(isinstance(value, bool) or not isinstance(value, int) for value in (first, last)) or not 1 <= first <= last <= length:
        raise ValueError("Residue pair must be ordered and within the parent")
    # A one-based start must be <= first and >= last-width+1.
    return max(0, min(first, starts) - max(1, last - width + 1) + 1) / starts


def _triangle(value: int) -> int:
    value = max(value, 0)
    return value * (value + 1) // 2


def _residue_start_pairs(end: int, width: int, starts: int) -> int:
    # Sum_{r=1..end} number of crop starts covering r. The trapezoidal
    # convolution is the finite difference of a triangular-number function.
    return _triangle(end) - _triangle(end - width) - _triangle(end - starts) + _triangle(end - width - starts)


def _segments(intervals: Sequence[Mapping[str, Any]], length: int) -> list[tuple[int, int]]:
    result = []
    for interval in intervals:
        first, last = interval["start"], interval["end"]
        if (
            isinstance(first, bool) or isinstance(last, bool)
            or not isinstance(first, int) or not isinstance(last, int)
            or not 1 <= first <= last <= length
        ):
            raise ValueError("Label segments require one-based inclusive integer positions within the parent")
        result.append((first, last))
    result.sort()
    if any(left[1] >= right[0] for left, right in zip(result, result[1:], strict=False)):
        raise ValueError("Domain label segments overlap")
    return result


def _start_intervals(segments: Sequence[tuple[int, int]], width: int, starts: int) -> list[tuple[int, int]]:
    admissible = sorted((max(1, first - width + 1), min(starts, last)) for first, last in segments)
    merged: list[tuple[int, int]] = []
    for first, last in admissible:
        if merged and first <= merged[-1][1] + 1:
            merged[-1] = merged[-1][0], max(merged[-1][1], last)
        else:
            merged.append((first, last))
    return merged


def domain_crop_exposure(length: int, crop_size: int, label_segments: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    """Expected residue exposure and domain intactness, including discontinuity.

    An intact discontinuous domain requires its entire sequence span to fit;
    fitting only the sum of its segment lengths is insufficient.
    """
    width, starts = _dimensions(length, crop_size)
    segments = _segments(label_segments, length)
    if not segments:
        raise ValueError("Domain crop exposure requires mapped label segments")
    residue_pairs = sum(
        _residue_start_pairs(last, width, starts) - _residue_start_pairs(first - 1, width, starts)
        for first, last in segments
    )
    covered_starts = sum(last - first + 1 for first, last in _start_intervals(segments, width, starts))
    present = covered_starts / starts
    intact = crop_pair_probability(length, crop_size, segments[0][0], segments[-1][1])
    return {
        "expected_residues": residue_pairs / starts,
        "expected_residue_fraction": residue_pairs / (starts * width),
        "presence_probability": present,
        "intact_probability": intact,
        "fragment_probability": present - intact,
    }


def chain_crop_exposure(record: Mapping[str, Any], *, crop_size: int = 256) -> dict[str, Any]:
    """Return verified domain exposure; all remaining residue mass is unknown.

    Unmapped domains retain their IDs in diagnostics, but receive no location or
    exposure estimate. Overlap between mapped domains fails instead of counting
    the same residue twice.
    """
    length = record["length"]
    width, starts = _dimensions(length, crop_size)
    domains: list[dict[str, Any]] = []
    mapped_segments: list[tuple[int, int]] = []
    mapped_fraction = 0.0
    events: dict[int, list[tuple[int, bool]]] = {1: [], starts + 1: []}
    for domain in record.get("cath_domains", []):
        item = {key: domain.get(key) for key in (
            "domain_id", "topology_id", "homologous_superfamily_id", "domain_length",
        )}
        intervals = domain.get("label_segments")
        if domain.get("domain_length") is None:
            item["exposure"] = None
        else:
            if not intervals:
                raise ValueError("A mapped domain length requires verified label segments for crop exposure")
            segments = _segments(intervals, length)
            if sum(last - first + 1 for first, last in segments) != domain["domain_length"]:
                raise ValueError("Mapped domain length does not match its label segments")
            mapped_segments.extend(segments)
            for first, last in _start_intervals(segments, width, starts):
                events.setdefault(first, []).append((len(domains), True))
                events.setdefault(last + 1, []).append((len(domains), False))
            item["exposure"] = domain_crop_exposure(length, crop_size, intervals)
            mapped_fraction += item["exposure"]["expected_residue_fraction"]
        item["presence_by_mapped_domain_count"] = {}
        domains.append(item)
    mapped_segments.sort()
    if any(left[1] >= right[0] for left, right in zip(mapped_segments, mapped_segments[1:], strict=False)):
        raise ValueError("Mapped domains overlap; crop annotation mass would be ambiguous")
    active: set[int] = set()
    domain_count_probabilities: dict[str, float] = {}
    composition_probabilities: dict[str, float] = {}
    positions = sorted(events)
    for first, next_first in zip(positions, positions[1:], strict=False):
        for index, present in events[first]:
            if present:
                active.add(index)
            else:
                active.remove(index)
        probability = (next_first - first) / starts
        count_bin = str(len(active)) if len(active) < 4 else "4+"
        domain_count_probabilities[count_bin] = domain_count_probabilities.get(count_bin, 0.0) + probability
        labels: dict[str, int] = {}
        for index in sorted(active):
            label = str(domains[index]["topology_id"] or "unknown")
            labels[label] = labels.get(label, 0) + 1
            joint = domains[index]["presence_by_mapped_domain_count"]
            joint[count_bin] = joint.get(count_bin, 0.0) + probability
        composition = ";".join(f"{label}*{count}" for label, count in sorted(labels.items())) or "no_mapped_domain"
        composition_probabilities[composition] = composition_probabilities.get(composition, 0.0) + probability
    return {
        "crop_sampling_policy": "uniform_contiguous_integer_start_v1",
        "parent_length": length,
        "requested_crop_size": crop_size,
        "effective_crop_size": width,
        "possible_starts": starts,
        "domains": domains,
        "expected_unknown_residue_fraction": max(0.0, 1.0 - mapped_fraction),
        "unmapped_domain_count": sum(item["exposure"] is None for item in domains),
        "mapped_domain_count_probabilities": domain_count_probabilities,
        "mapped_domain_composition_probabilities": composition_probabilities,
    }


def chain_observed_ca_crop_exposure(record: Mapping[str, Any], *, crop_size: int = 256) -> dict[str, Any]:
    """Intersect annotated sequence exposure with the measured CA mask.

    Each family partitions the expected crop width into labeled observed CA,
    unknown observed CA, and positions without observed CA. This is a measure
    of CA observation, not complete backbone or all-atom supervision. Missing
    measured masks are unavailable, rather than treated as unobserved atoms.
    """
    length = record["length"]
    width, starts = _dimensions(length, crop_size)
    geometry = record.get("crop_geometry")
    if geometry is None or (isinstance(geometry, Mapping) and "observed_ca_segments" not in geometry):
        return {"status": "unavailable", "reason": "measured_parent_CA_intervals_not_supplied"}
    if not isinstance(geometry, Mapping) or any((
        geometry.get("mapping_status") != "verified",
        geometry.get("sampling") != "uniform_contiguous_integer_start_v1",
        geometry.get("parent_length") != length,
        geometry.get("crop_size") != crop_size,
        geometry.get("effective_crop_size") != width,
    )):
        raise ValueError("Observed CA geometry does not match the parent and crop contract")
    intervals = geometry["observed_ca_segments"]
    if not isinstance(intervals, (list, tuple)) or any(not isinstance(item, Mapping) for item in intervals):
        raise ValueError("Observed CA segments must be a list of measured intervals")
    observed = _segments(intervals, length)
    if (observed != [(item["start"], item["end"]) for item in intervals]
            or any(left[1] + 1 == right[0] for left, right in zip(observed, observed[1:], strict=False))):
        raise ValueError("Observed CA segments must be sorted maximal intervals")
    count = geometry.get("parent_valid_ca_count")
    if type(count) is not int or count != sum(last - first + 1 for first, last in observed):
        raise ValueError("Observed CA segments disagree with the measured parent count")

    def fraction(segments: Sequence[tuple[int, int]]) -> float:
        return sum(_residue_start_pairs(last, width, starts) - _residue_start_pairs(first - 1, width, starts)
                   for first, last in segments) / (starts * width)

    observed_fraction = fraction(observed)
    expected_count = geometry.get("expected_valid_ca_count")
    if (isinstance(expected_count, bool) or not isinstance(expected_count, (int, float))
            or not math.isclose(observed_fraction * width, expected_count, rel_tol=0, abs_tol=1e-9)):
        raise ValueError("Observed CA intervals disagree with measured crop exposure")
    # Reuse the owning domain validator, including overlap and length checks.
    chain_crop_exposure(record, crop_size=crop_size)
    families: dict[str, dict[str, float]] = {
        family: {"unknown_observed": observed_fraction, "unobserved_ca": 1.0 - observed_fraction}
        for family in ("topology", "superfamily", "architecture")
    }
    mapped_observed = 0.0
    for domain in record.get("cath_domains", []):
        if domain.get("domain_length") is None:
            continue
        domain_segments = _segments(domain["label_segments"], length)
        intersection = [(max(first, ca_first), min(last, ca_last))
                        for first, last in domain_segments for ca_first, ca_last in observed
                        if max(first, ca_first) <= min(last, ca_last)]
        exposure = fraction(intersection)
        mapped_observed += exposure
        topology = domain.get("topology_id")
        superfamily = domain.get("homologous_superfamily_id")
        for label, parts in ((topology, 3), (superfamily, 4)):
            if label not in (None, "unknown") and (
                not isinstance(label, str) or re.fullmatch(r"[1-9]\d*(?:\.[1-9]\d*){" + str(parts - 1) + r"}", label) is None
            ):
                raise ValueError("Observed CA exposure requires valid numeric CATH labels")
        if (topology not in (None, "unknown") and superfamily not in (None, "unknown")
                and superfamily.rsplit(".", 1)[0] != topology):
            raise ValueError("Observed CA CATH topology and superfamily disagree")
        labels = {"topology": topology, "superfamily": superfamily,
                  "architecture": str(topology).rsplit(".", 1)[0] if topology else None}
        for family, label in labels.items():
            if label is not None and label != "unknown":
                values = families[family]
                values[str(label)] = values.get(str(label), 0.0) + exposure
                values["unknown_observed"] = max(0.0, values["unknown_observed"] - exposure)
    return {
        "status": "available",
        "scope": "expected_CA_observation_fraction_per_crop_position_not_all_atom_supervision",
        "crop_sampling_policy": "uniform_contiguous_integer_start_v1",
        "parent_length": length,
        "crop_size": crop_size,
        "effective_crop_size": width,
        "expected_observed_ca_fraction": observed_fraction,
        "expected_mapped_observed_ca_fraction": mapped_observed,
        "expected_unknown_observed_ca_fraction": max(0.0, observed_fraction - mapped_observed),
        "expected_unobserved_ca_fraction": 1.0 - observed_fraction,
        "families": families,
    }
