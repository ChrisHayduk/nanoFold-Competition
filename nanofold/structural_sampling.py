"""Select source-qualified chains against the pre-collapse structural population.

The input is the complete source-qualified training population. This module
does not assign split ownership, relax source qualification, or fit training
weights. Distinct sequences have equal reference probability by default; an
explicit independently supplied reference measure can also be diagnosed.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from scipy.sparse import csr_matrix

from nanofold.crop_exposure import chain_crop_exposure

POLICY = "topology_moments_group_coverage_v1"
# Predeclared priorities, shared by selection and its diagnostic report.
FAMILY_WEIGHTS = {
    "topology": 4.0,
    "domain_count": 2.0,
    "superfamily": 1.0,
    "topology_domain_length": 1.0,
    "topology_domain_count": 1.0,
    "domain_composition": 1.0,
    "annotation_coverage": 1.0,
    "chain_length": 1.0,
    "architecture": 0.5,
    "topology_geometry": 0.5,
    "resolution": 0.25,
    "topology_presence": 1.0,
    "topology_intact": 1.0,
    "topology_fragment": 1.0,
    "crop_observation": 1.0,
    "crop_contact_exposure": 0.5,
    "parent_length": 0.25,
    "diversity_group": 1.0,
}


def _bin(value: float, edges: Sequence[float]) -> str:
    bounds = (0.0, *edges, math.inf)
    for low, high in zip(bounds, bounds[1:], strict=False):
        if low <= value < high:
            return f"[{low:g},{high:g})"
    raise ValueError(f"Non-finite or negative value cannot be binned: {value}")


def _quality_resolution(record: Mapping[str, Any]) -> float:
    value = record.get("resolution")
    if value is None:
        return math.inf
    resolution = float(value)
    if not math.isfinite(resolution) or resolution <= 0:
        raise ValueError(f"Invalid resolution for {record['chain_id']}")
    return resolution


def structural_moments(record: Mapping[str, Any]) -> dict[str, Counter[str]]:
    """Return chain contributions; repeated domains contribute repeatedly.

    Domain lengths must come from verified mappings supplied by metadata prep.
    Unknown boundaries are retained as unknown, never inferred from chain length.
    These are chain/domain-instance moments, not residue-crop exposure estimates.
    """
    out: dict[str, Counter[str]] = defaultdict(Counter)
    domains = record.get("cath_domains", [])
    if not isinstance(domains, (list, tuple)):
        raise ValueError("cath_domains must be a list of domain records")
    domain_count = len(domains)
    count_bin = str(domain_count) if domain_count < 4 else "4+"
    out["domain_count"][count_bin] = 1
    out["chain_length"][_bin(float(record["length"]), (64, 96, 128, 192, 256))] = 1
    resolution = _quality_resolution(record)
    out["resolution"]["unknown" if math.isinf(resolution) else _bin(resolution, (1.5, 2.0, 2.5, 3.0))] = 1
    seen_ids: set[str] = set()
    composition: Counter[str] = Counter()
    for domain in domains:
        domain_id = domain.get("domain_id")
        if domain_id is not None:
            if domain_id in seen_ids:
                raise ValueError(f"Repeated domain ID: {domain_id}")
            seen_ids.add(domain_id)
        topology = str(domain.get("topology_id") or "unknown")
        superfamily = str(domain.get("homologous_superfamily_id") or "unknown")
        if topology != "unknown" and not re.fullmatch(r"[1-9]\d*\.[1-9]\d*\.[1-9]\d*", topology):
            raise ValueError(f"Invalid CATH topology: {topology}")
        if superfamily != "unknown":
            if not re.fullmatch(r"[1-9]\d*\.[1-9]\d*\.[1-9]\d*\.[1-9]\d*", superfamily):
                raise ValueError(f"Invalid CATH superfamily: {superfamily}")
            if topology != "unknown" and superfamily.rsplit(".", 1)[0] != topology:
                raise ValueError("CATH topology and superfamily disagree")
        architecture = topology.rsplit(".", 1)[0] if topology != "unknown" else "unknown"
        length = domain.get("domain_length")
        if length is not None and (isinstance(length, bool) or not isinstance(length, int) or length <= 0):
            raise ValueError("Mapped domain length must be a positive integer or None")
        length_bin = "unknown" if length is None else _bin(length, (40, 80, 120, 160, 240))
        out["topology"][topology] += 1
        out["superfamily"][superfamily] += 1
        out["architecture"][architecture] += 1
        out["topology_domain_length"][f"{topology}|{length_bin}"] += 1
        out["topology_domain_count"][f"{topology}|{count_bin}"] += 1
        composition[topology] += 1
    if not domains:
        for family in ("topology", "superfamily", "architecture"):
            out[family]["unknown"] = 1
        out["topology_domain_length"]["unknown|unknown"] = 1
        out["topology_domain_count"]["unknown|0"] = 1
    composition_label = ";".join(f"{key}*{value}" for key, value in sorted(composition.items())) or "unknown"
    out["domain_composition"][composition_label] = 1
    unannotated = record.get("cath_unannotated_fraction")
    if unannotated is None:
        coverage = "unknown"
    else:
        value = float(unannotated)
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError("cath_unannotated_fraction must be in [0, 1]")
        coverage = "fully_annotated" if value == 0 else _bin(value, (0.25, 0.5, 0.75, 1.0))
    out["annotation_coverage"][coverage] = 1
    # Whole-chain descriptors can stand in for a domain only when a single
    # mapped domain occupies nearly the entire observed chain.
    geometry = record.get("source_geometry") or {}
    domain_length = domains[0].get("domain_length") if len(domains) == 1 else None
    geometry_label = "unavailable"
    if (
        geometry.get("mapping_status") == "verified"
        and domain_length is not None
        and domain_length >= 0.9 * record["length"]
        and float(geometry.get("valid_ca_fraction", 0)) >= 0.9
    ):
        fraction = float(geometry["valid_ca_fraction"])
        radius = float(geometry["normalized_radius_of_gyration"])
        if not math.isfinite(fraction) or fraction > 1 or not math.isfinite(radius) or radius < 0:
            raise ValueError("Invalid verified CA geometry")
        contact_order = geometry.get("relative_contact_order")
        if contact_order is None:
            if geometry.get("contact_count") != 0:
                raise ValueError("Missing contact order for a chain with contacts")
            order_bin = "no_contacts"
        else:
            order = float(contact_order)
            if not math.isfinite(order) or not 0 <= order <= 1:
                raise ValueError("Invalid relative contact order")
            order_bin = _bin(order, (0.1, 0.2, 0.3, 0.4, 0.6))
        topology = str(domains[0].get("topology_id") or "unknown")
        if topology != "unknown":
            geometry_label = f"{topology}|rg:{_bin(radius, (1, 1.5, 2, 2.5, 3, 4))}|co:{order_bin}"
    out["topology_geometry"][geometry_label] = 1
    return dict(out)


def crop_structural_moments(record: Mapping[str, Any], *, crop_size: int = 256) -> dict[str, dict[str, float]]:
    """Expected per-crop structural contributions without family normalization.

    Topology/superfamily are residue fractions. Presence, intactness and
    fragmentation are expected numbers of domain instances and can exceed one.
    All expectations use every permitted contiguous crop start exactly.
    """
    parent = structural_moments(record)  # Validate scientific labels and missingness.
    crop = chain_crop_exposure(record, crop_size=crop_size)
    out: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    width = crop["effective_crop_size"]
    out["chain_length"][_bin(width, (64, 96, 128, 192, 256))] = 1.0
    out["parent_length"][_bin(record["length"], (64, 128, 256, 512, 1024, 2048, 4096))] = 1.0
    out["resolution"].update(parent["resolution"])
    out["domain_count"].update(crop["mapped_domain_count_probabilities"])
    out["domain_composition"].update(crop["mapped_domain_composition_probabilities"])
    unknown = crop["expected_unknown_residue_fraction"]
    for family in ("topology", "superfamily", "architecture"):
        out[family]["unknown"] = unknown
    out["annotation_coverage"].update({"mapped": 1.0 - unknown, "unknown": unknown})
    for domain in crop["domains"]:
        exposure = domain["exposure"]
        if exposure is None:
            continue
        topology = str(domain["topology_id"] or "unknown")
        superfamily = str(domain["homologous_superfamily_id"] or "unknown")
        architecture = topology.rsplit(".", 1)[0] if topology != "unknown" else "unknown"
        fraction = exposure["expected_residue_fraction"]
        out["topology"][topology] += fraction
        out["superfamily"][superfamily] += fraction
        out["architecture"][architecture] += fraction
        for family, field in (("topology_presence", "presence_probability"),
                              ("topology_intact", "intact_probability"),
                              ("topology_fragment", "fragment_probability")):
            out[family][topology] += exposure[field]
        length_bin = _bin(domain["domain_length"], (40, 80, 120, 160, 240, 320, 512))
        out["topology_domain_length"][f"{topology}|{length_bin}"] += fraction
        for count_bin, probability in domain["presence_by_mapped_domain_count"].items():
            out["topology_domain_count"][f"{topology}|{count_bin}"] += probability
    out["topology_domain_length"]["unknown|unknown"] = unknown
    geometry = record.get("crop_geometry")
    if geometry is None:
        out["crop_observation"]["unavailable"] = 1.0
        out["crop_contact_exposure"]["unavailable"] = 1.0
    else:
        if (
            geometry.get("sampling") != "uniform_contiguous_integer_start_v1"
            or geometry.get("mapping_status") != "verified"
            or geometry.get("crop_size") != crop_size
            or geometry.get("effective_crop_size") != width
            or geometry.get("parent_length") != record["length"]
        ):
            raise ValueError("Crop geometry does not match the declared parent and crop sampler")
        observed = float(geometry["expected_valid_ca_count"])
        contacts = float(geometry["expected_contacts"])
        if not math.isfinite(observed) or not 0 <= observed <= width or not math.isfinite(contacts) or contacts < 0:
            raise ValueError("Invalid expected crop geometry exposure")
        out["crop_observation"].update({"observed_ca": observed / width, "missing_ca": 1 - observed / width})
        out["crop_contact_exposure"]["contacts_per_residue"] = contacts / width
    return {family: dict(values) for family, values in out.items()}


def qualified_sequence_reference(records: Sequence[Mapping[str, Any]], *, seed: int = 0) -> list[Mapping[str, Any]]:
    """Resolve deposited copies by source quality without collapsing groups.

    Split prep must supply admitted training inputs only. Group labels describe
    the caller's declared grouping policy; these checks cannot reconstruct an
    admission barrier or establish transitive train/held-out separation.
    """
    by_sequence: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    chain_ids: set[str] = set()
    for record in records:
        chain_id = record["chain_id"]
        sequence = record["sequence"]
        group = record["group_id"]
        if not isinstance(chain_id, str) or not chain_id or chain_id in chain_ids:
            raise ValueError("Chain IDs must be nonempty and unique")
        if not isinstance(sequence, str) or not sequence or len(sequence) != record["length"]:
            raise ValueError(f"Sequence/length mismatch for {chain_id}")
        if not isinstance(group, str) or not group:
            raise ValueError("Every candidate requires an explicit group identifier")
        _quality_resolution(record)
        chain_ids.add(chain_id)
        by_sequence[sequence].append(record)
    reference: list[Mapping[str, Any]] = []
    for copies in by_sequence.values():
        if len({record["group_id"] for record in copies}) != 1:
            raise ValueError("Exact sequence spans supplied groups")
        reference.append(min(copies, key=lambda row: (_quality_resolution(row), row["chain_id"])))

    def order(record: Mapping[str, Any]) -> tuple[float, str, str]:
        tie = hashlib.sha256(f"{seed}\0{record['chain_id']}".encode()).hexdigest()
        return _quality_resolution(record), tie, record["chain_id"]

    return sorted(reference, key=order)


def select_structural_sample(
    records: Sequence[Mapping[str, Any]],
    *,
    size: int,
    seed: int = 0,
    swap_passes: int = 2,
    max_swap_attempts: int = 2048,
    reference_weights: Mapping[str, float] | None = None,
    reference_weighting: str = "uniform_distinct_sequence",
    crop_size: int | None = None,
    group_policy: str = "all_groups",
) -> tuple[list[str], dict[str, Any]]:
    """Choose exactly ``size`` unique sequences under an explicit group policy.

    Greedy anchors and additions use the same progressive moment objective.
    Anchors are replaceable: bounded coordinate swaps minimize the final-size
    objective subject to group coverage. The algorithm is a heuristic, not a
    claim of optimality. Time is bounded without an all-pairs similarity matrix.
    """
    if size <= 0 or swap_passes < 0 or max_swap_attempts < 0:
        raise ValueError("size must be positive; swap controls must be nonnegative")
    if group_policy not in {"all_groups", "soft_diversity"}:
        raise ValueError("group_policy must be all_groups or soft_diversity")
    reference = qualified_sequence_reference(records, seed=seed)
    population = len(reference)
    groups = sorted({record["group_id"] for record in reference})
    if size > population or (group_policy == "all_groups" and len(groups) > size):
        mandatory = len(groups) if group_policy == "all_groups" else 0
        raise ValueError(f"Infeasible sample size {size}: {mandatory} mandatory groups, {population} distinct sequences")
    group_ids = {group: index for index, group in enumerate(groups)}
    group = np.asarray([group_ids[row["group_id"]] for row in reference], dtype=np.int64)
    if reference_weights is None:
        if reference_weighting != "uniform_distinct_sequence":
            raise ValueError("A nonuniform reference label requires independently supplied weights")
        probabilities = np.full(population, 1.0 / population)
    else:
        if set(reference_weights) != {row["chain_id"] for row in reference}:
            raise ValueError("Reference weights must cover exactly the qualified distinct-sequence representatives")
        probabilities = np.asarray([float(reference_weights[row["chain_id"]]) for row in reference])
        if not np.isfinite(probabilities).all() or np.any(probabilities < 0) or not np.any(probabilities > 0):
            raise ValueError("Reference weights must be finite, nonnegative, and have positive total mass")
        probabilities /= math.fsum(probabilities)
        if reference_weighting == "uniform_distinct_sequence" and not np.allclose(probabilities, 1.0 / population, rtol=0, atol=1e-15):
            raise ValueError("Nonuniform supplied weights require an explicit reference_weighting label")
    features: list[Mapping[str, Mapping[str, float]]]
    if crop_size is None:
        features = [structural_moments(row) for row in reference]
    else:
        features = [crop_structural_moments(row, crop_size=crop_size) for row in reference]
    if group_policy == "soft_diversity":
        features = [{**feature, "diversity_group": {row["group_id"]: 1.0}}
                    for row, feature in zip(reference, features, strict=True)]
    keys = sorted({(family, label) for feature in features for family, labels in feature.items() for label in labels})
    columns = {key: column for column, key in enumerate(keys)}
    values: list[float] = []
    indices: list[int] = []
    indptr = [0]
    for feature in features:
        for family, labels in feature.items():
            for label, count in labels.items():
                indices.append(columns[family, label])
                values.append(float(count))
        indptr.append(len(indices))
    matrix = csr_matrix((values, indices, indptr), shape=(population, len(keys)), dtype=np.float64)
    matrix.sort_indices()
    total = np.asarray(matrix.sum(axis=0)).ravel()
    mean = np.asarray(matrix.T.dot(probabilities)).ravel()
    target = mean * size
    # A positive minimum in every member of a group is unavoidable because one
    # member must be selected. This is a lower bound, not a feasibility solver.
    group_minima: dict[int, dict[int, float]] = {}
    for row_index, group_index in enumerate(group):
        start, end = matrix.indptr[row_index:row_index + 2]
        contributions = dict(zip(matrix.indices[start:end], matrix.data[start:end], strict=True))
        if int(group_index) not in group_minima:
            group_minima[int(group_index)] = contributions
        else:
            current = group_minima[int(group_index)]
            for column in list(current):
                if column not in contributions:
                    del current[column]
                else:
                    current[column] = min(current[column], contributions[column])
    minimum = np.zeros(len(keys))
    if group_policy == "all_groups":
        for contributions in group_minima.values():
            for column, contribution in contributions.items():
                minimum[column] += contribution
    family_mass: dict[str, float] = defaultdict(float)
    for (family, _), expected in zip(keys, target, strict=True):
        family_mass[family] += float(expected)
    weight = np.asarray([
        FAMILY_WEIGHTS[family] / (max(family_mass[family], 1.0) * max(float(expected), 1.0))
        for (family, _), expected in zip(keys, target, strict=True)
    ])
    norm = np.asarray(matrix.multiply(matrix).dot(weight)).ravel()
    drift = 2 * matrix.dot(weight * mean)
    scores = norm - drift
    counts = np.zeros(len(keys))
    selected = np.zeros(population, dtype=bool)
    group_counts = np.zeros(len(groups), dtype=np.int64)
    row_vector = np.zeros(len(keys))

    def row_terms(index: int) -> tuple[np.ndarray, np.ndarray]:
        start, end = matrix.indptr[index:index + 2]
        return matrix.indices[start:end], matrix.data[start:end]

    for step in range(size):
        eligible = ~selected
        if group_policy == "all_groups" and step < len(groups):
            eligible &= group_counts[group] == 0
        index = int(np.argmin(np.where(eligible, scores, np.inf)))
        selected[index] = True
        group_counts[group[index]] += 1
        cols, vals = row_terms(index)
        counts[cols] += vals
        row_vector[cols] = weight[cols] * vals
        scores += 2 * matrix.dot(row_vector) - drift
        row_vector[cols] = 0
    greedy_objective = float(np.dot(weight, (counts - target) ** 2))
    addition = norm + 2 * matrix.dot(weight * (counts - target))
    accepted = attempts = 0
    passes_completed = 0
    for _ in range(swap_passes):
        before = accepted
        # Prioritize the selected chains whose removal most reduces discrepancy.
        removal = norm - 2 * matrix.dot(weight * (counts - target))
        order = np.flatnonzero(selected)
        order = order[np.argsort(removal[order], kind="stable")]
        for outgoing in order:
            if attempts >= max_swap_attempts:
                break
            attempts += 1
            cols, vals = row_terms(int(outgoing))
            row_vector[cols] = weight[cols] * vals
            after_removal = addition - 2 * matrix.dot(row_vector)
            row_vector[cols] = 0
            eligible = ~selected
            if group_policy == "all_groups" and group_counts[group[outgoing]] == 1:
                eligible &= group == group[outgoing]
            if not eligible.any():
                continue
            incoming = int(np.argmin(np.where(eligible, after_removal, np.inf)))
            removal_cost = norm[outgoing] - 2 * np.dot(vals * weight[cols], (counts - target)[cols])
            if float(removal_cost + after_removal[incoming]) >= -1e-12:
                continue
            in_cols, in_vals = row_terms(incoming)
            counts[cols] -= vals
            counts[in_cols] += in_vals
            selected[outgoing] = False
            selected[incoming] = True
            group_counts[group[outgoing]] -= 1
            group_counts[group[incoming]] += 1
            row_vector[cols] -= weight[cols] * vals
            row_vector[in_cols] += weight[in_cols] * in_vals
            addition += 2 * matrix.dot(row_vector)
            row_vector[cols] = 0
            row_vector[in_cols] = 0
            accepted += 1
        passes_completed += 1
        if accepted == before or attempts >= max_swap_attempts:
            break
    chosen = np.flatnonzero(selected)
    if len(chosen) != size or (group_policy == "all_groups" and np.any(group_counts == 0)):
        raise RuntimeError("Structural allocation violated size or group coverage")
    counts = np.asarray(matrix[chosen].sum(axis=0)).ravel()
    family_report: dict[str, Any] = {}
    for family in FAMILY_WEIGHTS:
        subset = np.asarray([i for i, key in enumerate(keys) if key[0] == family], dtype=np.int64)
        if not len(subset):
            continue
        family_report[family] = {
            "weight": FAMILY_WEIGHTS[family],
            "objective": float(np.dot(weight[subset], (counts[subset] - target[subset]) ** 2)),
            "reference_categories": int(np.count_nonzero(total[subset])),
            "selected_categories": int(np.count_nonzero(counts[subset])),
            "missing_reference_mass": (float(mean[subset[counts[subset] <= 1e-12]].sum() / mean[subset].sum())
                                       if mean[subset].sum() > 0 else None),
            "reference_feature_mass_per_crop_or_chain": float(mean[subset].sum()),
            "selected_feature_mass_per_crop_or_chain": float(counts[subset].sum() / size),
            "targets_below_mandatory_group_minimum": int(np.count_nonzero(target[subset] + 1e-12 < minimum[subset])),
            "counts": {
                keys[i][1]: {"reference": float(total[i]), "target": float(target[i]), "selected": float(counts[i]),
                             "reference_weighted_mean": float(mean[i]),
                             "mandatory_group_minimum": float(minimum[i])}
                for i in subset
            },
        }
    digest = hashlib.sha256()
    feature_digest = hashlib.sha256()
    for index in sorted(range(population), key=lambda i: reference[i]["chain_id"]):
        row = reference[index]
        digest.update(f"{row['chain_id']}\t{row['group_id']}\t{row['sequence']}\n".encode())
        feature_digest.update(json.dumps([row["chain_id"], features[index]], sort_keys=True).encode())
        feature_digest.update(b"\n")
    report = {
        "policy": POLICY if crop_size is None else "crop_topology_exposure_v1",
        "crop_size": crop_size,
        "crop_sampling_policy": None if crop_size is None else "uniform_contiguous_integer_start_v1",
        "group_policy": group_policy,
        "group_semantics": "full_leakage_groups" if group_policy == "all_groups" else "training_diversity_groups_not_leakage_certificate",
        "seed": seed,
        "reference": "distinct exact sequences in full source-qualified training-owned population",
        "reference_weighting": reference_weighting,
        "reference_probabilities": {reference[i]["chain_id"]: float(probabilities[i]) for i in sorted(range(population), key=lambda i: reference[i]["chain_id"])},
        "reference_probability_effective_size": float(1 / np.square(probabilities).sum()),
        "reference_identity_sha256": digest.hexdigest(),
        "reference_moments_sha256": feature_digest.hexdigest(),
        "input_chains": len(records),
        "reference_sequences": population,
        "selected_sequences": size,
        "mandatory_groups": len(groups) if group_policy == "all_groups" else 0,
        "reference_groups": len(groups),
        "covered_groups": int(np.count_nonzero(group_counts)),
        "group_coverage_fraction": float(np.count_nonzero(group_counts) / len(groups)),
        "uncovered_group_reference_probability": float(probabilities[group_counts[group] == 0].sum()),
        "maximum_selected_group_fraction": float(group_counts.max() / size),
        "objective_definition": "sum_f weight_f/max(sum(target_f),1) * sum_j (selected_j-target_j)^2/max(target_j,1)",
        "greedy_objective": greedy_objective,
        "final_objective": float(np.dot(weight, (counts - target) ** 2)),
        "swap_attempts": attempts,
        "accepted_swaps": accepted,
        "swap_passes_completed": passes_completed,
        "swap_attempt_limit": max_swap_attempts,
        "swap_pass_limit": swap_passes,
        "optimization_guarantee": "feasible heuristic; no global optimum or full local convergence claimed",
        "tradeoff": ("Every full group must be represented; this can prevent matching sequence-weighted proportions"
                     if group_policy == "all_groups" else
                     "Group distribution is a soft source-weighted objective; distinct group or structural category coverage is not guaranteed"),
        "geometry": ("Measured chain CA compactness/contact-order bins conditioned on topology only for one mapped domain covering >=90% of chain and >=90% valid CA; other cases unavailable"
                     if crop_size is None else "Expected observed CA fraction and contacts per crop residue from measured coordinates; no whole-chain geometry substituted for crop geometry"),
        "geometry_reference_eligible": population - int(family_report["topology_geometry" if crop_size is None else "crop_observation"]["counts"].get("unavailable", {}).get("reference", 0)),
        "exposure_scope": ("uniform chain selection with domain-instance multiplicity; not crop- or residue-weighted exposure"
                           if crop_size is None else
                           "exact expected structural contributions under declared reference parent probabilities and uniform contiguous crop starts; selected parents remain uniformly sampled without training weights"),
        "families": family_report,
    }
    return sorted(reference[int(index)]["chain_id"] for index in chosen), report
