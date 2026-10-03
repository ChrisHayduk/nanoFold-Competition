"""Compare fixed structural subsets with qualified random controls.

This maintainer analysis consumes hash-pinned private inputs. It never generates
labels, changes manifests, downloads data, or infers embeddings. Every output is
written to a new directory. See --help for the input contract.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
for directory in (REPO_ROOT, REPO_ROOT / "scripts"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from build_full_source_metadata import validate_full_source_metadata
from build_structural_population_plan import copy_rank, training_copies

from nanofold.crop_exposure import chain_observed_ca_crop_exposure
from nanofold.revised_parent_leakage import POLICY as REPAIRED_PUBLIC_POLICY
from nanofold.revised_parent_leakage import validate_parent_frame
from nanofold.structural_population import sequence_plan, validate_qualification_predicate, validate_receipt
from nanofold.structural_sampling import crop_structural_moments

METHODS = ("uniform_sequence", "cluster40_uniform_sequence", "group_cover_random_sequence")
UNKNOWN = "__unknown__"
SET_FIELDS = ("cath_architecture_ids", "cath_topology_ids", "cath_homologous_superfamily_ids")
DOMAIN_FIELDS = {"topology": "topology_id", "superfamily": "homologous_superfamily_id"}


def deterministic_crop(sequence: str, seed: int = 0, crop_size: int = 256) -> dict[str, Any]:
    """Outcome-independent exact-parent keyed crop, with unbiased hash rejection."""
    if not sequence or seed < 0 or crop_size < 1:
        raise ValueError("Crop requires a nonempty sequence, nonnegative seed and positive size")
    parent_sha256 = hashlib.sha256(sequence.encode("ascii")).hexdigest()
    starts = max(1, len(sequence) - crop_size + 1)
    limit = 2 ** 256 - (2 ** 256 % starts)
    counter = 0
    while True:
        payload = f"nanofold-esmc-crop-v1\0{seed}\0{parent_sha256}\0{counter}".encode("ascii")
        value = int.from_bytes(hashlib.sha256(payload).digest(), "big")
        if value < limit:
            start = value % starts
            break
        counter += 1
    crop = sequence[start:start + crop_size]
    return {"parent_sequence_sha256": parent_sha256, "crop_seed": seed,
            "start": start, "stop": start + len(crop), "sequence": crop,
            "crop_sequence_sha256": hashlib.sha256(crop.encode("ascii")).hexdigest()}


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def pin(path: Path) -> dict[str, Any]:
    return {"path": str(path.resolve()), "sha256": digest(path), "bytes": path.stat().st_size}


def checked_input(record: dict[str, Any]) -> Path:
    path = Path(record["path"])
    actual = pin(path)
    if actual["sha256"] != record["sha256"] or actual["bytes"] != record["bytes"]:
        raise ValueError(f"Input differs from frozen identity: {path}")
    return path


def write_json(path: Path, value: Any) -> None:
    with path.open("x") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def read_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if path.suffix in {".tsv", ".csv"}:
        with path.open(newline="") as handle:
            return list(csv.DictReader(handle, delimiter="\t" if path.suffix == ".tsv" else ","))
    value = json.loads(path.read_text())
    return value if isinstance(value, list) else value["chains"]


def manifest(path: Path) -> list[str]:
    values = [line.strip() for line in path.read_text().splitlines() if line.strip() and not line.startswith("#")]
    if not values or len(values) != len(set(values)):
        raise ValueError(f"Manifest is empty or duplicated: {path}")
    return values


def as_list(value: Any) -> list[Any]:
    if isinstance(value, str):
        return json.loads(value) if value else []
    return list(value or [])


def canonical_row(row: dict[str, Any], metadata: dict[str, Any]) -> dict[str, Any]:
    # Domain annotation owns its labels; qualification/frame exports own
    # sequence, grouping and measured geometry. Reject conflicting identities.
    for key in ("chain_id", "sequence", "sequence_sha256", "length"):
        left, right = row.get(key), metadata.get(key)
        if key == "length" and left is not None and right is not None:
            left, right = int(left), int(right)
        if left is not None and right is not None and left != right:
            raise ValueError(f"Metadata identity conflicts with qualification: {key}")
    result = {**row, **metadata}
    for key in ("group_id", "cluster40_id", "source_geometry", "crop_geometry", "qualification"):
        if key in row:
            result[key] = row[key]
    for key in ("chain_id", "group_id", "cluster40_id"):
        if not result.get(key):
            raise ValueError(f"Missing required chain field: {key}")
    sequence = result.get("sequence")
    if sequence:
        sha1 = hashlib.sha1(sequence.encode("ascii")).hexdigest()
        sha256 = hashlib.sha256(sequence.encode("ascii")).hexdigest()
        if result.get("sequence_sha1", sha1) != sha1 or result.get("sequence_sha256", sha256) != sha256:
            raise ValueError("Sequence hashes differ from exact sequence")
        result["sequence_sha1"] = sha1
        result["sequence_sha256"] = sha256
        if int(result.get("length", len(sequence))) != len(sequence):
            raise ValueError("Sequence length does not match metadata")
        result["length"] = len(sequence)
    if not result.get("sequence_sha1"):
        raise ValueError("Missing exact sequence identity")
    result["length"] = int(result["length"])
    if result["length"] <= 0:
        raise ValueError("Invalid chain length")
    resolution = result.get("resolution")
    result["resolution"] = float(resolution) if resolution not in (None, "", "unknown") else None
    if result["resolution"] is not None and (not math.isfinite(result["resolution"]) or result["resolution"] <= 0):
        raise ValueError("Invalid resolution must be explicit missing")
    for key in (*SET_FIELDS, "cath_domains"):
        result[key] = as_list(result.get(key))
    geometry = result.get("source_geometry") or {}
    if ("crop_geometry" not in result and result["length"] <= 256
            and geometry.get("mapping_status") == "verified"
            and geometry.get("target_length") == result["length"]
            and geometry.get("minimum_sequence_separation") == 3
            and geometry.get("contact_cutoff_angstrom") == 8.0):
        # For a short parent the only crop is the entire parent, so these
        # measured counts are already exact crop expectations.
        result["crop_geometry"] = {
            "sampling": "uniform_contiguous_integer_start_v1", "mapping_status": "verified",
            "crop_size": 256, "effective_crop_size": result["length"], "parent_length": result["length"],
            "expected_valid_ca_count": geometry["valid_ca_count"], "expected_contacts": geometry["contact_count"],
            "derivation": "Only possible crop is complete measured parent",
        }
    return result


def draw_controls(rows: list[dict[str, Any]], size: int, seeds: list[int], namespace: int,
                  *, group_policy: str = "all_groups") -> tuple[list[dict[str, Any]], dict[str, str]]:
    """Three distinct controls on a one-qualified-chain-per-sequence frame."""
    if size <= 0 or len(rows) < size or len({r["sequence_sha1"] for r in rows}) != len(rows):
        raise ValueError("Qualified frame must contain sufficient distinct exact sequences")
    if [r["chain_id"] for r in rows] != sorted(r["chain_id"] for r in rows):
        raise ValueError("Sampling frame must be sorted by chain ID")
    if len(set(seeds)) != len(seeds) or any(seed < 0 for seed in seeds):
        raise ValueError("Seeds must be distinct nonnegative integers")
    if group_policy not in {"all_groups", "soft_diversity"}:
        raise ValueError("Unknown group policy")
    clusters: dict[str, list[int]] = defaultdict(list)
    groups: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        clusters[row["cluster40_id"]].append(index)
        groups[row["group_id"]].append(index)
    arms, unavailable = [], {}
    for method_index, method in enumerate(METHODS, start=1):
        if method == "cluster40_uniform_sequence" and len(clusters) < size:
            unavailable[method] = f"Only {len(clusters)} qualified clusters for {size} required targets"
            continue
        if method == "group_cover_random_sequence" and len(groups) > size:
            if group_policy == "all_groups":
                unavailable[method] = f"Cannot cover {len(groups)} groups with {size} targets"
                continue
            method = "group_uniform_distinct"
        for seed in seeds:
            rng = np.random.default_rng(np.random.SeedSequence([namespace, method_index, seed]))
            if method == "uniform_sequence":
                indices = rng.choice(len(rows), size=size, replace=False)
            elif method == "cluster40_uniform_sequence":
                keys = sorted(clusters)
                chosen = rng.choice(len(keys), size=size, replace=False)
                indices = np.array([rng.choice(clusters[keys[i]]) for i in chosen], dtype=np.int64)
            elif method == "group_uniform_distinct":
                keys = sorted(groups)
                chosen = rng.choice(len(keys), size=size, replace=False)
                indices = np.array([rng.choice(groups[keys[i]]) for i in chosen], dtype=np.int64)
            else:
                anchors = np.array([rng.choice(groups[key]) for key in sorted(groups)], dtype=np.int64)
                remaining = np.setdiff1d(np.arange(len(rows)), anchors, assume_unique=True)
                indices = np.concatenate([anchors, rng.choice(remaining, size=size - len(anchors), replace=False)])
            arms.append({"name": f"{method}_seed{seed}", "method": method, "seed": seed,
                         "indices": np.sort(indices).tolist()})
    return arms, unavailable


def bin_value(value: float | int | None, edges: list[float]) -> str:
    if value is None:
        return UNKNOWN
    if not math.isfinite(value) or value <= 0:
        raise ValueError("Histogram observations must be finite and positive")
    for low, high in zip(edges[:-1], edges[1:], strict=True):
        if low < value <= high:
            return f"({low},{high}]"
    raise ValueError(f"Observation outside predefined bins: {value}")


def tokens(row: dict[str, Any], protocol: dict[str, Any]) -> dict[str, list[str]]:
    values = {key: ["|".join(sorted(set(row[key]))) or UNKNOWN] for key in SET_FIELDS}
    values["length"] = [bin_value(row["length"], protocol["chain_length_bins"])]
    values["resolution"] = [bin_value(row["resolution"], protocol["resolution_bins"])]
    domains = row["cath_domains"]
    values["domain_count"] = [str(len(domains))]
    values["annotation_status"] = ["annotated" if domains else "unannotated"]
    for name, field in DOMAIN_FIELDS.items():
        values[f"domain_{name}"] = [domain.get(field) or UNKNOWN for domain in domains] or [UNKNOWN]
    values["domain_boundary_status"] = [domain.get("boundary_status") or UNKNOWN for domain in domains] or [UNKNOWN]
    values["topology_x_domain_length"] = [
        f"{domain.get('topology_id') or UNKNOWN}|{bin_value(domain.get('domain_length'), protocol['domain_length_bins'])}"
        for domain in domains
    ] or [UNKNOWN]
    values["topology_x_domain_count"] = [
        f"{domain.get('topology_id') or UNKNOWN}|{len(domains)}" for domain in domains
    ] or [UNKNOWN]
    return values


def weights(rows: list[dict[str, Any]], mode: str) -> np.ndarray:
    if not rows:
        raise ValueError("Cannot weight empty population")
    if mode == "sequence_uniform":
        return np.full(len(rows), 1 / len(rows), dtype=np.float64)
    if mode in {"group_uniform", "cluster40_uniform"}:
        field = "group_id" if mode == "group_uniform" else "cluster40_id"
        counts = Counter(row[field] for row in rows)
        return np.array([1 / (len(counts) * counts[row[field]]) for row in rows], dtype=np.float64)
    raise ValueError(f"Unknown weighting: {mode}")


def crop_means(rows: list[dict[str, Any]], mode: str) -> dict[str, dict[str, float]]:
    """Population means of raw crop moments; preserve expected domain counts."""
    means: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for row, weight in zip(rows, weights(rows, mode), strict=True):
        for family, moments in row["_crop_moments"].items():
            for category, value in moments.items():
                if not math.isfinite(value) or value < 0:
                    raise ValueError("Crop moments must be finite and nonnegative")
                means[family][category] += float(weight * value)
    return {family: dict(values) for family, values in means.items()}


def crop_discrepancy(observed: dict[str, dict[str, float]], expected: dict[str, dict[str, float]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for family in sorted(observed.keys() | expected.keys()):
        left, right = observed.get(family, {}), expected.get(family, {})
        observed_total, reference_total = sum(left.values()), sum(right.values())
        result[f"{family}.expected_total"] = observed_total
        result[f"{family}.reference_expected_total"] = reference_total
        result[f"{family}.raw_l1"] = sum(abs(left.get(key, 0) - right.get(key, 0)) for key in left.keys() | right.keys())
        result[f"{family}.total_difference"] = observed_total - reference_total
        # Missing measurement tokens cannot constitute geometric agreement.
        geometry_family = family.startswith(("crop_", "observed_ca_"))
        measurable = not (geometry_family and (left.get("unavailable", 0) or right.get("unavailable", 0)))
        if geometry_family:
            result[f"{family}.unavailable_fraction"] = left.get("unavailable", 0)
            result[f"{family}.reference_unavailable_fraction"] = right.get("unavailable", 0)
        if observed_total > 0 and reference_total > 0 and measurable:
            js, tvd = divergence({key: value / observed_total for key, value in left.items()},
                                 {key: value / reference_total for key, value in right.items()})
            result[f"{family}.shape_js_bits"], result[f"{family}.shape_tvd"] = js, tvd
        else:
            result[f"{family}.shape_js_bits"] = result[f"{family}.shape_tvd"] = None
            if not measurable:
                result[f"{family}.raw_l1"] = result[f"{family}.total_difference"] = None
                result[f"{family}.expected_total"] = result[f"{family}.reference_expected_total"] = None
    return result


def analyze_crop_moments(reference: list[dict[str, Any]], arms: list[dict[str, Any]],
                         arm_rows: list[list[dict[str, Any]]], out: Path, crop_size: int = 256,
                         *, baseline_comparable: bool = True) -> None:
    for row in reference + arm_rows[0] + arm_rows[1]:
        row["_crop_moments"] = crop_structural_moments(row, crop_size=crop_size)
        observed_ca = chain_observed_ca_crop_exposure(row, crop_size=crop_size)
        row["_observed_ca_available"] = observed_ca["status"] == "available"
        if row["_observed_ca_available"]:
            for family, values in observed_ca["families"].items():
                row["_crop_moments"][f"observed_ca_{family}"] = values
            row["_crop_moments"]["observed_ca_coverage"] = {
                "mapped_observed": observed_ca["expected_mapped_observed_ca_fraction"],
                "unknown_observed": observed_ca["expected_unknown_observed_ca_fraction"],
                "unobserved_ca": observed_ca["expected_unobserved_ca_fraction"],
            }
        else:
            for family in ("topology", "superfamily", "architecture", "coverage"):
                row["_crop_moments"][f"observed_ca_{family}"] = {"unavailable": 1.0}
    references = {mode: crop_means(reference, mode) for mode in ("sequence_uniform", "cluster40_uniform")}
    write_json(out / "crop-reference-raw-means.json", references)
    records = []
    candidates = {}
    for arm, rows in zip(arms, arm_rows, strict=True):
        # The selected corpus still trains uniformly. Changing the reference
        # sensitivity never silently changes the subset's sampling probabilities.
        observed = crop_means(rows, "sequence_uniform")
        values = {f"uniform_subset_vs_{mode}.{key}": value for mode, expected in references.items()
                  for key, value in crop_discrepancy(observed, expected).items()}
        values["annotation.unannotated_parent_fraction"] = sum(not row["cath_domains"] for row in rows) / len(rows)
        values["annotation.parent_any_unmapped_domain_fraction"] = sum(
            any(not domain.get("label_segments") for domain in row["cath_domains"]) for row in rows) / len(rows)
        values["annotation.expected_unmapped_domain_count_per_parent"] = sum(
            not domain.get("label_segments") for row in rows for domain in row["cath_domains"]) / len(rows)
        values["annotation.observed_ca_measured_parent_fraction"] = sum(row["_observed_ca_available"] for row in rows) / len(rows)
        record = {key: arm[key] for key in ("name", "method", "seed")} | {"metrics": values}
        records.append(record)
        if arm["name"] in {"candidate06", "candidate07"}:
            candidates[arm["name"]] = observed
    write_json(out / "crop-candidate-raw-means.json", candidates)
    write_json(out / "crop-metrics.json", records)
    write_json(out / "crop-summary.json", summarize(records, baseline_comparable=baseline_comparable))


def source_stage_statistics(rows: list[dict[str, Any]], full_rows: list[dict[str, Any]], crop_size: int = 256) -> dict[str, Any]:
    """Metadata support, with deposited-copy and distinct-parent denominators."""
    if not rows or not full_rows:
        raise ValueError("Source stages must be nonempty")
    if any(not row.get("sequence_sha256") or int(row["length"]) <= 0 for row in full_rows):
        raise ValueError("Full-source metadata lacks exact-parent identities or valid lengths")
    full_parents = {row["sequence_sha256"]: int(row["length"]) for row in full_rows}
    parents = {row["sequence_sha256"]: int(row["length"]) for row in rows}
    counts = Counter(row["sequence_sha256"] for row in rows)
    represented = [row for row in full_rows if row["sequence_sha256"] in parents]
    result: dict[str, Any] = {
        "deposited_chains": len(rows), "distinct_parent_sequences": len(parents),
        "chain_copy_residues": sum(int(row["length"]) for row in rows),
        "distinct_parent_residues": sum(parents.values()),
        "distinct_parent_crop_residue_mass": sum(min(length, crop_size) for length in parents.values()),
        "fraction_distinct_parents_longer_than_crop": sum(length > crop_size for length in parents.values()) / len(parents),
        "full_source_unique_parent_mass_represented": len(parents.keys() & full_parents.keys()) / len(full_parents),
        "full_source_chain_mass_represented_by_parent_identity": len(represented) / len(full_rows),
        "full_source_residue_mass_represented_by_parent_identity": sum(int(row["length"]) for row in represented) / sum(int(row["length"]) for row in full_rows),
        "annotation_chain_status_counts": dict(Counter(row.get("cath_boundary_status", UNKNOWN) for row in rows)),
        "annotation_domain_boundary_status_counts": dict(Counter(
            domain.get("boundary_status", UNKNOWN) for row in rows for domain in row.get("cath_domains", []))),
    }
    for mode in ("deposited_chain_uniform", "unique_parent_then_uniform_copy"):
        totals: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
        for row in rows:
            weight = 1 / len(rows) if mode == "deposited_chain_uniform" else 1 / (len(parents) * counts[row["sequence_sha256"]])
            domains = row.get("cath_domains") or []
            for name, field in DOMAIN_FIELDS.items():
                for category in ([domain.get(field) or UNKNOWN for domain in domains] or [UNKNOWN]):
                    totals[name][category] += weight
        result[mode] = {}
        for name, values in totals.items():
            total = sum(values.values())
            result[mode][name] = {"raw_expected_domain_instances_including_unknown_placeholder": total,
                                  "known_categories": len(set(values) - {UNKNOWN}),
                                  "distribution": {key: value / total for key, value in values.items()}}
    return result


def analyze_source_stages(metadata_rows: list[dict[str, Any]], training_owned_ids: set[str],
                          reference: list[dict[str, Any]], selected: list[dict[str, Any]], out: Path) -> None:
    full = {row["chain_id"]: row for row in metadata_rows}
    eligible = {row["chain_id"] for row in metadata_rows if row.get("accepted") is True}
    qualified = {row["chain_id"] for row in reference}
    chosen = {row["chain_id"] for row in selected}
    if not chosen <= qualified <= training_owned_ids <= eligible <= set(full):
        raise ValueError("Full→eligible→barrier-owned→qualified→selected chain stages are not nested")
    stages = {"full_pinned_experimental_chain_cache": metadata_rows,
              "quality_eligible": [full[chain] for chain in sorted(eligible)],
              "barrier_training_owned": [full[chain] for chain in sorted(training_owned_ids)],
              "source_qualified_unique_parent": reference, "selected_unique_parent": selected}
    results = {name: source_stage_statistics(rows, metadata_rows) for name, rows in stages.items()}
    full_stats = results["full_pinned_experimental_chain_cache"]
    for stage in results.values():
        for mode in ("deposited_chain_uniform", "unique_parent_then_uniform_copy"):
            for name in DOMAIN_FIELDS:
                expected = full_stats[mode][name]["distribution"]
                observed = stage[mode][name]["distribution"]
                stage[mode][name]["missing_known_full_source_domain_mass"] = sum(
                    value for key, value in expected.items() if key != UNKNOWN and key not in observed)
                stage[mode][name]["js_bits_vs_full_source"], stage[mode][name]["tvd_vs_full_source"] = divergence(observed, expected)
    write_json(out / "source-support-stages.json", {
        "scope": "Pinned experimental PDB chain-cache component only; not all roughly16M OpenProteinSet MSAs or an empirical natural protein distribution",
        "weighting": "Unique-parent sensitivity first draws a parent uniformly, then a deposited copy uniformly; this source description is not the qualified training copy policy",
        "boundary_scope": "Full-source category support uses annotation labels; ineligible or unresolved domain boundaries do not imply known crop exposure",
        "stages": results})


def validate_qualification_frame(qualification: dict[str, Any], frame: dict[str, Any]) -> None:
    """Qualification must cover every admitted parent and ranked deposited copy."""
    plan = json.loads(Path(qualification["plan"]["path"]).read_text())
    if plan.get("full_frame_receipt") != frame["receipt"]:
        raise ValueError("Qualification plan does not bind the current complete barrier frame")
    expected = sequence_plan(training_copies(frame), rank=copy_rank)
    observed = read_rows(Path(qualification["sequence_plan"]["path"]))
    if observed != expected:
        raise ValueError("Qualification plan does not cover the exact current admitted parent and ranked-copy population")


def validate_expanded_frame(paths: dict[str, Path], reference: list[dict[str, Any]],
                            qualification: dict[str, Any], *, repaired_public: bool = False) -> set[str]:
    """Replay directed admission and bind qualified chains to current groups."""
    frame = validate_parent_frame(paths["barrier_receipt"])
    if (frame.get("policy") == REPAIRED_PUBLIC_POLICY) != repaired_public:
        raise ValueError("Current admission frame does not match the frozen public-cohort repair policy")
    validate_qualification_frame(qualification, frame)
    if frame["source_bindings"]["chain_data_cache"]["sha256"] != digest(paths["chain_data_cache"]):
        raise ValueError("Full-source metadata and admission barrier use different underlying chain caches")
    admitted = set(frame["eligible_training_ids"])
    if admitted != set(manifest(paths["training_owned_manifest"])):
        raise ValueError("Training-owned manifest differs from validated barrier export")
    frame_rows = {row["chain_id"]: row for row in frame["records"]}
    for row in reference:
        current = frame_rows.get(row["chain_id"])
        if row["chain_id"] not in admitted or current is None or current["training_exclusion_reasons"]:
            raise ValueError("Qualified reference escapes the validated admission barrier")
        if any(row[key] != current[key] for key in ("sequence", "group_id", "cluster40_id")):
            raise ValueError("Qualified identity/group differs from validated training frame")
    return admitted


def distributions(rows: list[dict[str, Any]], protocol: dict[str, Any], mode: str) -> dict[str, dict[str, float]]:
    totals: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for row, weight in zip(rows, weights(rows, mode), strict=True):
        for field, categories in row["_tokens"].items():
            for category in categories:
                totals[field][category] += float(weight)
        geometry = row.get("source_geometry") or {}
        profile = geometry.get("ca_distance_histogram")
        if geometry.get("mapping_status") == "verified" and profile is not None:
            values = np.asarray(profile, dtype=np.float64)
            if values.shape != (8,) or not np.isfinite(values).all() or np.any(values < 0) or not np.isclose(values.sum(), 1):
                raise ValueError("Measured CA distance histogram must be an eight-bin probability vector")
            for index, value in enumerate(values):
                totals["measured_chain_ca_distance_profile"][str(index)] += float(weight * value)
        else:
            totals["measured_chain_ca_distance_profile"][UNKNOWN] += float(weight)
    result = {}
    for field, counts in totals.items():
        total = sum(counts.values())
        result[field] = {key: value / total for key, value in counts.items()}
    return result


def divergence(left: dict[str, float], right: dict[str, float]) -> tuple[float, float]:
    keys = sorted(left.keys() | right.keys())
    p = np.array([left.get(key, 0) for key in keys], dtype=np.float64)
    q = np.array([right.get(key, 0) for key in keys], dtype=np.float64)
    if not np.isclose(p.sum(), 1) or not np.isclose(q.sum(), 1) or np.any(p < 0) or np.any(q < 0):
        raise ValueError("Probability distributions must be normalized and nonnegative")
    middle = (p + q) / 2
    pm, qm = p > 0, q > 0
    js = (np.sum(p[pm] * np.log2(p[pm] / middle[pm])) + np.sum(q[qm] * np.log2(q[qm] / middle[qm]))) / 2
    return float(js), float(np.sum(np.abs(p - q)) / 2)


def independent_support(rows: list[dict[str, Any]], field: str) -> dict[str, int]:
    groups: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        for domain in row["cath_domains"]:
            if domain.get(field):
                groups[domain[field]].add(row["group_id"])
    return {key: len(value) for key, value in groups.items()}


def describe_arm(rows: list[dict[str, Any]], reference: list[dict[str, Any]], protocol: dict[str, Any],
                 reference_distributions: dict[str, Any], reference_support: dict[str, dict[str, int]]) -> dict[str, Any]:
    groups = Counter(row["group_id"] for row in rows)
    fractions = np.array(list(groups.values()), dtype=np.float64) / len(rows)
    result: dict[str, Any] = {
        "chains": len(rows), "unique_sequences": len({row["sequence_sha1"] for row in rows}),
        "groups": len(groups), "cluster40": len({row["cluster40_id"] for row in rows}),
        "pdb_entries": len({row["chain_id"].split("_")[0] for row in rows}),
        "group_largest_fraction": float(fractions.max()), "group_hhi": float(np.sum(fractions ** 2)),
        "group_effective_count": float(1 / np.sum(fractions ** 2)),
        "unannotated_chain_fraction": sum(not row["cath_domains"] for row in rows) / len(rows),
        "reference_chain_membership_fraction": len({r["chain_id"] for r in rows} & {r["chain_id"] for r in reference}) / len(rows),
        "reference_sequence_membership_fraction": len({r["sequence_sha1"] for r in rows} & {r["sequence_sha1"] for r in reference}) / len(rows),
    }
    lengths = [domain.get("domain_length") for row in rows for domain in row["cath_domains"]]
    result["domain_instances"] = len(lengths)
    result["measured_geometry_available_fraction"] = sum(
        (row.get("source_geometry") or {}).get("mapping_status") == "verified" for row in rows) / len(rows)
    result["domain_length_available_fraction"] = sum(value is not None for value in lengths) / len(lengths) if lengths else None
    fractions_known = [row["cath_unannotated_fraction"] for row in rows if row.get("cath_unannotated_fraction") is not None]
    result["unannotated_residue_fraction_available_chains"] = len(fractions_known)
    result["mean_unannotated_residue_fraction_available_only"] = float(np.mean(fractions_known)) if fractions_known else None
    for mode in protocol["weighting"]:
        observed = distributions(rows, protocol, mode)
        for field, expected in reference_distributions[mode].items():
            if field == "measured_chain_ca_distance_profile" and (expected.get(UNKNOWN, 0) > 0 or observed[field].get(UNKNOWN, 0) > 0):
                # Different measurement availability is not evidence of improved
                # geometry. Preserve missing mass in distributions and withhold
                # the population comparison until both sides are measured.
                result[f"{mode}.{field}.js_bits"] = None
                result[f"{mode}.{field}.tvd"] = None
                continue
            js, tvd = divergence(observed[field], expected)
            result[f"{mode}.{field}.js_bits"] = js
            result[f"{mode}.{field}.tvd"] = tvd
        for name, field in DOMAIN_FIELDS.items():
            support = independent_support(rows, field)
            feasible = reference_support[name]
            expected = reference_distributions[mode][f"domain_{name}"]
            known_mass = sum(value for key, value in expected.items() if key != UNKNOWN)
            result[f"{mode}.{name}.distinct_known"] = len(support)
            result[f"{mode}.{name}.reference_known_categories"] = len(feasible)
            result[f"{mode}.{name}.known_reference_mass"] = known_mass
            for k in (1, 2, 5):
                eligible = {key for key, count in feasible.items() if count >= k}
                covered = {key for key in eligible if support.get(key, 0) >= k}
                result[f"{mode}.{name}.attainable_categories_at_{k}_groups"] = len(eligible)
                result[f"{mode}.{name}.coverage_at_{k}_groups"] = len(covered) / len(eligible) if eligible else None
                result[f"{mode}.{name}.missing_known_mass_at_{k}_groups"] = (
                    sum(expected.get(key, 0) for key in eligible - covered) / known_mass if known_mass else None)
    return result


def summarize(records: list[dict[str, Any]], metric_key: str = "metrics", *, baseline_comparable: bool = True) -> dict[str, Any]:
    methods: dict[str, list[dict[str, Any]]] = defaultdict(list)
    candidates = {}
    for row in records:
        if row["method"] in {"candidate06", "candidate07"}:
            candidates[row["method"]] = row[metric_key]
        else:
            methods[row["method"]].append(row[metric_key])
    summary: dict[str, Any] = {"candidates": candidates, "controls": {}, "candidate06_primary_comparable": baseline_comparable}
    for method, values in methods.items():
        summary["controls"][method] = {}
        for metric in values[0]:
            present = [v[metric] for v in values if v.get(metric) is not None]
            if not present:
                summary["controls"][method][metric] = {"n": 0}
                continue
            array = np.array(present, dtype=np.float64)
            summary["controls"][method][metric] = {
                "n": len(array), "mean": float(array.mean()), "sd": float(array.std(ddof=1)) if len(array) > 1 else 0.,
                "median": float(np.median(array)), "p05": float(np.quantile(array, .05)), "p95": float(np.quantile(array, .95)),
            }
    if set(candidates) == {"candidate06", "candidate07"} and baseline_comparable:
        summary["candidate07_minus_candidate06"] = {
            key: (candidates["candidate07"][key] - candidates["candidate06"][key]
                  if candidates["candidate07"].get(key) is not None and candidates["candidate06"].get(key) is not None else None)
            for key in candidates["candidate06"].keys() | candidates["candidate07"].keys()
        }
    return summary


def matched_control_evidence(summary: dict[str, Any], prefix: str) -> dict[str, Any]:
    """Expose every primary discrepancy and adverse direction, without a win score."""
    selected = {key: value for key, value in summary["candidates"]["candidate07"].items()
                if key.startswith(prefix) and key.endswith(("raw_l1", "shape_js_bits", "shape_tvd"))}
    comparisons: dict[str, Any] = {}
    for method, metrics in summary["controls"].items():
        comparisons[method] = {}
        for key, value in selected.items():
            control = metrics[key]
            comparisons[method][key] = {"candidate07": value, "control": control,
                                        "candidate07_minus_control_mean": value - control["mean"]
                                        if value is not None and control["n"] else None}
    return {"direction": "Lower discrepancy is closer to the fixed reference; no automatic universal superiority threshold",
            "seed_spread": "Random subset variability, not a biological confidence interval",
            "calibration_scope": "Annotated-sequence crop moments optimized by selection are calibration objectives. Observed-CA exposure is a separate diagnostic, not a selector objective or all-atom completeness measure. Neither establishes folding improvement.",
            "candidate07_vs_controls": comparisons}


def metric_vector(nearest: np.ndarray, radii: np.ndarray, global_radii: np.ndarray, weight: np.ndarray) -> dict[str, float]:
    if nearest.shape != weight.shape or radii.shape != (len(nearest), 2) or not np.isfinite(nearest).all():
        raise ValueError("Nearest-distance denominator mismatch or nonfinite result")
    weight = weight / weight.sum()
    return {"mean_distance": float(np.dot(nearest, weight)),
            "p50_sequence_unweighted": float(np.quantile(nearest, .5)),
            "p90_sequence_unweighted": float(np.quantile(nearest, .9)),
            "p95_sequence_unweighted": float(np.quantile(nearest, .95)),
            "local10_coverage": float(np.dot(nearest <= radii[:, 0], weight)),
            "local50_coverage": float(np.dot(nearest <= radii[:, 1], weight)),
            "global10_coverage": float(np.dot(nearest <= global_radii[0], weight)),
            "global50_coverage": float(np.dot(nearest <= global_radii[1], weight))}


def validate_historical_baseline(paths: dict[str, Path], protocol: dict[str, Any]) -> dict[str, Any]:
    historical = protocol["historical_controls"]
    historical_protocol = paths.get("historical_protocol", Path(historical["protocol_path"]))
    if digest(historical_protocol) != historical["protocol_sha256"]:
        raise ValueError("Frozen historical protocol changed")
    accepted = json.loads(historical_protocol.read_text())["inputs"]
    if digest(paths["candidate06"]) != accepted["nanofold_train"]["sha256"]:
        raise ValueError("Candidate06 differs from the frozen historical manifest")
    return accepted


def validate_embedding_inputs(paths: dict[str, Path], protocol: dict[str, Any]) -> None:
    """Bind reused measurements to the accepted historical study, not just new pins."""
    historical = protocol["historical_controls"]
    accepted = validate_historical_baseline(paths, protocol)
    for key, historical_key in {"embedding_reference": "reference", "embedding_cache": "embedding_cache",
                                "embedding_metadata": "embedding_metadata", "exact_helper": "exact_module",
                                "candidate06": "nanofold_train"}.items():
        if digest(paths[key]) != accepted[historical_key]["sha256"]:
            raise ValueError(f"Embedding input differs from original study: {key}")
    completion = json.loads(paths["historical_completion"].read_text())
    if completion.get("status") != "complete" or completion["inputs"]["protocol"]["sha256"] != historical["protocol_sha256"]:
        raise ValueError("Historical comparison is incomplete or uses another protocol")
    records = list(completion["inputs"].values()) + completion["artifacts"]
    accepted_hashes = {record["sha256"] for record in records}
    for key in ("historical_embedding_distances", "embedding_cohorts", "embedding_radii"):
        if digest(paths[key]) not in accepted_hashes:
            raise ValueError(f"Reused artifact not bound by original comparison: {key}")
    prepared_binding = completion["inputs"]["prepared"]
    if digest(Path(prepared_binding["path"])) != prepared_binding["sha256"]:
        raise ValueError("Historical membership preparation changed")
    prepared = json.loads(Path(prepared_binding["path"]).read_text())
    checked_input(prepared["files"]["memberships_npz"])


def validate_historical_arrays(names: list[str], nearest: np.ndarray, leave_self: np.ndarray, queries: int) -> None:
    methods = ("uniform_chain", *METHODS)
    expected = ["nanofold", *[f"{method}_seed{seed}" for method in methods for seed in range(100)]]
    if names != expected or nearest.shape != (queries, 401) or leave_self.shape != nearest.shape:
        raise ValueError("Historical controls must retain all four methods and 100 unique seeds in original order")
    if not np.isfinite(nearest).all() or not np.isfinite(leave_self).all():
        raise ValueError("Historical nearest distances must be finite")


def baseline_requalification_rows(paths: dict[str, Path], size: int) -> list[dict[str, Any]]:
    receipt = json.loads(paths["candidate06_requalification"].read_text())
    if receipt.get("policy") == "fixed_baseline_verified_source_replay_v1":
        validate_fixed_baseline_replay(receipt, paths, size)
    elif (receipt.get("status") != "complete" or receipt["counts"]["train"] != size
            or receipt["outputs"]["train-qualified"]["sha256"] != digest(paths["candidate06_rows"])):
        raise ValueError("Baseline requalification does not bind the complete candidate06 export")
    if receipt.get("policy") != "fixed_baseline_verified_source_replay_v1":
        components = {}
        for key in ("structure_query_geometry_replay", "full_a3m_audit"):
            binding = receipt[key]
            if digest(Path(binding["path"])) != binding["sha256"]:
                raise ValueError("Baseline component receipt changed")
            components[key] = json.loads(Path(binding["path"]).read_text())
            if components[key].get("status") != "complete":
                raise ValueError("Baseline component qualification is incomplete")
        structure = components["structure_query_geometry_replay"]
        if structure["inputs"]["train"]["sha256"] != digest(paths["candidate06"]):
            raise ValueError("Baseline requalification does not bind the frozen manifest")
    rows = read_rows(paths["candidate06_rows"])
    if (len(rows) != size or {row["chain_id"] for row in rows} != set(manifest(paths["candidate06"]))
            or any(row.get("status") != "passed" or not row.get("qualification", {}).get("raw_full_audit") for row in rows)):
        raise ValueError("Baseline requalification is incomplete or contains failed chains")
    return rows


def validate_fixed_baseline_replay(receipt: dict[str, Any], paths: dict[str, Path], size: int) -> None:
    """Bind the final unified replay to current predicates and original targets."""
    if (receipt.get("schema_version") != 1 or receipt.get("status") != "complete"
            or receipt["counts"] != {"train": {"passed": size}, "val": {"passed": 1000}, "hidden": {"passed": 1000}}
            or receipt["outputs"]["train-qualified"]["sha256"] != digest(paths["candidate06_rows"])):
        raise ValueError("Final baseline replay is incomplete or uses another qualification predicate")
    validate_qualification_predicate(receipt["predicate"])
    for record in [receipt["binding"], *receipt["inputs"].values(), *receipt["outputs"].values()]:
        checked_input(record)
    binding = json.loads(Path(receipt["binding"]["path"]).read_text())
    if binding["inputs"] != receipt["inputs"] or binding["predicate"] != receipt["predicate"]:
        raise ValueError("Final baseline replay differs from its initial binding")
    inputs = receipt["inputs"]
    prior = json.loads(Path(inputs["input_receipt"]["path"]).read_text())
    original = json.loads(Path(inputs["original_replay_receipt"]["path"]).read_text())
    if (prior.get("status") != "complete" or original.get("status") != "complete"
            or prior["outputs"]["train-qualified"]["sha256"] != inputs["train"]["sha256"]
            or prior["outputs"]["heldout-chains"]["sha256"] != inputs["heldout"]["sha256"]
            or prior["structure_query_geometry_replay"]["sha256"] != inputs["original_replay_receipt"]["sha256"]
            or original["inputs"]["train"]["sha256"] != digest(paths["candidate06"])
            or inputs["train_manifest"]["sha256"] != digest(paths["candidate06"])):
        raise ValueError("Final baseline replay does not preserve the original frozen manifest")


def compute_nearest_arrays(helper: Any, vectors: np.ndarray, selected: np.ndarray,
                           progress_path: Path | None = None) -> tuple[np.ndarray, np.ndarray]:
    engine = helper.MatrixEngine("numpy", vectors, vectors)
    nearest = np.empty((len(vectors), len(selected)), dtype=np.float32)
    leave_self = np.empty_like(nearest)
    for start in range(0, len(vectors), 128):
        end = min(start + 128, len(vectors))
        distances = helper.checked_distances(engine.similarities(vectors[start:end], "reference"))
        distances[np.arange(end - start), np.arange(start, end)] = 0.
        nearest[start:end], _ = helper.arm_nearest(distances, selected)
        distances[np.arange(end - start), np.arange(start, end)] = np.inf
        leave_self[start:end], _ = helper.arm_nearest(distances, selected)
        if progress_path is not None:
            progress_path.write_text(json.dumps({"completed_queries": end, "total_queries": len(vectors)}))
    return nearest, leave_self


def validate_cohorts(cohorts: dict[str, np.ndarray], reference_size: int) -> None:
    if any(mask.shape != (reference_size,) or mask.dtype != bool or not mask.any() for mask in cohorts.values()):
        raise ValueError("Invalid fixed embedding cohort")
    if (reference_size != 56658 or int(cohorts["public_val"].sum()) != 1000
            or int(cohorts["eval_group_reference"].sum()) != 5161
            or not cohorts["all_reference"].all()
            or np.any(cohorts["public_val"] & ~cohorts["eval_group_reference"])):
        raise ValueError("Embedding cohorts differ from frozen 56658/1000/5161 denominators")


def embedding_analysis(paths: dict[str, Path], arms: list[dict[str, Any]], arm_rows: list[list[dict[str, Any]]], out: Path) -> None:
    """Reuse pinned, independently reviewed exact engine and fixed radii."""
    spec = importlib.util.spec_from_file_location("frozen_exact_sampling_engine", paths["exact_helper"])
    if spec is None or spec.loader is None:
        raise ValueError("Cannot load pinned exact metric engine")
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    reference = read_rows(paths["embedding_reference"])
    sha = np.array([row["sequence_sha1"] for row in reference])
    if len(set(sha.tolist())) != len(sha):
        raise ValueError("Embedding reference must contain unique sequences")
    by_sha = {value: i for i, value in enumerate(sha.tolist())}
    with np.load(paths["embedding_cache"], allow_pickle=False) as cache:
        vectors = helper.align_embeddings(read_rows(paths["embedding_metadata"]), cache["ids"], cache["embeddings"], reference, 2560)
    with np.load(paths["embedding_radii"], allow_pickle=False) as cached:
        if not np.array_equal(cached["query_sequence_sha1"], sha):
            raise ValueError("Cached radius ordering differs from reference")
        radii = cached["local_radii"].copy()
    if radii.shape != (len(sha), 2) or not np.isfinite(radii).all() or np.any(radii < 0):
        raise ValueError("Invalid frozen local radii")
    with np.load(paths["embedding_cohorts"], allow_pickle=False) as cohort_file:
        if not np.array_equal(cohort_file["reference_sequence_sha1"], sha):
            raise ValueError("Cohort ordering differs from reference")
        cohorts = {name: cohort_file[name].copy() for name in ("public_val", "eval_group_reference", "all_reference")}
    validate_cohorts(cohorts, len(sha))
    selected = np.zeros((len(arms), len(reference)), dtype=bool)
    for index, rows in enumerate(arm_rows):
        try:
            selected[index, [by_sha[row["sequence_sha1"]] for row in rows]] = True
        except KeyError as error:
            raise ValueError("Arm exact sequence missing from accepted embedding cache") from error
    if np.any(selected[:, cohorts["eval_group_reference"]]):
        raise ValueError("Training arm overlaps evaluation-owned sequence cohort")
    nearest, leave_self = compute_nearest_arrays(helper, vectors, selected, out / "embedding-progress.json")
    np.savez_compressed(out / "embedding-distances.npz", query_sequence_sha1=sha,
                        arm_names=np.array([arm["name"] for arm in arms]), nearest_distances=nearest,
                        leave_exact_self_out=leave_self, local_radii=radii)
    global_radii = np.median(radii, axis=0)
    ref_weights = {"sequence_uniform": np.ones(len(reference)),
                   "group_uniform": weights(reference, "group_uniform"),
                   "source_chain_abundance": np.array([float(row["chain_count"]) for row in reference])}
    cohort_weight_vectors = {}
    for cohort_name, mask in {**cohorts, "all_reference_leave_exact_self_out": cohorts["all_reference"]}.items():
        cohort_weight_vectors[cohort_name] = {**ref_weights, "group_uniform": np.zeros(len(reference))}
        cohort_weight_vectors[cohort_name]["group_uniform"][mask] = weights(
            [row for row, included in zip(reference, mask, strict=True) if included], "group_uniform")
    results = []
    for i, arm in enumerate(arms):
        values = {}
        for cohort_name, mask in {**cohorts, "all_reference_leave_exact_self_out": cohorts["all_reference"]}.items():
            cohort_weights = cohort_weight_vectors[cohort_name]
            matrix = leave_self if cohort_name.endswith("leave_exact_self_out") else nearest
            for mode, weight in cohort_weights.items():
                for key, value in metric_vector(matrix[mask, i], radii[mask], global_radii, weight[mask]).items():
                    values[f"{cohort_name}.{mode}.{key}"] = value
        results.append({key: arm[key] for key in ("name", "method", "seed")} | {"metrics": values})
    write_json(out / "embedding-metrics.json", results)
    write_json(out / "embedding-summary.json", summarize(results))
    # Historical controls preserve their original membership and unqualified source
    # frame. Their full-reference nearest distances are reused, never redrawn.
    if "historical_embedding_distances" in paths:
        with np.load(paths["historical_embedding_distances"], allow_pickle=False) as cached:
            if not np.array_equal(cached["query_sequence_sha1"], sha):
                raise ValueError("Historical nearest distances have different query ordering")
            historical = cached["nearest_distances"]
            historical_leave_self = cached["nearest_distances_leave_exact_self_out"]
            names = cached["arm_names"].astype(str).tolist()
        validate_historical_arrays(names, historical, historical_leave_self, len(reference))
        legacy = []
        for i, name in enumerate(names):
            if name == "nanofold":
                continue
            method, seed_text = name.rsplit("_seed", 1)
            values = {}
            for cohort_name, mask in {**cohorts, "all_reference_leave_exact_self_out": cohorts["all_reference"]}.items():
                cohort_weights = cohort_weight_vectors[cohort_name]
                matrix = historical_leave_self if cohort_name.endswith("leave_exact_self_out") else historical
                for mode, weight in cohort_weights.items():
                    for key, value in metric_vector(matrix[mask, i], radii[mask], global_radii, weight[mask]).items():
                        values[f"{cohort_name}.{mode}.{key}"] = value
            legacy.append({"name": name, "method": method, "seed": int(seed_text), "metrics": values})
        write_json(out / "historical-broad-embedding-summary.json", {
            "scope": "Historical metadata-qualified frame; not equivalent to new full source qualification",
            "summary": summarize(results[:2] + legacy),
            "leave_exact_self_out": "Reuse separately retained historical distances excluding every deposited copy of the exact query sequence",
        })


def run(config_path: Path, protocol_path: Path, expected_protocol_sha: str, out: Path, *, skip_embeddings: bool = False) -> None:
    if digest(protocol_path) != expected_protocol_sha:
        raise ValueError("Comparison protocol differs from declared freeze")
    protocol = json.loads(protocol_path.read_text())
    expanded = bool(protocol.get("expanded_parent_crop_amendment"))
    group_policy = "soft_diversity" if expanded else "all_groups"
    config = json.loads(config_path.read_text())
    paths = {key: checked_input(value) for key, value in config["inputs"].items()}
    initial_pins = {"protocol": pin(protocol_path), "config": pin(config_path),
                    "implementation": pin(Path(__file__)), "inputs": config["inputs"],
                    "implementation_dependencies": [pin(REPO_ROOT / name) for name in
                        ("nanofold/crop_exposure.py", "nanofold/structural_sampling.py",
                         "nanofold/full_parent_leakage.py", "nanofold/revised_parent_leakage.py",
                         "nanofold/structural_population.py", "scripts/build_revised_parent_frame.py",
                         "scripts/build_full_parent_frame.py", "scripts/build_structural_population_plan.py",
                         "scripts/build_manifests.py", "scripts/build_full_source_metadata.py",
                         "scripts/build_structure_metadata.py", "nanofold/cath_boundaries.py",
                         "nanofold/structure_metadata.py")]}
    receipt = validate_receipt(paths["qualification_receipt"])
    if (Path(receipt["qualified_chains"]["path"]).resolve() != paths["qualified_rows"].resolve()
            or receipt["qualified_chains"]["sha256"] != digest(paths["qualified_rows"])):
        raise ValueError("Qualified comparison rows differ from complete qualification receipt")
    validate_historical_baseline(paths, protocol)
    if expanded and not skip_embeddings:
        raise ValueError("Expanded parents require separate crop-embedding analysis; use --skip-embeddings for exact structural analysis")
    if not skip_embeddings:
        validate_embedding_inputs(paths, protocol)
    if expanded:
        sources_dir = Path(config.get("metadata_sources_dir", paths["metadata_source_lock"].parent))
        validated_metadata = validate_full_source_metadata(
            metadata_path=paths["structure_metadata"], receipt_path=paths["metadata_receipt"],
            chain_data_cache=paths["chain_data_cache"], metadata_sources_dir=sources_dir,
            metadata_source_lock=paths["metadata_source_lock"], replay=True)
        metadata_rows = validated_metadata["chains"]
        metadata_receipt = json.loads(paths["metadata_receipt"].read_text())
        source_pins = []
        for binding in metadata_receipt["input_pins"]["metadata_sources"].values():
            actual = pin(sources_dir / binding["path"])
            if actual["sha256"] != binding["sha256"]:
                raise ValueError("Metadata source changed after owning full-source replay")
            source_pins.append(actual)
        initial_pins["metadata_source_files"] = source_pins
    else:
        metadata_rows = read_rows(paths["structure_metadata"])
    metadata = {row["chain_id"]: row for row in metadata_rows}
    if len(metadata) != len(metadata_rows):
        raise ValueError("Duplicate chain metadata")
    source = read_rows(paths["historical_source_rows"])
    source_lookup = {row["chain_id"]: row for row in source}
    if len(source_lookup) != len(source):
        raise ValueError("Duplicate source chain rows")
    qualified = read_rows(paths["qualified_rows"])
    reference = sorted([canonical_row(row, metadata[row["chain_id"]]) for row in qualified], key=lambda row: row["chain_id"])
    repaired_public = bool(protocol.get("repaired_public_amendment"))
    training_owned_ids = validate_expanded_frame(paths, reference, receipt, repaired_public=repaired_public) if expanded else {row["chain_id"] for row in reference}
    lookup = dict(source_lookup)
    lookup.update({row["chain_id"]: row for row in reference})
    baseline_requalification = None
    baseline_current_predicate_verified = False
    if "candidate06_rows" in paths:
        baseline_rows = baseline_requalification_rows(paths, protocol["target_size"])
        baseline_requalification = paths["candidate06_requalification"]
        baseline_current_predicate_verified = json.loads(baseline_requalification.read_text()).get("policy") == "fixed_baseline_verified_source_replay_v1"
        for row in baseline_rows:
            lookup[row["chain_id"]] = {**lookup[row["chain_id"]], **row}
    evaluation_groups = set(manifest(paths["fixed_eval_groups"])) if not expanded else set()
    if not expanded and any(row["group_id"] in evaluation_groups for row in reference):
        raise ValueError("Qualified training reference contains fixed evaluation groups")
    arms, unavailable = draw_controls(reference, protocol["target_size"], protocol["seeds"], protocol["seed_namespace"], group_policy=group_policy)
    reference_by_sequence = {row["sequence_sha1"]: row for row in reference}
    baseline_incompatible = []
    candidate_arms, candidate_rows = [], []
    for name in ("candidate06", "candidate07"):
        members = manifest(paths[name])
        if len(members) != protocol["target_size"]:
            raise ValueError("Candidate manifest size differs from protocol")
        if name == "candidate07" and not set(members) <= {row["chain_id"] for row in reference}:
            raise ValueError("Candidate07 contains a chain outside the qualified training pool")
        rows = [canonical_row(lookup[chain], metadata[chain]) for chain in members]
        if any(row["group_id"] in evaluation_groups for row in rows) and not (name == "candidate06" and expanded):
            raise ValueError("Candidate includes an evaluation-owned group")
        if name == "candidate06" and expanded:
            baseline_incompatible = [row["chain_id"] for row in rows
                                     if row["chain_id"] not in training_owned_ids or row["sequence_sha1"] not in reference_by_sequence]
            for row in rows:
                current = reference_by_sequence.get(row["sequence_sha1"])
                if current is not None:
                    row["group_id"], row["cluster40_id"] = current["group_id"], current["cluster40_id"]
        if name == "candidate07":
            if group_policy == "all_groups" and {row["group_id"] for row in rows} != {row["group_id"] for row in reference}:
                raise ValueError("Candidate07 does not preserve the exact required training group set")
            if len({row["sequence_sha1"] for row in rows}) != len(rows):
                raise ValueError("Candidate07 repeats an exact sequence")
        candidate_arms.append({"name": name, "method": name, "seed": None})
        candidate_rows.append(rows)
    all_rows = candidate_rows + [[reference[i] for i in arm["indices"]] for arm in arms]
    arms = candidate_arms + arms
    baseline_comparable = not baseline_incompatible and (not expanded or baseline_current_predicate_verified)
    # Compile feature tokens once, without reducing repeat-domain multiplicity.
    for row in reference + candidate_rows[0] + candidate_rows[1]:
        row["_tokens"] = tokens(row, protocol)
    reference_distributions = {mode: distributions(reference, protocol, mode) for mode in protocol["weighting"]}
    reference_support = {name: independent_support(reference, field) for name, field in DOMAIN_FIELDS.items()}
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / "input-pins.json", initial_pins)
    write_json(out / "memberships.json", [{key: arm[key] for key in ("name", "method", "seed")} |
                                          {"arm_id": arm["name"],
                                           "kind": "historical" if arm["name"] == "candidate06" and not baseline_comparable else
                                                   "candidate" if arm["name"] in {"candidate06", "candidate07"} else "control",
                                           "primary_compatible": arm["name"] != "candidate06" or baseline_comparable,
                                           "chains": [row["chain_id"] for row in rows],
                                           "parent_sequence_sha256": [row["sequence_sha256"] for row in rows]}
                                          for arm, rows in zip(arms, all_rows, strict=True)])
    write_json(out / "reference-distributions.json", reference_distributions)
    records = []
    for arm, rows in zip(arms, all_rows, strict=True):
        metrics = describe_arm(rows, reference, protocol, reference_distributions, reference_support)
        if arm["name"] == "candidate06" and not baseline_comparable:
            # Excluded parents have no current training-diversity component.
            # Old group IDs cannot supply an apparent current-group metric.
            for key in metrics:
                if key in {"groups", "cluster40"} or key.startswith(("group_", "group_uniform.")) or key.endswith("_groups"):
                    metrics[key] = None
        records.append({key: arm[key] for key in ("name", "method", "seed")} | {"metrics": metrics})
    write_json(out / "metadata-metrics.json", records)
    write_json(out / "metadata-summary.json", summarize(records, baseline_comparable=baseline_comparable))
    reference_chains = {row["chain_id"] for row in reference}
    reference_sequences = {row["sequence_sha1"] for row in reference}
    write_json(out / "denominators.json", {"reference_chains": len(reference), "reference_groups": len({r["group_id"] for r in reference}),
                                          "controls_unavailable": unavailable, "cath_known_support": reference_support,
                                          "candidate06_chains_absent_from_qualified_representatives": [r["chain_id"] for r in candidate_rows[0] if r["chain_id"] not in reference_chains],
                                          "candidate06_sequences_absent_from_qualified_reference": [r["sequence_sha1"] for r in candidate_rows[0] if r["sequence_sha1"] not in reference_sequences],
                                          "candidate06_incompatible_with_current_training_frame": baseline_incompatible,
                                          "candidate06_primary_comparable": baseline_comparable, "group_policy": group_policy,
                                          "candidate06_current_predicate_verified": baseline_current_predicate_verified,
                                          "geometry": "Measured projected whole-chain CA distance profiles; annotation and missingness remain explicit. This is not domain-alignment accuracy."})
    if not skip_embeddings:
        embedding_analysis(paths, arms, all_rows, out)
    if expanded:
        analyze_crop_moments(reference, arms, all_rows, out, baseline_comparable=baseline_comparable)
        analyze_source_stages(metadata_rows, training_owned_ids, reference, candidate_rows[1], out)
    summary = summarize(records, baseline_comparable=baseline_comparable)
    deltas = summary.get("candidate07_minus_candidate06", {})
    primary = {key: value for key, value in deltas.items() if
               key.startswith("sequence_uniform.") and any(part in key for part in
               ("domain_topology.", "domain_superfamily.", "topology_x_domain_length.", "topology_x_domain_count."))}
    acceptance: dict[str, Any] = {"status": "requires_scientific_review", "reference_qualification": "complete",
                  "candidate07_exact_group_coverage": {r["group_id"] for r in candidate_rows[1]} == {r["group_id"] for r in reference},
                  "group_policy": group_policy, "candidate06_primary_comparable": baseline_comparable,
                  "primary_candidate07_minus_candidate06": primary,
                  "primary_discrepancy_increases": {key: value for key, value in primary.items() if value is not None and value > 0},
                  "labels_and_full_dataset_integrity": "Must be supplied by separate dataset validation; not established by this analysis",
                  "baseline_qualification": "Membership comparison records compatibility; historical candidate06 is not silently certified under the new predicate",
                  "static_metrics_do_not_establish_folding_improvement": True}
    if baseline_requalification is not None:
        acceptance["baseline_qualification"] = "All actual candidate06 train chains passed the separately pinned requalification; predicate provenance is retained in that receipt"
    if expanded:
        crop_summary = json.loads((out / "crop-summary.json").read_text())
        acceptance["expanded_crop_primary"] = matched_control_evidence(crop_summary, "uniform_subset_vs_sequence_uniform.")
        acceptance["primary_candidate07_minus_candidate06"] = {
            key: value for key, value in crop_summary.get("candidate07_minus_candidate06", {}).items()
            if key.startswith("uniform_subset_vs_sequence_uniform.") and key.endswith(("raw_l1", "shape_js_bits", "shape_tvd"))}
        acceptance["primary_discrepancy_increases"] = {
            key: value for key, value in acceptance["primary_candidate07_minus_candidate06"].items() if value is not None and value > 0}
        acceptance["embedding_assessment"] = "Requires separately pinned crop ESMC results for all three fixed crop seeds before scientific review"
        acceptance["public_evaluation_policy"] = (
            "Repaired active public is primary; original public remains protected and is reported separately as historical sensitivity"
            if repaired_public else "Original frozen public cohort")
    if not skip_embeddings:
        embedding_summary = json.loads((out / "embedding-summary.json").read_text())
        heldout = {key: value for key, value in embedding_summary["candidate07_minus_candidate06"].items()
                   if key.startswith("public_val.sequence_uniform.")}
        acceptance["public_val_candidate07_minus_candidate06"] = heldout
        acceptance["adverse_public_val_directions"] = {key: value for key, value in heldout.items() if
                                                       ("coverage" in key and value < 0) or ("distance" in key and value > 0)}
    write_json(out / "acceptance-evidence.json", acceptance)
    # A changed input or implementation invalidates completion, including changes
    # during a long blocked-GEMM pass. Preserve partial outputs for diagnosis.
    for record in [initial_pins["protocol"], initial_pins["config"], initial_pins["implementation"],
                   *initial_pins["implementation_dependencies"], *initial_pins.get("metadata_source_files", []),
                   *config["inputs"].values()]:
        checked_input(record)
    write_json(out / "completion.json", {"status": "complete", "arms": len(arms), "numpy": np.__version__,
                                         "protocol_sha256": expected_protocol_sha, "embeddings_evaluated": not skip_embeddings,
                                         "scientific_scope": "Descriptive selection diagnostics; no training or accuracy experiment"})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, epilog=(
        "Config JSON inputs maps names to {path,sha256,bytes}. Required: qualified_rows (JSONL), "
        "qualification_receipt (status complete), structure_metadata, historical_source_rows (TSV), "
        "candidate06/candidate07 (manifests), fixed_eval_groups (one group ID per line). Expanded-parent studies "
        "also require barrier_receipt (native full-parent or additive public-repair replay, matching the protocol) and training_owned_manifest "
        "(barrier-admitted chains before qualification), metadata_receipt, chain_data_cache, and metadata_source_lock. "
        "Metadata sources default to the lock's directory; config metadata_sources_dir may relocate them. Optional "
        "candidate06_rows and candidate06_requalification bind the combined historical requalification. Embedding inputs: "
        "exact_helper, embedding_reference, embedding_cache, embedding_metadata, embedding_radii, "
        "embedding_cohorts, historical_completion, historical_embedding_distances. Optional historical_protocol "
        "may relocate the byte-identical accepted historical protocol for another machine. No private inputs belong in Git."))
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--skip-embeddings", action="store_true", help="Explicit metadata-only partial measurement; recorded in completion")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args.config, args.protocol, args.protocol_sha256, args.out, skip_embeddings=args.skip_embeddings)
