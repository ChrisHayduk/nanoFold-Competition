"""Exact parent-aware ESMC crop coverage from completed, hash-pinned artifacts.

Use --config CONFIG --output-dir NEW_DIRECTORY. This postprocessor performs no
inference/downloads. CONFIG (schema_version=1) pins protocol, reference_rows,
embeddings, embedding_provenance, inference_validator, inference_completion,
barrier_receipt, qualification_receipt, memberships and query_cohorts using
path/sha256 records (optional bytes). Paths are relative to CONFIG. Embeddings
and provenance are the original uniform-CUDA three-seed producer artifacts;
inference_completion is its successful scheduler receipt. The pinned native
validator must expose validate_full(Path)->dict and verify all original chunks.

Reference JSONL has one crop per parent: parent_sequence_sha256,
crop_sequence_sha256, vector_record_id, crop_seed, length, source_chain_count,
group_id, cluster40_id, pool_roles, and boolean source_qualified/training_owned/
evaluation_owned. Native barrier/qualification replay binds every parent,
chain alias, training label and flag. Evaluation-owned means actual frozen
heldout parents, including protected historical public targets after a repair;
admission_excluded is derived separately. pool_roles.public_val identifies the
active public targets; revised frames also require pool_roles.historical_public
for every original public target. No global grouping
is inferred from training diversity groups. Deduplicated vectors join through
the producer's exact per-seed parent/crop mapping, including every parent.

Membership JSON is a list of arms with arm_id, method, seed, kind, explicit
primary_compatible, and parallel chains/parent_sequence_sha256 lists. New arms
must use the actual qualified representative; controls are independently
replayed with the frozen seeds and policies. Incompatible historical arms are descriptive; a compatible candidate06 baseline retains primary comparisons.
Query JSON has a cohorts list: name, semantics, expected_count, and parent SHA
list. Required cohorts are public_val (heldout), one full_reference, one
leave_parent_out, and qualified_training_reference (qualified_training).
Revised public frames additionally require historical_public (historical_public)
and preserved_public (preserved_public): the original public set and its exact
intersection with the active set, both descriptive with separate denominators.
Optional heldout and barrier_excluded cohorts retain their distinct meaning.
All means/coverage have parent-uniform and source-chain-abundance versions;
group-uniform applies only to the complete qualified training query cohort.

CONFIG declares crop_seed, arm_size, control_methods, control_seeds and
candidate_arm_id. An optional candidate06 baseline also pins candidate06 (manifest), candidate06_rows and candidate06_requalification. Available controls require every frozen seed; unavailable
methods must match the actual qualified-frame feasibility. Production requires
three methods/100 seeds and 1000 active whole-short public queries, plus all
1000 historical and 786 preserved public queries when the revised frame is in use. pilot=true
permits smaller tests but never reports full completion. batch_size is at most
128. Optional radius_cache/radius_cache_contract pins must match the exact
reference, vectors, seed, normalization, NumPy version and chunk contract.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import platform
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
for directory in (REPO_ROOT, REPO_ROOT / "scripts"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from analyze_structural_sampling import (
    baseline_requalification_rows,
    deterministic_crop,
    draw_controls,
    validate_historical_baseline,
    validate_qualification_frame,
)

from nanofold.revised_parent_leakage import POLICY as REVISED_FRAME_POLICY
from nanofold.revised_parent_leakage import validate_parent_frame
from nanofold.structural_population import validate_receipt

KS = (10, 50)
METRICS = ("mean_distance", "distance_p50", "distance_p90", "distance_p95",
           "query_local_10nn_coverage", "query_local_50nn_coverage",
           "fixed_global_10nn_radius_coverage", "fixed_global_50nn_radius_coverage")


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def pin(path: Path) -> dict[str, Any]:
    return {"path": str(path.resolve()), "sha256": digest(path), "bytes": path.stat().st_size}


def write_json(path: Path, value: Any) -> None:
    with path.open("x") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def load_pins(config: dict[str, Any], base: Path) -> dict[str, dict[str, Any]]:
    names = ["protocol", "reference_rows", "embeddings", "embedding_provenance", "inference_validator", "inference_completion", "barrier_receipt",
             "qualification_receipt", "memberships", "query_cohorts"]
    if ("radius_cache" in config) != ("radius_cache_contract" in config):
        raise ValueError("Radius cache and exact contract must be supplied together")
    if "radius_cache" in config:
        names.extend(["radius_cache", "radius_cache_contract"])
    names.extend(name for name in ("candidate06", "candidate06_rows", "candidate06_requalification", "historical_protocol") if name in config)
    result = {}
    for name in names:
        wanted = config[name]
        actual = pin(base / wanted["path"])
        if actual["sha256"] != wanted["sha256"] or ("bytes" in wanted and actual["bytes"] != wanted["bytes"]):
            raise ValueError(f"Frozen artifact changed: {name}")
        result[name] = actual
    return result


def validate_rows(rows: list[dict[str, Any]], crop_seed: int) -> None:
    if not rows:
        raise ValueError("Empty reference")
    for key in ("parent_sequence_sha256", "vector_record_id"):
        values = [row[key] for row in rows]
        if len(set(values)) != len(values):
            raise ValueError(f"Duplicate {key}: a study must have one crop per parent")
    for row in rows:
        for key in ("parent_sequence_sha256", "crop_sequence_sha256"):
            if not isinstance(row[key], str) or len(row[key]) != 64 or any(c not in "0123456789abcdef" for c in row[key]):
                raise ValueError(f"Invalid exact sequence SHA256: {key}")
        if type(row["crop_seed"]) is not int or row["crop_seed"] != crop_seed:
            raise ValueError("Reference combines different crop seeds")
        for key in ("source_chain_count", "length"):
            if type(row[key]) is not int or row[key] < 1:
                raise ValueError(f"Invalid positive integer {key}")
        for key in ("source_qualified", "training_owned", "evaluation_owned"):
            if type(row[key]) is not bool:
                raise ValueError(f"Ownership and qualification must be boolean: {key}")
        if row["training_owned"] and row["evaluation_owned"]:
            raise ValueError("Conflicting train/evaluation ownership")
        if not isinstance(row["vector_record_id"], str) or not row["vector_record_id"]:
            raise ValueError("Missing vector identity")
        for key in ("group_id", "cluster40_id"):
            if row[key] is not None and (not isinstance(row[key], str) or not row[key]):
                raise ValueError("Group/cluster identity must be a string or explicit null")
            if row["training_owned"] and row[key] is None:
                raise ValueError("Training-owned parents require native group/cluster identities")
        if not isinstance(row["pool_roles"], list) or any(not isinstance(role, str) for role in row["pool_roles"]):
            raise ValueError("pool_roles must be an explicit list")
        if "public_val" in row["pool_roles"] and not row["evaluation_owned"]:
            raise ValueError("Public validation row must be evaluation-owned")


def normalize(values: np.ndarray) -> np.ndarray:
    if values.dtype.kind != "f":
        raise ValueError("Embedding vectors must have a real floating-point dtype")
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("Embeddings must be finite and two-dimensional")
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if not np.isfinite(norms).all() or np.any(norms <= 0):
        raise ValueError("Zero or invalid embedding norm")
    return np.ascontiguousarray(values / norms)


def align_embeddings(ids: np.ndarray, values: np.ndarray, rows: list[dict[str, Any]], dimensions: int = 2560) -> np.ndarray:
    if ids.ndim != 1 or ids.dtype.kind not in {"U", "S"}:
        raise ValueError("Vector IDs must be a one-dimensional string array")
    names = ids.astype(str).tolist()
    if len(set(names)) != len(names) or values.shape != (len(names), dimensions):
        raise ValueError("Duplicate vector IDs or invalid embedding shape")
    lookup = {name: index for index, name in enumerate(names)}
    wanted = [row["vector_record_id"] for row in rows]
    if set(names) != set(wanted):
        raise ValueError("Embedding artifact must contain the exact complete reference vector IDs")
    return normalize(values[np.array([lookup[name] for name in wanted])])



def align_crop_embeddings(ids: np.ndarray, values: np.ndarray, mapping: list[dict], rows: list[dict], seed: int) -> np.ndarray:
    """Join the receipted deduplicated multi-seed cache without intersecting populations."""
    if ids.ndim != 1 or ids.dtype.kind not in {"U", "S"} or values.shape != (len(ids), 2560):
        raise ValueError("Invalid full inference vector array")
    names = ids.astype(str).tolist()
    if len(set(names)) != len(names):
        raise ValueError("Full inference repeats crop vector IDs")
    index = {name: position for position, name in enumerate(names)}
    by_parent = {row["parent_sequence_sha256"]: row for row in mapping}
    if len(by_parent) != len(mapping) or set(by_parent) != {row["parent_sequence_sha256"] for row in rows}:
        raise ValueError("Seed mapping does not cover exactly the complete reference parents")
    wanted = []
    for row in rows:
        actual = by_parent[row["parent_sequence_sha256"]]
        if (actual["crop_seed"] != seed or actual["crop_sequence_sha256"] != row["crop_sequence_sha256"]
                or actual["source_vector_record_id"] != actual["crop_sequence_sha256"]
                or actual["parent_length"] != row["length"]
                or actual["stop"] - actual["start"] != min(row["length"], 256)
                or actual["start"] < 0 or actual["stop"] > row["length"]
                or actual["start"] != row.get("_expected_crop_start", actual["start"])
                or actual["stop"] != row.get("_expected_crop_stop", actual["stop"])):
            raise ValueError("Seed mapping differs from the reference crop identity")
        if actual["source_vector_record_id"] not in index:
            raise ValueError("Seed crop is missing from complete inference artifact")
        wanted.append(index[actual["source_vector_record_id"]])
    return normalize(values[np.asarray(wanted, dtype=np.int64)])


def validate_inference(paths: dict[str, Path], protocol_sha: str) -> dict:
    """Use the separately pinned producer validator, including scheduler completion."""
    if sys.flags.optimize:
        raise ValueError("Native inference receipts require Python assertions enabled")
    spec = importlib.util.spec_from_file_location("nanofold_pinned_crop_inference_validator", paths["inference_validator"])
    if spec is None or spec.loader is None:
        raise ValueError("Cannot load pinned inference validator")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    validated = module.validate_full(paths["embedding_provenance"].parent)
    for key, name in (("embeddings", "embeddings"), ("provenance", "embedding_provenance"),
                      ("scheduler", "inference_completion")):
        actual = pin(paths[name])
        if (Path(validated[key]["path"]).resolve() != paths[name].resolve()
                or validated[key]["sha256"] != actual["sha256"]):
            raise ValueError("Inference validation does not bind the configured source artifact")
    contract = validated["contract"]
    if (contract.get("schema") != "nanofold-esmc-uniform-cuda-v1" or contract.get("model_id") != "biohub/ESMC-6B"
            or contract.get("model_revision") != "af1602ba7406f521b11bf8f81d52af378cde09e4"
            or contract.get("esm_code_revision") != "43b4548b86762edfa747b07d5f440aad3c33acee"
            or contract.get("dimensions") != 2560 or contract.get("protocol_sha256") != protocol_sha
            or contract.get("seeds") != [0, 1, 2] or not contract.get("pooling")
            or contract.get("environment", {}).get("cuda") is None):
        raise ValueError("Inference contract differs from frozen ESMC6B uniform-CUDA extraction")
    return validated


def load_cohorts(document: dict[str, Any], rows: list[dict[str, Any]], *, pilot: bool = False,
                 repaired_public: bool = False) -> list[dict[str, Any]]:
    by_parent = {row["parent_sequence_sha256"]: index for index, row in enumerate(rows)}
    cohorts, seen = [], set()
    for cohort in document["cohorts"]:
        name, semantics = cohort["name"], cohort["semantics"]
        if name in seen or semantics not in {"heldout", "historical_public", "preserved_public", "barrier_excluded", "full_reference", "leave_parent_out", "qualified_training"}:
            raise ValueError("Duplicate cohort name or unknown cohort semantics")
        seen.add(name)
        parents = cohort["parent_sequence_sha256"]
        if not parents or len(set(parents)) != len(parents) or len(parents) != cohort["expected_count"]:
            raise ValueError("Empty, duplicated or incomplete fixed query cohort")
        if set(parents) - by_parent.keys():
            raise ValueError("Query parent is absent from the complete reference")
        indices = np.array([by_parent[parent] for parent in parents], dtype=np.int64)
        if semantics in {"full_reference", "leave_parent_out"} and set(parents) != set(by_parent):
            raise ValueError("Full-reference and leave-parent-out cohorts must include every parent")
        if semantics in {"heldout", "historical_public", "preserved_public"} and any(not rows[index]["evaluation_owned"] for index in indices):
            raise ValueError("Held-out query cohort includes a non-evaluation-owned parent")
        if semantics == "barrier_excluded" and any(rows[index]["training_owned"] for index in indices):
            raise ValueError("Barrier-excluded cohort includes admitted training parents")
        if name == "qualified_training_reference":
            expected_qualified = {row["parent_sequence_sha256"] for row in rows if row["source_qualified"] and row["training_owned"]}
            if semantics != "qualified_training" or set(parents) != expected_qualified:
                raise ValueError("Qualified training cohort must include the complete qualified admitted population")
        elif semantics == "qualified_training":
            raise ValueError("Qualified training cohort requires the frozen qualified_training_reference name")
        if name in {"public_val", "historical_public", "preserved_public"}:
            expected_semantics = "heldout" if name == "public_val" else name
            if name != "public_val" and not repaired_public:
                raise ValueError("Historical and preserved public queries require the revised public protocol and frame")
            roles = {"public_val", "historical_public"} if name == "preserved_public" else {name}
            expected = {row["parent_sequence_sha256"] for row in rows if roles <= set(row["pool_roles"])}
            expected_n = 786 if name == "preserved_public" else 1000
            if semantics != expected_semantics or set(parents) != expected or (not pilot and len(parents) != expected_n):
                raise ValueError(f"{name} must be the complete fixed {expected_n}-parent cohort")
            if any(rows[index]["length"] > 256 or rows[index]["crop_sequence_sha256"] != rows[index]["parent_sequence_sha256"] for index in indices):
                raise ValueError("Fixed public validation queries must be whole sequences of at most 256 residues")
        elif semantics in {"historical_public", "preserved_public"}:
            raise ValueError("Historical and preserved public queries require their frozen cohort names")
        role = ("primary_public" if name == "public_val" else "historical_descriptive" if name == "historical_public"
                else "preserved_descriptive" if name == "preserved_public"
                else "reference_diagnostic")
        cohorts.append({**cohort, "indices": indices, "cohort_role": role})
    if repaired_public and "historical_public" not in seen:
        raise ValueError("Missing required original historical_public cohort after public repair")
    if repaired_public and "preserved_public" not in seen:
        raise ValueError("Missing required common preserved_public cohort after public repair")
    if "qualified_training_reference" not in seen:
        raise ValueError("Missing prespecified qualified_training_reference cohort")
    if "public_val" not in seen:
        raise ValueError("Missing fixed public_val query cohort")
    if not {"full_reference", "leave_parent_out"} <= {cohort["semantics"] for cohort in cohorts}:
        raise ValueError("Missing prespecified full-reference or leave-parent-out cohort")
    return cohorts


def validate_arms(document: Any, rows: list[dict[str, Any]], arm_size: int, methods: list[str],
                  seeds: list[int], candidate: str, statuses: dict[str, Any] | None = None,
                  disjointness_fields: tuple[str, ...] = ("parent_sequence_sha256",)) -> tuple[list[dict], np.ndarray]:
    arms = document if isinstance(document, list) else document["arms"]
    if arm_size < 1 or not methods or len(set(methods)) != len(methods):
        raise ValueError("Invalid arm size or control methods")
    if not seeds or len(set(seeds)) != len(seeds) or any(type(seed) is not int or seed < 0 for seed in seeds):
        raise ValueError("Control seeds must be distinct nonnegative integers")
    statuses = statuses or {}
    if set(statuses) - set(methods):
        raise ValueError("Control status names an unfrozen method")
    unavailable = set()
    for method, status in statuses.items():
        if status.get("status") not in {"available", "unavailable"}:
            raise ValueError("Unknown frozen control status")
        if status["status"] == "unavailable":
            if not status.get("reason"):
                raise ValueError("Unavailable control must have an explicit reason")
            unavailable.add(method)
    if "parent_sequence_sha256" not in disjointness_fields or set(disjointness_fields) - {"parent_sequence_sha256", "group_id", "cluster40_id"}:
        raise ValueError("Invalid barrier disjointness contract")
    by_parent = {row["parent_sequence_sha256"]: index for index, row in enumerate(rows)}
    evaluation = [row for row in rows if row["evaluation_owned"]]
    barriers = {key: {row[key] for row in evaluation} for key in ("parent_sequence_sha256", "group_id", "cluster40_id")}
    names: set[str] = set()
    observed: Counter[tuple[str, int]] = Counter()
    output, masks = [], []
    for arm in arms:
        name, kind = arm["arm_id"], arm["kind"]
        if not name or name in names or kind not in {"candidate", "control", "historical"}:
            raise ValueError("Duplicate arm or invalid arm kind")
        names.add(name)
        if type(arm["primary_compatible"]) is not bool:
            raise ValueError("Arm compatibility must be explicit boolean")
        if (kind == "historical") == arm["primary_compatible"]:
            raise ValueError("Only historical arms may be excluded from primary comparisons")
        parents = arm["parent_sequence_sha256"]
        if len(parents) != arm_size or len(set(parents)) != arm_size:
            raise ValueError("Arm has incorrect size or duplicate exact parents")
        if set(parents) - by_parent.keys():
            raise ValueError("Arm member has no reference vector; incomplete controls are not admissible")
        indices = [by_parent[parent] for parent in parents]
        if "chains" in arm:
            chains = arm["chains"]
            if len(chains) != arm_size or len(set(chains)) != arm_size:
                raise ValueError("Arm chain identities are incomplete or duplicated")
            for chain, index in zip(chains, indices, strict=True):
                aliases = rows[index].get("chain_ids", [rows[index].get("chain_id")])
                if chain not in aliases:
                    raise ValueError("Membership chain identity is not bound to its exact parent")
        violations: Counter[str] = Counter()
        coincidences: Counter[str] = Counter()
        for index in indices:
            row = rows[index]
            if not row["source_qualified"]:
                violations["not_source_qualified"] += 1
            if not row["training_owned"]:
                violations["not_training_owned"] += 1
            for key, values in barriers.items():
                if row[key] in values:
                    coincidences[f"evaluation_{key}_overlap"] += 1
                    if key in disjointness_fields:
                        violations[f"evaluation_{key}_overlap"] += 1
        if violations and kind != "historical":
            raise ValueError(f"Training arm violates qualification or evaluation barrier: {name}: {dict(violations)}")
        if kind == "control":
            if (arm["method"] not in methods or arm["method"] in unavailable
                    or type(arm["seed"]) is not int or arm["seed"] not in seeds):
                raise ValueError("Control method/seed differs from frozen contract")
            observed[(arm["method"], arm["seed"])] += 1
        mask = np.zeros(len(rows), dtype=bool)
        mask[indices] = True
        masks.append(mask)
        output.append({key: value for key, value in arm.items() if key not in {"parent_sequence_sha256", "chains"}} |
                      {"barrier_violations": dict(violations), "evaluation_identity_coincidences": dict(coincidences)})
    expected = Counter({(method, seed): 1 for method in methods if method not in unavailable for seed in seeds})
    if observed != expected:
        raise ValueError("Incomplete or duplicated frozen control method/seed set")
    if sum(arm["kind"] == "candidate" and arm["arm_id"] == candidate for arm in output) != 1:
        raise ValueError("Missing primary candidate arm")
    if any(arm["kind"] == "candidate" and arm["arm_id"] not in {candidate, "candidate06"} for arm in output):
        raise ValueError("Additional candidate must be the fixed candidate06 baseline")
    return output, np.stack(masks)



def baseline_exposure(pairs: dict[str, str], by_chain: dict[str, dict], admitted: set[str],
                      cohort_chains: dict[str, set[str]]) -> dict:
    """Describe exact exposure and native exclusion without treating absence as independence."""
    baseline_chains, baseline_parents = set(pairs), set(pairs.values())
    excluded = sorted(baseline_chains - admitted)
    cohorts = {}
    for name, chains in cohort_chains.items():
        parents = {by_chain[chain]["sequence_sha256"] for chain in chains}
        cohorts[name] = {"query_chain_n": len(chains), "query_parent_n": len(parents),
                         "direct_chain_overlap_n": len(chains & baseline_chains),
                         "exact_parent_overlap_n": len(parents & baseline_parents)}
    return {"status": "complete", "cohorts": cohorts,
            "final_frame_barrier": {"excluded_chain_n": len(excluded),
                                    "excluded_parent_n": len({pairs[chain] for chain in excluded}),
                                    "excluded_chains": excluded,
                                    "reason_chain_counts": dict(sorted(Counter(
                                        reason for chain in excluded for reason in set(by_chain[chain]["training_exclusion_reasons"])
                                    ).items()))},
            "interpretation": "Descriptive exact exposure and final native admission only; zero exact overlaps do not establish generalization independence."}


def validate_admission(paths: dict[str, Path], rows: list[dict], document: Any, protocol: dict) -> dict:
    """Replay the native directed barrier and source qualification, then bind every parent."""
    frame = validate_parent_frame(paths["barrier_receipt"])
    repaired_public = bool(protocol.get("repaired_public_amendment"))
    if (frame.get("policy") == REVISED_FRAME_POLICY) != repaired_public:
        raise ValueError("Public repair protocol and owning parent-frame policy differ")
    qualification = validate_receipt(paths["qualification_receipt"])
    validate_qualification_frame(qualification, frame)
    qualified = [json.loads(line) for line in Path(qualification["qualified_chains"]["path"]).read_text().splitlines() if line.strip()]
    by_chain = {row["chain_id"]: row for row in frame["records"]}
    by_parent: dict[str, list[dict]] = {}
    for row in frame["records"]:
        by_parent.setdefault(row["sequence_sha256"], []).append(row)
    admitted = set(frame["eligible_training_ids"])
    qualified_by_parent = {row["sequence_sha256"]: row for row in qualified}
    if len(qualified_by_parent) != len(qualified):
        raise ValueError("Qualification export repeats exact parents")
    public = {by_chain[chain]["sequence_sha256"] for chain in frame["frozen_splits"]["val"]}
    hidden = {by_chain[chain]["sequence_sha256"] for chain in frame["frozen_splits"]["hidden_val"]}
    historical_public: set[str] = set()
    if repaired_public:
        native_cohorts = frame.get("evaluation_cohorts", {})
        if (set(native_cohorts) != {"historical_public", "active_public", "hidden"}
                or native_cohorts["active_public"] != frame["frozen_splits"]["val"]
                or native_cohorts["hidden"] != frame["frozen_splits"]["hidden_val"]):
            raise ValueError("Revised native evaluation cohorts do not bind the active and fixed hidden splits")
        historical_public = {by_chain[chain]["sequence_sha256"] for chain in native_cohorts["historical_public"]}
    evaluation = public | historical_public | hidden
    if any(by_chain[chain]["sequence_sha256"] in evaluation for chain in admitted):
        raise ValueError("Native admission includes a protected active, historical or hidden evaluation parent")
    if {row["parent_sequence_sha256"] for row in rows} != set(by_parent):
        raise ValueError("Embedding reference is not the complete native metadata-eligible parent population")
    for row in rows:
        parent = row["parent_sequence_sha256"]
        copies = by_parent[parent]
        retained = [copy for copy in copies if copy["chain_id"] in admitted]
        if (row["training_owned"] != bool(retained) or row["evaluation_owned"] != (parent in evaluation)
                or row["source_qualified"] != (parent in qualified_by_parent)):
            raise ValueError("Reference ownership/qualification flags differ from native completed receipts")
        if ("public_val" in row["pool_roles"]) != (parent in public):
            raise ValueError("Public query identities differ from the active frozen heldout split")
        if ("historical_public" in row["pool_roles"]) != (parent in historical_public):
            raise ValueError("Historical public identities differ from the protected original public cohort")
        if row["source_chain_count"] != len(copies) or row["length"] != copies[0]["length"]:
            raise ValueError("Reference abundance/length differs from native source population")
        crop = deterministic_crop(copies[0]["sequence"], row["crop_seed"])
        if row["crop_sequence_sha256"] != crop["crop_sequence_sha256"]:
            raise ValueError("Reference crop identity differs from deterministic exact-parent crop")
        if retained and any(row[key] != copy[key] for copy in retained for key in ("group_id", "cluster40_id")):
            raise ValueError("Training diversity labels differ from reconstructed native frame")
        native_chains = sorted(copy["chain_id"] for copy in copies)
        if "chain_ids" in row and sorted(row["chain_ids"]) != native_chains:
            raise ValueError("Reference chain aliases differ from native exact-parent copies")
        row["chain_ids"] = native_chains  # Derived only from verified evidence, never caller assertions.
        row["admission_excluded"] = not bool(retained)
        row["_expected_crop_start"], row["_expected_crop_stop"] = crop["start"], crop["stop"]
    for row in qualified:
        chain = row["chain_id"]
        if (chain not in admitted or by_chain[chain]["sequence_sha256"] != row["sequence_sha256"]
                or any(by_chain[chain][key] != row[key] for key in ("group_id", "cluster40_id"))):
            raise ValueError("Qualified representative is not admitted by the directed barrier")
    arms = document if isinstance(document, list) else document["arms"]
    baselines = [arm for arm in arms if arm["arm_id"] == "candidate06"]
    exposure: dict[str, Any] = {"status": "unavailable", "reason": "No fixed candidate06 baseline supplied"}
    if len(baselines) > 1 or any(arm["kind"] == "historical" and arm["arm_id"] != "candidate06" for arm in arms):
        raise ValueError("Historical comparison must identify only the fixed candidate06 baseline")
    if baselines:
        if not {"candidate06", "candidate06_rows", "candidate06_requalification"} <= paths.keys():
            raise ValueError("Candidate06 requires original manifest and source-requalification pins")
        validate_historical_baseline(paths, protocol)
        baseline_rows = baseline_requalification_rows(paths, protocol["target_size"])
        baseline = baselines[0]
        pairs = {chain: parent for chain, parent in zip(baseline["chains"], baseline["parent_sequence_sha256"], strict=True)}
        if set(pairs) != {row["chain_id"] for row in baseline_rows} or any(
                chain not in by_chain or by_chain[chain]["sequence_sha256"] != parent for chain, parent in pairs.items()):
            raise ValueError("Candidate06 memberships differ from the fixed original exact parents")
        baseline_receipt = json.loads(paths["candidate06_requalification"].read_text())
        current_predicate_replayed = baseline_receipt.get("policy") == "fixed_baseline_verified_source_replay_v1"
        compatible = current_predicate_replayed and all(chain in admitted and parent in qualified_by_parent for chain, parent in pairs.items())
        if baseline["primary_compatible"] != compatible or baseline["kind"] != ("candidate" if compatible else "historical"):
            raise ValueError("Candidate06 compatibility differs from native barrier and qualification evidence")
        cohort_chains = {"public_val": set(frame["frozen_splits"]["val"]),
                         "hidden": set(frame["frozen_splits"]["hidden_val"])}
        if repaired_public:
            cohort_chains["historical_public"] = set(frame["evaluation_cohorts"]["historical_public"])
            cohort_chains["preserved_public"] = {
                chain for chain in cohort_chains["public_val"] if by_chain[chain]["sequence_sha256"] in public & historical_public}
        exposure = baseline_exposure(pairs, by_chain, admitted, cohort_chains)
        exposure["primary_compatible"] = compatible
        exposure["current_source_predicate_replayed"] = current_predicate_replayed
    for arm in arms:
        if arm["arm_id"] == "candidate06":
            continue
        if "chains" not in arm or len(arm["chains"]) != len(arm["parent_sequence_sha256"]):
            raise ValueError("New arms require exact qualified chain-to-parent bindings")
        for parent, chain in zip(arm["parent_sequence_sha256"], arm["chains"], strict=True):
            if parent not in qualified_by_parent or qualified_by_parent[parent]["chain_id"] != chain:
                raise ValueError("New arm member differs from actual source-qualified representative")
    control_frame = []
    for row in sorted(qualified, key=lambda value: value["chain_id"]):
        native = by_chain[row["chain_id"]]
        control_frame.append({**row, "sequence_sha1": hashlib.sha1(native["sequence"].encode("ascii")).hexdigest()})
    expected, unavailable = draw_controls(control_frame, protocol["target_size"], protocol["seeds"],
                                          protocol["seed_namespace"], group_policy="soft_diversity")
    expected_members = {(arm["method"], arm["seed"]): {control_frame[index]["chain_id"] for index in arm["indices"]}
                        for arm in expected}
    observed_members = {(arm["method"], arm["seed"]): set(arm["chains"]) for arm in arms if arm["kind"] == "control"}
    if observed_members != expected_members:
        raise ValueError("Frozen control memberships differ from deterministic qualified-frame draws")
    group_count = len({row["group_id"] for row in qualified})
    methods = ["uniform_sequence", "cluster40_uniform_sequence",
               "group_uniform_distinct" if group_count > protocol["target_size"] else "group_cover_random_sequence"]
    return {"methods": methods, "unavailable": unavailable, "public_parents": sorted(public),
            "repaired_public": repaired_public, "historical_public_parents": sorted(historical_public),
            "hidden_parents": sorted(hidden), "preserved_public_parents": sorted(public & historical_public),
            "baseline_exposure": exposure,
            "qualified_parent_n": len(qualified), "training_parent_n": len({by_chain[chain]["sequence_sha256"] for chain in admitted})}


def checked_distances(similarities: np.ndarray) -> np.ndarray:
    if not np.isfinite(similarities).all() or np.max(similarities) > 1.0001 or np.min(similarities) < -1.0001:
        raise ValueError("Invalid cosine similarities outside float32 tolerance")
    return np.clip(1.0 - similarities, 0.0, 2.0)


def exclude_parents(distances: np.ndarray, query_parents: np.ndarray, reference_parents: np.ndarray) -> np.ndarray:
    """Remove every row sharing the exact parent, without excluding other zero distances."""
    if distances.shape != (len(query_parents), len(reference_parents)):
        raise ValueError("Parent identity vectors do not match distance matrix")
    work = distances.copy()
    work[query_parents[:, None] == reference_parents[None, :]] = np.inf
    return work


def local_radii(distances: np.ndarray, query_parents: np.ndarray, reference_parents: np.ndarray,
                ks: tuple[int, ...] = KS) -> np.ndarray:
    if not ks or min(ks) < 1 or max(ks) >= distances.shape[1]:
        raise ValueError("Reference is too small for requested neighborhood")
    work = exclude_parents(distances, query_parents, reference_parents)
    result = np.partition(work, [k - 1 for k in ks], axis=1)[:, [k - 1 for k in ks]]
    if not np.isfinite(result).all():
        raise ValueError("Too few distinct-parent neighbors for exact radius")
    return result


def arm_nearest(distances: np.ndarray, selected: np.ndarray, shortlist: int = 64) -> tuple[np.ndarray, int]:
    """Exact shortlist minima with exhaustive fallback, including after parent exclusion."""
    if selected.ndim != 2 or selected.shape[1] != distances.shape[1] or selected.dtype != bool:
        raise ValueError("Invalid arm membership masks")
    if shortlist < 1 or np.any(selected.sum(axis=1) == 0) or np.isnan(distances).any():
        raise ValueError("Empty arm or invalid distances/shortlist")
    k = min(shortlist, distances.shape[1])
    candidates = np.argpartition(distances, k - 1, axis=1)[:, :k]
    short_distances = np.take_along_axis(distances, candidates, axis=1)
    result = np.empty((len(distances), len(selected)), dtype=np.float32)
    fallbacks = 0
    for index, mask in enumerate(selected):
        included = mask[candidates]
        nearest = np.min(np.where(included, short_distances, np.inf), axis=1)
        unresolved = ~np.isfinite(nearest)
        if unresolved.any():
            nearest[unresolved] = np.min(distances[unresolved][:, mask], axis=1)
            fallbacks += int(unresolved.sum())
        result[:, index] = nearest
    if not np.isfinite(result).all():
        raise ValueError("Arm has no remaining member after parent exclusion")
    return result, fallbacks


def compute(embeddings: np.ndarray, parents: np.ndarray, selected: np.ndarray, batch_size: int = 128,
            cached_radii: np.ndarray | None = None, output_dir: Path | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    if not 1 <= batch_size <= 128 or len(parents) != len(embeddings):
        raise ValueError("Query chunks must be between 1 and 128 and match parent identities")
    shape = (len(embeddings), len(selected))
    def allocate(name: str, array_shape: tuple[int, ...]) -> np.ndarray:
        if output_dir is None:
            return np.empty(array_shape, dtype=np.float32)
        return np.lib.format.open_memmap(output_dir / f"{name}.npy", mode="w+", dtype=np.float32, shape=array_shape)
    nearest = allocate("nearest", shape)
    leave_parent_out = allocate("leave_parent_out", shape)
    radii = allocate("local_radii", (len(embeddings), len(KS)))
    transpose = np.ascontiguousarray(embeddings.T)
    started, fallbacks = time.perf_counter(), 0
    for start in range(0, len(embeddings), batch_size):
        stop = min(start + batch_size, len(embeddings))
        distances = checked_distances(embeddings[start:stop] @ transpose)
        distances[np.arange(stop - start), np.arange(start, stop)] = 0.0
        nearest[start:stop], count = arm_nearest(distances, selected)
        fallbacks += count
        if cached_radii is None:
            radii[start:stop] = local_radii(distances, parents[start:stop], parents)
        else:
            radii[start:stop] = cached_radii[start:stop]
        distances = exclude_parents(distances, parents[start:stop], parents)
        leave_parent_out[start:stop], count = arm_nearest(distances, selected)
        fallbacks += count
        if output_dir is not None:
            progress = {"completed_queries": stop, "total_queries": len(embeddings),
                        "elapsed_seconds": time.perf_counter() - started}
            temporary = output_dir / "progress.tmp.json"
            temporary.write_text(json.dumps(progress) + "\n")
            temporary.replace(output_dir / "progress.json")
            print(json.dumps(progress), flush=True)
    for array in (nearest, leave_parent_out, radii):
        if isinstance(array, np.memmap):
            array.flush()
    return nearest, leave_parent_out, radii, {"seconds": time.perf_counter() - started,
                                           "exhaustive_fallback_rows": fallbacks, "all_queries_resolved": True}


def summarize(distances: np.ndarray, radii: np.ndarray, weights: np.ndarray, global_radii: np.ndarray) -> dict[str, Any]:
    if (not len(distances) or radii.shape != (len(distances), 2) or weights.shape != distances.shape
            or not np.isfinite(distances).all() or not np.isfinite(radii).all()
            or not np.isfinite(weights).all() or np.any(weights <= 0)):
        raise ValueError("Invalid full-denominator metric population")
    weights = weights / weights.sum()
    positive = radii[:, 1] > 0
    ratio = distances[positive] / radii[positive, 1]
    return {"n": len(distances), "mean_distance": float(np.sum(weights * distances)),
            "distance_p50": float(np.quantile(distances, .5)), "distance_p90": float(np.quantile(distances, .9)),
            "distance_p95": float(np.quantile(distances, .95)), "distance_quantiles_weighting": "parent_uniform_numpy_linear",
            "query_local_10nn_coverage": float(np.sum(weights * (distances <= radii[:, 0]))),
            "query_local_50nn_coverage": float(np.sum(weights * (distances <= radii[:, 1]))),
            "fixed_global_10nn_radius_coverage": float(np.sum(weights * (distances <= global_radii[0]))),
            "fixed_global_50nn_radius_coverage": float(np.sum(weights * (distances <= global_radii[1]))),
            "zero_local_10nn_radius_n": int(np.sum(radii[:, 0] == 0)),
            "zero_local_50nn_radius_n": int(np.sum(~positive)),
            "zero_local_50nn_radius_nonzero_distance_n": int(np.sum((~positive) & (distances > 0))),
            "normalized_gap_50_p50": float(np.quantile(ratio, .5)) if len(ratio) else None,
            "normalized_gap_50_p95": float(np.quantile(ratio, .95)) if len(ratio) else None,
            "normalized_gap_population_n": int(positive.sum()),
            "normalized_gap_policy": "positive_radius_only_parent_uniform; zero_radius_excluded_and_counted",
            "denominator_policy": "complete_fixed_query_cohort_no_unresolved_rows"}


def metric_rows(nearest: np.ndarray, leave_out: np.ndarray, radii: np.ndarray, rows: list[dict],
                arms: list[dict], cohorts: list[dict], global_radii: np.ndarray) -> list[dict]:
    result = []
    for cohort in cohorts:
        indices = cohort["indices"]
        subset = [rows[index] for index in indices]
        weightings = {"parent_uniform": np.ones(len(indices)),
                      "source_chain_abundance": np.array([row["source_chain_count"] for row in subset], dtype=np.float64)}
        group_available = cohort["semantics"] == "qualified_training"
        group_definition = ("Current native training diversity groups within the complete qualified admitted population" if group_available
                            else "Unavailable: current training diversity groups do not define full-reference or evaluation grouping")
        if group_available:
            labels = [row["group_id"] for row in subset]
            counts = Counter(labels)
            weightings["group_uniform"] = np.array([1 / counts[label] for label in labels])
        values = leave_out if cohort["semantics"] == "leave_parent_out" else nearest
        for index, arm in enumerate(arms):
            if not group_available:
                result.append({**arm, "cohort": cohort["name"], "semantics": cohort["semantics"],
                               "cohort_role": cohort["cohort_role"],
                               "weighting": "group_uniform", "status": "unavailable", "reason": group_definition,
                               "n": len(indices)})
            for weighting, weights in weightings.items():
                result.append({**arm, "cohort": cohort["name"], "semantics": cohort["semantics"],
                               "cohort_role": cohort["cohort_role"],
                               "weighting": weighting, "status": "complete", "primary_weighting": weighting == "parent_uniform",
                               "grouping_definition": group_definition if weighting == "group_uniform" else None,
                               "cohort_interpretation": "historical_descriptive_only" if arm["kind"] == "historical" else cohort["semantics"],
                               **summarize(values[indices, index], radii[indices], weights, global_radii)})
    return result


def control_summary(rows: list[dict], candidate: str) -> list[dict]:
    rows = [row for row in rows if row.get("status") != "unavailable"]
    result = []
    for cohort, weighting in sorted({(row["cohort"], row["weighting"]) for row in rows}):
        subset = [row for row in rows if row["cohort"] == cohort and row["weighting"] == weighting]
        actual = next(row for row in subset if row["arm_id"] == candidate and row["primary_compatible"])
        for method in sorted({row["method"] for row in subset if row["kind"] == "control"}):
            controls = [row for row in subset if row["method"] == method and row["kind"] == "control"]
            for metric in METRICS:
                values = np.array([row[metric] for row in controls])
                candidate_value = actual[metric]
                lower = "distance" in metric
                better = values > candidate_value if lower else values < candidate_value
                result.append({"cohort": cohort, "cohort_role": actual.get("cohort_role"),
                               "weighting": weighting, "control_method": method,
                               "metric": metric, "candidate_arm_id": candidate, "candidate": candidate_value,
                               "controls_n": len(values), "control_mean": float(np.mean(values)), "control_sd": float(np.std(values, ddof=1)) if len(values) > 1 else None,
                               "control_p05": float(np.quantile(values, .05)), "control_p50": float(np.median(values)),
                               "control_p95": float(np.quantile(values, .95)),
                               "candidate_minus_control_mean": float(candidate_value - np.mean(values)),
                               "candidate_minus_control_median": float(candidate_value - np.median(values)),
                               "candidate_better_midrank_fraction": float(np.mean(better) + .5 * np.mean(values == candidate_value)),
                               "direction": "lower_is_better" if lower else "higher_is_better"})
    return result



def baseline_comparison(rows: list[dict], candidate: str) -> dict:
    baseline = [row for row in rows if row["arm_id"] == "candidate06"]
    if not baseline:
        return {"status": "unavailable", "reason": "No fixed candidate06 baseline supplied", "rows": []}
    if any(not row["primary_compatible"] for row in baseline):
        return {"status": "unavailable", "reason": "Historical candidate06 is incompatible with the current admitted frame", "rows": []}
    result = []
    lookup = {(row["cohort"], row["weighting"]): row for row in baseline if row.get("status") != "unavailable"}
    for actual in rows:
        if actual["arm_id"] != candidate or actual.get("status") == "unavailable":
            continue
        control = lookup[(actual["cohort"], actual["weighting"])]
        for metric in METRICS:
            result.append({"cohort": actual["cohort"], "cohort_role": actual.get("cohort_role"),
                           "weighting": actual["weighting"], "metric": metric,
                           "candidate": actual[metric], "candidate06": control[metric],
                           "candidate_minus_candidate06": actual[metric] - control[metric]})
    return {"status": "complete", "scope": "same_fixed_cohort_descriptive_compatible_baseline", "rows": result}


def radius_contract(inputs: dict[str, dict], crop_seed: int, batch_size: int = 128) -> dict:
    return {"schema_version": 1, "reference_rows_sha256": inputs["reference_rows"]["sha256"],
            "embeddings_sha256": inputs["embeddings"]["sha256"],
            "embedding_provenance_sha256": inputs["embedding_provenance"]["sha256"],
            "crop_seed": crop_seed, "ks": list(KS), "batch_size": batch_size, "numpy_version": np.__version__,
            "distance": "cosine_l2_normalized_float32_numpy_gemm",
            "exclusion": "all_rows_with_same_exact_parent", "row_order": "reference_rows"}


def load_cached_radii(path: Path, contract_path: Path, expected_contract: dict, rows: list[dict]) -> np.ndarray:
    if json.loads(contract_path.read_text()) != expected_contract:
        raise ValueError("Cached radii contract differs from exact current reference/model contract")
    with np.load(path, allow_pickle=False) as data:
        if data["vector_record_ids"].astype(str).tolist() != [row["vector_record_id"] for row in rows]:
            raise ValueError("Cached radii rows differ from current reference order")
        radii = np.asarray(data["radii"], dtype=np.float32)
    if (radii.shape != (len(rows), len(KS)) or not np.isfinite(radii).all()
            or np.any(radii < 0) or np.any(radii > 2) or np.any(radii[:, 0] > radii[:, 1])):
        raise ValueError("Invalid cached exact radii")
    return radii


def run(config_path: Path, output_dir: Path) -> dict:
    config_pin, code_pin = pin(config_path), pin(Path(__file__))
    implementation_pins = {name: pin(REPO_ROOT / name) for name in (
        "scripts/analyze_structural_sampling.py", "nanofold/full_parent_leakage.py",
        "scripts/build_full_parent_frame.py", "nanofold/structural_population.py",
        "scripts/build_structural_population_plan.py", "scripts/build_manifests.py",
        "nanofold/revised_parent_leakage.py", "scripts/build_revised_parent_frame.py")}
    config = json.loads(config_path.read_text())
    if config.get("schema_version") != 1 or type(config["crop_seed"]) is not int or config["crop_seed"] < 0:
        raise ValueError("Unsupported configuration or crop seed")
    pilot = config.get("pilot", False)
    if type(pilot) is not bool:
        raise ValueError("Pilot flag must be boolean")
    if not pilot and (len(config["control_methods"]) != 3 or len(config["control_seeds"]) != 100):
        raise ValueError("Full comparison requires three control methods and 100 frozen seeds")
    inputs = load_pins(config, config_path.parent)
    paths = {name: Path(value["path"]) for name, value in inputs.items()}
    protocol = json.loads(paths["protocol"].read_text())
    if not isinstance(protocol, dict) or not protocol:
        raise ValueError("Missing frozen comparison protocol")
    amendment = protocol.get("expanded_parent_crop_amendment", {})
    authorized = amendment.get("authorized_full_inference", {})
    allowed_seeds = authorized.get("crop_seeds", [amendment.get("crop_seed")])
    if (config["crop_seed"] not in allowed_seeds or amendment.get("crop_size") != 256
            or protocol.get("target_size") != config["arm_size"] or protocol.get("seeds") != config["control_seeds"]):
        raise ValueError("Runtime crop, arm size or control seeds differ from frozen protocol")
    rows = [json.loads(line) for line in paths["reference_rows"].read_text().splitlines() if line.strip()]
    validate_rows(rows, config["crop_seed"])
    document = json.loads(paths["memberships"].read_text())
    admission = validate_admission(paths, rows, document, protocol)
    cohorts = load_cohorts(json.loads(paths["query_cohorts"].read_text()), rows, pilot=pilot,
                           repaired_public=admission["repaired_public"])
    if config["control_methods"] != admission["methods"]:
        raise ValueError("Control methods differ from the frozen qualified-frame policies")
    declared_unavailable = {key for key, value in config.get("control_status", {}).items() if value["status"] == "unavailable"}
    if declared_unavailable != set(admission["unavailable"]):
        raise ValueError("Unavailable controls differ from actual qualified-frame feasibility")
    arms, selected = validate_arms(document, rows, config["arm_size"],
                                  config["control_methods"], config["control_seeds"], config["candidate_arm_id"],
                                  config.get("control_status"))
    inference_protocol_sha = protocol.get("embedding_inference_protocol_sha256", inputs["protocol"]["sha256"])
    inference = validate_inference(paths, inference_protocol_sha)
    provenance = json.loads(paths["embedding_provenance"].read_text())
    mapping_pin = inference["mappings"][str(config["crop_seed"])]
    mapping_path = Path(mapping_pin["path"])
    if digest(mapping_path) != mapping_pin["sha256"]:
        raise ValueError("Validated seed mapping changed")
    inputs["crop_mapping"] = pin(mapping_path)
    mapping = [json.loads(line) for line in mapping_path.read_text().splitlines() if line.strip()]
    with np.load(paths["embeddings"], allow_pickle=False) as data:
        embeddings = align_crop_embeddings(data["ids"], data["embeddings"], mapping, rows, config["crop_seed"])
    contract = radius_contract(inputs, config["crop_seed"], config.get("batch_size", 128))
    cached = load_cached_radii(paths["radius_cache"], paths["radius_cache_contract"], contract, rows) if "radius_cache" in paths else None
    output_dir.mkdir(parents=True, exist_ok=False)
    launch = {"status": "running", "pilot": pilot, "config": config_pin, "code": code_pin, "inputs": inputs,
              "python": platform.python_version(), "numpy": np.__version__, "implementation_helpers": implementation_pins,
              "reference_n": len(rows),
              "arms_n": len(arms), "embedding_provenance": provenance, "inference_verification": inference["verification"], "radius_contract": contract, "embedding_inference_protocol_sha256": inference_protocol_sha,
              "crop_seed": config["crop_seed"],
              "crop_study_role": "primary" if config["crop_seed"] == authorized.get("primary_crop_seed", amendment["crop_seed"]) else "prespecified_robustness",
              "control_status": config.get("control_status", {}), "admission_validation": admission,
              "group_weighting_scope": "qualified_training_reference_only", "distance_scope": "exact_float32_all_reference_pairs_no_ANN"}
    write_json(output_dir / "launch.json", launch)
    parents = np.array([row["parent_sequence_sha256"] for row in rows])
    nearest, leave_out, radii, timing = compute(embeddings, parents, selected, config.get("batch_size", 128), cached, output_dir)
    global_radii = np.median(radii, axis=0)
    metrics = metric_rows(nearest, leave_out, radii, rows, arms, cohorts, global_radii)
    write_json(output_dir / "metrics.json", {"rows": metrics})
    write_json(output_dir / "control_summary.json", {
        "scope": "descriptive_frozen_seed_distribution_not_model_uncertainty",
        "historical_arms_excluded": [arm["arm_id"] for arm in arms if arm["kind"] == "historical"],
        "rows": control_summary(metrics, config["candidate_arm_id"])})
    write_json(output_dir / "baseline_comparison.json", {
        **baseline_comparison(metrics, config["candidate_arm_id"]), "exposure": admission["baseline_exposure"]})
    np.savez(output_dir / "radii_cache.npz", vector_record_ids=np.array([row["vector_record_id"] for row in rows]), radii=radii)
    write_json(output_dir / "radii_contract.json", contract)
    write_json(output_dir / "array_index.json", {"vector_record_ids": [row["vector_record_id"] for row in rows],
                                               "parent_sequence_sha256": parents.tolist(),
                                               "arms": [arm["arm_id"] for arm in arms]})
    validate_inference(paths, inference_protocol_sha)
    validate_parent_frame(paths["barrier_receipt"])
    validate_receipt(paths["qualification_receipt"])
    if "candidate06" in paths:
        validate_historical_baseline(paths, protocol)
        baseline_requalification_rows(paths, protocol["target_size"])
    for name, before in {**inputs, **implementation_pins, "config": config_pin, "code": code_pin}.items():
        if pin(Path(before["path"])) != before:
            raise ValueError(f"Frozen artifact changed during analysis: {name}")
    completion = {**launch, "status": "complete_pilot" if pilot else "complete", "timing": timing,
                  "global_radii": global_radii.tolist(), "global_radii_policy": "parent_uniform_median_full_reference_local_radii",
                  "artifacts": [pin(path) for path in sorted(output_dir.iterdir()) if path.name != "progress.json"],
                  "limitations": ["Embedding coverage does not establish folding accuracy or causal sampling superiority.",
                                  "Full-reference coverage includes selected-parent self matches; leave-parent-out excludes all such matches.",
                                  "Historical incompatible arms are descriptive only and excluded from primary deltas.",
                                  "Repaired active and protected historical public cohorts have separate fixed denominators; historical public metrics are descriptive only.",
                                  "Preserved public queries are the fixed active/historical intersection and a descriptive sensitivity; exact exposure counts do not establish generalization independence.",
                                  "Parent/source-chain weighting applies throughout; training-group weighting only to complete qualified training queries; quantiles remain parent-uniform.",
                                  "Source-chain abundance describes deposition frequency, not biological protein abundance or loader exposure.",
                                  "Each study contains one crop per parent, so it does not measure all within-parent crop diversity."]}
    write_json(output_dir / "completion.json", completion)
    return completion


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    run(args.config, args.output_dir)


if __name__ == "__main__":
    main()
