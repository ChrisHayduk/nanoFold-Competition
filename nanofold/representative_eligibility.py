"""Source-bound qualification of ranked members of immutable sampling units.

This module validates private receipts. It never downloads sources, repairs
missing evidence, or treats an acquisition failure as a biological rejection.
"""
from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import json
import math
import platform
import re
from pathlib import Path
from typing import Any

from nanofold.chain_paths import chain_data_dir
from nanofold.mmcif import extract_chain_atoms

POLICY = "cath-v4"
SOURCE_POLICY = "pinned_OpenProteinSet_historical_structures"
ARCHIVE_URL = "https://openfold.s3.amazonaws.com/pdb_mmcif.zip"
THRESHOLDS = {
    "projection_seq_identity": 0.90, "projection_alignment_coverage": 0.70,
    "projection_aligned_fraction": 0.70, "projection_valid_ca_count": 32,
}
REPO_ROOT = Path(__file__).resolve().parents[1]
PREDICATE_FILES = (
    "scripts/preprocess.py", "scripts/audit_target_identity.py", "scripts/audit_cache_targets.py",
    "nanofold/mmcif.py", "nanofold/residue_constants.py", "nanofold/a3m.py",
    "nanofold/structure_metadata.py", "nanofold/chain_paths.py",
    "nanofold/representative_eligibility.py", "scripts/build_representative_eligibility.py",
    "scripts/build_homology_groups.py",
)
STRUCTURE_REJECTIONS = (
    "Ambiguous author chain ID", "Ambiguous label chain ID", "Ambiguous sequence mapping",
    "contains multiple polymer label chains", "Polymer label_seq_id must be positive",
    "No polymer residues found", "Ambiguous observed residues: duplicate polymer label_seq_id",
    "Ambiguous sequence alignment: coordinate-to-query correspondence is not unique",
    "Observed label_seq_id lies outside the deposited polymer sequence",
    "Observed residue identity disagrees with its deposited polymer position",
    "Could not align structure sequence to query sequence",
)
RAW_QUERY_REJECTION = "A3M ungapped query differs from the pinned cache target"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _unique_object(items: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in items:
        require(key not in result, "Duplicate JSON key in eligibility evidence")
        result[key] = value
    return result


def _reject_constant(value: str) -> Any:
    raise ValueError("Nonfinite JSON value in eligibility evidence")


def read_json(path: Path) -> dict[str, Any]:
    result = json.loads(path.read_text(), object_pairs_hook=_unique_object, parse_constant=_reject_constant)
    require(isinstance(result, dict), "Eligibility evidence must be a JSON object")
    return result


def snapshot(path: Path) -> tuple[int, int, int, int, int]:
    stat = path.stat()
    require(path.is_file(), "Eligibility source must be a regular file")
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns


def sha256_file(path: Path) -> str:
    before = snapshot(path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    require(snapshot(path) == before, "Eligibility source changed while hashing")
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, str]:
    return {"path": str(path.resolve()), "sha256": sha256_file(path)}


def json_digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def assignment_digest(mapping: dict[str, str]) -> str:
    return hashlib.sha256("".join(f"{chain}\t{group}\n" for chain, group in sorted(mapping.items())).encode()).hexdigest()


def predicate_record() -> dict[str, Any]:
    return {
        "source_policy": SOURCE_POLICY, "thresholds": dict(THRESHOLDS), "require_full_match": False,
        "code": {name: sha256_file(REPO_ROOT / name) for name in PREDICATE_FILES},
        "runtime": {"python": platform.python_version(),
                    **{name: importlib.metadata.version(name) for name in ("gemmi", "biopython", "numpy")}},
    }


def metadata_semantic_sha256(path: Path) -> str:
    # Use the graph proof's owning canonicalization, without importing Candidate
    # or loading a second statically named copy of a scripts module.
    return str(importlib.import_module("scripts.build_homology_groups").metadata_semantic_sha256(path))


def _raw_source(roots: list[Path], chain: str) -> Path:
    paths: set[Path] = set()
    for root in roots:
        for path in chain_data_dir(root / "roda_pdb", chain).rglob("uniref90_hits.a3m"):
            require(path.resolve().is_relative_to(root), "Raw A3M resolves outside its bound source root")
            if path.is_file():
                paths.add(path.resolve())
    require(bool(paths), "Required raw query source is missing; acquisition is incomplete")
    if len(paths) > 1:
        require(len({sha256_file(path) for path in paths}) == 1, "Required raw query has conflicting source bytes")
    return min(paths)


def raw_query_record(path: Path) -> dict[str, Any]:
    """Hash the entire file, parsing only its first query (not all MSA rows)."""
    before = snapshot(path)
    digest = hashlib.sha256()
    query: list[str] = []
    header = False
    with path.open("rb") as handle:
        for line in handle:
            digest.update(line)
            text = line.decode("utf-8").strip()
            if not text:
                continue
            if text.startswith(">"):
                require(bool(text[1:].strip()), "Raw query header is empty")
                if header:
                    break
                header = True
            else:
                require(header and re.fullmatch(r"[A-Za-z-]+", text) is not None, "Raw query has invalid syntax")
                query.append(text)
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    sequence = "".join(letter for letter in "".join(query) if not letter.islower() and letter != "-")
    require(header and bool(sequence), "Raw query is missing or empty")
    require(snapshot(path) == before, "Raw A3M changed during query qualification")
    return {"path": str(path.resolve()), "sha256": digest.hexdigest(), "bytes": before[2],
            "query_length": len(sequence), "query_sha256": hashlib.sha256(sequence.encode()).hexdigest(),
            "scope": "first_query_identity_only"}


def inspect_candidate(
    *, chain: str, sequence: str, rank: int, structure: dict[str, str],
    raw_roots: list[Path] | None = None, raw_path: Path | None = None,
) -> dict[str, Any]:
    path = Path(structure["path"])
    before = snapshot(path)
    require(sha256_file(path) == structure["sha256"], "Historical structure differs from its pinned acquisition")
    row: dict[str, Any] = {
        "chain_id": chain, "rank": rank, "sequence_sha256": hashlib.sha256(sequence.encode()).hexdigest(),
        "target_length": len(sequence), "structure": structure, "raw": None,
        "raw_status": "not_evaluated_structure_failed", "projection_statistics": None,
        "structure_rejection": None, "structure_status": "failed", "status": "failed",
    }
    try:
        pdb, chain_id = chain.split("_", 1)
        parsed = extract_chain_atoms(
            mmcif_path=path, pdb_id=pdb, chain_id=chain_id, expected_sequence=sequence, require_full_match=False,
        )
        projection = importlib.import_module("scripts.preprocess")._project_atom14_to_query
        _positions, _mask, statistics = projection(
            query_seq=sequence, structure_seq=parsed.sequence,
            structure_atom14_positions=parsed.atom14_positions, structure_atom14_mask=parsed.atom14_mask,
            polymer_sequence=parsed.polymer_sequence, polymer_residue_index=parsed.polymer_residue_index,
        )
        row["projection_statistics"] = statistics
        row["structure_status"] = "passed" if all(statistics[key] >= value for key, value in THRESHOLDS.items()) else "failed"
    except ValueError as exc:
        rejection = next((reason for reason in STRUCTURE_REJECTIONS if reason in str(exc)), None)
        if rejection is None:
            raise
        row["structure_rejection"] = rejection
    require(snapshot(path) == before, "Historical structure changed during qualification")
    if row["structure_status"] == "passed":
        raw = raw_query_record(raw_path if raw_path is not None else _raw_source(raw_roots or [], chain))
        row["raw"] = raw
        if raw["query_sha256"] == row["sequence_sha256"]:
            row.update(raw_status="passed", status="passed")
        else:
            row.update(raw_status="failed", raw_rejection=RAW_QUERY_REJECTION)
    return row


def _group_membership_binding(mapping: dict[str, str], groups: list[str]) -> dict[str, Any]:
    return {group: {"member_count": len(members),
                    "members_sha256": hashlib.sha256(("\n".join(members) + "\n").encode()).hexdigest()}
            for group in sorted(groups)
            for members in [sorted(chain for chain, assigned in mapping.items() if assigned == group)]}


def _read_group_map(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for line in path.read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        parts = line.split("\t")
        require(len(parts) == 2 and all(parts) and parts[1] not in result, "Invalid approval full-group map")
        result[parts[1]] = parts[0]
    require(bool(result), "Approval full-group map is empty")
    return result


def bind_approval_decision(
    decision_path: Path, *, split_group_tsv_path: Path, split_group_receipt_path: Path,
) -> dict[str, Any]:
    decision = read_json(decision_path)
    require("original_graph" not in decision, "Approval decision already has an original graph binding")
    mapping = _read_group_map(split_group_tsv_path)
    groups: Any = decision.get("excluded_group_ids")
    require(isinstance(groups, list) and all(isinstance(group, str) and group in set(mapping.values()) for group in groups),
            "Approved exclusion IDs must belong to the original full graph")
    require(decision.get("original_group_count") == len(set(mapping.values())), "Approval original group count differs")
    decision["decision_source"] = file_record(decision_path)
    decision["original_graph"] = {
        "split_group_tsv": file_record(split_group_tsv_path), "split_group_receipt": file_record(split_group_receipt_path),
        "assignments_sha256": assignment_digest(mapping), "excluded_groups": _group_membership_binding(mapping, groups),
    }
    return decision


def approved_exclusion_catalog(decision_path: Path) -> dict[str, Any]:
    decision = read_json(decision_path)
    groups: Any = decision.get("excluded_group_ids")
    require(isinstance(groups, list) and all(isinstance(group, str) and group for group in groups)
            and len(set(groups)) == len(groups), "Approval must name a unique explicit set of excluded groups")
    require(decision.get("thresholds") == THRESHOLDS and decision.get("source_policy") == SOURCE_POLICY,
            "Approval does not bind the strict historical-source predicate")
    require(decision.get("all_retained_groups_required") is True and decision.get("whole_group_split_ownership") is True,
            "Approval must preserve retained group coverage and complete group ownership")
    require(isinstance(decision.get("authorization"), str) and bool(decision["authorization"].strip()),
            "Group exclusions require explicit recorded authorization")
    original_count, retained_count = decision.get("original_group_count"), decision.get("retained_group_count")
    require(type(original_count) is int and type(retained_count) is int
            and original_count > 0 and retained_count == original_count - len(groups),
            "Approval must bind consistent original and retained group counts")
    graph = decision.get("original_graph", {})
    require(isinstance(graph, dict) and isinstance(decision.get("decision_source"), dict),
            "Exclusion approval must bind original graph membership")
    original = decision["decision_source"]
    require(file_record(Path(original["path"])) == original, "Original exclusion authorization changed")
    require(decision == bind_approval_decision(
        Path(original["path"]), split_group_tsv_path=Path(graph["split_group_tsv"]["path"]),
        split_group_receipt_path=Path(graph["split_group_receipt"]["path"]),
    ), "Derived approval differs from its authorization or original graph")
    return {
        "schema_version": 1, "policy": POLICY, "excluded_group_ids": sorted(groups),
        "source_policy": SOURCE_POLICY, "thresholds": dict(THRESHOLDS),
        "approval": file_record(decision_path),
        "original_group_count": original_count, "retained_group_count": retained_count,
        "original_graph": graph,
    }


def validate_exclusion_catalog(path: Path) -> list[str]:
    catalog = read_json(path)
    approval = catalog.get("approval", {})
    require(isinstance(approval, dict) and isinstance(approval.get("path"), str), "Missing group exclusion approval")
    expected = approved_exclusion_catalog(Path(approval["path"]))
    require(catalog == expected, "Group exclusion catalog differs from its bound approval")
    return list(catalog["excluded_group_ids"])


def historical_source_catalog(receipt_paths: list[Path]) -> tuple[dict[str, Any], dict[str, dict[str, str]]]:
    """Validate tracked acquisition receipts and index pinned member bytes by PDB."""
    require(bool(receipt_paths), "Historical acquisition receipts are required")
    source: dict[str, Any] = {}
    members: dict[str, dict[str, str]] = {}
    receipt_records: list[dict[str, str]] = []
    manifest_records: list[dict[str, str]] = []
    for path in sorted({path.resolve() for path in receipt_paths}):
        record = file_record(path)
        raw = read_json(path)
        require(raw.get("status") == "retrieved_unvalidated_structures" and raw.get("source_inputs_unchanged") is True
                and raw.get("scope") == "selective_archive_retrieval_only", "Historical acquisition is incomplete")
        pin = raw.get("archive_pin", {})
        require(raw.get("archive_url") == ARCHIVE_URL and isinstance(pin, dict)
                and type(pin.get("size")) is int and pin["size"] > 22
                and isinstance(pin.get("etag"), str) and re.fullmatch(r'"[\x21\x23-\x7e]+"', pin["etag"]) is not None
                and (pin.get("version_id") is None or isinstance(pin.get("version_id"), str)),
                "Historical acquisition must bind the immutable OpenProteinSet archive")
        directory = raw.get("directory", {})
        identity = {"archive_url": ARCHIVE_URL, "archive_pin": pin,
                    "central_directory_sha256": directory.get("central_directory_sha256")}
        require(isinstance(identity["central_directory_sha256"], str)
                and re.fullmatch(r"[a-f0-9]{64}", identity["central_directory_sha256"]) is not None,
                "Historical acquisition lacks a central-directory digest")
        require(not source or source == identity, "Historical acquisition receipts mix archive versions")
        source = identity
        inputs = raw.get("inputs", {})
        require(inputs.get("script_sha256") == sha256_file(REPO_ROOT / "scripts/fetch_openproteinset_mmcif.py")
                and inputs.get("manifest_validator_sha256") == sha256_file(REPO_ROOT / "nanofold/structure_metadata.py"),
                "Historical acquisition code differs from the tracked source validator")
        require(isinstance(inputs.get("manifest_path"), str), "Historical acquisition lacks its requested manifest")
        manifest = Path(inputs["manifest_path"])
        manifest_record = file_record(manifest)
        require(manifest_record["sha256"] == inputs.get("manifest_sha256"),
                "Historical acquisition request manifest changed")
        requested = [line.strip() for line in manifest.read_text().splitlines()
                     if line.strip() and not line.lstrip().startswith("#")]
        require(bool(requested) and len(requested) == len(set(requested)), "Historical acquisition request is invalid")
        requested_pdbs = {chain.split("_", 1)[0] for chain in requested}
        manifest_records.append(manifest_record)
        rows: Any = raw.get("members")
        require(type(raw.get("retrieved_structure_count")) is int and type(raw.get("requested_structure_count")) is int
                and isinstance(rows, list) and len(rows) == raw.get("retrieved_structure_count")
                == raw.get("requested_structure_count") and bool(rows), "Historical acquisition membership is incomplete")
        seen: set[str] = set()
        for row in rows:
            require(isinstance(row, dict) and row.get("crc32_verified") is True and row.get("local_header_verified") is True,
                    "Historical member lacks successful ZIP integrity checks")
            relative = row.get("output_path")
            require(isinstance(relative, str)
                    and re.fullmatch(r"pdb_data/mmcif_files/[a-z0-9]{4}\.cif", relative) is not None,
                    "Historical member output path is invalid")
            relative = str(relative)
            pdb = Path(relative).stem
            require(pdb not in seen, "Duplicate historical member in acquisition receipt")
            seen.add(pdb)
            member = row.get("archive_member", {})
            require(str(member.get("name", "")).lower() == f"mmcif_files/{pdb}.cif",
                    "Historical member identity differs from its output")
            asset_path = (path.parent / relative).resolve()
            require(asset_path.is_relative_to(path.parent), "Historical member resolves outside acquisition root")
            digest = row.get("output_sha256")
            require(isinstance(digest, str) and re.fullmatch(r"[a-f0-9]{64}", digest) is not None,
                    "Historical member digest is invalid")
            asset = {"path": str(asset_path), "sha256": digest, "retrieval_receipt_sha256": record["sha256"]}
            require(pdb not in members or members[pdb]["sha256"] == digest, "Conflicting bytes for historical structure")
            members.setdefault(pdb, asset)
        require(seen == requested_pdbs, "Historical acquisition did not retrieve exactly its requested structures")
        require(sha256_file(path) == record["sha256"], "Historical receipt changed during validation")
        receipt_records.append(record)
    return {**source, "retrieval_receipts": receipt_records, "requested_manifests": manifest_records}, members


def population_record(
    *, chain_data_cache_path: Path, structure_metadata_path: Path, cluster_map: dict[str, str],
    split_group_map: dict[str, str], split_group_receipt_path: Path, original_units: dict[str, list[str]], seed: int,
) -> dict[str, Any]:
    members = [chain for ranked in original_units.values() for chain in ranked]
    require(bool(members) and len(set(members)) == len(members) and set(members) == set(cluster_map) == set(split_group_map),
            "Original unit membership must partition the complete original candidate universe")
    for unit, ranked in original_units.items():
        require(bool(ranked) and unit == hashlib.sha256(("\n".join(sorted(ranked)) + "\n").encode()).hexdigest(),
                "Original unit identity does not match its complete membership")
        require(len({split_group_map[chain] for chain in ranked}) == 1, "Original unit crosses full groups")
    require(type(seed) is int, "Representative ranking seed must be an integer")
    return {
        "seed": seed, "chain_data_cache_sha256": sha256_file(chain_data_cache_path),
        "structure_metadata_semantic_sha256": metadata_semantic_sha256(structure_metadata_path),
        "cluster_assignments_sha256": assignment_digest(cluster_map),
        "split_group_assignments_sha256": assignment_digest(split_group_map),
        "split_group_receipt_sha256": sha256_file(split_group_receipt_path),
        "ranked_units_sha256": json_digest(original_units), "original_candidate_count": len(members),
        "original_unit_count": len(original_units), "original_group_count": len(set(split_group_map.values())),
    }


def _verify_file(record: dict[str, Any], verified: dict[str, str]) -> None:
    require(isinstance(record, dict) and isinstance(record.get("path"), str)
            and isinstance(record.get("sha256"), str), "Source record must contain a path and digest")
    path, expected = record["path"], record["sha256"]
    if path not in verified:
        verified[path] = sha256_file(Path(path))
    require(verified[path] == expected, "Audited source bytes differ from eligibility evidence")


def validate_outcome(row: dict[str, Any], *, historical_members: dict[str, dict[str, str]], verified: dict[str, str]) -> str:
    structure = row.get("structure", {})
    require(structure == historical_members.get(str(row.get("chain_id", "")).split("_", 1)[0]),
            "Candidate structure is not the bound historical archive member")
    _verify_file(structure, verified)
    state = row.get("structure_status")
    require(state in {"passed", "failed"}, "Candidate structure audit is incomplete")
    if row.get("structure_rejection") is not None:
        require(state == "failed" and row["structure_rejection"] in STRUCTURE_REJECTIONS
                and row.get("projection_statistics") is None, "Unrecognized definitive structure rejection")
    else:
        stats: Any = row.get("projection_statistics")
        require(isinstance(stats, dict) and set(stats) == set(THRESHOLDS), "Incomplete strict projection statistics")
        require(all(type(value) in (int, float) and math.isfinite(value) for value in stats.values()),
                "Projection statistics must be finite numeric values")
        require(all(0 <= stats[key] <= 1 for key in THRESHOLDS if key != "projection_valid_ca_count")
                and 0 <= stats["projection_valid_ca_count"] <= row["target_length"]
                and float(stats["projection_valid_ca_count"]).is_integer(), "Projection statistics are out of range")
        expected_state = "passed" if all(stats[key] >= limit for key, limit in THRESHOLDS.items()) else "failed"
        require(state == expected_state, "Structure outcome disagrees with the fixed strict thresholds")
    raw: Any = row.get("raw")
    if state == "failed":
        require(raw is None and row.get("raw_status") == "not_evaluated_structure_failed",
                "Failed structure outcome has inconsistent raw audit state")
        outcome = "failed"
    else:
        require(isinstance(raw, dict), "Passing structure requires an exact raw-query audit")
        _verify_file(raw, verified)
        if row.get("raw_status") == "failed":
            require(row.get("raw_rejection") == RAW_QUERY_REJECTION,
                    "Only a definitive raw query mismatch may reject a candidate")
            outcome = "failed"
        else:
            require(type(raw.get("query_length")) is int and raw.get("query_length") == row["target_length"]
                    and type(raw.get("bytes")) is int and raw["bytes"] > 0
                    and raw.get("query_sha256") == row["sequence_sha256"]
                    and raw.get("scope") == "first_query_identity_only"
                    and row.get("raw_status") == "passed" and row.get("raw_rejection") is None,
                    "Passing structure requires a complete exact raw-query audit")
            outcome = "passed"
    require(row.get("status") == outcome, "Candidate eligibility contradicts its source audits")
    return outcome


def validate_representative_eligibility_receipt(
    receipt_path: Path, *, approved_group_exclusions_path: Path, chain_data_cache_path: Path,
    structure_metadata_path: Path, cluster_map: dict[str, str], split_group_map: dict[str, str],
    split_group_receipt_path: Path, original_units: dict[str, list[str]], seed: int,
) -> dict[str, Any]:
    initial_inputs = {str(path.resolve()): sha256_file(path) for path in (
        receipt_path, approved_group_exclusions_path, chain_data_cache_path, structure_metadata_path,
        split_group_receipt_path,
    )}
    receipt_hash = sha256_file(receipt_path)
    receipt = read_json(receipt_path)
    require(type(receipt.get("schema_version")) is int and receipt["schema_version"] == 1
            and receipt.get("policy") == POLICY and receipt.get("status") == "complete",
            "Representative eligibility receipt is incomplete or has the wrong policy")
    population = population_record(
        chain_data_cache_path=chain_data_cache_path, structure_metadata_path=structure_metadata_path,
        cluster_map=cluster_map, split_group_map=split_group_map, split_group_receipt_path=split_group_receipt_path,
        original_units=original_units, seed=seed,
    )
    require(json_digest(receipt.get("population")) == json_digest(population),
            "Eligibility receipt differs from the original population or ranking")
    predicate = predicate_record()
    require(receipt.get("predicate") == predicate, "Eligibility predicate or owning source changed")
    excluded = validate_exclusion_catalog(approved_group_exclusions_path)
    exclusion_catalog = read_json(approved_group_exclusions_path)
    require(exclusion_catalog["original_group_count"] == population["original_group_count"],
            "Approved exclusions were authorized for a different full group universe")
    approved_graph = exclusion_catalog["original_graph"]
    require(approved_graph["assignments_sha256"] == population["split_group_assignments_sha256"]
            and approved_graph["split_group_receipt"]["sha256"] == population["split_group_receipt_sha256"]
            and approved_graph["excluded_groups"] == _group_membership_binding(split_group_map, excluded),
            "Approved exclusion scope differs from original graph membership")
    approval_hash = sha256_file(approved_group_exclusions_path)
    require(receipt.get("approved_group_exclusions_sha256") == approval_hash, "Eligibility exclusion approval differs")
    source = receipt.get("historical_sources", {})
    source_records = source.get("retrieval_receipts", [])
    require(isinstance(source_records, list) and bool(source_records), "Missing historical source receipts")
    expected_sources, historical_members = historical_source_catalog([Path(item["path"]) for item in source_records])
    require(source == expected_sources, "Historical acquisition receipts changed")
    cache = read_json(chain_data_cache_path)
    rows: Any = receipt.get("units")
    require(isinstance(rows, list) and len(rows) == len(original_units), "Eligibility receipt omits original units")
    selected: dict[str, str | None] = {}
    exhausted: list[str] = []
    passing_groups: set[str] = set()
    verified: dict[str, str] = {}
    original_metadata = receipt.get("original_structure_metadata", {})
    _verify_file(original_metadata, verified)
    require(metadata_semantic_sha256(Path(original_metadata["path"])) == population["structure_metadata_semantic_sha256"],
            "Original metadata provenance differs from the current semantic metadata")
    for unit in rows:
        unit_id = unit.get("unit_id")
        require(unit_id in original_units and unit_id not in selected, "Unknown or duplicate eligibility unit")
        ranked = original_units[unit_id]
        require(unit.get("ranked_members") == ranked, "Eligibility unit membership or rank order changed")
        group = split_group_map[ranked[0]]
        require(unit.get("full_group_id") == group, "Eligibility unit full group changed")
        outcomes = unit.get("outcomes")
        require(isinstance(outcomes, list) and 0 < len(outcomes) <= len(ranked), "Missing eligibility rank prefix")
        chosen: str | None = None
        for rank, outcome in enumerate(outcomes):
            chain = ranked[rank]
            require(outcome.get("chain_id") == chain and outcome.get("rank") == rank,
                    "Eligibility outcomes do not form the complete ranked prefix")
            metadata = cache.get(chain, {})
            sequences = {metadata[key].upper() for key in ("sequence", "seq", "seqres", "aatype_sequence")
                         if isinstance(metadata.get(key), str) and metadata[key]}
            require(len(sequences) == 1, "Pinned cache candidate sequence is missing or ambiguous")
            sequence = next(iter(sequences))
            require(outcome.get("target_length") == len(sequence)
                    and outcome.get("sequence_sha256") == hashlib.sha256(sequence.encode()).hexdigest(),
                    "Audited candidate target differs from the pinned cache")
            state = validate_outcome(outcome, historical_members=historical_members, verified=verified)
            actual = inspect_candidate(
                chain=chain, sequence=sequence, rank=rank, structure=historical_members[chain.split("_", 1)[0]],
                raw_path=Path(outcome["raw"]["path"]) if isinstance(outcome.get("raw"), dict) else None,
            )
            require(json_digest(actual) == json_digest(outcome),
                    "Eligibility outcome does not replay from the bound source bytes")
            if state == "passed":
                require(rank == len(outcomes) - 1, "Eligibility audit continues after the first passing rank")
                chosen = chain
        require(chosen is not None or len(outcomes) == len(ranked), "Failed unit lacks exhaustive member evidence")
        require(unit.get("selected_chain_id") == chosen, "Selected representative is not the first passing rank")
        selected[unit_id] = chosen
        if chosen is None:
            exhausted.append(unit_id)
        else:
            passing_groups.add(group)
    empty_groups = set(split_group_map.values()) - passing_groups
    require(empty_groups == set(excluded), "Approved exclusions must equal the exactly exhausted full groups")
    counts = {
        "eligible_unit_count": len(selected) - len(exhausted), "exhausted_unit_count": len(exhausted),
        "excluded_group_count": len(empty_groups), "retained_group_count": len(passing_groups),
    }
    require(json_digest(receipt.get("counts")) == json_digest(counts),
            "Eligibility counts disagree with complete unit evidence")
    for path, expected in {**initial_inputs, **verified}.items():
        require(sha256_file(Path(path)) == expected, "Eligibility input or source changed during validation")
    require(predicate_record() == predicate, "Eligibility predicate runtime or source changed during validation")
    require(validate_exclusion_catalog(approved_group_exclusions_path) == excluded,
            "Eligibility exclusion approval changed during validation")
    final_sources, _ = historical_source_catalog([Path(item["path"]) for item in source_records])
    require(final_sources == source, "Historical acquisition receipt changed during validation")
    binding = {
        "schema_version": 1, "policy": POLICY, "receipt_sha256": receipt_hash,
        "approved_group_exclusions_sha256": approval_hash, "source_policy": SOURCE_POLICY,
        **population, "predicate_sha256": json_digest(predicate), "historical_sources_sha256": json_digest(source), **counts,
    }
    source_paths = set(initial_inputs) | set(verified) | {str(REPO_ROOT / name) for name in PREDICATE_FILES}
    source_paths.update(item["path"] for item in source_records)
    source_paths.update(item["path"] for item in source["requested_manifests"])
    source_paths.update(approved_graph[key]["path"] for key in ("split_group_tsv", "split_group_receipt"))
    source_paths.add(exclusion_catalog["approval"]["path"])
    source_paths.add(read_json(Path(exclusion_catalog["approval"]["path"]))["decision_source"]["path"])
    source_paths.add(str(REPO_ROOT / "scripts/fetch_openproteinset_mmcif.py"))
    return {"selected_chain_by_original_unit": selected, "exhausted_original_units": sorted(exhausted),
            "approved_excluded_group_ids": sorted(excluded), "binding": binding, **counts,
            "validated_source_paths": sorted(source_paths)}
