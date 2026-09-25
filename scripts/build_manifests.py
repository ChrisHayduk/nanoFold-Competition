from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
from bisect import bisect_left
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from nanofold.structure_metadata import validate_author_chain_id

"""Create train/val/hidden manifests from an OpenFold/OpenProteinSet chain_data_cache.

Maintainer-only workflow:
- use this script to generate/refresh official manifests and lock metadata.
- participants should use committed manifests via scripts/setup_official_data.sh.

Expected input:
- chain_data_cache.json (available from RODA as described in OpenFold docs)

We filter by:
- single-chain prediction targets (biological monomers only with --require-monomer)
- length range
- resolution cutoff
and then sample fixed-size split chain sets with leakage-aware grouping:
- no sequence cluster is allowed to appear in multiple splits
- no PDB entry is allowed to appear in multiple splits
- CATH architecture compositions, then length and resolution, receive proportional quotas

The chain_data_cache parser accepts the field names present in OpenFold's
published cache metadata and requires source-bound versioned structural metadata.
"""

STANDARD_AAS = set("ARNDCQEGHILKMFPSTWYV")
SAMPLING_POLICY = "cath-v2"
GROUP_SAMPLING_POLICY = "cath-v3"
PROCESSABLE_GROUP_SAMPLING_POLICY = "cath-v4"
GROUP_SAMPLING_POLICIES = (GROUP_SAMPLING_POLICY, PROCESSABLE_GROUP_SAMPLING_POLICY)
CATH_SAMPLING_POLICIES = (SAMPLING_POLICY, *GROUP_SAMPLING_POLICIES)
GROUP_ALLOCATOR_POLICY = "joint_heldout_groups_mandatory_train_v1"
DEFAULT_STRATIFY_FIELDS = ("cath_architecture_ids", "length_bin", "resolution_bin")
LEGACY_STRATIFY_FIELDS = (
    "secondary_structure_class",
    "domain_architecture_class",
    "length_bin",
    "resolution_bin",
)
MAX_SPLIT_JS_DIVERGENCE = 0.35
MAX_UNKNOWN_DOMAIN_ARCHITECTURE_FRACTION = 0.75


@dataclass(frozen=True)
class Candidate:
    chain_id: str
    sequence: str
    length: int
    resolution: float
    release_date: str | None
    cluster_id: str
    pdb_id: str
    secondary_structure_class: str
    secondary_helix_fraction: float | None
    secondary_beta_fraction: float | None
    secondary_coil_fraction: float | None
    domain_architecture_class: str
    domain_architecture_source: str
    metadata_source_count: int
    cath_architecture_ids: tuple[str, ...] = ()
    cath_topology_ids: tuple[str, ...] = ()
    cath_homologous_superfamily_ids: tuple[str, ...] = ()
    cath_annotation_status: str = "unknown"
    cath_domain_count: int = 0


@dataclass(frozen=True)
class SplitResult:
    train_ids: list[str]
    val_ids: list[str]
    hidden_val_ids: list[str]
    cluster_map: dict[str, str]
    selected_candidates: dict[str, Candidate]
    candidate_count: int
    unit_count: int
    cluster_method: str
    cluster_command: list[str] | None
    structural_metadata_source: str
    group_policy: dict[str, bool]
    quality_report: dict[str, Any]
    split_group_map: dict[str, str] | None = None
    approved_excluded_group_ids: frozenset[str] = frozenset()


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sampling-policy", choices=CATH_SAMPLING_POLICIES, required=True,
                    help="Versioned CATH sampling: cath-v4 covers every retained full group with source-validated representatives.")
    ap.add_argument("--split-group-tsv", default="",
                    help="Required for cath-v3/v4: complete original eligible-chain full-group TSV, nesting original clusters, PDBs and exact sequences.")
    ap.add_argument("--split-group-receipt", default="",
                    help="Required for cath-v3/v4: validated original full-universe homology graph provenance receipt.")
    ap.add_argument("--representative-eligibility-receipt", default="",
                    help="Required for cath-v4: source-validated ranked representative eligibility for every original unit.")
    ap.add_argument("--approved-group-exclusions", default="",
                    help="Required for cath-v4: explicit approved catalog of fully exhausted original full groups.")
    ap.add_argument("--chain-data-cache", type=str, required=True, help="Path to chain_data_cache.json")
    ap.add_argument("--out-dir", type=str, required=True, help="Where to write public train.txt, val.txt, and all.txt")
    ap.add_argument(
        "--hidden-out-dir",
        type=str,
        default=".nanofold_private/manifests",
        help=(
            "Maintainer-only directory for hidden_val.txt and the full split quality report. "
            "Keep this path outside tracked public files."
        ),
    )
    ap.add_argument("--train-size", type=int, default=10000)
    ap.add_argument("--val-size", type=int, default=1000)
    ap.add_argument(
        "--hidden-val-size",
        type=int,
        default=1000,
        help=(
            "Hidden validation size. Official generation must keep this split cluster/PDB-disjoint "
            "from train and public val."
        ),
    )
    ap.add_argument("--min-len", type=int, default=40)
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--max-resolution", type=float, default=3.0)
    ap.add_argument("--require-monomer", action="store_true",
                    help="Require explicit biological oligomeric_count=1; absent state is unknown.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--hidden-split-salt-env",
        type=str,
        default="NANOFOLD_HIDDEN_SPLIT_SALT",
        help=(
            "Environment variable containing the maintainer-only hidden split salt. "
            "Required when --hidden-val-size is positive."
        ),
    )
    ap.add_argument(
        "--hidden-split-salt-file",
        type=str,
        default="",
        help=(
            "Optional private file containing the hidden split salt. Prefer the environment "
            "variable in automation so the salt path does not become public metadata."
        ),
    )
    ap.add_argument("--min-seq-id", type=float, default=0.30)
    ap.add_argument("--coverage", type=float, default=0.80)
    ap.add_argument("--mmseqs-bin", type=str, default="mmseqs")
    ap.add_argument(
        "--cluster-tsv",
        type=str,
        default="",
        help=(
            "Precomputed cluster TSV in MMseqs2 representative<TAB>member format. If omitted, "
            "--mmseqs-bin is run with the official identity/coverage parameters."
        ),
    )
    ap.add_argument(
        "--structure-metadata",
        type=str,
        required=True,
        help=(
            "Canonical structure metadata JSON from scripts/build_structure_metadata.py or "
            "scripts/filter_openproteinset.py. Schema 2 preserves CATH domains and measured secondary-structure missingness."
        ),
    )
    ap.add_argument(
        "--expected-chain-cache-sha256",
        type=str,
        default="",
        help=(
            "Optional expected SHA256 of chain_data_cache.json. "
            "If provided, generation fails on mismatch."
        ),
    )
    ap.add_argument(
        "--lock-file",
        type=str,
        required=True,
        help="Required public JSON path capturing non-hidden generation inputs/hashes.",
    )
    ap.add_argument(
        "--private-lock-file",
        type=str,
        required=True,
        help="Required maintainer-only JSON path capturing hidden split metadata and salt digest.",
    )
    return ap.parse_args()


def _pdb_id(chain_id: str) -> str:
    if "_" not in chain_id:
        raise ValueError(f"Invalid chain id (expected <pdb>_<chain>): {chain_id}")
    return chain_id.split("_", 1)[0].lower()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _extract_sequence(meta: Dict[str, object]) -> str | None:
    for key in ("sequence", "seq", "seqres", "aatype_sequence"):
        value = meta.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip().upper()
    return None


def _extract_release_date(meta: Dict[str, object]) -> str | None:
    for key in ("release_date", "pdb_release_date", "deposition_date"):
        value = meta.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _unknown_aa_fraction(sequence: str) -> float:
    if not sequence:
        return 1.0
    unknown = sum(1 for ch in sequence.upper() if ch not in STANDARD_AAS)
    return unknown / float(len(sequence))


def _length_bin(length: int) -> str:
    if length < 64:
        return "len_040_063"
    if length < 96:
        return "len_064_095"
    if length < 128:
        return "len_096_127"
    if length < 192:
        return "len_128_191"
    return "len_192_256"


def _resolution_bin(resolution: float) -> str:
    if not resolution or resolution >= 999.0:
        return "res_unknown"
    if resolution <= 1.5:
        return "res_le_1p5"
    if resolution <= 2.0:
        return "res_1p5_2p0"
    if resolution <= 2.5:
        return "res_2p0_2p5"
    return "res_2p5_3p0"


def _candidate_field(candidate: Candidate, field_name: str) -> str:
    if field_name == "length_bin":
        return _length_bin(candidate.length)
    if field_name == "resolution_bin":
        return _resolution_bin(candidate.resolution)
    if field_name == "metadata_source_count_bin":
        if candidate.metadata_source_count >= 4:
            return "sources_ge_4"
        if candidate.metadata_source_count >= 2:
            return "sources_2_3"
        return "sources_0_1"
    if field_name.startswith("cath_") and field_name.endswith("_ids"):
        return "+".join(getattr(candidate, field_name)) or "unknown"
    value = getattr(candidate, field_name, None)
    if value is None:
        return "unknown"
    return str(value)


def _stable_random_score(seed: int, salt: str, chain_id: str) -> int:
    payload = f"{seed}:{salt}:{chain_id}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], byteorder="big", signed=False)


def _read_hidden_split_salt(*, salt_env: str, salt_file: str) -> str:
    if salt_file:
        salt = Path(salt_file).read_text().strip()
        source = f"file `{salt_file}`"
    else:
        salt = os.environ.get(salt_env, "").strip()
        source = f"environment variable `{salt_env}`"
    if not salt:
        raise RuntimeError(
            "Hidden split generation requires a maintainer-only salt. "
            f"Set {source} before generating official hidden manifests."
        )
    if len(salt) < 32:
        raise RuntimeError(
            "Hidden split salt must be at least 32 characters so it cannot be guessed from public metadata."
        )
    return salt


def _salt_digest(salt: str) -> str:
    return hashlib.sha256(salt.encode("utf-8")).hexdigest()


def _stratum(candidate: Candidate, fields: tuple[str, ...]) -> tuple[str, ...]:
    if not fields:
        return ("all",)
    return tuple(_candidate_field(candidate, field_name) for field_name in fields)


def _normalize_secondary_class(value: object) -> str:
    text = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "alpha": "alpha",
        "all_alpha": "alpha",
        "mainly_alpha": "alpha",
        "helix": "alpha",
        "beta": "beta",
        "all_beta": "beta",
        "mainly_beta": "beta",
        "sheet": "beta",
        "strand": "beta",
        "alpha_beta": "alpha_beta",
        "mixed_alpha_beta": "alpha_beta",
        "alpha/beta": "alpha_beta",
        "alpha+beta": "alpha_beta",
        "mixed": "alpha_beta",
        "coil": "coil_or_sparse",
        "few_secondary_structures": "coil_or_sparse",
        "little_secondary_structure": "coil_or_sparse",
    }
    return aliases.get(text, text if text else "unknown")


def _secondary_class_from_fractions(
    helix_fraction: float | None,
    beta_fraction: float | None,
    coil_fraction: float | None,
) -> str:
    if helix_fraction is None or beta_fraction is None:
        return "unknown"
    helix = float(helix_fraction)
    beta = float(beta_fraction)
    if helix >= 0.40 and beta < 0.20:
        return "alpha"
    if beta >= 0.30 and helix < 0.20:
        return "beta"
    if helix >= 0.20 and beta >= 0.15:
        return "alpha_beta"
    if coil_fraction is not None and float(coil_fraction) >= 0.70:
        return "coil_or_sparse"
    return "mixed_low_confidence"


def _parse_secondary_payload(payload: dict[str, Any]) -> tuple[str, float | None, float | None, float | None]:
    cls = payload.get("secondary_structure_class", payload.get("structural_class"))
    helix = payload.get("secondary_helix_fraction", payload.get("helix_fraction"))
    beta = payload.get("secondary_beta_fraction", payload.get("beta_fraction"))
    coil = payload.get("secondary_coil_fraction", payload.get("coil_fraction"))
    helix_f = _to_optional_float(helix)
    beta_f = _to_optional_float(beta)
    coil_f = _to_optional_float(coil)
    if cls is None:
        cls = _secondary_class_from_fractions(helix_f, beta_f, coil_f)
    return _normalize_secondary_class(cls), helix_f, beta_f, coil_f


def _parse_domain_payload(payload: dict[str, Any]) -> tuple[str, str, int]:
    cls = str(payload.get("domain_architecture_class") or "").strip().lower()
    source = str(payload.get("domain_architecture_source") or "").strip() or "unknown"
    source_count_raw = payload.get("metadata_source_count", 0)
    try:
        source_count = int(source_count_raw) if isinstance(source_count_raw, (str, bytes, bytearray, int, float)) else 0
    except (TypeError, ValueError):
        source_count = 0
    return cls or "unknown", source, max(0, source_count)


def _to_optional_float(value: object) -> float | None:
    if value is None:
        return None
    if not isinstance(value, (str, bytes, bytearray, int, float)):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(out) or out < 0.0 or isinstance(value, bool):
        return None
    return out


def _load_structure_metadata(
    path: str | Path, *, require_schema_version: int | None = None,
) -> tuple[set[str], dict[str, dict[str, Any]]]:
    metadata_path = Path(path)
    raw = json.loads(metadata_path.read_text())
    if not isinstance(raw, dict) or not isinstance(raw.get("chains"), list):
        raise ValueError(f"Structure metadata must contain a `chains` list: {metadata_path}")
    version = raw.get("schema_version", 1)
    if version not in (1, 2) or (require_schema_version is not None and version != require_schema_version):
        raise ValueError(f"Structure metadata requires schema {require_schema_version or '1 or 2'}: {metadata_path}")
    accepted: set[str] = set()
    metadata: dict[str, dict[str, Any]] = {}
    for item in raw["chains"]:
        if not isinstance(item, dict) or not isinstance(item.get("chain_id"), str):
            raise ValueError("Structure metadata rows must contain a chain_id string.")
        chain_id = item["chain_id"].strip()
        if not chain_id or chain_id in metadata:
            raise ValueError(f"Missing or duplicate structure metadata chain: {chain_id}")
        if not isinstance(item.get("accepted"), bool):
            raise ValueError(f"Structure metadata accepted must be boolean: {chain_id}")
        metadata[chain_id] = item
        if not item["accepted"]:
            continue
        cls, helix_f, beta_f, coil_f = _parse_secondary_payload(item)
        fractions = (helix_f, beta_f, coil_f)
        if version == 1:
            if cls == "unknown" or any(value is None for value in fractions):
                raise ValueError(f"Accepted schema 1 metadata requires secondary structure: {chain_id}")
        else:
            raw_fractions = [item.get(f"secondary_{name}_fraction") for name in ("helix", "beta", "coil")]
            if any(value is not None for value in raw_fractions):
                if any(value is None or value > 1 for value in fractions):
                    raise ValueError(f"Invalid secondary structure fractions: {chain_id}")
                if not math.isclose(sum(value for value in fractions if value is not None), 1.0, abs_tol=0.02):
                    raise ValueError(f"Secondary structure fractions must sum to one: {chain_id}")
            elif cls != "unknown":
                raise ValueError(f"Secondary structure class requires measured fractions: {chain_id}")
            _parse_cath_payload(item)
        accepted.add(chain_id)
    if not accepted:
        raise ValueError(f"Structure metadata has no accepted chains: {metadata_path}")
    return accepted, metadata


def _parse_cath_payload(payload: dict[str, Any]) -> dict[str, Any]:
    values: dict[str, Any] = {}
    for field, levels in (("cath_architecture_ids", 2), ("cath_topology_ids", 3),
                          ("cath_homologous_superfamily_ids", 4)):
        codes = payload.get(field, [])
        if not isinstance(codes, list) or any(
            not isinstance(code, str) or len(code.split(".")) != levels
            or any(not part.isdigit() or int(part) <= 0 for part in code.split("."))
            for code in codes
        ):
            raise ValueError(f"Invalid {field} for {payload.get('chain_id')}")
        values[field] = tuple(sorted(set(codes)))
    status = payload.get("cath_annotation_status", "unknown")
    if status not in ("annotated", "unknown"):
        raise ValueError("Invalid cath_annotation_status")
    if (status == "annotated") != bool(values["cath_architecture_ids"]):
        raise ValueError("CATH annotation status disagrees with architecture IDs")
    if {".".join(code.split(".")[:2]) for code in values["cath_topology_ids"]} != set(values["cath_architecture_ids"]):
        raise ValueError("CATH topology IDs disagree with architecture IDs")
    if {".".join(code.split(".")[:3]) for code in values["cath_homologous_superfamily_ids"]} != set(values["cath_topology_ids"]):
        raise ValueError("CATH superfamily IDs disagree with topology IDs")
    domains = payload.get("cath_domains", [])
    if not isinstance(domains, list) or any(not isinstance(domain, dict) for domain in domains):
        raise ValueError("Invalid cath_domains")
    for field, domain_field in (("cath_architecture_ids", "architecture_id"), ("cath_topology_ids", "topology_id"),
                                ("cath_homologous_superfamily_ids", "homologous_superfamily_id")):
        if {domain.get(domain_field) for domain in domains} != set(values[field]):
            raise ValueError(f"CATH domains disagree with {field}")
    domain_ids = [domain.get("domain_id") for domain in domains]
    if any(not isinstance(domain_id, str) or not domain_id for domain_id in domain_ids) or len(set(domain_ids)) != len(domain_ids):
        raise ValueError("Invalid or duplicate CATH domain IDs")
    values["cath_annotation_status"] = status
    values["cath_domain_count"] = len(domains)
    return values


def _structure_metadata_source_record(path: str | Path) -> dict[str, Any]:
    metadata_path = Path(path)
    record: dict[str, Any] = {
        "path": str(metadata_path),
        "sha256": _sha256(metadata_path),
    }
    try:
        raw = json.loads(metadata_path.read_text())
    except json.JSONDecodeError:
        return record
    if isinstance(raw, dict):
        source = raw.get("source")
        if isinstance(source, dict):
            record["source"] = source
    return record


def _load_cluster_tsv(path: str | Path, candidates: set[str]) -> Dict[str, str]:
    cluster_path = Path(path)
    cluster_map: Dict[str, str] = {}
    with cluster_path.open() as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 2 or any(not part.strip() for part in parts):
                raise ValueError(f"Malformed cluster TSV row {line_number}")
            representative, member = (part.strip() for part in parts)
            if member in cluster_map and cluster_map[member] != representative:
                raise ValueError(f"Conflicting cluster assignments for {member}")
            cluster_map[member] = representative
    for representative in set(cluster_map.values()):
        if representative in cluster_map and cluster_map[representative] != representative:
            raise ValueError(f"Cluster representative is assigned to a different cluster: {representative}")
    cluster_map = {member: representative for member, representative in cluster_map.items() if member in candidates}
    missing = sorted(candidates - set(cluster_map))
    if missing:
        sample = ", ".join(missing[:8])
        raise ValueError(
            f"Cluster TSV does not cover {len(missing)} candidate chains. Examples: {sample}"
        )
    return cluster_map


def _cluster_sequences_mmseqs(
    *,
    sequences: Dict[str, str],
    mmseqs_bin: str,
    min_seq_id: float,
    coverage: float,
    alignment_mode: int | None = 3,
) -> tuple[Dict[str, str] | None, List[str] | None]:
    mmseqs_path = shutil.which(mmseqs_bin)
    if mmseqs_path is None:
        return None, None

    with tempfile.TemporaryDirectory(prefix="nanofold_mmseqs_") as tmp_str:
        tmp_dir = Path(tmp_str)
        fasta_path = tmp_dir / "chains.fasta"
        fasta_lines: List[str] = []
        for chain_id, seq in sorted(sequences.items()):
            fasta_lines.append(f">{chain_id}")
            fasta_lines.append(seq)
        fasta_path.write_text("\n".join(fasta_lines) + "\n")

        cluster_prefix = tmp_dir / "clusters"
        work_dir = tmp_dir / "work"
        actual_cmd = [
            mmseqs_path,
            "easy-cluster",
            str(fasta_path),
            str(cluster_prefix),
            str(work_dir),
            "--min-seq-id",
            str(min_seq_id),
            "-c",
            str(coverage),
            "--cov-mode",
            "0",
            *([] if alignment_mode is None else ["--alignment-mode", str(alignment_mode)]),
            *([] if alignment_mode is None else ["--dbtype", "1"]),
            "--threads",
            "1",
        ]
        lock_cmd = [
            mmseqs_bin,
            "easy-cluster",
            "<chains_fasta>",
            "<cluster_prefix>",
            "<work_dir>",
            "--min-seq-id",
            str(min_seq_id),
            "-c",
            str(coverage),
            "--cov-mode",
            "0",
            *([] if alignment_mode is None else ["--alignment-mode", str(alignment_mode)]),
            *([] if alignment_mode is None else ["--dbtype", "1"]),
            "--threads",
            "1",
        ]
        proc = subprocess.run(actual_cmd, capture_output=True, text=True, check=False)
        if proc.returncode != 0:
            return None, lock_cmd

        cluster_tsv = Path(str(cluster_prefix) + "_cluster.tsv")
        if not cluster_tsv.exists():
            return None, lock_cmd

        return _load_cluster_tsv(cluster_tsv, set(sequences)), lock_cmd


def _mmseqs_version(mmseqs_bin: str) -> str | None:
    proc = subprocess.run([mmseqs_bin, "version"], capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        return None
    return proc.stdout.strip() or None


def _cluster_assignments_sha256(cluster_map: Dict[str, str]) -> str:
    hasher = hashlib.sha256()
    for member, representative in sorted(cluster_map.items()):
        hasher.update(member.encode("utf-8"))
        hasher.update(b"\t")
        hasher.update(representative.encode("utf-8"))
        hasher.update(b"\n")
    return hasher.hexdigest()


def _build_candidates(
    data: dict[str, Any],
    *,
    min_len: int,
    max_len: int,
    max_resolution: float,
    structure_metadata: dict[str, dict[str, Any]],
    structure_metadata_accepted: set[str],
    sampling_policy: str = SAMPLING_POLICY,
    require_monomer: bool = False,
) -> tuple[list[Candidate], Counter[str], str]:
    candidates: list[Candidate] = []
    reject_reasons: Counter[str] = Counter()
    secondary_source = "structure_metadata"

    for chain_id, meta_raw in sorted(data.items()):
        if sampling_policy in CATH_SAMPLING_POLICIES:
            validate_author_chain_id(chain_id)
        elif not re.fullmatch(r"[0-9A-Za-z]{4}_[^\s_]+", chain_id):
            reject_reasons["invalid_chain_id"] += 1
            continue
        if chain_id not in structure_metadata_accepted:
            reject_reasons["not_accepted_in_structure_metadata"] += 1
            continue
        if not isinstance(meta_raw, dict):
            reject_reasons["bad_metadata"] += 1
            continue
        meta: Dict[str, object] = meta_raw
        seq = _extract_sequence(meta)
        if not seq:
            reject_reasons["missing_sequence"] += 1
            continue
        seq = seq.upper()
        if sampling_policy in CATH_SAMPLING_POLICIES and _unknown_aa_fraction(seq) != 0:
            reject_reasons["nonstandard_amino_acids"] += 1
            continue

        raw_length = meta.get("seq_length", meta.get("sequence_length", len(seq)))
        raw_resolution = meta.get("resolution")
        oligomer_count = meta.get("oligomeric_count")
        if sampling_policy in CATH_SAMPLING_POLICIES:
            length_value = _to_optional_float(raw_length)
            res_value = _to_optional_float(raw_resolution)
            olig_value = _to_optional_float(oligomer_count)
            if length_value is None or not length_value.is_integer() or length_value != len(seq):
                reject_reasons["invalid_or_mismatched_sequence_length"] += 1
                continue
            if res_value is None or res_value <= 0:
                res_value = 999.0
            if require_monomer and (olig_value is None or not olig_value.is_integer()):
                reject_reasons["missing_or_invalid_oligomeric_count"] += 1
                continue
            L, res, olig = int(length_value), res_value, olig_value
        else:
            L = int(_to_optional_float(raw_length) or len(seq))
            res = _to_optional_float(raw_resolution) or 999.0
            olig = int(_to_optional_float(oligomer_count) or 1)
        if not (min_len <= L <= max_len):
            reject_reasons["length"] += 1
            continue
        if res != 999.0 and res > max_resolution:
            reject_reasons["resolution"] += 1
            continue
        if (require_monomer or sampling_policy not in CATH_SAMPLING_POLICIES) and olig != 1:
            reject_reasons["oligomeric_count"] += 1
            continue

        metadata_entry = structure_metadata[chain_id]
        ss_class, helix_fraction, beta_fraction, coil_fraction = _parse_secondary_payload(metadata_entry)
        domain_class, domain_source, metadata_source_count = _parse_domain_payload(metadata_entry)
        if sampling_policy not in CATH_SAMPLING_POLICIES and (
            ss_class == "unknown" or any(value is None for value in (helix_fraction, beta_fraction, coil_fraction))
            or domain_class == "unknown"
        ):
            reject_reasons["missing_structure_metadata"] += 1
            continue
        cath = _parse_cath_payload(metadata_entry) if sampling_policy in CATH_SAMPLING_POLICIES else {}

        candidates.append(
            Candidate(
                chain_id=chain_id,
                sequence=seq,
                length=L,
                resolution=res,
                release_date=_extract_release_date(meta),
                cluster_id=chain_id,
                pdb_id=_pdb_id(chain_id),
                secondary_structure_class=ss_class,
                secondary_helix_fraction=helix_fraction,
                secondary_beta_fraction=beta_fraction,
                secondary_coil_fraction=coil_fraction,
                domain_architecture_class=domain_class,
                domain_architecture_source=domain_source,
                metadata_source_count=metadata_source_count,
                **cath,
            )
        )
    return candidates, reject_reasons, secondary_source


def _representative_rank(candidate: Candidate, seed: int) -> tuple[float, int, str, int, str]:
    return (
        candidate.resolution if candidate.resolution and candidate.resolution < 999.0 else 999.0,
        -candidate.length,
        candidate.release_date or "",
        _stable_random_score(seed, "representative", candidate.chain_id),
        candidate.chain_id,
    )


def _choose_representative(candidates: list[Candidate], seed: int) -> Candidate:
    return min(candidates, key=lambda candidate: _representative_rank(candidate, seed))


def _disjoint_components(
    candidates: list[Candidate],
    cluster_map: dict[str, str],
    *,
    disallow_pdb_overlap: bool,
) -> list[list[Candidate]]:
    chain_ids = sorted(candidate.chain_id for candidate in candidates)
    parent = {chain_id: chain_id for chain_id in chain_ids}

    def find(chain_id: str) -> str:
        while parent[chain_id] != chain_id:
            parent[chain_id] = parent[parent[chain_id]]
            chain_id = parent[chain_id]
        return chain_id

    def union(a: str, b: str) -> None:
        ra = find(a)
        rb = find(b)
        if ra != rb:
            parent[rb] = ra

    by_cluster: defaultdict[str, list[str]] = defaultdict(list)
    by_pdb: defaultdict[str, list[str]] = defaultdict(list)
    by_sequence: defaultdict[str, list[str]] = defaultdict(list)
    for candidate in candidates:
        by_cluster[cluster_map[candidate.chain_id]].append(candidate.chain_id)
        by_pdb[candidate.pdb_id].append(candidate.chain_id)
        by_sequence[candidate.sequence].append(candidate.chain_id)
    for cluster_members in list(by_cluster.values()) + list(by_sequence.values()):
        first = cluster_members[0]
        for member in cluster_members[1:]:
            union(first, member)
    if disallow_pdb_overlap:
        for pdb_members in by_pdb.values():
            first = pdb_members[0]
            for member in pdb_members[1:]:
                union(first, member)

    by_component: defaultdict[str, list[Candidate]] = defaultdict(list)
    by_id = {candidate.chain_id: candidate for candidate in candidates}
    for chain_id in chain_ids:
        by_component[find(chain_id)].append(by_id[chain_id])

    return list(by_component.values())


def _disjoint_units(
    candidates: list[Candidate], cluster_map: dict[str, str], *, disallow_pdb_overlap: bool, seed: int,
) -> list[Candidate]:
    return [_choose_representative(members, seed) for members in _disjoint_components(
        candidates, cluster_map, disallow_pdb_overlap=disallow_pdb_overlap,
    )]


def _original_unit_memberships(
    candidates: list[Candidate], cluster_map: dict[str, str], *, seed: int,
) -> dict[str, list[Candidate]]:
    """Return the unchanged original components and their deterministic candidate order."""
    ids = {candidate.chain_id for candidate in candidates}
    if len(ids) != len(candidates) or set(cluster_map) != ids:
        raise ValueError("Original unit memberships require the complete eligible candidate universe exactly.")
    units = {}
    for members in _disjoint_components(candidates, cluster_map, disallow_pdb_overlap=True):
        unit_id = hashlib.sha256("".join(chain + "\n" for chain in sorted(
            candidate.chain_id for candidate in members
        )).encode()).hexdigest()
        units[unit_id] = sorted(members, key=lambda candidate: _representative_rank(candidate, seed))
    return dict(sorted(units.items()))


def _apportion_counts(
    capacities: dict[str, int], weights: dict[str, int], *, size: int, seed: int, salt: str,
) -> dict[str, int]:
    """Capped proportional quotas with seeded largest-remainder rounding.

    Saturated groups are removed and their quota redistributed proportionally.
    If no reference-supported group remains, residual capacity supplies the weights.
    """
    if size < 0 or size > sum(capacities.values()):
        raise ValueError("Requested sample size exceeds available capacity or is negative.")
    allocation = dict.fromkeys(capacities, 0)
    available = {key: value for key, value in capacities.items() if value > 0}
    remaining = size
    while remaining and available:
        active_weights = {key: weights.get(key, 0) for key in available}
        if not sum(active_weights.values()):
            active_weights = dict(available)
        total_weight = sum(active_weights.values())
        numerators = {key: remaining * active_weights[key] for key in available}
        saturated = [key for key in available if numerators[key] >= available[key] * total_weight]
        if saturated:
            for key in saturated:
                allocation[key] += available[key]
                remaining -= available.pop(key)
            continue
        quotas = {key: divmod(value, total_weight) for key, value in numerators.items()}
        floors = {key: quota[0] for key, quota in quotas.items()}
        for key, count in floors.items():
            allocation[key] += count
            remaining -= count
        ranked = sorted(available, key=lambda key: (
            -quotas[key][1], _stable_random_score(seed, salt, key), key,
        ))
        for key in ranked[:remaining]:
            allocation[key] += 1
        remaining = 0
    return allocation


def _allocate_hierarchical(
    pool: list[Candidate], reference: list[Candidate], *, size: int,
    stratify_fields: tuple[str, ...], seed: int, salt: str,
) -> dict[tuple[str, ...], int]:
    """Apportion biological axes first, then length/resolution within each quota."""
    if size < 0 or size > len(pool):
        raise ValueError(f"Requested {size} chains but only {len(pool)} disjoint candidate units are available.")
    if not reference and size:
        raise ValueError("Cannot apportion a nonempty sample without a reference population.")
    allocation: dict[tuple[str, ...], int] = {}

    def visit(members: list[Candidate], targets: list[Candidate], quota: int, prefix: tuple[str, ...]) -> None:
        if len(prefix) == len(stratify_fields):
            allocation[prefix or ("all",)] = quota
            return
        field = stratify_fields[len(prefix)]
        groups: defaultdict[str, list[Candidate]] = defaultdict(list)
        target_groups: defaultdict[str, list[Candidate]] = defaultdict(list)
        for member in members:
            groups[_candidate_field(member, field)].append(member)
        for target in targets:
            target_groups[_candidate_field(target, field)].append(target)
        counts = _apportion_counts(
            {key: len(value) for key, value in groups.items()},
            {key: len(value) for key, value in target_groups.items()},
            size=quota, seed=seed, salt=f"{salt}:{json.dumps(prefix)}",
        )
        for key in sorted(groups):
            visit(groups[key], target_groups.get(key, []), counts[key], (*prefix, key))

    visit(pool, reference, size, ())
    return allocation


def _allocate_counts_by_stratum(
    units: list[Candidate], *, size: int, stratify_fields: tuple[str, ...], seed: int = 0, salt: str = "allocation",
) -> dict[tuple[str, ...], int]:
    return _allocate_hierarchical(
        units, units, size=size, stratify_fields=stratify_fields, seed=seed, salt=salt,
    )


def _select_stratified(
    units: list[Candidate],
    *,
    size: int,
    seed: int,
    salt: str,
    stratify_fields: tuple[str, ...],
    reference_units: list[Candidate] | None = None,
) -> list[Candidate]:
    if size <= 0:
        return []
    by_stratum: defaultdict[tuple[str, ...], list[Candidate]] = defaultdict(list)
    for unit in units:
        by_stratum[_stratum(unit, stratify_fields)].append(unit)
    allocation = _allocate_hierarchical(
        units, units if reference_units is None else reference_units,
        size=size, stratify_fields=stratify_fields, seed=seed, salt=salt,
    )
    selected: list[Candidate] = []
    for key, count in sorted(allocation.items()):
        members = sorted(
            by_stratum[key],
            key=lambda item: (_stable_random_score(seed, f"{salt}:{'|'.join(key)}", item.chain_id), item.chain_id),
        )
        selected.extend(members[:count])
    if len(selected) != size:
        raise RuntimeError(f"Internal stratified sampling error: requested {size}, selected {len(selected)}")
    return sorted(selected, key=lambda item: (_stable_random_score(seed, f"{salt}:selected", item.chain_id), item.chain_id))


def _validate_group_nesting(
    candidates: list[Candidate], cluster_map: dict[str, str], split_group_map: dict[str, str],
) -> None:
    ids = {candidate.chain_id for candidate in candidates}
    if len(ids) != len(candidates) or set(cluster_map) != ids or set(split_group_map) != ids:
        raise ValueError("Original clusters and full split groups must cover the complete eligible candidate universe exactly.")
    if any(not isinstance(group, str) or not group.strip() for group in split_group_map.values()):
        raise ValueError("Full split group identifiers must be nonempty strings.")
    links: tuple[dict[str, str], ...] = ({}, {}, {})
    for candidate in candidates:
        group = split_group_map[candidate.chain_id]
        for seen, value in zip(links, (cluster_map[candidate.chain_id], candidate.pdb_id, candidate.sequence), strict=True):
            if value in seen and seen[value] != group:
                raise ValueError("Full split groups must contain every original cluster, PDB entry and exact-sequence component.")
            seen[value] = group


def _group_sampling_units(
    candidates: list[Candidate], cluster_map: dict[str, str], split_group_map: dict[str, str], *, seed: int,
) -> tuple[list[Candidate], dict[str, list[Candidate]]]:
    """Validate full-universe nesting and retain one representative per original unit."""
    _validate_group_nesting(candidates, cluster_map, split_group_map)
    clustered = [replace(candidate, cluster_id=cluster_map[candidate.chain_id]) for candidate in candidates]
    units = _disjoint_units(clustered, cluster_map, disallow_pdb_overlap=True, seed=seed)
    groups: defaultdict[str, list[Candidate]] = defaultdict(list)
    for candidate in units:
        groups[split_group_map[candidate.chain_id]].append(candidate)
    if set(groups) != set(split_group_map.values()):
        raise RuntimeError("Original representative population does not cover every full split group.")
    return units, dict(groups)


def _qualified_group_sampling_units(
    candidates: list[Candidate], cluster_map: dict[str, str], split_group_map: dict[str, str], *, seed: int,
    selected_chain_by_original_unit: dict[str, str | None], approved_excluded_group_ids: set[str],
) -> tuple[list[Candidate], dict[str, list[Candidate]]]:
    """Apply validated eligibility inside immutable units without deleting graph vertices."""
    _validate_group_nesting(candidates, cluster_map, split_group_map)
    memberships = _original_unit_memberships(candidates, cluster_map, seed=seed)
    if set(selected_chain_by_original_unit) != set(memberships):
        raise ValueError("Representative eligibility must account for every original unit exactly.")
    units: list[Candidate] = []
    groups: defaultdict[str, list[Candidate]] = defaultdict(list)
    for unit_id, members in memberships.items():
        selected = selected_chain_by_original_unit[unit_id]
        if selected is None:
            continue
        by_id = {candidate.chain_id: candidate for candidate in members}
        if not isinstance(selected, str) or selected not in by_id:
            raise ValueError("Qualified representative does not belong to its immutable original unit.")
        candidate = replace(by_id[selected], cluster_id=cluster_map[selected])
        units.append(candidate)
        groups[split_group_map[selected]].append(candidate)
    original_groups = set(split_group_map.values())
    if not approved_excluded_group_ids <= original_groups:
        raise ValueError("Approved exclusions include a group outside the original full graph.")
    exhausted_groups = original_groups - set(groups)
    if exhausted_groups != approved_excluded_group_ids:
        raise ValueError("Fully exhausted groups must exactly match the approved group exclusion catalog.")
    return units, dict(groups)


def _select_group_anchors(
    anchors: list[Candidate], reference: list[Candidate], costs: dict[str, int], *, size: int,
    budget: int, seed: int,
) -> list[Candidate]:
    """Select stratified held-out groups without exhausting train capacity.

    A held-out group contributes one target, so its remaining members consume
    the unused-target budget. Each seeded choice reserves the exact minimum
    completion cost of the remaining quotas.
    """
    salt = "heldout-group-ownership"
    allocation = _allocate_hierarchical(
        anchors, reference, size=size, stratify_fields=DEFAULT_STRATIFY_FIELDS, seed=seed, salt=salt,
    )
    strata: defaultdict[tuple[str, ...], list[Candidate]] = defaultdict(list)
    for candidate in anchors:
        strata[_stratum(candidate, DEFAULT_STRATIFY_FIELDS)].append(candidate)
    minimum_cost = {
        key: sum(sorted(costs[candidate.chain_id] for candidate in strata[key])[:quota])
        for key, quota in allocation.items()
    }
    minimum_remaining = sum(minimum_cost.values())
    if minimum_remaining > budget:
        raise ValueError("Held-out CATH quotas exceed the available train-capacity budget; this group allocation is infeasible.")
    selected: list[Candidate] = []
    for key, quota in sorted(allocation.items()):
        minimum_remaining -= minimum_cost[key]
        remaining_costs = sorted(costs[candidate.chain_id] for candidate in strata[key])
        ranked = sorted(strata[key], key=lambda candidate: (
            _stable_random_score(seed, f"{salt}:{'|'.join(key)}", candidate.chain_id), candidate.chain_id,
        ))
        for candidate in ranked:
            if quota == 0:
                break
            cost = costs[candidate.chain_id]
            remaining_costs.pop(bisect_left(remaining_costs, cost))
            if len(remaining_costs) < quota - 1:
                raise RuntimeError("Held-out quota completion lost available groups.")
            completion = sum(remaining_costs[:quota - 1]) + minimum_remaining
            if cost + completion <= budget:
                selected.append(candidate)
                budget -= cost
                quota -= 1
        if quota:
            raise RuntimeError("Held-out group allocation did not satisfy its exact quotas.")
    return selected


def _select_with_mandatory(
    pool: list[Candidate], mandatory: list[Candidate], reference: list[Candidate], *, size: int, seed: int,
) -> list[Candidate]:
    """Preserve every mandatory group anchor and apportion the remaining slots."""
    mandatory_ids = {candidate.chain_id for candidate in mandatory}
    if (len(mandatory_ids) != len(mandatory) or not mandatory_ids <= {candidate.chain_id for candidate in pool}
            or not len(mandatory) <= size <= len(pool)):
        raise ValueError("Mandatory group coverage is incompatible with the requested train size or candidate pool.")
    selected = list(mandatory)
    extra = [candidate for candidate in pool if candidate.chain_id not in mandatory_ids]

    def visit(members: list[Candidate], anchors: list[Candidate], targets: list[Candidate],
              quota: int, prefix: tuple[str, ...]) -> None:
        if len(prefix) == len(DEFAULT_STRATIFY_FIELDS):
            ranked = sorted(members, key=lambda candidate: (
                _stable_random_score(seed, f"train-mandatory:{'|'.join(prefix)}", candidate.chain_id), candidate.chain_id,
            ))
            selected.extend(ranked[:quota - len(anchors)])
            return
        field = DEFAULT_STRATIFY_FIELDS[len(prefix)]
        groups: defaultdict[str, list[Candidate]] = defaultdict(list)
        lower: defaultdict[str, list[Candidate]] = defaultdict(list)
        target_groups: defaultdict[str, list[Candidate]] = defaultdict(list)
        for candidate in members:
            groups[_candidate_field(candidate, field)].append(candidate)
        for candidate in anchors:
            lower[_candidate_field(candidate, field)].append(candidate)
        for candidate in targets:
            target_groups[_candidate_field(candidate, field)].append(candidate)
        keys = sorted(set(groups) | set(lower))
        denominator = max(1, len(targets))
        # Integer numerators of deficits relative to the full target quota;
        # mandatory counts are lower bounds, not a new reference population.
        deficits = {key: max(0, quota * len(target_groups[key]) - denominator * len(lower[key])) for key in keys}
        counts = _apportion_counts(
            {key: len(groups[key]) for key in keys}, deficits, size=quota - len(anchors), seed=seed,
            salt=f"train-mandatory:{json.dumps(prefix)}",
        )
        for key in keys:
            visit(groups[key], lower[key], target_groups[key], len(lower[key]) + counts[key], (*prefix, key))

    visit(extra, mandatory, reference, size, ())
    if len(selected) != size or len({candidate.chain_id for candidate in selected}) != size:
        raise RuntimeError("Mandatory group sampling did not produce the exact requested train size.")
    return sorted(selected, key=lambda candidate: (
        _stable_random_score(seed, "train-mandatory:selected", candidate.chain_id), candidate.chain_id,
    ))


def _plan_group_splits(
    units: list[Candidate], groups: dict[str, list[Candidate]], *,
    train_size: int, val_size: int, hidden_val_size: int, seed: int,
) -> tuple[list[Candidate], list[Candidate], list[list[Candidate]]]:
    """Fix all group ownership using public inputs, before any hidden salt is read."""
    sizes = (train_size, val_size, hidden_val_size)
    if any(type(size) is not int or size <= 0 for size in sizes):
        raise ValueError("Group sampling requires positive integer split sizes.")
    grouped_ids = [candidate.chain_id for members in groups.values() for candidate in members]
    if (any(not members for members in groups.values()) or len(grouped_ids) != len(set(grouped_ids))
            or set(grouped_ids) != {candidate.chain_id for candidate in units} or len(grouped_ids) != len(units)):
        raise ValueError("Full groups must partition the original representative population exactly.")
    heldout_size = val_size + hidden_val_size
    budget = len(units) - sum(sizes)
    train_group_count = len(groups) - heldout_size
    if budget < 0 or train_group_count <= 0 or train_group_count > train_size:
        raise ValueError("Split sizes cannot cover every full group with one target per held-out group and mandatory train coverage.")
    anchors = {group: _choose_representative(members, seed) for group, members in sorted(groups.items())}
    anchor_to_group = {candidate.chain_id: group for group, candidate in anchors.items()}
    costs = {candidate.chain_id: len(groups[group]) - 1 for group, candidate in anchors.items()}
    eligible = [candidate for candidate in anchors.values() if costs[candidate.chain_id] <= budget]
    if len(eligible) < heldout_size:
        raise ValueError("Not enough full groups can enter held-out splits without exhausting train capacity.")
    heldout = _select_group_anchors(eligible, units, costs, size=heldout_size, budget=budget, seed=seed)
    val = _select_stratified(
        heldout, size=val_size, seed=seed, salt="val-group-ownership", stratify_fields=DEFAULT_STRATIFY_FIELDS,
        reference_units=units,
    )
    val_ids = {candidate.chain_id for candidate in val}
    heldout_groups = {anchor_to_group[candidate.chain_id] for candidate in heldout}
    hidden_group_ids = sorted(anchor_to_group[candidate.chain_id] for candidate in heldout if candidate.chain_id not in val_ids)
    train_group_ids = sorted(set(groups) - heldout_groups)
    train = _select_with_mandatory(
        [candidate for group in train_group_ids for candidate in groups[group]],
        [anchors[group] for group in train_group_ids], units, size=train_size, seed=seed,
    )
    return train, val, [groups[group] for group in hidden_group_ids]


def _sample_hidden_groups(
    hidden_groups: list[list[Candidate]], *, seed: int, hidden_split_salt_digest: str,
) -> list[Candidate]:
    if hidden_groups and not hidden_split_salt_digest:
        raise ValueError("Hidden group member selection requires a private salt digest.")
    salt = f"hidden-group-member:{hidden_split_salt_digest}"
    hidden = [min(members, key=lambda candidate: (
        _stable_random_score(seed, salt, candidate.chain_id), candidate.chain_id,
    )) for members in hidden_groups]
    return sorted(hidden, key=lambda candidate: (
        _stable_random_score(seed, f"{salt}:selected", candidate.chain_id), candidate.chain_id,
    ))


def _split_candidates(
    *,
    candidates: list[Candidate],
    cluster_map: dict[str, str],
    train_size: int,
    val_size: int,
    hidden_val_size: int,
    seed: int,
    hidden_split_salt_digest: str,
    stratify_fields: tuple[str, ...],
    disallow_pdb_overlap: bool,
    cluster_method: str,
    cluster_command: list[str] | None,
    structural_metadata_source: str,
    sampling_policy: str = SAMPLING_POLICY,
    split_group_map: dict[str, str] | None = None,
    representative_eligibility: dict[str, Any] | None = None,
) -> SplitResult:
    candidate_by_id = {candidate.chain_id: candidate for candidate in candidates}
    candidates_with_clusters = [replace(candidate, cluster_id=cluster_map[candidate.chain_id]) for candidate in candidates]
    original_representatives: list[Candidate] | None = None
    approved_excluded_groups: set[str] = set()
    if sampling_policy in GROUP_SAMPLING_POLICIES:
        if split_group_map is None or not disallow_pdb_overlap or stratify_fields != DEFAULT_STRATIFY_FIELDS:
            raise ValueError("Grouped CATH sampling requires complete split groups, PDB disjointness and the declared CATH hierarchy.")
        if sampling_policy == PROCESSABLE_GROUP_SAMPLING_POLICY:
            if representative_eligibility is None:
                raise ValueError("cath-v4 requires validated representative eligibility.")
            original_representatives = [members[0] for members in _original_unit_memberships(
                candidates_with_clusters, cluster_map, seed=seed,
            ).values()]
            approved_excluded_groups = set(representative_eligibility["approved_excluded_group_ids"])
            units, groups = _qualified_group_sampling_units(
                candidates_with_clusters, cluster_map, split_group_map, seed=seed,
                selected_chain_by_original_unit=representative_eligibility["selected_chain_by_original_unit"],
                approved_excluded_group_ids=approved_excluded_groups,
            )
        else:
            if representative_eligibility is not None:
                raise ValueError("Representative eligibility requires cath-v4.")
            units, groups = _group_sampling_units(candidates_with_clusters, cluster_map, split_group_map, seed=seed)
    elif sampling_policy == SAMPLING_POLICY and split_group_map is None and representative_eligibility is None:
        units = _disjoint_units(candidates_with_clusters, cluster_map, disallow_pdb_overlap=disallow_pdb_overlap, seed=seed)
        groups = {}
    else:
        raise ValueError("Unsupported sampling policy or unexpected full split groups.")
    required = int(train_size) + int(val_size) + int(hidden_val_size)
    if len(units) < required:
        raise ValueError(
            f"Not enough disjoint units after clustering/grouping: need {required}, have {len(units)}. "
            "The candidate source set is too small for the official split contract."
        )

    if sampling_policy in GROUP_SAMPLING_POLICIES:
        train, val, hidden_groups = _plan_group_splits(
            units, groups, train_size=train_size, val_size=val_size, hidden_val_size=hidden_val_size, seed=seed,
        )
        hidden = _sample_hidden_groups(hidden_groups, seed=seed, hidden_split_salt_digest=hidden_split_salt_digest)
    else:
        # Public choices must not reveal holes left by privately selected hidden chains.
        val = _select_stratified(units, size=int(val_size), seed=seed, salt="val", stratify_fields=stratify_fields)
        val_ids = {candidate.chain_id for candidate in val}
        train_pool = [candidate for candidate in units if candidate.chain_id not in val_ids]
        train = _select_stratified(train_pool, size=int(train_size), seed=seed, salt="train", stratify_fields=stratify_fields)
        public_ids = val_ids | {candidate.chain_id for candidate in train}
        hidden_pool = [candidate for candidate in units if candidate.chain_id not in public_ids]
        if hidden_val_size > 0:
            if not hidden_split_salt_digest:
                raise RuntimeError("Hidden split salt digest is required when hidden_val_size is positive.")
            hidden = _select_stratified(
                hidden_pool, size=int(hidden_val_size), seed=seed,
                salt=f"hidden_val:{hidden_split_salt_digest}", stratify_fields=stratify_fields,
            )
        else:
            hidden = []

    selected = {candidate.chain_id: candidate for candidate in train + val + hidden}
    quality_report = _build_quality_report(
        selected_candidates=selected,
        split_ids={
            "train": [candidate.chain_id for candidate in train],
            "val": [candidate.chain_id for candidate in val],
            "hidden_val": [candidate.chain_id for candidate in hidden],
        },
        stratify_fields=stratify_fields,
        reference_candidates=units,
    )
    _assert_quality_report(quality_report)
    group_policy = {"cluster_disjoint": True, "pdb_disjoint": disallow_pdb_overlap, "exact_sequence_disjoint": True}
    if split_group_map is not None:
        split_counts = {name: Counter(split_group_map[candidate.chain_id] for candidate in members)
                        for name, members in (("train", train), ("val", val), ("hidden_val", hidden))}
        quality_report["group_sampling"] = {
            "allocator_policy": GROUP_ALLOCATOR_POLICY,
            "reference_population": "original_cluster_pdb_exact_representatives",
            "selection_unit_count": len(units), "full_group_count": len(groups),
            "selection_unit_ids_sha256": hashlib.sha256(
                "".join(candidate.chain_id + "\n" for candidate in sorted(units, key=lambda item: item.chain_id)).encode()
            ).hexdigest(),
            "split_group_counts": {name: len(counts) for name, counts in split_counts.items()},
            "selected_counts_by_group": {name: dict(sorted(counts.items())) for name, counts in split_counts.items()},
            "max_train_group_share": max(split_counts["train"].values(), default=0) / max(1, len(train)),
            "hidden_membership_privacy": "Group coverage reveals the hidden group complement when the full universe is known; singleton target membership is inferable.",
        }
        if sampling_policy == PROCESSABLE_GROUP_SAMPLING_POLICY:
            assert original_representatives is not None
            group_report = quality_report["group_sampling"]
            group_report.update(
                reference_population="source_validated_original_unit_representatives",
                full_group_count=len(set(split_group_map.values())),
                original_group_count=len(set(split_group_map.values())),
                retained_group_count=len(groups),
                approved_excluded_group_count=len(approved_excluded_groups),
                original_unit_count=len(original_representatives),
                original_representative_ids_sha256=hashlib.sha256("".join(
                    chain + "\n" for chain in sorted(candidate.chain_id for candidate in original_representatives)
                ).encode()).hexdigest(),
                selectable_unit_count=len(units),
                exhausted_original_unit_count=len(original_representatives) - len(units),
                replacement_representative_count=len(
                    {candidate.chain_id for candidate in units} - {candidate.chain_id for candidate in original_representatives}
                ),
            )
            group_policy.update(full_group_disjoint=True, all_retained_groups_covered=True)
        else:
            group_policy.update(full_group_disjoint=True, all_groups_covered=True)
    return SplitResult(
        train_ids=[candidate.chain_id for candidate in train],
        val_ids=[candidate.chain_id for candidate in val],
        hidden_val_ids=[candidate.chain_id for candidate in hidden],
        cluster_map={chain_id: cluster_map[chain_id] for chain_id in candidate_by_id},
        selected_candidates=selected,
        candidate_count=len(candidate_by_id),
        unit_count=len(units),
        cluster_method=cluster_method,
        cluster_command=cluster_command,
        structural_metadata_source=structural_metadata_source,
        group_policy=group_policy,
        quality_report=quality_report,
        split_group_map=split_group_map,
        approved_excluded_group_ids=frozenset(approved_excluded_groups),
    )


def _assert_split_disjointness(result: SplitResult) -> None:
    split_ids = {
        "train": set(result.train_ids),
        "val": set(result.val_ids),
        "hidden_val": set(result.hidden_val_ids),
    }
    for left_name, left_ids in split_ids.items():
        for right_name, right_ids in split_ids.items():
            if left_name >= right_name:
                continue
            overlap = left_ids & right_ids
            if overlap:
                raise RuntimeError(f"Split chain overlap between {left_name} and {right_name}: {sorted(overlap)[:8]}")

    checks = [("cluster_id", "cluster"), ("sequence", "exact sequence")]
    if result.group_policy.get("pdb_disjoint", False):
        checks.append(("pdb_id", "PDB entry"))
    for attr_name, label in checks:
        seen: dict[str, str] = {}
        for split_name, ids in split_ids.items():
            for chain_id in ids:
                candidate = result.selected_candidates[chain_id]
                value = str(getattr(candidate, attr_name))
                other_split = seen.get(value)
                if other_split is not None and other_split != split_name:
                    raise RuntimeError(
                        f"{label} leakage across splits: {value} appears in {other_split} and {split_name}"
                    )
                seen[value] = split_name
    if result.split_group_map is not None:
        ownership: dict[str, str] = {}
        for split_name, ids in split_ids.items():
            for chain_id in ids:
                group = result.split_group_map[chain_id]
                if group in ownership and ownership[group] != split_name:
                    raise RuntimeError("Full split group appears in multiple selected splits.")
                ownership[group] = split_name
        if set(ownership) != set(result.split_group_map.values()) - result.approved_excluded_group_ids:
            raise RuntimeError("Selected splits do not cover exactly every retained full eligible group.")


def _split_distribution(result: SplitResult, ids: list[str]) -> dict[str, Any]:
    classes = Counter(result.selected_candidates[chain_id].secondary_structure_class for chain_id in ids)
    domains = Counter(result.selected_candidates[chain_id].domain_architecture_class for chain_id in ids)
    lengths = Counter(_length_bin(result.selected_candidates[chain_id].length) for chain_id in ids)
    resolutions = Counter(_resolution_bin(result.selected_candidates[chain_id].resolution) for chain_id in ids)
    return {
        "cath_architecture_ids": _distribution_for_field([result.selected_candidates[chain_id] for chain_id in ids], "cath_architecture_ids"),
        "secondary_structure_class": dict(sorted(classes.items())),
        "domain_architecture_class": dict(sorted(domains.items())),
        "length_bin": dict(sorted(lengths.items())),
        "resolution_bin": dict(sorted(resolutions.items())),
    }


def _distribution_for_field(candidates: list[Candidate], field_name: str) -> dict[str, int]:
    counts = Counter(_candidate_field(candidate, field_name) for candidate in candidates)
    return dict(sorted(counts.items()))


def _js_divergence(left: dict[str, int], right: dict[str, int]) -> float:
    keys = sorted(set(left) | set(right))
    left_total = float(sum(left.values()))
    right_total = float(sum(right.values()))
    if left_total <= 0.0 or right_total <= 0.0:
        return 0.0
    divergence = 0.0
    for key in keys:
        p = left.get(key, 0) / left_total
        q = right.get(key, 0) / right_total
        m = 0.5 * (p + q)
        if p > 0.0 and m > 0.0:
            divergence += 0.5 * p * math.log2(p / m)
        if q > 0.0 and m > 0.0:
            divergence += 0.5 * q * math.log2(q / m)
    return float(divergence)


def _build_quality_report(
    *, selected_candidates: dict[str, Candidate], split_ids: dict[str, list[str]],
    stratify_fields: tuple[str, ...], reference_candidates: list[Candidate] | None = None,
) -> dict[str, Any]:
    selected = [selected_candidates[chain_id] for ids in split_ids.values() for chain_id in ids]
    reference = reference_candidates if reference_candidates is not None else selected
    all_distribution = {field: _distribution_for_field(selected, field) for field in stratify_fields}
    reference_distribution = {field: _distribution_for_field(reference, field) for field in stratify_fields}
    split_distribution = {}
    js_by_split = {}
    coverage = {}
    for split_name, ids in split_ids.items():
        members = [selected_candidates[chain_id] for chain_id in ids]
        split_distribution[split_name] = {field: _distribution_for_field(members, field) for field in stratify_fields}
        js_by_split[split_name] = {
            field: _js_divergence(split_distribution[split_name][field], reference_distribution[field])
            for field in stratify_fields
        }
        coverage[split_name] = _annotation_coverage(members)
    domain_counts = _distribution_for_field(selected, "domain_architecture_class")
    cath_policy = "cath_architecture_ids" in stratify_fields
    unknown_fraction = sum(not candidate.cath_architecture_ids for candidate in selected) / max(1, len(selected))
    max_js = max((value for fields in js_by_split.values() for value in fields.values()), default=0.0)
    return {
        "stratify_fields": list(stratify_fields),
        "selected_distribution": all_distribution,
        "reference_distribution": reference_distribution,
        "reference_population": "eligible_disjoint_units" if reference_candidates is not None else "selected_chains",
        "reference_annotation_coverage": _annotation_coverage(reference),
        "split_distribution": split_distribution,
        "annotation_coverage": coverage,
        "training_familiarity": _training_familiarity(selected_candidates, split_ids),
        "jensen_shannon_divergence": js_by_split,
        "split_sizes": {name: len(ids) for name, ids in split_ids.items()},
        "gates": {
            "max_split_js_divergence": MAX_SPLIT_JS_DIVERGENCE,
            "max_unknown_domain_architecture_fraction": MAX_UNKNOWN_DOMAIN_ARCHITECTURE_FRACTION,
        },
        "observed": {
            "max_split_js_divergence": max_js,
            "unknown_domain_architecture_fraction": unknown_fraction if cath_policy else (
                domain_counts.get("unknown", 0) / max(1, len(selected))
            ),
            "min_split_size": min((len(ids) for ids in split_ids.values() if ids), default=0),
        },
    }


def _training_familiarity(
    candidates: dict[str, Candidate], split_ids: dict[str, list[str]],
) -> dict[str, dict[str, dict[str, int]]]:
    result = {}
    train = [candidates[chain_id] for chain_id in split_ids.get("train", [])]
    for name, ids in split_ids.items():
        if name == "train":
            continue
        fields = {}
        for field in ("cath_topology_ids", "cath_homologous_superfamily_ids"):
            known = {code for candidate in train for code in getattr(candidate, field)}
            counts = dict.fromkeys(("UNANNOTATED", "NONE_SEEN", "SOME_SEEN", "ALL_SEEN"), 0)
            for chain_id in ids:
                codes = set(getattr(candidates[chain_id], field))
                overlap = codes & known
                category = ("UNANNOTATED" if not codes else "NONE_SEEN" if not overlap
                            else "ALL_SEEN" if overlap == codes else "SOME_SEEN")
                counts[category] += 1
            fields[field] = counts
        result[name] = fields
    return result


def _annotation_coverage(candidates: list[Candidate]) -> dict[str, Any]:
    count = len(candidates)
    return {
        "chain_count": count,
        "residue_count": sum(candidate.length for candidate in candidates),
        "cath_domain_count": sum(candidate.cath_domain_count for candidate in candidates),
        "cath_multidomain_chain_count": sum(candidate.cath_domain_count > 1 for candidate in candidates),
        "cath_class_count": len({code.split(".")[0] for candidate in candidates for code in candidate.cath_architecture_ids}),
        "cath_annotated_count": sum(bool(candidate.cath_architecture_ids) for candidate in candidates),
        "cath_unknown_fraction": sum(not candidate.cath_architecture_ids for candidate in candidates) / max(1, count),
        "measured_secondary_structure_count": sum(
            candidate.secondary_helix_fraction is not None and candidate.secondary_beta_fraction is not None
            and candidate.secondary_coil_fraction is not None for candidate in candidates
        ),
        "cath_architecture_count": len({code for candidate in candidates for code in candidate.cath_architecture_ids}),
        "cath_topology_count": len({code for candidate in candidates for code in candidate.cath_topology_ids}),
        "cath_superfamily_count": len({code for candidate in candidates for code in candidate.cath_homologous_superfamily_ids}),
        "label_chain_counts": {
            field: dict(sorted(Counter(code for candidate in candidates for code in getattr(candidate, field)).items()))
            for field in ("cath_architecture_ids", "cath_topology_ids", "cath_homologous_superfamily_ids")
        },
    }


def _assert_quality_report(report: dict[str, Any]) -> None:
    observed = report.get("observed", {})
    max_js = float(observed.get("max_split_js_divergence", 0.0))
    unknown_domain = float(observed.get("unknown_domain_architecture_fraction", 0.0))
    min_split_size = int(observed.get("min_split_size", 0))
    split_sizes = report.get("split_sizes", {})
    if split_sizes:
        max_js = max((value for name, fields in report.get("jensen_shannon_divergence", {}).items()
                      if split_sizes.get(name, 0) >= 100 for value in fields.values()), default=0.0)
    if (split_sizes or min_split_size >= 100) and max_js > MAX_SPLIT_JS_DIVERGENCE:
        raise ValueError(
            "Split stratification quality gate failed: "
            f"max JS divergence {max_js:.4f} exceeds {MAX_SPLIT_JS_DIVERGENCE:.4f}."
        )
    if "cath_architecture_ids" in report.get("stratify_fields", []):
        unknown_domain = max((coverage["cath_unknown_fraction"] for coverage in report.get("annotation_coverage", {}).values()
                              if coverage["chain_count"] >= 100), default=unknown_domain)
    if unknown_domain > MAX_UNKNOWN_DOMAIN_ARCHITECTURE_FRACTION:
        raise ValueError(
            "Split metadata quality gate failed: "
            f"unknown domain architecture fraction {unknown_domain:.4f} exceeds "
            f"{MAX_UNKNOWN_DOMAIN_ARCHITECTURE_FRACTION:.4f}."
        )


def _validate_metadata_identity(path: str | Path, cache_sha256: str, args: argparse.Namespace) -> None:
    metadata = json.loads(Path(path).read_text())
    if not isinstance(metadata, dict) or not isinstance(metadata.get("source"), dict) or not isinstance(metadata.get("config"), dict):
        raise ValueError("Structure metadata requires source and config provenance objects.")
    if metadata.get("source", {}).get("chain_data_cache_sha256") != cache_sha256:
        raise ValueError("Structure metadata chain_data_cache SHA256 mismatch.")
    if metadata.get("source", {}).get("chain_id_namespace") != "author_asym_id":
        raise ValueError("Structure metadata must use the author_asym_id chain namespace.")
    config = metadata["config"]
    for field in ("min_len", "max_len", "max_resolution", "require_monomer"):
        expected_type = (int, float) if field == "max_resolution" else (bool,) if field == "require_monomer" else (int,)
        if type(config.get(field)) not in expected_type or config.get(field) != getattr(args, field):
            raise ValueError(f"Structure metadata filter mismatch for {field}.")
    sample_limit = config.get("sample_limit")
    if type(sample_limit) is not int or sample_limit != 0:
        raise ValueError("CATH sampling requires complete metadata with config.sample_limit=0; rebuild without debug sampling.")
    unknown_fraction = config.get("max_unknown_aa_fraction")
    if type(unknown_fraction) not in (int, float) or unknown_fraction != 0:
        raise ValueError("CATH sampling requires metadata with max_unknown_aa_fraction=0 (standard amino acids only).")
    summary = metadata.get("summary", {})
    if not isinstance(summary, dict) or not isinstance(metadata.get("chains"), list):
        raise ValueError("Structure metadata requires full-universe provenance and a chains list.")
    for field in ("eligible_candidate_count", "candidate_count"):
        value = summary.get(field)
        if type(value) is not int or value < 0 or value != len(metadata["chains"]):
            raise ValueError(f"Structure metadata full-universe provenance mismatch for summary.{field}.")


def _validate_generation_args(args: argparse.Namespace) -> None:
    for name in ("train_size", "val_size", "hidden_val_size"):
        value = getattr(args, name, 1)
        if type(value) is not int or value <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive.")
    if type(args.min_len) is not int or type(args.max_len) is not int or args.min_len <= 0 or args.max_len < args.min_len:
        raise ValueError("Length limits must be positive and ordered.")
    if type(args.seed) is not int:
        raise ValueError("Sampling seed must be an integer.")
    if type(args.require_monomer) is not bool:
        raise ValueError("require_monomer must be boolean.")
    if type(args.max_resolution) not in (int, float) or not math.isfinite(args.max_resolution) or args.max_resolution <= 0:
        raise ValueError("Resolution cutoff must be positive and finite.")
    if any(type(value) not in (int, float) or not math.isfinite(value) or not 0 < value <= 1
           for value in (args.min_seq_id, args.coverage)):
        raise ValueError("Sequence identity and coverage must be in (0, 1].")


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, prefix=f".{path.name}.", delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _write_manifest(path: Path, chain_ids: list[str]) -> None:
    _atomic_write(path, "\n".join(chain_ids) + ("\n" if chain_ids else ""))


def main() -> None:
    args = parse_args()
    _validate_generation_args(args)
    if args.sampling_policy in GROUP_SAMPLING_POLICIES:
        if not args.split_group_tsv or not args.split_group_receipt:
            raise ValueError("Grouped CATH sampling requires --split-group-tsv and --split-group-receipt.")
    elif args.split_group_tsv or args.split_group_receipt:
        raise ValueError("Full split-group inputs require --sampling-policy cath-v3 or cath-v4.")
    if args.sampling_policy == PROCESSABLE_GROUP_SAMPLING_POLICY:
        if not args.representative_eligibility_receipt or not args.approved_group_exclusions:
            raise ValueError("cath-v4 requires --representative-eligibility-receipt and --approved-group-exclusions.")
    elif args.representative_eligibility_receipt or args.approved_group_exclusions:
        raise ValueError("Representative eligibility and approved group exclusions require --sampling-policy cath-v4.")
    if not args.lock_file.strip() or not args.private_lock_file.strip():
        raise ValueError("CATH sampling requires nonempty --lock-file and --private-lock-file paths.")

    cache_path = Path(args.chain_data_cache)
    cache_sha = _sha256(cache_path)
    expected_cache_sha = args.expected_chain_cache_sha256.strip().lower()
    if expected_cache_sha:
        if len(expected_cache_sha) != 64 or any(ch not in "0123456789abcdef" for ch in expected_cache_sha):
            raise ValueError("--expected-chain-cache-sha256 must be a 64-char lowercase hex digest.")
        if cache_sha != expected_cache_sha:
            raise ValueError(
                "chain_data_cache SHA256 mismatch.\n"
                f"expected: {expected_cache_sha}\n"
                f"actual:   {cache_sha}\n"
                f"path:     {cache_path.resolve()}"
            )

    data = json.loads(cache_path.read_text())
    if not isinstance(data, dict):
        raise ValueError(
            "chain_data_cache.json must be a JSON object keyed by chain id. "
            f"Got {type(data).__name__}."
        )

    _validate_metadata_identity(args.structure_metadata, cache_sha, args)
    structure_metadata_accepted, structure_metadata = _load_structure_metadata(args.structure_metadata, require_schema_version=2)

    candidates, reject_reasons, structural_metadata_source = _build_candidates(
        data,
        min_len=int(args.min_len),
        max_len=int(args.max_len),
        max_resolution=float(args.max_resolution),
        structure_metadata=structure_metadata,
        structure_metadata_accepted=structure_metadata_accepted,
        require_monomer=args.require_monomer,
        sampling_policy=args.sampling_policy,
    )
    required_count = int(args.train_size) + int(args.val_size) + int(args.hidden_val_size)
    if int(args.hidden_val_size) <= 0:
        raise ValueError("--hidden-val-size must be positive for official split generation.")
    if len(candidates) < required_count:
        raise ValueError(f"Not enough candidates after filtering: need {required_count}, have {len(candidates)}")
    hidden_split_salt = _read_hidden_split_salt(
        salt_env=str(args.hidden_split_salt_env),
        salt_file=str(args.hidden_split_salt_file),
    )
    hidden_split_salt_digest = _salt_digest(hidden_split_salt)

    candidate_ids = [candidate.chain_id for candidate in candidates]
    candidate_sequences: Dict[str, str] = {candidate.chain_id: candidate.sequence for candidate in candidates}

    cluster_method = "mmseqs2"
    cluster_command: List[str] | None = None
    cluster_map: Dict[str, str]
    if args.cluster_tsv:
        cluster_method = "precomputed_tsv"
        cluster_map = _load_cluster_tsv(args.cluster_tsv, set(candidate_ids))
    elif len(candidate_sequences) == len(candidates):
        mmseqs_cluster_map, cluster_command = _cluster_sequences_mmseqs(
            sequences={cid: candidate_sequences[cid] for cid in candidate_ids},
            mmseqs_bin=args.mmseqs_bin,
            min_seq_id=float(args.min_seq_id),
            coverage=float(args.coverage),
        )
        if mmseqs_cluster_map is not None:
            cluster_method = "mmseqs2"
            cluster_map = mmseqs_cluster_map
        else:
            raise RuntimeError(
                "Official split generation requires MMseqs2 clustering or a locked --cluster-tsv. "
                f"Install `{args.mmseqs_bin}` or provide a TSV generated with --min-seq-id "
                f"{args.min_seq_id} and coverage {args.coverage}."
            )
    else:
        raise RuntimeError("All accepted structure metadata candidates must have sequences in chain_data_cache.json.")

    split_group_map = _load_cluster_tsv(args.split_group_tsv, set(candidate_ids)) if args.split_group_tsv else None
    group_receipt: dict[str, Any] | None = None
    if split_group_map is not None:
        from build_homology_groups import validate_group_receipt

        group_receipt = validate_group_receipt(
            Path(args.split_group_receipt), split_group_tsv=Path(args.split_group_tsv),
            chain_data_cache=cache_path, structure_metadata=Path(args.structure_metadata),
            cluster_map=cluster_map, candidate_ids=set(candidate_ids),
            min_seq_id=float(args.min_seq_id), coverage=float(args.coverage),
        )
    representative_eligibility: dict[str, Any] | None = None
    if args.sampling_policy == PROCESSABLE_GROUP_SAMPLING_POLICY:
        from nanofold.representative_eligibility import validate_representative_eligibility_receipt

        assert split_group_map is not None
        memberships = _original_unit_memberships(candidates, cluster_map, seed=int(args.seed))
        representative_eligibility = validate_representative_eligibility_receipt(
            Path(args.representative_eligibility_receipt),
            approved_group_exclusions_path=Path(args.approved_group_exclusions),
            chain_data_cache_path=cache_path, structure_metadata_path=Path(args.structure_metadata),
            cluster_map=cluster_map, split_group_map=split_group_map,
            split_group_receipt_path=Path(args.split_group_receipt),
            original_units={unit_id: [candidate.chain_id for candidate in members] for unit_id, members in memberships.items()},
            seed=int(args.seed),
        )
    stratify_fields = DEFAULT_STRATIFY_FIELDS
    result = _split_candidates(
        candidates=candidates,
        cluster_map=cluster_map,
        train_size=int(args.train_size),
        val_size=int(args.val_size),
        hidden_val_size=int(args.hidden_val_size),
        seed=int(args.seed),
        hidden_split_salt_digest=hidden_split_salt_digest,
        stratify_fields=stratify_fields,
        disallow_pdb_overlap=True,
        cluster_method=cluster_method,
        cluster_command=cluster_command,
        structural_metadata_source=structural_metadata_source,
        sampling_policy=args.sampling_policy,
        split_group_map=split_group_map,
        representative_eligibility=representative_eligibility,
    )
    _assert_split_disjointness(result)
    original_unit_count = result.quality_report.get("group_sampling", {}).get("original_unit_count", result.unit_count)
    if group_receipt is not None and group_receipt.get("original_unit_count") != original_unit_count:
        raise ValueError("Full-group receipt original-unit count disagrees with the reconstructed sampling population.")
    split_groups_record = None if split_group_map is None else {
        "path": str(args.split_group_tsv), "sha256": _sha256(Path(args.split_group_tsv)),
        "assignments_sha256": _cluster_assignments_sha256(split_group_map),
        "group_count": len(set(split_group_map.values())), "coverage": "all_groups",
        "selection_unit_count": result.unit_count, "allocator_policy": GROUP_ALLOCATOR_POLICY,
        "selection_unit_ids_sha256": result.quality_report["group_sampling"]["selection_unit_ids_sha256"],
        "receipt_path": str(args.split_group_receipt), "receipt_sha256": _sha256(Path(args.split_group_receipt)),
    }
    if representative_eligibility is not None:
        assert split_groups_record is not None
        group_report = result.quality_report["group_sampling"]
        split_groups_record.update(
            coverage="all_retained_groups", original_unit_count=original_unit_count,
            original_representative_ids_sha256=group_report["original_representative_ids_sha256"],
            retained_group_count=group_report["retained_group_count"],
            approved_excluded_group_count=group_report["approved_excluded_group_count"],
        )

    out_dir = Path(args.out_dir)
    hidden_out_dir = Path(args.hidden_out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    hidden_out_dir.mkdir(parents=True, exist_ok=True)
    train_path = out_dir / "train.txt"
    val_path = out_dir / "val.txt"
    hidden_val_path = hidden_out_dir / "hidden_val.txt"
    all_path = out_dir / "all.txt"
    quality_report_path = hidden_out_dir / "split_quality_report.json"
    output_paths = [train_path, val_path, hidden_val_path, all_path, quality_report_path]
    output_paths.extend(Path(value) for value in (args.lock_file, args.private_lock_file) if value)
    resolved_paths = [path.resolve() for path in output_paths]
    if len(set(resolved_paths)) != len(resolved_paths):
        raise ValueError("Sampling output paths must be distinct.")
    for path in output_paths:
        if path.exists():
            raise ValueError(f"Refusing to overwrite existing sampling output; use a new version directory: {path}")
    _write_manifest(train_path, result.train_ids)
    _write_manifest(val_path, result.val_ids)
    _write_manifest(hidden_val_path, result.hidden_val_ids)
    all_ids = sorted(set(result.train_ids + result.val_ids))
    _write_manifest(all_path, all_ids)
    _atomic_write(quality_report_path, json.dumps(result.quality_report, indent=2, sort_keys=True) + "\n")

    print(
        f"Wrote {len(result.train_ids)} train + {len(result.val_ids)} val chains to {out_dir} "
        f"and {len(result.hidden_val_ids)} hidden_val chains to {hidden_out_dir} "
        "(cluster/PDB-disjoint: 0 overlaps)"
    )

    if args.lock_file:
        split_distributions = {
            "train": _split_distribution(result, result.train_ids),
            "val": _split_distribution(result, result.val_ids),
        }
        public_selected = {
            chain_id: result.selected_candidates[chain_id]
            for chain_id in result.train_ids + result.val_ids
        }
        public_quality_report = _build_quality_report(
            selected_candidates=public_selected,
            split_ids={
                "train": result.train_ids,
                "val": result.val_ids,
            },
            stratify_fields=stratify_fields,
        )
        lock = {
            "sampling_policy": args.sampling_policy,
            "chain_data_cache_path": str(cache_path),
            "chain_data_cache_sha256": cache_sha,
            "args": {
                "sampling_policy": args.sampling_policy,
                "train_size": int(args.train_size),
                "val_size": int(args.val_size),
                "hidden_val_size": int(args.hidden_val_size),
                "min_len": int(args.min_len),
                "max_len": int(args.max_len),
                "max_resolution": float(args.max_resolution),
                "require_monomer": args.require_monomer,
                "seed": int(args.seed),
                "min_seq_id": float(args.min_seq_id),
                "coverage": float(args.coverage),
                "mmseqs_bin": str(args.mmseqs_bin),
                "cluster_tsv": str(args.cluster_tsv) if args.cluster_tsv else None,
                "structure_metadata": str(args.structure_metadata),
                "stratify_fields": list(stratify_fields),
                "expected_chain_cache_sha256": expected_cache_sha or None,
            },
            "filtering": {
                "candidate_count": result.candidate_count,
                "disjoint_unit_count": original_unit_count,
                "target_policy": "known_monomer" if args.require_monomer else "single_chain",
                "candidate_annotation_coverage": _annotation_coverage(candidates),
                "representative_policy": "best_resolution_then_longest_then_release_date_seeded_tie",
                "reject_reasons": dict(sorted(reject_reasons.items())),
            },
            "structure_metadata": _structure_metadata_source_record(args.structure_metadata),
            "clustering": {
                "method": cluster_method,
                "cluster_tsv_sha256": _sha256(Path(args.cluster_tsv)) if args.cluster_tsv else None,
                "guarantee": "assigned_cluster_disjointness; not an all-pairs sequence identity bound",
                "cluster_count": len(set(result.cluster_map.values())),
                "command": cluster_command,
                "mmseqs": _mmseqs_version(str(args.mmseqs_bin)) if cluster_method == "mmseqs2" else None,
                "cluster_assignments_sha256": _cluster_assignments_sha256(result.cluster_map),
            },
            "grouping": result.group_policy,
            "stratification": {
                "fields": list(stratify_fields),
                "structural_metadata_source": result.structural_metadata_source,
                "split_distributions": split_distributions,
                "quality_report": public_quality_report,
            },
            "outputs": {
                "train_manifest": str(train_path),
                "val_manifest": str(val_path),
                "all_manifest": str(all_path),
                "train_manifest_sha256": _sha256(train_path),
                "val_manifest_sha256": _sha256(val_path),
                "all_manifest_sha256": _sha256(all_path),
                "train_count": len(result.train_ids),
                "val_count": len(result.val_ids),
                "unique_count": len(all_ids),
            },
        }
        if split_groups_record is not None:
            lock["split_groups"] = split_groups_record
            lock["args"].update(split_group_tsv=str(args.split_group_tsv), split_group_receipt=str(args.split_group_receipt))
        if representative_eligibility is not None:
            lock["representative_eligibility"] = representative_eligibility["binding"]
            lock["args"].update(
                representative_eligibility_receipt=str(args.representative_eligibility_receipt),
                approved_group_exclusions=str(args.approved_group_exclusions),
            )
            lock["filtering"].update(
                selectable_unit_count=result.unit_count,
                representative_policy="source_validated_best_resolution_then_longest_then_release_date_seeded_tie",
            )
        lock_path = Path(args.lock_file)
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write(lock_path, json.dumps(lock, indent=2) + "\n")
        print(f"Wrote manifest lock file: {lock_path.resolve()}")

    if args.private_lock_file:
        private_lock = {
            "sampling_policy": args.sampling_policy,
            "notes": "Maintainer-only hidden manifest lock. Do not commit this file.",
            "chain_data_cache_sha256": cache_sha,
            "hidden_split_salt_sha256": hidden_split_salt_digest,
            "hidden_val_count": len(result.hidden_val_ids),
            "hidden_val_manifest": str(hidden_val_path),
            "hidden_val_manifest_sha256": _sha256(hidden_val_path),
            "split_quality_report": str(quality_report_path),
            "split_quality_report_sha256": _sha256(quality_report_path),
            "hidden_split_distribution": _split_distribution(result, result.hidden_val_ids),
            "source_lock_sha256": _sha256(Path(args.lock_file)),
            "public_contract": {
                "train_manifest_sha256": _sha256(train_path), "train_count": len(result.train_ids),
                "val_manifest_sha256": _sha256(val_path), "val_count": len(result.val_ids),
                "all_manifest_sha256": _sha256(all_path), "all_count": len(all_ids),
            },
            "structure_metadata": _structure_metadata_source_record(args.structure_metadata),
            "clustering": {"cluster_assignments_sha256": _cluster_assignments_sha256(result.cluster_map)},
            "stratification": {"reference": "eligible_disjoint_units", "fields": list(stratify_fields)},
            "quality_report": result.quality_report,
        }
        if split_groups_record is not None:
            private_lock["split_groups"] = split_groups_record
            private_lock["filtering"] = {
                "private_processability_exclusion_count": 0,
                "private_processability_exclusions": {
                    "path": None, "sha256": None,
                    "canonical_ids_sha256": hashlib.sha256(b"").hexdigest(),
                },
            }
        if representative_eligibility is not None:
            private_lock["representative_eligibility"] = representative_eligibility["binding"]
            private_lock["stratification"]["reference"] = "source_validated_disjoint_units"
        private_lock_path = Path(args.private_lock_file)
        private_lock_path.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write(private_lock_path, json.dumps(private_lock, indent=2) + "\n")
        print(f"Wrote private hidden manifest lock file: {private_lock_path.resolve()}")


if __name__ == "__main__":
    main()
