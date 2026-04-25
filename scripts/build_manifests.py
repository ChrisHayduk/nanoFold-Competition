from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

"""Create train/val/hidden manifests from an OpenFold/OpenProteinSet chain_data_cache.

Maintainer-only workflow:
- use this script to generate/refresh official manifests and lock metadata.
- participants should use committed manifests via scripts/setup_official_data.sh.

Expected input:
- chain_data_cache.json (available from RODA as described in OpenFold docs)

We filter by:
- monomer chains
- length range
- resolution cutoff
and then sample fixed-size split chain sets with leakage-aware grouping:
- no sequence cluster is allowed to appear in multiple splits
- no PDB entry is allowed to appear in multiple splits
- required secondary-structure metadata is used to stratify by
  secondary structure, domain architecture, length, and resolution bins

The chain_data_cache parser accepts the field names present in OpenFold's
published cache metadata and fails closed when required structure metadata is absent.
"""

STANDARD_AAS = set("ARNDCQEGHILKMFPSTWYV")
DEFAULT_STRATIFY_FIELDS = (
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


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
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
            "scripts/filter_openproteinset.py. Accepted chains must include secondary-structure metadata."
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
        default="",
        help="Optional public JSON path capturing non-hidden generation inputs/hashes.",
    )
    ap.add_argument(
        "--private-lock-file",
        type=str,
        default="",
        help="Optional maintainer-only JSON path capturing hidden split metadata and salt digest.",
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
    if out < 0.0:
        return None
    return out


def _load_structure_metadata(path: str | Path) -> tuple[set[str], dict[str, dict[str, Any]]]:
    metadata_path = Path(path)
    raw = json.loads(metadata_path.read_text())
    if not isinstance(raw, dict) or not isinstance(raw.get("chains"), list):
        raise ValueError(f"Structure metadata must contain a `chains` list: {metadata_path}")
    accepted: set[str] = set()
    metadata: dict[str, dict[str, Any]] = {}
    for item in raw["chains"]:
        if not isinstance(item, dict):
            continue
        chain_id = str(item.get("chain_id") or "").strip()
        if not chain_id:
            continue
        metadata[chain_id] = item
        if bool(item.get("accepted", False)):
            cls, helix_f, beta_f, coil_f = _parse_secondary_payload(item)
            if cls == "unknown" or helix_f is None or beta_f is None or coil_f is None:
                raise ValueError(
                    "Accepted structure metadata rows must include secondary-structure class and "
                    f"helix/beta/coil fractions. Bad chain: {chain_id}"
                )
            domain_class, _domain_source, _source_count = _parse_domain_payload(item)
            if domain_class == "unknown":
                raise ValueError(
                    "Accepted structure metadata rows must include a domain_architecture_class. "
                    f"Bad chain: {chain_id}"
                )
            accepted.add(chain_id)
    if not accepted:
        raise ValueError(f"Structure metadata has no accepted chains: {metadata_path}")
    return accepted, metadata


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
        for line in handle:
            parts = line.strip().split("\t")
            if len(parts) < 2:
                continue
            representative, member = parts[0].strip(), parts[1].strip()
            if member in candidates:
                cluster_map[member] = representative
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
            "--threads",
            "1",
        ]
        proc = subprocess.run(actual_cmd, capture_output=True, text=True, check=False)
        if proc.returncode != 0:
            return None, lock_cmd

        cluster_tsv = Path(str(cluster_prefix) + "_cluster.tsv")
        if not cluster_tsv.exists():
            return None, lock_cmd

        cluster_map: Dict[str, str] = {}
        for line in cluster_tsv.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            rep, member = line.split("\t")
            cluster_map[member] = rep
        if set(cluster_map.keys()) != set(sequences.keys()):
            return None, lock_cmd
        return cluster_map, lock_cmd


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
) -> tuple[list[Candidate], Counter[str], str]:
    candidates: list[Candidate] = []
    reject_reasons: Counter[str] = Counter()
    secondary_source = "structure_metadata"

    for chain_id, meta_raw in data.items():
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

        raw_length = meta.get("seq_length", meta.get("sequence_length", len(seq)))
        try:
            L = int(raw_length) if isinstance(raw_length, (str, bytes, bytearray, int, float)) else len(seq)
        except Exception:
            L = len(seq)
        raw_resolution = meta.get("resolution", 999.0)
        try:
            res = float(raw_resolution) if isinstance(raw_resolution, (str, bytes, bytearray, int, float)) else 999.0
        except Exception:
            res = 999.0

        olig = meta.get("oligomeric_count", meta.get("num_chains", 1))
        try:
            olig = int(olig) if isinstance(olig, (str, bytes, bytearray, int, float)) else 1
        except Exception:
            olig = 1

        if not (min_len <= L <= max_len):
            reject_reasons["length"] += 1
            continue
        if res != 999.0 and res > max_resolution:
            reject_reasons["resolution"] += 1
            continue
        if olig != 1:
            reject_reasons["oligomeric_count"] += 1
            continue

        metadata_entry = structure_metadata[chain_id]
        ss_class, helix_fraction, beta_fraction, coil_fraction = _parse_secondary_payload(metadata_entry)
        if ss_class == "unknown" or helix_fraction is None or beta_fraction is None or coil_fraction is None:
            reject_reasons["missing_secondary_structure_metadata"] += 1
            continue
        domain_class, domain_source, metadata_source_count = _parse_domain_payload(metadata_entry)
        if domain_class == "unknown":
            reject_reasons["missing_domain_architecture_metadata"] += 1
            continue

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
            )
        )
    return candidates, reject_reasons, secondary_source


def _choose_representative(candidates: list[Candidate], seed: int) -> Candidate:
    return min(
        candidates,
        key=lambda item: (
            1 if item.secondary_structure_class == "unknown" else 0,
            item.resolution if item.resolution and item.resolution < 999.0 else 999.0,
            -item.length,
            item.release_date or "",
            _stable_random_score(seed, "representative", item.chain_id),
            item.chain_id,
        ),
    )


def _disjoint_units(
    candidates: list[Candidate],
    cluster_map: dict[str, str],
    *,
    disallow_pdb_overlap: bool,
    seed: int,
) -> list[Candidate]:
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
    for candidate in candidates:
        by_cluster[cluster_map[candidate.chain_id]].append(candidate.chain_id)
        by_pdb[candidate.pdb_id].append(candidate.chain_id)
    for cluster_members in by_cluster.values():
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

    units: list[Candidate] = []
    for component_members in by_component.values():
        units.append(_choose_representative(component_members, seed))
    return units


def _allocate_counts_by_stratum(
    units: list[Candidate],
    *,
    size: int,
    stratify_fields: tuple[str, ...],
) -> dict[tuple[str, ...], int]:
    if size <= 0:
        return {}
    by_stratum: defaultdict[tuple[str, ...], list[Candidate]] = defaultdict(list)
    for unit in units:
        by_stratum[_stratum(unit, stratify_fields)].append(unit)
    if size > len(units):
        raise ValueError(f"Requested {size} chains but only {len(units)} disjoint candidate units are available.")

    total = len(units)
    allocation: dict[tuple[str, ...], int] = {}
    remainders: list[tuple[float, int, tuple[str, ...]]] = []
    allocated = 0
    for key, members in sorted(by_stratum.items()):
        exact = size * len(members) / float(total)
        count = min(len(members), int(exact))
        allocation[key] = count
        allocated += count
        remainders.append((exact - count, len(members), key))

    remaining = size - allocated
    for _, _, key in sorted(remainders, key=lambda item: (-item[0], -item[1], item[2])):
        if remaining <= 0:
            break
        capacity = len(by_stratum[key]) - allocation[key]
        if capacity <= 0:
            continue
        allocation[key] += 1
        remaining -= 1

    if remaining > 0:
        for key, members in sorted(by_stratum.items(), key=lambda item: (-len(item[1]), item[0])):
            if remaining <= 0:
                break
            capacity = len(members) - allocation[key]
            take = min(capacity, remaining)
            allocation[key] += take
            remaining -= take
    return allocation


def _select_stratified(
    units: list[Candidate],
    *,
    size: int,
    seed: int,
    salt: str,
    stratify_fields: tuple[str, ...],
) -> list[Candidate]:
    if size <= 0:
        return []
    by_stratum: defaultdict[tuple[str, ...], list[Candidate]] = defaultdict(list)
    for unit in units:
        by_stratum[_stratum(unit, stratify_fields)].append(unit)
    allocation = _allocate_counts_by_stratum(units, size=size, stratify_fields=stratify_fields)
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
) -> SplitResult:
    candidate_by_id = {candidate.chain_id: candidate for candidate in candidates}
    candidates_with_clusters = [
        Candidate(
            chain_id=candidate.chain_id,
            sequence=candidate.sequence,
            length=candidate.length,
            resolution=candidate.resolution,
            release_date=candidate.release_date,
            cluster_id=cluster_map[candidate.chain_id],
            pdb_id=candidate.pdb_id,
            secondary_structure_class=candidate.secondary_structure_class,
            secondary_helix_fraction=candidate.secondary_helix_fraction,
            secondary_beta_fraction=candidate.secondary_beta_fraction,
            secondary_coil_fraction=candidate.secondary_coil_fraction,
            domain_architecture_class=candidate.domain_architecture_class,
            domain_architecture_source=candidate.domain_architecture_source,
            metadata_source_count=candidate.metadata_source_count,
        )
        for candidate in candidates
    ]
    units = _disjoint_units(
        candidates_with_clusters,
        cluster_map,
        disallow_pdb_overlap=disallow_pdb_overlap,
        seed=seed,
    )
    required = int(train_size) + int(val_size) + int(hidden_val_size)
    if len(units) < required:
        raise ValueError(
            f"Not enough disjoint units after clustering/grouping: need {required}, have {len(units)}. "
            "The candidate source set is too small for the official split contract."
        )

    if hidden_val_size > 0:
        if not hidden_split_salt_digest:
            raise RuntimeError("Hidden split salt digest is required when hidden_val_size is positive.")
        hidden = _select_stratified(
            units,
            size=int(hidden_val_size),
            seed=seed,
            salt=f"hidden_val:{hidden_split_salt_digest}",
            stratify_fields=stratify_fields,
        )
        hidden_ids = {candidate.chain_id for candidate in hidden}
    else:
        hidden = []
        hidden_ids = set()

    public_pool = [candidate for candidate in units if candidate.chain_id not in hidden_ids]
    val = _select_stratified(public_pool, size=int(val_size), seed=seed, salt="val", stratify_fields=stratify_fields)
    val_ids = {candidate.chain_id for candidate in val}
    train_pool = [candidate for candidate in public_pool if candidate.chain_id not in val_ids]
    train = _select_stratified(train_pool, size=int(train_size), seed=seed, salt="train", stratify_fields=stratify_fields)

    selected = {candidate.chain_id: candidate for candidate in train + val + hidden}
    quality_report = _build_quality_report(
        selected_candidates=selected,
        split_ids={
            "train": [candidate.chain_id for candidate in train],
            "val": [candidate.chain_id for candidate in val],
            "hidden_val": [candidate.chain_id for candidate in hidden],
        },
        stratify_fields=stratify_fields,
    )
    _assert_quality_report(quality_report)
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
        group_policy={"cluster_disjoint": True, "pdb_disjoint": disallow_pdb_overlap},
        quality_report=quality_report,
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

    checks = [("cluster_id", "cluster")]
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


def _split_distribution(result: SplitResult, ids: list[str]) -> dict[str, Any]:
    classes = Counter(result.selected_candidates[chain_id].secondary_structure_class for chain_id in ids)
    domains = Counter(result.selected_candidates[chain_id].domain_architecture_class for chain_id in ids)
    lengths = Counter(_length_bin(result.selected_candidates[chain_id].length) for chain_id in ids)
    resolutions = Counter(_resolution_bin(result.selected_candidates[chain_id].resolution) for chain_id in ids)
    return {
        "secondary_structure_class": dict(sorted(classes.items())),
        "domain_architecture_class": dict(sorted(domains.items())),
        "length_bin": dict(sorted(lengths.items())),
        "resolution_bin": dict(sorted(resolutions.items())),
    }


def _distribution_for_field(candidates: list[Candidate], field_name: str) -> dict[str, int]:
    counts = Counter(_candidate_field(candidate, field_name) for candidate in candidates)
    return dict(sorted(counts.items()))


def _js_divergence(left: dict[str, int], right: dict[str, int]) -> float:
    keys = set(left) | set(right)
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
    *,
    selected_candidates: dict[str, Candidate],
    split_ids: dict[str, list[str]],
    stratify_fields: tuple[str, ...],
) -> dict[str, Any]:
    selected = [selected_candidates[chain_id] for ids in split_ids.values() for chain_id in ids]
    all_distribution = {
        field_name: _distribution_for_field(selected, field_name)
        for field_name in stratify_fields
    }
    split_distribution: dict[str, dict[str, dict[str, int]]] = {}
    js_by_split: dict[str, dict[str, float]] = {}
    for split_name, ids in split_ids.items():
        members = [selected_candidates[chain_id] for chain_id in ids]
        split_distribution[split_name] = {
            field_name: _distribution_for_field(members, field_name)
            for field_name in stratify_fields
        }
        js_by_split[split_name] = {
            field_name: _js_divergence(split_distribution[split_name][field_name], all_distribution[field_name])
            for field_name in stratify_fields
        }
    domain_counts = all_distribution.get("domain_architecture_class", {})
    unknown_domain_fraction = domain_counts.get("unknown", 0) / float(max(1, sum(domain_counts.values())))
    max_js = max(
        (value for fields in js_by_split.values() for value in fields.values()),
        default=0.0,
    )
    return {
        "stratify_fields": list(stratify_fields),
        "selected_distribution": all_distribution,
        "split_distribution": split_distribution,
        "jensen_shannon_divergence": js_by_split,
        "gates": {
            "max_split_js_divergence": MAX_SPLIT_JS_DIVERGENCE,
            "max_unknown_domain_architecture_fraction": MAX_UNKNOWN_DOMAIN_ARCHITECTURE_FRACTION,
        },
        "observed": {
            "max_split_js_divergence": max_js,
            "unknown_domain_architecture_fraction": unknown_domain_fraction,
            "min_split_size": min((len(ids) for ids in split_ids.values() if ids), default=0),
        },
    }


def _assert_quality_report(report: dict[str, Any]) -> None:
    observed = report.get("observed", {})
    max_js = float(observed.get("max_split_js_divergence", 0.0))
    unknown_domain = float(observed.get("unknown_domain_architecture_fraction", 0.0))
    min_split_size = int(observed.get("min_split_size", 0))
    if min_split_size >= 100 and max_js > MAX_SPLIT_JS_DIVERGENCE:
        raise ValueError(
            "Split stratification quality gate failed: "
            f"max JS divergence {max_js:.4f} exceeds {MAX_SPLIT_JS_DIVERGENCE:.4f}."
        )
    if unknown_domain > MAX_UNKNOWN_DOMAIN_ARCHITECTURE_FRACTION:
        raise ValueError(
            "Split metadata quality gate failed: "
            f"unknown domain architecture fraction {unknown_domain:.4f} exceeds "
            f"{MAX_UNKNOWN_DOMAIN_ARCHITECTURE_FRACTION:.4f}."
        )


def _write_manifest(path: Path, chain_ids: list[str]) -> None:
    path.write_text("\n".join(chain_ids) + ("\n" if chain_ids else ""))


def main() -> None:
    args = parse_args()

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

    structure_metadata_accepted, structure_metadata = _load_structure_metadata(args.structure_metadata)

    candidates, reject_reasons, structural_metadata_source = _build_candidates(
        data,
        min_len=int(args.min_len),
        max_len=int(args.max_len),
        max_resolution=float(args.max_resolution),
        structure_metadata=structure_metadata,
        structure_metadata_accepted=structure_metadata_accepted,
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
    )
    _assert_split_disjointness(result)

    out_dir = Path(args.out_dir)
    hidden_out_dir = Path(args.hidden_out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    hidden_out_dir.mkdir(parents=True, exist_ok=True)
    train_path = out_dir / "train.txt"
    val_path = out_dir / "val.txt"
    hidden_val_path = hidden_out_dir / "hidden_val.txt"
    all_path = out_dir / "all.txt"
    quality_report_path = hidden_out_dir / "split_quality_report.json"
    _write_manifest(train_path, result.train_ids)
    _write_manifest(val_path, result.val_ids)
    _write_manifest(hidden_val_path, result.hidden_val_ids)
    all_ids = sorted(set(result.train_ids + result.val_ids))
    _write_manifest(all_path, all_ids)
    quality_report_path.write_text(json.dumps(result.quality_report, indent=2, sort_keys=True) + "\n")

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
            "chain_data_cache_path": str(cache_path),
            "chain_data_cache_sha256": cache_sha,
            "args": {
                "train_size": int(args.train_size),
                "val_size": int(args.val_size),
                "hidden_val_size": int(args.hidden_val_size),
                "min_len": int(args.min_len),
                "max_len": int(args.max_len),
                "max_resolution": float(args.max_resolution),
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
                "disjoint_unit_count": result.unit_count,
                "reject_reasons": dict(sorted(reject_reasons.items())),
            },
            "structure_metadata": _structure_metadata_source_record(args.structure_metadata),
            "clustering": {
                "method": cluster_method,
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
        lock_path = Path(args.lock_file)
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        lock_path.write_text(json.dumps(lock, indent=2) + "\n")
        print(f"Wrote manifest lock file: {lock_path.resolve()}")

    if args.private_lock_file:
        private_lock = {
            "notes": "Maintainer-only hidden manifest lock. Do not commit this file.",
            "chain_data_cache_sha256": cache_sha,
            "hidden_split_salt_sha256": hidden_split_salt_digest,
            "hidden_val_count": len(result.hidden_val_ids),
            "hidden_val_manifest": str(hidden_val_path),
            "hidden_val_manifest_sha256": _sha256(hidden_val_path),
            "split_quality_report": str(quality_report_path),
            "split_quality_report_sha256": _sha256(quality_report_path),
            "hidden_split_distribution": _split_distribution(result, result.hidden_val_ids),
            "quality_report": result.quality_report,
        }
        private_lock_path = Path(args.private_lock_file)
        private_lock_path.parent.mkdir(parents=True, exist_ok=True)
        private_lock_path.write_text(json.dumps(private_lock, indent=2) + "\n")
        print(f"Wrote private hidden manifest lock file: {private_lock_path.resolve()}")


if __name__ == "__main__":
    main()
