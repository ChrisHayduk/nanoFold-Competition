"""Build canonical chain-level structure metadata for official split generation."""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from nanofold.structure_metadata import (
    STRUCTURE_METADATA_SCHEMA_VERSION,
    domain_architecture_from_cath_class,
    domain_architecture_from_scop_sccs,
    secondary_fractions_from_mmcif_annotations,
    validate_author_chain_id,
)

STANDARD_AAS = set("ARNDCQEGHILKMFPSTWYV")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build required split metadata from chain_data_cache and pinned structural classification sources. "
            "This is the maintainer input consumed by scripts/build_manifests.py."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--chain-data-cache", type=Path, required=True)
    parser.add_argument("--metadata-out", type=Path, required=True)
    parser.add_argument(
        "--metadata-sources-dir",
        type=Path,
        default=Path("data/metadata_sources"),
        help="Directory populated by scripts/download_structure_metadata_sources.py.",
    )
    parser.add_argument(
        "--metadata-source-lock",
        type=Path,
        default=Path("data/metadata_sources/structure_metadata_sources.lock.json"),
        help="Source lock JSON produced by scripts/download_structure_metadata_sources.py.",
    )
    parser.add_argument(
        "--feature-exclusion-list",
        type=Path,
        default=Path("data/manifests/openfold_required_feature_exclusions.txt"),
        help="Pinned chain IDs whose required official feature assets are unavailable.",
    )
    parser.add_argument(
        "--processability-exclusion-list",
        type=Path,
        default=Path("data/manifests/official_processability_exclusions.txt"),
        help="Pinned chain IDs that fail the official atom14 label processability gate.",
    )
    parser.add_argument(
        "--candidate-manifest-out",
        type=Path,
        default=None,
        help="Optional text manifest of accepted chain IDs available for split generation.",
    )
    parser.add_argument("--min-len", type=int, default=40)
    parser.add_argument("--max-len", type=int, default=256)
    parser.add_argument("--max-resolution", type=float, default=3.0)
    parser.add_argument("--max-unknown-aa-fraction", type=float, default=0.0)
    parser.add_argument(
        "--require-monomer",
        action="store_true",
        help="Require an explicit biological oligomeric_count of 1; unknown assembly state is rejected.",
    )
    parser.add_argument(
        "--sample-limit",
        type=int,
        default=0,
        help="Debug-only limit on eligible chains processed; cath-v2 and cath-v3 require 0 (the complete eligible universe).",
    )
    parser.add_argument("--fail-fast", action="store_true")
    return parser.parse_args(argv)


def _pdb_id(chain_id: str) -> str:
    if "_" not in chain_id:
        raise ValueError(f"Invalid chain id (expected <pdb>_<chain>): {chain_id}")
    return chain_id.split("_", 1)[0].lower()


def _chain_name(chain_id: str) -> str:
    if "_" not in chain_id:
        raise ValueError(f"Invalid chain id (expected <pdb>_<chain>): {chain_id}")
    return chain_id.split("_", 1)[1]


def _extract_sequence(meta: dict[str, Any]) -> str | None:
    for key in ("sequence", "seq", "seqres", "aatype_sequence"):
        value = meta.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip().upper()
    return None


def _unknown_aa_fraction(sequence: str) -> float:
    if not sequence:
        return 1.0
    unknown = sum(1 for ch in sequence.upper() if ch not in STANDARD_AAS)
    return unknown / float(len(sequence))


def _numeric(value: object, default: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (str, bytes, bytearray, int, float)):
        return default
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _candidate_rows(
    data: dict[str, Any],
    *,
    min_len: int,
    max_len: int,
    max_resolution: float,
    max_unknown_aa_fraction: float,
    require_monomer: bool = False,
) -> tuple[list[dict[str, Any]], Counter[str]]:
    if min_len <= 0 or max_len < min_len:
        raise ValueError("Invalid sequence length bounds")
    if not math.isfinite(max_resolution) or max_resolution <= 0:
        raise ValueError("max_resolution must be positive and finite")
    if not math.isfinite(max_unknown_aa_fraction) or not 0 <= max_unknown_aa_fraction <= 1:
        raise ValueError("max_unknown_aa_fraction must be in [0, 1]")
    candidates: list[dict[str, Any]] = []
    reject_counts: Counter[str] = Counter()
    for chain_id, raw_meta in sorted(data.items()):
        validate_author_chain_id(chain_id)
        if not isinstance(raw_meta, dict):
            reject_counts["bad_metadata"] += 1
            continue
        sequence = _extract_sequence(raw_meta)
        if not sequence:
            reject_counts["missing_sequence"] += 1
            continue
        length = len(sequence)
        declared_length = _numeric(raw_meta.get("seq_length", raw_meta.get("sequence_length", length)), -1)
        resolution = _numeric(raw_meta.get("resolution"), 999.0)
        # OpenFold uses zero for unavailable resolution, not a high-quality measurement.
        if resolution <= 0:
            resolution = 999.0
        oligomer = _numeric(raw_meta.get("oligomeric_count"), -1)
        oligomeric_count = int(oligomer) if oligomer > 0 and float(oligomer).is_integer() else None
        oligomeric_state = (
            "unknown" if oligomeric_count is None else ("known_monomer" if oligomeric_count == 1 else "known_multimer")
        )
        reasons: list[str] = []
        if declared_length != length:
            reasons.append("sequence_length_mismatch")
        if length < min_len:
            reasons.append("min_length")
        if length > max_len:
            reasons.append("max_length")
        if resolution != 999.0 and resolution > max_resolution:
            reasons.append("resolution")
        if require_monomer and oligomeric_count != 1:
            reasons.append("unknown_oligomeric_count" if oligomeric_count is None else "oligomeric_count")
        if _unknown_aa_fraction(sequence) > max_unknown_aa_fraction:
            reasons.append("unknown_aa_fraction")
        if reasons:
            reject_counts.update(reasons)
            continue
        candidates.append(
            {
                "chain_id": chain_id,
                "sequence": sequence,
                "length": length,
                "resolution": resolution,
                "oligomeric_count": oligomeric_count,
                "oligomeric_state": oligomeric_state,
            }
        )
    return candidates, reject_counts


def _chain_key(pdb_id: str, chain_id: str) -> str:
    return f"{pdb_id.lower()}_{chain_id}"


def _add_source_value(out: dict[str, set[str]], key: str, value: str) -> None:
    normalized = value.strip()
    if not normalized or normalized == "unknown":
        return
    out.setdefault(key, set()).add(normalized)


def _load_cath_annotations(path: Path) -> dict[str, list[dict[str, str]]]:
    """Read CATH domains without collapsing hierarchy levels or multiple domains.

    IDs contain a PDB ID, case-sensitive author chain, and two-digit domain
    number. The list has no boundaries and does not imply residue ordering.
    """
    annotations: dict[str, list[dict[str, str]]] = {}
    seen: dict[str, tuple[str, ...]] = {}
    for line_number, raw_line in enumerate(path.read_text().splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) != 12 or not re.fullmatch(r"[0-9A-Za-z]{4}[0-9A-Za-z][0-9]{2}", parts[0]):
            raise ValueError(f"Malformed CATH domain-list row at {path.name}:{line_number}")
        if any(not value.isdigit() for value in parts[1:11]):
            raise ValueError(f"Invalid CATH integer field at {path.name}:{line_number}")
        hierarchy = [int(value) for value in parts[1:5]]
        if any(value <= 0 for value in hierarchy) or int(parts[10]) <= 0:
            raise ValueError(f"Invalid CATH hierarchy or domain length at {path.name}:{line_number}")
        if _numeric(parts[11], -1) < 0:
            raise ValueError(f"Invalid CATH resolution at {path.name}:{line_number}")
        domain_id = parts[0][:4].lower() + parts[0][4:]
        record = tuple(parts[1:])
        if domain_id in seen:
            if seen[domain_id] != record:
                raise ValueError(f"Conflicting CATH annotations for domain {domain_id}")
            continue
        seen[domain_id] = record
        codes = [".".join(str(value) for value in hierarchy[:depth]) for depth in range(1, 5)]
        annotations.setdefault(_chain_key(domain_id[:4], domain_id[4]), []).append(
            {
                "domain_id": domain_id,
                "class_id": codes[0],
                "architecture_id": codes[1],
                "topology_id": codes[2],
                "homologous_superfamily_id": codes[3],
            }
        )
    if not annotations:
        raise ValueError(f"CATH domain list contains no annotations: {path}")
    return {key: sorted(domains, key=lambda domain: domain["domain_id"]) for key, domains in annotations.items()}


def _load_scope_annotations(path: Path) -> dict[str, set[str]]:
    annotations: dict[str, set[str]] = {}
    seen: dict[str, tuple[str, ...]] = {}
    for line_number, raw_line in enumerate(path.read_text().splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) != 6 or not re.fullmatch(r"[a-z]\.[0-9]+\.[0-9]+\.[0-9]+", parts[3]):
            raise ValueError(f"Malformed SCOPe row at {path.name}:{line_number}")
        record = tuple(parts[1:])
        if parts[0] in seen and seen[parts[0]] != record:
            raise ValueError(f"Conflicting SCOPe annotations for domain {parts[0]}")
        seen[parts[0]] = record
        chains = {segment.split(":", 1)[0] for segment in parts[2].split(",") if ":" in segment}
        for chain in sorted(chains - {"", "-", ".", "_"}):
            _add_source_value(annotations, _chain_key(parts[1], chain), domain_architecture_from_scop_sccs(parts[3]))
    return annotations


def _load_ecod_annotations(path: Path) -> dict[str, set[str]]:
    """Read named ECOD columns, retaining only explicitly identified PDB chains.

    ECOD distributions include both PDB domains and non-PDB predicted domains.
    The latter have empty PDB/chain fields and cannot annotate our PDB targets.
    """
    annotations: dict[str, set[str]] = {}
    columns: dict[str, int] | None = None
    counts: Counter[str] = Counter()
    required = {"uid", "ecod_domain_id", "manual_rep", "pdb", "chain", "pdb_range", "seqid_range"}
    with path.open() as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.rstrip("\r\n")
            if not line.strip():
                continue
            parts = line.removeprefix("#").split("\t")
            if parts[0] == "uid":
                if (
                    columns is not None or len(parts) != len(set(parts)) or any(not field for field in parts)
                    or not required.issubset(parts) or len({"f_id", "t_id"}.intersection(parts)) != 1
                ):
                    raise ValueError(f"Invalid ECOD header at {path.name}:{line_number}")
                columns = {name: index for index, name in enumerate(parts)}
                continue
            if line.lstrip().startswith("#"):
                continue
            if columns is None or len(parts) != len(columns):
                raise ValueError(f"Malformed ECOD row or missing header at {path.name}:{line_number}")
            hierarchy = parts[columns["f_id" if "f_id" in columns else "t_id"]]
            if (
                not parts[columns["uid"]].isdigit()
                or not parts[columns["ecod_domain_id"]]
                or any(char.isspace() for char in parts[columns["ecod_domain_id"]])
                or parts[columns["manual_rep"]] not in {"True", "False", "MANUAL_REP", "MANUAL_NONREP", "AUTO_REP", "AUTO_NONREP"}
                or not re.fullmatch(r"[0-9]+(?:\.[0-9]+){2,3}", hierarchy)
                or ("valid_structure" in columns and parts[columns["valid_structure"]] not in {"True", "False"})
            ):
                raise ValueError(f"Malformed ECOD identity fields at {path.name}:{line_number}")
            counts["domains"] += 1
            pdb_id, chain_field = parts[columns["pdb"]], parts[columns["chain"]]
            if not pdb_id and not chain_field:
                counts["non_pdb_domains"] += 1
                continue
            if not re.fullmatch(r"[0-9A-Za-z]{4}", pdb_id) or not re.fullmatch(r"[^\s_:,]+", chain_field):
                raise ValueError(f"Malformed ECOD PDB/chain fields at {path.name}:{line_number}")
            if "valid_structure" in columns and parts[columns["valid_structure"]] == "False":
                counts["invalid_structure_domains"] += 1
                continue
            chains = set(chain_field.split("-"))
            if not all(chains) or chains.intersection({".", "?"}):
                raise ValueError(f"Malformed ECOD chain identifiers at {path.name}:{line_number}")
            if len(chains) > 1:
                # A cross-chain domain uses e.g. P-B; corroborate each chain in
                # the explicit ranges rather than inventing an author chain P-B.
                range_chains = {
                    segment.split(":", 1)[0]
                    for segment in parts[columns["seqid_range"]].split(",") if ":" in segment
                }
                if chains != range_chains:
                    raise ValueError(f"Ambiguous ECOD cross-chain mapping at {path.name}:{line_number}")
            for chain in sorted(chains):
                _add_source_value(annotations, _chain_key(pdb_id, chain), "annotated")
            counts["pdb_domains"] += 1
    if columns is None or not counts["domains"]:
        raise ValueError(f"ECOD domain list has no header or domain records: {path.name}")
    print(
        f"ECOD metadata: {counts['pdb_domains']} PDB domains; "
        f"excluded {counts['non_pdb_domains']} non-PDB and "
        f"{counts['invalid_structure_domains']} invalid-structure domains.",
        file=sys.stderr,
    )
    return annotations


def _load_rcsb_annotations(path: Path) -> tuple[dict[str, set[str]], dict[str, dict[str, Any]]]:
    annotations: dict[str, set[str]] = {}
    metadata: dict[str, dict[str, Any]] = {}
    for line_number, raw_line in enumerate(path.read_text().splitlines(), start=1):
        if not raw_line.strip():
            continue
        row = json.loads(raw_line)
        if not isinstance(row, dict) or not isinstance(row.get("chain_id"), str):
            raise ValueError(f"Malformed RCSB record at {path.name}:{line_number}")
        chain_id = row["chain_id"]
        validate_author_chain_id(chain_id)
        if chain_id in metadata:
            raise ValueError(f"Duplicate RCSB chain record: {chain_id}")
        entry = row.get("entry") if isinstance(row.get("entry"), dict) else {}
        entity = row.get("entity") if isinstance(row.get("entity"), dict) else {}
        instance = row.get("instance")
        identifiers = instance.get("rcsb_polymer_entity_instance_container_identifiers") if isinstance(instance, dict) else None
        verified_chain = (
            isinstance(identifiers, dict)
            and identifiers.get("auth_asym_id") == _chain_name(chain_id)
            and str(identifiers.get("entry_id", "")).lower() == _pdb_id(chain_id)
            and bool(entity)
            and not row.get("errors")
        )
        chain_entity = entity if verified_chain else {}
        metadata[chain_id] = {
            "experimental_method": _first_nested_value(entry, ("exptl", "method")),
            "taxonomy_id": _first_nested_value(chain_entity, ("rcsb_entity_source_organism", "ncbi_taxonomy_id")),
            "source_organism": _first_nested_value(chain_entity, ("rcsb_entity_source_organism", "scientific_name")),
            "errors": row.get("errors", []),
            "chain_mapping_status": "verified_author_chain" if verified_chain else "unresolved",
        }
        if verified_chain:
            _add_source_value(annotations, chain_id, "annotated")
    return annotations, metadata


def _first_nested_value(obj: Any, path: tuple[str, ...]) -> Any:
    current = obj
    for key in path:
        if isinstance(current, list):
            current = current[0] if current else None
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    if isinstance(current, list):
        return current[0] if current else None
    return current


SOURCE_FILENAMES = {
    "cath_domain_list": "cath-domain-list.txt",
    "scope_classification": "dir.cla.scope.txt",
    "ecod_domains": "ecod.latest.domains.txt",
    "rcsb_chain_metadata": "rcsb_chain_metadata.jsonl",
    "dssp_mmcif": "dssp_mmcif",
}


def _tree_sha256(root: Path) -> str:
    import hashlib

    hasher = hashlib.sha256()
    for path in sorted(path for path in root.rglob("*") if path.is_file()):
        hasher.update(path.relative_to(root).as_posix().encode("utf-8"))
        hasher.update(b"\t")
        hasher.update(_sha256(path).encode("ascii"))
        hasher.update(b"\n")
    return hasher.hexdigest()


def _verify_source_lock(source_lock: Path, metadata_sources_dir: Path, chain_data_cache: Path) -> dict[str, Any]:
    if not source_lock.is_file():
        raise ValueError(f"Missing metadata source lock: {source_lock}")
    raw = json.loads(source_lock.read_text())
    if not isinstance(raw, dict) or not isinstance(raw.get("sources"), dict):
        raise ValueError("Metadata source lock must contain a sources object")
    if raw.get("schema_version", 1) not in {1, 2}:
        raise ValueError("Unsupported metadata source lock schema version")
    cache_record = raw.get("chain_data_cache")
    if not isinstance(cache_record, dict) or cache_record.get("sha256") != _sha256(chain_data_cache):
        raise ValueError("Metadata source lock chain_data_cache SHA-256 mismatch")
    verified: dict[str, Any] = {}
    for name, record in raw["sources"].items():
        if name not in SOURCE_FILENAMES:
            raise ValueError(f"Unsupported metadata source: {name}")
        if not isinstance(record, dict):
            raise ValueError(f"Invalid metadata source record: {name}")
        if record.get("skipped") is True:
            if name == "cath_domain_list":
                raise ValueError("CATH domain list is a required metadata source")
            continue
        path = metadata_sources_dir / SOURCE_FILENAMES[name]
        digest_field = "tree_sha256" if name == "dssp_mmcif" else "sha256"
        expected = record.get(digest_field)
        if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected):
            raise ValueError(f"Missing valid {digest_field} for metadata source {name}")
        if not path.exists() or (not path.is_dir() if name == "dssp_mmcif" else not path.is_file()):
            raise ValueError(f"Missing metadata source {name}: {path}")
        observed = _tree_sha256(path) if name == "dssp_mmcif" else _sha256(path)
        if observed != expected:
            raise ValueError(f"Metadata source SHA-256 mismatch: {name}")
        verified[name] = record
    if "cath_domain_list" not in verified:
        raise ValueError("CATH domain list is a required metadata source")
    return verified


def _load_external_sources(
    metadata_sources_dir: Path,
    verified_sources: dict[str, Any],
) -> tuple[dict[str, dict[str, set[str]]], dict[str, dict[str, Any]], dict[str, list[dict[str, str]]]]:
    cath_domains = _load_cath_annotations(metadata_sources_dir / "cath-domain-list.txt")
    cath: dict[str, set[str]] = {}
    for chain_id, domains in cath_domains.items():
        # CATH '0' denotes an absent PDB chain field, not a literal author chain 0.
        if _chain_name(chain_id) == "0":
            continue
        for domain in domains:
            raw_class = domain["class_id"]
            broad_class = (
                domain_architecture_from_cath_class(raw_class) if raw_class in {"1", "2", "3", "4"} else "other"
            )
            cath.setdefault(chain_id, set()).add(broad_class)
    scope = (
        _load_scope_annotations(metadata_sources_dir / "dir.cla.scope.txt")
        if "scope_classification" in verified_sources
        else {}
    )
    ecod = (
        _load_ecod_annotations(metadata_sources_dir / "ecod.latest.domains.txt")
        if "ecod_domains" in verified_sources
        else {}
    )
    rcsb, rcsb_meta = (
        _load_rcsb_annotations(metadata_sources_dir / "rcsb_chain_metadata.jsonl")
        if "rcsb_chain_metadata" in verified_sources
        else ({}, {})
    )
    return {"cath": cath, "scope": scope, "ecod": ecod, "rcsb": rcsb}, rcsb_meta, cath_domains


def _select_domain_architecture(
    chain_id: str,
    external_sources: dict[str, dict[str, set[str]]],
) -> tuple[str, str, dict[str, list[str]]]:
    if _chain_name(chain_id) == "0":
        return "unknown", "unknown", {}
    values_by_source = {
        name: sorted(values[chain_id]) for name, values in external_sources.items() if values.get(chain_id)
    }
    for source_name in ("cath", "scope"):
        values = values_by_source.get(source_name, [])
        if values:
            return values[0] if len(values) == 1 else "multi_class", source_name, values_by_source
    return "unknown", "unknown", values_by_source


def _load_feature_exclusions(path: Path) -> set[str]:
    if not path.is_file():
        raise ValueError(f"Missing required exclusion list: {path}")
    out: set[str] = set()
    for raw_line in path.read_text().splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        validate_author_chain_id(line)
        out.add(line)
    return out


def _feature_exclusion_source(path: Path, chain_ids: set[str]) -> dict[str, Any]:
    return {
        "path": str(path),
        "sha256": _sha256(path) if path.exists() else None,
        "chain_count": len(chain_ids),
    }


def _source_metadata_fields(
    *,
    chain_id: str,
    length: int,
    resolution: float,
    external_sources: dict[str, dict[str, set[str]]],
    rcsb_metadata: dict[str, dict[str, Any]],
    cath_domains_by_chain: dict[str, list[dict[str, str]]],
    dssp_dir: Path | None,
) -> tuple[dict[str, Any], str, str, dict[str, bool]]:
    domain_class, domain_source, domain_source_values = _select_domain_architecture(chain_id, external_sources)
    raw_domains = cath_domains_by_chain.get(chain_id, [])
    ambiguous_chain = _chain_name(chain_id) == "0" and bool(raw_domains)
    cath_domains = [] if ambiguous_chain else raw_domains
    secondary: tuple[str, float | None, float | None, float | None, str | None] = ("unknown", None, None, None, None)
    if dssp_dir is not None:
        mmcif_path = dssp_dir / f"{_pdb_id(chain_id)}.cif"
        if mmcif_path.is_file():
            secondary = secondary_fractions_from_mmcif_annotations(
                str(mmcif_path), chain_id=_chain_name(chain_id), length=length
            )
    secondary_class, helix, beta, coil, secondary_source = secondary
    source_coverage = {
        source_name: bool(external_sources.get(source_name, {}).get(chain_id))
        for source_name in ("cath", "scope", "ecod", "rcsb")
    }
    source_coverage["secondary_structure"] = secondary_source is not None
    nonempty_class_sets = {
        tuple(values) for source, values in domain_source_values.items() if source in {"cath", "scope"}
    }
    fields = {
        "accepted": True,
        "reject_reasons": [],
        "secondary_structure_class": secondary_class,
        "secondary_helix_fraction": helix,
        "secondary_beta_fraction": beta,
        "secondary_coil_fraction": coil,
        "secondary_structure_source": secondary_source,
        "domain_architecture_class": domain_class,
        "domain_architecture_source": domain_source,
        "domain_architecture_source_values": domain_source_values,
        "domain_class_source_conflict": len(nonempty_class_sets) > 1,
        "cath_domains": cath_domains,
        "cath_class_ids": sorted({domain["class_id"] for domain in cath_domains}),
        "cath_architecture_ids": sorted({domain["architecture_id"] for domain in cath_domains}),
        "cath_topology_ids": sorted({domain["topology_id"] for domain in cath_domains}),
        "cath_homologous_superfamily_ids": sorted({domain["homologous_superfamily_id"] for domain in cath_domains}),
        "cath_annotation_status": "annotated" if cath_domains else "unknown",
        "cath_mapping_status": "ambiguous_blank_chain"
        if ambiguous_chain
        else ("exact_author_chain" if cath_domains else "unannotated"),
        "cath_unresolved_domains": raw_domains if ambiguous_chain else [],
        "metadata_source_coverage": source_coverage,
        "metadata_source_count": sum(source_coverage.values()),
        "rcsb_metadata": rcsb_metadata.get(chain_id, {}),
        "structure_resolution": float(resolution),
    }
    return fields, secondary_class, domain_class, source_coverage


def _sha256(path: Path) -> str:
    import hashlib

    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def build_metadata(
    *,
    chain_data_cache: Path,
    metadata_sources_dir: Path,
    metadata_source_lock: Path,
    feature_exclusion_list: Path,
    processability_exclusion_list: Path,
    min_len: int,
    max_len: int,
    max_resolution: float,
    max_unknown_aa_fraction: float,
    sample_limit: int = 0,
    fail_fast: bool = False,
    require_monomer: bool = False,
) -> dict[str, Any]:
    """Build metadata with the eligible-universe size and any debug truncation recorded.

    ``sample_limit`` is for debugging only. Official cath-v2 generation requires
    zero, which processes every eligible chain, including subsequent exclusions.
    """
    if isinstance(sample_limit, bool) or not isinstance(sample_limit, int) or sample_limit < 0:
        raise ValueError("sample_limit must be a nonnegative integer")
    verified_sources = _verify_source_lock(metadata_source_lock, metadata_sources_dir, chain_data_cache)
    raw = json.loads(chain_data_cache.read_text())
    if not isinstance(raw, dict):
        raise ValueError(f"chain_data_cache must be a JSON object: {chain_data_cache}")
    candidates, initial_reject_counts = _candidate_rows(
        raw,
        min_len=min_len,
        max_len=max_len,
        max_resolution=max_resolution,
        max_unknown_aa_fraction=max_unknown_aa_fraction,
        require_monomer=require_monomer,
    )
    eligible_candidate_count = len(candidates)
    if sample_limit > 0:
        candidates = candidates[:sample_limit]

    entries: list[dict[str, Any]] = []
    reject_counts = Counter(initial_reject_counts)
    secondary_counts: Counter[str] = Counter()
    domain_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    external_sources, rcsb_metadata, cath_domains = _load_external_sources(metadata_sources_dir, verified_sources)
    dssp_dir = metadata_sources_dir / "dssp_mmcif" if "dssp_mmcif" in verified_sources else None
    feature_exclusions = _load_feature_exclusions(feature_exclusion_list)
    processability_exclusions = _load_feature_exclusions(processability_exclusion_list)

    for candidate in tqdm(candidates, desc="structure-metadata"):
        chain_id = str(candidate["chain_id"])
        pdb_id = _pdb_id(chain_id)
        _chain_name(chain_id)
        entry: dict[str, Any] = {
            "chain_id": chain_id,
            "pdb_id": pdb_id,
            "accepted": False,
            "reject_reasons": [],
            "length": int(candidate["length"]),
            "resolution": float(candidate["resolution"]),
            "oligomeric_count": candidate["oligomeric_count"],
            "oligomeric_state": candidate["oligomeric_state"],
        }
        if chain_id in feature_exclusions:
            entry["reject_reasons"] = ["missing_required_feature_asset"]
            reject_counts["missing_required_feature_asset"] += 1
            entries.append(entry)
            continue
        if chain_id in processability_exclusions:
            entry["reject_reasons"] = ["failed_official_label_processability"]
            reject_counts["failed_official_label_processability"] += 1
            entries.append(entry)
            continue
        try:
            fields, cls, domain_class, source_coverage = _source_metadata_fields(
                chain_id=chain_id,
                length=int(candidate["length"]),
                cath_domains_by_chain=cath_domains,
                dssp_dir=dssp_dir,
                resolution=float(candidate["resolution"]),
                external_sources=external_sources,
                rcsb_metadata=rcsb_metadata,
            )
            entry.update(fields)
            secondary_counts[cls] += 1
            domain_counts[domain_class] += 1
            source_counts.update(source for source, present in source_coverage.items() if present)
        except Exception as exc:
            reason = str(exc).split(":", 1)[0] or type(exc).__name__
            entry["reject_reasons"] = [reason]
            reject_counts[reason] += 1
            if fail_fast:
                raise
        entries.append(entry)

    return {
        "schema_version": STRUCTURE_METADATA_SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": {
            "chain_data_cache": str(chain_data_cache),
            "chain_data_cache_sha256": _sha256(chain_data_cache),
            "chain_id_namespace": "author_asym_id",
            "cath_annotation_semantics": "domain presence, not residue coverage",
            "cath_source_domain_count": sum(len(domains) for domains in cath_domains.values()),
            "cath_source_blank_chain_domain_count": sum(
                len(domains) for chain, domains in cath_domains.items() if _chain_name(chain) == "0"
            ),
            "metadata_sources_dir": str(metadata_sources_dir),
            "metadata_source_lock": {
                "lock_path": str(metadata_source_lock),
                "lock_sha256": _sha256(metadata_source_lock),
                "sources": verified_sources,
            },
            "feature_exclusion_list": _feature_exclusion_source(feature_exclusion_list, feature_exclusions),
            "processability_exclusion_list": _feature_exclusion_source(
                processability_exclusion_list,
                processability_exclusions,
            ),
        },
        "config": {
            "sample_limit": sample_limit,
            "require_monomer": require_monomer,
            "prediction_target": "single_chain",
            "min_len": int(min_len),
            "max_len": int(max_len),
            "max_resolution": float(max_resolution),
            "max_unknown_aa_fraction": float(max_unknown_aa_fraction),
            "required_feature_assets": ["uniref90_hits.a3m"],
            "label_processability": {
                "min_projection_seq_identity": 0.90,
                "min_projection_coverage": 0.70,
                "min_projection_aligned_fraction": 0.70,
                "min_projection_valid_ca": 32,
            },
        },
        "summary": {
            "eligible_candidate_count": eligible_candidate_count,
            "candidate_count": len(candidates),
            "oligomeric_state": dict(sorted(Counter(item["oligomeric_state"] for item in candidates).items())),
            "cath_annotation_status": dict(
                sorted(Counter(item.get("cath_annotation_status", "rejected") for item in entries).items())
            ),
            "accepted": sum(1 for item in entries if item["accepted"]),
            "rejected": sum(1 for item in entries if not item["accepted"]),
            "reject_reasons": dict(sorted(reject_counts.items())),
            "secondary_structure_class": dict(sorted(secondary_counts.items())),
            "domain_architecture_class": dict(sorted(domain_counts.items())),
            "metadata_source_coverage": dict(sorted(source_counts.items())),
        },
        "chains": entries,
    }


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    metadata = build_metadata(
        chain_data_cache=args.chain_data_cache,
        metadata_sources_dir=args.metadata_sources_dir,
        metadata_source_lock=args.metadata_source_lock,
        feature_exclusion_list=args.feature_exclusion_list,
        processability_exclusion_list=args.processability_exclusion_list,
        min_len=int(args.min_len),
        max_len=int(args.max_len),
        max_resolution=float(args.max_resolution),
        max_unknown_aa_fraction=float(args.max_unknown_aa_fraction),
        sample_limit=int(args.sample_limit),
        fail_fast=bool(args.fail_fast),
        require_monomer=bool(args.require_monomer),
    )
    args.metadata_out.parent.mkdir(parents=True, exist_ok=True)
    args.metadata_out.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")

    if args.candidate_manifest_out is not None:
        candidates = [entry["chain_id"] for entry in metadata["chains"] if entry["accepted"]]
        args.candidate_manifest_out.parent.mkdir(parents=True, exist_ok=True)
        args.candidate_manifest_out.write_text("".join(f"{chain_id}\n" for chain_id in candidates))

    summary = metadata["summary"]
    print(
        f"[structure-metadata] candidates={summary['candidate_count']} accepted={summary['accepted']} "
        f"rejected={summary['rejected']}"
    )
    print(f"[structure-metadata] metadata -> {args.metadata_out}")
    if args.candidate_manifest_out is not None:
        print(f"[structure-metadata] candidates -> {args.candidate_manifest_out}")


if __name__ == "__main__":
    main()
