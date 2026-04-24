"""Build canonical chain-level structure metadata for official split generation."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from nanofold.mmcif import extract_chain_atoms
from nanofold.structure_metadata import (
    STRUCTURE_METADATA_SCHEMA_VERSION,
    domain_architecture_from_cath_class,
    domain_architecture_from_scop_sccs,
    normalize_domain_architecture_class,
    secondary_fractions_from_atom14,
    secondary_fractions_from_mmcif_annotations,
)

STANDARD_AAS = set("ARNDCQEGHILKMFPSTWYV")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build required secondary-structure metadata from chain_data_cache + mmCIF atom14 geometry. "
            "This is the maintainer input consumed by scripts/build_manifests.py."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--chain-data-cache", type=Path, required=True)
    parser.add_argument("--mmcif-root", type=Path, required=True)
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
        "--candidate-manifest-out",
        type=Path,
        default=None,
        help="Optional text manifest of filtered candidate chain IDs, useful for downloading missing mmCIFs.",
    )
    parser.add_argument("--min-len", type=int, default=40)
    parser.add_argument("--max-len", type=int, default=256)
    parser.add_argument("--max-resolution", type=float, default=3.0)
    parser.add_argument("--max-unknown-aa-fraction", type=float, default=0.0)
    parser.add_argument("--sample-limit", type=int, default=0)
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
    if not isinstance(value, (str, bytes, bytearray, int, float)):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _candidate_reject_reasons(
    *,
    sequence: str | None,
    length: int,
    resolution: float,
    oligomeric_count: int,
    min_len: int,
    max_len: int,
    max_resolution: float,
    max_unknown_aa_fraction: float,
) -> list[str]:
    reasons: list[str] = []
    if not sequence:
        reasons.append("missing_sequence")
        return reasons
    if length < min_len:
        reasons.append("min_length")
    if length > max_len:
        reasons.append("max_length")
    if resolution != 999.0 and resolution > max_resolution:
        reasons.append("resolution")
    if oligomeric_count != 1:
        reasons.append("oligomeric_count")
    if _unknown_aa_fraction(sequence) > max_unknown_aa_fraction:
        reasons.append("unknown_aa_fraction")
    return reasons


def _candidate_rows(
    data: dict[str, Any],
    *,
    min_len: int,
    max_len: int,
    max_resolution: float,
    max_unknown_aa_fraction: float,
) -> tuple[list[dict[str, Any]], Counter[str]]:
    candidates: list[dict[str, Any]] = []
    reject_counts: Counter[str] = Counter()
    for chain_id, raw_meta in sorted(data.items()):
        if not isinstance(raw_meta, dict):
            reject_counts["bad_metadata"] += 1
            continue
        meta: dict[str, Any] = raw_meta
        sequence = _extract_sequence(meta)
        length = int(_numeric(meta.get("seq_length", meta.get("sequence_length", len(sequence or ""))), len(sequence or "")))
        resolution = _numeric(meta.get("resolution", 999.0), 999.0)
        oligomeric_count = int(_numeric(meta.get("oligomeric_count", meta.get("num_chains", 1)), 1.0))
        reasons = _candidate_reject_reasons(
            sequence=sequence,
            length=length,
            resolution=resolution,
            oligomeric_count=oligomeric_count,
            min_len=min_len,
            max_len=max_len,
            max_resolution=max_resolution,
            max_unknown_aa_fraction=max_unknown_aa_fraction,
        )
        if reasons:
            reject_counts.update(reasons)
            continue
        candidates.append(
            {
                "chain_id": chain_id,
                "sequence": sequence,
                "length": length,
                "resolution": resolution,
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


def _extract_chain_from_domain_id(domain_id: str) -> tuple[str, str] | None:
    text = domain_id.strip()
    if len(text) < 6:
        return None
    if text[0].lower() == "e":
        text = text[1:]
    pdb_id = text[:4].lower()
    chain = text[4:-1] if text[-1].isdigit() else text[4:]
    if not pdb_id or not chain:
        return None
    return pdb_id, chain


def _load_cath_annotations(path: Path) -> dict[str, set[str]]:
    annotations: dict[str, set[str]] = {}
    if not path.exists():
        return annotations
    for raw_line in path.read_text(errors="ignore").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        domain_id = parts[0].strip()
        if len(domain_id) < 5:
            continue
        pdb_id, chain = domain_id[:4].lower(), domain_id[4]
        _add_source_value(annotations, _chain_key(pdb_id, chain), domain_architecture_from_cath_class(parts[1]))
    return annotations


def _load_scope_annotations(path: Path) -> dict[str, set[str]]:
    annotations: dict[str, set[str]] = {}
    if not path.exists():
        return annotations
    for raw_line in path.read_text(errors="ignore").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 4:
            continue
        pdb_id = parts[1].lower()
        chain_range = parts[2]
        chain = chain_range.split(":", 1)[0].strip("_")
        if not chain or chain in {"-", "."}:
            parsed = _extract_chain_from_domain_id(parts[0])
            if parsed is None:
                continue
            pdb_id, chain = parsed
        _add_source_value(annotations, _chain_key(pdb_id, chain), domain_architecture_from_scop_sccs(parts[3]))
    return annotations


def _classify_ecod_line(line: str) -> str:
    lowered = line.lower()
    if "alpha" in lowered and "beta" in lowered:
        return "alpha_beta"
    if "a+b" in lowered or "a/b" in lowered:
        return "alpha_beta"
    if "alpha" in lowered or "helical" in lowered or "helix" in lowered:
        return "alpha"
    if "beta" in lowered or "sandwich" in lowered or "barrel" in lowered:
        return "beta"
    return "unknown"


def _load_ecod_annotations(path: Path) -> dict[str, set[str]]:
    annotations: dict[str, set[str]] = {}
    if not path.exists():
        return annotations
    for raw_line in path.read_text(errors="ignore").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parsed = None
        for token in line.replace("\t", " ").split():
            if token.startswith("e") and len(token) >= 6:
                parsed = _extract_chain_from_domain_id(token)
                if parsed is not None:
                    break
        if parsed is None:
            continue
        pdb_id, chain = parsed
        _add_source_value(annotations, _chain_key(pdb_id, chain), _classify_ecod_line(line))
    return annotations


def _walk_values(obj: Any) -> list[Any]:
    values: list[Any] = []
    if isinstance(obj, dict):
        for value in obj.values():
            values.extend(_walk_values(value))
    elif isinstance(obj, list):
        for value in obj:
            values.extend(_walk_values(value))
    else:
        values.append(obj)
    return values


def _load_rcsb_annotations(path: Path) -> tuple[dict[str, set[str]], dict[str, dict[str, Any]]]:
    annotations: dict[str, set[str]] = {}
    metadata: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return annotations, metadata
    for raw_line in path.read_text(errors="ignore").splitlines():
        if not raw_line.strip():
            continue
        try:
            row = json.loads(raw_line)
        except json.JSONDecodeError:
            continue
        chain_id = str(row.get("chain_id") or "").strip()
        if not chain_id:
            continue
        values = [str(value).lower() for value in _walk_values(row)]
        joined = " ".join(values)
        classes: set[str] = set()
        if "cath" in joined:
            classes.add("cath_annotated")
        if "scop" in joined:
            classes.add("scope_annotated")
        if "alpha" in joined and "beta" in joined:
            classes.add("alpha_beta")
        elif "helix" in joined or "alpha" in joined:
            classes.add("alpha")
        elif "sheet" in joined or "strand" in joined or "beta" in joined:
            classes.add("beta")
        for cls in classes:
            _add_source_value(annotations, chain_id, normalize_domain_architecture_class(cls))
        entry = row.get("entry") if isinstance(row.get("entry"), dict) else {}
        entity = row.get("entity") if isinstance(row.get("entity"), dict) else {}
        metadata[chain_id] = {
            "experimental_method": _first_nested_value(entry, ("exptl", "method")),
            "taxonomy_id": _first_nested_value(entity, ("rcsb_entity_source_organism", "ncbi_taxonomy_id")),
            "source_organism": _first_nested_value(entity, ("rcsb_entity_source_organism", "scientific_name")),
        }
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


def _load_external_sources(metadata_sources_dir: Path) -> tuple[dict[str, dict[str, set[str]]], dict[str, dict[str, Any]]]:
    cath = _load_cath_annotations(metadata_sources_dir / "cath-domain-list.txt")
    scope = _load_scope_annotations(metadata_sources_dir / "dir.cla.scope.txt")
    ecod = _load_ecod_annotations(metadata_sources_dir / "ecod.latest.domains.txt")
    rcsb, rcsb_meta = _load_rcsb_annotations(metadata_sources_dir / "rcsb_chain_metadata.jsonl")
    return {"cath": cath, "scope": scope, "ecod": ecod, "rcsb": rcsb}, rcsb_meta


def _select_domain_architecture(
    chain_id: str,
    external_sources: dict[str, dict[str, set[str]]],
    fallback_secondary_class: str,
) -> tuple[str, str, dict[str, list[str]]]:
    values_by_source: dict[str, list[str]] = {}
    for source_name in ("cath", "scope", "ecod", "rcsb"):
        values = sorted(external_sources.get(source_name, {}).get(chain_id, set()))
        if values:
            values_by_source[source_name] = values
    for source_name in ("cath", "scope", "ecod", "rcsb"):
        source_values = values_by_source.get(source_name)
        if source_values:
            chosen = normalize_domain_architecture_class(source_values[0])
            if chosen not in {"unknown", "cath_annotated", "scope_annotated"}:
                return chosen, source_name, values_by_source
    fallback = normalize_domain_architecture_class(fallback_secondary_class)
    return fallback, "secondary_structure_fallback", values_by_source


def _metadata_source_hashes(source_lock: Path) -> dict[str, Any]:
    if not source_lock.exists():
        return {}
    try:
        raw = json.loads(source_lock.read_text())
    except json.JSONDecodeError:
        return {}
    if not isinstance(raw, dict):
        return {}
    return {
        "lock_path": str(source_lock),
        "lock_sha256": _sha256(source_lock),
        "sources": raw.get("sources", {}),
    }


def _sha256(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_metadata(
    *,
    chain_data_cache: Path,
    mmcif_root: Path,
    metadata_sources_dir: Path,
    metadata_source_lock: Path,
    min_len: int,
    max_len: int,
    max_resolution: float,
    max_unknown_aa_fraction: float,
    sample_limit: int = 0,
    fail_fast: bool = False,
) -> dict[str, Any]:
    raw = json.loads(chain_data_cache.read_text())
    if not isinstance(raw, dict):
        raise ValueError(f"chain_data_cache must be a JSON object: {chain_data_cache}")
    candidates, initial_reject_counts = _candidate_rows(
        raw,
        min_len=min_len,
        max_len=max_len,
        max_resolution=max_resolution,
        max_unknown_aa_fraction=max_unknown_aa_fraction,
    )
    if sample_limit > 0:
        candidates = candidates[:sample_limit]

    entries: list[dict[str, Any]] = []
    reject_counts = Counter(initial_reject_counts)
    secondary_counts: Counter[str] = Counter()
    domain_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    external_sources, rcsb_metadata = _load_external_sources(metadata_sources_dir)

    for candidate in tqdm(candidates, desc="structure-metadata"):
        chain_id = str(candidate["chain_id"])
        pdb_id = _pdb_id(chain_id)
        chain_name = _chain_name(chain_id)
        mmcif_path = mmcif_root / f"{pdb_id}.cif"
        dssp_mmcif_path = metadata_sources_dir / "dssp_mmcif" / f"{pdb_id}.cif"
        entry: dict[str, Any] = {
            "chain_id": chain_id,
            "pdb_id": pdb_id,
            "accepted": False,
            "reject_reasons": [],
            "length": int(candidate["length"]),
            "resolution": float(candidate["resolution"]),
        }
        try:
            if not mmcif_path.exists():
                raise FileNotFoundError(f"missing_mmcif:{mmcif_path}")
            atoms = extract_chain_atoms(
                mmcif_path=mmcif_path,
                pdb_id=pdb_id,
                chain_id=chain_name,
                expected_sequence=str(candidate["sequence"]),
                require_full_match=True,
            )
            cls, helix_fraction, beta_fraction, coil_fraction, secondary_source = (
                secondary_fractions_from_mmcif_annotations(
                    str(dssp_mmcif_path),
                    chain_id=chain_name,
                    length=int(candidate["length"]),
                )
                if dssp_mmcif_path.exists()
                else ("unknown", None, None, None, None)
            )
            if cls == "unknown":
                cls, helix_fraction, beta_fraction, coil_fraction, secondary_source = (
                    secondary_fractions_from_mmcif_annotations(
                        str(mmcif_path),
                        chain_id=chain_name,
                        length=int(candidate["length"]),
                    )
                )
            if cls == "unknown":
                cls, helix_fraction, beta_fraction, coil_fraction = secondary_fractions_from_atom14(
                    atoms.atom14_positions,
                    atoms.atom14_mask,
                )
                secondary_source = "atom14_torsion"
            if cls == "unknown" or helix_fraction is None or beta_fraction is None or coil_fraction is None:
                raise ValueError("secondary_structure_unknown")
            domain_class, domain_source, domain_source_values = _select_domain_architecture(
                chain_id,
                external_sources,
                fallback_secondary_class=cls,
            )
            source_coverage = {
                source_name: bool(external_sources.get(source_name, {}).get(chain_id))
                for source_name in ("cath", "scope", "ecod", "rcsb")
            }
            source_coverage["dssp"] = dssp_mmcif_path.exists()
            source_coverage["mmcif_annotations"] = secondary_source == "mmcif_annotations"
            source_coverage["atom14_torsion"] = secondary_source == "atom14_torsion"
            metadata_source_count = sum(1 for present in source_coverage.values() if present)
            entry.update(
                {
                    "accepted": True,
                    "reject_reasons": [],
                    "secondary_structure_class": cls,
                    "secondary_helix_fraction": round(float(helix_fraction), 4),
                    "secondary_beta_fraction": round(float(beta_fraction), 4),
                    "secondary_coil_fraction": round(float(coil_fraction), 4),
                    "secondary_structure_source": secondary_source,
                    "domain_architecture_class": domain_class,
                    "domain_architecture_source": domain_source,
                    "domain_architecture_source_values": domain_source_values,
                    "metadata_source_coverage": source_coverage,
                    "metadata_source_count": metadata_source_count,
                    "rcsb_metadata": rcsb_metadata.get(chain_id, {}),
                    "structure_resolution": float(atoms.resolution),
                }
            )
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
            "mmcif_root": str(mmcif_root),
            "metadata_sources_dir": str(metadata_sources_dir),
            "metadata_source_lock": _metadata_source_hashes(metadata_source_lock),
        },
        "config": {
            "min_len": int(min_len),
            "max_len": int(max_len),
            "max_resolution": float(max_resolution),
            "max_unknown_aa_fraction": float(max_unknown_aa_fraction),
        },
        "summary": {
            "candidate_count": len(candidates),
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
        mmcif_root=args.mmcif_root,
        metadata_sources_dir=args.metadata_sources_dir,
        metadata_source_lock=args.metadata_source_lock,
        min_len=int(args.min_len),
        max_len=int(args.max_len),
        max_resolution=float(args.max_resolution),
        max_unknown_aa_fraction=float(args.max_unknown_aa_fraction),
        sample_limit=int(args.sample_limit),
        fail_fast=bool(args.fail_fast),
    )
    args.metadata_out.parent.mkdir(parents=True, exist_ok=True)
    args.metadata_out.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")

    if args.candidate_manifest_out is not None:
        candidates = [entry["chain_id"] for entry in metadata["chains"]]
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
        if summary["reject_reasons"].get("missing_mmcif", 0):
            print(
                "[structure-metadata] missing mmCIFs detected; use the candidate manifest with "
                "scripts/prepare_data.py --download-mmcif-subset to fetch them."
            )


if __name__ == "__main__":
    main()
