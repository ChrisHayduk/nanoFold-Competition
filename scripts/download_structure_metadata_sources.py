from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from build_structure_metadata import _candidate_rows

from nanofold.cath_boundaries import CATH_BOUNDARY_SOURCES

BULK_SOURCES = {
    "cath_domain_list": {
        "url": "https://download.cathdb.info/cath/releases/all-releases/v4_4_0/cath-classification-data/cath-domain-list-v4_4_0.txt",
        "path": "cath-domain-list.txt",
        "version": "4.4.0",
    },
    "scope_classification": {
        "url": "http://scop.berkeley.edu/downloads/parse/dir.cla.scope.2.08-stable.txt",
        "path": "dir.cla.scope.txt",
    },
    "ecod_domains": {
        "url": "http://prodata.swmed.edu/ecod/distributions/ecod.latest.domains.txt",
        "path": "ecod.latest.domains.txt",
    },
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download and pin structural metadata sources for official split curation. "
            "Existing locks are verified without repinning; use a fresh output directory and lock for new sources. "
            "CATH is required; optional source failures are recorded in the lock."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--chain-data-cache", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("data/metadata_sources"))
    parser.add_argument(
        "--source-lock",
        type=Path,
        default=Path("data/metadata_sources/structure_metadata_sources.lock.json"),
    )
    parser.add_argument("--min-len", type=int, default=40)
    parser.add_argument("--max-len", type=int, default=256)
    parser.add_argument("--max-resolution", type=float, default=3.0)
    parser.add_argument("--max-unknown-aa-fraction", type=float, default=0.0)
    parser.add_argument(
        "--require-monomer", action="store_true", help="Require an explicitly documented biological monomer."
    )
    parser.add_argument("--download-retries", type=int, default=2)
    parser.add_argument("--download-retry-delay-seconds", type=float, default=2.0)
    parser.add_argument("--include-rcsb-rest", action="store_true")
    parser.add_argument("--include-dssp-mmcif", action="store_true")
    parser.add_argument(
        "--include-cath-boundaries", action="store_true",
        help="Also require CATH 4.4 SEQRES domain boundaries and full domain sequences for exposure metadata.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def _sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _tree_sha256(root: Path) -> str | None:
    if not root.exists():
        return None
    files = sorted(path for path in root.rglob("*") if path.is_file())
    if not files:
        return None
    hasher = hashlib.sha256()
    for path in files:
        rel = str(path.relative_to(root)).replace("\\", "/")
        hasher.update(rel.encode("utf-8"))
        hasher.update(b"\t")
        hasher.update(_sha256(path).encode("ascii"))
        hasher.update(b"\n")
    return hasher.hexdigest()


def _download_url(
    url: str,
    destination: Path,
    *,
    dry_run: bool,
    retries: int,
    retry_delay_seconds: float,
) -> bool:
    print("+", url, "->", destination, flush=True)
    if dry_run:
        return True
    if destination.exists():
        raise ValueError(f"Refusing to reuse an unverified source file: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    attempt = 0
    while True:
        try:
            with urlopen(url, timeout=60) as response:
                tmp = destination.with_suffix(destination.suffix + ".tmp")
                contents = response.read()
                if not contents:
                    raise ValueError(f"Empty source response: {url}")
                tmp.write_bytes(contents)
                tmp.replace(destination)
            return True
        except (HTTPError, URLError, TimeoutError, OSError, ValueError) as exc:
            attempt += 1
            if attempt <= retries:
                wait_s = max(0.0, retry_delay_seconds) * attempt
                print(f"WARNING: download failed ({exc}), retrying {attempt}/{retries} after {wait_s:.1f}s", flush=True)
                time.sleep(wait_s)
                continue
            print(f"ERROR: download failed ({exc})", flush=True)
            return False


def _candidate_chain_ids(
    chain_data_cache: Path,
    *,
    min_len: int,
    max_len: int,
    max_resolution: float,
    max_unknown_aa_fraction: float,
    require_monomer: bool = False,
) -> list[str]:
    raw = json.loads(chain_data_cache.read_text())
    if not isinstance(raw, dict):
        raise ValueError(f"chain_data_cache must be a JSON object: {chain_data_cache}")
    rows, _ = _candidate_rows(
        raw,
        min_len=min_len,
        max_len=max_len,
        max_resolution=max_resolution,
        max_unknown_aa_fraction=max_unknown_aa_fraction,
        require_monomer=require_monomer,
    )
    return [str(row["chain_id"]) for row in rows]


def _pdb_id(chain_id: str) -> str:
    return chain_id.split("_", 1)[0].lower()


def _chain_name(chain_id: str) -> str:
    return chain_id.split("_", 1)[1]


def _read_json_url(url: str, *, retries: int, retry_delay_seconds: float) -> tuple[dict[str, Any] | None, str | None]:
    attempt = 0
    while True:
        try:
            with urlopen(url, timeout=30) as response:
                payload = json.loads(response.read().decode("utf-8"))
            if not isinstance(payload, dict):
                raise ValueError(f"Expected JSON object from {url}")
            return payload, None
        except (HTTPError, URLError, TimeoutError, OSError, ValueError, UnicodeError) as exc:
            attempt += 1
            if attempt <= retries:
                time.sleep(max(0.0, retry_delay_seconds) * attempt)
                continue
            return None, str(exc)


def _rcsb_entry_chain_map(
    pdb_id: str, *, retries: int, retry_delay_seconds: float,
) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]], list[str]]:
    """Resolve author chains through all polymer entities and label-asym instances."""
    entry, error = _read_json_url(
        f"https://data.rcsb.org/rest/v1/core/entry/{pdb_id}", retries=retries, retry_delay_seconds=retry_delay_seconds,
    )
    if error or entry is None:
        return {}, {}, [error or "Missing RCSB entry"]
    identifiers = entry.get("rcsb_entry_container_identifiers", {})
    entity_ids = identifiers.get("polymer_entity_ids") if isinstance(identifiers, dict) else None
    if not isinstance(entity_ids, list) or not entity_ids or any(not isinstance(item, str) for item in entity_ids):
        return entry, {}, ["Missing RCSB polymer entity identifiers"]
    if len(entity_ids) != len(set(entity_ids)):
        return entry, {}, ["Duplicate RCSB polymer entity identifiers"]
    mapped: dict[str, list[dict[str, Any]]] = {}
    errors: list[str] = []
    seen_asym_ids: set[str] = set()
    for entity_id in entity_ids:
        entity, error = _read_json_url(
            f"https://data.rcsb.org/rest/v1/core/polymer_entity/{pdb_id}/{entity_id}", retries=retries, retry_delay_seconds=retry_delay_seconds,
        )
        if error or entity is None:
            errors.append(error or "Missing RCSB polymer entity")
            continue
        entity_identifiers = entity.get("rcsb_polymer_entity_container_identifiers", {})
        asym_ids = entity_identifiers.get("asym_ids") if isinstance(entity_identifiers, dict) else None
        if not isinstance(asym_ids, list) or not asym_ids or any(not isinstance(item, str) for item in asym_ids):
            errors.append("Missing RCSB polymer instance identifiers")
            continue
        for asym_id in asym_ids:
            if asym_id in seen_asym_ids:
                errors.append("Duplicate RCSB polymer instance identifiers")
                continue
            seen_asym_ids.add(asym_id)
            instance_url = f"https://data.rcsb.org/rest/v1/core/polymer_entity_instance/{pdb_id}/{asym_id}"
            instance, error = _read_json_url(instance_url, retries=retries, retry_delay_seconds=retry_delay_seconds)
            if error or instance is None:
                errors.append(error or "Missing RCSB polymer instance")
                continue
            instance_identifiers = instance.get("rcsb_polymer_entity_instance_container_identifiers", {})
            if not isinstance(instance_identifiers, dict):
                errors.append("Missing RCSB instance chain mapping")
                continue
            author_chain = instance_identifiers.get("auth_asym_id")
            if (
                not isinstance(author_chain, str) or not author_chain
                or instance_identifiers.get("asym_id") != asym_id
                or instance_identifiers.get("entity_id") != entity_id
                or str(instance_identifiers.get("entry_id", "")).lower() != pdb_id
            ):
                errors.append("Unverified author-chain mapping in RCSB instance")
                continue
            mapped.setdefault(author_chain, []).append({
                "instance_url": instance_url, "instance": instance, "entity": entity,
            })
    return entry, mapped, errors


def _download_rcsb_jsonl(
    chain_ids: list[str],
    destination: Path,
    *,
    dry_run: bool,
    retries: int,
    retry_delay_seconds: float,
) -> dict[str, Any]:
    print("+ RCSB REST metadata ->", destination, flush=True)
    if dry_run:
        return {"path": destination.name, "chain_count": len(chain_ids), "sha256": None, "errors": 0}
    if destination.exists():
        raise ValueError(f"Refusing to reuse unverified RCSB metadata: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    errors = 0
    entry_cache: dict[str, tuple[dict[str, Any], dict[str, list[dict[str, Any]]], list[str]]] = {}
    tmp = destination.with_suffix(destination.suffix + ".tmp")
    with tmp.open("w") as handle:
        for idx, chain_id in enumerate(chain_ids, start=1):
            if idx == 1 or idx % 1000 == 0:
                print(f"  RCSB metadata {idx}/{len(chain_ids)}", flush=True)
            pdb_id = _pdb_id(chain_id)
            if pdb_id not in entry_cache:
                entry_cache[pdb_id] = _rcsb_entry_chain_map(
                    pdb_id, retries=retries, retry_delay_seconds=retry_delay_seconds,
                )
                errors += len(entry_cache[pdb_id][2])
            entry, mapped, entry_errors = entry_cache[pdb_id]
            matches = mapped.get(_chain_name(chain_id), [])
            row_errors = list(entry_errors)
            if not row_errors and len(matches) != 1:
                row_errors.append("No unique RCSB author-chain mapping")
                errors += 1
            selected = matches[0] if not row_errors and len(matches) == 1 else {}
            handle.write(
                json.dumps(
                    {
                        "chain_id": chain_id,
                        "entry_url": f"https://data.rcsb.org/rest/v1/core/entry/{pdb_id}",
                        "instance_url": selected.get("instance_url"),
                        "entry": entry,
                        "instance": selected.get("instance", {}),
                        "entity": selected.get("entity", {}),
                        "errors": row_errors,
                    },
                    sort_keys=True,
                )
                + "\n"
            )
    tmp.replace(destination)
    return {
        "path": destination.name,
        "chain_count": len(chain_ids),
        "sha256": _sha256(destination),
        "errors": errors,
    }


def _download_dssp_mmcifs(
    chain_ids: list[str],
    destination_dir: Path,
    *,
    dry_run: bool,
    retries: int,
    retry_delay_seconds: float,
) -> dict[str, Any]:
    pdb_ids = sorted({_pdb_id(chain_id) for chain_id in chain_ids})
    failures = 0
    for idx, pdb_id in enumerate(pdb_ids, start=1):
        if idx == 1 or idx % 1000 == 0:
            print(f"  DSSP mmCIF {idx}/{len(pdb_ids)}", flush=True)
        url = f"https://pdb-redo.eu/dssp/db/{pdb_id}/mmcif"
        destination = destination_dir / f"{pdb_id}.cif"
        ok = _download_url(
            url,
            destination,
            dry_run=dry_run,
            retries=retries,
            retry_delay_seconds=retry_delay_seconds,
        )
        if not ok:
            failures += 1
    return {
        "path": destination_dir.name,
        "pdb_count": len(pdb_ids),
        "failed_pdb_count": failures,
        "tree_sha256": None if dry_run else _tree_sha256(destination_dir),
    }


def _source_paths() -> dict[str, str]:
    return {
        **{name: spec["path"] for name, spec in BULK_SOURCES.items()},
        **{name: spec["path"] for name, spec in CATH_BOUNDARY_SOURCES.items()},
        "rcsb_chain_metadata": "rcsb_chain_metadata.jsonl",
        "dssp_mmcif": "dssp_mmcif",
    }


def _verify_existing_lock(args: argparse.Namespace, candidate_filters: dict[str, Any], chain_count: int) -> None:
    lock = json.loads(args.source_lock.read_text())
    if not isinstance(lock, dict) or lock.get("schema_version") != 2:
        raise ValueError("Expected structural metadata source lock schema_version=2; use a fresh directory and lock.")
    cache = lock.get("chain_data_cache")
    if not isinstance(cache, dict) or cache.get("sha256") != _sha256(args.chain_data_cache):
        raise ValueError("chain_data_cache hash mismatch; refusing to repin changed input.")
    if lock.get("candidate_filters") != candidate_filters or lock.get("candidate_chain_count") != chain_count:
        raise ValueError("Candidate selection differs from the source lock; use a fresh directory and lock.")
    options = {"include_rcsb_rest": args.include_rcsb_rest, "include_dssp_mmcif": args.include_dssp_mmcif}
    if args.include_cath_boundaries:
        options["include_cath_boundaries"] = True
    if lock.get("download_options") != options:
        raise ValueError("Download options differ from the source lock; use a fresh directory and lock.")
    sources = lock.get("sources")
    if not isinstance(sources, dict) or set(sources) - _source_paths().keys():
        raise ValueError("Invalid sources in structural metadata lock.")
    cath = sources.get("cath_domain_list")
    if not isinstance(cath, dict) or cath.get("skipped"):
        raise ValueError("Source lock must include the required CATH domain list.")
    if args.include_cath_boundaries:
        for name in CATH_BOUNDARY_SOURCES:
            if not isinstance(sources.get(name), dict) or sources[name].get("skipped"):
                raise ValueError(f"Source lock must include the requested CATH boundary source: {name}")
    for name, source in sources.items():
        if not isinstance(source, dict) or source.get("path") != _source_paths()[name]:
            raise ValueError(f"Invalid source path in lock: {name}")
        if source.get("skipped"):
            continue
        path = args.out_dir / source["path"]
        digest_key = "tree_sha256" if name == "dssp_mmcif" else "sha256"
        digest = source.get(digest_key)
        if not isinstance(digest, str) or len(digest) != 64:
            raise ValueError(f"Missing source checksum: {name}")
        if not path.exists() or path.is_symlink():
            raise ValueError(f"Missing or invalid source: {name}")
        if name == "dssp_mmcif":
            if not path.is_dir():
                raise ValueError(f"Expected a source directory: {name}")
            actual = _tree_sha256(path)
        else:
            if not path.is_file():
                raise ValueError(f"Expected a source file: {name}")
            actual = _sha256(path)
        if actual != digest:
            raise ValueError(f"Source hash mismatch: {name}; refusing to repin changed bytes.")


def _collect_sources(args: argparse.Namespace, chain_ids: list[str], destination_dir: Path) -> dict[str, Any]:
    sources: dict[str, Any] = {}
    requested = {**BULK_SOURCES, **(CATH_BOUNDARY_SOURCES if args.include_cath_boundaries else {})}
    for name, spec in requested.items():
        destination = destination_dir / spec["path"]
        ok = _download_url(
            str(spec["url"]),
            destination,
            dry_run=bool(args.dry_run),
            retries=max(0, int(args.download_retries)),
            retry_delay_seconds=max(0.0, float(args.download_retry_delay_seconds)),
        )
        sources[name] = {
            **spec,
            "sha256": None if args.dry_run or not destination.exists() else _sha256(destination),
        }
        if not ok:
            if name == "cath_domain_list" or name in CATH_BOUNDARY_SOURCES:
                raise SystemExit(f"Required metadata source download failed: {name}; no lock was published.")
            sources[name].update({"skipped": True, "errors": 1, "reason": "optional source download failed"})

    if args.include_rcsb_rest:
        sources["rcsb_chain_metadata"] = _download_rcsb_jsonl(
            chain_ids,
            destination_dir / "rcsb_chain_metadata.jsonl",
            dry_run=bool(args.dry_run),
            retries=max(0, int(args.download_retries)),
            retry_delay_seconds=max(0.0, float(args.download_retry_delay_seconds)),
        )
    else:
        sources["rcsb_chain_metadata"] = {
            "path": "rcsb_chain_metadata.jsonl",
            "chain_count": len(chain_ids),
            "sha256": None,
            "skipped": True,
            "reason": "bulk split metadata does not require per-chain REST calls",
        }
    if args.include_dssp_mmcif:
        sources["dssp_mmcif"] = _download_dssp_mmcifs(
            chain_ids,
            destination_dir / "dssp_mmcif",
            dry_run=bool(args.dry_run),
            retries=max(0, int(args.download_retries)),
            retry_delay_seconds=max(0.0, float(args.download_retry_delay_seconds)),
        )
        if not args.dry_run and not sources["dssp_mmcif"]["tree_sha256"]:
            sources["dssp_mmcif"].update({"skipped": True, "reason": "no DSSP source downloads succeeded"})
    else:
        sources["dssp_mmcif"] = {
            "path": "dssp_mmcif",
            "pdb_count": len({_pdb_id(chain_id) for chain_id in chain_ids}),
            "failed_pdb_count": None,
            "tree_sha256": None,
            "skipped": True,
            "reason": "selected-chain mmCIF annotations are derived during preprocessing",
        }
    return sources


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    cache_digest = _sha256(args.chain_data_cache)
    candidate_filters = {
        "min_len": args.min_len,
        "max_len": args.max_len,
        "max_resolution": args.max_resolution,
        "max_unknown_aa_fraction": args.max_unknown_aa_fraction,
        "require_monomer": args.require_monomer,
    }
    chain_ids = _candidate_chain_ids(args.chain_data_cache, **candidate_filters)
    if _sha256(args.chain_data_cache) != cache_digest:
        raise ValueError("Chain data cache changed during candidate selection.")
    print(
        f"Metadata source candidate scope: {len(chain_ids)} chains, "
        f"{len({_pdb_id(chain_id) for chain_id in chain_ids})} PDB entries",
        flush=True,
    )
    if not chain_ids:
        raise ValueError("No eligible chains for structural metadata download.")
    if args.source_lock.exists():
        _verify_existing_lock(args, candidate_filters, len(chain_ids))
        print(f"Verified existing metadata source lock: {args.source_lock}")
        return
    source_paths = [args.out_dir / path for path in _source_paths().values()]
    if args.source_lock.resolve() in {args.chain_data_cache.resolve(), *(path.resolve() for path in source_paths)}:
        raise ValueError("Source lock must not overwrite a source or chain data cache.")
    if args.dry_run:
        _collect_sources(args, chain_ids, args.out_dir)
        print("+ write source lock", args.source_lock)
        return
    for path in source_paths:
        if path.exists() or path.is_symlink():
            raise ValueError(f"Unpinned source already exists: {path}; use a fresh directory and lock.")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".structure-metadata-", dir=args.out_dir) as staging_dir:
        staging = Path(staging_dir)
        sources = _collect_sources(args, chain_ids, staging)
        if _sha256(args.chain_data_cache) != cache_digest:
            raise ValueError("Chain data cache changed during source download; no lock was published.")
        lock: dict[str, Any] = {
            "schema_version": 2,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "chain_data_cache": {"path": args.chain_data_cache.name, "sha256": cache_digest},
            "candidate_chain_count": len(chain_ids),
            "candidate_filters": candidate_filters,
            "download_options": {
                "include_rcsb_rest": args.include_rcsb_rest,
                "include_dssp_mmcif": args.include_dssp_mmcif,
            },
            "sources": sources,
        }
        if args.include_cath_boundaries:
            lock["download_options"]["include_cath_boundaries"] = True
        for source in sources.values():
            if not source.get("skipped"):
                shutil.move(str(staging / source["path"]), str(args.out_dir / source["path"]))
        args.source_lock.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(mode="w", dir=args.source_lock.parent, delete=False) as handle:
            temp_lock = Path(handle.name)
            try:
                handle.write(json.dumps(lock, indent=2, sort_keys=True) + "\n")
                handle.close()
                temp_lock.replace(args.source_lock)
            finally:
                temp_lock.unlink(missing_ok=True)
        print(f"Wrote metadata source lock: {args.source_lock}")


if __name__ == "__main__":
    main()
