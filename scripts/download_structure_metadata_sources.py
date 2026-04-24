from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

STANDARD_AAS = set("ARNDCQEGHILKMFPSTWYV")

BULK_SOURCES = {
    "cath_domain_list": {
        "url": "https://download.cathdb.info/cath/releases/latest-release/cath-classification-data/cath-domain-list.txt",
        "path": "cath-domain-list.txt",
    },
    "scope_classification": {
        "url": "https://scop.berkeley.edu/downloads/parse/dir.cla.scope.2.08-stable.txt",
        "path": "dir.cla.scope.txt",
    },
    "ecod_domains": {
        "url": "http://prodata.swmed.edu/ecod/distributions/ecod.latest.domains.txt",
        "path": "ecod.latest.domains.txt",
    },
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download and pin structural metadata sources for official split curation.",
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
    parser.add_argument("--download-retries", type=int, default=2)
    parser.add_argument("--download-retry-delay-seconds", type=float, default=2.0)
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
    print("+", url, "->", destination)
    if dry_run:
        return True
    destination.parent.mkdir(parents=True, exist_ok=True)
    attempt = 0
    while True:
        try:
            with urlopen(url, timeout=60) as response:
                destination.write_bytes(response.read())
            return True
        except (HTTPError, URLError, TimeoutError, OSError) as exc:
            attempt += 1
            if attempt <= retries:
                wait_s = max(0.0, retry_delay_seconds) * attempt
                print(f"WARNING: download failed ({exc}), retrying {attempt}/{retries} after {wait_s:.1f}s")
                time.sleep(wait_s)
                continue
            print(f"ERROR: download failed ({exc})")
            return False


def _extract_sequence(meta: dict[str, Any]) -> str | None:
    for key in ("sequence", "seq", "seqres", "aatype_sequence"):
        value = meta.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip().upper()
    return None


def _numeric(value: object, default: float) -> float:
    if not isinstance(value, (str, bytes, bytearray, int, float)):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _unknown_fraction(sequence: str) -> float:
    if not sequence:
        return 1.0
    return sum(1 for ch in sequence if ch not in STANDARD_AAS) / float(len(sequence))


def _candidate_chain_ids(
    chain_data_cache: Path,
    *,
    min_len: int,
    max_len: int,
    max_resolution: float,
    max_unknown_aa_fraction: float,
) -> list[str]:
    raw = json.loads(chain_data_cache.read_text())
    if not isinstance(raw, dict):
        raise ValueError(f"chain_data_cache must be a JSON object: {chain_data_cache}")
    chain_ids: list[str] = []
    for chain_id, meta_raw in sorted(raw.items()):
        if not isinstance(meta_raw, dict):
            continue
        seq = _extract_sequence(meta_raw)
        if not seq:
            continue
        length = int(_numeric(meta_raw.get("seq_length", meta_raw.get("sequence_length", len(seq))), len(seq)))
        resolution = _numeric(meta_raw.get("resolution", 999.0), 999.0)
        oligomeric_count = int(_numeric(meta_raw.get("oligomeric_count", meta_raw.get("num_chains", 1)), 1.0))
        if not (min_len <= length <= max_len):
            continue
        if resolution != 999.0 and resolution > max_resolution:
            continue
        if oligomeric_count != 1:
            continue
        if _unknown_fraction(seq) > max_unknown_aa_fraction:
            continue
        chain_ids.append(str(chain_id))
    return chain_ids


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
            return payload if isinstance(payload, dict) else None, None
        except (HTTPError, URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
            attempt += 1
            if attempt <= retries:
                time.sleep(max(0.0, retry_delay_seconds) * attempt)
                continue
            return None, str(exc)


def _download_rcsb_jsonl(
    chain_ids: list[str],
    destination: Path,
    *,
    dry_run: bool,
    retries: int,
    retry_delay_seconds: float,
) -> dict[str, Any]:
    print("+ RCSB REST metadata ->", destination)
    if dry_run:
        return {"path": str(destination), "chain_count": len(chain_ids), "sha256": None, "errors": 0}
    destination.parent.mkdir(parents=True, exist_ok=True)
    rows: list[str] = []
    errors = 0
    entry_cache: dict[str, dict[str, Any] | None] = {}
    for chain_id in chain_ids:
        pdb_id = _pdb_id(chain_id)
        chain_name = _chain_name(chain_id)
        entry_url = f"https://data.rcsb.org/rest/v1/core/entry/{pdb_id}"
        instance_url = f"https://data.rcsb.org/rest/v1/core/polymer_entity_instance/{pdb_id}/{chain_name}"
        entry = entry_cache.get(pdb_id)
        if pdb_id not in entry_cache:
            entry, error = _read_json_url(entry_url, retries=retries, retry_delay_seconds=retry_delay_seconds)
            entry_cache[pdb_id] = entry
            if error:
                errors += 1
        instance, instance_error = _read_json_url(
            instance_url,
            retries=retries,
            retry_delay_seconds=retry_delay_seconds,
        )
        entity = None
        entity_error = None
        entity_id = None
        if isinstance(instance, dict):
            identifiers = instance.get("rcsb_polymer_entity_instance_container_identifiers")
            if isinstance(identifiers, dict):
                entity_id = identifiers.get("entity_id")
        if entity_id is not None:
            entity_url = f"https://data.rcsb.org/rest/v1/core/polymer_entity/{pdb_id}/{entity_id}"
            entity, entity_error = _read_json_url(entity_url, retries=retries, retry_delay_seconds=retry_delay_seconds)
        if instance_error:
            errors += 1
        if entity_error:
            errors += 1
        rows.append(
            json.dumps(
                {
                    "chain_id": chain_id,
                    "entry_url": entry_url,
                    "instance_url": instance_url,
                    "entry": entry or {},
                    "instance": instance or {},
                    "entity": entity or {},
                    "errors": [err for err in (instance_error, entity_error) if err],
                },
                sort_keys=True,
            )
        )
    destination.write_text("\n".join(rows) + ("\n" if rows else ""))
    return {
        "path": str(destination),
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
    for pdb_id in pdb_ids:
        url = f"https://pdb-redo.eu/dssp/db/{pdb_id}/mmcif"
        destination = destination_dir / f"{pdb_id}.cif"
        if destination.exists():
            continue
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
        "path": str(destination_dir),
        "pdb_count": len(pdb_ids),
        "failed_pdb_count": failures,
        "tree_sha256": None if dry_run else _tree_sha256(destination_dir),
    }


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    chain_ids = _candidate_chain_ids(
        args.chain_data_cache,
        min_len=int(args.min_len),
        max_len=int(args.max_len),
        max_resolution=float(args.max_resolution),
        max_unknown_aa_fraction=float(args.max_unknown_aa_fraction),
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    sources: dict[str, Any] = {}
    failures: list[str] = []

    for name, spec in BULK_SOURCES.items():
        destination = args.out_dir / str(spec["path"])
        ok = _download_url(
            str(spec["url"]),
            destination,
            dry_run=bool(args.dry_run),
            retries=max(0, int(args.download_retries)),
            retry_delay_seconds=max(0.0, float(args.download_retry_delay_seconds)),
        )
        if not ok:
            failures.append(name)
        sources[name] = {
            "url": spec["url"],
            "path": str(destination),
            "sha256": None if args.dry_run or not destination.exists() else _sha256(destination),
        }

    sources["rcsb_chain_metadata"] = _download_rcsb_jsonl(
        chain_ids,
        args.out_dir / "rcsb_chain_metadata.jsonl",
        dry_run=bool(args.dry_run),
        retries=max(0, int(args.download_retries)),
        retry_delay_seconds=max(0.0, float(args.download_retry_delay_seconds)),
    )
    sources["dssp_mmcif"] = _download_dssp_mmcifs(
        chain_ids,
        args.out_dir / "dssp_mmcif",
        dry_run=bool(args.dry_run),
        retries=max(0, int(args.download_retries)),
        retry_delay_seconds=max(0.0, float(args.download_retry_delay_seconds)),
    )

    lock = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "chain_data_cache": {
            "path": str(args.chain_data_cache),
            "sha256": None if args.dry_run else _sha256(args.chain_data_cache),
        },
        "candidate_chain_count": len(chain_ids),
        "sources": sources,
    }
    if args.dry_run:
        print("+ write source lock", args.source_lock)
    else:
        args.source_lock.parent.mkdir(parents=True, exist_ok=True)
        args.source_lock.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n")
        print(f"Wrote metadata source lock: {args.source_lock}")
    if failures:
        raise SystemExit(f"Required metadata source downloads failed: {', '.join(failures)}")


if __name__ == "__main__":
    main()
