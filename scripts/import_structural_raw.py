"""Import selected acquired raw files into a stable append-only qualification pool.

Only an explicit manifest is imported. Resolve each chain through the owning raw
source lookup or a JSONL index containing chain_id/path/sha256, optional bytes,
and optional alias_chain_ids. Index aliases are explicit acquisition identities,
not biological qualification. First-accepted per-chain hashes are retained in
exclusive bindings; existing files and new sources must match those hashes.
Conflicts are fatal. Hardlinks require one filesystem and never change source permissions.
Source owners must avoid in-place writes: hardlinks share an inode, and later
source qualification/replay must still verify the accepted bytes.
"""
from __future__ import annotations

import argparse
import errno
import fcntl
import json
import os
import re
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from nanofold.chain_paths import chain_data_dir, chain_id_to_stem
from nanofold.representative_eligibility import _raw_source, file_record, read_json, require, sha256_file
from nanofold.structural_population import read_rows, validate_file_binding, write_json
from nanofold.structure_metadata import validate_author_chain_id


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def no_symlinks(path: Path) -> None:
    require(not any(part.is_symlink() for part in (path, *path.parents)), "Raw pool output paths must not contain symlinks")


def read_manifest(path: Path) -> tuple[list[str], dict[str, str]]:
    binding = file_record(path)
    chains = [line.strip() for line in path.read_text().splitlines() if line.strip() and not line.lstrip().startswith("#")]
    validate_file_binding(binding)
    require(bool(chains) and len(chains) == len(set(chains)), "Raw import manifest requires unique nonempty chain IDs")
    for chain in chains:
        validate_author_chain_id(chain)
    return sorted(chains), binding


def source_index(paths: list[Path], selected: set[str]) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, str]]]:
    indexed: dict[str, list[dict[str, Any]]] = {}
    pins = []
    for path in paths:
        pin = file_record(path)
        rows = read_rows(path)
        validate_file_binding(pin)
        pins.append(pin)
        for row in rows:
            aliases = row.get("alias_chain_ids", [])
            require(isinstance(aliases, list) and all(isinstance(chain, str) for chain in aliases), "Invalid explicit raw aliases")
            chains = [row["chain_id"], *aliases]
            require(len(chains) == len(set(chains)), "Repeated raw source alias")
            for chain in chains:
                validate_author_chain_id(chain)
            wanted = set(chains) & selected
            if not wanted:
                continue
            require(isinstance(row.get("path"), str) and isinstance(row.get("sha256"), str)
                    and re.fullmatch(r"[a-f0-9]{64}", row["sha256"]) is not None, "Raw index requires an explicit path and SHA256")
            actual = file_record(Path(row["path"]))
            require(actual["sha256"] == row["sha256"], "Indexed raw source bytes changed")
            size = Path(actual["path"]).stat().st_size
            require(size > 0 and ("bytes" not in row or type(row["bytes"]) is int and row["bytes"] == size),
                    "Indexed raw source size is empty or differs from its binding")
            for chain in wanted:
                indexed.setdefault(chain, []).append(actual | {"bytes": size, "declared_path": row["path"],
                    "declared_chain_id": row["chain_id"], "explicit_alias": chain != row["chain_id"]})
    return indexed, pins


def select_sources(chains: list[str], roots: list[Path], indexed: dict[str, list[dict[str, Any]]]) -> dict[str, dict[str, Any]]:
    selected = {}
    for chain in chains:
        candidates = list(indexed.get(chain, []))
        if roots:
            try:
                path = _raw_source(roots, chain)
            except ValueError as error:
                if str(error) != "Required raw query source is missing; acquisition is incomplete":
                    raise
            else:
                require(path.stat().st_size > 0, "Raw import source is empty")
                candidates.append(file_record(path) | {"bytes": path.stat().st_size, "declared_path": str(path),
                                                       "declared_chain_id": chain, "explicit_alias": False})
        require(bool(candidates), "Selected chain has no explicitly acquired raw source")
        require(len({row["sha256"] for row in candidates}) == 1, "Conflicting acquired bytes for selected raw chain")
        selected[chain] = min(candidates, key=lambda row: (row["path"], row["declared_chain_id"]))
    return selected


def publish_link(source: dict[str, Any], destination: Path) -> str:
    """Publish a verified complete inode without replacing any existing path."""
    no_symlinks(destination)
    destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    path = Path(source["path"])
    require(path.stat().st_size == source["bytes"] and sha256_file(path) == source["sha256"], "Raw source changed before import")
    if destination.exists():
        require(destination.is_file() and sha256_file(destination) == source["sha256"], "Conflicting existing raw pool bytes; refusing overwrite")
        return "reused_identical"
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.pending")
    try:
        try:
            os.link(path, temporary, follow_symlinks=False)
        except OSError as error:
            if error.errno == errno.EXDEV:
                raise ValueError("Raw pool imports require source and pool on the same filesystem") from error
            raise
        require(temporary.is_file() and not temporary.is_symlink()
                and temporary.stat().st_size == source["bytes"] and sha256_file(temporary) == source["sha256"],
                "Linked raw source differs from its accepted bytes")
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        try:
            os.link(temporary, destination, follow_symlinks=False)
            action = "linked"
        except FileExistsError:
            no_symlinks(destination)
            require(destination.is_file() and sha256_file(destination) == source["sha256"],
                    "Conflicting concurrent raw pool bytes; refusing overwrite")
            action = "reused_identical"
        require(sha256_file(path) == source["sha256"] and sha256_file(destination) == source["sha256"],
                "Raw source changed during publication; no import receipt can be accepted")
        fsync_directory(destination.parent)
        fsync_directory(destination.parent.parent)
        return action
    finally:
        temporary.unlink(missing_ok=True)


def import_chain(chain: str, source: dict[str, Any], output: Path) -> dict[str, Any]:
    """Preserve the first accepted hash even when hardlinked sources mutate."""
    destination = chain_data_dir(output / "roda_pdb", chain) / "uniref90_hits.a3m"
    acceptance_path = output / "accepted" / f"{chain_id_to_stem(chain)}.json"
    no_symlinks(acceptance_path)
    no_symlinks(destination)
    if acceptance_path.exists():
        accepted = read_json(acceptance_path)
        require(accepted.get("chain_id") == chain and accepted.get("destination", {}).get("path") == str(destination),
                "Raw pool acceptance binding differs from its chain and destination")
        require(destination.is_file() and file_record(destination) == accepted["destination"],
                "Previously accepted raw pool bytes changed")
        require(source["sha256"] == accepted["destination"]["sha256"]
                and source["bytes"] == accepted["bytes"], "Incoming raw source conflicts with the original pool acceptance")
    action = publish_link(source, destination)
    destination_pin = file_record(destination)
    require(destination_pin["sha256"] == source["sha256"], "Raw pool bytes changed before import acceptance")
    if not acceptance_path.exists():
        acceptance_path.parent.mkdir(exist_ok=True, mode=0o700)
        write_json(acceptance_path, {"schema_version": 1, "chain_id": chain, "destination": destination_pin,
                                    "bytes": source["bytes"], "first_source": source, "accepted_at": now()})
        fsync_directory(acceptance_path.parent)
        fsync_directory(output)
    return {"chain_id": chain, "source": source, "destination": destination_pin, "action": action,
            "pool_acceptance": file_record(acceptance_path)}


def run(args: argparse.Namespace) -> dict[str, Any]:
    os.umask(0o077)
    require(bool(args.raw_root or args.source_index), "Provide raw roots or explicit source indexes")
    output = args.output.absolute()
    no_symlinks(output)
    output = output.resolve()
    require(not output.is_relative_to(REPO_ROOT) or output.is_relative_to(REPO_ROOT / ".nanofold_private"),
            "Raw pool and import receipts belong in the ignored private root")
    output.mkdir(parents=True, exist_ok=True, mode=0o700)
    lock_path = output / ".import.lock"
    no_symlinks(lock_path)
    with lock_path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        import_root = output / "imports" / uuid.uuid4().hex
        no_symlinks(import_root)
        import_root.mkdir(parents=True, mode=0o700)
        files_dir = import_root / "files"
        files_dir.mkdir()
        roots = [path.resolve() for path in args.raw_root]
        start = {"status": "running", "started_at": now(), "manifest": str(args.manifest.resolve()),
                 "raw_roots": list(map(str, roots)), "source_indexes": [str(path.resolve()) for path in args.source_index]}
        write_json(import_root / "start.json", start)
        accepted: list[dict[str, Any]] = []
        receipt: dict[str, Any] = start.copy()
        try:
            chains, manifest_pin = read_manifest(args.manifest)
            indexed, index_pins = source_index(args.source_index, set(chains))
            sources = select_sources(chains, roots, indexed)
            pins = [manifest_pin, *index_pins, file_record(Path(__file__)),
                    file_record(REPO_ROOT / "nanofold/representative_eligibility.py"),
                    file_record(REPO_ROOT / "nanofold/chain_paths.py")]
            request = {"schema_version": 1, "manifest": manifest_pin, "source_indexes": index_pins,
                       "input_bindings": pins, "selected_sources": sources, "raw_roots": list(map(str, roots)),
                       "pool_root": str(output), "biological_identity_validated": False}
            write_json(import_root / "request.json", request)
            for chain, source in sources.items():
                record = import_chain(chain, source, output)
                write_json(files_dir / f"{chain_id_to_stem(chain)}.json", record)
                fsync_directory(files_dir)
                accepted.append(record)
            for pin in pins:
                validate_file_binding(pin)
            for record in accepted:
                validate_file_binding(record["pool_acceptance"])
                validate_file_binding(record["destination"])
                require(sha256_file(Path(record["source"]["path"])) == record["source"]["sha256"],
                        "Accepted raw source changed during this import")
            receipt.update(status="imported_unqualified", biological_identity_validated=False, pool_root=str(output),
                           request=file_record(import_root / "request.json"), imported_chain_count=len(accepted), files=accepted,
                           hardlink_mutability="source and destination share bytes; future source replay remains required")
        except BaseException as error:
            receipt.update(status="failed_or_interrupted", files=accepted, imported_chain_count=len(accepted),
                           error_type=type(error).__name__, error=str(error))
            raise
        finally:
            receipt["finished_at"] = now()
            write_json(import_root / "receipt.json", receipt)
            fsync_directory(import_root)
            fsync_directory(import_root.parent)
            fsync_directory(output)
    return receipt | {"receipt": file_record(import_root / "receipt.json")}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True, help="Explicit selected chain IDs; no full source scan.")
    parser.add_argument("--raw-root", type=Path, action="append", default=[])
    parser.add_argument("--source-index", type=Path, action="append", default=[], help="JSONL explicit file bindings and optional aliases.")
    parser.add_argument("--output", type=Path, required=True, help="Stable raw pool, shared by all acquisition waves.")
    return parser.parse_args(argv)


def main() -> int:
    result = run(parse_args())
    print(json.dumps({key: value for key, value in result.items() if key != "files"}, sort_keys=True))
    return 0


if __name__ == "__main__":
    main()
