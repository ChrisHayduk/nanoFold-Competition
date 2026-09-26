"""Acquire and prepare a source-locked OpenProteinSet snapshot without deleting inputs."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
import tempfile
import unicodedata
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any

DOWNLOADS = (
    ("s3://openfold/pdb/", "alignment_data/alignment_dir_roda", True),
    ("s3://openfold/pdb_mmcif.zip", "pdb_data/pdb_mmcif.zip", False),
    ("s3://openfold/duplicate_pdb_chains.txt", "pdb_data/duplicate_pdb_chains.txt", False),
    ("s3://openfold/data_caches/", "pdb_data/data_caches", True),
)
CHAIN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*\Z")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _relative_path(value: str) -> PurePosixPath:
    path = PurePosixPath(value)
    if not value or "\\" in value or path.is_absolute() or any(part in {"", ".", ".."} for part in value.split("/")):
        raise ValueError(f"Unsafe relative path: {value!r}")
    return path


def _file_identity(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    return {"sha256": digest, "size_bytes": path.stat().st_size}


def _files(root: Path) -> list[Path]:
    if root.is_symlink() or not root.is_dir():
        raise ValueError(f"Expected a real directory: {root}")
    files = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink() or not (path.is_dir() or path.is_file()):
            raise ValueError(f"Source tree contains a symlink or special file: {path}")
        if path.is_file():
            files.append(path)
    return files


def read_lock(path: Path) -> tuple[dict[str, dict[str, Any]], str]:
    raw = path.read_bytes()
    lock = json.loads(raw, object_pairs_hook=_unique_object)
    if not isinstance(lock, dict) or set(lock) != {"schema_version", "files"} or type(lock["schema_version"]) is not int or lock["schema_version"] != 1:
        raise ValueError("Source lock must contain schema_version=1 and files")
    files = lock["files"]
    if not isinstance(files, dict) or not files:
        raise ValueError("Source lock files must be a nonempty object")
    for name, expected in files.items():
        _relative_path(name)
        if not any(name.startswith(prefix + "/") if recursive else name == prefix for _, prefix, recursive in DOWNLOADS):
            raise ValueError(f"Source lock path is outside the supported downloads: {name}")
        if not isinstance(expected, dict) or set(expected) != {"sha256", "size_bytes"}:
            raise ValueError(f"Source lock requires sha256 and size_bytes for {name}")
        if not isinstance(expected["sha256"], str) or not re.fullmatch(r"[0-9a-f]{64}", expected["sha256"]):
            raise ValueError(f"Invalid SHA256 for {name}")
        if type(expected["size_bytes"]) is not int or expected["size_bytes"] < 0:
            raise ValueError(f"Invalid byte size for {name}")
    for _, prefix, recursive in DOWNLOADS:
        if not any(name.startswith(prefix + "/") if recursive else name == prefix for name in files):
            raise ValueError(f"Source lock has no files for {prefix}")
    _check_destinations([PurePosixPath(name) for name in files])
    return files, hashlib.sha256(raw).hexdigest()


def verify_sources(root: Path, expected: dict[str, dict[str, Any]]) -> None:
    actual = {str(path.relative_to(root)): _file_identity(path) for path in _files(root)}
    if actual != expected:
        missing = sorted(set(expected) - actual.keys())
        extra = sorted(set(actual) - expected.keys())
        changed = sorted(name for name in actual.keys() & expected.keys() if actual[name] != expected[name])
        raise ValueError(f"Source lock mismatch: missing={missing}, extra={extra}, changed={changed}")


def _check_destinations(paths: list[PurePosixPath], directories: list[PurePosixPath] | None = None) -> None:
    """Validate both explicit and implied directories before creating outputs."""
    def key(path: PurePosixPath) -> str:
        # macOS can equate Unicode-normalized spellings as well as case.
        return unicodedata.normalize("NFC", str(path)).casefold()

    directories = directories or []
    files: set[str] = set()
    directory_keys: set[str] = set()
    explicit: set[str] = set()
    spellings: dict[str, str] = {}
    for path, is_directory in [(path, False) for path in paths] + [(path, True) for path in directories]:
        for prefix in [*path.parents, path]:
            canonical = key(prefix)
            if canonical in spellings and spellings[canonical] != str(prefix):
                raise ValueError(f"Case-colliding output path: {path}")
            spellings[canonical] = str(prefix)
        canonical = key(path)
        if canonical in explicit:
            raise ValueError(f"Output path collision: {path}")
        explicit.add(canonical)
        directory_keys.update(key(parent) for parent in path.parents)
        if is_directory:
            directory_keys.add(canonical)
        else:
            files.add(canonical)
    if files.intersection(directory_keys):
        raise ValueError(f"File/directory output path collision: {sorted(files.intersection(directory_keys))}")


def extract_mmcif(archive: Path, output: Path) -> None:
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Refusing existing extraction directory: {output}")
    with zipfile.ZipFile(archive) as source:
        members = []
        directories = []
        names = set()
        for member in source.infolist():
            path = _relative_path(member.filename[:-1] if member.is_dir() else member.filename)
            mode = member.external_attr >> 16
            if path.parts[0] != "mmcif_files" or stat.S_ISLNK(mode) or (stat.S_IFMT(mode) not in {0, stat.S_IFREG, stat.S_IFDIR}):
                raise ValueError(f"Unsupported mmCIF archive member: {member.filename}")
            if (member.is_dir() and (stat.S_ISREG(mode) or member.file_size != 0)) or (not member.is_dir() and stat.S_ISDIR(mode)):
                raise ValueError(f"Inconsistent archive file/directory metadata: {member.filename}")
            if str(path).casefold() in names:
                raise ValueError(f"Duplicate archive member: {member.filename}")
            names.add(str(path).casefold())
            if member.is_dir():
                directories.append(path)
            else:
                if len(path.parts) < 2:
                    raise ValueError("mmcif_files must be an archive directory")
                members.append((member, path))
        if not members:
            raise ValueError("mmCIF archive contains no files")
        _check_destinations([path for _, path in members], directories)
        output.mkdir(parents=True)
        for member, path in members:
            destination = output.joinpath(*path.parts)
            destination.parent.mkdir(parents=True, exist_ok=True)
            with source.open(member) as incoming, destination.open("xb") as outgoing:
                shutil.copyfileobj(incoming, outgoing)


def _copy_checked(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as incoming, destination.open("xb") as outgoing:
        shutil.copyfileobj(incoming, outgoing)
    if _file_identity(source) != _file_identity(destination):
        raise ValueError(f"Copied bytes differ from source: {source}")


def flatten(source: Path, output: Path) -> None:
    if source.resolve() == output.resolve() or source.resolve() in output.resolve().parents:
        raise ValueError("Flatten output must be outside the source tree")
    plan: list[tuple[Path, PurePosixPath]] = []
    for path in _files(source):
        relative = path.relative_to(source)
        if len(relative.parts) < 3 or not CHAIN.fullmatch(relative.parts[0]):
            raise ValueError(f"Unexpected RODA alignment path: {relative}")
        chain, provider, *rest = relative.parts
        relative_output = PurePosixPath("data", *rest) if provider in {"pdb", "cif"} else PurePosixPath("alignments", chain, *rest)
        plan.append((path, relative_output))
    if not plan:
        raise ValueError("RODA alignment source is empty")
    _check_destinations([destination for _, destination in plan])
    if output.is_symlink():
        raise ValueError("Flatten output may not be a symlink")
    for name in ("data", "alignments"):
        destination = output / name
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(f"Refusing existing flatten output: {destination}")
    output.mkdir(parents=True, exist_ok=True)
    for path, relative_destination in plan:
        destination = output.joinpath(*relative_destination.parts)
        _copy_checked(path, destination)


def expand_duplicates(source: Path, alignments: Path) -> None:
    groups = []
    seen: set[str] = set()
    for line in source.read_text().splitlines():
        chains = line.split()
        if not chains:
            continue
        if any(not CHAIN.fullmatch(chain) for chain in chains) or len(set(chains)) != len(chains) or seen.intersection(chains):
            raise ValueError("Duplicate-chain groups contain an unsafe or repeated chain ID")
        seen.update(chains)
        representatives = [chain for chain in chains if (alignments / chain).is_dir() and not (alignments / chain).is_symlink()]
        if not representatives:
            raise ValueError("Duplicate-chain group has no downloaded representative")
        groups.append((chains, representatives[0]))
    for chains, representative in groups:
        for chain in chains:
            path = alignments / chain
            if path.is_symlink():
                raise ValueError(f"Unexpected existing duplicate symlink: {chain}")
            if not path.exists():
                path.symlink_to(representative, target_is_directory=True)


def _outputs(root: Path) -> dict[str, dict[str, Any]]:
    result = {}
    for path in sorted(root.rglob("*")):
        if path.parts[len(root.parts)] == "sources":
            continue
        if path.is_symlink():
            path.resolve(strict=True).relative_to((root / "alignment_data" / "alignments").resolve())
            result[str(path.relative_to(root))] = {"symlink": os.readlink(path)}
        elif path.is_file():
            result[str(path.relative_to(root))] = _file_identity(path)
    return result


def setup(data_root: Path, source_lock: Path) -> None:
    expected, lock_hash = read_lock(source_lock)
    if data_root.is_symlink():
        raise ValueError("Data root may not be a symlink")
    # Exclusivity prevents mixing old and new snapshots, including reruns after failure.
    data_root.mkdir(parents=True, exist_ok=False)
    sources = data_root / "sources"
    sources.mkdir()
    for uri, relative, recursive in DOWNLOADS:
        destination = sources / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if recursive:
            destination.mkdir()
        command = ["aws", "s3", "cp", uri, str(destination) + ("/" if recursive else ""), "--no-sign-request"]
        if recursive:
            command.append("--recursive")
        subprocess.run(command, check=True)
    prepare_sources(sources, data_root, expected, lock_hash)


def prepare_sources(sources: Path, data_root: Path, expected: dict[str, dict[str, Any]], lock_hash: str) -> None:
    verify_sources(sources, expected)
    extract_mmcif(sources / "pdb_data/pdb_mmcif.zip", data_root / "pdb_data")
    flatten(sources / "alignment_data/alignment_dir_roda", data_root / "alignment_data")
    expand_duplicates(sources / "pdb_data/duplicate_pdb_chains.txt", data_root / "alignment_data/alignments")
    _copy_checked(sources / "pdb_data/duplicate_pdb_chains.txt", data_root / "pdb_data/duplicate_pdb_chains.txt")
    for path in _files(sources / "pdb_data/data_caches"):
        _copy_checked(path, data_root / path.relative_to(sources))
    outputs = _outputs(data_root)
    verify_sources(sources, expected)
    receipt = {
        "schema_version": 1, "status": "complete", "source_lock_sha256": lock_hash,
        "preparation_source_sha256": _file_identity(Path(__file__))["sha256"],
        "source_directory": str(sources.resolve()),
        "source_objects": expected, "source_locations": [uri for uri, _, _ in DOWNLOADS], "outputs": outputs,
    }
    with tempfile.NamedTemporaryFile(mode="w", prefix=".receipt-", dir=data_root, delete=False) as handle:
        json.dump(receipt, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
        temporary = Path(handle.name)
    os.replace(temporary, data_root / "source_receipt.json")
    print(f"Prepared source-locked OpenProteinSet snapshot: {data_root}")
    print(f"Receipt: {data_root / 'source_receipt.json'}")


def prepare_local(sources: Path, data_root: Path, source_lock: Path) -> None:
    expected, lock_hash = read_lock(source_lock)
    if data_root.is_symlink() or sources.resolve() == data_root.resolve() or sources.resolve() in data_root.resolve().parents:
        raise ValueError("Prepared output must be separate from the source tree")
    verify_sources(sources, expected)
    data_root.mkdir(parents=True, exist_ok=False)
    prepare_sources(sources, data_root, expected, lock_hash)


def propose_lock(sources: Path, output: Path) -> None:
    """Record observed bytes for review; this does not accept or prepare a dataset."""
    if sources.resolve() == output.resolve() or sources.resolve() in output.resolve().parents:
        raise ValueError("Write proposed locks outside the source tree")
    provenance = output.with_name(output.name + ".provenance.json")
    if output.exists() or provenance.exists() or output.is_symlink() or provenance.is_symlink():
        raise FileExistsError("Proposed lock/provenance output already exists")
    observed = {str(path.relative_to(sources)): _file_identity(path) for path in _files(sources)}
    payload = {"schema_version": 1, "files": observed}
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as temporary:
        candidate = Path(temporary) / "lock.json"
        candidate.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        read_lock(candidate)
        payload["status"] = "unverified_unaccepted"
        proposed_bytes = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
        digest = hashlib.sha256(proposed_bytes).hexdigest()
        with output.open("xb") as handle:
            handle.write(proposed_bytes)
    with provenance.open("x") as handle:
        json.dump({"schema_version": 1, "status": "unverified_unaccepted", "source_directory": str(sources.resolve()),
                   "proposed_lock_sha256": digest,
                   "scope": "Observed local bytes only; no independent authenticity or dataset acceptance verified."},
                  handle, indent=2, sort_keys=True)
        handle.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    acquire = commands.add_parser("setup", help="Download and prepare a new source-locked snapshot")
    acquire.add_argument("data_root", type=Path, help="New, nonexistent output directory; partial inputs are retained on failure")
    acquire.add_argument("--source-lock", type=Path, required=True, help="Prespecified schema-v1 exact source SHA256/size lock")
    local = commands.add_parser("prepare-local", help="Verify a supplied frozen lock and prepare existing downloads offline")
    local.add_argument("--sources", type=Path, required=True)
    local.add_argument("--data-root", type=Path, required=True)
    local.add_argument("--source-lock", type=Path, required=True)
    proposal = commands.add_parser("propose-lock", help="Hash an existing download for review without accepting it")
    proposal.add_argument("--sources", type=Path, required=True)
    proposal.add_argument("--output", type=Path, required=True)
    flatten_parser = commands.add_parser("flatten", help="Copy RODA inputs into new data/alignments directories")
    flatten_parser.add_argument("source", type=Path)
    flatten_parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.command == "setup":
        setup(args.data_root, args.source_lock)
    elif args.command == "prepare-local":
        prepare_local(args.sources, args.data_root, args.source_lock)
    elif args.command == "propose-lock":
        propose_lock(args.sources, args.output)
    else:
        flatten(args.source, args.output)


if __name__ == "__main__":
    main()
