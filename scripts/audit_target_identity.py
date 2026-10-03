"""Verify raw A3M and processed target identity against a pinned chain cache.

This gate checks every selected target without repairing inputs or accepting
missing assets. Receipts are private; terminal output contains only counts.
Labels, geometry and cross-split homology require their separate corpus gates.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import zipfile
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
for import_root in (REPO_ROOT, SCRIPT_DIR):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from audit_cache_targets import (
    _reject_json_constant,
    _unique_json_object,
    read_manifest,
    sequences_for,
    sha256,
    write_json,
)

from nanofold.chain_paths import chain_data_dir, chain_npz_path
from nanofold.dataset_integrity import validate_feature_npz_schema

# Independent statement of the on-disk atom/MSA token contract.
TOKENS = {aa: index for index, aa in enumerate("ARNDCQEGHILKMFPSTWYV")}
AUDIT_ERRORS = (ValueError, OSError, KeyError, EOFError, UnicodeError, zipfile.BadZipFile)


def snapshot(path: Path) -> tuple[int, int, int, int, int]:
    info = path.stat()
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns


def raw_record(path: Path, expected: str) -> dict[str, Any]:
    """Read and hash all rows while retaining only one row's text in memory."""
    before = snapshot(path)
    hasher = hashlib.sha256()
    first: str | None = None
    pieces: list[str] = []
    have_header = False
    rows = 0
    width: int | None = None

    def finish() -> None:
        nonlocal first, rows, width
        raw = "".join(pieces)
        if not raw or re.fullmatch(r"[A-Za-z.\-]+", raw) is None:
            raise ValueError("A3M row is empty or contains an invalid character")
        aligned = "".join(aa for aa in raw if not aa.islower() and aa != ".")
        if not aligned:
            raise ValueError("A3M row has no aligned positions")
        if width is None:
            width = len(aligned)
            first = aligned.replace("-", "")
        elif len(aligned) != width:
            raise ValueError("A3M rows have inconsistent aligned widths")
        rows += 1

    with path.open("rb") as handle:
        for line in handle:
            hasher.update(line)
            text = line.decode("utf-8").strip()
            if not text:
                continue
            if text.startswith(">"):
                if not text[1:].strip():
                    raise ValueError("A3M contains an empty record header")
                if have_header:
                    finish()
                have_header = True
                pieces = []
            else:
                if not have_header:
                    raise ValueError("A3M sequence precedes its first header")
                pieces.append(text)
    if not have_header:
        raise ValueError("A3M contains no sequence records")
    finish()
    if first != expected:
        raise ValueError("A3M ungapped query differs from the pinned cache target")
    if snapshot(path) != before:
        raise ValueError("A3M changed while being audited")
    return {"path": str(path), "sha256": hasher.hexdigest(), "bytes": before[2], "rows": rows,
            "query_length": len(expected), "aligned_width": width}


def find_raw(root: Path, chain: str, name: str) -> Path:
    chain_root = chain_data_dir(root / "roda_pdb", chain)
    paths = {path.resolve() for path in chain_root.rglob(name) if path.is_file()}
    if len(paths) != 1:
        raise ValueError("Required A3M is missing or has multiple distinct source files")
    path = next(iter(paths))
    if not path.is_relative_to(root):
        raise ValueError("A3M resolves outside the bound raw root")
    return path


def find_processed(roots: list[Path], chain: str) -> Path:
    paths = {path.resolve() for root in roots if (path := chain_npz_path(root, chain)).is_file()}
    if len(paths) != 1:
        raise ValueError("Processed features are missing or appear in multiple distinct roots")
    path = next(iter(paths))
    if not any(path.is_relative_to(root) for root in roots):
        raise ValueError("Processed features resolve outside the bound feature roots")
    return path


def processed_record(path: Path, expected: str, chain: str) -> dict[str, Any]:
    before = snapshot(path)
    file_hash = sha256(path)
    # NumPy silently selects one entry for duplicate ZIP names. Reject that
    # ambiguity before examining arrays, including unaccessed object members.
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)) or any(
            not name.endswith(".npy") or Path(name).name != name or name[:-4] in names for name in names
        ):
            raise ValueError("Processed archive contains duplicate or invalid members")
    loaded = np.load(path, allow_pickle=False)
    if not isinstance(loaded, np.lib.npyio.NpzFile):
        raise ValueError("Processed features must be an NPZ archive")
    with loaded as arrays:
        if not {"aatype", "msa", "chain_id"} <= set(arrays.files):
            raise ValueError("Processed features lack target, query MSA or chain identity")
        for name in arrays.files:
            value = arrays[name]
            if not isinstance(value, np.ndarray) or value.dtype.hasobject:
                raise ValueError("Processed features contain unsafe or non-array members")
            if np.issubdtype(value.dtype, np.number) and not np.isfinite(value).all():
                raise ValueError("Processed features contain nonfinite values")
        actual, msa, identity = arrays["aatype"], arrays["msa"], arrays["chain_id"]
        tokens = np.asarray([TOKENS[aa] for aa in expected], dtype=np.int32)
        if identity.shape != () or identity.dtype.kind not in {"U", "S"}:
            raise ValueError("Processed feature chain identity has invalid shape or dtype")
        actual_chain = identity.item()
        if isinstance(actual_chain, bytes):
            actual_chain = actual_chain.decode("utf-8")
        if actual_chain != chain:
            raise ValueError("Processed feature chain identity disagrees with manifest")
        if actual.dtype not in (np.dtype("int32"), np.dtype("int64")) or actual.shape != tokens.shape:
            raise ValueError("Processed target has an invalid dtype or shape")
        if not np.array_equal(actual, tokens):
            raise ValueError("Processed target differs from the pinned cache target")
        if (msa.dtype not in (np.dtype("int32"), np.dtype("int64")) or msa.ndim != 2
                or msa.shape[0] == 0 or msa.shape[1:] != tokens.shape or not np.array_equal(msa[0], tokens)):
            raise ValueError("Processed query MSA differs from the pinned cache target")
        rows = int(msa.shape[0])
    if validate_feature_npz_schema(path):
        raise ValueError("Processed features violate the complete feature schema")
    if snapshot(path) != before:
        raise ValueError("Processed feature file changed while being audited")
    return {"path": str(path), "sha256": file_hash, "bytes": before[2], "target_length": len(expected), "msa_rows": rows}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chain-data-cache", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, action="append", required=True,
                        help="Complete mutually disjoint manifests; repeat for public all and hidden.")
    parser.add_argument("--expected-counts", type=int, nargs="+", required=True,
                        help="Exact required count for each manifest, in argument order.")
    parser.add_argument("--raw-root", type=Path)
    parser.add_argument("--features-dir", type=Path, action="append", default=[])
    parser.add_argument("--msa-name", action="append", default=[])
    parser.add_argument("--min-len", type=int, default=40)
    parser.add_argument("--max-len", type=int, default=256)
    parser.add_argument("--out-dir", type=Path, required=True, help="Fresh private directory for the immutable receipt.")
    return parser.parse_args(argv)


def audit_targets(args: argparse.Namespace) -> int:
    out = args.out_dir.resolve()
    if args.out_dir.is_symlink() or (out.is_relative_to(REPO_ROOT) and not out.is_relative_to(REPO_ROOT / ".nanofold_private")):
        raise ValueError("Audit outputs must remain in a fresh private directory")
    if not (args.raw_root or args.features_dir):
        raise ValueError("At least one raw or processed asset root is required")
    if (len(args.expected_counts) != len(args.manifest) or any(count < 1 for count in args.expected_counts)
            or args.min_len < 1 or args.max_len < args.min_len):
        raise ValueError("Invalid manifest counts or target length bounds")
    names = args.msa_name or ["uniref90_hits.a3m"]
    if len(names) != len(set(names)) or any(
        re.fullmatch(r"[A-Za-z0-9_.-]+", name) is None or name in {".", ".."} for name in names
    ):
        raise ValueError("MSA names must be unique plain filenames")
    roots = [path.resolve() for path in args.features_dir]
    raw_root = args.raw_root.resolve() if args.raw_root else None
    asset_roots = [*roots, *([raw_root] if raw_root else [])]
    if any(not root.is_dir() or out.is_relative_to(root) or root.is_relative_to(out) for root in asset_roots):
        raise ValueError("Asset roots must exist and be separate from the audit output")
    inputs = {"chain_data_cache": args.chain_data_cache.resolve(), "audit_script": Path(__file__).resolve()}
    for relative in ("scripts/audit_cache_targets.py", "nanofold/chain_paths.py", "nanofold/structure_metadata.py",
                     "nanofold/dataset_integrity.py", "nanofold/a3m.py", "nanofold/residue_constants.py"):
        inputs[relative] = REPO_ROOT / relative
    for index, path in enumerate(args.manifest):
        inputs[f"manifest_{index}"] = path.resolve()
    if any(not path.is_file() or path.is_relative_to(out) for path in inputs.values()):
        raise ValueError("Audit inputs must exist outside the output directory")
    original_umask = os.umask(0o077)
    try:
        out.mkdir(mode=0o700, parents=True, exist_ok=False)
        receipt: dict[str, Any] = {
            "schema_version": 1, "status": "running", "scope": "selected_target_identity_and_feature_schema",
            "raw_audited": raw_root is not None, "processed_audited": bool(roots), "labels_audited": False,
            "asset_roots": {"raw": str(raw_root) if raw_root else None, "features": [str(path) for path in roots]},
            "expected_counts": args.expected_counts, "sequence_contract": {"min_len": args.min_len, "max_len": args.max_len},
            "manifests": [], "records": [], "failures": [], "selected_chain_count": 0,
        }
        write_json(out / "summary.json", receipt)
        try:
            receipt["inputs"] = {name: {"path": str(path), "sha256": sha256(path)} for name, path in inputs.items()}
            cache = json.loads(inputs["chain_data_cache"].read_text(), object_pairs_hook=_unique_json_object,
                               parse_constant=_reject_json_constant)
            if not isinstance(cache, dict):
                raise ValueError("Chain cache must be a JSON object")
            seen: set[str] = set()
            manifests: list[list[str]] = []
            for index, count in enumerate(args.expected_counts):
                chains = read_manifest(inputs[f"manifest_{index}"])
                if len(chains) != count or seen & set(chains):
                    raise ValueError("Manifest counts differ from requirements or manifests overlap")
                seen.update(chains)
                manifests.append(chains)
                receipt["manifests"].append({"input": f"manifest_{index}", "count": count})
            receipt["selected_chain_count"] = len(seen)
            for index, chains in enumerate(manifests):
                for row_index, chain in enumerate(chains):
                    row: dict[str, Any] = {"manifest_index": index, "row_index": row_index,
                                          "chain_id_sha256": hashlib.sha256(chain.encode()).hexdigest()}
                    try:
                        sequence = sequences_for([chain], cache, min_len=args.min_len, max_len=args.max_len)[0]
                        row["cache_target_sha256"] = hashlib.sha256(sequence.encode()).hexdigest()
                        row["target_length"] = len(sequence)
                        if raw_root:
                            row["raw"] = {name: raw_record(find_raw(raw_root, chain, name), sequence) for name in names}
                        if roots:
                            row["processed"] = processed_record(find_processed(roots, chain), sequence, chain)
                        receipt["records"].append(row)
                    except AUDIT_ERRORS as exc:
                        receipt["failures"].append({**row, "error_type": type(exc).__name__, "reason": str(exc)})
            # Rehash the complete inventory after the last target: a source or
            # early asset changed during the audit must not receive a pass.
            for record in receipt["records"]:
                chain = manifests[record["manifest_index"]][record["row_index"]]
                if raw_root and any(str(find_raw(raw_root, chain, name)) != record["raw"][name]["path"] for name in names):
                    raise ValueError("An audited raw source changed before the gate completed")
                if roots and str(find_processed(roots, chain)) != record["processed"]["path"]:
                    raise ValueError("An audited processed source changed before the gate completed")
                assets = [*record.get("raw", {}).values()]
                if "processed" in record:
                    assets.append(record["processed"])
                if any(sha256(Path(asset["path"])) != asset["sha256"] for asset in assets):
                    raise ValueError("An audited asset changed before the gate completed")
            if any(sha256(path) != receipt["inputs"][name]["sha256"] for name, path in inputs.items()):
                raise ValueError("An audit input changed before the gate completed")
            receipt["status"] = "failed" if receipt["failures"] else "passed"
        except AUDIT_ERRORS as exc:
            receipt["status"] = "error"
            receipt["failures"].append({"error_type": type(exc).__name__, "reason": str(exc)})
        receipt["passed_chain_count"] = len(receipt["records"])
        receipt["failure_count"] = len(receipt["failures"])
        receipt["record_inventory_sha256"] = hashlib.sha256(
            json.dumps(receipt["records"], sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        ).hexdigest()
        receipt["raw_bytes"] = sum(asset["bytes"] for row in receipt["records"] for asset in row.get("raw", {}).values())
        receipt["raw_rows"] = sum(asset["rows"] for row in receipt["records"] for asset in row.get("raw", {}).values())
        write_json(out / "summary.json", receipt)
        print(f"Target identity audit: {receipt['status']}; selected={receipt['selected_chain_count']}; "
              f"passed={receipt['passed_chain_count']}; failures={receipt['failure_count']}")
        return 0 if receipt["status"] == "passed" else (1 if receipt["status"] == "failed" else 2)
    finally:
        os.umask(original_umask)


def main(argv: list[str] | None = None) -> int:
    try:
        return audit_targets(parse_args(argv))
    except AUDIT_ERRORS:
        print("Target identity audit: preflight error; no success receipt was issued", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
