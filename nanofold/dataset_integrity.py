from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List

from .data import read_manifest


FINGERPRINT_SCHEMA_VERSION = 1

FINGERPRINT_COMPARISON_KEYS = (
    "schema_version",
    "train_manifest_chain_count",
    "val_manifest_chain_count",
    "unique_chain_count",
    "present_chain_count",
    "missing_chain_count",
    "chain_ids_sha256",
    "npz_files_sha256",
)


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    hasher = hashlib.sha256()
    with Path(path).open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def _chain_ids_sha256(chain_ids: List[str]) -> str:
    hasher = hashlib.sha256()
    for chain_id in chain_ids:
        hasher.update(chain_id.encode("utf-8"))
        hasher.update(b"\n")
    return hasher.hexdigest()


def _npz_files_sha256(processed_dir: Path, chain_ids: List[str], missing_chain_ids: List[str]) -> str:
    hasher = hashlib.sha256()
    for chain_id in chain_ids:
        npz_path = processed_dir / f"{chain_id}.npz"
        if not npz_path.exists():
            missing_chain_ids.append(chain_id)
            continue
        file_hash = sha256_file(npz_path)
        hasher.update(chain_id.encode("utf-8"))
        hasher.update(b"\t")
        hasher.update(file_hash.encode("ascii"))
        hasher.update(b"\n")
    return hasher.hexdigest()


def build_dataset_fingerprint(
    *,
    processed_dir: str | Path,
    train_manifest: str | Path,
    val_manifest: str | Path,
    require_no_missing: bool,
) -> Dict[str, Any]:
    processed_dir = Path(processed_dir).resolve()
    train_manifest = Path(train_manifest).resolve()
    val_manifest = Path(val_manifest).resolve()

    train_chain_ids = read_manifest(train_manifest)
    val_chain_ids = read_manifest(val_manifest)
    all_chain_ids = sorted(set(train_chain_ids + val_chain_ids))

    missing_chain_ids: List[str] = []
    npz_hash = _npz_files_sha256(processed_dir=processed_dir, chain_ids=all_chain_ids, missing_chain_ids=missing_chain_ids)
    if require_no_missing and missing_chain_ids:
        sample = ", ".join(missing_chain_ids[:8])
        raise FileNotFoundError(
            f"{len(missing_chain_ids)} manifest chains are missing preprocessed files in {processed_dir}. "
            f"Examples: {sample}"
        )

    return {
        "schema_version": FINGERPRINT_SCHEMA_VERSION,
        "train_manifest_chain_count": len(train_chain_ids),
        "val_manifest_chain_count": len(val_chain_ids),
        "unique_chain_count": len(all_chain_ids),
        "present_chain_count": len(all_chain_ids) - len(missing_chain_ids),
        "missing_chain_count": len(missing_chain_ids),
        "missing_chain_ids": missing_chain_ids,
        "chain_ids_sha256": _chain_ids_sha256(all_chain_ids),
        "npz_files_sha256": npz_hash,
    }


def load_fingerprint(path: str | Path) -> Dict[str, Any]:
    raw = json.loads(Path(path).read_text())
    if not isinstance(raw, dict):
        raise ValueError(f"Fingerprint file must contain a JSON object: {path}")
    return raw


def compare_fingerprints(actual: Dict[str, Any], expected: Dict[str, Any]) -> List[str]:
    mismatches: List[str] = []
    for key in FINGERPRINT_COMPARISON_KEYS:
        if actual.get(key) != expected.get(key):
            mismatches.append(f"Mismatch `{key}`: expected={expected.get(key)!r}, actual={actual.get(key)!r}")

    if "missing_chain_ids" in expected:
        expected_missing = list(expected.get("missing_chain_ids") or [])
        actual_missing = list(actual.get("missing_chain_ids") or [])
        if expected_missing != actual_missing:
            mismatches.append(
                "Mismatch `missing_chain_ids`: "
                f"expected={expected_missing!r}, actual={actual_missing!r}"
            )
    return mismatches


def verify_dataset_against_fingerprint(
    *,
    processed_dir: str | Path,
    train_manifest: str | Path,
    val_manifest: str | Path,
    expected_fingerprint_path: str | Path,
    require_no_missing: bool,
) -> Dict[str, Any]:
    expected = load_fingerprint(expected_fingerprint_path)
    actual = build_dataset_fingerprint(
        processed_dir=processed_dir,
        train_manifest=train_manifest,
        val_manifest=val_manifest,
        require_no_missing=require_no_missing,
    )
    mismatches = compare_fingerprints(actual=actual, expected=expected)
    if mismatches:
        joined = "\n".join(f"- {m}" for m in mismatches)
        raise ValueError(f"Dataset fingerprint mismatch:\n{joined}")
    return actual
