from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from .data import read_manifest


FINGERPRINT_SCHEMA_VERSION = 2

FINGERPRINT_COMPARISON_KEYS = (
    "schema_version",
    "track_id",
    "train_manifest_chain_count",
    "val_manifest_chain_count",
    "unique_chain_count",
    "present_feature_chain_count",
    "missing_feature_chain_count",
    "present_label_chain_count",
    "missing_label_chain_count",
    "chain_ids_sha256",
    "train_manifest_sha256",
    "val_manifest_sha256",
    "feature_files_sha256",
    "label_files_sha256",
)

REQUIRED_FEATURE_KEYS = (
    "aatype",
    "msa",
    "deletions",
    "template_aatype",
    "template_ca_coords",
    "template_ca_mask",
)
REQUIRED_LABEL_KEYS = ("ca_coords", "ca_mask")


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


def _dtype_name(dtype: np.dtype) -> str:
    return str(np.dtype(dtype))


def validate_feature_npz_schema(npz_path: str | Path) -> List[str]:
    npz_path = Path(npz_path)
    errors: List[str] = []

    with np.load(npz_path) as data:
        keys = set(data.keys())
        for key in REQUIRED_FEATURE_KEYS:
            if key not in keys:
                errors.append(f"Missing key `{key}`")
        if errors:
            return errors

        aatype = data["aatype"]
        msa = data["msa"]
        deletions = data["deletions"]
        template_aatype = data["template_aatype"]
        template_ca_coords = data["template_ca_coords"]
        template_ca_mask = data["template_ca_mask"]

        if _dtype_name(aatype.dtype) not in {"int32", "int64"}:
            errors.append(f"`aatype` dtype must be int32/int64 (got {_dtype_name(aatype.dtype)})")
        if _dtype_name(msa.dtype) not in {"int32", "int64"}:
            errors.append(f"`msa` dtype must be int32/int64 (got {_dtype_name(msa.dtype)})")
        if _dtype_name(deletions.dtype) not in {"int32", "int64"}:
            errors.append(f"`deletions` dtype must be int32/int64 (got {_dtype_name(deletions.dtype)})")
        if _dtype_name(template_aatype.dtype) not in {"int32", "int64"}:
            errors.append(
                f"`template_aatype` dtype must be int32/int64 (got {_dtype_name(template_aatype.dtype)})"
            )
        if _dtype_name(template_ca_coords.dtype) not in {"float32", "float64"}:
            errors.append(
                "`template_ca_coords` dtype must be float32/float64 "
                f"(got {_dtype_name(template_ca_coords.dtype)})"
            )
        if _dtype_name(template_ca_mask.dtype) not in {"bool"}:
            errors.append(f"`template_ca_mask` dtype must be bool (got {_dtype_name(template_ca_mask.dtype)})")

        if aatype.ndim != 1:
            errors.append(f"`aatype` must have shape (L,), got {aatype.shape}")
            return errors
        L = int(aatype.shape[0])

        if msa.ndim != 2:
            errors.append(f"`msa` must have shape (N, L), got {msa.shape}")
        elif msa.shape[1] != L:
            errors.append(f"`msa` second dimension must equal len(aatype)={L}, got {msa.shape[1]}")

        if deletions.ndim != 2:
            errors.append(f"`deletions` must have shape (N, L), got {deletions.shape}")
        elif deletions.shape != msa.shape:
            errors.append(f"`deletions` shape must match `msa` shape, got {deletions.shape} vs {msa.shape}")

        if template_aatype.ndim != 2:
            errors.append(f"`template_aatype` must have shape (T,L), got {template_aatype.shape}")
        elif template_aatype.shape[1] != L:
            errors.append(
                "`template_aatype` second dimension must equal len(aatype)="
                f"{L}, got {template_aatype.shape[1]}"
            )

        if template_ca_coords.ndim != 3 or template_ca_coords.shape[-1] != 3:
            errors.append(f"`template_ca_coords` must have shape (T,L,3), got {template_ca_coords.shape}")
        else:
            if template_ca_coords.shape[1] != L:
                errors.append(
                    "`template_ca_coords` second dimension must equal len(aatype)="
                    f"{L}, got {template_ca_coords.shape[1]}"
                )
            if template_ca_coords.shape[0] != template_aatype.shape[0]:
                errors.append(
                    "`template_ca_coords` first dimension must match `template_aatype` "
                    f"({template_ca_coords.shape[0]} vs {template_aatype.shape[0]})"
                )

        if template_ca_mask.ndim != 2:
            errors.append(f"`template_ca_mask` must have shape (T,L), got {template_ca_mask.shape}")
        else:
            if template_ca_mask.shape[1] != L:
                errors.append(
                    "`template_ca_mask` second dimension must equal len(aatype)="
                    f"{L}, got {template_ca_mask.shape[1]}"
                )
            if template_ca_mask.shape[0] != template_aatype.shape[0]:
                errors.append(
                    "`template_ca_mask` first dimension must match `template_aatype` "
                    f"({template_ca_mask.shape[0]} vs {template_aatype.shape[0]})"
                )
    return errors


def validate_label_npz_schema(npz_path: str | Path) -> List[str]:
    npz_path = Path(npz_path)
    errors: List[str] = []
    with np.load(npz_path) as data:
        keys = set(data.keys())
        for key in REQUIRED_LABEL_KEYS:
            if key not in keys:
                errors.append(f"Missing key `{key}`")
        if errors:
            return errors

        ca_coords = data["ca_coords"]
        ca_mask = data["ca_mask"]

        if _dtype_name(ca_coords.dtype) not in {"float32", "float64"}:
            errors.append(f"`ca_coords` dtype must be float32/float64 (got {_dtype_name(ca_coords.dtype)})")
        if _dtype_name(ca_mask.dtype) not in {"bool"}:
            errors.append(f"`ca_mask` dtype must be bool (got {_dtype_name(ca_mask.dtype)})")

        if ca_coords.ndim != 2 or ca_coords.shape[1] != 3:
            errors.append(f"`ca_coords` must have shape (L,3), got {ca_coords.shape}")
        if ca_mask.ndim != 1:
            errors.append(f"`ca_mask` must have shape (L,), got {ca_mask.shape}")
        elif ca_coords.ndim == 2 and ca_coords.shape[0] != ca_mask.shape[0]:
            errors.append(
                "`ca_coords` first dimension must equal `ca_mask` length "
                f"({ca_coords.shape[0]} vs {ca_mask.shape[0]})"
            )
    return errors


def validate_npz_schema(npz_path: str | Path) -> List[str]:
    # Backward-compatible export used by tests/utilities.
    return validate_feature_npz_schema(npz_path)


def _files_sha256(
    *,
    base_dir: Path | None,
    chain_ids: List[str],
    missing_chain_ids: List[str],
    validate_schema: bool,
    schema_validator,
) -> str | None:
    if base_dir is None:
        return None
    hasher = hashlib.sha256()
    for chain_id in chain_ids:
        npz_path = base_dir / f"{chain_id}.npz"
        if not npz_path.exists():
            missing_chain_ids.append(chain_id)
            continue

        if validate_schema:
            schema_errors = schema_validator(npz_path)
            if schema_errors:
                joined = "; ".join(schema_errors[:8])
                raise ValueError(f"Invalid schema for {npz_path}: {joined}")

        file_hash = sha256_file(npz_path)
        hasher.update(chain_id.encode("utf-8"))
        hasher.update(b"\t")
        hasher.update(file_hash.encode("ascii"))
        hasher.update(b"\n")
    return hasher.hexdigest()


def build_dataset_fingerprint(
    *,
    processed_features_dir: str | Path,
    train_manifest: str | Path,
    val_manifest: str | Path,
    require_no_missing: bool,
    processed_labels_dir: str | Path | None = None,
    require_labels: bool = True,
    validate_schema: bool = True,
    track_id: str | None = None,
    source_lock_path: str | Path | None = None,
) -> Dict[str, Any]:
    processed_features_dir = Path(processed_features_dir).resolve()
    processed_labels_dir_path = Path(processed_labels_dir).resolve() if processed_labels_dir else None
    train_manifest = Path(train_manifest).resolve()
    val_manifest = Path(val_manifest).resolve()

    train_chain_ids = read_manifest(train_manifest)
    val_chain_ids = read_manifest(val_manifest)
    all_chain_ids = sorted(set(train_chain_ids + val_chain_ids))

    missing_feature_chain_ids: List[str] = []
    missing_label_chain_ids: List[str] = []

    feature_hash = _files_sha256(
        base_dir=processed_features_dir,
        chain_ids=all_chain_ids,
        missing_chain_ids=missing_feature_chain_ids,
        validate_schema=validate_schema,
        schema_validator=validate_feature_npz_schema,
    )
    label_hash = _files_sha256(
        base_dir=processed_labels_dir_path if require_labels else None,
        chain_ids=all_chain_ids,
        missing_chain_ids=missing_label_chain_ids,
        validate_schema=validate_schema,
        schema_validator=validate_label_npz_schema,
    )

    if require_no_missing and missing_feature_chain_ids:
        sample = ", ".join(missing_feature_chain_ids[:8])
        raise FileNotFoundError(
            f"{len(missing_feature_chain_ids)} manifest chains are missing feature files in {processed_features_dir}. "
            f"Examples: {sample}"
        )
    if require_no_missing and require_labels and missing_label_chain_ids:
        sample = ", ".join(missing_label_chain_ids[:8])
        raise FileNotFoundError(
            f"{len(missing_label_chain_ids)} manifest chains are missing label files in {processed_labels_dir_path}. "
            f"Examples: {sample}"
        )

    return {
        "schema_version": FINGERPRINT_SCHEMA_VERSION,
        "track_id": track_id,
        "train_manifest_chain_count": len(train_chain_ids),
        "val_manifest_chain_count": len(val_chain_ids),
        "unique_chain_count": len(all_chain_ids),
        "present_feature_chain_count": len(all_chain_ids) - len(missing_feature_chain_ids),
        "missing_feature_chain_count": len(missing_feature_chain_ids),
        "missing_feature_chain_ids": missing_feature_chain_ids,
        "present_label_chain_count": len(all_chain_ids) - len(missing_label_chain_ids) if require_labels else 0,
        "missing_label_chain_count": len(missing_label_chain_ids) if require_labels else 0,
        "missing_label_chain_ids": missing_label_chain_ids if require_labels else [],
        "chain_ids_sha256": _chain_ids_sha256(all_chain_ids),
        "train_manifest_sha256": sha256_file(train_manifest),
        "val_manifest_sha256": sha256_file(val_manifest),
        "feature_files_sha256": feature_hash,
        "label_files_sha256": label_hash,
        "require_labels": bool(require_labels),
        "source_lock_path": str(Path(source_lock_path).resolve()) if source_lock_path else None,
        "source_lock_sha256": sha256_file(source_lock_path) if source_lock_path else None,
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

    for key in ("missing_feature_chain_ids", "missing_label_chain_ids"):
        if key in expected and list(actual.get(key) or []) != list(expected.get(key) or []):
            mismatches.append(
                f"Mismatch `{key}`: expected={list(expected.get(key) or [])!r}, "
                f"actual={list(actual.get(key) or [])!r}"
            )
    for key in ("source_lock_sha256",):
        if key in expected and actual.get(key) != expected.get(key):
            mismatches.append(f"Mismatch `{key}`: expected={expected.get(key)!r}, actual={actual.get(key)!r}")
    return mismatches


def verify_dataset_against_fingerprint(
    *,
    processed_features_dir: str | Path,
    train_manifest: str | Path,
    val_manifest: str | Path,
    expected_fingerprint_path: str | Path,
    require_no_missing: bool,
    processed_labels_dir: str | Path | None = None,
    require_labels: bool = True,
    validate_schema: bool = True,
    track_id: str | None = None,
    source_lock_path: str | Path | None = None,
) -> Dict[str, Any]:
    expected = load_fingerprint(expected_fingerprint_path)
    actual = build_dataset_fingerprint(
        processed_features_dir=processed_features_dir,
        processed_labels_dir=processed_labels_dir,
        train_manifest=train_manifest,
        val_manifest=val_manifest,
        require_no_missing=require_no_missing,
        require_labels=require_labels,
        validate_schema=validate_schema,
        track_id=track_id,
        source_lock_path=source_lock_path,
    )
    mismatches = compare_fingerprints(actual=actual, expected=expected)
    if mismatches:
        joined = "\n".join(f"- {m}" for m in mismatches)
        raise ValueError(f"Dataset fingerprint mismatch:\n{joined}")
    return actual
