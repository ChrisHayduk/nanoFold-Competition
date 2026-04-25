from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping

import numpy as np

from .chain_paths import chain_npz_path
from .data import read_manifest
from .utils import sha256_file

FINGERPRINT_COMPARISON_KEYS = (
    "track_id",
    "split_names",
    "unique_chain_count",
    "present_feature_chain_count",
    "missing_feature_chain_count",
    "present_label_chain_count",
    "missing_label_chain_count",
    "chain_ids_sha256",
    "feature_files_sha256",
    "label_files_sha256",
    "require_labels",
    "source_lock_sha256",
    "preprocess_config_sha256",
)

PREPROCESS_META_FILENAME = "preprocess_meta.json"

REQUIRED_FEATURE_KEYS = (
    "aatype",
    "msa",
    "deletions",
    "template_aatype",
    "template_ca_coords",
    "template_ca_mask",
)
OPTIONAL_FEATURE_KEYS = (
    "residue_index",
    "between_segment_residues",
)
REQUIRED_LABEL_KEYS = ("ca_coords", "ca_mask", "atom14_positions", "atom14_mask")
OPTIONAL_LABEL_KEYS = (
    "residue_index",
    "resolution",
)


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

        for key in OPTIONAL_FEATURE_KEYS:
            if key not in keys:
                continue
            value = data[key]
            if _dtype_name(value.dtype) not in {"int32", "int64"}:
                errors.append(f"`{key}` dtype must be int32/int64 (got {_dtype_name(value.dtype)})")
            if value.ndim != 1:
                errors.append(f"`{key}` must have shape (L,), got {value.shape}")
            elif value.shape[0] != L:
                errors.append(f"`{key}` first dimension must equal len(aatype)={L}, got {value.shape[0]}")
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

        residue_count = int(ca_mask.shape[0]) if ca_mask.ndim == 1 else None

        if "atom14_positions" in keys:
            atom14_positions = data["atom14_positions"]
            if _dtype_name(atom14_positions.dtype) not in {"float32", "float64"}:
                errors.append(
                    f"`atom14_positions` dtype must be float32/float64 (got {_dtype_name(atom14_positions.dtype)})"
                )
            if atom14_positions.ndim != 3 or atom14_positions.shape[1:] != (14, 3):
                errors.append(
                    f"`atom14_positions` must have shape (L,14,3), got {atom14_positions.shape}"
                )
            elif residue_count is not None and atom14_positions.shape[0] != residue_count:
                errors.append(
                    "`atom14_positions` first dimension must equal `ca_mask` length "
                    f"({atom14_positions.shape[0]} vs {residue_count})"
                )

        if "atom14_mask" in keys:
            atom14_mask = data["atom14_mask"]
            if _dtype_name(atom14_mask.dtype) != "bool":
                errors.append(f"`atom14_mask` dtype must be bool (got {_dtype_name(atom14_mask.dtype)})")
            if atom14_mask.ndim != 2 or atom14_mask.shape[1] != 14:
                errors.append(f"`atom14_mask` must have shape (L,14), got {atom14_mask.shape}")
            elif residue_count is not None and atom14_mask.shape[0] != residue_count:
                errors.append(
                    "`atom14_mask` first dimension must equal `ca_mask` length "
                    f"({atom14_mask.shape[0]} vs {residue_count})"
                )

        if "residue_index" in keys:
            residue_index = data["residue_index"]
            if _dtype_name(residue_index.dtype) not in {"int32", "int64"}:
                errors.append(
                    f"`residue_index` dtype must be int32/int64 (got {_dtype_name(residue_index.dtype)})"
                )
            if residue_index.ndim != 1:
                errors.append(f"`residue_index` must have shape (L,), got {residue_index.shape}")
            elif residue_count is not None and residue_index.shape[0] != residue_count:
                errors.append(
                    "`residue_index` first dimension must equal `ca_mask` length "
                    f"({residue_index.shape[0]} vs {residue_count})"
                )

        if "resolution" in keys:
            resolution = data["resolution"]
            if _dtype_name(resolution.dtype) not in {"float32", "float64"}:
                errors.append(
                    f"`resolution` dtype must be float32/float64 (got {_dtype_name(resolution.dtype)})"
                )
            if resolution.ndim not in (0, 1):
                errors.append(f"`resolution` must be a scalar, got shape {resolution.shape}")
            elif resolution.ndim == 1 and resolution.shape[0] != 1:
                errors.append(f"`resolution` rank-1 form must be length 1, got {resolution.shape}")
    return errors


def validate_npz_schema(npz_path: str | Path) -> List[str]:
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
        npz_path = chain_npz_path(base_dir, chain_id)
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


def _normalize_manifest_map(manifest_paths: Mapping[str, str | Path]) -> Dict[str, Path]:
    normalized: Dict[str, Path] = {}
    for split_name, manifest_path in manifest_paths.items():
        key = str(split_name).strip()
        if key == "":
            raise ValueError("Manifest split names must be non-empty.")
        normalized[key] = Path(manifest_path).resolve()
    if not normalized:
        raise ValueError("At least one manifest path is required to build a fingerprint.")
    return normalized


def _display_path(path: str | Path) -> str:
    resolved = Path(path).resolve()
    try:
        return str(resolved.relative_to(Path.cwd().resolve())).replace("\\", "/")
    except ValueError:
        return str(path)


def build_split_fingerprint(
    *,
    processed_features_dir: str | Path,
    manifest_paths: Mapping[str, str | Path],
    require_no_missing: bool,
    processed_labels_dir: str | Path | None = None,
    require_labels: bool = True,
    validate_schema: bool = True,
    track_id: str | None = None,
    source_lock_path: str | Path | None = None,
) -> Dict[str, Any]:
    processed_features_dir = Path(processed_features_dir).resolve()
    processed_labels_dir_path = Path(processed_labels_dir).resolve() if processed_labels_dir else None
    manifest_map = _normalize_manifest_map(manifest_paths)

    manifest_meta: Dict[str, Dict[str, Any]] = {}
    all_chain_ids_unsorted: List[str] = []
    for split_name, manifest_path in manifest_map.items():
        chain_ids = read_manifest(manifest_path)
        manifest_meta[split_name] = {
            "chain_count": len(chain_ids),
            "sha256": sha256_file(manifest_path),
        }
        all_chain_ids_unsorted.extend(chain_ids)
    all_chain_ids = sorted(set(all_chain_ids_unsorted))

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

    preprocess_meta_path = processed_features_dir / PREPROCESS_META_FILENAME
    preprocess_config_sha256 = sha256_file(preprocess_meta_path) if preprocess_meta_path.exists() else None

    result: Dict[str, Any] = {
        "track_id": track_id,
        "split_names": list(manifest_map.keys()),
        "manifests": manifest_meta,
        "unique_chain_count": len(all_chain_ids),
        "present_feature_chain_count": len(all_chain_ids) - len(missing_feature_chain_ids),
        "missing_feature_chain_count": len(missing_feature_chain_ids),
        "missing_feature_chain_ids": missing_feature_chain_ids,
        "present_label_chain_count": len(all_chain_ids) - len(missing_label_chain_ids) if require_labels else 0,
        "missing_label_chain_count": len(missing_label_chain_ids) if require_labels else 0,
        "missing_label_chain_ids": missing_label_chain_ids if require_labels else [],
        "chain_ids_sha256": _chain_ids_sha256(all_chain_ids),
        "feature_files_sha256": feature_hash,
        "label_files_sha256": label_hash,
        "require_labels": bool(require_labels),
        "source_lock_path": _display_path(source_lock_path) if source_lock_path else None,
        "source_lock_sha256": sha256_file(source_lock_path) if source_lock_path else None,
        "preprocess_config_sha256": preprocess_config_sha256,
    }
    if list(manifest_map.keys()) == ["train", "val"]:
        result["train_manifest_chain_count"] = manifest_meta["train"]["chain_count"]
        result["val_manifest_chain_count"] = manifest_meta["val"]["chain_count"]
        result["train_manifest_sha256"] = manifest_meta["train"]["sha256"]
        result["val_manifest_sha256"] = manifest_meta["val"]["sha256"]
    return result


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
    return build_split_fingerprint(
        processed_features_dir=processed_features_dir,
        processed_labels_dir=processed_labels_dir,
        manifest_paths={"train": train_manifest, "val": val_manifest},
        require_no_missing=require_no_missing,
        require_labels=require_labels,
        validate_schema=validate_schema,
        track_id=track_id,
        source_lock_path=source_lock_path,
    )


def load_fingerprint(path: str | Path) -> Dict[str, Any]:
    raw = json.loads(Path(path).read_text())
    if not isinstance(raw, dict):
        raise ValueError(f"Fingerprint file must contain a JSON object: {path}")
    return raw


def _source_lock_path_for_verification(
    *,
    explicit_source_lock_path: str | Path | None,
    expected_fingerprint: Mapping[str, Any],
    expected_fingerprint_path: str | Path,
) -> str | Path | None:
    if explicit_source_lock_path:
        return explicit_source_lock_path
    source_lock = expected_fingerprint.get("source_lock_path")
    if not isinstance(source_lock, str) or not source_lock.strip():
        return None
    source_lock_path = Path(source_lock.strip())
    if source_lock_path.is_absolute():
        return source_lock_path

    cwd_candidate = Path.cwd() / source_lock_path
    if cwd_candidate.exists():
        return cwd_candidate

    fingerprint_relative_candidate = Path(expected_fingerprint_path).resolve().parent / source_lock_path
    if fingerprint_relative_candidate.exists():
        return fingerprint_relative_candidate
    return source_lock_path


def compare_fingerprints(
    actual: Dict[str, Any],
    expected: Dict[str, Any],
    *,
    comparison_mode: str = "exact",
) -> List[str]:
    if comparison_mode not in {"exact", "features_only"}:
        raise ValueError(f"Unsupported fingerprint comparison_mode={comparison_mode!r}")

    mismatches: List[str] = []
    keys: tuple[str, ...]

    if "manifests" not in expected:
        mismatches.append("Expected fingerprint is missing required `manifests` metadata.")
    else:
        if comparison_mode == "exact":
            keys = FINGERPRINT_COMPARISON_KEYS
        else:
            keys = (
                "track_id",
                "split_names",
                "unique_chain_count",
                "present_feature_chain_count",
                "missing_feature_chain_count",
                "chain_ids_sha256",
                "feature_files_sha256",
                "source_lock_sha256",
                "preprocess_config_sha256",
            )
        for key in keys:
            if actual.get(key) != expected.get(key):
                mismatches.append(f"Mismatch `{key}`: expected={expected.get(key)!r}, actual={actual.get(key)!r}")

        if dict(actual.get("manifests") or {}) != dict(expected.get("manifests") or {}):
            mismatches.append(
                f"Mismatch `manifests`: expected={dict(expected.get('manifests') or {})!r}, "
                f"actual={dict(actual.get('manifests') or {})!r}"
            )

    list_keys: tuple[str, ...] = ("missing_feature_chain_ids", "missing_label_chain_ids")
    if comparison_mode == "features_only":
        list_keys = ("missing_feature_chain_ids",)
    for key in list_keys:
        if key in expected and list(actual.get(key) or []) != list(expected.get(key) or []):
            mismatches.append(
                f"Mismatch `{key}`: expected={list(expected.get(key) or [])!r}, "
                f"actual={list(actual.get(key) or [])!r}"
            )
    return mismatches


def verify_split_against_fingerprint(
    *,
    processed_features_dir: str | Path,
    manifest_paths: Mapping[str, str | Path],
    expected_fingerprint_path: str | Path,
    require_no_missing: bool,
    processed_labels_dir: str | Path | None = None,
    require_labels: bool = True,
    validate_schema: bool = True,
    track_id: str | None = None,
    source_lock_path: str | Path | None = None,
    comparison_mode: str = "exact",
) -> Dict[str, Any]:
    expected = load_fingerprint(expected_fingerprint_path)
    source_lock_path = _source_lock_path_for_verification(
        explicit_source_lock_path=source_lock_path,
        expected_fingerprint=expected,
        expected_fingerprint_path=expected_fingerprint_path,
    )
    actual = build_split_fingerprint(
        processed_features_dir=processed_features_dir,
        processed_labels_dir=processed_labels_dir,
        manifest_paths=manifest_paths,
        require_no_missing=require_no_missing,
        require_labels=require_labels,
        validate_schema=validate_schema,
        track_id=track_id,
        source_lock_path=source_lock_path,
    )
    mismatches = compare_fingerprints(actual=actual, expected=expected, comparison_mode=comparison_mode)
    if mismatches:
        joined = "\n".join(f"- {m}" for m in mismatches)
        raise ValueError(f"Dataset fingerprint mismatch:\n{joined}")
    return actual


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
    comparison_mode: str = "exact",
) -> Dict[str, Any]:
    return verify_split_against_fingerprint(
        processed_features_dir=processed_features_dir,
        processed_labels_dir=processed_labels_dir,
        manifest_paths={"train": train_manifest, "val": val_manifest},
        expected_fingerprint_path=expected_fingerprint_path,
        require_no_missing=require_no_missing,
        require_labels=require_labels,
        validate_schema=validate_schema,
        track_id=track_id,
        source_lock_path=source_lock_path,
        comparison_mode=comparison_mode,
    )
