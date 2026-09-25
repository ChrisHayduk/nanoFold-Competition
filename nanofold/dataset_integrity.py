from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping

import numpy as np

from .a3m import GAP_ID, UNK_ID
from .chain_paths import chain_npz_path
from .data import LABEL_ONLY_KEYS, read_manifest
from .residue_constants import CA_ATOM14_SLOT
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
        for key in LABEL_ONLY_KEYS:
            if key in keys:
                errors.append(f"Feature archives must not contain supervision key `{key}`")
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
        if L == 0:
            errors.append("`aatype` must contain at least one residue")
        token_arrays = (("aatype", aatype, UNK_ID), ("msa", msa, GAP_ID), ("template_aatype", template_aatype, UNK_ID))
        for key, value, maximum in token_arrays:
            if np.issubdtype(value.dtype, np.number):
                if not np.isfinite(value).all() or np.any(value < 0) or np.any(value > maximum):
                    errors.append(f"`{key}` tokens must be in [0,{maximum}]")
        if np.issubdtype(deletions.dtype, np.number) and (not np.isfinite(deletions).all() or np.any(deletions < 0)):
            errors.append("`deletions` must contain nonnegative finite counts")
        if np.issubdtype(template_ca_coords.dtype, np.number) and not np.isfinite(template_ca_coords).all():
            errors.append("`template_ca_coords` must contain only finite coordinates")

        if msa.ndim != 2:
            errors.append(f"`msa` must have shape (N, L), got {msa.shape}")
        elif msa.shape[1] != L:
            errors.append(f"`msa` second dimension must equal len(aatype)={L}, got {msa.shape[1]}")
        elif msa.shape[0] == 0:
            errors.append("`msa` must include a query row")
        elif not np.array_equal(msa[0], aatype):
            errors.append("`msa` query row must equal `aatype`")

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
            if template_aatype.ndim == 2 and template_ca_coords.shape[0] != template_aatype.shape[0]:
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
            if template_aatype.ndim == 2 and template_ca_mask.shape[0] != template_aatype.shape[0]:
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
        for key in ("ca_coords", "atom14_positions", "resolution"):
            if key in keys:
                value = data[key]
                if np.issubdtype(value.dtype, np.number) and not np.isfinite(value).all():
                    errors.append(f"`{key}` must contain only finite values")

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

        if ca_mask.ndim == 1 and ca_mask.shape[0] == 0:
            errors.append("Labels must contain at least one residue")
        if atom14_mask.ndim == 2 and atom14_mask.shape[1] == 14:
            if not np.array_equal(ca_mask, atom14_mask[:, CA_ATOM14_SLOT]):
                errors.append("`ca_mask` must equal the CA slot of `atom14_mask`")
        if atom14_positions.ndim == 3 and atom14_positions.shape[1:] == (14, 3):
            if not np.array_equal(ca_coords, atom14_positions[:, CA_ATOM14_SLOT, :]):
                errors.append("`ca_coords` must equal the CA slot of `atom14_positions`")

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
            if np.issubdtype(resolution.dtype, np.number) and np.any(resolution < 0):
                errors.append("`resolution` must be nonnegative")
            if resolution.ndim not in (0, 1):
                errors.append(f"`resolution` must be a scalar, got shape {resolution.shape}")
            elif resolution.ndim == 1 and resolution.shape[0] != 1:
                errors.append(f"`resolution` rank-1 form must be length 1, got {resolution.shape}")
    return errors


def _chain_identity_errors(value: np.ndarray | None, expected_chain_id: str, *, required: bool) -> List[str]:
    if value is None:
        return [f"Missing `chain_id` for requested chain {expected_chain_id}"] if required else []
    if value.shape != () or str(value.item()) != expected_chain_id:
        return [f"`chain_id` must match requested chain {expected_chain_id}"]
    return []


def validate_feature_label_pair(
    feature_path: str | Path,
    label_path: str | Path,
    *,
    expected_chain_id: str | None = None,
    require_chain_id: bool = False,
) -> List[str]:
    """Check that individually valid arrays describe the same example and projection."""
    errors: List[str] = []
    with np.load(feature_path) as features, np.load(label_path) as labels:
        if expected_chain_id is not None:
            for data in (features, labels):
                value = data["chain_id"] if "chain_id" in data.files else None
                errors.extend(_chain_identity_errors(value, expected_chain_id, required=require_chain_id))
        if features["aatype"].shape != labels["ca_mask"].shape:
            errors.append("Feature and label residue counts must match")
        for key in (
            "chain_id", "residue_index", "preprocess_config_sha256", "source_inputs_sha256",
            "resolved_author_chain_id", "chain_mapping_method",
        ):
            if key in features.files and key in labels.files:
                if not np.array_equal(features[key], labels[key]):
                    errors.append(f"Feature and label `{key}` must match")
            elif key in {"preprocess_config_sha256", "source_inputs_sha256"} and (key in features.files or key in labels.files):
                errors.append(f"Both features and labels must record `{key}`")
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
    expected_preprocess_config_sha256: str | None = None,
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

        if validate_schema:
            with np.load(npz_path) as data:
                if expected_preprocess_config_sha256 is None and any(
                    key in data.files for key in ("preprocess_config_sha256", "source_inputs_sha256")
                ):
                    raise ValueError(
                        "Provenance-bearing outputs require schema-version 3 preprocessing metadata: "
                        f"{npz_path}"
                    )
                value = data["chain_id"] if "chain_id" in data.files else None
                identity_errors = _chain_identity_errors(
                    value, chain_id, required=expected_preprocess_config_sha256 is not None,
                )
                if identity_errors:
                    raise ValueError(f"Invalid identity for {npz_path}: {'; '.join(identity_errors)}")

        if expected_preprocess_config_sha256 is not None:
            with np.load(npz_path) as data:
                for key in ("preprocess_config_sha256", "source_inputs_sha256"):
                    if key not in data.files or data[key].shape != ():
                        raise ValueError(f"Missing scalar `{key}` in {npz_path}")
                    value = str(data[key].item())
                    if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
                        raise ValueError(f"Invalid `{key}` in {npz_path}")
                if str(data["preprocess_config_sha256"].item()) != expected_preprocess_config_sha256:
                    raise ValueError(f"Preprocessing provenance mismatch in {npz_path}")

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
        if key in normalized:
            raise ValueError(f"Duplicate normalized manifest split name: {key!r}")
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


def _stable_preprocess_meta_sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    try:
        raw = json.loads(path.read_text())
    except json.JSONDecodeError:
        return sha256_file(path)
    if not isinstance(raw, dict):
        return sha256_file(path)

    return preprocess_config_sha256(raw)


def preprocess_config_sha256(raw: Mapping[str, Any]) -> str:
    """Hash data-producing settings independently of machine-local paths."""
    raw_cli_args = raw.get("cli_args")
    cli_args: Mapping[str, Any] = raw_cli_args if isinstance(raw_cli_args, dict) else {}
    stable_cli_keys = (
        "allow_failures",
        "chain_data_cache_sha256",
        "disable_templates",
        "fail_fast",
        "max_msa_seqs",
        "max_templates",
        "min_projection_aligned_fraction",
        "min_projection_coverage",
        "min_projection_seq_identity",
        "min_projection_valid_ca",
        "msa_name",
        "msa_names",
        "msa_row_filter_sha256",
        "strict",
        "template_hhr_name",
    )
    stable = {
        "aligner": raw.get("aligner"),
        "atom14_num_slots": raw.get("atom14_num_slots"),
        "ca_atom14_slot": raw.get("ca_atom14_slot"),
        "cli_args": {key: cli_args.get(key) for key in stable_cli_keys if key in cli_args},
        "schema_version": raw.get("schema_version"),
    }
    if "implementation_sha256" in raw:
        stable["implementation_sha256"] = raw["implementation_sha256"]
    payload = json.dumps(stable, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


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
    if require_labels and not processed_labels_dir:
        raise ValueError("Label fingerprinting requires `processed_labels_dir`.")
    processed_features_dir = Path(processed_features_dir).resolve()
    processed_labels_dir_path = Path(processed_labels_dir).resolve() if processed_labels_dir else None
    manifest_map = _normalize_manifest_map(manifest_paths)
    preprocess_meta_path = processed_features_dir / PREPROCESS_META_FILENAME
    preprocess_config_hash = _stable_preprocess_meta_sha256(preprocess_meta_path)
    expected_preprocess_config_hash = None
    if validate_schema and preprocess_meta_path.exists():
        raw_meta = json.loads(preprocess_meta_path.read_text())
        if isinstance(raw_meta, dict) and raw_meta.get("schema_version") == 3:
            expected_preprocess_config_hash = preprocess_config_hash

    manifest_meta: Dict[str, Dict[str, Any]] = {}
    manifest_chain_ids: Dict[str, set[str]] = {}
    all_chain_ids_unsorted: List[str] = []
    for split_name, manifest_path in manifest_map.items():
        chain_ids = read_manifest(manifest_path)
        current_chain_ids = set(chain_ids)
        for previous_split, previous_chain_ids in manifest_chain_ids.items():
            overlap_count = len(current_chain_ids & previous_chain_ids)
            if overlap_count:
                raise ValueError(
                    "Manifest splits must be chain-disjoint: "
                    f"{previous_split!r} and {split_name!r} share {overlap_count} chain IDs."
                )
        manifest_chain_ids[split_name] = current_chain_ids
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
        expected_preprocess_config_sha256=expected_preprocess_config_hash,
    )
    label_hash = _files_sha256(
        base_dir=processed_labels_dir_path if require_labels else None,
        chain_ids=all_chain_ids,
        missing_chain_ids=missing_label_chain_ids,
        validate_schema=validate_schema,
        schema_validator=validate_label_npz_schema,
        expected_preprocess_config_sha256=expected_preprocess_config_hash,
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

    if validate_schema and require_labels and processed_labels_dir_path is not None:
        for chain_id in all_chain_ids:
            feature_path = chain_npz_path(processed_features_dir, chain_id)
            label_path = chain_npz_path(processed_labels_dir_path, chain_id)
            if feature_path.exists() and label_path.exists():
                pair_errors = validate_feature_label_pair(
                    feature_path, label_path, expected_chain_id=chain_id,
                    require_chain_id=expected_preprocess_config_hash is not None,
                )
                if pair_errors:
                    raise ValueError(f"Invalid feature/label pair for {chain_id}: {'; '.join(pair_errors)}")

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
        "preprocess_config_sha256": preprocess_config_hash,
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
