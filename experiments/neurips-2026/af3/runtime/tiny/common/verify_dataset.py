#!/usr/bin/env python3
"""Verify a staged NanoFold train/public-validation tensor snapshot.

The tree digest is stable across filesystems: it hashes each relative path,
file size, and file SHA-256 in lexicographic order. The script is intentionally
read-only except for its optional JSON report.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np


FEATURE_KEYS = {
    "aatype",
    "msa",
    "deletions",
    "residue_index",
    "between_segment_residues",
}
LABEL_KEYS = {"ca_coords", "ca_mask", "atom14_positions", "atom14_mask"}


def sha256_file(path: Path, chunk_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def read_manifest(path: Path) -> list[str]:
    ids = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if len(ids) != len(set(ids)):
        raise ValueError(f"duplicate chain IDs in {path}")
    return ids


def chain_stem(chain_id: str) -> str:
    return "chain_" + chain_id.encode("utf-8").hex()


def inspect_npz(feature_path: Path, label_path: Path) -> dict[str, Any]:
    with np.load(feature_path, allow_pickle=False) as features:
        missing = sorted(FEATURE_KEYS.difference(features.files))
        if missing:
            raise ValueError(f"{feature_path}: missing feature keys {missing}")
        length = int(features["aatype"].shape[0])
        msa_shape = [int(v) for v in features["msa"].shape]
        if msa_shape[-1] != length:
            raise ValueError(f"{feature_path}: MSA length does not match aatype")
        feature_schema = {
            key: {"shape": list(features[key].shape), "dtype": str(features[key].dtype)}
            for key in sorted(FEATURE_KEYS)
        }

    with np.load(label_path, allow_pickle=False) as labels:
        missing = sorted(LABEL_KEYS.difference(labels.files))
        if missing:
            raise ValueError(f"{label_path}: missing label keys {missing}")
        if labels["atom14_positions"].shape != (length, 14, 3):
            raise ValueError(f"{label_path}: unexpected atom14_positions shape")
        if labels["atom14_mask"].shape != (length, 14):
            raise ValueError(f"{label_path}: unexpected atom14_mask shape")
        label_schema = {
            key: {"shape": list(labels[key].shape), "dtype": str(labels[key].dtype)}
            for key in sorted(LABEL_KEYS)
        }

    return {
        "length": length,
        "msa_shape": msa_shape,
        "feature_schema": feature_schema,
        "label_schema": label_schema,
    }


def tree_inventory(root: Path, relative_paths: list[Path]) -> dict[str, Any]:
    tree_digest = hashlib.sha256()
    entries: list[dict[str, Any]] = []
    total_bytes = 0
    for relative_path in sorted(relative_paths, key=lambda value: value.as_posix()):
        path = root / relative_path
        size = path.stat().st_size
        file_digest = sha256_file(path)
        total_bytes += size
        entry = {
            "path": relative_path.as_posix(),
            "bytes": size,
            "sha256": file_digest,
        }
        entries.append(entry)
        tree_digest.update(
            f"{entry['path']}\0{entry['bytes']}\0{entry['sha256']}\n".encode("utf-8")
        )
    return {
        "file_count": len(entries),
        "total_bytes": total_bytes,
        "tree_sha256": tree_digest.hexdigest(),
        "files": entries,
    }


def dataset_relative_paths(
    root: Path,
    train_ids: list[str],
    validation_ids: list[str],
) -> list[Path]:
    """Resolve the exact manifest-bound tensor tree and reject extras."""
    relative_paths = [
        Path("manifests/train.txt"),
        Path("manifests/val.txt"),
    ]
    for chain_id in train_ids + validation_ids:
        stem = chain_stem(chain_id) + ".npz"
        relative_paths.extend(
            (
                Path("processed_features") / stem,
                Path("processed_labels") / stem,
            )
        )
    missing = [path for path in relative_paths if not (root / path).is_file()]
    if missing:
        raise FileNotFoundError(root / missing[0])

    expected_tensors = {path for path in relative_paths if path.suffix == ".npz"}
    observed_tensors = {
        path.relative_to(root)
        for directory in ("processed_features", "processed_labels")
        for path in (root / directory).glob("*.npz")
    }
    missing_tensors = sorted(expected_tensors - observed_tensors)
    extra_tensors = sorted(observed_tensors - expected_tensors)
    if missing_tensors or extra_tensors:
        raise ValueError(
            "staged tensor files differ from manifest-bound tree: "
            f"missing={missing_tensors[:5]} ({len(missing_tensors)}), "
            f"extra={extra_tensors[:5]} ({len(extra_tensors)})"
        )
    return relative_paths


def verify_dataset_snapshot(
    root: Path,
    expected_report: dict[str, Any],
) -> dict[str, Any]:
    """Rehash a staged dataset and compare it with an immutable report."""
    root = root.resolve()
    train_path = root / "manifests" / "train.txt"
    validation_path = root / "manifests" / "val.txt"
    train_ids = read_manifest(train_path)
    validation_ids = read_manifest(validation_path)
    overlap = set(train_ids).intersection(validation_ids)
    if overlap:
        raise ValueError(
            f"train and validation manifests overlap: {sorted(overlap)[:5]}"
        )

    observed_counts = {
        "train_count": len(train_ids),
        "validation_count": len(validation_ids),
    }
    expected_counts = {name: int(expected_report[name]) for name in observed_counts}
    if observed_counts != expected_counts:
        raise ValueError(
            "staged dataset counts differ from verification report: "
            f"expected={expected_counts}, observed={observed_counts}"
        )

    observed_manifests = {
        "train": sha256_file(train_path),
        "validation": sha256_file(validation_path),
    }
    expected_manifests = {
        name: str(expected_report["manifest_sha256"][name])
        for name in observed_manifests
    }
    if observed_manifests != expected_manifests:
        raise ValueError(
            "staged dataset manifests differ from verification report: "
            f"expected={expected_manifests}, observed={observed_manifests}"
        )

    relative_paths = dataset_relative_paths(root, train_ids, validation_ids)
    inventory = tree_inventory(root, relative_paths)
    observed_inventory = {
        name: inventory[name] for name in ("file_count", "total_bytes", "tree_sha256")
    }
    expected_inventory = {
        name: expected_report["inventory"][name] for name in observed_inventory
    }
    expected_inventory["file_count"] = int(expected_inventory["file_count"])
    expected_inventory["total_bytes"] = int(expected_inventory["total_bytes"])
    expected_inventory["tree_sha256"] = str(expected_inventory["tree_sha256"])
    if observed_inventory != expected_inventory:
        raise ValueError(
            "staged tensor tree differs from verification report: "
            f"expected={expected_inventory}, observed={observed_inventory}"
        )
    return {
        **observed_counts,
        "manifest_sha256": observed_manifests,
        "inventory": observed_inventory,
    }


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--include-file-inventory", action="store_true")
    args = parser.parse_args()

    root = args.data_root.resolve()
    feature_root = root / "processed_features"
    label_root = root / "processed_labels"
    manifest_root = root / "manifests"
    train_path = manifest_root / "train.txt"
    val_path = manifest_root / "val.txt"

    train_ids = read_manifest(train_path)
    val_ids = read_manifest(val_path)
    if set(train_ids).intersection(val_ids):
        raise ValueError("train and validation manifests overlap")
    if len(train_ids) != 10_000 or len(val_ids) != 1_000:
        raise ValueError(
            f"expected 10,000/1,000 train/validation IDs, got "
            f"{len(train_ids):,}/{len(val_ids):,}"
        )

    relative_paths = dataset_relative_paths(root, train_ids, val_ids)
    inventory = tree_inventory(root, relative_paths)
    samples = {}
    for split_name, chain_id in (("train", train_ids[0]), ("validation", val_ids[0])):
        stem = chain_stem(chain_id) + ".npz"
        samples[split_name] = {
            "chain_id": chain_id,
            **inspect_npz(feature_root / stem, label_root / stem),
        }

    report: dict[str, Any] = {
        "schema_version": 1,
        "data_root": str(root),
        "train_count": len(train_ids),
        "validation_count": len(val_ids),
        "manifest_sha256": {
            "train": sha256_file(train_path),
            "validation": sha256_file(val_path),
        },
        "inventory": inventory,
        "sample_schema": samples,
    }
    if not args.include_file_inventory:
        report["inventory"] = {
            key: value for key, value in inventory.items() if key != "files"
        }

    if args.output:
        atomic_write_json(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
