from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from nanofold.dataset_integrity import (
    build_dataset_fingerprint,
    build_split_fingerprint,
    validate_label_npz_schema,
    validate_npz_schema,
    verify_dataset_against_fingerprint,
    verify_split_against_fingerprint,
)


def _write_feature_npz(path: Path, *, bad_dtype: bool = False) -> None:
    np.savez_compressed(
        path,
        aatype=np.zeros((6,), dtype=np.float32 if bad_dtype else np.int32),
        msa=np.zeros((4, 6), dtype=np.int32),
        deletions=np.zeros((4, 6), dtype=np.int32),
        template_aatype=np.zeros((1, 6), dtype=np.int32),
        template_ca_coords=np.zeros((1, 6, 3), dtype=np.float32),
        template_ca_mask=np.ones((1, 6), dtype=bool),
    )


def _write_label_npz(path: Path) -> None:
    np.savez_compressed(
        path,
        ca_coords=np.zeros((6, 3), dtype=np.float32),
        ca_mask=np.ones((6,), dtype=bool),
    )


def test_validate_npz_schema_flags_bad_dtype(tmp_path: Path) -> None:
    bad = tmp_path / "bad.npz"
    _write_feature_npz(bad, bad_dtype=True)
    errors = validate_npz_schema(bad)
    assert any("aatype" in e for e in errors)


def test_validate_label_npz_schema_roundtrip(tmp_path: Path) -> None:
    labels = tmp_path / "labels.npz"
    _write_label_npz(labels)
    assert validate_label_npz_schema(labels) == []


def test_build_fingerprint_requires_present_files_when_requested(tmp_path: Path) -> None:
    manifests = tmp_path / "manifests"
    manifests.mkdir()
    features = tmp_path / "processed_features"
    labels = tmp_path / "processed_labels"
    features.mkdir()
    labels.mkdir()

    (manifests / "train.txt").write_text("1abc_A\n")
    (manifests / "val.txt").write_text("2def_B\n")
    _write_feature_npz(features / "1abc_A.npz")
    _write_label_npz(labels / "1abc_A.npz")
    # 2def_B intentionally missing

    with pytest.raises(FileNotFoundError):
        build_dataset_fingerprint(
            processed_features_dir=features,
            processed_labels_dir=labels,
            train_manifest=manifests / "train.txt",
            val_manifest=manifests / "val.txt",
            require_no_missing=True,
            require_labels=True,
        )


def test_verify_dataset_against_fingerprint_roundtrip(tmp_path: Path) -> None:
    manifests = tmp_path / "manifests"
    manifests.mkdir()
    features = tmp_path / "processed_features"
    labels = tmp_path / "processed_labels"
    features.mkdir()
    labels.mkdir()

    (manifests / "train.txt").write_text("1abc_A\n")
    (manifests / "val.txt").write_text("2def_B\n")
    _write_feature_npz(features / "1abc_A.npz")
    _write_feature_npz(features / "2def_B.npz")
    _write_label_npz(labels / "1abc_A.npz")
    _write_label_npz(labels / "2def_B.npz")

    fingerprint = build_dataset_fingerprint(
        processed_features_dir=features,
        processed_labels_dir=labels,
        train_manifest=manifests / "train.txt",
        val_manifest=manifests / "val.txt",
        require_no_missing=True,
        require_labels=True,
        track_id="limited_large_v3",
    )
    fp_path = tmp_path / "fp.json"
    fp_path.write_text(json.dumps(fingerprint))

    out = verify_dataset_against_fingerprint(
        processed_features_dir=features,
        processed_labels_dir=labels,
        train_manifest=manifests / "train.txt",
        val_manifest=manifests / "val.txt",
        expected_fingerprint_path=fp_path,
        require_no_missing=True,
        require_labels=True,
        track_id="limited_large_v3",
    )
    assert out["missing_feature_chain_count"] == 0
    assert out["missing_label_chain_count"] == 0


def test_verify_split_against_fingerprint_uses_explicit_manifest_names(tmp_path: Path) -> None:
    manifests = tmp_path / "manifests"
    manifests.mkdir()
    features = tmp_path / "processed_features"
    features.mkdir()

    hidden_manifest = manifests / "hidden_val.txt"
    hidden_manifest.write_text("9xyz_A\n")
    _write_feature_npz(features / "9xyz_A.npz")

    fingerprint = build_split_fingerprint(
        processed_features_dir=features,
        manifest_paths={"hidden_val": hidden_manifest},
        require_no_missing=True,
        require_labels=False,
    )
    fp_path = tmp_path / "hidden_fp.json"
    fp_path.write_text(json.dumps(fingerprint))

    out = verify_split_against_fingerprint(
        processed_features_dir=features,
        manifest_paths={"hidden_val": hidden_manifest},
        expected_fingerprint_path=fp_path,
        require_no_missing=True,
        require_labels=False,
    )
    assert out["split_names"] == ["hidden_val"]
    assert out["manifests"]["hidden_val"]["chain_count"] == 1


def test_verify_split_against_fingerprint_supports_features_only_comparison(tmp_path: Path) -> None:
    manifests = tmp_path / "manifests"
    manifests.mkdir()
    features = tmp_path / "processed_features"
    labels = tmp_path / "processed_labels"
    features.mkdir()
    labels.mkdir()

    (manifests / "train.txt").write_text("1abc_A\n")
    (manifests / "val.txt").write_text("2def_B\n")
    _write_feature_npz(features / "1abc_A.npz")
    _write_feature_npz(features / "2def_B.npz")
    _write_label_npz(labels / "1abc_A.npz")
    _write_label_npz(labels / "2def_B.npz")

    fingerprint = build_dataset_fingerprint(
        processed_features_dir=features,
        processed_labels_dir=labels,
        train_manifest=manifests / "train.txt",
        val_manifest=manifests / "val.txt",
        require_no_missing=True,
        require_labels=True,
        track_id="limited_large_v3",
    )
    fp_path = tmp_path / "fp.json"
    fp_path.write_text(json.dumps(fingerprint))

    out = verify_split_against_fingerprint(
        processed_features_dir=features,
        processed_labels_dir=None,
        manifest_paths={"train": manifests / "train.txt", "val": manifests / "val.txt"},
        expected_fingerprint_path=fp_path,
        require_no_missing=True,
        require_labels=False,
        track_id="limited_large_v3",
        comparison_mode="features_only",
    )
    assert out["present_feature_chain_count"] == 2
