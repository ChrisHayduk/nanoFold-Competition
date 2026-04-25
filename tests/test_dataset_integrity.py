from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from nanofold.chain_paths import chain_npz_path
from nanofold.dataset_integrity import (
    PREPROCESS_META_FILENAME,
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
        atom14_positions=np.zeros((6, 14, 3), dtype=np.float32),
        atom14_mask=np.ones((6, 14), dtype=bool),
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
    _write_feature_npz(chain_npz_path(features, "1abc_A"))
    _write_label_npz(chain_npz_path(labels, "1abc_A"))
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
    _write_feature_npz(chain_npz_path(features, "1abc_A"))
    _write_feature_npz(chain_npz_path(features, "2def_B"))
    _write_label_npz(chain_npz_path(labels, "1abc_A"))
    _write_label_npz(chain_npz_path(labels, "2def_B"))

    fingerprint = build_dataset_fingerprint(
        processed_features_dir=features,
        processed_labels_dir=labels,
        train_manifest=manifests / "train.txt",
        val_manifest=manifests / "val.txt",
        require_no_missing=True,
        require_labels=True,
        track_id="limited",
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
        track_id="limited",
    )
    assert out["missing_feature_chain_count"] == 0
    assert out["missing_label_chain_count"] == 0


def test_verify_dataset_against_fingerprint_uses_recorded_source_lock(tmp_path: Path) -> None:
    manifests = tmp_path / "manifests"
    manifests.mkdir()
    features = tmp_path / "processed_features"
    labels = tmp_path / "processed_labels"
    features.mkdir()
    labels.mkdir()
    source_lock = tmp_path / "official_manifest_source.lock.json"
    source_lock.write_text('{"source": "test"}\n')

    (manifests / "train.txt").write_text("1abc_A\n")
    (manifests / "val.txt").write_text("2def_B\n")
    _write_feature_npz(chain_npz_path(features, "1abc_A"))
    _write_feature_npz(chain_npz_path(features, "2def_B"))
    _write_label_npz(chain_npz_path(labels, "1abc_A"))
    _write_label_npz(chain_npz_path(labels, "2def_B"))

    fingerprint = build_dataset_fingerprint(
        processed_features_dir=features,
        processed_labels_dir=labels,
        train_manifest=manifests / "train.txt",
        val_manifest=manifests / "val.txt",
        require_no_missing=True,
        require_labels=True,
        track_id="limited",
        source_lock_path=source_lock,
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
        track_id="limited",
    )

    assert out["source_lock_sha256"] == fingerprint["source_lock_sha256"]


def test_verify_split_against_fingerprint_uses_explicit_manifest_names(tmp_path: Path) -> None:
    manifests = tmp_path / "manifests"
    manifests.mkdir()
    features = tmp_path / "processed_features"
    features.mkdir()

    hidden_manifest = manifests / "hidden_val.txt"
    hidden_manifest.write_text("9xyz_A\n")
    _write_feature_npz(chain_npz_path(features, "9xyz_A"))

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
    _write_feature_npz(chain_npz_path(features, "1abc_A"))
    _write_feature_npz(chain_npz_path(features, "2def_B"))
    _write_label_npz(chain_npz_path(labels, "1abc_A"))
    _write_label_npz(chain_npz_path(labels, "2def_B"))

    fingerprint = build_dataset_fingerprint(
        processed_features_dir=features,
        processed_labels_dir=labels,
        train_manifest=manifests / "train.txt",
        val_manifest=manifests / "val.txt",
        require_no_missing=True,
        require_labels=True,
        track_id="limited",
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
        track_id="limited",
        comparison_mode="features_only",
    )
    assert out["present_feature_chain_count"] == 2


def test_validate_label_npz_schema_accepts_atom14_fields(tmp_path: Path) -> None:
    labels = tmp_path / "labels_rich.npz"
    L = 5
    np.savez_compressed(
        labels,
        ca_coords=np.zeros((L, 3), dtype=np.float32),
        ca_mask=np.ones((L,), dtype=bool),
        atom14_positions=np.zeros((L, 14, 3), dtype=np.float32),
        atom14_mask=np.ones((L, 14), dtype=bool),
        residue_index=np.arange(L, dtype=np.int32),
        resolution=np.asarray(2.3, dtype=np.float32),
    )
    assert validate_label_npz_schema(labels) == []


def test_validate_label_npz_schema_rejects_wrong_atom14_shape(tmp_path: Path) -> None:
    labels = tmp_path / "labels_bad.npz"
    L = 5
    np.savez_compressed(
        labels,
        ca_coords=np.zeros((L, 3), dtype=np.float32),
        ca_mask=np.ones((L,), dtype=bool),
        atom14_positions=np.zeros((L, 12, 3), dtype=np.float32),  # wrong — should be 14 slots
        atom14_mask=np.ones((L, 14), dtype=bool),
    )
    errors = validate_label_npz_schema(labels)
    assert any("atom14_positions" in e for e in errors)


def test_build_fingerprint_captures_preprocess_config_sha256(tmp_path: Path) -> None:
    manifests = tmp_path / "manifests"
    manifests.mkdir()
    features = tmp_path / "processed_features"
    labels = tmp_path / "processed_labels"
    features.mkdir()
    labels.mkdir()

    (manifests / "train.txt").write_text("1abc_A\n")
    (manifests / "val.txt").write_text("2def_B\n")
    _write_feature_npz(chain_npz_path(features, "1abc_A"))
    _write_feature_npz(chain_npz_path(features, "2def_B"))
    _write_label_npz(chain_npz_path(labels, "1abc_A"))
    _write_label_npz(chain_npz_path(labels, "2def_B"))

    meta_path = features / PREPROCESS_META_FILENAME
    meta_path.write_text('{"cli_args": {"strict": true}}\n')

    fingerprint = build_dataset_fingerprint(
        processed_features_dir=features,
        processed_labels_dir=labels,
        train_manifest=manifests / "train.txt",
        val_manifest=manifests / "val.txt",
        require_no_missing=True,
        require_labels=True,
        track_id="limited",
    )
    assert isinstance(fingerprint["preprocess_config_sha256"], str)
    assert len(fingerprint["preprocess_config_sha256"]) == 64

    # Changing the meta file should change the fingerprint field.
    meta_path.write_text('{"cli_args": {"strict": false}}\n')
    fingerprint_changed = build_dataset_fingerprint(
        processed_features_dir=features,
        processed_labels_dir=labels,
        train_manifest=manifests / "train.txt",
        val_manifest=manifests / "val.txt",
        require_no_missing=True,
        require_labels=True,
        track_id="limited",
    )
    assert fingerprint_changed["preprocess_config_sha256"] != fingerprint["preprocess_config_sha256"]
