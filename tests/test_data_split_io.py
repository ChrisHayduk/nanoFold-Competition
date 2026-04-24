from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from nanofold.data import ProcessedNPZDataset


def _write_feature_npz(path: Path) -> None:
    np.savez_compressed(
        path,
        aatype=np.zeros((8,), dtype=np.int32),
        msa=np.zeros((4, 8), dtype=np.int32),
        deletions=np.zeros((4, 8), dtype=np.int32),
        residue_index=np.arange(8, dtype=np.int32),
        between_segment_residues=np.zeros((8,), dtype=np.int32),
        template_aatype=np.zeros((1, 8), dtype=np.int32),
        template_ca_coords=np.zeros((1, 8, 3), dtype=np.float32),
        template_ca_mask=np.ones((1, 8), dtype=bool),
    )


def _write_label_npz(path: Path) -> None:
    np.savez_compressed(
        path,
        ca_coords=np.zeros((8, 3), dtype=np.float32),
        ca_mask=np.ones((8,), dtype=bool),
        atom14_positions=np.zeros((8, 14, 3), dtype=np.float32),
        atom14_mask=np.ones((8, 14), dtype=bool),
    )


def test_train_path_requires_labels(tmp_path: Path) -> None:
    features = tmp_path / "features"
    labels = tmp_path / "labels"
    features.mkdir()
    labels.mkdir()
    manifest = tmp_path / "train.txt"
    manifest.write_text("1abc_A\n")
    _write_feature_npz(features / "1abc_A.npz")

    with pytest.raises(FileNotFoundError):
        _ = ProcessedNPZDataset(
            processed_features_dir=features,
            processed_labels_dir=labels,
            include_labels=True,
            manifest_path=manifest,
            allow_missing=False,
        )


def test_features_only_eval_rejects_labels_when_forbidden(tmp_path: Path) -> None:
    features = tmp_path / "features"
    labels = tmp_path / "labels"
    features.mkdir()
    labels.mkdir()
    manifest = tmp_path / "val.txt"
    manifest.write_text("1abc_A\n")
    _write_feature_npz(features / "1abc_A.npz")
    _write_label_npz(labels / "1abc_A.npz")

    with pytest.raises(RuntimeError, match="Labels are mounted"):
        _ = ProcessedNPZDataset(
            processed_features_dir=features,
            processed_labels_dir=labels,
            include_labels=False,
            fail_if_labels_present=True,
            manifest_path=manifest,
            allow_missing=False,
        )


def test_features_only_eval_has_no_supervision_keys(tmp_path: Path) -> None:
    features = tmp_path / "features"
    features.mkdir()
    manifest = tmp_path / "val.txt"
    manifest.write_text("1abc_A\n")
    _write_feature_npz(features / "1abc_A.npz")

    ds = ProcessedNPZDataset(
        processed_features_dir=features,
        processed_labels_dir=None,
        include_labels=False,
        manifest_path=manifest,
        allow_missing=False,
    )
    item = ds[0]
    assert "ca_coords" not in item
    assert "ca_mask" not in item
    assert item["residue_index"].shape == (8,)
    assert item["between_segment_residues"].shape == (8,)
