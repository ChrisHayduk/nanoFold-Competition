from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from nanofold.chain_paths import chain_npz_path
from nanofold.data import LABEL_ONLY_KEYS, ProcessedNPZDataset


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
    _write_feature_npz(chain_npz_path(features, "1abc_A"))

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
    _write_feature_npz(chain_npz_path(features, "1abc_A"))
    _write_label_npz(chain_npz_path(labels, "1abc_A"))

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
    _write_feature_npz(chain_npz_path(features, "1abc_A"))

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


@pytest.mark.parametrize("key", LABEL_ONLY_KEYS)
def test_features_only_dataset_rejects_label_members_without_fingerprint(tmp_path: Path, key: str) -> None:
    features = tmp_path / "features"
    features.mkdir()
    manifest = tmp_path / "val.txt"
    manifest.write_text("1abc_A\n")
    contents: dict[str, Any] = {key: np.zeros(1)}
    np.savez_compressed(chain_npz_path(features, "1abc_A"), **contents)
    with pytest.raises(ValueError, match="label-only fields"):
        ProcessedNPZDataset(features, manifest, include_labels=False)


@pytest.mark.parametrize("defect", ["feature_identity", "label_identity", "length", "residue_index",
                                  "mask_type", "mask_disagreement", "coordinate_disagreement", "nonfinite"])
def test_dataset_rejects_mispaired_supervision(tmp_path: Path, defect: str) -> None:
    features, labels = tmp_path / "features", tmp_path / "labels"
    features.mkdir()
    labels.mkdir()
    manifest = tmp_path / "train.txt"
    manifest.write_text("1abc_A\n")
    feature_path, label_path = chain_npz_path(features, "1abc_A"), chain_npz_path(labels, "1abc_A")
    _write_feature_npz(feature_path)
    _write_label_npz(label_path)
    path = feature_path if defect == "feature_identity" else label_path
    with np.load(path) as archive:
        contents = dict(archive)
    if defect.endswith("identity"):
        contents["chain_id"] = np.asarray("2def_A")
        match = "chain_id does not match"
    elif defect == "length":
        contents["atom14_positions"] = np.zeros((7, 14, 3), dtype=np.float32)
        match = "must have shape"
    elif defect == "residue_index":
        contents["residue_index"] = np.arange(8, dtype=np.int32) + 2
        match = "residue_index disagree"
    elif defect == "mask_type":
        contents["ca_mask"] = np.ones(8, dtype=np.int32)
        match = "boolean dtype"
    elif defect == "mask_disagreement":
        contents["atom14_mask"][0, 1] = False
        match = "matching CA masks"
    elif defect == "coordinate_disagreement":
        contents["ca_coords"][0, 0] = 1
        match = "coordinates disagree"
    else:
        contents["atom14_positions"][0, 0, 0] = np.nan
        match = "finite floating point"
    np.savez_compressed(path, **contents)
    dataset = ProcessedNPZDataset(features, manifest, processed_labels_dir=labels)
    with pytest.raises(ValueError, match=match):
        dataset[0]


def test_dataset_rejects_label_only_residue_gaps_when_feature_indices_absent(tmp_path: Path) -> None:
    features, labels = tmp_path / "features", tmp_path / "labels"
    features.mkdir()
    labels.mkdir()
    manifest = tmp_path / "train.txt"
    manifest.write_text("1abc_A\n")
    feature_path, label_path = chain_npz_path(features, "1abc_A"), chain_npz_path(labels, "1abc_A")
    _write_feature_npz(feature_path)
    _write_label_npz(label_path)
    with np.load(feature_path) as archive:
        contents = {key: archive[key] for key in archive.files if key != "residue_index"}
    np.savez_compressed(feature_path, **contents)
    indices = np.asarray([0, 1, 2, 5, 6, 7, 8, 9], dtype=np.int32)
    with np.load(label_path) as archive:
        contents = dict(archive)
    contents["residue_index"] = indices
    np.savez_compressed(label_path, **contents)
    # Inference on these feature bytes would see arange(8). Label-only geometry
    # must not change the model inputs on the labeled evaluation path.
    with pytest.raises(ValueError, match="residue_index disagree"):
        ProcessedNPZDataset(features, manifest, processed_labels_dir=labels)[0]
