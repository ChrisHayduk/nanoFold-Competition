from __future__ import annotations

from pathlib import Path

import numpy as np

from nanofold.data import ProcessedNPZDataset, collate_batch
from nanofold.dataset_integrity import validate_label_npz_schema, validate_npz_schema


def _write_feature_npz(path: Path, L: int = 8, N: int = 4, T: int = 1) -> None:
    np.savez_compressed(
        path,
        aatype=np.zeros((L,), dtype=np.int32),
        msa=np.zeros((N, L), dtype=np.int32),
        deletions=np.zeros((N, L), dtype=np.int32),
        template_aatype=np.zeros((T, L), dtype=np.int32),
        template_ca_coords=np.zeros((T, L, 3), dtype=np.float32),
        template_ca_mask=np.ones((T, L), dtype=bool),
    )


def _write_label_npz(path: Path, L: int = 8) -> None:
    arrays = {
        "ca_coords": np.zeros((L, 3), dtype=np.float32),
        "ca_mask": np.ones((L,), dtype=bool),
        "atom14_positions": np.zeros((L, 14, 3), dtype=np.float32),
        "atom14_mask": np.ones((L, 14), dtype=bool),
        "residue_index": np.arange(L, dtype=np.int32),
        "resolution": np.asarray(2.0, dtype=np.float32),
    }
    np.savez_compressed(path, **arrays)


def test_processed_npz_shapes_and_collate(tmp_path: Path) -> None:
    processed_features = tmp_path / "processed_features"
    processed_labels = tmp_path / "processed_labels"
    processed_features.mkdir()
    processed_labels.mkdir()
    manifest = tmp_path / "train.txt"
    cid = "1abc_A"
    manifest.write_text(f"{cid}\n")
    feature_path = processed_features / f"{cid}.npz"
    label_path = processed_labels / f"{cid}.npz"
    _write_feature_npz(feature_path, L=12, N=5, T=2)
    _write_label_npz(label_path, L=12)

    assert validate_npz_schema(feature_path) == []
    assert validate_label_npz_schema(label_path) == []

    ds = ProcessedNPZDataset(
        processed_features_dir=processed_features,
        processed_labels_dir=processed_labels,
        include_labels=True,
        manifest_path=manifest,
        allow_missing=False,
    )
    item = ds[0]
    batch = collate_batch([item], crop_size=12, msa_depth=5, crop_mode="center", msa_sample_mode="top")
    assert batch["aatype"].shape == (1, 12)
    assert batch["msa"].shape == (1, 5, 12)
    assert batch["ca_coords"].shape == (1, 12, 3)
    assert batch["template_ca_coords"].shape == (1, 2, 12, 3)
    assert batch["atom14_positions"].shape == (1, 12, 14, 3)
    assert batch["atom14_mask"].shape == (1, 12, 14)
    assert batch["residue_index"].shape == (1, 12)
    assert batch["resolution"].shape == (1,)


def test_processed_npz_carries_atom14_when_present(tmp_path: Path) -> None:
    processed_features = tmp_path / "processed_features"
    processed_labels = tmp_path / "processed_labels"
    processed_features.mkdir()
    processed_labels.mkdir()
    manifest = tmp_path / "train.txt"
    cid = "2xyz_B"
    manifest.write_text(f"{cid}\n")
    _write_feature_npz(processed_features / f"{cid}.npz", L=10, N=3, T=1)
    _write_label_npz(processed_labels / f"{cid}.npz", L=10)

    assert validate_label_npz_schema(processed_labels / f"{cid}.npz") == []

    ds = ProcessedNPZDataset(
        processed_features_dir=processed_features,
        processed_labels_dir=processed_labels,
        include_labels=True,
        manifest_path=manifest,
        allow_missing=False,
    )
    item = ds[0]
    batch = collate_batch([item], crop_size=10, msa_depth=3, crop_mode="center", msa_sample_mode="top")
    assert batch["atom14_positions"].shape == (1, 10, 14, 3)
    assert batch["atom14_mask"].shape == (1, 10, 14)
    assert bool(batch["atom14_mask"].all())
    assert batch["residue_index"].shape == (1, 10)
    assert batch["resolution"].shape == (1,)
    assert float(batch["resolution"][0]) == 2.0


def test_processed_npz_crop_preserves_atom14(tmp_path: Path) -> None:
    processed_features = tmp_path / "processed_features"
    processed_labels = tmp_path / "processed_labels"
    processed_features.mkdir()
    processed_labels.mkdir()
    manifest = tmp_path / "train.txt"
    cid = "3qqq_C"
    manifest.write_text(f"{cid}\n")
    _write_feature_npz(processed_features / f"{cid}.npz", L=20, N=2, T=1)
    _write_label_npz(processed_labels / f"{cid}.npz", L=20)

    ds = ProcessedNPZDataset(
        processed_features_dir=processed_features,
        processed_labels_dir=processed_labels,
        include_labels=True,
        manifest_path=manifest,
        allow_missing=False,
    )
    item = ds[0]
    batch = collate_batch([item], crop_size=7, msa_depth=2, crop_mode="center", msa_sample_mode="top")
    assert batch["aatype"].shape == (1, 7)
    assert batch["atom14_positions"].shape == (1, 7, 14, 3)
    assert batch["atom14_mask"].shape == (1, 7, 14)
    assert batch["residue_index"].shape == (1, 7)
