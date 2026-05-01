from __future__ import annotations

from pathlib import Path

import numpy as np

from nanofold.chain_paths import chain_npz_path
from scripts.upload_hf_public_dataset import generate_rows, render_dataset_card


def _write_npz_pair(features_dir: Path, labels_dir: Path, chain_id: str, *, length: int = 4, msa_depth: int = 2) -> None:
    feature_path = chain_npz_path(features_dir, chain_id)
    label_path = chain_npz_path(labels_dir, chain_id)
    np.savez_compressed(
        feature_path,
        chain_id=np.asarray(chain_id),
        aatype=np.arange(length, dtype=np.int32),
        msa=np.arange(msa_depth * length, dtype=np.int32).reshape(msa_depth, length),
        deletions=np.zeros((msa_depth, length), dtype=np.int32),
        residue_index=np.arange(length, dtype=np.int32),
        between_segment_residues=np.zeros((length,), dtype=np.int32),
        projection_seq_identity=np.asarray(1.0, dtype=np.float32),
        projection_alignment_coverage=np.asarray(1.0, dtype=np.float32),
        projection_aligned_fraction=np.asarray(1.0, dtype=np.float32),
        projection_valid_ca_count=np.asarray(length, dtype=np.int32),
        template_aatype=np.zeros((0, length), dtype=np.int32),
        template_ca_coords=np.zeros((0, length, 3), dtype=np.float32),
        template_ca_mask=np.zeros((0, length), dtype=bool),
    )
    np.savez_compressed(
        label_path,
        chain_id=np.asarray(chain_id),
        ca_coords=np.zeros((length, 3), dtype=np.float32),
        ca_mask=np.ones((length,), dtype=bool),
        atom14_positions=np.zeros((length, 14, 3), dtype=np.float32),
        atom14_mask=np.ones((length, 14), dtype=bool),
        residue_index=np.arange(length, dtype=np.int32),
        resolution=np.asarray(1.5, dtype=np.float32),
    )


def test_generate_rows_unrolls_npz_fields(tmp_path: Path) -> None:
    features_dir = tmp_path / "features"
    labels_dir = tmp_path / "labels"
    features_dir.mkdir()
    labels_dir.mkdir()
    manifest = tmp_path / "train.txt"
    manifest.write_text("1abc_A\n")
    _write_npz_pair(features_dir, labels_dir, "1abc_A")

    row = next(
        generate_rows(
            manifest_path=str(manifest),
            split="train",
            processed_features_dir=str(features_dir),
            processed_labels_dir=str(labels_dir),
        )
    )

    assert row["chain_id"] == "1abc_A"
    assert row["pdb_id"] == "1abc"
    assert row["pdb_chain_id"] == "A"
    assert row["split"] == "train"
    assert row["length"] == 4
    assert row["msa_depth"] == 2
    assert row["template_count"] == 0
    assert row["msa"].shape == (2, 4)
    assert row["atom14_positions"].shape == (4, 14, 3)
    assert len(row["feature_sha256"]) == 64
    assert len(row["label_sha256"]) == 64


def test_render_dataset_card_documents_columns_and_sampling() -> None:
    card = render_dataset_card(
        {
            "train_count": 10_000,
            "validation_count": 1_000,
            "total_count": 11_000,
            "train_manifest_sha256": "a" * 64,
            "validation_manifest_sha256": "b" * 64,
            "dataset_fingerprint": {
                "feature_files_sha256": "c" * 64,
                "label_files_sha256": "d" * 64,
            },
        }
    )

    assert "OpenProteinSet" in card
    assert "structural stratification" in card
    assert "`msa`" in card
    assert "`atom14_positions`" in card
    assert "smaller protein-folding models" in card
