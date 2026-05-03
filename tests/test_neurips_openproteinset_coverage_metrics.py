from __future__ import annotations

import csv

import numpy as np

from neurips_paper.scripts import compute_openproteinset_coverage_metrics as metrics


def _metadata() -> list[metrics.EmbeddingMetadata]:
    return [
        metrics.EmbeddingMetadata("train_0", "1abc_A", "train", 80, 1, metrics.sequence_hash("A" * 80)),
        metrics.EmbeddingMetadata("background_0", "2abc_A", "background", 90, 1, metrics.sequence_hash("C" * 90)),
        metrics.EmbeddingMetadata("val_0", "3abc_A", "public_val", 100, 1, metrics.sequence_hash("D" * 100)),
    ]


def test_load_aligned_embeddings_orders_by_metadata(tmp_path) -> None:
    path = tmp_path / "embeddings.npz"
    np.savez(
        path,
        ids=np.asarray(["val_0", "train_0", "background_0"]),
        embeddings=np.asarray([[3.0, 0.0], [1.0, 0.0], [2.0, 0.0]], dtype=np.float16),
    )

    aligned = metrics.load_aligned_embeddings(path, _metadata())

    np.testing.assert_allclose(aligned, np.asarray([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]], dtype=np.float32))


def test_l2_normalize_handles_zero_rows() -> None:
    normalized = metrics.l2_normalize(np.asarray([[3.0, 4.0], [0.0, 0.0]], dtype=np.float32))

    np.testing.assert_allclose(normalized[0], np.asarray([0.6, 0.8], dtype=np.float32))
    np.testing.assert_allclose(normalized[1], np.asarray([0.0, 0.0], dtype=np.float32))


def test_coverage_at_radii_rows_reports_each_split() -> None:
    nearest_train = np.asarray([0.0, 0.2, 0.4, 0.8], dtype=np.float32)
    splits = np.asarray(["train", "background", "public_val", "background"])

    rows = metrics.coverage_at_radii_rows(nearest_train, splits, [("r", 0.5)])

    assert rows == [
        {
            "radius_name": "r",
            "radius": 0.5,
            "all_n": 4,
            "all_covered_fraction": 0.75,
            "non_train_n": 3,
            "non_train_covered_fraction": 2 / 3,
            "background_n": 2,
            "background_covered_fraction": 0.5,
            "public_val_n": 1,
            "public_val_covered_fraction": 1.0,
        }
    ]


def test_hidden_manifest_overlay_uses_sequence_hashes_without_ids(tmp_path) -> None:
    hidden_manifest = tmp_path / "hidden.txt"
    chain_cache = tmp_path / "chain_data_cache.json"
    hidden_manifest.write_text("hidden_chain\nmissing_chain\n")
    chain_cache.write_text('{"hidden_chain": {"seq": "CCCC"}}')
    metadata = [
        metrics.EmbeddingMetadata("train_0", "1abc_A", "train", 4, 1, metrics.sequence_hash("AAAA")),
        metrics.EmbeddingMetadata("background_0", "2abc_A", "background", 4, 1, metrics.sequence_hash("CCCC")),
        metrics.EmbeddingMetadata("background_1", "3abc_A", "background", 4, 1, metrics.sequence_hash("DDDD")),
    ]

    hidden_hashes, stats = metrics.load_hidden_sequence_hashes(
        hidden_manifest=hidden_manifest,
        chain_data_cache=chain_cache,
    )
    hidden_mask = metrics.hidden_validation_mask(metadata, hidden_hashes)
    nearest_rows = metrics.nearest_train_quantile_rows(
        np.asarray([0.0, 0.2, 0.4], dtype=np.float32), metrics.split_array(metadata), hidden_mask
    )

    assert stats["manifest_records"] == 2
    assert stats["missing_cache_records"] == 1
    assert hidden_mask.tolist() == [False, True, False]
    assert {row["set"]: row["n"] for row in nearest_rows}["hidden_val"] == 1
    assert {row["set"]: row["n"] for row in nearest_rows}["non_train_source_pool"] == 1


def test_density_decile_rows_includes_train_fraction_and_normalized_gap() -> None:
    nearest_train = np.linspace(0.0, 0.9, 10, dtype=np.float32)
    local_d50 = np.linspace(0.1, 1.0, 10, dtype=np.float32)
    splits = np.asarray(["train", "background", "public_val", "background", "background"] * 2)

    rows = metrics.density_decile_rows(nearest_train, local_d50, splits)

    assert len(rows) == 10
    assert rows[0]["density_label"] == "densest"
    assert rows[-1]["density_label"] == "sparsest"
    assert rows[0]["train_fraction"] == 1.0
    assert rows[1]["non_train_within_local_50_fraction"] == 1.0


def test_cluster_rows_from_labels_counts_splits() -> None:
    labels = np.asarray([0, 0, 1, 1, 1])
    splits = np.asarray(["train", "background", "background", "public_val", "train"])

    rows = metrics.cluster_rows_from_labels(labels, splits, k=2)

    assert rows[0]["n"] == 3
    assert rows[0]["train_n"] == 1
    assert rows[0]["public_val_n"] == 1
    assert rows[1]["train_fraction"] == 0.5


def test_write_csv_roundtrip(tmp_path) -> None:
    path = tmp_path / "rows.csv"

    metrics.write_csv(path, [{"name": "a", "value": 1.25}])

    rows = list(csv.DictReader(path.open()))
    assert rows == [{"name": "a", "value": "1.25"}]
