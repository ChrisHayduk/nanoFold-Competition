from __future__ import annotations

import csv
from xml.etree import ElementTree as ET

import numpy as np

from neurips_paper.scripts import make_openproteinset_coverage as coverage


def test_build_coverage_records_deduplicates_and_prefers_train_representative() -> None:
    chain_cache = {
        "1aaa_A": {"seq": "ACDEFGHIKL", "resolution": 1.5},
        "1bbb_A": {"seq": "ACDEFGHIKL", "resolution": 2.0},
        "2ccc_A": {"seq": "MMMMMMMMMM", "resolution": 999.0},
        "3ddd_A": {"seq": "MMMMMMMMMX", "resolution": 1.0},
        "4eee_A": {"seq": "GGGG", "resolution": 1.0},
        "5fff_A": {"seq": "TTTTTTTTTT", "resolution": 4.0},
    }

    records = coverage.build_coverage_records(
        chain_cache,
        train_ids={"1bbb_A"},
        val_ids={"2ccc_A"},
        min_len=5,
        max_len=12,
        max_resolution=3.0,
    )

    assert [(record.representative_chain_id, record.split) for record in records] == [
        ("1bbb_A", "train"),
        ("2ccc_A", "public_val"),
    ]
    train_record = records[0]
    assert train_record.chain_count == 2
    assert train_record.train_chain_count == 1


def test_sample_records_keeps_all_train_and_caps_background_deterministically() -> None:
    records = [
        coverage.CoverageRecord(
            record_id=f"background_{index}",
            representative_chain_id=f"{index:04d}_A",
            sequence="ACDEFGHIKL",
            length=10,
            split="background",
            chain_count=1,
            train_chain_count=0,
            val_chain_count=0,
            sequence_sha1=str(index),
        )
        for index in range(8)
    ]
    records.append(
        coverage.CoverageRecord(
            record_id="train_0",
            representative_chain_id="9zzz_A",
            sequence="MMMMMMMMMM",
            length=10,
            split="train",
            chain_count=1,
            train_chain_count=1,
            val_chain_count=0,
            sequence_sha1="train",
        )
    )

    sampled_a = coverage.sample_records(records, max_background_records=3, seed=7)
    sampled_b = coverage.sample_records(records, max_background_records=3, seed=7)

    assert sampled_a == sampled_b
    assert sum(record.split == "background" for record in sampled_a) == 3
    assert any(record.record_id == "train_0" for record in sampled_a)


def test_write_fasta_and_metadata(tmp_path) -> None:
    record = coverage.CoverageRecord(
        record_id="1abc_A__seq_test",
        representative_chain_id="1abc_A",
        sequence="ACDEFGHIKL",
        length=10,
        split="train",
        chain_count=2,
        train_chain_count=1,
        val_chain_count=0,
        sequence_sha1="abc123",
    )

    fasta_path = tmp_path / "seqs.fasta"
    metadata_path = tmp_path / "metadata.csv"
    coverage.write_fasta([record], fasta_path)
    coverage.write_metadata([record], metadata_path)

    assert fasta_path.read_text().splitlines() == [
        ">1abc_A__seq_test|chain=1abc_A|split=train|len=10",
        "ACDEFGHIKL",
    ]
    rows = list(csv.DictReader(metadata_path.open()))
    assert rows[0]["record_id"] == "1abc_A__seq_test"
    assert rows[0]["split"] == "train"
    assert rows[0]["chain_count"] == "2"


def test_write_merged_embeddings_reuses_by_sequence_sha(tmp_path) -> None:
    old_record = coverage.CoverageRecord(
        record_id="old_chain__seq_test",
        representative_chain_id="old_chain",
        sequence="ACDEFGHIKL",
        length=10,
        split="background",
        chain_count=1,
        train_chain_count=0,
        val_chain_count=0,
        sequence_sha1=coverage.sequence_hash("ACDEFGHIKL"),
    )
    new_record = coverage.CoverageRecord(
        record_id="new_chain__seq_test",
        representative_chain_id="new_chain",
        sequence=old_record.sequence,
        length=10,
        split="background",
        chain_count=2,
        train_chain_count=0,
        val_chain_count=0,
        sequence_sha1=old_record.sequence_sha1,
    )
    metadata_path = tmp_path / "old_metadata.csv"
    old_embeddings_path = tmp_path / "old_embeddings.npz"
    merged_path = tmp_path / "merged_embeddings.npz"

    coverage.write_metadata([old_record], metadata_path)
    np.savez(
        old_embeddings_path,
        ids=np.asarray([old_record.record_id]),
        embeddings=np.asarray([[1.0, 2.0, 3.0]], dtype=np.float32),
    )

    reusable = coverage.load_reusable_embeddings_by_sha(metadata_path, old_embeddings_path)
    coverage.write_merged_embeddings(
        [new_record],
        out_path=merged_path,
        embedding_dtype="float32",
        reusable_by_sha=reusable,
        precomputed_by_id={},
    )

    merged = np.load(merged_path, allow_pickle=False)
    assert merged["ids"].tolist() == [new_record.record_id]
    np.testing.assert_allclose(merged["embeddings"], np.asarray([[1.0, 2.0, 3.0]], dtype=np.float32))


def test_pca_reduce_returns_deterministic_two_dimensional_projection() -> None:
    embeddings = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )

    first = coverage._pca_reduce(embeddings, 2)
    second = coverage._pca_reduce(embeddings, 2)

    assert first.shape == (4, 2)
    np.testing.assert_allclose(first, second)


def test_write_svg_uses_density_layer_and_highlight_points(tmp_path) -> None:
    records = [
        coverage.CoverageRecord(
            record_id=f"background_{index}",
            representative_chain_id=f"{index:04d}_A",
            sequence="ACDEFGHIKL",
            length=10,
            split="background",
            chain_count=1,
            train_chain_count=0,
            val_chain_count=0,
            sequence_sha1=f"background_{index}",
        )
        for index in range(6)
    ]
    records.extend(
        [
            coverage.CoverageRecord(
                record_id="train_0",
                representative_chain_id="1abc_A",
                sequence="MMMMMMMMMM",
                length=10,
                split="train",
                chain_count=1,
                train_chain_count=1,
                val_chain_count=0,
                sequence_sha1="train",
            ),
            coverage.CoverageRecord(
                record_id="val_0",
                representative_chain_id="2abc_A",
                sequence="GGGGGGGGGG",
                length=10,
                split="public_val",
                chain_count=1,
                train_chain_count=0,
                val_chain_count=1,
                sequence_sha1="val",
            ),
        ]
    )
    coords = np.asarray(
        [
            [0.0, 0.0],
            [0.1, 0.0],
            [0.0, 0.1],
            [0.2, 0.2],
            [0.3, 0.2],
            [0.2, 0.3],
            [0.1, 0.1],
            [0.25, 0.25],
        ],
        dtype=np.float32,
    )
    svg_path = tmp_path / "coverage.svg"

    coverage.write_svg(records, coords, svg_path, title="Coverage", reducer_name="umap")

    ET.parse(svg_path)
    svg = svg_path.read_text()
    assert "OpenProteinSet density" in svg
    assert "NanoFold train" in svg
    assert "Warm regions mark" in svg
