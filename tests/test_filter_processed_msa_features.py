from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

from nanofold.a3m import sequence_to_ids
from nanofold.chain_paths import chain_npz_path


def _load_filter_processed_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "filter_processed_msa_features.py"
    spec = importlib.util.spec_from_file_location("filter_processed_msa_features_for_tests", script_path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _sequence_sha256(sequence: str) -> str:
    return hashlib.sha256(sequence.upper().encode("utf-8")).hexdigest()


def _write_filter(path: Path, sequences: list[str]) -> None:
    payload = {
        "schema_version": 1,
        "excluded_sequence_sha256": sorted(_sequence_sha256(sequence) for sequence in sequences),
    }
    path.write_text(json.dumps(payload, sort_keys=True) + "\n")


def _write_feature(features_dir: Path, chain_id: str, rows: list[str]) -> None:
    msa = np.stack([sequence_to_ids(row) for row in rows]).astype(np.int32)
    deletions = np.stack(
        [np.full((msa.shape[1],), row_index, dtype=np.int32) for row_index in range(len(rows))]
    )
    feature_path = chain_npz_path(features_dir, chain_id)
    feature_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        feature_path,
        chain_id=np.asarray(chain_id),
        aatype=sequence_to_ids(rows[0]).astype(np.int32),
        msa=msa,
        deletions=deletions,
        residue_index=np.arange(msa.shape[1], dtype=np.int32),
        between_segment_residues=np.zeros((msa.shape[1],), dtype=np.int32),
        template_aatype=np.zeros((0, msa.shape[1]), dtype=np.int32),
        template_ca_coords=np.zeros((0, msa.shape[1], 3), dtype=np.float32),
        template_ca_mask=np.zeros((0, msa.shape[1]), dtype=bool),
    )


def test_filter_processed_features_removes_non_query_rows_and_preserves_query(tmp_path: Path) -> None:
    module = _load_filter_processed_module()
    manifest = tmp_path / "train.txt"
    manifest.write_text("toy_A\n")
    source_features = tmp_path / "features"
    out_features = tmp_path / "filtered"
    query = "AAAAAAAAAAAAAAAAAAAA"
    forbidden = "CCCCCCCCCCCCCCCCCCCC"
    allowed = "DDDDDDDDDDDDDDDDDDDD"
    _write_feature(source_features, "toy_A", [query, forbidden, allowed])
    filter_path = tmp_path / "filter.json"
    _write_filter(filter_path, [query, forbidden])
    row_filter = module._load_msa_row_filter(filter_path)

    audit = module.filter_processed_features(
        manifest_paths=[manifest],
        source_features_dirs=[source_features],
        out_features_dir=out_features,
        row_filter=row_filter,
    )

    output_path = chain_npz_path(out_features, "toy_A")
    with np.load(output_path) as data:
        assert data["msa"].shape == (2, 20)
        assert np.array_equal(data["msa"][0], sequence_to_ids(query))
        assert np.array_equal(data["msa"][1], sequence_to_ids(allowed))
        assert np.array_equal(data["deletions"][:, 0], np.asarray([0, 2], dtype=np.int32))
        assert str(data["msa_row_filter_sha256"].item()) == row_filter.sha256
        assert int(data["msa_rows_before_filter"].item()) == 3
        assert int(data["msa_rows_removed_by_filter"].item()) == 1

    assert audit["total_msa_rows_before_filter"] == 3
    assert audit["total_msa_rows_removed"] == 1
    assert audit["total_msa_rows_after_filter"] == 2
    assert audit["per_chain"][0]["status"] == "written"
    meta = json.loads((out_features / "preprocess_meta.json").read_text())
    assert meta["cli_args"]["msa_row_filter_sha256"] == row_filter.sha256


def test_filter_processed_features_searches_multiple_source_roots_and_skips_existing(tmp_path: Path) -> None:
    module = _load_filter_processed_module()
    manifest = tmp_path / "train.txt"
    manifest.write_text("first_A\nsecond_A\n")
    first_source = tmp_path / "features_a"
    second_source = tmp_path / "features_b"
    out_features = tmp_path / "filtered"
    forbidden = "CCCCCCCCCCCCCCCCCCCC"
    _write_feature(first_source, "first_A", ["AAAAAAAAAAAAAAAAAAAA", forbidden])
    _write_feature(second_source, "second_A", ["DDDDDDDDDDDDDDDDDDDD", "EEEEEEEEEEEEEEEEEEEE"])
    filter_path = tmp_path / "filter.json"
    _write_filter(filter_path, [forbidden])
    row_filter = module._load_msa_row_filter(filter_path)

    module.filter_processed_features(
        manifest_paths=[manifest],
        source_features_dirs=[first_source, second_source],
        out_features_dir=out_features,
        row_filter=row_filter,
    )
    audit = module.filter_processed_features(
        manifest_paths=[manifest],
        source_features_dirs=[first_source, second_source],
        out_features_dir=out_features,
        row_filter=row_filter,
        skip_existing=True,
    )

    assert audit["filtered_chain_count"] == 2
    assert [row["status"] for row in audit["per_chain"]] == ["skipped_existing", "skipped_existing"]
    with np.load(chain_npz_path(out_features, "first_A")) as data:
        assert data["msa"].shape == (1, 20)
    with np.load(chain_npz_path(out_features, "second_A")) as data:
        assert data["msa"].shape == (2, 20)


def test_skip_existing_rebuilds_changed_source_and_modified_output(tmp_path: Path) -> None:
    module = _load_filter_processed_module()
    manifest = tmp_path / "train.txt"
    manifest.write_text("toy_A\n")
    source, output = tmp_path / "source", tmp_path / "output"
    _write_feature(source, "toy_A", ["A" * 20, "C" * 20])
    filter_path = tmp_path / "filter.json"
    _write_filter(filter_path, ["C" * 20])
    kwargs = dict(manifest_paths=[manifest], source_features_dirs=[source], out_features_dir=output,
                  row_filter=module._load_msa_row_filter(filter_path), skip_existing=True)
    module.filter_processed_features(**kwargs)
    _write_feature(source, "toy_A", ["A" * 20, "C" * 20, "D" * 20])
    rebuilt = module.filter_processed_features(**kwargs)
    assert rebuilt["per_chain"][0]["status"] == "written"
    output_path = chain_npz_path(output, "toy_A")
    with np.load(output_path) as data:
        assert data["msa"].shape[0] == 2
        arrays = dict(data)
    arrays["residue_index"][0] = 99
    np.savez(output_path, **arrays)
    repaired = module.filter_processed_features(**kwargs)
    assert repaired["per_chain"][0]["status"] == "written"
    with np.load(output_path) as data:
        assert data["residue_index"][0] == 0


def test_filter_rejects_in_place_and_false_provenance(tmp_path: Path) -> None:
    import pytest

    module = _load_filter_processed_module()
    manifest = tmp_path / "train.txt"
    manifest.write_text("toy_A\n")
    source = tmp_path / "source"
    _write_feature(source, "toy_A", ["A" * 20, "C" * 20])
    filter_path = tmp_path / "filter.json"
    _write_filter(filter_path, ["C" * 20])
    row_filter = module._load_msa_row_filter(filter_path)
    path = chain_npz_path(source, "toy_A")
    original = path.read_bytes()
    with pytest.raises(ValueError, match="separate output"):
        module.filter_processed_features(manifest_paths=[manifest], source_features_dirs=[source],
                                         out_features_dir=source, row_filter=row_filter)
    assert path.read_bytes() == original
    with np.load(path) as data:
        arrays = dict(data)
    arrays["preprocess_config_sha256"] = np.asarray("a" * 64)
    np.savez(path, **arrays)
    with pytest.raises(ValueError, match="re-running preprocess"):
        module.filter_processed_features(manifest_paths=[manifest], source_features_dirs=[source],
                                         out_features_dir=tmp_path / "out", row_filter=row_filter)
    assert not chain_npz_path(tmp_path / "out", "toy_A").exists()


def test_processed_filter_refuses_to_choose_between_conflicting_variants(tmp_path: Path) -> None:
    import pytest

    module = _load_filter_processed_module()
    manifest = tmp_path / "train.txt"
    manifest.write_text("toy_A\n")
    first, second, output = tmp_path / "first", tmp_path / "second", tmp_path / "out"
    _write_feature(first, "toy_A", ["A" * 20, "C" * 20])
    _write_feature(second, "toy_A", ["A" * 20, "D" * 20])
    filter_path = tmp_path / "filter.json"
    _write_filter(filter_path, ["C" * 20])
    with pytest.raises(ValueError, match="different source feature variants"):
        module.filter_processed_features(manifest_paths=[manifest], source_features_dirs=[first, second],
                                         out_features_dir=output, row_filter=module._load_msa_row_filter(filter_path))
    assert not chain_npz_path(output, "toy_A").exists()
