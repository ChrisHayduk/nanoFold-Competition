from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from nanofold.chain_paths import chain_npz_path
from nanofold.dataset_integrity import (
    PREPROCESS_META_FILENAME,
    build_dataset_fingerprint,
    build_split_fingerprint,
    preprocess_config_sha256,
    validate_label_npz_schema,
    validate_npz_schema,
    verify_dataset_against_fingerprint,
    verify_split_against_fingerprint,
)
from nanofold.utils import sha256_file


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


def test_preprocess_meta_hash_ignores_local_paths_and_dependency_versions(tmp_path: Path) -> None:
    manifests = tmp_path / "manifests"
    manifests.mkdir()
    manifest = manifests / "hidden_val.txt"
    manifest.write_text("1abc_A\n")

    features_a = tmp_path / "features_a"
    features_b = tmp_path / "features_b"
    labels_a = tmp_path / "labels_a"
    labels_b = tmp_path / "labels_b"
    for path in (features_a, features_b, labels_a, labels_b):
        path.mkdir()

    for features, labels in ((features_a, labels_a), (features_b, labels_b)):
        _write_feature_npz(chain_npz_path(features, "1abc_A"))
        _write_label_npz(chain_npz_path(labels, "1abc_A"))

    stable_meta: dict[str, Any] = {
        "aligner": {"match_score": 2.0, "mode": "global"},
        "atom14_num_slots": 14,
        "ca_atom14_slot": 1,
        "cli_args": {
            "disable_templates": True,
            "max_msa_seqs": 2048,
            "max_templates": 1,
            "msa_name": "uniref90_hits.a3m",
            "processed_features_dir": "local/path/one",
            "processed_labels_dir": "local/path/one",
            "raw_root": "data/openproteinset",
            "strict": False,
        },
        "dependency_metadata": {"numpy": "1.0", "python": "3.11"},
        "git_sha": "abc",
        "schema_version": 2,
    }
    changed_environment_meta = {
        **stable_meta,
        "cli_args": {
            **stable_meta["cli_args"],
            "processed_features_dir": "different/local/path",
            "processed_labels_dir": "different/local/path",
            "raw_root": "/tmp/private/data",
        },
        "dependency_metadata": {"numpy": "2.0", "python": "3.13"},
        "git_sha": "def",
    }
    (features_a / PREPROCESS_META_FILENAME).write_text(json.dumps(stable_meta))
    (features_b / PREPROCESS_META_FILENAME).write_text(json.dumps(changed_environment_meta))

    fp_a = build_split_fingerprint(
        processed_features_dir=features_a,
        processed_labels_dir=labels_a,
        manifest_paths={"hidden_val": manifest},
        require_no_missing=True,
        require_labels=True,
    )
    fp_b = build_split_fingerprint(
        processed_features_dir=features_b,
        processed_labels_dir=labels_b,
        manifest_paths={"hidden_val": manifest},
        require_no_missing=True,
        require_labels=True,
    )
    assert fp_a["preprocess_config_sha256"] == fp_b["preprocess_config_sha256"]


def test_preprocess_meta_hash_captures_msa_row_filter_sha256(tmp_path: Path) -> None:
    manifests = tmp_path / "manifests"
    manifests.mkdir()
    manifest = manifests / "train.txt"
    manifest.write_text("1abc_A\n")

    features_a = tmp_path / "features_a"
    features_b = tmp_path / "features_b"
    labels_a = tmp_path / "labels_a"
    labels_b = tmp_path / "labels_b"
    for path in (features_a, features_b, labels_a, labels_b):
        path.mkdir()

    for features, labels in ((features_a, labels_a), (features_b, labels_b)):
        _write_feature_npz(chain_npz_path(features, "1abc_A"))
        _write_label_npz(chain_npz_path(labels, "1abc_A"))

    base_meta: dict[str, Any] = {
        "aligner": {"match_score": 2.0, "mode": "global"},
        "atom14_num_slots": 14,
        "ca_atom14_slot": 1,
        "cli_args": {
            "disable_templates": True,
            "max_msa_seqs": 2048,
            "msa_name": "uniref90_hits.a3m",
            "msa_row_filter_sha256": "a" * 64,
            "strict": False,
        },
        "schema_version": 2,
    }
    changed_meta = {
        **base_meta,
        "cli_args": {
            **base_meta["cli_args"],
            "msa_row_filter_sha256": "b" * 64,
        },
    }
    (features_a / PREPROCESS_META_FILENAME).write_text(json.dumps(base_meta))
    (features_b / PREPROCESS_META_FILENAME).write_text(json.dumps(changed_meta))

    fp_a = build_split_fingerprint(
        processed_features_dir=features_a,
        processed_labels_dir=labels_a,
        manifest_paths={"train": manifest},
        require_no_missing=True,
        require_labels=True,
    )
    fp_b = build_split_fingerprint(
        processed_features_dir=features_b,
        processed_labels_dir=labels_b,
        manifest_paths={"train": manifest},
        require_no_missing=True,
        require_labels=True,
    )
    assert fp_a["preprocess_config_sha256"] != fp_b["preprocess_config_sha256"]


def _replace_npz_array(path: Path, key: str, value: np.ndarray) -> None:
    with np.load(path) as data:
        arrays = dict(data)
    arrays[key] = value
    np.savez_compressed(path, **arrays)


@pytest.mark.parametrize(
    "key,value,error",
    [
        ("aatype", np.full(6, 21, dtype=np.int32), "tokens"),
        ("msa", np.full((4, 6), 22, dtype=np.int32), "tokens"),
        ("deletions", np.full((4, 6), -1, dtype=np.int32), "nonnegative"),
        ("template_aatype", np.full((1, 6), 21, dtype=np.int32), "tokens"),
        ("template_ca_coords", np.full((1, 6, 3), np.nan, dtype=np.float32), "finite"),
        ("msa", np.zeros((0, 6), dtype=np.int32), "query row"),
        ("msa", np.ones((4, 6), dtype=np.int32), "query row"),
        ("template_aatype", np.asarray(0, dtype=np.int32), "shape"),
    ],
)
def test_feature_validation_rejects_invalid_model_inputs(
    tmp_path: Path, key: str, value: np.ndarray, error: str,
) -> None:
    path = tmp_path / "features.npz"
    _write_feature_npz(path)
    _replace_npz_array(path, key, value)
    assert any(error in message for message in validate_npz_schema(path))


@pytest.mark.parametrize(
    "key,value,error",
    [
        ("ca_coords", np.full((6, 3), np.inf, dtype=np.float32), "finite"),
        ("atom14_positions", np.full((6, 14, 3), np.nan, dtype=np.float32), "finite"),
        ("resolution", np.asarray(np.nan, dtype=np.float32), "finite"),
        ("resolution", np.asarray(-1, dtype=np.float32), "nonnegative"),
        ("ca_coords", np.ones((6, 3), dtype=np.float32), "CA slot"),
        ("ca_mask", np.zeros(6, dtype=bool), "CA slot"),
    ],
)
def test_label_validation_rejects_nonfinite_or_inconsistent_supervision(
    tmp_path: Path, key: str, value: np.ndarray, error: str,
) -> None:
    path = tmp_path / "labels.npz"
    _write_label_npz(path)
    _replace_npz_array(path, key, value)
    assert any(error in message for message in validate_label_npz_schema(path))


def test_fingerprint_rejects_feature_label_identity_mismatch(tmp_path: Path) -> None:
    features = tmp_path / "features"
    labels = tmp_path / "labels"
    features.mkdir()
    labels.mkdir()
    manifest = tmp_path / "train.txt"
    manifest.write_text("1abc_A\n")
    feature_path = chain_npz_path(features, "1abc_A")
    label_path = chain_npz_path(labels, "1abc_A")
    _write_feature_npz(feature_path)
    _write_label_npz(label_path)
    _replace_npz_array(feature_path, "chain_id", np.asarray("1abc_A"))
    _replace_npz_array(label_path, "chain_id", np.asarray("2xyz_A"))
    with pytest.raises(ValueError, match="chain_id.*must match"):
        build_split_fingerprint(
            processed_features_dir=features,
            processed_labels_dir=labels,
            manifest_paths={"train": manifest},
            require_no_missing=True,
            require_labels=True,
        )


@pytest.mark.parametrize("schema_version", [1, 2])
def test_frozen_schema_optional_chain_ids_are_checked_when_present(tmp_path: Path, schema_version: int) -> None:
    features = tmp_path / "features"
    features.mkdir()
    manifest = tmp_path / "train.txt"
    manifest.write_text("1abc_A\n")
    path = chain_npz_path(features, "1abc_A")
    _write_feature_npz(path)
    (features / PREPROCESS_META_FILENAME).write_text(json.dumps({"schema_version": schema_version}))
    args: dict[str, Any] = {
        "processed_features_dir": features, "manifest_paths": {"train": manifest},
        "require_no_missing": True, "require_labels": False,
    }
    build_split_fingerprint(**args)
    _replace_npz_array(path, "chain_id", np.asarray("1abc_B"))
    with pytest.raises(ValueError, match="chain_id.*requested chain 1abc_A"):
        build_split_fingerprint(**args)


@pytest.mark.parametrize("label_key", ["ca_coords", "ca_mask", "atom14_positions", "atom14_mask", "resolution"])
def test_fingerprint_rejects_supervision_in_features_only_archive(tmp_path: Path, label_key: str) -> None:
    features = tmp_path / "features"
    features.mkdir()
    manifest = tmp_path / "hidden.txt"
    manifest.write_text("1abc_A\n")
    feature_path = chain_npz_path(features, "1abc_A")
    _write_feature_npz(feature_path)
    label_path = tmp_path / "labels.npz"
    _write_label_npz(label_path)
    with np.load(label_path) as data:
        label_value = data[label_key] if label_key in data.files else np.asarray(2.0, dtype=np.float32)
    _replace_npz_array(feature_path, label_key, label_value)

    with pytest.raises(ValueError, match=f"supervision key `{label_key}`"):
        build_split_fingerprint(
            processed_features_dir=features,
            manifest_paths={"hidden": manifest},
            require_no_missing=True,
            require_labels=False,
        )


@pytest.mark.parametrize("require_no_missing", [False, True])
@pytest.mark.parametrize("validate_schema", [False, True])
def test_fingerprint_cannot_claim_label_coverage_without_a_label_directory(
    tmp_path: Path, require_no_missing: bool, validate_schema: bool,
) -> None:
    features = tmp_path / "features"
    features.mkdir()
    manifest = tmp_path / "train.txt"
    manifest.write_text("1abc_A\n")
    _write_feature_npz(chain_npz_path(features, "1abc_A"))

    with pytest.raises(ValueError, match="requires `processed_labels_dir`"):
        build_split_fingerprint(
            processed_features_dir=features,
            manifest_paths={"train": manifest},
            require_no_missing=require_no_missing,
            require_labels=True,
            validate_schema=validate_schema,
        )


@pytest.mark.parametrize("split_names", [("train", "val"), ("public", "hidden")])
def test_fingerprint_rejects_chain_overlap_between_splits(
    tmp_path: Path, split_names: tuple[str, str],
) -> None:
    train = tmp_path / "train.txt"
    heldout = tmp_path / "heldout.txt"
    train.write_text("1abc_A\n2def_B\n")
    heldout.write_text("2def_B\n3ghi_C\n")
    with pytest.raises(ValueError, match="chain-disjoint.*share 1 chain IDs") as caught:
        build_split_fingerprint(
            processed_features_dir=tmp_path / "features",
            manifest_paths={split_names[0]: train, split_names[1]: heldout},
            require_no_missing=False,
            require_labels=False,
        )
    assert "2def_B" not in str(caught.value)


def test_fingerprint_rejects_duplicate_manifest_records(tmp_path: Path) -> None:
    manifest = tmp_path / "train.txt"
    manifest.write_text("1abc_A\n1abc_A\n")
    with pytest.raises(ValueError, match="[Dd]uplicate"):
        build_split_fingerprint(
            processed_features_dir=tmp_path / "features",
            manifest_paths={"train": manifest},
            require_no_missing=False,
            require_labels=False,
        )


def test_fingerprint_rejects_colliding_normalized_split_names(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Duplicate normalized manifest split name"):
        build_split_fingerprint(
            processed_features_dir=tmp_path / "features",
            manifest_paths={"train": tmp_path / "first.txt", " train ": tmp_path / "second.txt"},
            require_no_missing=False,
            require_labels=False,
        )


@pytest.fixture
def directed_dataset(tmp_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    features, labels = tmp_path / "features", tmp_path / "labels"
    features.mkdir()
    labels.mkdir()
    manifests = {role: tmp_path / f"{role}.txt" for role in ("train", "val")}
    chains = {"train": ["1abc_A", "2def_B"], "val": ["3ghi_C"]}
    filters = {"train": "a" * 64, "val": "b" * 64}
    for role, path in manifests.items():
        path.write_text("".join(f"{chain}\n" for chain in chains[role]))
    meta = {
        "schema_version": 3,
        "cli_args": {"max_msa_seqs": 2048, "msa_row_filter_sha256": None},
        "msa_filter_policy": {
            "schema_version": 1, "policy_type": "directed-heldout-v1", "policy_sha256": "c" * 64,
            "splits": {role: {"manifest_sha256": sha256_file(path), "chain_count": len(chains[role]),
                              "filter_sha256": filters[role]} for role, path in manifests.items()},
        },
    }
    config_hash = preprocess_config_sha256(meta)
    (features / PREPROCESS_META_FILENAME).write_text(json.dumps(meta))
    for role, chain_ids in chains.items():
        for chain_id in chain_ids:
            feature_path, label_path = chain_npz_path(features, chain_id), chain_npz_path(labels, chain_id)
            _write_feature_npz(feature_path)
            _write_label_npz(label_path)
            for path in (feature_path, label_path):
                with np.load(path) as data:
                    arrays = dict(data)
                arrays.update(chain_id=np.asarray(chain_id), preprocess_config_sha256=np.asarray(config_hash),
                              source_inputs_sha256=np.asarray("d" * 64))
                if path == feature_path:
                    arrays["msa_row_filter_sha256"] = np.asarray(filters[role])
                np.savez_compressed(path, **arrays)
    return {
        "processed_features_dir": features, "processed_labels_dir": labels, "manifest_paths": manifests,
        "require_no_missing": True, "require_labels": True,
    }, meta


def test_directed_policy_fingerprints_accept_exact_public_roles_and_single_role_subsets(
    directed_dataset: tuple[dict[str, Any], dict[str, Any]],
) -> None:
    args, meta = directed_dataset
    full = build_split_fingerprint(**args)
    assert full["present_feature_chain_count"] == 3
    assert full["present_label_chain_count"] == 3
    assert full["preprocess_config_sha256"] == preprocess_config_sha256(meta)
    for role in ("train", "val"):
        subset = build_split_fingerprint(**{**args, "manifest_paths": {role: args["manifest_paths"][role]}})
        assert subset["unique_chain_count"] == meta["msa_filter_policy"]["splits"][role]["chain_count"]
        assert subset["preprocess_config_sha256"] == full["preprocess_config_sha256"]


@pytest.mark.parametrize("chain_id", ["1abc_A", "2def_B", "3ghi_C"])
@pytest.mark.parametrize("validate_schema", [True, False])
def test_directed_policy_checks_actual_filter_on_every_feature(
    directed_dataset: tuple[dict[str, Any], dict[str, Any]], chain_id: str, validate_schema: bool,
) -> None:
    args, _ = directed_dataset
    path = chain_npz_path(args["processed_features_dir"], chain_id)
    wrong_filter = "a" * 64 if chain_id == "3ghi_C" else "b" * 64
    _replace_npz_array(path, "msa_row_filter_sha256", np.asarray(wrong_filter))
    with pytest.raises(ValueError, match="MSA filter provenance.*manifest split"):
        build_split_fingerprint(**args, validate_schema=validate_schema)


@pytest.mark.parametrize("value", [None, np.asarray(["a" * 64]), np.asarray(""), np.asarray(7)])
def test_directed_policy_rejects_missing_or_invalid_feature_filter_provenance(
    directed_dataset: tuple[dict[str, Any], dict[str, Any]], value: np.ndarray | None,
) -> None:
    args, _ = directed_dataset
    path = chain_npz_path(args["processed_features_dir"], "2def_B")
    with np.load(path) as data:
        arrays = dict(data)
    if value is None:
        del arrays["msa_row_filter_sha256"]
    else:
        arrays["msa_row_filter_sha256"] = value
    np.savez_compressed(path, **arrays)
    with pytest.raises(ValueError, match="msa_row_filter_sha256|MSA filter provenance"):
        build_split_fingerprint(**args)


@pytest.mark.parametrize("assignment", ["swap", "unknown", "subset", "reordered"])
def test_directed_policy_rejects_manifests_without_exact_role_binding(
    directed_dataset: tuple[dict[str, Any], dict[str, Any]], tmp_path: Path, assignment: str,
) -> None:
    args, _ = directed_dataset
    manifests = args["manifest_paths"]
    if assignment == "swap":
        changed = {"train": manifests["val"], "val": manifests["train"]}
    elif assignment == "unknown":
        changed = {"hidden_val": manifests["val"]}
    else:
        subset = tmp_path / "subset.txt"
        subset.write_text("2def_B\n1abc_A\n" if assignment == "reordered" else "1abc_A\n")
        changed = {"train": subset}
    with pytest.raises(ValueError, match="[Mm]anifest|known train or val split"):
        build_split_fingerprint(**{**args, "manifest_paths": changed})


@pytest.mark.parametrize("change", ["filter", "policy_hash", "manifest_hash", "chain_count"])
def test_directed_policy_changes_invalidate_common_preprocessing_config(
    directed_dataset: tuple[dict[str, Any], dict[str, Any]], change: str,
) -> None:
    args, meta = directed_dataset
    original = preprocess_config_sha256(meta)
    reordered = json.loads(json.dumps(meta, sort_keys=True))
    reordered["cli_args"]["raw_root"] = "/different/local/path"
    assert preprocess_config_sha256(reordered) == original
    policy = meta["msa_filter_policy"]
    if change == "policy_hash":
        policy["policy_sha256"] = "e" * 64
    else:
        field = {"filter": "filter_sha256", "manifest_hash": "manifest_sha256", "chain_count": "chain_count"}[change]
        policy["splits"]["train"][field] = 3 if change == "chain_count" else "e" * 64
    assert preprocess_config_sha256(meta) != original
    (args["processed_features_dir"] / PREPROCESS_META_FILENAME).write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="provenance mismatch|does not match the directed"):
        build_split_fingerprint(**args)


@pytest.mark.parametrize("change", ["private_path", "unknown_split", "bad_digest", "bool_count", "null", "old_schema"])
def test_fingerprint_rejects_invalid_public_policy_metadata_even_without_array_validation(
    directed_dataset: tuple[dict[str, Any], dict[str, Any]], change: str,
) -> None:
    args, meta = directed_dataset
    policy = meta["msa_filter_policy"]
    if change == "private_path":
        policy["path"] = "/private/hidden-policy.json"
    elif change == "unknown_split":
        policy["splits"]["hidden"] = dict(policy["splits"]["val"])
    elif change == "bad_digest":
        policy["splits"]["train"]["filter_sha256"] = "NOT_A_DIGEST"
    elif change == "bool_count":
        policy["splits"]["train"]["chain_count"] = True
    elif change == "null":
        meta["msa_filter_policy"] = None
    else:
        meta["schema_version"] = 2
    (args["processed_features_dir"] / PREPROCESS_META_FILENAME).write_text(json.dumps(meta))
    with pytest.raises(ValueError):
        build_split_fingerprint(**args, validate_schema=False)
