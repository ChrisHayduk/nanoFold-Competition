from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from nanofold.chain_paths import chain_npz_path


def _load_filter_module():
    module_path = Path("scripts/filter_openproteinset.py").resolve()
    spec = importlib.util.spec_from_file_location("filter_openproteinset_script", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module spec for {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_npzs(feature_dir: Path, label_dir: Path, chain_id: str, *, aatype: list[int], resolution: float) -> None:
    np.savez_compressed(
        chain_npz_path(feature_dir, chain_id),
        aatype=np.asarray(aatype, dtype=np.int32),
        msa=np.asarray([aatype], dtype=np.int32),
        deletions=np.zeros((1, len(aatype)), dtype=np.int32),
        template_aatype=np.zeros((0, len(aatype)), dtype=np.int32),
        template_ca_coords=np.zeros((0, len(aatype), 3), dtype=np.float32),
        template_ca_mask=np.zeros((0, len(aatype)), dtype=bool),
    )
    np.savez_compressed(
        chain_npz_path(label_dir, chain_id),
        ca_coords=np.zeros((len(aatype), 3), dtype=np.float32),
        ca_mask=np.ones((len(aatype),), dtype=bool),
        atom14_positions=np.zeros((len(aatype), 14, 3), dtype=np.float32),
        atom14_mask=np.ones((len(aatype), 14), dtype=bool),
        resolution=np.asarray(resolution, dtype=np.float32),
    )


def test_build_manifest_filters_length_resolution_and_single_aa(tmp_path: Path) -> None:
    module = _load_filter_module()
    build_manifest = getattr(module, "build_manifest")

    features = tmp_path / "features"
    labels = tmp_path / "labels"
    features.mkdir()
    labels.mkdir()

    _write_npzs(features, labels, "1oky_A", aatype=list(range(20)) * 3, resolution=2.0)
    _write_npzs(features, labels, "1tiny_A", aatype=list(range(10)), resolution=2.0)
    _write_npzs(features, labels, "1loq_A", aatype=list(range(20)) * 3, resolution=3.5)
    _write_npzs(features, labels, "1poly_A", aatype=[7] * 50 + [1, 2], resolution=2.0)

    manifest = build_manifest(
        processed_features_dir=features,
        processed_labels_dir=labels,
        manifest_path=None,
        min_length=40,
        max_length=256,
        max_resolution=3.0,
        max_single_aa_fraction_threshold=0.8,
    )

    entries = {entry["chain_id"]: entry for entry in manifest["chains"]}
    assert entries["1oky_A"]["accepted"] is True
    assert "secondary_structure_class" in entries["1oky_A"]
    assert "secondary_helix_fraction" in entries["1oky_A"]
    assert entries["1tiny_A"]["reject_reasons"] == ["min_length", "insufficient_valid_ca"]
    assert entries["1loq_A"]["reject_reasons"] == ["resolution"]
    assert entries["1poly_A"]["reject_reasons"] == ["single_aa"]
    assert manifest["summary"]["accepted"] == 1
    assert manifest["summary"]["rejected"] == 3


def test_cluster_tsv_and_accepted_out(tmp_path: Path) -> None:
    module = _load_filter_module()
    build_manifest = getattr(module, "build_manifest")
    main = getattr(module, "main")

    features = tmp_path / "features"
    labels = tmp_path / "labels"
    features.mkdir()
    labels.mkdir()
    _write_npzs(features, labels, "1oky_A", aatype=list(range(20)) * 3, resolution=2.0)
    cluster_tsv = tmp_path / "clusters.tsv"
    cluster_tsv.write_text("rep1\t1oky_A\nrep1\t1other_A\n")

    manifest = build_manifest(
        processed_features_dir=features,
        processed_labels_dir=labels,
        manifest_path=None,
        min_length=40,
        max_length=256,
        max_resolution=3.0,
        max_single_aa_fraction_threshold=0.8,
        mmseqs_cluster_tsv=cluster_tsv,
    )
    assert manifest["chains"][0]["cluster_id"] == "rep1"
    assert manifest["chains"][0]["cluster_size"] == 2

    manifest_out = tmp_path / "filter.json"
    accepted_out = tmp_path / "accepted.txt"
    main(
        [
            "--processed-features-dir",
            str(features),
            "--processed-labels-dir",
            str(labels),
            "--manifest-out",
            str(manifest_out),
            "--accepted-out",
            str(accepted_out),
        ]
    )
    written = json.loads(manifest_out.read_text())
    assert written["summary"]["accepted"] == 1
    assert accepted_out.read_text() == "1oky_A\n"


def test_ca_only_labels_are_rejected(tmp_path: Path) -> None:
    module = _load_filter_module()
    build_manifest = getattr(module, "build_manifest")

    features = tmp_path / "features"
    labels = tmp_path / "labels"
    features.mkdir()
    labels.mkdir()
    np.savez_compressed(
        chain_npz_path(features, "1ca_A"),
        aatype=np.asarray(list(range(20)) * 3, dtype=np.int32),
    )
    np.savez_compressed(
        chain_npz_path(labels, "1ca_A"),
        ca_coords=np.zeros((60, 3), dtype=np.float32),
        ca_mask=np.ones((60,), dtype=bool),
    )

    manifest = build_manifest(
        processed_features_dir=features,
        processed_labels_dir=labels,
        manifest_path=None,
        min_length=40,
        max_length=256,
        max_resolution=3.0,
        max_single_aa_fraction_threshold=0.8,
    )

    entry = manifest["chains"][0]
    assert entry["accepted"] is False
    assert entry["reject_reasons"] == ["invalid_feature_label_schema"]
    assert any("atom14_positions" in error for error in entry["schema_errors"])


@pytest.mark.parametrize("mutation", ["missing_msa", "query_mismatch", "nonfinite", "wrong_chain", "wrong_length", "no_ca", "corrupt"])
def test_filter_cannot_accept_invalid_or_unsupervised_training_pairs(tmp_path: Path, mutation: str) -> None:
    module = _load_filter_module()
    features, labels = tmp_path / "features", tmp_path / "labels"
    features.mkdir()
    labels.mkdir()
    _write_npzs(features, labels, "1abc_A", aatype=list(range(20)) * 3, resolution=2.0)
    feature_path, label_path = chain_npz_path(features, "1abc_A"), chain_npz_path(labels, "1abc_A")
    with np.load(feature_path) as data:
        feature_data = dict(data)
    with np.load(label_path) as data:
        label_data = dict(data)
    if mutation == "missing_msa":
        del feature_data["msa"]
    elif mutation == "query_mismatch":
        feature_data["msa"][0, 0] = 1
    elif mutation == "nonfinite":
        label_data["ca_coords"][:] = np.nan
        label_data["atom14_positions"][:] = np.nan
        label_data["resolution"] = np.asarray(np.nan)
    elif mutation == "wrong_chain":
        feature_data["chain_id"] = np.asarray("2abc_A")
        label_data["chain_id"] = np.asarray("2abc_A")
    elif mutation == "wrong_length":
        for key in ("ca_coords", "ca_mask", "atom14_positions", "atom14_mask"):
            label_data[key] = label_data[key][:-1]
    elif mutation == "no_ca":
        label_data["ca_mask"][:] = False
        label_data["atom14_mask"][:] = False
    np.savez(feature_path, **feature_data)
    np.savez(label_path, **label_data)
    if mutation == "corrupt":
        label_path.write_bytes(b"not a numpy archive")
    result = module.evaluate_chain(
        "1abc_A", processed_features_dir=features, processed_labels_dir=labels,
        min_length=40, max_length=256, max_resolution=3.0, max_single_aa_fraction_threshold=0.8,
    )
    assert result["accepted"] is False
    assert result["reject_reasons"]


@pytest.mark.parametrize("rows", ["rep\tchain\nrep\tchain\n", "rep1\tchain\nrep2\tchain\n", "broken\n", "rep\t\n"])
def test_filter_cluster_audit_rejects_ambiguous_assignments(tmp_path: Path, rows: str) -> None:
    module = _load_filter_module()
    path = tmp_path / "clusters.tsv"
    path.write_text(rows)
    with pytest.raises(ValueError, match="cluster"):
        module.load_cluster_tsv(path)


def test_filter_requires_cluster_coverage_and_reports_unknown_resolution(tmp_path: Path) -> None:
    module = _load_filter_module()
    features, labels = tmp_path / "features", tmp_path / "labels"
    features.mkdir()
    labels.mkdir()
    _write_npzs(features, labels, "1abc_A", aatype=list(range(20)) * 3, resolution=0.0)
    kwargs = dict(
        processed_features_dir=features, processed_labels_dir=labels, manifest_path=None,
        min_length=40, max_length=256, max_resolution=3.0, max_single_aa_fraction_threshold=0.8,
    )
    entry = module.build_manifest(**kwargs)["chains"][0]
    assert entry["accepted"] is True
    assert entry["resolution_status"] == "unknown"
    path = tmp_path / "clusters.tsv"
    path.write_text("rep\t2abc_A\n")
    with pytest.raises(ValueError, match="cover every"):
        module.build_manifest(**kwargs, mmseqs_cluster_tsv=path)
