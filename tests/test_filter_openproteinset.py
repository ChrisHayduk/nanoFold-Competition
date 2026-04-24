from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np


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
        feature_dir / f"{chain_id}.npz",
        aatype=np.asarray(aatype, dtype=np.int32),
        msa=np.zeros((1, len(aatype)), dtype=np.int32),
        deletions=np.zeros((1, len(aatype)), dtype=np.int32),
        template_aatype=np.zeros((0, len(aatype)), dtype=np.int32),
        template_ca_coords=np.zeros((0, len(aatype), 3), dtype=np.float32),
        template_ca_mask=np.zeros((0, len(aatype)), dtype=bool),
    )
    np.savez_compressed(
        label_dir / f"{chain_id}.npz",
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
    assert entries["1tiny_A"]["reject_reasons"] == ["min_length"]
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
        features / "1ca_A.npz",
        aatype=np.asarray(list(range(20)) * 3, dtype=np.int32),
    )
    np.savez_compressed(
        labels / "1ca_A.npz",
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
    assert entry["reject_reasons"] == ["missing_label_key"]
    assert "atom14_positions" in entry["missing_label_keys"]
