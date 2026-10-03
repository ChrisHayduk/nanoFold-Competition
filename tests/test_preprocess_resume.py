from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import gemmi
import numpy as np
import pytest

from nanofold.chain_paths import chain_data_dir, chain_npz_path


def _write_structure(path: Path, offset: float = 0, length: int = 3) -> None:
    structure = gemmi.Structure()
    model = gemmi.Model(1)
    chain = gemmi.Chain("A")
    for index in range(length):
        residue = gemmi.Residue()
        residue.name = "ALA"
        residue.seqid = gemmi.SeqId(index + 1, " ")
        residue.label_seq = index + 1
        residue.entity_type = gemmi.EntityType.Polymer
        atom = gemmi.Atom()
        atom.name = "CA"
        atom.element = gemmi.Element("C")
        atom.pos = gemmi.Position(offset + index * 3.8, 0, 0)
        atom.occ = 1
        residue.add_atom(atom)
        chain.add_residue(residue)
    model.add_chain(chain)
    structure.add_model(model)
    structure.setup_entities()
    structure.make_mmcif_document().write_file(str(path))


@pytest.fixture
def preprocessing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    spec = importlib.util.spec_from_file_location("preprocess_resume_script", Path("scripts/preprocess.py").resolve())
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    alignments = tmp_path / "alignments"
    chain_dir = chain_data_dir(alignments, "1abc_A")
    chain_dir.mkdir(parents=True)
    msa = chain_dir / "uniref90_hits.a3m"
    msa.write_text(">query\nAAA\n>hit\nAA-\n")
    mmcifs = tmp_path / "mmcif"
    mmcifs.mkdir()
    cif = mmcifs / "1abc.cif"
    _write_structure(cif)
    manifest = tmp_path / "train.txt"
    manifest.write_text("1abc_A\n")
    features = tmp_path / "features"
    labels = tmp_path / "labels"
    argv = [
        "preprocess.py", "--manifest", str(manifest), "--alignments-root", str(alignments),
        "--mmcif-root", str(mmcifs), "--processed-features-dir", str(features),
        "--processed-labels-dir", str(labels), "--disable-templates", "--fail-fast",
        "--min-projection-valid-ca", "3",
    ]

    def run(*extra: str, strict: bool = True) -> None:
        monkeypatch.setattr(sys, "argv", [*argv, *(["--strict"] if strict else []), *extra])
        module.main()

    return module, run, features, labels, msa, cif


def test_pinned_cache_rejects_wrong_fallback_query_that_passes_projection(preprocessing, tmp_path: Path) -> None:
    _, run, features, labels, msa, cif = preprocessing
    cached_sequence = "A" * 40
    fallback_query = "A" * 39 + "R"
    msa.write_text(">query\n" + fallback_query + "\n")
    _write_structure(cif, length=40)
    cache = tmp_path / "cache.json"
    cache.write_text(json.dumps({"1abc_A": {"seq": cached_sequence, "seq_length": 40}}))
    control_features = tmp_path / "unbound_features"
    control_labels = tmp_path / "unbound_labels"
    run("--processed-features-dir", str(control_features), "--processed-labels-dir", str(control_labels),
        "--min-projection-valid-ca", "32", strict=False)
    with np.load(chain_npz_path(control_labels, "1abc_A")) as data:
        assert np.count_nonzero(data["ca_mask"]) == 39
    with pytest.raises(ValueError, match="Acquired MSA query differs from the pinned chain cache target"):
        run("--chain-data-cache", str(cache), "--min-projection-valid-ca", "32", strict=False)
    assert not list(features.glob("*.npz"))
    assert not list(labels.glob("*.npz"))


@pytest.mark.parametrize("contents", [
    "[]", "{}", '{"1abc_a":{"seq":"AAA"}}', '{"1abc_A":[]}',
    '{"1abc_A":{"seq":123}}', '{"1abc_A":{"seq":""}}',
    '{"1abc_A":{"seq":"AAA","sequence":"AAR"}}',
    '{"1abc_A":{"seq":"AAA","sequence":null}}',
    '{"1abc_A":{"seq":"AAX"}}', '{"1abc_A":{"seq":"A-A"}}',
    '{"1abc_A":{"seq":"AAA","seq_length":2}}',
    '{"1abc_A":{"seq":"AAA","seq_length":3,"sequence_length":4}}',
    '{"1abc_A":{"seq":"AAA","seq_length":"3"}}',
    '{"1abc_A":{"seq":"AAA","seq_length":3.0}}',
    '{"1abc_A":{"seq":"AAA","seq_length":true}}',
    '{"1abc_A":{"seq":"AAA"},"1abc_A":{"seq":"AAR"}}',
    '{"1abc_A":{"seq":"AAR","seq":"AAA"}}',
])
def test_invalid_selected_cache_fails_before_outputs(preprocessing, tmp_path: Path, contents: str) -> None:
    _, run, features, labels, _, _ = preprocessing
    cache = tmp_path / "cache.json"
    cache.write_text(contents)
    with pytest.raises(ValueError):
        run("--chain-data-cache", str(cache))
    assert not features.exists()
    assert not labels.exists()


def test_cache_provenance_binds_bytes_and_resume_checks_target(preprocessing, tmp_path: Path) -> None:
    from nanofold.dataset_integrity import preprocess_config_sha256

    _, run, features, labels, _, _ = preprocessing
    cache = tmp_path / "cache.json"
    cache.write_text(json.dumps({"1abc_A": {"seq": "AAA", "seq_length": 3}}))
    run("--chain-data-cache", str(cache))
    feature_path = chain_npz_path(features, "1abc_A")
    label_path = chain_npz_path(labels, "1abc_A")
    meta_path = features / "preprocess_meta.json"
    metadata = json.loads(meta_path.read_bytes())
    assert metadata["cli_args"]["chain_data_cache_sha256"] == hashlib.sha256(cache.read_bytes()).hexdigest()
    bound_config = preprocess_config_sha256(metadata)
    del metadata["cli_args"]["chain_data_cache_sha256"]
    assert preprocess_config_sha256(metadata) != bound_config
    before_mtime = feature_path.stat().st_mtime_ns
    run("--chain-data-cache", str(cache), "--skip-existing")
    assert feature_path.stat().st_mtime_ns == before_mtime
    with np.load(feature_path) as data:
        arrays = dict(data)
    arrays["aatype"][:] = 1
    arrays["msa"][0] = arrays["aatype"]
    np.savez_compressed(feature_path, **arrays)
    run("--chain-data-cache", str(cache), "--skip-existing")
    with np.load(feature_path) as data:
        assert np.array_equal(data["aatype"], np.zeros(3, dtype=np.int32))
    before = (feature_path.read_bytes(), label_path.read_bytes(), meta_path.read_bytes())
    cache.write_text(cache.read_text() + "\n")
    with pytest.raises(ValueError, match="different preprocessing provenance"):
        run("--chain-data-cache", str(cache), "--skip-existing")
    assert (feature_path.read_bytes(), label_path.read_bytes(), meta_path.read_bytes()) == before


def test_uncached_outputs_cannot_be_resumed_as_cache_bound(preprocessing, tmp_path: Path) -> None:
    _, run, features, _, _, _ = preprocessing
    run()
    meta_path = features / "preprocess_meta.json"
    before = meta_path.read_bytes()
    assert "chain_data_cache_sha256" not in json.loads(before)["cli_args"]
    cache = tmp_path / "cache.json"
    cache.write_text(json.dumps({"1abc_A": {"seq": "AAA", "seq_length": 3}}))
    with pytest.raises(ValueError, match="different preprocessing provenance"):
        run("--chain-data-cache", str(cache), "--skip-existing")
    assert meta_path.read_bytes() == before


@pytest.mark.parametrize("output_name", ["preprocess_meta.json", "msa_row_filter_audit.json", "chain_316162635f41.error.txt"])
@pytest.mark.parametrize("hardlink", [False, True])
def test_pinned_cache_cannot_alias_written_outputs(preprocessing, tmp_path: Path, output_name: str, hardlink: bool) -> None:
    _, run, features, labels, _, _ = preprocessing
    features.mkdir()
    output = features / output_name
    output.write_text(json.dumps({"1abc_A": {"seq": "AAA", "seq_length": 3}}))
    cache = output
    if hardlink:
        cache = tmp_path / "pinned-cache.json"
        cache.hardlink_to(output)
    before = cache.read_bytes()
    with pytest.raises(ValueError, match="Pinned chain cache must be separate"):
        run("--chain-data-cache", str(cache))
    assert cache.read_bytes() == output.read_bytes() == before
    assert not labels.exists()
    assert list(features.iterdir()) == [output]


def test_resume_preserves_metadata_and_skips_only_unchanged_inputs(preprocessing) -> None:
    _, run, features, labels, msa, cif = preprocessing
    run()
    feature_path = chain_npz_path(features, "1abc_A")
    label_path = chain_npz_path(labels, "1abc_A")
    before = feature_path.stat().st_mtime_ns
    metadata_before = (features / "preprocess_meta.json").read_bytes()
    run("--skip-existing")
    assert feature_path.stat().st_mtime_ns == before
    assert (features / "preprocess_meta.json").read_bytes() == metadata_before

    msa.write_text(">query\nAAA\n>hit\nAA-\n>new\nA-A\n")
    run("--skip-existing")
    with np.load(feature_path) as data:
        assert data["msa"].shape == (3, 3)
        msa_source_hash = data["source_inputs_sha256"].item()

    _write_structure(cif, offset=4)
    run("--skip-existing")
    with np.load(feature_path) as data, np.load(label_path) as ground_truth:
        assert data["source_inputs_sha256"].item() != msa_source_hash
        assert data["source_inputs_sha256"].item() == ground_truth["source_inputs_sha256"].item()
        assert ground_truth["ca_coords"][0, 0] == 4


@pytest.mark.parametrize("manifest_text", ["", "# no targets\n", "1abc_A\n1abc_A\n", "bad-id\n", "../../bad_A\n"])
def test_invalid_manifest_fails_before_creating_outputs(preprocessing, tmp_path: Path, manifest_text: str) -> None:
    _, run, features, labels, _, _ = preprocessing
    manifest = tmp_path / "invalid.txt"
    manifest.write_text(manifest_text)
    with pytest.raises(ValueError):
        run("--manifest", str(manifest))
    assert not features.exists()
    assert not labels.exists()


@pytest.mark.parametrize("relationship", ["same", "labels_inside_features", "features_inside_labels"])
def test_feature_and_label_roots_must_be_disjoint(preprocessing, relationship: str) -> None:
    _, run, features, labels, _, _ = preprocessing
    if relationship == "same":
        labels = features
    elif relationship == "labels_inside_features":
        labels = features / "labels"
    else:
        features = labels / "features"
    with pytest.raises(ValueError, match="must not contain each other"):
        run("--processed-features-dir", str(features), "--processed-labels-dir", str(labels))
    assert not features.exists()
    assert not labels.exists()


def test_resume_rejects_changed_config_without_relabeling_existing_output(preprocessing) -> None:
    _, run, features, _, _, _ = preprocessing
    run()
    meta = features / "preprocess_meta.json"
    before = meta.read_bytes()
    with pytest.raises(ValueError, match="different preprocessing provenance"):
        run("--skip-existing", "--max-msa-seqs", "1")
    assert meta.read_bytes() == before
    with np.load(chain_npz_path(features, "1abc_A")) as data:
        assert data["msa"].shape[0] == 2


@pytest.mark.parametrize("missing_metadata", [True, False])
def test_populated_outputs_require_current_explicit_provenance(preprocessing, missing_metadata: bool) -> None:
    _, run, features, _, _, _ = preprocessing
    run()
    meta = features / "preprocess_meta.json"
    if missing_metadata:
        meta.unlink()
    else:
        recorded = json.loads(meta.read_text())
        recorded["schema_version"] = 2
        meta.write_text(json.dumps(recorded))
    with pytest.raises(ValueError, match="fresh output directories"):
        run("--skip-existing")


def test_resume_reprocesses_corrupt_shapes_and_nonfinite_labels(preprocessing) -> None:
    _, run, features, labels, _, _ = preprocessing
    run()
    feature_path = chain_npz_path(features, "1abc_A")
    label_path = chain_npz_path(labels, "1abc_A")
    with np.load(feature_path) as data:
        arrays = dict(data)
    arrays["deletions"] = np.zeros((1, 1), dtype=np.int32)
    np.savez_compressed(feature_path, **arrays)
    with np.load(label_path) as data:
        arrays = dict(data)
    arrays["atom14_positions"][0, 0, 0] = np.nan
    np.savez_compressed(label_path, **arrays)
    run("--skip-existing")
    with np.load(feature_path) as data, np.load(label_path) as ground_truth:
        assert data["deletions"].shape == data["msa"].shape
        assert np.isfinite(ground_truth["atom14_positions"]).all()


@pytest.mark.parametrize("valid_ca,recorded_count", [(0, 3), (2, 2), (3, 999)])
def test_resume_reprocesses_labels_below_quality_or_with_false_counts(
    preprocessing, valid_ca: int, recorded_count: int,
) -> None:
    _, run, features, labels, _, _ = preprocessing
    run()
    feature_path = chain_npz_path(features, "1abc_A")
    label_path = chain_npz_path(labels, "1abc_A")
    with np.load(feature_path) as data, np.load(label_path) as ground_truth:
        feature_arrays, label_arrays = dict(data), dict(ground_truth)
    # Keep provenance and all structural shapes valid; only supervision quality
    # or its recorded diagnostic changes.
    label_arrays["ca_mask"][valid_ca:] = False
    label_arrays["atom14_mask"][valid_ca:] = False
    feature_arrays["projection_valid_ca_count"] = np.asarray(recorded_count, dtype=np.int32)
    np.savez_compressed(feature_path, **feature_arrays)
    np.savez_compressed(label_path, **label_arrays)
    run("--skip-existing")
    with np.load(feature_path) as data, np.load(label_path) as ground_truth:
        assert np.count_nonzero(ground_truth["ca_mask"]) == 3
        assert data["projection_valid_ca_count"].item() == 3


@pytest.mark.parametrize("metric", [
    "projection_seq_identity", "projection_alignment_coverage", "projection_aligned_fraction",
])
def test_resume_reprocesses_cached_projection_below_configured_threshold(preprocessing, metric: str) -> None:
    _, run, features, _, _, _ = preprocessing
    run(strict=False)
    feature_path = chain_npz_path(features, "1abc_A")
    with np.load(feature_path) as data:
        arrays = dict(data)
    arrays[metric] = np.asarray(0.1, dtype=np.float32)
    np.savez_compressed(feature_path, **arrays)
    run("--skip-existing", strict=False)
    with np.load(feature_path) as data:
        assert data[metric].item() == 1.0


def test_resume_preserves_valid_float32_projection_threshold_boundary(preprocessing) -> None:
    _, run, features, _, msa, cif = preprocessing
    _write_structure(cif, length=10)
    msa.write_text(">query\nAAAAAAAAAR\n")
    run(strict=False)
    feature_path = chain_npz_path(features, "1abc_A")
    with np.load(feature_path) as data:
        assert data["projection_seq_identity"].item() == np.float32(0.9)
    modified = feature_path.stat().st_mtime_ns
    run("--skip-existing", strict=False)
    assert feature_path.stat().st_mtime_ns == modified


def test_invalid_generated_arrays_never_replace_valid_files(preprocessing) -> None:
    module, run, features, labels, _, _ = preprocessing
    run()
    feature_path = chain_npz_path(features, "1abc_A")
    label_path = chain_npz_path(labels, "1abc_A")
    before = (feature_path.read_bytes(), label_path.read_bytes())
    with np.load(feature_path) as data, np.load(label_path) as ground_truth:
        feature_arrays, label_arrays = dict(data), dict(ground_truth)
    feature_arrays["msa"][0, 0] = 1
    with pytest.raises(ValueError, match="query row"):
        module._save_validated_outputs(feature_path, feature_arrays, label_path, label_arrays)
    assert (feature_path.read_bytes(), label_path.read_bytes()) == before
    assert list(features.glob("*.npz")) == [feature_path]
    assert list(labels.glob("*.npz")) == [label_path]


def test_fingerprint_cannot_legitimize_mixed_preprocessing_configurations(preprocessing, tmp_path: Path) -> None:
    from nanofold.dataset_integrity import build_split_fingerprint

    _, run, features, labels, _, _ = preprocessing
    run()
    manifest = tmp_path / "fingerprint_manifest.txt"
    manifest.write_text("1abc_A\n")
    fingerprint_args = {
        "processed_features_dir": features,
        "processed_labels_dir": labels,
        "manifest_paths": {"train": manifest},
        "require_no_missing": True,
        "require_labels": True,
    }
    build_split_fingerprint(**fingerprint_args)
    # Matching pair-level hashes still cannot override the run-level provenance.
    for path in (chain_npz_path(features, "1abc_A"), chain_npz_path(labels, "1abc_A")):
        with np.load(path) as data:
            arrays = dict(data)
        arrays["preprocess_config_sha256"] = np.asarray("a" * 64)
        np.savez_compressed(path, **arrays)
    with pytest.raises(ValueError, match="Preprocessing provenance mismatch"):
        build_split_fingerprint(**fingerprint_args)


def test_preprocessing_records_resolved_author_for_label_chain_alias(preprocessing) -> None:
    _, run, features, labels, _, cif = preprocessing
    structure = gemmi.read_structure(str(cif))
    structure[0][0].name = "AUTHOR"
    for residue in structure[0][0]:
        residue.subchain = "A"
    structure.make_mmcif_document().write_file(str(cif))
    run()
    for path in (chain_npz_path(features, "1abc_A"), chain_npz_path(labels, "1abc_A")):
        with np.load(path) as data:
            assert data["chain_id"].item() == "1abc_A"
            assert data["resolved_author_chain_id"].item() == "AUTHOR"
            assert data["chain_mapping_method"].item() == "label_id"


def test_resume_and_fingerprint_reject_swapped_homomer_pairs(preprocessing, tmp_path: Path) -> None:
    from nanofold.dataset_integrity import build_split_fingerprint

    _, run, features, labels, msa, cif = preprocessing
    structure = gemmi.read_structure(str(cif))
    second_chain = structure[0][0].clone()
    second_chain.name = "B"
    for residue in second_chain:
        residue.subchain = "B_label"
        for atom in residue:
            atom.pos.x += 100
    structure[0].add_chain(second_chain)
    structure.make_mmcif_document().write_file(str(cif))
    second_msa = chain_data_dir(msa.parent.parent, "1abc_B") / msa.name
    second_msa.parent.mkdir()
    second_msa.write_bytes(msa.read_bytes())
    manifest = tmp_path / "homomers.txt"
    manifest.write_text("1abc_A\n1abc_B\n")
    run("--manifest", str(manifest))
    with np.load(chain_npz_path(features, "1abc_A")) as first, np.load(chain_npz_path(features, "1abc_B")) as second:
        assert first["source_inputs_sha256"].item() != second["source_inputs_sha256"].item()
    for directory in (features, labels):
        chain_npz_path(directory, "1abc_B").write_bytes(chain_npz_path(directory, "1abc_A").read_bytes())
    for require_labels in (True, False):
        with pytest.raises(ValueError, match="chain_id.*requested chain 1abc_B"):
            build_split_fingerprint(
                processed_features_dir=features,
                processed_labels_dir=labels if require_labels else None,
                manifest_paths={"train": manifest},
                require_no_missing=True,
                require_labels=require_labels,
            )
    run("--manifest", str(manifest), "--skip-existing")
    with np.load(chain_npz_path(labels, "1abc_B")) as data:
        assert data["chain_id"].item() == "1abc_B"
        assert data["ca_coords"][0, 0] == 100


@pytest.mark.parametrize("template_pdb,template_chain,expected_count", [
    ("1abc", "A", 0), ("1abc", "X", 0), ("1abc", "B", 0), ("2def", "A", 1),
])
def test_templates_exclude_target_entry_including_aliases_and_homomers(
    preprocessing, template_pdb: str, template_chain: str, expected_count: int,
) -> None:
    module, _, _, _, msa, cif = preprocessing
    _write_structure(cif, length=5)
    structure = gemmi.read_structure(str(cif))
    for residue in structure[0][0]:
        residue.subchain = "X"
    second_chain = structure[0][0].clone()
    second_chain.name = "B"
    for residue in second_chain:
        residue.subchain = "Y"
    structure[0].add_chain(second_chain)
    structure.make_mmcif_document().write_file(str(cif))
    (cif.parent / "2def.cif").write_bytes(cif.read_bytes())
    (msa.parent / "pdb70_hits.hhr").write_text(
        f"No 1\n>{template_pdb}_{template_chain} synthetic\n"
        f"Q query 1 AAAAA 5 (5)\nT {template_pdb}_{template_chain} 1 AAAAA 5 (5)\n"
    )
    templates = module._extract_template_features(
        chain_dir=msa.parent, mmcif_root=cif.parent, target_pdb_id="1abc", target_chain_id="A",
        target_seq="AAAAA", target_L=5, template_hhr_name="pdb70_hits.hhr", max_templates=1,
    )
    assert templates["template_ca_coords"].shape == (expected_count, 5, 3)


def test_strict_preprocessing_enforces_valid_coordinate_minimum(preprocessing) -> None:
    _, run, features, labels, _, _ = preprocessing
    with pytest.raises(ValueError, match="valid C-alpha count below threshold"):
        run("--min-projection-valid-ca", "4")
    assert not list(features.glob("*.npz"))
    assert not list(labels.glob("*.npz"))


@pytest.mark.parametrize("flag,value", [
    ("--min-projection-seq-identity", "nan"),
    ("--min-projection-coverage", "inf"),
    ("--min-projection-aligned-fraction", "-1"),
    ("--min-projection-valid-ca", "0"),
    ("--max-msa-seqs", "-1"),
    ("--max-templates", "-1"),
])
def test_invalid_preprocessing_thresholds_fail_before_creating_outputs(preprocessing, flag: str, value: str) -> None:
    _, run, features, labels, _, _ = preprocessing
    with pytest.raises(ValueError, match="must be"):
        run(flag, value)
    assert not features.exists()
    assert not labels.exists()


@pytest.mark.parametrize("metadata_change", ["missing", "downgraded"])
@pytest.mark.parametrize("require_labels", [True, False])
def test_fingerprint_requires_run_provenance_for_new_pairs(
    preprocessing, tmp_path: Path, metadata_change: str, require_labels: bool,
) -> None:
    from nanofold.dataset_integrity import build_split_fingerprint

    _, run, features, labels, _, _ = preprocessing
    run()
    meta = features / "preprocess_meta.json"
    if metadata_change == "missing":
        meta.unlink()
    else:
        raw = json.loads(meta.read_text())
        raw["schema_version"] = 2
        meta.write_text(json.dumps(raw))
    manifest = tmp_path / "fingerprint_manifest.txt"
    manifest.write_text("1abc_A\n")
    with pytest.raises(ValueError, match="require schema-version 3 preprocessing metadata"):
        build_split_fingerprint(
            processed_features_dir=features,
            processed_labels_dir=labels if require_labels else None,
            manifest_paths={"train": manifest},
            require_no_missing=True,
            require_labels=require_labels,
        )


def test_template_coordinates_follow_polymer_positions_after_missing_residues(preprocessing) -> None:
    module, _, _, _, msa, cif = preprocessing
    template_path = cif.parent / "2def.cif"
    _write_structure(template_path, length=6)
    structure = gemmi.read_structure(str(template_path))
    del structure[0][0][2]
    structure.make_mmcif_document().write_file(str(template_path))
    (msa.parent / "pdb70_hits.hhr").write_text(
        "No 1\n>2def_A synthetic\nQ query 1 AAAAAA 6 (6)\nT 2def_A 1 AAAAAA 6 (6)\n"
    )
    templates = module._extract_template_features(
        chain_dir=msa.parent, mmcif_root=cif.parent, target_pdb_id="1abc", target_chain_id="A",
        target_seq="AAAAAA", target_L=6, template_hhr_name="pdb70_hits.hhr", max_templates=1,
    )
    np.testing.assert_array_equal(templates["template_ca_mask"], [[True, True, False, True, True, True]])
    np.testing.assert_allclose(templates["template_ca_coords"][0, :, 0], [0, 3.8, 0, 11.4, 15.2, 19.0])


@pytest.mark.parametrize("invalid_mapping", ["missing_indices", "duplicate_indices", "query_mismatch", "template_mismatch"])
def test_templates_omit_unverifiable_residue_mappings(preprocessing, invalid_mapping: str) -> None:
    module, _, _, _, msa, cif = preprocessing
    template_path = cif.parent / "2def.cif"
    _write_structure(template_path, length=6)
    structure = gemmi.read_structure(str(template_path))
    if invalid_mapping == "missing_indices":
        for residue in structure[0][0]:
            residue.label_seq = None
    elif invalid_mapping == "duplicate_indices":
        structure[0][0][1].label_seq = 1
    structure.make_mmcif_document().write_file(str(template_path))
    query = "GGGGGG" if invalid_mapping == "query_mismatch" else "AAAAAA"
    template = "GGGGGG" if invalid_mapping == "template_mismatch" else "AAAAAA"
    (msa.parent / "pdb70_hits.hhr").write_text(
        f"No 1\n>2def_A synthetic\nQ query 1 {query} 6 (6)\nT 2def_A 1 {template} 6 (6)\n"
    )
    templates = module._extract_template_features(
        chain_dir=msa.parent, mmcif_root=cif.parent, target_pdb_id="1abc", target_chain_id="A",
        target_seq="AAAAAA", target_L=6, template_hhr_name="pdb70_hits.hhr", max_templates=1,
    )
    assert templates["template_ca_coords"].shape == (0, 6, 3)


@pytest.mark.parametrize("changed_input", ["mmcif", "msa"])
def test_inputs_changed_during_parsing_cannot_publish_stale_provenance(
    preprocessing, monkeypatch: pytest.MonkeyPatch, changed_input: str,
) -> None:
    module, run, features, labels, msa, cif = preprocessing
    original_reader = module._read_merged_msa

    def change_bound_input(*args, **kwargs):
        if changed_input == "mmcif":
            _write_structure(cif, offset=40)
        else:
            msa.write_text(">query\nAAA\n>different_hit\nA-A\n")
        return original_reader(*args, **kwargs)

    monkeypatch.setattr(module, "_read_merged_msa", change_bound_input)
    with pytest.raises(ValueError, match="Source inputs changed during preprocessing"):
        run()
    assert not list(features.glob("*.npz"))
    assert not list(labels.glob("*.npz"))


def test_inputs_changed_during_resume_cannot_be_reported_as_unchanged(
    preprocessing, monkeypatch: pytest.MonkeyPatch,
) -> None:
    module, run, features, labels, _, cif = preprocessing
    run()
    original_pairs = {
        path: path.read_bytes() for root in (features, labels) for path in root.glob("*.npz")
    }
    original_validator = module._existing_outputs_are_valid

    def change_bound_input(*args, **kwargs):
        valid = original_validator(*args, **kwargs)
        assert valid
        _write_structure(cif, offset=40)
        return valid

    monkeypatch.setattr(module, "_existing_outputs_are_valid", change_bound_input)
    with pytest.raises(ValueError, match="Source inputs changed while validating existing outputs"):
        run("--skip-existing")
    assert all(path.read_bytes() == contents for path, contents in original_pairs.items())
