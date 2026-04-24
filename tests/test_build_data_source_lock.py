from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


def _load_module():
    module_path = Path("scripts/build_data_source_lock.py").resolve()
    spec = importlib.util.spec_from_file_location("build_data_source_lock_script", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module spec for {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_data_source_lock_pins_raw_trees_and_annotates_manifest_lock(tmp_path: Path) -> None:
    module = _load_module()
    main = getattr(module, "main")

    data_root = tmp_path / "data"
    manifests = tmp_path / "manifests"
    mmcif_root = data_root / "pdb_data" / "mmcif_files"
    chain_dir = data_root / "roda_pdb" / "1abc_A" / "a3m"
    mmcif_root.mkdir(parents=True)
    chain_dir.mkdir(parents=True)
    (mmcif_root / "1abc.cif").write_text("data_1abc\n")
    (chain_dir / "uniref90_hits.a3m").write_text(">q\nAAAA\n")
    manifests.mkdir()
    (manifests / "train.txt").write_text("1abc_A\n")
    (manifests / "val.txt").write_text("")
    (manifests / "all.txt").write_text("1abc_A\n")
    chain_cache = data_root / "pdb_data" / "data_caches" / "chain_data_cache.json"
    chain_cache.parent.mkdir(parents=True)
    chain_cache.write_text("{}")
    structure_metadata = manifests / "structure_metadata.json"
    structure_metadata.write_text('{"chains":[]}\n')
    source_lock = tmp_path / "metadata_sources.lock.json"
    source_lock.write_text("{}\n")
    manifest_lock = tmp_path / "manifest.lock.json"
    manifest_lock.write_text("{}\n")
    out = tmp_path / "official_data_source.lock.json"

    main(
        [
            "--data-root",
            str(data_root),
            "--manifests-dir",
            str(manifests),
            "--chain-data-cache",
            str(chain_cache),
            "--structure-metadata",
            str(structure_metadata),
            "--metadata-source-lock",
            str(source_lock),
            "--manifest-lock",
            str(manifest_lock),
            "--output",
            str(out),
            "--require-complete",
        ]
    )

    lock = json.loads(out.read_text())
    assert lock["raw_assets"]["missing_mmcif_count"] == 0
    assert lock["raw_assets"]["missing_alignment_dir_count"] == 0
    assert lock["raw_assets"]["missing_alignment_file_count"] == 0
    assert lock["raw_assets"]["required_msa_names"] == ["uniref90_hits.a3m"]
    assert lock["raw_assets"]["mmcif_tree_sha256"]
    assert json.loads(manifest_lock.read_text())["data_source_lock_sha256"]


def test_data_source_lock_requires_declared_msa_files(tmp_path: Path) -> None:
    module = _load_module()
    main = getattr(module, "main")

    data_root = tmp_path / "data"
    manifests = tmp_path / "manifests"
    mmcif_root = data_root / "pdb_data" / "mmcif_files"
    chain_dir = data_root / "roda_pdb" / "1abc_A" / "a3m"
    mmcif_root.mkdir(parents=True)
    chain_dir.mkdir(parents=True)
    (mmcif_root / "1abc.cif").write_text("data_1abc\n")
    (chain_dir / "uniref90_hits.a3m").write_text(">q\nAAAA\n")
    manifests.mkdir()
    (manifests / "train.txt").write_text("1abc_A\n")
    (manifests / "val.txt").write_text("")
    (manifests / "all.txt").write_text("1abc_A\n")
    chain_cache = data_root / "pdb_data" / "data_caches" / "chain_data_cache.json"
    chain_cache.parent.mkdir(parents=True)
    chain_cache.write_text("{}")
    structure_metadata = manifests / "structure_metadata.json"
    structure_metadata.write_text('{"chains":[]}\n')
    source_lock = tmp_path / "metadata_sources.lock.json"
    source_lock.write_text("{}\n")
    manifest_lock = tmp_path / "manifest.lock.json"
    manifest_lock.write_text("{}\n")
    out = tmp_path / "official_data_source.lock.json"

    try:
        main(
            [
                "--data-root",
                str(data_root),
                "--manifests-dir",
                str(manifests),
                "--chain-data-cache",
                str(chain_cache),
                "--structure-metadata",
                str(structure_metadata),
                "--metadata-source-lock",
                str(source_lock),
                "--manifest-lock",
                str(manifest_lock),
                "--output",
                str(out),
                "--msa-names",
                "uniref90_hits.a3m,mgnify_hits.a3m",
                "--require-complete",
            ]
        )
    except SystemExit as exc:
        assert "missing_alignment_file_count=1" in str(exc)
    else:
        raise AssertionError("missing required MSA file did not fail the official source lock")


def test_data_source_lock_requires_hidden_manifest_when_included(tmp_path: Path) -> None:
    module = _load_module()
    main = getattr(module, "main")

    data_root = tmp_path / "data"
    manifests = tmp_path / "manifests"
    manifests.mkdir()
    (manifests / "train.txt").write_text("")
    (manifests / "val.txt").write_text("")
    (manifests / "all.txt").write_text("")
    chain_cache = data_root / "pdb_data" / "data_caches" / "chain_data_cache.json"
    chain_cache.parent.mkdir(parents=True)
    chain_cache.write_text("{}")
    structure_metadata = manifests / "structure_metadata.json"
    structure_metadata.write_text('{"chains":[]}\n')
    source_lock = tmp_path / "metadata_sources.lock.json"
    source_lock.write_text("{}\n")
    manifest_lock = tmp_path / "manifest.lock.json"
    manifest_lock.write_text("{}\n")
    out = tmp_path / "official_data_source.lock.json"

    try:
        main(
            [
                "--data-root",
                str(data_root),
                "--manifests-dir",
                str(manifests),
                "--chain-data-cache",
                str(chain_cache),
                "--structure-metadata",
                str(structure_metadata),
                "--metadata-source-lock",
                str(source_lock),
                "--manifest-lock",
                str(manifest_lock),
                "--output",
                str(out),
                "--include-hidden",
                "--require-complete",
            ]
        )
    except SystemExit as exc:
        assert "missing_manifest_count=1" in str(exc)
    else:
        raise AssertionError("missing included hidden manifest did not fail the official source lock")
