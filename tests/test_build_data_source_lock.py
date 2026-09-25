from __future__ import annotations

import json
from pathlib import Path

import pytest

from nanofold.chain_paths import chain_data_dir
from scripts import build_data_source_lock as sources


def _fixture(tmp_path: Path, *, include_hidden: bool = False) -> tuple[list[str], Path]:
    data_root = tmp_path / "raw"
    manifests = tmp_path / "manifests"
    manifests.mkdir()
    (manifests / "train.txt").write_text("1abc_A\n")
    (manifests / "val.txt").write_text("2abc_A\n")
    (manifests / "all.txt").write_text("1abc_A\n2abc_A\n")
    for chain_id in ("1abc_A", "2abc_A", "3abc_A"):
        mmcif = data_root / "pdb_data/mmcif_files" / f"{chain_id[:4]}.cif"
        mmcif.parent.mkdir(parents=True, exist_ok=True)
        mmcif.write_text(f"data_{chain_id[:4]}\n")
        msa = chain_data_dir(data_root / "roda_pdb", chain_id) / "uniref90_hits.a3m"
        msa.parent.mkdir(parents=True)
        msa.write_text(">query\nAAAA\n")
    cache, metadata, source_lock, manifest_lock = [tmp_path / name for name in (
        "cache.json", "metadata.json", "metadata_sources.lock.json", "manifest.lock.json",
    )]
    for path in (cache, metadata, source_lock, manifest_lock):
        path.write_text("{}\n")
    out = tmp_path / "candidate-source.lock.json"
    argv = ["--data-root", str(data_root), "--manifests-dir", str(manifests),
            "--chain-data-cache", str(cache), "--structure-metadata", str(metadata),
            "--metadata-source-lock", str(source_lock), "--manifest-lock", str(manifest_lock),
            "--output", str(out), "--require-complete"]
    if include_hidden:
        hidden = tmp_path / "private/manifests/hidden_val.txt"
        hidden.parent.mkdir(parents=True)
        hidden.write_text("3abc_A\n")
        argv += ["--include-hidden", "--hidden-manifest", str(hidden)]
    return argv, out


def test_data_source_lock_pins_raw_trees_without_public_manifest_annotation(tmp_path: Path) -> None:
    argv, out = _fixture(tmp_path)
    sources.main(argv)
    lock = json.loads(out.read_text())
    assert lock["raw_assets"]["missing_mmcif_count"] == 0
    assert lock["raw_assets"]["missing_alignment_dir_count"] == 0
    assert lock["raw_assets"]["missing_alignment_file_count"] == 0
    assert lock["raw_assets"]["unique_chain_count"] == 2
    assert lock["raw_assets"]["required_msa_names"] == ["uniref90_hits.a3m"]
    assert lock["raw_assets"]["mmcif_tree_sha256"]
    assert "data_source_lock_sha256" not in json.loads((tmp_path / "manifest.lock.json").read_text())


def test_data_source_lock_requires_declared_msa_files(tmp_path: Path) -> None:
    argv, out = _fixture(tmp_path)
    with pytest.raises(SystemExit, match="missing_alignment_file_count=2"):
        sources.main([*argv, "--msa-names", "uniref90_hits.a3m,mgnify_hits.a3m"])
    assert not out.exists()


def test_data_source_lock_requires_hidden_manifest_when_included(tmp_path: Path) -> None:
    argv, out = _fixture(tmp_path, include_hidden=True)
    (tmp_path / "private/manifests/hidden_val.txt").unlink()
    with pytest.raises(SystemExit, match="missing_manifest_count=1"):
        sources.main(argv)
    assert not out.exists()


def test_data_source_lock_uses_private_hidden_manifest_path(tmp_path: Path) -> None:
    argv, out = _fixture(tmp_path, include_hidden=True)
    sources.main(argv)
    lock = json.loads(out.read_text())
    assert lock["manifests"]["hidden_val"]["path"].endswith("private/manifests/hidden_val.txt")
    assert lock["manifests"]["hidden_val"]["chain_count"] == 1
    assert lock["raw_assets"]["unique_chain_count"] == 3


@pytest.mark.parametrize("source", ["cache.json", "metadata.json", "metadata_sources.lock.json"])
@pytest.mark.parametrize("kind", ["missing", "empty", "directory"])
def test_require_complete_rejects_missing_source_evidence(tmp_path: Path, source: str, kind: str) -> None:
    argv, out = _fixture(tmp_path)
    path = tmp_path / source
    path.unlink()
    if kind == "empty":
        path.touch()
    elif kind == "directory":
        path.mkdir()
    with pytest.raises(SystemExit, match="missing_sources"):
        sources.main(argv)
    assert not out.exists()


@pytest.mark.parametrize("invalid", ["empty", "duplicate", "overlap", "wrong_union", "hidden_overlap"])
def test_require_complete_rejects_invalid_splits(tmp_path: Path, invalid: str) -> None:
    argv, out = _fixture(tmp_path, include_hidden=True)
    manifests = tmp_path / "manifests"
    if invalid == "empty":
        for split in ("train", "val", "all"):
            (manifests / f"{split}.txt").write_text("")
    elif invalid == "duplicate":
        (manifests / "train.txt").write_text("1abc_A\n1abc_A\n")
    elif invalid == "overlap":
        (manifests / "val.txt").write_text("1abc_A\n")
    elif invalid == "wrong_union":
        (manifests / "all.txt").write_text("1abc_A\n3abc_A\n")
    else:
        (tmp_path / "private/manifests/hidden_val.txt").write_text("1abc_A\n")
    with pytest.raises(SystemExit, match="manifest_errors"):
        sources.main(argv)
    assert not out.exists()


@pytest.mark.parametrize("asset", ["msa", "mmcif"])
@pytest.mark.parametrize("kind", ["empty", "directory"])
def test_require_complete_rejects_invalid_raw_asset_files(tmp_path: Path, asset: str, kind: str) -> None:
    argv, out = _fixture(tmp_path)
    path = (chain_data_dir(tmp_path / "raw/roda_pdb", "1abc_A") / "uniref90_hits.a3m"
            if asset == "msa" else tmp_path / "raw/pdb_data/mmcif_files/1abc.cif")
    path.unlink()
    if kind == "empty":
        path.touch()
    else:
        path.mkdir()
    with pytest.raises(SystemExit, match="incomplete"):
        sources.main(argv)
    assert not out.exists()


@pytest.mark.parametrize("alias", ["same_path", "symlink", "hardlink"])
def test_source_lock_cannot_overwrite_input(tmp_path: Path, alias: str) -> None:
    argv, _ = _fixture(tmp_path)
    source = tmp_path / "cache.json"
    output = source
    if alias != "same_path":
        output = tmp_path / "aliased-output.json"
        if alias == "symlink":
            output.symlink_to(source)
        else:
            output.hardlink_to(source)
    original = source.read_bytes()
    with pytest.raises(ValueError, match="alias or overwrite"):
        sources.main([*argv, "--output", str(output)])
    assert source.read_bytes() == original
