from __future__ import annotations

from pathlib import Path

import pytest

from nanofold.chain_paths import chain_error_path, chain_npz_path
from scripts.sync_processed_npz_files import main


def _touch_npz(directory: Path, chain_id: str) -> None:
    path = chain_npz_path(directory, chain_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"npz placeholder")


def test_sync_processed_npz_files_prunes_to_manifests(tmp_path: Path) -> None:
    features = tmp_path / "features"
    labels = tmp_path / "labels"
    manifest_a = tmp_path / "train.txt"
    manifest_b = tmp_path / "val.txt"
    manifest_a.write_text("1abc_A\n")
    manifest_b.write_text("2def_B\n")

    for directory in (features, labels):
        directory.mkdir()
        _touch_npz(directory, "1abc_A")
        _touch_npz(directory, "2def_B")
        _touch_npz(directory, "9xyz_C")
        chain_error_path(directory, "9xyz_C").write_text("projection failed\n")

    assert (
        main(
            [
                "--features-dir",
                str(features),
                "--labels-dir",
                str(labels),
                "--manifest",
                str(manifest_a),
                "--manifest",
                str(manifest_b),
                "--remove-errors",
            ]
        )
        == 0
    )

    for directory in (features, labels):
        assert chain_npz_path(directory, "1abc_A").is_file()
        assert chain_npz_path(directory, "2def_B").is_file()
        assert not chain_npz_path(directory, "9xyz_C").exists()
        assert not chain_error_path(directory, "9xyz_C").exists()


def test_sync_processed_npz_files_requires_manifest_coverage(tmp_path: Path) -> None:
    features = tmp_path / "features"
    labels = tmp_path / "labels"
    manifest = tmp_path / "hidden_val.txt"
    manifest.write_text("1abc_A\n")
    features.mkdir()
    labels.mkdir()
    _touch_npz(features, "1abc_A")

    with pytest.raises(FileNotFoundError, match="missing label NPZs"):
        main(
            [
                "--features-dir",
                str(features),
                "--labels-dir",
                str(labels),
                "--manifest",
                str(manifest),
            ]
        )
