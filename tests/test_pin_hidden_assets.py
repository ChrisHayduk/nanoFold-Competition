from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from nanofold.chain_paths import chain_npz_path
from nanofold.dataset_integrity import build_split_fingerprint
from scripts import pin_hidden_assets as pin


def _fixture(tmp_path: Path) -> tuple[list[str], Path]:
    manifest = tmp_path / "hidden.txt"
    manifest.write_text("1abc_A\n")
    features, labels = tmp_path / "features", tmp_path / "labels"
    features.mkdir()
    labels.mkdir()
    np.savez_compressed(chain_npz_path(features, "1abc_A"),
                        aatype=np.zeros(6, dtype=np.int32), msa=np.zeros((1, 6), dtype=np.int32),
                        deletions=np.zeros((1, 6), dtype=np.int32),
                        template_aatype=np.zeros((0, 6), dtype=np.int32),
                        template_ca_coords=np.zeros((0, 6, 3), dtype=np.float32),
                        template_ca_mask=np.zeros((0, 6), dtype=bool))
    np.savez_compressed(chain_npz_path(labels, "1abc_A"),
                        ca_coords=np.zeros((6, 3), dtype=np.float32), ca_mask=np.ones(6, dtype=bool),
                        atom14_positions=np.zeros((6, 14, 3), dtype=np.float32),
                        atom14_mask=np.ones((6, 14), dtype=bool))
    fingerprint = tmp_path / "fingerprint.json"
    fingerprint.write_text(json.dumps(build_split_fingerprint(
        processed_features_dir=features, processed_labels_dir=labels, manifest_paths={"hidden_val": manifest},
        require_no_missing=True, require_labels=True, track_id="limited")))
    output = tmp_path / "private/lock.json"
    return ["--hidden-manifest", str(manifest), "--hidden-features-dir", str(features),
            "--hidden-labels-dir", str(labels), "--hidden-fingerprint", str(fingerprint),
            "--lock-file", str(output)], output


def test_pin_validates_complete_assets_and_writes_actual_count(tmp_path: Path) -> None:
    argv, output = _fixture(tmp_path)
    pin.main([*argv, "--hidden-chain-count", "1"])
    lock = json.loads(output.read_text())
    assert lock["hidden_chain_count"] == 1
    assert lock["track"] == "limited"
    assert lock["hidden_manifest_sha256"] == pin._sha256(tmp_path / "hidden.txt")
    assert lock["hidden_features_fingerprint_sha256"] == pin._tree_sha256(tmp_path / "features")
    assert sorted(path.name for path in output.parent.iterdir()) == ["lock.json"]


@pytest.mark.parametrize("invalid", ["missing_feature", "missing_label", "corrupt_feature", "bad_fingerprint",
                                     "empty_manifest", "duplicate_manifest", "wrong_count", "wrong_track"])
def test_invalid_hidden_assets_never_replace_existing_lock(tmp_path: Path, invalid: str) -> None:
    argv, output = _fixture(tmp_path)
    output.parent.mkdir()
    output.write_text("previous validated lock\n")
    if invalid == "missing_feature":
        chain_npz_path(tmp_path / "features", "1abc_A").unlink()
    elif invalid == "missing_label":
        chain_npz_path(tmp_path / "labels", "1abc_A").unlink()
    elif invalid == "corrupt_feature":
        chain_npz_path(tmp_path / "features", "1abc_A").write_bytes(b"corrupt")
    elif invalid == "bad_fingerprint":
        (tmp_path / "fingerprint.json").write_text("{}")
    elif invalid == "empty_manifest":
        (tmp_path / "hidden.txt").write_text("")
    elif invalid == "duplicate_manifest":
        (tmp_path / "hidden.txt").write_text("1abc_A\n1abc_A\n")
    elif invalid == "wrong_count":
        argv.extend(["--hidden-chain-count", "99"])
    else:
        argv.extend(["--track-id", "unlimited"])
    with pytest.raises((ValueError, FileNotFoundError)):
        pin.main(argv)
    assert output.read_text() == "previous validated lock\n"


@pytest.mark.parametrize("destination", ["hidden.txt", "fingerprint.json", "features/lock.json", "labels/lock.json"])
def test_pin_output_cannot_overwrite_or_change_pinned_inputs(tmp_path: Path, destination: str) -> None:
    argv, _ = _fixture(tmp_path)
    before = {path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    with pytest.raises(ValueError, match="alias|outside"):
        pin.main([*argv, "--lock-file", str(tmp_path / destination)])
    assert {path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()} == before
