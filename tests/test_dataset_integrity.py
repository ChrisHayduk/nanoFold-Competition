from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from nanofold.dataset_integrity import (
    build_dataset_fingerprint,
    validate_npz_schema,
    verify_dataset_against_fingerprint,
)


def _write_npz(path: Path, *, bad_dtype: bool = False) -> None:
    np.savez_compressed(
        path,
        aatype=np.zeros((6,), dtype=np.float32 if bad_dtype else np.int32),
        msa=np.zeros((4, 6), dtype=np.int32),
        deletions=np.zeros((4, 6), dtype=np.int32),
        ca_coords=np.zeros((6, 3), dtype=np.float32),
        ca_mask=np.ones((6,), dtype=bool),
        template_aatype=np.zeros((1, 6), dtype=np.int32),
        template_ca_coords=np.zeros((1, 6, 3), dtype=np.float32),
        template_ca_mask=np.ones((1, 6), dtype=bool),
    )


def test_validate_npz_schema_flags_bad_dtype(tmp_path: Path) -> None:
    bad = tmp_path / "bad.npz"
    _write_npz(bad, bad_dtype=True)
    errors = validate_npz_schema(bad)
    assert any("aatype" in e for e in errors)


def test_build_fingerprint_requires_present_files_when_requested(tmp_path: Path) -> None:
    manifests = tmp_path / "manifests"
    manifests.mkdir()
    processed = tmp_path / "processed"
    processed.mkdir()

    (manifests / "train.txt").write_text("1abc_A\n")
    (manifests / "val.txt").write_text("2def_B\n")
    _write_npz(processed / "1abc_A.npz")
    # 2def_B intentionally missing

    with pytest.raises(FileNotFoundError):
        build_dataset_fingerprint(
            processed_dir=processed,
            train_manifest=manifests / "train.txt",
            val_manifest=manifests / "val.txt",
            require_no_missing=True,
        )


def test_verify_dataset_against_fingerprint_roundtrip(tmp_path: Path) -> None:
    manifests = tmp_path / "manifests"
    manifests.mkdir()
    processed = tmp_path / "processed"
    processed.mkdir()

    (manifests / "train.txt").write_text("1abc_A\n")
    (manifests / "val.txt").write_text("2def_B\n")
    _write_npz(processed / "1abc_A.npz")
    _write_npz(processed / "2def_B.npz")

    fingerprint = build_dataset_fingerprint(
        processed_dir=processed,
        train_manifest=manifests / "train.txt",
        val_manifest=manifests / "val.txt",
        require_no_missing=True,
    )
    fp_path = tmp_path / "fp.json"
    fp_path.write_text(json.dumps(fingerprint))

    out = verify_dataset_against_fingerprint(
        processed_dir=processed,
        train_manifest=manifests / "train.txt",
        val_manifest=manifests / "val.txt",
        expected_fingerprint_path=fp_path,
        require_no_missing=True,
    )
    assert out["missing_chain_count"] == 0
