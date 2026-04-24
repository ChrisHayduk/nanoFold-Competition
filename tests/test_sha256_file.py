"""Confirm the single shared sha256_file is used by every consumer."""

from __future__ import annotations

import hashlib
from pathlib import Path

import nanofold.competition_policy as competition_policy
import nanofold.dataset_integrity as dataset_integrity
import nanofold.submission_runtime as submission_runtime
from nanofold.utils import sha256_file


def test_all_modules_reference_the_same_helper() -> None:
    assert dataset_integrity.sha256_file is sha256_file
    assert submission_runtime.sha256_file is sha256_file
    assert competition_policy.sha256_file is sha256_file


def test_sha256_file_matches_hashlib(tmp_path: Path) -> None:
    payload = b"the competition fingerprint keeps things honest\n"
    p = tmp_path / "sample.bin"
    p.write_bytes(payload)

    expected = hashlib.sha256(payload).hexdigest()
    assert sha256_file(p) == expected


def test_sha256_file_streams_large_files(tmp_path: Path) -> None:
    chunk = b"abcdef\n" * 10000
    p = tmp_path / "big.bin"
    p.write_bytes(chunk * 4)
    expected = hashlib.sha256(chunk * 4).hexdigest()
    assert sha256_file(p, chunk_size=128) == expected
