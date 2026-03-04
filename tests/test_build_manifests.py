from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path


def _cache_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _make_chain_cache(path: Path) -> None:
    # Keep insertion order stable and include enough chains for deterministic sampling.
    payload = {
        "1aaa_A": {"seq_length": 100, "resolution": 2.0, "oligomeric_count": 1},
        "1aaa_B": {"seq_length": 120, "resolution": 2.2, "oligomeric_count": 1},
        "2bbb_A": {"seq_length": 140, "resolution": 2.0, "oligomeric_count": 1},
        "3ccc_A": {"seq_length": 160, "resolution": 1.8, "oligomeric_count": 1},
        "4ddd_A": {"seq_length": 200, "resolution": 2.7, "oligomeric_count": 1},
        "5eee_A": {"seq_length": 220, "resolution": 2.9, "oligomeric_count": 1},
        "6fff_A": {"seq_length": 240, "resolution": 2.5, "oligomeric_count": 1},
    }
    path.write_text(json.dumps(payload))


def _run_build(cache_path: Path, out_dir: Path, expected_sha: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            "scripts/build_manifests.py",
            "--chain-data-cache",
            str(cache_path),
            "--out-dir",
            str(out_dir),
            "--train-size",
            "4",
            "--val-size",
            "2",
            "--seed",
            "0",
            "--expected-chain-cache-sha256",
            expected_sha,
        ],
        text=True,
        capture_output=True,
        check=False,
    )


def test_build_manifests_rejects_wrong_cache_sha(tmp_path: Path) -> None:
    cache_path = tmp_path / "chain_data_cache.json"
    _make_chain_cache(cache_path)

    out_dir = tmp_path / "out_bad"
    proc = _run_build(cache_path, out_dir, "0" * 64)
    assert proc.returncode != 0
    assert "SHA256 mismatch" in (proc.stderr + proc.stdout)


def test_build_manifests_deterministic_with_expected_cache_sha(tmp_path: Path) -> None:
    cache_path = tmp_path / "chain_data_cache.json"
    _make_chain_cache(cache_path)
    expected_sha = _cache_sha(cache_path)

    out_a = tmp_path / "out_a"
    out_b = tmp_path / "out_b"
    proc_a = _run_build(cache_path, out_a, expected_sha)
    proc_b = _run_build(cache_path, out_b, expected_sha)
    assert proc_a.returncode == 0, proc_a.stderr
    assert proc_b.returncode == 0, proc_b.stderr

    train_a = (out_a / "train.txt").read_text()
    val_a = (out_a / "val.txt").read_text()
    all_a = (out_a / "all.txt").read_text()

    train_b = (out_b / "train.txt").read_text()
    val_b = (out_b / "val.txt").read_text()
    all_b = (out_b / "all.txt").read_text()

    assert train_a == train_b
    assert val_a == val_b
    assert all_a == all_b
