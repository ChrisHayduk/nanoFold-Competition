from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import yaml


def _write_public_repo_fixture(root: Path) -> Path:
    (root / "tracks").mkdir()
    (root / "leaderboard").mkdir()
    track = {
        "id": "limited",
        "official": True,
        "dataset": {
            "hidden_manifest": None,
            "hidden_manifest_sha256": None,
            "hidden_fingerprint": None,
            "hidden_fingerprint_sha256": None,
            "hidden_lock_file": None,
        },
    }
    (root / "tracks/limited.yaml").write_text(yaml.safe_dump(track, sort_keys=False))
    (root / "leaderboard/official_hidden_assets.lock.json").write_text(
        json.dumps(
            {
                "hidden_manifest_sha256": None,
                "hidden_features_fingerprint_sha256": None,
                "hidden_labels_fingerprint_sha256": None,
                "hidden_fingerprint_sha256": None,
            }
        )
    )
    (root / "leaderboard/official_manifest_source.lock.json").write_text(
        json.dumps({"args": {"hidden_val_size": 1000}, "outputs": {"train_manifest": "data/manifests/train.txt"}})
    )
    tracked = root / "tracked.txt"
    tracked.write_text(
        "\n".join(
            [
                "tracks/limited.yaml",
                "leaderboard/official_hidden_assets.lock.json",
                "leaderboard/official_manifest_source.lock.json",
            ]
        )
        + "\n"
    )
    return tracked


def _run_check(root: Path, tracked: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            "scripts/check_public_release_leaks.py",
            "--root",
            str(root),
            "--tracked-files-from",
            str(tracked),
        ],
        text=True,
        capture_output=True,
        check=False,
    )


def test_public_release_leak_check_accepts_sanitized_public_metadata(tmp_path: Path) -> None:
    tracked = _write_public_repo_fixture(tmp_path)
    proc = _run_check(tmp_path, tracked)
    assert proc.returncode == 0, proc.stderr + proc.stdout


def test_public_release_leak_check_rejects_hidden_hashes_and_tracked_hidden_manifest(tmp_path: Path) -> None:
    tracked = _write_public_repo_fixture(tmp_path)
    track_path = tmp_path / "tracks/limited.yaml"
    raw = yaml.safe_load(track_path.read_text())
    raw["dataset"]["hidden_manifest_sha256"] = "a" * 64
    track_path.write_text(yaml.safe_dump(raw, sort_keys=False))
    (tmp_path / "data/manifests").mkdir(parents=True)
    (tmp_path / "data/manifests/hidden_val.txt").write_text("1abc_A\n")
    tracked.write_text(tracked.read_text() + "data/manifests/hidden_val.txt\n")

    proc = _run_check(tmp_path, tracked)
    assert proc.returncode != 0
    output = proc.stderr + proc.stdout
    assert "dataset.hidden_manifest_sha256" in output
    assert "data/manifests/hidden_val.txt" in output


def test_public_release_leak_check_rejects_private_workspace_files(tmp_path: Path) -> None:
    tracked = _write_public_repo_fixture(tmp_path)
    private_manifest = tmp_path / ".nanofold_private/manifests/hidden_val.txt"
    private_manifest.parent.mkdir(parents=True)
    private_manifest.write_text("1abc_A\n")
    tracked.write_text(tracked.read_text() + ".nanofold_private/manifests/hidden_val.txt\n")

    proc = _run_check(tmp_path, tracked)
    assert proc.returncode != 0
    assert ".nanofold_private/manifests/hidden_val.txt" in (proc.stderr + proc.stdout)
