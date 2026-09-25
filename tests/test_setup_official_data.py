from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts/setup_official_data.sh"


def _args(root: Path) -> list[str]:
    return ["--data-root", str(root / "raw"), "--manifests-dir", str(root / "manifests"),
            "--processed-features-dir", str(root / "features"), "--processed-labels-dir", str(root / "labels")]


def _env() -> dict[str, str]:
    return dict(os.environ, PATH=f"{Path(sys.executable).parent}:{os.environ['PATH']}", PYTHONPATH=str(REPO_ROOT))


def test_official_dry_run_validates_hashes_without_any_writes(tmp_path: Path) -> None:
    root = tmp_path / 'assets with "quotes" and spaces'
    result = subprocess.run(["bash", str(SCRIPT), *_args(root), "--dry-run"], cwd=REPO_ROOT,
                            env=_env(), capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "Verified official manifest hashes" in result.stdout
    assert "--strict-downloads" in result.stdout
    assert not root.exists()


@pytest.mark.parametrize("corruption", ["missing_union", "changed_train"])
def test_invalid_source_manifests_fail_before_copy_or_output_creation(tmp_path: Path, corruption: str) -> None:
    checkout = tmp_path / "checkout"
    script = checkout / "scripts/setup_official_data.sh"
    script.parent.mkdir(parents=True)
    shutil.copyfile(SCRIPT, script)
    manifests = checkout / "data/manifests"
    manifests.mkdir(parents=True)
    for split in ("train", "val", "all"):
        shutil.copyfile(REPO_ROOT / f"data/manifests/{split}.txt", manifests / f"{split}.txt")
    if corruption == "missing_union":
        (manifests / "all.txt").unlink()
    else:
        (manifests / "train.txt").write_text("changed\n")
    output = tmp_path / "output"
    result = subprocess.run(["bash", str(script), *_args(output), "--dry-run"], cwd=REPO_ROOT,
                            env=_env(), capture_output=True, text=True)
    assert result.returncode != 0
    assert "Missing committed" in result.stdout or "hash mismatch" in result.stderr
    assert not output.exists()


@pytest.mark.parametrize("overlap", ["labels_are_features", "features_under_raw", "features_alias_manifests"])
def test_official_setup_rejects_aliasing_output_directories(tmp_path: Path, overlap: str) -> None:
    args = _args(tmp_path / "output")
    if overlap == "labels_are_features":
        args.extend(["--processed-labels-dir", str(tmp_path / "output/features")])
    elif overlap == "features_under_raw":
        args.extend(["--processed-features-dir", str(tmp_path / "output/raw/features")])
    else:
        alias = tmp_path / "alias"
        alias.symlink_to(REPO_ROOT / "data/manifests", target_is_directory=True)
        args.extend(["--processed-features-dir", str(alias)])
    result = subprocess.run(["bash", str(SCRIPT), *args, "--dry-run"], cwd=REPO_ROOT,
                            env=_env(), capture_output=True, text=True)
    assert result.returncode != 0
    assert "must be separate" in result.stderr
    assert not (tmp_path / "output").exists()


def test_standalone_all_manifest_is_not_overwritten_without_force(tmp_path: Path) -> None:
    root = tmp_path / "output"
    manifest = root / "manifests/all.txt"
    manifest.parent.mkdir(parents=True)
    manifest.write_text("existing union\n")
    result = subprocess.run(["bash", str(SCRIPT), *_args(root), "--dry-run"], cwd=REPO_ROOT,
                            env=_env(), capture_output=True, text=True)
    assert result.returncode != 0
    assert "Refusing to overwrite" in result.stdout
    assert manifest.read_text() == "existing union\n"
    assert not (root / "raw").exists()
