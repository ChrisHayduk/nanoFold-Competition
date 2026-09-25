from __future__ import annotations

import hashlib
import json
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
    preprocessing = [line for line in result.stdout.splitlines() if "/preprocess.py" in line]
    assert len(preprocessing) == 1
    assert f"--manifest {root / 'manifests/all.txt'}" in preprocessing[0]
    assert all(f"--chain-data-cache {root / 'raw/pdb_data/data_caches/chain_data_cache.json'}" in line
               for line in preprocessing)
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


def _locked_checkout(tmp_path: Path) -> tuple[Path, Path]:
    checkout = tmp_path / "checkout"
    script = checkout / "scripts/setup_official_data.sh"
    script.parent.mkdir(parents=True)
    shutil.copyfile(SCRIPT, script)
    manifests = checkout / "data/manifests"
    manifests.mkdir(parents=True)
    for split in ("train", "val", "all"):
        shutil.copyfile(REPO_ROOT / f"data/manifests/{split}.txt", manifests / f"{split}.txt")
    cache = tmp_path / "pinned cache.json"
    cache.write_text('{"1abc_A": {"sequence": "ACD"}}\n')
    lock = checkout / "leaderboard/official_manifest_source.lock.json"
    lock.parent.mkdir()
    lock.write_text(json.dumps({"chain_data_cache_sha256": hashlib.sha256(cache.read_bytes()).hexdigest()}))
    return script, cache


@pytest.mark.parametrize("broken", ["changed", "missing"])
def test_explicit_official_cache_is_checked_before_writes(tmp_path: Path, broken: str) -> None:
    script, cache = _locked_checkout(tmp_path)
    if broken == "changed":
        cache.write_text("{}")
    else:
        cache.unlink()
    root = tmp_path / "output"
    result = subprocess.run(["bash", str(script), *_args(root), "--chain-data-cache", str(cache), "--dry-run"],
                            cwd=REPO_ROOT, env=_env(), capture_output=True, text=True)
    assert result.returncode != 0
    assert "cache" in result.stderr
    assert not root.exists()


@pytest.mark.parametrize("location", ["raw", "features", "labels", "symlink"])
def test_official_cache_cannot_overlap_asset_downloads_or_outputs(tmp_path: Path, location: str) -> None:
    script, cache = _locked_checkout(tmp_path)
    root = tmp_path / "output"
    destination = root / ("raw/pdb_data/duplicate_pdb_chains.txt" if location in {"raw", "symlink"}
                          else f"{location}/cache.json")
    destination.parent.mkdir(parents=True)
    shutil.copyfile(cache, destination)
    selected = destination
    if location == "symlink":
        selected = tmp_path / "cache-alias.json"
        selected.symlink_to(destination)
    before = {path: path.read_bytes() for path in root.rglob("*") if path.is_file()}
    result = subprocess.run(["bash", str(script), *_args(root), "--chain-data-cache", str(selected), "--dry-run"],
                            cwd=REPO_ROOT, env=_env(), capture_output=True, text=True)
    assert result.returncode != 0
    assert "chain cache must be outside asset output directories" in result.stderr
    assert {path: path.read_bytes() for path in root.rglob("*") if path.is_file()} == before
    assert "[1/5]" not in result.stdout


@pytest.mark.parametrize("cache_mode", ["explicit", "download", "derived_explicit", "corrupt_download"])
def test_official_preprocessing_uses_verified_cache(tmp_path: Path, cache_mode: str) -> None:
    script, cache = _locked_checkout(tmp_path)
    root = tmp_path / "output"
    command_log = tmp_path / "commands.log"
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    python = fake_bin / "python"
    python.write_text(
        '#!/bin/bash\nif [[ "$1" == "-" || "$1" == "-c" ]]; then exec "$REAL_PYTHON" "$@"; fi\n'
        'printf \'%s\\n\' "$*" >> "$COMMAND_LOG"\n'
    )
    python.chmod(0o755)
    aws = fake_bin / "aws"
    aws.write_text(
        '#!/bin/bash\nprintf \'aws %s\\n\' "$*" >> "$COMMAND_LOG"\n'
        'if [[ "$3" == "s3://openfold/data_caches/" && "$*" != *--exclude* ]]; then\n'
        '  if [[ "$CORRUPT_CACHE_DOWNLOAD" == "1" ]]; then\n'
        '    printf \'{}\\n\' > "$4/chain_data_cache.json"\n'
        '  else cp "$PINNED_CACHE" "$4/chain_data_cache.json"; fi\nfi\n'
    )
    aws.chmod(0o755)
    env = dict(_env(), PATH=f"{fake_bin}:{_env()['PATH']}", REAL_PYTHON=sys.executable,
               COMMAND_LOG=str(command_log), PINNED_CACHE=str(cache),
               CORRUPT_CACHE_DOWNLOAD=str(int(cache_mode == "corrupt_download")))
    selected_cache = cache if cache_mode == "explicit" else root / "raw/pdb_data/data_caches/chain_data_cache.json"
    extra = ["--chain-data-cache", str(selected_cache)] if cache_mode in {"explicit", "derived_explicit"} else []
    result = subprocess.run(["bash", str(script), *_args(root), *extra], cwd=REPO_ROOT, env=env,
                            capture_output=True, text=True)
    commands = command_log.read_text().splitlines()
    if cache_mode == "corrupt_download":
        assert result.returncode != 0
        assert "does not match the official manifest source lock" in result.stderr
        assert not any("prepare_data.py" in line or "preprocess.py" in line for line in commands)
    else:
        assert result.returncode == 0, result.stderr
        preprocessing = [line for line in commands if "/preprocess.py" in line]
        assert len(preprocessing) == 1
        assert f"--manifest {root / 'manifests/all.txt'}" in preprocessing[0]
        assert all(f"--chain-data-cache {selected_cache}" in line for line in preprocessing)
        assert ("--exclude chain_data_cache.json" in commands[0]) == (cache_mode == "explicit")
        assert selected_cache.read_bytes() == cache.read_bytes()
        assert cache.read_text() == '{"1abc_A": {"sequence": "ACD"}}\n'
