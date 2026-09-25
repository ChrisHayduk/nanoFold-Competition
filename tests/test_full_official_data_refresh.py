from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_full_refresh_dry_run_builds_and_applies_msa_row_filter() -> None:
    proc = subprocess.run(
        [
            "bash",
            "scripts/full_official_data_refresh.sh",
            "--dry-run",
            "--skip-manifest-regen",
            "--skip-fingerprint",
            "--skip-hidden",
            "--download-workers",
            "2",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )

    stdout = proc.stdout
    filter_path = ".nanofold_private/asset_rebuild/msa_row_filters/official_heldout_target_homology_filter.json"
    assert "--skip-preprocess" in stdout
    assert "scripts/build_msa_row_filter.py" in stdout
    assert "--source-manifest data/manifests/train.txt" in stdout
    assert "--source-manifest data/manifests/val.txt" in stdout
    assert "--heldout-manifest data/manifests/val.txt" in stdout
    assert f"--output {filter_path}" in stdout
    assert f"--msa-row-filter {filter_path}" in stdout


def test_full_refresh_runs_both_strict_msa_gates_before_fingerprints() -> None:
    proc = subprocess.run(
        ["bash", "scripts/full_official_data_refresh.sh", "--dry-run", "--skip-manifest-regen"],
        cwd=REPO_ROOT, capture_output=True, text=True, check=True,
    )
    stdout = proc.stdout
    assert stdout.count("--strict --run-mmseqs --skip-plot") == 2
    assert "--val-features-dir .nanofold_private/hidden_processed_features" in stdout
    assert stdout.index("audits/hidden_msa_split") < stdout.index("scripts/build_fingerprint.py")


def test_hidden_preprocess_failure_does_not_rewrite_exclusions_or_retry(tmp_path: Path) -> None:
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    refresh = scripts / "full_official_data_refresh.sh"
    shutil.copyfile(REPO_ROOT / "scripts/full_official_data_refresh.sh", refresh)
    (scripts / "setup_official_data.sh").write_text("#!/bin/bash\nexit 0\n")
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    command_log = tmp_path / "commands.log"
    python = fake_bin / "python"
    python.write_text(
        "#!/bin/bash\nprintf '%s\\n' \"$*\" >> \"$TEST_COMMAND_LOG\"\n"
        "if [[ \"$1\" == */preprocess.py && \"$*\" == *hidden_val.txt* ]]; then exit 1; fi\nexit 0\n"
    )
    python.chmod(0o755)
    exclusions = tmp_path / "exclusions.txt"
    exclusions.write_text("public_A\n")
    env = dict(os.environ, PATH=f"{fake_bin}:{os.environ['PATH']}", TEST_COMMAND_LOG=str(command_log))
    proc = subprocess.run(
        ["bash", str(refresh), "--skip-manifest-regen", "--processability-exclusion-list", str(exclusions)],
        env=env, capture_output=True, text=True, check=False,
    )
    assert proc.returncode != 0
    assert exclusions.read_text() == "public_A\n"
    commands = command_log.read_text()
    assert "update_processability_exclusions.py" not in commands
    assert "build_fingerprint.py" not in commands
    assert commands.count("/preprocess.py") == 3
    assert "will not be changed" in proc.stdout


def test_official_wrappers_refuse_regeneration_before_any_file_writes(tmp_path: Path) -> None:
    sentinel = tmp_path / "source.lock.json"
    sentinel.write_text("frozen\n")
    for script, args in (
        ("full_official_data_refresh.sh", ["--dry-run", "--lock-file", str(sentinel)]),
        ("regenerate_official_manifests.sh", ["--rewrite-lock", "--lock-file", str(sentinel)]),
    ):
        proc = subprocess.run(["bash", str(REPO_ROOT / "scripts" / script), *args], capture_output=True, text=True)
        assert proc.returncode != 0
        assert "cath-v2" in proc.stderr
        assert sentinel.read_text() == "frozen\n"
        assert sorted(path.name for path in tmp_path.iterdir()) == ["source.lock.json"]


def test_fingerprints_and_asset_locks_default_to_candidates() -> None:
    proc = subprocess.run(
        ["bash", "scripts/full_official_data_refresh.sh", "--dry-run", "--skip-manifest-regen"],
        cwd=REPO_ROOT, capture_output=True, text=True, check=True,
    )
    fingerprint_commands = [line for line in proc.stdout.splitlines() if "build_fingerprint.py" in line]
    assert len(fingerprint_commands) == 4
    for command in fingerprint_commands:
        assert "--output .nanofold_private/asset_rebuild/leaderboard/" in command
    assert "--lock-file .nanofold_private/asset_rebuild/leaderboard/private_hidden_assets.lock.json" in proc.stdout
    assert "--output .nanofold_private/asset_rebuild/leaderboard/official_data_source.lock.json" in proc.stdout


def test_custom_candidate_fingerprint_stages_other_tracks_beside_it(tmp_path: Path) -> None:
    candidate = tmp_path / "candidate" / "limited.json"
    proc = subprocess.run(
        ["bash", "scripts/full_official_data_refresh.sh", "--dry-run", "--skip-manifest-regen",
         "--fingerprint-out", str(candidate)],
        cwd=REPO_ROOT, capture_output=True, text=True, check=True,
    )
    assert f"--output {candidate.parent}/research_large_dataset_fingerprint.json" in proc.stdout
    assert f"--output {candidate.parent}/unlimited_dataset_fingerprint.json" in proc.stdout
    assert not candidate.parent.exists()


@pytest.mark.parametrize("alias", ["direct", "symlink", "hardlink"])
def test_refresh_rejects_published_fingerprint_alias_before_writes(tmp_path: Path, alias: str) -> None:
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    refresh = scripts / "full_official_data_refresh.sh"
    shutil.copyfile(REPO_ROOT / "scripts/full_official_data_refresh.sh", refresh)
    published = tmp_path / "leaderboard" / "official_dataset_fingerprint.json"
    published.parent.mkdir()
    published.write_text("frozen fingerprint\n")
    output = published
    if alias != "direct":
        output = tmp_path / "candidate.json"
        if alias == "symlink":
            output.symlink_to(published)
        else:
            output.hardlink_to(published)
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    (fake_bin / "python").symlink_to(sys.executable)
    before = set(tmp_path.rglob("*"))
    proc = subprocess.run(
        ["bash", str(refresh), "--skip-manifest-regen", "--fingerprint-out", str(output)],
        env=dict(os.environ, PATH=f"{fake_bin}:{os.environ['PATH']}"),
        capture_output=True, text=True, check=False,
    )
    assert proc.returncode != 0
    assert "frozen dataset contract" in proc.stderr
    assert "Download public split" not in proc.stdout
    assert published.read_text() == "frozen fingerprint\n"
    assert set(tmp_path.rglob("*")) == before


def test_refresh_rejects_existing_hidden_reference_before_writes(tmp_path: Path) -> None:
    sealed = tmp_path / "leaderboard" / "private_hidden_assets.lock.json"
    sealed.parent.mkdir()
    sealed.write_text("sealed assets\n")
    proc = subprocess.run(
        ["bash", "scripts/full_official_data_refresh.sh", "--skip-manifest-regen", "--private-root", str(tmp_path),
         "--hidden-lock-file", str(sealed)],
        cwd=REPO_ROOT, capture_output=True, text=True, check=False,
    )
    assert proc.returncode != 0
    assert "frozen dataset contract" in proc.stderr
    assert sealed.read_text() == "sealed assets\n"
    assert not (tmp_path / "asset_rebuild").exists()


def test_refresh_rejects_colliding_candidate_fingerprints_before_writes(tmp_path: Path) -> None:
    proc = subprocess.run(
        ["bash", "scripts/full_official_data_refresh.sh", "--skip-manifest-regen",
         "--fingerprint-out", str(tmp_path / "research_large_dataset_fingerprint.json")],
        cwd=REPO_ROOT, capture_output=True, text=True, check=False,
    )
    assert proc.returncode != 0
    assert "Candidate output paths must be distinct" in proc.stderr
    assert not list(tmp_path.iterdir())


def test_custom_raw_root_propagates_to_default_chain_cache(tmp_path: Path) -> None:
    raw = tmp_path / "custom-raw"
    result = subprocess.run(["bash", "scripts/full_official_data_refresh.sh", "--dry-run",
                             "--skip-manifest-regen", "--data-root", str(raw)],
                            cwd=REPO_ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    cache = raw / "pdb_data/data_caches/chain_data_cache.json"
    commands = [line for line in result.stdout.splitlines() if "--chain-data-cache" in line]
    assert len(commands) == 2
    assert all(f"--chain-data-cache {cache}" in command for command in commands)
    assert not raw.exists()


@pytest.mark.parametrize("overlap", ["public_hidden_features", "hidden_features_labels", "features_under_raw"])
def test_refresh_rejects_overlapping_asset_roots_before_work(tmp_path: Path, overlap: str) -> None:
    root = tmp_path / "assets"
    args = ["--data-root", str(root / "raw"), "--processed-features-dir", str(root / "public_features"),
            "--processed-labels-dir", str(root / "public_labels"),
            "--hidden-features-dir", str(root / "hidden_features"), "--hidden-labels-dir", str(root / "hidden_labels")]
    if overlap == "public_hidden_features":
        args.extend(["--hidden-features-dir", str(root / "public_features")])
    elif overlap == "hidden_features_labels":
        args.extend(["--hidden-labels-dir", str(root / "hidden_features")])
    else:
        args.extend(["--processed-features-dir", str(root / "raw/features")])
    result = subprocess.run(["bash", "scripts/full_official_data_refresh.sh", "--dry-run",
                             "--skip-manifest-regen", *args], cwd=REPO_ROOT, capture_output=True, text=True)
    assert result.returncode != 0
    assert "must be separate" in result.stderr
    assert "[1/5]" not in result.stdout
    assert not root.exists()
