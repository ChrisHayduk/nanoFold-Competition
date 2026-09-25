from __future__ import annotations

import os
import shlex
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
    filter_root = ".nanofold_private/asset_rebuild/msa_row_filters"
    assert "--skip-preprocess" in stdout
    filters = [shlex.split(line)[1:] for line in stdout.splitlines() if "/build_msa_row_filter.py" in line]
    assert len(filters) == 2
    for command, source, heldout, checkpoint in (
        (filters[0], "train", ["data/manifests/val.txt", ".nanofold_private/manifests/hidden_val.txt"], "train"),
        (filters[1], "val", [".nanofold_private/manifests/hidden_val.txt"], "val"),
    ):
        assert [command[i + 1] for i, part in enumerate(command) if part == "--source-manifest"] == [
            f"data/manifests/{source}.txt"
        ]
        assert [command[i + 1] for i, part in enumerate(command) if part == "--heldout-manifest"] == heldout
        assert "--exhaustive" in command
        assert command[command.index("--min-seq-id") + 1] == "0.30"
        assert command[command.index("--coverage") + 1] == "0.80"
        assert command[command.index("--checkpoint-dir") + 1] == f"{filter_root}/checkpoints/{checkpoint}"
    assert f"--output {filter_root}/train_filter.json" in stdout
    assert f"--output {filter_root}/val_filter.json" in stdout
    policy_commands = [line for line in stdout.splitlines() if "/build_msa_filter_policy.py" in line]
    assert len(policy_commands) == 1
    assert f"--train-msa-row-filter {filter_root}/train_filter.json" in policy_commands[0]
    assert f"--val-msa-row-filter {filter_root}/val_filter.json" in policy_commands[0]
    preprocessing = [line for line in stdout.splitlines() if "/preprocess.py" in line]
    assert len(preprocessing) == 1
    assert f"--msa-filter-policy {filter_root}/public_directed_policy.json" in preprocessing[0]
    assert "--msa-row-filter" not in preprocessing[0]
    assert preprocessing[0].endswith("--manifest data/manifests/all.txt")


def test_full_refresh_runs_both_strict_msa_gates_before_fingerprints() -> None:
    proc = subprocess.run(
        ["bash", "scripts/full_official_data_refresh.sh", "--dry-run", "--skip-manifest-regen"],
        cwd=REPO_ROOT, capture_output=True, text=True, check=True,
    )
    stdout = proc.stdout
    assert stdout.count("--strict --run-mmseqs --exhaustive --skip-plot") == 2
    assert stdout.count("--min-seq-id 0.30 --coverage 0.80") == 4
    assert "/checkpoints/residual_train_val" in stdout
    assert "/checkpoints/residual_public_hidden" in stdout
    assert "--val-features-dir .nanofold_private/hidden_processed_features" in stdout
    assert stdout.index("audits/hidden_msa_split") < stdout.index("scripts/build_fingerprint.py")
    preprocessing = [line for line in stdout.splitlines() if "/preprocess.py" in line]
    assert len(preprocessing) == 2
    assert "--processed-features-dir data/processed_features" in preprocessing[0]
    assert preprocessing[0].endswith("--manifest data/manifests/all.txt")
    assert "--processed-features-dir .nanofold_private/hidden_processed_features" in preprocessing[1]
    assert "--manifest .nanofold_private/manifests/hidden_val.txt" in preprocessing[1]
    assert "--msa-filter-policy" not in preprocessing[1]
    assert "--msa-row-filter" not in preprocessing[1]
    assert stdout.index("scripts/sync_official_manifest_hashes.py --check") < stdout.index(preprocessing[0])


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
    assert commands.count("/preprocess.py") == 2
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
        assert "frozen" in proc.stderr
        assert "regeneration" in proc.stderr
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


@pytest.mark.parametrize("source_option,output_option", [
    ("--chain-data-cache", "--data-source-lock"),
    ("--train-msa-row-filter", "--fingerprint-out"),
    ("--val-msa-row-filter", "--msa-filter-policy-out"),
])
@pytest.mark.parametrize("alias", ["direct", "symlink", "hardlink"])
def test_refresh_rejects_source_input_alias_before_writes(
    tmp_path: Path, source_option: str, output_option: str, alias: str,
) -> None:
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    refresh = scripts / "full_official_data_refresh.sh"
    shutil.copyfile(REPO_ROOT / "scripts/full_official_data_refresh.sh", refresh)
    source = tmp_path / "pinned input.json"
    source.write_text('{"source": "preserve these bytes"}\n')
    original = source.read_bytes()
    output = source
    if alias != "direct":
        output = tmp_path / "candidate.json"
        if alias == "symlink":
            output.symlink_to(source)
        else:
            output.hardlink_to(source)
    extra = []
    if source_option in {"--train-msa-row-filter", "--val-msa-row-filter"}:
        counterpart = tmp_path / "other filter.json"
        counterpart.write_text("{}\n")
        other = "--val-msa-row-filter" if source_option == "--train-msa-row-filter" else "--train-msa-row-filter"
        extra = [other, str(counterpart)]
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    (fake_bin / "python").symlink_to(sys.executable)
    before = set(tmp_path.rglob("*"))
    proc = subprocess.run(
        ["bash", str(refresh), "--skip-manifest-regen", source_option, str(source), output_option, str(output), *extra],
        env=dict(os.environ, PATH=f"{fake_bin}:{os.environ['PATH']}"),
        capture_output=True, text=True, check=False,
    )
    assert proc.returncode != 0
    assert "frozen dataset contract" in proc.stderr
    assert "[1/5]" not in proc.stdout
    assert source.read_bytes() == original
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


@pytest.mark.parametrize("cache_mode", ["default", "derived", "explicit"])
def test_refresh_propagates_same_chain_cache_to_all_consumers(tmp_path: Path, cache_mode: str) -> None:
    args: list[str] = []
    raw = Path("data/openproteinset")
    if cache_mode != "default":
        raw = tmp_path / "custom raw"
        args.extend(["--data-root", str(raw)])
    cache = raw / "pdb_data/data_caches/chain_data_cache.json"
    if cache_mode == "explicit":
        cache = tmp_path / "pinned cache.json"
        args.extend(["--chain-data-cache", str(cache)])
    result = subprocess.run(["bash", "scripts/full_official_data_refresh.sh", "--dry-run",
                             "--skip-manifest-regen", *args],
                            cwd=REPO_ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    consumers = ["setup_official_data.sh", "build_msa_row_filter.py", "preprocess.py", "build_data_source_lock.py"]
    commands = [line for line in result.stdout.splitlines()
                if line.startswith("+ ") and any(f"scripts/{name}" in line for name in consumers)]
    assert len(commands) == 6  # Setup, two row filters, public/hidden preprocessing, source lock.
    assert all(f"--chain-data-cache {cache}" in command for command in commands)
    assert not list(tmp_path.iterdir())


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


@pytest.mark.parametrize("args,message", [
    (["--msa-row-filter", "global.json"], "Global MSA filters"),
    (["--msa-row-filter-out", "global.json"], "Global MSA filters"),
    (["--train-msa-row-filter", "train.json"], "Supply both"),
    (["--val-msa-row-filter", "val.json"], "Supply both"),
    (["--skip-msa-row-filter-build"], "requires existing filters"),
    (["--skip-setup"], "use --resume-preprocess for validated reuse"),
])
def test_refresh_rejects_incomplete_or_global_filter_policy_before_work(args: list[str], message: str) -> None:
    result = subprocess.run(
        ["bash", "scripts/full_official_data_refresh.sh", "--dry-run", "--skip-manifest-regen", *args],
        cwd=REPO_ROOT, capture_output=True, text=True,
    )
    assert result.returncode != 0
    assert message in result.stderr
    assert "[1/5]" not in result.stdout


def test_refresh_reuses_two_filters_but_rebuilds_and_validates_their_policy(tmp_path: Path) -> None:
    train, val = tmp_path / "train.json", tmp_path / "val.json"
    train.write_text('{"preserve": "train"}\n')
    val.write_text('{"preserve": "val"}\n')
    policy = tmp_path / "candidate" / "policy.json"
    checkpoint = tmp_path / "checkpoints"
    result = subprocess.run(
        ["bash", "scripts/full_official_data_refresh.sh", "--dry-run", "--skip-manifest-regen",
         "--train-msa-row-filter", str(train), "--val-msa-row-filter", str(val),
         "--skip-msa-row-filter-build", "--msa-filter-policy-out", str(policy),
         "--msa-filter-checkpoint-dir", str(checkpoint)],
        cwd=REPO_ROOT, capture_output=True, text=True, check=True,
    )
    assert "/build_msa_row_filter.py" not in result.stdout
    policy_commands = [line for line in result.stdout.splitlines() if "/build_msa_filter_policy.py" in line]
    assert len(policy_commands) == 1
    assert f"--train-msa-row-filter {train}" in policy_commands[0]
    assert f"--val-msa-row-filter {val}" in policy_commands[0]
    assert f"--output {policy}" in policy_commands[0]
    assert f"--checkpoint-dir {checkpoint}/residual_train_val" in result.stdout
    assert f"--checkpoint-dir {checkpoint}/residual_public_hidden" in result.stdout
    assert train.read_text() == '{"preserve": "train"}\n'
    assert val.read_text() == '{"preserve": "val"}\n'
    assert not policy.parent.exists()
    assert not checkpoint.exists()


@pytest.mark.parametrize("collision", ["raw_policy", "raw_checkpoint", "policy_checkpoint", "filter_policy"])
def test_refresh_rejects_filter_output_and_checkpoint_collisions_before_work(tmp_path: Path, collision: str) -> None:
    raw = tmp_path / "raw"
    policy = tmp_path / "filters" / "policy.json"
    checkpoint = tmp_path / "checkpoints"
    if collision == "raw_policy":
        policy = raw / "policy.json"
    elif collision == "raw_checkpoint":
        checkpoint = raw / "checkpoints"
    elif collision == "policy_checkpoint":
        checkpoint = policy.parent
    else:
        policy = policy.parent / "train_filter.json"
    result = subprocess.run(
        ["bash", "scripts/full_official_data_refresh.sh", "--dry-run", "--skip-manifest-regen",
         "--data-root", str(raw), "--msa-filter-policy-out", str(policy),
         "--msa-filter-checkpoint-dir", str(checkpoint)],
        cwd=REPO_ROOT, capture_output=True, text=True,
    )
    assert result.returncode != 0
    assert "must be separate" in result.stderr or "must be distinct" in result.stderr
    assert "[1/5]" not in result.stdout
    assert not list(tmp_path.iterdir())
