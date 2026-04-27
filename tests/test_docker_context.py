from __future__ import annotations

import os
import subprocess
from pathlib import Path


def _dockerignore_patterns() -> set[str]:
    return {
        line.strip()
        for line in Path(".dockerignore").read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }


def test_official_docker_context_excludes_generated_data_and_private_assets() -> None:
    patterns = _dockerignore_patterns()
    required_patterns = {
        ".nanofold_private/",
        ".venv/",
        ".conda/",
        "data/openproteinset/",
        "data/processed_features/",
        "data/processed_labels/",
        "data/hidden_processed_features/",
        "data/hidden_processed_labels/",
        "runs/",
        "artifacts/",
        "*.npz",
        "*.pt",
        "*.ckpt",
    }
    assert required_patterns <= patterns


def test_official_dockerfile_installs_from_requirements_before_copying_source() -> None:
    dockerfile = Path("Dockerfile.official").read_text()
    assert "COPY requirements.txt requirements-dev.txt ./" in dockerfile
    assert "RUN pip install --no-cache-dir -r requirements.txt -r requirements-dev.txt" in dockerfile
    assert "RUN mkdir -p /nanofold_hidden" in dockerfile
    assert "COPY . ." in dockerfile


def test_official_docker_script_is_compatible_with_macos_bash() -> None:
    script = Path("scripts/run_official_docker.sh").read_text()
    assert "local -n" not in script
    assert "declare -n" not in script


def test_official_docker_script_runs_with_bash3_and_fake_docker(tmp_path: Path) -> None:
    docker_bin = tmp_path / "docker"
    docker_log = tmp_path / "docker.log"
    docker_bin.write_text(
        "#!/usr/bin/env bash\n"
        "printf '%s\\n' \"$*\" >> \"$DOCKER_LOG\"\n"
    )
    docker_bin.chmod(0o755)

    env = dict(os.environ)
    env["PATH"] = f"{tmp_path}:{env['PATH']}"
    env["DOCKER_LOG"] = str(docker_log)
    env["IMAGE_NAME"] = "nanofold-test-runner"

    proc = subprocess.run(
        [
            "bash",
            "scripts/run_official_docker.sh",
            "--disable-hidden",
            "--submission",
            "submissions/example",
            "--track",
            "limited",
        ],
        text=True,
        capture_output=True,
        env=env,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr + proc.stdout
    log = docker_log.read_text()
    assert "build -f" in log
    assert "run --rm --network=none --cap-drop=ALL" in log
    assert "--disable-hidden --submission submissions/example --track limited --skip-hidden-scoring" in log


def test_official_docker_script_mounts_hidden_assets_with_bash3_and_fake_docker(tmp_path: Path) -> None:
    private_root = tmp_path / "private"
    (private_root / "manifests").mkdir(parents=True)
    (private_root / "hidden_processed_features").mkdir()
    (private_root / "hidden_processed_labels").mkdir()
    (private_root / "leaderboard").mkdir()
    (private_root / "manifests/hidden_val.txt").write_text("1abc_A\n")
    (private_root / "leaderboard/official_hidden_fingerprint.json").write_text("{}\n")
    (private_root / "leaderboard/private_hidden_assets.lock.json").write_text("{}\n")

    docker_bin = tmp_path / "docker"
    docker_log = tmp_path / "docker.log"
    docker_bin.write_text(
        "#!/usr/bin/env bash\n"
        "printf '%s\\n' \"$*\" >> \"$DOCKER_LOG\"\n"
    )
    docker_bin.chmod(0o755)

    env = dict(os.environ)
    env["PATH"] = f"{tmp_path}:{env['PATH']}"
    env["DOCKER_LOG"] = str(docker_log)
    env["IMAGE_NAME"] = "nanofold-test-runner"
    env["NANOFOLD_PRIVATE_ROOT"] = str(private_root)

    proc = subprocess.run(
        [
            "bash",
            "scripts/run_official_docker.sh",
            "--submission",
            "submissions/example",
            "--track",
            "limited",
        ],
        text=True,
        capture_output=True,
        env=env,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr + proc.stdout
    lines = docker_log.read_text().splitlines()
    run_lines = [line for line in lines if line.startswith("run ")]
    assert len(run_lines) == 2
    assert "/nanofold_hidden/hidden_val.txt:ro" in run_lines[0]
    assert "/nanofold_hidden/official_hidden_fingerprint.json:ro" in run_lines[0]
    assert "/nanofold_hidden/features:ro" in run_lines[0]
    assert "/nanofold_hidden/labels:ro" not in run_lines[0]
    assert "/nanofold_hidden/private_hidden_assets.lock.json:ro" not in run_lines[0]
    assert "/nanofold_hidden/labels:ro" in run_lines[1]
    assert "/nanofold_hidden/private_hidden_assets.lock.json:ro" in run_lines[1]
    assert "/workspace/.nanofold_hidden" not in "\n".join(run_lines)
    assert "--score-hidden-only --skip-train" in run_lines[1]
