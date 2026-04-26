from __future__ import annotations

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
    assert "COPY . ." in dockerfile
