from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

import yaml

FORBIDDEN_TRACKED_EXACT = {
    "data/manifests/hidden_val.txt",
    "data/manifests/split_quality_report.json",
    "data/manifests/structure_candidates.txt",
    "data/manifests/structure_metadata.json",
    "leaderboard/official_data_source.lock.json",
    "leaderboard/official_hidden_fingerprint.json",
    "leaderboard/private_hidden_assets.lock.json",
    "leaderboard/private_hidden_manifest_source.lock.json",
}

FORBIDDEN_TRACKED_PREFIXES = (
    "data/openproteinset/",
    "data/processed_features/",
    "data/processed_labels/",
    "data/hidden_processed_features/",
    "data/hidden_processed_labels/",
    "data/metadata_sources/",
    "runs/",
    ".venv/",
)

PUBLIC_TRACK_HIDDEN_FIELDS = (
    "hidden_manifest",
    "hidden_manifest_sha256",
    "hidden_fingerprint",
    "hidden_fingerprint_sha256",
    "hidden_lock_file",
)

ABSOLUTE_LOCAL_PATH_MARKERS = (
    "/Users/",
    "/var/folders/",
    "/private/var/folders/",
)

TEXT_SUFFIXES = {
    "",
    ".cfg",
    ".ini",
    ".json",
    ".lock",
    ".md",
    ".py",
    ".sh",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fail if public tracked files expose hidden validation assets or local private paths."
    )
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument(
        "--tracked-files-from",
        type=Path,
        default=None,
        help="Optional newline-delimited tracked-file list for tests.",
    )
    return parser.parse_args()


def _git_tracked_files(root: Path) -> list[str]:
    proc = subprocess.run(
        ["git", "ls-files"],
        cwd=root,
        text=True,
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or "git ls-files failed")
    return [line.strip() for line in proc.stdout.splitlines() if line.strip()]


def _tracked_files(root: Path, tracked_files_from: Path | None) -> list[str]:
    if tracked_files_from is None:
        return _git_tracked_files(root)
    return [line.strip() for line in tracked_files_from.read_text().splitlines() if line.strip()]


def _is_forbidden_tracked_path(path: str) -> bool:
    if path in FORBIDDEN_TRACKED_EXACT:
        return True
    return any(path.startswith(prefix) for prefix in FORBIDDEN_TRACKED_PREFIXES)


def _is_null_like(value: Any) -> bool:
    return value is None or value == ""


def _load_yaml(path: Path) -> Any:
    return yaml.safe_load(path.read_text())


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def _check_track_file(path: Path, rel: str, errors: list[str]) -> None:
    raw = _load_yaml(path)
    if not isinstance(raw, dict):
        errors.append(f"{rel}: track file must be a mapping.")
        return
    if not bool(raw.get("official", False)):
        return
    dataset = raw.get("dataset")
    if not isinstance(dataset, dict):
        errors.append(f"{rel}: official track is missing dataset mapping.")
        return
    for field in PUBLIC_TRACK_HIDDEN_FIELDS:
        value = dataset.get(field)
        if not _is_null_like(value):
            errors.append(f"{rel}: dataset.{field} must be null in the public repo.")


def _check_hidden_placeholder_lock(path: Path, rel: str, errors: list[str]) -> None:
    raw = _load_json(path)
    if not isinstance(raw, dict):
        errors.append(f"{rel}: hidden asset placeholder must be a JSON object.")
        return
    for key in (
        "hidden_manifest_sha256",
        "hidden_features_fingerprint_sha256",
        "hidden_labels_fingerprint_sha256",
        "hidden_fingerprint_sha256",
    ):
        if not _is_null_like(raw.get(key)):
            errors.append(f"{rel}: {key} must be null in the public placeholder.")


def _walk_hidden_keys(value: Any, *, rel: str, path: tuple[str, ...], errors: list[str]) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            child_path = (*path, str(key))
            dotted = ".".join(child_path)
            if "hidden" in str(key).lower() and dotted != "args.hidden_val_size":
                errors.append(f"{rel}: public manifest lock must not expose {dotted}.")
            _walk_hidden_keys(child, rel=rel, path=child_path, errors=errors)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _walk_hidden_keys(child, rel=rel, path=(*path, f"[{index}]"), errors=errors)


def _check_public_manifest_lock(path: Path, rel: str, errors: list[str]) -> None:
    raw = _load_json(path)
    if not isinstance(raw, dict):
        errors.append(f"{rel}: manifest source lock must be a JSON object.")
        return
    for key in ("data_source_lock", "data_source_lock_sha256"):
        if key in raw:
            errors.append(f"{rel}: {key} belongs in maintainer-only source locks, not public metadata.")
    _walk_hidden_keys(raw, rel=rel, path=(), errors=errors)


def _check_text_for_local_paths(path: Path, rel: str, errors: list[str]) -> None:
    if path.suffix not in TEXT_SUFFIXES:
        return
    try:
        text = path.read_text()
    except UnicodeDecodeError:
        return
    for marker in ABSOLUTE_LOCAL_PATH_MARKERS:
        if marker in text:
            errors.append(f"{rel}: contains local absolute path marker `{marker}`.")


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    tracked = _tracked_files(root, args.tracked_files_from)
    errors: list[str] = []

    for rel in tracked:
        if _is_forbidden_tracked_path(rel):
            errors.append(f"{rel}: must not be tracked in the public repo.")

    for rel in tracked:
        path = root / rel
        if not path.exists() or not path.is_file():
            continue
        if rel.startswith("tracks/") and rel.endswith((".yaml", ".yml")):
            _check_track_file(path, rel, errors)
        if rel == "leaderboard/official_hidden_assets.lock.json":
            _check_hidden_placeholder_lock(path, rel, errors)
        if rel == "leaderboard/official_manifest_source.lock.json":
            _check_public_manifest_lock(path, rel, errors)
        _check_text_for_local_paths(path, rel, errors)

    if errors:
        print("Public release leak check failed:")
        for error in errors:
            print(f"- {error}")
        raise SystemExit(1)
    print("Public release leak check passed.")


if __name__ == "__main__":
    main()
