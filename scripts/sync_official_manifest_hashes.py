from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import yaml


@dataclass(frozen=True)
class ManifestHashes:
    train: str
    val: str
    all: str
    train_count: int
    val_count: int
    unique_count: int


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=(
            "Synchronize official manifest SHA256 values across track policy, lock metadata, and docs."
        )
    )
    ap.add_argument("--manifests-dir", type=str, default="data/manifests")
    ap.add_argument("--track-file", type=str, default="tracks/limited_large.yaml")
    ap.add_argument("--lock-file", type=str, default="leaderboard/official_manifest_source.lock.json")
    ap.add_argument("--readme", type=str, default="README.md")
    ap.add_argument("--competition-doc", type=str, default="COMPETITION.md")
    ap.add_argument(
        "--check",
        action="store_true",
        help="Check-only mode. Exit non-zero if any file is out of sync.",
    )
    return ap.parse_args()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_manifest_ids(path: Path) -> list[str]:
    ids: list[str] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        ids.append(line)
    return ids


def _compute_hashes(manifests_dir: Path) -> ManifestHashes:
    train_path = manifests_dir / "train.txt"
    val_path = manifests_dir / "val.txt"
    all_path = manifests_dir / "all.txt"
    for p in (train_path, val_path, all_path):
        if not p.exists():
            raise FileNotFoundError(f"Missing manifest file: {p}")

    train_ids = _read_manifest_ids(train_path)
    val_ids = _read_manifest_ids(val_path)
    all_ids = _read_manifest_ids(all_path)
    return ManifestHashes(
        train=_sha256(train_path),
        val=_sha256(val_path),
        all=_sha256(all_path),
        train_count=len(train_ids),
        val_count=len(val_ids),
        unique_count=len(set(all_ids)),
    )


def _replace_required(text: str, pattern: str, repl: str, *, label: str) -> str:
    updated, n = re.subn(pattern, repl, text, count=1, flags=re.MULTILINE)
    if n != 1:
        raise ValueError(f"Could not locate {label} pattern for replacement.")
    return updated


def _update_readme(path: Path, hashes: ManifestHashes) -> str:
    text = path.read_text()
    text = _replace_required(
        text,
        r"(- `train\.txt`: `)[0-9a-f]{64}(`)",
        rf"\g<1>{hashes.train}\2",
        label="README train hash",
    )
    text = _replace_required(
        text,
        r"(- `val\.txt`: `)[0-9a-f]{64}(`)",
        rf"\g<1>{hashes.val}\2",
        label="README val hash",
    )
    text = _replace_required(
        text,
        r"(- `all\.txt`: `)[0-9a-f]{64}(`)",
        rf"\g<1>{hashes.all}\2",
        label="README all hash",
    )
    return text


def _update_competition_doc(path: Path, hashes: ManifestHashes) -> str:
    text = path.read_text()
    text = _replace_required(
        text,
        r"(- train: `)[0-9a-f]{64}(`)",
        rf"\g<1>{hashes.train}\2",
        label="COMPETITION train hash",
    )
    text = _replace_required(
        text,
        r"(- val: `)[0-9a-f]{64}(`)",
        rf"\g<1>{hashes.val}\2",
        label="COMPETITION val hash",
    )
    text = _replace_required(
        text,
        r"(- all: `)[0-9a-f]{64}(`)",
        rf"\g<1>{hashes.all}\2",
        label="COMPETITION all hash",
    )
    return text


def _update_track_yaml(path: Path, hashes: ManifestHashes) -> str:
    raw = yaml.safe_load(path.read_text())
    if not isinstance(raw, dict):
        raise ValueError(f"Track file must be a YAML mapping: {path}")
    dataset = raw.get("dataset")
    if not isinstance(dataset, dict):
        raise ValueError(f"Track file missing `dataset` mapping: {path}")
    dataset["train_manifest_sha256"] = hashes.train
    dataset["val_manifest_sha256"] = hashes.val
    dataset["all_manifest_sha256"] = hashes.all
    dataset["train_chain_count"] = hashes.train_count
    dataset["val_chain_count"] = hashes.val_count
    return yaml.safe_dump(raw, sort_keys=False)


def _update_lock_json(path: Path, hashes: ManifestHashes, manifests_dir: Path) -> str:
    if path.exists():
        raw = json.loads(path.read_text())
    else:
        raw = {}
    if not isinstance(raw, dict):
        raise ValueError(f"Lock file must be a JSON object: {path}")

    outputs = raw.get("outputs")
    if not isinstance(outputs, dict):
        outputs = {}
        raw["outputs"] = outputs

    train_path = (manifests_dir / "train.txt").resolve()
    val_path = (manifests_dir / "val.txt").resolve()
    all_path = (manifests_dir / "all.txt").resolve()
    outputs["train_manifest"] = str(train_path)
    outputs["val_manifest"] = str(val_path)
    outputs["all_manifest"] = str(all_path)
    outputs["train_manifest_sha256"] = hashes.train
    outputs["val_manifest_sha256"] = hashes.val
    outputs["all_manifest_sha256"] = hashes.all
    outputs["train_count"] = hashes.train_count
    outputs["val_count"] = hashes.val_count
    outputs["unique_count"] = hashes.unique_count
    return json.dumps(raw, indent=2) + "\n"


def _write_or_check(path: Path, new_text: str, *, check_only: bool) -> bool:
    old_text = path.read_text() if path.exists() else ""
    changed = old_text != new_text
    if changed and not check_only:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(new_text)
    return changed


def main() -> None:
    args = parse_args()
    manifests_dir = Path(args.manifests_dir).resolve()
    track_file = Path(args.track_file).resolve()
    lock_file = Path(args.lock_file).resolve()
    readme = Path(args.readme).resolve()
    competition_doc = Path(args.competition_doc).resolve()

    hashes = _compute_hashes(manifests_dir)
    updates: Dict[Path, str] = {
        track_file: _update_track_yaml(track_file, hashes),
        lock_file: _update_lock_json(lock_file, hashes, manifests_dir),
        readme: _update_readme(readme, hashes),
        competition_doc: _update_competition_doc(competition_doc, hashes),
    }

    changed_paths: list[Path] = []
    for path, new_text in updates.items():
        if _write_or_check(path, new_text, check_only=bool(args.check)):
            changed_paths.append(path)

    if changed_paths:
        rel = [str(p.relative_to(Path.cwd())) if p.is_relative_to(Path.cwd()) else str(p) for p in changed_paths]
        if args.check:
            print("Out-of-sync official hash references:")
            for item in rel:
                print(f"- {item}")
            print("\nRun:\n  python scripts/sync_official_manifest_hashes.py")
            raise SystemExit(1)
        print("Updated files:")
        for item in rel:
            print(f"- {item}")
    else:
        print("All official hash references are already in sync.")


if __name__ == "__main__":
    main()
