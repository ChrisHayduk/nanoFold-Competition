from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import yaml


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Pin hidden asset digests into lock + track metadata.")
    ap.add_argument("--hidden-manifest", type=str, required=True)
    ap.add_argument("--hidden-features-dir", type=str, required=True)
    ap.add_argument("--hidden-labels-dir", type=str, required=True)
    ap.add_argument("--hidden-fingerprint", type=str, required=True)
    ap.add_argument("--track-file", type=str, default="tracks/limited_large.yaml")
    ap.add_argument("--lock-file", type=str, default="leaderboard/official_hidden_assets.lock.json")
    ap.add_argument("--hidden-chain-count", type=int, default=-1)
    return ap.parse_args()


def _sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def _tree_sha256(root: Path) -> str:
    hasher = hashlib.sha256()
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        rel = str(path.relative_to(root)).replace("\\", "/")
        hasher.update(rel.encode("utf-8"))
        hasher.update(b"\t")
        hasher.update(_sha256(path).encode("ascii"))
        hasher.update(b"\n")
    return hasher.hexdigest()


def _read_chain_count(manifest_path: Path) -> int:
    return sum(
        1
        for line in manifest_path.read_text().splitlines()
        if line.strip() and not line.startswith("#")
    )


def main() -> None:
    args = parse_args()
    hidden_manifest = Path(args.hidden_manifest).resolve()
    hidden_features_dir = Path(args.hidden_features_dir).resolve()
    hidden_labels_dir = Path(args.hidden_labels_dir).resolve()
    hidden_fingerprint = Path(args.hidden_fingerprint).resolve()
    track_file = Path(args.track_file).resolve()
    lock_file = Path(args.lock_file).resolve()

    for path in (hidden_manifest, hidden_fingerprint):
        if not path.exists() or not path.is_file():
            raise FileNotFoundError(path)
    for path in (hidden_features_dir, hidden_labels_dir):
        if not path.exists() or not path.is_dir():
            raise FileNotFoundError(path)

    lock_obj = {
        "track": track_file.stem,
        "notes": "Maintainer-only hidden asset lock. Keep paths private; commit hashes/metadata only.",
        "hidden_manifest_sha256": _sha256(hidden_manifest),
        "hidden_features_fingerprint_sha256": _tree_sha256(hidden_features_dir),
        "hidden_labels_fingerprint_sha256": _tree_sha256(hidden_labels_dir),
        "hidden_fingerprint_sha256": _sha256(hidden_fingerprint),
    }
    lock_file.parent.mkdir(parents=True, exist_ok=True)
    lock_file.write_text(json.dumps(lock_obj, indent=2) + "\n")

    raw_track = yaml.safe_load(track_file.read_text())
    if not isinstance(raw_track, dict):
        raise ValueError(f"Track file must be a YAML mapping: {track_file}")
    dataset = raw_track.get("dataset")
    if not isinstance(dataset, dict):
        raise ValueError(f"Track file missing dataset mapping: {track_file}")

    dataset["hidden_manifest_sha256"] = lock_obj["hidden_manifest_sha256"]
    dataset["hidden_fingerprint_sha256"] = lock_obj["hidden_fingerprint_sha256"]
    dataset["hidden_chain_count"] = (
        int(args.hidden_chain_count)
        if args.hidden_chain_count > 0
        else _read_chain_count(hidden_manifest)
    )
    track_file.write_text(yaml.safe_dump(raw_track, sort_keys=False))

    print(f"Wrote hidden lock: {lock_file}")
    print(f"Updated hidden digests in track: {track_file}")


if __name__ == "__main__":
    main()
