from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from nanofold.data import read_manifest
from nanofold.dataset_integrity import verify_split_against_fingerprint


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Verify complete hidden assets against their fingerprint, then atomically pin their digests into a private lock.")
    ap.add_argument("--hidden-manifest", type=str, required=True)
    ap.add_argument("--hidden-features-dir", type=str, required=True)
    ap.add_argument("--hidden-labels-dir", type=str, required=True)
    ap.add_argument("--hidden-fingerprint", type=str, required=True)
    ap.add_argument("--track-id", type=str, default="limited")
    ap.add_argument("--lock-file", type=str, default=".nanofold_private/leaderboard/private_hidden_assets.lock.json")
    ap.add_argument("--hidden-chain-count", type=int, default=None,
                    help="Optional expected count; must equal the nonempty unique hidden manifest.")
    return ap.parse_args(argv)


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


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    hidden_manifest = Path(args.hidden_manifest).resolve()
    hidden_features_dir = Path(args.hidden_features_dir).resolve()
    hidden_labels_dir = Path(args.hidden_labels_dir).resolve()
    hidden_fingerprint = Path(args.hidden_fingerprint).resolve()
    lock_file = Path(args.lock_file).resolve()

    for path in (hidden_manifest, hidden_fingerprint):
        if not path.exists() or not path.is_file():
            raise FileNotFoundError(path)
    for path in (hidden_features_dir, hidden_labels_dir):
        if not path.exists() or not path.is_dir():
            raise FileNotFoundError(path)
    inputs = [hidden_manifest, hidden_fingerprint, hidden_features_dir, hidden_labels_dir]
    if any(lock_file == path or (lock_file.exists() and lock_file.samefile(path)) for path in inputs):
        raise ValueError("Hidden lock output must not alias source inputs.")
    if any(lock_file.is_relative_to(path) for path in (hidden_features_dir, hidden_labels_dir)):
        raise ValueError("Hidden lock output must be outside the pinned asset directories.")
    if hidden_features_dir.is_relative_to(hidden_labels_dir) or hidden_labels_dir.is_relative_to(hidden_features_dir):
        raise ValueError("Hidden feature and label directories must be separate.")
    if lock_file.is_relative_to(REPO_ROOT) and not lock_file.is_relative_to(REPO_ROOT / ".nanofold_private"):
        raise ValueError("In-repository hidden locks must remain under .nanofold_private.")
    chain_ids = read_manifest(hidden_manifest)
    if not chain_ids or len(chain_ids) != len(set(chain_ids)):
        raise ValueError("Hidden manifest must contain a nonempty set of unique chain IDs.")
    if args.hidden_chain_count is not None and args.hidden_chain_count != len(chain_ids):
        raise ValueError("Expected hidden chain count does not match the manifest.")
    if not args.track_id.strip():
        raise ValueError("Track ID must be nonempty.")
    verify_split_against_fingerprint(
        processed_features_dir=hidden_features_dir, processed_labels_dir=hidden_labels_dir,
        manifest_paths={"hidden_val": hidden_manifest}, expected_fingerprint_path=hidden_fingerprint,
        require_no_missing=True, require_labels=True, validate_schema=True, track_id=args.track_id,
    )

    lock_obj = {
        "track": str(args.track_id),
        "notes": "Maintainer-only hidden asset lock. Do not commit this file.",
        "hidden_chain_count": len(chain_ids),
        "hidden_manifest_sha256": _sha256(hidden_manifest),
        "hidden_features_fingerprint_sha256": _tree_sha256(hidden_features_dir),
        "hidden_labels_fingerprint_sha256": _tree_sha256(hidden_labels_dir),
        "hidden_fingerprint_sha256": _sha256(hidden_fingerprint),
    }
    lock_file.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=lock_file.parent, prefix=f".{lock_file.name}.", delete=False) as handle:
        temporary = Path(handle.name)
        try:
            handle.write(json.dumps(lock_obj, indent=2) + "\n")
            handle.flush()
            temporary.replace(lock_file)
        finally:
            temporary.unlink(missing_ok=True)

    print(f"Wrote hidden lock: {lock_file}")


if __name__ == "__main__":
    main()
