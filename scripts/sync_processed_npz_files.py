from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from nanofold.chain_paths import chain_id_to_stem
from nanofold.data import read_manifest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Keep processed feature and label NPZ directories aligned to selected manifests.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--features-dir", type=Path, required=True)
    parser.add_argument("--labels-dir", type=Path, required=True)
    parser.add_argument(
        "--manifest",
        type=Path,
        action="append",
        required=True,
        help="Manifest path. Repeat for combined splits such as train+val.",
    )
    parser.add_argument("--remove-errors", action="store_true", help="Remove generated preprocess error markers.")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def _manifest_chain_ids(manifest_paths: list[Path]) -> list[str]:
    chain_ids: list[str] = []
    seen: set[str] = set()
    for manifest_path in manifest_paths:
        for chain_id in read_manifest(manifest_path):
            if chain_id in seen:
                raise ValueError(f"Duplicate chain ID across selected manifests: {chain_id}")
            seen.add(chain_id)
            chain_ids.append(chain_id)
    return chain_ids


def _unlink(path: Path, *, dry_run: bool) -> None:
    if not dry_run:
        path.unlink()


def _sync_npz_dir(directory: Path, *, expected_stems: set[str], dry_run: bool) -> tuple[int, int]:
    directory.mkdir(parents=True, exist_ok=True)
    kept = 0
    removed = 0
    for path in sorted(directory.glob("chain_*.npz")):
        if path.stem in expected_stems:
            kept += 1
            continue
        _unlink(path, dry_run=dry_run)
        removed += 1
    return kept, removed


def _remove_error_markers(directory: Path, *, dry_run: bool) -> int:
    if not directory.exists():
        return 0
    removed = 0
    for path in sorted(directory.glob("chain_*.error.txt")):
        _unlink(path, dry_run=dry_run)
        removed += 1
    return removed


def _missing_chain_ids(directory: Path, chain_ids: list[str]) -> list[str]:
    return [chain_id for chain_id in chain_ids if not (directory / f"{chain_id_to_stem(chain_id)}.npz").is_file()]


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    chain_ids = _manifest_chain_ids(args.manifest)
    expected_stems = {chain_id_to_stem(chain_id) for chain_id in chain_ids}

    feature_kept, feature_removed = _sync_npz_dir(args.features_dir, expected_stems=expected_stems, dry_run=args.dry_run)
    label_kept, label_removed = _sync_npz_dir(args.labels_dir, expected_stems=expected_stems, dry_run=args.dry_run)
    feature_error_removed = _remove_error_markers(args.features_dir, dry_run=args.dry_run) if args.remove_errors else 0
    label_error_removed = _remove_error_markers(args.labels_dir, dry_run=args.dry_run) if args.remove_errors else 0

    missing_features = _missing_chain_ids(args.features_dir, chain_ids)
    missing_labels = _missing_chain_ids(args.labels_dir, chain_ids)
    if missing_features or missing_labels:
        pieces = []
        if missing_features:
            pieces.append(f"missing feature NPZs: {', '.join(missing_features[:10])}")
        if missing_labels:
            pieces.append(f"missing label NPZs: {', '.join(missing_labels[:10])}")
        raise FileNotFoundError("; ".join(pieces))

    print(
        "Processed NPZ sync complete: "
        f"chains={len(chain_ids)}, "
        f"features_kept={feature_kept}, features_removed={feature_removed}, "
        f"labels_kept={label_kept}, labels_removed={label_removed}, "
        f"feature_errors_removed={feature_error_removed}, label_errors_removed={label_error_removed}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
