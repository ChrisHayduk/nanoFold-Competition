from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

# Allow running as `python scripts/build_fingerprint.py` from repo root.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from nanofold.competition_policy import OFFICIAL_DATASET_FINGERPRINT_PATH
from nanofold.dataset_integrity import build_dataset_fingerprint, build_split_fingerprint


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Canonical dataset fingerprint builder.")
    ap.add_argument("--config", type=str, default="", help="Optional config YAML path to read data paths from.")
    ap.add_argument("--processed-features-dir", type=str, default="", help="Path to preprocessed feature .npz directory.")
    ap.add_argument("--processed-labels-dir", type=str, default="", help="Path to preprocessed label .npz directory.")
    ap.add_argument("--train-manifest", type=str, default="", help="Path to train manifest.")
    ap.add_argument("--val-manifest", type=str, default="", help="Path to val manifest.")
    ap.add_argument(
        "--manifest",
        action="append",
        default=[],
        help="Split manifest in NAME=PATH form. Repeat for arbitrary split fingerprints.",
    )
    ap.add_argument(
        "--output",
        type=str,
        default=OFFICIAL_DATASET_FINGERPRINT_PATH,
        help=f"Output fingerprint JSON path (default: {OFFICIAL_DATASET_FINGERPRINT_PATH}).",
    )
    ap.add_argument(
        "--allow-missing",
        action="store_true",
        help="Allow missing .npz files when building the fingerprint.",
    )
    ap.add_argument(
        "--skip-schema-checks",
        action="store_true",
        help="Skip NPZ schema validation (not recommended).",
    )
    ap.add_argument(
        "--features-only",
        action="store_true",
        help="Build fingerprint without label file hashes.",
    )
    ap.add_argument(
        "--track",
        type=str,
        default="",
        help="Optional track identifier to embed in the fingerprint metadata.",
    )
    ap.add_argument(
        "--source-lock",
        type=str,
        default="",
        help="Optional lock file path whose SHA256 is embedded in fingerprint metadata.",
    )
    return ap.parse_args()


def _parse_manifest_args(values: list[str]) -> dict[str, str]:
    manifests: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"--manifest entries must use NAME=PATH form (got `{value}`).")
        name, path = value.split("=", 1)
        name = name.strip()
        path = path.strip()
        if not name or not path:
            raise ValueError(f"--manifest entries must use non-empty NAME=PATH form (got `{value}`).")
        manifests[name] = path
    return manifests


def _resolve_data_paths(args: argparse.Namespace) -> tuple[str, str | None, dict[str, str]]:
    if args.config:
        cfg = yaml.safe_load(Path(args.config).read_text())
        if not isinstance(cfg, dict) or not isinstance(cfg.get("data"), dict):
            raise ValueError(f"Config must contain a `data` mapping: {args.config}")
        data_cfg = cfg["data"]
        processed_features_dir = str(data_cfg["processed_features_dir"])
        processed_labels_dir = str(data_cfg.get("processed_labels_dir", "")).strip() or None
        manifest_paths = {"train": str(data_cfg["train_manifest"]), "val": str(data_cfg["val_manifest"])}
    else:
        processed_features_dir = args.processed_features_dir
        processed_labels_dir = args.processed_labels_dir or None
        manifest_paths = _parse_manifest_args(list(args.manifest))
        if not manifest_paths and args.train_manifest and args.val_manifest:
            manifest_paths = {"train": args.train_manifest, "val": args.val_manifest}

    if not processed_features_dir or not manifest_paths:
        raise ValueError(
            "Must provide either --config, or --processed-features-dir plus --manifest NAME=PATH "
            "(or legacy --train-manifest and --val-manifest)."
        )
    return processed_features_dir, processed_labels_dir, manifest_paths


def main() -> None:
    args = parse_args()
    processed_features_dir, processed_labels_dir, manifest_paths = _resolve_data_paths(args)
    require_labels = not bool(args.features_only)
    if require_labels and not processed_labels_dir:
        raise ValueError("Label fingerprinting requested but no processed labels dir was provided.")

    if list(manifest_paths.keys()) == ["train", "val"]:
        fingerprint = build_dataset_fingerprint(
            processed_features_dir=processed_features_dir,
            processed_labels_dir=processed_labels_dir,
            train_manifest=manifest_paths["train"],
            val_manifest=manifest_paths["val"],
            require_no_missing=not bool(args.allow_missing),
            require_labels=require_labels,
            validate_schema=not bool(args.skip_schema_checks),
            track_id=args.track.strip() or None,
            source_lock_path=args.source_lock.strip() or None,
        )
    else:
        fingerprint = build_split_fingerprint(
            processed_features_dir=processed_features_dir,
            processed_labels_dir=processed_labels_dir,
            manifest_paths=manifest_paths,
            require_no_missing=not bool(args.allow_missing),
            require_labels=require_labels,
            validate_schema=not bool(args.skip_schema_checks),
            track_id=args.track.strip() or None,
            source_lock_path=args.source_lock.strip() or None,
        )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(fingerprint, indent=2, sort_keys=True) + "\n")
    print(f"Wrote fingerprint to {output_path.resolve()}")


if __name__ == "__main__":
    main()
