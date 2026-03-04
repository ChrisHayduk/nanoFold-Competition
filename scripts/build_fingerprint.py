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
from nanofold.dataset_integrity import build_dataset_fingerprint


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Canonical dataset fingerprint builder.")
    ap.add_argument("--config", type=str, default="", help="Optional config YAML path to read data paths from.")
    ap.add_argument("--processed-dir", type=str, default="", help="Path to preprocessed .npz directory.")
    ap.add_argument("--train-manifest", type=str, default="", help="Path to train manifest.")
    ap.add_argument("--val-manifest", type=str, default="", help="Path to val manifest.")
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
    return ap.parse_args()


def _resolve_data_paths(args: argparse.Namespace) -> tuple[str, str, str]:
    if args.config:
        cfg = yaml.safe_load(Path(args.config).read_text())
        if not isinstance(cfg, dict) or not isinstance(cfg.get("data"), dict):
            raise ValueError(f"Config must contain a `data` mapping: {args.config}")
        data_cfg = cfg["data"]
        processed_dir = str(data_cfg["processed_dir"])
        train_manifest = str(data_cfg["train_manifest"])
        val_manifest = str(data_cfg["val_manifest"])
    else:
        processed_dir = args.processed_dir
        train_manifest = args.train_manifest
        val_manifest = args.val_manifest

    if not processed_dir or not train_manifest or not val_manifest:
        raise ValueError(
            "Must provide either --config, or all of --processed-dir, --train-manifest, and --val-manifest."
        )
    return processed_dir, train_manifest, val_manifest


def main() -> None:
    args = parse_args()
    processed_dir, train_manifest, val_manifest = _resolve_data_paths(args)

    fingerprint = build_dataset_fingerprint(
        processed_dir=processed_dir,
        train_manifest=train_manifest,
        val_manifest=val_manifest,
        require_no_missing=not bool(args.allow_missing),
        validate_schema=not bool(args.skip_schema_checks),
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(fingerprint, indent=2, sort_keys=True) + "\n")
    print(f"Wrote fingerprint to {output_path.resolve()}")


if __name__ == "__main__":
    main()
