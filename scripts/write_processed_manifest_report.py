"""Write a processed-manifest report from requested chains and NPZ outputs."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nanofold.chain_paths import chain_error_path, chain_npz_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--processed-features-dir", type=Path, required=True)
    parser.add_argument("--processed-labels-dir", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--require-complete", action="store_true")
    parser.add_argument("--max-examples", type=int, default=50)
    return parser.parse_args()


def _read_manifest(path: Path) -> list[str]:
    return [
        line.strip()
        for line in path.read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def _error_message(features_dir: Path, chain_id: str) -> str | None:
    marker = chain_error_path(features_dir, chain_id)
    if not marker.exists():
        return None
    first_line = marker.read_text(errors="replace").splitlines()
    if not first_line:
        return ""
    return first_line[0].strip()


def build_report(
    *,
    source_manifest: Path,
    processed_features_dir: Path,
    processed_labels_dir: Path,
    output_manifest: Path,
    max_examples: int = 50,
) -> dict[str, Any]:
    requested = _read_manifest(source_manifest)
    present: list[str] = []
    missing: list[str] = []
    feature_only: list[str] = []
    label_only: list[str] = []
    error_examples: list[dict[str, str]] = []
    error_summary: Counter[str] = Counter()

    for chain_id in requested:
        feature_exists = chain_npz_path(processed_features_dir, chain_id).exists()
        label_exists = chain_npz_path(processed_labels_dir, chain_id).exists()
        if feature_exists and label_exists:
            present.append(chain_id)
            continue
        missing.append(chain_id)
        if feature_exists:
            feature_only.append(chain_id)
        if label_exists:
            label_only.append(chain_id)
        message = _error_message(processed_features_dir, chain_id)
        if message is not None:
            error_summary[message] += 1
            if len(error_examples) < max_examples:
                error_examples.append({"chain_id": chain_id, "error": message})

    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    output_manifest.write_text("".join(f"{chain_id}\n" for chain_id in present))

    report = {
        "schema_version": 1,
        "source_manifest": str(source_manifest),
        "processed_manifest": str(output_manifest),
        "processed_features_dir": str(processed_features_dir),
        "processed_labels_dir": str(processed_labels_dir),
        "requested_count": len(requested),
        "processed_count": len(present),
        "missing_count": len(missing),
        "feature_only_count": len(feature_only),
        "label_only_count": len(label_only),
        "missing_examples": missing[:max_examples],
        "feature_only_examples": feature_only[:max_examples],
        "label_only_examples": label_only[:max_examples],
        "error_summary": dict(error_summary.most_common()),
        "error_examples": error_examples,
    }
    return report


def main() -> None:
    args = parse_args()
    report = build_report(
        source_manifest=args.source_manifest,
        processed_features_dir=args.processed_features_dir,
        processed_labels_dir=args.processed_labels_dir,
        output_manifest=args.output_manifest,
        max_examples=int(args.max_examples),
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["processed_count"]:
        raise SystemExit("No processed examples are available for eval.")
    if args.require_complete and report["missing_count"]:
        raise SystemExit(
            "Processed manifest is incomplete: "
            f"{report['missing_count']} / {report['requested_count']} requested chains are missing."
        )


if __name__ == "__main__":
    main()
