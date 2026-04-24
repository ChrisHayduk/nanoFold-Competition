"""Build a deterministic processed OpenProteinSet filter manifest.

This script is intentionally post-preprocess: it reads nanoFold feature and
label NPZs, records why each chain is accepted or rejected, and can emit a
plain-text accepted-chain manifest for downstream official split generation.

The defaults match nanoFold's competition curation target: length 40-256,
resolution <= 3.0 angstrom, and no single amino acid taking more than 80% of
the primary sequence. Optional MMseqs2 cluster TSV metadata can be embedded so
later tooling can audit or sample by sequence cluster without re-running
clustering.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

# Allow running as `python scripts/filter_openproteinset.py` from repo root.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from nanofold.data import read_manifest


DEFAULT_MIN_LENGTH = 40
DEFAULT_MAX_LENGTH = 256
DEFAULT_MAX_RESOLUTION_ANGSTROMS = 3.0
DEFAULT_MAX_SINGLE_AA_FRACTION = 0.8


def max_single_aa_fraction(aatype: np.ndarray) -> float:
    """Return the largest per-token share in a primary sequence."""
    if aatype.size == 0:
        return 1.0
    counts = Counter(int(value) for value in aatype.tolist())
    return max(counts.values()) / float(aatype.size)


def load_cluster_tsv(path: str | Path) -> dict[str, tuple[str, int]]:
    """Parse an MMseqs2 easy-cluster TSV as {chain_id: (cluster_id, size)}."""
    cluster_path = Path(path)
    members: dict[str, str] = {}
    cluster_sizes: Counter[str] = Counter()
    with cluster_path.open() as handle:
        for line in handle:
            parts = line.strip().split("\t")
            if len(parts) < 2:
                continue
            representative, member = parts[0], parts[1]
            members[member] = representative
            cluster_sizes[representative] += 1
    return {
        member: (representative, cluster_sizes[representative])
        for member, representative in members.items()
    }


def discover_chain_ids(
    processed_features_dir: str | Path,
    *,
    manifest_path: str | Path | None = None,
) -> list[str]:
    if manifest_path is not None:
        return read_manifest(manifest_path)
    features_dir = Path(processed_features_dir)
    return sorted(path.stem for path in features_dir.glob("*.npz"))


def _schema_reject_reasons(
    feature_path: Path,
    label_path: Path,
) -> list[str]:
    reject_reasons: list[str] = []
    if not feature_path.exists():
        reject_reasons.append("missing_feature")
    if not label_path.exists():
        reject_reasons.append("missing_label")
    return reject_reasons


def evaluate_chain(
    chain_id: str,
    *,
    processed_features_dir: str | Path,
    processed_labels_dir: str | Path,
    min_length: int,
    max_length: int | None,
    max_resolution: float,
    max_single_aa_fraction_threshold: float,
    cluster_info: dict[str, tuple[str, int]] | None = None,
) -> dict[str, Any]:
    """Evaluate one chain and return a JSON-serializable manifest entry."""
    features_dir = Path(processed_features_dir)
    labels_dir = Path(processed_labels_dir)
    feature_path = features_dir / f"{chain_id}.npz"
    label_path = labels_dir / f"{chain_id}.npz"

    reject_reasons = _schema_reject_reasons(feature_path, label_path)
    entry: dict[str, Any] = {
        "chain_id": chain_id,
        "accepted": False,
        "reject_reasons": reject_reasons,
    }
    if reject_reasons:
        return entry

    with np.load(feature_path) as features, np.load(label_path) as labels:
        missing_feature_keys = [key for key in ("aatype",) if key not in features]
        missing_label_keys = [
            key
            for key in ("ca_coords", "ca_mask", "atom14_positions", "atom14_mask", "resolution")
            if key not in labels
        ]
        if missing_feature_keys:
            reject_reasons.append("missing_feature_key")
        if missing_label_keys:
            reject_reasons.append("missing_label_key")
        if reject_reasons:
            entry["missing_feature_keys"] = missing_feature_keys
            entry["missing_label_keys"] = missing_label_keys
            return entry

        aatype = np.asarray(features["aatype"])
        length = int(aatype.shape[0])
        resolution = float(np.asarray(labels["resolution"]).reshape(()).item())
        single_aa_fraction = max_single_aa_fraction(aatype)

    if length < min_length:
        reject_reasons.append("min_length")
    if max_length is not None and length > max_length:
        reject_reasons.append("max_length")
    if np.isfinite(resolution) and resolution > max_resolution:
        reject_reasons.append("resolution")
    if single_aa_fraction > max_single_aa_fraction_threshold:
        reject_reasons.append("single_aa")

    entry.update(
        {
            "length": length,
            "resolution": resolution,
            "max_single_aa_fraction": round(single_aa_fraction, 4),
            "accepted": not reject_reasons,
            "reject_reasons": reject_reasons,
        }
    )
    if cluster_info and chain_id in cluster_info:
        cluster_id, cluster_size = cluster_info[chain_id]
        entry["cluster_id"] = cluster_id
        entry["cluster_size"] = cluster_size
    return entry


def build_manifest(
    *,
    processed_features_dir: str | Path,
    processed_labels_dir: str | Path,
    manifest_path: str | Path | None,
    min_length: int,
    max_length: int | None,
    max_resolution: float,
    max_single_aa_fraction_threshold: float,
    mmseqs_cluster_tsv: str | Path | None = None,
    sample_limit: int | None = None,
) -> dict[str, Any]:
    chain_ids = discover_chain_ids(processed_features_dir, manifest_path=manifest_path)
    if sample_limit is not None:
        chain_ids = chain_ids[:sample_limit]

    cluster_info = load_cluster_tsv(mmseqs_cluster_tsv) if mmseqs_cluster_tsv else {}
    entries = [
        evaluate_chain(
            chain_id,
            processed_features_dir=processed_features_dir,
            processed_labels_dir=processed_labels_dir,
            min_length=min_length,
            max_length=max_length,
            max_resolution=max_resolution,
            max_single_aa_fraction_threshold=max_single_aa_fraction_threshold,
            cluster_info=cluster_info,
        )
        for chain_id in chain_ids
    ]

    reason_counts: Counter[str] = Counter()
    for entry in entries:
        reason_counts.update(entry.get("reject_reasons", []))

    summary: dict[str, Any] = {
        "total": len(entries),
        "accepted": sum(1 for entry in entries if entry["accepted"]),
        "rejected": sum(1 for entry in entries if not entry["accepted"]),
        "reject_reasons": dict(sorted(reason_counts.items())),
    }
    return {
        "config": {
            "min_length": min_length,
            "max_length": max_length,
            "max_resolution_angstroms": max_resolution,
            "max_single_aa_fraction": max_single_aa_fraction_threshold,
            "mmseqs_cluster_tsv": str(mmseqs_cluster_tsv) if mmseqs_cluster_tsv else None,
        },
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_features_dir": str(processed_features_dir),
        "source_labels_dir": str(processed_labels_dir),
        "source_manifest": str(manifest_path) if manifest_path else None,
        "summary": summary,
        "chains": entries,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Filter processed nanoFold/OpenProteinSet chains and write a JSON audit manifest.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--processed-features-dir", type=Path, default=Path("data/processed_features"))
    parser.add_argument("--processed-labels-dir", type=Path, default=Path("data/processed_labels"))
    parser.add_argument("--manifest", type=Path, default=None, help="Optional source chain-id manifest to preserve order.")
    parser.add_argument("--manifest-out", type=Path, required=True, help="Destination JSON manifest.")
    parser.add_argument("--accepted-out", type=Path, default=None, help="Optional text file of accepted chain IDs.")
    parser.add_argument("--min-length", type=int, default=DEFAULT_MIN_LENGTH)
    parser.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH)
    parser.add_argument("--max-resolution-angstroms", type=float, default=DEFAULT_MAX_RESOLUTION_ANGSTROMS)
    parser.add_argument("--max-single-aa-fraction", type=float, default=DEFAULT_MAX_SINGLE_AA_FRACTION)
    parser.add_argument("--mmseqs-cluster-tsv", type=Path, default=None)
    parser.add_argument("--sample-limit", type=int, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    max_length = int(args.max_length) if int(args.max_length) > 0 else None
    manifest = build_manifest(
        processed_features_dir=args.processed_features_dir,
        processed_labels_dir=args.processed_labels_dir,
        manifest_path=args.manifest,
        min_length=int(args.min_length),
        max_length=max_length,
        max_resolution=float(args.max_resolution_angstroms),
        max_single_aa_fraction_threshold=float(args.max_single_aa_fraction),
        mmseqs_cluster_tsv=args.mmseqs_cluster_tsv,
        sample_limit=args.sample_limit,
    )

    args.manifest_out.parent.mkdir(parents=True, exist_ok=True)
    args.manifest_out.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    if args.accepted_out is not None:
        accepted_ids = [entry["chain_id"] for entry in manifest["chains"] if entry["accepted"]]
        args.accepted_out.parent.mkdir(parents=True, exist_ok=True)
        args.accepted_out.write_text("".join(f"{chain_id}\n" for chain_id in accepted_ids))

    summary = manifest["summary"]
    print(
        f"[filter] total={summary['total']} accepted={summary['accepted']} "
        f"rejected={summary['rejected']}"
    )
    print(f"[filter] manifest -> {args.manifest_out}")
    if args.accepted_out is not None:
        print(f"[filter] accepted -> {args.accepted_out}")


if __name__ == "__main__":
    main()
