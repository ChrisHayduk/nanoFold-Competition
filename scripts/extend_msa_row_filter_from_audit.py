"""Extend an MSA-row filter with residual rows from an MMseqs audit.

``scripts/audit_msa_split.py --run-mmseqs`` writes anonymous source headers in
the form ``trainmsa_<index>`` unless identifier output is explicitly enabled.
This helper reconstructs that same unique source-row order from a processed
feature root, maps residual MMseqs hits back to ungapped sequence hashes, and
writes a new de-identified filter JSON.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from nanofold.a3m import GAP_ID, MASK_ID, RESTYPES
from nanofold.chain_paths import chain_npz_path
from nanofold.data import read_manifest
from nanofold.mmseqs import iter_mmseqs_hits, load_msa_for_audit, sequence_records_sha256

MSA_TOKEN_TO_AA = {idx: aa for idx, aa in enumerate(RESTYPES)}
MSA_TOKEN_TO_AA[20] = "X"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base-filter", type=Path, required=True)
    ap.add_argument(
        "--source-manifest",
        action="append",
        required=True,
        help="Manifest(s) used as the source side of the audit MMseqs run.",
    )
    ap.add_argument("--features-dir", type=Path, required=True)
    ap.add_argument("--m8", type=Path, required=True, help="MMseqs m8 file from audit_msa_split.py.")
    ap.add_argument("--source-prefix", default="trainmsa_")
    ap.add_argument("--audit-summary", type=Path, required=True,
                    help="Original msa_split_audit_summary.json binding the m8 and ordered source sequences.")
    ap.add_argument("--output", type=Path, required=True)
    return ap.parse_args()


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sequence_sha256(sequence: str) -> str:
    return hashlib.sha256(sequence.upper().encode("utf-8")).hexdigest()


def _manifest_record(path: Path) -> dict[str, Any]:
    chain_ids = read_manifest(path)
    return {
        "path": str(path),
        "sha256": _sha256_file(path),
        "chain_count": len(chain_ids),
    }


def _msa_row_to_ungapped_sequence(row: np.ndarray) -> str:
    chars: list[str] = []
    for value in row.tolist():
        token = int(value)
        if token in (GAP_ID, MASK_ID):
            continue
        chars.append(MSA_TOKEN_TO_AA.get(token, "X"))
    return "".join(chars)


def _source_records(*, manifest_paths: Sequence[Path], features_dir: Path) -> list[tuple[str, str]]:
    source_records: list[tuple[str, str]] = []
    seen: set[str] = set()
    for manifest_path in manifest_paths:
        for chain_id in read_manifest(manifest_path):
            feature_path = chain_npz_path(features_dir, chain_id)
            if not feature_path.exists():
                raise FileNotFoundError(f"Missing processed feature NPZ for {chain_id}: {feature_path}")
            msa = load_msa_for_audit(feature_path, chain_id=chain_id)
            for row in msa[1:]:
                sequence = _msa_row_to_ungapped_sequence(row)
                if len(sequence) < 20:
                    continue
                digest = _sequence_sha256(sequence)
                if digest in seen:
                    continue
                seen.add(digest)
                source_records.append((f"trainmsa_{len(source_records)}", sequence))
    return source_records


def extend_filter_from_audit(
    *,
    base_filter_path: Path,
    source_manifest_paths: Sequence[Path],
    features_dir: Path,
    m8_path: Path,
    source_prefix: str,
    output_path: Path,
    audit_summary_path: Path,
) -> dict[str, Any]:
    payload = json.loads(base_filter_path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"Base filter must be a JSON object: {base_filter_path}")
    raw_excluded = payload.get("excluded_sequence_sha256")
    if not isinstance(raw_excluded, list):
        raise ValueError(f"Base filter missing `excluded_sequence_sha256`: {base_filter_path}")

    bad_hashes = [value for value in raw_excluded if not isinstance(value, str) or re.fullmatch(r"[0-9a-fA-F]{64}", value) is None]
    base_excluded = {str(value).lower() for value in raw_excluded}
    if bad_hashes:
        raise ValueError(f"Base filter contains malformed SHA256 values: {bad_hashes[:3]}")

    if source_prefix != "trainmsa_":
        raise ValueError("Residual closure requires anonymous trainmsa_ source records")
    summary = json.loads(audit_summary_path.read_text())
    report = summary.get("mmseqs_homology", {}) if isinstance(summary, dict) else {}
    binding = report.get("input_binding", {})
    if not isinstance(binding, dict) or binding.get("schema_version") != 1 or report.get("identifiers_in_m8") is not False:
        raise ValueError("Audit summary must bind an anonymous MSA homology search")
    source_records = _source_records(manifest_paths=source_manifest_paths, features_dir=features_dir)
    if binding.get("source_records_sha256") != sequence_records_sha256(source_records):
        raise ValueError("Source MSA sequences or ordering changed since the audit")
    if binding.get("m8_sha256") != _sha256_file(m8_path):
        raise ValueError("MMseqs output does not match the bound audit")
    query_ids = binding.get("query_ids")
    if not isinstance(query_ids, list) or not query_ids or any(not isinstance(value, str) for value in query_ids):
        raise ValueError("Audit summary is missing its query identifiers")
    threshold = report.get("threshold", {})
    if any(threshold.get(key) != value for key, value in (("alignment_mode", 3), ("seq_id_mode", 0), ("cov_mode", 0))):
        raise ValueError("Audit uses an unsupported alignment contract")
    source_hashes = {header: _sequence_sha256(sequence) for header, sequence in source_records}
    residual_hashes: set[str] = set()
    hit_count = 0
    for _query, target in iter_mmseqs_hits(
        m8_path, query_ids=set(query_ids), target_ids=set(source_hashes),
        min_seq_id=float(threshold["min_seq_id"]), coverage=float(threshold["coverage"]),
    ):
        residual_hashes.add(source_hashes[target])
        hit_count += 1
    merged = base_excluded | residual_hashes
    supplemental_record = {
        "type": "mmseqs_audit_residual_closure",
        "base_filter_path": str(base_filter_path),
        "base_filter_sha256": _sha256_file(base_filter_path),
        "features_dir": str(features_dir),
        "source_manifests": [_manifest_record(path) for path in source_manifest_paths],
        "m8_path": str(m8_path),
        "m8_sha256": _sha256_file(m8_path),
        "source_prefix": source_prefix,
        "source_record_count": len(source_hashes),
        "m8_hit_count": hit_count,
        "residual_source_record_count": len(residual_hashes),
        "audit_summary_sha256": _sha256_file(audit_summary_path),
        "source_records_sha256": binding["source_records_sha256"],
        "new_excluded_sequence_count": len(residual_hashes - base_excluded),
        "already_excluded_sequence_count": len(residual_hashes & base_excluded),
    }

    payload = dict(payload)
    payload["excluded_sequence_sha256"] = sorted(merged)
    payload["excluded_sequence_count"] = len(merged)
    supplemental = payload.get("supplemental_filters")
    if not isinstance(supplemental, list):
        supplemental = []
    supplemental.append(supplemental_record)
    payload["supplemental_filters"] = supplemental

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def main() -> None:
    args = parse_args()
    payload = extend_filter_from_audit(
        base_filter_path=args.base_filter,
        source_manifest_paths=[Path(path) for path in args.source_manifest],
        features_dir=args.features_dir,
        m8_path=args.m8,
        source_prefix=str(args.source_prefix),
        output_path=args.output,
        audit_summary_path=args.audit_summary,
    )
    latest = payload["supplemental_filters"][-1]
    print(
        "Wrote extended MSA row filter: "
        f"+{latest['new_excluded_sequence_count']} new hashes, "
        f"{payload['excluded_sequence_count']} total -> {args.output}"
    )


if __name__ == "__main__":
    main()
