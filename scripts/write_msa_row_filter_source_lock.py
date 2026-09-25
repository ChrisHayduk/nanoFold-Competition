"""Write a compact audit/source-lock record for an MSA-row filter.

The row-filter JSON can be very large because it stores every excluded sequence
hash. This helper produces a small, public-safe summary that records the
thresholds, source manifests, MMseqs settings, row-filter hash, and aggregate
processed-feature filtering counts without copying raw sequences, excluded hash
lists, hidden chain IDs, or private paths into the summary by default.
Completed bounded searches retain validated aggregate counts while batch receipt
paths and hashes remain private.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from nanofold.msa_filter_summary import validate_batching_summary

PUBLIC_PATHS = frozenset({
    "data/manifests/train.txt", "data/manifests/val.txt", "data/manifests/all.txt",
    "data/openproteinset", "data/openproteinset/pdb_data/data_caches/chain_data_cache.json",
    "data/processed_features",
})
SOURCE_STAT_KEYS = ("chains", "missing_features", "missing_msa_files", "rows", "non_query_rows", "unique_non_query_rows")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--msa-row-filter", type=Path, required=True)
    ap.add_argument(
        "--processed-filter-audit",
        type=Path,
        action="append",
        default=[],
        help="processed_msa_row_filter_audit.json from each filtering pass. Repeat in order.",
    )
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument(
        "--include-private-paths",
        action="store_true",
        help="Do not redact private paths/hashes. Use only for sealed maintainer-local records.",
    )
    return ap.parse_args()


def _sha256_file(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    raw = json.loads(path.read_text())
    if not isinstance(raw, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return raw


def _is_private_path(value: str | None) -> bool:
    if not value:
        return False
    return value.replace("\\", "/") not in PUBLIC_PATHS


def _safe_path(value: str | None, *, include_private_paths: bool) -> str | None:
    if value is None:
        return None
    if include_private_paths or not _is_private_path(value):
        return value
    return "<private>"


def _manifest_role(path: str | None) -> str:
    if path is None:
        return "manifest"
    normalized = path.replace("\\", "/").lower()
    if "hidden" in normalized or ".nanofold_private" in normalized:
        return "hidden"
    if normalized.endswith("val.txt"):
        return "public_validation"
    if normalized.endswith("train.txt"):
        return "train"
    return "manifest"


def _safe_manifest(record: Any, *, include_private_paths: bool) -> dict[str, Any]:
    if not isinstance(record, dict):
        raise ValueError(f"Manifest record must be an object, got {record!r}")
    path = record.get("path")
    if path is not None and not isinstance(path, str):
        raise ValueError(f"Manifest path must be a string, got {path!r}")
    private = _is_private_path(path)
    safe: dict[str, Any] = {
        "role": _manifest_role(path),
        "path": _safe_path(path, include_private_paths=include_private_paths),
        "chain_count": record.get("chain_count"),
    }
    if include_private_paths or not private:
        safe["sha256"] = record.get("sha256")
    else:
        safe["sha256"] = "<private>"
    return safe


def _safe_manifests(records: Any, *, include_private_paths: bool) -> list[dict[str, Any]]:
    if records is None:
        return []
    if not isinstance(records, list):
        raise ValueError("Manifest records must be a list.")
    return [_safe_manifest(record, include_private_paths=include_private_paths) for record in records]


def _normalize_mmseqs_command(command: Any) -> list[str]:
    if not isinstance(command, list):
        return []
    normalized = [str(part) for part in command]
    try:
        search_index = normalized.index("easy-search")
    except ValueError:
        return []
    # Only retain the search contract, never executable paths or arbitrary arguments.
    safe = ["mmseqs", "easy-search"]
    replacements = ["<heldout_targets.fasta>", "<source_msa_rows.fasta>", "<output.m8>", "<tmp_dir>"]
    safe.extend(replacements)
    numeric_flags = {
        "--min-seq-id", "-c", "--cov-mode", "--alignment-mode", "--seq-id-mode",
        "--threads", "--max-seqs", "-e", "-s", "--prefilter-mode", "--mask",
        "--comp-bias-corr", "--search-type", "--dbtype", "--max-accept", "--max-rejected",
    }
    for index in range(search_index + 5, len(normalized) - 1):
        flag = normalized[index]
        if flag in numeric_flags:
            try:
                float(normalized[index + 1])
            except ValueError:
                continue
            safe.extend([flag, normalized[index + 1]])
    return safe


def _safe_search_parameters(value: Any) -> dict[str, int | float]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError("MMseqs search parameters must be an object")
    fields = ("prefilter_mode", "evalue", "sensitivity", "mask", "comp_bias_corr", "search_type", "dbtype",
              "max_accept", "max_rejected", "max_seqs")
    result = {}
    for field in fields:
        if field not in value:
            continue
        number = value[field]
        if isinstance(number, bool) or not isinstance(number, (int, float)) or not math.isfinite(number):
            raise ValueError(f"MMseqs search parameter {field} must be finite numeric data")
        result[field] = number
    return result


def _safe_input_binding(value: Any, *, include_private_paths: bool) -> dict[str, str]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError("Row-filter input binding must be an object")
    result = {}
    for field in ("source_fasta_sha256", "heldout_fasta_sha256", "m8_sha256", "batch_receipt_sha256",
                  "source_sequence_set_sha256", "heldout_sequence_set_sha256", "source_asset_manifest_sha256"):
        if field not in value:
            continue
        digest = value[field]
        if not isinstance(digest, str) or len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            raise ValueError(f"Row-filter input binding {field} must be a SHA256 digest")
        result[field] = digest if include_private_paths else "<private>"
    return result


def _summarize_row_filter(filter_path: Path, *, include_private_paths: bool) -> dict[str, Any]:
    payload = _load_json(filter_path)
    if payload.get("inventory_type") is not None:
        raise ValueError("Search inventory is not a row filter; run the search before writing a filter source lock")
    search_contract = payload.get("search_contract")
    if search_contract not in (None, "mmseqs-local-all-pairs-v1", "mmseqs-prefiltered-v1"):
        raise ValueError("Unsupported row-filter search contract")
    source = payload.get("source", {})
    heldout = payload.get("heldout", {})
    mmseqs = payload.get("mmseqs", {})
    chain_data_cache = payload.get("chain_data_cache", {})

    if not isinstance(source, dict) or not isinstance(heldout, dict):
        raise ValueError("Row filter is missing source/heldout objects.")
    if not isinstance(mmseqs, dict):
        raise ValueError("Row filter is missing mmseqs object.")
    if not isinstance(chain_data_cache, dict):
        raise ValueError("Row filter has invalid chain_data_cache object.")
    source_stats = source.get("stats", {})
    if not isinstance(source_stats, dict):
        raise ValueError("Row filter source stats must be an object.")
    validate_batching_summary(payload, public_safe=False)

    summary: dict[str, Any] = {
        "path": _safe_path(str(filter_path), include_private_paths=include_private_paths),
        "sha256": _sha256_file(filter_path),
        "filter_type": payload.get("filter_type"),
        "search_contract": search_contract,
        "exhaustive": payload.get("exhaustive") if isinstance(payload.get("exhaustive"), bool) else None,
        "input_binding": _safe_input_binding(payload.get("input_binding"), include_private_paths=include_private_paths),
        "threshold": payload.get("threshold"),
        "excluded_sequence_count": payload.get("excluded_sequence_count"),
        "excluded_sequence_sha256_included": False,
        "mmseqs": {
            "binary": mmseqs.get("binary") if include_private_paths else "mmseqs",
            "version": mmseqs.get("version"),
            "command": _normalize_mmseqs_command(mmseqs.get("command")),
            "max_seqs": mmseqs.get("max_seqs"),
            "search_parameters": _safe_search_parameters(mmseqs.get("search_parameters")),
            "possible_pair_count": mmseqs.get("possible_pair_count"),
            "hit_count": mmseqs.get("hit_count"),
        },
        "source": {
            "mode": source.get("mode"),
            "complete": source.get("complete") if isinstance(source.get("complete"), bool) else None,
            "minimum_non_query_sequence_length": source.get("minimum_non_query_sequence_length"),
            "manifests": _safe_manifests(source.get("manifests"), include_private_paths=include_private_paths),
            "raw_root": _safe_path(source.get("raw_root"), include_private_paths=include_private_paths),
            "processed_features_dirs": [
                _safe_path(str(path), include_private_paths=include_private_paths)
                for path in source.get("processed_features_dirs", [])
            ],
            "msa_names": source.get("msa_names"),
            "stats": {
                key: value for key, value in source_stats.items()
                if key in SOURCE_STAT_KEYS and isinstance(value, int) and not isinstance(value, bool)
            },
        },
        "heldout": {
            "manifests": _safe_manifests(heldout.get("manifests"), include_private_paths=include_private_paths),
            "unique_target_sequence_count": heldout.get("unique_target_sequence_count"),
        },
        "chain_data_cache": {
            "path": _safe_path(chain_data_cache.get("path"), include_private_paths=include_private_paths),
            "sha256": chain_data_cache.get("sha256"),
        },
    }
    if "batching" in mmseqs:
        batching = dict(mmseqs["batching"])
        if not include_private_paths:
            batching.pop("receipt_path")
            batching["receipt_sha256"] = "<private>"
        summary["mmseqs"]["batching"] = batching
    return summary


def _summarize_processed_audit(path: Path, *, include_private_paths: bool) -> dict[str, Any]:
    payload = _load_json(path)
    return {
        "path": _safe_path(str(path), include_private_paths=include_private_paths),
        "sha256": _sha256_file(path),
        "operation": payload.get("operation"),
        "out_features_dir": _safe_path(payload.get("out_features_dir"), include_private_paths=include_private_paths),
        "source_processed_features_dirs": [
            _safe_path(str(source_path), include_private_paths=include_private_paths)
            for source_path in payload.get("source_processed_features_dirs", [])
        ],
        "manifests": _safe_manifests(payload.get("manifests"), include_private_paths=include_private_paths),
        "chain_count": payload.get("chain_count"),
        "filtered_chain_count": payload.get("filtered_chain_count"),
        "missing_chain_count": payload.get("missing_chain_count"),
        "excluded_sequence_count": payload.get("excluded_sequence_count"),
        "msa_row_filter_sha256": payload.get("msa_row_filter_sha256"),
        "total_msa_rows_before_filter": payload.get("total_msa_rows_before_filter"),
        "total_msa_rows_after_filter": payload.get("total_msa_rows_after_filter"),
        "total_msa_rows_removed": payload.get("total_msa_rows_removed"),
        "per_chain_included": False,
    }


def _sum_int(records: Sequence[dict[str, Any]], key: str) -> int | None:
    values: list[int] = []
    for record in records:
        value = record.get(key)
        if value is None:
            continue
        values.append(int(value))
    return sum(values) if values else None


def build_source_lock(
    *,
    row_filter_path: Path,
    processed_filter_audit_paths: Sequence[Path],
    include_private_paths: bool = False,
) -> dict[str, Any]:
    filter_summary = _summarize_row_filter(row_filter_path, include_private_paths=include_private_paths)
    audit_summaries = [
        _summarize_processed_audit(path, include_private_paths=include_private_paths)
        for path in processed_filter_audit_paths
    ]
    if audit_summaries and audit_summaries[-1]["msa_row_filter_sha256"] != filter_summary["sha256"]:
        raise ValueError("Final processed-filter audit does not reference the supplied row-filter bytes")
    for audit in audit_summaries:
        before, after, removed = (audit[key] for key in (
            "total_msa_rows_before_filter", "total_msa_rows_after_filter", "total_msa_rows_removed"
        ))
        if not all(isinstance(value, int) and not isinstance(value, bool) and value >= 0 for value in (before, after, removed)):
            raise ValueError("Processed-filter audit must contain nonnegative integer row counts")
        if before - after != removed:
            raise ValueError("Processed-filter audit row counts are inconsistent")
    return {
        "schema_version": 1,
        "artifact_type": "msa_row_filter_source_lock",
        "public_safe": not include_private_paths,
        "validation": {
            "final_audit_filter_hash_checked": bool(audit_summaries),
            "residual_homology_audited": False,
            "scope": "Filtering counts and source provenance only; use audit_msa_split.py --strict "
            "--run-mmseqs --exhaustive on the final features for all-pair local-alignment checks of "
            "residual held-out target homology.",
        },
        "safety": {
            "raw_sequences_included": False,
            "excluded_sequence_sha256_included": False,
            "per_chain_processed_audit_included": False,
            "private_paths_redacted": not include_private_paths,
        },
        "msa_row_filter": filter_summary,
        "processed_feature_filter_passes": audit_summaries,
        "aggregate_processed_filtering": {
            "pass_count": len(audit_summaries),
            "cumulative_msa_rows_removed": _sum_int(audit_summaries, "total_msa_rows_removed"),
            "final_total_msa_rows_after_filter": (
                audit_summaries[-1].get("total_msa_rows_after_filter") if audit_summaries else None
            ),
            "final_filtered_chain_count": audit_summaries[-1].get("filtered_chain_count") if audit_summaries else None,
            "final_msa_row_filter_sha256": audit_summaries[-1].get("msa_row_filter_sha256") if audit_summaries else None,
        },
    }


def main() -> None:
    args = parse_args()
    payload = build_source_lock(
        row_filter_path=args.msa_row_filter,
        processed_filter_audit_paths=args.processed_filter_audit,
        include_private_paths=bool(args.include_private_paths),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"Wrote MSA row-filter source lock to {args.output}")


if __name__ == "__main__":
    main()
