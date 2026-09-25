"""Apply an MSA-row filter to already processed feature NPZs.

This is a maintainer/rebuttal utility for existing processed features. The
preferred official data-refresh path is ``scripts/preprocess.py
--msa-filter-policy`` for the public split union, with separate unfiltered hidden
preprocessing. Filtering raw A3M rows before depth capping can backfill with
deeper permitted rows. This script accepts features without raw preprocessing provenance and preserves every feature
array except ``msa`` and ``deletions`` and is useful when raw alignments are
incomplete locally or when a fast sanitized-feature root is needed for audits or
experimental reruns.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from nanofold.a3m import GAP_ID, MASK_ID, RESTYPES
from nanofold.chain_paths import chain_npz_path
from nanofold.data import read_manifest
from nanofold.mmseqs import validate_msa_arrays

MSA_TOKEN_TO_AA = {idx: aa for idx, aa in enumerate(RESTYPES)}
MSA_TOKEN_TO_AA[20] = "X"


@dataclass(frozen=True)
class MSARowFilter:
    path: Path
    sha256: str
    excluded_sequence_sha256: frozenset[str]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--manifest",
        action="append",
        required=True,
        help="Manifest to filter. Repeat to process multiple splits into one output feature root.",
    )
    ap.add_argument(
        "--source-features-dir",
        type=Path,
        action="append",
        required=True,
        help="Processed feature root to read from. Repeat for disjoint roots; conflicting chain variants are rejected.",
    )
    ap.add_argument("--out-features-dir", type=Path, required=True)
    ap.add_argument("--msa-row-filter", type=Path, required=True)
    ap.add_argument(
        "--allow-missing",
        action="store_true",
        help="Record missing source feature NPZs instead of failing.",
    )
    ap.add_argument(
        "--skip-existing",
        action="store_true",
        help="Reuse validated output NPZs only when both source bytes and filter digest match.",
    )
    ap.add_argument(
        "--no-preprocess-meta",
        action="store_true",
        help="Do not write preprocess_meta.json into the output feature root.",
    )
    ap.add_argument(
        "--audit-name",
        default="processed_msa_row_filter_audit.json",
        help="Name of the JSON audit file written in --out-features-dir.",
    )
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


def _load_msa_row_filter(path: Path) -> MSARowFilter:
    raw = json.loads(path.read_text())
    if not isinstance(raw, dict):
        raise ValueError(f"MSA row filter must be a JSON object: {path}")
    excluded_raw = raw.get("excluded_sequence_sha256")
    if not isinstance(excluded_raw, list):
        raise ValueError(f"MSA row filter missing `excluded_sequence_sha256` list: {path}")
    excluded: set[str] = set()
    for value in excluded_raw:
        if not isinstance(value, str) or re.fullmatch(r"[0-9a-fA-F]{64}", value) is None:
            raise ValueError(f"Bad sequence SHA256 in MSA row filter {path}: {value!r}")
        excluded.add(value.lower())
    return MSARowFilter(path=path, sha256=_sha256_file(path), excluded_sequence_sha256=frozenset(excluded))


def _msa_row_to_ungapped_sequence(row: np.ndarray) -> str:
    chars: list[str] = []
    for value in row.tolist():
        token = int(value)
        if token in (GAP_ID, MASK_ID):
            continue
        chars.append(MSA_TOKEN_TO_AA.get(token, "X"))
    return "".join(chars)


def _find_feature_path(source_features_dirs: Sequence[Path], chain_id: str) -> Path | None:
    candidates = list(dict.fromkeys(
        chain_npz_path(root, chain_id).resolve()
        for root in source_features_dirs if chain_npz_path(root, chain_id).exists()
    ))
    if len(candidates) > 1 and len({_sha256_file(path) for path in candidates}) != 1:
        raise ValueError("Multiple different source feature variants require separate output roots")
    return candidates[0] if candidates else None


def _filter_arrays(
    arrays: dict[str, np.ndarray],
    row_filter: MSARowFilter,
) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    if "msa" not in arrays or "deletions" not in arrays:
        missing = [key for key in ("msa", "deletions") if key not in arrays]
        raise KeyError(f"Feature NPZ is missing required MSA keys: {', '.join(missing)}")

    if any(key in arrays for key in ("preprocess_config_sha256", "source_inputs_sha256")):
        raise ValueError("Provenance-bound features must be filtered by re-running preprocess.py from raw inputs")
    msa = validate_msa_arrays(arrays)
    deletions = np.asarray(arrays["deletions"])

    keep_indices = [0]
    for row_index, row in enumerate(msa[1:], start=1):
        sequence = _msa_row_to_ungapped_sequence(row)
        if _sequence_sha256(sequence) not in row_filter.excluded_sequence_sha256:
            keep_indices.append(row_index)

    keep = np.asarray(keep_indices, dtype=np.int64)
    removed = int(msa.shape[0]) - int(keep.shape[0])
    filtered = dict(arrays)
    filtered["msa"] = msa[keep]
    filtered["deletions"] = deletions[keep]
    filtered["msa_row_filter_sha256"] = np.asarray(row_filter.sha256)
    filtered["msa_rows_before_filter"] = np.asarray(int(msa.shape[0]), dtype=np.int32)
    filtered["msa_rows_removed_by_filter"] = np.asarray(removed, dtype=np.int32)
    return filtered, {
        "input_rows": int(msa.shape[0]),
        "removed_rows": removed,
        "output_rows": int(filtered["msa"].shape[0]),
    }


def _load_feature_arrays(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: np.array(data[key]) for key in data.files}


def _filter_one_chain(
    *,
    chain_id: str,
    source_features_dirs: Sequence[Path],
    out_features_dir: Path,
    row_filter: MSARowFilter,
    skip_existing: bool,
) -> dict[str, Any]:
    source_path = _find_feature_path(source_features_dirs, chain_id)
    if source_path is None:
        raise FileNotFoundError(f"Missing processed feature NPZ for {chain_id}")

    out_path = chain_npz_path(out_features_dir, chain_id)
    if out_path.resolve() == source_path.resolve():
        raise ValueError("Processed MSA filtering requires a separate output feature root")
    arrays = _load_feature_arrays(source_path)
    filtered, stats = _filter_arrays(arrays, row_filter)
    source_sha256 = _sha256_file(source_path)
    filtered["msa_row_filter_source_sha256"] = np.asarray(source_sha256)
    if skip_existing and out_path.exists():
        existing = _load_feature_arrays(out_path)
        validate_msa_arrays(existing)
        # Compare the complete expected transformation, including original non-MSA
        # arrays, so a modified output cannot pass on self-declared metadata alone.
        if existing.keys() == filtered.keys() and all(
            np.array_equal(existing[key], value) for key, value in filtered.items()
        ):
            return {
                "chain_id": chain_id, "source_path": str(source_path), "output_path": str(out_path),
                **stats, "status": "skipped_existing",
            }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{out_path.name}.", suffix=".tmp", dir=out_path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            np.savez_compressed(handle, **filtered)  # type: ignore[arg-type]
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, out_path)
    finally:
        Path(tmp_name).unlink(missing_ok=True)
    return {
        "chain_id": chain_id, "source_path": str(source_path), "output_path": str(out_path),
        **stats, "status": "written",
    }


def _write_preprocess_meta(
    *,
    out_features_dir: Path,
    manifest_paths: Sequence[Path],
    source_features_dirs: Sequence[Path],
    row_filter: MSARowFilter,
) -> None:
    payload = {
        "schema_version": 1,
        "operation": "processed_feature_msa_row_filter",
        "cli_args": {
            "msa_row_filter_sha256": row_filter.sha256,
        },
        "source_processed_features_dirs": [str(path) for path in source_features_dirs],
        "manifests": [_manifest_record(path) for path in manifest_paths],
        "msa_row_filter": {
            "path": str(row_filter.path),
            "sha256": row_filter.sha256,
            "excluded_sequence_count": len(row_filter.excluded_sequence_sha256),
        },
    }
    (out_features_dir / "preprocess_meta.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def filter_processed_features(
    *,
    manifest_paths: Sequence[Path],
    source_features_dirs: Sequence[Path],
    out_features_dir: Path,
    row_filter: MSARowFilter,
    allow_missing: bool = False,
    skip_existing: bool = False,
    write_preprocess_meta: bool = True,
    audit_name: str = "processed_msa_row_filter_audit.json",
) -> dict[str, Any]:
    if any(out_features_dir.resolve() == source.resolve() for source in source_features_dirs):
        raise ValueError("Processed MSA filtering requires a separate output feature root")
    if Path(audit_name).name != audit_name or audit_name in {"", ".", "..", "preprocess_meta.json"}:
        raise ValueError("Audit name must be a distinct filename inside the output feature root")
    out_features_dir.mkdir(parents=True, exist_ok=True)
    per_chain: list[dict[str, Any]] = []
    missing_chain_ids: list[str] = []

    seen: set[str] = set()
    chain_ids: list[str] = []
    for manifest_path in manifest_paths:
        for chain_id in read_manifest(manifest_path):
            if chain_id not in seen:
                seen.add(chain_id)
                chain_ids.append(chain_id)

    for chain_id in chain_ids:
        try:
            per_chain.append(
                _filter_one_chain(
                    chain_id=chain_id,
                    source_features_dirs=source_features_dirs,
                    out_features_dir=out_features_dir,
                    row_filter=row_filter,
                    skip_existing=skip_existing,
                )
            )
        except FileNotFoundError:
            missing_chain_ids.append(chain_id)
            if not allow_missing:
                raise

    total_input = sum(int(row["input_rows"]) for row in per_chain)
    total_removed = sum(int(row["removed_rows"]) for row in per_chain)
    total_output = sum(int(row["output_rows"]) for row in per_chain)
    payload = {
        "schema_version": 1,
        "operation": "processed_feature_msa_row_filter",
        "msa_row_filter_path": str(row_filter.path),
        "msa_row_filter_sha256": row_filter.sha256,
        "excluded_sequence_count": len(row_filter.excluded_sequence_sha256),
        "source_processed_features_dirs": [str(path) for path in source_features_dirs],
        "out_features_dir": str(out_features_dir),
        "manifests": [_manifest_record(path) for path in manifest_paths],
        "chain_count": len(chain_ids),
        "filtered_chain_count": len(per_chain),
        "missing_chain_count": len(missing_chain_ids),
        "missing_chain_ids": missing_chain_ids,
        "total_msa_rows_before_filter": total_input,
        "total_msa_rows_removed": total_removed,
        "total_msa_rows_after_filter": total_output,
        "per_chain": per_chain,
    }
    (out_features_dir / audit_name).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    if write_preprocess_meta:
        _write_preprocess_meta(
            out_features_dir=out_features_dir,
            manifest_paths=manifest_paths,
            source_features_dirs=source_features_dirs,
            row_filter=row_filter,
        )
    return payload


def main() -> None:
    args = parse_args()
    row_filter = _load_msa_row_filter(args.msa_row_filter)
    payload = filter_processed_features(
        manifest_paths=[Path(path) for path in args.manifest],
        source_features_dirs=[Path(path) for path in args.source_features_dir],
        out_features_dir=args.out_features_dir,
        row_filter=row_filter,
        allow_missing=bool(args.allow_missing),
        skip_existing=bool(args.skip_existing),
        write_preprocess_meta=not bool(args.no_preprocess_meta),
        audit_name=str(args.audit_name),
    )
    print(
        "Filtered processed MSA features: "
        f"{payload['filtered_chain_count']}/{payload['chain_count']} chains, "
        f"removed {payload['total_msa_rows_removed']} rows -> {args.out_features_dir}"
    )


if __name__ == "__main__":
    main()
