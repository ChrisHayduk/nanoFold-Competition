"""Build deterministic MSA-row filters against held-out target homologs.

The output JSON contains only SHA256 hashes of ungapped MSA-row sequences that
should be removed during preprocessing. It intentionally does not store raw
sequences or held-out chain IDs, so the same format can be used in private
maintainer workflows without exposing hidden identifiers.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from nanofold.a3m import GAP_ID, MASK_ID, RESTYPES, read_a3m, ungap_query_columns
from nanofold.chain_paths import chain_data_dir, chain_npz_path
from nanofold.mmseqs import canonical_target_sequence, iter_mmseqs_hits, load_msa_for_audit, validate_search_parameters

MSA_TOKEN_TO_AA = {idx: aa for idx, aa in enumerate(RESTYPES)}
MSA_TOKEN_TO_AA[20] = "X"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--source-manifest",
        action="append",
        required=True,
        help="Manifest whose MSA rows should be scanned. Repeat for multiple source splits.",
    )
    ap.add_argument(
        "--heldout-manifest",
        action="append",
        required=True,
        help="Held-out target manifest. MSA rows homologous to these targets are filtered.",
    )
    ap.add_argument("--chain-data-cache", type=Path, required=True)
    ap.add_argument("--raw-root", type=Path, default=Path("data/openproteinset"))
    ap.add_argument(
        "--source-processed-features-dir",
        type=Path,
        action="append",
        default=None,
        help=(
            "Optional processed feature directory to scan instead of raw A3M files. "
            "Repeat to scan the union of all matching source feature variants across roots. "
            "Use raw A3Ms for official regeneration; processed features are useful for audits."
        ),
    )
    ap.add_argument(
        "--allow-missing-source-msas",
        action="store_true",
        help="Allow missing raw MSA files while scanning source manifests. Official refreshes should leave this off.",
    )
    ap.add_argument(
        "--allow-missing-source-features",
        action="store_true",
        help="Allow missing processed feature NPZs when --source-processed-features-dir is used.",
    )
    ap.add_argument("--msa-name", default="uniref90_hits.a3m")
    ap.add_argument(
        "--msa-names",
        default="",
        help="Comma-separated MSA names to scan. Defaults to --msa-name.",
    )
    ap.add_argument("--min-seq-id", type=float, default=0.30)
    ap.add_argument("--coverage", type=float, default=0.80)
    ap.add_argument("--mmseqs-bin", default="mmseqs")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument(
        "--max-seqs",
        type=int,
        default=1_000_000,
        help="MMseqs max reported hits per held-out target. Keep high for filtering.",
    )
    ap.add_argument("--tmp-dir", type=Path, default=None)
    ap.add_argument("--output", type=Path, required=True)
    return ap.parse_args()


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sequence_sha256(sequence: str) -> str:
    return hashlib.sha256(sequence.upper().encode("utf-8")).hexdigest()


def _read_manifest(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text().splitlines() if line.strip() and not line.lstrip().startswith("#")]


def _manifest_record(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "sha256": _sha256_file(path),
        "chain_count": len(_read_manifest(path)),
    }


def _load_chain_data_cache(path: Path) -> dict[str, Any]:
    raw = json.loads(path.read_text())
    if not isinstance(raw, dict):
        raise ValueError(f"chain_data_cache must be a JSON object: {path}")
    return raw


def _chain_sequence(cache: dict[str, Any], chain_id: str) -> str:
    raw = cache.get(chain_id)
    if not isinstance(raw, dict):
        raise KeyError(f"Missing chain in cache: {chain_id}")
    for key in ("sequence", "seq", "seqres", "aatype_sequence"):
        value = raw.get(key)
        if isinstance(value, str) and value.strip():
            return canonical_target_sequence(value)
    raise KeyError(f"Missing sequence in chain_data_cache for {chain_id}")


def _resolve_msa_names(msa_name: str, msa_names: str | Sequence[str] | None = None) -> tuple[str, ...]:
    if msa_names is None:
        return (msa_name,)
    if isinstance(msa_names, str):
        parsed = tuple(token.strip() for token in msa_names.split(",") if token.strip())
    else:
        parsed = tuple(token.strip() for token in msa_names if token.strip())
    return parsed or (msa_name,)


def _find_msa_path(chain_dir: Path, msa_name: str) -> Path | None:
    for candidate in (chain_dir / msa_name, chain_dir / "a3m" / msa_name):
        if candidate.exists():
            return candidate
    for path in chain_dir.rglob(msa_name):
        return path
    return None


def _msa_row_to_ungapped_sequence(row: np.ndarray) -> str:
    chars: list[str] = []
    for value in row.tolist():
        token = int(value)
        if token in (GAP_ID, MASK_ID):
            continue
        chars.append(MSA_TOKEN_TO_AA.get(token, "X"))
    return "".join(chars)


def _add_sequence(sequences_by_hash: dict[str, str], sequence: str) -> None:
    sequence = sequence.upper()
    if len(sequence) < 20:
        return
    digest = _sequence_sha256(sequence)
    sequences_by_hash.setdefault(digest, sequence)


def _scan_processed_msa_rows(
    *,
    manifest_paths: Sequence[Path],
    features_dirs: Sequence[Path],
) -> tuple[dict[str, str], dict[str, int]]:
    sequences_by_hash: dict[str, str] = {}
    stats = {"chains": 0, "feature_variants": 0, "missing_features": 0, "rows": 0, "non_query_rows": 0}
    seen_chains: set[str] = set()
    for manifest_path in manifest_paths:
        for chain_id in _read_manifest(manifest_path):
            if chain_id in seen_chains:
                continue
            seen_chains.add(chain_id)
            feature_paths = list(dict.fromkeys(
                chain_npz_path(features_dir, chain_id).resolve()
                for features_dir in features_dirs if chain_npz_path(features_dir, chain_id).exists()
            ))
            if not feature_paths:
                stats["missing_features"] += 1
                continue
            stats["chains"] += 1
            for feature_path in feature_paths:
                msa = load_msa_for_audit(feature_path, chain_id=chain_id)
                stats["feature_variants"] += 1
                stats["rows"] += int(msa.shape[0])
                for row in msa[1:]:
                    stats["non_query_rows"] += 1
                    _add_sequence(sequences_by_hash, _msa_row_to_ungapped_sequence(row))
    stats["unique_non_query_rows"] = len(sequences_by_hash)
    return sequences_by_hash, stats


def _scan_raw_msa_rows(
    *,
    manifest_paths: Sequence[Path],
    raw_root: Path,
    msa_names: Sequence[str],
) -> tuple[dict[str, str], dict[str, int]]:
    sequences_by_hash: dict[str, str] = {}
    stats = {"chains": 0, "missing_msa_files": 0, "rows": 0, "non_query_rows": 0}
    roda_root = raw_root / "roda_pdb"
    for manifest_path in manifest_paths:
        for chain_id in _read_manifest(manifest_path):
            chain_dir = chain_data_dir(roda_root, chain_id)
            loaded = False
            for msa_name in msa_names:
                msa_path = _find_msa_path(chain_dir, msa_name)
                if msa_path is None:
                    stats["missing_msa_files"] += 1
                    continue
                loaded = True
                a3m = read_a3m(msa_path)
                msa, deletions = a3m.to_tokens(max_seqs=None)
                aligned_msa, _ = a3m.to_aligned_msa()
                msa, _deletions, _query = ungap_query_columns(
                    msa=msa,
                    deletions=deletions,
                    query_aligned=aligned_msa[0],
                )
                stats["rows"] += int(msa.shape[0])
                for row in msa[1:]:
                    stats["non_query_rows"] += 1
                    _add_sequence(sequences_by_hash, _msa_row_to_ungapped_sequence(row))
            if loaded:
                stats["chains"] += 1
    stats["unique_non_query_rows"] = len(sequences_by_hash)
    return sequences_by_hash, stats


def _write_heldout_fasta(path: Path, *, manifest_paths: Sequence[Path], cache: dict[str, Any]) -> int:
    seen: set[str] = set()
    with path.open("w") as handle:
        for manifest_path in manifest_paths:
            for chain_id in _read_manifest(manifest_path):
                sequence = _chain_sequence(cache, chain_id)
                digest = _sequence_sha256(sequence)
                if digest in seen:
                    continue
                seen.add(digest)
                handle.write(f">heldout_{len(seen)}\n{sequence}\n")
    return len(seen)


def _write_source_fasta(path: Path, sequences_by_hash: dict[str, str]) -> None:
    with path.open("w") as handle:
        for digest, sequence in sorted(sequences_by_hash.items()):
            handle.write(f">seq_{digest}\n{sequence}\n")


def _mmseqs_version(mmseqs_bin: str) -> str | None:
    proc = subprocess.run([mmseqs_bin, "version"], capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        return None
    return proc.stdout.strip() or None


def _run_mmseqs(
    *,
    mmseqs_bin: str,
    heldout_fasta: Path,
    source_fasta: Path,
    output_m8: Path,
    tmp_dir: Path,
    min_seq_id: float,
    coverage: float,
    threads: int,
    max_seqs: int,
) -> list[str]:
    validate_search_parameters(min_seq_id=min_seq_id, coverage=coverage, threads=threads, max_seqs=max_seqs)
    output_m8.unlink(missing_ok=True)
    cmd = [
        mmseqs_bin,
        "easy-search",
        str(heldout_fasta),
        str(source_fasta),
        str(output_m8),
        str(tmp_dir),
        "--min-seq-id",
        str(min_seq_id),
        "-c",
        str(coverage),
        "--cov-mode",
        "0",
        "--alignment-mode",
        "3",
        "--seq-id-mode",
        "0",
        "--search-type", "1",
        "--prefilter-mode", "0",
        "-e", "0.001",
        "-s", "5.7",
        "--mask", "1",
        "--comp-bias-corr", "1",
        "--threads",
        str(threads),
        "--max-seqs",
        str(max_seqs),
        "--format-output",
        "query,target,pident,alnlen,qcov,tcov,evalue,bits",
    ]
    proc = subprocess.run(cmd, text=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"MMseqs failed with exit code {proc.returncode}: {' '.join(cmd)}")
    if not output_m8.is_file():
        raise RuntimeError("MMseqs completed without writing the expected alignment output")
    return cmd


def _parse_excluded_hashes(
    output_m8: Path, *, query_ids: set[str], source_hashes: set[str], min_seq_id: float, coverage: float,
) -> tuple[set[str], int]:
    excluded: set[str] = set()
    hit_count = 0
    for _query, target in iter_mmseqs_hits(
        output_m8, query_ids=query_ids, target_ids={f"seq_{digest}" for digest in source_hashes},
        min_seq_id=min_seq_id, coverage=coverage,
    ):
        excluded.add(target.removeprefix("seq_"))
        hit_count += 1
    return excluded, hit_count


def main() -> None:
    args = parse_args()
    validate_search_parameters(
        min_seq_id=args.min_seq_id, coverage=args.coverage, threads=args.threads, max_seqs=args.max_seqs,
    )
    mmseqs_path = shutil.which(args.mmseqs_bin)
    if mmseqs_path is None:
        raise SystemExit(f"MMseqs binary not found: {args.mmseqs_bin}")

    source_manifests = [Path(path) for path in args.source_manifest]
    heldout_manifests = [Path(path) for path in args.heldout_manifest]
    cache = _load_chain_data_cache(args.chain_data_cache)
    resolved_msa_names = _resolve_msa_names(args.msa_name, args.msa_names)

    source_processed_features_dirs = tuple(args.source_processed_features_dir or ())
    if source_processed_features_dirs:
        sequences_by_hash, source_stats = _scan_processed_msa_rows(
            manifest_paths=source_manifests,
            features_dirs=source_processed_features_dirs,
        )
        source_mode = "processed_features"
        if source_stats["missing_features"] and not args.allow_missing_source_features:
            raise SystemExit(
                "Missing processed feature NPZs while building MSA row filter: "
                f"{source_stats['missing_features']} "
                "(pass --allow-missing-source-features only for exploratory audits)."
            )
    else:
        sequences_by_hash, source_stats = _scan_raw_msa_rows(
            manifest_paths=source_manifests,
            raw_root=args.raw_root,
            msa_names=resolved_msa_names,
        )
        source_mode = "raw_a3m"
        if source_stats["missing_msa_files"] and not args.allow_missing_source_msas:
            raise SystemExit(
                "Missing raw MSA files while building MSA row filter: "
                f"{source_stats['missing_msa_files']} "
                "(pass --allow-missing-source-msas only for exploratory audits)."
            )
    if not sequences_by_hash:
        raise SystemExit("No source MSA rows were found for filtering.")

    work_parent = args.tmp_dir
    with tempfile.TemporaryDirectory(prefix="nanofold_msa_row_filter_", dir=str(work_parent) if work_parent else None) as tmp:
        tmp_path = Path(tmp)
        heldout_fasta = tmp_path / "heldout_targets.fasta"
        source_fasta = tmp_path / "source_msa_rows.fasta"
        output_m8 = tmp_path / "heldout_vs_source_msa_rows.m8"
        mmseqs_tmp = tmp_path / "mmseqs_tmp"
        heldout_unique_count = _write_heldout_fasta(heldout_fasta, manifest_paths=heldout_manifests, cache=cache)
        if heldout_unique_count == 0:
            raise ValueError("Held-out manifests must contain target sequences")
        _write_source_fasta(source_fasta, sequences_by_hash)
        cmd = _run_mmseqs(
            mmseqs_bin=args.mmseqs_bin,
            heldout_fasta=heldout_fasta,
            source_fasta=source_fasta,
            output_m8=output_m8,
            tmp_dir=mmseqs_tmp,
            min_seq_id=float(args.min_seq_id),
            coverage=float(args.coverage),
            threads=int(args.threads),
            max_seqs=int(args.max_seqs),
        )
        excluded, hit_count = _parse_excluded_hashes(
            output_m8,
            query_ids={f"heldout_{index}" for index in range(1, heldout_unique_count + 1)},
            source_hashes=set(sequences_by_hash), min_seq_id=args.min_seq_id, coverage=args.coverage,
        )

    # Composition masking and unknown residues can hide even identical rows from
    # MMseqs. Exact filtering must be independent of heuristic search sensitivity.
    heldout_hashes = {
        _sequence_sha256(_chain_sequence(cache, chain_id))
        for manifest in heldout_manifests for chain_id in _read_manifest(manifest)
    }
    exact_excluded = set(sequences_by_hash) & heldout_hashes
    excluded.update(exact_excluded)

    payload = {
        "schema_version": 1,
        "filter_type": "heldout_target_homology",
        "threshold": {
            "min_seq_id": float(args.min_seq_id),
            "coverage": float(args.coverage),
            "cov_mode": 0,
            "alignment_mode": 3,
            "seq_id_mode": 0,
        },
        "mmseqs": {
            "binary": args.mmseqs_bin,
            "version": _mmseqs_version(args.mmseqs_bin),
            "command": cmd,
            "max_seqs": int(args.max_seqs),
            "search_parameters": {"prefilter_mode": 0, "evalue": 0.001, "sensitivity": 5.7,
                                  "mask": 1, "comp_bias_corr": 1, "search_type": 1},
            "hit_count": hit_count,
        },
        "source": {
            "mode": source_mode,
            "manifests": [_manifest_record(path) for path in source_manifests],
            "raw_root": str(args.raw_root) if source_mode == "raw_a3m" else None,
            "processed_features_dir": str(source_processed_features_dirs[0]) if source_processed_features_dirs else None,
            "processed_features_dirs": [str(path) for path in source_processed_features_dirs],
            "msa_names": list(resolved_msa_names),
            "stats": source_stats,
        },
        "heldout": {
            "manifests": [_manifest_record(path) for path in heldout_manifests],
            "unique_target_sequence_count": heldout_unique_count,
        },
        "chain_data_cache": {
            "path": str(args.chain_data_cache),
            "sha256": _sha256_file(args.chain_data_cache),
        },
        "exact_match_excluded_sequence_count": len(exact_excluded),
        "excluded_sequence_count": len(excluded),
        "excluded_sequence_sha256": sorted(excluded),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"Wrote MSA row filter with {len(excluded)} excluded sequence hashes to {args.output}")


if __name__ == "__main__":
    main()
