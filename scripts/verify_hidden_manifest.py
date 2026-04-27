from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from build_hidden_manifest import (  # noqa: E402
    _assert_public_manifests,
    _load_lock_args,
    _read_manifest,
    _with_cluster_ids,
)
from build_manifests import (  # noqa: E402
    DEFAULT_STRATIFY_FIELDS,
    _assert_quality_report,
    _build_candidates,
    _build_quality_report,
    _cluster_sequences_mmseqs,
    _load_cluster_tsv,
    _load_structure_metadata,
    _sha256,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Verify that maintainer-only hidden validation is disjoint from the fixed public "
            "manifests and respects the official structural stratification gates."
        )
    )
    parser.add_argument(
        "--chain-data-cache",
        type=Path,
        default=Path("data/openproteinset/pdb_data/data_caches/chain_data_cache.json"),
    )
    parser.add_argument("--lock-file", type=Path, default=Path("leaderboard/official_manifest_source.lock.json"))
    parser.add_argument("--public-train-manifest", type=Path, default=Path("data/manifests/train.txt"))
    parser.add_argument("--public-val-manifest", type=Path, default=Path("data/manifests/val.txt"))
    parser.add_argument("--public-all-manifest", type=Path, default=Path("data/manifests/all.txt"))
    parser.add_argument("--hidden-manifest", type=Path, default=Path(".nanofold_private/manifests/hidden_val.txt"))
    parser.add_argument(
        "--private-lock-file",
        type=Path,
        default=Path(".nanofold_private/leaderboard/private_hidden_manifest_source.lock.json"),
    )
    parser.add_argument("--structure-metadata", type=Path, default=None)
    parser.add_argument("--cluster-tsv", type=Path, default=None)
    parser.add_argument("--mmseqs-bin", type=str, default=None)
    parser.add_argument("--json-out", type=Path, default=None)
    return parser.parse_args()


def _load_private_lock(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    raw = json.loads(path.read_text())
    if not isinstance(raw, dict):
        raise ValueError(f"Private hidden manifest lock must be a JSON object: {path}")
    return raw


def _fail_if_nonempty(label: str, values: set[str]) -> None:
    if values:
        raise RuntimeError(f"{label} overlap is non-zero: {len(values)}")


def verify(args: argparse.Namespace) -> dict[str, Any]:
    lock_args = _load_lock_args(args.lock_file)
    structure_metadata_path = args.structure_metadata or Path(lock_args.structure_metadata)
    cluster_tsv = str(args.cluster_tsv if args.cluster_tsv is not None else lock_args.cluster_tsv)
    mmseqs_bin = str(args.mmseqs_bin if args.mmseqs_bin is not None else lock_args.mmseqs_bin)

    train_ids = _read_manifest(args.public_train_manifest)
    val_ids = _read_manifest(args.public_val_manifest)
    all_ids = _read_manifest(args.public_all_manifest)
    hidden_ids = _read_manifest(args.hidden_manifest)
    _assert_public_manifests(train_ids, val_ids, all_ids)

    if len(hidden_ids) != lock_args.hidden_val_size:
        raise ValueError(
            f"Hidden manifest count mismatch: expected {lock_args.hidden_val_size}, got {len(hidden_ids)}"
        )

    cache_sha = _sha256(args.chain_data_cache)
    expected_sha = lock_args.expected_chain_cache_sha256.strip().lower()
    if expected_sha and cache_sha != expected_sha:
        raise ValueError(
            "chain_data_cache SHA256 mismatch.\n"
            f"expected: {expected_sha}\n"
            f"actual:   {cache_sha}\n"
            f"path:     {args.chain_data_cache.resolve()}"
        )

    data = json.loads(args.chain_data_cache.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"chain_data_cache.json must be a JSON object: {args.chain_data_cache}")

    structure_metadata_accepted, structure_metadata = _load_structure_metadata(structure_metadata_path)
    candidates, _reject_reasons, _structural_metadata_source = _build_candidates(
        data,
        min_len=lock_args.min_len,
        max_len=lock_args.max_len,
        max_resolution=lock_args.max_resolution,
        structure_metadata=structure_metadata,
        structure_metadata_accepted=structure_metadata_accepted,
    )
    candidate_ids = {candidate.chain_id for candidate in candidates}
    required_ids = set(all_ids) | set(hidden_ids)
    missing = sorted(required_ids - candidate_ids)
    if missing:
        raise ValueError(f"{len(missing)} manifest chains are not eligible candidates. Examples: {missing[:8]}")

    candidate_sequences = {candidate.chain_id: candidate.sequence for candidate in candidates}
    if cluster_tsv:
        cluster_method = "precomputed_tsv"
        cluster_map = _load_cluster_tsv(cluster_tsv, candidate_ids)
    else:
        cluster_method = "mmseqs2"
        cluster_map_or_none, _cluster_command = _cluster_sequences_mmseqs(
            sequences={chain_id: candidate_sequences[chain_id] for chain_id in candidate_ids},
            mmseqs_bin=mmseqs_bin,
            min_seq_id=lock_args.min_seq_id,
            coverage=lock_args.coverage,
        )
        if cluster_map_or_none is None:
            raise RuntimeError(
                "Hidden manifest verification requires MMseqs2 clustering or a locked --cluster-tsv. "
                f"Install `{mmseqs_bin}` or provide a TSV generated with --min-seq-id "
                f"{lock_args.min_seq_id} and coverage {lock_args.coverage}."
            )
        cluster_map = cluster_map_or_none

    clustered_candidates = _with_cluster_ids(candidates, cluster_map)
    candidate_by_id = {candidate.chain_id: candidate for candidate in clustered_candidates}

    public_ids = set(all_ids)
    hidden_set = set(hidden_ids)
    chain_overlap = public_ids & hidden_set
    public_clusters = {candidate_by_id[chain_id].cluster_id for chain_id in public_ids}
    hidden_clusters = {candidate_by_id[chain_id].cluster_id for chain_id in hidden_set}
    cluster_overlap = public_clusters & hidden_clusters
    public_pdbs = {candidate_by_id[chain_id].pdb_id for chain_id in public_ids}
    hidden_pdbs = {candidate_by_id[chain_id].pdb_id for chain_id in hidden_set}
    pdb_overlap = public_pdbs & hidden_pdbs

    _fail_if_nonempty("Public/hidden chain", chain_overlap)
    _fail_if_nonempty("Public/hidden MMseqs2 cluster", cluster_overlap)
    _fail_if_nonempty("Public/hidden PDB entry", pdb_overlap)

    selected_candidates = {
        chain_id: candidate_by_id[chain_id]
        for chain_id in train_ids + val_ids + hidden_ids
    }
    quality_report = _build_quality_report(
        selected_candidates=selected_candidates,
        split_ids={"train": train_ids, "val": val_ids, "hidden_val": hidden_ids},
        stratify_fields=DEFAULT_STRATIFY_FIELDS,
    )
    _assert_quality_report(quality_report)

    private_lock = _load_private_lock(args.private_lock_file)
    if private_lock is not None:
        expected_hidden_sha = str(private_lock.get("hidden_val_manifest_sha256") or "")
        if expected_hidden_sha and expected_hidden_sha != _sha256(args.hidden_manifest):
            raise ValueError("Private hidden manifest lock hash does not match hidden manifest.")
        expected_count = int(private_lock.get("hidden_val_count", len(hidden_ids)))
        if expected_count != len(hidden_ids):
            raise ValueError("Private hidden manifest lock count does not match hidden manifest.")
        public_contract = private_lock.get("public_contract", {})
        if isinstance(public_contract, dict):
            expected_all_sha = str(public_contract.get("all_manifest_sha256") or "")
            if expected_all_sha and expected_all_sha != _sha256(args.public_all_manifest):
                raise ValueError("Private hidden manifest lock public all.txt hash does not match.")

    js = quality_report["jensen_shannon_divergence"]
    max_js = float(quality_report["observed"]["max_split_js_divergence"])
    return {
        "status": "ok",
        "counts": {
            "train": len(train_ids),
            "val": len(val_ids),
            "hidden_val": len(hidden_ids),
        },
        "overlap": {
            "chains": 0,
            "mmseqs2_clusters": 0,
            "pdb_entries": 0,
        },
        "clustering": {
            "method": cluster_method,
            "cluster_count": len(set(cluster_map.values())),
        },
        "stratification": {
            "fields": list(DEFAULT_STRATIFY_FIELDS),
            "max_split_js_divergence": max_js,
            "jensen_shannon_divergence": js,
        },
        "manifest_sha256": {
            "train": _sha256(args.public_train_manifest),
            "val": _sha256(args.public_val_manifest),
            "all": _sha256(args.public_all_manifest),
            "hidden_val": _sha256(args.hidden_manifest),
        },
    }


def main() -> None:
    args = parse_args()
    result = verify(args)
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")

    js = result["stratification"]["jensen_shannon_divergence"]
    print("Hidden manifest verification passed.")
    print(
        "Counts: "
        f"train={result['counts']['train']} "
        f"val={result['counts']['val']} "
        f"hidden_val={result['counts']['hidden_val']}"
    )
    print("Overlap: chains=0 mmseqs2_clusters=0 pdb_entries=0")
    print(f"Max JS divergence: {result['stratification']['max_split_js_divergence']:.8g}")
    print("Hidden JS divergence by field:")
    for field_name, value in js["hidden_val"].items():
        print(f"- {field_name}: {float(value):.8g}")


if __name__ == "__main__":
    main()
