from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Dict, List

"""Create train/val manifests from an OpenFold/OpenProteinSet chain_data_cache.

Maintainer-only workflow:
- use this script to generate/refresh official manifests and lock metadata.
- participants should use committed manifests via scripts/setup_official_data.sh.

Expected input:
- chain_data_cache.json (available from RODA as described in OpenFold docs)

We filter by:
- monomer chains
- length range
- resolution cutoff
and then sample fixed-size train/val chain sets with protein-disjoint splits:
- no PDB ID is allowed to appear in both train and val

NOTE: The exact schema of chain_data_cache.json can vary across OpenFold versions.
This script is intentionally conservative and will likely need small tweaks once you inspect the cache file.
"""


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chain-data-cache", type=str, required=True, help="Path to chain_data_cache.json")
    ap.add_argument("--out-dir", type=str, required=True, help="Where to write train.txt and val.txt")
    ap.add_argument("--train-size", type=int, default=10000)
    ap.add_argument("--val-size", type=int, default=1000)
    ap.add_argument("--min-len", type=int, default=40)
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--max-resolution", type=float, default=3.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--lock-file",
        type=str,
        default="",
        help="Optional JSON path capturing generation inputs/hashes for reproducibility.",
    )
    return ap.parse_args()


def _pdb_id(chain_id: str) -> str:
    if "_" not in chain_id:
        raise ValueError(f"Invalid chain id (expected <pdb>_<chain>): {chain_id}")
    return chain_id.split("_", 1)[0].lower()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    cache_path = Path(args.chain_data_cache)
    data = json.loads(cache_path.read_text())

    # Heuristic: cache might be a dict keyed by chain_id, with a nested dict of fields.
    # We'll try to pull:
    # - seq_length
    # - resolution
    # - oligomeric_count or similar
    # If fields missing, we keep the chain and let downstream preprocessing filter.
    candidates: List[str] = []
    for chain_id, meta in data.items():
        try:
            L = int(meta.get("seq_length", meta.get("sequence_length", -1)))
        except Exception:
            L = -1
        try:
            res = float(meta.get("resolution", 999.0))
        except Exception:
            res = 999.0

        # Oligomeric state isn't always present; if present, enforce monomer.
        olig = meta.get("oligomeric_count", meta.get("num_chains", 1))
        try:
            olig = int(olig)
        except Exception:
            olig = 1

        if L != -1 and not (args.min_len <= L <= args.max_len):
            continue
        if res != 999.0 and res > args.max_resolution:
            continue
        if olig != 1:
            continue

        candidates.append(chain_id)

    if len(candidates) < args.train_size + args.val_size:
        raise ValueError(f"Not enough candidates after filtering: {len(candidates)}")

    random.shuffle(candidates)

    # Sample val first, then restrict train to proteins not present in val.
    val_ids = candidates[: args.val_size]
    val_pdbs = {_pdb_id(cid) for cid in val_ids}

    train_pool = [cid for cid in candidates[args.val_size :] if _pdb_id(cid) not in val_pdbs]
    if len(train_pool) < args.train_size:
        raise ValueError(
            "Not enough protein-disjoint train candidates after selecting val: "
            f"need {args.train_size}, have {len(train_pool)}. "
            "Try reducing --val-size or adjusting filters."
        )

    random.shuffle(train_pool)
    train_ids = train_pool[: args.train_size]

    # Safety check to enforce split contract.
    train_pdbs = {_pdb_id(cid) for cid in train_ids}
    overlap = train_pdbs & val_pdbs
    if overlap:
        raise RuntimeError(f"Internal error: train/val protein overlap detected for {len(overlap)} PDB IDs")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_path = out_dir / "train.txt"
    val_path = out_dir / "val.txt"
    all_path = out_dir / "all.txt"
    train_path.write_text("\n".join(train_ids) + "\n")
    val_path.write_text("\n".join(val_ids) + "\n")
    all_ids = sorted(set(train_ids + val_ids))
    all_path.write_text("\n".join(all_ids) + "\n")

    print(
        f"Wrote {len(train_ids)} train + {len(val_ids)} val chains to {out_dir} "
        f"(protein-disjoint: {len(overlap)} overlaps)"
    )

    if args.lock_file:
        cache_sha = hashlib.sha256(cache_path.read_bytes()).hexdigest()
        lock = {
            "chain_data_cache_path": str(cache_path.resolve()),
            "chain_data_cache_sha256": cache_sha,
            "args": {
                "train_size": int(args.train_size),
                "val_size": int(args.val_size),
                "min_len": int(args.min_len),
                "max_len": int(args.max_len),
                "max_resolution": float(args.max_resolution),
                "seed": int(args.seed),
            },
            "outputs": {
                "train_manifest": str(train_path.resolve()),
                "val_manifest": str(val_path.resolve()),
                "all_manifest": str(all_path.resolve()),
                "train_count": len(train_ids),
                "val_count": len(val_ids),
                "unique_count": len(all_ids),
            },
        }
        lock_path = Path(args.lock_file)
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        lock_path.write_text(json.dumps(lock, indent=2) + "\n")
        print(f"Wrote manifest lock file: {lock_path.resolve()}")


if __name__ == "__main__":
    main()
