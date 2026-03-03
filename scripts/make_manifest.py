from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List

"""Create train/val manifests from an OpenFold/OpenProteinSet chain_data_cache.

This is *one* reasonable way to define a fixed 10k benchmark. You can replace this logic as you like,
but once you publish a competition, treat the resulting manifests as immutable.

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
    ap.add_argument("--val-size", type=int, default=500)
    ap.add_argument("--min-len", type=int, default=40)
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--max-resolution", type=float, default=3.0)
    ap.add_argument("--seed", type=int, default=0)
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
    (out_dir / "train.txt").write_text("\n".join(train_ids) + "\n")
    (out_dir / "val.txt").write_text("\n".join(val_ids) + "\n")

    print(
        f"Wrote {len(train_ids)} train + {len(val_ids)} val chains to {out_dir} "
        f"(protein-disjoint: {len(overlap)} overlaps)"
    )


if __name__ == "__main__":
    main()
