from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import random
import shutil
import subprocess
import tempfile
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
and then sample fixed-size train/val chain sets with homology-disjoint splits:
- no sequence cluster is allowed to appear in both train and val

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
    ap.add_argument("--min-seq-id", type=float, default=0.30)
    ap.add_argument("--coverage", type=float, default=0.80)
    ap.add_argument("--mmseqs-bin", type=str, default="mmseqs")
    ap.add_argument(
        "--require-mmseqs",
        action="store_true",
        help="Require MMseqs2 clustering instead of falling back to internal heuristics.",
    )
    ap.add_argument(
        "--expected-chain-cache-sha256",
        type=str,
        default="",
        help=(
            "Optional expected SHA256 of chain_data_cache.json. "
            "If provided, generation fails on mismatch."
        ),
    )
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


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _extract_sequence(meta: Dict[str, object]) -> str | None:
    for key in ("sequence", "seq", "seqres", "aatype_sequence"):
        value = meta.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip().upper()
    return None


def _sequence_cluster_match(seq_a: str, seq_b: str, *, min_seq_id: float, coverage: float) -> bool:
    if not seq_a or not seq_b:
        return False
    matcher = difflib.SequenceMatcher(a=seq_a, b=seq_b, autojunk=False)
    matching_blocks = matcher.get_matching_blocks()
    matches = sum(block.size for block in matching_blocks)
    aligned_pairs = min(len(seq_a), len(seq_b))
    if aligned_pairs <= 0:
        return False
    seq_id = matches / float(aligned_pairs)
    aligned_coverage = aligned_pairs / float(max(1, min(len(seq_a), len(seq_b))))
    return seq_id >= float(min_seq_id) and aligned_coverage >= float(coverage)


def _cluster_sequences_internal(
    *,
    sequences: Dict[str, str],
    min_seq_id: float,
    coverage: float,
) -> Dict[str, str]:
    chain_ids = sorted(sequences.keys())
    parent = {cid: cid for cid in chain_ids}

    def find(cid: str) -> str:
        while parent[cid] != cid:
            parent[cid] = parent[parent[cid]]
            cid = parent[cid]
        return cid

    def union(a: str, b: str) -> None:
        ra = find(a)
        rb = find(b)
        if ra != rb:
            parent[rb] = ra

    for i, cid_a in enumerate(chain_ids):
        for cid_b in chain_ids[i + 1 :]:
            if _sequence_cluster_match(
                sequences[cid_a],
                sequences[cid_b],
                min_seq_id=min_seq_id,
                coverage=coverage,
            ):
                union(cid_a, cid_b)

    return {cid: find(cid) for cid in chain_ids}


def _cluster_sequences_mmseqs(
    *,
    sequences: Dict[str, str],
    mmseqs_bin: str,
    min_seq_id: float,
    coverage: float,
) -> tuple[Dict[str, str] | None, List[str] | None]:
    mmseqs_path = shutil.which(mmseqs_bin)
    if mmseqs_path is None:
        return None, None

    with tempfile.TemporaryDirectory(prefix="nanofold_mmseqs_") as tmp_str:
        tmp_dir = Path(tmp_str)
        fasta_path = tmp_dir / "chains.fasta"
        fasta_lines: List[str] = []
        for chain_id, seq in sorted(sequences.items()):
            fasta_lines.append(f">{chain_id}")
            fasta_lines.append(seq)
        fasta_path.write_text("\n".join(fasta_lines) + "\n")

        cluster_prefix = tmp_dir / "clusters"
        work_dir = tmp_dir / "work"
        cmd = [
            mmseqs_path,
            "easy-cluster",
            str(fasta_path),
            str(cluster_prefix),
            str(work_dir),
            "--min-seq-id",
            str(min_seq_id),
            "-c",
            str(coverage),
            "--cov-mode",
            "0",
            "--threads",
            "1",
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if proc.returncode != 0:
            return None, cmd

        cluster_tsv = Path(str(cluster_prefix) + "_cluster.tsv")
        if not cluster_tsv.exists():
            return None, cmd

        cluster_map: Dict[str, str] = {}
        for line in cluster_tsv.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            rep, member = line.split("\t")
            cluster_map[member] = rep
        if set(cluster_map.keys()) != set(sequences.keys()):
            return None, cmd
        return cluster_map, cmd


def _cluster_assignments_sha256(cluster_map: Dict[str, str]) -> str:
    hasher = hashlib.sha256()
    for member, representative in sorted(cluster_map.items()):
        hasher.update(member.encode("utf-8"))
        hasher.update(b"\t")
        hasher.update(representative.encode("utf-8"))
        hasher.update(b"\n")
    return hasher.hexdigest()


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    cache_path = Path(args.chain_data_cache)
    cache_sha = _sha256(cache_path)
    expected_cache_sha = args.expected_chain_cache_sha256.strip().lower()
    if expected_cache_sha:
        if len(expected_cache_sha) != 64 or any(ch not in "0123456789abcdef" for ch in expected_cache_sha):
            raise ValueError("--expected-chain-cache-sha256 must be a 64-char lowercase hex digest.")
        if cache_sha != expected_cache_sha:
            raise ValueError(
                "chain_data_cache SHA256 mismatch.\n"
                f"expected: {expected_cache_sha}\n"
                f"actual:   {cache_sha}\n"
                f"path:     {cache_path.resolve()}"
            )

    data = json.loads(cache_path.read_text())
    if not isinstance(data, dict):
        raise ValueError(
            "chain_data_cache.json must be a JSON object keyed by chain id. "
            f"Got {type(data).__name__}."
        )

    # Heuristic: cache might be a dict keyed by chain_id, with a nested dict of fields.
    # We'll try to pull:
    # - seq_length
    # - resolution
    # - oligomeric_count or similar
    # If fields missing, we keep the chain and let downstream preprocessing filter.
    candidates: List[str] = []
    candidate_sequences: Dict[str, str] = {}
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
        seq = _extract_sequence(meta)
        if seq:
            candidate_sequences[chain_id] = seq

    if len(candidates) < args.train_size + args.val_size:
        raise ValueError(f"Not enough candidates after filtering: {len(candidates)}")

    cluster_method = "internal_pairwise"
    cluster_command: List[str] | None = None
    cluster_map: Dict[str, str]
    if len(candidate_sequences) == len(candidates):
        mmseqs_cluster_map, cluster_command = _cluster_sequences_mmseqs(
            sequences={cid: candidate_sequences[cid] for cid in candidates},
            mmseqs_bin=args.mmseqs_bin,
            min_seq_id=float(args.min_seq_id),
            coverage=float(args.coverage),
        )
        if mmseqs_cluster_map is not None:
            cluster_method = "mmseqs2"
            cluster_map = mmseqs_cluster_map
        else:
            if args.require_mmseqs:
                raise RuntimeError(
                    "MMseqs2 clustering was required but unavailable or failed. "
                    f"Expected executable `{args.mmseqs_bin}`."
                )
            cluster_map = _cluster_sequences_internal(
                sequences={cid: candidate_sequences[cid] for cid in candidates},
                min_seq_id=float(args.min_seq_id),
                coverage=float(args.coverage),
            )
    else:
        if args.require_mmseqs:
            raise RuntimeError(
                "MMseqs2 clustering was required but some candidate chains were missing sequences in chain_data_cache.json."
            )
        cluster_method = "pdb_id_fallback"
        cluster_map = {cid: _pdb_id(cid) for cid in candidates}

    rng.shuffle(candidates)

    # Sample val first, then restrict train to chains outside the selected homology clusters.
    val_ids = candidates[: args.val_size]
    val_clusters = {cluster_map[cid] for cid in val_ids}

    train_pool = [cid for cid in candidates[args.val_size :] if cluster_map[cid] not in val_clusters]
    if len(train_pool) < args.train_size:
        raise ValueError(
            "Not enough cluster-disjoint train candidates after selecting val: "
            f"need {args.train_size}, have {len(train_pool)}. "
            "Try reducing --val-size or adjusting filters."
        )

    rng.shuffle(train_pool)
    train_ids = train_pool[: args.train_size]

    train_clusters = {cluster_map[cid] for cid in train_ids}
    overlap = train_clusters & val_clusters
    if overlap:
        raise RuntimeError(f"Internal error: train/val homology overlap detected for {len(overlap)} clusters")

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
        f"(cluster-disjoint: {len(overlap)} overlaps)"
    )

    if args.lock_file:
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
                "min_seq_id": float(args.min_seq_id),
                "coverage": float(args.coverage),
                "mmseqs_bin": str(args.mmseqs_bin),
                "require_mmseqs": bool(args.require_mmseqs),
                "expected_chain_cache_sha256": expected_cache_sha or None,
            },
            "clustering": {
                "method": cluster_method,
                "cluster_count": len(set(cluster_map.values())),
                "command": cluster_command,
                "mmseqs_version": subprocess.run(
                    [args.mmseqs_bin, "--version"],
                    capture_output=True,
                    text=True,
                    check=False,
                ).stdout.strip() if cluster_method == "mmseqs2" else None,
                "cluster_assignments_sha256": _cluster_assignments_sha256(cluster_map),
            },
            "outputs": {
                "train_manifest": str(train_path.resolve()),
                "val_manifest": str(val_path.resolve()),
                "all_manifest": str(all_path.resolve()),
                "train_manifest_sha256": _sha256(train_path),
                "val_manifest_sha256": _sha256(val_path),
                "all_manifest_sha256": _sha256(all_path),
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
