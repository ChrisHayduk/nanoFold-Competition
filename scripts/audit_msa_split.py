"""Audit MSA availability, target separation and cross-split MSA-row leakage.

This script is intended for maintainer/rebuttal analysis. It can summarize MSA
depths for two split manifests, verify target chain/cluster disjointness, check
exact MSA-row sequence overlap, and optionally run separate MMseqs2 searches
from held-out targets to source targets and source non-query MSA rows.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Sequence, cast

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from nanofold.a3m import GAP_ID, MASK_ID, RESTYPES
from nanofold.chain_paths import chain_data_dir, chain_npz_path
from nanofold.mmseqs import (
    batched_mmseqs_summary,
    canonical_target_sequence,
    iter_batched_mmseqs_hits,
    iter_mmseqs_hits,
    load_msa_for_audit,
    run_batched_mmseqs,
    sequence_records_sha256,
    validate_search_parameters,
)

MSA_TOKEN_TO_AA = {idx: aa for idx, aa in enumerate(RESTYPES)}
MSA_TOKEN_TO_AA[20] = "X"
DEFAULT_DEPTH_BIN_EDGES = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train-manifest", type=Path, default=Path("data/manifests/train.txt"))
    ap.add_argument("--val-manifest", type=Path, default=Path("data/manifests/val.txt"))
    ap.add_argument("--extra-heldout-manifest", type=Path, action="append", default=[],
                    help="Additional protected target queries from the cache; repeat for historical or hidden targets. "
                    "Their MSA features are not inspected. Requires exhaustive MMseqs and --heldout-chain-data-cache.")
    ap.add_argument("--heldout-chain-data-cache", type=Path, default=None,
                    help="Sequence cache for primary and extra protected targets; requires --extra-heldout-manifest.")
    ap.add_argument("--features-dir", type=Path, default=Path("data/processed_features"))
    ap.add_argument("--val-features-dir", type=Path, default=None, help="Held-out features root; defaults to --features-dir.")
    ap.add_argument(
        "--feature-exclusion-list",
        type=Path,
        default=Path("data/manifests/openfold_required_feature_exclusions.txt"),
    )
    ap.add_argument("--cluster-tsv", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument(
        "--depth-bin-edges",
        default=",".join(str(edge) for edge in DEFAULT_DEPTH_BIN_EDGES),
        help="Comma-separated inclusive upper edges for MSA-depth bins used in JS-divergence reporting.",
    )
    ap.add_argument(
        "--audit-raw-identifiers",
        action="store_true",
        help="Also scan raw A3M headers and report aggregate train/validation hit-identifier overlap.",
    )
    ap.add_argument("--raw-root", type=Path, default=Path("data/openproteinset"))
    ap.add_argument("--raw-msa-name", default="uniref90_hits.a3m")
    ap.add_argument(
        "--raw-msa-names",
        default="",
        help="Comma-separated raw A3M names to scan for identifier overlap. Defaults to --raw-msa-name.",
    )
    ap.add_argument(
        "--run-mmseqs", action="store_true",
        help="Align all held-out/source target pairs with a permissive E-value limit; separately search non-query MSA rows heuristically. Gate both with --strict.",
    )
    ap.add_argument(
        "--exhaustive", action="store_true",
        help="With --run-mmseqs, also compare every held-out target/non-query MSA-row pair without "
        "masking or a heuristic prefilter. Requires max-seqs to cover all unique source rows or each checkpoint batch.",
    )
    ap.add_argument("--mmseqs-bin", default="mmseqs")
    ap.add_argument("--min-seq-id", type=float, default=0.30)
    ap.add_argument("--coverage", type=float, default=0.80)
    ap.add_argument("--coverage-mode", type=int, choices=(0, 2), default=0,
                    help="0: require both sequence coverages; 2: require held-out query coverage only (requires --exhaustive).")
    ap.add_argument("--max-sequence-length", type=int, default=256,
                    help="Maximum sequence length admitted to either search database (1..65535); longer inputs fail.")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--max-seqs", type=int, default=1_000_000)
    ap.add_argument("--checkpoint-dir", type=Path, default=None,
                    help="Private resumable exhaustive search directory; requires --exhaustive and anonymous headers.")
    ap.add_argument("--source-batch-size", type=int, default=1_000_000, help="Maximum source records per batch (1..1000000).")
    ap.add_argument("--query-batch-size", type=int, default=256, help="Maximum held-out records per batch (1..256).")
    ap.add_argument("--search-disk-reserve-gib", type=int, default=100, help="Free-disk reserve during bounded search.")
    ap.add_argument("--search-max-rss-gib", type=int, default=48, help="Sampled caller plus native process-tree RSS ceiling.")
    ap.add_argument("--tmp-dir", type=Path, default=None)
    ap.add_argument(
        "--include-examples",
        action="store_true",
        help="Include public chain IDs and sequence prefixes in JSON examples. Do not use for hidden reports.",
    )
    ap.add_argument(
        "--include-identifiers-in-mmseqs",
        action="store_true",
        help="Write chain/row identifiers into MMseqs FASTA headers. Do not use for hidden reports.",
    )
    ap.add_argument("--skip-plot", action="store_true")
    ap.add_argument(
        "--include-depth-identifiers",
        action="store_true",
        help="Include chain IDs in the depth CSV. Use only for public or sealed private reports.",
    )
    ap.add_argument(
        "--strict",
        action="store_true",
        help="Fail on missing/empty features, duplicate targets, target chain/PDB/cluster overlap, "
        "or held-out target leakage. MMseqs homology is gated only with --run-mmseqs.",
    )
    return ap.parse_args()


def read_manifest(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text().splitlines() if line.strip() and not line.lstrip().startswith("#")]


def _read_optional_set(path: Path | None) -> set[str]:
    if path is None:
        return set()
    return set(read_manifest(path))


def _sha256_file(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()



def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, prefix=f".{path.name}.", delete=False) as handle:
        temporary = Path(handle.name)
        try:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _assert_inputs_unchanged(snapshots: dict[Path, str]) -> None:
    for path, expected in snapshots.items():
        if not path.is_file() or _sha256_file(path) != expected:
            raise ValueError("MSA audit input changed during scanning or search; no report was published")


def _sequence_sha256(sequence: str) -> str:
    return hashlib.sha256(sequence.upper().encode("utf-8")).hexdigest()


def _msa_row_to_ungapped_sequence(row: np.ndarray) -> str:
    chars: list[str] = []
    for value in row.tolist():
        token = int(value)
        if token in (GAP_ID, MASK_ID):
            continue
        chars.append(MSA_TOKEN_TO_AA.get(token, "X"))
    return "".join(chars)


def _msa_row_digest_and_length(row: np.ndarray) -> tuple[str, int]:
    filtered = row[(row != GAP_ID) & (row != MASK_ID)].astype(np.uint8, copy=False)
    return hashlib.sha256(filtered.tobytes()).hexdigest(), int(filtered.shape[0])


def _percentiles(values: Sequence[int]) -> dict[str, float | int | None]:
    if not values:
        return {
            "min": None,
            "p25": None,
            "median": None,
            "p75": None,
            "max": None,
        }
    arr = np.asarray(values, dtype=np.float64)
    return {
        "min": int(np.min(arr)),
        "p25": float(np.percentile(arr, 25)),
        "median": float(np.percentile(arr, 50)),
        "p75": float(np.percentile(arr, 75)),
        "max": int(np.max(arr)),
    }


def _parse_depth_bin_edges(raw: str) -> tuple[int, ...]:
    edges = tuple(sorted({int(token.strip()) for token in raw.split(",") if token.strip()}))
    if not edges or edges[0] < 1:
        raise ValueError("--depth-bin-edges must contain positive integer edges")
    return edges


def _depth_bin(depth: int, edges: Sequence[int]) -> str:
    previous = 0
    for edge in edges:
        if depth <= edge:
            if previous == 0:
                return f"<= {edge}"
            if previous + 1 == edge:
                return str(edge)
            return f"{previous + 1}-{edge}"
        previous = edge
    return f"> {edges[-1]}"


def _depth_bin_labels(edges: Sequence[int]) -> list[str]:
    labels: list[str] = []
    previous = 0
    for edge in edges:
        if previous == 0:
            labels.append(f"<= {edge}")
        elif previous + 1 == edge:
            labels.append(str(edge))
        else:
            labels.append(f"{previous + 1}-{edge}")
        previous = edge
    labels.append(f"> {edges[-1]}")
    return labels


def _jensen_shannon_divergence(counts_a: Sequence[int], counts_b: Sequence[int]) -> float | None:
    total_a = float(sum(counts_a))
    total_b = float(sum(counts_b))
    if total_a <= 0 or total_b <= 0:
        return None
    p = np.asarray(counts_a, dtype=np.float64) / total_a
    q = np.asarray(counts_b, dtype=np.float64) / total_b
    m = 0.5 * (p + q)

    def kl_divergence(left: np.ndarray, right: np.ndarray) -> float:
        mask = left > 0
        return float(np.sum(left[mask] * np.log2(left[mask] / right[mask])))

    return 0.5 * kl_divergence(p, m) + 0.5 * kl_divergence(q, m)


def summarize_msa_depth_distribution(
    *,
    train_rows: Sequence[dict[str, Any]],
    val_rows: Sequence[dict[str, Any]],
    depth_bin_edges: Sequence[int],
) -> dict[str, Any]:
    labels = _depth_bin_labels(depth_bin_edges)
    train_counts = Counter(str(row["msa_depth_bin"]) for row in train_rows)
    val_counts = Counter(str(row["msa_depth_bin"]) for row in val_rows)
    train_values = [int(train_counts[label]) for label in labels]
    val_values = [int(val_counts[label]) for label in labels]
    return {
        "bin_edges": list(depth_bin_edges),
        "bins": labels,
        "train_counts": dict(zip(labels, train_values, strict=True)),
        "val_counts": dict(zip(labels, val_values, strict=True)),
        "jensen_shannon_divergence_bits": _jensen_shannon_divergence(train_values, val_values),
    }


def _load_feature_msa(features_dir: Path, chain_id: str) -> tuple[np.ndarray, int] | None:
    path = chain_npz_path(features_dir, chain_id)
    if not path.exists():
        return None
    msa = load_msa_for_audit(path, chain_id=chain_id)
    return msa, int(msa.shape[1])


def summarize_split_features(
    *,
    split_name: str,
    manifest_ids: Sequence[str],
    features_dir: Path,
    feature_exclusions: set[str],
    depth_bin_edges: Sequence[int],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    depths: list[int] = []
    lengths: list[int] = []
    missing_features = 0
    missing_msa_key = 0
    depth_rows: list[dict[str, Any]] = []

    for chain_id in manifest_ids:
        loaded = _load_feature_msa(features_dir, chain_id)
        if loaded is None:
            if chain_npz_path(features_dir, chain_id).exists():
                missing_msa_key += 1
            else:
                missing_features += 1
            continue
        msa, length = loaded
        depth = int(msa.shape[0])
        depths.append(depth)
        lengths.append(length)
        depth_rows.append(
            {
                "split": split_name,
                "chain_id": chain_id,
                "length": length,
                "msa_depth": depth,
                "msa_depth_bin": _depth_bin(depth, depth_bin_edges),
            }
        )

    msa_stats = _percentiles(depths)
    length_stats = _percentiles(lengths)
    summary = {
        "manifest_chains": len(manifest_ids),
        "missing_feature_files": missing_features,
        "missing_msa_key": missing_msa_key,
        "feature_exclusion_intersection_count": len(set(manifest_ids) & feature_exclusions),
        "msa_depth_min": msa_stats["min"],
        "msa_depth_p25": msa_stats["p25"],
        "msa_depth_median": msa_stats["median"],
        "msa_depth_p75": msa_stats["p75"],
        "msa_depth_max": msa_stats["max"],
        "length_min": length_stats["min"],
        "length_median": length_stats["median"],
        "length_max": length_stats["max"],
    }
    return summary, depth_rows


def _load_cluster_map(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}
    cluster_map: dict[str, str] = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) != 2:
            raise ValueError("Cluster TSV must contain exactly representative and member columns")
        rep, member = parts[0], parts[1]
        for chain_id in (rep, member):
            if chain_id in cluster_map and cluster_map[chain_id] != rep:
                raise ValueError("Cluster TSV assigns a chain to conflicting clusters")
            cluster_map[chain_id] = rep
    return cluster_map


def target_disjointness(
    *,
    train_ids: Sequence[str],
    val_ids: Sequence[str],
    cluster_tsv: Path | None,
) -> dict[str, Any]:
    train_set = set(train_ids)
    val_set = set(val_ids)
    cluster_map = _load_cluster_map(cluster_tsv)
    train_clusters = {cluster_map[chain_id] for chain_id in train_ids if chain_id in cluster_map}
    val_clusters = {cluster_map[chain_id] for chain_id in val_ids if chain_id in cluster_map}
    missing_cluster_count = len((train_set | val_set) - cluster_map.keys())
    cluster_complete = cluster_tsv is not None and missing_cluster_count == 0
    return {
        "train_chains": len(train_ids),
        "val_chains": len(val_ids),
        "train_duplicate_chain_count": len(train_ids) - len(train_set),
        "val_duplicate_chain_count": len(val_ids) - len(val_set),
        "train_clusters": len(train_clusters) if cluster_tsv is not None else None,
        "val_clusters": len(val_clusters) if cluster_tsv is not None else None,
        "train_val_chain_overlap": len(train_set & val_set),
        "train_val_pdb_overlap": len(
            {chain_id.split("_", 1)[0].lower() for chain_id in train_ids}
            & {chain_id.split("_", 1)[0].lower() for chain_id in val_ids}
        ),
        "train_val_cluster_overlap": len(train_clusters & val_clusters) if cluster_tsv is not None else None,
        "cluster_map_complete": cluster_complete,
        "missing_cluster_assignments": missing_cluster_count if cluster_tsv is not None else None,
        "cluster_tsv_supplied": cluster_tsv is not None,
    }


def _first_origin(origins: dict[str, list[dict[str, Any]]], digest: str) -> dict[str, Any] | None:
    values = origins.get(digest)
    if not values:
        return None
    return values[0]


def _scan_msa_rows(
    *,
    manifest_ids: Sequence[str],
    features_dir: Path,
    split_name: str,
    include_examples: bool,
) -> dict[str, Any]:
    all_hashes: set[str] = set()
    non_query_hashes: set[str] = set()
    target_hashes: set[str] = set()
    target_hash_to_chain: dict[str, str] = {}
    origins: dict[str, list[dict[str, Any]]] = {}
    rows = 0
    non_query_rows = 0
    short_rows = 0
    missing_npz = 0

    for chain_id in manifest_ids:
        loaded = _load_feature_msa(features_dir, chain_id)
        if loaded is None:
            missing_npz += 1
            continue
        msa, _length = loaded
        for row_index, row in enumerate(msa):
            digest, sequence_length = _msa_row_digest_and_length(row)
            rows += 1
            if sequence_length < 20:
                short_rows += 1
            all_hashes.add(digest)
            if row_index == 0:
                target_hashes.add(digest)
                target_hash_to_chain[digest] = chain_id
            else:
                non_query_rows += 1
                non_query_hashes.add(digest)
            if include_examples and len(origins.get(digest, [])) < 3:
                origins.setdefault(digest, []).append(
                    {"chain_id": chain_id, "row": int(row_index), "split": split_name, "length": sequence_length}
                )

    return {
        "hashes": all_hashes,
        "non_query_hashes": non_query_hashes,
        "target_hashes": target_hashes,
        "target_hash_to_chain": target_hash_to_chain,
        "origins": origins,
        "summary": {
            "chains": len(manifest_ids),
            "rows": rows,
            "non_query_rows": non_query_rows,
            "unique_ungapped_rows": len(all_hashes),
            "unique_non_query_ungapped_rows": len(non_query_hashes),
            "short_rows": short_rows,
            "missing_npz": missing_npz,
        },
    }


def audit_processed_msa_exact(
    *,
    train_ids: Sequence[str],
    val_ids: Sequence[str],
    features_dir: Path,
    include_examples: bool = False,
    val_features_dir: Path | None = None,
) -> dict[str, Any]:
    train = _scan_msa_rows(
        manifest_ids=train_ids,
        features_dir=features_dir,
        split_name="train",
        include_examples=include_examples,
    )
    val = _scan_msa_rows(
        manifest_ids=val_ids,
        features_dir=val_features_dir or features_dir,
        split_name="val",
        include_examples=include_examples,
    )

    val_targets_in_train_non_query = val["target_hashes"] & train["non_query_hashes"]
    exact_non_query_overlap = train["non_query_hashes"] & val["non_query_hashes"]
    exact_all_overlap = train["hashes"] & val["hashes"]
    payload: dict[str, Any] = {
        "train": train["summary"],
        "val": val["summary"],
        "val_target_sequences_seen_in_train_non_query_msa": len(val_targets_in_train_non_query),
        "train_val_target_sequence_overlap": len(train["target_hashes"] & val["target_hashes"]),
        "train_val_non_query_msa_unique_sequence_overlap": len(exact_non_query_overlap),
        "train_val_all_msa_unique_sequence_overlap": len(exact_all_overlap),
    }

    if include_examples:
        target_examples: dict[str, list[dict[str, Any]]] = {}
        for digest in sorted(val_targets_in_train_non_query)[:20]:
            chain_id = val["target_hash_to_chain"].get(digest, "unknown")
            target_examples[chain_id] = train["origins"].get(digest, [])[:3]
        overlap_examples: list[dict[str, Any]] = []
        for digest in sorted(exact_non_query_overlap)[:20]:
            train_origin = _first_origin(train["origins"], digest)
            val_origin = _first_origin(val["origins"], digest)
            if train_origin is not None and val_origin is not None:
                overlap_examples.append({"train_origin": train_origin, "val_origin": val_origin})
        payload["val_target_sequence_examples"] = target_examples
        payload["train_val_non_query_msa_overlap_examples"] = overlap_examples
    return payload


def _write_depth_csv(path: Path, rows: Iterable[dict[str, Any]], *, include_identifiers: bool = False) -> None:
    fields = ["split", "length", "msa_depth", "msa_depth_bin"]
    if include_identifiers:
        fields.insert(1, "chain_id")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


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


def _iter_a3m_headers(path: Path) -> Iterable[str]:
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if line.startswith(">"):
                yield line[1:].strip()


def _canonical_raw_msa_identifier(header: str) -> str | None:
    token = header.strip().split(None, 1)[0] if header.strip() else ""
    if not token or token.lower() == "query":
        return None
    return token.split("/", 1)[0]


def _scan_raw_msa_identifiers(
    *,
    manifest_ids: Sequence[str],
    raw_root: Path,
    msa_names: Sequence[str],
    split_name: str,
    include_examples: bool,
) -> dict[str, Any]:
    identifiers: set[str] = set()
    origins: dict[str, list[dict[str, Any]]] = {}
    roda_root = raw_root / "roda_pdb"
    files = 0
    missing_msa_files = 0
    non_query_headers = 0

    for chain_id in manifest_ids:
        chain_dir = chain_data_dir(roda_root, chain_id)
        loaded = False
        for msa_name in msa_names:
            msa_path = _find_msa_path(chain_dir, msa_name)
            if msa_path is None:
                continue
            loaded = True
            files += 1
            for row_index, header in enumerate(_iter_a3m_headers(msa_path)):
                if row_index == 0:
                    continue
                non_query_headers += 1
                identifier = _canonical_raw_msa_identifier(header)
                if identifier is None:
                    continue
                identifiers.add(identifier)
                if include_examples and len(origins.get(identifier, [])) < 3:
                    origins.setdefault(identifier, []).append(
                        {
                            "chain_id": chain_id,
                            "row": int(row_index),
                            "split": split_name,
                            "msa_name": msa_name,
                        }
                    )
        if not loaded:
            missing_msa_files += 1

    return {
        "identifiers": identifiers,
        "origins": origins,
        "summary": {
            "chains": len(manifest_ids),
            "files": files,
            "missing_msa_files": missing_msa_files,
            "non_query_headers": non_query_headers,
            "unique_identifiers": len(identifiers),
            "msa_names": list(msa_names),
        },
    }


def audit_raw_msa_identifier_overlap(
    *,
    train_ids: Sequence[str],
    val_ids: Sequence[str],
    raw_root: Path,
    msa_names: Sequence[str],
    include_examples: bool = False,
) -> dict[str, Any]:
    train = _scan_raw_msa_identifiers(
        manifest_ids=train_ids,
        raw_root=raw_root,
        msa_names=msa_names,
        split_name="train",
        include_examples=include_examples,
    )
    val = _scan_raw_msa_identifiers(
        manifest_ids=val_ids,
        raw_root=raw_root,
        msa_names=msa_names,
        split_name="val",
        include_examples=include_examples,
    )
    overlap = train["identifiers"] & val["identifiers"]
    payload: dict[str, Any] = {
        "train": train["summary"],
        "val": val["summary"],
        "train_val_unique_identifier_overlap": len(overlap),
        "raw_root": str(raw_root),
        "identifiers_are_included": False,
    }
    if include_examples:
        payload["overlap_examples"] = [
            {
                "identifier": identifier,
                "train_origin": train["origins"].get(identifier, [])[:3],
                "val_origin": val["origins"].get(identifier, [])[:3],
            }
            for identifier in sorted(overlap)[:20]
        ]
        payload["identifiers_are_included"] = True
    return payload


def _plot_depths(depth_csv: Path, out_dir: Path) -> None:
    os.environ.setdefault("MPLCONFIGDIR", tempfile.gettempdir())
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    values: dict[str, list[int]] = {"train": [], "val": []}
    with depth_csv.open() as handle:
        for row in csv.DictReader(handle):
            values.setdefault(row["split"], []).append(int(row["msa_depth"]))

    fig, ax = plt.subplots(figsize=(5.2, 3.4), dpi=180)
    labels = [split for split in ("train", "val") if values.get(split)]
    data = [values[split] for split in labels]
    boxplot = ax.boxplot(data, tick_labels=labels, showfliers=False, widths=0.55, patch_artist=True)
    for patch, color in zip(cast(list[Any], boxplot["boxes"]), ["#4C78A8", "#F58518"], strict=False):
        patch.set_facecolor(color)
        patch.set_alpha(0.5)
    ax.set_ylabel("Processed MSA rows")
    ax.set_xlabel("Split")
    ax.set_title("MSA Depth By Public Split")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    for ext in ("png", "pdf", "svg"):
        fig.savefig(out_dir / f"msa_depth_by_split.{ext}")
    plt.close(fig)


def _write_fasta(path: Path, records: Sequence[tuple[str, str]]) -> None:
    with path.open("w") as handle:
        for header, sequence in records:
            handle.write(f">{header}\n{sequence}\n")


def _mmseqs_version(mmseqs_bin: str) -> str | None:
    proc = subprocess.run([mmseqs_bin, "version"], capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        return None
    return proc.stdout.strip() or None


def _extra_heldout_targets(
    *, val_ids: Sequence[str], manifests: Sequence[Path], cache_path: Path | None,
    snapshots: dict[Path, str], exhaustive: bool,
) -> tuple[dict[str, str], dict[str, Any]]:
    if bool(manifests) != (cache_path is not None):
        raise ValueError("--extra-heldout-manifest and --heldout-chain-data-cache are required together")
    primary_ids = set(val_ids)
    scope: dict[str, Any] = {
        "primary_chain_count": len(primary_ids), "extra_chain_count": 0, "additional_chain_count": 0,
        "protected_chain_count": len(primary_ids), "extra_unique_query_count": 0,
        "extra_manifest_sha256": [], "chain_data_cache_sha256": None,
        "feature_statistics_scope": "Primary validation manifest only; extra targets contribute cached queries only.",
        "extra_msa_features_checked": False,
    }
    if cache_path is None:
        return {}, scope
    if not exhaustive:
        raise ValueError("Extra held-out targets require --exhaustive")
    cache_path = cache_path.resolve(strict=True)
    manifests = [path.resolve(strict=True) for path in manifests]
    for path in (*manifests, cache_path):
        digest = _sha256_file(path)
        if path in snapshots and snapshots[path] != digest:
            raise ValueError("MSA audit protected-target input changed during scanning")
        snapshots[path] = digest
    cache = json.loads(cache_path.read_text())
    if not isinstance(cache, dict):
        raise ValueError("Held-out chain-data cache must be a JSON object")
    extra_ids = {chain_id for path in manifests for chain_id in read_manifest(path)}
    if not extra_ids:
        raise ValueError("Extra held-out manifests must contain target chains")
    sequences: dict[str, str] = {}
    for chain_id in sorted(primary_ids | extra_ids):
        row = cache.get(chain_id)
        if not isinstance(row, dict):
            raise ValueError("Protected target is missing from the held-out chain-data cache")
        sequences_for_chain = {
            canonical_target_sequence(value)
            for key in ("sequence", "seq", "seqres", "aatype_sequence")
            if isinstance(value := row.get(key), str) and value.strip()
        }
        if len(sequences_for_chain) != 1:
            raise ValueError("Protected target cache must contain one unambiguous canonical sequence per chain")
        sequences[chain_id] = sequences_for_chain.pop()
    scope.update({
        "extra_chain_count": len(extra_ids), "additional_chain_count": len(extra_ids - primary_ids),
        "protected_chain_count": len(primary_ids | extra_ids),
        "extra_unique_query_count": len({sequences[chain_id] for chain_id in extra_ids}),
        "extra_manifest_sha256": sorted(snapshots[path] for path in manifests),
        "chain_data_cache_sha256": snapshots[cache_path],
        "extra_manifest_pins": [{"path": str(path), "sha256": snapshots[path]} for path in manifests],
        "chain_data_cache_path": str(cache_path),
    })
    _assert_inputs_unchanged(snapshots)
    return sequences, scope


def _collect_mmseqs_records(
    *,
    train_ids: Sequence[str],
    val_ids: Sequence[str],
    features_dir: Path,
    include_identifiers: bool,
    val_features_dir: Path | None = None,
    source_targets_only: bool = False,
    input_snapshots: dict[Path, str] | None = None,
    require_complete: bool = False,
    protected_cache_sequences: dict[str, str] | None = None,
    heldout_query_scope: dict[str, Any] | None = None,
) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    def load_bound(root: Path, chain_id: str) -> tuple[np.ndarray, int] | None:
        path = chain_npz_path(root, chain_id)
        if not path.exists():
            if require_complete:
                raise ValueError("Protected-target audit requires every source and primary held-out feature file")
            return None
        path = path.resolve(strict=True)
        if input_snapshots is not None:
            digest = _sha256_file(path)
            if path in input_snapshots and input_snapshots[path] != digest:
                raise ValueError("MSA audit input changed during record collection")
            input_snapshots[path] = digest
        msa = load_msa_for_audit(path, chain_id=chain_id)
        return msa, int(msa.shape[1])

    target_records: list[tuple[str, str]] = []
    seen_targets: set[str] = set()
    for target_index, chain_id in enumerate(val_ids):
        loaded = load_bound(val_features_dir or features_dir, chain_id)
        if loaded is None:
            continue
        sequence = _msa_row_to_ungapped_sequence(loaded[0][0])
        if protected_cache_sequences and sequence != protected_cache_sequences.get(chain_id):
            raise ValueError("Primary held-out feature query does not match the protected target cache")
        digest = _sequence_sha256(sequence)
        if digest in seen_targets:
            continue
        seen_targets.add(digest)
        header = f"valtarget_{target_index}"
        if include_identifiers:
            header += f"|origin={chain_id}"
        target_records.append((header, sequence))

    primary_query_count = len(target_records)
    for index, (chain_id, sequence) in enumerate(sorted((protected_cache_sequences or {}).items())):
        digest = _sequence_sha256(sequence)
        if digest in seen_targets:
            continue
        seen_targets.add(digest)
        header = f"valtarget_{len(val_ids) + index}"
        if include_identifiers:
            header += f"|origin={chain_id}"
        target_records.append((header, sequence))
    if heldout_query_scope is not None:
        heldout_query_scope.update({
            "primary_unique_query_count": primary_query_count,
            "additional_unique_query_count": len(target_records) - primary_query_count,
            "protected_unique_query_count": len(target_records),
            "searched_query_records_sha256": sequence_records_sha256(target_records),
        })

    source_records: list[tuple[str, str]] = []
    seen_sources: set[str] = set()
    for chain_id in train_ids:
        loaded = load_bound(features_dir, chain_id)
        if loaded is None:
            continue
        msa = loaded[0]
        rows = enumerate(msa[:1]) if source_targets_only else enumerate(msa[1:], start=1)
        for row_index, row in rows:
            sequence = _msa_row_to_ungapped_sequence(row)
            if not source_targets_only and len(sequence) < 20:
                continue
            digest = _sequence_sha256(sequence)
            if digest in seen_sources:
                continue
            seen_sources.add(digest)
            prefix = "traintarget" if source_targets_only else "trainmsa"
            header = f"{prefix}_{len(source_records)}"
            if include_identifiers:
                header += f"|origin={chain_id}|row={row_index}"
            source_records.append((header, sequence))
    return target_records, source_records


def _validate_search_scope(*, coverage_mode: int, max_sequence_length: int, exhaustive: bool) -> None:
    if type(coverage_mode) is not int or coverage_mode not in (0, 2):
        raise ValueError("--coverage-mode must be 0 or 2")
    if type(max_sequence_length) is not int or not 1 <= max_sequence_length <= 65535:
        raise ValueError("--max-sequence-length must be in 1..65535")
    if coverage_mode == 2 and not exhaustive:
        raise ValueError("--coverage-mode 2 requires --exhaustive")


def run_mmseqs_homology(
    *,
    train_ids: Sequence[str],
    val_ids: Sequence[str],
    features_dir: Path,
    out_dir: Path,
    mmseqs_bin: str,
    min_seq_id: float,
    coverage: float,
    threads: int,
    max_seqs: int,
    tmp_dir: Path | None,
    include_identifiers: bool,
    val_features_dir: Path | None = None,
    source_targets_only: bool = False,
    exhaustive: bool = False,
    checkpoint_dir: Path | None = None,
    source_batch_size: int = 1_000_000,
    query_batch_size: int = 256,
    reserve_disk_bytes: int = 100 * 1024**3,
    max_rss_bytes: int = 48 * 1024**3,
    input_binding_sha256: str | None = None,
    coverage_mode: int = 0,
    max_sequence_length: int = 256,
    extra_heldout_manifests: Sequence[Path] = (),
    heldout_chain_data_cache: Path | None = None,
) -> dict[str, Any]:
    _validate_search_scope(coverage_mode=coverage_mode, max_sequence_length=max_sequence_length, exhaustive=exhaustive)
    validate_search_parameters(min_seq_id=min_seq_id, coverage=coverage, threads=threads, max_seqs=max_seqs)
    if shutil.which(mmseqs_bin) is None:
        raise SystemExit(f"MMseqs binary not found: {mmseqs_bin}")
    if checkpoint_dir is not None and (not exhaustive or include_identifiers):
        raise ValueError("Checkpointed audit requires --exhaustive and anonymous MMseqs headers")
    snapshots: dict[Path, str] = {}
    protected_sequences, heldout_query_scope = _extra_heldout_targets(
        val_ids=val_ids, manifests=extra_heldout_manifests, cache_path=heldout_chain_data_cache,
        snapshots=snapshots, exhaustive=exhaustive,
    )
    target_records, source_records = _collect_mmseqs_records(
        train_ids=train_ids,
        val_ids=val_ids,
        features_dir=features_dir,
        include_identifiers=include_identifiers,
        val_features_dir=val_features_dir,
        source_targets_only=source_targets_only,
        input_snapshots=snapshots,
        require_complete=checkpoint_dir is not None or bool(extra_heldout_manifests),
        protected_cache_sequences=protected_sequences, heldout_query_scope=heldout_query_scope,
    )
    if not target_records:
        raise SystemExit("Could not collect held-out target records for MMseqs audit.")
    if any(len(sequence) > max_sequence_length
           for records in (target_records, source_records) for _header, sequence in records):
        raise ValueError(f"MMseqs audit sequence exceeds --max-sequence-length {max_sequence_length}")
    source_count_key = "source_target_records" if source_targets_only else "source_non_query_records"
    hit_count_key = (
        "val_targets_with_train_target_homolog_hits" if source_targets_only
        else "val_targets_with_train_msa_homolog_hits"
    )
    all_pairs = source_targets_only or exhaustive
    search_contract = "mmseqs-local-all-pairs-v1" if all_pairs else "mmseqs-prefiltered-v1"
    search_parameters = {
        "prefilter_mode": 2 if all_pairs else 0,
        "evalue": 1e308 if all_pairs else 1e-3,
        "sensitivity": 5.7,
        "mask": 0 if all_pairs else 1,
        "comp_bias_corr": 0 if all_pairs else 1,
        "search_type": 1,
        "dbtype": 1,
        "max_seqs": max_seqs,
        "max_accept": max_seqs if all_pairs else 2147483647,
        "max_rejected": max_seqs if all_pairs else 2147483647,
        "max_sequence_length": max_sequence_length,
    }
    threshold = {
        "min_seq_id": min_seq_id, "coverage": coverage, "cov_mode": coverage_mode, "alignment_mode": 3, "seq_id_mode": 0,
    }
    if all_pairs and checkpoint_dir is None and max_seqs < len(source_records):
        raise ValueError(
            f"Exhaustive audit requires --max-seqs >= {len(source_records)} unique source records; got {max_seqs}."
        )
    if not source_records:
        _assert_inputs_unchanged(snapshots)
        return {
            "heldout_query_scope": heldout_query_scope,
            "threshold": threshold,
            "coverage_mode": coverage_mode, "max_sequence_length": max_sequence_length,
            "search_contract": search_contract,
            "exhaustive": all_pairs,
            "possible_pair_count": 0,
            "search_parameters": search_parameters,
            "target_records": len(target_records),
            source_count_key: 0,
            "hit_count": 0,
            hit_count_key: 0,
            "search_executed": False,
            "reason": "No source targets" if source_targets_only else "No non-query source rows of at least 20 residues",
        }

    source_name = "train_targets" if source_targets_only else "train_msa"
    if checkpoint_dir is not None:
        binding_records = [{"path": str(path), "sha256": digest} for path, digest in sorted(snapshots.items())]
        bound_inputs = hashlib.sha256(json.dumps({
            "source_assets": binding_records, "upstream_input_binding_sha256": input_binding_sha256,
        }, sort_keys=True).encode()).hexdigest()
        with tempfile.TemporaryDirectory(prefix="nanofold_msa_split_audit_", dir=str(tmp_dir) if tmp_dir else None) as tmp:
            target_fasta = Path(tmp) / "val_targets.fasta"
            source_fasta = Path(tmp) / f"{source_name}.fasta"
            _write_fasta(target_fasta, sorted(target_records))
            _write_fasta(source_fasta, sorted(source_records))
            result = run_batched_mmseqs(
                query_fasta=target_fasta, source_fasta=source_fasta, checkpoint_dir=checkpoint_dir / source_name,
                mmseqs_bin=mmseqs_bin, min_seq_id=min_seq_id, coverage=coverage, threads=threads, max_seqs=max_seqs,
                source_batch_size=source_batch_size, query_batch_size=query_batch_size,
                reserve_disk_bytes=reserve_disk_bytes, max_rss_bytes=max_rss_bytes,
                input_binding_sha256=bound_inputs, coverage_mode=coverage_mode,
                max_sequence_length=max_sequence_length,
            )
            unique_target_hits: set[str] = set()
            hit_count = 0
            for query, _target in iter_batched_mmseqs_hits(
                result, min_seq_id=min_seq_id, coverage=coverage, coverage_mode=coverage_mode,
            ):
                unique_target_hits.add(query)
                hit_count += 1
        _assert_inputs_unchanged(snapshots)
        return {
            "heldout_query_scope": heldout_query_scope,
            "threshold": threshold, "search_contract": search_contract, "exhaustive": True,
            "coverage_mode": coverage_mode, "max_sequence_length": max_sequence_length,
            "possible_pair_count": len(target_records) * len(source_records), "search_parameters": search_parameters,
            "mmseqs": {"binary": mmseqs_bin, "version": _mmseqs_version(mmseqs_bin), "command": [],
                       "batching": batched_mmseqs_summary(result)},
            "target_records": len(target_records), source_count_key: len(source_records),
            "hit_count": hit_count, hit_count_key: len(unique_target_hits), "m8_path": None,
            "input_binding": {
                "schema_version": 1, "source_records_sha256": sequence_records_sha256(source_records),
                "target_records_sha256": sequence_records_sha256(target_records),
                "query_ids": [header for header, _ in target_records],
                "source_assets": binding_records, "upstream_input_binding_sha256": input_binding_sha256,
                "batch_receipt_sha256": result["receipt_sha256"],
            },
            "identifiers_in_m8": False, "search_executed": True,
        }
    m8_out = out_dir / f"val_targets_vs_{source_name}.mmseqs_m8"
    m8_out.unlink(missing_ok=True)
    with tempfile.TemporaryDirectory(prefix="nanofold_msa_split_audit_", dir=str(tmp_dir) if tmp_dir else None) as tmp:
        tmp_path = Path(tmp)
        target_fasta = tmp_path / "val_targets.fasta"
        source_fasta = tmp_path / f"{source_name}.fasta"
        mmseqs_tmp = tmp_path / "mmseqs_tmp"
        _write_fasta(target_fasta, target_records)
        _write_fasta(source_fasta, source_records)
        cmd = [
            mmseqs_bin,
            "easy-search",
            str(target_fasta),
            str(source_fasta),
            str(m8_out),
            str(mmseqs_tmp),
            "--min-seq-id",
            str(min_seq_id),
            "-c",
            str(coverage),
            "--cov-mode",
            str(coverage_mode),
            "--max-seq-len", str(max_sequence_length),
            "--alignment-mode",
            "3",
            "--seq-id-mode",
            "0",
            "--search-type", "1",
            "--dbtype", "1",
            "--prefilter-mode", str(search_parameters["prefilter_mode"]),
            "-e", str(search_parameters["evalue"]),
            "-s", str(search_parameters["sensitivity"]),
            "--mask", str(search_parameters["mask"]),
            "--comp-bias-corr", str(search_parameters["comp_bias_corr"]),
            "--threads",
            str(threads),
            "--max-seqs",
            str(max_seqs),
            "--format-output",
            "query,target,pident,alnlen,qcov,tcov,evalue,bits",
        ]
        if all_pairs:
            cmd.extend(["--max-accept", str(max_seqs), "--max-rejected", str(max_seqs)])
        proc = subprocess.run(cmd, text=True, check=False)
        if proc.returncode != 0:
            raise RuntimeError(f"MMseqs failed with exit code {proc.returncode}: {' '.join(cmd)}")

    unique_target_hits = set()
    hit_count = 0
    for query, _target in iter_mmseqs_hits(
        m8_out, query_ids={header for header, _ in target_records},
        target_ids={header for header, _ in source_records}, min_seq_id=min_seq_id, coverage=coverage,
        coverage_mode=coverage_mode,
    ):
        hit_count += 1
        unique_target_hits.add(query)
    _assert_inputs_unchanged(snapshots)
    return {
        "heldout_query_scope": heldout_query_scope,
        "threshold": threshold,
        "coverage_mode": coverage_mode, "max_sequence_length": max_sequence_length,
        "search_contract": search_contract,
        "exhaustive": all_pairs,
        "possible_pair_count": len(target_records) * len(source_records),
        "search_parameters": search_parameters,
        "mmseqs": {"binary": mmseqs_bin, "version": _mmseqs_version(mmseqs_bin), "command": cmd},
        "target_records": len(target_records),
        source_count_key: len(source_records),
        "hit_count": hit_count,
        hit_count_key: len(unique_target_hits),
        "m8_path": str(m8_out),
        "input_binding": {
            "schema_version": 1,
            "source_records_sha256": sequence_records_sha256(source_records),
            "target_records_sha256": sequence_records_sha256(target_records),
            "query_ids": [header for header, _ in target_records],
            "m8_sha256": _sha256_file(m8_out),
            "upstream_input_binding_sha256": input_binding_sha256,
        },
        "identifiers_in_m8": include_identifiers,
        "search_executed": True,
    }


def strict_failures(summary: dict[str, Any]) -> list[str]:
    """Evaluate target leakage and completeness; shared background MSA rows remain descriptive."""
    failures: list[str] = []
    for split, stats in summary["splits"].items():
        if stats["manifest_chains"] == 0:
            failures.append(f"{split}: empty manifest")
        for key in ("missing_feature_files", "missing_msa_key", "feature_exclusion_intersection_count"):
            if stats[key]:
                failures.append(f"{split}: {key}={stats[key]}")
    disjointness = summary["target_manifest_disjointness"]
    for key in (
        "train_duplicate_chain_count", "val_duplicate_chain_count", "train_val_chain_overlap",
        "train_val_pdb_overlap", "train_val_cluster_overlap", "missing_cluster_assignments",
    ):
        if disjointness[key]:
            failures.append(f"{key}={disjointness[key]}")
    protected_disjointness = summary.get("protected_target_manifest_disjointness", {})
    for key in ("train_val_chain_overlap", "train_val_pdb_overlap", "train_val_cluster_overlap", "missing_cluster_assignments"):
        if protected_disjointness.get(key):
            failures.append(f"protected_targets: {key}={protected_disjointness[key]}")
    exact = summary["processed_msa_exact"]
    for key in ("val_target_sequences_seen_in_train_non_query_msa", "train_val_target_sequence_overlap"):
        if exact[key]:
            failures.append(f"{key}={exact[key]}")
    for report, key in (
        ("mmseqs_homology", "val_targets_with_train_msa_homolog_hits"),
        ("mmseqs_target_homology", "val_targets_with_train_target_homolog_hits"),
    ):
        if report in summary and summary[report][key]:
            failures.append(f"{key}={summary[report][key]}")
    return failures


def main() -> None:
    args = parse_args()
    if bool(args.extra_heldout_manifest) != (args.heldout_chain_data_cache is not None):
        raise ValueError("--extra-heldout-manifest and --heldout-chain-data-cache are required together")
    if args.extra_heldout_manifest and not (args.run_mmseqs and args.exhaustive):
        raise ValueError("Extra held-out targets require --run-mmseqs --exhaustive")
    if args.exhaustive and not args.run_mmseqs:
        raise SystemExit("--exhaustive requires --run-mmseqs")
    _validate_search_scope(coverage_mode=args.coverage_mode, max_sequence_length=args.max_sequence_length,
                           exhaustive=args.exhaustive)
    validate_search_parameters(
        min_seq_id=args.min_seq_id, coverage=args.coverage, threads=args.threads, max_seqs=args.max_seqs,
    )
    if args.checkpoint_dir is not None and (not args.exhaustive or args.include_identifiers_in_mmseqs):
        raise ValueError("--checkpoint-dir requires --exhaustive and anonymous MMseqs headers")
    if not 1 <= args.source_batch_size <= 1_000_000 or not 1 <= args.query_batch_size <= 256:
        raise ValueError("Source/query batch sizes must be in 1..1000000 and 1..256 respectively")
    if args.search_disk_reserve_gib < 1 or args.search_max_rss_gib < 1:
        raise ValueError("Search disk reserve and RSS limits must be positive GiB counts")
    args.train_manifest = args.train_manifest.resolve(strict=True)
    args.val_manifest = args.val_manifest.resolve(strict=True)
    metadata_paths = [args.train_manifest, args.val_manifest]
    args.extra_heldout_manifest = [path.resolve(strict=True) for path in args.extra_heldout_manifest]
    metadata_paths.extend(args.extra_heldout_manifest)
    if args.heldout_chain_data_cache is not None:
        args.heldout_chain_data_cache = args.heldout_chain_data_cache.resolve(strict=True)
        metadata_paths.append(args.heldout_chain_data_cache)
    if args.cluster_tsv is not None:
        args.cluster_tsv = args.cluster_tsv.resolve(strict=True)
        metadata_paths.append(args.cluster_tsv)
    if args.feature_exclusion_list is not None:
        args.feature_exclusion_list = args.feature_exclusion_list.resolve(strict=True)
        metadata_paths.append(args.feature_exclusion_list)
    snapshots = {path: _sha256_file(path) for path in metadata_paths}
    train_ids = read_manifest(args.train_manifest)
    val_ids = read_manifest(args.val_manifest)
    feature_origins: dict[Path, Path] = {}
    if args.checkpoint_dir is not None:
        for root, chain_ids in ((args.features_dir, train_ids), (args.val_features_dir or args.features_dir, val_ids)):
            for chain_id in chain_ids:
                origin = chain_npz_path(root, chain_id).absolute()
                resolved = origin.resolve(strict=True)
                feature_origins[origin] = resolved
                snapshots[resolved] = _sha256_file(resolved)
    upstream_binding = hashlib.sha256(json.dumps(
        [{"path": str(path), "sha256": digest} for path, digest in sorted(snapshots.items())], sort_keys=True,
    ).encode()).hexdigest()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    depth_bin_edges = _parse_depth_bin_edges(str(args.depth_bin_edges))
    raw_msa_names = _resolve_msa_names(args.raw_msa_name, args.raw_msa_names)

    feature_exclusions = _read_optional_set(args.feature_exclusion_list)
    train_summary, train_depth_rows = summarize_split_features(
        split_name="train",
        manifest_ids=train_ids,
        features_dir=args.features_dir,
        feature_exclusions=feature_exclusions,
        depth_bin_edges=depth_bin_edges,
    )
    val_summary, val_depth_rows = summarize_split_features(
        split_name="val",
        manifest_ids=val_ids,
        features_dir=args.val_features_dir or args.features_dir,
        feature_exclusions=feature_exclusions,
        depth_bin_edges=depth_bin_edges,
    )
    depth_csv = args.out_dir / "msa_split_audit_depths.csv"
    _write_depth_csv(depth_csv, [*train_depth_rows, *val_depth_rows], include_identifiers=args.include_depth_identifiers)
    if not args.skip_plot:
        _plot_depths(depth_csv, args.out_dir)

    summary: dict[str, Any] = {
        "splits": {"train": train_summary, "val": val_summary},
        "target_manifest_disjointness": target_disjointness(
            train_ids=train_ids,
            val_ids=val_ids,
            cluster_tsv=args.cluster_tsv,
        ),
        "msa_depth_distribution": summarize_msa_depth_distribution(
            train_rows=train_depth_rows,
            val_rows=val_depth_rows,
            depth_bin_edges=depth_bin_edges,
        ),
        "processed_msa_exact": audit_processed_msa_exact(
            train_ids=train_ids,
            val_ids=val_ids,
            features_dir=args.features_dir,
            include_examples=args.include_examples,
            val_features_dir=args.val_features_dir,
        ),
    }
    if args.extra_heldout_manifest:
        protected_ids = sorted(set(val_ids) | {
            chain_id for path in args.extra_heldout_manifest for chain_id in read_manifest(path)
        })
        summary["protected_target_manifest_disjointness"] = target_disjointness(
            train_ids=train_ids, val_ids=protected_ids, cluster_tsv=args.cluster_tsv,
        )
    if args.audit_raw_identifiers:
        summary["raw_msa_identifier_overlap"] = audit_raw_msa_identifier_overlap(
            train_ids=train_ids,
            val_ids=val_ids,
            raw_root=args.raw_root,
            msa_names=raw_msa_names,
            include_examples=args.include_examples,
        )
    if args.run_mmseqs:
        summary["mmseqs_target_homology"] = run_mmseqs_homology(
            train_ids=train_ids,
            val_ids=val_ids,
            features_dir=args.features_dir,
            out_dir=args.out_dir,
            mmseqs_bin=args.mmseqs_bin,
            min_seq_id=float(args.min_seq_id),
            coverage=float(args.coverage),
            threads=int(args.threads),
            max_seqs=int(args.max_seqs),
            tmp_dir=args.tmp_dir,
            include_identifiers=bool(args.include_identifiers_in_mmseqs),
            val_features_dir=args.val_features_dir,
            source_targets_only=True,
            exhaustive=bool(args.exhaustive),
            checkpoint_dir=args.checkpoint_dir,
            source_batch_size=args.source_batch_size, query_batch_size=args.query_batch_size,
            reserve_disk_bytes=args.search_disk_reserve_gib * 1024**3,
            max_rss_bytes=args.search_max_rss_gib * 1024**3,
            input_binding_sha256=upstream_binding,
            coverage_mode=args.coverage_mode, max_sequence_length=args.max_sequence_length,
            extra_heldout_manifests=args.extra_heldout_manifest, heldout_chain_data_cache=args.heldout_chain_data_cache,
        )
        summary["mmseqs_homology"] = run_mmseqs_homology(
            train_ids=train_ids,
            val_ids=val_ids,
            features_dir=args.features_dir,
            out_dir=args.out_dir,
            mmseqs_bin=args.mmseqs_bin,
            min_seq_id=float(args.min_seq_id),
            coverage=float(args.coverage),
            threads=int(args.threads),
            max_seqs=int(args.max_seqs),
            tmp_dir=args.tmp_dir,
            include_identifiers=bool(args.include_identifiers_in_mmseqs),
            val_features_dir=args.val_features_dir,
            exhaustive=bool(args.exhaustive),
            checkpoint_dir=args.checkpoint_dir,
            source_batch_size=args.source_batch_size, query_batch_size=args.query_batch_size,
            reserve_disk_bytes=args.search_disk_reserve_gib * 1024**3,
            max_rss_bytes=args.search_max_rss_gib * 1024**3,
            input_binding_sha256=upstream_binding,
            coverage_mode=args.coverage_mode, max_sequence_length=args.max_sequence_length,
            extra_heldout_manifests=args.extra_heldout_manifest, heldout_chain_data_cache=args.heldout_chain_data_cache,
        )
        if summary["mmseqs_target_homology"]["heldout_query_scope"] != summary["mmseqs_homology"]["heldout_query_scope"]:
            raise ValueError("Protected target query scope changed between target and MSA searches")
        summary["heldout_query_scope"] = summary["mmseqs_target_homology"]["heldout_query_scope"]
    failures = strict_failures(summary)
    summary["validation"] = {
        "strict": bool(args.strict),
        "coverage_mode": args.coverage_mode,
        "max_sequence_length": args.max_sequence_length,
        "passed": not failures,
        "failures": failures,
        "cluster_disjointness_checked": summary["target_manifest_disjointness"]["cluster_map_complete"],
        "mmseqs_homology_checked": bool(args.run_mmseqs),
        "target_target_homology_checked": bool(args.run_mmseqs),
        "exhaustive_msa_homology_checked": bool(args.run_mmseqs and args.exhaustive),
        "msa_search_contract": summary.get("mmseqs_homology", {}).get("search_contract"),
        "homology_scope": "Target pairs use all-pair MMseqs local alignments without composition correction "
        "or prefilter masking, with E-value limit 1e308. Source non-query rows of at least 20 residues use "
        + ("the same all-pair contract. " if args.exhaustive else "a separate heuristic search. ")
        + ("Identity and bidirectional coverage gates " if args.coverage_mode == 0
           else "Identity and held-out query coverage gates (source coverage is not required) ")
        + "apply to the returned local alignments; zero hits do not "
        "prove absence of every possible alignment or all homology.",
        "depth_identifiers_included": bool(args.include_depth_identifiers),
        "feature_statistics_scope": "Primary train and validation manifests only; extra protected targets add search queries only.",
    }
    if not args.run_mmseqs:
        summary["validation"]["homology_scope"] = "MMseqs homology was not checked; --run-mmseqs was not requested."
    if any(origin.resolve(strict=True) != resolved for origin, resolved in feature_origins.items()):
        raise ValueError("MSA audit feature path changed during scanning or search")
    _assert_inputs_unchanged(snapshots)
    _write_json_atomic(args.out_dir / "msa_split_audit_summary.json", summary)
    print(f"Wrote MSA split audit to {args.out_dir}")
    if args.strict and failures:
        raise SystemExit("MSA split audit failed: " + "; ".join(failures))


if __name__ == "__main__":
    main()
