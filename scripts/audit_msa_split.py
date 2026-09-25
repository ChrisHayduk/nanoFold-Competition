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
    iter_mmseqs_hits,
    load_msa_for_audit,
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
    ap.add_argument("--mmseqs-bin", default="mmseqs")
    ap.add_argument("--min-seq-id", type=float, default=0.30)
    ap.add_argument("--coverage", type=float, default=0.80)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--max-seqs", type=int, default=1_000_000)
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


def _collect_mmseqs_records(
    *,
    train_ids: Sequence[str],
    val_ids: Sequence[str],
    features_dir: Path,
    include_identifiers: bool,
    val_features_dir: Path | None = None,
    source_targets_only: bool = False,
) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    target_records: list[tuple[str, str]] = []
    seen_targets: set[str] = set()
    for target_index, chain_id in enumerate(val_ids):
        loaded = _load_feature_msa(val_features_dir or features_dir, chain_id)
        if loaded is None:
            continue
        sequence = _msa_row_to_ungapped_sequence(loaded[0][0])
        digest = _sequence_sha256(sequence)
        if digest in seen_targets:
            continue
        seen_targets.add(digest)
        header = f"valtarget_{target_index}"
        if include_identifiers:
            header += f"|origin={chain_id}"
        target_records.append((header, sequence))

    source_records: list[tuple[str, str]] = []
    seen_sources: set[str] = set()
    for chain_id in train_ids:
        loaded = _load_feature_msa(features_dir, chain_id)
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
) -> dict[str, Any]:
    validate_search_parameters(min_seq_id=min_seq_id, coverage=coverage, threads=threads, max_seqs=max_seqs)
    if shutil.which(mmseqs_bin) is None:
        raise SystemExit(f"MMseqs binary not found: {mmseqs_bin}")
    target_records, source_records = _collect_mmseqs_records(
        train_ids=train_ids,
        val_ids=val_ids,
        features_dir=features_dir,
        include_identifiers=include_identifiers,
        val_features_dir=val_features_dir,
        source_targets_only=source_targets_only,
    )
    if not target_records:
        raise SystemExit("Could not collect held-out target records for MMseqs audit.")
    source_count_key = "source_target_records" if source_targets_only else "source_non_query_records"
    hit_count_key = (
        "val_targets_with_train_target_homolog_hits" if source_targets_only
        else "val_targets_with_train_msa_homolog_hits"
    )
    search_parameters = {
        "prefilter_mode": 2 if source_targets_only else 0,
        "evalue": 1e308 if source_targets_only else 1e-3,
        "sensitivity": 5.7,
        "mask": 0 if source_targets_only else 1,
        "comp_bias_corr": 0 if source_targets_only else 1,
        "search_type": 1,
        "dbtype": 1,
    }
    threshold = {
        "min_seq_id": min_seq_id, "coverage": coverage, "cov_mode": 0, "alignment_mode": 3, "seq_id_mode": 0,
    }
    if source_targets_only and max_seqs < len(source_records):
        raise ValueError("Target-pair audit max-seqs must cover every source target")
    if not source_records:
        return {
            "threshold": threshold,
            "search_parameters": search_parameters,
            "target_records": len(target_records),
            source_count_key: 0,
            "hit_count": 0,
            hit_count_key: 0,
            "search_executed": False,
            "reason": "No source targets" if source_targets_only else "No non-query source rows of at least 20 residues",
        }

    source_name = "train_targets" if source_targets_only else "train_msa"
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
            "0",
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
        proc = subprocess.run(cmd, text=True, check=False)
        if proc.returncode != 0:
            raise RuntimeError(f"MMseqs failed with exit code {proc.returncode}: {' '.join(cmd)}")

    unique_target_hits: set[str] = set()
    hit_count = 0
    for query, _target in iter_mmseqs_hits(
        m8_out, query_ids={header for header, _ in target_records},
        target_ids={header for header, _ in source_records}, min_seq_id=min_seq_id, coverage=coverage,
    ):
        hit_count += 1
        unique_target_hits.add(query)
    return {
        "threshold": threshold,
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
            "m8_sha256": hashlib.sha256(m8_out.read_bytes()).hexdigest(),
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
    validate_search_parameters(
        min_seq_id=args.min_seq_id, coverage=args.coverage, threads=args.threads, max_seqs=args.max_seqs,
    )
    train_ids = read_manifest(args.train_manifest)
    val_ids = read_manifest(args.val_manifest)
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
        )
    failures = strict_failures(summary)
    summary["validation"] = {
        "strict": bool(args.strict),
        "passed": not failures,
        "failures": failures,
        "cluster_disjointness_checked": summary["target_manifest_disjointness"]["cluster_map_complete"],
        "mmseqs_homology_checked": bool(args.run_mmseqs),
        "target_target_homology_checked": bool(args.run_mmseqs),
        "homology_scope": "Target pairs use all-pair MMseqs local alignments without composition correction "
        "or prefilter masking, with E-value limit 1e308. Source non-query rows of at least 20 residues use "
        "a separate heuristic search. Identity and bidirectional coverage gates apply to the returned "
        "local alignments; zero hits do not prove absence of every possible alignment or all homology.",
        "depth_identifiers_included": bool(args.include_depth_identifiers),
    }
    (args.out_dir / "msa_split_audit_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(f"Wrote MSA split audit to {args.out_dir}")
    if args.strict and failures:
        raise SystemExit("MSA split audit failed: " + "; ".join(failures))


if __name__ == "__main__":
    main()
