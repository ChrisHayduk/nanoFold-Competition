"""Full-parent heldout barriers with explicitly separate training diversity groups.

A heldout domain may occupy a small fraction of a long training parent. The
barrier therefore requires coverage of the heldout query only. Transitive
training diversity groups are not claims about global split independence.
"""
from __future__ import annotations

import hashlib
import importlib
import json
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from nanofold.mmseqs import canonical_target_sequence

POLICY = "full_parent_query_barrier_plus_training_diversity_v1"


def sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def file_record(path: Path) -> dict[str, str]:
    return {"path": str(path.resolve()), "sha256": sha256(path)}


def sequence_id(sequence: str) -> str:
    return "s_" + hashlib.sha256(sequence.encode()).hexdigest()


def read_manifest(path: Path) -> list[str]:
    rows = [line.strip() for line in path.read_text().splitlines() if line.strip() and not line.startswith("#")]
    if not rows or len(rows) != len(set(rows)):
        raise ValueError("Manifest is empty or contains repeated chains")
    return rows


def read_groups(path: Path) -> dict[str, str]:
    groups = {}
    for line in path.read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        columns = line.split("\t")
        if len(columns) != 2 or columns[1] in groups:
            raise ValueError("Malformed or duplicate original group assignment")
        groups[columns[1]] = columns[0]
    return groups


def read_population(path: Path) -> list[dict[str, Any]]:
    records = [json.loads(line) for line in path.open()]
    ids = [row["chain_id"] for row in records]
    if not ids or ids != sorted(set(ids)):
        raise ValueError("Parent population must contain every chain once in sorted order")
    for row in records:
        sequence = row["sequence"]
        if canonical_target_sequence(sequence) != sequence or "X" in sequence:
            raise ValueError("Parent population requires canonical 20-amino-acid sequences")
        if row["length"] != len(sequence) or row["sequence_sha256"] != hashlib.sha256(sequence.encode()).hexdigest():
            raise ValueError("Parent sequence identity is inconsistent")
    return records


def validate_population_source(records: Sequence[Mapping[str, Any]], cache_path: Path) -> None:
    """Replay the complete metadata frame before any source-copy reduction."""
    manifest_module = importlib.import_module("scripts.build_manifests")
    metadata_module = importlib.import_module("scripts.build_structure_metadata")
    cache = json.loads(cache_path.read_text())
    maximum = max((len(metadata_module._extract_sequence(row) or "") for row in cache.values() if isinstance(row, dict)), default=0)
    expected, _ = metadata_module._candidate_rows(cache, min_len=40, max_len=maximum, max_resolution=3.0,
                                  max_unknown_aa_fraction=0.0, require_monomer=False)
    if len(records) != len(expected):
        raise ValueError("Parent population does not cover the complete eligible cache")
    for actual, row in zip(records, expected, strict=True):
        projected = {key: row[key] for key in ("chain_id", "sequence", "length", "resolution")}
        projected.update(sequence_sha256=hashlib.sha256(row["sequence"].encode()).hexdigest(),
                         release_date=manifest_module._extract_release_date(cache[row["chain_id"]]))
        if actual != projected:
            raise ValueError("Parent population differs from independently replayed full-cache eligibility")


class Components:
    def __init__(self, ids: Iterable[str]):
        self.parent = {chain: chain for chain in ids}

    def find(self, chain: str) -> str:
        while self.parent[chain] != chain:
            self.parent[chain] = self.parent[self.parent[chain]]
            chain = self.parent[chain]
        return chain

    def union(self, left: str, right: str) -> None:
        left, right = self.find(left), self.find(right)
        if left != right:
            self.parent[max(left, right)] = min(left, right)


def parse_clusters(path: Path, sequences: set[str]) -> dict[str, str]:
    expected = {sequence_id(sequence) for sequence in sequences}
    mapping = read_groups(path)
    if set(mapping) != expected or not set(mapping.values()) <= expected:
        raise ValueError("Fresh clustering does not cover the complete exact-sequence population")
    if any(mapping[root] != root for root in set(mapping.values())):
        raise ValueError("Fresh cluster representative belongs to a different cluster")
    return mapping


def derive_training_frame(
    records: Sequence[Mapping[str, Any]], inherited_groups: Mapping[str, str],
    frozen_splits: Mapping[str, Sequence[str]], clusters: Mapping[str, str],
    barrier_hits: Iterable[tuple[str, str]],
) -> dict[str, Any]:
    """Replay direct exclusions, then group only the retained training subgraph."""
    if set(frozen_splits) != {"val", "hidden_val"}:
        raise ValueError("Both frozen heldout splits are required")
    by_id = {row["chain_id"]: row for row in records}
    if len(by_id) != len(records):
        raise ValueError("Parent population repeats a chain")
    heldout = set(frozen_splits["val"]) | set(frozen_splits["hidden_val"])
    if (set(frozen_splits["val"]) & set(frozen_splits["hidden_val"]) or not heldout <= set(by_id)
            or not heldout <= set(inherited_groups)):
        raise ValueError("Heldout membership escapes the original graph or overlaps")
    heldout_sequences = {by_id[chain]["sequence"] for chain in heldout}
    heldout_pdbs = {chain.split("_", 1)[0].lower() for chain in heldout}
    heldout_groups = {inherited_groups[chain] for chain in heldout}
    parent_ids = {sequence_id(row["sequence"]) for row in records}
    if set(clusters) != parent_ids:
        raise ValueError("Fresh clustering does not cover the complete parent population")
    hit_parents: set[str] = set()
    hit_queries: set[str] = set()
    seen_pairs: set[tuple[str, str]] = set()
    for query, parent in barrier_hits:
        if query not in heldout or parent not in parent_ids or (query, parent) in seen_pairs:
            raise ValueError("Unknown or duplicate parent barrier alignment")
        seen_pairs.add((query, parent))
        hit_parents.add(parent)
        if parent == sequence_id(by_id[query]["sequence"]):
            hit_queries.add(query)
    if hit_queries != heldout:
        raise ValueError("Exhaustive parent barrier is missing heldout self alignments")
    exclusions: dict[str, list[str]] = {}
    for chain, row in by_id.items():
        reasons = []
        if row["sequence"] in heldout_sequences:
            reasons.append("exact_heldout_sequence")
        if chain.split("_", 1)[0].lower() in heldout_pdbs:
            reasons.append("same_heldout_pdb")
        if inherited_groups.get(chain) in heldout_groups:
            reasons.append("inherited_heldout_group")
        if sequence_id(row["sequence"]) in hit_parents:
            reasons.append("heldout_query_homology")
        if reasons:
            exclusions[chain] = reasons
    retained = sorted(set(by_id) - set(exclusions))
    graph = Components(retained)
    for key in ("sequence", "pdb", "cluster40", "inherited"):
        anchors: dict[str, str] = {}
        for chain in retained:
            row = by_id[chain]
            value = (row["sequence"] if key == "sequence" else chain.split("_", 1)[0].lower() if key == "pdb"
                     else clusters[sequence_id(row["sequence"])] if key == "cluster40" else inherited_groups.get(chain))
            if value is None:
                continue
            if value in anchors:
                graph.union(chain, anchors[value])
            else:
                anchors[value] = chain
    cluster_roots: dict[str, str] = {}
    for chain in retained:
        cluster_roots.setdefault(clusters[sequence_id(by_id[chain]["sequence"])], chain)
    exported = []
    for row in records:
        chain = row["chain_id"]
        value = dict(row)
        value.update(group_id=graph.find(chain) if chain in graph.parent else None,
                     cluster40_id=cluster_roots.get(clusters[sequence_id(row["sequence"])]),
                     training_exclusion_reasons=exclusions.get(chain, []))
        exported.append(value)
    diagnostics = {
        "metadata_eligible_chains": len(records), "metadata_eligible_sequences": len(parent_ids),
        "retained_training_chains": len(retained),
        "retained_training_sequences": len({by_id[chain]["sequence"] for chain in retained}),
        "training_diversity_group_count": len({graph.find(chain) for chain in retained}),
        "exclusion_chain_counts_nonexclusive": dict(Counter(reason for reasons in exclusions.values() for reason in reasons)),
        "excluded_chains": len(exclusions), "barrier_hit_pairs": len(seen_pairs),
        "barrier_hit_parent_sequences": len(hit_parents),
        "maximum_parent_length": max(row["length"] for row in records),
        "training_diversity_groups_are_global_split_components": False,
    }
    return {"policy": POLICY, "records": exported, "eligible_training_ids": retained,
            "frozen_splits": {name: list(ids) for name, ids in frozen_splits.items()}, "diagnostics": diagnostics}


def validate_parent_frame(receipt_path: Path) -> dict[str, Any]:
    """Validate native evidence and independently reconstruct every exported row."""
    return dict(importlib.import_module("scripts.build_full_parent_frame").validate_parent_frame_receipt(receipt_path))
