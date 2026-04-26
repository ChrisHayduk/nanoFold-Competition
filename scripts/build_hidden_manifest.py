from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from build_manifests import (  # noqa: E402
    DEFAULT_STRATIFY_FIELDS,
    Candidate,
    _assert_quality_report,
    _build_candidates,
    _build_quality_report,
    _candidate_field,
    _choose_representative,
    _cluster_assignments_sha256,
    _cluster_sequences_mmseqs,
    _load_cluster_tsv,
    _load_structure_metadata,
    _mmseqs_version,
    _read_hidden_split_salt,
    _salt_digest,
    _sha256,
    _stable_random_score,
    _structure_metadata_source_record,
)


@dataclass(frozen=True)
class SplitArgs:
    hidden_val_size: int
    min_len: int
    max_len: int
    max_resolution: float
    seed: int
    min_seq_id: float
    coverage: float
    mmseqs_bin: str
    cluster_tsv: str
    structure_metadata: str
    expected_chain_cache_sha256: str


@dataclass(frozen=True)
class DisjointComponent:
    representative: Candidate
    members: tuple[Candidate, ...]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=(
            "Generate maintainer-only hidden validation manifests while preserving the "
            "committed public train/public validation manifests."
        )
    )
    ap.add_argument(
        "--chain-data-cache",
        type=Path,
        default=Path("data/openproteinset/pdb_data/data_caches/chain_data_cache.json"),
    )
    ap.add_argument("--lock-file", type=Path, default=Path("leaderboard/official_manifest_source.lock.json"))
    ap.add_argument("--public-train-manifest", type=Path, default=Path("data/manifests/train.txt"))
    ap.add_argument("--public-val-manifest", type=Path, default=Path("data/manifests/val.txt"))
    ap.add_argument("--public-all-manifest", type=Path, default=Path("data/manifests/all.txt"))
    ap.add_argument("--hidden-out-dir", type=Path, default=Path(".nanofold_private/manifests"))
    ap.add_argument(
        "--private-lock-file",
        type=Path,
        default=Path(".nanofold_private/leaderboard/private_hidden_manifest_source.lock.json"),
    )
    ap.add_argument(
        "--private-processability-exclusion-list",
        type=Path,
        default=Path(".nanofold_private/manifests/hidden_processability_exclusions.txt"),
        help=(
            "Maintainer-only chain ID list to exclude from hidden selection after private "
            "preprocessing failures. This file must stay outside tracked public files."
        ),
    )
    ap.add_argument("--hidden-split-salt-env", type=str, default="NANOFOLD_HIDDEN_SPLIT_SALT")
    ap.add_argument("--hidden-split-salt-file", type=Path, default=None)
    ap.add_argument("--hidden-val-size", type=int, default=None)
    ap.add_argument("--structure-metadata", type=Path, default=None)
    ap.add_argument("--cluster-tsv", type=Path, default=None)
    ap.add_argument("--mmseqs-bin", type=str, default=None)
    ap.add_argument("--seed", type=int, default=None)
    return ap.parse_args()


def _read_manifest(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"Missing manifest: {path}")
    ids = [line.strip() for line in path.read_text().splitlines() if line.strip() and not line.startswith("#")]
    duplicates = sorted(chain_id for chain_id, count in Counter(ids).items() if count > 1)
    if duplicates:
        raise ValueError(f"Manifest contains duplicate chain IDs: {path}; examples: {duplicates[:8]}")
    return ids


def _read_optional_chain_list(path: Path) -> list[str]:
    if not path.exists():
        return []
    return [line.strip() for line in path.read_text().splitlines() if line.strip() and not line.startswith("#")]


def _load_lock_args(path: Path) -> SplitArgs:
    raw = json.loads(path.read_text())
    if not isinstance(raw, dict) or not isinstance(raw.get("args"), dict):
        raise ValueError(f"Manifest lock must contain an args object: {path}")
    args = raw["args"]
    required = (
        "hidden_val_size",
        "min_len",
        "max_len",
        "max_resolution",
        "seed",
        "min_seq_id",
        "coverage",
        "mmseqs_bin",
        "structure_metadata",
    )
    for key in required:
        if key not in args:
            raise ValueError(f"Manifest lock missing args.{key}: {path}")
    expected_sha = str(raw.get("chain_data_cache_sha256") or args.get("expected_chain_cache_sha256") or "")
    return SplitArgs(
        hidden_val_size=int(args["hidden_val_size"]),
        min_len=int(args["min_len"]),
        max_len=int(args["max_len"]),
        max_resolution=float(args["max_resolution"]),
        seed=int(args["seed"]),
        min_seq_id=float(args["min_seq_id"]),
        coverage=float(args["coverage"]),
        mmseqs_bin=str(args["mmseqs_bin"]),
        cluster_tsv=str(args.get("cluster_tsv") or ""),
        structure_metadata=str(args["structure_metadata"]),
        expected_chain_cache_sha256=expected_sha,
    )


def _with_cluster_ids(candidates: list[Candidate], cluster_map: dict[str, str]) -> list[Candidate]:
    out: list[Candidate] = []
    for candidate in candidates:
        out.append(
            Candidate(
                chain_id=candidate.chain_id,
                sequence=candidate.sequence,
                length=candidate.length,
                resolution=candidate.resolution,
                release_date=candidate.release_date,
                cluster_id=cluster_map[candidate.chain_id],
                pdb_id=candidate.pdb_id,
                secondary_structure_class=candidate.secondary_structure_class,
                secondary_helix_fraction=candidate.secondary_helix_fraction,
                secondary_beta_fraction=candidate.secondary_beta_fraction,
                secondary_coil_fraction=candidate.secondary_coil_fraction,
                domain_architecture_class=candidate.domain_architecture_class,
                domain_architecture_source=candidate.domain_architecture_source,
                metadata_source_count=candidate.metadata_source_count,
            )
        )
    return out


def _componentize(
    candidates: list[Candidate],
    cluster_map: dict[str, str],
    *,
    seed: int,
) -> list[DisjointComponent]:
    chain_ids = sorted(candidate.chain_id for candidate in candidates)
    parent = {chain_id: chain_id for chain_id in chain_ids}

    def find(chain_id: str) -> str:
        while parent[chain_id] != chain_id:
            parent[chain_id] = parent[parent[chain_id]]
            chain_id = parent[chain_id]
        return chain_id

    def union(left: str, right: str) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    by_cluster: defaultdict[str, list[str]] = defaultdict(list)
    by_pdb: defaultdict[str, list[str]] = defaultdict(list)
    for candidate in candidates:
        by_cluster[cluster_map[candidate.chain_id]].append(candidate.chain_id)
        by_pdb[candidate.pdb_id].append(candidate.chain_id)
    for members in by_cluster.values():
        first = members[0]
        for member in members[1:]:
            union(first, member)
    for members in by_pdb.values():
        first = members[0]
        for member in members[1:]:
            union(first, member)

    by_component: defaultdict[str, list[Candidate]] = defaultdict(list)
    candidate_by_id = {candidate.chain_id: candidate for candidate in candidates}
    for chain_id in chain_ids:
        by_component[find(chain_id)].append(candidate_by_id[chain_id])

    components = [
        DisjointComponent(
            representative=_choose_representative(members, seed),
            members=tuple(sorted(members, key=lambda item: item.chain_id)),
        )
        for members in by_component.values()
    ]
    return sorted(components, key=lambda component: component.representative.chain_id)


def _stratum(candidate: Candidate, fields: tuple[str, ...]) -> tuple[str, ...]:
    if not fields:
        return ("all",)
    return tuple(_candidate_field(candidate, field_name) for field_name in fields)


def _allocate_against_reference(
    pool: list[Candidate],
    reference: list[Candidate],
    *,
    size: int,
    stratify_fields: tuple[str, ...],
) -> dict[tuple[str, ...], int]:
    if size <= 0:
        return {}
    if size > len(pool):
        raise ValueError(f"Requested {size} hidden chains but only {len(pool)} disjoint public-excluded units exist.")

    by_pool: defaultdict[tuple[str, ...], list[Candidate]] = defaultdict(list)
    for candidate in pool:
        by_pool[_stratum(candidate, stratify_fields)].append(candidate)
    reference_counts = Counter(_stratum(candidate, stratify_fields) for candidate in reference)
    reference_total = sum(reference_counts.values())
    if reference_total <= 0:
        raise ValueError("Cannot stratify hidden split without public reference candidates.")

    allocation: dict[tuple[str, ...], int] = {}
    remainders: list[tuple[float, int, int, tuple[str, ...]]] = []
    allocated = 0
    for key, members in sorted(by_pool.items()):
        exact = size * reference_counts.get(key, 0) / float(reference_total)
        count = min(len(members), int(exact))
        allocation[key] = count
        allocated += count
        remainders.append((exact - count, reference_counts.get(key, 0), len(members), key))

    remaining = size - allocated
    for _remainder, _reference_count, _capacity, key in sorted(
        remainders,
        key=lambda item: (-item[0], -item[1], -item[2], item[3]),
    ):
        if remaining <= 0:
            break
        if reference_counts.get(key, 0) <= 0:
            continue
        capacity = len(by_pool[key]) - allocation[key]
        if capacity <= 0:
            continue
        allocation[key] += 1
        remaining -= 1

    if remaining > 0:
        for key, members in sorted(
            by_pool.items(),
            key=lambda item: (-reference_counts.get(item[0], 0), -len(item[1]), item[0]),
        ):
            if remaining <= 0:
                break
            capacity = len(members) - allocation[key]
            if capacity <= 0:
                continue
            take = min(capacity, remaining)
            allocation[key] += take
            remaining -= take

    if remaining > 0:
        raise RuntimeError(f"Internal allocation error: {remaining} hidden slots were not assigned.")
    return allocation


def _sample_by_allocation(
    pool: list[Candidate],
    allocation: dict[tuple[str, ...], int],
    *,
    seed: int,
    salt: str,
    stratify_fields: tuple[str, ...],
) -> list[Candidate]:
    by_stratum: defaultdict[tuple[str, ...], list[Candidate]] = defaultdict(list)
    for candidate in pool:
        by_stratum[_stratum(candidate, stratify_fields)].append(candidate)

    selected: list[Candidate] = []
    for key, count in sorted(allocation.items()):
        members = sorted(
            by_stratum[key],
            key=lambda item: (_stable_random_score(seed, f"{salt}:{'|'.join(key)}", item.chain_id), item.chain_id),
        )
        selected.extend(members[:count])
    expected = sum(allocation.values())
    if len(selected) != expected:
        raise RuntimeError(f"Internal sampling error: requested {expected}, selected {len(selected)}")
    return sorted(selected, key=lambda item: (_stable_random_score(seed, f"{salt}:selected", item.chain_id), item.chain_id))


def _assert_public_manifests(train_ids: list[str], val_ids: list[str], all_ids: list[str]) -> None:
    train_set = set(train_ids)
    val_set = set(val_ids)
    all_set = set(all_ids)
    if train_set & val_set:
        raise ValueError("Public train and public validation manifests overlap.")
    if all_set != train_set | val_set:
        raise ValueError("Public all.txt must equal the union of train.txt and val.txt.")


def _build_hidden_manifest(args: argparse.Namespace) -> dict[str, Any]:
    lock_args = _load_lock_args(args.lock_file)
    hidden_val_size = int(args.hidden_val_size if args.hidden_val_size is not None else lock_args.hidden_val_size)
    seed = int(args.seed if args.seed is not None else lock_args.seed)
    structure_metadata_path = args.structure_metadata or Path(lock_args.structure_metadata)
    mmseqs_bin = str(args.mmseqs_bin if args.mmseqs_bin is not None else lock_args.mmseqs_bin)
    cluster_tsv = str(args.cluster_tsv if args.cluster_tsv is not None else lock_args.cluster_tsv)

    train_ids = _read_manifest(args.public_train_manifest)
    val_ids = _read_manifest(args.public_val_manifest)
    all_ids = _read_manifest(args.public_all_manifest)
    _assert_public_manifests(train_ids, val_ids, all_ids)
    public_ids = set(all_ids)
    private_processability_exclusions = set(_read_optional_chain_list(args.private_processability_exclusion_list))
    if private_processability_exclusions & public_ids:
        raise ValueError("Private hidden processability exclusions must not contain public manifest chains.")

    cache_sha = _sha256(args.chain_data_cache)
    expected_sha = lock_args.expected_chain_cache_sha256.strip().lower()
    if expected_sha:
        if cache_sha != expected_sha:
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
    candidates, reject_reasons, structural_metadata_source = _build_candidates(
        data,
        min_len=lock_args.min_len,
        max_len=lock_args.max_len,
        max_resolution=lock_args.max_resolution,
        structure_metadata=structure_metadata,
        structure_metadata_accepted=structure_metadata_accepted,
    )
    candidate_by_id = {candidate.chain_id: candidate for candidate in candidates}
    missing_public = sorted(public_ids - set(candidate_by_id))
    if missing_public:
        raise ValueError(
            f"{len(missing_public)} public manifest chains are not eligible candidates. Examples: {missing_public[:8]}"
        )

    candidate_ids = [candidate.chain_id for candidate in candidates]
    candidate_sequences = {candidate.chain_id: candidate.sequence for candidate in candidates}
    cluster_method = "mmseqs2"
    cluster_command: list[str] | None = None
    if cluster_tsv:
        cluster_method = "precomputed_tsv"
        cluster_map = _load_cluster_tsv(cluster_tsv, set(candidate_ids))
    else:
        cluster_map_or_none, cluster_command = _cluster_sequences_mmseqs(
            sequences={chain_id: candidate_sequences[chain_id] for chain_id in candidate_ids},
            mmseqs_bin=mmseqs_bin,
            min_seq_id=lock_args.min_seq_id,
            coverage=lock_args.coverage,
        )
        if cluster_map_or_none is None:
            raise RuntimeError(
                "Hidden manifest generation requires MMseqs2 clustering or a locked --cluster-tsv. "
                f"Install `{mmseqs_bin}` or provide a TSV generated with --min-seq-id "
                f"{lock_args.min_seq_id} and coverage {lock_args.coverage}."
            )
        cluster_map = cluster_map_or_none

    clustered_candidates = _with_cluster_ids(candidates, cluster_map)
    candidate_by_id = {candidate.chain_id: candidate for candidate in clustered_candidates}
    components = _componentize(clustered_candidates, cluster_map, seed=seed)

    public_components = [
        component
        for component in components
        if any(member.chain_id in public_ids for member in component.members)
    ]
    hidden_pool = [
        component.representative
        for component in components
        if not any(member.chain_id in public_ids for member in component.members)
        and not any(member.chain_id in private_processability_exclusions for member in component.members)
    ]
    if hidden_val_size > len(hidden_pool):
        raise ValueError(
            f"Not enough public-disjoint hidden units: need {hidden_val_size}, have {len(hidden_pool)}."
        )

    hidden_split_salt = _read_hidden_split_salt(
        salt_env=str(args.hidden_split_salt_env),
        salt_file=str(args.hidden_split_salt_file or ""),
    )
    hidden_split_salt_digest = _salt_digest(hidden_split_salt)
    stratify_fields = DEFAULT_STRATIFY_FIELDS
    public_reference = [candidate_by_id[chain_id] for chain_id in train_ids + val_ids]
    allocation = _allocate_against_reference(
        hidden_pool,
        public_reference,
        size=hidden_val_size,
        stratify_fields=stratify_fields,
    )
    hidden = _sample_by_allocation(
        hidden_pool,
        allocation,
        seed=seed,
        salt=f"hidden_val_fixed_public:{hidden_split_salt_digest}",
        stratify_fields=stratify_fields,
    )
    hidden_ids = [candidate.chain_id for candidate in hidden]
    hidden_set = set(hidden_ids)
    if hidden_set & public_ids:
        raise RuntimeError("Hidden split overlaps public manifests.")

    public_clusters = {candidate_by_id[chain_id].cluster_id for chain_id in public_ids}
    hidden_clusters = {candidate_by_id[chain_id].cluster_id for chain_id in hidden_ids}
    cluster_overlap = public_clusters & hidden_clusters
    if cluster_overlap:
        raise RuntimeError(f"Hidden split leaks public sequence clusters: {sorted(cluster_overlap)[:8]}")

    public_pdbs = {candidate_by_id[chain_id].pdb_id for chain_id in public_ids}
    hidden_pdbs = {candidate_by_id[chain_id].pdb_id for chain_id in hidden_ids}
    pdb_overlap = public_pdbs & hidden_pdbs
    if pdb_overlap:
        raise RuntimeError(f"Hidden split leaks public PDB entries: {sorted(pdb_overlap)[:8]}")

    selected_candidates = {
        chain_id: candidate_by_id[chain_id]
        for chain_id in train_ids + val_ids + hidden_ids
    }
    quality_report = _build_quality_report(
        selected_candidates=selected_candidates,
        split_ids={"train": train_ids, "val": val_ids, "hidden_val": hidden_ids},
        stratify_fields=stratify_fields,
    )
    _assert_quality_report(quality_report)

    args.hidden_out_dir.mkdir(parents=True, exist_ok=True)
    hidden_manifest = args.hidden_out_dir / "hidden_val.txt"
    split_quality_report = args.hidden_out_dir / "split_quality_report.json"
    hidden_manifest.write_text("\n".join(hidden_ids) + "\n")
    split_quality_report.write_text(json.dumps(quality_report, indent=2, sort_keys=True) + "\n")

    private_lock = {
        "notes": "Maintainer-only hidden manifest lock. Do not commit this file.",
        "chain_data_cache_sha256": cache_sha,
        "hidden_split_salt_sha256": hidden_split_salt_digest,
        "hidden_val_count": len(hidden_ids),
        "hidden_val_manifest": str(hidden_manifest),
        "hidden_val_manifest_sha256": _sha256(hidden_manifest),
        "split_quality_report": str(split_quality_report),
        "split_quality_report_sha256": _sha256(split_quality_report),
        "public_contract": {
            "train_manifest": str(args.public_train_manifest),
            "train_manifest_sha256": _sha256(args.public_train_manifest),
            "train_count": len(train_ids),
            "val_manifest": str(args.public_val_manifest),
            "val_manifest_sha256": _sha256(args.public_val_manifest),
            "val_count": len(val_ids),
            "all_manifest": str(args.public_all_manifest),
            "all_manifest_sha256": _sha256(args.public_all_manifest),
            "all_count": len(all_ids),
        },
        "filtering": {
            "candidate_count": len(candidates),
            "disjoint_unit_count": len(components),
            "public_component_count": len(public_components),
            "hidden_candidate_unit_count": len(hidden_pool),
            "private_processability_exclusion_count": len(private_processability_exclusions),
            "reject_reasons": dict(sorted(reject_reasons.items())),
        },
        "structure_metadata": _structure_metadata_source_record(structure_metadata_path),
        "clustering": {
            "method": cluster_method,
            "cluster_count": len(set(cluster_map.values())),
            "command": cluster_command,
            "mmseqs": _mmseqs_version(mmseqs_bin) if cluster_method == "mmseqs2" else None,
            "cluster_assignments_sha256": _cluster_assignments_sha256(cluster_map),
        },
        "grouping": {
            "cluster_disjoint": True,
            "pdb_disjoint": True,
            "public_components_excluded": True,
        },
        "stratification": {
            "fields": list(stratify_fields),
            "structural_metadata_source": structural_metadata_source,
            "reference": "public_train_plus_public_val",
            "allocation": {"|".join(key): count for key, count in sorted(allocation.items())},
            "quality_report": quality_report,
        },
    }
    args.private_lock_file.parent.mkdir(parents=True, exist_ok=True)
    args.private_lock_file.write_text(json.dumps(private_lock, indent=2, sort_keys=True) + "\n")

    return {
        "hidden_manifest": hidden_manifest,
        "hidden_count": len(hidden_ids),
        "private_lock": args.private_lock_file,
        "quality_report": split_quality_report,
        "hidden_manifest_sha256": private_lock["hidden_val_manifest_sha256"],
    }


def main() -> None:
    result = _build_hidden_manifest(parse_args())
    print(
        "Wrote "
        f"{result['hidden_count']} hidden validation chains to {result['hidden_manifest']} "
        "with zero public chain/cluster/PDB overlap."
    )
    print(f"Wrote private hidden manifest lock: {result['private_lock']}")
    print(f"Wrote hidden split quality report: {result['quality_report']}")
    print(f"hidden_val.txt SHA256: {result['hidden_manifest_sha256']}")


if __name__ == "__main__":
    main()
