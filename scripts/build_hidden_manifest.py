from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from build_homology_groups import validate_group_receipt  # noqa: E402
from build_manifests import (  # noqa: E402
    CATH_SAMPLING_POLICIES,
    DEFAULT_STRATIFY_FIELDS,
    GROUP_ALLOCATOR_POLICY,
    GROUP_SAMPLING_POLICIES,
    PROCESSABLE_GROUP_SAMPLING_POLICY,
    Candidate,
    _allocate_hierarchical,
    _assert_quality_report,
    _atomic_write,
    _build_candidates,
    _build_quality_report,
    _candidate_field,
    _choose_representative,
    _cluster_assignments_sha256,
    _cluster_sequences_mmseqs,
    _group_sampling_units,
    _load_cluster_tsv,
    _load_structure_metadata,
    _mmseqs_version,
    _original_unit_memberships,
    _plan_group_splits,
    _qualified_group_sampling_units,
    _read_hidden_split_salt,
    _salt_digest,
    _sample_hidden_groups,
    _select_stratified,
    _sha256,
    _stable_random_score,
    _structure_metadata_source_record,
    _validate_generation_args,
    _validate_metadata_identity,
    validate_author_chain_id,
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
    sampling_policy: str = "legacy-v1"
    require_monomer: bool = False
    train_size: int = 1
    val_size: int = 1
    split_group_tsv: str = ""
    split_group_receipt: str = ""
    representative_eligibility_receipt: str = ""
    approved_group_exclusions: str = ""


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
    ap.add_argument("--sampling-policy", choices=sorted(CATH_SAMPLING_POLICIES), required=True,
                    help="Must match the versioned public source lock; writes a new private hidden version.")
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
        default=None,
        help=(
            "Maintainer-only chain ID list to exclude from hidden selection after private "
            "preprocessing failures. A supplied path must exist; omission means no private exclusions. "
            "This file must stay outside tracked public files."
        ),
    )
    ap.add_argument("--hidden-split-salt-env", type=str, default="NANOFOLD_HIDDEN_SPLIT_SALT")
    ap.add_argument("--hidden-split-salt-file", type=Path, default=None)
    ap.add_argument("--hidden-val-size", type=int, default=None)
    ap.add_argument("--structure-metadata", type=Path, default=None)
    ap.add_argument("--cluster-tsv", type=Path, default=None)
    ap.add_argument("--split-group-tsv", type=Path, default=None,
                    help="Relocated cath-v3/cath-v4 full-universe group assignments; bytes must match the source lock.")
    ap.add_argument("--split-group-receipt", type=Path, default=None,
                    help="Relocated cath-v3/cath-v4 group-construction receipt; bytes must match the source lock.")
    ap.add_argument("--representative-eligibility-receipt", type=Path, default=None,
                    help="Relocated cath-v4 representative qualification receipt; bytes must match the source lock.")
    ap.add_argument("--approved-group-exclusions", type=Path, default=None,
                    help="Relocated cath-v4 approved whole-group exclusion catalogue; bytes must match the source lock.")
    ap.add_argument("--mmseqs-bin", type=str, default=None)
    ap.add_argument("--seed", type=int, default=None)
    return ap.parse_args()


def _read_manifest(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"Missing manifest: {path}")
    ids = [line.strip() for line in path.read_text().splitlines() if line.strip() and not line.startswith("#")]
    duplicates = sorted(chain_id for chain_id, count in Counter(ids).items() if count > 1)
    if duplicates:
        raise ValueError(f"Manifest contains {len(duplicates)} duplicate chain IDs: {path}")
    return ids


def _read_optional_chain_list(path: Path | None) -> list[str]:
    if path is None:
        return []
    if not path.is_file():
        raise ValueError("A supplied private processability exclusion list must be an existing file.")
    ids = [line.strip() for line in path.read_text().splitlines() if line.strip() and not line.lstrip().startswith("#")]
    for chain_id in ids:
        validate_author_chain_id(chain_id)
    return ids


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
    sampling_policy = raw.get("sampling_policy", args.get("sampling_policy", "legacy-v1"))
    if sampling_policy not in {"legacy-v1", *CATH_SAMPLING_POLICIES}:
        raise ValueError("Manifest lock has an unsupported sampling policy.")
    if args.get("sampling_policy", sampling_policy) != sampling_policy:
        raise ValueError("Manifest lock sampling policy disagrees with args.sampling_policy.")
    if sampling_policy in CATH_SAMPLING_POLICIES:
        for key in ("sampling_policy", "train_size", "val_size", "require_monomer", "stratify_fields"):
            if key not in args:
                raise ValueError(f"Manifest lock missing args.{key}: {path}")
        if args["stratify_fields"] != list(DEFAULT_STRATIFY_FIELDS):
            raise ValueError("Manifest lock args.stratify_fields disagrees with CATH sampling policy.")
    if sampling_policy in GROUP_SAMPLING_POLICIES:
        for key in ("split_group_tsv", "split_group_receipt"):
            if not isinstance(args.get(key), str) or not args[key].strip():
                raise ValueError(f"Manifest lock missing args.{key} for {sampling_policy}.")
    if sampling_policy == PROCESSABLE_GROUP_SAMPLING_POLICY:
        for key in ("representative_eligibility_receipt", "approved_group_exclusions"):
            if not isinstance(args.get(key), str) or not args[key].strip():
                raise ValueError(f"Manifest lock missing args.{key} for cath-v4.")
    _validate_generation_args(argparse.Namespace(**{**args, "require_monomer": args.get("require_monomer", False)}))
    expected_sha = str(raw.get("chain_data_cache_sha256") or args.get("expected_chain_cache_sha256") or "")
    if sampling_policy in CATH_SAMPLING_POLICIES:
        recorded_sha = raw.get("chain_data_cache_sha256")
        if not isinstance(recorded_sha, str) or len(recorded_sha) != 64 or any(
            char not in "0123456789abcdef" for char in recorded_sha
        ):
            raise ValueError("Manifest lock chain_data_cache_sha256 is missing or invalid.")
        if args.get("expected_chain_cache_sha256") not in (None, "", recorded_sha):
            raise ValueError("Manifest lock cache digest disagrees with args.expected_chain_cache_sha256.")
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
        sampling_policy=sampling_policy,
        require_monomer=args.get("require_monomer", False),
        train_size=int(args.get("train_size", 1)),
        val_size=int(args.get("val_size", 1)),
        split_group_tsv=str(args.get("split_group_tsv") or ""),
        split_group_receipt=str(args.get("split_group_receipt") or ""),
        representative_eligibility_receipt=str(args.get("representative_eligibility_receipt") or ""),
        approved_group_exclusions=str(args.get("approved_group_exclusions") or ""),
    )


def _validate_public_sampling_contract(
    lock_path: Path, lock_args: SplitArgs, components: list[DisjointComponent],
    train_ids: list[str], val_ids: list[str],
    *, split_group_map: dict[str, str] | None = None, eligibility: dict[str, Any] | None = None,
) -> list[list[Candidate]] | None:
    """Replay public selection so a lock cannot silently describe another sampling run."""
    if lock_args.sampling_policy not in CATH_SAMPLING_POLICIES:
        return None
    if len(train_ids) != lock_args.train_size or len(val_ids) != lock_args.val_size:
        raise ValueError("Public manifest counts disagree with locked sampling sizes.")
    lock = json.loads(lock_path.read_text())
    filtering = lock.get("filtering", {})
    expected = {
        "candidate_count": sum(len(component.members) for component in components),
        "disjoint_unit_count": len(components),
        "target_policy": "known_monomer" if lock_args.require_monomer else "single_chain",
        "representative_policy": "best_resolution_then_longest_then_release_date_seeded_tie",
    }
    if lock_args.sampling_policy == PROCESSABLE_GROUP_SAMPLING_POLICY:
        if eligibility is None:
            raise ValueError("cath-v4 requires validated representative eligibility.")
        expected["representative_policy"] = "source_validated_best_resolution_then_longest_then_release_date_seeded_tie"
        expected["selectable_unit_count"] = sum(value is not None for value in eligibility["selected_chain_by_original_unit"].values())
    if not isinstance(filtering, dict) or any(filtering.get(key) != value for key, value in expected.items()):
        raise ValueError("Public source lock candidate universe or representative policy is missing or inconsistent.")
    grouping = lock.get("grouping", {})
    if not isinstance(grouping, dict) or any(grouping.get(key) is not True for key in (
        "cluster_disjoint", "pdb_disjoint", "exact_sequence_disjoint",
    )):
        raise ValueError("Public source lock grouping policy disagrees with CATH sampling policy.")
    units = [component.representative for component in components]
    if lock_args.sampling_policy in GROUP_SAMPLING_POLICIES:
        if split_group_map is None:
            raise ValueError("Group sampling requires complete split-group assignments.")
        candidates = [member for component in components for member in component.members]
        cluster_map = {candidate.chain_id: candidate.cluster_id for candidate in candidates}
        original_ids_digest = hashlib.sha256(
            "".join(f"{chain}\n" for chain in sorted(candidate.chain_id for candidate in units)).encode()
        ).hexdigest()
        if eligibility is not None:
            units, groups = _qualified_group_sampling_units(
                candidates, cluster_map, split_group_map, seed=lock_args.seed,
                selected_chain_by_original_unit=eligibility["selected_chain_by_original_unit"],
                approved_excluded_group_ids=set(eligibility["approved_excluded_group_ids"]),
            )
        else:
            units, groups = _group_sampling_units(candidates, cluster_map, split_group_map, seed=lock_args.seed)
        record = lock.get("split_groups", {})
        expected_groups = {
            "group_count": len(set(split_group_map.values())), "selection_unit_count": len(units), "coverage": "all_groups",
            "allocator_policy": GROUP_ALLOCATOR_POLICY,
            "selection_unit_ids_sha256": hashlib.sha256(
                "".join(f"{chain}\n" for chain in sorted(candidate.chain_id for candidate in units)).encode()
            ).hexdigest(),
        }
        if eligibility is not None:
            expected_groups.update({
                "coverage": "all_retained_groups", "retained_group_count": len(groups),
                "approved_excluded_group_count": len(eligibility["approved_excluded_group_ids"]),
                "original_unit_count": len(components), "original_representative_ids_sha256": original_ids_digest,
            })
        if not isinstance(record, dict) or any(record.get(key) != value for key, value in expected_groups.items()):
            raise ValueError("Public source lock split-group population or coverage requirement is inconsistent.")
        coverage_field = "all_retained_groups_covered" if eligibility is not None else "all_groups_covered"
        if grouping.get("full_group_disjoint") is not True or grouping.get(coverage_field) is not True:
            raise ValueError("Public source lock must require disjoint full groups and complete group coverage.")
        train, val, hidden_groups = _plan_group_splits(
            units, groups, train_size=lock_args.train_size, val_size=lock_args.val_size,
            hidden_val_size=lock_args.hidden_val_size, seed=lock_args.seed,
        )
        if val_ids != [candidate.chain_id for candidate in val] or train_ids != [candidate.chain_id for candidate in train]:
            raise ValueError("Public manifests do not reproduce the locked full-group sampling policy.")
        return hidden_groups
    val = _select_stratified(units, size=lock_args.val_size, seed=lock_args.seed, salt="val",
                             stratify_fields=DEFAULT_STRATIFY_FIELDS)
    val_set = {candidate.chain_id for candidate in val}
    train = _select_stratified([candidate for candidate in units if candidate.chain_id not in val_set],
                               size=lock_args.train_size, seed=lock_args.seed, salt="train",
                               stratify_fields=DEFAULT_STRATIFY_FIELDS)
    if val_ids != [candidate.chain_id for candidate in val] or train_ids != [candidate.chain_id for candidate in train]:
        raise ValueError("Public manifests do not reproduce the locked seed, representative, and sampling policy.")
    return None


def _with_cluster_ids(candidates: list[Candidate], cluster_map: dict[str, str]) -> list[Candidate]:
    return [replace(candidate, cluster_id=cluster_map[candidate.chain_id]) for candidate in candidates]


def _load_split_group_map(
    lock_path: Path, lock_args: SplitArgs, candidate_ids: set[str], *, override_path: Path | None,
    receipt_override_path: Path | None, chain_data_cache_path: Path, structure_metadata_path: Path,
    cluster_map: dict[str, str],
) -> dict[str, str] | None:
    if lock_args.sampling_policy not in GROUP_SAMPLING_POLICIES:
        if override_path is not None or receipt_override_path is not None:
            raise ValueError("Split-group overrides require a group-sampling source lock.")
        return None
    path = override_path or Path(lock_args.split_group_tsv)
    record = json.loads(lock_path.read_text()).get("split_groups")
    if not isinstance(record, dict) or not path.is_file() or record.get("sha256") != _sha256(path):
        raise ValueError("Split-group TSV SHA256 disagrees with source lock.")
    group_map = _load_cluster_tsv(path, candidate_ids)
    if record.get("assignments_sha256") != _cluster_assignments_sha256(group_map):
        raise ValueError("Split-group assignments SHA256 disagrees with source lock.")
    receipt_path = receipt_override_path or Path(lock_args.split_group_receipt)
    if not receipt_path.is_file() or record.get("receipt_sha256") != _sha256(receipt_path):
        raise ValueError("Split-group receipt SHA256 disagrees with source lock.")
    receipt = validate_group_receipt(
        receipt_path, split_group_tsv=path, chain_data_cache=chain_data_cache_path,
        structure_metadata=structure_metadata_path, cluster_map=cluster_map,
        candidate_ids=candidate_ids, min_seq_id=lock_args.min_seq_id, coverage=lock_args.coverage,
    )
    unit_count_field = "original_unit_count" if lock_args.sampling_policy == PROCESSABLE_GROUP_SAMPLING_POLICY else "selection_unit_count"
    if receipt.get("original_unit_count") != record.get(unit_count_field):
        raise ValueError("Split-group receipt original-unit count disagrees with source lock.")
    return group_map


def _load_representative_eligibility(
    args: argparse.Namespace, lock_args: SplitArgs, candidates: list[Candidate], *,
    cluster_map: dict[str, str], split_group_map: dict[str, str] | None, structure_metadata_path: Path,
) -> dict[str, Any] | None:
    receipt_override = getattr(args, "representative_eligibility_receipt", None)
    exclusions_override = getattr(args, "approved_group_exclusions", None)
    if lock_args.sampling_policy != PROCESSABLE_GROUP_SAMPLING_POLICY:
        if receipt_override is not None or exclusions_override is not None:
            raise ValueError("Representative eligibility overrides require a cath-v4 source lock.")
        return None
    from nanofold.representative_eligibility import validate_representative_eligibility_receipt

    if split_group_map is None:
        raise ValueError("cath-v4 requires the unchanged complete split-group graph.")
    receipt_path = receipt_override or Path(lock_args.representative_eligibility_receipt)
    exclusions_path = exclusions_override or Path(lock_args.approved_group_exclusions)
    binding = json.loads(args.lock_file.read_text()).get("representative_eligibility")
    if not isinstance(binding, dict):
        raise ValueError("Source lock representative eligibility binding is missing or invalid.")
    for path, field in ((receipt_path, "receipt_sha256"), (exclusions_path, "approved_group_exclusions_sha256")):
        if not path.is_file() or binding.get(field) != _sha256(path):
            raise ValueError(f"Representative eligibility {field} disagrees with source lock.")
    original_units = _original_unit_memberships(candidates, cluster_map, seed=lock_args.seed)
    eligibility = validate_representative_eligibility_receipt(
        receipt_path, approved_group_exclusions_path=exclusions_path,
        chain_data_cache_path=args.chain_data_cache, structure_metadata_path=structure_metadata_path,
        cluster_map=cluster_map, split_group_map=split_group_map,
        split_group_receipt_path=getattr(args, "split_group_receipt", None) or Path(lock_args.split_group_receipt),
        original_units={unit_id: [member.chain_id for member in members] for unit_id, members in original_units.items()},
        seed=lock_args.seed,
    )
    if eligibility["binding"] != binding:
        raise ValueError("Source lock representative eligibility provenance does not match the validated receipt.")
    return eligibility


def _validate_group_selection(
    components: list[DisjointComponent], split_group_map: dict[str, str],
    splits: dict[str, list[str]], private_exclusions: set[str],
    *, eligibility: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Check group coverage and ownership independently of the selection algorithm."""
    universe = {member.chain_id for component in components for member in component.members}
    if set(split_group_map) != universe:
        raise ValueError("Split-group assignments must cover the complete eligible universe exactly.")
    selected_ids = [chain_id for chain_ids in splits.values() for chain_id in chain_ids]
    selected = set(selected_ids)
    if len(selected) != len(selected_ids):
        raise RuntimeError("Selected targets contain repeated chain IDs.")
    representatives = {component.representative.chain_id for component in components}
    original_groups = set(split_group_map.values())
    approved_excluded_groups = set() if eligibility is None else set(eligibility["approved_excluded_group_ids"])
    if not approved_excluded_groups <= original_groups:
        raise RuntimeError("Approved exclusions reference groups outside the original complete graph.")
    if any(split_group_map[chain] in approved_excluded_groups for chain in selected if chain in split_group_map):
        raise RuntimeError("Selected manifests contain an approved excluded full-group member.")
    if eligibility is not None:
        qualified = [chain for chain in eligibility["selected_chain_by_original_unit"].values() if chain is not None]
        representatives = set(qualified)
        if len(representatives) != len(qualified) or not representatives <= universe:
            raise RuntimeError("Qualified representatives must be unique members of the original eligible universe.")
    if not selected <= representatives:
        raise RuntimeError("Selected targets must be the locked original-unit quality representatives.")
    for component in components:
        if len({split_group_map[member.chain_id] for member in component.members}) != 1:
            raise RuntimeError("A full split group divides an original cluster/PDB/exact-sequence unit.")
        if len({member.chain_id for member in component.members} & selected) > 1:
            raise RuntimeError("Selected targets repeat an original cluster/PDB/exact-sequence unit.")
    owners: defaultdict[str, set[str]] = defaultdict(set)
    for split_name, chain_ids in splits.items():
        for chain_id in chain_ids:
            owners[split_group_map[chain_id]].add(split_name)
    overlapping = sum(len(split_names) > 1 for split_names in owners.values())
    if overlapping:
        raise RuntimeError(f"Selected splits share {overlapping} full homology groups.")
    retained_groups = original_groups - approved_excluded_groups
    missing = len(retained_groups - owners.keys())
    if missing:
        raise RuntimeError(f"Selected manifests leave {missing} full homology groups uncovered.")
    hidden_groups = {split_group_map[chain_id] for chain_id in splits["hidden_val"]}
    excluded_groups = {split_group_map[chain_id] for chain_id in private_exclusions if chain_id in split_group_map}
    if hidden_groups & excluded_groups:
        raise RuntimeError("Private exclusions touch fixed hidden full groups; complete group coverage cannot be preserved.")
    if eligibility is not None:
        return {
            "required": "all_retained_groups", "retained_group_coverage_verified": True,
            "original_group_count": len(original_groups), "approved_excluded_group_count": len(approved_excluded_groups),
            "retained_group_count": len(retained_groups), "covered_group_count": len(owners),
            "original_unit_count": len(components), "selection_unit_count": len(representatives),
            "selection_unit_ids_sha256": hashlib.sha256(
                "".join(f"{chain}\n" for chain in sorted(representatives)).encode()
            ).hexdigest(),
            "split_group_counts": {name: sum(name in split_names for split_names in owners.values()) for name in splits},
        }
    return {
        "required": "all_groups", "all_group_coverage_verified": True,
        "complete_group_count": len(set(split_group_map.values())), "covered_group_count": len(owners),
        "selection_unit_count": len(components),
        "selection_unit_ids_sha256": hashlib.sha256(
            "".join(f"{chain}\n" for chain in sorted(representatives)).encode()
        ).hexdigest(),
        "split_group_counts": {name: sum(name in split_names for split_names in owners.values()) for name in splits},
    }


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
    by_sequence: defaultdict[str, list[str]] = defaultdict(list)
    for candidate in candidates:
        by_cluster[cluster_map[candidate.chain_id]].append(candidate.chain_id)
        by_pdb[candidate.pdb_id].append(candidate.chain_id)
        by_sequence[candidate.sequence].append(candidate.chain_id)
    for members in list(by_cluster.values()) + list(by_sequence.values()):
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
    pool: list[Candidate], reference: list[Candidate], *, size: int,
    stratify_fields: tuple[str, ...], seed: int = 0, salt: str = "hidden_allocation",
) -> dict[tuple[str, ...], int]:
    return _allocate_hierarchical(
        pool, reference, size=size, stratify_fields=stratify_fields, seed=seed, salt=salt,
    )


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


def _validate_source_lock(
    lock_path: Path, *, train_path: Path, val_path: Path, all_path: Path,
    structure_metadata_path: Path, cluster_map: dict[str, str], cluster_tsv_path: Path | None = None,
) -> None:
    lock = json.loads(lock_path.read_text())
    outputs = lock.get("outputs", {})
    for name, path in (("train", train_path), ("val", val_path), ("all", all_path)):
        expected = outputs.get(f"{name}_manifest_sha256")
        if expected != _sha256(path):
            raise ValueError(f"Public {name} manifest SHA256 disagrees with source lock.")
        count_key = "unique_count" if name == "all" else f"{name}_count"
        if outputs.get(count_key) != len(_read_manifest(path)):
            raise ValueError(f"Public {name} manifest count disagrees with source lock.")
    if lock.get("structure_metadata", {}).get("sha256") != _sha256(structure_metadata_path):
        raise ValueError("Structure metadata SHA256 disagrees with source lock.")
    if lock.get("clustering", {}).get("cluster_assignments_sha256") != _cluster_assignments_sha256(cluster_map):
        raise ValueError("Cluster assignments SHA256 disagrees with source lock.")
    expected_cluster_tsv_sha = lock.get("clustering", {}).get("cluster_tsv_sha256")
    if expected_cluster_tsv_sha is not None:
        if cluster_tsv_path is None or not cluster_tsv_path.is_file() or _sha256(cluster_tsv_path) != expected_cluster_tsv_sha:
            raise ValueError("Cluster TSV SHA256 disagrees with source lock.")


def _build_hidden_manifest(args: argparse.Namespace) -> dict[str, Any]:
    lock_args = _load_lock_args(args.lock_file)
    if lock_args.sampling_policy not in CATH_SAMPLING_POLICIES:
        raise ValueError("Hidden generation requires a versioned CATH public source lock; preserve frozen legacy splits.")
    if getattr(args, "sampling_policy", lock_args.sampling_policy) != lock_args.sampling_policy:
        raise ValueError("Requested sampling policy differs from the public source lock.")
    _validate_generation_args(argparse.Namespace(**vars(lock_args)))
    for field in ("hidden_val_size", "seed"):
        override = getattr(args, field)
        if override is not None and override != getattr(lock_args, field):
            raise ValueError(f"{field} override differs from source lock; create a new versioned source contract.")
    hidden_val_size = int(args.hidden_val_size if args.hidden_val_size is not None else lock_args.hidden_val_size)
    if hidden_val_size <= 0:
        raise ValueError("--hidden-val-size must be positive.")
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

    _validate_metadata_identity(structure_metadata_path, cache_sha, argparse.Namespace(**vars(lock_args)))
    structure_metadata_accepted, structure_metadata = _load_structure_metadata(structure_metadata_path, require_schema_version=2)
    candidates, reject_reasons, structural_metadata_source = _build_candidates(
        data,
        min_len=lock_args.min_len,
        max_len=lock_args.max_len,
        max_resolution=lock_args.max_resolution,
        structure_metadata=structure_metadata,
        structure_metadata_accepted=structure_metadata_accepted,
        sampling_policy=lock_args.sampling_policy,
        require_monomer=lock_args.require_monomer,
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

    split_group_map = _load_split_group_map(
        args.lock_file, lock_args, set(candidate_ids), override_path=getattr(args, "split_group_tsv", None),
        receipt_override_path=getattr(args, "split_group_receipt", None),
        chain_data_cache_path=args.chain_data_cache, structure_metadata_path=structure_metadata_path,
        cluster_map=cluster_map,
    )
    _validate_source_lock(
        args.lock_file, train_path=args.public_train_manifest, val_path=args.public_val_manifest,
        all_path=args.public_all_manifest, structure_metadata_path=structure_metadata_path, cluster_map=cluster_map,
        cluster_tsv_path=Path(cluster_tsv) if cluster_tsv else None,
    )
    clustered_candidates = _with_cluster_ids(candidates, cluster_map)
    candidate_by_id = {candidate.chain_id: candidate for candidate in clustered_candidates}
    components = _componentize(clustered_candidates, cluster_map, seed=seed)
    eligibility = _load_representative_eligibility(
        args, lock_args, clustered_candidates, cluster_map=cluster_map, split_group_map=split_group_map,
        structure_metadata_path=structure_metadata_path,
    )
    fixed_hidden_groups = _validate_public_sampling_contract(
        args.lock_file, lock_args, components, train_ids, val_ids, split_group_map=split_group_map,
        eligibility=eligibility,
    )

    public_components = [
        component
        for component in components
        if any(member.chain_id in public_ids for member in component.members)
    ]
    if fixed_hidden_groups is not None:
        assert split_group_map is not None
        hidden_pool = [candidate for group in fixed_hidden_groups for candidate in group]
        fixed_group_ids = {split_group_map[candidate.chain_id] for candidate in hidden_pool}
        excluded_group_ids = {split_group_map[chain] for chain in private_processability_exclusions if chain in split_group_map}
        if fixed_group_ids & excluded_group_ids:
            raise RuntimeError("Private exclusions touch fixed hidden full groups; complete group coverage cannot be preserved.")
    else:
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
    if fixed_hidden_groups is not None:
        hidden = _sample_hidden_groups(fixed_hidden_groups, seed=seed, hidden_split_salt_digest=hidden_split_salt_digest)
        allocation = dict(Counter(_stratum(candidate, stratify_fields) for candidate in hidden))
    else:
        allocation = _allocate_against_reference(
            hidden_pool, public_reference, size=hidden_val_size, stratify_fields=stratify_fields,
            seed=seed, salt=f"hidden_val_fixed_public:{hidden_split_salt_digest}",
        )
        hidden = _sample_by_allocation(
            hidden_pool, allocation, seed=seed, salt=f"hidden_val_fixed_public:{hidden_split_salt_digest}",
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

    public_sequences = {candidate_by_id[chain_id].sequence for chain_id in public_ids}
    if public_sequences & {candidate.sequence for candidate in hidden}:
        raise RuntimeError("Hidden split leaks exact public sequences.")
    if split_group_map is not None:
        _validate_group_selection(
            components, split_group_map, {"train": train_ids, "val": val_ids, "hidden_val": hidden_ids},
            private_processability_exclusions, eligibility=eligibility,
        )

    selected_candidates = {
        chain_id: candidate_by_id[chain_id]
        for chain_id in train_ids + val_ids + hidden_ids
    }
    reference_candidates = [component.representative for component in components] if fixed_hidden_groups is not None else public_reference
    reference_population = "eligible_disjoint_units" if fixed_hidden_groups is not None else "fixed_public_chains"
    if eligibility is not None:
        reference_candidates = [candidate_by_id[chain] for chain in eligibility["selected_chain_by_original_unit"].values() if chain is not None]
        reference_population = "source_validated_disjoint_units"
    quality_report = _build_quality_report(
        selected_candidates=selected_candidates,
        split_ids={"train": train_ids, "val": val_ids, "hidden_val": hidden_ids},
        stratify_fields=stratify_fields,
        reference_candidates=reference_candidates,
    )
    quality_report["reference_population"] = reference_population
    _assert_quality_report(quality_report)

    args.hidden_out_dir.mkdir(parents=True, exist_ok=True)
    hidden_manifest = args.hidden_out_dir / "hidden_val.txt"
    split_quality_report = args.hidden_out_dir / "split_quality_report.json"
    output_paths = (hidden_manifest, split_quality_report, args.private_lock_file)
    if len({path.resolve() for path in output_paths}) != len(output_paths):
        raise ValueError("Sampling output paths must be distinct.")
    for path in output_paths:
        if path.exists():
            raise ValueError(f"Refusing to overwrite existing sampling output; use a new version directory: {path}")
    _atomic_write(hidden_manifest, "\n".join(hidden_ids) + "\n")
    _atomic_write(split_quality_report, json.dumps(quality_report, indent=2, sort_keys=True) + "\n")

    private_lock: dict[str, Any] = {
        "sampling_policy": lock_args.sampling_policy,
        "source_lock_sha256": _sha256(args.lock_file),
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
            "private_processability_exclusions": {
                "path": str(args.private_processability_exclusion_list) if args.private_processability_exclusion_list else None,
                "sha256": _sha256(args.private_processability_exclusion_list)
                if args.private_processability_exclusion_list else None,
                "canonical_ids_sha256": hashlib.sha256(
                    "".join(f"{chain_id}\n" for chain_id in sorted(private_processability_exclusions)).encode()
                ).hexdigest(),
            },
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
            "exact_sequence_disjoint": True,
        },
        "stratification": {
            "fields": list(stratify_fields),
            "structural_metadata_source": structural_metadata_source,
            "reference": reference_population if fixed_hidden_groups is not None else "public_train_plus_public_val",
            "allocation": {"|".join(key): count for key, count in sorted(allocation.items())},
            "quality_report": quality_report,
        },
    }
    if split_group_map is not None:
        private_lock["split_groups"] = json.loads(args.lock_file.read_text())["split_groups"]
        private_lock["grouping"]["full_group_disjoint"] = True
        private_lock["grouping"]["all_retained_groups_covered" if eligibility is not None else "all_groups_covered"] = True
    if eligibility is not None:
        private_lock["representative_eligibility"] = eligibility["binding"]
    args.private_lock_file.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write(args.private_lock_file, json.dumps(private_lock, indent=2, sort_keys=True) + "\n")

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
