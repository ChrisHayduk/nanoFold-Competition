from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from build_hidden_manifest import (  # noqa: E402
    _allocate_against_reference,
    _assert_public_manifests,
    _componentize,
    _load_lock_args,
    _load_split_group_map,
    _read_manifest,
    _read_optional_chain_list,
    _sample_by_allocation,
    _validate_group_selection,
    _validate_public_sampling_contract,
    _validate_source_lock,
    _with_cluster_ids,
)
from build_manifests import (  # noqa: E402
    CATH_SAMPLING_POLICIES,
    DEFAULT_STRATIFY_FIELDS,
    GROUP_SAMPLING_POLICY,
    LEGACY_STRATIFY_FIELDS,
    SAMPLING_POLICY,
    _assert_quality_report,
    _atomic_write,
    _build_candidates,
    _build_quality_report,
    _cluster_assignments_sha256,
    _cluster_sequences_mmseqs,
    _load_cluster_tsv,
    _load_structure_metadata,
    _sample_hidden_groups,
    _select_stratified,
    _sha256,
    _validate_metadata_identity,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Verify that maintainer-only hidden validation is disjoint from the fixed public "
            "manifests and respects structural stratification gates. For CATH sampling policies, replay public and "
            "hidden selection against the locked candidate universe, sizes, seed, and private salt digest."
            " cath-v3 additionally requires complete full-group coverage and disjoint group ownership."
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
    parser.add_argument("--split-group-tsv", type=Path, default=None,
                        help="Relocated cath-v3 group assignments; bytes must match the source lock.")
    parser.add_argument("--split-group-receipt", type=Path, default=None,
                        help="Relocated cath-v3 group-construction receipt; bytes must match the source lock.")
    parser.add_argument("--mmseqs-bin", type=str, default=None)
    parser.add_argument(
        "--private-processability-exclusion-list", type=Path, default=None,
        help="Relocated private exclusion input; its bytes and canonical IDs must match the private lock.",
    )
    parser.add_argument("--json-out", type=Path, default=None,
                        help="Atomically write the report; output must not alias any audited input.")
    return parser.parse_args()


def _load_private_lock(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError("A private hidden manifest source lock is required for verification.")
    raw = json.loads(path.read_text())
    if not isinstance(raw, dict):
        raise ValueError(f"Private hidden manifest lock must be a JSON object: {path}")
    return raw


def _validate_private_lock(
    private_lock: dict[str, Any], *, args: argparse.Namespace, sampling_policy: str,
    cache_sha: str, structure_metadata_path: Path, cluster_map: dict[str, str], hidden_count: int,
) -> str:
    """Bind the private selection to the same inputs and public release being audited."""
    required = sampling_policy in CATH_SAMPLING_POLICIES

    def check(container: dict[str, Any], key: str, actual: object, *, mandatory: bool = True) -> None:
        if (mandatory or key in container) and container.get(key) != actual:
            raise ValueError(f"Private hidden manifest lock {key} is missing or does not match.")

    check(private_lock, "hidden_val_manifest_sha256", _sha256(args.hidden_manifest))
    check(private_lock, "hidden_val_count", hidden_count)
    check(private_lock, "chain_data_cache_sha256", cache_sha)
    check(private_lock, "sampling_policy", sampling_policy, mandatory=required)
    check(private_lock, "source_lock_sha256", _sha256(args.lock_file), mandatory=required)
    if sampling_policy == GROUP_SAMPLING_POLICY:
        public_groups = json.loads(args.lock_file.read_text()).get("split_groups")
        if not isinstance(public_groups, dict) or private_lock.get("split_groups") != public_groups:
            raise ValueError("Private hidden manifest lock split-group provenance does not match the public source lock.")
    contract = private_lock.get("public_contract", {})
    if not isinstance(contract, dict):
        raise ValueError("Private hidden manifest lock public_contract must be an object.")
    for name, path in (("train", args.public_train_manifest), ("val", args.public_val_manifest),
                       ("all", args.public_all_manifest)):
        check(contract, f"{name}_manifest_sha256", _sha256(path), mandatory=required)
        check(contract, f"{name}_count", len(_read_manifest(path)), mandatory=required)
    for section, key, actual in (
        ("structure_metadata", "sha256", _sha256(structure_metadata_path)),
        ("clustering", "cluster_assignments_sha256", _cluster_assignments_sha256(cluster_map)),
    ):
        record = private_lock.get(section, {})
        if not isinstance(record, dict):
            raise ValueError(f"Private hidden manifest lock {section} must be an object.")
        check(record, key, actual, mandatory=required)
    stratification = private_lock.get("stratification", {})
    if not isinstance(stratification, dict):
        raise ValueError("Private hidden manifest lock stratification must be an object.")
    if required and stratification.get("fields") != list(DEFAULT_STRATIFY_FIELDS):
        raise ValueError("Private hidden manifest lock stratification fields disagree with CATH sampling policy.")
    reference = stratification.get("reference", "selected_chains" if not required else None)
    permitted = {"eligible_disjoint_units", "public_train_plus_public_val"} if required else {
        "selected_chains", "public_train_plus_public_val",
    }
    if reference not in permitted:
        raise ValueError("Private hidden manifest lock has an unsupported stratification reference.")
    if sampling_policy == GROUP_SAMPLING_POLICY and reference != "eligible_disjoint_units":
        raise ValueError("cath-v3 quality reference must be the original eligible selection units.")
    # Frozen hidden-only locks describe their allocation reference, but their
    # quality gate was evaluated against pooled selected chains.
    return str(reference) if required else "selected_chains"


def _fail_if_nonempty(label: str, values: set[str]) -> None:
    if values:
        raise RuntimeError(f"{label} overlap is non-zero: {len(values)}")


def _validate_private_exclusions(
    private_lock: dict[str, Any], *, override_path: Path | None, required: bool,
) -> set[str]:
    filtering = private_lock.get("filtering", {})
    if not isinstance(filtering, dict):
        raise ValueError("Private hidden manifest lock filtering must be an object.")
    record = filtering.get("private_processability_exclusions")
    if record is None and not required and override_path is None:
        return set()
    if not isinstance(record, dict):
        raise ValueError("Private hidden manifest lock private exclusions are missing or invalid.")
    if "path" not in record:
        raise ValueError("Private hidden manifest lock private exclusion path is missing.")
    recorded_path = record.get("path")
    if recorded_path is None:
        if override_path is not None or record.get("sha256", "missing") is not None:
            raise ValueError("Private processability exclusions are unconfigured but a path or digest was supplied.")
        path = None
    elif not isinstance(recorded_path, str) or not recorded_path:
        raise ValueError("Private hidden manifest lock private exclusion path is missing or invalid.")
    else:
        path = override_path if override_path is not None else Path(recorded_path)
    if path is not None and not path.is_file() and (required or record.get("sha256") is not None):
        raise ValueError("Private processability exclusions must be an existing file.")
    actual_sha = _sha256(path) if path is not None and path.is_file() else None
    if "sha256" not in record or record["sha256"] != actual_sha:
        raise ValueError("Private processability exclusion SHA256 is missing or does not match.")
    exclusions = set(_read_optional_chain_list(path if actual_sha is not None else None))
    canonical_sha = hashlib.sha256("".join(f"{chain}\n" for chain in sorted(exclusions)).encode()).hexdigest()
    if record.get("canonical_ids_sha256") != canonical_sha:
        raise ValueError("Private processability exclusion canonical IDs SHA256 does not match.")
    if filtering.get("private_processability_exclusion_count") != len(exclusions):
        raise ValueError("Private processability exclusion count does not match.")
    return exclusions


def verify(args: argparse.Namespace) -> dict[str, Any]:
    lock_args = _load_lock_args(args.lock_file)
    if lock_args.sampling_policy not in {*CATH_SAMPLING_POLICIES, "legacy-v1"}:
        raise ValueError("Unsupported sampling policy in source lock.")
    structure_metadata_path = args.structure_metadata or Path(lock_args.structure_metadata)
    cluster_tsv = str(args.cluster_tsv if args.cluster_tsv is not None else lock_args.cluster_tsv)
    mmseqs_bin = str(args.mmseqs_bin if args.mmseqs_bin is not None else lock_args.mmseqs_bin)

    report_path = getattr(args, "json_out", None)
    if report_path is not None:
        inputs = [args.chain_data_cache, args.lock_file, args.private_lock_file, args.public_train_manifest,
                  args.public_val_manifest, args.public_all_manifest, args.hidden_manifest, structure_metadata_path]
        if cluster_tsv:
            inputs.append(Path(cluster_tsv))
        if lock_args.sampling_policy == GROUP_SAMPLING_POLICY:
            inputs.extend([
                getattr(args, "split_group_tsv", None) or Path(lock_args.split_group_tsv),
                getattr(args, "split_group_receipt", None) or Path(lock_args.split_group_receipt),
            ])
        private_record = _load_private_lock(args.private_lock_file)
        filtering = private_record.get("filtering", {})
        exclusion_record = filtering.get("private_processability_exclusions", {}) if isinstance(filtering, dict) else {}
        if not isinstance(exclusion_record, dict):
            exclusion_record = {}
        exclusion_path = getattr(args, "private_processability_exclusion_list", None) or exclusion_record.get("path")
        if exclusion_path:
            inputs.append(Path(exclusion_path))
        if any(report_path.resolve() == path.resolve() or (
            report_path.exists() and path.exists() and report_path.samefile(path)
        ) for path in inputs):
            raise ValueError("Verification report output must not alias an audited input.")

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
    if not expected_sha or cache_sha != expected_sha:
        raise ValueError(
            "chain_data_cache SHA256 mismatch.\n"
            f"expected: {expected_sha}\n"
            f"actual:   {cache_sha}\n"
            f"path:     {args.chain_data_cache.resolve()}"
        )

    data = json.loads(args.chain_data_cache.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"chain_data_cache.json must be a JSON object: {args.chain_data_cache}")

    structure_metadata_accepted, structure_metadata = _load_structure_metadata(
        structure_metadata_path, require_schema_version=2 if lock_args.sampling_policy in CATH_SAMPLING_POLICIES else None,
    )
    if lock_args.sampling_policy in CATH_SAMPLING_POLICIES:
        _validate_metadata_identity(structure_metadata_path, cache_sha256=cache_sha, args=argparse.Namespace(**vars(lock_args)))
    candidates, _reject_reasons, _structural_metadata_source = _build_candidates(
        data,
        min_len=lock_args.min_len,
        max_len=lock_args.max_len,
        max_resolution=lock_args.max_resolution,
        structure_metadata=structure_metadata,
        structure_metadata_accepted=structure_metadata_accepted,
        sampling_policy=lock_args.sampling_policy,
        require_monomer=lock_args.require_monomer,
    )
    candidate_ids = {candidate.chain_id for candidate in candidates}
    required_ids = set(all_ids) | set(hidden_ids)
    missing = sorted(required_ids - candidate_ids)
    if missing:
        raise ValueError(f"{len(missing)} manifest chains are not eligible candidates.")

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
            alignment_mode=3 if lock_args.sampling_policy in CATH_SAMPLING_POLICIES else None,
        )
        if cluster_map_or_none is None:
            raise RuntimeError(
                "Hidden manifest verification requires MMseqs2 clustering or a locked --cluster-tsv. "
                f"Install `{mmseqs_bin}` or provide a TSV generated with --min-seq-id "
                f"{lock_args.min_seq_id} and coverage {lock_args.coverage}."
            )
        cluster_map = cluster_map_or_none

    split_group_map = _load_split_group_map(
        args.lock_file, lock_args, candidate_ids, override_path=getattr(args, "split_group_tsv", None),
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

    splits = {"train": train_ids, "val": val_ids, "hidden_val": hidden_ids}
    for left, right in (("train", "val"), ("train", "hidden_val"), ("val", "hidden_val")):
        for label, attribute in (("MMseqs2 cluster", "cluster_id"), ("PDB entry", "pdb_id"),
                                 ("exact sequence", "sequence")):
            left_values = {getattr(candidate_by_id[chain_id], attribute) for chain_id in splits[left]}
            right_values = {getattr(candidate_by_id[chain_id], attribute) for chain_id in splits[right]}
            _fail_if_nonempty(f"{left}/{right} {label}", left_values & right_values)

    private_lock = _load_private_lock(args.private_lock_file)
    reference_population = _validate_private_lock(
        private_lock, args=args, sampling_policy=lock_args.sampling_policy, cache_sha=cache_sha,
        structure_metadata_path=structure_metadata_path, cluster_map=cluster_map, hidden_count=len(hidden_ids),
    )
    private_exclusions = _validate_private_exclusions(
        private_lock, override_path=getattr(args, "private_processability_exclusion_list", None),
        required=lock_args.sampling_policy == GROUP_SAMPLING_POLICY or (
            lock_args.sampling_policy == SAMPLING_POLICY and reference_population == "public_train_plus_public_val"
        ),
    )
    _fail_if_nonempty("Public/private processability exclusion", public_ids & private_exclusions)
    components = _componentize(clustered_candidates, cluster_map, seed=lock_args.seed)
    split_by_id = {chain_id: name for name, ids in splits.items() for chain_id in ids}
    shared_components = 0
    repeated_units = 0
    excluded_hidden_components = 0
    for component in components:
        member_ids = {member.chain_id for member in component.members}
        selected_ids = member_ids & split_by_id.keys()
        shared_components += len({split_by_id[chain_id] for chain_id in selected_ids}) > 1
        repeated_units += len(selected_ids) > 1
        excluded_hidden_components += bool(member_ids & hidden_set) and bool(member_ids & private_exclusions)
    if shared_components:
        raise RuntimeError(f"Selected splits share {shared_components} transitive cluster/PDB/exact-sequence components.")
    if lock_args.sampling_policy in CATH_SAMPLING_POLICIES and repeated_units:
        raise RuntimeError(f"Selected manifests contain multiple representatives of {repeated_units} connected components.")
    if excluded_hidden_components:
        raise RuntimeError(f"Hidden split touches {excluded_hidden_components} privately excluded connected components.")
    group_coverage = None
    if split_group_map is not None:
        group_coverage = _validate_group_selection(components, split_group_map, splits, private_exclusions)
    fixed_hidden_groups = _validate_public_sampling_contract(
        args.lock_file, lock_args, components, train_ids, val_ids, split_group_map=split_group_map,
    )
    if lock_args.sampling_policy in CATH_SAMPLING_POLICIES:
        digest = private_lock.get("hidden_split_salt_sha256")
        if not isinstance(digest, str) or len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            raise ValueError("Private hidden manifest lock hidden salt digest is missing or invalid.")
        hidden_pool = [component.representative for component in components
                       if not any(member.chain_id in public_ids or member.chain_id in private_exclusions
                                  for member in component.members)]
        if fixed_hidden_groups is not None:
            replay = _sample_hidden_groups(fixed_hidden_groups, seed=lock_args.seed, hidden_split_salt_digest=digest)
        elif reference_population == "eligible_disjoint_units":
            replay = _select_stratified(hidden_pool, size=lock_args.hidden_val_size, seed=lock_args.seed,
                                       salt=f"hidden_val:{digest}", stratify_fields=DEFAULT_STRATIFY_FIELDS)
        else:
            salt = f"hidden_val_fixed_public:{digest}"
            allocation = _allocate_against_reference(
                hidden_pool, [candidate_by_id[chain] for chain in train_ids + val_ids],
                size=lock_args.hidden_val_size, seed=lock_args.seed, salt=salt, stratify_fields=DEFAULT_STRATIFY_FIELDS,
            )
            replay = _sample_by_allocation(hidden_pool, allocation, seed=lock_args.seed, salt=salt,
                                          stratify_fields=DEFAULT_STRATIFY_FIELDS)
        if hidden_ids != [candidate.chain_id for candidate in replay]:
            raise ValueError("Hidden manifest does not reproduce the locked salt, seed, exclusions, and sampling policy.")
    reference_candidates = None
    if reference_population == "eligible_disjoint_units":
        reference_candidates = [component.representative for component in components]
    elif reference_population == "public_train_plus_public_val":
        reference_candidates = [candidate_by_id[chain_id] for chain_id in train_ids + val_ids]
    stratify_fields = DEFAULT_STRATIFY_FIELDS if lock_args.sampling_policy in CATH_SAMPLING_POLICIES else LEGACY_STRATIFY_FIELDS

    selected_candidates = {
        chain_id: candidate_by_id[chain_id]
        for chain_id in train_ids + val_ids + hidden_ids
    }
    quality_report = _build_quality_report(
        selected_candidates=selected_candidates,
        split_ids=splits,
        stratify_fields=stratify_fields,
        reference_candidates=reference_candidates,
    )
    _assert_quality_report(quality_report)

    js = quality_report["jensen_shannon_divergence"]
    max_js = float(quality_report["observed"]["max_split_js_divergence"])
    return {
        "status": "ok",
        "sampling_policy": lock_args.sampling_policy,
        **({"group_coverage": group_coverage} if group_coverage is not None else {}),
        "counts": {
            "train": len(train_ids),
            "val": len(val_ids),
            "hidden_val": len(hidden_ids),
        },
        "overlap": {
            "chains": 0,
            "mmseqs2_clusters": 0,
            "pdb_entries": 0,
            "exact_sequences": 0,
            "connected_components": 0,
            "private_exclusion_components": 0,
        },
        "clustering": {
            "method": cluster_method,
            "cluster_count": len(set(cluster_map.values())),
        },
        "stratification": {
            "fields": list(stratify_fields),
            "reference_population": reference_population,
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
        _atomic_write(args.json_out, json.dumps(result, indent=2, sort_keys=True) + "\n")

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
