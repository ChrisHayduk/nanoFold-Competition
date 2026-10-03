"""Selection invariants for full parents under a directed heldout barrier.

A fresh 40-percent cluster is an ownership heuristic. The graph receipt's
query-coverage barrier supplies the training/heldout safeguard; this module does
not call those components an all-pairs homology certificate.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from nanofold.representative_eligibility import require


def metadata_rows(value: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    rows = value.get("chains")
    if not isinstance(rows, list):
        raise ValueError("Metadata chains must be a list")
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        require(isinstance(row, dict) and isinstance(row.get("chain_id"), str), "Invalid metadata row")
        require(row["chain_id"] not in result, "Duplicate chain in structural metadata")
        result[row["chain_id"]] = row
    return result


def join_qualified_parents(
    *, expected_copies: Mapping[str, Mapping[str, Any]],
    ranks: Mapping[str, tuple[Any, ...]], plans: Sequence[Mapping[str, Any]],
    qualified: Sequence[Mapping[str, Any]], metadata: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Prove every barrier-admitted copy was attempted in a complete sequence plan.

    The population receipt separately replays attempted structures and raw MSAs.
    The owning full-frame validator supplies expected_copies and ranks, rather
    than trusting the qualified export to define its own denominator.
    """
    expected_ids = set(expected_copies)
    seen: set[str] = set()
    sequence_groups: dict[str, str] = {}
    for plan in plans:
        sequence, group = plan["sequence"], plan["group_id"]
        require(sequence not in sequence_groups, "Full-parent plan repeats an exact sequence")
        sequence_groups[sequence] = group
        copies = plan["ranked_copies"]
        require(bool(copies), "Full-parent sequence has no deposited copies")
        for copy in copies:
            chain = copy["chain_id"]
            require(chain in expected_ids and chain not in seen, "Full-parent plan escapes or repeats admitted frame")
            require(copy == expected_copies[chain], "Full-parent copy identity or scientific metadata changed")
            require(copy["sequence"] == sequence and copy["group_id"] == group,
                    "Full-parent sequence plan disagrees with its copies")
            seen.add(chain)
        require(copies == sorted(copies, key=lambda row: ranks[row["chain_id"]]),
                "Full-parent plan changed deposited-copy quality rank")
    require(seen == expected_ids, "Full-parent sequence plan omitted admitted source chains")
    records = []
    selected_sequences: set[str] = set()
    selected_chains: set[str] = set()
    for row in qualified:
        chain, sequence = row["chain_id"], row["sequence"]
        require(chain in expected_ids and chain not in selected_chains, "Qualified parent escapes or repeats admitted frame")
        require(sequence not in selected_sequences, "Qualified parents repeat an exact sequence")
        require(all(row.get(key) == value for key, value in expected_copies[chain].items()),
                "Qualified parent differs from authoritative frame")
        require(sequence_groups.get(sequence) == row["group_id"], "Qualified parent has a different ownership component")
        selected_chains.add(chain)
        selected_sequences.add(sequence)
        annotations = {key: value for key, value in metadata[chain].items() if key.startswith("cath_")}
        resolution = row["resolution"]
        records.append(dict(row) | annotations | {"resolution": None if resolution == 999.0 else resolution})
    require(bool(records), "No source-qualified parents remain")
    return sorted(records, key=lambda row: row["chain_id"])


def verify_parent_selection(
    train: Sequence[str], *, qualified: Sequence[Mapping[str, Any]],
    frozen_splits: Mapping[str, Sequence[str]], evaluation_cohorts: Mapping[str, Sequence[str]],
    all_sequences: Mapping[str, str],
    size: int, required_groups: set[str], coverage_policy: str,
) -> dict[str, Any]:
    """Check selection without imposing a false expanded heldout-group claim."""
    require(set(frozen_splits) == {"val", "hidden_val"}, "Both frozen heldouts are required")
    require(set(evaluation_cohorts) == {"historical_public", "active_public", "hidden"},
            "Historical, active public, and hidden evaluation cohorts are required")
    require(list(evaluation_cohorts["active_public"]) == list(frozen_splits["val"])
            and list(evaluation_cohorts["hidden"]) == list(frozen_splits["hidden_val"]),
            "Active evaluation cohorts differ from staged validation splits")
    historical = evaluation_cohorts["historical_public"]
    require(bool(historical) and len(set(historical)) == len(historical)
            and len(historical) == len(frozen_splits["val"]), "Historical public cohort is incomplete or repeated")
    require(not set(historical) & set(frozen_splits["hidden_val"]), "Historical public and hidden chains overlap")
    require(len(train) == size and len(set(train)) == size, "Incorrect parent count or duplicate training parents")
    by_chain = {row["chain_id"]: row for row in qualified}
    require(len(by_chain) == len(qualified) and set(train) <= set(by_chain), "Training includes an unqualified parent")
    require(len({by_chain[chain]["sequence"] for chain in train}) == size, "Training repeats an exact parent sequence")
    train_pdbs = {chain.split("_", 1)[0].lower() for chain in train}
    train_sequences = {by_chain[chain]["sequence"] for chain in train}
    heldouts: set[str] = set()
    for ids in frozen_splits.values():
        require(bool(ids) and len(set(ids)) == len(ids), "Frozen heldout is empty or repeats chains")
        require(not heldouts.intersection(ids), "Frozen validation and hidden chains overlap")
        heldouts.update(ids)
    heldouts.update(historical)
    require(not set(train) & heldouts, "Training and heldout chain overlap")
    require(not train_pdbs & {chain.split("_", 1)[0].lower() for chain in heldouts}, "Training and heldout PDB overlap")
    require(not train_sequences & {all_sequences[chain] for chain in heldouts}, "Training and heldout exact sequence overlap")
    eligible_groups = {row["group_id"] for row in qualified}
    selected_groups = {by_chain[chain]["group_id"] for chain in train}
    require(required_groups <= eligible_groups, "Required coverage group has no qualified parent")
    require(required_groups <= selected_groups, "Training omitted a required coverage group")
    require(coverage_policy in {"all_qualified_training_groups", "explicit_required_groups", "soft_diversity"}, "Unknown parent coverage policy")
    if coverage_policy == "soft_diversity":
        require(not required_groups, "Soft-diversity selection must not hide mandatory group anchors")
    if coverage_policy == "all_qualified_training_groups":
        require(required_groups == eligible_groups, "All-group policy did not require all qualified groups")
    return {
        "training_parent_count": size,
        "distinct_training_parent_sequences": size,
        "qualified_training_group_count": len(eligible_groups),
        "selected_training_group_count": len(selected_groups),
        "required_group_count": len(required_groups),
        "coverage_policy": coverage_policy,
        "frozen_heldout_counts": {key: len(ids) for key, ids in frozen_splits.items()},
        "evaluation_cohort_counts": {key: len(ids) for key, ids in evaluation_cohorts.items()},
        "protected_public_union_count": len(set(historical) | set(frozen_splits["val"])),
        "historical_public_protected_from_training": True,
        "training_heldout_chain_pdb_exact_sequence_overlap": 0,
        "leakage_scope": "full_parent_query_barrier_protected_public_revision_v1",
        "expanded_all_pairs_homology_disjointness_claimed": False,
    }
