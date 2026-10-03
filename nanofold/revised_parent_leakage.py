"""Additive parent exclusions for a versioned public evaluation repair.

Historical public targets stay protected from training. Only the active public
cohort must pass the new symmetric boundary against the unchanged hidden cohort.
"""
from __future__ import annotations

import importlib
import json
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from nanofold.full_parent_leakage import derive_training_frame, sequence_id

POLICY = "full_parent_query_barrier_protected_public_revision_v1"


def derive_revised_frame(
    records: Sequence[Mapping[str, Any]], inherited_groups: Mapping[str, str],
    baseline_splits: Mapping[str, Sequence[str]], active_public: Sequence[str],
    clusters: Mapping[str, str], baseline_hits: Iterable[tuple[str, str]],
    added_hits: Iterable[tuple[str, str]],
) -> dict[str, Any]:
    """Replay both protected cohorts without dropping any baseline exclusion."""
    historical = set(baseline_splits["val"])
    hidden = set(baseline_splits["hidden_val"])
    active = set(active_public)
    by_id = {row["chain_id"]: row for row in records}
    if (len(active) != len(active_public) or len(active) != len(historical)
            or active & hidden or not active <= by_id.keys() or not active <= inherited_groups.keys()):
        raise ValueError("Active evaluation cohort has incomplete, duplicate, or overlapping membership")
    if any(not 40 <= by_id[chain]["length"] <= 256 for chain in active):
        raise ValueError("Active evaluation targets must remain whole chains of 40--256 residues")
    if len({inherited_groups[chain] for chain in active}) != len(active):
        raise ValueError("Active public targets must occupy distinct original groups")
    if {inherited_groups[chain] for chain in active} & {inherited_groups[chain] for chain in hidden}:
        raise ValueError("Active public and hidden targets share original ownership")
    if ({chain.split("_", 1)[0].lower() for chain in active} & {chain.split("_", 1)[0].lower() for chain in hidden}
            or {by_id[chain]["sequence"] for chain in active} & {by_id[chain]["sequence"] for chain in hidden}):
        raise ValueError("Active public and hidden targets share PDB or exact sequence")
    historical_by_sequence = {sequence_id(by_id[chain]["sequence"]): chain for chain in historical}
    hidden_sequences = {sequence_id(by_id[chain]["sequence"]) for chain in hidden}
    active_sequences = {sequence_id(by_id[chain]["sequence"]) for chain in active}
    previous_hits = list(baseline_hits)
    new_hits = list(added_hits)
    affected = set()
    for query, target in previous_hits:
        if query in hidden and target in historical_by_sequence:
            affected.add(historical_by_sequence[target])
        elif query in historical and target in hidden_sequences:
            affected.add(query)
    if not historical - affected <= active:
        raise ValueError("Evaluation repair removed an unaffected historical public target")
    added = active - historical
    if any(query not in added for query, _ in new_hits):
        raise ValueError("Additive barrier contains a query outside the new public targets")
    combined = previous_hits + new_hits
    if any((query in hidden and target in active_sequences) or (query in active and target in hidden_sequences)
           for query, target in combined):
        raise ValueError("Active public and hidden targets fail the symmetric query-coverage barrier")
    protected_splits = {"val": sorted(historical | active), "hidden_val": list(baseline_splits["hidden_val"])}
    frame = derive_training_frame(records, inherited_groups, protected_splits, clusters, combined)
    frame["policy"] = POLICY
    frame["frozen_splits"] = {"val": list(active_public), "hidden_val": list(baseline_splits["hidden_val"])}
    frame["evaluation_cohorts"] = {"historical_public": list(baseline_splits["val"]),
                                  "active_public": list(active_public), "hidden": list(baseline_splits["hidden_val"])}
    frame["diagnostics"].update(historical_public_targets=len(historical), active_public_targets=len(active),
                                added_public_targets=len(added), retired_public_targets=len(historical - active),
                                protected_public_targets=len(historical | active),
                                preserved_unaffected_public_targets=len(historical - affected),
                                active_public_hidden_symmetric_boundary_passed=True)
    return frame


def validate_parent_frame(receipt_path: Path) -> dict[str, Any]:
    """Dispatch to the original or additive owning evidence validator."""
    receipt = json.loads(receipt_path.read_text())
    if receipt.get("policy") == POLICY:
        return dict(importlib.import_module("scripts.build_revised_parent_frame").validate_revised_frame_receipt(receipt_path))
    return dict(importlib.import_module("nanofold.full_parent_leakage").validate_parent_frame(receipt_path))
