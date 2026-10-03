from __future__ import annotations

import hashlib
import importlib
from pathlib import Path

import pytest

from nanofold.full_parent_leakage import derive_training_frame, sequence_id
from nanofold.revised_parent_leakage import derive_revised_frame


def fixture() -> tuple:
    sequences = {"1aaa_A": "A" * 80, "2aaa_A": "D" * 50, "3aaa_A": "C" * 40,
                 "4aaa_A": "E" * 80, "5aaa_A": "F" * 80, "6aaa_A": "G" * 80, "7aaa_A": "I" * 80}
    rows = [{"chain_id": chain, "sequence": sequence, "length": len(sequence),
             "sequence_sha256": hashlib.sha256(sequence.encode()).hexdigest(),
             "resolution": 2.0, "release_date": "2020-01-01"} for chain, sequence in sequences.items()]
    groups = {chain: chain for chain in sequences}
    groups["4aaa_A"] = groups["1aaa_A"]
    clusters = {sequence_id(sequence): sequence_id(sequence) for sequence in sequences.values()}
    baseline = {"val": ["1aaa_A", "2aaa_A"], "hidden_val": ["3aaa_A"]}
    old_hits = [(chain, sequence_id(sequences[chain])) for chain in ["1aaa_A", "2aaa_A", "3aaa_A"]]
    old_hits += [("3aaa_A", sequence_id(sequences["1aaa_A"])), ("1aaa_A", sequence_id(sequences["5aaa_A"]))]
    new_hits = [("4aaa_A", sequence_id(sequences[chain])) for chain in ["4aaa_A", "6aaa_A"]]
    return rows, groups, baseline, ["2aaa_A", "4aaa_A"], clusters, old_hits, new_hits


def test_revision_keeps_retired_public_protection_and_adds_new_parent_exclusions() -> None:
    frame = derive_revised_frame(*fixture())
    assert frame["eligible_training_ids"] == ["7aaa_A"]
    assert frame["frozen_splits"] == {"val": ["2aaa_A", "4aaa_A"], "hidden_val": ["3aaa_A"]}
    assert frame["evaluation_cohorts"]["historical_public"] == ["1aaa_A", "2aaa_A"]
    assert frame["diagnostics"]["protected_public_targets"] == 3
    assert frame["diagnostics"]["preserved_unaffected_public_targets"] == 1
    excluded = {row["chain_id"]: row["training_exclusion_reasons"] for row in frame["records"]}
    assert "heldout_query_homology" in excluded["5aaa_A"]
    assert "heldout_query_homology" in excluded["6aaa_A"]


@pytest.mark.parametrize("direction", ["hidden_query", "active_query"])
def test_revised_evaluation_requires_both_query_coverage_directions(direction: str) -> None:
    rows, groups, baseline, active, clusters, old_hits, new_hits = fixture()
    sequence = {row["chain_id"]: sequence_id(row["sequence"]) for row in rows}
    if direction == "hidden_query":
        old_hits.append(("3aaa_A", sequence["4aaa_A"]))
    else:
        new_hits.append(("4aaa_A", sequence["3aaa_A"]))
    with pytest.raises(ValueError, match="symmetric query-coverage"):
        derive_revised_frame(rows, groups, baseline, active, clusters, old_hits, new_hits)


def test_repair_cannot_replace_unaffected_targets_or_omit_additive_self_hits() -> None:
    rows, groups, baseline, active, clusters, old_hits, new_hits = fixture()
    with pytest.raises(ValueError, match="unaffected historical"):
        derive_revised_frame(rows, groups, baseline, ["4aaa_A", "7aaa_A"], clusters, old_hits, new_hits)
    with pytest.raises(ValueError, match="missing heldout self"):
        derive_revised_frame(rows, groups, baseline, active, clusters, old_hits, new_hits[1:])


def test_declared_additive_binary_and_resource_limits_must_match_native_execution(tmp_path: Path) -> None:
    driver = importlib.import_module("scripts.build_revised_parent_frame")
    binary = tmp_path / "mmseqs"
    binary.write_text("immutable executable")
    plan = {"binary": driver.file_record(binary), "source_batch_size": 10000, "query_batch_size": 64,
            "reserve_disk_bytes": 200, "max_rss_bytes": 500, "maximum_sequence_length": 3245, "threads": 8}
    native = {**plan, "max_sequence_length": 3245,
              "flags": ["--threads", "8", "--max-seqs", "10000", "--max-accept", "10000", "--max-rejected", "10000"]}
    driver.verify_execution_binding(plan, native)
    with pytest.raises(ValueError, match="declared execution bounds"):
        driver.verify_execution_binding(plan, {**native, "max_rss_bytes": 501})
    with pytest.raises(ValueError, match="different declared native binary"):
        driver.verify_execution_binding(plan, {**native, "binary": {**plan["binary"], "sha256": "0" * 64}})


def test_capacity_guard_counts_distinct_sequences_not_deposited_copies(monkeypatch: pytest.MonkeyPatch) -> None:
    driver = importlib.import_module("scripts.build_revised_parent_frame")
    rows, groups, baseline, active, clusters, old_hits, new_hits = fixture()
    template = rows[-1]
    for index in range(10000):
        chain = f"copy{index}_A"
        rows.append({**template, "chain_id": chain})
        groups[chain] = "retained_copies"
    base = derive_training_frame(rows, groups, baseline, clusters, old_hits)
    assert base["diagnostics"]["retained_training_chains"] > 10000
    context = {"rows": rows, "groups": groups, "baseline": base, "active": active,
               "clusters": clusters, "base_receipt": {"barrier": {"hits": old_hits}}}
    monkeypatch.setattr(driver, "iter_batched_mmseqs_hits", lambda result, **kwargs: iter(result["hits"]))
    with pytest.raises(ValueError, match="only 1 unique training parents"):
        driver.reconstruct(context, {"hits": new_hits})


def test_sorted_selection_accepts_unsorted_original_manifests_without_rewriting_them() -> None:
    driver = importlib.import_module("scripts.build_revised_parent_frame")
    baseline = {"val": ["2aaa_A", "1aaa_A"], "hidden_val": ["9aaa_A", "8aaa_A"]}
    selection = {"historical_public_ids": ["1aaa_A", "2aaa_A"], "hidden_ids": ["8aaa_A", "9aaa_A"],
                 "active_public_ids": ["2aaa_A", "4aaa_A"], "added_public_ids": ["4aaa_A"],
                 "qualified_added_ids": ["4aaa_A"]}
    assert driver.validate_selection_cohorts(baseline, selection) == ["4aaa_A"]
    assert baseline["val"] == ["2aaa_A", "1aaa_A"]
    assert baseline["hidden_val"] == ["9aaa_A", "8aaa_A"]
    with pytest.raises(ValueError, match="preserve and qualify"):
        driver.validate_selection_cohorts(baseline, {**selection, "hidden_ids": ["8aaa_A", "8aaa_A"]})
