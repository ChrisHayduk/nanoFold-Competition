from __future__ import annotations

import hashlib
import importlib
import json
from pathlib import Path

import pytest

from nanofold.full_parent_leakage import derive_training_frame, parse_clusters, sequence_id, validate_population_source


def row(chain: str, sequence: str) -> dict:
    return {"chain_id": chain, "sequence": sequence, "sequence_sha256": hashlib.sha256(sequence.encode()).hexdigest(),
            "length": len(sequence), "resolution": 2.0, "release_date": "2020-01-01"}


def test_embedded_domain_and_original_ownership_excluded_without_transitive_bridge_loss() -> None:
    val = "ACDEFGHIKLMNPQRSTVWY" * 3
    hidden = "YWVTSRQPNMLKIHGFEDCA" * 3
    keep = "GKLHIM" * 10
    records = [row("1aaa_A", val), row("1aaa_B", keep), row("2aaa_A", hidden), row("3aaa_A", keep),
               row("3aaa_C", "GKLHIM" * 11), row("4aaa_A", keep * 5 + val + keep * 5),
               row("5aaa_A", "STNFQ" * 10), row("7aaa_A", val)]
    original = {"1aaa_A": "original_val", "2aaa_A": "original_hidden", "3aaa_A": "original_train", "5aaa_A": "original_val"}
    frozen = {"val": ["1aaa_A"], "hidden_val": ["2aaa_A"]}
    clusters = {sequence_id(r["sequence"]): sequence_id(r["sequence"]) for r in records}
    hits = [("1aaa_A", sequence_id(val)), ("2aaa_A", sequence_id(hidden)), ("1aaa_A", sequence_id(records[5]["sequence"]))]
    frame = derive_training_frame(records, original, frozen, clusters, hits)
    assert frame["eligible_training_ids"] == ["3aaa_A", "3aaa_C"]
    assert frame["diagnostics"]["training_diversity_group_count"] == 1
    by_id = {r["chain_id"]: r for r in frame["records"]}
    assert by_id["4aaa_A"]["training_exclusion_reasons"] == ["heldout_query_homology"]
    assert by_id["5aaa_A"]["training_exclusion_reasons"] == ["inherited_heldout_group"]
    # The unrelated long-PDB bridge does not propagate heldout ownership through
    # the entire exact/PDB/homology component after direct exclusions.
    assert by_id["1aaa_B"]["training_exclusion_reasons"] == ["same_heldout_pdb"]
    assert by_id["3aaa_A"]["group_id"] == by_id["3aaa_C"]["group_id"] == "3aaa_A"
    assert by_id["7aaa_A"]["group_id"] is None
    assert not frame["diagnostics"]["training_diversity_groups_are_global_split_components"]


@pytest.mark.parametrize("failure", ["missing_self", "unknown_query", "unknown_parent", "duplicate"])
def test_parent_barrier_rejects_incomplete_or_unbound_alignments(failure: str) -> None:
    records = [row("1aaa_A", "A" * 40), row("2aaa_A", "C" * 40), row("3aaa_A", "D" * 40)]
    clusters = {sequence_id(r["sequence"]): sequence_id(r["sequence"]) for r in records}
    hits = [(r["chain_id"], sequence_id(r["sequence"])) for r in records[:2]]
    if failure == "missing_self":
        hits.pop()
    elif failure == "unknown_query":
        hits.append(("not_frozen", sequence_id(records[2]["sequence"])))
    elif failure == "unknown_parent":
        hits.append(("1aaa_A", "unbound"))
    else:
        hits.append(hits[0])
    with pytest.raises(ValueError):
        derive_training_frame(records, {r["chain_id"]: r["chain_id"] for r in records},
                              {"val": ["1aaa_A"], "hidden_val": ["2aaa_A"]}, clusters, hits)


def test_complete_population_replay_supports_owning_sequence_aliases_and_rejects_missing_rows(tmp_path: Path) -> None:
    cache = tmp_path / "cache.json"
    cache.write_text(json.dumps({"1aaa_A": {"sequence": "A" * 400, "resolution": 2.0, "release_date": "2020-01-01"},
                                "2aaa_A": {"seq": "C" * 60, "resolution": 2.0, "release_date": "2020-01-01"},
                                "3aaa_A": None}))
    expected = [row("1aaa_A", "A" * 400), row("2aaa_A", "C" * 60)]
    validate_population_source(expected, cache)
    with pytest.raises(ValueError, match="complete eligible cache"):
        validate_population_source(expected[:1], cache)
    expected[0]["sequence"] = "A" * 256
    with pytest.raises(ValueError, match="independently replayed"):
        validate_population_source(expected, cache)


def test_cluster_membership_cannot_omit_parent_or_point_to_another_cluster(tmp_path: Path) -> None:
    first, second = sequence_id("A" * 40), sequence_id("C" * 40)
    path = tmp_path / "clusters.tsv"
    path.write_text(f"{first}\t{first}\n")
    with pytest.raises(ValueError, match="complete exact-sequence"):
        parse_clusters(path, {"A" * 40, "C" * 40})
    path.write_text(f"{second}\t{first}\n{first}\t{second}\n")
    with pytest.raises(ValueError, match="different cluster"):
        parse_clusters(path, {"A" * 40, "C" * 40})


def test_changed_verifier_requires_preserved_corrective_provenance(tmp_path: Path) -> None:
    driver = importlib.import_module("scripts.build_full_parent_frame")
    original = {"code_bindings": {"scripts/build_full_parent_frame.py": {"sha256": "0" * 64}},
                "scientific_payload": {"alignment_count": 200}}
    pending = tmp_path / "receipt.incomplete.json"
    pending.write_text(json.dumps(original))
    final = tmp_path / "receipt.json"
    with pytest.raises(ValueError, match="recorded corrective verification"):
        driver._validate_verifier_provenance(final, original)
    with pytest.raises(ValueError, match="original unpublished"):
        driver._validate_verifier_provenance(final, original, allow_unpublished_verifier_drift=True)
    driver._validate_verifier_provenance(pending, original, allow_unpublished_verifier_drift=True)

    historical_verifier = tmp_path / "historical-verifier.py"
    historical_verifier.write_text("# Immutable historical corrective verifier\n")
    corrected = {**original, "verification": {"original_unpublished_receipt": driver.file_record(pending),
                                              "verifier": driver.file_record(historical_verifier)}}
    # A newer verifier may replay an older corrective certificate; it must not
    # erase the original receipt, historical verifier, or scientific payload.
    driver._validate_verifier_provenance(final, corrected)
    modified = {**corrected, "scientific_payload": {"alignment_count": 199}}
    with pytest.raises(ValueError, match="changed the original scientific"):
        driver._validate_verifier_provenance(final, modified)
    historical_verifier.write_text("# changed\n")
    with pytest.raises(ValueError):
        driver._validate_verifier_provenance(final, corrected)
