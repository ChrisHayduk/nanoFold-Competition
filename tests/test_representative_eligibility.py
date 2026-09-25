from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest
from representative_eligibility_helpers import source_backed_eligibility_fixture, unit_id, write_json

from nanofold.representative_eligibility import (
    historical_source_catalog,
    population_record,
    raw_query_record,
    validate_representative_eligibility_receipt,
)


@pytest.fixture
def eligibility(tmp_path: Path) -> dict[str, Any]:
    sequences = {"1aaa_A": "A" * 40, "2bbb_A": "A" * 40, "3ccc_A": "C" * 40, "4ddd_A": "D" * 40}
    cache, metadata, graph, groups_path = (tmp_path / name for name in ("cache.json", "metadata.json", "graph.json", "groups.tsv"))
    write_json(cache, {chain: {"sequence": sequence, "seq_length": len(sequence),
                              "resolution": 1.0 if chain == "1aaa_A" else 2.0}
                       for chain, sequence in sequences.items()})
    write_json(metadata, {
        "schema_version": 2, "generated_at": "original", "source": {},
        "chains": [{"chain_id": chain, "accepted": True, "domain_architecture_class": "alpha",
                    "domain_architecture_source": "cath", "metadata_source_count": 1,
                    "cath_annotation_status": "annotated", "cath_architecture_ids": ["1.10"],
                    "cath_topology_ids": ["1.10.10"], "cath_homologous_superfamily_ids": ["1.10.10.10"],
                    "cath_domains": [{"domain_id": chain.replace("_", "") + "01", "class_id": "1",
                                      "architecture_id": "1.10", "topology_id": "1.10.10",
                                      "homologous_superfamily_id": "1.10.10.10"}]} for chain in sequences],
    })
    write_json(graph, {"synthetic_graph": "independently verified by the consumer"})
    clusters = {chain: ("1aaa_A" if chain == "2bbb_A" else chain) for chain in sequences}
    groups = dict(clusters)
    groups_path.write_text("".join(f"{group}\t{chain}\n" for chain, group in groups.items()))
    units = {unit_id(members): members for members in (["1aaa_A", "2bbb_A"], ["3ccc_A"], ["4ddd_A"])}
    fixture = source_backed_eligibility_fixture(
        tmp_path / "sources", cache_path=cache, metadata_path=metadata, groups_path=groups_path,
        graph_receipt_path=graph, sequences=sequences, cluster_map=clusters, group_map=groups,
        original_units=units, observed={"1aaa_A": 20, "4ddd_A": 20}, excluded_groups=["4ddd_A"],
    )
    fixture["kwargs"] = {
        "approved_group_exclusions_path": fixture["catalog_path"], "chain_data_cache_path": cache,
        "structure_metadata_path": metadata, "cluster_map": clusters, "split_group_map": groups,
        "split_group_receipt_path": graph, "original_units": units, "seed": 0,
    }
    return fixture


def validate(fixture: dict[str, Any]) -> dict[str, Any]:
    return validate_representative_eligibility_receipt(fixture["receipt_path"], **fixture["kwargs"])


def test_real_source_replay_selects_first_passing_rank_and_exact_exhausted_groups(eligibility: dict[str, Any]) -> None:
    result = validate(eligibility)
    assert result["selected_chain_by_original_unit"][unit_id(["1aaa_A", "2bbb_A"])] == "2bbb_A"
    assert result["selected_chain_by_original_unit"][unit_id(["4ddd_A"])] is None
    assert result["approved_excluded_group_ids"] == ["4ddd_A"]
    assert result["binding"]["original_candidate_count"] == 4
    assert result["binding"]["original_group_count"] == 3
    assert result["binding"]["retained_group_count"] == 2
    assert str(eligibility["decision"].resolve()) in result["validated_source_paths"]
    receipt = json.loads(eligibility["receipt_path"].read_text())
    assert set(receipt["predicate"]["runtime"]) == {"python", "gemmi", "biopython", "numpy"}


def test_coherently_forged_statistics_cannot_adopt_a_failed_structure(eligibility: dict[str, Any]) -> None:
    path = eligibility["receipt_path"]
    receipt = json.loads(path.read_text())
    unit = next(row for row in receipt["units"] if row["ranked_members"] == ["1aaa_A", "2bbb_A"])
    forged = copy.deepcopy(unit["outcomes"][1])
    forged.update(chain_id="1aaa_A", rank=0, structure=unit["outcomes"][0]["structure"])
    unit.update(outcomes=[forged], selected_chain_id="1aaa_A")
    write_json(path, receipt)
    with pytest.raises(ValueError, match="does not replay"):
        validate(eligibility)


def test_coherently_forged_rejection_cannot_skip_a_passing_rank(eligibility: dict[str, Any]) -> None:
    path = eligibility["receipt_path"]
    receipt = json.loads(path.read_text())
    unit = next(row for row in receipt["units"] if row["ranked_members"] == ["3ccc_A"])
    outcome = unit["outcomes"][0]
    outcome.update(status="failed", structure_status="failed", raw=None, raw_status="not_evaluated_structure_failed")
    outcome["projection_statistics"]["projection_valid_ca_count"] = 0.0
    unit["selected_chain_id"] = None
    write_json(path, receipt)
    with pytest.raises(ValueError, match="source is missing"):
        validate(eligibility)


@pytest.mark.parametrize("mutation", ["omit_rank", "omit_unit", "pending", "bool_count", "runtime"])
def test_incomplete_or_changed_qualification_is_rejected(eligibility: dict[str, Any], mutation: str) -> None:
    path = eligibility["receipt_path"]
    receipt = json.loads(path.read_text())
    unit = next(row for row in receipt["units"] if row["ranked_members"] == ["1aaa_A", "2bbb_A"])
    if mutation == "omit_rank":
        unit["outcomes"] = unit["outcomes"][1:]
    elif mutation == "omit_unit":
        receipt["units"].pop()
    elif mutation == "pending":
        unit["outcomes"][0]["structure_status"] = "pending"
    elif mutation == "bool_count":
        unit["outcomes"][1]["raw"]["query_length"] = True
    elif mutation == "runtime":
        receipt["predicate"]["runtime"]["gemmi"] = "changed"
    write_json(path, receipt)
    with pytest.raises(ValueError):
        validate(eligibility)


def test_semantic_metadata_relocation_allowed_but_changed_annotation_rejected(eligibility: dict[str, Any], tmp_path: Path) -> None:
    original = eligibility["kwargs"]["structure_metadata_path"]
    data = json.loads(original.read_text())
    data.update(generated_at="later")
    relocated = tmp_path / "relocated.json"
    write_json(relocated, data)
    eligibility["kwargs"]["structure_metadata_path"] = relocated
    assert validate(eligibility)["retained_group_count"] == 2
    data["chains"] = [{"changed": "annotation"}]
    write_json(relocated, data)
    with pytest.raises(ValueError, match="original population"):
        validate(eligibility)


def test_mutated_source_bytes_and_missing_acquisition_are_not_rejections(eligibility: dict[str, Any]) -> None:
    structure = eligibility["historical_root"] / "pdb_data/mmcif_files/2bbb.cif"
    original = structure.read_bytes()
    structure.write_bytes(original + b"\n# changed\n")
    with pytest.raises(ValueError, match="source bytes"):
        validate(eligibility)
    structure.unlink()
    with pytest.raises(FileNotFoundError):
        validate(eligibility)


def test_historical_sources_must_share_one_archive_version(eligibility: dict[str, Any], tmp_path: Path) -> None:
    first = eligibility["retrieval"]
    second = tmp_path / "other-retrieval.json"
    record = json.loads(first.read_text())
    record["archive_pin"]["etag"] = '"another-version"'
    write_json(second, record)
    with pytest.raises(ValueError, match="mix archive versions"):
        historical_source_catalog([first, second])


def test_raw_query_audit_hashes_full_bytes_without_claiming_all_rows(tmp_path: Path) -> None:
    path = tmp_path / "query.a3m"
    path.write_text(">query\nAAaaAA--\n>second\ninvalid downstream row!\n")
    record = raw_query_record(path)
    assert record["scope"] == "first_query_identity_only"
    assert record["query_length"] == 4
    from nanofold.representative_eligibility import sha256_file
    assert record["sha256"] == sha256_file(path)


def test_generator_reconstructs_and_audits_ranked_population(
    eligibility: dict[str, Any], monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    import importlib

    module = importlib.import_module("scripts.build_representative_eligibility")
    # The graph proof has its own end-to-end suite. This integration exercises
    # reconstruction and physical source audits, never substitutes outcome rows.
    checked_graph = []
    monkeypatch.setattr(module, "validate_group_receipt", lambda *args, **kwargs: checked_graph.append(kwargs))
    kwargs = eligibility["kwargs"]
    clusters = tmp_path / "clusters.tsv"
    groups = tmp_path / "groups.tsv"
    clusters.write_text("".join(f"{group}\t{chain}\n" for chain, group in kwargs["cluster_map"].items()))
    output = tmp_path / "generated.json"
    args = module.parse_args([
        "--chain-data-cache", str(kwargs["chain_data_cache_path"]),
        "--structure-metadata", str(kwargs["structure_metadata_path"]), "--cluster-tsv", str(clusters),
        "--split-group-tsv", str(groups), "--split-group-receipt", str(kwargs["split_group_receipt_path"]),
        "--approved-group-exclusions", str(eligibility["catalog_path"]),
        "--historical-retrieval-receipt", str(eligibility["retrieval"]),
        "--raw-root", str(eligibility["raw_root"]), "--output", str(output),
    ])
    result = module.run(args)
    assert len(checked_graph) == 1
    assert result["counts"] == {"eligible_unit_count": 2, "exhausted_unit_count": 1,
                               "excluded_group_count": 1, "retained_group_count": 2}
    assert output.is_file()
    assert validate_representative_eligibility_receipt(output, **kwargs)["retained_group_count"] == 2
    with pytest.raises(ValueError, match="fresh private path"):
        module.run(args)


def test_group_names_cannot_reuse_approval_after_membership_changes(eligibility: dict[str, Any]) -> None:
    kwargs = eligibility["kwargs"]
    kwargs["cluster_map"]["2bbb_A"] = "2bbb_A"
    kwargs["split_group_map"]["2bbb_A"] = "3ccc_A"
    kwargs["original_units"] = {unit_id([chain]): [chain] for chain in kwargs["cluster_map"]}
    receipt = json.loads(eligibility["receipt_path"].read_text())
    receipt["population"] = population_record(**{key: value for key, value in kwargs.items()
                                                 if key != "approved_group_exclusions_path"})
    write_json(eligibility["receipt_path"], receipt)
    with pytest.raises(ValueError, match="exclusion scope differs"):
        validate(eligibility)


def test_validator_rehashes_inputs_after_source_replay(
    eligibility: dict[str, Any], monkeypatch: pytest.MonkeyPatch,
) -> None:
    import nanofold.representative_eligibility as module

    original = module.inspect_candidate
    changed = False

    def changing_input(**kwargs: Any) -> dict[str, Any]:
        nonlocal changed
        result = original(**kwargs)
        if not changed:
            cache = eligibility["kwargs"]["chain_data_cache_path"]
            cache.write_text(cache.read_text() + "\n")
            changed = True
        return result

    monkeypatch.setattr(module, "inspect_candidate", changing_input)
    with pytest.raises(ValueError, match="input or source changed"):
        validate(eligibility)
