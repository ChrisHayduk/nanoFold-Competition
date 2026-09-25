from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from scripts import build_structure_metadata as metadata


def _write_sources(tmp_path: Path, cache: Path, cath_text: str | None = None) -> tuple[Path, Path]:
    sources = tmp_path / "sources"
    sources.mkdir(exist_ok=True)
    cath = sources / "cath-domain-list.txt"
    cath.write_text(cath_text or "1okyA00 1 10 20 30 1 1 1 1 1 60 2.0\n")
    lock = tmp_path / "sources.lock.json"
    lock.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "chain_data_cache": {"sha256": hashlib.sha256(cache.read_bytes()).hexdigest()},
                "sources": {
                    "cath_domain_list": {"path": cath.name, "sha256": hashlib.sha256(cath.read_bytes()).hexdigest()}
                },
            }
        )
    )
    return sources, lock


def _build(tmp_path: Path, data: dict[str, Any], cath_text: str | None = None, **kwargs: Any) -> dict[str, Any]:
    cache = tmp_path / "cache.json"
    cache.write_text(json.dumps(data))
    sources, lock = _write_sources(tmp_path, cache, cath_text)
    feature_exclusions = tmp_path / "features.txt"
    processability_exclusions = tmp_path / "processability.txt"
    feature_exclusions.touch(exist_ok=True)
    processability_exclusions.touch(exist_ok=True)
    return metadata.build_metadata(
        chain_data_cache=cache,
        metadata_sources_dir=sources,
        metadata_source_lock=lock,
        feature_exclusion_list=feature_exclusions,
        processability_exclusion_list=processability_exclusions,
        min_len=4,
        max_len=256,
        max_resolution=3,
        max_unknown_aa_fraction=0,
        **kwargs,
    )


def test_cath_preserves_hierarchy_multiple_domains_and_chain_case(tmp_path: Path) -> None:
    cath = tmp_path / "cath.txt"
    cath.write_text(
        "# independently specified CATH list records\n"
        "1abcA02 2 40 50 60 1 1 1 1 1 90 2.0\n"
        "1abcA01 1 10 20 30 1 1 1 1 1 70 2.0\n"
        "1abcA03 1 10 20 30 1 1 1 1 1 80 2.0\n"
        "1abca00 6 10 20 30 1 1 1 1 1 70 2.0\n"
    )
    observed = metadata._load_cath_annotations(cath)
    assert observed["1abc_A"] == [
        {
            "domain_id": "1abcA01",
            "class_id": "1",
            "architecture_id": "1.10",
            "topology_id": "1.10.20",
            "homologous_superfamily_id": "1.10.20.30",
        },
        {
            "domain_id": "1abcA02",
            "class_id": "2",
            "architecture_id": "2.40",
            "topology_id": "2.40.50",
            "homologous_superfamily_id": "2.40.50.60",
        },
        {
            "domain_id": "1abcA03",
            "class_id": "1",
            "architecture_id": "1.10",
            "topology_id": "1.10.20",
            "homologous_superfamily_id": "1.10.20.30",
        },
    ]
    assert observed["1abc_a"][0]["homologous_superfamily_id"] == "6.10.20.30"


@pytest.mark.parametrize(
    "line",
    [
        "<html>download unavailable</html>",
        "1abcA00 1 10 20 30",
        "1abcAA00 1 10 20 30 1 1 1 1 1 60 2.0",
        "1abcA00 1 10 NaN 30 1 1 1 1 1 60 2.0",
        "1abcA00 1 0 20 30 1 1 1 1 1 60 2.0",
        "1abcA00 1 10 20 30 1 1 1 1 1 0 2.0",
        "1abcA00 1 10 20 30 1 1 1 1 1 60 NaN",
    ],
)
def test_cath_rejects_malformed_records(tmp_path: Path, line: str) -> None:
    path = tmp_path / "cath.txt"
    path.write_text(line + "\n")
    with pytest.raises(ValueError):
        metadata._load_cath_annotations(path)


def test_cath_duplicate_conflicts_fail_but_identical_rows_deduplicate(tmp_path: Path) -> None:
    path = tmp_path / "cath.txt"
    row = "1abcA00 1 10 20 30 1 1 1 1 1 60 2.0\n"
    path.write_text(row * 2)
    assert len(metadata._load_cath_annotations(path)["1abc_A"]) == 1
    path.write_text(row + row.replace("20 30", "20 40"))
    with pytest.raises(ValueError, match="Conflicting CATH"):
        metadata._load_cath_annotations(path)


def test_metadata_does_not_fabricate_secondary_measurements_or_unknown_annotation(tmp_path: Path) -> None:
    result = _build(
        tmp_path,
        {
            "1oky_A": {"seq": "AAAA", "resolution": 2.0},
            "2xyz_AA": {"seq": "AAAA", "resolution": 0.0},
        },
    )
    assert result["schema_version"] == 2
    assert result["source"]["chain_id_namespace"] == "author_asym_id"
    annotated, unknown = result["chains"]
    assert annotated["cath_architecture_ids"] == ["1.10"]
    assert annotated["cath_topology_ids"] == ["1.10.20"]
    assert annotated["cath_homologous_superfamily_ids"] == ["1.10.20.30"]
    for row in result["chains"]:
        assert row["secondary_structure_class"] == "unknown"
        assert row["secondary_structure_source"] is None
        assert row["secondary_helix_fraction"] is None
        assert row["secondary_beta_fraction"] is None
        assert row["secondary_coil_fraction"] is None
        assert row["oligomeric_state"] == "unknown"
    assert unknown["cath_annotation_status"] == "unknown"
    assert unknown["domain_architecture_class"] == "unknown"
    assert unknown["metadata_source_count"] == 0
    assert unknown["resolution"] == 999.0


def test_multidomain_chain_preserves_domain_multiplicity(tmp_path: Path) -> None:
    result = _build(
        tmp_path,
        {"1oky_A": {"seq": "AAAA"}},
        "1okyA01 1 10 20 30 1 1 1 1 1 60 2.0\n"
        "1okyA02 1 10 20 30 1 1 1 1 1 60 2.0\n"
        "1okyA03 2 40 50 60 1 1 1 1 1 60 2.0\n",
    )
    row = result["chains"][0]
    assert row["domain_architecture_class"] == "multi_class"
    assert row["cath_architecture_ids"] == ["1.10", "2.40"]
    assert [domain["architecture_id"] for domain in row["cath_domains"]] == ["1.10", "1.10", "2.40"]


def test_cath_blank_chain_code_is_not_mapped_to_literal_zero(tmp_path: Path) -> None:
    result = _build(tmp_path, {"1oky_0": {"seq": "AAAA"}}, "1oky000 1 10 20 30 1 1 1 1 1 60 2.0\n")
    row = result["chains"][0]
    assert row["cath_domains"] == []
    assert row["cath_mapping_status"] == "ambiguous_blank_chain"
    assert row["cath_annotation_status"] == "unknown"
    assert row["cath_unresolved_domains"][0]["domain_id"] == "1oky000"


@pytest.mark.parametrize(
    "mutation,error",
    [
        ("cache", "chain_data_cache SHA-256 mismatch"),
        ("cath", "SHA-256 mismatch"),
        ("missing", "Missing metadata source"),
        ("lock", "Missing metadata source lock"),
    ],
)
def test_source_verification_rejects_stale_changed_and_missing_inputs(
    tmp_path: Path, mutation: str, error: str
) -> None:
    cache = tmp_path / "cache.json"
    cache.write_text("{}")
    sources, lock = _write_sources(tmp_path, cache)
    if mutation == "cache":
        cache.write_text('{"changed": true}')
    elif mutation == "cath":
        (sources / "cath-domain-list.txt").write_text("changed")
    elif mutation == "missing":
        (sources / "cath-domain-list.txt").unlink()
    else:
        lock.unlink()
    with pytest.raises(ValueError, match=error):
        metadata._verify_source_lock(lock, sources, cache)


def test_unlocked_optional_files_are_not_consumed(tmp_path: Path) -> None:
    cache = tmp_path / "cache.json"
    cache.write_text("{}")
    sources, lock = _write_sources(tmp_path, cache)
    (sources / "dir.cla.scope.txt").write_text("malformed unlocked source")
    verified = metadata._verify_source_lock(lock, sources, cache)
    external, _, _ = metadata._load_external_sources(sources, verified)
    assert external["scope"] == {}


def test_source_conflicts_reported_without_picking_one_domain(tmp_path: Path) -> None:
    cls, source, values = metadata._select_domain_architecture(
        "1abc_A",
        {
            "cath": {"1abc_A": {"alpha", "beta"}},
            "scope": {"1abc_A": {"alpha_beta"}},
        },
    )
    assert (cls, source) == ("multi_class", "cath")
    assert values == {"cath": ["alpha", "beta"], "scope": ["alpha_beta"]}


def test_candidate_validation_is_explicit_about_assembly_and_length() -> None:
    data = {
        "1abc_A": {"seq": "AAAA", "num_chains": 1},
        "2abc_A": {"seq": "AAAA", "oligomeric_count": 1},
        "3abc_A": {"seq": "AAAA", "oligomeric_count": 2},
        "4abc_A": {"seq": "AAAA", "seq_length": 4.9},
        "5abc_A": {"seq": "AXXX"},
    }
    kwargs: dict[str, Any] = dict(min_len=4, max_len=256, max_resolution=3, max_unknown_aa_fraction=0)
    rows, rejected = metadata._candidate_rows(data, **kwargs)
    assert [row["chain_id"] for row in rows] == ["1abc_A", "2abc_A", "3abc_A"]
    assert rejected == {"sequence_length_mismatch": 1, "unknown_aa_fraction": 1}
    rows, rejected = metadata._candidate_rows(data, require_monomer=True, **kwargs)
    assert [row["chain_id"] for row in rows] == ["2abc_A"]
    assert rejected["unknown_oligomeric_count"] == 3
    assert rejected["oligomeric_count"] == 1


@pytest.mark.parametrize(
    "filename,reason",
    [
        ("features.txt", "missing_required_feature_asset"),
        ("processability.txt", "failed_official_label_processability"),
    ],
)
def test_required_feature_and_label_exclusions(tmp_path: Path, filename: str, reason: str) -> None:
    (tmp_path / filename).write_text("2bad_A\n")
    result = _build(tmp_path, {"1oky_A": {"seq": "AAAA"}, "2bad_A": {"seq": "AAAA"}})
    assert result["summary"]["accepted"] == 1
    assert result["chains"][1]["reject_reasons"] == [reason]


@pytest.mark.parametrize("sample_limit,processed_count,accepted_count", [(0, 4, 2), (3, 3, 1)])
def test_metadata_records_full_eligible_universe_before_debug_truncation(
    tmp_path: Path, sample_limit: int, processed_count: int, accepted_count: int
) -> None:
    (tmp_path / "features.txt").write_text("2bad_A\n")
    (tmp_path / "processability.txt").write_text("3bad_A\n")
    result = _build(
        tmp_path,
        {
            "1oky_A": {"seq": "AAAA"},
            "2bad_A": {"seq": "AAAA"},
            "3bad_A": {"seq": "AAAA"},
            "4oky_A": {"seq": "AAAA"},
            "5bad_A": {"seq": "A"},
        },
        sample_limit=sample_limit,
    )
    assert result["config"]["sample_limit"] == sample_limit
    assert result["summary"]["eligible_candidate_count"] == 4
    assert result["summary"]["candidate_count"] == processed_count == len(result["chains"])
    assert result["summary"]["accepted"] == accepted_count
    assert result["summary"]["rejected"] == 2
    assert result["summary"]["reject_reasons"] == {
        "min_length": 1,
        "missing_required_feature_asset": 1,
        "failed_official_label_processability": 1,
    }
    assert [row["chain_id"] for row in result["chains"] if not row["accepted"]] == ["2bad_A", "3bad_A"]


@pytest.mark.parametrize("sample_limit", [-1, True, False, 1.5, "2", None])
def test_metadata_rejects_invalid_debug_sample_limit(tmp_path: Path, sample_limit: Any) -> None:
    with pytest.raises(ValueError, match="sample_limit must be a nonnegative integer"):
        _build(tmp_path, {"1oky_A": {"seq": "AAAA"}}, sample_limit=sample_limit)


def test_cli_outputs_repeatable_sorted_candidates(tmp_path: Path) -> None:
    cache = tmp_path / "cache.json"
    cache.write_text(json.dumps({"2xyz_A": {"seq": "A" * 40}, "1oky_A": {"seq": "A" * 40}}))
    sources, lock = _write_sources(tmp_path, cache)
    output, candidates = tmp_path / "metadata.json", tmp_path / "candidates.txt"
    argv = [
        "--chain-data-cache",
        str(cache),
        "--metadata-sources-dir",
        str(sources),
        "--metadata-source-lock",
        str(lock),
        "--metadata-out",
        str(output),
        "--candidate-manifest-out",
        str(candidates),
    ]
    metadata.main(argv)
    first = json.loads(output.read_text())
    metadata.main(argv)
    second = json.loads(output.read_text())
    first.pop("generated_at")
    second.pop("generated_at")
    assert first == second
    assert candidates.read_text() == "1oky_A\n2xyz_A\n"


def test_declared_optional_source_requires_a_hash_even_when_file_exists(tmp_path: Path) -> None:
    cache = tmp_path / "cache.json"
    cache.write_text("{}")
    sources, lock = _write_sources(tmp_path, cache)
    obj = json.loads(lock.read_text())
    obj["sources"]["scope_classification"] = {"path": "dir.cla.scope.txt", "sha256": None}
    lock.write_text(json.dumps(obj))
    (sources / "dir.cla.scope.txt").write_text("unverified")
    with pytest.raises(ValueError, match="Missing valid sha256"):
        metadata._verify_source_lock(lock, sources, cache)
    obj["sources"]["scope_classification"]["skipped"] = True
    lock.write_text(json.dumps(obj))
    assert set(metadata._verify_source_lock(lock, sources, cache)) == {"cath_domain_list"}


def test_locked_secondary_annotations_are_measured_and_tampering_is_rejected(tmp_path: Path) -> None:
    cache = tmp_path / "cache.json"
    cache.write_text(json.dumps({"1oky_A": {"seq": "A" * 10}}))
    sources, lock = _write_sources(tmp_path, cache)
    dssp = sources / "dssp_mmcif"
    dssp.mkdir()
    cif = dssp / "1oky.cif"
    cif.write_text("""data_test
loop_
_struct_conf.conf_type_id
_struct_conf.beg_auth_asym_id
_struct_conf.end_auth_asym_id
_struct_conf.beg_label_seq_id
_struct_conf.end_label_seq_id
HELX_P A A 1 5
""")
    with cif.open("a") as handle:
        handle.write(
            "loop_\n_atom_site.auth_asym_id\n_atom_site.label_asym_id\n_atom_site.label_atom_id\n_atom_site.label_seq_id\n"
        )
        handle.write("_atom_site.Cartn_x\n_atom_site.Cartn_y\n_atom_site.Cartn_z\n")
        handle.write("".join(f"A X CA {position} {position}.0 0.0 0.0\n" for position in range(1, 11)))
    obj = json.loads(lock.read_text())
    # Construct the tree digest independently of the implementation under test.
    tree_record = f"1oky.cif\t{hashlib.sha256(cif.read_bytes()).hexdigest()}\n"
    obj["sources"]["dssp_mmcif"] = {"tree_sha256": hashlib.sha256(tree_record.encode()).hexdigest()}
    lock.write_text(json.dumps(obj))
    exclusions = tmp_path / "exclusions.txt"
    exclusions.touch()
    result = metadata.build_metadata(
        chain_data_cache=cache,
        metadata_sources_dir=sources,
        metadata_source_lock=lock,
        feature_exclusion_list=exclusions,
        processability_exclusion_list=exclusions,
        min_len=4,
        max_len=256,
        max_resolution=3,
        max_unknown_aa_fraction=0,
    )
    row = result["chains"][0]
    assert row["secondary_helix_fraction"] == 0.5
    assert row["secondary_structure_source"] == "mmcif_annotations"
    cif.write_text(cif.read_text().replace("1 5", "1 8"))
    with pytest.raises(ValueError, match="SHA-256 mismatch: dssp_mmcif"):
        metadata._verify_source_lock(lock, sources, cache)


def test_missing_exclusion_list_fails_closed(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Missing required exclusion list"):
        metadata._load_feature_exclusions(tmp_path / "absent.txt")


def test_noncanonical_pdb_aliases_cannot_hide_annotation_or_bypass_exclusions(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="lowercase"):
        _build(tmp_path, {"1OKY_A": {"seq": "AAAA"}, "1oky_A": {"seq": "AAAA"}})
    exclusions = tmp_path / "upper.txt"
    exclusions.write_text("1OKY_A\n")
    with pytest.raises(ValueError, match="lowercase"):
        metadata._load_feature_exclusions(exclusions)


def test_rcsb_entry_only_metadata_does_not_claim_chain_coverage(tmp_path: Path) -> None:
    path = tmp_path / "rcsb.jsonl"
    path.write_text(json.dumps({
        "chain_id": "2or1_L", "entry": {"exptl": [{"method": "X-RAY DIFFRACTION"}]},
        "entity": {}, "instance": {}, "errors": ["instance unavailable"],
    }) + "\n")
    annotations, details = metadata._load_rcsb_annotations(path)
    assert annotations == {}
    assert details["2or1_L"]["experimental_method"] == "X-RAY DIFFRACTION"
    assert details["2or1_L"]["chain_mapping_status"] == "unresolved"


def test_scope_small_protein_and_coiled_coil_do_not_become_unstructured_coil(tmp_path: Path) -> None:
    path = tmp_path / "scope.txt"
    path.write_text(
        "d1abca_ 1abc A:1-60 g.1.1.1 1 cl=1\n"
        "d2abcb_ 2abc B:1-60 h.1.1.1 2 cl=2\n"
    )
    assert metadata._load_scope_annotations(path) == {
        "1abc_A": {"small_protein"}, "2abc_B": {"coiled_coil"},
    }
    from nanofold.structure_metadata import normalize_domain_architecture_class

    assert normalize_domain_architecture_class("small protein") == "small_protein"
