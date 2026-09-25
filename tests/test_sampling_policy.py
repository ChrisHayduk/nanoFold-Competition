from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from scripts.build_hidden_manifest import _allocate_against_reference, _componentize, _load_lock_args
from scripts.build_manifests import (
    DEFAULT_STRATIFY_FIELDS,
    Candidate,
    _allocate_counts_by_stratum,
    _assert_quality_report,
    _assert_split_disjointness,
    _atomic_write,
    _build_candidates,
    _build_quality_report,
    _disjoint_units,
    _load_cluster_tsv,
    _load_structure_metadata,
    _parse_cath_payload,
    _resolution_bin,
    _select_stratified,
    _split_candidates,
    _validate_metadata_identity,
)


def _candidate(index: int, architecture: str = "1.10", *, length: int = 100) -> Candidate:
    sequence = "A" * (length - 12) + "".join("C" if index & (1 << bit) else "A" for bit in range(12))
    return Candidate(
        chain_id=f"{index:04d}_A", sequence=sequence, length=length, resolution=2.0,
        release_date=None, cluster_id=f"cluster-{index}", pdb_id=f"{index:04d}",
        secondary_structure_class="unknown", secondary_helix_fraction=None,
        secondary_beta_fraction=None, secondary_coil_fraction=None,
        domain_architecture_class="unknown", domain_architecture_source="cath", metadata_source_count=1,
        cath_architecture_ids=(architecture,), cath_topology_ids=(f"{architecture}.1",),
        cath_homologous_superfamily_ids=(f"{architecture}.1.1",),
        cath_annotation_status="annotated", cath_domain_count=1,
    )


def _metadata(chain_id: str = "0001_A") -> dict:
    return {
        "chain_id": chain_id, "accepted": True,
        "secondary_structure_class": "unknown", "secondary_helix_fraction": None,
        "secondary_beta_fraction": None, "secondary_coil_fraction": None,
        "domain_architecture_class": "alpha",
        "cath_annotation_status": "annotated", "cath_architecture_ids": ["1.10"],
        "cath_topology_ids": ["1.10.1"], "cath_homologous_superfamily_ids": ["1.10.1.1"],
        "cath_domains": [{"domain_id": "0001A01", "class_id": "1", "architecture_id": "1.10",
                          "topology_id": "1.10.1", "homologous_superfamily_id": "1.10.1.1"}],
    }


def test_hierarchical_quotas_preserve_architecture_under_sparse_joint_strata() -> None:
    # Each architecture has many distinct length/resolution cells; architecture comes first.
    candidates = [replace(_candidate(index, "1.10" if index < 80 else "2.20", length=40 + index),
                          resolution=1.0 + (index % 5) * 0.5) for index in range(100)]
    selected = _select_stratified(candidates, size=10, seed=8, salt="test", stratify_fields=DEFAULT_STRATIFY_FIELDS)
    assert sum(candidate.cath_architecture_ids == ("2.20",) for candidate in selected) == 2
    assert len({candidate.chain_id for candidate in selected}) == 10
    reverse = _select_stratified(list(reversed(candidates)), size=10, seed=8, salt="test",
                                 stratify_fields=DEFAULT_STRATIFY_FIELDS)
    assert selected == reverse


def test_equal_sparse_strata_receive_seeded_rather_than_lexical_priority() -> None:
    candidates = [_candidate(index, f"1.{index + 1}") for index in range(20)]
    allocations = [_allocate_counts_by_stratum(candidates, size=3, seed=seed,
                    stratify_fields=DEFAULT_STRATIFY_FIELDS) for seed in range(40)]
    assert all(sum(allocation.values()) == 3 for allocation in allocations)
    supported = {key[0] for allocation in allocations for key, count in allocation.items() if count}
    assert len(supported) == 20
    assert len({tuple(sorted(key for key, count in allocation.items() if count)) for allocation in allocations}) > 30


def test_fixed_public_capacity_shortfall_redistributes_proportionally() -> None:
    reference = [_candidate(index, "1.10" if index < 80 else "2.20" if index < 90 else "3.30")
                 for index in range(100)]
    pool = [_candidate(200, "1.10")] + [_candidate(201 + i, "2.20") for i in range(20)]
    pool += [_candidate(301 + i, "3.30") for i in range(20)]
    allocation = _allocate_against_reference(pool, reference, size=21, stratify_fields=DEFAULT_STRATIFY_FIELDS)
    assert {key[0]: count for key, count in allocation.items()} == {"1.10": 1, "2.20": 10, "3.30": 10}


def test_pdb_cluster_and_exact_sequence_links_form_transitive_components() -> None:
    a, b, c, d = [_candidate(index) for index in range(4)]
    b = replace(b, sequence=a.sequence)
    c = replace(c, pdb_id=b.pdb_id)
    mapping = {a.chain_id: "a", b.chain_id: "b", c.chain_id: "d", d.chain_id: "d"}
    units = _disjoint_units([d, c, b, a], mapping, disallow_pdb_overlap=True, seed=0)
    components = _componentize([a, b, c, d], mapping, seed=0)
    assert len(units) == len(components) == 1
    assert {member.chain_id for member in components[0].members} == {a.chain_id, b.chain_id, c.chain_id, d.chain_id}
    assert components[0].representative == units[0]


def test_split_is_reproducible_disjoint_and_uses_pool_reference() -> None:
    candidates = [_candidate(index, "1.10" if index % 3 else "2.20") for index in range(60)]
    cluster_map = {candidate.chain_id: candidate.cluster_id for candidate in candidates}
    kwargs: dict[str, Any] = dict(cluster_map=cluster_map, train_size=30, val_size=10, hidden_val_size=10, seed=17,
                  hidden_split_salt_digest="private-test", stratify_fields=DEFAULT_STRATIFY_FIELDS,
                  disallow_pdb_overlap=True, cluster_method="test", cluster_command=None,
                  structural_metadata_source="fixture")
    result = _split_candidates(candidates=candidates, **kwargs)
    shuffled = _split_candidates(candidates=list(reversed(candidates)), **kwargs)
    _assert_split_disjointness(result)
    assert result == shuffled
    assert result.quality_report["reference_annotation_coverage"]["chain_count"] == 60
    assert result.quality_report["reference_population"] == "eligible_disjoint_units"


def test_quality_gate_detects_shared_selection_bias_and_is_not_disabled_by_small_split() -> None:
    reference = [_candidate(index, "1.10" if index < 100 else "2.20") for index in range(1000)]
    selected = {candidate.chain_id: candidate for candidate in reference[:101]}
    split_ids = {"train": list(selected)[:100], "val": list(selected)[100:]}
    report = _build_quality_report(selected_candidates=selected, split_ids=split_ids,
                                   stratify_fields=DEFAULT_STRATIFY_FIELDS, reference_candidates=reference)
    with pytest.raises(ValueError, match="stratification quality gate"):
        _assert_quality_report(report)


def test_unknown_annotations_remain_unknown_and_partial_familiarity_is_not_hidden() -> None:
    train = _candidate(0)
    partial = replace(_candidate(1), cath_topology_ids=("1.10.1", "2.20.1"),
                      cath_homologous_superfamily_ids=("1.10.1.1", "2.20.1.1"))
    unknown = replace(_candidate(2), cath_architecture_ids=(), cath_topology_ids=(),
                      cath_homologous_superfamily_ids=(), cath_annotation_status="unknown")
    report = _build_quality_report(selected_candidates={candidate.chain_id: candidate for candidate in (train, partial, unknown)},
                                   split_ids={"train": [train.chain_id], "val": [partial.chain_id, unknown.chain_id]},
                                   stratify_fields=DEFAULT_STRATIFY_FIELDS)
    assert report["training_familiarity"]["val"]["cath_topology_ids"] == {
        "ALL_SEEN": 0, "NONE_SEEN": 0, "SOME_SEEN": 1, "UNANNOTATED": 1,
    }
    assert report["annotation_coverage"]["val"]["cath_unknown_fraction"] == 0.5


def test_cath_domains_preserve_multiplicity_and_reject_conflicting_summary() -> None:
    payload = _metadata()
    payload["cath_domains"].append({**payload["cath_domains"][0], "domain_id": "0001A02"})
    assert _parse_cath_payload(payload)["cath_domain_count"] == 2
    payload["cath_topology_ids"] = ["1.10.2"]
    with pytest.raises(ValueError, match="disagree"):
        _parse_cath_payload(payload)


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), -0.1, 1.1, True])
def test_schema2_rejects_invalid_measured_fractions(tmp_path: Path, invalid: float) -> None:
    payload = _metadata()
    payload.update(secondary_helix_fraction=invalid, secondary_beta_fraction=0.2, secondary_coil_fraction=0.1)
    path = tmp_path / "metadata.json"
    path.write_text(json.dumps({"schema_version": 2, "chains": [payload]}))
    with pytest.raises(ValueError, match="fractions"):
        _load_structure_metadata(path, require_schema_version=2)


def test_schema2_allows_unmeasured_secondary_and_rejects_duplicate_rows(tmp_path: Path) -> None:
    path = tmp_path / "metadata.json"
    path.write_text(json.dumps({"schema_version": 2, "chains": [_metadata()]}))
    assert _load_structure_metadata(path, require_schema_version=2)[0] == {"0001_A"}
    path.write_text(json.dumps({"schema_version": 2, "chains": [_metadata(), _metadata()]}))
    with pytest.raises(ValueError, match="duplicate"):
        _load_structure_metadata(path, require_schema_version=2)


def _filter(cache_entry: dict, **kwargs):
    return _build_candidates({"0001_A": cache_entry}, min_len=40, max_len=256, max_resolution=3,
                             structure_metadata={"0001_A": _metadata()}, structure_metadata_accepted={"0001_A"}, **kwargs)


def test_single_chain_target_does_not_imply_monomer_and_asu_count_is_not_assembly_count() -> None:
    cache = {"sequence": "A" * 100, "seq_length": 100, "resolution": 2.0, "num_chains": 1}
    assert len(_filter(cache)[0]) == 1
    assert not _filter(cache, require_monomer=True)[0]
    assert len(_filter({**cache, "oligomeric_count": 1}, require_monomer=True)[0]) == 1
    assert not _filter({**cache, "oligomeric_count": 1.5}, require_monomer=True)[0]


@pytest.mark.parametrize("resolution", [None, 0, -1, float("nan"), float("inf"), 999])
def test_unknown_resolution_does_not_become_high_quality(resolution: float | None) -> None:
    candidates, _, _ = _filter({"sequence": "A" * 100, "seq_length": 100, "resolution": resolution})
    assert len(candidates) == 1
    assert _resolution_bin(candidates[0].resolution) == "res_unknown"


@pytest.mark.parametrize("length", [101, 100.5, float("nan"), True, "bad"])
def test_cache_declared_length_cannot_silently_disagree_with_sequence(length: object) -> None:
    assert not _filter({"sequence": "A" * 100, "seq_length": length, "resolution": 2})[0]


@pytest.mark.parametrize("residue", ["X", "B", "Z", "U", "O", "-", "*", " "])
def test_candidate_gate_rejects_nonstandard_residues_even_if_metadata_accepts(residue: str) -> None:
    candidates, rejections, _ = _filter({"sequence": "A" * 49 + residue + "A" * 50, "resolution": 2})
    assert candidates == []
    assert rejections["nonstandard_amino_acids"] == 1


@pytest.mark.parametrize("text", ["a\tb\nsecond\tb\n", "a\tb\textra\n", "a\t\n", "a\n"])
def test_cluster_tsv_rejects_malformed_or_conflicting_assignments(tmp_path: Path, text: str) -> None:
    path = tmp_path / "clusters.tsv"
    path.write_text(text)
    with pytest.raises(ValueError, match="Malformed|Conflicting"):
        _load_cluster_tsv(path, {"b"})


def test_metadata_identity_binds_cache_namespace_and_sampling_filters(tmp_path: Path) -> None:
    args = argparse.Namespace(min_len=40, max_len=256, max_resolution=3.0, require_monomer=False)
    raw = {"source": {"chain_data_cache_sha256": "cache", "chain_id_namespace": "author_asym_id"},
           "config": {**vars(args), "max_unknown_aa_fraction": 0, "sample_limit": 0},
           "summary": {"eligible_candidate_count": 1, "candidate_count": 1}, "chains": [_metadata()]}
    path = tmp_path / "metadata.json"
    path.write_text(json.dumps(raw))
    _validate_metadata_identity(path, "cache", args)
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        _validate_metadata_identity(path, "different", args)
    with pytest.raises(ValueError, match="filter mismatch"):
        _validate_metadata_identity(path, "cache", argparse.Namespace(**{**vars(args), "min_len": 50}))


@pytest.mark.parametrize("section,field,value", [
    ("config", "sample_limit", None), ("config", "sample_limit", 1),
    ("config", "sample_limit", 100), ("config", "sample_limit", False),
    ("config", "max_unknown_aa_fraction", None), ("config", "max_unknown_aa_fraction", 0.1),
    ("config", "max_unknown_aa_fraction", False), ("config", "max_unknown_aa_fraction", float("nan")),
    ("summary", "eligible_candidate_count", None), ("summary", "eligible_candidate_count", 2),
    ("summary", "candidate_count", None), ("summary", "candidate_count", 0),
    ("summary", "candidate_count", True), ("config", "require_monomer", 0),
])
def test_cath_v2_rejects_incomplete_or_permissive_metadata(
    tmp_path: Path, section: str, field: str, value: object,
) -> None:
    args = argparse.Namespace(min_len=40, max_len=256, max_resolution=3.0, require_monomer=False)
    raw: dict[str, Any] = {
        "source": {"chain_data_cache_sha256": "cache", "chain_id_namespace": "author_asym_id"},
        "config": {**vars(args), "max_unknown_aa_fraction": 0, "sample_limit": 0},
        "summary": {"eligible_candidate_count": 1, "candidate_count": 1}, "chains": [_metadata()],
    }
    if value is None:
        del raw[section][field]
    else:
        raw[section][field] = value
    path = tmp_path / "metadata.json"
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="sample_limit|max_unknown_aa_fraction|full-universe|filter mismatch"):
        _validate_metadata_identity(path, "cache", args)


@pytest.mark.parametrize("field,value", [
    ("hidden_val_size", True), ("hidden_val_size", 1.5), ("min_len", 0),
    ("max_len", 39), ("max_resolution", float("nan")), ("max_resolution", 0),
    ("coverage", float("inf")), ("min_seq_id", False), ("seed", 1.5),
    ("require_monomer", "false"), ("require_monomer", 0),
])
def test_source_lock_rejects_coercible_or_invalid_sampling_arguments(tmp_path: Path, field: str, value: object) -> None:
    args = {"hidden_val_size": 1, "min_len": 40, "max_len": 256, "max_resolution": 3.0,
            "seed": 0, "min_seq_id": 0.3, "coverage": 0.8, "mmseqs_bin": "mmseqs",
            "structure_metadata": "metadata.json", "require_monomer": False}
    path = tmp_path / "source.lock.json"
    path.write_text(json.dumps({"sampling_policy": "cath-v2", "args": {**args, field: value}}))
    with pytest.raises(ValueError):
        _load_lock_args(path)


def test_atomic_write_preserves_existing_file_when_replace_fails(tmp_path: Path, monkeypatch) -> None:
    path = tmp_path / "manifest.txt"
    path.write_text("old\n")
    def fail_replace(*_args):
        raise OSError("simulated replacement failure")
    monkeypatch.setattr("scripts.build_manifests.os.replace", fail_replace)
    with pytest.raises(OSError, match="simulated"):
        _atomic_write(path, "new\n")
    assert path.read_text() == "old\n"
    assert list(tmp_path.iterdir()) == [path]


def test_private_salt_cannot_change_public_selection() -> None:
    candidates = [_candidate(index, "1.10" if index % 3 else "2.20") for index in range(100)]
    kwargs: dict[str, Any] = dict(candidates=candidates, cluster_map={c.chain_id: c.cluster_id for c in candidates},
                  train_size=60, val_size=20, hidden_val_size=10, seed=0,
                  stratify_fields=DEFAULT_STRATIFY_FIELDS, disallow_pdb_overlap=True,
                  cluster_method="test", cluster_command=None, structural_metadata_source="fixture")
    first = _split_candidates(hidden_split_salt_digest="private-salt-one", **kwargs)
    second = _split_candidates(hidden_split_salt_digest="private-salt-two", **kwargs)
    assert first.train_ids == second.train_ids
    assert first.val_ids == second.val_ids
    assert first.hidden_val_ids != second.hidden_val_ids
