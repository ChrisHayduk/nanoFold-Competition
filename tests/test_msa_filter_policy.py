from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
from msa_filter_policy_helpers import digest_file, refresh_filter_binding, write_valid_policy_case

from nanofold.msa_filter_policy import (
    build_msa_filter_policy,
    load_msa_filter_policy,
    load_msa_row_filter,
    validate_msa_filter_policy_output,
    validate_msa_filter_policy_provenance,
)


@pytest.fixture
def policy_case(tmp_path: Path) -> dict[str, Path]:
    return write_valid_policy_case(tmp_path)


def _edit_filter(paths: dict[str, Path], edit: Any, role: str = "train") -> None:
    path = paths[f"{role}_filter"]
    payload = json.loads(path.read_text())
    edit(payload)
    path.write_text(json.dumps(payload, sort_keys=True))
    refresh_filter_binding(paths, role)


def test_policy_enforces_direction_and_keeps_hidden_provenance_private(policy_case: dict[str, Path]) -> None:
    policy = load_msa_filter_policy(policy_case["policy_path"], expected_chain_ids=["trn1_A", "val1_A"])
    validation_hash = hashlib.sha256(("C" * 20).encode()).hexdigest()
    hidden_hash = hashlib.sha256(("D" * 20).encode()).hexdigest()
    assert policy.filter_for("trn1_A").excluded_sequence_sha256 == {validation_hash, hidden_hash}
    assert policy.filter_for("val1_A").excluded_sequence_sha256 == {hidden_hash}
    assert policy.chain_data_cache_sha256 == digest_file(policy_case["cache"])
    public = json.dumps(policy.provenance)
    for private in ("hid1_A", str(policy_case["hidden_manifest"]), digest_file(policy_case["hidden_manifest"])):
        assert private not in public
    assert set(policy.manifest_paths) == {"train", "val"}
    with pytest.raises(ValueError, match="no public"):
        policy.filter_for("hid1_A")
    policy.assert_unchanged()


def test_builder_validates_and_returns_canonical_policy(policy_case: dict[str, Path], tmp_path: Path) -> None:
    inputs: dict[str, Any] = {key: policy_case[key] for key in ("train_manifest", "val_manifest", "hidden_manifest", "train_filter", "val_filter")}
    payload = build_msa_filter_policy(**inputs)
    assert payload == json.loads(policy_case["policy_path"].read_text())
    output = tmp_path / "new" / "policy.json"
    assert validate_msa_filter_policy_output(output, payload) == output


@pytest.mark.parametrize("chain_ids", [["trn1_A"], ["trn1_A", "val1_A", "hid1_A"], ["trn1_A", "val1_A", "val1_A"]])
def test_manifest_requires_exact_public_union(policy_case: dict[str, Path], chain_ids: list[str]) -> None:
    with pytest.raises(ValueError, match="exactly cover"):
        load_msa_filter_policy(policy_case["policy_path"], expected_chain_ids=chain_ids)


@pytest.mark.parametrize("kind", ["swapped", "missing_role", "extra_role", "overlap", "duplicate_chain", "unknown_key"])
def test_invalid_assignments_are_rejected(policy_case: dict[str, Path], kind: str) -> None:
    path = policy_case["policy_path"]
    payload = json.loads(path.read_text())
    if kind == "swapped":
        payload["filters"]["train"], payload["filters"]["val"] = payload["filters"]["val"], payload["filters"]["train"]
    elif kind == "missing_role":
        del payload["filters"]["val"]
    elif kind == "extra_role":
        payload["filters"]["hidden"] = payload["filters"]["val"]
    elif kind == "unknown_key":
        payload["chain_assignments"] = {}
    else:
        role = "hidden" if kind == "overlap" else "train"
        manifest = policy_case[f"{role}_manifest"]
        manifest.write_text("trn1_A\n" if kind == "overlap" else "trn1_A\ntrn1_A\n")
        payload["manifests"][role]["sha256"] = digest_file(manifest)
        payload["manifests"][role]["chain_count"] = 1 if kind == "overlap" else 2
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        load_msa_filter_policy(path)


@pytest.mark.parametrize("kind", ["exhaustive", "source_complete", "missing_msa", "minimum_length", "threshold", "wrong_targets", "excluded_count", "duplicate_exclusion", "uppercase_hash", "native_capacity", "heuristic", "source_assets", "source_set", "heldout_set", "no_checkpoint"])
def test_incoherent_filter_proof_is_rejected(policy_case: dict[str, Path], kind: str) -> None:
    def edit(payload: dict[str, Any]) -> None:
        if kind == "exhaustive":
            payload["exhaustive"] = False
        elif kind == "source_complete":
            payload["source"]["complete"] = False
        elif kind == "missing_msa":
            payload["source"]["stats"]["missing_msa_files"] = 1
        elif kind == "minimum_length":
            payload["source"]["minimum_non_query_sequence_length"] = 21
        elif kind == "threshold":
            payload["threshold"]["coverage"] = 0.99
        elif kind == "wrong_targets":
            payload["heldout"]["manifests"] = payload["heldout"]["manifests"][1:]
        elif kind == "excluded_count":
            payload["excluded_sequence_count"] -= 1
        elif kind == "duplicate_exclusion":
            payload["excluded_sequence_sha256"].append(payload["excluded_sequence_sha256"][0])
            payload["excluded_sequence_count"] += 1
        elif kind == "uppercase_hash":
            payload["excluded_sequence_sha256"][0] = payload["excluded_sequence_sha256"][0].upper()
        elif kind == "native_capacity":
            payload["mmseqs"]["max_seqs"] = 1
        elif kind == "heuristic":
            payload["mmseqs"]["search_parameters"]["prefilter_mode"] = 0
        elif kind == "source_assets":
            payload["input_binding"]["source_assets"] = []
            payload["input_binding"]["source_asset_manifest_sha256"] = hashlib.sha256(b"[]").hexdigest()
        elif kind in ("source_set", "heldout_set"):
            key = "source_sequence_set_sha256" if kind == "source_set" else "heldout_sequence_set_sha256"
            payload["input_binding"][key] = "a" * 64
        else:
            del payload["mmseqs"]["batching"]
            del payload["input_binding"]["batch_receipt_sha256"]
    _edit_filter(policy_case, edit)
    with pytest.raises(ValueError):
        load_msa_filter_policy(policy_case["policy_path"])


def test_count_preserving_forged_exclusion_set_is_rejected(policy_case: dict[str, Path]) -> None:
    _edit_filter(policy_case, lambda payload: payload.update(excluded_sequence_sha256=["a" * 64, "b" * 64]))
    with pytest.raises(ValueError, match="exclusions do not equal"):
        load_msa_filter_policy(policy_case["policy_path"])


def test_exact_exclusion_count_is_replayed(policy_case: dict[str, Path]) -> None:
    _edit_filter(policy_case, lambda payload: payload.update(exact_match_excluded_sequence_count=1))
    with pytest.raises(ValueError, match="exclusions do not equal"):
        load_msa_filter_policy(policy_case["policy_path"])


@pytest.mark.parametrize("dependency", ["train_filter", "hidden_manifest", "cache", "train_msa"])
def test_dependency_mutations_fail_loaded_snapshot(policy_case: dict[str, Path], dependency: str) -> None:
    policy = load_msa_filter_policy(policy_case["policy_path"])
    policy_case[dependency].write_bytes(policy_case[dependency].read_bytes() + b"\n")
    with pytest.raises(ValueError, match="dependency changed"):
        policy.assert_unchanged()


def test_preprocessing_consumes_only_the_proven_msa_bytes(policy_case: dict[str, Path], tmp_path: Path) -> None:
    policy = load_msa_filter_policy(policy_case["policy_path"])
    original = policy_case["train_msa"]
    assert policy.assert_msa_inputs("trn1_A", {"uniref90_hits.a3m": original}) == {"uniref90_hits.a3m": original}
    copied = tmp_path / "copied.a3m"
    copied.write_bytes(original.read_bytes())
    with pytest.raises(ValueError, match="not bound"):
        policy.assert_msa_inputs("trn1_A", {"uniref90_hits.a3m": copied})
    with pytest.raises(ValueError, match="names differ"):
        policy.assert_msa_names(["different.a3m"])
    with pytest.raises(ValueError, match="missing"):
        policy.assert_msa_inputs("trn1_A", {"uniref90_hits.a3m": None})
    original.write_text(original.read_text() + ">added_row\n" + "A" * 20 + "\n")
    with pytest.raises(ValueError, match="not bound"):
        policy.assert_msa_inputs("trn1_A", {"uniref90_hits.a3m": original})


@pytest.mark.parametrize("dependency", ["policy_path", "train_filter", "hidden_manifest", "cache", "train_msa", "checkpoint_m8"])
@pytest.mark.parametrize("alias", ["direct", "symlink", "hardlink"])
def test_output_aliases_cannot_overwrite_evidence(policy_case: dict[str, Path], tmp_path: Path, dependency: str, alias: str) -> None:
    policy = load_msa_filter_policy(policy_case["policy_path"])
    if dependency == "checkpoint_m8":
        receipt = json.loads((tmp_path / "train-checkpoint" / "complete.json").read_text())
        original = Path(receipt["jobs"][0]["m8_path"])
    else:
        original = policy_case[dependency]
    output = original
    if alias != "direct":
        output = tmp_path / "output.json"
        output.symlink_to(original) if alias == "symlink" else output.hardlink_to(original)
    before = original.read_bytes()
    with pytest.raises(ValueError, match="immutable"):
        policy.validate_output_paths([output])
    assert original.read_bytes() == before


@pytest.mark.parametrize("alias", ["symlink", "hardlink"])
def test_existing_output_metadata_alias_is_rejected_before_writing(policy_case: dict[str, Path], tmp_path: Path, alias: str) -> None:
    policy = load_msa_filter_policy(policy_case["policy_path"])
    output = tmp_path / "features"
    output.mkdir()
    target = output / "preprocess_meta.json"
    target.symlink_to(policy_case["hidden_manifest"]) if alias == "symlink" else target.hardlink_to(policy_case["hidden_manifest"])
    with pytest.raises(ValueError, match="aliases"):
        policy.assert_output_roots_safe([output])


def test_scalar_loader_preserves_exploratory_usage_with_consistent_resolved_bytes(tmp_path: Path) -> None:
    original = tmp_path / "original.json"
    original.write_text(json.dumps({"excluded_sequence_sha256": ["A" * 64, "a" * 64]}))
    alias = tmp_path / "alias.json"
    alias.symlink_to(original)
    loaded = load_msa_row_filter(alias)
    assert loaded.path == original
    assert loaded.sha256 == digest_file(original)
    assert loaded.excluded_sequence_sha256 == {"a" * 64}


@pytest.mark.parametrize("mutation", ["hidden", "path", "bool_count", "extra_root", "uppercase_digest"])
def test_public_provenance_rejects_private_or_malformed_fields(policy_case: dict[str, Path], mutation: str) -> None:
    metadata = load_msa_filter_policy(policy_case["policy_path"]).provenance
    if mutation == "hidden":
        metadata["splits"]["hidden"] = metadata["splits"]["val"]
    elif mutation == "path":
        metadata["splits"]["train"]["path"] = str(policy_case["train_manifest"])
    elif mutation == "bool_count":
        metadata["splits"]["train"]["chain_count"] = True
    elif mutation == "extra_root":
        metadata["private"] = "hidden"
    else:
        metadata["policy_sha256"] = "A" * 64
    with pytest.raises(ValueError):
        validate_msa_filter_policy_provenance(metadata)
