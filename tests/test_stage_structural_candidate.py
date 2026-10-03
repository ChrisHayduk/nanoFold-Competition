from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from nanofold import revised_parent_leakage
from nanofold.representative_eligibility import file_record
from scripts import stage_structural_candidate as stage


def write(path: Path, value: Any) -> Path:
    path.write_text(json.dumps(value))
    return path


def context_fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path, dict[str, Any], dict[str, Any]]:
    rows: list[dict[str, Any]] = [dict(chain_id=chain, sequence=letter * length, length=length, resolution=resolution,
                 release_date=None, group_id=group, cluster40_id=group)
            for chain, letter, length, resolution, group in (
                ("aaaa_A", "A", 600, 2., "ga"), ("bbbb_A", "C", 80, 999., "gb"),
                ("vvvv_A", "D", 40, 1., None), ("hhhh_A", "E", 40, 1., None), ("rrrr_A", "G", 40, 1., None))]
    metadata = {"chains": [dict(row, accepted=True, cath_domains=[],
                                sequence_sha256=hashlib.sha256(row["sequence"].encode()).hexdigest()) for row in rows]}
    frozen_val, frozen_hidden = tmp_path / "val.txt", tmp_path / "hidden.txt"
    frozen_val.write_text("vvvv_A\n")
    frozen_hidden.write_text("hhhh_A\n")
    historical_public = tmp_path / "historical-public.txt"
    historical_public.write_text("rrrr_A\n")
    baseline_frame = write(tmp_path / "baseline-frame.json", {})
    repair_selection = write(tmp_path / "repair-selection.json", {"policy": "source_qualified_public_heldout_repair_v1"})
    cache = write(tmp_path / "cache.json", {})
    frame_receipt = write(tmp_path / "frame.json", {})
    frame = {"policy": stage.BARRIER_POLICY, "records": rows, "eligible_training_ids": ["aaaa_A", "bbbb_A"],
             "frozen_splits": {"val": ["vvvv_A"], "hidden_val": ["hhhh_A"]},
             "evaluation_cohorts": {"historical_public": ["rrrr_A"], "active_public": ["vvvv_A"], "hidden": ["hhhh_A"]},
             "diagnostics": {"active_public_hidden_symmetric_boundary_passed": True},
             "source_bindings": {"historical_public": file_record(historical_public),
                                 "baseline_frame_receipt": file_record(baseline_frame), "selection_receipt": file_record(repair_selection),
                                 "chain_data_cache": file_record(cache), "frozen_val": file_record(frozen_val),
                                 "frozen_hidden": file_record(frozen_hidden)}}
    plans = [{"sequence": row["sequence"], "group_id": row["group_id"], "ranked_copies": [row]} for row in rows[:2]]
    sequence_plan = tmp_path / "plan.jsonl"
    sequence_plan.write_text("".join(json.dumps(row) + "\n" for row in plans))
    qualified = tmp_path / "qualified.jsonl"
    qualified.write_text("".join(json.dumps(row) + "\n" for row in rows[:2]))
    plan = write(tmp_path / "plan.json", {"seed": 0, "inputs": [file_record(frame_receipt)]})
    population = {"plan": file_record(plan), "sequence_plan": file_record(sequence_plan), "qualified_chains": file_record(qualified)}
    population_path = write(tmp_path / "population.json", population)
    meta_path, meta_receipt, source_lock = [write(tmp_path / name, {}) for name in ("meta.json", "meta-receipt.json", "sources.json")]
    runtime = {"full_frame_receipt": str(frame_receipt), "population_receipt": str(population_path),
               "metadata": {"path": str(meta_path), "receipt": str(meta_receipt), "chain_data_cache": str(cache),
                            "metadata_sources_dir": str(tmp_path), "metadata_source_lock": str(source_lock)}}
    config = write(tmp_path / "runtime.json", runtime)
    monkeypatch.setattr(revised_parent_leakage, "validate_parent_frame", lambda _: frame)
    monkeypatch.setattr(stage, "validate_full_source_metadata", lambda **_: metadata)
    monkeypatch.setattr(stage, "validate_receipt", lambda *_args, **_kwargs: population)
    read_protocol = stage.read_protocol

    def small_protocol(path: Path) -> dict[str, Any]:
        protocol = read_protocol(path)
        protocol["evaluation_cohorts"]["target_count_per_cohort"] = 1
        return protocol

    monkeypatch.setattr(stage, "read_protocol", small_protocol)
    return config, stage.REPO_ROOT / "configs/structural-sampling.yaml", frame, metadata


def test_full_parent_driver_joins_authoritative_sources_and_correct_frozen_binding_keys(tmp_path, monkeypatch):
    config, protocol, _, _ = context_fixture(tmp_path, monkeypatch)
    context = stage.load_context(config, protocol, verify_sources=True)
    assert [row["length"] for row in context["records"]] == [600, 80]
    assert context["records"][1]["resolution"] is None
    assert set(context["frozen_paths"]) == {"val", "hidden_val"}
    assert context["pins"]["historical_public"]["sha256"] == file_record(tmp_path / "historical-public.txt")["sha256"]
    assert context["pins"]["frozen_val"]["sha256"] == file_record(tmp_path / "val.txt")["sha256"]


def test_driver_rejects_frame_from_another_cache_even_when_sequences_match(tmp_path, monkeypatch):
    config, protocol, frame, _ = context_fixture(tmp_path, monkeypatch)
    frame["source_bindings"]["chain_data_cache"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="different source caches"):
        stage.load_context(config, protocol, verify_sources=True)


def test_driver_rejects_omitted_metadata_eligible_parent(tmp_path, monkeypatch):
    config, protocol, frame, _ = context_fixture(tmp_path, monkeypatch)
    frame["records"].pop()
    with pytest.raises(ValueError, match="exactly metadata-eligible parents"):
        stage.load_context(config, protocol, verify_sources=True)


@pytest.mark.parametrize("mutation", ["exposure", "population", "hidden_count"])
def test_rehashed_source_lock_cannot_change_verified_contract(tmp_path, monkeypatch, mutation):
    config, protocol, _, _ = context_fixture(tmp_path, monkeypatch)
    context = stage.load_context(config, protocol, verify_sources=True)
    # Keep the real small allocator and stage/verify flow; fixture input validators
    # already isolate the expensive native source preparation from this boundary.
    context["protocol"]["selection"]["size"] = 2
    monkeypatch.setattr(stage, "load_context", lambda *_args, **_kwargs: context)
    candidate = tmp_path / "candidate"
    status_path = stage.stage_candidate(config, protocol, candidate)
    assert stage.verify_stage(candidate)["verification"]["training_parent_count"] == 2
    public_path, private_path = candidate / "public-source.lock.json", candidate / "private-source.lock.json"
    public, private = json.loads(public_path.read_text()), json.loads(private_path.read_text())
    if mutation == "exposure":
        public["training_exposure"]["parent_sampling"] = "nonuniform"
    elif mutation == "population":
        public["population_receipt"]["sha256"] = "0" * 64
    else:
        private["hidden_val_count"] += 1
    write(public_path, public)
    private["source_lock_sha256"] = file_record(public_path)["sha256"]
    write(private_path, private)
    status = json.loads(status_path.read_text())
    status["outputs"]["public_source_lock"] = file_record(public_path)
    status["outputs"]["private_source_lock"] = file_record(private_path)
    write(status_path, status)
    with pytest.raises(ValueError, match="complete verified contract"):
        stage.verify_stage(candidate)


def test_driver_rejects_active_and_historical_cohort_role_confusion(tmp_path, monkeypatch):
    config, protocol, frame, _ = context_fixture(tmp_path, monkeypatch)
    frame["evaluation_cohorts"]["active_public"] = ["rrrr_A"]
    with pytest.raises(ValueError, match="differ from active staged splits"):
        stage.load_context(config, protocol, verify_sources=True)


def test_driver_requires_both_directions_of_repaired_public_hidden_barrier(tmp_path, monkeypatch):
    config, protocol, frame, _ = context_fixture(tmp_path, monkeypatch)
    frame["diagnostics"]["active_public_hidden_symmetric_boundary_passed"] = False
    with pytest.raises(ValueError, match="symmetric native boundary proof"):
        stage.load_context(config, protocol, verify_sources=True)
