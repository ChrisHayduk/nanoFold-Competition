from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from scripts import analyze_crop_embeddings as engine
from scripts import analyze_structural_sampling as structural

METHODS = ["uniform_sequence", "cluster40_uniform_sequence", "group_uniform_distinct"]


def rows(n: int = 54, *, revised: bool = False) -> list[dict]:
    result = []
    for index in range(n):
        is_eval = index >= n - (4 if revised else 2)
        roles = ["public_val"] if index >= n - 2 else ["train"]
        if revised and is_eval:
            roles = (["hidden"] if index == n - 4 else ["public_val"] if index == n - 3
                     else ["public_val", "historical_public"] if index == n - 2 else ["historical_public"])
        alphabet = "ACDEFGHIKLMNPQRSTVWY"
        sequence = alphabet[index // 20] + alphabet[index % 20] + "A" * 254
        parent = hashlib.sha256(sequence.encode()).hexdigest()
        result.append({"parent_sequence_sha256": parent, "crop_sequence_sha256": parent, "sequence": sequence,
                       "vector_record_id": f"v{index}", "chain_id": f"chain{index}", "crop_seed": 0,
                       "length": 256, "source_chain_count": 1, "resolution": 2., "release_date": "2020-01-01",
                       "group_id": f"g{index // 2}", "cluster40_id": f"c{index}",
                       "pool_roles": roles,
                       "source_qualified": not is_eval, "training_owned": not is_eval, "evaluation_owned": is_eval})
    return result


def memberships(frame: list[dict]) -> list[dict]:
    def arm(name: str, kind: str, method: str, seed: int | None, indices: list[int]) -> dict:
        return {"arm_id": name, "kind": kind, "method": method, "seed": seed,
                "primary_compatible": kind != "historical",
                "chains": [frame[index]["chain_id"] for index in indices],
                "parent_sequence_sha256": [frame[index]["parent_sequence_sha256"] for index in indices]}
    result = [arm("candidate07", "candidate", "candidate07", None, [0, 1, 2]),
              arm("candidate06", "historical", "candidate06", None, [0, 1, len(frame) - 1])]
    control_frame = [{**row, "sequence_sha1": hashlib.sha1(row["sequence"].encode()).hexdigest()}
                     for row in sorted(frame, key=lambda row: row["chain_id"]) if row["training_owned"]]
    controls, _ = engine.draw_controls(control_frame, 3, [0, 1], 20261003, group_policy="soft_diversity")
    lookup = {row["chain_id"]: index for index, row in enumerate(frame)}
    for control in controls:
        indices = [lookup[control_frame[index]["chain_id"]] for index in control["indices"]]
        result.append(arm(control["name"], "control", control["method"], control["seed"], indices))
    return result


def query_document(frame: list[dict]) -> dict:
    parents = [row["parent_sequence_sha256"] for row in frame]
    public = [row["parent_sequence_sha256"] for row in frame if "public_val" in row["pool_roles"]]
    result = {"cohorts": [
        {"name": "public_val", "semantics": "heldout", "expected_count": len(public), "parent_sequence_sha256": public},
        {"name": "reference", "semantics": "full_reference", "expected_count": len(frame), "parent_sequence_sha256": parents},
        {"name": "reference_leave_parent_out", "semantics": "leave_parent_out", "expected_count": len(frame),
         "parent_sequence_sha256": parents},
        {"name": "qualified_training_reference", "semantics": "qualified_training",
         "expected_count": sum(row["source_qualified"] and row["training_owned"] for row in frame),
         "parent_sequence_sha256": [row["parent_sequence_sha256"] for row in frame if row["source_qualified"] and row["training_owned"]]}]}
    historical = [row["parent_sequence_sha256"] for row in frame if "historical_public" in row["pool_roles"]]
    if historical:
        result["cohorts"].append({"name": "historical_public", "semantics": "historical_public",
                                  "expected_count": len(historical), "parent_sequence_sha256": historical})
        preserved = sorted(set(public) & set(historical))
        result["cohorts"].append({"name": "preserved_public", "semantics": "preserved_public",
                                  "expected_count": len(preserved), "parent_sequence_sha256": preserved})
    return result


def validate(document: list[dict], frame: list[dict], **kwargs: Any) -> tuple:
    return engine.validate_arms(document, frame, 3, METHODS, [0, 1], "candidate07", **kwargs)


def fixture_config(tmp_path: Path, *, revised: bool = False) -> Path:
    frame = rows(revised=revised)
    files = {
        "reference_rows": tmp_path / "reference.jsonl", "embeddings": tmp_path / "embeddings.npz",
        "memberships": tmp_path / "memberships.json", "query_cohorts": tmp_path / "queries.json",
        "protocol": tmp_path / "protocol.json", "barrier_receipt": tmp_path / "barrier.json",
        "qualification_receipt": tmp_path / "qualification.json",
        "inference_validator": tmp_path / "validate_full.py", "inference_completion": tmp_path / "scheduler.tsv",
        "candidate06": tmp_path / "candidate06.txt", "candidate06_rows": tmp_path / "candidate06.jsonl",
        "candidate06_requalification": tmp_path / "candidate06-requalification.json",
        "embedding_provenance": tmp_path / "provenance.json"}
    files["reference_rows"].write_text("".join(json.dumps(row) + "\n" for row in frame))
    vectors = np.random.default_rng(42).normal(size=(len(frame), 2560)).astype(np.float32)
    np.savez(files["embeddings"], ids=np.array([row["crop_sequence_sha256"] for row in frame]), embeddings=vectors)
    files["inference_validator"].write_text("# Mocked native producer validator in adapter tests\n")
    files["inference_completion"].write_text("123|COMPLETED|00:01|0:0\n")
    for seed in (0, 1, 2):
        mapping = [{"parent_sequence_sha256": row["parent_sequence_sha256"],
                    "crop_sequence_sha256": row["crop_sequence_sha256"], "source_vector_record_id": row["crop_sequence_sha256"],
                    "crop_seed": seed, "parent_length": 256, "start": 0, "stop": 256} for row in frame]
        (tmp_path / f"mapping-seed-{seed}.jsonl").write_text("".join(json.dumps(row) + "\n" for row in mapping))
    files["memberships"].write_text(json.dumps(memberships(frame)))
    baseline = memberships(frame)[1]
    files["candidate06"].write_text("\n".join(baseline["chains"]) + "\n")
    files["candidate06_rows"].write_text("".join(json.dumps(row) + "\n" for row in frame if row["chain_id"] in baseline["chains"]))
    files["candidate06_requalification"].write_text('{"status":"complete"}')
    files["query_cohorts"].write_text(json.dumps(query_document(frame)))
    protocol = {"target_size": 3, "seeds": [0, 1], "seed_namespace": 20261003,
                "expanded_parent_crop_amendment": {"crop_seed": 0, "crop_size": 256}}
    if revised:
        protocol["repaired_public_amendment"] = {"policy": engine.REVISED_FRAME_POLICY}
    files["protocol"].write_text(json.dumps(protocol))
    native = [{**row, "sequence_sha256": row["parent_sequence_sha256"],
               "training_exclusion_reasons": [] if row["training_owned"] else ["exact_heldout_sequence"]} for row in frame]
    public_ids = [row["chain_id"] for row in frame if "public_val" in row["pool_roles"]]
    hidden_ids = [row["chain_id"] for row in frame if "hidden" in row["pool_roles"]]
    native_frame = {"status": "complete", "records": native,
        "eligible_training_ids": [row["chain_id"] for row in frame if row["training_owned"]],
        "frozen_splits": {"val": public_ids, "hidden_val": hidden_ids}}
    if revised:
        native_frame.update(policy=engine.REVISED_FRAME_POLICY,
                            evaluation_cohorts={"historical_public": [row["chain_id"] for row in frame if "historical_public" in row["pool_roles"]],
                                                "active_public": public_ids, "hidden": hidden_ids})
    files["barrier_receipt"].write_text(json.dumps(native_frame))
    qualified = tmp_path / "qualified.jsonl"
    qualified.write_text("".join(json.dumps(row) + "\n" for row in native if row["source_qualified"]))
    sequence_plan = tmp_path / "sequence-plan.jsonl"
    native_frame = json.loads(files["barrier_receipt"].read_text())
    planned = structural.sequence_plan(structural.training_copies(native_frame), rank=structural.copy_rank)
    sequence_plan.write_text("".join(json.dumps(row) + "\n" for row in planned))
    plan = tmp_path / "population-plan.json"
    plan.write_text(json.dumps({"full_frame_receipt": engine.pin(files["barrier_receipt"])}))
    files["qualification_receipt"].write_text(json.dumps({"status": "complete", "qualified_chains": engine.pin(qualified),
                                                         "sequence_plan": engine.pin(sequence_plan), "plan": engine.pin(plan)}))
    files["embedding_provenance"].write_text(json.dumps({
        "model_id": "esmc-test", "model_revision": "test-revision", "pooling": "post_layer_norm_mean",
        "dimensions": 2560, "crop_seed": 0,
        "reference_rows_sha256": engine.digest(files["reference_rows"]),
        "embeddings_sha256": engine.digest(files["embeddings"]),
        "barrier_receipt_sha256": engine.digest(files["barrier_receipt"])}))
    config = {"schema_version": 1, "pilot": True, "crop_seed": 0, "arm_size": 3,
              "control_methods": METHODS, "control_seeds": [0, 1],
              "candidate_arm_id": "candidate07", "batch_size": 17,
              **{key: engine.pin(path) for key, path in files.items()}}
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    return path



@pytest.fixture
def native_receipts(monkeypatch: pytest.MonkeyPatch) -> None:
    # Native receipt replay has its own integration tests. These small frames
    # isolate this adapter's binding checks after a trusted replay returns.
    monkeypatch.setattr(engine, "validate_parent_frame", lambda path: {**json.loads(path.read_text()), "receipt": engine.pin(path)})
    monkeypatch.setattr(engine, "validate_receipt", lambda path: json.loads(path.read_text()))
    def inference(paths: dict, protocol_sha: str) -> dict:
        root = paths["embedding_provenance"].parent
        return {"mappings": {str(seed): engine.pin(root / f"mapping-seed-{seed}.jsonl") for seed in (0, 1, 2)},
                "verification": {"status": "fixture_trusted_native_replay"}}
    monkeypatch.setattr(engine, "validate_inference", inference)
    monkeypatch.setattr(engine, "validate_historical_baseline", lambda paths, protocol: {})
    monkeypatch.setattr(engine, "baseline_requalification_rows", lambda paths, size: [json.loads(line) for line in paths["candidate06_rows"].read_text().splitlines()])


def test_exact_parent_exclusion_keeps_zero_distance_other_parent() -> None:
    parents = np.array(["a", "a", "b", "c"])
    distances = np.array([[0., .03, 0., .6], [.2, .1, .5, 0.]], dtype=np.float32)
    queries = np.array(["a", "c"])
    excluded = engine.exclude_parents(distances, queries, parents)
    assert np.isinf(excluded[0, :2]).all()
    assert excluded[0, 2] == 0
    assert np.isinf(excluded[1, 3])
    np.testing.assert_allclose(engine.local_radii(distances, queries, parents, (1, 2)), [[0., .6], [.1, .2]])
    assert np.isfinite(distances).all()
    with pytest.raises(ValueError, match="Too few"):
        engine.local_radii(distances[:1], queries[:1], parents, (3,))


def test_arm_shortlist_and_exhaustive_fallback_are_exact() -> None:
    distances = np.array([[0., .1, .2, .2, .8, .9], [.7, .5, 0., .1, .2, np.inf]], dtype=np.float32)
    selected = np.array([[False, True, True, False, False, False],
                         [False, False, False, False, True, True]])
    actual, fallbacks = engine.arm_nearest(distances, selected, shortlist=1)
    expected = np.stack([np.min(distances[:, mask], axis=1) for mask in selected], axis=1)
    np.testing.assert_array_equal(actual, expected)
    assert fallbacks == 3
    distances[0, selected[1]] = np.inf
    with pytest.raises(ValueError, match="no remaining member"):
        engine.arm_nearest(distances, selected)


def test_chunked_engine_matches_dense_parent_aware_reference() -> None:
    rng = np.random.default_rng(51)
    vectors = engine.normalize(rng.normal(size=(61, 7)).astype(np.float32))
    vectors[1] = vectors[0]
    parents = np.array([f"p{i}" for i in range(len(vectors))])
    selected = np.zeros((2, len(vectors)), dtype=bool)
    selected[0, [0, 1, 4, 10]] = True
    selected[1, [2, 8, 20, 55]] = True
    nearest, leave_out, radii, receipt = engine.compute(vectors, parents, selected, batch_size=13)
    dense = engine.checked_distances(vectors @ vectors.T)
    np.fill_diagonal(dense, 0.)
    excluded = engine.exclude_parents(dense, parents, parents)
    np.testing.assert_allclose(nearest, np.stack([dense[:, mask].min(axis=1) for mask in selected], axis=1), atol=2e-7)
    np.testing.assert_allclose(leave_out, np.stack([excluded[:, mask].min(axis=1) for mask in selected], axis=1), atol=2e-7)
    np.testing.assert_allclose(radii, np.sort(excluded, axis=1)[:, [9, 49]], atol=2e-7)
    assert leave_out[0, 0] < 2e-7  # The other exact parent remains despite an identical crop/vector.
    assert receipt["all_queries_resolved"] is True
    with pytest.raises(ValueError, match="128"):
        engine.compute(vectors, parents, selected, batch_size=129)


def test_rows_reject_duplicate_parent_cross_seed_and_string_ownership() -> None:
    frame = rows()
    engine.validate_rows(frame, 0)
    duplicate = copy.deepcopy(frame)
    duplicate[1]["parent_sequence_sha256"] = duplicate[0]["parent_sequence_sha256"]
    with pytest.raises(ValueError, match="one crop per parent"):
        engine.validate_rows(duplicate, 0)
    frame[0]["crop_seed"] = 1
    with pytest.raises(ValueError, match="crop seeds"):
        engine.validate_rows(frame, 0)
    frame[0]["crop_seed"] = 0
    frame[0]["training_owned"] = "false"
    with pytest.raises(ValueError, match="must be boolean"):
        engine.validate_rows(frame, 0)


def test_vector_join_rejects_missing_duplicate_extra_and_invalid_vectors() -> None:
    frame = rows(3)
    ids = np.array([row["vector_record_id"] for row in frame])
    values = np.array([[1., 0.], [0., 1.], [1., 1.]], dtype=np.float32)
    aligned = engine.align_embeddings(ids[::-1], values[::-1], frame, 2)
    np.testing.assert_allclose(aligned, engine.normalize(values))
    with pytest.raises(ValueError, match="complete reference"):
        engine.align_embeddings(ids[:-1], values[:-1], frame, 2)
    with pytest.raises(ValueError, match="Duplicate"):
        engine.align_embeddings(np.array([ids[0]] * 3), values, frame, 2)
    values[0] = 0
    with pytest.raises(ValueError, match="Zero"):
        engine.align_embeddings(ids, values, frame, 2)
    values[0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        engine.align_embeddings(ids, values, frame, 2)


def test_new_arms_enforce_directed_ownership_and_historical_is_separate() -> None:
    frame = rows()
    # Directed barriers can leave transitive group/cluster labels on both sides.
    frame[0]["group_id"] = frame[-1]["group_id"]
    frame[0]["cluster40_id"] = frame[-1]["cluster40_id"]
    arms, _ = validate(memberships(frame), frame)
    assert arms[0]["barrier_violations"] == {}
    assert arms[0]["evaluation_identity_coincidences"]["evaluation_group_id_overlap"] == 1
    assert arms[1]["primary_compatible"] is False
    assert arms[1]["barrier_violations"]["evaluation_parent_sequence_sha256_overlap"] == 1
    with pytest.raises(ValueError, match="evaluation barrier"):
        validate(memberships(frame), frame, disjointness_fields=("parent_sequence_sha256", "group_id"))
    bad = memberships(frame)
    bad[0]["chains"][-1] = frame[-1]["chain_id"]
    bad[0]["parent_sequence_sha256"][-1] = frame[-1]["parent_sequence_sha256"]
    with pytest.raises(ValueError, match="evaluation barrier"):
        validate(bad, frame)
    frame[0]["source_qualified"] = False
    with pytest.raises(ValueError, match="not_source_qualified"):
        validate(memberships(frame), frame)


def test_missing_control_membership_seed_and_unbound_chain_fail_closed() -> None:
    frame = rows()
    arms = memberships(frame)
    with pytest.raises(ValueError, match="Incomplete or duplicated"):
        validate(arms[:-1], frame)
    with pytest.raises(ValueError, match="Duplicate arm"):
        validate(arms + [arms[-1]], frame)
    arms[2]["parent_sequence_sha256"][0] = "f" * 64
    with pytest.raises(ValueError, match="no reference vector"):
        validate(arms, frame)
    arms = memberships(frame)
    arms[2]["chains"][0] = "wrong-chain"
    with pytest.raises(ValueError, match="not bound"):
        validate(arms, frame)
    arms = [arm for arm in memberships(frame) if arm["method"] != "cluster40_uniform_sequence"]
    valid, _ = validate(arms, frame, statuses={"cluster40_uniform_sequence": {"status": "unavailable", "reason": "Too few distinct clusters"}})
    assert len(valid) == 6
    with pytest.raises(ValueError, match="explicit reason"):
        validate(arms, frame, statuses={"cluster40_uniform_sequence": {"status": "unavailable"}})


def test_fixed_query_cohorts_cannot_silently_shrink_or_leak() -> None:
    frame = rows()
    query = query_document(frame)
    assert len(engine.load_cohorts(query, frame, pilot=True)) == 4
    with pytest.raises(ValueError, match="1000"):
        engine.load_cohorts(query, frame)
    query["cohorts"][1]["parent_sequence_sha256"] = query["cohorts"][1]["parent_sequence_sha256"][:-1]
    query["cohorts"][1]["expected_count"] -= 1
    with pytest.raises(ValueError, match="every parent"):
        engine.load_cohorts(query, frame, pilot=True)
    query = query_document(frame)
    query["cohorts"][0]["parent_sequence_sha256"][0] = frame[0]["parent_sequence_sha256"]
    with pytest.raises(ValueError, match="non-evaluation-owned"):
        engine.load_cohorts(query, frame, pilot=True)


def test_weighted_means_and_coverage_keep_unweighted_quantiles_and_zero_radii_explicit() -> None:
    result = engine.summarize(np.array([0., .5, 1.]), np.array([[0., 0.], [.5, .5], [.1, .2]]),
                              np.array([1., 2., 7.]), np.array([.2, .6]))
    assert result["mean_distance"] == pytest.approx(.8)
    assert result["distance_p50"] == .5
    assert result["query_local_50nn_coverage"] == pytest.approx(.3)
    assert result["fixed_global_50nn_radius_coverage"] == pytest.approx(.3)
    assert result["zero_local_50nn_radius_n"] == 1
    assert result["normalized_gap_population_n"] == 2
    assert result["normalized_gap_50_p50"] == 3.


def test_integration_pins_outputs_historical_separation_and_radius_cache(tmp_path: Path, native_receipts: None) -> None:
    config_path = fixture_config(tmp_path)
    result = engine.run(config_path, tmp_path / "first")
    assert result["status"] == "complete_pilot"
    assert result["reference_n"] == 54
    summaries = json.loads((tmp_path / "first/control_summary.json").read_text())
    assert summaries["historical_arms_excluded"] == ["candidate06"]
    assert all(row["controls_n"] == 2 for row in summaries["rows"])
    assert len(summaries["rows"]) == (3 * 2 + 3) * 3 * len(engine.METRICS)
    config = json.loads(config_path.read_text())
    config["radius_cache"] = engine.pin(tmp_path / "first/radii_cache.npz")
    config["radius_cache_contract"] = engine.pin(tmp_path / "first/radii_contract.json")
    config_path.write_text(json.dumps(config))
    engine.run(config_path, tmp_path / "cached")
    np.testing.assert_array_equal(np.load(tmp_path / "first/local_radii.npy"), np.load(tmp_path / "cached/local_radii.npy"))
    frame = rows()
    expected = json.loads((tmp_path / "first/radii_contract.json").read_text())
    expected["crop_seed"] = 1
    with pytest.raises(ValueError, match="contract differs"):
        engine.load_cached_radii(tmp_path / "first/radii_cache.npz", tmp_path / "first/radii_contract.json", expected, frame)
    config["embeddings"]["sha256"] = "0" * 64
    config_path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="Frozen artifact changed"):
        engine.run(config_path, tmp_path / "bad")
    assert not (tmp_path / "bad").exists()


def test_midrun_pin_change_cannot_receive_completion_receipt(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, native_receipts: None) -> None:
    config = fixture_config(tmp_path)
    original = engine.compute
    def changed(*args: Any, **kwargs: Any) -> tuple:
        result = original(*args, **kwargs)
        with (tmp_path / "reference.jsonl").open("a") as handle:
            handle.write("\n")
        return result
    monkeypatch.setattr(engine, "compute", changed)
    with pytest.raises(ValueError, match="changed during analysis"):
        engine.run(config, tmp_path / "mutated")
    assert not (tmp_path / "mutated/completion.json").exists()


def test_production_contract_and_protocol_mismatch_rejected_before_work(tmp_path: Path) -> None:
    path = fixture_config(tmp_path)
    config = json.loads(path.read_text())
    config["pilot"] = False
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="100 frozen seeds"):
        engine.run(path, tmp_path / "bad-count")
    config["pilot"] = True
    config["arm_size"] = 4
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="differ from frozen protocol"):
        engine.run(path, tmp_path / "bad-protocol")


def test_empty_native_barrier_receipt_cannot_certify_caller_flags(tmp_path: Path) -> None:
    path = fixture_config(tmp_path)
    config = json.loads(path.read_text())
    barrier = tmp_path / "barrier.json"
    barrier.write_text("{}")
    config["barrier_receipt"] = engine.pin(barrier)
    provenance_path = tmp_path / "provenance.json"
    provenance = json.loads(provenance_path.read_text())
    provenance["barrier_receipt_sha256"] = engine.digest(barrier)
    provenance_path.write_text(json.dumps(provenance))
    config["embedding_provenance"] = engine.pin(provenance_path)
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="Incomplete or unknown full-parent"):
        engine.run(path, tmp_path / "forged")
    assert not (tmp_path / "forged").exists()


def test_native_admission_rejects_self_asserted_flags_crops_public_ids_and_changed_controls(tmp_path: Path, native_receipts: None) -> None:
    path = fixture_config(tmp_path)
    config = json.loads(path.read_text())
    paths = {name: Path(value["path"]) for name, value in engine.load_pins(config, path.parent).items()}
    frame = rows()
    arms = memberships(frame)
    protocol = json.loads(paths["protocol"].read_text())
    assert engine.validate_admission(paths, frame, arms, protocol)["qualified_parent_n"] == 52
    for key, value, message in (("source_qualified", False, "flags differ"),
                                ("crop_sequence_sha256", "f" * 64, "crop identity"),
                                ("source_chain_count", 200, "abundance/length"),
                                ("pool_roles", ["public_val"], "active frozen")):
        bad = rows()
        bad[0][key] = value
        with pytest.raises(ValueError, match=message):
            engine.validate_admission(paths, bad, arms, protocol)
    bad_arms = memberships(rows())
    first = bad_arms[2]
    absent = next(row for row in rows()[:52] if row["chain_id"] not in first["chains"])
    first["chains"][0] = absent["chain_id"]
    first["parent_sequence_sha256"][0] = absent["parent_sequence_sha256"]
    with pytest.raises(ValueError, match="deterministic qualified-frame draws"):
        engine.validate_admission(paths, rows(), bad_arms, protocol)


def test_missing_required_query_semantics_and_qualified_denominator_fail() -> None:
    frame = rows()
    query = query_document(frame)
    query["cohorts"] = [cohort for cohort in query["cohorts"] if cohort["semantics"] != "leave_parent_out"]
    with pytest.raises(ValueError, match="prespecified full-reference"):
        engine.load_cohorts(query, frame, pilot=True)
    query = query_document(frame)
    query["cohorts"][-1]["parent_sequence_sha256"].pop()
    query["cohorts"][-1]["expected_count"] -= 1
    with pytest.raises(ValueError, match="complete qualified admitted"):
        engine.load_cohorts(query, frame, pilot=True)


def test_authorized_robustness_seed_is_separate_and_group_weighting_has_honest_scope(tmp_path: Path, native_receipts: None) -> None:
    path = fixture_config(tmp_path)
    config = json.loads(path.read_text())
    config["crop_seed"] = 1
    frame = rows()
    for row in frame:
        row["crop_seed"] = 1  # Whole short sequences are identical, but the study identity is distinct.
    (tmp_path / "reference.jsonl").write_text("".join(json.dumps(row) + "\n" for row in frame))
    protocol = json.loads((tmp_path / "protocol.json").read_text())
    protocol["expanded_parent_crop_amendment"]["authorized_full_inference"] = {"crop_seeds": [0, 1, 2], "primary_crop_seed": 0}
    (tmp_path / "protocol.json").write_text(json.dumps(protocol))
    provenance = json.loads((tmp_path / "provenance.json").read_text())
    provenance.update(crop_seed=1, reference_rows_sha256=engine.digest(tmp_path / "reference.jsonl"))
    (tmp_path / "provenance.json").write_text(json.dumps(provenance))
    for key in ("reference_rows", "protocol", "embedding_provenance"):
        config[key] = engine.pin(Path(config[key]["path"]))
    path.write_text(json.dumps(config))
    result = engine.run(path, tmp_path / "seed1")
    assert result["crop_study_role"] == "prespecified_robustness"
    metrics = json.loads((tmp_path / "seed1/metrics.json").read_text())["rows"]
    groups = [row for row in metrics if row["weighting"] == "group_uniform"]
    assert all(row["status"] == "unavailable" for row in groups if row["cohort"] != "qualified_training_reference")
    assert all(row["status"] == "complete" for row in groups if row["cohort"] == "qualified_training_reference")


def test_deduplicated_three_seed_cache_joins_all_parents_without_excluding_equal_crops() -> None:
    frame = rows(3)
    frame[1]["crop_sequence_sha256"] = frame[0]["crop_sequence_sha256"]
    mapping = [{"parent_sequence_sha256": row["parent_sequence_sha256"], "crop_sequence_sha256": row["crop_sequence_sha256"],
                "source_vector_record_id": row["crop_sequence_sha256"], "crop_seed": 0,
                "parent_length": row["length"], "start": 0, "stop": 256} for row in frame]
    ids = np.array([frame[2]["crop_sequence_sha256"], "unused-other-seed-crop", frame[0]["crop_sequence_sha256"]])
    vectors = np.random.default_rng(3).normal(size=(3, 2560)).astype(np.float16)
    result = engine.align_crop_embeddings(ids, vectors, mapping, frame, 0)
    np.testing.assert_array_equal(result[0], result[1])
    np.testing.assert_array_equal(result[2], engine.normalize(vectors[[0]])[0])
    with pytest.raises(ValueError, match="exactly the complete reference"):
        engine.align_crop_embeddings(ids, vectors, mapping[:-1], frame, 0)
    mapping[0]["crop_seed"] = 1
    with pytest.raises(ValueError, match="crop identity"):
        engine.align_crop_embeddings(ids, vectors, mapping, frame, 0)


def test_native_inference_binding_rejects_wrong_model_contract(tmp_path: Path) -> None:
    paths = {key: tmp_path / name for key, name in (("inference_validator", "validator.py"),
             ("embeddings", "embeddings.npz"), ("embedding_provenance", "provenance.json"),
             ("inference_completion", "scheduler.tsv"))}
    for key in ("embeddings", "embedding_provenance", "inference_completion"):
        paths[key].write_text("test fixture")
    proof = {"embeddings": engine.pin(paths["embeddings"]), "provenance": engine.pin(paths["embedding_provenance"]),
             "scheduler": engine.pin(paths["inference_completion"]), "contract": {"model_id": "wrong-model"}}
    paths["inference_validator"].write_text("def validate_full(root):\n    return " + repr(proof) + "\n")
    with pytest.raises(ValueError, match="frozen ESMC6B"):
        engine.validate_inference(paths, "a" * 64)


def test_compatible_original_candidate06_is_allowed_but_arbitrary_extra_candidate_is_not(tmp_path: Path, native_receipts: None) -> None:
    config_path = fixture_config(tmp_path)
    config = json.loads(config_path.read_text())
    paths = {name: Path(record["path"]) for name, record in engine.load_pins(config, tmp_path).items()}
    frame = rows()
    arms = memberships(frame)
    arms[1].update(kind="candidate", primary_compatible=True,
                   chains=arms[0]["chains"].copy(), parent_sequence_sha256=arms[0]["parent_sequence_sha256"].copy())
    paths["candidate06_rows"].write_text("".join(json.dumps(row) + "\n" for row in frame[:3]))
    paths["candidate06_requalification"].write_text(json.dumps({"status": "complete", "policy": "fixed_baseline_verified_source_replay_v1"}))
    engine.validate_admission(paths, frame, arms, json.loads(paths["protocol"].read_text()))
    validated, _ = validate(arms, frame)
    assert sum(arm["kind"] == "candidate" for arm in validated) == 2
    paths["candidate06_requalification"].write_text(json.dumps({"status": "complete", "policy": "older_combined_replay"}))
    with pytest.raises(ValueError, match="compatibility differs"):
        engine.validate_admission(paths, frame, arms, json.loads(paths["protocol"].read_text()))
    arms[1]["arm_id"] = "unfrozen-extra-candidate"
    with pytest.raises(ValueError, match="fixed candidate06"):
        validate(arms, frame)


def test_baseline_comparison_omits_incompatible_history_and_retains_compatible_delta() -> None:
    def metric(arm: str, value: float) -> dict:
        return {"arm_id": arm, "cohort": "public_val", "weighting": "parent_uniform",
                "status": "complete", "primary_compatible": True, **{key: value for key in engine.METRICS}}
    primary, baseline = metric("candidate07", .3), metric("candidate06", .4)
    compared = engine.baseline_comparison([primary, baseline], "candidate07")
    assert compared["status"] == "complete"
    assert compared["rows"][0]["candidate_minus_candidate06"] == pytest.approx(-.1)
    baseline["primary_compatible"] = False
    assert engine.baseline_comparison([primary, baseline], "candidate07")["rows"] == []


def test_valid_qualification_of_a_subset_cannot_replace_the_complete_admitted_population(tmp_path: Path, native_receipts: None) -> None:
    config_path = fixture_config(tmp_path)
    config = json.loads(config_path.read_text())
    paths = {name: Path(record["path"]) for name, record in engine.load_pins(config, tmp_path).items()}
    sequence_plan = tmp_path / "sequence-plan.jsonl"
    planned = [json.loads(line) for line in sequence_plan.read_text().splitlines()]
    sequence_plan.write_text("".join(json.dumps(row) + "\n" for row in planned[:-1]))
    with pytest.raises(ValueError, match="exact current admitted parent"):
        engine.validate_admission(paths, rows(), memberships(rows()), json.loads(paths["protocol"].read_text()))
    planned[0]["ranked_copies"] *= 2
    sequence_plan.write_text("".join(json.dumps(row) + "\n" for row in planned))
    with pytest.raises(ValueError, match="exact current admitted parent"):
        engine.validate_admission(paths, rows(), memberships(rows()), json.loads(paths["protocol"].read_text()))


def test_repaired_public_pipeline_keeps_active_and_original_denominators_separate(tmp_path: Path, native_receipts: None) -> None:
    config = fixture_config(tmp_path, revised=True)
    output = tmp_path / "repaired"
    result = engine.run(config, output)
    frame = rows(revised=True)
    assert result["admission_validation"]["repaired_public"] is True
    active = [index for index, row in enumerate(frame) if "public_val" in row["pool_roles"]]
    historical = [index for index, row in enumerate(frame) if "historical_public" in row["pool_roles"]]
    preserved = sorted(set(active) & set(historical))
    assert len(set(active) & set(historical)) == 1
    assert len(set(historical) - set(active)) == 1
    metrics = json.loads((output / "metrics.json").read_text())["rows"]
    selected = {row["cohort"]: row for row in metrics if row["arm_id"] == "candidate07" and row["weighting"] == "parent_uniform"}
    nearest = np.load(output / "nearest.npy")
    assert selected["public_val"]["mean_distance"] == pytest.approx(float(nearest[active, 0].mean()))
    assert selected["historical_public"]["mean_distance"] == pytest.approx(float(nearest[historical, 0].mean()))
    assert selected["preserved_public"]["mean_distance"] == pytest.approx(float(nearest[preserved, 0].mean()))
    assert selected["public_val"]["cohort_role"] == "primary_public"
    assert selected["historical_public"]["cohort_role"] == "historical_descriptive"
    assert selected["preserved_public"]["cohort_role"] == "preserved_descriptive"
    summaries = json.loads((output / "control_summary.json").read_text())["rows"]
    historical_summaries = [row for row in summaries if row["cohort"] == "historical_public"]
    assert historical_summaries and all(row["cohort_role"] == "historical_descriptive" for row in historical_summaries)
    baseline = json.loads((output / "baseline_comparison.json").read_text())
    assert baseline["status"] == "unavailable"
    assert baseline["exposure"]["final_frame_barrier"]["excluded_chain_n"] == 1
    assert baseline["exposure"]["cohorts"]["historical_public"]["exact_parent_overlap_n"] == 1
    assert baseline["exposure"]["cohorts"]["public_val"]["exact_parent_overlap_n"] == 0
    assert baseline["exposure"]["cohorts"]["preserved_public"]["query_parent_n"] == 1


def test_repaired_queries_cannot_omit_or_mix_the_original_and_active_cohorts() -> None:
    frame = rows(revised=True)
    query = query_document(frame)
    assert len(engine.load_cohorts(query, frame, pilot=True, repaired_public=True)) == 6
    no_historical = copy.deepcopy(query)
    no_historical["cohorts"] = [cohort for cohort in no_historical["cohorts"] if cohort["name"] != "historical_public"]
    with pytest.raises(ValueError, match="required original historical_public"):
        engine.load_cohorts(no_historical, frame, pilot=True, repaired_public=True)
    historical = query["cohorts"][-2]["parent_sequence_sha256"]
    active = query["cohorts"][0]["parent_sequence_sha256"]
    mixed = copy.deepcopy(query)
    mixed["cohorts"][0]["parent_sequence_sha256"] = historical
    with pytest.raises(ValueError, match="public_val must be the complete"):
        engine.load_cohorts(mixed, frame, pilot=True, repaired_public=True)
    mixed = copy.deepcopy(query)
    mixed["cohorts"][-2]["parent_sequence_sha256"] = active
    with pytest.raises(ValueError, match="historical_public must be the complete"):
        engine.load_cohorts(mixed, frame, pilot=True, repaired_public=True)
    no_preserved = copy.deepcopy(query)
    no_preserved["cohorts"].pop()
    with pytest.raises(ValueError, match="required common preserved_public"):
        engine.load_cohorts(no_preserved, frame, pilot=True, repaired_public=True)
    mixed = copy.deepcopy(query)
    mixed["cohorts"][-1]["parent_sequence_sha256"] = [historical[-1]]
    with pytest.raises(ValueError, match="preserved_public must be the complete"):
        engine.load_cohorts(mixed, frame, pilot=True, repaired_public=True)


def test_repaired_protocol_cannot_use_stale_frame_or_qualification_plan(tmp_path: Path, native_receipts: None) -> None:
    config_path = fixture_config(tmp_path, revised=True)
    config = json.loads(config_path.read_text())
    paths = {name: Path(record["path"]) for name, record in engine.load_pins(config, tmp_path).items()}
    protocol = json.loads(paths["protocol"].read_text())
    frame = rows(revised=True)
    arms = memberships(frame)
    no_repair = copy.deepcopy(protocol)
    no_repair.pop("repaired_public_amendment")
    with pytest.raises(ValueError, match="protocol and owning parent-frame policy"):
        engine.validate_admission(paths, frame, arms, no_repair)
    original = json.loads(paths["barrier_receipt"].read_text())
    base = copy.deepcopy(original)
    base.pop("policy")
    paths["barrier_receipt"].write_text(json.dumps(base))
    with pytest.raises(ValueError, match="protocol and owning parent-frame policy"):
        engine.validate_admission(paths, frame, arms, protocol)
    paths["barrier_receipt"].write_text(json.dumps(original))
    population_plan = tmp_path / "population-plan.json"
    plan = json.loads(population_plan.read_text())
    plan["full_frame_receipt"]["sha256"] = "0" * 64
    population_plan.write_text(json.dumps(plan))
    with pytest.raises(ValueError, match="current complete barrier frame"):
        engine.validate_admission(paths, frame, arms, protocol)


def test_retired_public_parents_remain_protected_and_cannot_enter_new_arms(tmp_path: Path, native_receipts: None) -> None:
    config_path = fixture_config(tmp_path, revised=True)
    config = json.loads(config_path.read_text())
    paths = {name: Path(record["path"]) for name, record in engine.load_pins(config, tmp_path).items()}
    protocol = json.loads(paths["protocol"].read_text())
    frame = rows(revised=True)
    arms = memberships(frame)
    retired = frame[-1]
    assert "historical_public" in retired["pool_roles"] and "public_val" not in retired["pool_roles"]
    admission = engine.validate_admission(paths, frame, arms, protocol)
    assert retired["parent_sequence_sha256"] in admission["historical_public_parents"]
    assert retired["parent_sequence_sha256"] not in admission["public_parents"]
    bad = copy.deepcopy(arms)
    bad[0]["chains"][0] = retired["chain_id"]
    bad[0]["parent_sequence_sha256"][0] = retired["parent_sequence_sha256"]
    with pytest.raises(ValueError, match="actual source-qualified representative"):
        engine.validate_admission(paths, frame, bad, protocol)
    with pytest.raises(ValueError, match="evaluation barrier"):
        validate(bad, frame)
    forged = rows(revised=True)
    forged[-1]["evaluation_owned"] = False
    with pytest.raises(ValueError, match="flags differ"):
        engine.validate_admission(paths, forged, arms, protocol)
    forged = rows(revised=True)
    forged[-1]["pool_roles"] = []
    with pytest.raises(ValueError, match="protected original public cohort"):
        engine.validate_admission(paths, forged, arms, protocol)


def test_baseline_exposure_distinguishes_chain_aliases_and_native_homology_exclusions() -> None:
    by_chain = {chain: {"sequence_sha256": parent, "training_exclusion_reasons": reasons}
                for chain, parent, reasons in [
                    ("new_public", "a", ["exact_heldout_sequence"]),
                    ("old_training_alias", "a", ["exact_heldout_sequence"]),
                    ("retained", "b", ["exact_heldout_sequence"]),
                    ("retired", "c", ["exact_heldout_sequence"]),
                    ("hidden", "d", ["exact_heldout_sequence"]),
                    ("directed_exclusion", "e", ["heldout_query_homology"])]}
    result = engine.baseline_exposure(
        {"old_training_alias": "a", "retained": "b", "directed_exclusion": "e"}, by_chain, set(),
        {"public_val": {"new_public", "retained"}, "historical_public": {"retained", "retired"},
         "preserved_public": {"retained"}, "hidden": {"hidden"}})
    assert result["cohorts"]["public_val"]["direct_chain_overlap_n"] == 1
    assert result["cohorts"]["public_val"]["exact_parent_overlap_n"] == 2
    assert result["cohorts"]["historical_public"]["exact_parent_overlap_n"] == 1
    assert result["cohorts"]["hidden"]["exact_parent_overlap_n"] == 0
    assert result["final_frame_barrier"]["excluded_chain_n"] == 3
    assert result["final_frame_barrier"]["excluded_parent_n"] == 3
    assert result["final_frame_barrier"]["reason_chain_counts"] == {
        "exact_heldout_sequence": 2, "heldout_query_homology": 1}


def test_preserved_exposure_follows_parent_intersection_when_active_chain_alias_changes(tmp_path: Path, native_receipts: None) -> None:
    config_path = fixture_config(tmp_path, revised=True)
    config = json.loads(config_path.read_text())
    paths = {name: Path(record["path"]) for name, record in engine.load_pins(config, tmp_path).items()}
    frame = rows(revised=True)
    preserved = frame[-2]
    original_chain = preserved["chain_id"]
    active_alias = "active_chain_alias"
    native = json.loads(paths["barrier_receipt"].read_text())
    alias_record = next(row for row in native["records"] if row["chain_id"] == original_chain)
    native["records"].append({**alias_record, "chain_id": active_alias})
    native["frozen_splits"]["val"] = [active_alias if chain == original_chain else chain for chain in native["frozen_splits"]["val"]]
    native["evaluation_cohorts"]["active_public"] = native["frozen_splits"]["val"]
    paths["barrier_receipt"].write_text(json.dumps(native))
    plan_path = tmp_path / "population-plan.json"
    plan_path.write_text(json.dumps({"full_frame_receipt": engine.pin(paths["barrier_receipt"])}))
    preserved["source_chain_count"] = 2
    arms = memberships(frame)
    arms[1]["chains"][-1] = original_chain
    arms[1]["parent_sequence_sha256"][-1] = preserved["parent_sequence_sha256"]
    paths["candidate06_rows"].write_text("".join(json.dumps(row) + "\n" for row in [*frame[:2], preserved]))
    admission = engine.validate_admission(paths, frame, arms, json.loads(paths["protocol"].read_text()))
    exposure = admission["baseline_exposure"]["cohorts"]["preserved_public"]
    assert admission["preserved_public_parents"] == [preserved["parent_sequence_sha256"]]
    assert exposure == {"query_chain_n": 1, "query_parent_n": 1, "direct_chain_overlap_n": 0, "exact_parent_overlap_n": 1}
