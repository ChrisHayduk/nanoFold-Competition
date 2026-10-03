from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from scripts import analyze_structural_sampling as analyzer
from scripts.analyze_structural_sampling import (
    canonical_row,
    compute_nearest_arrays,
    describe_arm,
    digest,
    distributions,
    divergence,
    draw_controls,
    independent_support,
    metric_vector,
    pin,
    run,
    tokens,
    validate_cohorts,
    validate_historical_arrays,
    weights,
)


def protocol() -> dict:
    return {"target_size": 3, "seeds": [0, 1], "seed_namespace": 20261003,
            "weighting": ["sequence_uniform", "group_uniform"],
            "chain_length_bins": [0, 64, 128, 1000], "domain_length_bins": [0, 64, 128, 1000],
            "resolution_bins": [0, 1.5, 2, 3, 1000]}


def rows() -> list[dict]:
    result = []
    for index, sequence in enumerate(("A", "C", "D", "E", "F", "G")):
        domain = {"domain_id": str(index), "topology_id": "1.1.1" if index < 4 else "2.1.1",
                  "homologous_superfamily_id": "1.1.1.1" if index < 4 else "2.1.1.1",
                  "domain_length": 20, "boundary_status": "verified"}
        row = canonical_row({"chain_id": f"{index}aaa_A", "group_id": f"g{index // 2}",
                             "cluster40_id": f"c{index}", "sequence": sequence * 40,
                             "resolution": 2., "cath_domains": [domain]}, {})
        row["_tokens"] = tokens(row, protocol())
        result.append(row)
    return result


def test_controls_enforce_unique_frame_and_never_silently_backfill() -> None:
    frame = rows()
    arms, unavailable = draw_controls(frame, 3, [0, 1], 42)
    assert unavailable == {}
    assert arms == draw_controls(frame, 3, [0, 1], 42)[0]
    assert len(arms) == 6
    for arm in arms:
        assert len(set(arm["indices"])) == 3
        if arm["method"] == "group_cover_random_sequence":
            assert len({frame[i]["group_id"] for i in arm["indices"]}) == 3
    for row in frame:
        row["cluster40_id"] = "one"
    arms, unavailable = draw_controls(frame, 3, [0], 42)
    assert set(unavailable) == {"cluster40_uniform_sequence"}
    assert len(arms) == 2
    frame[0]["sequence_sha1"] = frame[1]["sequence_sha1"]
    with pytest.raises(ValueError, match="distinct exact sequences"):
        draw_controls(frame, 3, [0], 42)


def test_domain_multiplicity_unknowns_and_group_support_have_distinct_meanings() -> None:
    frame = rows()
    repeated = copy.deepcopy(frame[0])
    repeated["cath_domains"] *= 2
    repeated["_tokens"] = tokens(repeated, protocol())
    missing = copy.deepcopy(frame[4])
    missing["cath_domains"] = []
    missing["_tokens"] = tokens(missing, protocol())
    distribution = distributions([repeated, missing], protocol(), "sequence_uniform")
    assert distribution["domain_topology"] == {"1.1.1": pytest.approx(2 / 3), "__unknown__": pytest.approx(1 / 3)}
    assert distribution["topology_x_domain_count"]["1.1.1|2"] == pytest.approx(2 / 3)
    assert independent_support([repeated, frame[1]], "topology_id") == {"1.1.1": 1}
    assert divergence({"a": 1}, {"b": 1}) == (1., 1.)
    assert divergence({"a": 1}, {"a": 1}) == (0., 0.)


def test_macro_group_weights_and_known_mass_are_not_chain_frequency() -> None:
    frame = rows()[:3]
    assert weights(frame, "group_uniform").tolist() == [.25, .25, .5]
    reference = rows()
    refdist = {mode: distributions(reference, protocol(), mode) for mode in protocol()["weighting"]}
    support = {"topology": independent_support(reference, "topology_id"),
               "superfamily": independent_support(reference, "homologous_superfamily_id")}
    metrics = describe_arm(frame, reference, protocol(), refdist, support)
    assert metrics["sequence_uniform.topology.coverage_at_1_groups"] == .5
    assert metrics["sequence_uniform.topology.missing_known_mass_at_1_groups"] == pytest.approx(1 / 3)
    assert metrics["sequence_uniform.topology.attainable_categories_at_2_groups"] == 1
    assert metrics["sequence_uniform.topology.coverage_at_2_groups"] == 1
    assert metrics["sequence_uniform.topology.coverage_at_5_groups"] is None


def test_distance_coverage_uses_common_denominator_and_explicit_unweighted_quantiles() -> None:
    distances = np.array([.1, .2, .3])
    radii = np.array([[.1, .15], [.15, .25], [.25, .4]])
    result = metric_vector(distances, radii, np.array([.15, .25]), np.array([1., 2., 1.]))
    assert result["mean_distance"] == pytest.approx(.2)
    assert result["local50_coverage"] == 1
    assert result["global50_coverage"] == .75
    with pytest.raises(ValueError, match="denominator"):
        metric_vector(distances, radii[:2], np.array([.15, .25]), np.ones(3))


def test_sequence_identity_corruption_is_rejected() -> None:
    row = {"chain_id": "1aaa_A", "group_id": "g", "cluster40_id": "c", "sequence": "A" * 40,
           "sequence_sha256": "incorrect", "resolution": 2}
    with pytest.raises(ValueError, match="hashes"):
        canonical_row(row, {})


def fixture_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    frame = [{k: v for k, v in row.items() if k != "_tokens"} for row in rows()]
    inputs = {}
    values = {"qualified_rows": frame, "historical_source_rows": frame,
              "structure_metadata": [{"chain_id": row["chain_id"], "cath_domains": row["cath_domains"]} for row in frame],
              }
    for name, value in values.items():
        path = tmp_path / (f"{name}.jsonl" if name == "qualified_rows" else f"{name}.json")
        path.write_text("".join(json.dumps(row) + "\n" for row in value) if name == "qualified_rows" else json.dumps(value))
        inputs[name] = pin(path)
    # Qualification replay is tested by the source-population suite. Keep this
    # integration fixture focused on the consumer's binding/leakage/metric path.
    qualified_record = inputs["qualified_rows"]
    receipt = {"status": "complete", "qualified_chains": qualified_record}
    receipt_path = tmp_path / "receipt.json"
    receipt_path.write_text(json.dumps(receipt))
    inputs["qualification_receipt"] = pin(receipt_path)
    monkeypatch.setattr(analyzer, "validate_receipt", lambda path: json.loads(path.read_text()))
    for name, members in {"candidate06": [frame[i]["chain_id"] for i in (0, 2, 4)],
                          "candidate07": [frame[i]["chain_id"] for i in (1, 3, 5)],
                          "fixed_eval_groups": ["held_out"]}.items():
        path = tmp_path / f"{name}.txt"
        path.write_text("\n".join(members) + "\n")
        inputs[name] = pin(path)
    configuration, frozen = tmp_path / "config.json", tmp_path / "protocol.json"
    configuration.write_text(json.dumps({"inputs": inputs}))
    historical = tmp_path / "historical-protocol.json"
    historical.write_text(json.dumps({"inputs": {"nanofold_train": inputs["candidate06"]}}))
    frozen_protocol = protocol()
    frozen_protocol["historical_controls"] = {"protocol_path": str(historical), "protocol_sha256": digest(historical)}
    frozen.write_text(json.dumps(frozen_protocol))
    return configuration, frozen


def test_end_to_end_records_memberships_before_metrics_and_preserves_inputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config, frozen = fixture_config(tmp_path, monkeypatch)
    before = digest(config)
    out = tmp_path / "comparison"
    run(config, frozen, digest(frozen), out, skip_embeddings=True)
    assert digest(config) == before
    assert json.loads((out / "completion.json").read_text())["arms"] == 8
    assert json.loads((out / "completion.json").read_text())["embeddings_evaluated"] is False
    members = json.loads((out / "memberships.json").read_text())
    assert len(members) == 8 and all(len(arm["chains"]) == 3 for arm in members)
    with pytest.raises(FileExistsError):
        run(config, frozen, digest(frozen), out, skip_embeddings=True)
    values = json.loads(config.read_text())
    Path(values["inputs"]["candidate07"]["path"]).write_text("modified\n")
    with pytest.raises(ValueError, match="frozen identity"):
        run(config, frozen, digest(frozen), tmp_path / "other", skip_embeddings=True)


def test_evaluation_ownership_leak_fails_before_output(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config, frozen = fixture_config(tmp_path, monkeypatch)
    values = json.loads(config.read_text())
    eval_groups = Path(values["inputs"]["fixed_eval_groups"]["path"])
    eval_groups.write_text("g0\n")
    values["inputs"]["fixed_eval_groups"] = pin(eval_groups)
    config.write_text(json.dumps(values))
    with pytest.raises(ValueError, match="evaluation groups"):
        run(config, frozen, digest(frozen), tmp_path / "bad", skip_embeddings=True)
    assert not (tmp_path / "bad").exists()


def test_missing_selected_group_is_rejected_without_refilling(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config, frozen = fixture_config(tmp_path, monkeypatch)
    settings = json.loads(config.read_text())
    selected = Path(settings["inputs"]["candidate07"]["path"])
    selected.write_text("0aaa_A\n1aaa_A\n2aaa_A\n")
    settings["inputs"]["candidate07"] = pin(selected)
    config.write_text(json.dumps(settings))
    with pytest.raises(ValueError, match="exact required training group set"):
        run(config, frozen, digest(frozen), tmp_path / "bad", skip_embeddings=True)
    assert not (tmp_path / "bad").exists()


def test_receipt_is_not_accepted_on_status_label_alone(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config, frozen = fixture_config(tmp_path, monkeypatch)
    monkeypatch.undo()
    with pytest.raises(ValueError, match="not complete"):
        run(config, frozen, digest(frozen), tmp_path / "bad", skip_embeddings=True)


def test_input_mutation_during_metrics_prevents_completion(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config, frozen = fixture_config(tmp_path, monkeypatch)
    original = analyzer.describe_arm

    def mutate_input(*args: object, **kwargs: object) -> dict:
        config.write_text(config.read_text() + " ")
        return original(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(analyzer, "describe_arm", mutate_input)
    out = tmp_path / "partial"
    with pytest.raises(ValueError, match="frozen identity"):
        run(config, frozen, digest(frozen), out, skip_embeddings=True)
    assert (out / "metadata-metrics.json").exists()
    assert not (out / "completion.json").exists()


def test_blocked_nearest_glue_preserves_self_exclusion_and_exact_minima() -> None:
    from types import SimpleNamespace

    calls = []

    class Engine:
        def __init__(self, backend: str, reference: np.ndarray, pool: np.ndarray):
            assert backend == "numpy" and np.array_equal(reference, pool)
            self.reference = reference

        def similarities(self, query: np.ndarray, which: str) -> np.ndarray:
            assert which == "reference"
            calls.append(len(query))
            return query @ self.reference.T

    def brute_nearest(distances: np.ndarray, selected: np.ndarray) -> tuple[np.ndarray, int]:
        return np.stack([distances[:, mask].min(axis=1) for mask in selected], axis=1), 0

    helper = SimpleNamespace(MatrixEngine=Engine, checked_distances=lambda similarity: np.clip(1 - similarity, 0, 2),
                             arm_nearest=brute_nearest)
    rng = np.random.default_rng(7)
    vectors = rng.standard_normal((131, 4)).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    masks = np.zeros((2, 131), dtype=bool)
    masks[0, :20] = True
    masks[1, 100:] = True
    nearest, leave_self = compute_nearest_arrays(helper, vectors, masks)
    assert calls == [128, 3]
    brute = np.clip(1 - vectors @ vectors.T, 0, 2)
    np.fill_diagonal(brute, 0)
    np.testing.assert_allclose(nearest, brute_nearest(brute, masks)[0], atol=2e-7)
    assert np.all(nearest[:20, 0] == 0)
    np.fill_diagonal(brute, np.inf)
    np.testing.assert_allclose(leave_self, brute_nearest(brute, masks)[0], atol=2e-7)
    assert np.all(leave_self[:20, 0] > 0)


def test_original_cohort_and_historical_seed_completeness_are_mandatory() -> None:
    index = np.arange(56658)
    cohorts = {"public_val": index < 1000, "eval_group_reference": index < 5161,
               "all_reference": np.ones(len(index), dtype=bool)}
    validate_cohorts(cohorts, len(index))
    cohorts["eval_group_reference"] = (index > 0) & (index <= 5161)
    with pytest.raises(ValueError, match="denominators"):
        validate_cohorts(cohorts, len(index))
    names = ["nanofold", *[f"{method}_seed{seed}" for method in ("uniform_chain", *analyzer.METHODS) for seed in range(100)]]
    distances = np.zeros((3, len(names)), dtype=np.float32)
    validate_historical_arrays(names, distances, distances, 3)
    with pytest.raises(ValueError, match="all four methods"):
        validate_historical_arrays(names[:-1], distances[:, :-1], distances[:, :-1], 3)


def test_measured_geometry_has_explicit_missing_mass() -> None:
    frame = rows()[:2]
    frame[0]["source_geometry"] = {"mapping_status": "verified", "ca_distance_histogram": [1., 0, 0, 0, 0, 0, 0, 0]}
    profile = distributions(frame, protocol(), "sequence_uniform")["measured_chain_ca_distance_profile"]
    assert profile["0"] == .5 and profile["__unknown__"] == .5
    frame[0]["source_geometry"]["ca_distance_histogram"] = [0.] * 8
    with pytest.raises(ValueError, match="probability vector"):
        distributions(frame, protocol(), "sequence_uniform")


def test_geometry_comparison_is_unavailable_when_measurement_coverage_differs() -> None:
    frame = rows()
    profile = {"mapping_status": "verified", "ca_distance_histogram": [1., 0, 0, 0, 0, 0, 0, 0]}
    for row in frame:
        row["source_geometry"] = profile
    reference_distributions = {mode: distributions(frame, protocol(), mode) for mode in protocol()["weighting"]}
    support = {"topology": independent_support(frame, "topology_id"), "superfamily": independent_support(frame, "homologous_superfamily_id")}
    observed = copy.deepcopy(frame[:3])
    observed[0].pop("source_geometry")
    result = describe_arm(observed, frame, protocol(), reference_distributions, support)
    assert result["sequence_uniform.measured_chain_ca_distance_profile.js_bits"] is None
    assert result["measured_geometry_available_fraction"] == pytest.approx(2 / 3)


def test_metadata_only_mode_still_requires_original_baseline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config, frozen = fixture_config(tmp_path, monkeypatch)
    settings = json.loads(config.read_text())
    baseline = Path(settings["inputs"]["candidate06"]["path"])
    baseline.write_text("1aaa_A\n3aaa_A\n5aaa_A\n")
    settings["inputs"]["candidate06"] = pin(baseline)
    config.write_text(json.dumps(settings))
    with pytest.raises(ValueError, match="frozen historical manifest"):
        run(config, frozen, digest(frozen), tmp_path / "bad", skip_embeddings=True)


def test_expanded_group_control_is_explicit_and_does_not_require_all_groups() -> None:
    frame = rows()
    for index, row in enumerate(frame):
        row["group_id"] = f"g{index}"
    controls, unavailable = draw_controls(frame, 3, [0], 20261003, group_policy="soft_diversity")
    assert not unavailable
    group = next(arm for arm in controls if arm["method"] == "group_uniform_distinct")
    assert len({frame[index]["group_id"] for index in group["indices"]}) == 3
    assert len(group["indices"]) == 3


def test_raw_crop_counts_are_not_hidden_by_normalized_shape() -> None:
    result = analyzer.crop_discrepancy({"topology_intact": {"1.1.1": .2}}, {"topology_intact": {"1.1.1": 1.}})
    assert result["topology_intact.shape_js_bits"] == 0
    assert result["topology_intact.raw_l1"] == .8
    assert result["topology_intact.expected_total"] == .2
    assert result["topology_intact.total_difference"] == -.8


def test_source_support_reports_chain_and_parent_mass_separately() -> None:
    first, second = rows()[0], rows()[4]
    full = [{**first, "chain_id": f"copy{i}_A"} for i in range(9)] + [second]
    result = analyzer.source_stage_statistics([first], full)
    assert result["full_source_unique_parent_mass_represented"] == .5
    assert result["full_source_chain_mass_represented_by_parent_identity"] == .9
    assert result["full_source_residue_mass_represented_by_parent_identity"] == .9
    reference = analyzer.source_stage_statistics(full, full)
    assert reference["deposited_chain_uniform"]["topology"]["distribution"]["1.1.1"] == pytest.approx(.9)
    assert reference["unique_parent_then_uniform_copy"]["topology"]["distribution"]["1.1.1"] == pytest.approx(.5)


def test_historical_incompatibility_disables_primary_paired_delta() -> None:
    records = [{"method": "candidate06", "metrics": {"score": 1}}, {"method": "candidate07", "metrics": {"score": 2}}]
    summary = analyzer.summarize(records, baseline_comparable=False)
    assert summary["candidate06_primary_comparable"] is False
    assert "candidate07_minus_candidate06" not in summary


def test_crop_identity_is_parent_keyed_and_short_sequences_pass_through() -> None:
    sequence = "ACDEFGHIKLMNPQRSTVWY" * 40
    first = analyzer.deterministic_crop(sequence, 0)
    assert first == analyzer.deterministic_crop(sequence, 0)
    assert first["start"] != analyzer.deterministic_crop(sequence, 1)["start"]
    assert first["stop"] - first["start"] == 256
    assert first["sequence"] == sequence[first["start"]:first["stop"]]
    short = analyzer.deterministic_crop(sequence[:80], 123)
    assert short["start"] == 0 and short["sequence"] == sequence[:80]


def test_metadata_cannot_override_current_group_or_measured_qualification() -> None:
    row = rows()[0]
    merged = canonical_row(row | {"crop_geometry": {"measured": True}},
                           {"chain_id": row["chain_id"], "length": str(row["length"]),
                            "group_id": "obsolete", "crop_geometry": None})
    assert merged["group_id"] == row["group_id"]
    assert merged["crop_geometry"] == {"measured": True}
    with pytest.raises(ValueError, match="Metadata identity conflicts"):
        canonical_row(row, {"sequence": "WRONG"})


def test_missing_crop_geometry_does_not_report_unavailable_token_as_physical_mass() -> None:
    result = analyzer.crop_discrepancy({"crop_contact_exposure": {"unavailable": 1.}},
                                      {"crop_contact_exposure": {"contacts_per_residue": 2.}})
    assert result["crop_contact_exposure.expected_total"] is None
    assert result["crop_contact_exposure.reference_expected_total"] is None
    assert result["crop_contact_exposure.raw_l1"] is None
    assert result["crop_contact_exposure.unavailable_fraction"] == 1


def test_expanded_pipeline_binds_admission_and_keeps_incompatible_baseline_historical(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, frozen = fixture_config(tmp_path, monkeypatch)
    settings = json.loads(config.read_text())
    raw = [{k: v for k, v in row.items() if k != "_tokens"} for row in rows()]
    for index, row in enumerate(raw):
        row.update(accepted=True, group_id=f"current{index}", release_date="2020-01-01")
        row["cath_domains"][0].update(label_segments=[{"start": 1, "end": 20}],
                                      boundary_status="verified_domain_sequence")
    admitted = raw[1:]
    for name, records in (("structure_metadata", raw), ("qualified_rows", admitted)):
        path = Path(settings["inputs"][name]["path"])
        path.write_text("".join(json.dumps(row) + "\n" for row in records) if name == "qualified_rows" else json.dumps(records))
        settings["inputs"][name] = pin(path)
    receipt = Path(settings["inputs"]["qualification_receipt"]["path"])
    receipt.write_text(json.dumps({"status": "complete", "qualified_chains": settings["inputs"]["qualified_rows"]}))
    settings["inputs"]["qualification_receipt"] = pin(receipt)
    train = tmp_path / "training-owned.txt"
    train.write_text("\n".join(row["chain_id"] for row in admitted) + "\n")
    settings["inputs"]["training_owned_manifest"] = pin(train)
    barrier = tmp_path / "barrier.json"
    barrier.write_text("{}\n")
    settings["inputs"]["barrier_receipt"] = pin(barrier)
    for name, value in (("chain_data_cache", raw), ("metadata_source_lock", {}),
                        ("metadata_receipt", {"input_pins": {"metadata_sources": {}}})):
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(value))
        settings["inputs"][name] = pin(path)
    def metadata_replay(**kwargs: object) -> dict:
        assert kwargs["replay"] is True
        return {"chains": raw}

    monkeypatch.setattr(analyzer, "validate_full_source_metadata", metadata_replay)
    # Native graph replay is tested by its own suite. This consumer test checks
    # exact exported identity binding, not just a caller-supplied status flag.
    frame = {"eligible_training_ids": [row["chain_id"] for row in admitted],
             "records": [row | {"training_exclusion_reasons": []} for row in admitted], "receipt": pin(barrier),
             "source_bindings": {"chain_data_cache": settings["inputs"]["chain_data_cache"]}}
    sequence_plan = tmp_path / "sequence-plan.jsonl"
    expected_plan = analyzer.sequence_plan(analyzer.training_copies(frame), rank=analyzer.copy_rank)
    sequence_plan.write_text("".join(json.dumps(row) + "\n" for row in expected_plan))
    population_plan = tmp_path / "population-plan.json"
    population_plan.write_text(json.dumps({"full_frame_receipt": frame["receipt"]}))
    qualification = {"status": "complete", "qualified_chains": settings["inputs"]["qualified_rows"],
                     "plan": pin(population_plan), "sequence_plan": pin(sequence_plan)}
    receipt.write_text(json.dumps(qualification))
    settings["inputs"]["qualification_receipt"] = pin(receipt)
    monkeypatch.setattr(analyzer, "validate_parent_frame", lambda path: frame)
    config.write_text(json.dumps(settings))
    frozen_value = json.loads(frozen.read_text())
    frozen_value["expanded_parent_crop_amendment"] = {"crop_size": 256}
    frozen.write_text(json.dumps(frozen_value))
    out = tmp_path / "expanded"
    run(config, frozen, digest(frozen), out, skip_embeddings=True)
    summary = json.loads((out / "crop-summary.json").read_text())
    assert "candidate07_minus_candidate06" not in summary
    assert set(summary["controls"]) == {"uniform_sequence", "cluster40_uniform_sequence", "group_uniform_distinct"}
    memberships = json.loads((out / "memberships.json").read_text())
    assert memberships[0]["kind"] == "historical" and not memberships[0]["primary_compatible"]
    acceptance = json.loads((out / "acceptance-evidence.json").read_text())
    assert acceptance["group_policy"] == "soft_diversity"
    assert acceptance["candidate07_exact_group_coverage"] is False
    assert set(acceptance["expanded_crop_primary"]["candidate07_vs_controls"]) == set(summary["controls"])
    metadata = json.loads((out / "metadata-summary.json").read_text())
    assert metadata["candidates"]["candidate06"]["groups"] is None
    stages = json.loads((out / "source-support-stages.json").read_text())["stages"]
    assert stages["full_pinned_experimental_chain_cache"]["distinct_parent_sequences"] == 6
    assert stages["barrier_training_owned"]["distinct_parent_sequences"] == 5
    frozen_value["repaired_public_amendment"] = {"policy": analyzer.REPAIRED_PUBLIC_POLICY}
    frozen.write_text(json.dumps(frozen_value))
    with pytest.raises(ValueError, match="frozen public-cohort repair policy"):
        run(config, frozen, digest(frozen), tmp_path / "unrevised-frame", skip_embeddings=True)
    assert not (tmp_path / "unrevised-frame").exists()
    frame["policy"] = analyzer.REPAIRED_PUBLIC_POLICY
    repaired_out = tmp_path / "repaired-frame"
    run(config, frozen, digest(frozen), repaired_out, skip_embeddings=True)
    acceptance = json.loads((repaired_out / "acceptance-evidence.json").read_text())
    assert acceptance["public_evaluation_policy"].startswith("Repaired active public is primary")
    frame["records"][0]["group_id"] = "changed-current-group"
    with pytest.raises(ValueError, match="exact current admitted parent"):
        run(config, frozen, digest(frozen), tmp_path / "bad-frame", skip_embeddings=True)
    assert not (tmp_path / "bad-frame").exists()


def test_final_baseline_replay_binds_current_predicate_and_original_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    def save(name: str, value: object) -> dict:
        path = tmp_path / name
        path.write_text(json.dumps(value))
        return pin(path)

    train_manifest = tmp_path / "train.txt"
    train_manifest.write_text("0aaa_A\n2aaa_A\n4aaa_A\n")
    manifest_record = pin(train_manifest)
    qualified = [row | {"status": "passed", "qualification": {"raw_full_audit": {"rows": 1}}}
                 for row in rows()[::2]]
    final_rows = save("final-rows.json", qualified)
    input_train = save("input-train.json", qualified)
    input_heldout = save("input-heldout.json", [])
    original = save("original.json", {"status": "complete", "inputs": {"train": manifest_record}})
    prior = save("prior.json", {"status": "complete", "outputs": {"train-qualified": input_train, "heldout-chains": input_heldout},
                                "structure_query_geometry_replay": original})
    inputs = {"train": input_train, "heldout": input_heldout, "input_receipt": prior,
              "original_replay_receipt": original, "train_manifest": manifest_record}
    predicate = {"final": "frozen-predicate"}
    current_predicate = predicate

    def validate_predicate(record: dict) -> None:
        if record != current_predicate:
            raise ValueError("Another qualification predicate")

    monkeypatch.setattr(analyzer, "validate_qualification_predicate", validate_predicate)
    binding = save("bindings.json", {"inputs": inputs, "predicate": predicate})
    receipt = {"schema_version": 1, "policy": "fixed_baseline_verified_source_replay_v1", "status": "complete",
               "counts": {"train": {"passed": 3}, "val": {"passed": 1000}, "hidden": {"passed": 1000}},
               "predicate": predicate, "outputs": {"train-qualified": final_rows}, "inputs": inputs, "binding": binding}
    receipt_pin = save("receipt.json", receipt)
    paths = {"candidate06": train_manifest, "candidate06_rows": Path(final_rows["path"]),
             "candidate06_requalification": Path(receipt_pin["path"])}
    result = analyzer.baseline_requalification_rows(paths, 3)
    assert {row["chain_id"] for row in result} == set(analyzer.manifest(train_manifest))
    current_predicate = {"changed": True}
    with pytest.raises(ValueError, match="Another qualification predicate"):
        analyzer.baseline_requalification_rows(paths, 3)


def test_observed_ca_diagnostic_is_separate_from_annotated_sequence_exposure(tmp_path: Path) -> None:
    row = rows()[0]
    row["cath_domains"][0].update(label_segments=[{"start": 1, "end": 20}], boundary_status="verified_domain_sequence")
    row["crop_geometry"] = {"sampling": "uniform_contiguous_integer_start_v1", "mapping_status": "verified",
                            "parent_length": 40, "crop_size": 256, "effective_crop_size": 40,
                            "expected_valid_ca_count": 20., "expected_contacts": 0., "parent_valid_ca_count": 20,
                            "observed_ca_segments": [{"start": 21, "end": 40}]}
    arms = [{"name": name, "method": name, "seed": None} for name in ("candidate06", "candidate07")]
    analyzer.analyze_crop_moments([row], arms, [[row], [row]], tmp_path)
    means = json.loads((tmp_path / "crop-candidate-raw-means.json").read_text())["candidate07"]
    assert means["topology"]["1.1.1"] == .5  # CATH domain exists in sequence.
    assert means["observed_ca_topology"]["1.1.1"] == 0.  # Its coordinates are absent.
    assert means["observed_ca_coverage"] == {"mapped_observed": 0., "unknown_observed": .5, "unobserved_ca": .5}
    assert sum(means["observed_ca_topology"].values()) == 1.
