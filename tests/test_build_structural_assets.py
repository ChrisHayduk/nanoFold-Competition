from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pytest

from scripts.build_structural_assets import (
    Step,
    build_steps,
    digest,
    execute_step,
    materialize_sources,
    native_sequence_bound,
    parse_args,
    semantic_gate,
    source_rows,
)


def arguments(tmp_path: Path) -> argparse.Namespace:
    values: list[str] = []
    for name in ("candidate", "qualified-chains", "heldout-chains", "heldout-source-receipt", "chain-data-cache", "baseline-source-receipt", "mmseqs-bin", "output-dir"):
        values.extend([f"--{name}", str(tmp_path / name)])
    args = parse_args(values)
    args.min_len = 40
    args.max_len = args.max_sequence_length = 500
    return args


def qualified_row(tmp_path: Path, chain: str = "1abc_A") -> dict:
    sequence = "ACDE"
    raw, structure = tmp_path / f"{chain}.a3m", tmp_path / f"{chain}.cif"
    raw.write_text(">query\nACDE\n")
    structure.write_text("data_source\n")
    sequence_hash = hashlib.sha256(sequence.encode()).hexdigest()
    result = {"chain_id": chain, "sequence": sequence, "qualification": {
        "chain_id": chain, "status": "passed", "raw_status": "passed", "structure_status": "passed",
        "sequence_sha256": sequence_hash,
        "raw": {"path": str(raw), "sha256": digest(raw), "bytes": raw.stat().st_size, "query_sha256": sequence_hash},
        "structure": {"path": str(structure), "sha256": digest(structure)}}}
    result["qualification"]["raw_full_audit"] = {**result["qualification"]["raw"], "rows": 1, "query_length": 4, "aligned_width": 4}
    return result


def test_plan_preserves_directed_native_and_strict_preprocess_contract(tmp_path: Path) -> None:
    args = arguments(tmp_path)
    steps = {step.name: step for step in build_steps(args)}
    assert len(steps) == 21
    for split, excluded_count in (("train", 3), ("val", 1)):
        argv = steps[f"{split}_filter"].argv
        assert argv.count("--heldout-manifest") == excluded_count
        assert "--exhaustive" in argv
        assert argv[argv.index("--coverage-mode") + 1] == "2"
        assert argv[argv.index("--max-sequence-length") + 1] == "500"
        assert argv[argv.index("--min-seq-id") + 1] == "0.30"
        assert argv[argv.index("--coverage") + 1] == "0.80"
        assert argv[argv.index("--max-seqs") + 1] == argv[argv.index("--source-batch-size") + 1]
    for split in ("public", "hidden"):
        argv = steps[f"preprocess_{split}"].argv
        assert "--disable-templates" in argv and "--allow-failures" not in argv
        assert argv[argv.index("--max-msa-seqs") + 1] == "2048"
        assert ("--msa-filter-policy" in argv) == (split == "public")
    train_filter = steps["train_filter"].argv
    assert {train_filter[index + 1] for index, option in enumerate(train_filter) if option == "--heldout-manifest"} == {
        str(args.candidate / "public/val.txt"), str(args.candidate / "evaluation/historical_public.txt"),
        str(args.candidate / "hidden/hidden_val.txt")}
    residual = steps["residual_train_val"].argv
    assert {residual[index + 1] for index, option in enumerate(residual) if option == "--extra-heldout-manifest"} == {
        str(args.candidate / "evaluation/historical_public.txt"), str(args.candidate / "hidden/hidden_val.txt")}
    assert "--extra-heldout-manifest" not in steps["residual_public_hidden"].argv
    assert len([name for name in steps if name.startswith("fingerprint_")]) == 6
    assert len([name for name in steps if name.startswith("pin_")]) == 3


def test_materialization_preserves_sources_and_rejects_drift(tmp_path: Path) -> None:
    row = qualified_row(tmp_path)
    raw_root = tmp_path / "selected"
    first = materialize_sources({row["chain_id"]: row}, raw_root)
    assert first == materialize_sources({row["chain_id"]: row}, raw_root)
    assert all((raw_root / pin["output"]).is_symlink() for pin in first["files"])
    Path(row["qualification"]["raw"]["path"]).write_text(">changed\nACDE\n")
    with pytest.raises(ValueError, match="source bytes changed"):
        materialize_sources({row["chain_id"]: row}, raw_root)


def test_native_bound_includes_retired_public_query() -> None:
    rows = {"train_A": {"sequence": "A" * 100}}
    frame = {"evaluation_cohorts": {"historical_public": ["retired_A"], "active_public": ["active_A"], "hidden": ["hidden_A"]},
             "records": [{"chain_id": chain, "sequence": "A" * length}
                         for chain, length in (("retired_A", 256), ("active_A", 80), ("hidden_A", 40))]}
    assert native_sequence_bound(rows, frame) == 256
    frame["records"].pop()
    with pytest.raises(ValueError, match="Protected query is absent"):
        native_sequence_bound(rows, frame)


def test_materialization_rejects_conflicting_same_pdb_structures(tmp_path: Path) -> None:
    a, b = qualified_row(tmp_path, "1abc_A"), qualified_row(tmp_path, "1abc_B")
    path = Path(b["qualification"]["structure"]["path"])
    path.write_text("another source")
    b["qualification"]["structure"]["sha256"] = digest(path)
    with pytest.raises(ValueError, match="Conflicting structures"):
        materialize_sources({a["chain_id"]: a, b["chain_id"]: b}, tmp_path / "selected")
    assert not (tmp_path / "selected").exists()


def test_source_index_requires_complete_qualified_identity(tmp_path: Path) -> None:
    row = qualified_row(tmp_path)
    index = tmp_path / "index.jsonl"
    index.write_text(json.dumps(row) + "\n")
    assert source_rows([index], {row["chain_id"]}) == {row["chain_id"]: row}
    with pytest.raises(ValueError, match="exact selected"):
        source_rows([index], {row["chain_id"], "missing_A"})
    row["sequence"] = "AAAA"
    index.write_text(json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="target sequence"):
        source_rows([index], {row["chain_id"]})


def test_step_resume_rejects_changed_outputs_without_reexecution(tmp_path: Path) -> None:
    args = arguments(tmp_path)
    args.output_dir.mkdir()
    output = args.output_dir / "produced.json"
    source = f"import pathlib; pathlib.Path({str(output)!r}).write_text('{{\"ok\":true}}')"
    step = Step("producer", [sys.executable, "-c", source], [str(output)])
    first = execute_step(step, args, "plan", "previous")
    assert execute_step(step, args, "plan", "previous") == first
    output.write_text('{"ok":false}')
    with pytest.raises(ValueError, match="Accepted outputs changed"):
        execute_step(step, args, "plan", "previous")
    assert output.read_text() == '{"ok":false}'


def test_failed_process_is_never_accepted(tmp_path: Path) -> None:
    args = arguments(tmp_path)
    step = Step("failure", [sys.executable, "-c", "raise SystemExit(3)"], [str(args.output_dir / "missing")])
    with pytest.raises(ValueError, match="Producer failed"):
        execute_step(step, args, "plan", "previous")
    assert not (args.output_dir / "receipts/failure.json").exists()
    receipt = json.loads((args.output_dir / "receipts/failure.attempt-001.json").read_text())
    assert receipt["status"] == "failed" and receipt["exit_code"] == 3
    assert receipt["process_stopped"] is False
    with pytest.raises(ValueError, match="state is unresolved"):
        execute_step(step, args, "plan", "previous")


def test_semantic_gate_rejects_success_exit_with_incomplete_audit(tmp_path: Path) -> None:
    path = tmp_path / "audit.json"
    path.write_text(json.dumps({"status": "passed", "failure_count": 0, "passed_chain_count": 11999}))
    step = Step("audit", [], [str(path)], "audit")
    with pytest.raises(ValueError, match="full corpus"):
        semantic_gate(step, arguments(tmp_path))


def test_tensor_gate_preserves_distinct_deletion_profiles(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import numpy as np

    from nanofold.chain_paths import chain_npz_path
    from scripts import build_structural_assets as assets

    args = arguments(tmp_path)
    (args.candidate / "public").mkdir(parents=True)
    (args.candidate / "public/all.txt").write_text("1abc_A\n")
    features, labels = args.output_dir / "public_features", args.output_dir / "public_labels"
    features.mkdir(parents=True)
    labels.mkdir()
    (features / "preprocess_meta.json").write_text(json.dumps({"cli_args": {"disable_templates": True, "max_msa_seqs": 2048}}))
    np.savez(chain_npz_path(labels, "1abc_A"), unused=np.array([1]))
    monkeypatch.setattr(assets, "validate_feature_label_pair", lambda *a, **kw: [])
    msa = np.array([[0, 1], [0, 1]], dtype=np.int32)
    deletions = np.array([[0, 0], [1, 0]], dtype=np.float32)
    path = chain_npz_path(features, "1abc_A")
    np.savez(path, msa=msa, deletions=deletions, template_aatype=np.empty((0, 2)))
    assets.tensor_gate(args, "public")
    np.savez(path, msa=msa, deletions=np.zeros_like(deletions), template_aatype=np.empty((0, 2)))
    with pytest.raises(ValueError, match="deduplicated"):
        assets.tensor_gate(args, "public")


def test_writable_symlink_is_rejected_before_a_producer_runs(tmp_path: Path) -> None:
    from scripts.build_structural_assets import assert_output_containment

    args = arguments(tmp_path)
    args.output_dir.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (args.output_dir / "hidden_labels").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        assert_output_containment(args.output_dir)
    assert list(outside.iterdir()) == []


def test_root_without_plan_must_be_fresh(tmp_path: Path) -> None:
    from scripts.build_structural_assets import admit_output_root

    root = tmp_path / "old-assets"
    root.mkdir()
    original = root / "hidden_labels.npz"
    original.write_bytes(b"unowned")
    with pytest.raises(ValueError, match="must be fresh"):
        admit_output_root(root)
    assert original.read_bytes() == b"unowned"


def test_input_manifest_drift_blocks_producer(tmp_path: Path) -> None:
    from scripts.build_structural_assets import atomic_json, file_pin

    args = arguments(tmp_path)
    args.output_dir.mkdir()
    manifest = tmp_path / "train.txt"
    manifest.write_text("original_A\n")
    plan = {"inputs": [file_pin(manifest)], "code": {}}
    atomic_json(args.output_dir / "plan.json", plan)
    manifest.write_text("replacement_A\n")
    output = args.output_dir / "made.txt"
    step = Step("blocked", [sys.executable, "-c", f"open({str(output)!r}, 'w').write('unsafe')"], [str(output)])
    with pytest.raises(ValueError, match="bound producer input changed"):
        execute_step(step, args, "plan", "previous")
    assert not output.exists()


def test_unresolved_running_attempt_blocks_restart_even_without_outputs(tmp_path: Path) -> None:
    from scripts.build_structural_assets import atomic_json

    args = arguments(tmp_path)
    output = args.output_dir / "produced.txt"
    step = Step("interrupted", [sys.executable, "-c", f"open({str(output)!r}, 'w').write('bad')"], [str(output)])
    atomic_json(args.output_dir / "receipts/interrupted.attempt-001.json", {
        "status": "running", "process_group_id": 99999999, "process_stopped": False,
        "plan_sha256": "plan", "previous_receipt_sha256": "previous", "argv": step.argv})
    with pytest.raises(ValueError, match="state is unresolved"):
        execute_step(step, args, "plan", "previous")
    assert not output.exists()


def test_repaired_public_sources_bind_added_proof_and_keep_hidden_fixed() -> None:
    from scripts.build_structural_assets import validate_evaluation_sources

    def qualification(digest: str) -> dict:
        return {"structure": {"sha256": digest}, "raw": {"sha256": digest}}

    original = {"retained_A": qualification("retained"), "hidden_A": qualification("hidden")}
    frame = {"evaluation_cohorts": {"historical_public": ["retained_A", "retired_A"],
                                   "active_public": ["retained_A", "new_A"], "hidden": ["hidden_A"]},
             "selected_qualifications": {"new_A": qualification("new")}}
    rows = {chain: {"qualification": values} for chain, values in
            {**original, "new_A": qualification("new")}.items()}
    validate_evaluation_sources(frame, rows, original)
    rows["hidden_A"] = {"qualification": qualification("changed")}
    with pytest.raises(ValueError, match="acquisition proof"):
        validate_evaluation_sources(frame, rows, original)
    rows["hidden_A"] = {"qualification": original["hidden_A"]}
    rows["new_A"] = {"qualification": qualification("unapproved")}
    with pytest.raises(ValueError, match="acquisition proof"):
        validate_evaluation_sources(frame, rows, original)
