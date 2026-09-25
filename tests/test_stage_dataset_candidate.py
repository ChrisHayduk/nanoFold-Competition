from __future__ import annotations

import hashlib
import json
import os
import random
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from scripts import stage_dataset_candidate as staging


def _inputs(tmp_path: Path, *, related: bool = False) -> list[str]:
    cache = tmp_path / "cache.json"
    sources = tmp_path / "sources"
    sources.mkdir()
    rows = {
        f"{index:04d}_A": {
            "seq": ("A" * 40 + "".join("C" if index & (1 << bit) else "A" for bit in range(8))
                    if related else "".join(random.Random(index + 987).choices("ARNDCQEGHILKMFPSTWYV", k=48))),
            "resolution": 2.0, "oligomeric_count": 1,
        }
        for index in range(12)
    }
    cache.write_text(json.dumps(rows))
    cath = sources / "cath-domain-list.txt"
    cath.write_text("".join(f"{index:04d}A00 1 10 20 30 1 1 1 1 1 48 2.0\n" for index in range(12)))
    source_lock = tmp_path / "sources.lock.json"
    source_lock.write_text(json.dumps({
        "schema_version": 2,
        "chain_data_cache": {"sha256": hashlib.sha256(cache.read_bytes()).hexdigest()},
        "sources": {"cath_domain_list": {"path": cath.name, "sha256": hashlib.sha256(cath.read_bytes()).hexdigest()}},
    }))
    cluster_tsv = tmp_path / "clusters.tsv"
    cluster_tsv.write_text("".join(f"{chain}\t{chain}\n" for chain in rows))
    exclusions = tmp_path / "exclusions.txt"
    exclusions.touch()
    salt = tmp_path / "salt.txt"
    salt.write_text("stage-fixture-secret-never-in-public-contract")
    return [
        "--sampling-policy", "cath-v2",
        "--stage-dir", str(tmp_path / "candidate"), "--chain-data-cache", str(cache),
        "--metadata-sources-dir", str(sources), "--metadata-source-lock", str(source_lock),
        "--feature-exclusion-list", str(exclusions), "--processability-exclusion-list", str(exclusions),
        "--cluster-tsv", str(cluster_tsv), "--train-size", "6", "--val-size", "2", "--hidden-val-size", "2",
        "--require-monomer", "--hidden-split-salt-file", str(salt),
        "--mmseqs-bin", sys.executable,
    ]


def _mmseqs_binary() -> str:
    binary = os.environ.get("MMSEQS_TEST_BINARY") or shutil.which("mmseqs")
    if not binary:
        pytest.skip("Set MMSEQS_TEST_BINARY to run real staged target-audit integration")
    return binary


def test_staging_runs_real_builders_and_verifier_without_changing_release(tmp_path: Path) -> None:
    args = staging.parse_args([*_inputs(tmp_path), "--mmseqs-bin", _mmseqs_binary()])
    frozen = staging.REPO_ROOT / "leaderboard/official_manifest_source.lock.json"
    frozen_bytes = frozen.read_bytes()
    receipt = staging.stage_candidate(args)
    status = json.loads(receipt.read_text())
    assert status["status"] == "manifests_verified"
    assert status["processed_assets_audited"] is False
    assert status["target_sequences_audited"] is True
    target_audit = json.loads((receipt.parent / "target_audit/summary.json").read_text())
    assert {name: pair["requested_pair_count"] for name, pair in target_audit["pairs"].items()} == {
        "public": 12, "hidden": 16,
    }
    assert target_audit["status"] == "passed"
    for relative_path, digest in status["artifact_sha256"].items():
        assert hashlib.sha256((receipt.parent / relative_path).read_bytes()).hexdigest() == digest
    verification = json.loads((receipt.parent / "manifest-verification.json").read_text())
    assert verification["counts"] == {"train": 6, "val": 2, "hidden_val": 2}
    assert (receipt.parent / "public/train.txt").is_file()
    assert (receipt.parent / "hidden/hidden_val.txt").is_file()
    public_lock = (receipt.parent / "public-source.lock.json").read_text()
    assert "stage-fixture-secret-never-in-public-contract" not in public_lock
    assert "hidden_split_salt_sha256" not in public_lock
    assert frozen.read_bytes() == frozen_bytes
    assert receipt.parent.stat().st_mode & 0o777 == 0o700
    with pytest.raises(FileExistsError):
        staging.stage_candidate(args)


def test_full_coverage_staging_with_real_homology_graph_and_extra_train_target(tmp_path: Path) -> None:
    from scripts import build_homology_groups as graph

    args = staging.parse_args([*_inputs(tmp_path), "--sampling-policy", "cath-v3",
                               "--mmseqs-bin", _mmseqs_binary()])
    cache = json.loads(args.chain_data_cache.read_text())
    template = cache["0000_A"]["seq"]
    for index in range(1, 4):
        sequence = list(template)
        sequence[index] = "A" if sequence[index] != "A" else "C"
        cache[f"{index:04d}_A"]["seq"] = "".join(sequence)
    args.chain_data_cache.write_text(json.dumps(cache))
    source_lock = json.loads(args.metadata_source_lock.read_text())
    source_lock["chain_data_cache"]["sha256"] = staging._sha256(args.chain_data_cache)
    args.metadata_source_lock.write_text(json.dumps(source_lock))
    metadata = tmp_path / "graph-metadata.json"
    subprocess.run([
        sys.executable, str(staging.SCRIPT_DIR / "build_structure_metadata.py"),
        "--chain-data-cache", str(args.chain_data_cache),
        "--metadata-sources-dir", str(args.metadata_sources_dir),
        "--metadata-source-lock", str(args.metadata_source_lock),
        "--feature-exclusion-list", str(args.feature_exclusion_list),
        "--processability-exclusion-list", str(args.processability_exclusion_list),
        "--metadata-out", str(metadata), "--require-monomer",
    ], check=True, capture_output=True, text=True)
    receipt = graph.build_groups(graph.parse_args([
        "--chain-data-cache", str(args.chain_data_cache), "--structure-metadata", str(metadata),
        "--cluster-tsv", str(args.cluster_tsv), "--out-dir", str(tmp_path / "groups"),
        "--mmseqs-bin", args.mmseqs_bin,
    ]))
    args.split_group_receipt = receipt
    args.split_group_tsv = receipt.parent / "split-groups.tsv"
    status = json.loads(staging.stage_candidate(args).read_text())
    assert status["status"] == "manifests_verified"
    assert status["target_sequences_audited"] is True
    assert status["all_group_coverage_verified"] is True
    assert status["group_coverage"]["complete_group_count"] == 9
    assert status["group_coverage"]["covered_group_count"] == 9
    assert status["group_coverage"]["split_group_counts"] == {"train": 5, "val": 2, "hidden_val": 2}
    assert len((args.stage_dir / "public/train.txt").read_text().splitlines()) == 6
    group_map = graph.load_map(args.split_group_tsv, set(cache), canonical=True)
    train = (args.stage_dir / "public/train.txt").read_text().splitlines()
    assert sum(group_map[chain] == group_map["0000_A"] for chain in train) == 2


def test_cluster_disjoint_but_related_targets_fail_staging_and_retain_both_audits(tmp_path: Path) -> None:
    args = staging.parse_args([*_inputs(tmp_path, related=True), "--mmseqs-bin", _mmseqs_binary()])
    with pytest.raises(subprocess.CalledProcessError):
        staging.stage_candidate(args)
    status = json.loads((args.stage_dir / "candidate_status.json").read_text())
    assert status["status"] == "failed"
    assert status["step"] == "target_audit"
    assert status["target_sequences_audited"] is False
    assert "artifact_sha256" not in status
    verification = json.loads((args.stage_dir / "manifest-verification.json").read_text())
    assert verification["status"] == "ok"
    audit = json.loads((args.stage_dir / "target_audit/summary.json").read_text())
    assert audit["status"] == "failed"
    assert set(audit["pairs"]) == {"public", "hidden"}
    assert all(pair["qualifying_alignment_count"] > 0 for pair in audit["pairs"].values())


def test_precomputed_clusters_still_require_an_audit_binary_before_outputs(tmp_path: Path) -> None:
    args = staging.parse_args([*_inputs(tmp_path), "--mmseqs-bin", str(tmp_path / "missing-mmseqs")])
    with pytest.raises(ValueError, match="MMseqs2 is required"):
        staging.stage_candidate(args)
    assert not args.stage_dir.exists()


def test_full_coverage_policy_requires_both_group_map_and_receipt_before_outputs(tmp_path: Path) -> None:
    options = _inputs(tmp_path)
    args = staging.parse_args([*options, "--sampling-policy", "cath-v3"])
    with pytest.raises(ValueError, match="requires --split-group-tsv and --split-group-receipt"):
        staging.stage_candidate(args)
    assert not args.stage_dir.exists()
    args.split_group_tsv = args.cluster_tsv
    with pytest.raises(ValueError, match="requires --split-group-tsv and --split-group-receipt"):
        staging.stage_candidate(args)
    assert not args.stage_dir.exists()


def test_cath_v2_cannot_silently_ignore_full_group_contract(tmp_path: Path) -> None:
    args = staging.parse_args(_inputs(tmp_path))
    args.split_group_tsv = args.cluster_tsv
    args.split_group_receipt = args.metadata_source_lock
    with pytest.raises(ValueError, match="only supported for cath-v3"):
        staging.stage_candidate(args)
    assert not args.stage_dir.exists()


@pytest.mark.parametrize("mutation", ["omitted_group", "wrong_owner_count", "status_only", "false_coverage", "boolean_count"])
def test_full_coverage_receipt_requires_every_locked_group(tmp_path: Path, mutation: str) -> None:
    args = staging.parse_args(_inputs(tmp_path))
    lock = {"split_groups": {"group_count": 9, "selection_unit_count": 12, "selection_unit_ids_sha256": "0" * 64}}
    coverage = {
        "required": "all_groups", "all_group_coverage_verified": True,
        "complete_group_count": 9, "covered_group_count": 9, "selection_unit_count": 12,
        "selection_unit_ids_sha256": "0" * 64,
        "split_group_counts": {"train": 5, "val": 2, "hidden_val": 2},
    }
    staging._validate_group_coverage({"group_coverage": coverage}, lock, args)
    if mutation == "omitted_group":
        coverage["covered_group_count"] = 8
    elif mutation == "wrong_owner_count":
        coverage["split_group_counts"] = {"train": 6, "val": 2, "hidden_val": 1}
    elif mutation == "status_only":
        coverage = {"all_group_coverage_verified": True}
    elif mutation == "false_coverage":
        coverage["all_group_coverage_verified"] = 1
    else:
        coverage["split_group_counts"] = {"train": 5, "val": 2, "hidden_val": True}
    with pytest.raises(ValueError, match="cover every locked group"):
        staging._validate_group_coverage({"group_coverage": coverage}, lock, args)


def test_inputs_changed_after_audit_cannot_receive_verified_receipt(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    args = staging.parse_args([*_inputs(tmp_path), "--mmseqs-bin", _mmseqs_binary()])
    original_run = subprocess.run

    def mutate_after_audit(command, **kwargs):
        result = original_run(command, **kwargs)
        if Path(command[1]).name == "audit_cache_targets.py":
            args.chain_data_cache.write_text(args.chain_data_cache.read_text() + "\n")
        return result

    monkeypatch.setattr(staging.subprocess, "run", mutate_after_audit)
    with pytest.raises(ValueError, match="inputs changed between"):
        staging.stage_candidate(args)
    status = json.loads((args.stage_dir / "candidate_status.json").read_text())
    assert status["status"] == "failed"
    assert "artifact_sha256" not in status


def test_failed_allocation_never_receives_verified_status_and_cannot_be_reused(tmp_path: Path) -> None:
    args = staging.parse_args([*_inputs(tmp_path), "--train-size", "20"])
    with pytest.raises(subprocess.CalledProcessError):
        staging.stage_candidate(args)
    status = json.loads((args.stage_dir / "candidate_status.json").read_text())
    assert status["status"] == "failed"
    assert status["step"] == "manifests"
    assert "artifact_sha256" not in status
    assert not (args.stage_dir / "manifest-verification.json").exists()
    assert not (args.stage_dir / "verification.log").exists()
    with pytest.raises(FileExistsError):
        staging.stage_candidate(args)


def test_corrupt_pinned_metadata_stops_before_manifest_generation(tmp_path: Path) -> None:
    args = staging.parse_args(_inputs(tmp_path))
    (args.metadata_sources_dir / "cath-domain-list.txt").write_text("changed source\n")
    with pytest.raises(subprocess.CalledProcessError):
        staging.stage_candidate(args)
    status = json.loads((args.stage_dir / "candidate_status.json").read_text())
    assert status["status"] == "failed"
    assert status["step"] == "metadata"
    assert not (args.stage_dir / "public").exists()
    assert not (args.stage_dir / "private-source.lock.json").exists()


def test_unignored_repository_stage_is_rejected_before_creating_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = staging.parse_args(_inputs(tmp_path))
    monkeypatch.setattr(staging, "REPO_ROOT", tmp_path)
    with pytest.raises(ValueError, match="hidden artifacts untracked"):
        staging.stage_candidate(args)
    assert not args.stage_dir.exists()


def test_private_root_symlink_cannot_redirect_hidden_outputs_into_tracked_tree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = staging.parse_args(_inputs(tmp_path))
    checkout = tmp_path / "repo"
    tracked = checkout / "data"
    tracked.mkdir(parents=True)
    (checkout / ".nanofold_private").symlink_to(tracked, target_is_directory=True)
    args.stage_dir = checkout / ".nanofold_private/candidate"
    monkeypatch.setattr(staging, "REPO_ROOT", checkout)
    with pytest.raises(ValueError, match="hidden artifacts untracked"):
        staging.stage_candidate(args)
    assert not (tracked / "candidate").exists()


@pytest.fixture(scope="module")
def completed_target_audit(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, dict[str, Any]]:
    root = tmp_path_factory.mktemp("complete-target-receipt")
    binary = os.environ.get("MMSEQS_TEST_BINARY") or shutil.which("mmseqs")
    if binary is None:
        executable = root / "mmseqs"
        executable.write_text(
            f"#!{sys.executable}\n"
            "import pathlib, sys\n"
            "if sys.argv[1] == 'version':\n print('receipt-fixture'); sys.exit(0)\n"
            "pathlib.Path(sys.argv[4]).write_text('')\n"
        )
        executable.chmod(0o700)
        binary = str(executable)
    args = staging.parse_args([*_inputs(root), "--mmseqs-bin", binary])
    status = staging.stage_candidate(args)
    stage = status.parent
    inputs = {
        "chain_data_cache": args.chain_data_cache,
        "manifest_train": stage / "public/train.txt", "manifest_val": stage / "public/val.txt",
        "manifest_public_all": stage / "public/all.txt", "manifest_hidden": stage / "hidden/hidden_val.txt",
        "audit_script": staging.SCRIPT_DIR / "audit_cache_targets.py",
        "shared_parser": staging.REPO_ROOT / "nanofold/mmseqs.py", "mmseqs_binary": Path(args.mmseqs_bin),
    }
    expected = {
        "inputs": inputs, "input_hashes": {name: staging._sha256(path) for name, path in inputs.items()},
        "expected_counts": {"train": 6, "val": 2, "hidden": 2},
        "min_seq_id": 0.30, "coverage": 0.80, "min_len": 40, "max_len": 256,
        "threads": 1, "max_seqs": 1_000_000,
    }
    return stage / "target_audit", expected


def test_complete_target_receipt_matches_the_requested_audit(completed_target_audit) -> None:
    source, expected = completed_target_audit
    staging._validate_target_audit_receipt(source / "summary.json", **expected)


@pytest.mark.parametrize("mutation", [
    "status_only", "stale_cache_hash", "wrong_manifest_path", "missing_code_binding", "wrong_binary_hash",
    "wrong_threshold", "wrong_coverage", "wrong_sequence_contract", "wrong_expected_count",
    "wrong_pair_count", "truncated_query_count", "boolean_overlap_count", "nonzero_overlap", "missing_pair_count",
    "missing_artifact", "unsafe_artifact_name", "unexpected_artifact", "artifact_bytes_changed", "artifact_symlink",
    "pair_summary_changed", "pair_summary_rebound", "returned_alignment_with_rebound_hashes",
])
def test_target_receipt_rejects_stale_contracts_and_corrupted_artifacts(
    completed_target_audit, tmp_path: Path, mutation: str,
) -> None:
    source, expected = completed_target_audit
    copied = tmp_path / "audit"
    shutil.copytree(source, copied)
    receipt = copied / "summary.json"
    report = json.loads(receipt.read_text())
    pair = report["pairs"]["public"]
    work = copied / "public"
    if mutation == "status_only":
        report = {"status": "passed", "pairs": {"public": {"status": "passed"}, "hidden": {"status": "passed"}}}
    elif mutation == "stale_cache_hash":
        report["inputs"]["chain_data_cache"]["sha256"] = "0" * 64
    elif mutation == "wrong_manifest_path":
        report["inputs"]["manifest_train"]["path"] = str(tmp_path / "other_train.txt")
    elif mutation == "missing_code_binding":
        del report["inputs"]["audit_script"]
    elif mutation == "wrong_binary_hash":
        report["inputs"]["mmseqs_binary"]["sha256"] = "0" * 64
    elif mutation == "wrong_threshold":
        report["alignment_contract"]["min_seq_id"] = 0.99
    elif mutation == "wrong_coverage":
        report["alignment_contract"]["coverage"] = 0.10
    elif mutation == "wrong_sequence_contract":
        report["sequence_contract"]["max_len"] = 128
    elif mutation == "wrong_expected_count":
        report["expected_counts"]["train"] = 6.0
    elif mutation == "wrong_pair_count":
        pair["requested_pair_count"] -= 1
    elif mutation == "truncated_query_count":
        pair["query_records"] -= 1
    elif mutation == "boolean_overlap_count":
        pair["pdb_overlap_count"] = False
    elif mutation == "nonzero_overlap":
        pair["exact_sequence_overlap_query_count"] = 1
    elif mutation == "missing_pair_count":
        del pair["chain_overlap_count"]
    elif mutation == "missing_artifact":
        del pair["artifact_sha256"]["queries.fasta"]
    elif mutation == "unsafe_artifact_name":
        pair["artifact_sha256"]["../queries.fasta"] = pair["artifact_sha256"].pop("queries.fasta")
    elif mutation == "unexpected_artifact":
        (work / "unbound.txt").write_text("unverified output\n")
    elif mutation == "artifact_bytes_changed":
        with (work / "sources.fasta").open("a") as handle:
            handle.write("\n")
    elif mutation == "artifact_symlink":
        (work / "queries.fasta").unlink()
        (work / "queries.fasta").symlink_to(source / "public/queries.fasta")
    elif mutation in {"pair_summary_changed", "pair_summary_rebound"}:
        (work / "summary.json").write_text('{"status": "failed"}\n')
        if mutation == "pair_summary_rebound":
            report["pair_summary_sha256"]["public"] = staging._sha256(work / "summary.json")
    elif mutation == "returned_alignment_with_rebound_hashes":
        (work / "alignments.m8").write_text("query_0\tsource_0\t90\t48\t1\t1\t0\t80\n")
        pair["artifact_sha256"]["alignments.m8"] = staging._sha256(work / "alignments.m8")
        (work / "summary.json").write_text(json.dumps(pair))
        report["pair_summary_sha256"]["public"] = staging._sha256(work / "summary.json")
    else:
        raise AssertionError(mutation)
    receipt.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="Target audit|complete target|Passed target"):
        staging._validate_target_audit_receipt(receipt, **expected)


def test_staging_never_accepts_status_only_audit_receipts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    args = staging.parse_args(_inputs(tmp_path))
    original_run = subprocess.run

    def substitute_receipt(command, **kwargs):
        if Path(command[1]).name == "audit_cache_targets.py":
            output = Path(command[command.index("--out-dir") + 1])
            output.mkdir()
            (output / "summary.json").write_text(json.dumps({
                "status": "passed", "pairs": {"public": {"status": "passed"}, "hidden": {"status": "passed"}},
            }))
            return subprocess.CompletedProcess(command, 0)
        return original_run(command, **kwargs)

    monkeypatch.setattr(staging.subprocess, "run", substitute_receipt)
    with pytest.raises(ValueError, match="receipt is incomplete"):
        staging.stage_candidate(args)
    status = json.loads((args.stage_dir / "candidate_status.json").read_text())
    assert status["status"] == "failed"
    assert status["target_sequences_audited"] is False
    assert "artifact_sha256" not in status
