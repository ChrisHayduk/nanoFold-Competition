from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
import threading
from pathlib import Path
from typing import Any

import pytest
from representative_eligibility_helpers import write_structure
from test_source_relocation import relocated_sources as relocated_sources

from nanofold.chain_paths import chain_data_dir
from nanofold.representative_eligibility import file_record
from scripts import acquire_structural_sources as acquire


def sequence_plan() -> list[dict[str, Any]]:
    sequence = "A" * 40
    return [{"sequence": sequence, "sequence_sha256": hashlib.sha256(sequence.encode()).hexdigest(),
             "group_id": "train-group", "ranked_copies": [
                 {"chain_id": chain, "sequence": sequence, "length": len(sequence), "group_id": "train-group"}
                 for chain in ("1aaa_A", "2bbb_A", "3ccc_A")]}]


def request(plan: dict[str, Any]) -> dict[str, Any]:
    return {"status": "sources_needed", "sequence_sha256": plan["sequence_sha256"], "group_id": plan["group_id"],
            "pending_chain_id": "2bbb_A", "missing": "historical_structure", "attempts": [
                {"chain_id": "1aaa_A", "rank": 0, "sequence_sha256": plan["sequence_sha256"], "status": "failed",
                 "structure_status": "failed", "projection_statistics": None,
                 "structure_rejection": "No polymer residues found"}]}


def test_requests_acquire_only_first_or_proven_next_rank() -> None:
    plans = sequence_plan()
    assert acquire.select_requests(plans, None) == ["1aaa_A"]
    assert acquire.select_requests(plans, [request(plans[0])]) == ["2bbb_A"]


@pytest.mark.parametrize("mutation", ["foreign", "group", "skip", "prefix", "duplicate", "passed"])
def test_forged_fallback_requests_do_not_expand_the_frame(mutation: str) -> None:
    plans = sequence_plan()
    row = request(plans[0])
    if mutation == "foreign":
        row["sequence_sha256"] = "0" * 64
    elif mutation == "group":
        row["group_id"] = "heldout"
    elif mutation == "skip":
        row["pending_chain_id"] = "3ccc_A"
    elif mutation == "prefix":
        row["attempts"][0]["chain_id"] = "3ccc_A"
    elif mutation == "passed":
        row["attempts"][0]["status"] = "passed"
    rows = [row, copy.deepcopy(row)] if mutation == "duplicate" else [row]
    with pytest.raises(ValueError):
        acquire.select_requests(plans, rows)


def test_structure_batches_deduplicate_pdbs_and_bind_exact_manifests(tmp_path: Path) -> None:
    selected = sorted([f"{number:04x}_A" for number in range(1, 1026)] + ["0001_B"])
    jobs = acquire.make_jobs(tmp_path, selected, {"members": {}}, [tmp_path / "empty"])
    structures = [job for job in jobs if job["kind"] == "structures"]
    assert [len(job["chain_ids"]) for job in structures] == [1024, 1]
    assert len(jobs[-1]["chain_ids"]) == 1026
    acquire.validate_jobs(jobs, tmp_path, selected, {"members": {}})
    # Coherently changing the manifest and stored digest cannot request a
    # foreign source outside the selected immutable population.
    foreign = structures[0]
    foreign["chain_ids"][0] = "zzzz_Z"
    manifest = Path(foreign["manifest"]["path"])
    manifest.write_text("\n".join(foreign["chain_ids"]) + "\n")
    foreign["manifest"] = file_record(manifest)
    with pytest.raises(ValueError, match="foreign or duplicate"):
        acquire.validate_jobs(jobs, tmp_path, selected, {"members": {}})


def arguments(tmp_path: Path) -> argparse.Namespace:
    return argparse.Namespace(acquisition_code_root=tmp_path / "frozen code", duplicate_chains_file=tmp_path / "duplicates.txt",
                              raw_output_root=tmp_path / "raw-output", max_attempts=2)


def test_commands_use_pinned_owners_without_live_structures_or_templates(tmp_path: Path) -> None:
    args = arguments(tmp_path)
    pin = {"size": 123456789, "etag": '"literal-quoted-etag"', "version_id": None}
    job = {"kind": "structures", "manifest": {"path": str(tmp_path / "manifest with spaces.txt")}}
    argv = acquire.command(job, tmp_path / "fresh-output", args, pin)
    assert argv[:2] == [sys.executable, str(args.acquisition_code_root / "scripts/fetch_openproteinset_mmcif.py")]
    assert argv[argv.index("--archive-etag") + 1] == '"literal-quoted-etag"'
    assert argv[argv.index("--archive-version-id") + 1] == "absent"
    job["kind"] = "raw"
    argv = acquire.command(job, tmp_path / "fresh-raw", args, pin)
    assert argv[:2] == [sys.executable, str(acquire.REPO_ROOT / "scripts/prepare_data.py")]
    assert argv[argv.index("--download-workers") + 1] == "8"
    assert "--strict-downloads" in argv and "--no-template-hits" in argv
    assert "--download-mmcif-subset" not in argv and "--include-mmcif-zip" not in argv


def test_existing_raw_sources_are_reused_but_empty_files_are_not_accepted(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    source = chain_data_dir(raw / "roda_pdb", "1aaa_A") / "uniref90_hits.a3m"
    source.parent.mkdir(parents=True)
    source.write_text(">query\nAAAA\n")
    structure = tmp_path / "structure.cif"
    structure.write_text("data_1aaa\n")
    catalog = {"members": {"1aaa": file_record(structure)}}
    assert acquire.make_jobs(tmp_path, ["1aaa_A"], catalog, [raw]) == []
    source.write_text("")
    with pytest.raises(ValueError, match="Existing raw source is empty"):
        acquire.make_jobs(tmp_path / "another", ["1aaa_A"], catalog, [raw])


def test_failed_download_preserves_attempt_and_fresh_retry_is_bounded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    args = arguments(tmp_path)
    manifest = tmp_path / "raw.txt"
    manifest.write_text("1aaa_A\n")
    job = {"name": "raw", "kind": "raw", "chain_ids": ["1aaa_A"], "manifest": file_record(manifest)}
    monkeypatch.setattr(acquire, "check_disk", lambda path: None)
    # Exercise durable subprocess handling with a local process, never a downloader.
    monkeypatch.setattr(acquire, "command", lambda *args: [sys.executable, "-c", "raise SystemExit(7)"])
    catalog = {"historical_sources": {"archive_pin": {}}}
    for number in (1, 2):
        with pytest.raises(ValueError, match="subprocess failed"):
            acquire.run_job(job, args, tmp_path, {"files": []}, catalog)
        attempt = tmp_path / "jobs/raw" / f"attempt-{number:03d}"
        result = json.loads((attempt / "result.json").read_text())
        assert result["exit_code"] == 7 and result["status"] == "failed_or_interrupted"
        assert (attempt / "stdout.log").exists() and (attempt / "stderr.log").exists()
        assert not (tmp_path / "jobs/raw/complete.json").exists()
    destinations = [json.loads(path.read_text())["destination"] for path in (tmp_path / "jobs/raw").glob("attempt-*/start.json")]
    assert len(set(destinations)) == 2
    with pytest.raises(ValueError, match="attempt limit exhausted"):
        acquire.run_job(job, args, tmp_path, {"files": []}, catalog)


def test_input_drift_and_disk_guard_prevent_subprocess_launch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = tmp_path / "source"
    source.write_text("frozen")
    binding = {"files": [file_record(source)]}
    source.write_text("changed")
    with pytest.raises(ValueError, match="Bound source file changed"):
        acquire.check_inputs(binding)
    monkeypatch.setattr(acquire.shutil, "disk_usage", lambda path: argparse.Namespace(free=acquire.MIN_FREE_BYTES - 1))
    with pytest.raises(ValueError, match="at least 50 GiB"):
        acquire.check_disk(tmp_path)


def test_unrecorded_or_live_interrupted_child_is_not_retried(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    args = arguments(tmp_path)
    manifest = tmp_path / "raw.txt"
    manifest.write_text("1aaa_A\n")
    job = {"name": "raw", "kind": "raw", "chain_ids": ["1aaa_A"], "manifest": file_record(manifest)}
    attempt = tmp_path / "jobs/raw/attempt-001"
    attempt.mkdir(parents=True)
    with pytest.raises(ValueError, match="no child identity"):
        acquire.run_job(job, args, tmp_path, {"files": []}, {})
    (attempt / "process.json").write_text(json.dumps({"pid": 123456}))
    monkeypatch.setattr(acquire.os, "kill", lambda *args: None)
    with pytest.raises(ValueError, match="may still be running"):
        acquire.run_job(job, args, tmp_path, {"files": []}, {})


def test_subprocess_receipt_success_never_hides_nonzero_exit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    args = arguments(tmp_path)
    manifest = tmp_path / "raw.txt"
    manifest.write_text("1aaa_A\n")
    job = {"name": "raw", "kind": "raw", "chain_ids": ["1aaa_A"], "manifest": file_record(manifest)}
    monkeypatch.setattr(acquire, "check_disk", lambda path: None)
    monkeypatch.setattr(acquire, "command", lambda *args: [sys.executable, "-c", "raise SystemExit(1)"])
    monkeypatch.setattr(acquire, "verify_job", lambda *args: pytest.fail("nonzero exit must not verify outputs"))
    with pytest.raises(ValueError, match="subprocess failed"):
        acquire.run_job(job, args, tmp_path, {"files": []}, {"historical_sources": {"archive_pin": {}}})


def test_worker_cancellation_terminates_and_reaps_its_subprocess(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    args = arguments(tmp_path)
    manifest = tmp_path / "raw.txt"
    manifest.write_text("1aaa_A\n")
    job = {"name": "raw", "kind": "raw", "chain_ids": ["1aaa_A"], "manifest": file_record(manifest)}
    monkeypatch.setattr(acquire, "check_disk", lambda path: None)
    monkeypatch.setattr(acquire, "command", lambda *args: [sys.executable, "-c", "import time; time.sleep(30)"])
    canceled = threading.Event()
    timer = threading.Timer(0.1, canceled.set)
    timer.start()
    try:
        with pytest.raises(ValueError, match="canceled after"):
            acquire.run_job(job, args, tmp_path, {"files": []}, {"historical_sources": {"archive_pin": {}}}, canceled)
    finally:
        timer.cancel()
    attempt = tmp_path / "jobs/raw/attempt-001"
    assert json.loads((attempt / "result.json").read_text())["status"] == "failed_or_interrupted"
    pid = json.loads((attempt / "process.json").read_text())["pid"]
    with pytest.raises(ProcessLookupError):
        acquire.os.kill(pid, 0)


def full_arguments(tmp_path: Path, fixture: dict[str, Any]) -> argparse.Namespace:
    (fixture["code_root"] / "nanofold/__init__.py").write_text("")
    rows = sequence_plan()
    rows[0]["ranked_copies"] = rows[0]["ranked_copies"][:1]
    frame = tmp_path / "sequences.jsonl"
    frame.write_text("".join(json.dumps(row) + "\n" for row in rows))
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps({"policy": acquire.POLICY, "sequence_plan": file_record(frame),
                               "sequence_count": 1, "inputs": []}))
    catalog = tmp_path / "catalog.json"
    catalog.write_text(json.dumps(fixture["catalog"]))
    duplicates = tmp_path / "duplicates.txt"
    duplicates.write_text("")
    return acquire.parse_args(["--plan", str(plan), "--source-catalog", str(catalog),
                               "--raw-root", str(fixture["raw_root"]), "--raw-output-root", str(tmp_path / "new-raw"),
                               "--duplicate-chains-file", str(duplicates), "--acquisition-code-root", str(fixture["code_root"]),
                               "--output", str(tmp_path / "acquisition")])


def test_complete_reused_sources_emit_unqualified_catalog_and_pinned_resume(
    tmp_path: Path, relocated_sources: dict[str, Any], monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = full_arguments(tmp_path, relocated_sources)
    monkeypatch.setattr(acquire.subprocess, "Popen", lambda *args, **kwargs: pytest.fail("all sources already acquired"))
    planned = acquire.run(args)
    assert planned["status"] == "planned" and planned["job_count"] == 0
    args.execute = True
    complete = acquire.run(args)
    assert complete["status"] == "retrieved_unqualified"
    assert complete["biological_identity_validated"] is False
    assert acquire.run(args) == complete
    args.duplicate_chains_file.write_text("1aaa_A 2bbb_A\n")
    with pytest.raises(ValueError, match="Resume differs from frozen"):
        acquire.run(args)


def test_input_change_while_reading_cannot_bind_new_bytes_to_old_plan(
    tmp_path: Path, relocated_sources: dict[str, Any], monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = full_arguments(tmp_path, relocated_sources)
    original_read = acquire.read_json

    def change_after_read(path: Path) -> dict[str, Any]:
        result = original_read(path)
        if path == args.plan:
            path.write_text(path.read_text() + "\n")
        return result

    monkeypatch.setattr(acquire, "read_json", change_after_read)
    with pytest.raises(ValueError, match="Bound source file changed"):
        acquire.freeze_inputs(args)


def test_completed_independent_receipt_is_merged_before_missing_source_planning(
    tmp_path: Path, relocated_sources: dict[str, Any],
) -> None:
    args = full_arguments(tmp_path, relocated_sources)
    frame = tmp_path / "sequences.jsonl"
    row = json.loads(frame.read_text())
    row["ranked_copies"][0]["chain_id"] = "4ddd_A"
    frame.write_text(json.dumps(row) + "\n")
    plan = json.loads(args.plan.read_text())
    plan["sequence_plan"] = file_record(frame)
    args.plan.write_text(json.dumps(plan))
    acquired = tmp_path / "independent"
    structure = acquired / "pdb_data/mmcif_files/4ddd.cif"
    write_structure(structure, row["sequence"])
    manifest = acquired / "requested.txt"
    manifest.write_text("4ddd_A\n")
    original = json.loads(relocated_sources["original_bytes"]["receipt"])
    original["inputs"]["manifest_path"] = str(manifest)
    original["inputs"]["manifest_sha256"] = file_record(manifest)["sha256"]
    original["members"] = [{"output_path": "pdb_data/mmcif_files/4ddd.cif", "output_sha256": file_record(structure)["sha256"],
                             "crc32_verified": True, "local_header_verified": True,
                             "archive_member": {"name": "mmcif_files/4ddd.cif"}}]
    receipt = acquired / "retrieval-receipt.json"
    receipt.write_text(json.dumps(original))
    args.retrieval_receipt = [receipt]
    binding, catalog, selected = acquire.freeze_inputs(args)
    assert selected == ["4ddd_A"] and set(catalog["members"]) == {"1aaa", "4ddd"}
    assert file_record(receipt) in binding["files"] and file_record(manifest) in binding["files"]
    assert [job["kind"] for job in acquire.make_jobs(tmp_path, selected, catalog, args.raw_root)] == ["raw"]
    original["status"] = "failed"
    receipt.write_text(json.dumps(original))
    with pytest.raises(ValueError, match="acquisition is incomplete"):
        acquire.freeze_inputs(args)
