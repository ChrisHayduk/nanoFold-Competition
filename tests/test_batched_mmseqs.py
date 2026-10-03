from __future__ import annotations

import importlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from nanofold import mmseqs


def _fasta(path: Path, rows: list[tuple[str, str]]) -> Path:
    path.write_text("".join(f">{header}\n{sequence}\n" for header, sequence in rows))
    return path


@pytest.fixture
def batch_case(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    executable = tmp_path / "mmseqs"
    executable.write_text("#!/bin/sh\nprintf 'test-native-version\\n'\n")
    executable.chmod(0o755)
    monkeypatch.setattr(mmseqs, "_batch_resource_check", lambda *args, **kwargs: (10**15, 1024))
    return {
        "query_fasta": _fasta(tmp_path / "query.fasta", [("q0", "A" * 20), ("q1", "C" * 25), ("q2", "D" * 26)]),
        "source_fasta": _fasta(tmp_path / "source.fasta", [("s0", "A" * 20), ("s1", "D" * 26), ("s2", "C" * 25)]),
        "checkpoint_dir": tmp_path / "checkpoint", "mmseqs_bin": str(executable),
        "min_seq_id": 0.3, "coverage": 0.8, "threads": 1, "max_seqs": 2,
        "source_batch_size": 2, "query_batch_size": 2, "reserve_disk_bytes": 0, "max_rss_bytes": 10**12,
        "input_binding_sha256": "a" * 64,
    }


def _fake_native(command: list[str], log: Path, root: Path, **kwargs: Any) -> dict[str, Any]:
    query = list(mmseqs._batch_records(Path(command[2])))
    source = list(mmseqs._batch_records(Path(command[3])))
    rows = [f"{qh}\t{sh}\t100\t{len(qs)}\t1\t1\t0\t40\n"
            for qh, qs in query for sh, ss in source if qs == ss]
    Path(command[4]).write_text("".join(rows))
    # Independent count from individual lengths, including the 20/25 boundary.
    alignments = sum(min(np.float32(len(qs) / len(ss)), np.float32(len(ss) / len(qs))) >= np.float32(0.8)
                     for _, qs in query for _, ss in source)
    log.write_text(f"Query database size: {len(query)} type: Aminoacid\n"
                   f"Target database size: {len(source)} type: Aminoacid\n"
                   f"{alignments} alignments calculated\n"
                   f"{len(rows)} sequence pairs passed the thresholds (1.00 of overall calculated)\n")
    return {"elapsed_seconds": 0.1, "sampled_peak_rss_bytes": 1024, "minimum_sampled_free_disk_bytes": 10**15}


def test_complete_cartesian_search_and_equal_content_resume(batch_case: dict[str, Any], monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls = []

    def execute(*args: Any, **kwargs: Any) -> dict[str, Any]:
        calls.append(args[0])
        return _fake_native(*args, **kwargs)

    monkeypatch.setattr(mmseqs, "_batch_execute", execute)
    result = mmseqs.run_batched_mmseqs(**batch_case)
    assert result["status"] == "complete"
    assert result["batch_count"] == 4
    assert result["possible_pair_count"] == 9
    assert result["alignments_calculated"] == 7
    assert set(mmseqs.iter_batched_mmseqs_hits(result, min_seq_id=0.3, coverage=0.8)) == {("q0", "s0"), ("q1", "s2"), ("q2", "s1")}
    assert mmseqs.batched_mmseqs_summary(result)["source_partition_count"] == 2
    for name in ("query_fasta", "source_fasta"):
        moved = tmp_path / (name + ".moved")
        shutil.copyfile(batch_case[name], moved)
        batch_case[name] = moved
    assert mmseqs.run_batched_mmseqs(**batch_case) == result
    assert len(calls) == 4


def test_interrupt_retains_attempt_and_resumes_only_missing_jobs(batch_case: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    calls = 0

    def interrupted(*args: Any, **kwargs: Any) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        result = _fake_native(*args, **kwargs)
        if calls == 2:
            raise KeyboardInterrupt
        return result

    monkeypatch.setattr(mmseqs, "_batch_execute", interrupted)
    with pytest.raises(KeyboardInterrupt):
        mmseqs.run_batched_mmseqs(**batch_case)
    root = batch_case["checkpoint_dir"]
    assert not (root / "complete.json").exists()
    assert len(list(root.glob("job-*/complete.json"))) == 1
    result = mmseqs.run_batched_mmseqs(**batch_case)
    assert result["batch_count"] == 4
    assert calls == 5
    assert len(list(root.glob("job-*/attempt-*"))) == 5


def test_earlier_job_mutation_during_later_search_prevents_final_receipt(batch_case: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    outputs = []

    def mutate_earlier(command: list[str], *args: Any, **kwargs: Any) -> dict[str, Any]:
        result = _fake_native(command, *args, **kwargs)
        outputs.append(Path(command[4]))
        if len(outputs) == 2:
            outputs[0].write_text("")
        return result

    monkeypatch.setattr(mmseqs, "_batch_execute", mutate_earlier)
    with pytest.raises(ValueError, match="before final receipt"):
        mmseqs.run_batched_mmseqs(**batch_case)
    assert not (batch_case["checkpoint_dir"] / "complete.json").exists()


@pytest.mark.parametrize("kind", ["m8", "log", "source_fasta", "query_fasta", "command", "job_receipt", "plan", "missing_plan", "input", "binary", "binding"])
def test_resume_rejects_corruption_without_launch(batch_case: dict[str, Any], monkeypatch: pytest.MonkeyPatch, kind: str) -> None:
    monkeypatch.setattr(mmseqs, "_batch_execute", _fake_native)
    result = mmseqs.run_batched_mmseqs(**batch_case)
    job = result["jobs"][0]
    if kind in ("m8", "log", "source_fasta", "query_fasta"):
        Path(job[f"{kind}_path"]).write_text("")
    elif kind == "command":
        path = Path(job["receipt_path"])
        altered = json.loads(path.read_text())
        altered["command"][altered["command"].index("--prefilter-mode") + 1] = "0"
        path.write_text(json.dumps(altered))
    elif kind == "job_receipt":
        Path(job["receipt_path"]).unlink()
    elif kind == "plan":
        path = Path(result["plan_path"])
        altered = json.loads(path.read_text())
        altered["source_partitions"] = altered["source_partitions"][:-1]
        path.write_text(json.dumps(altered))
    elif kind == "missing_plan":
        Path(result["plan_path"]).unlink()
    elif kind == "input":
        batch_case["source_fasta"].write_text(">s0\n" + "A" * 21 + "\n")
    elif kind == "binary":
        path = Path(batch_case["mmseqs_bin"])
        path.write_text(path.read_text() + "# changed\n")
    else:
        batch_case["input_binding_sha256"] = "b" * 64
    monkeypatch.setattr(mmseqs, "_batch_execute", lambda *args, **kwargs: pytest.fail("search launched with corrupted evidence"))
    with pytest.raises((ValueError, FileNotFoundError)):
        mmseqs.run_batched_mmseqs(**batch_case)


@pytest.mark.parametrize("kind", ["m8", "log", "source_fasta", "query_fasta", "receipt", "plan"])
def test_streaming_rejects_changed_evidence(batch_case: dict[str, Any], monkeypatch: pytest.MonkeyPatch, kind: str) -> None:
    monkeypatch.setattr(mmseqs, "_batch_execute", _fake_native)
    result = mmseqs.run_batched_mmseqs(**batch_case)
    path = result["plan_path"] if kind == "plan" else result["jobs"][0][f"{kind}_path"]
    Path(path).write_text("")
    with pytest.raises((ValueError, FileNotFoundError)):
        list(mmseqs.iter_batched_mmseqs_hits(result, min_seq_id=0.3, coverage=0.8))


def test_streaming_detects_mutation_after_first_hit(batch_case: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(mmseqs, "_batch_execute", _fake_native)
    result = mmseqs.run_batched_mmseqs(**batch_case)
    hits = mmseqs.iter_batched_mmseqs_hits(result, min_seq_id=0.3, coverage=0.8)
    next(hits)
    Path(result["jobs"][0]["m8_path"]).write_text("")
    with pytest.raises(ValueError, match="changed during"):
        list(hits)
    with pytest.raises(ValueError, match="thresholds differ"):
        list(mmseqs.iter_batched_mmseqs_hits(result, min_seq_id=0.2, coverage=0.8))


@pytest.mark.parametrize("corruption", ["truncated", "duplicate", "wrong_db_count", "wrong_alignment_count", "missing_log_count", "duplicate_log_count"])
def test_native_accounting_rejects_incomplete_or_duplicated_outputs(batch_case: dict[str, Any], monkeypatch: pytest.MonkeyPatch, corruption: str) -> None:
    def corrupt(command: list[str], log: Path, root: Path, **kwargs: Any) -> dict[str, Any]:
        result = _fake_native(command, log, root, **kwargs)
        m8 = Path(command[4])
        if corruption == "truncated":
            m8.write_text("")
        elif corruption == "duplicate":
            m8.write_text(m8.read_text() * 2)
        elif corruption == "wrong_db_count":
            log.write_text(log.read_text().replace("Query database size: 2", "Query database size: 9"))
        elif corruption == "wrong_alignment_count":
            log.write_text(log.read_text().replace("alignments calculated", "1 alignments calculated"))
        elif corruption == "missing_log_count":
            log.write_text("\n".join(line for line in log.read_text().splitlines() if "calculated" not in line))
        else:
            log.write_text(log.read_text() + "1 alignments calculated\n")
        return result

    monkeypatch.setattr(mmseqs, "_batch_execute", corrupt)
    with pytest.raises(ValueError):
        mmseqs.run_batched_mmseqs(**batch_case)
    assert not (batch_case["checkpoint_dir"] / "complete.json").exists()
    assert not list(batch_case["checkpoint_dir"].glob("job-*/complete.json"))


@pytest.mark.parametrize("contents", ["", ">q0 words\nAAAA\n", ">q0\nAXZ\n", ">q0\n" + "A" * 257 + "\n", ">q0\nAA\n>q0\nAA\n", ">q2\nAA\n>q1\nAA\n", ">q0\nAA\nAA\n", ">q0\nAA"])
def test_invalid_fasta_rejected_before_search(batch_case: dict[str, Any], monkeypatch: pytest.MonkeyPatch, contents: str) -> None:
    batch_case["query_fasta"].write_text(contents)
    monkeypatch.setattr(mmseqs, "_batch_execute", lambda *args, **kwargs: pytest.fail("invalid FASTA launched"))
    with pytest.raises(ValueError):
        mmseqs.run_batched_mmseqs(**batch_case)


@pytest.mark.parametrize(("option", "value"), [("source_batch_size", 1_000_001), ("query_batch_size", 257), ("source_batch_size", 0), ("max_seqs", 1), ("max_seqs", 2**31), ("threads", True), ("input_binding_sha256", "bad"), ("max_sequence_length", 0), ("max_sequence_length", 65536), ("coverage_mode", 1)])
def test_invalid_bounds_rejected_before_search(batch_case: dict[str, Any], monkeypatch: pytest.MonkeyPatch, option: str, value: Any) -> None:
    batch_case[option] = value
    monkeypatch.setattr(mmseqs, "_batch_execute", lambda *args, **kwargs: pytest.fail("invalid bounds launched"))
    with pytest.raises(ValueError):
        mmseqs.run_batched_mmseqs(**batch_case)


def test_checkpoint_lock_prevents_parallel_mutation(batch_case: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    import fcntl

    root = batch_case["checkpoint_dir"]
    root.mkdir()
    with (root / ".lock").open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(RuntimeError, match="checkpoint lock"):
            mmseqs.run_batched_mmseqs(**batch_case)
    assert not (root / "plan.json").exists()


def test_symlinked_job_directory_rejected_before_launch(batch_case: dict[str, Any], monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def interrupt(*args: Any, **kwargs: Any) -> dict[str, Any]:
        raise KeyboardInterrupt

    monkeypatch.setattr(mmseqs, "_batch_execute", interrupt)
    with pytest.raises(KeyboardInterrupt):
        mmseqs.run_batched_mmseqs(**batch_case)
    root = batch_case["checkpoint_dir"]
    external = tmp_path / "external"
    job = root / "job-000000-000000"
    job.rename(external)
    job.symlink_to(external, target_is_directory=True)
    before = list(external.rglob("*"))
    monkeypatch.setattr(mmseqs, "_batch_execute", lambda *args, **kwargs: pytest.fail("search escaped checkpoint"))
    with pytest.raises(ValueError, match="symlink"):
        mmseqs.run_batched_mmseqs(**batch_case)
    assert list(external.rglob("*")) == before


def test_resource_monitor_counts_caller_and_native_group(monkeypatch: pytest.MonkeyPatch) -> None:
    output = f"{os.getpid()} 99 100\n41 41 250\n42 41 75\n500 500 900\n"
    monkeypatch.setattr(mmseqs.subprocess, "run", lambda *args, **kwargs: subprocess.CompletedProcess(args[0], 0, output, ""))
    assert mmseqs._batch_rss_bytes(41) == 425 * 1024
    assert mmseqs._batch_rss_bytes(None) == 100 * 1024


@pytest.mark.parametrize("output", ["", "not valid process data\n", "4 4 100\n"])
def test_unavailable_or_malformed_monitor_fails_closed(monkeypatch: pytest.MonkeyPatch, output: str) -> None:
    monkeypatch.setattr(mmseqs.subprocess, "run", lambda *args, **kwargs: subprocess.CompletedProcess(args[0], 0, output, ""))
    with pytest.raises(RuntimeError):
        mmseqs._batch_rss_bytes(None)


def test_disk_and_rss_guards_fail_before_admission(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from types import SimpleNamespace

    monkeypatch.setattr(mmseqs.shutil, "disk_usage", lambda path: SimpleNamespace(free=100))
    monkeypatch.setattr(mmseqs, "_batch_rss_bytes", lambda group: 10)
    with pytest.raises(RuntimeError, match="free-disk"):
        mmseqs._batch_resource_check(tmp_path, reserve=101, max_rss=100)
    with pytest.raises(RuntimeError, match="RSS"):
        mmseqs._batch_resource_check(tmp_path, reserve=100, max_rss=9)


def test_resource_monitor_fails_closed_and_guard_kills_group(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    checks = 0

    def resources(*args: Any, **kwargs: Any) -> tuple[int, int]:
        nonlocal checks
        checks += 1
        if checks > 1:
            raise RuntimeError("RSS guard")
        return 10**15, 1024

    monkeypatch.setattr(mmseqs, "_batch_resource_check", resources)
    launched = []
    original = subprocess.Popen

    def launch(*args: Any, **kwargs: Any) -> subprocess.Popen[Any]:
        process = original(*args, **kwargs)
        launched.append(process)
        return process

    monkeypatch.setattr(mmseqs.subprocess, "Popen", launch)
    with pytest.raises(RuntimeError, match="RSS guard"):
        mmseqs._batch_execute([sys.executable, "-c", "import time; time.sleep(30)"], tmp_path / "native.log", tmp_path, reserve=0, max_rss=100)
    assert launched[0].poll() is not None
    with pytest.raises(ProcessLookupError):
        os.killpg(launched[0].pid, 0)


def test_exited_leader_with_live_child_is_rejected_and_child_killed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ready, heartbeat = tmp_path / "ready", tmp_path / "heartbeat"
    child = (
        "import os,signal,sys,time; from pathlib import Path\n"
        "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
        "Path(sys.argv[1]).write_text(str(os.getpid()))\n"
        "deadline = time.monotonic() + 10\n"
        "while time.monotonic() < deadline:\n"
        " Path(sys.argv[2]).write_text(str(time.monotonic()))\n"
        " time.sleep(0.02)\n"
    )
    leader = (
        "import subprocess,sys,time; from pathlib import Path\n"
        f"subprocess.Popen([sys.executable, '-c', {child!r}, *sys.argv[1:]])\n"
        "while not Path(sys.argv[1]).exists(): time.sleep(0.01)\n"
    )
    monkeypatch.setattr(mmseqs, "_batch_resource_check", lambda *args, **kwargs: (10**15, 1024))
    with pytest.raises(RuntimeError, match="process group remains active"):
        mmseqs._batch_execute([sys.executable, "-c", leader, str(ready), str(heartbeat)],
                              tmp_path / "native.log", tmp_path, reserve=0, max_rss=100)
    time.sleep(0.05)
    stopped = heartbeat.read_text()
    time.sleep(0.1)
    assert heartbeat.read_text() == stopped


def test_real_batched_search_matches_unpartitioned_exhaustive_search(tmp_path: Path) -> None:
    binary = os.environ.get("MMSEQS_TEST_BINARY") or shutil.which("mmseqs")
    if not binary:
        pytest.skip("Set MMSEQS_TEST_BINARY to run the native batched-search regression")
    first = "SDNKAWGPALHQYCLWHIMIYVKYYMFIMLYEMNLNEHRA"
    second = "SMPKAWGRWPHWYGLWFYFSYPMYYMFIMYYTTAWNIHRW"
    third = "ACDEFGHIKLMNPQRSTVWYACDEFGHIKLMNPQRSTVWY"
    kwargs = {"query_fasta": _fasta(tmp_path / "query.fasta", [("q0", first), ("q1", third)]),
              "source_fasta": _fasta(tmp_path / "source.fasta", [("s0", second), ("s1", third), ("s2", "A" * 20)]),
              "mmseqs_bin": binary, "min_seq_id": 0.3, "coverage": 0.8, "threads": 1, "max_seqs": 3,
              "reserve_disk_bytes": 0, "max_rss_bytes": 4 * 1024**3}
    full = mmseqs.run_batched_mmseqs(**kwargs, checkpoint_dir=tmp_path / "full", source_batch_size=3, query_batch_size=2)
    batched = mmseqs.run_batched_mmseqs(**kwargs, checkpoint_dir=tmp_path / "batched", source_batch_size=2, query_batch_size=1)
    full_hits = set(mmseqs.iter_batched_mmseqs_hits(full, min_seq_id=0.3, coverage=0.8))
    assert ("q0", "s0") in full_hits
    assert ("q1", "s1") in full_hits
    assert set(mmseqs.iter_batched_mmseqs_hits(batched, min_seq_id=0.3, coverage=0.8)) == full_hits
    assert batched["alignments_calculated"] == full["alignments_calculated"]
    assert mmseqs.run_batched_mmseqs(**kwargs, checkpoint_dir=tmp_path / "batched", source_batch_size=2, query_batch_size=1) == batched


def test_real_query_coverage_finds_short_domain_in_long_parent(tmp_path: Path) -> None:
    binary = os.environ.get("MMSEQS_TEST_BINARY") or shutil.which("mmseqs")
    if not binary:
        pytest.skip("Set MMSEQS_TEST_BINARY to test the full-parent leakage boundary")
    domain = "SDNKAWGPALHQYCLWHIMIYVKYYMFIMLYEMNLNEHRA"
    parent = "ACDEFGHIKLMNPQRSTVWY" * 15 + domain + "YWVTSRQPNMLKIHGFEDCA" * 15
    arguments = {
        "query_fasta": _fasta(tmp_path / "heldout.fasta", [("q0", domain)]),
        "source_fasta": _fasta(tmp_path / "parents.fasta", [("s0", domain), ("s1", parent), ("s2", domain[:20])]),
        "mmseqs_bin": binary, "min_seq_id": 0.3, "coverage": 0.8, "threads": 1, "max_seqs": 3,
        "reserve_disk_bytes": 0, "max_rss_bytes": 4 * 1024**3, "max_sequence_length": len(parent),
        "source_batch_size": 3, "input_binding_sha256": "a" * 64,
    }
    directional = mmseqs.run_batched_mmseqs(**arguments, coverage_mode=2, checkpoint_dir=tmp_path / "directional")
    # The real runner records -e 1e308. Verify the exact produced receipt,
    # not a synthetic command with a different equivalent float spelling.
    importlib.import_module("scripts.build_full_parent_frame").verify_barrier(
        directional, arguments["query_fasta"], arguments["source_fasta"], "a" * 64)
    assert directional["alignments_calculated"] == 2
    assert set(mmseqs.iter_batched_mmseqs_hits(directional, min_seq_id=0.3, coverage=0.8, coverage_mode=2)) == {("q0", "s0"), ("q0", "s1")}
    with pytest.raises(ValueError, match="thresholds differ"):
        list(mmseqs.iter_batched_mmseqs_hits(directional, min_seq_id=0.3, coverage=0.8))
    bilateral = mmseqs.run_batched_mmseqs(**arguments, checkpoint_dir=tmp_path / "bilateral")
    assert bilateral["alignments_calculated"] == 1
    assert set(mmseqs.iter_batched_mmseqs_hits(bilateral, min_seq_id=0.3, coverage=0.8)) == {("q0", "s0")}
    arguments["max_sequence_length"] -= 1
    with pytest.raises(ValueError, match="plan does not match"):
        mmseqs.run_batched_mmseqs(**arguments, coverage_mode=2, checkpoint_dir=tmp_path / "directional")


def test_long_record_native_size_guard_precedes_search(batch_case: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    batch_case.update(max_sequence_length=65535, source_batch_size=1_000_000)
    monkeypatch.setattr(mmseqs, "_batch_execute", lambda *args, **kwargs: pytest.fail("native bounds exceeded"))
    with pytest.raises(ValueError, match="native record-size"):
        mmseqs.run_batched_mmseqs(**batch_case)
