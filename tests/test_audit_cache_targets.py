from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

import pytest

from scripts import audit_cache_targets as audit


def _fake_mmseqs(path: Path, mode: str = "empty") -> Path:
    path.write_text(
        f"#!{sys.executable}\n"
        "import pathlib, sys\n"
        "if sys.argv[1] == 'version':\n print('fixture-mmseqs'); sys.exit(0)\n"
        "out = pathlib.Path(sys.argv[4])\n"
        f"mode = {mode!r}\n"
        "if mode == 'public_error' and out.parent.name == 'public': sys.exit(7)\n"
        "if mode == 'malformed': out.write_text('query_0\\tunknown\\t99\\t48\\t1\\t1\\t0\\t80\\n')\n"
        "elif mode == 'hits': out.write_text('query_0\\tsource_0\\t90\\t48\\t1\\t1\\t0\\t80\\n')\n"
        "else: out.write_text('')\n"
    )
    path.chmod(0o700)
    return path


def _fixture(tmp_path: Path, *, binary: Path | None = None) -> list[str]:
    cache = tmp_path / "cache.json"
    cache.write_text(json.dumps({f"{index:04d}_A": {"seq": residue * 48, "seq_length": 48}
                                 for index, residue in enumerate("ACDE", start=1)}))
    splits = {"train": ["0001_A", "0002_A"], "val": ["0003_A"],
              "public-all": ["0001_A", "0002_A", "0003_A"], "hidden": ["0004_A"]}
    argv = ["--chain-data-cache", str(cache), "--out-dir", str(tmp_path / "audit"), "--expected-counts", "2", "1", "1",
            "--mmseqs-bin", str(binary or _fake_mmseqs(tmp_path / "mmseqs"))]
    for name, ids in splits.items():
        path = tmp_path / f"{name}.txt"
        path.write_text("".join(chain + "\n" for chain in ids))
        argv.extend([f"--{name}-manifest", str(path)])
    return argv


def _report(tmp_path: Path) -> dict[str, Any]:
    return json.loads((tmp_path / "audit/summary.json").read_text())


def test_success_receipt_binds_every_input_and_private_pair_artifact(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    argv = _fixture(tmp_path)
    assert audit.main(argv) == 0
    report = _report(tmp_path)
    assert report["status"] == "passed"
    assert report["processed_assets_audited"] is False
    assert report["expected_counts"] == {"train": 2, "val": 1, "hidden": 1}
    assert report["pairs"]["public"]["requested_pair_count"] == 2
    assert report["pairs"]["hidden"]["requested_pair_count"] == 3
    for item in report["inputs"].values():
        assert hashlib.sha256(Path(item["path"]).read_bytes()).hexdigest() == item["sha256"]
    for name, pair in report["pairs"].items():
        for filename, digest in pair["artifact_sha256"].items():
            assert hashlib.sha256((tmp_path / "audit" / name / filename).read_bytes()).hexdigest() == digest
        assert hashlib.sha256((tmp_path / "audit" / name / "summary.json").read_bytes()).hexdigest() == report["pair_summary_sha256"][name]
    assert (tmp_path / "audit").stat().st_mode & 0o777 == 0o700
    assert (tmp_path / "audit/public/private_index_mapping.json").stat().st_mode & 0o077 == 0
    stdout = capsys.readouterr().out
    assert "0001_A" not in stdout and "0004_A" not in stdout and str(tmp_path) not in stdout
    original = (tmp_path / "audit/summary.json").read_bytes()
    assert audit.main(argv) == 2
    assert (tmp_path / "audit/summary.json").read_bytes() == original


@pytest.mark.parametrize("mode,code,public_status,hidden_status", [
    ("hits", 1, "failed", "failed"), ("public_error", 2, "error", "passed"), ("malformed", 2, "error", "error"),
])
def test_both_comparisons_are_retained_even_when_first_fails(
    tmp_path: Path, mode: str, code: int, public_status: str, hidden_status: str,
) -> None:
    binary = _fake_mmseqs(tmp_path / "mmseqs", mode)
    assert audit.main(_fixture(tmp_path, binary=binary)) == code
    report = _report(tmp_path)
    assert report["status"] != "passed"
    assert report["pairs"]["public"]["status"] == public_status
    assert report["pairs"]["hidden"]["status"] == hidden_status
    assert (tmp_path / "audit/public/summary.json").exists()
    assert (tmp_path / "audit/hidden/summary.json").exists()


@pytest.mark.parametrize("overlap", ["chain", "pdb", "sequence"])
def test_independent_overlap_checks_fail_even_without_returned_alignments(tmp_path: Path, overlap: str) -> None:
    argv = _fixture(tmp_path)
    cache = json.loads((tmp_path / "cache.json").read_text())
    if overlap == "chain":
        (tmp_path / "val.txt").write_text("0001_A\n")
        (tmp_path / "public-all.txt").write_text("0001_A\n0002_A\n")
    elif overlap == "pdb":
        cache["0001_B"] = cache.pop("0003_A")
        (tmp_path / "val.txt").write_text("0001_B\n")
        (tmp_path / "public-all.txt").write_text("0001_A\n0002_A\n0001_B\n")
    else:
        cache["0003_A"]["seq"] = cache["0001_A"]["seq"]
    (tmp_path / "cache.json").write_text(json.dumps(cache))
    assert audit.main(argv) == 1
    public = _report(tmp_path)["pairs"]["public"]
    key = "exact_sequence_overlap_query_count" if overlap == "sequence" else f"{overlap}_overlap_count"
    assert public[key] == 1
    assert public["qualifying_alignment_count"] == 0


@pytest.mark.parametrize("mutation", [
    "duplicate_manifest", "union", "count", "missing_sequence", "unknown_aa", "declared_length", "string_length",
    "float_length", "boolean_length", "null_alias", "empty_alias", "conflicting_alias", "unselected_bad_record", "duplicate_json",
])
def test_invalid_complete_population_fails_before_any_search(tmp_path: Path, mutation: str) -> None:
    argv = _fixture(tmp_path)
    cache = json.loads((tmp_path / "cache.json").read_text())
    selected = cache["0004_A"]
    if mutation == "duplicate_manifest":
        (tmp_path / "hidden.txt").write_text("0004_A\n0004_A\n")
    elif mutation == "union":
        (tmp_path / "public-all.txt").write_text("0001_A\n0002_A\n0004_A\n")
    elif mutation == "count":
        argv.extend(["--expected-counts", "3", "1", "1"])
    elif mutation == "missing_sequence":
        del selected["seq"]
    elif mutation == "unknown_aa":
        selected["seq"] = "X" * 48
    elif mutation in {"declared_length", "string_length", "float_length", "boolean_length"}:
        selected["seq_length"] = {"declared_length": 47, "string_length": "48", "float_length": 48.0, "boolean_length": True}[mutation]
    elif mutation in {"null_alias", "empty_alias", "conflicting_alias"}:
        selected["sequence"] = {"null_alias": None, "empty_alias": "", "conflicting_alias": "A" * 48}[mutation]
    elif mutation == "unselected_bad_record":
        cache["9999_A"] = False
    (tmp_path / "cache.json").write_text(json.dumps(cache))
    if mutation == "duplicate_json":
        (tmp_path / "cache.json").write_text('{"0001_A": {}, "0001_A": {}}')
    assert audit.main(argv) == 2
    report = _report(tmp_path)
    assert report["status"] == "error" and report["pairs"] == {}
    assert not (tmp_path / "audit/public").exists()


@pytest.mark.parametrize("output", ["cache.json", ".", "symlink"])
def test_output_aliases_cannot_overwrite_inputs(tmp_path: Path, output: str) -> None:
    argv = _fixture(tmp_path)
    original = (tmp_path / "cache.json").read_bytes()
    if output == "symlink":
        (tmp_path / "symlink").symlink_to(tmp_path / "cache.json")
    argv.extend(["--out-dir", str(tmp_path / output)])
    assert audit.main(argv) == 2
    assert (tmp_path / "cache.json").read_bytes() == original


def test_changed_inputs_cannot_receive_passed_receipt(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    argv = _fixture(tmp_path)
    original_pair = audit.run_pair

    def mutating_pair(**kwargs: Any) -> dict[str, Any]:
        result = original_pair(**kwargs)
        if kwargs["work"].name == "hidden":
            with (tmp_path / "cache.json").open("a") as handle:
                handle.write("\n")
        return result

    monkeypatch.setattr(audit, "run_pair", mutating_pair)
    assert audit.main(argv) == 2
    assert _report(tmp_path)["status"] == "error"


def test_changed_pair_summary_cannot_be_rebound_as_a_passed_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    argv = _fixture(tmp_path)
    original_pair = audit.run_pair

    def mutating_pair(**kwargs: Any) -> dict[str, Any]:
        result = original_pair(**kwargs)
        if kwargs["work"].name == "hidden":
            (tmp_path / "audit/public/summary.json").write_text(json.dumps({
                "status": "failed", "qualifying_alignment_count": 99,
            }))
        return result

    monkeypatch.setattr(audit, "run_pair", mutating_pair)
    assert audit.main(argv) == 2
    report = _report(tmp_path)
    assert report["status"] == "error"
    assert report["pair_summary_sha256"]["public"] != hashlib.sha256(
        (tmp_path / "audit/public/summary.json").read_bytes()
    ).hexdigest()


def test_real_mmseqs_gates_complete_target_pairs(tmp_path: Path) -> None:
    binary = os.environ.get("MMSEQS_TEST_BINARY") or shutil.which("mmseqs")
    if not binary:
        pytest.skip("Set MMSEQS_TEST_BINARY to run the real target-alignment integration")
    passing = tmp_path / "passing"
    passing.mkdir()
    assert audit.main(_fixture(passing, binary=Path(binary))) == 0
    failing = tmp_path / "failing"
    failing.mkdir()
    argv = _fixture(failing, binary=Path(binary))
    cache = json.loads((failing / "cache.json").read_text())
    cache["0003_A"]["seq"] = "A" * 40 + "C" * 8
    cache["0004_A"]["seq"] = "A" * 40 + "D" * 8
    (failing / "cache.json").write_text(json.dumps(cache))
    assert audit.main(argv) == 1
    report = _report(failing)
    assert all(pair["queries_with_qualifying_hits"] == 1 for pair in report["pairs"].values())
    assert all(pair["exact_sequence_overlap_query_count"] == 0 for pair in report["pairs"].values())
