from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from nanofold.chain_paths import chain_data_dir
from nanofold.representative_eligibility import file_record
from nanofold.source_availability import CODE_FILES, POLICY, validate_source_availability


def write_json(path: Path, value: Any) -> dict[str, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n")
    return file_record(path)


def make_source_availability_case(tmp_path: Path) -> tuple[Path, list[dict[str, Any]], list[str]]:
    """A complete synthetic acquisition, using the actual pinned owning transport."""
    root = Path(__file__).resolve().parents[1]
    raw = tmp_path / "raw"
    journal = raw / "roda_pdb/.s3-acquisition.jsonl"
    journal.parent.mkdir(parents=True)
    chains = ["1abc_B", "1abc_C", "1abc_A"]
    plans = [{"sequence": "ACDE", "sequence_sha256": hashlib.sha256(b"ACDE").hexdigest(),
              "group_id": "fixture_group", "ranked_copies": [
                  {"chain_id": chain, "sequence": "ACDE", "length": 4, "resolution": 2.0,
                   "release_date": "2020-01-01", "group_id": "fixture_group", "cluster40_id": "fixture_cluster"}
                  for chain in chains]}]
    sequence_plan = tmp_path / "sequence-plan.jsonl"
    write_json(sequence_plan, plans[0])
    acquisition_plan = tmp_path / "plan.json"
    write_json(acquisition_plan, {"sequence_plan": file_record(sequence_plan)})
    duplicate_map = tmp_path / "duplicates.txt"
    duplicate_map.write_text("1abc_A 1abc_B 1abc_C\n")
    manifest = tmp_path / "manifest.txt"
    manifest.write_text("1abc_B\n")
    code = {relative: file_record(root / relative) for relative in CODE_FILES}
    binding = {"kind": "raw", "scope": "acquisition_superset_not_finalqualification", "selected_count": 1,
               "selected_manifest": file_record(manifest),
               "files": [file_record(path) for path in (acquisition_plan, duplicate_map, manifest)]
                        + list(code.values())}
    binding_path = tmp_path / "input-bindings.json"
    write_json(binding_path, binding)
    attempt = tmp_path / "attempts/000"
    start = {"source_scope": binding["scope"],
             "argv": ["python", code["scripts/prepare_data.py"]["path"], "--data-root", str(raw),
                      "--manifest", str(manifest), "--duplicate-chains-file", str(duplicate_map),
                      "--no-template-hits", "--download-workers", "8", "--strict-downloads",
                      "--download-transport", "pooled-s3"]}
    write_json(attempt / "start.json", start)
    write_json(attempt / "result.json", {"exit_code": 1})
    events = [{"status": "source_objects_pinned", "bucket": "openfold", "prefix": f"pdb/{chain}/", "objects": []}
              for chain in chains]
    journal.write_text("".join(json.dumps(event) + "\n" for event in events))
    certificate = {"policy": POLICY, "status": "complete", "requested_chain_ids": ["1abc_B"],
                   "raw_root": str(raw), "code": code,
                   **{name: file_record(path) for name, path in {
                       "input_bindings": binding_path, "acquisition_plan": acquisition_plan,
                       "sequence_plan": sequence_plan, "duplicate_map": duplicate_map, "manifest": manifest,
                       "attempt_start": attempt / "start.json", "attempt_result": attempt / "result.json",
                       "journal": journal}.items()}}
    path = tmp_path / "certificate.json"
    write_json(path, certificate)
    return path, plans, [str(raw)]


def refresh_pin(certificate_path: Path, field: str) -> None:
    certificate = json.loads(certificate_path.read_text())
    certificate[field] = file_record(Path(certificate[field]["path"]))
    write_json(certificate_path, certificate)


def test_empty_complete_routes_cover_each_ranked_copy_and_preserve_scope(tmp_path: Path) -> None:
    path, plans, roots = make_source_availability_case(tmp_path)
    records = validate_source_availability(path, plans, roots)
    assert list(records) == ["1abc_B", "1abc_C", "1abc_A"]
    for chain, record in records.items():
        assert record["scope"] == "frozen_successful_filtered_listing"
        assert record["certificate"] == file_record(path)
        assert record["source_routes"][0]["source_chain_id"] == chain
        assert {row["source_chain_id"] for row in record["source_routes"]} == set(records)
    # Final leakage exclusion can remove a copy without changing the source proof.
    reduced = [plans[0] | {"group_id": "revised_group", "ranked_copies": [
        plans[0]["ranked_copies"][1] | {"group_id": "revised_group", "cluster40_id": "revised_cluster"}]}]
    assert list(validate_source_availability(path, reduced, roots)) == ["1abc_C"]
    # A previously unlisted deposited copy cannot inherit absence through sequence identity.
    expanded = [plans[0] | {"ranked_copies": [*plans[0]["ranked_copies"],
                                            plans[0]["ranked_copies"][0] | {"chain_id": "1abc_D"}]}]
    assert "1abc_D" not in validate_source_availability(path, expanded, roots)


@pytest.mark.parametrize("fault", ["missing", "partial", "transport_error", "nonempty", "published", "wrong_bucket",
                                   "unknown_route", "truncated"])
def test_never_certify_unproven_or_contradicted_routes(tmp_path: Path, fault: str) -> None:
    path, plans, roots = make_source_availability_case(tmp_path)
    certificate = json.loads(path.read_text())
    journal = Path(certificate["journal"]["path"])
    events = [json.loads(line) for line in journal.read_text().splitlines()]
    if fault == "missing":
        events.pop()
    elif fault == "partial":
        events[-1]["status"] = "listing_started"
    elif fault == "transport_error":
        events.append({"status": "failed_attempt", "prefix": "pdb/1abc_A/", "source_chain_id": "1abc_A"})
    elif fault == "nonempty":
        events[-1]["objects"] = [{"key": "pdb/1abc_A/uniref90_hits.a3m"}]
    elif fault == "published":
        events[-1]["status"] = "published"
    elif fault == "wrong_bucket":
        events[-1]["bucket"] = "another-bucket"
    elif fault == "unknown_route":
        events[-1]["source_chain_id"] = "1abc_A"
        events[-1]["prefix"] = "unrelated/1abc_A/"
    text = "".join(json.dumps(event) + "\n" for event in events)
    journal.write_text(text.rstrip("\n") if fault == "truncated" else text)
    refresh_pin(path, "journal")
    with pytest.raises(ValueError, match="Source availability"):
        validate_source_availability(path, plans, roots)


@pytest.mark.parametrize("fault", ["changed_journal", "changed_code", "request_policy", "other_manifest",
                                   "unfinished_attempt", "another_attempt", "sequence_identity", "raw_present",
                                   "unbound_raw_root", "omitted_alias", "sequence_bytes", "copy_metadata"])
def test_bound_inputs_and_live_payloads_are_independently_verified(tmp_path: Path, fault: str) -> None:
    path, plans, roots = make_source_availability_case(tmp_path)
    certificate = json.loads(path.read_text())
    if fault == "changed_journal":
        Path(certificate["journal"]["path"]).write_text("")
    elif fault == "changed_code":
        copied = tmp_path / "code/scripts/prepare_data.py"
        copied.parent.mkdir(parents=True)
        copied.write_text("# Arbitrary byte-pinned code is not owning evidence.\n")
        certificate["code"]["scripts/prepare_data.py"] = file_record(copied)
        write_json(path, certificate)
    elif fault == "request_policy":
        start = Path(certificate["attempt_start"]["path"])
        value = json.loads(start.read_text())
        value["argv"][-1] = "aws-cli"
        write_json(start, value)
        refresh_pin(path, "attempt_start")
    elif fault == "other_manifest":
        manifest = tmp_path / "other-manifest.txt"
        manifest.write_text("1abc_B\n")
        certificate["manifest"] = file_record(manifest)
        write_json(path, certificate)
    elif fault == "unfinished_attempt":
        write_json(Path(certificate["attempt_result"]["path"]), {"exit_code": -15})
        refresh_pin(path, "attempt_result")
    elif fault == "another_attempt":
        (tmp_path / "attempts/001").mkdir()
    elif fault == "sequence_identity":
        plans[0]["sequence_sha256"] = "wrong"
    elif fault == "sequence_bytes":
        plans[0]["sequence"] = "AAAA"
    elif fault == "copy_metadata":
        plans[0]["ranked_copies"][0]["resolution"] = 9.0
    elif fault == "raw_present":
        directory = chain_data_dir(Path(roots[0]) / "roda_pdb", "1abc_C")
        directory.mkdir(parents=True)
        (directory / "uniref90_hits.a3m").write_text(">query\nACDE\n")
    elif fault == "unbound_raw_root":
        roots = [str(tmp_path / "different-raw")]
    elif fault == "omitted_alias":
        certificate["source_routes"] = ["1abc_B"]
        write_json(path, certificate)
    with pytest.raises(ValueError, match="[Ss]ource availability"):
        validate_source_availability(path, plans, roots)
