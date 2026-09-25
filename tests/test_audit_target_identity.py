from __future__ import annotations

import hashlib
import io
import json
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from nanofold.chain_paths import chain_data_dir, chain_npz_path
from scripts import audit_target_identity as audit

SEQUENCE = "ARNDCQEGHILKMFPSTWYV" * 2
CHAIN = "0001_A"


def arrays(chain: str = CHAIN) -> dict[str, Any]:
    tokens = np.asarray([audit.TOKENS[aa] for aa in SEQUENCE], dtype=np.int32)
    return {
        "aatype": tokens, "msa": np.stack([tokens, tokens]), "chain_id": np.asarray(chain),
        "deletions": np.zeros((2, len(tokens)), np.int32),
        "template_aatype": np.zeros((0, len(tokens)), np.int32),
        "template_ca_coords": np.zeros((0, len(tokens), 3), np.float32),
        "template_ca_mask": np.zeros((0, len(tokens)), bool),
    }


def fixture(tmp_path: Path) -> list[str]:
    (tmp_path / "cache.json").write_text(json.dumps({CHAIN: {"seq": SEQUENCE, "seq_length": len(SEQUENCE)}}))
    (tmp_path / "manifest.txt").write_text(CHAIN + "\n")
    source = chain_data_dir(tmp_path / "raw/roda_pdb", CHAIN)
    source.mkdir(parents=True)
    (source / "uniref90_hits.a3m").write_text(f">query\n{SEQUENCE}\n>homolog\n{SEQUENCE}\n")
    (tmp_path / "features").mkdir()
    np.savez(chain_npz_path(tmp_path / "features", CHAIN), **arrays())
    return ["--chain-data-cache", str(tmp_path / "cache.json"), "--manifest", str(tmp_path / "manifest.txt"),
            "--expected-counts", "1", "--raw-root", str(tmp_path / "raw"),
            "--features-dir", str(tmp_path / "features"), "--out-dir", str(tmp_path / "audit")]


def report(tmp_path: Path) -> dict[str, Any]:
    return json.loads((tmp_path / "audit/summary.json").read_text())


def test_complete_gate_binds_assets_inputs_and_hides_terminal_identifiers(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    args = fixture(tmp_path)
    assert audit.main(args) == 0
    result = report(tmp_path)
    assert result["status"] == "passed"
    assert result["selected_chain_count"] == result["passed_chain_count"] == 1
    assert result["raw_rows"] == 2 and result["raw_bytes"] > 80
    assert result["raw_audited"] and result["processed_audited"] and not result["labels_audited"]
    for item in result["inputs"].values():
        assert hashlib.sha256(Path(item["path"]).read_bytes()).hexdigest() == item["sha256"]
    row = result["records"][0]
    for item in [row["processed"], *row["raw"].values()]:
        assert hashlib.sha256(Path(item["path"]).read_bytes()).hexdigest() == item["sha256"]
    assert CHAIN not in capsys.readouterr().out
    assert (tmp_path / "audit").stat().st_mode & 0o777 == 0o700
    assert (tmp_path / "audit/summary.json").stat().st_mode & 0o077 == 0
    before = (tmp_path / "audit/summary.json").read_bytes()
    assert audit.main(args) == 2
    assert (tmp_path / "audit/summary.json").read_bytes() == before


@pytest.mark.parametrize("text", [
    "", "AAAA\n", ">\nAAAA\n", ">query\n", ">query\nAAAA*\n", ">query\nAAAA\n>other\nAAA\n",
    ">query\naaaa\n", ">query\nAA\u00c4A\n", ">query\nAAAA\n",
])
def test_raw_parser_rejects_incomplete_malformed_or_wrong_targets(tmp_path: Path, text: str) -> None:
    path = tmp_path / "input.a3m"
    path.write_text(text)
    with pytest.raises(ValueError):
        audit.raw_record(path, SEQUENCE)


def test_raw_parser_handles_multiline_insertions_and_query_gap_columns(tmp_path: Path) -> None:
    path = tmp_path / "input.a3m"
    path.write_text(">query\nAR-\nND\n>row\nAaaR-ND\n")
    result = audit.raw_record(path, "ARND")
    assert result["rows"] == 2 and result["aligned_width"] == 5 and result["query_length"] == 4


@pytest.mark.parametrize("mutation", ["float_target", "float_msa", "wrong_target", "wrong_query", "empty_msa", "shape",
                                         "wrong_chain", "chain_array", "object", "nonfinite", "out_of_range", "missing_key"])
def test_processed_assets_reject_identity_schema_and_unsafe_values(tmp_path: Path, mutation: str) -> None:
    args = fixture(tmp_path)
    values = arrays()
    if mutation == "float_target":
        values["aatype"] = values["aatype"].astype(np.float32)
    elif mutation == "float_msa":
        values["msa"] = values["msa"].astype(np.float32)
    elif mutation == "wrong_target":
        values["aatype"][0] = 1
    elif mutation == "wrong_query":
        values["msa"][0, 0] = 1
    elif mutation == "empty_msa":
        values["msa"] = values["msa"][:0]
    elif mutation == "shape":
        values["aatype"] = values["aatype"][None]
    elif mutation == "wrong_chain":
        values["chain_id"] = np.asarray("0002_A")
    elif mutation == "chain_array":
        values["chain_id"] = np.asarray([CHAIN])
    elif mutation == "object":
        values["unused_metadata"] = np.asarray({"unsafe": 1}, dtype=object)
    elif mutation == "nonfinite":
        values["unused_metadata"] = np.asarray(float("nan"))
    elif mutation == "out_of_range":
        values["msa"][1, 0] = 200
    else:
        del values["deletions"]
    np.savez(chain_npz_path(tmp_path / "features", CHAIN), **values)
    assert audit.main(args) == 1
    assert report(tmp_path)["failure_count"] == 1


@pytest.mark.parametrize("malformation", ["duplicate", "nonarray", "nested", "broken", "aliased_object"])
def test_rejects_ambiguous_or_malformed_zip_members(tmp_path: Path, malformation: str) -> None:
    args = fixture(tmp_path)
    path = chain_npz_path(tmp_path / "features", CHAIN)
    if malformation == "broken":
        path.write_bytes(b"not a ZIP")
    else:
        with zipfile.ZipFile(path, "a") as archive:
            if malformation == "duplicate":
                value = io.BytesIO()
                np.save(value, arrays()["aatype"])
                with pytest.warns(UserWarning, match="Duplicate name"):
                    archive.writestr("aatype.npy", value.getvalue())
            elif malformation == "aliased_object":
                value = io.BytesIO()
                np.save(value, np.asarray({"unsafe": 1}, dtype=object))
                archive.writestr("aatype.npy.npy", value.getvalue())
            else:
                archive.writestr("nested/a.npy" if malformation == "nested" else "metadata.txt", "bad")
    assert audit.main(args) == 1


@pytest.mark.parametrize("case", ["missing_raw", "ambiguous_raw", "missing_npz", "ambiguous_npz", "escaped_raw", "escaped_npz"])
def test_assets_must_be_present_unique_and_under_bound_roots(tmp_path: Path, case: str) -> None:
    args = fixture(tmp_path)
    raw = chain_data_dir(tmp_path / "raw/roda_pdb", CHAIN) / "uniref90_hits.a3m"
    feature = chain_npz_path(tmp_path / "features", CHAIN)
    if case == "missing_raw":
        raw.unlink()
    elif case == "ambiguous_raw":
        (raw.parent / "a3m").mkdir()
        (raw.parent / "a3m" / raw.name).write_bytes(raw.read_bytes())
    elif case == "missing_npz":
        feature.unlink()
    elif case == "ambiguous_npz":
        (tmp_path / "features2").mkdir()
        chain_npz_path(tmp_path / "features2", CHAIN).write_bytes(feature.read_bytes())
        args += ["--features-dir", str(tmp_path / "features2")]
    else:
        path = raw if case == "escaped_raw" else feature
        outside = tmp_path / "outside"
        path.rename(outside)
        path.symlink_to(outside)
    assert audit.main(args) == 1


@pytest.mark.parametrize("case", ["wrong_count", "duplicate_chain", "overlap", "uppercase_pdb", "bad_alias", "bad_length",
                                         "duplicate_json", "nonfinite_cache"])
def test_input_contract_cannot_be_satisfied_by_silent_normalization(tmp_path: Path, case: str) -> None:
    args = fixture(tmp_path)
    if case == "wrong_count":
        args[args.index("--expected-counts") + 1] = "2"
    elif case == "duplicate_chain":
        (tmp_path / "manifest.txt").write_text((CHAIN + "\n") * 2)
    elif case == "overlap":
        args += ["--manifest", str(tmp_path / "manifest.txt")]
        args[args.index("--expected-counts") + 1:args.index("--expected-counts") + 2] = ["1", "1"]
    elif case == "uppercase_pdb":
        (tmp_path / "manifest.txt").write_text("1ABC_A\n")
    elif case == "duplicate_json":
        (tmp_path / "cache.json").write_text('{"same": 1, "same": 2}')
    elif case == "nonfinite_cache":
        (tmp_path / "cache.json").write_text('{"bad": NaN}')
    else:
        cache = json.loads((tmp_path / "cache.json").read_text())
        if case == "bad_alias":
            cache[CHAIN]["sequence"] = "C" * 40
        else:
            cache[CHAIN]["seq_length"] = 40.0
        (tmp_path / "cache.json").write_text(json.dumps(cache))
    assert audit.main(args) in {1, 2}
    assert report(tmp_path)["status"] != "passed"


@pytest.mark.parametrize("changed", ["asset", "cache", "manifest"])
def test_end_to_end_rehash_detects_changes_after_successful_target_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, changed: str,
) -> None:
    args = fixture(tmp_path)
    original = audit.processed_record

    def mutate(path: Path, expected: str, chain: str) -> dict[str, Any]:
        result = original(path, expected, chain)
        target = {"asset": path, "cache": tmp_path / "cache.json", "manifest": tmp_path / "manifest.txt"}[changed]
        with target.open("ab") as handle:
            handle.write(b" ")
        return result

    monkeypatch.setattr(audit, "processed_record", mutate)
    assert audit.main(args) == 2
    assert report(tmp_path)["status"] == "error"


@pytest.mark.parametrize("source", ["cache", "manifest"])
def test_reads_bound_resolved_input_even_if_original_symlink_is_retargeted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, source: str,
) -> None:
    args = fixture(tmp_path)
    path = tmp_path / ("cache.json" if source == "cache" else "manifest.txt")
    pinned = path.with_name("pinned-" + path.name)
    path.rename(pinned)
    path.symlink_to(pinned)
    alternate = path.with_name("alternate-" + path.name)
    alternate.write_text(json.dumps({CHAIN: {"seq": "C" * 40}}) if source == "cache" else "0002_A\n")
    original = audit.sha256
    changed = False

    def retarget(input_path: Path) -> str:
        nonlocal changed
        digest = original(input_path)
        if input_path == pinned and not changed:
            path.unlink()
            path.symlink_to(alternate)
            changed = True
        return digest

    monkeypatch.setattr(audit, "sha256", retarget)
    assert audit.main(args) == 0
    key = "chain_data_cache" if source == "cache" else "manifest_0"
    assert report(tmp_path)["inputs"][key]["path"] == str(pinned)


@pytest.mark.parametrize("source", ["raw", "processed"])
def test_final_inventory_rejects_new_ambiguous_sources(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, source: str,
) -> None:
    args = fixture(tmp_path)
    if source == "processed":
        (tmp_path / "other-features").mkdir()
        args += ["--features-dir", str(tmp_path / "other-features")]
    original = audit.processed_record

    def add_ambiguous_asset(path: Path, expected: str, chain: str) -> dict[str, Any]:
        record = original(path, expected, chain)
        if source == "raw":
            raw = chain_data_dir(tmp_path / "raw/roda_pdb", chain) / "uniref90_hits.a3m"
            (raw.parent / "extra").mkdir()
            (raw.parent / "extra" / raw.name).write_bytes(raw.read_bytes())
        else:
            chain_npz_path(tmp_path / "other-features", chain).write_bytes(path.read_bytes())
        return record

    monkeypatch.setattr(audit, "processed_record", add_ambiguous_asset)
    assert audit.main(args) == 2
    assert report(tmp_path)["status"] == "error"
