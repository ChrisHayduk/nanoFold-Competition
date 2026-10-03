from __future__ import annotations

import argparse
import errno
import json
import os
from pathlib import Path
from typing import Any

import pytest

from nanofold.chain_paths import chain_data_dir
from nanofold.representative_eligibility import file_record
from scripts import import_structural_raw as importer


def raw_file(root: Path, chain: str, content: str = ">query\nAAAA\n") -> Path:
    path = chain_data_dir(root / "roda_pdb", chain) / "uniref90_hits.a3m"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    return path


def arguments(tmp_path: Path, chains: list[str]) -> argparse.Namespace:
    manifest = tmp_path / "manifest.txt"
    manifest.write_text("\n".join(chains) + "\n")
    return importer.parse_args(["--manifest", str(manifest), "--raw-root", str(tmp_path / "acquired"),
                                "--output", str(tmp_path / "pool")])


def test_pool_is_append_only_with_identical_reuse_and_durable_source_bindings(tmp_path: Path) -> None:
    source = raw_file(tmp_path / "acquired", "1aaa_A")
    original_mode = source.stat().st_mode
    args = arguments(tmp_path, ["1aaa_A"])
    first = importer.run(args)
    destination = Path(first["files"][0]["destination"]["path"])
    assert first["status"] == "imported_unqualified" and first["biological_identity_validated"] is False
    assert destination.stat().st_ino == source.stat().st_ino
    assert source.stat().st_mode == original_mode
    second = importer.run(args)
    assert second["files"][0]["action"] == "reused_identical"
    assert first["receipt"]["path"] != second["receipt"]["path"]
    saved = json.loads(Path(second["receipt"]["path"]).read_text())
    assert saved["files"][0]["chain_id"] == "1aaa_A"
    assert saved["files"][0]["source"]["path"] == str(source)
    assert saved["files"][0]["source"]["sha256"] == file_record(source)["sha256"]
    # A later wave adds another ID under the identical pool root.
    raw_file(tmp_path / "acquired", "2bbb_A", ">query\nCCCC\n")
    third = importer.run(arguments(tmp_path, ["2bbb_A"]))
    assert third["pool_root"] == first["pool_root"]
    assert destination.read_text() == ">query\nAAAA\n"


def test_conflicting_pool_bytes_are_preserved_and_never_overwritten(tmp_path: Path) -> None:
    raw_file(tmp_path / "acquired", "1aaa_A")
    destination = raw_file(tmp_path / "pool", "1aaa_A", ">query\nWRONG\n")
    with pytest.raises(ValueError, match="Conflicting existing"):
        importer.run(arguments(tmp_path, ["1aaa_A"]))
    assert destination.read_text() == ">query\nWRONG\n"
    receipts = list((tmp_path / "pool/imports").glob("*/receipt.json"))
    assert len(receipts) == 1
    assert json.loads(receipts[0].read_text())["status"] == "failed_or_interrupted"


def test_interrupted_staging_never_publishes_partial_file_and_retry_succeeds(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = raw_file(tmp_path / "acquired", "1aaa_A")
    args = arguments(tmp_path, ["1aaa_A"])
    real_link = importer.os.link
    calls = 0

    def interrupted_link(*values: Any, **options: Any) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise InterruptedError("fixture interruption before publication")
        real_link(*values, **options)

    monkeypatch.setattr(importer.os, "link", interrupted_link)
    with pytest.raises(InterruptedError, match="before publication"):
        importer.run(args)
    destination = chain_data_dir(tmp_path / "pool/roda_pdb", "1aaa_A") / "uniref90_hits.a3m"
    assert not destination.exists()
    assert not list(destination.parent.glob("*.pending"))
    assert source.read_text() == ">query\nAAAA\n"
    monkeypatch.setattr(importer.os, "link", real_link)
    assert importer.run(args)["status"] == "imported_unqualified"


def test_crash_after_atomic_link_can_be_recovered_by_identical_reuse(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = raw_file(tmp_path / "acquired", "1aaa_A")
    args = arguments(tmp_path, ["1aaa_A"])
    real_write = importer.write_json

    def fail_chain_record(path: Path, value: object) -> None:
        if path.parent.name == "files":
            raise InterruptedError("fixture crash after publishing complete link")
        real_write(path, value)

    monkeypatch.setattr(importer, "write_json", fail_chain_record)
    with pytest.raises(InterruptedError, match="complete link"):
        importer.run(args)
    destination = chain_data_dir(tmp_path / "pool/roda_pdb", "1aaa_A") / "uniref90_hits.a3m"
    assert destination.read_bytes() == source.read_bytes()
    monkeypatch.setattr(importer, "write_json", real_write)
    assert importer.run(args)["files"][0]["action"] == "reused_identical"


def test_explicit_aliases_are_imported_only_when_selected_and_hash_matches(tmp_path: Path) -> None:
    source = raw_file(tmp_path / "sources", "1aaa_A")
    index = tmp_path / "index.jsonl"
    row = file_record(source) | {"chain_id": "1aaa_A", "alias_chain_ids": ["1aaa_B", "1aaa_C"]}
    index.write_text(json.dumps(row) + "\n")
    manifest = tmp_path / "manifest.txt"
    manifest.write_text("1aaa_A\n1aaa_B\n")
    args = importer.parse_args(["--manifest", str(manifest), "--source-index", str(index), "--output", str(tmp_path / "pool")])
    result = importer.run(args)
    assert {record["chain_id"] for record in result["files"]} == {"1aaa_A", "1aaa_B"}
    alias = next(record for record in result["files"] if record["chain_id"] == "1aaa_B")
    assert alias["source"]["explicit_alias"] is True and alias["source"]["declared_chain_id"] == "1aaa_A"
    assert not chain_data_dir(tmp_path / "pool/roda_pdb", "1aaa_C").exists()
    row["sha256"] = "0" * 64
    index.write_text(json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="Indexed raw source bytes changed"):
        importer.run(args)


def test_cross_filesystem_requires_operator_choice_instead_of_copy_fallback(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = raw_file(tmp_path / "acquired", "1aaa_A")
    destination = tmp_path / "pool/file.a3m"

    def cross_device(*args: object, **kwargs: object) -> None:
        raise OSError(errno.EXDEV, "different devices")

    monkeypatch.setattr(importer.os, "link", cross_device)
    with pytest.raises(ValueError, match="same filesystem"):
        importer.publish_link(file_record(source) | {"bytes": source.stat().st_size}, destination)
    assert not destination.exists()


def test_symlinked_pool_target_is_rejected(tmp_path: Path) -> None:
    raw_file(tmp_path / "acquired", "1aaa_A")
    outside = tmp_path / "outside"
    outside.mkdir()
    (tmp_path / "pool").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="must not contain symlinks"):
        importer.run(arguments(tmp_path, ["1aaa_A"]))
    assert not list(outside.iterdir())


def test_raw_lookup_alias_resolves_inside_source_root_without_full_scan(tmp_path: Path) -> None:
    root = tmp_path / "acquired"
    source = raw_file(root, "1aaa_A")
    alias_dir = chain_data_dir(root / "roda_pdb", "1aaa_B")
    alias_dir.symlink_to(source.parent, target_is_directory=True)
    # A malformed unrelated source is deliberately outside this selected wave.
    raw_file(root, "2bbb_A", "")
    result = importer.run(arguments(tmp_path, ["1aaa_B"]))
    assert result["imported_chain_count"] == 1
    assert Path(result["files"][0]["destination"]["path"]).read_bytes() == source.read_bytes()
    assert os.stat(result["files"][0]["destination"]["path"]).st_ino == source.stat().st_ino


def test_later_in_place_source_mutation_cannot_rewrite_pool_acceptance(tmp_path: Path) -> None:
    source = raw_file(tmp_path / "acquired", "1aaa_A")
    args = arguments(tmp_path, ["1aaa_A"])
    result = importer.run(args)
    binding = result["files"][0]["destination"]
    acceptance_path = Path(result["files"][0]["pool_acceptance"]["path"])
    acceptance_bytes = acceptance_path.read_bytes()
    source.write_text(">query\nCCCC\n")
    # Both links change together, but a later import must still compare them
    # against the original per-chain acceptance rather than adopt new bytes.
    assert Path(binding["path"]).read_bytes() == source.read_bytes()
    with pytest.raises(ValueError, match="Previously accepted raw pool bytes changed"):
        importer.run(args)
    assert acceptance_path.read_bytes() == acceptance_bytes


def test_publication_without_acceptance_binding_recovers_from_verified_source(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = raw_file(tmp_path / "acquired", "1aaa_A")
    args = arguments(tmp_path, ["1aaa_A"])
    real_write = importer.write_json

    def fail_acceptance(path: Path, value: object) -> None:
        if path.parent.name == "accepted":
            raise InterruptedError("fixture crash before immutable acceptance")
        real_write(path, value)

    monkeypatch.setattr(importer, "write_json", fail_acceptance)
    with pytest.raises(InterruptedError, match="before immutable acceptance"):
        importer.run(args)
    destination = chain_data_dir(tmp_path / "pool/roda_pdb", "1aaa_A") / "uniref90_hits.a3m"
    assert destination.read_bytes() == source.read_bytes()
    assert not list((tmp_path / "pool/accepted").glob("*.json"))
    monkeypatch.setattr(importer, "write_json", real_write)
    result = importer.run(args)
    assert result["files"][0]["action"] == "reused_identical"
    acceptance = json.loads(Path(result["files"][0]["pool_acceptance"]["path"]).read_text())
    assert acceptance["destination"]["sha256"] == file_record(source)["sha256"]
