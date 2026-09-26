from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import os
import shutil
import stat
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest


@pytest.fixture
def preparation():
    path = Path(__file__).resolve().parents[1] / "scripts/prepare_openproteinset_roda.py"
    spec = importlib.util.spec_from_file_location("roda_preparation", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _zip(entries: list[tuple[str | zipfile.ZipInfo, bytes]]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, content in entries:
            archive.writestr(name, content)
    return buffer.getvalue()


@pytest.fixture
def snapshot(tmp_path, preparation, monkeypatch):
    source = tmp_path / "remote"
    files = {
        "alignment_data/alignment_dir_roda/1abc_A/uniref90/uniref90_hits.a3m": b">query\nACDE\n",
        "pdb_data/pdb_mmcif.zip": _zip([("mmcif_files/1abc.cif", b"data_1abc\n")]),
        "pdb_data/duplicate_pdb_chains.txt": b"1abc_A 1abc_B\n",
        "pdb_data/data_caches/chain_data_cache.json": b"{}\n",
    }
    for name, content in files.items():
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    lock = tmp_path / "lock.json"
    lock.write_text(json.dumps({"schema_version": 1, "files": {
        name: preparation._file_identity(source / name) for name in files
    }}))
    commands = []

    def download(command, *, check):
        assert check is True and command[:3] == ["aws", "s3", "cp"]
        commands.append(command)
        uri, destination = command[3:5]
        relative, recursive = next((path, recursive) for remote, path, recursive in preparation.DOWNLOADS if remote == uri)
        if recursive:
            shutil.copytree(source / relative, destination, dirs_exist_ok=True)
        else:
            shutil.copyfile(source / relative, destination)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(preparation.subprocess, "run", download)
    return source, lock, commands


def test_setup_preserves_sources_and_receipts_exact_prepared_bytes(tmp_path, preparation, snapshot):
    source, lock, commands = snapshot
    root = tmp_path / "prepared"
    preparation.setup(root, lock)
    assert len(commands) == 4
    receipt = json.loads((root / "source_receipt.json").read_text())
    assert receipt["status"] == "complete"
    assert receipt["source_objects"] == json.loads(lock.read_text())["files"]
    for name, expected in receipt["source_objects"].items():
        assert preparation._file_identity(root / "sources" / name) == expected
    for name, expected in receipt["outputs"].items():
        path = root / name
        if "symlink" in expected:
            assert path.readlink() == Path("1abc_A")
            assert path.resolve().is_dir()
        else:
            assert preparation._file_identity(path) == expected
    assert (root / "pdb_data/mmcif_files/1abc.cif").read_bytes() == b"data_1abc\n"
    assert (root / "alignment_data/alignments/1abc_B/uniref90_hits.a3m").read_bytes() == b">query\nACDE\n"
    with pytest.raises(FileExistsError):
        preparation.setup(root, lock)
    assert len(commands) == 4


@pytest.mark.parametrize("mutation", ["changed", "extra", "missing"])
def test_source_lock_mismatch_retains_download_without_completion(tmp_path, preparation, snapshot, mutation):
    source, lock, _ = snapshot
    path = source / "pdb_data/data_caches/chain_data_cache.json"
    if mutation == "changed":
        path.write_text("changed")
    elif mutation == "extra":
        path.with_name("extra.json").write_text("{}")
    else:
        path.unlink()
    root = tmp_path / "prepared"
    with pytest.raises(ValueError, match="Source lock mismatch"):
        preparation.setup(root, lock)
    assert (root / "sources/pdb_data/pdb_mmcif.zip").is_file()
    assert not (root / "source_receipt.json").exists()
    assert not (root / "pdb_data").exists()


@pytest.mark.parametrize("kind", ["escape", "absolute", "symlink", "duplicate", "case_collision", "file_parent"])
def test_archive_rejects_unsafe_members_before_extraction(tmp_path, preparation, kind):
    entries: list[tuple[str | zipfile.ZipInfo, bytes]] = [("mmcif_files/a.cif", b"data_a")]
    if kind == "escape":
        entries.append(("mmcif_files/../../escaped", b"bad"))
    elif kind == "absolute":
        entries.append(("/escaped", b"bad"))
    elif kind == "symlink":
        link = zipfile.ZipInfo("mmcif_files/link")
        link.create_system = 3
        link.external_attr = (stat.S_IFLNK | 0o777) << 16
        entries.append((link, b"../../escaped"))
    elif kind == "duplicate":
        entries.append(entries[0])
    elif kind == "case_collision":
        entries.extend([("mmcif_files/Sub/a", b"a"), ("mmcif_files/sub/b", b"b")])
    else:
        entries.extend([("mmcif_files/a", b"a"), ("mmcif_files/a/b", b"b")])
    archive = tmp_path / "source.zip"
    with pytest.warns(UserWarning) if kind == "duplicate" else contextlib.nullcontext():
        archive.write_bytes(_zip(entries))
    with pytest.raises(ValueError):
        preparation.extract_mmcif(archive, tmp_path / "output")
    assert not (tmp_path / "output").exists()
    assert not (tmp_path / "escaped").exists()
    assert archive.exists()


def test_flatten_collision_does_not_modify_source_or_existing_output(tmp_path, preparation):
    source = tmp_path / "roda"
    for provider in ["one", "two"]:
        path = source / "1abc_A" / provider / "same.a3m"
        path.parent.mkdir(parents=True)
        path.write_text(provider)
    with pytest.raises(ValueError, match="collision"):
        preparation.flatten(source, tmp_path / "output")
    assert len(list(source.rglob("*.a3m"))) == 2
    assert not (tmp_path / "output").exists()


def test_copy_failure_keeps_sources_and_never_publishes_completion(tmp_path, preparation, snapshot, monkeypatch):
    _, lock, _ = snapshot
    original = preparation._copy_checked

    def fail_on_alignment(source, destination):
        if destination.suffix == ".a3m":
            raise OSError("simulated full disk")
        original(source, destination)

    monkeypatch.setattr(preparation, "_copy_checked", fail_on_alignment)
    root = tmp_path / "prepared"
    with pytest.raises(OSError, match="full disk"):
        preparation.setup(root, lock)
    preparation.verify_sources(root / "sources", json.loads(lock.read_text())["files"])
    assert not (root / "source_receipt.json").exists()


def test_flatten_shell_fails_without_deleting_input(tmp_path):
    source = tmp_path / "roda"
    path = source / "1abc_A/provider/hits.a3m"
    path.parent.mkdir(parents=True)
    path.write_text("retained")
    output = tmp_path / "output"
    (output / "alignments").mkdir(parents=True)
    script = Path(__file__).resolve().parents[1] / "scripts/flatten_roda.sh"
    result = subprocess.run(["sh", str(script), str(source), str(output)], env={**os.environ, "PYTHON": sys.executable}, capture_output=True)
    assert result.returncode != 0
    assert path.read_text() == "retained"


def test_proposed_lock_remains_unaccepted_and_usable_after_explicit_freeze(tmp_path, preparation, snapshot):
    source, _, commands = snapshot
    output = tmp_path / "proposed.json"
    preparation.propose_lock(source, output)
    provenance = json.loads(output.with_name(output.name + ".provenance.json").read_text())
    assert provenance["status"] == "unverified_unaccepted"
    assert json.loads(output.read_text())["status"] == "unverified_unaccepted"
    with pytest.raises(ValueError, match="Source lock"):
        preparation.read_lock(output)
    assert commands == []
    assert not (source / "source_receipt.json").exists()


def test_duplicate_path_escape_rejected_without_symlink(tmp_path, preparation):
    alignments = tmp_path / "alignments"
    (alignments / "1abc_A").mkdir(parents=True)
    duplicates = tmp_path / "duplicates.txt"
    duplicates.write_text("1abc_A ../../escaped\n")
    with pytest.raises(ValueError, match="unsafe"):
        preparation.expand_duplicates(duplicates, alignments)
    assert list(alignments.iterdir()) == [alignments / "1abc_A"]


def test_offline_preparation_uses_explicit_lock_without_redownload(tmp_path, preparation, snapshot):
    source, lock, commands = snapshot
    root = tmp_path / "offline"
    preparation.prepare_local(source, root, lock)
    assert commands == []
    assert json.loads((root / "source_receipt.json").read_text())["source_directory"] == str(source.resolve())
    preparation.verify_sources(source, json.loads(lock.read_text())["files"])


@pytest.mark.parametrize("kind", ["directory_case_prefix", "file_directory_parent", "directory_type", "directory_payload", "unicode_collision", "duplicate_separator"])
def test_explicit_zip_directories_participate_in_collision_validation(tmp_path, preparation, kind):
    entries: list[tuple[str | zipfile.ZipInfo, bytes]] = [("mmcif_files/a.cif", b"data_a")]
    if kind == "directory_case_prefix":
        entries.extend([("mmcif_files/Sub/", b""), ("mmcif_files/sub/b.cif", b"data_b")])
    elif kind == "file_directory_parent":
        entries.extend([("mmcif_files/parent", b"file"), ("mmcif_files/parent/child/", b"")])
    elif kind == "directory_type":
        member = zipfile.ZipInfo("mmcif_files/mismatched/")
        member.create_system = 3
        member.external_attr = (stat.S_IFREG | 0o644) << 16
        entries.append((member, b""))
    elif kind == "directory_payload":
        entries.append(("mmcif_files/hidden/", b"unrepresented bytes"))
    elif kind == "unicode_collision":
        entries.extend([("mmcif_files/\u00e9.cif", b"a"), ("mmcif_files/e\u0301.cif", b"b")])
    else:
        entries.append(("mmcif_files/sub//", b""))
    archive = tmp_path / "source.zip"
    original = _zip(entries)
    archive.write_bytes(original)
    with pytest.raises(ValueError):
        preparation.extract_mmcif(archive, tmp_path / "output")
    assert not (tmp_path / "output").exists()
    assert archive.read_bytes() == original


def test_explicit_shared_zip_directories_remain_valid(tmp_path, preparation):
    archive = tmp_path / "source.zip"
    archive.write_bytes(_zip([
        ("mmcif_files/", b""), ("mmcif_files/sub/", b""),
        ("mmcif_files/sub/a.cif", b"data_a"), ("mmcif_files/sub/b.cif", b"data_b"),
    ]))
    preparation.extract_mmcif(archive, tmp_path / "output")
    assert (tmp_path / "output/mmcif_files/sub/a.cif").read_bytes() == b"data_a"
    assert (tmp_path / "output/mmcif_files/sub/b.cif").read_bytes() == b"data_b"


@pytest.mark.parametrize("kind", ["file_parent", "case_prefix", "unicode"])
def test_impossible_source_lock_fails_before_download_or_output(tmp_path, preparation, snapshot, kind):
    source, lock, commands = snapshot
    record = json.loads(lock.read_text())
    identity = record["files"]["pdb_data/data_caches/chain_data_cache.json"]
    if kind == "file_parent":
        names = ["pdb_data/data_caches/chain_data_cache.json/child"]
    elif kind == "case_prefix":
        names = ["pdb_data/data_caches/Sub/a", "pdb_data/data_caches/sub/b"]
    else:
        names = ["pdb_data/data_caches/\u00e9", "pdb_data/data_caches/e\u0301"]
    record["files"].update({name: identity for name in names})
    lock.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="colli"):
        preparation.setup(tmp_path / "output", lock)
    assert commands == []
    assert not (tmp_path / "output").exists()
    assert (source / "pdb_data/data_caches/chain_data_cache.json").read_bytes() == b"{}\n"


@pytest.mark.parametrize("kind", ["root", "file", "directory"])
def test_flatten_source_symlinks_fail_before_writing(tmp_path, preparation, kind):
    source = tmp_path / "source"
    path = source / "1abc_A/provider/hits.a3m"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"retained")
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "external.a3m").write_bytes(b"untouched")
    if kind == "root":
        alias = tmp_path / "source-alias"
        alias.symlink_to(source, target_is_directory=True)
        source = alias
    elif kind == "file":
        path.with_name("alias.a3m").symlink_to(outside / "external.a3m")
    else:
        (path.parent / "nested").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink|real directory"):
        preparation.flatten(source, tmp_path / "output")
    assert not (tmp_path / "output").exists()
    assert path.read_bytes() == b"retained"
    assert (outside / "external.a3m").read_bytes() == b"untouched"


@pytest.mark.parametrize("nested", [False, True])
def test_flatten_cannot_publish_inside_raw_source_tree(tmp_path, preparation, nested):
    source = tmp_path / "source"
    path = source / "1abc_A/provider/hits.a3m"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"retained")
    output = source / "prepared" if nested else source
    with pytest.raises(ValueError, match="outside the source tree"):
        preparation.flatten(source, output)
    assert path.read_bytes() == b"retained"
    assert not (source / "alignments").exists()
    assert not (source / "prepared").exists()
