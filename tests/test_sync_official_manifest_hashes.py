from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.fixture
def sync_case(tmp_path: Path) -> tuple[list[str], dict[str, Path]]:
    manifests = tmp_path / "manifests"
    manifests.mkdir()
    (manifests / "train.txt").write_text("1abc_A\n2def_A\n")
    (manifests / "val.txt").write_text("3ghi_A\n")
    (manifests / "all.txt").write_text("1abc_A\n2def_A\n3ghi_A\n")

    track = tmp_path / "track.yaml"
    track.write_text(
        "\n".join(
            [
                "id: limited",
                "dataset:",
                "  train_manifest_sha256: " + "0" * 64,
                "  val_manifest_sha256: " + "0" * 64,
                "  all_manifest_sha256: " + "0" * 64,
                "  hidden_manifest_sha256: null",
                "  train_chain_count: 0",
                "  val_chain_count: 0",
                "  hidden_chain_count: 1000",
                "scoring:",
                "  foldscore_weights:",
                "    gdt_ha_ca: 0.25",
                "    lddt_atom14: 0.09375",
                "    cad_atom14: 0.09375",
                "    sg_atom14: 0.09375",
                "    sc_atom14: 0.09375",
                "    molprobity_clash_atom14: 0.125",
                "    bb_atom14: 0.125",
                "    dipdiff_atom14: 0.125",
            ]
        )
        + "\n"
    )
    lock = tmp_path / "lock.json"
    lock.write_text(json.dumps({
        "sampling_policy": "cath-v3",
        "outputs": {
            **{f"{split}_manifest_sha256": hashlib.sha256((manifests / f"{split}.txt").read_bytes()).hexdigest()
               for split in ("train", "val", "all")},
            "train_count": 2,
            "val_count": 1,
            "unique_count": 3,
        },
    }))
    fingerprint = tmp_path / "fingerprint.json"
    fingerprint.write_text(
        json.dumps(
            {
                "source_lock_path": "old-lock.json",
                "source_lock_sha256": "0" * 64,
            }
        )
        + "\n"
    )
    readme = tmp_path / "README.md"
    readme.write_text(
        "- `train.txt`: `" + "0" * 64 + "`\n"
        "- `val.txt`: `" + "0" * 64 + "`\n"
        "- `all.txt`: `" + "0" * 64 + "`\n"
    )
    competition = tmp_path / "COMPETITION.md"
    competition.write_text(
        "- train: `" + "0" * 64 + "`\n"
        "- val: `" + "0" * 64 + "`\n"
        "- all: `" + "0" * 64 + "`\n"
    )

    command = [
        sys.executable,
        "scripts/sync_official_manifest_hashes.py",
        "--manifests-dir",
        str(manifests),
        "--hidden-manifest",
        str(tmp_path / "hidden.txt"),
        "--track-file",
        str(track),
        "--lock-file",
        str(lock),
        "--fingerprint-file",
        str(fingerprint),
        "--readme",
        str(readme),
        "--competition-doc",
        str(competition),
    ]
    return command, {
        "manifests": manifests, "track": track, "lock": lock, "fingerprint": fingerprint,
        "readme": readme, "competition": competition, "hidden": tmp_path / "hidden.txt",
    }


def _run(command: list[str], *options: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run([*command, *options], text=True, capture_output=True, check=False)


def _outputs(files: dict[str, Path]) -> dict[Path, bytes]:
    return {files[key]: files[key].read_bytes()
            for key in ("track", "lock", "fingerprint", "readme", "competition")}


def test_sync_requires_explicit_write_to_repair_locked_references(sync_case: tuple[list[str], dict[str, Path]]) -> None:
    command, files = sync_case
    before = _outputs(files)
    for options in ((), ("--check",)):
        proc = _run(command, *options)
        assert proc.returncode == 1, proc.stderr
        assert "--write" in proc.stdout
        assert _outputs(files) == before

    proc = _run(command, "--write")
    assert proc.returncode == 0, proc.stderr
    track, lock, fingerprint = (files[key] for key in ("track", "lock", "fingerprint"))
    assert "gdt_ha_ca: 0.25" in track.read_text()
    assert "dipdiff_atom14: 0.125" in track.read_text()

    lock_obj = json.loads(lock.read_text())
    outputs = lock_obj["outputs"]
    assert not Path(outputs["train_manifest"]).is_absolute()
    assert outputs["train_count"] == 2
    assert outputs["val_count"] == 1
    fingerprint_obj = json.loads(fingerprint.read_text())
    assert fingerprint_obj["source_lock_path"] == "lock.json"
    assert fingerprint_obj["source_lock_sha256"] == hashlib.sha256(lock.read_bytes()).hexdigest()

    after = _outputs(files)
    for options in ((), ("--check",)):
        check_proc = _run(command, *options)
        assert check_proc.returncode == 0, check_proc.stderr
        assert _outputs(files) == after


@pytest.mark.parametrize("changed", ["train", "val", "all", "train_count", "val_count", "unique_count",
                                     "missing_bindings", "boolean_count", "missing_lock"])
def test_sync_cannot_rebind_existing_provenance(
    sync_case: tuple[list[str], dict[str, Path]], changed: str,
) -> None:
    command, files = sync_case
    if changed in ("train", "val", "all"):
        # Even a formatting-only change produces bytes not bound by the source lock.
        path = files["manifests"] / f"{changed}.txt"
        path.write_text(path.read_text() + "\n")
    elif changed == "missing_lock":
        files["lock"].unlink()
    else:
        lock = json.loads(files["lock"].read_text())
        if changed == "missing_bindings":
            lock["outputs"].pop("train_manifest_sha256")
        elif changed == "boolean_count":
            lock["outputs"]["val_count"] = True
        else:
            lock["outputs"][changed] += 1
        files["lock"].write_text(json.dumps(lock))
    before = {path: path.read_bytes() for path in files.values() if path.is_file()}
    for options in ((), ("--write",)):
        proc = _run(command, *options)
        assert proc.returncode != 0
        assert "fresh candidate" in proc.stderr
        assert all(path.read_bytes() == content for path, content in before.items())
        if changed == "missing_lock":
            assert not files["lock"].exists()


@pytest.mark.parametrize("changed", ["bytes", "count", "missing_binding", "missing_file", None])
def test_hidden_sync_requires_existing_matching_binding(
    sync_case: tuple[list[str], dict[str, Path]], changed: str | None,
) -> None:
    command, files = sync_case
    files["hidden"].write_text("4jkl_A\n")
    lock = json.loads(files["lock"].read_text())
    lock["outputs"].update({
        "hidden_val_manifest_sha256": hashlib.sha256(files["hidden"].read_bytes()).hexdigest(),
        "hidden_val_count": 1,
    })
    if changed == "bytes":
        files["hidden"].write_text("5mno_A\n")
    elif changed == "count":
        lock["outputs"]["hidden_val_count"] = 2
    elif changed == "missing_binding":
        lock["outputs"].pop("hidden_val_manifest_sha256")
    elif changed == "missing_file":
        files["hidden"].unlink()
    files["lock"].write_text(json.dumps(lock))
    before = _outputs(files)
    proc = _run(command, "--include-hidden", "--write")
    if changed is None:
        assert proc.returncode == 0, proc.stderr
        outputs = json.loads(files["lock"].read_text())["outputs"]
        assert outputs["hidden_val_manifest_sha256"] == hashlib.sha256(files["hidden"].read_bytes()).hexdigest()
        assert outputs["hidden_val_count"] == 1
    else:
        assert proc.returncode != 0
        assert _outputs(files) == before


def test_sync_modes_are_mutually_exclusive(sync_case: tuple[list[str], dict[str, Path]]) -> None:
    command, files = sync_case
    before = _outputs(files)
    proc = _run(command, "--check", "--write")
    assert proc.returncode == 2
    assert "not allowed with argument" in proc.stderr
    assert _outputs(files) == before
