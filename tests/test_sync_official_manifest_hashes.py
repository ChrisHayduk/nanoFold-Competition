from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path


def test_sync_manifest_hashes_preserves_yaml_format_and_writes_relative_lock_paths(tmp_path: Path) -> None:
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
    lock.write_text(json.dumps({"outputs": {}}))
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

    proc = subprocess.run(
        [
            sys.executable,
            "scripts/sync_official_manifest_hashes.py",
            "--manifests-dir",
            str(manifests),
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
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
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

    check_proc = subprocess.run(
        [
            sys.executable,
            "scripts/sync_official_manifest_hashes.py",
            "--manifests-dir",
            str(manifests),
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
            "--check",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert check_proc.returncode == 0, check_proc.stderr
