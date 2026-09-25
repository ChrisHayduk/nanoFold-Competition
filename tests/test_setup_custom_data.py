from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import gemmi
import numpy as np
import pytest

from nanofold.chain_paths import chain_data_dir, chain_npz_path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts/setup_custom_data.sh"


def _fixture(tmp_path: Path) -> tuple[list[str], Path, Path]:
    manifests = tmp_path / "candidate public"
    manifests.mkdir()
    contents = {"train": "1abc_A\n", "val": "2abc_A\n", "all": "1abc_A\n2abc_A\n"}
    outputs: dict[str, object] = {"train_count": 1, "val_count": 1, "unique_count": 2}
    for split, content in contents.items():
        path = manifests / f"{split}.txt"
        path.write_text(content)
        outputs[f"{split}_manifest_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    lock = tmp_path / "public-source.lock.json"
    cache = tmp_path / "chain_data_cache.json"
    cache.write_text(json.dumps({"1abc_A": {"sequence": "ACD"}, "2abc_A": {"sequence": "EFG"}}))
    lock.write_text(json.dumps({"outputs": outputs, "chain_data_cache_sha256": hashlib.sha256(cache.read_bytes()).hexdigest()}))
    args = [
        "--manifests-dir", str(manifests), "--manifest-lock-file", str(lock),
        "--data-root", str(tmp_path / "raw"), "--processed-features-dir", str(tmp_path / "features"),
        "--processed-labels-dir", str(tmp_path / "labels"),
        "--chain-data-cache", str(cache),
    ]
    return args, manifests, lock


def _stub_commands(tmp_path: Path) -> tuple[dict[str, str], Path]:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    log = tmp_path / "commands.log"
    python = fake_bin / "python"
    python.write_text(
        "#!/bin/bash\n"
        'if [[ "$1" == "-" || "$1" == "-c" ]]; then exec "$REAL_PYTHON" "$@"; fi\n'
        'printf \'%s\\n\' "$*" >> "$COMMAND_LOG"\n'
    )
    python.chmod(0o755)
    aws = fake_bin / "aws"
    aws.write_text('#!/bin/bash\nprintf \'aws %s\\n\' "$*" >> "$COMMAND_LOG"\n')
    aws.chmod(0o755)
    return dict(os.environ, PATH=f"{fake_bin}:{os.environ['PATH']}", REAL_PYTHON=sys.executable,
                COMMAND_LOG=str(log)), log


def _run(args: list[str], env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["bash", str(SCRIPT), *args], cwd=REPO_ROOT, env=env,
                          capture_output=True, text=True, check=False)


def test_custom_dry_run_uses_locked_manifests_without_writes(tmp_path: Path) -> None:
    args, manifests, lock = _fixture(tmp_path)
    env, log = _stub_commands(tmp_path)
    before = {path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    proc = _run([*args, "--dry-run"], env)
    assert proc.returncode == 0, proc.stderr
    assert f"--manifest {manifests}/all.txt" in proc.stdout
    assert "--download-mmcif-subset --strict-downloads" in proc.stdout
    assert "--disable-templates" in proc.stdout
    assert "build_manifests.py" not in proc.stdout
    assert "pdb_mmcif.zip" not in proc.stdout
    assert {path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()} == before
    assert not log.exists()
    assert not (tmp_path / "raw").exists()
    assert not (tmp_path / "features").exists()
    assert not (tmp_path / "labels").exists()
    assert lock.read_bytes() == before[lock]


def test_custom_setup_preprocesses_locked_public_union_once_for_complete_audit(tmp_path: Path) -> None:
    args, manifests, lock = _fixture(tmp_path)
    env, log = _stub_commands(tmp_path)
    originals = {path: path.read_bytes() for path in [*manifests.iterdir(), lock]}
    filter_path = tmp_path / "private-filter.json"
    filter_path.write_text("{}")
    proc = _run([*args, "--msa-row-filter", str(filter_path), "--resume-preprocess"], env)
    assert proc.returncode == 0, proc.stderr
    commands = log.read_text().splitlines()
    downloads = [line for line in commands if "prepare_data.py" in line]
    assert len(downloads) == 1
    assert f"--manifest {manifests}/all.txt" in downloads[0]
    preprocessing = [line for line in commands if "preprocess.py" in line]
    assert len(preprocessing) == 1
    assert f"--manifest {manifests}/all.txt" in preprocessing[0]
    assert all(f"--msa-row-filter {filter_path} --skip-existing" in command for command in preprocessing)
    assert all(f"--chain-data-cache {tmp_path / 'chain_data_cache.json'}" in command for command in preprocessing)
    assert all(path.read_bytes() == content for path, content in originals.items())


def test_custom_setup_writes_complete_public_msa_filter_audit(tmp_path: Path) -> None:
    args, manifests, lock = _fixture(tmp_path)
    env, _log = _stub_commands(tmp_path)
    python = tmp_path / "bin/python"
    python.write_text(
        '#!/bin/bash\nif [[ "$1" == "-" || "$1" == "-c" ]]; then exec "$REAL_PYTHON" "$@"; fi\n'
        'printf \'%s\\n\' "$*" >> "$COMMAND_LOG"\n'
        'if [[ "$1" == */preprocess.py ]]; then exec "$REAL_PYTHON" "$@"; fi\n'
    )
    sequences = {"1abc_A": "A" * 40, "2abc_A": "R" * 40}
    cache = tmp_path / "chain_data_cache.json"
    cache.write_text(json.dumps({chain: {"sequence": sequence} for chain, sequence in sequences.items()}))
    source_lock = json.loads(lock.read_bytes())
    source_lock["chain_data_cache_sha256"] = hashlib.sha256(cache.read_bytes()).hexdigest()
    lock.write_text(json.dumps(source_lock))
    mmcifs = tmp_path / "raw/pdb_data/mmcif_files"
    mmcifs.mkdir(parents=True)
    excluded = "G" * 40
    for chain_id, sequence in sequences.items():
        alignment = chain_data_dir(tmp_path / "raw/roda_pdb", chain_id)
        alignment.mkdir(parents=True)
        (alignment / "uniref90_hits.a3m").write_text(
            f">query\n{sequence}\n>excluded\n{excluded}\n>retained\n{sequence[:-1]}-\n"
        )
        structure = gemmi.Structure()
        model = gemmi.Model(1)
        chain = gemmi.Chain("A")
        for index in range(len(sequence)):
            residue = gemmi.Residue()
            residue.name = "ALA" if sequence[index] == "A" else "ARG"
            residue.seqid = gemmi.SeqId(index + 1, " ")
            residue.label_seq = index + 1
            residue.entity_type = gemmi.EntityType.Polymer
            atom = gemmi.Atom()
            atom.name = "CA"
            atom.element = gemmi.Element("C")
            atom.pos = gemmi.Position(index * 3.8, 0, 0)
            atom.occ = 1
            residue.add_atom(atom)
            chain.add_residue(residue)
        model.add_chain(chain)
        structure.add_model(model)
        structure.setup_entities()
        structure.make_mmcif_document().write_file(str(mmcifs / f"{chain_id[:4]}.cif"))
    filter_path = tmp_path / "private-filter.json"
    filter_path.write_text(json.dumps({"excluded_sequence_sha256": [hashlib.sha256(excluded.encode()).hexdigest()]}))
    result = _run([*args, "--msa-row-filter", str(filter_path)], env)
    assert result.returncode == 0, result.stderr
    features = tmp_path / "features"
    audit = json.loads((features / "msa_row_filter_audit.json").read_bytes())
    assert audit["manifest"] == str(manifests / "all.txt")
    assert audit["chain_count"] == 2
    assert {row["chain_id"] for row in audit["per_chain"]} == set(sequences)
    assert audit["total_msa_rows_before_filter"] == 6
    assert audit["total_msa_rows_removed"] == 2
    assert audit["total_msa_rows_after_filter"] == 4
    for chain_id in sequences:
        with np.load(chain_npz_path(features, chain_id)) as arrays:
            assert arrays["msa"].shape == (2, 40)
        with np.load(chain_npz_path(tmp_path / "labels", chain_id)) as arrays:
            assert np.count_nonzero(arrays["ca_mask"]) == 40


@pytest.mark.parametrize("broken", ["train", "val", "all", "missing_union"])
def test_invalid_locked_manifest_stops_before_downloads_or_writes(tmp_path: Path, broken: str) -> None:
    args, manifests, _lock = _fixture(tmp_path)
    env, log = _stub_commands(tmp_path)
    if broken == "missing_union":
        (manifests / "all.txt").unlink()
    else:
        (manifests / f"{broken}.txt").write_text("3abc_A\n")
    proc = _run(args, env)
    assert proc.returncode != 0
    assert "source lock" in proc.stderr or "all.txt is never generated" in proc.stderr
    assert not log.exists()
    assert not (tmp_path / "raw").exists()
    assert not (tmp_path / "features").exists()
    assert not (tmp_path / "labels").exists()


def test_hash_bound_but_incorrect_union_is_rejected(tmp_path: Path) -> None:
    args, manifests, lock = _fixture(tmp_path)
    env, log = _stub_commands(tmp_path)
    union = manifests / "all.txt"
    union.write_text("1abc_A\n3abc_A\n")
    source = json.loads(lock.read_text())
    source["outputs"]["all_manifest_sha256"] = hashlib.sha256(union.read_bytes()).hexdigest()
    lock.write_text(json.dumps(source))
    proc = _run(args, env)
    assert proc.returncode != 0
    assert "exact train/val union" in proc.stderr
    assert not log.exists()
    assert not (tmp_path / "raw").exists()


def test_missing_custom_path_and_generation_flags_fail_before_work(tmp_path: Path) -> None:
    args, _manifests, _lock = _fixture(tmp_path)
    env, log = _stub_commands(tmp_path)
    args_without_data = args[:4] + args[6:]
    proc = _run(args_without_data, env)
    assert proc.returncode != 0
    assert "Explicit custom" in proc.stderr
    proc = _run([*args, "--train-size", "100"], env)
    assert proc.returncode != 0
    assert "stage_dataset_candidate.py" in proc.stderr
    assert not log.exists()
    assert not (tmp_path / "raw").exists()


def test_custom_outputs_cannot_alias_published_assets(tmp_path: Path) -> None:
    args, _manifests, _lock = _fixture(tmp_path)
    env, log = _stub_commands(tmp_path)
    alias = tmp_path / "public-data-alias"
    alias.symlink_to(REPO_ROOT / "data", target_is_directory=True)
    proc = _run([*args, "--processed-features-dir", str(alias / "processed_features")], env)
    assert proc.returncode != 0
    assert "separate from published paths" in proc.stderr
    assert not log.exists()
    assert not (tmp_path / "raw").exists()


@pytest.mark.parametrize("broken", ["changed", "missing", "omitted", "unlocked"])
def test_custom_cache_binding_fails_before_downloads_or_writes(tmp_path: Path, broken: str) -> None:
    args, _manifests, lock = _fixture(tmp_path)
    env, log = _stub_commands(tmp_path)
    cache = tmp_path / "chain_data_cache.json"
    if broken == "changed":
        cache.write_text("{}")
    elif broken == "missing":
        cache.unlink()
    elif broken == "omitted":
        args = args[:-2]
    else:
        source = json.loads(lock.read_text())
        source.pop("chain_data_cache_sha256")
        lock.write_text(json.dumps(source))
    result = _run(args, env)
    assert result.returncode != 0
    assert "cache" in result.stderr
    assert not log.exists()
    assert not (tmp_path / "raw").exists()
    assert not (tmp_path / "features").exists()
