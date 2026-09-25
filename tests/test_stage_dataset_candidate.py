from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from scripts import stage_dataset_candidate as staging


def _inputs(tmp_path: Path) -> list[str]:
    cache = tmp_path / "cache.json"
    sources = tmp_path / "sources"
    sources.mkdir()
    rows = {
        f"{index:04d}_A": {
            "seq": "A" * 40 + "".join("C" if index & (1 << bit) else "A" for bit in range(8)),
            "resolution": 2.0, "oligomeric_count": 1,
        }
        for index in range(12)
    }
    cache.write_text(json.dumps(rows))
    cath = sources / "cath-domain-list.txt"
    cath.write_text("".join(f"{index:04d}A00 1 10 20 30 1 1 1 1 1 48 2.0\n" for index in range(12)))
    source_lock = tmp_path / "sources.lock.json"
    source_lock.write_text(json.dumps({
        "schema_version": 2,
        "chain_data_cache": {"sha256": hashlib.sha256(cache.read_bytes()).hexdigest()},
        "sources": {"cath_domain_list": {"path": cath.name, "sha256": hashlib.sha256(cath.read_bytes()).hexdigest()}},
    }))
    cluster_tsv = tmp_path / "clusters.tsv"
    cluster_tsv.write_text("".join(f"{chain}\t{chain}\n" for chain in rows))
    exclusions = tmp_path / "exclusions.txt"
    exclusions.touch()
    salt = tmp_path / "salt.txt"
    salt.write_text("stage-fixture-secret-never-in-public-contract")
    return [
        "--stage-dir", str(tmp_path / "candidate"), "--chain-data-cache", str(cache),
        "--metadata-sources-dir", str(sources), "--metadata-source-lock", str(source_lock),
        "--feature-exclusion-list", str(exclusions), "--processability-exclusion-list", str(exclusions),
        "--cluster-tsv", str(cluster_tsv), "--train-size", "6", "--val-size", "2", "--hidden-val-size", "2",
        "--require-monomer", "--hidden-split-salt-file", str(salt),
    ]


def test_staging_runs_real_builders_and_verifier_without_changing_release(tmp_path: Path) -> None:
    args = staging.parse_args(_inputs(tmp_path))
    frozen = staging.REPO_ROOT / "leaderboard/official_manifest_source.lock.json"
    frozen_bytes = frozen.read_bytes()
    receipt = staging.stage_candidate(args)
    status = json.loads(receipt.read_text())
    assert status["status"] == "manifests_verified"
    assert status["processed_assets_audited"] is False
    for relative_path, digest in status["artifact_sha256"].items():
        assert hashlib.sha256((receipt.parent / relative_path).read_bytes()).hexdigest() == digest
    verification = json.loads((receipt.parent / "manifest-verification.json").read_text())
    assert verification["counts"] == {"train": 6, "val": 2, "hidden_val": 2}
    assert (receipt.parent / "public/train.txt").is_file()
    assert (receipt.parent / "hidden/hidden_val.txt").is_file()
    public_lock = (receipt.parent / "public-source.lock.json").read_text()
    assert "stage-fixture-secret-never-in-public-contract" not in public_lock
    assert "hidden_split_salt_sha256" not in public_lock
    assert frozen.read_bytes() == frozen_bytes
    assert receipt.parent.stat().st_mode & 0o777 == 0o700
    with pytest.raises(FileExistsError):
        staging.stage_candidate(args)


def test_failed_allocation_never_receives_verified_status_and_cannot_be_reused(tmp_path: Path) -> None:
    args = staging.parse_args([*_inputs(tmp_path), "--train-size", "20"])
    with pytest.raises(subprocess.CalledProcessError):
        staging.stage_candidate(args)
    status = json.loads((args.stage_dir / "candidate_status.json").read_text())
    assert status["status"] == "failed"
    assert status["step"] == "manifests"
    assert "artifact_sha256" not in status
    assert not (args.stage_dir / "manifest-verification.json").exists()
    assert not (args.stage_dir / "verification.log").exists()
    with pytest.raises(FileExistsError):
        staging.stage_candidate(args)


def test_corrupt_pinned_metadata_stops_before_manifest_generation(tmp_path: Path) -> None:
    args = staging.parse_args(_inputs(tmp_path))
    (args.metadata_sources_dir / "cath-domain-list.txt").write_text("changed source\n")
    with pytest.raises(subprocess.CalledProcessError):
        staging.stage_candidate(args)
    status = json.loads((args.stage_dir / "candidate_status.json").read_text())
    assert status["status"] == "failed"
    assert status["step"] == "metadata"
    assert not (args.stage_dir / "public").exists()
    assert not (args.stage_dir / "private-source.lock.json").exists()


def test_unignored_repository_stage_is_rejected_before_creating_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = staging.parse_args(_inputs(tmp_path))
    monkeypatch.setattr(staging, "REPO_ROOT", tmp_path)
    with pytest.raises(ValueError, match="hidden artifacts untracked"):
        staging.stage_candidate(args)
    assert not args.stage_dir.exists()
