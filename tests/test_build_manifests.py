from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path


def _cache_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _make_chain_cache(path: Path) -> None:
    # Keep insertion order stable and include enough chains for deterministic sampling.
    payload = {
        "1aaa_A": {"seq_length": 100, "resolution": 2.0, "oligomeric_count": 1, "sequence": "A" * 100},
        "1aaa_B": {"seq_length": 120, "resolution": 2.2, "oligomeric_count": 1, "sequence": "A" * 120},
        "2bbb_A": {"seq_length": 140, "resolution": 2.0, "oligomeric_count": 1, "sequence": "C" * 140},
        "3ccc_A": {"seq_length": 160, "resolution": 1.8, "oligomeric_count": 1, "sequence": "D" * 160},
        "4ddd_A": {"seq_length": 200, "resolution": 2.7, "oligomeric_count": 1, "sequence": "E" * 200},
        "5eee_A": {"seq_length": 220, "resolution": 2.9, "oligomeric_count": 1, "sequence": "F" * 220},
        "6fff_A": {"seq_length": 240, "resolution": 2.5, "oligomeric_count": 1, "sequence": "G" * 240},
    }
    path.write_text(json.dumps(payload))


def _write_structure_metadata(path: Path, chain_ids: list[str], class_by_chain: dict[str, str] | None = None) -> None:
    chains = []
    for chain_id in chain_ids:
        cls = (class_by_chain or {}).get(chain_id, "alpha_beta")
        chains.append(
            {
                "chain_id": chain_id,
                "accepted": True,
                "secondary_structure_class": cls,
                "secondary_helix_fraction": 0.7 if cls == "alpha" else 0.2,
                "secondary_beta_fraction": 0.7 if cls == "beta" else 0.2,
                "secondary_coil_fraction": 0.1,
                "domain_architecture_class": cls,
                "domain_architecture_source": "test",
                "metadata_source_count": 2,
            }
        )
    path.write_text(json.dumps({"schema_version": 1, "chains": chains}))


def _write_cluster_tsv(path: Path, chain_ids: list[str]) -> None:
    lines = []
    for chain_id in chain_ids:
        representative = "1aaa_A" if chain_id in {"1aaa_A", "1aaa_B"} else chain_id
        lines.append(f"{representative}\t{chain_id}")
    path.write_text("\n".join(lines) + "\n")


def _run_build(cache_path: Path, out_dir: Path, expected_sha: str) -> subprocess.CompletedProcess[str]:
    chain_ids = list(json.loads(cache_path.read_text()).keys())
    structure_metadata = cache_path.parent / "structure_metadata.json"
    cluster_tsv = cache_path.parent / "clusters.tsv"
    _write_structure_metadata(structure_metadata, chain_ids)
    _write_cluster_tsv(cluster_tsv, chain_ids)
    return subprocess.run(
        [
            sys.executable,
            "scripts/build_manifests.py",
            "--chain-data-cache",
            str(cache_path),
            "--out-dir",
            str(out_dir),
            "--train-size",
            "3",
            "--val-size",
            "2",
            "--hidden-val-size",
            "1",
            "--seed",
            "0",
            "--structure-metadata",
            str(structure_metadata),
            "--cluster-tsv",
            str(cluster_tsv),
            "--expected-chain-cache-sha256",
            expected_sha,
        ],
        text=True,
        capture_output=True,
        check=False,
    )


def _run_build_without_cluster_tsv(cache_path: Path, out_dir: Path, expected_sha: str) -> subprocess.CompletedProcess[str]:
    chain_ids = list(json.loads(cache_path.read_text()).keys())
    structure_metadata = cache_path.parent / "structure_metadata.json"
    _write_structure_metadata(structure_metadata, chain_ids)
    return subprocess.run(
        [
            sys.executable,
            "scripts/build_manifests.py",
            "--chain-data-cache",
            str(cache_path),
            "--out-dir",
            str(out_dir),
            "--train-size",
            "3",
            "--val-size",
            "2",
            "--hidden-val-size",
            "1",
            "--seed",
            "0",
            "--structure-metadata",
            str(structure_metadata),
            "--expected-chain-cache-sha256",
            expected_sha,
        ],
        text=True,
        capture_output=True,
        check=False,
    )


def _make_balanced_chain_cache(path: Path) -> None:
    payload = {}
    residues = "ARNDCQEGHILK"
    for cls_idx, _cls in enumerate(("alpha", "beta", "alpha_beta")):
        for item_idx in range(4):
            pdb = f"{cls_idx + 1}{item_idx + 1:03d}"
            chain_id = f"{pdb}_A"
            residue = residues[(cls_idx * 4) + item_idx]
            payload[chain_id] = {
                "seq_length": 96 + item_idx,
                "resolution": 1.6 + (0.1 * item_idx),
                "oligomeric_count": 1,
                "sequence": residue * (96 + item_idx),
            }
    path.write_text(json.dumps(payload))


def test_build_manifests_rejects_wrong_cache_sha(tmp_path: Path) -> None:
    cache_path = tmp_path / "chain_data_cache.json"
    _make_chain_cache(cache_path)

    out_dir = tmp_path / "out_bad"
    proc = _run_build(cache_path, out_dir, "0" * 64)
    assert proc.returncode != 0
    assert "SHA256 mismatch" in (proc.stderr + proc.stdout)


def test_build_manifests_deterministic_with_expected_cache_sha(tmp_path: Path) -> None:
    cache_path = tmp_path / "chain_data_cache.json"
    _make_chain_cache(cache_path)
    expected_sha = _cache_sha(cache_path)

    out_a = tmp_path / "out_a"
    out_b = tmp_path / "out_b"
    proc_a = _run_build(cache_path, out_a, expected_sha)
    proc_b = _run_build(cache_path, out_b, expected_sha)
    assert proc_a.returncode == 0, proc_a.stderr
    assert proc_b.returncode == 0, proc_b.stderr

    train_a = (out_a / "train.txt").read_text()
    val_a = (out_a / "val.txt").read_text()
    all_a = (out_a / "all.txt").read_text()

    train_b = (out_b / "train.txt").read_text()
    val_b = (out_b / "val.txt").read_text()
    all_b = (out_b / "all.txt").read_text()

    assert train_a == train_b
    assert val_a == val_b
    assert all_a == all_b


def test_build_manifests_requires_mmseqs_or_locked_cluster_tsv(tmp_path: Path) -> None:
    cache_path = tmp_path / "chain_data_cache.json"
    _make_chain_cache(cache_path)
    expected_sha = _cache_sha(cache_path)

    out_dir = tmp_path / "out_mmseqs"
    proc = _run_build_without_cluster_tsv(cache_path, out_dir, expected_sha)
    assert proc.returncode != 0
    assert "requires MMseqs2 clustering or a locked --cluster-tsv" in (proc.stderr + proc.stdout)


def test_build_manifests_writes_hidden_split_and_stratifies_secondary_classes(tmp_path: Path) -> None:
    cache_path = tmp_path / "chain_data_cache.json"
    _make_balanced_chain_cache(cache_path)
    expected_sha = _cache_sha(cache_path)

    structure_metadata = tmp_path / "structure_metadata.json"
    cluster_tsv = tmp_path / "clusters.tsv"
    chain_ids = []
    class_by_chain = {}
    for cls_idx, cls in enumerate(("alpha", "beta", "alpha_beta")):
        for item_idx in range(4):
            pdb = f"{cls_idx + 1}{item_idx + 1:03d}"
            chain_id = f"{pdb}_A"
            chain_ids.append(chain_id)
            class_by_chain[chain_id] = cls
    _write_structure_metadata(structure_metadata, chain_ids, class_by_chain)
    _write_cluster_tsv(cluster_tsv, chain_ids)

    out_dir = tmp_path / "out_hidden"
    lock_file = tmp_path / "lock.json"
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/build_manifests.py",
            "--chain-data-cache",
            str(cache_path),
            "--out-dir",
            str(out_dir),
            "--train-size",
            "3",
            "--val-size",
            "3",
            "--hidden-val-size",
            "3",
            "--seed",
            "7",
            "--expected-chain-cache-sha256",
            expected_sha,
            "--structure-metadata",
            str(structure_metadata),
            "--cluster-tsv",
            str(cluster_tsv),
            "--lock-file",
            str(lock_file),
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert (out_dir / "hidden_val.txt").exists()
    assert len((out_dir / "train.txt").read_text().splitlines()) == 3
    assert len((out_dir / "val.txt").read_text().splitlines()) == 3
    assert len((out_dir / "hidden_val.txt").read_text().splitlines()) == 3

    lock = json.loads(lock_file.read_text())
    distributions = lock["stratification"]["split_distributions"]
    for split_name in ("train", "val", "hidden_val"):
        classes = distributions[split_name]["secondary_structure_class"]
        assert classes == {"alpha": 1, "alpha_beta": 1, "beta": 1}
    assert lock["grouping"] == {"cluster_disjoint": True, "pdb_disjoint": True}
    assert lock["outputs"]["hidden_val_count"] == 3
    assert (out_dir / "split_quality_report.json").exists()
    assert lock["outputs"]["split_quality_report_sha256"]
