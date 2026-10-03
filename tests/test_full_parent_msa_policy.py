from __future__ import annotations

import hashlib
import json
import os
import random
import shutil
import sys
from pathlib import Path

import pytest

from nanofold.chain_paths import chain_data_dir
from nanofold.msa_filter_policy import FULL_PARENT_POLICY_TYPE, build_msa_filter_policy, load_msa_filter_policy
from scripts import build_msa_row_filter
from scripts.preprocess import _read_merged_msa


@pytest.mark.skipif(not (os.environ.get("NANOFOLD_MMSEQS_TEST_BINARY") or shutil.which("mmseqs")), reason="Native MMseqs test binary not configured")
def test_native_long_msa_embedded_heldout_domain_is_removed_by_verified_policy(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    binary = os.environ.get("NANOFOLD_MMSEQS_TEST_BINARY") or shutil.which("mmseqs")
    assert binary is not None
    rng = random.Random(102)
    alphabet = "ARNDCQEGHILKMFPSTWYV"

    def sequence(length: int) -> str:
        return "".join(rng.choice(alphabet) for _ in range(length))

    hidden, validation, train = sequence(64), sequence(80), sequence(400)
    embedded = sequence(160) + hidden + sequence(176)
    chains = {"train": "train_A", "val": "valid_A", "hidden": "hidden_A"}
    cache = tmp_path / "cache.json"
    cache.write_text(json.dumps({chains[role]: {"sequence": seq} for role, seq in
                                 (("train", train), ("val", validation), ("hidden", hidden))}))
    manifests = {}
    for role, chain in chains.items():
        manifests[role] = tmp_path / f"{role}.txt"
        manifests[role].write_text(chain + "\n")
    raw = tmp_path / "raw"
    train_dir = chain_data_dir(raw / "roda_pdb", chains["train"])
    val_dir = chain_data_dir(raw / "roda_pdb", chains["val"])
    train_dir.mkdir(parents=True)
    val_dir.mkdir(parents=True)
    (train_dir / "uniref90_hits.a3m").write_text(f">query\n{train}\n>heldout_domain_in_long_row\n{embedded}\n")
    (val_dir / "uniref90_hits.a3m").write_text(f">query\n{validation}\n>hidden\n{hidden + sequence(16)}\n")
    filters = {}
    for role in ("train", "val"):
        output = tmp_path / f"{role}-filter.json"
        argv = ["build_msa_row_filter.py", "--source-manifest", str(manifests[role]),
                "--chain-data-cache", str(cache), "--raw-root", str(raw), "--output", str(output),
                "--mmseqs-bin", binary, "--threads", "1", "--exhaustive", "--coverage-mode", "2",
                "--max-sequence-length", "400", "--checkpoint-dir", str(tmp_path / f"{role}-checkpoint"),
                "--source-batch-size", "2", "--query-batch-size", "2", "--max-seqs", "2",
                "--search-disk-reserve-gib", "1", "--search-max-rss-gib", "8"]
        for heldout in (("val", "hidden") if role == "train" else ("hidden",)):
            argv += ["--heldout-manifest", str(manifests[heldout])]
        monkeypatch.setattr(sys, "argv", argv)
        build_msa_row_filter.main()
        filters[role] = output
    train_filter = json.loads(filters["train"].read_text())
    assert hashlib.sha256(embedded.encode()).hexdigest() in train_filter["excluded_sequence_sha256"]
    assert train_filter["threshold"]["cov_mode"] == 2
    kwargs = {"train_manifest": manifests["train"], "val_manifest": manifests["val"],
              "hidden_manifest": manifests["hidden"], "train_filter": filters["train"], "val_filter": filters["val"]}
    with pytest.raises(ValueError, match="threshold semantics"):
        build_msa_filter_policy(**kwargs)
    payload = build_msa_filter_policy(**kwargs, policy_type=FULL_PARENT_POLICY_TYPE)
    policy_path = tmp_path / "policy.json"
    policy_path.write_text(json.dumps(payload))
    policy = load_msa_filter_policy(policy_path)
    msa, _, query, stats = _read_merged_msa(train_dir, msa_name="uniref90_hits.a3m", msa_names=None,
        max_msa_seqs=2048, msa_row_filter=policy.filter_for(chains["train"]))
    assert query == train and msa.shape == (1, 400) and stats.removed_rows == 1
    assert policy.provenance["policy_type"] == FULL_PARENT_POLICY_TYPE
    # A declared long-row bound cannot drift independently of its native proof.
    train_filter["mmseqs"]["maximum_sequence_length"] = 401
    filters["train"].write_text(json.dumps(train_filter))
    payload["filters"]["train"]["sha256"] = hashlib.sha256(filters["train"].read_bytes()).hexdigest()
    policy_path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="length/coverage contract"):
        load_msa_filter_policy(policy_path)


@pytest.mark.skipif(not (os.environ.get("NANOFOLD_MMSEQS_TEST_BINARY") or shutil.which("mmseqs")), reason="Native MMseqs test binary not configured")
def test_native_protected_public_policy_adds_historical_rows_without_changing_v1(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from nanofold.msa_filter_policy import PROTECTED_PUBLIC_POLICY_TYPE

    binary = os.environ.get("NANOFOLD_MMSEQS_TEST_BINARY") or shutil.which("mmseqs")
    assert binary is not None
    rng = random.Random(519)

    def sequence(length: int) -> str:
        return "".join(rng.choice("ARNDCQEGHILKMFPSTWYV") for _ in range(length))

    seqs = {"train_A": sequence(400), "active_A": sequence(80), "retired_A": sequence(64), "hidden_A": sequence(64)}
    cache = tmp_path / "cache.json"
    cache.write_text(json.dumps({chain: {"sequence": seq} for chain, seq in seqs.items()}))
    manifests = {}
    for role, chains in {"train": ["train_A"], "val": ["active_A"], "hidden": ["hidden_A"],
                         "historical_public": ["active_A", "retired_A"]}.items():
        manifests[role] = tmp_path / f"{role}.txt"
        manifests[role].write_text("".join(chain + "\n" for chain in chains))
    embedded = sequence(100) + seqs["retired_A"] + sequence(236)
    raw = tmp_path / "raw"
    for chain, row in (("train_A", embedded), ("active_A", seqs["hidden_A"] + sequence(16))):
        directory = chain_data_dir(raw / "roda_pdb", chain)
        directory.mkdir(parents=True)
        (directory / "uniref90_hits.a3m").write_text(f">query\n{seqs[chain]}\n>row\n{row}\n")

    def run_filter(name: str, role: str, targets: tuple[str, ...]) -> Path:
        output = tmp_path / f"{name}.json"
        argv = ["build_msa_row_filter.py", "--source-manifest", str(manifests[role]),
                "--chain-data-cache", str(cache), "--raw-root", str(raw), "--output", str(output),
                "--mmseqs-bin", binary, "--threads", "1", "--exhaustive", "--coverage-mode", "2",
                "--max-sequence-length", "400", "--checkpoint-dir", str(tmp_path / f"{name}-checkpoint"),
                "--source-batch-size", "2", "--query-batch-size", "2", "--max-seqs", "2",
                "--search-disk-reserve-gib", "1", "--search-max-rss-gib", "8"]
        for target in targets:
            argv += ["--heldout-manifest", str(manifests[target])]
        monkeypatch.setattr(sys, "argv", argv)
        build_msa_row_filter.main()
        return output

    val_filter = run_filter("val", "val", ("hidden",))
    old_filter = run_filter("train-v1", "train", ("val", "hidden"))
    new_filter = run_filter("train-v2", "train", ("val", "historical_public", "hidden"))
    target_hash = hashlib.sha256(embedded.encode()).hexdigest()
    assert target_hash not in json.loads(old_filter.read_text())["excluded_sequence_sha256"]
    assert target_hash in json.loads(new_filter.read_text())["excluded_sequence_sha256"]
    for policy_type, train_filter, expected_rows in ((FULL_PARENT_POLICY_TYPE, old_filter, 2),
                                                    (PROTECTED_PUBLIC_POLICY_TYPE, new_filter, 1)):
        payload = build_msa_filter_policy(train_manifest=manifests["train"], val_manifest=manifests["val"],
            hidden_manifest=manifests["hidden"], train_filter=train_filter, val_filter=val_filter, policy_type=policy_type,
            historical_public_manifest=manifests["historical_public"] if policy_type == PROTECTED_PUBLIC_POLICY_TYPE else None)
        path = tmp_path / f"{policy_type}.json"
        path.write_text(json.dumps(payload))
        policy = load_msa_filter_policy(path)
        msa, _, _, stats = _read_merged_msa(chain_data_dir(raw / "roda_pdb", "train_A"),
            msa_name="uniref90_hits.a3m", msa_names=None, max_msa_seqs=2048,
            msa_row_filter=policy.filter_for("train_A"))
        assert len(msa) == expected_rows and stats.removed_rows == 2 - expected_rows
    # A historical cohort cannot be omitted from a v2 proof or attached to v1.
    with pytest.raises(ValueError, match="historical public manifest"):
        build_msa_filter_policy(train_manifest=manifests["train"], val_manifest=manifests["val"],
            hidden_manifest=manifests["hidden"], train_filter=new_filter, val_filter=val_filter,
            policy_type=PROTECTED_PUBLIC_POLICY_TYPE)
