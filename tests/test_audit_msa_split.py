from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

from nanofold.a3m import GAP_ID, sequence_to_ids
from nanofold.chain_paths import chain_data_dir, chain_npz_path


def _load_audit_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "audit_msa_split.py"
    spec = importlib.util.spec_from_file_location("audit_msa_split_for_tests", script_path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _write_feature(features_dir: Path, chain_id: str, rows: list[str]) -> None:
    max_len = max(len(row) for row in rows)
    msa = np.full((len(rows), max_len), GAP_ID, dtype=np.int32)
    for row_index, row in enumerate(rows):
        msa[row_index, : len(row)] = sequence_to_ids(row)
    path = chain_npz_path(features_dir, chain_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        aatype=sequence_to_ids(rows[0]),
        msa=msa,
        deletions=np.zeros_like(msa),
    )


def test_audit_processed_msa_exact_counts_target_leakage(tmp_path: Path) -> None:
    module = _load_audit_module()
    features_dir = tmp_path / "features"
    _write_feature(features_dir, "trn1_A", ["AAAAAAAAAAAAAAAAAAAA", "CCCCCCCCCCCCCCCCCCCC"])
    _write_feature(features_dir, "val1_A", ["CCCCCCCCCCCCCCCCCCCC", "DDDDDDDDDDDDDDDDDDDD"])

    summary = module.audit_processed_msa_exact(
        train_ids=["trn1_A"],
        val_ids=["val1_A"],
        features_dir=features_dir,
        include_examples=True,
    )

    assert summary["train"]["rows"] == 2
    assert summary["val"]["rows"] == 2
    assert summary["val_target_sequences_seen_in_train_non_query_msa"] == 1
    assert summary["train_val_non_query_msa_unique_sequence_overlap"] == 0
    assert summary["val_target_sequence_examples"]["val1_A"][0]["chain_id"] == "trn1_A"


def test_msa_depth_distribution_reports_js_divergence() -> None:
    module = _load_audit_module()
    depth_bin_edges = (1, 2, 4)
    train_rows = [
        {"msa_depth": 1, "msa_depth_bin": module._depth_bin(1, depth_bin_edges)},
        {"msa_depth": 2, "msa_depth_bin": module._depth_bin(2, depth_bin_edges)},
    ]
    val_rows = [
        {"msa_depth": 4, "msa_depth_bin": module._depth_bin(4, depth_bin_edges)},
        {"msa_depth": 5, "msa_depth_bin": module._depth_bin(5, depth_bin_edges)},
    ]

    summary = module.summarize_msa_depth_distribution(
        train_rows=train_rows,
        val_rows=val_rows,
        depth_bin_edges=depth_bin_edges,
    )

    assert summary["bins"] == ["<= 1", "2", "3-4", "> 4"]
    assert summary["train_counts"]["<= 1"] == 1
    assert summary["val_counts"]["> 4"] == 1
    assert summary["jensen_shannon_divergence_bits"] > 0


def test_raw_msa_identifier_overlap_canonicalizes_ranges(tmp_path: Path) -> None:
    module = _load_audit_module()
    raw_root = tmp_path / "raw"
    train_dir = chain_data_dir(raw_root / "roda_pdb", "trn1_A")
    val_dir = chain_data_dir(raw_root / "roda_pdb", "val1_A")
    train_dir.mkdir(parents=True)
    val_dir.mkdir(parents=True)
    (train_dir / "uniref90_hits.a3m").write_text(
        ">query\nAAAAAAAAAAAAAAAAAAAA\n"
        ">UniRef90_SHARED/1-20 shared hit\nAAAAAAAAAAAAAAAAAAAA\n"
        ">UniRef90_TRAIN_ONLY/1-20 train hit\nAAAAAAAAAAAAAAAAAAAA\n"
    )
    (val_dir / "uniref90_hits.a3m").write_text(
        ">query\nCCCCCCCCCCCCCCCCCCCC\n"
        ">UniRef90_SHARED/5-24 shared hit different range\nCCCCCCCCCCCCCCCCCCCC\n"
        ">UniRef90_VAL_ONLY/1-20 val hit\nCCCCCCCCCCCCCCCCCCCC\n"
    )

    summary = module.audit_raw_msa_identifier_overlap(
        train_ids=["trn1_A"],
        val_ids=["val1_A"],
        raw_root=raw_root,
        msa_names=("uniref90_hits.a3m",),
        include_examples=True,
    )

    assert summary["train"]["unique_identifiers"] == 2
    assert summary["val"]["unique_identifiers"] == 2
    assert summary["train_val_unique_identifier_overlap"] == 1
    assert summary["overlap_examples"][0]["identifier"] == "UniRef90_SHARED"


def test_target_cluster_audit_does_not_impute_missing_assignments(tmp_path: Path) -> None:
    module = _load_audit_module()
    absent = module.target_disjointness(train_ids=["same_A"], val_ids=["same_B"], cluster_tsv=None)
    assert absent["train_val_cluster_overlap"] is None
    assert absent["cluster_map_complete"] is False
    assert absent["train_val_pdb_overlap"] == 1
    clusters = tmp_path / "clusters.tsv"
    clusters.write_text("same_A\tsame_A\n")
    incomplete = module.target_disjointness(train_ids=["same_A"], val_ids=["other_A"], cluster_tsv=clusters)
    assert incomplete["missing_cluster_assignments"] == 1
    assert incomplete["cluster_map_complete"] is False


def test_depth_csv_identifiers_are_opt_in(tmp_path: Path) -> None:
    module = _load_audit_module()
    rows = [{"split": "val", "chain_id": "secret_A", "length": 20, "msa_depth": 2, "msa_depth_bin": "2"}]
    path = tmp_path / "depth.csv"
    module._write_depth_csv(path, rows)
    assert "secret_A" not in path.read_text()
    assert "chain_id" not in path.read_text()
    module._write_depth_csv(path, rows, include_identifiers=True)
    assert "secret_A" in path.read_text()


def test_cli_strict_audit_checks_heldout_root_and_retains_failure_report(tmp_path: Path) -> None:
    import json
    import subprocess
    import sys

    public = tmp_path / "public"
    private = tmp_path / "private"
    _write_feature(public, "trn1_A", ["A" * 20, "C" * 20])
    _write_feature(private, "secret_A", ["C" * 20])
    train = tmp_path / "train.txt"
    val = tmp_path / "hidden.txt"
    train.write_text("trn1_A\n")
    val.write_text("secret_A\nmissing_A\n")
    report = tmp_path / "report"
    cmd = [
        sys.executable, str(Path(__file__).resolve().parents[1] / "scripts/audit_msa_split.py"),
        "--train-manifest", str(train), "--val-manifest", str(val),
        "--features-dir", str(public), "--val-features-dir", str(private),
        "--out-dir", str(report), "--skip-plot",
    ]
    descriptive = subprocess.run(cmd, capture_output=True, text=True, check=False)
    assert descriptive.returncode == 0, descriptive.stderr
    strict = subprocess.run([*cmd, "--strict"], capture_output=True, text=True, check=False)
    assert strict.returncode != 0
    summary = json.loads((report / "msa_split_audit_summary.json").read_text())
    assert summary["validation"]["passed"] is False
    assert "val: missing_feature_files=1" in summary["validation"]["failures"]
    assert "val_target_sequences_seen_in_train_non_query_msa=1" in summary["validation"]["failures"]
    assert "secret_A" not in (report / "msa_split_audit_depths.csv").read_text()
    assert "secret_A" not in json.dumps(summary)


def test_strict_cli_gates_mmseqs_residual_hits(tmp_path: Path) -> None:
    import json
    import subprocess
    import sys

    features = tmp_path / "features"
    _write_feature(features, "trn1_A", ["A" * 20, "C" * 20])
    _write_feature(features, "val1_A", ["D" * 20])
    train = tmp_path / "train.txt"
    val = tmp_path / "val.txt"
    train.write_text("trn1_A\n")
    val.write_text("val1_A\n")
    mmseqs = tmp_path / "mmseqs"
    mmseqs.write_text(
        "#!/bin/sh\nif [ \"$1\" = version ]; then echo test; exit 0; fi\n"
        "case \"$3\" in *train_targets.fasta) : > \"$4\"; exit 0;; esac\n"
        "printf 'valtarget_0\\ttrainmsa_0\\t50\\t20\\t1\\t1\\t0.01\\t20\\n' > \"$4\"\n"
    )
    mmseqs.chmod(0o755)
    report = tmp_path / "report"
    proc = subprocess.run([
        sys.executable, str(Path(__file__).resolve().parents[1] / "scripts/audit_msa_split.py"),
        "--train-manifest", str(train), "--val-manifest", str(val), "--features-dir", str(features),
        "--out-dir", str(report), "--strict", "--run-mmseqs", "--mmseqs-bin", str(mmseqs), "--skip-plot",
    ], capture_output=True, text=True, check=False)
    assert proc.returncode != 0
    summary = json.loads((report / "msa_split_audit_summary.json").read_text())
    assert summary["processed_msa_exact"]["val_target_sequences_seen_in_train_non_query_msa"] == 0
    assert "val_targets_with_train_msa_homolog_hits=1" in summary["validation"]["failures"]


def test_strict_audit_gates_target_homology_despite_disjoint_clusters_and_query_only_msas(tmp_path: Path) -> None:
    import json
    import subprocess
    import sys

    public = tmp_path / "public"
    private = tmp_path / "private"
    _write_feature(public, "trn1_A", ["A" * 20])
    _write_feature(private, "secret_A", ["A" * 10 + "C" * 10])
    train = tmp_path / "train.txt"
    val = tmp_path / "hidden.txt"
    clusters = tmp_path / "clusters.tsv"
    train.write_text("trn1_A\n")
    val.write_text("secret_A\n")
    clusters.write_text("trn1_A\ttrn1_A\nsecret_A\tsecret_A\n")
    mmseqs = tmp_path / "mmseqs"
    mmseqs.write_text(
        "#!/bin/sh\nif [ \"$1\" = version ]; then echo test; exit 0; fi\n"
        "printf 'valtarget_0\\ttraintarget_0\\t50\\t20\\t1\\t1\\t0.01\\t20\\n' > \"$4\"\n"
    )
    mmseqs.chmod(0o755)
    report = tmp_path / "report"
    proc = subprocess.run([
        sys.executable, str(Path(__file__).resolve().parents[1] / "scripts/audit_msa_split.py"),
        "--train-manifest", str(train), "--val-manifest", str(val),
        "--features-dir", str(public), "--val-features-dir", str(private), "--cluster-tsv", str(clusters),
        "--out-dir", str(report), "--strict", "--run-mmseqs", "--mmseqs-bin", str(mmseqs), "--skip-plot",
    ], capture_output=True, text=True, check=False)
    assert proc.returncode != 0
    summary = json.loads((report / "msa_split_audit_summary.json").read_text())
    assert summary["target_manifest_disjointness"]["train_val_cluster_overlap"] == 0
    assert summary["processed_msa_exact"]["train_val_target_sequence_overlap"] == 0
    assert summary["mmseqs_homology"]["search_executed"] is False
    assert summary["mmseqs_target_homology"]["search_executed"] is True
    assert summary["validation"]["failures"] == ["val_targets_with_train_target_homolog_hits=1"]
    command = summary["mmseqs_target_homology"]["mmseqs"]["command"]
    assert command[command.index("--alignment-mode") + 1] == "3"
    assert command[command.index("--seq-id-mode") + 1] == "0"
    assert "secret_A" not in json.dumps(summary)
    assert "secret_A" not in (report / "val_targets_vs_train_targets.mmseqs_m8").read_text()


def test_mmseqs_cannot_reuse_stale_zero_hit_output(tmp_path: Path) -> None:
    import pytest

    module = _load_audit_module()
    features = tmp_path / "features"
    _write_feature(features, "trn1_A", ["A" * 20, "C" * 20])
    _write_feature(features, "val1_A", ["D" * 20])
    mmseqs = tmp_path / "mmseqs"
    mmseqs.write_text("#!/bin/sh\nexit 0\n")
    mmseqs.chmod(0o755)
    (tmp_path / "val_targets_vs_train_msa.mmseqs_m8").write_text("")
    with pytest.raises(RuntimeError, match="without writing"):
        module.run_mmseqs_homology(
            train_ids=["trn1_A"], val_ids=["val1_A"], features_dir=features, out_dir=tmp_path,
            mmseqs_bin=str(mmseqs), min_seq_id=0.3, coverage=0.8, threads=1, max_seqs=100,
            tmp_dir=None, include_identifiers=False,
        )


def test_conflicting_cluster_membership_and_invalid_msa_are_rejected(tmp_path: Path) -> None:
    import pytest

    module = _load_audit_module()
    clusters = tmp_path / "clusters.tsv"
    clusters.write_text("trn1_A\tval1_A\nother_A\tval1_A\n")
    with pytest.raises(ValueError, match="conflicting clusters"):
        module.target_disjointness(train_ids=["trn1_A"], val_ids=["val1_A"], cluster_tsv=clusters)
    path = chain_npz_path(tmp_path, "trn1_A")
    for bad_tokens in (np.array([[2**32]], dtype=np.int64), np.array([[1.5]], dtype=np.float32)):
        np.savez_compressed(path, aatype=np.array([1]), msa=bad_tokens)
        with pytest.raises(ValueError, match="tokens"):
            module._load_feature_msa(tmp_path, "trn1_A")
    np.savez_compressed(path, aatype=np.array([1]), msa=np.array([[2]]))
    with pytest.raises(ValueError, match="query row does not match"):
        module._load_feature_msa(tmp_path, "trn1_A")
