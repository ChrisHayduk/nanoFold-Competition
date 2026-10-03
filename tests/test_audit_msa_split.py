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


def test_real_mmseqs_target_audit_finds_prefilter_missed_pair(tmp_path: Path) -> None:
    import os
    import shutil

    import pytest

    binary = os.environ.get("MMSEQS_TEST_BINARY") or shutil.which("mmseqs")
    if not binary:
        pytest.skip("Set MMSEQS_TEST_BINARY to run the real target alignment regression")
    module = _load_audit_module()
    features = tmp_path / "features"
    _write_feature(features, "trn1_A", ["SDNKAWGPALHQYCLWHIMIYVKYYMFIMLYEMNLNEHRA"])
    _write_feature(features, "val1_A", ["SMPKAWGRWPHWYGLWFYFSYPMYYMFIMYYTTAWNIHRW"])
    result = module.run_mmseqs_homology(
        train_ids=["trn1_A"], val_ids=["val1_A"], features_dir=features, out_dir=tmp_path,
        mmseqs_bin=str(binary), min_seq_id=0.3, coverage=0.8, threads=1, max_seqs=100,
        tmp_dir=None, include_identifiers=False, source_targets_only=True,
    )
    assert result["val_targets_with_train_target_homolog_hits"] == 1
    assert result["search_parameters"]["prefilter_mode"] == 2
    assert result["search_parameters"]["evalue"] == 1e308
    assert result["search_parameters"]["mask"] == 0
    assert result["input_binding"]["m8_sha256"]


def test_real_mmseqs_treats_ac_only_target_sequences_as_proteins(tmp_path: Path) -> None:
    import os
    import shutil

    import pytest

    binary = os.environ.get("MMSEQS_TEST_BINARY") or shutil.which("mmseqs")
    if not binary:
        pytest.skip("Set MMSEQS_TEST_BINARY to run the protein database regression")
    module = _load_audit_module()
    features = tmp_path / "features"
    _write_feature(features, "trn1_A", ["A" * 48])
    for name, query, expected_hits in (("unrelated", "C" * 48, 0), ("homolog", "A" * 40 + "C" * 8, 1)):
        _write_feature(features, "val1_A", [query])
        output = tmp_path / name
        output.mkdir()
        result = module.run_mmseqs_homology(
            train_ids=["trn1_A"], val_ids=["val1_A"], features_dir=features, out_dir=output,
            mmseqs_bin=str(binary), min_seq_id=0.3, coverage=0.8, threads=1, max_seqs=100,
            tmp_dir=None, include_identifiers=False, source_targets_only=True,
        )
        assert result["val_targets_with_train_target_homolog_hits"] == expected_hits
        assert result["search_parameters"]["dbtype"] == 1


def test_exhaustive_audit_requires_search_and_reports_declared_contract(tmp_path: Path) -> None:
    import json
    import subprocess
    import sys

    script = str(Path(__file__).resolve().parents[1] / "scripts/audit_msa_split.py")
    invalid = subprocess.run([sys.executable, script, "--out-dir", str(tmp_path), "--exhaustive"],
                             capture_output=True, text=True)
    assert invalid.returncode != 0
    assert "requires --run-mmseqs" in invalid.stderr
    features = tmp_path / "features"
    _write_feature(features, "trn1_A", ["A" * 40, "C" * 40])
    _write_feature(features, "val1_A", ["D" * 40])
    train, heldout = tmp_path / "train", tmp_path / "heldout"
    train.write_text("trn1_A\n")
    heldout.write_text("val1_A\n")
    mmseqs = tmp_path / "mmseqs"
    mmseqs.write_text('#!/bin/sh\nif [ "$1" = version ]; then echo test; else : > "$4"; fi\n')
    mmseqs.chmod(0o755)
    out = tmp_path / "report"
    proc = subprocess.run([
        sys.executable, script, "--train-manifest", str(train), "--val-manifest", str(heldout),
        "--features-dir", str(features), "--out-dir", str(out), "--skip-plot", "--strict",
        "--run-mmseqs", "--exhaustive", "--max-seqs", "1", "--mmseqs-bin", str(mmseqs),
    ], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    report = json.loads((out / "msa_split_audit_summary.json").read_text())
    assert report["validation"]["exhaustive_msa_homology_checked"] is True
    assert report["validation"]["msa_search_contract"] == "mmseqs-local-all-pairs-v1"
    assert "separate heuristic search" not in report["validation"]["homology_scope"]
    assert report["mmseqs_homology"]["search_parameters"]["prefilter_mode"] == 2
    assert report["mmseqs_homology"]["possible_pair_count"] == 1
    assert report["mmseqs_homology"]["input_binding"]["m8_sha256"]


def test_exhaustive_msa_audit_rejects_source_hit_cap(tmp_path: Path) -> None:
    import pytest

    features = tmp_path / "features"
    _write_feature(features, "trn1_A", ["A" * 40, "C" * 40, "D" * 40])
    _write_feature(features, "val1_A", ["E" * 40])
    mmseqs = tmp_path / "mmseqs"
    mmseqs.write_text("#!/bin/sh\nexit 99\n")
    mmseqs.chmod(0o755)
    with pytest.raises(ValueError, match="max-seqs >= 2"):
        _load_audit_module().run_mmseqs_homology(
            train_ids=["trn1_A"], val_ids=["val1_A"], features_dir=features, out_dir=tmp_path,
            mmseqs_bin=str(mmseqs), min_seq_id=0.3, coverage=0.8, threads=1, max_seqs=1,
            tmp_dir=None, include_identifiers=False, exhaustive=True,
        )


def test_real_exhaustive_msa_audit_recovers_prefilter_missed_pair(tmp_path: Path) -> None:
    import os
    import shutil

    import pytest

    binary = os.environ.get("MMSEQS_TEST_BINARY") or shutil.which("mmseqs")
    if not binary:
        pytest.skip("Set MMSEQS_TEST_BINARY to run the all-pair MSA audit regression")
    features = tmp_path / "features"
    _write_feature(features, "trn1_A", ["D" * 40, "SDNKAWGPALHQYCLWHIMIYVKYYMFIMLYEMNLNEHRA"])
    _write_feature(features, "val1_A", ["SMPKAWGRWPHWYGLWFYFSYPMYYMFIMYYTTAWNIHRW"])
    module = _load_audit_module()
    for exhaustive, expected_hits in ((False, 0), (True, 1)):
        result = module.run_mmseqs_homology(
            train_ids=["trn1_A"], val_ids=["val1_A"], features_dir=features, out_dir=tmp_path,
            mmseqs_bin=str(binary), min_seq_id=0.3, coverage=0.8, threads=1, max_seqs=1,
            tmp_dir=None, include_identifiers=False, exhaustive=exhaustive,
        )
        assert result["val_targets_with_train_msa_homolog_hits"] == expected_hits
        assert result["exhaustive"] == exhaustive


def test_directional_audit_cli_records_scope_and_omits_global_cluster_check(tmp_path: Path) -> None:
    import json
    import subprocess
    import sys

    features = tmp_path / "features"
    _write_feature(features, "trn1_A", ["A" * 320, "C" * 320])
    _write_feature(features, "val1_A", ["D" * 40])
    train, heldout, exclusions = (tmp_path / name for name in ("train", "heldout", "exclusions"))
    train.write_text("trn1_A\n")
    heldout.write_text("val1_A\n")
    exclusions.write_text("")
    mmseqs = tmp_path / "mmseqs"
    mmseqs.write_text(
        '#!/bin/sh\nif [ "$1" = version ]; then echo test; exit 0; fi\n'
        'previous=; for argument in "$@"; do\n'
        'if [ "$previous" = --cov-mode ] && [ "$argument" = 0 ]; then : > "$4"; exit 0; fi\n'
        'previous=$argument; done\n'
        'case "$3" in *train_targets.fasta) source=traintarget_0;; *) source=trainmsa_0;; esac\n'
        "printf 'valtarget_0\\t%s\\t100\\t40\\t1\\t0.125\\t0\\t100\\n' \"$source\" > \"$4\"\n"
    )
    mmseqs.chmod(0o755)
    script = Path(__file__).resolve().parents[1] / "scripts/audit_msa_split.py"
    command = [
        sys.executable, str(script), "--train-manifest", str(train), "--val-manifest", str(heldout),
        "--feature-exclusion-list", str(exclusions), "--features-dir", str(features),
        "--skip-plot", "--strict", "--run-mmseqs", "--exhaustive", "--max-sequence-length", "512",
        "--mmseqs-bin", str(mmseqs),
    ]
    for mode, expected_status in ((0, 0), (2, 1)):
        output = tmp_path / f"audit-{mode}"
        result = subprocess.run([*command, "--out-dir", str(output), "--coverage-mode", str(mode)],
                                text=True, capture_output=True)
        assert bool(result.returncode) == bool(expected_status), result.stderr
        report = json.loads((output / "msa_split_audit_summary.json").read_text())
        assert report["validation"]["coverage_mode"] == mode
        assert report["validation"]["max_sequence_length"] == 512
        assert report["validation"]["cluster_disjointness_checked"] is False
        assert report["target_manifest_disjointness"]["train_val_cluster_overlap"] is None
        assert len(report["validation"]["failures"]) == (2 if mode == 2 else 0)
        for key in ("mmseqs_homology", "mmseqs_target_homology"):
            assert report[key]["coverage_mode"] == mode
            assert report[key]["threshold"]["cov_mode"] == mode
            assert report[key]["max_sequence_length"] == 512
            native_command = report[key]["mmseqs"]["command"]
            assert native_command[native_command.index("--cov-mode") + 1] == str(mode)
            assert native_command[native_command.index("--max-seq-len") + 1] == "512"
        assert ("held-out query coverage" in report["validation"]["homology_scope"]) == (mode == 2)


def test_directional_audit_rejects_heuristics_and_unbounded_sequences(tmp_path: Path) -> None:
    import pytest

    module = _load_audit_module()
    with pytest.raises(ValueError, match="requires --exhaustive"):
        module._validate_search_scope(coverage_mode=2, max_sequence_length=512, exhaustive=False)
    for maximum in (0, 65536, 256.0):
        with pytest.raises(ValueError, match="1..65535"):
            module._validate_search_scope(coverage_mode=2, max_sequence_length=maximum, exhaustive=True)
    features = tmp_path / "features"
    _write_feature(features, "trn1_A", ["A" * 320])
    _write_feature(features, "val1_A", ["C" * 40])
    binary = tmp_path / "mmseqs"
    binary.write_text("#!/bin/sh\nexit 99\n")
    binary.chmod(0o755)
    with pytest.raises(ValueError, match="exceeds --max-sequence-length 256"):
        module.run_mmseqs_homology(
            train_ids=["trn1_A"], val_ids=["val1_A"], features_dir=features, out_dir=tmp_path,
            mmseqs_bin=str(binary), min_seq_id=0.3, coverage=0.8, threads=1, max_seqs=1,
            tmp_dir=None, include_identifiers=False, source_targets_only=True,
        )


def test_real_directional_audit_detects_embedded_domain_in_full_parent(tmp_path: Path) -> None:
    import os
    import random
    import shutil

    import pytest

    binary = os.environ.get("MMSEQS_TEST_BINARY") or shutil.which("mmseqs")
    if not binary:
        pytest.skip("Set MMSEQS_TEST_BINARY to run the full-parent directional regression")
    randomizer = random.Random(9823)
    alphabet = "ACDEFGHIKLMNPQRSTVWY"
    query = "".join(randomizer.choices(alphabet, k=64))
    parent = "".join(randomizer.choices(alphabet, k=144)) + query + "".join(randomizer.choices(alphabet, k=144))
    unrelated = "".join(randomizer.choices(alphabet, k=len(parent)))
    features = tmp_path / "features"
    module = _load_audit_module()
    for targets_only in (True, False):
        _write_feature(features, "trn1_A", [parent] if targets_only else [unrelated, parent])
        _write_feature(features, "val1_A", [query])
        hit_key = ("val_targets_with_train_target_homolog_hits" if targets_only
                   else "val_targets_with_train_msa_homolog_hits")
        for checkpointed in (False, True):
            for coverage_mode in (0, 2):
                output = tmp_path / f"audit-{targets_only}-{checkpointed}-{coverage_mode}"
                output.mkdir()
                result = module.run_mmseqs_homology(
                    train_ids=["trn1_A"], val_ids=["val1_A"], features_dir=features, out_dir=output,
                    mmseqs_bin=str(binary), min_seq_id=0.3, coverage=0.8, threads=1, max_seqs=1,
                    tmp_dir=None, include_identifiers=False, source_targets_only=targets_only, exhaustive=True,
                    checkpoint_dir=output / "checkpoints" if checkpointed else None,
                    source_batch_size=1, query_batch_size=1, reserve_disk_bytes=0,
                    coverage_mode=coverage_mode, max_sequence_length=512,
                )
                assert result[hit_key] == (1 if coverage_mode == 2 else 0)
                assert result["coverage_mode"] == coverage_mode
                assert result["max_sequence_length"] == 512
                if checkpointed:
                    assert result["mmseqs"]["batching"]["alignments_calculated"] == (1 if coverage_mode == 2 else 0)
                elif coverage_mode == 2:
                    fields = Path(result["m8_path"]).read_text().strip().split("\t")
                    assert float(fields[4]) == 1.0
                    assert float(fields[5]) < 0.8


def test_extra_protected_queries_bind_inputs_and_preserve_primary_feature_scope(tmp_path: Path, monkeypatch) -> None:
    import json
    import sys

    import pytest

    import nanofold.mmseqs as engine

    module = _load_audit_module()
    query, active = "C" * 64, "D" * 64
    parent = "A" * 144 + query + "A" * 144
    features = tmp_path / "features"
    _write_feature(features, "trn1_A", ["E" * len(parent), parent])
    _write_feature(features, "val1_A", [active])
    train, primary, historical, hidden, cache, exclusions = (
        tmp_path / name for name in ("train.txt", "primary.txt", "historical.txt", "hidden.txt", "cache.json", "excluded.txt")
    )
    train.write_text("trn1_A\n")
    primary.write_text("val1_A\n")
    historical.write_text("val1_A\nold1_A\n")
    hidden.write_text("hid1_A\n")
    # Deliberate active/historical overlap and identical historical/hidden sequences
    # must add only one query, while all four manifest/cache inputs stay pinned.
    cache.write_text(json.dumps({"val1_A": {"sequence": active}, "old1_A": {"sequence": query},
                                 "hid1_A": {"sequence": query}}))
    exclusions.write_text("")
    binary = tmp_path / "mmseqs-fixture"
    binary.write_text(f"#!{sys.executable}\n" + '''import pathlib, sys
if sys.argv[1] == "version":
    print("extra-protected-fixture-v1")
    raise SystemExit(0)
def records(path):
    lines = pathlib.Path(path).read_text().splitlines()
    return [(lines[i][1:], lines[i+1]) for i in range(0, len(lines), 2)]
queries, sources = records(sys.argv[2]), records(sys.argv[3])
rows = [f"{q}\\t{s}\\t100\\t{len(qs)}\\t1\\t{len(qs)/len(ss)}\\t0\\t50\\n"
        for q, qs in queries for s, ss in sources if qs in ss]
pathlib.Path(sys.argv[4]).write_text("".join(rows))
print(f"Query database size: {len(queries)} type: Aminoacid")
print(f"Target database size: {len(sources)} type: Aminoacid")
print(f"{len(queries) * len(sources)} alignments calculated")
print(f"{len(rows)} sequence pairs passed the thresholds")
''')
    binary.chmod(0o755)
    monkeypatch.setattr(engine, "_batch_resource_check", lambda *args, **kwargs: (1, 10**12))
    output = tmp_path / "report"
    argv = [
        "audit_msa_split.py", "--train-manifest", str(train), "--val-manifest", str(primary),
        "--extra-heldout-manifest", str(historical), "--extra-heldout-manifest", str(hidden),
        "--heldout-chain-data-cache", str(cache), "--features-dir", str(features),
        "--feature-exclusion-list", str(exclusions), "--out-dir", str(output), "--skip-plot", "--strict",
        "--run-mmseqs", "--exhaustive", "--coverage-mode", "2", "--max-sequence-length", "512",
        "--checkpoint-dir", str(tmp_path / "checkpoints"), "--source-batch-size", "1", "--query-batch-size", "2",
        "--max-seqs", "1", "--threads", "1", "--mmseqs-bin", str(binary),
    ]
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit, match="val_targets_with_train_msa_homolog_hits=1"):
        module.main()
    report_path = output / "msa_split_audit_summary.json"
    original_report = report_path.read_bytes()
    report = json.loads(original_report)
    scope = report["heldout_query_scope"]
    assert scope["primary_chain_count"] == 1
    assert scope["extra_chain_count"] == 3
    assert scope["additional_chain_count"] == 2
    assert scope["protected_chain_count"] == 3
    assert scope["primary_unique_query_count"] == 1
    assert scope["extra_unique_query_count"] == 2
    assert scope["additional_unique_query_count"] == 1
    assert scope["protected_unique_query_count"] == 2
    assert scope["extra_manifest_sha256"] == sorted(module._sha256_file(path) for path in (historical, hidden))
    assert scope["chain_data_cache_sha256"] == module._sha256_file(cache)
    assert scope["extra_msa_features_checked"] is False
    assert report["splits"]["val"]["manifest_chains"] == 1
    assert report["splits"]["val"]["missing_feature_files"] == 0
    assert report["processed_msa_exact"]["val_target_sequences_seen_in_train_non_query_msa"] == 0
    assert report["protected_target_manifest_disjointness"]["val_chains"] == 3
    for key in ("mmseqs_target_homology", "mmseqs_homology"):
        assert report[key]["target_records"] == 2
        assert report[key]["heldout_query_scope"] == scope
        assert report[key]["input_binding"]["target_records_sha256"] == scope["searched_query_records_sha256"]
    assert "old1_A" not in json.dumps(report)
    assert "hid1_A" not in json.dumps(report)
    # Exact bytes, including whitespace with no sequence change, invalidate resume.
    for path in (cache, historical, hidden):
        original = path.read_bytes()
        path.write_bytes(original + b"\n")
        with pytest.raises(ValueError, match="binding|inputs|plan"):
            module.main()
        assert report_path.read_bytes() == original_report
        path.write_bytes(original)
    with pytest.raises(SystemExit, match="val_targets_with_train_msa_homolog_hits=1"):
        module.main()
    assert report_path.read_bytes() == original_report


def test_extra_protected_queries_require_complete_canonical_target_identity(tmp_path: Path) -> None:
    import json

    import pytest

    module = _load_audit_module()
    manifest, cache = tmp_path / "extra.txt", tmp_path / "cache.json"
    manifest.write_text("old1_A\n")
    kwargs = dict(val_ids=["val1_A"], manifests=[manifest], cache_path=cache, snapshots={}, exhaustive=True)
    with pytest.raises(ValueError, match="required together"):
        module._extra_heldout_targets(**{**kwargs, "cache_path": None})
    with pytest.raises(ValueError, match="requires|require"):
        module._extra_heldout_targets(**{**kwargs, "exhaustive": False})
    cache.write_text(json.dumps({"old1_A": {"sequence": "C" * 40}}))
    with pytest.raises(ValueError, match="missing"):
        module._extra_heldout_targets(**kwargs)
    for invalid in ({"sequence": "A-C"}, {"sequence": "C" * 40, "seq": "D" * 40}):
        cache.write_text(json.dumps({"old1_A": invalid, "val1_A": {"sequence": "A" * 40}}))
        with pytest.raises(ValueError, match="ungapped|unambiguous"):
            module._extra_heldout_targets(**{**kwargs, "snapshots": {}})
    features = tmp_path / "features"
    _write_feature(features, "val1_A", ["A" * 40])
    with pytest.raises(ValueError, match="does not match"):
        module._collect_mmseqs_records(
            train_ids=[], val_ids=["val1_A"], features_dir=features, include_identifiers=False,
            protected_cache_sequences={"val1_A": "C" * 40},
        )


def test_real_directional_extra_query_detects_historical_target_without_features(tmp_path: Path) -> None:
    import json
    import os
    import random
    import shutil

    import pytest

    binary = os.environ.get("MMSEQS_TEST_BINARY") or shutil.which("mmseqs")
    if not binary:
        pytest.skip("Set MMSEQS_TEST_BINARY to run the extra protected query regression")
    randomizer = random.Random(9342)
    alphabet = "ACDEFGHIKLMNPQRSTVWY"
    historical_query = "".join(randomizer.choices(alphabet, k=64))
    active_query = "".join(randomizer.choices(alphabet, k=64))
    parent = "".join(randomizer.choices(alphabet, k=144)) + historical_query + "".join(randomizer.choices(alphabet, k=144))
    features = tmp_path / "features"
    _write_feature(features, "trn1_A", ["".join(randomizer.choices(alphabet, k=len(parent))), parent])
    _write_feature(features, "val1_A", [active_query])
    historical, cache = tmp_path / "historical.txt", tmp_path / "cache.json"
    historical.write_text("old1_A\n")
    cache.write_text(json.dumps({"val1_A": {"sequence": active_query}, "old1_A": {"sequence": historical_query}}))
    module = _load_audit_module()
    kwargs = dict(
        train_ids=["trn1_A"], val_ids=["val1_A"], features_dir=features, out_dir=tmp_path,
        mmseqs_bin=str(binary), min_seq_id=0.3, coverage=0.8, threads=1, max_seqs=1,
        tmp_dir=None, include_identifiers=False, exhaustive=True, coverage_mode=2, max_sequence_length=512,
    )
    primary_only = module.run_mmseqs_homology(**kwargs)
    assert primary_only["val_targets_with_train_msa_homolog_hits"] == 0
    protected = module.run_mmseqs_homology(
        **kwargs, extra_heldout_manifests=[historical], heldout_chain_data_cache=cache,
    )
    assert protected["val_targets_with_train_msa_homolog_hits"] == 1
    assert protected["target_records"] == 2
    assert protected["heldout_query_scope"]["extra_msa_features_checked"] is False
