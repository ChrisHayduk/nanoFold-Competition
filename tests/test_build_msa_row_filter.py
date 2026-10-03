from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

from nanofold.a3m import GAP_ID, sequence_to_ids
from nanofold.chain_paths import chain_npz_path


def _load_filter_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "build_msa_row_filter.py"
    spec = importlib.util.spec_from_file_location("build_msa_row_filter_for_tests", script_path)
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
    np.savez_compressed(path, msa=msa, aatype=msa[0])


def test_parse_excluded_hashes_from_mmseqs_targets(tmp_path: Path) -> None:
    module = _load_filter_module()
    digest = "a" * 64
    m8 = tmp_path / "hits.m8"
    m8.write_text(f"heldout_1\tseq_{digest}\t40.0\t80\t0.8\t0.8\t1e-6\t50\n")

    assert module._parse_excluded_hashes(
        m8, query_ids={"heldout_1"}, source_hashes={digest}, min_seq_id=0.3, coverage=0.8,
    ) == ({digest}, 1)


def test_scan_processed_msa_rows_skips_query_rows(tmp_path: Path) -> None:
    module = _load_filter_module()
    features_dir = tmp_path / "features"
    manifest = tmp_path / "train.txt"
    manifest.write_text("trn1_A\n")
    _write_feature(
        features_dir,
        "trn1_A",
        ["AAAAAAAAAAAAAAAAAAAA", "CCCCCCCCCCCCCCCCCCCC", "CCCCCCCCCCCCCCCCCCCC"],
    )

    sequences_by_hash, stats = module._scan_processed_msa_rows(
        manifest_paths=[manifest],
        features_dirs=[features_dir],
    )

    assert stats["chains"] == 1
    assert stats["rows"] == 3
    assert stats["non_query_rows"] == 2
    assert stats["unique_non_query_rows"] == 1
    assert list(sequences_by_hash.values()) == ["CCCCCCCCCCCCCCCCCCCC"]


def test_scan_processed_msa_rows_searches_multiple_roots(tmp_path: Path) -> None:
    module = _load_filter_module()
    first_features_dir = tmp_path / "features_a"
    second_features_dir = tmp_path / "features_b"
    manifest = tmp_path / "train.txt"
    manifest.write_text("trn1_A\ntrn2_A\n")
    _write_feature(first_features_dir, "trn1_A", ["AAAAAAAAAAAAAAAAAAAA", "CCCCCCCCCCCCCCCCCCCC"])
    _write_feature(second_features_dir, "trn2_A", ["DDDDDDDDDDDDDDDDDDDD", "EEEEEEEEEEEEEEEEEEEE"])

    sequences_by_hash, stats = module._scan_processed_msa_rows(
        manifest_paths=[manifest],
        features_dirs=[first_features_dir, second_features_dir],
    )

    assert stats["chains"] == 2
    assert stats["missing_features"] == 0
    assert sorted(sequences_by_hash.values()) == ["CCCCCCCCCCCCCCCCCCCC", "EEEEEEEEEEEEEEEEEEEE"]


def test_write_heldout_fasta_deduplicates_without_chain_ids(tmp_path: Path) -> None:
    module = _load_filter_module()
    manifest = tmp_path / "heldout.txt"
    manifest.write_text("val1_A\nval2_A\n")
    cache = {
        "val1_A": {"seq": "AAAAAAAAAAAAAAAAAAAA"},
        "val2_A": {"seq": "AAAAAAAAAAAAAAAAAAAA"},
    }
    fasta = tmp_path / "heldout.fasta"

    count = module._write_heldout_fasta(fasta, manifest_paths=[manifest], cache=cache)

    assert count == 1
    text = fasta.read_text()
    assert ">heldout_1" in text
    assert "val1_A" not in text
    assert "val2_A" not in text


def test_filter_cli_pins_actual_identity_and_rejects_missing_search_output(tmp_path: Path) -> None:
    import hashlib
    import json
    import subprocess
    import sys

    features = tmp_path / "features"
    _write_feature(features, "trn1_A", ["A" * 20, "C" * 20])
    train = tmp_path / "train.txt"
    heldout = tmp_path / "heldout.txt"
    cache = tmp_path / "cache.json"
    output = tmp_path / "filter.json"
    train.write_text("trn1_A\n")
    heldout.write_text("val1_A\n")
    cache.write_text(json.dumps({"val1_A": {"sequence": "D" * 20}}))
    digest = hashlib.sha256(("C" * 20).encode()).hexdigest()
    mmseqs = tmp_path / "mmseqs"
    mmseqs.write_text(
        "#!/bin/sh\nif [ \"$1\" = version ]; then echo test; exit 0; fi\n"
        f"printf 'heldout_1\\tseq_{digest}\\t40\\t20\\t1\\t1\\t0.01\\t20\\n' > \"$4\"\n"
    )
    mmseqs.chmod(0o755)
    command = [
        sys.executable, str(Path(__file__).resolve().parents[1] / "scripts/build_msa_row_filter.py"),
        "--source-manifest", str(train), "--heldout-manifest", str(heldout),
        "--chain-data-cache", str(cache), "--source-processed-features-dir", str(features),
        "--mmseqs-bin", str(mmseqs), "--output", str(output),
    ]
    success = subprocess.run(command, capture_output=True, text=True, check=False)
    assert success.returncode == 0, success.stderr
    payload = json.loads(output.read_text())
    assert payload["excluded_sequence_sha256"] == [digest]
    assert payload["threshold"]["alignment_mode"] == 3
    assert payload["threshold"]["seq_id_mode"] == 0
    search = payload["mmseqs"]["command"]
    assert search[search.index("--alignment-mode") + 1] == "3"
    previous = output.read_bytes()
    repeated = subprocess.run(command, capture_output=True, text=True, check=False)
    assert repeated.returncode == 0, repeated.stderr
    assert output.read_bytes() == previous
    mmseqs.write_text("#!/bin/sh\nexit 0\n")
    failure = subprocess.run(command, capture_output=True, text=True, check=False)
    assert failure.returncode != 0
    assert "without writing" in failure.stderr
    assert output.read_bytes() == previous


def test_filter_rejects_unknown_sequence_in_search_output(tmp_path: Path) -> None:
    import pytest

    module = _load_filter_module()
    m8 = tmp_path / "hits.m8"
    m8.write_text(f"heldout_1\tseq_{'b' * 64}\t40\t20\t1\t1\t0.01\t20\n")
    with pytest.raises(ValueError, match="unknown sequence"):
        module._parse_excluded_hashes(
            m8, query_ids={"heldout_1"}, source_hashes={"a" * 64}, min_seq_id=0.3, coverage=0.8,
        )


def test_exact_filter_does_not_depend_on_mmseqs_masking(tmp_path: Path) -> None:
    import hashlib
    import json
    import subprocess
    import sys

    features = tmp_path / "features"
    _write_feature(features, "trn1_A", ["D" * 20, "A" * 20, "X" * 20])
    train, heldout, cache, output = [tmp_path / name for name in ("train", "heldout", "cache", "filter")]
    train.write_text("trn1_A\n")
    heldout.write_text("val1_A\nval2_A\n")
    cache.write_text(json.dumps({"val1_A": {"sequence": "A" * 20}, "val2_A": {"sequence": "B" * 20}}))
    mmseqs = tmp_path / "mmseqs"
    mmseqs.write_text('#!/bin/sh\nif [ "$1" = version ]; then echo test; else : > "$4"; fi\n')
    mmseqs.chmod(0o755)
    proc = subprocess.run([
        sys.executable, str(Path(__file__).resolve().parents[1] / "scripts/build_msa_row_filter.py"),
        "--source-manifest", str(train), "--heldout-manifest", str(heldout),
        "--chain-data-cache", str(cache), "--source-processed-features-dir", str(features),
        "--mmseqs-bin", str(mmseqs), "--output", str(output),
    ], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    result = json.loads(output.read_text())
    assert set(result["excluded_sequence_sha256"]) == {
        hashlib.sha256((residue * 20).encode()).hexdigest() for residue in ("A", "X")
    }
    assert result["exact_match_excluded_sequence_count"] == 2
    assert result["mmseqs"]["hit_count"] == 0


def test_filter_with_real_mmseqs_identity_coverage_and_masked_exact_rows(tmp_path: Path) -> None:
    import hashlib
    import json
    import os
    import random
    import shutil
    import subprocess
    import sys

    import pytest

    binary = os.environ.get("MMSEQS_TEST_BINARY") or shutil.which("mmseqs")
    if not binary:
        pytest.skip("Set MMSEQS_TEST_BINARY to run the real MMseqs integration regression")
    rng = random.Random(43)
    alphabet = "ARNDCQEGHILKMFPSTWYV"
    target = "".join(rng.choices(alphabet, k=100))
    homolog = "".join(aa if i % 2 == 0 else rng.choice(alphabet.replace(aa, "")) for i, aa in enumerate(target))
    fragment = target[:70]
    features = tmp_path / "features"
    _write_feature(features, "trn1_A", ["D" * 100, homolog, fragment, "A" * 100, "X" * 100])
    train, heldout, cache, output = [tmp_path / name for name in ("train", "heldout", "cache", "filter")]
    train.write_text("trn1_A\n")
    heldout.write_text("val1_A\nval2_A\nval3_A\n")
    cache.write_text(json.dumps({"val1_A": {"sequence": target}, "val2_A": {"sequence": "A" * 100},
                                "val3_A": {"sequence": "B" * 100}}))
    proc = subprocess.run([
        sys.executable, str(Path(__file__).resolve().parents[1] / "scripts/build_msa_row_filter.py"),
        "--source-manifest", str(train), "--heldout-manifest", str(heldout),
        "--chain-data-cache", str(cache), "--source-processed-features-dir", str(features),
        "--mmseqs-bin", str(binary), "--threads", "1", "--output", str(output),
    ], capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, proc.stderr + proc.stdout[-2000:]
    result = json.loads(output.read_text())
    expected = {hashlib.sha256(sequence.encode()).hexdigest() for sequence in (homolog, "A" * 100, "X" * 100)}
    assert set(result["excluded_sequence_sha256"]) == expected
    assert hashlib.sha256(fragment.encode()).hexdigest() not in expected
    assert result["mmseqs"]["version"]


def test_real_mmseqs_filter_accepts_ac_only_protein_source_database(tmp_path: Path) -> None:
    import os
    import shutil

    import pytest

    binary = os.environ.get("MMSEQS_TEST_BINARY") or shutil.which("mmseqs")
    if not binary:
        pytest.skip("Set MMSEQS_TEST_BINARY to run the protein database regression")
    heldout, source, output = [tmp_path / name for name in ("heldout.fa", "source.fa", "hits.m8")]
    heldout.write_text(f">heldout_1\n{'D' * 48}\n")
    source.write_text(f">source_1\n{'A' * 48}\n>source_2\n{'C' * 48}\n")
    command = _load_filter_module()._run_mmseqs(
        mmseqs_bin=str(binary), heldout_fasta=heldout, source_fasta=source, output_m8=output,
        tmp_dir=tmp_path / "mmseqs-work", min_seq_id=0.3, coverage=0.8, threads=1, max_seqs=100,
    )
    assert output.read_text() == ""
    assert command[command.index("--dbtype") + 1] == "1"


def test_source_scanning_includes_all_feature_variants_for_the_same_chain(tmp_path: Path) -> None:
    module = _load_filter_module()
    manifest = tmp_path / "train.txt"
    manifest.write_text("trn1_A\n")
    first, second = tmp_path / "first", tmp_path / "second"
    _write_feature(first, "trn1_A", ["A" * 20, "C" * 20])
    _write_feature(second, "trn1_A", ["A" * 20, "D" * 20])
    rows, stats = module._scan_processed_msa_rows(manifest_paths=[manifest], features_dirs=[first, second])
    assert set(rows.values()) == {"C" * 20, "D" * 20}
    assert stats["chains"] == 1
    assert stats["feature_variants"] == 2


def test_inventory_sizes_exhaustive_search_without_mmseqs_and_is_not_a_filter(tmp_path: Path) -> None:
    import json
    import subprocess
    import sys

    features = tmp_path / "features"
    _write_feature(features, "trn1_A", ["A" * 40, "C" * 40, "D" * 40, "C" * 40, "E" * 19])
    train, heldout, cache, output = [tmp_path / name for name in ("train", "heldout", "cache", "inventory")]
    train.write_text("trn1_A\n")
    heldout.write_text("val1_A\nval2_A\n")
    cache.write_text(json.dumps({"val1_A": {"sequence": "F" * 40}, "val2_A": {"sequence": "G" * 40}}))
    result = subprocess.run([
        sys.executable, str(Path(__file__).resolve().parents[1] / "scripts/build_msa_row_filter.py"),
        "--source-manifest", str(train), "--heldout-manifest", str(heldout),
        "--chain-data-cache", str(cache), "--source-processed-features-dir", str(features),
        "--mmseqs-bin", "__does_not_exist__", "--output", str(output),
        "--inventory-only", "--exhaustive", "--max-seqs", "1",
    ], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    inventory = json.loads(output.read_text())
    assert inventory["inventory_type"] == "msa_search_inventory"
    assert "filter_type" not in inventory and "excluded_sequence_sha256" not in inventory
    assert inventory["source_unique_sequence_count"] == inventory["required_exhaustive_max_seqs"] == 2
    assert inventory["heldout_unique_sequence_count"] == 2
    assert inventory["all_pair_comparisons"] == 4
    assert inventory["mmseqs_search_executed"] is False
    assert inventory["exhaustive_hit_capacity_sufficient"] is False
    assert inventory["source_complete"] is True
    assert inventory["source_sequence_set_sha256"]
    assert inventory["heldout_sequence_set_sha256"]


def test_exhaustive_filter_rejects_truncating_hit_capacity_before_search(tmp_path: Path) -> None:
    import pytest

    heldout, source, output = [tmp_path / name for name in ("heldout.fa", "source.fa", "hits.m8")]
    heldout.write_text(f">heldout_1\n{'A' * 40}\n")
    source.write_text(f">source_1\n{'C' * 40}\n>source_2\n{'D' * 40}\n")
    output.write_text("previous\n")
    with pytest.raises(ValueError, match="max-seqs >= 2"):
        _load_filter_module()._run_mmseqs(
            mmseqs_bin="__must_not_execute__", heldout_fasta=heldout, source_fasta=source,
            output_m8=output, tmp_dir=tmp_path / "work", min_seq_id=0.3, coverage=0.8,
            threads=1, max_seqs=1, exhaustive=True,
        )
    assert output.read_text() == "previous\n"


def test_real_exhaustive_filter_recovers_prefilter_missed_homolog(tmp_path: Path) -> None:
    import hashlib
    import json
    import os
    import shutil
    import subprocess
    import sys

    import pytest

    binary = os.environ.get("MMSEQS_TEST_BINARY") or shutil.which("mmseqs")
    if not binary:
        pytest.skip("Set MMSEQS_TEST_BINARY to run the all-pair filter regression")
    source_sequence = "SDNKAWGPALHQYCLWHIMIYVKYYMFIMLYEMNLNEHRA"
    heldout_sequence = "SMPKAWGRWPHWYGLWFYFSYPMYYMFIMYYTTAWNIHRW"
    features = tmp_path / "features"
    _write_feature(features, "trn1_A", ["D" * len(source_sequence), source_sequence])
    train, heldout, cache, output = [tmp_path / name for name in ("train", "heldout", "cache", "filter")]
    train.write_text("trn1_A\n")
    heldout.write_text("val1_A\n")
    cache.write_text(json.dumps({"val1_A": {"sequence": heldout_sequence}}))
    command = [
        sys.executable, str(Path(__file__).resolve().parents[1] / "scripts/build_msa_row_filter.py"),
        "--source-manifest", str(train), "--heldout-manifest", str(heldout),
        "--chain-data-cache", str(cache), "--source-processed-features-dir", str(features),
        "--mmseqs-bin", str(binary), "--threads", "1", "--output", str(output), "--max-seqs", "1",
    ]
    heuristic = subprocess.run(command, capture_output=True, text=True, timeout=120)
    assert heuristic.returncode == 0, heuristic.stderr
    assert json.loads(output.read_text())["excluded_sequence_count"] == 0
    exhaustive = subprocess.run([*command, "--exhaustive"], capture_output=True, text=True, timeout=120)
    assert exhaustive.returncode == 0, exhaustive.stderr
    result = json.loads(output.read_text())
    assert result["excluded_sequence_sha256"] == [hashlib.sha256(source_sequence.encode()).hexdigest()]
    assert result["exhaustive"] is True
    assert result["search_contract"] == "mmseqs-local-all-pairs-v1"
    assert result["mmseqs"]["possible_pair_count"] == 1
    assert result["mmseqs"]["search_parameters"]["max_accept"] == 1
    assert result["input_binding"]["source_fasta_sha256"]
    assert result["input_binding"]["heldout_fasta_sha256"]
    assert result["input_binding"]["m8_sha256"]


def test_inventory_resolves_cache_once_and_binds_actual_source_assets(tmp_path: Path, monkeypatch) -> None:
    import hashlib
    import json
    import sys

    features = tmp_path / "features"
    _write_feature(features, "trn1_A", ["A" * 40, "C" * 40])
    train, heldout = tmp_path / "train", tmp_path / "heldout"
    train.write_text("trn1_A\n")
    heldout.write_text("val1_A\n")
    first, second, alias, output = [tmp_path / name for name in ("first.json", "second.json", "alias.json", "inventory")]
    first.write_text(json.dumps({"val1_A": {"sequence": "D" * 40}}))
    second.write_text(json.dumps({"val1_A": {"sequence": "E" * 40}}))
    alias.symlink_to(first)
    module = _load_filter_module()
    original_load = module._load_chain_data_cache
    def retarget_then_load(path):
        alias.unlink()
        alias.symlink_to(second)
        assert path == first.resolve()
        return original_load(path)
    monkeypatch.setattr(module, "_load_chain_data_cache", retarget_then_load)
    monkeypatch.setattr(sys, "argv", [
        "build_msa_row_filter.py", "--source-manifest", str(train), "--heldout-manifest", str(heldout),
        "--chain-data-cache", str(alias), "--source-processed-features-dir", str(features),
        "--output", str(output), "--inventory-only", "--exhaustive",
    ])
    module.main()
    result = json.loads(output.read_text())
    assert result["chain_data_cache_sha256"] == hashlib.sha256(first.read_bytes()).hexdigest()
    assert result["heldout_sequence_set_sha256"] == module._hash_sequence_set({module._sequence_sha256("D" * 40)})
    feature = chain_npz_path(features, "trn1_A").resolve()
    assert result["input_binding"]["source_assets"] == [
        {"path": str(feature), "sha256": hashlib.sha256(feature.read_bytes()).hexdigest()},
    ]


def test_inventory_fails_if_source_asset_changes_during_scan(tmp_path: Path, monkeypatch) -> None:
    import json
    import sys

    import pytest

    features = tmp_path / "features"
    _write_feature(features, "trn1_A", ["A" * 40, "C" * 40])
    train, heldout, cache, output = [tmp_path / name for name in ("train", "heldout", "cache", "inventory")]
    train.write_text("trn1_A\n")
    heldout.write_text("val1_A\n")
    cache.write_text(json.dumps({"val1_A": {"sequence": "D" * 40}}))
    module = _load_filter_module()
    original_load = module.load_msa_for_audit
    def mutate_after_load(path, **kwargs):
        result = original_load(path, **kwargs)
        path.write_bytes(path.read_bytes() + b"changed during scan")
        return result
    monkeypatch.setattr(module, "load_msa_for_audit", mutate_after_load)
    monkeypatch.setattr(sys, "argv", [
        "build_msa_row_filter.py", "--source-manifest", str(train), "--heldout-manifest", str(heldout),
        "--chain-data-cache", str(cache), "--source-processed-features-dir", str(features),
        "--output", str(output), "--inventory-only",
    ])
    with pytest.raises(ValueError, match="input changed"):
        module.main()
    assert not output.exists()


def test_streaming_raw_projection_matches_token_semantics_and_checks_late_rows(tmp_path: Path) -> None:
    import pytest

    from nanofold.chain_paths import chain_data_dir

    module = _load_filter_module()
    raw = tmp_path / "raw"
    chain = chain_data_dir(raw / "roda_pdb", "train_A")
    chain.mkdir(parents=True)
    manifest = tmp_path / "train.txt"
    manifest.write_text("train_A\n")
    path = chain / "uniref90_hits.a3m"
    # Query-gap columns drop before row ungapping. Unknown uppercase residues
    # map to X and lowercase/dot insertion states contribute no searched AA.
    path.write_text(">query\n" + "A-" + "C" * 20 + "\n>row\n" + "Bf.Q" + "D" * 20 + "\n")
    sequences, counts = module._scan_raw_msa_rows(manifest_paths=[manifest], raw_root=raw, msa_names=[path.name])
    assert list(sequences.values()) == ["X" + "D" * 20]
    assert counts["rows"] == 2 and counts["non_query_rows"] == 1
    with path.open("a") as handle:
        handle.write(">malformed_last_row\nA\n")
    with pytest.raises(ValueError, match="aligned"):
        module._scan_raw_msa_rows(manifest_paths=[manifest], raw_root=raw, msa_names=[path.name])
