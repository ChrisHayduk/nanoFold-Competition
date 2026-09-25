"""Publication and source-identity integration for bounded MSA searches."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

import nanofold.mmseqs as engine
from nanofold.a3m import sequence_to_ids
from nanofold.chain_paths import chain_npz_path
from nanofold.msa_filter_summary import validate_batching_summary


def _module(name: str):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).resolve().parents[1] / "scripts" / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _feature(root: Path, chain_id: str, rows: list[str]) -> Path:
    path = chain_npz_path(root, chain_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    msa = np.stack([sequence_to_ids(row) for row in rows])
    np.savez_compressed(path, msa=msa, aatype=msa[0])
    return path


@pytest.fixture
def native_fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    # Exercise the native output/accounting protocol; biological search correctness
    # and resource guards have independent engine/pinned-binary tests.
    binary = tmp_path / "mmseqs-fixture"
    binary.write_text(f"#!{sys.executable}\n" + '''import pathlib, sys
if sys.argv[1] == "version":
    print("fixture-v1")
    raise SystemExit(0)
def records(path):
    lines = pathlib.Path(path).read_text().splitlines()
    return [(lines[i][1:], lines[i+1]) for i in range(0, len(lines), 2)]
queries, sources = records(sys.argv[2]), records(sys.argv[3])
rows = [f"{q}\\t{s}\\t100\\t{len(qs)}\\t1\\t1\\t0\\t50\\n"
        for q, qs in queries for s, ss in sources if qs == ss]
pathlib.Path(sys.argv[4]).write_text("".join(rows))
print(f"Query database size: {len(queries)} type: Aminoacid")
print(f"Target database size: {len(sources)} type: Aminoacid")
print(f"{len(queries) * len(sources)} alignments calculated")
print(f"{len(rows)} sequence pairs passed the thresholds")
''')
    binary.chmod(0o755)
    monkeypatch.setattr(engine, "_batch_resource_check", lambda *args, **kwargs: (1, 10**12))
    return binary


def _filter_inputs(tmp_path: Path, binary: Path) -> tuple[list[str], Path, Path, Path]:
    features = tmp_path / "features"
    _feature(features, "trn1_A", ["A" * 20, "C" * 20, "D" * 20, "E" * 20])
    source = tmp_path / "source.txt"
    source.write_text("trn1_A\n")
    heldout = tmp_path / "heldout.txt"
    # More than nine headers ensures the numeric IDs are sorted lexically for batching.
    targets = {f"val{i}_A": {"sequence": aa * 20} for i, aa in enumerate("CFGHIKLMNPQ")}
    heldout.write_text("\n".join(targets) + "\n")
    cache = tmp_path / "cache.json"
    cache.write_text(json.dumps(targets))
    output = tmp_path / "filter.json"
    checkpoint = tmp_path / "checkpoint"
    args = ["build_msa_row_filter.py", "--source-manifest", str(source), "--heldout-manifest", str(heldout),
            "--chain-data-cache", str(cache), "--source-processed-features-dir", str(features),
            "--mmseqs-bin", str(binary), "--output", str(output), "--exhaustive", "--checkpoint-dir", str(checkpoint),
            "--source-batch-size", "2", "--query-batch-size", "4", "--max-seqs", "2", "--threads", "1",
            "--search-disk-reserve-gib", "1"]
    return args, output, checkpoint, cache


def test_filter_checkpoint_union_resume_and_raw_input_identity(
    tmp_path: Path, native_fixture: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module("build_msa_row_filter")
    args, output, checkpoint, cache = _filter_inputs(tmp_path, native_fixture)
    monkeypatch.setattr(sys, "argv", args)
    module.main()
    original = output.read_bytes()
    payload = json.loads(original)
    assert payload["excluded_sequence_sha256"] == [hashlib.sha256(("C" * 20).encode()).hexdigest()]
    assert payload["mmseqs"]["batching"]["batch_count"] == 6
    assert payload["mmseqs"]["batching"]["alignments_calculated"] == 33
    assert payload["mmseqs"]["command"] == []
    assert "m8_sha256" not in payload["input_binding"]
    assert payload["input_binding"]["batch_receipt_sha256"] == payload["mmseqs"]["batching"]["receipt_sha256"]
    # Check the actual generated payload against the independently owned source-lock projection.
    writer = _module("write_msa_row_filter_source_lock")
    public_summary = writer._summarize_row_filter(output, include_private_paths=False)
    validate_batching_summary(public_summary, public_safe=True)
    assert public_summary["mmseqs"]["batching"]["receipt_sha256"] == "<private>"
    assert "receipt_path" not in public_summary["mmseqs"]["batching"]
    attempts = sorted(checkpoint.glob("job-*/attempt-*"))
    module.main()
    assert output.read_bytes() == original
    assert sorted(checkpoint.glob("job-*/attempt-*")) == attempts
    # Same sequences, changed cache bytes: original source identity must still reject resume.
    cache.write_text(cache.read_text() + "\n")
    with pytest.raises(ValueError, match="binding|inputs|plan"):
        module.main()
    assert output.read_bytes() == original


def test_filter_never_publishes_partial_stream(
    tmp_path: Path, native_fixture: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module("build_msa_row_filter")
    args, output, _checkpoint, _cache = _filter_inputs(tmp_path, native_fixture)
    monkeypatch.setattr(sys, "argv", args)
    original_stream = module.iter_batched_mmseqs_hits

    def failing_stream(*args, **kwargs):
        yield from original_stream(*args, **kwargs)
        raise ValueError("late evidence mutation")

    monkeypatch.setattr(module, "iter_batched_mmseqs_hits", failing_stream)
    with pytest.raises(ValueError, match="late evidence mutation"):
        module.main()
    assert not output.exists()


def test_filter_source_changes_during_search_prevent_publication(
    tmp_path: Path, native_fixture: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module("build_msa_row_filter")
    args, output, _checkpoint, cache = _filter_inputs(tmp_path, native_fixture)
    monkeypatch.setattr(sys, "argv", args)
    original_search = module.run_batched_mmseqs

    def mutating_search(**kwargs):
        result = original_search(**kwargs)
        cache.write_text(cache.read_text() + "\n")
        return result

    monkeypatch.setattr(module, "run_batched_mmseqs", mutating_search)
    with pytest.raises(ValueError, match="input changed"):
        module.main()
    assert not output.exists()


def test_audit_separates_target_and_msa_checkpoints_and_requires_complete_features(
    tmp_path: Path, native_fixture: Path,
) -> None:
    module = _module("audit_msa_split")
    features = tmp_path / "features"
    _feature(features, "trn1_A", ["A" * 20, "C" * 20, "D" * 20, "E" * 20])
    _feature(features, "val1_A", ["C" * 20])
    out = tmp_path / "audit"
    out.mkdir()
    kwargs = dict(train_ids=["trn1_A"], val_ids=["val1_A"], features_dir=features, out_dir=out,
                  mmseqs_bin=str(native_fixture), min_seq_id=0.3, coverage=0.8, threads=1, max_seqs=2,
                  tmp_dir=None, include_identifiers=False, exhaustive=True,
                  checkpoint_dir=tmp_path / "checkpoints", source_batch_size=2, query_batch_size=1,
                  reserve_disk_bytes=0)
    targets = module.run_mmseqs_homology(**kwargs, source_targets_only=True)
    rows = module.run_mmseqs_homology(**kwargs)
    assert targets["hit_count"] == 0
    assert rows["hit_count"] == 1
    assert rows["mmseqs"]["batching"]["batch_count"] == 2
    assert Path(targets["mmseqs"]["batching"]["receipt_path"]).parent.name == "train_targets"
    assert Path(rows["mmseqs"]["batching"]["receipt_path"]).parent.name == "train_msa"
    assert rows["m8_path"] is None
    assert "m8_sha256" not in rows["input_binding"]
    assert module.run_mmseqs_homology(**kwargs) == rows
    with pytest.raises(ValueError, match="every source"):
        module.run_mmseqs_homology(**{**kwargs, "train_ids": ["missing_A"]})
    with pytest.raises(ValueError, match="anonymous"):
        module.run_mmseqs_homology(**{**kwargs, "include_identifiers": True})


def test_audit_does_not_publish_after_only_target_stage_completes(
    tmp_path: Path, native_fixture: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module("audit_msa_split")
    features = tmp_path / "features"
    _feature(features, "trn1_A", ["A" * 20, "C" * 20])
    _feature(features, "val1_A", ["D" * 20])
    train, val, exclusions = (tmp_path / name for name in ("train.txt", "val.txt", "exclusions.txt"))
    train.write_text("trn1_A\n")
    val.write_text("val1_A\n")
    exclusions.write_text("")
    out = tmp_path / "audit"
    out.mkdir()
    report = out / "msa_split_audit_summary.json"
    report.write_text('{"previous": true}\n')
    monkeypatch.setattr(sys, "argv", [
        "audit_msa_split.py", "--train-manifest", str(train), "--val-manifest", str(val),
        "--feature-exclusion-list", str(exclusions), "--features-dir", str(features), "--out-dir", str(out),
        "--run-mmseqs", "--exhaustive", "--mmseqs-bin", str(native_fixture), "--skip-plot",
        "--checkpoint-dir", str(tmp_path / "checkpoints"), "--source-batch-size", "2", "--query-batch-size", "1",
    ])
    original_search = module.run_batched_mmseqs

    def fail_second_stage(**kwargs):
        if kwargs["checkpoint_dir"].name == "train_msa":
            raise RuntimeError("second stage failed")
        return original_search(**kwargs)

    monkeypatch.setattr(module, "run_batched_mmseqs", fail_second_stage)
    with pytest.raises(RuntimeError, match="second stage failed"):
        module.main()
    assert report.read_text() == '{"previous": true}\n'
    assert (tmp_path / "checkpoints" / "train_targets" / "complete.json").is_file()


def test_batched_inventory_reports_per_partition_capacity_without_search(
    tmp_path: Path, native_fixture: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module("build_msa_row_filter")
    args, output, checkpoint, _cache = _filter_inputs(tmp_path, native_fixture)
    args[args.index("--mmseqs-bin") + 1] = "intentionally-unavailable-mmseqs"
    monkeypatch.setattr(sys, "argv", [*args, "--inventory-only"])
    module.main()
    payload = json.loads(output.read_text())
    assert payload["source_unique_sequence_count"] == 3
    assert payload["required_exhaustive_max_seqs"] == 2
    assert payload["exhaustive_hit_capacity_sufficient"] is True
    assert payload["mmseqs_search_executed"] is False
    assert not checkpoint.exists()
