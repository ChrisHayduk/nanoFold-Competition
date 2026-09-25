from __future__ import annotations

import importlib.util
import json
from pathlib import Path


def _load_lock_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "write_msa_row_filter_source_lock.py"
    spec = importlib.util.spec_from_file_location("write_msa_row_filter_source_lock_for_tests", script_path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_filter(path: Path) -> None:
    _write_json(
        path,
        {
            "schema_version": 1,
            "filter_type": "heldout_target_homology",
            "threshold": {"min_seq_id": 0.3, "coverage": 0.8, "cov_mode": 0},
            "excluded_sequence_count": 2,
            "excluded_sequence_sha256": ["a" * 64, "b" * 64],
            "mmseqs": {
                "binary": "mmseqs",
                "version": "18-test",
                "command": [
                    "mmseqs",
                    "easy-search",
                    "/tmp/heldout.fasta",
                    "/tmp/source.fasta",
                    "/tmp/out.m8",
                    "/tmp/mmseqs",
                    "--min-seq-id",
                    "0.3",
                ],
                "max_seqs": 1000000,
                "hit_count": 5,
            },
            "source": {
                "mode": "processed_features",
                "manifests": [
                    {"path": "data/manifests/train.txt", "sha256": "trainhash", "chain_count": 10}
                ],
                "processed_features_dirs": ["neurips_paper/results_private/features"],
                "msa_names": ["uniref90_hits.a3m"],
                "stats": {"chains": 10, "rows": 100, "unique_non_query_rows": 50},
            },
            "heldout": {
                "manifests": [
                    {"path": "data/manifests/val.txt", "sha256": "valhash", "chain_count": 1},
                    {
                        "path": ".nanofold_private/manifests/hidden_val.txt",
                        "sha256": "hiddenhash",
                        "chain_count": 1,
                    },
                ],
                "unique_target_sequence_count": 2,
            },
            "chain_data_cache": {"path": "data/openproteinset/cache.json", "sha256": "cachehash"},
        },
    )


def _write_audit(path: Path) -> None:
    _write_json(
        path,
        {
            "schema_version": 1,
            "operation": "processed_feature_msa_row_filter",
            "manifests": [
                {"path": "data/manifests/train.txt", "sha256": "trainhash", "chain_count": 10},
                {
                    "path": ".nanofold_private/manifests/hidden_val.txt",
                    "sha256": "hiddenhash",
                    "chain_count": 1,
                },
            ],
            "chain_count": 11,
            "filtered_chain_count": 11,
            "missing_chain_count": 0,
            "excluded_sequence_count": 2,
            "msa_row_filter_sha256": "filterhash",
            "out_features_dir": "neurips_paper/results_private/out",
            "source_processed_features_dirs": ["neurips_paper/results_private/source"],
            "total_msa_rows_before_filter": 100,
            "total_msa_rows_after_filter": 90,
            "total_msa_rows_removed": 10,
            "per_chain": [{"chain_id": "secret_H", "removed_rows": 10}],
        },
    )


def test_source_lock_summary_redacts_private_details_by_default(tmp_path: Path) -> None:
    module = _load_lock_module()
    row_filter = tmp_path / "private" / "filter.json"
    audit = tmp_path / "private" / "audit.json"
    _write_filter(row_filter)
    _write_audit(audit)

    summary = module.build_source_lock(row_filter_path=row_filter, processed_filter_audit_paths=[audit])

    rendered = json.dumps(summary, sort_keys=True)
    assert summary["public_safe"] is True
    assert summary["safety"]["raw_sequences_included"] is False
    assert summary["safety"]["excluded_sequence_sha256_included"] is False
    assert summary["safety"]["per_chain_processed_audit_included"] is False
    assert summary["aggregate_processed_filtering"]["cumulative_msa_rows_removed"] == 10
    assert "<heldout_targets.fasta>" in summary["msa_row_filter"]["mmseqs"]["command"]
    assert "hiddenhash" not in rendered
    assert ".nanofold_private" not in rendered
    assert "secret_H" not in rendered
    assert "a" * 64 not in rendered


def test_source_lock_summary_can_keep_private_manifest_hashes_for_sealed_records(tmp_path: Path) -> None:
    module = _load_lock_module()
    row_filter = tmp_path / "filter.json"
    audit = tmp_path / "audit.json"
    _write_filter(row_filter)
    _write_audit(audit)

    summary = module.build_source_lock(
        row_filter_path=row_filter,
        processed_filter_audit_paths=[audit],
        include_private_paths=True,
    )

    rendered = json.dumps(summary, sort_keys=True)
    assert summary["public_safe"] is False
    assert "hiddenhash" in rendered
    assert ".nanofold_private/manifests/hidden_val.txt" in rendered
    assert "secret_H" not in rendered
    assert "a" * 64 not in rendered
