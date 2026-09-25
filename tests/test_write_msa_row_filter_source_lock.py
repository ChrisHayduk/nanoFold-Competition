from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


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
                    "--alignment-mode",
                    "3",
                    "--seq-id-mode",
                    "0",
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


def _write_audit(path: Path, row_filter: Path) -> None:
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
            "msa_row_filter_sha256": hashlib.sha256(row_filter.read_bytes()).hexdigest(),
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
    _write_audit(audit, row_filter)

    summary = module.build_source_lock(row_filter_path=row_filter, processed_filter_audit_paths=[audit])

    rendered = json.dumps(summary, sort_keys=True)
    assert summary["public_safe"] is True
    assert summary["safety"]["raw_sequences_included"] is False
    assert summary["safety"]["excluded_sequence_sha256_included"] is False
    assert summary["safety"]["per_chain_processed_audit_included"] is False
    assert summary["aggregate_processed_filtering"]["cumulative_msa_rows_removed"] == 10
    assert "<heldout_targets.fasta>" in summary["msa_row_filter"]["mmseqs"]["command"]
    command = summary["msa_row_filter"]["mmseqs"]["command"]
    assert command[command.index("--alignment-mode") + 1] == "3"
    assert command[command.index("--seq-id-mode") + 1] == "0"
    assert "hiddenhash" not in rendered
    assert ".nanofold_private" not in rendered
    assert "secret_H" not in rendered
    assert "a" * 64 not in rendered


def test_source_lock_summary_can_keep_private_manifest_hashes_for_sealed_records(tmp_path: Path) -> None:
    module = _load_lock_module()
    row_filter = tmp_path / "filter.json"
    audit = tmp_path / "audit.json"
    _write_filter(row_filter)
    _write_audit(audit, row_filter)

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


def test_public_summary_redacts_custom_private_paths_and_manifest_hashes(tmp_path: Path) -> None:
    module = _load_lock_module()
    row_filter = tmp_path / "secret_workspace" / "filter.json"
    _write_filter(row_filter)
    payload = json.loads(row_filter.read_text())
    payload["heldout"]["manifests"] = [{"path": "custom/heldout.txt", "sha256": "customhiddenhash", "chain_count": 1}]
    payload["mmseqs"]["binary"] = "/home/private-person/bin/mmseqs"
    payload["mmseqs"]["command"][0] = payload["mmseqs"]["binary"]
    payload["source"]["stats"]["secret_chain"] = "secret_A"
    _write_json(row_filter, payload)
    summary = module.build_source_lock(row_filter_path=row_filter, processed_filter_audit_paths=[])
    rendered = json.dumps(summary)
    assert "secret_workspace" not in rendered
    assert "custom/heldout.txt" not in rendered
    assert "customhiddenhash" not in rendered
    assert "private-person" not in rendered
    assert "secret_A" not in rendered
    assert summary["msa_row_filter"]["sha256"] == hashlib.sha256(row_filter.read_bytes()).hexdigest()
    assert summary["validation"]["residual_homology_audited"] is False


def test_source_lock_rejects_unrelated_final_filter_audit(tmp_path: Path) -> None:
    import pytest

    module = _load_lock_module()
    row_filter = tmp_path / "filter.json"
    audit = tmp_path / "audit.json"
    _write_filter(row_filter)
    _write_audit(audit, row_filter)
    payload = json.loads(audit.read_text())
    payload["msa_row_filter_sha256"] = "0" * 64
    _write_json(audit, payload)
    with pytest.raises(ValueError, match="does not reference"):
        module.build_source_lock(row_filter_path=row_filter, processed_filter_audit_paths=[audit])


def test_source_lock_rejects_inconsistent_filtering_counts(tmp_path: Path) -> None:
    import pytest

    module = _load_lock_module()
    row_filter = tmp_path / "filter.json"
    audit = tmp_path / "audit.json"
    _write_filter(row_filter)
    _write_audit(audit, row_filter)
    payload = json.loads(audit.read_text())
    payload["total_msa_rows_removed"] = 15
    _write_json(audit, payload)
    with pytest.raises(ValueError, match="counts are inconsistent"):
        module.build_source_lock(row_filter_path=row_filter, processed_filter_audit_paths=[audit])


def test_source_lock_retains_exhaustive_contract_and_redacts_bound_private_inputs(tmp_path: Path) -> None:
    module = _load_lock_module()
    row_filter = tmp_path / "filter.json"
    _write_filter(row_filter)
    payload = json.loads(row_filter.read_text())
    payload["search_contract"] = "mmseqs-local-all-pairs-v1"
    payload["exhaustive"] = True
    payload["input_binding"] = {
        "source_fasta_sha256": "c" * 64, "heldout_fasta_sha256": "d" * 64, "m8_sha256": "e" * 64,
        "source_sequence_set_sha256": "f" * 64, "heldout_sequence_set_sha256": "1" * 64,
        "secret_sequence": "private_target_chain",
    }
    payload["mmseqs"]["search_parameters"] = {
        "prefilter_mode": 2, "evalue": 1e308, "mask": 0, "comp_bias_corr": 0, "dbtype": 1,
        "max_accept": 1_000_000, "max_rejected": 1_000_000, "secret_sequence": "private_target_chain",
    }
    payload["mmseqs"]["command"].extend([
        "--prefilter-mode", "2", "--mask", "0", "--comp-bias-corr", "0", "-e", "1e308",
        "--max-accept", "1000000", "--max-rejected", "1000000",
    ])
    _write_json(row_filter, payload)
    public = module.build_source_lock(row_filter_path=row_filter, processed_filter_audit_paths=[])
    summary = public["msa_row_filter"]
    assert summary["search_contract"] == "mmseqs-local-all-pairs-v1"
    assert summary["exhaustive"] is True
    assert summary["mmseqs"]["search_parameters"]["prefilter_mode"] == 2
    command = summary["mmseqs"]["command"]
    assert command[command.index("--comp-bias-corr") + 1] == "0"
    assert command[command.index("--max-rejected") + 1] == "1000000"
    assert set(summary["input_binding"].values()) == {"<private>"}
    assert "private_target_chain" not in json.dumps(public)
    assert "d" * 64 not in json.dumps(public)
    private = module.build_source_lock(row_filter_path=row_filter, processed_filter_audit_paths=[], include_private_paths=True)
    assert private["msa_row_filter"]["input_binding"]["heldout_fasta_sha256"] == "d" * 64
    assert "private_target_chain" not in json.dumps(private)


def test_source_lock_rejects_search_inventory_as_a_completed_filter(tmp_path: Path) -> None:
    import pytest

    inventory = tmp_path / "inventory.json"
    _write_json(inventory, {"schema_version": 1, "inventory_type": "msa_search_inventory"})
    with pytest.raises(ValueError, match="inventory is not a row filter"):
        _load_lock_module().build_source_lock(row_filter_path=inventory, processed_filter_audit_paths=[])


def _write_batched_filter(path: Path) -> dict:
    _write_filter(path)
    payload = json.loads(path.read_text())
    payload.update(search_contract="mmseqs-local-all-pairs-v1", exhaustive=True,
                   input_binding={"batch_receipt_sha256": "c" * 64})
    payload["mmseqs"].update(command=[], possible_pair_count=100, batching={
        "schema_version": 1, "status": "complete", "source_batch_size": 20, "query_batch_size": 1,
        "source_partition_count": 3, "query_partition_count": 2, "batch_count": 6,
        "source_count": 50, "query_count": 2, "alignments_calculated": 80,
        "receipt_path": "/private/checkpoints/search/receipt.json", "receipt_sha256": "c" * 64,
    })
    _write_json(path, payload)
    return payload


def test_batched_source_lock_retains_only_public_aggregates_and_redacted_bindings(tmp_path: Path) -> None:
    module = _load_lock_module()
    row_filter = tmp_path / "filter.json"
    payload = _write_batched_filter(row_filter)
    public = module.build_source_lock(row_filter_path=row_filter, processed_filter_audit_paths=[])
    summary = public["msa_row_filter"]
    expected = dict(payload["mmseqs"]["batching"])
    expected.pop("receipt_path")
    expected["receipt_sha256"] = "<private>"
    assert summary["mmseqs"]["batching"] == expected
    assert summary["input_binding"]["batch_receipt_sha256"] == "<private>"
    assert summary["mmseqs"]["command"] == []
    assert "/private/checkpoints" not in json.dumps(public)
    assert "c" * 64 not in json.dumps(public)
    sealed = module.build_source_lock(row_filter_path=row_filter, processed_filter_audit_paths=[], include_private_paths=True)
    assert sealed["msa_row_filter"]["mmseqs"]["batching"] == payload["mmseqs"]["batching"]
    assert sealed["msa_row_filter"]["input_binding"]["batch_receipt_sha256"] == "c" * 64


@pytest.mark.parametrize("corruption", [
    "incomplete", "unbound", "unknown_jobs", "partial_grid", "mismatched_sources", "boolean_count",
    "bad_receipt", "single_result", "orphan_receipt",
])
def test_batched_source_lock_rejects_incomplete_or_inconsistent_receipts(tmp_path: Path, corruption: str) -> None:
    row_filter = tmp_path / "filter.json"
    payload = _write_batched_filter(row_filter)
    batching = payload["mmseqs"]["batching"]
    if corruption == "incomplete":
        batching["status"] = "running"
    elif corruption == "unbound":
        payload["input_binding"]["batch_receipt_sha256"] = "d" * 64
    elif corruption == "unknown_jobs":
        batching["jobs"] = []
    elif corruption == "partial_grid":
        batching["batch_count"] = 5
    elif corruption == "mismatched_sources":
        batching["source_count"] = 49
    elif corruption == "boolean_count":
        batching["query_batch_size"] = True
    elif corruption == "bad_receipt":
        batching["receipt_sha256"] = "invalid"
    elif corruption == "single_result":
        payload["input_binding"]["m8_sha256"] = "e" * 64
    else:
        del payload["mmseqs"]["batching"]
    _write_json(row_filter, payload)
    with pytest.raises(ValueError, match="MSA batching|MSA batch receipt"):
        _load_lock_module().build_source_lock(row_filter_path=row_filter, processed_filter_audit_paths=[])
