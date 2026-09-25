from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

from nanofold.a3m import sequence_to_ids
from nanofold.chain_paths import chain_npz_path


def _load_extend_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "extend_msa_row_filter_from_audit.py"
    spec = importlib.util.spec_from_file_location("extend_msa_row_filter_from_audit_for_tests", script_path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _sequence_sha256(sequence: str) -> str:
    return hashlib.sha256(sequence.upper().encode("utf-8")).hexdigest()


def _write_feature(features_dir: Path, chain_id: str, rows: list[str]) -> None:
    msa = np.stack([sequence_to_ids(row) for row in rows]).astype(np.int32)
    path = chain_npz_path(features_dir, chain_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, msa=msa, aatype=msa[0], deletions=np.zeros_like(msa))



def _write_audit_binding(module, tmp_path: Path, manifest: Path, features: Path, m8: Path) -> Path:
    from nanofold.mmseqs import sequence_records_sha256

    records = module._source_records(manifest_paths=[manifest], features_dir=features)
    audit = tmp_path / "audit.json"
    audit.write_text(json.dumps({"mmseqs_homology": {
        "identifiers_in_m8": False,
        "threshold": {"min_seq_id": 0.3, "coverage": 0.8, "alignment_mode": 3, "seq_id_mode": 0, "cov_mode": 0},
        "input_binding": {"schema_version": 1,
                          "source_records_sha256": sequence_records_sha256(records),
                          "query_ids": ["valtarget_1"],
                          "m8_sha256": hashlib.sha256(m8.read_bytes()).hexdigest()},
    }}))
    return audit

def test_extend_filter_from_anonymous_audit_indices(tmp_path: Path) -> None:
    module = _load_extend_module()
    manifest = tmp_path / "train.txt"
    manifest.write_text("a_A\nb_A\n")
    features_dir = tmp_path / "features"
    first = "CCCCCCCCCCCCCCCCCCCC"
    second = "DDDDDDDDDDDDDDDDDDDD"
    duplicate_first = "CCCCCCCCCCCCCCCCCCCC"
    _write_feature(features_dir, "a_A", ["AAAAAAAAAAAAAAAAAAAA", first, second])
    _write_feature(features_dir, "b_A", ["EEEEEEEEEEEEEEEEEEEE", duplicate_first])

    base_filter = tmp_path / "base_filter.json"
    base_payload = {
        "schema_version": 1,
        "excluded_sequence_count": 1,
        "excluded_sequence_sha256": [_sequence_sha256(first)],
    }
    base_filter.write_text(json.dumps(base_payload, sort_keys=True) + "\n")
    m8 = tmp_path / "residual.m8"
    m8.write_text("valtarget_1\ttrainmsa_1\t30.0\t20\t1.0\t1.0\t1e-5\t40\n")
    out_filter = tmp_path / "extended_filter.json"
    audit = _write_audit_binding(module, tmp_path, manifest, features_dir, m8)

    payload = module.extend_filter_from_audit(
        base_filter_path=base_filter,
        source_manifest_paths=[manifest],
        features_dir=features_dir,
        m8_path=m8,
        source_prefix="trainmsa_",
        output_path=out_filter,
        audit_summary_path=audit,
    )

    excluded = set(payload["excluded_sequence_sha256"])
    assert _sequence_sha256(first) in excluded
    assert _sequence_sha256(second) in excluded
    assert payload["excluded_sequence_count"] == 2
    supplemental = payload["supplemental_filters"][-1]
    assert supplemental["m8_hit_count"] == 1
    assert supplemental["source_record_count"] == 2
    assert supplemental["new_excluded_sequence_count"] == 1


def test_residual_closure_rejects_changed_source_order_or_alignment(tmp_path: Path) -> None:
    import pytest

    module = _load_extend_module()
    manifest = tmp_path / "train.txt"
    manifest.write_text("toy_A\n")
    features = tmp_path / "features"
    _write_feature(features, "toy_A", ["A" * 20, "C" * 20, "D" * 20])
    base = tmp_path / "base.json"
    base.write_text(json.dumps({"excluded_sequence_sha256": []}))
    m8 = tmp_path / "residual.m8"
    m8.write_text("valtarget_1\ttrainmsa_0\t90\t20\t1\t1\t1e-10\t40\n")
    audit = _write_audit_binding(module, tmp_path, manifest, features, m8)
    output = tmp_path / "filter.json"
    kwargs = dict(base_filter_path=base, source_manifest_paths=[manifest], features_dir=features,
                  m8_path=m8, source_prefix="trainmsa_", output_path=output, audit_summary_path=audit)
    _write_feature(features, "toy_A", ["A" * 20, "D" * 20, "C" * 20])
    with pytest.raises(ValueError, match="ordering changed"):
        module.extend_filter_from_audit(**kwargs)
    assert not output.exists()
    _write_feature(features, "toy_A", ["A" * 20, "C" * 20, "D" * 20])
    m8.write_text("")
    with pytest.raises(ValueError, match="does not match the bound audit"):
        module.extend_filter_from_audit(**kwargs)
    assert not output.exists()
