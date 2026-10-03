from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import pytest

from scripts import build_full_source_metadata as full


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    cache = tmp_path / "chain_data_cache.json"
    cache.write_text(json.dumps({
        "1aaa_A": {"seq": "A" * 600, "resolution": 2.0},
        "2aaa_A": {"seq": "C" * 50, "resolution": 2.0},
        "3aaa_A": {"seq": "D" * 1000, "resolution": 4.0},
        "4aaa_A": {"seq": "E" * 39, "resolution": 2.0},
        "5aaa_A": {"seq": "AX" * 20, "resolution": 2.0},
        "6aaa_A": {"seq": "G" * 80, "resolution": 0.0},
    }))
    sources = tmp_path / "sources"
    sources.mkdir()
    texts = {
        "cath_domain_list": "1aaaA00 1 10 20 30 1 1 1 1 1 600 2.0\n3aaaA00 2 20 30 40 1 1 1 1 1 1000 4.0\n",
        "cath_domain_boundaries_seqres": "1aaaA00 1-600\n3aaaA00 1-1000\n",
        "cath_domain_sequences": ">cath|4_4_0|1aaaA00/1-600\n" + "A" * 600 + "\n>cath|4_4_0|3aaaA00/1-1000\n" + "D" * 1000 + "\n",
    }
    records = {}
    for name, text in texts.items():
        path = sources / full.base.SOURCE_FILENAMES[name]
        path.write_text(text)
        records[name] = {"path": path.name, "sha256": full.base._sha256(path), "version": "4.4.0"}
    lock = sources / "sources.lock.json"
    lock.write_text(json.dumps({"schema_version": 2, "chain_data_cache": {"sha256": full.base._sha256(cache)}, "sources": records}))
    return cache, sources, lock


def test_all_snapshot_parents_and_rejection_mass_are_explicit(tmp_path: Path) -> None:
    cache, sources, lock = _fixture(tmp_path)
    metadata, _ = full.compute_full_source_metadata(
        chain_data_cache=cache, metadata_sources_dir=sources, metadata_source_lock=lock,
    )
    scope = metadata["source"]["full_source_scope"]
    assert scope["source_chain_count"] == 6
    assert scope["source_unique_sequence_count"] == 6
    assert scope["source_observed_max_parent_length"] == 1000
    assert scope["metadata_eligible_max_parent_length"] == 600
    assert scope["metadata_eligible_chain_count"] == 3
    assert scope["metadata_eligible_unknown_resolution_count"] == 1
    assert scope["metadata_reject_reason_counts_overlap"] == {"min_length": 1, "resolution": 1, "unknown_aa_fraction": 1}
    assert scope["parent_upper_length_filter"] is None
    assert metadata["summary"]["candidate_count"] == 6
    assert metadata["summary"]["accepted"] == 3
    assert metadata["summary"]["rejected"] == 3
    assert len(metadata["chains"]) == 6
    by_id = {row["chain_id"]: row for row in metadata["chains"]}
    assert by_id["1aaa_A"]["cath_domains"][0]["domain_length"] == 600
    assert by_id["1aaa_A"]["sequence_sha256"] == hashlib.sha256(b"A" * 600).hexdigest()
    assert by_id["3aaa_A"]["accepted"] is False
    assert by_id["3aaa_A"]["reject_reasons"] == ["resolution"]
    assert by_id["3aaa_A"]["cath_domains"][0]["topology_id"] == "2.20.30"
    assert by_id["3aaa_A"]["cath_domains"][0]["domain_length"] is None


def test_portable_transport_replay_and_tamper_detection(tmp_path: Path) -> None:
    cache, sources, lock = _fixture(tmp_path)
    out = tmp_path / "output"
    full.main(["--chain-data-cache", str(cache), "--metadata-sources-dir", str(sources),
               "--metadata-source-lock", str(lock), "--out-dir", str(out)])
    moved = tmp_path / "other-machine"
    moved.mkdir()
    shutil.copytree(sources, moved / "sources")
    shutil.copyfile(cache, moved / "renamed-cache.json")
    kwargs: dict[str, Any] = dict(metadata_path=out / "structure_metadata.json", receipt_path=out / "receipt.json",
                  chain_data_cache=moved / "renamed-cache.json", metadata_sources_dir=moved / "sources",
                  metadata_source_lock=moved / "sources" / lock.name)
    full.validate_full_source_metadata(**kwargs)
    metadata = json.loads((out / "structure_metadata.json").read_text())
    metadata["chains"][0]["cath_domains"][0]["domain_length"] = 599
    (out / "structure_metadata.json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="checksum mismatch"):
        full.validate_full_source_metadata(**kwargs)
    receipt = json.loads((out / "receipt.json").read_text())
    receipt["metadata"]["sha256"] = full.base._sha256(out / "structure_metadata.json")
    (out / "receipt.json").write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="differs from source replay"):
        full.validate_full_source_metadata(**kwargs)


def test_source_mutation_prevents_publication(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cache, sources, lock = _fixture(tmp_path)
    original = full.base.build_metadata
    def mutate(*args: Any, **kwargs: Any) -> dict[str, Any]:
        result = original(*args, **kwargs)
        cache.write_text(cache.read_text() + "\n")
        return result
    monkeypatch.setattr(full.base, "build_metadata", mutate)
    out = tmp_path / "never-published"
    with pytest.raises(ValueError, match="SHA-256 mismatch|changed during processing"):
        full.main(["--chain-data-cache", str(cache), "--metadata-sources-dir", str(sources),
                   "--metadata-source-lock", str(lock), "--out-dir", str(out)])
    assert not out.exists()


def test_active_narrow_source_without_skipped_flag_is_rejected(tmp_path: Path) -> None:
    cache, sources, lock = _fixture(tmp_path)
    narrow = sources / "rcsb_chain_metadata.jsonl"
    narrow.write_text("{}\n")
    payload = json.loads(lock.read_text())
    payload["sources"]["rcsb_chain_metadata"] = {"path": narrow.name, "sha256": full.base._sha256(narrow)}
    lock.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="bulk-only classification"):
        full.compute_full_source_metadata(chain_data_cache=cache, metadata_sources_dir=sources, metadata_source_lock=lock)
