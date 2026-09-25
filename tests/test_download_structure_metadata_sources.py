from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts import download_structure_metadata_sources as downloader


def _fixture_args(tmp_path: Path) -> tuple[list[str], Path, Path, Path]:
    cache = tmp_path / "chains.json"
    cache.write_text(json.dumps({"1abc_A": {"sequence": "A" * 50, "resolution": 2.0}}))
    out = tmp_path / "sources"
    lock = out / "sources.lock.json"
    return ["--chain-data-cache", str(cache), "--out-dir", str(out), "--source-lock", str(lock)], cache, out, lock


def _fake_download(url: str, destination: Path, **kwargs: object) -> bool:
    if not kwargs.get("dry_run"):
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(f"fixture source {url}\n")
    return True


def _no_network(*args: object, **kwargs: object) -> bool:
    pytest.fail("Verified source-lock reuse must not download or repin any source")


def test_download_pins_source_bytes_and_reuses_verified_lock(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    args, cache, out, lock = _fixture_args(tmp_path)
    monkeypatch.setattr(downloader, "_download_url", _fake_download)
    downloader.main(args)
    original_lock = lock.read_bytes()
    payload = json.loads(original_lock)
    assert payload["schema_version"] == 2
    assert payload["chain_data_cache"]["sha256"] == hashlib.sha256(cache.read_bytes()).hexdigest()
    assert payload["candidate_chain_count"] == 1
    assert payload["candidate_filters"]["require_monomer"] is False
    cath = payload["sources"]["cath_domain_list"]
    assert "/all-releases/v4_4_0/" in cath["url"]
    assert cath["version"] == "4.4.0"
    for name in downloader.BULK_SOURCES:
        source = payload["sources"][name]
        assert Path(source["path"]).name == source["path"]
        assert source["sha256"] == hashlib.sha256((out / source["path"]).read_bytes()).hexdigest()
    monkeypatch.setattr(downloader, "_download_url", _no_network)
    downloader.main(args)
    assert lock.read_bytes() == original_lock


@pytest.mark.parametrize("changed_input", ["cache", "source", "deleted_source"])
def test_changed_inputs_cannot_be_silently_repinned(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, changed_input: str
) -> None:
    args, cache, out, lock = _fixture_args(tmp_path)
    monkeypatch.setattr(downloader, "_download_url", _fake_download)
    downloader.main(args)
    original_lock = lock.read_bytes()
    source = out / "cath-domain-list.txt"
    if changed_input == "cache":
        cache.write_text(cache.read_text() + "\n")
    elif changed_input == "source":
        source.write_text("different classification bytes\n")
    else:
        source.unlink()
    monkeypatch.setattr(downloader, "_download_url", _no_network)
    with pytest.raises(ValueError, match="hash mismatch|Missing or invalid source"):
        downloader.main(args)
    assert lock.read_bytes() == original_lock


def test_required_failure_publishes_nothing_and_can_retry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    args, _, out, lock = _fixture_args(tmp_path)
    monkeypatch.setattr(downloader, "_download_url", lambda *args, **kwargs: False)
    with pytest.raises(SystemExit, match="Required metadata source download failed"):
        downloader.main(args)
    assert not lock.exists()
    assert list(out.iterdir()) == []
    monkeypatch.setattr(downloader, "_download_url", _fake_download)
    downloader.main(args)
    assert lock.exists()


def test_cache_changed_during_download_does_not_publish_stale_scope(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args, cache, out, lock = _fixture_args(tmp_path)

    def fake_download(url: str, destination: Path, **kwargs: object) -> bool:
        cache.write_text(cache.read_text() + "\n")
        return _fake_download(url, destination, **kwargs)

    monkeypatch.setattr(downloader, "_download_url", fake_download)
    with pytest.raises(ValueError, match="cache changed during source download"):
        downloader.main(args)
    assert not lock.exists()
    assert list(out.iterdir()) == []


def test_optional_failure_is_preserved_in_verified_lock(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    args, _, out, lock = _fixture_args(tmp_path)

    def fake_download(url: str, destination: Path, **kwargs: object) -> bool:
        return False if destination.name == "dir.cla.scope.txt" else _fake_download(url, destination, **kwargs)

    monkeypatch.setattr(downloader, "_download_url", fake_download)
    downloader.main(args)
    payload = json.loads(lock.read_text())
    assert payload["sources"]["scope_classification"]["skipped"] is True
    assert payload["sources"]["scope_classification"]["errors"] == 1
    assert not (out / "dir.cla.scope.txt").exists()
    monkeypatch.setattr(downloader, "_download_url", _no_network)
    downloader.main(args)
    assert json.loads(lock.read_text()) == payload


def test_dry_run_preserves_existing_rcsb_file_and_creates_no_directories(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args, _, out, lock = _fixture_args(tmp_path)
    monkeypatch.setattr(downloader, "_download_url", _fake_download)
    downloader.main([*args, "--dry-run"])
    assert not out.exists()
    out.mkdir()
    rcsb = out / "rcsb_chain_metadata.jsonl"
    rcsb.write_text('{"preserve": true}\n')
    downloader.main([*args, "--dry-run"])
    assert rcsb.read_text() == '{"preserve": true}\n'
    assert not lock.exists()
    assert list(out.iterdir()) == [rcsb]


def test_existing_unpinned_sources_are_not_adopted(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    args, _, out, lock = _fixture_args(tmp_path)
    out.mkdir()
    source = out / "cath-domain-list.txt"
    source.write_text("unverifiable provenance\n")
    monkeypatch.setattr(downloader, "_download_url", _no_network)
    with pytest.raises(ValueError, match="Unpinned source already exists"):
        downloader.main(args)
    assert not lock.exists()
    assert source.read_text() == "unverifiable provenance\n"


def test_incomplete_old_lock_is_not_replaced(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    args, _, out, lock = _fixture_args(tmp_path)
    out.mkdir()
    lock.write_text('{"sources": {}}\n')
    monkeypatch.setattr(downloader, "_download_url", _no_network)
    with pytest.raises(ValueError, match="schema_version=2"):
        downloader.main(args)
    assert lock.read_text() == '{"sources": {}}\n'


@pytest.mark.parametrize("extra_args", [["--include-rcsb-rest"], ["--min-len", "41"]])
def test_changed_download_scope_requires_a_new_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, extra_args: list[str]
) -> None:
    args, _, _, lock = _fixture_args(tmp_path)
    monkeypatch.setattr(downloader, "_download_url", _fake_download)
    downloader.main(args)
    original_lock = lock.read_bytes()
    monkeypatch.setattr(downloader, "_download_url", _no_network)
    with pytest.raises(ValueError, match="differ.*source lock"):
        downloader.main([*args, *extra_args])
    assert lock.read_bytes() == original_lock


def test_optional_service_failures_remain_visible_on_reuse(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    args, cache, out, lock = _fixture_args(tmp_path)
    cache.write_text(
        json.dumps({chain: {"sequence": "A" * 50, "resolution": 2.0} for chain in ["1abc_A", "1abc_B", "2abc_A"]})
    )

    def fake_download(url: str, destination: Path, **kwargs: object) -> bool:
        return False if destination.name == "2abc.cif" else _fake_download(url, destination, **kwargs)

    def fake_json(url: str, **kwargs: object) -> tuple[dict[str, object] | None, str | None]:
        if "/entry/" in url:
            return None, "entry service unavailable"
        return {
            "rcsb_polymer_entity_instance_container_identifiers": {
                "entity_id": "1",
                "auth_asym_id": url.rsplit("/", 1)[1],
            }
        }, None

    monkeypatch.setattr(downloader, "_download_url", fake_download)
    monkeypatch.setattr(downloader, "_read_json_url", fake_json)
    args.extend(["--include-rcsb-rest", "--include-dssp-mmcif"])
    downloader.main(args)
    payload = json.loads(lock.read_text())
    assert payload["sources"]["rcsb_chain_metadata"]["errors"] == 2
    rcsb_records = [json.loads(line) for line in (out / "rcsb_chain_metadata.jsonl").read_text().splitlines()]
    assert all(record["errors"] == ["entry service unavailable"] for record in rcsb_records)
    assert payload["sources"]["dssp_mmcif"]["failed_pdb_count"] == 1
    assert payload["sources"]["dssp_mmcif"]["tree_sha256"] == downloader._tree_sha256(out / "dssp_mmcif")
    monkeypatch.setattr(downloader, "_download_url", _no_network)
    monkeypatch.setattr(downloader, "_read_json_url", _no_network)
    downloader.main(args)
    assert json.loads(lock.read_text()) == payload
    (out / "dssp_mmcif" / "1abc.cif").write_text("changed mmcif bytes\n")
    with pytest.raises(ValueError, match="Source hash mismatch: dssp_mmcif"):
        downloader.main(args)


@pytest.mark.parametrize("author_chain", [None, "B", "A"])
def test_rcsb_instance_requires_verified_author_chain_mapping(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, author_chain: str | None
) -> None:
    requested_urls: list[str] = []

    def fake_json(url: str, **kwargs: object) -> tuple[dict[str, object], None]:
        requested_urls.append(url)
        if "/entry/" in url:
            return {"rcsb_entry_container_identifiers": {"polymer_entity_ids": ["1"]}}, None
        if "/polymer_entity_instance/" in url:
            identifiers = {"entity_id": "1", "asym_id": "A", "entry_id": "1ABC"}
            if author_chain is not None:
                identifiers["auth_asym_id"] = author_chain
            return {"rcsb_polymer_entity_instance_container_identifiers": identifiers}, None
        return {"rcsb_polymer_entity_container_identifiers": {"asym_ids": ["A"]}}, None

    monkeypatch.setattr(downloader, "_read_json_url", fake_json)
    destination = tmp_path / "rcsb.jsonl"
    summary = downloader._download_rcsb_jsonl(["1abc_A"], destination, dry_run=False, retries=0, retry_delay_seconds=0)
    record = json.loads(destination.read_text())
    assert record["entry"] == {"rcsb_entry_container_identifiers": {"polymer_entity_ids": ["1"]}}
    if author_chain == "A":
        assert record["instance"]
        assert record["entity"] == {"rcsb_polymer_entity_container_identifiers": {"asym_ids": ["A"]}}
        assert record["errors"] == []
        assert summary["errors"] == 0
        assert len(requested_urls) == 3
    else:
        assert record["instance"] == {}
        assert record["entity"] == {}
        assert "mapping" in record["errors"][0]
        assert summary["errors"] == 1
        assert len(requested_urls) == 3


@pytest.mark.parametrize("ambiguous", [False, True])
def test_rcsb_resolves_author_chain_through_label_asym_crosswalk(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, ambiguous: bool,
) -> None:
    """RCSB 2OR1 uses label C/D for author L/R, not the same-letter endpoint."""
    requested: list[str] = []

    def fake_json(url: str, **kwargs: object) -> tuple[dict[str, object], None]:
        requested.append(url)
        if "/entry/" in url:
            return {"rcsb_entry_container_identifiers": {"polymer_entity_ids": ["3"]}}, None
        if "/polymer_entity/" in url:
            return {
                "rcsb_polymer_entity_container_identifiers": {"asym_ids": ["C", "D"]},
                "rcsb_entity_source_organism": [{"ncbi_taxonomy_id": 10712}],
            }, None
        label = url.rsplit("/", 1)[1]
        assert label in {"C", "D"}
        return {"rcsb_polymer_entity_instance_container_identifiers": {
            "entity_id": "3", "entry_id": "2OR1", "asym_id": label,
            "auth_asym_id": "L" if ambiguous or label == "C" else "R",
        }}, None

    monkeypatch.setattr(downloader, "_read_json_url", fake_json)
    destination = tmp_path / "rcsb.jsonl"
    summary = downloader._download_rcsb_jsonl(
        ["2or1_L", "2or1_R"], destination, dry_run=False, retries=0, retry_delay_seconds=0,
    )
    records = [json.loads(line) for line in destination.read_text().splitlines()]
    assert len(requested) == 4  # Both candidate chains share a single verified entry crosswalk.
    if ambiguous:
        assert all(record["instance"] == {} and record["entity"] == {} for record in records)
        assert summary["errors"] == 2
    else:
        assert summary["errors"] == 0
        assert records[0]["instance_url"].endswith("/2or1/C")
        assert records[1]["instance_url"].endswith("/2or1/D")
        assert records[0]["entity"]["rcsb_entity_source_organism"][0]["ncbi_taxonomy_id"] == 10712


def test_all_dssp_failures_are_explicitly_skipped_in_reusable_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args, _, out, lock = _fixture_args(tmp_path)

    def fake_download(url: str, destination: Path, **kwargs: object) -> bool:
        return False if destination.suffix == ".cif" else _fake_download(url, destination, **kwargs)

    monkeypatch.setattr(downloader, "_download_url", fake_download)
    args.append("--include-dssp-mmcif")
    downloader.main(args)
    payload = json.loads(lock.read_text())
    assert payload["sources"]["dssp_mmcif"]["skipped"] is True
    assert payload["sources"]["dssp_mmcif"]["failed_pdb_count"] == 1
    assert payload["sources"]["dssp_mmcif"]["tree_sha256"] is None
    assert not (out / "dssp_mmcif").exists()
    monkeypatch.setattr(downloader, "_download_url", _no_network)
    downloader.main(args)
    assert json.loads(lock.read_text()) == payload


def test_candidate_scope_uses_shared_monomer_and_length_rules(tmp_path: Path) -> None:
    cache = tmp_path / "chains.json"
    cache.write_text(
        json.dumps(
            {
                "1abc_A": {"sequence": "A" * 50, "resolution": 2.0},
                "2abc_A": {"sequence": "A" * 50, "resolution": 2.0, "oligomeric_count": 1},
                "3abc_A": {"sequence": "A" * 50, "resolution": 2.0, "seq_length": 40},
                "4abc_A": {"sequence": "A" * 50, "resolution": 2.0, "num_chains": 1},
            }
        )
    )
    filters = {"min_len": 40, "max_len": 256, "max_resolution": 3.0, "max_unknown_aa_fraction": 0.0}
    assert downloader._candidate_chain_ids(cache, **filters) == ["1abc_A", "2abc_A", "4abc_A"]
    assert downloader._candidate_chain_ids(cache, require_monomer=True, **filters) == ["2abc_A"]
