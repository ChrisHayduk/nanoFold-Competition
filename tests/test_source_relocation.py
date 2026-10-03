from __future__ import annotations

import copy
import json
import shutil
from pathlib import Path
from typing import Any

import pytest
from representative_eligibility_helpers import source_backed_eligibility_fixture, unit_id, write_json

from nanofold.representative_eligibility import file_record, historical_source_catalog, inspect_candidate
from nanofold.structural_population import validate_source_catalog


@pytest.fixture
def relocated_sources(tmp_path: Path) -> dict[str, Any]:
    """Physically move a real acquisition without rewriting its saved evidence."""
    cache, metadata, groups, graph = [tmp_path / name for name in ("cache.json", "metadata.json", "groups.tsv", "graph.json")]
    sequence = "A" * 40
    write_json(cache, {"1aaa_A": {"sequence": sequence}})
    write_json(metadata, {"schema_version": 2, "chains": []})
    groups.write_text("group\t1aaa_A\n")
    write_json(graph, {"fixture": "independently owned graph proof"})
    original = source_backed_eligibility_fixture(
        tmp_path / "original", cache_path=cache, metadata_path=metadata, groups_path=groups,
        graph_receipt_path=graph, sequences={"1aaa_A": sequence}, cluster_map={"1aaa_A": "1aaa_A"},
        group_map={"1aaa_A": "group"}, original_units={unit_id(["1aaa_A"]): ["1aaa_A"]},
    )
    portable = tmp_path / "portable"
    code_root = portable / "acquisition-code"
    repo = Path(__file__).resolve().parents[1]
    code_bindings = []
    for relative in ("scripts/fetch_openproteinset_mmcif.py", "nanofold/structure_metadata.py"):
        target = code_root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(repo / relative, target)
        code_bindings.append(file_record(target))
    paths = {
        "receipt": (original["retrieval"], portable / "receipts/immutable.json"),
        "manifest": (original["historical_root"] / "manifest.txt", portable / "requests/chains.list"),
        "structure": (original["historical_root"] / "pdb_data/mmcif_files/1aaa.cif", portable / "structures/source.cif"),
    }
    original_bytes = {key: first.read_bytes() for key, (first, _) in paths.items()}
    relocation_rows = []
    for first, last in paths.values():
        original_binding = file_record(first)
        last.parent.mkdir(parents=True, exist_ok=True)
        first.rename(last)
        relocation_rows.append(file_record(last) | {
            "original_path": str(first), "original_sha256": original_binding["sha256"],
        })
    relocation_path = portable / "source-relocations.json"
    write_json(relocation_path, {"status": "verified_byte_identical_relocation", "files": relocation_rows})
    source_map = {row["original_path"]: row["path"] for row in relocation_rows}
    sources, members = historical_source_catalog(
        [paths["receipt"][0]], acquisition_code_root=code_root, source_path_map=source_map,
    )
    catalog = {"status": "complete", "inputs": code_bindings, "source_relocations": file_record(relocation_path),
               "historical_sources": sources, "members": members}
    return {"paths": paths, "code_root": code_root, "source_map": source_map, "catalog": catalog,
            "relocation_path": relocation_path, "original_bytes": original_bytes,
            "raw_root": original["raw_root"], "sequence": sequence}


def test_immutable_acquisition_evidence_replays_after_real_relocation(relocated_sources: dict[str, Any]) -> None:
    fixture = relocated_sources
    for key, (original, current) in fixture["paths"].items():
        assert not original.exists()
        assert current.read_bytes() == fixture["original_bytes"][key]
    receipt = fixture["paths"]["receipt"][1]
    assert json.loads(receipt.read_text())["inputs"]["manifest_path"] == str(fixture["paths"]["manifest"][0])
    # Either receipt location spelling must resolve to the same immutable evidence.
    for path in fixture["paths"]["receipt"]:
        sources, members = historical_source_catalog(
            [path], acquisition_code_root=fixture["code_root"], source_path_map=fixture["source_map"],
        )
        assert sources == fixture["catalog"]["historical_sources"]
        assert members == fixture["catalog"]["members"]
    validate_source_catalog(fixture["catalog"])
    result = inspect_candidate(chain="1aaa_A", sequence=fixture["sequence"], rank=0,
                               structure=members["1aaa"], raw_roots=[fixture["raw_root"]])
    assert result["status"] == "passed"
    assert result["structure"]["path"] == str(fixture["paths"]["structure"][1])


@pytest.mark.parametrize("changed", ["manifest", "acquisition_code"])
def test_relocated_manifest_and_code_still_match_owning_receipt_hashes(
    relocated_sources: dict[str, Any], changed: str,
) -> None:
    fixture = relocated_sources
    target = (fixture["paths"]["manifest"][1] if changed == "manifest"
              else fixture["code_root"] / "nanofold/structure_metadata.py")
    target.write_bytes(target.read_bytes() + b"\n# changed after transfer\n")
    with pytest.raises(ValueError, match="request manifest changed|code differs"):
        historical_source_catalog([fixture["paths"]["receipt"][1]], acquisition_code_root=fixture["code_root"],
                                  source_path_map=fixture["source_map"])


def test_changed_relocated_structure_rejected_by_source_qualification(relocated_sources: dict[str, Any]) -> None:
    fixture = relocated_sources
    target = fixture["paths"]["structure"][1]
    target.write_bytes(target.read_bytes() + b"\n# changed after transfer\n")
    # Catalog construction binds expected archive hashes; physical structure
    # hashing belongs to qualification and final replay, including after a move.
    with pytest.raises(ValueError, match="differs from its pinned acquisition"):
        inspect_candidate(chain="1aaa_A", sequence=fixture["sequence"], rank=0,
                          structure=fixture["catalog"]["members"]["1aaa"], raw_roots=[fixture["raw_root"]])


@pytest.mark.parametrize("source", ["manifest", "structure"])
@pytest.mark.parametrize("mutation", ["unmapped", "forged_origin"])
def test_relocation_requires_the_exact_origin_bound_by_acquisition(
    relocated_sources: dict[str, Any], source: str, mutation: str,
) -> None:
    fixture = relocated_sources
    relocation = json.loads(fixture["relocation_path"].read_text())
    original = str(fixture["paths"][source][0])
    if mutation == "unmapped":
        relocation["files"] = [row for row in relocation["files"] if row["original_path"] != original]
    else:
        next(row for row in relocation["files"] if row["original_path"] == original)["original_path"] = str(
            Path(original).parent / "forged-origin"
        )
    write_json(fixture["relocation_path"], relocation)
    catalog = copy.deepcopy(fixture["catalog"])
    # Rebinding the mapping itself cannot substitute a different logical source.
    catalog["source_relocations"] = file_record(fixture["relocation_path"])
    error = "No such file" if source == "manifest" else "without a bound relocation"
    with pytest.raises((FileNotFoundError, ValueError), match=error):
        validate_source_catalog(catalog)
