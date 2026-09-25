"""Real raw A3M/mmCIF fixtures for source-replayed representative qualification."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import gemmi

from nanofold.chain_paths import chain_data_dir
from nanofold.representative_eligibility import (
    ARCHIVE_URL,
    SOURCE_POLICY,
    THRESHOLDS,
    approved_exclusion_catalog,
    bind_approval_decision,
    file_record,
    historical_source_catalog,
    inspect_candidate,
    population_record,
    predicate_record,
)

RESIDUES = dict(zip("ARNDCQEGHILKMFPSTWYV", (
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
), strict=True))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")


def write_structure(path: Path, sequence: str, *, observed: int | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    structure = gemmi.Structure()
    structure.name = path.stem
    model, chain = gemmi.Model(1), gemmi.Chain("A")
    for index, letter in enumerate(sequence[:observed]):
        residue = gemmi.Residue()
        residue.name = RESIDUES[letter]
        residue.seqid = gemmi.SeqId(index + 1, " ")
        residue.label_seq = index + 1
        residue.subchain = "Apoly"
        residue.entity_type = gemmi.EntityType.Polymer
        for name, dx in (("N", -1.0), ("CA", 0.0), ("C", 1.0), ("O", 2.0)):
            atom = gemmi.Atom()
            atom.name = name
            atom.element = gemmi.Element(name[0])
            atom.pos = gemmi.Position(index * 3.8, dx, 0.0)
            atom.occ = 1.0
            residue.add_atom(atom)
        chain.add_residue(residue)
    model.add_chain(chain)
    structure.add_model(model)
    structure.setup_entities()
    for entity in structure.entities:
        entity.full_sequence = [RESIDUES[letter] for letter in sequence]
    structure.make_mmcif_document().write_file(str(path))


def source_backed_eligibility_fixture(
    root: Path, *, cache_path: Path, metadata_path: Path, groups_path: Path, graph_receipt_path: Path,
    sequences: dict[str, str], cluster_map: dict[str, str], group_map: dict[str, str],
    original_units: dict[str, list[str]], observed: dict[str, int] | None = None,
    excluded_groups: list[str] | None = None, seed: int = 0,
) -> dict[str, Any]:
    """Create physical sources and fresh outcomes, never hand-written pass/fail."""
    repo = Path(__file__).resolve().parents[1]
    acquisition, raw_root = root / "historical", root / "raw"
    acquisition.mkdir(parents=True)
    manifest = acquisition / "manifest.txt"
    manifest.write_text("\n".join(sorted(sequences)) + "\n")
    members = []
    for chain, sequence in sorted(sequences.items()):
        pdb = chain.split("_", 1)[0]
        cif = acquisition / "pdb_data/mmcif_files" / f"{pdb}.cif"
        write_structure(cif, sequence, observed=(observed or {}).get(chain))
        members.append({"output_path": str(cif.relative_to(acquisition)), "output_sha256": file_record(cif)["sha256"],
                        "crc32_verified": True, "local_header_verified": True,
                        "archive_member": {"name": f"mmcif_files/{pdb}.cif"}})
        a3m = chain_data_dir(raw_root / "roda_pdb", chain) / "uniref90_hits.a3m"
        a3m.parent.mkdir(parents=True, exist_ok=True)
        a3m.write_text(f">query\n{sequence}\n>homolog\n{sequence}\n")
    retrieval = acquisition / "retrieval-receipt.json"
    write_json(retrieval, {
        "status": "retrieved_unvalidated_structures", "scope": "selective_archive_retrieval_only",
        "source_inputs_unchanged": True, "archive_url": ARCHIVE_URL,
        "archive_pin": {"size": 10000000, "etag": '"fixture-pin"', "version_id": "null"},
        "directory": {"central_directory_sha256": "d" * 64}, "members": members,
        "retrieved_structure_count": len(members), "requested_structure_count": len(members),
        "inputs": {"manifest_path": str(manifest), "manifest_sha256": file_record(manifest)["sha256"],
                   "script_sha256": file_record(repo / "scripts/fetch_openproteinset_mmcif.py")["sha256"],
                   "manifest_validator_sha256": file_record(repo / "nanofold/structure_metadata.py")["sha256"]},
    })
    decision = root / "decision.json"
    write_json(decision, {"excluded_group_ids": excluded_groups or [], "source_policy": SOURCE_POLICY,
                          "thresholds": THRESHOLDS, "authorization": "Explicit synthetic fixture approval",
                          "all_retained_groups_required": True, "whole_group_split_ownership": True,
                          "original_group_count": len(set(group_map.values())),
                          "retained_group_count": len(set(group_map.values())) - len(excluded_groups or [])})
    bound_decision = root / "bound-decision.json"
    write_json(bound_decision, bind_approval_decision(
        decision, split_group_tsv_path=groups_path, split_group_receipt_path=graph_receipt_path,
    ))
    catalog_path = root / "approved-exclusions.json"
    write_json(catalog_path, approved_exclusion_catalog(bound_decision))
    historical, structure_members = historical_source_catalog([retrieval])
    records = []
    for unit_id, ranked in sorted(original_units.items()):
        outcomes, selected = [], None
        for rank, chain in enumerate(ranked):
            result = inspect_candidate(
                chain=chain, sequence=sequences[chain], rank=rank,
                structure=structure_members[chain.split("_", 1)[0]], raw_roots=[raw_root.resolve()],
            )
            outcomes.append(result)
            if result["status"] == "passed":
                selected = chain
                break
        records.append({"unit_id": unit_id, "full_group_id": group_map[ranked[0]], "ranked_members": ranked,
                        "outcomes": outcomes, "selected_chain_id": selected})
    exhausted = sum(record["selected_chain_id"] is None for record in records)
    passing_groups = {record["full_group_id"] for record in records if record["selected_chain_id"] is not None}
    receipt_path = root / "eligibility.json"
    population = population_record(
        chain_data_cache_path=cache_path, structure_metadata_path=metadata_path,
        cluster_map=cluster_map, split_group_map=group_map, split_group_receipt_path=graph_receipt_path,
        original_units=original_units, seed=seed,
    )
    write_json(receipt_path, {
        "schema_version": 1, "policy": "cath-v4", "status": "complete", "population": population,
        "predicate": predicate_record(), "original_structure_metadata": file_record(metadata_path),
        "historical_sources": historical, "approved_group_exclusions_sha256": file_record(catalog_path)["sha256"],
        "units": records, "counts": {
            "eligible_unit_count": len(records) - exhausted, "exhausted_unit_count": exhausted,
            "excluded_group_count": len(set(group_map.values()) - passing_groups), "retained_group_count": len(passing_groups),
        },
    })
    return {"receipt_path": receipt_path, "catalog_path": catalog_path, "retrieval": retrieval, "raw_root": raw_root,
            "decision": decision, "bound_decision": bound_decision, "historical_root": acquisition}


def unit_id(members: list[str]) -> str:
    return hashlib.sha256(("\n".join(sorted(members)) + "\n").encode()).hexdigest()
