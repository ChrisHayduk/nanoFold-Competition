"""Build portable, pinned metadata for every eligible parent in an OPS snapshot."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
for search_path in (REPO_ROOT, Path(__file__).resolve().parent):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

import build_structure_metadata as base

IMPLEMENTATION_FILES = (
    "scripts/build_full_source_metadata.py", "scripts/build_structure_metadata.py",
    "nanofold/cath_boundaries.py", "nanofold/structure_metadata.py",
)
SCHEMA_VERSION = 1


def _pins(cache: Path, sources_dir: Path, source_lock: Path) -> dict[str, Any]:
    verified = base._verify_source_lock(source_lock, sources_dir, cache)
    return {
        "chain_data_cache": base._sha256(cache),
        "metadata_source_lock": base._sha256(source_lock),
        "metadata_sources": {
            name: {"path": base.SOURCE_FILENAMES[name], "sha256": record["sha256"]}
            for name, record in verified.items() if name != "dssp_mmcif"
        },
        "implementation": {name: base._sha256(REPO_ROOT / name) for name in IMPLEMENTATION_FILES},
    }


def _check_pins(expected: dict[str, Any], cache: Path, sources_dir: Path, source_lock: Path) -> None:
    if _pins(cache, sources_dir, source_lock) != expected:
        raise ValueError("Full-source metadata inputs or implementation changed during processing")


def _length_bin(length: int) -> str:
    for ceiling in (39, 256, 512, 1024):
        if length <= ceiling:
            return {39: "under_40", 256: "40_to_256", 512: "257_to_512", 1024: "513_to_1024"}[ceiling]
    return "over_1024"


def _scope(
    raw: dict[str, Any], eligible: list[dict[str, Any]], rejects: Counter[str], observed_max: int,
) -> dict[str, Any]:
    sequences = [sequence for row in raw.values() if isinstance(row, dict)
                 and (sequence := base._extract_sequence(row))]
    eligible_sequences = {row["sequence"] for row in eligible}
    all_sequences = set(sequences)
    return {
        "population": "All chains present in the pinned OpenProteinSet PDB chain_data_cache snapshot",
        "source_chain_count": len(raw),
        "source_unique_sequence_count": len(all_sequences),
        "source_chain_count_with_sequence": len(sequences),
        "source_observed_max_parent_length": observed_max,
        "source_length_bins": dict(sorted(Counter(_length_bin(len(sequence)) for sequence in sequences).items())),
        "metadata_eligible_chain_count": len(eligible),
        "metadata_eligible_unique_sequence_count": len(eligible_sequences),
        "metadata_ineligible_chain_count": len(raw) - len(eligible),
        "unrepresented_unique_sequences_after_metadata_filters": len(all_sequences - eligible_sequences),
        "metadata_reject_reason_counts_overlap": dict(sorted(rejects.items())),
        "metadata_eligible_length_bins": dict(sorted(Counter(_length_bin(row["length"]) for row in eligible).items())),
        "metadata_eligible_max_parent_length": max((row["length"] for row in eligible), default=0),
        "metadata_eligible_unknown_resolution_count": sum(row["resolution"] == 999.0 for row in eligible),
        "parent_upper_length_filter": None,
        "qualification_status": "metadata_eligibility_only; atom14 and full MSA qualification not yet performed",
        "split_ownership_status": "not assigned by this artifact; apply the separately pinned heldout barrier",
        "scope_gap": "This PDB-chain snapshot is not a census of other OpenProteinSet components or storage objects",
    }


def compute_full_source_metadata(
    *, chain_data_cache: Path, metadata_sources_dir: Path, metadata_source_lock: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    pins = _pins(chain_data_cache, metadata_sources_dir, metadata_source_lock)
    source_lock = json.loads(metadata_source_lock.read_text())
    # Per-chain supplements acquired for a narrower universe cannot establish
    # measurements across the full snapshot. This artifact uses bulk sources.
    for name in ("rcsb_chain_metadata", "dssp_mmcif"):
        if name in source_lock["sources"] and source_lock["sources"][name].get("skipped") is not True:
            raise ValueError("Full-source metadata requires bulk-only classification sources")
    raw = json.loads(chain_data_cache.read_text())
    if not isinstance(raw, dict) or not raw:
        raise ValueError("Expected a nonempty chain_data_cache snapshot")
    lengths = [len(sequence) for row in raw.values() if isinstance(row, dict)
               and (sequence := base._extract_sequence(row))]
    observed_max = max(lengths, default=0)
    if observed_max < 40:
        raise ValueError("Snapshot contains no parent of at least 40 residues")
    eligible, rejects = base._candidate_rows(
        raw, min_len=40, max_len=observed_max, max_resolution=3.0, max_unknown_aa_fraction=0.0,
    )
    scope = _scope(raw, eligible, rejects, observed_max)
    eligible_ids = {row["chain_id"] for row in eligible}
    del eligible, lengths
    with tempfile.TemporaryDirectory(prefix="nanofold-full-source-") as temporary:
        empty_exclusions = Path(temporary) / "empty-exclusions.txt"
        empty_exclusions.write_text("")
        metadata = base.build_metadata(
            chain_data_cache=chain_data_cache, metadata_sources_dir=metadata_sources_dir,
            metadata_source_lock=metadata_source_lock, feature_exclusion_list=empty_exclusions,
            processability_exclusion_list=empty_exclusions, min_len=40, max_len=observed_max,
            max_resolution=3.0, max_unknown_aa_fraction=0.0, include_cath_boundaries=True, fail_fast=True,
        )
    if metadata["summary"]["accepted"] != scope["metadata_eligible_chain_count"]:
        raise ValueError("Domain annotation unexpectedly changed the metadata-eligible population")
    external, rcsb, classifications = base._load_external_sources(
        metadata_sources_dir, {name: record for name, record in source_lock["sources"].items()
                               if not record.get("skipped")},
    )
    by_id = {row["chain_id"]: row for row in metadata["chains"]}
    for chain_id, raw_row in sorted(raw.items()):
        sequence = base._extract_sequence(raw_row) if isinstance(raw_row, dict) else None
        sequence_digest = hashlib.sha256(sequence.encode()).hexdigest() if sequence else None
        if chain_id in eligible_ids:
            by_id[chain_id]["sequence_sha256"] = sequence_digest
            continue
        _, reasons = base._candidate_rows(
            {chain_id: raw_row}, min_len=40, max_len=observed_max,
            max_resolution=3.0, max_unknown_aa_fraction=0.0,
        )
        length = len(sequence) if sequence else 0
        resolution = base._numeric(raw_row.get("resolution"), 999.0) if isinstance(raw_row, dict) else 999.0
        if resolution <= 0:
            resolution = 999.0
        fields, _, _, _ = base._source_metadata_fields(
            chain_id=chain_id, length=length, resolution=resolution, external_sources=external,
            rcsb_metadata=rcsb, cath_domains_by_chain=classifications, dssp_dir=None,
        )
        domains = [{**domain, "domain_length": None, "author_segments": [], "seqres_segments": [],
                    "label_segments": [], "boundary_status": "not_evaluated_metadata_ineligible"}
                   for domain in fields["cath_domains"]]
        metadata["chains"].append({
            **fields, "chain_id": chain_id, "pdb_id": base._pdb_id(chain_id), "length": length,
            "resolution": resolution, "accepted": False, "reject_reasons": sorted(reasons),
            "sequence_sha256": sequence_digest, "cath_domains": domains,
            "cath_domain_count": len(domains),
            "cath_topology_counts": dict(sorted(Counter(domain["topology_id"] for domain in domains).items())),
            "cath_superfamily_counts": dict(sorted(Counter(domain["homologous_superfamily_id"] for domain in domains).items())),
            "cath_boundary_status": "not_evaluated_metadata_ineligible",
            "cath_annotated_residue_count": None, "cath_unannotated_residue_count": None,
            "cath_unannotated_fraction": None, "cath_domain_order": None,
        })
    metadata["chains"].sort(key=lambda row: row["chain_id"])
    metadata["summary"]["candidate_count"] = len(raw)
    metadata["summary"]["rejected"] = len(raw) - scope["metadata_eligible_chain_count"]
    metadata["summary"]["cath_boundary_status"] = dict(sorted(Counter(
        row["cath_boundary_status"] for row in metadata["chains"]
    ).items()))
    for key in ("cath_annotation_status", "secondary_structure_class", "domain_architecture_class"):
        metadata["summary"][key] = dict(sorted(Counter(row[key] for row in metadata["chains"]).items()))
    metadata["summary"]["metadata_source_coverage"] = dict(sorted(Counter(
        name for row in metadata["chains"] for name, present in row["metadata_source_coverage"].items() if present
    ).items()))
    metadata["summary"]["oligomeric_state_population"] = "metadata_eligible_chains"
    del raw, by_id, classifications, external, rcsb
    # Logical source names keep deterministic replay valid after transport.
    metadata.pop("generated_at", None)
    source = metadata["source"]
    source["chain_data_cache"] = "chain_data_cache.json"
    source["metadata_sources_dir"] = "metadata_sources"
    source["metadata_source_lock"]["lock_path"] = "metadata_sources/structure_metadata_sources.lock.json"
    for key in ("feature_exclusion_list", "processability_exclusion_list"):
        source[key] = {"status": "not_applied_at_metadata_stage", "chain_count": 0}
    source["full_source_scope"] = scope
    metadata["config"]["parent_upper_length_filter"] = None
    metadata["config"]["max_len_semantics"] = "Observed finite snapshot maximum; no parents removed by upper length"
    metadata["config"]["prediction_target"] = "parent_chain_for_subsequent_crop_sampling"
    metadata["config"]["required_feature_assets"] = []
    metadata["config"]["label_processability"] = {"status": "deferred_to_source_qualification"}
    metadata["config"]["full_source_metadata_schema_version"] = SCHEMA_VERSION
    metadata["summary"]["cath_domain_boundary_status"] = dict(sorted(Counter(
        domain["boundary_status"] for row in metadata["chains"] for domain in row["cath_domains"]
    ).items()))
    _check_pins(pins, chain_data_cache, metadata_sources_dir, metadata_source_lock)
    return metadata, pins


def validate_full_source_metadata(
    *, metadata_path: Path, receipt_path: Path, chain_data_cache: Path,
    metadata_sources_dir: Path, metadata_source_lock: Path, replay: bool = True,
) -> dict[str, Any]:
    """Validate transported artifact identities and, by default, replay all rows."""
    receipt_digest = base._sha256(receipt_path)
    receipt = json.loads(receipt_path.read_text())
    if receipt.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported full-source metadata receipt")
    metadata_digest = base._sha256(metadata_path)
    if metadata_digest != receipt.get("metadata", {}).get("sha256"):
        raise ValueError("Full-source metadata checksum mismatch")
    _check_pins(receipt["input_pins"], chain_data_cache, metadata_sources_dir, metadata_source_lock)
    metadata = json.loads(metadata_path.read_text())
    if receipt.get("source_scope") != metadata.get("source", {}).get("full_source_scope"):
        raise ValueError("Full-source scope and receipt differ")
    if replay:
        expected, pins = compute_full_source_metadata(
            chain_data_cache=chain_data_cache, metadata_sources_dir=metadata_sources_dir,
            metadata_source_lock=metadata_source_lock,
        )
        if metadata != expected or receipt["input_pins"] != pins:
            raise ValueError("Full-source metadata differs from source replay")
    if base._sha256(metadata_path) != metadata_digest or base._sha256(receipt_path) != receipt_digest:
        raise ValueError("Full-source artifact changed during validation")
    return metadata


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chain-data-cache", type=Path, required=True)
    parser.add_argument("--metadata-sources-dir", type=Path, required=True)
    parser.add_argument("--metadata-source-lock", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True, help="New private output directory.")
    args = parser.parse_args(argv)
    if args.out_dir.exists():
        raise ValueError("Full-source output directory already exists; use a fresh path")
    args.out_dir.parent.mkdir(parents=True, exist_ok=True)
    metadata, pins = compute_full_source_metadata(
        chain_data_cache=args.chain_data_cache, metadata_sources_dir=args.metadata_sources_dir,
        metadata_source_lock=args.metadata_source_lock,
    )
    with tempfile.TemporaryDirectory(prefix=".full-source-", dir=args.out_dir.parent) as temporary:
        staging = Path(temporary)
        metadata_path = staging / "structure_metadata.json"
        with metadata_path.open("w") as handle:
            json.dump(metadata, handle, sort_keys=True, separators=(",", ":"))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        receipt = {
            "schema_version": SCHEMA_VERSION,
            "metadata": {"path": metadata_path.name, "sha256": base._sha256(metadata_path),
                         "bytes": metadata_path.stat().st_size},
            "input_pins": pins,
            "source_scope": metadata["source"]["full_source_scope"],
            "runtime": {"python": platform.python_version()},
        }
        (staging / "receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
        _check_pins(pins, args.chain_data_cache, args.metadata_sources_dir, args.metadata_source_lock)
        # Exclusive creation avoids replacing a concurrently published artifact.
        args.out_dir.mkdir()
        for name in ("structure_metadata.json", "receipt.json"):
            os.link(staging / name, args.out_dir / name)
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
