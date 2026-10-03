"""Stage a full-parent sample under the pinned heldout query barrier.

The public protocol contains scientific choices; a private runtime JSON resolves
source paths. Sampling verification is separate from processed-asset acceptance.
Run this on the host where the bound structures and alignments are accessible.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
for directory in (REPO_ROOT, REPO_ROOT / "scripts"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from build_full_source_metadata import validate_full_source_metadata

from nanofold.representative_eligibility import file_record, read_json, require, sha256_file
from nanofold.structural_candidate import join_qualified_parents, metadata_rows, verify_parent_selection
from nanofold.structural_population import read_rows, validate_receipt, write_json
from nanofold.structural_sampling import select_structural_sample

POLICY = "cath-v5-full-parent"
BARRIER_POLICY = "full_parent_query_barrier_protected_public_revision_v1"
IMPLEMENTATIONS = (
    "scripts/stage_structural_candidate.py", "nanofold/structural_candidate.py",
    "nanofold/structural_sampling.py", "nanofold/crop_exposure.py",
    "nanofold/full_parent_leakage.py", "nanofold/revised_parent_leakage.py",
    "scripts/build_full_parent_frame.py", "scripts/build_revised_parent_frame.py",
    "nanofold/heldout_repair_selection.py", "nanofold/mmseqs.py",
    "scripts/build_full_source_metadata.py",
)


def _checked(binding: dict[str, Any]) -> Path:
    path = Path(binding["path"])
    require(sha256_file(path) == binding["sha256"], "Bound full-parent staging artifact changed")
    return path


def _read_manifest(path: Path) -> list[str]:
    ids = [row.strip() for row in path.read_text().splitlines() if row.strip() and not row.startswith("#")]
    require(bool(ids) and len(set(ids)) == len(ids), "Manifest is empty or repeats parent IDs")
    return ids


def _implementations() -> dict[str, dict[str, str]]:
    return {name: file_record(REPO_ROOT / name) for name in IMPLEMENTATIONS}


def _rank(row: dict[str, Any]) -> tuple[float, int, str, int, str]:
    chain = row["chain_id"]
    tie = int.from_bytes(hashlib.sha256(f"0:representative:{chain}".encode()).digest()[:8], "big")
    resolution = row["resolution"]
    return resolution if resolution and resolution < 999.0 else 999., -row["length"], row["release_date"] or "", tie, chain


def read_protocol(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text())
    require(isinstance(value, dict) and value.get("schema_version") == 1
            and value.get("sampling_policy") == POLICY, "Unsupported full-parent sampling protocol")
    require(value.get("coverage") == {"mode": "soft_diversity",
                                      "interpretation": "training_diversity_groups_on_retained_training_subgraph"},
            "Full-parent protocol must explicitly declare soft training diversity")
    require(value["heldout_barrier"] == {
        "policy": BARRIER_POLICY, "min_sequence_identity": 0.30, "min_heldout_query_coverage": 0.80,
        "min_parent_coverage": None, "exclude_heldout_PDB_entries": True,
        "exclude_exact_heldout_parent_sequences": True, "exclude_original_heldout_group_members": True,
        "propagate_through_new_full_parent_components": False,
    }, "Full-parent protocol changed the approved heldout safeguard")
    require(value["evaluation_cohorts"] == {
        "selection_policy": "source_qualified_public_heldout_repair_v1",
        "historical_public": "preserved_and_protected_from_training",
        "active_public": "deterministic_source_qualified_repair",
        "hidden": "unchanged_original_membership", "preserve_unaffected_public": True,
        "active_public_hidden_barrier": "both_query_directions",
        "target_count_per_cohort": 1000, "metric_based_replacement_selection": False,
    }, "Full-parent protocol changed the evaluation repair contract")
    require(value["parent_eligibility"] == {
        "min_length": 40, "max_length": None, "standard_amino_acids_only": True,
        "max_known_resolution": 3., "allow_unknown_resolution": True,
        "source_qualification": "complete_structure_and_raw_MSA_checks_before_selection",
    }, "Full-parent protocol changed parent eligibility")
    require(value["training_exposure"] == {"parent_sampling": "uniform_selected_parent",
                                           "crop_sampling": "uniform_integer_start", "training_loss_reweighting": False},
            "Full-parent protocol changed uniform training exposure")
    selection = value["selection"]
    require(set(selection) == {"size", "seed", "crop_size", "swap_passes", "max_swap_attempts", "reference_weighting"}
            and all(type(selection[key]) is int and selection[key] >= 0
                    for key in ("size", "seed", "crop_size", "swap_passes", "max_swap_attempts")),
            "Full-parent selection parameters must be explicit nonnegative integers")
    require(selection["crop_size"] == 256 and selection["size"] == 10000,
            "Full-parent protocol must select 10000 parents with 256-residue crops")
    return value


def load_context(config_path: Path, protocol_path: Path, *, verify_sources: bool) -> dict[str, Any]:
    # Import lazily so protocol validation and help do not require graph assets.
    from nanofold.revised_parent_leakage import validate_parent_frame

    runtime, protocol = read_json(config_path), read_protocol(protocol_path)
    require(set(runtime) == {"full_frame_receipt", "population_receipt", "metadata"},
            "Runtime config must supply only the full-frame, population, and metadata source locations")
    meta_paths = runtime["metadata"]
    require(set(meta_paths) == {"path", "receipt", "chain_data_cache", "metadata_sources_dir", "metadata_source_lock"},
            "Runtime metadata paths are incomplete or ambiguous")
    paths = {"runtime_config": config_path, "protocol": protocol_path,
             "full_frame_receipt": Path(runtime["full_frame_receipt"]),
             "population_receipt": Path(runtime["population_receipt"]),
             **{"metadata_" + key: Path(value) for key, value in meta_paths.items() if key != "metadata_sources_dir"}}
    pins = {key: file_record(path) for key, path in paths.items()}
    metadata = validate_full_source_metadata(
        metadata_path=Path(meta_paths["path"]), receipt_path=Path(meta_paths["receipt"]),
        chain_data_cache=Path(meta_paths["chain_data_cache"]), metadata_sources_dir=Path(meta_paths["metadata_sources_dir"]),
        metadata_source_lock=Path(meta_paths["metadata_source_lock"]), replay=True,
    )
    frame = validate_parent_frame(paths["full_frame_receipt"])
    require(frame["policy"] == BARRIER_POLICY, "Full-parent frame has a different leakage scope")
    require(frame["source_bindings"]["chain_data_cache"]["sha256"] == pins["metadata_chain_data_cache"]["sha256"],
            "Full-parent barrier and metadata use different source caches")
    rows = metadata_rows(metadata)
    frame_records = frame["records"]
    by_chain = {row["chain_id"]: row for row in frame_records}
    require(len(by_chain) == len(frame_records), "Full-parent frame repeats chain IDs")
    require(set(by_chain) == {chain for chain, row in rows.items() if row["accepted"]},
            "Full-parent frame does not cover exactly metadata-eligible parents")
    for chain, record in by_chain.items():
        require(len(record["sequence"]) == record["length"] == rows[chain]["length"], "Full-parent length differs from metadata")
        require(hashlib.sha256(record["sequence"].encode()).hexdigest() == rows[chain]["sequence_sha256"],
                "Full-parent sequence differs from metadata")
    admitted = set(frame["eligible_training_ids"])
    require(admitted <= set(by_chain) and len(admitted) == len(frame["eligible_training_ids"]),
            "Full-parent training frame escapes or repeats eligible parents")
    copy_keys = ("chain_id", "sequence", "length", "resolution", "release_date", "group_id", "cluster40_id")
    expected = {chain: {key: by_chain[chain][key] for key in copy_keys} for chain in admitted}
    for row in expected.values():
        if row["resolution"] is None:
            row["resolution"] = 999.0
    receipt = validate_receipt(paths["population_receipt"], verify_sources=verify_sources)
    plan = read_json(Path(receipt["plan"]["path"]))
    require(plan.get("seed") == 0, "Source copies must retain the frozen quality-ranking seed")
    require(any(binding.get("path") == pins["full_frame_receipt"]["path"]
                and binding.get("sha256") == pins["full_frame_receipt"]["sha256"] for binding in plan["inputs"]),
            "Qualification plan does not bind this full-parent frame")
    records = join_qualified_parents(expected_copies=expected, ranks={chain: _rank(row) for chain, row in expected.items()},
                                     plans=read_rows(Path(receipt["sequence_plan"]["path"])),
                                     qualified=read_rows(Path(receipt["qualified_chains"]["path"])), metadata=rows)
    require(protocol["selection"]["reference_weighting"] == "uniform_distinct_sequence",
            "This staged protocol requires the independently qualified unique-sequence reference")
    frozen = frame["frozen_splits"]
    require(set(frozen) == {"val", "hidden_val"}, "Full-parent frame lacks both active heldouts")
    cohorts = frame["evaluation_cohorts"]
    require(set(cohorts) == {"historical_public", "active_public", "hidden"}
            and cohorts["active_public"] == frozen["val"] and cohorts["hidden"] == frozen["hidden_val"],
            "Evaluation cohorts differ from active staged splits")
    require(all(len(ids) == protocol["evaluation_cohorts"]["target_count_per_cohort"] for ids in cohorts.values()),
            "Evaluation cohort count differs from the approved repair protocol")
    require(frame["diagnostics"].get("active_public_hidden_symmetric_boundary_passed") is True,
            "Repaired active public cohort lacks its symmetric native boundary proof")
    for role in ("baseline_frame_receipt", "selection_receipt", "historical_public"):
        pins[role] = file_record(_checked(frame["source_bindings"][role]))
    require(read_json(Path(pins["selection_receipt"]["path"])).get("policy")
            == protocol["evaluation_cohorts"]["selection_policy"],
            "Public repair selection policy differs from the sampling protocol")
    require(_read_manifest(Path(pins["historical_public"]["path"])) == cohorts["historical_public"],
            "Historical public cohort differs from its pinned original manifest")
    frozen_paths = {}
    for split in frozen:
        source = frame["source_bindings"]["frozen_val" if split == "val" else "frozen_hidden"]
        path = _checked(source)
        require(_read_manifest(path) == frozen[split], "Full-parent frame differs from pinned heldout manifest")
        frozen_paths[split] = path
        pins["frozen_" + split] = file_record(path)
    for pin in pins.values():
        _checked(pin)
    return {"pins": pins, "protocol": protocol, "records": records, "frame": frame,
            "metadata": metadata, "frozen_paths": frozen_paths, "population": receipt}


def _selection(context: dict[str, Any]) -> tuple[list[str], dict[str, Any]]:
    selection = context["protocol"]["selection"]
    return select_structural_sample(context["records"], size=selection["size"], seed=selection["seed"],
                                    swap_passes=selection["swap_passes"], max_swap_attempts=selection["max_swap_attempts"],
                                    crop_size=selection["crop_size"], group_policy="soft_diversity",
                                    reference_weighting=selection["reference_weighting"])


def _verify_selection(context: dict[str, Any], train: list[str]) -> dict[str, Any]:
    return verify_parent_selection(train, qualified=context["records"], frozen_splits=context["frame"]["frozen_splits"],
                                   evaluation_cohorts=context["frame"]["evaluation_cohorts"],
                                   all_sequences={row["chain_id"]: row["sequence"] for row in context["frame"]["records"]},
                                   size=context["protocol"]["selection"]["size"], required_groups=set(),
                                   coverage_policy="soft_diversity")


def _output_paths(stage: Path) -> dict[str, Path]:
    return {"train": stage / "public/train.txt", "val": stage / "public/val.txt",
               "all": stage / "public/all.txt", "hidden_val": stage / "hidden/hidden_val.txt",
               "historical_public": stage / "evaluation/historical_public.txt",
               "selection_report": stage / "selection-report.json",
               "public_source_lock": stage / "public-source.lock.json", "private_source_lock": stage / "private-source.lock.json"}


def _public_lock(context: dict[str, Any], implementation: dict[str, Any], outputs: dict[str, Path],
                 verification: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": 1, "sampling_policy": POLICY, "protocol": context["pins"]["protocol"],
        "selection_parameters": context["protocol"]["selection"],
        "coverage": context["protocol"]["coverage"], "training_exposure": context["protocol"]["training_exposure"],
        "chain_data_cache_path": context["pins"]["metadata_chain_data_cache"]["path"],
        "chain_data_cache_sha256": context["pins"]["metadata_chain_data_cache"]["sha256"],
        "structure_metadata": context["pins"]["metadata_path"],
        "full_frame_receipt": context["pins"]["full_frame_receipt"],
        "population_receipt": context["pins"]["population_receipt"],
        "evaluation_repair": {"baseline_frame_receipt": context["pins"]["baseline_frame_receipt"],
                              "selection_receipt": context["pins"]["selection_receipt"],
                              "protocol": context["protocol"]["evaluation_cohorts"]},
        "implementation": implementation, "verification": verification,
        "outputs": {key + "_manifest": str(outputs[key]) for key in ("train", "val", "all")}
                   | {key + "_manifest_sha256": sha256_file(outputs[key]) for key in ("train", "val", "all")}
                   | {"train_count": verification["training_parent_count"], "val_count": len(context["frame"]["frozen_splits"]["val"]),
                      "unique_count": verification["training_parent_count"] + len(context["frame"]["frozen_splits"]["val"])},
    }


def _private_lock(context: dict[str, Any], outputs: dict[str, Path]) -> dict[str, Any]:
    return {
        "schema_version": 1, "sampling_policy": POLICY, "source_lock_sha256": sha256_file(outputs["public_source_lock"]),
        "hidden_val_manifest": str(outputs["hidden_val"]), "hidden_val_manifest_sha256": sha256_file(outputs["hidden_val"]),
        "hidden_val_count": len(context["frame"]["frozen_splits"]["hidden_val"]),
        "evaluation_cohorts": {role: file_record(outputs[key]) | {"count": len(context["frame"]["evaluation_cohorts"][role])}
                               for role, key in (("historical_public", "historical_public"),
                                                 ("active_public", "val"), ("hidden", "hidden_val"))},
    }


def stage_candidate(config_path: Path, protocol_path: Path, stage: Path) -> Path:
    os.umask(0o077)
    stage = stage.resolve()
    require(not stage.exists(), "Full-parent candidate staging directory must be fresh")
    require(not stage.is_relative_to(REPO_ROOT) or stage.is_relative_to(REPO_ROOT / ".nanofold_private"),
            "In-repository full-parent candidates belong in the ignored private root")
    implementation = _implementations()
    context = load_context(config_path.resolve(), protocol_path.resolve(), verify_sources=True)
    train, selection_report = _selection(context)
    verification = _verify_selection(context, train)
    stage.mkdir(parents=True, mode=0o700)
    (stage / "public").mkdir()
    (stage / "hidden").mkdir()
    (stage / "evaluation").mkdir()
    outputs = _output_paths(stage)
    outputs["train"].write_text("".join(chain + "\n" for chain in sorted(train)))
    outputs["all"].write_text("".join(chain + "\n" for chain in sorted(set(train) | set(context["frame"]["frozen_splits"]["val"]))))
    for split in ("val", "hidden_val"):
        outputs[split].write_bytes(context["frozen_paths"][split].read_bytes())
    outputs["historical_public"].write_bytes(Path(context["pins"]["historical_public"]["path"]).read_bytes())
    write_json(outputs["selection_report"], selection_report)
    source_lock = _public_lock(context, implementation, outputs, verification)
    write_json(outputs["public_source_lock"], source_lock)
    write_json(outputs["private_source_lock"], _private_lock(context, outputs))
    for pin in context["pins"].values():
        _checked(pin)
    require(_implementations() == implementation, "Full-parent staging implementation changed during selection")
    status = stage / "candidate_status.json"
    write_json(status, {"schema_version": 1, "sampling_policy": POLICY, "status": "manifest_verified",
                        "processed_assets_audited": False,
                        "scope": "Full-parent sampling and directed heldout barrier; all historical and active public targets protected; processed tensors and MSAs not certified",
                        "immutable_inputs": context["pins"], "implementation": implementation,
                        "outputs": {key: file_record(path) for key, path in outputs.items()}, "verification": verification})
    return status


def verify_stage(stage: Path, *, replay_selection: bool = True, verify_sources: bool = True) -> dict[str, Any]:
    receipt = read_json(stage / "candidate_status.json")
    require(receipt.get("schema_version") == 1 and receipt.get("sampling_policy") == POLICY
            and receipt.get("status") == "manifest_verified" and receipt.get("processed_assets_audited") is False,
            "Full-parent candidate lacks complete manifest verification")
    require(receipt["implementation"] == _implementations(), "Full-parent stage implementation changed")
    for binding in [*receipt["immutable_inputs"].values(), *receipt["outputs"].values()]:
        _checked(binding)
    inputs, outputs = receipt["immutable_inputs"], receipt["outputs"]
    expected_paths = _output_paths(stage.resolve())
    require(outputs == {key: file_record(path) for key, path in expected_paths.items()},
            "Candidate outputs escape their canonical staging paths or omit required artifacts")
    context = load_context(Path(inputs["runtime_config"]["path"]), Path(inputs["protocol"]["path"]), verify_sources=verify_sources)
    require(context["pins"] == inputs, "Full-parent derivation inputs changed")
    manifests = {key: _read_manifest(Path(outputs[key]["path"])) for key in ("train", "val", "all", "hidden_val")}
    for split in ("val", "hidden_val"):
        require(outputs[split]["sha256"] == inputs["frozen_" + split]["sha256"], "Full-parent candidate changed heldout bytes")
    require(outputs["historical_public"]["sha256"] == inputs["historical_public"]["sha256"],
            "Historical public manifest differs from its protected original bytes")
    require(set(manifests["all"]) == set(manifests["train"]) | set(manifests["val"]), "Full-parent public union differs")
    verification = _verify_selection(context, manifests["train"])
    require(verification == receipt["verification"], "Full-parent verification summary differs")
    public, private = read_json(Path(outputs["public_source_lock"]["path"])), read_json(Path(outputs["private_source_lock"]["path"]))
    require(public == _public_lock(context, receipt["implementation"], expected_paths, verification),
            "Full-parent public lock differs from the complete verified contract")
    require(private == _private_lock(context, expected_paths),
            "Full-parent private lock differs from the complete verified contract")
    if replay_selection:
        train, selection = _selection(context)
        require(sorted(train) == manifests["train"], "Full-parent selection did not reproduce")
        require(selection == read_json(Path(outputs["selection_report"]["path"])), "Full-parent selection report did not reproduce")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest="action", required=True)
    stage = actions.add_parser("stage")
    stage.add_argument("--config", type=Path, required=True, help="Private runtime JSON with source locations")
    stage.add_argument("--protocol", type=Path, default=REPO_ROOT / "configs/structural-sampling.yaml")
    stage.add_argument("--stage-dir", type=Path, required=True)
    verify = actions.add_parser("verify")
    verify.add_argument("--stage-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.action == "stage":
        print(stage_candidate(args.config, args.protocol, args.stage_dir))
    else:
        receipt = verify_stage(args.stage_dir.resolve())
        print(json.dumps({"status": receipt["status"], "verification": receipt["verification"]}, sort_keys=True))


if __name__ == "__main__":
    main()
