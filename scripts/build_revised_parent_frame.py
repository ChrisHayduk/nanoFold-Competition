"""Extend the immutable parent barrier to a source-qualified public repair.

The original public cohort remains protected from training. Native work covers
only newly added public queries; the original complete barrier is independently
replayed and retained as evidence. Evaluation selection is a separate receipt.
"""
from __future__ import annotations

import argparse
import importlib
import json
import shutil
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
for entry in (ROOT, ROOT / "scripts"):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

from build_full_parent_frame import json_once, verify_barrier, verify_record, write_once

from nanofold.full_parent_leakage import (
    file_record,
    parse_clusters,
    read_groups,
    read_manifest,
    read_population,
    sha256,
)
from nanofold.full_parent_leakage import (
    validate_parent_frame as validate_baseline,
)
from nanofold.mmseqs import iter_batched_mmseqs_hits, run_batched_mmseqs
from nanofold.revised_parent_leakage import POLICY, derive_revised_frame

CODE_FILES = ("scripts/build_revised_parent_frame.py", "nanofold/revised_parent_leakage.py",
              "nanofold/heldout_repair_selection.py", "scripts/build_full_parent_frame.py",
              "nanofold/full_parent_leakage.py", "nanofold/mmseqs.py")
MINIMUM_TRAINING_SEQUENCES = 10000


def validate_selection_cohorts(baseline_splits: dict[str, Any], selection: dict[str, Any]) -> list[str]:
    """Compare membership while preserving original manifest order separately."""
    historical, hidden = baseline_splits["val"], baseline_splits["hidden_val"]
    added = sorted(set(selection["active_public_ids"]) - set(historical))
    if (sorted(selection["historical_public_ids"]) != sorted(historical)
            or sorted(selection["hidden_ids"]) != sorted(hidden)
            or not added or set(selection["added_public_ids"]) != set(added)
            or set(selection["qualified_added_ids"]) != set(added)):
        raise ValueError("Repair selection does not preserve and qualify the declared evaluation cohorts")
    return added


def load_context(base_path: Path, selection_path: Path) -> dict[str, Any]:
    base_pin = file_record(base_path)
    baseline = validate_baseline(base_path)
    receipt = json.loads(base_path.read_text())
    selection = importlib.import_module("nanofold.heldout_repair_selection").validate_selection_receipt(
        selection_path, verify_sources=True)
    if selection["source_bindings"]["baseline_frame_receipt"] != base_pin:
        raise ValueError("Evaluation repair was selected against another baseline frame")
    active = selection["active_public_ids"]
    added = validate_selection_cohorts(baseline["frozen_splits"], selection)
    rows = read_population(verify_record(baseline["source_bindings"]["population_jsonl"]))
    by_id = {row["chain_id"]: row for row in rows}
    query_text = "".join(f">{chain}\n{by_id[chain]['sequence']}\n" for chain in added)
    source = base_path.resolve().parent / "parents.fasta"
    cluster_receipt = json.loads(verify_record(receipt["cluster_receipt"]).read_text())
    clusters = parse_clusters(verify_record(cluster_receipt["output"]), {row["sequence"] for row in rows})
    return {"baseline": baseline, "base_receipt": receipt, "selection": selection, "rows": rows, "clusters": clusters,
            "active": active, "added": added, "query_text": query_text, "source": source,
            "groups": read_groups(verify_record(baseline["source_bindings"]["inherited_groups_tsv"]))}


def reconstruct(context: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    frame = derive_revised_frame(
        context["rows"], context["groups"], context["baseline"]["frozen_splits"], context["active"], context["clusters"],
        iter_batched_mmseqs_hits(context["base_receipt"]["barrier"], min_seq_id=0.3, coverage=0.8, coverage_mode=2),
        iter_batched_mmseqs_hits(result, min_seq_id=0.3, coverage=0.8, coverage_mode=2))
    if not set(frame["eligible_training_ids"]) <= set(context["baseline"]["eligible_training_ids"]):
        raise ValueError("Additive barrier admitted a previously excluded training parent")
    original = context["baseline"]["diagnostics"]
    diagnostics = frame["diagnostics"]
    diagnostics.update(baseline_training_chains=original["retained_training_chains"],
                       baseline_training_sequences=original["retained_training_sequences"],
                       additional_excluded_chains=original["retained_training_chains"] - diagnostics["retained_training_chains"],
                       additional_excluded_sequences=original["retained_training_sequences"] - diagnostics["retained_training_sequences"])
    if diagnostics["retained_training_sequences"] < MINIMUM_TRAINING_SEQUENCES:
        raise ValueError(f"Additive quarantine leaves only {diagnostics['retained_training_sequences']} unique training parents; "
                         f"at least {MINIMUM_TRAINING_SEQUENCES} are required. Preserve this failed attempt before revising the protocol.")
    return frame


def verify_execution_binding(plan: dict[str, Any], native: dict[str, Any]) -> None:
    """Bind the declared execution to the actual native search contract."""
    verify_record(plan["binary"])
    if {key: native["binary"][key] for key in ("path", "sha256")} != plan["binary"]:
        raise ValueError("Additive search used a different declared native binary")
    for name in ("source_batch_size", "query_batch_size", "reserve_disk_bytes", "max_rss_bytes"):
        if native[name] != plan[name]:
            raise ValueError("Additive search differs from declared execution bounds")
    flags = dict(zip(native["flags"][::2], native["flags"][1::2], strict=True))
    if (native["max_sequence_length"] != plan["maximum_sequence_length"]
            or flags["--threads"] != str(plan["threads"])
            or any(flags[key] != str(plan["source_batch_size"]) for key in ("--max-seqs", "--max-accept", "--max-rejected"))):
        raise ValueError("Additive native flags differ from the declared execution plan")


def build(args: argparse.Namespace) -> Path:
    import fcntl

    out = args.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=True, mode=0o700)
    with (out / ".lock").open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        inputs = {"baseline_frame_receipt": file_record(args.baseline_frame_receipt),
                  "selection_receipt": file_record(args.selection_receipt)}
        context = load_context(args.baseline_frame_receipt, args.selection_receipt)
        code = {name: file_record(ROOT / name) for name in CODE_FILES}
        binary_name = shutil.which(args.mmseqs_bin)
        if binary_name is None:
            raise ValueError("MMseqs executable unavailable")
        binary = Path(binary_name).resolve()
        query, active = out / "added-public.fasta", out / "active-public.txt"
        write_once(query, context["query_text"])
        write_once(active, "".join(chain + "\n" for chain in context["active"]))
        maximum = max(row["length"] for row in context["rows"])
        plan = {"schema_version": 1, "policy": POLICY, "inputs": inputs, "code": code,
                "query": file_record(query), "source": file_record(context["source"]),
                "active_public": file_record(active), "binary": file_record(binary),
                "minimum_training_sequences": MINIMUM_TRAINING_SEQUENCES,
                "threads": args.threads, "source_batch_size": args.source_batch_size,
                "query_batch_size": args.query_batch_size, "maximum_sequence_length": maximum,
                "reserve_disk_bytes": args.reserve_disk_bytes, "max_rss_bytes": args.max_rss_bytes}
        plan_path = out / "plan.json"
        json_once(plan_path, plan)
        print(json.dumps({"phase": "additive_public_parent_barrier", "added_queries": len(context["added"]),
                          "source_sequences": len(context["clusters"])}), flush=True)
        result = run_batched_mmseqs(query_fasta=query, source_fasta=context["source"], checkpoint_dir=out / "barrier",
                                   mmseqs_bin=str(binary), min_seq_id=0.3, coverage=0.8, coverage_mode=2,
                                   max_sequence_length=maximum, threads=args.threads, max_seqs=args.source_batch_size,
                                   source_batch_size=args.source_batch_size, query_batch_size=args.query_batch_size,
                                   reserve_disk_bytes=args.reserve_disk_bytes, max_rss_bytes=args.max_rss_bytes,
                                   input_binding_sha256=sha256(plan_path))
        frame = reconstruct(context, result)
        rows_path, train_path = out / "parent-frame.jsonl", out / "eligible-training.txt"
        write_once(rows_path, "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in frame["records"]))
        write_once(train_path, "".join(chain + "\n" for chain in frame["eligible_training_ids"]))
        receipt = {"schema_version": 1, "status": "complete", "policy": POLICY, "inputs": inputs,
                   "code_bindings": code, "plan": file_record(plan_path), "barrier": result,
                   "records": file_record(rows_path), "eligible_training_ids": file_record(train_path),
                   "evaluation_cohorts": frame["evaluation_cohorts"], "diagnostics": frame["diagnostics"],
                   "scope": "Historical and active public cohorts remain protected from training. Active public and fixed hidden pass both query-covered alignment directions. Original public conflicts are retained as historical audit evidence. Training diversity groups are not global split components."}
        pending = out / "receipt.incomplete.json"
        json_once(pending, receipt)
        validate_revised_frame_receipt(pending)
        final = out / "receipt.json"
        json_once(final, receipt)
        print(json.dumps({"phase": "complete", "receipt": file_record(final), "diagnostics": frame["diagnostics"]}), flush=True)
        return final


def validate_revised_frame_receipt(path: Path) -> dict[str, Any]:
    pin = file_record(path)
    receipt = json.loads(path.read_text())
    if receipt.get("schema_version") != 1 or receipt.get("policy") != POLICY or receipt.get("status") != "complete":
        raise ValueError("Incomplete or unknown revised parent frame receipt")
    plan_path = verify_record(receipt["plan"])
    plan = json.loads(plan_path.read_text())
    if (plan.get("policy") != POLICY or plan.get("schema_version") != 1
            or plan.get("minimum_training_sequences") != MINIMUM_TRAINING_SEQUENCES
            or plan["inputs"] != receipt["inputs"] or plan["code"] != receipt["code_bindings"]
            or set(plan["code"]) != set(CODE_FILES)):
        raise ValueError("Revised frame differs from its frozen input and code plan")
    for name, binding in plan["code"].items():
        verify_record(binding)
        if sha256(ROOT / name) != binding["sha256"]:
            raise ValueError("Current revised-frame implementation differs from its producer")
    context = load_context(verify_record(receipt["inputs"]["baseline_frame_receipt"]),
                           verify_record(receipt["inputs"]["selection_receipt"]))
    query, source = verify_record(plan["query"]), verify_record(plan["source"])
    active_path = verify_record(plan["active_public"])
    if (query.read_text() != context["query_text"] or file_record(source) != file_record(context["source"])
            or read_manifest(active_path) != context["active"]
            or plan["maximum_sequence_length"] != max(row["length"] for row in context["rows"])):
        raise ValueError("Additive query/source or active cohort differs from validated selection")
    verify_barrier(receipt["barrier"], query, source, sha256(plan_path))
    native_plan = json.loads(verify_record({"path": receipt["barrier"]["plan_path"],
                                           "sha256": receipt["barrier"]["plan_sha256"]}).read_text())
    verify_execution_binding(plan, native_plan["binding"])
    frame = reconstruct(context, receipt["barrier"])
    observed = [json.loads(line) for line in verify_record(receipt["records"]).open()]
    if (observed != frame["records"] or read_manifest(verify_record(receipt["eligible_training_ids"])) != frame["eligible_training_ids"]
            or receipt["evaluation_cohorts"] != frame["evaluation_cohorts"] or receipt["diagnostics"] != frame["diagnostics"]):
        raise ValueError("Revised frame export differs from complete independently replayed exclusions")
    for binding in [*plan["inputs"].values(), *plan["code"].values(), plan["binary"], plan["query"], plan["source"], plan["active_public"], receipt["plan"]]:
        verify_record(binding)
    if file_record(path) != pin:
        raise ValueError("Revised parent frame receipt changed during validation")
    sources = dict(context["baseline"]["source_bindings"])
    sources.update(historical_public=sources["frozen_val"], frozen_val=file_record(active_path),
                   baseline_frame_receipt=receipt["inputs"]["baseline_frame_receipt"], selection_receipt=receipt["inputs"]["selection_receipt"])
    return {**frame, "source_bindings": sources, "receipt": pin,
            "selected_qualifications": context["selection"]["selected_qualifications"]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("baseline-frame-receipt", "selection-receipt", "out-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--mmseqs-bin", required=True)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--source-batch-size", type=int, default=10000)
    parser.add_argument("--query-batch-size", type=int, default=64)
    parser.add_argument("--reserve-disk-bytes", type=int, default=20 * 1024**3)
    parser.add_argument("--max-rss-bytes", type=int, default=48 * 1024**3)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("threads must be positive")
    build(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
