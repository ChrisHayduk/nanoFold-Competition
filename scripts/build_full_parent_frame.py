"""Certify full-parent training eligibility against frozen heldout targets.

Fresh 40% clusters supply diversity groups only. Exhaustive query-covered local
alignments, exact identity, PDB identity, and inherited heldout ownership supply
training/heldout exclusions. This is not an all-pairs transitive homology graph.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
import uuid
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nanofold.full_parent_leakage import (
    POLICY,
    derive_training_frame,
    file_record,
    parse_clusters,
    read_groups,
    read_manifest,
    read_population,
    sequence_id,
    sha256,
    validate_population_source,
)
from nanofold.mmseqs import (
    _batch_execute,
    _batch_validate_job,
    _batch_verify_partitions,
    iter_batched_mmseqs_hits,
    run_batched_mmseqs,
)

CLUSTER_FLAGS = ["--min-seq-id", "0.4", "-c", "0.8", "--cov-mode", "0", "--alignment-mode", "3",
                 "--seq-id-mode", "0", "--cluster-mode", "0", "--single-step-clustering", "1",
                 "--cluster-reassign", "1", "--max-seqs", "1000", "--mask", "0", "--comp-bias-corr", "0", "-s", "5.7"]


def write_once(path: Path, content: str) -> None:
    if path.exists():
        if path.read_text() != content:
            raise ValueError(f"Immutable frame evidence changed: {path.name}")
    else:
        with path.open("x") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())


def json_once(path: Path, value: Any) -> None:
    write_once(path, json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")


def verify_record(value: dict[str, str]) -> Path:
    path = Path(value["path"])
    if not path.is_file() or sha256(path) != value["sha256"]:
        raise ValueError("A bound parent frame artifact is absent or changed")
    return path


def cluster_command(binary: Path, fasta: Path, out: Path, threads: int, maximum: int) -> list[str]:
    return [str(binary), "easy-cluster", str(fasta), str(out / "cluster"), str(out / "cluster-tmp"),
            *CLUSTER_FLAGS, "--threads", str(threads), "--max-seq-len", str(maximum)]


def verify_cluster(receipt: dict[str, Any], fasta: Path, out: Path, sequences: set[str]) -> dict[str, str]:
    if receipt.get("status") != "complete":
        raise ValueError("Fresh cluster execution is incomplete")
    binary = verify_record(receipt["binary"])
    attempt = Path(receipt["output"]["path"]).parent
    if attempt.parent != out or not attempt.name.startswith("cluster-attempt-"):
        raise ValueError("Fresh cluster outputs must belong to a preserved private attempt")
    command = cluster_command(binary, fasta, attempt, receipt["threads"], max(map(len, sequences)))
    if receipt["command"] != command or receipt["input"] != file_record(fasta):
        raise ValueError("Fresh cluster execution differs from the declared source and grouping contract")
    verify_record(receipt["log"])
    return parse_clusters(verify_record(receipt["output"]), sequences)


def verify_barrier(result: dict[str, Any], query: Path, source: Path, binding: str) -> None:
    """Replay complete checkpoint evidence without permitting any native launch."""
    receipt_path = verify_record({"path": result["receipt_path"], "sha256": result["receipt_sha256"]})
    if json.loads(receipt_path.read_text()) != {k: v for k, v in result.items() if k not in {"receipt_path", "receipt_sha256"}}:
        raise ValueError("Barrier result differs from its native completion receipt")
    plan_path = verify_record({"path": result["plan_path"], "sha256": result["plan_sha256"]})
    plan = json.loads(plan_path.read_text())
    contract = plan["binding"]
    if (contract["query_input"]["sha256"] != sha256(query) or contract["source_input"]["sha256"] != sha256(source)
            or contract["input_binding_sha256"] != binding or contract["coverage_mode"] != 2
            or contract["engine_sha256"] != sha256(ROOT / "nanofold/mmseqs.py")):
        raise ValueError("Barrier search does not certify this complete source/query frame")
    flags = contract["flags"]
    options = dict(zip(flags[::2], flags[1::2], strict=True))
    expected = {"--min-seq-id": "0.3", "-c": "0.8", "--cov-mode": "2", "--alignment-mode": "3",
                "--seq-id-mode": "0", "--search-type": "1", "--dbtype": "1", "--prefilter-mode": "2",
                "-e": "1e308", "-s": "5.7", "--mask": "0", "--comp-bias-corr": "0",
                "--format-output": "query,target,pident,alnlen,qcov,tcov,evalue,bits",
                "--max-seq-len": str(contract["max_sequence_length"])}
    if (len(options) != len(flags) // 2 or set(options) != set(expected) | {"--threads", "--max-seqs", "--max-accept", "--max-rejected"}
            or not math.isfinite(float(options["-e"])) or float(options["-e"]) != 1e308
            or any(options[key] != value for key, value in expected.items() if key != "-e")):
        raise ValueError("Barrier native flags do not enforce the complete directional contract")
    if (int(options["--threads"]) < 1 or any(int(options[key]) < max(p["count"] for p in plan["source_partitions"])
                                           for key in ("--max-seqs", "--max-accept", "--max-rejected"))):
        raise ValueError("Barrier native count limits truncate the parent population")
    root = receipt_path.parent
    binary = verify_record({key: contract["binary"][key] for key in ("path", "sha256")})
    expected_jobs = len(plan["source_partitions"]) * len(plan["query_partitions"])
    if len(result["jobs"]) != expected_jobs:
        raise ValueError("Barrier completion omits a partition pair")
    for role in ("query", "source"):
        _batch_verify_partitions(plan[f"{role}_partitions"], {**contract[f"{role}_input"], "count": plan[f"{role}_count"]},
                                 contract[f"{role}_batch_size"], root, max_sequence_length=contract["max_sequence_length"])
    index = 0
    for si, sources in enumerate(plan["source_partitions"]):
        for qi, queries in enumerate(plan["query_partitions"]):
            job = result["jobs"][index]
            job_root = root / f"job-{si:06d}-{qi:06d}"
            if job["receipt_path"] != str(job_root / "complete.json"):
                raise ValueError("Barrier jobs repeat or omit a partition pair")
            _batch_validate_job(job, queries, sources, plan_sha256=result["plan_sha256"], root=root,
                                job_root=job_root, binary=binary, flags=flags, min_seq_id=0.3, coverage=0.8,
                                coverage_mode=2, max_sequence_length=contract["max_sequence_length"])
            index += 1
    if (result["source_count"] != plan["source_count"] or result["query_count"] != plan["query_count"]
            or result["possible_pair_count"] != plan["source_count"] * plan["query_count"]
            or result["alignments_calculated"] != sum(job["alignment_count"] for job in result["jobs"])
            or result["hit_count"] != sum(job["hit_count"] for job in result["jobs"])):
        raise ValueError("Barrier aggregate accounting differs from its verified partition jobs")


def source_inputs(args: argparse.Namespace) -> dict[str, dict[str, str]]:
    names = ("population_jsonl", "chain_data_cache", "inherited_groups_tsv", "frozen_val", "frozen_hidden",
             "baseline_public_lock", "baseline_private_lock", "baseline_graph_receipt")
    return {name: file_record(getattr(args, name)) for name in names}


def validate_frozen_baseline(sources: dict[str, dict[str, str]], expected_public: str, expected_private: str) -> None:
    paths = {name: verify_record(pin) for name, pin in sources.items()}
    public = json.loads(paths["baseline_public_lock"].read_text())
    private = json.loads(paths["baseline_private_lock"].read_text())
    graph = json.loads(paths["baseline_graph_receipt"].read_text())
    if sha256(paths["baseline_public_lock"]) != expected_public or sha256(paths["baseline_private_lock"]) != expected_private:
        raise ValueError("Original split locks differ from the explicitly pinned baseline")
    if (public["chain_data_cache_sha256"] != sha256(paths["chain_data_cache"])
            or private["source_lock_sha256"] != expected_public
            or public["split_groups"]["receipt_sha256"] != sha256(paths["baseline_graph_receipt"])
            or public["split_groups"]["sha256"] != sha256(paths["inherited_groups_tsv"])
            or graph["split_groups"]["sha256"] != sha256(paths["inherited_groups_tsv"])
            or graph["status"] != "complete"):
        raise ValueError("Original graph provenance differs from the frozen split locks")
    if (sha256(paths["frozen_val"]) != public["outputs"]["val_manifest_sha256"]
            or sha256(paths["frozen_hidden"]) != private["hidden_val_manifest_sha256"]
            or len(read_manifest(paths["frozen_val"])) != 1000
            or len(read_manifest(paths["frozen_hidden"])) != 1000):
        raise ValueError("Frozen heldout manifests differ from the original 1000+1000 baseline")
    groups = read_groups(paths["inherited_groups_tsv"])
    if len(groups) != graph["universe"]["chain_count"] or len(set(groups.values())) != graph["split_groups"]["group_count"]:
        raise ValueError("Inherited original group assignments are incomplete")


def inputs_and_fastas(sources: dict[str, dict[str, str]]) -> tuple[list[dict[str, Any]], dict[str, list[str]], str, str]:
    paths = {name: verify_record(pin) for name, pin in sources.items()}
    rows = read_population(paths["population_jsonl"])
    validate_population_source(rows, paths["chain_data_cache"])
    frozen = {"val": read_manifest(paths["frozen_val"]), "hidden_val": read_manifest(paths["frozen_hidden"])}
    by_id = {row["chain_id"]: row for row in rows}
    source_records = sorted({(sequence_id(row["sequence"]), row["sequence"]) for row in rows})
    query_records = [(chain, by_id[chain]["sequence"]) for chain in sorted(frozen["val"] + frozen["hidden_val"])]
    return rows, frozen, "".join(f">{key}\n{seq}\n" for key, seq in query_records), "".join(f">{key}\n{seq}\n" for key, seq in source_records)


def build(args: argparse.Namespace) -> Path:
    import fcntl

    out = args.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=True, mode=0o700)
    with (out / ".lock").open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        sources = source_inputs(args)
        validate_frozen_baseline(sources, args.expected_baseline_public_sha256, args.expected_baseline_private_sha256)
        code = {name: file_record(ROOT / name) for name in ("scripts/build_full_parent_frame.py", "nanofold/full_parent_leakage.py",
                                                          "nanofold/mmseqs.py", "scripts/build_structure_metadata.py", "scripts/build_manifests.py")}
        rows, frozen, query_text, source_text = inputs_and_fastas(sources)
        maximum = max(row["length"] for row in rows)
        binary_name = shutil.which(args.mmseqs_bin)
        if binary_name is None:
            raise ValueError("MMseqs executable unavailable")
        binary = Path(binary_name).resolve()
        query, source = out / "heldout.fasta", out / "parents.fasta"
        write_once(query, query_text)
        write_once(source, source_text)
        plan = {"policy": POLICY, "sources": sources, "code": code, "binary": file_record(binary),
                "expected_baseline_public_sha256": args.expected_baseline_public_sha256,
                "expected_baseline_private_sha256": args.expected_baseline_private_sha256,
                "threads": args.threads, "maximum_sequence_length": maximum,
                "source_batch_size": args.source_batch_size, "query_batch_size": args.query_batch_size,
                "reserve_disk_bytes": args.reserve_disk_bytes, "max_rss_bytes": args.max_rss_bytes}
        json_once(out / "plan.json", plan)
        binding = sha256(out / "plan.json")
        cluster_path = out / "clusters.receipt.json"
        if not cluster_path.exists():
            attempt = out / ("cluster-attempt-" + uuid.uuid4().hex)
            attempt.mkdir()
            command = cluster_command(binary, source, attempt, args.threads, maximum)
            log = attempt / "clusters.log"
            print(json.dumps({"phase": "fresh_40_percent_clusters", "parent_sequences": len(set(row["sequence"] for row in rows))}), flush=True)
            metrics = _batch_execute(command, log, out, reserve=args.reserve_disk_bytes, max_rss=args.max_rss_bytes)
            cluster_receipt = {"status": "complete", "command": command, "threads": args.threads,
                               "binary": file_record(binary), "input": file_record(source), "log": file_record(log),
                               "output": file_record(attempt / "cluster_cluster.tsv"), "metrics": metrics,
                               "scope": "Heuristic 40 percent identity bilateral 80 percent coverage clustering for diversity; not an exhaustive leakage certificate."}
            verify_cluster(cluster_receipt, source, out, {row["sequence"] for row in rows})
            json_once(cluster_path, cluster_receipt)
        clusters = verify_cluster(json.loads(cluster_path.read_text()), source, out, {row["sequence"] for row in rows})
        print(json.dumps({"phase": "exhaustive_heldout_parent_barrier", "query_count": len(frozen["val"] + frozen["hidden_val"]),
                          "source_count": len(clusters)}), flush=True)
        result = run_batched_mmseqs(query_fasta=query, source_fasta=source, checkpoint_dir=out / "barrier", mmseqs_bin=str(binary),
                                   min_seq_id=0.3, coverage=0.8, threads=args.threads, max_seqs=args.source_batch_size,
                                   source_batch_size=args.source_batch_size, query_batch_size=args.query_batch_size,
                                   reserve_disk_bytes=args.reserve_disk_bytes, max_rss_bytes=args.max_rss_bytes,
                                   input_binding_sha256=binding, coverage_mode=2, max_sequence_length=maximum)
        frame = derive_training_frame(rows, read_groups(Path(sources["inherited_groups_tsv"]["path"])), frozen, clusters,
                                      iter_batched_mmseqs_hits(result, min_seq_id=0.3, coverage=0.8, coverage_mode=2))
        rows_path = out / "parent-frame.jsonl"
        write_once(rows_path, "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in frame["records"]))
        train_path = out / "eligible-training.txt"
        write_once(train_path, "".join(chain + "\n" for chain in frame["eligible_training_ids"]))
        receipt = {"schema_version": 1, "status": "complete", "policy": POLICY, "plan": file_record(out / "plan.json"),
                   "scope": "All heldout-query/full-parent pairs submitted without prefiltering; query-only coverage accepts embedded domains. This certifies the returned local-alignment contract, not absence of every possible alternative alignment or biological homology. Diversity groups are not globally disjoint split components.",
                   "source_bindings": sources, "code_bindings": code, "cluster_receipt": file_record(cluster_path),
                   "barrier": result, "records": file_record(rows_path), "eligible_training_ids": file_record(train_path),
                   "diagnostics": frame["diagnostics"]}
        for pin in [*sources.values(), *code.values()]:
            verify_record(pin)
        pending = out / "receipt.incomplete.json"
        json_once(pending, receipt)
        validate_parent_frame_receipt(pending)
        json_once(out / "receipt.json", receipt)
        print(json.dumps({"phase": "complete", "diagnostics": frame["diagnostics"]}), flush=True)
        return out / "receipt.json"


def _validate_verifier_provenance(
    path: Path, receipt: dict[str, Any], *, allow_unpublished_verifier_drift: bool = False,
) -> None:
    if allow_unpublished_verifier_drift and (path.name != "receipt.incomplete.json" or "verification" in receipt):
        raise ValueError("Verifier drift exception requires the original unpublished receipt")
    verification = receipt.get("verification")
    if verification is not None:
        original = json.loads(verify_record(verification["original_unpublished_receipt"]).read_text())
        if original != {key: value for key, value in receipt.items() if key != "verification"}:
            raise ValueError("Corrected verification changed the original scientific receipt")
        # Retain the historical corrective verifier even when a newer verifier
        # independently replays its certificate and all scientific evidence.
        verify_record(verification["verifier"])
    producer_sha = receipt["code_bindings"]["scripts/build_full_parent_frame.py"]["sha256"]
    if producer_sha != sha256(Path(__file__)) and verification is None and not allow_unpublished_verifier_drift:
        raise ValueError("Changed verifier requires recorded corrective verification provenance")


def validate_parent_frame_receipt(path: Path, *, _allow_unpublished_verifier_drift: bool = False) -> dict[str, Any]:
    digest = sha256(path)
    receipt = json.loads(path.read_text())
    if receipt.get("policy") != POLICY or receipt.get("status") != "complete" or receipt.get("schema_version") != 1:
        raise ValueError("Incomplete or unknown full-parent frame receipt")
    _validate_verifier_provenance(path, receipt, allow_unpublished_verifier_drift=_allow_unpublished_verifier_drift)
    out = path.resolve().parent
    plan_path = verify_record(receipt["plan"])
    plan = json.loads(plan_path.read_text())
    if plan["sources"] != receipt["source_bindings"] or plan["code"] != receipt["code_bindings"] or plan["policy"] != POLICY:
        raise ValueError("Parent frame receipt differs from its immutable input plan")
    validate_frozen_baseline(receipt["source_bindings"], plan["expected_baseline_public_sha256"], plan["expected_baseline_private_sha256"])
    for name, pin in receipt["code_bindings"].items():
        verify_record(pin)
        # The verifier may repair a validation bug without rewriting the
        # original producer or rerunning unchanged scientific alignments.
        if name != "scripts/build_full_parent_frame.py" and sha256(ROOT / name) != pin["sha256"]:
            raise ValueError("Current parent frame implementation differs from the completed producer")
    rows, frozen, query_text, source_text = inputs_and_fastas(receipt["source_bindings"])
    query, source = out / "heldout.fasta", out / "parents.fasta"
    if query.read_text() != query_text or source.read_text() != source_text:
        raise ValueError("Parent barrier FASTAs do not cover the complete input frame")
    clusters = verify_cluster(json.loads(verify_record(receipt["cluster_receipt"]).read_text()), source, out,
                              {row["sequence"] for row in rows})
    verify_barrier(receipt["barrier"], query, source, sha256(plan_path))
    frame = derive_training_frame(rows, read_groups(verify_record(receipt["source_bindings"]["inherited_groups_tsv"])), frozen, clusters,
                                  iter_batched_mmseqs_hits(receipt["barrier"], min_seq_id=0.3, coverage=0.8, coverage_mode=2))
    observed = [json.loads(line) for line in verify_record(receipt["records"]).open()]
    train_ids = read_manifest(verify_record(receipt["eligible_training_ids"]))
    if observed != frame["records"] or train_ids != frame["eligible_training_ids"] or receipt["diagnostics"] != frame["diagnostics"]:
        raise ValueError("Parent frame export differs from independently replayed exclusions and diversity groups")
    for pin in [*receipt["source_bindings"].values(), *receipt["code_bindings"].values(), receipt["records"], receipt["eligible_training_ids"], receipt["plan"]]:
        verify_record(pin)
    if sha256(path) != digest:
        raise ValueError("Parent frame receipt changed during validation")
    return {**frame, "source_bindings": receipt["source_bindings"], "receipt": file_record(path)}


def finalize_existing(pending_path: Path) -> Path:
    """Publish only after replaying preserved native evidence with the fixed verifier."""
    if pending_path.name != "receipt.incomplete.json":
        raise ValueError("Expected the original unpublished completion receipt")
    validate_parent_frame_receipt(pending_path, _allow_unpublished_verifier_drift=True)
    receipt = json.loads(pending_path.read_text())
    receipt["verification"] = {
        "reason": "Corrected equivalent MMseqs E-value literal validation (1e308 versus 1e+308); producer, inputs, alignments, memberships, and parameters unchanged.",
        "original_unpublished_receipt": file_record(pending_path),
        "verifier": file_record(Path(__file__)),
    }
    checked = pending_path.parent / "receipt.corrected-verification.json"
    json_once(checked, receipt)
    validate_parent_frame_receipt(checked)
    final = pending_path.parent / "receipt.json"
    json_once(final, receipt)
    print(json.dumps({"status": "complete", "receipt": file_record(final), "diagnostics": receipt["diagnostics"]}), flush=True)
    return final


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("population-jsonl", "chain-data-cache", "inherited-groups-tsv", "frozen-val", "frozen-hidden", "out-dir",
                 "baseline-public-lock", "baseline-private-lock", "baseline-graph-receipt"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--mmseqs-bin", required=True)
    parser.add_argument("--expected-baseline-public-sha256", required=True)
    parser.add_argument("--expected-baseline-private-sha256", required=True)
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
