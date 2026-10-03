"""Build private structural-sampling assets with resumable, byte-bound producer receipts.

A staged candidate and freshly qualified source indexes are prerequisites. This
coordinator invokes the ordinary production producers; it does not relax their
schemas, projection thresholds, or native exhaustive leakage checks. Outputs and
receipts stay private. Planning is the default; pass --execute to run producers.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
for directory in (REPO_ROOT, REPO_ROOT / "scripts"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from nanofold.chain_paths import chain_id_to_stem, chain_npz_path
from nanofold.data import read_manifest
from nanofold.dataset_integrity import validate_feature_label_pair
from nanofold.msa_filter_policy import PROTECTED_PUBLIC_POLICY_TYPE

COUNTS = {"train": 10000, "val": 1000, "hidden_val": 1000}
TRACKS = ("limited", "research_large", "unlimited")


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def file_pin(path: Path) -> dict[str, Any]:
    before = path.stat()
    result = {"path": str(path.absolute()), "bytes": before.st_size, "sha256": digest(path)}
    after = path.stat()
    if (before.st_ino, before.st_size, before.st_mtime_ns) != (after.st_ino, after.st_size, after.st_mtime_ns):
        raise ValueError(f"Input changed during hashing: {path}")
    return result


def output_pin(path: Path) -> dict[str, Any]:
    if path.is_file():
        return file_pin(path)
    if not path.is_dir():
        raise FileNotFoundError(path)
    files = [{"name": str(p.relative_to(path)), "sha256": digest(p), "bytes": p.stat().st_size}
             for p in sorted(path.rglob("*")) if p.is_file()]
    if not files:
        raise ValueError(f"Expected populated output directory: {path}")
    return {"path": str(path), "files": files}


def atomic_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        try:
            json.dump(data, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def json_digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def require(condition: bool, reason: str) -> None:
    if not condition:
        raise ValueError(reason)


@dataclass(frozen=True)
class Step:
    name: str
    argv: list[str]
    outputs: list[str]
    gate: str = "exists"
    split: str = ""


def build_steps(args: argparse.Namespace) -> list[Step]:
    """Return the complete ordered producer argv, independent of execution state."""
    candidate, out = args.candidate, args.output_dir
    raw = out / "raw"
    manifests = {"train": candidate / "public/train.txt", "val": candidate / "public/val.txt",
                 "all": candidate / "public/all.txt", "hidden_val": candidate / "hidden/hidden_val.txt",
                 "historical_public": candidate / "evaluation/historical_public.txt"}
    filters, audits, leader = out / "msa_filters", out / "audits", out / "leaderboard"
    public_features, public_labels = out / "public_features", out / "public_labels"
    hidden_features, hidden_labels = out / "hidden_features", out / "hidden_labels"
    steps: list[Step] = []

    def add(name: str, script: str, options: list[Any], outputs: list[Path], gate: str = "exists", split: str = "") -> None:
        steps.append(Step(name, [sys.executable, str(REPO_ROOT / "scripts" / script), *map(str, options)],
                          list(map(str, outputs)), gate, split))

    def search_options(name: str) -> list[Any]:
        return ["--mmseqs-bin", args.mmseqs_bin, "--threads", args.threads,
                "--min-seq-id", "0.30", "--coverage", "0.80", "--exhaustive",
                "--coverage-mode", "2", "--max-sequence-length", args.max_sequence_length,
                "--source-batch-size", args.source_batch_size, "--query-batch-size", args.query_batch_size,
                "--max-seqs", args.source_batch_size, "--search-max-rss-gib", args.search_max_rss_gib,
                "--search-disk-reserve-gib", args.search_disk_reserve_gib,
                "--checkpoint-dir", out / "checkpoints" / name, "--tmp-dir", out / "tmp" / name]

    identity = ["--chain-data-cache", args.chain_data_cache,
                "--manifest", manifests["all"], "--manifest", manifests["hidden_val"],
                "--expected-counts", 11000, 1000, "--min-len", args.min_len, "--max-len", args.max_len]
    add("raw_identity", "audit_target_identity.py",
        identity + ["--raw-root", raw, "--msa-name", "uniref90_hits.a3m", "--out-dir", audits / "raw_identity"],
        [audits / "raw_identity/summary.json"], "audit")
    for split, excluded in (("train", ("val", "historical_public", "hidden_val")), ("val", ("hidden_val",))):
        options: list[Any] = ["--source-manifest", manifests[split]]
        for other in excluded:
            options.extend(["--heldout-manifest", manifests[other]])
        add(f"{split}_filter", "build_msa_row_filter.py", options + [
            "--chain-data-cache", args.chain_data_cache, "--raw-root", raw, "--msa-name", "uniref90_hits.a3m",
            *search_options(split), "--output", filters / f"{split}.json"],
            [filters / f"{split}.json", out / "checkpoints" / split / "complete.json"])
    policy = filters / "public_directed_policy.json"
    add("directed_policy", "build_msa_filter_policy.py", [
        "--policy-type", PROTECTED_PUBLIC_POLICY_TYPE, "--historical-public-manifest", manifests["historical_public"],
        "--train-manifest", manifests["train"], "--val-manifest", manifests["val"],
        "--hidden-manifest", manifests["hidden_val"], "--train-msa-row-filter", filters / "train.json",
        "--val-msa-row-filter", filters / "val.json", "--output", policy], [policy], "policy")
    preprocess = ["--raw-root", raw, "--mmcif-root", raw / "pdb_data/mmcif_files",
                  "--chain-data-cache", args.chain_data_cache, "--msa-name", "uniref90_hits.a3m",
                  "--max-msa-seqs", 2048, "--disable-templates", "--skip-existing", "--min-projection-seq-identity", "0.90",
                  "--min-projection-coverage", "0.70", "--min-projection-aligned-fraction", "0.70",
                  "--min-projection-valid-ca", 32]
    for split, features, labels, manifest in (("public", public_features, public_labels, manifests["all"]),
                                            ("hidden", hidden_features, hidden_labels, manifests["hidden_val"])):
        options = preprocess + ["--manifest", manifest, "--processed-features-dir", features,
                                "--processed-labels-dir", labels]
        if split == "public":
            options += ["--msa-filter-policy", policy]
        add(f"preprocess_{split}", "preprocess.py", options, [features, labels], "tensors", split)
    add("processed_identity", "audit_target_identity.py", identity + [
        "--features-dir", public_features, "--features-dir", hidden_features,
        "--out-dir", audits / "processed_identity"], [audits / "processed_identity/summary.json"], "audit")
    geometry: list[Any] = ["--chain-data-cache", args.chain_data_cache]
    for split in COUNTS:
        features, labels = (hidden_features, hidden_labels) if split == "hidden_val" else (public_features, public_labels)
        geometry += ["--split", split, manifests[split], features, labels, COUNTS[split]]
    add("label_geometry", "audit_label_geometry.py", geometry + [
        "--min-len", args.min_len, "--max-len", args.max_len,
        "--ca-distance-range", "2.5", "4.5", "--peptide-distance-range", "1.0", "1.8",
        "--out-dir", audits / "label_geometry"], [audits / "label_geometry/summary.json"], "audit")
    for name, source, heldout, heldout_features in (("residual_train_val", "train", "val", public_features),
                                                 ("residual_public_hidden", "all", "hidden_val", hidden_features)):
        additional = (["--extra-heldout-manifest", manifests["historical_public"],
                       "--extra-heldout-manifest", manifests["hidden_val"],
                       "--heldout-chain-data-cache", args.chain_data_cache] if source == "train" else [])
        add(name, "audit_msa_split.py", [
            "--train-manifest", manifests[source], "--val-manifest", manifests[heldout],
            "--features-dir", public_features, "--val-features-dir", heldout_features,
            "--feature-exclusion-list", REPO_ROOT / "data/manifests/openfold_required_feature_exclusions.txt",
            "--out-dir", audits / name, "--strict", "--run-mmseqs",
            "--skip-plot", *search_options(name), *additional], [audits / name / "msa_split_audit_summary.json"], "residual")
    for split in ("train", "val"):
        add(f"{split}_source_summary", "write_msa_row_filter_source_lock.py", [
            "--msa-row-filter", filters / f"{split}.json", "--msa-filter-policy", policy, "--split", split,
            "--processed-filter-audit", public_features / f"msa_row_filter_audit.{split}.json",
            "--output", leader / f"{split}_msa_filter_source.lock.json"],
            [leader / f"{split}_msa_filter_source.lock.json"], "source_summary", split)
    for track in TRACKS:
        for split, features, labels in (("public", public_features, public_labels), ("hidden", hidden_features, hidden_labels)):
            fingerprint = leader / f"{track}_{split}_fingerprint.json"
            options = ["--processed-features-dir", features, "--processed-labels-dir", labels,
                       "--track", track, "--source-lock", candidate / "public-source.lock.json", "--output", fingerprint]
            options += (["--train-manifest", manifests["train"], "--val-manifest", manifests["val"]] if split == "public"
                        else ["--manifest", f"hidden_val={manifests['hidden_val']}"])
            add(f"fingerprint_{track}_{split}", "build_fingerprint.py", options, [fingerprint], "fingerprint", split)
            if split == "hidden":
                lock = leader / f"private_{track}_hidden_assets.lock.json"
                add(f"pin_{track}_hidden", "pin_hidden_assets.py", [
                    "--hidden-manifest", manifests["hidden_val"], "--hidden-features-dir", hidden_features,
                    "--hidden-labels-dir", hidden_labels, "--hidden-fingerprint", fingerprint, "--track-id", track,
                    "--hidden-chain-count", 1000, "--lock-file", lock], [lock], "hidden_lock")
    return steps


def source_rows(paths: list[Path], selected: set[str]) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for path in paths:
        for line in path.open():
            row = json.loads(line)
            chain = row["chain_id"]
            if chain not in selected:
                continue
            require(chain not in rows, "Selected target occurs twice in source indexes")
            qualification = row["qualification"]
            require(qualification["status"] == "passed" and qualification["raw_status"] == "passed"
                    and qualification["structure_status"] == "passed", "Selected source was not fully qualified")
            require(qualification["chain_id"] == chain, "Source qualification belongs to another target")
            require(hashlib.sha256(row["sequence"].encode()).hexdigest() == qualification["sequence_sha256"],
                    "Qualified target sequence differs from source index")
            require(qualification["sequence_sha256"] == qualification["raw"]["query_sha256"], "Raw query identity differs")
            audit = qualification.get("raw_full_audit", {})
            require(audit.get("sha256") == qualification["raw"]["sha256"]
                    and audit.get("bytes") == qualification["raw"]["bytes"]
                    and audit.get("path") == qualification["raw"]["path"]
                    and audit.get("query_length") == len(row["sequence"])
                    and type(audit.get("rows")) is int and audit["rows"] > 0
                    and type(audit.get("aligned_width")) is int and audit["aligned_width"] >= len(row["sequence"]),
                    "Selected source lacks a complete bound raw MSA audit")
            rows[chain] = row
    require(set(rows) == selected, "Source indexes do not cover the exact selected manifest union")
    return rows


def materialize_sources(rows: dict[str, dict[str, Any]], raw: Path) -> dict[str, Any]:
    """Link byte-verified source files without copying or modifying their caches."""
    expected: dict[Path, dict[str, Any]] = {}
    for chain, row in rows.items():
        qualification = row["qualification"]
        pdb = chain.split("_", 1)[0]
        for key, destination in (("raw", raw / "roda_pdb" / chain_id_to_stem(chain) / "uniref90_hits.a3m"),
                                 ("structure", raw / "pdb_data/mmcif_files" / f"{pdb}.cif")):
            source = qualification[key]
            actual = file_pin(Path(source["path"]).resolve(strict=True))
            require(actual["sha256"] == source["sha256"] and actual["bytes"] == source.get("bytes", actual["bytes"]),
                    "Qualified source bytes changed")
            if destination in expected:
                require(expected[destination]["sha256"] == actual["sha256"], "Conflicting structures share a PDB ID")
                continue
            expected[destination] = actual
    for destination, source in expected.items():
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists() or destination.is_symlink():
            require(destination.is_symlink() and destination.resolve(strict=True) == Path(source["path"]),
                    "Materialized destination already exists with another source")
        else:
            destination.symlink_to(source["path"])
    actual_files = {p for p in raw.rglob("*") if p.is_file() or p.is_symlink()}
    require(actual_files == set(expected), "Raw directory contains unbound sources")
    return {"status": "complete", "selected_chain_count": len(rows), "mode": "verified_source_symlinks",
            "files": [{"output": str(path.relative_to(raw)), **pin} for path, pin in sorted(expected.items())]}


def validate_evaluation_sources(frame: dict[str, Any], rows: dict[str, dict[str, Any]],
                                original: dict[str, dict[str, Any]]) -> None:
    """Keep inherited bytes fixed and bind replacements to their selection proof."""
    cohorts = frame["evaluation_cohorts"]
    historical, active, hidden = (set(cohorts[name]) for name in ("historical_public", "active_public", "hidden"))
    added = frame["selected_qualifications"]
    require(set(added) == active - historical, "Replacement qualification proof differs from the active cohort")
    for chain in active | hidden:
        expected = original[chain] if chain in historical | hidden else added[chain]
        actual = rows[chain]["qualification"]
        for source_type in ("structure", "raw"):
            require(actual[source_type]["sha256"] == expected[source_type]["sha256"],
                    "Evaluation source differs from its inherited or replacement acquisition proof")


def native_sequence_bound(rows: dict[str, dict[str, Any]], frame: dict[str, Any]) -> int:
    """Cover selected sources and every protected query, including retired public targets."""
    protected = set().union(*frame["evaluation_cohorts"].values())
    queries = {row["chain_id"]: row["sequence"] for row in frame["records"] if row["chain_id"] in protected}
    require(set(queries) == protected, "Protected query is absent from the validated full-parent frame")
    return max(max(len(row["sequence"]) for row in rows.values()), max(map(len, queries.values())))


def tensor_gate(args: argparse.Namespace, split: str) -> None:
    import numpy as np

    root = args.output_dir
    features, labels = root / f"{split}_features", root / f"{split}_labels"
    manifest = args.candidate / ("public/all.txt" if split == "public" else "hidden/hidden_val.txt")
    chains = read_manifest(manifest)
    expected = {chain_npz_path(features, chain).name for chain in chains}
    require({p.name for p in features.glob("*.npz")} == expected, "Feature coverage differs from manifest")
    require({p.name for p in labels.glob("*.npz")} == expected, "Label coverage differs from manifest")
    require(not list(features.glob("*.error.txt")) and not list(labels.glob("*.error.txt")), "Preprocessing failures remain")
    meta = json.loads((features / "preprocess_meta.json").read_text())
    config = meta["cli_args"]
    # The owning fingerprint builder validates the full per-NPZ configuration binding.
    require(config.get("disable_templates") is True and config.get("max_msa_seqs") == 2048,
            "Preprocessing does not use the fixed template/MSA contract")
    for chain in chains:
        feature, label = chain_npz_path(features, chain), chain_npz_path(labels, chain)
        require(not validate_feature_label_pair(feature, label, expected_chain_id=chain, require_chain_id=True),
                "Invalid paired feature/label identity")
        with np.load(feature, allow_pickle=False) as arrays:
            msa = arrays["msa"]
            distinct_rows = {(row.tobytes(), deletion.tobytes()) for row, deletion in zip(msa, arrays["deletions"], strict=True)}
            require(1 <= len(msa) <= 2048 and len(distinct_rows) == len(msa), "MSA is not capped/deduplicated")
            require(arrays["template_aatype"].shape[0] == 0, "Templates must be disabled")


def semantic_gate(step: Step, args: argparse.Namespace) -> None:
    if step.gate == "tensors":
        tensor_gate(args, step.split)
        return
    if step.gate == "exists":
        return
    result = json.loads(Path(step.outputs[0]).read_text())
    if step.gate == "audit":
        require(result.get("status") == "passed" and result.get("failure_count") == 0
                and result.get("passed_chain_count") == 12000, "Identity/geometry audit did not pass the full corpus")
    elif step.gate == "residual":
        validation = result["validation"]
        require(all(validation.get(key) is True for key in ("passed", "strict",
                    "target_target_homology_checked", "exhaustive_msa_homology_checked")), "Residual boundary audit incomplete")
        require(validation.get("coverage_mode") == 2 and validation.get("max_sequence_length") == args.max_sequence_length,
                "Residual audit does not prove the complete heldout-query coverage contract")
        if step.name == "residual_train_val":
            extra = [args.candidate / "evaluation/historical_public.txt", args.candidate / "hidden/hidden_val.txt"]
            protected = set(read_manifest(args.candidate / "public/val.txt"))
            for manifest in extra:
                protected.update(read_manifest(manifest))
            scope = result.get("heldout_query_scope", {})
            require(scope.get("extra_manifest_sha256") == sorted(digest(path) for path in extra)
                    and scope.get("chain_data_cache_sha256") == digest(args.chain_data_cache)
                    and scope.get("protected_chain_count") == len(protected),
                    "Training residual audit omitted a protected evaluation cohort")
            require(all(result[key].get("heldout_query_scope") == scope
                        for key in ("mmseqs_homology", "mmseqs_target_homology")),
                    "Target and MSA searches did not audit the same protected query union")
    elif step.gate == "policy":
        require(result.get("policy_type") == PROTECTED_PUBLIC_POLICY_TYPE, "Invalid directed filter policy")
    elif step.gate == "fingerprint":
        count = 11000 if step.split == "public" else 1000
        require(result.get("require_labels") is True and result.get("unique_chain_count") == count
                and result.get("present_feature_chain_count") == count and result.get("present_label_chain_count") == count
                and result.get("missing_feature_chain_count") == 0 and result.get("missing_label_chain_count") == 0,
                "Fingerprint is incomplete")
    elif step.gate == "hidden_lock":
        require(result.get("hidden_chain_count") == 1000 and result.get("track") in TRACKS, "Hidden lock incomplete")
    elif step.gate == "source_summary":
        require(result.get("artifact_type") == "msa_row_filter_source_lock" and result.get("public_safe") is True,
                "Filter source summary did not validate")


def process_group_alive(group: int) -> bool:
    try:
        os.killpg(group, 0)
        return True
    except ProcessLookupError:
        return False


def stop_owned_process_group(process: subprocess.Popen[bytes]) -> None:
    """Stop only the process group created by this active coordinator attempt."""
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(process.pid, sig)
        except ProcessLookupError:
            break
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            continue
        for _ in range(20):
            if not process_group_alive(process.pid):
                break
            time.sleep(0.05)
        if not process_group_alive(process.pid):
            break
    if process.poll() is None:
        process.wait(timeout=5)


def execute_step(step: Step, args: argparse.Namespace, plan_sha: str, previous_sha: str) -> str:
    assert_output_containment(args.output_dir)
    require(all(Path(path).absolute().is_relative_to(args.output_dir) for path in step.outputs),
            "Producer outputs must belong to the owned asset root")
    plan_path = args.output_dir / "plan.json"
    if plan_path.exists():
        validate_plan_inputs(json.loads(plan_path.read_text()))
    receipts = args.output_dir / "receipts"
    accepted = receipts / f"{step.name}.json"
    if accepted.exists():
        result = json.loads(accepted.read_text())
        require(result.get("status") == "complete" and result.get("process_stopped") is True and result.get("plan_sha256") == plan_sha
                and result.get("previous_receipt_sha256") == previous_sha and result.get("argv") == step.argv,
                "Existing receipt belongs to another run")
        require(result["outputs"] == [output_pin(Path(path)) for path in step.outputs], "Accepted outputs changed")
        semantic_gate(step, args)
        return digest(accepted)
    attempts = sorted(receipts.glob(f"{step.name}.attempt-*.json")) if receipts.exists() else []
    if attempts:
        prior = json.loads(attempts[-1].read_text())
        require(prior.get("status") in ("complete", "failed") and prior.get("process_stopped") is True,
                "Prior producer state is unresolved; reconcile its recorded process before resuming")
    if any(Path(path).exists() for path in step.outputs):
        require(bool(attempts), "Unowned producer outputs already exist")
        previous = json.loads(attempts[-1].read_text())
        require(previous.get("plan_sha256") == plan_sha and previous.get("argv") == step.argv
                and previous.get("previous_receipt_sha256") == previous_sha,
                "Existing producer outputs belong to another attempt")
        if previous["status"] == "complete":
            require(previous["outputs"] == [output_pin(Path(path)) for path in step.outputs], "Completed attempt outputs changed")
            semantic_gate(step, args)
            atomic_json(accepted, previous)
            return digest(accepted)
    receipts.mkdir(parents=True, exist_ok=True)
    attempt = 1
    while (receipts / f"{step.name}.attempt-{attempt:03d}.json").exists():
        attempt += 1
    prefix = receipts / f"{step.name}.attempt-{attempt:03d}"
    result = {"status": "running", "argv": step.argv, "cwd": str(REPO_ROOT), "plan_sha256": plan_sha,
              "previous_receipt_sha256": previous_sha, "started_at": datetime.now(timezone.utc).isoformat(),
              "process_stopped": False}
    attempt_path = Path(f"{prefix}.json")
    atomic_json(attempt_path, result)
    # Native search and preprocessing own validated resumption; no historical
    # producer output is imported merely because it exists.
    process: subprocess.Popen[bytes] | None = None
    try:
        with Path(f"{prefix}.stdout.log").open("xb") as stdout, Path(f"{prefix}.stderr.log").open("xb") as stderr:
            process = subprocess.Popen(step.argv, cwd=REPO_ROOT, stdout=stdout, stderr=stderr, start_new_session=True)
            result["process_group_id"] = process.pid
            atomic_json(attempt_path, result)
            result["exit_code"] = process.wait()
        require(not process_group_alive(process.pid), "Producer exited with active descendants")
        result["process_stopped"] = True
        require(process.returncode == 0, f"Producer failed: {step.name}; see {prefix}.stderr.log")
        assert_output_containment(args.output_dir)
        if plan_path.exists():
            validate_plan_inputs(json.loads(plan_path.read_text()))
        semantic_gate(step, args)
        result["outputs"] = [output_pin(Path(path)) for path in step.outputs]
        result["status"] = "complete"
    except BaseException as exc:
        if process is not None:
            stop_owned_process_group(process)
        # Native producers may create their own sessions. Disappearance of
        # this group cannot certify detached descendants after any failure.
        # Keep the attempt unresolved until an independent process audit.
        result["process_stopped"] = process is None
        result.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        result["finished_at"] = datetime.now(timezone.utc).isoformat()
        result["logs"] = [file_pin(path) for path in (Path(f"{prefix}.stdout.log"), Path(f"{prefix}.stderr.log")) if path.exists()]
        atomic_json(attempt_path, result)
    atomic_json(accepted, result)
    return digest(accepted)


def assert_output_containment(root: Path) -> None:
    """Reject writable aliases, including aliases introduced after admission."""
    require(root.absolute() == root.resolve(), "Output root or an ancestor is a symlink")
    if not root.exists():
        return
    require(root.is_dir(), "Output root must be a directory")
    for path in root.rglob("*"):
        relative = path.relative_to(root)
        if relative.parts[0] == "raw":
            # The materializer owns and independently replays this read-only
            # source-link inventory. Directories may never be redirected.
            require(not (path.is_symlink() and path.is_dir()), "Raw source directories may not be symlinked")
            continue
        if path.is_symlink():
            require(relative.parts[0] in ("checkpoints", "tmp") and path.resolve().is_relative_to(root),
                    "Writable output path is a symlink or escapes its owned root")
        elif path.is_file() and relative.parts[0] not in ("checkpoints", "tmp"):
            require(path.stat().st_nlink == 1, "Writable output is a hardlink")


def validate_plan_inputs(plan: dict[str, Any]) -> None:
    for pin in plan["inputs"]:
        require(file_pin(Path(pin["path"])) == pin, "A bound producer input changed")
    for relative, expected in plan["code"].items():
        require(digest(REPO_ROOT / relative) == expected, "Producer implementation changed")


def admit_output_root(root: Path) -> None:
    assert_output_containment(root)
    if root.exists() and not (root / "plan.json").exists():
        require(not any(path.name != ".build.lock" for path in root.iterdir()),
                "An asset root without its matching plan must be fresh")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("candidate", "qualified-chains", "heldout-chains", "heldout-source-receipt", "chain-data-cache", "baseline-source-receipt", "mmseqs-bin", "output-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=12)
    parser.add_argument("--source-batch-size", type=int, default=250000)
    parser.add_argument("--query-batch-size", type=int, default=128)
    parser.add_argument("--search-max-rss-gib", type=int, default=32)
    parser.add_argument("--search-disk-reserve-gib", type=int, default=64)
    parser.add_argument("--execute", action="store_true", help="Execute the reviewed plan and resume verified completed steps.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    os.umask(0o077)
    args = parse_args(argv)
    args.output_dir = args.output_dir.absolute()
    admit_output_root(args.output_dir)
    for key, value in vars(args).items():
        if isinstance(value, Path) and key != "output_dir":
            setattr(args, key, value.resolve())
    require(args.output_dir.is_relative_to(REPO_ROOT / ".nanofold_private"), "Assets and hidden evidence must remain private")
    require(not args.candidate.is_relative_to(args.output_dir) and not args.output_dir.is_relative_to(args.candidate),
            "Output and candidate directories must be separate")
    require(all(getattr(args, name) > 0 for name in ("threads", "source_batch_size", "query_batch_size",
                 "search_max_rss_gib", "search_disk_reserve_gib")), "Resource settings must be positive")
    from stage_structural_candidate import verify_stage

    stage = verify_stage(args.candidate)
    require(stage["sampling_policy"] == "cath-v5-full-parent", "Assets require the staged full-parent protocol")
    bound = stage["immutable_inputs"]
    from nanofold.heldout_sources import validate_heldout_sources

    heldout_receipt = validate_heldout_sources(args.heldout_source_receipt, verify_sources=True)
    require(heldout_receipt["status"] == "complete"
            and heldout_receipt["heldout_chains"]["path"] == str(args.heldout_chains)
            and heldout_receipt["heldout_chains"]["sha256"] == digest(args.heldout_chains),
            "Heldout source index differs from its completed source receipt")
    require(heldout_receipt["full_frame_receipt"]["sha256"] == bound["full_frame_receipt"]["sha256"]
            and heldout_receipt["public_manifest"]["sha256"] == stage["outputs"]["val"]["sha256"]
            and heldout_receipt["hidden_manifest"]["sha256"] == stage["outputs"]["hidden_val"]["sha256"],
            "Heldout source receipt belongs to another evaluation cohort or frame")
    require(str(args.chain_data_cache) == bound["metadata_chain_data_cache"]["path"]
            and digest(args.chain_data_cache) == bound["metadata_chain_data_cache"]["sha256"], "Asset cache differs from staged cache")
    population = json.loads(Path(bound["population_receipt"]["path"]).read_text())
    require(str(args.qualified_chains) == population["qualified_chains"]["path"]
            and digest(args.qualified_chains) == population["qualified_chains"]["sha256"], "Qualified index differs from staged population")
    selected: set[str] = set()
    split_ids: dict[str, list[str]] = {}
    for split, count in COUNTS.items():
        chains = read_manifest(Path(stage["outputs"][split]["path"]))
        require(len(chains) == count and len(set(chains)) == count and not selected.intersection(chains), "Invalid frozen split counts")
        selected.update(chains)
        split_ids[split] = chains
    rows = source_rows([args.qualified_chains], set(split_ids["train"]))
    rows.update(source_rows([args.heldout_chains], set(split_ids["val"]) | set(split_ids["hidden_val"])))
    from nanofold.revised_parent_leakage import POLICY as REVISED_FRAME_POLICY
    from nanofold.revised_parent_leakage import validate_parent_frame

    frame = validate_parent_frame(Path(bound["full_frame_receipt"]["path"]))
    require(frame["policy"] == REVISED_FRAME_POLICY, "Assets require the final additive public-evaluation frame")
    for split in ("val", "hidden_val"):
        require(split_ids[split] == frame["frozen_splits"][split], "Staged evaluation differs from the additive frame")
    require(read_manifest(Path(stage["outputs"]["historical_public"]["path"])) == frame["evaluation_cohorts"]["historical_public"],
            "Historical public evidence differs from the additive frame")
    baseline_pin = frame["source_bindings"]["baseline_public_lock"]
    require(digest(Path(baseline_pin["path"])) == baseline_pin["sha256"], "Baseline source lock changed")
    baseline = json.loads(Path(baseline_pin["path"]).read_text())
    require(digest(args.baseline_source_receipt) == baseline["representative_eligibility"]["receipt_sha256"],
            "Baseline source receipt changed")
    old_rows = {row["chain_id"]: row for unit in json.loads(args.baseline_source_receipt.read_text())["units"]
                for row in unit["outcomes"] if row["status"] == "passed"}
    validate_evaluation_sources(frame, rows, old_rows)
    # Full targets remain intact in NPZs. The staged training protocol samples
    # all integer starts 0..L-min(256,L), including both endpoints, at training.
    args.min_len = 40
    args.max_len = max(len(row["sequence"]) for row in rows.values())
    args.max_sequence_length = native_sequence_bound(rows, frame)
    require(args.max_sequence_length <= 65535, "Selected source or protected query exceeds the native audited sequence-length bound")
    requested_batch_size = args.source_batch_size
    args.source_batch_size = min(args.source_batch_size, (2**32 - 2) // args.max_sequence_length)
    steps = build_steps(args)
    paths = [args.candidate / "candidate_status.json", args.qualified_chains, args.heldout_chains, args.heldout_source_receipt,
             args.chain_data_cache, args.mmseqs_bin, args.baseline_source_receipt,
             REPO_ROOT / "data/manifests/openfold_required_feature_exclusions.txt",
             *[Path(pin["path"]) for pin in bound.values()],
             *[Path(pin["path"]) for pin in stage["outputs"].values()],
             *[Path(pin["path"]) for pin in frame["source_bindings"].values()]]
    code_paths = [p for directory in (REPO_ROOT / "nanofold", REPO_ROOT / "scripts") for p in sorted(directory.rglob("*.py"))]
    source_lock = json.loads((args.candidate / "public-source.lock.json").read_text())
    plan = {"schema_version": 1, "inputs": [file_pin(path) for path in sorted(set(paths))],
            "code": {str(path.relative_to(REPO_ROOT)): digest(path) for path in code_paths},
            "python": sys.executable, "steps": [asdict(step) for step in steps],
            "maximum_parent_length": args.max_len, "maximum_native_sequence_length": args.max_sequence_length,
            "requested_source_batch_size": requested_batch_size,
            "source_batch_size": args.source_batch_size, "training_exposure": source_lock["training_exposure"],
            "crop_size": source_lock["selection_parameters"]["crop_size"], "assets_store_full_parents": True,
            "evaluation_cohorts": {name: {"chain_count": len(chains),
                "membership_sha256": json_digest(sorted(chains))} for name, chains in frame["evaluation_cohorts"].items()}}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / ".build.lock").open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        plan_path = args.output_dir / "plan.json"
        if plan_path.exists():
            require(json.loads(plan_path.read_text()) == plan, "Plan inputs/code changed; use a new private asset root")
        else:
            atomic_json(plan_path, plan)
        validate_plan_inputs(plan)
        if not args.execute:
            print(f"Wrote {len(steps)} producer commands to {plan_path}; no assets generated.")
            return
        materialization = materialize_sources(rows, args.output_dir / "raw")
        materialization_path = args.output_dir / "materialization.json"
        if materialization_path.exists():
            require(json.loads(materialization_path.read_text()) == materialization, "Materialized source bindings changed")
        else:
            atomic_json(materialization_path, materialization)
        for name in ("train", "val", "residual_train_val", "residual_public_hidden"):
            (args.output_dir / "tmp" / name).mkdir(parents=True, exist_ok=True)
        for name in ("msa_filters", "leaderboard", "audits"):
            (args.output_dir / name).mkdir(parents=True, exist_ok=True)
        previous = digest(materialization_path)
        for step in steps:
            print(f"Running/verifying {step.name}", flush=True)
            previous = execute_step(step, args, digest(plan_path), previous)
        require(materialize_sources(rows, args.output_dir / "raw") == materialization, "Source bytes changed during generation")
        validate_plan_inputs(plan)
        require(verify_stage(args.candidate, replay_selection=False) == stage, "Stage changed during asset generation")
        atomic_json(args.output_dir / "completion.json", {
            "schema_version": 1, "status": "complete", "processed_assets_audited": True,
            "plan_sha256": digest(plan_path), "materialization_sha256": digest(materialization_path),
            "last_receipt_sha256": previous, "split_counts": COUNTS,
            "training_exposure": plan["training_exposure"], "crop_size": plan["crop_size"],
            "assets_store_full_parents": True, "evaluation_cohorts": plan["evaluation_cohorts"],
            "geometry_outliers": "Preserved in audits/label_geometry/summary.json; audit passage does not mean zero outliers.",
            "training_improvement_validated": False})
        print(f"Complete verified assets: {args.output_dir}")


if __name__ == "__main__":
    main()
