"""Plan or acquire missing sources for a frozen structural population.

Planning is the default; --execute delegates retrieval to the pinned owners.
Sources remain unqualified until the population qualifier replays them. Each
invocation makes one fresh attempt per unfinished job; reruns may retry up to
--max-attempts, preserving previous directories and failures.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from nanofold.representative_eligibility import (
    _raw_source,
    file_record,
    historical_source_catalog,
    read_json,
    require,
    sha256_file,
)
from nanofold.structural_population import (
    POLICY,
    read_rows,
    validate_file_binding,
    validate_outcome,
    validate_source_catalog,
    write_json,
)
from nanofold.structure_metadata import validate_author_chain_id

BATCH_SIZE = 1024
MIN_FREE_BYTES = 50 * 1024**3
ACQUISITION_FILES = ("scripts/fetch_openproteinset_mmcif.py", "nanofold/structure_metadata.py")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_state(path: Path, value: Any) -> None:
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        try:
            json.dump(value, handle, sort_keys=True, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)


def check_disk(path: Path) -> None:
    while not path.exists():
        path = path.parent
    require(shutil.disk_usage(path).free >= MIN_FREE_BYTES, "Acquisition requires at least 50 GiB free on each output filesystem")


def select_requests(plans: list[dict[str, Any]], requests: list[dict[str, Any]] | None) -> list[str]:
    """Bind each acquisition to the next copy in an immutable sequence rank."""
    indexed = {}
    chains = set()
    for plan in plans:
        sequence, digest, copies = plan["sequence"], plan["sequence_sha256"], plan["ranked_copies"]
        require(isinstance(sequence, str) and bool(sequence)
                and hashlib.sha256(sequence.encode()).hexdigest() == digest, "Sequence plan hash differs from its sequence")
        require(digest not in indexed and bool(copies), "Repeated sequence or empty ranked copy list")
        for candidate in copies:
            chain = candidate["chain_id"]
            validate_author_chain_id(chain)
            require(chain not in chains and candidate["sequence"] == sequence and candidate["length"] == len(sequence)
                    and candidate["group_id"] == plan["group_id"], "Copy differs from its frozen sequence/group ownership")
            chains.add(chain)
        indexed[digest] = plan
    if requests is None:
        return sorted(plan["ranked_copies"][0]["chain_id"] for plan in plans)
    require(bool(requests), "Requested fallback acquisition is empty")
    selected = []
    seen = set()
    for request in requests:
        digest = request.get("sequence_sha256")
        require(digest in indexed and digest not in seen, "Acquisition request is foreign or repeats a sequence")
        seen.add(digest)
        plan = indexed[digest]
        attempts: Any = request.get("attempts")
        require(request.get("status") == "sources_needed" and request.get("group_id") == plan["group_id"]
                and request.get("missing") in {"historical_structure", "raw_query"}
                and isinstance(attempts, list) and len(attempts) < len(plan["ranked_copies"]),
                "Acquisition request does not bind a pending ranked copy")
        if attempts:
            # This validates only the supplied failed prefix, never excludes the
            # full sequence or substitutes for qualification of remaining copies.
            prefix = plan | {"ranked_copies": plan["ranked_copies"][:len(attempts)]}
            validate_outcome(prefix, request | {"selected_chain_id": None}, verify_sources=False)
        chain = plan["ranked_copies"][len(attempts)]["chain_id"]
        require(request.get("pending_chain_id") == chain, "Acquisition request skipped the next ranked copy")
        selected.append(chain)
    return sorted(selected)


def relocation_map(catalog: dict[str, Any]) -> dict[str, str]:
    if catalog.get("source_relocations") is None:
        return {}
    mapping = read_json(Path(catalog["source_relocations"]["path"]))
    return {row["original_path"]: row["path"] for row in mapping["files"]}


def freeze_inputs(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any], list[str]]:
    first_pins = [file_record(args.plan), file_record(args.source_catalog)]
    first_pins.extend(file_record(path) for path in args.retrieval_receipt)
    if args.requests:
        first_pins.append(file_record(args.requests))
    plan = read_json(args.plan)
    require(plan.get("policy") == POLICY, "Unexpected population plan policy")
    sequence_pin = plan["sequence_plan"]
    for pin in [*plan["inputs"], sequence_pin]:
        validate_file_binding(pin)
    plans = read_rows(Path(sequence_pin["path"]))
    require(bool(plans) and len(plans) == plan["sequence_count"], "Sequence plan size differs from frozen plan")
    requests = read_rows(args.requests) if args.requests else None
    selected = select_requests(plans, requests)
    catalog = read_json(args.source_catalog)
    validate_source_catalog(catalog)
    if args.retrieval_receipt:
        receipts = [Path(row["path"]) for row in catalog["historical_sources"]["retrieval_receipts"]]
        sources, members = historical_source_catalog(
            receipts + args.retrieval_receipt, acquisition_code_root=args.acquisition_code_root,
            source_path_map=relocation_map(catalog),
        )
        catalog = catalog | {"historical_sources": sources, "members": members}
    first_pins.extend([sequence_pin, *plan["inputs"], *catalog["inputs"],
                       *catalog["historical_sources"]["retrieval_receipts"],
                       *catalog["historical_sources"]["requested_manifests"]])
    if catalog.get("source_relocations"):
        first_pins.append(catalog["source_relocations"])
    # The supplied frozen checkout must replay the original acquisition hashes.
    historical_source_catalog([Path(row["path"]) for row in catalog["historical_sources"]["retrieval_receipts"]],
                              acquisition_code_root=args.acquisition_code_root, source_path_map=relocation_map(catalog))
    paths = [args.plan, Path(sequence_pin["path"]), args.source_catalog, args.duplicate_chains_file,
             *args.retrieval_receipt,
             *(args.acquisition_code_root / name for name in ACQUISITION_FILES),
             REPO_ROOT / "scripts/prepare_data.py", REPO_ROOT / "nanofold/chain_paths.py",
             REPO_ROOT / "nanofold/representative_eligibility.py", REPO_ROOT / "nanofold/structural_population.py",
             REPO_ROOT / "nanofold/structure_metadata.py", REPO_ROOT / "nanofold/__init__.py",
             args.acquisition_code_root / "nanofold/__init__.py", Path(__file__), Path(sys.executable)]
    if args.requests:
        paths.append(args.requests)
    if catalog.get("source_relocations"):
        paths.append(Path(catalog["source_relocations"]["path"]))
    paths.extend(Path(row["path"]) for row in [*catalog["inputs"], *plan["inputs"],
                                             *catalog["historical_sources"]["retrieval_receipts"],
                                             *catalog["historical_sources"]["requested_manifests"]])
    known = {str(Path(pin["path"]).resolve()): pin for pin in first_pins}
    pins = []
    for path in sorted(set(path.resolve() for path in paths)):
        pin = known[str(path)] if str(path) in known else file_record(path)
        validate_file_binding(pin)
        pins.append(pin)
    binding = {"schema_version": 1, "policy": POLICY, "files": pins,
               "selected_chain_ids": selected, "raw_roots": sorted(str(path.resolve()) for path in args.raw_root),
               "raw_output_root": str(args.raw_output_root.resolve()), "python": sys.version,
               "acquisition_code_root": str(args.acquisition_code_root.resolve()),
               "structure_batch_size": BATCH_SIZE, "structure_workers": 4, "raw_download_workers": 8,
               "max_attempts": args.max_attempts, "minimum_free_bytes": MIN_FREE_BYTES}
    return binding, catalog, selected


def check_inputs(binding: dict[str, Any]) -> None:
    for pin in binding["files"]:
        validate_file_binding(pin)


def write_manifest(path: Path, chains: list[str]) -> dict[str, str]:
    with path.open("x") as handle:
        handle.write("\n".join(chains) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    return file_record(path)


def make_jobs(output: Path, selected: list[str], catalog: dict[str, Any], raw_roots: list[Path]) -> list[dict[str, Any]]:
    missing_pdbs: dict[str, str] = {}
    raw_missing = []
    for chain in selected:
        pdb = chain.split("_", 1)[0]
        if pdb not in catalog["members"]:
            missing_pdbs.setdefault(pdb, chain)
        else:
            require(Path(catalog["members"][pdb]["path"]).is_file(), "Catalog member is missing; reconcile original source acquisition")
        try:
            existing = _raw_source(raw_roots, chain)
            require(existing.stat().st_size > 0, "Existing raw source is empty; reconcile incomplete acquisition")
        except ValueError as error:
            if str(error) != "Required raw query source is missing; acquisition is incomplete":
                raise
            raw_missing.append(chain)
    manifests = output / "manifests"
    manifests.mkdir()
    jobs = []
    missing = [missing_pdbs[pdb] for pdb in sorted(missing_pdbs)]
    for start in range(0, len(missing), BATCH_SIZE):
        chains = missing[start:start + BATCH_SIZE]
        name = f"structures-{start // BATCH_SIZE:05d}"
        jobs.append({"name": name, "kind": "structures", "chain_ids": chains,
                     "manifest": write_manifest(manifests / f"{name}.txt", chains)})
    if raw_missing:
        jobs.append({"name": "raw", "kind": "raw", "chain_ids": raw_missing,
                     "manifest": write_manifest(manifests / "raw.txt", raw_missing)})
    return jobs


def validate_jobs(jobs: list[dict[str, Any]], output: Path, selected: list[str], catalog: dict[str, Any]) -> None:
    expected_pdbs = {chain.split("_", 1)[0] for chain in selected} - set(catalog["members"])
    requested_pdbs: set[str] = set()
    seen_names = set()
    for job in jobs:
        name, kind, chains = job["name"], job["kind"], job["chain_ids"]
        require(kind in {"structures", "raw"} and name not in seen_names, "Invalid or repeated acquisition job")
        seen_names.add(name)
        require(name == "raw" if kind == "raw" else name == f"structures-{len(seen_names) - 1:05d}",
                "Acquisition job names or ordering changed")
        require(bool(chains) and chains == sorted(set(chains)) and set(chains) <= set(selected),
                "Acquisition manifest contains foreign or duplicate chain IDs")
        manifest = output / "manifests" / f"{name}.txt"
        require(job["manifest"]["path"] == str(manifest) and file_record(manifest) == job["manifest"],
                "Acquisition manifest binding changed")
        require(manifest.read_text() == "\n".join(chains) + "\n", "Acquisition manifest differs from its selected batch")
        if kind == "structures":
            pdbs = {chain.split("_", 1)[0] for chain in chains}
            require(len(chains) <= BATCH_SIZE and len(pdbs) == len(chains) and not (pdbs & requested_pdbs),
                    "Structure acquisition duplicated PDBs or exceeds its batch size")
            requested_pdbs |= pdbs
    require(requested_pdbs == expected_pdbs, "Acquisition jobs differ from missing historical structures")


def command(job: dict[str, Any], destination: Path, args: argparse.Namespace, pin: dict[str, Any]) -> list[str]:
    manifest = job["manifest"]["path"]
    if job["kind"] == "structures":
        return [sys.executable, str(args.acquisition_code_root / ACQUISITION_FILES[0]),
                "--manifest", manifest, "--out-dir", str(destination), "--archive-size", str(pin["size"]),
                "--archive-etag", pin["etag"], "--archive-version-id", "absent" if pin["version_id"] is None else pin["version_id"]]
    return [sys.executable, str(REPO_ROOT / "scripts/prepare_data.py"), "--manifest", manifest,
            "--data-root", str(destination), "--duplicate-chains-file", str(args.duplicate_chains_file),
            "--msa-name", "uniref90_hits.a3m", "--no-template-hits", "--strict-downloads", "--download-workers", "8"]


def verify_job(job: dict[str, Any], destination: Path, args: argparse.Namespace, catalog: dict[str, Any]) -> dict[str, Any]:
    if job["kind"] == "raw":
        rows = []
        for chain in job["chain_ids"]:
            source = _raw_source([destination], chain)
            require(source.stat().st_size > 0, "Retrieved raw source is empty")
            rows.append(file_record(source))
        return {"raw_root": str(destination), "raw_sources": rows}
    receipt = destination / "retrieval-receipt.json"
    sources, members = historical_source_catalog([receipt], acquisition_code_root=args.acquisition_code_root)
    original = catalog["historical_sources"]
    require(all(sources[key] == original[key] for key in ("archive_url", "archive_pin", "central_directory_sha256")),
            "New receipt differs from the frozen archive")
    require(sources["requested_manifests"] == [job["manifest"]]
            and set(members) == {chain.split("_", 1)[0] for chain in job["chain_ids"]},
            "Retrieval receipt differs from its exact requested batch")
    for member in members.values():
        require(sha256_file(Path(member["path"])) == member["sha256"], "Retrieved structure bytes differ from owning receipt")
    return {"retrieval_receipt": file_record(receipt)}


def run_job(job: dict[str, Any], args: argparse.Namespace, output: Path,
            binding: dict[str, Any], catalog: dict[str, Any],
            cancel: threading.Event | None = None) -> dict[str, Any]:
    require(cancel is None or not cancel.is_set(), "Acquisition canceled before subprocess launch")
    root = output / "jobs" / job["name"]
    root.mkdir(parents=True, exist_ok=True)
    accepted = root / "complete.json"
    validate_file_binding(job["manifest"])
    if accepted.exists():
        previous = read_json(accepted)
        require(previous["evidence"] == verify_job(job, Path(previous["destination"]), args, catalog), "Completed acquisition outputs changed")
        return previous
    attempts = sorted(root.glob("attempt-*"))
    for attempt in attempts:
        if not (attempt / "result.json").exists():
            process_path = attempt / "process.json"
            require(process_path.exists(), "Interrupted attempt has no child identity; reconcile before retry")
            pid = read_json(process_path)["pid"]
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                pass
            else:
                raise ValueError("Interrupted acquisition child may still be running; reconcile before retry")
    require(len(attempts) < args.max_attempts, "Acquisition attempt limit exhausted; preserve failures and investigate")
    check_inputs(binding)
    attempt = root / f"attempt-{len(attempts) + 1:03d}"
    raw_namespace = hashlib.sha256(str(output).encode()).hexdigest()[:16]
    destination = (attempt / "retrieved" if job["kind"] == "structures" else
                   args.raw_output_root / raw_namespace / attempt.name)
    require(not destination.exists(), "Acquisition destination is not fresh")
    check_disk(output)
    check_disk(destination)
    attempt.mkdir()
    destination.parent.mkdir(parents=True, exist_ok=True)
    argv = command(job, destination, args, catalog["historical_sources"]["archive_pin"])
    start = {"status": "starting", "argv": argv, "destination": str(destination), "started_at": now()}
    write_json(attempt / "start.json", start)
    process = None
    result: dict[str, Any] = start.copy()
    try:
        with (attempt / "stdout.log").open("xb") as stdout, (attempt / "stderr.log").open("xb") as stderr:
            code_root = args.acquisition_code_root if job["kind"] == "structures" else REPO_ROOT
            environment = os.environ.copy()
            environment["PYTHONPATH"] = str(code_root)
            process = subprocess.Popen(argv, cwd=code_root, env=environment, stdout=stdout, stderr=stderr, start_new_session=True)
            write_json(attempt / "process.json", {"pid": process.pid, "argv": argv, "started_at": now()})
            while True:
                require(cancel is None or not cancel.is_set(), "Acquisition canceled after another worker stopped")
                check_disk(output)
                check_disk(destination)
                try:
                    exit_code = process.wait(timeout=1)
                    break
                except subprocess.TimeoutExpired:
                    continue
            stdout.flush()
            stderr.flush()
            os.fsync(stdout.fileno())
            os.fsync(stderr.fileno())
        result["exit_code"] = exit_code
        require(exit_code == 0, "Source acquisition subprocess failed; no biological exclusion is inferred")
        check_inputs(binding)
        validate_file_binding(job["manifest"])
        result.update(status="retrieved_unqualified", evidence=verify_job(job, destination, args, catalog))
    except BaseException as error:
        if process is not None and process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
        result.update(status="failed_or_interrupted", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        result["finished_at"] = now()
        write_json(attempt / "result.json", result)
    write_json(accepted, result)
    return result


def run(args: argparse.Namespace) -> dict[str, Any]:
    os.umask(0o077)
    for name in ("plan", "source_catalog", "duplicate_chains_file", "acquisition_code_root", "raw_output_root", "output"):
        setattr(args, name, getattr(args, name).resolve())
    if args.requests:
        args.requests = args.requests.resolve()
    args.raw_root = [path.resolve() for path in args.raw_root]
    args.retrieval_receipt = [path.resolve() for path in args.retrieval_receipt]
    output = args.output
    for path in (output, args.raw_output_root):
        require(not path.is_relative_to(REPO_ROOT) or path.is_relative_to(REPO_ROOT / ".nanofold_private"),
                "Acquisition evidence and raw outputs belong in the ignored private root")
    output.mkdir(parents=True, exist_ok=True, mode=0o700)
    with (output / ".writer.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        binding, catalog, selected = freeze_inputs(args)
        frozen = output / "acquisition-plan.json"
        if frozen.exists():
            plan = read_json(frozen)
            require(plan["bindings"] == binding, "Resume differs from frozen acquisition inputs")
        else:
            plan = {"schema_version": 1, "status": "planned", "bindings": binding,
                    "jobs": make_jobs(output, selected, catalog, args.raw_root)}
            write_json(frozen, plan)
        frozen_pin = file_record(frozen)
        validate_jobs(plan["jobs"], output, selected, catalog)
        if not args.execute:
            return {"status": "planned", "plan": file_record(frozen), "job_count": len(plan["jobs"])}
        state = output / "status.json"
        atomic_state(state, {"status": "running", "started_at": now(), "pid": os.getpid()})
        try:
            jobs = plan["jobs"]
            cancel = threading.Event()
            with ThreadPoolExecutor(max_workers=4) as structure_workers, ThreadPoolExecutor(max_workers=1) as raw_workers:
                futures = {(structure_workers if job["kind"] == "structures" else raw_workers).submit(
                    run_job, job, args, output, binding, catalog, cancel,
                ): job["name"] for job in jobs}
                try:
                    completed_jobs = {futures[future]: future.result() for future in as_completed(futures)}
                    results = [completed_jobs[job["name"]] for job in jobs]
                except BaseException:
                    cancel.set()
                    raise
            check_inputs(binding)
            validate_file_binding(frozen_pin)
            validate_jobs(jobs, output, selected, catalog)
            receipts = [Path(row["path"]) for row in catalog["historical_sources"]["retrieval_receipts"]]
            receipts.extend(Path(result["evidence"]["retrieval_receipt"]["path"]) for result in results
                            if "retrieval_receipt" in result["evidence"])
            sources, members = historical_source_catalog(receipts, acquisition_code_root=args.acquisition_code_root,
                                                         source_path_map=relocation_map(catalog))
            raw_roots = sorted(set(binding["raw_roots"] + [result["evidence"]["raw_root"] for result in results
                                                          if "raw_root" in result["evidence"]]))
            for chain in selected:
                require(chain.split("_", 1)[0] in members, "Selected chain lacks an acquired historical structure")
                raw_path = _raw_source([Path(root) for root in raw_roots], chain)
                require(raw_path.stat().st_size > 0, "Selected chain lacks nonempty acquired raw bytes")
            merged = catalog | {"historical_sources": sources, "members": members}
            old_code = {str(Path(row["path"])) for row in catalog["inputs"]
                        if any(Path(row["path"]).as_posix().endswith('/' + name) for name in ACQUISITION_FILES)}
            merged["inputs"] = [row for row in catalog["inputs"] if row["path"] not in old_code]
            merged["inputs"] += [file_record(args.acquisition_code_root / name) for name in ACQUISITION_FILES]
            merged["inputs"] += [file_record(frozen)]
            validate_source_catalog(merged)
            catalog_path = output / "source-catalog.json"
            if catalog_path.exists():
                require(read_json(catalog_path) == merged, "Completed source catalog changed")
            else:
                write_json(catalog_path, merged)
            completed = {"status": "retrieved_unqualified", "biological_identity_validated": False,
                         "source_catalog": file_record(catalog_path), "plan": file_record(frozen),
                         "raw_roots": raw_roots, "selected_chain_count": len(selected), "job_count": len(jobs)}
            receipt_path = output / "receipt.json"
            if receipt_path.exists():
                require(read_json(receipt_path) == completed, "Completed acquisition receipt changed")
            else:
                write_json(receipt_path, completed)
            atomic_state(state, completed | {"finished_at": now()})
            return completed
        except BaseException as error:
            atomic_state(state, {"status": "failed_or_interrupted", "finished_at": now(),
                                 "error_type": type(error).__name__, "error": str(error)})
            raise


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("plan", "source-catalog", "raw-output-root", "duplicate-chains-file", "acquisition-code-root", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--raw-root", action="append", type=Path, required=True, help="Existing raw source root; repeat as needed.")
    parser.add_argument("--requests", type=Path, help="Qualification sources-needed JSONL; defaults to first-ranked copies.")
    parser.add_argument("--retrieval-receipt", action="append", type=Path, default=[],
                        help="Additional completed owning mmCIF retrieval receipt to merge before planning; repeat as needed.")
    parser.add_argument("--max-attempts", type=int, default=2, help="Maximum fresh attempts per job across invocations (1–3).")
    parser.add_argument("--execute", action="store_true", help="Run owning retrievers after freezing the plan; otherwise only plan.")
    args = parser.parse_args(argv)
    require(1 <= args.max_attempts <= 3, "Maximum acquisition attempts must be between one and three")
    return args


def main() -> int:
    result = run(parse_args())
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
