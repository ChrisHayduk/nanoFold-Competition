"""Validate the alignment contract shared by dataset homology searches."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import signal
import struct
import subprocess
import time
import uuid
from collections.abc import Iterator, Mapping, Sequence, Set
from pathlib import Path
from typing import Any

import numpy as np

from nanofold.a3m import GAP_ID, RESTYPES, UNK_ID


def validate_msa_arrays(arrays: Mapping[str, np.ndarray]) -> np.ndarray:
    """Validate the unmasked, query-aligned MSA consumed by homology audits."""
    if "msa" not in arrays or "aatype" not in arrays:
        raise ValueError("Processed MSA auditing requires msa and aatype arrays")
    msa = np.asarray(arrays["msa"])
    query = np.asarray(arrays["aatype"])
    if msa.ndim != 2 or not all(msa.shape):
        raise ValueError("Processed MSA must be a nonempty two-dimensional array")
    if not np.issubdtype(msa.dtype, np.integer) or np.any(msa < 0) or np.any(msa > GAP_ID):
        raise ValueError("Processed MSA must contain integer residue/gap tokens without masks")
    if query.ndim != 1 or not np.issubdtype(query.dtype, np.integer):
        raise ValueError("Processed query must be a one-dimensional integer array")
    if np.any(query < 0) or np.any(query > UNK_ID):
        raise ValueError("Processed query must contain residue tokens without gaps or masks")
    if not np.array_equal(msa[0], query):
        raise ValueError("Processed MSA query row does not match aatype")
    if "deletions" in arrays:
        deletions = np.asarray(arrays["deletions"])
        if deletions.shape != msa.shape or not np.issubdtype(deletions.dtype, np.integer) or np.any(deletions < 0):
            raise ValueError("Processed deletion counts must be nonnegative integers matching the MSA")
    return msa


def load_msa_for_audit(path: Path, *, chain_id: str) -> np.ndarray:
    with np.load(path, allow_pickle=False) as data:
        msa = validate_msa_arrays(data)
        if "chain_id" in data.files:
            identity = np.asarray(data["chain_id"])
            if identity.shape != () or str(identity.item()) != chain_id:
                raise ValueError("Processed feature chain identity does not match its manifest")
        return msa


def canonical_target_sequence(sequence: str) -> str:
    """Match preprocessing's unknown-residue normalization without accepting FASTA syntax."""
    normalized = sequence.strip().upper()
    if not normalized or any(char not in RESTYPES + "XBZJUO" for char in normalized):
        raise ValueError("Target sequence must be a nonempty ungapped amino-acid sequence")
    return "".join(char if char in RESTYPES else "X" for char in normalized)


def sequence_records_sha256(records: Sequence[tuple[str, str]]) -> str:
    """Bind anonymous record indices to their ordered sequence contents."""
    rows = [(header, hashlib.sha256(sequence.encode()).hexdigest()) for header, sequence in records]
    return hashlib.sha256(json.dumps(rows, separators=(",", ":")).encode()).hexdigest()


def validate_search_parameters(*, min_seq_id: float, coverage: float, threads: int, max_seqs: int) -> None:
    if not 0 < min_seq_id <= 1 or not 0 < coverage <= 1:
        raise ValueError("MMseqs identity and coverage thresholds must be in (0, 1]")
    # MMseqs parses these CLI counts with atoi, including size_t max-seqs.
    # Exhaustive searches also reuse max-seqs for max-accept/max-rejected.
    if not 1 <= threads <= 2_147_483_647 or not 1 <= max_seqs <= 2_147_483_647:
        raise ValueError("MMseqs threads and max-seqs must be in [1, 2147483647]")


def iter_mmseqs_hits(
    path: Path,
    *,
    query_ids: Set[str],
    target_ids: Set[str],
    min_seq_id: float,
    coverage: float,
) -> Iterator[tuple[str, str]]:
    """Read known query/target pairs from validated eight-column MMseqs output.

    Searches request pident in percent and qcov/tcov as fractions. A tolerance
    of 0.001 in fractional units accommodates MMseqs' printed-value rounding.
    The caller must use alignment mode 3, sequence identity mode 0, and coverage
    mode 0 so these checks have the same meaning as the search thresholds.
    """
    if not path.is_file():
        raise RuntimeError("MMseqs completed without writing the expected alignment output")
    with path.open() as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            columns = line.rstrip("\r\n").split("\t")
            if len(columns) != 8:
                raise ValueError(f"MMseqs alignment row {line_number} must contain eight columns")
            query, target = columns[:2]
            if query not in query_ids or target not in target_ids:
                raise ValueError(f"MMseqs alignment row {line_number} references an unknown sequence")
            try:
                identity = float(columns[2]) / 100
                alignment_length = int(columns[3])
                query_coverage, target_coverage, evalue, bits = map(float, columns[4:])
            except ValueError as exc:
                raise ValueError(f"MMseqs alignment row {line_number} contains invalid numeric fields") from exc
            values = (identity, query_coverage, target_coverage, evalue, bits)
            if (
                not all(math.isfinite(value) for value in values)
                or alignment_length < 1
                or not 0 <= identity <= 1
                or not 0 <= query_coverage <= 1
                or not 0 <= target_coverage <= 1
                or evalue < 0
                or bits < 0
            ):
                raise ValueError(f"MMseqs alignment row {line_number} contains out-of-range numeric fields")
            if identity + 0.001 < min_seq_id or min(query_coverage, target_coverage) + 0.001 < coverage:
                raise ValueError(f"MMseqs alignment row {line_number} does not meet the requested thresholds")
            yield query, target


_BATCH_CONTRACT = "mmseqs-local-all-pairs-v1"
_BATCH_HEADER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}")
_BATCH_SEQUENCE = re.compile(r"[ACDEFGHIKLMNPQRSTVWYX]{1,256}")


def _batch_sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _batch_json(path: Path, value: Any) -> None:
    """Publish a complete file, never replace existing checkpoint evidence."""
    data = (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    with temporary.open("xb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.link(temporary, path)
    finally:
        temporary.unlink()


def _batch_records(path: Path) -> Iterator[tuple[str, str]]:
    previous = ""
    with path.open("r", encoding="ascii", newline="") as handle:
        while header_line := handle.readline():
            sequence_line = handle.readline()
            header, sequence = header_line[1:-1], sequence_line[:-1]
            if (
                not header_line.startswith(">") or not header_line.endswith("\n")
                or not sequence_line.endswith("\n") or not _BATCH_HEADER.fullmatch(header)
                or not _BATCH_SEQUENCE.fullmatch(sequence)
            ):
                raise ValueError("Batched MMseqs requires safe single-line canonical FASTA records of length 1–256")
            if header <= previous:
                raise ValueError("Batched MMseqs FASTA headers must be unique and lexicographically increasing")
            previous = header
            yield header, sequence


def _batch_partitions(path: Path, directory: Path, size: int) -> list[dict[str, Any]]:
    partitions: list[dict[str, Any]] = []
    records = iter(_batch_records(path))
    while first := next(records, None):
        output = directory / f"{len(partitions):06d}.fasta"
        histogram: dict[str, int] = {}
        count = 0
        max_header_length = 0
        with output.open("x", encoding="ascii", newline="\n") as handle:
            record: tuple[str, str] | None = first
            while record is not None:
                header, sequence = record
                max_header_length = max(max_header_length, len(header))
                handle.write(f">{header}\n{sequence}\n")
                length = str(len(sequence))
                histogram[length] = histogram.get(length, 0) + 1
                count += 1
                record = next(records, None) if count < size else None
        partitions.append({"path": str(output), "sha256": _batch_sha256(output), "count": count,
                           "length_histogram": histogram, "bytes": output.stat().st_size,
                           "max_header_length": max_header_length})
    return partitions


def _batch_verify_partitions(parts: list[dict[str, Any]], expected: dict[str, Any], size: int, root: Path) -> None:
    digest = hashlib.sha256()
    previous = ""
    total = 0
    for index, part in enumerate(parts):
        path = _batch_evidence_path(part["path"], root)
        histogram: dict[str, int] = {}
        count = 0
        max_header_length = 0
        for header, sequence in _batch_records(path):
            if header <= previous:
                raise ValueError("Checkpoint partitions overlap or have changed order")
            previous = header
            max_header_length = max(max_header_length, len(header))
            count += 1
            length = str(len(sequence))
            histogram[length] = histogram.get(length, 0) + 1
        if not 1 <= count <= size or (index < len(parts) - 1 and count != size):
            raise ValueError("Checkpoint partition is incomplete")
        if (count != part["count"] or histogram != part["length_histogram"] or path.stat().st_size != part["bytes"]
                or max_header_length != part["max_header_length"]):
            raise ValueError("Checkpoint partition metadata mismatch")
        if _batch_sha256(path) != part["sha256"]:
            raise ValueError("Checkpoint partition digest mismatch")
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
        total += count
    if digest.hexdigest() != expected["sha256"] or total != expected["count"]:
        raise ValueError("Checkpoint partitions do not cover the complete input FASTA")


def _batch_evidence_path(value: str, root: Path) -> Path:
    path = Path(value)
    if path.is_symlink() or path.resolve() != path or not path.is_relative_to(root) or not path.is_file():
        raise ValueError("Checkpoint evidence must be a regular file inside its checkpoint directory")
    return path


def _batch_expected_alignments(query: dict[str, int], source: dict[str, int], coverage: float) -> int:
    def f32(value: float) -> float:
        return struct.unpack("f", struct.pack("f", value))[0]

    return sum(
        nq * ns for qlen, nq in query.items() for slen, ns in source.items()
        if min(f32(int(qlen) / int(slen)), f32(int(slen) / int(qlen))) >= f32(coverage)
    )


def _batch_log_counts(path: Path) -> dict[str, int]:
    contents = path.read_text(errors="strict").replace("\r\n", "\n").replace("\r", "\n")
    patterns = {
        "query_count": r"^Query database size: ([0-9]+) type: Aminoacid$",
        "source_count": r"^Target database size: ([0-9]+) type: Aminoacid$",
        "alignment_count": r"^([0-9]+) alignments calculated$",
        "hit_count": r"^([0-9]+) sequence pairs passed the thresholds(?: \([^\r\n]*\))?$",
    }
    result = {}
    for name, pattern in patterns.items():
        matches = re.findall(pattern, contents, flags=re.MULTILINE)
        if len(matches) != 1:
            raise ValueError(f"MMseqs log must contain exactly one {name} accounting line")
        result[name] = int(matches[0])
    return result


def _batch_unique_hits(path: Path, query_path: Path, source_path: Path, *, min_seq_id: float, coverage: float) -> Iterator[tuple[str, str]]:
    query_ids = {header for header, _ in _batch_records(query_path)}
    source_ids = {header for header, _ in _batch_records(source_path)}
    seen_queries: set[str] = set()
    seen_targets: set[str] = set()
    current = None
    for query, target in iter_mmseqs_hits(path, query_ids=query_ids, target_ids=source_ids,
                                        min_seq_id=min_seq_id, coverage=coverage):
        if query != current:
            if query in seen_queries:
                raise ValueError("MMseqs output repeats a query block")
            seen_queries.add(query)
            seen_targets.clear()
            current = query
        if target in seen_targets:
            raise ValueError("MMseqs output contains a duplicate query/target pair")
        seen_targets.add(target)
        yield query, target


def _batch_rss_bytes(process_group: int | None) -> int:
    """Sample the caller plus its launched process group; RSS can double-count shared pages."""
    proc = subprocess.run(["/bin/ps", "-axo", "pid=,pgid=,rss="], capture_output=True, text=True,
                          check=True, timeout=10, env={**os.environ, "LC_ALL": "C"})
    total = 0
    found_caller = False
    for line in proc.stdout.splitlines():
        columns = line.split()
        if len(columns) != 3 or not all(value.isdecimal() for value in columns):
            raise RuntimeError("Cannot safely parse process RSS monitoring output")
        pid, pgid, rss = map(int, columns)
        if pid == os.getpid() or pgid == process_group:
            total += rss * 1024
        found_caller |= pid == os.getpid()
    if not found_caller:
        raise RuntimeError("Process RSS monitoring could not observe its calling process")
    return total


def _batch_resource_check(root: Path, *, reserve: int, max_rss: int, process_group: int | None = None) -> tuple[int, int]:
    free = shutil.disk_usage(root).free
    rss = _batch_rss_bytes(process_group)
    if free < reserve:
        raise RuntimeError("Batched MMseqs stopped at its free-disk guard")
    if rss > max_rss:
        raise RuntimeError("Batched MMseqs stopped at its sampled process-tree RSS guard")
    return free, rss


def _batch_stop_process(proc: subprocess.Popen[bytes]) -> None:
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        pass
    # An intermediate shell can exit before its children. Kill the group even
    # after wait() reports that the group leader has exited.
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    proc.wait(timeout=5)


def _batch_execute(command: list[str], log: Path, root: Path, *, reserve: int, max_rss: int) -> dict[str, Any]:
    minimum_free, peak_rss = _batch_resource_check(root, reserve=reserve, max_rss=max_rss)
    started = time.monotonic()
    with log.open("xb") as output:
        proc = subprocess.Popen(command, stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            while True:
                free, rss = _batch_resource_check(root, reserve=reserve, max_rss=max_rss, process_group=proc.pid)
                minimum_free, peak_rss = min(minimum_free, free), max(peak_rss, rss)
                if proc.poll() is not None:
                    try:
                        os.killpg(proc.pid, 0)
                    except ProcessLookupError:
                        break
                    raise RuntimeError("MMseqs leader exited while its process group remains active")
                time.sleep(0.25)
            if proc.returncode != 0:
                raise RuntimeError(f"MMseqs batch exited with status {proc.returncode}; its attempt is retained")
        except BaseException:
            _batch_stop_process(proc)
            raise
    return {"elapsed_seconds": time.monotonic() - started, "sampled_peak_rss_bytes": peak_rss,
            "minimum_sampled_free_disk_bytes": minimum_free}


def _batch_validate_job(job: dict[str, Any], query: dict[str, Any], source: dict[str, Any], *,
                        plan_sha256: str, root: Path, job_root: Path, binary: Path, flags: list[str],
                        min_seq_id: float, coverage: float) -> None:
    if job["schema_version"] != 1 or job["plan_sha256"] != plan_sha256 or job["status"] != "complete":
        raise ValueError("Completed job does not belong to the immutable search plan")
    m8, log = Path(job["m8_path"]), Path(job["log_path"])
    if (m8.name != "alignments.m8" or log != m8.parent / "mmseqs.log" or m8.parent.parent != job_root
            or not m8.parent.name.startswith("attempt-")):
        raise ValueError("Completed job output paths do not belong to its attempt")
    expected_command = [str(binary), "easy-search", query["path"], source["path"], str(m8), str(m8.parent / "native-tmp"), *flags]
    if job["command"] != expected_command:
        raise ValueError("Completed job command does not match the immutable exhaustive contract")
    for prefix, partition in (("query", query), ("source", source)):
        if job[f"{prefix}_count"] != partition["count"]:
            raise ValueError("Completed job count does not match its partition")
        if job[f"{prefix}_fasta_path"] != partition["path"] or job[f"{prefix}_fasta_sha256"] != partition["sha256"]:
            raise ValueError("Completed job does not bind its input partition")
        if _batch_sha256(_batch_evidence_path(partition["path"], root)) != partition["sha256"]:
            raise ValueError("Completed job input digest mismatch")
    for kind in ("m8", "log"):
        if _batch_sha256(_batch_evidence_path(job[f"{kind}_path"], root)) != job[f"{kind}_sha256"]:
            raise ValueError("Completed job output digest mismatch")
    counts = _batch_log_counts(Path(job["log_path"]))
    expected = _batch_expected_alignments(query["length_histogram"], source["length_histogram"], coverage)
    hits = sum(1 for _ in _batch_unique_hits(Path(job["m8_path"]), Path(query["path"]), Path(source["path"]),
                                            min_seq_id=min_seq_id, coverage=coverage))
    if counts != {"query_count": query["count"], "source_count": source["count"], "alignment_count": expected, "hit_count": hits}:
        raise ValueError("MMseqs native accounting disagrees with the complete batch inputs or validated output")
    if job["hit_count"] != hits or job["alignment_count"] != expected:
        raise ValueError("Completed job receipt counts disagree with verified evidence")
    for kind in ("m8", "log"):
        if _batch_sha256(Path(job[f"{kind}_path"])) != job[f"{kind}_sha256"]:
            raise ValueError("Completed job evidence changed during verification")
    for partition in (query, source):
        if _batch_sha256(Path(partition["path"])) != partition["sha256"]:
            raise ValueError("Completed job input changed during verification")


def run_batched_mmseqs(*, query_fasta: Path, source_fasta: Path, checkpoint_dir: Path, mmseqs_bin: str,
                       min_seq_id: float, coverage: float, threads: int, max_seqs: int,
                       source_batch_size: int = 1_000_000, query_batch_size: int = 256,
                       reserve_disk_bytes: int = 100 * 1024**3, max_rss_bytes: int = 48 * 1024**3,
                       input_binding_sha256: str | None = None) -> dict[str, Any]:
    """Search every partition pair and resume only hash-verified completed jobs.

    FASTAs must have sorted unique safe headers and canonical sequences of 1–256
    residues. Checkpoints are private audit evidence. A POSIX process lock and
    process-group control are required; unavailable RSS monitoring fails closed.
    Disk admission reserves 400 bytes plus twice the combined header lengths
    per candidate pair, plus database overhead. RSS guards are sampled, not
    operating-system memory caps.
    """
    import fcntl

    validate_search_parameters(min_seq_id=min_seq_id, coverage=coverage, threads=threads, max_seqs=max_seqs)
    if any(type(value) is not int for value in (threads, max_seqs, source_batch_size, query_batch_size, reserve_disk_bytes, max_rss_bytes)):
        raise ValueError("Batch sizes, resource limits, and native count options must be integers")
    if not 1 <= source_batch_size <= 1_000_000 or not 1 <= query_batch_size <= 256:
        raise ValueError("Source/query batch sizes must be in 1–1000000 and 1–256")
    if 32 * source_batch_size + 1 >= 2**32 or 256 * source_batch_size + 1 >= 2**32:
        raise ValueError("Source batch exceeds native record-size bounds")
    if reserve_disk_bytes < 0 or max_rss_bytes <= 0:
        raise ValueError("Disk reserve must be nonnegative and RSS limit positive")
    if input_binding_sha256 is not None and not re.fullmatch(r"[0-9a-f]{64}", input_binding_sha256):
        raise ValueError("Input binding must be a lowercase SHA-256 digest")
    query_fasta, source_fasta = query_fasta.resolve(strict=True), source_fasta.resolve(strict=True)
    executable = shutil.which(mmseqs_bin)
    if executable is None:
        raise ValueError("MMseqs executable was not found")
    binary = Path(executable).resolve(strict=True)
    version = subprocess.run([str(binary), "version"], capture_output=True, text=True, check=True, timeout=30).stdout.strip()
    if not version:
        raise ValueError("MMseqs must report a nonempty version")
    flags = ["--min-seq-id", str(min_seq_id), "-c", str(coverage), "--cov-mode", "0", "--alignment-mode", "3",
             "--seq-id-mode", "0", "--search-type", "1", "--dbtype", "1", "--prefilter-mode", "2", "-e", "1e308",
             "-s", "5.7", "--mask", "0", "--comp-bias-corr", "0", "--threads", str(threads),
             "--max-seqs", str(max_seqs), "--max-accept", str(max_seqs), "--max-rejected", str(max_seqs),
             "--format-output", "query,target,pident,alnlen,qcov,tcov,evalue,bits"]
    binding = {"schema_version": 1, "search_contract": _BATCH_CONTRACT,
               "query_input": {"sha256": _batch_sha256(query_fasta)},
               "source_input": {"sha256": _batch_sha256(source_fasta)},
               "binary": {"path": str(binary), "sha256": _batch_sha256(binary), "version": version},
               "engine_sha256": _batch_sha256(Path(__file__)), "flags": flags,
               "source_batch_size": source_batch_size, "query_batch_size": query_batch_size,
               "reserve_disk_bytes": reserve_disk_bytes, "max_rss_bytes": max_rss_bytes,
               "input_binding_sha256": input_binding_sha256}
    if checkpoint_dir.is_symlink():
        raise ValueError("Checkpoint directory must not be a symlink")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    root = checkpoint_dir.resolve(strict=True)
    lock_path = root / ".lock"
    if lock_path.is_symlink():
        raise ValueError("Checkpoint lock must not be a symlink")
    with lock_path.open("a+b") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Another process holds this MMseqs checkpoint lock") from exc
        return _run_batched_locked(root, binding, binary, flags, min_seq_id=min_seq_id, coverage=coverage,
                                   max_seqs=max_seqs, reserve=reserve_disk_bytes, max_rss=max_rss_bytes,
                                   input_paths={"query": query_fasta, "source": source_fasta})


def _run_batched_locked(root: Path, binding: dict[str, Any], binary: Path, flags: list[str], *,
                        min_seq_id: float, coverage: float, max_seqs: int, reserve: int, max_rss: int,
                        input_paths: dict[str, Path]) -> dict[str, Any]:
    plan_path = root / "plan.json"
    plan: dict[str, Any]
    if plan_path.is_symlink() or (not plan_path.exists() and ((root / "complete.json").exists() or any(root.glob("job-*")))):
        raise ValueError("Checkpoint has completed or attempted work without its immutable plan")
    if plan_path.exists():
        plan = json.loads(_batch_evidence_path(str(plan_path), root).read_text())
        if plan["binding"] != binding:
            raise ValueError("Immutable MMseqs checkpoint plan does not match current inputs, binary, engine, or parameters")
    else:
        _batch_resource_check(root, reserve=reserve + sum(path.stat().st_size for path in input_paths.values()), max_rss=max_rss)
        staging = root / ("partitions-" + uuid.uuid4().hex)
        staging.mkdir()
        plan = {"binding": binding}
        for prefix in ("query", "source"):
            directory = staging / prefix
            directory.mkdir()
            parts = _batch_partitions(input_paths[prefix], directory, binding[f"{prefix}_batch_size"])
            if not parts:
                raise ValueError("Batched MMseqs requires nonempty query and source FASTAs")
            plan[f"{prefix}_partitions"] = parts
            plan[f"{prefix}_count"] = sum(part["count"] for part in parts)
        if any(max_seqs < part["count"] for part in plan["source_partitions"]):
            raise ValueError("Exhaustive max-seqs must cover every source in the largest batch")
        for prefix in ("query", "source"):
            if _batch_sha256(input_paths[prefix]) != binding[f"{prefix}_input"]["sha256"]:
                raise ValueError("Input FASTA changed during partitioning")
        _batch_json(plan_path, plan)
    for prefix in ("query", "source"):
        if not plan[f"{prefix}_partitions"] or plan[f"{prefix}_count"] < 1:
            raise ValueError("Batched MMseqs requires nonempty query and source FASTAs")
        _batch_verify_partitions(plan[f"{prefix}_partitions"], {**binding[f"{prefix}_input"], "count": plan[f"{prefix}_count"]},
                                 binding[f"{prefix}_batch_size"], root)
    plan_sha = _batch_sha256(plan_path)
    if (root / "complete.json").exists() and any(
        not (root / f"job-{source_index:06d}-{query_index:06d}" / "complete.json").is_file()
        for source_index in range(len(plan["source_partitions"]))
        for query_index in range(len(plan["query_partitions"]))
    ):
        raise ValueError("Final checkpoint is missing a completed job receipt")
    jobs = []
    completed: dict[str, dict[str, Any]] = {}
    # Validate all existing completion evidence before scheduling missing work.
    for source_index, source in enumerate(plan["source_partitions"]):
        for query_index, query in enumerate(plan["query_partitions"]):
            job_root = root / f"job-{source_index:06d}-{query_index:06d}"
            if job_root.is_symlink() or job_root.resolve() != job_root:
                raise ValueError("Checkpoint job directory must not follow a symlink")
            receipt_path = job_root / "complete.json"
            if receipt_path.exists():
                job = json.loads(_batch_evidence_path(str(receipt_path), root).read_text())
                _batch_validate_job(job, query, source, plan_sha256=plan_sha, root=root, job_root=job_root,
                                    binary=binary, flags=flags, min_seq_id=min_seq_id, coverage=coverage)
                completed[str(job_root)] = job
    for source_index, source in enumerate(plan["source_partitions"]):
        for query_index, query in enumerate(plan["query_partitions"]):
            job_root = root / f"job-{source_index:06d}-{query_index:06d}"
            job_root.mkdir(exist_ok=True)
            receipt_path = job_root / "complete.json"
            if str(job_root) in completed:
                job = completed[str(job_root)]
            else:
                pair_bytes = 400 + 2 * (source["max_header_length"] + query["max_header_length"])
                required = reserve + pair_bytes * source["count"] * query["count"] + 4 * (source["bytes"] + query["bytes"]) + 1024**2
                _batch_resource_check(root, reserve=required, max_rss=max_rss)
                attempt = job_root / ("attempt-" + uuid.uuid4().hex)
                attempt.mkdir()
                m8_path, log_path = attempt / "alignments.m8", attempt / "mmseqs.log"
                command = [str(binary), "easy-search", query["path"], source["path"], str(m8_path), str(attempt / "native-tmp"), *flags]
                metrics = _batch_execute(command, log_path, root, reserve=reserve, max_rss=max_rss)
                counts = _batch_log_counts(log_path)
                job = {"schema_version": 1, "status": "complete", "plan_sha256": plan_sha, "command": command,
                       "query_fasta_path": query["path"], "query_fasta_sha256": query["sha256"], "query_count": query["count"],
                       "source_fasta_path": source["path"], "source_fasta_sha256": source["sha256"], "source_count": source["count"],
                       "m8_path": str(m8_path), "m8_sha256": _batch_sha256(m8_path), "log_path": str(log_path),
                       "log_sha256": _batch_sha256(log_path), "hit_count": counts["hit_count"],
                       "alignment_count": counts["alignment_count"], "metrics": metrics}
                _batch_validate_job(job, query, source, plan_sha256=plan_sha, root=root, job_root=job_root,
                                    binary=binary, flags=flags, min_seq_id=min_seq_id, coverage=coverage)
            if not receipt_path.exists():
                _batch_json(receipt_path, job)
            receipt_sha = _batch_sha256(receipt_path)
            if json.loads(receipt_path.read_text()) != job or _batch_sha256(receipt_path) != receipt_sha:
                raise ValueError("Completed job receipt changed during verification")
            jobs.append({**job, "receipt_path": str(receipt_path), "receipt_sha256": receipt_sha})
    evidence = {}
    for job in jobs:
        for kind in ("m8", "log", "query_fasta", "source_fasta", "receipt"):
            evidence[job[f"{kind}_path"]] = job[f"{kind}_sha256"]
    for evidence_path, expected_sha in evidence.items():
        if _batch_sha256(_batch_evidence_path(evidence_path, root)) != expected_sha:
            raise ValueError("Completed batch evidence changed before final receipt creation")
    for item in ({"path": str(input_paths[prefix]), **binding[f"{prefix}_input"]} for prefix in ("query", "source")):
        if _batch_sha256(Path(item["path"])) != item["sha256"]:
            raise ValueError("Search input or executable changed during the batched search")
    if (_batch_sha256(binary) != binding["binary"]["sha256"] or _batch_sha256(Path(__file__)) != binding["engine_sha256"]
            or _batch_sha256(plan_path) != plan_sha):
        raise ValueError("Search engine or immutable plan changed during the batched search")
    result = {"schema_version": 1, "status": "complete", "search_contract": _BATCH_CONTRACT,
              "plan_path": str(plan_path), "plan_sha256": plan_sha, "batch_count": len(jobs),
              "source_batch_size": binding["source_batch_size"], "query_batch_size": binding["query_batch_size"],
              "source_partition_count": len(plan["source_partitions"]), "query_partition_count": len(plan["query_partitions"]),
              "source_count": plan["source_count"], "query_count": plan["query_count"],
              "possible_pair_count": plan["source_count"] * plan["query_count"],
              "alignments_calculated": sum(job["alignment_count"] for job in jobs),
              "hit_count": sum(job["hit_count"] for job in jobs), "jobs": jobs}
    final_path = root / "complete.json"
    if final_path.exists():
        if json.loads(_batch_evidence_path(str(final_path), root).read_text()) != result:
            raise ValueError("Final checkpoint receipt does not match its verified jobs")
    else:
        _batch_json(final_path, result)
    return {**result, "receipt_path": str(final_path), "receipt_sha256": _batch_sha256(final_path)}


def batched_mmseqs_summary(result: dict[str, Any]) -> dict[str, Any]:
    """Return the aggregate checkpoint binding without sequence or job details."""
    fields = ("schema_version", "status", "source_batch_size", "query_batch_size", "batch_count",
              "source_partition_count", "query_partition_count", "source_count", "query_count",
              "alignments_calculated", "receipt_path", "receipt_sha256")
    return {field: result[field] for field in fields}


def iter_batched_mmseqs_hits(result: dict[str, Any], *, min_seq_id: float, coverage: float) -> Iterator[tuple[str, str]]:
    """Stream verified batches using at most one partition's identifier sets."""
    receipt_path = Path(result["receipt_path"])
    if result.get("status") != "complete" or _batch_sha256(receipt_path) != result["receipt_sha256"]:
        raise ValueError("Batched search result is not a verified complete receipt")
    if json.loads(receipt_path.read_text()) != {key: value for key, value in result.items() if key not in ("receipt_path", "receipt_sha256")}:
        raise ValueError("Batched search result differs from its receipt")
    root = receipt_path.parent
    plan_path = _batch_evidence_path(result["plan_path"], root)
    if _batch_sha256(plan_path) != result["plan_sha256"]:
        raise ValueError("Batched search plan changed before output streaming")
    plan = json.loads(plan_path.read_text())
    flags = plan["binding"]["flags"]
    if flags[flags.index("--min-seq-id") + 1] != str(min_seq_id) or flags[flags.index("-c") + 1] != str(coverage):
        raise ValueError("Streamed thresholds differ from the immutable search plan")
    all_evidence = {str(receipt_path): result["receipt_sha256"], str(plan_path): result["plan_sha256"]}
    total = 0
    for job in result["jobs"]:
        job_receipt = _batch_evidence_path(job["receipt_path"], root)
        if _batch_sha256(job_receipt) != job["receipt_sha256"]:
            raise ValueError("Completed job receipt changed before output streaming")
        if json.loads(job_receipt.read_text()) != {key: value for key, value in job.items() if key not in ("receipt_path", "receipt_sha256")}:
            raise ValueError("Completed job differs from its receipt")
        all_evidence[str(job_receipt)] = job["receipt_sha256"]
        log_path = _batch_evidence_path(job["log_path"], root)
        if _batch_sha256(log_path) != job["log_sha256"]:
            raise ValueError("Completed job log changed before output streaming")
        all_evidence[str(log_path)] = job["log_sha256"]
        paths = [(Path(job[f"{kind}_path"]), job[f"{kind}_sha256"]) for kind in ("m8", "query_fasta", "source_fasta")]
        for path, digest in paths:
            if _batch_sha256(_batch_evidence_path(str(path), root)) != digest:
                raise ValueError("Batch evidence changed before output streaming")
            all_evidence[str(path)] = digest
        count = 0
        for pair in _batch_unique_hits(paths[0][0], paths[1][0], paths[2][0], min_seq_id=min_seq_id, coverage=coverage):
            count += 1
            yield pair
        for path, digest in paths:
            if _batch_sha256(path) != digest:
                raise ValueError("Batch evidence changed during output streaming")
        if count != job["hit_count"]:
            raise ValueError("Streamed batch hit count differs from its receipt")
        total += count
    if total != result["hit_count"] or _batch_sha256(receipt_path) != result["receipt_sha256"]:
        raise ValueError("Streamed search differs from its complete receipt")
    for evidence_name, digest in all_evidence.items():
        if _batch_sha256(_batch_evidence_path(evidence_name, root)) != digest:
            raise ValueError("Batch evidence changed during output streaming")
