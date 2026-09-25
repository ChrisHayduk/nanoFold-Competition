"""Audit complete selected cache targets before expensive processed-asset preparation."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from nanofold.mmseqs import iter_mmseqs_hits, sequence_records_sha256, validate_search_parameters
from nanofold.structure_metadata import validate_author_chain_id

SCOPE = (
    "Complete selected target sequences from the bound chain cache. All source/query pairs are submitted "
    "to MMseqs local alignment under the recorded contract; returned qualifying alignments and independent "
    "exact-sequence, chain and PDB overlaps fail the gate. Passing does not prove absence of every possible "
    "alignment or homology. Processed features, MSA rows, labels and tensor coverage remain unaudited."
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, data: dict[str, Any]) -> None:
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        try:
            json.dump(data, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
            handle.close()
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)


def read_manifest(path: Path) -> list[str]:
    ids = [line.strip() for line in path.read_text().splitlines()
           if line.strip() and not line.lstrip().startswith("#")]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("A manifest is empty or contains duplicate IDs")
    for chain_id in ids:
        validate_author_chain_id(chain_id)
    return ids


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Chain cache JSON contains duplicate object keys")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise ValueError("Chain cache JSON contains a nonfinite numeric constant")


def sequences_for(ids: list[str], cache: dict[str, Any], *, min_len: int, max_len: int) -> list[str]:
    sequences: list[str] = []
    for chain_id in ids:
        record = cache.get(chain_id)
        if not isinstance(record, dict):
            raise ValueError("A selected chain is absent from the chain cache")
        aliases: list[str] = []
        for key in ("sequence", "seq", "seqres", "aatype_sequence"):
            if key not in record:
                continue
            value = record[key]
            if not isinstance(value, str) or not value.strip():
                raise ValueError("A selected chain has an invalid cache sequence alias")
            aliases.append(value.strip().upper())
        if not aliases or len(set(aliases)) != 1:
            raise ValueError("A selected chain has missing or conflicting cache sequences")
        sequence = aliases[0]
        if not min_len <= len(sequence) <= max_len or set(sequence) - set("ARNDCQEGHILKMFPSTWYV"):
            raise ValueError("A selected cache sequence violates the standard-amino-acid length contract")
        for key in ("seq_length", "sequence_length"):
            length = record.get(key, len(sequence))
            if isinstance(length, bool) or not isinstance(length, int) or length != len(sequence):
                raise ValueError("A selected chain cache length does not match its sequence")
        sequences.append(sequence)
    return sequences


def _write_fasta(path: Path, records: list[tuple[str, str]]) -> None:
    with path.open("w") as handle:
        for header, sequence in records:
            handle.write(f">{header}\n{sequence}\n")


def run_pair(*, source_ids: list[str], query_ids: list[str], source_sequences: list[str],
             query_sequences: list[str], work: Path, binary: str, threads: int,
             min_seq_id: float, coverage: float, max_seqs: int) -> dict[str, Any]:
    source_records = [(f"source_{index}", sequence) for index, sequence in enumerate(source_sequences)]
    query_records = [(f"query_{index}", sequence) for index, sequence in enumerate(query_sequences)]
    source_fasta, query_fasta, m8 = work / "sources.fasta", work / "queries.fasta", work / "alignments.m8"
    _write_fasta(source_fasta, source_records)
    _write_fasta(query_fasta, query_records)
    write_json(work / "private_index_mapping.json", {"source_ids": source_ids, "query_ids": query_ids})
    with tempfile.TemporaryDirectory(prefix="mmseqs-", dir=work) as temporary:
        command = [binary, "easy-search", str(query_fasta), str(source_fasta), str(m8), temporary,
                   "--min-seq-id", str(min_seq_id), "-c", str(coverage), "--cov-mode", "0", "--alignment-mode", "3",
                   "--seq-id-mode", "0", "--search-type", "1", "--dbtype", "1", "--prefilter-mode", "2", "-e", "1e308",
                   "-s", "5.7", "--mask", "0", "--comp-bias-corr", "0", "--threads", str(threads),
                   "--max-seqs", str(max_seqs), "--format-output", "query,target,pident,alnlen,qcov,tcov,evalue,bits"]
        write_json(work / "command.json", {"argv": command})
        with (work / "mmseqs.log").open("w") as log:
            process = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False)
        if process.returncode:
            raise RuntimeError(f"MMseqs target audit failed with exit code {process.returncode}")
    hits = list(iter_mmseqs_hits(
        m8, query_ids={header for header, _ in query_records}, target_ids={header for header, _ in source_records},
        min_seq_id=min_seq_id, coverage=coverage,
    ))
    if len(hits) != len(set(hits)):
        raise ValueError("MMseqs returned duplicate source/query pairs")
    source_sequence_set = set(source_sequences)
    exact_query_count = sum(sequence in source_sequence_set for sequence in query_sequences)
    chain_overlap = len(set(source_ids) & set(query_ids))
    pdb_overlap = len({chain.split("_", 1)[0] for chain in source_ids}
                      & {chain.split("_", 1)[0] for chain in query_ids})
    return {
        "status": "failed" if hits or exact_query_count or chain_overlap or pdb_overlap else "passed",
        "source_records": len(source_records), "query_records": len(query_records),
        "requested_pair_count": len(source_records) * len(query_records),
        "qualifying_alignment_count": len(hits), "queries_with_qualifying_hits": len({query for query, _ in hits}),
        "exact_sequence_overlap_query_count": exact_query_count,
        "chain_overlap_count": chain_overlap, "pdb_overlap_count": pdb_overlap,
        "source_records_sha256": sequence_records_sha256(source_records),
        "query_records_sha256": sequence_records_sha256(query_records),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--chain-data-cache", type=Path, required=True)
    for name in ("train", "val", "public-all", "hidden"):
        parser.add_argument(f"--{name}-manifest", type=Path, required=True)
    parser.add_argument("--expected-counts", type=int, nargs=3, required=True, metavar=("TRAIN", "VAL", "HIDDEN"))
    parser.add_argument("--out-dir", type=Path, required=True, help="Fresh private directory for reports and anonymous alignments.")
    parser.add_argument("--mmseqs-bin", default="mmseqs", help="MMseqs executable path or name, resolved and hashed before auditing.")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--max-seqs", type=int, default=1_000_000, help="Must cover every public source target.")
    parser.add_argument("--min-seq-id", type=float, default=0.30)
    parser.add_argument("--coverage", type=float, default=0.80)
    parser.add_argument("--min-len", type=int, default=40)
    parser.add_argument("--max-len", type=int, default=256)
    return parser.parse_args(argv)


def audit_targets(args: argparse.Namespace) -> int:
    out = args.out_dir.resolve()
    if args.out_dir.is_symlink() or (out.is_relative_to(REPO_ROOT) and not out.is_relative_to(REPO_ROOT / ".nanofold_private")):
        raise ValueError("Audit outputs must remain in a fresh private directory")
    manifests = {name: getattr(args, f"{name}_manifest").resolve() for name in ("train", "val", "public_all", "hidden")}
    inputs = {"chain_data_cache": args.chain_data_cache.resolve(), **{f"manifest_{name}": path for name, path in manifests.items()}}
    if any(path.is_relative_to(out) for path in inputs.values()):
        raise ValueError("Audit output must not contain or overwrite an input")
    for path in inputs.values():
        if not path.is_file():
            raise ValueError("An audit input is missing or is not a file")
    validate_search_parameters(min_seq_id=args.min_seq_id, coverage=args.coverage, threads=args.threads, max_seqs=args.max_seqs)
    if args.threads > (os.cpu_count() or 1):
        raise ValueError("Requested audit threads exceed local CPU capacity")
    if args.min_len < 1 or args.max_len < args.min_len or any(value < 1 for value in args.expected_counts):
        raise ValueError("Invalid target length bounds or expected split counts")
    binary = shutil.which(args.mmseqs_bin)
    if binary is None:
        raise ValueError("MMseqs executable is unavailable")
    binary = str(Path(binary).resolve())
    inputs.update({"audit_script": Path(__file__).resolve(), "shared_parser": REPO_ROOT / "nanofold/mmseqs.py", "mmseqs_binary": Path(binary)})
    original_umask = os.umask(0o077)
    try:
        out.mkdir(mode=0o700, parents=True, exist_ok=False)
        report: dict[str, Any] = {
            "schema_version": 1, "status": "running", "scope": SCOPE, "processed_assets_audited": False,
            "expected_counts": dict(zip(("train", "val", "hidden"), args.expected_counts, strict=True)),
            "pairs": {}, "pair_summary_sha256": {},
            "alignment_contract": {
                "min_seq_id": args.min_seq_id, "coverage": args.coverage, "cov_mode": 0, "alignment_mode": 3,
                "seq_id_mode": 0, "search_type": 1, "dbtype": 1, "prefilter_mode": 2, "evalue": 1e308, "sensitivity": 5.7,
                "mask": 0, "comp_bias_corr": 0, "threads": args.threads, "max_seqs": args.max_seqs,
            },
            "sequence_contract": {"min_len": args.min_len, "max_len": args.max_len, "standard_amino_acids_only": True},
        }
        write_json(out / "summary.json", report)
        try:
            report["inputs"] = {name: {"path": str(path), "sha256": sha256(path)} for name, path in inputs.items()}
            version = subprocess.run([binary, "version"], capture_output=True, text=True, check=True).stdout.strip()
            if not version:
                raise ValueError("MMseqs did not report its version")
            report["mmseqs_version"] = version
            ids = {name: read_manifest(path) for name, path in manifests.items()}
            if [len(ids[name]) for name in ("train", "val", "hidden")] != args.expected_counts:
                raise ValueError("Manifest sizes differ from the expected complete split population")
            if set(ids["public_all"]) != set(ids["train"]) | set(ids["val"]):
                raise ValueError("Public all manifest differs from the exact train/validation union")
            if args.max_seqs < len(ids["public_all"]):
                raise ValueError("max-seqs must cover every source target")
            cache = json.loads(inputs["chain_data_cache"].read_text(), object_pairs_hook=_unique_json_object,
                               parse_constant=_reject_json_constant)
            if not isinstance(cache, dict):
                raise ValueError("Chain cache must be a JSON object")
            for chain_id, record in cache.items():
                validate_author_chain_id(chain_id)
                if not isinstance(record, dict):
                    raise ValueError("Every chain cache record must be a JSON object")
            sequences = {name: sequences_for(chains, cache, min_len=args.min_len, max_len=args.max_len)
                         for name, chains in ids.items()}
            report["cache_record_count"] = len(cache)
            del cache
            write_json(out / "summary.json", report)
            for name, source, query in (("public", "train", "val"), ("hidden", "public_all", "hidden")):
                work = out / name
                work.mkdir(mode=0o700)
                try:
                    pair = run_pair(
                        source_ids=ids[source], query_ids=ids[query], source_sequences=sequences[source],
                        query_sequences=sequences[query], work=work, binary=binary, threads=args.threads,
                        min_seq_id=args.min_seq_id, coverage=args.coverage, max_seqs=args.max_seqs,
                    )
                except Exception as exc:
                    (work / "error.log").write_text(traceback.format_exc())
                    pair = {"status": "error", "error_type": type(exc).__name__,
                            "source_records": len(ids[source]), "query_records": len(ids[query]),
                            "requested_pair_count": len(ids[source]) * len(ids[query])}
                pair["artifact_sha256"] = {path.name: sha256(path) for path in sorted(work.iterdir()) if path.is_file()}
                write_json(work / "summary.json", pair)
                report["pair_summary_sha256"][name] = sha256(work / "summary.json")
                report["pairs"][name] = pair
                write_json(out / "summary.json", report)
                print(json.dumps({"audit": name, **{key: pair[key] for key in (
                    "status", "source_records", "query_records", "requested_pair_count",
                    "qualifying_alignment_count", "queries_with_qualifying_hits",
                ) if key in pair}}), flush=True)
            if any(sha256(path) != report["inputs"][name]["sha256"] for name, path in inputs.items()):
                raise ValueError("Bound inputs or implementation changed during the target audit")
            for name, pair in report["pairs"].items():
                if any(sha256(out / name / filename) != digest for filename, digest in pair["artifact_sha256"].items()):
                    raise ValueError("Bound target-audit artifacts changed during a subsequent pair audit")
                if sha256(out / name / "summary.json") != report["pair_summary_sha256"][name]:
                    raise ValueError("Bound target-audit pair summary changed during a subsequent pair audit")
            report["status"] = ("passed" if all(pair["status"] == "passed" for pair in report["pairs"].values())
                                else "error" if any(pair["status"] == "error" for pair in report["pairs"].values()) else "failed")
        except BaseException as exc:
            report["status"] = "error"
            report["error_type"] = type(exc).__name__
            (out / "error.log").write_text(traceback.format_exc())
        write_json(out / "summary.json", report)
        print(json.dumps({"target_audit_status": report["status"], "processed_assets_audited": False}), flush=True)
        return {"passed": 0, "failed": 1, "error": 2}[report["status"]]
    finally:
        os.umask(original_umask)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        return audit_targets(args)
    except (OSError, ValueError) as exc:
        print(json.dumps({"target_audit_status": "error", "error_type": type(exc).__name__}), flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
