"""Build and verify a private CATH candidate without changing a published release."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from build_manifests import _atomic_write, _read_hidden_split_salt, _sha256

AUDIT_ARTIFACTS = frozenset({
    "sources.fasta", "queries.fasta", "alignments.m8", "private_index_mapping.json", "command.json", "mmseqs.log",
})


def _validate_group_coverage(report: dict[str, Any], lock: dict[str, Any], args: argparse.Namespace) -> None:
    coverage = report.get("group_coverage")
    groups = lock.get("split_groups")
    if not isinstance(coverage, dict) or not isinstance(groups, dict):
        raise ValueError("cath-v3 verification must prove complete group coverage.")
    count, unit_count = groups.get("group_count"), groups.get("selection_unit_count")
    unit_digest = groups.get("selection_unit_ids_sha256")
    if (type(count) is not int or count <= args.val_size + args.hidden_val_size
            or type(unit_count) is not int or unit_count < args.train_size + args.val_size + args.hidden_val_size
            or not isinstance(unit_digest, str) or re.fullmatch(r"[0-9a-f]{64}", unit_digest) is None):
        raise ValueError("cath-v3 source lock has invalid group or representative counts.")
    expected = {
        "required": "all_groups", "all_group_coverage_verified": True,
        "complete_group_count": count, "covered_group_count": count, "selection_unit_count": unit_count,
        "selection_unit_ids_sha256": unit_digest,
        "split_group_counts": {"train": count - args.val_size - args.hidden_val_size,
                               "val": args.val_size, "hidden_val": args.hidden_val_size},
    }
    if (set(coverage) != set(expected)
            or any(type(coverage[key]) is not type(value) or coverage[key] != value for key, value in expected.items())
            or any(type(value) is not int for value in coverage["split_group_counts"].values())):
        raise ValueError("cath-v3 verification did not cover every locked group with the required split ownership.")


def _read_audit_json(path: Path) -> dict[str, Any]:
    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("Target audit receipt contains duplicate object keys.")
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise ValueError("Target audit receipt contains nonfinite numeric values.")

    if path.is_symlink() or not path.is_file():
        raise ValueError("Target audit receipt is not a regular file.")
    raw = json.loads(path.read_text(), object_pairs_hook=unique_object, parse_constant=reject_constant)
    if not isinstance(raw, dict):
        raise ValueError("Target audit receipt must be a JSON object.")
    return raw


def _validate_target_audit_receipt(
    receipt: Path, *, inputs: dict[str, Path], input_hashes: dict[str, str],
    expected_counts: dict[str, int], min_seq_id: float, coverage: float,
    min_len: int, max_len: int, threads: int, max_seqs: int,
) -> None:
    report = _read_audit_json(receipt)
    if (type(report.get("schema_version")) is not int or report["schema_version"] != 1
            or report.get("status") != "passed" or report.get("processed_assets_audited") is not False
            or not isinstance(report.get("scope"), str) or not report["scope"]
            or not isinstance(report.get("mmseqs_version"), str) or not report["mmseqs_version"].strip()):
        raise ValueError("Target audit receipt is incomplete or did not pass.")

    def require_contract(field: str, expected: dict[str, Any]) -> None:
        actual = report.get(field)
        if (not isinstance(actual, dict) or set(actual) != set(expected)
                or any(type(actual[key]) is not type(value) or actual[key] != value for key, value in expected.items())):
            raise ValueError(f"Target audit receipt does not match the requested {field}.")

    require_contract("expected_counts", expected_counts)
    require_contract("alignment_contract", {
        "min_seq_id": min_seq_id, "coverage": coverage, "cov_mode": 0, "alignment_mode": 3,
        "seq_id_mode": 0, "search_type": 1, "dbtype": 1, "prefilter_mode": 2,
        "evalue": 1e308, "sensitivity": 5.7, "mask": 0, "comp_bias_corr": 0,
        "threads": threads, "max_seqs": max_seqs,
    })
    require_contract("sequence_contract", {
        "min_len": min_len, "max_len": max_len, "standard_amino_acids_only": True,
    })
    reported_inputs = report.get("inputs")
    if not isinstance(reported_inputs, dict) or set(reported_inputs) != set(inputs):
        raise ValueError("Target audit receipt does not bind all requested inputs and implementations.")
    for name, path in inputs.items():
        expected = {"path": str(path.resolve()), "sha256": input_hashes[name]}
        if reported_inputs[name] != expected or _sha256(path) != input_hashes[name]:
            raise ValueError("Target audit receipt input or implementation binding does not match this candidate.")

    pairs, summary_hashes = report.get("pairs"), report.get("pair_summary_sha256")
    if (not isinstance(pairs, dict) or set(pairs) != {"public", "hidden"}
            or not isinstance(summary_hashes, dict) or set(summary_hashes) != set(pairs)):
        raise ValueError("Both complete target-sequence audits must be present.")

    def checked_hash(path: Path, expected: Any) -> None:
        if (not isinstance(expected, str) or re.fullmatch(r"[0-9a-f]{64}", expected) is None
                or path.is_symlink() or not path.is_file() or _sha256(path) != expected):
            raise ValueError("Target audit artifact content does not match its bound hash.")

    for name, source_count, query_count in (
        ("public", expected_counts["train"], expected_counts["val"]),
        ("hidden", expected_counts["train"] + expected_counts["val"], expected_counts["hidden"]),
    ):
        pair = pairs[name]
        if not isinstance(pair, dict) or pair.get("status") != "passed":
            raise ValueError("Both complete target-sequence audits must pass.")
        counts = {
            "source_records": source_count, "query_records": query_count,
            "requested_pair_count": source_count * query_count,
            "qualifying_alignment_count": 0, "queries_with_qualifying_hits": 0,
            "exact_sequence_overlap_query_count": 0, "chain_overlap_count": 0, "pdb_overlap_count": 0,
        }
        if any(type(pair.get(key)) is not int or pair[key] != value for key, value in counts.items()):
            raise ValueError("Target audit pair counts do not prove the complete requested population passed.")
        for key in ("source_records_sha256", "query_records_sha256"):
            if not isinstance(pair.get(key), str) or re.fullmatch(r"[0-9a-f]{64}", pair[key]) is None:
                raise ValueError("Target audit pair lacks its sequence-record binding.")
        work = receipt.parent / name
        if work.is_symlink() or not work.is_dir() or {path.name for path in work.iterdir()} != AUDIT_ARTIFACTS | {"summary.json"}:
            raise ValueError("Target audit pair has an unexpected artifact directory or filenames.")
        artifacts = pair.get("artifact_sha256")
        if not isinstance(artifacts, dict) or set(artifacts) != AUDIT_ARTIFACTS:
            raise ValueError("Target audit pair does not bind the complete expected artifact set.")
        for filename, digest in artifacts.items():
            checked_hash(work / filename, digest)
        checked_hash(work / "summary.json", summary_hashes[name])
        private_summary = _read_audit_json(work / "summary.json")
        if json.dumps(private_summary, sort_keys=True) != json.dumps(pair, sort_keys=True):
            raise ValueError("Target audit pair summary disagrees with the completion receipt.")
        with (work / "alignments.m8").open() as handle:
            if any(line.strip() for line in handle):
                raise ValueError("Passed target audit unexpectedly contains returned alignments.")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build complete schema-2 metadata, generate versioned CATH public/hidden manifests and both source locks, "
            "then independently verify them and audit both target-sequence split boundaries in a fresh private directory. "
            "Requires pinned annotation sources and MMseqs2, including when using precomputed clusters. "
            "Failures retain private logs and never receive a verified completion receipt. "
            "Processed-asset and homology audits are still required before adopting a dataset."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--stage-dir", type=Path, required=True,
                        help="Fresh private output directory; in-repository paths must be under .nanofold_private/.")
    parser.add_argument("--sampling-policy", choices=("cath-v2", "cath-v3"), default="cath-v3",
                        help="cath-v3 covers every full group and keeps whole groups in one split.")
    parser.add_argument("--split-group-tsv", type=Path,
                        help="Required for cath-v3: complete homology-closed groups for the eligible universe.")
    parser.add_argument("--split-group-receipt", type=Path,
                        help="Required for cath-v3: source-bound complete homology-group audit receipt.")
    parser.add_argument("--chain-data-cache", type=Path, required=True)
    parser.add_argument("--metadata-sources-dir", type=Path, required=True)
    parser.add_argument("--metadata-source-lock", type=Path, required=True)
    parser.add_argument("--feature-exclusion-list", type=Path,
                        default=REPO_ROOT / "data/manifests/openfold_required_feature_exclusions.txt")
    parser.add_argument("--processability-exclusion-list", type=Path,
                        default=REPO_ROOT / "data/manifests/official_processability_exclusions.txt")
    parser.add_argument("--train-size", type=int, default=10000)
    parser.add_argument("--val-size", type=int, default=1000)
    parser.add_argument("--hidden-val-size", type=int, default=1000)
    parser.add_argument("--min-len", type=int, default=40)
    parser.add_argument("--max-len", type=int, default=256)
    parser.add_argument("--max-resolution", type=float, default=3.0)
    parser.add_argument("--require-monomer", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--min-seq-id", type=float, default=0.30)
    parser.add_argument("--coverage", type=float, default=0.80)
    parser.add_argument("--cluster-tsv", type=Path, default=None,
                        help="Optional complete, independently pinned cluster assignments; otherwise run MMseqs2.")
    parser.add_argument("--mmseqs-bin", default="mmseqs")
    parser.add_argument("--audit-threads", type=int, default=1,
                        help="CPU threads for the mandatory all-pair cache-target audits.")
    parser.add_argument("--hidden-split-salt-file", type=Path, default=None,
                        help="Otherwise read NANOFOLD_HIDDEN_SPLIT_SALT; the secret is never placed in a command.")
    return parser.parse_args(argv)


def stage_candidate(args: argparse.Namespace) -> Path:
    stage = args.stage_dir.resolve()
    repo = REPO_ROOT.resolve()
    private_root = repo / ".nanofold_private"
    if stage.is_relative_to(repo) and not stage.is_relative_to(private_root):
        raise ValueError("In-repository --stage-dir must be under .nanofold_private to keep hidden artifacts untracked.")
    # Validate local prerequisites before creating outputs or starting expensive clustering.
    source_hashes: dict[Path, str] = {}
    for name in ("chain_data_cache", "metadata_source_lock", "feature_exclusion_list",
                 "processability_exclusion_list", "cluster_tsv", "split_group_tsv", "split_group_receipt",
                 "hidden_split_salt_file"):
        value = getattr(args, name)
        if value is not None:
            path = value.resolve()
            if not path.is_file():
                raise ValueError(f"Missing input for --{name.replace('_', '-')}: {path}")
            setattr(args, name, path)
            source_hashes[path] = _sha256(path)
    if args.sampling_policy == "cath-v3" and (args.split_group_tsv is None or args.split_group_receipt is None):
        raise ValueError("cath-v3 requires --split-group-tsv and --split-group-receipt for the complete eligible universe.")
    if args.sampling_policy != "cath-v3" and (args.split_group_tsv is not None or args.split_group_receipt is not None):
        raise ValueError("--split-group-tsv and --split-group-receipt are only supported for cath-v3.")
    args.metadata_sources_dir = args.metadata_sources_dir.resolve()
    if not args.metadata_sources_dir.is_dir():
        raise ValueError("--metadata-sources-dir must be a directory containing pinned annotation sources.")
    if args.audit_threads < 1:
        raise ValueError("--audit-threads must be positive.")
    binary = shutil.which(args.mmseqs_bin)
    if binary is None:
        raise ValueError("MMseqs2 is required for the mandatory target audit, including with --cluster-tsv.")
    args.mmseqs_bin = str(Path(binary).resolve())
    _read_hidden_split_salt(
        salt_env="NANOFOLD_HIDDEN_SPLIT_SALT",
        salt_file=str(args.hidden_split_salt_file) if args.hidden_split_salt_file else "",
    )
    stage.mkdir(mode=0o700, parents=True, exist_ok=False)
    status_path = stage / "candidate_status.json"
    status: dict[str, object] = {
        "schema_version": 1, "sampling_policy": args.sampling_policy, "status": "running",
        "processed_assets_audited": False,
        "target_sequences_audited": False,
    }

    def write_status() -> None:
        _atomic_write(status_path, json.dumps(status, indent=2, sort_keys=True) + "\n")

    def run_step(step: str, script: str, options: list[str]) -> None:
        status["step"] = step
        write_status()
        print(f"[candidate] {step}", flush=True)
        with (stage / f"{step}.log").open("w") as log:
            subprocess.run([sys.executable, str(SCRIPT_DIR / script), *options], cwd=REPO_ROOT,
                           stdout=log, stderr=subprocess.STDOUT, check=True)
        if any(_sha256(path) != digest for path, digest in source_hashes.items()):
            raise ValueError("Candidate inputs changed between staging steps; use immutable sources and a fresh directory.")

    policy_options = ["--min-len", str(args.min_len), "--max-len", str(args.max_len),
                      "--max-resolution", str(args.max_resolution)]
    if args.require_monomer:
        policy_options.append("--require-monomer")
    metadata = stage / "structure_metadata.json"
    public_lock = stage / "public-source.lock.json"
    private_lock = stage / "private-source.lock.json"
    verification = stage / "manifest-verification.json"
    try:
        run_step("metadata", "build_structure_metadata.py", [
            "--chain-data-cache", str(args.chain_data_cache),
            "--metadata-sources-dir", str(args.metadata_sources_dir),
            "--metadata-source-lock", str(args.metadata_source_lock),
            "--feature-exclusion-list", str(args.feature_exclusion_list),
            "--processability-exclusion-list", str(args.processability_exclusion_list),
            "--metadata-out", str(metadata), "--candidate-manifest-out", str(stage / "candidates.txt"),
            *policy_options,
        ])
        cluster_options = ["--cluster-tsv", str(args.cluster_tsv)] if args.cluster_tsv else []
        group_options = (["--split-group-tsv", str(args.split_group_tsv),
                          "--split-group-receipt", str(args.split_group_receipt)] if args.split_group_tsv else [])
        salt_options = (["--hidden-split-salt-file", str(args.hidden_split_salt_file)]
                        if args.hidden_split_salt_file else [])
        run_step("manifests", "build_manifests.py", [
            "--sampling-policy", args.sampling_policy, "--chain-data-cache", str(args.chain_data_cache),
            "--structure-metadata", str(metadata), "--out-dir", str(stage / "public"),
            "--hidden-out-dir", str(stage / "hidden"), "--lock-file", str(public_lock),
            "--private-lock-file", str(private_lock), "--train-size", str(args.train_size),
            "--val-size", str(args.val_size), "--hidden-val-size", str(args.hidden_val_size),
            "--seed", str(args.seed), "--min-seq-id", str(args.min_seq_id), "--coverage", str(args.coverage),
            "--mmseqs-bin", args.mmseqs_bin, *policy_options, *cluster_options, *group_options, *salt_options,
        ])
        artifacts = [metadata, public_lock, private_lock,
                     stage / "public/train.txt", stage / "public/val.txt", stage / "public/all.txt",
                     stage / "hidden/hidden_val.txt"]
        verified_inputs = {path: _sha256(path) for path in [args.chain_data_cache, *artifacts]}
        run_step("verification", "verify_hidden_manifest.py", [
            "--chain-data-cache", str(args.chain_data_cache), "--lock-file", str(public_lock),
            "--private-lock-file", str(private_lock), "--public-train-manifest", str(stage / "public/train.txt"),
            "--public-val-manifest", str(stage / "public/val.txt"),
            "--public-all-manifest", str(stage / "public/all.txt"),
            "--hidden-manifest", str(stage / "hidden/hidden_val.txt"),
            "--json-out", str(verification), "--mmseqs-bin", args.mmseqs_bin,
        ])
        verified_report = _read_audit_json(verification)
        if verified_report.get("status") != "ok" or verified_report.get("sampling_policy") != args.sampling_policy:
            raise ValueError("Manifest verification did not produce a successful report.")
        if args.sampling_policy == "cath-v3":
            _validate_group_coverage(verified_report, _read_audit_json(public_lock), args)
            status["all_group_coverage_verified"] = True
            status["group_coverage"] = verified_report["group_coverage"]
        verified_inputs[verification] = _sha256(verification)
        target_audit = stage / "target_audit"
        audit_inputs = {
            "chain_data_cache": args.chain_data_cache,
            "manifest_train": stage / "public/train.txt", "manifest_val": stage / "public/val.txt",
            "manifest_public_all": stage / "public/all.txt", "manifest_hidden": stage / "hidden/hidden_val.txt",
            "audit_script": SCRIPT_DIR / "audit_cache_targets.py",
            "shared_parser": REPO_ROOT / "nanofold/mmseqs.py", "mmseqs_binary": Path(args.mmseqs_bin),
        }
        audit_input_hashes = {name: _sha256(path) for name, path in audit_inputs.items()}
        audit_max_seqs = max(1_000_000, args.train_size + args.val_size)
        run_step("target_audit", "audit_cache_targets.py", [
            "--chain-data-cache", str(args.chain_data_cache),
            "--train-manifest", str(stage / "public/train.txt"),
            "--val-manifest", str(stage / "public/val.txt"),
            "--public-all-manifest", str(stage / "public/all.txt"),
            "--hidden-manifest", str(stage / "hidden/hidden_val.txt"),
            "--expected-counts", str(args.train_size), str(args.val_size), str(args.hidden_val_size),
            "--out-dir", str(target_audit), "--mmseqs-bin", args.mmseqs_bin,
            "--threads", str(args.audit_threads),
            "--max-seqs", str(audit_max_seqs),
            "--min-seq-id", str(args.min_seq_id), "--coverage", str(args.coverage),
            "--min-len", str(args.min_len), "--max-len", str(args.max_len),
        ])
        audit_receipt = target_audit / "summary.json"
        if any(_sha256(path) != digest for path, digest in verified_inputs.items()):
            raise ValueError("Candidate inputs changed between manifest verification and target auditing.")
        _validate_target_audit_receipt(
            audit_receipt, inputs=audit_inputs, input_hashes=audit_input_hashes,
            expected_counts={"train": args.train_size, "val": args.val_size, "hidden": args.hidden_val_size},
            min_seq_id=args.min_seq_id, coverage=args.coverage, min_len=args.min_len, max_len=args.max_len,
            threads=args.audit_threads, max_seqs=audit_max_seqs,
        )
        status["target_sequences_audited"] = True
        status["artifact_sha256"] = {
            str(path.relative_to(stage)): _sha256(path)
            for path in [*artifacts, verification, audit_receipt]
        }
        status["status"] = "manifests_verified"
        write_status()
    except BaseException:
        status["status"] = "failed"
        write_status()
        raise
    return status_path


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    try:
        receipt = stage_candidate(args)
    except (OSError, ValueError, subprocess.CalledProcessError) as exc:
        if isinstance(exc, subprocess.CalledProcessError):
            raise SystemExit("Candidate staging failed. Inspect the private stage status/logs; use a fresh directory.") from None
        raise SystemExit(str(exc)) from None
    print(f"[candidate] manifests verified and target audits passed; private receipt: {receipt}")
    print("[candidate] processed-asset validation and strict homology audits are required before release.")


if __name__ == "__main__":
    main()
