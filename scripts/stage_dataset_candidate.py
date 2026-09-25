"""Build and verify a private CATH candidate without changing a published release."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from build_manifests import _atomic_write, _read_hidden_split_salt, _sha256


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build complete schema-2 metadata, generate cath-v2 public/hidden manifests and both source locks, "
            "then independently verify them in a fresh private directory. Requires pinned annotation sources. "
            "Failures retain private logs and never receive a verified completion receipt. "
            "Processed-asset and homology audits are still required before adopting a dataset."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--stage-dir", type=Path, required=True,
                        help="Fresh private output directory; in-repository paths must be under .nanofold_private/.")
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
    for name in ("chain_data_cache", "metadata_source_lock", "feature_exclusion_list",
                 "processability_exclusion_list", "cluster_tsv", "hidden_split_salt_file"):
        value = getattr(args, name)
        if value is not None:
            path = value.resolve()
            if not path.is_file():
                raise ValueError(f"Missing input for --{name.replace('_', '-')}: {path}")
            setattr(args, name, path)
    args.metadata_sources_dir = args.metadata_sources_dir.resolve()
    if not args.metadata_sources_dir.is_dir():
        raise ValueError("--metadata-sources-dir must be a directory containing pinned annotation sources.")
    _read_hidden_split_salt(
        salt_env="NANOFOLD_HIDDEN_SPLIT_SALT",
        salt_file=str(args.hidden_split_salt_file) if args.hidden_split_salt_file else "",
    )
    stage.mkdir(mode=0o700, parents=True, exist_ok=False)
    status_path = stage / "candidate_status.json"
    status: dict[str, object] = {
        "schema_version": 1, "sampling_policy": "cath-v2", "status": "running",
        "processed_assets_audited": False,
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
        salt_options = (["--hidden-split-salt-file", str(args.hidden_split_salt_file)]
                        if args.hidden_split_salt_file else [])
        run_step("manifests", "build_manifests.py", [
            "--sampling-policy", "cath-v2", "--chain-data-cache", str(args.chain_data_cache),
            "--structure-metadata", str(metadata), "--out-dir", str(stage / "public"),
            "--hidden-out-dir", str(stage / "hidden"), "--lock-file", str(public_lock),
            "--private-lock-file", str(private_lock), "--train-size", str(args.train_size),
            "--val-size", str(args.val_size), "--hidden-val-size", str(args.hidden_val_size),
            "--seed", str(args.seed), "--min-seq-id", str(args.min_seq_id), "--coverage", str(args.coverage),
            "--mmseqs-bin", args.mmseqs_bin, *policy_options, *cluster_options, *salt_options,
        ])
        run_step("verification", "verify_hidden_manifest.py", [
            "--chain-data-cache", str(args.chain_data_cache), "--lock-file", str(public_lock),
            "--private-lock-file", str(private_lock), "--public-train-manifest", str(stage / "public/train.txt"),
            "--public-val-manifest", str(stage / "public/val.txt"),
            "--public-all-manifest", str(stage / "public/all.txt"),
            "--hidden-manifest", str(stage / "hidden/hidden_val.txt"),
            "--json-out", str(verification), "--mmseqs-bin", args.mmseqs_bin,
        ])
        if json.loads(verification.read_text()).get("status") != "ok":
            raise ValueError("Manifest verification did not produce a successful report.")
        status["artifact_sha256"] = {
            str(path.relative_to(stage)): _sha256(path)
            for path in (metadata, public_lock, private_lock, verification,
                         stage / "public/train.txt", stage / "public/val.txt", stage / "public/all.txt",
                         stage / "hidden/hidden_val.txt")
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
    print(f"[candidate] manifests verified; private receipt: {receipt}")
    print("[candidate] processed-asset validation and strict homology audits are required before release.")


if __name__ == "__main__":
    main()
