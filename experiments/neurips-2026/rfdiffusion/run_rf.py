#!/usr/bin/env python3
"""Launch the published NanoFold RFdiffusion-style training and evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_LOCK = ROOT / "data-lock.json"
LARGE = ROOT / "large/architecture-reconstructions/rfdiffusion-style"
SMALL = ROOT / "small/architecture-scaling/rfdiffusion-style"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_provenance() -> dict:
    provenance = json.loads((ROOT / "PROVENANCE.json").read_text())
    for relative, entry in provenance["exact_sources"].items():
        if sha256(ROOT / relative) != entry["sha256"]:
            raise ValueError(f"Released RF source differs from its recorded hash: {relative}")
    if json.loads(DATA_LOCK.read_text()) != provenance["dataset"]:
        raise ValueError("Dataset verification lock differs from the released provenance")
    return provenance


def verify_data(data_root: Path) -> None:
    sys.path.insert(0, str(LARGE.parent / "common"))
    from verify_dataset import verify_dataset_snapshot

    report = json.loads(DATA_LOCK.read_text())
    print(json.dumps(verify_dataset_snapshot(data_root, report), indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    verification = commands.add_parser("verify-data", help="Check the complete public tensor tree")
    verification.add_argument("--data-root", type=Path, required=True)
    for action in ("train", "evaluate"):
        command = commands.add_parser(action)
        command.add_argument("profile", choices=("tiny", "medium", "large"))
        command.add_argument("--data-root", type=Path, required=True)
        command.add_argument("--dry-run", action="store_true")
        if action == "train":
            command.add_argument("--run-dir", type=Path, required=True)
        else:
            command.add_argument("--checkpoint", type=Path, required=True)
            command.add_argument("--scorer-root", type=Path, required=True)
            command.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    provenance = read_provenance()
    if args.command == "verify-data":
        verify_data(args.data_root)
        return

    profile = provenance["profiles"][args.profile]
    large = args.profile == "large"
    if args.command == "train":
        ranks = profile["physical_training_ranks"]
        script = LARGE / "train.py" if large else SMALL / "runtime/train_accumulated.py"
        config = ROOT / profile["training_config"]
        extra = ["--run-dir", str(args.run_dir.resolve())]
        if not large:
            extra += ["--physical-world-size", str(ranks)]
    else:
        ranks = 8 if large else 4
        script = LARGE / "evaluate.py" if large else SMALL / "runtime/evaluate_accumulated.py"
        config = LARGE / "configs/eval-final.json" if large else SMALL / "runtime/eval-world4.json"
        extra = [
            "--checkpoint", str(args.checkpoint.resolve()),
            "--nanofold-source-root", str(args.scorer_root.resolve()),
            "--output-dir", str(args.output_dir.resolve()),
        ]
        if not args.dry_run:
            for file in ("metrics.py", "residue_constants.py"):
                path = args.scorer_root / "nanofold" / file
                if sha256(path) != provenance["scorer"][path.stem + "_sha256"]:
                    raise ValueError(f"Use the recorded NanoFold scorer file: {file}")

    argv = [
        sys.executable, "-m", "torch.distributed.run", "--standalone",
        f"--nproc_per_node={ranks}", str(script),
        "--config", str(config),
        "--data-root", str(args.data_root.resolve()),
        "--dataset-verification", str(DATA_LOCK),
        *extra,
    ]
    print(shlex.join(argv), flush=True)
    if not args.dry_run:
        subprocess.run(argv, check=True)


if __name__ == "__main__":
    main()
