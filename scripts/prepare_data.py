from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

"""Download a fixed subset of OpenProteinSet/OpenFold training data.

This script is deliberately implementation-light and delegates heavy lifting to the AWS CLI
(RODA uses an anonymous S3 bucket).

You have two main options:

A) Download everything (not recommended) and then subset.
B) Download only the chain directories referenced by your manifests (recommended).

This script assumes:
- you have `aws` installed
- your manifests contain chain IDs like `7KDX_A`

The OpenFold docs describe the bucket layout and a helper script to flatten the RODA directory:
https://openfold.readthedocs.io/en/latest/OpenFold_Training_Setup.html

You will likely want to run `scripts/preprocess.py` after downloading to convert raw A3M + mmCIF into .npz.
"""


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=str, default="data/raw", help="Where to store downloaded raw data")
    ap.add_argument("--manifest", type=str, required=True, help="train.txt or val.txt listing chain IDs")
    ap.add_argument("--bucket", type=str, default="s3://openfold", help="RODA bucket for OpenProteinSet")
    ap.add_argument(
        "--include-mmcif-zip",
        action="store_true",
        help="Also download pdb_mmcif.zip (very large). Consider hosting a smaller mmCIF subset yourself.",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without running.",
    )
    return ap.parse_args()


def run(cmd: list[str], dry_run: bool) -> None:
    print("+", " ".join(cmd))
    if not dry_run:
        subprocess.check_call(cmd)


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root)
    data_root.mkdir(parents=True, exist_ok=True)

    manifest = Path(args.manifest)
    chain_ids = [ln.strip() for ln in manifest.read_text().splitlines() if ln.strip() and not ln.startswith("#")]

    # Download per-chain alignment directories from s3://openfold/pdb/<chain_id>/
    # NOTE: this assumes the bucket uses chain IDs as top-level folder names under `pdb/`.
    # If the upstream layout changes, tweak here.
    for cid in chain_ids:
        out_dir = data_root / "roda_pdb" / cid
        out_dir.parent.mkdir(parents=True, exist_ok=True)
        s3_path = f"{args.bucket}/pdb/{cid}"
        run(["aws", "s3", "cp", s3_path, str(out_dir), "--recursive", "--no-sign-request"], args.dry_run)

    if args.include_mmcif_zip:
        run(["aws", "s3", "cp", f"{args.bucket}/pdb_mmcif.zip", str(data_root), "--no-sign-request"], args.dry_run)

    print("Done. Next: run scripts/preprocess.py to build data/processed/*.npz")


if __name__ == "__main__":
    main()
