"""Bind directed train and public-validation MSA filters to the frozen split trio."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from nanofold.msa_filter_policy import build_msa_filter_policy, validate_msa_filter_policy_output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--val-manifest", type=Path, required=True)
    parser.add_argument("--hidden-manifest", type=Path, required=True)
    parser.add_argument("--train-msa-row-filter", type=Path, required=True,
                        help="Complete exhaustive train-MSA filter against public and hidden targets.")
    parser.add_argument("--val-msa-row-filter", type=Path, required=True,
                        help="Complete exhaustive public-validation-MSA filter against hidden targets.")
    parser.add_argument("--output", type=Path, required=True, help="Private policy JSON for public preprocessing.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = build_msa_filter_policy(
        train_manifest=args.train_manifest,
        val_manifest=args.val_manifest,
        hidden_manifest=args.hidden_manifest,
        train_filter=args.train_msa_row_filter,
        val_filter=args.val_msa_row_filter,
    )
    output = validate_msa_filter_policy_output(args.output, payload)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=output.parent, prefix=f".{output.name}.", delete=False) as handle:
        temporary = Path(handle.name)
        try:
            json.dump(payload, handle, sort_keys=True, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    try:
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"Wrote directed public MSA filter policy to {output}")


if __name__ == "__main__":
    main()
