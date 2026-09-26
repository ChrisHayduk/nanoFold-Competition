"""Prepare public-only private-experiment metadata for Modal; never upload or launch."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nanofold.modal_training import prepare_bundle


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, action="append", required=True)
    parser.add_argument("--fingerprint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="Fresh ignored metadata bundle directory.")
    args = parser.parse_args()
    result = prepare_bundle(repo=Path(__file__).resolve().parents[1], configs=args.config,
                            fingerprint=args.fingerprint, output=args.output)
    print(f"Prepared {len(result['runs'])} public-only training configs; no upload or launch performed.")


if __name__ == "__main__":
    main()
