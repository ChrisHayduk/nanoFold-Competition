"""Export verified active public and unchanged hidden source rows into a fresh private directory."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nanofold.heldout_sources import export_heldout_sources


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-replay-receipt", type=Path, required=True,
                        help="Final fixed-baseline replay receipt with all original 12,000 targets")
    parser.add_argument("--full-frame-receipt", type=Path, required=True,
                        help="Verified revised frame binding the 214 source-qualified public replacements")
    parser.add_argument("--output", type=Path, required=True, help="Fresh private output directory; existing paths fail")
    args = parser.parse_args()
    receipt = export_heldout_sources(args.baseline_replay_receipt, args.full_frame_receipt, args.output)
    print(json.dumps({"status": receipt["status"], "counts": receipt["counts"],
                      "receipt": str((args.output / "receipt.json").resolve())}, sort_keys=True))


if __name__ == "__main__":
    main()
