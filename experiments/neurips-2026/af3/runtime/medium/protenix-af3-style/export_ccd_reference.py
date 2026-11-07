#!/usr/bin/env python3
"""Export the pinned Protenix 20-residue CCD table as a versioned JSON artifact."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

from adapter import ccd_reference_sha256, load_protenix_ccd_reference
from config import PROTENIX_COMMIT


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    table = load_protenix_ccd_reference()
    payload = {
        "schema_version": 2,
        "protenix_commit": PROTENIX_COMMIT,
        "ccd_reference_sha256": ccd_reference_sha256(table),
        "residues": list(table),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{args.output.name}.", suffix=".tmp", dir=args.output.parent
    )
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary_name, args.output)
    print(json.dumps({key: value for key, value in payload.items() if key != "residues"}))


if __name__ == "__main__":
    main()
