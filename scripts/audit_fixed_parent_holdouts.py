"""Audit both directional homology boundaries between the fixed evaluation sets."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nanofold.full_parent_leakage import file_record, sequence_id, validate_parent_frame
from nanofold.mmseqs import iter_batched_mmseqs_hits


def audit(receipt_path: Path, out_path: Path) -> dict[str, Any]:
    receipt_pin = file_record(receipt_path)
    code_pin = file_record(Path(__file__))
    frame = validate_parent_frame(receipt_path)
    receipt = json.loads(receipt_path.read_text())
    frozen = frame["frozen_splits"]
    by_id = {row["chain_id"]: row for row in frame["records"]}
    split_by_query = {chain: split for split, ids in frozen.items() for chain in ids}
    target_by_sequence: dict[str, tuple[str, str]] = {}
    for split, ids in frozen.items():
        for chain in ids:
            key = sequence_id(by_id[chain]["sequence"])
            if key in target_by_sequence:
                raise ValueError("Frozen evaluation targets repeat an exact sequence")
            target_by_sequence[key] = (split, chain)
    hits = []
    for query, target in iter_batched_mmseqs_hits(receipt["barrier"], min_seq_id=0.3, coverage=0.8, coverage_mode=2):
        target_record = target_by_sequence.get(target)
        if target_record is None or target_record[0] == split_by_query[query]:
            continue
        hits.append({"query_split": split_by_query[query], "target_split": target_record[0],
                     "query": query, "target": target_record[1],
                     "query_length": by_id[query]["length"], "target_length": by_id[target_record[1]]["length"]})
    hits.sort(key=lambda row: (row["query_split"], row["query"], row["target"]))
    directions = {}
    for query_split, target_split in (("hidden_val", "val"), ("val", "hidden_val")):
        selected = [row for row in hits if row["query_split"] == query_split]
        directions[f"{query_split}_query_to_{target_split}_target"] = {
            "qualifying_pairs": len(selected), "unique_queries": len({r["query"] for r in selected}),
            "unique_targets": len({r["target"] for r in selected}),
        }
    result = {
        "status": "complete_conflicts_detected" if hits else "complete_zero_cross_split_hits",
        "separation_passed": not hits,
        "scope": "Both directional full-target query-covered local-alignment contracts between the unchanged evaluation sets. No evaluation membership was changed.",
        "alignment_contract": {"min_seq_id": 0.3, "coverage": 0.8, "coverage_mode": 2},
        "inputs": {"full_parent_frame_receipt": receipt_pin, "auditor": code_pin},
        "verified_barrier_batches": receipt["barrier"]["batch_count"],
        "frozen_split_counts": {split: len(ids) for split, ids in frozen.items()},
        "directions": directions, "hits": hits,
        "unique_chains_by_split": dict(Counter(split for split, ids in frozen.items()
                                                for chain in ids if any(chain in (r["query"], r["target"]) for r in hits))),
    }
    if file_record(receipt_path) != receipt_pin or file_record(Path(__file__)) != code_pin:
        raise ValueError("Audit inputs changed during evaluation")
    if out_path.exists():
        raise ValueError("Refusing to replace an existing evaluation audit")
    payload = json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n"
    with out_path.open("x") as handle:
        handle.write(payload)
    print(json.dumps({"status": result["status"], "directions": directions,
                      "report": str(out_path), "sha256": hashlib.sha256(payload.encode()).hexdigest()}), flush=True)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-frame-receipt", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    audit(args.full_frame_receipt, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
