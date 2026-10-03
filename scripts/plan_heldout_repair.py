"""Freeze private repair rankings and source requests before choosing new evaluation targets.

The proposal is metadata-only and does not modify a manifest. A provisional
selection requires actual source qualification and complete hidden-safety
proofs; a separate additive full-parent barrier must still accept it.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nanofold.full_parent_leakage import file_record
from nanofold.heldout_repair_selection import (
    POLICY,
    REQUEST_POLICY,
    _qualification_outcomes,
    _safety_states,
    build_plan,
    build_safety_plan,
    optimistic_proposal,
    replay_selection,
    selection_payload,
    validate_plan,
    validate_selection_receipt,
)
from nanofold.structural_population import read_rows, write_json


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    freeze = subparsers.add_parser("freeze", help="Write a new immutable candidate ranking and initial source request")
    freeze.add_argument("--baseline-frame-receipt", type=Path, required=True)
    freeze.add_argument("--complete-feasibility", type=Path, required=True)
    freeze.add_argument("--structural-metadata", type=Path, required=True)
    freeze.add_argument("--structural-metadata-receipt", type=Path, required=True)
    freeze.add_argument("--metadata-source-lock", type=Path, required=True)
    freeze.add_argument("--output", type=Path, required=True)
    freeze.add_argument("--seed", type=int, default=0)
    safety = subparsers.add_parser("safety-plan", help="Write exact reverse candidate-query/hidden search inputs; do not run a search")
    safety.add_argument("--request", type=Path, required=True)
    safety.add_argument("--output", type=Path, required=True)
    accept = subparsers.add_parser("select", help="Replay actual source and safety proofs; accept only a complete ranked selection")
    accept.add_argument("--plan", type=Path, required=True)
    accept.add_argument("--qualification-receipt", type=Path, action="append", default=[])
    accept.add_argument("--safety-proof", type=Path, action="append", default=[], help="JSON object with pinned native safety plan and result")
    accept.add_argument("--output", type=Path, required=True)
    verify = subparsers.add_parser("verify")
    verify.add_argument("--selection-receipt", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "freeze":
        path = build_plan(baseline_frame_receipt=args.baseline_frame_receipt, complete_feasibility=args.complete_feasibility,
                          structural_metadata=args.structural_metadata, structural_metadata_receipt=args.structural_metadata_receipt,
                          metadata_source_lock=args.metadata_source_lock, output=args.output, seed=args.seed)
        plan = json.loads(path.read_text())
        sequences = read_rows(Path(plan["sequence_plan"]["path"]))
        proposal = optimistic_proposal(plan, sequences)
        write_json(args.output / "proposal.json", {"plan": file_record(path), **proposal})
        request = {"schema_version": 1, "policy": REQUEST_POLICY, "plan": file_record(path),
                   "sequence_plan": plan["sequence_plan"], "requested_sequence_sha256": proposal["requested_sequence_sha256"]}
        write_json(args.output / "request.json", request)
        print(json.dumps({"status": proposal["status"], "plan": file_record(path), "requested_sequences": len(request["requested_sequence_sha256"])}))
    elif args.command == "safety-plan":
        request = json.loads(args.request.read_text())
        path = build_safety_plan(Path(request["plan"]["path"]), request["requested_sequence_sha256"], args.output)
        print(json.dumps({"status": "search_inputs_only", "plan": file_record(path)}))
    elif args.command == "select":
        plan, sequences, frame = validate_plan(args.plan)
        plan_pin = file_record(args.plan)
        source_pins = [file_record(path) for path in args.qualification_receipt]
        safety_pins = [json.loads(path.read_text()) for path in args.safety_proof]
        outcomes = _qualification_outcomes(source_pins, plan_pin, plan, sequences, verify_sources=True)
        safety_states = _safety_states(safety_pins, plan_pin, plan, sequences, frame)
        replay = replay_selection(plan, sequences, outcomes, safety_states)
        args.output.mkdir(parents=True, exist_ok=True)
        if replay["status"] != "complete":
            write_json(args.output / "pending-selection.json", {"plan": plan_pin, **replay})
            proposal = optimistic_proposal(plan, sequences, known_outcomes=outcomes, known_safety=safety_states)
            write_json(args.output / "proposal.json", {"plan": plan_pin, **proposal})
            if proposal["requested_sequence_sha256"]:
                write_json(args.output / "request.json", {"schema_version": 1, "policy": REQUEST_POLICY, "plan": plan_pin,
                           "sequence_plan": plan["sequence_plan"], "requested_sequence_sha256": proposal["requested_sequence_sha256"]})
            if proposal["safety_sequence_sha256"]:
                write_json(args.output / "safety-request.json", {"schema_version": 1, "policy": REQUEST_POLICY, "plan": plan_pin,
                           "sequence_plan": plan["sequence_plan"], "requested_sequence_sha256": proposal["safety_sequence_sha256"]})
            print(json.dumps({"status": replay["status"], "selected_count": len(replay["selected"]),
                              "pending_count": len(replay["pending_sequence_sha256"]), "reason": replay.get("pending_reason")}))
            return 2
        payload = selection_payload(plan, replay, frame)
        receipt = {"schema_version": 1, "policy": POLICY, "status": "provisional_source_and_hidden_safety_verified",
                   "plan": plan_pin, "qualification_receipts": source_pins, "safety_receipts": safety_pins,
                   "selection": replay, "source_bindings": plan["source_bindings"], **payload}
        pending = args.output / "selection.receipt.incomplete.json"
        write_json(pending, receipt)
        validate_selection_receipt(pending, verify_sources=True)
        path = args.output / "selection.receipt.json"
        write_json(path, receipt)
        print(json.dumps({"status": receipt["status"], "receipt": file_record(path), "added_count": len(payload["added_public_ids"]),
                          "capacity_after_provenance_only": payload["capacity_after_provenance_only"]}))
    else:
        result = validate_selection_receipt(args.selection_receipt, verify_sources=True)
        print(json.dumps({"status": "verified_provisional_selection", "added_count": len(result["added_public_ids"]), "receipt": result["receipt"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
