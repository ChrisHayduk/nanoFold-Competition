"""Replay frozen, successful empty raw-source listings without biological claims.

The certificate contains acquisition input pins and requested chain IDs only.
Routes and covered copies are reconstructed here; diagnostic summaries are not
certificates. This policy supports one completed strict pooled-S3 acquisition,
with its whole journal, for the default MSA-only request.
"""
from __future__ import annotations

import hashlib
import importlib
import json
from pathlib import Path
from typing import Any

from nanofold.representative_eligibility import (
    _raw_source,
    file_record,
    read_json,
    require,
    sha256_file,
    snapshot,
)

POLICY = "frozen_raw_source_unavailability_v1"
SCOPE = "frozen_successful_filtered_listing"
CODE_FILES = ("scripts/prepare_data.py", "nanofold/s3_transport.py")
PIN_FIELDS = (
    "input_bindings", "acquisition_plan", "sequence_plan", "duplicate_map", "manifest",
    "attempt_start", "attempt_result", "journal",
)


def _identity(pin: dict[str, Any]) -> dict[str, str]:
    require(set(pin) in ({"path", "sha256"}, {"path", "sha256", "bytes"}),
            "Source availability has invalid file binding fields")
    return {"path": pin["path"], "sha256": pin["sha256"]}


def _validate_pin(pin: dict[str, Any]) -> None:
    identity = _identity(pin)
    path = Path(identity["path"])
    before = snapshot(path)
    require(file_record(path) == identity and snapshot(path) == before,
            "Source availability bound file changed")
    require("bytes" not in pin or pin["bytes"] == before[2],
            "Source availability bound byte count changed")


def _acquisition(certificate: dict[str, Any], raw_roots: list[str]) -> dict[str, tuple[int, int, int, int, int]]:
    """Bind the exact request policy to a completed, single acquisition attempt."""
    snapshots: dict[str, tuple[int, int, int, int, int]] = {}

    def check(pin: dict[str, Any]) -> None:
        snapshots.setdefault(pin["path"], snapshot(Path(pin["path"])))
        _validate_pin(pin)

    for field in PIN_FIELDS:
        if field != "journal":
            check(certificate[field])
    code = certificate["code"]
    require(set(code) == set(CODE_FILES), "Source availability acquisition code pins are incomplete")
    root = Path(__file__).resolve().parents[1]
    for relative in CODE_FILES:
        check(code[relative])
        require(code[relative]["sha256"] == sha256_file(root / relative),
                "Source availability acquisition implementation differs from owning code")
    # The driver imports the transport from its own repository root.
    prepare_path = Path(code["scripts/prepare_data.py"]["path"])
    require(Path(code["nanofold/s3_transport.py"]["path"])
            == prepare_path.parents[1] / "nanofold/s3_transport.py",
            "Source availability transport does not belong to its bound driver")
    bindings = read_json(Path(certificate["input_bindings"]["path"]))
    require(bindings.get("kind") == "raw", "Source availability attempt is not raw acquisition")
    files = bindings["files"]
    require(len({pin["path"] for pin in files}) == len(files), "Acquisition repeats an input binding")
    for pin in files:
        check(pin)
    identities = [_identity(pin) for pin in files]
    for pin in [certificate[name] for name in ("acquisition_plan", "duplicate_map", "manifest")] + list(code.values()):
        require(_identity(pin) in identities, "Source availability input is outside frozen acquisition bindings")
    require(_identity(bindings["selected_manifest"]) == _identity(certificate["manifest"]),
            "Source availability manifest differs from acquisition")
    plan = read_json(Path(certificate["acquisition_plan"]["path"]))
    require(_identity(plan["sequence_plan"]) == _identity(certificate["sequence_plan"]),
            "Source availability exact-sequence plan differs from acquisition")
    raw_root = Path(certificate["raw_root"])
    require(str(raw_root.resolve()) == str(raw_root)
            and str(raw_root) in [str(Path(path).resolve()) for path in raw_roots],
            "Source availability raw root is outside qualification bindings")
    require(Path(certificate["journal"]["path"]) == raw_root / "roda_pdb/.s3-acquisition.jsonl",
            "Source availability journal is outside its raw acquisition")
    start_path = Path(certificate["attempt_start"]["path"])
    result_path = Path(certificate["attempt_result"]["path"])
    require(start_path.name == "start.json" and result_path == start_path.parent / "result.json"
            and start_path.parent.parent == Path(certificate["input_bindings"]["path"]).parent / "attempts",
            "Source availability attempt files do not share the acquisition owner")
    require(list(start_path.parent.parent.iterdir()) == [start_path.parent],
            "Source availability requires a single completed acquisition attempt")
    start = read_json(start_path)
    argv = start.get("argv", [])
    require(isinstance(argv, list) and len(argv) == 14 and isinstance(argv[0], str)
            and isinstance(argv[10], str) and argv[10].isdigit() and int(argv[10]) > 0,
            "Source availability request is not the frozen strict MSA-only command")
    expected = [argv[0], str(prepare_path), "--data-root", str(raw_root),
                "--manifest", certificate["manifest"]["path"],
                "--duplicate-chains-file", certificate["duplicate_map"]["path"],
                "--no-template-hits", "--download-workers", argv[10],
                "--strict-downloads", "--download-transport", "pooled-s3"]
    require(argv == expected, "Source availability acquisition request policy differs")
    require(start.get("source_scope") == bindings.get("scope"),
            "Source availability attempt scope differs from acquisition")
    result = read_json(result_path)
    require(result == {"exit_code": 1},
            "Source availability requires a completed strict missing-source exit")
    return snapshots


def _sequence_index(path: Path) -> dict[str, dict[str, Any]]:
    by_chain: dict[str, dict[str, Any]] = {}
    with path.open() as handle:
        for line in handle:
            require(line.endswith("\n"), "Source availability sequence plan is truncated")
            plan = json.loads(line)
            require(hashlib.sha256(plan["sequence"].encode()).hexdigest() == plan["sequence_sha256"],
                    "Source availability sequence identity differs from its plan")
            for copy in plan["ranked_copies"]:
                chain = copy["chain_id"]
                require(chain not in by_chain, "Source availability plan repeats a deposited copy")
                by_chain[chain] = plan
    return by_chain


def _empty_listings(pin: dict[str, Any], chains: set[str]) -> dict[str, list[int]]:
    """Only the pinned transport's post-paginator empty event proves absence."""
    path = Path(pin["path"])
    identity = _identity(pin)
    before = snapshot(path)
    prefixes = {f"pdb/{chain}/": chain for chain in chains}
    lines: dict[str, list[int]] = {chain: [] for chain in chains}
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for number, line in enumerate(handle, 1):
            digest.update(line)
            require(line.endswith(b"\n"), "Source availability journal is truncated")
            event = json.loads(line)
            require(isinstance(event, dict), "Source availability journal event is invalid")
            chain = prefixes.get(event.get("prefix"))
            source_chain = event.get("source_chain_id")
            if chain is None and source_chain not in chains:
                continue
            if chain is None or source_chain not in (None, chain):
                raise ValueError("Source availability journal has an inconsistent source route")
            require(event == {"status": "source_objects_pinned", "bucket": "openfold",
                              "prefix": f"pdb/{chain}/", "objects": []},
                    "Source availability route has errors, payloads, or incomplete listing evidence")
            lines[chain].append(number)
    require(snapshot(path) == before and digest.hexdigest() == identity["sha256"]
            and str(path.resolve()) == identity["path"]
            and ("bytes" not in pin or pin["bytes"] == before[2]),
            "Source availability journal differs from its frozen snapshot")
    require(all(lines.values()), "Source availability has not exhausted every original-map route")
    return lines


def validate_source_availability(
    path: Path, sequence_plans: list[dict[str, Any]], raw_roots: list[str],
) -> dict[str, dict[str, Any]]:
    """Return exact copy-specific records justified by a frozen source certificate.

    Validation never writes files or contacts the source. A certificate is a
    statement about a completed filtered listing, not perpetual upstream absence.
    Missing proof remains an acquisition request in the qualification owner.
    """
    certificate_pin = file_record(path)
    certificate = read_json(path)
    require(set(certificate) == {"policy", "status", "requested_chain_ids", "raw_root", "code", *PIN_FIELDS}
            and certificate["policy"] == POLICY and certificate["status"] == "complete",
            "Unsupported source availability certificate")
    snapshots = _acquisition(certificate, raw_roots)
    manifest = [line.strip() for line in Path(certificate["manifest"]["path"]).read_text().splitlines()
                if line.strip() and not line.lstrip().startswith("#")]
    requested = certificate["requested_chain_ids"]
    require(isinstance(requested, list) and bool(requested) and len(set(requested)) == len(requested)
            and len(set(manifest)) == len(manifest) and set(requested) <= set(manifest),
            "Source availability requests are outside the frozen acquisition manifest")
    bindings = read_json(Path(certificate["input_bindings"]["path"]))
    require(bindings["selected_count"] == len(manifest), "Source availability manifest count changed")
    source_plans = _sequence_index(Path(certificate["sequence_plan"]["path"]))
    prepare = importlib.import_module("scripts.prepare_data")
    require(prepare.__file__ is not None
            and sha256_file(Path(prepare.__file__)) == certificate["code"]["scripts/prepare_data.py"]["sha256"],
            "Source availability imported acquisition owner differs from frozen code")
    duplicates = prepare._load_duplicate_map(certificate["duplicate_map"]["path"])
    routes: dict[str, list[str]] = {}
    covered: dict[str, str] = {}
    for chain in requested:
        require(chain in source_plans and source_plans[chain]["ranked_copies"][0]["chain_id"] == chain,
                "Source availability request differs from its frozen first copy")
        candidates = prepare._dedupe_keep_order([chain, *duplicates.get(chain, [])])
        for candidate in candidates:
            require(candidate in source_plans
                    and source_plans[candidate]["sequence_sha256"] == source_plans[chain]["sequence_sha256"],
                    "Source availability original-map routes differ from exact-sequence membership")
            routes[candidate] = prepare._dedupe_keep_order([candidate, *duplicates.get(candidate, [])])
            covered[candidate] = source_plans[chain]["sequence_sha256"]
    all_routes = {chain for candidates in routes.values() for chain in candidates}
    snapshots[certificate["journal"]["path"]] = snapshot(Path(certificate["journal"]["path"]))
    listings = _empty_listings(certificate["journal"], all_routes)
    for chain in sorted(all_routes):
        try:
            _raw_source([Path(root).resolve() for root in raw_roots], chain)
        except ValueError as exc:
            if str(exc) != "Required raw query source is missing; acquisition is incomplete":
                raise
        else:
            raise ValueError("Source availability is contradicted by a currently available raw payload")
    result: dict[str, dict[str, Any]] = {}
    for plan in sequence_plans:
        for copy in plan["ranked_copies"]:
            chain = copy["chain_id"]
            if chain not in covered:
                continue
            original = source_plans[chain]
            require(plan["sequence_sha256"] == covered[chain] and plan["sequence"] == original["sequence"],
                    "Source availability copy differs from current exact-sequence membership")
            original_copy = next(candidate for candidate in original["ranked_copies"] if candidate["chain_id"] == chain)
            # Diversity groups are rederived after final heldout provenance exclusion.
            def metadata(row: dict[str, Any]) -> dict[str, Any]:
                return {key: value for key, value in row.items() if key not in {"group_id", "cluster40_id"}}
            require(metadata(copy) == metadata(original_copy),
                    "Source availability deposited-copy metadata differs from acquisition")
            require(chain not in result, "Qualification plan repeats a source-unavailable copy")
            result[chain] = {"status": "source_unavailable", "scope": SCOPE, "chain_id": chain,
                             "sequence_sha256": plan["sequence_sha256"], "certificate": certificate_pin,
                             "source_routes": [{"source_chain_id": source, "bucket": "openfold",
                                                "prefix": f"pdb/{source}/", "listing_lines": listings[source]}
                                               for source in routes[chain]]}
    require(all(snapshot(Path(name)) == before for name, before in snapshots.items()),
            "Source availability acquisition evidence changed during validation")
    require(file_record(path) == certificate_pin, "Source availability certificate changed during validation")
    return result
