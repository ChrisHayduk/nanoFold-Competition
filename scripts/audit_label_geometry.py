"""Audit residue-specific label support and report masked backbone geometry.

Invalid identities, schemas and chemically nonexistent atom14 support fail the
gate. Bond-distance outliers are diagnostics, never automatic exclusions.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import zipfile
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
for import_root in (REPO_ROOT, Path(__file__).resolve().parent):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from audit_cache_targets import (
    _reject_json_constant,
    _unique_json_object,
    read_manifest,
    sequences_for,
    sha256,
    write_json,
)
from audit_target_identity import AUDIT_ERRORS, find_processed, processed_record, snapshot

from nanofold.dataset_integrity import validate_feature_label_pair, validate_label_npz_schema
from nanofold.residue_constants import STANDARD_ATOM14_MASK
from nanofold.structure_metadata import validate_author_chain_id


def label_diagnostics(feature_path: Path, label_path: Path, *, chain: str,
                      ca_bounds: tuple[float, float], peptide_bounds: tuple[float, float]) -> dict[str, Any]:
    before = snapshot(label_path)
    digest = sha256(label_path)
    with zipfile.ZipFile(label_path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)) or any(
            not name.endswith(".npy") or Path(name).name != name or name[:-4] in names for name in names
        ):
            raise ValueError("Label archive contains duplicate or invalid members")
    with np.load(label_path, allow_pickle=False) as labels:
        for name in labels.files:
            value = labels[name]
            if not isinstance(value, np.ndarray) or value.dtype.hasobject:
                raise ValueError("Labels contain unsafe or non-array members")
            if np.issubdtype(value.dtype, np.number) and not np.isfinite(value).all():
                raise ValueError("Labels contain nonfinite values")
        if "chain_id" not in labels or labels["chain_id"].shape != () or labels["chain_id"].dtype.kind != "U":
            raise ValueError("Labels require a scalar Unicode chain identity")
        if labels["chain_id"].item() != chain:
            raise ValueError("Label chain identity differs from the selected target")
        if validate_label_npz_schema(label_path) or validate_feature_label_pair(
            feature_path, label_path, expected_chain_id=chain, require_chain_id=True,
        ):
            raise ValueError("Labels violate their schema or paired feature identity")
        positions, mask = labels["atom14_positions"], labels["atom14_mask"]
        with np.load(feature_path, allow_pickle=False) as features:
            aatype = features["aatype"]
            if np.any(mask & ~STANDARD_ATOM14_MASK[aatype].astype(bool)):
                raise ValueError("Observed atom14 mask claims atoms absent from the target residue type")
            if "residue_index" not in features or "residue_index" not in labels:
                raise ValueError("Geometry requires explicit paired residue indices")
            residue_index = features["residue_index"]
            breaks = features["between_segment_residues"] if "between_segment_residues" in features else np.zeros_like(aatype)
            if breaks.shape != aatype.shape or breaks.dtype.kind not in {"i", "u"} or np.any((breaks != 0) & (breaks != 1)):
                raise ValueError("Segment-break flags must be binary integers with target length")
            if residue_index.shape != aatype.shape or residue_index.dtype.kind not in {"i", "u"}:
                raise ValueError("Residue indices must be integers with target length")
            if np.any(residue_index[1:] <= residue_index[:-1]):
                raise ValueError("Residue indices must increase strictly")
            adjacent = (np.diff(residue_index.astype(np.int64)) == 1) & (breaks[1:] == 0)
        diagnostics: dict[str, Any] = {}
        for name, source_slot, target_slot, bounds in (
            ("ca", 1, 1, ca_bounds), ("peptide", 2, 0, peptide_bounds),
        ):
            valid = adjacent & mask[:-1, source_slot] & mask[1:, target_slot]
            indices = np.flatnonzero(valid)
            distances = np.linalg.norm(
                positions[:-1, source_slot].astype(np.float64) - positions[1:, target_slot].astype(np.float64), axis=-1,
            )[valid]
            unusual = (distances < bounds[0]) | (distances > bounds[1])
            diagnostics[name] = {
                "checked_pairs": len(indices), "outlier_count": int(unusual.sum()),
                "min_distance": float(distances.min()) if len(distances) else None,
                "max_distance": float(distances.max()) if len(distances) else None,
                "outliers": [{"left_residue_index": int(residue_index[index]),
                              "right_residue_index": int(residue_index[index + 1]), "distance": float(distance)}
                             for index, distance, flag in zip(indices, distances, unusual, strict=True) if flag],
            }
        result = {"path": str(label_path), "sha256": digest, "bytes": before[2],
                  "residues": len(aatype), "observed_atoms": int(mask.sum()), "observed_ca": int(mask[:, 1].sum()),
                  "excluded_nonadjacent_or_break_pairs": int((~adjacent).sum()), "distances": diagnostics}
    if snapshot(label_path) != before:
        raise ValueError("Label file changed while being audited")
    return result


def _groups(path: Path, selected: set[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for line in path.read_text().splitlines():
        columns = line.split("\t")
        if len(columns) != 2:
            raise ValueError("Split-group map requires exactly two tab-separated columns")
        group, member = columns
        validate_author_chain_id(group)
        validate_author_chain_id(member)
        if member in result:
            raise ValueError("Split-group map repeats a member")
        result[member] = group
    if not selected <= result.keys():
        raise ValueError("Split-group map omits selected targets")
    return result


def _totals(records: list[dict[str, Any]]) -> dict[str, int]:
    return {"targets": len(records), "observed_atoms": sum(row["label"]["observed_atoms"] for row in records),
            "observed_ca": sum(row["label"]["observed_ca"] for row in records),
            **{f"{kind}_{field}": sum(row["label"]["distances"][kind][field] for row in records)
               for kind in ("ca", "peptide") for field in ("checked_pairs", "outlier_count")}}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chain-data-cache", type=Path, required=True)
    parser.add_argument("--split", nargs=5, action="append", required=True,
                        metavar=("NAME", "MANIFEST", "FEATURES", "LABELS", "COUNT"))
    parser.add_argument("--split-group-tsv", type=Path, help="Optional verified group/member TSV for group summaries.")
    parser.add_argument("--ca-distance-range", type=float, nargs=2, default=(2.5, 4.5), metavar=("MIN", "MAX"),
                        help="Angstrom diagnostic bounds only; outliers never fail the gate.")
    parser.add_argument("--peptide-distance-range", type=float, nargs=2, default=(1.0, 1.8), metavar=("MIN", "MAX"))
    parser.add_argument("--min-len", type=int, default=40)
    parser.add_argument("--max-len", type=int, default=256)
    parser.add_argument("--out-dir", type=Path, required=True, help="Fresh private directory for all detail and receipts.")
    return parser.parse_args(argv)


def audit_labels(args: argparse.Namespace) -> int:
    out = args.out_dir.resolve()
    if args.out_dir.is_symlink() or (out.is_relative_to(REPO_ROOT) and not out.is_relative_to(REPO_ROOT / ".nanofold_private")):
        raise ValueError("Geometry audit outputs must remain in a fresh private directory")
    if args.min_len < 1 or args.max_len < args.min_len:
        raise ValueError("Invalid target length bounds")
    for bounds in (args.ca_distance_range, args.peptide_distance_range):
        if not np.isfinite(bounds).all() or not 0 < bounds[0] < bounds[1]:
            raise ValueError("Distance bounds must be positive, finite and increasing")
    inputs = {"cache": args.chain_data_cache.resolve(), "audit_script": Path(__file__).resolve()}
    for relative in ("scripts/audit_target_identity.py", "scripts/audit_cache_targets.py", "nanofold/dataset_integrity.py",
                     "nanofold/residue_constants.py", "nanofold/structure_metadata.py", "nanofold/chain_paths.py",
                     "nanofold/a3m.py", "nanofold/data.py", "nanofold/utils.py"):
        inputs[relative] = REPO_ROOT / relative
    splits: dict[str, dict[str, Any]] = {}
    for name, manifest, features, labels, count in args.split:
        if name in splits or not name or not name.replace("_", "").replace("-", "").isalnum() or int(count) < 1:
            raise ValueError("Split names must be unique simple names and expected counts positive")
        roots = [Path(features).resolve(), Path(labels).resolve()]
        if roots[0] == roots[1] or any(
            not root.is_dir() or out.is_relative_to(root) or root.is_relative_to(out) for root in roots
        ):
            raise ValueError("Separate feature/label roots must exist outside the output directory")
        splits[name] = {"features": roots[0], "labels": roots[1], "count": int(count)}
        inputs[f"manifest_{name}"] = Path(manifest).resolve()
    if args.split_group_tsv:
        inputs["groups"] = args.split_group_tsv.resolve()
    if any(not path.is_file() or path.is_relative_to(out) for path in inputs.values()):
        raise ValueError("Audit inputs must exist outside the output directory")
    old_umask = os.umask(0o077)
    try:
        out.mkdir(mode=0o700, parents=True, exist_ok=False)
        receipt: dict[str, Any] = {"schema_version": 1, "status": "running", "records": [], "failures": [],
                                  "distance_outliers_are_diagnostic_only": True,
                                  "distance_ranges_angstrom": {"ca": args.ca_distance_range, "peptide": args.peptide_distance_range},
                                  "expected_counts": {name: split["count"] for name, split in splits.items()},
                                  "sequence_contract": {"min_len": args.min_len, "max_len": args.max_len}}
        write_json(out / "summary.json", receipt)
        try:
            receipt["inputs"] = {name: {"path": str(path), "sha256": sha256(path)} for name, path in inputs.items()}
            cache = json.loads(inputs["cache"].read_text(), object_pairs_hook=_unique_json_object, parse_constant=_reject_json_constant)
            if not isinstance(cache, dict):
                raise ValueError("Chain cache must be an object")
            selected: set[str] = set()
            for name, split in splits.items():
                chains = read_manifest(inputs[f"manifest_{name}"])
                if len(chains) != split["count"] or selected & set(chains):
                    raise ValueError("Split counts differ from requirements or split targets overlap")
                selected.update(chains)
                split["chains"] = chains
            groups = _groups(inputs["groups"], selected) if "groups" in inputs else {}
            group_owners: dict[str, str] = {}
            for name, split in splits.items():
                for index, chain in enumerate(split["chains"]):
                    group = groups.get(chain)
                    if group is not None:
                        if group in group_owners and group_owners[group] != name:
                            raise ValueError("A complete group crosses selected splits")
                        group_owners[group] = name
                    row: dict[str, Any] = {"split": name, "row_index": index,
                                          "chain_id_sha256": hashlib.sha256(chain.encode()).hexdigest(),
                                          "group_sha256": hashlib.sha256(group.encode()).hexdigest() if group else None}
                    try:
                        sequence = sequences_for([chain], cache, min_len=args.min_len, max_len=args.max_len)[0]
                        feature_path = find_processed([split["features"]], chain)
                        label_path = find_processed([split["labels"]], chain)
                        feature_state = snapshot(feature_path)
                        row["feature"] = processed_record(feature_path, sequence, chain)
                        row["label"] = label_diagnostics(feature_path, label_path, chain=chain,
                                                         ca_bounds=tuple(args.ca_distance_range),
                                                         peptide_bounds=tuple(args.peptide_distance_range))
                        if snapshot(feature_path) != feature_state:
                            raise ValueError("Feature identity changed between target validation and label diagnostics")
                        row["feature"]["snapshot"] = list(feature_state)
                        receipt["records"].append(row)
                    except AUDIT_ERRORS as exc:
                        receipt["failures"].append({**row, "reason": str(exc), "error_type": type(exc).__name__})
            for row in receipt["records"]:
                split = splits[row["split"]]
                chain = split["chains"][row["row_index"]]
                for key, root_key in (("feature", "features"), ("label", "labels")):
                    path = find_processed([split[root_key]], chain)
                    if (str(path) != row[key]["path"] or sha256(path) != row[key]["sha256"]
                            or (key == "feature" and list(snapshot(path)) != row[key]["snapshot"])):
                        raise ValueError("An audited asset changed before the geometry gate completed")
            if any(sha256(path) != receipt["inputs"][name]["sha256"] for name, path in inputs.items()):
                raise ValueError("An audit source changed before the geometry gate completed")
            receipt["status"] = "failed" if receipt["failures"] else "passed"
        except AUDIT_ERRORS as exc:
            receipt["status"] = "error"
            receipt["failures"].append({"reason": str(exc), "error_type": type(exc).__name__})
        receipt["split_summaries"] = {name: _totals([row for row in receipt["records"] if row["split"] == name]) for name in splits}
        grouped: dict[str, list[dict[str, Any]]] = {}
        for row in receipt["records"]:
            if row["group_sha256"]:
                grouped.setdefault(row["group_sha256"], []).append(row)
        receipt["group_summaries"] = {key: {"split": rows[0]["split"], **_totals(rows)} for key, rows in sorted(grouped.items())}
        receipt["passed_chain_count"] = len(receipt["records"])
        receipt["failure_count"] = len(receipt["failures"])
        receipt["record_inventory_sha256"] = hashlib.sha256(
            json.dumps(receipt["records"], sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        ).hexdigest()
        write_json(out / "summary.json", receipt)
        print(f"Label geometry audit: {receipt['status']}; passed={receipt['passed_chain_count']}; failures={receipt['failure_count']}")
        return 0 if receipt["status"] == "passed" else (1 if receipt["status"] == "failed" else 2)
    finally:
        os.umask(old_umask)


def main(argv: list[str] | None = None) -> int:
    try:
        return audit_labels(parse_args(argv))
    except AUDIT_ERRORS:
        print("Label geometry audit: preflight error; no success receipt was issued", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
