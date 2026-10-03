from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from nanofold import heldout_sources as sources
from nanofold.structural_population import qualification_predicate
from scripts.audit_target_identity import raw_record


def pin(path: Path) -> dict[str, Any]:
    return {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "bytes": path.stat().st_size}


def put_json(path: Path, value: Any) -> dict[str, Any]:
    path.write_text(json.dumps(value, sort_keys=True) + "\n")
    return pin(path)


def put_rows(path: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    return pin(path)


@pytest.fixture(scope="module")
def baseline_case(tmp_path_factory):
    root = tmp_path_factory.mktemp("heldout-source-fixture")
    structure = root / "structure.cif"
    structure.write_text("immutable synthetic deposited bytes\n")
    structure_pin = pin(structure)
    raw_root = root / "raw"
    raw_root.mkdir()
    geometry = {"mapping_status": "verified", "valid_ca_count": 40}

    def row(chain: str, index: int, split: str, *, need_raw: bool) -> dict[str, Any]:
        sequence = "".join("ARNDCQEGHILKMFPSTWYV"[(index // 20**position) % 20] for position in range(5)) + "A" * 35
        sequence_sha = hashlib.sha256(sequence.encode()).hexdigest()
        path = raw_root / (chain + ".a3m")
        if need_raw:
            path.write_text(f">query\n{sequence}\n>support\n{sequence[:-1]}-\n")
            audit = raw_record(path, sequence)
        else:
            audit = {"path": str(path), "sha256": "a" * 64, "bytes": 100, "query_length": 40,
                     "rows": 2, "aligned_width": 40}
        raw = {key: audit[key] for key in ("path", "sha256", "bytes", "query_length")}
        raw.update(query_sha256=sequence_sha, scope="first_query_identity_only")
        attempt = {"chain_id": chain, "sequence_sha256": sequence_sha, "target_length": 40, "rank": 0,
                   "structure": structure_pin, "raw": raw, "raw_full_audit": audit,
                   "status": "passed", "structure_status": "passed", "raw_status": "passed",
                   "source_geometry": geometry, "crop_geometry": geometry}
        return {"chain_id": chain, "sequence": sequence, "sequence_sha256": sequence_sha, "length": 40,
                "split": split, "group_id": chain, "status": "passed", "qualification": attempt,
                "source_geometry": geometry, "crop_geometry": geometry}

    train = [row(f"t{index:04d}_A", index, "train", need_raw=index == 0) for index in range(10000)]
    public = [row(f"v{index:04d}_A", index + 10000, "val", need_raw=True) for index in range(1000)]
    hidden = [row(f"h{index:04d}_A", index + 11000, "hidden", need_raw=True) for index in range(1000)]
    public = public[::2] + public[1::2]
    hidden.reverse()
    # A selected repair may have belonged to the old train cohort. Its new
    # qualification rank must be preserved; it is excluded from new training.
    repairs = [copy.deepcopy(train[0])] + [row(f"r{index:04d}_A", index + 12000, "val", need_raw=True) for index in range(213)]
    for repair in repairs:
        repair["qualification"]["rank"] = 2
    prior_train = put_rows(root / "prior-train.jsonl", train)
    prior_heldout = put_rows(root / "prior-heldout.jsonl", public + hidden)
    final = []
    for original in train + public + hidden:
        final.append(original | {"original_source_bindings": {key: original["qualification"][key] for key in ("structure", "raw")}})
    manifests = {}
    for split, rows in (("train", train), ("val", public), ("hidden", hidden)):
        path = root / (split + ".txt")
        path.write_text("".join(row["chain_id"] + "\n" for row in rows))
        manifests[split] = pin(path)
    original_pin = put_json(root / "original-replay.json", {"status": "complete", "inputs": manifests})
    prior_pin = put_json(root / "prior-receipt.json", {"status": "complete",
        "outputs": {"train-qualified": prior_train, "heldout-chains": prior_heldout},
        "structure_query_geometry_replay": original_pin})
    inputs = {"train": prior_train, "heldout": prior_heldout, "input_receipt": prior_pin,
              "original_replay_receipt": original_pin,
              **{split + "_manifest": value for split, value in manifests.items()}}
    predicate = qualification_predicate()
    binding_pin = put_json(root / "bindings.json", {"inputs": inputs, "predicate": predicate})
    baseline = {"schema_version": 1, "policy": sources.BASELINE_POLICY, "status": "complete",
                "counts": {"train": {"passed": 10000}, "val": {"passed": 1000}, "hidden": {"passed": 1000}},
                "inputs": inputs, "predicate": predicate, "binding": binding_pin,
                "outputs": {"train-qualified": put_rows(root / "train-qualified.jsonl", final[:10000]),
                            "heldout-chains": put_rows(root / "heldout-chains.jsonl", final[10000:])}}
    baseline_path = root / "baseline-replay.json"
    put_json(baseline_path, baseline)
    active = [row["chain_id"] for row in public[:786] + repairs]
    active_path = root / "active.txt"
    active_path.write_text("".join(chain + "\n" for chain in active))
    selection_path = root / "selection.json"
    put_json(selection_path, {"status": "selected"})
    lock_pin = put_json(root / "public-lock.json", {"outputs": {"train_manifest_sha256": manifests["train"]["sha256"]}})
    frame_path = root / "frame.json"
    frame = {"policy": sources.FRAME_POLICY,
             "records": [{key: value for key, value in row.items() if key not in
                         {"qualification", "source_geometry", "crop_geometry", "status", "split"}} | {"group_id": None}
                         for row in public + hidden + repairs],
             "frozen_splits": {"val": active, "hidden_val": [row["chain_id"] for row in hidden]},
             "evaluation_cohorts": {"historical_public": [row["chain_id"] for row in public],
                                    "active_public": active, "hidden": [row["chain_id"] for row in hidden]},
             "source_bindings": {"frozen_val": pin(active_path), "frozen_hidden": manifests["hidden"],
                                 "historical_public": manifests["val"], "baseline_public_lock": lock_pin,
                                 "selection_receipt": pin(selection_path)},
             "selected_qualifications": {row["chain_id"]: row["qualification"] for row in repairs}}
    put_json(frame_path, frame)
    return {"root": root, "baseline": baseline_path, "frame": frame_path, "public": public, "final": final,
            "structure": structure}


@pytest.fixture
def exported(baseline_case, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    # Native homology/repair selection has its own integration coverage. This
    # boundary stands in for its verified reconstructed frame only; the actual
    # final baseline replay validator runs against genuine 3-field file pins.
    monkeypatch.setattr(sources, "validate_parent_frame", lambda path: json.loads(path.read_text()))
    out = tmp_path / "export"
    sources.export_heldout_sources(baseline_case["baseline"], baseline_case["frame"], out)
    return baseline_case, out / "receipt.json"


def test_export_preserves_exact_originals_and_selected_repair_qualifications(exported) -> None:
    case, path = exported
    receipt = sources.validate_heldout_sources(path)
    rows = sources.read_rows(Path(receipt["heldout_chains"]["path"]))
    assert len(rows) == 2000
    assert rows[:786] == case["final"][10000:10786]
    assert rows[1000:] == case["final"][11000:]
    assert rows[786]["chain_id"] == "t0000_A" and rows[786]["qualification"]["rank"] == 2
    assert receipt["counts"] == sources.COUNTS
    assert Path(receipt["public_manifest"]["path"]).stat().st_mode & 0o077 == 0
    with pytest.raises(ValueError, match="fresh output"):
        sources.export_heldout_sources(case["baseline"], case["frame"], path.parent)


@pytest.mark.parametrize("mutation", ["sequence", "source", "geometry", "retired_target", "duplicate"])
def test_updated_export_hash_cannot_authorize_changed_rows(exported, mutation: str) -> None:
    case, path = exported
    receipt = json.loads(path.read_text())
    rows_path = Path(receipt["heldout_chains"]["path"])
    rows = sources.read_rows(rows_path)
    if mutation == "sequence":
        rows[0]["sequence"] = "R" * 40
    elif mutation == "source":
        rows[0]["qualification"]["raw"]["sha256"] = "f" * 64
    elif mutation == "geometry":
        rows[0]["source_geometry"]["valid_ca_count"] = 39
    elif mutation == "retired_target":
        rows[0] = case["final"][10999]
    else:
        rows[-1] = rows[0]
    receipt["heldout_chains"] = put_rows(rows_path, rows)
    put_json(path, receipt)
    with pytest.raises(ValueError, match="rows differ"):
        sources.validate_heldout_sources(path, verify_sources=False)


def test_updated_manifest_hash_cannot_authorize_old_public_cohort(exported) -> None:
    case, path = exported
    receipt = json.loads(path.read_text())
    manifest = Path(receipt["public_manifest"]["path"])
    manifest.write_bytes((case["root"] / "val.txt").read_bytes())
    receipt["public_manifest"] = pin(manifest)
    put_json(path, receipt)
    with pytest.raises(ValueError, match="manifest differs"):
        sources.validate_heldout_sources(path, verify_sources=False)


def test_baseline_must_still_include_retired_original_public_targets(
    baseline_case, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sources, "validate_parent_frame", lambda path: json.loads(path.read_text()))
    baseline = json.loads(baseline_case["baseline"].read_text())
    rows = sources.read_rows(Path(baseline["outputs"]["heldout-chains"]["path"]))
    rows[999] = rows[0]  # Remove a retired target while retaining the declared row count.
    baseline["outputs"]["heldout-chains"] = put_rows(tmp_path / "changed-heldout.jsonl", rows)
    changed = tmp_path / "changed-baseline.json"
    put_json(changed, baseline)
    with pytest.raises(ValueError, match="repeats or omits original"):
        sources.export_heldout_sources(changed, baseline_case["frame"], tmp_path / "export")
    assert not (tmp_path / "export").exists()


def test_selected_repair_qualification_must_match_its_exact_frame_target(
    baseline_case, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = json.loads(baseline_case["frame"].read_text())
    frame["selected_qualifications"]["t0000_A"]["chain_id"] = "wrong_A"
    monkeypatch.setattr(sources, "validate_parent_frame", lambda _: frame)
    with pytest.raises(ValueError, match="exact passing target"):
        sources.export_heldout_sources(baseline_case["baseline"], baseline_case["frame"], tmp_path / "export")
    assert not (tmp_path / "export").exists()


@pytest.mark.parametrize("kind", ["structure", "raw"])
def test_changed_consumed_source_is_rejected(exported, kind: str) -> None:
    case, receipt_path = exported
    path = case["structure"] if kind == "structure" else Path(case["public"][0]["qualification"]["raw"]["path"])
    before = path.read_bytes()
    try:
        path.write_bytes(before + b"\n")
        with pytest.raises(ValueError, match="source changed|complete MSA audit"):
            sources.validate_heldout_sources(receipt_path, verify_sources=True)
    finally:
        path.write_bytes(before)


def test_private_destination_is_checked_before_any_recipe_read(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    monkeypatch.setattr(sources, "ROOT", root)
    with pytest.raises(ValueError, match="tracked repository"):
        sources.export_heldout_sources(Path("absent"), Path("absent"), root / "public-data")
    outside = tmp_path / "outside"
    outside.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        sources.export_heldout_sources(Path("absent"), Path("absent"), alias)
    (outside / "repo-alias").symlink_to(root, target_is_directory=True)
    with pytest.raises(ValueError, match="tracked repository"):
        sources.export_heldout_sources(Path("absent"), Path("absent"), outside / "repo-alias" / "public-data")


def test_manifest_change_between_validation_and_copy_cannot_publish_receipt(
    baseline_case, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sources, "validate_parent_frame", lambda path: json.loads(path.read_text()))
    manifest = baseline_case["root"] / "active.txt"
    original = manifest.read_bytes()
    output = tmp_path / "export"
    read_bytes = Path.read_bytes

    def changed_read(path: Path) -> bytes:
        if path == manifest and output.exists():
            return original + b"# changed during source read\n"
        return read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", changed_read)
    with pytest.raises(ValueError, match="manifest changed during export copying"):
        sources.export_heldout_sources(baseline_case["baseline"], baseline_case["frame"], output)
    assert not (output / "receipt.json").exists()
