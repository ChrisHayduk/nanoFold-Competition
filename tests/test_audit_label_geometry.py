from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from nanofold.chain_paths import chain_npz_path
from scripts import audit_label_geometry as audit

CHAIN = "0001_A"


def fixture(tmp_path: Path) -> tuple[list[str], dict[str, Any], dict[str, Any]]:
    length = 40
    target = np.zeros(length, np.int32)
    features: dict[str, Any] = {
        "aatype": target, "msa": target[None], "deletions": np.zeros((1, length), np.int32),
        "chain_id": np.asarray(CHAIN), "residue_index": np.arange(length, dtype=np.int32),
        "between_segment_residues": np.zeros(length, np.int32),
        "template_aatype": np.zeros((0, length), np.int32),
        "template_ca_coords": np.zeros((0, length, 3), np.float32), "template_ca_mask": np.zeros((0, length), bool),
    }
    xyz = np.zeros((length, 14, 3), np.float32)
    xyz[:, :5, 0] = np.arange(length)[:, None] * 3.8
    xyz[:, 0, 0] -= 1.25
    xyz[:, 2, 0] += 1.22
    mask = np.zeros((length, 14), bool)
    mask[:, :5] = True
    labels: dict[str, Any] = {
        "atom14_positions": xyz, "atom14_mask": mask, "ca_coords": xyz[:, 1].copy(), "ca_mask": mask[:, 1].copy(),
        "chain_id": np.asarray(CHAIN), "residue_index": features["residue_index"].copy(),
    }
    for root, values in (("features", features), ("labels", labels)):
        (tmp_path / root).mkdir()
        np.savez(chain_npz_path(tmp_path / root, CHAIN), **values)
    (tmp_path / "cache.json").write_text(json.dumps({CHAIN: {"seq": "A" * length}}))
    (tmp_path / "manifest.txt").write_text(CHAIN + "\n")
    (tmp_path / "groups.tsv").write_text(f"{CHAIN}\t{CHAIN}\n")
    return (["--chain-data-cache", str(tmp_path / "cache.json"), "--split", "train", str(tmp_path / "manifest.txt"),
             str(tmp_path / "features"), str(tmp_path / "labels"), "1", "--split-group-tsv", str(tmp_path / "groups.tsv"),
             "--out-dir", str(tmp_path / "audit")], features, labels)


def save(tmp_path: Path, features: dict[str, Any], labels: dict[str, Any]) -> None:
    for root, values in (("features", features), ("labels", labels)):
        np.savez(chain_npz_path(tmp_path / root, CHAIN), **values)


def report(tmp_path: Path) -> dict[str, Any]:
    return json.loads((tmp_path / "audit/summary.json").read_text())


def test_complete_geometry_receipt_binds_labels_and_reports_split_group_counts(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    args, _, _ = fixture(tmp_path)
    assert audit.main(args) == 0
    result = report(tmp_path)
    assert result["distance_outliers_are_diagnostic_only"] is True
    assert result["split_summaries"]["train"]["ca_checked_pairs"] == 39
    assert result["split_summaries"]["train"]["peptide_checked_pairs"] == 39
    assert result["split_summaries"]["train"]["ca_outlier_count"] == 0
    assert len(result["group_summaries"]) == 1
    for row in result["records"]:
        for key in ("feature", "label"):
            assert audit.sha256(Path(row[key]["path"])) == row[key]["sha256"]
    assert CHAIN not in capsys.readouterr().out
    assert (tmp_path / "audit/summary.json").stat().st_mode & 0o077 == 0
    original = (tmp_path / "audit/summary.json").read_bytes()
    assert audit.main(args) == 2
    assert (tmp_path / "audit/summary.json").read_bytes() == original


@pytest.mark.parametrize("kind", ["atom_support", "nonfinite", "wrong_chain", "wrong_ca", "wrong_mask", "wrong_length",
                                  "object", "index_mismatch", "invalid_break", "missing_index", "repeated_index", "overflow_index"])
def test_invalid_support_identity_schema_or_projection_is_a_hard_error(tmp_path: Path, kind: str) -> None:
    args, features, labels = fixture(tmp_path)
    if kind == "atom_support":
        labels["atom14_mask"][0, 5] = True  # Alanine has no atom in this slot.
    elif kind == "nonfinite":
        labels["atom14_positions"][0, 13] = float("nan")  # Invalid even though masked.
    elif kind == "wrong_chain":
        labels["chain_id"] = np.asarray("0002_A")
    elif kind == "wrong_ca":
        labels["ca_coords"][0, 0] += 1
    elif kind == "wrong_mask":
        labels["ca_mask"][0] = False
    elif kind == "wrong_length":
        labels["ca_coords"] = labels["ca_coords"][:-1]
    elif kind == "object":
        labels["unused"] = np.asarray({"unsafe": 1}, dtype=object)
    elif kind == "index_mismatch":
        labels["residue_index"][1] += 2
    elif kind == "invalid_break":
        features["between_segment_residues"][10] = 3
    elif kind == "missing_index":
        del labels["residue_index"]
    elif kind == "overflow_index":
        indices = np.asarray([np.iinfo(np.int64).max, *range(np.iinfo(np.int64).min, np.iinfo(np.int64).min + 39)], dtype=np.int64)
        features["residue_index"] = indices
        labels["residue_index"] = indices.copy()
    else:
        features["residue_index"][1] = labels["residue_index"][1] = 0
    save(tmp_path, features, labels)
    assert audit.main(args) == 1
    assert report(tmp_path)["failure_count"] == 1


def test_experimental_distance_outliers_are_reported_without_rejecting_target(tmp_path: Path) -> None:
    args, features, labels = fixture(tmp_path)
    labels["atom14_positions"][5, 1, 0] += 100
    labels["ca_coords"] = labels["atom14_positions"][:, 1].copy()
    labels["atom14_positions"][7, 2, 0] += 10
    save(tmp_path, features, labels)
    assert audit.main(args) == 0
    result = report(tmp_path)
    assert result["split_summaries"]["train"]["ca_outlier_count"] == 2
    assert result["split_summaries"]["train"]["peptide_outlier_count"] == 1
    assert len(result["records"][0]["label"]["distances"]["ca"]["outliers"]) == 2


def test_missing_atoms_breaks_and_discontinuous_indices_remove_only_uninterpretable_pairs(tmp_path: Path) -> None:
    args, features, labels = fixture(tmp_path)
    features["between_segment_residues"][10] = 1
    features["residue_index"][20:] += 5
    labels["residue_index"] = features["residue_index"].copy()
    labels["atom14_mask"][2, 1] = False
    labels["ca_mask"][2] = False
    labels["atom14_mask"][1, 0] = False
    save(tmp_path, features, labels)
    assert audit.main(args) == 0
    diagnostics = report(tmp_path)["records"][0]["label"]
    assert diagnostics["excluded_nonadjacent_or_break_pairs"] == 2
    assert diagnostics["distances"]["ca"]["checked_pairs"] == 35
    assert diagnostics["distances"]["peptide"]["checked_pairs"] == 36


@pytest.mark.parametrize("mode", ["missing", "escaped", "alias_object", "duplicate"])
def test_missing_escaped_or_ambiguous_label_archives_fail(tmp_path: Path, mode: str) -> None:
    args, _, _ = fixture(tmp_path)
    path = chain_npz_path(tmp_path / "labels", CHAIN)
    if mode == "missing":
        path.unlink()
    elif mode == "escaped":
        outside = tmp_path / "outside.npz"
        path.rename(outside)
        path.symlink_to(outside)
    else:
        value = io.BytesIO()
        np.save(value, np.asarray({"bad": 1}, dtype=object))
        with zipfile.ZipFile(path, "a") as archive:
            if mode == "duplicate":
                with pytest.warns(UserWarning, match="Duplicate name"):
                    archive.writestr("atom14_mask.npy", value.getvalue())
            else:
                archive.writestr("atom14_mask.npy.npy", value.getvalue())
    assert audit.main(args) == 1


@pytest.mark.parametrize("mode", ["wrong_count", "missing_group", "duplicate_member", "overlap", "wrong_cache"])
def test_complete_selected_population_and_cache_bindings_are_required(tmp_path: Path, mode: str) -> None:
    args, _, _ = fixture(tmp_path)
    if mode == "wrong_count":
        args[args.index("--split") + 5] = "2"
    elif mode == "missing_group":
        (tmp_path / "groups.tsv").write_text("0002_A\t0002_A\n")
    elif mode == "duplicate_member":
        with (tmp_path / "groups.tsv").open("a") as handle:
            handle.write(f"{CHAIN}\t{CHAIN}\n")
    elif mode == "overlap":
        args += ["--split", "val", str(tmp_path / "manifest.txt"), str(tmp_path / "features"), str(tmp_path / "labels"), "1"]
    else:
        (tmp_path / "cache.json").write_text(json.dumps({CHAIN: {"seq": "C" * 40}}))
    assert audit.main(args) in {1, 2}
    assert report(tmp_path)["status"] != "passed"


def test_label_mutation_after_read_cannot_receive_success(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    args, _, _ = fixture(tmp_path)
    original = audit.label_diagnostics

    def mutate(feature_path: Path, label_path: Path, **kwargs: Any) -> dict[str, Any]:
        result = original(feature_path, label_path, **kwargs)
        with label_path.open("ab") as handle:
            handle.write(b"changed")
        return result

    monkeypatch.setattr(audit, "label_diagnostics", mutate)
    assert audit.main(args) == 2
    assert report(tmp_path)["status"] == "error"


def test_changed_and_restored_feature_cannot_validate_wrong_residue_atom_support(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    args, features, labels = fixture(tmp_path)
    labels["atom14_mask"][0, 5] = True
    save(tmp_path, features, labels)
    original = audit.label_diagnostics

    def swap_feature(feature_path: Path, label_path: Path, **kwargs: Any) -> dict[str, Any]:
        pinned_bytes = feature_path.read_bytes()
        changed_features = dict(features)
        changed_features["aatype"] = features["aatype"].copy()
        changed_features["aatype"][0] = 1  # Arg supports slot 5; the actual pinned Ala does not.
        try:
            np.savez(feature_path, **changed_features)
            return original(feature_path, label_path, **kwargs)
        finally:
            feature_path.write_bytes(pinned_bytes)

    monkeypatch.setattr(audit, "label_diagnostics", swap_feature)
    assert audit.main(args) == 1
    assert report(tmp_path)["status"] == "failed"
