from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from nanofold.representative_eligibility import file_record
from nanofold.structural_population import read_rows
from scripts import build_structural_population_plan as builder
from scripts.stage_structural_candidate import _rank as stage_rank


def _copy(chain: str, resolution: float | None, *, sequence: str = "A" * 40,
          date: str = "2020-01-01", excluded: bool = False) -> dict[str, Any]:
    return {"chain_id": chain, "sequence": sequence, "length": len(sequence), "resolution": resolution,
            "release_date": date, "group_id": None if excluded else "training-group",
            "cluster40_id": "cluster", "training_exclusion_reasons": ["heldout_query_homology"] if excluded else []}


def _frame(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    receipt = tmp_path / "frame.json"
    receipt.write_text("frozen full-parent validator fixture\n")
    rows = [_copy("1abc_A", 1.0, excluded=True), _copy("2abc_A", 2.0, date="2021-01-01"),
            _copy("3abc_A", 2.0, date="2019-01-01"), _copy("4abc_A", None),
            _copy("5abc_A", 1.5, sequence="C" * 400)]
    frame = {"records": rows, "eligible_training_ids": [row["chain_id"] for row in rows[1:]],
             "source_bindings": {}, "receipt": file_record(receipt)}
    monkeypatch.setattr(builder, "validate_parent_frame", lambda path: frame)
    return receipt


def test_plan_partitions_all_admitted_copies_and_preserves_quality_rank(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    receipt = _frame(tmp_path, monkeypatch)
    plan = builder.build(receipt, tmp_path / "population")
    sequences = read_rows(Path(plan["sequence_plan"]["path"]))
    copies = next(row["ranked_copies"] for row in sequences if row["sequence"] == "A" * 40)
    assert [row["chain_id"] for row in copies] == ["3abc_A", "2abc_A", "4abc_A"]
    assert copies[-1]["resolution"] == 999.0
    assert all(set(row) == set(builder.COPY_FIELDS) for sequence in sequences for row in sequence["ranked_copies"])
    assert all(builder.copy_rank(row) == stage_rank(row) for sequence in sequences for row in sequence["ranked_copies"])
    assert plan["source_chain_count"] == 4 and plan["sequence_count"] == 2
    assert any(len(row["sequence"]) == 400 for row in sequences)
    assert plan["seed"] == 0 and plan["require_all_groups"] is False
    assert file_record(receipt) in plan["inputs"]


def test_changed_frame_is_not_published_as_a_complete_plan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    receipt = _frame(tmp_path, monkeypatch)
    write_rows = builder.write_rows

    def changed_receipt(path: Path, rows: list[dict[str, Any]]) -> None:
        write_rows(path, rows)
        receipt.write_text("changed frame\n")

    monkeypatch.setattr(builder, "write_rows", changed_receipt)
    with pytest.raises(ValueError, match="Bound source file changed"):
        builder.build(receipt, tmp_path / "population")
    assert not (tmp_path / "population/plan.json").exists()
