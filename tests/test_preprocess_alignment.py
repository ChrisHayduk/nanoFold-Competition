from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np


def _load_preprocess_module():
    module_path = Path("scripts/preprocess.py").resolve()
    spec = importlib.util.spec_from_file_location("preprocess_script", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module spec for {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_ungap_query_columns_removes_query_gap_positions() -> None:
    module = _load_preprocess_module()
    fn = getattr(module, "_ungap_query_columns")

    msa = np.asarray([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=np.int32)
    deletions = np.asarray([[0, 1, 0, 0], [0, 0, 2, 0]], dtype=np.int32)

    out_msa, out_deletions, target_seq = fn(msa=msa, deletions=deletions, query_aligned="A-CD")
    assert target_seq == "ACD"
    assert out_msa.shape == (2, 3)
    assert out_deletions.shape == (2, 3)
    assert np.array_equal(out_msa[:, 0], np.array([1, 5], dtype=np.int32))
    assert np.array_equal(out_msa[:, 1], np.array([3, 7], dtype=np.int32))


def test_parse_hhr_hits_preserves_file_order_and_alignment_strings(tmp_path: Path) -> None:
    module = _load_preprocess_module()
    parse_hits = getattr(module, "_parse_hhr_hits")
    pairs_from_strings = getattr(module, "_pairs_from_aligned_strings")

    hhr = tmp_path / "toy.hhr"
    hhr.write_text(
        "\n".join(
            [
                "No 1",
                ">1ABC_A synthetic",
                "Q query             1 AC-D   3 (3)",
                "T 1ABC_A            5 ACGD   8 (8)",
                "No 2",
                ">2BCD_B synthetic",
                "Q query             1 AC-D   3 (3)",
                "T 2BCD_B            1 AC-D   3 (3)",
            ]
        )
        + "\n"
    )

    hits = parse_hits(hhr)
    assert [(hit.pdb_id, hit.chain_id) for hit in hits] == [("1abc", "A"), ("2bcd", "B")]
    assert hits[0].query_aligned == "AC-D"
    assert hits[0].template_aligned == "ACGD"

    pairs, matches = pairs_from_strings(hits[0].query_aligned, hits[0].template_aligned)
    assert pairs == [(0, 0), (1, 1), (2, 3)]
    assert matches == 3


def test_project_structure_to_query_reports_alignment_provenance() -> None:
    module = _load_preprocess_module()
    project = getattr(module, "_project_structure_to_query")

    query_seq = "ACDE"
    structure_seq = "ACDF"
    coords = np.zeros((4, 3), dtype=np.float32)
    mask = np.asarray([True, True, True, False], dtype=bool)

    out_coords, out_mask, stats = project(
        query_seq=query_seq,
        structure_seq=structure_seq,
        structure_ca_coords=coords,
        structure_ca_mask=mask,
    )

    assert out_coords.shape == (4, 3)
    assert out_mask.shape == (4,)
    assert float(stats["projection_seq_identity"]) == 1.0
    assert float(stats["projection_alignment_coverage"]) == 0.75
    assert float(stats["projection_aligned_fraction"]) == 0.75
    assert float(stats["projection_valid_ca_count"]) == 3.0
