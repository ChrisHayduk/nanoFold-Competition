from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np

from nanofold.a3m import ungap_query_columns
from nanofold.residue_constants import ATOM14_NUM_SLOTS, CA_ATOM14_SLOT


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
    msa = np.asarray([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=np.int32)
    deletions = np.asarray([[0, 1, 0, 0], [0, 0, 2, 0]], dtype=np.int32)

    out_msa, out_deletions, target_seq = ungap_query_columns(
        msa=msa,
        deletions=deletions,
        query_aligned="A-CD",
    )
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


def test_project_atom14_to_query_reports_alignment_provenance() -> None:
    module = _load_preprocess_module()
    project = getattr(module, "_project_atom14_to_query")

    query_seq = "ACDE"
    structure_seq = "ACDF"
    L_s = len(structure_seq)
    atom14_positions = np.zeros((L_s, ATOM14_NUM_SLOTS, 3), dtype=np.float32)
    atom14_mask = np.zeros((L_s, ATOM14_NUM_SLOTS), dtype=bool)
    # Mark CA present for the first 3 residues.
    atom14_mask[:3, CA_ATOM14_SLOT] = True
    # Give each CA a distinct coordinate so we can verify copying.
    atom14_positions[:3, CA_ATOM14_SLOT, 0] = np.arange(3, dtype=np.float32)

    out_positions, out_mask, stats = project(
        query_seq=query_seq,
        structure_seq=structure_seq,
        structure_atom14_positions=atom14_positions,
        structure_atom14_mask=atom14_mask,
    )

    assert out_positions.shape == (len(query_seq), ATOM14_NUM_SLOTS, 3)
    assert out_mask.shape == (len(query_seq), ATOM14_NUM_SLOTS)
    # BioPython global aligner aligns all 4 positions (ACDE vs ACDF): 3 matches, 1 mismatch.
    assert float(stats["projection_seq_identity"]) == 0.75
    assert float(stats["projection_alignment_coverage"]) == 1.0
    assert float(stats["projection_aligned_fraction"]) == 1.0
    # Only 3 residues have CA coordinates present in the structure mask.
    assert float(stats["projection_valid_ca_count"]) == 3.0
    # Check CA coordinates copied through for aligned residues.
    assert float(out_positions[0, CA_ATOM14_SLOT, 0]) == 0.0
    assert float(out_positions[2, CA_ATOM14_SLOT, 0]) == 2.0
