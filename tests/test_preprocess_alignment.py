from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

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
    assert hits[0].aligned_pairs == ((0, 4), (1, 5), (2, 7))

    pairs, matches = pairs_from_strings(hits[0].query_aligned, hits[0].template_aligned)
    assert pairs == [(0, 0), (1, 1), (2, 3)]
    assert matches == 3


def test_read_merged_msa_deduplicates_multiple_sources(tmp_path: Path) -> None:
    module = _load_preprocess_module()
    read_merged_msa = getattr(module, "_read_merged_msa")

    chain_dir = tmp_path / "1abc_A"
    a3m_dir = chain_dir / "a3m"
    a3m_dir.mkdir(parents=True)
    (a3m_dir / "uniref90_hits.a3m").write_text(
        ">query\nACD\n"
        ">dup\nACD\n"
        ">unique1\nA-D\n"
    )
    (a3m_dir / "mgnify_hits.a3m").write_text(
        ">query\nACD\n"
        ">dup-again\nACD\n"
        ">unique2\n-CD\n"
    )

    msa, deletions, query_sequence, filter_stats = read_merged_msa(
        chain_dir,
        msa_name="uniref90_hits.a3m",
        msa_names="uniref90_hits.a3m,mgnify_hits.a3m",
        max_msa_seqs=8,
    )

    assert query_sequence == "ACD"
    assert msa.shape == (3, 3)
    assert deletions.shape == (3, 3)
    assert filter_stats.input_rows == 6
    assert filter_stats.removed_rows == 0


def test_read_merged_msa_filters_non_query_rows_before_depth_cap(tmp_path: Path) -> None:
    module = _load_preprocess_module()
    read_merged_msa = getattr(module, "_read_merged_msa")
    row_filter_cls = getattr(module, "MSARowFilter")
    sequence_sha256 = getattr(module, "_sequence_sha256")

    chain_dir = tmp_path / "1abc_A"
    a3m_dir = chain_dir / "a3m"
    a3m_dir.mkdir(parents=True)
    (a3m_dir / "uniref90_hits.a3m").write_text(
        ">query\nACD\n"
        ">forbidden\nAAA\n"
        ">allowed\nAC-\n"
    )
    filter_path = tmp_path / "filter.json"
    filter_path.write_text("{}\n")
    row_filter = row_filter_cls(
        path=filter_path,
        sha256="0" * 64,
        excluded_sequence_sha256=frozenset({sequence_sha256("AAA"), sequence_sha256("ACD")}),
    )

    msa, deletions, query_sequence, filter_stats = read_merged_msa(
        chain_dir,
        msa_name="uniref90_hits.a3m",
        msa_names="",
        max_msa_seqs=2,
        msa_row_filter=row_filter,
    )

    assert query_sequence == "ACD"
    assert msa.shape == (2, 3)
    assert deletions.shape == (2, 3)
    assert filter_stats.input_rows == 3
    assert filter_stats.removed_rows == 1
    # Query row is preserved even though its sequence hash is in the filter.
    assert np.array_equal(msa[0], np.array([0, 4, 3], dtype=np.int32))
    # The forbidden row was removed before depth capping, so the next valid row backfills it.
    assert np.array_equal(msa[1], np.array([0, 4, 21], dtype=np.int32))


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


def test_projection_masks_mismatched_residue_including_sidechain_slots() -> None:
    module = _load_preprocess_module()
    positions = np.arange(4 * 14 * 3, dtype=np.float32).reshape(4, 14, 3)
    mask = np.ones((4, 14), dtype=bool)
    projected, projected_mask, stats = module._project_atom14_to_query("ACDE", "ACDF", positions, mask)

    # A phenylalanine ring must never supervise a glutamate carboxylate slot.
    assert not projected_mask[3].any()
    assert not projected[3].any()
    np.testing.assert_array_equal(projected[:3], positions[:3])
    assert stats["projection_seq_identity"] == 0.75
    assert stats["projection_valid_ca_count"] == 3


def test_projection_uses_polymer_positions_for_unresolved_repeated_residues() -> None:
    module = _load_preprocess_module()
    positions = np.zeros((9, 14, 3), dtype=np.float32)
    observed_indices = np.array([0, 1, 2, 3, 5, 6, 7, 8, 9], dtype=np.int32)
    positions[:, CA_ATOM14_SLOT, 0] = observed_indices
    mask = np.zeros((9, 14), dtype=bool)
    mask[:, CA_ATOM14_SLOT] = True

    projected, projected_mask, stats = module._project_atom14_to_query(
        "M" + "A" * 10, "A" * 9, positions, mask,
        polymer_sequence="A" * 10, polymer_residue_index=observed_indices,
    )
    # The query's leading M is absent from the deposited polymer. The missing
    # internal alanine remains missing instead of shifting a repeated segment.
    assert np.flatnonzero(~projected_mask[:, CA_ATOM14_SLOT]).tolist() == [0, 5]
    np.testing.assert_array_equal(projected[observed_indices + 1, CA_ATOM14_SLOT, 0], observed_indices)
    assert stats["projection_valid_ca_count"] == 9


def test_projection_rejects_ambiguous_observed_sequence_without_polymer_context() -> None:
    module = _load_preprocess_module()
    with pytest.raises(ValueError, match="Ambiguous sequence alignment"):
        module._project_atom14_to_query(
            "AAAAAAAAAA", "AAAAAAAAA", np.zeros((9, 14, 3)), np.ones((9, 14), dtype=bool),
        )


def test_projection_rejects_ambiguous_query_mapping_even_with_polymer_context() -> None:
    module = _load_preprocess_module()
    with pytest.raises(ValueError, match="Ambiguous sequence alignment"):
        module._project_atom14_to_query(
            "AAAAAAAAA", "AAAAAAAAA", np.zeros((9, 14, 3)), np.ones((9, 14), dtype=bool),
            polymer_sequence="AAAAAAAAAA", polymer_residue_index=np.arange(9),
        )


def test_projection_rejects_residue_identity_at_wrong_polymer_position() -> None:
    module = _load_preprocess_module()
    with pytest.raises(ValueError, match="identity disagrees"):
        module._project_atom14_to_query(
            "ACD", "CAD", np.zeros((3, 14, 3)), np.ones((3, 14), dtype=bool),
            polymer_sequence="ACD", polymer_residue_index=np.arange(3),
        )


@pytest.mark.parametrize("indices", [[0, 0, 2], [0, -1, 2], [0, 1, 3]])
def test_projection_rejects_invalid_deposited_coordinate_mapping(indices: list[int]) -> None:
    module = _load_preprocess_module()
    with pytest.raises(ValueError, match="label_seq_id"):
        module._project_atom14_to_query(
            "AAA", "AAA", np.zeros((3, 14, 3)), np.ones((3, 14), dtype=bool),
            polymer_sequence="AAA", polymer_residue_index=np.asarray(indices),
        )


def test_preprocess_preserves_unresolved_polymer_position_with_author_insertion_codes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import gemmi

    from nanofold.chain_paths import chain_data_dir, chain_npz_path

    module = _load_preprocess_module()
    structure = gemmi.Structure()
    model = gemmi.Model(1)
    chain = gemmi.Chain("A")
    for index in range(10):
        if index == 4:
            continue
        residue = gemmi.Residue()
        residue.name = "ALA"
        residue.seqid = gemmi.SeqId(100 + index // 2, "A" if index % 2 else " ")
        residue.label_seq = index + 1
        residue.entity_type = gemmi.EntityType.Polymer
        atom = gemmi.Atom()
        atom.name = "CA"
        atom.element = gemmi.Element("C")
        atom.pos = gemmi.Position(index * 3.8, 0, 0)
        atom.occ = 1
        residue.add_atom(atom)
        chain.add_residue(residue)
    model.add_chain(chain)
    structure.add_model(model)
    structure.setup_entities()
    structure.get_entity_of(structure[0][0].get_polymer()).full_sequence = ["ALA"] * 10
    mmcif_root = tmp_path / "mmcif"
    mmcif_root.mkdir()
    structure.make_mmcif_document().write_file(str(mmcif_root / "1abc.cif"))
    alignments = tmp_path / "alignments"
    chain_dir = chain_data_dir(alignments, "1abc_A")
    chain_dir.mkdir(parents=True)
    (chain_dir / "uniref90_hits.a3m").write_text(">query\nAAAAAAAAAA\n")
    manifest = tmp_path / "manifest.txt"
    manifest.write_text("1abc_A\n")
    features, labels = tmp_path / "features", tmp_path / "labels"
    monkeypatch.setattr(sys, "argv", [
        "preprocess.py", "--manifest", str(manifest), "--alignments-root", str(alignments),
        "--mmcif-root", str(mmcif_root), "--processed-features-dir", str(features),
        "--processed-labels-dir", str(labels), "--disable-templates", "--fail-fast",
        "--min-projection-valid-ca", "3",
    ])
    module.main()
    with np.load(chain_npz_path(labels, "1abc_A")) as data:
        assert np.flatnonzero(~data["ca_mask"]).tolist() == [4]
        observed = np.array([0, 1, 2, 3, 5, 6, 7, 8, 9])
        np.testing.assert_allclose(data["ca_coords"][observed, 0], observed * 3.8)
