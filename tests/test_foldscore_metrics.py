from __future__ import annotations

import math

import pytest
import torch

from nanofold.metrics import (
    FOLDSCORE_COMPONENT_NAMES,
    FOLDSCORE_WEIGHT_BY_COMPONENT,
    _kabsch_align,
    backbone_atom14_score,
    dipdiff_atom14_score,
    foldscore_auc,
    foldscore_components,
    foldscore_from_components,
    gdt_ca,
    gdt_ha_ca,
    sidechain_atom14_score,
    spheregrinder_atom14_score,
)


def _atom14_chain(length: int = 6) -> torch.Tensor:
    coords = torch.zeros((length, 14, 3), dtype=torch.float32)
    coords[..., 0] = torch.arange(length, dtype=torch.float32)[:, None] * 3.8
    coords[..., 1] = torch.arange(14, dtype=torch.float32)[None, :] * 0.1
    coords[..., 2] = torch.arange(14, dtype=torch.float32)[None, :] * 0.05
    return coords


def test_foldscore_components_perfect_atom14_is_one() -> None:
    true_atom14 = _atom14_chain()
    pred_atom14 = true_atom14.clone()
    mask = torch.ones((true_atom14.shape[0], 14), dtype=torch.bool)
    aatype = torch.zeros((true_atom14.shape[0],), dtype=torch.long)

    comps = foldscore_components(pred_atom14, true_atom14, mask, aatype)

    for name in FOLDSCORE_COMPONENT_NAMES:
        assert torch.isclose(comps[name], torch.tensor(1.0), atol=1e-6), name


def test_foldscore_penalizes_collapsed_nonadjacent_atoms() -> None:
    true_atom14 = _atom14_chain()
    pred_atom14 = true_atom14.clone()
    pred_atom14[4] = pred_atom14[0]
    mask = torch.ones((true_atom14.shape[0], 14), dtype=torch.bool)
    aatype = torch.zeros((true_atom14.shape[0],), dtype=torch.long)

    comps = foldscore_components(pred_atom14, true_atom14, mask, aatype)

    assert comps["molprobity_clash_atom14"] < 1.0
    assert comps["cad_atom14"] < 1.0
    assert comps["foldscore"] < 1.0


def test_foldscore_weighting_contract() -> None:
    score = foldscore_from_components(
        gdt_ha_ca_score=torch.tensor(1.0),
        lddt_atom14_score=torch.tensor(0.0),
        cad_atom14_score=torch.tensor(0.0),
        sg_atom14_score=torch.tensor(0.0),
        sc_atom14_score=torch.tensor(0.0),
        molprobity_clash_atom14_score=torch.tensor(0.0),
        bb_atom14_score=torch.tensor(0.0),
        dipdiff_atom14_score=torch.tensor(0.0),
    )
    assert torch.isclose(
        score,
        torch.tensor(FOLDSCORE_WEIGHT_BY_COMPONENT["gdt_ha_ca"]),
        atol=1e-6,
    )

    all_one_score = foldscore_from_components(
        gdt_ha_ca_score=torch.tensor(1.0),
        lddt_atom14_score=torch.tensor(1.0),
        cad_atom14_score=torch.tensor(1.0),
        sg_atom14_score=torch.tensor(1.0),
        sc_atom14_score=torch.tensor(1.0),
        molprobity_clash_atom14_score=torch.tensor(1.0),
        bb_atom14_score=torch.tensor(1.0),
        dipdiff_atom14_score=torch.tensor(1.0),
    )
    assert torch.isclose(all_one_score, torch.tensor(1.0), atol=1e-6)


def test_gdt_ha_ca_is_rigid_transform_invariant() -> None:
    true_ca = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 1.0],
        ],
        dtype=torch.float32,
    )
    rotation = torch.tensor(
        [
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
    )
    pred_ca = true_ca @ rotation.T + torch.tensor([7.0, -3.0, 2.0])
    mask = torch.ones((4,), dtype=torch.bool)

    assert torch.isclose(gdt_ha_ca(pred_ca, true_ca, mask), torch.tensor(1.0), atol=1e-5)


def test_gdt_threshold_optimizes_over_partial_superpositions() -> None:
    true_ca = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [8.0, 0.0, 0.0],
            [9.0, 0.0, 0.0],
            [8.0, 1.0, 0.0],
            [8.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
    )
    pred_ca = true_ca.clone()
    pred_ca[4:] += torch.tensor([30.0, 5.0, -7.0])
    mask = torch.ones((8,), dtype=torch.bool)

    score = gdt_ca(pred_ca, true_ca, mask, thresholds=(0.5,))

    assert score >= 0.5


def test_spheregrinder_uses_local_superposition_and_rmsd_cutoffs() -> None:
    true_atom14 = _atom14_chain(length=6)
    pred_atom14 = true_atom14.clone()
    mask = torch.ones((6, 14), dtype=torch.bool)

    perfect = spheregrinder_atom14_score(pred_atom14, true_atom14, mask)
    pred_atom14[2, :8] += torch.tensor([0.0, 0.0, 12.0])
    perturbed = spheregrinder_atom14_score(pred_atom14, true_atom14, mask)

    assert torch.isclose(perfect, torch.tensor(1.0), atol=1e-6)
    assert perturbed < perfect


def test_backbone_score_equal_weights_phi_psi_omega_errors() -> None:
    torch.manual_seed(0)
    true_atom14 = torch.randn((5, 14, 3), dtype=torch.float32)
    pred_atom14 = true_atom14.clone()
    mask = torch.zeros((5, 14), dtype=torch.bool)
    mask[:, :4] = True

    perfect = backbone_atom14_score(pred_atom14, true_atom14, mask)
    pred_atom14[2, 1] += torch.tensor([3.0, -2.0, 1.5])
    perturbed = backbone_atom14_score(pred_atom14, true_atom14, mask)

    assert torch.isclose(perfect, torch.tensor(1.0), atol=1e-6)
    assert perturbed < perfect


def test_sidechain_score_uses_chi_angles_and_symmetry() -> None:
    true_atom14 = torch.zeros((1, 14, 3), dtype=torch.float32)
    pred_atom14 = true_atom14.clone()
    mask = torch.zeros((1, 14), dtype=torch.bool)
    # VAL: N, CA, C, O, CB, CG1, CG2. chi1 is pi-periodic under CG1/CG2 symmetry.
    true_atom14[0, 0] = torch.tensor([0.0, 0.0, 0.0])
    true_atom14[0, 1] = torch.tensor([1.0, 0.0, 0.0])
    true_atom14[0, 2] = torch.tensor([2.0, 0.1, 0.0])
    true_atom14[0, 3] = torch.tensor([2.4, -0.8, 0.0])
    true_atom14[0, 4] = torch.tensor([1.0, 1.0, 0.0])
    true_atom14[0, 5] = torch.tensor([1.0, 1.0, 1.0])
    true_atom14[0, 6] = torch.tensor([1.0, 2.0, 0.0])
    pred_atom14.copy_(true_atom14)
    pred_atom14[0, 5] = torch.tensor([1.0, 1.0, -1.0])
    mask[0, :7] = True
    val_aatype = torch.tensor([19], dtype=torch.long)

    assert torch.isclose(
        sidechain_atom14_score(pred_atom14, true_atom14, mask, val_aatype),
        torch.tensor(1.0),
        atol=1e-6,
    )

    cys_aatype = torch.tensor([4], dtype=torch.long)
    assert sidechain_atom14_score(pred_atom14, true_atom14, mask, cys_aatype) < 1.0


def test_dipdiff_uses_three_residue_ca_o_distance_windows() -> None:
    true_atom14 = _atom14_chain(length=5)
    pred_atom14 = true_atom14.clone()
    mask = torch.zeros((5, 14), dtype=torch.bool)
    mask[:, [1, 3]] = True

    perfect = dipdiff_atom14_score(pred_atom14, true_atom14, mask)
    pred_atom14[2, 3] += torch.tensor([0.0, 0.0, 5.0])
    perturbed = dipdiff_atom14_score(pred_atom14, true_atom14, mask)

    assert torch.isclose(perfect, torch.tensor(1.0), atol=1e-6)
    assert perturbed < perfect


def test_foldscore_auc_uses_trapezoids_over_samples() -> None:
    points = [(0, 0, 0.0), (100, 100, 1.0), (200, 200, 0.0)]
    auc = foldscore_auc(points, sample_budget=200)
    assert math.isclose(auc, 0.5, abs_tol=1e-8)


def test_foldscore_auc_rejects_duplicate_sample_axis() -> None:
    points = [(0, 0, 0.0), (100, 0, 1.0)]
    try:
        foldscore_auc(points, sample_budget=100)
    except ValueError as exc:
        assert "strictly increasing" in str(exc)
    else:
        raise AssertionError("duplicate cumulative sample counts should fail")


@pytest.mark.parametrize("points,budget", [
    ([], 100), ([(0, 0, .5)], 100), ([(1, 10, .5), (2, 100, .6)], 100),
    ([(0, 0, .5), (1, 50, .6)], 100), ([(0, 0, .5), (1, 100, float("nan"))], 100),
    ([(0, 0, .5), (1, 100, 1.5)], 100), ([(0, 0, .5), (1, 100, .5)], 0),
    ([(0, 0, .5), (1, 100, .5)], float("inf")),
])
def test_auc_rejects_partial_or_invalid_curves(points, budget) -> None:
    with pytest.raises(ValueError):
        foldscore_auc(points, sample_budget=budget)


def test_integral_zero_component_does_not_truncate_fractional_components() -> None:
    scores = {f"{name}_score": .5 for name in FOLDSCORE_WEIGHT_BY_COMPONENT}
    scores["gdt_ha_ca_score"] = 0
    assert float(foldscore_from_components(**scores)) == .375


def test_scoring_upcasts_amp_outputs_and_rejects_unsupervised_targets() -> None:
    reference = _atom14_chain().half()
    mask = torch.ones((6, 14), dtype=torch.bool)
    residues = torch.zeros(6, dtype=torch.long)
    half_scores = foldscore_components(reference, reference, mask, residues)
    float_scores = foldscore_components(reference.float(), reference.float(), mask, residues)
    assert all(torch.equal(half_scores[key], float_scores[key]) for key in half_scores)
    with pytest.raises(ValueError, match="observed CA"):
        foldscore_components(reference, reference, torch.zeros_like(mask), residues)


def test_kabsch_preserves_chirality_and_ignores_masked_points() -> None:
    target = torch.tensor([[0., 0., 0.], [4., 0., 0.], [0., 5., 0.], [0., 0., 6.], [100., 100., 100.]])
    mask = torch.tensor([True, True, True, True, False])
    translated = target + torch.tensor([9., 8., -3.])
    translated[-1] = -10000
    aligned, active = _kabsch_align(translated, target, mask)
    assert torch.allclose(aligned[active], target[active], atol=1e-5)
    mirror = target * torch.tensor([-1., 1., 1.])
    aligned, active = _kabsch_align(mirror, target, mask)
    assert torch.linalg.vector_norm(aligned[active] - target[active], dim=-1).mean() > 1


@pytest.mark.parametrize("metadata", [
    {"residue_index": torch.tensor([0, 1, 8, 9])},
    {"between_segment_residues": torch.tensor([0, 0, 1, 0])},
])
def test_local_backbone_scores_do_not_bridge_segment_breaks(metadata) -> None:
    generator = torch.Generator().manual_seed(19)
    reference = torch.randn((4, 14, 3), generator=generator)
    prediction = reference.clone()
    prediction[2:] += torch.tensor([8., -4., 6.])
    mask = torch.zeros((4, 14), dtype=torch.bool)
    mask[:, :4] = True
    residues = torch.zeros(4, dtype=torch.long)
    connected = foldscore_components(prediction, reference, mask, residues)
    disconnected = foldscore_components(prediction, reference, mask, residues, **metadata)
    assert disconnected["bb_atom14"] == pytest.approx(1., abs=1e-6)
    assert disconnected["dipdiff_atom14"] == 1
    assert connected["bb_atom14"] < .99
    assert connected["dipdiff_atom14"] < 1


def test_clashes_across_segment_break_are_not_exempted_as_bonded_neighbors() -> None:
    reference = torch.zeros((2, 14, 3))
    mask = torch.zeros((2, 14), dtype=torch.bool)
    mask[:, 1] = True
    residues = torch.zeros(2, dtype=torch.long)
    connected = foldscore_components(reference, reference, mask, residues)
    disconnected = foldscore_components(reference, reference, mask, residues, residue_index=torch.tensor([0, 4]))
    assert connected["molprobity_clash_atom14"] == 1
    assert disconnected["molprobity_clash_atom14"] < .2
