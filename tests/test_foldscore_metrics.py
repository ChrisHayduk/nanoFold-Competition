from __future__ import annotations

import math

import torch

from nanofold.metrics import (
    FOLDSCORE_COMPONENT_NAMES,
    FOLDSCORE_WEIGHT_BY_COMPONENT,
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
