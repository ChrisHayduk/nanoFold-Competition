from __future__ import annotations

import math

import torch

from nanofold.metrics import (
    foldscore_auc,
    foldscore_components,
    foldscore_from_components,
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

    comps = foldscore_components(pred_atom14, true_atom14, mask)

    assert torch.isclose(comps["lddt_ca"], torch.tensor(1.0), atol=1e-6)
    assert torch.isclose(comps["lddt_backbone_atom14"], torch.tensor(1.0), atol=1e-6)
    assert torch.isclose(comps["lddt_atom14"], torch.tensor(1.0), atol=1e-6)
    assert torch.isclose(comps["foldscore"], torch.tensor(1.0), atol=1e-6)


def test_foldscore_weighting_contract() -> None:
    score = foldscore_from_components(
        lddt_ca_score=torch.tensor(1.0),
        lddt_backbone_atom14_score=torch.tensor(0.5),
        lddt_atom14_score=torch.tensor(0.0),
    )
    assert torch.isclose(score, torch.tensor(0.70), atol=1e-6)


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
