from __future__ import annotations

import torch

from nanofold.metrics import lddt_ca


def test_lddt_perfect_prediction_is_one() -> None:
    true_ca = torch.randn(10, 3)
    pred_ca = true_ca.clone()
    mask = torch.ones(10, dtype=torch.bool)
    score = lddt_ca(pred_ca, true_ca, mask)
    assert torch.isclose(score, torch.tensor(1.0), atol=1e-6)


def test_lddt_rigid_transform_invariance() -> None:
    true_ca = torch.randn(12, 3)
    theta = torch.tensor(0.7)
    rot = torch.tensor(
        [
            [torch.cos(theta), -torch.sin(theta), 0.0],
            [torch.sin(theta), torch.cos(theta), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    pred_ca = (true_ca @ rot.T) + torch.tensor([3.0, -2.0, 1.0])
    mask = torch.ones(12, dtype=torch.bool)
    score = lddt_ca(pred_ca, true_ca, mask)
    assert torch.isclose(score, torch.tensor(1.0), atol=1e-6)


def test_lddt_masking_with_too_few_residues_returns_zero() -> None:
    true_ca = torch.randn(4, 3)
    pred_ca = true_ca.clone()
    mask = torch.tensor([True, False, False, False])
    score = lddt_ca(pred_ca, true_ca, mask)
    assert torch.isclose(score, torch.tensor(0.0), atol=1e-6)
