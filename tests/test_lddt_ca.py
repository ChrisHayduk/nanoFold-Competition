from __future__ import annotations

import torch

from nanofold.metrics import lddt_ca


def test_lddt_ca_identity_is_one() -> None:
    true_ca = torch.randn(9, 3)
    pred_ca = true_ca.clone()
    mask = torch.ones(9, dtype=torch.bool)
    score = lddt_ca(pred_ca, true_ca, mask)
    assert torch.isclose(score, torch.tensor(1.0), atol=1e-6)


def test_lddt_ca_translation_invariant() -> None:
    true_ca = torch.randn(9, 3)
    pred_ca = true_ca + torch.tensor([4.0, -3.0, 1.0])
    mask = torch.ones(9, dtype=torch.bool)
    score = lddt_ca(pred_ca, true_ca, mask)
    assert torch.isclose(score, torch.tensor(1.0), atol=1e-6)


def test_lddt_ca_masked_too_small_is_zero() -> None:
    true_ca = torch.randn(5, 3)
    pred_ca = true_ca.clone()
    mask = torch.tensor([True, False, False, False, False])
    score = lddt_ca(pred_ca, true_ca, mask)
    assert torch.isclose(score, torch.tensor(0.0), atol=1e-6)


def test_lddt_ca_controlled_perturbation_drops_score() -> None:
    true_ca = torch.randn(10, 3)
    pred_ca = true_ca.clone()
    pred_ca[0] += torch.tensor([8.0, 0.0, 0.0])
    mask = torch.ones(10, dtype=torch.bool)
    score = lddt_ca(pred_ca, true_ca, mask)
    assert float(score) < 1.0
