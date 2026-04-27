from __future__ import annotations

import pytest
import torch

from score import _labels_to_device, _resolve_score_device, _score_chain


def test_score_requires_atom14_labels() -> None:
    labels = {
        "ca_coords": torch.zeros((4, 3), dtype=torch.float32),
        "ca_mask": torch.ones((4,), dtype=torch.bool),
    }

    with pytest.raises(ValueError, match="atom14"):
        _score_chain(
            pred_atom14=torch.zeros((4, 14, 3), dtype=torch.float32),
            labels=labels,
        )


def test_score_uses_atom14_contract() -> None:
    labels = {
        "ca_coords": torch.zeros((4, 3), dtype=torch.float32),
        "ca_mask": torch.ones((4,), dtype=torch.bool),
        "atom14_positions": torch.zeros((4, 14, 3), dtype=torch.float32),
        "atom14_mask": torch.ones((4, 14), dtype=torch.bool),
    }

    metrics = _score_chain(
        pred_atom14=torch.zeros((4, 14, 3), dtype=torch.float32),
        labels=labels,
    )

    assert metrics["foldscore"] == 1.0
    assert metrics["lddt_ca"] == 1.0


def test_score_device_helpers_default_and_move_labels() -> None:
    device = _resolve_score_device("cpu")
    labels = {
        "ca_coords": torch.zeros((4, 3), dtype=torch.float32),
        "ca_mask": torch.ones((4,), dtype=torch.bool),
        "atom14_positions": torch.zeros((4, 14, 3), dtype=torch.float32),
        "atom14_mask": torch.ones((4, 14), dtype=torch.bool),
    }
    moved = _labels_to_device(labels, device=device)
    assert all(value.device.type == "cpu" for value in moved.values())
