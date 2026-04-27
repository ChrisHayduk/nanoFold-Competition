from __future__ import annotations

import pytest
import torch

from score import _labels_to_device, _resolve_score_device, _score_chain


def _atom14_chain(length: int = 4) -> torch.Tensor:
    coords = torch.zeros((length, 14, 3), dtype=torch.float32)
    coords[..., 0] = torch.arange(length, dtype=torch.float32)[:, None] * 3.8
    coords[..., 1] = torch.arange(14, dtype=torch.float32)[None, :] * 0.1
    coords[..., 2] = torch.arange(14, dtype=torch.float32)[None, :] * 0.05
    return coords


def test_score_requires_atom14_labels() -> None:
    labels = {
        "ca_coords": torch.zeros((4, 3), dtype=torch.float32),
        "ca_mask": torch.ones((4,), dtype=torch.bool),
    }

    with pytest.raises(ValueError, match="atom14"):
        _score_chain(
            pred_atom14=torch.zeros((4, 14, 3), dtype=torch.float32),
            labels=labels,
            features={"aatype": torch.zeros((4,), dtype=torch.long)},
        )


def test_score_uses_atom14_contract() -> None:
    atom14_positions = _atom14_chain()
    labels = {
        "ca_coords": atom14_positions[:, 1, :],
        "ca_mask": torch.ones((4,), dtype=torch.bool),
        "atom14_positions": atom14_positions,
        "atom14_mask": torch.ones((4, 14), dtype=torch.bool),
    }
    features = {"aatype": torch.zeros((4,), dtype=torch.long)}

    metrics = _score_chain(
        pred_atom14=atom14_positions.clone(),
        labels=labels,
        features=features,
    )

    assert metrics["foldscore"] == 1.0
    assert metrics["gdt_ha_ca"] == 1.0
    assert metrics["lddt_atom14"] == 1.0
    assert metrics["cad_atom14"] == 1.0
    assert metrics["sg_atom14"] == 1.0
    assert metrics["sc_atom14"] == 1.0
    assert metrics["molprobity_clash_atom14"] == 1.0
    assert metrics["bb_atom14"] == 1.0
    assert metrics["dipdiff_atom14"] == 1.0
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
