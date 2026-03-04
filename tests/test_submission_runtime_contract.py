from __future__ import annotations

from types import ModuleType
from typing import Any, Dict

import pytest
import torch

from nanofold.submission_runtime import SubmissionHooks, run_submission_batch, strip_supervision_from_batch


class TinyModel(torch.nn.Module):
    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        B, L = batch["aatype"].shape
        return torch.zeros((B, L, 3), dtype=torch.float32)


def _batch() -> Dict[str, Any]:
    B, L, N, T = 1, 8, 4, 1
    return {
        "chain_id": ["TEST_A"],
        "aatype": torch.zeros((B, L), dtype=torch.long),
        "msa": torch.zeros((B, N, L), dtype=torch.long),
        "deletions": torch.zeros((B, N, L), dtype=torch.long),
        "template_aatype": torch.zeros((B, T, L), dtype=torch.long),
        "template_ca_coords": torch.zeros((B, T, L, 3), dtype=torch.float32),
        "template_ca_mask": torch.ones((B, T, L), dtype=torch.bool),
        "ca_coords": torch.zeros((B, L, 3), dtype=torch.float32),
        "ca_mask": torch.ones((B, L), dtype=torch.bool),
        "residue_mask": torch.ones((B, L), dtype=torch.bool),
    }


def _hooks(run_batch_fn):
    mod = ModuleType("submission")
    return SubmissionHooks(
        module_ref="test:submission",
        module=mod,
        build_model=lambda cfg: TinyModel(),
        build_optimizer=lambda cfg, model: torch.optim.Adam(model.parameters(), lr=1e-3),
        run_batch=run_batch_fn,
        build_scheduler=None,
    )


def test_run_submission_batch_accepts_valid_output() -> None:
    def run_batch(model, batch, cfg, training):  # noqa: ANN001, ANN201
        pred = model(batch)
        out = {"pred_ca": pred}
        if training:
            out["loss"] = pred.sum() * 0.0
        return out

    hooks = _hooks(run_batch)
    model = TinyModel()
    out = run_submission_batch(hooks, model=model, batch=_batch(), cfg={}, training=True)
    assert "pred_ca" in out
    assert "loss" in out


def test_run_submission_batch_rejects_wrong_shape() -> None:
    def run_batch(model, batch, cfg, training):  # noqa: ANN001, ANN201
        return {"pred_ca": torch.zeros((1, 8, 4), dtype=torch.float32), "loss": torch.tensor(0.0)}

    hooks = _hooks(run_batch)
    with pytest.raises(ValueError):
        run_submission_batch(hooks, model=TinyModel(), batch=_batch(), cfg={}, training=True)


def test_run_submission_batch_rejects_nan_prediction() -> None:
    def run_batch(model, batch, cfg, training):  # noqa: ANN001, ANN201
        pred = torch.zeros((1, 8, 3), dtype=torch.float32)
        pred[0, 0, 0] = float("nan")
        return {"pred_ca": pred, "loss": torch.tensor(0.0)}

    hooks = _hooks(run_batch)
    with pytest.raises(ValueError):
        run_submission_batch(hooks, model=TinyModel(), batch=_batch(), cfg={}, training=True)


def test_strip_supervision_from_batch_for_inference() -> None:
    batch = _batch()
    out = strip_supervision_from_batch(batch)
    assert "ca_coords" not in out
    assert "ca_mask" not in out
