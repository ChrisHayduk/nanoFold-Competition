"""Integration tests for official NanoFold v1 submission package.

Verifies:
  1. Compliance with validate_submission.py under --strict mode.
  2. build_model constructs the full model within the < 3.0M budget.
  3. build_optimizer constructs a valid optimizer.
  4. run_batch(training=True) executes and computes multi-task loss.
  5. run_batch(training=False) runs strictly without supervision labels.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch
import yaml

from nanofold.competition_policy import load_track_spec
from nanofold.submission_runtime import strip_supervision_from_batch
from nanofold.utils import count_parameters
from scripts.validate_submission import validate_submission
from submissions.nanofold_v1 import submission


@pytest.fixture
def submission_cfg() -> dict[str, Any]:
    config_path = Path("submissions/nanofold_v1/config.yaml")
    return yaml.safe_load(config_path.read_text())


def _synthetic_batch(B: int = 1, L: int = 16, N: int = 8) -> dict[str, Any]:
    """Generates synthetic supervised batch conforming to data loader interface."""
    torch.manual_seed(42)
    aatype = torch.randint(0, 20, (B, L))
    msa = torch.randint(0, 21, (B, N, L))
    deletions = torch.zeros((B, N, L), dtype=torch.long)
    residue_mask = torch.ones((B, L), dtype=torch.bool)
    ca_coords = torch.randn(B, L, 3)
    ca_mask = torch.ones((B, L), dtype=torch.bool)
    atom14_positions = torch.randn(B, L, 14, 3)
    atom14_mask = torch.ones((B, L, 14), dtype=torch.bool)

    return {
        "chain_id": ["TEST_CHAIN_A"],
        "aatype": aatype,
        "msa": msa,
        "deletions": deletions,
        "residue_mask": residue_mask,
        "residue_index": torch.arange(L).unsqueeze(0).expand(B, -1),
        "ca_coords": ca_coords,
        "ca_mask": ca_mask,
        "atom14_positions": atom14_positions,
        "atom14_mask": atom14_mask,
    }


def test_nanofold_submission_passes_official_validator():
    """Verify that submissions/nanofold_v1 passes validate_submission with 0 errors/warnings."""
    code = validate_submission(
        submission_dir=Path("submissions/nanofold_v1"),
        strict=True,
        track_id="limited",
    )
    assert code == 0, "validate_submission failed for submissions/nanofold_v1"


def test_nanofold_submission_model_parameter_limit(submission_cfg):
    """Verify model parameter limit under track limited (< 3,000,000)."""
    track_spec = load_track_spec("limited")
    model = submission.build_model(submission_cfg)
    n_params = count_parameters(model)

    if track_spec.max_params is not None:
        assert n_params <= track_spec.max_params
    assert n_params < 3_000_000
    assert 2_700_000 <= n_params <= 2_950_000


def test_nanofold_submission_optimizer(submission_cfg):
    """Verify build_optimizer creates an optimizer with step and zero_grad."""
    model = submission.build_model(submission_cfg)
    opt = submission.build_optimizer(submission_cfg, model)

    assert hasattr(opt, "step") and callable(opt.step)
    assert hasattr(opt, "zero_grad") and callable(opt.zero_grad)


def test_nanofold_submission_run_batch_training(submission_cfg):
    """Verify run_batch in training mode returns pred_atom14 and scalar loss."""
    model = submission.build_model(submission_cfg)
    model.train()
    batch = _synthetic_batch(B=1, L=16, N=8)

    out = submission.run_batch(model, batch, submission_cfg, training=True)

    assert "pred_atom14" in out
    assert out["pred_atom14"].shape == (1, 16, 14, 3)
    assert "loss" in out
    assert torch.isfinite(out["loss"])
    assert out["loss"].requires_grad


def test_nanofold_submission_run_batch_inference_sealed(submission_cfg):
    """Verify run_batch in inference mode has no access to ground truth supervision."""
    model = submission.build_model(submission_cfg)
    model.eval()
    batch = _synthetic_batch(B=1, L=16, N=8)
    unsupervised_batch = strip_supervision_from_batch(batch)

    assert "ca_coords" not in unsupervised_batch
    assert "ca_mask" not in unsupervised_batch
    assert "atom14_positions" not in unsupervised_batch
    assert "atom14_mask" not in unsupervised_batch

    with torch.no_grad():
        out = submission.run_batch(model, unsupervised_batch, submission_cfg, training=False)

    assert "pred_atom14" in out
    assert out["pred_atom14"].shape == (1, 16, 14, 3)
    assert not torch.isnan(out["pred_atom14"]).any()
    assert "loss" not in out


def test_nanofold_submission_scheduler(submission_cfg):
    """Verify build_scheduler creates a scheduler that steps and saves state."""
    model = submission.build_model(submission_cfg)
    opt = submission.build_optimizer(submission_cfg, model)
    sched = submission.build_scheduler(submission_cfg, opt)

    assert hasattr(sched, "step") and callable(sched.step)
    assert hasattr(sched, "state_dict") and callable(sched.state_dict)

    initial_lr = opt.param_groups[0]["lr"]
    opt.step()
    sched.step()
    stepped_lr = opt.param_groups[0]["lr"]

    assert stepped_lr > 0.0
    assert stepped_lr >= initial_lr
    state = sched.state_dict()
    sched.load_state_dict(state)


def test_nanofold_submission_run_batch_supervised_evaluation(submission_cfg):
    """Verify run_batch in evaluation mode returns loss when supervision labels are present."""
    model = submission.build_model(submission_cfg)
    model.eval()
    batch = _synthetic_batch(B=1, L=16, N=8)

    with torch.no_grad():
        out = submission.run_batch(model, batch, submission_cfg, training=False)

    assert "pred_atom14" in out
    assert "loss" in out
    assert torch.isfinite(out["loss"])


def test_nanofold_submission_amp_finite_gradients(submission_cfg):
    """Verify forward and backward passes produce strictly finite gradients across all parameters."""
    model = submission.build_model(submission_cfg)
    model.train()
    batch = _synthetic_batch(B=1, L=16, N=8)

    opt = submission.build_optimizer(submission_cfg, model)
    opt.zero_grad()

    out = submission.run_batch(model, batch, submission_cfg, training=True)
    loss = out["loss"]
    loss.backward()

    for name, param in model.named_parameters():
        if param.grad is not None:
            assert torch.isfinite(param.grad).all(), f"Non-finite gradient in {name}"


