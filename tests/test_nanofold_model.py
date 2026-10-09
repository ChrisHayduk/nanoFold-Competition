"""Phase F Acceptance Tests: Integrated NanoFold Architecture.

Validates the full end-to-end differentiable system wiring:
  - Micro-Pairformer Trunk (Phase A)
  - JEPA Token Predictor (Phase C)
  - Recurrent SE(3) Workspace Thinker (Phase D)
  - Torsion Head + Atom14 Kinematics Assembly (Phase E)

Test suite verifies:
  1. Parameter count strictly complies with < 3.0M budget (~2.32M actual).
  2. Output shape strictly matches the atom14 competition contract: (B, L, 14, 3).
  3. Padding masking: masked residues are strictly zeroed out.
  4. End-to-end multi-task loss computation and gradient backpropagation.
  5. Test-time compute scaling: variable recurrence steps R in {1, 2, 4, 8}.
  6. Inference with single sequence MSA (S=1).
"""

from __future__ import annotations

import pytest
import torch

from nanofold.models.nanofold_model import (
    NanoFoldConfig,
    NanoFoldLoss,
    NanoFoldModel,
)
from nanofold.utils import count_parameters


@pytest.fixture
def small_config() -> NanoFoldConfig:
    """Returns a lightweight configuration for fast test execution."""
    cfg = NanoFoldConfig()
    cfg.pairformer.n_blocks = 2
    cfg.thinker.d_workspace = 64
    cfg.thinker.num_steps = 2
    return cfg


def _make_dummy_batch(
    B: int = 2,
    L: int = 16,
    S: int = 4,
    device: str = "cpu",
) -> dict[str, torch.Tensor]:
    """Creates a dummy batch matching official data shapes."""
    torch.manual_seed(42)
    aatype = torch.randint(0, 20, (B, L), device=device)
    msa = torch.randint(0, 21, (B, S, L), device=device)
    deletions = torch.zeros((B, S, L), dtype=torch.float32, device=device)
    residue_mask = torch.ones((B, L), dtype=torch.bool, device=device)
    # Mask out last 3 residues of second batch item
    if B > 1 and L > 4:
        residue_mask[1, -3:] = False

    ca_coords = torch.randn(B, L, 3, device=device)
    ca_mask = residue_mask.clone()
    atom14_positions = torch.randn(B, L, 14, 3, device=device)
    atom14_mask = torch.ones(B, L, 14, dtype=torch.bool, device=device)

    return {
        "aatype": aatype,
        "msa": msa,
        "deletions": deletions,
        "residue_mask": residue_mask,
        "ca_coords": ca_coords,
        "ca_mask": ca_mask,
        "atom14_positions": atom14_positions,
        "atom14_mask": atom14_mask,
    }


def test_nanofold_model_parameter_budget():
    """Verify default model has ~2.83M parameters and strictly < 3.0M budget."""
    cfg = NanoFoldConfig()
    model = NanoFoldModel(cfg)
    total_params = count_parameters(model)

    assert total_params < 3_000_000, f"Exceeded parameter budget: {total_params} >= 3.0M"
    assert 2_700_000 <= total_params <= 2_950_000, f"Expected ~2.83M parameters, got {total_params}"


def test_nanofold_model_forward_atom14_contract(small_config):
    """Verify forward output shape strictly follows (B, L, 14, 3)."""
    model = NanoFoldModel(small_config)
    model.eval()

    batch = _make_dummy_batch(B=2, L=16, S=4)
    with torch.no_grad():
        out = model(
            aatype=batch["aatype"],
            msa=batch["msa"],
            deletions=batch["deletions"],
            residue_mask=batch["residue_mask"],
        )

    pred_atom14 = out["pred_atom14"]
    atom14_mask = out["atom14_mask"]

    assert isinstance(pred_atom14, torch.Tensor)
    assert pred_atom14.shape == (2, 16, 14, 3), f"Wrong shape: {pred_atom14.shape}"
    assert atom14_mask.shape == (2, 16, 14)
    assert not torch.isnan(pred_atom14).any(), "NaNs found in pred_atom14"
    assert not torch.isinf(pred_atom14).any(), "Infs found in pred_atom14"

    # Intermediate outputs check
    assert out["final_frames"].rot.shape == (2, 16, 3, 3)
    assert out["final_frames"].trans.shape == (2, 16, 3)
    assert out["tokens_pred"].shape == (2, 16, 64)
    assert len(out["trajectory"]) == small_config.thinker.num_steps
    assert out["distogram_logits"].shape == (2, 16, 16, 64)
    assert out["contact_logits"].shape == (2, 16, 16)


def test_nanofold_model_padding_zeroing(small_config):
    """Verify that masked padding residues have strictly zeroed coordinates."""
    model = NanoFoldModel(small_config)
    model.eval()

    batch = _make_dummy_batch(B=2, L=16, S=4)
    with torch.no_grad():
        out = model(
            aatype=batch["aatype"],
            msa=batch["msa"],
            deletions=batch["deletions"],
            residue_mask=batch["residue_mask"],
        )

    pred_atom14 = out["pred_atom14"]
    # Residues 13..15 of batch item 1 are masked out
    masked_coords = pred_atom14[1, -3:]
    assert torch.all(masked_coords == 0.0), "Padding residues must have 0 coordinates"

    # Valid residues must have non-zero coordinates
    valid_coords = pred_atom14[1, :10]
    assert not torch.all(valid_coords == 0.0), "Valid residues should not be all zeros"


def test_nanofold_model_loss_and_backward(small_config):
    """Verify end-to-end differentiable loss computation and non-zero gradients."""
    model = NanoFoldModel(small_config)
    model.train()
    loss_fn = NanoFoldLoss(small_config)

    batch = _make_dummy_batch(B=2, L=12, S=4)
    out = model(
        aatype=batch["aatype"],
        msa=batch["msa"],
        deletions=batch["deletions"],
        residue_mask=batch["residue_mask"],
    )

    loss_dict = loss_fn(out, batch)
    loss = loss_dict["loss"]

    assert torch.isfinite(loss), f"Loss is not finite: {loss.item()}"
    assert loss.item() > 0.0

    loss.backward()

    # Check gradients exist in each subsystem
    # 1. MicroPairformer trunk
    p_grad = model.pairformer.blocks[0].single_attention.q_proj.weight.grad
    assert p_grad is not None and torch.isfinite(p_grad).all()

    # 2. Token Predictor
    tp_grad = model.token_predictor.query_slots.grad
    assert tp_grad is not None and torch.isfinite(tp_grad).all()

    # 3. Workspace Thinker (verify dh, dt, and domega all receive active, finite gradients)
    wt_grad = model.workspace_thinker.reasoning_block.head_dh.weight.grad
    assert wt_grad is not None and torch.isfinite(wt_grad).all()

    wt_dt_grad = model.workspace_thinker.reasoning_block.head_dt.weight.grad
    assert wt_dt_grad is not None and torch.isfinite(wt_dt_grad).all()
    assert wt_dt_grad.norm().item() > 0.0, "head_dt must receive non-zero gradients"

    wt_domega_grad = model.workspace_thinker.reasoning_block.head_domega.weight.grad
    assert wt_domega_grad is not None and torch.isfinite(wt_domega_grad).all()
    assert wt_domega_grad.norm().item() > 0.0, "head_domega must receive non-zero gradients"

    # 4. Torsion Head
    th_grad = model.torsion_head.net[-1].weight.grad
    assert th_grad is not None and torch.isfinite(th_grad).all()


def test_nanofold_test_time_compute_scaling(small_config):
    """Verify dynamic test-time recursion steps R in {1, 2, 4, 8} without errors."""
    model = NanoFoldModel(small_config)
    model.eval()

    batch = _make_dummy_batch(B=1, L=10, S=2)
    for r_steps in (1, 2, 4, 8):
        with torch.no_grad():
            out = model(
                aatype=batch["aatype"],
                msa=batch["msa"],
                deletions=batch["deletions"],
                residue_mask=batch["residue_mask"],
                eval_steps=r_steps,
            )
        assert len(out["trajectory"]) == r_steps
        assert out["pred_atom14"].shape == (1, 10, 14, 3)
        assert not torch.isnan(out["pred_atom14"]).any()


def test_nanofold_single_sequence_inference(small_config):
    """Verify that inference works even when S=1 (no homologous sequences)."""
    model = NanoFoldModel(small_config)
    model.eval()

    batch = _make_dummy_batch(B=1, L=14, S=1)
    with torch.no_grad():
        out = model(
            aatype=batch["aatype"],
            msa=batch["msa"],
            deletions=batch["deletions"],
            residue_mask=batch["residue_mask"],
        )
    assert out["pred_atom14"].shape == (1, 14, 14, 3)
    assert not torch.isnan(out["pred_atom14"]).any()
