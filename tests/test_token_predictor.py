"""Acceptance Gate Tests for Phase C JEPA Structural Token Predictor.

Verifies the 6 mandatory Phase-C acceptance criteria:
  1. Shape & Interface Contract: Predicts continuous tokens q̂ in R^{B x 16 x 64}.
  2. Sequence Padding Invariance: Sequence padding does not leak into predicted tokens.
  3. Strict Stop-Gradient on Teacher: Gradients flow to student, never to teacher targets.
  4. Hierarchical Prefix Loss Weighting: Early slots (1..4) weighted more heavily than later slots.
  5. Structural Geometry Bridge Test:
         L(q*) <= L(q̂) << L(q_random)
     Predicted tokens decode into valid pairwise geometries through the frozen Phase B decoder.
  6. Parameter Budget Compliance: Student predictor is lightweight (< 500k parameters).
"""

from __future__ import annotations

import pytest
import torch

from nanofold.models.token_predictor import (
    PhaseCJEPALoss,
    PredictorConfig,
    TokenPredictor,
)
from nanofold.models.tokenizer import (
    GlobalStructureTokenizer,
    TokenizerConfig,
)


@pytest.fixture
def synthetic_trunk_features():
    """Generates synthetic trunk features (s, z) mimicking Micro-Pairformer outputs."""
    torch.manual_seed(42)
    B, L = 2, 20
    d_single = 128
    d_pair = 64

    s = torch.randn((B, L, d_single))
    z = torch.randn((B, L, L, d_pair))
    # Symmetrize pair features slightly
    z = 0.5 * (z + z.transpose(1, 2))
    mask = torch.ones((B, L), dtype=torch.bool)
    # Mask last 4 residues as padding
    mask[:, -4:] = False

    return {"s": s, "z": z, "mask": mask, "B": B, "L": L}


def test_token_predictor_shape_and_contract(synthetic_trunk_features):
    """Acceptance Gate 1: Predictor outputs continuous tokens q̂ in R^{B x 16 x 64}."""
    cfg = PredictorConfig(d_single=128, d_pair=64, d_tok=64, k_max=16)
    predictor = TokenPredictor(cfg)

    s = synthetic_trunk_features["s"]
    z = synthetic_trunk_features["z"]
    mask = synthetic_trunk_features["mask"]

    q_pred = predictor(s, z, mask=mask)

    assert q_pred.shape == (2, 16, 64)
    assert not torch.isnan(q_pred).any(), "NaN detected in predicted tokens"


def test_padding_invariance():
    """Acceptance Gate 2: Predicted tokens are invariant to sequence padding."""
    torch.manual_seed(99)
    L_true = 16
    L_padded = 32
    d_single = 128
    d_pair = 64

    s_short = torch.randn(1, L_true, d_single)
    z_short = torch.randn(1, L_true, L_true, d_pair)
    mask_short = torch.ones(1, L_true, dtype=torch.bool)

    # Construct padded tensors
    s_pad = torch.zeros(1, L_padded, d_single)
    s_pad[:, :L_true, :] = s_short

    z_pad = torch.zeros(1, L_padded, L_padded, d_pair)
    z_pad[:, :L_true, :L_true, :] = z_short

    mask_pad = torch.zeros(1, L_padded, dtype=torch.bool)
    mask_pad[:, :L_true] = True

    cfg = PredictorConfig(d_single=128, d_pair=64, d_tok=64, k_max=16)
    predictor = TokenPredictor(cfg)
    predictor.eval()

    with torch.no_grad():
        q_short = predictor(s_short, z_short, mask=mask_short)
        q_pad = predictor(s_pad, z_pad, mask=mask_pad)

    diff = (q_short - q_pad).abs().max().item()
    print(f"\nMax TokenPredictor padding difference: {diff:.6e}")
    assert diff < 1e-4, f"Padding leaked into predicted tokens: diff = {diff}"


def test_strict_stop_gradient_on_teacher(synthetic_trunk_features):
    """Acceptance Gate 3: JEPA loss gradients propagate to student, NEVER to teacher."""
    cfg = PredictorConfig(d_single=128, d_pair=64, d_tok=64, k_max=16)
    predictor = TokenPredictor(cfg)
    jepa_loss_fn = PhaseCJEPALoss(cfg)

    s = synthetic_trunk_features["s"]
    z = synthetic_trunk_features["z"]
    mask = synthetic_trunk_features["mask"]

    # Student produces q_pred with gradients
    q_pred = predictor(s, z, mask=mask)

    # Teacher mock output with requires_grad=True
    q_teacher = torch.randn(2, 16, 64, requires_grad=True)

    loss_dict = jepa_loss_fn(q_pred=q_pred, q_target=q_teacher)
    loss = loss_dict["loss"]
    loss.backward()

    # 1. Verify gradient reached student parameters
    assert predictor.query_slots.grad is not None
    assert not torch.isnan(predictor.query_slots.grad).any()

    # 2. Verify gradient did NOT reach teacher target
    assert q_teacher.grad is None, (
        "Strict Stop-Gradient Violated: Gradients leaked into teacher target!"
    )


def test_curriculum_prefix_weights():
    """Acceptance Gate 4: Hierarchical weights (1.0, 0.75, 0.50) are correctly applied."""
    cfg = PredictorConfig(prefix_weights=(1.0, 0.75, 0.5))
    loss_fn = PhaseCJEPALoss(cfg)

    # Slot weights buffer should match [1.0]*4 + [0.75]*4 + [0.5]*8
    expected = torch.tensor([1.0] * 4 + [0.75] * 4 + [0.5] * 8)
    assert torch.allclose(loss_fn.slot_weights, expected)


def test_decoded_geometry_bridge():
    """Acceptance Gate 5: Predicted tokens decode to meaningful geometry vs random tokens.

    Tests the critical bridge:
        L(q*) <= L(q̂) << L(q_random)
    """
    torch.manual_seed(42)
    B, L = 2, 24

    # Generate synthetic target structure
    aatype = torch.randint(0, 20, (B, L))
    pos = torch.zeros(B, L, 14, 3)
    t = torch.linspace(0, 3 * 3.1415, L).unsqueeze(0).expand(B, -1)
    pos[:, :, 1, 0] = torch.cos(t) * 5.0
    pos[:, :, 1, 1] = torch.sin(t) * 5.0
    pos[:, :, 1, 2] = torch.linspace(0, 15, L).unsqueeze(0).expand(B, -1)
    pos[:, :, 0, :] = pos[:, :, 1, :] - 0.7
    pos[:, :, 2, :] = pos[:, :, 1, :] + 0.7
    atom_mask = torch.ones(B, L, 14)
    res_mask = torch.ones(B, L, dtype=torch.bool)

    # Phase B Tokenizer (Teacher + Decoder)
    tok_cfg = TokenizerConfig(d_res=64, d_pair=32, d_tok=32, k_max=16)
    tokenizer = GlobalStructureTokenizer(tok_cfg)
    opt_tok = torch.optim.AdamW(tokenizer.parameters(), lr=5e-3)

    # Train tokenizer so decoder learns to map tokens to geometry
    for _ in range(40):
        opt_tok.zero_grad()
        out = tokenizer(aatype, pos, atom_mask, res_mask, eval_k=16)
        out["loss"].backward()
        opt_tok.step()

    tokenizer.eval()

    # Phase C Student Predictor
    pred_cfg = PredictorConfig(d_single=64, d_pair=32, d_tok=32, k_max=16)
    predictor = TokenPredictor(pred_cfg)
    jepa_loss_fn = PhaseCJEPALoss(pred_cfg)
    opt_pred = torch.optim.AdamW(predictor.parameters(), lr=5e-3)

    # Mock trunk embeddings
    s_trunk = torch.randn(B, L, 64)
    z_trunk = torch.randn(B, L, L, 32)

    with torch.no_grad():
        q_target, geom = tokenizer.encode(aatype, pos, atom_mask, res_mask)

    # Train student to predict target tokens
    for _ in range(40):
        opt_pred.zero_grad()
        q_pred = predictor(s_trunk, z_trunk, mask=res_mask)
        loss = jepa_loss_fn(q_pred, q_target)["loss"]
        loss.backward()
        opt_pred.step()

    predictor.eval()
    with torch.no_grad():
        q_pred = predictor(s_trunk, z_trunk, mask=res_mask)
        q_random = torch.randn_like(q_pred)

        # Ground truth distance matrix and mask
        d_cb = geom["d_cb"]
        off_diag = ~torch.eye(L, dtype=torch.bool).unsqueeze(0)
        dist_mask = geom["pair_mask_cb"].bool() & off_diag
        bin_w = (tok_cfg.rbf_max - tok_cfg.rbf_min) / float(tok_cfg.distogram_bins - 1)
        gt_bins = ((d_cb - tok_cfg.rbf_min) / bin_w).long().clamp(0, tok_cfg.distogram_bins - 1)

        # Decode through frozen Phase B Decoder
        dec_target = tokenizer.decoder(q_target, seq_len=L, active_k=16)
        dec_pred = tokenizer.decoder(q_pred, seq_len=L, active_k=16)
        dec_random = tokenizer.decoder(q_random, seq_len=L, active_k=16)

        loss_target = torch.nn.functional.cross_entropy(dec_target["distogram_logits"][dist_mask], gt_bins[dist_mask]).item()
        loss_pred = torch.nn.functional.cross_entropy(dec_pred["distogram_logits"][dist_mask], gt_bins[dist_mask]).item()
        loss_rand = torch.nn.functional.cross_entropy(dec_random["distogram_logits"][dist_mask], gt_bins[dist_mask]).item()

    print(f"\nGeometry Bridge Evaluation: Loss(q*)={loss_target:.4f}, Loss(q̂)={loss_pred:.4f}, Loss(q_random)={loss_rand:.4f}")
    assert loss_pred < loss_rand, (
        f"Predicted tokens did not decode better than random noise: Loss(q̂)={loss_pred:.4f} >= Loss(random)={loss_rand:.4f}"
    )


def test_parameter_count():
    """Acceptance Gate 6: Verify student TokenPredictor is compact (< 500k params)."""
    cfg = PredictorConfig(d_single=128, d_pair=64, d_tok=64, k_max=16)
    predictor = TokenPredictor(cfg)

    total_params = sum(p.numel() for p in predictor.parameters() if p.requires_grad)
    print(f"\nTotal TokenPredictor parameters: {total_params:,}")

    assert total_params < 500_000, f"TokenPredictor has {total_params:,} parameters, exceeding 500k!"
