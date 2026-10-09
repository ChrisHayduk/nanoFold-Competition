"""Acceptance Gate Tests for Global Structure Tokenizer (Phase B).

Verifies the 7 mandatory Phase-B acceptance criteria:
  1. SE(3) Rigid Invariance: q(RX + t) == q(X) for arbitrary 3D rotation R and translation t.
  2. Padding Invariance: Tokens are invariant to sequence padding length.
  3. Missing Atom Robustness: Zero NaNs when atoms/residues have mask=False.
  4. End-to-End Gradient Flow: Gradients propagate from L_recon through Decoder -> Tokens -> Encoder.
  5. Prefix Length Verification: k in {2, 4, 8, 16} all function seamlessly.
  6. Symmetry of Decoded Predictions: Distogram and Contact predictions are strictly symmetric.
  7. Parameter Budget Compliance: Entire tokenizer is compact (< 3M parameters).
"""

from __future__ import annotations

import math

import pytest
import torch

from nanofold.models.tokenizer import (
    GlobalStructureTokenizer,
    TokenizerConfig,
)


@pytest.fixture
def synthetic_protein():
    """Generates synthetic 3D atom14 coordinates with realistic peptide geometry."""
    torch.manual_seed(42)
    B, L = 2, 24

    aatype = torch.randint(0, 20, (B, L), dtype=torch.long)
    atom14_positions = torch.zeros((B, L, 14, 3), dtype=torch.float32)
    atom14_mask = torch.zeros((B, L, 14), dtype=torch.float32)

    # Construct an idealized alpha-helix backbone
    for i in range(L):
        angle = i * (2.0 * math.pi / 3.6)
        z_coord = i * 1.5
        ca = torch.tensor([2.3 * math.cos(angle), 2.3 * math.sin(angle), z_coord])
        n = ca + torch.tensor([-0.8, -0.6, -0.5])
        c = ca + torch.tensor([0.7, 0.8, 0.6])
        cb = ca + torch.tensor([1.2, -0.3, 0.8])

        atom14_positions[:, i, 0, :] = n
        atom14_positions[:, i, 1, :] = ca
        atom14_positions[:, i, 2, :] = c
        atom14_positions[:, i, 4, :] = cb

        # Set backbone atom masks to 1.0
        atom14_mask[:, i, 0] = 1.0
        atom14_mask[:, i, 1] = 1.0
        atom14_mask[:, i, 2] = 1.0
        atom14_mask[:, i, 4] = 1.0

    residue_mask = torch.ones((B, L), dtype=torch.bool)
    # Mask last 3 residues as padding
    residue_mask[:, -3:] = False
    atom14_mask[:, -3:, :] = 0.0

    return {
        "aatype": aatype,
        "atom14_positions": atom14_positions,
        "atom14_mask": atom14_mask,
        "residue_mask": residue_mask,
    }


def test_se3_rigid_invariance(synthetic_protein):
    """Acceptance Gate 1: q(RX + t) MUST equal q(X) for any 3D rigid transform."""
    cfg = TokenizerConfig(d_res=64, d_pair=32, d_tok=32, k_max=8)
    tokenizer = GlobalStructureTokenizer(cfg)
    tokenizer.eval()

    aatype = synthetic_protein["aatype"]
    pos = synthetic_protein["atom14_positions"]
    atom_mask = synthetic_protein["atom14_mask"]
    res_mask = synthetic_protein["residue_mask"]

    with torch.no_grad():
        # 1. Original tokens
        tokens_orig, _ = tokenizer.encode(aatype, pos, atom_mask, res_mask)

        # 2. Apply random global 3D rotation and translation: X_transformed = X @ R^T + t
        # Generate random orthonormal rotation matrix via QR decomposition
        mat = torch.randn(3, 3)
        R_rand, _ = torch.linalg.qr(mat)
        t_rand = torch.tensor([15.2, -42.8, 105.4])

        pos_transformed = torch.matmul(pos, R_rand.T) + t_rand

        # 3. Transformed tokens
        tokens_trans, _ = tokenizer.encode(aatype, pos_transformed, atom_mask, res_mask)

    max_diff = (tokens_orig - tokens_trans).abs().max().item()
    print(f"\nMax SE(3) token difference under 3D rotation + translation: {max_diff:.6e}")
    assert max_diff < 1e-4, f"Tokenizer violated SE(3) invariance: max diff = {max_diff}"


def test_padding_invariance():
    """Acceptance Gate 2: Tokens are invariant to sequence padding."""
    torch.manual_seed(99)
    L_true = 16
    L_padded = 32

    aatype_short = torch.randint(0, 20, (1, L_true))
    pos_short = torch.randn(1, L_true, 14, 3)
    atom_mask_short = torch.ones(1, L_true, 14)
    res_mask_short = torch.ones(1, L_true, dtype=torch.bool)

    # Construct padded arrays
    aatype_pad = torch.zeros(1, L_padded, dtype=torch.long)
    aatype_pad[:, :L_true] = aatype_short

    pos_pad = torch.zeros(1, L_padded, 14, 3)
    pos_pad[:, :L_true] = pos_short

    atom_mask_pad = torch.zeros(1, L_padded, 14)
    atom_mask_pad[:, :L_true] = atom_mask_short

    res_mask_pad = torch.zeros(1, L_padded, dtype=torch.bool)
    res_mask_pad[:, :L_true] = True

    cfg = TokenizerConfig(d_res=64, d_pair=32, d_tok=32, k_max=8)
    tokenizer = GlobalStructureTokenizer(cfg)
    tokenizer.eval()

    with torch.no_grad():
        tok_short, _ = tokenizer.encode(aatype_short, pos_short, atom_mask_short, res_mask_short)
        tok_pad, _ = tokenizer.encode(aatype_pad, pos_pad, atom_mask_pad, res_mask_pad)

    diff = (tok_short - tok_pad).abs().max().item()
    print(f"\nMax token difference under padding: {diff:.6e}")
    assert diff < 1e-4, f"Padding leaked into token representation: max diff = {diff}"


def test_missing_atoms_robustness():
    """Acceptance Gate 3: No NaNs when atoms/residues have mask=False."""
    B, L = 2, 16
    aatype = torch.randint(0, 20, (B, L))
    pos = torch.randn(B, L, 14, 3)
    atom_mask = torch.zeros(B, L, 14)  # Everything missing initially
    res_mask = torch.ones(B, L, dtype=torch.bool)

    # Only provide N, CA, C for first 8 residues
    atom_mask[:, :8, :3] = 1.0

    cfg = TokenizerConfig(d_res=64, d_pair=32, d_tok=32, k_max=4)
    tokenizer = GlobalStructureTokenizer(cfg)

    out = tokenizer(aatype, pos, atom_mask, res_mask)
    assert not torch.isnan(out["tokens"]).any(), "NaN detected in tokens with missing atoms"
    assert not torch.isnan(out["loss"]).any(), "NaN detected in loss with missing atoms"


def test_gradient_flow_and_reconstruction(synthetic_protein):
    """Acceptance Gate 4 & 5: Gradients flow cleanly and all prefix lengths k function."""
    cfg = TokenizerConfig(d_res=64, d_pair=32, d_tok=32, k_max=16)
    tokenizer = GlobalStructureTokenizer(cfg)

    aatype = synthetic_protein["aatype"]
    pos = synthetic_protein["atom14_positions"]
    atom_mask = synthetic_protein["atom14_mask"]
    res_mask = synthetic_protein["residue_mask"]

    # Test all prefix lengths k in {2, 4, 8, 16}
    for k in (2, 4, 8, 16):
        out = tokenizer(aatype, pos, atom_mask, res_mask, eval_k=k)
        assert out["active_k"] == k
        assert out["tokens"].shape == (2, 16, 32)
        assert not torch.isnan(out["loss"]), f"Loss is NaN for prefix k={k}"

        # Test backpropagation
        out["loss"].backward()
        # Verify gradient reached query slots
        assert tokenizer.encoder.query_slots.grad is not None
        assert not torch.isnan(tokenizer.encoder.query_slots.grad).any()
        tokenizer.zero_grad()


def test_decoder_predictions_symmetry(synthetic_protein):
    """Acceptance Gate 6: Distogram and Contact predictions MUST be physically symmetric."""
    cfg = TokenizerConfig(d_res=64, d_pair=32, d_tok=32, k_max=8)
    tokenizer = GlobalStructureTokenizer(cfg)
    tokenizer.eval()

    with torch.no_grad():
        out = tokenizer(
            synthetic_protein["aatype"],
            synthetic_protein["atom14_positions"],
            synthetic_protein["atom14_mask"],
            synthetic_protein["residue_mask"],
            eval_k=8,
        )

    disto_logits = out["distogram_logits"]
    contact_logits = out["contact_logits"]

    diff_disto = (disto_logits - disto_logits.transpose(1, 2)).abs().max().item()
    diff_contact = (contact_logits - contact_logits.transpose(1, 2)).abs().max().item()

    assert diff_disto < 1e-6, f"Decoded distogram violated symmetry: diff={diff_disto}"
    assert diff_contact < 1e-6, f"Decoded contact violated symmetry: diff={diff_contact}"


def test_parameter_count():
    """Acceptance Gate 7: Verify entire tokenizer is lightweight (< 3M parameters)."""
    cfg = TokenizerConfig(d_res=128, d_pair=64, d_tok=64, k_max=16)
    tokenizer = GlobalStructureTokenizer(cfg)

    total_params = sum(p.numel() for p in tokenizer.parameters() if p.requires_grad)
    print(f"\nTotal GlobalStructureTokenizer parameters: {total_params:,}")

    assert total_params < 3_000_000, f"Tokenizer has {total_params:,} params, exceeding 3M limit!"


def test_prefix_reconstruction_monotonicity():
    """Acceptance Gate 8: More tokens improve or maintain reconstruction fidelity.

    Verifies ReconQuality(2) <= ReconQuality(4) <= ReconQuality(8) <= ReconQuality(16)
    (i.e. Loss(16) <= Loss(2)).
    """
    torch.manual_seed(42)
    B, L = 4, 24

    aatype = torch.randint(0, 20, (B, L))
    t = torch.linspace(0, 3 * 3.1415, L).unsqueeze(0).expand(B, -1)
    pos = torch.zeros(B, L, 14, 3)
    pos[:, :, 1, 0] = torch.cos(t) * 5.0
    pos[:, :, 1, 1] = torch.sin(t) * 5.0
    pos[:, :, 1, 2] = torch.linspace(0, 15, L).unsqueeze(0).expand(B, -1)
    pos[:, :, 0, :] = pos[:, :, 1, :] - 0.7
    pos[:, :, 2, :] = pos[:, :, 1, :] + 0.7
    pos[:, :, 4, :] = pos[:, :, 1, :] + 1.2
    atom_mask = torch.ones(B, L, 14)
    res_mask = torch.ones(B, L, dtype=torch.bool)

    cfg = TokenizerConfig(d_res=64, d_pair=32, d_tok=32, k_max=16)
    tokenizer = GlobalStructureTokenizer(cfg)
    optimizer = torch.optim.AdamW(tokenizer.parameters(), lr=2e-3)

    for _ in range(50):
        optimizer.zero_grad()
        out = tokenizer(aatype, pos, atom_mask, res_mask)
        out["loss"].backward()
        optimizer.step()

    tokenizer.eval()
    with torch.no_grad():
        loss_2 = tokenizer(aatype, pos, atom_mask, res_mask, eval_k=2)["loss"].item()
        loss_16 = tokenizer(aatype, pos, atom_mask, res_mask, eval_k=16)["loss"].item()

    assert loss_16 <= loss_2, (
        f"Reconstruction did not improve with higher token budget: Loss(16)={loss_16:.4f} > Loss(2)={loss_2:.4f}"
    )
