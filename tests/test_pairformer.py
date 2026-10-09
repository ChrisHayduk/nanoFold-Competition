"""Scientific Invariant Tests for the Compact Micro-Pairformer Trunk (Stage 1).

Verifies:
  1. Forward pass shapes for arbitrary batch size B, sequence length L, and MSA depth N.
  2. Physical symmetry of Distogram and Contact predictions (d_ij == d_ji).
  3. Padding Invariance: Valid residue embeddings are identical regardless of padding length.
  4. MSA Row Permutation Invariance: Permuting non-query homolog rows yields identical outputs.
  5. Relative Position Clipping: Boundary behavior for sequences up to L=256 (|j - i| <= 255).
  6. Activation Magnitude Stability: Checks that std(z) remains stable across all 6 blocks.
  7. Mixed Precision (BF16): Forward and backward pass stability in bfloat16 autocast.
  8. Parameter count budget compliance (< 15M parameters).
"""

from __future__ import annotations

import pytest
import torch

from nanofold.models.pairformer import (
    ContactHead,
    DistogramHead,
    InputEmbedder,
    MicroPairformer,
    PairformerConfig,
    TriangularMultiplicativeUpdate,
)


@pytest.fixture
def dummy_batch():
    """Generates synthetic biological tensors matching NanoFold input shapes."""
    torch.manual_seed(42)
    B, N, L = 2, 16, 32

    aatype = torch.randint(0, 20, (B, L), dtype=torch.long)
    msa = torch.randint(0, 22, (B, N, L), dtype=torch.long)
    deletions = torch.randint(0, 5, (B, N, L), dtype=torch.long)
    residue_index = torch.arange(L, dtype=torch.long).unsqueeze(0).expand(B, L)

    # Residue mask: last 4 residues in sequence are padding
    residue_mask = torch.ones((B, L), dtype=torch.bool)
    residue_mask[:, -4:] = False

    return {
        "aatype": aatype,
        "msa": msa,
        "deletions": deletions,
        "residue_index": residue_index,
        "residue_mask": residue_mask,
    }


def test_input_embedder_shapes(dummy_batch):
    """Test that InputEmbedder produces correctly shaped single and pair tensors."""
    cfg = PairformerConfig(d_single=64, d_pair=32, n_blocks=2)
    embedder = InputEmbedder(cfg)

    s, z = embedder(
        aatype=dummy_batch["aatype"],
        msa=dummy_batch["msa"],
        deletions=dummy_batch["deletions"],
        residue_index=dummy_batch["residue_index"],
        residue_mask=dummy_batch["residue_mask"],
    )

    B, L = dummy_batch["aatype"].shape
    assert s.shape == (B, L, cfg.d_single)
    assert z.shape == (B, L, L, cfg.d_pair)

    # Verify that padded positions are zeroed out
    assert torch.all(s[:, -4:] == 0.0), "Padded positions in single representation must be zero"
    assert torch.all(z[:, -4:, :] == 0.0), "Padded rows in pair representation must be zero"
    assert torch.all(z[:, :, -4:] == 0.0), "Padded columns in pair representation must be zero"


def test_triangular_multiplicative_update():
    """Verify outgoing and incoming triangular updates preserve shapes and handle masks."""
    B, L, d_pair, d_hidden = 2, 16, 32, 16
    z = torch.randn(B, L, L, d_pair)
    mask = torch.ones(B, L, L, dtype=torch.bool)
    mask[:, -2:, :] = False
    mask[:, :, -2:] = False

    tri_out = TriangularMultiplicativeUpdate(d_pair=d_pair, d_hidden=d_hidden, mode="outgoing")
    tri_in = TriangularMultiplicativeUpdate(d_pair=d_pair, d_hidden=d_hidden, mode="incoming")

    z_out = tri_out(z, mask=mask)
    z_in = tri_in(z, mask=mask)

    assert z_out.shape == (B, L, L, d_pair)
    assert z_in.shape == (B, L, L, d_pair)
    assert not torch.isnan(z_out).any()
    assert not torch.isnan(z_in).any()


def test_micro_pairformer_forward_and_backward(dummy_batch):
    """Verify complete forward pass and gradient backpropagation through MicroPairformer."""
    cfg = PairformerConfig(d_single=64, d_pair=32, d_tri_hidden=32, n_blocks=2)
    model = MicroPairformer(cfg)

    s, z = model(
        aatype=dummy_batch["aatype"],
        msa=dummy_batch["msa"],
        deletions=dummy_batch["deletions"],
        residue_index=dummy_batch["residue_index"],
        residue_mask=dummy_batch["residue_mask"],
    )

    B, L = dummy_batch["aatype"].shape
    assert s.shape == (B, L, cfg.d_single)
    assert z.shape == (B, L, L, cfg.d_pair)

    disto_head = DistogramHead(d_pair=cfg.d_pair, num_bins=64)
    contact_head = ContactHead(d_pair=cfg.d_pair)

    disto_logits = disto_head(z)
    contact_logits = contact_head(z, return_logits=True)

    assert disto_logits.shape == (B, L, L, 64)
    assert contact_logits.shape == (B, L, L)

    # Test backpropagation
    dummy_loss = disto_logits.sum() + contact_logits.sum()
    dummy_loss.backward()

    assert model.input_embedder.aa_embedding.weight.grad is not None
    assert not torch.isnan(model.input_embedder.aa_embedding.weight.grad).any()


def test_distogram_and_contact_symmetry(dummy_batch):
    """Scientific Invariant: Distance and contact predictions MUST be physically symmetric."""
    cfg = PairformerConfig(d_single=64, d_pair=32, n_blocks=2)
    model = MicroPairformer(cfg)

    _, z = model(
        aatype=dummy_batch["aatype"],
        msa=dummy_batch["msa"],
        deletions=dummy_batch["deletions"],
        residue_index=dummy_batch["residue_index"],
        residue_mask=dummy_batch["residue_mask"],
    )

    disto_head = DistogramHead(d_pair=cfg.d_pair, num_bins=64)
    contact_head = ContactHead(d_pair=cfg.d_pair)

    disto_logits = disto_head(z)
    contact_logits = contact_head(z, return_logits=True)

    # Numerically verify: max_{ij} |l_{ij} - l_{ji}| < 1e-6
    diff_disto = (disto_logits - disto_logits.transpose(1, 2)).abs().max().item()
    diff_contact = (contact_logits - contact_logits.transpose(1, 2)).abs().max().item()

    assert diff_disto < 1e-6, f"Distogram predictions violated symmetry: max diff = {diff_disto}"
    assert diff_contact < 1e-6, f"Contact predictions violated symmetry: max diff = {diff_contact}"


def test_msa_row_permutation_invariance():
    """Scientific Invariant: Permuting non-query MSA rows MUST yield identical outputs.

    Homologous sequences in an MSA have no intrinsic canonical ordering.
    A valid coevolution extractor must produce the exact same pair features regardless of row order.
    """
    torch.manual_seed(123)
    B, N, L = 1, 10, 16
    aatype = torch.randint(0, 20, (B, L))
    msa = torch.randint(0, 22, (B, N, L))
    deletions = torch.zeros((B, N, L), dtype=torch.long)
    residue_index = torch.arange(L).unsqueeze(0)
    residue_mask = torch.ones((B, L), dtype=torch.bool)

    cfg = PairformerConfig(d_single=32, d_pair=16, n_blocks=1)
    model = MicroPairformer(cfg)
    model.eval()

    with torch.no_grad():
        s1, z1 = model(aatype, msa, deletions, residue_index, residue_mask)

        # Randomly permute the non-query MSA rows (rows 1 to N-1)
        perm = torch.randperm(N - 1) + 1
        permuted_indices = torch.cat([torch.tensor([0]), perm])
        msa_permuted = msa[:, permuted_indices, :]
        deletions_permuted = deletions[:, permuted_indices, :]

        s2, z2 = model(aatype, msa_permuted, deletions_permuted, residue_index, residue_mask)

    max_s_diff = (s1 - s2).abs().max().item()
    max_z_diff = (z1 - z2).abs().max().item()

    assert max_s_diff < 1e-5, f"MSA row permutation changed single embeddings: diff = {max_s_diff}"
    assert max_z_diff < 1e-5, f"MSA row permutation changed pair embeddings: diff = {max_z_diff}"


def test_padding_invariance():
    """Scientific Invariant: Valid residue embeddings must be invariant to padding length.

    A protein of length L=16 must produce approximately identical embeddings whether
    it is processed standalone or padded to length L=32 with mask=False.
    """
    torch.manual_seed(999)
    L_true = 16
    L_padded = 32
    N = 8

    aatype_short = torch.randint(0, 20, (1, L_true))
    msa_short = torch.randint(0, 22, (1, N, L_true))
    deletions_short = torch.zeros((1, N, L_true), dtype=torch.long)
    residue_index_short = torch.arange(L_true).unsqueeze(0)
    mask_short = torch.ones((1, L_true), dtype=torch.bool)

    # Construct padded version
    aatype_pad = torch.zeros((1, L_padded), dtype=torch.long)
    aatype_pad[:, :L_true] = aatype_short
    msa_pad = torch.zeros((1, N, L_padded), dtype=torch.long)
    msa_pad[:, :, :L_true] = msa_short
    deletions_pad = torch.zeros((1, N, L_padded), dtype=torch.long)
    residue_index_pad = torch.arange(L_padded).unsqueeze(0)
    mask_pad = torch.zeros((1, L_padded), dtype=torch.bool)
    mask_pad[:, :L_true] = True

    cfg = PairformerConfig(d_single=32, d_pair=16, n_blocks=2)
    model = MicroPairformer(cfg)
    model.eval()

    with torch.no_grad():
        s_short, z_short = model(aatype_short, msa_short, deletions_short, residue_index_short, mask_short)
        s_pad, z_pad = model(aatype_pad, msa_pad, deletions_pad, residue_index_pad, mask_pad)

    # Compare unpadded region: [0 : L_true]
    diff_s = (s_short - s_pad[:, :L_true, :]).abs().max().item()
    diff_z = (z_short - z_pad[:, :L_true, :L_true, :]).abs().max().item()

    assert diff_s < 1e-4, f"Padding leaked into single representations: diff = {diff_s}"
    assert diff_z < 1e-4, f"Padding leaked into pair representations: diff = {diff_z}"


def test_relative_position_clipping():
    """Scientific Invariant: Verify relative sequence offsets up to L=256 (|j - i| <= 255)."""
    L = 256
    aatype = torch.randint(0, 20, (1, L))
    msa = torch.randint(0, 22, (1, 4, L))
    deletions = torch.zeros((1, 4, L), dtype=torch.long)
    residue_index = torch.arange(L).unsqueeze(0)
    residue_mask = torch.ones((1, L), dtype=torch.bool)

    cfg = PairformerConfig(d_single=32, d_pair=16, max_relative_pos=32, n_blocks=1)
    embedder = InputEmbedder(cfg)

    # Should run smoothly without IndexError or out-of-bounds embedding lookup
    s, z = embedder(aatype, msa, deletions, residue_index, residue_mask)
    assert s.shape == (1, L, 32)
    assert z.shape == (1, L, L, 16)
    assert not torch.isnan(z).any()


def test_activation_magnitude_stability():
    """Scientific Invariant: Verify std(z) does not explode across all 6 blocks."""
    torch.manual_seed(42)
    B, N, L = 2, 8, 32
    aatype = torch.randint(0, 20, (B, L))
    msa = torch.randint(0, 22, (B, N, L))
    deletions = torch.zeros((B, N, L), dtype=torch.long)
    residue_index = torch.arange(L).unsqueeze(0).expand(B, L)
    residue_mask = torch.ones((B, L), dtype=torch.bool)

    cfg = PairformerConfig(d_single=64, d_pair=32, d_tri_hidden=32, n_blocks=6)
    model = MicroPairformer(cfg)
    model.eval()

    with torch.no_grad():
        s, z = model.input_embedder(aatype, msa, deletions, residue_index, residue_mask)
        pair_mask = residue_mask.unsqueeze(2) & residue_mask.unsqueeze(1)

        print(f"\nInitial z std: {z.std().item():.3f}")
        for i, block in enumerate(model.blocks):
            s, z = block(s, z, residue_mask, pair_mask)
            z_std = z.std().item()
            print(f"Block {i+1} z std: {z_std:.3f}")
            # Activation variance must stay healthy (< 3.0) and not explode exponentially
            assert z_std < 3.0, f"Activation explosion at block {i+1}: std={z_std:.3f}"


def test_mixed_precision_bf16(dummy_batch):
    """Scientific Invariant: Forward and backward stability in bfloat16 mixed precision."""
    cfg = PairformerConfig(d_single=64, d_pair=32, d_tri_hidden=32, n_blocks=2)
    model = MicroPairformer(cfg)

    # Test under autocast bfloat16
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        s, z = model(
            aatype=dummy_batch["aatype"],
            msa=dummy_batch["msa"],
            deletions=dummy_batch["deletions"],
            residue_index=dummy_batch["residue_index"],
            residue_mask=dummy_batch["residue_mask"],
        )
        disto_head = DistogramHead(d_pair=cfg.d_pair)
        logits = disto_head(z)
        loss = logits.sum()

    assert not torch.isnan(s).any(), "NaN in BF16 single representation"
    assert not torch.isnan(z).any(), "NaN in BF16 pair representation"
    assert not torch.isnan(logits).any(), "NaN in BF16 distogram logits"

    loss.backward()
    assert model.input_embedder.aa_embedding.weight.grad is not None
    assert not torch.isnan(model.input_embedder.aa_embedding.weight.grad).any()


def test_parameter_count():
    """Verify that our Micro-Pairformer is within the compact budget (< 15M parameters)."""
    cfg = PairformerConfig(d_single=128, d_pair=64, d_tri_hidden=64, n_blocks=6)
    model = MicroPairformer(cfg)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total MicroPairformer trainable parameters: {total_params:,}")

    # Standard AF2 Evoformer is 93M+ parameters. Our compact budget is < 15M.
    assert total_params < 15_000_000, f"Model has {total_params:,} params, exceeding 15M limit!"
