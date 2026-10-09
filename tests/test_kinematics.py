"""Acceptance Gate Tests for Phase E Stereochemical Kinematics & Full Atom14 Assembly.

Verifies the 6 mandatory Phase-E acceptance criteria:
  1. TorsionHead S^1 Unit Norm Contract:
     Every predicted torsion angle is strictly on the unit circle (sin^2 + cos^2 = 1.0).
  2. Atom14 Assembly Output Shape Contract:
     pred_atom14 has shape (B, L, 14, 3) and atom14_mask has shape (B, L, 14).
  3. Amino Acid Stereochemical Mask Contract:
     Glycine has slot 4 (Cβ) masked 0, Alanine has slot 4 (Cβ) active 1, etc.
  4. Backbone Coordinate Preservation:
     Cα coordinate in slot 1 strictly matches the backbone frame translation.
  5. End-to-End Differentiability:
     Gradients from all-atom loss flow back into backbone frames and torsion weights.
  6. Compact Parameter Budget:
     TorsionHead has < 100k parameters.
"""

from __future__ import annotations

import pytest
import torch

from nanofold.models.geometry_se3 import make_extended_linear_chain
from nanofold.models.kinematics import (
    Atom14KinematicsAssembly,
    KinematicsConfig,
    StereochemicalLoss,
    TorsionHead,
)

# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def synthetic_kinematics_inputs():
    """Generates synthetic representations mimicking Phase D output."""
    torch.manual_seed(42)
    B, L = 2, 20
    d_h = 128

    h = torch.randn(B, L, d_h)
    aatype = torch.randint(0, 20, (B, L))
    # Ensure Glycine (7) and Alanine (0) and Tryptophan (17) are present
    aatype[0, 0] = 7   # GLY
    aatype[0, 1] = 0   # ALA
    aatype[0, 2] = 17  # TRP

    frames = make_extended_linear_chain(batch_size=B, seq_len=L, bond_length=3.8)

    return {
        "h": h,
        "aatype": aatype,
        "frames": frames,
        "B": B,
        "L": L,
    }


# =============================================================================
# Gate 1: TorsionHead Unit Circle Norm Contract
# =============================================================================

def test_torsion_head_unit_norm(synthetic_kinematics_inputs):
    """Gate 1: Verify predicted torsion angles satisfy sin^2 α + cos^2 α = 1.0."""
    cfg = KinematicsConfig(d_h=128, d_hidden=128)
    head = TorsionHead(cfg)

    h = synthetic_kinematics_inputs["h"]
    angles = head(h)  # [B, L, 7, 2]

    assert angles.shape == (2, 20, 7, 2)

    # Compute sin^2 + cos^2
    norm_sq = torch.sum(angles ** 2, dim=-1)  # [B, L, 7]
    diff = (norm_sq - 1.0).abs().max().item()

    print(f"\nMax deviation from S^1 unit circle: {diff:.6e}")
    assert diff < 1e-5, f"Torsion angles violate unit norm: max diff = {diff}"


# =============================================================================
# Gate 2: Atom14 Assembly Shape & Contract
# =============================================================================

def test_atom14_assembly_shapes(synthetic_kinematics_inputs):
    """Gate 2: Verify Atom14KinematicsAssembly outputs (B, L, 14, 3) and mask (B, L, 14)."""
    cfg = KinematicsConfig()
    head = TorsionHead(cfg)
    assembly = Atom14KinematicsAssembly()

    h = synthetic_kinematics_inputs["h"]
    aatype = synthetic_kinematics_inputs["aatype"]
    frames = synthetic_kinematics_inputs["frames"]

    angles = head(h)
    atom14_coords, atom14_mask = assembly(
        backbone_frames=frames,
        torsion_angles=angles,
        aatype=aatype,
    )

    assert atom14_coords.shape == (2, 20, 14, 3)
    assert atom14_mask.shape == (2, 20, 14)
    assert not torch.isnan(atom14_coords).any(), "NaN detected in atom14 coordinates"


# =============================================================================
# Gate 3: Stereochemical Mask Validity
# =============================================================================

def test_amino_acid_mask_validity(synthetic_kinematics_inputs):
    """Gate 3: Verify restype-specific atom masks (Glycine no CB, Alanine has CB, TRP has 14 atoms)."""
    cfg = KinematicsConfig()
    head = TorsionHead(cfg)
    assembly = Atom14KinematicsAssembly()

    h = synthetic_kinematics_inputs["h"]
    aatype = synthetic_kinematics_inputs["aatype"]
    frames = synthetic_kinematics_inputs["frames"]

    angles = head(h)
    coords, mask = assembly(frames, angles, aatype)

    # Check GLY at [0, 0]: Slot 4 (CB) must be 0
    assert mask[0, 0, 4].item() == 0.0, "Glycine must not have a CB atom"
    assert (coords[0, 0, 4, :] == 0.0).all(), "Unused atom slot in Glycine must be zeroed"

    # Check ALA at [0, 1]: Slot 4 (CB) must be 1, Slots 5..13 must be 0
    assert mask[0, 1, 4].item() == 1.0, "Alanine must have a CB atom"
    assert (mask[0, 1, 5:] == 0.0).all(), "Alanine must not have atoms beyond CB"

    # Check TRP at [0, 2]: All 14 atom slots must be 1
    assert (mask[0, 2, :] == 1.0).all(), "Tryptophan must have all 14 atom slots active"


# =============================================================================
# Gate 4: Backbone Coordinate Preservation
# =============================================================================

def test_backbone_ca_preservation(synthetic_kinematics_inputs):
    """Gate 4: Cα coordinate in slot 1 strictly matches backbone frame translation."""
    cfg = KinematicsConfig()
    head = TorsionHead(cfg)
    assembly = Atom14KinematicsAssembly()

    h = synthetic_kinematics_inputs["h"]
    aatype = synthetic_kinematics_inputs["aatype"]
    frames = synthetic_kinematics_inputs["frames"]

    angles = head(h)
    coords, _ = assembly(frames, angles, aatype)

    # Cα atom is in slot 1
    ca_assembled = coords[:, :, 1, :]
    ca_input = frames.trans

    diff = (ca_assembled - ca_input).abs().max().item()
    print(f"\nMax Cα coordinate discrepancy between frame and Atom14: {diff:.6e} Å")
    assert diff < 1e-4, f"Cα atom was displaced from backbone frame translation: diff = {diff}"


# =============================================================================
# Gate 5: End-to-End Differentiability & Gradient Flow
# =============================================================================

def test_kinematics_differentiability(synthetic_kinematics_inputs):
    """Gate 5: Verify gradients propagate back to both backbone frames and torsion weights."""
    cfg = KinematicsConfig()
    head = TorsionHead(cfg)
    assembly = Atom14KinematicsAssembly()
    loss_fn = StereochemicalLoss(cfg)

    h = synthetic_kinematics_inputs["h"]
    aatype = synthetic_kinematics_inputs["aatype"]
    frames = synthetic_kinematics_inputs["frames"]

    # Allow gradient on backbone translations
    frames.trans.requires_grad_(True)

    angles = head(h)
    coords, mask = assembly(frames, angles, aatype)

    target_coords = coords.detach() + torch.randn_like(coords) * 0.5
    loss_dict = loss_fn(
        pred_atom14=coords,
        target_atom14=target_coords,
        atom14_mask=mask,
        pred_torsions=angles,
    )

    loss = loss_dict["loss"]
    loss.backward()

    # 1. Gradients reach TorsionHead parameters
    assert head.net[-1].weight.grad is not None
    assert not torch.isnan(head.net[-1].weight.grad).any()

    # 2. Gradients reach backbone frames
    assert frames.trans.grad is not None
    assert not torch.isnan(frames.trans.grad).any()


# =============================================================================
# Gate 6: Parameter Budget Compliance
# =============================================================================

def test_torsion_head_parameter_budget():
    """Gate 6: Verify TorsionHead is compact (< 100k parameters)."""
    cfg = KinematicsConfig(d_h=128, d_hidden=128)
    head = TorsionHead(cfg)

    total_params = sum(p.numel() for p in head.parameters() if p.requires_grad)
    print(f"\nTotal TorsionHead parameters: {total_params:,}")

    assert total_params < 100_000, f"TorsionHead has {total_params:,} parameters, exceeding 100k budget!"
