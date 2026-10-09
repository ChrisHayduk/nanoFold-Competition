"""Differentiable SE(3) Rigid Body Geometry and Lie Algebra Operations for NanoFold.

===============================================================================
MATHEMATICAL & BIOLOGICAL FOUNDATIONS:
===============================================================================
Proteins are continuous polymer chains of amino acid residues. Every residue i
possesses a local rigid backbone frame T_i = (R_i, t_i) ∈ SE(3):
  - R_i ∈ SO(3): 3x3 orthonormal rotation matrix describing backbone orientation
                 (derived from N, Cα, C atoms: Cα -> C defines x-axis, Cα -> N
                 defines xy-plane).
  - t_i ∈ R^3:   3D Euclidean coordinate of the residue's Cα atom.

RIGID LIE GROUP UPDATES ON SO(3):
Standard neural networks output unconstrained vectors in R^3 or matrices in R^{3x3}.
If a network directly predicts a 3x3 matrix, it almost never lies on the SO(3)
manifold (det(R) ≠ 1, R^T R ≠ I), producing sheared and unphysical geometries.

Instead, we use the Lie algebra so(3) via the exponential map:
  1. The network predicts an unconstrained tangent update vector:
         Δω = (ω_x, ω_y, ω_z) ∈ R^3
  2. Δω corresponds to a skew-symmetric matrix [Δω]_× ∈ so(3):
         [Δω]_× = [  0   -ω_z   ω_y ]
                  [  ω_z   0   -ω_x ]
                  [ -ω_y  ω_x    0  ]
  3. The exact matrix exponential exp([Δω]_×) ∈ SO(3) is computed via Rodrigues' formula:
         θ = ||Δω||_2
         exp([Δω]_×) = I + (sin θ / θ) [Δω]_× + ((1 - cos θ) / θ^2) [Δω]_×^2
     For small θ (θ < 1e-4), we use the Taylor expansion to guarantee numerical stability:
         exp([Δω]_×) ≈ I + [Δω]_× + 0.5 [Δω]_×^2
  4. The orientation update is strictly manifold-preserving:
         R_{new} = R_{old} · exp([Δω]_×) ∈ SO(3)
===============================================================================
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

# =============================================================================
# 1. SO(3) Lie Algebra Matrix Exponential
# =============================================================================

def skew_symmetric(omega: torch.Tensor) -> torch.Tensor:
    """Constructs skew-symmetric matrices [omega]_x in so(3) from tangent vectors.

    Args:
        omega: [..., 3] Tangent vectors (axis * angle).

    Returns:
        [..., 3, 3] Skew-symmetric matrices.
    """
    wx = omega[..., 0]
    wy = omega[..., 1]
    wz = omega[..., 2]
    zeros = torch.zeros_like(wx)

    # Row 0: [  0, -wz,  wy]
    # Row 1: [ wz,   0, -wx]
    # Row 2: [-wy,  wx,   0]
    row0 = torch.stack([zeros, -wz, wy], dim=-1)
    row1 = torch.stack([wz, zeros, -wx], dim=-1)
    row2 = torch.stack([-wy, wx, zeros], dim=-1)

    return torch.stack([row0, row1, row2], dim=-2)  # [..., 3, 3]


def so3_exp_map(omega: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """Computes orthonormal rotation matrices in SO(3) from 3D update vectors.

    Implements AlphaFold2 Algorithm 23 (Backbone update):
      Maps unconstrained update vector v = (b, c, d) into a unit quaternion
      q = (1, b, c, d) / sqrt(1 + b^2 + c^2 + d^2), then computes the rotation matrix.

    Guarantees:
      1. Strictly positive denominator >= 1.0 (division by zero is impossible).
      2. Zero trigonometric functions (no catastrophic float cancellation).
      3. Infinitely differentiable C^inf and 100% finite gradients across FP32, FP16, BF16.
      4. v = 0 yields exact identity matrix I.

    Args:
        omega: [..., 3] Rotation update vectors (interpreted as (b, c, d)).
        eps: Unused legacy parameter retained for API compatibility.

    Returns:
        R: [..., 3, 3] Orthonormal rotation matrices in SO(3).
    """
    orig_dtype = omega.dtype
    v = omega.float()
    b = v[..., 0]
    c = v[..., 1]
    d = v[..., 2]

    a = torch.ones_like(b)
    norm = torch.sqrt(1.0 + b * b + c * c + d * d)

    a, b, c, d = a / norm, b / norm, c / norm, d / norm

    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    ab, ac, ad = a * b, a * c, a * d
    bc, bd, cd = b * c, b * d, c * d

    r11 = aa + bb - cc - dd
    r12 = 2.0 * (bc - ad)
    r13 = 2.0 * (bd + ac)

    r21 = 2.0 * (bc + ad)
    r22 = aa - bb + cc - dd
    r23 = 2.0 * (cd - ab)

    r31 = 2.0 * (bd - ac)
    r32 = 2.0 * (cd + ab)
    r33 = aa - bb - cc + dd

    row1 = torch.stack([r11, r12, r13], dim=-1)
    row2 = torch.stack([r21, r22, r23], dim=-1)
    row3 = torch.stack([r31, r32, r33], dim=-1)
    R = torch.stack([row1, row2, row3], dim=-2)
    return R.to(dtype=orig_dtype)


# =============================================================================
# 2. Rigid Body Transformations SE(3)
# =============================================================================

@dataclass
class RigidFrames:
    """Rigid transformations in SE(3) representing backbone residue frames.

    Attributes:
        rot: [B, L, 3, 3] Orthonormal rotation matrices in SO(3).
        trans: [B, L, 3] Cα coordinate translations in Euclidean 3D space.
    """
    rot: torch.Tensor
    trans: torch.Tensor

    def compose(self, delta_rot: torch.Tensor, delta_trans: torch.Tensor) -> RigidFrames:
        """Applies local rigid body update: (R_new, t_new) = (R · R_delta, t + R · t_delta)."""
        new_rot = torch.matmul(self.rot, delta_rot)
        # Transform delta_trans into global frame using current orientation
        trans_offset = torch.matmul(self.rot, delta_trans.unsqueeze(-1)).squeeze(-1)
        new_trans = self.trans + trans_offset
        return RigidFrames(rot=new_rot, trans=new_trans)

    def inverse(self) -> RigidFrames:
        """Computes inverse transformation: T^{-1} = (R^T, -R^T t)."""
        rot_inv = self.rot.transpose(-1, -2)
        trans_inv = -torch.matmul(rot_inv, self.trans.unsqueeze(-1)).squeeze(-1)
        return RigidFrames(rot=rot_inv, trans=trans_inv)

    def apply_to_points(self, points: torch.Tensor) -> torch.Tensor:
        """Applies local frame transformation to points in local frame: x_global = R x_local + t.

        Args:
            points: [B, L, ..., 3] Point coordinates.

        Returns:
            [B, L, ..., 3] Transformed point coordinates.
        """
        # rot: [B, L, 3, 3], points: [B, L, 3]
        pts_rot = torch.matmul(self.rot, points.unsqueeze(-1)).squeeze(-1)
        return pts_rot + self.trans

    def invert_apply_to_points(self, points_global: torch.Tensor) -> torch.Tensor:
        """Transforms global points into local frame: x_local = R^T (x_global - t)."""
        diff = points_global - self.trans
        return torch.matmul(self.rot.transpose(-1, -2), diff.unsqueeze(-1)).squeeze(-1)


# =============================================================================
# 3. Initialization: Extended Linear Polypeptide Chain
# =============================================================================

def make_extended_linear_chain(
    batch_size: int,
    seq_len: int,
    bond_length: float = 3.8,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> RigidFrames:
    """Initializes an extended linear polypeptide chain with standard Cα spacing (3.8Å).

    Biological Motivation:
      Initializing all residue translations at (0, 0, 0) creates geometric singularity
      and pairwise distance collapse. An extended chain:
          t_i^{(0)} = (3.8 · i, 0, 0)
          R_i^{(0)} = I_{3x3}
      provides a non-degenerate, well-conditioned initial state for iterative folding.

    Args:
        batch_size: Number of proteins in batch.
        seq_len: Number of residues L.
        bond_length: Typical Cα-Cα virtual bond distance (default: 3.8Å).
        device: Torch device.
        dtype: Float dtype.

    Returns:
        RigidFrames initialized to an extended linear chain.
    """
    device = device or torch.device("cpu")

    # Translations: [B, L, 3] along x-axis
    indices = torch.arange(seq_len, dtype=dtype, device=device)  # [L]
    trans_x = indices * bond_length                              # [L]
    zeros = torch.zeros_like(trans_x)                            # [L]
    trans_single = torch.stack([trans_x, zeros, zeros], dim=-1)  # [L, 3]
    trans = trans_single.unsqueeze(0).expand(batch_size, -1, -1).contiguous()

    # Rotations: [B, L, 3, 3] Identity matrices
    eye = torch.eye(3, dtype=dtype, device=device)
    rot = eye.unsqueeze(0).unsqueeze(0).expand(batch_size, seq_len, -1, -1).contiguous()

    return RigidFrames(rot=rot, trans=trans)


# =============================================================================
# 4. Frame Aligned Point Error (FAPE) & Backbone Frame Loss
# =============================================================================

def compute_fape_loss(
    pred_frames: RigidFrames,
    target_frames: RigidFrames,
    mask: torch.Tensor,
    clamp_distance: float = 10.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    r"""Computes Frame-Aligned Point Error (FAPE) between predicted and native frames.

    FAPE aligns the structure to frame i and measures the Euclidean error at residue j:
        d_{ij} = || T_{pred, i}^{-1} t_{pred, j} - T_{target, i}^{-1} t_{target, j} ||_2
        L_{FAPE} = (1 / Z) \sum_{ij} min(d_{ij}, clamp_distance)

    Invariant to global SE(3) rigid body rotations and translations.

    Args:
        pred_frames: Predicted RigidFrames.
        target_frames: Target native RigidFrames.
        mask: [B, L] Boolean residue validity mask.
        clamp_distance: Maximum distance penalty (clamped at 10Å for stability).
        eps: Small epsilon for sqrt numerical stability.

    Returns:
        Scalar FAPE loss.
    """
    B, L = mask.shape
    valid_mask = mask.float()
    pair_mask = valid_mask.unsqueeze(2) * valid_mask.unsqueeze(1)  # [B, L, L]

    # Perform all geometric alignment and distance calculations in float32 for complete numerical stability
    pred_trans = pred_frames.trans.float()
    pred_rot = pred_frames.rot.float()
    target_trans = target_frames.trans.float()
    target_rot = target_frames.rot.float()

    # Project all positions t_j into each local frame i:
    # diff: [B, L, 1, 3] - [B, 1, L, 3] -> [B, L, L, 3]
    # local_pos_{ij} = R_i^T (t_j - t_i)
    diff_pred = pred_trans.unsqueeze(1) - pred_trans.unsqueeze(2)   # [B, L, L, 3] (t_j - t_i)
    local_pred = torch.matmul(pred_rot.unsqueeze(2).transpose(-1, -2), diff_pred.unsqueeze(-1)).squeeze(-1)

    diff_target = target_trans.unsqueeze(1) - target_trans.unsqueeze(2)
    local_target = torch.matmul(target_rot.unsqueeze(2).transpose(-1, -2), diff_target.unsqueeze(-1)).squeeze(-1)

    # Euclidean distance between frame-aligned points: [B, L, L]
    dist_sq = torch.sum((local_pred - local_target) ** 2, dim=-1)
    dist = torch.sqrt(torch.clamp(dist_sq, min=1e-8) + eps)

    # Clamped loss
    clamped_dist = torch.clamp(dist, max=clamp_distance)

    # Normalized over valid pairs
    loss = (clamped_dist * pair_mask).sum() / pair_mask.sum().clamp(min=1.0)
    return loss.to(dtype=pred_frames.trans.dtype)


def make_backbone_frames_from_atom14(
    atom14_positions: torch.Tensor,
    mask: torch.Tensor,
    eps: float = 1e-8,
) -> RigidFrames:
    r"""Constructs orthonormal backbone RigidFrames from Atom14 positions.

    Uses standard N (slot 0), Cα (slot 1), C (slot 2) coordinates via Gram-Schmidt:
      - Origin t = Cα
      - e1 = (C - Cα) / ||C - Cα||
      - e2 = Gram-Schmidt orthonormalized (N - Cα)
      - e3 = e1 x e2

    Args:
        atom14_positions: [B, L, 14, 3] Heavy atom coordinates.
        mask: [B, L] Residue validity mask.
        eps: Small epsilon to prevent division by zero in normalization.

    Returns:
        RigidFrames containing [B, L, 3, 3] rotations and [B, L, 3] Cα translations.
    """
    n = atom14_positions[..., 0, :].float()
    ca = atom14_positions[..., 1, :].float()
    c = atom14_positions[..., 2, :].float()

    t = ca

    v1 = c - ca
    e1 = v1 / (torch.norm(v1, dim=-1, keepdim=True) + eps)

    v2 = n - ca
    u2 = v2 - torch.sum(v2 * e1, dim=-1, keepdim=True) * e1
    e2 = u2 / (torch.norm(u2, dim=-1, keepdim=True) + eps)

    e3 = torch.cross(e1, e2, dim=-1)
    rot = torch.stack([e1, e2, e3], dim=-1)  # [B, L, 3, 3]

    return RigidFrames(rot=rot.to(atom14_positions.dtype), trans=t.to(atom14_positions.dtype))


def compute_smooth_lddt_loss(
    pred_ca: torch.Tensor,
    target_ca: torch.Tensor,
    mask: torch.Tensor,
    cutoff: float = 15.0,
    eps: float = 1e-8,
) -> torch.Tensor:
    r"""Computes differentiable Smooth lDDT loss between predicted and native Cα coordinates.

    Inspired by AlphaFold3 / Protenix Algorithm 27:
      Evaluates distance differences |d_{pred, ij} - d_{target, ij}| across all native pairs within 15Å.
      Thresholds at {0.5, 1.0, 2.0, 4.0}Å using smooth sigmoids.
      Directly aligns training loss with the FoldScore / lDDT evaluation metric.

    Args:
        pred_ca: [B, L, 3] Predicted Cα coordinates.
        target_ca: [B, L, 3] Target native Cα coordinates.
        mask: [B, L] Boolean residue validity mask.
        cutoff: Inclusion distance in native structure (default: 15.0 Å).
        eps: Numerical epsilon.

    Returns:
        Scalar smooth lDDT loss (1.0 - smooth_lddt_score).
    """
    pred_f = pred_ca.float()
    target_f = target_ca.float()
    valid_f = mask.float()

    pair_mask = (valid_f.unsqueeze(2) * valid_f.unsqueeze(1)).bool()  # [B, L, L]
    L = pred_ca.shape[1]
    eye = torch.eye(L, device=pred_ca.device, dtype=torch.bool).unsqueeze(0)

    d_pred = torch.cdist(pred_f, pred_f)      # [B, L, L]
    d_target = torch.cdist(target_f, target_f)  # [B, L, L]

    inclusion = pair_mask & ~eye & (d_target < cutoff)

    dist_diff = torch.abs(d_pred - d_target)
    score = torch.zeros_like(dist_diff)
    for threshold in (0.5, 1.0, 2.0, 4.0):
        score = score + 0.25 * torch.sigmoid((threshold - dist_diff) / 0.1)

    n_valid = inclusion.sum().clamp(min=1.0)
    lddt = (score * inclusion.float()).sum() / n_valid
    return 1.0 - lddt

