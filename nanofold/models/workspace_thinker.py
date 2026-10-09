"""Phase D: Recurrent SE(3) Structural Workspace Thinker for NanoFold.

===============================================================================
BIOLOGICAL & ARCHITECTURAL FOUNDATIONS:
===============================================================================
Full AlphaFold2 / OpenFold relies on 8 distinct, non-shared Invariant Point
Attention (IPA) blocks to construct 3D coordinates. In a data-scarce benchmark
like NanoFold (10,000 proteins, 30k steps), non-shared layer stacks suffer from
severe sample inefficiency and gradient vanishing in early blocks.

Phase D introduces a structurally inspired alternative:
A SINGLE WEIGHT-SHARED RECURRENT WORKSPACE BLOCK F_θ:

                          F_1 ≡ F_2 ≡ F_3 ≡ ... ≡ F_R ≡ F_θ

At each reasoning iteration r ∈ {0, ..., R - 1}, the model maintains a physical
state for every residue i:
              H_i^{(r)} = (h_i^{(r)}, R_i^{(r)}, t_i^{(r)})
  - h_i^{(r)} ∈ R^{d_h}: Latent workspace representation
  - R_i^{(r)} ∈ SO(3):   Current backbone orientation frame
  - t_i^{(r)} ∈ R^3:     Current Cα 3D coordinate

DETERMINISTIC EXTENDED INITIALIZATION:
  Instead of collapsing all residues to (0, 0, 0), the chain begins as an extended
  linear polypeptide:
              t_i^{(0)} = (3.8 · i, 0, 0) Å
              R_i^{(0)} = I_{3x3}
  The recurrent block F_θ iteratively folds this extended chain into native geometry.

TRIPLE CONDITIONING:
  The shared block F_θ is jointly conditioned on:
    1. Single representation s_i (sequence & evolutionary context from Phase A)
    2. Pair representation z_{ij} (pairwise relational geometry from Phase A)
    3. Predicted tokens q̂_{1:16} (global structural imagination from Phase C)
    4. Step embedding r (reasoning progress)

RIGID LIE ALGEBRA SO(3) EXPONENTIAL MAP:
  The block outputs updates (Δh_i, Δt_i, Δω_i). Rotations are updated strictly
  on the SO(3) manifold via the Rodrigues exponential map:
              R_i^{(r+1)} = R_i^{(r)} · exp([α_r Δω_i]_×) ∈ SO(3)
              t_i^{(r+1)} = t_i^{(r)} + α_r Δt_i ∈ R^3

TEST-TIME REASONING HYPOTHESIS:
  Because F_θ is weight-shared, we can run it for R ∈ {1, 2, 4, 8} iterations
  at test time. A genuinely learned iterative thinker should exhibit monotonic
  improvement:
              L(R=8) ≤ L(R=4) < L(R=2) < L(R=1)
===============================================================================
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from nanofold.models.geometry_se3 import (
    RigidFrames,
    compute_fape_loss,
    so3_exp_map,
)

# =============================================================================
# 1. Configuration Dataclass
# =============================================================================

@dataclass
class WorkspaceThinkerConfig:
    """Hyperparameters configuring the Recurrent Structural Workspace Thinker."""
    d_h: int = 128               # Dimension of latent residue workspace
    d_single: int = 128          # Dimension of single representation from Pairformer
    d_pair: int = 64             # Dimension of pair representation from Pairformer
    d_tok: int = 64              # Dimension of global structural tokens from TokenPredictor
    k_tokens: int = 16           # Number of global structural tokens K = 16
    num_steps: int = 4           # Default number of recurrent thinking steps R = 4
    n_heads: int = 4             # Number of attention heads in workspace self-attention
    step_weights: Tuple[float, float, float, float] = (0.1, 0.2, 0.3, 1.0)  # Multi-step supervision weights
    trans_scale: float = 2.5     # Step size scaling factor for translation updates
    rot_scale: float = 0.5       # Step size scaling factor for rotation updates


# =============================================================================
# 2. Sinusoidal Step Embedding
# =============================================================================

class SinusoidalStepEmbedding(nn.Module):
    """Encodes recurrence iteration index r into continuous frequency features."""

    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        half_dim = d_model // 2
        freqs = torch.exp(-math.log(10000.0) * torch.arange(half_dim, dtype=torch.float32) / (half_dim - 1))
        self.register_buffer("freqs", freqs)
        self.proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

    def forward(self, step_idx: int, device: torch.device) -> torch.Tensor:
        """Returns [1, 1, d_model] embedding for step r."""
        freqs = self.freqs.to(device)
        step_val = torch.tensor([step_idx], dtype=torch.float32, device=device)
        args = step_val.unsqueeze(-1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        return self.proj(emb).unsqueeze(1)  # [1, 1, d_model]


# =============================================================================
# 3. Shared Recurrent Reasoning Block F_θ
# =============================================================================

class WorkspaceReasoningBlock(nn.Module):
    """The single weight-shared recurrent block F_θ applied at every step r."""

    def __init__(self, cfg: WorkspaceThinkerConfig):
        super().__init__()
        self.cfg = cfg
        self.d_h = cfg.d_h
        self.n_heads = cfg.n_heads
        self.head_dim = self.d_h // self.n_heads

        # 1. Residue self-attention with pairwise bias from z_ij
        self.norm_h = nn.LayerNorm(self.d_h)
        self.q_proj = nn.Linear(self.d_h, self.d_h, bias=False)
        self.k_proj = nn.Linear(self.d_h, self.d_h, bias=False)
        self.v_proj = nn.Linear(self.d_h, self.d_h, bias=False)
        self.pair_bias_proj = nn.Linear(cfg.d_pair, self.n_heads, bias=False)
        self.attn_out = nn.Linear(self.d_h, self.d_h, bias=False)

        # 2. Cross-attention to global structural tokens q̂_{1:16}
        self.norm_cross = nn.LayerNorm(self.d_h)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=self.d_h,
            kdim=cfg.d_tok,
            vdim=cfg.d_tok,
            num_heads=self.n_heads,
            batch_first=True,
        )

        # 3. Transition MLP
        self.norm_mlp = nn.LayerNorm(self.d_h)
        self.mlp = nn.Sequential(
            nn.Linear(self.d_h, 2 * self.d_h),
            nn.GELU(),
            nn.Linear(2 * self.d_h, self.d_h),
        )

        # 4. Geometric Update Heads: Δh, Δt, Δω
        self.head_dh = nn.Linear(self.d_h, self.d_h)
        self.head_dt = nn.Linear(self.d_h, 3)
        self.head_domega = nn.Linear(self.d_h, 3)

        # Initialize geometric update heads:
        # Normal initialization for dt allows active spatial expansion away from (0, 0, 0)
        nn.init.normal_(self.head_dt.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.head_dt.bias)
        nn.init.zeros_(self.head_domega.weight)
        nn.init.zeros_(self.head_domega.bias)

    def forward(
        self,
        h: torch.Tensor,
        z: torch.Tensor,
        q_tokens: torch.Tensor,
        step_emb: torch.Tensor,
        mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Performs one recurrent reasoning pass.

        Args:
            h: [B, L, d_h] Current latent workspace representations.
            z: [B, L, L, d_pair] Pairwise geometric representations from Phase A.
            q_tokens: [B, 16, d_tok] Structural tokens from Phase C.
            step_emb: [1, 1, d_h] Recurrence step embedding.
            mask: [B, L] Boolean residue validity mask.

        Returns:
            delta_h: [B, L, d_h] Latent workspace update.
            delta_trans: [B, L, 3] Coordinate translation update.
            delta_omega: [B, L, 3] Axis-angle rotation update in so(3).
        """
        B, L, _ = h.shape

        # Inject recurrence step embedding
        h_step = h + step_emb

        # 1. Residue Self-Attention with Pair Bias
        h_norm = self.norm_h(h_step)
        q = self.q_proj(h_norm).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(h_norm).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(h_norm).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)

        # Pair bias: [B, L, L, n_heads] -> [B, n_heads, L, L]
        pair_bias = self.pair_bias_proj(z).permute(0, 3, 1, 2)
        scale = 1.0 / math.sqrt(self.head_dim)
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale + pair_bias

        # Mask invalid residues
        padding_mask = (~mask).unsqueeze(1).unsqueeze(2)  # [B, 1, 1, L]
        scores = scores.masked_fill(padding_mask, float("-inf"))
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0)

        attn_out = torch.matmul(attn_weights, v).transpose(1, 2).contiguous().view(B, L, self.d_h)
        h = h + self.attn_out(attn_out)

        # 2. Cross-Attention to Global Structural Tokens
        h_norm2 = self.norm_cross(h)
        cross_out, _ = self.cross_attn(
            query=h_norm2,
            key=q_tokens,
            value=q_tokens,
        )
        h = h + cross_out

        # 3. Transition MLP
        h = h + self.mlp(self.norm_mlp(h))

        # 4. Geometric Update Heads
        delta_h = self.head_dh(h) * mask.unsqueeze(-1).float()
        delta_trans = (self.head_dt(h) * self.cfg.trans_scale) * mask.unsqueeze(-1).float()
        delta_omega = (self.head_domega(h) * self.cfg.rot_scale) * mask.unsqueeze(-1).float()

        return delta_h, delta_trans, delta_omega


# =============================================================================
# 4. Recurrent Structural Workspace Thinker
# =============================================================================

class RecurrentWorkspaceThinker(nn.Module):
    """The Complete Recurrent Structural Workspace Thinker (Phase D).

    Manages the recurrent iterative reasoning loop across R steps.
    """

    def __init__(self, cfg: Optional[WorkspaceThinkerConfig] = None):
        super().__init__()
        self.cfg = cfg or WorkspaceThinkerConfig()
        self.d_h = self.cfg.d_h

        # Input integration: fuses Phase A s_i and global token summary
        self.init_fuse = nn.Sequential(
            nn.Linear(self.cfg.d_single + self.cfg.d_tok, self.d_h),
            nn.GELU(),
            nn.LayerNorm(self.d_h),
        )

        # Step embedding
        self.step_embed = SinusoidalStepEmbedding(self.d_h)

        # The single weight-shared reasoning block F_θ
        self.reasoning_block = WorkspaceReasoningBlock(self.cfg)

    def init_state(
        self,
        s: torch.Tensor,
        q_tokens: torch.Tensor,
        mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, RigidFrames]:
        """Initializes the physical workspace state H^{(0)} = (h^{(0)}, R^{(0)}, t^{(0)}).

        Args:
            s: [B, L, d_single] Single residue features from Phase A.
            q_tokens: [B, 16, d_tok] Structural tokens from Phase C.
            mask: [B, L] Residue mask.

        Returns:
            h_0: [B, L, d_h] Initial latent workspace.
            frames_0: RigidFrames with extended linear chain.
        """
        B, L, _ = s.shape
        device = s.device

        # Global token context: average over 16 slots [B, 1, d_tok]
        q_summary = q_tokens.mean(dim=1, keepdim=True).expand(-1, L, -1)
        fused = torch.cat([s, q_summary], dim=-1)
        h_0 = self.init_fuse(fused) * mask.unsqueeze(-1).float()

        # Standard AlphaFold 2 structural initialization: t_i = (0, 0, 0), R_i = I
        eye = torch.eye(3, dtype=s.dtype, device=device)
        rot_0 = eye.unsqueeze(0).unsqueeze(0).expand(B, L, -1, -1).contiguous()
        trans_0 = torch.zeros((B, L, 3), dtype=s.dtype, device=device)
        frames_0 = RigidFrames(rot=rot_0, trans=trans_0)

        return h_0, frames_0

    def forward(
        self,
        s: torch.Tensor,
        z: torch.Tensor,
        q_tokens: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        eval_steps: Optional[int] = None,
    ) -> Dict[str, object]:
        """Executes recurrent structural reasoning across R steps.

        Args:
            s: [B, L, d_single] Single residue features from Phase A.
            z: [B, L, L, d_pair] Pairwise geometric features from Phase A.
            q_tokens: [B, 16, d_tok] Predicted continuous tokens from Phase C.
            mask: [B, L] Boolean residue mask.
            eval_steps: Optional step count override for test-time compute evaluation (e.g. 1, 2, 4, 8).

        Returns:
            dict containing:
              - final_frames: RigidFrames at step R.
              - trajectory: List[RigidFrames] of intermediate frames for each step r.
              - num_steps: Actual number of reasoning steps executed.
        """
        B, L, _ = s.shape
        device = s.device

        if mask is None:
            mask = torch.ones((B, L), dtype=torch.bool, device=device)
        else:
            mask = mask.bool()

        num_steps = eval_steps if eval_steps is not None else self.cfg.num_steps

        # 1. Initialize physical workspace: H^{(0)}
        h, frames = self.init_state(s, q_tokens, mask)

        trajectory: List[RigidFrames] = []

        # 2. Recurrent reasoning loop: F_θ applied R times with shared weights
        for r in range(num_steps):
            step_emb = self.step_embed(step_idx=r, device=device)

            # Apply shared block F_θ
            delta_h, delta_t, delta_omega = self.reasoning_block(
                h=h,
                z=z,
                q_tokens=q_tokens,
                step_emb=step_emb,
                mask=mask,
            )

            # Latent update: h^{(r+1)} = h^{(r)} + Δh
            h = h + delta_h

            # Rigid Lie Group update: R^{(r+1)} = R^{(r)} · exp([Δω]_×), t^{(r+1)} = t^{(r)} + Δt
            delta_rot = so3_exp_map(delta_omega)
            frames = frames.compose(delta_rot=delta_rot, delta_trans=delta_t)

            trajectory.append(frames)

        return {
            "final_frames": frames,
            "trajectory": trajectory,
            "num_steps": num_steps,
        }

    def compute_loss(
        self,
        trajectory: List[RigidFrames],
        target_frames: RigidFrames,
        mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        r"""Computes multi-step Frame-Aligned Point Error (FAPE) loss across the trajectory.

        L_D = \sum_{r=1}^R w_r L_{FAPE}^{(r)}

        Args:
            trajectory: List of RigidFrames predicted at each iteration r = 1..R.
            target_frames: Ground-truth native RigidFrames.
            mask: [B, L] Residue mask.

        Returns:
            dict containing:
              - total_loss: Weighted sum of multi-step FAPE losses.
              - step_losses: List of scalar FAPE losses per step.
        """
        step_losses: List[torch.Tensor] = []
        num_steps = len(trajectory)

        # Adjust step weights if num_steps differs from config
        if num_steps == len(self.cfg.step_weights):
            weights = list(self.cfg.step_weights)
        else:
            # Linear ramp up to 1.0
            weights = [float(r + 1) / num_steps for r in range(num_steps)]

        total_loss = torch.tensor(0.0, device=mask.device)

        for r, pred_frames in enumerate(trajectory):
            fape = compute_fape_loss(pred_frames, target_frames, mask)
            step_losses.append(fape)
            total_loss = total_loss + weights[r] * fape

        return {
            "loss": total_loss,
            "step_losses": [loss.item() for loss in step_losses],
            "final_fape": step_losses[-1].detach(),
        }
