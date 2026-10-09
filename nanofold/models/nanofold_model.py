"""Phase F: Complete End-to-End NanoFold Architecture.

===============================================================================
THE COMPLETE 4-ACTION BIOLOGICAL ARCHITECTURE:
===============================================================================
NanoFold integrates the four core actions of protein structure reasoning:

  1. UNDERSTAND (Phase A):
     InputEmbedder + 6-block MicroPairformer:
     Raw sequence + true MSA coevolution -> s_i ∈ R^{128}, z_ij ∈ R^{64}

  2. IMAGINE (Phase C):
     Perceiver-style TokenPredictor:
     16 learned query slots cross-attend over pair-conditioned residue features:
     (s_i, z_ij) -> Continuous Structural Tokens q̂_{1:16} ∈ R^{16 x 64}

  3. THINK (Phase D):
     Recurrent SE(3) Structural Workspace Thinker:
     Weight-shared block F_θ applied across R=4 iterations on rigid frames
     T_i = (R_i, t_i) ∈ SE(3) initialized as an extended linear chain:
     (s_i, z_ij, q̂) -> Evolved Backbone Frames (R_i, t_i) + Workspace h_i

  4. CONSTRUCT (Phase E):
     TorsionHead + Atom14KinematicsAssembly:
     Forward kinematics along the side-chain torsion tree (Algorithm 24):
     (R_i, t_i, h_i) -> All 14 Heavy Atoms pred_atom14 ∈ R^{B x L x 14 x 3}

TOTAL PARAMETER FOOTPRINT:
  ~2.83M parameters (strictly within the < 3.0M budget).
===============================================================================
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from nanofold.models.geometry_se3 import (
    RigidFrames,
    compute_fape_loss,
    compute_smooth_lddt_loss,
    make_backbone_frames_from_atom14,
)
from nanofold.models.kinematics import (
    Atom14KinematicsAssembly,
    KinematicsConfig,
    TorsionHead,
)
from nanofold.models.pairformer import (
    ContactHead,
    DistogramHead,
    MicroPairformer,
    PairformerConfig,
)
from nanofold.models.token_predictor import (
    PredictorConfig,
    TokenPredictor,
)
from nanofold.models.workspace_thinker import (
    RecurrentWorkspaceThinker,
    WorkspaceThinkerConfig,
)

# =============================================================================
# 1. Complete Model Configuration Dataclass
# =============================================================================

@dataclass
class NanoFoldConfig:
    """Hyperparameters configuring the entire integrated NanoFold architecture."""
    pairformer: PairformerConfig = field(default_factory=lambda: PairformerConfig(n_blocks=8))
    predictor: PredictorConfig = field(default_factory=PredictorConfig)
    thinker: WorkspaceThinkerConfig = field(default_factory=lambda: WorkspaceThinkerConfig(d_h=160))
    kinematics: KinematicsConfig = field(default_factory=KinematicsConfig)

    # Loss weights for joint multi-task training
    loss_weight_fape: float = 1.0        # Backbone multi-step FAPE weight
    loss_weight_lddt: float = 1.0        # Differentiable Smooth lDDT weight
    loss_weight_atom14: float = 0.2      # Centered all-atom Huber coordinate weight
    loss_weight_distogram: float = 0.3   # Auxiliary distogram CE weight
    loss_weight_torsion: float = 0.5     # Side-chain torsion angle weight


# =============================================================================
# 2. Integrated NanoFold Model Module
# =============================================================================

class NanoFoldModel(nn.Module):
    """The Complete Integrated NanoFold Architecture.

    Wires Phase A -> Phase C -> Phase D -> Phase E into a unified differentiable system.
    """

    def __init__(self, cfg: Optional[NanoFoldConfig] = None):
        super().__init__()
        self.cfg = cfg or NanoFoldConfig()

        # Phase A: Micro-Pairformer Trunk (~1.78M params)
        self.pairformer = MicroPairformer(self.cfg.pairformer)

        # Auxiliary heads from pair representations
        self.distogram_head = DistogramHead(
            d_pair=self.cfg.pairformer.d_pair,
            num_bins=self.cfg.pairformer.distogram_bins,
        )
        self.contact_head = ContactHead(d_pair=self.cfg.pairformer.d_pair)

        # Phase C: JEPA Structural Token Predictor (~51k params)
        self.token_predictor = TokenPredictor(self.cfg.predictor)

        # Phase D: Recurrent SE(3) Structural Workspace Thinker (~257k params)
        self.workspace_thinker = RecurrentWorkspaceThinker(self.cfg.thinker)

        # Phase E: Stereochemical Torsion Head + Atom14 Kinematics (~35k params)
        self.torsion_head = TorsionHead(self.cfg.kinematics)
        self.atom14_assembly = Atom14KinematicsAssembly()

    def forward(
        self,
        aatype: torch.Tensor,
        msa: torch.Tensor,
        deletions: torch.Tensor,
        residue_mask: torch.Tensor,
        residue_index: Optional[torch.Tensor] = None,
        eval_steps: Optional[int] = None,
    ) -> Dict[str, object]:
        """Complete forward pass from sequence + MSA to 3D Atom14 coordinates.

        Args:
            aatype: [B, L] Integer amino acid sequence indices.
            msa: [B, S, L] Raw MSA sequence tokens.
            deletions: [B, S, L] Float deletion matrix.
            residue_mask: [B, L] Boolean residue validity mask.
            residue_index: Optional [B, L] residue sequence numbers (defaults to 0..L-1).
            eval_steps: Optional recurrent reasoning steps override (e.g. 1, 2, 4, 8).

        Returns:
            dict containing:
              - pred_atom14: [B, L, 14, 3] All 14 heavy atom coordinates in Å.
              - atom14_mask: [B, L, 14] Binary validity mask.
              - final_frames: RigidFrames containing backbone (R_i, t_i).
              - tokens_pred: [B, 16, d_tok] Predicted continuous structural tokens.
              - trajectory: List[RigidFrames] of intermediate frames per thinking step.
              - distogram_logits: [B, L, L, 64] Auxiliary pairwise distance logits.
              - contact_logits: [B, L, L] Auxiliary pairwise contact logits.
        """
        B, L = aatype.shape
        device = aatype.device

        if residue_index is None:
            residue_index = torch.arange(L, device=device, dtype=torch.long).unsqueeze(0).expand(B, -1)

        # 1. UNDERSTAND: Micro-Pairformer Trunk
        s, z = self.pairformer(
            aatype=aatype,
            msa=msa,
            deletions=deletions,
            residue_index=residue_index,
            residue_mask=residue_mask,
        )

        # Auxiliary pairwise predictions
        dist_logits = self.distogram_head(z)
        cont_logits = self.contact_head(z)

        # 2. IMAGINE: JEPA Structural Token Predictor
        q_tokens = self.token_predictor(s, z, mask=residue_mask)

        # 3. THINK: Recurrent SE(3) Workspace Thinker
        thinker_out = self.workspace_thinker(
            s=s,
            z=z,
            q_tokens=q_tokens,
            mask=residue_mask,
            eval_steps=eval_steps,
        )
        final_frames: RigidFrames = thinker_out["final_frames"]
        trajectory: List[RigidFrames] = thinker_out["trajectory"]

        # 4. CONSTRUCT: Torsion Head + Atom14 Assembly
        torsion_angles = self.torsion_head(s)  # [B, L, 7, 2]
        pred_atom14, atom14_mask = self.atom14_assembly(
            backbone_frames=final_frames,
            torsion_angles=torsion_angles,
            aatype=aatype,
        )

        # Mask padding residues
        res_mask_f = residue_mask.unsqueeze(-1).unsqueeze(-1).float()
        pred_atom14 = pred_atom14 * res_mask_f

        return {
            "pred_atom14": pred_atom14,
            "atom14_mask": atom14_mask,
            "final_frames": final_frames,
            "tokens_pred": q_tokens,
            "trajectory": trajectory,
            "torsion_angles": torsion_angles,
            "distogram_logits": dist_logits,
            "contact_logits": cont_logits,
            "s": s,
            "z": z,
        }


# =============================================================================
# 3. Multi-Task Training Loss Suite
# =============================================================================

class NanoFoldLoss(nn.Module):
    """Computes the joint multi-task loss for training the integrated NanoFoldModel."""

    def __init__(self, cfg: Optional[NanoFoldConfig] = None):
        super().__init__()
        self.cfg = cfg or NanoFoldConfig()

    def forward(
        self,
        model_out: Dict[str, object],
        batch: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Calculates multi-task training loss against target ground-truth data.

        Integrates:
          1. Multi-Step Backbone FAPE (AlphaFold 2 Algorithm 20):
             Supervises both translations and orientations across all thinker steps.
          2. Differentiable Smooth lDDT (AlphaFold 3 / Protenix Algorithm 27):
             Directly aligns learning with CASP15 lDDT evaluation metric.
          3. Centered All-Atom Huber Coordinate Loss:
             Subtracts Cα centroids to eliminate the 10Å coordinate barrier and FP16 overflow.
          4. Auxiliary Distogram Cross-Entropy:
             Guides pair coevolutionary distance bins.

        Args:
            model_out: Dictionary output from NanoFoldModel.forward.
            batch: Batch dictionary with ground truth coordinates, masks, etc.

        Returns:
            dict containing total 'loss' and individual component losses.
        """
        pred_atom14: torch.Tensor = model_out["pred_atom14"]  # [B, L, 14, 3]
        target_atom14: torch.Tensor = batch["atom14_positions"]  # [B, L, 14, 3]
        res_mask = batch["residue_mask"].bool()  # [B, L]
        target_atom_mask = batch["atom14_mask"].float() * res_mask.unsqueeze(-1).float()  # [B, L, 14]

        # 1. Multi-Step Backbone FAPE Loss on SE(3) Thinker Trajectory
        target_frames = make_backbone_frames_from_atom14(target_atom14, res_mask)
        trajectory = model_out.get("trajectory", None)
        if trajectory is not None and isinstance(trajectory, list) and len(trajectory) > 0:
            fape_loss = torch.tensor(0.0, device=pred_atom14.device)
            num_steps = len(trajectory)
            weight_sum = sum(range(1, num_steps + 1))
            for r, frames_r in enumerate(trajectory):
                w_r = float(r + 1) / weight_sum
                fape_r = compute_fape_loss(frames_r, target_frames, res_mask, clamp_distance=10.0)
                fape_loss = fape_loss + w_r * fape_r
        else:
            final_frames = model_out.get("final_frames", None)
            if final_frames is not None:
                fape_loss = compute_fape_loss(final_frames, target_frames, res_mask, clamp_distance=10.0)
            else:
                fape_loss = torch.tensor(0.0, device=pred_atom14.device)

        # 2. Differentiable Smooth lDDT Loss on Cα (slot 1)
        pred_ca = pred_atom14[:, :, 1, :]
        target_ca = target_atom14[:, :, 1, :]
        lddt_loss = compute_smooth_lddt_loss(pred_ca, target_ca, res_mask, cutoff=15.0)

        # 3. Centered All-Atom Huber Coordinate Loss
        # Subtract Cα centroid so coordinates are scale-invariant and never overflow FP16
        mask_f = res_mask.unsqueeze(-1).float()  # [B, L, 1]
        n_res = mask_f.sum(dim=1, keepdim=True).clamp(min=1.0)
        centroid_pred = (pred_ca.float() * mask_f).sum(dim=1, keepdim=True) / n_res  # [B, 1, 3]
        centroid_tgt = (target_ca.float() * mask_f).sum(dim=1, keepdim=True) / n_res  # [B, 1, 3]

        pred_centered = pred_atom14.float() - centroid_pred.unsqueeze(2)  # [B, L, 14, 3]
        tgt_centered = target_atom14.float() - centroid_tgt.unsqueeze(2)  # [B, L, 14, 3]

        huber = F.smooth_l1_loss(pred_centered, tgt_centered, beta=1.0, reduction="none").mean(dim=-1)  # [B, L, 14]
        atom_loss = (huber * target_atom_mask).sum() / target_atom_mask.sum().clamp(min=1.0)

        # 4. Auxiliary Distogram Cross-Entropy Loss (Cβ-Cβ distance)
        dist_logits = model_out["distogram_logits"]  # [B, L, L, 64]
        ca_coords = batch["ca_coords"]               # [B, L, 3]
        ca_mask = batch["ca_mask"].float()           # [B, L]
        pair_ca_mask = (ca_mask.unsqueeze(2) * ca_mask.unsqueeze(1)).bool()

        d_ca = torch.norm(ca_coords.unsqueeze(2) - ca_coords.unsqueeze(1), dim=-1)
        eye = torch.eye(d_ca.shape[1], device=d_ca.device, dtype=torch.bool).unsqueeze(0)
        valid_pair = pair_ca_mask & ~eye

        bin_w = (22.0 - 2.0) / 63.0
        gt_bins = ((d_ca - 2.0) / bin_w).long().clamp(0, 63)
        dist_loss = F.cross_entropy(dist_logits[valid_pair], gt_bins[valid_pair])

        total_loss = (
            self.cfg.loss_weight_fape * fape_loss +
            self.cfg.loss_weight_lddt * lddt_loss +
            self.cfg.loss_weight_atom14 * atom_loss +
            self.cfg.loss_weight_distogram * dist_loss
        )

        return {
            "loss": total_loss,
            "loss_fape": fape_loss.detach(),
            "loss_lddt": lddt_loss.detach(),
            "loss_atom14": atom_loss.detach(),
            "loss_distogram": dist_loss.detach(),
        }
