"""Phase C: Sequence-to-Structural Token Predictor (JEPA Predictor) for NanoFold.

===============================================================================
BIOLOGICAL & MATHEMATICAL FOUNDATIONS:
===============================================================================
In LeCun's Joint Embedding Predictive Architecture (JEPA), a model never tries to
predict raw, noisy, sub-angstrom coordinates directly from sequence representations.
Instead, it predicts in an abstract, compact, SE(3)-invariant structural latent space:

                    Trunk Embeddings (s_i, z_ij)  [From Phase A]
                                  │
                                  ▼
                         Pair-to-Residue Pooling
                                  │
                                  ▼
                         Conditioned Residue h_i
                                  │
                                  ▼
                    16 Learned Student Query Slots
                                  │
                           Cross-Attention
                                  │
                                  ▼
                     Predicted Tokens  q̂ ∈ R^{16 x 64}
                                  │
                                  ▼
                 L_JEPA(q̂, stop_gradient(q^*))   [Toward Phase B Teacher]

KEY PRINCIPLES LOCKED FOR PHASE C:
  1. Global-Query Philosophy:
     Instead of a flat MLP over Pairformer features, 16 learned student query slots
     cross-attend over residue representations conditioned by pairwise geometry.
     This gives the student and teacher matching Perceiver bottleneck geometry
     without sharing weights.
  2. Strict Stop-Gradient on Teacher Targets:
     q^* = stop_gradient(E_structure(X_native)).
     ∇_teacher L_PhaseC = 0.
     The teacher's latent manifold is defined by 3D structural reconstruction.
     The student must learn to imagine that manifold; the student must NEVER pull
     the teacher manifold toward an easy sequence triviality.
  3. Curriculum Prefix Loss Weighting:
     Because Phase B shows early tokens (q_{1:4}) concentrate global topology,
     we supervise slots hierarchically:
         w_{1:4}  = 1.00  (Global topology anchor)
         w_{5:8}  = 0.75  (Secondary packing)
         w_{9:16} = 0.50  (Fine geometric details)
  4. Composite Latent Metric:
     L_k = λ_MSE ||q̂_k - q^*_k||_2^2 + λ_cos (1 - cos(q̂_k, q^*_k)).
===============================================================================
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# =============================================================================
# Configuration Dataclass
# =============================================================================

@dataclass
class PredictorConfig:
    """Hyperparameters configuring the JEPA Structural Token Predictor (Phase C)."""
    d_single: int = 128            # Dimension of single representation s_i from Pairformer
    d_pair: int = 64               # Dimension of pair representation z_ij from Pairformer
    d_tok: int = 64                # Dimension of global structural tokens q_k (matching Phase B)
    k_max: int = 16                # Number of global structural tokens K = 16
    n_heads: int = 4               # Number of cross-attention heads
    loss_weight_mse: float = 1.0   # Weight for L2 MSE latent distance
    loss_weight_cos: float = 1.0   # Weight for Cosine distance (1 - cos)
    prefix_weights: Tuple[float, float, float] = (1.0, 0.75, 0.5)  # Weights for (q_1..4, q_5..8, q_9..16)


# =============================================================================
# 1. JEPA Token Predictor Module
# =============================================================================

class TokenPredictor(nn.Module):
    """Predicts continuous structural tokens q̂_{1:16} from sequence/MSA trunk features.

    Architecture:
      1. Aggregates pairwise coevolution context z_{ij} into single residue features s_i:
             z_pooled_i = (1 / L_valid) * sum_j [ LayerNorm(z_{ij}) W_pair ]
             h_i = LayerNorm(s_i + z_pooled_i)
      2. 16 learnable Perceiver query slots cross-attend over h_i to generate q̂.
      3. Outputs continuous tokens normalized via LayerNorm in R^{B x 16 x d_tok}.
    """

    def __init__(self, cfg: Optional[PredictorConfig] = None):
        super().__init__()
        self.cfg = cfg or PredictorConfig()
        self.d_single = self.cfg.d_single
        self.d_pair = self.cfg.d_pair
        self.d_tok = self.cfg.d_tok
        self.k_max = self.cfg.k_max

        # Pair-to-residue context integration
        self.pair_norm = nn.LayerNorm(self.d_pair)
        self.pair_pool_proj = nn.Linear(self.d_pair, self.d_single)
        self.fuse_norm = nn.LayerNorm(self.d_single)

        # 16 learnable student query slots in R^{16 x d_tok}
        self.query_slots = nn.Parameter(torch.randn(self.k_max, self.d_tok) * 0.02)

        # Cross-attention: 16 student slots attend over single residue representations h_i
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=self.d_tok,
            kdim=self.d_single,
            vdim=self.d_single,
            num_heads=self.cfg.n_heads,
            batch_first=True,
        )
        self.token_norm = nn.LayerNorm(self.d_tok)
        self.token_mlp = nn.Sequential(
            nn.Linear(self.d_tok, 2 * self.d_tok),
            nn.GELU(),
            nn.Linear(2 * self.d_tok, self.d_tok),
        )
        self.final_norm = nn.LayerNorm(self.d_tok)

    def forward(
        self,
        s: torch.Tensor,
        z: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass for TokenPredictor.

        Args:
            s: [B, L, d_single] Single residue features from Pairformer.
            z: [B, L, L, d_pair] Pairwise geometric features from Pairformer.
            mask: [B, L] Optional boolean or float residue validity/padding mask.

        Returns:
            q_pred: [B, 16, d_tok] Predicted continuous structural tokens.
        """
        B, L, _ = s.shape
        device = s.device

        if mask is None:
            mask = torch.ones((B, L), dtype=torch.bool, device=device)
        else:
            mask = mask.bool()

        valid_mask_f = mask.float()  # [B, L]
        pair_mask = valid_mask_f.unsqueeze(2) * valid_mask_f.unsqueeze(1)  # [B, L, L]

        # 1. Integrate pair geometry context into residue representations
        # Apply LayerNorm and project, masking padding elements to avoid bias leakage
        z_normed = self.pair_norm(z) * pair_mask.unsqueeze(-1)
        z_proj = self.pair_pool_proj(z_normed) * pair_mask.unsqueeze(-1)  # [B, L, L, d_single]

        # Length-normalized sum over columns j
        col_counts = valid_mask_f.sum(dim=1, keepdim=True).unsqueeze(-1).clamp(min=1.0)  # [B, 1, 1]
        z_pooled = (z_proj.sum(dim=2) / col_counts) * valid_mask_f.unsqueeze(-1)         # [B, L, d_single]

        h = self.fuse_norm(s + z_pooled) * valid_mask_f.unsqueeze(-1)                    # [B, L, d_single]

        # 2. Expand 16 student query slots across batch: [B, 16, d_tok]
        q_queries = self.query_slots.unsqueeze(0).expand(B, -1, -1)

        # 3. Cross-attention: Q_slots attend over conditioned sequence H
        key_padding_mask = ~mask  # True where invalid/padded
        tokens_pred, _ = self.cross_attn(
            query=q_queries,
            key=h,
            value=h,
            key_padding_mask=key_padding_mask,
        )

        tokens_pred = self.token_norm(tokens_pred + q_queries)
        tokens_pred = self.final_norm(tokens_pred + self.token_mlp(tokens_pred))

        return tokens_pred  # [B, 16, d_tok]


# =============================================================================
# 2. JEPA Latent Loss Module
# =============================================================================

class PhaseCJEPALoss(nn.Module):
    """Computes hierarchical latent distance loss between student q̂ and teacher q*.

    Key Technical Properties:
      1. Detaches Teacher Targets: Enforces stop_gradient(q*) to prevent the student
         from corrupting the teacher's geometric manifold.
      2. Dual Distance Metric: Combines L2 Euclidean MSE and Directional Cosine Error:
             L_k = λ_MSE ||q̂_k - q^*_k||_2^2 + λ_cos (1 - cos(q̂_k, q^*_k))
      3. Prefix Hierarchy Curriculum: Weights early topological slots higher:
             w_{1:4} = 1.00,  w_{5:8} = 0.75,  w_{9:16} = 0.50
    """

    def __init__(self, cfg: Optional[PredictorConfig] = None):
        super().__init__()
        self.cfg = cfg or PredictorConfig()

        # Slot weight vector in R^{16}
        w1, w2, w3 = self.cfg.prefix_weights
        weights = torch.tensor(
            [w1] * 4 + [w2] * 4 + [w3] * 8,
            dtype=torch.float32,
        )
        self.register_buffer("slot_weights", weights)

    def forward(
        self,
        q_pred: torch.Tensor,
        q_target: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Calculates hierarchical JEPA loss between predicted and target tokens.

        Args:
            q_pred: [B, 16, d_tok] Predicted continuous tokens from TokenPredictor.
            q_target: [B, 16, d_tok] Native structural tokens from StructureEncoder.

        Returns:
            dict containing:
              - loss: Total scalar JEPA loss.
              - loss_mse: Mean weighted L2 distance.
              - loss_cos: Mean weighted Cosine distance (1 - cos).
              - token_cos_sim: Mean raw cosine similarity across all tokens.
        """
        # 1. Enforce strict stop_gradient on the teacher target
        q_target = q_target.detach()

        # Ensure slot weights are on the correct device
        weights = self.slot_weights.to(device=q_pred.device)  # [16]

        # 2. Per-token L2 squared distance: [B, 16]
        diff_sq = torch.sum((q_pred - q_target) ** 2, dim=-1)  # [B, 16]
        weighted_mse = (diff_sq * weights.unsqueeze(0)).mean()

        # 3. Per-token Cosine distance: [B, 16]
        # cos_sim = (q_pred . q_target) / (||q_pred|| * ||q_target|| + eps)
        q_pred_norm = F.normalize(q_pred, p=2, dim=-1)
        q_target_norm = F.normalize(q_target, p=2, dim=-1)
        cos_sim = torch.sum(q_pred_norm * q_target_norm, dim=-1)  # [B, 16]
        cos_dist = 1.0 - cos_sim                                  # [B, 16]
        weighted_cos = (cos_dist * weights.unsqueeze(0)).mean()

        # 4. Total Composite JEPA Loss
        total_loss = (
            self.cfg.loss_weight_mse * weighted_mse +
            self.cfg.loss_weight_cos * weighted_cos
        )

        return {
            "loss": total_loss,
            "loss_mse": weighted_mse.detach(),
            "loss_cos": weighted_cos.detach(),
            "mean_cos_sim": cos_sim.mean().detach(),
        }
