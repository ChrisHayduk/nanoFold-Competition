"""Phase B: Global Structure-Token Teacher (APT Structural Tokenizer) for NanoFold.

===============================================================================
BIOLOGICAL & MATHEMATICAL FOUNDATIONS:
===============================================================================
Under data scarcity (10,000 proteins), predicting 3D coordinates directly from
2D pair matrices forces a neural network to solve two coupled problems simultaneously:
  1. Global topology search: Is the protein an all-alpha bundle, beta-barrel,
     or alpha/beta sandwich? Which domains pack against each other?
  2. Sub-angstrom coordinate placement: Placing every atom into realistic local
     covalent geometry and torsion angles.

Direct coordinate loss often gets trapped in global topological inversions (knots).

The Adaptive Protein Tokenization (APT) Solution:
  We train a lightweight, training-only structural autoencoder that compresses
  a native 3D structure into a short sequence of K=16 global continuous tokens:
      X_native  -->  q_1, q_2, ..., q_16  in R^{16 x 64}

Nested Prefix Training (Coarse-to-Fine Ordering):
  During training, the decoder is randomly exposed to prefixes of different lengths:
      k in {2, 4, 8, 16}
      P(k=2)  = 0.10  --> Forces q_{1:2} to encode the lowest-bandwidth global shape.
      P(k=4)  = 0.20  --> Forces q_{1:4} to encode coarse topology.
      P(k=8)  = 0.30  --> Forces q_{1:8} to add intermediate domain packing.
      P(k=16) = 0.40  --> Forces q_{1:16} to provide the highest-fidelity V1 structural geometry description.

Three Multimodal Reconstruction Targets:
  1. Cβ Distogram: 64 bins spanning [2.0Å, 22.0Å] (symmetric global metric).
  2. Binary Contact Map: d(Cβ, Cβ) < 8.0Å (symmetric fold topology).
  3. Relative Backbone Orientation: R_i^T R_j and local direction R_i^T (t_j - t_i) / d_ij
     (breaks chirality / mirror-image symmetry that distances alone cannot resolve).

Competition Compliance Note:
  The tokenizer is trained strictly on the allowed 10,000 NanoFold training proteins.
  At evaluation/inference time, the native structure teacher encoder disappears;
  the sequence/MSA-to-token predictor (Phase C) infers the tokens directly.
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
class TokenizerConfig:
    """Hyperparameters configuring the Global Structure Tokenizer (Phase B)."""
    d_res: int = 128              # Dimension of residue representations h_i
    d_pair: int = 64              # Dimension of pair geometry representations z_ij
    d_tok: int = 64               # Dimension of global structural tokens q_k
    k_max: int = 16               # Maximum token count K = 16
    prefix_lengths: Tuple[int, ...] = (2, 4, 8, 16)
    prefix_probs: Tuple[float, ...] = (0.10, 0.20, 0.30, 0.40)
    n_heads: int = 4              # Attention heads for cross-attention
    num_rbf: int = 16             # Number of Gaussian radial basis functions for distances
    rbf_min: float = 2.0          # Minimum distance for RBF kernels (in Angstroms)
    rbf_max: float = 22.0         # Maximum distance for RBF kernels (in Angstroms)
    distogram_bins: int = 64      # Number of Cβ distogram bins
    loss_weight_dist: float = 1.0       # Weight for distogram cross-entropy
    loss_weight_contact: float = 0.25   # Weight for binary contact BCE
    loss_weight_orient: float = 0.25    # Weight for relative orientation MSE


# =============================================================================
# 1. Geometry Adapter (SE(3)-Invariant Feature Extraction)
# =============================================================================

def safe_normalize(v: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Safely normalizes vectors along the last dimension with an epsilon floor."""
    return v / torch.sqrt(torch.sum(v ** 2, dim=-1, keepdim=True) + eps)


def compute_gaussian_rbf(
    dists: torch.Tensor,
    num_rbf: int = 16,
    d_min: float = 2.0,
    d_max: float = 22.0,
) -> torch.Tensor:
    """Encodes pairwise Euclidean distances into Gaussian Radial Basis Functions.

    Equation:
        RBF_k(d) = exp( - (d - mu_k)^2 / (2 * sigma^2) )
    where centers mu_k are uniformly spaced between d_min and d_max.
    """
    device = dists.device
    centers = torch.linspace(d_min, d_max, num_rbf, device=device)
    sigma = (d_max - d_min) / float(num_rbf)
    diff = dists.unsqueeze(-1) - centers  # [..., num_rbf]
    return torch.exp(-0.5 * (diff / sigma) ** 2)


class GeometryAdapter(nn.Module):
    """Extracts strictly SE(3)-invariant residue and pair geometric features.

    Inputs:
      - aatype: [B, L]
      - atom14_positions: [B, L, 14, 3] (Slot 0=N, Slot 1=CA, Slot 2=C, Slot 4=CB)
      - atom14_mask: [B, L, 14]
      - residue_mask: [B, L]

    Extracted Invariants:
      1. Backbone orthonormal frames: R_i in SO(3), t_i = CA_i in R^3
      2. Pseudo-Cβ coordinates: slot 4 if present, else CA for Glycine
      3. Backbone dihedrals: (cos φ, sin φ, cos ψ, sin ψ, cos ω, sin ω)
      4. Pair distances: RBF(d_CA) and RBF(d_CB)
      5. Relative orientation: R_i^T R_j (9 matrix entries) and local direction R_i^T (t_j - t_i) / d_ij (3 entries)
    """

    def __init__(self, cfg: TokenizerConfig):
        super().__init__()
        self.cfg = cfg
        # Relative position encoding window: [-32, ..., +32]
        self.max_rel_pos = 32
        self.num_rel_pos_bins = 2 * self.max_rel_pos + 1
        self.rel_pos_embed = nn.Embedding(self.num_rel_pos_bins, 16)

    def forward(
        self,
        aatype: torch.Tensor,
        atom14_positions: torch.Tensor,
        atom14_mask: torch.Tensor,
        residue_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        B, L, _, _ = atom14_positions.shape
        device = atom14_positions.device

        # ---------------------------------------------------------------------
        # 1A. Backbone Atom Coordinates & Validity Mask
        # ---------------------------------------------------------------------
        pos_n = atom14_positions[:, :, 0, :]   # [B, L, 3]
        pos_ca = atom14_positions[:, :, 1, :]  # [B, L, 3]
        pos_c = atom14_positions[:, :, 2, :]   # [B, L, 3]

        mask_n = atom14_mask[:, :, 0]   # [B, L]
        mask_ca = atom14_mask[:, :, 1]  # [B, L]
        mask_c = atom14_mask[:, :, 2]   # [B, L]
        valid_bb = mask_n * mask_ca * mask_c * residue_mask.float()  # [B, L]

        # ---------------------------------------------------------------------
        # 1B. Gram-Schmidt Orthonormal Backbone Frames T_i = (R_i, t_i)
        # ---------------------------------------------------------------------
        # +x axis points from Cα toward C
        v1 = pos_c - pos_ca
        e1 = safe_normalize(v1)
        # +y axis orthogonalized from N toward Cα
        v2 = pos_n - pos_ca
        u2 = v2 - torch.sum(v2 * e1, dim=-1, keepdim=True) * e1
        e2 = safe_normalize(u2)
        # +z axis completes right-handed orthonormal triad
        e3 = safe_normalize(torch.cross(e1, e2, dim=-1))
        # Rotation matrix R = [e1, e2, e3] (columns in global coordinates)
        R = torch.stack([e1, e2, e3], dim=-1)  # [B, L, 3, 3]

        # ---------------------------------------------------------------------
        # 1C. Pseudo-Cβ Coordinates with Virtual Stereochemical Fallback
        # ---------------------------------------------------------------------
        # Ideal tetrahedral Cβ from N, CA, C backbone geometry:
        # b = CA - N, c = C - CA, a = b x c
        # Cβ_ideal = CA + (-0.58273431 * b + 0.56802827 * c - 0.54067466 * a)
        vec_b = pos_ca - pos_n
        vec_c = pos_c - pos_ca
        vec_a = safe_normalize(torch.cross(vec_b, vec_c, dim=-1))
        cb_ideal = pos_ca + (-0.58273431 * vec_b + 0.56802827 * vec_c - 0.54067466 * vec_a)

        # For Glycine (aatype == 7), pseudo-Cβ is Cα.
        # For non-Glycine: use experimental Cβ (slot 4) if present, else virtual Cβ!
        is_gly = (aatype == 7).unsqueeze(-1)  # [B, L, 1]
        has_exp_cb = (atom14_mask[:, :, 4:5] > 0)  # [B, L, 1]
        pos_cb_exp = atom14_positions[:, :, 4, :]  # [B, L, 3]

        pos_cb = torch.where(
            is_gly,
            pos_ca,
            torch.where(has_exp_cb, pos_cb_exp, cb_ideal),
        )
        # Cβ is valid whenever backbone is valid (since virtual Cβ is always reconstructible)
        valid_cb = valid_bb  # [B, L]

        # ---------------------------------------------------------------------
        # 1D. Backbone Dihedral Angles (φ, ψ, ω) as (cos, sin)
        # ---------------------------------------------------------------------
        # φ_i = Dihedral(C_{i-1}, N_i, Cα_i, C_i)
        # ψ_i = Dihedral(N_i, Cα_i, C_i, N_{i+1})
        # ω_i = Dihedral(Cα_{i-1}, C_{i-1}, N_i, Cα_i)
        dihedrals = torch.zeros((B, L, 6), device=device, dtype=atom14_positions.dtype)

        def _dihedral_sin_cos(p1: torch.Tensor, p2: torch.Tensor, p3: torch.Tensor, p4: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            b1 = p2 - p1
            b2 = safe_normalize(p3 - p2)
            b3 = p4 - p3
            v = b1 - torch.sum(b1 * b2, dim=-1, keepdim=True) * b2
            w = b3 - torch.sum(b3 * b2, dim=-1, keepdim=True) * b2
            x = torch.sum(v * w, dim=-1)
            y = torch.sum(torch.cross(b2, v, dim=-1) * w, dim=-1)
            norm = torch.sqrt(x ** 2 + y ** 2 + 1e-8)
            return x / norm, y / norm  # cos, sin

        # Compute φ, ψ, ω with boundary/padding-safe valid masks
        if L > 1:
            # Pair valid mask between consecutive residues: True iff both i and i+1 are valid
            valid_pair_adj = valid_bb[:, :-1] * valid_bb[:, 1:]  # [B, L-1]

            # φ_i for i in [1, L-1] uses residues i-1 and i
            cos_phi, sin_phi = _dihedral_sin_cos(pos_c[:, :-1], pos_n[:, 1:], pos_ca[:, 1:], pos_c[:, 1:])
            dihedrals[:, 1:, 0] = cos_phi * valid_pair_adj
            dihedrals[:, 1:, 1] = sin_phi * valid_pair_adj

            # ψ_i for i in [0, L-2] uses residues i and i+1
            cos_psi, sin_psi = _dihedral_sin_cos(pos_n[:, :-1], pos_ca[:, :-1], pos_c[:, :-1], pos_n[:, 1:])
            dihedrals[:, :-1, 2] = cos_psi * valid_pair_adj
            dihedrals[:, :-1, 3] = sin_psi * valid_pair_adj

            # ω_i for i in [1, L-1] uses residues i-1 and i
            cos_omega, sin_omega = _dihedral_sin_cos(pos_ca[:, :-1], pos_c[:, :-1], pos_n[:, 1:], pos_ca[:, 1:])
            dihedrals[:, 1:, 4] = cos_omega * valid_pair_adj
            dihedrals[:, 1:, 5] = sin_omega * valid_pair_adj

        # Mask invalid dihedral positions
        dihedrals = dihedrals * valid_bb.unsqueeze(-1)

        # ---------------------------------------------------------------------
        # 1E. Pairwise Distances and Relative Orientations
        # ---------------------------------------------------------------------
        pair_mask_bb = valid_bb.unsqueeze(2) * valid_bb.unsqueeze(1)  # [B, L, L]
        pair_mask_cb = valid_cb.unsqueeze(2) * valid_cb.unsqueeze(1)  # [B, L, L]

        # Pairwise Cα and Cβ distances (explicit difference avoids cdist dimension-dependent GEMM precision drift)
        diff_ca = pos_ca.unsqueeze(2) - pos_ca.unsqueeze(1)
        d_ca = torch.sqrt(torch.sum(diff_ca ** 2, dim=-1) + 1e-8)  # [B, L, L]
        diff_cb = pos_cb.unsqueeze(2) - pos_cb.unsqueeze(1)
        d_cb = torch.sqrt(torch.sum(diff_cb ** 2, dim=-1) + 1e-8)  # [B, L, L]

        rbf_ca = compute_gaussian_rbf(d_ca, self.cfg.num_rbf, self.cfg.rbf_min, self.cfg.rbf_max)
        rbf_cb = compute_gaussian_rbf(d_cb, self.cfg.num_rbf, self.cfg.rbf_min, self.cfg.rbf_max)

        # Relative orientation matrix: R_{ij} = R_i^T R_j in SO(3)
        # Shape: [B, L, L, 3, 3] -> flatten to 9 entries
        R_i = R.unsqueeze(2)  # [B, L, 1, 3, 3]
        R_j = R.unsqueeze(1)  # [B, 1, L, 3, 3]
        R_rel = torch.matmul(R_i.transpose(-1, -2), R_j)  # [B, L, L, 3, 3]
        R_rel_flat = R_rel.reshape(B, L, L, 9)

        # Local unit direction vector: R_i^T (t_j - t_i) / (||t_j - t_i|| + eps)
        diff_t = pos_ca.unsqueeze(1) - pos_ca.unsqueeze(2)  # t_j - t_i: [B, L, L, 3]
        unit_dir = safe_normalize(diff_t)
        local_dir = torch.einsum("bijk,bikm->bijm", unit_dir, R)  # [B, L, L, 3]

        orientation_feat = torch.cat([R_rel_flat, local_dir], dim=-1)  # [B, L, L, 12]

        # Relative sequence separation embedding |j - i|
        seq_idx = torch.arange(L, device=device)
        rel_pos = (seq_idx.unsqueeze(1) - seq_idx.unsqueeze(0)).clamp(-self.max_rel_pos, self.max_rel_pos) + self.max_rel_pos
        rel_pos_emb = self.rel_pos_embed(rel_pos).unsqueeze(0).expand(B, -1, -1, -1)  # [B, L, L, 16]

        # Mask each pair feature according to its physical validity
        rbf_ca = rbf_ca * pair_mask_bb.unsqueeze(-1)
        rbf_cb = rbf_cb * pair_mask_cb.unsqueeze(-1)
        orientation_feat = orientation_feat * pair_mask_bb.unsqueeze(-1)
        rel_pos_emb = rel_pos_emb * pair_mask_bb.unsqueeze(-1)

        # Combine pair features
        pair_feat = torch.cat([rbf_ca, rbf_cb, orientation_feat, rel_pos_emb], dim=-1)  # [B, L, L, 16+16+12+16=60]

        return {
            "dihedrals": dihedrals,                # [B, L, 6]
            "pair_feat": pair_feat,                # [B, L, L, 60]
            "pos_cb": pos_cb,                      # [B, L, 3]
            "pos_ca": pos_ca,                      # [B, L, 3]
            "d_cb": d_cb,                          # [B, L, L]
            "orientation_target": orientation_feat,# [B, L, L, 12]
            "valid_bb": valid_bb,                  # [B, L]
            "valid_cb": valid_cb,                  # [B, L]
            "pair_mask_bb": pair_mask_bb,          # [B, L, L]
            "pair_mask_cb": pair_mask_cb,          # [B, L, L]
        }


# =============================================================================
# 2. Structure Encoder (Compresses Native 3D Structure into K Tokens)
# =============================================================================

class StructureEncoder(nn.Module):
    """Encodes invariant residue and pair geometry into K global continuous tokens.

    Geometry-Only Teacher:
      Sees strictly dihedral angles (phi, psi, omega) and pairwise invariant geometry.
      Zero one-hot amino acid identity is fed into this module, guaranteeing that the
      latent tokens represent a pure structural language q* = f(structure).
    """

    def __init__(self, cfg: TokenizerConfig):
        super().__init__()
        self.cfg = cfg
        self.d_res = cfg.d_res
        self.d_pair = cfg.d_pair
        self.d_tok = cfg.d_tok
        self.k_max = cfg.k_max

        # Input projections:
        # Pure backbone dihedrals (6: cos/sin for phi, psi, omega) -> d_res
        self.res_proj = nn.Linear(6, self.d_res)
        # pair features (60) -> d_pair
        self.pair_proj = nn.Linear(60, self.d_pair)

        # Invariant interaction layers: Residue updates biased by pair geometry
        self.res_norm = nn.LayerNorm(self.d_res)
        self.pair_norm = nn.LayerNorm(self.d_pair)
        self.pair_pool_proj = nn.Linear(self.d_pair, self.d_res)

        # 16 learnable Perceiver query slots
        self.query_slots = nn.Parameter(torch.randn(self.k_max, self.d_tok) * 0.02)

        # Cross-attention: Query slots attend over residue representations h_i
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=self.d_tok,
            kdim=self.d_res,
            vdim=self.d_res,
            num_heads=cfg.n_heads,
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
        dihedrals: torch.Tensor,
        pair_feat: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass for StructureEncoder (Geometry-Only).

        Args:
            dihedrals: [B, L, 6] Backbone dihedral angles (cos/sin of phi, psi, omega).
            pair_feat: [B, L, L, 60] Pairwise invariant geometric features.
            valid_mask: [B, L] Backbone validity mask.

        Returns:
            q: [B, K, d_tok] K global continuous structural tokens.
        """
        B, L, _ = dihedrals.shape

        # Mask inputs so padding tokens don't leak into projections
        pair_mask = valid_mask.unsqueeze(2) * valid_mask.unsqueeze(1)  # [B, L, L]
        h_proj = self.res_proj(dihedrals) * valid_mask.unsqueeze(-1)  # [B, L, d_res]

        z = self.pair_proj(pair_feat) * pair_mask.unsqueeze(-1)        # [B, L, L, d_pair]
        z_normed = self.pair_norm(z) * pair_mask.unsqueeze(-1)
        z_proj = self.pair_pool_proj(z_normed) * pair_mask.unsqueeze(-1)

        # Integrate pair context via length-normalized sum over valid columns only
        col_counts = valid_mask.sum(dim=1, keepdim=True).unsqueeze(-1).clamp(min=1.0)  # [B, 1, 1]
        z_pooled = (z_proj.sum(dim=2) / col_counts) * valid_mask.unsqueeze(-1)        # [B, L, d_res]

        h = self.res_norm(h_proj + z_pooled) * valid_mask.unsqueeze(-1)

        # Expand query slots across batch: [B, K, d_tok]
        q_queries = self.query_slots.unsqueeze(0).expand(B, -1, -1)

        # Key padding mask: True where residue is invalid / padding
        key_padding_mask = (~valid_mask.bool())

        # Cross-attention: Q_slots attend over H_residues
        tokens, _ = self.cross_attn(
            query=q_queries,
            key=h,
            value=h,
            key_padding_mask=key_padding_mask,
        )

        tokens = self.token_norm(tokens + q_queries)
        tokens = self.final_norm(tokens + self.token_mlp(tokens))

        return tokens  # [B, K, d_tok]


# =============================================================================
# 3. Structure Decoder (Decodes Tokens into Distogram, Contacts & Orientations)
# =============================================================================

class StructureDecoder(nn.Module):
    """Decodes a prefix of global tokens q_{1:k} back into 3 structural targets:

      1. Cβ Distogram: 64-bin symmetric probability distribution
      2. Contact Map: Binary symmetric contact probabilities (d < 8.0Å)
      3. Relative Orientation: 12-dimensional invariant orientation features

    Perceiver-style Architecture:
      - Query residues (from sequence positions) attend to the active structural tokens q_{1:k}.
      - The decoded residue representation is strictly token-driven (no direct bypass of
        position embeddings), ensuring that the structural tokens genuinely carry the 3D geometry.
    """

    def __init__(self, cfg: TokenizerConfig):
        super().__init__()
        self.cfg = cfg
        self.d_tok = cfg.d_tok
        self.d_res = cfg.d_res
        self.d_pair = cfg.d_pair

        # Position embeddings for query residues
        self.pos_embed = nn.Embedding(512, self.d_res)

        # Cross-attention: Residues attend to the active structural tokens q_{1:k}
        self.res_token_attn = nn.MultiheadAttention(
            embed_dim=self.d_res,
            kdim=self.d_tok,
            vdim=self.d_tok,
            num_heads=cfg.n_heads,
            batch_first=True,
        )
        self.res_norm = nn.LayerNorm(self.d_res)
        self.res_mlp = nn.Sequential(
            nn.Linear(self.d_res, 2 * self.d_res),
            nn.GELU(),
            nn.Linear(2 * self.d_res, self.d_res),
        )
        self.res_norm2 = nn.LayerNorm(self.d_res)

        # Pairwise decoder projections
        self.pair_left = nn.Linear(self.d_res, self.d_pair)
        self.pair_right = nn.Linear(self.d_res, self.d_pair)
        self.pair_norm = nn.LayerNorm(self.d_pair)

        # Decoder Target Heads:
        # A. Cβ Distogram Head (64 bins)
        self.distogram_head = nn.Linear(self.d_pair, cfg.distogram_bins)
        # B. Contact Head (1 scalar)
        self.contact_head = nn.Linear(self.d_pair, 1)
        # C. Relative Orientation Head (12 dimensions: 9 matrix + 3 local direction)
        self.orient_head = nn.Linear(self.d_pair, 12)

    def forward(
        self,
        tokens: torch.Tensor,
        seq_len: int,
        active_k: int,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass for StructureDecoder.

        Args:
            tokens: [B, K, d_tok] Full or prefix-masked global tokens.
            seq_len: Length of target sequence L to decode.
            active_k: Number of active tokens in prefix k in {2, 4, 8, 16}.

        Returns:
            dict with:
              - distogram_logits: [B, L, L, 64] (symmetrized)
              - contact_logits:   [B, L, L] (symmetrized)
              - orient_pred:      [B, L, L, 12]
        """
        B = tokens.shape[0]
        device = tokens.device

        # Slice active prefix: [B, active_k, d_tok]
        active_tokens = tokens[:, :active_k, :]

        # Initialize residue queries from position embeddings
        pos_idx = torch.arange(seq_len, device=device).clamp(0, 511)
        res_queries = self.pos_embed(pos_idx).unsqueeze(0).expand(B, -1, -1)  # [B, L, d_res]

        # Cross-attend residues to active structural tokens
        # The query position i asks "what is my structural context in this fold?"
        # The tokens provide keys and values
        h_tokens, _ = self.res_token_attn(
            query=res_queries,
            key=active_tokens,
            value=active_tokens,
        )
        # Token-conditioned residue representation
        h_normed = self.res_norm(h_tokens)
        h_decoded = self.res_norm2(h_normed + self.res_mlp(h_normed))

        # Construct pairwise representation: z_{ij} = LayerNorm(Linear(h_i) + Linear(h_j))
        z_dec = self.pair_left(h_decoded).unsqueeze(2) + self.pair_right(h_decoded).unsqueeze(1)
        z_dec = self.pair_norm(z_dec)  # [B, L, L, d_pair]

        # A. Symmetrized Distogram Logits
        disto_logits = self.distogram_head(z_dec)  # [B, L, L, 64]
        disto_logits = 0.5 * (disto_logits + disto_logits.transpose(1, 2))

        # B. Symmetrized Contact Logits
        contact_logits = self.contact_head(z_dec).squeeze(-1)  # [B, L, L]
        contact_logits = 0.5 * (contact_logits + contact_logits.transpose(1, 2))

        # C. Relative Orientation Predictions
        orient_pred = self.orient_head(z_dec)  # [B, L, L, 12]

        return {
            "distogram_logits": disto_logits,
            "contact_logits": contact_logits,
            "orient_pred": orient_pred,
        }


# =============================================================================
# 4. Complete Structure Tokenizer Autoencoder (Teacher Module)
# =============================================================================

class GlobalStructureTokenizer(nn.Module):
    """The Complete Global Structure Tokenizer (Phase B).

    Combines:
      - GeometryAdapter: Invariant extraction of distances, dihedrals, orientations
      - StructureEncoder: Compression to K=16 continuous tokens
      - StructureDecoder: Coarse-to-fine reconstruction of distogram, contacts, and orientations
    """

    def __init__(self, cfg: Optional[TokenizerConfig] = None):
        super().__init__()
        self.cfg = cfg or TokenizerConfig()

        self.adapter = GeometryAdapter(self.cfg)
        self.encoder = StructureEncoder(self.cfg)
        self.decoder = StructureDecoder(self.cfg)

    def encode(
        self,
        aatype: torch.Tensor,
        atom14_positions: torch.Tensor,
        atom14_mask: torch.Tensor,
        residue_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Encodes native 3D Atom14 structure into K global continuous tokens."""
        geom = self.adapter(
            aatype=aatype,
            atom14_positions=atom14_positions,
            atom14_mask=atom14_mask,
            residue_mask=residue_mask,
        )

        tokens = self.encoder(
            dihedrals=geom["dihedrals"],
            pair_feat=geom["pair_feat"],
            valid_mask=geom["valid_bb"],
        )
        return tokens, geom

    def sample_prefix_length(self, eval_k: Optional[int] = None) -> int:
        """Samples a prefix length k in {2, 4, 8, 16} according to config probabilities."""
        if eval_k is not None:
            return eval_k
        if not self.training:
            return self.cfg.k_max  # Full budget at evaluation

        r = torch.rand(1).item()
        cum_prob = 0.0
        for k, prob in zip(self.cfg.prefix_lengths, self.cfg.prefix_probs, strict=True):
            cum_prob += prob
            if r <= cum_prob:
                return k
        return self.cfg.k_max

    def forward(
        self,
        aatype: torch.Tensor,
        atom14_positions: torch.Tensor,
        atom14_mask: torch.Tensor,
        residue_mask: torch.Tensor,
        eval_k: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """Complete forward pass with nested prefix sampling and loss computation."""
        B, L = aatype.shape

        # 1. Encode native structure to K tokens
        tokens, geom = self.encode(
            aatype=aatype,
            atom14_positions=atom14_positions,
            atom14_mask=atom14_mask,
            residue_mask=residue_mask,
        )

        # 2. Sample coarse-to-fine prefix length k
        active_k = self.sample_prefix_length(eval_k=eval_k)

        # 3. Decode from prefix tokens
        decoded = self.decoder(
            tokens=tokens,
            seq_len=L,
            active_k=active_k,
        )

        # 4. Compute Multimodal Reconstruction Losses:
        pair_mask_bb = geom["pair_mask_bb"]  # [B, L, L]
        pair_mask_cb = geom["pair_mask_cb"]  # [B, L, L]

        # Ignore diagonal i == j in loss as trivial supervision
        off_diag = ~torch.eye(L, device=aatype.device, dtype=torch.bool).unsqueeze(0)
        dist_mask = pair_mask_cb.bool() & off_diag
        orient_mask = (pair_mask_bb.bool() & off_diag).unsqueeze(-1).expand_as(decoded["orient_pred"])

        # Target A: Cβ Distogram Cross-Entropy
        d_cb = geom["d_cb"]  # [B, L, L]
        bin_width = (self.cfg.rbf_max - self.cfg.rbf_min) / float(self.cfg.distogram_bins - 1)
        gt_bins = ((d_cb - self.cfg.rbf_min) / bin_width).long().clamp(0, self.cfg.distogram_bins - 1)

        # Flatten masked positions for cross-entropy
        logits_flat = decoded["distogram_logits"][dist_mask]  # [N_valid, 64]
        labels_flat = gt_bins[dist_mask]                      # [N_valid]
        if logits_flat.numel() > 0:
            loss_dist = F.cross_entropy(logits_flat, labels_flat)
        else:
            loss_dist = torch.tensor(0.0, device=aatype.device)

        # Target B: Binary Contact Loss (d_CB < 8.0Å)
        contact_targets = (d_cb < 8.0).float()
        contact_logits_flat = decoded["contact_logits"][dist_mask]
        contact_targets_flat = contact_targets[dist_mask]
        if contact_logits_flat.numel() > 0:
            loss_contact = F.binary_cross_entropy_with_logits(contact_logits_flat, contact_targets_flat)
        else:
            loss_contact = torch.tensor(0.0, device=aatype.device)

        # Target C: Relative Orientation Loss (MSE on 12 invariant features)
        orient_pred = decoded["orient_pred"]                  # [B, L, L, 12]
        orient_gt = geom["orientation_target"]                # [B, L, L, 12]
        if orient_mask.any():
            loss_orient = F.mse_loss(orient_pred[orient_mask], orient_gt[orient_mask])
        else:
            loss_orient = torch.tensor(0.0, device=aatype.device)

        # Total Composite Reconstruction Loss
        total_loss = (
            self.cfg.loss_weight_dist * loss_dist +
            self.cfg.loss_weight_contact * loss_contact +
            self.cfg.loss_weight_orient * loss_orient
        )

        return {
            "tokens": tokens,                          # [B, K, d_tok]
            "active_k": active_k,
            "loss": total_loss,
            "loss_dist": loss_dist.detach(),
            "loss_contact": loss_contact.detach(),
            "loss_orient": loss_orient.detach(),
            "distogram_logits": decoded["distogram_logits"],
            "contact_logits": decoded["contact_logits"],
            "orient_pred": decoded["orient_pred"],
        }
