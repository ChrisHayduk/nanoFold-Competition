"""Stage 1: Compact Micro-Pairformer Trunk for NanoFold.

===============================================================================
BIOLOGICAL & MATHEMATICAL FOUNDATIONS:
===============================================================================
In protein structure prediction, 1D amino acid sequences fold into 3D shapes
primarily through pairwise physical contacts (hydrogen bonds, salt bridges,
hydrophobic packing, and disulfide bonds).

1. Why 2D Pair Representations?
   While 1D sequence representations s_i capture local residue properties
   (hydrophobicity, secondary structure propensity), 3D tertiary structure is
   fundamentally governed by 2D pairwise spatial relationships z_{ij}.
   The pair matrix z_{ij} acts as a learned representation of the spatial
   distance and orientation between residue i and residue j.

2. Three-Residue Geometric Reasoning (Triangular Updates):
   Triangular multiplicative updates allow the representation of residue pair
   (i, j) to reason over all intermediate residues k, providing a strong
   inductive bias for geometrically consistent three-residue relationships:
       (i, k), (j, k) -> (i, j)   [Outgoing / Starting Node]
       (k, i), (k, j) -> (i, j)   [Incoming / Ending Node]
   The operation can learn relationships related to triangle geometry and
   path consistency, providing dense relational message-passing across the
   entire contact graph.

3. Preserving True MSA Coevolution (Without a 48-Block MSA Stack):
   A 1D amino-acid position frequency profile p(a_i) captures conservation,
   but destroys correlated mutations:
       Cov(a_i, a_j) = p(a_i, a_j) - p(a_i) * p(a_j)
   Correlated mutations (e.g. compensatory charge swaps Lys<->Asp or
   hydrophobic volume packing) are the primary evolutionary signal for 3D contacts.
   Instead of carrying an expensive MSA tensor (B, N, L, c) through 48 blocks,
   we extract low-rank pairwise coevolution at the input:
       C_{ij} = (1 / N) * sum_{n=1}^N a(m_{n, i}) (x) b(m_{n, j})
   and inject it directly into z_{ij}^{(0)}, then discard the raw MSA!

4. Why a "Micro" Pairformer (6-8 Blocks)?
   Standard AlphaFold2 uses 48 heavy Evoformer blocks (~93M parameters).
   On a limited 10,000 protein budget (240,000 samples), 48 blocks severely
   underfit or overfit and consume 80% of compute.
   Our Micro-Pairformer uses 6-8 compact blocks (~1.8M parameters), designed
   to improve optimization and sample efficiency under the 30,000-step budget.
===============================================================================
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from nanofold.a3m import GAP_ID, MSA_ALPHABET_SIZE, SEQ_ALPHABET_SIZE
from nanofold.model import msa_profile

# =============================================================================
# Configuration Dataclass
# =============================================================================

@dataclass
class PairformerConfig:
    """Hyperparameters configuring the Micro-Pairformer architecture."""
    d_single: int = 128           # Dimension of 1D single residue representation (s_i)
    d_pair: int = 64              # Dimension of 2D pairwise representation (z_ij)
    d_tri_hidden: int = 64        # Intermediate projection channel size for triangular updates
    d_msa_emb: int = 16           # Low-rank embedding dimension for raw MSA tokens
    d_msa_opm: int = 16           # Inner projection dimension for MSA coevolution outer-product
    n_blocks: int = 6             # Number of stacked Pairformer blocks (6-8 is optimal for 10k budget)
    n_heads_pair: int = 4         # Number of attention heads for pair self-attention
    n_heads_single: int = 8       # Number of attention heads for single residue self-attention
    dropout: float = 0.0          # Dropout probability
    max_relative_pos: int = 32    # Window for relative positional encoding (clamps |i - j| > 32)
    distogram_bins: int = 64      # Number of distance histogram bins (spanning 2.0A to 22.0A)
    min_dist: float = 2.0         # Minimum distance threshold for distogram (in Angstroms)
    max_dist: float = 22.0        # Maximum distance threshold for distogram (in Angstroms)


# =============================================================================
# 1. Input Embeddings (Sequence + True MSA Coevolution + Relative Positions)
# =============================================================================

class InputEmbedder(nn.Module):
    """Embeds raw biological features into single (s_i) and pair (z_ij) representations.

    Key biological signals incorporated:
      - Primary sequence identity: Embedding of aatype (20 standard AAs + UNK)
      - Conservation profile: 1D frequency profile across homologous sequences
      - True pairwise coevolution: Low-rank outer product of raw MSA sequences
        capturing correlated mutations across homologs
      - Deletion statistics: Log-transformed deletion counts
      - Relative 1D sequence separation: Clamped position offsets (j - i)
    """

    def __init__(self, cfg: PairformerConfig):
        super().__init__()
        self.cfg = cfg
        self.d_single = cfg.d_single
        self.d_pair = cfg.d_pair

        # ---------------------------------------------------------------------
        # 1A. Single Residue Embeddings (1D: per-residue)
        # ---------------------------------------------------------------------
        self.aa_embedding = nn.Embedding(SEQ_ALPHABET_SIZE, self.d_single)
        self.profile_projection = nn.Linear(SEQ_ALPHABET_SIZE, self.d_single, bias=False)
        self.deletion_projection = nn.Linear(1, self.d_single, bias=False)
        self.single_layer_norm = nn.LayerNorm(self.d_single)

        # ---------------------------------------------------------------------
        # 1B. Relative Positional Embeddings (2D: per-pair)
        # ---------------------------------------------------------------------
        self.num_rel_pos_bins = 2 * cfg.max_relative_pos + 1
        self.rel_pos_embedding = nn.Embedding(self.num_rel_pos_bins, self.d_pair)

        # ---------------------------------------------------------------------
        # 1C. True MSA Pairwise Coevolution Extractor (Low-Rank OPM)
        # ---------------------------------------------------------------------
        # Embeds raw MSA tokens into low-rank channels c_m
        self.msa_embedding = nn.Embedding(MSA_ALPHABET_SIZE, cfg.d_msa_emb)
        # Linear projections for low-rank outer-product factors
        self.msa_proj_a = nn.Linear(cfg.d_msa_emb, cfg.d_msa_opm, bias=False)
        self.msa_proj_b = nn.Linear(cfg.d_msa_emb, cfg.d_msa_opm, bias=False)
        # Projects flattened outer product (d_msa_opm * d_msa_opm) into d_pair
        self.msa_cov_to_pair = nn.Linear(cfg.d_msa_opm * cfg.d_msa_opm, self.d_pair, bias=False)
        # Initialize coevolution projection with small weights to prevent initial shock
        nn.init.normal_(self.msa_cov_to_pair.weight, std=0.02)

        # ---------------------------------------------------------------------
        # 1D. Sequence-to-Pair Bias Projections
        # ---------------------------------------------------------------------
        self.single_to_pair_left = nn.Linear(self.d_single, self.d_pair, bias=False)
        self.single_to_pair_right = nn.Linear(self.d_single, self.d_pair, bias=False)
        self.pair_layer_norm = nn.LayerNorm(self.d_pair)

    def forward(
        self,
        aatype: torch.Tensor,
        msa: torch.Tensor,
        deletions: torch.Tensor,
        residue_index: torch.Tensor,
        residue_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for InputEmbedder.

        Args:
            aatype: [B, L] Integer tensor of amino acid identities.
            msa: [B, N, L] Integer tensor of MSA tokens.
            deletions: [B, N, L] Integer tensor of deletion counts.
            residue_index: [B, L] Integer tensor of residue position indices.
            residue_mask: [B, L] Boolean mask (True = real residue, False = padding).

        Returns:
            s: [B, L, d_single] Initial single representation.
            z: [B, L, L, d_pair] Initial pair representation with true MSA coevolution.
        """
        B, L = aatype.shape
        _, N, _ = msa.shape

        # ---------------------------------------------------------------------
        # Step 1: Construct Single Embedding s_i
        # ---------------------------------------------------------------------
        clamped_aatype = aatype.clamp(min=0, max=SEQ_ALPHABET_SIZE - 1)
        s = self.aa_embedding(clamped_aatype)  # [B, L, d_single]

        # Profile frequencies across MSA
        prof = msa_profile(msa_tokens=msa, residue_mask=residue_mask)
        s = s + self.profile_projection(prof)

        # Deletion counts (log-transformed)
        del_feat = torch.log1p(deletions.float()).mean(dim=1, keepdim=True)
        del_feat = torch.where(residue_mask.unsqueeze(1), del_feat, torch.zeros_like(del_feat))
        s = s + self.deletion_projection(del_feat.permute(0, 2, 1))

        s = self.single_layer_norm(s)
        s = s * residue_mask.unsqueeze(-1).float()

        # ---------------------------------------------------------------------
        # Step 2: Construct Relative Positional Matrix RelPos(i, j)
        # ---------------------------------------------------------------------
        pos_i = residue_index.unsqueeze(2)  # [B, L, 1]
        pos_j = residue_index.unsqueeze(1)  # [B, 1, L]
        rel_offset = pos_j - pos_i          # [B, L, L]

        clamped_offset = rel_offset.clamp(
            min=-self.cfg.max_relative_pos,
            max=self.cfg.max_relative_pos,
        ) + self.cfg.max_relative_pos
        z = self.rel_pos_embedding(clamped_offset)  # [B, L, L, d_pair]

        # ---------------------------------------------------------------------
        # Step 3: Extract True Pairwise MSA Coevolution Covariance
        # ---------------------------------------------------------------------
        # Correlated mutations: sum_n a(m_{n, i}) (x) b(m_{n, j}) / N
        clamped_msa = msa.clamp(min=0, max=MSA_ALPHABET_SIZE - 1)
        m_emb = self.msa_embedding(clamped_msa)  # [B, N, L, d_msa_emb]

        # Mask out gap tokens (GAP_ID = 21) and padding residues
        is_valid = (clamped_msa != GAP_ID) & residue_mask.unsqueeze(1)  # [B, N, L]
        m_emb = m_emb * is_valid.unsqueeze(-1).float()

        m_a = self.msa_proj_a(m_emb)  # [B, N, L, d_msa_opm]
        m_b = self.msa_proj_b(m_emb)  # [B, N, L, d_msa_opm]

        # Batched outer-product contraction across sequence dimension N:
        # Sum is completely permutation-invariant with respect to non-query rows!
        cov = torch.einsum("bnic,bnjd->bijcd", m_a, m_b) / max(1, N)  # [B, L, L, c, c]
        cov = cov.reshape(B, L, L, self.cfg.d_msa_opm * self.cfg.d_msa_opm)
        z = z + self.msa_cov_to_pair(cov)

        # ---------------------------------------------------------------------
        # Step 4: Sequence Outer Product into Pair Space
        # ---------------------------------------------------------------------
        s_left = self.single_to_pair_left(s)
        s_right = self.single_to_pair_right(s)
        z = z + s_left.unsqueeze(2) + s_right.unsqueeze(1)

        # Normalize and mask padding pairs
        z = self.pair_layer_norm(z)
        pair_mask = residue_mask.unsqueeze(2) & residue_mask.unsqueeze(1)
        z = z * pair_mask.unsqueeze(-1).float()

        return s, z


# =============================================================================
# 2. Triangular Multiplicative Update (3-Residue Relational Reasoning)
# =============================================================================

class TriangularMultiplicativeUpdate(nn.Module):
    """Triangular Multiplicative Update for Residue Pair Relationships.

    Mathematical Formulation:
      Allows pair representation (i, j) to receive information from all two-edge
      paths through an intermediate residue k:

      Outgoing Mode:
          (i, k), (j, k) -> (i, j)
          a_{ik} = LayerNorm(z_{ik}) W_A,   b_{jk} = LayerNorm(z_{jk}) W_B
          contracted_{ij} = Sum_{k=1}^L a_{ik} * b_{jk}

      Incoming Mode:
          (k, i), (k, j) -> (i, j)
          a_{ki} = LayerNorm(z_{ki}) W_A,   b_{kj} = LayerNorm(z_{kj}) W_B
          contracted_{ij} = Sum_{k=1}^L a_{ki} * b_{kj}

    Numerical Stability:
      - The contraction is scaled by 1 / sqrt(L) to prevent activation variance
        from scaling with sequence length.
      - Output projection W_O is initialized to zeros, ensuring residual identity
        at step 0.
    """

    def __init__(self, d_pair: int, d_hidden: int, mode: str = "outgoing"):
        super().__init__()
        assert mode in ("outgoing", "incoming"), f"Invalid mode {mode!r}; expected 'outgoing' or 'incoming'."
        self.mode = mode
        self.d_pair = d_pair
        self.d_hidden = d_hidden

        self.layer_norm_in = nn.LayerNorm(d_pair)
        self.proj_a = nn.Linear(d_pair, d_hidden, bias=False)
        self.proj_b = nn.Linear(d_pair, d_hidden, bias=False)
        self.gate_a = nn.Linear(d_pair, d_hidden, bias=False)
        self.gate_b = nn.Linear(d_pair, d_hidden, bias=False)
        self.gate_out = nn.Linear(d_pair, d_pair, bias=False)

        self.layer_norm_mid = nn.LayerNorm(d_hidden)
        self.proj_out = nn.Linear(d_hidden, d_pair, bias=False)

        # Zero-initialize output projection for smooth residual gradient flow
        nn.init.zeros_(self.proj_out.weight)

    def forward(self, z: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass for TriangularMultiplicativeUpdate."""
        B, L, _, _ = z.shape

        z_norm = self.layer_norm_in(z)
        a = self.proj_a(z_norm) * torch.sigmoid(self.gate_a(z_norm))  # [B, L, L, d_hidden]
        b = self.proj_b(z_norm) * torch.sigmoid(self.gate_b(z_norm))  # [B, L, L, d_hidden]

        if mask is not None:
            mask_float = mask.unsqueeze(-1).float()
            a = a * mask_float
            b = b * mask_float

        # Contraction over intermediate residues k with length-scaling normalization
        scale = 1.0 / math.sqrt(max(1, L))
        if self.mode == "outgoing":
            contracted = torch.einsum("bikc,bjkc->bijc", a, b) * scale
        else:
            contracted = torch.einsum("bkic,bkjc->bijc", a, b) * scale

        contracted = self.layer_norm_mid(contracted)
        out = self.proj_out(contracted)

        gate = torch.sigmoid(self.gate_out(z_norm))
        z_update = out * gate

        return z + z_update


# =============================================================================
# 3. Outer Product Mean (1D Sequence -> 2D Pair Communication)
# =============================================================================

class OuterProductMean(nn.Module):
    """Communicates single residue properties into 2D pair relationships."""

    def __init__(self, d_single: int, d_pair: int, d_hidden: int = 32):
        super().__init__()
        self.d_single = d_single
        self.d_pair = d_pair
        self.d_hidden = d_hidden

        self.layer_norm = nn.LayerNorm(d_single)
        self.proj_a = nn.Linear(d_single, d_hidden, bias=False)
        self.proj_b = nn.Linear(d_single, d_hidden, bias=False)
        self.proj_out = nn.Linear(d_hidden * d_hidden, d_pair, bias=False)

        # Zero-initialize output projection for smooth residual accumulation
        nn.init.zeros_(self.proj_out.weight)

    def forward(self, s: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass for OuterProductMean."""
        B, L, _ = s.shape

        s_norm = self.layer_norm(s)
        a = self.proj_a(s_norm)  # [B, L, d_hidden]
        b = self.proj_b(s_norm)  # [B, L, d_hidden]

        if mask is not None:
            mask_float = mask.unsqueeze(-1).float()
            a = a * mask_float
            b = b * mask_float

        outer = torch.einsum("bic,bjd->bijcd", a, b)
        outer = outer.reshape(B, L, L, self.d_hidden * self.d_hidden)
        delta_z = self.proj_out(outer)
        return delta_z


# =============================================================================
# 4. Single-Residue Attention with Pair Bias (2D Pair -> 1D Sequence)
# =============================================================================

class SingleAttentionWithPairBias(nn.Module):
    """Updates 1D residue representations steered by 2D pairwise relationships."""

    def __init__(self, d_single: int, d_pair: int, n_heads: int):
        super().__init__()
        self.d_single = d_single
        self.d_pair = d_pair
        self.n_heads = n_heads
        self.head_dim = d_single // n_heads

        assert d_single % n_heads == 0, "d_single must be divisible by n_heads"

        self.layer_norm = nn.LayerNorm(d_single)
        self.q_proj = nn.Linear(d_single, d_single, bias=False)
        self.k_proj = nn.Linear(d_single, d_single, bias=False)
        self.v_proj = nn.Linear(d_single, d_single, bias=False)
        self.pair_bias_proj = nn.Linear(d_pair, n_heads, bias=False)
        self.gate_proj = nn.Linear(d_single, d_single, bias=False)
        self.out_proj = nn.Linear(d_single, d_single, bias=False)

        # Zero-initialize output projection
        nn.init.zeros_(self.out_proj.weight)

    def forward(
        self,
        s: torch.Tensor,
        z: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass for SingleAttentionWithPairBias."""
        B, L, _ = s.shape

        s_norm = self.layer_norm(s)

        q = self.q_proj(s_norm).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(s_norm).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(s_norm).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)

        # Directional pair bias: z_{ij} provides head-specific attention affinity
        pair_bias = self.pair_bias_proj(z).permute(0, 3, 1, 2)  # [B, n_heads, L, L]

        scale = 1.0 / math.sqrt(self.head_dim)
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale + pair_bias

        if mask is not None:
            padding_mask = (~mask).unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(padding_mask, float("-inf"))

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0)

        context = torch.matmul(attn_weights, v)
        context = context.transpose(1, 2).contiguous().view(B, L, self.d_single)

        gate = torch.sigmoid(self.gate_proj(s_norm))
        out = self.out_proj(context * gate)

        return s + out


# =============================================================================
# 5. Transition Feed-Forward MLP
# =============================================================================

class TransitionMLP(nn.Module):
    """Two-layer feed-forward network with GELU activation and residual connection."""

    def __init__(self, d_in: int, expansion_factor: int = 2):
        super().__init__()
        d_hidden = d_in * expansion_factor
        self.layer_norm = nn.LayerNorm(d_in)
        self.linear1 = nn.Linear(d_in, d_hidden, bias=False)
        self.linear2 = nn.Linear(d_hidden, d_in, bias=False)

        # Zero-initialize second linear projection for residual stability
        nn.init.zeros_(self.linear2.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.linear2(F.gelu(self.linear1(self.layer_norm(x))))


# =============================================================================
# 6. Complete Pairformer Block
# =============================================================================

class PairformerBlock(nn.Module):
    """A single integrated Micro-Pairformer Block."""

    def __init__(self, cfg: PairformerConfig):
        super().__init__()
        self.cfg = cfg

        self.tri_outgoing = TriangularMultiplicativeUpdate(
            d_pair=cfg.d_pair,
            d_hidden=cfg.d_tri_hidden,
            mode="outgoing",
        )
        self.tri_incoming = TriangularMultiplicativeUpdate(
            d_pair=cfg.d_pair,
            d_hidden=cfg.d_tri_hidden,
            mode="incoming",
        )
        self.pair_mlp = TransitionMLP(d_in=cfg.d_pair, expansion_factor=2)

        self.outer_product_mean = OuterProductMean(
            d_single=cfg.d_single,
            d_pair=cfg.d_pair,
            d_hidden=32,
        )
        self.single_attention = SingleAttentionWithPairBias(
            d_single=cfg.d_single,
            d_pair=cfg.d_pair,
            n_heads=cfg.n_heads_single,
        )
        self.single_mlp = TransitionMLP(d_in=cfg.d_single, expansion_factor=2)

    def forward(
        self,
        s: torch.Tensor,
        z: torch.Tensor,
        residue_mask: torch.Tensor,
        pair_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through a single Pairformer block."""
        # 1. 3-Residue Geometric Relational Reasoning
        z = self.tri_outgoing(z, mask=pair_mask)
        z = self.tri_incoming(z, mask=pair_mask)
        z = self.pair_mlp(z)

        # 2. Sequence to Pair Communication
        z = z + self.outer_product_mean(s, mask=residue_mask)

        # 3. Pair to Sequence Communication
        s = self.single_attention(s, z=z, mask=residue_mask)
        s = self.single_mlp(s)

        return s, z


# =============================================================================
# 7. Complete Micro-Pairformer Trunk
# =============================================================================

class MicroPairformer(nn.Module):
    """The Complete Compact Micro-Pairformer Trunk (Stage 1)."""

    def __init__(self, cfg: Optional[PairformerConfig] = None):
        super().__init__()
        self.cfg = cfg or PairformerConfig()

        self.input_embedder = InputEmbedder(self.cfg)
        self.blocks = nn.ModuleList([
            PairformerBlock(self.cfg) for _ in range(self.cfg.n_blocks)
        ])

        self.norm_single = nn.LayerNorm(self.cfg.d_single)
        self.norm_pair = nn.LayerNorm(self.cfg.d_pair)

    def forward(
        self,
        aatype: torch.Tensor,
        msa: torch.Tensor,
        deletions: torch.Tensor,
        residue_index: torch.Tensor,
        residue_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the entire Micro-Pairformer trunk."""
        s, z = self.input_embedder(
            aatype=aatype,
            msa=msa,
            deletions=deletions,
            residue_index=residue_index,
            residue_mask=residue_mask,
        )

        pair_mask = residue_mask.unsqueeze(2) & residue_mask.unsqueeze(1)

        for block in self.blocks:
            s, z = block(s=s, z=z, residue_mask=residue_mask, pair_mask=pair_mask)

        s = self.norm_single(s) * residue_mask.unsqueeze(-1).float()
        z = self.norm_pair(z) * pair_mask.unsqueeze(-1).float()

        return s, z


# =============================================================================
# 8. Auxiliary Output Heads (Distogram & Contact Prediction)
# =============================================================================

class DistogramHead(nn.Module):
    """Predicts binned Cβ-Cβ distance distributions from pair representations.

    Distance Bins:
      - 64 bins uniformly spaced between min_dist (2.0Å) and max_dist (22.0Å).
      - Bin 64 represents distances > 22.0Å (no contact).

    Symmetry Guarantee:
      Physical distances are strictly symmetric: d(i, j) = d(j, i).
      We enforce this in representation space by symmetrizing the logits:
          logits_{ij} = 0.5 * ( L(z_{ij}) + L(z_{ji}) )
    """

    def __init__(self, d_pair: int, num_bins: int = 64):
        super().__init__()
        self.num_bins = num_bins
        self.layer_norm = nn.LayerNorm(d_pair)
        self.proj = nn.Linear(d_pair, num_bins)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Forward pass for DistogramHead."""
        z_norm = self.layer_norm(z)
        logits = self.proj(z_norm)
        # Symmetrize logits: distance from i to j equals distance from j to i
        symmetrized_logits = 0.5 * (logits + logits.transpose(1, 2))
        return symmetrized_logits


class ContactHead(nn.Module):
    """Predicts binary residue-residue contact probability (d_ij < 8.0Å).

    Enforces exact physical symmetry: C_ij == C_ji.
    Notice: We symmetrize the logits of this head, while leaving the underlying
    pair representation z_{ij} free to maintain directional information.
    """

    def __init__(self, d_pair: int):
        super().__init__()
        self.layer_norm = nn.LayerNorm(d_pair)
        self.proj = nn.Linear(d_pair, 1)

    def forward(self, z: torch.Tensor, return_logits: bool = False) -> torch.Tensor:
        """Forward pass for ContactHead.

        Args:
            z: [B, L, L, d_pair] Pair representation tensor.
            return_logits: If True, returns symmetrized logits for BCEWithLogitsLoss.
                           If False, returns sigmoid probabilities in [0, 1].

        Returns:
            contacts: [B, L, L] Symmetrized logits or probabilities.
        """
        z_norm = self.layer_norm(z)
        logits = self.proj(z_norm).squeeze(-1)  # [B, L, L]
        # Enforce exact physical contact symmetry: C_ij == C_ji
        symmetrized_logits = 0.5 * (logits + logits.transpose(1, 2))
        if return_logits:
            return symmetrized_logits
        return torch.sigmoid(symmetrized_logits)
