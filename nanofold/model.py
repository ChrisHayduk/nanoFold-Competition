from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .a3m import MSA_ALPHABET_SIZE, SEQ_ALPHABET_SIZE, GAP_ID, UNK_ID
from .metrics import lddt_ca


def msa_profile(msa_tokens: torch.Tensor, residue_mask: torch.Tensor) -> torch.Tensor:
    """Compute a simple MSA profile: per-position AA frequencies.

    msa_tokens: (B, N, L) int
    residue_mask: (B, L) bool (True for real residues, False for pad)

    Returns:
      profile: (B, L, 21) frequencies over {20 AAs + UNK}, ignoring gaps.
    """
    B, N, L = msa_tokens.shape
    device = msa_tokens.device

    # Map gaps to UNK for counting, then we'll explicitly ignore gaps via a mask.
    tokens = msa_tokens.clamp(min=0, max=MSA_ALPHABET_SIZE - 1)

    # Ignore padding positions (residue_mask False) by setting everything to GAP.
    tokens = torch.where(residue_mask[:, None, :], tokens, torch.full_like(tokens, GAP_ID))

    # Mask gaps
    is_gap = tokens == GAP_ID
    tokens_nogap = torch.where(is_gap, torch.full_like(tokens, UNK_ID), tokens)
    # One-hot over 23, then take first 21 dims
    one_hot = F.one_hot(tokens_nogap, num_classes=MSA_ALPHABET_SIZE).float()[..., :SEQ_ALPHABET_SIZE]  # (B,N,L,21)
    weights = (~is_gap).float()[..., None]  # (B,N,L,1)

    prof = (one_hot * weights).sum(dim=1)  # (B,L,21)
    denom = weights.sum(dim=1).clamp(min=1.0)
    prof = prof / denom
    return prof


class NanoFoldBaseline(nn.Module):
    """A deliberately small baseline: Transformer encoder -> Cα coordinates.

    This is *not* AlphaFold. It's just a lightweight starting point so the benchmark runs end-to-end.
    Competitors are expected to replace this with better folding architectures.

    Inputs:
      aatype: (B,L) int in [0,20]
      msa: (B,N,L) int in [0,22]
      residue_mask: (B,L) bool

    Outputs:
      pred_ca: (B,L,3)
    """

    def __init__(self, d_model: int, n_layers: int, n_heads: int, dropout: float):
        super().__init__()
        self.d_model = d_model

        self.aa_embed = nn.Embedding(SEQ_ALPHABET_SIZE, d_model)
        self.prof_proj = nn.Linear(SEQ_ALPHABET_SIZE, d_model, bias=False)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)

        self.coord_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 3),
        )

    def forward(self, aatype: torch.Tensor, msa: torch.Tensor, residue_mask: torch.Tensor) -> torch.Tensor:
        # aatype: (B,L)
        x = self.aa_embed(aatype.clamp(min=0, max=SEQ_ALPHABET_SIZE - 1))  # (B,L,d)

        prof = msa_profile(msa, residue_mask=residue_mask)                # (B,L,21)
        x = x + self.prof_proj(prof)

        # Transformer expects padding mask with True meaning "ignore"
        key_padding_mask = ~residue_mask
        x = self.encoder(x, src_key_padding_mask=key_padding_mask)

        pred_ca = self.coord_head(x)                                      # (B,L,3)
        return pred_ca


def distogram_loss(
    pred_ca: torch.Tensor,
    true_ca: torch.Tensor,
    ca_mask: torch.Tensor,
    residue_mask: torch.Tensor,
    cutoff: float = 15.0,
) -> torch.Tensor:
    """Simple distance-matrix MSE on pairs with true distance < cutoff.

    This is a *toy* loss that correlates with lDDT-Cα and is SE(3)-invariant.
    """
    B, L, _ = pred_ca.shape
    device = pred_ca.device

    # Compute pairwise distances
    d_true = torch.cdist(true_ca, true_ca, p=2)   # (B,L,L)
    d_pred = torch.cdist(pred_ca, pred_ca, p=2)

    # Valid pairs: within cutoff in true structure, both residues present, not padding, not i==j
    valid_res = ca_mask & residue_mask
    eye = torch.eye(L, dtype=torch.bool, device=device)[None, :, :]
    pair_mask = (d_true < cutoff) & valid_res[:, :, None] & valid_res[:, None, :] & (~eye)

    diff2 = (d_pred - d_true) ** 2
    if pair_mask.any():
        loss = diff2[pair_mask].mean()
    else:
        # Return a differentiable zero so backward() is still valid for batches
        # where no residue pairs satisfy the supervision mask.
        loss = d_pred.sum() * 0.0
    return loss


@torch.no_grad()
def batch_lddt_ca(pred_ca: torch.Tensor, true_ca: torch.Tensor, ca_mask: torch.Tensor, residue_mask: torch.Tensor) -> torch.Tensor:
    scores = []
    B = pred_ca.shape[0]
    for b in range(B):
        scores.append(lddt_ca(pred_ca[b], true_ca[b], ca_mask[b] & residue_mask[b]))
    return torch.stack(scores).mean()
