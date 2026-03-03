from __future__ import annotations

import math
from typing import Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


# Portions of this file are adapted from publicly available ESMFold code:
# https://github.com/facebookresearch/esm
# Specifically inspired by:
# - esm/esmfold/v1/trunk.py (RelativePosition and seq/pair trunk structure)
# - esm/esmfold/v1/tri_self_attn_block.py (pair-biased sequence attention idea)
# The ESM repository is MIT licensed (see THIRD_PARTY_LICENSES.md in this folder).


class RelativePosition(nn.Module):
    """Relative positional encoding adapted from ESMFold trunk.py."""

    def __init__(self, bins: int, pairwise_state_dim: int):
        super().__init__()
        self.bins = bins
        self.embedding = nn.Linear(2 * bins + 2, pairwise_state_dim)

    def forward(self, residue_index: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        diff = residue_index[:, None, :] - residue_index[:, :, None]
        diff = diff.clamp(-self.bins, self.bins)
        diff = diff + self.bins + 1
        if mask is not None:
            pair_mask = mask[:, :, None] & mask[:, None, :]
            diff = diff * pair_mask + (pair_mask == 0) * 0
        one_hot = F.one_hot(diff, num_classes=2 * self.bins + 2).float()
        return self.embedding(one_hot)


class PairBiasedSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, pair_dim: int, dropout: float):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads})")
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = self.head_dim ** -0.5
        self.dropout = float(dropout)

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)
        self.pair_bias = nn.Linear(pair_dim, n_heads, bias=False)

    def forward(self, x: torch.Tensor, z: torch.Tensor, residue_mask: torch.Tensor) -> torch.Tensor:
        # x: (B,L,C), z: (B,L,L,Cz), residue_mask: (B,L)
        B, L, C = x.shape
        H = self.n_heads
        D = self.head_dim

        q = self.q_proj(x).view(B, L, H, D).transpose(1, 2)
        k = self.k_proj(x).view(B, L, H, D).transpose(1, 2)
        v = self.v_proj(x).view(B, L, H, D).transpose(1, 2)

        bias = self.pair_bias(z).permute(0, 3, 1, 2).contiguous()
        attn_logits = torch.matmul(q, k.transpose(-2, -1)) * self.scale + bias

        key_ok = residue_mask[:, None, None, :]
        query_ok = residue_mask[:, None, :, None]
        attn_logits = attn_logits.masked_fill(~key_ok, -1.0e4)
        attn_logits = attn_logits.masked_fill(~query_ok, -1.0e4)

        attn = torch.softmax(attn_logits, dim=-1)
        if self.training and self.dropout > 0:
            attn = F.dropout(attn, p=self.dropout)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, L, C)
        out = self.o_proj(out)
        return out


class ESMFoldSeedBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, pair_dim: int, dropout: float):
        super().__init__()
        self.seq_ln = nn.LayerNorm(d_model)
        self.seq_attn = PairBiasedSelfAttention(d_model=d_model, n_heads=n_heads, pair_dim=pair_dim, dropout=dropout)
        self.seq_ff_ln = nn.LayerNorm(d_model)
        self.seq_ff = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
        )

        self.outer_left = nn.Linear(d_model, pair_dim)
        self.outer_right = nn.Linear(d_model, pair_dim)
        self.pair_ln = nn.LayerNorm(pair_dim)
        self.pair_ff = nn.Sequential(
            nn.Linear(pair_dim, 2 * pair_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * pair_dim, pair_dim),
        )

    def forward(self, s: torch.Tensor, z: torch.Tensor, residue_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        s = s + self.seq_attn(self.seq_ln(s), z, residue_mask=residue_mask)
        s = s + self.seq_ff(self.seq_ff_ln(s))

        left = self.outer_left(s)
        right = self.outer_right(s)
        outer = left[:, :, None, :] * right[:, None, :, :]
        z = z + outer / math.sqrt(max(z.shape[-1], 1))
        z = z + self.pair_ff(self.pair_ln(z))

        pair_mask = residue_mask[:, :, None] & residue_mask[:, None, :]
        z = z * pair_mask[..., None]
        s = s * residue_mask[..., None]
        return s, z


class ESMFoldSeedModel(nn.Module):
    """Single-sequence, ESMFold-style trunk with C-alpha and auxiliary heads."""

    def __init__(
        self,
        d_model: int,
        n_layers: int,
        n_heads: int,
        pair_dim: int,
        relpos_bins: int,
        distogram_bins: int,
        plddt_bins: int,
        max_seq_len: int,
        dropout: float,
    ):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.distogram_bins = distogram_bins
        self.plddt_bins = plddt_bins

        self.aa_embed = nn.Embedding(21, d_model)
        self.pos_embed = nn.Embedding(max_seq_len, d_model)
        self.state_in = nn.LayerNorm(d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.single_encoder = nn.TransformerEncoder(enc_layer, num_layers=max(1, n_layers // 2))

        self.relpos = RelativePosition(bins=relpos_bins, pairwise_state_dim=pair_dim)
        self.pair_init_left = nn.Linear(d_model, pair_dim)
        self.pair_init_right = nn.Linear(d_model, pair_dim)
        self.pair_init_ln = nn.LayerNorm(pair_dim)

        self.blocks = nn.ModuleList(
            [ESMFoldSeedBlock(d_model=d_model, n_heads=n_heads, pair_dim=pair_dim, dropout=dropout) for _ in range(n_layers)]
        )
        self.out_ln = nn.LayerNorm(d_model)
        self.coord_head = nn.Linear(d_model, 3)

        # ESMFold-style auxiliary prediction heads.
        self.distogram_head = nn.Linear(pair_dim, distogram_bins)
        self.plddt_head = nn.Linear(d_model, plddt_bins)

    def forward(self, aatype: torch.Tensor, residue_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        # aatype: (B,L), residue_mask: (B,L)
        B, L = aatype.shape
        if L > self.max_seq_len:
            raise ValueError(f"Sequence length {L} exceeds model.max_seq_len={self.max_seq_len}")

        pos_ids = torch.arange(L, device=aatype.device, dtype=torch.long)[None, :].expand(B, L)
        s = self.aa_embed(aatype.clamp(min=0, max=20)) + self.pos_embed(pos_ids)
        s = self.state_in(s)

        key_padding_mask = ~residue_mask
        s = self.single_encoder(s, src_key_padding_mask=key_padding_mask)
        s = s * residue_mask[..., None]

        residue_index = pos_ids
        z = self.relpos(residue_index=residue_index, mask=residue_mask)
        z = z + self.pair_init_left(s)[:, :, None, :] + self.pair_init_right(s)[:, None, :, :]
        z = self.pair_init_ln(z)

        for block in self.blocks:
            s, z = block(s, z, residue_mask=residue_mask)

        s_out = self.out_ln(s)
        pred_ca = self.coord_head(s_out) * residue_mask[..., None]

        dist_logits = self.distogram_head(z)
        dist_logits = 0.5 * (dist_logits + dist_logits.transpose(1, 2))

        plddt_logits = self.plddt_head(s_out)
        return {
            "pred_ca": pred_ca,
            "distogram_logits": dist_logits,
            "plddt_logits": plddt_logits,
        }


def build_model(cfg: Dict[str, Any]) -> nn.Module:
    model_cfg = cfg["model"]
    return ESMFoldSeedModel(
        d_model=int(model_cfg.get("d_model", 256)),
        n_layers=int(model_cfg.get("n_layers", 6)),
        n_heads=int(model_cfg.get("n_heads", 8)),
        pair_dim=int(model_cfg.get("pair_dim", 128)),
        relpos_bins=int(model_cfg.get("relpos_bins", 32)),
        distogram_bins=int(model_cfg.get("distogram_bins", 64)),
        plddt_bins=int(model_cfg.get("plddt_bins", 50)),
        max_seq_len=int(model_cfg.get("max_seq_len", 1024)),
        dropout=float(model_cfg.get("dropout", 0.1)),
    )
