from __future__ import annotations

import math
from typing import Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


# This file is an OpenFold/AlphaFold2-style architecture adaptation for the benchmark.
# It is inspired by publicly available OpenFold code and AF2 architectural descriptions:
# - OpenFold repo: https://github.com/aqlaboratory/openfold
# - AlphaFold2 paper/supplement (Evoformer + Structure Module concepts)
# See THIRD_PARTY_LICENSES.md for attribution.


def _sequence_mask_to_pair(mask: torch.Tensor) -> torch.Tensor:
    return mask[:, :, None] & mask[:, None, :]


class RelativePosition(nn.Module):
    def __init__(self, bins: int, c_z: int):
        super().__init__()
        self.bins = bins
        self.linear = nn.Linear(2 * bins + 2, c_z)

    def forward(self, residue_index: torch.Tensor, residue_mask: torch.Tensor) -> torch.Tensor:
        diff = residue_index[:, None, :] - residue_index[:, :, None]
        diff = diff.clamp(-self.bins, self.bins) + self.bins + 1
        one_hot = F.one_hot(diff, num_classes=2 * self.bins + 2).float()
        z = self.linear(one_hot)
        z = z * _sequence_mask_to_pair(residue_mask)[..., None]
        return z


class TemplatePairEmbedder(nn.Module):
    def __init__(self, c_z: int, dist_bins: int, min_bin: float, max_bin: float):
        super().__init__()
        self.dist_bins = dist_bins
        self.min_bin = min_bin
        self.max_bin = max_bin
        # distogram onehot + pair mask + aatype_i onehot + aatype_j onehot
        self.linear = nn.Linear(dist_bins + 1 + 21 + 21, c_z)

    def forward(
        self,
        template_aatype: torch.Tensor,      # (B,T,L)
        template_ca_coords: torch.Tensor,   # (B,T,L,3)
        template_ca_mask: torch.Tensor,     # (B,T,L)
        residue_mask: torch.Tensor,         # (B,L)
    ) -> torch.Tensor:
        B, T, L = template_aatype.shape
        device = template_aatype.device
        dtype = template_ca_coords.dtype

        if T == 0:
            return torch.zeros((B, L, L, self.linear.out_features), device=device, dtype=dtype)

        d = torch.cdist(template_ca_coords, template_ca_coords, p=2)  # (B,T,L,L)
        boundaries = torch.linspace(self.min_bin, self.max_bin, self.dist_bins - 1, device=device, dtype=dtype)
        d_bin = torch.bucketize(d, boundaries).long()
        d_oh = F.one_hot(d_bin, num_classes=self.dist_bins).float()  # (B,T,L,L,D)

        tmask_pair = template_ca_mask[:, :, :, None] & template_ca_mask[:, :, None, :]  # (B,T,L,L)
        tmask_pair_f = tmask_pair.float()[..., None]  # (B,T,L,L,1)

        aatype_oh = F.one_hot(template_aatype.clamp(min=0, max=20), num_classes=21).float()  # (B,T,L,21)
        aai = aatype_oh[:, :, :, None, :].expand(B, T, L, L, 21)
        aaj = aatype_oh[:, :, None, :, :].expand(B, T, L, L, 21)

        feat = torch.cat([d_oh, tmask_pair_f, aai, aaj], dim=-1)
        emb = self.linear(feat)  # (B,T,L,L,Cz)
        emb = emb * tmask_pair_f

        denom = tmask_pair_f.sum(dim=1).clamp(min=1.0)  # (B,L,L,1)
        z_tpl = emb.sum(dim=1) / denom  # (B,L,L,Cz)
        z_tpl = z_tpl * _sequence_mask_to_pair(residue_mask)[..., None]
        return z_tpl


class MSARowAttentionWithPairBias(nn.Module):
    def __init__(self, c_m: int, c_z: int, n_heads: int, dropout: float):
        super().__init__()
        if c_m % n_heads != 0:
            raise ValueError(f"c_m ({c_m}) must be divisible by n_heads ({n_heads})")
        self.n_heads = n_heads
        self.head_dim = c_m // n_heads
        self.scale = self.head_dim ** -0.5
        self.dropout = float(dropout)

        self.q_proj = nn.Linear(c_m, c_m)
        self.k_proj = nn.Linear(c_m, c_m)
        self.v_proj = nn.Linear(c_m, c_m)
        self.o_proj = nn.Linear(c_m, c_m)
        self.pair_bias = nn.Linear(c_z, n_heads, bias=False)
        self.ln = nn.LayerNorm(c_m)

    def forward(self, m: torch.Tensor, z: torch.Tensor, residue_mask: torch.Tensor, msa_mask: torch.Tensor) -> torch.Tensor:
        # m: (B,N,L,Cm), z: (B,L,L,Cz), residue_mask: (B,L), msa_mask: (B,N)
        x = self.ln(m)
        B, N, L, C = x.shape
        H = self.n_heads
        D = self.head_dim

        q = self.q_proj(x).view(B, N, L, H, D).permute(0, 1, 3, 2, 4)  # (B,N,H,L,D)
        k = self.k_proj(x).view(B, N, L, H, D).permute(0, 1, 3, 2, 4)
        v = self.v_proj(x).view(B, N, L, H, D).permute(0, 1, 3, 2, 4)

        bias = self.pair_bias(z).permute(0, 3, 1, 2)[:, None, :, :, :]  # (B,1,H,L,L)
        logits = torch.matmul(q, k.transpose(-2, -1)) * self.scale + bias  # (B,N,H,L,L)

        key_ok = residue_mask[:, None, None, None, :]
        query_ok = residue_mask[:, None, None, :, None]
        row_ok = msa_mask[:, :, None, None, None]
        logits = logits.masked_fill(~key_ok, -1.0e4)
        logits = logits.masked_fill(~query_ok, -1.0e4)
        logits = logits.masked_fill(~row_ok, -1.0e4)

        attn = torch.softmax(logits, dim=-1)
        if self.training and self.dropout > 0:
            attn = F.dropout(attn, p=self.dropout)

        out = torch.matmul(attn, v)  # (B,N,H,L,D)
        out = out.permute(0, 1, 3, 2, 4).contiguous().view(B, N, L, C)
        out = self.o_proj(out)
        out = out * msa_mask[:, :, None, None]
        return out


class MSAColumnAttention(nn.Module):
    def __init__(self, c_m: int, n_heads: int, dropout: float):
        super().__init__()
        self.ln = nn.LayerNorm(c_m)
        self.attn = nn.MultiheadAttention(c_m, n_heads, dropout=dropout, batch_first=True)

    def forward(self, m: torch.Tensor, msa_mask: torch.Tensor) -> torch.Tensor:
        # m: (B,N,L,Cm), msa_mask: (B,N)
        x = self.ln(m)
        B, N, L, C = x.shape

        # Attend over N for each residue position L.
        x_col = x.permute(0, 2, 1, 3).contiguous().view(B * L, N, C)  # (B*L,N,C)
        key_padding = (~msa_mask[:, None, :]).expand(B, L, N).reshape(B * L, N)
        out, _ = self.attn(x_col, x_col, x_col, key_padding_mask=key_padding, need_weights=False)
        out = out.view(B, L, N, C).permute(0, 2, 1, 3).contiguous()
        out = out * msa_mask[:, :, None, None]
        return out


class Transition(nn.Module):
    def __init__(self, c: int, mult: int, dropout: float):
        super().__init__()
        self.ln = nn.LayerNorm(c)
        self.net = nn.Sequential(
            nn.Linear(c, mult * c),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mult * c, c),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(self.ln(x))


class OuterProductMean(nn.Module):
    def __init__(self, c_m: int, c_z: int, c_hidden: int):
        super().__init__()
        self.ln = nn.LayerNorm(c_m)
        self.left = nn.Linear(c_m, c_hidden)
        self.right = nn.Linear(c_m, c_hidden)
        self.out = nn.Linear(c_hidden, c_z)

    def forward(self, m: torch.Tensor, msa_mask: torch.Tensor, residue_mask: torch.Tensor) -> torch.Tensor:
        # m: (B,N,L,Cm)
        x = self.ln(m)
        row_mask = msa_mask[:, :, None, None].float()
        denom = row_mask.sum(dim=1).clamp(min=1.0)  # (B,1,1)

        l = (self.left(x) * row_mask).sum(dim=1) / denom  # (B,L,H)
        r = (self.right(x) * row_mask).sum(dim=1) / denom  # (B,L,H)
        opm = l[:, :, None, :] * r[:, None, :, :]  # (B,L,L,H)
        z = self.out(opm)
        z = z * _sequence_mask_to_pair(residue_mask)[..., None]
        return z


class TriangleMultiplicativeUpdate(nn.Module):
    def __init__(self, c_z: int, c_hidden: int, outgoing: bool):
        super().__init__()
        self.outgoing = outgoing
        self.ln = nn.LayerNorm(c_z)
        self.a = nn.Linear(c_z, c_hidden)
        self.b = nn.Linear(c_z, c_hidden)
        self.out = nn.Linear(c_hidden, c_z)

    def forward(self, z: torch.Tensor, residue_mask: torch.Tensor) -> torch.Tensor:
        # z: (B,L,L,Cz)
        x = self.ln(z)
        a = self.a(x)
        b = self.b(x)
        if self.outgoing:
            # i-k and j-k interactions
            u = torch.einsum("bikc,bjkc->bijc", a, b)
        else:
            # k-i and k-j interactions
            u = torch.einsum("bkic,bkjc->bijc", a, b)
        u = self.out(u / math.sqrt(max(z.shape[1], 1)))
        u = u * _sequence_mask_to_pair(residue_mask)[..., None]
        return u


class PairAxialAttention(nn.Module):
    def __init__(self, c_z: int, n_heads: int, dropout: float, along_row: bool):
        super().__init__()
        self.along_row = along_row
        self.ln = nn.LayerNorm(c_z)
        self.attn = nn.MultiheadAttention(c_z, n_heads, dropout=dropout, batch_first=True)

    def forward(self, z: torch.Tensor, residue_mask: torch.Tensor) -> torch.Tensor:
        # z: (B,L,L,Cz)
        x = self.ln(z)
        B, L, _, C = x.shape
        if self.along_row:
            # sequences are j for each i
            x_in = x.view(B * L, L, C)
            key_padding = (~residue_mask[:, None, :]).expand(B, L, L).reshape(B * L, L)
            out, _ = self.attn(x_in, x_in, x_in, key_padding_mask=key_padding, need_weights=False)
            out = out.view(B, L, L, C)
        else:
            xt = x.transpose(1, 2).contiguous()
            x_in = xt.view(B * L, L, C)
            key_padding = (~residue_mask[:, None, :]).expand(B, L, L).reshape(B * L, L)
            out, _ = self.attn(x_in, x_in, x_in, key_padding_mask=key_padding, need_weights=False)
            out = out.view(B, L, L, C).transpose(1, 2).contiguous()
        out = out * _sequence_mask_to_pair(residue_mask)[..., None]
        return out


class EvoformerBlock(nn.Module):
    def __init__(
        self,
        c_m: int,
        c_z: int,
        n_heads_msa: int,
        n_heads_pair: int,
        c_hidden_opm: int,
        c_hidden_tri: int,
        dropout: float,
    ):
        super().__init__()
        self.msa_row_attn = MSARowAttentionWithPairBias(c_m=c_m, c_z=c_z, n_heads=n_heads_msa, dropout=dropout)
        self.msa_col_attn = MSAColumnAttention(c_m=c_m, n_heads=n_heads_msa, dropout=dropout)
        self.msa_transition = Transition(c=c_m, mult=4, dropout=dropout)

        self.opm = OuterProductMean(c_m=c_m, c_z=c_z, c_hidden=c_hidden_opm)
        self.tri_out = TriangleMultiplicativeUpdate(c_z=c_z, c_hidden=c_hidden_tri, outgoing=True)
        self.tri_in = TriangleMultiplicativeUpdate(c_z=c_z, c_hidden=c_hidden_tri, outgoing=False)
        self.pair_attn_row = PairAxialAttention(c_z=c_z, n_heads=n_heads_pair, dropout=dropout, along_row=True)
        self.pair_attn_col = PairAxialAttention(c_z=c_z, n_heads=n_heads_pair, dropout=dropout, along_row=False)
        self.pair_transition = Transition(c=c_z, mult=2, dropout=dropout)

    def forward(
        self,
        m: torch.Tensor,
        z: torch.Tensor,
        residue_mask: torch.Tensor,
        msa_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        m = m + self.msa_row_attn(m, z, residue_mask=residue_mask, msa_mask=msa_mask)
        m = m + self.msa_col_attn(m, msa_mask=msa_mask)
        m = m + self.msa_transition(m)
        m = m * msa_mask[:, :, None, None]

        z = z + self.opm(m, msa_mask=msa_mask, residue_mask=residue_mask)
        z = z + self.tri_out(z, residue_mask=residue_mask)
        z = z + self.tri_in(z, residue_mask=residue_mask)
        z = z + self.pair_attn_row(z, residue_mask=residue_mask)
        z = z + self.pair_attn_col(z, residue_mask=residue_mask)
        z = z + self.pair_transition(z)
        z = z * _sequence_mask_to_pair(residue_mask)[..., None]
        return m, z


class PairBiasedSelfAttention(nn.Module):
    def __init__(self, c_s: int, c_z: int, n_heads: int, dropout: float):
        super().__init__()
        if c_s % n_heads != 0:
            raise ValueError(f"c_s ({c_s}) must be divisible by n_heads ({n_heads})")
        self.n_heads = n_heads
        self.head_dim = c_s // n_heads
        self.scale = self.head_dim ** -0.5
        self.dropout = float(dropout)

        self.q_proj = nn.Linear(c_s, c_s)
        self.k_proj = nn.Linear(c_s, c_s)
        self.v_proj = nn.Linear(c_s, c_s)
        self.o_proj = nn.Linear(c_s, c_s)
        self.pair_bias = nn.Linear(c_z, n_heads, bias=False)
        self.ln = nn.LayerNorm(c_s)

    def forward(self, s: torch.Tensor, z: torch.Tensor, residue_mask: torch.Tensor) -> torch.Tensor:
        x = self.ln(s)
        B, L, C = x.shape
        H = self.n_heads
        D = self.head_dim
        q = self.q_proj(x).view(B, L, H, D).transpose(1, 2)
        k = self.k_proj(x).view(B, L, H, D).transpose(1, 2)
        v = self.v_proj(x).view(B, L, H, D).transpose(1, 2)

        bias = self.pair_bias(z).permute(0, 3, 1, 2)
        logits = torch.matmul(q, k.transpose(-2, -1)) * self.scale + bias
        key_ok = residue_mask[:, None, None, :]
        query_ok = residue_mask[:, None, :, None]
        logits = logits.masked_fill(~key_ok, -1.0e4)
        logits = logits.masked_fill(~query_ok, -1.0e4)
        attn = torch.softmax(logits, dim=-1)
        if self.training and self.dropout > 0:
            attn = F.dropout(attn, p=self.dropout)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, L, C)
        return self.o_proj(out) * residue_mask[..., None]


class StructureBlock(nn.Module):
    def __init__(self, c_s: int, c_z: int, n_heads: int, dropout: float):
        super().__init__()
        self.attn = PairBiasedSelfAttention(c_s=c_s, c_z=c_z, n_heads=n_heads, dropout=dropout)
        self.ff = Transition(c=c_s, mult=4, dropout=dropout)

    def forward(self, s: torch.Tensor, z: torch.Tensor, residue_mask: torch.Tensor) -> torch.Tensor:
        s = s + self.attn(s, z, residue_mask=residue_mask)
        s = s + self.ff(s)
        s = s * residue_mask[..., None]
        return s


class OpenFoldSeedModel(nn.Module):
    """OpenFold-like seed architecture using MSA + template features."""

    def __init__(
        self,
        c_m: int,
        c_z: int,
        c_s: int,
        n_evoformer_blocks: int,
        n_structure_blocks: int,
        n_heads_msa: int,
        n_heads_pair: int,
        n_heads_structure: int,
        relpos_bins: int,
        template_dist_bins: int,
        template_dist_min: float,
        template_dist_max: float,
        distogram_bins: int,
        plddt_bins: int,
        dropout: float,
    ):
        super().__init__()
        self.distogram_bins = distogram_bins
        self.plddt_bins = plddt_bins

        self.target_embed = nn.Embedding(21, c_m)
        self.msa_embed = nn.Embedding(23, c_m)
        self.deletion_embed = nn.Linear(1, c_m)
        self.query_proj = nn.Linear(c_m, c_m, bias=False)
        self.msa_ln = nn.LayerNorm(c_m)

        self.single_proj = nn.Linear(c_m, c_s)
        self.pair_i_proj = nn.Linear(c_m, c_z, bias=False)
        self.pair_j_proj = nn.Linear(c_m, c_z, bias=False)
        self.relpos = RelativePosition(bins=relpos_bins, c_z=c_z)
        self.template_embed = TemplatePairEmbedder(
            c_z=c_z,
            dist_bins=template_dist_bins,
            min_bin=template_dist_min,
            max_bin=template_dist_max,
        )

        self.evoformer = nn.ModuleList(
            [
                EvoformerBlock(
                    c_m=c_m,
                    c_z=c_z,
                    n_heads_msa=n_heads_msa,
                    n_heads_pair=n_heads_pair,
                    c_hidden_opm=max(16, c_z // 2),
                    c_hidden_tri=max(16, c_z // 2),
                    dropout=dropout,
                )
                for _ in range(n_evoformer_blocks)
            ]
        )

        self.structure_blocks = nn.ModuleList(
            [
                StructureBlock(c_s=c_s, c_z=c_z, n_heads=n_heads_structure, dropout=dropout)
                for _ in range(n_structure_blocks)
            ]
        )
        self.single_out_ln = nn.LayerNorm(c_s)
        self.coord_head = nn.Linear(c_s, 3)
        self.distogram_head = nn.Linear(c_z, distogram_bins)
        self.plddt_head = nn.Linear(c_s, plddt_bins)

    def forward(
        self,
        aatype: torch.Tensor,            # (B,L)
        msa: torch.Tensor,               # (B,N,L)
        deletions: torch.Tensor,         # (B,N,L)
        residue_mask: torch.Tensor,      # (B,L)
        template_aatype: torch.Tensor,   # (B,T,L)
        template_ca_coords: torch.Tensor,  # (B,T,L,3)
        template_ca_mask: torch.Tensor,  # (B,T,L)
    ) -> Dict[str, torch.Tensor]:
        B, L = aatype.shape
        N = msa.shape[1]
        device = aatype.device

        target = self.target_embed(aatype.clamp(min=0, max=20))  # (B,L,Cm)
        msa_tok = self.msa_embed(msa.clamp(min=0, max=22))       # (B,N,L,Cm)
        del_feat = self.deletion_embed(deletions.float().unsqueeze(-1))
        m = self.msa_ln(msa_tok + del_feat + self.query_proj(target)[:, None, :, :])

        # MSA row mask: padded rows are all GAP(21).
        msa_mask = (msa != 21).any(dim=-1)  # (B,N)
        if N > 0:
            msa_mask[:, 0] = True  # ensure query row always considered.
        m = m * msa_mask[:, :, None, None]

        residue_index = torch.arange(L, device=device, dtype=torch.long)[None, :].expand(B, L)
        z = self.relpos(residue_index=residue_index, residue_mask=residue_mask)
        z = z + self.pair_i_proj(target)[:, :, None, :] + self.pair_j_proj(target)[:, None, :, :]
        z = z + self.template_embed(
            template_aatype=template_aatype,
            template_ca_coords=template_ca_coords,
            template_ca_mask=template_ca_mask,
            residue_mask=residue_mask,
        )
        z = z * _sequence_mask_to_pair(residue_mask)[..., None]

        for block in self.evoformer:
            m, z = block(m, z, residue_mask=residue_mask, msa_mask=msa_mask)

        s = self.single_proj(m[:, 0]) * residue_mask[..., None]
        for block in self.structure_blocks:
            s = block(s, z, residue_mask=residue_mask)

        s_out = self.single_out_ln(s)
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
    mc = cfg["model"]
    return OpenFoldSeedModel(
        c_m=int(mc.get("c_m", 128)),
        c_z=int(mc.get("c_z", 64)),
        c_s=int(mc.get("c_s", 128)),
        n_evoformer_blocks=int(mc.get("n_evoformer_blocks", 4)),
        n_structure_blocks=int(mc.get("n_structure_blocks", 4)),
        n_heads_msa=int(mc.get("n_heads_msa", 8)),
        n_heads_pair=int(mc.get("n_heads_pair", 4)),
        n_heads_structure=int(mc.get("n_heads_structure", 8)),
        relpos_bins=int(mc.get("relpos_bins", 32)),
        template_dist_bins=int(mc.get("template_dist_bins", 39)),
        template_dist_min=float(mc.get("template_dist_min", 3.0)),
        template_dist_max=float(mc.get("template_dist_max", 50.0)),
        distogram_bins=int(mc.get("distogram_bins", 64)),
        plddt_bins=int(mc.get("plddt_bins", 50)),
        dropout=float(mc.get("dropout", 0.1)),
    )
