from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .a3m import GAP_ID, MSA_ALPHABET_SIZE, SEQ_ALPHABET_SIZE, UNK_ID
from .metrics import lddt_ca


def msa_profile(msa_tokens: torch.Tensor, residue_mask: torch.Tensor) -> torch.Tensor:
    """Compute a simple MSA profile: per-position AA frequencies.

    msa_tokens: (B, N, L) int
    residue_mask: (B, L) bool (True for real residues, False for pad)

    Returns:
      profile: (B, L, 21) frequencies over {20 AAs + UNK}, ignoring gaps.
    """
    tokens = msa_tokens.clamp(min=0, max=MSA_ALPHABET_SIZE - 1)

    # Ignore padding positions (residue_mask False) by setting everything to GAP.
    tokens = torch.where(residue_mask[:, None, :], tokens, torch.full_like(tokens, GAP_ID))

    # Mask gaps
    is_gap = tokens == GAP_ID
    tokens_nogap = torch.where(is_gap, torch.full_like(tokens, UNK_ID), tokens)
    one_hot = F.one_hot(tokens_nogap, num_classes=MSA_ALPHABET_SIZE).float()[..., :SEQ_ALPHABET_SIZE]  # (B,N,L,21)
    weights = (~is_gap).float()[..., None]  # (B,N,L,1)

    prof = (one_hot * weights).sum(dim=1)  # (B,L,21)
    denom = weights.sum(dim=1).clamp(min=1.0)
    prof = prof / denom
    return prof


class NanoFoldBaseline(nn.Module):
    """A deliberately small baseline: Transformer encoder -> Cα coordinates.

    Inputs:
      aatype: (B,L) int in [0,20]
      msa: (B,N,L) int in [0,22]
      deletions: (B,N,L) int
      residue_mask: (B,L) bool

    Outputs:
      pred_ca: (B,L,3)
    """

    def __init__(self, d_model: int, n_layers: int, n_heads: int, dropout: float, max_seq_len: int = 1024):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = int(max_seq_len)

        self.aa_embed = nn.Embedding(SEQ_ALPHABET_SIZE, d_model)
        self.pos_embed = nn.Embedding(self.max_seq_len, d_model)
        self.prof_proj = nn.Linear(SEQ_ALPHABET_SIZE, d_model, bias=False)
        self.del_proj = nn.Linear(1, d_model, bias=False)

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

    def forward(
        self,
        aatype: torch.Tensor,
        msa: torch.Tensor,
        deletions: torch.Tensor,
        residue_mask: torch.Tensor,
    ) -> torch.Tensor:
        # aatype: (B,L), msa/deletions: (B,N,L)
        B, L = aatype.shape
        if L > self.max_seq_len:
            raise ValueError(f"Sequence length {L} exceeds model.max_seq_len={self.max_seq_len}")

        pos_idx = torch.arange(L, device=aatype.device, dtype=torch.long)[None, :].expand(B, L)
        x = self.aa_embed(aatype.clamp(min=0, max=SEQ_ALPHABET_SIZE - 1)) + self.pos_embed(pos_idx)  # (B,L,d)

        prof = msa_profile(msa, residue_mask=residue_mask)  # (B,L,21)
        x = x + self.prof_proj(prof)

        del_feat = torch.log1p(deletions.float()).mean(dim=1)  # (B,L)
        del_feat = torch.where(residue_mask, del_feat, torch.zeros_like(del_feat))
        x = x + self.del_proj(del_feat.unsqueeze(-1))

        key_padding_mask = ~residue_mask
        x = self.encoder(x, src_key_padding_mask=key_padding_mask)

        pred_ca = self.coord_head(x) * residue_mask[..., None]
        return pred_ca


def _valid_pair_mask(valid_res: torch.Tensor, *, upper_only: bool) -> torch.Tensor:
    B, L = valid_res.shape
    device = valid_res.device
    eye = torch.eye(L, dtype=torch.bool, device=device)[None, :, :]
    pair_mask = valid_res[:, :, None] & valid_res[:, None, :] & (~eye)
    if upper_only:
        tri = torch.triu(torch.ones((L, L), dtype=torch.bool, device=device), diagonal=1)[None, :, :]
        pair_mask = pair_mask & tri
    return pair_mask


def distogram_loss(
    pred_ca: torch.Tensor,
    true_ca: torch.Tensor,
    ca_mask: torch.Tensor,
    residue_mask: torch.Tensor,
    cutoff: float = 15.0,
) -> torch.Tensor:
    """Distance-matrix MSE on pairs with true distance < cutoff."""
    d_true = torch.cdist(true_ca, true_ca, p=2)  # (B,L,L)
    d_pred = torch.cdist(pred_ca, pred_ca, p=2)

    valid_res = ca_mask & residue_mask
    pair_mask = _valid_pair_mask(valid_res, upper_only=False) & (d_true < cutoff)

    diff2 = (d_pred - d_true) ** 2
    if pair_mask.any():
        return diff2[pair_mask].mean()
    return d_pred.sum() * 0.0


def global_distance_smooth_l1_loss(
    pred_ca: torch.Tensor,
    true_ca: torch.Tensor,
    ca_mask: torch.Tensor,
    residue_mask: torch.Tensor,
    max_dist: float = 32.0,
    beta: float = 1.0,
) -> torch.Tensor:
    d_true = torch.cdist(true_ca, true_ca, p=2).clamp(max=max_dist)
    d_pred = torch.cdist(pred_ca, pred_ca, p=2).clamp(max=max_dist)

    valid_res = ca_mask & residue_mask
    pair_mask = _valid_pair_mask(valid_res, upper_only=True)
    if pair_mask.any():
        return F.smooth_l1_loss(d_pred[pair_mask], d_true[pair_mask], beta=beta, reduction="mean")
    return d_pred.sum() * 0.0


def adjacent_bond_smooth_l1_loss(
    pred_ca: torch.Tensor,
    true_ca: torch.Tensor,
    ca_mask: torch.Tensor,
    residue_mask: torch.Tensor,
    beta: float = 0.25,
) -> torch.Tensor:
    valid_res = ca_mask & residue_mask
    pair_valid = valid_res[:, :-1] & valid_res[:, 1:]
    pred_bond = torch.linalg.norm(pred_ca[:, 1:] - pred_ca[:, :-1], dim=-1)
    true_bond = torch.linalg.norm(true_ca[:, 1:] - true_ca[:, :-1], dim=-1)
    if pair_valid.any():
        return F.smooth_l1_loss(pred_bond[pair_valid], true_bond[pair_valid], beta=beta, reduction="mean")
    return pred_bond.sum() * 0.0


def baseline_composite_loss(
    pred_ca: torch.Tensor,
    true_ca: torch.Tensor,
    ca_mask: torch.Tensor,
    residue_mask: torch.Tensor,
    *,
    local_cutoff: float = 15.0,
    global_max_dist: float = 32.0,
    local_weight: float = 1.0,
    global_weight: float = 0.25,
    bond_weight: float = 0.10,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    local = distogram_loss(
        pred_ca=pred_ca,
        true_ca=true_ca,
        ca_mask=ca_mask,
        residue_mask=residue_mask,
        cutoff=local_cutoff,
    )
    global_pair = global_distance_smooth_l1_loss(
        pred_ca=pred_ca,
        true_ca=true_ca,
        ca_mask=ca_mask,
        residue_mask=residue_mask,
        max_dist=global_max_dist,
    )
    bond = adjacent_bond_smooth_l1_loss(
        pred_ca=pred_ca,
        true_ca=true_ca,
        ca_mask=ca_mask,
        residue_mask=residue_mask,
    )
    total = local_weight * local + global_weight * global_pair + bond_weight * bond
    return total, {
        "local_loss": local,
        "global_loss": global_pair,
        "bond_loss": bond,
    }


@torch.no_grad()
def batch_lddt_ca(pred_ca: torch.Tensor, true_ca: torch.Tensor, ca_mask: torch.Tensor, residue_mask: torch.Tensor) -> torch.Tensor:
    scores = []
    B = pred_ca.shape[0]
    for b in range(B):
        scores.append(lddt_ca(pred_ca[b], true_ca[b], ca_mask[b] & residue_mask[b]))
    return torch.stack(scores).mean()
