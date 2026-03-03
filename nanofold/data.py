from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class ExamplePaths:
    # For a chain id like '7KDX_A'
    chain_id: str
    a3m_path: Path
    mmcif_path: Path


def read_manifest(manifest_path: str | Path) -> List[str]:
    manifest_path = Path(manifest_path)
    ids: List[str] = []
    for line in manifest_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        ids.append(line)
    if not ids:
        raise ValueError(f"Empty manifest: {manifest_path}")
    return ids


class ProcessedNPZDataset(Dataset):
    """Loads per-chain preprocessed .npz examples produced by scripts/preprocess.py."""

    def __init__(self, processed_dir: str | Path, manifest_path: str | Path):
        self.processed_dir = Path(processed_dir)
        self.chain_ids = read_manifest(manifest_path)

    def __len__(self) -> int:
        return len(self.chain_ids)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        chain_id = self.chain_ids[idx]
        path = self.processed_dir / f"{chain_id}.npz"
        data = np.load(path)
        L = int(data["aatype"].shape[0])

        if "template_aatype" in data:
            template_aatype = torch.from_numpy(data["template_aatype"]).long()
        else:
            template_aatype = torch.zeros((0, L), dtype=torch.long)

        if "template_ca_coords" in data:
            template_ca_coords = torch.from_numpy(data["template_ca_coords"]).float()
        else:
            template_ca_coords = torch.zeros((0, L, 3), dtype=torch.float32)

        if "template_ca_mask" in data:
            template_ca_mask = torch.from_numpy(data["template_ca_mask"]).bool()
        else:
            template_ca_mask = torch.zeros((0, L), dtype=torch.bool)

        out = {
            "chain_id": chain_id,
            "aatype": torch.from_numpy(data["aatype"]).long(),                  # (L,)
            "msa": torch.from_numpy(data["msa"]).long(),                        # (N,L)
            "deletions": torch.from_numpy(data["deletions"]).long(),            # (N,L)
            "ca_coords": torch.from_numpy(data["ca_coords"]).float(),           # (L,3)
            "ca_mask": torch.from_numpy(data["ca_mask"]).bool(),                # (L,)
            "template_aatype": template_aatype,                                  # (T,L)
            "template_ca_coords": template_ca_coords,                            # (T,L,3)
            "template_ca_mask": template_ca_mask,                                # (T,L)
        }
        return out


def random_crop(
    aatype: torch.Tensor,
    msa: torch.Tensor,
    deletions: torch.Tensor,
    ca_coords: torch.Tensor,
    ca_mask: torch.Tensor,
    template_aatype: torch.Tensor,
    template_ca_coords: torch.Tensor,
    template_ca_mask: torch.Tensor,
    crop_size: int,
) -> Dict[str, torch.Tensor]:
    L = aatype.shape[0]
    if L <= crop_size:
        return {
            "aatype": aatype,
            "msa": msa,
            "deletions": deletions,
            "ca_coords": ca_coords,
            "ca_mask": ca_mask,
            "template_aatype": template_aatype,
            "template_ca_coords": template_ca_coords,
            "template_ca_mask": template_ca_mask,
        }

    # Simple contiguous crop (AlphaFold uses more complex masking; keep it simple).
    start = torch.randint(low=0, high=L - crop_size + 1, size=(1,)).item()
    end = start + crop_size

    return {
        "aatype": aatype[start:end],
        "msa": msa[:, start:end],
        "deletions": deletions[:, start:end],
        "ca_coords": ca_coords[start:end],
        "ca_mask": ca_mask[start:end],
        "template_aatype": template_aatype[:, start:end],
        "template_ca_coords": template_ca_coords[:, start:end],
        "template_ca_mask": template_ca_mask[:, start:end],
    }


def sample_msa(msa: torch.Tensor, deletions: torch.Tensor, msa_depth: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Subsample MSA rows to a fixed depth, always keeping row 0 (query)."""
    N = msa.shape[0]
    if N <= msa_depth:
        return msa, deletions

    # Keep query at index 0; sample the rest.
    perm = torch.randperm(N - 1) + 1
    keep = torch.cat([torch.zeros(1, dtype=torch.long), perm[: msa_depth - 1]])
    keep = keep.sort().values

    return msa[keep], deletions[keep]


def collate_batch(
    examples: List[Dict[str, torch.Tensor]],
    crop_size: int,
    msa_depth: int,
) -> Dict[str, torch.Tensor]:
    # This baseline uses batch_size=1 in the config by default.
    # Still implement stacking for convenience.
    batch = []
    chain_ids = []
    for ex in examples:
        chain_ids.append(ex["chain_id"])
        cropped = random_crop(
            ex["aatype"],
            ex["msa"],
            ex["deletions"],
            ex["ca_coords"],
            ex["ca_mask"],
            ex["template_aatype"],
            ex["template_ca_coords"],
            ex["template_ca_mask"],
            crop_size=crop_size,
        )
        msa_s, del_s = sample_msa(cropped["msa"], cropped["deletions"], msa_depth=msa_depth)
        cropped["msa"] = msa_s
        cropped["deletions"] = del_s
        batch.append(cropped)

    # Stack with padding (only needed if batch>1 or variable length). We'll pad to max L in batch.
    max_L = max(item["aatype"].shape[0] for item in batch)
    max_N = max(item["msa"].shape[0] for item in batch)
    max_T = max(item["template_aatype"].shape[0] for item in batch)

    def pad_1d(x: torch.Tensor, pad_value: int = 0) -> torch.Tensor:
        L = x.shape[0]
        if L == max_L:
            return x
        out = torch.full((max_L,), pad_value, dtype=x.dtype)
        out[:L] = x
        return out

    def pad_2d(x: torch.Tensor, pad_value: int = 0) -> torch.Tensor:
        N, L = x.shape
        out = torch.full((max_N, max_L), pad_value, dtype=x.dtype)
        out[:N, :L] = x
        return out

    def pad_coords(x: torch.Tensor) -> torch.Tensor:
        L = x.shape[0]
        out = torch.zeros((max_L, 3), dtype=x.dtype)
        out[:L] = x
        return out

    def pad_template_2d(x: torch.Tensor, pad_value: int = 0) -> torch.Tensor:
        T, L = x.shape
        out = torch.full((max_T, max_L), pad_value, dtype=x.dtype)
        out[:T, :L] = x
        return out

    def pad_template_coords(x: torch.Tensor) -> torch.Tensor:
        T, L, _ = x.shape
        out = torch.zeros((max_T, max_L, 3), dtype=x.dtype)
        out[:T, :L] = x
        return out

    aatype = torch.stack([pad_1d(item["aatype"], pad_value=0) for item in batch], dim=0)          # (B,L)
    msa = torch.stack([pad_2d(item["msa"], pad_value=21) for item in batch], dim=0)              # (B,N,L), pad with GAP_ID=21
    deletions = torch.stack([pad_2d(item["deletions"], pad_value=0) for item in batch], dim=0)    # (B,N,L)
    ca_coords = torch.stack([pad_coords(item["ca_coords"]) for item in batch], dim=0)            # (B,L,3)
    ca_mask = torch.stack([pad_1d(item["ca_mask"].to(torch.int64), pad_value=0).bool() for item in batch], dim=0)  # (B,L)
    template_aatype = torch.stack([pad_template_2d(item["template_aatype"], pad_value=20) for item in batch], dim=0)  # (B,T,L)
    template_ca_coords = torch.stack([pad_template_coords(item["template_ca_coords"]) for item in batch], dim=0)        # (B,T,L,3)
    template_ca_mask = torch.stack(
        [pad_template_2d(item["template_ca_mask"].to(torch.int64), pad_value=0).bool() for item in batch], dim=0
    )  # (B,T,L)

    # Additionally provide an overall residue mask for padding positions.
    residue_mask = torch.stack([pad_1d(torch.ones_like(item["aatype"], dtype=torch.bool), pad_value=0) for item in batch], dim=0)

    return {
        "chain_id": chain_ids,
        "aatype": aatype,
        "msa": msa,
        "deletions": deletions,
        "ca_coords": ca_coords,
        "ca_mask": ca_mask,
        "template_aatype": template_aatype,
        "template_ca_coords": template_ca_coords,
        "template_ca_mask": template_ca_mask,
        "residue_mask": residue_mask,
    }
