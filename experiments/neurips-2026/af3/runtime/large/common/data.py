"""Deterministic NanoFold tensor loading for distributed reconstruction runs."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


# Canonical AlphaFold atom14 slot occupancy for the 20 standard amino acids in
# ``ARNDCQEGHILKMFPSTWYV`` order plus an all-zero unknown row. This is a
# chemistry-derived feature: unlike the target ``atom14_mask``, it does not
# reveal which atoms happened to be resolved in a deposited structure.
_ATOM14_COUNTS = (
    5,
    11,
    8,
    8,
    6,
    9,
    9,
    4,
    10,
    8,
    8,
    9,
    8,
    11,
    7,
    6,
    7,
    14,
    12,
    7,
)
STANDARD_ATOM14_MASK = np.zeros((21, 14), dtype=np.bool_)
for _restype_index, _atom_count in enumerate(_ATOM14_COUNTS):
    STANDARD_ATOM14_MASK[_restype_index, :_atom_count] = True


def chemistry_atom14_mask(aatype: np.ndarray, token_mask: np.ndarray) -> np.ndarray:
    """Return atom14 occupancy from residue identity, never target resolution."""
    residue_types = np.asarray(aatype, dtype=np.int64)
    valid_tokens = np.asarray(token_mask, dtype=np.bool_)
    if residue_types.shape != valid_tokens.shape:
        raise ValueError("aatype and token_mask must have identical shapes")
    if np.any((residue_types < 0) | (residue_types > 20)):
        bad = np.unique(
            residue_types[(residue_types < 0) | (residue_types > 20)]
        ).tolist()
        raise ValueError(f"unexpected aatype tokens: {bad}")
    return STANDARD_ATOM14_MASK[residue_types] & valid_tokens[..., None]


def read_manifest(path: str | Path) -> list[str]:
    values = [
        line.strip()
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if not values:
        raise ValueError(f"empty manifest: {path}")
    if len(values) != len(set(values)):
        raise ValueError(f"duplicate chain IDs in manifest: {path}")
    return values


def chain_npz_path(root: str | Path, chain_id: str) -> Path:
    return Path(root) / f"chain_{chain_id.encode('utf-8').hex()}.npz"


def _seed64(*parts: object) -> int:
    digest = hashlib.sha256(
        "\x1f".join(str(part) for part in parts).encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "little", signed=False)


@dataclass(frozen=True)
class DeterministicBatchSchedule:
    """Maps (optimizer step, rank) to a chain without mutable sampler state."""

    chain_ids: tuple[str, ...]
    seed: int
    world_size: int

    def __post_init__(self) -> None:
        if not self.chain_ids:
            raise ValueError("chain_ids must not be empty")
        if self.world_size < 1:
            raise ValueError("world_size must be positive")
        if len(self.chain_ids) % self.world_size:
            raise ValueError(
                "manifest length must be divisible by world_size for exact epochs"
            )

    def chain_at(self, step: int, rank: int) -> tuple[str, int, int]:
        if step < 0 or not 0 <= rank < self.world_size:
            raise ValueError("invalid step or rank")
        global_position = step * self.world_size + rank
        epoch, offset = divmod(global_position, len(self.chain_ids))
        permutation = np.random.default_rng(
            _seed64("nanofold-permutation", self.seed, epoch)
        ).permutation(len(self.chain_ids))
        chain_id = self.chain_ids[int(permutation[offset])]
        return chain_id, epoch, offset


class NanoFoldTensorLoader:
    """Loads one reproducible cropped chain at a time from official NPZ tensors."""

    def __init__(
        self,
        *,
        data_root: str | Path,
        crop_size: int = 256,
        msa_depth: int = 192,
        seed: int = 0,
    ) -> None:
        self.data_root = Path(data_root)
        self.feature_root = self.data_root / "processed_features"
        self.label_root = self.data_root / "processed_labels"
        self.crop_size = int(crop_size)
        self.msa_depth = int(msa_depth)
        self.seed = int(seed)
        if self.crop_size < 1 or self.msa_depth < 1:
            raise ValueError("crop_size and msa_depth must be positive")

    @staticmethod
    def _map_msa_tokens(msa: np.ndarray) -> np.ndarray:
        mapped = np.asarray(msa, dtype=np.int64).copy()
        if np.any((mapped < 0) | (mapped > 21)):
            bad = np.unique(mapped[(mapped < 0) | (mapped > 21)]).tolist()
            raise ValueError(f"unexpected NanoFold MSA tokens: {bad}")
        mapped[mapped == 21] = 31
        return mapped

    def load(
        self,
        chain_id: str,
        *,
        train: bool,
        step: int = 0,
        rank: int = 0,
    ) -> dict[str, Any]:
        rng = np.random.default_rng(
            _seed64("nanofold-example", self.seed, step, rank, chain_id, int(train))
        )
        feature_path = chain_npz_path(self.feature_root, chain_id)
        label_path = chain_npz_path(self.label_root, chain_id)
        with np.load(feature_path, allow_pickle=False) as features:
            aatype_full = np.asarray(features["aatype"], dtype=np.int64)
            msa_full = self._map_msa_tokens(features["msa"])
            deletions_full = np.asarray(features["deletions"], dtype=np.float32)
            residue_index_full = np.asarray(features["residue_index"], dtype=np.int64)
        with np.load(label_path, allow_pickle=False) as labels:
            atom14_positions_full = np.asarray(
                labels["atom14_positions"], dtype=np.float32
            )
            atom14_mask_full = np.asarray(labels["atom14_mask"], dtype=np.bool_)

        length = int(aatype_full.shape[0])
        if msa_full.ndim != 2 or msa_full.shape[1] != length:
            raise ValueError(f"{feature_path}: inconsistent MSA dimensions")
        if deletions_full.shape != msa_full.shape:
            raise ValueError(f"{feature_path}: MSA/deletion shape mismatch")
        if atom14_positions_full.shape != (length, 14, 3):
            raise ValueError(f"{label_path}: unexpected atom14 coordinate shape")
        if atom14_mask_full.shape != (length, 14):
            raise ValueError(f"{label_path}: unexpected atom14 mask shape")

        # AF3 summary features are defined over the complete supplied MSA before
        # row subsampling and residue cropping.
        profile_full = np.zeros((length, 32), dtype=np.float32)
        for token in range(32):
            profile_full[:, token] = np.mean(msa_full == token, axis=0)
        deletion_mean_full = np.mean(deletions_full, axis=0, dtype=np.float32)

        crop_length = min(length, self.crop_size)
        if train and length > crop_length:
            crop_start = int(rng.integers(0, length - crop_length + 1))
        else:
            crop_start = (length - crop_length) // 2
        crop_slice = slice(crop_start, crop_start + crop_length)

        n_msa = int(msa_full.shape[0])
        if n_msa < 1:
            raise ValueError(f"{feature_path}: MSA must contain the query row")
        keep_count = min(n_msa, self.msa_depth)
        if train and n_msa > keep_count:
            non_query = rng.choice(
                np.arange(1, n_msa, dtype=np.int64),
                size=keep_count - 1,
                replace=False,
            )
            msa_rows = np.concatenate((np.array([0], dtype=np.int64), non_query))
        else:
            msa_rows = np.arange(keep_count, dtype=np.int64)

        aatype = aatype_full[crop_slice]
        msa = msa_full[msa_rows, crop_slice]
        deletions = deletions_full[msa_rows, crop_slice]
        residue_index = residue_index_full[crop_slice]
        profile = profile_full[crop_slice]
        deletion_mean = deletion_mean_full[crop_slice]
        atom14_positions = atom14_positions_full[crop_slice]
        atom14_mask = atom14_mask_full[crop_slice]

        token_mask = np.zeros((self.crop_size,), dtype=np.bool_)
        token_mask[:crop_length] = True
        padded_aatype = np.zeros((self.crop_size,), dtype=np.int64)
        padded_aatype[:crop_length] = aatype
        padded_residue_index = np.zeros((self.crop_size,), dtype=np.int64)
        padded_residue_index[:crop_length] = residue_index
        padded_profile = np.zeros((self.crop_size, 32), dtype=np.float32)
        padded_profile[:crop_length] = profile
        padded_deletion_mean = np.zeros((self.crop_size,), dtype=np.float32)
        padded_deletion_mean[:crop_length] = deletion_mean

        padded_msa = np.full((self.msa_depth, self.crop_size), 31, dtype=np.int64)
        padded_deletions = np.zeros((self.msa_depth, self.crop_size), dtype=np.float32)
        msa_row_mask = np.zeros((self.msa_depth,), dtype=np.bool_)
        padded_msa[:keep_count, :crop_length] = msa
        padded_deletions[:keep_count, :crop_length] = deletions
        msa_row_mask[:keep_count] = True

        padded_positions = np.zeros((self.crop_size, 14, 3), dtype=np.float32)
        padded_atom_mask = np.zeros((self.crop_size, 14), dtype=np.bool_)
        padded_positions[:crop_length] = atom14_positions
        padded_atom_mask[:crop_length] = atom14_mask
        # Preserve the raw deposited coordinate array for corruption-frame
        # reconstruction. The loss/scoring copy below is centred with the
        # resolved-atom mask, but that mask must not influence model inputs.
        padded_positions_uncentered = padded_positions.copy()
        padded_chemistry_mask = chemistry_atom14_mask(padded_aatype, token_mask)

        # Centre only by observed atoms. Random rigid rotation belongs in the
        # model-specific diffusion augmentation, not in data provenance.
        observed = padded_positions[padded_atom_mask]
        coordinate_centre = (
            observed.mean(axis=0, dtype=np.float32)
            if observed.size
            else np.zeros((3,), dtype=np.float32)
        )
        padded_positions[padded_atom_mask] -= coordinate_centre

        mapped_restype = F.one_hot(
            torch.from_numpy(padded_aatype), num_classes=32
        ).float()
        deletion_value = (2.0 / math.pi) * torch.atan(
            torch.from_numpy(padded_deletions) / 3.0
        )
        return {
            "chain_id": chain_id,
            "source_length": length,
            "crop_start": crop_start,
            "crop_length": crop_length,
            "aatype": torch.from_numpy(padded_aatype).long(),
            "restype": mapped_restype,
            "msa": torch.from_numpy(padded_msa).long(),
            "msa_mask": torch.from_numpy(msa_row_mask[:, None] & token_mask[None, :]),
            "has_deletion": torch.from_numpy(padded_deletions > 0).float(),
            "deletion_value": deletion_value.float(),
            "profile": torch.from_numpy(padded_profile).float(),
            "deletion_mean": torch.from_numpy(padded_deletion_mean).float(),
            "residue_index": torch.from_numpy(padded_residue_index).long(),
            "token_mask": torch.from_numpy(token_mask),
            "atom14_positions": torch.from_numpy(padded_positions).float(),
            "atom14_positions_uncentered": torch.from_numpy(
                padded_positions_uncentered
            ).float(),
            # Supervision-only mask: records which target coordinates exist.
            "atom14_mask": torch.from_numpy(padded_atom_mask),
            # Public-input mask: determined solely by aatype and padding.
            "atom14_chemistry_mask": torch.from_numpy(padded_chemistry_mask),
            "coordinate_centre": torch.from_numpy(coordinate_centre).float(),
        }
