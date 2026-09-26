"""Validated deterministic crops shared by both scoring entrypoints."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .chain_paths import chain_npz_path
from .residue_constants import CA_ATOM14_SLOT


def _load_arrays(directory: Path, chain_id: str) -> dict[str, np.ndarray]:
    path = chain_npz_path(directory, chain_id)
    with np.load(path, allow_pickle=False) as data:
        wanted = {"chain_id", "ca_coords", "ca_mask", "atom14_positions", "atom14_mask", "aatype", "residue_index", "between_segment_residues"}
        arrays = {key: np.asarray(data[key]) for key in data.files if key in wanted}
    if "chain_id" in arrays:
        identity = arrays["chain_id"]
        if identity.shape != () or str(identity.item()) != chain_id:
            raise ValueError(f"Stored chain_id does not match scoring chain {chain_id}.")
    return arrays


def _crop(
    values: dict[str, torch.Tensor], arrays: dict[str, np.ndarray], *, length: int, crop_size: int, crop_mode: str,
) -> dict[str, torch.Tensor]:
    if length <= 0 or crop_size <= 0 or crop_mode != "center":
        raise ValueError("External scoring requires nonempty inputs and a positive deterministic center crop.")
    if "residue_index" in arrays:
        indices = arrays["residue_index"]
        if indices.shape != (length,) or indices.dtype.kind not in "iu":
            raise ValueError("Scoring residue_index must contain one integer index per residue.")
        values["residue_index"] = torch.from_numpy(indices.astype(np.int64))
    if "between_segment_residues" in arrays:
        breaks = arrays["between_segment_residues"]
        if breaks.shape != (length,) or breaks.dtype.kind not in "biu" or np.any((breaks != 0) & (breaks != 1)):
            raise ValueError("Scoring between_segment_residues must contain one binary flag per residue.")
        values["between_segment_residues"] = torch.from_numpy(breaks.astype(np.int64))
    start = max(0, (length - crop_size) // 2)
    out = {key: value[start : start + crop_size] for key, value in values.items()}
    out["full_length"] = torch.tensor(length)
    return out


def load_label_crop(*, labels_dir: Path, chain_id: str, crop_size: int, crop_mode: str) -> dict[str, torch.Tensor]:
    arrays = _load_arrays(labels_dir, chain_id)
    required = ("ca_coords", "ca_mask", "atom14_positions", "atom14_mask")
    if any(key not in arrays for key in required):
        raise ValueError("Scoring labels require ca_coords, ca_mask, atom14_positions, and atom14_mask.")
    coords = arrays["atom14_positions"]
    if coords.ndim != 3 or coords.shape[1:] != (14, 3):
        raise ValueError("Label atom14_positions must have shape (L, 14, 3).")
    length = coords.shape[0]
    for key, shape in (("ca_coords", (length, 3)), ("ca_mask", (length,)), ("atom14_mask", (length, 14))):
        if arrays[key].shape != shape:
            raise ValueError(f"Label {key} must have shape {shape}.")
    for key in ("ca_coords", "atom14_positions"):
        if arrays[key].dtype.kind != "f" or not np.isfinite(arrays[key]).all():
            raise ValueError(f"Label {key} must contain finite floating point coordinates.")
    for key in ("ca_mask", "atom14_mask"):
        if arrays[key].dtype != np.bool_:
            raise ValueError(f"Label {key} must have boolean dtype.")
    if not np.array_equal(arrays["ca_mask"], arrays["atom14_mask"][:, CA_ATOM14_SLOT]):
        raise ValueError("Label ca_mask and atom14 CA mask disagree.")
    ca_mask = arrays["ca_mask"]
    if not ca_mask.any():
        raise ValueError("Scoring labels must contain at least one observed CA atom.")
    if not np.allclose(arrays["ca_coords"][ca_mask], coords[ca_mask, CA_ATOM14_SLOT], rtol=0, atol=1e-5):
        raise ValueError("Label ca_coords and atom14 CA coordinates disagree.")
    values = {key: torch.from_numpy(arrays[key].astype(np.float32) if "mask" not in key else arrays[key]) for key in required}
    if any(not bool(torch.isfinite(values[key]).all()) for key in ("ca_coords", "atom14_positions")):
        raise ValueError("Label coordinates must remain finite in float32.")
    return _crop(values, arrays, length=length, crop_size=crop_size, crop_mode=crop_mode)


def load_feature_crop(*, features_dir: Path, chain_id: str, crop_size: int, crop_mode: str) -> dict[str, torch.Tensor]:
    arrays = _load_arrays(features_dir, chain_id)
    if "aatype" not in arrays:
        raise ValueError("Scoring features require aatype.")
    tokens = arrays["aatype"]
    if tokens.ndim != 1 or tokens.dtype.kind not in "iu" or np.any(tokens < 0) or np.any(tokens > 20):
        raise ValueError("Feature aatype must contain integer residue tokens in [0, 20].")
    arrays.setdefault("residue_index", np.arange(tokens.shape[0], dtype=np.int64))
    arrays.setdefault("between_segment_residues", np.zeros(tokens.shape[0], dtype=np.int64))
    return _crop(
        {"aatype": torch.from_numpy(tokens.astype(np.int64))}, arrays,
        length=tokens.shape[0], crop_size=crop_size, crop_mode=crop_mode,
    )


def validate_scoring_alignment(labels: dict[str, torch.Tensor], features: dict[str, torch.Tensor]) -> None:
    for key in ("full_length", "residue_index", "between_segment_residues"):
        if key in labels and key in features and not torch.equal(labels[key].cpu(), features[key].cpu()):
            raise ValueError(f"Scoring feature and label {key} disagree.")
