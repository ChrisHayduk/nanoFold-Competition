from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from .chain_paths import chain_npz_path
from .residue_constants import CA_ATOM14_SLOT

FEATURE_KEYS = (
    "aatype",
    "msa",
    "deletions",
    "residue_index",
    "between_segment_residues",
    "template_aatype",
    "template_ca_coords",
    "template_ca_mask",
)

# Required keys every label NPZ must carry.
LABEL_KEYS = ("ca_coords", "ca_mask", "atom14_positions", "atom14_mask")
LABEL_ONLY_KEYS = (*LABEL_KEYS, "resolution")

# Extra keys produced by the current preprocessing pipeline.
OPTIONAL_LABEL_KEYS = (
    "residue_index",
    "resolution",
)


@dataclass(frozen=True)
class ExamplePaths:
    # For a chain id like '7KDX_A'
    chain_id: str
    a3m_path: Path
    mmcif_path: Path


def read_manifest(manifest_path: str | Path) -> List[str]:
    manifest_path = Path(manifest_path)
    ids: List[str] = []
    seen: set[str] = set()
    for line_number, line in enumerate(manifest_path.read_text().splitlines(), start=1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line in seen:
            raise ValueError(f"Duplicate chain ID at line {line_number} in manifest: {manifest_path}")
        seen.add(line)
        ids.append(line)
    if not ids:
        raise ValueError(f"Empty manifest: {manifest_path}")
    return ids


class ProcessedNPZDataset(Dataset):
    """Loads split per-chain features/labels .npz examples produced by scripts/preprocess.py."""

    def __init__(
        self,
        processed_features_dir: str | Path,
        manifest_path: str | Path,
        *,
        processed_labels_dir: str | Path | None = None,
        include_labels: bool = True,
        allow_missing: bool = False,
        fail_if_labels_present: bool = False,
    ):
        self.processed_features_dir = Path(processed_features_dir)
        self.processed_labels_dir = Path(processed_labels_dir) if processed_labels_dir else None
        self.include_labels = bool(include_labels)
        self.fail_if_labels_present = bool(fail_if_labels_present)
        if self.include_labels and self.processed_labels_dir is None:
            raise ValueError("`include_labels=True` requires `processed_labels_dir`.")

        requested_chain_ids = read_manifest(manifest_path)
        present_chain_ids: List[str] = []
        missing_chain_ids: List[str] = []
        label_conflict_chain_ids: List[str] = []

        for chain_id in requested_chain_ids:
            feature_path = chain_npz_path(self.processed_features_dir, chain_id)
            label_path = chain_npz_path(self.processed_labels_dir, chain_id) if self.processed_labels_dir else None
            has_feature = feature_path.exists()
            has_label = bool(label_path and label_path.exists())

            if self.include_labels:
                if has_feature and has_label:
                    present_chain_ids.append(chain_id)
                else:
                    missing_chain_ids.append(chain_id)
                continue

            if has_feature:
                with np.load(feature_path, allow_pickle=False) as feature_data:
                    if set(feature_data.files).intersection(LABEL_ONLY_KEYS):
                        raise ValueError("Features-only input archive contains label-only fields.")
                if self.fail_if_labels_present and has_label:
                    label_conflict_chain_ids.append(chain_id)
                else:
                    present_chain_ids.append(chain_id)
            else:
                missing_chain_ids.append(chain_id)

        self.missing_chain_ids = missing_chain_ids
        self.label_conflict_chain_ids = label_conflict_chain_ids

        if label_conflict_chain_ids:
            sample = ", ".join(label_conflict_chain_ids[:8])
            raise RuntimeError(
                "Labels are mounted for examples that should be features-only. "
                f"Examples: {sample}. Remove label mount/path for official eval."
            )

        if missing_chain_ids and not allow_missing:
            sample = ", ".join(missing_chain_ids[:8])
            if self.include_labels:
                raise FileNotFoundError(
                    "Missing split preprocessed features/labels for manifest chains in "
                    f"{self.processed_features_dir} and {self.processed_labels_dir}. "
                    f"Examples: {sample}"
                )
            raise FileNotFoundError(
                f"{len(missing_chain_ids)} chains from manifest are missing feature files in "
                f"{self.processed_features_dir}. Examples: {sample}"
            )

        self.chain_ids = present_chain_ids if allow_missing else requested_chain_ids
        if not self.chain_ids:
            raise ValueError(
                "No preprocessed examples available for manifest "
                f"{manifest_path} using features dir {self.processed_features_dir}."
            )

    def __len__(self) -> int:
        return len(self.chain_ids)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        chain_id = self.chain_ids[idx]
        features_path = chain_npz_path(self.processed_features_dir, chain_id)
        if not features_path.exists():
            raise FileNotFoundError(f"Missing feature file: {features_path}")

        with np.load(features_path, allow_pickle=False) as feature_data:
            if "chain_id" in feature_data:
                identity = feature_data["chain_id"]
                if identity.shape != () or str(identity.item()) != chain_id:
                    raise ValueError(f"Feature chain_id does not match manifest chain {chain_id}.")
            L = int(feature_data["aatype"].shape[0])

            template_aatype = (
                torch.from_numpy(feature_data["template_aatype"]).long()
                if "template_aatype" in feature_data
                else torch.zeros((0, L), dtype=torch.long)
            )
            template_ca_coords = (
                torch.from_numpy(feature_data["template_ca_coords"]).float()
                if "template_ca_coords" in feature_data
                else torch.zeros((0, L, 3), dtype=torch.float32)
            )
            template_ca_mask = (
                torch.from_numpy(feature_data["template_ca_mask"]).bool()
                if "template_ca_mask" in feature_data
                else torch.zeros((0, L), dtype=torch.bool)
            )
            residue_index = (
                torch.from_numpy(np.asarray(feature_data["residue_index"], dtype=np.int64))
                if "residue_index" in feature_data
                else torch.arange(L, dtype=torch.long)
            )
            between_segment_residues = (
                torch.from_numpy(np.asarray(feature_data["between_segment_residues"], dtype=np.int64))
                if "between_segment_residues" in feature_data
                else torch.zeros((L,), dtype=torch.long)
            )

            out: Dict[str, Any] = {
                "chain_id": chain_id,
                "aatype": torch.from_numpy(feature_data["aatype"]).long(),  # (L,)
                "msa": torch.from_numpy(feature_data["msa"]).long(),  # (N,L)
                "deletions": torch.from_numpy(feature_data["deletions"]).long(),  # (N,L)
                "residue_index": residue_index,  # (L,)
                "between_segment_residues": between_segment_residues,  # (L,)
                "template_aatype": template_aatype,  # (T,L)
                "template_ca_coords": template_ca_coords,  # (T,L,3)
                "template_ca_mask": template_ca_mask,  # (T,L)
            }

        if self.include_labels:
            assert self.processed_labels_dir is not None
            labels_path = chain_npz_path(self.processed_labels_dir, chain_id)
            if not labels_path.exists():
                raise FileNotFoundError(f"Missing label file: {labels_path}")
            with np.load(labels_path, allow_pickle=False) as label_data:
                if "chain_id" in label_data:
                    identity = label_data["chain_id"]
                    if identity.shape != () or str(identity.item()) != chain_id:
                        raise ValueError(f"Label chain_id does not match manifest chain {chain_id}.")
                missing = [key for key in LABEL_KEYS if key not in label_data]
                if missing:
                    raise KeyError(f"Label file {labels_path} is missing required keys: {', '.join(missing)}")
                expected_shapes = {
                    "ca_coords": (L, 3), "ca_mask": (L,),
                    "atom14_positions": (L, 14, 3), "atom14_mask": (L, 14),
                }
                label_arrays = {key: label_data[key] for key in LABEL_KEYS}
                for key, shape in expected_shapes.items():
                    if label_arrays[key].shape != shape:
                        raise ValueError(f"Label {key} for {chain_id} must have shape {shape}.")
                for key in ("ca_coords", "atom14_positions"):
                    if label_arrays[key].dtype.kind != "f" or not np.isfinite(label_arrays[key]).all():
                        raise ValueError(f"Label {key} must contain finite floating point coordinates.")
                for key in ("ca_mask", "atom14_mask"):
                    if label_arrays[key].dtype != np.bool_:
                        raise ValueError(f"Label {key} must have boolean dtype.")
                ca_mask = label_arrays["ca_mask"]
                if not ca_mask.any() or not np.array_equal(ca_mask, label_arrays["atom14_mask"][:, CA_ATOM14_SLOT]):
                    raise ValueError("Labels require observed CA atoms and matching CA masks.")
                if not np.allclose(label_arrays["ca_coords"][ca_mask], label_arrays["atom14_positions"][ca_mask, CA_ATOM14_SLOT],
                                   rtol=0, atol=1e-5):
                    raise ValueError("Label CA coordinates disagree with atom14 positions.")

                out["ca_coords"] = torch.from_numpy(label_arrays["ca_coords"]).float()  # (L,3)
                out["ca_mask"] = torch.from_numpy(label_arrays["ca_mask"]).bool()  # (L,)
                out["atom14_positions"] = torch.from_numpy(
                    np.asarray(label_arrays["atom14_positions"], dtype=np.float32)
                )  # (L,14,3)
                out["atom14_mask"] = torch.from_numpy(
                    np.asarray(label_arrays["atom14_mask"], dtype=bool)
                )  # (L,14)
                if not torch.isfinite(out["ca_coords"]).all() or not torch.isfinite(out["atom14_positions"]).all():
                    raise ValueError("Label coordinates must remain finite in float32.")
                if "residue_index" in label_data:
                    label_residue_index = torch.from_numpy(
                        np.asarray(label_data["residue_index"], dtype=np.int64)
                    )
                    if label_residue_index.shape != (L,):
                        raise ValueError(f"Label residue_index for {chain_id} must have shape {(L,)}.")
                    if not torch.equal(residue_index, label_residue_index):
                        raise ValueError(f"Feature and label residue_index disagree for {chain_id}.")
                if "resolution" in label_data:
                    out["resolution"] = torch.tensor(
                        float(np.asarray(label_data["resolution"]).item()),
                        dtype=torch.float32,
                    )  # scalar

        return out


def _crop_single_example(example: Dict[str, torch.Tensor], crop_size: int, crop_mode: str) -> Dict[str, torch.Tensor]:
    if type(crop_size) is not int or crop_size <= 0:
        raise ValueError("crop_size must be a positive integer.")
    if crop_mode not in {"random", "center"}:
        raise ValueError(f"Unsupported crop_mode={crop_mode!r}; expected 'random' or 'center'.")
    aatype = example["aatype"]
    L = int(aatype.shape[0])
    if L <= crop_size:
        return dict(example)

    if crop_mode == "random":
        start = int(torch.randint(low=0, high=L - crop_size + 1, size=(1,)).item())
    else:
        start = (L - crop_size) // 2
    end = start + crop_size

    cropped = dict(example)
    cropped["aatype"] = example["aatype"][start:end]
    cropped["msa"] = example["msa"][:, start:end]
    cropped["deletions"] = example["deletions"][:, start:end]
    cropped["template_aatype"] = example["template_aatype"][:, start:end]
    cropped["template_ca_coords"] = example["template_ca_coords"][:, start:end]
    cropped["template_ca_mask"] = example["template_ca_mask"][:, start:end]
    if "between_segment_residues" in example:
        cropped["between_segment_residues"] = example["between_segment_residues"][start:end]
    if "ca_coords" in example:
        cropped["ca_coords"] = example["ca_coords"][start:end]
    if "ca_mask" in example:
        cropped["ca_mask"] = example["ca_mask"][start:end]
    if "atom14_positions" in example:
        cropped["atom14_positions"] = example["atom14_positions"][start:end]
    if "atom14_mask" in example:
        cropped["atom14_mask"] = example["atom14_mask"][start:end]
    if "residue_index" in example:
        cropped["residue_index"] = example["residue_index"][start:end]
    # resolution is a scalar, passes through unchanged.
    return cropped


def sample_msa(
    msa: torch.Tensor,
    deletions: torch.Tensor,
    msa_depth: int,
    sample_mode: str = "random",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Subsample MSA rows to a fixed depth, always keeping row 0 (query)."""
    if type(msa_depth) is not int or msa_depth <= 0:
        raise ValueError("msa_depth must be a positive integer.")
    if sample_mode not in {"random", "top"}:
        raise ValueError(f"Unsupported msa sample_mode={sample_mode!r}; expected 'random' or 'top'.")
    if msa.ndim != 2 or msa.shape != deletions.shape or min(msa.shape) <= 0:
        raise ValueError("msa and deletions must have matching nonempty (N, L) shapes.")
    N = msa.shape[0]
    if N <= msa_depth:
        return msa, deletions

    if sample_mode == "top":
        keep = torch.arange(msa_depth, dtype=torch.long)
        return msa[keep], deletions[keep]

    # Keep query at index 0; sample the rest.
    perm = torch.randperm(N - 1) + 1
    keep = torch.cat([torch.zeros(1, dtype=torch.long), perm[: msa_depth - 1]])
    keep = keep.sort().values
    return msa[keep], deletions[keep]


def collate_batch(
    examples: List[Dict[str, torch.Tensor]],
    crop_size: int,
    msa_depth: int,
    crop_mode: str = "random",
    msa_sample_mode: str = "random",
) -> Dict[str, torch.Tensor]:
    if not examples:
        raise ValueError("Cannot collate an empty batch.")
    # This baseline uses batch_size=1 in the config by default.
    # Still implement stacking for convenience.
    batch: List[Dict[str, torch.Tensor]] = []
    chain_ids: List[str] = []
    has_labels = all(("ca_coords" in ex and "ca_mask" in ex) for ex in examples)
    if any(("ca_coords" in ex or "ca_mask" in ex) for ex in examples) and not has_labels:
        raise ValueError("Inconsistent supervision presence inside one batch.")

    has_atom14 = all(
        ("atom14_positions" in ex and "atom14_mask" in ex) for ex in examples
    )
    if has_labels and not has_atom14:
        raise ValueError("Labelled batches require atom14 supervision.")
    if any(("atom14_positions" in ex or "atom14_mask" in ex) for ex in examples) and not has_atom14:
        raise ValueError("Inconsistent atom14 supervision presence inside one batch.")

    has_residue_index = all("residue_index" in ex for ex in examples)
    if any("residue_index" in ex for ex in examples) and not has_residue_index:
        raise ValueError("Inconsistent residue_index presence inside one batch.")

    has_between_segment_residues = all("between_segment_residues" in ex for ex in examples)
    if any("between_segment_residues" in ex for ex in examples) and not has_between_segment_residues:
        raise ValueError("Inconsistent between_segment_residues presence inside one batch.")

    has_resolution = all("resolution" in ex for ex in examples)
    if any("resolution" in ex for ex in examples) and not has_resolution:
        raise ValueError("Inconsistent resolution presence inside one batch.")

    for ex in examples:
        chain_ids.append(str(ex["chain_id"]))
        cropped = _crop_single_example(ex, crop_size=crop_size, crop_mode=crop_mode)
        msa_s, del_s = sample_msa(
            cropped["msa"],
            cropped["deletions"],
            msa_depth=msa_depth,
            sample_mode=msa_sample_mode,
        )
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

    def pad_atom14_coords(x: torch.Tensor) -> torch.Tensor:
        L, A, _ = x.shape
        out = torch.zeros((max_L, A, 3), dtype=x.dtype)
        out[:L] = x
        return out

    def pad_atom14_mask(x: torch.Tensor) -> torch.Tensor:
        L, A = x.shape
        out = torch.zeros((max_L, A), dtype=x.dtype)
        out[:L] = x
        return out

    aatype = torch.stack([pad_1d(item["aatype"], pad_value=0) for item in batch], dim=0)  # (B,L)
    msa = torch.stack([pad_2d(item["msa"], pad_value=21) for item in batch], dim=0)  # (B,N,L)
    deletions = torch.stack([pad_2d(item["deletions"], pad_value=0) for item in batch], dim=0)  # (B,N,L)
    template_aatype = torch.stack(
        [pad_template_2d(item["template_aatype"], pad_value=20) for item in batch], dim=0
    )  # (B,T,L)
    template_ca_coords = torch.stack(
        [pad_template_coords(item["template_ca_coords"]) for item in batch], dim=0
    )  # (B,T,L,3)
    template_ca_mask = torch.stack(
        [pad_template_2d(item["template_ca_mask"].to(torch.int64), pad_value=0).bool() for item in batch],
        dim=0,
    )  # (B,T,L)

    # Overall residue mask for padding positions.
    residue_mask = torch.stack(
        [pad_1d(torch.ones_like(item["aatype"], dtype=torch.bool), pad_value=0) for item in batch],
        dim=0,
    )
    # Biological gaps remain valid observations. Only batch padding is masked.
    msa_mask = torch.stack(
        [pad_2d(torch.ones_like(item["msa"], dtype=torch.bool), pad_value=0) for item in batch],
        dim=0,
    )

    out: Dict[str, Any] = {
        "chain_id": chain_ids,
        "aatype": aatype,
        "msa": msa,
        "msa_mask": msa_mask,
        "deletions": deletions,
        "template_aatype": template_aatype,
        "template_ca_coords": template_ca_coords,
        "template_ca_mask": template_ca_mask,
        "residue_mask": residue_mask,
    }
    if has_labels:
        out["ca_coords"] = torch.stack([pad_coords(item["ca_coords"]) for item in batch], dim=0)  # (B,L,3)
        out["ca_mask"] = torch.stack(
            [pad_1d(item["ca_mask"].to(torch.int64), pad_value=0).bool() for item in batch], dim=0
        )  # (B,L)
    if has_atom14:
        out["atom14_positions"] = torch.stack(
            [pad_atom14_coords(item["atom14_positions"]) for item in batch], dim=0
        )  # (B,L,14,3)
        out["atom14_mask"] = torch.stack(
            [pad_atom14_mask(item["atom14_mask"].to(torch.int64)).bool() for item in batch], dim=0
        )  # (B,L,14)
    if has_residue_index:
        out["residue_index"] = torch.stack(
            [pad_1d(item["residue_index"].to(torch.int64), pad_value=0) for item in batch], dim=0
        )  # (B,L)
    if has_between_segment_residues:
        out["between_segment_residues"] = torch.stack(
            [pad_1d(item["between_segment_residues"].to(torch.int64), pad_value=0) for item in batch], dim=0
        )  # (B,L)
    if has_resolution:
        out["resolution"] = torch.stack(
            [item["resolution"].float().reshape(()) for item in batch], dim=0
        )  # (B,)
    return out
