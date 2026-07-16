from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from nanofold.chain_paths import chain_npz_path
from nanofold.data import LengthBucketBatchSampler
from train import make_loader


def test_length_bucket_sampler_is_deterministic_and_keeps_batches_local() -> None:
    lengths = [40, 41, 42, 43, 120, 121, 122, 123, 250, 251]
    first = LengthBucketBatchSampler(
        lengths,
        batch_size=2,
        generator=torch.Generator().manual_seed(7),
        bucket_size=2,
        drop_last=True,
    )
    second = LengthBucketBatchSampler(
        lengths,
        batch_size=2,
        generator=torch.Generator().manual_seed(7),
        bucket_size=2,
        drop_last=True,
    )
    first_batches = list(first)
    second_batches = list(second)
    assert first_batches == second_batches
    assert len(first_batches) == len(lengths) // 2
    assert all(max(lengths[index] for index in batch) - min(lengths[index] for index in batch) <= 3 for batch in first_batches)


def test_length_bucket_sampler_keeps_partial_batch_when_requested() -> None:
    sampler = LengthBucketBatchSampler(
        [1, 2, 3, 4, 5],
        batch_size=2,
        generator=torch.Generator().manual_seed(0),
        bucket_size=2,
        drop_last=False,
    )
    batches = list(sampler)
    assert len(batches) == 3
    assert sorted(index for batch in batches for index in batch) == [0, 1, 2, 3, 4]


def test_make_loader_uses_length_buckets_for_training(tmp_path: Path) -> None:
    features = tmp_path / "features"
    labels = tmp_path / "labels"
    features.mkdir()
    labels.mkdir()
    chain_ids = ["1abc_A", "2abc_A", "3abc_A", "4abc_A"]
    lengths = [8, 9, 20, 21]
    for chain_id, length in zip(chain_ids, lengths, strict=True):
        np.savez(
            chain_npz_path(features, chain_id),
            aatype=np.zeros((length,), dtype=np.int64),
            msa=np.zeros((2, length), dtype=np.int64),
            deletions=np.zeros((2, length), dtype=np.int64),
            residue_index=np.arange(length, dtype=np.int64),
            between_segment_residues=np.zeros((length,), dtype=np.int64),
            template_aatype=np.zeros((0, length), dtype=np.int64),
            template_ca_coords=np.zeros((0, length, 3), dtype=np.float32),
            template_ca_mask=np.zeros((0, length), dtype=bool),
        )
        np.savez(
            chain_npz_path(labels, chain_id),
            ca_coords=np.zeros((length, 3), dtype=np.float32),
            ca_mask=np.ones((length,), dtype=bool),
            atom14_positions=np.zeros((length, 14, 3), dtype=np.float32),
            atom14_mask=np.ones((length, 14), dtype=bool),
        )
    manifest = tmp_path / "train.txt"
    manifest.write_text("\n".join(chain_ids) + "\n")
    cfg = {
        "data": {
            "processed_features_dir": str(features),
            "processed_labels_dir": str(labels),
            "train_manifest": str(manifest),
            "val_manifest": str(manifest),
            "crop_size": 32,
            "msa_depth": 2,
            "batch_size": 2,
            "num_workers": 0,
            "train_crop_mode": "random",
            "val_crop_mode": "center",
            "train_msa_sample_mode": "top",
            "val_msa_sample_mode": "top",
            "bucket_by_length": True,
            "length_bucket_size": 2,
        }
    }
    loader = make_loader(
        cfg,
        "train",
        device=torch.device("cpu"),
        include_labels=True,
        fail_if_labels_present=False,
        allow_missing=False,
        generator_seed=0,
    )
    batches = list(loader)
    assert len(batches) == 2
    assert sorted(tuple(batch["aatype"].shape) for batch in batches) == [(2, 9), (2, 21)]
