from __future__ import annotations

from pathlib import Path

import numpy as np

from nanofold.data import ProcessedNPZDataset, collate_batch
from nanofold.dataset_integrity import validate_npz_schema


def _write_example_npz(path: Path, L: int = 8, N: int = 4, T: int = 1) -> None:
    np.savez_compressed(
        path,
        aatype=np.zeros((L,), dtype=np.int32),
        msa=np.zeros((N, L), dtype=np.int32),
        deletions=np.zeros((N, L), dtype=np.int32),
        ca_coords=np.zeros((L, 3), dtype=np.float32),
        ca_mask=np.ones((L,), dtype=bool),
        template_aatype=np.zeros((T, L), dtype=np.int32),
        template_ca_coords=np.zeros((T, L, 3), dtype=np.float32),
        template_ca_mask=np.ones((T, L), dtype=bool),
    )


def test_processed_npz_shapes_and_collate(tmp_path: Path) -> None:
    processed = tmp_path / "processed"
    processed.mkdir()
    manifest = tmp_path / "train.txt"
    cid = "1abc_A"
    manifest.write_text(f"{cid}\n")
    npz_path = processed / f"{cid}.npz"
    _write_example_npz(npz_path, L=12, N=5, T=2)

    assert validate_npz_schema(npz_path) == []

    ds = ProcessedNPZDataset(processed_dir=processed, manifest_path=manifest, allow_missing=False)
    item = ds[0]
    batch = collate_batch([item], crop_size=12, msa_depth=5, crop_mode="center", msa_sample_mode="top")
    assert batch["aatype"].shape == (1, 12)
    assert batch["msa"].shape == (1, 5, 12)
    assert batch["ca_coords"].shape == (1, 12, 3)
    assert batch["template_ca_coords"].shape == (1, 2, 12, 3)
