from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from eval import _load_feature_crop, _load_label_crop
from nanofold.chain_paths import chain_npz_path
from train import (
    _cfg_with_runtime,
    _scalar_output_metrics,
    _summarize_eval_metrics,
    masked_kabsch_rmsd,
)


def test_summarize_eval_metrics_omits_loss_when_submission_does_not_return_one() -> None:
    metrics = _summarize_eval_metrics([torch.tensor(0.25), torch.tensor(0.75)], [])

    assert metrics == {"val_lddt_ca": 0.5}


def test_summarize_eval_metrics_includes_loss_when_available() -> None:
    metrics = _summarize_eval_metrics(
        [torch.tensor(0.25), torch.tensor(0.75)],
        [torch.tensor(1.0), torch.tensor(3.0)],
    )

    assert metrics == {"val_lddt_ca": 0.5, "val_loss": 2.0}


def test_summarize_eval_metrics_includes_rmsd_when_available() -> None:
    metrics = _summarize_eval_metrics(
        [torch.tensor(0.25), torch.tensor(0.75)],
        [torch.tensor(1.0), torch.tensor(3.0)],
        [torch.tensor(2.0), torch.tensor(4.0)],
        [torch.tensor(6.0), torch.tensor(8.0)],
    )

    assert metrics == {
        "val_lddt_ca": 0.5,
        "val_loss": 2.0,
        "val_rmsd_ca": 3.0,
        "val_rmsd_atom14": 7.0,
    }


def test_summarize_eval_metrics_includes_loss_components_and_foldscore_components() -> None:
    metrics = _summarize_eval_metrics(
        [torch.tensor(0.25), torch.tensor(0.75)],
        [torch.tensor(1.0), torch.tensor(3.0)],
        scalar_metrics={
            "loss_fape_loss": [torch.tensor(2.0), torch.tensor(4.0)],
            "loss_distogram_loss": [torch.tensor(0.5), torch.tensor(1.5)],
        },
        foldscore_metric_values={
            "foldscore": [torch.tensor(0.2), torch.tensor(0.6)],
            "gdt_ha_ca": [torch.tensor(0.1), torch.tensor(0.3)],
        },
    )

    assert metrics["val_loss"] == 2.0
    assert metrics["val_loss_fape_loss"] == 3.0
    assert metrics["val_loss_distogram_loss"] == 1.0
    assert metrics["val_foldscore"] == pytest.approx(0.4)
    assert metrics["val_gdt_ha_ca"] == pytest.approx(0.2)


def test_scalar_output_metrics_keeps_finite_scalar_tensors_only() -> None:
    metrics = _scalar_output_metrics(
        {
            "loss": torch.tensor(1.0),
            "pred_atom14": torch.zeros(1, 2, 14, 3),
            "pred_ca": torch.zeros(1, 2, 3),
            "loss_fape_loss": torch.tensor([1.0, 3.0]),
            "loss_nan_term": torch.tensor(float("nan")),
            "count": 7,
        }
    )

    assert set(metrics) == {"loss_fape_loss"}
    assert metrics["loss_fape_loss"].item() == 2.0


def test_summarize_eval_metrics_reports_nan_score_for_empty_eval_loader() -> None:
    metrics = _summarize_eval_metrics([], [])

    assert "val_loss" not in metrics
    assert math.isnan(metrics["val_lddt_ca"])


def test_eval_external_score_loaders_use_encoded_chain_paths(tmp_path) -> None:
    chain_id = "2mda_B"
    labels_dir = tmp_path / "labels"
    features_dir = tmp_path / "features"
    labels_dir.mkdir()
    features_dir.mkdir()
    np.savez_compressed(
        chain_npz_path(labels_dir, chain_id),
        ca_coords=np.zeros((4, 3), dtype=np.float32),
        ca_mask=np.ones((4,), dtype=bool),
        atom14_positions=np.zeros((4, 14, 3), dtype=np.float32),
        atom14_mask=np.ones((4, 14), dtype=bool),
    )
    np.savez_compressed(
        chain_npz_path(features_dir, chain_id),
        aatype=np.zeros((4,), dtype=np.int64),
    )

    labels = _load_label_crop(labels_dir=labels_dir, chain_id=chain_id, crop_size=8, crop_mode="center")
    features = _load_feature_crop(features_dir=features_dir, chain_id=chain_id, crop_size=8, crop_mode="center")

    assert not (labels_dir / f"{chain_id}.npz").exists()
    assert labels["atom14_positions"].shape == (4, 14, 3)
    assert features["aatype"].shape == (4,)


def test_masked_kabsch_rmsd_is_invariant_to_global_frame() -> None:
    true_points = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 0.0, 3.0],
            [1.0, 2.0, 3.0],
        ]
    )
    theta = 0.7
    rotation = torch.tensor(
        [
            [math.cos(theta), -math.sin(theta), 0.0],
            [math.sin(theta), math.cos(theta), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    pred_points = true_points @ rotation.T + torch.tensor([100.0, -25.0, 7.0])

    rmsd = masked_kabsch_rmsd(pred_points, true_points, torch.ones(5, dtype=torch.bool))

    assert rmsd.item() < 1.0e-5


def test_cfg_with_runtime_does_not_mutate_base_config() -> None:
    cfg = {"train": {"max_steps": 10000}}

    runtime_cfg = _cfg_with_runtime(
        cfg,
        step=123,
        cumulative_samples_seen=246,
        max_steps=10000,
        sample_budget=20000,
    )

    assert "_runtime" not in cfg
    assert runtime_cfg["_runtime"] == {
        "step": 123,
        "cumulative_samples_seen": 246,
        "max_steps": 10000,
        "sample_budget": 20000,
    }
