from __future__ import annotations

import math

import torch

from train import _cfg_with_runtime, _summarize_eval_metrics, masked_kabsch_rmsd


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


def test_summarize_eval_metrics_reports_nan_score_for_empty_eval_loader() -> None:
    metrics = _summarize_eval_metrics([], [])

    assert "val_loss" not in metrics
    assert math.isnan(metrics["val_lddt_ca"])


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
