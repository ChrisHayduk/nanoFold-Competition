from __future__ import annotations

import math
import random

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from eval import _load_feature_crop, _load_label_crop
from nanofold.chain_paths import chain_npz_path
from train import (
    WeightedDistributedSampler,
    _append_scalar_output_metrics,
    _capture_rng_state,
    _cfg_with_runtime,
    _epoch_aware_sampler,
    _isolated_validation,
    _scalar_output_metrics,
    _summarize_eval_metrics,
    _validate_train_sample_weights_opt_in,
    batch_lddt_ca,
    batch_rmsd_ca,
    masked_kabsch_rmsd,
)


def test_summarize_eval_metrics_omits_loss_when_submission_does_not_return_one() -> None:
    metrics = _summarize_eval_metrics([torch.tensor(0.25), torch.tensor(0.75)], [])

    assert metrics["val_lddt_ca"] == 0.5
    assert metrics["val_target_count"] == metrics["val_lddt_ca_valid_count"] == 2
    assert metrics["val_lddt_ca_nonfinite_count"] == metrics["val_lddt_ca_missing_count"] == 0
    assert "val_loss" not in metrics


def test_summarize_eval_metrics_includes_loss_when_available() -> None:
    metrics = _summarize_eval_metrics(
        [torch.tensor(0.25), torch.tensor(0.75)],
        [torch.tensor(1.0), torch.tensor(3.0)],
    )

    assert metrics["val_lddt_ca"] == 0.5
    assert metrics["val_loss"] == 2.0
    assert metrics["val_loss_valid_count"] == 2


def test_summarize_eval_metrics_includes_rmsd_when_available() -> None:
    metrics = _summarize_eval_metrics(
        [torch.tensor(0.25), torch.tensor(0.75)],
        [torch.tensor(1.0), torch.tensor(3.0)],
        [torch.tensor(2.0), torch.tensor(4.0)],
        [torch.tensor(6.0), torch.tensor(8.0)],
    )

    assert {key: metrics[key] for key in ("val_lddt_ca", "val_loss", "val_rmsd_ca", "val_rmsd_atom14")} == {
        "val_lddt_ca": 0.5,
        "val_loss": 2.0,
        "val_rmsd_ca": 3.0,
        "val_rmsd_atom14": 7.0,
    }


def test_public_eval_metrics_weight_each_valid_target_equally_across_unequal_batches() -> None:
    true = torch.tensor([
        [[0., 0., 0.], [1., 0., 0.], [0., 2., 0.], [0., 0., 3.]],
    ]).repeat(3, 1, 1)
    predicted = true.clone()
    predicted[1] *= 3
    predicted[2] *= 2
    mask = torch.ones((3, 4), dtype=torch.bool)
    mask[1, 2:] = False  # This target has no defined Kabsch RMSD.
    residue_mask = torch.ones_like(mask)

    def aggregate(slices):
        scores, rmsds = [], []
        for indices in slices:
            args = (predicted[indices], true[indices], mask[indices], residue_mask[indices])
            scores.append(batch_lddt_ca(*args))
            rmsds.append(batch_rmsd_ca(*args))
        return _summarize_eval_metrics(scores, [], rmsds)

    individual = aggregate([slice(i, i + 1) for i in range(3)])
    assert individual["val_rmsd_ca"] > 0
    assert individual["val_rmsd_ca_valid_count"] == 2
    assert individual["val_rmsd_ca_nonfinite_count"] == 1
    assert aggregate([slice(0, 2), slice(2, 3)]) == pytest.approx(individual)
    assert aggregate([slice(0, 1), slice(1, 3)]) == pytest.approx(individual)


def test_public_eval_scalar_outputs_weight_batch_means_by_target_count_and_keep_valid_items() -> None:
    scalar_metrics = {}
    _append_scalar_output_metrics(scalar_metrics, {
        "loss_component": torch.tensor(2.0),
        "per_target": torch.tensor([1.0, float("nan")]),
    }, batch_size=2)
    _append_scalar_output_metrics(scalar_metrics, {
        "loss_component": torch.tensor(5.0), "per_target": torch.tensor([3.0]),
    }, batch_size=1)
    metrics = _summarize_eval_metrics(
        [torch.ones(3)], [torch.tensor([2., 2.]), torch.tensor([5.])], scalar_metrics=scalar_metrics,
    )
    assert metrics["val_loss"] == pytest.approx(3.0)
    assert metrics["val_loss_component"] == pytest.approx(3.0)
    assert metrics["val_per_target"] == pytest.approx(2.0)
    assert metrics["val_per_target_valid_count"] == 2
    assert metrics["val_per_target_nonfinite_count"] == 1


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


@pytest.mark.parametrize("scores,expected", [([], None), ([torch.ones(2)], 3), ([torch.tensor([1., float('nan')])], 2)])
def test_public_eval_rejects_empty_missing_or_nonfinite_primary_scores(scores, expected) -> None:
    with pytest.raises(ValueError, match="every target|missing or nonfinite"):
        _summarize_eval_metrics(scores, [], expected_targets=expected)


def test_public_eval_diagnostics_cannot_overwrite_trusted_scores_or_denominators() -> None:
    metrics = _summarize_eval_metrics(
        [torch.tensor([0.2, 0.6])], [], expected_targets=2,
        scalar_metrics={"lddt_ca": [torch.tensor([99., 99.])], "target_count": [torch.tensor(99.)],
                        "foldscore": [torch.tensor(99.)], "lddt_ca_valid_count": [torch.tensor(99.)],
                        "auxiliary": [torch.tensor([float("nan"), float("nan")])]},
        foldscore_metric_values={"foldscore": [torch.tensor([0.1, 0.3])]},
    )
    assert metrics["val_target_count"] == metrics["val_lddt_ca_valid_count"] == 2
    assert metrics["val_lddt_ca"] == pytest.approx(0.4)
    assert metrics["val_foldscore"] == pytest.approx(0.2)
    assert metrics["val_auxiliary_valid_count"] == 0
    assert metrics["val_auxiliary_nonfinite_count"] == 2
    assert math.isnan(metrics["val_auxiliary"])


def test_failed_validation_restores_training_rng_loader_and_mixed_module_modes() -> None:
    model = torch.nn.Sequential(torch.nn.Linear(1, 1), torch.nn.Dropout())
    model.train()
    model[0].eval()
    generator = torch.Generator().manual_seed(43)
    loader = DataLoader(TensorDataset(torch.arange(3)), generator=generator)
    device = torch.device("cpu")
    before = _capture_rng_state(device)
    loader_before = generator.get_state().clone()
    with pytest.raises(RuntimeError, match="scoring failure"):
        with _isolated_validation(model, loader, device):
            assert not any(module.training for module in model.modules())
            random.random()
            np.random.random()
            torch.rand(3)
            next(iter(loader))
            raise RuntimeError("scoring failure")
    after = _capture_rng_state(device)
    assert before["python"] == after["python"]
    assert before["numpy"] == after["numpy"]
    assert torch.equal(before["torch"], after["torch"])
    assert torch.equal(loader_before, generator.get_state())
    assert model.training and not model[0].training and model[1].training


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


def test_weighted_distributed_sampler_is_epoch_aware() -> None:
    weights = torch.arange(1, 21, dtype=torch.float32)
    sampler = WeightedDistributedSampler(weights, num_replicas=2, rank=0, seed=17)

    epoch0 = list(iter(sampler))
    sampler.set_epoch(1)
    epoch1 = list(iter(sampler))

    assert len(epoch0) == 10
    assert len(epoch1) == 10
    assert epoch0 != epoch1
    assert _epoch_aware_sampler(sampler) is sampler


def test_train_sample_weights_require_explicit_opt_in() -> None:
    with pytest.raises(ValueError, match="replacement-based weighted sampling"):
        _validate_train_sample_weights_opt_in({"train_sample_weights": "weights.json"})

    _validate_train_sample_weights_opt_in(
        {"train_sample_weights": "weights.json", "allow_train_sample_weights": True}
    )
