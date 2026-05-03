from __future__ import annotations

import torch

from neurips_paper.submission_common.minalphafold2_experiment import build_optimizer, build_scheduler
from neurips_paper.submission_common.muon_optimizer import MuonWithAuxAdam, zeropower_via_newtonschulz5


def _muon_cfg() -> dict:
    return {
        "optim": {
            "name": "muon",
            "lr": 1.0e-3,
            "muon_lr": 2.0e-2,
            "muon_momentum": 0.95,
            "muon_ns_steps": 5,
            "muon_nesterov": True,
            "muon_weight_decay": 0.0,
            "aux_lr": 1.0e-3,
            "aux_beta1": 0.9,
            "aux_beta2": 0.999,
            "aux_eps": 1.0e-6,
            "aux_weight_decay": 0.0,
            "lr_decay_factor": 0.5,
        },
        "train": {
            "max_steps": 12,
            "finetune_start_step": 8,
            "finetune_ramp_steps": 2,
            "finetune_lr_scale": 0.1,
            "warmup_steps": 2,
            "lr_decay_step": 6,
        },
    }


def test_newton_schulz_keeps_matrix_shape_and_finite_values() -> None:
    grad = torch.randn(7, 3)

    update = zeropower_via_newtonschulz5(grad)

    assert update.shape == grad.shape
    assert torch.isfinite(update).all()


def test_muon_with_aux_adam_updates_matrix_and_auxiliary_parameters() -> None:
    matrix = torch.nn.Parameter(torch.randn(4, 4))
    bias = torch.nn.Parameter(torch.randn(4))
    optimizer = MuonWithAuxAdam(
        [
            {"params": [matrix], "use_muon": True, "lr": 0.02},
            {"params": [bias], "use_muon": False, "lr": 0.001, "betas": (0.9, 0.999), "eps": 1.0e-6},
        ]
    )
    matrix_before = matrix.detach().clone()
    bias_before = bias.detach().clone()
    matrix.grad = torch.randn_like(matrix)
    bias.grad = torch.randn_like(bias)

    optimizer.step()

    assert not torch.allclose(matrix, matrix_before)
    assert not torch.allclose(bias, bias_before)
    assert "momentum_buffer" in optimizer.state[matrix]
    assert "exp_avg" in optimizer.state[bias]


def test_muon_builder_splits_2d_parameters_and_preserves_groupwise_schedule() -> None:
    model = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.LayerNorm(4))
    optimizer = build_optimizer(_muon_cfg(), model)

    assert [group["name"] for group in optimizer.param_groups] == ["muon_matrix", "aux_adam"]
    assert optimizer.param_groups[0]["lr"] == 0.02
    assert optimizer.param_groups[1]["lr"] == 0.001

    scheduler = build_scheduler(_muon_cfg(), optimizer)
    assert optimizer.param_groups[0]["lr"] == 0.0
    assert optimizer.param_groups[1]["lr"] == 0.0

    scheduler.step()
    assert optimizer.param_groups[0]["lr"] == 0.01
    assert optimizer.param_groups[1]["lr"] == 0.0005

    scheduler.step()
    assert optimizer.param_groups[0]["lr"] == 0.02
    assert optimizer.param_groups[1]["lr"] == 0.001

    for _ in range(4):
        scheduler.step()
    assert optimizer.param_groups[0]["lr"] == 0.01
    assert optimizer.param_groups[1]["lr"] == 0.0005

    for _ in range(2):
        scheduler.step()
    assert optimizer.param_groups[0]["lr"] == 0.001
    assert optimizer.param_groups[1]["lr"] == 0.00005
