from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch
import yaml

from nanofold.competition_policy import load_track_spec
from nanofold.utils import count_parameters
from submissions.minalphafold2 import submission


def _synthetic_supervised_batch(length: int = 8, msa_depth: int = 4) -> dict[str, Any]:
    residue_index = torch.arange(length, dtype=torch.float32)
    atom14_positions = torch.zeros((1, length, 14, 3), dtype=torch.float32)
    atom14_positions[0, :, 0, :] = torch.stack(
        (residue_index * 3.8, torch.zeros(length), torch.zeros(length)),
        dim=-1,
    )
    atom14_positions[0, :, 1, :] = torch.stack(
        (residue_index * 3.8 + 1.45, torch.full((length,), 0.5), torch.zeros(length)),
        dim=-1,
    )
    atom14_positions[0, :, 2, :] = torch.stack(
        (residue_index * 3.8 + 2.55, torch.zeros(length), torch.full((length,), 0.2)),
        dim=-1,
    )
    atom14_positions[0, :, 3, :] = torch.stack(
        (residue_index * 3.8 + 3.2, torch.full((length,), -0.6), torch.full((length,), 0.2)),
        dim=-1,
    )
    atom14_mask = torch.zeros((1, length, 14), dtype=torch.bool)
    atom14_mask[:, :, :4] = True
    aatype = torch.full((1, length), 7, dtype=torch.long)
    return {
        "chain_id": ["TEST_A"],
        "aatype": aatype,
        "msa": aatype.unsqueeze(1).expand(-1, msa_depth, -1).contiguous(),
        "deletions": torch.zeros((1, msa_depth, length), dtype=torch.long),
        "residue_index": torch.arange(length, dtype=torch.long).expand(1, -1),
        "between_segment_residues": torch.zeros((1, length), dtype=torch.long),
        "residue_mask": torch.ones((1, length), dtype=torch.bool),
        "ca_coords": atom14_positions[:, :, 1, :],
        "ca_mask": atom14_mask[:, :, 1],
        "atom14_positions": atom14_positions,
        "atom14_mask": atom14_mask,
        "resolution": torch.tensor([2.0], dtype=torch.float32),
    }


def test_minalphafold2_reference_loads_upstream_tiny_toml() -> None:
    cfg = yaml.safe_load(Path("submissions/minalphafold2/config.yaml").read_text())

    model = submission.build_model(cfg)
    expected = submission.load_model_config(Path("third_party/minAlphaFold2/configs/tiny.toml"))

    assert cfg["model"]["profile_path"] == "third_party/minAlphaFold2/configs/tiny.toml"
    assert "c_m" not in cfg["model"]
    assert model.config == expected


def test_minalphafold2_full_profile_fits_limited_param_cap() -> None:
    cfg = yaml.safe_load(Path("submissions/minalphafold2_full/config.yaml").read_text())
    track = load_track_spec("limited")

    n_params = count_parameters(submission.build_model(cfg))

    assert cfg["model"]["profile_path"] == "third_party/minAlphaFold2/configs/alphafold2.toml"
    assert n_params > 50_000_000
    assert track.max_params == 100_000_000
    assert n_params <= int(track.max_params)


def test_minalphafold2_budget_schedule_scales_af2_protocol_to_track_budget() -> None:
    cfg = yaml.safe_load(Path("submissions/minalphafold2/config.yaml").read_text())

    schedule = submission._af2_budget_schedule(cfg)

    assert schedule.max_steps == 10000
    assert schedule.finetune_start_step == 8696
    assert schedule.finetune_ramp_steps == 500
    assert schedule.warmup_steps == 111
    assert schedule.lr_decay_step == 5565
    assert not submission._use_finetune_loss({**cfg, "_runtime": {"step": 8695}})
    assert submission._use_finetune_loss({**cfg, "_runtime": {"step": 8696}})


def test_minalphafold2_finetune_ramp_weight_scales_linearly() -> None:
    cfg = yaml.safe_load(Path("submissions/minalphafold2/config.yaml").read_text())

    assert submission._finetune_ramp_weight({**cfg, "_runtime": {"step": 8695}}) == pytest.approx(0.0)
    assert submission._finetune_ramp_weight({**cfg, "_runtime": {"step": 8696}}) == pytest.approx(0.0)
    assert submission._finetune_ramp_weight({**cfg, "_runtime": {"step": 8946}}) == pytest.approx(0.5)
    assert submission._finetune_ramp_weight({**cfg, "_runtime": {"step": 9196}}) == pytest.approx(1.0)
    assert submission._finetune_ramp_weight({**cfg, "_runtime": {"step": 10000}}) == pytest.approx(1.0)


def test_minalphafold2_finetune_auxiliary_weights_follow_ramp() -> None:
    loss_fn = submission.AlphaFoldLoss(finetune=True)
    target_weights = {
        attr: float(getattr(loss_fn, attr))
        for attr in submission.FINETUNE_AUXILIARY_WEIGHT_ATTRS
    }

    submission._apply_finetune_ramp(loss_fn, 0.0)
    for attr in submission.FINETUNE_AUXILIARY_WEIGHT_ATTRS:
        assert float(getattr(loss_fn, attr)) == pytest.approx(0.0)

    submission._apply_finetune_ramp(loss_fn, 0.5)
    for attr in submission.FINETUNE_AUXILIARY_WEIGHT_ATTRS:
        assert float(getattr(loss_fn, attr)) == pytest.approx(target_weights[attr] * 0.5)

    submission._apply_finetune_ramp(loss_fn, 1.5)
    for attr in submission.FINETUNE_AUXILIARY_WEIGHT_ATTRS:
        assert float(getattr(loss_fn, attr)) == pytest.approx(target_weights[attr])


def test_minalphafold2_handoff_blends_initial_and_finetune_losses(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = yaml.safe_load(Path("submissions/minalphafold2/config.yaml").read_text())

    class ConstantLoss(torch.nn.Module):
        def __init__(self, value: float) -> None:
            super().__init__()
            self.value = torch.nn.Parameter(torch.tensor(value))

        def forward(self, **_: Any) -> torch.Tensor:
            return self.value.reshape(1)

    model = torch.nn.Module()
    model.nanofold_initial_loss_fn = ConstantLoss(2.0)
    model.nanofold_finetune_loss_fn = ConstantLoss(6.0)
    features = {"aatype": torch.zeros((1, 1), dtype=torch.long)}
    monkeypatch.setattr(submission, "loss_inputs_from_batch", lambda features, model_out: {})

    cfg["_runtime"] = {"step": 8695}
    assert float(submission._alphafold_loss(model, features, {}, cfg).detach()) == pytest.approx(2.0)

    cfg["_runtime"] = {"step": 8946}
    assert float(submission._alphafold_loss(model, features, {}, cfg).detach()) == pytest.approx(4.0)

    cfg["_runtime"] = {"step": 9196}
    assert float(submission._alphafold_loss(model, features, {}, cfg).detach()) == pytest.approx(6.0)


def test_minalphafold2_scheduler_uses_af2_lr_stages() -> None:
    cfg = yaml.safe_load(Path("submissions/minalphafold2/config.yaml").read_text())
    param = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = submission.build_optimizer(cfg, torch.nn.ParameterList([param]))
    scheduler = submission.build_scheduler(cfg, optimizer)

    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.0)

    scheduler.load_state_dict({"completed_steps": 111})
    assert optimizer.param_groups[0]["lr"] == pytest.approx(1.0e-3)

    scheduler.load_state_dict({"completed_steps": 5565})
    assert optimizer.param_groups[0]["lr"] == pytest.approx(9.5e-4)

    scheduler.load_state_dict({"completed_steps": 8696})
    assert optimizer.param_groups[0]["lr"] == pytest.approx(4.75e-4)


def test_minalphafold2_inputs_include_af2_supervision_and_masked_msa_targets() -> None:
    cfg = yaml.safe_load(Path("submissions/minalphafold2/config.yaml").read_text())
    batch = _synthetic_supervised_batch()

    features = submission._build_minalphafold_inputs(batch, cfg, training=True)

    expected_keys = {
        "masked_msa_target",
        "masked_msa_mask",
        "true_rotations",
        "true_translations",
        "true_atom_positions",
        "true_atom_mask",
        "true_torsion_angles",
        "true_rigid_group_frames_R",
        "true_rigid_group_exists",
        "resolution",
    }
    assert expected_keys <= set(features)
    assert features["true_atom_positions"].shape == (1, 8, 14, 3)
    assert features["masked_msa_target"].shape[:2] == features["msa_feat"].shape[:2]


def test_minalphafold2_run_batch_returns_af2_loss() -> None:
    cfg = yaml.safe_load(Path("submissions/minalphafold2/config.yaml").read_text())
    model = submission.build_model(cfg)
    batch = _synthetic_supervised_batch()

    out = submission.run_batch(model, batch, cfg, training=True)

    assert out["pred_atom14"].shape == (1, 8, 14, 3)
    assert out["loss"].requires_grad
    assert torch.isfinite(out["loss"])


def test_minalphafold2_run_batch_returns_finetune_loss_at_handoff() -> None:
    cfg = yaml.safe_load(Path("submissions/minalphafold2/config.yaml").read_text())
    cfg["_runtime"] = {"step": 8696}
    model = submission.build_model(cfg)
    batch = _synthetic_supervised_batch()

    out = submission.run_batch(model, batch, cfg, training=True)

    assert out["pred_atom14"].shape == (1, 8, 14, 3)
    assert out["loss"].requires_grad
    assert torch.isfinite(out["loss"])
    loss_fn = model.nanofold_finetune_loss_fn
    for attr in submission.FINETUNE_AUXILIARY_WEIGHT_ATTRS:
        assert float(getattr(loss_fn, attr)) > 0.0
