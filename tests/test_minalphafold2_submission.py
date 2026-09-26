from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch
import yaml

from nanofold.competition_policy import load_track_spec
from nanofold.data import collate_batch
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


def test_minalphafold2_full_profile_reports_trainable_parameters_without_track_cap() -> None:
    cfg = yaml.safe_load(Path("submissions/minalphafold2_full/config.yaml").read_text())
    track = load_track_spec("limited")

    n_params = count_parameters(submission.build_model(cfg))

    assert cfg["model"]["profile_path"] == "third_party/minAlphaFold2/configs/alphafold2.toml"
    assert n_params > 50_000_000
    assert track.max_params is None


def test_minalphafold2_budget_schedule_scales_af2_protocol_to_track_budget() -> None:
    cfg = yaml.safe_load(Path("submissions/minalphafold2/config.yaml").read_text())

    schedule = submission._af2_budget_schedule(cfg)

    assert schedule.max_steps == 30000
    assert schedule.finetune_start_step == 26087
    assert schedule.finetune_ramp_steps == 500
    assert schedule.warmup_steps == 334
    assert schedule.lr_decay_step == 16696
    assert not submission._use_finetune_loss({**cfg, "_runtime": {"step": 26086}})
    assert submission._use_finetune_loss({**cfg, "_runtime": {"step": 26087}})


def test_minalphafold2_finetune_ramp_weight_scales_linearly() -> None:
    cfg = yaml.safe_load(Path("submissions/minalphafold2/config.yaml").read_text())

    assert submission._finetune_ramp_weight({**cfg, "_runtime": {"step": 26086}}) == pytest.approx(0.0)
    assert submission._finetune_ramp_weight({**cfg, "_runtime": {"step": 26087}}) == pytest.approx(0.0)
    assert submission._finetune_ramp_weight({**cfg, "_runtime": {"step": 26337}}) == pytest.approx(0.5)
    assert submission._finetune_ramp_weight({**cfg, "_runtime": {"step": 26587}}) == pytest.approx(1.0)
    assert submission._finetune_ramp_weight({**cfg, "_runtime": {"step": 30000}}) == pytest.approx(1.0)


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

    cfg["_runtime"] = {"step": 26086}
    assert float(submission._alphafold_loss(model, features, {}, cfg).detach()) == pytest.approx(2.0)

    cfg["_runtime"] = {"step": 26337}
    assert float(submission._alphafold_loss(model, features, {}, cfg).detach()) == pytest.approx(4.0)

    cfg["_runtime"] = {"step": 26587}
    assert float(submission._alphafold_loss(model, features, {}, cfg).detach()) == pytest.approx(6.0)


def test_minalphafold2_scheduler_uses_af2_lr_stages() -> None:
    cfg = yaml.safe_load(Path("submissions/minalphafold2/config.yaml").read_text())
    param = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = submission.build_optimizer(cfg, torch.nn.ParameterList([param]))
    scheduler = submission.build_scheduler(cfg, optimizer)

    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.0)

    scheduler.load_state_dict({"completed_steps": 334})
    assert optimizer.param_groups[0]["lr"] == pytest.approx(1.0e-3)

    scheduler.load_state_dict({"completed_steps": 16696})
    assert optimizer.param_groups[0]["lr"] == pytest.approx(9.5e-4)

    scheduler.load_state_dict({"completed_steps": 26087})
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


@pytest.mark.parametrize("training", [False, True])
def test_minalphafold2_msa_features_ignore_companion_padding(training: bool) -> None:
    def example(length: int, depth: int) -> dict[str, Any]:
        msa = torch.zeros((depth, length), dtype=torch.long)
        msa[-1] = 21  # Biological gaps must survive padding removal.
        return {
            "chain_id": "TEST_A",
            "aatype": torch.zeros(length, dtype=torch.long),
            "msa": msa,
            "deletions": torch.ones_like(msa),
            "template_aatype": torch.zeros((0, length), dtype=torch.long),
            "template_ca_coords": torch.zeros((0, length, 3)),
            "template_ca_mask": torch.zeros((0, length), dtype=torch.bool),
        }

    cfg = {"model": {"msa_depth": 2, "extra_msa_depth": 4}}
    short = example(3, 2)
    alone = collate_batch([short], 5, 5)
    together = collate_batch([short, example(5, 5)], 5, 5)
    torch.manual_seed(41)
    expected = submission._build_minalphafold_inputs(alone, cfg, training=training)
    torch.manual_seed(41)
    actual = submission._build_minalphafold_inputs(together, cfg, training=training)
    for key in ("msa_feat", "msa_mask", "masked_msa_target", "masked_msa_mask"):
        torch.testing.assert_close(actual[key][0, :2, :3], expected[key][0])
        assert torch.count_nonzero(actual[key][0, :, 3:]) == 0
    assert bool(actual["msa_mask"][0, :2, :3].all())
    assert torch.count_nonzero(actual["extra_msa_mask"][0]) == 0


def test_minalphafold2_run_batch_returns_af2_loss() -> None:
    cfg = yaml.safe_load(Path("submissions/minalphafold2/config.yaml").read_text())
    model = submission.build_model(cfg)
    batch = _synthetic_supervised_batch()

    out = submission.run_batch(model, batch, cfg, training=True)

    assert out["pred_atom14"].shape == (1, 8, 14, 3)
    assert out["loss"].requires_grad
    assert torch.isfinite(out["loss"])
    out["loss"].backward()
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    assert gradients and all(torch.isfinite(grad).all() for grad in gradients)
    assert any(torch.count_nonzero(grad) for grad in gradients)


def test_minalphafold2_run_batch_returns_finetune_loss_at_handoff() -> None:
    cfg = yaml.safe_load(Path("submissions/minalphafold2/config.yaml").read_text())
    cfg["_runtime"] = {"step": 26087}
    model = submission.build_model(cfg)
    batch = _synthetic_supervised_batch()

    out = submission.run_batch(model, batch, cfg, training=True)

    assert out["pred_atom14"].shape == (1, 8, 14, 3)
    assert out["loss"].requires_grad
    assert torch.isfinite(out["loss"])
    loss_fn = model.nanofold_finetune_loss_fn
    for attr in ("structural_violation_weight", "experimentally_resolved_weight", "tm_score_weight"):
        assert float(getattr(loss_fn, attr)) > 0.0


@pytest.mark.parametrize("state", [{}, {"completed_steps": True}, {"completed_steps": 1.5},
                                    {"completed_steps": -1}, {"completed_steps": 30001},
                                    {"completed_steps": 1, "ignored": 2}])
def test_minalphafold2_scheduler_rejects_invalid_resume_without_changing_lr(state: dict[str, Any]) -> None:
    cfg = yaml.safe_load(Path("submissions/minalphafold2/config.yaml").read_text())
    optimizer = submission.build_optimizer(cfg, torch.nn.Linear(1, 1))
    scheduler = submission.build_scheduler(cfg, optimizer)
    scheduler.load_state_dict({"completed_steps": 500})
    with pytest.raises(ValueError):
        scheduler.load_state_dict(state)
    assert scheduler.state_dict() == {"completed_steps": 500}
    assert optimizer.param_groups[0]["lr"] == pytest.approx(1.0e-3)


def test_minalphafold2_explicit_fixed_recycling_and_inference_mode() -> None:
    cfg = yaml.safe_load(Path("submissions/minalphafold2/config.yaml").read_text())
    cfg["model"]["n_cycles"] = 4
    cfg["experiment"] = {"recycle": {"train_mode": "fixed", "max_train_cycles": 4}}
    model = submission.build_model(cfg)
    batch = _synthetic_supervised_batch(length=4)
    trained = submission.run_batch(model, batch, cfg, training=True)
    assert trained["recycle_cycles"].item() == 4
    trained["loss"].backward()
    recycle_gradient = dict(model.named_parameters())["recycle_norm_s.weight"].grad
    assert recycle_gradient is not None and torch.isfinite(recycle_gradient).all()

    cfg["experiment"]["recycle"]["train_mode"] = "sample_uniform"
    with torch.no_grad():
        evaluated = submission.run_batch(model, batch, cfg, training=False)
        again = submission.run_batch(model, batch, cfg, training=False)
    assert not model.training
    assert evaluated["recycle_cycles"].item() == 4
    torch.testing.assert_close(evaluated["pred_atom14"], again["pred_atom14"], rtol=0, atol=0)


def test_minalphafold2_fape_policy_controls_real_geometric_loss() -> None:
    cfg = yaml.safe_load(Path("submissions/minalphafold2/config.yaml").read_text())
    cfg["experiment"] = {"fape": {"backbone_clamp_mode": "fully_clamped"}}
    model = submission.build_model(cfg)
    batch = _synthetic_supervised_batch(length=8)
    # A large geometric error must remain visible under the unclamped objective.
    batch["atom14_positions"] *= 20.0
    with torch.no_grad():
        clamped = submission.run_batch(model, batch, cfg, training=False)["loss"]
        cfg["experiment"]["fape"]["backbone_clamp_mode"] = "fully_unclamped"
        unclamped = submission.run_batch(model, batch, cfg, training=False)["loss"]
        cfg["experiment"]["fape"] = {"backbone_clamp_mode": "batchwise", "clamped_probability": 0.9}
        mixture = submission.run_batch(model, batch, cfg, training=False)["loss"]
    assert unclamped > clamped
    torch.testing.assert_close(mixture, 0.9 * clamped + 0.1 * unclamped)


def test_minalphafold2_batchwise_fape_shares_step_decision_without_consuming_rng() -> None:
    cfg = {"seed": 2, "_runtime": {"step": 10}, "experiment": {"fape": {"backbone_clamp_mode": "batchwise"}}}
    features = {"aatype": torch.zeros((2, 3), dtype=torch.long)}
    state = torch.get_rng_state().clone()
    first = submission._fape_clamp_weight(cfg, features, training=True)
    second = submission._fape_clamp_weight(cfg, features, training=True)
    assert first is not None and second is not None
    assert first.item() in (0.0, 1.0)
    torch.testing.assert_close(first, second)
    assert torch.equal(state, torch.get_rng_state())


def test_minalphafold2_initial_only_does_not_activate_at_final_checkpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = yaml.safe_load(Path("submissions/minalphafold2/config.yaml").read_text())
    cfg["experiment"] = {"loss_schedule": {"mode": "initial_only"}}
    cfg["_runtime"] = {"step": 30000}
    model = submission.build_model(cfg)
    def unexpected_finetune(*args: Any, **kwargs: Any) -> torch.Tensor:
        pytest.fail("Initial-only recipes must never use the fine-tuning loss.")
    monkeypatch.setattr(model.nanofold_finetune_loss_fn, "forward", unexpected_finetune)
    out = submission.run_batch(model, _synthetic_supervised_batch(), cfg, training=False)
    assert torch.isfinite(out["loss"])


def test_minalphafold2_supervision_masks_padding_and_chain_breaks() -> None:
    cfg = yaml.safe_load(Path("submissions/minalphafold2/config.yaml").read_text())
    batch = _synthetic_supervised_batch(length=6)
    batch["residue_mask"][:, 5:] = False
    batch["residue_index"][:, 2:] += 10
    batch["between_segment_residues"][:, 4] = 1
    features = submission._build_minalphafold_inputs(batch, cfg, training=False)
    assert torch.count_nonzero(features["atom37_exists"][:, 5:]) == 0
    assert features["true_torsion_mask"][0, 1, :2].all()
    assert torch.count_nonzero(features["true_torsion_mask"][0, [2, 4], :2]) == 0
    assert features["loss_residue_index"][0, 4] - features["loss_residue_index"][0, 3] == 2
    # Model relative-position inputs retain the deposited residue numbering.
    torch.testing.assert_close(features["residue_index"], batch["residue_index"])


def test_minalphafold2_finetune_loss_ignores_companion_padding() -> None:
    def example(length: int) -> dict[str, Any]:
        batch = _synthetic_supervised_batch(length=length)
        item = {key: value[0] for key, value in batch.items()}
        item.update(
            template_aatype=torch.zeros((0, length), dtype=torch.long),
            template_ca_coords=torch.zeros((0, length, 3)),
            template_ca_mask=torch.zeros((0, length), dtype=torch.bool),
        )
        return item

    cfg = yaml.safe_load(Path("submissions/minalphafold2/config.yaml").read_text())
    cfg["_runtime"] = {"step": 30000}
    model = submission.build_model(cfg)
    short, long = example(4), example(7)
    with torch.no_grad():
        alone = [submission.run_batch(model, collate_batch([item], 8, 4), cfg, training=False)
                 for item in (short, long)]
        together = submission.run_batch(model, collate_batch([short, long], 8, 4), cfg, training=False)
    assert torch.count_nonzero(together["pred_atom14"][0, 4:]) == 0
    torch.testing.assert_close(together["pred_atom14"][0, :4], alone[0]["pred_atom14"][0], rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(together["loss"], (alone[0]["loss"] + alone[1]["loss"]) / 2, rtol=1e-5, atol=1e-5)


def test_minalphafold2_reports_real_extra_msa_capacity() -> None:
    cfg = yaml.safe_load(Path("submissions/minalphafold2/config.yaml").read_text())
    cfg["model"].update(msa_depth=4, extra_msa_depth=2)
    model = submission.build_model(cfg)
    with torch.no_grad():
        shallow = submission.run_batch(model, _synthetic_supervised_batch(length=4, msa_depth=4), cfg, training=False)
        deep = submission.run_batch(model, _synthetic_supervised_batch(length=4, msa_depth=8), cfg, training=False)
    assert shallow["msa_cluster_rows"].item() == 4
    assert shallow["msa_extra_rows"].item() == 0
    assert deep["msa_cluster_rows"].item() == 4
    assert deep["msa_extra_rows"].item() == 2


@pytest.mark.parametrize("section,value", [
    ("fape", {"backbone_clamp_mode": "unclamp_typo"}),
    ("fape", {"backbone_clamp_mdoe": "fully_unclamped"}),
    ("fape", {"clamped_probability": float("nan")}),
    ("fape", {"clamped_probability": 1.1}),
    ("recycle", {"train_mode": "fixed_typo"}),
    ("recycle", {"max_train_cycles": 4}),
    ("loss_schedule", {"mode": "initial_typo"}),
])
def test_minalphafold2_rejects_invalid_recipe_controls(section: str, value: dict[str, Any]) -> None:
    cfg = yaml.safe_load(Path("submissions/minalphafold2/config.yaml").read_text())
    cfg["experiment"] = {section: value}
    with pytest.raises(ValueError):
        submission.validate_config(cfg)
