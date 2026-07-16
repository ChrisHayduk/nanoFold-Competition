from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pytest
import torch
import yaml

from nanofold.utils import load_torch_checkpoint
from submissions.compact_pairformer_ipa import submission


def _tiny_cfg() -> dict[str, Any]:
    cfg = yaml.safe_load(Path("submissions/compact_pairformer_ipa/config.yaml").read_text())
    cfg["data"]["crop_size"] = 16
    cfg["data"]["msa_depth"] = 4
    cfg["model"].update(
        {
            "msa_depth": 4,
            "c_m": 32,
            "c_s": 32,
            "c_z": 16,
            "msa_blocks": 1,
            "msa_heads": 4,
            "msa_head_dim": 8,
            "outer_product_dim": 8,
            "pairformer_blocks": 1,
            "triangle_mult_c": 16,
            "triangle_heads": 2,
            "triangle_head_dim": 8,
            "single_attention_heads": 4,
            "structure_c": 16,
            "structure_layers": 2,
            "ipa_heads": 4,
            "ipa_head_dim": 8,
            "bf16": False,
        }
    )
    cfg["train"].update({"max_steps": 12, "warmup_steps": 2, "lr_decay_step": 8})
    return cfg


def _batch(length: int = 8, msa_depth: int = 4, *, supervised: bool = True) -> dict[str, Any]:
    residue_index = torch.arange(length, dtype=torch.float32)
    aatype = torch.full((1, length), 7, dtype=torch.long)
    batch: dict[str, Any] = {
        "chain_id": ["TEST_A"],
        "aatype": aatype,
        "msa": aatype.unsqueeze(1).expand(-1, msa_depth, -1).contiguous(),
        "deletions": torch.zeros((1, msa_depth, length), dtype=torch.long),
        "residue_index": torch.arange(length, dtype=torch.long).expand(1, -1),
        "between_segment_residues": torch.zeros((1, length), dtype=torch.long),
        "residue_mask": torch.ones((1, length), dtype=torch.bool),
    }
    if not supervised:
        return batch

    atom14_positions = torch.zeros((1, length, 14, 3), dtype=torch.float32)
    atom14_positions[0, :, 0, 0] = residue_index * 3.8
    atom14_positions[0, :, 1, 0] = residue_index * 3.8 + 1.45
    atom14_positions[0, :, 2, 0] = residue_index * 3.8 + 2.55
    atom14_positions[0, :, 3, 0] = residue_index * 3.8 + 3.2
    atom14_mask = torch.zeros((1, length, 14), dtype=torch.bool)
    atom14_mask[:, :, :4] = True
    batch.update(
        {
            "ca_coords": atom14_positions[:, :, 1, :],
            "ca_mask": atom14_mask[:, :, 1],
            "atom14_positions": atom14_positions,
            "atom14_mask": atom14_mask,
            "resolution": torch.tensor([2.0], dtype=torch.float32),
        }
    )
    return batch


def test_compact_pairformer_train_forward_backward_is_finite() -> None:
    cfg = _tiny_cfg()
    model = submission.build_model(cfg)

    out = submission.run_batch(model, _batch(), cfg, training=True)
    out["loss"].backward()

    assert out["pred_atom14"].shape == (1, 8, 14, 3)
    assert torch.isfinite(out["pred_atom14"]).all()
    assert torch.isfinite(out["loss"])
    assert any(param.grad is not None for param in model.parameters())


def test_compact_pairformer_inference_needs_no_supervision() -> None:
    cfg = _tiny_cfg()
    model = submission.build_model(cfg).eval()

    with torch.no_grad():
        out = submission.run_batch(model, _batch(supervised=False), cfg, training=False)

    assert set(out) == {"pred_atom14"}
    assert out["pred_atom14"].shape == (1, 8, 14, 3)
    assert torch.isfinite(out["pred_atom14"]).all()


def test_pairformer_residual_outputs_start_at_zero() -> None:
    model = cast(submission.CompactPairformerIPA, submission.build_model(_tiny_cfg()))
    block = cast(submission.CompactPairformerBlock, model.pairformer_blocks[0])

    assert torch.count_nonzero(block.single_attention.linear_output.weight) == 0
    assert torch.count_nonzero(block.single_transition.linear_down.weight) == 0
    assert torch.count_nonzero(block.triangle_mult_out.out_linear.weight) == 0


def test_geometry_loss_ramp_targets_the_high_weight_checkpoint() -> None:
    cfg = _tiny_cfg()
    cfg["train"].update(
        {
            "max_steps": 30000,
            "finetune_start_step": 1000,
            "finetune_ramp_steps": 4000,
        }
    )

    assert submission._finetune_ramp_weight({**cfg, "_runtime": {"step": 0}}) == pytest.approx(0.0)
    assert submission._finetune_ramp_weight({**cfg, "_runtime": {"step": 1000}}) == pytest.approx(0.0)
    assert submission._finetune_ramp_weight({**cfg, "_runtime": {"step": 3000}}) == pytest.approx(0.5)
    assert submission._finetune_ramp_weight({**cfg, "_runtime": {"step": 5000}}) == pytest.approx(1.0)


def test_muon_split_is_complete_disjoint_and_updates_both_groups() -> None:
    cfg = _tiny_cfg()
    model = submission.build_model(cfg)
    muon_params, auxiliary_params = submission._muon_parameter_split(model)
    trainable = {id(param) for param in model.parameters() if param.requires_grad}

    assert muon_params
    assert auxiliary_params
    assert {id(param) for param in muon_params}.isdisjoint({id(param) for param in auxiliary_params})
    assert {id(param) for param in [*muon_params, *auxiliary_params]} == trainable

    optimizer = submission.build_optimizer(cfg, model)
    matrix = muon_params[0]
    auxiliary = auxiliary_params[0]
    matrix_before = matrix.detach().clone()
    auxiliary_before = auxiliary.detach().clone()
    matrix.grad = torch.randn_like(matrix)
    auxiliary.grad = torch.randn_like(auxiliary)
    optimizer.step()

    assert not torch.allclose(matrix, matrix_before)
    assert not torch.allclose(auxiliary, auxiliary_before)
    assert "momentum_buffer" in optimizer.state[matrix]
    assert "exp_avg" in optimizer.state[auxiliary]


def test_groupwise_scheduler_preserves_muon_to_adam_lr_ratio() -> None:
    cfg = _tiny_cfg()
    cfg["train"].update(
        {
            "max_steps": 12,
            "warmup_steps": 2,
            "lr_decay_step": 8,
            "finetune_start_step": 4,
            "finetune_lr_scale": 1.0,
        }
    )
    optimizer = submission.build_optimizer(cfg, submission.build_model(cfg))
    scheduler = submission.build_scheduler(cfg, optimizer)

    assert [group["lr"] for group in optimizer.param_groups] == [0.0, 0.0]
    scheduler.step()
    assert [group["lr"] for group in optimizer.param_groups] == pytest.approx([0.005, 0.0005])
    scheduler.step()
    assert [group["lr"] for group in optimizer.param_groups] == pytest.approx([0.01, 0.001])
    for _ in range(6):
        scheduler.step()
    assert [group["lr"] for group in optimizer.param_groups] == pytest.approx([0.005, 0.0005])


def test_optimizer_and_scheduler_checkpoint_round_trip(tmp_path: Path) -> None:
    cfg = _tiny_cfg()
    model = submission.build_model(cfg)
    optimizer = submission.build_optimizer(cfg, model)
    scheduler = submission.build_scheduler(cfg, optimizer)

    out = submission.run_batch(model, _batch(), cfg, training=True)
    out["loss"].backward()
    optimizer.step()
    scheduler.step()

    checkpoint_path = tmp_path / "checkpoint.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "opt": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
        },
        checkpoint_path,
    )
    checkpoint = load_torch_checkpoint(checkpoint_path, map_location="cpu")

    restored_model = submission.build_model(cfg)
    restored_optimizer = submission.build_optimizer(cfg, restored_model)
    restored_scheduler = submission.build_scheduler(cfg, restored_optimizer)
    restored_model.load_state_dict(checkpoint["model"], strict=True)
    restored_optimizer.load_state_dict(checkpoint["opt"])
    restored_scheduler.load_state_dict(checkpoint["scheduler"])

    assert restored_scheduler.state_dict() == scheduler.state_dict()
    assert [group["lr"] for group in restored_optimizer.param_groups] == pytest.approx(
        [group["lr"] for group in optimizer.param_groups]
    )
    assert sum(bool(state) for state in restored_optimizer.state.values()) == sum(
        bool(state) for state in optimizer.state.values()
    )
