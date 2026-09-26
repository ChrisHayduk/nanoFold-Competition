from __future__ import annotations

import json
import random
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import torch

import train
from nanofold.chain_paths import chain_npz_path
from nanofold.submission_runtime import INFERENCE_BATCH_KEYS, SubmissionHooks


class TinyStochasticModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = torch.nn.Linear(3, 3)
        self.dropout = torch.nn.Dropout(0.25)

    def forward(self, batch):
        features = torch.stack(
            [batch["aatype"].float() / 20, batch["msa"].float().mean(dim=1) / 20,
             batch["residue_index"].float() / 20],
            dim=-1,
        )
        return self.projection(self.dropout(features))


def _write_proteins(root):
    features, labels = root / "features", root / "labels"
    features.mkdir()
    labels.mkdir()
    manifests = {}
    for split, count in (("train", 22), ("val", 3)):
        chains = [f"{split}_{index}" for index in range(count)]
        manifest = root / f"{split}.txt"
        manifest.write_text("\n".join(chains) + "\n")
        manifests[split] = str(manifest)
        for index, chain in enumerate(chains):
            length, depth = 5 + index % 9, 2 + index % 7
            positions = np.arange(length, dtype=np.int64)
            aatype = (positions + index) % 20
            msa = (aatype[None, :] + np.arange(depth, dtype=np.int64)[:, None]) % 20
            coordinates = np.stack(
                [positions * 0.3, np.sin(positions), np.cos(positions)], axis=-1,
            ).astype(np.float32)
            atom14 = np.repeat(coordinates[:, None, :], 14, axis=1)
            np.savez(
                chain_npz_path(features, chain), chain_id=chain, aatype=aatype, msa=msa,
                deletions=np.zeros_like(msa), residue_index=positions,
            )
            np.savez(
                chain_npz_path(labels, chain), chain_id=chain, ca_coords=coordinates,
                ca_mask=np.ones(length, dtype=bool), atom14_positions=atom14,
                atom14_mask=np.ones((length, 14), dtype=bool), residue_index=positions,
                resolution=np.float32(2.0),
            )
    return {
        "processed_features_dir": str(features), "processed_labels_dir": str(labels),
        "train_manifest": manifests["train"], "val_manifest": manifests["val"],
        "crop_size": 8, "msa_depth": 4, "batch_size": 2, "num_workers": 0,
        "train_crop_mode": "random", "train_msa_sample_mode": "random",
        "val_crop_mode": "center", "val_msa_sample_mode": "top",
    }


@pytest.fixture
def loop_harness(monkeypatch, tmp_path):
    data = _write_proteins(tmp_path)
    observation = {"clock": 0.0, "train_calls": [], "eval_calls": 0}

    def run_batch(model, batch, cfg, training):
        if training:
            assert "atom14_positions" in batch
            observation["clock"] = float(cfg["_runtime"]["step"] + 1)
            observation["train_calls"].append(tuple(batch["chain_id"]))
        else:
            # Exercise the real submission-runtime boundary, including label metadata.
            assert set(batch) <= INFERENCE_BATCH_KEYS
            observation["eval_calls"] += 1
        # Evaluation deliberately consumes all RNGs used by training submissions.
        stochastic_scale = 0.5 + random.random() + float(np.random.random()) + float(torch.rand(()))
        ca = model(batch) * stochastic_scale
        atom14 = ca.unsqueeze(2).expand(-1, -1, 14, -1).contiguous()
        output = {"pred_atom14": atom14}
        if training:
            mask = batch["residue_mask"][:, :, None, None]
            output["loss"] = ((atom14 - batch["atom14_positions"]).square() * mask).mean()
        return output

    hooks = SubmissionHooks(
        module_ref="test:tiny-stochastic", module=ModuleType("tiny_stochastic"),
        source_path=__file__, source_sha256=train._sha256_file(__file__),
        build_model=lambda cfg: TinyStochasticModel(),
        build_optimizer=lambda cfg, model: torch.optim.Adam(model.parameters(), lr=0.01),
        run_batch=run_batch,
        build_scheduler=lambda cfg, optimizer: torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5),
    )
    monkeypatch.setattr(train, "load_submission_hooks", lambda *args, **kwargs: hooks)
    monkeypatch.setattr(train, "default_torch_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(train, "get_env_metadata", lambda device: {"device_type": "cpu", "torch": torch.__version__})
    monkeypatch.setattr(train, "time", SimpleNamespace(perf_counter=lambda: observation["clock"]))
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    monkeypatch.delenv("RANK", raising=False)
    monkeypatch.delenv("LOCAL_RANK", raising=False)

    def config(name, *, num_workers=0, eval_every=2):
        run_dir = tmp_path / name
        cfg = {
            "run_name": str(run_dir), "seed": 37,
            "data": {**data, "num_workers": num_workers}, "optim": {"grad_clip_norm": 0.5},
            "train": {"max_steps": 7, "grad_accum_steps": 2, "log_every": 20,
                      "eval_every": eval_every, "save_every": 4, "amp": False},
        }
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(cfg))
        return path, run_dir

    def execute(config_path, *, resume=None, deadline=None):
        observation.update(clock=0.0, train_calls=[], eval_calls=0)
        argv = ["train.py", "--config", str(config_path), "--track", "unlimited", "--deterministic"]
        if resume is not None:
            argv.extend(["--resume", str(resume)])
        if deadline is not None:
            argv.extend(["--max-runtime-seconds", str(deadline)])
        monkeypatch.setattr(sys, "argv", argv)
        train.main()

    yield SimpleNamespace(config=config, execute=execute, observation=observation)


def _assert_nested_equal(actual, expected):
    if isinstance(expected, torch.Tensor):
        assert torch.equal(actual, expected)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            _assert_nested_equal(actual[key], expected[key])
    elif isinstance(expected, (tuple, list)):
        assert type(actual) is type(expected)
        assert len(actual) == len(expected)
        for actual_item, expected_item in zip(actual, expected, strict=True):
            _assert_nested_equal(actual_item, expected_item)
    else:
        assert actual == expected


def _assert_training_state_equal(actual, expected, *, compare_sampling=True):
    fields = ["step", "model", "opt", "scheduler", "scaler", "cumulative_samples_seen",
              "cumulative_cropped_residues_seen", "cumulative_nonpad_residues_seen", "cumulative_residues_seen"]
    if compare_sampling:
        fields.append("sampling_state")
    for field in fields:
        _assert_nested_equal(actual[field], expected[field])


@pytest.mark.parametrize("num_workers", [0, 2])
def test_main_deadline_resume_matches_uninterrupted_training(loop_harness, num_workers):
    baseline_config, baseline_dir = loop_harness.config("baseline", num_workers=num_workers)
    loop_harness.execute(baseline_config)
    expected = train.load_torch_checkpoint(baseline_dir / "checkpoints/ckpt_last.pt")
    expected_calls = loop_harness.observation["train_calls"].copy()

    resume_config, resume_dir = loop_harness.config("resumed", num_workers=num_workers)
    with pytest.raises(SystemExit) as stopped:
        loop_harness.execute(resume_config, deadline=3)
    assert stopped.value.code == 75
    checkpoint_path = resume_dir / "checkpoints/ckpt_last.pt"
    paused = train.load_torch_checkpoint(checkpoint_path)
    assert paused["step"] == 3
    assert paused["cumulative_samples_seen"] == 12
    assert paused["sampling_state"]["rank_states"][0]["batches_consumed"] == 6
    # Stop is mid-epoch, after the worker prefetch window, and off save cadence.
    assert (resume_dir / "checkpoints/ckpt_step_3.pt").is_file()
    paused_metrics = json.loads((resume_dir / "metrics.json").read_text())
    assert paused_metrics["status"] == "paused"
    assert paused_metrics["stop_reason"] == "max_runtime_seconds"
    paused_calls = loop_harness.observation["train_calls"].copy()

    loop_harness.execute(resume_config, resume=checkpoint_path)
    actual = train.load_torch_checkpoint(checkpoint_path)
    _assert_training_state_equal(actual, expected)
    assert paused_calls + loop_harness.observation["train_calls"] == expected_calls
    assert actual["step"] == 7
    assert actual["cumulative_samples_seen"] == 28
    assert actual["cumulative_nonpad_residues_seen"] < 28 * 8
    assert actual["sampling_state"]["rank_states"][0]["epoch"] == 1
    assert (resume_dir / "checkpoints/ckpt_step_7.pt").is_file()
    final_metrics = json.loads((resume_dir / "metrics.json").read_text())
    assert final_metrics["status"] == "complete"
    assert final_metrics.get("stop_reason") != "max_runtime_seconds"


def test_validation_frequency_does_not_change_training_and_never_exposes_labels(loop_harness):
    frequent_config, frequent_dir = loop_harness.config("frequent_validation", eval_every=1)
    loop_harness.execute(frequent_config)
    frequent = train.load_torch_checkpoint(frequent_dir / "checkpoints/ckpt_last.pt")
    frequent_calls = loop_harness.observation["eval_calls"]
    sparse_config, sparse_dir = loop_harness.config("sparse_validation", eval_every=4)
    loop_harness.execute(sparse_config)
    sparse = train.load_torch_checkpoint(sparse_dir / "checkpoints/ckpt_last.pt")
    assert frequent_calls > loop_harness.observation["eval_calls"] > 0
    _assert_training_state_equal(sparse, frequent, compare_sampling=False)
    # The final update is saved even though step 7 is not a scheduled save step.
    assert (sparse_dir / "checkpoints/ckpt_step_7.pt").is_file()


def test_deadline_at_completed_budget_does_not_report_paused(loop_harness):
    config, run_dir = loop_harness.config("complete_at_deadline")
    loop_harness.execute(config, deadline=7)
    checkpoint = train.load_torch_checkpoint(run_dir / "checkpoints/ckpt_last.pt")
    assert checkpoint["step"] == 7
    assert json.loads((run_dir / "metrics.json").read_text())["status"] == "complete"
    saved_outputs = {path.relative_to(run_dir): path.read_bytes() for path in run_dir.rglob("*") if path.is_file()}
    with pytest.raises(FileExistsError, match="resume.*reset-run"):
        loop_harness.execute(config)
    assert loop_harness.observation["train_calls"] == []
    assert {path.relative_to(run_dir): path.read_bytes() for path in run_dir.rglob("*") if path.is_file()} == saved_outputs
    with pytest.raises(ValueError, match="newer checkpoints remain"):
        loop_harness.execute(config, resume=run_dir / "checkpoints/ckpt_step_4.pt")
    assert loop_harness.observation["train_calls"] == []
    assert {path.relative_to(run_dir): path.read_bytes() for path in run_dir.rglob("*") if path.is_file()} == saved_outputs


@pytest.mark.parametrize("directory", ["processed_features_dir", "processed_labels_dir"])
def test_missing_validation_input_fails_before_training(loop_harness, directory):
    config, run_dir = loop_harness.config("missing_validation")
    data = json.loads(config.read_text())["data"]
    chain_npz_path(data[directory], "val_0").unlink()
    with pytest.raises(FileNotFoundError, match="Missing.*manifest"):
        loop_harness.execute(config)
    assert loop_harness.observation["train_calls"] == []
    assert not list((run_dir / "checkpoints").glob("*.pt"))


def test_public_foldscore_receives_unpadded_sequences_and_segment_metadata(loop_harness, monkeypatch):
    config, run_dir = loop_harness.config("foldscore_boundaries")
    cfg = json.loads(config.read_text())
    cfg["train"].update(max_steps=1, eval_foldscore_components=True)
    config.write_text(json.dumps(cfg))
    lengths = []

    def score(**kwargs):
        length = len(kwargs["aatype"])
        lengths.append(length)
        for key in ("pred_atom14", "true_atom14", "atom14_mask", "residue_index", "between_segment_residues"):
            assert len(kwargs[key]) == length
        assert kwargs["atom14_mask"].any(dim=1).all()
        assert torch.equal(kwargs["residue_index"], torch.arange(length))
        return {name: torch.tensor(0.5) for name in train.FOLDSCORE_COMPONENT_NAMES}

    monkeypatch.setattr(train, "foldscore_components", score)
    loop_harness.execute(config)
    assert lengths == [5, 6, 7]
    metrics = json.loads((run_dir / "metrics.json").read_text())
    assert metrics["foldscore_version"] == train.FOLDSCORE_VERSION
    assert metrics["history"][-1]["val_foldscore_valid_count"] == 3
