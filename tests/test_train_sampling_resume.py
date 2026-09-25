from __future__ import annotations

import json
import random
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, Dataset, DistributedSampler

import train


class SamplingDataset(Dataset):
    """Small, distinguishable protein/MSA examples using the production collator."""

    def __init__(self, *, manifest_path, **kwargs):
        self.chain_ids = Path(manifest_path).read_text().splitlines()

    def __len__(self):
        return len(self.chain_ids)

    def __getitem__(self, index):
        length, depth = 17, 9
        positions = torch.arange(length)
        rows = torch.arange(depth)[:, None]
        msa = (rows + positions[None, :]) % 20
        return {
            "chain_id": self.chain_ids[index],
            "aatype": positions,
            "msa": msa,
            "deletions": rows * 100 + positions[None, :],
            "template_aatype": positions[None, :],
            "template_ca_coords": torch.zeros((1, length, 3)),
            "template_ca_mask": torch.zeros((1, length), dtype=torch.bool),
        }


def _config(tmp_path, *, num_workers=0, weighted=False):
    chain_ids = [f"chain_{index}" for index in range(8)]
    manifest = tmp_path / "manifest.txt"
    manifest.write_text("\n".join(chain_ids) + "\n")
    weights = tmp_path / "weights.json"
    weights.write_text(json.dumps({chain: index + 1 for index, chain in enumerate(chain_ids)}))
    return {"data": {
        "processed_features_dir": "unused",
        "train_manifest": str(manifest),
        "val_manifest": str(manifest),
        "crop_size": 6,
        "msa_depth": 4,
        "batch_size": 2,
        "num_workers": num_workers,
        "train_sample_weights": str(weights) if weighted else "",
        "allow_train_sample_weights": weighted,
    }}


def _loader(config, split="train", *, rank=0, world_size=1):
    return train.make_loader(
        config, split, device=torch.device("cpu"), include_labels=False,
        fail_if_labels_present=False, allow_missing=False,
        generator_seed=73 if split == "train" else 74,
        distributed_rank=rank, distributed_world_size=world_size,
    )


@dataclass
class TrainingStream:
    loader: DataLoader
    iterator: Any
    epoch_start: torch.Tensor
    epoch: int = 0
    batches_consumed: int = 0

    def advance(self):
        try:
            batch = next(self.iterator)
        except StopIteration:
            self.epoch += 1
            self.batches_consumed = 0
            self.iterator, self.epoch_start = train._start_train_epoch(self.loader, self.epoch)
            batch = next(self.iterator)
        self.batches_consumed += 1
        # Model-side randomness must continue independently of worker prefetch and
        # replay. Dropout, recycling and user submissions can use all three RNGs.
        model_draws = (random.random(), float(np.random.random()), torch.rand(5).tolist())
        return {
            "chains": batch["chain_id"],
            "crop": batch["aatype"].tolist(),
            "msa": batch["msa"].tolist(),
            "deletions": batch["deletions"].tolist(),
            "model_draws": model_draws,
        }

    def close(self):
        shutdown = getattr(self.iterator, "_shutdown_workers", None)
        if shutdown is not None:
            shutdown()


def _checkpoint(stream, val_loader, *, step, grad_accum_steps=2):
    contract = train._training_sampling_contract(
        stream.loader, world_size=1, local_grad_accum_steps=grad_accum_steps,
    )
    state = train._capture_training_rank_state(
        train_loader=stream.loader, val_loader=val_loader,
        epoch=stream.epoch, batches_consumed=stream.batches_consumed,
        epoch_start_generator_state=stream.epoch_start,
        rank=0, device=torch.device("cpu"),
    )
    return {
        "step": step,
        "sampling_state": train._collect_training_sampling_state(state, contract, distributed=False),
    }, contract


@pytest.mark.parametrize("num_workers,weighted", [(0, False), (0, True), (2, False), (2, True)])
def test_resume_continues_samples_crops_msa_and_model_rng_across_epochs(
    monkeypatch, tmp_path, num_workers, weighted,
) -> None:
    monkeypatch.setattr(train, "ProcessedNPZDataset", SamplingDataset)
    config = _config(tmp_path, num_workers=num_workers, weighted=weighted)
    train.set_seed(37)
    loader, val_loader = _loader(config), _loader(config, "val")
    iterator, epoch_start = train._start_train_epoch(loader, 0)
    uninterrupted = TrainingStream(loader, iterator, epoch_start)
    resumed = None
    try:
        # Three optimizer steps with accumulation=2 cross an epoch boundary.
        for _ in range(6):
            uninterrupted.advance()
        assert (uninterrupted.epoch, uninterrupted.batches_consumed) == (1, 2)
        torch.rand(3, generator=val_loader.generator)
        checkpoint, contract = _checkpoint(uninterrupted, val_loader, step=3)
        checkpoint_path = tmp_path / "checkpoint.pt"
        torch.save(checkpoint, checkpoint_path)
        checkpoint = train.load_torch_checkpoint(checkpoint_path)
        expected = [uninterrupted.advance() for _ in range(4)]
        expected_val_draw = torch.rand(3, generator=val_loader.generator)
        uninterrupted.close()

        # A fresh process starts with arbitrary global RNGs and loader state.
        train.set_seed(999)
        resumed_loader, resumed_val_loader = _loader(config), _loader(config, "val")
        resumed_contract = train._training_sampling_contract(
            resumed_loader, world_size=1, local_grad_accum_steps=2,
        )
        assert resumed_contract == contract
        state = train._resume_rank_sampling_state(checkpoint, resumed_contract, rank=0)
        resumed = TrainingStream(
            resumed_loader,
            train._restore_training_iterator(
                resumed_loader, resumed_val_loader, state, device=torch.device("cpu"),
            ),
            state["epoch_start_generator_state"],
            epoch=state["epoch"], batches_consumed=state["batches_consumed"],
        )
        assert [resumed.advance() for _ in range(4)] == expected
        assert (resumed.epoch, resumed.batches_consumed) == (2, 2)
        assert torch.equal(torch.rand(3, generator=resumed_val_loader.generator), expected_val_draw)
    finally:
        uninterrupted.close()
        if resumed is not None:
            resumed.close()


@pytest.fixture
def saved_sampling(monkeypatch, tmp_path):
    monkeypatch.setattr(train, "ProcessedNPZDataset", SamplingDataset)
    config = _config(tmp_path, weighted=True)
    loader, val_loader = _loader(config), _loader(config, "val")
    iterator, epoch_start = train._start_train_epoch(loader, 0)
    stream = TrainingStream(loader, iterator, epoch_start)
    for _ in range(2):
        stream.advance()
    checkpoint, contract = _checkpoint(stream, val_loader, step=1)
    return config, checkpoint, contract


def test_resume_binds_aligned_weight_values_instead_of_file_format(saved_sampling) -> None:
    config, checkpoint, contract = saved_sampling
    path = Path(config["data"]["train_sample_weights"])
    original = json.loads(path.read_text())
    path.write_text(json.dumps(dict(reversed(list(original.items()))), indent=4))
    reformatted_contract = train._training_sampling_contract(
        _loader(config), world_size=1, local_grad_accum_steps=2,
    )
    assert reformatted_contract == contract
    train._resume_rank_sampling_state(checkpoint, reformatted_contract, rank=0)

    original["chain_3"] *= 2
    path.write_text(json.dumps(original))
    changed_contract = train._training_sampling_contract(
        _loader(config), world_size=1, local_grad_accum_steps=2,
    )
    assert changed_contract != contract
    with pytest.raises(ValueError, match="sampling contract"):
        train._resume_rank_sampling_state(checkpoint, changed_contract, rank=0)


@pytest.mark.parametrize("setting,value", [
    ("world_size", 2), ("local_grad_accum_steps", 1), ("num_workers", 2),
    ("batch_size", 1), ("crop_size", 5), ("msa_depth", 3),
    ("train_crop_mode", "center"), ("train_msa_sample_mode", "top"),
])
def test_resume_rejects_changes_that_alter_sampling(saved_sampling, setting, value) -> None:
    config, checkpoint, _ = saved_sampling
    kwargs = {"world_size": 1, "local_grad_accum_steps": 2}
    if setting in kwargs:
        kwargs[setting] = value
    else:
        config["data"][setting] = value
    contract = train._training_sampling_contract(
        _loader(config), world_size=kwargs["world_size"], local_grad_accum_steps=kwargs["local_grad_accum_steps"],
    )
    with pytest.raises(ValueError, match="sampling contract"):
        train._resume_rank_sampling_state(checkpoint, contract, rank=0)


def test_resume_rejects_changed_manifest_order(saved_sampling) -> None:
    config, checkpoint, _ = saved_sampling
    path = Path(config["data"]["train_manifest"])
    path.write_text("\n".join(reversed(path.read_text().splitlines())) + "\n")
    contract = train._training_sampling_contract(_loader(config), world_size=1, local_grad_accum_steps=2)
    with pytest.raises(ValueError, match="sampling contract"):
        train._resume_rank_sampling_state(checkpoint, contract, rank=0)


def test_resume_requires_sampling_state(saved_sampling) -> None:
    _, _, contract = saved_sampling
    with pytest.raises(ValueError, match="sampling state"):
        train._resume_rank_sampling_state({"step": 1}, contract, rank=0)


@pytest.mark.parametrize("field,value", [
    ("epoch", -1), ("epoch", True), ("batches_consumed", 1), ("batches_consumed", 5),
    ("rank", 1), ("rng_state", None), ("loader_generator_state", torch.ones(4)),
])
def test_resume_rejects_corrupted_rank_cursor_or_rng(saved_sampling, field, value) -> None:
    _, checkpoint, contract = saved_sampling
    checkpoint = deepcopy(checkpoint)
    checkpoint["sampling_state"]["rank_states"][0][field] = value
    with pytest.raises(ValueError, match="sampling|rank|generator_state|RNG"):
        train._resume_rank_sampling_state(checkpoint, contract, rank=0)


@pytest.mark.parametrize("weighted", [False, True])
def test_distributed_workers_use_distinct_streams_but_one_shared_sampler_seed(
    monkeypatch, tmp_path, weighted,
) -> None:
    monkeypatch.setattr(train, "ProcessedNPZDataset", SamplingDataset)
    config = _config(tmp_path, num_workers=2, weighted=weighted)
    rank0, rank1 = [_loader(config, rank=rank, world_size=2) for rank in (0, 1)]
    assert isinstance(rank0.sampler, (DistributedSampler, train.WeightedDistributedSampler))
    assert isinstance(rank1.sampler, (DistributedSampler, train.WeightedDistributedSampler))
    assert rank0.sampler.seed == rank1.sampler.seed == 73
    assert not torch.equal(
        torch.rand(16, generator=rank0.generator),
        torch.rand(16, generator=rank1.generator),
    )


@pytest.mark.parametrize("field", [
    "epoch_start_generator_state", "loader_generator_state", "val_loader_generator_state", "rng_state",
])
def test_resume_rejects_corruption_on_another_rank_before_restoring_any_rng(saved_sampling, field) -> None:
    _, checkpoint, contract = saved_sampling
    checkpoint, contract = deepcopy(checkpoint), deepcopy(contract)
    contract["world_size"] = 2
    checkpoint["sampling_state"]["contract"] = contract
    second = deepcopy(checkpoint["sampling_state"]["rank_states"][0])
    second["rank"] = 1
    checkpoint["sampling_state"]["rank_states"].append(second)
    if field == "rng_state":
        second[field]["numpy"] = ("invalid", [], 0, 0, 0.0)
    else:
        # Shape and dtype look correct, but this is not a valid generator state.
        second[field] = torch.zeros(4, dtype=torch.uint8)
    before = train._capture_rng_state(torch.device("cpu"))
    with pytest.raises(ValueError, match="generator_state|RNG state"):
        train._resume_rank_sampling_state(checkpoint, contract, rank=0, device=torch.device("cpu"))
    after = train._capture_rng_state(torch.device("cpu"))
    assert before["python"] == after["python"]
    assert before["numpy"] == after["numpy"]
    assert torch.equal(before["torch"], after["torch"])


def test_invalid_model_rng_is_rejected_without_changing_any_rng(saved_sampling) -> None:
    _, checkpoint, _ = saved_sampling
    state = deepcopy(checkpoint["sampling_state"]["rank_states"][0]["rng_state"])
    state["torch"] = torch.zeros(4, dtype=torch.uint8)
    train.set_seed(993)
    before = train._capture_rng_state(torch.device("cpu"))
    with pytest.raises(ValueError, match="RNG state"):
        train._restore_rng_state(state, torch.device("cpu"))
    after = train._capture_rng_state(torch.device("cpu"))
    assert before["python"] == after["python"]
    assert before["numpy"] == after["numpy"]
    assert torch.equal(before["torch"], after["torch"])


@pytest.mark.parametrize("stage", ["serialize", "sync"])
def test_failed_checkpoint_write_preserves_previous_complete_checkpoint(monkeypatch, tmp_path, stage) -> None:
    path = tmp_path / "ckpt_last.pt"
    train._atomic_torch_save({"step": 3, "model": torch.arange(5)}, path)
    original_bytes = path.read_bytes()

    def interrupted_write(payload, handle):
        handle.write(b"partial checkpoint")
        raise OSError("disk full")

    def failed_sync(fd):
        raise OSError("sync failed")

    if stage == "serialize":
        monkeypatch.setattr(torch, "save", interrupted_write)
    else:
        monkeypatch.setattr(train.os, "fsync", failed_sync)
    with pytest.raises(OSError, match="disk full|sync failed"):
        train._atomic_torch_save({"step": 4, "model": torch.arange(6)}, path)
    assert path.read_bytes() == original_bytes
    assert train.load_torch_checkpoint(path)["step"] == 3
    assert sorted(item.name for item in tmp_path.iterdir()) == ["ckpt_last.pt"]


@pytest.mark.parametrize("field,value", [
    ("scale", float("nan")), ("scale", -1.0), ("scale", "1"),
    ("growth_factor", 1.0), ("backoff_factor", 1.5),
    ("growth_interval", 0), ("_growth_tracker", -1),
])
def test_amp_resume_rejects_invalid_scaler_state_before_mutation(field, value) -> None:
    scaler = torch.amp.GradScaler("cpu", init_scale=128.0)
    before = scaler.state_dict()
    invalid = {**before, field: value}
    with pytest.raises(ValueError, match="gradient-scaler"):
        train._restore_scaler_state(scaler, {"scaler": invalid}, enabled=True)
    assert scaler.state_dict() == before


def test_amp_resume_requires_and_restores_scaler_state() -> None:
    scaler = torch.amp.GradScaler("cpu", init_scale=128.0)
    for checkpoint in ({}, {"scaler": None}, {"scaler": {}}):
        with pytest.raises(ValueError, match="gradient-scaler"):
            train._restore_scaler_state(scaler, checkpoint, enabled=True)
    saved = {**scaler.state_dict(), "scale": 256.0, "_growth_tracker": 17}
    train._restore_scaler_state(scaler, {"scaler": saved}, enabled=True)
    assert scaler.state_dict() == saved


@pytest.mark.parametrize("checkpoint", [{}, {"scheduler": None}, {"scheduler": {}}, {"scheduler": {"last_epoch": 9}}])
def test_scheduler_resume_rejects_missing_or_incompatible_state_without_mutation(checkpoint) -> None:
    optimizer = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=0.25)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.2)
    original = deepcopy(scheduler.state_dict())
    with pytest.raises(ValueError, match="scheduler"):
        train._restore_scheduler_state(scheduler, checkpoint)
    assert scheduler.state_dict() == original
    assert optimizer.param_groups[0]["lr"] == 0.25


def test_scheduler_resume_preserves_next_learning_rate_updates() -> None:
    optimizer = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=0.25)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.2)
    for _ in range(4):
        optimizer.step()
        scheduler.step()
    checkpoint = {"scheduler": deepcopy(scheduler.state_dict()), "opt": deepcopy(optimizer.state_dict())}
    expected = []
    for _ in range(4):
        optimizer.step()
        scheduler.step()
        expected.append(optimizer.param_groups[0]["lr"])
    resumed_optimizer = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=0.25)
    resumed_scheduler = torch.optim.lr_scheduler.StepLR(resumed_optimizer, step_size=3, gamma=0.2)
    train._restore_scheduler_state(resumed_scheduler, checkpoint)
    resumed_optimizer.load_state_dict(checkpoint["opt"])
    actual = []
    for _ in range(4):
        resumed_optimizer.step()
        resumed_scheduler.step()
        actual.append(resumed_optimizer.param_groups[0]["lr"])
    assert actual == expected


def test_scheduler_resume_rejects_corrupt_clock_before_mutation() -> None:
    optimizer = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=0.25)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.2)
    original = deepcopy(scheduler.state_dict())
    with pytest.raises(ValueError, match="scheduler last_epoch"):
        train._restore_scheduler_state(scheduler, {"scheduler": {**original, "last_epoch": "4"}})
    assert scheduler.state_dict() == original


@pytest.mark.parametrize("saved", [{}, {"updates": 7}])
def test_custom_scheduler_may_restore_empty_or_dynamic_state(saved) -> None:
    class CustomScheduler:
        def __init__(self):
            self.updates = 0

        def step(self):
            self.updates += 1

        def state_dict(self):
            return {"updates": self.updates} if self.updates else {}

        def load_state_dict(self, state):
            self.updates = state.get("updates", 0)

    scheduler = CustomScheduler()
    assert scheduler.state_dict() == {}
    train._restore_scheduler_state(scheduler, {"scheduler": saved})
    scheduler.step()
    assert scheduler.updates == saved.get("updates", 0) + 1


def _exposure_checkpoint():
    return {
        "step": 3,
        "cumulative_samples_seen": 12,
        "cumulative_cropped_residues_seen": 70,
        "cumulative_nonpad_residues_seen": 65,
        "cumulative_residues_seen": 65,
    }


@pytest.mark.parametrize("field,value", [
    ("step", -1), ("cumulative_samples_seen", -999), ("cumulative_samples_seen", True),
    ("cumulative_samples_seen", 11), ("cumulative_cropped_residues_seen", "70"),
    ("cumulative_cropped_residues_seen", 73), ("cumulative_nonpad_residues_seen", 71),
    ("cumulative_nonpad_residues_seen", 11), ("cumulative_residues_seen", 64),
    ("cumulative_residues_seen", None),
])
def test_resume_rejects_corrupt_or_impossible_exposure_counters(field, value) -> None:
    checkpoint = {**_exposure_checkpoint(), field: value}
    with pytest.raises(ValueError, match="exposure"):
        train._resume_exposure_counters(checkpoint, effective_batch_size=4, crop_size=6)


def test_resume_requires_explicit_exposure_counters_and_preserves_actual_residue_counts() -> None:
    checkpoint = _exposure_checkpoint()
    assert train._resume_exposure_counters(checkpoint, effective_batch_size=4, crop_size=6) == (12, 70, 65)
    for field in checkpoint:
        incomplete = {key: value for key, value in checkpoint.items() if key != field}
        with pytest.raises(ValueError, match="exposure"):
            train._resume_exposure_counters(incomplete, effective_batch_size=4, crop_size=6)
    assert train._resume_exposure_counters(
        dict.fromkeys(checkpoint, 0), effective_batch_size=4, crop_size=6,
    ) == (0, 0, 0)


def test_stateless_custom_scheduler_still_requires_explicit_checkpoint_state() -> None:
    class StatelessScheduler:
        def state_dict(self):
            return {}

        def load_state_dict(self, state):
            assert state == {}

    with pytest.raises(ValueError, match="scheduler"):
        train._restore_scheduler_state(StatelessScheduler(), {})
    train._restore_scheduler_state(StatelessScheduler(), {"scheduler": {}})


@pytest.mark.parametrize("split,kind", [("train", "features"), ("train", "labels"), ("val", "features"), ("val", "labels")])
def test_unfingerprinted_resume_binds_processed_input_bytes(tmp_path, split, kind) -> None:
    features, labels = tmp_path / "features", tmp_path / "labels"
    features.mkdir()
    labels.mkdir()
    loaders = {}
    paths = {}
    for split_name, chain in (("train", "1abc_A"), ("val", "2def_B")):
        manifest = tmp_path / f"{split_name}.txt"
        manifest.write_text(chain + "\n")
        for name, directory in (("features", features), ("labels", labels)):
            path = train.chain_npz_path(directory, chain)
            path.write_bytes(b"original input")
            paths[split_name, name] = path
        dataset = train.ProcessedNPZDataset(
            features, manifest, processed_labels_dir=labels, include_labels=True,
        )
        loaders[split_name] = DataLoader(dataset, generator=torch.Generator().manual_seed(0))
    data_hash = train._training_data_sha256(loaders["train"], loaders["val"])
    original_contract = train._training_sampling_contract(
        loaders["train"], world_size=1, local_grad_accum_steps=1, data_sha256=data_hash,
    )
    paths[split, kind].write_bytes(b"modified input")
    changed_hash = train._training_data_sha256(loaders["train"], loaders["val"])
    assert changed_hash != data_hash
    changed_contract = train._training_sampling_contract(
        loaders["train"], world_size=1, local_grad_accum_steps=1, data_sha256=changed_hash,
    )
    checkpoint = {"sampling_state": {"schema_version": 1, "contract": original_contract}}
    with pytest.raises(ValueError, match="processed inputs"):
        train._resume_rank_sampling_state(checkpoint, changed_contract, rank=0)
