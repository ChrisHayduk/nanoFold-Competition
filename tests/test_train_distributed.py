from __future__ import annotations

import json
import socket
from datetime import timedelta
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import Dataset, DistributedSampler, RandomSampler, SequentialSampler, WeightedRandomSampler

import train


@pytest.mark.parametrize("weights", [[], [[1.0]], [-1.0, 2.0], [0.0], [float("nan")], [float("inf")], [1e308, 1e308]])
def test_weighted_sampler_rejects_invalid_weights(weights) -> None:
    with pytest.raises(ValueError, match="weights"):
        train.WeightedDistributedSampler(torch.tensor(weights, dtype=torch.float64), num_replicas=2, rank=0, seed=7)


@pytest.mark.parametrize("replicas,rank", [(0, 0), (-1, 0), (2, -1), (2, 2)])
def test_weighted_sampler_rejects_invalid_rank_or_replica_count(replicas, rank) -> None:
    with pytest.raises(ValueError, match="num_replicas|rank"):
        train.WeightedDistributedSampler(torch.ones(4), num_replicas=replicas, rank=rank, seed=7)


def test_weighted_sampler_ranks_partition_one_reproducible_global_draw() -> None:
    weights = torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0], dtype=torch.float64)
    samplers = [train.WeightedDistributedSampler(weights, num_replicas=2, rank=rank, seed=11) for rank in range(2)]
    for epoch in (0, 1, 5):
        for sampler in samplers:
            sampler.set_epoch(epoch)
        shards = [list(sampler) for sampler in samplers]
        combined = [value for pair in zip(*shards, strict=True) for value in pair]
        expected = torch.multinomial(weights, 6, replacement=True, generator=torch.Generator().manual_seed(11 + epoch))
        assert combined == expected.tolist()
        assert all(len(shard) == len(sampler) == 3 for shard, sampler in zip(shards, samplers, strict=True))
        assert 0 not in combined
        assert list(samplers[0]) == shards[0]


@pytest.mark.parametrize("suffix,contents", [
    (".json", '{"a": 2, "b": 0, "extra": 3}'),
    (".csv", "# comment\na, 2\n\nb,0\nextra,3\n"),
    (".txt", "a 2\nb 0\nextra 3\n"),
])
def test_weight_files_align_weights_to_manifest_order(tmp_path, suffix, contents) -> None:
    path = tmp_path / f"weights{suffix}"
    path.write_text(contents)
    assert train._load_train_sample_weights(path, ["b", "a"]).tolist() == [0.0, 2.0]


@pytest.mark.parametrize("contents,error", [
    ("[]", "JSON mapping"),
    ('{"a": 1}', "missing"),
    ('{"a": -1, "b": 2}', "Invalid"),
    ('{"a": 0, "b": 0}', "Invalid"),
    ('{"a": NaN, "b": 1}', "Invalid"),
    ('{"a": 1e308, "b": 1e308}', "Invalid"),
])
def test_weight_files_reject_incomplete_or_invalid_weights(tmp_path, contents, error) -> None:
    path = tmp_path / "weights.json"
    path.write_text(contents)
    with pytest.raises(ValueError, match=error):
        train._load_train_sample_weights(path, ["a", "b"])


def test_weight_loader_rejects_empty_manifest_and_missing_file(tmp_path) -> None:
    with pytest.raises(ValueError, match="empty manifest"):
        train._load_train_sample_weights(tmp_path / "absent", [])
    with pytest.raises(FileNotFoundError, match="Missing train sample weights"):
        train._load_train_sample_weights(tmp_path / "absent", ["a"])


@pytest.mark.parametrize("value", [False, "false", "0", "no", "off", None])
def test_weighted_sampling_opt_in_rejects_false_values(value) -> None:
    with pytest.raises(ValueError, match="explicitly labeled"):
        train._validate_train_sample_weights_opt_in({"train_sample_weights": "weights.json", "allow_train_sample_weights": value})


@pytest.mark.parametrize("value", [True, "true", "1", "yes", "ON"])
def test_weighted_sampling_opt_in_accepts_true_values(value) -> None:
    train._validate_train_sample_weights_opt_in({"train_sample_weights": "weights.json", "allow_train_sample_weights": value})
    train._validate_train_sample_weights_opt_in({})


@pytest.mark.parametrize("split,weighted,world_size,expected_sampler", [
    ("train", False, 1, RandomSampler),
    ("train", False, 2, DistributedSampler),
    ("train", True, 1, WeightedRandomSampler),
    ("train", True, 2, train.WeightedDistributedSampler),
    ("val", True, 2, SequentialSampler),
])
def test_loader_uses_sampling_only_for_training(monkeypatch, tmp_path, split, weighted, world_size, expected_sampler) -> None:
    class StubDataset(Dataset):
        chain_ids = ["a", "b", "c", "d"]

        def __init__(self, **kwargs):
            pass

        def __len__(self):
            return len(self.chain_ids)

        def __getitem__(self, index):
            return self.chain_ids[index]

    monkeypatch.setattr(train, "ProcessedNPZDataset", StubDataset)
    weights_path = tmp_path / "weights.json"
    weights_path.write_text(json.dumps(dict.fromkeys(StubDataset.chain_ids, 1)))
    config = {"data": {
        "processed_features_dir": "features", "train_manifest": "train", "val_manifest": "val",
        "crop_size": 2, "msa_depth": 1, "batch_size": 1,
        "train_sample_weights": str(weights_path) if weighted else "", "allow_train_sample_weights": True,
    }}
    loader = train.make_loader(
        config, split, device=torch.device("cpu"), include_labels=True, fail_if_labels_present=False,
        allow_missing=False, generator_seed=13, distributed_rank=0, distributed_world_size=world_size,
    )
    assert isinstance(loader.sampler, expected_sampler)
    assert loader.drop_last is (split == "train")
    assert len(loader) == (2 if split == "train" and world_size == 2 else 4)
    assert (train._epoch_aware_sampler(loader.sampler) is not None) is (split == "train" and world_size == 2)


@pytest.mark.parametrize("value", [False, True])
def test_single_device_metrics_do_not_allocate_float64_on_mps(monkeypatch, value) -> None:
    def unexpected_collective(*args, **kwargs):
        raise AssertionError("Single-device training must not invoke a collective")

    monkeypatch.setattr(dist, "all_reduce", unexpected_collective)
    device = torch.device("mps")
    total = train._distributed_sum_tensor([1.25, 3.0], device=device, enabled=False)
    assert total.device.type == "cpu"
    assert total.tolist() == [1.25, 3.0]
    assert train._distributed_any(value, device=device, enabled=False) is value
    assert train._distributed_metric_names({"z": 1.0, "a": 2.0}, enabled=False) == ["a", "z"]


@pytest.mark.parametrize("batch_size,world_size,expected_local_steps", [(1, 1, 8), (1, 2, 4), (2, 2, 2), (1, 8, 1)])
def test_distributed_accumulation_preserves_the_global_sample_budget(batch_size, world_size, expected_local_steps) -> None:
    local_steps = train._distributed_grad_accum_steps(
        effective_batch_size=8, batch_size=batch_size, world_size=world_size,
    )
    assert local_steps == expected_local_steps
    assert batch_size * local_steps * world_size == 8
    assert train.compute_sample_budget(100, batch_size * local_steps * world_size) == 800


@pytest.mark.parametrize("effective_batch,batch_size,world_size", [(8, 1, 3), (8, 1, 16), (8, 3, 2), (0, 1, 2), (8, 0, 2), (8, 1, 0)])
def test_distributed_accumulation_rejects_inexact_or_invalid_global_batches(effective_batch, batch_size, world_size) -> None:
    with pytest.raises(ValueError, match="positive|preserve"):
        train._distributed_grad_accum_steps(
            effective_batch_size=effective_batch, batch_size=batch_size, world_size=world_size,
        )


@pytest.mark.parametrize("timeout", [None, "37"])
def test_distributed_initialization_sets_cuda_device_and_nccl_timeout(monkeypatch, timeout) -> None:
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setenv("RANK", "1")
    monkeypatch.setenv("LOCAL_RANK", "1")
    monkeypatch.delenv("NANOFOLD_DDP_TIMEOUT_SECONDS", raising=False)
    if timeout is not None:
        monkeypatch.setenv("NANOFOLD_DDP_TIMEOUT_SECONDS", timeout)
    calls = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "set_device", lambda device: calls.append(device))
    monkeypatch.setattr(dist, "init_process_group", lambda **kwargs: calls.append(kwargs))
    info = train._init_distributed()
    expected_init: dict[str, object] = {"backend": "nccl"}
    if timeout is not None:
        expected_init["timeout"] = timedelta(seconds=37)
    assert calls == [1, expected_init]
    assert info == {"enabled": True, "world_size": 2, "rank": 1, "local_rank": 1, "is_main": False}


def test_distributed_initialization_requires_cuda_and_positive_timeout(monkeypatch) -> None:
    monkeypatch.setenv("WORLD_SIZE", "1")
    assert train._init_distributed()["enabled"] is False
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA"):
        train._init_distributed()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "set_device", lambda device: None)
    monkeypatch.setenv("NANOFOLD_DDP_TIMEOUT_SECONDS", "0")
    with pytest.raises(ValueError, match="positive"):
        train._init_distributed()


def _gloo_metric_worker(rank: int, rendezvous: str, result_dir: str) -> None:
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2, timeout=timedelta(seconds=20))
    try:
        local_metrics = {"first": 2.0} if rank == 0 else {"second": 6.0}
        names = train._distributed_metric_names(local_metrics, enabled=True)
        totals = train._distributed_sum_tensor(
            [float(rank + 1), *[local_metrics.get(name, 0.0) for name in names]],
            device=torch.device("cpu"), enabled=True,
        )
        any_rank = train._distributed_any(rank == 1, device=torch.device("cpu"), enabled=True)
        no_rank = train._distributed_any(False, device=torch.device("cpu"), enabled=True)
        train.set_seed(100 + rank)
        sampling = train._collect_training_sampling_state(
            {"rank": rank, "rng_state": train._capture_rng_state(torch.device("cpu"))},
            {"world_size": 2}, distributed=True,
        )
        expected_random = torch.rand(5)
        train.set_seed(999)
        train._restore_rng_state(sampling["rank_states"][rank]["rng_state"], torch.device("cpu"))
        rank_rng_restored = torch.equal(torch.rand(5), expected_random)
        separate_rank_states = not torch.equal(
            sampling["rank_states"][0]["rng_state"]["torch"],
            sampling["rank_states"][1]["rng_state"]["torch"],
        )
        write_failed_on_all_ranks = False
        try:
            train._checkpoint_write_error("disk full" if rank == 0 else None, distributed=True)
        except RuntimeError as exc:
            write_failed_on_all_ranks = "disk full" in str(exc)
        train._distributed_barrier(True)
        Path(result_dir, f"rank{rank}.json").write_text(json.dumps({
            "names": names, "totals": totals.tolist(), "any": any_rank, "none": no_rank,
            "rank_rng_restored": rank_rng_restored, "separate_rank_states": separate_rank_states,
            "write_failed_on_all_ranks": write_failed_on_all_ranks,
        }))
    finally:
        train._destroy_distributed(True)


@pytest.mark.skipif(not dist.is_available() or not dist.is_gloo_available(), reason="PyTorch gloo backend unavailable")
def test_distributed_helpers_reduce_unequal_metric_sets_across_processes(monkeypatch, tmp_path) -> None:
    loopback = next((name for _, name in socket.if_nameindex() if name in {"lo", "lo0"}), None)
    if loopback is not None:
        monkeypatch.setenv("GLOO_SOCKET_IFNAME", loopback)
    rendezvous = (tmp_path / "rendezvous").as_uri()
    mp.spawn(_gloo_metric_worker, args=(rendezvous, str(tmp_path)), nprocs=2, join=True)
    expected = {
        "names": ["first", "second"], "totals": [3.0, 2.0, 6.0], "any": True, "none": False,
        "rank_rng_restored": True, "separate_rank_states": True,
        "write_failed_on_all_ranks": True,
    }
    assert json.loads((tmp_path / "rank0.json").read_text()) == expected
    assert json.loads((tmp_path / "rank1.json").read_text()) == expected
