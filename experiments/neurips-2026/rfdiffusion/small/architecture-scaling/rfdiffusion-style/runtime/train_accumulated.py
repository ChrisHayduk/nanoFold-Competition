#!/usr/bin/env python3
"""Additive RF trainer: eight logical examples per update on 1/2/4/8 GPUs.

Copied from the SHA256-pinned completed trainer; scientific model, data,
losses and optimizer are unchanged. Physical-rank accumulation is operational.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import socket
import sys
import time
from collections.abc import Callable
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel


PACKAGE_DIR = Path(__file__).resolve().parents[1]
THIS_DIR = PACKAGE_DIR.parents[1] / "architecture-reconstructions" / "rfdiffusion-style"
COMMON_DIR = THIS_DIR.parent / "common"
sys.path.insert(0, str(COMMON_DIR))
sys.path.insert(0, str(THIS_DIR))

from data import (  # noqa: E402
    DeterministicBatchSchedule,
    NanoFoldTensorLoader,
    read_manifest,
)
from geometry import (  # noqa: E402
    apply_rotation,
    kabsch_align,
    random_rotation,
    smooth_lddt_loss,
)
from model import (  # noqa: E402
    DIFFUSION_CONTRACT_ID,
    IMPUTATION_CONTRACT_ID,
    MODEL_FEATURE_KEYS,
    RFStyleConfig,
    RFdiffusionStyleModel,
    diffuse_backbone,
)
from runtime import (  # noqa: E402
    atomic_json,
    atomic_torch_save,
    capture_rng_state,
    gather_rank_rng_states,
    initialize_distributed,
    restore_rng_state,
    seed_process,
    sha256_file,
    verify_sha256_sidecar,
    write_heartbeat,
)
from verify_dataset import verify_dataset_snapshot  # noqa: E402


EFFECTIVE_BATCH_SIZE = 8
ACCUMULATION_CONTRACT_ID = "rf-effective8-logical-rank-accumulation-v1"


def execution_contract(physical_world_size: int) -> dict[str, Any]:
    if physical_world_size not in (1, 2, 4, 8):
        raise ValueError("physical world size must be one of 1, 2, 4, 8")
    return {
        "contract_id": ACCUMULATION_CONTRACT_ID,
        "physical_world_size": physical_world_size,
        "logical_world_size": EFFECTIVE_BATCH_SIZE,
        "effective_batch_size": EFFECTIVE_BATCH_SIZE,
        "gradient_accumulation_steps": EFFECTIVE_BATCH_SIZE // physical_world_size,
        "logical_rank_mapping": "microstep * physical_world_size + physical_rank",
        "example_rng": "sha256(base_seed,optimizer_step,logical_rank)",
        "ddp_static_graph": False,
        "boundary": "same effective batch/data schedule; not bitwise RNG-equivalent to original eight-rank trainer",
    }


def logical_rank_for(*, physical_rank: int, microstep: int, physical_world_size: int) -> int:
    contract = execution_contract(physical_world_size)
    if not 0 <= physical_rank < physical_world_size:
        raise ValueError("invalid physical rank")
    if not 0 <= microstep < contract["gradient_accumulation_steps"]:
        raise ValueError("invalid accumulation microstep")
    return microstep * physical_world_size + physical_rank


def example_seed(base_seed: int, step: int, logical_rank: int) -> int:
    if step < 0 or not 0 <= logical_rank < EFFECTIVE_BATCH_SIZE:
        raise ValueError("invalid logical example coordinate")
    value = hashlib.sha256(f"{ACCUMULATION_CONTRACT_ID}\0{base_seed}\0{step}\0{logical_rank}".encode()).digest()
    return int.from_bytes(value[:8], "little") & ((1 << 63) - 1)


def accumulation_context(model: torch.nn.Module, microstep: int, accumulation_steps: int):
    if isinstance(model, DistributedDataParallel) and microstep + 1 < accumulation_steps:
        return model.no_sync()
    return nullcontext()


def execution_metadata(config: dict[str, Any], world_size: int, step: int) -> dict[str, Any]:
    contract = execution_contract(world_size)
    if config.get("_execution_contract") != contract:
        raise ValueError("missing or mismatched accumulation execution contract")
    return {
        **contract,
        "samples_seen": int(step) * EFFECTIVE_BATCH_SIZE,
        "optimizer_updates": int(step),
    }


def validate_production_roster(config: dict[str, Any], report: dict[str, Any]) -> None:
    """Bound every normal 30k run to the exact frozen public train/val corpus.

    Short configurations are explicit diagnostics. The subsequent existing
    tensor-tree validator still rehashes their complete declared dataset.
    """
    if int(config["training"]["steps"]) != 30000:
        return
    expected = {
        "train_count": 10000,
        "validation_count": 1000,
        "train_manifest_sha256": "d36d1f77ba43b7c4509a6e9dfd3f9414e1ce60f8364b24e0086c1734ba6aef6d",
        "validation_manifest_sha256": "d4a0265bcd0a021e116c0c889f21e86bc24006460bcc42dec2f9a80b70c8812b",
        "tree_sha256": "98c141bafcb51c2a8532ec3eba115ff170e9a9da9c54e4f18e5e7c83b4ef1e00",
    }
    actual = dataset_identity(report)
    for name, value in expected.items():
        if actual[name] != value:
            raise ValueError(f"production frozen-roster mismatch: {name}")


def to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            output[key] = value.unsqueeze(0).to(device, non_blocking=True)
        else:
            output[key] = value
    return output


def distance_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    atom_mask: torch.Tensor,
) -> torch.Tensor:
    pred_ca = prediction[:, :, 1].float()
    true_ca = target[:, :, 1].float()
    ca_mask = atom_mask[:, :, 1]
    pred_distance = torch.cdist(pred_ca, pred_ca)
    true_distance = torch.cdist(true_ca, true_ca)
    pair_mask = ca_mask[:, :, None] & ca_mask[:, None, :]
    pair_mask &= true_distance < 20.0
    diagonal = torch.eye(
        true_distance.shape[-1], device=true_distance.device, dtype=torch.bool
    )
    pair_mask &= ~diagonal.unsqueeze(0)
    error = F.smooth_l1_loss(
        pred_distance / 10.0,
        true_distance / 10.0,
        reduction="none",
        beta=0.1,
    )
    weight = pair_mask.to(error.dtype)
    return (error * weight).sum() / weight.sum().clamp_min(1.0)


def aligned_atom_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    atom_mask: torch.Tensor,
) -> torch.Tensor:
    aligned = kabsch_align(prediction.float(), target.float(), atom_mask)
    error = torch.sqrt((aligned - target.float()).square().sum(-1) + 1e-6)
    error = torch.clamp(error, max=10.0) / 10.0
    weight = atom_mask.to(error.dtype)
    return (error * weight).sum() / weight.sum().clamp_min(1.0)


def rotation_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    token_mask: torch.Tensor,
) -> torch.Tensor:
    relative = prediction.float().transpose(-1, -2) @ target.float()
    cosine = ((relative.diagonal(dim1=-2, dim2=-1).sum(-1) - 1.0) / 2.0).clamp(
        -1.0, 1.0
    )
    loss = 1.0 - cosine
    weight = token_mask.to(loss.dtype)
    return (loss * weight).sum() / weight.sum().clamp_min(1.0)


def translation_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    token_mask: torch.Tensor,
) -> torch.Tensor:
    error = F.smooth_l1_loss(
        prediction.float() / 10.0,
        target.float() / 10.0,
        reduction="none",
        beta=0.1,
    ).sum(-1)
    weight = token_mask.to(error.dtype)
    return (error * weight).sum() / weight.sum().clamp_min(1.0)


def learning_rate(step: int, config: dict[str, Any]) -> float:
    base = float(config["training"]["learning_rate"])
    warmup = int(config["training"]["warmup_steps"])
    total = int(config["training"]["steps"])
    if step < warmup:
        return base * float(step + 1) / float(max(1, warmup))
    progress = float(step - warmup) / float(max(1, total - warmup))
    return base * 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


def deterministic_overfit_seed(
    run_seed: int,
    chain_id: str,
    rank: int,
    timestep: int,
) -> int:
    """Bind one exact overfit example to its run, chain, rank, and timestep."""
    digest = hashlib.sha256(
        (
            "nanofold-rfdiffusion-fixed-overfit-v1"
            f"\0{run_seed}\0{chain_id}\0{rank}\0{timestep}"
        ).encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "little") & ((1 << 63) - 1)


def validate_fixed_overfit_contract(config: dict[str, Any]) -> int | None:
    """Validate and return the fixed timestep for an exact memorization gate."""
    training = config["training"]
    if not bool(training.get("overfit_fixed_example", False)):
        return None
    if not bool(training.get("overfit_first_rank_batch", False)):
        raise ValueError(
            "overfit_fixed_example requires overfit_first_rank_batch"
        )
    if float(config["model"].get("dropout", 0.0)) != 0.0:
        raise ValueError("overfit_fixed_example requires model dropout == 0")
    timestep = int(training.get("overfit_fixed_timestep", 0))
    diffusion_steps = int(training["diffusion_steps"])
    if not 1 <= timestep <= diffusion_steps:
        raise ValueError(
            "overfit_fixed_timestep must lie inside the diffusion schedule"
        )
    return timestep


def all_reduce_metrics(metrics: dict[str, torch.Tensor]) -> dict[str, float]:
    reduced: dict[str, float] = {}
    for key, value in metrics.items():
        detached = value.detach().float()
        if dist.is_initialized():
            dist.all_reduce(detached, op=dist.ReduceOp.SUM)
            detached /= dist.get_world_size()
        reduced[key] = float(detached.cpu())
    return reduced


def append_history(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def dataset_identity(record: dict[str, Any]) -> dict[str, Any]:
    """Extract the immutable scientific identity from a verification report."""
    inventory = record["inventory"]
    manifests = record["manifest_sha256"]
    return {
        "train_count": int(record["train_count"]),
        "validation_count": int(record["validation_count"]),
        "train_manifest_sha256": str(manifests["train"]),
        "validation_manifest_sha256": str(manifests["validation"]),
        "file_count": int(inventory["file_count"]),
        "total_bytes": int(inventory["total_bytes"]),
        "tree_sha256": str(inventory["tree_sha256"]),
    }


def validate_staged_dataset(
    *,
    data_root: Path,
    dataset_verification: dict[str, Any],
) -> None:
    """Rehash the complete manifest-bound staged tensor tree."""
    verify_dataset_snapshot(data_root, dataset_verification)


def verify_checkpoint_sidecar(path: Path) -> dict[str, Any]:
    """Verify a checkpoint's atomic-save sidecar before unpickling it."""
    return verify_sha256_sidecar(path)


def _config_without_source_provenance(
    config: dict[str, Any],
) -> dict[str, Any]:
    return {key: value for key, value in config.items() if key != "_source_provenance"}


def _source_hashes(config: dict[str, Any]) -> dict[str, str]:
    provenance = config.get("_source_provenance")
    if not isinstance(provenance, dict):
        raise ValueError("config lacks source provenance")
    hashes: dict[str, str] = {}
    for name, record in provenance.items():
        if not isinstance(record, dict) or not record.get("sha256"):
            raise ValueError(f"source provenance is incomplete for {name}")
        hashes[str(name)] = str(record["sha256"])
    return hashes


def validate_resume_contract(
    *,
    checkpoint: dict[str, Any],
    config: dict[str, Any],
    dataset_verification: dict[str, Any],
    world_size: int,
) -> None:
    """Reject any resume that would silently mix scientific run identities."""
    if checkpoint.get("architecture") != "nanofold-rfdiffusion-style":
        raise ValueError("resume checkpoint has a different architecture")
    if int(checkpoint.get("world_size", -1)) != world_size:
        raise ValueError("resume world size differs from checkpoint")
    expected_execution = execution_metadata(config, world_size, int(checkpoint.get("step", -1)))
    for key, expected in expected_execution.items():
        if checkpoint.get(key) != expected:
            raise ValueError(f"resume accumulation metadata mismatch: {key}")
    checkpoint_config = checkpoint.get("config")
    if not isinstance(checkpoint_config, dict):
        raise ValueError("resume checkpoint lacks its training config")
    if _config_without_source_provenance(
        checkpoint_config
    ) != _config_without_source_provenance(config):
        raise ValueError("resume training/data/model config differs from checkpoint")
    if _source_hashes(checkpoint_config) != _source_hashes(config):
        raise ValueError("resume source provenance differs from checkpoint")
    checkpoint_dataset = checkpoint.get("dataset_verification")
    if not isinstance(checkpoint_dataset, dict):
        raise ValueError("resume checkpoint lacks dataset verification")
    if dataset_identity(checkpoint_dataset) != dataset_identity(dataset_verification):
        raise ValueError("resume dataset identity differs from checkpoint")
    step = int(checkpoint.get("step", -1))
    total_steps = int(config["training"]["steps"])
    if not 0 <= step <= total_steps:
        raise ValueError(
            f"resume step {step} is outside configured range [0, {total_steps}]"
        )
    rank_rng_states = checkpoint.get("rank_rng_states")
    if not isinstance(rank_rng_states, list) or len(rank_rng_states) != world_size:
        raise ValueError("resume checkpoint has incomplete per-rank RNG states")
    if not isinstance(checkpoint.get("model"), dict):
        raise ValueError("resume checkpoint lacks model state")
    if not isinstance(checkpoint.get("optimizer"), dict):
        raise ValueError("resume checkpoint lacks optimizer state")


def run_directory_has_state(run_dir: Path) -> bool:
    return any(
        (
            (run_dir / "run-state.json").exists(),
            (run_dir / "config.json").exists(),
            (run_dir / "status.json").exists(),
            (run_dir / "logs" / "history.jsonl").exists(),
            (
                any((run_dir / "checkpoints").glob("*.pt"))
                if (run_dir / "checkpoints").is_dir()
                else False
            ),
        )
    )


def read_run_state(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "run-state.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"{run_dir} has no run-state.json and is not a verified resumable run"
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def update_run_state(run_dir: Path, **updates: Any) -> dict[str, Any]:
    path = run_dir / "run-state.json"
    payload: dict[str, Any] = {}
    if path.is_file():
        payload = read_run_state(run_dir)
    payload.update(updates)
    payload["schema_version"] = 1
    payload["updated_unix_time"] = time.time()
    atomic_json(path, payload)
    return payload


def prepare_run_directory(
    *,
    run_dir: Path,
    config: dict[str, Any],
    resume_path: Path | None,
    world_size: int,
) -> None:
    """Validate and initialize run state. This must execute on rank zero only."""
    has_state = run_directory_has_state(run_dir)
    previous_state: dict[str, Any] | None = None
    if resume_path is None:
        if has_state:
            raise FileExistsError(
                f"{run_dir} already contains run state; use a new run ID. "
                "Only a run explicitly marked resumable may be continued with "
                "--resume and a verified checkpoint."
            )
    elif has_state:
        existing_config_path = run_dir / "config.json"
        if not existing_config_path.is_file():
            raise ValueError(
                f"{run_dir} contains partial run state without config.json; "
                "use a new run ID"
            )
        existing_config = json.loads(existing_config_path.read_text(encoding="utf-8"))
        if existing_config != config:
            raise ValueError(
                f"{run_dir} contains a different run config; refusing to mix "
                "resume histories"
            )
        previous_state = read_run_state(run_dir)
        if previous_state.get("resumable") is not True:
            raise ValueError(
                f"{run_dir} is marked non-resumable; use a new run ID and a "
                "verified checkpoint"
            )

    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (run_dir / "logs").mkdir(parents=True, exist_ok=True)
    (run_dir / "predictions").mkdir(parents=True, exist_ok=True)

    # Write the conservative lifecycle marker before any claim-bearing run
    # metadata. A brand-new directory remains non-resumable until either a
    # checkpoint is committed or a supplied resume checkpoint is fully loaded.
    initial_state = dict(previous_state or {})
    initial_state.update(
        {
            "state": "resuming" if resume_path is not None else "initializing",
            "resumable": bool(
                previous_state is not None and previous_state.get("resumable") is True
            ),
            "world_size": int(world_size),
            **execution_metadata(config, world_size, int(initial_state.get("step", 0))),
            "requested_resume": (
                str(resume_path.resolve()) if resume_path is not None else None
            ),
        }
    )
    update_run_state(run_dir, **initial_state)
    atomic_json(run_dir / "config.json", config)
    write_heartbeat(
        run_dir,
        state="initializing",
        step=0,
        rank=0,
        world_size=world_size,
        extra={
            "architecture": "RFdiffusion-style reconstruction",
            "pod_id": os.environ.get("RUNPOD_POD_ID"),
            "resumable": initial_state["resumable"],
            **execution_metadata(config, world_size, int(initial_state.get("step", 0))),
        },
    )


def distributed_rank_zero_action(
    *,
    rank: int,
    description: str,
    action: Callable[[], None],
) -> None:
    """Run one filesystem action on rank zero and propagate its outcome."""
    result: list[dict[str, Any] | None] = [None]
    if rank == 0:
        try:
            action()
            result[0] = {"ok": True}
        except BaseException as error:
            result[0] = {
                "ok": False,
                "error_type": type(error).__name__,
                "error": str(error),
            }
    if dist.is_initialized():
        dist.broadcast_object_list(result, src=0)
    outcome = result[0]
    if not outcome or outcome.get("ok") is not True:
        raise RuntimeError(f"{description} failed on rank zero: {outcome}")
    # All ranks skip the barrier on the same broadcast failure path. On
    # success, every rank enters it only after seeing the same committed state.
    if dist.is_initialized():
        dist.barrier()


def mark_training_complete(
    *,
    run_dir: Path,
    step: int,
    world_size: int,
    parameter_count: int,
    metrics: dict[str, float] | None = None,
    resume_from: dict[str, Any] | None = None,
    execution: dict[str, Any] | None = None,
) -> None:
    write_heartbeat(
        run_dir,
        state="training-complete",
        step=step,
        rank=0,
        world_size=world_size,
        metrics=metrics,
        extra={
            "parameter_count": parameter_count,
            "resume_from": resume_from,
            "resumable": True,
            **(execution or {}),
        },
    )
    update_run_state(
        run_dir,
        state="training-complete",
        resumable=True,
        step=step,
        resume_from=resume_from,
        **(execution or {}),
    )


_STEP_CHECKPOINT_PATTERN = re.compile(r"^step-(\d{7})\.pt$")


def prune_numbered_checkpoints(
    checkpoint_dir: Path,
    *,
    milestone_every: int,
    keep_recent_non_milestones: int,
) -> dict[str, list[int]]:
    """Retain every milestone and a bounded rolling recovery window."""
    if milestone_every < 1 or keep_recent_non_milestones < 1:
        raise ValueError("checkpoint retention values must be positive")
    checkpoints: list[tuple[int, Path]] = []
    for path in checkpoint_dir.glob("step-*.pt"):
        match = _STEP_CHECKPOINT_PATTERN.fullmatch(path.name)
        if match:
            checkpoints.append((int(match.group(1)), path))
    checkpoints.sort()
    milestones = {
        step for step, _ in checkpoints if step == 0 or step % milestone_every == 0
    }
    non_milestones = [step for step, _ in checkpoints if step not in milestones]
    retained = milestones | set(non_milestones[-keep_recent_non_milestones:])
    removed: list[int] = []
    for step, path in checkpoints:
        if step in retained:
            continue
        sidecar = path.with_suffix(path.suffix + ".sha256.json")
        path.unlink()
        sidecar.unlink(missing_ok=True)
        removed.append(step)
    payload = {
        "retained_steps": sorted(retained),
        "removed_steps": removed,
    }
    atomic_json(checkpoint_dir / "retention.json", payload)
    return payload


def save_checkpoint(
    *,
    run_dir: Path,
    step: int,
    model: DistributedDataParallel | RFdiffusionStyleModel,
    optimizer: torch.optim.Optimizer,
    config: dict[str, Any],
    dataset_verification: dict[str, Any],
    rank: int,
    world_size: int,
    milestone: bool,
    milestone_every: int,
    keep_recent_non_milestones: int,
    resume_from: dict[str, Any] | None = None,
) -> Path | None:
    rank_states = gather_rank_rng_states(capture_rng_state())
    if rank != 0:
        return None
    raw_model = model.module if isinstance(model, DistributedDataParallel) else model
    destination = run_dir / "checkpoints" / f"step-{step:07d}.pt"
    payload = {
        "schema_version": 3,
        "architecture": "nanofold-rfdiffusion-style",
        "step": int(step),
        "world_size": int(world_size),
        **execution_metadata(config, world_size, step),
        "model": raw_model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "rank_rng_states": rank_states,
        "config": config,
        "dataset_verification": dataset_verification,
        "resume_from": resume_from,
        "source": {
            "hostname": socket.gethostname(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
        },
    }
    checkpoint_sha256 = atomic_torch_save(destination, payload)
    latest = run_dir / "checkpoints" / "latest.pt"
    atomic_torch_save(latest, payload)
    retention = prune_numbered_checkpoints(
        run_dir / "checkpoints",
        milestone_every=milestone_every,
        keep_recent_non_milestones=keep_recent_non_milestones,
    )
    append_history(
        run_dir / "logs" / "history.jsonl",
        {
            "event": "checkpoint",
            "step": int(step),
            "milestone": bool(milestone),
            "path": str(destination),
            "retention": retention,
            "time": time.time(),
        },
    )
    update_run_state(
        run_dir,
        state="checkpoint-ready",
        resumable=True,
        step=int(step),
        **execution_metadata(config, world_size, step),
        last_checkpoint={
            "path": str(destination),
            "bytes": destination.stat().st_size,
            "sha256": checkpoint_sha256,
            "step": int(step),
        },
        resume_from=resume_from,
    )
    return destination


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--dataset-verification", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--physical-world-size", type=int, choices=(1, 2, 4, 8), required=True)
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    fixed_overfit_timestep = validate_fixed_overfit_contract(config)
    dataset_verification = json.loads(
        args.dataset_verification.read_text(encoding="utf-8")
    )
    validate_production_roster(config, dataset_verification)
    rank, local_rank, world_size, device = initialize_distributed()
    if int(config["training"]["world_size"]) != EFFECTIVE_BATCH_SIZE:
        raise ValueError("profile must retain the original logical world size/effective batch eight")
    config["_execution_contract"] = execution_contract(args.physical_world_size)
    accumulation_steps = config["_execution_contract"]["gradient_accumulation_steps"]
    expected_world_size = args.physical_world_size
    if world_size != expected_world_size:
        raise RuntimeError(
            f"config requires world_size={expected_world_size}, got {world_size}"
        )
    distributed_rank_zero_action(
        rank=rank,
        description="staged dataset verification",
        action=lambda: validate_staged_dataset(
            data_root=args.data_root.resolve(),
            dataset_verification=dataset_verification,
        ),
    )
    seed = int(config["training"]["seed"])
    seed_process(seed, rank)
    source_paths = {
        "train_accumulated.py": Path(__file__).resolve(),
        "parent_train.py": THIS_DIR / "train.py",
        "rf-contract.json": PACKAGE_DIR / "rf-contract.json",
        "model.py": THIS_DIR / "model.py",
        "common/data.py": COMMON_DIR / "data.py",
        "common/geometry.py": COMMON_DIR / "geometry.py",
        "common/runtime.py": COMMON_DIR / "runtime.py",
        "common/verify_dataset.py": COMMON_DIR / "verify_dataset.py",
        "config": args.config.resolve(),
        "dataset-verification": args.dataset_verification.resolve(),
        "train-manifest": args.data_root.resolve() / "manifests" / "train.txt",
        "validation-manifest": args.data_root.resolve() / "manifests" / "val.txt",
    }
    source_provenance = {
        name: {"path": str(path), "sha256": sha256_file(path)}
        for name, path in source_paths.items()
    }
    config["_source_provenance"] = source_provenance
    config["_scientific_contract"] = {
        "diffusion": DIFFUSION_CONTRACT_ID,
        "imputation": IMPUTATION_CONTRACT_ID,
    }

    run_dir = args.run_dir.resolve()
    resume_path = args.resume.resolve() if args.resume is not None else None
    distributed_rank_zero_action(
        rank=rank,
        description="run-directory initialization",
        action=lambda: prepare_run_directory(
            run_dir=run_dir,
            config=config,
            resume_path=resume_path,
            world_size=world_size,
        ),
    )

    data_config = config["data"]
    loader = NanoFoldTensorLoader(
        data_root=args.data_root,
        crop_size=int(data_config["crop_size"]),
        msa_depth=int(data_config["msa_depth"]),
        seed=seed,
    )
    train_ids = tuple(read_manifest(args.data_root / "manifests" / "train.txt"))
    schedule = DeterministicBatchSchedule(train_ids, seed=seed, world_size=EFFECTIVE_BATCH_SIZE)

    model_config = RFStyleConfig(**config["model"])
    raw_model = RFdiffusionStyleModel(model_config).to(device)
    parameter_count = sum(parameter.numel() for parameter in raw_model.parameters())
    model: DistributedDataParallel | RFdiffusionStyleModel
    if world_size > 1:
        model = DistributedDataParallel(
            raw_model,
            device_ids=[local_rank],
            output_device=local_rank,
            static_graph=False,
        )
    else:
        model = raw_model
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        betas=tuple(config["training"]["betas"]),
        eps=float(config["training"]["epsilon"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )

    start_step = 0
    resume_from: dict[str, Any] | None = None
    if resume_path is not None:
        verification_result: list[dict[str, Any] | None] = [None]
        if rank == 0:
            try:
                verification_result[0] = {
                    "ok": True,
                    **verify_checkpoint_sidecar(resume_path),
                }
            except BaseException as error:
                verification_result[0] = {
                    "ok": False,
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
        if dist.is_initialized():
            dist.broadcast_object_list(verification_result, src=0)
        checkpoint_verification = verification_result[0]
        if not checkpoint_verification or not checkpoint_verification["ok"]:
            raise RuntimeError(
                "resume checkpoint verification failed: " f"{checkpoint_verification}"
            )
        checkpoint_payload = torch.load(
            resume_path, map_location="cpu", weights_only=False
        )
        validate_resume_contract(
            checkpoint=checkpoint_payload,
            config=config,
            dataset_verification=dataset_verification,
            world_size=world_size,
        )
        raw_model.load_state_dict(checkpoint_payload["model"], strict=True)
        optimizer.load_state_dict(checkpoint_payload["optimizer"])
        start_step = int(checkpoint_payload["step"])
        restore_rng_state(checkpoint_payload["rank_rng_states"][rank])
        resume_from = {
            "path": str(resume_path),
            "bytes": int(checkpoint_verification["bytes"]),
            "sha256": str(checkpoint_verification["sha256"]),
            "step": start_step,
        }
        distributed_rank_zero_action(
            rank=rank,
            description="resume-state commit",
            action=lambda: update_run_state(
                run_dir,
                state="resume-loaded",
                resumable=True,
                step=start_step,
                last_checkpoint=resume_from,
            ),
        )

    if rank == 0:
        append_history(
            run_dir / "logs" / "history.jsonl",
            {
                "event": "start",
                "step": start_step,
                "world_size": world_size,
                **execution_metadata(config, world_size, start_step),
                "parameter_count": parameter_count,
                "dataset_tree_sha256": dataset_verification["inventory"]["tree_sha256"],
                "source_provenance": source_provenance,
                "time": time.time(),
            },
        )

    training = config["training"]
    total_steps = int(training["steps"])
    checkpoint_every = int(training["checkpoint_every"])
    milestone_every = int(training["milestone_every"])
    keep_recent_checkpoints = int(
        training.get("keep_recent_non_milestone_checkpoints", 2)
    )
    heartbeat_every = int(training["heartbeat_every"])
    log_every = int(training["log_every"])
    loss_weights = training["loss_weights"]
    use_bf16 = device.type == "cuda" and bool(training["bf16"])
    previous_time = time.perf_counter()

    if start_step == total_steps:
        distributed_rank_zero_action(
            rank=rank,
            description="training-complete state commit",
            action=lambda: mark_training_complete(
                run_dir=run_dir,
                step=total_steps,
                world_size=world_size,
                parameter_count=parameter_count,
                resume_from=resume_from,
                execution=execution_metadata(config, world_size, total_steps),
            ),
        )
        if dist.is_initialized():
            dist.destroy_process_group()
        return

    if start_step == 0:
        save_checkpoint(
            run_dir=run_dir,
            step=0,
            model=model,
            optimizer=optimizer,
            config=config,
            dataset_verification=dataset_verification,
            rank=rank,
            world_size=world_size,
            milestone=True,
            milestone_every=milestone_every,
            keep_recent_non_milestones=keep_recent_checkpoints,
            resume_from=resume_from,
        )
        if dist.is_initialized():
            dist.barrier()

    for step in range(start_step, total_steps):
        lr = learning_rate(step, config)
        for parameter_group in optimizer.param_groups:
            parameter_group["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        metric_sums = {name: torch.zeros((), device=device) for name in (
            "loss", "atom_loss", "lddt_loss", "distance_loss", "rotation_loss", "translation_loss"
        )}
        logical_examples = []
        if step < 20 or step % 1000 == 0:
            global_ids = [schedule.chain_at(step, logical_rank)[0] for logical_rank in range(EFFECTIVE_BATCH_SIZE)]
            if len(set(global_ids)) != EFFECTIVE_BATCH_SIZE:
                raise RuntimeError(f"duplicate virtual-batch chain IDs at update {step}: {global_ids}")
        for microstep in range(accumulation_steps):
            logical_rank = logical_rank_for(physical_rank=rank, microstep=microstep, physical_world_size=world_size)
            microbatch_seed = example_seed(seed, step, logical_rank)
            torch.manual_seed(microbatch_seed)
            if device.type == "cuda":
                torch.cuda.manual_seed(microbatch_seed)
            if bool(training.get("overfit_first_rank_batch", False)):
                chain_id, epoch, offset = train_ids[logical_rank], 0, logical_rank
            else:
                chain_id, epoch, offset = schedule.chain_at(step, logical_rank)
            loader_step = 0 if fixed_overfit_timestep is not None else step
            batch = to_device(
                loader.load(chain_id, train=True, step=loader_step, rank=logical_rank),
                device,
            )
            fixed_example_seed: int | None = None
            if fixed_overfit_timestep is not None:
                fixed_example_seed = deterministic_overfit_seed(
                    seed,
                    chain_id,
                    logical_rank,
                    fixed_overfit_timestep,
                )
                # Dropout is required to be zero in this mode. Resetting the
                # per-process generators here therefore makes the target rigid
                # augmentation, frame imputation, and diffusion draws exactly
                # identical on every optimizer step and in the paired evaluator.
                torch.manual_seed(fixed_example_seed)
                if device.type == "cuda":
                    torch.cuda.manual_seed_all(fixed_example_seed)
            clean_atom14 = batch["atom14_positions"]
            corruption_atom14 = batch["atom14_positions_uncentered"]
            rotation_augmentation = random_rotation(
                clean_atom14.shape[0], device=device, dtype=clean_atom14.dtype
            )
            clean_atom14 = apply_rotation(clean_atom14, rotation_augmentation)
            corruption_atom14 = apply_rotation(corruption_atom14, rotation_augmentation)
            batch["atom14_positions"] = clean_atom14
            if fixed_overfit_timestep is not None:
                timestep = torch.full(
                    (clean_atom14.shape[0],),
                    fixed_overfit_timestep,
                    dtype=torch.long,
                    device=device,
                )
            else:
                timestep = torch.randint(
                    1,
                    int(training["diffusion_steps"]) + 1,
                    (clean_atom14.shape[0],),
                    device=device,
                )
            diffusion = diffuse_backbone(
                corruption_atom14,
                batch["atom14_chemistry_mask"],
                timestep,
                total_steps=int(training["diffusion_steps"]),
                strict_complete_backbone=bool(
                    training.get("strict_complete_backbone", False)
                ),
            )
            model_batch = {key: batch[key] for key in MODEL_FEATURE_KEYS}

            with accumulation_context(model, microstep, accumulation_steps):
                autocast_context = (
                    torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                    if use_bf16
                    else nullcontext()
                )
                with autocast_context:
                    output = model(model_batch, diffusion["noisy_backbone"], timestep)
                pred_atom14 = output["pred_atom14"].float()
                supervision_atom_mask = batch["atom14_mask"] & batch["atom14_chemistry_mask"]
                atom = aligned_atom_loss(pred_atom14, clean_atom14, supervision_atom_mask)
                lddt = smooth_lddt_loss(pred_atom14, clean_atom14, supervision_atom_mask)
                distances = distance_loss(pred_atom14, clean_atom14, supervision_atom_mask)
                valid_frame_mask = (
                    supervision_atom_mask[:, :, :3].all(dim=-1)
                    & diffusion["clean_frame_geometry_mask"]
                )
                rotations = rotation_loss(
                    output["rotation"], diffusion["clean_rotation"], valid_frame_mask
                )
                translations = translation_loss(
                    output["translation"],
                    diffusion["clean_translation"],
                    valid_frame_mask,
                )
                loss = (
                    float(loss_weights["atom"]) * atom
                    + float(loss_weights["lddt"]) * lddt
                    + float(loss_weights["distance"]) * distances
                    + float(loss_weights["rotation"]) * rotations
                    + float(loss_weights["translation"]) * translations
                )
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"non-finite loss at step {step}")
                (loss / accumulation_steps).backward()
                for name, value in {
                    "loss": loss, "atom_loss": atom, "lddt_loss": lddt,
                    "distance_loss": distances, "rotation_loss": rotations,
                    "translation_loss": translations,
                }.items():
                    metric_sums[name] += value.detach().float() / accumulation_steps
            logical_examples.append({"logical_rank": logical_rank, "chain_id": chain_id, "epoch": epoch, "offset": offset, "crop_start": int(batch["crop_start"]), "timestep": int(timestep.item()), "example_seed": microbatch_seed})
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), float(training["gradient_clip_norm"])
        )
        if not torch.isfinite(gradient_norm):
            raise FloatingPointError(f"non-finite gradient norm at step {step}")
        optimizer.step()

        metrics = all_reduce_metrics(
            {**metric_sums, "gradient_norm": gradient_norm}
        )
        now = time.perf_counter()
        metrics["seconds_per_step"] = now - previous_time
        metrics["learning_rate"] = lr
        previous_time = now
        completed_step = step + 1
        metrics["samples_seen"] = completed_step * EFFECTIVE_BATCH_SIZE
        metrics["effective_batch_size"] = EFFECTIVE_BATCH_SIZE
        metrics["physical_world_size"] = world_size
        metrics["gradient_accumulation_steps"] = accumulation_steps

        if rank == 0 and (completed_step % log_every == 0 or completed_step <= 10):
            append_history(
                run_dir / "logs" / "history.jsonl",
                {
                    "event": "train",
                    "step": completed_step,
                    "epoch": epoch,
                    "offset": offset,
                    "chain_id_rank0": chain_id,
                    "logical_examples_rank0": logical_examples,
                    **execution_metadata(config, world_size, completed_step),
                    "crop_start_rank0": int(batch["crop_start"]),
                    "timestep_rank0": int(timestep.item()),
                    "fixed_example_seed_rank0": fixed_example_seed,
                    "metrics": metrics,
                    "time": time.time(),
                },
            )
            print(
                json.dumps(
                    {"step": completed_step, "metrics": metrics}, sort_keys=True
                ),
                flush=True,
            )
        if completed_step % heartbeat_every == 0 or completed_step <= 10:
            write_heartbeat(
                run_dir,
                state="training",
                step=completed_step,
                rank=rank,
                world_size=world_size,
                metrics=metrics,
                extra={
                    "parameter_count": parameter_count,
                    **execution_metadata(config, world_size, completed_step),
                    "dataset_tree_sha256": dataset_verification["inventory"][
                        "tree_sha256"
                    ],
                },
            )
        if completed_step % checkpoint_every == 0 or completed_step == total_steps:
            save_checkpoint(
                run_dir=run_dir,
                step=completed_step,
                model=model,
                optimizer=optimizer,
                config=config,
                dataset_verification=dataset_verification,
                rank=rank,
                world_size=world_size,
                milestone=completed_step % milestone_every == 0
                or completed_step == total_steps,
                milestone_every=milestone_every,
                keep_recent_non_milestones=keep_recent_checkpoints,
                resume_from=resume_from,
            )
            if dist.is_initialized():
                dist.barrier()

    distributed_rank_zero_action(
        rank=rank,
        description="training-complete state commit",
        action=lambda: mark_training_complete(
            run_dir=run_dir,
            step=total_steps,
            world_size=world_size,
            parameter_count=parameter_count,
            metrics=metrics,
            resume_from=resume_from,
            execution=execution_metadata(config, world_size, total_steps),
        ),
    )
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
