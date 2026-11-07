"""Run-state, checkpoint, and distributed helpers for architecture experiments."""

from __future__ import annotations

import hashlib
import inspect
import json
import os
import random
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist


def fsync_directory(path: str | Path) -> None:
    descriptor = os.open(Path(path), os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def distributed_context() -> tuple[int, int, int]:
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    return rank, local_rank, world_size


def initialize_distributed() -> tuple[int, int, int, torch.device]:
    rank, local_rank, world_size = distributed_context()
    if not torch.cuda.is_available():
        if world_size != 1:
            raise RuntimeError("multi-process training requires CUDA/NCCL")
        return rank, local_rank, world_size, torch.device("cpu")
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    if world_size > 1 and not dist.is_initialized():
        try:
            init_parameters = inspect.signature(dist.init_process_group).parameters
        except (TypeError, ValueError):
            init_parameters = {}
        init_kwargs: dict[str, Any] = {"backend": "nccl"}
        if "device_id" in init_parameters:
            # Passing the process-local device eagerly initializes NCCL on the
            # correct GPU and avoids process-group barrier device guessing.
            init_kwargs["device_id"] = device
        dist.init_process_group(**init_kwargs)
    return rank, local_rank, world_size, device


def seed_process(seed: int, rank: int) -> None:
    process_seed = int(
        hashlib.sha256(f"{seed}\0{rank}".encode("utf-8")).hexdigest()[:8], 16
    )
    random.seed(process_seed)
    np.random.seed(process_seed)
    torch.manual_seed(process_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(process_seed)


def capture_rng_state() -> dict[str, Any]:
    return {
        "rank": int(os.environ.get("RANK", "0")),
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        # One torchrun process owns one current CUDA device. Capturing every
        # visible device would make each rank initialize contexts on all GPUs.
        "torch_cuda": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
    }


def restore_rng_state(state: dict[str, Any]) -> None:
    expected_rank = int(os.environ.get("RANK", "0"))
    if type(state.get("rank")) is not int or int(state["rank"]) != expected_rank:
        raise ValueError(
            f"RNG state rank differs: {state.get('rank')!r} != {expected_rank}"
        )
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available() and state.get("torch_cuda") is not None:
        cuda_state = state["torch_cuda"]
        # Backward-compatible read for pre-schema-2 local smoke checkpoints.
        if isinstance(cuda_state, list):
            if not cuda_state:
                return
            local_rank = int(os.environ.get("LOCAL_RANK", "0"))
            torch.cuda.set_rng_state(cuda_state[local_rank])
        else:
            torch.cuda.set_rng_state(cuda_state)


def sha256_file(path: Path, chunk_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def verify_sha256_sidecar(path: str | Path) -> dict[str, Any]:
    """Verify an atomic checkpoint sidecar before any deserialization."""
    checkpoint_path = Path(path).resolve()
    sidecar_path = checkpoint_path.with_suffix(checkpoint_path.suffix + ".sha256.json")
    if not sidecar_path.is_file():
        raise FileNotFoundError(
            f"checkpoint checksum sidecar is missing: {sidecar_path}"
        )
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    observed = {
        "path": checkpoint_path.name,
        "bytes": checkpoint_path.stat().st_size,
        "sha256": sha256_file(checkpoint_path),
    }
    expected = {
        "path": sidecar.get("path"),
        "bytes": int(sidecar.get("bytes", -1)),
        "sha256": sidecar.get("sha256"),
    }
    if observed != expected:
        raise ValueError(
            "checkpoint checksum sidecar mismatch: "
            f"expected={expected}, observed={observed}"
        )
    return {
        **observed,
        "sidecar_path": str(sidecar_path),
        "sidecar_sha256": sha256_file(sidecar_path),
    }


def atomic_json(path: str | Path, payload: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, destination)
        fsync_directory(destination.parent)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def atomic_torch_save(path: str | Path, payload: Any) -> str:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        torch.save(payload, temporary_path)
        with temporary_path.open("rb") as handle:
            os.fsync(handle.fileno())
        digest = sha256_file(temporary_path)
        os.replace(temporary_path, destination)
        fsync_directory(destination.parent)
        atomic_json(
            destination.with_suffix(destination.suffix + ".sha256.json"),
            {
                "path": destination.name,
                "bytes": destination.stat().st_size,
                "sha256": digest,
            },
        )
        return digest
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def gather_rank_rng_states(local_state: dict[str, Any]) -> list[dict[str, Any]]:
    if not dist.is_initialized():
        return [local_state]
    gathered: list[dict[str, Any] | None] = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, local_state)
    return [state for state in gathered if state is not None]


def write_heartbeat(
    run_dir: str | Path,
    *,
    state: str,
    step: int,
    rank: int,
    world_size: int,
    metrics: dict[str, float] | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    if rank != 0:
        return
    payload: dict[str, Any] = {
        "schema_version": 1,
        "state": state,
        "step": int(step),
        "world_size": int(world_size),
        "unix_time": time.time(),
    }
    if metrics is not None:
        payload["metrics"] = {key: float(value) for key, value in metrics.items()}
    if extra:
        payload.update(extra)
    atomic_json(Path(run_dir) / "status.json", payload)
