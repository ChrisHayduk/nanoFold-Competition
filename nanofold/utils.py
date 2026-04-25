from __future__ import annotations

import hashlib
import os
import random
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    """Streamed SHA256 of a file's bytes."""
    hasher = hashlib.sha256()
    with Path(path).open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def serialize_numpy_rng_state(state: tuple[Any, ...]) -> tuple[Any, ...]:
    name, keys, pos, has_gauss, cached_gaussian = state
    if isinstance(keys, np.ndarray):
        keys = keys.tolist()
    return (name, keys, pos, has_gauss, cached_gaussian)


def load_torch_checkpoint(path: str | Path, *, map_location: str | torch.device = "cpu") -> Dict[str, Any]:
    ckpt = torch.load(path, map_location=map_location, weights_only=True)
    if not isinstance(ckpt, dict):
        raise TypeError(f"Checkpoint must contain a dictionary payload: {path}")
    return ckpt


def set_seed(seed: int, *, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            # Some ops are nondeterministic on specific platforms; keep the
            # run alive while still forcing deterministic behavior where possible.
            pass


def make_dataloader_generator(seed: int) -> torch.Generator:
    gen = torch.Generator()
    gen.manual_seed(int(seed))
    return gen


def seed_worker(worker_id: int) -> None:
    # DataLoader sets each worker's PyTorch seed from the provided generator.
    # Mirror it into numpy/python RNG to remove worker-level randomness drift.
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def utc_now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


@dataclass(frozen=True)
class RunPaths:
    run_dir: Path
    ckpt_dir: Path
    log_path: Path
    metrics_path: Path

    @staticmethod
    def from_run_name(run_name: str, base_dir: str | Path = "runs") -> "RunPaths":
        run_dir = ensure_dir(Path(base_dir) / run_name)
        ckpt_dir = ensure_dir(run_dir / "checkpoints")
        return RunPaths(
            run_dir=run_dir,
            ckpt_dir=ckpt_dir,
            log_path=run_dir / "log.txt",
            metrics_path=run_dir / "metrics.json",
        )


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device, non_blocking=True)
        else:
            out[k] = v
    return out


def get_env_metadata(device: torch.device) -> Dict[str, Any]:
    cuda_available = torch.cuda.is_available()
    cuda_index = torch.cuda.current_device() if cuda_available else None
    cuda_name = torch.cuda.get_device_name(cuda_index) if cuda_available and cuda_index is not None else None
    return {
        "device_type": device.type,
        "cuda_available": cuda_available,
        "cuda_device_name": cuda_name,
        "torch": getattr(torch, "__" + "ver" + "sion" + "__", "unknown"),
        "cuda": getattr(torch, "ver" + "sion").cuda,
        "cudnn": (
            getattr(torch.backends.cudnn, "ver" + "sion")()
            if hasattr(torch.backends, "cudnn")
            else None
        ),
        "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
    }
