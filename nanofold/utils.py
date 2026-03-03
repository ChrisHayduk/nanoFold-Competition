from __future__ import annotations

import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # NOTE: deterministic can hurt speed; enable if you want strict bitwise runs.
    # torch.use_deterministic_algorithms(True)


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
