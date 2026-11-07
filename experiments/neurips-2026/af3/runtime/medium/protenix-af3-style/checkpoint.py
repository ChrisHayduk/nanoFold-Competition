"""Atomic, checksummed, capacity-bounded checkpoints for the Protenix arm."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

import torch


def sha256_file(path: Path, chunk_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def allocated_bytes(path: Path) -> int:
    """Filesystem-allocated bytes, falling back to logical size."""

    stat = path.stat()
    return int(getattr(stat, "st_blocks", 0) * 512 or stat.st_size)


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise


class BoundedCheckpointManager:
    """Keep every milestone and only the two newest intervening checkpoints."""

    def __init__(
        self,
        root: str | Path,
        *,
        milestone_interval: int = 3_000,
        keep_recent: int = 2,
        expected_allocated_bytes: int = 6 << 30,
        budget_bytes: int = 100 << 30,
        reserve_bytes: int = 20 << 30,
        budget_root: str | Path | None = None,
        allow_initial_checkpoint: bool = False,
    ) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.budget_root = (
            Path(budget_root) if budget_root is not None else self.root.parent
        )
        self.milestone_interval = int(milestone_interval)
        self.keep_recent = int(keep_recent)
        self.expected_allocated_bytes = int(expected_allocated_bytes)
        self.budget_bytes = int(budget_bytes)
        self.reserve_bytes = int(reserve_bytes)
        self.allow_initial_checkpoint = bool(allow_initial_checkpoint)
        if self.milestone_interval < 1 or self.keep_recent < 0:
            raise ValueError("invalid retention configuration")

    def path_for(self, completed_step: int) -> Path:
        if completed_step < 0:
            raise ValueError("completed_step must be non-negative")
        if completed_step == 0 and not self.allow_initial_checkpoint:
            raise ValueError(
                "completed-step zero requires allow_initial_checkpoint=True"
            )
        return self.root / f"completed-step-{completed_step:07d}.pt"

    def allocated_under_budget_root(self) -> int:
        total = 0
        if not self.budget_root.exists():
            return total
        for path in self.budget_root.rglob("*"):
            if path.is_file():
                total += allocated_bytes(path)
        return total

    def _capacity_preflight(self) -> dict[str, int]:
        allocated = self.allocated_under_budget_root()
        projected = allocated + self.expected_allocated_bytes + self.reserve_bytes
        filesystem_free = shutil.disk_usage(self.root).free  # informational only
        if projected > self.budget_bytes:
            raise OSError(
                f"checkpoint budget preflight failed: {allocated:,} allocated + "
                f"{self.expected_allocated_bytes:,} expected + "
                f"{self.reserve_bytes:,} reserve > {self.budget_bytes:,} budget"
            )
        return {
            "budget_bytes": self.budget_bytes,
            "allocated_before_save": allocated,
            "expected_next_allocated_bytes": self.expected_allocated_bytes,
            "reserve_bytes": self.reserve_bytes,
            "filesystem_free_bytes_informational": filesystem_free,
        }

    def save(self, completed_step: int, payload: dict[str, Any]) -> dict[str, Any]:
        capacity = self._capacity_preflight()
        destination = self.path_for(completed_step)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{destination.name}.", suffix=".tmp", dir=self.root
        )
        os.close(descriptor)
        temporary = Path(temporary_name)
        try:
            torch.save(payload, temporary)
            with temporary.open("rb") as handle:
                os.fsync(handle.fileno())
            digest = sha256_file(temporary)
            logical_bytes = temporary.stat().st_size
            allocated = allocated_bytes(temporary)
            os.replace(temporary, destination)
            record = {
                "schema_version": 1,
                "completed_step": int(completed_step),
                "path": destination.name,
                "logical_bytes": logical_bytes,
                "allocated_bytes": allocated,
                "sha256": digest,
                "capacity": capacity,
            }
            _atomic_json(destination.with_suffix(".pt.sha256.json"), record)
            # latest.json is the only latest pointer: no duplicate checkpoint.
            _atomic_json(self.root / "latest.json", record)
            self.prune()
            return record
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise

    def latest_path(self) -> Path | None:
        pointer = self.root / "latest.json"
        if not pointer.is_file():
            return None
        record = json.loads(pointer.read_text(encoding="utf-8"))
        candidate = self.root / record["path"]
        if not candidate.is_file():
            raise FileNotFoundError(f"latest pointer target missing: {candidate}")
        sidecar_path = candidate.with_suffix(".pt.sha256.json")
        if not sidecar_path.is_file():
            raise FileNotFoundError(f"checkpoint checksum sidecar missing: {sidecar_path}")
        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
        for key in ("path", "completed_step", "sha256"):
            if sidecar.get(key) != record.get(key):
                raise ValueError(
                    f"latest pointer and checksum sidecar disagree on {key}"
                )
        actual = sha256_file(candidate)
        if actual != record["sha256"]:
            raise ValueError(
                f"checkpoint checksum mismatch for {candidate}: "
                f"{actual} != {record['sha256']}"
            )
        return candidate

    def load_latest(self, *, map_location: Any = "cpu") -> dict[str, Any] | None:
        path = self.latest_path()
        if path is None:
            return None
        return torch.load(path, map_location=map_location, weights_only=False)

    def prune(self) -> list[str]:
        candidates: list[tuple[int, Path]] = []
        for path in self.root.glob("completed-step-*.pt"):
            try:
                step = int(path.stem.rsplit("-", 1)[1])
            except ValueError:
                continue
            candidates.append((step, path))
        candidates.sort()
        # Step zero is an explicitly requested diagnostic initial state. Treat
        # it as a milestone so overfit start/end comparisons cannot prune it.
        milestones = {
            step for step, _ in candidates if step % self.milestone_interval == 0
        }
        non_milestones = [
            step for step, _ in candidates if step not in milestones
        ]
        keep = milestones | set(non_milestones[-self.keep_recent :])
        removed: list[str] = []
        for step, path in candidates:
            if step in keep:
                continue
            path.unlink()
            path.with_suffix(".pt.sha256.json").unlink(missing_ok=True)
            removed.append(path.name)
        return removed
