#!/usr/bin/env python3
"""Public-validation evaluator for the NanoFold RFdiffusion-style model.

This evaluates the clean-room, sequence-conditioned RFdiffusion-style
reconstruction in this directory. It is not an evaluator for the official
RosettaCommons RFdiffusion model.

Prediction and scoring are deliberately separated. The model receives public
features and an amino-acid-chemistry atom mask; validation coordinates and
resolved-atom masks enter only the authoritative NanoFold scorer.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import platform
import socket
import subprocess
import sys
import tempfile
import time
import types
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.distributed as dist


PACKAGE_DIR = Path(__file__).resolve().parents[1]
THIS_DIR = PACKAGE_DIR.parents[1] / "architecture-reconstructions" / "rfdiffusion-style"
COMMON_DIR = THIS_DIR.parent / "common"
sys.path.insert(0, str(COMMON_DIR))
sys.path.insert(0, str(THIS_DIR))

from data import (  # noqa: E402
    STANDARD_ATOM14_MASK,
    NanoFoldTensorLoader,
    read_manifest,
)
from model import (  # noqa: E402
    DIFFUSION_CONTRACT_ID,
    IMPUTATION_CONTRACT_ID,
    MODEL_FEATURE_KEYS,
    RFStyleConfig,
    RFdiffusionStyleModel,
    complete_corruption_frames,
    corrupt_frames,
    terminal_translation_sigma,
)
from runtime import (  # noqa: E402
    atomic_json,
    initialize_distributed,
    sha256_file,
    verify_sha256_sidecar,
)
from verify_dataset import verify_dataset_snapshot  # noqa: E402


EXPECTED_COMPONENTS = (
    "foldscore",
    "gdt_ha_ca",
    "lddt_atom14",
    "cad_atom14",
    "sg_atom14",
    "sc_atom14",
    "molprobity_clash_atom14",
    "bb_atom14",
    "dipdiff_atom14",
    "gdt_ts_ca",
    "lddt_ca",
    "lddt_backbone_atom14",
)
METRIC_NAMES = (*EXPECTED_COMPONENTS, "rmsd_ca")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate an RFdiffusion-style checkpoint on NanoFold public "
            "validation with the exact repository FoldScore implementation."
        )
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--dataset-verification", type=Path, required=True)
    parser.add_argument(
        "--nanofold-source-root",
        type=Path,
        required=True,
        help=(
            "Read-only NanoFold competition source root containing "
            "nanofold/metrics.py and nanofold/residue_constants.py."
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--max-chains",
        type=int,
        default=None,
        help="Diagnostic prefix limit; omit for a claim-bearing evaluation.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Verify and reuse completed chain/sample artifacts.",
    )
    return parser.parse_args()


def _seed64(*parts: object) -> int:
    digest = hashlib.sha256(
        "\x1f".join(str(part) for part in parts).encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "little", signed=False) & ((1 << 63) - 1)


def sample_seed(global_seed: int, chain_id: str, sample_index: int) -> int:
    """Stable chain/sample seed independent of rank and world size."""
    return _seed64(
        "nanofold-rfdiffusion-style-eval",
        global_seed,
        chain_id,
        sample_index,
    )


def reverse_schedule(diffusion_steps: int, sampling_steps: int) -> tuple[int, ...]:
    """Return a strictly decreasing integer schedule ending at one."""
    if diffusion_steps < 1:
        raise ValueError("diffusion_steps must be positive")
    if not 1 <= sampling_steps <= diffusion_steps:
        raise ValueError("sampling_steps must be in [1, diffusion_steps]")
    if sampling_steps == 1:
        return (diffusion_steps,)
    values = np.linspace(diffusion_steps, 1, sampling_steps)
    schedule: list[int] = []
    for value in values:
        step = int(round(float(value)))
        if not schedule or step < schedule[-1]:
            schedule.append(step)
    if schedule[0] != diffusion_steps:
        schedule.insert(0, diffusion_steps)
    if schedule[-1] != 1:
        schedule.append(1)
    return tuple(schedule)


def _load_module(name: str, path: Path) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _git_revision(path: Path) -> dict[str, Any]:
    try:
        commit = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            [
                "git",
                "-C",
                str(path),
                "status",
                "--porcelain",
                "--",
                "nanofold/metrics.py",
                "nanofold/residue_constants.py",
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        return {"commit": commit, "metric_files_dirty": bool(status.strip())}
    except (FileNotFoundError, subprocess.CalledProcessError):
        return {"commit": None, "metric_files_dirty": None}


def load_official_nanofold_metrics(
    source_root: str | Path,
) -> tuple[types.ModuleType, types.ModuleType, dict[str, Any]]:
    """Load exact metric sources in an isolated package and hash them."""
    root = Path(source_root).resolve()
    package_root = root / "nanofold"
    metrics_path = package_root / "metrics.py"
    constants_path = package_root / "residue_constants.py"
    for path in (metrics_path, constants_path):
        if not path.is_file():
            raise FileNotFoundError(f"authoritative NanoFold source missing: {path}")

    package_name = f"_nanofold_official_{sha256_file(metrics_path)[:16]}"
    package = types.ModuleType(package_name)
    package.__path__ = [str(package_root)]  # type: ignore[attr-defined]
    package.__package__ = package_name
    sys.modules[package_name] = package
    constants = _load_module(f"{package_name}.residue_constants", constants_path)
    metrics = _load_module(f"{package_name}.metrics", metrics_path)
    components = tuple(metrics.FOLDSCORE_COMPONENT_NAMES)
    if components != EXPECTED_COMPONENTS:
        raise RuntimeError(
            "NanoFold metric contract differs from evaluator expectation: "
            f"{components!r}"
        )
    if not np.array_equal(
        np.asarray(constants.STANDARD_ATOM14_MASK, dtype=np.bool_),
        STANDARD_ATOM14_MASK,
    ):
        raise RuntimeError(
            "reconstruction chemistry mask differs from authoritative NanoFold "
            "STANDARD_ATOM14_MASK"
        )
    provenance = {
        "source_root": str(root),
        "metrics_path": str(metrics_path),
        "metrics_sha256": sha256_file(metrics_path),
        "residue_constants_path": str(constants_path),
        "residue_constants_sha256": sha256_file(constants_path),
        "foldscore_components": list(components),
        "chemistry_mask_equal": True,
        **_git_revision(root),
    }
    return metrics, constants, provenance


def make_model_batch(
    example: dict[str, Any], *, device: torch.device
) -> dict[str, torch.Tensor]:
    """Construct a sealed model batch with no target tensors."""
    batch: dict[str, torch.Tensor] = {}
    for key in MODEL_FEATURE_KEYS:
        value = example[key]
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"model feature {key} is not a tensor")
        batch[key] = value.unsqueeze(0).to(device, non_blocking=True)
    forbidden = {
        "atom14_positions",
        "atom14_positions_uncentered",
        "atom14_mask",
        "ca_coords",
        "ca_mask",
    }
    if forbidden.intersection(batch):
        raise AssertionError("target tensor leaked into model batch")
    return batch


def _randn(
    shape: Iterable[int],
    *,
    seed: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    return torch.randn(tuple(shape), generator=generator, device=device, dtype=dtype)


def _rand(
    shape: Iterable[int],
    *,
    seed: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    return torch.rand(tuple(shape), generator=generator, device=device, dtype=dtype)


def initial_noisy_backbone(
    batch: dict[str, torch.Tensor],
    *,
    seed: int,
    total_steps: int,
    translation_noise: torch.Tensor | None = None,
    rotation_quaternion: torch.Tensor | None = None,
) -> torch.Tensor:
    """Draw the exact terminal distribution of the shared forward process."""
    batch_size, length = batch["aatype"].shape
    device = batch["aatype"].device
    if translation_noise is None:
        translation_noise = _randn(
            (batch_size, length, 3),
            seed=_seed64(seed, "initial-translation"),
            device=device,
        )
    if rotation_quaternion is None:
        rotation_quaternion = _randn(
            (batch_size, length, 4),
            seed=_seed64(seed, "initial-rotation"),
            device=device,
        )
    identity = torch.eye(3, device=device).expand(batch_size, length, 3, 3)
    zero_translation = torch.zeros(batch_size, length, 3, device=device)
    support = batch["atom14_chemistry_mask"][:, :, :3].all(dim=-1)
    corrupted = corrupt_frames(
        identity,
        zero_translation,
        support,
        torch.full(
            (batch_size,),
            total_steps,
            device=device,
            dtype=torch.long,
        ),
        total_steps=total_steps,
        translation_noise=translation_noise,
        rotation_quaternion=rotation_quaternion,
    )
    return corrupted["noisy_backbone"]


def renoise_prediction(
    prediction: torch.Tensor,
    chemistry_mask: torch.Tensor,
    timestep: torch.Tensor,
    *,
    total_steps: int,
    seed: int,
) -> torch.Tensor:
    """Apply the trainer's forward corruption schedule with explicit RNG."""
    (
        _clean_rotation,
        _clean_translation,
        corruption_support,
        _clean_frame_geometry_mask,
        corruption_rotation,
        corruption_translation,
        _imputation_mask,
    ) = complete_corruption_frames(
        prediction,
        chemistry_mask,
        replacement_uniform=_rand(
            prediction.shape[:2],
            seed=_seed64(seed, "replacement"),
            device=prediction.device,
        ),
        imputation_translation_noise=_randn(
            prediction.shape[:2] + (3,),
            seed=_seed64(seed, "imputation-translation"),
            device=prediction.device,
        ),
        imputation_quaternion=_randn(
            prediction.shape[:2] + (4,),
            seed=_seed64(seed, "imputation-rotation"),
            device=prediction.device,
        ),
    )
    translation_noise = _randn(
        corruption_translation.shape,
        seed=_seed64(seed, "translation"),
        device=corruption_translation.device,
    )
    rotation_quaternion = _randn(
        (*corruption_translation.shape[:-1], 4),
        seed=_seed64(seed, "rotation"),
        device=corruption_translation.device,
    )
    return corrupt_frames(
        corruption_rotation,
        corruption_translation,
        corruption_support,
        timestep,
        total_steps=total_steps,
        translation_noise=translation_noise,
        rotation_quaternion=rotation_quaternion,
    )["noisy_backbone"]


@torch.inference_mode()
def sample_structure(
    model: RFdiffusionStyleModel,
    batch: dict[str, torch.Tensor],
    *,
    seed: int,
    diffusion_steps: int,
    sampling_steps: int,
    use_bf16: bool,
) -> torch.Tensor:
    """Run the declared annealed x0-denoise/re-noise sampler."""
    schedule = reverse_schedule(diffusion_steps, sampling_steps)
    current = initial_noisy_backbone(
        batch,
        seed=seed,
        total_steps=diffusion_steps,
    )
    prediction: torch.Tensor | None = None
    for index, step in enumerate(schedule):
        timestep = torch.full(
            (current.shape[0],),
            step,
            device=current.device,
            dtype=torch.long,
        )
        autocast = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if use_bf16 and current.device.type == "cuda"
            else nullcontext()
        )
        with autocast:
            prediction = model(batch, current, timestep)["pred_atom14"].float()
        if index + 1 < len(schedule):
            next_step = schedule[index + 1]
            current = renoise_prediction(
                prediction,
                batch["atom14_chemistry_mask"],
                torch.full_like(timestep, next_step),
                total_steps=diffusion_steps,
                seed=_seed64(seed, "renoise", step, next_step),
            )
    if prediction is None:
        raise AssertionError("empty sampling schedule")
    return prediction


def masked_kabsch_rmsd(
    pred: torch.Tensor, true: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    active = mask.to(device=pred.device, dtype=torch.bool)
    if int(active.sum().item()) < 3:
        return torch.full((), float("nan"), device=pred.device, dtype=pred.dtype)
    pred_active = pred[active]
    true_active = true.to(device=pred.device, dtype=pred.dtype)[active]
    pred_centered = pred_active - pred_active.mean(dim=0, keepdim=True)
    true_centered = true_active - true_active.mean(dim=0, keepdim=True)
    covariance = pred_centered.transpose(0, 1) @ true_centered
    u, _, vh = torch.linalg.svd(covariance, full_matrices=False)
    correction = torch.ones(3, device=pred.device, dtype=pred.dtype)
    if torch.det(u @ vh) < 0:
        correction[-1] = -1
    rotation = u @ torch.diag(correction) @ vh
    aligned = pred_centered @ rotation
    return torch.sqrt(
        (aligned - true_centered).square().sum(dim=-1).mean().clamp_min(0)
    )


def score_prediction(
    prediction: torch.Tensor,
    example: dict[str, Any],
    official_metrics: types.ModuleType,
) -> dict[str, float]:
    """Score one prediction only after the sealed model call."""
    crop_length = int(example["crop_length"])
    device = prediction.device
    pred = prediction[0, :crop_length].float()
    true = example["atom14_positions"][:crop_length].to(device).float()
    target_mask = example["atom14_mask"][:crop_length].to(device).bool()
    aatype = example["aatype"][:crop_length].to(device).long()
    components = official_metrics.foldscore_components(
        pred_atom14=pred,
        true_atom14=true,
        atom14_mask=target_mask,
        aatype=aatype,
    )
    values = {
        name: float(components[name].detach().float().cpu())
        for name in EXPECTED_COMPONENTS
    }
    values["rmsd_ca"] = float(
        masked_kabsch_rmsd(pred[:, 1], true[:, 1], target_mask[:, 1]).detach().cpu()
    )
    for name, value in values.items():
        if not math.isfinite(value):
            raise FloatingPointError(f"non-finite metric {name}: {value}")
    return values


def atomic_save_prediction(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def prediction_path(root: Path, chain_id: str, sample_index: int) -> Path:
    encoded = chain_id.encode("utf-8").hex()
    return root / f"sample-{sample_index:03d}" / f"chain_{encoded}.npz"


def read_completed_rows(output_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted((output_dir / "shards").glob("rank-*.jsonl")):
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"invalid resume shard {path}:{line_number}: {error}"
                ) from error
    return rows


def validate_resume_rows(
    rows: list[dict[str, Any]],
    *,
    output_dir: Path,
    checkpoint_step: int,
    checkpoint_sha256: str,
    num_samples: int,
) -> None:
    """Refuse stale, mixed-checkpoint, or corrupt prediction artifacts."""
    for row in rows:
        if row.get("architecture") != "RFdiffusion-style":
            raise ValueError("resume shard contains a different architecture")
        if bool(row.get("official_rfdiffusion", True)):
            raise ValueError("resume shard lacks RFdiffusion-style disclaimer")
        if int(row.get("checkpoint_step", -1)) != checkpoint_step:
            raise ValueError("resume shard checkpoint step differs")
        if row.get("checkpoint_sha256") != checkpoint_sha256:
            raise ValueError("resume shard checkpoint hash differs")
        sample_index = int(row.get("sample_index", -1))
        if not 0 <= sample_index < num_samples:
            raise ValueError(f"invalid resume sample index {sample_index}")
        relative_prediction = Path(str(row.get("prediction_path", "")))
        if relative_prediction.is_absolute() or ".." in relative_prediction.parts:
            raise ValueError(f"unsafe resume prediction path: {relative_prediction}")
        prediction_file = (output_dir / relative_prediction).resolve()
        prediction_root = (output_dir / "predictions").resolve()
        if prediction_root not in prediction_file.parents:
            raise ValueError("resume prediction is outside prediction root")
        if not prediction_file.is_file():
            raise FileNotFoundError(f"resume prediction is missing: {prediction_file}")
        if row.get("prediction_sha256") != sha256_file(prediction_file):
            raise ValueError(f"resume prediction hash mismatch: {relative_prediction}")


def validate_resume_provenance(
    existing: dict[str, Any], current: dict[str, Any]
) -> None:
    """Require identical scientific inputs before appending to a run."""
    file_names = (
        "checkpoint",
        "checkpoint_sidecar",
        "eval_config",
        "evaluate.py",
        "model.py",
        "common/data.py",
        "common/runtime.py",
        "common/verify_dataset.py",
        "validation_manifest",
        "dataset_verification",
    )
    comparisons = {
        name: (
            existing.get("source_files", {}).get(name, {}).get("sha256"),
            current.get("source_files", {}).get(name, {}).get("sha256"),
        )
        for name in file_names
    }
    comparisons.update(
        {
            "architecture": (
                existing.get("architecture"),
                current.get("architecture"),
            ),
            "official metrics": (
                existing.get("metric_source", {}).get("metrics_sha256"),
                current.get("metric_source", {}).get("metrics_sha256"),
            ),
            "official residue constants": (
                existing.get("metric_source", {}).get("residue_constants_sha256"),
                current.get("metric_source", {}).get("residue_constants_sha256"),
            ),
        }
    )
    mismatches = [
        name for name, (old, new) in comparisons.items() if old is None or old != new
    ]
    if mismatches:
        raise ValueError(
            "cannot resume because provenance differs for: " + ", ".join(mismatches)
        )


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def _mean_std(values: Iterable[float]) -> tuple[float, float]:
    array = np.asarray(list(values), dtype=np.float64)
    if not len(array):
        raise ValueError("cannot summarize empty values")
    return (
        float(array.mean()),
        float(array.std(ddof=1)) if len(array) > 1 else 0.0,
    )


def consolidate_rows(
    rows: list[dict[str, Any]],
    *,
    expected_chain_ids: list[str],
    num_samples: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Validate completeness and compute chain-first macro summaries."""
    ordered = sorted(rows, key=lambda row: (row["chain_id"], row["sample_index"]))
    expected = {
        (chain_id, sample_index)
        for chain_id in expected_chain_ids
        for sample_index in range(num_samples)
    }
    observed = [(str(row["chain_id"]), int(row["sample_index"])) for row in ordered]
    if len(observed) != len(set(observed)):
        raise ValueError("duplicate chain/sample records in evaluation shards")
    missing = sorted(expected - set(observed))
    extra = sorted(set(observed) - expected)
    if missing or extra:
        raise ValueError(
            f"incomplete evaluation: missing={missing[:5]} ({len(missing)}), "
            f"extra={extra[:5]} ({len(extra)})"
        )

    by_chain: dict[str, list[dict[str, Any]]] = {}
    for row in ordered:
        by_chain.setdefault(str(row["chain_id"]), []).append(row)
    chain_rows: list[dict[str, Any]] = []
    for chain_id in expected_chain_ids:
        samples = sorted(by_chain[chain_id], key=lambda row: row["sample_index"])
        chain_row: dict[str, Any] = {
            "chain_id": chain_id,
            "source_length": samples[0]["source_length"],
            "crop_start": samples[0]["crop_start"],
            "crop_length": samples[0]["crop_length"],
            "num_samples": len(samples),
        }
        for metric in METRIC_NAMES:
            mean, std = _mean_std(float(row[metric]) for row in samples)
            chain_row[f"{metric}_mean"] = mean
            chain_row[f"{metric}_sample_std"] = std
        chain_rows.append(chain_row)

    aggregate_rows: list[dict[str, Any]] = []
    for sample_index in range(num_samples):
        sample_rows = [
            row for row in ordered if int(row["sample_index"]) == sample_index
        ]
        aggregate: dict[str, Any] = {
            "scope": "sample",
            "sample_index": sample_index,
            "num_chains": len(sample_rows),
        }
        for metric in METRIC_NAMES:
            mean, std = _mean_std(float(row[metric]) for row in sample_rows)
            aggregate[f"{metric}_mean"] = mean
            aggregate[f"{metric}_chain_std"] = std
            aggregate[f"{metric}_chain_sem"] = std / math.sqrt(len(sample_rows))
        aggregate_rows.append(aggregate)

    macro: dict[str, Any] = {
        "scope": "all-sample-chain-macro",
        "sample_index": "all",
        "num_chains": len(chain_rows),
        "num_samples_per_chain": num_samples,
    }
    for metric in METRIC_NAMES:
        mean, std = _mean_std(float(row[f"{metric}_mean"]) for row in chain_rows)
        macro[f"{metric}_mean"] = mean
        macro[f"{metric}_chain_std"] = std
        macro[f"{metric}_chain_sem"] = std / math.sqrt(len(chain_rows))
    aggregate_rows.append(macro)
    return ordered, chain_rows, {"rows": aggregate_rows, "macro": macro}


def _source_provenance(
    *,
    args: argparse.Namespace,
    config_path: Path,
    checkpoint_path: Path,
    checkpoint_verification: dict[str, Any],
) -> dict[str, Any]:
    paths = {
        "evaluate.py": Path(__file__).resolve(),
        "model.py": THIS_DIR / "model.py",
        "common/data.py": COMMON_DIR / "data.py",
        "common/runtime.py": COMMON_DIR / "runtime.py",
        "common/verify_dataset.py": COMMON_DIR / "verify_dataset.py",
        "eval_config": config_path,
        "dataset_verification": args.dataset_verification.resolve(),
        "validation_manifest": args.data_root.resolve() / "manifests" / "val.txt",
    }
    provenance = {
        name: {
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for name, path in paths.items()
    }
    provenance["checkpoint"] = {
        "path": str(checkpoint_path),
        "bytes": int(checkpoint_verification["bytes"]),
        "sha256": str(checkpoint_verification["sha256"]),
    }
    sidecar_path = Path(str(checkpoint_verification["sidecar_path"]))
    provenance["checkpoint_sidecar"] = {
        "path": str(sidecar_path),
        "bytes": sidecar_path.stat().st_size,
        "sha256": str(checkpoint_verification["sidecar_sha256"]),
    }
    return provenance


def validate_checkpoint_training_sources(checkpoint: dict[str, Any]) -> None:
    """Require the checkpoint to have been trained with this model/data contract."""
    recorded = checkpoint.get("config", {}).get("_source_provenance")
    if not isinstance(recorded, dict):
        raise ValueError(
            "checkpoint lacks training source provenance; refusing evaluation"
        )
    current_paths = {
        "train_accumulated.py": PACKAGE_DIR / "runtime/train_accumulated.py",
        "parent_train.py": THIS_DIR / "train.py",
        "rf-contract.json": PACKAGE_DIR / "rf-contract.json",
        "model.py": THIS_DIR / "model.py",
        "common/data.py": COMMON_DIR / "data.py",
        "common/geometry.py": COMMON_DIR / "geometry.py",
        "common/runtime.py": COMMON_DIR / "runtime.py",
        "common/verify_dataset.py": COMMON_DIR / "verify_dataset.py",
    }
    for name, path in current_paths.items():
        recorded_hash = recorded.get(name, {}).get("sha256")
        current_hash = sha256_file(path)
        if recorded_hash != current_hash:
            raise ValueError(
                f"checkpoint training source mismatch for {name}: "
                f"recorded={recorded_hash}, current={current_hash}"
            )


    from train_accumulated import execution_metadata
    expected_execution = execution_metadata(checkpoint["config"], int(checkpoint["world_size"]), int(checkpoint["step"]))
    for name, expected in expected_execution.items():
        if checkpoint.get(name) != expected:
            raise ValueError(f"checkpoint accumulation metadata mismatch: {name}")


def verify_evaluation_inputs(
    *,
    rank: int,
    data_root: Path,
    dataset_verification: dict[str, Any],
    checkpoint_path: Path,
) -> dict[str, Any]:
    """Rehash dataset/checkpoint inputs on rank zero and broadcast the result."""
    result: list[dict[str, Any] | None] = [None]
    if rank == 0:
        try:
            observed_dataset = verify_dataset_snapshot(data_root, dataset_verification)
            checkpoint_verification = verify_sha256_sidecar(checkpoint_path)
            result[0] = {
                "ok": True,
                "dataset": observed_dataset,
                "checkpoint": checkpoint_verification,
            }
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
        raise RuntimeError(f"evaluation input verification failed: {outcome}")
    return {
        "checkpoint": dict(outcome["checkpoint"]),
        "dataset": dict(outcome["dataset"]),
    }


def _validate_contract(
    *,
    config: dict[str, Any],
    checkpoint: dict[str, Any],
    dataset_verification: dict[str, Any],
    manifest_path: Path,
    actual_manifest_count: int,
) -> None:
    if checkpoint.get("architecture") != "nanofold-rfdiffusion-style":
        raise ValueError("checkpoint is not NanoFold RFdiffusion-style")
    validate_checkpoint_training_sources(checkpoint)
    expected_scientific_contract = {
        "diffusion": DIFFUSION_CONTRACT_ID,
        "imputation": IMPUTATION_CONTRACT_ID,
    }
    if (
        checkpoint.get("config", {}).get("_scientific_contract")
        != expected_scientific_contract
    ):
        raise ValueError(
            "checkpoint scientific contract differs from evaluator: "
            f"expected={expected_scientific_contract}, "
            f"recorded={checkpoint.get('config', {}).get('_scientific_contract')}"
        )
    current_tree = dataset_verification["inventory"]["tree_sha256"]
    checkpoint_tree = checkpoint["dataset_verification"]["inventory"]["tree_sha256"]
    if current_tree != checkpoint_tree:
        raise ValueError(
            f"dataset tree mismatch: current={current_tree}, "
            f"checkpoint={checkpoint_tree}"
        )
    actual_hash = sha256_file(manifest_path)
    expected_hash = dataset_verification["manifest_sha256"]["validation"]
    if actual_hash != expected_hash:
        raise ValueError(
            f"validation manifest hash mismatch: expected {expected_hash}, "
            f"got {actual_hash}"
        )
    expected_count = int(dataset_verification["validation_count"])
    if actual_manifest_count != expected_count:
        raise ValueError(
            f"validation count mismatch: expected {expected_count}, "
            f"got {actual_manifest_count}"
        )
    checkpoint_diffusion = int(checkpoint["config"]["training"]["diffusion_steps"])
    evaluation_diffusion = int(config["sampling"]["diffusion_steps"])
    if checkpoint_diffusion != evaluation_diffusion:
        raise ValueError(
            f"diffusion step mismatch: checkpoint={checkpoint_diffusion}, "
            f"evaluation={evaluation_diffusion}"
        )
    derived_prior_sigma = terminal_translation_sigma(total_steps=evaluation_diffusion)
    declared_prior_sigma = config["sampling"].get("initial_translation_sigma")
    if declared_prior_sigma is not None and not math.isclose(
        float(declared_prior_sigma),
        derived_prior_sigma,
        rel_tol=0.0,
        abs_tol=1e-8,
    ):
        raise ValueError(
            "initial translation prior must equal the training terminal "
            f"sigma: declared={declared_prior_sigma}, "
            f"derived={derived_prior_sigma}"
        )
    checkpoint_data = checkpoint["config"]["data"]
    for name in ("crop_size", "msa_depth"):
        if int(config["data"][name]) != int(checkpoint_data[name]):
            raise ValueError(
                f"{name} mismatch: checkpoint={checkpoint_data[name]}, "
                f"evaluation={config['data'][name]}"
            )


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    checkpoint_path = args.checkpoint.resolve()
    data_root = args.data_root.resolve()
    output_dir = args.output_dir.resolve()
    if args.max_chains is not None and args.max_chains < 1:
        raise ValueError("--max-chains must be positive")
    if output_dir.exists() and not args.resume:
        if list((output_dir / "shards").glob("rank-*.jsonl")):
            raise FileExistsError(
                f"{output_dir} has shards; pass --resume or use a new output"
            )
    output_dir.mkdir(parents=True, exist_ok=True)

    config = json.loads(config_path.read_text(encoding="utf-8"))
    dataset_verification = json.loads(
        args.dataset_verification.read_text(encoding="utf-8")
    )
    rank, _local_rank, world_size, device = initialize_distributed()
    configured_world_size = int(config["runtime"]["world_size"])
    if world_size != configured_world_size:
        raise RuntimeError(
            f"config requires world_size={configured_world_size}, got {world_size}"
        )
    input_verification = verify_evaluation_inputs(
        rank=rank,
        data_root=data_root,
        dataset_verification=dataset_verification,
        checkpoint_path=checkpoint_path,
    )
    checkpoint_verification = input_verification["checkpoint"]
    manifest_path = data_root / "manifests" / "val.txt"
    full_chain_ids = read_manifest(manifest_path)
    _checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    _validate_contract(
        config=config,
        checkpoint=_checkpoint,
        dataset_verification=dataset_verification,
        manifest_path=manifest_path,
        actual_manifest_count=len(full_chain_ids),
    )
    chain_ids = (
        full_chain_ids if args.max_chains is None else full_chain_ids[: args.max_chains]
    )
    claim_bearing = (
        args.max_chains is None
        and len(full_chain_ids) == 1000
        and sha256_file(manifest_path) == "d4a0265bcd0a021e116c0c889f21e86bc24006460bcc42dec2f9a80b70c8812b"
        and int(_checkpoint.get("step", -1)) == 30000
        and int(_checkpoint["config"]["training"]["steps"]) == 30000
        and int(_checkpoint["config"]["training"]["seed"]) == 0
        and int(_checkpoint.get("samples_seen", -1)) == 240000
        and config["sampling"] == {"global_seed": 1729, "feature_seed": 0, "num_samples": 5, "diffusion_steps": 50, "sampling_steps": 50}
    )

    metrics, _constants, metric_provenance = load_official_nanofold_metrics(
        args.nanofold_source_root
    )
    model = RFdiffusionStyleModel(RFStyleConfig(**_checkpoint["config"]["model"])).to(
        device
    )
    model.load_state_dict(_checkpoint["model"], strict=True)
    model.eval()
    parameter_count = sum(parameter.numel() for parameter in model.parameters())

    loader = NanoFoldTensorLoader(
        data_root=data_root,
        crop_size=int(config["data"]["crop_size"]),
        msa_depth=int(config["data"]["msa_depth"]),
        seed=int(config["sampling"]["feature_seed"]),
    )
    num_samples = int(config["sampling"]["num_samples"])
    if num_samples < 1:
        raise ValueError("num_samples must be positive")
    diffusion_steps = int(config["sampling"]["diffusion_steps"])
    sampling_steps = int(config["sampling"]["sampling_steps"])
    prior_sigma = terminal_translation_sigma(total_steps=diffusion_steps)
    use_bf16 = bool(config["runtime"]["bf16"])
    source_provenance = _source_provenance(
        args=args,
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        checkpoint_verification=checkpoint_verification,
    )
    provenance = {
        "schema_version": 1,
        "architecture": "RFdiffusion-style NanoFold reconstruction",
        "official_rfdiffusion": False,
        "claim_bearing_full_public_validation": claim_bearing,
        "checkpoint_step": int(_checkpoint["step"]),
        "parameter_count": parameter_count,
        "world_size": world_size,
        "config": config,
        "dataset_verification": dataset_verification,
        "observed_dataset_verification": input_verification["dataset"],
        "scientific_contract": _checkpoint["config"]["_scientific_contract"],
        "metric_source": metric_provenance,
        "source_files": source_provenance,
        "environment": {
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "python": sys.version,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": str(device),
        },
        "sampler": {
            "name": "annealed-x0-denoise-renoise",
            "official_rfdiffusion_reverse_sde": False,
            "schedule": list(reverse_schedule(diffusion_steps, sampling_steps)),
            "initial_translation_sigma": prior_sigma,
            "initial_translation_sigma_source": (
                "derived from the checkpoint training terminal timestep"
            ),
            "seed_derivation": (
                "sha256('nanofold-rfdiffusion-style-eval', global_seed, "
                "chain_id, sample_index), low 63 bits"
            ),
        },
    }

    completed_rows = read_completed_rows(output_dir) if args.resume else []
    if args.resume:
        existing_provenance_path = output_dir / "provenance.json"
        if completed_rows and not existing_provenance_path.is_file():
            raise FileNotFoundError("resume shards exist without provenance")
        if existing_provenance_path.is_file():
            validate_resume_provenance(
                json.loads(existing_provenance_path.read_text(encoding="utf-8")),
                provenance,
            )
        validate_resume_rows(
            completed_rows,
            output_dir=output_dir,
            checkpoint_step=int(_checkpoint["step"]),
            checkpoint_sha256=source_provenance["checkpoint"]["sha256"],
            num_samples=num_samples,
        )
    completed = {
        (str(row["chain_id"]), int(row["sample_index"])) for row in completed_rows
    }
    if rank == 0:
        atomic_json(output_dir / "provenance.json", provenance)
    if dist.is_initialized():
        dist.barrier()

    rank_chain_ids = [
        chain_id
        for index, chain_id in enumerate(chain_ids)
        if index % world_size == rank
    ]
    shard_path = output_dir / "shards" / f"rank-{rank:03d}.jsonl"
    shard_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    processed_chains = 0
    with shard_path.open("a" if args.resume else "w", encoding="utf-8") as shard:
        for chain_id in rank_chain_ids:
            example = loader.load(chain_id, train=False, step=0, rank=0)
            model_batch = make_model_batch(example, device=device)
            for sample_index in range(num_samples):
                if (chain_id, sample_index) in completed:
                    continue
                seed = sample_seed(
                    int(config["sampling"]["global_seed"]),
                    chain_id,
                    sample_index,
                )
                sample_started = time.perf_counter()
                prediction = sample_structure(
                    model,
                    model_batch,
                    seed=seed,
                    diffusion_steps=diffusion_steps,
                    sampling_steps=sampling_steps,
                    use_bf16=use_bf16,
                )
                metric_values = score_prediction(prediction, example, metrics)
                crop_length = int(example["crop_length"])
                prediction_file = prediction_path(
                    output_dir / "predictions", chain_id, sample_index
                )
                atomic_save_prediction(
                    prediction_file,
                    pred_atom14=prediction[0, :crop_length]
                    .detach()
                    .float()
                    .cpu()
                    .numpy()
                    .astype(np.float32),
                    atom14_chemistry_mask=model_batch["atom14_chemistry_mask"][
                        0, :crop_length
                    ]
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.bool_),
                    chain_id=np.asarray(chain_id),
                    sample_index=np.asarray(sample_index, dtype=np.int32),
                    sample_seed=np.asarray(seed, dtype=np.int64),
                    checkpoint_sha256=np.asarray(
                        source_provenance["checkpoint"]["sha256"]
                    ),
                    checkpoint_step=np.asarray(
                        int(_checkpoint["step"]), dtype=np.int64
                    ),
                    crop_start=np.asarray(int(example["crop_start"]), dtype=np.int32),
                    crop_length=np.asarray(crop_length, dtype=np.int32),
                )
                row = {
                    "architecture": "RFdiffusion-style",
                    "official_rfdiffusion": False,
                    "checkpoint_step": int(_checkpoint["step"]),
                    "checkpoint_sha256": source_provenance["checkpoint"]["sha256"],
                    "chain_id": chain_id,
                    "source_length": int(example["source_length"]),
                    "crop_start": int(example["crop_start"]),
                    "crop_length": crop_length,
                    "sample_index": sample_index,
                    "sample_seed": seed,
                    "sampling_seconds": time.perf_counter() - sample_started,
                    "prediction_path": str(prediction_file.relative_to(output_dir)),
                    "prediction_sha256": sha256_file(prediction_file),
                    **metric_values,
                }
                shard.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
            shard.flush()
            os.fsync(shard.fileno())
            processed_chains += 1
            atomic_json(
                output_dir / "progress" / f"rank-{rank:03d}.json",
                {
                    "rank": rank,
                    "world_size": world_size,
                    "state": "evaluating",
                    "processed_chains_this_invocation": processed_chains,
                    "assigned_chains": len(rank_chain_ids),
                    "last_chain_id": chain_id,
                    "elapsed_seconds": time.perf_counter() - started,
                },
            )

    atomic_json(
        output_dir / "progress" / f"rank-{rank:03d}.json",
        {
            "rank": rank,
            "world_size": world_size,
            "state": "complete",
            "processed_chains_this_invocation": processed_chains,
            "assigned_chains": len(rank_chain_ids),
            "elapsed_seconds": time.perf_counter() - started,
        },
    )
    if dist.is_initialized():
        dist.barrier()

    if rank == 0:
        sample_rows, chain_rows, aggregate = consolidate_rows(
            read_completed_rows(output_dir),
            expected_chain_ids=chain_ids,
            num_samples=num_samples,
        )
        sample_fields = [
            "architecture",
            "official_rfdiffusion",
            "checkpoint_step",
            "checkpoint_sha256",
            "chain_id",
            "source_length",
            "crop_start",
            "crop_length",
            "sample_index",
            "sample_seed",
            "sampling_seconds",
            "prediction_path",
            "prediction_sha256",
            *METRIC_NAMES,
        ]
        chain_fields = [
            "chain_id",
            "source_length",
            "crop_start",
            "crop_length",
            "num_samples",
            *[
                field
                for metric in METRIC_NAMES
                for field in (
                    f"{metric}_mean",
                    f"{metric}_sample_std",
                )
            ],
        ]
        aggregate_fields = [
            "scope",
            "sample_index",
            "num_chains",
            "num_samples_per_chain",
            *[
                field
                for metric in METRIC_NAMES
                for field in (
                    f"{metric}_mean",
                    f"{metric}_chain_std",
                    f"{metric}_chain_sem",
                )
            ],
        ]
        write_csv(
            output_dir / "per-sample-metrics.csv",
            sample_rows,
            sample_fields,
        )
        write_csv(
            output_dir / "per-chain-metrics.csv",
            chain_rows,
            chain_fields,
        )
        write_csv(
            output_dir / "aggregate-metrics.csv",
            aggregate["rows"],
            aggregate_fields,
        )
        summary = {
            "schema_version": 1,
            "architecture": "RFdiffusion-style NanoFold reconstruction",
            "official_rfdiffusion": False,
            "claim_bearing_full_public_validation": claim_bearing,
            "checkpoint_step": int(_checkpoint["step"]),
            "checkpoint_sha256": source_provenance["checkpoint"]["sha256"],
            "num_chains": len(chain_ids),
            "num_samples_per_chain": num_samples,
            "world_size": world_size,
            "macro": aggregate["macro"],
            "output_files": {
                "per_sample_metrics": "per-sample-metrics.csv",
                "per_chain_metrics": "per-chain-metrics.csv",
                "aggregate_metrics": "aggregate-metrics.csv",
                "provenance": "provenance.json",
                "prediction_root": "predictions",
            },
            "elapsed_seconds_rank0": time.perf_counter() - started,
        }
        atomic_json(output_dir / "summary.json", summary)
        print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
