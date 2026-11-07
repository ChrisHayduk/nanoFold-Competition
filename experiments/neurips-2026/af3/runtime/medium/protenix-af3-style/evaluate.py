#!/usr/bin/env python3
"""Fail-closed public-validation evaluation for the Protenix AF3-style arm.

Prediction and scoring are deliberately separate.  A model invocation receives
only the public feature NPZ through ``load_target_free_example`` and
``nanofold_inputs_to_protenix``.  The prediction is atomically persisted before
the validation label NPZ is opened by the scorer.

This evaluates the modified NanoFold Protenix reconstruction in this directory.
It is not DeepMind AlphaFold 3 and does not use released AlphaFold 3 weights.
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


HERE = Path(__file__).resolve().parent
ARCH_ROOT = HERE.parent
COMMON_DIR = ARCH_ROOT / "common"
sys.path.insert(0, str(COMMON_DIR))
sys.path.insert(0, str(HERE))

from adapter import (  # noqa: E402
    ccd_reference_sha256,
    load_protenix_ccd_reference,
    nanofold_inputs_to_protenix,
    prediction_to_nanofold,
)
from checkpoint import sha256_file  # noqa: E402
from config import PROTENIX_COMMIT, build_protenix_config  # noqa: E402
from data import (  # noqa: E402
    STANDARD_ATOM14_MASK,
    DeterministicBatchSchedule,
    chain_npz_path,
    read_manifest,
)
from predict_one import load_target_free_example, seed_inference  # noqa: E402
from runtime import atomic_json, initialize_distributed  # noqa: E402
from effective_batch import accumulation_contract  # noqa: E402
from train import (  # noqa: E402
    build_training_schedule_contract,
    canonical_json_sha256,
    to_device,
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
ARCHITECTURE_ID = "Protenix-AF3-style"
AGGREGATION_POLICY_ID = "sample-mean-then-equal-chain-macro-v1"
AGGREGATION_POLICY = {
    "id": AGGREGATION_POLICY_ID,
    "primary": (
        "For each chain, take the arithmetic mean of every predeclared "
        "(model_seed_index, diffusion_sample_index) metric; then take the "
        "equal-weight arithmetic mean across validation chains."
    ),
    "sample_selection": "none",
    "best_of_samples": False,
    "confidence_selection": False,
    "target_metric_selection": False,
    "secondary": (
        "Fixed sample-index macro means and within-chain sample standard "
        "deviations are descriptive diagnostics only."
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a random-init Protenix AF3-style checkpoint on the "
            "NanoFold public-validation split."
        )
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--training-provenance", type=Path, required=True)
    parser.add_argument("--protenix-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--dataset-verification", type=Path, required=True)
    parser.add_argument("--nanofold-source-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument(
        "--max-chains",
        type=int,
        default=None,
        help=(
            "Evaluate only the manifest prefix as a non-claim-bearing "
            "diagnostic. Omit for all 1,000 public-validation chains."
        ),
    )
    selection.add_argument(
        "--overfit-train-schedule-step0",
        action="store_true",
        help=(
            "Non-claim-bearing diagnostic on exactly the eight chains assigned "
            "at optimizer step zero by the recorded training schedule."
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Verify and reuse atomic per-chain/sample records.",
    )
    return parser.parse_args()


def _seed64(*parts: object) -> int:
    digest = hashlib.sha256(
        "\x1f".join(str(part) for part in parts).encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "little", signed=False) & ((1 << 63) - 1)


def sample_seed(
    global_seed: int,
    chain_id: str,
    model_seed_index: int,
    diffusion_sample_index: int,
) -> int:
    """Stable seed with no rank or world-size dependency."""

    if model_seed_index < 0 or diffusion_sample_index < 0:
        raise ValueError("sample indices must be non-negative")
    return _seed64(
        "nanofold-protenix-af3-style-eval-v1",
        global_seed,
        chain_id,
        model_seed_index,
        diffusion_sample_index,
    )


def flat_sample_index(
    model_seed_index: int,
    diffusion_sample_index: int,
    *,
    diffusion_samples: int,
) -> int:
    if model_seed_index < 0:
        raise ValueError("model_seed_index must be non-negative")
    if not 0 <= diffusion_sample_index < diffusion_samples:
        raise ValueError("diffusion_sample_index outside declared sample count")
    return model_seed_index * diffusion_samples + diffusion_sample_index


def partition_chain_ids(
    chain_ids: list[str], *, rank: int, world_size: int
) -> list[str]:
    """Manifest-order modulo partition; every chain belongs to exactly one rank."""

    if world_size < 1 or not 0 <= rank < world_size:
        raise ValueError("invalid distributed rank/world size")
    return [
        chain_id
        for manifest_index, chain_id in enumerate(chain_ids)
        if manifest_index % world_size == rank
    ]


def is_claim_bearing_evaluation(
    *,
    full_manifest_count: int,
    selected_count: int,
    max_chains: int | None,
    world_size: int,
) -> bool:
    """Only the complete official 1,000-chain, eight-rank run can bear claims."""

    return (
        max_chains is None
        and full_manifest_count == 1_000
        and selected_count == 1_000
        and world_size in (1, 2, 4, 8)
    )


def step_zero_training_schedule(
    training_chain_ids: tuple[str, ...], *, seed: int, world_size: int = 8
) -> list[dict[str, Any]]:
    """Return the exact chains assigned to ranks at optimizer step zero."""

    schedule = DeterministicBatchSchedule(
        training_chain_ids,
        seed=seed,
        world_size=world_size,
    )
    selected: list[dict[str, Any]] = []
    for rank in range(world_size):
        chain_id, epoch, offset = schedule.chain_at(0, rank)
        selected.append(
            {
                "rank": rank,
                "chain_id": chain_id,
                "epoch": epoch,
                "offset": offset,
            }
        )
    if len({item["chain_id"] for item in selected}) != world_size:
        raise ValueError("step-zero training schedule did not yield distinct chains")
    return selected


def overfit_step_zero_training_schedule(
    full_training_chain_ids: tuple[str, ...],
    *,
    training_provenance: dict[str, Any],
    world_size: int = 8,
) -> list[dict[str, Any]]:
    """Reconstruct and verify the diagnostic schedule before selecting chains."""

    if training_provenance.get("save_initial_checkpoint") is not True:
        raise ValueError(
            "overfit evaluation requires an explicitly saved initial checkpoint"
        )
    recorded = training_provenance.get("schedule")
    if not isinstance(recorded, dict):
        raise ValueError("overfit training provenance lacks a schedule contract")
    if recorded.get("selection") != "verified_manifest_prefix_diagnostic":
        raise ValueError("overfit evaluation requires a diagnostic-prefix schedule")
    if int(recorded.get("train_chain_limit", -1)) != world_size:
        raise ValueError(
            "overfit schedule must contain exactly one training chain per rank"
        )
    scheduled_chain_ids, expected = build_training_schedule_contract(
        full_training_chain_ids,
        train_chain_limit=world_size,
        world_size=world_size,
        max_steps=int(training_provenance.get("max_steps", -1)),
        smoke_steps=0,
    )
    if recorded != expected:
        raise ValueError(
            "recorded overfit schedule does not match the verified manifest prefix"
        )
    if training_provenance.get("scheduled_train_count") != world_size:
        raise ValueError("overfit provenance scheduled_train_count mismatch")
    return step_zero_training_schedule(
        scheduled_chain_ids,
        seed=int(training_provenance["seed"]),
        world_size=world_size,
    )


def is_claim_eligible_training_checkpoint(
    *,
    checkpoint: dict[str, Any],
    training_provenance: dict[str, Any],
) -> bool:
    """Require the exact completed full-manifest production contract."""

    schedule = training_provenance.get("schedule")
    return bool(
        isinstance(schedule, dict)
        and schedule.get("claim_eligible") is True
        and schedule.get("selection") == "full_verified_manifest"
        and schedule.get("full_manifest_count") == 10_000
        and schedule.get("scheduled_count") == 10_000
        and schedule.get("train_chain_limit") is None
        and training_provenance.get("train_count") == 10_000
        and training_provenance.get("scheduled_train_count") == 10_000
        and training_provenance.get("max_steps") == 30_000
        and training_provenance.get("save_initial_checkpoint") is False
        and checkpoint.get("schedule") == schedule
        and checkpoint.get("completed_step") == 30_000
    )


def _load_module(name: str, path: Path) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load Python module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _git_metric_revision(root: Path) -> dict[str, Any]:
    try:
        commit = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            [
                "git",
                "-C",
                str(root),
                "status",
                "--porcelain",
                "--",
                "nanofold/metrics.py",
                "nanofold/residue_constants.py",
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (FileNotFoundError, subprocess.CalledProcessError) as error:
        raise RuntimeError(
            f"authoritative NanoFold metric root is not a readable Git checkout: {root}"
        ) from error
    return {"commit": commit, "metric_files_dirty": bool(status)}


def load_official_nanofold_metrics(
    source_root: str | Path,
    *,
    expected_commit: str,
    expected_metrics_sha256: str,
    expected_residue_constants_sha256: str,
) -> tuple[types.ModuleType, dict[str, Any]]:
    """Load and hash the exact predeclared NanoFold metric implementation."""

    root = Path(source_root).resolve()
    metrics_path = root / "nanofold" / "metrics.py"
    constants_path = root / "nanofold" / "residue_constants.py"
    for path in (metrics_path, constants_path):
        if not path.is_file():
            raise FileNotFoundError(f"authoritative NanoFold source missing: {path}")
    observed_hashes = {
        "metrics_sha256": sha256_file(metrics_path),
        "residue_constants_sha256": sha256_file(constants_path),
    }
    expected_hashes = {
        "metrics_sha256": expected_metrics_sha256,
        "residue_constants_sha256": expected_residue_constants_sha256,
    }
    if observed_hashes != expected_hashes:
        raise ValueError(
            "NanoFold metric source hash mismatch: "
            f"expected={expected_hashes}, observed={observed_hashes}"
        )
    revision = _git_metric_revision(root)
    if revision["commit"] != expected_commit:
        raise ValueError(
            f"NanoFold source commit mismatch: {revision['commit']} != {expected_commit}"
        )
    if revision["metric_files_dirty"]:
        raise RuntimeError("NanoFold metric files have uncommitted changes")

    package_name = f"_nanofold_official_{observed_hashes['metrics_sha256'][:16]}"
    package = types.ModuleType(package_name)
    package.__path__ = [str(metrics_path.parent)]  # type: ignore[attr-defined]
    package.__package__ = package_name
    sys.modules[package_name] = package
    constants = _load_module(f"{package_name}.residue_constants", constants_path)
    metrics = _load_module(f"{package_name}.metrics", metrics_path)
    components = tuple(metrics.FOLDSCORE_COMPONENT_NAMES)
    if components != EXPECTED_COMPONENTS:
        raise RuntimeError(
            "NanoFold FoldScore component contract changed: "
            f"{components!r} != {EXPECTED_COMPONENTS!r}"
        )
    if not np.array_equal(
        np.asarray(constants.STANDARD_ATOM14_MASK, dtype=np.bool_),
        STANDARD_ATOM14_MASK,
    ):
        raise RuntimeError(
            "adapter chemistry mask differs from authoritative NanoFold constants"
        )
    provenance = {
        "source_root": str(root),
        "commit": revision["commit"],
        "metric_files_dirty": False,
        "metrics_path": str(metrics_path),
        "residue_constants_path": str(constants_path),
        **observed_hashes,
        "foldscore_components": list(components),
        "chemistry_mask_equal": True,
    }
    return metrics, provenance


def verify_protenix_checkout(protenix_root: Path) -> dict[str, Any]:
    root = protenix_root.resolve()
    commit = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    tree = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD^{tree}"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    tracked_changes = subprocess.run(
        [
            "git",
            "-C",
            str(root),
            "status",
            "--porcelain",
            "--untracked-files=no",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if commit != PROTENIX_COMMIT:
        raise RuntimeError(f"Protenix commit mismatch: {commit} != {PROTENIX_COMMIT}")
    if tracked_changes:
        raise RuntimeError(
            "pinned Protenix checkout has tracked working-tree changes:\n"
            f"{tracked_changes}"
        )
    return {
        "root": str(root),
        "commit": commit,
        "git_tree": tree,
        "tracked_worktree_clean": True,
    }


def verify_checkpoint_sidecar(checkpoint_path: Path) -> dict[str, Any]:
    """Verify the custom BoundedCheckpointManager sidecar before torch.load."""

    path = checkpoint_path.resolve()
    sidecar_path = path.with_suffix(".pt.sha256.json")
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint missing: {path}")
    if not sidecar_path.is_file():
        raise FileNotFoundError(f"checkpoint checksum sidecar missing: {sidecar_path}")
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    observed = {
        "path": path.name,
        "logical_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }
    expected = {
        "path": sidecar.get("path"),
        "logical_bytes": int(sidecar.get("logical_bytes", -1)),
        "sha256": sidecar.get("sha256"),
    }
    if observed != expected:
        raise ValueError(
            "checkpoint checksum sidecar mismatch: "
            f"expected={expected}, observed={observed}"
        )
    completed_step = int(sidecar.get("completed_step", -1))
    if completed_step < 0:
        raise ValueError("checkpoint sidecar has invalid completed_step")
    return {
        **observed,
        "completed_step": completed_step,
        "sidecar_path": str(sidecar_path),
        "sidecar_sha256": sha256_file(sidecar_path),
    }


def load_and_verify_training_provenance(path: Path) -> dict[str, Any]:
    provenance = json.loads(path.read_text(encoding="utf-8"))
    recorded_digest = provenance.get("provenance_sha256")
    unsigned = dict(provenance)
    unsigned.pop("provenance_sha256", None)
    observed_digest = canonical_json_sha256(unsigned)
    if recorded_digest != observed_digest:
        raise ValueError(
            "training provenance self-digest mismatch: "
            f"{recorded_digest} != {observed_digest}"
        )
    return provenance


def training_source_paths() -> dict[str, Path]:
    return {
        "adapter": HERE / "adapter.py",
        "checkpoint": HERE / "checkpoint.py",
        "config_python": HERE / "config.py",
        "config_yaml": HERE / "config.yaml",
        "ccd_exporter": HERE / "export_ccd_reference.py",
        "trainer": HERE / "train.py",
        "prediction_export": HERE / "predict_one.py",
        "launch_8xh100": HERE / "launch_8xh100.sh",
        "launch_overfit_8xh100": HERE / "launch_overfit_8xh100.sh",
        "pilot_8xh100": HERE / "pilot_8xh100.sh",
        "common_data": COMMON_DIR / "data.py",
        "effective_batch": COMMON_DIR / "effective_batch.py",
        "common_runtime": COMMON_DIR / "runtime.py",
        "common_dataset_verifier": COMMON_DIR / "verify_dataset.py",
    }


def validate_training_source_hashes(
    checkpoint: dict[str, Any], training_provenance: dict[str, Any]
) -> None:
    expected = checkpoint.get("source_sha256")
    if not isinstance(expected, dict) or expected != training_provenance.get(
        "source_sha256"
    ):
        raise ValueError("checkpoint and training provenance source hashes differ")
    paths = training_source_paths()
    if set(expected) != set(paths):
        raise ValueError(
            "training source manifest differs from evaluator expectation: "
            f"recorded={sorted(expected)}, expected={sorted(paths)}"
        )
    for name, path in paths.items():
        observed = sha256_file(path)
        if observed != expected[name]:
            raise ValueError(
                f"checkpoint training source mismatch for {name}: "
                f"{observed} != {expected[name]}"
            )


def validate_eval_config(config: dict[str, Any]) -> None:
    data = config.get("data", {})
    sampling = config.get("sampling", {})
    runtime = config.get("runtime", {})
    metrics = config.get("metrics", {})
    if int(data.get("crop_size", 0)) < 1 or int(data.get("msa_depth", 0)) < 1:
        raise ValueError("evaluation crop_size and msa_depth must be positive")
    for key in ("model_seeds", "diffusion_samples", "diffusion_steps"):
        if int(sampling.get(key, 0)) < 1:
            raise ValueError(f"sampling.{key} must be positive")
    if float(sampling.get("mc_dropout_apply_rate", -1.0)) != 0.0:
        raise ValueError("claim-bearing contract requires MC dropout exactly zero")
    if int(runtime.get("world_size", 0)) not in (1, 2, 4, 8):
        raise ValueError("evaluation config must declare 1, 2, 4 or 8 ranks")
    if config.get("aggregation_policy") != AGGREGATION_POLICY_ID:
        raise ValueError(
            "evaluation aggregation policy differs from the predeclared policy"
        )
    for key in (
        "nanofold_source_commit",
        "metrics_sha256",
        "residue_constants_sha256",
    ):
        value = str(metrics.get(key, ""))
        required_length = 40 if key == "nanofold_source_commit" else 64
        if len(value) != required_length or any(
            character not in "0123456789abcdef" for character in value
        ):
            kind = "full Git commit" if required_length == 40 else "lowercase SHA-256"
            raise ValueError(f"metrics.{key} is not a {kind}")


def validate_checkpoint_contract(
    *,
    checkpoint: dict[str, Any],
    checkpoint_verification: dict[str, Any],
    training_provenance: dict[str, Any],
    protenix_checkout: dict[str, Any],
    resolved_training_config_sha256: str,
    ccd_digest: str,
    dataset_verification: dict[str, Any],
    dataset_verification_path: Path,
    validation_manifest_path: Path,
    validation_count: int,
) -> None:
    if int(checkpoint.get("completed_step", -1)) != int(
        checkpoint_verification["completed_step"]
    ):
        raise ValueError("checkpoint body and checksum sidecar completed_step differ")
    expected_pairs = (
        (
            "training provenance",
            checkpoint.get("provenance_sha256"),
            training_provenance.get("provenance_sha256"),
        ),
        ("Protenix commit", checkpoint.get("protenix_commit"), PROTENIX_COMMIT),
        ("Protenix Git tree", checkpoint.get("protenix_git_tree"), protenix_checkout["git_tree"]),
        (
            "resolved training config",
            checkpoint.get("resolved_config_sha256"),
            resolved_training_config_sha256,
        ),
        (
            "CCD reference",
            checkpoint.get("ccd_reference_sha256"),
            ccd_digest,
        ),
    )
    for name, observed, expected in expected_pairs:
        if observed != expected:
            raise ValueError(f"{name} mismatch: {observed} != {expected}")
    validate_training_source_hashes(checkpoint, training_provenance)
    if checkpoint.get("schedule") != training_provenance.get("schedule"):
        raise ValueError("checkpoint and training provenance schedules differ")
    if checkpoint.get("save_initial_checkpoint") != training_provenance.get(
        "save_initial_checkpoint"
    ):
        raise ValueError(
            "checkpoint and training provenance initial-checkpoint flags differ"
        )
    physical_world = int(checkpoint.get("world_size", -1))
    expected_batch = accumulation_contract(physical_world)
    if checkpoint.get("batch_execution") != expected_batch or training_provenance.get("batch_execution") != expected_batch:
        raise ValueError("effective-eight accumulation contract differs")
    if checkpoint.get("samples_seen") != int(checkpoint["completed_step"]) * 8:
        raise ValueError("checkpoint sample count differs from effective batch eight")

    if int(dataset_verification.get("validation_count", -1)) != 1_000:
        raise ValueError("dataset verification does not certify 1,000 validation chains")
    if validation_count != 1_000:
        raise ValueError(f"public-validation manifest must contain 1,000 chains, got {validation_count}")
    manifest_hash = sha256_file(validation_manifest_path)
    if manifest_hash != dataset_verification["manifest_sha256"]["validation"]:
        raise ValueError("validation manifest disagrees with dataset verification")
    if manifest_hash != training_provenance.get("validation_manifest_sha256"):
        raise ValueError("validation manifest disagrees with training provenance")
    report_digest = sha256_file(dataset_verification_path)
    recorded_dataset = training_provenance.get("dataset_verification", {})
    if report_digest != recorded_dataset.get("report_sha256"):
        raise ValueError("evaluation dataset report differs from the training report")
    tree_digest = dataset_verification["inventory"]["tree_sha256"]
    for name, value in (
        ("training provenance", recorded_dataset.get("tree_sha256")),
        ("checkpoint", checkpoint.get("dataset_tree_sha256")),
    ):
        if value != tree_digest:
            raise ValueError(
                f"dataset tree mismatch for {name}: {value} != {tree_digest}"
            )


def verify_evaluation_inputs(
    *,
    rank: int,
    data_root: Path,
    dataset_verification: dict[str, Any],
    checkpoint_path: Path,
) -> dict[str, Any]:
    """Rank zero rehashes immutable inputs and broadcasts success or failure."""

    result: list[dict[str, Any] | None] = [None]
    if rank == 0:
        try:
            result[0] = {
                "ok": True,
                "dataset": verify_dataset_snapshot(data_root, dataset_verification),
                "checkpoint": verify_checkpoint_sidecar(checkpoint_path),
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
        "dataset": dict(outcome["dataset"]),
        "checkpoint": dict(outcome["checkpoint"]),
    }


def predict_target_free_sample(
    model: Any,
    adapted_item: dict[str, Any],
    *,
    device: torch.device,
    seed: int,
    use_bf16: bool,
) -> torch.Tensor:
    """Run one seeded sample while labels are absent from every model argument."""

    if "label_dict" in adapted_item or "label_full_dict" in adapted_item:
        raise AssertionError("target-free inference item unexpectedly contains labels")
    features = adapted_item["input_feature_dict"]
    forbidden = {
        "coordinate",
        "coordinate_mask",
        "atom14_positions",
        "atom14_positions_uncentered",
        "atom14_mask",
        "ca_coords",
        "ca_mask",
    }
    leaked = forbidden.intersection(features)
    if leaked:
        raise AssertionError(f"target tensors leaked into Protenix features: {leaked}")
    device_features = to_device(features, device)
    amp = (
        torch.autocast("cuda", dtype=torch.bfloat16)
        if use_bf16 and device.type == "cuda"
        else nullcontext()
    )
    # Model construction may consume RNG. Reset immediately before every
    # independently addressable chain/sample call.
    seed_inference(seed)
    with torch.inference_mode(), amp:
        prediction, _, _ = model(
            input_feature_dict=device_features,
            label_full_dict=None,
            label_dict=None,
            mode="inference",
            current_step=None,
            symmetric_permutation=None,
            mc_dropout_apply_rate=0.0,
        )
    exported = prediction_to_nanofold(prediction, adapted_item, sample_index=0)
    atom14 = exported["pred_atom14"].detach().float().cpu()
    if atom14.ndim != 3 or atom14.shape[-2:] != (14, 3):
        raise ValueError(f"invalid exported atom14 shape: {tuple(atom14.shape)}")
    if not torch.isfinite(atom14).all():
        raise FloatingPointError("Protenix prediction contains non-finite coordinates")
    return atom14


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


def prediction_path(
    root: Path,
    chain_id: str,
    model_seed_index: int,
    diffusion_sample_index: int,
) -> Path:
    encoded = chain_id.encode("utf-8").hex()
    return (
        root
        / f"model-seed-{model_seed_index:03d}"
        / f"diffusion-sample-{diffusion_sample_index:03d}"
        / f"chain_{encoded}.npz"
    )


def record_path(
    root: Path,
    chain_id: str,
    model_seed_index: int,
    diffusion_sample_index: int,
) -> Path:
    encoded = chain_id.encode("utf-8").hex()
    return (
        root
        / f"model-seed-{model_seed_index:03d}"
        / f"diffusion-sample-{diffusion_sample_index:03d}"
        / f"chain_{encoded}.json"
    )


def load_scoring_target(
    data_root: Path,
    chain_id: str,
    *,
    source_length: int,
    crop_start: int,
    crop_length: int,
) -> tuple[torch.Tensor, torch.Tensor, Path]:
    """Open validation coordinates only after prediction persistence."""

    label_path = chain_npz_path(data_root / "processed_labels", chain_id)
    if not label_path.is_file():
        raise FileNotFoundError(f"validation label missing: {label_path}")
    with np.load(label_path, allow_pickle=False) as labels:
        missing = {"atom14_positions", "atom14_mask"}.difference(labels.files)
        if missing:
            raise ValueError(f"{label_path}: missing scoring labels {sorted(missing)}")
        positions = np.asarray(labels["atom14_positions"], dtype=np.float32)
        mask = np.asarray(labels["atom14_mask"], dtype=np.bool_)
    if positions.shape != (source_length, 14, 3):
        raise ValueError(f"{label_path}: invalid atom14_positions shape {positions.shape}")
    if mask.shape != (source_length, 14):
        raise ValueError(f"{label_path}: invalid atom14_mask shape {mask.shape}")
    crop_end = crop_start + crop_length
    if not 0 <= crop_start < crop_end <= source_length:
        raise ValueError("prediction crop is outside source chain")
    return (
        torch.from_numpy(positions[crop_start:crop_end]).float(),
        torch.from_numpy(mask[crop_start:crop_end]).bool(),
        label_path,
    )


def masked_kabsch_rmsd(
    pred: torch.Tensor, true: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    active = mask.to(device=pred.device, dtype=torch.bool)
    if int(active.sum().item()) < 3:
        raise ValueError("fewer than three resolved C-alpha atoms for RMSD")
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
    return torch.sqrt(
        ((pred_centered @ rotation) - true_centered)
        .square()
        .sum(dim=-1)
        .mean()
        .clamp_min(0)
    )


def score_prediction_artifact(
    prediction_file: Path,
    *,
    data_root: Path,
    official_metrics: types.ModuleType,
    device: torch.device,
) -> tuple[dict[str, float], dict[str, Any]]:
    """Score an already persisted atom14 artifact with exact pinned metrics."""

    if not prediction_file.is_file():
        raise FileNotFoundError(
            f"prediction must be persisted before scoring: {prediction_file}"
        )
    with np.load(prediction_file, allow_pickle=False) as prediction:
        required = {
            "pred_atom14",
            "aatype",
            "chain_id",
            "source_length",
            "crop_start",
            "crop_length",
        }
        missing = required.difference(prediction.files)
        if missing:
            raise ValueError(
                f"{prediction_file}: missing prediction fields {sorted(missing)}"
            )
        pred = torch.from_numpy(
            np.asarray(prediction["pred_atom14"], dtype=np.float32)
        )
        aatype = torch.from_numpy(np.asarray(prediction["aatype"], dtype=np.int64))
        chain_id = str(np.asarray(prediction["chain_id"]).item())
        source_length = int(np.asarray(prediction["source_length"]).item())
        crop_start = int(np.asarray(prediction["crop_start"]).item())
        crop_length = int(np.asarray(prediction["crop_length"]).item())
    if pred.shape != (crop_length, 14, 3):
        raise ValueError(f"{prediction_file}: invalid pred_atom14 shape {tuple(pred.shape)}")
    if aatype.shape != (crop_length,):
        raise ValueError(f"{prediction_file}: invalid aatype shape {tuple(aatype.shape)}")
    if not torch.isfinite(pred).all():
        raise FloatingPointError(f"{prediction_file}: non-finite prediction")

    # This is the first semantic access to validation coordinates.
    true, target_mask, label_path = load_scoring_target(
        data_root,
        chain_id,
        source_length=source_length,
        crop_start=crop_start,
        crop_length=crop_length,
    )
    pred = pred.to(device)
    true = true.to(device)
    target_mask = target_mask.to(device)
    aatype = aatype.to(device)
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
        masked_kabsch_rmsd(pred[:, 1], true[:, 1], target_mask[:, 1])
        .detach()
        .float()
        .cpu()
    )
    for name, value in values.items():
        if not math.isfinite(value):
            raise FloatingPointError(f"non-finite metric {name}: {value}")
    return values, {
        "label_path": str(label_path),
        "label_sha256": sha256_file(label_path),
    }


def read_completed_rows(output_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted((output_dir / "records").glob("**/chain_*.json")):
        try:
            row = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid atomic evaluation record {path}: {error}") from error
        if not isinstance(row, dict):
            raise ValueError(f"evaluation record is not an object: {path}")
        rows.append(row)
    return rows


def validate_resume_rows(
    rows: list[dict[str, Any]],
    *,
    output_dir: Path,
    checkpoint_step: int,
    checkpoint_sha256: str,
    evaluation_contract_sha256: str,
    global_seed: int,
    model_seeds: int,
    diffusion_samples: int,
    allowed_chain_ids: set[str],
) -> None:
    """Refuse stale, mixed-checkpoint, wrongly seeded, or corrupt artifacts."""

    for row in rows:
        if row.get("architecture") != ARCHITECTURE_ID:
            raise ValueError("resume record contains a different architecture")
        if bool(row.get("official_alphafold3", True)):
            raise ValueError("resume record lacks the AF3-style disclaimer")
        if int(row.get("checkpoint_completed_step", -1)) != checkpoint_step:
            raise ValueError("resume record checkpoint step differs")
        if row.get("checkpoint_sha256") != checkpoint_sha256:
            raise ValueError("resume record checkpoint hash differs")
        if row.get("evaluation_contract_sha256") != evaluation_contract_sha256:
            raise ValueError("resume record evaluation contract differs")
        chain_id = str(row.get("chain_id", ""))
        if chain_id not in allowed_chain_ids:
            raise ValueError(f"resume record chain is outside selection: {chain_id}")
        model_seed_index = int(row.get("model_seed_index", -1))
        diffusion_sample_index = int(row.get("diffusion_sample_index", -1))
        if not 0 <= model_seed_index < model_seeds:
            raise ValueError("resume record model_seed_index is invalid")
        if not 0 <= diffusion_sample_index < diffusion_samples:
            raise ValueError("resume record diffusion_sample_index is invalid")
        expected_sample_index = flat_sample_index(
            model_seed_index,
            diffusion_sample_index,
            diffusion_samples=diffusion_samples,
        )
        if int(row.get("sample_index", -1)) != expected_sample_index:
            raise ValueError("resume record flattened sample_index is invalid")
        expected_seed = sample_seed(
            global_seed,
            chain_id,
            model_seed_index,
            diffusion_sample_index,
        )
        if int(row.get("sample_seed", -1)) != expected_seed:
            raise ValueError("resume record sample seed differs from deterministic policy")
        relative_prediction = Path(str(row.get("prediction_path", "")))
        if relative_prediction.is_absolute() or ".." in relative_prediction.parts:
            raise ValueError(f"unsafe resume prediction path: {relative_prediction}")
        prediction_file = (output_dir / relative_prediction).resolve()
        prediction_root = (output_dir / "predictions").resolve()
        if prediction_root not in prediction_file.parents:
            raise ValueError("resume prediction is outside the prediction root")
        expected_relative_prediction = prediction_path(
            Path("predictions"),
            chain_id,
            model_seed_index,
            diffusion_sample_index,
        )
        if relative_prediction != expected_relative_prediction:
            raise ValueError(
                "resume prediction path differs from its chain/sample address"
            )
        if not prediction_file.is_file():
            raise FileNotFoundError(f"resume prediction missing: {prediction_file}")
        if row.get("prediction_sha256") != sha256_file(prediction_file):
            raise ValueError(f"resume prediction hash mismatch: {relative_prediction}")
        with np.load(prediction_file, allow_pickle=False) as prediction:
            metadata = {
                "chain_id": str(np.asarray(prediction["chain_id"]).item()),
                "model_seed_index": int(
                    np.asarray(prediction["model_seed_index"]).item()
                ),
                "diffusion_sample_index": int(
                    np.asarray(prediction["diffusion_sample_index"]).item()
                ),
                "sample_index": int(np.asarray(prediction["sample_index"]).item()),
                "sample_seed": int(np.asarray(prediction["sample_seed"]).item()),
                "checkpoint_completed_step": int(
                    np.asarray(prediction["checkpoint_completed_step"]).item()
                ),
                "checkpoint_sha256": str(
                    np.asarray(prediction["checkpoint_sha256"]).item()
                ),
                "evaluation_contract_sha256": str(
                    np.asarray(prediction["evaluation_contract_sha256"]).item()
                ),
                "target_files_accessed": bool(
                    np.asarray(prediction["target_files_accessed"]).item()
                ),
            }
            pred_atom14 = np.asarray(prediction["pred_atom14"])
            crop_length = int(np.asarray(prediction["crop_length"]).item())
        expected_metadata = {
            "chain_id": chain_id,
            "model_seed_index": model_seed_index,
            "diffusion_sample_index": diffusion_sample_index,
            "sample_index": expected_sample_index,
            "sample_seed": expected_seed,
            "checkpoint_completed_step": checkpoint_step,
            "checkpoint_sha256": checkpoint_sha256,
            "evaluation_contract_sha256": evaluation_contract_sha256,
            "target_files_accessed": False,
        }
        if metadata != expected_metadata:
            raise ValueError(
                "resume prediction embedded metadata differs from record: "
                f"{metadata} != {expected_metadata}"
            )
        if pred_atom14.shape != (crop_length, 14, 3):
            raise ValueError("resume prediction has invalid atom14 shape")
        if not np.isfinite(pred_atom14).all():
            raise FloatingPointError("resume prediction has non-finite coordinates")


def validate_resume_provenance(
    existing: dict[str, Any], current: dict[str, Any]
) -> None:
    if existing.get("evaluation_contract_sha256") != current.get(
        "evaluation_contract_sha256"
    ):
        raise ValueError("cannot resume because the evaluation contract differs")
    if existing.get("aggregation_policy") != current.get("aggregation_policy"):
        raise ValueError("cannot resume because the aggregation policy differs")


def _mean_std(values: Iterable[float]) -> tuple[float, float]:
    array = np.asarray(list(values), dtype=np.float64)
    if not len(array):
        raise ValueError("cannot summarize empty metrics")
    return (
        float(array.mean()),
        float(array.std(ddof=1)) if len(array) > 1 else 0.0,
    )


def consolidate_rows(
    rows: list[dict[str, Any]],
    *,
    expected_chain_ids: list[str],
    model_seeds: int,
    diffusion_samples: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Check completeness and apply the predeclared sample-mean policy."""

    manifest_order = {chain_id: index for index, chain_id in enumerate(expected_chain_ids)}
    ordered = sorted(
        rows,
        key=lambda row: (
            manifest_order.get(str(row["chain_id"]), len(manifest_order)),
            int(row["model_seed_index"]),
            int(row["diffusion_sample_index"]),
        ),
    )
    expected = {
        (chain_id, model_seed_index, diffusion_sample_index)
        for chain_id in expected_chain_ids
        for model_seed_index in range(model_seeds)
        for diffusion_sample_index in range(diffusion_samples)
    }
    observed = [
        (
            str(row["chain_id"]),
            int(row["model_seed_index"]),
            int(row["diffusion_sample_index"]),
        )
        for row in ordered
    ]
    if len(observed) != len(set(observed)):
        raise ValueError("duplicate chain/model-seed/diffusion-sample records")
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
        samples = by_chain[chain_id]
        metadata_fields = ("source_length", "crop_start", "crop_length")
        for field in metadata_fields:
            if len({int(row[field]) for row in samples}) != 1:
                raise ValueError(f"{chain_id}: inconsistent {field} across samples")
        chain_row: dict[str, Any] = {
            "chain_id": chain_id,
            "source_length": int(samples[0]["source_length"]),
            "crop_start": int(samples[0]["crop_start"]),
            "crop_length": int(samples[0]["crop_length"]),
            "num_samples": len(samples),
        }
        for metric in METRIC_NAMES:
            mean, std = _mean_std(float(row[metric]) for row in samples)
            chain_row[f"{metric}_mean"] = mean
            chain_row[f"{metric}_sample_std"] = std
        chain_rows.append(chain_row)

    aggregate_rows: list[dict[str, Any]] = []
    for model_seed_index in range(model_seeds):
        for diffusion_sample_index in range(diffusion_samples):
            sample_rows = [
                row
                for row in ordered
                if int(row["model_seed_index"]) == model_seed_index
                and int(row["diffusion_sample_index"]) == diffusion_sample_index
            ]
            aggregate: dict[str, Any] = {
                "scope": "fixed-sample-index-chain-macro",
                "model_seed_index": model_seed_index,
                "diffusion_sample_index": diffusion_sample_index,
                "num_chains": len(sample_rows),
                "num_samples_per_chain": 1,
            }
            for metric in METRIC_NAMES:
                mean, std = _mean_std(float(row[metric]) for row in sample_rows)
                aggregate[f"{metric}_mean"] = mean
                aggregate[f"{metric}_chain_std"] = std
                aggregate[f"{metric}_chain_sem"] = std / math.sqrt(len(sample_rows))
            aggregate_rows.append(aggregate)

    primary: dict[str, Any] = {
        "scope": "primary-sample-mean-then-equal-chain-macro",
        "model_seed_index": "all",
        "diffusion_sample_index": "all",
        "num_chains": len(chain_rows),
        "num_samples_per_chain": model_seeds * diffusion_samples,
        "aggregation_policy": AGGREGATION_POLICY_ID,
    }
    for metric in METRIC_NAMES:
        mean, std = _mean_std(float(row[f"{metric}_mean"]) for row in chain_rows)
        primary[f"{metric}_mean"] = mean
        primary[f"{metric}_chain_std"] = std
        primary[f"{metric}_chain_sem"] = std / math.sqrt(len(chain_rows))
    aggregate_rows.append(primary)
    if any("best" in key.lower() for key in primary):
        raise AssertionError("target-based best-of-sample aggregation is prohibited")
    return ordered, chain_rows, {"rows": aggregate_rows, "primary": primary}


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


def _file_record(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    return {
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def source_provenance(
    *,
    args: argparse.Namespace,
    config_path: Path,
    checkpoint_verification: dict[str, Any],
) -> dict[str, Any]:
    paths = {
        "evaluate.py": Path(__file__).resolve(),
        "adapter.py": HERE / "adapter.py",
        "predict_one.py": HERE / "predict_one.py",
        "config.py": HERE / "config.py",
        "checkpoint.py": HERE / "checkpoint.py",
        "common/data.py": COMMON_DIR / "data.py",
        "common/runtime.py": COMMON_DIR / "runtime.py",
        "common/verify_dataset.py": COMMON_DIR / "verify_dataset.py",
        "eval_config": config_path,
        "training_provenance": args.training_provenance.resolve(),
        "dataset_verification": args.dataset_verification.resolve(),
        "training_manifest": args.data_root.resolve() / "manifests" / "train.txt",
        "validation_manifest": args.data_root.resolve() / "manifests" / "val.txt",
    }
    records = {name: _file_record(path) for name, path in paths.items()}
    checkpoint_path = args.checkpoint.resolve()
    records["checkpoint"] = {
        "path": str(checkpoint_path),
        "bytes": int(checkpoint_verification["logical_bytes"]),
        "sha256": str(checkpoint_verification["sha256"]),
    }
    sidecar_path = Path(str(checkpoint_verification["sidecar_path"]))
    records["checkpoint_sidecar"] = _file_record(sidecar_path)
    return records


def build_evaluation_contract(
    *,
    config: dict[str, Any],
    checkpoint_verification: dict[str, Any],
    training_provenance: dict[str, Any],
    dataset_verification: dict[str, Any],
    metric_provenance: dict[str, Any],
    protenix_checkout: dict[str, Any],
    sources: dict[str, Any],
    selection: dict[str, Any],
) -> dict[str, Any]:
    contract = {
        "schema_version": 1,
        "architecture": ARCHITECTURE_ID,
        "official_alphafold3": False,
        "checkpoint_completed_step": int(checkpoint_verification["completed_step"]),
        "checkpoint_sha256": checkpoint_verification["sha256"],
        "training_provenance_sha256": training_provenance["provenance_sha256"],
        "dataset_tree_sha256": dataset_verification["inventory"]["tree_sha256"],
        "validation_manifest_sha256": dataset_verification["manifest_sha256"][
            "validation"
        ],
        "protenix_commit": protenix_checkout["commit"],
        "protenix_git_tree": protenix_checkout["git_tree"],
        "metrics_sha256": metric_provenance["metrics_sha256"],
        "residue_constants_sha256": metric_provenance[
            "residue_constants_sha256"
        ],
        "eval_config": config,
        "selection": selection,
        "aggregation_policy": AGGREGATION_POLICY,
        "source_sha256": {
            name: record["sha256"] for name, record in sources.items()
        },
    }
    contract["sha256"] = canonical_json_sha256(contract)
    return contract


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    checkpoint_path = args.checkpoint.resolve()
    data_root = args.data_root.resolve()
    output_dir = args.output_dir.resolve()
    if args.max_chains is not None and args.max_chains < 1:
        raise ValueError("--max-chains must be positive")
    existing_records = (
        list((output_dir / "records").glob("**/chain_*.json"))
        if output_dir.exists()
        else []
    )
    if existing_records and not args.resume:
        raise FileExistsError(
            f"{output_dir} contains evaluation records; pass --resume or use a new path"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    config_payload = json.loads(config_path.read_text(encoding="utf-8"))
    validate_eval_config(config_payload)
    dataset_verification = json.loads(
        args.dataset_verification.read_text(encoding="utf-8")
    )
    training_provenance = load_and_verify_training_provenance(
        args.training_provenance.resolve()
    )

    rank, _local_rank, world_size, device = initialize_distributed()
    configured_world_size = int(config_payload["runtime"]["world_size"])
    if world_size != configured_world_size:
        raise RuntimeError(
            f"evaluation requires world_size={configured_world_size}, got {world_size}"
        )
    input_verification = verify_evaluation_inputs(
        rank=rank,
        data_root=data_root,
        dataset_verification=dataset_verification,
        checkpoint_path=checkpoint_path,
    )
    checkpoint_verification = input_verification["checkpoint"]
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    protenix_checkout = verify_protenix_checkout(args.protenix_root)
    sys.path.insert(0, str(args.protenix_root.resolve()))
    ccd_reference = load_protenix_ccd_reference()
    ccd_digest = ccd_reference_sha256(ccd_reference)
    training_config = build_protenix_config()
    training_config.max_steps = int(training_provenance["max_steps"])
    training_config.seed = int(training_provenance["seed"])
    resolved_training_config_sha256 = canonical_json_sha256(training_config.to_dict())

    validation_manifest_path = data_root / "manifests" / "val.txt"
    full_chain_ids = read_manifest(validation_manifest_path)
    validate_checkpoint_contract(
        checkpoint=checkpoint,
        checkpoint_verification=checkpoint_verification,
        training_provenance=training_provenance,
        protenix_checkout=protenix_checkout,
        resolved_training_config_sha256=resolved_training_config_sha256,
        ccd_digest=ccd_digest,
        dataset_verification=dataset_verification,
        dataset_verification_path=args.dataset_verification.resolve(),
        validation_manifest_path=validation_manifest_path,
        validation_count=len(full_chain_ids),
    )

    metrics_config = config_payload["metrics"]
    official_metrics, metric_provenance = load_official_nanofold_metrics(
        args.nanofold_source_root,
        expected_commit=str(metrics_config["nanofold_source_commit"]),
        expected_metrics_sha256=str(metrics_config["metrics_sha256"]),
        expected_residue_constants_sha256=str(
            metrics_config["residue_constants_sha256"]
        ),
    )

    # Validate the training configuration before applying inference-only
    # overrides. Each outer-loop call produces one independently seeded sample.
    training_config.sample_diffusion.N_step = int(
        config_payload["sampling"]["diffusion_steps"]
    )
    training_config.sample_diffusion.N_sample = 1
    training_config.model.N_model_seed = 1
    training_config.deterministic = True
    training_config.deterministic_seed = True
    from protenix.model.protenix import Protenix

    model = Protenix(training_config).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    parameter_count = sum(parameter.numel() for parameter in model.parameters())

    if args.overfit_train_schedule_step0:
        training_chain_ids = tuple(
            read_manifest(data_root / "manifests" / "train.txt")
        )
        scheduled = overfit_step_zero_training_schedule(
            training_chain_ids,
            training_provenance=training_provenance,
            world_size=8,
        )
        chain_ids = [str(item["chain_id"]) for item in scheduled]
        evaluation_selection = {
            "scope": "overfit-training-schedule-step0",
            "claim_bearing": False,
            "manifest": "train",
            "training_seed": int(training_provenance["seed"]),
            "training_world_size": 8,
            "training_optimizer_step": 0,
            "scheduled": scheduled,
            "chain_ids": chain_ids,
            "evaluation_crop_mode": "center",
        }
        claim_bearing = False
    else:
        chain_ids = (
            full_chain_ids
            if args.max_chains is None
            else full_chain_ids[: args.max_chains]
        )
        training_checkpoint_claim_eligible = is_claim_eligible_training_checkpoint(
            checkpoint=checkpoint,
            training_provenance=training_provenance,
        )
        claim_bearing = training_checkpoint_claim_eligible and is_claim_bearing_evaluation(
            full_manifest_count=len(full_chain_ids),
            selected_count=len(chain_ids),
            max_chains=args.max_chains,
            world_size=world_size,
        )
        evaluation_selection = {
            "scope": (
                "full-public-validation"
                if args.max_chains is None
                else "public-validation-prefix-diagnostic"
            ),
            "claim_bearing": claim_bearing,
            "manifest": "val",
            "max_chains": args.max_chains,
            "chain_ids": chain_ids,
            "evaluation_crop_mode": "center",
            "training_checkpoint_claim_eligible": (
                training_checkpoint_claim_eligible
            ),
        }
    del checkpoint
    sampling = config_payload["sampling"]
    model_seeds = int(sampling["model_seeds"])
    diffusion_samples = int(sampling["diffusion_samples"])
    total_samples = model_seeds * diffusion_samples
    global_seed = int(sampling["global_seed"])
    use_bf16 = bool(config_payload["runtime"]["bf16"])
    for config_name, provenance_name in (
        ("crop_size", "crop_size"),
        ("msa_depth", "msa_depth"),
    ):
        observed = int(config_payload["data"][config_name])
        expected = int(training_provenance[provenance_name])
        if observed != expected:
            raise ValueError(
                f"evaluation {config_name} differs from training provenance: "
                f"{observed} != {expected}"
            )
    sources = source_provenance(
        args=args,
        config_path=config_path,
        checkpoint_verification=checkpoint_verification,
    )
    evaluation_contract = build_evaluation_contract(
        config=config_payload,
        checkpoint_verification=checkpoint_verification,
        training_provenance=training_provenance,
        dataset_verification=dataset_verification,
        metric_provenance=metric_provenance,
        protenix_checkout=protenix_checkout,
        sources=sources,
        selection=evaluation_selection,
    )
    provenance = {
        "schema_version": 1,
        "architecture": "Protenix AF3-style NanoFold reconstruction",
        "architecture_id": ARCHITECTURE_ID,
        "official_alphafold3": False,
        "claim_bearing_full_public_validation": claim_bearing,
        "full_public_validation_requirement": (
            "Exactly all 1,000 manifest chains, no --max-chains, and 8 ranks."
        ),
        "evaluation_selection": evaluation_selection,
        "checkpoint_completed_step": checkpoint_verification["completed_step"],
        "parameter_count": parameter_count,
        "world_size": world_size,
        "config": config_payload,
        "sampling_execution": {
            "outer_model_seed_repeats": model_seeds,
            "outer_diffusion_samples_per_model_seed": diffusion_samples,
            "upstream_N_model_seed_per_call": 1,
            "upstream_N_sample_per_call": 1,
            "diffusion_steps": int(sampling["diffusion_steps"]),
            "mc_dropout_apply_rate": 0.0,
            "seed_derivation": (
                "low 63 bits of sha256(namespace, global_seed, chain_id, "
                "model_seed_index, diffusion_sample_index); rank and world "
                "size are absent"
            ),
        },
        "aggregation_policy": AGGREGATION_POLICY,
        "dataset_verification": dataset_verification,
        "observed_dataset_verification": input_verification["dataset"],
        "metric_source": metric_provenance,
        "protenix_source": protenix_checkout,
        "source_files": sources,
        "evaluation_contract": evaluation_contract,
        "evaluation_contract_sha256": evaluation_contract["sha256"],
        "environment": {
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "python": sys.version,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": str(device),
        },
    }

    completed_rows = read_completed_rows(output_dir) if args.resume else []
    existing_provenance_path = output_dir / "provenance.json"
    if args.resume:
        if completed_rows and not existing_provenance_path.is_file():
            raise FileNotFoundError("resume records exist without provenance.json")
        if existing_provenance_path.is_file():
            validate_resume_provenance(
                json.loads(existing_provenance_path.read_text(encoding="utf-8")),
                provenance,
            )
        validate_resume_rows(
            completed_rows,
            output_dir=output_dir,
            checkpoint_step=int(checkpoint_verification["completed_step"]),
            checkpoint_sha256=str(checkpoint_verification["sha256"]),
            evaluation_contract_sha256=evaluation_contract["sha256"],
            global_seed=global_seed,
            model_seeds=model_seeds,
            diffusion_samples=diffusion_samples,
            allowed_chain_ids=set(chain_ids),
        )
    completed = {
        (
            str(row["chain_id"]),
            int(row["model_seed_index"]),
            int(row["diffusion_sample_index"]),
        )
        for row in completed_rows
    }
    if rank == 0:
        atomic_json(existing_provenance_path, provenance)
    if dist.is_initialized():
        dist.barrier()

    rank_chain_ids = partition_chain_ids(chain_ids, rank=rank, world_size=world_size)
    started = time.perf_counter()
    processed_samples = 0
    for chain_id in rank_chain_ids:
        source, feature_path = load_target_free_example(
            data_root,
            chain_id,
            crop_size=int(config_payload["data"]["crop_size"]),
            msa_depth=int(config_payload["data"]["msa_depth"]),
        )
        for model_seed_index in range(model_seeds):
            for diffusion_sample_index in range(diffusion_samples):
                key = (chain_id, model_seed_index, diffusion_sample_index)
                if key in completed:
                    continue
                seed = sample_seed(
                    global_seed,
                    chain_id,
                    model_seed_index,
                    diffusion_sample_index,
                )
                sample_started = time.perf_counter()
                adapted_item = nanofold_inputs_to_protenix(
                    source,
                    seed=seed,
                    step=0,
                    rank=0,
                    augment_reference=False,
                    ccd_reference=ccd_reference,
                )
                pred_atom14 = predict_target_free_sample(
                    model,
                    adapted_item,
                    device=device,
                    seed=seed,
                    use_bf16=use_bf16,
                )
                crop_length = int(source["crop_length"])
                prediction_file = prediction_path(
                    output_dir / "predictions",
                    chain_id,
                    model_seed_index,
                    diffusion_sample_index,
                )
                sample_index = flat_sample_index(
                    model_seed_index,
                    diffusion_sample_index,
                    diffusion_samples=diffusion_samples,
                )
                atomic_save_prediction(
                    prediction_file,
                    pred_atom14=pred_atom14.numpy().astype(np.float32),
                    aatype=source["aatype"].cpu().numpy().astype(np.int64),
                    chain_id=np.asarray(chain_id),
                    source_length=np.asarray(
                        int(source["source_length"]), dtype=np.int32
                    ),
                    crop_start=np.asarray(int(source["crop_start"]), dtype=np.int32),
                    crop_length=np.asarray(crop_length, dtype=np.int32),
                    model_seed_index=np.asarray(model_seed_index, dtype=np.int32),
                    diffusion_sample_index=np.asarray(
                        diffusion_sample_index, dtype=np.int32
                    ),
                    sample_index=np.asarray(sample_index, dtype=np.int32),
                    sample_seed=np.asarray(seed, dtype=np.int64),
                    checkpoint_completed_step=np.asarray(
                        int(checkpoint_verification["completed_step"]), dtype=np.int64
                    ),
                    checkpoint_sha256=np.asarray(checkpoint_verification["sha256"]),
                    evaluation_contract_sha256=np.asarray(
                        evaluation_contract["sha256"]
                    ),
                    target_files_accessed=np.asarray(False),
                )
                prediction_digest = sha256_file(prediction_file)
                metric_values, scoring_provenance = score_prediction_artifact(
                    prediction_file,
                    data_root=data_root,
                    official_metrics=official_metrics,
                    device=device,
                )
                row = {
                    "schema_version": 1,
                    "architecture": ARCHITECTURE_ID,
                    "official_alphafold3": False,
                    "checkpoint_completed_step": int(
                        checkpoint_verification["completed_step"]
                    ),
                    "checkpoint_sha256": checkpoint_verification["sha256"],
                    "evaluation_contract_sha256": evaluation_contract["sha256"],
                    "chain_id": chain_id,
                    "source_length": int(source["source_length"]),
                    "crop_start": int(source["crop_start"]),
                    "crop_length": crop_length,
                    "model_seed_index": model_seed_index,
                    "diffusion_sample_index": diffusion_sample_index,
                    "sample_index": sample_index,
                    "sample_seed": seed,
                    "feature_path": str(feature_path),
                    "feature_sha256": sha256_file(feature_path),
                    "prediction_path": str(prediction_file.relative_to(output_dir)),
                    "prediction_sha256": prediction_digest,
                    "target_files_accessed_by_model": False,
                    "scoring_label_sha256": scoring_provenance["label_sha256"],
                    "sampling_and_scoring_seconds": time.perf_counter()
                    - sample_started,
                    **metric_values,
                }
                atomic_json(
                    record_path(
                        output_dir / "records",
                        chain_id,
                        model_seed_index,
                        diffusion_sample_index,
                    ),
                    row,
                )
                processed_samples += 1
                atomic_json(
                    output_dir / "progress" / f"rank-{rank:03d}.json",
                    {
                        "rank": rank,
                        "world_size": world_size,
                        "state": "evaluating",
                        "processed_samples_this_invocation": processed_samples,
                        "assigned_chains": len(rank_chain_ids),
                        "last_chain_id": chain_id,
                        "last_model_seed_index": model_seed_index,
                        "last_diffusion_sample_index": diffusion_sample_index,
                        "elapsed_seconds": time.perf_counter() - started,
                    },
                )

    atomic_json(
        output_dir / "progress" / f"rank-{rank:03d}.json",
        {
            "rank": rank,
            "world_size": world_size,
            "state": "complete",
            "processed_samples_this_invocation": processed_samples,
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
            model_seeds=model_seeds,
            diffusion_samples=diffusion_samples,
        )
        sample_fields = [
            "architecture",
            "official_alphafold3",
            "checkpoint_completed_step",
            "checkpoint_sha256",
            "evaluation_contract_sha256",
            "chain_id",
            "source_length",
            "crop_start",
            "crop_length",
            "model_seed_index",
            "diffusion_sample_index",
            "sample_index",
            "sample_seed",
            "feature_sha256",
            "prediction_path",
            "prediction_sha256",
            "target_files_accessed_by_model",
            "scoring_label_sha256",
            "sampling_and_scoring_seconds",
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
                for field in (f"{metric}_mean", f"{metric}_sample_std")
            ],
        ]
        aggregate_fields = [
            "scope",
            "model_seed_index",
            "diffusion_sample_index",
            "num_chains",
            "num_samples_per_chain",
            "aggregation_policy",
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
        write_csv(output_dir / "per-sample-metrics.csv", sample_rows, sample_fields)
        write_csv(output_dir / "per-chain-metrics.csv", chain_rows, chain_fields)
        write_csv(
            output_dir / "aggregate-metrics.csv",
            aggregate["rows"],
            aggregate_fields,
        )
        summary = {
            "schema_version": 1,
            "architecture": "Protenix AF3-style NanoFold reconstruction",
            "architecture_id": ARCHITECTURE_ID,
            "official_alphafold3": False,
            "claim_bearing_full_public_validation": claim_bearing,
            "checkpoint_completed_step": checkpoint_verification["completed_step"],
            "checkpoint_sha256": checkpoint_verification["sha256"],
            "evaluation_contract_sha256": evaluation_contract["sha256"],
            "num_chains": len(chain_ids),
            "model_seeds": model_seeds,
            "diffusion_samples_per_model_seed": diffusion_samples,
            "num_samples_per_chain": total_samples,
            "world_size": world_size,
            "evaluation_selection": evaluation_selection,
            "aggregation_policy": AGGREGATION_POLICY,
            "primary": aggregate["primary"],
            "output_files": {
                "per_sample_metrics": "per-sample-metrics.csv",
                "per_chain_metrics": "per-chain-metrics.csv",
                "aggregate_metrics": "aggregate-metrics.csv",
                "provenance": "provenance.json",
                "prediction_root": "predictions",
                "atomic_record_root": "records",
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
