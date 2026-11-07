#!/usr/bin/env python3
"""Custom NanoFold trainer around the pinned Protenix implementation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
import traceback
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
import torch.utils.checkpoint
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

HERE = Path(__file__).resolve().parent
ARCH_ROOT = HERE.parent
sys.path.insert(0, str(ARCH_ROOT / "common"))
sys.path.insert(0, str(HERE))

from adapter import (  # noqa: E402
    ccd_reference_sha256,
    load_protenix_ccd_reference,
    nanofold_to_protenix,
)
from checkpoint import BoundedCheckpointManager  # noqa: E402
from config import PROTENIX_COMMIT, build_protenix_config  # noqa: E402
from data import DeterministicBatchSchedule, NanoFoldTensorLoader, read_manifest  # noqa: E402
from runtime import (  # noqa: E402
    atomic_json,
    capture_rng_state,
    gather_rank_rng_states,
    initialize_distributed,
    restore_rng_state,
    seed_process,
    sha256_file,
    write_heartbeat,
)
from verify_dataset import verify_dataset_snapshot  # noqa: E402


def to_device(value: Any, device: torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device, non_blocking=True)
    if isinstance(value, dict):
        return {key: to_device(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [to_device(item, device) for item in value]
    return value


def append_metric(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--protenix-root", type=Path, required=True)
    parser.add_argument("--dataset-verification-report", type=Path, required=True)
    parser.add_argument("--max-steps", type=int, default=30_000)
    parser.add_argument("--checkpoint-interval", type=int, default=250)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--crop-size", type=int, default=256)
    parser.add_argument("--msa-depth", type=int, default=192)
    parser.add_argument("--seed", type=int, default=20_260_725)
    parser.add_argument("--resume", choices=("auto", "never", "required"), default="auto")
    parser.add_argument("--expected-checkpoint-gib", type=float, default=6.0)
    parser.add_argument("--checkpoint-budget-gib", type=float, default=100.0)
    parser.add_argument("--checkpoint-reserve-gib", type=float, default=20.0)
    parser.add_argument("--smoke-steps", type=int, default=0)
    parser.add_argument(
        "--train-chain-limit",
        type=int,
        default=0,
        help=(
            "diagnostic-only manifest-prefix size; 0 schedules the verified full "
            "10,000-chain training manifest"
        ),
    )
    parser.add_argument(
        "--save-initial-checkpoint",
        action="store_true",
        help=(
            "diagnostic subsets only: atomically preserve completed-step zero "
            "for a later fixed-seed start/end overfit comparison"
        ),
    )
    parser.add_argument("--skip-dataset-rehash", action="store_true")
    return parser.parse_args()


def rank0_guard(rank: int, action: Any) -> None:
    error = None
    if rank == 0:
        try:
            action()
        except BaseException:
            error = traceback.format_exc()
    payload = [error]
    if dist.is_initialized():
        dist.broadcast_object_list(payload, src=0)
    if payload[0] is not None:
        raise RuntimeError(f"rank-0 initialization failed:\n{payload[0]}")


def require_all_ranks_finite(total_loss: torch.Tensor) -> None:
    """Fail every rank before backward if any rank produced a non-finite loss."""

    finite = torch.isfinite(total_loss.detach()).all().to(
        device=total_loss.device, dtype=torch.int32
    )
    if dist.is_initialized():
        dist.all_reduce(finite, op=dist.ReduceOp.MIN)
    if not bool(finite.item()):
        raise FloatingPointError("at least one DDP rank produced a non-finite loss")


def collective_call(action: Any, *, context: str) -> Any:
    """Propagate rank-local preparation failures before entering DDP forward."""

    value = None
    error = None
    try:
        value = action()
    except BaseException:
        error = traceback.format_exc()
    if dist.is_initialized():
        errors: list[str | None] = [None] * dist.get_world_size()
        dist.all_gather_object(errors, error)
    else:
        errors = [error]
    failures = [
        f"rank {rank}:\n{message}"
        for rank, message in enumerate(errors)
        if message is not None
    ]
    if failures:
        raise RuntimeError(
            f"{context} failed before distributed forward:\n" + "\n".join(failures)
        )
    return value


def canonical_json_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), default=str
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_training_schedule_contract(
    full_chain_ids: tuple[str, ...],
    *,
    train_chain_limit: int,
    world_size: int,
    max_steps: int,
    smoke_steps: int,
) -> tuple[tuple[str, ...], dict[str, Any]]:
    """Resolve a diagnostic prefix only after the full dataset was verified."""

    if len(full_chain_ids) != 10_000:
        raise ValueError(
            f"expected 10,000 verified NanoFold training chains, got {len(full_chain_ids)}"
        )
    if world_size < 1:
        raise ValueError("world_size must be positive")
    if train_chain_limit < 0 or train_chain_limit > len(full_chain_ids):
        raise ValueError(
            "--train-chain-limit must be 0 or between 1 and 10,000 inclusive"
        )
    normalized_limit = train_chain_limit or None
    scheduled_chain_ids = (
        full_chain_ids
        if normalized_limit is None
        else full_chain_ids[:normalized_limit]
    )
    if len(scheduled_chain_ids) % world_size:
        raise ValueError(
            "scheduled training-chain count must be divisible by world_size"
        )
    scheduled_bytes = "".join(
        f"{chain_id}\n" for chain_id in scheduled_chain_ids
    ).encode("utf-8")
    contract = {
        "schema_version": 1,
        "selection": (
            "full_verified_manifest"
            if normalized_limit is None
            else "verified_manifest_prefix_diagnostic"
        ),
        "full_manifest_count": len(full_chain_ids),
        "scheduled_count": len(scheduled_chain_ids),
        "train_chain_limit": normalized_limit,
        "scheduled_chain_ids_sha256": hashlib.sha256(scheduled_bytes).hexdigest(),
        # This is a necessary, not sufficient, claim gate. Other run/evaluation
        # gates remain binding even when this field is true.
        "claim_eligible": (
            normalized_limit is None
            and max_steps == 30_000
        ),
    }
    return scheduled_chain_ids, contract


def require_matching_schedule_contract(
    observed: Any, expected: dict[str, Any], *, context: str
) -> None:
    """Prevent a full-manifest run from resuming a diagnostic subset or vice versa."""

    if observed != expected:
        raise ValueError(f"{context} training schedule contract mismatch")


def optimizer_scheduler_contract(
    config: Any,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
) -> dict[str, Any]:
    """Return and, when objects are supplied, verify the executed recipe."""

    expected_optimizer = "AdamW" if bool(config.adam.use_adamw) else "Adam"
    scheduler_names = {
        "af3": "AlphaFold3LRScheduler",
        "cosine_annealing": "CosineAnnealingWithWarmup",
        "constant": "ConstantLRScheduler",
    }
    if config.lr_scheduler not in scheduler_names:
        raise ValueError(f"unsupported Protenix scheduler: {config.lr_scheduler}")
    contract = {
        "implementation": "pinned_protenix_helpers",
        "optimizer": {
            "class": expected_optimizer,
            "learning_rate": float(config.adam.lr),
            "beta1": float(config.adam.beta1),
            "beta2": float(config.adam.beta2),
            "weight_decay": float(config.adam.weight_decay),
        },
        "scheduler": {
            "class": scheduler_names[str(config.lr_scheduler)],
            "name": str(config.lr_scheduler),
            "learning_rate": float(config.af3_lr_scheduler.lr),
            "warmup_steps": int(config.af3_lr_scheduler.warmup_steps),
            "decay_every_n_steps": int(
                config.af3_lr_scheduler.decay_every_n_steps
            ),
            "decay_factor": float(config.af3_lr_scheduler.decay_factor),
        },
        "gradient_clip_norm": float(config.grad_clip_norm),
        "loss_autocast_disabled": bool(config.skip_amp.loss),
    }
    if optimizer is not None:
        observed = optimizer.__class__.__name__
        if observed != expected_optimizer:
            raise TypeError(
                f"optimizer implementation mismatch: {observed} != {expected_optimizer}"
            )
        defaults = optimizer.defaults
        observed_betas = tuple(float(value) for value in defaults["betas"])
        expected_betas = (
            contract["optimizer"]["beta1"],
            contract["optimizer"]["beta2"],
        )
        if observed_betas != expected_betas:
            raise ValueError(
                f"optimizer beta mismatch: {observed_betas} != {expected_betas}"
            )
        if float(defaults["lr"]) != contract["optimizer"]["learning_rate"]:
            raise ValueError("optimizer learning rate disagrees with resolved config")
        if float(defaults["weight_decay"]) != contract["optimizer"]["weight_decay"]:
            raise ValueError("optimizer weight decay disagrees with resolved config")
    if scheduler is not None:
        observed = scheduler.__class__.__name__
        expected = contract["scheduler"]["class"]
        if observed != expected:
            raise TypeError(
                f"scheduler implementation mismatch: {observed} != {expected}"
            )
    return contract


def build_fp32_protenix_loss(
    loss_module: Any,
    config: Any,
    autocasting_disable_decorator: Any,
) -> Any:
    """Build the pinned upstream FP32 loss callable and fail closed on drift."""

    if not bool(config.skip_amp.loss):
        raise ValueError(
            "NanoFold AF3-style contract requires skip_amp.loss=True"
        )
    return autocasting_disable_decorator(True)(loss_module)


def runtime_environment(device: torch.device) -> dict[str, Any]:
    """Capture dependency/runtime versions that can affect fused GPU results."""

    payload: dict[str, Any] = {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": str(torch.__version__),
        "cuda_runtime": torch.version.cuda,
        "cudnn": (
            torch.backends.cudnn.version()
            if hasattr(torch.backends, "cudnn")
            else None
        ),
        "device_type": device.type,
    }
    if device.type == "cuda":
        payload.update(
            {
                "device_name": torch.cuda.get_device_name(device),
                "device_capability": list(torch.cuda.get_device_capability(device)),
            }
        )
    return payload


def validate_dataset_report(
    report_path: Path,
    train_manifest: Path,
    validation_manifest: Path,
    *,
    data_root: Path,
    rehash: bool,
) -> dict[str, Any]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("train_count") != 10_000 or report.get("validation_count") != 1_000:
        raise ValueError("dataset verification report does not certify 10,000/1,000")
    expected = report["manifest_sha256"]
    actual_train = sha256_file(train_manifest)
    actual_validation = sha256_file(validation_manifest)
    if expected["train"] != actual_train or expected["validation"] != actual_validation:
        raise ValueError("staged manifests disagree with dataset verification report")
    tree_digest = report.get("inventory", {}).get("tree_sha256")
    if not tree_digest:
        raise ValueError("dataset verification report lacks inventory.tree_sha256")
    result = {
        "report_path": str(report_path.resolve()),
        "report_sha256": sha256_file(report_path),
        "tree_sha256": tree_digest,
        "manifest_sha256": expected,
        "file_count": report["inventory"].get("file_count"),
        "total_bytes": report["inventory"].get("total_bytes"),
        "rehash_performed": bool(rehash),
    }
    if rehash:
        observed = verify_dataset_snapshot(data_root, report)
        result["observed_rehash"] = observed
        result["observed_rehash_sha256"] = canonical_json_sha256(observed)
    return result


def enforce_run_state(args: argparse.Namespace) -> None:
    run_dir = args.run_dir
    entries = list(run_dir.iterdir()) if run_dir.exists() else []
    latest_pointer = run_dir / "checkpoints" / "latest.json"
    if args.resume == "never" and entries:
        raise FileExistsError(
            f"--resume never refuses non-empty run directory: {run_dir}"
        )
    if args.resume in {"auto", "required"} and entries and not latest_pointer.is_file():
        raise RuntimeError(
            f"partial run has no verified latest checkpoint: {run_dir}"
        )
    if args.resume == "required" and not latest_pointer.is_file():
        raise FileNotFoundError("resume required but latest.json does not exist")
    state_path = run_dir / "run-state.json"
    if state_path.is_file():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("state") == "complete" and args.resume != "never":
            raise RuntimeError("refusing to resume a run already marked complete")


def main() -> None:
    args = parse_args()
    if args.save_initial_checkpoint and args.train_chain_limit <= 0:
        raise ValueError(
            "--save-initial-checkpoint is restricted to --train-chain-limit diagnostics"
        )
    sys.path.insert(0, str(args.protenix_root))
    rank, local_rank, world_size, device = initialize_distributed()
    if world_size != 8 and args.smoke_steps == 0:
        raise RuntimeError(f"production contract requires world_size=8, got {world_size}")
    seed_process(args.seed, rank)

    from protenix.model.loss import ProtenixLoss
    from protenix.model.protenix import Protenix
    from protenix.utils.lr_scheduler import get_lr_scheduler
    from protenix.utils.permutation.permutation import SymmetricPermutation
    from protenix.utils.torch_utils import autocasting_disable_decorator
    from protenix.utils.training import get_optimizer

    ccd_reference = load_protenix_ccd_reference()
    ccd_digest = ccd_reference_sha256(ccd_reference)

    git_head = subprocess.run(
        ["git", "-C", str(args.protenix_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if git_head != PROTENIX_COMMIT:
        raise RuntimeError(f"Protenix commit mismatch: {git_head} != {PROTENIX_COMMIT}")
    git_tree = subprocess.run(
        ["git", "-C", str(args.protenix_root), "rev-parse", "HEAD^{tree}"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    tracked_changes = subprocess.run(
        [
            "git",
            "-C",
            str(args.protenix_root),
            "status",
            "--porcelain",
            "--untracked-files=no",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if tracked_changes:
        raise RuntimeError(
            "pinned Protenix checkout has tracked working-tree changes:\n"
            f"{tracked_changes}"
        )

    train_manifest = args.data_root / "manifests" / "train.txt"
    validation_manifest = args.data_root / "manifests" / "val.txt"
    full_chain_ids = tuple(read_manifest(train_manifest))
    if len(full_chain_ids) != 10_000:
        raise ValueError(
            f"expected 10,000 NanoFold training chains, got {len(full_chain_ids)}"
        )
    loader = NanoFoldTensorLoader(
        data_root=args.data_root,
        crop_size=args.crop_size,
        msa_depth=args.msa_depth,
        seed=args.seed,
    )

    config = build_protenix_config()
    config.max_steps = args.max_steps
    config.seed = args.seed
    model = Protenix(config).to(device)
    # Templates are explicitly excluded; freeze the dormant module so DDP does
    # not wait for gradients from a branch that is intentionally never called.
    for parameter in model.template_embedder.parameters():
        parameter.requires_grad_(False)
    if world_size > 1:
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            static_graph=True,
            find_unused_parameters=False,
        )
    loss_module = ProtenixLoss(config).to(device)
    permutation = SymmetricPermutation(
        config, error_dir=str(args.run_dir / "permutation-errors")
    )
    optimizer = get_optimizer(
        config,
        model,
        param_names=config.get("finetune_params_with_substring", [""]),
    )
    scheduler = get_lr_scheduler(config, optimizer)
    execution_contract = optimizer_scheduler_contract(
        config, optimizer=optimizer, scheduler=scheduler
    )
    loss_forward = build_fp32_protenix_loss(
        loss_module, config, autocasting_disable_decorator
    )

    rank0_guard(rank, lambda: enforce_run_state(args))
    if rank == 0:
        args.run_dir.mkdir(parents=True, exist_ok=True)
    if dist.is_initialized():
        dist.barrier()
    checkpoint_manager = BoundedCheckpointManager(
        args.run_dir / "checkpoints",
        milestone_interval=3_000,
        keep_recent=2,
        expected_allocated_bytes=int(args.expected_checkpoint_gib * (1 << 30)),
        budget_bytes=int(args.checkpoint_budget_gib * (1 << 30)),
        reserve_bytes=int(args.checkpoint_reserve_gib * (1 << 30)),
        budget_root=args.run_dir,
        allow_initial_checkpoint=bool(args.save_initial_checkpoint),
    )
    # Rank 0 validates pointer+sidecar+payload checksum before any rank loads.
    rank0_guard(
        rank,
        lambda: (
            checkpoint_manager.latest_path()
            if args.resume != "never"
            and (args.run_dir / "checkpoints" / "latest.json").is_file()
            else None
        ),
    )
    start_step = 0
    checkpoint = None if args.resume == "never" else checkpoint_manager.load_latest()
    if args.resume == "required" and checkpoint is None:
        raise FileNotFoundError("resume required but no latest checkpoint exists")
    if checkpoint is not None:
        raw_model = model.module if isinstance(model, DDP) else model
        raw_model.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_step = int(checkpoint["completed_step"])
        if checkpoint["world_size"] != world_size:
            raise ValueError("resume world size differs from checkpoint")
        restore_rng_state(checkpoint["rank_rng_states"][rank])

    # The canonical report and all 10,000 manifest entries are verified before
    # any diagnostic manifest prefix is selected for scheduling.
    dataset_verification_holder: dict[str, Any] = {}

    def verify_full_dataset() -> None:
        dataset_verification_holder["value"] = validate_dataset_report(
            args.dataset_verification_report,
            train_manifest,
            validation_manifest,
            data_root=args.data_root,
            rehash=not args.skip_dataset_rehash,
        )

    rank0_guard(rank, verify_full_dataset)
    dataset_payload = [
        dataset_verification_holder.get("value") if rank == 0 else None
    ]
    if dist.is_initialized():
        dist.broadcast_object_list(dataset_payload, src=0)
    dataset_verification = dataset_payload[0]
    chain_ids, schedule_contract = build_training_schedule_contract(
        full_chain_ids,
        train_chain_limit=args.train_chain_limit,
        world_size=world_size,
        max_steps=args.max_steps,
        smoke_steps=args.smoke_steps,
    )
    schedule = DeterministicBatchSchedule(chain_ids, args.seed, world_size)

    provenance_holder: dict[str, Any] = {}

    def initialize_provenance() -> None:
        source_paths = {
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
            "common_data": ARCH_ROOT / "common" / "data.py",
            "common_runtime": ARCH_ROOT / "common" / "runtime.py",
            "common_dataset_verifier": ARCH_ROOT / "common" / "verify_dataset.py",
        }
        current = {
            "schema_version": 1,
            "protenix_commit": PROTENIX_COMMIT,
            "protenix_git_tree": git_tree,
            "protenix_tracked_worktree_clean": True,
            "manifest": str(train_manifest),
            "manifest_sha256": sha256_file(train_manifest),
            "validation_manifest_sha256": sha256_file(validation_manifest),
            "dataset_verification": dataset_verification,
            "source_sha256": {
                name: sha256_file(path) for name, path in source_paths.items()
            },
            "resolved_config_sha256": canonical_json_sha256(config.to_dict()),
            "optimizer_scheduler": execution_contract,
            "environment": runtime_environment(device),
            "ccd_reference_sha256": ccd_digest,
            "train_count": len(full_chain_ids),
            "scheduled_train_count": len(chain_ids),
            "schedule": schedule_contract,
            "world_size": world_size,
            "crop_size": args.crop_size,
            "msa_depth": args.msa_depth,
            "seed": args.seed,
            "ema": False,
            "templates": False,
            "esm": False,
            "save_initial_checkpoint": bool(args.save_initial_checkpoint),
            "start_step": start_step,
            "max_steps": args.max_steps,
        }
        current["provenance_sha256"] = canonical_json_sha256(current)
        if checkpoint is not None:
            existing_path = args.run_dir / "provenance.json"
            if not existing_path.is_file():
                raise FileNotFoundError("resumed run lacks provenance.json")
            existing = json.loads(existing_path.read_text(encoding="utf-8"))
            for key in (
                "protenix_commit",
                "protenix_git_tree",
                "manifest_sha256",
                "validation_manifest_sha256",
                "source_sha256",
                "resolved_config_sha256",
                "optimizer_scheduler",
                "environment",
                "ccd_reference_sha256",
                "save_initial_checkpoint",
            ):
                if existing.get(key) != current.get(key):
                    raise ValueError(f"resume provenance mismatch for {key}")
            require_matching_schedule_contract(
                existing.get("schedule"),
                current["schedule"],
                context="provenance resume",
            )
            if (
                existing.get("dataset_verification", {}).get("tree_sha256")
                != current["dataset_verification"]["tree_sha256"]
            ):
                raise ValueError("resume NanoFold tensor-tree digest mismatch")
            provenance_holder["value"] = existing
        else:
            atomic_json(args.run_dir / "provenance.json", current)
            provenance_holder["value"] = current
        provenance = provenance_holder["value"]
        atomic_json(
            args.run_dir / "run-state.json",
            {
                "schema_version": 1,
                "state": "running",
                "completed_step": start_step,
                "provenance_sha256": provenance["provenance_sha256"],
            },
        )

    rank0_guard(rank, initialize_provenance)
    provenance = provenance_holder.get("value", {})
    payload = [provenance if rank == 0 else None]
    if dist.is_initialized():
        dist.broadcast_object_list(payload, src=0)
    provenance = payload[0]
    if checkpoint is not None:
        require_matching_schedule_contract(
            checkpoint.get("schedule"),
            provenance["schedule"],
            context="checkpoint resume",
        )
        for key, expected in (
            ("provenance_sha256", provenance["provenance_sha256"]),
            ("protenix_git_tree", provenance["protenix_git_tree"]),
            ("resolved_config_sha256", provenance["resolved_config_sha256"]),
            ("optimizer_scheduler", provenance["optimizer_scheduler"]),
            ("environment", provenance["environment"]),
            ("ccd_reference_sha256", provenance["ccd_reference_sha256"]),
            ("source_sha256", provenance["source_sha256"]),
            (
                "save_initial_checkpoint",
                provenance["save_initial_checkpoint"],
            ),
            (
                "dataset_tree_sha256",
                provenance["dataset_verification"]["tree_sha256"],
            ),
        ):
            if checkpoint.get(key) != expected:
                raise ValueError(f"checkpoint resume mismatch for {key}")

    def save_training_checkpoint(
        completed_step: int, rank_rng_states: list[dict[str, Any]]
    ) -> None:
        raw_model = model.module if isinstance(model, DDP) else model
        checkpoint_manager.save(
            completed_step,
            {
                "schema_version": 1,
                "completed_step": completed_step,
                "samples_seen": completed_step * world_size,
                "world_size": world_size,
                "model": raw_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "rank_rng_states": rank_rng_states,
                "protenix_commit": PROTENIX_COMMIT,
                "protenix_git_tree": provenance["protenix_git_tree"],
                "manifest_sha256": sha256_file(train_manifest),
                "dataset_tree_sha256": provenance["dataset_verification"][
                    "tree_sha256"
                ],
                "source_sha256": provenance["source_sha256"],
                "schedule": provenance["schedule"],
                "resolved_config_sha256": provenance["resolved_config_sha256"],
                "optimizer_scheduler": provenance["optimizer_scheduler"],
                "environment": provenance["environment"],
                "ccd_reference_sha256": provenance["ccd_reference_sha256"],
                "provenance_sha256": provenance["provenance_sha256"],
                "save_initial_checkpoint": provenance[
                    "save_initial_checkpoint"
                ],
                "ema": None,
            },
        )

    if args.save_initial_checkpoint and checkpoint is None:
        initial_rank_rng_states = gather_rank_rng_states(capture_rng_state())
        rank0_guard(
            rank,
            lambda: save_training_checkpoint(0, initial_rank_rng_states),
        )
    write_heartbeat(
        args.run_dir,
        state="running",
        step=start_step,
        rank=rank,
        world_size=world_size,
    )

    max_steps = min(
        args.max_steps,
        start_step + args.smoke_steps if args.smoke_steps > 0 else args.max_steps,
    )
    model.train()
    for step in range(start_step, max_steps):
        def prepare_batch() -> tuple[str, int, int, dict[str, Any]]:
            chain_id, epoch, offset = schedule.chain_at(step, rank)
            source = loader.load(chain_id, train=True, step=step, rank=rank)
            batch = nanofold_to_protenix(
                source,
                seed=args.seed,
                step=step,
                rank=rank,
                augment_reference=True,
                ccd_reference=ccd_reference,
            )
            return chain_id, epoch, offset, batch

        chain_id, epoch, offset, batch = collective_call(
            prepare_batch, context=f"step {step} batch preparation"
        )
        features = to_device(batch["input_feature_dict"], device)
        label = to_device(batch["label_dict"], device)
        label_full = to_device(batch["label_full_dict"], device)
        optimizer.zero_grad(set_to_none=True)
        amp = (
            torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False)
            if device.type == "cuda"
            else nullcontext()
        )
        started = time.monotonic()
        with amp:
            prediction, label, _ = model(
                input_feature_dict=features,
                label_full_dict=label_full,
                label_dict=label,
                mode="train",
                current_step=step,
                symmetric_permutation=permutation,
                mc_dropout_apply_rate=0,
            )
            # Match pinned runner/train.py: ProtenixLoss executes with CUDA
            # autocast disabled (skip_amp.loss=True), while the model forward
            # remains under BF16 autocast.
            total_loss, metrics = loss_forward(
                feat_dict=features,
                pred_dict=prediction,
                label_dict=label,
                mode="train",
            )
        require_all_ranks_finite(total_loss)
        total_loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), float(config.grad_clip_norm)
        )
        require_all_ranks_finite(gradient_norm)
        optimizer.step()
        scheduler.step()
        completed_step = step + 1
        elapsed = time.monotonic() - started

        if completed_step % args.log_interval == 0 or completed_step == 1:
            reduced = torch.tensor(
                [float(total_loss.detach()), float(gradient_norm), elapsed],
                device=device,
            )
            if dist.is_initialized():
                dist.all_reduce(reduced)
                reduced /= world_size
            if rank == 0:
                append_metric(
                    args.run_dir / "metrics.jsonl",
                    {
                        "completed_step": completed_step,
                        "samples_seen": completed_step * world_size,
                        "loss": float(reduced[0]),
                        "gradient_norm": float(reduced[1]),
                        "seconds_per_step": float(reduced[2]),
                        "learning_rate": optimizer.param_groups[0]["lr"],
                        "epoch": epoch,
                        "offset": offset,
                    },
                )
            write_heartbeat(
                args.run_dir,
                state="running",
                step=completed_step,
                rank=rank,
                world_size=world_size,
                metrics={"loss": float(reduced[0])},
            )

        save_now = (
            completed_step % args.checkpoint_interval == 0
            or completed_step == max_steps
        )
        if save_now:
            rank_rng_states = gather_rank_rng_states(capture_rng_state())
            rank0_guard(
                rank,
                lambda: save_training_checkpoint(completed_step, rank_rng_states),
            )

    final_state = "complete" if max_steps >= args.max_steps else "smoke_complete"
    if rank == 0:
        atomic_json(
            args.run_dir / "run-state.json",
            {
                "schema_version": 1,
                "state": final_state,
                "completed_step": max_steps,
                "provenance_sha256": provenance["provenance_sha256"],
            },
        )
    write_heartbeat(
        args.run_dir,
        state=final_state,
        step=max_steps,
        rank=rank,
        world_size=world_size,
    )
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
