from __future__ import annotations

import argparse
import json
import math
import sys
import time
from functools import partial
from pathlib import Path
from typing import Any, Dict, Iterable

import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from nanofold.competition_policy import (
    DEFAULT_TRACK_ID,
    OFFICIAL_DATASET_FINGERPRINT_PATH,
    TrackSpec,
    apply_track_policy,
    assert_track_policy,
    compute_effective_batch_size,
    compute_residue_budget,
    compute_sample_budget,
    enforce_model_param_limit,
    load_track_spec,
)
from nanofold.data import ProcessedNPZDataset, collate_batch
from nanofold.dataset_integrity import verify_dataset_against_fingerprint
from nanofold.metrics import lddt_ca
from nanofold.submission_runtime import (
    load_submission_hooks,
    run_submission_batch,
)
from nanofold.utils import (
    RunPaths,
    count_parameters,
    default_torch_device,
    ensure_dir,
    get_env_metadata,
    load_torch_checkpoint,
    make_dataloader_generator,
    seed_worker,
    serialize_numpy_rng_state,
    set_seed,
    should_pin_memory,
    to_device,
    utc_now_iso,
)
from nanofold.utils import (
    sha256_file as _sha256_file,
)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--track", type=str, default=DEFAULT_TRACK_ID, help=f"Track id (default: {DEFAULT_TRACK_ID})")
    ap.add_argument("--official", action="store_true", help="Enable strict official track enforcement.")
    ap.add_argument(
        "--fingerprint",
        type=str,
        default="",
        help="Expected dataset fingerprint JSON path (defaults to track fingerprint).",
    )
    ap.add_argument(
        "--verify-fingerprint",
        action="store_true",
        help="Verify dataset fingerprint even outside official mode.",
    )
    ap.add_argument(
        "--processed-features-dir",
        type=str,
        default="",
        help="Runtime override for data.processed_features_dir.",
    )
    ap.add_argument(
        "--processed-labels-dir",
        type=str,
        default="",
        help="Runtime override for data.processed_labels_dir.",
    )
    ap.add_argument(
        "--resume",
        type=str,
        default="",
        help="Optional checkpoint path to resume from.",
    )
    ap.add_argument(
        "--reset-run",
        action="store_true",
        help="Clear an existing run directory before starting from step 0.",
    )
    ap.add_argument(
        "--deterministic",
        action="store_true",
        help="Enable deterministic training settings (can reduce speed).",
    )
    ap.add_argument(
        "--allow-resume-mismatch",
        action="store_true",
        help="Allow resume metadata mismatches (maintainer-only override).",
    )
    return ap.parse_args()


def load_config(path: str | Path) -> Dict[str, Any]:
    return yaml.safe_load(Path(path).read_text())


def make_autocast_ctx(device: torch.device, enabled: bool):
    try:
        amp = getattr(torch, "amp")
        return amp.autocast(device_type=device.type, enabled=enabled)
    except Exception:
        return torch.cuda.amp.autocast(enabled=enabled)


def make_grad_scaler(enabled: bool):
    try:
        amp = getattr(torch, "amp")
        return amp.GradScaler("cuda", enabled=enabled)
    except Exception:
        return torch.cuda.amp.GradScaler(enabled=enabled)


def empty_device_cache(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.empty_cache()
        return
    if device.type == "mps":
        mps = getattr(torch, "mps", None)
        if mps is not None and hasattr(mps, "empty_cache"):
            mps.empty_cache()


def normalize_num_workers(n: int) -> int:
    n = int(n)
    if n <= 0:
        return 0
    if sys.platform == "darwin" and sys.version_info >= (3, 13):
        print("Forcing data.num_workers=0 on macOS with Python 3.13+ for DataLoader stability.")
        return 0
    return n


def _mean_tensors(values: Iterable[torch.Tensor]) -> float:
    finite_values = [
        value.float().reshape(())
        for value in values
        if torch.isfinite(value.float().reshape(()))
    ]
    if not finite_values:
        return float("nan")
    return float(torch.stack(finite_values).mean())


def _summarize_eval_metrics(
    scores: list[torch.Tensor],
    losses: list[torch.Tensor],
    ca_rmsds: list[torch.Tensor] | None = None,
    atom14_rmsds: list[torch.Tensor] | None = None,
) -> Dict[str, float]:
    metrics = {
        "val_lddt_ca": _mean_tensors(scores),
    }
    if losses:
        metrics["val_loss"] = _mean_tensors(losses)
    if ca_rmsds is not None:
        metrics["val_rmsd_ca"] = _mean_tensors(ca_rmsds)
    if atom14_rmsds is not None:
        metrics["val_rmsd_atom14"] = _mean_tensors(atom14_rmsds)
    return metrics


def _cfg_with_runtime(
    cfg: Dict[str, Any],
    *,
    step: int,
    cumulative_samples_seen: int,
    max_steps: int,
    sample_budget: int,
) -> Dict[str, Any]:
    runtime_cfg = dict(cfg)
    runtime_cfg["_runtime"] = {
        "step": int(step),
        "cumulative_samples_seen": int(cumulative_samples_seen),
        "max_steps": int(max_steps),
        "sample_budget": int(sample_budget),
    }
    return runtime_cfg


def _guidance_for_missing_data(track_spec: TrackSpec) -> str:
    return (
        "Official mode requires fully preprocessed data for every chain in the official manifests.\n"
        f"Track: {track_spec.track_id}\n"
        "Run:\n"
        "  bash scripts/setup_official_data.sh\n"
        "or preprocess any missing chains listed in the error message."
    )


def resume_metadata_mismatches(
    *,
    ckpt_obj: Dict[str, Any],
    submission_entrypoint_sha256: str | None,
    config_sha256: str,
    track_id: str,
    fingerprint_sha256: str | None,
    n_params: int,
) -> list[str]:
    expected_pairs = {
        "submission_entrypoint_sha256": submission_entrypoint_sha256,
        "config_sha256": config_sha256,
        "track_id": track_id,
        "fingerprint_sha256": fingerprint_sha256,
        "n_params": n_params,
    }
    mismatches: list[str] = []
    for key, expected in expected_pairs.items():
        actual = ckpt_obj.get(key)
        if actual != expected:
            mismatches.append(f"{key}: expected={expected!r}, actual={actual!r}")
    return mismatches


def make_loader(
    cfg: Dict[str, Any],
    split: str,
    *,
    device: torch.device,
    include_labels: bool,
    fail_if_labels_present: bool,
    allow_missing: bool,
    generator_seed: int,
) -> DataLoader:
    data_cfg = cfg["data"]
    processed_features_dir = data_cfg["processed_features_dir"]
    processed_labels_dir = data_cfg.get("processed_labels_dir")
    manifest_path = data_cfg["train_manifest"] if split == "train" else data_cfg["val_manifest"]
    if split == "train":
        crop_mode = str(data_cfg.get("train_crop_mode", "random"))
        msa_sample_mode = str(data_cfg.get("train_msa_sample_mode", "random"))
    else:
        crop_mode = str(data_cfg.get("val_crop_mode", "center"))
        msa_sample_mode = str(data_cfg.get("val_msa_sample_mode", "top"))

    ds = ProcessedNPZDataset(
        processed_features_dir=processed_features_dir,
        processed_labels_dir=processed_labels_dir,
        include_labels=include_labels,
        fail_if_labels_present=fail_if_labels_present,
        manifest_path=manifest_path,
        allow_missing=allow_missing,
    )
    if getattr(ds, "missing_chain_ids", None):
        print(
            f"[{split}] Skipping {len(ds.missing_chain_ids)} missing preprocessed chains "
            f"(first: {', '.join(ds.missing_chain_ids[:6])})"
        )
    collate_fn = partial(
        collate_batch,
        crop_size=int(data_cfg["crop_size"]),
        msa_depth=int(data_cfg["msa_depth"]),
        crop_mode=crop_mode,
        msa_sample_mode=msa_sample_mode,
    )
    num_workers = normalize_num_workers(int(data_cfg.get("num_workers", 0)))
    generator = make_dataloader_generator(generator_seed)

    return DataLoader(
        ds,
        batch_size=data_cfg.get("batch_size", 1),
        shuffle=(split == "train"),
        num_workers=num_workers,
        pin_memory=should_pin_memory(device),
        collate_fn=collate_fn,
        drop_last=(split == "train"),
        worker_init_fn=seed_worker if num_workers > 0 else None,
        generator=generator,
    )


@torch.no_grad()
def batch_lddt_ca(
    pred_ca: torch.Tensor,
    true_ca: torch.Tensor,
    ca_mask: torch.Tensor,
    residue_mask: torch.Tensor,
) -> torch.Tensor:
    scores = []
    B = pred_ca.shape[0]
    for b in range(B):
        scores.append(lddt_ca(pred_ca[b], true_ca[b], ca_mask[b] & residue_mask[b]))
    return torch.stack(scores).mean()


@torch.no_grad()
def masked_kabsch_rmsd(
    pred_points: torch.Tensor,
    true_points: torch.Tensor,
    point_mask: torch.Tensor,
) -> torch.Tensor:
    if pred_points.shape != true_points.shape:
        raise ValueError("`pred_points` and `true_points` must have the same shape.")
    if pred_points.ndim != 2 or pred_points.shape[-1] != 3:
        raise ValueError(f"`pred_points` must have shape (N, 3), got {tuple(pred_points.shape)}")
    if point_mask.ndim != 1 or point_mask.shape[0] != pred_points.shape[0]:
        raise ValueError("`point_mask` must have shape (N,).")

    device = torch.device("cpu")
    pred_points = pred_points.detach().float().cpu()
    true_points = true_points.detach().float().cpu()
    point_mask = point_mask.detach().cpu()
    mask = point_mask.to(device=device, dtype=torch.bool)
    if int(mask.sum().item()) < 3:
        return torch.full((), float("nan"), device=device, dtype=pred_points.dtype)

    pred = pred_points[mask]
    true = true_points[mask]
    pred_centered = pred - pred.mean(dim=0, keepdim=True)
    true_centered = true - true.mean(dim=0, keepdim=True)

    covariance = pred_centered.transpose(0, 1) @ true_centered
    u, _, vh = torch.linalg.svd(covariance, full_matrices=False)
    correction = torch.ones(3, device=device, dtype=pred_centered.dtype)
    if torch.det(u @ vh) < 0:
        correction[-1] = -1.0
    rotation = u @ torch.diag(correction) @ vh
    aligned = pred_centered @ rotation
    squared_error = (aligned - true_centered).square().sum(dim=-1)
    return torch.sqrt(squared_error.mean().clamp_min(0.0))


@torch.no_grad()
def batch_rmsd_ca(
    pred_ca: torch.Tensor,
    true_ca: torch.Tensor,
    ca_mask: torch.Tensor,
    residue_mask: torch.Tensor,
) -> torch.Tensor:
    rmsds = [
        masked_kabsch_rmsd(
            pred_ca[b],
            true_ca[b],
            ca_mask[b] & residue_mask[b],
        )
        for b in range(pred_ca.shape[0])
    ]
    return torch.nanmean(torch.stack(rmsds))


@torch.no_grad()
def batch_rmsd_atom14(
    pred_atom14: torch.Tensor,
    true_atom14: torch.Tensor,
    atom14_mask: torch.Tensor,
    residue_mask: torch.Tensor,
) -> torch.Tensor:
    rmsds = []
    for b in range(pred_atom14.shape[0]):
        mask = atom14_mask[b] & residue_mask[b, :, None]
        rmsds.append(
            masked_kabsch_rmsd(
                pred_atom14[b].reshape(-1, 3),
                true_atom14[b].reshape(-1, 3),
                mask.reshape(-1),
            )
        )
    return torch.nanmean(torch.stack(rmsds))


def optimizer_zero_grad(optimizer: Any) -> None:
    try:
        optimizer.zero_grad(set_to_none=True)
    except TypeError:
        optimizer.zero_grad()


def _resolve_fingerprint_path(args: argparse.Namespace, track_spec: TrackSpec) -> str:
    if args.fingerprint:
        return args.fingerprint
    if track_spec.fingerprint_path:
        return track_spec.fingerprint_path
    if args.official or args.verify_fingerprint:
        raise ValueError(
            f"Track `{track_spec.track_id}` does not define a fingerprint path. "
            "Pass --fingerprint explicitly."
        )
    return OFFICIAL_DATASET_FINGERPRINT_PATH


def _grad_norm(model: torch.nn.Module) -> float:
    total = 0.0
    for p in model.parameters():
        if p.grad is None:
            continue
        gn = float(torch.linalg.vector_norm(p.grad.detach(), ord=2).item())
        total += gn * gn
    return math.sqrt(total) if total > 0 else 0.0


def _first_nonfinite_gradient(model: torch.nn.Module) -> str | None:
    for name, param in model.named_parameters():
        if param.grad is not None and not torch.isfinite(param.grad).all():
            return name
    return None


def _current_lr(optimizer: Any) -> float:
    groups = getattr(optimizer, "param_groups", None)
    if isinstance(groups, list) and groups:
        try:
            return float(groups[0]["lr"])
        except Exception:
            return float("nan")
    return float("nan")


def _format_duration(seconds: float) -> str:
    if not math.isfinite(seconds) or seconds < 0:
        return "unknown"
    seconds_i = int(round(seconds))
    hours, rem = divmod(seconds_i, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{secs:02d}s"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


def _format_train_status(
    *,
    step_record: Dict[str, Any],
    max_steps: int,
    run_elapsed_seconds: float,
    steps_completed_this_run: int,
) -> str:
    step = int(step_record["step"])
    step_fraction = step / float(max_steps) if max_steps > 0 else float("nan")
    sample_fraction = float(step_record["sample_budget_fraction"])
    residue_fraction = float(step_record["nonpad_residue_budget_fraction"])
    avg_step_seconds = (
        run_elapsed_seconds / float(steps_completed_this_run)
        if steps_completed_this_run > 0
        else float("nan")
    )
    eta_seconds = max(max_steps - step, 0) * avg_step_seconds
    return (
        f"[train] step {step}/{max_steps} ({step_fraction:.1%}) "
        f"loss={float(step_record['train_loss']):.4f} "
        f"lr={float(step_record['lr']):.3e} "
        f"grad_norm={float(step_record['grad_norm']):.4f} "
        f"samples={int(step_record['cumulative_samples_seen'])} ({sample_fraction:.1%}) "
        f"residues={int(step_record['cumulative_nonpad_residues_seen'])} ({residue_fraction:.1%}) "
        f"step_time={float(step_record['step_seconds']):.2f}s "
        f"samples/s={float(step_record['samples_per_sec']):.2f} "
        f"elapsed={_format_duration(run_elapsed_seconds)} "
        f"eta={_format_duration(eta_seconds)}"
    )


def _format_eval_status(val_metrics: Dict[str, Any]) -> str:
    return (
        f"[eval] step {int(val_metrics['step'])} "
        f"val_loss={float(val_metrics.get('val_loss', float('nan'))):.4f} "
        f"val_lddt_ca={float(val_metrics.get('val_lddt_ca', float('nan'))):.4f} "
        f"val_rmsd_ca={float(val_metrics.get('val_rmsd_ca', float('nan'))):.3f} "
        f"val_rmsd_atom14={float(val_metrics.get('val_rmsd_atom14', float('nan'))):.3f} "
        f"train_loss={float(val_metrics.get('train_loss', float('nan'))):.4f}"
    )


def _reset_run_outputs(paths: RunPaths, step_metrics_path: Path) -> None:
    for path in (paths.metrics_path, paths.log_path, step_metrics_path):
        if path.exists():
            path.unlink()
    for path in paths.ckpt_dir.glob("ckpt_step_*.pt"):
        path.unlink()
    last_ckpt = paths.ckpt_dir / "ckpt_last.pt"
    if last_ckpt.exists():
        last_ckpt.unlink()


def _truncate_step_metrics(path: Path, *, max_step: int) -> int:
    if not path.exists():
        return 0
    kept_lines: list[str] = []
    removed = 0
    for line in path.read_text().splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        try:
            record = json.loads(stripped)
            record_step = int(record.get("step", max_step))
        except Exception:
            kept_lines.append(line)
            continue
        if record_step <= max_step:
            kept_lines.append(line)
        else:
            removed += 1
    if removed:
        path.write_text("\n".join(kept_lines) + ("\n" if kept_lines else ""))
    return removed


def _truncate_metric_history(metrics: Dict[str, Any], *, max_step: int) -> int:
    history = metrics.get("history")
    if not isinstance(history, list):
        return 0
    kept_history = []
    removed = 0
    for item in history:
        if not isinstance(item, dict):
            kept_history.append(item)
            continue
        try:
            item_step = int(item.get("step", max_step))
        except Exception:
            kept_history.append(item)
            continue
        if item_step <= max_step:
            kept_history.append(item)
        else:
            removed += 1
    if removed:
        metrics["history"] = kept_history
    return removed


def _verify_dataset(
    *,
    cfg: Dict[str, Any],
    fingerprint_path: str,
    require_no_missing: bool,
    track_id: str | None = None,
) -> None:
    data_cfg = cfg["data"]
    verify_dataset_against_fingerprint(
        processed_features_dir=data_cfg["processed_features_dir"],
        processed_labels_dir=data_cfg.get("processed_labels_dir"),
        train_manifest=data_cfg["train_manifest"],
        val_manifest=data_cfg["val_manifest"],
        expected_fingerprint_path=fingerprint_path,
        require_no_missing=require_no_missing,
        track_id=track_id,
    )


def main() -> None:
    args = parse_args()
    if args.reset_run and args.resume:
        raise ValueError("`--reset-run` cannot be combined with `--resume`.")

    config_path = Path(args.config).resolve()
    raw_cfg = load_config(args.config)
    track_spec = load_track_spec(args.track)
    cfg = apply_track_policy(raw_cfg, track_spec=track_spec) if args.official else raw_cfg
    data_cfg = cfg.setdefault("data", {})
    if args.processed_features_dir:
        data_cfg["processed_features_dir"] = args.processed_features_dir
    if args.processed_labels_dir:
        data_cfg["processed_labels_dir"] = args.processed_labels_dir

    deterministic = bool(args.deterministic or args.official)

    fingerprint_path = _resolve_fingerprint_path(args, track_spec)
    if args.official:
        assert_track_policy(
            cfg=cfg,
            track_spec=track_spec,
            enforce_manifest_paths=True,
            enforce_manifest_hashes=True,
        )
        print(
            f"Official mode enabled for track `{track_spec.track_id}`. "
            f"Verifying dataset fingerprint: {Path(fingerprint_path).resolve()}",
            flush=True,
        )
        try:
            _verify_dataset(
                cfg=cfg,
                fingerprint_path=fingerprint_path,
                require_no_missing=True,
                track_id=track_spec.track_id,
            )
        except FileNotFoundError as exc:
            raise RuntimeError(f"{exc}\n\n{_guidance_for_missing_data(track_spec)}") from exc
        print(f"Dataset fingerprint matched: {Path(fingerprint_path).resolve()}", flush=True)
    elif args.verify_fingerprint:
        _verify_dataset(
            cfg=cfg,
            fingerprint_path=fingerprint_path,
            require_no_missing=False,
            track_id=track_spec.track_id,
        )
        print(f"Fingerprint verification succeeded: {Path(fingerprint_path).resolve()}")

    hooks = load_submission_hooks(cfg, config_path, allowed_root=config_path.parent)
    run_name = cfg.get("run_name", "run")
    paths = RunPaths.from_run_name(run_name)
    ensure_dir(paths.run_dir)
    ensure_dir(paths.ckpt_dir)
    step_metrics_path = paths.run_dir / "train_metrics.jsonl"
    if args.reset_run:
        _reset_run_outputs(paths, step_metrics_path)

    seed = int(cfg.get("seed", 0))
    set_seed(seed, deterministic=deterministic)

    device = default_torch_device()
    env_meta = get_env_metadata(device)
    print(f"Using device: {device}")

    allow_missing = not bool(args.official)
    try:
        train_loader = make_loader(
            cfg,
            split="train",
            device=device,
            include_labels=True,
            fail_if_labels_present=False,
            allow_missing=allow_missing,
            generator_seed=seed,
        )
        val_loader = make_loader(
            cfg,
            split="val",
            device=device,
            include_labels=True,
            fail_if_labels_present=False,
            allow_missing=allow_missing,
            generator_seed=seed + 1,
        )
    except FileNotFoundError as exc:
        if args.official:
            raise RuntimeError(f"{exc}\n\n{_guidance_for_missing_data(track_spec)}") from exc
        raise

    model = hooks.build_model(cfg)
    if not isinstance(model, nn.Module):
        raise TypeError("`build_model(cfg)` must return a torch.nn.Module")
    model = model.to(device)

    n_params = count_parameters(model)
    print(f"Model params: {n_params:,}")
    print(f"Submission module: {hooks.module_ref}")
    if args.official:
        enforce_model_param_limit(track_spec=track_spec, n_params=n_params)

    opt = hooks.build_optimizer(cfg, model)
    if not callable(getattr(opt, "step", None)) or not callable(getattr(opt, "zero_grad", None)):
        raise TypeError("`build_optimizer(cfg, model)` must return an optimizer-like object with `step` and `zero_grad`.")
    if not callable(getattr(opt, "state_dict", None)):
        raise TypeError("Optimizer must implement `state_dict()` for checkpointing.")

    scheduler = hooks.build_scheduler(cfg, opt) if hooks.build_scheduler is not None else None
    if scheduler is not None and not callable(getattr(scheduler, "step", None)):
        raise TypeError("`build_scheduler` must return an object with `step()` or None.")

    tcfg = cfg["train"]
    max_steps = int(tcfg["max_steps"])
    log_every = int(tcfg.get("log_every", 50))
    eval_every = int(tcfg.get("eval_every", 500))
    save_every = int(tcfg.get("save_every", 500))
    grad_accum_steps = int(tcfg.get("grad_accum_steps", 1))
    if log_every <= 0:
        raise ValueError("`train.log_every` must be >= 1")
    if eval_every <= 0:
        raise ValueError("`train.eval_every` must be >= 1")
    if save_every <= 0:
        raise ValueError("`train.save_every` must be >= 1")
    if grad_accum_steps <= 0:
        raise ValueError("`train.grad_accum_steps` must be >= 1")
    grad_clip = float(cfg.get("optim", {}).get("grad_clip_norm", 0.0))
    use_amp = bool(tcfg.get("amp", False)) and device.type == "cuda"

    scaler = make_grad_scaler(enabled=use_amp)

    batch_size = int(cfg["data"]["batch_size"])
    crop_size = int(cfg["data"]["crop_size"])
    effective_batch_size = compute_effective_batch_size(batch_size, grad_accum_steps)
    sample_budget = compute_sample_budget(max_steps, effective_batch_size)
    residue_budget = compute_residue_budget(max_steps, effective_batch_size, crop_size)
    config_sha256 = _sha256_file(config_path)
    fingerprint_sha256 = (
        _sha256_file(fingerprint_path) if (args.official or args.verify_fingerprint) and Path(fingerprint_path).exists() else None
    )

    metrics: Dict[str, Any] = {
        "run_name": run_name,
        "track": track_spec.track_id,
        "official_mode": bool(args.official),
        "seed": seed,
        "deterministic": deterministic,
        "n_params": n_params,
        "submission_module": hooks.module_ref,
        "submission_entrypoint_path": hooks.source_path,
        "submission_entrypoint_sha256": hooks.source_sha256,
        "config_path": str(config_path),
        "config_sha256": config_sha256,
        "config": cfg,
        "effective_batch_size": effective_batch_size,
        "sample_budget": sample_budget,
        "residue_budget": residue_budget,
        "fingerprint_path": str(Path(fingerprint_path).resolve()) if (args.official or args.verify_fingerprint) else None,
        "fingerprint_sha256": fingerprint_sha256,
        "env": env_meta,
        "started_at": utc_now_iso(),
        "updated_at": utc_now_iso(),
        "history": [],
        "step_metrics_jsonl": str(step_metrics_path),
        "cumulative_samples_seen": 0,
        "cumulative_cropped_residues_seen": 0,
        "cumulative_nonpad_residues_seen": 0,
        "cumulative_residues_seen": 0,
    }

    if args.reset_run:
        print(f"Reset run outputs: {paths.run_dir}")
    if not args.resume:
        step_metrics_path.write_text("")
        Path(paths.metrics_path).write_text(json.dumps(metrics, indent=2))

    start_step = 0
    cumulative_samples_seen = 0
    cumulative_cropped_residues_seen = 0
    cumulative_nonpad_residues_seen = 0

    def _checkpoint_payload(*, step_value: int) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "step": step_value,
            "model": model.state_dict(),
            "opt": opt.state_dict(),
            "submission_module": hooks.module_ref,
            "submission_entrypoint_path": hooks.source_path,
            "submission_entrypoint_sha256": hooks.source_sha256,
            "config": cfg,
            "config_sha256": config_sha256,
            "track_id": track_spec.track_id,
            "n_params": n_params,
            "effective_batch_size": effective_batch_size,
            "sample_budget": sample_budget,
            "residue_budget": residue_budget,
            "fingerprint_path": str(Path(fingerprint_path).resolve()) if (args.official or args.verify_fingerprint) else None,
            "fingerprint_sha256": fingerprint_sha256,
            "scaler": scaler.state_dict() if use_amp else None,
            "cumulative_samples_seen": cumulative_samples_seen,
            "cumulative_cropped_residues_seen": cumulative_cropped_residues_seen,
            "cumulative_nonpad_residues_seen": cumulative_nonpad_residues_seen,
            "cumulative_residues_seen": cumulative_nonpad_residues_seen,
            "rng_state": {
                "python": __import__("random").getstate(),
                "numpy": serialize_numpy_rng_state(__import__("numpy").random.get_state()),
                "torch": torch.get_rng_state(),
                "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            },
        }
        if scheduler is not None and callable(getattr(scheduler, "state_dict", None)):
            payload["scheduler"] = scheduler.state_dict()
        return payload

    def _save_checkpoint_files(*, step_value: int) -> None:
        ckpt = _checkpoint_payload(step_value=step_value)
        ckpt_path = paths.ckpt_dir / f"ckpt_step_{step_value}.pt"
        torch.save(ckpt, ckpt_path)
        torch.save(ckpt, paths.ckpt_dir / "ckpt_last.pt")
    if args.resume:
        resume_path = Path(args.resume)
        if not resume_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
        ckpt = load_torch_checkpoint(resume_path, map_location="cpu")
        resume_mismatches = resume_metadata_mismatches(
            ckpt_obj=ckpt,
            submission_entrypoint_sha256=hooks.source_sha256,
            config_sha256=config_sha256,
            track_id=track_spec.track_id,
            fingerprint_sha256=fingerprint_sha256,
            n_params=n_params,
        )
        if resume_mismatches and not args.allow_resume_mismatch:
            joined = "\n".join(f"- {item}" for item in resume_mismatches)
            raise ValueError(
                "Resume checkpoint metadata does not match the current run.\n"
                "Pass --allow-resume-mismatch to override.\n"
                f"{joined}"
            )
        model.load_state_dict(ckpt["model"], strict=True)
        opt.load_state_dict(ckpt["opt"])
        if scheduler is not None and "scheduler" in ckpt and callable(getattr(scheduler, "load_state_dict", None)):
            scheduler.load_state_dict(ckpt["scheduler"])
        if use_amp and "scaler" in ckpt:
            try:
                scaler.load_state_dict(ckpt["scaler"])
            except Exception:
                pass
        start_step = int(ckpt.get("step", 0))
        cumulative_samples_seen = int(ckpt.get("cumulative_samples_seen", start_step * effective_batch_size))
        cumulative_cropped_residues_seen = int(
            ckpt.get("cumulative_cropped_residues_seen", start_step * effective_batch_size * crop_size)
        )
        cumulative_nonpad_residues_seen = int(
            ckpt.get(
                "cumulative_nonpad_residues_seen",
                ckpt.get("cumulative_residues_seen", cumulative_cropped_residues_seen),
            )
        )
        if "rng_state" in ckpt:
            rng_state = ckpt["rng_state"]
            try:
                import random as _random
                _random.setstate(rng_state["python"])
                import numpy as _np
                _np.random.set_state(rng_state["numpy"])
                torch.set_rng_state(rng_state["torch"])
                cuda_state = rng_state.get("torch_cuda")
                if cuda_state is not None and torch.cuda.is_available():
                    torch.cuda.set_rng_state_all(cuda_state)
            except Exception:
                pass
        print(f"Resumed from {resume_path} at step {start_step}")
        removed_step_metrics = _truncate_step_metrics(step_metrics_path, max_step=start_step)
        if removed_step_metrics:
            print(f"Truncated {removed_step_metrics} stale per-step metric rows newer than checkpoint step {start_step}.")
        if paths.metrics_path.exists():
            try:
                existing_metrics = json.loads(paths.metrics_path.read_text())
                if isinstance(existing_metrics, dict):
                    metrics.update(existing_metrics)
                    removed_history = _truncate_metric_history(metrics, max_step=start_step)
                    if removed_history:
                        print(f"Truncated {removed_history} stale eval history rows newer than checkpoint step {start_step}.")
                    metrics["resumed_from"] = str(resume_path.resolve())
                    metrics["updated_at"] = utc_now_iso()
                    metrics["submission_entrypoint_path"] = hooks.source_path
                    metrics["submission_entrypoint_sha256"] = hooks.source_sha256
                    metrics["config_path"] = str(config_path)
                    metrics["config_sha256"] = config_sha256
                    metrics["effective_batch_size"] = effective_batch_size
                    metrics["sample_budget"] = sample_budget
                    metrics["residue_budget"] = residue_budget
                    metrics["fingerprint_path"] = (
                        str(Path(fingerprint_path).resolve()) if (args.official or args.verify_fingerprint) else None
                    )
                    metrics["fingerprint_sha256"] = fingerprint_sha256
            except Exception:
                pass
        metrics["cumulative_samples_seen"] = cumulative_samples_seen
        metrics["cumulative_cropped_residues_seen"] = cumulative_cropped_residues_seen
        metrics["cumulative_nonpad_residues_seen"] = cumulative_nonpad_residues_seen
        metrics["cumulative_residues_seen"] = cumulative_nonpad_residues_seen
        Path(paths.metrics_path).write_text(json.dumps(metrics, indent=2))
    else:
        _save_checkpoint_files(step_value=0)

    step = start_step
    if step >= max_steps:
        print(f"Checkpoint step {step} already reached max_steps={max_steps}; nothing to do.")
        metrics["cumulative_samples_seen"] = cumulative_samples_seen
        metrics["cumulative_cropped_residues_seen"] = cumulative_cropped_residues_seen
        metrics["cumulative_nonpad_residues_seen"] = cumulative_nonpad_residues_seen
        metrics["cumulative_residues_seen"] = cumulative_nonpad_residues_seen
        metrics["finished_at"] = utc_now_iso()
        metrics["updated_at"] = utc_now_iso()
        Path(paths.metrics_path).write_text(json.dumps(metrics, indent=2))
        return

    model.train()
    print(
        "Run summary: "
        f"track={track_spec.track_id} max_steps={max_steps} "
        f"effective_batch_size={effective_batch_size} sample_budget={sample_budget} "
        f"crop_size={crop_size} residue_budget={residue_budget} "
        f"log_every={log_every} eval_every={eval_every} save_every={save_every}"
    )
    print(f"Per-step metrics: {step_metrics_path}")
    print(f"Checkpoints: {paths.ckpt_dir}")
    sys.stdout.flush()
    show_progress = sys.stderr.isatty()

    def log_line(message: str) -> None:
        if show_progress:
            tqdm.write(message)
        else:
            print(message)

    pbar = tqdm(
        total=max_steps - step,
        desc="train",
        dynamic_ncols=True,
        disable=not show_progress,
    )

    train_iter = iter(train_loader)
    step_start = time.perf_counter()
    run_start = time.perf_counter()

    def run_eval() -> Dict[str, float]:
        model.eval()
        scores = []
        losses = []
        ca_rmsds = []
        atom14_rmsds = []
        runtime_cfg = _cfg_with_runtime(
            cfg,
            step=step,
            cumulative_samples_seen=cumulative_samples_seen,
            max_steps=max_steps,
            sample_budget=sample_budget,
        )
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="val", leave=False, disable=not show_progress):
                batch = to_device(batch, device)
                with make_autocast_ctx(device=device, enabled=use_amp):
                    out = run_submission_batch(
                        hooks,
                        model=model,
                        batch=batch,
                        cfg=runtime_cfg,
                        training=False,
                        expose_supervision=True,
                    )
                pred_ca = out["pred_ca"]
                score = batch_lddt_ca(pred_ca, batch["ca_coords"], batch["ca_mask"], batch["residue_mask"])
                scores.append(score.detach().cpu())
                ca_rmsd = batch_rmsd_ca(pred_ca, batch["ca_coords"], batch["ca_mask"], batch["residue_mask"])
                atom14_rmsd = batch_rmsd_atom14(
                    out["pred_atom14"],
                    batch["atom14_positions"],
                    batch["atom14_mask"],
                    batch["residue_mask"],
                )
                ca_rmsds.append(ca_rmsd.detach().cpu())
                atom14_rmsds.append(atom14_rmsd.detach().cpu())
                if "loss" in out:
                    losses.append(out["loss"].detach().cpu())
        model.train()
        empty_device_cache(device)
        return _summarize_eval_metrics(scores, losses, ca_rmsds, atom14_rmsds)

    while step < max_steps:
        optimizer_zero_grad(opt)
        running_loss = 0.0
        samples_this_step = 0
        cropped_residues_this_step = 0
        nonpad_residues_this_step = 0
        for _ in range(grad_accum_steps):
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                batch = next(train_iter)

            batch = to_device(batch, device)
            samples_this_step += int(batch["aatype"].shape[0])
            cropped_residues_this_step += int(batch["aatype"].shape[0] * batch["aatype"].shape[1])
            nonpad_residues_this_step += int(batch["residue_mask"].sum().item())

            with make_autocast_ctx(device=device, enabled=use_amp):
                runtime_cfg = _cfg_with_runtime(
                    cfg,
                    step=step,
                    cumulative_samples_seen=cumulative_samples_seen,
                    max_steps=max_steps,
                    sample_budget=sample_budget,
                )
                out = run_submission_batch(
                    hooks,
                    model=model,
                    batch=batch,
                    cfg=runtime_cfg,
                    training=True,
                )
                raw_loss = out["loss"]
                if not raw_loss.requires_grad:
                    # Some batches may have no valid supervision and return a constant
                    # scalar loss. Anchor it to model outputs so backward() stays valid.
                    pred_atom14 = out.get("pred_atom14", None)
                    if torch.is_tensor(pred_atom14):
                        raw_loss = raw_loss + pred_atom14.sum() * 0.0
                    else:
                        raise RuntimeError(
                            "Submission returned a non-differentiable `loss` and no `pred_atom14` tensor "
                            "to anchor gradients."
                        )
                loss = raw_loss / grad_accum_steps

            scaler.scale(loss).backward()
            running_loss += float(raw_loss.detach().cpu())

        scaler.unscale_(opt)
        bad_grad = _first_nonfinite_gradient(model)
        if bad_grad is not None:
            raise RuntimeError(f"Non-finite gradient detected in `{bad_grad}` before optimizer step.")
        if grad_clip and grad_clip > 0:
            grad_norm = float(nn.utils.clip_grad_norm_(model.parameters(), grad_clip).item())
        else:
            grad_norm = _grad_norm(model)
        if not math.isfinite(grad_norm):
            raise RuntimeError("Non-finite gradient norm detected before optimizer step.")

        scaler.step(opt)
        scaler.update()
        if scheduler is not None:
            scheduler.step()

        step += 1
        pbar.update(1)
        cumulative_samples_seen += samples_this_step
        cumulative_cropped_residues_seen += cropped_residues_this_step
        cumulative_nonpad_residues_seen += nonpad_residues_this_step
        train_loss = running_loss / grad_accum_steps
        lr = _current_lr(opt)
        now = time.perf_counter()
        step_seconds = max(now - step_start, 1e-8)
        step_start = now
        run_elapsed_seconds = now - run_start
        steps_completed_this_run = step - start_step
        steps_per_sec = 1.0 / step_seconds
        theoretical_residues_per_sec = (effective_batch_size * crop_size) / step_seconds
        samples_per_sec = samples_this_step / step_seconds
        cropped_residues_per_sec = cropped_residues_this_step / step_seconds
        actual_residues_per_sec = nonpad_residues_this_step / step_seconds

        step_record = {
            "timestamp": utc_now_iso(),
            "step": step,
            "train_loss": train_loss,
            "lr": lr,
            "grad_norm": grad_norm,
            "step_seconds": step_seconds,
            "steps_per_sec": steps_per_sec,
            "samples_per_sec": samples_per_sec,
            "cropped_residues_per_sec": cropped_residues_per_sec,
            "actual_residues_per_sec": actual_residues_per_sec,
            "residues_per_sec": theoretical_residues_per_sec,
            "samples_this_step": samples_this_step,
            "cropped_residues_this_step": cropped_residues_this_step,
            "nonpad_residues_this_step": nonpad_residues_this_step,
            "residues_this_step": nonpad_residues_this_step,
            "cumulative_samples_seen": cumulative_samples_seen,
            "cumulative_cropped_residues_seen": cumulative_cropped_residues_seen,
            "cumulative_nonpad_residues_seen": cumulative_nonpad_residues_seen,
            "cumulative_residues_seen": cumulative_nonpad_residues_seen,
            "sample_budget_fraction": (cumulative_samples_seen / float(sample_budget)) if sample_budget > 0 else float("nan"),
            "cropped_residue_budget_fraction": (
                cumulative_cropped_residues_seen / float(residue_budget)
            ) if residue_budget > 0 else float("nan"),
            "nonpad_residue_budget_fraction": (
                cumulative_nonpad_residues_seen / float(residue_budget)
            ) if residue_budget > 0 else float("nan"),
            "budget_fraction": (
                cumulative_nonpad_residues_seen / float(residue_budget)
            ) if residue_budget > 0 else float("nan"),
        }
        with step_metrics_path.open("a") as f:
            f.write(json.dumps(step_record) + "\n")

        should_log = (
            step == start_step + 1
            or steps_completed_this_run <= 5
            or step % log_every == 0
            or step == max_steps
        )
        if should_log:
            pbar.set_postfix(loss=f"{train_loss:.4f}", lr=f"{lr:.3e}", gnorm=f"{grad_norm:.3f}")
            log_line(
                _format_train_status(
                    step_record=step_record,
                    max_steps=max_steps,
                    run_elapsed_seconds=run_elapsed_seconds,
                    steps_completed_this_run=steps_completed_this_run,
                )
            )
            sys.stdout.flush()

        if step % eval_every == 0 or step == max_steps:
            log_line(f"[eval] step {step}: starting public validation")
            sys.stdout.flush()
            val_metrics = run_eval()
            val_metrics["step"] = step
            val_metrics["train_loss"] = train_loss
            val_metrics["lr"] = lr
            val_metrics["grad_norm"] = grad_norm
            val_metrics["steps_per_sec"] = steps_per_sec
            val_metrics["samples_per_sec"] = samples_per_sec
            val_metrics["cropped_residues_per_sec"] = cropped_residues_per_sec
            val_metrics["actual_residues_per_sec"] = actual_residues_per_sec
            val_metrics["residues_per_sec"] = theoretical_residues_per_sec
            val_metrics["cumulative_samples_seen"] = cumulative_samples_seen
            val_metrics["cumulative_cropped_residues_seen"] = cumulative_cropped_residues_seen
            val_metrics["cumulative_nonpad_residues_seen"] = cumulative_nonpad_residues_seen
            val_metrics["sample_budget_fraction"] = (
                cumulative_samples_seen / float(sample_budget)
            ) if sample_budget > 0 else float("nan")
            val_metrics["cropped_residue_budget_fraction"] = (
                cumulative_cropped_residues_seen / float(residue_budget)
            ) if residue_budget > 0 else float("nan")
            val_metrics["nonpad_residue_budget_fraction"] = (
                cumulative_nonpad_residues_seen / float(residue_budget)
            ) if residue_budget > 0 else float("nan")
            metrics["history"].append(val_metrics)
            metrics["updated_at"] = utc_now_iso()
            metrics["cumulative_samples_seen"] = cumulative_samples_seen
            metrics["cumulative_cropped_residues_seen"] = cumulative_cropped_residues_seen
            metrics["cumulative_nonpad_residues_seen"] = cumulative_nonpad_residues_seen
            metrics["cumulative_residues_seen"] = cumulative_nonpad_residues_seen
            Path(paths.metrics_path).write_text(json.dumps(metrics, indent=2))
            log_line(_format_eval_status(val_metrics))
            sys.stdout.flush()

        if step % save_every == 0 or step == max_steps:
            _save_checkpoint_files(step_value=step)
            log_line(f"[checkpoint] step {step}: wrote {paths.ckpt_dir / 'ckpt_last.pt'}")
            sys.stdout.flush()

    pbar.close()
    metrics["cumulative_samples_seen"] = cumulative_samples_seen
    metrics["cumulative_cropped_residues_seen"] = cumulative_cropped_residues_seen
    metrics["cumulative_nonpad_residues_seen"] = cumulative_nonpad_residues_seen
    metrics["cumulative_residues_seen"] = cumulative_nonpad_residues_seen
    metrics["wall_time_seconds"] = float(time.perf_counter() - run_start)
    metrics["finished_at"] = utc_now_iso()
    metrics["updated_at"] = utc_now_iso()
    Path(paths.metrics_path).write_text(json.dumps(metrics, indent=2))
    print("Done. Metrics:", paths.metrics_path)
    print("Checkpoint:", paths.ckpt_dir / "ckpt_last.pt")


if __name__ == "__main__":
    main()
