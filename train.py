from __future__ import annotations

import argparse
import json
import math
import time
from functools import partial
from pathlib import Path
import sys
from typing import Any, Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml

from nanofold.competition_policy import (
    DEFAULT_TRACK_ID,
    OFFICIAL_DATASET_FINGERPRINT_PATH,
    TrackSpec,
    assert_config_matches_track,
    compute_effective_batch_size,
    compute_residue_budget,
    load_track_spec,
)
from nanofold.data import ProcessedNPZDataset, collate_batch
from nanofold.dataset_integrity import verify_dataset_against_fingerprint
from nanofold.metrics import lddt_ca
from nanofold.submission_runtime import (
    load_submission_hooks,
    run_submission_batch,
    strip_supervision_from_batch,
)
from nanofold.utils import (
    RunPaths,
    count_parameters,
    ensure_dir,
    get_env_metadata,
    make_dataloader_generator,
    seed_worker,
    set_seed,
    to_device,
    utc_now_iso,
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
        "--resume",
        type=str,
        default="",
        help="Optional checkpoint path to resume from.",
    )
    ap.add_argument(
        "--deterministic",
        action="store_true",
        help="Enable deterministic training settings (can reduce speed).",
    )
    return ap.parse_args()


def load_config(path: str | Path) -> Dict[str, Any]:
    return yaml.safe_load(Path(path).read_text())


def make_autocast_ctx(device: torch.device, enabled: bool):
    try:
        return torch.amp.autocast(device_type=device.type, enabled=enabled)
    except Exception:
        return torch.cuda.amp.autocast(enabled=enabled)


def make_grad_scaler(enabled: bool):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except Exception:
        return torch.cuda.amp.GradScaler(enabled=enabled)


def normalize_num_workers(n: int) -> int:
    n = int(n)
    if n <= 0:
        return 0
    if sys.platform == "darwin" and sys.version_info >= (3, 13):
        print("Forcing data.num_workers=0 on macOS with Python 3.13+ for DataLoader stability.")
        return 0
    return n


def _guidance_for_missing_data(track_spec: TrackSpec) -> str:
    return (
        "Official mode requires fully preprocessed data for every chain in the official manifests.\n"
        f"Track: {track_spec.track_id}\n"
        "Run:\n"
        "  bash scripts/setup_official_data.sh\n"
        "or preprocess any missing chains listed in the error message."
    )


def make_loader(
    cfg: Dict[str, Any],
    split: str,
    *,
    allow_missing: bool,
    generator_seed: int,
) -> DataLoader:
    data_cfg = cfg["data"]
    processed_dir = data_cfg["processed_dir"]
    manifest_path = data_cfg["train_manifest"] if split == "train" else data_cfg["val_manifest"]
    if split == "train":
        crop_mode = str(data_cfg.get("train_crop_mode", "random"))
        msa_sample_mode = str(data_cfg.get("train_msa_sample_mode", "random"))
    else:
        crop_mode = str(data_cfg.get("val_crop_mode", "center"))
        msa_sample_mode = str(data_cfg.get("val_msa_sample_mode", "top"))

    ds = ProcessedNPZDataset(processed_dir=processed_dir, manifest_path=manifest_path, allow_missing=allow_missing)
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
        pin_memory=torch.cuda.is_available(),
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


def _current_lr(optimizer: Any) -> float:
    groups = getattr(optimizer, "param_groups", None)
    if isinstance(groups, list) and groups:
        try:
            return float(groups[0]["lr"])
        except Exception:
            return float("nan")
    return float("nan")


def _verify_dataset(
    *,
    cfg: Dict[str, Any],
    fingerprint_path: str,
    require_no_missing: bool,
) -> None:
    data_cfg = cfg["data"]
    verify_dataset_against_fingerprint(
        processed_dir=data_cfg["processed_dir"],
        train_manifest=data_cfg["train_manifest"],
        val_manifest=data_cfg["val_manifest"],
        expected_fingerprint_path=fingerprint_path,
        require_no_missing=require_no_missing,
    )


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    track_spec = load_track_spec(args.track)

    deterministic = bool(args.deterministic or args.official)

    fingerprint_path = _resolve_fingerprint_path(args, track_spec)
    if args.official:
        assert_config_matches_track(cfg, track_spec=track_spec, enforce_manifest_paths=True)
        try:
            _verify_dataset(
                cfg=cfg,
                fingerprint_path=fingerprint_path,
                require_no_missing=True,
            )
        except FileNotFoundError as exc:
            raise RuntimeError(f"{exc}\n\n{_guidance_for_missing_data(track_spec)}") from exc
        print(
            f"Official mode enabled for track `{track_spec.track_id}`. "
            f"Dataset fingerprint matched: {Path(fingerprint_path).resolve()}"
        )
    elif args.verify_fingerprint:
        _verify_dataset(
            cfg=cfg,
            fingerprint_path=fingerprint_path,
            require_no_missing=False,
        )
        print(f"Fingerprint verification succeeded: {Path(fingerprint_path).resolve()}")

    hooks = load_submission_hooks(cfg, args.config)

    run_name = cfg.get("run_name", "run")
    paths = RunPaths.from_run_name(run_name)
    ensure_dir(paths.run_dir)
    ensure_dir(paths.ckpt_dir)
    step_metrics_path = paths.run_dir / "train_metrics.jsonl"

    seed = int(cfg.get("seed", 0))
    set_seed(seed, deterministic=deterministic)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env_meta = get_env_metadata(device)

    allow_missing = not bool(args.official)
    try:
        train_loader = make_loader(
            cfg,
            split="train",
            allow_missing=allow_missing,
            generator_seed=seed,
        )
        val_loader = make_loader(
            cfg,
            split="val",
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
    if grad_accum_steps <= 0:
        raise ValueError("`train.grad_accum_steps` must be >= 1")
    grad_clip = float(cfg.get("optim", {}).get("grad_clip_norm", 0.0))
    use_amp = bool(tcfg.get("amp", False)) and device.type == "cuda"

    scaler = make_grad_scaler(enabled=use_amp)

    batch_size = int(cfg["data"]["batch_size"])
    crop_size = int(cfg["data"]["crop_size"])
    effective_batch_size = compute_effective_batch_size(batch_size, grad_accum_steps)
    residue_budget = compute_residue_budget(max_steps, effective_batch_size, crop_size)

    metrics: Dict[str, Any] = {
        "run_name": run_name,
        "track": track_spec.track_id,
        "official_mode": bool(args.official),
        "seed": seed,
        "deterministic": deterministic,
        "n_params": n_params,
        "submission_module": hooks.module_ref,
        "config_path": str(Path(args.config).resolve()),
        "config": cfg,
        "effective_batch_size": effective_batch_size,
        "residue_budget": residue_budget,
        "fingerprint_path": str(Path(fingerprint_path).resolve()) if (args.official or args.verify_fingerprint) else None,
        "env": env_meta,
        "started_at": utc_now_iso(),
        "updated_at": utc_now_iso(),
        "history": [],
        "step_metrics_jsonl": str(step_metrics_path),
    }

    start_step = 0
    if args.resume:
        resume_path = Path(args.resume)
        if not resume_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
        ckpt = torch.load(resume_path, map_location="cpu")
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
        print(f"Resumed from {resume_path} at step {start_step}")
        if paths.metrics_path.exists():
            try:
                existing_metrics = json.loads(paths.metrics_path.read_text())
                if isinstance(existing_metrics, dict):
                    metrics.update(existing_metrics)
                    metrics["resumed_from"] = str(resume_path.resolve())
                    metrics["updated_at"] = utc_now_iso()
            except Exception:
                pass

    step = start_step
    if step >= max_steps:
        print(f"Checkpoint step {step} already reached max_steps={max_steps}; nothing to do.")
        metrics["finished_at"] = utc_now_iso()
        metrics["updated_at"] = utc_now_iso()
        Path(paths.metrics_path).write_text(json.dumps(metrics, indent=2))
        return

    model.train()
    pbar = tqdm(total=max_steps - step, desc="train", dynamic_ncols=True)

    train_iter = iter(train_loader)
    step_start = time.perf_counter()
    run_start = time.perf_counter()

    def run_eval() -> Dict[str, float]:
        model.eval()
        scores = []
        losses = []
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="val", leave=False):
                batch = to_device(batch, device)
                inference_batch = strip_supervision_from_batch(batch)
                with make_autocast_ctx(device=device, enabled=use_amp):
                    out = run_submission_batch(hooks, model=model, batch=inference_batch, cfg=cfg, training=False)
                pred_ca = out["pred_ca"]
                score = batch_lddt_ca(pred_ca, batch["ca_coords"], batch["ca_mask"], batch["residue_mask"])
                scores.append(score.detach().cpu())
                if "loss" in out:
                    losses.append(out["loss"].detach().cpu())
        model.train()
        return {
            "val_loss": float(torch.stack(losses).mean()) if losses else float("nan"),
            "val_lddt_ca": float(torch.stack(scores).mean()) if scores else float("nan"),
        }

    while step < max_steps:
        optimizer_zero_grad(opt)
        running_loss = 0.0
        for _ in range(grad_accum_steps):
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                batch = next(train_iter)

            batch = to_device(batch, device)

            with make_autocast_ctx(device=device, enabled=use_amp):
                out = run_submission_batch(hooks, model=model, batch=batch, cfg=cfg, training=True)
                raw_loss = out["loss"]
                if not raw_loss.requires_grad:
                    # Some batches may have no valid supervision and return a constant
                    # scalar loss. Anchor it to model outputs so backward() stays valid.
                    pred_ca = out.get("pred_ca", None)
                    if torch.is_tensor(pred_ca):
                        raw_loss = raw_loss + pred_ca.sum() * 0.0
                    else:
                        raise RuntimeError(
                            "Submission returned a non-differentiable `loss` and no `pred_ca` tensor "
                            "to anchor gradients."
                        )
                loss = raw_loss / grad_accum_steps

            scaler.scale(loss).backward()
            running_loss += float(raw_loss.detach().cpu())

        if grad_clip and grad_clip > 0:
            scaler.unscale_(opt)
            grad_norm = float(nn.utils.clip_grad_norm_(model.parameters(), grad_clip).item())
        else:
            scaler.unscale_(opt)
            grad_norm = _grad_norm(model)

        scaler.step(opt)
        scaler.update()
        if scheduler is not None:
            scheduler.step()

        step += 1
        pbar.update(1)
        train_loss = running_loss / grad_accum_steps
        lr = _current_lr(opt)
        now = time.perf_counter()
        step_seconds = max(now - step_start, 1e-8)
        step_start = now
        steps_per_sec = 1.0 / step_seconds
        residues_per_sec = (effective_batch_size * crop_size) / step_seconds

        step_record = {
            "timestamp": utc_now_iso(),
            "step": step,
            "train_loss": train_loss,
            "lr": lr,
            "grad_norm": grad_norm,
            "step_seconds": step_seconds,
            "steps_per_sec": steps_per_sec,
            "residues_per_sec": residues_per_sec,
        }
        with step_metrics_path.open("a") as f:
            f.write(json.dumps(step_record) + "\n")

        if step % log_every == 0:
            pbar.set_postfix(loss=train_loss, lr=f"{lr:.3e}", gnorm=f"{grad_norm:.3f}")

        if step % eval_every == 0 or step == max_steps:
            val_metrics = run_eval()
            val_metrics["step"] = step
            val_metrics["train_loss"] = train_loss
            val_metrics["lr"] = lr
            val_metrics["grad_norm"] = grad_norm
            val_metrics["steps_per_sec"] = steps_per_sec
            val_metrics["residues_per_sec"] = residues_per_sec
            metrics["history"].append(val_metrics)
            metrics["updated_at"] = utc_now_iso()
            Path(paths.metrics_path).write_text(json.dumps(metrics, indent=2))
            print("Eval:", val_metrics)

        if step % save_every == 0 or step == max_steps:
            ckpt = {
                "step": step,
                "model": model.state_dict(),
                "opt": opt.state_dict(),
                "submission_module": hooks.module_ref,
                "config": cfg,
                "track": track_spec.track_id,
                "scaler": scaler.state_dict() if use_amp else None,
            }
            if scheduler is not None and callable(getattr(scheduler, "state_dict", None)):
                ckpt["scheduler"] = scheduler.state_dict()
            ckpt_path = paths.ckpt_dir / f"ckpt_step_{step}.pt"
            torch.save(ckpt, ckpt_path)
            torch.save(ckpt, paths.ckpt_dir / "ckpt_last.pt")

    pbar.close()
    metrics["wall_time_seconds"] = float(time.perf_counter() - run_start)
    metrics["finished_at"] = utc_now_iso()
    metrics["updated_at"] = utc_now_iso()
    Path(paths.metrics_path).write_text(json.dumps(metrics, indent=2))
    print("Done. Metrics:", paths.metrics_path)
    print("Checkpoint:", paths.ckpt_dir / "ckpt_last.pt")


if __name__ == "__main__":
    main()
