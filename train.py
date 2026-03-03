from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml

from nanofold.data import ProcessedNPZDataset, collate_batch
from nanofold.metrics import lddt_ca
from nanofold.submission_runtime import load_submission_hooks, run_submission_batch
from nanofold.utils import RunPaths, count_parameters, ensure_dir, set_seed, to_device


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    return ap.parse_args()


def load_config(path: str | Path) -> Dict[str, Any]:
    return yaml.safe_load(Path(path).read_text())


def make_loader(cfg: Dict[str, Any], split: str) -> DataLoader:
    data_cfg = cfg["data"]
    processed_dir = data_cfg["processed_dir"]
    manifest_path = data_cfg["train_manifest"] if split == "train" else data_cfg["val_manifest"]

    ds = ProcessedNPZDataset(processed_dir=processed_dir, manifest_path=manifest_path)

    return DataLoader(
        ds,
        batch_size=data_cfg.get("batch_size", 1),
        shuffle=(split == "train"),
        num_workers=data_cfg.get("num_workers", 0),
        pin_memory=True,
        collate_fn=lambda exs: collate_batch(exs, crop_size=data_cfg["crop_size"], msa_depth=data_cfg["msa_depth"]),
        drop_last=(split == "train"),
    )


@torch.no_grad()
def batch_lddt_ca(pred_ca: torch.Tensor, true_ca: torch.Tensor, ca_mask: torch.Tensor, residue_mask: torch.Tensor) -> torch.Tensor:
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


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    hooks = load_submission_hooks(cfg, args.config)

    run_name = cfg.get("run_name", "run")
    paths = RunPaths.from_run_name(run_name)
    ensure_dir(paths.run_dir)
    ensure_dir(paths.ckpt_dir)

    seed = int(cfg.get("seed", 0))
    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader = make_loader(cfg, split="train")
    val_loader = make_loader(cfg, split="val")

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
    use_amp = bool(tcfg.get("amp", False))

    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    step = 0
    model.train()
    pbar = tqdm(total=max_steps, desc="train", dynamic_ncols=True)

    train_iter = iter(train_loader)

    def run_eval() -> Dict[str, float]:
        model.eval()
        scores = []
        losses = []
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="val", leave=False):
                batch = to_device(batch, device)
                with torch.cuda.amp.autocast(enabled=use_amp):
                    out = run_submission_batch(hooks, model=model, batch=batch, cfg=cfg, training=False)
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

    metrics: Dict[str, Any] = {
        "run_name": run_name,
        "seed": seed,
        "n_params": n_params,
        "config": cfg,
        "history": [],
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

            with torch.cuda.amp.autocast(enabled=use_amp):
                out = run_submission_batch(hooks, model=model, batch=batch, cfg=cfg, training=True)
                raw_loss = out["loss"]
                loss = raw_loss / grad_accum_steps

            scaler.scale(loss).backward()
            running_loss += float(raw_loss.detach().cpu())

        if grad_clip and grad_clip > 0:
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        scaler.step(opt)
        scaler.update()
        if scheduler is not None:
            scheduler.step()

        step += 1
        pbar.update(1)
        train_loss = running_loss / grad_accum_steps

        if step % log_every == 0:
            pbar.set_postfix(loss=train_loss)

        if step % eval_every == 0 or step == max_steps:
            val_metrics = run_eval()
            val_metrics["step"] = step
            val_metrics["train_loss"] = train_loss
            metrics["history"].append(val_metrics)
            Path(paths.metrics_path).write_text(json.dumps(metrics, indent=2))
            print("Eval:", val_metrics)

        if step % save_every == 0 or step == max_steps:
            ckpt = {
                "step": step,
                "model": model.state_dict(),
                "opt": opt.state_dict(),
                "submission_module": hooks.module_ref,
                "config": cfg,
            }
            if scheduler is not None and callable(getattr(scheduler, "state_dict", None)):
                ckpt["scheduler"] = scheduler.state_dict()
            ckpt_path = paths.ckpt_dir / f"ckpt_step_{step}.pt"
            torch.save(ckpt, ckpt_path)
            torch.save(ckpt, paths.ckpt_dir / "ckpt_last.pt")

    pbar.close()
    print("Done. Metrics:", paths.metrics_path)
    print("Checkpoint:", paths.ckpt_dir / "ckpt_last.pt")


if __name__ == "__main__":
    main()
