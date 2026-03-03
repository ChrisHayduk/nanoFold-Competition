from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml

from nanofold.data import ProcessedNPZDataset, collate_batch
from nanofold.metrics import lddt_ca
from nanofold.submission_runtime import load_submission_hooks, run_submission_batch
from nanofold.utils import to_device


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--split", type=str, default="val", choices=["train", "val"])
    return ap.parse_args()


def load_config(path: str | Path) -> Dict[str, Any]:
    return yaml.safe_load(Path(path).read_text())


@torch.no_grad()
def batch_lddt_ca(pred_ca: torch.Tensor, true_ca: torch.Tensor, ca_mask: torch.Tensor, residue_mask: torch.Tensor) -> torch.Tensor:
    scores = []
    B = pred_ca.shape[0]
    for b in range(B):
        scores.append(lddt_ca(pred_ca[b], true_ca[b], ca_mask[b] & residue_mask[b]))
    return torch.stack(scores).mean()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    hooks = load_submission_hooks(cfg, args.config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = bool(cfg.get("train", {}).get("amp", False))

    data_cfg = cfg["data"]
    manifest_path = data_cfg["train_manifest"] if args.split == "train" else data_cfg["val_manifest"]
    ds = ProcessedNPZDataset(processed_dir=data_cfg["processed_dir"], manifest_path=manifest_path)

    loader = DataLoader(
        ds,
        batch_size=data_cfg.get("batch_size", 1),
        shuffle=False,
        num_workers=data_cfg.get("num_workers", 0),
        pin_memory=True,
        collate_fn=lambda exs: collate_batch(exs, crop_size=data_cfg["crop_size"], msa_depth=data_cfg["msa_depth"]),
    )

    model = hooks.build_model(cfg)
    if not isinstance(model, torch.nn.Module):
        raise TypeError("`build_model(cfg)` must return a torch.nn.Module")
    model = model.to(device)

    ckpt = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()

    losses = []
    scores = []

    with torch.no_grad():
        for batch in tqdm(loader, desc=f"eval:{args.split}"):
            batch = to_device(batch, device)
            with torch.cuda.amp.autocast(enabled=use_amp):
                out = run_submission_batch(hooks, model=model, batch=batch, cfg=cfg, training=False)
            pred_ca = out["pred_ca"]
            score = batch_lddt_ca(pred_ca, batch["ca_coords"], batch["ca_mask"], batch["residue_mask"])
            scores.append(score.detach().cpu())
            if "loss" in out:
                losses.append(out["loss"].detach().cpu())

    out = {
        "split": args.split,
        "mean_loss": float(torch.stack(losses).mean()) if losses else float("nan"),
        "mean_lddt_ca": float(torch.stack(scores).mean()) if scores else float("nan"),
        "ckpt": str(args.ckpt),
        "submission_module": hooks.module_ref,
    }
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
