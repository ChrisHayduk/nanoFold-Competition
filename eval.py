from __future__ import annotations

import argparse
import json
from functools import partial
from pathlib import Path
import sys
from typing import Any, Dict

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml

from nanofold.competition_policy import (
    OFFICIAL_DATASET_FINGERPRINT_PATH,
    assert_official_limited_config,
)
from nanofold.data import ProcessedNPZDataset, collate_batch
from nanofold.dataset_integrity import verify_dataset_against_fingerprint
from nanofold.metrics import lddt_ca
from nanofold.submission_runtime import (
    load_submission_hooks,
    run_submission_batch,
    strip_supervision_from_batch,
)
from nanofold.utils import to_device


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--split", type=str, default="val", choices=["train", "val"])
    ap.add_argument("--official", action="store_true", help="Enable strict official limited-track enforcement.")
    ap.add_argument(
        "--fingerprint",
        type=str,
        default=OFFICIAL_DATASET_FINGERPRINT_PATH,
        help=f"Expected dataset fingerprint JSON path (default: {OFFICIAL_DATASET_FINGERPRINT_PATH}).",
    )
    return ap.parse_args()


def load_config(path: str | Path) -> Dict[str, Any]:
    return yaml.safe_load(Path(path).read_text())


def make_autocast_ctx(device: torch.device, enabled: bool):
    try:
        return torch.amp.autocast(device_type=device.type, enabled=enabled)
    except Exception:
        return torch.cuda.amp.autocast(enabled=enabled)


def normalize_num_workers(n: int) -> int:
    n = int(n)
    if n <= 0:
        return 0
    if sys.platform == "darwin" and sys.version_info >= (3, 13):
        print("Forcing data.num_workers=0 on macOS with Python 3.13+ for DataLoader stability.")
        return 0
    return n


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
    if args.official:
        assert_official_limited_config(cfg)
        data_cfg = cfg["data"]
        verify_dataset_against_fingerprint(
            processed_dir=data_cfg["processed_dir"],
            train_manifest=data_cfg["train_manifest"],
            val_manifest=data_cfg["val_manifest"],
            expected_fingerprint_path=args.fingerprint,
            require_no_missing=True,
        )
        print(f"Official mode enabled. Dataset fingerprint matched: {Path(args.fingerprint).resolve()}")

    hooks = load_submission_hooks(cfg, args.config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = bool(cfg.get("train", {}).get("amp", False)) and device.type == "cuda"

    data_cfg = cfg["data"]
    manifest_path = data_cfg["train_manifest"] if args.split == "train" else data_cfg["val_manifest"]
    ds = ProcessedNPZDataset(
        processed_dir=data_cfg["processed_dir"],
        manifest_path=manifest_path,
        allow_missing=not bool(args.official),
    )
    if getattr(ds, "missing_chain_ids", None):
        print(
            f"[{args.split}] Skipping {len(ds.missing_chain_ids)} missing preprocessed chains "
            f"(first: {', '.join(ds.missing_chain_ids[:6])})"
        )
    if args.split == "train":
        crop_mode = str(data_cfg.get("train_crop_mode", "random"))
        msa_sample_mode = str(data_cfg.get("train_msa_sample_mode", "random"))
    else:
        crop_mode = str(data_cfg.get("val_crop_mode", "center"))
        msa_sample_mode = str(data_cfg.get("val_msa_sample_mode", "top"))
    collate_fn = partial(
        collate_batch,
        crop_size=int(data_cfg["crop_size"]),
        msa_depth=int(data_cfg["msa_depth"]),
        crop_mode=crop_mode,
        msa_sample_mode=msa_sample_mode,
    )
    num_workers = normalize_num_workers(int(data_cfg.get("num_workers", 0)))

    loader = DataLoader(
        ds,
        batch_size=data_cfg.get("batch_size", 1),
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_fn,
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
            inference_batch = strip_supervision_from_batch(batch)
            with make_autocast_ctx(device=device, enabled=use_amp):
                out = run_submission_batch(hooks, model=model, batch=inference_batch, cfg=cfg, training=False)
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
