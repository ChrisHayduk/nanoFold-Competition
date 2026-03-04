from __future__ import annotations

import argparse
import json
import time
from functools import partial
from pathlib import Path
import sys
from typing import Any, Dict, List

import torch
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
from nanofold.utils import get_env_metadata, make_dataloader_generator, seed_worker, to_device, utc_now_iso


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--split", type=str, default="val", choices=["train", "val"])
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
        "--per-chain-out",
        type=str,
        default="",
        help="Optional JSONL output path for per-chain lDDT-Ca records.",
    )
    ap.add_argument(
        "--save",
        type=str,
        default="",
        help="Optional path to write eval summary JSON.",
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


def _guidance_for_missing_data(track_spec: TrackSpec) -> str:
    return (
        "Official mode requires fully preprocessed data for every chain in the official manifests.\n"
        f"Track: {track_spec.track_id}\n"
        "Run:\n"
        "  bash scripts/setup_official_data.sh\n"
        "or preprocess any missing chains listed in the error message."
    )


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


@torch.no_grad()
def batch_lddt_per_chain(
    pred_ca: torch.Tensor,
    true_ca: torch.Tensor,
    ca_mask: torch.Tensor,
    residue_mask: torch.Tensor,
) -> List[float]:
    scores: List[float] = []
    B = pred_ca.shape[0]
    for b in range(B):
        score = lddt_ca(pred_ca[b], true_ca[b], ca_mask[b] & residue_mask[b])
        scores.append(float(score.detach().cpu()))
    return scores


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    track_spec = load_track_spec(args.track)
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = bool(cfg.get("train", {}).get("amp", False)) and device.type == "cuda"

    data_cfg = cfg["data"]
    manifest_path = data_cfg["train_manifest"] if args.split == "train" else data_cfg["val_manifest"]
    num_workers = normalize_num_workers(int(data_cfg.get("num_workers", 0)))
    seed = int(cfg.get("seed", 0))

    try:
        ds = ProcessedNPZDataset(
            processed_dir=data_cfg["processed_dir"],
            manifest_path=manifest_path,
            allow_missing=not bool(args.official),
        )
    except FileNotFoundError as exc:
        if args.official:
            raise RuntimeError(f"{exc}\n\n{_guidance_for_missing_data(track_spec)}") from exc
        raise

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

    loader = DataLoader(
        ds,
        batch_size=data_cfg.get("batch_size", 1),
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_fn,
        worker_init_fn=seed_worker if num_workers > 0 else None,
        generator=make_dataloader_generator(seed + 17),
    )

    model = hooks.build_model(cfg)
    if not isinstance(model, torch.nn.Module):
        raise TypeError("`build_model(cfg)` must return a torch.nn.Module")
    model = model.to(device)

    ckpt = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()

    losses: List[torch.Tensor] = []
    per_chain: List[Dict[str, Any]] = []

    start = time.perf_counter()
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"eval:{args.split}"):
            chain_ids = list(batch["chain_id"])
            batch = to_device(batch, device)
            inference_batch = strip_supervision_from_batch(batch)
            with make_autocast_ctx(device=device, enabled=use_amp):
                out = run_submission_batch(hooks, model=model, batch=inference_batch, cfg=cfg, training=False)
            pred_ca = out["pred_ca"]
            scores = batch_lddt_per_chain(pred_ca, batch["ca_coords"], batch["ca_mask"], batch["residue_mask"])
            if "loss" in out:
                losses.append(out["loss"].detach().cpu())

            residue_mask = batch["residue_mask"].detach().cpu()
            for idx, chain_id in enumerate(chain_ids):
                length = int(residue_mask[idx].numel())
                masked_length = int(residue_mask[idx].sum().item())
                per_chain.append(
                    {
                        "chain_id": chain_id,
                        "lddt_ca": float(scores[idx]),
                        "length": length,
                        "masked_length": masked_length,
                    }
                )

    lddt_values = [float(item["lddt_ca"]) for item in per_chain]
    mean_lddt = float(sum(lddt_values) / len(lddt_values)) if lddt_values else float("nan")
    eval_seconds = float(time.perf_counter() - start)

    batch_size = int(cfg["data"]["batch_size"])
    grad_accum_steps = int(cfg["train"].get("grad_accum_steps", 1))
    crop_size = int(cfg["data"]["crop_size"])

    out = {
        "split": args.split,
        "track": track_spec.track_id,
        "official_mode": bool(args.official),
        "mean_loss": float(torch.stack(losses).mean()) if losses else float("nan"),
        "mean_lddt_ca": mean_lddt,
        "num_chains": len(per_chain),
        "ckpt": str(Path(args.ckpt).resolve()),
        "submission_module": hooks.module_ref,
        "config_path": str(Path(args.config).resolve()),
        "fingerprint_path": str(Path(fingerprint_path).resolve()) if (args.official or args.verify_fingerprint) else None,
        "effective_batch_size": compute_effective_batch_size(batch_size, grad_accum_steps),
        "residue_budget": compute_residue_budget(
            int(cfg["train"]["max_steps"]),
            compute_effective_batch_size(batch_size, grad_accum_steps),
            crop_size,
        ),
        "env": get_env_metadata(device),
        "eval_wall_time_seconds": eval_seconds,
        "finished_at": utc_now_iso(),
    }
    print(json.dumps(out, indent=2))

    if args.per_chain_out:
        out_path = Path(args.per_chain_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w") as f:
            for row in per_chain:
                f.write(json.dumps(row) + "\n")
        print(f"Wrote per-chain scores to {out_path.resolve()}")

    if args.save:
        save_path = Path(args.save)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_path.write_text(json.dumps(out, indent=2) + "\n")
        print(f"Wrote eval summary to {save_path.resolve()}")


if __name__ == "__main__":
    main()
