from __future__ import annotations

import argparse
import json
import time
from functools import partial
from pathlib import Path
import sys
from typing import Any, Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml

from nanofold.competition_policy import (
    DEFAULT_TRACK_ID,
    OFFICIAL_DATASET_FINGERPRINT_PATH,
    TrackSpec,
    apply_track_policy,
    assert_track_policy,
    compute_effective_batch_size,
    compute_residue_budget,
    enforce_model_param_limit,
    load_track_spec,
)
from nanofold.data import ProcessedNPZDataset, collate_batch
from nanofold.dataset_integrity import verify_dataset_against_fingerprint
from nanofold.metrics import lddt_ca
from nanofold.submission_runtime import load_submission_hooks, run_submission_batch
from nanofold.utils import (
    count_parameters,
    get_env_metadata,
    make_dataloader_generator,
    seed_worker,
    to_device,
    utc_now_iso,
)


HIDDEN_SPLITS = {"hidden_val", "test_hidden"}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--ckpt", type=str, default="", help="Single checkpoint path to evaluate.")
    ap.add_argument(
        "--ckpt-list",
        type=str,
        default="",
        help="Comma-separated checkpoint paths for multi-checkpoint evaluation.",
    )
    ap.add_argument(
        "--ckpt-dir",
        type=str,
        default="",
        help="Checkpoint directory for --ckpt-steps (uses ckpt_step_<step>.pt and optional ckpt_last.pt).",
    )
    ap.add_argument(
        "--ckpt-steps",
        type=str,
        default="",
        help="Comma-separated steps for multi-checkpoint eval (example: 1000,2000,5000,10000,last).",
    )
    ap.add_argument("--split", type=str, default="val", choices=["train", "val", "hidden_val", "test_hidden"])
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
    ap.add_argument("--hidden-manifest", type=str, default="", help="Hidden split manifest path override.")
    ap.add_argument(
        "--score-labels-dir",
        type=str,
        default="",
        help=(
            "Optional labels dir used for scoring when dataset batch is features-only. "
            "This path is never passed to submission code."
        ),
    )
    ap.add_argument(
        "--forbid-labels-dir",
        type=str,
        default="",
        help="Optional labels dir that must not be mounted for official features-only eval.",
    )
    ap.add_argument(
        "--allow-labels-mounted",
        action="store_true",
        help="Allow labels to be mounted in official eval (maintainer-only, not leaderboard path).",
    )
    ap.add_argument(
        "--pred-out-dir",
        type=str,
        default="",
        help="Optional directory to write per-chain prediction .npz files.",
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


def _resolve_hidden_manifest(args: argparse.Namespace, track_spec: TrackSpec) -> str:
    if args.hidden_manifest:
        return args.hidden_manifest
    if track_spec.hidden_manifest:
        return track_spec.hidden_manifest
    env_manifest = str(__import__("os").environ.get("NANOFOLD_HIDDEN_MANIFEST", "")).strip()
    if env_manifest:
        return env_manifest
    raise ValueError(
        "Hidden split requested but no hidden manifest is set. Use --hidden-manifest or NANOFOLD_HIDDEN_MANIFEST."
    )


def _manifest_for_split(cfg: Dict[str, Any], args: argparse.Namespace, track_spec: TrackSpec) -> str:
    data_cfg = cfg["data"]
    if args.split == "train":
        return str(data_cfg["train_manifest"])
    if args.split == "val":
        return str(data_cfg["val_manifest"])
    return _resolve_hidden_manifest(args, track_spec)


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
    require_labels: bool,
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
        require_labels=require_labels,
        track_id=track_id,
    )


def _load_label_crop(
    *,
    labels_dir: Path,
    chain_id: str,
    crop_size: int,
    crop_mode: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    label_path = labels_dir / f"{chain_id}.npz"
    if not label_path.exists():
        raise FileNotFoundError(f"Missing label file for scoring: {label_path}")
    with np.load(label_path) as data:
        ca_coords = torch.from_numpy(data["ca_coords"]).float()
        ca_mask = torch.from_numpy(data["ca_mask"]).bool()
    if ca_coords.ndim != 2 or ca_coords.shape[-1] != 3:
        raise ValueError(f"Invalid ca_coords shape in {label_path}: {tuple(ca_coords.shape)}")
    if ca_mask.ndim != 1:
        raise ValueError(f"Invalid ca_mask shape in {label_path}: {tuple(ca_mask.shape)}")
    if ca_coords.shape[0] != ca_mask.shape[0]:
        raise ValueError(f"Label length mismatch in {label_path}")
    L = int(ca_coords.shape[0])
    if L <= crop_size:
        return ca_coords, ca_mask
    if crop_mode == "center":
        start = (L - crop_size) // 2
    elif crop_mode == "random":
        raise ValueError("Scoring external labels with random crop is unsupported; use deterministic crop mode.")
    else:
        raise ValueError(f"Unsupported crop_mode={crop_mode!r}")
    end = start + crop_size
    return ca_coords[start:end], ca_mask[start:end]


def _resolve_checkpoints(args: argparse.Namespace) -> List[Path]:
    explicit: List[Path] = []
    if args.ckpt:
        explicit.append(Path(args.ckpt).resolve())
    if args.ckpt_list:
        for token in args.ckpt_list.split(","):
            token = token.strip()
            if token:
                explicit.append(Path(token).resolve())

    steps = [tok.strip().lower() for tok in args.ckpt_steps.split(",") if tok.strip()]
    if steps:
        if not args.ckpt_dir:
            raise ValueError("--ckpt-steps requires --ckpt-dir.")
        ckpt_dir = Path(args.ckpt_dir).resolve()
        for token in steps:
            if token == "last":
                explicit.append(ckpt_dir / "ckpt_last.pt")
                continue
            if not token.isdigit():
                raise ValueError(f"Invalid checkpoint step token `{token}` in --ckpt-steps.")
            explicit.append(ckpt_dir / f"ckpt_step_{int(token)}.pt")

    if not explicit:
        raise ValueError("Provide at least one checkpoint via --ckpt, --ckpt-list, or --ckpt-steps with --ckpt-dir.")

    deduped: List[Path] = []
    seen: set[Path] = set()
    for ckpt in explicit:
        resolved = ckpt.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        deduped.append(resolved)
    for ckpt in deduped:
        if not ckpt.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
    return deduped


def _lddt_for_chain(pred_ca: torch.Tensor, true_ca: torch.Tensor, ca_mask: torch.Tensor) -> float:
    L = min(int(pred_ca.shape[0]), int(true_ca.shape[0]), int(ca_mask.shape[0]))
    if L <= 0:
        return float("nan")
    score = lddt_ca(pred_ca[:L], true_ca[:L], ca_mask[:L])
    return float(score.detach().cpu())


def _prediction_path_for_ckpt(pred_out_dir: Path, ckpt: Path, multi: bool) -> Path:
    if not multi:
        return pred_out_dir
    stem = ckpt.stem
    out = pred_out_dir / stem
    out.mkdir(parents=True, exist_ok=True)
    return out


def main() -> None:
    args = parse_args()
    raw_cfg = load_config(args.config)
    track_spec = load_track_spec(args.track)
    cfg = apply_track_policy(raw_cfg, track_spec=track_spec) if args.official else raw_cfg
    fingerprint_path = _resolve_fingerprint_path(args, track_spec)

    if args.official:
        assert_track_policy(
            cfg=cfg,
            track_spec=track_spec,
            enforce_manifest_paths=True,
            enforce_manifest_hashes=True,
        )
        try:
            _verify_dataset(
                cfg=cfg,
                fingerprint_path=fingerprint_path,
                require_no_missing=True,
                require_labels=True,
                track_id=track_spec.track_id,
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
            require_labels=bool(str(cfg.get("data", {}).get("processed_labels_dir", "")).strip()),
            track_id=track_spec.track_id,
        )
        print(f"Fingerprint verification succeeded: {Path(fingerprint_path).resolve()}")

    checkpoints = _resolve_checkpoints(args)

    hooks = load_submission_hooks(cfg, args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = bool(cfg.get("train", {}).get("amp", False)) and device.type == "cuda"
    seed = int(cfg.get("seed", 0))

    data_cfg = cfg["data"]
    manifest_path = _manifest_for_split(cfg, args, track_spec)
    num_workers = normalize_num_workers(int(data_cfg.get("num_workers", 0)))
    crop_mode = (
        str(data_cfg.get("train_crop_mode", "random"))
        if args.split == "train"
        else str(data_cfg.get("val_crop_mode", "center"))
    )
    msa_sample_mode = (
        str(data_cfg.get("train_msa_sample_mode", "random"))
        if args.split == "train"
        else str(data_cfg.get("val_msa_sample_mode", "top"))
    )

    include_labels = not bool(args.official) and not bool(args.score_labels_dir)
    labels_dir_for_dataset = str(data_cfg.get("processed_labels_dir", "")).strip() or None
    fail_if_labels_present = False
    if args.official:
        include_labels = False
        labels_dir_for_dataset = args.forbid_labels_dir.strip() or labels_dir_for_dataset
        fail_if_labels_present = not bool(args.allow_labels_mounted)

    try:
        ds = ProcessedNPZDataset(
            processed_features_dir=data_cfg["processed_features_dir"],
            processed_labels_dir=labels_dir_for_dataset,
            include_labels=include_labels,
            fail_if_labels_present=fail_if_labels_present,
            manifest_path=manifest_path,
            allow_missing=not bool(args.official),
        )
    except (FileNotFoundError, RuntimeError) as exc:
        if args.official:
            raise RuntimeError(f"{exc}\n\n{_guidance_for_missing_data(track_spec)}") from exc
        raise

    if getattr(ds, "missing_chain_ids", None):
        print(
            f"[{args.split}] Skipping {len(ds.missing_chain_ids)} missing preprocessed chains "
            f"(first: {', '.join(ds.missing_chain_ids[:6])})"
        )

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
    if args.official:
        enforce_model_param_limit(track_spec=track_spec, n_params=count_parameters(model))

    score_labels_dir = Path(args.score_labels_dir).resolve() if args.score_labels_dir else None
    pred_out_dir = Path(args.pred_out_dir).resolve() if args.pred_out_dir else None
    if pred_out_dir:
        pred_out_dir.mkdir(parents=True, exist_ok=True)
    per_chain_out_path = Path(args.per_chain_out).resolve() if args.per_chain_out else None

    all_ckpt_results: List[Dict[str, Any]] = []
    all_per_chain_rows: List[Dict[str, Any]] = []

    for ckpt_idx, ckpt_path in enumerate(checkpoints):
        ckpt = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(ckpt["model"], strict=True)
        model.eval()

        losses: List[torch.Tensor] = []
        per_chain_rows: List[Dict[str, Any]] = []

        save_pred_root = None
        if pred_out_dir:
            save_pred_root = _prediction_path_for_ckpt(pred_out_dir, ckpt_path, multi=(len(checkpoints) > 1))
            save_pred_root.mkdir(parents=True, exist_ok=True)

        start = time.perf_counter()
        with torch.no_grad():
            for batch in tqdm(loader, desc=f"eval:{args.split}:{ckpt_path.name}"):
                chain_ids = list(batch["chain_id"])
                batch_device = to_device(batch, device)
                with make_autocast_ctx(device=device, enabled=use_amp):
                    run_out = run_submission_batch(hooks, model=model, batch=batch_device, cfg=cfg, training=False)
                pred_ca = run_out["pred_ca"].detach().cpu()
                if "loss" in run_out:
                    losses.append(run_out["loss"].detach().cpu())

                residue_mask = batch["residue_mask"].detach().cpu()
                for idx, chain_id in enumerate(chain_ids):
                    masked_length = int(residue_mask[idx].sum().item())
                    length = int(residue_mask[idx].numel())
                    pred_chain = pred_ca[idx][:masked_length]

                    if save_pred_root is not None:
                        np.savez_compressed(
                            save_pred_root / f"{chain_id}.npz",
                            pred_ca=pred_chain.numpy().astype(np.float32),
                            masked_length=np.array(masked_length, dtype=np.int32),
                            ckpt=str(ckpt_path),
                        )

                    if include_labels:
                        true_ca = batch["ca_coords"][idx][:masked_length]
                        ca_mask = batch["ca_mask"][idx][:masked_length]
                        chain_lddt = _lddt_for_chain(pred_chain, true_ca, ca_mask)
                    elif score_labels_dir is not None:
                        true_ca, ca_mask = _load_label_crop(
                            labels_dir=score_labels_dir,
                            chain_id=chain_id,
                            crop_size=int(data_cfg["crop_size"]),
                            crop_mode=crop_mode,
                        )
                        chain_lddt = _lddt_for_chain(pred_chain, true_ca, ca_mask)
                    else:
                        chain_lddt = float("nan")

                    row = {
                        "ckpt": str(ckpt_path),
                        "chain_id": chain_id,
                        "lddt_ca": chain_lddt,
                        "length": length,
                        "masked_length": masked_length,
                    }
                    per_chain_rows.append(row)
                    all_per_chain_rows.append(row)

        lddt_values = [float(item["lddt_ca"]) for item in per_chain_rows if not np.isnan(item["lddt_ca"])]
        mean_lddt = float(sum(lddt_values) / len(lddt_values)) if lddt_values else float("nan")
        eval_seconds = float(time.perf_counter() - start)
        ckpt_result = {
            "ckpt": str(ckpt_path),
            "mean_loss": float(torch.stack(losses).mean()) if losses else float("nan"),
            "mean_lddt_ca": mean_lddt,
            "num_chains": len(per_chain_rows),
            "eval_wall_time_seconds": eval_seconds,
            "index": ckpt_idx,
        }
        all_ckpt_results.append(ckpt_result)
        print(f"[{ckpt_path.name}] mean_lDDT-Ca={mean_lddt:.6f} chains={len(per_chain_rows)}")

    if per_chain_out_path:
        per_chain_out_path.parent.mkdir(parents=True, exist_ok=True)
        with per_chain_out_path.open("w") as f:
            for row in all_per_chain_rows:
                f.write(json.dumps(row) + "\n")
        print(f"Wrote per-chain scores to {per_chain_out_path}")

    final_ckpt_result = all_ckpt_results[-1]
    batch_size = int(cfg["data"]["batch_size"])
    grad_accum_steps = int(cfg["train"].get("grad_accum_steps", 1))
    crop_size = int(cfg["data"]["crop_size"])
    effective_batch_size = compute_effective_batch_size(batch_size, grad_accum_steps)

    out: Dict[str, Any] = {
        "split": args.split,
        "track": track_spec.track_id,
        "official_mode": bool(args.official),
        "ckpt": final_ckpt_result["ckpt"],
        "num_checkpoints": len(all_ckpt_results),
        "checkpoints": all_ckpt_results,
        "mean_loss": final_ckpt_result["mean_loss"],
        "mean_lddt_ca": final_ckpt_result["mean_lddt_ca"],
        "num_chains": final_ckpt_result["num_chains"],
        "submission_module": hooks.module_ref,
        "config_path": str(Path(args.config).resolve()),
        "fingerprint_path": str(Path(fingerprint_path).resolve()) if (args.official or args.verify_fingerprint) else None,
        "effective_batch_size": effective_batch_size,
        "residue_budget": compute_residue_budget(
            int(cfg["train"]["max_steps"]),
            effective_batch_size,
            crop_size,
        ),
        "env": get_env_metadata(device),
        "pred_out_dir": str(pred_out_dir) if pred_out_dir else None,
        "score_labels_dir": str(score_labels_dir) if score_labels_dir else None,
        "finished_at": utc_now_iso(),
    }
    print(json.dumps(out, indent=2))

    if args.save:
        save_path = Path(args.save)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_path.write_text(json.dumps(out, indent=2) + "\n")
        print(f"Wrote eval summary to {save_path.resolve()}")


if __name__ == "__main__":
    main()
