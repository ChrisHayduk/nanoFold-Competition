from __future__ import annotations

import argparse
import copy
import json
import os
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
    compute_sample_budget,
    compute_residue_budget,
    enforce_model_param_limit,
    load_track_spec,
)
from nanofold.data import ProcessedNPZDataset, collate_batch
from nanofold.dataset_integrity import verify_split_against_fingerprint
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
SEALED_RUNTIME_ENV = "NANOFOLD_OFFICIAL_SEALED_RUNTIME"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Prediction-only evaluation entrypoint.")
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
        help="Comma-separated steps for multi-checkpoint eval (example: 0,1000,2000,last).",
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
        "--forbid-labels-dir",
        type=str,
        default="",
        help="Optional labels dir that must not be mounted for official features-only prediction.",
    )
    ap.add_argument(
        "--allow-labels-mounted",
        action="store_true",
        help="Allow labels to be mounted in official prediction (maintainer-only override).",
    )
    ap.add_argument(
        "--pred-out-dir",
        type=str,
        required=True,
        help="Directory to write per-chain prediction .npz files.",
    )
    ap.add_argument(
        "--save",
        type=str,
        default="",
        help="Optional path to write prediction summary JSON.",
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
    if args.split in HIDDEN_SPLITS and track_spec.hidden_fingerprint_path:
        return track_spec.hidden_fingerprint_path
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
    env_manifest = str(os.environ.get("NANOFOLD_HIDDEN_MANIFEST", "")).strip()
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
    processed_features_dir: str,
    manifest_paths: Dict[str, str],
    fingerprint_path: str,
    require_no_missing: bool,
    track_id: str | None = None,
) -> None:
    verify_split_against_fingerprint(
        processed_features_dir=processed_features_dir,
        processed_labels_dir=None,
        manifest_paths=manifest_paths,
        expected_fingerprint_path=fingerprint_path,
        require_no_missing=require_no_missing,
        require_labels=False,
        track_id=track_id,
        comparison_mode="features_only",
    )


def _sanitize_predict_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(cfg)
    data_cfg = out.get("data")
    if not isinstance(data_cfg, dict):
        raise ValueError("Config missing `data` section.")
    data_cfg["processed_labels_dir"] = ""
    return out


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


def _prediction_path_for_ckpt(pred_out_dir: Path, ckpt: Path, multi: bool) -> Path:
    if not multi:
        return pred_out_dir
    out = pred_out_dir / ckpt.stem
    out.mkdir(parents=True, exist_ok=True)
    return out


def main() -> None:
    args = parse_args()
    if args.official and args.split in HIDDEN_SPLITS and str(os.environ.get(SEALED_RUNTIME_ENV, "")).strip() != "1":
        raise ValueError(
            "Official hidden prediction requires a sealed runtime. "
            "Use scripts/run_official_docker.sh or set NANOFOLD_OFFICIAL_SEALED_RUNTIME=1 in a sealed environment."
        )

    raw_cfg = load_config(args.config)
    # Sanitize labels_dir immediately after load so no downstream code path —
    # including submission hooks, which see only the sanitized cfg — can leak it.
    sanitized_cfg = _sanitize_predict_config(raw_cfg)
    track_spec = load_track_spec(args.track)
    cfg = apply_track_policy(sanitized_cfg, track_spec=track_spec) if args.official else sanitized_cfg
    config_path = Path(args.config).resolve()
    fingerprint_path = _resolve_fingerprint_path(args, track_spec)

    if args.official and args.split in HIDDEN_SPLITS:
        hidden_labels_cfg_value = str(cfg.get("data", {}).get("processed_labels_dir", "")).strip()
        if hidden_labels_cfg_value:
            raise ValueError(
                "Official hidden prediction requires a sanitized config with empty `data.processed_labels_dir`."
            )

    predict_cfg = cfg
    predict_data_cfg = predict_cfg["data"]
    verify_manifest_paths = (
        {args.split: _manifest_for_split(cfg, args, track_spec)}
        if args.split in HIDDEN_SPLITS
        else {
            "train": str(cfg["data"]["train_manifest"]),
            "val": str(cfg["data"]["val_manifest"]),
        }
    )

    if args.official:
        assert_track_policy(
            cfg=cfg,
            track_spec=track_spec,
            enforce_manifest_paths=True,
            enforce_manifest_hashes=True,
        )
        try:
            _verify_dataset(
                processed_features_dir=str(cfg["data"]["processed_features_dir"]),
                manifest_paths=verify_manifest_paths,
                fingerprint_path=fingerprint_path,
                require_no_missing=True,
                track_id=track_spec.track_id,
            )
        except FileNotFoundError as exc:
            raise RuntimeError(f"{exc}\n\n{_guidance_for_missing_data(track_spec)}") from exc
        print(
            f"Official mode enabled for track `{track_spec.track_id}`. "
            f"Feature fingerprint matched: {Path(fingerprint_path).resolve()}"
        )
    elif args.verify_fingerprint:
        _verify_dataset(
            processed_features_dir=str(cfg["data"]["processed_features_dir"]),
            manifest_paths=verify_manifest_paths,
            fingerprint_path=fingerprint_path,
            require_no_missing=False,
            track_id=track_spec.track_id,
        )
        print(f"Feature fingerprint verification succeeded: {Path(fingerprint_path).resolve()}")

    checkpoints = _resolve_checkpoints(args)
    hooks = load_submission_hooks(predict_cfg, config_path, allowed_root=config_path.parent)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = bool(predict_cfg.get("train", {}).get("amp", False)) and device.type == "cuda"
    seed = int(predict_cfg.get("seed", 0))

    data_cfg = predict_data_cfg
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

    forbid_labels_dir = args.forbid_labels_dir.strip() or None
    try:
        ds = ProcessedNPZDataset(
            processed_features_dir=data_cfg["processed_features_dir"],
            processed_labels_dir=forbid_labels_dir,
            include_labels=False,
            fail_if_labels_present=bool(args.official) and bool(forbid_labels_dir) and not bool(args.allow_labels_mounted),
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

    model = hooks.build_model(predict_cfg)
    if not isinstance(model, torch.nn.Module):
        raise TypeError("`build_model(cfg)` must return a torch.nn.Module")
    model = model.to(device)
    if args.official:
        enforce_model_param_limit(track_spec=track_spec, n_params=count_parameters(model))

    pred_out_dir = Path(args.pred_out_dir).resolve()
    pred_out_dir.mkdir(parents=True, exist_ok=True)

    batch_size = int(cfg["data"]["batch_size"])
    grad_accum_steps = int(cfg["train"].get("grad_accum_steps", 1))
    crop_size = int(cfg["data"]["crop_size"])
    effective_batch_size = compute_effective_batch_size(batch_size, grad_accum_steps)
    sample_budget = compute_sample_budget(int(cfg["train"]["max_steps"]), effective_batch_size)
    residue_budget = compute_residue_budget(int(cfg["train"]["max_steps"]), effective_batch_size, crop_size)
    checkpoint_rows: List[Dict[str, Any]] = []
    for ckpt_idx, ckpt_path in enumerate(checkpoints):
        ckpt = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(ckpt["model"], strict=True)
        model.eval()

        pred_root = _prediction_path_for_ckpt(pred_out_dir, ckpt_path, multi=(len(checkpoints) > 1))
        pred_root.mkdir(parents=True, exist_ok=True)

        start = time.perf_counter()
        prediction_count = 0
        with torch.no_grad():
            for batch in tqdm(loader, desc=f"predict:{args.split}:{ckpt_path.name}"):
                chain_ids = list(batch["chain_id"])
                batch_device = to_device(batch, device)
                with make_autocast_ctx(device=device, enabled=use_amp):
                    run_out = run_submission_batch(
                        hooks,
                        model=model,
                        batch=batch_device,
                        cfg=predict_cfg,
                        training=False,
                    )
                pred_atom14_cpu = run_out["pred_atom14"].detach().cpu()
                residue_mask = batch["residue_mask"].detach().cpu()
                for idx, chain_id in enumerate(chain_ids):
                    masked_length = int(residue_mask[idx].sum().item())
                    arrays: Dict[str, Any] = {
                        "pred_atom14": pred_atom14_cpu[idx][:masked_length].numpy().astype(np.float32),
                        "masked_length": np.array(masked_length, dtype=np.int32),
                        "ckpt": str(ckpt_path),
                    }
                    np.savez_compressed(pred_root / f"{chain_id}.npz", **arrays)
                    prediction_count += 1

        step = int(ckpt.get("step", 0))
        cumulative_samples_seen = int(ckpt.get("cumulative_samples_seen", step * effective_batch_size))
        cumulative_cropped_residues_seen = int(
            ckpt.get("cumulative_cropped_residues_seen", step * effective_batch_size * crop_size)
        )
        cumulative_nonpad_residues_seen = int(
            ckpt.get(
                "cumulative_nonpad_residues_seen",
                ckpt.get("cumulative_residues_seen", cumulative_cropped_residues_seen),
            )
        )
        checkpoint_rows.append(
            {
                "ckpt": str(ckpt_path),
                "step": step,
                "num_chains": prediction_count,
                "prediction_dir": str(pred_root.resolve()),
                "predict_wall_time_seconds": float(time.perf_counter() - start),
                "cumulative_samples_seen": cumulative_samples_seen,
                "cumulative_cropped_residues_seen": cumulative_cropped_residues_seen,
                "cumulative_nonpad_residues_seen": cumulative_nonpad_residues_seen,
                "sample_budget_fraction": (
                    cumulative_samples_seen / float(sample_budget)
                ) if sample_budget > 0 else float("nan"),
                "cropped_residue_budget_fraction": (
                    cumulative_cropped_residues_seen / float(residue_budget)
                ) if residue_budget > 0 else float("nan"),
                "nonpad_residue_budget_fraction": (
                    cumulative_nonpad_residues_seen / float(residue_budget)
                ) if residue_budget > 0 else float("nan"),
                "index": ckpt_idx,
            }
        )
        print(f"[{ckpt_path.name}] wrote predictions for {prediction_count} chains")

    final_row = checkpoint_rows[-1]
    out: Dict[str, Any] = {
        "mode": "predict",
        "split": args.split,
        "track": track_spec.track_id,
        "official_mode": bool(args.official),
        "num_checkpoints": len(checkpoint_rows),
        "checkpoints": checkpoint_rows,
        "submission_module": hooks.module_ref,
        "submission_entrypoint_path": hooks.source_path,
        "config_path": str(config_path),
        "manifest_path": str(Path(manifest_path).resolve()),
        "fingerprint_path": str(Path(fingerprint_path).resolve()) if (args.official or args.verify_fingerprint) else None,
        "effective_batch_size": effective_batch_size,
        "sample_budget": sample_budget,
        "residue_budget": residue_budget,
        "crop_size": crop_size,
        "crop_mode": crop_mode,
        "cumulative_samples_seen": int(final_row["cumulative_samples_seen"]),
        "cumulative_cropped_residues_seen": int(final_row["cumulative_cropped_residues_seen"]),
        "cumulative_nonpad_residues_seen": int(final_row["cumulative_nonpad_residues_seen"]),
        "env": get_env_metadata(device),
        "pred_out_dir": str(pred_out_dir.resolve()),
        "predict_config_sanitized": True,
        "finished_at": utc_now_iso(),
    }
    print(json.dumps(out, indent=2))

    if args.save:
        save_path = Path(args.save)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_path.write_text(json.dumps(out, indent=2) + "\n")
        print(f"Wrote prediction summary to {save_path.resolve()}")


if __name__ == "__main__":
    main()
