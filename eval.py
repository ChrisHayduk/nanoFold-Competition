from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from functools import partial
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from nanofold.chain_paths import chain_npz_path
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
from nanofold.dataset_integrity import verify_split_against_fingerprint
from nanofold.metrics import FOLDSCORE_COMPONENT_NAMES, FOLDSCORE_VERSION, foldscore_components
from nanofold.prediction_contract import (
    PREDICTION_FORMAT_VERSION,
    prepare_prediction_directory,
    validate_checkpoint_for_prediction,
    validate_checkpoint_output_names,
    validate_complete_chain_ids,
)
from nanofold.residue_constants import CA_ATOM14_SLOT
from nanofold.scoring_io import (
    load_feature_crop as _load_feature_crop,
)
from nanofold.scoring_io import (
    load_label_crop as _load_label_crop,
)
from nanofold.scoring_io import (
    validate_scoring_alignment,
)
from nanofold.submission_runtime import load_submission_hooks, run_submission_batch
from nanofold.utils import (
    count_parameters,
    default_torch_device,
    get_env_metadata,
    load_torch_checkpoint,
    make_dataloader_generator,
    seed_worker,
    set_seed,
    sha256_file,
    should_pin_memory,
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
        help="Optional JSONL output path for per-chain FoldScore/component records.",
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
        amp = getattr(torch, "amp")
        return amp.autocast(device_type=device.type, enabled=enabled)
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
    processed_features_dir: str,
    processed_labels_dir: str | None,
    manifest_paths: Dict[str, str],
    fingerprint_path: str,
    require_no_missing: bool,
    require_labels: bool,
    track_id: str | None = None,
) -> None:
    verify_split_against_fingerprint(
        processed_features_dir=processed_features_dir,
        processed_labels_dir=processed_labels_dir,
        manifest_paths=manifest_paths,
        expected_fingerprint_path=fingerprint_path,
        require_no_missing=require_no_missing,
        require_labels=require_labels,
        track_id=track_id,
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
    validate_checkpoint_output_names(deduped)
    return deduped


def _score_chain(
    *,
    pred_atom14: torch.Tensor,
    labels: Dict[str, torch.Tensor],
    features: Dict[str, torch.Tensor],
) -> Dict[str, float]:
    validate_scoring_alignment(labels, features)
    has_atom14_labels = "atom14_positions" in labels and "atom14_mask" in labels
    if not has_atom14_labels:
        raise ValueError("FoldScore eval requires label atom14_positions/atom14_mask.")
    device = pred_atom14.device
    labels = {
        name: value.to(device=device, non_blocking=True) if torch.is_tensor(value) else value
        for name, value in labels.items()
    }
    features = {
        name: value.to(device=device, non_blocking=True) if torch.is_tensor(value) else value
        for name, value in features.items()
    }
    comps = foldscore_components(
        pred_atom14=pred_atom14,
        true_atom14=labels["atom14_positions"],
        atom14_mask=labels["atom14_mask"],
        aatype=features["aatype"],
        residue_index=features.get("residue_index"),
        between_segment_residues=features.get("between_segment_residues"),
    )
    metrics = {name: float(value.detach().cpu()) for name, value in comps.items()}
    ca_mask = labels["ca_mask"].to(dtype=torch.bool)
    rmsd_ca = _masked_kabsch_rmsd(
        pred_atom14[:, CA_ATOM14_SLOT, :],
        labels["ca_coords"],
        ca_mask,
    )
    metrics["rmsd_ca"] = float(rmsd_ca.detach().cpu())
    return metrics


def _masked_kabsch_rmsd(pred: torch.Tensor, true: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    pred = pred.float()
    true = true.to(device=pred.device, dtype=torch.float32)
    active = mask.to(dtype=torch.bool)
    if int(active.sum().item()) < 3:
        return torch.tensor(float("nan"), device=pred.device, dtype=pred.dtype)
    pred_active = pred[active]
    true_active = true[active].to(device=pred.device, dtype=pred.dtype)
    pred_centered = pred_active - pred_active.mean(dim=0, keepdim=True)
    true_centered = true_active - true_active.mean(dim=0, keepdim=True)
    covariance = pred_centered.transpose(0, 1) @ true_centered
    u, _, vh = torch.linalg.svd(covariance, full_matrices=False)
    correction = torch.ones(3, device=pred.device, dtype=pred.dtype)
    if torch.det(u @ vh) < 0:
        correction[-1] = -1.0
    rotation = u @ torch.diag(correction) @ vh
    aligned = pred_centered @ rotation
    squared_error = (aligned - true_centered).square().sum(dim=-1)
    return torch.sqrt(squared_error.mean().clamp_min(0.0))


def _scalar_output_metrics(out: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    metrics: Dict[str, torch.Tensor] = {}
    for key, value in out.items():
        if key in {"loss", "pred_atom14", "pred_ca"}:
            continue
        if not torch.is_tensor(value) or value.numel() == 0:
            continue
        scalar = value.detach().float().mean().cpu()
        metrics[str(key)] = scalar
    return metrics


def _summarize_chain_metrics(
    rows: List[Dict[str, Any]], *, expected_chain_ids: list[str], scored: bool,
) -> Dict[str, Any]:
    validate_complete_chain_ids([str(row["chain_id"]) for row in rows], expected_chain_ids)
    out: Dict[str, Any] = {"scored_chain_count": len(rows) if scored else 0}
    for name in FOLDSCORE_COMPONENT_NAMES:
        if not scored:
            out[f"mean_{name}"] = float("nan")
            continue
        values = [float(row.get(name, float("nan"))) for row in rows]
        if not values or any(not np.isfinite(value) or not 0.0 <= value <= 1.0 for value in values):
            raise ValueError(f"Every manifest chain must have a finite {name} in [0, 1]; refusing an incomplete mean.")
        out[f"mean_{name}"] = float(sum(values) / len(expected_chain_ids))
    rmsd = [float(row.get("rmsd_ca", float("nan"))) for row in rows]
    if any(np.isinf(value) or value < 0 for value in rmsd):
        raise ValueError("RMSD diagnostics contain an invalid value.")
    valid_rmsd = [value for value in rmsd if np.isfinite(value)]
    out["mean_rmsd_ca"] = float(sum(valid_rmsd) / len(valid_rmsd)) if valid_rmsd else float("nan")
    out["rmsd_ca_valid_count"] = len(valid_rmsd)
    out["rmsd_ca_undefined_count"] = len(rows) - len(valid_rmsd)
    return out


def _summarize_submission_metrics(metrics: Dict[str, list[tuple[torch.Tensor, int]]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for name, entries in metrics.items():
        valid = [(float(value), count) for value, count in entries if torch.isfinite(value)]
        valid_count = sum(count for _, count in valid)
        prefix = f"submission_{name}"
        out[f"mean_{prefix}"] = sum(value * count for value, count in valid) / valid_count if valid_count else float("nan")
        out[f"{prefix}_valid_count"] = valid_count
        out[f"{prefix}_nonfinite_count"] = sum(count for _, count in entries) - valid_count
    return out


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
    config_path = Path(args.config).resolve()
    fingerprint_path = _resolve_fingerprint_path(args, track_spec)
    if args.official and args.split in HIDDEN_SPLITS:
        hidden_labels_cfg_value = str(cfg.get("data", {}).get("processed_labels_dir", "")).strip()
        if hidden_labels_cfg_value:
            raise ValueError(
                "Official hidden prediction requires a sanitized config with empty `data.processed_labels_dir`."
            )
    predict_cfg = _sanitize_predict_config(cfg)
    predict_data_cfg = predict_cfg["data"]
    verify_manifest_paths = (
        {args.split: _manifest_for_split(cfg, args, track_spec)}
        if args.split in HIDDEN_SPLITS
        else {
            "train": str(cfg["data"]["train_manifest"]),
            "val": str(cfg["data"]["val_manifest"]),
        }
    )
    verify_labels_dir = None if args.split in HIDDEN_SPLITS else str(cfg["data"].get("processed_labels_dir", "")).strip() or None

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
                processed_labels_dir=verify_labels_dir,
                manifest_paths=verify_manifest_paths,
                fingerprint_path=fingerprint_path,
                require_no_missing=True,
                require_labels=verify_labels_dir is not None,
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
            processed_features_dir=str(cfg["data"]["processed_features_dir"]),
            processed_labels_dir=verify_labels_dir,
            manifest_paths=verify_manifest_paths,
            fingerprint_path=fingerprint_path,
            require_no_missing=True,
            require_labels=verify_labels_dir is not None,
            track_id=track_spec.track_id,
        )
        print(f"Fingerprint verification succeeded: {Path(fingerprint_path).resolve()}")

    checkpoints = _resolve_checkpoints(args)

    hooks = load_submission_hooks(predict_cfg, config_path, allowed_root=config_path.parent)
    device = default_torch_device()
    print(f"Using device: {device}")
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

    include_labels = not bool(args.official) and not bool(args.score_labels_dir)
    labels_dir_for_dataset = str(cfg["data"].get("processed_labels_dir", "")).strip() or None
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
            allow_missing=False,
        )
    except (FileNotFoundError, RuntimeError) as exc:
        if args.official:
            raise RuntimeError(f"{exc}\n\n{_guidance_for_missing_data(track_spec)}") from exc
        raise

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
        pin_memory=should_pin_memory(device),
        collate_fn=collate_fn,
        worker_init_fn=seed_worker if num_workers > 0 else None,
        generator=make_dataloader_generator(seed + 17),
    )

    score_labels_dir = Path(args.score_labels_dir).resolve() if args.score_labels_dir else None
    score_features_dir = Path(str(data_cfg["processed_features_dir"])).resolve()
    pred_out_dir = Path(args.pred_out_dir).resolve() if args.pred_out_dir else None
    if pred_out_dir:
        pred_out_dir.mkdir(parents=True, exist_ok=True)
    per_chain_out_path = Path(args.per_chain_out).resolve() if args.per_chain_out else None
    batch_size = int(cfg["data"]["batch_size"])
    grad_accum_steps = int(cfg["train"].get("grad_accum_steps", 1))
    crop_size = int(cfg["data"]["crop_size"])
    effective_batch_size = compute_effective_batch_size(batch_size, grad_accum_steps)
    sample_budget = compute_sample_budget(
        int(cfg["train"]["max_steps"]),
        effective_batch_size,
    )
    residue_budget = compute_residue_budget(
        int(cfg["train"]["max_steps"]),
        effective_batch_size,
        crop_size,
    )

    all_ckpt_results: List[Dict[str, Any]] = []
    all_per_chain_rows: List[Dict[str, Any]] = []

    for ckpt_idx, ckpt_path in enumerate(checkpoints):
        # Reset both main-process and worker crop/MSA randomness for each checkpoint.
        set_seed(seed + 17)
        assert loader.generator is not None
        loader.generator.manual_seed(seed + 17)
        checkpoint_sha256 = sha256_file(ckpt_path)
        ckpt = load_torch_checkpoint(ckpt_path, map_location="cpu")
        if sha256_file(ckpt_path) != checkpoint_sha256:
            raise ValueError("Checkpoint changed while it was being loaded.")
        checkpoint_model_contract_verified = validate_checkpoint_for_prediction(
            ckpt, predict_cfg, submission_source_sha256=hooks.source_sha256, require_metadata=bool(args.official),
            training_fingerprint_sha256=(sha256_file(fingerprint_path) if args.official and args.split not in HIDDEN_SPLITS else None),
            track_id=track_spec.track_id if args.official else None,
        )
        model = hooks.build_model(predict_cfg)
        if not isinstance(model, torch.nn.Module):
            raise TypeError("`build_model(cfg)` must return a torch.nn.Module")
        model = model.to(device)
        if args.official:
            enforce_model_param_limit(track_spec=track_spec, n_params=count_parameters(model))
        model.load_state_dict(ckpt["model"], strict=True)
        model.eval()

        scalar_metrics: Dict[str, list[tuple[torch.Tensor, int]]] = {}
        per_chain_rows: List[Dict[str, Any]] = []

        save_pred_root = None
        if pred_out_dir:
            save_pred_root = _prediction_path_for_ckpt(pred_out_dir, ckpt_path, multi=(len(checkpoints) > 1))
            prepare_prediction_directory(save_pred_root)

        start = time.perf_counter()
        with torch.no_grad():
            for batch in tqdm(loader, desc=f"eval:{args.split}:{ckpt_path.name}"):
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
                pred_atom14 = run_out["pred_atom14"].detach()
                pred_atom14_cpu = pred_atom14.detach().float().cpu() if save_pred_root is not None else None
                if "loss" in run_out:
                    scalar_metrics.setdefault("loss", []).append((run_out["loss"].detach().cpu(), len(chain_ids)))
                for metric_name, metric_value in _scalar_output_metrics(run_out).items():
                    scalar_metrics.setdefault(metric_name, []).append((metric_value, len(chain_ids)))

                residue_mask = batch["residue_mask"].detach().cpu()
                for idx, chain_id in enumerate(chain_ids):
                    masked_length = int(residue_mask[idx].sum().item())
                    length = int(residue_mask[idx].numel())

                    if save_pred_root is not None:
                        assert pred_atom14_cpu is not None
                        arrays: Dict[str, Any] = {
                            "pred_atom14": pred_atom14_cpu[idx][:masked_length].numpy().astype(np.float32),
                            "masked_length": np.array(masked_length, dtype=np.int32),
                            "ckpt": str(ckpt_path),
                            "ckpt_sha256": checkpoint_sha256,
                            "chain_id": str(chain_id),
                        }
                        with chain_npz_path(save_pred_root, chain_id).open("xb") as prediction_file:
                            np.savez_compressed(prediction_file, **arrays)

                    if include_labels:
                        label_tensors: Dict[str, torch.Tensor] = {
                            "ca_coords": batch_device["ca_coords"][idx][:masked_length],
                            "ca_mask": batch_device["ca_mask"][idx][:masked_length],
                        }
                        label_tensors["atom14_positions"] = batch_device["atom14_positions"][idx][:masked_length]
                        label_tensors["atom14_mask"] = batch_device["atom14_mask"][idx][:masked_length]
                        feature_tensors: Dict[str, torch.Tensor] = {
                            key: batch_device[key][idx][:masked_length]
                            for key in ("aatype", "residue_index", "between_segment_residues") if key in batch_device
                        }
                        chain_metrics = _score_chain(
                            pred_atom14=pred_atom14[idx][:masked_length],
                            labels=label_tensors,
                            features=feature_tensors,
                        )
                    elif score_labels_dir is not None:
                        labels = _load_label_crop(
                            labels_dir=score_labels_dir,
                            chain_id=chain_id,
                            crop_size=int(data_cfg["crop_size"]),
                            crop_mode=crop_mode,
                        )
                        features = _load_feature_crop(
                            features_dir=score_features_dir,
                            chain_id=chain_id,
                            crop_size=int(data_cfg["crop_size"]),
                            crop_mode=crop_mode,
                        )
                        chain_metrics = _score_chain(
                            pred_atom14=pred_atom14[idx][:masked_length],
                            labels=labels,
                            features=features,
                        )
                    else:
                        chain_metrics = {
                            "lddt_ca": float("nan"),
                            "lddt_backbone_atom14": float("nan"),
                            "lddt_atom14": float("nan"),
                            "foldscore": float("nan"),
                        }

                    row = {
                        "ckpt": str(ckpt_path),
                        "ckpt_sha256": checkpoint_sha256,
                        "foldscore_version": FOLDSCORE_VERSION,
                        "step": int(ckpt.get("step", 0)),
                        "chain_id": chain_id,
                        **chain_metrics,
                        "length": length,
                        "masked_length": masked_length,
                    }
                    per_chain_rows.append(row)
                    all_per_chain_rows.append(row)

        component_means = _summarize_chain_metrics(
            per_chain_rows, expected_chain_ids=ds.chain_ids, scored=include_labels or score_labels_dir is not None,
        )
        mean_lddt = component_means["mean_lddt_ca"]
        mean_foldscore = component_means["mean_foldscore"]
        scalar_means = _summarize_submission_metrics(scalar_metrics)
        eval_seconds = float(time.perf_counter() - start)
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
        ckpt_result = {
            "ckpt": str(ckpt_path),
            "ckpt_sha256": checkpoint_sha256,
            "checkpoint_model_contract_verified": checkpoint_model_contract_verified,
            "training_fingerprint_sha256": ckpt.get("fingerprint_sha256"),
            "training_fingerprint_verified": bool(args.official and args.split not in HIDDEN_SPLITS),
            "training_track_id": ckpt.get("track_id"),
            "step": step,
            "mean_loss": scalar_means.get("mean_submission_loss", float("nan")),
            "mean_foldscore": mean_foldscore,
            "mean_lddt_ca": mean_lddt,
            **component_means,
            **scalar_means,
            "num_chains": len(per_chain_rows),
            "eval_wall_time_seconds": eval_seconds,
            "cumulative_samples_seen": cumulative_samples_seen,
            "cumulative_cropped_residues_seen": cumulative_cropped_residues_seen,
            "cumulative_nonpad_residues_seen": cumulative_nonpad_residues_seen,
            "sample_budget_fraction": (cumulative_samples_seen / float(sample_budget)) if sample_budget > 0 else float("nan"),
            "cropped_residue_budget_fraction": (
                cumulative_cropped_residues_seen / float(residue_budget)
            ) if residue_budget > 0 else float("nan"),
            "nonpad_residue_budget_fraction": (
                cumulative_nonpad_residues_seen / float(residue_budget)
            ) if residue_budget > 0 else float("nan"),
            "index": ckpt_idx,
        }
        all_ckpt_results.append(ckpt_result)
        print(f"[{ckpt_path.name}] mean_FoldScore={mean_foldscore:.6f} mean_lDDT-Ca={mean_lddt:.6f} chains={len(per_chain_rows)}")

    if per_chain_out_path:
        per_chain_out_path.parent.mkdir(parents=True, exist_ok=True)
        with per_chain_out_path.open("w") as f:
            for row in all_per_chain_rows:
                f.write(json.dumps(row) + "\n")
        print(f"Wrote per-chain scores to {per_chain_out_path}")

    final_ckpt_result = all_ckpt_results[-1]

    out: Dict[str, Any] = {
        "split": args.split,
        "track": track_spec.track_id,
        "official_mode": bool(args.official),
        "foldscore_version": FOLDSCORE_VERSION,
        "ckpt": final_ckpt_result["ckpt"],
        "num_checkpoints": len(all_ckpt_results),
        "checkpoints": all_ckpt_results,
        "mean_loss": final_ckpt_result["mean_loss"],
        "mean_foldscore": final_ckpt_result["mean_foldscore"],
        "mean_lddt_ca": final_ckpt_result["mean_lddt_ca"],
        "num_chains": final_ckpt_result["num_chains"],
        "submission_module": hooks.module_ref,
        "config_path": str(config_path),
        "fingerprint_path": str(Path(fingerprint_path).resolve()) if (args.official or args.verify_fingerprint) else None,
        "effective_batch_size": effective_batch_size,
        "sample_budget": sample_budget,
        "residue_budget": residue_budget,
        "cumulative_samples_seen": int(final_ckpt_result["cumulative_samples_seen"]),
        "cumulative_cropped_residues_seen": int(final_ckpt_result["cumulative_cropped_residues_seen"]),
        "cumulative_nonpad_residues_seen": int(final_ckpt_result["cumulative_nonpad_residues_seen"]),
        "env": get_env_metadata(device),
        "pred_out_dir": str(pred_out_dir) if pred_out_dir else None,
        "score_labels_dir": str(score_labels_dir) if score_labels_dir else None,
        "predict_config_sanitized": True,
        "prediction_format_version": PREDICTION_FORMAT_VERSION,
        "finished_at": utc_now_iso(),
    }
    for key, value in final_ckpt_result.items():
        if key.startswith("mean_") and key not in out:
            out[key] = value
    print(json.dumps(out, indent=2))

    if args.save:
        save_path = Path(args.save)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_path.write_text(json.dumps(out, indent=2) + "\n")
        print(f"Wrote eval summary to {save_path.resolve()}")


if __name__ == "__main__":
    main()
