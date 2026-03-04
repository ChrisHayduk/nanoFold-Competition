from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Dict, List

import numpy as np
import torch
import yaml

# Allow running as `python scripts/run_official.py` from repo root.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from nanofold.competition_policy import DEFAULT_TRACK_ID, load_track_spec
from nanofold.data import read_manifest
from nanofold.metrics import lddt_ca


SCHEMA_VERSION = 2
DEFAULT_CHECKPOINT_STEPS = "1000,2000,5000,10000"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Canonical official runner for submission evaluation.")
    ap.add_argument("--submission", type=str, required=True, help="Submission directory, e.g. submissions/alice")
    ap.add_argument("--config", type=str, default="", help="Optional config path (defaults to <submission>/config.yaml)")
    ap.add_argument("--track", type=str, default=DEFAULT_TRACK_ID)
    ap.add_argument("--python", type=str, default=sys.executable, help="Python executable to use.")
    ap.add_argument("--commit", type=str, default="", help="Optional commit hash override.")
    ap.add_argument("--description", type=str, default="", help="Optional leaderboard description.")
    ap.add_argument("--update-leaderboard", action="store_true", help="Append result into leaderboard JSON and render README.")
    ap.add_argument("--leaderboard", type=str, default="leaderboard/leaderboard.json")
    ap.add_argument("--readme", type=str, default="README.md")
    ap.add_argument(
        "--checkpoint-steps",
        type=str,
        default=DEFAULT_CHECKPOINT_STEPS,
        help=f"Comma-separated checkpoint steps for hidden efficiency metrics (default: {DEFAULT_CHECKPOINT_STEPS}).",
    )
    ap.add_argument(
        "--hidden-lock-file",
        type=str,
        default="",
        help="Maintainer lock file containing expected hidden asset hashes.",
    )
    ap.add_argument(
        "--disable-hidden",
        action="store_true",
        help="Skip hidden evaluation/scoring and produce public-val-only artifact.",
    )
    ap.add_argument("--hidden-manifest", type=str, default="", help="Hidden manifest path override.")
    ap.add_argument("--hidden-features-dir", type=str, default="", help="Hidden processed features dir override.")
    ap.add_argument("--hidden-labels-dir", type=str, default="", help="Hidden processed labels dir override.")
    ap.add_argument("--hidden-fingerprint", type=str, default="", help="Hidden fingerprint JSON path override.")
    ap.add_argument("--skip-train", action="store_true", help="Skip train.py stage and reuse existing checkpoints.")
    return ap.parse_args()


def _run(cmd: List[str]) -> None:
    printable = " ".join(cmd)
    print(f"+ {printable}")
    subprocess.run(cmd, check=True)


def _resolve_commit(explicit: str) -> str:
    if explicit.strip():
        return explicit.strip()
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        return out
    except Exception:
        return "unknown"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _maybe_sha256(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    return _sha256(path)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def _read_per_chain(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def _parse_checkpoint_steps(text: str) -> List[str]:
    tokens = [tok.strip().lower() for tok in text.split(",") if tok.strip()]
    out: List[str] = []
    for tok in tokens:
        if tok == "last":
            out.append(tok)
            continue
        if not tok.isdigit():
            raise ValueError(f"Invalid checkpoint token `{tok}` in --checkpoint-steps.")
        out.append(str(int(tok)))
    return out


def _resolve_hidden_asset(
    *,
    cli_value: str,
    track_value: str | None,
    env_key: str,
    label: str,
) -> Path:
    env_value = str(__import__("os").environ.get(env_key, "")).strip()
    value = cli_value.strip() or env_value or (track_value or "").strip()
    if not value:
        raise ValueError(f"Missing hidden asset for {label}. Set --{label} or environment variable {env_key}.")
    path = Path(value).resolve()
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")
    return path


def _validate_hidden_lock(
    *,
    lock_path: Path,
    hidden_manifest: Path,
    hidden_features_dir: Path,
    hidden_labels_dir: Path,
    hidden_fingerprint: Path,
) -> Dict[str, Any]:
    if not lock_path.exists():
        return {"lock_file": str(lock_path), "status": "missing"}

    raw = _read_json(lock_path)
    if not isinstance(raw, dict):
        raise ValueError(f"Hidden lock file must contain a JSON object: {lock_path}")

    expected = {
        "hidden_manifest_sha256": raw.get("hidden_manifest_sha256"),
        "hidden_features_fingerprint_sha256": raw.get("hidden_features_fingerprint_sha256"),
        "hidden_labels_fingerprint_sha256": raw.get("hidden_labels_fingerprint_sha256"),
        "hidden_fingerprint_sha256": raw.get("hidden_fingerprint_sha256"),
    }

    actual = {
        "hidden_manifest_sha256": _sha256(hidden_manifest),
        "hidden_features_fingerprint_sha256": None,
        "hidden_labels_fingerprint_sha256": None,
        "hidden_fingerprint_sha256": _sha256(hidden_fingerprint),
    }

    # Optional file-level fingerprints for hidden dirs.
    # Maintainers can precompute/lock these in the lock file. If absent, they are advisory only.
    features_hash_path = hidden_features_dir / "_fingerprint.sha256"
    labels_hash_path = hidden_labels_dir / "_fingerprint.sha256"
    if features_hash_path.exists():
        actual["hidden_features_fingerprint_sha256"] = features_hash_path.read_text().strip().lower()
    if labels_hash_path.exists():
        actual["hidden_labels_fingerprint_sha256"] = labels_hash_path.read_text().strip().lower()

    for key, expected_value in expected.items():
        if expected_value is None:
            continue
        if not isinstance(expected_value, str):
            raise ValueError(f"Invalid `{key}` in {lock_path}; expected string or null.")
        if actual.get(key) != expected_value:
            raise ValueError(
                f"Hidden lock mismatch for `{key}`: expected `{expected_value}`, got `{actual.get(key)}`."
            )

    return {
        "lock_file": str(lock_path.resolve()),
        "status": "validated",
        "expected": expected,
        "actual": actual,
    }


def _parse_step_from_ckpt_path(ckpt_path: str, *, max_steps: int) -> int | None:
    name = Path(ckpt_path).name
    m = re.search(r"ckpt_step_(\d+)\.pt$", name)
    if m:
        return int(m.group(1))
    if name == "ckpt_last.pt":
        return int(max_steps)
    return None


def _prediction_subdir(pred_root: Path, ckpt_path: str, n_ckpts: int) -> Path:
    if n_ckpts <= 1:
        return pred_root
    return pred_root / Path(ckpt_path).stem


def _center_crop_labels(ca_coords: np.ndarray, ca_mask: np.ndarray, crop_size: int) -> tuple[np.ndarray, np.ndarray]:
    L = int(ca_coords.shape[0])
    if L <= crop_size:
        return ca_coords, ca_mask
    start = (L - crop_size) // 2
    end = start + crop_size
    return ca_coords[start:end], ca_mask[start:end]


def _score_hidden_predictions(
    *,
    hidden_manifest: Path,
    hidden_labels_dir: Path,
    pred_root: Path,
    checkpoint_entries: List[Dict[str, Any]],
    crop_size: int,
    max_steps: int,
    per_chain_out_path: Path,
) -> Dict[str, Any]:
    chain_ids = read_manifest(hidden_manifest)
    n_ckpts = len(checkpoint_entries)
    checkpoint_scores: List[Dict[str, Any]] = []
    per_chain_rows: List[Dict[str, Any]] = []

    for item in checkpoint_entries:
        ckpt_path = str(item["ckpt"])
        step = _parse_step_from_ckpt_path(ckpt_path, max_steps=max_steps)
        ckpt_pred_dir = _prediction_subdir(pred_root, ckpt_path, n_ckpts=n_ckpts)
        scores: List[float] = []

        for chain_id in chain_ids:
            pred_path = ckpt_pred_dir / f"{chain_id}.npz"
            label_path = hidden_labels_dir / f"{chain_id}.npz"
            if not pred_path.exists():
                raise FileNotFoundError(f"Missing prediction file for hidden scoring: {pred_path}")
            if not label_path.exists():
                raise FileNotFoundError(f"Missing hidden label file: {label_path}")

            with np.load(pred_path) as pred_npz:
                pred_ca = np.asarray(pred_npz["pred_ca"], dtype=np.float32)
            with np.load(label_path) as label_npz:
                true_ca = np.asarray(label_npz["ca_coords"], dtype=np.float32)
                ca_mask = np.asarray(label_npz["ca_mask"], dtype=bool)
            true_ca, ca_mask = _center_crop_labels(true_ca, ca_mask, crop_size=crop_size)

            L = min(pred_ca.shape[0], true_ca.shape[0], ca_mask.shape[0])
            if L <= 0:
                chain_score = float("nan")
            else:
                chain_score = float(
                    lddt_ca(
                        torch.from_numpy(pred_ca[:L]),
                        torch.from_numpy(true_ca[:L]),
                        torch.from_numpy(ca_mask[:L]),
                    )
                    .detach()
                    .cpu()
                )
            if not np.isnan(chain_score):
                scores.append(chain_score)
            per_chain_rows.append(
                {
                    "split": "hidden_val",
                    "ckpt": ckpt_path,
                    "step": step,
                    "chain_id": chain_id,
                    "lddt_ca": chain_score,
                }
            )

        mean_score = float(sum(scores) / len(scores)) if scores else float("nan")
        checkpoint_scores.append(
            {
                "ckpt": ckpt_path,
                "step": step,
                "mean_lddt_ca": mean_score,
                "num_chains": len(chain_ids),
            }
        )

    per_chain_out_path.parent.mkdir(parents=True, exist_ok=True)
    with per_chain_out_path.open("w") as f:
        for row in per_chain_rows:
            f.write(json.dumps(row) + "\n")

    sorted_points = [row for row in checkpoint_scores if row["step"] is not None]
    sorted_points.sort(key=lambda x: int(x["step"]))

    final_hidden = float("nan")
    if sorted_points:
        final_hidden = float(sorted_points[-1]["mean_lddt_ca"])

    auc_hidden = float("nan")
    if len(sorted_points) == 1:
        auc_hidden = float(sorted_points[0]["mean_lddt_ca"])
    elif len(sorted_points) >= 2:
        area = 0.0
        for i in range(1, len(sorted_points)):
            x0 = float(sorted_points[i - 1]["step"])
            x1 = float(sorted_points[i]["step"])
            y0 = float(sorted_points[i - 1]["mean_lddt_ca"])
            y1 = float(sorted_points[i]["mean_lddt_ca"])
            area += (x1 - x0) * (y0 + y1) * 0.5
        x_start = float(sorted_points[0]["step"])
        x_end = float(sorted_points[-1]["step"])
        denom = max(x_end - x_start, 1.0)
        auc_hidden = area / denom if denom > 0 else float("nan")

    lddt_at_steps = {
        str(int(row["step"])): float(row["mean_lddt_ca"])
        for row in sorted_points
        if row["step"] is not None
    }

    return {
        "final_hidden_lddt_ca": final_hidden,
        "lddt_auc_hidden": auc_hidden,
        "lddt_at_steps": lddt_at_steps,
        "checkpoint_metrics_hidden": checkpoint_scores,
        "per_chain_scores_hidden_path": str(per_chain_out_path.resolve()),
    }


def _write_temp_hidden_config(
    *,
    cfg: Dict[str, Any],
    hidden_features_dir: Path,
    hidden_labels_dir: Path,
    out_path: Path,
) -> Path:
    hidden_cfg = copy.deepcopy(cfg)
    data_cfg = hidden_cfg.get("data")
    if not isinstance(data_cfg, dict):
        raise ValueError("Config missing `data` section.")
    data_cfg["processed_features_dir"] = str(hidden_features_dir)
    data_cfg["processed_labels_dir"] = str(hidden_labels_dir)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(yaml.safe_dump(hidden_cfg, sort_keys=False))
    return out_path


def main() -> None:
    args = parse_args()
    submission_dir = Path(args.submission).resolve()
    if not submission_dir.exists():
        raise FileNotFoundError(f"Submission dir not found: {submission_dir}")

    config_path = Path(args.config).resolve() if args.config else (submission_dir / "config.yaml").resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    cfg = yaml.safe_load(config_path.read_text())
    if not isinstance(cfg, dict):
        raise ValueError(f"Config must contain a YAML mapping: {config_path}")
    run_name = str(cfg.get("run_name", "")).strip()
    if not run_name:
        raise ValueError("Config must define a non-empty `run_name`.")

    track_spec = load_track_spec(args.track)
    commit = _resolve_commit(args.commit)
    data_cfg = cfg.get("data", {}) if isinstance(cfg.get("data"), dict) else {}
    train_manifest_path = Path(str(data_cfg.get("train_manifest", track_spec.train_manifest))).resolve()
    val_manifest_path = Path(str(data_cfg.get("val_manifest", track_spec.val_manifest))).resolve()
    all_manifest_path = Path(str(track_spec.all_manifest)).resolve() if track_spec.all_manifest else None

    run_dir = Path("runs") / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    eval_public_summary_path = run_dir / "eval_val.json"
    per_chain_public_path = run_dir / "per_chain_scores_val.jsonl"
    result_path = run_dir / "result.json"
    ckpt_dir = run_dir / "checkpoints"
    ckpt_last_path = ckpt_dir / "ckpt_last.pt"

    _run(
        [
            args.python,
            "scripts/validate_submission.py",
            "--submission",
            str(submission_dir),
            "--track",
            track_spec.track_id,
            "--strict",
        ]
    )

    if not args.skip_train:
        _run(
            [
                args.python,
                "train.py",
                "--config",
                str(config_path),
                "--track",
                track_spec.track_id,
                "--official",
            ]
        )

    if not ckpt_last_path.exists():
        raise FileNotFoundError(f"Missing checkpoint after training: {ckpt_last_path}")

    forbid_labels_dir = run_dir / "_forbidden_eval_labels_mount"
    forbid_labels_dir.mkdir(parents=True, exist_ok=True)
    score_labels_dir = str(data_cfg.get("processed_labels_dir", "")).strip()
    if not score_labels_dir:
        raise ValueError(
            "Config data.processed_labels_dir is required for public validation scoring in the official runner."
        )

    _run(
        [
            args.python,
            "eval.py",
            "--config",
            str(config_path),
            "--ckpt",
            str(ckpt_last_path),
            "--split",
            "val",
            "--track",
            track_spec.track_id,
            "--official",
            "--forbid-labels-dir",
            str(forbid_labels_dir),
            "--score-labels-dir",
            score_labels_dir,
            "--save",
            str(eval_public_summary_path),
            "--per-chain-out",
            str(per_chain_public_path),
        ]
    )

    eval_public_summary = _read_json(eval_public_summary_path)
    public_per_chain_rows = _read_per_chain(per_chain_public_path)
    public_scores = [float(row["lddt_ca"]) for row in public_per_chain_rows if "lddt_ca" in row]
    public_summary = {
        "count": len(public_scores),
        "mean": float(mean(public_scores)) if public_scores else float("nan"),
        "std": float(pstdev(public_scores)) if len(public_scores) > 1 else 0.0,
        "min": float(min(public_scores)) if public_scores else float("nan"),
        "max": float(max(public_scores)) if public_scores else float("nan"),
    }

    hidden_results: Dict[str, Any] = {
        "final_hidden_lddt_ca": float("nan"),
        "lddt_auc_hidden": float("nan"),
        "lddt_at_steps": {},
        "checkpoint_metrics_hidden": [],
        "per_chain_scores_hidden_path": None,
    }
    hidden_assets_meta: Dict[str, Any] | None = None
    hidden_eval_summary_path: Path | None = None
    hidden_eval_cfg_path: Path | None = None
    hidden_lock_meta: Dict[str, Any] | None = None

    if not args.disable_hidden:
        hidden_manifest = _resolve_hidden_asset(
            cli_value=args.hidden_manifest,
            track_value=track_spec.hidden_manifest,
            env_key="NANOFOLD_HIDDEN_MANIFEST",
            label="hidden-manifest",
        )
        hidden_features_dir = _resolve_hidden_asset(
            cli_value=args.hidden_features_dir,
            track_value=None,
            env_key="NANOFOLD_HIDDEN_FEATURES_DIR",
            label="hidden-features-dir",
        )
        hidden_labels_dir = _resolve_hidden_asset(
            cli_value=args.hidden_labels_dir,
            track_value=None,
            env_key="NANOFOLD_HIDDEN_LABELS_DIR",
            label="hidden-labels-dir",
        )
        hidden_fingerprint = _resolve_hidden_asset(
            cli_value=args.hidden_fingerprint,
            track_value=track_spec.hidden_fingerprint_path,
            env_key="NANOFOLD_HIDDEN_FINGERPRINT",
            label="hidden-fingerprint",
        )

        lock_file_value = args.hidden_lock_file.strip() or track_spec.hidden_lock_file or "leaderboard/official_hidden_assets.lock.json"
        lock_file = Path(lock_file_value).resolve()
        hidden_lock_meta = _validate_hidden_lock(
            lock_path=lock_file,
            hidden_manifest=hidden_manifest,
            hidden_features_dir=hidden_features_dir,
            hidden_labels_dir=hidden_labels_dir,
            hidden_fingerprint=hidden_fingerprint,
        )

        hidden_eval_cfg_path = run_dir / "hidden_eval_config.yaml"
        _write_temp_hidden_config(
            cfg=cfg,
            hidden_features_dir=hidden_features_dir,
            hidden_labels_dir=hidden_labels_dir,
            out_path=hidden_eval_cfg_path,
        )

        checkpoint_tokens = _parse_checkpoint_steps(args.checkpoint_steps)
        checkpoint_steps_arg = ",".join(checkpoint_tokens)
        hidden_pred_dir = run_dir / "hidden_predictions"
        hidden_eval_summary_path = run_dir / "eval_hidden_raw.json"

        _run(
            [
                args.python,
                "eval.py",
                "--config",
                str(hidden_eval_cfg_path),
                "--split",
                "hidden_val",
                "--track",
                track_spec.track_id,
                "--official",
                "--fingerprint",
                str(hidden_fingerprint),
                "--hidden-manifest",
                str(hidden_manifest),
                "--forbid-labels-dir",
                str(forbid_labels_dir),
                "--ckpt-dir",
                str(ckpt_dir),
                "--ckpt-steps",
                checkpoint_steps_arg,
                "--pred-out-dir",
                str(hidden_pred_dir),
                "--save",
                str(hidden_eval_summary_path),
            ]
        )

        hidden_eval_summary = _read_json(hidden_eval_summary_path)
        checkpoint_entries = hidden_eval_summary.get("checkpoints", [])
        if not isinstance(checkpoint_entries, list) or not checkpoint_entries:
            raise ValueError("Hidden eval summary did not contain checkpoint results.")

        hidden_per_chain_path = run_dir / "per_chain_scores_hidden.jsonl"
        hidden_results = _score_hidden_predictions(
            hidden_manifest=hidden_manifest,
            hidden_labels_dir=hidden_labels_dir,
            pred_root=hidden_pred_dir,
            checkpoint_entries=checkpoint_entries,
            crop_size=int(cfg.get("data", {}).get("crop_size", 256)),
            max_steps=int(cfg.get("train", {}).get("max_steps", 0)),
            per_chain_out_path=hidden_per_chain_path,
        )

        hidden_assets_meta = {
            "manifest_path": str(hidden_manifest),
            "manifest_sha256": _sha256(hidden_manifest),
            "features_dir": str(hidden_features_dir),
            "labels_dir": str(hidden_labels_dir),
            "fingerprint_path": str(hidden_fingerprint),
            "fingerprint_sha256": _sha256(hidden_fingerprint),
            "lock": hidden_lock_meta,
        }

    fingerprint_path = Path(track_spec.fingerprint_path) if track_spec.fingerprint_path else None
    train_metrics_path = run_dir / "metrics.json"
    train_metrics = _read_json(train_metrics_path) if train_metrics_path.exists() else {}

    rank_metric = "final_hidden_lddt_ca" if not args.disable_hidden else "public_val_lddt_ca"
    rank_score = (
        float(hidden_results["final_hidden_lddt_ca"])
        if not args.disable_hidden
        else float(eval_public_summary["mean_lddt_ca"])
    )

    result = {
        "schema_version": SCHEMA_VERSION,
        "run_name": run_name,
        "submission_name": submission_dir.name,
        "submission_dir": str(submission_dir),
        "config_path": str(config_path),
        "track": track_spec.track_id,
        "rank_metric": rank_metric,
        "rank_score": rank_score,
        "final_hidden_lddt_ca": float(hidden_results["final_hidden_lddt_ca"]),
        "lddt_auc_hidden": float(hidden_results["lddt_auc_hidden"]),
        "lddt_at_steps": hidden_results["lddt_at_steps"],
        "checkpoint_metrics_hidden": hidden_results["checkpoint_metrics_hidden"],
        "public_val_lddt_ca": float(eval_public_summary["mean_lddt_ca"]),
        "public_val_per_chain_summary": public_summary,
        "num_chains_public_val": int(eval_public_summary.get("num_chains", len(public_per_chain_rows))),
        "eval_public_summary_path": str(eval_public_summary_path.resolve()),
        "per_chain_public_scores_path": str(per_chain_public_path.resolve()),
        "eval_hidden_summary_path": str(hidden_eval_summary_path.resolve()) if hidden_eval_summary_path else None,
        "per_chain_hidden_scores_path": hidden_results["per_chain_scores_hidden_path"],
        "hidden_eval_config_path": str(hidden_eval_cfg_path.resolve()) if hidden_eval_cfg_path else None,
        "train_metrics_path": str(train_metrics_path.resolve()),
        "train_wall_time_seconds": float(train_metrics.get("wall_time_seconds", float("nan")))
        if isinstance(train_metrics, dict)
        else float("nan"),
        "eval_public_wall_time_seconds": float(eval_public_summary.get("eval_wall_time_seconds", float("nan"))),
        "fingerprint_path": str(fingerprint_path.resolve()) if fingerprint_path else None,
        "fingerprint_sha256": _sha256(fingerprint_path) if fingerprint_path and fingerprint_path.exists() else None,
        "hidden_assets": hidden_assets_meta,
        "manifest_paths": {
            "train_manifest": str(train_manifest_path),
            "val_manifest": str(val_manifest_path),
            "all_manifest": str(all_manifest_path) if all_manifest_path else None,
        },
        "manifest_sha256": {
            "train_manifest": _maybe_sha256(train_manifest_path),
            "val_manifest": _maybe_sha256(val_manifest_path),
            "all_manifest": _maybe_sha256(all_manifest_path) if all_manifest_path else None,
        },
        "track_manifest_sha256": {
            "train_manifest": track_spec.train_manifest_sha256,
            "val_manifest": track_spec.val_manifest_sha256,
            "all_manifest": track_spec.all_manifest_sha256,
            "hidden_manifest": track_spec.hidden_manifest_sha256,
        },
        "config_sha256": _sha256(config_path),
        "commit": commit,
        "description": args.description.strip() or f"Official run for {submission_dir.name}",
        "created_at": datetime.now(tz=timezone.utc).isoformat(),
        "train_env": train_metrics.get("env", {}) if isinstance(train_metrics, dict) else {},
        "eval_public_env": eval_public_summary.get("env", {}) if isinstance(eval_public_summary, dict) else {},
    }
    result_path.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Wrote result artifact: {result_path.resolve()}")

    if args.update_leaderboard:
        _run(
            [
                args.python,
                "scripts/add_leaderboard_entry.py",
                "--result",
                str(result_path),
                "--leaderboard",
                args.leaderboard,
                "--readme",
                args.readme,
                "--description",
                result["description"],
            ]
        )


if __name__ == "__main__":
    main()
